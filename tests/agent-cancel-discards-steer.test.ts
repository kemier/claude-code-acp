/**
 * Reported against a claude-code-acp session: the user queued two messages and
 * clicked "Send Now" on the second. Both appeared in the chat panel, but the
 * work the agent actually did was the FIRST one — the message that had been
 * steered into the turn the Send Now cancelled.
 *
 * Mechanism: Claude CLI cannot inject mid-turn, so a steer is queued and
 * drained AFTER the main turn (drainSteerQueue). "Send Now" cancels that turn,
 * and the cancel exits runPrompt BEFORE the drain — leaving the steer in the
 * queue. The next turn (the Send Now message) then answers, and its own
 * post-turn drain delivers the stale steer as a follow-up: two bubbles, and the
 * last thing the model works on is the older message.
 *
 * drainSteerQueue already states the intended policy — "Cancellation discards
 * the rest — the user asked to stop" — it just never applied when the cancel
 * landed before a drain had started.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

/** Rejecter for the first (hanging) turn, invoked by runner.cancel(). */
let cancelReject: ((err: Error) => void) | undefined;
/** Resolver for the second turn, so a steer can be sent while it is in flight. */
let releaseTurn2: (() => void) | undefined;
/** Every prompt text handed to the runner, in order. */
let runnerPrompts: string[] = [];

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    let startCalls = 0;
    this.startSessionStreaming = vi
      .fn()
      .mockImplementation(async (_cwd: string, prompt: string) => {
        runnerPrompts.push(prompt);
        startCalls++;
        if (startCalls === 1) {
          // Turn 1 hangs until cancel() rejects it — the turn the user interrupts.
          return new Promise((_resolve, reject) => {
            cancelReject = reject;
          });
        }
        // Turn 2 (the "Send Now" message) stays in flight until released, so a
        // steer can arrive mid-turn, and establishes a Claude session id — so a
        // stale steer WOULD be drainable after it.
        return new Promise((resolve) => {
          releaseTurn2 = () => resolve({ text: "ok", sessionId: "uuid-1" });
        });
      });
    this.continueSessionStreaming = vi
      .fn()
      .mockImplementation(async (_sid: string, prompt: string) => {
        runnerPrompts.push(prompt);
        return { text: "ok", sessionId: "uuid-1" };
      });
    this.isRunning = vi.fn(() => false);
    this.cancel = vi.fn().mockImplementation(() => {
      if (cancelReject) {
        cancelReject(new Error("Process killed"));
        cancelReject = undefined;
      }
    });
  });
  return { ClaudeRunner: MockClaudeRunner };
});

function createMockConnection(): AgentSideConnection {
  return {
    sessionUpdate: vi.fn().mockResolvedValue(undefined),
    requestPermission: vi.fn(),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    createTerminal: vi.fn(),
    signal: new AbortController().signal,
    closed: new Promise(() => {}),
  } as unknown as AgentSideConnection;
}

/** Texts of every user_message_chunk the agent echoed to the client. */
function echoedUserTexts(conn: AgentSideConnection): string[] {
  const calls = (conn.sessionUpdate as unknown as { mock: { calls: any[][] } }).mock.calls;
  return calls
    .map((c) => c[0]?.update)
    .filter((u) => u?.sessionUpdate === "user_message_chunk")
    .map((u) => String(u?.content?.text ?? ""));
}

describe("cancel discards an undelivered steer", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    cancelReject = undefined;
    releaseTurn2 = undefined;
    runnerPrompts = [];
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  async function connectedSession(): Promise<string> {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    return session.sessionId;
  }

  /** Wait until the mock's turn-2 resolver exists, then release it. */
  async function finishTurn2(): Promise<void> {
    for (let i = 0; i < 50 && !releaseTurn2; i++) {
      await new Promise((r) => setTimeout(r, 2));
    }
    releaseTurn2?.();
  }

  it("does not deliver it at the end of the next turn", async () => {
    const sessionId = await connectedSession();

    // Turn 1: the long turn the user is watching.
    const turn1 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "TURN-ONE" }] });
    await new Promise((r) => setTimeout(r, 10));

    // Queued message A, auto-steered into the running turn by vscode-acp.
    await agent.extNotification!("session/steer", { sessionId, text: "MESSAGE-A" });

    // "Send Now" on message B: the client cancels, then prompts B.
    await agent.cancel({ sessionId });
    expect((await turn1).stopReason).toBe("cancelled");

    const turn2 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MESSAGE-B" }] });
    await finishTurn2();
    expect((await turn2).stopReason).toBe("end_turn");

    // B is the turn that ran. A must NOT have been resurrected as a follow-up:
    // the client still holds it queued and drains it as its own prompt.
    expect(runnerPrompts.some((p) => p.includes("MESSAGE-B"))).toBe(true);
    expect(runnerPrompts.filter((p) => p.includes("MESSAGE-A"))).toEqual([]);
    expect(echoedUserTexts(conn).filter((t) => t.includes("MESSAGE-A"))).toEqual([]);
  });

  it("still delivers a steer sent into a turn that is NOT cancelled", async () => {
    const sessionId = await connectedSession();

    // Turn 1 exists only to give the mock its cancel path; the delivery case is
    // turn 2, which completes normally.
    const turn1 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "TURN-ONE" }] });
    await new Promise((r) => setTimeout(r, 10));
    await agent.cancel({ sessionId });
    await turn1;

    const turn2 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MESSAGE-B" }] });
    await new Promise((r) => setTimeout(r, 10));
    // Steer while turn 2 is in flight — it must reach the model after it.
    await agent.extNotification!("session/steer", { sessionId, text: "MESSAGE-C" });
    await finishTurn2();
    expect((await turn2).stopReason).toBe("end_turn");

    expect(runnerPrompts.some((p) => p.includes("MESSAGE-C"))).toBe(true);
    expect(echoedUserTexts(conn).some((t) => t.includes("MESSAGE-C"))).toBe(true);
  });

  it("ignores a steer that races in after the cancel", async () => {
    const sessionId = await connectedSession();

    const turn1 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "TURN-ONE" }] });
    await new Promise((r) => setTimeout(r, 10));

    // Cancel first, then a steer arrives before the client has seen promptEnd —
    // queuing it would put the same stale message at the end of the next turn.
    await agent.cancel({ sessionId });
    await agent.extNotification!("session/steer", { sessionId, text: "MESSAGE-LATE" });
    await turn1;

    const turn2 = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MESSAGE-B" }] });
    await finishTurn2();
    await turn2;

    expect(runnerPrompts.filter((p) => p.includes("MESSAGE-LATE"))).toEqual([]);
  });
});
