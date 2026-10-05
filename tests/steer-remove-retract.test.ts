/**
 * The user types a message during a Claude Code turn; vscode-acp auto-steers it
 * (session/steer) into the running turn, and the agent queues it for a follow-up
 * delivery at promptEnd. The user then DELETES the queue chip. Without a retract
 * the agent would still deliver the deleted text as a follow-up turn — the model
 * answers a message the user removed.
 *
 * `session/steer-remove` lets the client retract a queued steer by (collapsed)
 * text. This test proves a retracted steer is dropped from the queue and never
 * delivered, while a non-retracted steer is still delivered at the turn's end.
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

/** Resolver for the in-flight turn, so steers can be sent mid-turn. */
let releaseTurn: (() => void) | undefined;
/** Every prompt text handed to the runner, in order. */
let runnerPrompts: string[] = [];

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi
      .fn()
      .mockImplementation(async (_cwd: string, prompt: string) => {
        runnerPrompts.push(prompt);
        // The turn stays in flight until released, so a steer (and a steer-remove)
        // can arrive while it is running. Establishes a Claude session id so a
        // non-retracted steer WOULD be drainable after it.
        return new Promise((resolve) => {
          releaseTurn = () => resolve({ text: "ok", sessionId: "uuid-1" });
        });
      });
    this.continueSessionStreaming = vi
      .fn()
      .mockImplementation(async (_sid: string, prompt: string) => {
        runnerPrompts.push(prompt);
        return { text: "ok", sessionId: "uuid-1" };
      });
    this.isRunning = vi.fn(() => true);
    this.writeSteerToMainTurn = vi.fn(() => false);
    this.cancel = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.setSessionPermissionMode = vi.fn();
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

/** Texts of every sessionUpdate of a given type the agent sent to the client. */
function sentTexts(conn: AgentSideConnection, kind: string): string[] {
  const calls = (conn.sessionUpdate as unknown as { mock: { calls: any[][] } }).mock.calls;
  return calls
    .map((c) => c[0]?.update)
    .filter((u) => u?.sessionUpdate === kind)
    .map((u) => String(u?.content?.text ?? ""));
}

describe("steer-remove retracts a queued steer", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    releaseTurn = undefined;
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

  async function finishTurn(): Promise<void> {
    for (let i = 0; i < 50 && !releaseTurn; i++) {
      await new Promise((r) => setTimeout(r, 2));
    }
    releaseTurn?.();
  }

  it("drops a retracted steer and still delivers a non-retracted one", async () => {
    const sessionId = await connectedSession();

    // Start the turn (in flight), steer two messages into it, retract one, THEN
    // let the turn end — so drainSteerQueue runs against the post-retract queue.
    const turn = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MAIN-TURN" }] });
    await new Promise((r) => setTimeout(r, 10));

    // Two messages auto-steered into the running turn.
    await agent.extNotification!("session/steer", { sessionId, text: "KEEP-ME" });
    await agent.extNotification!("session/steer", { sessionId, text: "DELETE-ME" });

    // The user deletes the second one before the turn ends.
    await agent.extNotification!("session/steer-remove", { sessionId, text: "DELETE-ME" });

    await finishTurn();
    expect((await turn).stopReason).toBe("end_turn");

    // The retracted steer never reaches the model…
    expect(runnerPrompts.filter((p) => p.includes("DELETE-ME"))).toEqual([]);
    // …and the agent acknowledged the removal to the client.
    expect(sentTexts(conn, "user_message_removed")).toContain("DELETE-ME");
    // The non-retracted steer IS delivered as a follow-up turn at promptEnd.
    expect(runnerPrompts.filter((p) => p.includes("KEEP-ME")).length).toBe(1);
    expect(sentTexts(conn, "user_message_chunk")).toContain("KEEP-ME");
  });

  it("is a no-op when the steer was already delivered (queue drained)", async () => {
    const sessionId = await connectedSession();

    // Start the turn, steer into it while it runs, then let it finish so
    // drainSteerQueue delivers the steer.
    const turn = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MAIN-TURN" }] });
    await new Promise((r) => setTimeout(r, 10));
    await agent.extNotification!("session/steer", { sessionId, text: "EARLY" });
    await finishTurn();
    expect((await turn).stopReason).toBe("end_turn");

    // Late retract after delivery: nothing left in the queue to remove.
    await agent.extNotification!("session/steer-remove", { sessionId, text: "EARLY" });
    expect(sentTexts(conn, "user_message_removed")).toEqual([]);
  });

  it("matches on collapsed whitespace, not exact bytes", async () => {
    const sessionId = await connectedSession();

    const turn = agent.prompt({ sessionId, prompt: [{ type: "text", text: "MAIN-TURN" }] });
    await new Promise((r) => setTimeout(r, 10));

    await agent.extNotification!("session/steer", { sessionId, text: "delete   me  now" });
    // Client sends a different whitespace form of the same text.
    await agent.extNotification!("session/steer-remove", { sessionId, text: "delete me now" });

    await finishTurn();
    expect((await turn).stopReason).toBe("end_turn");
    expect(runnerPrompts.filter((p) => p.includes("delete"))).toEqual([]);
    expect(sentTexts(conn, "user_message_removed").length).toBe(1);
  });
});
