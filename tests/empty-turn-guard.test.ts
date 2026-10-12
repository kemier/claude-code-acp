// Regression: a turn that returns end_turn with zero tokens, no text, and no
// tool call (the degenerate "No response requested." turn the CLI emits when
// answering a synthetic control frame) was forwarded as a silent successful
// end_turn — the user saw an empty bubble, the button flipped back, and they
// had to send "go on" a second time.
//
// Fix: a bounded retry guard in executePrompt. On match, re-deliver the same
// turn once (MAX_EMPTY_TURN_RETRIES = 1). If the retry is also empty, surface
// a visible agent_message_chunk notice instead of a silent bubble.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  readdirSync: vi.fn(() => []),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

// Each test stages a sequence of "drive" behaviours. The first prompt() call
// invokes drive[0]; a retry (if the guard fires) invokes drive[1].
let drives: ((onEvent: (e: any) => void) => Promise<any>)[] = [];

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    let driveIndex = 0;
    const drive = async (onEvent: (e: any) => void) => {
      const fn = drives[Math.min(driveIndex, drives.length - 1)] ?? drives[0];
      driveIndex++;
      return fn(onEvent);
    };
    this.startSessionStreaming = vi.fn().mockImplementation((_cwd: string, _p: string, onEvent: any) => drive(onEvent));
    this.continueSessionStreaming = vi.fn().mockImplementation((_sid: string, _p: string, onEvent: any) => drive(onEvent));
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, onEvent: any) => drive(onEvent),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, onEvent: any) => drive(onEvent),
    );
    this.cancel = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.isRunning = vi.fn().mockReturnValue(false);
    this.setSessionPermissionMode = vi.fn();
    // Track how many times the runner was invoked so tests can assert retry count.
    this._driveCount = () => driveIndex;
  });
  return { ClaudeRunner: MockClaudeRunner };
});

// A degenerate empty turn: result with zero usage, no text, no tool call.
const EMPTY_RESULT = {
  type: "result",
  usage: { input_tokens: 0, output_tokens: 0 },
  stopReason: "end_turn",
};
// A normal turn: result with real usage and text.
const TEXT_RESULT = {
  type: "result",
  text: "Here is the answer.",
  usage: { input_tokens: 100, output_tokens: 50 },
  stopReason: "end_turn",
};
// A tool-only turn: a tool_use event, then a zero-usage result.
const TOOL_RESULT = {
  type: "result",
  usage: { input_tokens: 0, output_tokens: 0 },
  stopReason: "end_turn",
};

function driveEmpty() {
  return async (onEvent: (e: any) => void) => {
    onEvent({ type: "assistant", apiCallIndex: 0 });
    onEvent(EMPTY_RESULT);
    return { text: "", sessionId: "claude-uuid-abc" };
  };
}
function driveText() {
  return async (onEvent: (e: any) => void) => {
    onEvent({ type: "assistant", apiCallIndex: 0 });
    onEvent(TEXT_RESULT);
    return { text: "Here is the answer.", sessionId: "claude-uuid-abc" };
  };
}
function driveToolOnly() {
  return async (onEvent: (e: any) => void) => {
    onEvent({ type: "assistant", apiCallIndex: 0 });
    onEvent({ type: "tool_use", toolName: "Bash", toolInput: { command: "ls" } });
    onEvent(TOOL_RESULT);
    return { text: "", sessionId: "claude-uuid-abc" };
  };
}

describe("degenerate empty-turn guard", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(async () => {
    vi.clearAllMocks();
    drives = [];
    conn = {
      sessionUpdate: vi.fn().mockResolvedValue(undefined),
      requestPermission: vi.fn(),
      readTextFile: vi.fn(),
      writeTextFile: vi.fn(),
      createTerminal: vi.fn(),
      extMethod: vi.fn().mockResolvedValue({}),
      extNotification: vi.fn().mockResolvedValue(undefined),
      signal: new AbortController().signal,
      closed: new Promise(() => {}),
    } as unknown as AgentSideConnection;
    agent = createClaudeCodeAgent(conn);
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
  });

  async function runPrompt() {
    const sid = (await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] })).sessionId;
    const result = await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "go on" }] });
    const frames = ((conn.sessionUpdate as any).mock.calls).map((c: any) => c[0]?.update);
    return { result, frames };
  }

  it("retries once when the first turn is degenerate and the retry succeeds", async () => {
    drives = [driveEmpty(), driveText()];
    const { result } = await runPrompt();
    expect(result.stopReason).toBe("end_turn");
    // The retry's text should have been streamed to the client.
    const textFrames = (
      ((conn.sessionUpdate as any).mock.calls).map((c: any) => c[0]?.update)
    ).filter((u: any) => u?.sessionUpdate === "agent_message_chunk");
    expect(textFrames.some((f: any) => f.content?.text === "Here is the answer.")).toBe(true);
  });

  it("does not retry when the turn has real text", async () => {
    drives = [driveText()];
    const { frames } = await runPrompt();
    // Only one agent_message_chunk (the real text), no "empty response" notice.
    const textFrames = frames.filter((u: any) => u?.sessionUpdate === "agent_message_chunk");
    expect(textFrames).toHaveLength(1);
    expect(textFrames[0].content?.text).toBe("Here is the answer.");
  });

  it("does not retry when the turn emitted a tool call", async () => {
    drives = [driveToolOnly()];
    const { frames } = await runPrompt();
    // The tool call should be present; no "empty response" notice.
    const toolFrames = frames.filter((u: any) => u?.sessionUpdate === "tool_call");
    expect(toolFrames.length).toBeGreaterThanOrEqual(1);
    const noticeFrames = frames.filter(
      (u: any) => u?.sessionUpdate === "agent_message_chunk" && u.content?.text?.includes("empty response"),
    );
    expect(noticeFrames).toHaveLength(0);
  });

  it("surfaces a notice on exhaustion and does not retry more than once", async () => {
    drives = [driveEmpty(), driveEmpty()];
    const { frames } = await runPrompt();
    const noticeFrames = frames.filter(
      (u: any) => u?.sessionUpdate === "agent_message_chunk" && u.content?.text?.includes("empty response"),
    );
    expect(noticeFrames).toHaveLength(1);
    expect(noticeFrames[0]._meta?.commandOutput).toBe(true);
  });

  it("a cancelled turn (stopReason cancelled) does not match the fingerprint", async () => {
    // The guard requires turnStopReason === "end_turn". A cancelled turn has
    // stopReason "cancelled", so even with zero usage and no text the guard
    // must not fire. We test this by driving a turn that emits a result with
    // stopReason "cancelled" — the guard must be a no-op.
    drives = [
      async (onEvent: (e: any) => void) => {
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({
          type: "result",
          usage: { input_tokens: 0, output_tokens: 0 },
          stopReason: "cancelled",
        });
        return { text: "", sessionId: "claude-uuid-abc" };
      },
    ];
    const sid = (await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] })).sessionId;
    const result = await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "go on" }] });
    // The adapter returns "end_turn" for a non-cancelled result; the guard
    // must NOT have fired because turnStopReason would have been "cancelled"
    // (set from event.stopReason in the result branch). No "empty response"
    // notice should appear.
    const frames = ((conn.sessionUpdate as any).mock.calls).map((c: any) => c[0]?.update);
    const noticeFrames = frames.filter(
      (u: any) => u?.sessionUpdate === "agent_message_chunk" && u.content?.text?.includes("empty response"),
    );
    expect(noticeFrames).toHaveLength(0);
  });
});
