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

/**
 * Regression: the "stop mid-stream, then the queued 'Send Now' message vanishes"
 * bug.
 *
 * Observed sequence (session 67d66e19, 2026-09-26 03:13:22, from the ACP log):
 *   03:13:22.846  Cancel request             (latches cancelledSessions, A in flight)
 *   03:13:22.886  Prompt A resolves end_turn  (A's cancel checks had already passed)
 *   03:13:23.112  Prompt B prompt() called    (vscode-acp drainQueue sent it late)
 *   03:13:23.304  "Queued prompt cancelled before start"  ← B wrongly dropped
 *
 * Root cause: cancel() latched cancelledSessions while A was in flight, recording
 * A's generation as the target. A then exited via its NORMAL end_turn path (which
 * does not clear cancelledSessions). ~226ms later vscode-acp's drainQueue sent B
 * (a fresh "Send Now" message); B's executePrompt reached the before-start check
 * against the STALE latch and was cancelled before it even started.
 *
 * Fix: the cancel latch is generation-scoped. cancel() records the generation of
 * the prompt that was in flight at cancel time (cancelledGen). A prompt is only
 * cancelled-before-start if the cancel's target generation matches its OWN. B
 * (a later generation) is not hit by the stale latch and runs as its own turn.
 */

// Turn A: resolves end_turn on its own after a short delay, so A passes its
// early-cancel site (no latch yet) and returns end_turn — mirroring the real
// flow where A's streaming had already finished before the Stop arrived.
// Turn B: hangs until the test settles it, so we can observe that B STARTED a
// real turn (the point of the regression).
let settleB: ((v: { text: string; sessionId: string }) => void) | undefined;
let turnCount = 0;

function makeTurnPromise() {
  turnCount += 1;
  const n = turnCount;
  if (n === 1) {
    return new Promise<{ text: string; sessionId: string }>((resolve) => {
      setTimeout(() => resolve({ text: "A done", sessionId: "uuid-1" }), 5);
    });
  }
  return new Promise<{ text: string; sessionId: string }>((resolve) => {
    settleB = resolve;
  });
}

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(async () => makeTurnPromise());
    this.continueSessionStreaming = vi.fn().mockImplementation(async () => makeTurnPromise());
    this.cancel = vi.fn(); // no-op: A resolves on its own; we test the latch, not the SIGINT
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

describe("Cancel latch must not cancel a fresh prompt queued behind a cancelled turn", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    turnCount = 0;
    settleB = undefined;
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  it("a stale cancel latch (turn A already ended) must not cancel the later-queued prompt B", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    const sid = session.sessionId;

    // Prompt A starts. Its turn resolves end_turn on its own (~5ms), so A
    // passes its early-cancel site (no latch yet) and returns end_turn —
    // mirroring the real flow where A's streaming had already finished.
    const promptA = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "first message" }],
    });
    // Let A's executePrompt reach the streaming loop and A's turn resolve.
    await new Promise((r) => setTimeout(r, 10));
    const resultA = await promptA;
    expect(resultA.stopReason).toBe("end_turn");
    // Flush so A's .finally cleanup and lock settle have fully run (A is gone,
    // inFlightGen cleared).
    await new Promise((r) => setTimeout(r, 10));

    // The user's Stop arrives a moment AFTER A already ended. The cancel
    // latches cancelledSessions (stale — no prompt in flight), but does not
    // record a target generation (inFlightGen is cleared). This is the exact
    // observed race: cancel latches, A has already exited via end_turn (which
    // does not clear the latch).
    await agent.cancel({ sessionId: sid });

    // NOW vscode-acp's drainQueue sends B (the fresh "Send Now" message),
    // ~226ms after the cancel in the real log. B is a NEW prompt (gen 2) that
    // was NOT in flight when the cancel latched. Its executePrompt runs against
    // the stale latch.
    const promptB = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "second message (send now)" }],
    });

    // B's turn starts (reaches the mock). With the generation-scoped latch,
    // B's before-start check must NOT fire (no cancel targeted B's generation).
    await new Promise((r) => setTimeout(r, 20));
    expect(settleB).toBeDefined(); // B started a real turn (reached the streaming mock)

    // Settle B so we can assert on its stopReason.
    settleB!({ text: "B done", sessionId: "uuid-2" });
    const resultB = await promptB;

    // The regression: B was cancelled-before-start (stopReason "cancelled"
    // returned synchronously, never reaching the streaming mock). With the fix,
    // B starts a real turn and settles with end_turn.
    expect(resultB.stopReason).toBe("end_turn");
  });
});
