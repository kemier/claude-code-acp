/**
 * Reported: the local model proxy was killed mid-session, the next turn's
 * `--resume` exited 1 with "API Error: Connection refused", and the resume
 * fallback treated that as an expired session — it cleared the persisted Claude
 * session id and started a FRESH conversation. The user resumed, said "go on",
 * and got "I don't have any prior task in this session to continue from".
 * 68 commands of context abandoned, while the transcript was still on disk.
 *
 * The session id is the only handle on that conversation and nothing in the UI
 * can restore it, so the fallback must be reserved for failures that actually
 * implicate the session id. A transient upstream failure is surfaced instead.
 */

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent, isTransientTurnFailure } from "../src/agent.js";
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

/** Error the mocked resume rejects with; set per test. */
let resumeError: Error | undefined;
/** Emit an API-error result event on the resume attempt before rejecting. */
let resumeEmitsApiError = false;
/** How many times a NEW (non-resume) session was started. */
let freshStarts = 0;

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(async () => {
      freshStarts++;
      return { text: "fresh", sessionId: "uuid-fresh" };
    });
    this.startSessionStreamingWithContent = this.startSessionStreaming;
    this.startSessionWithMcp = this.startSessionStreaming;
    this.continueSessionStreaming = vi
      .fn()
      .mockImplementation(async (_sid: string, _prompt: string, onEvent: any) => {
        if (resumeEmitsApiError) {
          // What the CLI really does: reports the API failure as a result event,
          // then exits non-zero.
          onEvent?.({ type: "result", isError: true, stopReason: "success" });
        }
        if (resumeError) {
          throw resumeError;
        }
        return { text: "resumed", sessionId: "uuid-1" };
      });
    this.continueSessionStreamingWithContent = this.continueSessionStreaming;
    this.isRunning = vi.fn(() => false);
    this.cancel = vi.fn();
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

describe("isTransientTurnFailure", () => {
  it("recognises the reported upstream failures", () => {
    expect(isTransientTurnFailure("claude exited with code 1: API Error: Connection refused")).toBe(true);
    expect(isTransientTurnFailure("connect ECONNREFUSED 127.0.0.1:8317")).toBe(true);
    expect(isTransientTurnFailure("claude exited with code 1: fetch failed")).toBe(true);
    expect(isTransientTurnFailure("Error: socket hang up")).toBe(true);
    expect(isTransientTurnFailure("upstream returned 503")).toBe(true);
    expect(isTransientTurnFailure("429 too many requests")).toBe(true);
    expect(isTransientTurnFailure("request timed out")).toBe(true);
  });

  it("treats an isError result as transient whatever the exit message says", () => {
    // The bare rejection carries no reason on some paths; the stream's own
    // isError result is the reliable signal.
    expect(isTransientTurnFailure("claude exited with code 1", true)).toBe(true);
  });

  it("does NOT claim a stale-session failure is transient", () => {
    expect(isTransientTurnFailure("No conversation found with session ID abc")).toBe(false);
    expect(isTransientTurnFailure("claude exited with code 1")).toBe(false);
    expect(isTransientTurnFailure("")).toBe(false);
  });
});

describe("a resume that fails on a dead upstream", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    resumeError = undefined;
    resumeEmitsApiError = false;
    freshStarts = 0;
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  async function sessionWithClaudeId(): Promise<string> {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    // First turn establishes the Claude session id that must survive.
    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "first" }] });
    freshStarts = 0;
    return session.sessionId;
  }

  it("keeps the conversation instead of starting fresh", async () => {
    const sessionId = await sessionWithClaudeId();
    resumeError = new Error("claude exited with code 1: API Error: Connection refused");
    resumeEmitsApiError = true;

    const res = await agent.prompt({ sessionId, prompt: [{ type: "text", text: "go on" }] });

    // The turn fails visibly — and no fresh conversation was started, so the
    // persisted session id still points at the real conversation.
    expect(res.stopReason).toBe("end_turn");
    expect(freshStarts).toBe(0);
  });

  it("still falls back when the session id itself is the problem", async () => {
    const sessionId = await sessionWithClaudeId();
    resumeError = new Error("No conversation found with session ID uuid-1");
    resumeEmitsApiError = false;

    await agent.prompt({ sessionId, prompt: [{ type: "text", text: "go on" }] });

    expect(freshStarts).toBe(1);
  });
});
