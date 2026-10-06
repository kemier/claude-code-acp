import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// These tests cover per-session prompt serialization (agent.ts
// `sessionPromptLocks`). The ACP SDK dispatches each `session/prompt`
// concurrently (its read loop calls #processMessage without awaiting it), so a
// second prompt arriving mid-turn would otherwise run a second prompt()
// handler in parallel — racing on shared per-session state and dropping the
// /skills agent_message_chunk (seen in ACP traffic log 20260731T022841: zero
// agent_message_chunk in the whole log).
//
// This file uses a runner mock whose startSessionStreaming BLOCKS on a
// deferred release, so we can fire a second prompt mid-turn and assert it
// waits (serialized) and still emits its output.

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

// Deferred release for the blocking first turn. Reassigned per call by the
// mock; tests read/write via the helpers below.
let releaseFirstTurn: () => void = () => {};
let firstTurnStarted: () => void = () => {};

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: any) => void) => {
        firstTurnStarted();
        // Block until the test releases this turn — simulates a long-running
        // LLM turn during which a second prompt arrives.
        await new Promise<void>((resolve) => {
          releaseFirstTurn = resolve;
        });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "first", sessionId: "claude-uuid-abc" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sessionId: string, _prompt: string, onEvent: (e: any) => void) => {
        firstTurnStarted();
        await new Promise<void>((resolve) => {
          releaseFirstTurn = resolve;
        });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "continued", sessionId: "claude-uuid-abc" };
      }
    );
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
    this.isRunning = vi.fn(() => true);
    this.writeSteerToMainTurn = vi.fn(() => false);
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

describe("per-session prompt serialization", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(async () => {
    vi.clearAllMocks();
    releaseFirstTurn = () => {};
    firstTurnStarted = () => {};
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
  });

  async function newSession() {
    const r = await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] });
    return r.sessionId;
  }

  it("a read-only slash command mid-turn is not lost — it responds immediately, not after the turn", async () => {
    const sid = await newSession();

    let started = false;
    firstTurnStarted = () => {
      started = true;
    };

    // Fire the first prompt WITHOUT awaiting — it blocks inside the runner.
    const first = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "long running" }],
    });
    await vi.waitFor(() => expect(started).toBe(true));

    // While the first turn is in-flight, fire /skills (as vscode-acp does).
    const skills = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "/skills" }],
    });

    // /skills is read-only and bypasses the per-session prompt lock, so its
    // chunk is emitted immediately — while the first turn is still in flight —
    // not serialized behind it. This is the whole point of the bypass: the
    // user sees the snapshot at send time, not after the in-flight turn ends.
    await vi.waitFor(() => {
      const chunks = (conn.sessionUpdate as any).mock.calls.filter(
        (c: any) =>
          c[0]?.update?.sessionUpdate === "agent_message_chunk" &&
          /Available skills|No skills configured/.test(c[0]?.update?.content?.text ?? "")
      );
      expect(chunks.length).toBeGreaterThan(0);
    });
    // The first turn is STILL in flight when the chunk lands.
    expect(started).toBe(true);

    // Release the first turn; both settle.
    releaseFirstTurn();
    await first;
    await skills;
  });

  it("two regular prompts run sequentially, never concurrently", async () => {
    const sid = await newSession();

    // `running` is true while a turn is inside the runner; `overlap` flips if
    // a second turn starts before the first ends.
    let running = false;
    let overlap = false;
    firstTurnStarted = () => {
      if (running) overlap = true;
      running = true;
    };

    const p1 = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "first" }],
    });
    await vi.waitFor(() => expect(running).toBe(true));

    const p2 = agent.prompt({
      sessionId: sid,
      prompt: [{ type: "text", text: "second" }],
    });
    // p2 is waiting — first turn still running, no overlap.
    await new Promise((r) => setTimeout(r, 10));
    expect(running).toBe(true);
    expect(overlap).toBe(false);

    // Release p1: it finishes, then p2 starts (still no overlap).
    releaseFirstTurn();
    await p1;
    running = false;
    await vi.waitFor(() => expect(running).toBe(true));
    releaseFirstTurn();
    await p2;

    expect(overlap).toBe(false);
  });
});
