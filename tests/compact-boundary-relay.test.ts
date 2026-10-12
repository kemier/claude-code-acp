// Regression: the Claude Code CLI runs its OWN auto-compact. Nothing in this
// adapter disables it, so it can rewrite the conversation mid-turn — the next
// API call reports a tiny input_tokens, the adapter forwards that usage_update,
// and the client's context ring drops with NO chip / notice / spinner. The CLI
// does announce the compaction (`system`/`compact_boundary`), but the runner had
// no branch for that subtype, so the announcement was discarded.
//
// Fix: the runner parses the frame and the agent relays it as the same
// completion lifecycle `doCompact` emits (tool_call_update +
// _meta.contextCompaction + status:"completed"), followed by a usage_update so
// the ring repaints at the notice — exactly what the adapter-owned path does.

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

// Emission is staged per test: the mocked runner replays whatever boundary
// events the test queues, mid-turn, exactly where the real runner would.
const staged: any[] = [];

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    const drive = async (onEvent: (e: any) => void) => {
      onEvent({ type: "assistant", apiCallIndex: 0 });
      for (const e of staged) onEvent(e);
      onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
      return { text: "streamed", sessionId: "claude-uuid-abc" };
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
  });
  return { ClaudeRunner: MockClaudeRunner };
});

describe("CLI native compact_boundary relays a compaction lifecycle", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(async () => {
    vi.clearAllMocks();
    staged.length = 0;
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

  async function turnFrames() {
    const sid = (await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] })).sessionId;
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "hello" }] });
    return ((conn.sessionUpdate as any).mock.calls).map((c: any) => c[0]?.update);
  }

  it("emits a completed contextCompaction frame followed by a usage_update seeded from post_tokens", async () => {
    staged.push({
      type: "compact_boundary",
      compactTrigger: "auto",
      compactPreTokens: 190000,
      compactPostTokens: 18000,
    });
    const frames = await turnFrames();

    const completedIdx = frames.findIndex(
      (u) => u?.sessionUpdate === "tool_call_update" && u?._meta?.contextCompaction && u?.status === "completed",
    );
    expect(completedIdx).toBeGreaterThanOrEqual(0);
    expect(frames[completedIdx].title).toBe("Context compacted");

    const next = frames[completedIdx + 1];
    expect(next?.sessionUpdate).toBe("usage_update");
    // The CLI's own post-compaction measurement seeds the estimator directly —
    // the ring drops to the fresh session size at the notice, not at the next
    // API call.
    expect(next.used).toBe(18000);
    expect(next.size).toBeGreaterThan(next.used);
  });

  it("ignores a sidechain (subagent) compaction so it cannot wipe the parent panel", async () => {
    staged.push({
      type: "compact_boundary",
      compactTrigger: "auto",
      compactPostTokens: 18000,
      isSidechain: true,
    });
    const frames = await turnFrames();
    expect(
      frames.some((u) => u?.sessionUpdate === "tool_call_update" && u?._meta?.contextCompaction),
    ).toBe(false);
  });

  it("a boundary with no post_tokens still relays the notice", async () => {
    staged.push({ type: "compact_boundary", compactTrigger: "manual", compactPreTokens: 190000 });
    const frames = await turnFrames();
    const completedIdx = frames.findIndex(
      (u) => u?.sessionUpdate === "tool_call_update" && u?._meta?.contextCompaction && u?.status === "completed",
    );
    expect(completedIdx).toBeGreaterThanOrEqual(0);
    // Falls back to the (reset) estimator rather than crashing on undefined.
    const next = frames[completedIdx + 1];
    expect(next?.sessionUpdate).toBe("usage_update");
  });
});
