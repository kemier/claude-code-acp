// Regression: manual /compact left the context ring showing the PRE-compaction
// value until the FIRST prompt after compaction finished (that turn's result
// event sent the next usage_update frame, which finally repainted the ring).
// The ring is fed only by usage_update frames; auto-compact sends one after
// its continuation turn, but manual /compact has no continuation — so the
// last frame the client saw was the pre-compaction usage, and the ring never
// dropped at the "compacted" notice.
//
// Fix: doCompact emits a post-compaction usage_update (from the reset+reseeded
// estimator, which now holds exactly the fresh session's summary) right after
// the "completed" lifecycle frame.

import { describe, it, expect, vi, beforeEach } from "vitest";
import { createRequire } from "node:module";
import { resolve } from "node:path";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

const require = createRequire(__filename);
const readTextFile: (p: string) => string = require("./src-file-reader.cjs").readTextFile;

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

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: any) => void) => {
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "streamed", sessionId: "claude-uuid-abc" };
      },
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sid: string, _prompt: string, onEvent: (e: any) => void) => {
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "a short summary", sessionId: "claude-uuid-abc" };
      },
    );
    // Content-block variants delegate to the text variants (the adapter
    // routes prompts through them when the prompt carries content blocks).
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string) =>
        this.startSessionStreaming(_cwd, typeof _content?.[0]?.text === "string" ? _content[0].text : "", _onEvent),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string) =>
        this.continueSessionStreaming(_sid, typeof _content?.[0]?.text === "string" ? _content[0].text : "", _onEvent),
    );
    this.cancel = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.isRunning = vi.fn().mockReturnValue(false);
    this.setSessionPermissionMode = vi.fn();
  });
  return { ClaudeRunner: MockClaudeRunner };
});

describe("post-compaction usage_update repaints the context ring", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(async () => {
    vi.clearAllMocks();
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

  async function compactFrames() {
    const sid = (await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] })).sessionId;
    // Establish a real turn first: creates the session estimator and emits a
    // pre-compaction usage_update (the value the ring would keep without the fix).
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "hello" }] });
    const before = ((conn.sessionUpdate as any).mock.calls).length;
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "/compact" }] });
    return (((conn.sessionUpdate as any).mock.calls).slice(before)).map((c: any) => c[0]?.update);
  }

  it("manual /compact emits a usage_update immediately after the completed lifecycle frame", async () => {
    const frames = await compactFrames();
    const completedIdx = frames.findIndex(
      (u) => u?.sessionUpdate === "tool_call_update" && u?._meta?.contextCompaction && u?.status === "completed",
    );
    expect(completedIdx).toBeGreaterThanOrEqual(0);
    const next = frames[completedIdx + 1];
    expect(next?.sessionUpdate).toBe("usage_update");
    // The estimate is the reseeded estimator (summary text only) — small but
    // non-zero, so the ring visibly drops at the notice instead of holding the
    // pre-compaction value until the next prompt.
    expect(next.used).toBeGreaterThan(0);
    expect(next.size).toBeGreaterThan(next.used);
  });

  it("the frame is guarded by !skipCompleted so deferred paths do not double-fire", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const fnAt = src.indexOf("async function doCompact(");
    expect(fnAt).toBeGreaterThan(0);
    const body = src.slice(fnAt, fnAt + 8000);
    const completedAt = body.indexOf('emitCompactionLifecycle(connection, sessionId, compactionId, "completed")');
    expect(completedAt).toBeGreaterThan(0);
    const usageAt = body.indexOf('sessionUpdate: "usage_update"', completedAt);
    expect(usageAt).toBeGreaterThan(completedAt);
    const guard = body.slice(Math.max(0, completedAt - 200), completedAt);
    expect(guard).toContain("if (!skipCompleted) {");
    const between = body.slice(completedAt, usageAt);
    expect(between.includes("skipCompleted")).toBe(false);
  });
});
