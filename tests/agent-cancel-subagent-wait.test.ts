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

// The turn ends cleanly (resolves), but a background subagent is still pending.
// Without the early cancel ack in prompt(), the agent would sit in the bounded
// subagent wait (pendingAgentWaitMs) BEFORE returning "cancelled" — past the
// host's 5s cancel-fallback, which then races a second promptEnd over the queue
// (the "stop shows the queued message but the turn never ends" strand).
//
// The mock holds the stream open (seeding a pending Task subagent on event)
// until cancel() is called, then resolves — mirroring a real turn where the
// model is still streaming when the user clicks Stop. That means prompt() is
// INSIDE the `await runner.startSessionStreaming(...)` when cancel sets the
// flag, so it only reaches the early-return check AFTER resolving.
vi.mock("../src/claude-runner.js", () => {
  let resolveStream: ((v: any) => void) | undefined;
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: any) => {
        // Seed a pending subagent: a Task tool_use with no task-notification.
        onEvent({
          type: "tool_use",
          toolName: "Task",
          toolUseId: "sub-agent-1",
          toolInput: { description: "verify thing", prompt: "do the work" },
        });
        // Hold the stream open until cancel() resolves it.
        return new Promise((resolve) => {
          resolveStream = resolve;
        });
      },
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "ok",
      sessionId: "uuid-1",
    });
    // Simulate the CLI: on cancel the in-flight stream ends (process SIGINT'd).
    this.cancel = vi.fn().mockImplementation(function () {
      if (resolveStream) {
        resolveStream({ text: "interrupted", sessionId: "uuid-1" });
        resolveStream = undefined;
      }
    });
    this.setSessionPermissionMode = vi.fn();
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

describe("Agent cancel with a settling subagent → fast cancelled ack", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    // Make the (now-skipped) subagent wait long enough that a regression would
    // be obvious: 30s. The fix returns well before that.
    agent = createClaudeCodeAgent(conn, { pendingAgentWaitMs: 30_000 });
  });

  it(
    "returns cancelled promptly instead of waiting out the subagent",
    async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      } as any);
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      const startedAt = Date.now();
      const promptPromise = agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "Hello" }],
      } as any);

      // Let the turn "finish" (mock resolves after seeding the subagent), then
      // cancel. The prompt should resolve to cancelled WITHOUT blocking on the
      // 30s subagent wait.
      await new Promise((r) => setTimeout(r, 50));
      await agent.cancel({ sessionId: session.sessionId });

      const result = await Promise.race([
        promptPromise,
        new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error("TIMED OUT waiting for cancelled ack")), 8_000),
        ),
      ]);

      const elapsed = Date.now() - startedAt;
      expect((result as any).stopReason).toBe("cancelled");
      // Regression guard: must not have waited out pendingAgentWaitMs (30s).
      expect(elapsed).toBeLessThan(8_000);
    },
    20_000,
  );
});
