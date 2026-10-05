import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { isProxyBaseUrl } from "../src/claude-runner.js";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

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

let scriptedEvents: StreamEvent[] = [];
let holdRelease: (() => void) | undefined;

vi.mock("../src/claude-runner.js", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/claude-runner.js")>();
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: StreamEvent) => void) => {
        for (const e of scriptedEvents) onEvent(e);
        // Hold the turn open so a mid-turn watchdog can fire before the
        // post-turn settle clears it — mirroring a tool whose result never
        // arrives while the CLI process is still running.
        await new Promise<void>((resolve) => {
          holdRelease = resolve;
        });
        return { text: "done", sessionId: "uuid-1" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "Continued",
      sessionId: "uuid-1",
    });
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
  return { ...actual, ClaudeRunner: MockClaudeRunner };
});

function createMockConnection() {
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

async function runWithEvents(events: StreamEvent[]) {
  scriptedEvents = events;
  const conn = createMockConnection();
  const agent = createClaudeCodeAgent(conn);
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: {},
    clientInfo: { name: "test", version: "1.0" },
  });
  const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  const promptPromise = agent.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "go" }],
  });
  // Let the mock replay its events and park on the hold.
  await Promise.resolve();
  await Promise.resolve();
  return {
    conn,
    finish: async () => {
      holdRelease?.();
      holdRelease = undefined;
      await promptPromise;
      return (conn.sessionUpdate as any).mock.calls.map((c: any) => c[0].update);
    },
  };
}

const savedBase = process.env.ANTHROPIC_BASE_URL;

afterEach(() => {
  if (savedBase === undefined) delete process.env.ANTHROPIC_BASE_URL;
  else process.env.ANTHROPIC_BASE_URL = savedBase;
});

describe("web-tool watchdog fails a hung server-side tool", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    process.env.ANTHROPIC_BASE_URL = "http://127.0.0.1:8317";
  });

  it("flips a hung WebSearch to failed with a named cause after the timeout", async () => {
    vi.useFakeTimers();
    try {
      const run = await runWithEvents([
        {
          type: "tool_use",
          toolName: "WebSearch",
          toolInput: { query: "weather stockholm" },
          toolUseId: "tu_web",
        },
      ]);
      // Advance past the watchdog (production 90s) while the turn is open.
      await vi.advanceTimersByTimeAsync(95_000);
      const updates = await run.finish();
      const frames = updates.filter(
        (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
      );
      expect(frames[0].status).toBe("in_progress");
      const failed = frames.find((u: any) => u.status === "failed");
      expect(failed, "watchdog must flip the hung web tool to failed").toBeTruthy();
      expect(failed.toolCallId).toBe(frames[0].toolCallId);
      expect(failed._meta?.claudeCode?.error).toMatch(/not available on this proxy route/i);
      // The watchdog's verdict must be final — the post-turn flush must not
      // re-complete the same toolCallId after it failed.
      const afterFailed = frames.slice(frames.indexOf(failed) + 1);
      expect(
        afterFailed.some((u: any) => u.toolCallId === failed.toolCallId && u.status === "completed"),
      ).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("leaves a WebFetch alone: the CLI fetches client-side on proxy routes", async () => {
    vi.useFakeTimers();
    try {
      const run = await runWithEvents([
        {
          type: "tool_use",
          toolName: "WebFetch",
          toolInput: { url: "https://example.com" },
          toolUseId: "tu_fetch",
        },
      ]);
      // Advance past the watchdog window while the turn is open — WebFetch
      // must NOT be failed, it returns results even on a proxy route.
      await vi.advanceTimersByTimeAsync(95_000);
      const updates = await run.finish();
      const frames = updates.filter(
        (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
      );
      expect(frames.some((u: any) => u.status === "failed")).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it("fails a stalled WebFetch when the operator names it in CLAUDE_ACP_DISALLOWED_WEB_TOOLS", async () => {
    vi.useFakeTimers();
    process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS = "WebSearch,WebFetch";
    try {
      const run = await runWithEvents([
        {
          type: "tool_use",
          toolName: "WebFetch",
          toolInput: { url: "https://example.com" },
          toolUseId: "tu_fetch_block",
        },
      ]);
      await vi.advanceTimersByTimeAsync(95_000);
      const updates = await run.finish();
      const frames = updates.filter(
        (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
      );
      const failed = frames.find((u: any) => u.status === "failed");
      expect(failed, "env-named WebFetch must be failed after the timeout").toBeTruthy();
      expect(failed._meta?.claudeCode?.error).toMatch(/not available on this proxy route/i);
    } finally {
      delete process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS;
      vi.useRealTimers();
    }
  });
});
