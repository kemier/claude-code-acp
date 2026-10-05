import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

// Mock node:fs so SessionStore doesn't read real persisted sessions
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

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (
        _cwd: string,
        _prompt: string,
        onEvent: (event: StreamEvent) => void
      ) => {
        for (const e of scriptedEvents) onEvent(e);
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
  return { ClaudeRunner: MockClaudeRunner };
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
  await agent.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "go" }],
  });
  const calls = (conn.sessionUpdate as any).mock.calls;
  const updates = calls.map((c: any) => c[0].update);
  return { updates };
}

describe("tool_progress → tool_call_update forwarding", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scriptedEvents = [];
  });

  it("refreshes a running tool with the elapsed time so the UI ticks", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: { command: "./poll_bear.sh" },
        toolUseId: "tu_slow",
      },
      {
        type: "tool_progress",
        toolUseId: "tu_slow",
        toolName: "Bash",
        toolElapsedSeconds: 61,
        toolSidechain: false,
      },
      { type: "user", text: "result", toolResultIds: ["tu_slow"] },
    ]);
    const frames = updates.filter(
      (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
    );
    // tool_call(in_progress) → tool_call_update(in_progress, elapsed) → completed
    expect(frames).toHaveLength(3);
    expect(frames[0].sessionUpdate).toBe("tool_call");
    expect(frames[0].status).toBe("in_progress");
    const progress = frames[1];
    expect(progress.sessionUpdate).toBe("tool_call_update");
    expect(progress.status).toBe("in_progress");
    expect(progress.toolCallId).toBe(frames[0].toolCallId);
    expect((progress as any)._meta?.claudeCode?.toolProgress?.elapsedSeconds).toBe(61);
    expect(frames[2].status).toBe("completed");
  });

  it("ignores a heartbeat for a tool this session never started", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_progress",
        toolUseId: "tu_unknown",
        toolName: "Bash",
        toolElapsedSeconds: 5,
        toolSidechain: false,
      },
    ]);
    const frames = updates.filter(
      (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
    );
    expect(frames).toHaveLength(0);
  });

  it("ignores a sidechain heartbeat (a subagent's tool, not the parent's)", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: { command: "sleep 600" },
        toolUseId: "tu_main",
      },
      {
        // Same-shape heartbeat, but for a tool inside a subagent: its id is
        // not in the parent's pending map and must not touch the parent row.
        type: "tool_progress",
        toolUseId: "tu_child",
        toolName: "Bash",
        toolElapsedSeconds: 120,
        toolSidechain: true,
      },
    ]);
    const frames = updates.filter(
      (u: any) => u.sessionUpdate === "tool_call" || u.sessionUpdate === "tool_call_update",
    );
    // Only the tool_use's own in_progress emit + the turn-end flush.
    expect(frames).toHaveLength(2);
    expect(frames[1].status).toBe("completed");
  });
});
