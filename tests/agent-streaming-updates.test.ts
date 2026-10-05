import { describe, it, expect, vi, beforeEach } from "vitest";
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

// Mock that emits thinking and usage events
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: StreamEvent) => void) => {
        onEvent({ type: "thinking", text: "Let me think about this..." } as any);
        onEvent({ type: "text_delta", text: "Here is the answer." });
        onEvent({
          type: "result",
          text: "Here is the answer.",
          sessionId: "uuid-stream",
          usage: {
            input_tokens: 100,
            output_tokens: 50,
          },
          totalTokens: 150,
        } as any);
        return { text: "Here is the answer.", sessionId: "uuid-stream" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "ok", sessionId: "uuid-stream",
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

describe("Agent streaming updates", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  it("should send agent_thought_chunk for thinking events", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "think about this" }],
    });

    const calls = (conn.sessionUpdate as any).mock.calls;
    const updates = calls.map((c: any) => c[0].update);

    const thoughtChunks = updates.filter(
      (u: any) => u.sessionUpdate === "agent_thought_chunk"
    );
    expect(thoughtChunks.length).toBeGreaterThanOrEqual(1);
    expect(thoughtChunks[0].content.type).toBe("text");
    expect(thoughtChunks[0].content.text).toBe("Let me think about this...");
  });

  it("should send session_info_update after prompt completes", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "hello" }],
    });

    const calls = (conn.sessionUpdate as any).mock.calls;
    const updates = calls.map((c: any) => c[0].update);

    const infoUpdates = updates.filter(
      (u: any) => u.sessionUpdate === "session_info_update"
    );
    expect(infoUpdates.length).toBeGreaterThanOrEqual(1);
    expect(infoUpdates[0].updatedAt).toBeDefined();
  });
});

// Regression for the intermittent empty-bubble bug: in --verbose mode the
// Claude CLI streams per-token content_block_delta text AND then a final
// `assistant` block re-emitting the FULL text immediately before the
// tool_use. Re-emitting that final-block text lands it ~1ms before the
// tool_call, beating vscode-acp's 80ms streaming render and leaving an empty
// bubble at every text→tool transition. The final-block text that duplicates
// already-streamed text must be dropped from the UI chunks.
describe("Agent --verbose duplicate final-block text dedup", () => {
  // Shared mock factory: emits the --verbose double-emission pattern.
  function makeVerboseRunnerMock(streamedDeltas: string[], finalText: string, withToolUse: boolean) {
    const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
      this.startSessionStreaming = vi.fn().mockImplementation(
        async (_cwd: string, _prompt: string, onEvent: (e: StreamEvent) => void) => {
          // 1. per-token streaming deltas (content_block_delta path)
          for (const d of streamedDeltas) {
            onEvent({ type: "text_delta", text: d });
          }
          // 2. final assistant block: boundary event, full text, then tool_use
          onEvent({ type: "assistant", apiCallIndex: 0 } as any);
          onEvent({ type: "text_delta", text: finalText, fromAssistantBlock: true });
          if (withToolUse) {
            onEvent({
              type: "tool_use",
              toolName: "Read",
              toolInput: { file_path: "/tmp/x" },
              toolUseId: "tu_1",
            } as any);
          }
          onEvent({
            type: "result",
            text: finalText,
            sessionId: "uuid-verbose",
            usage: { input_tokens: 100, output_tokens: 50 },
          } as any);
          return { text: finalText, sessionId: "uuid-verbose" };
        }
      );
      this.continueSessionStreaming = vi.fn().mockResolvedValue({
        text: finalText, sessionId: "uuid-verbose",
      });
      this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
      this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
      // Delegate the stream-json stdin path to the same event-emitting start
      // so the verbose double-emission pattern is exercised on the main-turn
      // (stdin) path too.
      this.startSessionStreamingWithContent = vi.fn().mockImplementation(
        (_cwd: string, _content: any, onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
          this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), onEvent, _trackingId, _m, _e)
      );
      this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
        (_sid: string, _content: any, onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
          this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), onEvent, _trackingId, _m, _e)
      );
      this.cancel = vi.fn();
    this.setSessionPermissionMode = vi.fn();
    });
    return { ClaudeRunner: MockClaudeRunner };
  }

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

  it("drops the final-block text chunk when text was already streamed token-by-token", async () => {
    vi.doMock("../src/claude-runner.js", () =>
      makeVerboseRunnerMock(["Hello ", "world"], "Hello world", true)
    );
    vi.resetModules();
    const { createClaudeCodeAgent } = await import("../src/agent.js");
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
      prompt: [{ type: "text", text: "hi" }],
    });

    const calls = (conn.sessionUpdate as any).mock.calls;
    const updates = calls.map((c: any) => c[0].update);
    const textChunks = updates.filter(
      (u: any) => u.sessionUpdate === "agent_message_chunk" && u.content?.type === "text"
    );
    const chunkTexts = textChunks.map((u: any) => u.content.text);

    // The two per-token deltas are emitted; the final-block "Hello world" is
    // NOT re-emitted (it duplicates the streamed tokens and would land
    // back-to-back with the tool_call, causing the empty bubble).
    expect(chunkTexts).toEqual(["Hello ", "world"]);
    expect(chunkTexts).not.toContain("Hello world");
  });

  it("still emits text when it appears ONLY in the final block (no streaming deltas)", async () => {
    // Non-verbose / non-streaming model: no content_block_delta, text lives
    // only in the final assistant block. Must not be dropped.
    vi.doMock("../src/claude-runner.js", () =>
      makeVerboseRunnerMock([], "Final-only text", true)
    );
    vi.resetModules();
    const { createClaudeCodeAgent } = await import("../src/agent.js");
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
      prompt: [{ type: "text", text: "hi" }],
    });

    const calls = (conn.sessionUpdate as any).mock.calls;
    const updates = calls.map((c: any) => c[0].update);
    const textChunks = updates.filter(
      (u: any) => u.sessionUpdate === "agent_message_chunk" && u.content?.type === "text"
    );
    const chunkTexts = textChunks.map((u: any) => u.content.text);

    expect(chunkTexts).toContain("Final-only text");
  });
});
