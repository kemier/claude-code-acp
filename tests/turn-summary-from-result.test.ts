import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

/**
 * A turn that ends on tool calls with NO streamed assistant text carries its
 * summary only in the result event's text. The CLI prints it; the adapter used
 * to drop it, so the panel showed tool calls + end_turn and the summary
 * appeared only after Reload Window (session/load replay renders it from the
 * transcript). The result handler now surfaces that text as the turn's final
 * assistant bubble. A turn that DID stream text must not get a second bubble.
 */

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

// Mock runner: a turn that runs a tool then ends with result text and NO
// streamed assistant text (the "tool calls + end_turn, summary missing live"
// case). The result text must surface as one agent_message_chunk.
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: StreamEvent) => void) => {
        onEvent({ type: "tool_use", toolName: "Bash", toolInput: { command: "ls" }, toolUseId: "toolu_1" } as any);
        onEvent({
          type: "result",
          text: "Here is the summary of what the tools found.",
          sessionId: "uuid-summary",
          usage: { input_tokens: 10, output_tokens: 5 },
          stopReason: "end_turn",
        } as any);
        return { text: "Here is the summary of what the tools found.", sessionId: "uuid-summary" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({ text: "ok", sessionId: "uuid-summary" });
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

describe("turn summary surfaces from result text when no text streamed", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

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

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  it("tool-only turn: result text surfaces as one agent_message_chunk", async () => {
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "t", version: "1" } });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "go" }] });

    const updates = ((conn.sessionUpdate as any).mock.calls as any[]).map((c) => c[0].update);
    const msgTexts = updates
      .filter((u: any) => u.sessionUpdate === "agent_message_chunk")
      .map((u: any) => (u.content as any).text);
    expect(msgTexts.some((t: string) => t.includes("summary of what the tools found"))).toBe(true);
  });
});
