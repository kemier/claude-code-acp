import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

/**
 * The agent must drop sidechain text/thinking emissions — a subagent's own
 * assistant text rendered as agent_message_chunk in the parent's chat
 * ("subagent assistant msg leaked into main agent chat"). Main-conversation
 * emissions must still render.
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

// Mock runner interleaving main + sidechain events.
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: StreamEvent) => void) => {
        onEvent({ type: "text_delta", text: "main says hi" } as any);
        onEvent({ type: "text_delta", text: "SUBAGENT LEAK", isSidechain: true } as any);
        onEvent({ type: "thinking", text: "main thought" } as any);
        onEvent({ type: "thinking", text: "SUBAGENT THOUGHT LEAK", isSidechain: true } as any);
        onEvent({
          type: "result",
          text: "main says hi",
          sessionId: "uuid-sidechain",
          usage: { input_tokens: 10, output_tokens: 5 },
        } as any);
        return { text: "main says hi", sessionId: "uuid-sidechain" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({ text: "ok", sessionId: "uuid-sidechain" });
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

describe("agent drops sidechain text/thinking emissions", () => {
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

  it("sidechain text_delta produces no agent_message_chunk; main text still renders", async () => {
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "t", version: "1" } });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "go" }] });

    const updates = ((conn.sessionUpdate as any).mock.calls as any[]).map((c) => c[0].update);
    const msgTexts = updates
      .filter((u: any) => u.sessionUpdate === "agent_message_chunk")
      .map((u: any) => (u.content as any).text);
    expect(msgTexts).toContain("main says hi");
    expect(msgTexts.some((t: string) => t.includes("SUBAGENT LEAK"))).toBe(false);
  });

  it("sidechain thinking produces no agent_thought_chunk; main thought still renders", async () => {
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "t", version: "1" } });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    await agent.prompt({ sessionId: session.sessionId, prompt: [{ type: "text", text: "go" }] });

    const updates = ((conn.sessionUpdate as any).mock.calls as any[]).map((c) => c[0].update);
    const thoughtTexts = updates
      .filter((u: any) => u.sessionUpdate === "agent_thought_chunk")
      .map((u: any) => (u.content as any).text);
    expect(thoughtTexts).toContain("main thought");
    expect(thoughtTexts.some((t: string) => t.includes("SUBAGENT THOUGHT LEAK"))).toBe(false);
  });
});
