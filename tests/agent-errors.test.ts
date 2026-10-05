import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

// Mock that fails on execution
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockRejectedValue(
      new Error("claude: command not found")
    );
    this.continueSessionStreaming = vi.fn().mockRejectedValue(
      new Error("claude exited with code 1")
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

describe("Agent error handling", () => {
  let conn: ReturnType<typeof createMockConnection>;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  it("should throw for unknown session", async () => {
    await expect(
      agent.prompt({
        sessionId: "nonexistent",
        prompt: [{ type: "text", text: "hello" }],
      })
    ).rejects.toThrow();
  });

  it("should throw for empty prompt", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({
      cwd: "/tmp",
      mcpServers: [],
    });

    await expect(
      agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "   " }],
      })
    ).rejects.toThrow();
  });

  it("should gracefully handle claude runner errors", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({
      cwd: "/tmp",
      mcpServers: [],
    });

    // Should not throw - error is sent as agent message
    const result = await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "hello" }],
    });

    expect(result.stopReason).toBe("end_turn");

    // Should have sent error message via sessionUpdate
    const calls = (conn.sessionUpdate as any).mock.calls;
    const errorUpdate = calls.find((c: any) =>
      c[0].update.content?.text?.includes("Error:")
    );
    expect(errorUpdate).toBeDefined();
  });
});
