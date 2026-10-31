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

/**
 * Zed and vscode-acp Send Now both wait until session/prompt settles, then
 * submit the queued text as a new session/prompt. Awaiting the subagent
 * auto follow-up inside that RPC kept the client on Generating, so the
 * queued message never started. The follow-up must run after the RPC
 * returns, and the queued prompt must preempt it.
 */
const forwarded: string[] = [];
let releaseFollowUp: (() => void) | undefined;
let followUpStarted: (() => void) | undefined;
const followUpStartedPromise = { current: null as Promise<void> | null };

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: any) => void) => {
        onEvent({
          type: "tool_use",
          toolName: "Task",
          toolInput: { description: "investigate", subagent_type: "general-purpose" },
          toolUseId: "tu_subagent",
        });
        onEvent({
          type: "result",
          stopReason: "end_turn",
          usage: { input_tokens: 10, output_tokens: 5 },
        });
        return { text: "planned", sessionId: "uuid-1" };
      },
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sid: string, prompt: string) => {
        forwarded.push(prompt);
        if (prompt.includes("[Auto]") && prompt.includes("background subagent")) {
          followUpStarted?.();
          return new Promise((resolve) => {
            releaseFollowUp = () => resolve({ text: "incorporated", sessionId: "uuid-1" });
          });
        }
        return { text: "ok", sessionId: "uuid-1" };
      },
    );
    this.cancel = vi.fn().mockImplementation(() => {
      releaseFollowUp?.();
      releaseFollowUp = undefined;
    });
    this.discardLiveSession = vi.fn();
    this.setSessionPermissionMode = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, content: any, onEvent: any, trackingId?: string, m?: string, e?: string, mcp?: any) =>
        this.startSessionStreaming(
          _cwd,
          typeof content?.[0]?.text === "string" ? content[0].text : "",
          onEvent,
          trackingId,
          m,
          e,
          mcp,
        ),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, content: any, onEvent: any, trackingId?: string, m?: string, e?: string, mcp?: any) =>
        this.continueSessionStreaming(
          _sid,
          typeof content?.[0]?.text === "string" ? content[0].text : "",
          onEvent,
          trackingId,
          m,
          e,
          mcp,
        ),
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

describe("Send Now preempts the post-turn subagent follow-up", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    forwarded.length = 0;
    releaseFollowUp = undefined;
    followUpStarted = undefined;
    followUpStartedPromise.current = new Promise<void>((resolve) => {
      followUpStarted = resolve;
    });
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  it("returns end_turn before the follow-up finishes, then the queued prompt runs", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    const resultA = await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "research the plan" }],
    });
    expect(resultA.stopReason).toBe("end_turn");

    await followUpStartedPromise.current;
    expect(forwarded.some((p) => p.includes("[Auto]"))).toBe(true);
    expect(forwarded.some((p) => p.includes("queued send now"))).toBe(false);

    const resultB = await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "queued send now" }],
    });
    expect(resultB.stopReason).toBe("end_turn");
    expect(forwarded.some((p) => p.includes("queued send now"))).toBe(true);
  });
});
