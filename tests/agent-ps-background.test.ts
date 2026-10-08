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

/**
 * `/ps` now lists REAL background Bash commands (started with
 * run_in_background:true) instead of the old hardcoded "No background
 * terminals running." stub. A command is shown while its completion has not
 * been detected — collectCompletedBackgroundBash removes it from
 * sessionBackgroundBash when it finishes, so the list reflects the live set.
 */
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      (_cwd: string, _prompt: string, onEvent: any) => {
        // Launch a background Bash command, then the launch-ack, then end the
        // turn WITHOUT a completion — so it stays in sessionBackgroundBash.
        onEvent({
          type: "tool_use",
          toolName: "Bash",
          toolUseId: "bash-ps-1",
          toolInput: { command: "npm run watch", run_in_background: true },
        });
        onEvent({
          type: "user",
          text: "Command running in background",
          toolResultIds: ["bash-ps-1"],
        });
        onEvent({ type: "result", stopReason: "end_turn", usage: {} });
        return Promise.resolve({ text: "ok", sessionId: "uuid-1" });
      },
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "ok",
      sessionId: "uuid-1",
    });
    this.cancel = vi.fn();
    this.setSessionPermissionMode = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _t?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _t, _m, _e)
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _t?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _t, _m, _e)
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

function commandOutputTexts(conn: AgentSideConnection): string[] {
  const cont = (conn as any).sessionUpdate as any;
  return cont.mock.calls
    .map((c: unknown[]) => c?.[0]?.update)
    .filter((u: any) => u?._meta?.commandOutput === true && u?.sessionUpdate === "agent_message_chunk")
    .map((u: any) => u.content.text);
}

describe("/ps lists real background Bash commands", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn, { pendingAgentWaitMs: 100 });
  });

  it("lists a running background command after a turn that started one", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    } as any);
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    // A turn that launches a background Bash and ends without a completion.
    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "start the watch" }],
    } as any);

    // /ps is a read-only command (turnSafe) — dispatch it as its own prompt.
    const psPromise = agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/ps" }],
    } as any);
    await psPromise;

    const out = commandOutputTexts(conn);
    const psLine = out.find((t) => /background command/.test(t));
    expect(psLine).toBeTruthy();
    expect(psLine).toContain("npm run watch");
    // Elapsed-time prefix is present (e.g. "[0s]").
    expect(psLine).toMatch(/\[\d+[smh][a-z]* [^\]]+\]|\[\d+s\]/);
  }, 20_000);

  it("reports none when no background command is running", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    } as any);
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/ps" }],
    } as any);

    const out = commandOutputTexts(conn);
    expect(out).toContain("No background terminals running.");
  }, 20_000);
});
