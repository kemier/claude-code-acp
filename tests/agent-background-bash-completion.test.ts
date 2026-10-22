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
 * Background Bash completion injection (Option B).
 *
 * In headless `claude -p` the CLI returns an immediate "Command running in
 * background…" tool_result for a `run_in_background: true` Bash command and
 * delivers the later completion as its own `<task-notification>` turn when
 * the process is idle. If that completion is already in hand when the turn
 * ends (no model turn followed it), the adapter injects the output. If the
 * command is still running, it must NOT inject a "still pending" turn — that
 * holds the process busy and the CLI absorbs the real notification into it.
 *
 * The mock stream fires the events the runner would: the tool_use (launch),
 * the launch-ack tool_result (same turn), the result/end_turn, and a SEPARATE
 * user event reusing the tool_use_id with the final completion text.
 */
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      (_cwd: string, _prompt: string, onEvent: any) => {
        // 1. Launch: a backgrounded Bash command.
        onEvent({
          type: "tool_use",
          toolName: "Bash",
          toolUseId: "bash-1",
          toolInput: { command: "node scripts/urlsweep.js", run_in_background: true },
        });
        // 2. Launch-ack tool_result (same turn) — "still running".
        onEvent({
          type: "user",
          text: "Command running in background",
          toolResultIds: ["bash-1"],
        });
        // 3. The model ends the turn.
        onEvent({ type: "result", stopReason: "end_turn", usage: {} });
        // 4. The completion re-injects as a separate user event reusing the
        //    same tool_use_id, with the final output.
        onEvent({
          type: "user",
          text: "node scripts/urlsweep.js finished, exit code 0",
          toolResultIds: ["bash-1"],
        });
        return Promise.resolve({ text: "I'll report the sweep results soon", sessionId: "uuid-1" });
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
    this.hasIdleNotificationWaiter = vi.fn().mockReturnValue(false);
    this.waitForIdleNotification = vi.fn().mockResolvedValue({ text: "", sessionId: "uuid-1", superseded: true });
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

describe("Background Bash completion injection (Option B)", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn, { pendingAgentWaitMs: 100 });
  });

  it("injects the finished background command's output as a synthetic follow-up turn", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    } as any);
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "Run the URL sweep in the background" }],
    } as any);

    // The completion (case 4) was injected as its OWN turn via
    // continueSessionStreaming, with a synthetic message that carries the
    // final output and explicitly tells the model NOT to assume it is running.
    // We assert on the user_message_chunk echoes: the synthetic "[Auto] …have
    // finished" text must appear as a user chunk.
    const cont = (conn as any).sessionUpdate as any;
    const userChunks = cont.mock.calls
      .map((c: unknown[]) => c?.[0]?.update)
      .filter((u: any) => u?.sessionUpdate === "user_message_chunk");
    const synthetic = userChunks.find((u: any) =>
      typeof u?.content?.text === "string" && u.content.text.includes("have finished"),
    );
    expect(synthetic).toBeTruthy();
    expect(synthetic.content.text).toContain("node scripts/urlsweep.js");
    expect(synthetic.content.text).toContain("Do not assume they are still running");
    // The completion's output was captured and included.
    expect(synthetic.content.text).toMatch(/finished, exit code 0/);
  }, 20_000);

  it("does NOT inject a launch-ack as a completion (same tool_use_id, non-completion text)", async () => {
    // Replace the mock so the ONLY user event is the launch-ack (no separate
    // completion). The turn must end without a synthetic follow-up; the
    // adapter arms an idle wait for the CLI's own task-notification.
    vi.resetModules();
    vi.doMock("../src/claude-runner.js", () => {
      const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
        this.startSessionStreaming = vi.fn().mockImplementation((_cwd: string, _prompt: string, onEvent: any) => {
          onEvent({ type: "tool_use", toolName: "Bash", toolUseId: "bash-2", toolInput: { command: "long-job.sh", run_in_background: true } });
          onEvent({ type: "user", text: "Command running in background", toolResultIds: ["bash-2"] });
          onEvent({ type: "result", stopReason: "end_turn", usage: {} });
          return Promise.resolve({ text: "started", sessionId: "uuid-2" });
        });
        this.continueSessionStreaming = vi.fn().mockResolvedValue({ text: "ok", sessionId: "uuid-2" });
        this.cancel = vi.fn();
        this.setSessionPermissionMode = vi.fn();
        this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
        this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
        this.hasIdleNotificationWaiter = vi.fn().mockReturnValue(false);
        this.waitForIdleNotification = vi.fn().mockResolvedValue({ text: "", sessionId: "uuid-2", superseded: true });
        this.startSessionStreamingWithContent = vi.fn().mockImplementation(
          (_cwd: string, _content: any, _onEvent: any, _t?: string, _m?: string, _e?: string, _mcp?: any) =>
            this.startSessionStreaming(_cwd, "", _onEvent, _t, _m, _e)
        );
        this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
          (_sid: string, _content: any, _onEvent: any, _t?: string, _m?: string, _e?: string, _mcp?: any) =>
            this.continueSessionStreaming(_sid, "", _onEvent, _t, _m, _e)
        );
      });
      return { ClaudeRunner: MockClaudeRunner };
    });
    const { createClaudeCodeAgent: fresh } = await import("../src/agent.js");
    const freshConn = createMockConnection();
    const freshAgent = fresh(freshConn, { pendingAgentWaitMs: 100 });
    await freshAgent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "t", version: "1" } } as any);
    const s = await freshAgent.newSession({ cwd: "/tmp", mcpServers: [] });
    await freshAgent.prompt({ sessionId: s.sessionId, prompt: [{ type: "text", text: "start" }] } as any);

    const cont = (freshConn as any).sessionUpdate as any;
    const userChunks = cont.mock.calls
      .map((c: unknown[]) => c?.[0]?.update)
      .filter((u: any) => u?.sessionUpdate === "user_message_chunk");
    const synthetic = userChunks.find((u: any) =>
      typeof u?.content?.text === "string" &&
      (u.content.text.includes("not yet reported") || u.content.text.includes("have finished")),
    );
    expect(synthetic).toBeFalsy();
    const { ClaudeRunner } = await import("../src/claude-runner.js");
    const runner = (ClaudeRunner as any).mock.results.at(-1)?.value;
    expect(runner.waitForIdleNotification).toHaveBeenCalled();
  }, 20_000);
});
