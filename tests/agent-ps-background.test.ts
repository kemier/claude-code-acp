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

type Gate = { release: () => void };

function makeGate(): Gate & Promise<{ text: string; sessionId: string }> {
  let resolveFn: (v: { text: string; sessionId: string }) => void;
  const promise = new Promise<{ text: string; sessionId: string }>((res) => {
    resolveFn = res;
  });
  return Object.assign(promise, {
    release: () => resolveFn({ text: "ok", sessionId: "uuid-1" }),
  });
}

// Each test configures the next startSessionStreaming invocation via this
// hook: it may emit tool events through onEvent and return a Promise that
// gates the turn's completion (so the turn is in flight while a mid-turn
// /ps is dispatched).
let nextStartTurn:
  | ((onEvent: (e: any) => void) => Promise<{ text: string; sessionId: string }>)
  | undefined;

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      (_cwd: string, _prompt: string, onEvent: any) =>
        (nextStartTurn ?? ((_e: any) => Promise.resolve({ text: "ok", sessionId: "uuid-1" })))(onEvent),
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

/** A turn that launches one background Bash command (launch-ack only, no
 *  completion) so it stays in sessionBackgroundBash. */
function emitBashLaunch(onEvent: (e: any) => void, toolUseId: string): void {
  onEvent({
    type: "tool_use",
    toolName: "Bash",
    toolUseId,
    toolInput: { command: "npm run watch", run_in_background: true },
  });
  onEvent({
    type: "user",
    text: "Command running in background",
    toolResultIds: [toolUseId],
  });
}

describe("/ps lists real background Bash commands", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    nextStartTurn = undefined;
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn, { pendingAgentWaitMs: 100 });
  });

  it("lists a running background command after a turn that started one", async () => {
    nextStartTurn = (onEvent) => {
      emitBashLaunch(onEvent, "bash-ps-1");
      onEvent({ type: "result", stopReason: "end_turn", usage: {} });
      return Promise.resolve({ text: "ok", sessionId: "uuid-1" });
    };

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
    nextStartTurn = (_onEvent) => Promise.resolve({ text: "ok", sessionId: "uuid-1" });

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

  it("resolves a mid-turn /ps promptly while the turn that started the bg command is still running", async () => {
    // Regression: the read-only fast path used to join the per-session
    // serialization chain BEFORE checking isReadOnlyCommand and then return
    // early, leaving its chain entry dangling until the in-flight turn
    // settled. A mid-turn /ps therefore took ~turn duration to answer, and
    // any prompt queued behind it (e.g. a second /ps) waited the same.
    const gate = makeGate();
    nextStartTurn = (onEvent) => {
      emitBashLaunch(onEvent, "bash-mid-1");
      // No result event: the turn stays in flight until the gate releases.
      return gate as any;
    };

    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    } as any);
    const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

    // Start the turn; it will not settle on its own.
    const turnPromise = agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "start the watch" }],
    } as any);

    // Let the turn's events (tool_use + launch-ack) be processed, and let the
    // first prompt join the per-session lock chain.
    await new Promise((r) => setTimeout(r, 100));

    // Mid-turn /ps: must resolve quickly and report the live bg command —
    // not wait for the in-flight turn to settle.
    const t0 = Date.now();
    const psPromise = agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/ps" }],
    } as any);
    const psResult = await psPromise;
    const psElapsed = Date.now() - t0;
    expect(psResult.stopReason).toBe("end_turn");
    expect(psElapsed).toBeLessThan(2000);

    const out = commandOutputTexts(conn);
    const psLine = out.find((t) => /background command/.test(t));
    expect(psLine).toBeTruthy();
    expect(psLine).toContain("npm run watch");

    // A SECOND /ps right after must also resolve promptly — with the old
    // bug it would have waited on the dangling chain entry for the turn.
    const t1 = Date.now();
    const ps2Result = await agent.prompt({
      sessionId: session.sessionId,
      prompt: [{ type: "text", text: "/ps" }],
    } as any);
    const ps2Elapsed = Date.now() - t1;
    expect(ps2Result.stopReason).toBe("end_turn");
    expect(ps2Elapsed).toBeLessThan(2000);

    // Release the gated turn so the test process can settle.
    gate.release();
    await turnPromise;
  }, 20_000);
});
