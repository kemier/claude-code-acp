/**
 * Regression: mid-turn auto-compact used to leave ghost entries in
 * sessionPendingAgents after SIGINT/discard. drainSteerQueue then deferred
 * forever ("N subagent(s) still in_progress"), so a steer queued before
 * compact never drained — vscode-acp's claude-drain hold stayed armed until
 * the next user prompt.
 *
 * Fix: clear pending agents when mid-turn compact arms / process drops, skip
 * the false auto-follow-up, and arm the post-turn steer drain AFTER compact.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createClaudeCodeAgent } from "../src/agent.js";
import { clearClaudeSettingsCache } from "../src/claude-settings.js";

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  readdirSync: vi.fn(() => []),
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

interface RecordedCall {
  kind: "start" | "continue";
  prompt: string;
}

let recorded: RecordedCall[] = [];
let emitCompactFixture = false;
let capturedAgentRef: any = null;
let capturedSessionRef: string | null = null;
const STEER_TEXT = "AFTER-COMPACT-STEER";

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    const stream = async (kind: "start" | "continue", prompt: string, onEvent: (e: any) => void) => {
      recorded.push({ kind, prompt });

      // Summary turn for doCompact — return non-empty text, no side effects.
      if (prompt.includes("concise summary of our conversation")) {
        onEvent({ type: "result", usage: { input_tokens: 100, output_tokens: 50 } });
        return { text: "summary of prior work", sessionId: "claude-uuid-abc" };
      }

      // Fresh session seed after compact.
      if (prompt.includes("Previous conversation summary")) {
        onEvent({ type: "result", usage: { input_tokens: 200, output_tokens: 20 } });
        return { text: "ready", sessionId: "claude-uuid-fresh" };
      }

      if (emitCompactFixture) {
        emitCompactFixture = false;
        onEvent({
          type: "tool_use",
          toolName: "Task",
          toolInput: { description: "investigate", subagent_type: "general-purpose" },
          toolUseId: "tu_subagent_compact",
        });
        onEvent({ type: "assistant", apiCallIndex: 0 });
        // Cross the auto-compact threshold → midTurnCompactArmed + cancel.
        onEvent({
          type: "message_delta",
          usage: { input_tokens: 6_000, output_tokens: 100 },
        });
        // Queue a steer while the turn is still in flight.
        void Promise.resolve().then(() => {
          capturedAgentRef?.extNotification("session/steer", {
            sessionId: capturedSessionRef,
            text: STEER_TEXT,
          });
        });
        onEvent({
          type: "result",
          stopReason: "end_turn",
          usage: { input_tokens: 6_000, output_tokens: 100 },
        });
        return { text: "", sessionId: "claude-uuid-abc" };
      }

      onEvent({ type: "assistant", apiCallIndex: 0 });
      onEvent({ type: "text_delta", text: "ok", fromAssistantBlock: true });
      onEvent({ type: "result", usage: { input_tokens: 100, output_tokens: 20 } });
      return { text: "ok", sessionId: "claude-uuid-abc" };
    };

    this.startSessionStreaming = vi
      .fn()
      .mockImplementation(async (_cwd: string, prompt: string, onEvent: any) =>
        stream("start", prompt, onEvent),
      );
    this.continueSessionStreaming = vi
      .fn()
      .mockImplementation(async (_sid: string, prompt: string, onEvent: any) =>
        stream("continue", prompt, onEvent),
      );
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, content: any, onEvent: any) =>
        this.startSessionStreaming(
          _cwd,
          typeof content?.[0]?.text === "string" ? content[0].text : "",
          onEvent,
        ),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, content: any, onEvent: any) =>
        this.continueSessionStreaming(
          _sid,
          typeof content?.[0]?.text === "string" ? content[0].text : "",
          onEvent,
        ),
    );
    this.startSession = vi.fn().mockResolvedValue({ text: "", sessionId: "claude-uuid-abc" });
    this.continueSession = vi.fn().mockResolvedValue({ text: "", sessionId: "claude-uuid-abc" });
    this.cancel = vi.fn();
    this.discardLiveSession = vi.fn();
    this.setSpawnEnvOverlay = vi.fn();
    this.isRunning = vi.fn().mockReturnValue(false);
    this.setSessionPermissionMode = vi.fn();
    this.cancelAll = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
  });
  return { ClaudeRunner: MockClaudeRunner };
});

const savedSettingsDir = process.env.CLAUDE_SETTINGS_DIR;
const fakeSettingsDir = join(tmpdir(), "acp-midturn-compact-no-settings");

beforeEach(() => {
  recorded = [];
  emitCompactFixture = false;
  capturedAgentRef = null;
  capturedSessionRef = null;
  process.env.CLAUDE_SETTINGS_DIR = fakeSettingsDir;
  process.env.ANTHROPIC_MODEL = "glm-5.2";
  process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "10000";
  process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "50"; // threshold = 5000
  delete process.env.DISABLE_AUTO_COMPACT;
  delete process.env.DISABLE_COMPACT;
  delete process.env.CLAUDE_ACP_GOAL_EVAL;
  clearClaudeSettingsCache();
});

afterEach(() => {
  if (savedSettingsDir === undefined) delete process.env.CLAUDE_SETTINGS_DIR;
  else process.env.CLAUDE_SETTINGS_DIR = savedSettingsDir;
  delete process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS;
  delete process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
  clearClaudeSettingsCache();
});

function createMockConnection(): any {
  return {
    sessionUpdate: vi.fn().mockResolvedValue(undefined),
    requestPermission: vi.fn(),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    createTerminal: vi.fn(),
    extMethod: vi.fn().mockResolvedValue({}),
    extNotification: vi.fn().mockResolvedValue(undefined),
    signal: new AbortController().signal,
    closed: new Promise(() => {}),
  };
}

describe("mid-turn compact + steer drain", () => {
  it("delivers a mid-turn steer after compact instead of deferring on ghost subagents", async () => {
    const conn = createMockConnection();
    const agent: any = createClaudeCodeAgent(conn);
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
    const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    capturedAgentRef = agent;
    capturedSessionRef = sessionId;
    emitCompactFixture = true;

    await agent.prompt({
      sessionId,
      prompt: [{ type: "text", text: "long-running research" }],
    });

    // Let the microtask'd steer + background drain settle.
    await vi.waitFor(
      () => {
        const chunks = (conn.sessionUpdate as any).mock.calls
          .map((c: any) => c[0]?.update)
          .filter(
            (u: any) =>
              u?.sessionUpdate === "user_message_chunk" &&
              typeof u?.content?.text === "string" &&
              u.content.text.includes(STEER_TEXT),
          );
        expect(chunks.length).toBeGreaterThanOrEqual(1);
      },
      { timeout: 5000 },
    );

    // No false auto-follow-up for a dying process.
    const autoChunks = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter(
        (u: any) =>
          u?.sessionUpdate === "user_message_chunk" &&
          typeof u?.content?.text === "string" &&
          u.content.text.startsWith("[Auto]") &&
          u.content.text.includes("subagent"),
      );
    expect(autoChunks).toHaveLength(0);

    // Ghost subagent marked failed (not left spinning).
    const failed = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter(
        (u: any) =>
          u?.sessionUpdate === "tool_call_update" && u?.status === "failed",
      );
    expect(failed.length).toBeGreaterThanOrEqual(1);

    // Compaction actually ran.
    expect(
      recorded.some((c) => c.prompt.includes("Previous conversation summary")),
    ).toBe(true);
  });
});
