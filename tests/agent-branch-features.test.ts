import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import { clearClaudeSettingsCache, MODEL_ENV_KEYS } from "../src/claude-settings.js";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

const ERICAI_MODELS =
  "glm-5.2,glm-5.1,Qwen3-Coder,deepseek-v4-pro,kimi-k2.7-code,kimi-k2.6,mimo-v2.5-pro,mimo-v2.5,qwen2.5-14b-1m";

// Throwaway path that replaces the user-layer settings dir (~/.claude) for the
// duration of each test. The vi.mock("node:fs") below does not reliably
// intercept claude-settings.ts's readFileSync under vitest 4 / vite 8, so the
// developer's real ~/.claude/settings.json (which sets ANTHROPIC_MODEL etc.)
// would leak into the cached settings env and override the process.env values
// these tests set. CLAUDE_SETTINGS_DIR redirects settingsPaths() to this path,
// which does not exist — readSettingsFile treats an absent file as {} — so
// getClaudeEnv falls back to process.env, which the tests fully control. No
// directory is created or removed: the path is intentionally non-existent.
const savedSettingsDir = process.env.CLAUDE_SETTINGS_DIR;
const fakeSettingsDir = join(tmpdir(), "acp-agent-no-settings");

beforeEach(() => {
  process.env.CLAUDE_SETTINGS_DIR = fakeSettingsDir;
  process.env.ANTHROPIC_MODEL = "glm-5.2";
  process.env.CLAUDE_ACP_AVAILABLE_MODELS = ERICAI_MODELS;
  clearClaudeSettingsCache();
});

afterEach(() => {
  if (savedSettingsDir === undefined) delete process.env.CLAUDE_SETTINGS_DIR;
  else process.env.CLAUDE_SETTINGS_DIR = savedSettingsDir;
  clearClaudeSettingsCache();
});

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

// Recorded streaming calls: each entry is the args passed to
// startSessionStreaming / continueSessionStreaming so tests can assert the
// model + effort overrides and the prompt text that actually reached the CLI.
interface RecordedCall {
  kind: "start" | "continue";
  cwd?: string;
  prompt: string;
  modelOverride?: string;
  effortOverride?: string;
}
let recorded: RecordedCall[] = [];
// The next text the mocked summary/fresh-session turns should return.
let nextStreamText = "streamed response";
// The usage figure the next mocked turn reports in its `result` event.
// Default mirrors the CLI's small usage; tests set it high to drive the
// pre-turn compaction guard.
let nextUsage: { input_tokens: number; output_tokens: number } = {
  input_tokens: 10,
  output_tokens: 5,
};
// Controls runner.isRunning() so tests can simulate "a turn is in flight" at
// steer time (handleSteer drops steers for idle sessions). Set true before
// steering, false before the draining prompt.
let running = false;
// Optional gate: when set, startSessionStreaming/continueSessionStreaming
// await it before resolving, letting a test fire a steer mid-turn (with
// running=false to simulate the subprocess-closed-but-prompt-in-flight drain
// tail window). `streamEntered` resolves once the gated streaming call has
// begun (so the test can steer only after executePrompt is truly in flight).
// Cleared after each use.
let streamGate: (() => void) | null = null;
let streamGatePromise: Promise<void> = new Promise(() => {});
let streamEntered: () => void = () => {};
let streamEnteredPromise: Promise<void> = new Promise(() => {});
function setStreamGate() {
  streamGatePromise = new Promise((resolve) => {
    streamGate = resolve;
  });
  streamEnteredPromise = new Promise((resolve) => {
    streamEntered = resolve;
  });
}

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSession = vi.fn().mockResolvedValue({
      text: "Hello",
      sessionId: "claude-uuid-abc",
    });
    this.continueSession = vi.fn().mockResolvedValue({
      text: "Continued",
      sessionId: "claude-uuid-abc",
    });
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (
        cwd: string,
        prompt: string,
        onEvent: (e: any) => void,
        _trackingId?: string,
        modelOverride?: string,
        effortOverride?: string
      ) => {
        recorded.push({ kind: "start", cwd, prompt, modelOverride, effortOverride });
        if (streamGate) streamEntered();
        onEvent({ type: "assistant", apiCallIndex: 0 });
        // message_delta is the authoritative per-call prompt token count
        // (agent.ts:2674). Without it the estimator never sees a real count
        // and the pre-turn guard can't project the next prompt size.
        onEvent({ type: "message_delta", usage: nextUsage });
        onEvent({ type: "text_delta", text: nextStreamText, fromAssistantBlock: true });
        onEvent({ type: "result", usage: nextUsage });
        if (streamGate) await streamGatePromise;
        return { text: nextStreamText, sessionId: "claude-uuid-abc" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (
        _sid: string,
        prompt: string,
        onEvent: (e: any) => void,
        _trackingId?: string,
        modelOverride?: string,
        effortOverride?: string
      ) => {
        recorded.push({ kind: "continue", prompt, modelOverride, effortOverride });
        if (streamGate) streamEntered();
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "message_delta", usage: nextUsage });
        onEvent({ type: "text_delta", text: nextStreamText, fromAssistantBlock: true });
        onEvent({ type: "result", usage: nextUsage });
        if (streamGate) await streamGatePromise;
        return { text: nextStreamText, sessionId: "claude-uuid-abc" };
      }
    );
    this.isRunning = vi.fn(() => running);
    this.writeSteerToMainTurn = vi.fn(() => false);
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

async function setup() {
  vi.clearAllMocks();
  recorded = [];
  nextStreamText = "streamed response";
  nextUsage = { input_tokens: 10, output_tokens: 5 };
  running = false;
  streamGate = null;
  streamGatePromise = new Promise(() => {});
  streamEntered = () => {};
  streamEnteredPromise = new Promise(() => {});
  const conn = createMockConnection();
  const agent: any = createClaudeCodeAgent(conn);
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: {},
    clientInfo: { name: "test", version: "1.0" },
  });
  const sid = (await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] })).sessionId;
  return { conn, agent, sid };
}

/** Queue a steer as if a turn were in flight (handleSteer drops idle steers). */
async function steer(agent: any, sid: string, text: string) {
  running = true;
  await agent.extNotification("session/steer", { sessionId: sid, text });
  running = false;
}

function prompt(agent: any, sid: string, text: string) {
  return agent.prompt({ sessionId: sid, prompt: [{ type: "text", text }] });
}

function messages(conn: AgentSideConnection): string[] {
  return (conn.sessionUpdate as any).mock.calls
    .filter((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk")
    .map((c: any) => c[0].update.content.text);
}

describe("model & effort pickers", () => {
  it("newSession advertises models from CLAUDE_ACP_AVAILABLE_MODELS and model-aware efforts", async () => {
    const { agent } = await setup();
    const r: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    const opts: any[] = r.configOptions;
    const modelPicker = opts.find((o) => o.id === "model");
    expect(modelPicker).toBeDefined();
    expect(modelPicker.options).toHaveLength(9);
    expect(modelPicker.options.map((o: any) => o.value)).toContain("glm-5.2");
    expect(modelPicker.options.map((o: any) => o.value)).toContain("qwen2.5-14b-1m");
    expect(modelPicker.description).toMatch(/Claude Code settings/);

    const effortPicker = opts.find((o) => o.id === "effort");
    // Default model is glm-5.2 (reasoning) → effort picker present with low/medium/high.
    expect(effortPicker).toBeDefined();
    expect(effortPicker.options.map((o: any) => o.value)).toEqual(["low", "medium", "high"]);
  });

  it("newSession advertises only ANTHROPIC_MODEL when no available-models list is set", async () => {
    // With no explicit CLAUDE_ACP_AVAILABLE_MODELS list, the picker is built
    // from every model-naming env key (MODEL_ENV_KEYS). The developer's real
    // env sets ANTHROPIC_SMALL_FAST_MODEL etc., so scrub them all before
    // pinning the single ANTHROPIC_MODEL this test asserts on, and restore
    // afterward so the scrub does not leak into later tests.
    const saved: Record<string, string | undefined> = {};
    for (const k of MODEL_ENV_KEYS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    try {
      delete process.env.CLAUDE_ACP_AVAILABLE_MODELS;
      process.env.ANTHROPIC_MODEL = "LongCat-2.0";
      clearClaudeSettingsCache();
      const { agent } = await setup();
      const r: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
      const modelPicker = r.configOptions.find((o: any) => o.id === "model");
      expect(modelPicker.options.map((o: any) => o.value)).toEqual(["LongCat-2.0"]);
      expect(modelPicker.currentValue).toBe("LongCat-2.0");
    } finally {
      for (const k of MODEL_ENV_KEYS) {
        if (saved[k] === undefined) delete process.env[k];
        else process.env[k] = saved[k];
      }
    }
  });

  it("Qwen3.8-Flash-Next effort picker offers only low/medium", async () => {
    // The Ray cluster's extra_body_suggestions for Qwen3.8-Flash-Next-FP8
    // advertise only low/medium (verified via `ericai api models.list`):
    // `high` 500s at the deployed chat template and upstream's `xhigh` 422s
    // at the vLLM schema. The picker must not offer a rejected tier.
    process.env.CLAUDE_ACP_AVAILABLE_MODELS = "Qwen3.8-Flash-Next";
    process.env.ANTHROPIC_MODEL = "Qwen3.8-Flash-Next";
    clearClaudeSettingsCache();
    const { agent } = await setup();
    const r: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    const effortPicker = r.configOptions.find((o: any) => o.id === "effort");
    expect(effortPicker).toBeDefined();
    expect(effortPicker.options.map((o: any) => o.value)).toEqual(["low", "medium"]);
  });

  it("selecting a non-reasoning model hides the effort picker", async () => {
    const { agent } = await setup();
    const r1: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    const sid = r1.sessionId;
    await agent.setSessionConfigOption({ sessionId: sid, configId: "model", value: "qwen2.5-14b-1m" });
    const r2: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
    // Re-fetch config for the session by setting mode is not needed; instead
    // drive a prompt and check the recorded effort override is undefined.
    recorded = [];
    await prompt(agent, sid, "hi");
    const call = recorded[recorded.length - 1];
    expect(call.effortOverride).toBeUndefined();
  });

  it("selecting a reasoning model + effort flows --effort to the CLI", async () => {
    const { agent, sid } = await setup();
    await agent.setSessionConfigOption({ sessionId: sid, configId: "model", value: "deepseek-v4-pro" });
    await agent.setSessionConfigOption({ sessionId: sid, configId: "effort", value: "xhigh" });
    recorded = [];
    await prompt(agent, sid, "hi");
    const call = recorded[recorded.length - 1];
    expect(call.modelOverride).toBe("deepseek-v4-pro");
    expect(call.effortOverride).toBe("xhigh");
  });

  it("defaults to high effort for a reasoning model with no override", async () => {
    const { agent, sid } = await setup();
    recorded = [];
    await prompt(agent, sid, "hi");
    const call = recorded[recorded.length - 1];
    // glm-5.2 default effort is "high" — consistent with what the picker shows.
    expect(call.effortOverride).toBe("high");
  });
});

describe("/goal subcommands", () => {
  it("/goal clear reports when no goal is set", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal clear");
    expect(messages(conn).some((m) => m === "Goal cleared.")).toBe(true);
  });

  it("/goal pause with no goal tells the user to set one", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal pause");
    expect(messages(conn).some((m) => m.includes("No goal to pause"))).toBe(true);
  });

  it("/goal resume with no goal tells the user to set one", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal resume");
    expect(messages(conn).some((m) => m.includes("No goal to resume"))).toBe(true);
  });

  it("/goal with over-length text is rejected", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, `/goal ${"x".repeat(4001)}`);
    expect(messages(conn).some((m) => m.includes("at most 4000 characters"))).toBe(true);
    // No LLM turn started for a rejected goal.
    expect(recorded.some((c) => c.prompt.includes("start working toward this goal"))).toBe(false);
  });

  it("an active goal is prepended to subsequent prompts", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "/goal ship the feature");
    recorded = [];
    await prompt(agent, sid, "what is the next step");
    const call = recorded[recorded.length - 1];
    expect(call.prompt).toContain("[Goal: ship the feature]");
    expect(call.prompt).toContain("what is the next step");
  });
});

describe("/ps, /skills, /status", () => {
  it("/ps reports no background terminals", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/ps");
    expect(messages(conn).some((m) => m.includes("No background terminals"))).toBe(true);
  });

  it("/skills lists configured skills or a helpful empty message", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/skills");
    const m = messages(conn).find((x) => x.includes("skills") || x.includes("Skills"));
    expect(m).toBeDefined();
  });

  it("/status shows current goal state", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal ship it");
    await prompt(agent, sid, "/status");
    const status = messages(conn).find((m) => m.includes("Model:"));
    expect(status).toContain("Goal: ship it");
  });
});

describe("goal_control extension method (chip buttons)", () => {
  // Helper: read the latest published goal snapshot from session_info_update
  // calls (the chip's data source), keyed by _meta.codex.goal.
  function latestGoal(conn: AgentSideConnection) {
    const calls = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "session_info_update");
    return calls.length ? (calls[calls.length - 1][0].update as any)._meta?.codex?.goal : null;
  }

  it("goal_control pause flips the snapshot to paused", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal ship the feature");
    await agent.extMethod("_codex/session/goal_control", { sessionId: sid, action: "pause" });
    expect(latestGoal(conn)?.status).toBe("paused");
  });

  it("goal_control clear removes the goal (null snapshot)", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "/goal ship the feature");
    await agent.extMethod("_codex/session/goal_control", { sessionId: sid, action: "clear" });
    expect(latestGoal(conn)).toBe(null);
  });

  it("goal_control on a session with no goal is a no-op (no throw)", async () => {
    const { conn, agent, sid } = await setup();
    await agent.extMethod("_codex/session/goal_control", { sessionId: sid, action: "pause" });
    expect(latestGoal(conn)).toBe(null);
  });

  it("unknown extension method returns {} rather than throwing", async () => {
    const { agent, sid } = await setup();
    const r = await agent.extMethod("_some/unknown_method", { sessionId: sid });
    expect(r).toEqual({});
  });
});

describe("/compact (doCompact)", () => {
  it("summarizes, clears the session, and starts fresh with the summary", async () => {
    const { conn, agent, sid } = await setup();
    // Establish a Claude session first.
    await prompt(agent, sid, "hello");
    recorded = [];
    nextStreamText = "SUMMARY-TEXT";
    await prompt(agent, sid, "/compact");

    // The compact flow: a continue (summary) turn, then a start (fresh session
    // seeded with the summary). The fresh-session prompt must embed the summary.
    const fresh = recorded.find(
      (c) => c.kind === "start" && c.prompt.includes("Previous conversation summary")
    );
    expect(fresh).toBeDefined();
    expect(fresh!.prompt).toContain("SUMMARY-TEXT");

    // Compaction is announced via the dedicated `_meta.contextCompaction`
    // lifecycle, NOT as assistant text. Assistant text would be appended to
    // the reply that just streamed and would never be dismissed.
    const compactionUpdates = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter((u: any) => u?._meta?.contextCompaction === true);
    expect(compactionUpdates).toHaveLength(2);
    expect(compactionUpdates[0]).toMatchObject({
      sessionUpdate: "tool_call",
      kind: "other",
      title: "Context compacting",
      status: "in_progress",
    });
    expect(compactionUpdates[1]).toMatchObject({
      sessionUpdate: "tool_call_update",
      title: "Context compacted",
      status: "completed",
    });
    // Same tool call id for both phases so the client can pair them.
    expect(compactionUpdates[1].toolCallId).toBe(compactionUpdates[0].toolCallId);
    // No compaction text leaks into the assistant stream.
    expect(messages(conn).some((m) => m.includes("Compacting conversation"))).toBe(false);
    expect(messages(conn).some((m) => m.includes("compacted successfully"))).toBe(false);
  });

  it("reports nothing to compact when there is no session", async () => {
    const { conn, agent, sid } = await setup();
    // No prior prompt → no Claude session yet.
    await prompt(agent, sid, "/compact");
    expect(messages(conn).some((m) => m.includes("No conversation to compact"))).toBe(true);
  });
});

describe("pre-turn compaction guard", () => {
  // Regression for the "1-token overflow" class: the post-turn auto-compact
  // check measures the PREVIOUS turn's real prompt count, so a turn that
  // finished at 176,000 (under the 176,947 threshold) yet whose NEXT prompt
  // grows the request past the threshold arrives at vLLM as input + maxOutput
  // > window → HTTP 500. The pre-turn guard projects the about-to-be-sent
  // prompt size (last real count + new prompt's char-estimated tokens) and
  // compacts BEFORE forwarding when it would cross the threshold.
  it("compacts before forwarding a prompt that would cross the threshold", async () => {
    // glm-5.2 window 196608, CLAUDE_AUTOCOMPACT_PCT_OVERRIDE default →
    // threshold = floor(196608 * 0.90) = 176947 (no reserve; see e0bf09d).
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    clearClaudeSettingsCache();
    const { conn, agent, sid } = await setup();

    // Turn 1: establish a real prompt count JUST under the threshold. The
    // estimator's realTokens becomes 176000; the post-turn check sees
    // 176000 < 176947 and does NOT compact.
    nextUsage = { input_tokens: 176_000, output_tokens: 100 };
    await prompt(agent, sid, "seed turn");
    recorded = [];

    // Turn 2: a prompt large enough that 176000 + ceil(chars/3) >= 176947,
    // i.e. >= 2841 chars. The pre-turn guard must fire doCompact before the
    // turn's normal continueSessionStreaming call.
    nextUsage = { input_tokens: 30_000, output_tokens: 100 };
    const bigPrompt = "x".repeat(3000);
    await prompt(agent, sid, bigPrompt);

    // doCompact ran: a continue (summary) then a start (fresh session seeded
    // with the summary) appear BEFORE the turn's own continue.
    const summaryStart = recorded.findIndex(
      (c) => c.kind === "start" && c.prompt.includes("Previous conversation summary"),
    );
    expect(summaryStart).toBeGreaterThanOrEqual(0);
    // The summary start must precede the final continue that runs the actual
    // turn (the bigPrompt), proving compaction happened before forwarding.
    // Suffix match: the adapter prepends the <collaboration_mode> block to the
    // default-mode prompt, so the forwarded prompt is block + bigPrompt.
    const finalContinue = recorded.findIndex(
      (c, i) => i > summaryStart && c.kind === "continue" && (c.prompt === bigPrompt || c.prompt.endsWith(bigPrompt)),
    );
    expect(finalContinue).toBeGreaterThan(summaryStart);

    // Compaction was announced via the contextCompaction lifecycle.
    const compactionUpdates = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter((u: any) => u?._meta?.contextCompaction === true);
    expect(compactionUpdates.length).toBeGreaterThanOrEqual(2);

    delete process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS;
    delete process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
    clearClaudeSettingsCache();
  });

  it("does not compact when the projected prompt stays under the threshold", async () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    clearClaudeSettingsCache();
    const { conn, agent, sid } = await setup();

    // Turn 1: well under threshold.
    nextUsage = { input_tokens: 100_000, output_tokens: 100 };
    await prompt(agent, sid, "seed turn");
    recorded = [];

    // Turn 2: small prompt, projected stays under 176947 → no compaction.
    nextUsage = { input_tokens: 100_500, output_tokens: 100 };
    await prompt(agent, sid, "small follow-up");

    const compactionUpdates = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter((u: any) => u?._meta?.contextCompaction === true);
    expect(compactionUpdates).toHaveLength(0);
    // Only the single continue for the turn — no summary start.
    expect(recorded.some((c) => c.prompt.includes("Previous conversation summary"))).toBe(false);

    delete process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS;
    delete process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE;
    clearClaudeSettingsCache();
  });
});


describe("steering (session/steer)", () => {
  it("does NOT echo a steer on receipt — the echo means delivered", async () => {
    // vscode-acp treats the echo as the receipt that releases the NEXT queued
    // message. Echoing here let several interjections be dequeued client-side while
    // the turn was still busy; they then arrived together and the model answered only
    // the last. The echo now happens in drainSteerQueue, as each message is handed to
    // the model.
    const { conn, agent, sid } = await setup();
    await steer(agent, sid, "please also do X");
    const userChunks = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "user_message_chunk")
      .map((c: any) => c[0].update.content?.text ?? c[0].update.text);
    expect(userChunks.some((t: string) => t.includes("please also do X"))).toBe(false);
  });

  it("drains a queued plain-text steer as a follow-up --resume prompt", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish Claude session
    await steer(agent, sid, "steered follow-up");
    recorded = [];
    await prompt(agent, sid, "next");
    // The steered text is delivered as a continue (resume) prompt.
    const steerCall = recorded.find(
      (c) => c.kind === "continue" && c.prompt.includes("steered follow-up")
    );
    expect(steerCall).toBeDefined();
  });

  it("drains a queued /status steer as a command, not raw text", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish session
    await steer(agent, sid, "/status");
    await prompt(agent, sid, "next");
    // /status dispatched as a command → status output appears, and the literal
    // "/status" is NOT sent to the CLI as a prompt.
    expect(messages(conn).some((m) => m.includes("Model:"))).toBe(true);
    expect(recorded.some((c) => c.prompt === "/status")).toBe(false);
  });

  it("drains a queued /goal <objective> steer and starts a turn for it", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish session
    await steer(agent, sid, "/goal queued goal");
    recorded = [];
    await prompt(agent, sid, "next");
    expect(messages(conn).some((m) => m.startsWith("Goal set:"))).toBe(true);
    expect(
      recorded.some((c) => c.prompt.includes("start working toward this goal"))
    ).toBe(true);
  });

  it("drains a queued /compact steer through the compact flow", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish session
    nextStreamText = "STEER-SUMMARY";
    await steer(agent, sid, "/compact");
    recorded = [];
    await prompt(agent, sid, "next");
    // /compact dispatched during drain → the compaction lifecycle was emitted
    // and the fresh-session seed embeds the summary.
    const compactionUpdates = (conn.sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter((u: any) => u?._meta?.contextCompaction === true);
    expect(compactionUpdates.length).toBeGreaterThanOrEqual(2);
    expect(compactionUpdates.some((u: any) => u.status === "completed")).toBe(true);
    expect(
      recorded.some((c) => c.kind === "start" && c.prompt.includes("Previous conversation summary"))
    ).toBe(true);
  });

  it("queues a steer arriving in the drain-tail window (subprocess closed, prompt in flight)", async () => {
    // Regression: handleSteer used to gate on runner.isRunning() alone, which
    // flips false the moment the Claude subprocess closes — BEFORE executePrompt's
    // steer drain runs. A steer typed in the last moments of streaming thus
    // arrived with running=false, was logged "no active turn", and silently
    // dropped instead of queued — so it never showed before the next turn.
    // The fix also treats a prompt-in-flight on the lock as "active".
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish Claude session

    // Gate the main turn so it stays in flight while we steer.
    setStreamGate();
    running = false; // subprocess "closed" — but executePrompt is still in flight
    const inflight = prompt(agent, sid, "main turn");
    // Wait until executePrompt has actually entered streaming (in-flight flag set).
    await streamEnteredPromise;
    // While the main turn is gated (in flight, runner NOT running), steer.
    await agent.extNotification("session/steer", { sessionId: sid, text: "late steer during drain tail" });
    // The steer must be QUEUED, not echoed yet: the echo now means "handed to the
    // model", which the client uses as the receipt that releases its next queued
    // message. Echoing here would release it while this one is still waiting.
    const chunksBeforeDrain = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "user_message_chunk")
      .map((c: any) => c[0].update.content?.text ?? c[0].update.text);
    expect(chunksBeforeDrain.some((t: string) => t.includes("late steer during drain tail"))).toBe(false);
    // Release the main turn → drain tail runs and should drain the queued steer.
    streamGate!();
    await inflight;

    const steerCall = recorded.find(
      (c) => c.kind === "continue" && c.prompt.includes("late steer during drain tail")
    );
    expect(steerCall).toBeDefined();
    // And the echo lands once it is actually delivered.
    const chunksAfterDrain = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "user_message_chunk")
      .map((c: any) => c[0].update.content?.text ?? c[0].update.text);
    expect(chunksAfterDrain.some((t: string) => t.includes("late steer during drain tail"))).toBe(true);
  });

  it("drops a steer for a genuinely idle session (no prompt in flight)", async () => {
    // When no turn is running AND no prompt is in flight, a steer is a no-op
    // (matches codex-acp's "no active turn" rejection).
    const { agent, sid } = await setup();
    running = false;
    await agent.extNotification("session/steer", { sessionId: sid, text: "orphan steer" });
    recorded = [];
    await prompt(agent, sid, "next");
    // The orphan steer was dropped, not drained.
    expect(recorded.some((c) => c.prompt.includes("orphan steer"))).toBe(false);
  });
});
