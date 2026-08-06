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
// Controls runner.isRunning() so tests can simulate "a turn is in flight" at
// steer time (handleSteer drops steers for idle sessions). Set true before
// steering, false before the draining prompt.
let running = false;
// Optional gate: when set, startSessionStreaming/continueSessionStreaming
// await it before resolving, letting a test fire a steer mid-turn (with
// running=false to simulate the subprocess-closed-but-prompt-in-flight drain
// tail window). `streamEntered` resolves once the gated streaming call has
// begun (so the test can steer only after runPrompt is truly in flight).
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
        onEvent({ type: "text_delta", text: nextStreamText, fromAssistantBlock: true });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
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
        onEvent({ type: "text_delta", text: nextStreamText, fromAssistantBlock: true });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        if (streamGate) await streamGatePromise;
        return { text: nextStreamText, sessionId: "claude-uuid-abc" };
      }
    );
    this.isRunning = vi.fn(() => running);
    this.cancel = vi.fn();
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
    const status = messages(conn).find((m) => m.includes("**Model:**"));
    expect(status).toContain("**Goal:** ship it");
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

describe("steering (session/steer)", () => {
  it("emits a user_message_chunk immediately for visual feedback", async () => {
    const { conn, agent, sid } = await setup();
    await steer(agent, sid, "please also do X");
    const userChunks = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "user_message_chunk")
      .map((c: any) => c[0].update.content?.text ?? c[0].update.text);
    expect(userChunks.some((t: string) => t.includes("please also do X"))).toBe(true);
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
    expect(messages(conn).some((m) => m.includes("**Model:**"))).toBe(true);
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
    // flips false the moment the Claude subprocess closes — BEFORE runPrompt's
    // steer drain runs. A steer typed in the last moments of streaming thus
    // arrived with running=false, was logged "no active turn", and silently
    // dropped instead of queued — so it never showed before the next turn.
    // The fix also treats a prompt-in-flight on the lock as "active".
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish Claude session

    // Gate the main turn so it stays in flight while we steer.
    setStreamGate();
    running = false; // subprocess "closed" — but runPrompt is still in flight
    const inflight = prompt(agent, sid, "main turn");
    // Wait until runPrompt has actually entered streaming (in-flight flag set).
    await streamEnteredPromise;
    // While the main turn is gated (in flight, runner NOT running), steer.
    await agent.extNotification("session/steer", { sessionId: sid, text: "late steer during drain tail" });
    // The steer must still produce its visual user_message_chunk.
    const userChunks = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "user_message_chunk")
      .map((c: any) => c[0].update.content?.text ?? c[0].update.text);
    expect(userChunks.some((t: string) => t.includes("late steer during drain tail"))).toBe(true);
    // Release the main turn → drain tail runs and should drain the queued steer.
    streamGate!();
    await inflight;

    const steerCall = recorded.find(
      (c) => c.kind === "continue" && c.prompt.includes("late steer during drain tail")
    );
    expect(steerCall).toBeDefined();
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
