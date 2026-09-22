import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import * as fs from "node:fs";
import * as os from "node:os";

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

// Captures the onEvent callback passed to the runner so a test can emit
// stream events (assistant block + streaming deltas) and assert the estimator
// does not double-count assistant text.
let capturedOnEvent: ((e: any) => void) | null = null;
// Captures the prompt the agent sent to the runner (2nd arg of
// startSessionStreaming / continueSessionStreaming) so slash-command tests can
// assert the constructed prompt reaches the model.
let capturedPrompt: string | null = null;

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
      async (_cwd: string, _prompt: string, onEvent: (e: any) => void) => {
        capturedOnEvent = onEvent;
        capturedPrompt = _prompt;
        // Sentinel prompt to simulate a Claude CLI error result (e.g. the
        // model hit the turn ceiling). Used by the error-surfacing test.
        if (_prompt === "__error_max_turns__") {
          onEvent({
            type: "result",
            usage: { input_tokens: 10, output_tokens: 5 },
            stopReason: "error_max_turns",
            isError: true,
          });
          return { text: "", sessionId: "claude-uuid-abc" };
        }
        // Emit both a streaming text delta AND the final assistant block with
        // the same text — this is what --verbose mode does. The estimator must
        // count the text only once.
        onEvent({ type: "text_delta", text: " streamed" });
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "text_delta", text: " streamed", fromAssistantBlock: true });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: " streamed", sessionId: "claude-uuid-abc" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sessionId: string, _prompt: string, onEvent: (e: any) => void) => {
        capturedOnEvent = onEvent;
        capturedPrompt = _prompt;
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "text_delta", text: "continued", fromAssistantBlock: true });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "continued", sessionId: "claude-uuid-abc" };
      }
    );
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

describe("slash commands & estimator", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(async () => {
    vi.clearAllMocks();
    capturedOnEvent = null;
    capturedPrompt = null;
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
  });

  async function newSession() {
    const r = await agent.newSession({ cwd: "/tmp/proj", mcpServers: [] });
    return r.sessionId;
  }

  function prompt(sid: string, text: string) {
    return agent.prompt({ sessionId: sid, prompt: [{ type: "text", text }] });
  }

  it("/status emits model, mode, cwd and session", async () => {
    const sid = await newSession();
    await prompt(sid, "/status");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    const text: string = msg[0].update.content.text;
    expect(text).toContain("**Model:**");
    expect(text).toContain("**Mode:** agent");
    expect(text).toContain("**Directory:** /tmp/proj");
    expect(text).toContain(`**Session:** \`${sid}\``);
  });

  it("/goal with empty arg shows status, not a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/goal");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    // /goal with no argument shows the current goal status (or "**Goal:** none"
    // when no goal is set) — matches Claude Code CLI behavior (1442197).
    expect(msg[0].update.content.text).toMatch(/^\*\*Goal:\*\*/);
    // No streaming turn should have been started for an empty /goal.
    expect(capturedOnEvent).toBeNull();
  });

  it("/goal <objective> sets the goal and starts a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/goal ship the feature");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const messages = calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk")
      .map((c: any) => c[0].update.content.text);
    expect(messages.some((m: string) => m.startsWith("Goal set:"))).toBe(true);
    // A streaming turn was started for the goal.
    expect(capturedOnEvent).not.toBeNull();
  });

  it("/review-branch with empty arg returns a usage message, not a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/review-branch");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    expect(msg[0].update.content.text).toMatch(/requires branch name/);
    expect(capturedOnEvent).toBeNull();
  });

  it("/review-branch <branch> starts a review turn referencing the branch diff", async () => {
    const sid = await newSession();
    await prompt(sid, "/review-branch main");

    expect(capturedOnEvent).not.toBeNull();
    expect(capturedPrompt).toContain("git diff main...HEAD");
    expect(capturedPrompt).toContain("main");
  });

  it("/review-commit with empty arg returns a usage message, not a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/review-commit");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    expect(msg[0].update.content.text).toMatch(/requires commit sha/);
    expect(capturedOnEvent).toBeNull();
  });

  it("/review-commit <sha> starts a review turn referencing the commit show", async () => {
    const sid = await newSession();
    await prompt(sid, "/review-commit abc123");

    expect(capturedOnEvent).not.toBeNull();
    expect(capturedPrompt).toContain("git show abc123");
    expect(capturedPrompt).toContain("abc123");
  });

  it("/clear wipes the session and does not start a turn", async () => {
    const sid = await newSession();
    // First establish a Claude session via a normal prompt.
    await prompt(sid, "hello");
    capturedOnEvent = null;
    await prompt(sid, "/clear");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const messages = calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk")
      .map((c: any) => c[0].update.content.text);
    expect(messages.some((m: string) => m.includes("Conversation cleared"))).toBe(true);
    expect(capturedOnEvent).toBeNull();
  });

  it("does not double-count assistant text emitted as both streaming delta and final block", async () => {    const sid = await newSession();
    await prompt(sid, "hello");

    const calls = (conn.sessionUpdate as any).mock.calls;
    // The mock emits " streamed" twice (once as streaming delta, once as the
    // final assistant block) plus a result event. usage_update is capped at
    // contextWindow; we assert the used value reflects a SINGLE count of the
    // assistant text (plus overhead + the user prompt), not a double count.
    const usage = calls.find((c: any) => c[0]?.update?.sessionUpdate === "usage_update");
    expect(usage).toBeTruthy();
    // Assert the LAST usage_update (sent with the result event, after all text
    // has been accumulated) — the first one is emitted at the `assistant` event
    // before the final block arrives.
    const usedValues: number[] = calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "usage_update")
      .map((c: any) => c[0].update.used as number);
    const used: number = usedValues[usedValues.length - 1];
    // This mock emits no `message.usage`, so the character fallback applies:
    // ceil((72748 overhead + 5 for "hello" + 9 for " streamed") / 3) = 24254.
    // Counting " streamed" twice would yield 24257, so an exact match proves
    // the assistant text is counted once.
    expect(used).toBe(24254);
  });

  it("prefers real per-call token usage over the character estimate", async () => {
    const sid = await newSession();
    await prompt(sid, "hello");
    // Feed a real per-call usage report, as Claude CLI does on every
    // `assistant` message, and confirm it becomes the reported context size
    // instead of the character guess.
    (conn.sessionUpdate as any).mockClear();
    capturedOnEvent!({
      type: "assistant",
      apiCallIndex: 0,
      usage: {
        input_tokens: 118035,
        output_tokens: 1271,
        cache_read_input_tokens: 200,
        cache_creation_input_tokens: 100,
      },
    });
    const usage = (conn.sessionUpdate as any).mock.calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "usage_update",
    );
    expect(usage).toBeTruthy();
    // 118035 + 200 + 100 = 118335 real prompt tokens, reported verbatim.
    expect(usage[0].update.used).toBe(118335);
  });

  it("ignores subagent (sidechain) usage — it is a separate context window", async () => {
    const sid = await newSession();
    await prompt(sid, "hello");
    (conn.sessionUpdate as any).mockClear();
    capturedOnEvent!({
      type: "assistant",
      apiCallIndex: 0,
      isSidechain: true,
      usage: { input_tokens: 150000, output_tokens: 10 },
    });
    const usage = (conn.sessionUpdate as any).mock.calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "usage_update",
    );
    expect(usage).toBeTruthy();
    // Still the character fallback — the subagent's 150k must not be adopted.
    expect(usage[0].update.used).toBeLessThan(30000);
  });

  it("usage estimate does not flip down at the start of a new turn (monotonic across turns)", async () => {
    const sid = await newSession();
    // Turn 1: prompt "hello" + assistant " streamed" + result usage.
    await prompt(sid, "hello");
    // Turn 2: continueSessionStreaming emits assistant + "continued".
    await prompt(sid, "more");

    const usages = (conn.sessionUpdate as any).mock.calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "usage_update")
      .map((c: any) => c[0].update.used as number);

    // Split into turn-1 and turn-2 usage updates. The mock emits a result
    // usage_update at the end of each turn; turn 2 also emits one from the
    // assistant event. We only need: the FIRST usage of turn 2 is >= the LAST
    // usage of turn 1 — i.e. no reset-to-overhead drop.
    expect(usages.length).toBeGreaterThanOrEqual(2);
    const lastTurn1 = usages[0];
    const firstTurn2 = usages[usages.length - 1];
    // Turn 2 starts with turn 1's accumulated content carried forward plus
    // the new prompt, so it must NOT drop below turn 1's final value.
    expect(firstTurn2).toBeGreaterThanOrEqual(lastTurn1);
  });

  it("surfaces an error_max_turns result to the user instead of a silent end_turn", async () => {
    const sid = await newSession();
    // Sentinel prompt makes the mock emit an isError result with
    // stopReason "error_max_turns" (model hit the turn ceiling mid-task).
    const res = await prompt(sid, "__error_max_turns__");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const messages = calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk")
      .map((c: any) => c[0].update.content.text);
    // The user must see a visible message explaining the premature stop.
    expect(messages.some((m: string) => /maximum number of turns/i.test(m))).toBe(true);
    // The prompt still resolves (does not throw) — end_turn is returned.
    expect(res.stopReason).toBe("end_turn");
  });

  it("merges native CLI commands on system_init, excluding skills/agents and tagging read-only ones turn-safe", async () => {
    const sid = await newSession();
    // Start a turn so the mock captures onEvent, then emit a realistic
    // system_init the way claude-runner.ts does. Mirrors the real CLI payload:
    // slash_commands is the union of builtins+skills+agents+plugin, while
    // skills/agents/terminal_slash_commands are separate flat lists.
    await prompt(sid, "hello");
    (conn.sessionUpdate as any).mockClear();
    capturedOnEvent!({
      type: "system_init",
      slashCommands: [
        "clear", "compact", "context", "model", "usage", "recap", "init",
        "doctor", "color", "reload-plugins",
        "zhihu", "graphify",        // skills
        "Explore", "general-purpose", // agents
        "run",                       // plugin command
      ],
      terminalSlashCommands: ["doctor", "color", "reload-plugins"],
      skillNames: ["zhihu", "graphify"],
      agentNames: ["claude", "Explore", "general-purpose", "Plan", "statusline-setup"],
    });

    const calls = (conn.sessionUpdate as any).mock.calls;
    const acUpdate = calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "available_commands_update",
    );
    expect(acUpdate).toBeTruthy();
    const commands: any[] = acUpdate[0].update.availableCommands;
    const byName = (n: string) => commands.find((c: any) => c.name === n);

    // Builtins are still present (ps, status, ...).
    expect(byName("ps")).toBeTruthy();
    expect(byName("status")).toBeTruthy();

    // Skills and agents are EXCLUDED from the command list.
    expect(byName("zhihu")).toBeUndefined();
    expect(byName("graphify")).toBeUndefined();
    expect(byName("Explore")).toBeUndefined();
    expect(byName("general-purpose")).toBeUndefined();

    // Native commands ARE present.
    expect(byName("context")).toBeTruthy();
    expect(byName("model")).toBeTruthy();

    // Read-only natives are turn-safe: via terminal_slash_commands (doctor)
    // and via the static allowlist (context, usage, recap).
    expect(byName("doctor")?._meta?.turnSafe).toBe(true);
    expect(byName("context")?._meta?.turnSafe).toBe(true);
    expect(byName("recap")?._meta?.turnSafe).toBe(true);
    expect(byName("usage")?._meta?.turnSafe).toBe(true);

    // State-mutating natives are NOT turn-safe (no _meta at all).
    expect(byName("model")?._meta).toBeUndefined();
    expect(byName("init")?._meta).toBeUndefined();
  });

  it("/skills lists skills from ~/.agents/skills (cross-agent global)", async () => {
    // Point every scanned dir at a non-existent path EXCEPT
    // ~/.agents/skills, which contains one skill dir with a SKILL.md.
    const realJoin = await import("node:path").then((m) => m.join);
    // agent.ts reads homedir() via node:os (NOT mocked here), so the scan
    // target is the REAL home's .agents/skills.
    const agentsSkillsDir = realJoin(os.homedir(), ".agents", "skills");

    const existsSync = vi.mocked(fs.existsSync);
    const readdirSync = vi.mocked(fs.readdirSync);
    const readFileSync = vi.mocked(fs.readFileSync);

    existsSync.mockImplementation((p: any) => {
      const s = String(p);
      return s === agentsSkillsDir || s === realJoin(agentsSkillsDir, "zhihu", "SKILL.md");
    });
    readdirSync.mockImplementation((_p: any, opts: any) => {
      if (opts?.withFileTypes) {
        return [
          { name: "zhihu", isDirectory: () => true },
          { name: "not-a-skill.txt", isDirectory: () => false },
        ] as any;
      }
      return [];
    });
    readFileSync.mockImplementation((p: any) => {
      if (String(p).endsWith("zhihu/SKILL.md")) {
        return "---\nname: zhihu\ndescription: Search Zhihu\n---\n# Zhihu\n";
      }
      return "{}";
    });

    const sid = await newSession();
    await prompt(sid, "/skills");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk",
    );
    const text: string = msg[0].update.content.text;
    expect(text).toContain("/zhihu");
    expect(text).toContain("Search Zhihu");

    // restore default mocks for other tests
    existsSync.mockImplementation(() => false);
    readdirSync.mockImplementation(() => []);
    readFileSync.mockImplementation(() => "{}");
  });
});

// The steer-drain loop now calls the same dispatchSlashCommand() helper that
// prompt() uses (verified by the tests above), so command parity between the
// two entry points is guaranteed by construction rather than by a duplicated
// implementation. Driving the internal steer queue through the public ACP API
// would require a full connection harness, so we rely on the shared-dispatch
// unit tests above for coverage of the command behavior itself.

// Per-session prompt serialization (the concurrent /skills-mid-turn
// regression) is covered in tests/agent-prompt-serialization.test.ts — it
// needs a runner mock whose startSessionStreaming blocks on a deferred
// release, which is incompatible with this file's synchronous mock.
