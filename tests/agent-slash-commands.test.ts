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
// When true, the FIRST startSessionStreaming/continueSessionStreaming call
// emits a Task (subagent) tool_use with no <task-notification> and ends the
// turn — the premature-end_turn case (subagent still pending at end_turn).
// Subsequent calls (the auto-follow-up turn, the post-turn steer drain) emit a
// normal result. Lets a test drive the auto-follow-up + steer-drain ordering.
// steerToInject is fired synchronously from within onEvent (guaranteeing a
// prompt is in flight) so the test doesn't depend on a 0ms timer race.
let emitPendingSubagentOnce = false;
let steerToInject: { text: string } | null = null;
// Set by the test so the fixture's queueMicrotask steer can reach the real
// agent's extNotification with the right session id.
let capturedAgentRef: any = null;
let capturedSessionRef: string | null = null;

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
        // Premature-end_turn fixture: emit a subagent (Task) tool_use with no
        // <task-notification>, so the turn ends with the subagent still
        // pending. Consumed once, then the auto-follow-up / steer-drain turns
        // below take the normal path. If steerToInject is set, fire it
        // synchronously from within onEvent — guaranteeing a prompt is in
        // flight when handleSteer runs (no 0ms timer race).
        if (emitPendingSubagentOnce) {
          emitPendingSubagentOnce = false;
          onEvent({ type: "tool_use", toolName: "Task", toolInput: { description: "investigate", subagent_type: "general-purpose" }, toolUseId: "tu_subagent" });
          onEvent({ type: "assistant", apiCallIndex: 0 });
          if (steerToInject) {
            const s = steerToInject;
            steerToInject = null;
            // Fire the steer on the NEXT microtask so the tool_use handler has
            // registered the pending subagent, but still during this prompt
            // (which is in flight).
            queueMicrotask(() => {
              (capturedAgentRef as any)?.extNotification("session/steer", { sessionId: capturedSessionRef, text: s.text });
            });
          }
          onEvent({ type: "result", stopReason: "end_turn", usage: { input_tokens: 10, output_tokens: 5 } });
          return { text: "", sessionId: "claude-uuid-abc" };
        }
        // Sentinel prompt to simulate a Claude CLI error result (e.g. the
        // model hit the turn ceiling). Used by the error-surfacing test.
        // The adapter prepends a <collaboration_mode> block to every default-mode
        // prompt, so match on the sentinel being a suffix, not an exact match.
        if (_prompt === "__error_max_turns__" || _prompt.endsWith("__error_max_turns__")) {
          onEvent({
            type: "result",
            usage: { input_tokens: 10, output_tokens: 5 },
            stopReason: "error_max_turns",
            isError: true,
          });
          return { text: "", sessionId: "claude-uuid-abc" };
        }
        // Sentinel prompt to simulate a Claude CLI `local` command result
        // (/heapdump, /context): the CLI services it itself with no LLM turn,
        // returning the output as the `result` of a single `result` event with
        // no stop reason and no usage (see processSlashCommand case 'local' →
        // { shouldQuery:false, resultText }). The adapter must surface that
        // text as a commandOutput row instead of dropping it.
        // (Suffix match: the adapter prepends the <collaboration_mode> block.)
        if (_prompt === "/heapdump" || _prompt.endsWith("/heapdump") || _prompt === "/context" || _prompt.endsWith("/context")) {
          onEvent({ type: "result", text: "Heap snapshot written to: /tmp/x.heapsnapshot", stopReason: null, usage: undefined });
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
        // Premature-end_turn fixture (mirrors startSessionStreaming): emit a
        // subagent tool_use with no <task-notification> so the turn ends with
        // the subagent still pending. Consumed once.
        if (emitPendingSubagentOnce) {
          emitPendingSubagentOnce = false;
          onEvent({ type: "tool_use", toolName: "Task", toolInput: { description: "investigate", subagent_type: "general-purpose" }, toolUseId: "tu_subagent" });
          onEvent({ type: "assistant", apiCallIndex: 0 });
          onEvent({ type: "result", stopReason: "end_turn", usage: { input_tokens: 10, output_tokens: 5 } });
          return { text: "", sessionId: "claude-uuid-abc" };
        }
        // Sentinel prompt to simulate a Claude CLI `local` command result
        // (/heapdump, /context): the CLI services it itself with no LLM turn,
        // returning the output as the `result` of a single `result` event with
        // no stop reason and no usage (see processSlashCommand case 'local' →
        // { shouldQuery:false, resultText }). The adapter must surface that
        // text as a commandOutput row instead of dropping it.
        // (Suffix match: the adapter prepends the <collaboration_mode> block.)
        if (_prompt === "/heapdump" || _prompt.endsWith("/heapdump") || _prompt === "/context" || _prompt.endsWith("/context")) {
          onEvent({ type: "result", text: "Heap snapshot written to: /tmp/x.heapsnapshot", stopReason: null, usage: undefined });
          return { text: "", sessionId: "claude-uuid-abc" };
        }
        onEvent({ type: "assistant", apiCallIndex: 0 });
        onEvent({ type: "text_delta", text: "continued", fromAssistantBlock: true });
        onEvent({ type: "result", usage: { input_tokens: 10, output_tokens: 5 } });
        return { text: "continued", sessionId: "claude-uuid-abc" };
      }
    );
    this.cancel = vi.fn();
    this.setSessionPermissionMode = vi.fn();
    // handleSteer checks whether a turn is active; report idle (the agent's
    // own sessionPromptInFlight set is the authoritative in-flight signal for
    // the steer-queueing decision, so a false here is fine for the test).
    this.isRunning = vi.fn().mockReturnValue(false);
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
    // Custom extension methods. The agent's real extNotification handles
    // session/steer; these mocks exist only so the AgentSideConnection shape
    // is complete for the tests that don't drive them.
    extMethod: vi.fn().mockResolvedValue({}),
    extNotification: vi.fn().mockResolvedValue(undefined),
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
    // Plain text, not markdown — command-output rows render via textContent,
    // so markdown source would show up verbatim.
    expect(text).toContain("Model:");
    expect(text).toContain("Mode: agent");
    expect(text).toContain("Directory: /tmp/proj");
    expect(text).toContain(`Session: ${sid}`);
  });

  it("a CLI-native local command (/heapdump) result is surfaced as a commandOutput row", async () => {
    // /heapdump is a Claude CLI-native `local` command the CLI services itself
    // with no LLM turn. It is NOT in isReadOnlyCommand, so it is queued and
    // drained like a normal prompt (not fired mid-turn as a side-channel) and
    // forwarded to the CLI. The CLI returns its output as the `result` field
    // of a single `result` event (no stop reason, no usage). The mock runner
    // is wired to emit exactly that for the /heapdump and /context prompts.
    // The adapter must surface the text as a transient commandOutput SNAPSHOT
    // row — the regression is that the result handler used to drop
    // `event.text`, so /heapdump and /context showed nothing in the ACP chat.
    const sid = await newSession();
    // First establish a Claude session via a normal prompt, so the
    // /heapdump turn resumes that session (continueSessionStreaming path).
    await prompt(sid, "hello");
    (conn.sessionUpdate as any).mockClear();
    const res = await prompt(sid, "/heapdump");

    // The turn resolves as a clean end_turn (a native command, not a failure).
    expect(res.stopReason).toBe("end_turn");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const cmdMsg = calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk"
        && c[0]?.update?._meta?.commandOutput === true,
    );
    expect(cmdMsg).toBeTruthy();
    expect(cmdMsg[0].update.content.text).toContain("Heap snapshot written to:");

    // No assistant bubble / LLM text should have been emitted for the command
    // — the only agent_message_chunk is the commandOutput snapshot row.
    const allMsgTexts = calls
      .filter((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk")
      .map((c: any) => c[0].update.content.text);
    expect(allMsgTexts.length).toBe(1);
  });

  it("/goal with empty arg shows status, not a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/goal");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    // /goal with no argument shows the current goal status (or "Goal: none"
    // when no goal is set) — matches Claude Code CLI behavior (1442197).
    // Plain text, not markdown: command-output rows render via textContent.
    expect(msg[0].update.content.text).toMatch(/^Goal:/);
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
    // This mock emits no `message.usage`, so the character fallback applies.
    // The adapter prepends the <collaboration_mode> Default-mode block (182
    // chars) to every default-mode prompt, so the counted prompt is
    // block + "hello" + " streamed":
    // ceil((72748 overhead + 182 block + 5 for "hello" + 9 for " streamed") / 3) = 24436.
    // Counting " streamed" twice would yield 24437, so an exact match proves
    // the assistant text is counted once.
    expect(used).toBe(24436);
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

    // Read-only natives are turn-safe via the static allowlist
    // (context, usage, recap).
    expect(byName("context")?._meta?.turnSafe).toBe(true);
    expect(byName("recap")?._meta?.turnSafe).toBe(true);
    expect(byName("usage")?._meta?.turnSafe).toBe(true);

    // State-mutating or cosmetic natives are NOT turn-safe (no _meta at all).
    // doctor/color/reload-plugins are in the CLI's terminal_slash_commands but
    // are not read-only introspection useful mid-turn, so they stay unsafe.
    expect(byName("model")?._meta).toBeUndefined();
    expect(byName("init")?._meta).toBeUndefined();
    expect(byName("doctor")?._meta).toBeUndefined();
    expect(byName("color")?._meta).toBeUndefined();
    expect(byName("reload-plugins")?._meta).toBeUndefined();
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

  it("/list-agents shows this session and other local Claude sessions", async () => {
    const realJoin = await import("node:path").then((m) => m.join);
    const sessionsDir = realJoin(os.homedir(), ".claude", "sessions");

    const selfEntry = JSON.stringify({
      pid: process.pid,
      sessionId: "01a00000-0000-0000-0000-000000000001",
      name: "test-agent",
      cwd: "/tmp/proj",
      startedAt: Date.now() - 60_000,
      status: "busy",
    });
    const otherEntry = JSON.stringify({
      pid: 99999,
      sessionId: "01b00000-0000-0000-0000-000000000002",
      name: "other-agent",
      cwd: "/workspace/other-repo",
      startedAt: Date.now() - 3_600_000,
      status: "idle",
    });

    const readdirSync = vi.mocked(fs.readdirSync);
    const readFileSync = vi.mocked(fs.readFileSync);

    readdirSync.mockImplementation((p: any, _opts: any) => {
      if (String(p) === sessionsDir) {
        return [
          { name: "123.json", isFile: () => true },
          { name: "456.json", isFile: () => true },
        ] as any;
      }
      return [] as any;
    });
    readFileSync.mockImplementation((p: any) => {
      const s = String(p);
      if (s.endsWith("123.json")) return selfEntry;
      if (s.endsWith("456.json")) return otherEntry;
      return "{}";
    });

    const sid = await newSession();
    await prompt(sid, "/list-agents");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find(
      (c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk",
    );
    const text: string = msg[0].update.content.text;
    expect(text).toContain("This session: test-agent");
    expect(text).toContain("Other Claude sessions (1)");
    expect(text).toContain("[idle]");
    expect(text).toContain("other-agent");
    expect(text).toContain("/workspace/other-repo");
    // commandOutput marker must be set for transient snapshot rendering
    expect(msg[0].update._meta?.commandOutput).toBe(true);

    // restore default mocks for other tests
    readdirSync.mockImplementation(() => []);
    readFileSync.mockImplementation(() => "{}");
  });

  it("a mid-turn user steer is NOT consumed by the premature-end_turn auto-follow-up", async () => {
    // Regression for the "deploy a2a" collision: a turn ends with a subagent
    // still pending (premature end_turn). The adapter enqueues a synthetic
    // "[Auto] …incorporate results…" follow-up. Previously it called
    // drainSteerQueue() which drained the ENTIRE steer queue — folding any
    // user steer typed mid-turn (e.g. "deploy a2a") into that synthetic turn
    // instead of letting it run as its own clean turn.
    //
    // Fixture: the first prompt emits a Task (subagent) tool_use with no
    // <task-notification> and ends the turn (subagent pending). During the
    // turn we fire a steer (session/steer) that handleSteer queues because a
    // prompt is in flight. The auto-follow-up must deliver ONLY its own
    // synthetic message as a turn; the user's steer must be delivered as a
    // SEPARATE turn by the post-turn drain.
    // Use a near-zero subagent wait (the production default is 15s, which
    // would blow the 5s vitest timeout during the bounded end_turn wait).
    agent = createClaudeCodeAgent(conn, { pendingAgentWaitMs: 1 });
    const sid = await newSession();
    capturedAgentRef = agent;
    capturedSessionRef = sid;
    emitPendingSubagentOnce = true;
    // The fixture fires this steer from within onEvent (mid-turn), so a prompt
    // is in flight when handleSteer queues it — no 0ms timer race.
    steerToInject = { text: "deploy a2a" };
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "research" }] });

    // Let the queueMicrotask'd steer + the post-turn drain settle.
    await new Promise((r) => setTimeout(r, 0));

    // The prompts forwarded to the runner (2nd arg of the streaming calls),
    // in order. The auto-follow-up delivers its synthetic message itself; the
    // post-turn drain delivers the user steer. Both must appear, and the user
    // steer must be its OWN turn (framed as a follow-up), NOT merged into the
    // auto-follow-up's synthetic text.
    const forwarded = ((conn as any).__forwardedPrompts ?? []);
    // Fallback: capture via the module-level capturedPrompt is not per-call,
    // so instead assert on the sessionUpdate user_message_chunk echoes, which
    // the drain + follow-up each emit for the message they send.
    const userChunks = ((conn as any).sessionUpdate as any).mock.calls
      .map((c: any) => c[0]?.update)
      .filter((u: any) => u?.sessionUpdate === "user_message_chunk")
      .map((u: any) => u.content.text);

    // The auto-follow-up synthetic message was delivered as its own turn.
    expect(userChunks.some((t: string) => t.startsWith("[Auto] 1 background subagent(s) finished"))).toBe(true);
    // The user steer was delivered as its OWN turn, framed as a follow-up —
    // i.e. it was NOT absorbed into the synthetic follow-up's prompt.
    expect(userChunks.some((t: string) => t.includes("deploy a2a"))).toBe(true);

    // And crucially: the user steer's forwarded prompt is the framed
    // "[Follow-up instruction from user] …" text, distinct from the synthetic
    // follow-up — proving two separate turns ran.
    expect(userChunks.filter((t: string) => t.includes("deploy a2a")).length).toBe(1);
    void forwarded;
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
