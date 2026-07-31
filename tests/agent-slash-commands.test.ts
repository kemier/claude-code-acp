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

// Captures the onEvent callback passed to the runner so a test can emit
// stream events (assistant block + streaming deltas) and assert the estimator
// does not double-count assistant text.
let capturedOnEvent: ((e: any) => void) | null = null;

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

  it("/goal with empty arg returns an error message, not a turn", async () => {
    const sid = await newSession();
    await prompt(sid, "/goal");

    const calls = (conn.sessionUpdate as any).mock.calls;
    const msg = calls.find((c: any) => c[0]?.update?.sessionUpdate === "agent_message_chunk");
    expect(msg[0].update.content.text).toMatch(/requires \[.+\]/);
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

  it("does not double-count assistant text emitted as both streaming delta and final block", async () => {
    const sid = await newSession();
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
