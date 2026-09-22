import { describe, it, expect, vi } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";

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
let running = false;
/** Fires once the given call index begins, so a test can steer mid-turn. */
let onStreamCall: ((call: RecordedCall) => void | Promise<void>) | null = null;

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    const record = async (kind: "start" | "continue", prompt: string) => {
      const call: RecordedCall = { kind, prompt };
      recorded.push(call);
      if (onStreamCall) await onStreamCall(call);
      return { text: "streamed", sessionId: "claude-uuid-abc" };
    };
    this.startSessionStreaming = vi
      .fn()
      .mockImplementation(async (_cwd: string, prompt: string) => record("start", prompt));
    this.continueSessionStreaming = vi
      .fn()
      .mockImplementation(async (_sid: string, prompt: string) => record("continue", prompt));
    this.startSession = vi.fn().mockResolvedValue({ text: "", sessionId: "claude-uuid-abc" });
    this.continueSession = vi.fn().mockResolvedValue({ text: "", sessionId: "claude-uuid-abc" });
    this.isRunning = vi.fn(() => running);
    this.cancel = vi.fn();
    this.cancelAll = vi.fn();
  });
  return { ClaudeRunner: MockClaudeRunner };
});

function createMockConnection(): any {
  return { sessionUpdate: vi.fn().mockResolvedValue(undefined) };
}

async function setup() {
  vi.clearAllMocks();
  recorded = [];
  running = false;
  onStreamCall = null;
  const conn = createMockConnection();
  const agent: any = createClaudeCodeAgent(conn);
  await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  return { conn, agent, sid: sessionId };
}

function prompt(agent: any, sid: string, text: string) {
  return agent.prompt({ sessionId: sid, prompt: [{ type: "text", text }] });
}

async function steer(agent: any, sid: string, text: string) {
  running = true;
  await agent.extNotification("session/steer", { sessionId: sid, text });
  running = false;
}

/** Prompts sent to the CLI, in order. */
function prompts() {
  return recorded.map((r) => r.prompt);
}

describe("steer queue drain ordering", () => {
  it("drains a steer queued before the next prompt BEFORE that prompt's turn", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "first"); // establish a Claude session

    // A steer accepted while the previous turn was winding down but NOT
    // delivered by its drain (e.g. it landed in the drain tail). It must go out
    // before the newer prompt, matching the order the user typed.
    await steer(agent, sid, "EARLIER-STEER");

    recorded = [];
    await prompt(agent, sid, "second");

    const idxSteer = prompts().findIndex((p) => p.includes("EARLIER-STEER"));
    const idxSecond = prompts().findIndex((p) => p.includes("second"));
    expect(idxSteer).toBeGreaterThanOrEqual(0);
    expect(idxSecond).toBeGreaterThanOrEqual(0);
    expect(idxSteer).toBeLessThan(idxSecond);
  });

  it("picks up a steer that arrives mid-drain in the same pass", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello");

    await steer(agent, sid, "STEER-A");
    // While STEER-A is being delivered, STEER-B arrives. The live-queue drain
    // must pick it up in the same pass instead of stranding it.
    onStreamCall = async (call) => {
      if (call.prompt.includes("STEER-A")) {
        onStreamCall = null;
        await steer(agent, sid, "STEER-B");
      }
    };
    recorded = [];
    await prompt(agent, sid, "go");
    expect(prompts().some((p) => p.includes("STEER-B"))).toBe(true);
  });

  it("delivers a steer queued during /compact after compaction completes", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish a Claude session

    recorded = [];
    // Steer while /compact is running its summary turn.
    onStreamCall = async (call) => {
      if (call.prompt.includes("concise summary")) {
        onStreamCall = null;
        await steer(agent, sid, "DURING-COMPACT");
      }
    };
    await prompt(agent, sid, "/compact");

    const all = prompts();
    const idxFresh = all.findIndex((p) => p.includes("Previous conversation summary"));
    const idxSteer = all.findIndex((p) => p.includes("DURING-COMPACT"));
    expect(idxFresh).toBeGreaterThanOrEqual(0);
    // Queued during compaction, delivered after it — not dropped, not before.
    expect(idxSteer).toBeGreaterThan(idxFresh);
  });

  it("keeps undelivered steers queued when the drain stops early", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello");

    // Two steers; /clear is dispatched first and ends the pass, so the second
    // must remain queued rather than being discarded with the taken batch.
    running = true;
    await agent.extNotification("session/steer", { sessionId: sid, text: "/clear" });
    await agent.extNotification("session/steer", { sessionId: sid, text: "SURVIVOR" });
    running = false;

    recorded = [];
    await prompt(agent, sid, "next");
    // SURVIVOR is delivered rather than lost (it may land after the new turn,
    // since /clear left no session to resume into at pre-turn drain time).
    expect(prompts().some((p) => p.includes("SURVIVOR"))).toBe(true);
  });

  it("discards queued steers when the session is cancelled", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello");

    running = true;
    await agent.extNotification("session/steer", { sessionId: sid, text: "DROP-ME" });
    running = false;
    await agent.cancel({ sessionId: sid });

    recorded = [];
    await prompt(agent, sid, "next");
    expect(prompts().some((p) => p.includes("DROP-ME"))).toBe(false);
  });

  it("ignores steers for an idle session (no active turn)", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello");

    running = false; // no turn in flight, no prompt in flight
    await agent.extNotification("session/steer", { sessionId: sid, text: "IGNORED" });

    recorded = [];
    await prompt(agent, sid, "next");
    expect(prompts().some((p) => p.includes("IGNORED"))).toBe(false);
  });

  it("emits a user_message_accepted (stage-1) on receipt, distinct from the delivered chunk", async () => {
    const { conn, agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish a Claude session

    // Steer while a turn is active: the agent must ack immediately with a
    // user_message_accepted so the client shows the mid-turn bubble, but it must
    // NOT emit a user_message_chunk on receipt (that would be treated as
    // delivered and prematurely release the next queued message).
    const updatesBefore: any[] = [];
    const spy = conn.sessionUpdate as any;
    spy.mockImplementation((u: any) => {
      updatesBefore.push(u.update);
      return Promise.resolve(undefined);
    });
    running = true;
    await agent.extNotification("session/steer", { sessionId: sid, text: "MID-TURN-MSG" });
    running = false;

    const accepted = updatesBefore.filter(
      (u) => u?.sessionUpdate === "user_message_accepted" &&
             u?.content?.text === "MID-TURN-MSG",
    );
    const chunksOnReceipt = updatesBefore.filter(
      (u) => u?.sessionUpdate === "user_message_chunk" &&
             u?.content?.text === "MID-TURN-MSG",
    );
    expect(accepted.length).toBe(1); // stage-1 ack fired on receipt
    expect(chunksOnReceipt.length).toBe(0); // no premature delivered-echo
  });
});
