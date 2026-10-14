import { describe, it, expect, vi } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";

/**
 * Flipping the collaboration picker plan→default WHILE the model is streaming
 * its plan summary used to SIGINT the running CLI (interruptForPlanMode on
 * EITHER direction). The in-flight assistant message was killed mid-sentence —
 * the partial plan was all the client saw, the killed CLI never wrote it to
 * the resume transcript, and the re-prompt had no plan to continue from:
 * premature end_turn + cut-off plan.
 *
 * The fix: plan→default no longer interrupts (the running Plan spawn is
 * already read-only, so nothing must be revoked). It queues the
 * <collaboration_mode> notice on the steer queue; drainSteerQueue chains a
 * follow-up turn at end_turn with the new mode's flags, and the model
 * implements with the FULL plan in its transcript. default→plan keeps
 * interrupting (the running Default spawn carries edit access until
 * re-spawned).
 */

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
/** Fires once the given call begins, so a test can flip the picker mid-turn. */
let onStreamCall: ((call: RecordedCall) => void | Promise<void>) | null = null;
/** runner.cancel() call count — the plan→default fix must NOT SIGINT. */
let cancelCalls = 0;

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
    this.cancel = vi.fn(() => {
      cancelCalls += 1;
    });
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.setSessionPermissionMode = vi.fn();
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
  cancelCalls = 0;
  const conn = createMockConnection();
  const agent: any = createClaudeCodeAgent(conn);
  await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  return { conn, agent, sid: sessionId };
}

function prompt(agent: any, sid: string, text: string) {
  return agent.prompt({ sessionId: sid, prompt: [{ type: "text", text }] });
}

/** Prompts sent to the CLI, in order. */
function prompts() {
  return recorded.map((r) => r.prompt);
}

function flipCollaboration(agent: any, sid: string, value: string) {
  return agent.setSessionConfigOption({
    sessionId: sid,
    configId: "collaboration_mode",
    value,
  });
}

describe("collaboration picker flip mid-turn", () => {
  it("plan→default lets the running turn finish and chains the notice as a follow-up turn (no SIGINT)", async () => {
    const { agent, sid } = await setup();
    // Enter plan mode while idle — plain applyModeChange, no announcement.
    await flipCollaboration(agent, sid, "plan");
    await prompt(agent, sid, "hello"); // establish a Claude session

    // Start the plan turn; flip the picker to default WHILE it streams.
    // The second prompt in a session rides continueSessionStreaming (persistent
    // CLI), so match on the prompt text, not the call kind.
    onStreamCall = async (call) => {
      if (call.prompt.includes("write-the-plan")) {
        onStreamCall = null;
        await flipCollaboration(agent, sid, "default");
      }
    };
    recorded = [];
    await prompt(agent, sid, "write-the-plan");

    // The running turn was NOT SIGINT'd...
    expect(cancelCalls).toBe(0);
    // The post-turn drain may be backgrounded (pendingDrainPromises), so poll.
    await vi.waitFor(
      () => {
        const followUps = prompts().filter(
          (p) => p.includes("Collaboration Mode: Default") && p.includes("no longer active")
        );
        expect(followUps.length).toBeGreaterThanOrEqual(1);
        const mainIdx = prompts().findIndex((p) => p.includes("write-the-plan"));
        const flipIdx = prompts().findIndex((p) => p.includes("Collaboration Mode: Default"));
        expect(flipIdx).toBeGreaterThan(mainIdx);
      },
      { timeout: 3000 },
    );
  });

  it("default→plan still interrupts and re-prompts with the fresh Plan block", async () => {
    const { agent, sid } = await setup();
    await prompt(agent, sid, "hello"); // establish a Claude session (default mode)

    // Fire-and-forget: interruptForPlanMode chains the re-prompt on the prompt
    // lock, so awaiting it from INSIDE the streaming callback would deadlock.
    // runner.cancel() still runs synchronously before the chain awaits.
    onStreamCall = async (call) => {
      if (call.prompt.includes("edit-the-code")) {
        onStreamCall = null;
        flipCollaboration(agent, sid, "plan");
      }
    };
    recorded = [];
    await prompt(agent, sid, "edit-the-code");

    // The Default spawn carries edit access — it MUST be revoked via SIGINT...
    expect(cancelCalls).toBe(1);
    // The re-prompt chains on the prompt lock AFTER the cancelled turn settles,
    // so poll like steer-queue-drain does.
    await vi.waitFor(
      () => {
        const rePrompts = prompts().filter(
          (p) => p.includes("Collaboration Mode: Plan") && p.includes("no longer active")
        );
        expect(rePrompts.length).toBeGreaterThanOrEqual(1);
      },
      { timeout: 3000 },
    );
  });
});
