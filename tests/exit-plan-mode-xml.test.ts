import { describe, it, expect, vi } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

/**
 * Proxy models write ExitPlanMode as chat XML because headless `claude -p`
 * has no approval dialog and plan mode disallows the tool. The adapter must
 * hide the tags, show the plan, and leave plan mode only after the user
 * approves — the same plan→default steer the collaboration picker queues.
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

const PLAN = "Implement all feasible deep-agent capability gaps.";
const XML = `<tool_call>
<function=ExitPlanMode>
<parameter=plan>
${PLAN}
</parameter>
</function>
</tool_call>`;

let recorded: string[] = [];
let permissionMode: string | null = null;
/** When set, the next streaming turn emits this instead of a plain result. */
let emit: ((onEvent: (e: StreamEvent) => void) => void) | null = null;

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    const run = async (_prompt: string, onEvent: (e: StreamEvent) => void) => {
      recorded.push(_prompt);
      if (emit) {
        const fire = emit;
        emit = null;
        fire(onEvent);
      }
      onEvent({ type: "result", text: "done", sessionId: "uuid-exit-plan" } as StreamEvent);
      return { text: "done", sessionId: "uuid-exit-plan" };
    };
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, prompt: string, onEvent: (e: StreamEvent) => void) => run(prompt, onEvent),
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sid: string, prompt: string, onEvent: (e: StreamEvent) => void) => run(prompt, onEvent),
    );
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, content: any, onEvent: any) =>
        this.startSessionStreaming(_cwd, content?.[0]?.text ?? "", onEvent),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, content: any, onEvent: any) =>
        this.continueSessionStreaming(_sid, content?.[0]?.text ?? "", onEvent),
    );
    this.cancel = vi.fn();
    this.isRunning = vi.fn(() => false);
    this.setSessionPermissionMode = vi.fn((mode: string) => {
      permissionMode = mode;
    });
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
  });
  return { ClaudeRunner: MockClaudeRunner, isProxyBaseUrl: () => false };
});

function createMockConnection(permissionResponse: any): AgentSideConnection {
  return {
    sessionUpdate: vi.fn().mockResolvedValue(undefined),
    requestPermission: vi.fn().mockResolvedValue(permissionResponse),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    createTerminal: vi.fn(),
    signal: new AbortController().signal,
    closed: new Promise(() => {}),
  } as unknown as AgentSideConnection;
}

function assistantText(conn: AgentSideConnection): string {
  return (conn.sessionUpdate as any).mock.calls
    .map((c: any) => c[0].update)
    .filter((u: any) => u.sessionUpdate === "agent_message_chunk" && u.content?.type === "text")
    .map((u: any) => u.content.text as string)
    .join("");
}

async function setup(permissionResponse: any) {
  vi.clearAllMocks();
  recorded = [];
  permissionMode = null;
  emit = null;
  const conn = createMockConnection(permissionResponse);
  const agent: any = createClaudeCodeAgent(conn);
  await agent.initialize({ protocolVersion: 1, clientCapabilities: {} });
  const { sessionId } = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  await agent.setSessionConfigOption({
    sessionId,
    configId: "collaboration_mode",
    value: "plan",
  });
  return { conn, agent, sid: sessionId };
}

function allow() {
  return { outcome: { type: "selected", optionId: "allow_once" } };
}
function reject() {
  return { outcome: { type: "selected", optionId: "reject_once" } };
}

describe("leaked ExitPlanMode XML", () => {
  it("hides the tags, shows the plan, and implements after approval", async () => {
    const { conn, agent, sid } = await setup(allow());
    emit = (onEvent) => {
      onEvent({ type: "text_delta", text: "Ready.\n" });
      // Split the open tag and the plan body across deltas.
      const cut = XML.indexOf(PLAN);
      onEvent({ type: "text_delta", text: XML.slice(0, cut) });
      onEvent({ type: "text_delta", text: XML.slice(cut) });
    };
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "plan the work" }] });

    const shown = assistantText(conn);
    expect(shown).toContain("Ready.");
    expect(shown).toContain(PLAN);
    expect(shown).not.toContain("<tool_call>");
    expect(shown).not.toContain("<function=ExitPlanMode>");
    expect(shown).not.toContain("<parameter=plan>");

    expect(conn.requestPermission).toHaveBeenCalledTimes(1);
    const perm = (conn.requestPermission as any).mock.calls[0][0];
    expect(perm.toolCall.title).toBe("Exit plan mode");
    expect(perm.toolCall.kind).toBe("switch_mode");
    expect(perm.options.map((o: any) => o.optionId)).toEqual(["allow_once", "reject_once"]);

    const updates = (conn.sessionUpdate as any).mock.calls.map((c: any) => c[0].update);
    const exitTool = updates.find(
      (u: any) => u.sessionUpdate === "tool_call" && u.title === "Exit plan mode",
    );
    expect(exitTool?.status).toBe("pending");
    expect(
      updates.some(
        (u: any) =>
          u.sessionUpdate === "tool_call_update" &&
          u.toolCallId === exitTool.toolCallId &&
          u.status === "completed",
      ),
    ).toBe(true);
    expect(permissionMode).toBe("agent");

    await vi.waitFor(
      () => {
        expect(
          recorded.some(
            (p) => p.includes("Collaboration Mode: Default") && p.includes("implement it now"),
          ),
        ).toBe(true);
      },
      { timeout: 3000 },
    );
  });

  it("keeps plan mode when the user rejects", async () => {
    const { conn, agent, sid } = await setup(reject());
    emit = (onEvent) => {
      onEvent({ type: "text_delta", text: XML });
    };
    await agent.prompt({ sessionId: sid, prompt: [{ type: "text", text: "plan the work" }] });

    const shown = assistantText(conn);
    expect(shown).toContain(PLAN);
    expect(shown).not.toContain("<tool_call>");
    expect(permissionMode).toBe("read-only");

    const updates = (conn.sessionUpdate as any).mock.calls.map((c: any) => c[0].update);
    const exitTool = updates.find(
      (u: any) => u.sessionUpdate === "tool_call" && u.title === "Exit plan mode",
    );
    expect(
      updates.some(
        (u: any) =>
          u.sessionUpdate === "tool_call_update" &&
          u.toolCallId === exitTool.toolCallId &&
          u.status === "failed",
      ),
    ).toBe(true);

    await new Promise((r) => setTimeout(r, 30));
    expect(recorded.some((p) => p.includes("implement it now"))).toBe(false);
  });
});
