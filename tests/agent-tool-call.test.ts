import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import type { StreamEvent } from "../src/claude-runner.js";

// Mock node:fs so SessionStore doesn't read real persisted sessions
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

// Shared, mutable event script. Each test sets `scriptedEvents` before
// driving a prompt; the mock runner replays them.
let scriptedEvents: StreamEvent[] = [];

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (
        _cwd: string,
        _prompt: string,
        onEvent: (event: StreamEvent) => void
      ) => {
        for (const e of scriptedEvents) onEvent(e);
        return { text: "done", sessionId: "uuid-1" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "Continued",
      sessionId: "uuid-1",
    });
    this.cancel = vi.fn();
  });
  return { ClaudeRunner: MockClaudeRunner };
});

function createMockConnection() {
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

async function runWithEvents(events: StreamEvent[]) {
  scriptedEvents = events;
  const conn = createMockConnection();
  const agent = createClaudeCodeAgent(conn);
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: {},
    clientInfo: { name: "test", version: "1.0" },
  });
  const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  await agent.prompt({
    sessionId: session.sessionId,
    prompt: [{ type: "text", text: "go" }],
  });
  const calls = (conn.sessionUpdate as any).mock.calls;
  const updates = calls.map((c: any) => c[0].update);
  return { updates, conn };
}

describe("Agent tool_call notifications", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scriptedEvents = [];
  });

  it("Read tool → descriptive title + kind:read + locations", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Read",
        toolInput: { file_path: "/tmp/test.ts" },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].title).toBe("Read file 'test.ts'");
    expect(toolCalls[0].kind).toBe("read");
    expect(toolCalls[0].locations).toEqual([{ path: "/tmp/test.ts" }]);
  });

  it("Edit tool → kind:edit + diff content (old/new) + locations", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Edit",
        toolInput: {
          file_path: "/tmp/foo.ts",
          old_string: "a",
          new_string: "b",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].title).toBe("Edit file 'foo.ts'");
    expect(toolCalls[0].kind).toBe("edit");
    expect(toolCalls[0].content).toEqual([
      {
        type: "diff",
        path: "/tmp/foo.ts",
        oldText: "a",
        newText: "b",
        _meta: { kind: "update" },
      },
    ]);
    expect(toolCalls[0].locations).toEqual([{ path: "/tmp/foo.ts" }]);
  });

  it("Write tool → kind:edit + diff content (oldText null) + _meta kind:add", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Write",
        toolInput: { file_path: "/tmp/new.ts", content: "hello" },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("Write file 'new.ts'");
    expect(toolCalls[0].kind).toBe("edit");
    expect(toolCalls[0].content[0].oldText).toBeNull();
    expect(toolCalls[0].content[0].newText).toBe("hello");
    expect(toolCalls[0].content[0]._meta).toEqual({ kind: "add" });
  });

  it("Bash tool → kind:execute + command title", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: { command: "npm test" },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("npm test");
    expect(toolCalls[0].kind).toBe("execute");
  });

  it("Grep tool → kind:search + 'Search for' title", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Grep",
        toolInput: { pattern: "TODO", path: "/tmp/src" },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].kind).toBe("search");
    expect(toolCalls[0].title).toContain("Search for 'TODO'");
  });

  it("unknown tool → falls back to tool name + kind:execute", async () => {
    const { updates } = await runWithEvents([
      { type: "tool_use", toolName: "CustomTool", toolInput: { x: 1 } },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("CustomTool");
    expect(toolCalls[0].kind).toBe("execute");
  });
});

describe("Agent todo/plan updates", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scriptedEvents = [];
  });

  it("TaskCreate emits a plan with one entry per task", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: {
          tasks: [
            { subject: "Write tests", description: "add vitest" },
            { subject: "Build", description: "npm run build" },
          ],
        },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    expect(plans).toHaveLength(1);
    expect(plans[0].entries).toHaveLength(2);
    expect(plans[0].entries[0].content).toBe("Write tests");
    expect(plans[0].entries[1].content).toBe("Build");
    expect(plans[0].entries[0].status).toBe("pending");
  });

  it("TaskUpdate mutates the existing list, not a stacked copy", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: {
          tasks: [{ subject: "First" }, { subject: "Second" }],
        },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        toolInput: { taskId: "task-1", status: "completed" },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    expect(plans).toHaveLength(2);
    // After the update the list is still 2 entries (no stacking).
    const lastPlan = plans[plans.length - 1];
    expect(lastPlan.entries).toHaveLength(2);
    expect(lastPlan.entries[0].status).toBe("completed");
    expect(lastPlan.entries[1].status).toBe("pending");
  });

  it("multiple TaskCreate calls accumulate into one growing list", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "A" }] },
      },
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "B" }, { subject: "C" }] },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    expect(lastPlan.entries).toHaveLength(3);
    expect(lastPlan.entries.map((e: any) => e.content)).toEqual([
      "A",
      "B",
      "C",
    ]);
  });

  it("TaskList does not reset or duplicate the list", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "Only" }] },
      },
      { type: "tool_use", toolName: "TaskList", toolInput: {} },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    // TaskList still emits a snapshot (same list, unchanged).
    const lastPlan = plans[plans.length - 1];
    expect(lastPlan.entries).toHaveLength(1);
    expect(lastPlan.entries[0].content).toBe("Only");
  });

  it("non-task tools do not emit a plan", async () => {
    const { updates } = await runWithEvents([
      { type: "tool_use", toolName: "Read", toolInput: { file_path: "/x" } },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    expect(plans).toHaveLength(0);
  });
});

describe("Subagent (Task tool) lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scriptedEvents = [];
  });

  it("emits in_progress on Task tool_use, completed on tool_result", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Task",
        toolInput: { description: "investigate", subagent_type: "general-purpose" },
        toolUseId: "tu_1",
      },
      { type: "user", text: "result text", toolResultIds: ["tu_1"] },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    // First: in_progress when the subagent starts.
    expect(toolCalls[0].status).toBe("in_progress");
    // Last: a bare completed flip keyed by the same toolCallId.
    const flip = toolCalls[toolCalls.length - 1];
    expect(flip.status).toBe("completed");
    expect(flip.toolCallId).toBe(toolCalls[0].toolCallId);
  });

  it("stays in_progress if no tool_result has arrived yet", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Task",
        toolInput: { description: "investigate" },
        toolUseId: "tu_2",
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0].status).toBe("in_progress");
  });

  it("non-subagent tools stay completed immediately (no lifecycle)", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Read",
        toolInput: { file_path: "/x" },
        toolUseId: "tu_3",
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].status).toBe("completed");
  });
});
