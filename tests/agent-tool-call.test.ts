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

  it("Bash tool → strips leading `cd <dir>\\n` so the title shows the real command", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command: "cd /workspace/git/ezennin/claude-code-acp\ngit diff --stat",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("git diff --stat");
    // Full command preserved in rawInput for the client.
    expect(toolCalls[0].rawInput.command).toBe(
      "cd /workspace/git/ezennin/claude-code-acp\ngit diff --stat",
    );
  });

  it("Bash tool → strips leading `cd <dir> && ` on a single line", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command: "cd /workspace/git/ezennin/claude-code-acp && npm run build",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("npm run build");
  });

  it("Bash tool → `cd <dir>\\n<cmd> && ...` keeps the real first command (&& must not cross newlines)", async () => {
    // Regression: a prior `[^&]*&&` regex was greedy across `\n`, so for
    // `cd <dir>\ngit diff --stat && echo ...` it ate up to the first `&&` on
    // the SECOND line and the title lost `git diff --stat`.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command:
            'cd /workspace/git/ezennin/claude-code-acp\ngit diff --stat && echo "=== diff ===" && git diff src/agent.ts',
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe('git diff --stat && echo "=== diff ===" && git diff src/agent.ts');
  });

  it("Bash tool → skips a `# comment` line after `cd` so the real command is the title", async () => {
    // Claude Code sometimes emits `cd <dir>\n# <note>\n<cmd>`. The title must
    // be the command, not the comment.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command:
            "cd /workspace/git/ezennin\n# codex-acp is TypeScript.\ngrep -rnE 'command' ../codex-acp/src",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("grep -rnE 'command' ../codex-acp/src");
  });

  it("Bash tool → multi-line `node -e '…'` uses description (unclosed quote on line 1)", async () => {
    // A multi-line quoted script opens its quote on line 1 (`node -e '`) and
    // closes it many lines later. The first line renders as a truncated
    // `node -e '` title; prefer the tool's `description` when present.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command: "node -e '\nfunction titleForCommand(command) {\n  return command;\n}\nconsole.log(titleForCommand(\"x\"));\n'",
          description: "Test titleForCommand edge cases",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("Test titleForCommand edge cases");
    // Full command still preserved in rawInput.
    expect(toolCalls[0].rawInput.command).toContain("node -e '");
  });

  it("Bash tool → single-line `node -e 'foo'` (closed quote) keeps the command", async () => {
    // A CLOSED quote on line 1 is a complete one-line command — must NOT be
    // replaced by the description.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Bash",
        toolInput: {
          command: "node -e 'console.log(1+1)'",
          description: "Should be ignored — quote is closed",
        },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("node -e 'console.log(1+1)'");
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

  it("Skill tool → 'Read skill: <name>' title (categorizes as a skill load)", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Skill",
        toolInput: { skill: "graphify", args: "some input" },
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("Read skill: graphify");
    // rawInput keeps the skill name so vscode-acp's detectSkillLoad can read it.
    expect(toolCalls[0].rawInput.skill).toBe("graphify");
  });

  it("Skill tool without a name → bare 'Read skill' title", async () => {
    const { updates } = await runWithEvents([
      { type: "tool_use", toolName: "Skill", toolInput: {} },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    expect(toolCalls[0].title).toBe("Read skill");
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

  it("TaskUpdate matches by subject when taskId is Claude-internal", async () => {
    // Claude's TaskUpdate references an internal id we never saw at create
    // time. It must match the existing entry by subject, NOT add a duplicate.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "Alpha" }, { subject: "Beta" }, { subject: "Gamma" }] },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        // Claude-internal id "10" we never issued; subject carries the match.
        toolInput: { taskId: "10", status: "in_progress", subject: "Beta" },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    // Still 3 — no spurious 4th entry from the unmatched id.
    expect(lastPlan.entries).toHaveLength(3);
    const beta = lastPlan.entries.find((e: any) => e.content === "Beta");
    expect(beta.status).toBe("in_progress");
  });

  it("TaskUpdate with an unmatched id and no subject does not inflate the list", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "Only" }] },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        // Bare Claude-internal id, no subject to match — must be ignored.
        toolInput: { taskId: "99", status: "completed" },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    expect(lastPlan.entries).toHaveLength(1);
    expect(lastPlan.entries[0].content).toBe("Only");
  });

  it("TaskUpdate with NO prior TaskCreate emits no plan (no empty 'Plan (no content)')", async () => {
    // Reproduces the memory-replay bug from ACP log 20260730T192213: in a new
    // session GLM-5.2 read prior context and called TaskUpdate {taskId:"1"/"2"/"3",
    // status:"completed"} WITHOUT any TaskCreate. sessionTodos was empty, so each
    // TaskUpdate emitted entries:[] — the UI flashed an empty "Plan (no content)".
    // Fix: when there is nothing tracked, skip the plan emission entirely.
    const { updates } = await runWithEvents([
      { type: "tool_use", toolName: "TaskUpdate", toolInput: { taskId: "1", status: "completed" } },
      { type: "tool_use", toolName: "TaskUpdate", toolInput: { taskId: "2", status: "completed" } },
      { type: "tool_use", toolName: "TaskUpdate", toolInput: { taskId: "3", status: "completed" } },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    // No empty plan frames — nothing to show.
    expect(plans).toHaveLength(0);
  });

  it("TaskCreate scalar form (top-level subject) populates the plan — the GLM-5.2 live shape", async () => {
    // Reproduces the exact traffic from ACP log 20260730T181212: GLM-5.2
    // calls TaskCreate three times, each with a single task described by
    // top-level subject/description/activeForm (NOT a `tasks` array).
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: {
          activeForm: "Creating initial tasks",
          description: "First test task — verify TaskCreate works.",
          subject: "Create initial test task",
        },
      },
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: {
          activeForm: "Creating dependent task",
          description: "Second test task that depends on task #1.",
          subject: "Create dependent task",
        },
      },
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: {
          activeForm: "Creating standalone task",
          description: "Third independent task for testing.",
          subject: "Create standalone cleanup task",
        },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    // All three scalar tasks must show up (previously emitted empty plans).
    expect(lastPlan.entries).toHaveLength(3);
    expect(lastPlan.entries.map((e: any) => e.content)).toEqual([
      "Create initial test task",
      "Create dependent task",
      "Create standalone cleanup task",
    ]);
    expect(lastPlan.entries[0].status).toBe("pending");
  });

  it("TaskUpdate by bare numeric id flips the Nth task status (live shape)", async () => {
    // Live traffic: TaskUpdate { taskId: "1", status: "in_progress" }.
    // The numeric "1" must map to the first synthetic task (task-1).
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "First" }, { subject: "Second" }] },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        toolInput: { taskId: "1", status: "in_progress", owner: "claude" },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    expect(lastPlan.entries).toHaveLength(2);
    expect(lastPlan.entries[0].status).toBe("in_progress");
    expect(lastPlan.entries[1].status).toBe("pending");
  });

  it("TaskUpdate with only addBlockedBy (no status) does not reset status", async () => {
    // Live traffic: TaskUpdate { taskId: "2", addBlockedBy: ["1"] }.
    // No status field — an in_progress task must NOT be reset to pending.
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "TaskCreate",
        toolInput: { tasks: [{ subject: "A" }, { subject: "B" }] },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        toolInput: { taskId: "1", status: "in_progress" },
      },
      {
        type: "tool_use",
        toolName: "TaskUpdate",
        toolInput: { taskId: "2", addBlockedBy: ["1"] },
      },
    ]);
    const plans = updates.filter((u: any) => u.sessionUpdate === "plan");
    const lastPlan = plans[plans.length - 1];
    // Task A stays in_progress; the dependency-only update didn't touch it.
    expect(lastPlan.entries[0].status).toBe("in_progress");
    // Task B is still pending (addBlockedBy doesn't carry a status).
    expect(lastPlan.entries[1].status).toBe("pending");
  });
});

describe("Subagent (Task tool) lifecycle", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    scriptedEvents = [];
  });

  it("emits in_progress on Task tool_use, completed at turn end (not on tool_result)", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Task",
        toolInput: { description: "investigate", subagent_type: "general-purpose" },
        toolUseId: "tu_1",
      },
      // Claude Code's Task tool_result is a launch ack ("Async agent launched
      // successfully..."), NOT a completion signal — the subagent keeps running
      // after it. So this must NOT flip the tool_call to completed.
      { type: "user", text: "result text", toolResultIds: ["tu_1"] },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    // First: in_progress when the subagent starts.
    expect(toolCalls[0].status).toBe("in_progress");
    // The tool_result launch ack must NOT produce a completed flip — only two
    // frames exist (the in_progress emit + the turn-end flush).
    expect(toolCalls).toHaveLength(2);
    // Last: the turn-end flush flips the same toolCallId to completed.
    const flip = toolCalls[toolCalls.length - 1];
    expect(flip.status).toBe("completed");
    expect(flip.toolCallId).toBe(toolCalls[0].toolCallId);
  });

  it("stays in_progress across the launch-ack tool_result, then completes at turn end", async () => {
    const { updates } = await runWithEvents([
      {
        type: "tool_use",
        toolName: "Task",
        toolInput: { description: "investigate" },
        toolUseId: "tu_2",
      },
    ]);
    const toolCalls = updates.filter((u: any) => u.sessionUpdate === "tool_call");
    // in_progress emit + turn-end flush = 2 frames.
    expect(toolCalls).toHaveLength(2);
    expect(toolCalls[0].status).toBe("in_progress");
    expect(toolCalls[1].status).toBe("completed");
    expect(toolCalls[1].toolCallId).toBe(toolCalls[0].toolCallId);
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
