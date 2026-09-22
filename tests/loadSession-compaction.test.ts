import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// Dynamically-controlled fs mock. Each test stages the transcript contents that
// readFileSync should return for the replay path, and decides which paths
// existSync reports as present.
const fsState: { files: Record<string, string>; exists: Set<string> } = {
  files: {},
  exists: new Set(),
};

vi.mock("node:fs", () => ({
  existsSync: vi.fn((p: string) => fsState.exists.has(String(p))),
  readFileSync: vi.fn((p: string) => fsState.files[String(p)] ?? ""),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  readdirSync: vi.fn(() => []),
  statSync: vi.fn(() => ({ mtimeMs: 0 })),
}));

vi.mock("node:os", () => ({
  homedir: () => "/home/test",
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockResolvedValue({
      text: "ok",
      sessionId: "bound-claude-session",
    });
    this.continueSessionStreaming = vi.fn().mockResolvedValue({
      text: "ok",
      sessionId: "bound-claude-session",
    });
    this.cancel = vi.fn();
  });
  return { ClaudeRunner: MockClaudeRunner };
});

function createMockConnection() {
  const updates: any[] = [];
  return {
    conn: {
      sessionUpdate: vi.fn(async (params: any) => {
        updates.push(params.update);
        return undefined;
      }),
      requestPermission: vi.fn(),
      readTextFile: vi.fn(),
      writeTextFile: vi.fn(),
      createTerminal: vi.fn(),
      signal: new AbortController().signal,
      closed: new Promise(() => {}),
    } as unknown as AgentSideConnection,
    updates,
  };
}

// Build a transcript where a `summary` (compaction) marker sits between an
// early pre-compaction exchange and a later post-compaction exchange.
function buildCompactedTranscript(): string {
  const lines: any[] = [
    {
      type: "user",
      message: { content: "pre-compact user message" },
      timestamp: "t0",
    },
    {
      type: "assistant",
      message: {
        content: [{ type: "text", text: "pre-compact assistant reply" }],
      },
      timestamp: "t1",
    },
    // Compaction marker: the CLI replaces the conversation above with this
    // summary. The replay must DROP the two preceding turns.
    { type: "summary", summary: "Compacted summary of earlier conversation", timestamp: "t2" },
    {
      type: "user",
      message: { content: "post-compact user message" },
      timestamp: "t3",
    },
    {
      type: "assistant",
      message: {
        content: [{ type: "text", text: "post-compact assistant reply" }],
      },
      timestamp: "t4",
    },
  ];
  return lines.map((l) => JSON.stringify(l)).join("\n");
}

describe("loadSession replay compaction truncation", () => {
  let agent: ReturnType<typeof createClaudeCodeAgent>;
  let conn: AgentSideConnection;
  let updates: any[];

  beforeEach(() => {
    vi.clearAllMocks();
    fsState.files = {};
    fsState.exists = new Set();
    ({ conn, updates } = createMockConnection());
    agent = createClaudeCodeAgent(conn);
  });

  it("drops turns before the last `summary` marker (only post-compact turns replay)", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "11111111-2222-3333-4444-555555555555";
    const cwd = "/tmp/proj";
    // loadSession constructs: /home/test/.claude/projects/-tmp-proj/<id>.jsonl
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    fsState.files[transcriptPath] = buildCompactedTranscript();
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    const texts = updates
      .filter(
        (u) =>
          u.sessionUpdate === "user_message_chunk" ||
          u.sessionUpdate === "agent_message_chunk",
      )
      .map((u) => u.content?.text)
      .filter(Boolean);

    // Post-compact turns must be present…
    expect(texts).toContain("post-compact user message");
    expect(texts).toContain("post-compact assistant reply");
    // …and the pre-compact turns must have been truncated by the summary marker.
    expect(texts).not.toContain("pre-compact user message");
    expect(texts).not.toContain("pre-compact assistant reply");
    // The summary itself must not be replayed as a message.
    expect(texts.some((t) => /Compacted summary/.test(t))).toBe(false);
  });

  it("replays the full transcript when there is no compaction marker", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    const cwd = "/tmp/proj";
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    const lines: any[] = [
      { type: "user", message: { content: "first user" } },
      {
        type: "assistant",
        message: { content: [{ type: "text", text: "first reply" }] },
      },
      { type: "user", message: { content: "second user" } },
    ];
    fsState.files[transcriptPath] = lines.map((l) => JSON.stringify(l)).join("\n");
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    const texts = updates
      .filter(
        (u) =>
          u.sessionUpdate === "user_message_chunk" ||
          u.sessionUpdate === "agent_message_chunk",
      )
      .map((u) => u.content?.text)
      .filter(Boolean);

    expect(texts).toEqual(["first user", "first reply", "second user"]);
  });

  it("skips a compact-summary continuation entry so it is not replayed as a user message", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "bbbbbbbb-aaaa-cccc-dddd-eeeeeeeeeeee";
    const cwd = "/tmp/proj";
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    // After a context overflow the CLI rewrites the conversation and records
    // the new leading turn as a user entry flagged isCompactSummary — the
    // machine "This session is being continued from a previous conversation…"
    // banner. It must not be replayed as a user message; only the real
    // post-compact prompt is.
    const lines: any[] = [
      {
        type: "user",
        isCompactSummary: true,
        isVisibleInTranscriptOnly: true,
        message: {
          content:
            "This session is being continued from a previous conversation that " +
            "ran out of context. The summary below covers the earlier portion " +
            "of the conversation.\n\nSummary:\n1. Primary Request and Intent:\n  …",
        },
        timestamp: "t2",
      },
      { type: "user", message: { content: "real post-compact prompt" }, timestamp: "t3" },
      {
        type: "assistant",
        message: { content: [{ type: "text", text: "post-compact reply" }] },
        timestamp: "t4",
      },
    ];
    fsState.files[transcriptPath] = lines.map((l) => JSON.stringify(l)).join("\n");
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    const texts = updates
      .filter((u) => u.sessionUpdate === "user_message_chunk")
      .map((u) => u.content?.text)
      .filter(Boolean);

    // The compact-summary banner must not surface as a user message…
    expect(
      texts.some((t) => /This session is being continued from a previous conversation/.test(t)),
    ).toBe(false);
    // …and the real prompt + reply must still replay.
    expect(texts).toContain("real post-compact prompt");
    expect(
      updates.some(
        (u) => u.sessionUpdate === "agent_message_chunk" && u.content?.text === "post-compact reply",
      ),
    ).toBe(true);
  });

  it("reconstructs the plan snapshot from replayed TaskCreate/TaskUpdate tool_use blocks", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "cccccccc-dddd-eeee-ffff-000000000001";
    const cwd = "/tmp/proj";
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    const lines: any[] = [
      {
        type: "user",
        message: { content: "do the tasks" },
      },
      {
        type: "assistant",
        message: {
          content: [
            {
              type: "tool_use",
              id: "toolu-create-1",
              name: "TaskCreate",
              input: {
                tasks: [
                  { subject: "Write tests" },
                  { subject: "Ship fix" },
                ],
              },
            },
          ],
        },
      },
      {
        type: "assistant",
        message: {
          content: [
            {
              type: "tool_use",
              id: "toolu-update-1",
              name: "TaskUpdate",
              input: { taskId: "1", status: "completed" },
            },
          ],
        },
      },
    ];
    fsState.files[transcriptPath] = lines.map((l) => JSON.stringify(l)).join("\n");
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    const planSnapshots = updates.filter((u) => u.sessionUpdate === "plan");
    // At least one plan snapshot must reach the client from the replay.
    expect(planSnapshots.length).toBeGreaterThan(0);
    const last = planSnapshots[planSnapshots.length - 1];
    // The final snapshot reflects the TaskUpdate: "Write tests" is completed,
    // "Ship fix" is still pending.
    const byContent: Record<string, string> = {};
    for (const e of last.entries) byContent[e.content] = e.status;
    expect(byContent["Write tests"]).toBe("completed");
    expect(byContent["Ship fix"]).toBe("pending");
  });
});
