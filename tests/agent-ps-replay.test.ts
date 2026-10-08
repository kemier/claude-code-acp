import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// Dynamically-controlled fs mock. Each test stages the transcript contents that
// readFileSync should return for the replay path, and decides which paths
// existsSync reports as present.
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
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e)
    );
    this.setSessionPermissionMode = vi.fn();
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

const SESSION = "11111111-2222-3333-4444-555555555555";
const CWD = "/tmp/proj";
const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${SESSION}.jsonl`;

/** A background command's <task-notification> record (how the CLI reports a
 *  finished background Bash command; the CLI's pre-resume teardown record uses
 *  status "stopped" with no completion record). */
function bgNotification(tuId: string, status: string, summary: string): string {
  return (
    `<task-notification>\n<task-id>task-${tuId}</task-id>\n` +
    `<tool-use-id>${tuId}</tool-use-id>\n` +
    `<status>${status}</status>\n<summary>${summary}</summary>\n</task-notification>`
  );
}

/** Transcript: one backgrounded command with NO completion record (should
 *  survive rehydration), one with a completion <task-notification> (should be
 *  dropped). */
function buildTranscript(): string {
  const lines: any[] = [
    { type: "user", message: { content: "run the sweep in the background" } },
    {
      type: "assistant",
      message: {
        content: [
          { type: "text", text: "Starting it." },
          {
            type: "tool_use",
            id: "tu-live",
            name: "Bash",
            input: { command: "npm run sweep", run_in_background: true },
          },
          {
            type: "tool_use",
            id: "tu-done",
            name: "Bash",
            input: { command: "sleep 2", run_in_background: true },
          },
        ],
      },
    },
    {
      type: "user",
      message: {
        content: [
          {
            type: "tool_result",
            tool_use_id: "tu-live",
            content: "Command running in background with ID: task-live.",
          },
          {
            type: "tool_result",
            tool_use_id: "tu-done",
            content: bgNotification("tu-done", "completed", "Background command \"sleep 2\" completed (exit code 0)"),
          },
        ],
      },
    },
  ];
  return lines.map((l) => JSON.stringify(l)).join("\n");
}

describe("loadSession replay rehydrates the background-Bash tracker", () => {
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

  async function loadAndPs(): Promise<string[]> {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    fsState.files[transcriptPath] = buildTranscript();
    fsState.exists.add(transcriptPath);
    await agent.loadSession!({ sessionId: SESSION, cwd: CWD, mcpServers: [] });
    await agent.prompt({
      sessionId: SESSION,
      prompt: [{ type: "text", text: "/ps" }],
    });
    return updates
      .filter(
        (u) =>
          u.sessionUpdate === "agent_message_chunk" &&
          u._meta?.commandOutput === true,
      )
      .map((u) => u.content.text);
  }

  it("lists a pre-resume background command with no completion record", async () => {
    const out = await loadAndPs();
    const ps = out.find((t) => /background command/.test(t));
    expect(ps).toBeTruthy();
    expect(ps).toContain("npm run sweep");
    // The command with a completion record must NOT be listed.
    expect(ps).not.toContain("sleep 2");
  }, 20_000);

  it("says no background terminals (with the restored-session notice) when the tracker is empty", async () => {
    // Transcript with a background command whose completion DID surface —
    // rehydration registers it, then the <task-notification> drops it.
    fsState.files = {};
    fsState.exists = new Set();
    const lines: any[] = [
      { type: "user", message: { content: "go" } },
      {
        type: "assistant",
        message: {
          content: [
            { type: "tool_use", id: "tu-done", name: "Bash", input: { command: "sleep 2", run_in_background: true } },
          ],
        },
      },
      {
        type: "user",
        message: {
          content: [
            { type: "tool_result", tool_use_id: "tu-done", content: bgNotification("tu-done", "completed", "done") },
          ],
        },
      },
    ];
    fsState.files[transcriptPath] = lines.map((l) => JSON.stringify(l)).join("\n");
    fsState.exists.add(transcriptPath);

    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    await agent.loadSession!({ sessionId: SESSION, cwd: CWD, mcpServers: [] });
    await agent.prompt({
      sessionId: SESSION,
      prompt: [{ type: "text", text: "/ps" }],
    });
    const out = updates
      .filter(
        (u) =>
          u.sessionUpdate === "agent_message_chunk" &&
          u._meta?.commandOutput === true,
      )
      .map((u) => u.content.text);
    const ps = out.find((t) => /No background terminals/.test(t));
    expect(ps).toBeTruthy();
    expect(ps).toContain("restored session");
  }, 20_000);

  it("drops a background command whose <task-notification> arrives as plain user text", async () => {
    // The CLI sometimes injects the completion as a plain user text message
    // (queued_command), not inside a tool_result — rehydration must drop it
    // by the <tool-use-id> embedded in the notification.
    const lines: any[] = [
      { type: "user", message: { content: "go" } },
      {
        type: "assistant",
        message: {
          content: [
            { type: "tool_use", id: "tu-text", name: "Bash", input: { command: "long-job", run_in_background: true } },
          ],
        },
      },
      {
        type: "user",
        message: { content: bgNotification("tu-text", "stopped", "no completion record found") },
      },
    ];
    fsState.files[transcriptPath] = lines.map((l) => JSON.stringify(l)).join("\n");
    fsState.exists.add(transcriptPath);

    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });
    await agent.loadSession!({ sessionId: SESSION, cwd: CWD, mcpServers: [] });
    await agent.prompt({
      sessionId: SESSION,
      prompt: [{ type: "text", text: "/ps" }],
    });
    const out = updates
      .filter(
        (u) =>
          u.sessionUpdate === "agent_message_chunk" &&
          u._meta?.commandOutput === true,
      )
      .map((u) => u.content.text);
    const ps = out.find((t) => /No background terminals/.test(t));
    expect(ps).toBeTruthy();
    expect(ps).not.toContain("long-job");
  }, 20_000);
});
