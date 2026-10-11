import { describe, it, expect, vi } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";
import { join } from "node:path";

// Dynamically-controlled fs mock. The session store reads its sessions.json
// through node:fs at construction, so the SAME mock that stages the replay
// transcript also seeds the store's cwd→Claude-UUID mapping — which is exactly
// the handle /clear empties (setClaudeSessionId(sessionId, "")).
const fsState: {
  files: Record<string, string>;
  exists: Set<string>;
  dirs: Record<string, string[]>;
  mtimes: Record<string, number>;
} = {
  files: {},
  exists: new Set(),
  dirs: {},
  mtimes: {},
};

vi.mock("node:fs", () => ({
  existsSync: vi.fn((p: string) => fsState.exists.has(String(p))),
  readFileSync: vi.fn((p: string) => fsState.files[String(p)] ?? ""),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  readdirSync: vi.fn((p: string) => fsState.dirs[String(p)] ?? []),
  statSync: vi.fn((p: string) => ({ mtimeMs: fsState.mtimes[String(p)] ?? 0 })),
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
    this.startSessionStreaming = vi.fn().mockResolvedValue({ text: "ok", sessionId: "bound-claude-session" });
    this.continueSessionStreaming = vi.fn().mockResolvedValue({ text: "ok", sessionId: "bound-claude-session" });
    this.cancel = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      (_cwd: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.startSessionStreaming(_cwd, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e),
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      (_sid: string, _content: any, _onEvent: any, _trackingId?: string, _m?: string, _e?: string, _mcp?: any) =>
        this.continueSessionStreaming(_sid, (typeof _content?.[0]?.text === "string" ? _content[0].text : ""), _onEvent, _trackingId, _m, _e),
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

// The store resolves STORE_DIR from CLAUDE_ACP_STORE_DIR at module load (set by
// tests/setup.ts). Stage sessions.json there so the store rehydrates with the
// exact claudeSessionId the test wants to exercise.
const storeFile = () => join(process.env.CLAUDE_ACP_STORE_DIR!, "sessions.json");

function seedStore(sessionId: string, cwd: string, claudeSessionId: undefined | string) {
  const data: any = {
    cwd,
    mcpServers: [],
    createdAt: "2020-01-01T00:00:00.000Z",
    updatedAt: "2020-01-01T00:00:00.000Z",
    configOverrides: {},
  };
  // undefined → genuinely-unknown mapping (pre-persistence sessions); "" → the
  // deliberate wipe /clear performs.
  if (claudeSessionId !== undefined) data.claudeSessionId = claudeSessionId;
  fsState.files[storeFile()] = JSON.stringify({
    version: 2,
    byPersistKey: { [`cwd:${cwd}`]: claudeSessionId ?? "stale" },
    sessions: { [sessionId]: data },
  });
  fsState.exists.add(storeFile());
}

// A transcript that survives on disk for the cwd — the newest-transcript
// fallback would replay it if the mapping were merely unknown.
function stageNewestTranscript(cwd: string, slug: string, uuid: string): string {
  void cwd;
  const projectDir = `/home/test/.claude/projects/${slug}`;
  const path = `${projectDir}/${uuid}.jsonl`;
  const lines = [
    { type: "user", message: { content: "pre-clear user message" } },
    { type: "assistant", message: { content: [{ type: "text", text: "pre-clear reply" }] } },
  ];
  fsState.files[path] = lines.map((l) => JSON.stringify(l)).join("\n");
  fsState.exists.add(path);
  fsState.exists.add(projectDir);
  fsState.dirs[projectDir] = [`${uuid}.jsonl`];
  fsState.mtimes[path] = 1000;
  return path;
}

function replayedTexts(updates: any[]): string[] {
  return updates
    .filter((u) => u.sessionUpdate === "user_message_chunk" || u.sessionUpdate === "agent_message_chunk")
    .map((u) => u.content?.text)
    .filter(Boolean);
}

describe("loadSession replay after /clear", () => {
  it("a cleared session replays NOTHING even when a transcript is on disk", async () => {
    const sessionId = "acp-cleared-1";
    const cwd = "/tmp/proj";
    // /clear set the mapping to "" — the conversation was wiped.
    seedStore(sessionId, cwd, "");
    // A stale transcript still sits on disk for this cwd (the pre-clear turns).
    stageNewestTranscript(cwd, "-tmp-proj", "deadbeef-0000-0000-0000-000000000000");

    // Seed BEFORE the agent is built: the store rehydrates at construction.
    const { conn, updates } = createMockConnection();
    const agent = createClaudeCodeAgent(conn);
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "test", version: "1.0" } });

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    // The resumed panel must be empty: the user cleared these turns, so replaying
    // them is the reported bug.
    expect(replayedTexts(updates)).toEqual([]);
  });

  it("an unknown mapping (undefined) still falls back to the newest transcript", async () => {
    const sessionId = "acp-legacy-1";
    const cwd = "/tmp/proj";
    // No claudeSessionId key at all — sessions written before full-record
    // persistence. The fallback must still render the conversation.
    seedStore(sessionId, cwd, undefined);
    stageNewestTranscript(cwd, "-tmp-proj", "cafebabe-0000-0000-0000-000000000000");

    const { conn, updates } = createMockConnection();
    const agent = createClaudeCodeAgent(conn);
    await agent.initialize({ protocolVersion: 1, clientCapabilities: {}, clientInfo: { name: "test", version: "1.0" } });

    await agent.loadSession!({ sessionId, cwd, mcpServers: [] });

    const texts = replayedTexts(updates);
    expect(texts).toContain("pre-clear user message");
    expect(texts).toContain("pre-clear reply");
  });
});
