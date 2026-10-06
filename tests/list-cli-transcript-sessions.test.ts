import { describe, it, expect, beforeEach, vi } from "vitest";

// The transcript scanner reads ~/.claude/projects/<encoded-cwd>/<uuid>.jsonl.
// Mock node:os + node:fs so tests serve a synthetic project directory without
// touching the developer's real Claude transcripts.
//
// vi.hoisted runs before the vi.mock factories below (which are themselves
// hoisted above imports), so the constants are initialized in time for the
// factory closures to read them.
const { FAKE_HOME, FAKE_CWD, ENCODED, PROJECT_DIR, fsFiles } = vi.hoisted(() => {
  const FAKE_HOME = "/fake/home";
  const FAKE_CWD = "/workspace/git/test-project";
  const ENCODED = FAKE_CWD.replace(/[^a-zA-Z0-9]/g, "-"); // -workspace-git-test-project
  return {
    FAKE_HOME,
    FAKE_CWD,
    ENCODED,
    PROJECT_DIR: `${FAKE_HOME}/.claude/projects/${ENCODED}`,
    // In-memory filesystem: a map of path -> content. readdirSync derives its
    // result from the keys present under a directory.
    fsFiles: new Map<string, string>(),
  };
});

import { createClaudeCodeAgent, encodeClaudeProjectSlug, listCliTranscriptSessions } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// Build a JSONL transcript from entry specs.
function transcript(entries: Array<Record<string, unknown>>): string {
  return entries.map((e) => JSON.stringify(e)).join("\n") + "\n";
}

vi.mock("node:os", () => ({
  homedir: () => FAKE_HOME,
}));

vi.mock("node:fs", () => ({
  // A path "exists" if it is a file key OR a directory that contains file
  // keys (so the project directory itself reports as existing even though
  // only the transcripts beneath it are stored).
  existsSync: (p: string) =>
    fsFiles.has(p) ||
    [...fsFiles.keys()].some((k) => k.startsWith(p + "/")),
  readFileSync: (p: string) => {
    const v = fsFiles.get(p);
    if (v === undefined) {
      const err = new Error(`ENOENT: ${p}`) as NodeJS.ErrnoException;
      err.code = "ENOENT";
      throw err;
    }
    return v;
  },
  readdirSync: (dir: string) =>
    [...fsFiles.keys()]
      .filter((k) => k.startsWith(dir + "/"))
      .map((k) => k.slice(dir.length + 1)),
  statSync: (_p: string) => ({
    mtimeMs: 0,
    mtime: new Date(0),
  }),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}));

// SessionStore must not read real persisted sessions; the fs mock above keeps
// the store empty by default (no sessions.json content key is set).
vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

vi.mock("../src/claude-runner.js", () => ({
  ClaudeRunner: vi.fn(),
}));

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

function writeTranscript(name: string, content: string): void {
  fsFiles.set(`${PROJECT_DIR}/${name}`, content);
}

beforeEach(() => {
  fsFiles.clear();
});

describe("encodeClaudeProjectSlug", () => {
  it("encodes a Windows cwd the way the Claude CLI names the project folder", () => {
    expect(encodeClaudeProjectSlug("e:\\projects\\vscode-acp")).toBe(
      "e--projects-vscode-acp",
    );
  });

  it("encodes a POSIX cwd by turning slashes into dashes", () => {
    expect(encodeClaudeProjectSlug("/workspace/git/test-project")).toBe(
      "-workspace-git-test-project",
    );
  });
});

describe("listCliTranscriptSessions", () => {
  it("returns nothing when no project directory exists", async () => {
    const sessions = await listCliTranscriptSessions(FAKE_CWD);
    expect(sessions).toEqual([]);
  });

  it("lists each transcript with sessionId derived from the filename", async () => {
    writeTranscript(
      "aaaa0000-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "summary", summary: "fix login bug", timestamp: "2026-08-01T10:00:00.000Z" },
        { type: "user", message: { content: "hi" }, timestamp: "2026-08-01T10:00:01.000Z" },
      ])
    );
    writeTranscript(
      "bbbb1111-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "user", message: { content: "explain this repo" }, timestamp: "2026-08-02T10:00:00.000Z" },
      ])
    );

    const sessions = await listCliTranscriptSessions(FAKE_CWD);
    expect(sessions).toHaveLength(2);
    const ids = sessions.map((s) => s.sessionId).sort();
    expect(ids).toEqual([
      "aaaa0000-0000-0000-0000-000000000000",
      "bbbb1111-0000-0000-0000-000000000000",
    ]);
    expect(sessions.every((s) => s.cwd === FAKE_CWD)).toBe(true);
  });

  it("prefers the summary entry as the title", async () => {
    writeTranscript(
      "cccc2222-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "summary", summary: "a short label", timestamp: "2026-08-01T10:00:00.000Z" },
        { type: "user", message: { content: "ignored first prompt" }, timestamp: "2026-08-01T10:00:01.000Z" },
      ])
    );
    const [s] = await listCliTranscriptSessions(FAKE_CWD);
    expect(s.title).toBe("a short label");
  });

  it("falls back to the first user message text when no summary exists", async () => {
    writeTranscript(
      "dddd3333-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "user", message: { content: "what does this function do?" }, timestamp: "2026-08-01T10:00:00.000Z" },
      ])
    );
    const [s] = await listCliTranscriptSessions(FAKE_CWD);
    expect(s.title).toBe("what does this function do?");
  });

  it("falls back to first user text-block content for array content", async () => {
    writeTranscript(
      "eeee4444-0000-0000-0000-000000000000.jsonl",
      transcript([
        {
          type: "user",
          message: { content: [{ type: "text", text: "array-style prompt" }] },
          timestamp: "2026-08-01T10:00:00.000Z",
        },
      ])
    );
    const [s] = await listCliTranscriptSessions(FAKE_CWD);
    expect(s.title).toBe("array-style prompt");
  });

  it("uses the last entry timestamp as updatedAt", async () => {
    writeTranscript(
      "ffff5555-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "user", message: { content: "first" }, timestamp: "2026-08-01T10:00:00.000Z" },
        { type: "assistant", message: { content: [] }, timestamp: "2026-08-01T10:05:00.000Z" },
        { type: "user", message: { content: "second" }, timestamp: "2026-08-01T10:10:00.000Z" },
      ])
    );
    const [s] = await listCliTranscriptSessions(FAKE_CWD);
    expect(s.updatedAt).toBe("2026-08-01T10:10:00.000Z");
  });

  it("skips unparseable lines without throwing", async () => {
    // A file whose lines are all non-JSON still yields an entry (with the
    // default title and stat-based updatedAt) — the per-line JSON.parse is
    // guarded, so corrupt lines are skipped rather than crashing the scan.
    fsFiles.set(
      `${PROJECT_DIR}/bad00000-0000-0000-0000-000000000000.jsonl`,
      "{ not json at all\nalso not json\n"
    );
    const sessions = await listCliTranscriptSessions(FAKE_CWD);
    expect(sessions).toHaveLength(1);
    expect(sessions[0].title).toBe("Claude CLI session");
  });

  it("returns nothing when cwd is undefined", async () => {
    const sessions = await listCliTranscriptSessions(undefined);
    expect(sessions).toEqual([]);
  });
});

describe("agent.listSessions merge", () => {
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    agent = createClaudeCodeAgent(createMockConnection());
  });

  it("surfaces CLI transcripts alongside ACP store sessions", async () => {
    writeTranscript(
      "cli00000-0000-0000-0000-000000000000.jsonl",
      transcript([
        { type: "summary", summary: "cli session", timestamp: "2026-08-01T10:00:00.000Z" },
      ])
    );

    const res = await agent.listSessions({ cwd: FAKE_CWD });
    expect(res.sessions).toHaveLength(1);
    expect(res.sessions[0].sessionId).toBe("cli00000-0000-0000-0000-000000000000");
    expect(res.sessions[0].title).toBe("cli session");
    expect(res.sessions[0].cwd).toBe(FAKE_CWD);
  });

  it("sorts merged sessions by updatedAt descending", async () => {
    writeTranscript(
      "older0000-0000-0000-0000-000000000000.jsonl",
      transcript([{ type: "summary", summary: "older", timestamp: "2026-08-01T10:00:00.000Z" }])
    );
    writeTranscript(
      "newer0000-0000-0000-0000-000000000000.jsonl",
      transcript([{ type: "summary", summary: "newer", timestamp: "2026-08-05T10:00:00.000Z" }])
    );

    const res = await agent.listSessions({ cwd: FAKE_CWD });
    expect(res.sessions.map((s) => s.title)).toEqual(["newer", "older"]);
  });

  it("does not strip required SessionInfo wire fields", async () => {
    writeTranscript(
      "wire00000-0000-0000-0000-000000000000.jsonl",
      transcript([{ type: "summary", summary: "wire", timestamp: "2026-08-01T10:00:00.000Z" }])
    );
    const res = await agent.listSessions({ cwd: FAKE_CWD });
    const s = res.sessions[0];
    // The internal claudeSessionId field must NOT appear on the wire shape.
    expect(s).not.toHaveProperty("claudeSessionId");
    expect(s).toHaveProperty("sessionId");
    expect(s).toHaveProperty("cwd");
    expect(s).toHaveProperty("title");
    expect(s).toHaveProperty("updatedAt");
  });
});
