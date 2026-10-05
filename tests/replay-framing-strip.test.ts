import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent, stripReplayPromptFraming } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// Dynamically-controlled fs mock (same pattern as loadSession-compaction.test).
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

describe("stripReplayPromptFraming (unit)", () => {
  it("strips the steer framing prefix", () => {
    expect(
      stripReplayPromptFraming("[Follow-up instruction from user]\nalso check the logs"),
    ).toBe("also check the logs");
  });

  it("strips the goal annotation prefix", () => {
    expect(stripReplayPromptFraming("[Goal: ship it]\n\nrun the tests")).toBe("run the tests");
  });

  it("strips the goal preamble", () => {
    expect(
      stripReplayPromptFraming("Please start working toward this goal now:\n\nship the fix"),
    ).toBe("ship the fix");
  });

  it("strips nested combinations to a fixed point", () => {
    // A steer during an active goal: follow-up wraps goal wraps text.
    expect(
      stripReplayPromptFraming(
        "[Follow-up instruction from user]\n[Goal: ship it]\n\nalso check the logs",
      ),
    ).toBe("also check the logs");
    // A steer of a steer stacks wrappers — all must go.
    expect(
      stripReplayPromptFraming(
        "[Follow-up instruction from user]\n[Follow-up instruction from user]\ngo on",
      ),
    ).toBe("go on");
    // Goal preamble wrapped by a steer.
    expect(
      stripReplayPromptFraming(
        "[Follow-up instruction from user]\nPlease start working toward this goal now:\n\nship it",
      ),
    ).toBe("ship it");
  });

  it("only strips leading framing; prose and interior text stay intact", () => {
    const mid = "first line\n[Follow-up instruction from user]\nnot leading, keep me";
    expect(stripReplayPromptFraming(mid)).toBe(mid);
    const plain = "a user message that merely mentions [Goal: x] inline";
    expect(stripReplayPromptFraming(plain)).toBe(plain);
  });

  // A leading <collaboration_mode> block (prepended to effectivePrompt and sent
  // to the CLI) must be stripped on replay so the user's own message comes back
  // clean. Interior blocks are preserved (the user may legitimately discuss the
  // tag in their text).
  const MODE_BLOCK =
    "<collaboration_mode># Collaboration Mode: Default\n\n" +
    "You are in Default mode: you may read and edit the codebase and execute commands.\n\n" +
    "</collaboration_mode>\n\n";

  it("strips a leading <collaboration_mode> block", () => {
    expect(stripReplayPromptFraming(MODE_BLOCK + "fix the bug")).toBe("fix the bug");
  });

  it("strips a nested <collaboration_mode> + steer framing to a fixed point", () => {
    expect(
      stripReplayPromptFraming(MODE_BLOCK + "[Follow-up instruction from user]\ngo on"),
    ).toBe("go on");
    // Steer framing outside the block (the order the CLI may persist).
    expect(
      stripReplayPromptFraming("[Follow-up instruction from user]\n" + MODE_BLOCK + "ship it"),
    ).toBe("ship it");
  });

  it("keeps an interior <collaboration_mode> block in user prose", () => {
    const mid = "before <collaboration_mode>x</collaboration_mode> after";
    expect(stripReplayPromptFraming(mid)).toBe(mid);
  });

  it("leaves messages without a mode block untouched", () => {
    expect(stripReplayPromptFraming("just a plain message")).toBe("just a plain message");
  });
});

describe("loadSession replay strips steer/goal framing", () => {
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

  it("replays user turns without the follow-up/goal wrappers", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee";
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    fsState.files[transcriptPath] = [
      { type: "user", message: { content: "plain first message" }, timestamp: "t0" },
      {
        type: "assistant",
        message: { content: [{ type: "text", text: "reply" }] },
        timestamp: "t1",
      },
      {
        type: "user",
        message: { content: "[Follow-up instruction from user]\nalso check the logs" },
        timestamp: "t2",
      },
      {
        type: "user",
        message: {
          content: "[Follow-up instruction from user]\n[Goal: ship it]\n\nrun the tests",
        },
        timestamp: "t3",
      },
      // Array-content form (what --input-format=stream-json writes for
      // block prompts): wrapper rides on the text block.
      {
        type: "user",
        message: {
          content: [
            { type: "text", text: "[Follow-up instruction from user]\nblock form steer" },
          ],
        },
        timestamp: "t4",
      },
    ]
      .map((l) => JSON.stringify(l))
      .join("\n");
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd: "/tmp/proj", mcpServers: [] });

    const userTexts = updates
      .filter((u) => u.sessionUpdate === "user_message_chunk")
      .map((u) => u.content?.text);
    expect(userTexts).toEqual([
      "plain first message",
      "also check the logs",
      "run the tests",
      "block form steer",
    ]);
    // No trace of the wrappers may survive into the replay.
    for (const t of userTexts) {
      expect(t).not.toMatch(/\[Follow-up instruction from user\]/);
      expect(t).not.toMatch(/^\[Goal:/);
    }
  });

  it("drops a block-form user turn that is only framing", async () => {
    await agent.initialize({
      protocolVersion: 1,
      clientCapabilities: {},
      clientInfo: { name: "test", version: "1.0" },
    });

    const sessionId = "ffffffff-1111-2222-3333-444444444444";
    const transcriptPath = `/home/test/.claude/projects/-tmp-proj/${sessionId}.jsonl`;
    fsState.files[transcriptPath] = [
      {
        type: "user",
        message: { content: [{ type: "text", text: "[Follow-up instruction from user]\n" }] },
        timestamp: "t0",
      },
    ]
      .map((l) => JSON.stringify(l))
      .join("\n");
    fsState.exists.add(transcriptPath);

    await agent.loadSession!({ sessionId, cwd: "/tmp/proj", mcpServers: [] });

    const userTexts = updates
      .filter((u) => u.sessionUpdate === "user_message_chunk")
      .map((u) => u.content?.text);
    expect(userTexts).toEqual([]);
  });
});
