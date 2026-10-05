import { describe, it, expect, vi, beforeEach } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";

vi.mock("node:child_process", () => ({
  spawn: vi.fn(),
}));

import { spawn } from "node:child_process";
const mockSpawn = vi.mocked(spawn);

function createMockProcess(output: string, exitCode = 0): ChildProcess {
  const proc = new EventEmitter() as ChildProcess & EventEmitter;
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  (proc as any).stdout = stdout;
  (proc as any).stderr = stderr;
  (proc as any).pid = 12345;
  (proc as any).kill = vi.fn();

  setTimeout(() => {
    stdout.emit("data", Buffer.from(output));
    setTimeout(() => proc.emit("close", exitCode), 5);
  }, 5);

  return proc;
}

describe("ClaudeRunner with config", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should add --model flag when model is set", async () => {
    const runner = new ClaudeRunner({
      model: "opus",
      allowedTools: [],
      maxTurns: undefined,
      timeout: 300000,
      dangerouslySkipPermissions: false,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }))
    );

    await runner.startSession("/tmp", "test");

    expect(mockSpawn).toHaveBeenCalledWith(
      "claude",
      expect.arrayContaining(["--model", "opus"]),
      expect.any(Object)
    );
  });

  it("should add --max-turns flag when set", async () => {
    const runner = new ClaudeRunner({
      model: undefined,
      allowedTools: [],
      maxTurns: 5,
      timeout: 300000,
      dangerouslySkipPermissions: false,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }))
    );

    await runner.startSession("/tmp", "test");

    expect(mockSpawn).toHaveBeenCalledWith(
      "claude",
      expect.arrayContaining(["--max-turns", "5"]),
      expect.any(Object)
    );
  });

  it("should add --dangerously-skip-permissions when set", async () => {
    const runner = new ClaudeRunner({
      model: undefined,
      allowedTools: [],
      maxTurns: undefined,
      timeout: 300000,
      dangerouslySkipPermissions: true,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }))
    );

    await runner.startSession("/tmp", "test");

    expect(mockSpawn).toHaveBeenCalledWith(
      "claude",
      expect.arrayContaining(["--dangerously-skip-permissions"]),
      expect.any(Object)
    );
  });

  it("should add --allowedTools for each tool", async () => {
    const runner = new ClaudeRunner({
      model: undefined,
      allowedTools: ["Bash", "Read"],
      maxTurns: undefined,
      timeout: 300000,
      dangerouslySkipPermissions: false,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }))
    );

    await runner.startSession("/tmp", "test");

    const call = mockSpawn.mock.calls[0];
    const args = call[1] as string[];
    expect(args.filter((a) => a === "--allowedTools")).toHaveLength(2);
    expect(args).toContain("Bash");
    expect(args).toContain("Read");
  });

  it("keeps ANTHROPIC_API_KEY but strips nested-session vars", async () => {
    // sanitizeEnv intentionally KEEPS ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
    // so the spawned claude CLI routes through the CCR gateway → ericai-proxy
    // → EricAI vLLM (API-key auth, not subscription auth). It strips the
    // CLAUDECODE* vars that would make claude refuse a nested session.
    const origKey = process.env.ANTHROPIC_API_KEY;
    const origClaudecode = process.env.CLAUDECODE;
    process.env.ANTHROPIC_API_KEY = "test-proxy-key";
    process.env.CLAUDECODE = "1";
    try {
      const runner = new ClaudeRunner({
        model: undefined,
        allowedTools: [],
        maxTurns: undefined,
        timeout: 300000,
        dangerouslySkipPermissions: false,
      });

      mockSpawn.mockReturnValue(
        createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }))
      );

      await runner.startSession("/tmp", "test");

      const spawnOptions = mockSpawn.mock.calls[0][2] as any;
      // The proxy API key is preserved for the gateway chain.
      expect(spawnOptions.env.ANTHROPIC_API_KEY).toBe("test-proxy-key");
      // Nested-session markers are stripped so claude doesn't refuse to run.
      expect(spawnOptions.env.CLAUDECODE).toBeUndefined();
      expect(spawnOptions.env.CLAUDE_CODE_SESSION_ID).toBeUndefined();
    } finally {
      // Restore the test process's real env so sibling tests are unaffected.
      if (origKey === undefined) delete process.env.ANTHROPIC_API_KEY;
      else process.env.ANTHROPIC_API_KEY = origKey;
      if (origClaudecode === undefined) delete process.env.CLAUDECODE;
      else process.env.CLAUDECODE = origClaudecode;
    }
  });
});
