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

// On the ericai-proxy branch, sanitizeEnv() keeps the entire process.env
// (including ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN) so the claude CLI
// routes through our CCR gateway → ericai-proxy → EricAI vLLM. The upstream
// repo deletes these to force subscription auth; this branch needs them.
describe("ClaudeRunner env preservation (ericai-proxy branch)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should keep ANTHROPIC_API_KEY in the subprocess env", async () => {
    process.env.ANTHROPIC_API_KEY = "sk-test-key";

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

    const spawnEnv = (mockSpawn.mock.calls[0][2] as any).env;
    expect(spawnEnv.ANTHROPIC_API_KEY).toBe("sk-test-key");

    delete process.env.ANTHROPIC_API_KEY;
  });

  it("should keep ANTHROPIC_AUTH_TOKEN in the subprocess env", async () => {
    process.env.ANTHROPIC_AUTH_TOKEN = "token-test";

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

    const spawnEnv = (mockSpawn.mock.calls[0][2] as any).env;
    expect(spawnEnv.ANTHROPIC_AUTH_TOKEN).toBe("token-test");

    delete process.env.ANTHROPIC_AUTH_TOKEN;
  });

  it("should pass through ANTHROPIC_BASE_URL and ANTHROPIC_MODEL", async () => {
    process.env.ANTHROPIC_BASE_URL = "http://127.0.0.1:3457";
    process.env.ANTHROPIC_MODEL = "glm-5.2";

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

    const spawnEnv = (mockSpawn.mock.calls[0][2] as any).env;
    expect(spawnEnv.ANTHROPIC_BASE_URL).toBe("http://127.0.0.1:3457");
    expect(spawnEnv.ANTHROPIC_MODEL).toBe("glm-5.2");

    delete process.env.ANTHROPIC_BASE_URL;
    delete process.env.ANTHROPIC_MODEL;
  });
});
