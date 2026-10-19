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

  it("disables CLI auto-compact on the child so ACP owns the threshold", async () => {
    const savedAuto = process.env.DISABLE_AUTO_COMPACT;
    const savedAll = process.env.DISABLE_COMPACT;
    delete process.env.DISABLE_AUTO_COMPACT;
    delete process.env.DISABLE_COMPACT;

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
    expect(spawnEnv.DISABLE_AUTO_COMPACT).toBe("1");
    expect(spawnEnv.DISABLE_COMPACT).toBeUndefined();

    if (savedAuto === undefined) delete process.env.DISABLE_AUTO_COMPACT;
    else process.env.DISABLE_AUTO_COMPACT = savedAuto;
    if (savedAll === undefined) delete process.env.DISABLE_COMPACT;
    else process.env.DISABLE_COMPACT = savedAll;
  });

  it("leaves an explicit DISABLE_AUTO_COMPACT from the user alone", async () => {
    const savedAuto = process.env.DISABLE_AUTO_COMPACT;
    process.env.DISABLE_AUTO_COMPACT = "0";

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
    expect(spawnEnv.DISABLE_AUTO_COMPACT).toBe("0");

    if (savedAuto === undefined) delete process.env.DISABLE_AUTO_COMPACT;
    else process.env.DISABLE_AUTO_COMPACT = savedAuto;
  });
});
