import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

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

describe("MCP config passthrough", () => {
  let runner: ClaudeRunner | undefined;

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    // startSessionWithMcp writes a claude-acp-mcp-* temp dir via buildMcpArgs;
    // cleanup() removes it. Without this the dir leaks into /tmp on every run.
    runner?.cleanup();
    runner = undefined;
  });

  it("should pass --mcp-config when mcpServers provided", async () => {
    runner = new ClaudeRunner({
      model: undefined,
      allowedTools: [],
      maxTurns: undefined,
      timeout: 300000,
      dangerouslySkipPermissions: false,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(
        JSON.stringify({ result: "ok", session_id: "s1" }) + "\n",
        0
      )
    );

    const mcpServers = [
      {
        name: "test-server",
        transport: {
          type: "stdio" as const,
          command: "node",
          args: ["server.js"],
        },
      },
    ];

    await runner.startSessionWithMcp("/tmp", "hello", mcpServers);

    const args = mockSpawn.mock.calls[0][1] as string[];
    expect(args).toContain("--mcp-config");
    // Should have written a temp config file
    const mcpConfigIdx = args.indexOf("--mcp-config");
    const configPath = args[mcpConfigIdx + 1];
    expect(configPath).toBeDefined();
  });

  it("should pass --model/--effort overrides on the MCP path (regression: picker override was dropped)", async () => {
    // Regression: startSessionWithMcp previously called buildExtraArgs() with
    // no modelOverride, so a session's per-session picker override never
    // reached the CLI. The CLI then fell back to the settings default
    // (ANTHROPIC_MODEL, e.g. GLM-5.2) even though the picker showed the
    // override (e.g. Qwen3.8). The fix threads runModel/runEffort through.
    runner = new ClaudeRunner({
      model: undefined, // no global default, so the override is the only source
      allowedTools: [],
      maxTurns: undefined,
      timeout: 300000,
      dangerouslySkipPermissions: false,
    });

    mockSpawn.mockReturnValue(
      createMockProcess(JSON.stringify({ result: "ok", session_id: "s1" }) + "\n", 0)
    );

    const mcpServers = [
      {
        name: "test-server",
        transport: { type: "stdio" as const, command: "node", args: ["server.js"] },
      },
    ];

    await runner.startSessionWithMcp(
      "/tmp",
      "hello",
      mcpServers,
      () => {},
      "tracking",
      "Qwen3.8-27B",
      "high"
    );

    const args = mockSpawn.mock.calls[0][1] as string[];
    const modelIdx = args.indexOf("--model");
    expect(modelIdx).toBeGreaterThan(-1);
    expect(args[modelIdx + 1]).toBe("Qwen3.8-27B");
    const effortIdx = args.indexOf("--effort");
    expect(effortIdx).toBeGreaterThan(-1);
    expect(args[effortIdx + 1]).toBe("high");
  });
});
