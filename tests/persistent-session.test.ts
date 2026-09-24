import { describe, it, expect, vi, beforeEach } from "vitest";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";
import { ClaudeRunner } from "../src/claude-runner.js";

vi.mock("node:child_process", () => ({
  spawn: vi.fn(),
}));

import { spawn } from "node:child_process";

const mockSpawn = vi.mocked(spawn);

function createPersistentProc(): ChildProcess & { stdout: EventEmitter; stdin: { write: ReturnType<typeof vi.fn>; writable: boolean } } {
  const proc = new EventEmitter() as ChildProcess & EventEmitter;
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  const stdin = { write: vi.fn(), writable: true, end: vi.fn() };
  (proc as any).stdout = stdout;
  (proc as any).stderr = stderr;
  (proc as any).stdin = stdin;
  (proc as any).pid = 4242;
  (proc as any).exitCode = null;
  (proc as any).kill = vi.fn(() => true);
  return proc as any;
}

describe("persistent Claude process", () => {
  let runner: ClaudeRunner;

  beforeEach(() => {
    vi.clearAllMocks();
    runner = new ClaudeRunner();
  });

  it("writes the follow-up prompt to the same process instead of spawning --resume", async () => {
    const proc = createPersistentProc();
    mockSpawn.mockReturnValue(proc as any);

    const first = runner.startSessionStreaming("/tmp/repo", "hi", () => {}, "acp-1");
    expect(mockSpawn).toHaveBeenCalledTimes(1);
    const firstArgs = mockSpawn.mock.calls[0][1] as string[];
    expect(firstArgs).toContain("--input-format");
    expect(firstArgs).not.toContain("--resume");

    proc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "hello", session_id: "sid-1" }) + "\n"),
    );
    await expect(first).resolves.toEqual({ text: "hello", sessionId: "sid-1" });

    const second = runner.continueSessionStreaming("sid-1", "next", () => {}, "acp-1");
    expect(mockSpawn).toHaveBeenCalledTimes(1);
    const written = (proc.stdin.write as ReturnType<typeof vi.fn>).mock.calls.map((c) => String(c[0])).join("");
    const lines = written.trim().split("\n").map((l) => JSON.parse(l));
    expect(lines).toHaveLength(2);
    expect(lines[1].message.content).toEqual([{ type: "text", text: "next" }]);

    proc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "ok2", session_id: "sid-1" }) + "\n"),
    );
    await expect(second).resolves.toEqual({ text: "ok2", sessionId: "sid-1" });
    expect(runner.isRunning("acp-1")).toBe(false);
  });

  it("respawns with --resume when the permission mode changed", async () => {
    const firstProc = createPersistentProc();
    const secondProc = createPersistentProc();
    mockSpawn.mockReturnValueOnce(firstProc as any).mockReturnValueOnce(secondProc as any);

    const first = runner.startSessionStreaming("/tmp/repo", "hi", () => {}, "acp-1");
    firstProc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "hello", session_id: "sid-1" }) + "\n"),
    );
    await first;

    runner.setSessionPermissionMode("read-only");
    const second = runner.continueSessionStreaming("sid-1", "plan", () => {}, "acp-1");
    expect(mockSpawn).toHaveBeenCalledTimes(2);
    const args = mockSpawn.mock.calls[1][1] as string[];
    expect(args).toContain("--resume");
    expect(args).toContain("sid-1");
    expect(args).toContain("plan");

    secondProc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "planned", session_id: "sid-1" }) + "\n"),
    );
    await expect(second).resolves.toEqual({ text: "planned", sessionId: "sid-1" });
  });
});
