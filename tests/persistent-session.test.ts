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

  it("resumes in the same cwd after the interrupted process is discarded", async () => {
    const firstProc = createPersistentProc();
    const secondProc = createPersistentProc();
    mockSpawn.mockReturnValueOnce(firstProc as any).mockReturnValueOnce(secondProc as any);

    const first = runner.startSessionStreaming("/tmp/repo", "hi", () => {}, "acp-1");
    firstProc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "hello", session_id: "sid-1" }) + "\n"),
    );
    await first;

    runner.discardLiveSession("acp-1");
    const second = runner.continueSessionStreaming("sid-1", "summarize", () => {}, "acp-1");
    expect(mockSpawn).toHaveBeenCalledTimes(2);
    const args = mockSpawn.mock.calls[1][1] as string[];
    expect(args).toContain("--resume");
    expect(args).toContain("sid-1");
    expect(mockSpawn.mock.calls[1][2]).toMatchObject({ cwd: "/tmp/repo" });

    secondProc.stdout.emit(
      "data",
      Buffer.from(JSON.stringify({ type: "result", result: "summary", session_id: "sid-1" }) + "\n"),
    );
    await expect(second).resolves.toEqual({ text: "summary", sessionId: "sid-1" });
  });

  it("keeps the prompt open when resume drains a task-notification result first", async () => {
    const proc = createPersistentProc();
    mockSpawn.mockReturnValue(proc as any);
    const events: { type?: string; toolName?: string; text?: string }[] = [];

    const turn = runner.startSessionStreaming(
      "/tmp/repo",
      "check vscode-acp",
      (event) => events.push(event),
      "acp-1",
    );

    const notification = JSON.stringify({
      type: "user",
      message: {
        role: "user",
        content: [
          {
            type: "text",
            text: "<task-notification>\n<task-id>bfbt5zeye</task-id>\n<status>stopped</status>\n</task-notification>",
          },
        ],
      },
      parent_tool_use_id: null,
      session_id: "sid-1",
    });
    const notificationResult = JSON.stringify({
      type: "result",
      subtype: "success",
      result: "notification done",
      session_id: "sid-1",
    });
    proc.stdout.emit("data", Buffer.from(`${notification}\n${notificationResult}\n`));

    let settled = false;
    void turn.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);

    const spawn = JSON.stringify({
      type: "assistant",
      parent_tool_use_id: null,
      session_id: "sid-1",
      message: {
        content: [
          {
            type: "tool_use",
            name: "Agent",
            id: "call_015a8cd",
            input: { description: "Trace vscode-acp memory wiring" },
          },
        ],
      },
    });
    const done = JSON.stringify({
      type: "result",
      subtype: "success",
      result: "waiting",
      session_id: "sid-1",
    });
    proc.stdout.emit("data", Buffer.from(`${spawn}\n${done}\n`));

    await expect(turn).resolves.toEqual({ text: "waiting", sessionId: "sid-1" });
    expect(events.some((event) => event.type === "tool_use" && event.toolName === "Agent")).toBe(true);
    expect(events.some((event) => event.type === "result" && event.text === "notification done")).toBe(false);
  });

  it("does not end the parent turn on a subagent result", async () => {
    const proc = createPersistentProc();
    mockSpawn.mockReturnValue(proc as any);

    const turn = runner.startSessionStreaming("/tmp/repo", "go", () => {}, "acp-1");
    const assistant = JSON.stringify({
      type: "assistant",
      parent_tool_use_id: null,
      session_id: "sid-1",
      message: { content: [{ type: "text", text: "starting" }] },
    });
    const sidechainResult = JSON.stringify({
      type: "result",
      subtype: "success",
      result: "subagent finished",
      session_id: "sid-1",
      parent_tool_use_id: "call_015a8cd",
    });
    proc.stdout.emit("data", Buffer.from(`${assistant}\n${sidechainResult}\n`));

    let settled = false;
    void turn.then(
      () => {
        settled = true;
      },
      () => {
        settled = true;
      },
    );
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(settled).toBe(false);

    proc.stdout.emit(
      "data",
      Buffer.from(
        JSON.stringify({
          type: "result",
          subtype: "success",
          result: "parent done",
          session_id: "sid-1",
        }) + "\n",
      ),
    );
    await expect(turn).resolves.toEqual({ text: "parent done", sessionId: "sid-1" });
  });
});
