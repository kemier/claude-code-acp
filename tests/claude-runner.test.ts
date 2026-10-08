import { describe, it, expect, vi, beforeEach } from "vitest";
import { ClaudeRunner, type ClaudeResult, type StreamEvent } from "../src/claude-runner.js";
import { EventEmitter } from "node:events";
import type { ChildProcess } from "node:child_process";

// Mock child_process.spawn
vi.mock("node:child_process", () => ({
  spawn: vi.fn(),
}));

import { spawn } from "node:child_process";
const mockSpawn = vi.mocked(spawn);

function createMockProcess(
  stdoutData: string,
  exitCode: number = 0
): ChildProcess {
  const proc = new EventEmitter() as ChildProcess & EventEmitter;
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  (proc as any).stdout = stdout;
  (proc as any).stderr = stderr;
  (proc as any).pid = 12345;
  (proc as any).kill = vi.fn(() => true);

  // Simulate async data and exit
  setTimeout(() => {
    stdout.emit("data", Buffer.from(stdoutData));
    setTimeout(() => {
      proc.emit("close", exitCode);
    }, 5);
  }, 5);

  return proc;
}

function createStreamMockProcess(lines: string[], exitCode = 0): ChildProcess {
  const proc = new EventEmitter() as ChildProcess & EventEmitter;
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  (proc as any).stdout = stdout;
  (proc as any).stderr = stderr;
  (proc as any).pid = 12345;
  (proc as any).kill = vi.fn(() => true);

  setTimeout(() => {
    for (const line of lines) {
      stdout.emit("data", Buffer.from(line + "\n"));
    }
    setTimeout(() => proc.emit("close", exitCode), 5);
  }, 5);

  return proc;
}

// Mock process for the stdin stream-json path. Captures whatever JSON line is
// written to stdin so tests can assert the exact SDK user message format.
function createStdinMockProcess(
  lines: string[],
  exitCode = 0
): ChildProcess & { capturedStdin: () => string } {
  const proc = new EventEmitter() as ChildProcess & EventEmitter;
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  const stdin = new EventEmitter() as EventEmitter & { end: (s?: string) => void; write: (s: string) => void };
  let stdinData = "";
  stdin.write = (s: string) => {
    if (s) stdinData += s;
  };
  stdin.end = (s?: string) => {
    if (s) stdinData += s;
  };
  (proc as any).stdout = stdout;
  (proc as any).stderr = stderr;
  (proc as any).stdin = stdin;
  (proc as any).pid = 12345;
  (proc as any).kill = vi.fn(() => true);

  setTimeout(() => {
    for (const line of lines) {
      stdout.emit("data", Buffer.from(line + "\n"));
    }
    setTimeout(() => proc.emit("close", exitCode), 5);
  }, 5);

  return proc as ChildProcess & { capturedStdin: () => string };
}

describe("ClaudeRunner", () => {
  let runner: ClaudeRunner;

  beforeEach(() => {
    vi.clearAllMocks();
    runner = new ClaudeRunner();
  });

  describe("startSession", () => {
    it("should run claude -p and return result with session_id", async () => {
      const jsonOutput = JSON.stringify({
        result: "Hello! How can I help?",
        session_id: "abc-123-uuid",
      });
      mockSpawn.mockReturnValue(createMockProcess(jsonOutput));

      const result = await runner.startSession("/tmp/project", "Hello");

      expect(mockSpawn).toHaveBeenCalledWith(
        "claude",
        expect.arrayContaining([
          "-p",
          "Hello",
          "--output-format",
          "json",
        ]),
        expect.objectContaining({ cwd: "/tmp/project" })
      );
      expect(result.text).toBe("Hello! How can I help?");
      expect(result.sessionId).toBe("abc-123-uuid");
    });

    it("should throw on non-zero exit code", async () => {
      mockSpawn.mockReturnValue(createMockProcess("error", 1));

      await expect(
        runner.startSession("/tmp", "test")
      ).rejects.toThrow();
    });
  });

  describe("continueSession", () => {
    it("should run claude -p with --resume flag", async () => {
      const jsonOutput = JSON.stringify({
        result: "Continued response",
        session_id: "abc-123-uuid",
      });
      mockSpawn.mockReturnValue(createMockProcess(jsonOutput));

      const result = await runner.continueSession("abc-123-uuid", "next question");

      expect(mockSpawn).toHaveBeenCalledWith(
        "claude",
        expect.arrayContaining([
          "-p",
          "next question",
          "--resume",
          "abc-123-uuid",
          "--output-format",
          "json",
        ]),
        expect.any(Object)
      );
      expect(result.text).toBe("Continued response");
    });
  });

  describe("startSessionStreaming", () => {
    it("should emit text delta events from stream-json output", async () => {
      const lines = [
        JSON.stringify({
          type: "assistant",
          message: {
            content: [{ type: "text", text: "Hello world" }],
          },
          session_id: "sess-456",
        }),
      ];
      mockSpawn.mockReturnValue(createStreamMockProcess(lines));

      const events: StreamEvent[] = [];
      const result = await runner.startSessionStreaming(
        "/tmp",
        "hi",
        (event) => events.push(event)
      );

      expect(result.sessionId).toBe("sess-456");
    });

    it("should handle content_block_delta events", async () => {
      const lines = [
        JSON.stringify({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "chunk1" },
        }),
        JSON.stringify({
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "chunk2" },
        }),
        JSON.stringify({
          type: "result",
          result: "chunk1chunk2",
          session_id: "sess-789",
        }),
      ];
      mockSpawn.mockReturnValue(createStreamMockProcess(lines));

      const events: StreamEvent[] = [];
      const result = await runner.startSessionStreaming(
        "/tmp",
        "hi",
        (event) => events.push(event)
      );

      const textEvents = events.filter((e) => e.type === "text_delta");
      expect(textEvents).toHaveLength(2);
      expect(textEvents[0].text).toBe("chunk1");
      expect(textEvents[1].text).toBe("chunk2");
      expect(result.sessionId).toBe("sess-789");
    });
  });

  describe("startSessionStreamingWithContent", () => {
    it("should write an SDK user message JSON line to stdin", async () => {
      const outLines = [
        JSON.stringify({ type: "result", result: "OK", session_id: "sess-img" }),
      ];
      const proc = createStdinMockProcess(outLines);
      mockSpawn.mockReturnValue(proc as any);

      const stdinData: string[] = [];
      (proc as any).stdin.write = (s: string) => {
        if (s) stdinData.push(s);
      };
      (proc as any).stdin.end = (s?: string) => {
        if (s) stdinData.push(s);
      };

      const result = await runner.startSessionStreamingWithContent(
        "/tmp",
        [
          { type: "text", text: "what's this?" },
          {
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
          },
        ],
        () => {}
      );

      expect(result.sessionId).toBe("sess-img");
      // Must pass --input-format=stream-json and NOT -p <string>.
      const args = mockSpawn.mock.calls[0][1] as string[];
      expect(args).toContain("--input-format");
      expect(args).toContain("stream-json");
      expect(args).not.toContain("what's this?");

      const sent = stdinData.join("");
      const parsed = JSON.parse(sent.trim());
      expect(parsed.type).toBe("user");
      expect(parsed.message?.role).toBe("user");
      expect(parsed.message?.content).toEqual([
        { type: "text", text: "what's this?" },
        {
          type: "image",
          source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
        },
      ]);
      expect(parsed.parent_tool_use_id).toBeNull();
    });
  });

  describe("continueSessionStreamingWithContent", () => {
    it("should write --resume and an SDK user message to stdin", async () => {
      const outLines = [
        JSON.stringify({ type: "result", result: "Continue", session_id: "sess-resume" }),
      ];
      const proc = createStdinMockProcess(outLines);
      mockSpawn.mockReturnValue(proc as any);

      const stdinData: string[] = [];
      (proc as any).stdin.write = (s: string) => {
        if (s) stdinData.push(s);
      };
      (proc as any).stdin.end = (s?: string) => {
        if (s) stdinData.push(s);
      };

      const result = await runner.continueSessionStreamingWithContent(
        "abc-123",
        [{ type: "text", text: "more" }],
        () => {}
      );

      expect(result.sessionId).toBe("sess-resume");
      const args = mockSpawn.mock.calls[0][1] as string[];
      expect(args).toContain("--resume");
      expect(args).toContain("abc-123");
      expect(args).toContain("--input-format");
      expect(args).toContain("stream-json");

      const sent = stdinData.join("");
      const parsed = JSON.parse(sent.trim());
      expect(parsed.type).toBe("user");
      expect(parsed.message?.content).toEqual([{ type: "text", text: "more" }]);
    });
  });

  describe("cancel", () => {
    it("should kill a running process", () => {
      const proc = createMockProcess("{}", 0);
      mockSpawn.mockReturnValue(proc);

      // Start a session to register the process
      runner.startSession("/tmp", "test");
      runner.cancel("test-session");

      // cancel with unknown session should not throw
      expect(() => runner.cancel("unknown")).not.toThrow();
    });
  });
});
