import { describe, it, expect, vi } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import type { StreamEvent } from "../src/claude-runner.js";

/** Drive the private processStreamLine with raw stream-json objects. */
function feed(lines: unknown[]): StreamEvent[] {
  const runner = new ClaudeRunner();
  const events: StreamEvent[] = [];
  for (const line of lines) {
    (runner as any).processStreamLine(line, (e: StreamEvent) => events.push(e));
  }
  return events;
}

describe("tool_progress heartbeat", () => {
  // While a tool runs (a Bash poll with sleep/ssh loops) the CLI emits
  // periodic {"type":"tool_progress",...} heartbeats and NOTHING ELSE for
  // that tool — no deltas, no tool_result. Dropping them made a long tool
  // look exactly like a hung session (the "session hangs, Stop fixes it"
  // symptom). They ride at the ENVELOPE level, not inside stream_event.
  it("forwards an envelope-level tool_progress with id, name and elapsed", () => {
    const events = feed([
      {
        type: "tool_progress",
        tool_use_id: "toolu_bash1",
        tool_name: "Bash",
        elapsed_time_seconds: 61,
        task_id: "t1",
        uuid: "u1",
        session_id: "sid",
        parent_tool_use_id: null,
      },
    ]);
    const tp = events.find((e) => e.type === "tool_progress");
    expect(tp).toBeDefined();
    expect(tp?.toolUseId).toBe("toolu_bash1");
    expect(tp?.toolName).toBe("Bash");
    expect(tp?.toolElapsedSeconds).toBe(61);
    expect(tp?.toolSidechain).toBe(false);
  });

  it("flags a sidechain heartbeat so it is not mapped onto the parent's tools", () => {
    const events = feed([
      {
        type: "tool_progress",
        tool_use_id: "toolu_sub",
        tool_name: "Read",
        elapsed_time_seconds: 3,
        parent_tool_use_id: "toolu_task",
      },
    ]);
    expect(events.find((e) => e.type === "tool_progress")?.toolSidechain).toBe(true);
  });

  it("tolerates missing optional fields", () => {
    const events = feed([{ type: "tool_progress", tool_use_id: "toolu_x" }]);
    const tp = events.find((e) => e.type === "tool_progress");
    expect(tp?.toolUseId).toBe("toolu_x");
    expect(tp?.toolElapsedSeconds).toBeUndefined();
    expect(tp?.toolSidechain).toBe(false);
  });
});

describe("result event sidechain marker", () => {
  it("marks a subagent's own result line", () => {
    const events = feed([
      {
        type: "result",
        subtype: "success",
        result: "ok",
        session_id: "sid",
        usage: {},
        parent_tool_use_id: "toolu_task",
      },
    ]);
    expect(events.find((e) => e.type === "result")?.isSidechain).toBe(true);
  });

  it("marks the main conversation's result", () => {
    const events = feed([
      { type: "result", subtype: "success", result: "ok", session_id: "sid", usage: {} },
    ]);
    expect(events.find((e) => e.type === "result")?.isSidechain).toBe(false);
  });
});

describe("cancel kills the whole process group", () => {
  // The streaming spawns are `detached: true`, so the claude child leads its
  // own process group. Signaling -pid reaps the turn's tool descendants
  // (bash → sleep/ssh trees) that would otherwise outlive the cancelled turn
  // as orphans. Falls back to the child-only kill when the group is gone.
  function cancelledRunner(): ClaudeRunner {
    const runner = new ClaudeRunner();
    const proc = {
      pid: 4242,
      exitCode: null,
      killed: [] as string[],
      kill: (sig: string) => {
        proc.killed.push("child:" + sig);
        return true;
      },
    };
    (runner as any).runningProcesses.set("t1", proc);
    return runner;
  }

  it("SIGINTs the process group first", () => {
    const runner = cancelledRunner();
    const killSpy = vi.spyOn(process, "kill").mockImplementation(() => true);
    try {
      runner.cancel("t1");
      expect(killSpy).toHaveBeenCalledWith(-4242, "SIGINT");
    } finally {
      killSpy.mockRestore();
      (runner as any).cancelEscalationTimers.forEach((t: any) => clearTimeout(t));
      (runner as any).cancelEscalationTimers.clear();
    }
  });

  it("falls back to the child when the group is already gone", () => {
    const runner = cancelledRunner();
    const killSpy = vi.spyOn(process, "kill").mockImplementation(() => {
      throw Object.assign(new Error("no such process"), { code: "ESRCH" });
    });
    try {
      runner.cancel("t1");
      const proc = (runner as any).runningProcesses.get("t1");
      expect(proc.killed).toEqual(["child:SIGINT"]);
    } finally {
      killSpy.mockRestore();
      (runner as any).cancelEscalationTimers.forEach((t: any) => clearTimeout(t));
      (runner as any).cancelEscalationTimers.clear();
    }
  });

  it("escalates to SIGTERM on the group after the grace window", async () => {
    vi.useFakeTimers();
    const runner = cancelledRunner();
    const killSpy = vi.spyOn(process, "kill").mockImplementation(() => true);
    try {
      runner.cancel("t1");
      vi.advanceTimersByTime(3000);
      expect(killSpy).toHaveBeenLastCalledWith(-4242, "SIGTERM");
    } finally {
      killSpy.mockRestore();
      vi.useRealTimers();
      (runner as any).cancelEscalationTimers.forEach((t: any) => clearTimeout(t));
      (runner as any).cancelEscalationTimers.clear();
    }
  });
});
