import { describe, it, expect } from "vitest";
import { extractTaskNotificationSummary } from "../src/agent.js";

/**
 * A finished subagent's result must leave this adapter, or it is lost.
 *
 * Claude Code delivers a completed subagent as a `user` text message carrying a
 * <task-notification> block with a <summary> — NOT as a tool_result reusing the
 * tool_use_id. This adapter already parsed that block, but only to match
 * spawn→completion by description, and then discarded the text.
 *
 * Nothing else carried it: subagent content never enters the main conversation's
 * prompt (see the note on getEstimatedTokens), and a client's memory capture
 * records tool_call TITLES plus the main assistant text — never tool results. So
 * the work was unrecoverable once the turn ended, and could never become memory
 * or a skill.
 *
 * The summary is now forwarded over `_acp/subagent/findings`, the same
 * adapter-neutral method the other adapters send, so a client needs one implementation
 * rather than one per adapter.
 */

describe("extractTaskNotificationSummary", () => {
  const desc = "Trace interrupt message wiring";

  it("prefers the <summary> block", () => {
    const notif = `<task-notification>
Agent "${desc}" completed
<summary>Interrupt flows through Op::Interrupt in turn_processor.rs:214</summary>
</task-notification>`;
    const out = extractTaskNotificationSummary(notif, desc);
    expect(out).not.toBeNull();
    expect(out!.summary).toBe("Interrupt flows through Op::Interrupt in turn_processor.rs:214");
    expect(out!.stopped).toBe(false);
  });

  it("marks a stopped subagent, keeping its partial work", () => {
    const notif = `<task-notification>
background agent "${desc}" ... [stopped]
<summary>Got through 3 of 40 files before stopping</summary>
</task-notification>`;
    const out = extractTaskNotificationSummary(notif, desc)!;
    expect(out.stopped).toBe(true);
    // Partial work is still worth remembering — it is what stops a later
    // session repeating the same dead end.
    expect(out.summary).toContain("3 of 40");
  });

  it("falls back to the body when there is no <summary>", () => {
    const notif = `<task-notification>
Agent "${desc}" completed
The handler lives in CodexAcpServer.ts and calls turnInterrupt().
</task-notification>`;
    const out = extractTaskNotificationSummary(notif, desc)!;
    expect(out.summary).toContain("CodexAcpServer.ts");
    // The agent-name line carries no findings and duplicates the task.
    expect(out.summary).not.toContain("completed");
    expect(out.summary).not.toContain("<task-notification>");
  });

  it("does not report a finding that has no content", () => {
    // An empty notification must not become a memory claiming a subagent found
    // something.
    expect(extractTaskNotificationSummary(`<task-notification>\nAgent "${desc}" completed\n</task-notification>`, desc)).toBeNull();
    expect(extractTaskNotificationSummary("", desc)).toBeNull();
    expect(extractTaskNotificationSummary("<task-notification></task-notification>", desc)).toBeNull();
  });

  it("strips the description, which is sent separately as the task", () => {
    const notif = `<task-notification><summary>${desc}: found it in main.ts</summary></task-notification>`;
    const out = extractTaskNotificationSummary(notif, desc)!;
    expect(out.summary).not.toContain(desc);
    expect(out.summary).toContain("found it in main.ts");
  });

  it("bounds a runaway summary instead of shipping a transcript", () => {
    const huge = "x".repeat(20000);
    const out = extractTaskNotificationSummary(
      `<task-notification><summary>${huge}</summary></task-notification>`,
      desc,
    )!;
    expect(out.summary.length).toBeLessThan(8200);
    expect(out.summary).toContain("(truncated)");
  });

  it("handles several notifications in one message independently", () => {
    // A single user message can carry multiple blocks when subagents finish
    // together; each is matched by its own description.
    const both = `<task-notification><summary>alpha result</summary></task-notification>
<task-notification><summary>beta result</summary></task-notification>`;
    const out = extractTaskNotificationSummary(both, "")!;
    expect(out.summary).toContain("alpha result");
  });
});

describe("the emitter", () => {
  it("uses the adapter-neutral method and does not block on the client", async () => {
    const fs = await import("fs");
    const src = fs.readFileSync(new URL("../src/agent.ts", import.meta.url), "utf8");
    expect(src).toContain('const SUBAGENT_FINDINGS_METHOD = "_acp/subagent/findings"');
    const at = src.indexOf("extNotification(SUBAGENT_FINDINGS_METHOD");
    expect(at).toBeGreaterThan(0);
    const around = src.slice(at, at + 500);
    // A client that ignores the method must not disturb a finished subagent.
    expect(around).toContain(".catch(() => {})");
    // The task must travel with the finding, or the summary answers no question.
    expect(around).toContain("task: info.description");
    // And nothing is sent when there is no content to report.
    expect(src.slice(Math.max(0, at - 400), at)).toContain("if (findings)");
  });
});
