import { describe, it, expect } from "vitest";
import * as fs from "fs";

/**
 * Claude Code's task list must not be labelled a plan.
 *
 * TaskCreate / TaskUpdate / TaskList manage a TODO list — this adapter's own note
 * says so ("Claude Code's TaskCreate/TaskUpdate/TaskList tools manage a todo
 * list"). They are mirrored into ACP as `sessionUpdate: "plan"` because that is
 * where the client's plan bar lives, and the client titles that bar "Plan" by
 * default, so a checklist read as a strategy.
 *
 * The frame carries no kind of its own, so the adapter now states it in
 * _meta.acp.plan — the adapter-neutral hint shared across adapters. A client that
 * ignores _meta renders exactly as before, so this is additive.
 *
 * Verified while writing this: the plan bar has ALWAYS rendered a checklist
 * (per-entry ✅/🔄/⬜ from entry.status), for every agent on that path. So the
 * rendering was never wrong — only the heading was.
 */

const src = fs.readFileSync(new URL("../src/agent.ts", import.meta.url), "utf8");

/**
 * The body of the plan snapshot — the single object literal built in
 * handleTaskTool and delivered either inline (live) or buffered on `out`
 * (replay / post-compaction re-push). Anchored on the `out.snapshot = snapshot`
 * branch, not on the object construction, so the assertions below cover the
 * _meta hint that travels with whichever path delivers it.
 */
function deliverCall(): string {
  const at = src.indexOf('sessionUpdate: "plan"');
  expect(at).toBeGreaterThan(0);
  const start = src.lastIndexOf("const snapshot = {", at);
  expect(start).toBeGreaterThan(0);
  return src.slice(start, src.indexOf("} as any;", at) + 9);
}

describe("the task list declares itself a todo list", () => {
  const call = deliverCall();

  it("carries the neutral kind hint", () => {
    expect(call).toContain('_meta: { acp: { plan: { kind: "todo", title: "Todos" } } }');
  });

  it("still uses the shared plan channel", () => {
    // The point is to keep reusing the plan bar, not to add a second surface.
    expect(call).toContain('sessionUpdate: "plan"');
    expect(call).toContain("entries,");
  });

  it("there is exactly one emit site, so live and replay agree", () => {
    // A second emitter would drift: one bar titled Todos, another Plan. The live
    // path and the session/load replay path share one `deliver`.
    expect((src.match(/sessionUpdate: "plan"/g) ?? []).length).toBe(1);
    expect((src.match(/const deliver = \(snapshot: any\) =>/g) ?? []).length).toBe(1);
  });

  it("entries keep their shape, so the change is additive", () => {
    // Built just ABOVE the deliver call, so assert on the construction rather
    // than on the delivered object literal.
    const at = src.indexOf("const entries = Array.from(todos.entries())");
    expect(at).toBeGreaterThan(0);
    const build = src.slice(at, src.indexOf("}));", at) + 4);
    expect(build).toMatch(/content: t\.content/);
    expect(build).toMatch(/status: t\.status/);
    // And it must still precede the emit, or nothing would be sent.
    expect(at).toBeLessThan(src.indexOf('sessionUpdate: "plan"'));
  });

  it("an empty todo set still emits nothing", () => {
    // Pre-existing behaviour worth not breaking: an empty snapshot used to flash
    // an empty bar when the model updated tasks it never created.
    const at = src.indexOf("if (todos.size === 0) {");
    expect(at).toBeGreaterThan(0);
    expect(at).toBeLessThan(src.indexOf('sessionUpdate: "plan"'));
  });
});
