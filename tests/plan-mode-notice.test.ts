import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { buildPlanModeChangeNotice, shouldAnnounceModeChange } from "../src/plan-mode-notice.js";

/**
 * Switching out of plan mode mid-turn was invisible to the running turn: the
 * <collaboration_mode> block is built once when the prompt starts (agent.ts reads
 * sessionCollaborationModes at that point), so the model kept planning, presented
 * the plan for review, and the user had to open a new turn and say "you are not in
 * plan mode now". Found in ericli first; this adapter had the same snapshot.
 */
describe("collaboration mode change mid-turn", () => {
  it("the leaving notice tells the model to implement in the same turn", () => {
    const notice = buildPlanModeChangeNotice(false);
    expect(notice).toContain("Plan mode is now OFF");
    expect(notice).toMatch(/THIS turn/);
    expect(notice).toMatch(/do not wait for another turn/);
  });

  it("the entering notice stops changes", () => {
    const notice = buildPlanModeChangeNotice(true);
    expect(notice).toContain("Plan mode is now ON");
    expect(notice).toContain("no file edits");
  });

  it("only announces an actual change", () => {
    expect(shouldAnnounceModeChange("plan", "default")).toBe(true);
    expect(shouldAnnounceModeChange(undefined, "plan")).toBe(true);
    expect(shouldAnnounceModeChange("plan", "plan")).toBe(false);
  });

  it("is queued only while a turn is in flight, on the same channel as steering", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const idx = src.indexOf("} else if (configId === COLLABORATION_MODE_CONFIG_ID) {");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 1600);
    // Same in-flight test handleSteer uses.
    expect(body).toContain("runner.isRunning(sessionId) || sessionPromptInFlight.has(sessionId)");
    expect(body).toContain("buildPlanModeChangeNotice(value === PLAN_COLLABORATION_MODE)");
    expect(body).toContain("shouldAnnounceModeChange(previousCollabMode, value)");
  });
});
