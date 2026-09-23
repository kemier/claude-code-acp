import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * The Claude CLI exposes a native EnterPlanMode tool the model calls when it
 * decides a task needs planning before acting (mirrors ericli's auto
 * enter_plan_mode). Before this was mapped, the adapter only handled the EXIT
 * side (ExitPlanMode → leavePlanMode): a model that self-entered plan mode left
 * the picker reading "Default" the whole time, so the user had no visibility
 * that planning was underway.
 *
 * enterPlanMode is the agent-initiated enter — assert it exists, is wired to the
 * EnterPlanMode tool, sets the stored collaboration mode to plan, and notifies
 * the client so the picker flips to "Plan" live.
 */
describe("EnterPlanMode", () => {
  const src = readFileSync(resolve(process.cwd(), "src/agent.ts"), "utf8");

  it("is mapped to an agent-initiated plan-mode enter", () => {
    expect(src).toContain('event.toolName === "EnterPlanMode"');
    expect(src).toContain("enterPlanMode(sessionId, connection, store)");
  });

  it("sets the stored collaboration mode to plan", () => {
    const idx = src.indexOf("async function enterPlanMode(");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 1600);
    expect(body).toContain("sessionCollaborationModes.set(sessionId, PLAN_COLLABORATION_MODE)");
    // A no-op when already in plan mode, so a repeated tool call cannot emit a
    // redundant picker update.
    expect(body).toContain("=== PLAN_COLLABORATION_MODE");
  });

  it("tells the client so the picker enters Plan", () => {
    const idx = src.indexOf("async function enterPlanMode(");
    const body = src.slice(idx, idx + 1600);
    expect(body).toContain('sessionUpdate: "current_mode_update"');
    expect(body).toContain("currentModeId: PLAN_COLLABORATION_MODE");
    expect(body).toContain('sessionUpdate: "config_option_update"');
  });

  it("default mode prompts the model it may self-enter Plan mode", () => {
    // The Default-mode <collaboration_mode> block tells the model to call
    // EnterPlanMode on complex tasks — the auto-entry trigger (mirrors ericli's
    // "USE WHEN: complex multi-step tasks").
    expect(src).toContain("# Collaboration Mode: Default");
    expect(src).toContain("call EnterPlanMode to switch to Plan mode");
    expect(src).toContain("call ExitPlanMode to return to Default mode");
  });
});
