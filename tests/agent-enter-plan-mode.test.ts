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
    expect(src).toContain("enterPlanMode(sessionId, connection, store, runner)");
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

  it("tells the client so both pickers enter Plan", () => {
    const idx = src.indexOf("async function enterPlanMode(");
    const body = src.slice(idx, idx + 2000);
    // Permission picker: current_mode_update must carry a valid AVAILABLE_MODES
    // id ("read-only"), not the collaboration id "plan" (which isn't in
    // AVAILABLE_MODES, so availableModes.find() would miss and the picker would
    // silently stay on the previous mode). store.setMode persists it.
    expect(body).toContain('sessionUpdate: "current_mode_update"');
    expect(body).toContain('currentModeId: "read-only"');
    expect(body).toContain('store.setMode(sessionId, "read-only")');
    // Runner flag: the next claude -p spawn gets --permission-mode plan (CLI
    // hard-blocks mutating tools at its own layer, not just via the prompt block).
    expect(body).toContain('runner?.setSessionPermissionMode("read-only")');
    // Collaboration picker: config_option_update flips "collaboration_mode"
    // to "plan".
    expect(body).toContain('sessionUpdate: "config_option_update"');
    expect(body).toContain("sessionCollaborationModes.set(sessionId, PLAN_COLLABORATION_MODE)");
  });

  it("pass-through: buildExtraArgs emits --permission-mode from the runner field", () => {
    // Verify claude-runner.ts threads sessionPermissionMode → --permission-mode.
    const runnerSrc = readFileSync(resolve(process.cwd(), "src/claude-runner.ts"), "utf8");
    expect(runnerSrc).toContain("setSessionPermissionMode(mode: string | null)");
    expect(runnerSrc).toContain('this.sessionPermissionMode === "read-only" ? "plan" : "acceptEdits"');
    expect(runnerSrc).toContain('--permission-mode');
  });

  it("suppresses --dangerously-skip-permissions in read-only mode", () => {
    // --dangerously-skip-permissions bypasses ALL permission checks, INCLUDING
    // plan mode's read-only gate. Passing both --permission-mode plan AND
    // --dangerously-skip-permissions lets the CLI silently re-enable full edit +
    // Bash access (rm -rf, file writes) even though the picker shows "read-only".
    // buildExtraArgs MUST omit the skip flag when the session is in read-only.
    const runnerSrc = readFileSync(resolve(process.cwd(), "src/claude-runner.ts"), "utf8");
    const idx = runnerSrc.indexOf("--dangerously-skip-permissions");
    expect(idx).toBeGreaterThan(0);
    // The push must be gated on the mode NOT being read-only.
    expect(runnerSrc).toContain(
      'this.config.dangerouslySkipPermissions && this.sessionPermissionMode !== "read-only"'
    );
  });

  it("prompt handler syncs runner permission mode from store before each spawn", () => {
    // A user-initiated picker flip (setSessionMode) only writes store.mode — it
    // does NOT call runner.setSessionPermissionMode. The prompt handler must
    // sync it before any spawn, or a manual "Read-only" flip would not propagate
    // to the next claude -p process.
    const idx = src.indexOf("async prompt(params: PromptRequest)");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 1200);
    expect(body).toContain("runner.setSessionPermissionMode(store.getMode(sessionId) ?? DEFAULT_MODE)");
  });

  it("default mode prompts the model to use the picker for Plan mode", () => {
    // The Default-mode <collaboration_mode> block tells the model to ask the
    // user to switch to Plan mode via the picker — EnterPlanMode/ExitPlanMode
    // are NOT available in headless -p mode, so the model cannot self-enter.
    expect(src).toContain("# Collaboration Mode: Default");
    expect(src).toContain("switch to Plan mode via the Collaboration Mode picker");
  });
});
