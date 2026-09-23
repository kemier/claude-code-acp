import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

/**
 * Claude Code signals plan completion with its native ExitPlanMode tool. Before
 * this was mapped, the adapter ignored it: the session stayed in Plan mode, the
 * injected <collaboration_mode> block kept telling the model that an execution
 * request is a request to PLAN the execution, and a follow-up "implement it" came
 * back as code in the chat instead of edits on disk.
 *
 * setSessionMode only ECHOES a client-initiated switch, so leavePlanMode is the
 * only agent-initiated exit — assert it exists, is called, and notifies the client.
 */
describe("ExitPlanMode", () => {
  const src = readFileSync(resolve(process.cwd(), "src/agent.ts"), "utf8");

  it("is mapped to an agent-initiated plan-mode exit", () => {
    expect(src).toContain('event.toolName === "ExitPlanMode"');
    expect(src).toContain("leavePlanMode(sessionId, connection, store, runner)");
  });

  it("clears the stored collaboration mode", () => {
    const idx = src.indexOf("async function leavePlanMode(");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 1600);
    expect(body).toContain("sessionCollaborationModes.set(sessionId, DEFAULT_COLLABORATION_MODE)");
    // A no-op when not in plan mode, so an unexpected tool call cannot disturb
    // the picker.
    expect(body).toContain("!== PLAN_COLLABORATION_MODE");
  });

  it("tells the client so both pickers leave Plan", () => {
    const idx = src.indexOf("async function leavePlanMode(");
    const body = src.slice(idx, idx + 2000);
    // Permission picker: restore full access ("agent") so the model can
    // implement the approved plan. current_mode_update carries a valid
    // AVAILABLE_MODES id.
    expect(body).toContain('sessionUpdate: "current_mode_update"');
    expect(body).toContain("currentModeId: DEFAULT_MODE");
    expect(body).toContain("store.setMode(sessionId, DEFAULT_MODE)");
    // Runner flag: the next claude -p spawn gets --permission-mode acceptEdits
    // (full access restored), not plan.
    expect(body).toContain("runner?.setSessionPermissionMode(DEFAULT_MODE)");
    // Collaboration picker: flip "collaboration_mode" back to "default".
    expect(body).toContain('sessionUpdate: "config_option_update"');
    expect(body).toContain("sessionCollaborationModes.set(sessionId, DEFAULT_COLLABORATION_MODE)");
  });
});
