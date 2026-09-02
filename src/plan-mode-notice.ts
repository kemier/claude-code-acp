/**
 * Notice delivered into a RUNNING turn when the user flips collaboration mode.
 *
 * The <collaboration_mode> block is built once, when the prompt starts, from
 * sessionCollaborationModes. Switching plan -> default mid-turn was therefore
 * invisible to the turn already in flight: the model kept following Plan mode,
 * finished by presenting the plan for review, and the work waited for the user to
 * open another turn and say "you are not in plan mode now".
 *
 * Same defect and same remedy as ericli, which rides its tool-iteration boundary.
 * Here the carrier is the existing steer queue, drained mid-turn.
 *
 * Plan mode in this adapter is prompt-level only — there is no per-query
 * permissionMode gate — so telling the model is enough for it to start editing.
 */
export function buildPlanModeChangeNotice(planActive: boolean): string {
  if (planActive) {
    return (
      "[Mode change]: Plan mode is now ON. Stop making changes from here on: no file edits, " +
      "no patches, no codegen. Continue with read-only investigation and present a plan."
    );
  }
  return (
    "[Mode change]: Plan mode is now OFF and you have write access. Do not stop to present " +
    "the plan for approval and do not wait for another turn — implement it now, in THIS turn, " +
    "using the tools. If you already presented a plan, start executing it."
  );
}

/** True when a collaboration-mode change should be announced to a running turn. */
export function shouldAnnounceModeChange(previous: string | undefined, next: string): boolean {
  return (previous ?? "") !== next;
}
