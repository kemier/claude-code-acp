/**
 * Notice delivered into a RUNNING turn when the user flips collaboration mode.
 *
 * The <collaboration_mode> block is built once, when the prompt starts, from
 * sessionCollaborationModes. Switching plan -> default mid-turn was therefore
 * invisible to the turn already in flight: the model kept following Plan mode,
 * finished by presenting the plan for review, and the work waited for the user
 * to open another turn and say "you are not in plan mode now".
 *
 * Same defect and same remedy as ericli, which rides its tool-iteration boundary.
 * Here the carrier is the existing steer queue, drained mid-turn.
 *
 * Plan mode in this adapter is prompt-level only — there is no per-query
 * permissionMode gate — so telling the model is enough for it to start editing.
 */

/**
 * Mid-turn announcement of a collaboration-mode flip, framed as a fresh
 * `<collaboration_mode>` developer block — the SAME authoritative mechanism that
 * sets the mode at turn start.
 *
 * A bare user-level notice ("Plan mode is now OFF") is NOT enough: the running
 * turn's original prompt still carries the previous mode's `<collaboration_mode>`
 * developer instruction ("In Plan mode you MUST NOT make changes") deep in its
 * context, and a short user message cannot reliably override a strong, explicit
 * developer instruction already in the transcript. The model "can't perceive"
 * the switch. Re-asserting the canonical block (the developer channel, which
 * outranks a user message) is what actually flips the model's behavior.
 *
 * @param planActive the mode the turn should now be running in
 */
export function buildMidTurnModeChangeNotice(planActive: boolean): string {
  return planActive
    ? `<collaboration_mode># Collaboration Mode: Plan\n\nYour active mode was just switched to Plan mid-turn by the user. All prior instructions for other modes are no longer active. From here on: investigate read-only and present a plan — no file edits, no patches, no codegen.\n</collaboration_mode>`
    : `<collaboration_mode># Collaboration Mode: Default\n\nYour active mode was just switched to Default mid-turn by the user. All prior Plan-mode instructions are no longer active: you may read and edit the codebase and execute commands again. Do not stop to present the plan for approval and do not wait for another turn — implement it now, in THIS turn, using the tools. If you already presented a plan, start executing it.\n</collaboration_mode>`;
}

/** True when a collaboration-mode change should be announced to a running turn. */
export function shouldAnnounceModeChange(previous: string | undefined, next: string): boolean {
  return (previous ?? "") !== next;
}
