import { describe, it, expect } from "vitest";
import * as fs from "node:fs";
import * as path from "node:path";
import { buildMidTurnModeChangeNotice, shouldAnnounceModeChange } from "../src/plan-mode-notice.js";

/**
 * Switching out of plan mode mid-turn was invisible to the running turn: the
 * <collaboration_mode> block is built once when the prompt starts (agent.ts reads
 * sessionCollaborationModes at that point), so the model kept planning, presented
 * the plan for review, and the user had to open a new turn and say "you are not in
 * plan mode now". Found in ericli first; this adapter had the same snapshot.
 */
describe("collaboration mode change mid-turn", () => {
  it("the leaving notice tells the model to implement in the same turn", () => {
    const notice = buildMidTurnModeChangeNotice(false);
    // Framed as a fresh <collaboration_mode> developer block (the authoritative
    // channel) — a bare user-level notice can't override the Plan-mode
    // instruction already in the running turn's context.
    expect(notice).toContain("<collaboration_mode># Collaboration Mode: Default");
    expect(notice).toContain("no longer active");
    expect(notice).toMatch(/THIS turn/);
    expect(notice).toMatch(/do not wait for another turn/);
    expect(notice).toContain("you may read and edit the codebase and execute commands again");
  });

  it("the entering notice stops changes", () => {
    const notice = buildMidTurnModeChangeNotice(true);
    expect(notice).toContain("<collaboration_mode># Collaboration Mode: Plan");
    expect(notice).toContain("no file edits");
  });

  it("only announces an actual change", () => {
    expect(shouldAnnounceModeChange("plan", "default")).toBe(true);
    expect(shouldAnnounceModeChange(undefined, "plan")).toBe(true);
    expect(shouldAnnounceModeChange("plan", "plan")).toBe(false);
  });

  it("cancels the running turn and re-prompts with the fresh block on a mid-turn flip", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const idx = src.indexOf("} else if (configId === COLLABORATION_MODE_CONFIG_ID) {");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 5000);
    // Same in-flight test handleSteer uses.
    expect(body).toContain("runner.isRunning(sessionId) || sessionPromptInFlight.has(sessionId)");
    // Genuine-flip gate (unchanged).
    expect(body).toContain("shouldAnnounceModeChange(previousCollabMode, nextCollab)");
    // The re-prompt notice describes the NEW mode (the one the turn will run in),
    // NOT the previous one. A Default→Plan flip passes true ("no more edits");
    // a Plan→Default flip passes false ("implement now"). (An earlier revision
    // passed prevWasPlan, inverting both directions — the model was told to
    // implement right after being put in plan mode.)
    expect(body).toContain("const newModeIsPlan = nextCollab === PLAN_COLLABORATION_MODE;");
    expect(body).toContain("await interruptForPlanMode(sessionId, newModeIsPlan, repromptModel, repromptEffort);");
    // It no longer pushes onto the steer queue for this path.
    expect(body).not.toContain("steerQueues");
  });

  it("the shared interruptForPlanMode core SIGINTs the running turn and re-prompts with the fresh block", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const idx = src.indexOf("const interruptForPlanMode = (");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 3000);
    // The fresh <collaboration_mode> block is built from the new-mode flag.
    expect(body).toContain("buildMidTurnModeChangeNotice(planActive)");
    // The interrupt: SIGINT the running turn…
    expect(body).toContain("runner.cancel(sessionId)");
    // …and chain a fresh executePrompt on the per-session lock with the notice.
    // (Passes its own generation so the cancel-latch scoping tracks it; an
    // empty promptImages array since a mode-change re-prompt has no images.)
    expect(body).toContain("executePrompt(sessionId, repromptText, repromptModel, repromptEffort, [], repromptGen)");
    expect(body).toContain("sessionPromptLocks.get(sessionId)");
    expect(body).toContain("promptGenCounter.get(sessionId)");
  });

  it("the model-initiated EnterPlanMode handler interrupts a running full-access turn", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const idx = src.indexOf('if (event.toolName === "EnterPlanMode") {');
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 2000);
    // Only interrupt on a GENUINE entry (enterPlanMode returns false for a
    // redundant re-call) and only while a turn is in flight.
    expect(body).toContain("void enterPlanMode(sessionId, connection, store, runner).then((entered) => {");
    expect(body).toContain("if (entered && (runner.isRunning(sessionId) || sessionPromptInFlight.has(sessionId))) {");
    // Re-prompt with the Plan block (newModeIsPlan = true) so read-only applies
    // NOW — the running turn was spawned with --dangerously-skip-permissions,
    // which overrides the CLI's plan gate.
    expect(body).toContain("void interruptForPlanMode(sessionId, true, repromptModel, repromptEffort);");
  });

  it("enterPlanMode reports whether a genuine flip into plan occurred", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const idx = src.indexOf("async function enterPlanMode(");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 2200);
    // Returns false for the redundant re-call (already in plan)…
    expect(body).toContain("return false;");
    // …and true after a genuine flip, so the caller can gate the interrupt.
    expect(body).toContain("return true;");
    // Declared to return a boolean, not void.
    expect(body).toContain("Promise<boolean>");
  });
});

/**
 * The "delivered" echo for a steered message must mean "sent to the model", not
 * "received".
 *
 * vscode-acp keeps a mid-turn message queued until the agent echoes it and
 * treats that echo as the receipt releasing the NEXT queued message. Echoing a
 * plain user_message_chunk on receipt let several interjections be dequeued
 * client-side while the turn was still busy; they then all arrived at once and
 * the model answered only the last.
 *
 * The two-stage fix (mirrors ericli): on receipt the agent emits a DISTINCT
 * `user_message_accepted` notification (visual-only — the client shows the
 * mid-turn bubble but keeps the entry queued), and emits the `user_message_chunk`
 * delivered-echo only when the message is actually drained to the model. The
 * accepted type decouples "shown" from "delivered" so the queue semantics stay
 * intact.
 */
describe("steer echo timing", () => {
  const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");

  it("receipt acks with user_message_accepted, NOT a delivered user_message_chunk", () => {
    const at = src.indexOf("steerQueues.get(sessionId)!.push(text);");
    expect(at).toBeGreaterThan(0);
    const handlerTail = src.slice(at, at + 1400);
    // Stage-1 visual ack IS emitted on receipt…
    expect(handlerTail).toContain("user_message_accepted");
    // …but the delivered-echo (the receipt that releases the next queued
    // message) is NOT — that only fires in drainSteerQueue.
    expect(handlerTail).not.toContain("user_message_chunk");
  });

  it("the drain echoes each message as it goes to the model", () => {
    const at = src.indexOf("const drainSteerQueue = async ()");
    expect(at).toBeGreaterThan(0);
    const body = src.slice(at, at + 2000);
    expect(body).toContain("sessionUpdate: 'user_message_chunk'");
    expect(body.indexOf("deliveredAny = true;")).toBeLessThan(body.indexOf("user_message_chunk"));
    expect(body).toContain(".catch(()");
  });
});

/**
 * The auto-compact lifecycle emits "completed" IMMEDIATELY once the CLI backend
 * finishes compacting (doCompact with skipCompleted=false), so the client clears
 * the panel and shows the "Context compacted" notice right away. The continuation
 * (re-sent original prompt) then streams into the fresh panel; the client's
 * pendingPostCompactChunks buffer absorbs any continuation chunks that arrive
 * inside the 400ms flush window.
 *
 * Because "completed" fires BEFORE the continuation starts, a failing
 * continuation (Claude CLI error, aborted turn) can no longer leave the
 * lifecycle stuck at "start" — the client already flushed. The finally block
 * only re-pushes the plan snapshot to the continuation's final board state.
 */
describe("immediate compaction completion", () => {
  it("auto-compact calls doCompact with skipCompleted=false (emit completed now)", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    // The auto-compact call site passes false for skipCompleted (8th arg).
    const at = src.indexOf("Emit \"completed\" IMMEDIATELY");
    expect(at).toBeGreaterThan(0);
    const body = src.slice(at, at + 1200);
    expect(body).toContain("false, // skipCompleted=false");
  });

  it("does NOT re-emit completed from the finally block (would double-fire)", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    // The auto-compact finally block must not call
    // emitCompactionLifecycle(..., "completed") — that would double-fire the
    // lifecycle and re-wipe the panel. doCompact already emitted it.
    const finallyIdx = src.indexOf("No \"completed\" re-emit here");
    expect(finallyIdx).toBeGreaterThan(0);
    const finallyBlock = src.slice(finallyIdx, finallyIdx + 600);
    expect(finallyBlock).not.toContain('emitCompactionLifecycle(connection, sessionId, compactResult.compactionId, "completed")');
  });

  it("emits a post-compaction usage_update so the ring drops immediately", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    const at = src.indexOf("Drop the pre-compaction usage");
    expect(at).toBeGreaterThan(0);
    const body = src.slice(at, at + 700);
    expect(body).toContain('sessionUpdate: "usage_update"');
    expect(body).toContain("sessionLastUsage.delete(sessionId)");
  });

  it("keeps the failure path for the non-deferred case", () => {
    const src = fs.readFileSync(path.resolve(process.cwd(), "src/agent.ts"), "utf8");
    expect(src).toContain('emitCompactionLifecycle(connection, sessionId, compactionId, "failed", detail)');
  });
});
