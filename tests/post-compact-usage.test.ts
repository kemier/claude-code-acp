// Regression: manual /compact left the context ring showing the PRE-compaction
// value. The ring is fed only by usage_update frames; auto-compact sends one
// after its continuation turn, but manual /compact has no continuation — the
// last frame the client saw was the pre-compaction usage, so the ring never
// dropped even though the "compacted" notice appeared.
//
// Fix: doCompact emits a post-compaction usage_update (from the reset+reseeded
// estimator, which now holds only the summary) right after the "completed"
// lifecycle frame, covering manual /compact, the pre-turn guard, and the
// post-turn auto-compact alike.

import { describe, it, expect } from "vitest";
import { createRequire } from "node:module";
import { resolve } from "node:path";

const require = createRequire(__filename);
const readTextFile: (p: string) => string = require("./src-file-reader.cjs").readTextFile;

describe("post-compaction usage_update repaints the context ring", () => {
  it("doCompact emits usage_update after the completed lifecycle frame", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const idx = src.indexOf('await emitCompactionLifecycle(connection, sessionId, compactionId, "completed");');
    expect(idx).toBeGreaterThan(0);
    const after = src.slice(idx, idx + 1600);
    expect(after).toContain('sessionUpdate: "usage_update"');
    expect(after).toContain("estimator.getEstimatedTokens()");
    expect(after).toContain("postWindow");
  });

  it("the emission sits inside the !skipCompleted guard (deferred paths do not double-fire)", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const fnAt = src.indexOf("async function doCompact(");
    expect(fnAt).toBeGreaterThan(0);
    const body = src.slice(fnAt, fnAt + 8000);
    const completedAt = body.indexOf('emitCompactionLifecycle(connection, sessionId, compactionId, "completed")');
    expect(completedAt).toBeGreaterThan(0);
    const usageAt = body.indexOf('sessionUpdate: "usage_update"', completedAt);
    expect(usageAt).toBeGreaterThan(completedAt);
    // The guard opens before the completed emission and the usage frame is
    // emitted before the guard closes — i.e. both are inside the same block.
    const guard = body.slice(Math.max(0, completedAt - 200), completedAt);
    expect(guard).toContain("if (!skipCompleted) {");
    // No nested skipCompleted check between the two emissions — they share the
    // one guard, so a deferred path skips both frames together.
    const between = body.slice(completedAt, usageAt);
    expect(between.includes("skipCompleted")).toBe(false);
  });

  it("the frame carries the fresh window, not a stale one", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const idx = src.indexOf("const postWindow = getContextWindowForModel(resolveSessionModel(store, sessionId, modelOverride));");
    expect(idx).toBeGreaterThan(0);
    expect(src).toContain("used: Math.min(estimator.getEstimatedTokens(), postWindow)");
    expect(src).toContain("size: postWindow");
  });
});
