/**
 * Inline images attached to a prompt (ACP `image` content blocks) are sent
 * to the Claude CLI as `stream-json` content blocks alongside the prompt
 * text, but their payload was invisible to `TurnContextEstimator` — which
 * only consumed strings via `addText`. On a multi-image prompt the pre-turn
 * compaction guard (`getEstimatedTokensWithPending`) therefore under-counted
 * by up to ~1500 tokens per image. The guard didn't trip, the request
 * reached the backend oversized, and the ring landed at 100% on the first
 * API response (the real `prompt_tokens` counted every image the guard had
 * missed).
 *
 * Reported by the user 2026-11-06 on claude-code-acp local/dev:
 *   "seems it because i sent too many picture in one prompt caused this"
 *
 * These tests pin the per-image accounting in `addImages`, verify the
 * lifecycle integration with `setRealTokens`/`reset`, and make sure
 * non-image blocks are ignored so this helper doesn't double-count text.
 */

import { describe, it, expect } from "vitest";
import { TurnContextEstimator } from "../src/agent.js";

const TOKENS_PER_IMAGE = 1500;

function imageBlock(size = 1024) {
  return {
    type: "image",
    source: { type: "base64", media_type: "image/png", data: "A".repeat(size) },
  };
}

describe("TurnContextEstimator.addImages", () => {
  it("adds TOKENS_PER_IMAGE per image block to the pre-real estimate", () => {
    const e = new TurnContextEstimator();
    const before = e.getEstimatedTokens();
    e.addImages([imageBlock(), imageBlock(), imageBlock()]);
    expect(e.getEstimatedTokens() - before).toBe(3 * TOKENS_PER_IMAGE);
  });

  it("ignores non-image blocks so text isn't double-counted (addText handles that)", () => {
    const e = new TurnContextEstimator();
    const before = e.getEstimatedTokens();
    e.addImages([{ type: "text" } as any, { type: "tool_use" } as any, imageBlock()]);
    expect(e.getEstimatedTokens() - before).toBe(TOKENS_PER_IMAGE);
  });

  it("accepts an empty / undefined list without changing state", () => {
    const e = new TurnContextEstimator();
    const before = e.getEstimatedTokens();
    e.addImages([]);
    e.addImages(undefined);
    expect(e.getEstimatedTokens()).toBe(before);
  });

  it("surfaces pending image tokens in getEstimatedTokensWithPending AFTER a real count arrives", () => {
    const e = new TurnContextEstimator();
    // Simulate a turn: real count from the first API response, then user
    // sends a new prompt with 4 images for the next turn (not yet sent).
    e.setRealTokens(100_000);
    expect(e.getEstimatedTokens()).toBe(100_000);

    e.addImages([imageBlock(), imageBlock(), imageBlock(), imageBlock()]);

    // getEstimatedTokens is verbatim-real — unaffected by pending images.
    expect(e.getEstimatedTokens()).toBe(100_000);

    // getEstimatedTokensWithPending adds the pending images so the pre-turn
    // compaction guard sees what the backend is about to receive.
    expect(e.getEstimatedTokensWithPending()).toBe(100_000 + 4 * TOKENS_PER_IMAGE);
  });

  it("setRealTokens clears the pending-image portion (same lifecycle as bytesSinceReal)", () => {
    const e = new TurnContextEstimator();
    e.addImages([imageBlock(), imageBlock()]);
    e.addText("hello");
    // Real count arrives — it already includes everything Claude CLI saw,
    // so sinceReal counters reset.
    e.setRealTokens(42_000);
    expect(e.getEstimatedTokens()).toBe(42_000);
    expect(e.getEstimatedTokensWithPending()).toBe(42_000);
  });

  it("reset wipes both the pre-real and sinceReal image token portions", () => {
    const e = new TurnContextEstimator();
    const baseline = new TurnContextEstimator().getEstimatedTokens();
    e.addImages([imageBlock(), imageBlock(), imageBlock()]);
    e.setRealTokens(50_000);
    e.addImages([imageBlock()]); // bumps imageTokensSinceReal only
    e.reset();
    // Reset drops contentBytes / realTokens / imageTokens* back to zero —
    // only the fixed CLAUDE_OVERHEAD_BYTES baseline remains (the comment on
    // reset() documents keeping the fixed overhead on purpose).
    expect(e.getEstimatedTokens()).toBe(baseline);
    expect(e.getEstimatedTokensWithPending()).toBe(baseline);
  });

  it("without real tokens, pre-real estimate stacks addText + addImages", () => {
    const e = new TurnContextEstimator();
    const before = e.getEstimatedTokens();
    e.addText("x".repeat(3000)); // ~1000 tokens @ chars/3
    e.addImages([imageBlock(), imageBlock()]); // 3000 tokens flat
    const delta = e.getEstimatedTokens() - before;
    expect(delta).toBeGreaterThanOrEqual(3000 + 1000);
    // And the pre-turn guard sees the same thing when no real count exists.
    expect(e.getEstimatedTokensWithPending()).toBe(e.getEstimatedTokens());
  });

  it("10-image prompt no longer under-counts by zero (the production symptom)", () => {
    // The field repro: user attaches 10 images in one prompt. Pre-fix the
    // guard saw the ~200 chars of text and said "~70 tokens added". The
    // real prompt_tokens landed at +15000 and the ring jumped to 100%.
    const e = new TurnContextEstimator();
    e.setRealTokens(170_000); // simulate approaching-threshold session
    const before = e.getEstimatedTokensWithPending();

    e.addText("here are the screenshots");
    const tenImages = Array.from({ length: 10 }, () => imageBlock());
    e.addImages(tenImages);

    const after = e.getEstimatedTokensWithPending();
    // Pre-fix: delta ≈ 8 tokens (text only). Post-fix: delta >= 15000 tokens.
    // This is what makes the pre-turn guard trip BEFORE sending oversized.
    expect(after - before).toBeGreaterThanOrEqual(10 * TOKENS_PER_IMAGE);
  });
});
