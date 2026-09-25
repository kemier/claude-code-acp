// Per-model effort (reasoning) tiers.
//
// The Effort picker is filtered by getEffortsForModel(modelId), where modelId
// is the raw /v1/models id passed through verbatim by the picker. Two failure
// modes have bitten this lookup:
//
//  1. Case sensitivity — the table mixes cases ("deepseek-v4-flash" vs the
//     live id "DeepSeek-V4.1-Flash") and a raw hasOwnProperty miss silently
//     dropped xhigh, falling back to DEFAULT_EFFORTS (low/medium/high).
//  2. Version drift — the backend was upgraded from "deepseek-v4-flash"
//     (alias) to "DeepSeek-V4.1-Flash"; a literal table key never matches the
//     new slug. ericli avoids both via a /deepseek.*flash/i regex
//     (supportedReasoningEffortsForModel in model-fallback.ts); this test
//     pins that claude-code-acp does the same.

import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

describe("getEffortsForModel — deepseek-flash version/case tolerance", () => {
  const src = readFileSync(resolve(process.cwd(), "src/agent.ts"), "utf8");

  // Slice the getEffortsForModel body out of the source so the assertions
  // below scope to that function (not a stray mention elsewhere).
  const fnIdx = src.indexOf("function getEffortsForModel(");
  const fn = src.slice(fnIdx, fnIdx + 1400);

  it("lowercases the incoming model id before the table lookup (case-insensitive)", () => {
    // The lookup must derive a `lower` from modelId and probe a lower-cased
    // table — a raw hasOwnProperty(MODEL_EFFORTS, modelId) is case-fragile.
    expect(fn).toContain(".toLowerCase()");
    expect(fn).toContain("MODEL_EFFORTS_LOWER");
  });

  it("matches the deepseek-flash family by regex, not a literal slug", () => {
    // Version-tolerant: catches "deepseek-v4-flash", "DeepSeek-V4.1-Flash",
    // and any future deepseek+flash id. Mirrors ericli's /deepseek.*flash/i.
    expect(fn).toMatch(/\/deepseek\.\*flash\//);
  });

  it("advertises the full ladder (including xhigh) for the deepseek-flash family", () => {
    // The regex branch must return low/medium/high/xhigh — the tier that was
    // missing for DeepSeek-V4.1-Flash before the fix.
    const xhighIdx = fn.indexOf("xhigh");
    expect(xhighIdx).toBeGreaterThan(0);
    const ladder = fn.slice(fn.lastIndexOf("[", xhighIdx), fn.indexOf("]", xhighIdx) + 1);
    expect(ladder).toContain("low");
    expect(ladder).toContain("medium");
    expect(ladder).toContain("high");
    expect(ladder).toContain("xhigh");
  });

  it("still falls back to DEFAULT_EFFORTS for unknown models", () => {
    expect(fn).toContain("DEFAULT_EFFORTS");
  });
});
