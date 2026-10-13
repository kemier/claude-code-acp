// Verifies the ACP layer's auto-compact threshold arithmetic.
//
// Both layers can compact. If their arithmetic disagrees, whichever fires first
// wins and the other silently never runs — so tuning compaction via the
// documented environment variables would have unpredictable effect. These tests
// pin the ACP side's formula:
//
//   effectiveWindow = min(window, AUTO_COMPACT_WINDOW)        (NO summary reserve)
//   threshold       = effectiveWindow - 13000
//   with PCT=N        threshold = min(floor(effectiveWindow * N/100), threshold)
//
// NOTE: the ACP layer intentionally does NOT subtract a summary output reserve
// (it used to subtract min(maxOutput, 20000), mirroring the Claude Code CLI).
// Removing it compacts later (~codex-acp's behavior) at the cost of the
// `threshold + maxOutput <= window` safety property: a turn at the threshold
// whose response uses the full maxOutput budget can overflow the window and be
// backend-rejected. That trade-off is accepted (see getEffectiveContextWindowSize).

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// These tests exercise the threshold *arithmetic*, so the settings-file layer is
// stubbed out to read the process environment directly. Without this the real
// ~/.claude/settings.json on the developer's machine would leak in and make the
// results depend on local configuration. The settings layer itself is covered
// by claude-settings.test.ts.
vi.mock("../src/claude-settings.js", () => ({
  getClaudeEnv: (key: string) => process.env[key],
  loadClaudeSettingsEnv: () => ({}),
  clearClaudeSettingsCache: () => {},
}));

import {
  getContextWindowForModel,
  getMaxOutputTokens,
  getAutoCompactWindow,
  getEffectiveContextWindowSize,
  getAutoCompactThreshold,
  isAutoCompactEnabled,
} from "../src/agent.js";

/** GLM-5.2's real window, as advertised by the proxy's /v1/models. */
const GLM_WINDOW = 196_608;

const ENV_KEYS = [
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
  "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
  "DISABLE_AUTO_COMPACT",
  "DISABLE_COMPACT",
] as const;

let saved: Record<string, string | undefined> = {};

beforeEach(() => {
  saved = {};
  for (const k of ENV_KEYS) {
    saved[k] = process.env[k];
    delete process.env[k];
  }
});

afterEach(() => {
  for (const k of ENV_KEYS) {
    if (saved[k] === undefined) delete process.env[k];
    else process.env[k] = saved[k];
  }
});

describe("context window resolution", () => {
  it("falls back to the default when nothing is configured", () => {
    // No proxy data loaded in tests and no override, so the default applies.
    expect(getContextWindowForModel("glm-5.2")).toBe(200_000);
  });

  it("honors CLAUDE_CODE_MAX_CONTEXT_TOKENS above all else", () => {
    // The documented override for models routed via ANTHROPIC_BASE_URL that the
    // CLI does not recognize as Claude models.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    expect(getContextWindowForModel("glm-5.2")).toBe(GLM_WINDOW);
    expect(getContextWindowForModel(undefined)).toBe(GLM_WINDOW);
  });

  it("ignores a malformed or non-positive override", () => {
    for (const bad of ["0", "-5", "abc", ""]) {
      process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = bad;
      expect(getContextWindowForModel("glm-5.2")).toBe(200_000);
    }
  });
});

describe("max output tokens", () => {
  it("defaults to the CLI's 32000 for unrecognized model ids", () => {
    expect(getMaxOutputTokens()).toBe(32_000);
  });

  it("honors CLAUDE_CODE_MAX_OUTPUT_TOKENS", () => {
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    expect(getMaxOutputTokens()).toBe(20_000);
  });
});

describe("auto-compact window", () => {
  it("is capped at the model's real window", () => {
    // The CLI caps this value, so it can only ever lower the capacity.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = "999999";
    expect(getAutoCompactWindow("glm-5.2")).toBe(GLM_WINDOW);
  });

  it("lowers the capacity when set below the real window", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = "100000";
    expect(getAutoCompactWindow("glm-5.2")).toBe(100_000);
  });
});

describe("effective window and threshold", () => {
  it("effective window is the (capped) model window with no reserve subtracted", () => {
    // No summary reserve: effective == the compaction window itself.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(GLM_WINDOW);
  });

  it("ignores the output budget — no reserve is subtracted", () => {
    // The old formula reserved min(maxOutput, 20000); that reserve is gone, so
    // a large output budget no longer lowers the effective window.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "64000";
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "8000";
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(GLM_WINDOW);
  });

  it("subtracts the 13000 buffer with no percentage override", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(GLM_WINDOW - 13_000); // 183608
  });

  it("applies the percentage to the effective (un-reserved) window", () => {
    // 90% of the un-reserved 196608 = 176947, which is below the 183608 base,
    // so the percentage wins.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(176_947);
  });

  it("only ever lowers the threshold — a high percentage has no effect", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "100";
    // floor(196608 * 1.0) = 196608, min with 183608 -> 183608
    expect(getAutoCompactThreshold("glm-5.2")).toBe(183_608);
  });

  it("compacts much earlier with a low percentage", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "50";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(Math.floor(GLM_WINDOW / 2));
  });

  it("ignores an out-of-range percentage", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    for (const bad of ["0", "101", "-1", "abc"]) {
      process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = bad;
      expect(getAutoCompactThreshold("glm-5.2")).toBe(183_608);
    }
  });

  it("documents the accepted overflow risk at the threshold", () => {
    // With the reserve removed, a turn at the threshold whose response uses
    // the full maxOutput budget CAN exceed the window — the backend rejects
    // rather than gracefully compacting. This is the trade-off of compaction
    // mode A (compact late, no reserve). Pinned as a known, accepted property.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "32000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    const threshold = getAutoCompactThreshold("glm-5.2");
    // threshold (176947) + maxOutput (32000) = 208947 > 196608 -> overflow.
    expect(threshold + getMaxOutputTokens()).toBeGreaterThan(GLM_WINDOW);
  });

  it("treats 91% of a 262144 window as over a 90% override, and 86% as under", () => {
    // A live turn sat at ~91% (239889/262144) after a previous completion at
    // 86% (225701). The 90% override must put the threshold between those two.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "262144";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    const threshold = getAutoCompactThreshold("qwen3.8-flash-next-coder-iq1_m");
    expect(threshold).toBe(Math.floor((262144 * 90) / 100));
    expect(239889 >= threshold).toBe(true);
    expect(225701 >= threshold).toBe(false);
  });
});

describe("auto-compact kill switches", () => {
  it("is enabled by default", () => {
    expect(isAutoCompactEnabled()).toBe(true);
  });

  it("honors DISABLE_AUTO_COMPACT", () => {
    for (const v of ["1", "true", "yes", "on", "TRUE", " On "]) {
      process.env.DISABLE_AUTO_COMPACT = v;
      expect(isAutoCompactEnabled()).toBe(false);
    }
  });

  it("honors DISABLE_COMPACT", () => {
    process.env.DISABLE_COMPACT = "1";
    expect(isAutoCompactEnabled()).toBe(false);
  });

  it("treats other values as not disabling", () => {
    for (const v of ["0", "false", "no", "off", ""]) {
      process.env.DISABLE_AUTO_COMPACT = v;
      expect(isAutoCompactEnabled()).toBe(true);
    }
  });
});
