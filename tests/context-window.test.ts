// Verifies the ACP layer computes the same auto-compact threshold Claude Code
// CLI does.
//
// Both layers can compact. If their arithmetic disagrees, whichever fires first
// wins and the other silently never runs — so tuning compaction via the
// documented environment variables would have unpredictable effect. These tests
// pin the ACP side to the CLI's formula:
//
//   effectiveWindow = min(window, AUTO_COMPACT_WINDOW) - min(maxOutput, 20000)
//   threshold       = effectiveWindow - 13000
//   with PCT=N        threshold = min(floor(effectiveWindow * N/100), threshold)
//
// The CLI implementation this mirrors (v2.1.220, minified):
//   function yfo(e,t){let r=e-13000,n=t.testPctOverride;
//     if(n!==void 0&&!isNaN(n)&&n>0&&n<=100)return Math.min(Math.floor(e*(n/100)),r);
//     return r}

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

describe("effective window and threshold (CLI parity)", () => {
  it("reproduces the effectiveWindow the CLI logs for our configuration", () => {
    // Claude Code's own debug log for this exact env prints
    // "autocompact: ... effectiveWindow=176608".
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(176_608);
  });

  it("caps the summary reserve at 20000 even with a larger output budget", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "64000";
    // reserve is min(64000, 20000) = 20000
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(GLM_WINDOW - 20_000);
  });

  it("reserves the full output budget when it is below the cap", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "8000";
    expect(getEffectiveContextWindowSize("glm-5.2")).toBe(GLM_WINDOW - 8_000);
  });

  it("subtracts the 13000 buffer with no percentage override", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(176_608 - 13_000); // 163608
  });

  it("applies the percentage to the effective window, matching the CLI", () => {
    // Not 90% of the raw 196608 (=176947) — 90% of the effective 176608.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(158_947);
  });

  it("only ever lowers the threshold — a high percentage has no effect", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "100";
    // floor(176608 * 1.0) = 176608, min with 163608 -> 163608
    expect(getAutoCompactThreshold("glm-5.2")).toBe(163_608);
  });

  it("compacts much earlier with a low percentage", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "50";
    expect(getAutoCompactThreshold("glm-5.2")).toBe(Math.floor(176_608 / 2));
  });

  it("ignores an out-of-range percentage", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    for (const bad of ["0", "101", "-1", "abc"]) {
      process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = bad;
      expect(getAutoCompactThreshold("glm-5.2")).toBe(163_608);
    }
  });

  it("leaves headroom so a request at the threshold cannot overflow", () => {
    // This is the property that actually prevents the failure: the backend
    // rejects when prompt + max_tokens exceeds the window.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW = String(GLM_WINDOW);
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    const worstCase = getAutoCompactThreshold("glm-5.2") + getMaxOutputTokens();
    expect(worstCase).toBeLessThanOrEqual(GLM_WINDOW);
  });

  it("would overflow with the old naive 90%-of-raw-window rule", () => {
    // Regression guard documenting why the previous formula was unsafe:
    // 90% of 196608 = 176947, plus a 32000 output budget = 208947 > 196608.
    const naive = Math.floor((GLM_WINDOW * 9) / 10);
    expect(naive + 32_000).toBeGreaterThan(GLM_WINDOW);
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
