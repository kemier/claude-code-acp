// Verifies how the per-model window from the proxy interacts with the CLI's
// global CLAUDE_CODE_MAX_CONTEXT_TOKENS override.
//
// Lives in its own file because loading the proxy's model list mutates
// module-level state (and only runs once), which would change the defaults the
// other context-window tests assert against.
//
// The hazard under test: the CLI applies its context override to every model.
// Our model picker offers windows from 131072 (glm-5.1) to 262144 (Qwen3-Coder,
// deepseek-v4-pro, deepseek-v4-flash — the whole DeepSeek V4 series shares one
// 262144 backend limit). The real proxy-advertised window always wins when
// known: it protects smaller-window models (a 196608 override must not inflate
// a 131072 model past its real limit) and preserves larger windows (the
// override must not clamp a 262144 model down to 196608, which would hide
// capacity and compact too early). The override only applies when the proxy
// never advertised a window for the model.

import { describe, it, expect, beforeAll, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/claude-settings.js", () => ({
  getClaudeEnv: (key: string) => process.env[key],
  loadClaudeSettingsEnv: () => ({}),
  clearClaudeSettingsCache: () => {},
}));

import {
  loadModelContextWindows,
  getContextWindowForModel,
  getAutoCompactThreshold,
} from "../src/agent.js";

/** Windows as the proxy really advertises them on /v1/models. */
const PROXY_WINDOWS: Record<string, number> = {
  "glm-5.2": 196_608,
  "glm-5.1": 131_072,
  "deepseek-v4-pro": 262_144,
  "deepseek-v4-flash": 262_144,
  "Qwen3-Coder": 262_144,
};

beforeAll(async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        data: Object.entries(PROXY_WINDOWS).map(([id, context_length]) => ({
          id,
          context_length,
        })),
      }),
    })),
  );
  await loadModelContextWindows();
});

const ENV_KEYS = [
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
  "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
  "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
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

describe("per-model windows from the proxy", () => {
  it("uses the proxy's window when no override is set", () => {
    expect(getContextWindowForModel("glm-5.2")).toBe(196_608);
    expect(getContextWindowForModel("glm-5.1")).toBe(131_072);
    expect(getContextWindowForModel("deepseek-v4-pro")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-flash")).toBe(262_144);
    expect(getContextWindowForModel("Qwen3-Coder")).toBe(262_144);
  });

  it("falls back to the default for a model the proxy did not advertise", () => {
    expect(getContextWindowForModel("some-unknown-model")).toBe(200_000);
  });
});

describe("global override vs per-model reality", () => {
  it("honors the override for the model it was written for", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("glm-5.2")).toBe(196_608);
  });

  it("does not inflate a model whose real window is smaller", () => {
    // Without the cap this would report 196608 for a 131072-token model and the
    // threshold would sit above the real limit — overflow with no compaction.
    // glm-5.1 is the 131072 model; deepseek-v4-pro is 262144 (see below).
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("glm-5.1")).toBe(131_072);
  });

  it("compacts at window - 13000 for a smaller model (no reserve)", () => {
    // With the summary reserve removed, the threshold is just window - 13000.
    // The output budget no longer factors in, so threshold + maxOutput can
    // exceed the real window (accepted overflow risk, compaction mode A).
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    const threshold = getAutoCompactThreshold("glm-5.1");
    // 131072 - 13000
    expect(threshold).toBe(118_072);
    // No longer guaranteed to fit: 118072 + 20000 = 138072 > 131072.
    expect(threshold + 20_000).toBeGreaterThan(131_072);
  });

  it("does not let the override lower a larger model's window", () => {
    // These models really have 262144; the override was written for a smaller
    // model (glm-5.2 @ 196608) and must not hide capacity from a larger one.
    // Clamping a larger window down to the override would compact far earlier
    // than necessary and misreport the denominator in the usage UI. The whole
    // DeepSeek V4 series shares the same 262144 backend limit.
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("Qwen3-Coder")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-pro")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-flash")).toBe(262_144);
  });

  it("applies the override to a model the proxy never advertised", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("some-unknown-model")).toBe(196_608);
  });
});
