// Per-model windows come from Claude Code's result.contextWindow, not a
// catalog fetch. A known CLI window wins over CLAUDE_CODE_MAX_CONTEXT_TOKENS.

import { describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/claude-settings.js", () => ({
  getClaudeEnv: (key: string) => process.env[key],
  loadClaudeSettingsEnv: () => ({}),
  clearClaudeSettingsCache: () => {},
}));

import {
  recordCliContextWindow,
  resetCliReportedContextWindows,
  getContextWindowForModel,
  getAutoCompactThreshold,
} from "../src/agent.js";

const CLI_WINDOWS: Record<string, number> = {
  "glm-5.2": 196_608,
  "glm-5.1": 131_072,
  "deepseek-v4-pro": 262_144,
  "deepseek-v4-flash": 262_144,
  "Qwen3-Coder": 262_144,
};

beforeAll(() => {
  for (const [id, window] of Object.entries(CLI_WINDOWS)) {
    recordCliContextWindow(id, window);
  }
});

afterAll(() => {
  resetCliReportedContextWindows();
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

describe("per-model windows from the CLI", () => {
  it("uses the CLI-reported window when no override is set", () => {
    expect(getContextWindowForModel("glm-5.2")).toBe(196_608);
    expect(getContextWindowForModel("glm-5.1")).toBe(131_072);
    expect(getContextWindowForModel("deepseek-v4-pro")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-flash")).toBe(262_144);
    expect(getContextWindowForModel("Qwen3-Coder")).toBe(262_144);
  });

  it("falls back to the default for a model the CLI has not reported", () => {
    expect(getContextWindowForModel("some-unknown-model")).toBe(200_000);
  });
});

describe("global override vs per-model CLI window", () => {
  it("honors the override for a model with a matching CLI window", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("glm-5.2")).toBe(196_608);
  });

  it("does not inflate a model whose CLI window is smaller", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("glm-5.1")).toBe(131_072);
  });

  it("compacts at window - 13000 for a smaller model (no reserve)", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS = "20000";
    const threshold = getAutoCompactThreshold("glm-5.1");
    expect(threshold).toBe(118_072);
    expect(threshold + 20_000).toBeGreaterThan(131_072);
  });

  it("does not let the override lower a larger model's window", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("Qwen3-Coder")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-pro")).toBe(262_144);
    expect(getContextWindowForModel("deepseek-v4-flash")).toBe(262_144);
  });

  it("applies the override to a model the CLI never reported", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("some-unknown-model")).toBe(196_608);
  });
});

describe("CLI-reported window vs env override", () => {
  it("prefers the CLI window over the override when the model is known", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    expect(getContextWindowForModel("deepseek-v4-flash")).toBe(262_144);
  });

  it("computes the 90% threshold against the real (larger) window", () => {
    process.env.CLAUDE_CODE_MAX_CONTEXT_TOKENS = "196608";
    process.env.CLAUDE_AUTOCOMPACT_PCT_OVERRIDE = "90";
    expect(getAutoCompactThreshold("deepseek-v4-flash")).toBe(
      Math.floor(262_144 * 0.9),
    );
  });
});
