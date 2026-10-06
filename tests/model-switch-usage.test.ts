// Regression: switching the model mid-conversation (setSessionConfigOption with
// configId "model") used to leave the context ring showing the OLD model's
// window. The ring's `size` denominator is only ever set by a usage_update
// frame, and those fire inside the per-turn result-event handler — a model
// switch does not start a turn, so the ring kept e.g. 262144 (Qwen3.8-27B)
// after the user picked the 1M variant (Qwen3.8-27B-FP8-1M).
//
// The fix: setSessionConfigOption re-emits a usage_update with the NEW model's
// window via reEmitUsageOnModelSwitch, carrying over the last known used tokens
// so the ring does not falsely reset to 0%.

import { describe, it, expect, beforeAll, afterAll, vi } from "vitest";
import { createRequire } from "node:module";
import { resolve } from "node:path";

const require = createRequire(__filename);
const readTextFile: (p: string) => string = require("./src-file-reader.cjs").readTextFile;

vi.mock("../src/claude-settings.js", () => ({
  getClaudeEnv: (key: string) => process.env[key],
  loadClaudeSettingsEnv: () => ({}),
  clearClaudeSettingsCache: () => {},
}));

import {
  recordCliContextWindow,
  resetCliReportedContextWindows,
  getContextWindowForModel,
  reEmitUsageOnModelSwitch,
} from "../src/agent.js";

const ENV_KEYS = [
  "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
  "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
  "DISABLE_COMPACT",
  "DISABLE_AUTO_COMPACT",
] as const;
const savedEnv: Record<string, string | undefined> = {};

beforeAll(() => {
  for (const k of ENV_KEYS) savedEnv[k] = process.env[k];
  for (const k of ENV_KEYS) delete process.env[k];
  recordCliContextWindow("Qwen3.8-27B", 262144);
  recordCliContextWindow("Qwen3.8-27B-FP8-1M", 1000000);
});

afterAll(() => {
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k]!;
  }
  resetCliReportedContextWindows();
});

describe("model switch context window", () => {
  it("resolves the 1M variant to 1000000 and the bare model to 262144", () => {
    expect(getContextWindowForModel("Qwen3.8-27B")).toBe(262144);
    expect(getContextWindowForModel("Qwen3.8-27B-FP8-1M")).toBe(1000000);
  });

  it("setSessionConfigOption re-emits usage on a model change (wiring)", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const idx = src.indexOf("async setSessionConfigOption(");
    expect(idx).toBeGreaterThan(0);
    const body = src.slice(idx, idx + 8000);
    expect(body).toContain('reEmitUsageOnModelSwitch(sessionId, connection, store, prevModel, String(value))');
    expect(body).toContain('configId === "model" && prevModel !== value');
  });

  it("reEmitUsageOnModelSwitch emits nothing when no turn has run yet (no false 0%)", async () => {
    const calls: any[] = [];
    const conn = {
      sessionUpdate: async (p: any) => {
        calls.push(p);
      },
    };
    reEmitUsageOnModelSwitch("no-usage-session", conn, {} as any, "Qwen3.8-27B", "Qwen3.8-27B-FP8-1M");
    expect(calls).toHaveLength(0);
  });
});
