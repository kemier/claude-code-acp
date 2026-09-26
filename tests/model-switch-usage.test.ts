// Regression: switching the model mid-conversation (setSessionConfigOption with
// configId "model") used to leave the context ring showing the OLD model's
// window. The ring's `size` denominator is only ever set by a usage_update
// frame, and those fire inside the per-turn result-event handler — a model
// switch does not start a turn, so the ring kept e.g. 262144 (Qwen3.8-27B)
// after the user picked the 1M variant (Qwen3.8-27B-FP8-1M).
//
// The fix: setSessionConfigOption re-emits a usage_update with the NEW model's
// (proxy-authoritative) window via reEmitUsageOnModelSwitch, carrying over the
// last known used tokens so the ring does not falsely reset to 0%.
//
// Lives in its own file because loadModelContextWindows mutates module-level
// state (the proxy window map) and only runs once — the same isolation
// context-window-models.test.ts uses.

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
  loadModelContextWindows,
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

beforeAll(async () => {
  for (const k of ENV_KEYS) savedEnv[k] = process.env[k];
  // Strip any env override so the proxy-advertised window is the only signal.
  for (const k of ENV_KEYS) delete process.env[k];

  // Seed the proxy map with both the bare and 1M Qwen variants, exactly as the
  // live proxy advertises them on /v1/models.
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({
        data: [
          { id: "Qwen3.8-27B", context_length: 262144 },
          { id: "Qwen3.8-27B-FP8-1M", context_length: 1000000 },
        ],
      }),
    })),
  );
  await loadModelContextWindows();
});

afterAll(() => {
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k]!;
  }
  vi.unstubAllGlobals();
});

describe("model switch context window", () => {
  it("resolves the 1M variant to 1000000 and the bare model to 262144", () => {
    // The core of the bug: the bare model resolves to 262144. If the switch
    // had not updated the ring, the user would keep seeing this after picking
    // the 1M variant.
    expect(getContextWindowForModel("Qwen3.8-27B")).toBe(262144);
    // The 1M variant must resolve to its real 1M window, not the bare model's.
    expect(getContextWindowForModel("Qwen3.8-27B-FP8-1M")).toBe(1000000);
  });

  it("setSessionConfigOption re-emits usage on a model change (wiring)", () => {
    const src = readTextFile(resolve(process.cwd(), "src/agent.ts"));
    const idx = src.indexOf("async setSessionConfigOption(\n");
    expect(idx).toBeGreaterThan(0);
    // The model-change re-emit lives in the trailing else branch (after the
    // mode + collaboration-mode handling), ~7.3k chars into the method, so
    // slice generously.
    const body = src.slice(idx, idx + 8000);
    // The model branch must call the re-emit helper on a genuine change
    // (prevModel !== value), with the captured previous model.
    expect(body).toContain('reEmitUsageOnModelSwitch(sessionId, connection, store, prevModel, String(value))');
    expect(body).toContain('configId === "model" && prevModel !== value');
  });

  it("reEmitUsageOnModelSwitch emits nothing when no turn has run yet (no false 0%)", async () => {
    // Fresh session id: sessionLastUsage has no entry, so used=0 and the helper
    // must NOT emit (the ring is hidden until the first real usage_update).
    const calls: any[] = [];
    const conn = {
      sessionUpdate: async (p: any) => {
        calls.push(p);
      },
    };
    // store is unused by the helper beyond the signature; pass a minimal stub.
    reEmitUsageOnModelSwitch("no-usage-session", conn, {} as any, "Qwen3.8-27B", "Qwen3.8-27B-FP8-1M");
    expect(calls).toHaveLength(0);
  });
});
