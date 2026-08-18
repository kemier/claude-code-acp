// Dynamic picker flag via the settings `env` block (regression).
//
// The CLAUDE_ACP_PROXY_MODELS=1 opt-in flag lives in the `env` block of
// ~/.claude/settings.json. The ACP parent process never inherits that block —
// Claude Code only applies it to its own subprocess — so reading the flag with
// `process.env[CLAUDE_ACP_PROXY_MODELS]` silently kept the static picker
// (which still advertised gpt-oss-120b / Qwen3.6-27B). The gate now resolves
// through getClaudeEnv, which reads the settings env block first.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { loadProxyPickerModels } from "../src/agent.js";
import { clearClaudeSettingsCache } from "../src/claude-settings.js";

const PROJECT_SETTINGS = JSON.stringify({
  env: { CLAUDE_ACP_PROXY_MODELS: "1" },
});

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  // Only the project-layer settings file exists; the user-layer read must not
  // leak the developer's real flag into the test.
  readFileSync: vi.fn((path: string) => {
    if (String(path).endsWith(process.env.CLAUDE_PROJECT_DIR + "/.claude/settings.json")) {
      return PROJECT_SETTINGS;
    }
    throw new Error("ENOENT");
  }),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  readdirSync: vi.fn(() => []),
}));

const ENV_KEYS = [
  "ANTHROPIC_MODEL",
  "ANTHROPIC_SMALL_FAST_MODEL",
  "CLAUDE_ACP_AVAILABLE_MODELS",
  "CLAUDE_ACP_PROXY_MODELS",
  "ERICAI_PROXY_URL",
  "ERICAI_PROXY_BASE_URL",
  "CLAUDE_PROJECT_DIR",
] as const;

const savedEnv: Record<string, string | undefined> = {};
let savedFetch: typeof globalThis.fetch;

beforeEach(() => {
  for (const k of ENV_KEYS) savedEnv[k] = process.env[k];
  for (const k of ENV_KEYS) delete process.env[k];
  process.env.ANTHROPIC_MODEL = "test-model";
  process.env.CLAUDE_PROJECT_DIR = "/tmp/acp-project";
  process.env.ERICAI_PROXY_BASE_URL = "http://proxy.test.invalid";
  savedFetch = globalThis.fetch;
  clearClaudeSettingsCache();
});

afterEach(() => {
  for (const k of ENV_KEYS) {
    if (savedEnv[k] === undefined) delete process.env[k];
    else process.env[k] = savedEnv[k];
  }
  globalThis.fetch = savedFetch;
  clearClaudeSettingsCache();
});

describe("CLAUDE_ACP_PROXY_MODELS set in the settings env block", () => {
  it("activates the dynamic proxy picker even though process.env lacks the flag", async () => {
    // The flag is NOT in the process environment — only in the project
    // settings env block (as production configures it).
    expect(process.env.CLAUDE_ACP_PROXY_MODELS).toBeUndefined();

    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url.includes("/v1/model-health")) {
          return {
            ok: true,
            status: 200,
            json: async () => ({
              data: [
                { id: "glm-5.2", health: { healthy: true, lifecycle: "Experimental" } },
                { id: "gpt-oss-120b", health: { healthy: true, lifecycle: "Recommended" } },
              ],
            }),
          };
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({
            data: [
              { id: "glm-5.2", tags: ["sota", "near_sota", "coding"] },
              { id: "gpt-oss-120b", tags: ["fast"] },
            ],
          }),
        };
      }),
    );

    // Must NOT no-op like the old process.env gate did: it should attempt the
    // proxy fetch and load the live filtered list.
    await expect(loadProxyPickerModels()).resolves.toBeUndefined();

    const fetchMock = vi.mocked(globalThis.fetch);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/v1/models"),
      expect.anything(),
    );
  });
});
