// Dynamic proxy model picker (Option B).
//
// By default the ACP model picker advertises the static list from
// ~/.claude/settings.json (CLAUDE_ACP_AVAILABLE_MODELS), which can drift from
// the live proxy catalog: gpt-oss-120b (tags=['fast'], no coding/near_sota)
// and Qwen3.6-27B (lifecycle=Deprecated) were hardcoded in settings, while a
// healthy multimodal model like Qwen3.8-27B-FP8-1M never got added. Setting
// CLAUDE_ACP_PROXY_MODELS=1 flips the picker to the live /v1/models +
// /v1/model-health catalog filtered through the same gate the codex catalog
// and ericli pickers use (healthy + non-Deprecated + coding/near_sota tag), so
// all four agents agree on the picker.

import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { createClaudeCodeAgent, loadProxyPickerModels } from "../src/agent.js";
import { clearClaudeSettingsCache, filterPickerModels } from "../src/claude-settings.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  readdirSync: vi.fn(() => []),
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

function createMockConnection() {
  return {
    sessionUpdate: vi.fn().mockResolvedValue(undefined),
    requestPermission: vi.fn(),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    createTerminal: vi.fn(),
    signal: new AbortController().signal,
    closed: new Promise(() => {}),
  } as unknown as AgentSideConnection;
}

const ENV_KEYS = [
  "ANTHROPIC_MODEL",
  "ANTHROPIC_SMALL_FAST_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "CLAUDE_ACP_MODEL",
  "CLAUDE_ACP_AVAILABLE_MODELS",
  "CLAUDE_ACP_PROXY_MODELS",
  "ERICAI_PROXY_URL",
  "ERICAI_PROXY_BASE_URL",
] as const;

const savedEnv: Record<string, string | undefined> = {};
let savedFetch: typeof globalThis.fetch;

beforeEach(() => {
  for (const k of ENV_KEYS) savedEnv[k] = process.env[k];
  // Isolate from the developer's shell/real proxy: default to the static
  // path and a failing fetch, then individual tests opt in.
  for (const k of ENV_KEYS) delete process.env[k];
  process.env.ANTHROPIC_MODEL = "test-model";
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

async function pickerOptions(): Promise<any> {
  const conn = createMockConnection();
  const agent: any = createClaudeCodeAgent(conn);
  await agent.initialize({
    protocolVersion: 1,
    clientCapabilities: {},
    clientInfo: { name: "test", version: "1.0" },
  });
  const r: any = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
  return (r.configOptions as any[]).find((o) => o.id === "model");
}

describe("filterPickerModels (shared picker gate)", () => {
  it("keeps healthy coding/near_sota models, drops fast-only and no-tag models", () => {
    const models = [
      { id: "glm-5.2", tags: ["sota", "near_sota", "coding"] },
      { id: "gpt-oss-120b", tags: ["fast"] },
      { id: "Qwen3.8-27B-FP8-1M", tags: ["1M", "multimodal", "coding"] },
      { id: "Qwen3-32B", tags: undefined },
      { id: "deepseek-v4-flash", tags: ["fast", "coding", "near_sota"] },
    ];
    const health = [
      { id: "glm-5.2", health: { healthy: true, lifecycle: "Experimental" } },
      { id: "gpt-oss-120b", health: { healthy: true, lifecycle: "Recommended" } },
      { id: "Qwen3.8-27B-FP8-1M", health: { healthy: true, lifecycle: "Experimental" } },
      { id: "deepseek-v4-flash", health: { healthy: true, lifecycle: "Experimental" } },
    ];
    expect(filterPickerModels(models, health)).toEqual([
      "glm-5.2",
      "Qwen3.8-27B-FP8-1M",
      "deepseek-v4-flash",
    ]);
  });

  it("drops Deprecated lifecycle even when healthy and tagged", () => {
    const models = [{ id: "Qwen3.6-27B", tags: ["fast", "multimodal", "coding"] }];
    const health = [
      { id: "Qwen3.6-27B", health: { healthy: true, lifecycle: "Deprecated" } },
    ];
    expect(filterPickerModels(models, health)).toEqual([]);
  });

  it("drops scaled-to-zero (unhealthy) models", () => {
    const models = [{ id: "glm-5.2", tags: ["coding"] }];
    const health = [{ id: "glm-5.2", health: { healthy: false, lifecycle: "Experimental" } }];
    expect(filterPickerModels(models, health)).toEqual([]);
  });

  it("accepts the ericai_backend_id as a health key (alias-shaped health)", () => {
    const models = [{ id: "glm-5.2", tags: ["coding"] }];
    const health = [
      { ericai_backend_id: "glm-5.2", health: { healthy: true, lifecycle: "Experimental" } },
    ];
    expect(filterPickerModels(models, health)).toEqual(["glm-5.2"]);
  });

  it("treats all models healthy when health data is empty (fallback)", () => {
    const models = [{ id: "glm-5.2", tags: ["coding"] }];
    expect(filterPickerModels(models, [], true)).toEqual(["glm-5.2"]);
  });

  it("still applies the tag gate in the all-healthy fallback", () => {
    const models = [{ id: "gpt-oss-120b", tags: ["fast"] }];
    expect(filterPickerModels(models, [], true)).toEqual([]);
  });
});

describe("dynamic picker via proxy catalog (CLAUDE_ACP_PROXY_MODELS=1)", () => {
  it("advertises the live filtered proxy list instead of the static settings", async () => {
    process.env.CLAUDE_ACP_PROXY_MODELS = "1";
    // Static list deliberately diverges from the live catalog (the bug).
    process.env.CLAUDE_ACP_AVAILABLE_MODELS =
      "Qwen3.6-27B,Qwen3.8-27B,gpt-oss-120b";
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
                { id: "Qwen3.8-27B-FP8-1M", health: { healthy: true, lifecycle: "Experimental" } },
                { id: "gpt-oss-120b", health: { healthy: true, lifecycle: "Recommended" } },
                { id: "Qwen3.6-27B", health: { healthy: true, lifecycle: "Deprecated" } },
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
              { id: "Qwen3.8-27B-FP8-1M", tags: ["1M", "multimodal", "coding"] },
              { id: "gpt-oss-120b", tags: ["fast"] },
              { id: "Qwen3.6-27B", tags: ["fast", "multimodal", "coding"] },
            ],
          }),
        };
      }),
    );
    await loadProxyPickerModels();
    const picker = await pickerOptions();
    const values: string[] = picker.options.map((o: any) => o.value);
    // gpt-oss-120b (no coding/near_sota) and Qwen3.6-27B (Deprecated) are
    // gone; Qwen3.8-27B-FP8-1M (healthy, coding) now appears.
    expect(values).toContain("glm-5.2");
    expect(values).toContain("Qwen3.8-27B-FP8-1M");
    expect(values).not.toContain("gpt-oss-120b");
    expect(values).not.toContain("Qwen3.6-27B");
  });

  it("falls back to the static settings list when the proxy fetch fails", async () => {
    process.env.CLAUDE_ACP_PROXY_MODELS = "1";
    process.env.CLAUDE_ACP_AVAILABLE_MODELS = "glm-5.2,Qwen3-Coder";
    // fetch rejects (proxy unreachable).
    vi.stubGlobal("fetch", vi.fn(async () => { throw new Error("ECONNREFUSED"); }));
    await loadProxyPickerModels();
    const picker = await pickerOptions();
    expect(picker.options.map((o: any) => o.value)).toEqual(["glm-5.2", "Qwen3-Coder"]);
  });

  it("keeps the static picker when CLAUDE_ACP_PROXY_MODELS is not set", async () => {
    process.env.CLAUDE_ACP_AVAILABLE_MODELS = "glm-5.2,gpt-oss-120b";
    // No proxy fetch should be attempted.
    const fetchSpy = vi.fn(async () => {
      throw new Error("unexpected fetch");
    });
    vi.stubGlobal("fetch", fetchSpy);
    await loadProxyPickerModels(); // no-op when flag off
    const picker = await pickerOptions();
    expect(picker.options.map((o: any) => o.value)).toEqual(["glm-5.2", "gpt-oss-120b"]);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
