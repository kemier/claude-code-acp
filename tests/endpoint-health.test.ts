import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { clearClaudeSettingsCache } from "../src/claude-settings.js";
import {
  clearEndpointHealthCache,
  filterHealthyEndpoints,
} from "../src/endpoint-health.js";
import type { AcpEndpoint } from "../src/claude-settings.js";

let userSettingsDir: string;
let savedSettingsDir: string | undefined;

const endpoints: AcpEndpoint[] = [
  {
    id: "strata-windows",
    name: "Win",
    model: "qwen",
    baseUrl: "http://192.168.124.20:8080",
  },
  {
    id: "strata-ubuntu",
    name: "Ubu",
    model: "qwen",
    baseUrl: "http://192.168.124.14:8080",
  },
];

beforeEach(() => {
  userSettingsDir = mkdtempSync(join(tmpdir(), "acp-health-user-"));
  savedSettingsDir = process.env.CLAUDE_SETTINGS_DIR;
  process.env.CLAUDE_SETTINGS_DIR = userSettingsDir;
  mkdirSync(join(userSettingsDir), { recursive: true });
  writeFileSync(
    join(userSettingsDir, "settings.json"),
    JSON.stringify({ env: { ANTHROPIC_AUTH_TOKEN: "test-key" } }),
  );
  clearClaudeSettingsCache();
  clearEndpointHealthCache();
});

afterEach(() => {
  vi.unstubAllGlobals();
  if (savedSettingsDir === undefined) delete process.env.CLAUDE_SETTINGS_DIR;
  else process.env.CLAUDE_SETTINGS_DIR = savedSettingsDir;
  clearClaudeSettingsCache();
  clearEndpointHealthCache();
  rmSync(userSettingsDir, { recursive: true, force: true });
});

describe("filterHealthyEndpoints", () => {
  it("keeps only endpoints whose /health returns status ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).includes("192.168.124.20")) {
          return {
            ok: true,
            json: async () => ({ status: "ok", service: "strata" }),
          };
        }
        return { ok: false, json: async () => ({}) };
      }),
    );

    const healthy = await filterHealthyEndpoints(endpoints);
    expect(healthy.map((e) => e.id)).toEqual(["strata-windows"]);
    expect(fetch).toHaveBeenCalledWith(
      "http://192.168.124.20:8080/health",
      expect.objectContaining({
        headers: { Authorization: "Bearer test-key" },
      }),
    );
  });

  it("returns empty when every probe fails", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("timeout");
      }),
    );
    const healthy = await filterHealthyEndpoints(endpoints);
    expect(healthy).toEqual([]);
  });

  it("reuses the TTL cache on the next call", async () => {
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url).includes("192.168.124.20")) {
        return { ok: true, json: async () => ({ status: "ok" }) };
      }
      return { ok: false, json: async () => ({}) };
    });
    vi.stubGlobal("fetch", fetchMock);

    await filterHealthyEndpoints(endpoints);
    await filterHealthyEndpoints(endpoints);
    // Two endpoints × one probe round (second call hits cache).
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
