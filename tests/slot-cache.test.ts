import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  getLlamaServerUrl,
  getSlotInfo,
  isSlotCacheEnabled,
  restoreSlotCache,
  saveSlotCache,
  saveSessionSlot,
  tryRestoreSlot,
} from "../src/slot-cache.js";

describe("isSlotCacheEnabled", () => {
  it("is off by default", () => {
    const saved = process.env.CLAUDE_SLOT_CACHE_ENABLED;
    delete process.env.CLAUDE_SLOT_CACHE_ENABLED;
    try {
      expect(isSlotCacheEnabled()).toBe(false);
    } finally {
      if (saved === undefined) delete process.env.CLAUDE_SLOT_CACHE_ENABLED;
      else process.env.CLAUDE_SLOT_CACHE_ENABLED = saved;
    }
  });

  it("is on only when set to exactly \"true\"", () => {
    const saved = process.env.CLAUDE_SLOT_CACHE_ENABLED;
    try {
      process.env.CLAUDE_SLOT_CACHE_ENABLED = "true";
      expect(isSlotCacheEnabled()).toBe(true);
      process.env.CLAUDE_SLOT_CACHE_ENABLED = "TRUE";
      expect(isSlotCacheEnabled()).toBe(false);
      process.env.CLAUDE_SLOT_CACHE_ENABLED = "1";
      expect(isSlotCacheEnabled()).toBe(false);
    } finally {
      if (saved === undefined) delete process.env.CLAUDE_SLOT_CACHE_ENABLED;
      else process.env.CLAUDE_SLOT_CACHE_ENABLED = saved;
    }
  });
});

describe("getLlamaServerUrl", () => {
  afterEach(() => {
    delete process.env.CLAUDE_LLAMA_SERVER_URL;
    delete process.env.ANTHROPIC_BASE_URL;
  });

  it("prefers CLAUDE_LLAMA_SERVER_URL and strips /v1", () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://192.168.124.14:8008/v1";
    process.env.ANTHROPIC_BASE_URL = "http://wrong:1/v1";
    expect(getLlamaServerUrl()).toBe("http://192.168.124.14:8008");
  });

  it("falls back to ANTHROPIC_BASE_URL and strips /v1 and trailing slash", () => {
    process.env.ANTHROPIC_BASE_URL = "http://10.0.0.1:9000/v1/";
    expect(getLlamaServerUrl()).toBe("http://10.0.0.1:9000");
  });

  it("returns null when neither env var is set", () => {
    expect(getLlamaServerUrl()).toBe(null);
  });
});

describe("getSlotInfo", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("reads the matching slot from the /slots list endpoint", async () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    const fetchMock = vi.fn(async (url: string) => {
      expect(String(url)).toBe("http://host:8008/slots");
      return {
        ok: true,
        json: async () => [
          {id: 0, n_prompt_tokens: 5000, n_prompt_tokens_cache: 100, is_processing: false},
          {id: 1, n_prompt_tokens: 0, n_prompt_tokens_cache: 0, is_processing: false},
        ],
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    const info = await getSlotInfo(0);
    expect(info).toEqual({
      id: 0,
      nPromptTokens: 5000,
      nPromptTokensCache: 100,
      isProcessing: false,
    });
  });

  it("returns null when the slot endpoint is unreachable", async () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("connection refused");
      }),
    );
    expect(await getSlotInfo(0)).toBe(null);
  });
});

describe("saveSlotCache", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.CLAUDE_LLAMA_SERVER_URL;
  });

  it("posts to /slots/{id}?action=save and reports token counts", async () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    const fetchMock = vi.fn(async (url: string, init: any) => {
      expect(String(url)).toBe("http://host:8008/slots/0?action=save");
      expect(init.method).toBe("POST");
      expect(init.body).toBe(JSON.stringify({filename: "sess-1.bin"}));
      return {
        ok: true,
        json: async () => ({filename: "sess-1.bin", n_saved: 42, n_written: 4200}),
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await saveSlotCache(0, "sess-1.bin");
    expect(result.success).toBe(true);
    expect(result.nTokens).toBe(42);
    expect(result.fileBytes).toBe(4200);
  });

  it("fails cleanly when no llama server URL is configured", async () => {
    const result = await saveSlotCache(0, "sess-1.bin");
    expect(result.success).toBe(false);
    expect(result.error).toMatch(/not set/);
  });
});

describe("restoreSlotCache", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.CLAUDE_LLAMA_SERVER_URL;
  });

  it("posts to /slots/{id}?action=restore and reports token counts", async () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        expect(String(url)).toBe("http://host:8008/slots/0?action=restore");
        return {
          ok: true,
          json: async () => ({filename: "sess-1.bin", n_restored: 42, n_read: 4200}),
        };
      }),
    );

    const result = await restoreSlotCache(0, "sess-1.bin");
    expect(result.success).toBe(true);
    expect(result.nTokens).toBe(42);
  });
});

describe("session wrappers", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    delete process.env.CLAUDE_SLOT_CACHE_ENABLED;
    delete process.env.CLAUDE_LLAMA_SERVER_URL;
  });

  it("saveSessionSlot is a no-op when the feature is disabled", async () => {
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    saveSessionSlot("sess-1");
    await new Promise((r) => setTimeout(r, 10));
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("tryRestoreSlot skips restore when the slot holds a conversation", async () => {
    process.env.CLAUDE_SLOT_CACHE_ENABLED = "true";
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    let restoreCalled = false;
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === "http://host:8008/slots") {
        return {
          ok: true,
          json: async () => [
            {id: 0, n_prompt_tokens: 5000, n_prompt_tokens_cache: 0, is_processing: false},
          ],
        };
      }
      restoreCalled = true;
      return {ok: true, json: async () => ({})};
    });
    vi.stubGlobal("fetch", fetchMock);

    await tryRestoreSlot("sess-1");
    expect(restoreCalled).toBe(false);
  });

  it("tryRestoreSlot restores from the session file when the slot is empty", async () => {
    process.env.CLAUDE_SLOT_CACHE_ENABLED = "true";
    process.env.CLAUDE_LLAMA_SERVER_URL = "http://host:8008";
    const fetchMock = vi.fn(async (url: string) => {
      if (String(url) === "http://host:8008/slots") {
        return {
          ok: true,
          json: async () => [
            {id: 0, n_prompt_tokens: 0, n_prompt_tokens_cache: 0, is_processing: false},
          ],
        };
      }
      return {
        ok: true,
        json: async () => ({filename: "sess-1.bin", n_restored: 10, n_read: 1000}),
      };
    });
    vi.stubGlobal("fetch", fetchMock);

    await tryRestoreSlot("sess-1");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });
});
