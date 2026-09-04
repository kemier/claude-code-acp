/**
 * Slot KV cache save/restore via llama-server's slot API.
 *
 * Enables fast session resume by persisting the KV cache to disk
 * instead of re-prefilling the full conversation history on the
 * llama-server.
 *
 * Requires:
 *   - llama-server started with `--slot-save-path <dir>`
 *   - CLAUDE_LLAMA_SERVER_URL env var (e.g. "http://192.168.124.14:8008"), or
 *     fallback to ANTHROPIC_BASE_URL (only valid when Claude Code is routed
 *     directly at llama-server, which natively serves /v1/messages)
 *   - CLAUDE_SLOT_CACHE_ENABLED env var set to "true" (feature gate)
 *
 * When the resolved URL is a proxy (no /slots endpoint), the probe fails and
 * every save/restore is skipped with a log line — the feature is inert, not
 * destructive.
 */

import { logger } from "./logger.js";

const SAVE_TIMEOUT_MS = 30_000;
const RESTORE_TIMEOUT_MS = 60_000;
const SLOT_INFO_TIMEOUT_MS = 5_000;

type SlotCacheResult = {
  success: boolean;
  filename: string;
  nTokens: number;
  fileBytes: number;
  elapsedMs: number;
  error?: string;
};

export function isSlotCacheEnabled(): boolean {
  return process.env.CLAUDE_SLOT_CACHE_ENABLED === "true";
}

export function getLlamaServerUrl(): string | null {
  const url =
    process.env.CLAUDE_LLAMA_SERVER_URL || process.env.ANTHROPIC_BASE_URL;
  if (!url) return null;
  return url.replace(/\/v1\/?$/, "").replace(/\/$/, "");
}

/**
 * Save the KV cache for a llama-server slot to a file.
 */
export async function saveSlotCache(
  slotId: number,
  filename: string,
): Promise<SlotCacheResult> {
  const base = getLlamaServerUrl();
  if (!base) {
    return {
      success: false,
      filename,
      nTokens: 0,
      fileBytes: 0,
      elapsedMs: 0,
      error:
        "CLAUDE_LLAMA_SERVER_URL not set and no ANTHROPIC_BASE_URL in env",
    };
  }

  const t0 = Date.now();
  try {
    const res = await fetch(`${base}/slots/${slotId}?action=save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
      signal: AbortSignal.timeout(SAVE_TIMEOUT_MS),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return {
        success: false,
        filename,
        nTokens: 0,
        fileBytes: 0,
        elapsedMs: Date.now() - t0,
        error: `HTTP ${res.status}: ${body.slice(0, 200)}`,
      };
    }

    const data = (await res.json()) as Record<string, unknown>;
    return {
      success: true,
      filename: (data["filename"] as string) ?? filename,
      nTokens: (data["n_saved"] as number) ?? 0,
      fileBytes: (data["n_written"] as number) ?? 0,
      elapsedMs: Date.now() - t0,
    };
  } catch (e) {
    const err = e as Error | undefined;
    return {
      success: false,
      filename,
      nTokens: 0,
      fileBytes: 0,
      elapsedMs: Date.now() - t0,
      error: err?.message ?? String(e),
    };
  }
}

/**
 * Restore the KV cache for a llama-server slot from a file.
 */
export async function restoreSlotCache(
  slotId: number,
  filename: string,
): Promise<SlotCacheResult> {
  const base = getLlamaServerUrl();
  if (!base) {
    return {
      success: false,
      filename,
      nTokens: 0,
      fileBytes: 0,
      elapsedMs: 0,
      error:
        "CLAUDE_LLAMA_SERVER_URL not set and no ANTHROPIC_BASE_URL in env",
    };
  }

  const t0 = Date.now();
  try {
    const res = await fetch(`${base}/slots/${slotId}?action=restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
      signal: AbortSignal.timeout(RESTORE_TIMEOUT_MS),
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      return {
        success: false,
        filename,
        nTokens: 0,
        fileBytes: 0,
        elapsedMs: Date.now() - t0,
        error: `HTTP ${res.status}: ${body.slice(0, 200)}`,
      };
    }

    const data = (await res.json()) as Record<string, unknown>;
    return {
      success: true,
      filename: (data["filename"] as string) ?? filename,
      nTokens: (data["n_restored"] as number) ?? 0,
      fileBytes: (data["n_read"] as number) ?? 0,
      elapsedMs: Date.now() - t0,
    };
  } catch (e) {
    const err = e as Error | undefined;
    return {
      success: false,
      filename,
      nTokens: 0,
      fileBytes: 0,
      elapsedMs: Date.now() - t0,
      error: err?.message ?? String(e),
    };
  }
}

/**
 * Get the current slot info from llama-server.
 * Uses the `/slots` list endpoint — the single-slot `GET /slots/{id}`
 * endpoint returns 404 on some llama-server builds.
 * Returns null if the slot endpoint is unreachable (feature effectively off).
 */
export async function getSlotInfo(
  slotId: number = 0,
): Promise<{
  id: number;
  nPromptTokens: number;
  nPromptTokensCache: number;
  isProcessing: boolean;
} | null> {
  const base = getLlamaServerUrl();
  if (!base) {
    return null;
  }

  try {
    const res = await fetch(`${base}/slots`, {
      method: "GET",
      signal: AbortSignal.timeout(SLOT_INFO_TIMEOUT_MS),
    });
    if (!res.ok) {
      return null;
    }
    const data = (await res.json()) as Array<Record<string, unknown>>;
    const slot = data.find((entry) => entry["id"] === slotId);
    if (!slot) {
      return null;
    }
    return {
      id: (slot["id"] as number) ?? slotId,
      nPromptTokens: (slot["n_prompt_tokens"] as number) ?? 0,
      nPromptTokensCache: (slot["n_prompt_tokens_cache"] as number) ?? 0,
      isProcessing: (slot["is_processing"] as boolean) ?? false,
    };
  } catch {
    logger.info(`[slot-cache] slot info request failed for slot ${slotId}`);
    return null;
  }
}

/** Save the llama-server slot KV cache for a session (fire-and-forget). */
export function saveSessionSlot(sessionId: string): void {
  if (!isSlotCacheEnabled()) {
    return;
  }
  void (async () => {
    try {
      const result = await saveSlotCache(0, `${sessionId}.bin`);
      if (result.success) {
        logger.info(
          `[slot-cache] saved session=${sessionId} tokens=${result.nTokens} file=${result.fileBytes}B (${result.elapsedMs}ms)`,
        );
      } else {
        logger.info(
          `[slot-cache] save failed session=${sessionId}: ${result.error}`,
        );
      }
    } catch (e) {
      const err = e as Error | undefined;
      logger.error(
        `[slot-cache] save error session=${sessionId}: ${err?.message ?? String(e)}`,
      );
    }
  })();
}

/** Attempt to restore the llama-server slot KV cache for a session (best-effort). */
export async function tryRestoreSlot(sessionId: string): Promise<void> {
  if (!isSlotCacheEnabled()) {
    return;
  }
  try {
    // Only restore if the slot is empty (no prompt has been sent yet).
    // If the slot already has tokens, the restore is pointless churn.
    const slotInfo = await getSlotInfo(0);
    if (slotInfo && slotInfo.nPromptTokens > 1000) {
      logger.info(
        `[slot-cache] restore skipped session=${sessionId}: slot has ${slotInfo.nPromptTokens} tokens (not empty)`,
      );
      return;
    }

    const result = await restoreSlotCache(0, `${sessionId}.bin`);
    if (result.success) {
      logger.info(
        `[slot-cache] restored session=${sessionId} tokens=${result.nTokens} file=${result.fileBytes}B (${result.elapsedMs}ms)`,
      );
    } else {
      // Restore failed — file may not exist (first session) or is stale, or
      // the URL points at a proxy without /slots support. Expected; the
      // normal prefill path handles it.
      logger.info(
        `[slot-cache] restore skipped session=${sessionId}: ${result.error}`,
      );
    }
  } catch (e) {
    const err = e as Error | undefined;
    logger.error(
      `[slot-cache] restore error session=${sessionId}: ${err?.message ?? String(e)}`,
    );
  }
}
