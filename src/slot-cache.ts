/**
 * Slot KV cache save/restore via llama-server's slot API.
 *
 * Enables fast session resume by persisting the KV cache to disk
 * instead of re-prefilling the full conversation history on the
 * llama-server.
 *
 * Requires:
 *   - llama-server started with `--slot-save-path <dir>`
 *   - CLAUDE_LLAMA_SERVER_URL env var (e.g. "http://192.168.124.14:8008")
 *   - CLAUDE_SLOT_CACHE_ENABLED=true
 *
 * ANTHROPIC_BASE_URL is not used: it may point at Strata, CCR, or any
 * OpenAI/Anthropic-compatible server that does not implement llama.cpp /slots.
 * When CLAUDE_LLAMA_SERVER_URL is unset, save/restore are no-ops.
 */

import { logger } from "./logger.js";

// The save request needs the slot lock server-side, so it queues behind any
// in-flight turn. With long agentic turns (minutes at large contexts) a short
// timeout loses the race whenever the next prompt/steer grabs the slot right
// at end_turn — the save then dies silently and no restore point is written.
// 90s + retries below cover the common starvation windows.
const SAVE_TIMEOUT_MS = 90_000;
const RESTORE_TIMEOUT_MS = 60_000;
const SLOT_INFO_TIMEOUT_MS = 5_000;
const SAVE_MAX_ATTEMPTS = 3;
const SAVE_RETRY_DELAYS_MS = [10_000, 60_000];

/** Sessions with a save currently running — dedupes rapid end_turn bursts. */
const savesInFlight = new Set<string>();

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type SlotCacheResult = {
  success: boolean;
  filename: string;
  nTokens: number;
  fileBytes: number;
  elapsedMs: number;
  error?: string;
};

export function isSlotCacheEnabled(): boolean {
  return (
    process.env.CLAUDE_SLOT_CACHE_ENABLED === "true" &&
    Boolean(process.env.CLAUDE_LLAMA_SERVER_URL)
  );
}

export function getLlamaServerUrl(): string | null {
  const url = process.env.CLAUDE_LLAMA_SERVER_URL;
  if (!url) return null;
  return url.replace(/\/v1\/?$/, "").replace(/\/$/, "");
}

/**
 * Auth header for the slot endpoint. llama.cpp's /slots is unauthenticated, but
 * Strata gates it behind the same API key as the chat API. Read the key from
 * CLAUDE_SLOT_CACHE_API_KEY, falling back to the token Claude Code already uses.
 */
function slotAuthHeaders(): Record<string, string> {
  const key =
    process.env.CLAUDE_SLOT_CACHE_API_KEY ||
    process.env.ANTHROPIC_AUTH_TOKEN ||
    process.env.ANTHROPIC_API_KEY;
  return key ? { Authorization: `Bearer ${key}` } : {};
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
        "CLAUDE_LLAMA_SERVER_URL not set",
    };
  }

  const t0 = Date.now();
  try {
    const res = await fetch(`${base}/slots/${slotId}?action=save`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...slotAuthHeaders() },
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
        "CLAUDE_LLAMA_SERVER_URL not set",
    };
  }

  const t0 = Date.now();
  try {
    const res = await fetch(`${base}/slots/${slotId}?action=restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...slotAuthHeaders() },
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
 * List all slots from llama-server.
 * Uses the `/slots` list endpoint — the single-slot `GET /slots/{id}`
 * endpoint returns 404 on some llama-server builds.
 * Returns null if the slot endpoint is unreachable (feature effectively off).
 */
export async function getAllSlots(): Promise<Array<{
  id: number;
  nPromptTokens: number;
  isProcessing: boolean;
}> | null> {
  const base = getLlamaServerUrl();
  if (!base) {
    return null;
  }

  try {
    const res = await fetch(`${base}/slots`, {
      method: "GET",
      headers: { ...slotAuthHeaders() },
      signal: AbortSignal.timeout(SLOT_INFO_TIMEOUT_MS),
    });
    if (!res.ok) {
      return null;
    }
    const data = (await res.json()) as Array<Record<string, unknown>>;
    return data.map((entry) => ({
      id: (entry["id"] as number) ?? 0,
      nPromptTokens: (entry["n_prompt_tokens"] as number) ?? 0,
      isProcessing: (entry["is_processing"] as boolean) ?? false,
    }));
  } catch {
    logger.info(`[slot-cache] slot list request failed`);
    return null;
  }
}

/** Save the llama-server slot KV cache for a session (fire-and-forget). */
export function saveSessionSlot(sessionId: string): void {
  if (!isSlotCacheEnabled()) {
    return;
  }
  // Rapid turn sequences (e.g. steer drains) can end several turns within
  // seconds; the first save captures the freshest server-side KV anyway.
  if (savesInFlight.has(sessionId)) {
    logger.info(
      `[slot-cache] save already in flight session=${sessionId} — skipping duplicate`,
    );
    return;
  }
  savesInFlight.add(sessionId);
  // Entry log: without this, a save that never reaches the server leaves no
  // trace in the adapter stderr at all (observed 2026-09-19: turns completed
  // for hours with zero bin files written and zero log lines).
  logger.info(`[slot-cache] save triggered session=${sessionId}`);
  void (async () => {
    try {
      for (let attempt = 1; attempt <= SAVE_MAX_ATTEMPTS; attempt++) {
        if (attempt > 1) {
          await sleep(SAVE_RETRY_DELAYS_MS[attempt - 2]);
        }
        const result = await saveSlotCache(0, `${sessionId}.bin`);
        if (result.success) {
          logger.info(
            `[slot-cache] saved session=${sessionId} tokens=${result.nTokens} file=${result.fileBytes}B (${result.elapsedMs}ms, attempt ${attempt}/${SAVE_MAX_ATTEMPTS})`,
          );
          if (result.nTokens < 1000) {
            // On a multi-slot server (--parallel N) the session's turn may
            // have run on a different slot; saving slot 0 then captures an
            // empty/foreign sequence. The restore side restores into every
            // empty slot, so a small save here is the signal to check this.
            logger.info(
              `[slot-cache] WARNING saved only ${result.nTokens} tokens — session may be on a non-zero slot`,
            );
          }
          return;
        }
        logger.info(
          `[slot-cache] save failed session=${sessionId} (attempt ${attempt}/${SAVE_MAX_ATTEMPTS}): ${result.error}`,
        );
      }
      logger.info(
        `[slot-cache] save gave up session=${sessionId} after ${SAVE_MAX_ATTEMPTS} attempts`,
      );
    } catch (e) {
      const err = e as Error | undefined;
      logger.error(
        `[slot-cache] save error session=${sessionId}: ${err?.message ?? String(e)}`,
      );
    } finally {
      savesInFlight.delete(sessionId);
    }
  })();
}

/**
 * Attempt to restore the llama-server slot KV cache for a session
 * (best-effort). On multi-slot servers (--parallel N) the bin is restored
 * into EVERY empty slot: after a server restart the session's first turn may
 * land on any slot, and a steer/concurrent request grabs whichever is free —
 * restoring only slot 0 leaves the other slots cold and forces a full
 * re-prefill there. A slot that already holds tokens is left alone.
 */
export async function tryRestoreSlot(sessionId: string): Promise<void> {
  if (!isSlotCacheEnabled()) {
    return;
  }
  try {
    const slots = await getAllSlots();
    // Only restore into empty slots (no prompt processed yet) — restoring
    // over a live slot is pointless churn. When the slot list is unavailable
    // (proxy without /slots support), fall back to trying slot 0 so the
    // failure mode is logged rather than silent.
    const targets = slots
      ? slots.filter((s) => s.nPromptTokens <= 1000).map((s) => s.id)
      : [0];
    if (slots && targets.length === 0) {
      logger.info(
        `[slot-cache] restore skipped session=${sessionId}: all ${slots.length} slots busy`,
      );
      return;
    }

    for (const slotId of targets) {
      const result = await restoreSlotCache(slotId, `${sessionId}.bin`);
      if (result.success) {
        logger.info(
          `[slot-cache] restored slot=${slotId} session=${sessionId} tokens=${result.nTokens} file=${result.fileBytes}B (${result.elapsedMs}ms)`,
        );
      } else {
        // Restore failed — file may not exist (first session) or is stale, the
        // URL points at a proxy without /slots support, or (kv-unified) the
        // shared cache has no room for a second copy. Expected; the normal
        // prefill path handles it.
        logger.info(
          `[slot-cache] restore skipped slot=${slotId} session=${sessionId}: ${result.error}`,
        );
      }
    }
  } catch (e) {
    const err = e as Error | undefined;
    logger.error(
      `[slot-cache] restore error session=${sessionId}: ${err?.message ?? String(e)}`,
    );
  }
}
