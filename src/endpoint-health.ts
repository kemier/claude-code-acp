/**
 * Health-filter acpEndpoints (Windows/Ubuntu Strata dual-boot, etc.).
 *
 * Probes GET {baseUrl}/health with the same Bearer token Claude Code uses.
 * Results are cached briefly so picker rebuilds stay cheap.
 */

import { getClaudeEnv, type AcpEndpoint } from "./claude-settings.js";
import { logger } from "./logger.js";

const HEALTH_TIMEOUT_MS = 1_500;
const CACHE_TTL_MS = 20_000;

type CacheEntry = {
  at: number;
  healthyIds: string[];
};

let cache: CacheEntry | undefined;

/** Drop the health cache (tests / after settings reload). */
export function clearEndpointHealthCache(): void {
  cache = undefined;
}

function authHeaders(): Record<string, string> {
  const key =
    getClaudeEnv("ANTHROPIC_AUTH_TOKEN") ||
    getClaudeEnv("ANTHROPIC_API_KEY") ||
    process.env.ANTHROPIC_AUTH_TOKEN ||
    process.env.ANTHROPIC_API_KEY;
  return key ? { Authorization: `Bearer ${key}` } : {};
}

async function probeEndpoint(ep: AcpEndpoint): Promise<boolean> {
  const url = `${ep.baseUrl.replace(/\/+$/, "")}/health`;
  try {
    const res = await fetch(url, {
      method: "GET",
      headers: authHeaders(),
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!res.ok) return false;
    const body = (await res.json()) as { status?: string };
    return body?.status === "ok";
  } catch {
    return false;
  }
}

/**
 * Return the subset of endpoints whose /health reports status "ok".
 * Parallel probe; cached for CACHE_TTL_MS.
 */
export async function filterHealthyEndpoints(
  endpoints: AcpEndpoint[],
): Promise<AcpEndpoint[]> {
  if (endpoints.length === 0) return [];

  const now = Date.now();
  if (cache && now - cache.at < CACHE_TTL_MS) {
    const set = new Set(cache.healthyIds);
    return endpoints.filter((e) => set.has(e.id));
  }

  const results = await Promise.all(
    endpoints.map(async (ep) => {
      const ok = await probeEndpoint(ep);
      return { ep, ok };
    }),
  );

  const healthy = results.filter((r) => r.ok).map((r) => r.ep);
  cache = {
    at: now,
    healthyIds: healthy.map((e) => e.id),
  };

  const unhealthy = results.filter((r) => !r.ok).map((r) => r.ep.id);
  if (unhealthy.length > 0) {
    logger.info(
      `acpEndpoints health: healthy=[${healthy.map((e) => e.id).join(",") || "(none)"}] ` +
        `unhealthy=[${unhealthy.join(",")}]`,
    );
  }

  return healthy;
}
