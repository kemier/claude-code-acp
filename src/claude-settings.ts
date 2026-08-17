/**
 * Read Claude Code CLI's own settings so the ACP layer and the CLI agree.
 *
 * Claude Code applies the `env` block of its settings files to its own process
 * at startup — and it wins over inherited environment variables. That means a
 * value configured there (context window, output-token budget, compaction
 * thresholds) governs what the CLI actually does, while the ACP process, being
 * the *parent* that spawns `claude`, never sees it.
 *
 * Both layers can trigger compaction. If the ACP layer computed its threshold
 * from a different configuration than the CLI, whichever fired first would win
 * and the other would silently never run. Reading the CLI's settings here keeps
 * `~/.claude/settings.json` the single place to configure this behaviour.
 */

import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { logger } from "./logger.js";

/** Shape of the parts of settings.json we care about. */
type ClaudeSettings = {
  env?: Record<string, unknown>;
  model?: unknown;
  effortLevel?: unknown;
};

type LoadedClaudeSettings = {
  env: Record<string, string>;
  model?: string;
  effortLevel?: string;
};

/** Env keys that name concrete backend models in Claude Code settings. */
export const MODEL_ENV_KEYS = [
  "ANTHROPIC_MODEL",
  "ANTHROPIC_SMALL_FAST_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "CLAUDE_ACP_MODEL",
] as const;

/** Optional comma-separated explicit picker list in settings env. */
export const AVAILABLE_MODELS_ENV_KEY = "CLAUDE_ACP_AVAILABLE_MODELS";

let cachedSettings: LoadedClaudeSettings | undefined;

/**
 * Settings files in increasing order of precedence, mirroring Claude Code:
 * user, then shared project, then project-local.
 *
 * Only the user file and the project directory the CLI advertises via
 * `CLAUDE_PROJECT_DIR` are consulted. Per-session working directories are not,
 * because these values (context window, token budgets) are model-level
 * configuration rather than per-directory settings.
 *
 * The user-layer directory defaults to `~/.claude`. Setting the
 * `CLAUDE_SETTINGS_DIR` environment variable redirects it elsewhere — used by
 * tests to point at a throwaway directory so the developer's real
 * `~/.claude/settings.json` cannot leak into model-picker assertions, and
 * usable in production to layer the ACP adapter over a non-default settings
 * location.
 */
function settingsPaths(): string[] {
  const userDir = process.env.CLAUDE_SETTINGS_DIR ?? join(homedir(), ".claude");
  const paths = [join(userDir, "settings.json")];
  const projectDir = process.env.CLAUDE_PROJECT_DIR;
  if (projectDir) {
    paths.push(join(projectDir, ".claude", "settings.json"));
    paths.push(join(projectDir, ".claude", "settings.local.json"));
  }
  return paths;
}

function readSettingsFile(path: string): LoadedClaudeSettings {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return { env: {} }; // absent is the normal case, not an error
  }
  let parsed: ClaudeSettings;
  try {
    parsed = JSON.parse(raw) as ClaudeSettings;
  } catch (err) {
    logger.warn(
      `Ignoring malformed Claude Code settings at ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
    return { env: {} };
  }
  const env: Record<string, string> = {};
  if (parsed.env && typeof parsed.env === "object") {
    for (const [key, value] of Object.entries(parsed.env)) {
      // Claude Code only supports string values here; skip anything else rather
      // than coercing, so a mistyped setting fails visibly in the CLI too.
      if (typeof value === "string") env[key] = value;
    }
  }
  return {
    env,
    model: typeof parsed.model === "string" ? parsed.model : undefined,
    effortLevel: typeof parsed.effortLevel === "string" ? parsed.effortLevel : undefined,
  };
}

/** Merged Claude Code settings (env + model + effortLevel). Cached. */
export function loadClaudeSettings(): LoadedClaudeSettings {
  if (cachedSettings) return cachedSettings;
  const merged: LoadedClaudeSettings = { env: {} };
  for (const path of settingsPaths()) {
    const layer = readSettingsFile(path);
    Object.assign(merged.env, layer.env);
    if (layer.model !== undefined) merged.model = layer.model;
    if (layer.effortLevel !== undefined) merged.effortLevel = layer.effortLevel;
  }
  cachedSettings = merged;
  return merged;
}

/** The merged `env` block from Claude Code's settings files. Cached. */
export function loadClaudeSettingsEnv(): Record<string, string> {
  return loadClaudeSettings().env;
}

/** Drop the cache. Exposed for tests and for reloading after a settings edit. */
export function clearClaudeSettingsCache(): void {
  cachedSettings = undefined;
}

/**
 * Resolve a configuration variable the way Claude Code will see it.
 *
 * The settings `env` block takes precedence over the inherited environment,
 * matching the CLI: exporting a variable in the shell does not override
 * settings.json (verified against Claude Code v2.1.220).
 */
export function getClaudeEnv(key: string): string | undefined {
  const fromSettings = loadClaudeSettingsEnv()[key];
  if (fromSettings !== undefined) return fromSettings;
  return process.env[key];
}

/**
 * Resolve a Claude Code model alias (`opus` / `sonnet` / `haiku`) to the
 * concrete id configured in settings env, or return the value unchanged.
 */
export function resolveModelAlias(model: string): string {
  const aliases: Record<string, string | undefined> = {
    opus: getClaudeEnv("ANTHROPIC_DEFAULT_OPUS_MODEL"),
    sonnet: getClaudeEnv("ANTHROPIC_DEFAULT_SONNET_MODEL"),
    haiku: getClaudeEnv("ANTHROPIC_DEFAULT_HAIKU_MODEL"),
  };
  const resolved = aliases[model];
  return resolved || model;
}

/**
 * Default model for new sessions, matching Claude Code / settings precedence:
 * CLAUDE_ACP_MODEL → ANTHROPIC_MODEL → top-level settings.model (alias-resolved).
 */
export function getDefaultModelFromSettings(): string {
  const fromAcp = getClaudeEnv("CLAUDE_ACP_MODEL");
  if (fromAcp) return resolveModelAlias(fromAcp);

  const fromAnthropic = getClaudeEnv("ANTHROPIC_MODEL");
  if (fromAnthropic) return resolveModelAlias(fromAnthropic);

  const topLevel = loadClaudeSettings().model;
  if (topLevel) return resolveModelAlias(topLevel);

  throw new Error(
    "No model configured: set env.ANTHROPIC_MODEL (or CLAUDE_ACP_MODEL / settings.model) in ~/.claude/settings.json",
  );
}

/**
 * Models to advertise in the ACP picker.
 *
 * Precedence:
 *  1. `CLAUDE_ACP_AVAILABLE_MODELS` (comma-separated) when set in settings/env
 *  2. Unique concrete ids from the ANTHROPIC_*_MODEL / CLAUDE_ACP_MODEL keys
 *     (and the resolved top-level `model` alias)
 *
 * Throws when nothing is configured — the picker must not invent a catalog.
 */
export function getAvailableModelsFromSettings(): Array<{ id: string; name: string }> {
  const explicit = getClaudeEnv(AVAILABLE_MODELS_ENV_KEY);
  const ids: string[] = [];

  if (explicit) {
    for (const part of explicit.split(",")) {
      const id = part.trim();
      if (id) ids.push(resolveModelAlias(id));
    }
  } else {
    for (const key of MODEL_ENV_KEYS) {
      const value = getClaudeEnv(key);
      if (value) ids.push(resolveModelAlias(value));
    }
    const topLevel = loadClaudeSettings().model;
    if (topLevel) ids.push(resolveModelAlias(topLevel));
  }

  const unique: string[] = [];
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) continue;
    seen.add(id);
    unique.push(id);
  }

  if (unique.length === 0) {
    throw new Error(
      "No models configured for the ACP picker: set env.ANTHROPIC_MODEL or env.CLAUDE_ACP_AVAILABLE_MODELS in ~/.claude/settings.json",
    );
  }

  return unique.map((id) => ({ id, name: id }));
}

/**
 * Cached proxy model discovery result. Populated by discoverProxyModels()
 * and used as a fallback when settings.json has no explicit model list
 * (CLAUDE_ACP_AVAILABLE_MODELS not set).
 */
let proxyModelsCache: Array<{ id: string; name: string }> | null = null;

/**
 * Picker rules for the ACP model dropdown.
 *
 * A model is shown only when ALL of these hold:
 *   1. healthy — /v1/model-health reports healthy === true (replicas > 0,
 *      actually loaded by the vLLM backend; not configured-but-scaled-zero)
 *   2. lifecycle !== 'Deprecated' — retired models stay out of the dropdown
 *   3. tagged `coding` (agent mainline) OR `near_sota` (flagship models like
 *      deepseek-v4-flash / glm-5.2)
 *   4. NOT tagged `fast` — the fast-tagged variants are lighter chat models,
 *      not the agent mainline
 *
 * Pure exported helper so the tag/lifecycle combination can be unit tested
 * without a live proxy. Mirrors ericli's filterProxyModelsForPicker so all
 * ACP pickers (ericli, claude-code-acp, codex-acp via update_catalog.sh)
 * agree on the same list.
 */
export function filterProxyModelsForPicker(
  rawModels: Array<{ id?: string; tags?: string[] }>,
  healthById: ReadonlySet<string>,
): string[] {
  return rawModels
    .map((m) => {
      const tags = Array.isArray(m.tags) ? m.tags : [];
      return {
        id: m.id ?? "",
        // Show any coding-tagged model (agent mainline) and any healthy
        // flagship (near_sota): a model like deepseek-v4-flash carries the
        // fast tag on top of coding/near_sota and must NOT be hidden by it.
        // Fast-only chat models (gpt-oss-120b) still fail this test.
        isCoding: tags.includes("coding") || tags.includes("near_sota"),
        isHealthy: m.id ? healthById.has(m.id) : false,
      };
    })
    .filter((e) => e.id && e.isHealthy && e.isCoding)
    .map((e) => e.id)
    .sort();
}

/**
 * Fetch the available model list from the proxy, filtered for the picker.
 *
 * Pulls /v1/model-health (healthy + lifecycle) and /v1/models (tags), then
 * keeps only healthy, non-Deprecated, coding/near_sota-tagged (no fast)
 * models — the same rules as ericli's picker. Called during session
 * initialization so the picker shows new models without editing settings.json.
 *
 * Safe to call multiple times — returns the cached result after the first
 * successful fetch. A failed fetch leaves the cache null so settings.json
 * remains the fallback.
 */
export async function discoverProxyModels(): Promise<void> {
  if (proxyModelsCache) return;
  try {
    const baseUrl =
      process.env.ERICAI_PROXY_URL ||
      process.env.ERICAI_PROXY_BASE_URL ||
      "http://127.0.0.1:8317";
    // Health/lifecycle metadata comes from /v1/model-health (the same source
    // update_catalog.sh uses for the codex catalog). Without the health
    // filter, raw /v1/models would surface Deprecated models (DeepSeek-V3.2,
    // Qwen3-32B) and health-less variants in the picker.
    const healthResp = await fetch(`${baseUrl}/v1/model-health`, { signal: AbortSignal.timeout(3000) });
    if (!healthResp.ok) return;
    const healthBody = await healthResp.json() as { data?: Array<{
      id?: string;
      ericai_backend_id?: string;
      health?: { healthy?: boolean; lifecycle?: string; running_replicas?: number; status?: string };
    }> };
    const healthyById = new Set<string>();
    for (const m of healthBody.data ?? []) {
      const h = m.health ?? {};
      if (h.healthy === true && h.lifecycle !== "Deprecated") {
        if (m.id) healthyById.add(m.id);
        if (m.ericai_backend_id) healthyById.add(m.ericai_backend_id);
      }
    }
    // Only apply the healthy gate when model-health responded with data (so
    // a health endpoint that errors out doesn't ghost the picker).
    if (healthyById.size > 0) {
      const modelsResp = await fetch(`${baseUrl}/v1/models`, { signal: AbortSignal.timeout(3000) });
      if (!modelsResp.ok) return;
      const modelsBody = await modelsResp.json() as { data?: Array<{ id?: string; tags?: string[] }> };
      const ids = filterProxyModelsForPicker(modelsBody.data ?? [], healthyById);
      if (ids.length > 0) {
        proxyModelsCache = ids.map((id) => ({ id, name: id }));
        logger.info(`Discovered ${ids.length} healthy coding/near_sota models from proxy`);
      }
    }
  } catch {
    // Proxy unreachable — settings.json remains the model source
  }
}

/**
 * Get available models, preferring settings.json, falling back to the
 * proxy's /v1/models endpoint (if discoverProxyModels was called).
 * Throws if no models are configured.
 */
export function getAvailableModels(): Array<{ id: string; name: string }> {
  // If CLAUDE_ACP_AVAILABLE_MODELS is explicitly set, use it (highest priority)
  if (getClaudeEnv(AVAILABLE_MODELS_ENV_KEY)) {
    return getAvailableModelsFromSettings();
  }

  // If any model env keys are set, use settings-based list
  const hasModelEnv = MODEL_ENV_KEYS.some((key) => getClaudeEnv(key));
  const hasTopLevelModel = loadClaudeSettings().model;
  if (hasModelEnv || hasTopLevelModel) {
    return getAvailableModelsFromSettings();
  }

  // No models in settings — use proxy discovery cache
  if (proxyModelsCache && proxyModelsCache.length > 0) {
    return proxyModelsCache;
  }

  // Last resort: try settings (will throw with a helpful message)
  return getAvailableModelsFromSettings();
}

/** Default effort from settings (`effortLevel`), if present. */
export function getDefaultEffortFromSettings(): string | undefined {
  return loadClaudeSettings().effortLevel;
}
