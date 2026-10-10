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

/** One labeled backend (e.g. Windows vs Ubuntu Strata) for the ACP model picker. */
export type AcpEndpoint = {
  id: string;
  name: string;
  model: string;
  baseUrl: string;
  /** Subagent model for sessions on this endpoint; injected as
   * CLAUDE_CODE_SUBAGENT_MODEL, overriding the global settings.env value
   * per endpoint. Lets a proxy endpoint route subagents to a different
   * machine (e.g. Mac Ollama) than the session's main model provider. */
  subagentModel?: string;
};

/** Shape of the parts of settings.json we care about. */
type ClaudeSettings = {
  env?: Record<string, unknown>;
  model?: unknown;
  effortLevel?: unknown;
  /** Dual-boot / multi-host backends. Later settings layers override by `id`. */
  acpEndpoints?: unknown;
};

type LoadedClaudeSettings = {
  env: Record<string, string>;
  model?: string;
  effortLevel?: string;
  acpEndpoints: AcpEndpoint[];
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

function parseAcpEndpoints(raw: unknown, source: string): AcpEndpoint[] {
  if (raw === undefined) return [];
  if (!Array.isArray(raw)) {
    logger.warn(`Ignoring non-array acpEndpoints in ${source}`);
    return [];
  }
  const out: AcpEndpoint[] = [];
  for (const entry of raw) {
    if (!entry || typeof entry !== "object") continue;
    const e = entry as Record<string, unknown>;
    const id = typeof e.id === "string" ? e.id.trim() : "";
    const name = typeof e.name === "string" ? e.name.trim() : "";
    const model = typeof e.model === "string" ? e.model.trim() : "";
    const baseUrl = typeof e.baseUrl === "string" ? e.baseUrl.trim().replace(/\/+$/, "") : "";
    if (!id || !model || !baseUrl) {
      logger.warn(
        `Ignoring incomplete acpEndpoints entry in ${source} (need id, model, baseUrl)`,
      );
      continue;
    }
    out.push({
      id,
      name: name || `${model} (${id})`,
      // Alias-resolved in loadClaudeSettings after env layers are merged.
      model,
      baseUrl,
      // Alias-resolved like model, below.
      subagentModel: typeof e.subagentModel === "string" && e.subagentModel.trim()
        ? e.subagentModel.trim()
        : undefined,
    });
  }
  return out;
}

/** Merge endpoint lists by id; later layers replace earlier ones with the same id. */
function mergeAcpEndpoints(into: AcpEndpoint[], layer: AcpEndpoint[]): AcpEndpoint[] {
  if (layer.length === 0) return into;
  const byId = new Map<string, AcpEndpoint>();
  for (const e of into) byId.set(e.id, e);
  for (const e of layer) byId.set(e.id, e);
  // Preserve first-seen order, append newly introduced ids at the end.
  const order: string[] = [];
  const seen = new Set<string>();
  for (const e of into) {
    if (!seen.has(e.id)) {
      order.push(e.id);
      seen.add(e.id);
    }
  }
  for (const e of layer) {
    if (!seen.has(e.id)) {
      order.push(e.id);
      seen.add(e.id);
    }
  }
  return order.map((id) => byId.get(id)!);
}

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "").toLowerCase();
}

function readSettingsFile(path: string): LoadedClaudeSettings {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return { env: {}, acpEndpoints: [] }; // absent is the normal case, not an error
  }
  // PowerShell's default UTF-8 writer prefixes a BOM. JSON.parse rejects it,
  // which used to drop the whole settings file and then fail session/load
  // with "No model configured".
  if (raw.charCodeAt(0) === 0xfeff) raw = raw.slice(1);
  let parsed: ClaudeSettings;
  try {
    parsed = JSON.parse(raw) as ClaudeSettings;
  } catch (err) {
    logger.warn(
      `Ignoring malformed Claude Code settings at ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
    return { env: {}, acpEndpoints: [] };
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
    // resolveModelAlias needs env; parse with raw model strings first, alias after merge.
    acpEndpoints: parseAcpEndpoints(parsed.acpEndpoints, path).map((e) => ({
      ...e,
      // Alias resolution deferred in loadClaudeSettings once env is merged.
      model: e.model,
    })),
  };
}

/** Merged Claude Code settings (env + model + effortLevel + acpEndpoints). Cached. */
export function loadClaudeSettings(): LoadedClaudeSettings {
  if (cachedSettings) return cachedSettings;
  const merged: LoadedClaudeSettings = { env: {}, acpEndpoints: [] };
  for (const path of settingsPaths()) {
    const layer = readSettingsFile(path);
    Object.assign(merged.env, layer.env);
    if (layer.model !== undefined) merged.model = layer.model;
    if (layer.effortLevel !== undefined) merged.effortLevel = layer.effortLevel;
    merged.acpEndpoints = mergeAcpEndpoints(merged.acpEndpoints, layer.acpEndpoints);
  }
  // Publish before alias resolution so resolveModelAlias → getClaudeEnv can
  // read the merged env without re-entering the load loop.
  cachedSettings = merged;
  merged.acpEndpoints = merged.acpEndpoints.map((e) => ({
    ...e,
    model: resolveModelAlias(e.model),
    subagentModel: e.subagentModel ? resolveModelAlias(e.subagentModel) : e.subagentModel,
  }));
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
 * CLAUDE_ACP_MODEL → ANTHROPIC_MODEL → top-level settings.model (alias-resolved)
 * → first acpEndpoints[].model.
 */
export function getDefaultModelFromSettings(): string {
  const fromAcp = getClaudeEnv("CLAUDE_ACP_MODEL");
  if (fromAcp) return resolveModelAlias(fromAcp);

  const fromAnthropic = getClaudeEnv("ANTHROPIC_MODEL");
  if (fromAnthropic) return resolveModelAlias(fromAnthropic);

  const topLevel = loadClaudeSettings().model;
  if (topLevel) return resolveModelAlias(topLevel);

  const endpoints = getAcpEndpoints();
  if (endpoints.length > 0) return endpoints[0].model;

  throw new Error(
    "No model configured: set env.ANTHROPIC_MODEL (or CLAUDE_ACP_MODEL / settings.model) in ~/.claude/settings.json",
  );
}

/** Configured labeled backends (Windows/Ubuntu Strata, etc.). Empty when unused. */
export function getAcpEndpoints(): AcpEndpoint[] {
  return loadClaudeSettings().acpEndpoints;
}

export function findAcpEndpoint(id: string): AcpEndpoint | undefined {
  if (!id) return undefined;
  return getAcpEndpoints().find((e) => e.id === id);
}

/**
 * Map a picker value to the backend model id. Endpoint ids resolve to their
 * `model`; plain model ids / aliases pass through resolveModelAlias.
 */
export function resolveBackendModel(pickerValue: string): string {
  const ep = findAcpEndpoint(pickerValue);
  if (ep) return ep.model;
  return resolveModelAlias(pickerValue);
}

/**
 * Default picker value when acpEndpoints is configured: prefer the endpoint
 * whose baseUrl matches a leftover settings.env ANTHROPIC_BASE_URL (migration
 * aid — process.env is ignored so a stale shell export cannot pin the dead
 * dual-boot host), else the first entry.
 */
export function getDefaultEndpointId(): string | undefined {
  const endpoints = getAcpEndpoints();
  if (endpoints.length === 0) return undefined;
  // settings.env only — not process.env (getClaudeEnv would also read the shell).
  const base = loadClaudeSettingsEnv()["ANTHROPIC_BASE_URL"];
  if (base) {
    const want = normalizeBaseUrl(base);
    const match = endpoints.find((e) => normalizeBaseUrl(e.baseUrl) === want);
    if (match) return match.id;
  }
  return endpoints[0].id;
}

/**
 * Spawn-env overrides for a picker selection. Sets ANTHROPIC_BASE_URL / MODEL
 * (and CLAUDE_LLAMA_SERVER_URL when slot-cache is on). Returns null when the
 * value is not an acpEndpoints id.
 *
 * Claude Code applies settings.json `env` over inherited process env — so
 * ANTHROPIC_BASE_URL must NOT remain in settings.env when using acpEndpoints,
 * or the CLI will ignore these overrides.
 */
export function buildEndpointSpawnEnv(
  pickerValue: string | undefined,
): Record<string, string> | null {
  if (!pickerValue) return null;
  const ep = findAcpEndpoint(pickerValue);
  if (!ep) return null;
  const out: Record<string, string> = {
    ANTHROPIC_BASE_URL: ep.baseUrl,
    ANTHROPIC_MODEL: ep.model,
  };
  // NOTE: the CLI applies settings.env OVER inherited process env (see the
  // static-ANTHROPIC_BASE_URL warning in agent.ts). For this per-endpoint
  // override to stick, settings.env must NOT set CLAUDE_CODE_SUBAGENT_MODEL —
  // keep it out of the env block and let the endpoint decide per session.
  if (ep.subagentModel) {
    out.CLAUDE_CODE_SUBAGENT_MODEL = ep.subagentModel;
  }
  if (getClaudeEnv("CLAUDE_SLOT_CACHE_ENABLED") === "true") {
    out.CLAUDE_LLAMA_SERVER_URL = ep.baseUrl;
  }
  return out;
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

/** Default effort from settings (`effortLevel`), if present. */
export function getDefaultEffortFromSettings(): string | undefined {
  return loadClaudeSettings().effortLevel;
}
