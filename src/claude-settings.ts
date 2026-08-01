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
const MODEL_ENV_KEYS = [
  "ANTHROPIC_MODEL",
  "ANTHROPIC_SMALL_FAST_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
  "CLAUDE_ACP_MODEL",
] as const;

/** Optional comma-separated explicit picker list in settings env. */
const AVAILABLE_MODELS_ENV_KEY = "CLAUDE_ACP_AVAILABLE_MODELS";

let cachedSettings: LoadedClaudeSettings | undefined;

/**
 * Settings files in increasing order of precedence, mirroring Claude Code:
 * user, then shared project, then project-local.
 *
 * Only the user file and the project directory the CLI advertises via
 * `CLAUDE_PROJECT_DIR` are consulted. Per-session working directories are not,
 * because these values (context window, token budgets) are model-level
 * configuration rather than per-directory settings.
 */
function settingsPaths(): string[] {
  const paths = [join(homedir(), ".claude", "settings.json")];
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

/** Default effort from settings (`effortLevel`), if present. */
export function getDefaultEffortFromSettings(): string | undefined {
  return loadClaudeSettings().effortLevel;
}
