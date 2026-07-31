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
};

let cachedEnv: Record<string, string> | undefined;

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

function readEnvBlock(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {}; // absent is the normal case, not an error
  }
  let parsed: ClaudeSettings;
  try {
    parsed = JSON.parse(raw) as ClaudeSettings;
  } catch (err) {
    logger.warn(
      `Ignoring malformed Claude Code settings at ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
    return {};
  }
  const env = parsed.env;
  if (!env || typeof env !== "object") return {};
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(env)) {
    // Claude Code only supports string values here; skip anything else rather
    // than coercing, so a mistyped setting fails visibly in the CLI too.
    if (typeof value === "string") out[key] = value;
  }
  return out;
}

/** The merged `env` block from Claude Code's settings files. Cached. */
export function loadClaudeSettingsEnv(): Record<string, string> {
  if (cachedEnv) return cachedEnv;
  const merged: Record<string, string> = {};
  for (const path of settingsPaths()) {
    Object.assign(merged, readEnvBlock(path));
  }
  cachedEnv = merged;
  return merged;
}

/** Drop the cache. Exposed for tests and for reloading after a settings edit. */
export function clearClaudeSettingsCache(): void {
  cachedEnv = undefined;
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
