// Tests for reading Claude Code CLI's own settings.
//
// Claude Code applies the `env` block of its settings files to its own process,
// and that block wins over inherited environment variables. The ACP process is
// the parent that spawns `claude`, so it never inherits those values — it has to
// read them. Without this, the ACP layer would compute its auto-compact
// threshold from different configuration than the CLI, and whichever fired
// first would win while the other silently never ran.
//
// Only the project layers are exercised directly here: the user layer lives at
// ~/.claude/settings.json, which these tests must not touch. Assertions use
// keys unique to this file so a real user settings file cannot affect them.

import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  getAvailableModelsFromSettings,
  getClaudeEnv,
  getDefaultEffortFromSettings,
  getDefaultModelFromSettings,
  loadClaudeSettingsEnv,
  clearClaudeSettingsCache,
  resolveModelAlias,
  MODEL_ENV_KEYS,
  AVAILABLE_MODELS_ENV_KEY,
} from "../src/claude-settings.js";

/** Key names unique to this test, so real settings files can't collide. */
const KEY = "ACP_TEST_ONLY_SETTING";
const KEY2 = "ACP_TEST_ONLY_SECOND";

let projectDir: string;
let userSettingsDir: string;
let savedProjectDir: string | undefined;
let savedSettingsDir: string | undefined;

function writeProjectSettings(file: string, settings: unknown): void {
  const dir = join(projectDir, ".claude");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, file), JSON.stringify(settings));
  clearClaudeSettingsCache(); // settings are cached after first read
}

beforeEach(() => {
  projectDir = mkdtempSync(join(tmpdir(), "acp-settings-test-"));
  // Isolate the user-layer settings file from the developer's real
  // ~/.claude/settings.json. CLAUDE_SETTINGS_DIR redirects settingsPaths() to
  // this empty throwaway dir so a real user file (which may set
  // ANTHROPIC_MODEL etc.) cannot leak into the assertions below.
  userSettingsDir = mkdtempSync(join(tmpdir(), "acp-settings-user-"));
  savedProjectDir = process.env.CLAUDE_PROJECT_DIR;
  savedSettingsDir = process.env.CLAUDE_SETTINGS_DIR;
  process.env.CLAUDE_PROJECT_DIR = projectDir;
  process.env.CLAUDE_SETTINGS_DIR = userSettingsDir;
  delete process.env[KEY];
  delete process.env[KEY2];
  clearClaudeSettingsCache();
});

afterEach(() => {
  if (savedProjectDir === undefined) delete process.env.CLAUDE_PROJECT_DIR;
  else process.env.CLAUDE_PROJECT_DIR = savedProjectDir;
  if (savedSettingsDir === undefined) delete process.env.CLAUDE_SETTINGS_DIR;
  else process.env.CLAUDE_SETTINGS_DIR = savedSettingsDir;
  delete process.env[KEY];
  delete process.env[KEY2];
  clearClaudeSettingsCache();
  rmSync(projectDir, { recursive: true, force: true });
  rmSync(userSettingsDir, { recursive: true, force: true });
});

describe("reading the settings env block", () => {
  it("reads a value from project settings", () => {
    writeProjectSettings("settings.json", { env: { [KEY]: "196608" } });
    expect(getClaudeEnv(KEY)).toBe("196608");
  });

  it("lets settings override the inherited environment", () => {
    // This is the CLI's behaviour: exporting a variable in the shell does not
    // beat settings.json, so predicting what the CLI will do means preferring
    // the settings value.
    process.env[KEY] = "from-shell";
    writeProjectSettings("settings.json", { env: { [KEY]: "from-settings" } });
    expect(getClaudeEnv(KEY)).toBe("from-settings");
  });

  it("falls back to the environment when settings do not define the key", () => {
    process.env[KEY] = "from-shell";
    writeProjectSettings("settings.json", { env: { [KEY2]: "other" } });
    expect(getClaudeEnv(KEY)).toBe("from-shell");
  });

  it("returns undefined when neither source defines the key", () => {
    writeProjectSettings("settings.json", { env: {} });
    expect(getClaudeEnv(KEY)).toBeUndefined();
  });

  it("gives settings.local.json precedence over settings.json", () => {
    writeProjectSettings("settings.json", { env: { [KEY]: "shared" } });
    writeProjectSettings("settings.local.json", { env: { [KEY]: "local" } });
    expect(getClaudeEnv(KEY)).toBe("local");
  });
});

describe("tolerating bad input", () => {
  it("ignores a malformed settings file rather than throwing", () => {
    const dir = join(projectDir, ".claude");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "settings.json"), "{ this is not json");
    clearClaudeSettingsCache();
    expect(() => loadClaudeSettingsEnv()).not.toThrow();
    process.env[KEY] = "fallback";
    expect(getClaudeEnv(KEY)).toBe("fallback");
  });

  it("ignores a missing env block", () => {
    writeProjectSettings("settings.json", { hooks: {} });
    expect(() => loadClaudeSettingsEnv()).not.toThrow();
    expect(getClaudeEnv(KEY)).toBeUndefined();
  });

  it("skips non-string values instead of coercing them", () => {
    // Claude Code only supports strings here; coercing would make the ACP layer
    // honor a setting the CLI itself would reject.
    writeProjectSettings("settings.json", {
      env: { [KEY]: 196608, [KEY2]: "196608" },
    });
    expect(getClaudeEnv(KEY)).toBeUndefined();
    expect(getClaudeEnv(KEY2)).toBe("196608");
  });

  it("survives a missing project directory", () => {
    process.env.CLAUDE_PROJECT_DIR = join(projectDir, "does-not-exist");
    clearClaudeSettingsCache();
    expect(() => loadClaudeSettingsEnv()).not.toThrow();
  });
});

describe("caching", () => {
  it("caches after the first read", () => {
    writeProjectSettings("settings.json", { env: { [KEY]: "first" } });
    expect(getClaudeEnv(KEY)).toBe("first");
    // Change the file without clearing the cache — the old value must persist,
    // so repeated lookups don't hit the filesystem on every token calculation.
    const dir = join(projectDir, ".claude");
    writeFileSync(join(dir, "settings.json"), JSON.stringify({ env: { [KEY]: "second" } }));
    expect(getClaudeEnv(KEY)).toBe("first");
  });

  it("picks up changes after the cache is cleared", () => {
    writeProjectSettings("settings.json", { env: { [KEY]: "first" } });
    expect(getClaudeEnv(KEY)).toBe("first");
    writeProjectSettings("settings.json", { env: { [KEY]: "second" } });
    expect(getClaudeEnv(KEY)).toBe("second");
  });
});

describe("model picker from settings", () => {
  // getAvailableModelsFromSettings / getDefaultModelFromSettings resolve model
  // ids via getClaudeEnv, which falls back to process.env when the settings env
  // block is silent. The developer's real environment commonly sets
  // ANTHROPIC_SMALL_FAST_MODEL (and other ANTHROPIC_DEFAULT_*_MODEL keys), so
  // scrubbing only ANTHROPIC_MODEL / CLAUDE_ACP_MODEL is not enough — those
  // sibling keys would leak in and pollute the picker. Save and clear every
  // model-naming env key for the duration of this suite.
  const savedEnv: Record<string, string | undefined> = {};
  const allModelKeys = [...MODEL_ENV_KEYS, AVAILABLE_MODELS_ENV_KEY];

  afterEach(() => {
    for (const k of allModelKeys) {
      if (savedEnv[k] === undefined) delete process.env[k];
      else process.env[k] = savedEnv[k];
    }
    clearClaudeSettingsCache();
  });

  beforeEach(() => {
    for (const k of allModelKeys) {
      savedEnv[k] = process.env[k];
      delete process.env[k];
    }
    clearClaudeSettingsCache();
  });

  it("lists unique models from ANTHROPIC_*_MODEL keys", () => {
    writeProjectSettings("settings.json", {
      env: {
        ANTHROPIC_MODEL: "LongCat-2.0",
        ANTHROPIC_SMALL_FAST_MODEL: "LongCat-2.0",
        ANTHROPIC_DEFAULT_OPUS_MODEL: "LongCat-2.0",
      },
    });
    expect(getAvailableModelsFromSettings().map((m) => m.id)).toEqual(["LongCat-2.0"]);
    expect(getDefaultModelFromSettings()).toBe("LongCat-2.0");
  });

  it("resolves top-level model alias via ANTHROPIC_DEFAULT_*_MODEL", () => {
    writeProjectSettings("settings.json", {
      model: "opus",
      env: { ANTHROPIC_DEFAULT_OPUS_MODEL: "LongCat-2.0" },
    });
    expect(resolveModelAlias("opus")).toBe("LongCat-2.0");
    expect(getDefaultModelFromSettings()).toBe("LongCat-2.0");
    expect(getAvailableModelsFromSettings().map((m) => m.id)).toEqual(["LongCat-2.0"]);
  });

  it("honors CLAUDE_ACP_AVAILABLE_MODELS when set", () => {
    writeProjectSettings("settings.json", {
      env: {
        ANTHROPIC_MODEL: "glm-5.2",
        CLAUDE_ACP_AVAILABLE_MODELS: "glm-5.2, Qwen3-Coder, deepseek-v4-pro",
      },
    });
    expect(getAvailableModelsFromSettings().map((m) => m.id)).toEqual([
      "glm-5.2",
      "Qwen3-Coder",
      "deepseek-v4-pro",
    ]);
  });

  it("reads effortLevel from settings", () => {
    writeProjectSettings("settings.json", {
      effortLevel: "high",
      env: { ANTHROPIC_MODEL: "LongCat-2.0" },
    });
    expect(getDefaultEffortFromSettings()).toBe("high");
  });

  it("throws when no model is configured", () => {
    writeProjectSettings("settings.json", { env: {} });
    expect(() => getDefaultModelFromSettings()).toThrow(/No model configured/);
    expect(() => getAvailableModelsFromSettings()).toThrow(/No models configured/);
  });
});

