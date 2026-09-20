/**
 * Ensure agent tests that call newSession (and thus buildConfigOptions) have
 * at least one model configured. Production reads ~/.claude/settings.json;
 * most unit tests mock fs to "{}", so process.env is the source of truth here.
 */
import { beforeEach } from "vitest";
import { clearClaudeSettingsCache } from "../src/claude-settings.js";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// Redirect the session store away from ~/.claude-code-acp BEFORE any test
// imports session-store.js (STORE_DIR is resolved at module load).
//
// Tests that do not mock node:fs write this file for real. A full `vitest run`
// therefore overwrote the developer's live sessions.json with fixture entries
// and destroyed the cwd → Claude-session mapping of every real session — the
// mapping is the only handle on a Claude conversation, and nothing in the UI can
// restore it. Measured on a real machine: a 16-session store reduced to one
// `cwd:/tmp` fixture record.
process.env.CLAUDE_ACP_STORE_DIR =
  process.env.CLAUDE_ACP_STORE_DIR || mkdtempSync(join(tmpdir(), "claude-acp-test-store-"));

beforeEach(() => {
  clearClaudeSettingsCache();
  if (!process.env.ANTHROPIC_MODEL && !process.env.CLAUDE_ACP_MODEL) {
    process.env.ANTHROPIC_MODEL = "test-model";
  }
});
