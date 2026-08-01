/**
 * Ensure agent tests that call newSession (and thus buildConfigOptions) have
 * at least one model configured. Production reads ~/.claude/settings.json;
 * most unit tests mock fs to "{}", so process.env is the source of truth here.
 */
import { beforeEach } from "vitest";
import { clearClaudeSettingsCache } from "../src/claude-settings.js";

beforeEach(() => {
  clearClaudeSettingsCache();
  if (!process.env.ANTHROPIC_MODEL && !process.env.CLAUDE_ACP_MODEL) {
    process.env.ANTHROPIC_MODEL = "test-model";
  }
});
