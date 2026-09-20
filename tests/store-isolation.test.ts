/**
 * The test suite must never write the developer's real session store.
 *
 * Tests that do not mock node:fs persist for real, and a full `vitest run`
 * replaced ~/.claude-code-acp/sessions.json with fixture entries — wiping the
 * cwd → Claude-session mapping of every live session on the machine (measured:
 * 16 records down to one `cwd:/tmp` fixture). That mapping is the only handle on
 * a Claude conversation and the UI cannot restore it, so this is silent,
 * unrecoverable data loss triggered by running the tests.
 *
 * tests/setup.ts redirects the store to a temp dir via CLAUDE_ACP_STORE_DIR.
 * These guards fail if either half of that arrangement is removed.
 */

import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";

describe("session store isolation in tests", () => {
  it("redirects the store away from the real home dir", () => {
    const dir = process.env.CLAUDE_ACP_STORE_DIR;
    expect(dir, "tests/setup.ts must set CLAUDE_ACP_STORE_DIR").toBeTruthy();
    expect(dir!.startsWith(join(homedir(), ".claude-code-acp"))).toBe(false);
    expect(dir!.startsWith(tmpdir())).toBe(true);
  });

  it("session-store honours the override instead of hardcoding homedir", () => {
    const src = readFileSync(join(__dirname, "../src/session-store.ts"), "utf8");
    expect(src).toContain("process.env.CLAUDE_ACP_STORE_DIR");
    // The bare homedir() form must not be the store path any more.
    expect(src).not.toContain('const STORE_DIR = join(homedir(), ".claude-code-acp")');
  });

  it("the setup file sets the override before any store import", () => {
    const setup = readFileSync(join(__dirname, "setup.ts"), "utf8");
    expect(setup).toContain("CLAUDE_ACP_STORE_DIR");
    expect(setup).toContain("mkdtempSync");
  });
});
