import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// SessionStore resolves its store path at import time, so point it at a scratch
// dir BEFORE importing it. CLAUDE_ACP_STORE_DIR is the documented override (see
// session-store.ts); HOME is still redirected for anything else that reads it.
let home: string;
home = mkdtempSync(join(tmpdir(), "ccacp-store-"));
process.env.HOME = home;
process.env.CLAUDE_ACP_STORE_DIR = join(home, ".claude-code-acp");
delete process.env.ACPX_SESSION_NAME;

const { SessionStore } = await import("../src/session-store.js");

const STORE_FILE = () => join(process.env.CLAUDE_ACP_STORE_DIR!, "sessions.json");

describe("SessionStore persistence across agent restarts", () => {
  beforeEach(() => {
    if (existsSync(STORE_FILE())) rmSync(STORE_FILE());
  });
  afterEach(() => {
    if (existsSync(STORE_FILE())) rmSync(STORE_FILE());
  });

  it("a new store instance sees sessions written by the previous one", () => {
    const first = new SessionStore();
    first.create("acp-1", "/work/proj");
    first.setClaudeSessionId("acp-1", "claude-uuid-1");
    first.setMode("acp-1", "architect");
    first.setTitle("acp-1", "My session");

    // Simulate the agent process being respawned (window reload / new session).
    const second = new SessionStore();
    expect(second.has("acp-1")).toBe(true);
    expect(second.getCwd("acp-1")).toBe("/work/proj");
    expect(second.getClaudeSessionId("acp-1")).toBe("claude-uuid-1");
    expect(second.getMode("acp-1")).toBe("architect");
    expect(second.getTitle("acp-1")).toBe("My session");
  });

  it("session/list works after a restart", () => {
    const first = new SessionStore();
    first.create("acp-1", "/work/a");
    first.create("acp-2", "/work/b");
    first.setClaudeSessionId("acp-1", "u1");

    const second = new SessionStore();
    expect(second.listAll().map((s) => s.sessionId).sort()).toEqual(["acp-1", "acp-2"]);
    expect(second.listAll("/work/b").map((s) => s.sessionId)).toEqual(["acp-2"]);
  });

  it("config overrides survive a restart", () => {
    const first = new SessionStore();
    first.create("acp-1", "/work/proj");
    first.setConfigOverride("acp-1", "model", "glm-5.2");
    first.setConfigOverride("acp-1", "effort", "high");

    const second = new SessionStore();
    expect(second.getConfigOverrides("acp-1")).toEqual({ model: "glm-5.2", effort: "high" });
  });

  it("deletion is persisted", () => {
    const first = new SessionStore();
    first.create("acp-1", "/work/proj");
    first.delete("acp-1");
    expect(new SessionStore().has("acp-1")).toBe(false);
  });

  it("migrates a legacy flat cwd→uuid store without losing the mapping", () => {
    // Pre-fix on-disk format.
    mkdirSync(join(process.env.HOME!, ".claude-code-acp"), { recursive: true });
    writeFileSync(STORE_FILE(), JSON.stringify({ "cwd:/work/legacy": "legacy-uuid" }, null, 2));

    const store = new SessionStore();
    // No full records existed, so nothing to rehydrate...
    expect(store.has("acp-x")).toBe(false);
    // session/new (plain create) must NOT inherit — otherwise "+" keeps the old chat.
    store.create("acp-fresh", "/work/legacy");
    expect(store.getClaudeSessionId("acp-fresh")).toBeUndefined();
    // loadSession rehydration (ensure) still resumes the last Claude UUID for the cwd.
    store.ensure("acp-x", "/work/legacy");
    expect(store.getClaudeSessionId("acp-x")).toBe("legacy-uuid");

    const onDisk = JSON.parse(readFileSync(STORE_FILE(), "utf-8"));
    expect(onDisk.version).toBe(2);
    expect(onDisk.byPersistKey["cwd:/work/legacy"]).toBe("legacy-uuid");
  });

  it("session/new create() does not attach the cwd's last Claude session", () => {
    const first = new SessionStore();
    first.create("acp-1", "/work/proj");
    first.setClaudeSessionId("acp-1", "old-claude-uuid");

    const second = new SessionStore();
    second.create("acp-2", "/work/proj");
    expect(second.getClaudeSessionId("acp-2")).toBeUndefined();
  });

  it("ensure() is a no-op for a session that already exists", () => {
    const store = new SessionStore();
    store.create("acp-1", "/work/proj");
    store.setClaudeSessionId("acp-1", "keep-me");
    store.ensure("acp-1", "/somewhere/else");
    expect(store.getCwd("acp-1")).toBe("/work/proj");
    expect(store.getClaudeSessionId("acp-1")).toBe("keep-me");
  });
});
