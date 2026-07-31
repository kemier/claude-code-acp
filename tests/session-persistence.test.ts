import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

// SessionStore resolves its store path from homedir() at import time, so point
// HOME at a scratch dir BEFORE importing it.
let home: string;
home = mkdtempSync(join(tmpdir(), "ccacp-store-"));
process.env.HOME = home;
delete process.env.ACPX_SESSION_NAME;

const { SessionStore } = await import("../src/session-store.js");

const STORE_FILE = () => join(process.env.HOME!, ".claude-code-acp", "sessions.json");

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
    // ...but a new session in that cwd still resumes the old Claude conversation.
    store.create("acp-x", "/work/legacy");
    expect(store.getClaudeSessionId("acp-x")).toBe("legacy-uuid");

    const onDisk = JSON.parse(readFileSync(STORE_FILE(), "utf-8"));
    expect(onDisk.version).toBe(2);
    expect(onDisk.byPersistKey["cwd:/work/legacy"]).toBe("legacy-uuid");
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
