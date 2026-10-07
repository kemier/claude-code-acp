import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { McpServerConfig } from "./claude-runner.js";
import { logger } from "./logger.js";

/**
 * Where session state lives.
 *
 * `CLAUDE_ACP_STORE_DIR` exists because the test suite writes this file for
 * real: any test that does not mock `node:fs` persisted into the developer's
 * own ~/.claude-code-acp/sessions.json, and a full `vitest run` replaced the
 * live map with fixture entries (`cwd:/tmp` → `uuid-perm`) — destroying the
 * cwd → Claude-session mapping of every real session on the machine. Measured:
 * a 16-session store reduced to one fixture record. tests/setup.ts points this
 * at a per-run temp dir so the suite can never touch live state again.
 */
const STORE_DIR = process.env.CLAUDE_ACP_STORE_DIR || join(homedir(), ".claude-code-acp");
const STORE_FILE = join(STORE_DIR, "sessions.json");

export interface SessionInfo {
  sessionId: string;
  cwd: string;
  title?: string;
  updatedAt?: string;
  /**
   * The underlying Claude CLI session UUID this ACP session replays, when
   * known. Used by listSessions to dedupe against CLI-spawned transcripts
   * scanned from ~/.claude/projects so the same conversation is not listed
   * twice (once as an ACP session, once as a raw transcript). Not part of
   * the ACP wire type — stripped before sending.
   */
  claudeSessionId?: string;
}

interface SessionData {
  cwd: string;
  /** Monotonic tool-call sequence, unique for the life of the session. */
  toolCallSeq?: number;
  claudeSessionId?: string;
  mcpServers: McpServerConfig[];
  createdAt: string;
  mode?: string;
  title?: string;
  updatedAt: string;
  configOverrides: Record<string, unknown>;
}

/**
 * On-disk shape.
 *
 * `sessions` keeps a FULL record per ACP session id. Without it the store was
 * in-memory only, so after the agent process restarted (VS Code window reload,
 * or a fresh spawn per session) `has()` returned false for a session the client
 * still knew about — `loadSession` threw resourceNotFound and the resumed
 * conversation rendered empty. It also makes `session/list` work across
 * restarts, which previously listed nothing.
 *
 * `byPersistKey` is the cwd→last-Claude-UUID mapping. Used when rehydrating an
 * unknown session via `ensure(..., { resumeLastClaude: true })` (loadSession
 * fallback). Intentionally NOT applied on a plain `create()` / `session/new` —
 * otherwise the ACP "+" new-conversation button keeps `--resume`-ing the old
 * Claude transcript in the same workspace.
 *
 * Legacy files are a bare `{ [persistKey]: claudeSessionId }` map; loadPersisted
 * migrates those on read.
 */
interface PersistedStore {
  version: 2;
  byPersistKey: { [key: string]: string };
  sessions: { [acpSessionId: string]: SessionData };
}

/**
 * Returns a stable key for persisting sessions.
 * Uses ACPX_SESSION_NAME (unique per thread/spawn) when available,
 * falls back to cwd (shared across threads with same directory).
 */
function getPersistKey(cwd: string): string {
  const sessionName = process.env.ACPX_SESSION_NAME;
  if (sessionName) {
    return `session:${sessionName}`;
  }
  return `cwd:${cwd}`;
}

export class SessionStore {
  private sessions = new Map<string, SessionData>();

  constructor() {
    // Rehydrate sessions written by a previous agent process. The agent is
    // respawned per client session (and on window reload), so without this the
    // map starts empty and every session the client tries to resume looks
    // unknown — loadSession threw resourceNotFound and the chat showed empty.
    try {
      const persisted = this.loadPersisted();
      for (const [sessionId, data] of Object.entries(persisted.sessions)) {
        if (data && typeof data.cwd === "string") {
          this.sessions.set(sessionId, {
            ...data,
            mcpServers: data.mcpServers ?? [],
            configOverrides: data.configOverrides ?? {},
          });
        }
      }
      if (this.sessions.size > 0) {
        logger.info(`Restored ${this.sessions.size} session(s) from ${STORE_FILE}`);
      }
    } catch (err) {
      logger.warn(
        `Could not restore persisted sessions: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  /**
   * Recreate an in-memory record for a session the client knows about but this
   * process does not (e.g. nothing persisted yet, or a different cwd). Falls
   * back to the cwd→Claude-UUID mapping so the transcript can still be found.
   * Returns the (possibly pre-existing) session's cwd.
   */
  ensure(sessionId: string, cwd: string, mcpServers: McpServerConfig[] = []): void {
    if (this.sessions.has(sessionId)) return;
    // loadSession rehydration: fall back to the last Claude UUID for this cwd
    // when the ACP session was never fully persisted.
    this.create(sessionId, cwd, mcpServers, { resumeLastClaude: true });
  }

  create(
    sessionId: string,
    cwd: string,
    mcpServers: McpServerConfig[] = [],
    opts: { resumeLastClaude?: boolean } = {},
  ): void {
    if (this.sessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} already exists`);
    }

    // Only attach the cwd→last-Claude mapping when explicitly requested
    // (ensure/loadSession). session/new must start with no Claude id so the
    // first prompt calls startSession (fresh conversation), not --resume.
    let claudeSessionId: string | undefined;
    if (opts.resumeLastClaude) {
      const persistKey = getPersistKey(cwd);
      const persisted = this.loadPersisted();
      claudeSessionId = persisted.byPersistKey[persistKey];
    }

    const now = new Date().toISOString();
    this.sessions.set(sessionId, {
      cwd,
      mcpServers,
      claudeSessionId,
      createdAt: now,
      updatedAt: now,
      configOverrides: {},
    });

    if (claudeSessionId) {
      logger.info(
        `Restored Claude Code session ${claudeSessionId} for ${getPersistKey(cwd)}`
      );
    }
    this.persistSessions();
  }

  getMcpServers(sessionId: string): McpServerConfig[] {
    return this.sessions.get(sessionId)?.mcpServers ?? [];
  }

  getCwd(sessionId: string): string | undefined {
    return this.sessions.get(sessionId)?.cwd;
  }

  /**
   * Next tool-call sequence number for a session — monotonic for its whole life.
   *
   * The counter used to be a local in the prompt loop, so it restarted at 1 every
   * turn AND was reset again whenever a steer message drained mid-turn. Clients
   * resolve tool elements by id: vscode-acp keys a map on the wire id and builds
   * DOM ids as 'tc-' + id, so a repeated call_1 made the new turn update the
   * PREVIOUS turn's row — the same defect found in ericli, where a cancelled turn
   * reached tc-8 and the replacement turn opened at tc-1 again.
   */
  nextToolCallSeq(sessionId: string): number {
    const session = this.sessions.get(sessionId);
    if (!session) {
      // No session (e.g. a late event after deletion): fall back to a clock-based
      // value, which is still unique.
      return Date.now();
    }
    session.toolCallSeq = (session.toolCallSeq ?? 0) + 1;
    return session.toolCallSeq;
  }

  setClaudeSessionId(sessionId: string, claudeId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.claudeSessionId = claudeId;

    // Persist to disk using stable key
    const persistKey = getPersistKey(session.cwd);
    this.savePersisted(persistKey, claudeId);
    this.persistSessions();
  }

  getClaudeSessionId(sessionId: string): string | undefined {
    return this.sessions.get(sessionId)?.claudeSessionId;
  }

  getMode(sessionId: string): string | undefined {
    return this.sessions.get(sessionId)?.mode;
  }

  setMode(sessionId: string, mode: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.mode = mode;
    session.updatedAt = new Date().toISOString();
    this.persistSessions();
  }

  getTitle(sessionId: string): string | undefined {
    return this.sessions.get(sessionId)?.title;
  }

  setTitle(sessionId: string, title: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.title = title;
    session.updatedAt = new Date().toISOString();
    this.persistSessions();
  }

  getUpdatedAt(sessionId: string): string | undefined {
    return this.sessions.get(sessionId)?.updatedAt;
  }

  touch(sessionId: string): void {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.updatedAt = new Date().toISOString();
    // Once per turn — cheap, and updatedAt is what orders session/list.
    this.persistSessions();
  }

  getConfigOverrides(sessionId: string): Record<string, unknown> {
    return this.sessions.get(sessionId)?.configOverrides ?? {};
  }

  setConfigOverride(sessionId: string, key: string, value: unknown): void {
    const session = this.sessions.get(sessionId);
    if (!session) {
      throw new Error(`Session ${sessionId} not found`);
    }
    session.configOverrides[key] = value;
    session.updatedAt = new Date().toISOString();
    this.persistSessions();
  }

  listAll(cwdFilter?: string): SessionInfo[] {
    const result: SessionInfo[] = [];
    for (const [sessionId, data] of this.sessions) {
      if (cwdFilter && data.cwd !== cwdFilter) continue;
      result.push({
        sessionId,
        cwd: data.cwd,
        title: data.title,
        updatedAt: data.updatedAt,
        claudeSessionId: data.claudeSessionId,
      });
    }
    return result;
  }

  delete(sessionId: string): void {
    this.sessions.delete(sessionId);
    this.persistSessions();
  }

  has(sessionId: string): boolean {
    return this.sessions.has(sessionId);
  }

  /** Clear the persisted Claude Code session for the current persist key. */
  clearPersistedSession(cwd: string): void {
    const persistKey = getPersistKey(cwd);
    const persisted = this.loadPersisted();
    delete persisted.byPersistKey[persistKey];
    this.writePersisted(persisted);
    logger.info(`Cleared persisted session for ${persistKey}`);
  }

  private loadPersisted(): PersistedStore {
    try {
      if (existsSync(STORE_FILE)) {
        const data = readFileSync(STORE_FILE, "utf-8");
        const parsed = JSON.parse(data);
        if (parsed && parsed.version === 2) {
          return {
            version: 2,
            byPersistKey: parsed.byPersistKey ?? {},
            sessions: parsed.sessions ?? {},
          };
        }
        // Legacy flat map: { "cwd:/x": "<claude-uuid>" }. Migrate in place.
        if (parsed && typeof parsed === "object") {
          const byPersistKey: { [k: string]: string } = {};
          for (const [k, v] of Object.entries(parsed)) {
            if (typeof v === "string") byPersistKey[k] = v;
          }
          return { version: 2, byPersistKey, sessions: {} };
        }
      }
    } catch {
      logger.warn(`Failed to load persisted sessions from ${STORE_FILE}`);
    }
    return { version: 2, byPersistKey: {}, sessions: {} };
  }

  private savePersisted(key: string, claudeSessionId: string): void {
    const persisted = this.loadPersisted();
    persisted.byPersistKey[key] = claudeSessionId;
    this.writePersisted(persisted);
    logger.debug(
      `Persisted Claude Code session ${claudeSessionId} for ${key}`
    );
  }

  /**
   * Write the full in-memory session set to disk. Called after every mutation
   * so a restarted agent process can answer session/load and session/list for
   * sessions the client still holds.
   */
  private persistSessions(): void {
    const persisted = this.loadPersisted();
    persisted.sessions = {};
    for (const [sessionId, data] of this.sessions) {
      persisted.sessions[sessionId] = data;
    }
    this.writePersisted(persisted);
  }

  private writePersisted(data: PersistedStore): void {
    try {
      if (!existsSync(STORE_DIR)) {
        mkdirSync(STORE_DIR, { recursive: true });
      }
      writeFileSync(STORE_FILE, JSON.stringify(data, null, 2));
    } catch (err) {
      logger.error(
        `Failed to write persisted sessions: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }
}
