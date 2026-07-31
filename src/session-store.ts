import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import type { McpServerConfig } from "./claude-runner.js";
import { logger } from "./logger.js";

const STORE_DIR = join(homedir(), ".claude-code-acp");
const STORE_FILE = join(STORE_DIR, "sessions.json");

export interface SessionInfo {
  sessionId: string;
  cwd: string;
  title?: string;
  updatedAt?: string;
}

interface SessionData {
  cwd: string;
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
 * `byPersistKey` is the original cwd→Claude-UUID mapping, kept so a NEW session
 * in a known directory still resumes the last Claude conversation there.
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
    this.create(sessionId, cwd, mcpServers);
  }

  create(
    sessionId: string,
    cwd: string,
    mcpServers: McpServerConfig[] = []
  ): void {
    if (this.sessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} already exists`);
    }

    // Restore Claude Code session ID from disk if available
    const persistKey = getPersistKey(cwd);
    const persisted = this.loadPersisted();
    const claudeSessionId = persisted.byPersistKey[persistKey];

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
        `Restored Claude Code session ${claudeSessionId} for ${persistKey}`
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
