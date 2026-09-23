import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
}));

vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, _prompt: string, onEvent: (e: any) => void) => {
        onEvent({ type: "text_delta", text: "response" });
        return { text: "response", sessionId: "uuid-1" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_id: string, _prompt: string, onEvent: (e: any) => void) => {
        onEvent({ type: "text_delta", text: "continued" });
        return { text: "continued", sessionId: "uuid-1" };
      }
    );
    this.cancel = vi.fn();
    this.setSessionPermissionMode = vi.fn();
  });
  return { ClaudeRunner: MockClaudeRunner };
});

function createMockConnection(): AgentSideConnection {
  return {
    sessionUpdate: vi.fn().mockResolvedValue(undefined),
    requestPermission: vi.fn(),
    readTextFile: vi.fn(),
    writeTextFile: vi.fn(),
    createTerminal: vi.fn(),
    signal: new AbortController().signal,
    closed: new Promise(() => {}),
  } as unknown as AgentSideConnection;
}

describe("Agent modes and config", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  describe("newSession response", () => {
    it("should include modes in newSession response", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const result = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      expect(result.sessionId).toBeDefined();
      expect((result as any).modes).toBeDefined();
      expect((result as any).modes.availableModes).toHaveLength(3);
      expect((result as any).modes.currentModeId).toBe("agent");

      const modeIds = (result as any).modes.availableModes.map((m: any) => m.id);
      expect(modeIds).toContain("read-only");
      expect(modeIds).toContain("agent");
      expect(modeIds).toContain("agent-full-access");
    });

    it("should include configOptions in newSession response", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const result = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      expect((result as any).configOptions).toBeDefined();
      expect(Array.isArray((result as any).configOptions)).toBe(true);
    });
  });

  describe("setSessionMode", () => {
    it("should change session mode", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      const result = await agent.setSessionMode!({
        sessionId: session.sessionId,
        modeId: "architect",
      });

      expect(result).toBeDefined();
    });

    it("should send current_mode_update notification", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      await agent.setSessionMode!({
        sessionId: session.sessionId,
        modeId: "ask",
      });

      const calls = (conn.sessionUpdate as any).mock.calls;
      const modeUpdate = calls.find(
        (c: any) => c[0].update.sessionUpdate === "current_mode_update"
      );
      expect(modeUpdate).toBeDefined();
      expect(modeUpdate[0].update.currentModeId).toBe("ask");
    });

    it("should throw for unknown session", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      await expect(
        agent.setSessionMode!({
          sessionId: "unknown",
          modeId: "ask",
        })
      ).rejects.toThrow();
    });
  });

  describe("setSessionConfigOption", () => {
    it("should set a config option and return all options", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      const result = await agent.setSessionConfigOption!({
        sessionId: session.sessionId,
        configId: "thought_level",
        value: "high",
      } as any);

      expect(result).toBeDefined();
      expect((result as any).configOptions).toBeDefined();
      expect(Array.isArray((result as any).configOptions)).toBe(true);
    });

    it("should throw for unknown session", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      await expect(
        agent.setSessionConfigOption!({
          sessionId: "unknown",
          configId: "thought_level",
          value: "high",
        } as any)
      ).rejects.toThrow();
    });

    it("persists the mode config option so loadSession restores it", async () => {
      // Regression: setSessionConfigOption previously wrote the "mode" value
      // to configOverrides["mode"], which no read path consulted. loadSession
      // reads store.getMode() (session.mode) and so reset to DEFAULT_MODE on
      // reload/resume — the picker selection was not persisted.
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      const result = await agent.setSessionConfigOption!({
        sessionId: session.sessionId,
        configId: "mode",
        value: "agent-full-access",
      } as any);

      // The returned configOptions reflect the just-selected mode ...
      const modeOpt = (result as any).configOptions.find(
        (o: any) => o.id === "mode"
      );
      expect(modeOpt.currentValue).toBe("agent-full-access");

      // ... and loadSession restores it (the field that was being lost).
      const loaded = await agent.loadSession!({ sessionId: session.sessionId });
      expect((loaded as any).modes.currentModeId).toBe("agent-full-access");
    });
  });

  describe("listSessions", () => {
    it("should return list of sessions", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      await agent.newSession({ cwd: "/tmp", mcpServers: [] });
      await agent.newSession({ cwd: "/home", mcpServers: [] });

      const result = await agent.listSessions!({});

      expect(result.sessions).toHaveLength(2);
      expect(result.sessions[0].sessionId).toBeDefined();
      expect(result.sessions[0].cwd).toBeDefined();
    });

    it("should filter by cwd", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      await agent.newSession({ cwd: "/tmp", mcpServers: [] });
      await agent.newSession({ cwd: "/home", mcpServers: [] });
      await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      const result = await agent.listSessions!({ cwd: "/tmp" });

      expect(result.sessions).toHaveLength(2);
      expect(result.sessions.every((s: any) => s.cwd === "/tmp")).toBe(true);
    });
  });

  describe("loadSession", () => {
    it("should restore a session by id", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      // First prompt to set claudeSessionId
      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "hello" }],
      });

      const result = await agent.loadSession!({
        sessionId: session.sessionId,
        cwd: "/tmp",
        mcpServers: [],
      });

      expect(result).toBeDefined();
      expect((result as any).modes).toBeDefined();
    });

    it("rehydrates a session unknown to this process when cwd is given", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      // The agent is respawned per client session, so a resumed session is
      // routinely unknown to the new process. Throwing here made the resumed
      // conversation render empty; it must load instead.
      const res: any = await agent.loadSession!({
        sessionId: "unknown",
        cwd: "/tmp",
        mcpServers: [],
      });
      expect(res).toBeDefined();
      expect(res.modes).toBeDefined();
    });

    it("still throws when there is no cwd to rehydrate from", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      await expect(
        agent.loadSession!({
          sessionId: "unknown",
          mcpServers: [],
        } as any)
      ).rejects.toThrow();
    });
  });

  describe("initialize capabilities", () => {
    it("should have loadSession method available", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      expect(agent.loadSession).toBeDefined();
      expect(typeof agent.loadSession).toBe("function");
    });

    it("should advertise sessionCapabilities.list", async () => {
      const result = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      expect(result.agentCapabilities?.sessionCapabilities?.list).toBeDefined();
    });
  });
});
