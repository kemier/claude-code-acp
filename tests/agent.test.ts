import { describe, it, expect, vi, beforeEach } from "vitest";
import { createClaudeCodeAgent, acpImageBlocksToStdinBlocks } from "../src/agent.js";
import type { AgentSideConnection } from "@agentclientprotocol/sdk";

// Captured runner calls for routing assertions (see mock below).
const capturedRunnerCalls: Array<{
  kind: string;
  prompt?: string;
  content?: unknown[];
}> = [];

// Mock node:fs so SessionStore doesn't read real persisted sessions
vi.mock("node:fs", () => ({
  existsSync: vi.fn(() => false),
  readFileSync: vi.fn(() => "{}"),
  writeFileSync: vi.fn(),
  mkdirSync: vi.fn(),
  mkdtempSync: vi.fn(() => "/tmp/claude-acp-test"),
}));

// Mock validation - agent unit tests should not hit filesystem
vi.mock("../src/validation.js", () => ({
  validateCwd: vi.fn((cwd: string) => cwd),
  validateMcpCommand: vi.fn(),
  validateMcpArgs: vi.fn(),
}));

// Mock ClaudeRunner
vi.mock("../src/claude-runner.js", () => {
  const MockClaudeRunner = vi.fn().mockImplementation(function (this: any) {
    this.startSession = vi.fn().mockResolvedValue({
      text: "Hello from Claude",
      sessionId: "claude-uuid-abc",
    });
    this.continueSession = vi.fn().mockResolvedValue({
      text: "Continued response",
      sessionId: "claude-uuid-abc",
    });
    this.startSessionStreamingWithContent = vi.fn().mockImplementation(
      async (_cwd: string, _content: unknown[], onEvent: (e: any) => void) => {
        capturedRunnerCalls.push({ kind: "startWithContent", content: _content });
        onEvent({ type: "text_delta", text: "Streamed with content" });
        return { text: "Streamed with content", sessionId: "claude-uuid-abc" };
      }
    );
    this.continueSessionStreamingWithContent = vi.fn().mockImplementation(
      async (_sessionId: string, _content: unknown[], onEvent: (e: any) => void) => {
        capturedRunnerCalls.push({ kind: "continueWithContent", content: _content });
        onEvent({ type: "text_delta", text: "Continued with content" });
        return { text: "Continued with content", sessionId: "claude-uuid-abc" };
      }
    );
    this.startSessionStreaming = vi.fn().mockImplementation(
      async (_cwd: string, prompt: string, onEvent: (e: any) => void) => {
        capturedRunnerCalls.push({ kind: "start", prompt });
        onEvent({ type: "text_delta", text: "Streamed response" });
        return { text: "Streamed response", sessionId: "claude-uuid-abc" };
      }
    );
    this.continueSessionStreaming = vi.fn().mockImplementation(
      async (_sessionId: string, prompt: string, onEvent: (e: any) => void) => {
        capturedRunnerCalls.push({ kind: "continue", prompt });
        onEvent({ type: "text_delta", text: "Continued streamed" });
        return { text: "Continued streamed", sessionId: "claude-uuid-abc" };
      }
    );
    this.cancel = vi.fn();
    this.discardLiveSession = vi.fn();
    this.writeSteerToMainTurn = vi.fn().mockReturnValue(false);
    this.hasOpenMainTurnStdin = vi.fn().mockReturnValue(false);
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

describe("ClaudeCodeAgent", () => {
  let conn: AgentSideConnection;
  let agent: ReturnType<typeof createClaudeCodeAgent>;

  beforeEach(() => {
    vi.clearAllMocks();
    capturedRunnerCalls.length = 0;
    conn = createMockConnection();
    agent = createClaudeCodeAgent(conn);
  });

  describe("initialize", () => {
    it("should return protocol version and capabilities", async () => {
      const result = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test-client", version: "1.0" },
      });

      expect(result.protocolVersion).toBe(1);
      expect(result.agentInfo?.name).toBe("claude-code-acp");
      expect(result.agentCapabilities).toBeDefined();
      expect(result.agentCapabilities?.loadSession).toBe(true);
    });

    it("should advertise image prompt capability", async () => {
      const result = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      expect(
        result.agentCapabilities?.promptCapabilities as unknown as { image: boolean; embeddedContext: boolean }
      ).toMatchObject({ image: true, embeddedContext: true });
    });

    it("should advertise no auth methods", async () => {
      const result = await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      expect(result.authMethods).toEqual([]);
    });
  });

  describe("newSession", () => {
    it("should create a session and return sessionId", async () => {
      // Must initialize first
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      const result = await agent.newSession({
        cwd: "/Users/test/project",
        mcpServers: [],
      });

      expect(result.sessionId).toBeDefined();
      expect(typeof result.sessionId).toBe("string");
      expect(result.sessionId.length).toBeGreaterThan(0);
    });

    it("should return unique session ids", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });

      const r1 = await agent.newSession({ cwd: "/tmp", mcpServers: [] });
      const r2 = await agent.newSession({ cwd: "/tmp", mcpServers: [] });

      expect(r1.sessionId).not.toBe(r2.sessionId);
    });
  });

  describe("authenticate", () => {
    it("should return empty response (no-op)", async () => {
      const result = await agent.authenticate({ methodId: "test" });
      expect(result).toEqual({});
    });
  });

  describe("prompt", () => {
    it("should send prompt to Claude and return completed stop reason", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      const result = await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "Hello" }],
      });

      expect(result.stopReason).toBe("end_turn");
    });

    it("should send session update with agent message", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "Hello" }],
      });

      // Should have called sessionUpdate at least once
      expect(conn.sessionUpdate).toHaveBeenCalled();
    });

    it("should throw for unknown session", async () => {
      await expect(
        agent.prompt({
          sessionId: "nonexistent",
          prompt: [{ type: "text", text: "Hello" }],
        })
      ).rejects.toThrow();
    });

    it("should use continueSession on second prompt", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      // First prompt
      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "First" }],
      });

      // Second prompt should use continueSession
      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "Second" }],
      });

      // Verify sessionUpdate was called for both prompts
      expect((conn.sessionUpdate as any).mock.calls.length).toBeGreaterThanOrEqual(2);
    });

    it("should route an image prompt through startSessionStreamingWithContent", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      const result = await agent.prompt({
        sessionId: session.sessionId,
        prompt: [
          { type: "text", text: "What's in this picture?" },
          {
            type: "image",
            data: "aGVsbG8=",
            mimeType: "image/png",
          },
        ],
      });

      expect(result.stopReason).toBe("end_turn");
      // The stream-json stdin path must be used (not the text-only one).
      const withContentCalls = capturedRunnerCalls.filter(
        (c) => c.kind === "startWithContent"
      );
      expect(withContentCalls).toHaveLength(1);
      expect(capturedRunnerCalls.some((c) => c.kind === "start")).toBe(false);

      // The image block is forwarded as an Anthropic image block. A text-only
      // backend does not see the pixels: the proxy vision-model describes them.
      // The text block carries the <collaboration_mode> Default-mode preamble
      // the adapter prepends + the user's question.
      const [textBlock, imageBlock] = withContentCalls[0].content;
      expect(textBlock.type).toBe("text");
      expect(textBlock.text).toContain("What's in this picture?");
      expect(textBlock.text).toContain("<collaboration_mode>");
      // Vision-capable harness model: the image block is forwarded, NOT degraded.
      expect(imageBlock).toEqual({
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
      });
    });

    it("inlines an editor selection resource into the CLI prompt", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [
          {
            type: "resource",
            resource: {
              uri: "file:///workspace/cta_scheduler.cu#L8:8",
              text: "__device__ void schedule() {}",
            },
          },
          { type: "text", text: "讲一下这段的原理" },
        ] as any,
      });

      const call = capturedRunnerCalls.find((c) => c.kind === "startWithContent");
      expect(call).toBeDefined();
      const body = (call!.content as Array<{ text?: string }>)[0].text ?? "";
      expect(body).toContain('<ide_selection path="/workspace/cta_scheduler.cu" lines="8">');
      expect(body).toContain("__device__ void schedule() {}");
      expect(body).toContain("讲一下这段的原理");
    });

    it("forwards an image even when the session model is text-only", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });
      // Text-only name. The image still goes to the CLI; the proxy describes it.
      agent.setSessionConfigOption({
        sessionId: session.sessionId,
        configId: "model",
        value: "gpt-3.5-turbo",
      } as any);

      const result = await agent.prompt({
        sessionId: session.sessionId,
        prompt: [
          { type: "text", text: "What's in this picture?" },
          {
            type: "image",
            data: "aGVsbG8=",
            mimeType: "image/png",
          },
        ],
      });

      expect(result.stopReason).toBe("end_turn");
      const withContentCalls = capturedRunnerCalls.filter(
        (c) => c.kind === "startWithContent"
      );
      expect(withContentCalls).toHaveLength(1);
      const [textBlock, imageBlock] = withContentCalls[0].content;
      expect(textBlock.type).toBe("text");
      expect(textBlock.text).toContain("What's in this picture?");
      expect(imageBlock).toEqual({
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
      });
    });

    it("should route text-only prompts through the stream-json stdin path", async () => {
      await agent.initialize({
        protocolVersion: 1,
        clientCapabilities: {},
        clientInfo: { name: "test", version: "1.0" },
      });
      const session = await agent.newSession({
        cwd: "/tmp",
        mcpServers: [],
      });

      await agent.prompt({
        sessionId: session.sessionId,
        prompt: [{ type: "text", text: "Just text" }],
      });

      // All main turns (text-only and multimodal) take the stream-json stdin
      // path so the CLI's stdin stays open for the duration of the turn and a
      // mid-turn steer can be written into the in-flight turn. Text-only
      // prompts are therefore delivered via startSessionStreamingWithContent
      // (a text content block), NOT the argv-prompt startSessionStreaming.
      expect(
        capturedRunnerCalls.filter((c) => c.kind === "startWithContent")
      ).toHaveLength(1);
      expect(capturedRunnerCalls.some((c) => c.kind === "start")).toBe(false);
    });
  });

  describe("cancel", () => {
    it("should cancel without throwing", async () => {
      await expect(
        agent.cancel({ sessionId: "some-session" })
      ).resolves.not.toThrow();
    });
  });
});

describe("acpImageBlocksToStdinBlocks", () => {
  it("returns an empty array when there are no image blocks", () => {
    expect(acpImageBlocksToStdinBlocks([])).toEqual([]);
    expect(acpImageBlocksToStdinBlocks([{ type: "text", text: "hi" }])).toEqual([]);
  });

  it("converts a single image block to an Anthropic image source", () => {
    expect(
      acpImageBlocksToStdinBlocks([
        { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
      ])
    ).toEqual([
      {
        type: "image",
        source: { type: "base64", media_type: "image/png", data: "aGVsbG8=" },
      },
    ]);
  });

  it("defaults mimeType to image/png when omitted", () => {
    const out = acpImageBlocksToStdinBlocks([{ type: "image", data: "QUJD" }]);
    expect(out[0]?.source).toMatchObject({ media_type: "image/png" });
  });

  it("skips image blocks with empty/missing base64 data", () => {
    expect(
      acpImageBlocksToStdinBlocks([
        { type: "image", data: "" },
        { type: "image", data: undefined },
        { type: "image", data: "aGk=", mimeType: "image/webp" },
      ])
    ).toEqual([
      {
        type: "image",
        source: { type: "base64", media_type: "image/webp", data: "aGk=" },
      },
    ]);
  });

  it("keeps image order and ignores non-image blocks", () => {
    const out = acpImageBlocksToStdinBlocks([
      { type: "text", text: "ignore me" },
      { type: "image", data: "QUJD", mimeType: "image/jpeg" },
      { type: "resource_link", uri: "file:///x" as unknown as string, name: "x" },
      { type: "image", data: "REVG", mimeType: "image/png" },
    ]);
    expect(out).toHaveLength(2);
    expect(out[0]?.source).toMatchObject({ media_type: "image/jpeg", data: "QUJD" });
    expect(out[1]?.source).toMatchObject({ media_type: "image/png", data: "REVG" });
  });
});
