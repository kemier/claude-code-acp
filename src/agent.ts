import {
  PROTOCOL_VERSION,
  RequestError,
  type Agent,
  type AgentSideConnection,
  type InitializeRequest,
  type InitializeResponse,
  type NewSessionRequest,
  type NewSessionResponse,
  type AuthenticateRequest,
  type AuthenticateResponse,
  type PromptRequest,
  type PromptResponse,
  type CancelNotification,
  type LoadSessionRequest,
  type LoadSessionResponse,
  type ListSessionsRequest,
  type ListSessionsResponse,
  type SetSessionModeRequest,
  type SetSessionModeResponse,
  type SetSessionConfigOptionRequest,
  type SetSessionConfigOptionResponse,
} from "@agentclientprotocol/sdk";
import { SessionStore } from "./session-store.js";
import { ClaudeRunner, type StreamEvent } from "./claude-runner.js";
import { logger } from "./logger.js";
import {
  validateCwd,
  validateMcpCommand,
  validateMcpArgs,
} from "./validation.js";

function generateSessionId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

const AVAILABLE_MODES = [
  { id: "code", name: "Code", description: "Write and edit code" },
  { id: "ask", name: "Ask", description: "Ask questions without making changes" },
  { id: "architect", name: "Architect", description: "Plan and design architecture" },
];

const DEFAULT_MODE = "code";

function buildModeState(currentModeId: string) {
  return {
    availableModes: AVAILABLE_MODES,
    currentModeId,
  };
}

/** Models available through ericai-proxy (queried from /v1/models). */
const PROXY_MODELS = [
  { id: "glm-5.2", name: "GLM-5.2" },
  { id: "glm-5.1", name: "GLM-5.1" },
  { id: "Qwen3-Coder", name: "Qwen3-Coder" },
  { id: "deepseek-v4-pro", name: "DeepSeek-V4-Pro" },
  { id: "kimi-k2.7-code", name: "Kimi-K2.7-Code" },
  { id: "kimi-k2.6", name: "Kimi-K2.6" },
  { id: "mimo-v2.5-pro", name: "MiMo-V2.5-Pro" },
  { id: "mimo-v2.5", name: "MiMo-V2.5" },
  { id: "qwen2.5-14b-1m", name: "Qwen2.5-14B-1M" },
];

/** All possible effort levels (Claude Code CLI --effort flag). */
const ALL_EFFORT_LEVELS: Record<string, { name: string; description: string }> = {
  low: { name: "Low", description: "Fast, minimal reasoning" },
  medium: { name: "Medium", description: "Balanced reasoning" },
  high: { name: "High", description: "Deep reasoning" },
  xhigh: { name: "XHigh", description: "Extra-deep reasoning" },
  max: { name: "Max", description: "Maximum reasoning effort" },
};

/**
 * Per-model effort support. Must match reasoning_profiles in
 * ericai-proxy/scripts/update_catalog.sh.
 * - Models with thinking support get their specific effort levels.
 * - Models without thinking get an empty array (effort picker hidden).
 */
const MODEL_EFFORTS: Record<string, string[]> = {
  "glm-5.2": ["low", "medium", "high"],
  "glm-5.1": ["low", "medium", "high"],
  "Qwen3-Coder": ["low", "medium", "high"],
  "deepseek-v4-pro": ["low", "medium", "high", "xhigh"],
  "kimi-k2.7-code": ["low", "high"],
  "kimi-k2.6": ["low", "high"],
  "mimo-v2.5-pro": ["low", "high"],
  "mimo-v2.5": ["low", "high"],
  // qwen2.5-14b-1m is a non-reasoning model — no effort picker
  "qwen2.5-14b-1m": [],
};

/** Default effort per model (first entry in MODEL_EFFORTS, or undefined). */
function getDefaultEffort(modelId: string): string | undefined {
  const efforts = MODEL_EFFORTS[modelId];
  if (!efforts || efforts.length === 0) return undefined;
  // GLM/DeepSeek default to high; Kimi/MiMo default to high
  return efforts.includes("high") ? "high" : efforts[0];
}

/** Built-in slash commands (mirrors codex-acp's CodexCommands). */
const BUILTIN_COMMANDS = [
  {
    name: "ps",
    description: "List background terminal processes.",
    input: null,
  },
  {
    name: "goal",
    description: "Set a goal to keep pursuing.",
    input: { hint: "[<objective>|clear|pause|resume]" },
  },
  {
    name: "skills",
    description: "List available skills.",
    input: null,
  },
  {
    name: "status",
    description: "Display session configuration and token usage.",
    input: null,
  },
];

/** Per-session goal state. */
const sessionGoals = new Map<string, { text: string; paused: boolean }>();

/** Parse a slash command from prompt text. Returns null if not a command. */
function parseSlashCommand(prompt: string): { name: string; rest: string } | null {
  const trimmed = prompt.trim();
  if (!trimmed.startsWith("/")) return null;
  const commandText = trimmed.slice(1).trim();
  if (!commandText) return null;
  const [name, ...restParts] = commandText.split(/\s+/);
  if (!name) return null;
  return {
    name: name.toLowerCase(),
    rest: restParts.join(" ").trim(),
  };
}

/** Check if a prompt is a read-only slash command (no LLM turn needed). */
function isReadOnlyCommand(name: string): boolean {
  return name === "ps" || name === "status" || name === "skills";
}

function buildConfigOptions(store: SessionStore, sessionId: string) {
  const modelOverride = store.getConfigOverrides(sessionId)["model"] as string | undefined;
  const currentModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
  const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
  const effortOverride = store.getConfigOverrides(sessionId)["effort"] as string | undefined;

  // Get effort levels supported by the CURRENT model
  const modelEfforts = MODEL_EFFORTS[currentModel] ?? [];
  // If the stored effort is not supported by this model, fall back to default
  const defaultEffort = getDefaultEffort(currentModel);
  const currentEffort =
    effortOverride && modelEfforts.includes(effortOverride)
      ? effortOverride
      : defaultEffort;

  const configOptions: any[] = [
    {
      id: "mode",
      name: "Mode",
      type: "select" as const,
      category: "mode",
      description: "Agent mode",
      options: AVAILABLE_MODES.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      })),
      currentValue: currentMode,
    },
    {
      id: "model",
      name: "Model",
      type: "select" as const,
      category: "model",
      description: "EricAI model (via ericai-proxy)",
      options: PROXY_MODELS.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.id,
      })),
      currentValue: currentModel,
    },
  ];

  // Only show effort picker if the model supports thinking
  if (modelEfforts.length > 0) {
    configOptions.push({
      id: "effort",
      name: "Effort",
      type: "select" as const,
      category: "effort",
      description: "Thinking effort level",
      options: modelEfforts.map((e) => ({
        value: e,
        name: ALL_EFFORT_LEVELS[e]?.name ?? e,
        description: ALL_EFFORT_LEVELS[e]?.description ?? "",
      })),
      currentValue: currentEffort,
    });
  }

  return configOptions;
}

export function createClaudeCodeAgent(
  connection: AgentSideConnection
): Agent {
  const store = new SessionStore();
  const runner = new ClaudeRunner();
  const cancelledSessions = new Set<string>();
  const steerQueues = new Map<string, string[]>();

  /**
   * Steering notification — injects a user message into the active turn.
   * Claude CLI doesn't support mid-turn injection, so we queue the message
   * and send it as a follow-up --resume prompt when the current process
   * finishes. The user_message_chunk is emitted immediately so the IDE
   * shows the steering message as a user bubble (visual context).
   *
   * Custom ACP extension: `session/steer`.
   */
  async function handleSteer(params: { sessionId: string; text: string }): Promise<void> {
    const { sessionId, text } = params;
    if (!store.has(sessionId)) {
      logger.warn(`Steer request for unknown session ${sessionId}`);
      return;
    }
    if (!runner.isRunning(sessionId)) {
      logger.info(`Steer request for idle session ${sessionId} — ignoring (no active turn)`);
      return;
    }
    // Queue the message for delivery after the current turn finishes.
    if (!steerQueues.has(sessionId)) {
      steerQueues.set(sessionId, []);
    }
    steerQueues.get(sessionId)!.push(text);
    logger.info(`Steer queued for session ${sessionId}: ${text.length} chars`);
    // Emit a user_message_chunk so the IDE shows the steering message
    // as a user bubble in the chat (visual context).
    try {
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "user_message_chunk",
          content: { type: "text", text },
        } as any,
      });
    } catch {
      // ignore — best effort visual feedback
    }
  }

  return {
    async initialize(
      _params: InitializeRequest
    ): Promise<InitializeResponse> {
      logger.info("Initialize request received");
      return {
        protocolVersion: PROTOCOL_VERSION,
        agentInfo: {
          name: "claude-code-acp",
          title: "Claude Code",
          version: "0.1.0",
        },
        agentCapabilities: {
          loadSession: false,
          promptCapabilities: {
            image: false,
            audio: false,
            embeddedContext: false,
          },
          mcpCapabilities: {
            http: false,
            sse: false,
          },
          sessionCapabilities: {
            list: {},
          },
        },
        authMethods: [],
      };
    },

    async newSession(
      params: NewSessionRequest
    ): Promise<NewSessionResponse> {
      logger.debug(`newSession params: ${JSON.stringify(params)}`);
      // Validate cwd
      let resolvedCwd: string;
      try {
        resolvedCwd = validateCwd(params.cwd);
      } catch (err) {
        throw RequestError.invalidParams(
          `Invalid cwd: ${err instanceof Error ? err.message : String(err)}`
        );
      }

      const sessionId = generateSessionId();
      // Convert and validate ACP MCP server format
      const mcpServers = (params.mcpServers ?? [])
        .filter((s: any) => s.transport?.type === "stdio")
        .map((s: any) => {
          const command = s.transport.command;
          const args = s.transport.args ?? [];
          try {
            validateMcpCommand(command);
            validateMcpArgs(args);
          } catch (err) {
            throw RequestError.invalidParams(
              `Invalid MCP server config: ${err instanceof Error ? err.message : String(err)}`
            );
          }
          return {
            name: s.name ?? s.id ?? "unknown",
            transport: {
              type: "stdio" as const,
              command,
              args,
              env: s.transport.env,
            },
          };
        });
      store.create(sessionId, resolvedCwd, mcpServers);
      logger.info(
        `Session created: ${sessionId} (cwd: ${resolvedCwd}, mcpServers: ${mcpServers.length})`
      );

      // Send available_commands_update notification so vscode-acp picks up
      // the slash commands. vscode-acp's SessionManager ignores the
      // availableCommands field in the newSession response (hardcodes to []),
      // and only populates via the available_commands_update notification.
      connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "available_commands_update",
          availableCommands: BUILTIN_COMMANDS,
        } as any,
      }).catch(() => {
        // Best effort — notification may arrive before session is registered,
        // vscode-acp buffers it via pendingAvailableCommands.
      });

      return {
        sessionId,
        modes: buildModeState(DEFAULT_MODE),
        configOptions: buildConfigOptions(store, sessionId),
        availableCommands: BUILTIN_COMMANDS,
      } as NewSessionResponse;
    },

    async loadSession(
      params: LoadSessionRequest
    ): Promise<LoadSessionResponse> {
      const { sessionId } = params;

      if (!store.has(sessionId)) {
        throw RequestError.resourceNotFound(
          `Session ${sessionId} not found`
        );
      }

      const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
      logger.info(`Load session: ${sessionId}`);

      return {
        modes: buildModeState(currentMode),
        configOptions: buildConfigOptions(store, sessionId),
        availableCommands: BUILTIN_COMMANDS,
      } as LoadSessionResponse;
    },

    async listSessions(
      params: ListSessionsRequest
    ): Promise<ListSessionsResponse> {
      const sessions = store.listAll(params.cwd ?? undefined);
      return { sessions };
    },

    async setSessionMode(
      params: SetSessionModeRequest
    ): Promise<SetSessionModeResponse> {
      const { sessionId, modeId } = params;

      if (!store.has(sessionId)) {
        throw RequestError.resourceNotFound(
          `Session ${sessionId} not found`
        );
      }

      store.setMode(sessionId, modeId);
      logger.info(`Session ${sessionId} mode set to ${modeId}`);

      // Send current_mode_update notification
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "current_mode_update",
          currentModeId: modeId,
        } as any,
      });

      return {};
    },

    async setSessionConfigOption(
      params: SetSessionConfigOptionRequest
    ): Promise<SetSessionConfigOptionResponse> {
      const { sessionId, configId } = params as any;

      if (!store.has(sessionId)) {
        throw RequestError.resourceNotFound(
          `Session ${sessionId} not found`
        );
      }

      const value = (params as any).value;
      store.setConfigOverride(sessionId, configId, value);
      logger.info(`Session ${sessionId} config ${configId} set to ${value}`);

      // Send config_option_update notification
      const configOptions = buildConfigOptions(store, sessionId);
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "config_option_update",
          configOptions,
        } as any,
      });

      return { configOptions } as SetSessionConfigOptionResponse;
    },

    async authenticate(
      _params: AuthenticateRequest
    ): Promise<AuthenticateResponse> {
      return {};
    },

    async prompt(params: PromptRequest): Promise<PromptResponse> {
      logger.debug(`prompt params keys: ${JSON.stringify(Object.keys(params))}`);
      const { sessionId, prompt } = params;

      if (!store.has(sessionId)) {
        throw RequestError.resourceNotFound(
          `Session ${sessionId} not found`
        );
      }

      // Extract text from content blocks
      const text = prompt
        .filter((block): block is { type: "text"; text: string } =>
          block.type === "text"
        )
        .map((block) => block.text)
        .join("\n");

      if (!text.trim()) {
        throw RequestError.invalidParams("Empty prompt text");
      }

      // ── Slash command handling ──────────────────────────────────
      // Intercept built-in commands (/ps, /goal, /skills, /status) before
      // starting an LLM turn. Read-only commands (ps, status, skills) return
      // immediately. /goal sets session state and may start a turn.
      const command = parseSlashCommand(text);
      if (command) {
        const { name, rest } = command;

        // /ps — list background processes (always empty for claude-code-acp,
        // but vscode-acp expects the agent to handle it when not done client-side)
        if (name === "ps") {
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "No background terminals running." },
            } as any,
          });
          return { stopReason: "end_turn" };
        }

        // /status — show session info
        if (name === "status") {
          const cwd = store.getCwd(sessionId)!;
          const configOverrides = store.getConfigOverrides(sessionId);
          const currentModel = (configOverrides["model"] as string) || process.env.ANTHROPIC_MODEL || "glm-5.2";
          const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
          const goal = sessionGoals.get(sessionId);
          const lines = [
            `**Model:** ${currentModel}`,
            `**Mode:** ${currentMode}`,
            `**Directory:** ${cwd}`,
            `**Session:** \`${sessionId}\``,
            goal ? `**Goal:** ${goal.paused ? "(paused) " : ""}${goal.text.slice(0, 100)}` : "**Goal:** none",
          ];
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: lines.join("\n") },
            } as any,
          });
          return { stopReason: "end_turn" };
        }

        // /skills — list available skills (from EriCLI skills system)
        if (name === "skills") {
          const { execSync } = await import("node:child_process");
          let skillsList = "No skills configured.";
          try {
            const output = execSync("ericli skills list 2>/dev/null || echo ''", {
              encoding: "utf-8",
              timeout: 5000,
            }).trim();
            if (output) {
              skillsList = `Available skills:\n${output}`;
            }
          } catch {
            skillsList = "Skills system not available. Install ericli to list skills.";
          }
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: skillsList },
            } as any,
          });
          return { stopReason: "end_turn" };
        }

        // /goal — set/clear/pause/resume a persistent goal
        if (name === "goal") {
          const argument = rest.trim();
          if (!argument) {
            await connection.sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: 'Command "/goal" requires [<objective>|clear|pause|resume].' },
              } as any,
            });
            return { stopReason: "end_turn" };
          }

          const lowerArg = argument.toLowerCase();
          if (lowerArg === "clear") {
            sessionGoals.delete(sessionId);
            await connection.sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "Goal cleared." },
              } as any,
            });
            return { stopReason: "end_turn" };
          }
          if (lowerArg === "pause") {
            const goal = sessionGoals.get(sessionId);
            if (goal) {
              goal.paused = true;
              await connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: "Goal paused. Use /goal resume to continue." },
                } as any,
              });
            } else {
              await connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: "No goal to pause. Set one with /goal <objective>." },
                } as any,
              });
            }
            return { stopReason: "end_turn" };
          }
          if (lowerArg === "resume") {
            const goal = sessionGoals.get(sessionId);
            if (goal) {
              goal.paused = false;
              await connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: `Goal resumed: ${goal.text.slice(0, 100)}` },
                } as any,
              });
            } else {
              await connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "agent_message_chunk",
                  content: { type: "text", text: "No goal to resume. Set one with /goal <objective>." },
                } as any,
              });
            }
            return { stopReason: "end_turn" };
          }

          // Set a new goal
          if (argument.length > 4000) {
            await connection.sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: 'Command "/goal" requires goal text of at most 4000 characters.' },
              } as any,
            });
            return { stopReason: "end_turn" };
          }
          sessionGoals.set(sessionId, { text: argument, paused: false });
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: `Goal set: ${argument.slice(0, 200)}` },
            } as any,
          });
          return { stopReason: "end_turn" };
        }
      }

      // If there's an active (non-paused) goal, prepend it to the prompt
      const activeGoal = sessionGoals.get(sessionId);
      let effectivePrompt = text;
      if (activeGoal && !activeGoal.paused && !command) {
        effectivePrompt = `[Goal: ${activeGoal.text}]\n\n${text}`;
      }

      const claudeSessionId = store.getClaudeSessionId(sessionId);
      const cwd = store.getCwd(sessionId)!;
      let toolCallCounter = 0;

      // Get model and effort overrides from session config (set via pickers)
      const configOverrides = store.getConfigOverrides(sessionId);
      const modelOverride = configOverrides["model"] as string | undefined;
      const effortOverride = configOverrides["effort"] as string | undefined;
      // Only pass effort if the selected model actually supports it
      const effectiveModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
      const modelEfforts = MODEL_EFFORTS[effectiveModel] ?? [];
      const effectiveEffort =
        effortOverride && modelEfforts.includes(effortOverride)
          ? effortOverride
          : undefined;

      logger.info(`Prompt for session ${sessionId}: ${text.length} chars`);
      logger.debug(
        `Prompt content: ${text.slice(0, 100)}${text.length > 100 ? "..." : ""}`
      );

      const permissionPromises: Promise<void>[] = [];

      const onEvent = (event: StreamEvent) => {
        if ((event as any).type === "thinking" && (event as any).text) {
          connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_thought_chunk",
              content: { type: "text", text: (event as any).text },
            } as any,
          });
        } else if (event.type === "text_delta" && event.text) {
          connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: event.text },
            },
          });
        } else if (event.type === "tool_use" && event.toolName) {
          const toolCallId = `call_${++toolCallCounter}`;
          logger.debug(`Tool call: ${event.toolName}`);
          connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "tool_call",
              toolCallId,
              title: event.toolName,
              kind: "execute",
              status: "completed",
              rawInput: event.toolInput ?? {},
            },
          });
        } else if (event.type === "permission_request" && event.toolName) {
          const toolCallId = `call_${++toolCallCounter}`;
          const toolName = event.toolName;
          const toolInput = event.toolInput ?? {};
          logger.info(`Permission request: ${toolName}`);

          // Send pending tool_call status
          connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "tool_call",
              toolCallId,
              title: toolName,
              kind: "execute",
              status: "pending",
              rawInput: toolInput,
            },
          });

          // Request permission from client (track promise for awaiting)
          const permPromise = connection
            .requestPermission({
              sessionId,
              toolCall: {
                toolCallId,
                title: toolName,
                kind: "execute",
                status: "pending",
                rawInput: toolInput,
              },
              options: [
                { optionId: "allow_once", kind: "allow_once", name: "Allow once" },
                { optionId: "allow_always", kind: "allow_always", name: "Allow always" },
                { optionId: "reject_once", kind: "reject_once", name: "Reject once" },
                { optionId: "reject_always", kind: "reject_always", name: "Reject always" },
              ],
            })
            .then((response) => {
              const outcome = response.outcome as any;
              const isSelected =
                outcome.outcome === "selected" || outcome.type === "selected";
              const optionId = outcome.optionId;
              const isApproved =
                isSelected &&
                (optionId === "allow_once" ||
                  optionId === "allow_always");

              connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "tool_call",
                  toolCallId,
                  title: toolName,
                  kind: "execute",
                  status: isApproved ? "completed" : "failed",
                  rawInput: toolInput,
                },
              });
            })
            .catch((err) => {
              logger.error(
                `Permission request failed: ${err instanceof Error ? err.message : String(err)}`
              );
              connection.sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "tool_call",
                  toolCallId,
                  title: toolName,
                  kind: "execute",
                  status: "failed",
                  rawInput: toolInput,
                },
              });
            });

          permissionPromises.push(permPromise);
        }
      };

      try {
        let result;
        if (claudeSessionId) {
          try {
            result = await runner.continueSessionStreaming(
              claudeSessionId,
              effectivePrompt,
              onEvent,
              sessionId,
              modelOverride,
              effectiveEffort
            );
          } catch (resumeErr) {
            // Resume failed (expired session, etc.) — fall back to new session
            logger.warn(
              `Resume failed for ${claudeSessionId}, starting fresh: ${resumeErr instanceof Error ? resumeErr.message : String(resumeErr)}`
            );
            store.clearPersistedSession(cwd);
            result = await runner.startSessionStreaming(
              cwd,
              effectivePrompt,
              onEvent,
              sessionId,
              modelOverride,
              effectiveEffort
            );
            store.setClaudeSessionId(sessionId, result.sessionId);
          }
        } else {
          const mcpServers = store.getMcpServers(sessionId);
          if (mcpServers.length > 0) {
            result = await runner.startSessionWithMcp(
              cwd,
              text,
              mcpServers,
              onEvent,
              sessionId
            );
          } else {
            result = await runner.startSessionStreaming(
              cwd,
              effectivePrompt,
              onEvent,
              sessionId,
              modelOverride,
              effectiveEffort
            );
          }
          store.setClaudeSessionId(sessionId, result.sessionId);
        }

        // Wait for all pending permission requests to resolve
        await Promise.all(permissionPromises);

        // Check if this session was cancelled while running
        if (cancelledSessions.has(sessionId)) {
          cancelledSessions.delete(sessionId);
          logger.info(`Prompt cancelled for session ${sessionId}`);
          return { stopReason: "cancelled" };
        }

        // Send session_info_update with updatedAt
        store.touch(sessionId);
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "session_info_update",
            updatedAt: store.getUpdatedAt(sessionId),
          } as any,
        });

       logger.info(`Prompt completed for session ${sessionId}`);

        // Drain steering queue — send each queued steer message as a
        // follow-up --resume prompt. Claude CLI doesn't support mid-turn
        // injection, so steered messages are delivered sequentially after
        // the current turn finishes.
        const steerQueue = steerQueues.get(sessionId);
        if (steerQueue && steerQueue.length > 0) {
          steerQueues.delete(sessionId);
          for (const steerText of steerQueue) {
            if (cancelledSessions.has(sessionId)) break;
            const claudeSid = store.getClaudeSessionId(sessionId);
            if (!claudeSid) break;
            toolCallCounter = 0;
            permissionPromises.length = 0;
            logger.info(`Draining steer message for session ${sessionId}: ${steerText.length} chars`);
            try {
              // Prepend goal to steer messages too
              const steerPrompt = activeGoal && !activeGoal.paused
                ? `[Goal: ${activeGoal.text}]\n\n${steerText}`
                : steerText;
              await runner.continueSessionStreaming(
                claudeSid,
                steerPrompt,
                onEvent,
                sessionId,
                modelOverride,
                effectiveEffort
              );
              await Promise.all(permissionPromises);
            } catch (steerErr) {
              logger.error(`Steer drain failed for session ${sessionId}: ${steerErr instanceof Error ? steerErr.message : String(steerErr)}`);
            }
          }
          store.touch(sessionId);
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "session_info_update",
              updatedAt: store.getUpdatedAt(sessionId),
            } as any,
          });
        }

        return { stopReason: "end_turn" };
      } catch (err) {
        // Check if cancelled
        if (cancelledSessions.has(sessionId)) {
          cancelledSessions.delete(sessionId);
          logger.info(`Prompt cancelled for session ${sessionId}`);
          return { stopReason: "cancelled" };
        }

        const message =
          err instanceof Error ? err.message : String(err);
        logger.error(`Prompt failed for session ${sessionId}: ${message}`);

        // Send error as agent message so client sees it
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: {
              type: "text",
              text: `Error: ${message}`,
            },
          },
        });

        return { stopReason: "end_turn" };
      }
    },

    async cancel(params: CancelNotification): Promise<void> {
      logger.info(`Cancel request for session ${params.sessionId}`);
      cancelledSessions.add(params.sessionId);
      runner.cancel(params.sessionId);
    },

    /**
     * Handle custom (non-standard) ACP notifications.
     * The SDK's AgentSideConnection dispatches unknown notification methods here.
     * Used for session/steer.
     */
    async extNotification(method: string, params: Record<string, unknown>): Promise<void> {
      if (method === "session/steer") {
        return handleSteer(params as { sessionId: string; text: string });
      }
      logger.warn(`Unknown notification method: ${method}`);
    },
  };
}
