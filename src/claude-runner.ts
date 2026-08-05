import { spawn, type ChildProcess } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type AgentConfig, loadConfig } from "./config.js";
import { logger } from "./logger.js";

export interface McpServerConfig {
  name: string;
  transport: {
    type: "stdio";
    command: string;
    args?: string[];
    env?: Record<string, string>;
  };
}

export interface ClaudeResult {
  text: string;
  sessionId: string;
}

export interface StreamEvent {
  type: "text_delta" | "tool_use" | "result" | "permission_request" | "thinking" | "system_init" | "message_delta" | "assistant" | "user";
 text?: string;
 toolName?: string;
 toolInput?: unknown;
 /** Claude's native tool_use id (from the assistant content block). */
 toolUseId?: string;
 /** Claude tool_use ids resolved by a following user/tool_result message. */
 toolResultIds?: string[];
 sessionId?: string;
 permissionId?: string;
 usage?: {
   input_tokens?: number;
   output_tokens?: number;
   /** Prompt tokens served from the cache — part of the real context size. */
   cache_read_input_tokens?: number;
   /** Prompt tokens written to the cache — part of the real context size. */
   cache_creation_input_tokens?: number;
 };
 /**
  * True when the event belongs to a subagent (sidechain) conversation, i.e.
  * Claude CLI reported a non-null `parent_tool_use_id`. Subagents have their
  * own context window, so their usage must not be attributed to the parent.
  */
 isSidechain?: boolean;
  slashCommands?: string[];
  /** Claude CLI result stop reason (e.g. "end_turn", "max_turns"). */
  stopReason?: string;
  /** True when Claude CLI reports the result as an error (e.g. max turns). */
  isError?: boolean;
  /** For assistant events: marks an API call boundary (for context estimation). */
  apiCallIndex?: number;
  /**
   * For result events: the context window Claude Code resolved for the model
   * (from `modelUsage[...].contextWindow`) and its max output tokens. These are
   * the CLI's own figures, used here so the usage denominator doesn't rely on a
   * hardcoded default.
   */
  contextWindow?: number;
  maxOutputTokens?: number;
  /**
   * For text_delta events: true when the text comes from the final `assistant`
   * message block (not a streaming content_block_delta). The context estimator
   * uses this to avoid double-counting assistant text in --verbose streams,
   * where the same content arrives both as streaming deltas and as the final
   * block. UI rendering treats both identically.
   */
  fromAssistantBlock?: boolean;
}

const SENSITIVE_FLAGS = new Set(["-p", "--print"]);
const REDACT_FLAGS = new Set(["--resume"]);

export function maskArgs(args: string[]): string {
  const masked: string[] = [];
  for (let i = 0; i < args.length; i++) {
    if (SENSITIVE_FLAGS.has(args[i]) && i + 1 < args.length) {
      masked.push(args[i], `<prompt: ${args[i + 1].length} chars>`);
      i++;
    } else if (REDACT_FLAGS.has(args[i]) && i + 1 < args.length) {
      masked.push(args[i], "<session-id>");
      i++;
    } else {
      masked.push(args[i]);
    }
  }
  return masked.join(" ");
}

export class ClaudeRunner {
  private runningProcesses = new Map<string, ChildProcess>();
  /**
   * Tracking ids whose process was explicitly killed by cancel(). On close,
   * runStreaming treats these as a clean cancellation (resolve with whatever
   * was captured, including the Claude session id) rather than a failure
   * (reject "claude exited with code N"). Without this, a user "Send Now"
   * (which cancels the in-flight turn) makes the *next* prompt's
   * continueSessionStreaming reject → the prompt handler catches it and
   * clearPersistedSession() wipes the Claude session id → the follow-up
   * starts fresh with no conversation history. See agent.ts prompt handler.
   */
  private cancelledProcesses = new Set<string>();
  private tempDirs: string[] = [];
  private config: AgentConfig;

  constructor(config?: AgentConfig) {
    this.config = config ?? loadConfig();
  }

  private buildExtraArgs(modelOverride?: string, effortOverride?: string): string[] {
    const extra: string[] = [];
    const model = modelOverride || this.config.model;
    if (model) {
      extra.push("--model", model);
    }
    if (effortOverride) {
      extra.push("--effort", effortOverride);
    }
    if (this.config.maxTurns) {
      extra.push("--max-turns", String(this.config.maxTurns));
    }
    if (this.config.dangerouslySkipPermissions) {
      extra.push("--dangerously-skip-permissions");
    }
    for (const tool of this.config.allowedTools) {
      extra.push("--allowedTools", tool);
    }
    return extra;
  }

  async startSession(
    cwd: string,
    prompt: string
  ): Promise<ClaudeResult> {
    const args = ["-p", prompt, "--output-format", "json", ...this.buildExtraArgs()];
    return this.runJson(args, cwd);
  }

  async continueSession(
    claudeSessionId: string,
    prompt: string
  ): Promise<ClaudeResult> {
    const args = [
      "-p",
      prompt,
      "--resume",
      claudeSessionId,
      "--output-format",
      "json",
      ...this.buildExtraArgs(),
    ];
    return this.runJson(args);
  }

  async startSessionStreaming(
    cwd: string,
    prompt: string,
    onEvent: (event: StreamEvent) => void,
    trackingId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): Promise<ClaudeResult> {
    const args = [
      "-p",
      prompt,
      "--output-format",
      "stream-json",
      "--verbose",
      // Emit per-token deltas (wrapped in `stream_event`) so assistant text
      // renders progressively instead of landing as one large chunk. Without
      // this the CLI only sends the completed `assistant` message, so the
      // client creates a bubble and immediately supersedes it — a visibly
      // empty bubble that flashes and disappears at each text→tool boundary.
      "--include-partial-messages",
      ...this.buildExtraArgs(modelOverride, effortOverride),
    ];
    return this.runStreaming(args, cwd, onEvent, trackingId);
  }

  async continueSessionStreaming(
    claudeSessionId: string,
    prompt: string,
    onEvent: (event: StreamEvent) => void,
    trackingId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): Promise<ClaudeResult> {
    const args = [
      "-p",
      prompt,
      "--resume",
      claudeSessionId,
      "--output-format",
      "stream-json",
      "--verbose",
      // Emit per-token deltas (wrapped in `stream_event`) so assistant text
      // renders progressively instead of landing as one large chunk. Without
      // this the CLI only sends the completed `assistant` message, so the
      // client creates a bubble and immediately supersedes it — a visibly
      // empty bubble that flashes and disappears at each text→tool boundary.
      "--include-partial-messages",
      ...this.buildExtraArgs(modelOverride, effortOverride),
    ];
    return this.runStreaming(args, undefined, onEvent, trackingId);
  }

  async startSessionWithMcp(
    cwd: string,
    prompt: string,
    mcpServers: McpServerConfig[],
    onEvent?: (event: StreamEvent) => void,
    trackingId?: string
  ): Promise<ClaudeResult> {
    const mcpArgs = this.buildMcpArgs(mcpServers);
    if (onEvent) {
      const args = [
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        ...this.buildExtraArgs(),
        ...mcpArgs,
      ];
      return this.runStreaming(args, cwd, onEvent, trackingId);
    } else {
      const args = [
        "-p",
        prompt,
        "--output-format",
        "json",
        ...this.buildExtraArgs(),
        ...mcpArgs,
      ];
      return this.runJson(args, cwd);
    }
  }

  private buildMcpArgs(mcpServers: McpServerConfig[]): string[] {
    if (mcpServers.length === 0) return [];

    // Write MCP config to temp file
    const tmpDir = mkdtempSync(join(tmpdir(), "claude-acp-mcp-"));
    const configPath = join(tmpDir, "mcp.json");
    const mcpConfig: Record<string, any> = {
      mcpServers: Object.fromEntries(
        mcpServers.map((s) => [
          s.name,
          {
            command: s.transport.command,
            args: s.transport.args ?? [],
            env: s.transport.env ?? {},
          },
        ])
      ),
    };
    writeFileSync(configPath, JSON.stringify(mcpConfig, null, 2));
    this.tempDirs.push(tmpDir);
    logger.debug(`MCP config written to ${configPath}`);
    return ["--mcp-config", configPath];
  }

  cleanup(): void {
    for (const dir of this.tempDirs) {
      try {
        rmSync(dir, { recursive: true, force: true });
      } catch {
        // ignore cleanup errors
      }
    }
    this.tempDirs = [];
  }

  private sanitizeEnv(): NodeJS.ProcessEnv {
    // Keep ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN so claude CLI
    // routes through our CCR gateway → ericai-proxy → EricAI vLLM.
    // The original code deleted these to force subscription auth, but
    // in our setup we use API key auth via the proxy chain.
    return { ...process.env };
  }

  cancel(trackingId: string): void {
    const proc = this.runningProcesses.get(trackingId);
    if (proc) {
      // Mark as explicitly cancelled BEFORE the kill so the close handler
      // resolves (clean cancel) rather than rejecting (treated as a crash /
      // resume failure). This preserves the captured Claude session id.
      this.cancelledProcesses.add(trackingId);
      proc.kill("SIGTERM");
      this.runningProcesses.delete(trackingId);
    }
  }

  isRunning(trackingId: string): boolean {
    return this.runningProcesses.has(trackingId);
  }

  private runJson(
    args: string[],
    cwd?: string
  ): Promise<ClaudeResult> {
    return new Promise((resolve, reject) => {
      logger.debug(`spawn: claude ${maskArgs(args)}`);
      const proc = spawn("claude", args, {
        cwd,
        env: this.sanitizeEnv(),
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";

      proc.stdout!.on("data", (data: Buffer) => {
        stdout += data.toString();
      });
      proc.stderr!.on("data", (data: Buffer) => {
        stderr += data.toString();
      });

      proc.on("close", (code) => {
        if (code !== 0) {
          reject(
            new Error(
              `claude exited with code ${code}: ${stderr || stdout}`
            )
          );
          return;
        }
        try {
          const parsed = JSON.parse(stdout.trim());
          resolve({
            text: parsed.result ?? "",
            sessionId: parsed.session_id ?? "",
          });
        } catch {
          reject(new Error(`Failed to parse claude output: ${stdout}`));
        }
      });

      proc.on("error", reject);
    });
  }

  private runStreaming(
    args: string[],
    cwd: string | undefined,
    onEvent: (event: StreamEvent) => void,
    trackingId?: string
  ): Promise<ClaudeResult> {
    return new Promise((resolve, reject) => {
      logger.debug(`spawn streaming: claude ${maskArgs(args)}`);
      const proc = spawn("claude", args, {
        cwd,
        env: this.sanitizeEnv(),
        stdio: ["ignore", "pipe", "pipe"],
      });

      if (trackingId) {
        this.runningProcesses.set(trackingId, proc);
      }

      let buffer = "";
      let resultText = "";
      let sessionId = "";

      proc.stdout!.on("data", (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            this.processStreamLine(parsed, onEvent);

            // Extract session_id and result from various event types
            if (parsed.session_id) {
              sessionId = parsed.session_id;
            }
            if (parsed.result !== undefined) {
              resultText = parsed.result;
            }
            if (parsed.type === "result") {
              sessionId = parsed.session_id ?? sessionId;
              resultText = parsed.result ?? resultText;
            }
          } catch {
            // Skip unparseable lines
          }
        }
      });

      proc.on("close", (code) => {
        if (trackingId) {
          this.runningProcesses.delete(trackingId);
        }
        const wasCancelled = trackingId
          ? this.cancelledProcesses.delete(trackingId)
          : false;

        // Process remaining buffer
        if (buffer.trim()) {
          try {
            const parsed = JSON.parse(buffer);
            this.processStreamLine(parsed, onEvent);
            if (parsed.session_id) sessionId = parsed.session_id;
            if (parsed.result !== undefined) resultText = parsed.result;
          } catch {
            // ignore
          }
        }

        // An explicit cancel() (user "Send Now" / interrupt) kills the
        // process with SIGTERM → non-zero exit. Treat that as a clean
        // cancellation, NOT a failure: resolve with whatever was captured
        // (notably the Claude session id, which the CLI emits early). The
        // prompt handler checks cancelledSessions and short-circuits to
        // { stopReason: "cancelled" } without touching the stored id, so
        // the follow-up prompt resumes correctly. Rejecting here would
        // make continueSessionStreaming's catch wipe the session id and
        // start fresh — losing all conversation history.
        if (wasCancelled) {
          resolve({ text: resultText, sessionId });
          return;
        }

        if (code !== 0 && code !== null) {
          reject(new Error(`claude exited with code ${code}`));
          return;
        }

        resolve({ text: resultText, sessionId });
      });

      proc.on("error", reject);
    });
  }

  private processStreamLine(
    parsed: any,
    onEvent: (event: StreamEvent) => void
  ): void {
    // Handle system/init — extract native slash commands from Claude CLI
    if (parsed.type === "system" && parsed.subtype === "init") {
      const slashCommands: string[] = Array.isArray(parsed.slash_commands)
        ? parsed.slash_commands.filter((c: any) => typeof c === "string")
        : [];
      onEvent({ type: "system_init", slashCommands });
      return;
    }

    // Unwrap the `--include-partial-messages` envelope. Claude CLI does NOT emit
    // bare `content_block_delta` / `message_delta` lines — it nests them as
    // {"type":"stream_event","event":{"type":"content_block_delta","delta":{…}}}.
    // The handlers below match on the INNER shape, so without this unwrap they
    // never fire and all assistant text arrives only as the completed
    // `assistant` message (one big chunk → the empty-bubble flash).
    //
    // `parent_tool_use_id` lives on the WRAPPER, so capture it before replacing
    // `parsed` — it is how a subagent's own API calls are told apart from the
    // main conversation's.
    let envelopeSidechain = false;
    if (parsed.type === "stream_event" && parsed.event && typeof parsed.event === "object") {
      envelopeSidechain = parsed.parent_tool_use_id != null;
      parsed = parsed.event;
    }

   // Handle content_block_delta (streaming text)
    if (
      parsed.type === "content_block_delta" &&
      parsed.delta?.type === "text_delta"
    ) {
      onEvent({ type: "text_delta", text: parsed.delta.text });
      return;
    }

   // Handle assistant message with content array
   if (parsed.type === "assistant" && parsed.message?.content) {
      // Emit an "assistant" event FIRST — this marks an API call boundary
      // for context estimation (each assistant message = one LLM API call).
      //
      // Each assistant message carries the REAL per-call usage for the API
      // call that produced it. `input_tokens` (+ cache reads/creations, which
      // are prompt tokens served from cache) is the exact prompt size the
      // provider counted — i.e. the true current context size. This is
      // authoritative and replaces character-based guessing. Unlike the
      // `result` event (whose usage is accumulated across every call in the
      // turn, so it over-reports on multi-tool turns), this value is per-call.
      //
      // `parent_tool_use_id` is non-null for subagent (sidechain) messages.
      // A subagent runs its own separate conversation, so its context size
      // must NOT be attributed to the parent session.
      const u = parsed.message.usage;
      const isSidechain = parsed.parent_tool_use_id != null;
      onEvent({
        type: "assistant",
        apiCallIndex: 0,
        isSidechain,
        ...(u
          ? {
              usage: {
                input_tokens: u.input_tokens,
                output_tokens: u.output_tokens,
                cache_read_input_tokens: u.cache_read_input_tokens,
                cache_creation_input_tokens: u.cache_creation_input_tokens,
              },
            }
          : {}),
      });
     for (const block of parsed.message.content) {
       if (block.type === "text") {
         onEvent({ type: "text_delta", text: block.text, fromAssistantBlock: true });
       } else if (block.type === "tool_use") {
         onEvent({
           type: "tool_use",
           toolName: block.name,
           toolInput: block.input,
           toolUseId: typeof block.id === "string" ? block.id : undefined,
         });
       } else if (block.type === "thinking") {
         onEvent({ type: "thinking", text: block.thinking });
       }
     }
     return;
   }

    // Handle user message (tool results) — emit for context estimation.
    // Claude CLI stream-json emits {"type":"user","message":{"content":[{"type":"tool_result","content":"..."}]}}
    if (parsed.type === "user" && parsed.message?.content) {
      // Collect the tool_use_ids this message resolves, so the agent can flip
      // any in_progress tool calls (e.g. a running subagent) to completed.
      const toolResultIds: string[] = [];
      // Emit a "user" event so the context estimator can accumulate tool results.
      const toolResultText = Array.isArray(parsed.message.content)
        ? parsed.message.content
            .map((b: any) => {
              if (typeof b === "string") return b;
              if (b.type === "tool_result") {
                if (typeof b.tool_use_id === "string") toolResultIds.push(b.tool_use_id);
                return typeof b.content === "string"
                  ? b.content
                  : JSON.stringify(b.content ?? "");
              }
              return JSON.stringify(b);
            })
            .join("")
        : "";
      onEvent({
        type: "user",
        text: toolResultText,
        ...(toolResultIds.length ? { toolResultIds } : {}),
      });
      return;
    }

   // Handle permission request from Claude CLI
   if (parsed.type === "permission_request") {
     onEvent({
       type: "permission_request",
       toolName: parsed.tool_name ?? parsed.toolName,
       toolInput: parsed.tool_input ?? parsed.toolInput,
       permissionId: parsed.permission_id ?? parsed.permissionId,
     });
     return;
   }

    // Handle thinking/extended thinking blocks
    if (
      parsed.type === "content_block_delta" &&
      parsed.delta?.type === "thinking_delta"
    ) {
      onEvent({ type: "thinking", text: parsed.delta.thinking });
      return;
    }

   // Handle message_delta — the ONLY real-time source of exact per-call usage.
   //
   // Verified against claude v2.1.220 through the CCR/proxy path: the completed
   // `assistant` message reports `usage.input_tokens: 0`, and the `result` event
   // reports usage ACCUMULATED over every API call in the turn (2 calls of ~22k
   // showed as 44193). Only `message_delta` carries the true prompt size for the
   // call that just finished — 20694 for a 20694-token prompt. This is the same
   // number Claude Code and the Anthropic SDK use for their own context
   // accounting, so surfacing it removes the need to estimate at all.
   if (parsed.type === "message_delta" && parsed.usage) {
      onEvent({
        type: "message_delta",
        isSidechain: envelopeSidechain,
        usage: {
          input_tokens: parsed.usage.input_tokens ?? parsed.usage.inputTokens,
          output_tokens: parsed.usage.output_tokens ?? parsed.usage.outputTokens,
          cache_read_input_tokens: parsed.usage.cache_read_input_tokens,
          cache_creation_input_tokens: parsed.usage.cache_creation_input_tokens,
        },
      });
      return;
    }

    // Handle result event
    if (parsed.type === "result") {
      // `modelUsage` reports the context window and max output tokens Claude
      // Code itself resolved for the model — the same figures it uses for its
      // own auto-compact threshold. Surfaced so the UI denominator comes from
      // the CLI rather than a hardcoded default.
      let contextWindow: number | undefined;
      let maxOutputTokens: number | undefined;
      const mu = parsed.modelUsage;
      if (mu && typeof mu === "object") {
        for (const entry of Object.values(mu) as any[]) {
          if (entry && typeof entry.contextWindow === "number" && entry.contextWindow > 0) {
            contextWindow = entry.contextWindow;
            if (typeof entry.maxOutputTokens === "number") maxOutputTokens = entry.maxOutputTokens;
            break;
          }
        }
      }
      onEvent({
        type: "result",
        text: parsed.result,
        sessionId: parsed.session_id,
        usage: parsed.usage,
        ...(contextWindow !== undefined ? { contextWindow } : {}),
        ...(maxOutputTokens !== undefined ? { maxOutputTokens } : {}),
        // Claude CLI's stop reason/subtype — e.g. "end_turn", "max_turns",
        // "stop_sequence". Surfaced so the agent can log/diagnose premature
        // turn termination (vs a model that kept calling tools).
        stopReason: parsed.subtype ?? parsed.stop_reason ?? parsed.stopReason,
        isError: parsed.is_error === true || parsed.subtype === "error_max_turns",
      });
    }
  }
}
