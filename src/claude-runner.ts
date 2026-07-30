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
 usage?: { input_tokens?: number; output_tokens?: number };
  slashCommands?: string[];
  /** Claude CLI result stop reason (e.g. "end_turn", "max_turns"). */
  stopReason?: string;
  /** True when Claude CLI reports the result as an error (e.g. max turns). */
  isError?: boolean;
  /** For assistant events: marks an API call boundary (for context estimation). */
  apiCallIndex?: number;
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
      onEvent({ type: "assistant", apiCallIndex: 0 });
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

   // Handle message_delta — carries per-call usage (the REAL current
   // context size), unlike the result event which has cumulative usage.
   if (parsed.type === "message_delta" && parsed.usage) {
      // NOTE: Claude CLI does NOT forward message_delta events to stdout in
      // stream-json mode (includePartialMessages defaults to false in
      // QueryEngine.ts). This handler exists for future-proofing but will
      // never fire with current Claude CLI versions. Usage tracking relies on
      // the result event instead (see agent.ts).
      onEvent({
        type: "message_delta",
        usage: {
          input_tokens: parsed.usage.input_tokens ?? parsed.usage.inputTokens,
          output_tokens: parsed.usage.output_tokens ?? parsed.usage.outputTokens,
        },
      });
      return;
    }

    // Handle result event
    if (parsed.type === "result") {
      onEvent({
        type: "result",
        text: parsed.result,
        sessionId: parsed.session_id,
        usage: parsed.usage,
        // Claude CLI's stop reason/subtype — e.g. "end_turn", "max_turns",
        // "stop_sequence". Surfaced so the agent can log/diagnose premature
        // turn termination (vs a model that kept calling tools).
        stopReason: parsed.subtype ?? parsed.stop_reason ?? parsed.stopReason,
        isError: parsed.is_error === true || parsed.subtype === "error_max_turns",
      });
    }
  }
}
