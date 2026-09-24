import { spawn, execFileSync, type ChildProcess } from "node:child_process";
import { writeFileSync, mkdtempSync, rmSync, existsSync } from "node:fs";
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

/** One long-lived `claude -p --input-format stream-json` process for an ACP session. */
interface LiveClaudeSession {
  proc: ChildProcess;
  trackingId: string;
  /** Claude CLI session id, filled from the first result when this process created it. */
  claudeSessionId: string;
  /** model + effort + permission mode captured at spawn. A mismatch forces --resume. */
  spawnKey: string;
  turn: {
    resolve: (result: ClaudeResult) => void;
    reject: (err: Error) => void;
    onEvent: (event: StreamEvent) => void;
    resultText: string;
    sessionId: string;
  } | null;
  buffer: string;
  stderrText: string;
}

/**
 * A user message content block following the Anthropic Messages API schema,
 * sent to the Claude CLI via `--input-format=stream-json` on stdin.
 *
 * The CLI's SDK input layer accepts the same content blocks as the Messages
 * API: `text`, `image` (with `source`), `tool_result`, `document`, etc. Only
 * `text` and `image` are used by this bridge today, but the union is kept
 * open so future media types need no type churn.
 */
export type StdinContentBlock =
  | { type: "text"; text: string }
  | {
      type: "image";
      source: {
        type: "base64";
        media_type: string;
        data: string;
      };
    }
  | { type: "tool_result"; tool_use_id?: string; content: string }
  | { type: "document"; source: { type: "base64"; media_type: string; data: string } }
  | { type: string; [key: string]: unknown };

export interface StreamEvent {
  type: "text_delta" | "tool_use" | "result" | "permission_request" | "thinking" | "system_init" | "message_delta" | "assistant" | "user" | "tool_progress";
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
  /**
   * Claude CLI's own list of slash commands it serves from the terminal
   * (e.g. "doctor", "color", "reload-plugins"). These are read-only,
   * non-LLM interactions, so agent.ts marks them turn-safe when merging the
   * native command set into available_commands_update.
   */
  terminalSlashCommands?: string[];
  /**
   * Claude CLI's skill names. They appear inside `slash_commands` (a skill is
   * invocable as `/<name>`), but they are NOT "commands" in the ACP sense —
   * agent.ts excludes them from the merged list so the picker doesn't show a
   * skill like `/zhihu` as a command with a bogus `Claude Code: /zhihu` label.
   */
  skillNames?: string[];
  /**
   * Claude CLI's agent (subagent) names — "claude", "Explore", "general-purpose",
   * "Plan", etc. The CLI lists them for the Task tool; they are NOT slash
   * commands a user types, so agent.ts excludes them from the command list too.
   */
  agentNames?: string[];
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
  /**
   * For tool_progress events: seconds the tool has been running, as reported
   * by Claude CLI's periodic heartbeat (`elapsed_time_seconds`). Surfaced so a
   * long-running tool (a Bash poll that sleeps for minutes) shows visible
   * progress in the chat UI instead of looking like a hung turn.
   */
  toolElapsedSeconds?: number;
  /**
   * For tool_progress events: true when the heartbeat carries a non-null
   * `parent_tool_use_id`, i.e. it belongs to a subagent's tool rather than
   * the main conversation's. Sidechain heartbeats must not be forwarded as
   * updates to the parent's tool calls.
   */
  toolSidechain?: boolean;
}

const SENSITIVE_FLAGS = new Set(["-p", "--print"]);
const REDACT_FLAGS = new Set(["--resume"]);

/**
 * True when the base URL routes LLM calls somewhere other than the real
 * Anthropic API — a proxy/gateway (ericai-proxy, a local relay, anything
 * non-anthropic). Empty/undefined or a *.anthropic.com host means real
 * Anthropic is in front (direct, or a pass-through on the official host),
 * where the server-side WebSearch tool does work.
 */
export function isProxyBaseUrl(baseUrl: string | undefined): boolean {
  if (!baseUrl) return false;
  try {
    const host = new URL(baseUrl).hostname;
    return !/(^|\.)anthropic\.com$/i.test(host);
  } catch {
    // Unparseable URL — treat as a proxy (it is certainly not the API).
    return true;
  }
}

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
  /**
   * Pending SIGINT→SIGTERM escalation timers, keyed by trackingId. cancel()
   * sends SIGINT first (a gentle turn interrupt that Claude Code's interactive
   * mode handles); if the process has not exited within the grace window, the
   * timer escalates to SIGTERM (hard kill). The timer is cleared when the
   * process closes naturally, so a process that honors SIGINT is never
   * SIGTERMed after the fact.
   */
  private cancelEscalationTimers = new Map<string, NodeJS.Timeout>();
  private tempDirs: string[] = [];
  private config: AgentConfig;
  /**
   * Session permission mode, set by the agent before each prompt spawn.
   * "read-only" (plan mode) → --permission-mode plan so the Claude CLI
   * hard-blocks mutating tools at its own layer, not just via the
   * <collaboration_mode> prompt block. "agent" / "agent-full-access" →
   * --permission-mode acceptEdits (the CLI's interactive-equivalent of the
   * default agent mode). When null, no flag is passed (the CLI uses its own
   * default), preserving prior behavior for callers that don't track mode.
   */
  private sessionPermissionMode: string | null = null;

  /**
   * Set the permission mode for the next spawn. Called by the agent from
   * enterPlanMode (→ "read-only") and leavePlanMode (→ "agent"). Because the
   * CLI is spawned per-prompt (--resume <id>), each turn's spawn picks up the
   * current mode without restarting the session or losing conversation history.
   */
  setSessionPermissionMode(mode: string | null): void {
    this.sessionPermissionMode = mode;
  }

  constructor(config?: AgentConfig) {
    this.config = config ?? loadConfig();
  }

  /**
   * Resolve the claude executable to spawn.
   *
   * On Windows `claude` ships as .cmd shims, which spawn() cannot run without
   * a shell — and cmd shell-quoting concatenates args with raw quotes, so any
   * prompt containing `"` (the memory wrapper always does) is re-tokenized
   * into garbage and claude exits 1. Resolving the real .exe and spawning it
   * directly keeps argv clean on every platform.
   */
  protected resolveClaudeBin(): string {
    if (process.platform !== "win32") return "claude";
    const candidates: string[] = [];
    try {
      const out = execFileSync("where", ["claude"], { encoding: "utf8", windowsHide: true });
      candidates.push(
        ...out
          .split(/\r?\n/)
          .map((s) => s.trim())
          .filter((s) => /\.exe$/i.test(s)),
      );
    } catch {
      // Not on PATH as an .exe — fall through to known install layouts.
    }
    if (process.env.APPDATA) {
      candidates.push(
        join(process.env.APPDATA, "npm", "node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe"),
      );
    }
    if (process.env.USERPROFILE) {
      candidates.push(join(process.env.USERPROFILE, ".local", "bin", "claude.exe"));
    }
    return candidates.find((p) => existsSync(p)) || "claude";
  }

  private buildExtraArgs(modelOverride?: string, effortOverride?: string): string[] {
    const extra: string[] = [];
    const model = modelOverride || this.config.model;
    if (model) {
      extra.push("--model", model);
    }
    // Thread the session permission mode through as a CLI flag so the Claude
    // CLI enforces read-only (plan mode) at its own layer, not just via the
    // <collaboration_mode> prompt block. "read-only" → plan (hard-blocks
    // mutating tools); everything else → acceptEdits (the CLI's interactive
    // default for an agent). A null mode (caller didn't track it) passes no
    // flag — the CLI uses its own default, so this is a no-op for any path
    // that doesn't call setSessionPermissionMode.
    if (this.sessionPermissionMode) {
      extra.push(
        "--permission-mode",
        this.sessionPermissionMode === "read-only" ? "plan" : "acceptEdits",
      );
    }
    // WebSearch is a server-side Anthropic tool: the search is executed by
    // Anthropic's API, not the CLI. With ANTHROPIC_BASE_URL pointed at a proxy
    // serving non-Claude models (ericai-proxy: GLM/DeepSeek/Qwen), nobody
    // executes it — the tool_use opens and waits for a tool_result that never
    // arrives, which the chat shows as a hung turn. Drop it from the advertised
    // set so the model never calls an unsupported tool. A direct Anthropic
    // route keeps it (real Claude models do execute it).
    // WebFetch is fetched by the CLI itself (client-side), so the proxy route
    // does not break it — but where the network blocks it (e.g. China), the
    // fetch stalls. Add it to the dropped set via
    // CLAUDE_ACP_DISALLOWED_WEB_TOOLS=WebSearch,WebFetch on such hosts.
    // CLAUDE_ACP_KEEP_WEB_TOOLS=1 drops nothing at all (proxy that does front
    // a web-search-capable backend).
    if (!process.env.CLAUDE_ACP_KEEP_WEB_TOOLS && isProxyBaseUrl(process.env.ANTHROPIC_BASE_URL)) {
      const dropped = (process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS || "WebSearch")
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean)
        .join(",");
      if (dropped) {
        extra.push("--disallowedTools", dropped);
      }
    }
    if (effortOverride) {
      extra.push("--effort", effortOverride);
    }
    if (this.config.maxTurns) {
      extra.push("--max-turns", String(this.config.maxTurns));
    }
    // --dangerously-skip-permissions bypasses ALL permission checks, which
    // includes plan mode's read-only gate. It is therefore mutually exclusive
    // with read-only: when the session is in plan mode we MUST omit it, or the
    // CLI silently re-enables full edit/Bash access (rm -rf, file writes) even
    // though --permission-mode plan is also passed. In every other mode ACP
    // relies on the flag to auto-approve without interactive prompts.
    if (this.config.dangerouslySkipPermissions && this.sessionPermissionMode !== "read-only") {
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
    if (trackingId) {
      return this.runPersistentTurn({
        trackingId,
        cwd,
        resumeId: undefined,
        replace: true,
        content: [{ type: "text", text: prompt }],
        onEvent,
        modelOverride,
        effortOverride,
      });
    }
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

  /**
   * Starts a new session with a multimodal user message via the Claude CLI's
   * `--input-format=stream-json` stdin protocol. Unlike the plain-string
   * `startSessionStreaming`, the message may contain `image` content blocks,
   * which the CLI forwards to vision-capable models.
   *
   * Backward compatible: callers that only pass text can keep using
   * `startSessionStreaming`; this method is only required when an image is
   * present in the ACP prompt.
   */
  async startSessionStreamingWithContent(
    cwd: string,
    content: StdinContentBlock[],
    onEvent: (event: StreamEvent) => void,
    trackingId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): Promise<ClaudeResult> {
    if (trackingId) {
      return this.runPersistentTurn({
        trackingId,
        cwd,
        resumeId: undefined,
        replace: true,
        content,
        onEvent,
        modelOverride,
        effortOverride,
      });
    }
    const args = this.buildStreamJsonArgs(undefined, modelOverride, effortOverride);
    return this.runStreamingJsonLine(args, cwd, content, onEvent, trackingId);
  }

  /**
   * Continues a session with a multimodal user message. See
   * `startSessionStreamingWithContent` for why this exists.
   */
  async continueSessionStreamingWithContent(
    claudeSessionId: string,
    content: StdinContentBlock[],
    onEvent: (event: StreamEvent) => void,
    trackingId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): Promise<ClaudeResult> {
    if (trackingId) {
      return this.runPersistentTurn({
        trackingId,
        cwd: undefined,
        resumeId: claudeSessionId,
        replace: false,
        content,
        onEvent,
        modelOverride,
        effortOverride,
      });
    }
    const args = this.buildStreamJsonArgs(claudeSessionId, modelOverride, effortOverride);
    return this.runStreamingJsonLine(args, undefined, content, onEvent, trackingId);
  }

  /**
   * Builds the CLI argv for the stream-json stdin input path. Mirrors the
   * plain-string streaming args, plus `--input-format=stream-json` (required
   * for stdin JSON messages) — and omits `-p`/`--print <prompt>`, since the
   * message now arrives as a JSON line on stdin instead.
   */
  private buildStreamJsonArgs(
    claudeSessionId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): string[] {
    const args: string[] = [
      "-p",
      "--input-format",
      "stream-json",
      "--output-format",
      "stream-json",
      "--verbose",
      // Emit per-token deltas (wrapped in `stream_event`) so assistant text
      // renders progressively instead of landing as one large chunk.
      "--include-partial-messages",
    ];
    if (claudeSessionId) {
      args.push("--resume", claudeSessionId);
    }
    args.push(...this.buildExtraArgs(modelOverride, effortOverride));
    return args;
  }

  async continueSessionStreaming(
    claudeSessionId: string,
    prompt: string,
    onEvent: (event: StreamEvent) => void,
    trackingId?: string,
    modelOverride?: string,
    effortOverride?: string
  ): Promise<ClaudeResult> {
    if (trackingId) {
      return this.runPersistentTurn({
        trackingId,
        cwd: undefined,
        resumeId: claudeSessionId,
        replace: false,
        content: [{ type: "text", text: prompt }],
        onEvent,
        modelOverride,
        effortOverride,
      });
    }
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
    const env = { ...process.env };
    // Never leak the parent Claude Code session into the spawned CLI: claude
    // refuses to run nested when CLAUDECODE is set ("Claude Code cannot be
    // launched inside another Claude Code session", exit code 1). This hits
    // when VS Code is launched from inside a Claude Code terminal.
    delete env.CLAUDECODE;
    delete env.CLAUDE_CODE_SESSION_ID;
    delete env.CLAUDE_CODE_CHILD_SESSION;
    return env;
  }

  /**
   * Signal a spawned claude process. The streaming spawns use `detached:
   * true`, so the child leads its own process group and signaling `-pid`
   * reaches every descendant the turn's tools forked (a Bash poll's
   * `bash → sleep/ssh` tree). Without the group signal those grandchildren
   * outlive the cancelled turn as orphans — continuing to run, consume the
   * remote resource and hold the network long after the UI says "cancelled".
   * Falls back to signaling just the child when the group is gone (already
   * reaped, ESRCH) or the pid is not a group leader (ESRCH/EPERM/unknown).
   */
  private signalProcessTree(proc: ChildProcess, signal: NodeJS.Signals): void {
    if (proc.pid !== undefined) {
      try {
        process.kill(-proc.pid, signal);
        return;
      } catch {
        // Group leader gone or not a group — fall through to the child only.
      }
    }
    try {
      proc.kill(signal);
    } catch {
      // Process already gone — nothing to signal.
    }
  }

  cancel(trackingId: string): void {
    const proc = this.runningProcesses.get(trackingId);
    if (!proc) {
      return;
    }
    // Mark as explicitly cancelled BEFORE the kill so the close handler
    // resolves (clean cancel) rather than rejecting (treated as a crash /
    // resume failure). This preserves the captured Claude session id.
    this.cancelledProcesses.add(trackingId);

    // SIGINT to the whole group — Claude Code's interactive mode treats SIGINT
    // as an in-turn interrupt and exits cleanly, keeping the session id valid;
    // the group signal also interrupts the tool processes the turn spawned.
    // If nothing has exited within the grace window, escalate the whole group
    // to SIGTERM (hard kill). There is no native "interrupt this turn" RPC on
    // the CLI (see anthropics/claude-code#51078), so this is the gentlest
    // reliable sequence available.
    if (this.cancelEscalationTimers.has(trackingId)) {
      // A cancel is already in progress for this trackingId — do not stack a
      // second escalation timer.
      return;
    }
    this.signalProcessTree(proc, "SIGINT");
    const escalation = setTimeout(() => {
      this.cancelEscalationTimers.delete(trackingId);
      const stillRunning = this.runningProcesses.get(trackingId);
      // Only escalate if the process is still alive and still tracked. If it
      // exited cleanly from SIGINT, the close handler already removed it from
      // runningProcesses and cleared the timer.
      if (stillRunning && stillRunning.exitCode === null) {
        this.signalProcessTree(stillRunning, "SIGTERM");
      }
    }, 3000);
    this.cancelEscalationTimers.set(trackingId, escalation);
  }

  isRunning(trackingId: string): boolean {
    return this.runningProcesses.has(trackingId);
  }

  /**
   * One Claude CLI process per ACP session. Follow-up prompts (the next user
   * message, a steer drained at end_turn) are written to its stdin instead of
   * spawning `claude -p --resume`. A new process re-reads `git status` into
   * the system prompt, so the llama.cpp prefix diverges and the whole
   * conversation is prefilled again. The process is built once, so that
   * snapshot stays byte-identical and only the new turn is prefilled.
   *
   * Dropped (and the next prompt falls back to --resume) when the process
   * exits, the user cancels, or model / effort / permission mode changes —
   * those are spawn flags and cannot be applied to a live process.
   */
  private liveSessions = new Map<string, LiveClaudeSession>();

  private userMessageLine(content: StdinContentBlock[]): string {
    return (
      JSON.stringify({
        type: "user",
        message: { role: "user", content },
        parent_tool_use_id: null,
        session_id: "",
      }) + "\n"
    );
  }

  private liveSpawnKey(modelOverride?: string, effortOverride?: string): string {
    const model = modelOverride || this.config.model || "";
    return `${model}\0${effortOverride ?? ""}\0${this.sessionPermissionMode ?? ""}`;
  }

  private canReuseLive(
    live: LiveClaudeSession,
    resumeId: string | undefined,
  ): boolean {
    if (live.turn) return false;
    // exitCode is null while the process is alive; a number once it has exited.
    if (typeof live.proc.exitCode === "number" || live.proc.stdin?.writable !== true) return false;
    if (!resumeId || live.claudeSessionId !== resumeId) return false;
    return true;
  }

  /** Kill an idle persistent CLI. In-flight turns are cancelled via cancel(). */
  private dropLiveSession(trackingId: string): void {
    const live = this.liveSessions.get(trackingId);
    if (!live) return;
    this.liveSessions.delete(trackingId);
    this.runningProcesses.delete(trackingId);
    try {
      live.proc.stdin?.end();
    } catch {
      // already closed
    }
    this.signalProcessTree(live.proc, "SIGTERM");
  }

  private pushLiveTurn(
    live: LiveClaudeSession,
    content: StdinContentBlock[],
    onEvent: (event: StreamEvent) => void,
  ): Promise<ClaudeResult> {
    return new Promise((resolve, reject) => {
      live.turn = {
        resolve,
        reject,
        onEvent,
        resultText: "",
        sessionId: live.claudeSessionId,
      };
      this.runningProcesses.set(live.trackingId, live.proc);
      live.proc.stdin!.write(this.userMessageLine(content));
    });
  }

  /**
   * Run one turn on the session's long-lived CLI. `replace` starts a brand-new
   * conversation (first prompt, or the post-compact session) and retires
   * whatever process this tracking id was holding.
   */
  private runPersistentTurn(opts: {
    trackingId: string;
    cwd: string | undefined;
    resumeId: string | undefined;
    replace: boolean;
    content: StdinContentBlock[];
    onEvent: (event: StreamEvent) => void;
    modelOverride?: string;
    effortOverride?: string;
  }): Promise<ClaudeResult> {
    const existing = this.liveSessions.get(opts.trackingId);
    const spawnKey = this.liveSpawnKey(opts.modelOverride, opts.effortOverride);
    if (
      !opts.replace &&
      existing &&
      existing.spawnKey === spawnKey &&
      this.canReuseLive(existing, opts.resumeId)
    ) {
      logger.info(
        `Reusing Claude process for session ${opts.trackingId} (no --resume; prefix stays cached)`,
      );
      return this.pushLiveTurn(existing, opts.content, opts.onEvent);
    }
    if (existing) this.dropLiveSession(opts.trackingId);

    const args = this.buildStreamJsonArgs(
      opts.resumeId,
      opts.modelOverride,
      opts.effortOverride,
    );
    logger.debug(`spawn persistent: claude ${maskArgs(args)}`);
    const proc = spawn(this.resolveClaudeBin(), args, {
      cwd: opts.cwd,
      env: this.sanitizeEnv(),
      stdio: ["pipe", "pipe", "pipe"],
      detached: true,
    });
    const live: LiveClaudeSession = {
      proc,
      trackingId: opts.trackingId,
      claudeSessionId: opts.resumeId ?? "",
      spawnKey,
      turn: null,
      buffer: "",
      stderrText: "",
    };
    this.liveSessions.set(opts.trackingId, live);

    const finishTurn = (code: number | null): void => {
      const turn = live.turn;
      live.turn = null;
      // The close handler outlives a replacement spawn. Only forget this
      // process when it is still the one the session is holding.
      if (this.liveSessions.get(opts.trackingId) === live) {
        this.liveSessions.delete(opts.trackingId);
      }
      if (this.runningProcesses.get(opts.trackingId) === live.proc) {
        this.runningProcesses.delete(opts.trackingId);
      }
      const esc = this.cancelEscalationTimers.get(opts.trackingId);
      if (esc) {
        clearTimeout(esc);
        this.cancelEscalationTimers.delete(opts.trackingId);
      }
      if (!turn) return;
      const wasCancelled = this.cancelledProcesses.delete(opts.trackingId);
      if (wasCancelled) {
        turn.resolve({ text: turn.resultText, sessionId: turn.sessionId || live.claudeSessionId });
        return;
      }
      if (code !== 0 && code !== null) {
        const detail = live.stderrText.trim() || turn.resultText.trim();
        turn.reject(
          new Error(`claude exited with code ${code}${detail ? `: ${detail.slice(0, 2000)}` : ""}`),
        );
        return;
      }
      turn.resolve({ text: turn.resultText, sessionId: turn.sessionId || live.claudeSessionId });
    };

    proc.stderr!.on("data", (data: Buffer) => {
      live.stderrText += data.toString();
    });

    proc.stdout!.on("data", (data: Buffer) => {
      live.buffer += data.toString();
      const lines = live.buffer.split("\n");
      live.buffer = lines.pop() ?? "";
      for (const line of lines) {
        if (!line.trim()) continue;
        this.consumeLiveLine(live, line);
      }
    });

    proc.on("close", (code) => {
      if (live.buffer.trim()) {
        this.consumeLiveLine(live, live.buffer);
        live.buffer = "";
      }
      finishTurn(code);
    });
    proc.on("error", (err) => {
      const turn = live.turn;
      live.turn = null;
      if (this.liveSessions.get(opts.trackingId) === live) {
        this.liveSessions.delete(opts.trackingId);
      }
      if (this.runningProcesses.get(opts.trackingId) === live.proc) {
        this.runningProcesses.delete(opts.trackingId);
      }
      turn?.reject(err);
    });

    return this.pushLiveTurn(live, opts.content, opts.onEvent);
  }

  /** Parse one stdout line from a persistent CLI. A `result` ends the turn; the process stays up. */
  private consumeLiveLine(live: LiveClaudeSession, line: string): void {
    let parsed: any;
    try {
      parsed = JSON.parse(line);
    } catch {
      return;
    }
    const turn = live.turn;
    if (turn) this.processStreamLine(parsed, turn.onEvent);
    if (parsed.session_id) {
      live.claudeSessionId = parsed.session_id;
      if (turn) turn.sessionId = parsed.session_id;
    }
    if (parsed.result !== undefined && turn) {
      turn.resultText = parsed.result;
    }
    if (parsed.type === "result" && turn) {
      turn.sessionId = parsed.session_id ?? turn.sessionId;
      turn.resultText = parsed.result ?? turn.resultText;
      // Clear the turn BEFORE resolving so a follow-up prompt queued on the
      // same tick can push the next user message into this same process.
      live.turn = null;
      this.runningProcesses.delete(live.trackingId);
      const esc = this.cancelEscalationTimers.get(live.trackingId);
      if (esc) {
        clearTimeout(esc);
        this.cancelEscalationTimers.delete(live.trackingId);
      }
      turn.resolve({ text: turn.resultText, sessionId: turn.sessionId || live.claudeSessionId });
    }
  }

  private runJson(
    args: string[],
    cwd?: string
  ): Promise<ClaudeResult> {
    return new Promise((resolve, reject) => {
      logger.debug(`spawn: claude ${maskArgs(args)}`);
      const proc = spawn(this.resolveClaudeBin(), args, {
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
      // detached puts claude in its own process group so cancel() can
      // signal(-pgid) the whole tree — a Bash tool's `bash → sleep/ssh`
      // grandchildren keep running (and keep the turn's output flowing) when
      // only the claude parent is signalled, orphaning the turn's work.
      const proc = spawn(this.resolveClaudeBin(), args, {
        cwd,
        env: this.sanitizeEnv(),
        stdio: ["ignore", "pipe", "pipe"],
        detached: true,
      });

      if (trackingId) {
        this.runningProcesses.set(trackingId, proc);
      }

      let buffer = "";
      let resultText = "";
      let sessionId = "";
      // Captured so a non-zero exit can say WHY. Without it the rejection was a
      // bare "claude exited with code 1", which the resume fallback could not
      // tell apart from a stale session id — so a dead upstream cost the whole
      // conversation. See isTransientTurnFailure in agent.ts.
      let stderrText = "";

      proc.stderr!.on("data", (data: Buffer) => {
        stderrText += data.toString();
      });

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
          // The process exited (either from SIGINT or a completed turn) — clear
          // any pending SIGINT→SIGTERM escalation timer so a process that
          // honored SIGINT is never SIGTERMed after the fact.
          const esc = this.cancelEscalationTimers.get(trackingId);
          if (esc) {
            clearTimeout(esc);
            this.cancelEscalationTimers.delete(trackingId);
          }
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
          const detail = stderrText.trim() || resultText.trim();
          reject(new Error(`claude exited with code ${code}${detail ? `: ${detail.slice(0, 2000)}` : ""}`));
          return;
        }

        resolve({ text: resultText, sessionId });
      });

      proc.on("error", reject);
    });
  }

  /**
   * Runs `claude` with the stream-json stdin input protocol: spawns with
   * `--input-format=stream-json`, writes a single SDK `user` message JSON line
   * to stdin (matching the format the CLI's `--input-format=stream-json`
   * consumes — see directConnectManager.sendMessage in Claude Code's source),
   * then streams the output exactly like `runStreaming`.
   *
   * The JSON envelope is:
   *   {"type":"user","message":{"role":"user","content":<blocks>},
   *    "parent_tool_use_id":null,"session_id":""}
   *
   * `content` blocks follow the Anthropic Messages API schema, so an image is
   *   {"type":"image","source":{"type":"base64","media_type":...,"data":...}}
   *
   * The stdin handle is closed after the write: with `--print` the CLI treats
   * a closed stdin as "no more input events", so a single-turn session ends
   * cleanly, and `--resume` batches as expected without hanging.
   */
  private runStreamingJsonLine(
    args: string[],
    cwd: string | undefined,
    content: StdinContentBlock[],
    onEvent: (event: StreamEvent) => void,
    trackingId?: string
  ): Promise<ClaudeResult> {
    return new Promise((resolve, reject) => {
      logger.debug(`spawn streaming-json: claude ${maskArgs(args)}`);
      // detached: own process group — see runStreaming for why cancel()
      // needs to signal the whole tree, not just the claude parent.
      const proc = spawn("claude", args, {
        cwd,
        env: this.sanitizeEnv(),
        stdio: ["pipe", "pipe", "pipe"],
        detached: true,
      });

      if (trackingId) {
        this.runningProcesses.set(trackingId, proc);
      }

      // Write the SDK user message as a single JSON line, then end stdin so the
      // CLI's --print mode knows there are no further input events.
      const message = JSON.stringify({
        type: "user",
        message: { role: "user", content },
        parent_tool_use_id: null,
        session_id: "",
      });
      proc.stdin!.end(message + "\n");

      let buffer = "";
      let resultText = "";
      let sessionId = "";
      // Captured so a non-zero exit can say WHY. Without it the rejection was a
      // bare "claude exited with code 1", which the resume fallback could not
      // tell apart from a stale session id — so a dead upstream cost the whole
      // conversation. See isTransientTurnFailure in agent.ts.
      let stderrText = "";

      proc.stderr!.on("data", (data: Buffer) => {
        stderrText += data.toString();
      });

      proc.stdout!.on("data", (data: Buffer) => {
        buffer += data.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const parsed = JSON.parse(line);
            this.processStreamLine(parsed, onEvent);

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
          // Clear any pending SIGINT→SIGTERM escalation timer (see cancel()).
          const esc = this.cancelEscalationTimers.get(trackingId);
          if (esc) {
            clearTimeout(esc);
            this.cancelEscalationTimers.delete(trackingId);
          }
        }
        const wasCancelled = trackingId
          ? this.cancelledProcesses.delete(trackingId)
          : false;

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

        if (wasCancelled) {
          resolve({ text: resultText, sessionId });
          return;
        }

        if (code !== 0 && code !== null) {
          const detail = stderrText.trim() || resultText.trim();
          reject(new Error(`claude exited with code ${code}${detail ? `: ${detail.slice(0, 2000)}` : ""}`));
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
      const terminalSlashCommands: string[] = Array.isArray(parsed.terminal_slash_commands)
        ? parsed.terminal_slash_commands.filter((c: any) => typeof c === "string")
        : [];
      const skillNames: string[] = Array.isArray(parsed.skills)
        ? parsed.skills.filter((c: any) => typeof c === "string")
        : [];
      const agentNames: string[] = Array.isArray(parsed.agents)
        ? parsed.agents.filter((c: any) => typeof c === "string")
        : [];
      onEvent({ type: "system_init", slashCommands, terminalSlashCommands, skillNames, agentNames });
      return;
    }

    // Handle tool_progress — a periodic heartbeat Claude CLI emits while a
    // tool runs ({"type":"tool_progress","tool_use_id":…,"tool_name":…
    // ,"elapsed_time_seconds":…}, see SDKToolProgressMessageSchema). It rides
    // at the ENVELOPE level (a top-level line, not nested inside
    // `stream_event`), so it must be matched before the unwrap below. While a
    // long Bash tool polls (sleep loops, ssh waits) the CLI emits NOTHING
    // else for it — no deltas, no tool_result — so without this the adapter
    // drops every heartbeat and the chat UI shows a frozen, silent in_progress
    // tool that looks exactly like a hung session. Forwarding it lets agent.ts
    // refresh the tool_call_update with the elapsed time.
    // Defense in depth: even if the tool still fires (e.g. an override kept
    // it, or a session started before the gate existed), relay the elapsed
    // time so agent.ts can surface a watchdog instead of a silent freeze.
    if (parsed.type === "tool_progress") {
      onEvent({
        type: "tool_progress",
        toolUseId: typeof parsed.tool_use_id === "string" ? parsed.tool_use_id : undefined,
        toolName: typeof parsed.tool_name === "string" ? parsed.tool_name : undefined,
        toolElapsedSeconds:
          typeof parsed.elapsed_time_seconds === "number" ? parsed.elapsed_time_seconds : undefined,
        toolSidechain: parsed.parent_tool_use_id != null,
      });
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
        // A subagent's own conversation ends with its own result line; the
        // envelope carries a non-null parent_tool_use_id for sidechains.
        isSidechain: parsed.parent_tool_use_id != null,
      });
    }
  }
}
