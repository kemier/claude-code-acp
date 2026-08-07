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
  type ToolKind,
  type ToolCallContent,
  type ToolCallLocation,
} from "@agentclientprotocol/sdk";
import { SessionStore } from "./session-store.js";
import { ClaudeRunner, type StreamEvent } from "./claude-runner.js";
import { logger } from "./logger.js";
import {
  getAvailableModelsFromSettings,
  getClaudeEnv,
  getDefaultEffortFromSettings,
  getDefaultModelFromSettings,
} from "./claude-settings.js";
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
  { id: "read-only", name: "Read-only", description: "Requires approval to edit files and run commands." },
  { id: "agent", name: "Agent", description: "Read and edit files, and run commands." },
  { id: "agent-full-access", name: "Agent (full access)", description: "Edit files outside workspace and run commands with network access." },
];

const DEFAULT_MODE = "agent";

// ─── Collaboration mode (Default / Plan) ───────────────────────────────────
const COLLABORATION_MODE_CONFIG_ID = "collaboration_mode";
const DEFAULT_COLLABORATION_MODE = "default";
const PLAN_COLLABORATION_MODE = "plan";

const sessionCollaborationModes = new Map<string, string>();

function buildModeState(currentModeId: string) {
  return {
    availableModes: AVAILABLE_MODES,
    currentModeId,
  };
}

/** All possible effort levels (Claude Code CLI --effort flag). */
const ALL_EFFORT_LEVELS: Record<string, { name: string; description: string }> = {
  low: { name: "Low", description: "Fast, minimal reasoning" },
  medium: { name: "Medium", description: "Balanced reasoning" },
  high: { name: "High", description: "Deep reasoning" },
  xhigh: { name: "XHigh", description: "Extra-deep reasoning" },
  max: { name: "Max", description: "Maximum reasoning effort" },
};

/**
 * Optional per-model effort support for known backends (e.g. EricAI via
 * CLAUDE_ACP_AVAILABLE_MODELS). Unknown models (LongCat, etc.) get the
 * standard Claude Code effort set below.
 * Empty array = non-reasoning model (effort picker hidden).
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
  "qwen2.5-14b-1m": [],
};

const DEFAULT_EFFORTS = ["low", "medium", "high"];

function getEffortsForModel(modelId: string): string[] {
  if (Object.prototype.hasOwnProperty.call(MODEL_EFFORTS, modelId)) {
    return MODEL_EFFORTS[modelId];
  }
  return DEFAULT_EFFORTS;
}

/** Default effort: settings.effortLevel when valid, else high if supported. */
function getDefaultEffort(modelId: string): string | undefined {
  const efforts = getEffortsForModel(modelId);
  if (efforts.length === 0) return undefined;
  const fromSettings = getDefaultEffortFromSettings();
  if (fromSettings && efforts.includes(fromSettings)) return fromSettings;
  return efforts.includes("high") ? "high" : efforts[0];
}

/** Session model override, else the model configured in Claude settings. */
function resolveSessionModel(
  store: SessionStore,
  sessionId: string,
  modelOverride?: string,
): string {
  const override =
    modelOverride ?? (store.getConfigOverrides(sessionId)["model"] as string | undefined);
  return override || getDefaultModelFromSettings();
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
  {
    name: "clear",
    description: "Clear conversation history and start fresh.",
    input: null,
  },
  {
    name: "review-branch",
    description: "Review changes relative to a base branch.",
    input: { hint: "branch name" },
  },
  {
    name: "review-commit",
    description: "Review a specific commit.",
    input: { hint: "commit sha" },
  },
  {
    name: "compact",
    description: "Summarize conversation to reduce context usage.",
    input: null,
  },
];

/** Per-session goal state. */
const sessionGoals = new Map<string, { text: string; paused: boolean }>();

// ─── Context window resolution ──────────────────────────────────────────────
// Default context window used when the proxy doesn't report one for a model.
const DEFAULT_CONTEXT_WINDOW = 200_000;

// Cached map of model-id → context_length, fetched once from the proxy's
// /v1/models endpoint at startup. Falls back to DEFAULT_CONTEXT_WINDOW.
const modelContextWindows = new Map<string, number>();
let modelContextWindowsLoaded = false;

/**
 * The proxy URL for direct /v1/models queries.
 * Claude CLI itself routes through ANTHROPIC_BASE_URL (a CCR gateway),
 * so we derive the direct proxy URL from ERICAI_PROXY_URL, falling back
 * to the default port 8317.
 */
function getProxyBaseUrl(): string {
  return (
    process.env.ERICAI_PROXY_URL ||
    process.env.ERICAI_PROXY_BASE_URL ||
    "http://127.0.0.1:8317"
  );
}

/**
 * Fetch /v1/models from the proxy and populate modelContextWindows.
 * Called once at module load; safe to call again to refresh.
 */
export async function loadModelContextWindows(): Promise<void> {
  if (modelContextWindowsLoaded) return;
  modelContextWindowsLoaded = true; // set early to avoid concurrent fetches
  try {
    const url = `${getProxyBaseUrl()}/v1/models`;
    const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) {
      logger.warn(`Failed to fetch /v1/models from proxy (${resp.status}): ${url}`);
      return;
    }
    const body = await resp.json() as { data?: Array<{ id: string; context_length?: number }> };
    if (body.data) {
      for (const m of body.data) {
        if (typeof m.context_length === "number" && m.context_length > 0) {
          modelContextWindows.set(m.id, m.context_length);
        }
      }
      logger.info(`Loaded ${modelContextWindows.size} model context windows from proxy`);
    }
  } catch (err) {
    logger.warn(`Could not load model context windows from proxy: ${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * Context windows reported by Claude Code itself (`modelUsage[...].contextWindow`
 * on the result event). Used when the proxy did not advertise a window for the
 * model, so the denominator comes from the CLI rather than a hardcoded default.
 * The proxy value still wins when present: it reflects the model's true window,
 * whereas the CLI falls back to 200000 unless CLAUDE_CODE_MAX_CONTEXT_TOKENS is
 * set.
 */
const cliReportedContextWindows = new Map<string, number>();

/**
 * Resolve the full context window (in tokens) for a given model.
 *
 * This is the *display* window — the denominator for `usage_update`, matching
 * how Claude Code's status line always uses the model's full window even when
 * the compaction threshold is decoupled from it.
 *
 * Precedence mirrors Claude Code CLI:
 *  1. `CLAUDE_CODE_MAX_CONTEXT_TOKENS` — the documented override for models the
 *     CLI does not recognize as Claude models (i.e. anything routed through
 *     `ANTHROPIC_BASE_URL`, which is exactly our case). Applied directly as of
 *     CLI v2.1.193.
 *  2. The proxy's advertised `context_length` from /v1/models — authoritative
 *     for the real backend model.
 *  3. The window Claude Code itself reported on the result event.
 *  4. DEFAULT_CONTEXT_WINDOW.
 *
 * One deliberate deviation: the override is capped by the model's real window
 * when we know it. The CLI applies the override globally to every model, which
 * is harmless for a larger window (it just compacts earlier than needed) but
 * unsafe for a smaller one — glm-5.1 and deepseek-v4-pro are 131072, so a
 * global 196608 would put the threshold above their real limit and the
 * conversation would overflow without ever compacting. Capping preserves exact
 * parity for the model the override was written for while staying safe when the
 * model picker switches to a smaller one.
 */
export function getContextWindowForModel(modelId: string | undefined): number {
  const known = modelId
    ? modelContextWindows.get(modelId) ?? cliReportedContextWindows.get(modelId)
    : undefined;
  const override = parsePositiveIntEnv(getClaudeEnv("CLAUDE_CODE_MAX_CONTEXT_TOKENS"));
  if (override !== undefined) {
    return known !== undefined ? Math.min(override, known) : override;
  }
  return known ?? DEFAULT_CONTEXT_WINDOW;
}

// ─── Claude Code CLI auto-compact arithmetic ────────────────────────────────
// These mirror the CLI's own constants (services/compact/autoCompact.ts) so the
// ACP layer computes the same threshold the CLI does. Keeping the two in sync
// matters because both can compact: if they disagree, whichever fires first
// wins and the other becomes unreachable, making the configured behaviour a
// coin flip that depends on undocumented internals.

/** Tokens the CLI reserves for the compaction summary's own output. */
const MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000;
/** Headroom the CLI keeps below the effective window before compacting. */
const AUTOCOMPACT_BUFFER_TOKENS = 13_000;
/** The CLI's max-output default for model ids it does not recognize. */
const DEFAULT_MAX_OUTPUT_TOKENS = 32_000;

/** Parse an env var as a positive integer, or undefined when unset/invalid. */
function parsePositiveIntEnv(value: string | undefined): number | undefined {
  if (!value) return undefined;
  const parsed = parseInt(value, 10);
  if (isNaN(parsed) || parsed <= 0) return undefined;
  return parsed;
}

/** Match the CLI's `isEnvTruthy`: 1 / true / yes / on (case-insensitive). */
function isEnvTruthy(value: string | undefined): boolean {
  if (!value) return false;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase().trim());
}

/**
 * Max output tokens per request, as the CLI resolves it:
 * `CLAUDE_CODE_MAX_OUTPUT_TOKENS` when set, else 32000 for unrecognized models.
 */
export function getMaxOutputTokens(): number {
  return (
    parsePositiveIntEnv(getClaudeEnv("CLAUDE_CODE_MAX_OUTPUT_TOKENS")) ??
    DEFAULT_MAX_OUTPUT_TOKENS
  );
}

/**
 * The capacity used for compaction maths.
 *
 * `CLAUDE_CODE_AUTO_COMPACT_WINDOW` lets the compaction threshold be computed
 * against a smaller window than the model really has; the CLI caps it at the
 * real window, so it can only lower the figure.
 */
export function getAutoCompactWindow(modelId: string | undefined): number {
  const full = getContextWindowForModel(modelId);
  const configured = parsePositiveIntEnv(getClaudeEnv("CLAUDE_CODE_AUTO_COMPACT_WINDOW"));
  return configured !== undefined ? Math.min(full, configured) : full;
}

/**
 * Effective window = compaction capacity minus the summary output reserve.
 * Mirrors the CLI's `getEffectiveContextWindowSize()`.
 */
export function getEffectiveContextWindowSize(modelId: string | undefined): number {
  const reserved = Math.min(getMaxOutputTokens(), MAX_OUTPUT_TOKENS_FOR_SUMMARY);
  return getAutoCompactWindow(modelId) - reserved;
}

/**
 * Auto-compact threshold, matching Claude Code CLI exactly:
 *
 *   threshold = effectiveWindow - 13000
 *   with CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=N:
 *     threshold = min(floor(effectiveWindow * N/100), threshold)
 *
 * The percentage applies to the *effective* window (not the raw context
 * window) and can only lower the threshold — values above the default have no
 * effect, per the CLI's own implementation.
 */
export function getAutoCompactThreshold(modelId: string | undefined): number {
  const effective = getEffectiveContextWindowSize(modelId);
  const base = effective - AUTOCOMPACT_BUFFER_TOKENS;
  const pct = parsePositiveIntEnv(getClaudeEnv("CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"));
  if (pct !== undefined && pct <= 100) {
    return Math.min(Math.floor((effective * pct) / 100), base);
  }
  return base;
}

/**
 * Whether auto-compaction should run at all, honoring the CLI's kill switches.
 * `DISABLE_COMPACT` disables everything; `DISABLE_AUTO_COMPACT` disables only
 * the automatic pass and leaves manual /compact working.
 */
export function isAutoCompactEnabled(): boolean {
  if (isEnvTruthy(getClaudeEnv("DISABLE_COMPACT"))) return false;
  if (isEnvTruthy(getClaudeEnv("DISABLE_AUTO_COMPACT"))) return false;
  return true;
}

// ─── Auto-compact ───────────────────────────────────────────────────────────
// Trigger compaction when token usage reaches the same threshold Claude Code
// CLI would use (see getAutoCompactThreshold above), so the two layers agree.
// The CLI checks mid-turn; Claude CLI manages turns internally here, so we
// check between turns — after the prompt finishes but before the next user
// prompt is processed. The compaction replaces the
// old Claude session with a fresh one seeded with a summary.
// Track the last reported usage per session so the auto-compact check
// can run after the turn finishes.
const sessionLastUsage = new Map<string, { input_tokens: number; output_tokens: number }>();

// Monotonic id source for compaction lifecycle tool calls.
let compactionCounter = 0;

// Per-session context estimator for the in-flight prompt() turn. Stored on a
// map (not a local) so doCompact() — which runs in its own scope — can reset
// it after compaction replaces the conversation with a short summary. Without
// a reset, the estimator keeps counting pre-compaction content and over-reports.
const sessionEstimators = new Map<string, TurnContextEstimator>();

// ─── Real-time context estimation ──────────────────────────────────────────
// Claude CLI reports the REAL prompt size on every `assistant` message
// (`message.usage.input_tokens` + cache read/creation tokens = the exact
// number of prompt tokens the provider counted for that API call). That value
// is authoritative and is used as the context size whenever available.
//
// Character-based estimation is only a fallback for the window before the
// first assistant message of a session arrives, and as a short-lived
// increment on top of the last real count so the usage ring still moves while
// a long tool sequence runs. Measured against a real 118k-token transcript,
// the old pure-character estimate was off by ~25% (it read 91,336 for a real
// 118,035), and it counted subagent traffic that never enters the parent's
// context at all — which is what made the ring diverge from reality and let
// auto-compact fire at the wrong time.
//
// Approximate token count for the fallback path: ~3 chars per token. (4 was
// measured to under-report by ~25% on code/JSON-heavy conversations.)
const CHARS_PER_TOKEN = 3;
// Claude CLI system prompt is ~7KB. Tools (24 defs) are ~65KB. These are sent
// on EVERY API call. Pre-computed: 7148 + 65600 = 72748 bytes overhead.
const CLAUDE_OVERHEAD_BYTES = 72748;

/** Estimate tokens from a character count (rough: 3 chars ≈ 1 token). */
function estimateTokensFromChars(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

/** Per-turn accumulator for estimating real-time context size. */
class TurnContextEstimator {
  private contentBytes = 0;
  private apiCallCount = 0;
  /**
   * Real prompt tokens from the most recent non-sidechain assistant message.
   * 0 until the first one arrives. Authoritative when > 0.
   */
  private realTokens = 0;
  /** Content accumulated since `realTokens` was last refreshed. */
  private bytesSinceReal = 0;

  /** Called when a new assistant event arrives (marks an API call boundary). */
  markApiCall(): void {
    this.apiCallCount++;
  }

  /**
   * Record the REAL prompt size reported by Claude CLI for one API call.
   * Supersedes character estimation from this point on. Callers must skip
   * sidechain (subagent) events — those measure a different conversation.
   */
  setRealTokens(tokens: number): void {
    if (!Number.isFinite(tokens) || tokens <= 0) return;
    this.realTokens = tokens;
    this.bytesSinceReal = 0;
  }

  /** True once a real per-call token count has been observed. */
  hasRealTokens(): boolean {
    return this.realTokens > 0;
  }

  /** Accumulate text content (assistant messages, thinking, tool results). */
  addText(text: string): void {
    if (text) {
      this.contentBytes += text.length;
      this.bytesSinceReal += text.length;
    }
  }

  /**
   * Reset accumulated conversation content (keep the fixed overhead).
   * Called after compaction, which replaces the entire conversation with a
   * short summary — without this, the estimator would keep counting the
   * pre-compaction content and over-report context size. The real count is
   * cleared too: it describes the pre-compaction conversation, so keeping it
   * would pin usage at its pre-compaction value until the next API call.
   */
  reset(): void {
    this.contentBytes = 0;
    this.apiCallCount = 0;
    this.realTokens = 0;
    this.bytesSinceReal = 0;
  }

  /** Accumulate tool call content (JSON-serialized input). */
  addToolCall(toolName: string, toolInput: unknown): void {
    let added = toolName.length;
    try {
      added += JSON.stringify(toolInput ?? {}).length;
    } catch {
      added += 50; // fallback estimate
    }
    this.contentBytes += added;
    this.bytesSinceReal += added;
  }

  /**
   * Context size in tokens.
   *
   * Once a real per-call count has arrived it is reported VERBATIM. Adding a
   * character-based increment on top used to inflate it: tool results and
   * subagent traffic were counted at a guessed chars-per-token ratio, and
   * subagent content never enters this conversation's prompt at all. The real
   * figure refreshes on every API call (message_delta), so holding it steady
   * between calls is both accurate and what Claude Code's own UI does.
   *
   * The character estimate is only used before the first real count arrives.
   */
  getEstimatedTokens(): number {
    if (this.realTokens > 0) return this.realTokens;
    return estimateTokensFromChars(CLAUDE_OVERHEAD_BYTES + this.contentBytes);
  }

  getApiCallCount(): number {
    return this.apiCallCount;
  }
}

/** Send a usage_update with the estimated context size (real-time, mid-turn). */
function sendEstimatedUsage(
  estimator: TurnContextEstimator,
  connection: AgentSideConnection,
  sessionId: string,
  modelId: string | undefined,
): void {
  const contextWindow = getContextWindowForModel(modelId);
  const estimated = estimator.getEstimatedTokens();
  // Cap at contextWindow so the UI never shows >100%.
  const capped = Math.min(estimated, contextWindow);
  connection.sessionUpdate({
    sessionId,
    update: {
      sessionUpdate: "usage_update",
      used: capped,
      size: contextWindow,
    } as any,
  }).catch(() => {
    // Best effort — don't fail the turn.
  });
}

/**
 * Basename of a file path for compact titles (e.g. "/a/b/c.ts" → "c.ts").
 * Falls back to the full path if it's already short or not a string.
 */
function shortName(filePath: unknown): string {
  const p = typeof filePath === "string" ? filePath : "";
  if (!p) return "";
  const base = p.split("/").pop() || p;
  return base.length <= 48 ? base : base.slice(0, 45) + "…";
}

/**
 * True if `line` ends with an unclosed single or double quote — i.e. an odd
 * number of that quote char on the line, so the quoted region opens here and
 * closes on a later line (e.g. `node -e '`, `bash -c "…`). Used to detect
 * multi-line quoted scripts whose first line would render as a truncated title.
 */
function hasUnclosedQuote(line: string): boolean {
  const single = (line.match(/'/g) ?? []).length;
  const double = (line.match(/"/g) ?? []).length;
  return single % 2 === 1 || double % 2 === 1;
}

/**
 * Split Claude Code's `cd <cwd>` prefix off a Bash command.
 *
 * Claude Code CLI prepends the working directory to EVERY Bash command, as
 * either `cd <dir> && <cmd>` or `cd <dir>\n<cmd>`. ACP clients surface the
 * command from `rawInput.command`; vscode-acp specifically takes its FIRST
 * line and, when it differs from the agent-supplied `title`, overwrites the
 * title with it (ChatWebviewProvider.ts: `firstLine = detailHint.split('\n')[0]`
 * → `displayTitle = firstLine`). With the `cd` still attached, every execute
 * tool call therefore renders as `cd /workspace/...` and the real command is
 * never shown.
 *
 * Reporting the directory out-of-band as `cwd` (the shape codex-acp uses:
 * `rawInput: { command, cwd }`) makes the first line of `command` the actual
 * command, so the client's override becomes a no-op and no information is lost.
 *
 * Returns the command with the `cd` clause removed plus the extracted `cwd`.
 * When no `cd` prefix is present, the command is returned unchanged and `cwd`
 * is undefined.
 */
export function splitCdPrefix(command: string): { command: string; cwd?: string } {
  const nl = command.indexOf("\n");
  const firstLine = (nl === -1 ? command : command.slice(0, nl)).trim();
  const rest = nl === -1 ? "" : command.slice(nl + 1);

  // `cd <dir> && <inline rest>` — cd and the command share the first line.
  const cdAnd = firstLine.match(/^cd\s+(\S[^&]*?)\s*&&\s*(.+)$/);
  if (cdAnd) {
    const inline = cdAnd[2].trim();
    const remainder = rest ? `${inline}\n${rest}` : inline;
    return { command: remainder, cwd: cdAnd[1].trim() };
  }

  // Bare `cd <dir>` on its own line — the command starts on the next line.
  const cdOnly = firstLine.match(/^cd\s+(\S.*)$/);
  if (cdOnly && rest.trim()) {
    return { command: rest.replace(/^\n+/, ""), cwd: cdOnly[1].trim() };
  }

  return { command };
}

/**
 * Derive a one-line title for a Bash command. Claude Code prefixes commands
 * with `cd <cwd>` (either as a leading `cd <dir> && <cmd>` on one line, or as
 * a `cd <dir>` line followed by `\n<cmd>`). vscode-acp renders only the first
 * line of the title, so the raw command would show as `✓ cd /workspace/...`
 * with the real command hidden. Strip the leading `cd` so the meaningful
 * command is visible. Multi-line commands collapse to their first non-empty,
 * non-`cd` line. The full command is still preserved in `rawInput`.
 */
function titleForCommand(command: string, description?: string): string {
  const c = command.trim();
  // Claude Code prefixes commands with `cd <cwd>`. Two shapes occur:
  //   1. `cd <dir> && <cmd>` — cd and the real command on ONE line.
  //   2. `cd <dir>\n<cmd>`  — cd alone on its own line, command on the next.
  // vscode-acp renders only the first line of the title, so the raw command
  // would show as `✓ cd /workspace/...` with the real command hidden. Strip
  // the leading `cd` clause and collapse to the first meaningful line.
  //
  // Process line-by-line from the top so the `&&` strip never crosses a
  // newline (a prior `[^&]*&&` regex ate across `\n` and dropped real
  // commands like `git diff --stat` when a later line contained `&&`).
  const lines = c.split("\n");
  for (const ln of lines) {
    let t = ln.trim();
    if (!t) continue;
    // `cd <dir> && <rest>` on a single line → keep just <rest>.
    const cdAnd = t.match(/^cd\s+\S[^&]*&&\s*(.*)$/);
    if (cdAnd) {
      t = cdAnd[1].trim();
      if (!t) continue; // `cd <dir> && ` with nothing after — next line.
    } else if (/^cd\s+\S/.test(t)) {
      // Bare leading `cd <dir>` line — the real command is on a later line.
      continue;
    }
    // Skip comment-only lines so a `# note` after `cd` isn't shown as the
    // title instead of the actual command.
    if (/^(#|\/\/)/.test(t)) continue;
    // A line ending in an UNCLOSED quote (e.g. `node -e '`, `bash -c "`)
    // opens a multi-line quoted script that closes many lines later. Showing
    // it verbatim renders as a truncated `node -e '` title. If the quote is
    // unclosed on this line, prefer the tool's `description` field; otherwise
    // append an ellipsis so it reads as a continuation rather than complete.
    if (hasUnclosedQuote(t) && description) {
      return description.length <= 120 ? description : description.slice(0, 117) + "…";
    }
    return t.length <= 120 ? t : t.slice(0, 117) + "…";
  }
  // Only `cd`/comment/blank lines (or empty) — no real command was found. This
  // happens for a standalone `cd <dir>` with no `&&` suffix and no following
  // command line: the loop skipped the `cd` line and found nothing after it,
  // so falling back to the last line would render the bare `cd /workspace/…`
  // path as the title. Claude Code supplies a meaningful `description` for
  // ~97% of Bash commands (e.g. "Change to project directory"), so prefer it;
  // only fall back to the raw last line when no description exists.
  if (description) {
    return description.length <= 120 ? description : description.slice(0, 117) + "…";
  }
  const last = lines[lines.length - 1]?.trim() ?? c;
  return last.length <= 120 ? last : last.slice(0, 117) + "…";
}

/**
 * Build an ACP tool_call session update from a Claude CLI tool_use event,
 * mirroring codex-acp's CodexToolCallMapper: a descriptive title, the correct
 * `kind`, and — for file edits — a diff `content` block plus `locations` so
 * vscode-acp renders the action verb and a click-to-open diff instead of a
 * bare filename + checkmark.
 *
 * Claude tool_input shapes:
 * - Read / Write / Edit: { file_path, ... }
 * - Edit: { file_path, old_string, new_string }
 * - MultiEdit: { file_path, edits: [{ old_string, new_string }] }
 * - Bash: { command }
 * - Glob: { pattern }, Grep: { pattern, path? }, etc.
 */
function buildToolCallUpdate(
  toolCallId: string,
  toolName: string,
  toolInput: unknown,
): {
  sessionUpdate: "tool_call";
  toolCallId: string;
  title: string;
  kind: ToolKind;
  status: "completed";
  rawInput: unknown;
  content?: ToolCallContent[];
  locations?: ToolCallLocation[];
} {
  const input = (toolInput ?? {}) as Record<string, any>;
  const filePath: string | undefined =
    typeof input.file_path === "string" ? input.file_path : undefined;
  const name = shortName(filePath);

  switch (toolName) {
    case "Read": {
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: filePath ? `Read file '${name}'` : "Read file",
        kind: "read",
        status: "completed",
        rawInput: toolInput ?? {},
        ...(filePath ? { locations: [{ path: filePath }] } : {}),
      };
    }
    case "Write": {
      // New/overwritten file: oldText null, newText = full content.
      const content = typeof input.content === "string" ? input.content : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: filePath ? `Write file '${name}'` : "Write file",
        kind: "edit",
        status: "completed",
        rawInput: toolInput ?? {},
        content: filePath
          ? [{ type: "diff", path: filePath, oldText: null, newText: content, _meta: { kind: "add" } }]
          : undefined,
        ...(filePath ? { locations: [{ path: filePath }] } : {}),
      };
    }
    case "Edit": {
      const oldText = typeof input.old_string === "string" ? input.old_string : "";
      const newText = typeof input.new_string === "string" ? input.new_string : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: filePath ? `Edit file '${name}'` : "Edit file",
        kind: "edit",
        status: "completed",
        rawInput: toolInput ?? {},
        content: filePath
          ? [{ type: "diff", path: filePath, oldText, newText, _meta: { kind: "update" } }]
          : undefined,
        ...(filePath ? { locations: [{ path: filePath }] } : {}),
      };
    }
    case "MultiEdit": {
      // Merge all edits into one diff: oldText = joined old_strings,
      // newText = joined new_strings (good enough for a visual diff preview).
      const edits: Array<{ old_string?: string; new_string?: string }> =
        Array.isArray(input.edits) ? input.edits : [];
      const oldText = edits.map((e) => e.old_string ?? "").join("\n");
      const newText = edits.map((e) => e.new_string ?? "").join("\n");
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: filePath ? `Edit file '${name}'` : "Edit file",
        kind: "edit",
        status: "completed",
        rawInput: toolInput ?? {},
        content: filePath
          ? [{ type: "diff", path: filePath, oldText, newText, _meta: { kind: "update" } }]
          : undefined,
        ...(filePath ? { locations: [{ path: filePath }] } : {}),
      };
    }
    case "Bash":
    case "bash": {
      const command =
        typeof input.command === "string"
          ? input.command
          : Array.isArray(input.command)
            ? input.command.join(" ")
            : toolName;
      const description =
        typeof input.description === "string" ? input.description : undefined;
      // Report the working directory out-of-band instead of leaving Claude
      // Code's `cd <cwd>` prefix inside `command`. ACP clients derive the
      // displayed command from the first line of `rawInput.command`, so the
      // prefix would mask every command (see splitCdPrefix).
      const split = splitCdPrefix(command);
      const rawInput: Record<string, unknown> = {
        ...(typeof toolInput === "object" && toolInput !== null ? input : {}),
        command: split.command,
      };
      if (split.cwd !== undefined) rawInput.cwd = split.cwd;
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: titleForCommand(command, description),
        kind: "execute",
        status: "completed",
        rawInput,
      };
    }
    case "Glob": {
      const pattern = typeof input.pattern === "string" ? input.pattern : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: pattern ? `Search for '${pattern}'` : "Search files",
        kind: "search",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    case "Grep": {
      const pattern = typeof input.pattern === "string" ? input.pattern : "";
      const path = typeof input.path === "string" ? input.path : "";
      const title = pattern && path
        ? `Search for '${pattern}' in ${shortName(path)}`
        : pattern
          ? `Search for '${pattern}'`
          : "Search";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title,
        kind: "search",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    case "Skill": {
      // Claude Code's Skill tool invokes a skill (loading its SKILL.md and
      // references). Mirror ericli's "Read skill: <name>" convention so
      // vscode-acp's detectSkillLoad() recognizes it and folds it into the
      // collapsed "loaded N skill(s)" summary instead of counting it as a
      // generic execute/command. rawInput keeps the skill name under `skill`
      // (detectSkillLoad's EricCLI branch reads ri.name || ri.skill).
      const skillName =
        typeof input.skill === "string" ? input.skill :
        typeof input.name === "string" ? input.name : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: skillName ? `Read skill: ${skillName}` : "Read skill",
        kind: "execute",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    case "NotebookEdit": {
      // Notebook cell edit — treat like a file edit (kind:edit) so it
      // categorizes with the other edits in the collapsed summary.
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: filePath ? `Edit notebook '${name}'` : "Edit notebook",
        kind: "edit",
        status: "completed",
        rawInput: toolInput ?? {},
        ...(filePath ? { locations: [{ path: filePath }] } : {}),
      };
    }
    case "WebSearch": {
      // Mirrors codex-acp/ericli: kind:search, "Web search: <query>".
      const query =
        typeof input.query === "string" ? input.query :
        typeof input.prompt === "string" ? input.prompt : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: query ? `Web search: ${query}` : "Web search",
        kind: "search",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    case "WebFetch": {
      // kind:fetch (the only ACP fetch kind), "Fetch '<url>'".
      const url = typeof input.url === "string" ? input.url : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: url ? `Fetch '${url}'` : "Fetch URL",
        kind: "fetch",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    case "Task":
    case "Agent": {
      // Subagent spawn. Title from the task description/prompt (codex-acp
      // uses "Start subagent <name>"); fall back to a generic label. Status
      // is overridden to in_progress by the caller's subagent lifecycle.
      const desc =
        typeof input.description === "string" ? input.description :
        typeof input.prompt === "string" ? input.prompt :
        typeof input.subagent_type === "string" ? input.subagent_type : "";
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: desc ? `Start subagent: ${desc.slice(0, 60)}` : "Start subagent",
        kind: "other",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
    default: {
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: toolName,
        kind: "execute",
        status: "completed",
        rawInput: toolInput ?? {},
      };
    }
  }
}

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

/**
 * Build the prompt for /review-branch and /review-commit. The model has Bash
 * tool access, so it runs the diff command itself and reviews the result in a
 * single turn — standing in for codex-acp's app-server-side review/start.
 */
function buildReviewTurnPrompt(diffCommand: string, scope: string): string {
  return [
    `Please review ${scope}.`,
    ``,
    `Run \`${diffCommand}\` to see the changes, then review them for:`,
    `- correctness and potential bugs`,
    `- security issues`,
    `- readability and style`,
    `- anything that could be simplified or is missing (e.g. tests, error handling)`,
    ``,
    `Report findings grouped by severity (critical → minor), each with the file and line. If there is nothing to review, say so.`,
  ].join("\n");
}

function buildConfigOptions(store: SessionStore, sessionId: string) {
  const availableModels = getAvailableModelsFromSettings();
  const modelOverride = store.getConfigOverrides(sessionId)["model"] as string | undefined;
  let currentModel = modelOverride || getDefaultModelFromSettings();
  // Picker options come strictly from settings/env. If the resolved default is
  // not in that list (e.g. settings.model vs CLAUDE_ACP_AVAILABLE_MODELS),
  // snap to the first available model. Session overrides outside the list stay
  // visible so a just-selected value is not dropped before settings update.
  if (!availableModels.some((m) => m.id === currentModel)) {
    if (modelOverride) {
      availableModels.unshift({ id: modelOverride, name: modelOverride });
    } else {
      currentModel = availableModels[0].id;
    }
  }
  const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
  const effortOverride = store.getConfigOverrides(sessionId)["effort"] as string | undefined;
  const currentCollabMode = sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE;

  // Get effort levels supported by the CURRENT model
  const modelEfforts = getEffortsForModel(currentModel);
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
      description: "Approval and sandboxing preset for the session",
      options: AVAILABLE_MODES.map((m) => ({
        value: m.id,
        name: m.name,
        description: m.description,
      })),
      currentValue: currentMode,
    },
    {
      id: COLLABORATION_MODE_CONFIG_ID,
      name: "Collaboration mode",
      type: "select" as const,
      category: "collaboration_mode",
      description: "How the agent collaborates for subsequent turns",
      options: [
        { value: DEFAULT_COLLABORATION_MODE, name: "Default" },
        { value: PLAN_COLLABORATION_MODE, name: "Plan", description: "Plan before making changes" },
      ],
      currentValue: currentCollabMode,
    },
    {
      id: "model",
      name: "Model",
      type: "select" as const,
      category: "model",
      description: "Model from Claude Code settings (~/.claude/settings.json)",
      options: availableModels.map((m) => ({
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
  // Sessions whose steer queue is currently being drained. drainSteerQueue()
  // awaits LLM turns, so without this a second call (pre-turn vs post-turn vs
  // post-compaction) could run concurrently and interleave two consumers over
  // the same queue, delivering messages out of order.
  const steerDraining = new Set<string>();

  // ─── Per-session prompt serialization ─────────────────────────────────────
  // The ACP SDK dispatches each incoming `session/prompt` request concurrently
  // — its read loop (acp.js #receive) calls #processMessage without awaiting
  // it, so a second prompt that arrives while the first is still streaming
  // (e.g. vscode-acp sends a read-only slash command like /skills mid-turn)
  // runs a SECOND prompt() handler on the same session in parallel. That race
  // corrupts shared per-session state (sessionTodos, sessionEstimators, the
  // Claude --resume session id, the onEvent closure) and can drop the slash
  // command's agent_message_chunk entirely. Serialize prompts per session: a
  // prompt that arrives mid-turn simply waits for the in-flight one to finish
  // before running. Read-only slash commands are fast (no LLM turn), so the
  // wait is short; mutating commands (/compact, /clear) and regular prompts
  // queue behind the active turn as the user expects.
  const sessionPromptLocks = new Map<string, Promise<unknown>>();

  // True for the entire duration of a runPrompt() call — from entry through
  // the LLM turn, steer drain, auto-compact, and the end_turn return. Unlike
  // runner.isRunning() (which flips false the moment the Claude subprocess
  // closes, before the drain runs) this spans the whole turn tail. Used by
  // handleSteer() to decide whether a steer should be queued vs. dropped: a
  // steer typed in the last moments of streaming arrives after the subprocess
  // closes but while a turn is still in flight, and must be queued (then
  // drained at line ~2056) rather than silently dropped. Cleared in a finally
  // so it can't get stuck true if runPrompt throws.
  const sessionPromptInFlight = new Set<string>();

  // ─── Subagent (Agent tool) lifecycle tracking ─────────────────────────────
  // The Agent tool spawns a subagent that runs for an extended period. We
  // must NOT mark it "completed" the instant its tool_use block appears (as
  // we do for Read/Edit/etc.) — the subagent is still running and the UI
  // should reflect that. Instead we emit status:"in_progress" when the
  // tool_use fires and flip to "completed" only when the matching tool_result
  // arrives (the user/tool_result message Claude emits once the subagent
  // finishes). Maps Claude's native tool_use id → our ACP toolCallId.
  const sessionPendingAgents = new Map<string, Map<string, string>>();

  // Non-subagent tools (Bash/Read/Edit/Grep/…) use the same in_progress →
  // completed lifecycle, but resolve on the tool_result message (not a
  // <task-notification>). The runner collects the tool_use_ids a user message
  // resolves (claude-runner.ts `toolResultIds`); we flip each matching pending
  // tool to "completed" when that user event arrives. Without this, every tool
  // is emitted as status:"completed" at the tool_use and the chat UI never
  // shows a running phase — the command text appears only after the tool ends.
  // Maps sessionId → (Claude tool_use_id → ACP toolCallId).
  const sessionPendingTools = new Map<string, Map<string, string>>();

  // ─── Todo list (plan) tracking ───────────────────────────────────────────
  // Claude Code's TaskCreate/TaskUpdate/TaskList tools manage a todo list.
  // We mirror them into ACP `session/update` with sessionUpdate:"plan" +
  // entries:PlanEntry[] so vscode-acp renders a plan bar with the task count
  // (like codex-acp's update_plan). The plan is a full snapshot replacement
  // each time, so the count reflects the current list size — not a running
  // total of updates (the user wants "how many todo list", not "how many
  // updates").
  // taskId → { content, status }. Order preserved by insertion.
  const sessionTodos = new Map<string, Map<string, { content: string; status: string }>>();

  /** Normalize Claude's task status to an ACP PlanEntryStatus. */
  function toPlanStatus(s: unknown): "pending" | "in_progress" | "completed" {
    const v = String(s ?? "").toLowerCase();
    if (v === "completed" || v === "done") return "completed";
    if (v === "in_progress" || v === "in-progress" || v === "running") return "in_progress";
    return "pending";
  }

  /**
   * Apply a TaskCreate/TaskUpdate/TaskList tool call to the per-session todo
   * map and emit a `plan` session update snapshot. Returns true if the event
   * was a task tool (and thus handled here).
   *
   * Claude tool_input shapes observed from GLM-5.2 (two valid TaskCreate forms):
   * - Batch:    { tasks: [{ subject, description, activeForm }] }
   * - Scalar:   { subject, description, activeForm }   (one task, top-level)
   *   ids are assigned by Claude CLI and surface only in the tool_result, so
   *   we assign synthetic ids `task-N` in creation order.
   * - TaskUpdate: { taskId, status?, subject?, addBlockedBy?, owner? }
   *   `taskId` is a numeric string ("1", "2") referencing the Nth task, OR a
   *   Claude-internal id we never saw. We match by synthetic id (`task-N`),
   *   then by `task-N` derived from a bare numeric id, then by subject.
   * - TaskList: {} (no-op; the list is already tracked)
   */
  function handleTaskTool(
    sessionId: string,
    toolName: string,
    toolInput: unknown,
  ): boolean {
    if (toolName !== "TaskCreate" && toolName !== "TaskUpdate" && toolName !== "TaskList") {
      return false;
    }
    let todos = sessionTodos.get(sessionId);
    if (!todos) {
      todos = new Map();
      sessionTodos.set(sessionId, todos);
    }

    const input = (toolInput ?? {}) as Record<string, any>;

    if (toolName === "TaskCreate") {
      // Accept BOTH schemas: a `tasks` array (batch) OR a single task
      // described by top-level subject/description/activeForm (scalar).
      // GLM-5.2 emits the scalar form most often.
      const batch: Array<Record<string, any>> = Array.isArray(input.tasks)
        ? input.tasks
        : [];
      const scalar =
        typeof input.subject === "string" || typeof input.description === "string"
          ? [input]
          : [];
      const tasks = batch.length ? batch : scalar;
      for (const t of tasks) {
        const content =
          (typeof t.subject === "string" && t.subject) ||
          (typeof t.description === "string" && t.description) ||
          (typeof t.activeForm === "string" && t.activeForm) ||
          "task";
        // Assign a synthetic id in creation order. Claude's real task ids
        // only surface in the tool_result (not the tool_use input), so we
        // can't key on them; we match TaskUpdate by derived id/subject below.
        const id = `task-${todos.size + 1}`;
        todos.set(id, { content, status: "pending" });
      }
      logger.debug(`TaskCreate: +${tasks.length} task(s), list now ${todos.size}`);
    } else if (toolName === "TaskUpdate") {
      const rawTaskId = String(input.taskId ?? input.task_id ?? "");
      const status = toPlanStatus(input.status);
      const subject =
        typeof input.subject === "string" && input.subject.trim()
          ? input.subject.trim()
          : "";
      // Resolve a bare numeric id ("1") to our synthetic `task-N` form so a
      // status flip keyed only by position still lands on the right entry.
      let synthId = rawTaskId;
      if (rawTaskId && !todos.has(rawTaskId) && /^\d+$/.test(rawTaskId)) {
        synthId = `task-${rawTaskId}`;
      }
      // Try to match an existing entry: synthetic id first, then by subject
      // content. Claude's TaskUpdate sometimes references Claude-internal ids
      // we never saw, so id matching can fail — subject matching is the
      // reliable fallback. We must NOT add a spurious new entry on a failed
      // match (that would inflate the count and is what caused "3 created,
      // only N shown" drift).
      let matchedId: string | null = null;
      if (synthId && todos.has(synthId)) {
        matchedId = synthId;
      } else if (subject) {
        for (const [id, t] of todos) {
          if (t.content === subject) { matchedId = id; break; }
        }
      }
      if (matchedId) {
        const existing = todos.get(matchedId)!;
        if (subject) existing.content = subject;
        // Only mutate status when the update actually carries one. A
        // TaskUpdate with only `addBlockedBy`/`owner` (no `status`) must not
        // reset an in_progress task back to pending.
        if (input.status !== undefined) existing.status = status;
      } else if (subject) {
        // Update for a task we didn't see created AND we can't match by
        // subject — only then add it, so the count reflects a genuinely new
        // tracked item. Bare taskId with no subject is ignored (can't render
        // it meaningfully and it's almost certainly a Claude-internal id).
        const id = `task-${todos.size + 1}`;
        todos.set(id, { content: subject, status });
      }
      logger.debug(`TaskUpdate: taskId=${rawTaskId} status=${input.status ?? "(none)"} matched=${matchedId ?? "none"}, list ${todos.size}`);
    }
    // TaskList: nothing to apply; we already track the list.

    // Emit a full plan snapshot — but ONLY when there is at least one tracked
    // todo. An empty snapshot would render as "Plan (no content)" in the UI,
    // which happens when the model calls TaskUpdate on tasks it never created
    // (e.g. GLM-5.2 replaying task ids it saw in memory/prior context across a
    // new session where sessionTodos is empty). In that case there is genuinely
    // nothing to plan, so we skip the emission rather than flash an empty bar.
    if (todos.size === 0) {
      return true;
    }
    const entries = Array.from(todos.entries()).map(([_id, t]) => ({
      content: t.content,
      priority: "medium" as const,
      status: t.status as "pending" | "in_progress" | "completed",
    }));
    connection.sessionUpdate({
      sessionId,
      update: {
        sessionUpdate: "plan",
        entries,
      } as any,
    }).catch(() => {
      // Best effort.
    });
    return true;
  }


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
    // A turn is "active" for steering while EITHER the Claude subprocess is
    // streaming OR a runPrompt() is still in flight. runner.isRunning() flips
    // false the moment the subprocess closes (claude-runner.ts proc.on('close'))
    // — but runPrompt() still runs the steer drain + auto-compact + return
    // end_turn after that. Without the in-flight check, a steer typed in the
    // last moments of streaming (subprocess closed, runPrompt in its drain
    // tail) would hit this guard, log "no active turn", and be silently
    // dropped instead of queued — so it never appeared in the chat before the
    // next turn.
    if (!runner.isRunning(sessionId) && !sessionPromptInFlight.has(sessionId)) {
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

  /**
   * Reusable compaction flow — used by both the /compact slash command
   * and the auto-compact trigger. Asks the LLM to summarize the current
   * conversation, then replaces the old Claude session with a fresh one
   * seeded with the summary.
   *
   * Returns true if compaction succeeded.
   */
  /**
   * Dispatch a built-in slash command (/ps, /clear, /compact, /status,
   * /skills, /goal). Shared by prompt() and the steer-drain loop so both entry
   * points behave identically — previously the steer path was a hand-rewritten
   * copy that silently dropped several cases.
   *
   * Returns:
   *   - { handled: false }            — not a slash command (or unknown); caller should send it as a plain prompt.
   *   - { handled: true, startTurn? } — command handled. If startTurn is set, the caller should start an LLM turn with that prompt (the /goal case). Otherwise the command was fully serviced (read-only, /clear, /compact, /goal clear|pause|resume) and the caller should do nothing further.
   */
  async function dispatchSlashCommand(
    sessionId: string,
    connection: AgentSideConnection,
    runner: ClaudeRunner,
    store: SessionStore,
    text: string,
  ): Promise<{ handled: boolean; startTurn?: string }> {
    const command = parseSlashCommand(text);
    if (!command) return { handled: false };
    const { name, rest } = command;

    if (name === "ps") {
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "No background terminals running." },
        } as any,
      });
      return { handled: true };
    }

    if (name === "clear") {
      const cwd = store.getCwd(sessionId)!;
      store.clearPersistedSession(cwd);
      store.setClaudeSessionId(sessionId, "");
      sessionGoals.delete(sessionId);
      sessionTodos.delete(sessionId);
      // Clearing drops the entire conversation, so the context estimate must
      // reset too — otherwise the ring would keep showing the pre-clear size.
      const clearedEstimator = sessionEstimators.get(sessionId);
      if (clearedEstimator) clearedEstimator.reset();
      sessionLastUsage.delete(sessionId);
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Conversation cleared. Starting fresh." },
        } as any,
      });
      return { handled: true };
    }

    if (name === "review-branch") {
      if (!rest) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: 'Command "/review-branch" requires branch name.' },
          } as any,
        });
        return { handled: true };
      }
      // Like /goal, return a prompt the caller starts an LLM turn with. The
      // model has Bash tool access, so it gathers the diff itself (mirrors how
      // codex-acp's review delegates diff-gathering to its app-server, which
      // here is just the agent's own tool use).
      return { handled: true, startTurn: buildReviewTurnPrompt(`git diff ${rest}...HEAD`, `changes on this branch relative to \`${rest}\``) };
    }

    if (name === "review-commit") {
      if (!rest) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: 'Command "/review-commit" requires commit sha.' },
          } as any,
        });
        return { handled: true };
      }
      return { handled: true, startTurn: buildReviewTurnPrompt(`git show ${rest}`, `commit ${rest}`) };
    }

    if (name === "compact") {
      const compactEffort = resolveEffort(store, sessionId);
      await doCompact(sessionId, connection, runner, store, store.getConfigOverrides(sessionId)["model"] as string | undefined, compactEffort, true);
      return { handled: true };
    }

    if (name === "status") {
      const cwd = store.getCwd(sessionId)!;
      const configOverrides = store.getConfigOverrides(sessionId);
      const currentModel = resolveSessionModel(store, sessionId, configOverrides["model"] as string | undefined);
      const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
      const goal = sessionGoals.get(sessionId);
      // Context usage, using the same figures that drive auto-compaction so
      // what /status reports and what actually triggers cannot disagree.
      const contextWindow = getContextWindowForModel(currentModel);
      const threshold = getAutoCompactThreshold(currentModel);
      const usage = sessionLastUsage.get(sessionId);
      const usedTokens = usage ? usage.input_tokens + usage.output_tokens : 0;
      const usedPct = contextWindow > 0 ? (usedTokens / contextWindow) * 100 : 0;
      const compactState = isAutoCompactEnabled()
        ? `at ${threshold.toLocaleString()} tokens (${((threshold / contextWindow) * 100).toFixed(1)}% of window)`
        : "disabled";
      const lines = [
        `**Model:** ${currentModel}`,
        `**Mode:** ${currentMode}`,
        `**Directory:** ${cwd}`,
        `**Session:** \`${sessionId}\``,
        goal ? `**Goal:** ${goal.paused ? "(paused) " : ""}${goal.text.slice(0, 100)}` : "**Goal:** none",
        `**Context:** ${usedTokens.toLocaleString()} / ${contextWindow.toLocaleString()} tokens (${usedPct.toFixed(1)}%)`,
        `**Max output:** ${getMaxOutputTokens().toLocaleString()} tokens`,
        `**Auto-compact:** ${compactState}`,
      ];
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: lines.join("\n") },
        } as any,
      });
      return { handled: true };
    }

    if (name === "skills") {
      const { readdirSync, readFileSync, existsSync } = await import("node:fs");
      const { join: joinPath, dirname } = await import("node:path");
      const { homedir } = await import("node:os");

      const cwd = store.getCwd(sessionId)!;
      const skillsMap = new Map<string, { description: string; source: string }>();

      const scanSkillsDir = (skillsDir: string, source: string) => {
        if (!existsSync(skillsDir)) return;
        let entries;
        try {
          entries = readdirSync(skillsDir, { withFileTypes: true });
        } catch {
          return;
        }
        for (const entry of entries) {
          if (!entry.isDirectory()) continue;
          const skillFile = joinPath(skillsDir, entry.name, "SKILL.md");
          let description = "";
          if (existsSync(skillFile)) {
            try {
              const content = readFileSync(skillFile, "utf-8");
              const descMatch = content.match(/^description:\s*(.+)$/m);
              if (descMatch) {
                description = descMatch[1].trim().replace(/^[\'"]|[\'"]$/g, "");
              } else {
                const headingMatch = content.match(/^#\s+(.+)$/m);
                if (headingMatch) description = headingMatch[1].trim();
              }
            } catch {
              // skip unreadable
            }
          }
          if (!skillsMap.has(entry.name)) {
            skillsMap.set(entry.name, { description, source });
          }
        }
      };

      scanSkillsDir(joinPath(homedir(), ".claude", "skills"), "global");
      scanSkillsDir(joinPath(cwd, ".agents", "skills"), "project");
      let dir = cwd;
      for (let i = 0; i < 10; i++) {
        const parent = dirname(dir);
        if (parent === dir) break;
        scanSkillsDir(joinPath(parent, ".agents", "skills"), "workspace");
        dir = parent;
      }

      let skillsList: string;
      if (skillsMap.size === 0) {
        skillsList = "No skills configured.\n\nSkills are loaded from:\n- ~/.claude/skills/\n- <project>/.agents/skills/\n- <workspace>/.agents/skills/";
      } else {
        const lines: string[] = ["Available skills:\n"];
        for (const [sname, info] of skillsMap) {
          const desc = info.description ? `: ${info.description}` : "";
          lines.push(`- /${sname}${desc}`);
        }
        skillsList = lines.join("\n");
      }

      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: skillsList },
        } as any,
      });
      return { handled: true };
    }

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
        return { handled: true };
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
        return { handled: true };
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
        return { handled: true };
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
        return { handled: true };
      }

      if (argument.length > 4000) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: 'Command "/goal" requires goal text of at most 4000 characters.' },
          } as any,
        });
        return { handled: true };
      }
      sessionGoals.set(sessionId, { text: argument, paused: false });
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `Goal set: ${argument.slice(0, 200)}` },
        } as any,
      });
      // Caller should start an LLM turn pursuing the goal.
      return { handled: true, startTurn: `Please start working toward this goal now:\n\n${argument}` };
    }

    // Unknown slash command — let the caller send it as a plain prompt
    // (Claude CLI may handle it natively).
    return { handled: false };
  }

  /**
   * Resolve the effort to pass to claude CLI for a session: the stored override
   * if the current model supports it, otherwise the model's default effort.
   * Centralized so /compact, normal turns, steer drains, and auto-compact all
   * agree on the effort for a given model+session (previously /compact via
   * prompt() used getDefaultEffort=high while steer/auto /compact passed
   * undefined for reasoning models).
   */
  function resolveEffort(store: SessionStore, sessionId: string): string | undefined {
    const effortOverride = store.getConfigOverrides(sessionId)["effort"] as string | undefined;
    const effectiveModel = resolveSessionModel(store, sessionId);
    const modelEfforts = getEffortsForModel(effectiveModel);
    if (effortOverride && modelEfforts.includes(effortOverride)) return effortOverride;
    return getDefaultEffort(effectiveModel);
  }

  /**
   * Emit the ACP compaction lifecycle that clients use to render a DEDICATED,
   * transient compaction notice — mirroring codex-acp's `contextCompaction`
   * tool-call pair instead of announcing compaction as assistant text.
   *
   * Why not `agent_message_chunk`: vscode-acp only treats compaction text as a
   * standalone notice when nothing is mid-stream (`currentAssistantText.trim()
   * .length === 0 && text.length < 200`). Auto-compact runs right after a turn
   * in which the assistant just streamed text, so that guard fails and the
   * notice is appended to the previous assistant bubble — concatenated with
   * older output, and never dismissed because it lands in normal history.
   *
   * With `_meta.contextCompaction` the client instead:
   *  - shows "Compacting conversation..." progress while `in_progress`;
   *  - on `completed`, clears the compacted-away history and renders the notice
   *    as a DOM-only `compact-notice` system message that is transient (not
   *    persisted to history, cleared when the next user message arrives);
   *  - on `failed`, hides progress and shows a "Compaction failed" error.
   *
   * The client supplies the notice text itself, so no assistant text is sent.
   */
  async function emitCompactionLifecycle(
    connection: AgentSideConnection,
    sessionId: string,
    toolCallId: string,
    phase: "start" | "completed" | "failed",
    detail?: string,
  ): Promise<void> {
    const update =
      phase === "start"
        ? {
            sessionUpdate: "tool_call",
            toolCallId,
            kind: "other",
            title: "Context compacting",
            status: "in_progress",
            _meta: { contextCompaction: true },
          }
        : {
            sessionUpdate: "tool_call_update",
            toolCallId,
            // On failure the client renders its own error text; the detail is
            // kept in the title so it still shows in the ACP traffic log.
            title:
              phase === "completed"
                ? "Context compacted"
                : `Compaction failed${detail ? `: ${detail}` : ""}`,
            status: phase === "completed" ? "completed" : "failed",
            _meta: { contextCompaction: true },
          };
    await connection
      .sessionUpdate({ sessionId, update: update as any })
      .catch(() => {
        // Best effort — a missing notice must not fail the compaction itself.
      });
  }

  async function doCompact(
    sessionId: string,
    connection: AgentSideConnection,
    runner: ClaudeRunner,
    store: SessionStore,
    modelOverride: string | undefined,
    effectiveEffort: string | undefined,
    announce: boolean,
  ): Promise<boolean> {
    const claudeSid = store.getClaudeSessionId(sessionId);
    if (!claudeSid) {
      if (announce) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "No conversation to compact yet." },
          } as any,
        });
      }
      return false;
    }
    // Announce compaction through the dedicated lifecycle instead of assistant
    // text, so the client can render a standalone, self-dismissing notice.
    // Emitted for BOTH manual /compact and auto-compact — the client tells them
    // apart by whether it already set `isCompacting` from the user's command.
    const compactionId = `compaction_${++compactionCounter}`;
    await emitCompactionLifecycle(connection, sessionId, compactionId, "start");
    // Ask the LLM to produce a concise summary of the conversation.
    const compactPrompt =
      "Please provide a concise summary of our conversation so far, including key decisions, context, and any pending tasks. This summary will be used to continue the conversation with reduced context.";
    try {
      const result = await runner.continueSessionStreaming(
        claudeSid,
        compactPrompt,
        () => {}, // no streaming events for compact
        sessionId,
        modelOverride,
        effectiveEffort,
      );
      // Now clear the old session and start fresh with the summary.
      const cwd = store.getCwd(sessionId)!;
      store.clearPersistedSession(cwd);
      store.setClaudeSessionId(sessionId, "");
      const newResult = await runner.startSessionStreaming(
        cwd,
        `Previous conversation summary:\n\n${result.text}\n\nPlease continue from this context.`,
        () => {},
        sessionId,
        modelOverride,
        effectiveEffort,
      );
      store.setClaudeSessionId(sessionId, newResult.sessionId);
      // Clear usage tracking so auto-compact doesn't immediately re-trigger.
      sessionLastUsage.delete(sessionId);
      // Compaction replaces the whole conversation with a summary, so any
      // todo list from the prior context no longer applies.
      sessionTodos.delete(sessionId);
      // Reset the context estimator: compaction replaced the entire
      // conversation with a short summary, so the pre-compaction content the
      // estimator accumulated no longer reflects the session's real context.
      // Re-seed with the summary prompt that started the fresh session.
      const estimator = sessionEstimators.get(sessionId);
      if (estimator) {
        estimator.reset();
        estimator.addText(
          `Previous conversation summary:\n\n${result.text}\n\nPlease continue from this context.`,
        );
      }
      // Completion drives the client's history clear + transient notice. The
      // client renders the notice text itself, so nothing is sent as assistant
      // text (which is what previously got concatenated onto the prior reply).
      await emitCompactionLifecycle(connection, sessionId, compactionId, "completed");
      logger.info(`Compaction succeeded for session ${sessionId}`);
      return true;
    } catch (compactErr) {
      const detail = compactErr instanceof Error ? compactErr.message : String(compactErr);
      logger.error(`Compaction failed for session ${sessionId}: ${detail}`);
      // `failed` makes the client hide the progress indicator and surface an
      // error instead of leaving a stuck "Compacting conversation..." banner.
      await emitCompactionLifecycle(connection, sessionId, compactionId, "failed", detail);
      return false;
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
          loadSession: true,
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

     // The agent is respawned per client session (and on window reload), so a
     // session the client is resuming may not be in this process's map. The
     // store rehydrates from disk at construction; if it still isn't there
     // (nothing persisted yet, or persisted under a different key), recreate it
     // from the cwd the client supplied. Throwing resourceNotFound here is what
     // made a resumed session render completely empty.
     if (!store.has(sessionId)) {
       if (typeof params.cwd === "string" && params.cwd) {
         logger.info(
           `Load session ${sessionId} unknown to this process — rehydrating from cwd ${params.cwd}`
         );
         store.ensure(sessionId, params.cwd, (params as any).mcpServers ?? []);
       } else {
         throw RequestError.resourceNotFound(
           `Session ${sessionId} not found`
         );
       }
     }

     const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
     logger.info(`Load session: ${sessionId}`);

      // Replay conversation history from Claude CLI's JSONL transcript.
      // The transcript lives at ~/.claude/projects/<encoded-cwd>/<claude-session-id>.jsonl
      let claudeSessionId = store.getClaudeSessionId(sessionId);
      const { homedir } = await import("node:os");
      const replayCwd = store.getCwd(sessionId)!;
      const projectDir = `${homedir()}/.claude/projects/${replayCwd.replace(/\//g, "-")}`;
      if (!claudeSessionId) {
        // Sessions created before full-record persistence have no mapping. Fall
        // back to the most recent transcript for this directory so the resumed
        // conversation still renders instead of coming up blank.
        try {
          const { readdirSync, statSync, existsSync } = await import("node:fs");
          if (existsSync(projectDir)) {
            const newest = readdirSync(projectDir)
              .filter((f) => f.endsWith(".jsonl"))
              .map((f) => ({ f, m: statSync(`${projectDir}/${f}`).mtimeMs }))
              .sort((a, b) => b.m - a.m)[0];
            if (newest) {
              claudeSessionId = newest.f.replace(/\.jsonl$/, "");
              logger.info(
                `No Claude session mapped for ${sessionId}; replaying newest transcript ${claudeSessionId}`
              );
            }
          }
        } catch {
          // Fall through — nothing to replay.
        }
      }
      if (claudeSessionId) {
        try {
          const cwd = replayCwd;
          const transcriptPath = `${projectDir}/${claudeSessionId}.jsonl`;
          const { readFileSync, existsSync } = await import("node:fs");
          if (existsSync(transcriptPath)) {
            const raw = readFileSync(transcriptPath, "utf-8");
            const lines = raw.split("\n").filter((l) => l.trim());
            logger.info(`Replaying ${lines.length} transcript lines from ${transcriptPath}`);
            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
                // Subagent (sidechain) entries are a separate conversation —
                // replaying them would interleave subagent chatter into the
                // main history.
                if (entry.isSidechain) continue;
                if (entry.type === "user") {
                  const msg = entry.message;
                  if (!msg) continue;
                  const content = msg.content;
                  if (typeof content === "string" && content.trim()) {
                    await connection.sessionUpdate({
                      sessionId,
                      update: {
                        sessionUpdate: "user_message_chunk",
                        content: { type: "text", text: content },
                      } as any,
                    });
                  } else if (Array.isArray(content)) {
                    for (const block of content) {
                      if (block.type === "text" && block.text) {
                        await connection.sessionUpdate({
                          sessionId,
                          update: {
                            sessionUpdate: "user_message_chunk",
                            content: { type: "text", text: block.text },
                          } as any,
                        });
                      }
                    }
                  }
                } else if (entry.type === "assistant") {
                  const msg = entry.message;
                  if (!msg) continue;
                  const content = msg.content;
                  if (Array.isArray(content)) {
                    for (const block of content) {
                      if (block.type === "text" && block.text) {
                        await connection.sessionUpdate({
                          sessionId,
                          update: {
                            sessionUpdate: "agent_message_chunk",
                            content: { type: "text", text: block.text },
                          } as any,
                        });
                      } else if (block.type === "tool_use") {
                        // Build the update with the SAME mapper the live path
                        // uses, so a resumed session shows "Read file 'x.ts'"
                        // and the right `kind`/diff content instead of a bare
                        // tool name like "Read".
                        await connection.sessionUpdate({
                          sessionId,
                          update: buildToolCallUpdate(
                            block.id ?? `replay-${Math.random().toString(36).slice(2, 8)}`,
                            block.name ?? "tool",
                            block.input ?? {},
                          ) as any,
                        });
                      }
                    }
                  }
                }
              } catch {
                // Skip unparseable lines
              }
            }
            logger.info(`Replay complete for session ${sessionId}`);
          } else {
            logger.warn(`Transcript not found: ${transcriptPath}`);
          }
        } catch (e) {
          logger.error(`Failed to replay transcript: ${e instanceof Error ? e.message : String(e)}`);
        }
      }

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

      // The mode picker (configId "mode") must write to the canonical
      // session.mode field — every read site (buildConfigOptions, /status,
      // loadSession) consults store.getMode(), not configOverrides["mode"].
      // Routing it through setMode() keeps a single source of truth and makes
      // the selection persist across agent restarts (setMode persists to disk;
      // setConfigOverride would write configOverrides["mode"], which nothing
      // reads back, so the picker would reset to the default after restart).
      if (configId === "mode") {
        store.setMode(sessionId, String(value));
        logger.info(`Session ${sessionId} config mode set to ${value}`);
      } else if (configId === COLLABORATION_MODE_CONFIG_ID) {
        // Handle collaboration_mode specially (stored in separate map)
        sessionCollaborationModes.set(sessionId, value);
        logger.info(`Session ${sessionId} collaboration_mode set to ${value}`);
      } else {
        store.setConfigOverride(sessionId, configId, value);
        logger.info(`Session ${sessionId} config ${configId} set to ${value}`);
      }

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

      // Serialize against any in-flight prompt on this session. See
      // sessionPromptLocks above: the SDK runs concurrent prompt() handlers,
      // so without this a mid-turn /skills (or any second prompt) would race
      // with the active turn. We chain on the lock and run the whole prompt
      // body — slash dispatch, LLM turn, steer drain, auto-compact — once the
      // prior prompt finishes. The lock is keyed by session and always
      // replaced, so it never holds a stale promise across /clear (which
      // resolves quickly and clears per-session state under the lock too).
      const runPrompt = async (): Promise<PromptResponse> => {
      // Mark this session as having a prompt in flight for the entire runPrompt
      // body (LLM turn + steer drain + auto-compact + return). Cleared when the
      // lock chain settles (below). See sessionPromptInFlight.
      sessionPromptInFlight.add(sessionId);
      // If this prompt was cancelled while waiting in the serialization queue
      // behind a prior turn, don't start a new turn — drain-and-return.
      if (cancelledSessions.has(sessionId)) {
        cancelledSessions.delete(sessionId);
        logger.info(`Queued prompt cancelled before start for session ${sessionId}`);
        return { stopReason: "cancelled" };
      }
      // ── Slash command handling ──────────────────────────────────
      // Intercept built-in commands (/ps, /goal, /skills, /status, /clear,
      // /compact) before starting an LLM turn. Shared dispatch logic lives in
      // dispatchSlashCommand() so prompt() and the steer-drain loop behave
      // identically. Read-only commands and /clear|/compact|/goal clear|pause
      // |resume are fully serviced and return immediately. /goal <objective>
      // returns a goalPrompt to start an LLM turn with.
      const dispatchResult = await dispatchSlashCommand(sessionId, connection, runner, store, text);
      // When /goal sets a new objective, this holds the prompt to send to the
      // LLM to start working on it immediately.
      const goalPrompt = dispatchResult.startTurn ?? null;
      // A fully-serviced command (read-only, /clear, /compact, /goal subcmds)
      // needs no main LLM turn — but we still drain any steers queued during it
      // (e.g. steers queued while /compact was summarizing) and run the
      // auto-compact check, so we skip the main turn rather than returning early.
      const skipMainTurn = dispatchResult.handled && !goalPrompt;
      const command = dispatchResult.handled ? parseSlashCommand(text) : null;

      // If there's an active (non-paused) goal, prepend it to the prompt.
     // goalPrompt is set when /goal just set a new objective — use it directly.
     // For subsequent prompts, prepend the goal as context.
     const activeGoal = sessionGoals.get(sessionId);
     let effectivePrompt = text;
     if (goalPrompt) {
       effectivePrompt = goalPrompt;
     } else if (activeGoal && !activeGoal.paused && !command) {
       effectivePrompt = `[Goal: ${activeGoal.text}]\n\n${text}`;
     }

      // Inject collaboration mode (plan mode) instructions
      const collabMode = sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE;
      if (collabMode === PLAN_COLLABORATION_MODE) {
        effectivePrompt = `<collaboration_mode># Collaboration Mode: Plan\n\nYou are now in Plan mode. Any previous instructions for other modes (e.g. Default mode) are no longer active.\n\nYour active mode changes only when new developer instructions with a different \`<collaboration_mode>...</collaboration_mode>\` change it; user requests or tool descriptions do not change mode by themselves.\n\nIn Plan mode, you MUST NOT make any changes to the codebase. Instead, analyze the request, investigate the codebase, and produce a detailed plan. Do not execute commands that modify files. Read-only investigation (searching, reading files, running read-only commands) is allowed.\n</collaboration_mode>\n\n${effectivePrompt}`;
      }

      const claudeSessionId = store.getClaudeSessionId(sessionId);
      const cwd = store.getCwd(sessionId)!;
     let toolCallCounter = 0;

      // Real-time context estimator — tracks conversation content to estimate
      // context size during the turn (before the result event arrives).
      // REUSE the session-persistent estimator across turns: a fresh estimator
      // each turn would reset to ~overhead-only and the usage ring would DROP
      // at the start of every new turn (then climb again as content streams),
      // making the context usage visibly flip down→up non-monotonically. By
      // persisting accumulated content across turns, the estimate stays
      // monotonic-growing (as a real context window does). Only /clear,
      // /compact, and auto-compact reset it (see dispatchSlashCommand /
      // doCompact).
      const existing = sessionEstimators.get(sessionId);
      const ctxEstimator = existing ?? new TurnContextEstimator();
      sessionEstimators.set(sessionId, ctxEstimator);
      // Accumulate the user's prompt text (it's part of the conversation).
      ctxEstimator.addText(effectivePrompt);

      // Get model and effort overrides from session config (set via pickers)
      const configOverrides = store.getConfigOverrides(sessionId);
      const modelOverride = configOverrides["model"] as string | undefined;
      // Only pass effort if the selected model actually supports it; otherwise
      // fall back to the model's default effort. Using resolveEffort() keeps
      // the actually-sent effort consistent with what the picker shows and
      // what /compact, steer drains, and auto-compact use (previously normal
      // turns fell back to undefined for reasoning models, while /compact via
      // prompt() used the default "high").
      const effectiveModel = resolveSessionModel(store, sessionId, modelOverride);
      const effectiveEffort = resolveEffort(store, sessionId);

      logger.info(`Prompt for session ${sessionId}: ${text.length} chars`);
      logger.debug(
        `Prompt content: ${text.slice(0, 100)}${text.length > 100 ? "..." : ""}`
      );

      const permissionPromises: Promise<void>[] = [];

      let lastUsage: { input_tokens?: number; output_tokens?: number } | null = null;
      // Output tokens of the most recent non-sidechain API call. The result
      // event's output_tokens is summed over every call in the turn, which
      // over-states how much the conversation actually grew.
      let lastRealOutputTokens = 0;
      // Track whether Claude CLI's final result was an error (e.g.
      // error_max_turns) so we can surface it to the user instead of
      // returning a silent end_turn. Mirrors codex-acp, which maps upstream
      // turn failures to a visible error rather than an empty success.
      let turnErrored = false;
      let turnStopReason: string | undefined;
      // Claude CLI emits per-token text deltas only when launched with
      // --include-partial-messages (claude-runner passes it), and wraps them in
      // a `stream_event` envelope that claude-runner unwraps. It ALSO emits the
      // completed `assistant` message block re-stating the FULL text. Emitting
      // both would duplicate the reply, so the final block is dropped for the
      // UI when deltas already delivered it (context estimation still counts
      // the final block — see below). Before the flag was passed there were no
      // deltas at all, so the whole reply arrived as one chunk and the client
      // painted a bubble it immediately superseded: the empty bubble that
      // flashed and vanished at every text→tool boundary.
      let streamedTextThisMessage = false;
      const onEvent = (event: StreamEvent) => {
        // Handle system/init — extract native Claude CLI slash commands and
        // surface them to vscode-acp via available_commands_update.
        if (event.type === "system_init" && event.slashCommands && event.slashCommands.length > 0) {
          const nativeCommands = event.slashCommands
            .filter((name) => !BUILTIN_COMMANDS.some((b) => b.name === name))
            .map((name) => ({
              name,
              description: `Claude Code: /${name}`,
              input: null,
            }));
          const allCommands = [...BUILTIN_COMMANDS, ...nativeCommands];
          connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "available_commands_update",
              availableCommands: allCommands,
            } as any,
         }).catch(() => {
           // Best effort
         });
       }
      // Handle assistant event — marks an API call boundary. Each assistant
      // message in Claude CLI's stream-json output = one LLM API call.
      // Send a real-time usage_update so the context ring shows during the turn.
      if (event.type === "assistant") {
        ctxEstimator.markApiCall();
        // Secondary source. Through the CCR/proxy path the completed assistant
        // message reports input_tokens: 0, so this usually contributes nothing;
        // setRealTokens ignores non-positive values. `message_delta` below is
        // the reliable source.
        if (event.usage && !event.isSidechain) {
          const real =
            (event.usage.input_tokens ?? 0) +
            (event.usage.cache_read_input_tokens ?? 0) +
            (event.usage.cache_creation_input_tokens ?? 0);
          ctxEstimator.setRealTokens(real);
          lastRealOutputTokens = event.usage.output_tokens ?? 0;
        }
        sendEstimatedUsage(ctxEstimator, connection, sessionId, effectiveModel);
        // NOTE: do NOT reset streamedTextThisMessage here. The `assistant`        // event fires at the START of the final assistant block (after the
        // per-token content_block_delta stream for the same message already
        // arrived). Resetting here would wipe the flag right before the
        // final-block text_delta below checks it. The flag is consumed and
        // cleared by the final-block text_delta itself.
      }
      // Handle message_delta — the authoritative per-call context size.
      //
      // This is the figure Claude Code and the Anthropic SDK use for their own
      // context accounting: the exact prompt token count the provider charged
      // for the API call that just completed. Adopting it verbatim is why the
      // usage ring no longer needs an estimate. Subagent calls (non-null
      // parent_tool_use_id on the stream envelope) measure a different
      // conversation and must not be attributed here.
      if (event.type === "message_delta" && event.usage && !event.isSidechain) {
        const realPrompt =
          (event.usage.input_tokens ?? 0) +
          (event.usage.cache_read_input_tokens ?? 0) +
          (event.usage.cache_creation_input_tokens ?? 0);
        if (realPrompt > 0) {
          ctxEstimator.setRealTokens(realPrompt);
          lastRealOutputTokens = event.usage.output_tokens ?? lastRealOutputTokens;
          sendEstimatedUsage(ctxEstimator, connection, sessionId, effectiveModel);
        }
      }
      // Handle user event (tool results) — accumulate for context estimation.
      if (event.type === "user" && event.text) {
        ctxEstimator.addText(event.text);
      }
      // Claude Code's Task/Agent tool runs subagents ASYNCHRONOUSLY: the
      // tool_result it emits ("Async agent launched successfully...") is a
      // launch acknowledgment, NOT a completion signal. The subagent keeps
      // running (and emitting its own nested tool calls) for seconds to
      // minutes after this tool_result lands. Flipping the parent tool_call
      // to "completed" here would mark it done at spawn time — leaving the UI
      // spinner stuck on a card whose nested work is still streaming.
      //
      // Instead we leave subagent tool_calls in_progress and flush them all to
      // "completed" at turn end (see the post-stream flush below). The parent
      // turn cannot end while a subagent it spawned is still running — Claude
      // awaits them — so turn end is a reliable completion signal.
      // Non-subagent tools: flip any in_progress tool whose Claude tool_use_id
      // is resolved by this user message's tool_results to "completed". The
      // runner emits toolResultIds on the user event (claude-runner.ts). This
      // is what surfaces the running → completed transition for Bash/Read/Edit.
      if (event.type === "user" && event.toolResultIds?.length) {
        const pendingTools = sessionPendingTools.get(sessionId);
        if (pendingTools && pendingTools.size > 0) {
          for (const tuId of event.toolResultIds) {
            const acpCallId = pendingTools.get(tuId);
            if (acpCallId) {
              pendingTools.delete(tuId);
              connection
                .sessionUpdate({
                  sessionId,
                  update: {
                    sessionUpdate: "tool_call_update",
                    toolCallId: acpCallId,
                    status: "completed",
                  } as any,
                })
                .catch(() => {});
            }
          }
          if (pendingTools.size === 0) {
            sessionPendingTools.delete(sessionId);
          }
        }
      }
    // Handle result event — Claude CLI reports CUMULATIVE usage across all
    // API calls in the turn (accumulateUsage in QueryEngine.ts). For a
    // single-call turn, input_tokens = real context size (accurate). For a
     // multi-call turn (tool calls), input_tokens = sum of all calls' input
     // tokens, which inflates the usage UI (e.g. 3 calls × 20k = 60k shown,
     // but real context is ~20k).
     //
     // Per-call usage comes from each `assistant` message's `message.usage`
     // (see the assistant handler above), which is exact. The result event's
     // usage is only a fallback for the pre-first-call window.
     //
    // Strategy:
    // - For the UI: use our real-time estimate (ctxEstimator), which is more
    //   accurate than the cumulative result value. Cap at contextWindow.
     // - For auto-compact: use ONLY our estimate. The cumulative result
     //   value (sum of all API calls) is inflated on multi-call turns and
     //   triggers compaction way too early (e.g. 340k for a 27k context).
     //   Our estimate tracks actual conversation growth and is accurate.
     if (event.type === "result") {
      // Capture the stop reason / error flag for the prompt return path.
      // This runs even when the result carries no usage (some error results
      // omit it), so an error_max_turns / failed turn is never silently
      // swallowed as a plain end_turn.
      turnStopReason = event.stopReason;
      if (event.isError) turnErrored = true;
      // Record the context window Claude Code resolved for this model, so the
      // usage denominator can fall back to the CLI's own figure instead of a
      // hardcoded 200k when the proxy advertises nothing.
      if (event.contextWindow && event.contextWindow > 0) {
        const cwModel = resolveSessionModel(store, sessionId, modelOverride);
        if (cliReportedContextWindows.get(cwModel) !== event.contextWindow) {
          cliReportedContextWindows.set(cwModel, event.contextWindow);
          logger.info(
            `Claude Code reports contextWindow=${event.contextWindow} for ${cwModel}`
          );
        }
      }
      // Log the stop reason so premature turn termination (end_turn vs
      // max_turns vs error) is diagnosable from the ACP stderr log.
      if (event.stopReason || event.isError) {
        logger.info(
          `Turn result: stopReason=${event.stopReason ?? "unknown"} isError=${!!event.isError}`,
        );
      }
      if (event.usage) {
      lastUsage = event.usage;
      const inputTokens = event.usage.input_tokens ?? 0;
      const outputTokens = event.usage.output_tokens ?? 0;
      const effectiveModel = resolveSessionModel(store, sessionId, modelOverride);
      const contextWindow = getContextWindowForModel(effectiveModel);
      // Context size for both the UI and auto-compact. Once a real per-call
      // prompt count has been seen (message.usage on an assistant event) the
      // estimator returns that; the result event's own usage is accumulated
      // across every API call in the turn and would over-report badly on
      // multi-tool turns, so it is never used for sizing.
      const estimatedTokens = ctxEstimator.getEstimatedTokens();
      // The next request's prompt = this prompt + this turn's final output, so
      // add the LAST call's output rather than the turn-cumulative figure.
      const trailingOutput = ctxEstimator.hasRealTokens()
        ? lastRealOutputTokens
        : outputTokens;
      sessionLastUsage.set(sessionId, { input_tokens: estimatedTokens, output_tokens: trailingOutput });
      // For the UI: use our real-time estimate (more accurate than cumulative).
      // Cap at contextWindow so it never shows >100%.
      const cappedUsed = Math.min(estimatedTokens, contextWindow);
       connection.sessionUpdate({
         sessionId,
         update: {
           sessionUpdate: "usage_update",
           used: cappedUsed,
           size: contextWindow,
         } as any,
       }).catch(() => {
         // Best effort — don't fail the turn if usage_update fails.
       });
      }
     }
       if ((event as any).type === "thinking" && (event as any).text) {
         ctxEstimator.addText((event as any).text);
         connection.sessionUpdate({
           sessionId,
           update: {
             sessionUpdate: "agent_thought_chunk",
             content: { type: "text", text: (event as any).text },
           } as any,
         });
       } else if (event.type === "text_delta" && event.text) {
         // In --verbose mode Claude CLI emits BOTH streaming content_block_delta
         // text AND the final assistant message block's full text. To avoid
         // double-counting assistant content in the context estimate, we count
         // only the final-block text (fromAssistantBlock) and ignore the
         // streaming deltas.
         if (event.fromAssistantBlock) {
           ctxEstimator.addText(event.text);
         } else {
           // Mark that this message streamed its text via per-token deltas.
           streamedTextThisMessage = true;
         }
         // Emit the chunk to the UI — EXCEPT for the final-block text when the
         // same message was already streamed token-by-token. The final block's
         // full text would land ~1ms before the tool_use in the same content
         // array (claude-runner.ts emits them back-to-back), beating vscode-acp's
         // 80ms streaming render and leaving an empty bubble at every
         // text→tool transition. The per-token deltas already delivered this
         // text; re-emitting it also duplicates it in currentAssistantText.
         // When there were NO streaming deltas (text only in the final block,
         // e.g. non-verbose or a non-streaming model), emit it as the sole
         // source of the text. The final-block text consumes and clears the
         // flag so it applies per message.
         const dropAsDuplicate = event.fromAssistantBlock && streamedTextThisMessage;
         if (event.fromAssistantBlock) streamedTextThisMessage = false;
         if (!dropAsDuplicate) {
           connection.sessionUpdate({
             sessionId,
             update: {
               sessionUpdate: "agent_message_chunk",
               content: { type: "text", text: event.text },
             },
           });
         }
       } else if (event.type === "tool_use" && event.toolName) {
         ctxEstimator.addToolCall(event.toolName, event.toolInput);
         const toolCallId = `call_${++toolCallCounter}`;
         logger.debug(`Tool call: ${event.toolName}`);
         // TaskCreate/TaskUpdate/TaskList maintain a todo list — mirror it
         // into an ACP `plan` update so the chat UI shows a plan bar with
         // the current task count (like codex-acp's update_plan). The plan
         // is a full snapshot, so the count reflects the live list size
         // rather than a running total of updates.
         handleTaskTool(sessionId, event.toolName, event.toolInput);
         // Build a descriptive title + kind + (for edits) diff content +
         // locations, mirroring codex-acp, so the chat UI shows the action
         // verb and a click-to-open diff instead of a bare filename.
         const isSubagent =
           event.toolName === "Task" || event.toolName === "Agent";
         const update = buildToolCallUpdate(
           toolCallId,
           event.toolName,
           event.toolInput,
         );
         // Emit every tool as in_progress at its tool_use so the chat UI shows
         // a running phase (command text + spinner) while the tool executes.
         // The tool flips to "completed" when its tool_result lands — subagents
         // via the <task-notification> path below, normal tools via the
         // tool_result path in the user-event handler. buildToolCallUpdate's
         // status:"completed" default is overridden here and only resolved on
         // completion; the post-turn flush is the safety net for tools whose
         // result never arrived before end_turn.
         const emitted = { ...update, status: "in_progress" as const };
         if (isSubagent && event.toolUseId) {
           let pending = sessionPendingAgents.get(sessionId);
           if (!pending) {
             pending = new Map();
             sessionPendingAgents.set(sessionId, pending);
           }
           pending.set(event.toolUseId, toolCallId);
         } else if (event.toolUseId) {
           // Non-subagent tool: register so the tool_result user-event can flip
           // it to completed. Subagents are excluded (their completion comes
           // through <task-notification>, not the tool_result id match).
           let pendingTools = sessionPendingTools.get(sessionId);
           if (!pendingTools) {
             pendingTools = new Map();
             sessionPendingTools.set(sessionId, pendingTools);
           }
           pendingTools.set(event.toolUseId, toolCallId);
         }
         connection.sessionUpdate({
           sessionId,
           update: emitted,
         });
       } else if (event.type === "permission_request" && event.toolName) {
         ctxEstimator.addToolCall(event.toolName, event.toolInput);
         const toolCallId = `call_${++toolCallCounter}`;
         const toolName = event.toolName;
         const toolInput = event.toolInput ?? {};
         logger.info(`Permission request: ${toolName}`);

          // Build the descriptive title + kind + diff content once, then reuse
          // across the pending → completed/failed lifecycle.
          const baseUpdate = buildToolCallUpdate(toolCallId, toolName, toolInput);

          // Send pending tool_call status
          connection.sessionUpdate({
            sessionId,
            update: { ...baseUpdate, status: "pending" },
          });

          // Request permission from client (track promise for awaiting)
          const permPromise = connection
            .requestPermission({
              sessionId,
              toolCall: {
                toolCallId,
                title: baseUpdate.title,
                kind: baseUpdate.kind as any,
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
                  ...baseUpdate,
                  sessionUpdate: "tool_call_update",
                  status: isApproved ? "completed" : "failed",
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
                  ...baseUpdate,
                  sessionUpdate: "tool_call_update",
                  status: "failed",
                },
              });
            });

          permissionPromises.push(permPromise);
        }
      };

  /**
   * Deliver every queued steer message as a follow-up `--resume` prompt.
   *
   * Claude CLI cannot inject mid-turn, so steers are queued (handleSteer) and
   * delivered here. Called at THREE points so delivery order always matches
   * the order the user typed in:
   *  1. BEFORE the main turn — a steer that landed in the drain tail of the
   *     previous turn (after that drain took the queue but before the prompt
   *     lock cleared) must go out before the newer prompt, not after it.
   *  2. AFTER the main turn — the normal case.
   *  3. AFTER compaction — /compact and auto-compact each run extra LLM calls,
   *     and anything steered during them would otherwise sit in the queue
   *     until the next user prompt.
   *
   * The queue is drained IN PLACE (shift from the live array) instead of being
   * copied out and deleted. That way a steer arriving mid-drain is picked up by
   * the same pass, and an early exit leaves the undelivered remainder queued
   * rather than silently dropping it.
   */
  const drainSteerQueue = async (): Promise<void> => {
    const queue = steerQueues.get(sessionId);
    if (!queue || queue.length === 0) return;
    // Re-entrancy guard: draining awaits LLM turns, so a nested call (e.g. the
    // post-compaction drain starting while the post-turn drain is still
    // awaiting) would interleave two consumers over one queue.
    if (steerDraining.has(sessionId)) return;
    steerDraining.add(sessionId);
    let deliveredAny = false;
    try {
      while (queue.length > 0) {
        const steerText = queue.shift()!;
        // Cancellation discards the rest — the user asked to stop.
        if (cancelledSessions.has(sessionId)) {
          queue.length = 0;
          break;
        }
        const claudeSid = store.getClaudeSessionId(sessionId);
        // No Claude session to resume into (e.g. straight after /clear). Put it
        // back so a later drain delivers it once a session exists again.
        if (!claudeSid) {
          queue.unshift(steerText);
          break;
        }
        deliveredAny = true;
       toolCallCounter = 0;
       permissionPromises.length = 0;
       logger.info(`Draining steer message for session ${sessionId}: ${steerText.length} chars`);

          // If the steered text is a slash command, dispatch it through the
          // SAME handler prompt() uses (dispatchSlashCommand) so the two entry
          // points behave identically. Previously this was a hand-rewritten
          // copy that silently dropped several cases (/goal with empty/over-
          // length arg, /goal pause|resume with no goal, and /ps|/status|
          // /skills fell through to raw text).
          const steerDispatch = await dispatchSlashCommand(sessionId, connection, runner, store, steerText);
          if (steerDispatch.handled) {
            logger.info(`Steer drain: handled slash command for session ${sessionId}`);
            if (steerDispatch.startTurn) {
              // /goal <objective> — start an LLM turn pursuing it.
              try {
                ctxEstimator.addText(steerDispatch.startTurn);
                await runner.continueSessionStreaming(claudeSid, steerDispatch.startTurn, onEvent, sessionId, modelOverride, effectiveEffort);
                await Promise.all(permissionPromises);
              } catch (goalErr) {
                logger.error(`Steer drain goal turn failed: ${goalErr instanceof Error ? goalErr.message : String(goalErr)}`);
              }
            }
            // /clear wipes the Claude session ID, so there is nothing to resume
            // into — stop this pass. Anything still queued stays queued (the
            // drain is non-destructive), and the `!claudeSid` guard puts it back
            // until a later drain runs against the fresh session.
            if (parseSlashCommand(steerText)?.name === "clear") break;
            continue;
          }

        try {
          // Steer-drain guard: a steered message is a FOLLOW-UP user
          // instruction delivered after the previous turn completed — Claude
          // CLI can't inject mid-turn, so we resume the session here. To stop
          // the model from misreading the prior turn's (possibly subagent-
          // produced) output as an interruption it must diagnose, frame the
          // steer as a fresh instruction. Also, if a subagent is somehow still
          // marked in_progress (result fired before its tool_result), defer
          // the drain rather than yanking the session into a new context.
          const stillPending = sessionPendingAgents.get(sessionId);
          if (stillPending && stillPending.size > 0) {
            logger.info(
              `Steer drain deferred for session ${sessionId}: ${stillPending.size} subagent(s) still in_progress`,
            );
            // Put it back at the front so it drains once the subagent resolves.
            queue.unshift(steerText);
            break;
          }

          // Prepend goal to steer messages too
          const basePrompt = activeGoal && !activeGoal.paused
            ? `[Goal: ${activeGoal.text}]\n\n${steerText}`
            : steerText;
          // Frame as a follow-up so the model treats this as a new user
          // instruction rather than a description of interrupted state.
          const steerPrompt = `[Follow-up instruction from user]\n${basePrompt}`;
          ctxEstimator.addText(steerPrompt);
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
    } finally {
      steerDraining.delete(sessionId);
    }
    if (deliveredAny) {
      store.touch(sessionId);
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "session_info_update",
          updatedAt: store.getUpdatedAt(sessionId),
        } as any,
      }).catch(() => {});
    }
  };

      try {
        // Deliver anything still queued from the PREVIOUS turn before starting
        // this one. A steer can land in the previous turn's drain tail — after
        // its drain emptied the queue but before the prompt lock cleared —
        // and handleSteer accepts it because a prompt is still in flight. Left
        // to the post-turn drain it would arrive AFTER this newer prompt, i.e.
        // out of the order the user typed.
        await drainSteerQueue();
        // Re-read the Claude session id: the drain above can dispatch slash
        // commands, and /clear wipes it. The id captured before the drain would
        // then resume a conversation the user just cleared.
        const claudeSessionIdAfterDrain = store.getClaudeSessionId(sessionId);
        let result;
        if (!skipMainTurn) {
        if (claudeSessionIdAfterDrain) {
          try {
            result = await runner.continueSessionStreaming(
              claudeSessionIdAfterDrain,
              effectivePrompt,
              onEvent,
              sessionId,
              modelOverride,
              effectiveEffort
            );
          } catch (resumeErr) {
            // Resume failed (expired session, etc.) — fall back to new session
            logger.warn(
              `Resume failed for ${claudeSessionIdAfterDrain}, starting fresh: ${resumeErr instanceof Error ? resumeErr.message : String(resumeErr)}`
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
        }

        // Wait for all pending permission requests to resolve
        await Promise.all(permissionPromises);

        // Turn end = all spawned subagents have finished (Claude awaits them
        // before ending the turn). Flush any still-in_progress subagent
        // tool_calls to "completed" now. This is the real completion signal
        // for the Task/Agent tool — its tool_result was only a launch ack.
        const pendingAgents = sessionPendingAgents.get(sessionId);
        if (pendingAgents && pendingAgents.size > 0) {
          for (const acpCallId of pendingAgents.values()) {
            connection
              .sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "tool_call_update",
                  toolCallId: acpCallId,
                  status: "completed",
                } as any,
              })
              .catch(() => {});
          }
          pendingAgents.clear();
          sessionPendingAgents.delete(sessionId);
        }

        // Post-turn normal-tool flush. Tools whose tool_result never arrived
        // before end_turn (e.g. a tool_use emitted in the final assistant
        // message with no following user/tool_result) would otherwise hang in
        // in_progress forever. Mark them completed as a fallback.
        const pendingTools = sessionPendingTools.get(sessionId);
        if (pendingTools && pendingTools.size > 0) {
          logger.info(
            `Turn ended with ${pendingTools.size} tool(s) still pending ` +
              `(no tool_result before end_turn) for session ${sessionId} ` +
              `— marking completed as fallback.`,
          );
          for (const acpCallId of pendingTools.values()) {
            connection
              .sessionUpdate({
                sessionId,
                update: {
                  sessionUpdate: "tool_call_update",
                  toolCallId: acpCallId,
                  status: "completed",
                } as any,
              })
              .catch(() => {});
          }
          pendingTools.clear();
          sessionPendingTools.delete(sessionId);
        }

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

        // Drain any steers queued during this turn (see drainSteerQueue).
        await drainSteerQueue();

        // ─── Auto-compact check ───────────────────────────────────────────
        // Uses the same threshold Claude Code CLI computes (context window,
        // output-token reserve, 13k buffer and CLAUDE_AUTOCOMPACT_PCT_OVERRIDE),
        // and honors the CLI's DISABLE_AUTO_COMPACT / DISABLE_COMPACT switches,
        // so enabling or tuning compaction in one place affects both layers.
        const lastUsageEntry = sessionLastUsage.get(sessionId);
        if (lastUsageEntry && isAutoCompactEnabled()) {
          const threshold = getAutoCompactThreshold(effectiveModel);
          const used = lastUsageEntry.input_tokens + lastUsageEntry.output_tokens;
          if (used >= threshold) {
            logger.info(
              `Auto-compact triggered for session ${sessionId}: ${used} >= ${threshold} (effectiveWindow=${getEffectiveContextWindowSize(effectiveModel)}, contextWindow=${getContextWindowForModel(effectiveModel)})`,
            );
            // The compaction notice is emitted by doCompact as a dedicated
            // `_meta.contextCompaction` lifecycle, not as assistant text. An
            // agent_message_chunk here would be appended to the reply that just
            // streamed (vscode-acp only treats compaction text as standalone
            // when nothing is mid-stream), which is exactly the concatenated,
            // never-dismissed notice this replaces.
           await doCompact(
             sessionId,
             connection,
             runner,
             store,
             modelOverride,
             effectiveEffort,
             false, // no "No conversation to compact yet." text for auto-compact
           );
           // After compaction, re-send the original user prompt so the LLM
           // continues the task in the fresh (compacted) session. This matches
           // Codex behavior where compaction is transparent to the user — the
           // task continues automatically after context is compressed.
           logger.info(`Re-sending original prompt after auto-compact for session ${sessionId}`);
           const claudeSid = store.getClaudeSessionId(sessionId);
           if (claudeSid) {
             const continueResult = await runner.continueSessionStreaming(
               claudeSid,
               effectivePrompt,
               onEvent,
               sessionId,
               modelOverride,
               effectiveEffort,
             );
             store.setClaudeSessionId(sessionId, continueResult.sessionId);
             await Promise.all(permissionPromises);
             // Update usage after continuation
             const contEstimate = ctxEstimator.getEstimatedTokens();
             const contContextWindow = getContextWindowForModel(effectiveModel);
             sessionLastUsage.set(sessionId, { input_tokens: contEstimate, output_tokens: 0 });
             await connection.sessionUpdate({
               sessionId,
               update: {
                 sessionUpdate: "usage_update",
                 used: Math.min(contEstimate, contContextWindow),
                 size: contContextWindow,
               } as any,
             }).catch(() => {});
           }
         }
        }

        // Compaction (manual /compact via dispatchSlashCommand above, or the
        // auto-compact just run) performs its own LLM calls, during which a
        // prompt is still in flight — so handleSteer keeps accepting steers.
        // Drain them now instead of leaving them queued until the next user
        // prompt. This mirrors the vscode-acp client, which holds its own
        // message queue while `isCompacting` and drains it once compaction
        // finishes. No-op when nothing was queued.
        await drainSteerQueue();

        // If Claude CLI reported an error result (e.g. error_max_turns — the
        // model hit the turn ceiling mid-task), surface it to the user as an
        // agent message instead of a silent end_turn. This is what makes a
        // genuinely premature stop distinguishable from a normal one; codex-acp
        // does the same by mapping upstream failures to a visible error.
        if (turnErrored) {
          const reason = turnStopReason ?? "error";
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: {
                type: "text",
                text:
                  reason === "error_max_turns"
                    ? `Reached the maximum number of turns (${reason}) before finishing. The task may be incomplete — continue to resume.`
                    : `Turn ended with status: ${reason}.`,
              },
            },
          }).catch(() => {});
        }
        return { stopReason: "end_turn", usage: (lastUsage as any) ? { inputTokens: (lastUsage as any).input_tokens ?? 0, outputTokens: (lastUsage as any).output_tokens ?? 0, totalTokens: ((lastUsage as any).input_tokens ?? 0) + ((lastUsage as any).output_tokens ?? 0) } : undefined };
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
      }; // end runPrompt

      // Chain on the per-session lock so this prompt waits for any in-flight
      // one. We always replace the lock with the new chain (success or
      // failure) so a rejected prior prompt never blocks the next forever.
      const prev = sessionPromptLocks.get(sessionId) ?? Promise.resolve();
      const next = prev.then(runPrompt, runPrompt);
      // Clear the in-flight flag once runPrompt settles (success or failure).
      // If a newer prompt queued behind this one has already re-added the flag
      // by the time this fires, the delete is a harmless no-op on a re-added
      // entry — but ordering (this finally registered before the newer prompt's
      // runPrompt runs) means this clears first, then the newer runPrompt
      // re-adds, so the flag stays accurate.
      next.finally(() => sessionPromptInFlight.delete(sessionId));
      sessionPromptLocks.set(sessionId, next.catch(() => {}));
      return next;
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
