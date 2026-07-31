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
  {
    name: "clear",
    description: "Clear conversation history and start fresh.",
    input: null,
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
 * Resolve the context window (in tokens) for a given model.
 * Uses the cached proxy data; falls back to DEFAULT_CONTEXT_WINDOW.
 */
function getContextWindowForModel(modelId: string | undefined): number {
  if (modelId && modelContextWindows.has(modelId)) {
    return modelContextWindows.get(modelId)!;
  }
  return DEFAULT_CONTEXT_WINDOW;
}

/**
 * Auto-compact threshold = context_window * 9 / 10 (90%), mirroring Codex.
 */
function getAutoCompactThreshold(modelId: string | undefined): number {
  return Math.floor((getContextWindowForModel(modelId) * 9) / 10);
}

// ─── Auto-compact ───────────────────────────────────────────────────────────
// Mirrors Codex core's auto-compact: trigger compaction when token usage
// reaches 90% of the context window (context_window * 9 / 10).
// Codex checks after each API response mid-turn; Claude CLI manages turns
// internally, so we check between turns — after the prompt finishes but
// before the next user prompt is processed. The compaction replaces the
// old Claude session with a fresh one seeded with a summary.
// Track the last reported usage per session so the auto-compact check
// can run after the turn finishes.
const sessionLastUsage = new Map<string, { input_tokens: number; output_tokens: number }>();

// Per-session context estimator for the in-flight prompt() turn. Stored on a
// map (not a local) so doCompact() — which runs in its own scope — can reset
// it after compaction replaces the conversation with a short summary. Without
// a reset, the estimator keeps counting pre-compaction content and over-reports.
const sessionEstimators = new Map<string, TurnContextEstimator>();

// ─── Real-time context estimation ──────────────────────────────────────────
// Claude CLI only reports usage in the result event at turn END (cumulative).
// To show usage DURING the turn, we estimate context size from stream events:
// accumulate conversation content (text, tool calls, tool results) and add
// the fixed overhead (system prompt + tool definitions).
// Approximate token count: ~4 chars per token (rough heuristic).
const CHARS_PER_TOKEN = 4;
// Claude CLI system prompt is ~7KB. Tools (24 defs) are ~65KB. These are sent
// on EVERY API call. Pre-computed: 7148 + 65600 = 72748 bytes overhead.
const CLAUDE_OVERHEAD_BYTES = 72748;

/** Estimate tokens from a character count (rough: 4 chars ≈ 1 token). */
function estimateTokensFromChars(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN);
}

/** Per-turn accumulator for estimating real-time context size. */
class TurnContextEstimator {
  private contentBytes = 0;
  private apiCallCount = 0;

  /** Called when a new assistant event arrives (marks an API call boundary). */
  markApiCall(): void {
    this.apiCallCount++;
  }

  /** Accumulate text content (assistant messages, thinking, tool results). */
  addText(text: string): void {
    if (text) this.contentBytes += text.length;
  }

  /**
   * Reset accumulated conversation content (keep the fixed overhead).
   * Called after compaction, which replaces the entire conversation with a
   * short summary — without this, the estimator would keep counting the
   * pre-compaction content and over-report context size.
   */
  reset(): void {
    this.contentBytes = 0;
    this.apiCallCount = 0;
  }

  /** Accumulate tool call content (JSON-serialized input). */
  addToolCall(toolName: string, toolInput: unknown): void {
    this.contentBytes += toolName.length;
    try {
      this.contentBytes += JSON.stringify(toolInput ?? {}).length;
    } catch {
      this.contentBytes += 50; // fallback estimate
    }
  }

  /** Get the estimated context size (overhead + conversation content). */
  getEstimatedTokens(): number {
    const totalBytes = CLAUDE_OVERHEAD_BYTES + this.contentBytes;
    return estimateTokensFromChars(totalBytes);
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
  // Only `cd`/comment/blank lines (or empty) — return whatever remains.
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
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: titleForCommand(command, description),
        kind: "execute",
        status: "completed",
        rawInput: toolInput ?? {},
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

function buildConfigOptions(store: SessionStore, sessionId: string) {
 const modelOverride = store.getConfigOverrides(sessionId)["model"] as string | undefined;
 const currentModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
 const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
 const effortOverride = store.getConfigOverrides(sessionId)["effort"] as string | undefined;
  const currentCollabMode = sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE;

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

  // ─── Subagent (Agent tool) lifecycle tracking ─────────────────────────────
  // The Agent tool spawns a subagent that runs for an extended period. We
  // must NOT mark it "completed" the instant its tool_use block appears (as
  // we do for Read/Edit/etc.) — the subagent is still running and the UI
  // should reflect that. Instead we emit status:"in_progress" when the
  // tool_use fires and flip to "completed" only when the matching tool_result
  // arrives (the user/tool_result message Claude emits once the subagent
  // finishes). Maps Claude's native tool_use id → our ACP toolCallId.
  const sessionPendingAgents = new Map<string, Map<string, string>>();

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

    if (name === "compact") {
      const compactEffort = resolveEffort(store, sessionId);
      await doCompact(sessionId, connection, runner, store, store.getConfigOverrides(sessionId)["model"] as string | undefined, compactEffort, true);
      return { handled: true };
    }

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
    const modelOverride = store.getConfigOverrides(sessionId)["model"] as string | undefined;
    const effortOverride = store.getConfigOverrides(sessionId)["effort"] as string | undefined;
    const effectiveModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
    const modelEfforts = MODEL_EFFORTS[effectiveModel] ?? [];
    if (effortOverride && modelEfforts.includes(effortOverride)) return effortOverride;
    return getDefaultEffort(effectiveModel);
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
    if (announce) {
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Compacting conversation..." },
        } as any,
      });
    }
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
      if (announce) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "Conversation compacted successfully." },
          } as any,
        });
      }
      logger.info(`Compaction succeeded for session ${sessionId}`);
      return true;
    } catch (compactErr) {
      logger.error(`Compaction failed for session ${sessionId}: ${compactErr instanceof Error ? compactErr.message : String(compactErr)}`);
      if (announce) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: `Compaction failed: ${compactErr instanceof Error ? compactErr.message : String(compactErr)}` },
          } as any,
        });
      }
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

     if (!store.has(sessionId)) {
       throw RequestError.resourceNotFound(
         `Session ${sessionId} not found`
       );
     }

     const currentMode = store.getMode(sessionId) ?? DEFAULT_MODE;
     logger.info(`Load session: ${sessionId}`);

      // Replay conversation history from Claude CLI's JSONL transcript.
      // The transcript lives at ~/.claude/projects/<encoded-cwd>/<claude-session-id>.jsonl
      const claudeSessionId = store.getClaudeSessionId(sessionId);
      if (claudeSessionId) {
        try {
          const { homedir } = await import("node:os");
          const cwd = store.getCwd(sessionId)!;
          const encodedCwd = cwd.replace(/\//g, "-");
          const transcriptPath = `${homedir()}/.claude/projects/${encodedCwd}/${claudeSessionId}.jsonl`;
          const { readFileSync, existsSync } = await import("node:fs");
          if (existsSync(transcriptPath)) {
            const raw = readFileSync(transcriptPath, "utf-8");
            const lines = raw.split("\n").filter((l) => l.trim());
            logger.info(`Replaying ${lines.length} transcript lines from ${transcriptPath}`);
            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
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
                        await connection.sessionUpdate({
                          sessionId,
                          update: {
                            sessionUpdate: "tool_call",
                            toolCallId: block.id ?? `replay-${Math.random().toString(36).slice(2, 8)}`,
                            title: block.name ?? "tool",
                            kind: "execute",
                            status: "completed",
                            rawInput: block.input ?? {},
                          } as any,
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

      // Handle collaboration_mode specially (stored in separate map)
      if (configId === COLLABORATION_MODE_CONFIG_ID) {
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
      const effectiveModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
      const effectiveEffort = resolveEffort(store, sessionId);

      logger.info(`Prompt for session ${sessionId}: ${text.length} chars`);
      logger.debug(
        `Prompt content: ${text.slice(0, 100)}${text.length > 100 ? "..." : ""}`
      );

      const permissionPromises: Promise<void>[] = [];

      let lastUsage: { input_tokens?: number; output_tokens?: number } | null = null;
      // Track whether Claude CLI's final result was an error (e.g.
      // error_max_turns) so we can surface it to the user instead of
      // returning a silent end_turn. Mirrors codex-acp, which maps upstream
      // turn failures to a visible error rather than an empty success.
      let turnErrored = false;
      let turnStopReason: string | undefined;
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
        sendEstimatedUsage(ctxEstimator, connection, sessionId, effectiveModel);
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
      // Non-subagent tool_result ids are ignored here (they were never added to
      // sessionPendingAgents; they completed inline at their tool_use).
    // Handle result event — Claude CLI reports CUMULATIVE usage across all
    // API calls in the turn (accumulateUsage in QueryEngine.ts). For a
    // single-call turn, input_tokens = real context size (accurate). For a
     // multi-call turn (tool calls), input_tokens = sum of all calls' input
     // tokens, which inflates the usage UI (e.g. 3 calls × 20k = 60k shown,
     // but real context is ~20k).
     //
     // Claude CLI does NOT forward message_delta events to stdout in
     // stream-json mode (includePartialMessages defaults to false), so we
     // cannot get per-call usage from the stream. The result event is our
     // only source.
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
      const effectiveModel = modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id;
      const contextWindow = getContextWindowForModel(effectiveModel);
      // For auto-compact: use ONLY our estimate (not the cumulative result value).
      const estimatedTokens = ctxEstimator.getEstimatedTokens();
      sessionLastUsage.set(sessionId, { input_tokens: estimatedTokens, output_tokens: outputTokens });
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
         // streaming deltas. UI rendering is unaffected — both still render.
         if (event.fromAssistantBlock) {
           ctxEstimator.addText(event.text);
         }
         connection.sessionUpdate({
           sessionId,
           update: {
             sessionUpdate: "agent_message_chunk",
             content: { type: "text", text: event.text },
           },
         });
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
         // A subagent runs long after its tool_use block appears. Emit it as
         // in_progress (not completed) and record the Claude tool_use id so
         // we can flip it to completed when the subagent's tool_result lands.
         const emitted = isSubagent
           ? { ...update, status: "in_progress" as const }
           : update;
         if (isSubagent && event.toolUseId) {
           let pending = sessionPendingAgents.get(sessionId);
           if (!pending) {
             pending = new Map();
             sessionPendingAgents.set(sessionId, pending);
           }
           pending.set(event.toolUseId, toolCallId);
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
                update: { ...baseUpdate, status: "failed" },
              });
            });

          permissionPromises.push(permPromise);
        }
      };

      try {
        let result;
        if (!skipMainTurn) {
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
                  sessionUpdate: "tool_call",
                  toolCallId: acpCallId,
                  status: "completed",
                } as any,
              })
              .catch(() => {});
          }
          pendingAgents.clear();
          sessionPendingAgents.delete(sessionId);
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
            // /clear wipes the Claude session ID — there's no session to
            // continue into, so stop draining remaining queued steers.
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
            // Re-queue at the front so it drains once the subagent resolves.
            const q = steerQueues.get(sessionId) ?? [];
            q.unshift(steerText);
            steerQueues.set(sessionId, q);
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
          store.touch(sessionId);
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "session_info_update",
              updatedAt: store.getUpdatedAt(sessionId),
            } as any,
          });
        }

        // ─── Auto-compact check ───────────────────────────────────────────
        // Mirrors Codex core: after the turn finishes (including steer
        // drains), check if token usage has reached 90% of the model's
        // context window. If so, run the compaction flow to replace the
        // session with a summary-seeded fresh one. This prevents context
        // exhaustion on long conversations.
        const lastUsageEntry = sessionLastUsage.get(sessionId);
        if (lastUsageEntry) {
          const threshold = getAutoCompactThreshold(
            modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id,
          );
          const used = lastUsageEntry.input_tokens + lastUsageEntry.output_tokens;
          if (used >= threshold) {
            logger.info(
              `Auto-compact triggered for session ${sessionId}: ${used} >= ${threshold} (90% of ${getContextWindowForModel(modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id)})`,
            );
            // Emit a compaction notice to the UI (matches Codex's style).
            await connection.sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "agent_message_chunk",
                content: { type: "text", text: "*Context compacted to fit the model's context window.*\n\n" },
              } as any,
            });
           await doCompact(
             sessionId,
             connection,
             runner,
             store,
             modelOverride,
             effectiveEffort,
             false, // don't announce "Compacting..." — we already emitted the notice above
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
             const contContextWindow = getContextWindowForModel(
               modelOverride || process.env.ANTHROPIC_MODEL || PROXY_MODELS[0].id,
             );
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
