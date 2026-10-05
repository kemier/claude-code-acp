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
  type ForkSessionRequest,
  type ForkSessionResponse,
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
import { SessionStore, type SessionInfo } from "./session-store.js";
import { buildMidTurnModeChangeNotice, shouldAnnounceModeChange } from "./plan-mode-notice.js";
import { ClaudeRunner, isProxyBaseUrl, type StreamEvent, type McpServerConfig } from "./claude-runner.js";
import { logger } from "./logger.js";
import { saveSessionSlot, tryRestoreSlot } from "./slot-cache.js";
import {
  getAvailableModelsFromSettings,
  getClaudeEnv,
  getDefaultEffortFromSettings,
  getDefaultModelFromSettings,
  filterPickerModels,
  type ProxyModelEntry,
  type ProxyModelHealthEntry,
} from "./claude-settings.js";
import {
  validateCwd,
  validateMcpCommand,
  validateMcpArgs,
} from "./validation.js";
import { isAbsolute as isAbsolutePath, relative as relativePath, resolve as resolvePath, join as joinPath } from "node:path";
import { randomUUID } from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";

function generateSessionId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Normalize the ACP `mcpServers` request param into internal
 * `McpServerConfig[]` (the shape `claude-runner.buildMcpArgs` consumes).
 *
 * Accepts the ACP *wire* shape (`McpServerStdio`: flat
 * `{name, command, args, env: EnvVariable[]}` — what vscode-acp sends in
 * newSession/loadSession/resumeSession) as well as the legacy nested
 * `{transport: {type, command, args, env}}` form, and normalizes `env` to a
 * `Record<string,string>`. Non-stdio entries are dropped.
 */
function normalizeMcpServers(servers: any[]): McpServerConfig[] {
  return servers
    .filter(
      (s: any) =>
        (s && s.transport && s.transport.type === "stdio") ||
        (s && s.command && typeof s.command === "string")
    )
    .map((s: any) => {
      const t = s.transport && s.transport.type === "stdio" ? s.transport : s;
      const command = t.command;
      const args = t.args ?? [];
      // env may be the ACP wire form (Array<{name,value}>) or a plain object.
      const rawEnv = t.env;
      const env: Record<string, string> = {};
      if (Array.isArray(rawEnv)) {
        for (const v of rawEnv) {
          if (v && typeof v.name === "string" && typeof v.value === "string") {
            env[v.name] = v.value;
          }
        }
      } else if (rawEnv && typeof rawEnv === "object") {
        for (const [k, val] of Object.entries(rawEnv)) {
          env[k] = String(val);
        }
      }
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
          env,
        },
      };
    });
}

// Monotonic id for one-shot agent_message_chunk emits (slash-command
// responses, status notices). Each gets a distinct id so a client grouping
// by messageId treats each as its own message (never merges them with
// adjacent assistant text). See ACP spec: ContentChunk.messageId.
let commandMessageCounter = 0;
function nextCommandMessageId(): string {
  return `cmd_msg_${++commandMessageCounter}`;
}

/**
 * Converts ACP `image` content blocks into Claude SDK stdin image blocks
 * (Anthropic Messages API shape) for the `--input-format=stream-json` path.
 *
 * ACP `ImageContent` carries base64 `data` + `mimeType`; the CLI's SDK input
 * expects `{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`.
 * Blocks with missing/empty base64 data are skipped (best-effort), matching
 * ericli's `extractPromptImages` behavior.
 */
export function acpImageBlocksToStdinBlocks(
  blocks: ReadonlyArray<{ type: string; data?: unknown; mimeType?: string | null }>,
): import("./claude-runner.js").StdinContentBlock[] {
  const out: import("./claude-runner.js").StdinContentBlock[] = [];
  for (const block of blocks) {
    if (block.type !== "image") continue;
    const data = block.data;
    if (typeof data !== "string" || data.length === 0) continue;
    const mediaType =
      typeof block.mimeType === "string" && block.mimeType.length > 0
        ? block.mimeType
        : "image/png";
    out.push({
      type: "image",
      source: { type: "base64", media_type: mediaType, data },
    });
  }
  return out;
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

/**
 * Leave Plan mode for `sessionId` and tell the client.
 *
 * Claude Code signals plan completion with its ExitPlanMode tool. `setSessionMode`
 * echoes client-initiated switches, so this is the only agent-initiated path.
 *
 * Two independent pickers must both flip back:
 *   - permission mode (store.mode, "agent"/"read-only") via current_mode_update
 *   - collaboration mode (sessionCollaborationModes) via config_option_update
 *
 * `enterPlanMode` sets the permission mode to "read-only" when the model
 * self-enters plan mode, so leaving must restore it to "agent" (full access) —
 * otherwise the permission picker stays stuck on "Read-only" after the plan is
 * approved and the model starts implementing.
 */
async function leavePlanMode(
  sessionId: string,
  connection: { sessionUpdate: (p: any) => Promise<unknown> },
  store: SessionStore,
  runner?: { setSessionPermissionMode: (m: string | null) => void },
): Promise<void> {
  if ((sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE) !== PLAN_COLLABORATION_MODE) {
    return;
  }
  sessionCollaborationModes.set(sessionId, DEFAULT_COLLABORATION_MODE);
  // Plan mode is read-only by design; restoring full access so the model can
  // implement the approved plan. Both the stored mode (persists across
  // restarts, drives the picker) and the runner's spawn flag (drives the
  // --permission-mode the next claude -p process receives) are updated.
  store.setMode(sessionId, DEFAULT_MODE);
  runner?.setSessionPermissionMode(DEFAULT_MODE);
  logger.info(`Session ${sessionId} left plan mode (ExitPlanMode)`);
  try {
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "current_mode_update", currentModeId: DEFAULT_MODE } as any,
    });
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "config_option_update", configOptions: buildConfigOptions(store, sessionId) } as any,
    });
  } catch {
    /* non-fatal — the next turn still runs without the plan-mode preamble */
  }
}

/**
 * Enter Plan mode for `sessionId` and tell the client.
 *
 * Mirror of {@link leavePlanMode}. The Claude CLI exposes a native EnterPlanMode
 * tool the model calls when it decides a task needs planning before acting
 * (mirrors ericli's auto enter_plan_mode). Without mapping it the picker kept
 * reading "Default" while the model was actually planning — the user had no way
 * to see that plan mode was active until they flipped the picker themselves.
 *
 * Two independent pickers must both flip to "plan":
 *   - permission mode → "read-only" (current_mode_update with a valid
 *     AVAILABLE_MODES id + store.setMode so it persists and buildConfigOptions
 *     reflects it)
 *   - collaboration mode → "plan" (config_option_update "collaboration_mode")
 *
 * The old code sent currentModeId: "plan" here, but "plan" is not an id in
 * AVAILABLE_MODES (which is read-only/agent/agent-full-access), so vscode-acp's
 * availableModes.find() missed and the permission picker silently stayed on the
 * previous mode. Sending "read-only" is the valid id that makes the picker
 * flip to "Read-only" in lockstep with the collaboration picker flipping to
 * "Plan".
 */
async function enterPlanMode(
  sessionId: string,
  connection: { sessionUpdate: (p: any) => Promise<unknown> },
  store: SessionStore,
  runner?: { setSessionPermissionMode: (m: string | null) => void },
): Promise<boolean> {
  if ((sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE) === PLAN_COLLABORATION_MODE) {
    // Already in plan mode — the model re-called EnterPlanMode (e.g. after a
    // sub-turn). Nothing to do; re-emitting would be a redundant picker update.
    // Return false so the caller does NOT re-interrupt a turn that is already
    // running under Plan.
    return false;
  }
  sessionCollaborationModes.set(sessionId, PLAN_COLLABORATION_MODE);
  // Plan mode is read-only: the model investigates and presents a plan but does
  // not edit files or run commands. Switching the permission mode to "read-only"
  // makes the IDE's permission picker reflect that (agent → read-only) AND sets
  // the runner's spawn flag so the next claude -p process is launched with
  // --permission-mode plan — the CLI then hard-blocks mutating tools at its own
  // layer, not just via the <collaboration_mode> prompt block.
  store.setMode(sessionId, "read-only");
  runner?.setSessionPermissionMode("read-only");
  logger.info(`Session ${sessionId} entered plan mode (EnterPlanMode)`);
  try {
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "current_mode_update", currentModeId: "read-only" } as any,
    });
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "config_option_update", configOptions: buildConfigOptions(store, sessionId) } as any,
    });
  } catch {
    /* non-fatal — the next turn still injects the plan-mode preamble */
  }
  return true;
}

/**
 * Apply a user-initiated permission-mode change (the Mode picker) and tell the
 * client, keeping every surface consistent:
 *
 *   - store.mode            → persists across restarts + drives buildConfigOptions
 *   - runner perm mode      → the --permission-mode the next claude -p spawn gets
 *                              (WITHOUT this the picker flip only took effect on
 *                              the next turn, and an in-flight turn kept its old
 *                              flags)
 *   - collaboration mode    → the two pickers flip in lockstep (read-only ↔ plan)
 *   - current_mode_update   → legacy `modes` display + permission picker label
 *   - config_option_update  → fresh configOptions so the config pickers re-render
 *
 * The model-initiated path (EnterPlanMode/ExitPlanMode) routes through
 * enterPlanMode/leavePlanMode, which do the same state writes with their own
 * no-op guards; this is the USER-initiated path that was previously missing the
 * runner sync and the current_mode_update — the reason flipping the Mode picker
 * to "Read-only" appeared to do nothing.
 */
async function applyModeChange(
  sessionId: string,
  connection: { sessionUpdate: (p: any) => Promise<unknown> },
  store: SessionStore,
  runner?: { setSessionPermissionMode: (m: string | null) => void },
  modeId?: string,
): Promise<void> {
  const mode = modeId ?? store.getMode(sessionId) ?? DEFAULT_MODE;
  // read-only ⇔ plan are the two halves of "plan mode": flipping one flips the
  // other so the permission and collaboration pickers never disagree.
  const collab = mode === "read-only" ? PLAN_COLLABORATION_MODE : DEFAULT_COLLABORATION_MODE;
  sessionCollaborationModes.set(sessionId, collab);
  store.setMode(sessionId, mode);
  runner?.setSessionPermissionMode(mode);
  logger.info(`Session ${sessionId} mode change applied: ${mode} (collab ${collab})`);
  try {
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "current_mode_update", currentModeId: mode } as any,
    });
    await connection.sessionUpdate({
      sessionId,
      update: { sessionUpdate: "config_option_update", configOptions: buildConfigOptions(store, sessionId) } as any,
    });
  } catch {
    /* non-fatal — the next turn still syncs the runner from store.getMode() */
  }
}

function buildModeState(currentModeId: string) {
  return {
    availableModes: AVAILABLE_MODES,
    currentModeId,
  };
}

/**
 * Adapter-neutral method for reporting what a finished subagent found.
 *
 * Not under a `_claude/` prefix: other adapters send the same method with the
 * same shape, and a client that persists findings should not have to learn one
 * dialect per adapter. The client treats it as a notification and may ignore it.
 */
const SUBAGENT_FINDINGS_METHOD = "_acp/subagent/findings";

/** Bound on a reported summary — enough to be useful, not a transcript dump. */
const MAX_FINDINGS_SUMMARY_CHARS = 8000;

/**
 * Pull a finished subagent's result out of its `<task-notification>`.
 *
 * Claude Code delivers a completed subagent as a user message like:
 *
 *   <task-notification>
 *   Agent "Trace interrupt message wiring" completed
 *   <summary>… what it found …</summary>
 *   </task-notification>
 *
 * The `<summary>` is the part worth keeping. When it is absent the notification
 * body is used instead, minus the agent-name line that only repeats the
 * description we already send as the task.
 *
 * Returns null when nothing substantive is left, so an empty notification is not
 * reported as a finding with no content.
 */
/**
 * True when a failed `claude` turn failed for a reason that says nothing about
 * the session id — a dead or unhappy upstream, not a stale conversation.
 *
 * This decides whether a failed `--resume` may destroy the session mapping.
 * Reported case: the local model proxy was killed mid-session, `--resume` exited
 * 1 with "API Error: Connection refused", and the resume fallback read that as
 * "expired session", cleared the persisted id and started a FRESH Claude
 * conversation. The user resumed, said "go on", and the agent answered "I don't
 * have any prior task in this session" — 68 commands of context abandoned while
 * the transcript still sat on disk. Retrying an outage costs nothing; losing the
 * conversation is unrecoverable through the UI.
 *
 * `sawApiError` comes from the stream's own result event (isError), which is the
 * most reliable signal: it is set whenever Claude reported an API-level failure,
 * whatever the wording.
 */
export function isTransientTurnFailure(message: string, sawApiError = false): boolean {
  if (sawApiError) {
    return true;
  }
  const m = String(message || '');
  return (
    /API Error/i.test(m) ||
    /connection refused|ECONNREFUSED|ECONNRESET|EPIPE/i.test(m) ||
    /ETIMEDOUT|ENETUNREACH|EHOSTUNREACH|EAI_AGAIN|ENOTFOUND/i.test(m) ||
    /socket hang up|fetch failed|network error|tunnel|proxy/i.test(m) ||
    /\b(?:429|5\d\d)\b/.test(m) ||
    /overloaded|rate.?limit|too many requests|timed? ?out|timeout/i.test(m) ||
    /unauthorized|forbidden|invalid.{0,12}api.{0,4}key|token expired/i.test(m)
  );
}

export function extractTaskNotificationSummary(
  notif: string,
  description: string,
): { summary: string; stopped: boolean } | null {
  const text = String(notif || "");
  // "… [stopped]" marks a subagent that did not run to completion. Its partial
  // work is still worth recording, but not as a success.
  const stopped = /\bstopped\b/i.test(text);

  let body = "";
  const sm = /<summary>([\s\S]*?)<\/summary>/i.exec(text);
  if (sm) {
    body = sm[1];
  } else {
    body = text
      .replace(/<\/?task-notification>/gi, "")
      // Drop the "Agent \"desc\" completed" / "background agent \"desc\" …" line:
      // it carries no findings and duplicates the task we already send.
      .replace(/^\s*(background\s+)?agent\s+"[^"]*"[^\n]*\n?/gim, "");
  }
  // The description appears verbatim in the notification (that is how the
  // spawn→completion match is made) and adds nothing next to `task`.
  if (description) {
    body = body.split(description).join("").trim();
  }
  body = body.trim();
  if (!body) {
    return null;
  }
  if (body.length > MAX_FINDINGS_SUMMARY_CHARS) {
    body = body.slice(0, MAX_FINDINGS_SUMMARY_CHARS) + "\n…(truncated)";
  }
  return { summary: body, stopped };
}


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
  // DeepSeek-Flash family (V4-Flash, V4.1-Flash, …): supports the full ladder
  // including xhigh — an EricAI-specific extension these deployments honour
  // (verified via ericli's supportedReasoningEffortsForModel). Matched by the
  // case-insensitive deepseek.*flash regex below, NOT this literal key, so a
  // version bump in the /v1/models id (e.g. "deepseek-v4-flash" →
  // "DeepSeek-V4.1-Flash") does not silently drop xhigh and fall back to
  // DEFAULT_EFFORTS. The literal is retained as documentation of the family.
  "deepseek-v4-flash": ["low", "medium", "high", "xhigh"],
  "kimi-k2.7-code": ["low", "high"],
  "kimi-k2.6": ["low", "high"],
  "mimo-v2.5-pro": ["low", "high"],
  "mimo-v2.5": ["low", "high"],
  "qwen2.5-14b-1m": [],
  // Qwen3.8-27B (standard + FP8-1M): only low/medium are sendable. Verified
  // raw against the backendsso vLLM (2026-09-22): "high" 500s at the chat
  // template ("Supported types are xhigh (default), medium, and low"),
  // "xhigh" 422s at the outer schema ("Input should be 'low', 'medium' or
  // 'high'"). The proxy drops high/xhigh before the wire, so advertising
  // them would make the picker show a "High" that silently maps to
  // enable_thinking:false — hide it. (Stale "qwen38-27b" slug removed; the
  // live model id from /v1/models is "Qwen3.8-27B".)
  "Qwen3.8-27B": ["low", "medium"],
  "Qwen3.8-27B-FP8-1M": ["low", "medium"],
  "qwythos": ["low", "medium", "xhigh"],
  // Qwen3.8-Flash-Next: the Ray cluster's extra_body_suggestions advertise
  // only low/medium (verified via `ericai api models.list`). `high` 500s at
  // the deployed chat template; upstream's `xhigh` 422s at the vLLM schema.
  "Qwen3.8-Flash-Next": ["low", "medium"],
};

const DEFAULT_EFFORTS = ["low", "medium", "high"];

// Case-insensitive lookup key: lowercases both the table keys (lazily, once)
// and the incoming model id. The /v1/models id is passed through verbatim by
// the picker (e.g. "DeepSeek-V4.1-Flash", "Qwen3.8-27B"), and the table above
// mixes cases, so a raw hasOwnProperty lookup is case-fragile.
const MODEL_EFFORTS_LOWER: Record<string, string[]> = (() => {
  const out: Record<string, string[]> = {};
  for (const [k, v] of Object.entries(MODEL_EFFORTS)) out[k.toLowerCase()] = v;
  return out;
})();

function getEffortsForModel(modelId: string): string[] {
  const lower = modelId.toLowerCase();
  // DeepSeek-Flash family: version-tolerant match mirroring ericli's
  // supportedReasoningEffortsForModel (/deepseek.*flash/i). The backend was
  // upgraded from "deepseek-v4-flash" (alias) to "DeepSeek-V4.1-Flash"
  // (/v1/models id); a literal table key would miss the new slug and silently
  // drop xhigh. Any deepseek+flash id — current or future — gets the full
  // ladder. (deepseek-v4-pro is NOT a flash model and stays on its table entry.)
  if (/deepseek.*flash/.test(lower)) {
    return ["low", "medium", "high", "xhigh"];
  }
  if (Object.prototype.hasOwnProperty.call(MODEL_EFFORTS_LOWER, lower)) {
    return MODEL_EFFORTS_LOWER[lower];
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

// Watchdog timers for web tools that can never return on the current route:
// WebSearch always (it executes on Anthropic's API, which a proxy route has no
// executor for), and WebFetch when the operator names it in
// CLAUDE_ACP_DISALLOWED_WEB_TOOLS (the CLI fetches client-side, so a
// network-blocked host — e.g. China — stalls it). Keyed by session so a
// settled turn can clear them wholesale. Each timer also no-ops if its tool
// completed or was flushed in the meantime.
const webToolWatchdogs = new Map<string, Set<NodeJS.Timeout>>();
const WEB_TOOL_TIMEOUT_MS = 90_000;

function watchedWebTools(): Set<string> {
  const names = new Set(["WebSearch"]);
  const drop = process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS;
  if (
    drop &&
    drop
      .split(",")
      .map((t) => t.trim())
      .includes("WebFetch")
  ) {
    names.add("WebFetch");
  }
  return names;
}

/** Built-in slash commands (mirrors codex-acp's CodexCommands). */
const BUILTIN_COMMANDS = [
  {
    name: "ps",
    description: "List background terminal processes.",
    input: null,
    // Read-only snapshot: dispatchSlashCommand services it from current state
    // with no LLM turn and no state change, so vscode-acp may send it mid-turn
    // as a side-channel request (surviving any streaming turn on the agent
    // side) instead of queuing it. Mirrors ericli's /ps declaration.
    _meta: { turnSafe: true },
  },
  {
    name: "list-agents",
    description: "List other Claude Code sessions on this machine.",
    input: null,
    // Read-only: scans ~/.claude/sessions/*.json and emits a snapshot.
    // No turn, no state change.
    _meta: { turnSafe: true },
  },
  {
    name: "goal",
    description: "Set a goal to keep pursuing.",
    input: { hint: "[<objective>|clear|pause|resume]" },
    // Turn-safe for the subcommands that need NO LLM turn (bare status, /clear
    // and its aliases, /pause, /resume). exactArgs (not a plain array, not
    // true) so that `/goal <objective>` still starts a turn and therefore
    // queues mid-turn — same declaration vscode-acp's tests and ericli use.
    _meta: { turnSafe: { exactArgs: ['', 'clear', 'stop', 'off', 'reset', 'none', 'cancel', 'pause', 'resume'] } },
  },
  {
    name: "skills",
    description: "List available skills.",
    input: null,
    // Read-only: scans the skill dirs and emits a snapshot. No turn, no state.
    _meta: { turnSafe: true },
  },
  {
    name: "status",
    description: "Display session configuration and token usage.",
    input: null,
    // Read-only snapshot of config/usage; no turn, no state change.
    _meta: { turnSafe: true },
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

/**
 * Claude CLI-native slash commands that are READ-ONLY introspection — they
 * inspect session/model/context state and neither start an LLM turn nor change
 * anything. When the CLI's system_init reports them we mark them turnSafe so
 * vscode-acp can fire them mid-turn as a side-channel (same rationale as the
 * builtin /ps, /status, /skills declarations). This is a static allowlist
 * because the CLI's system_init only sends flat name strings, not
 * descriptions or capability metadata, so the adapter must know which natives
 * are safe on its own.
 *
 * Anything NOT here is conservatively treated as potentially state-mutating
 * (e.g. /model, /config, /output-style) and left non-turnSafe, which makes
 * vscode-acp queue it behind the running turn.
 */
const NATIVE_TURN_SAFE_COMMANDS = new Set([
  "context",
  "usage",
  "insights",
  "recap",
  "heapdump",
]);

/** Per-session goal state. */
const sessionGoals = new Map<string, { text: string; paused: boolean; createdAt: number; pausedAt: number | null; tokensUsed: number; timeUsedSeconds: number }>();

/**
 * Publish the current goal snapshot to the IDE via session_info_update with
 * _meta.codex.goal. This mirrors codex-acp's publishGoalSnapshot so the
 * vscode-acp goal chip works across all agents.
 */
async function publishGoalSnapshot(
  sessionId: string,
  connection: AgentSideConnection,
): Promise<void> {
  const goal = sessionGoals.get(sessionId);
 const snapshot = goal
   ? {
       objective: goal.text,
       status: goal.paused ? "paused" as const : "active" as const,
      tokenBudget: null,
       tokensUsed: goal.tokensUsed,
       timeUsedSeconds: goal.paused && goal.pausedAt
         ? Math.floor((goal.pausedAt - goal.createdAt) / 1000)
         : Math.floor((Date.now() - goal.createdAt) / 1000),
       createdAt: Math.floor(goal.createdAt / 1000),
      controlMethod: "_codex/session/goal_control",
     }
   : null;
  await connection.sessionUpdate({
    sessionId,
    update: {
      sessionUpdate: "session_info_update",
      _meta: { codex: { goal: snapshot } },
    } as any,
  }).catch(() => {});
}

// ─── Context window resolution ──────────────────────────────────────────────
// Default context window used when the proxy doesn't report one for a model.
const DEFAULT_CONTEXT_WINDOW = 200_000;

// Cached map of model-id → context_length, fetched from the proxy's
// /v1/models endpoint. Falls back to DEFAULT_CONTEXT_WINDOW.
const modelContextWindows = new Map<string, number>();
// Cached map of model-id → vision-capable, derived from the proxy's
// /v1/models `tags` (the `multimodal` tag, matching ericli's
// hasMultimodalTag). Populated alongside modelContextWindows. When the proxy
// is not in use (no base-url override) the map stays empty and the
// name-pattern fallback in modelSupportsVision() decides — Claude's own
// models are vision-capable, so the default is true for a bare Claude session.
const modelVisionCapable = new Map<string, boolean>();
// Once a fetch has SUCCEEDED this is set so we don't re-fetch on every lookup.
// A FAILED fetch (proxy unreachable, non-200) deliberately leaves this false so
// the next caller retries — otherwise a transiently-unreachable proxy at
// process spawn would freeze the map empty for the whole session, and every
// model would fall through to the CLAUDE_CODE_MAX_CONTEXT_TOKENS override
// (pinned to one model's window), mis-reporting every other model's denominator
// and compaction threshold.
let modelContextWindowsLoaded = false;
// In-flight fetch promise so concurrent callers share one request without
// marking the load complete until it settles.
let modelContextWindowsLoadPromise: Promise<void> | null = null;

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
 * Called once at module load; safe to call again to refresh. Concurrent
 * callers share a single in-flight fetch. The "loaded" gate is only set on
 * SUCCESS — a failed fetch leaves the map retryable so a transiently
 * unreachable proxy at spawn doesn't freeze the wrong override in for the
 * whole session.
 */
export async function loadModelContextWindows(): Promise<void> {
  if (modelContextWindowsLoaded) return;
  if (modelContextWindowsLoadPromise) return modelContextWindowsLoadPromise;
  modelContextWindowsLoadPromise = (async () => {
    try {
      const url = `${getProxyBaseUrl()}/v1/models`;
      const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
      if (!resp.ok) {
        logger.warn(`Failed to fetch /v1/models from proxy (${resp.status}): ${url}`);
        return;
      }
      const body = await resp.json() as { data?: Array<{ id: string; context_length?: number; tags?: string[] }> };
      if (body.data) {
        for (const m of body.data) {
          if (typeof m.context_length === "number" && m.context_length > 0) {
            modelContextWindows.set(m.id, m.context_length);
          }
          // Capture the multimodal tag so image prompts can be gated on the
          // backend's actual vision capability (a proxy base-url override can
          // serve a text-only model behind the "Claude" wire format).
          if (Array.isArray(m.tags)) {
            modelVisionCapable.set(m.id, m.tags.some((t) => String(t).toLowerCase() === "multimodal"));
          }
        }
        logger.info(`Loaded ${modelContextWindows.size} model context windows from proxy`);
        // Only pin the load complete once we actually have data. A failed or
        // empty fetch stays retryable.
        if (modelContextWindows.size > 0) modelContextWindowsLoaded = true;
      }
    } catch (err) {
      logger.warn(`Could not load model context windows from proxy: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      modelContextWindowsLoadPromise = null;
    }
  })();
  return modelContextWindowsLoadPromise;
}

/**
 * Name-pattern fallback for vision capability, used when the proxy tag map has
 * no entry for the model (proxy not in use, tag fetch failed, or a model the
 * catalog didn't advertise). Mirrors pi-acp's modelSupportsImage: the common
 * multimodal families. A bare Claude session (no base-url override) is
 * vision-capable by default, so an unknown id on the native Claude path
 * defaults to true — only a PROXY-served model with a known text-only name
 * defaults to false.
 */
const VISION_MODEL_NAME_PATTERN =
  /(^|[^a-z])(claude|gpt-4o|gpt-4\.1|gpt-5|gemini|qwen.*vl|qwen3\.8-27b|llava|glm-4v|glm-4\.5v|phi-3-vision|o1|o3|o4|deepseek-vl|kimi-vl)/i;

/**
 * True when the model serving this session can accept image inputs.
 *
 * Priority:
 *  1. Proxy tag map (authoritative — the backend curator's `multimodal` tag).
 *  2. Name-pattern fallback when the map has no entry.
 *
 * When no proxy base-url is configured (native Claude), the map is empty and
 * the name pattern matches "claude" → true, so image prompts pass through
 * unchanged (zero behavior change for the common case).
 */
export function modelSupportsVision(modelId: string | undefined): boolean {
  const id = String(modelId ?? "").toLowerCase();
  if (!id) return true; // unknown model on native Claude → assume vision-capable
  if (modelVisionCapable.has(id)) return modelVisionCapable.get(id)!;
  // Map miss: only a proxy base-url override makes the tag map non-empty in
  // practice. For a proxy-served model we couldn't look up, fall back to the
  // name pattern. For native Claude (no proxy), "claude" matches → true.
  return VISION_MODEL_NAME_PATTERN.test(id);
}

const MIME_TO_EXT: Record<string, string> = {
  "image/png": "png",
  "image/jpeg": "jpg",
  "image/jpg": "jpg",
  "image/gif": "gif",
  "image/webp": "webp",
  "image/bmp": "bmp",
  "image/svg+xml": "svg",
};

/**
 * Graceful degradation for image prompts when the serving model is text-only
 * (e.g. a proxy base-url override routes "Claude" to a non-multimodal backend).
 * Instead of letting the upstream error on the image blocks, persist each image
 * to a temp file and return a text note describing what was attached and where
 * it lives on disk. The prompt proceeds text-only; the model/user can switch to
 * a vision model and re-attach, or use a tool that routes to one.
 *
 * Mirrors codex-acp's degradeImagePromptForTextOnlyModel and ericli's
 * writePromptImagesToTemp + buildNonVisionImageReminder: never lose the user's
 * input, degrade instead of fail.
 */
export function degradeImageBlocksForTextOnlyModel(
  images: Array<{ source: { media_type: string; data: string } }>,
  sessionId: string,
): { note: string; tempPaths: string[] } {
  const tempPaths: string[] = [];
  const notes: string[] = [];
  for (let i = 0; i < images.length; i++) {
    const img = images[i];
    const mimeType = img.source?.media_type;
    const data = img.source?.data;
    let tempPath: string | null = null;
    if (data) {
      try {
        const dir = fs.mkdtempSync(joinPath(os.tmpdir(), `claude-acp-img-${sessionId.slice(0, 8)}-`));
        const ext = MIME_TO_EXT[mimeType] ?? (mimeType?.split("/")[1] || "png");
        tempPath = joinPath(dir, `image-${i + 1}.${ext}`);
        fs.writeFileSync(tempPath, Buffer.from(data, "base64"));
        tempPaths.push(tempPath);
      } catch (err) {
        logger.warn(`Failed to persist attached image to temp file: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
    const pathNote = tempPath ? `Saved to: ${tempPath}` : "(could not persist to disk)";
    notes.push(
      `[Attached image #${i + 1} (${mimeType || "unknown"}). The current model does NOT support image input, so you cannot see it. ${pathNote}. If you need to analyze it, ask the user to switch to a vision-capable model and re-attach, or use a tool that routes to one.]`,
    );
  }
  return { note: notes.join("\n"), tempPaths };
}

// ─── Dynamic proxy model picker list ──────────────────────────────────────
// By default the ACP model picker advertises the static list from
// ~/.claude/settings.json (CLAUDE_ACP_AVAILABLE_MODELS), which can drift from
// the live proxy catalog (e.g. gpt-oss-120b / Qwen3.6-27B hardcoded while the
// proxy deprecates them, and healthy multimodal models like Qwen3.8-27B-FP8-1M
// never added). Setting CLAUDE_ACP_PROXY_MODELS=1 flips the picker to the live
// proxy catalog (/v1/models + /v1/model-health) filtered through the SAME gate
// the codex catalog and ericli pickers use (healthy + non-Deprecated +
// coding/near_sota tag), so all four agents agree on the picker. When the
// proxy list is unavailable or empty, the picker falls back to the static
// settings list so a proxy outage never empties the dropdown.
const PROXY_MODELS_ENV = "CLAUDE_ACP_PROXY_MODELS";

let proxyPickerModels: Array<{ id: string; name: string }> | null = null;
let proxyPickerModelsPromise: Promise<void> | null = null;

/**
 * Epoch ms of the last successful proxy picker refresh. Models can become
 * healthy (scaled back up) after a session starts (e.g. glm-5.2), so a
 * one-shot load at startup can leave a newly-healthy model out of the picker
 * until the whole process restarts. A TTL lets subsequent refresh calls re-fetch
 * so the picker catches up to the backend without hammering /v1/models.
 */
let lastProxyPickerModelsLoad = 0;
const PROXY_PICKER_MODELS_TTL_MS = 30_000;

/**
 * Reset the module-level proxy picker cache. Used by tests so each case starts
 * from a clean slate (the TTL would otherwise leak one test's successful load
 * into the next within the same process).
 */
export function resetProxyPickerModelsCache(): void {
  proxyPickerModels = null;
  proxyPickerModelsPromise = null;
  lastProxyPickerModelsLoad = 0;
}

/**
 * Whether the dynamic proxy picker is enabled.
 *
 * Reads through `getClaudeEnv` so the flag works both when exported into the
 * ACP parent process environment AND when set in the `env` block of
 * ~/.claude/settings.json (which only reaches the Claude CLI subprocess; the
 * ACP parent process never sees it in process.env).
 */
function isProxyModelsEnabled(): boolean {
  return getClaudeEnv(PROXY_MODELS_ENV) === "1";
}

/**
 * Fetch the live proxy catalog and filter it through the shared picker gate.
 * Result is cached in `proxyPickerModels`; a failed fetch leaves it null so
 * the next call retries. Throws when disabled (CLAUDE_ACP_PROXY_MODELS != "1")
 * so callers can skip gracefully.
 */
export async function loadProxyPickerModels(): Promise<void> {
  if (!isProxyModelsEnabled()) return; // opt-in (tests stay hermetic)
  // Within the TTL the cached list is still considered fresh, even if a
  // prior fetch succeeded — avoid re-polling the proxy on every session op.
  if (proxyPickerModels && Date.now() - lastProxyPickerModelsLoad < PROXY_PICKER_MODELS_TTL_MS) {
    return;
  }
  if (proxyPickerModelsPromise) return proxyPickerModelsPromise;
  proxyPickerModelsPromise = (async () => {
    try {
      const base = getProxyBaseUrl();
      const [modelsResp, healthResp] = await Promise.all([
        fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(5000) }),
        fetch(`${base}/v1/model-health`, { signal: AbortSignal.timeout(5000) }),
      ]);
      let models: ProxyModelEntry[] = [];
      if (modelsResp.ok) {
        const body = (await modelsResp.json()) as { data?: ProxyModelEntry[] };
        models = body.data ?? [];
      }
      let health: ProxyModelHealthEntry[] = [];
      let healthOk = false;
      if (healthResp.ok) {
        const body = (await healthResp.json()) as { data?: ProxyModelHealthEntry[] };
        health = body.data ?? [];
        healthOk = true;
      }
      // When model-health returned no data, treat every catalog model as
      // healthy so a health-endpoint error doesn't ghost the picker (mirrors
      // ericli's `healthyById.size === 0` fallback). The tag gate still applies.
      const ids = filterPickerModels(models, health, !healthOk || health.length === 0);
      logger.info(
        `Loaded ${ids.length} picker models from proxy` +
          (isProxyModelsEnabled() ? " (dynamic)" : ""),
      );
      proxyPickerModels = ids.map((id) => ({ id, name: id }));
      lastProxyPickerModelsLoad = Date.now();
    } catch (err) {
      proxyPickerModels = null;
      logger.warn(
        `Could not load picker models from proxy: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      proxyPickerModelsPromise = null;
    }
  })();
  return proxyPickerModelsPromise;
}

/**
 * The models to advertise in the ACP picker.
 *
 * When CLAUDE_ACP_PROXY_MODELS=1 and the proxy list has loaded, that live
 * filtered list wins. Otherwise (flag off, still loading, or proxy
 * unreachable) fall back to the static settings list — the pre-existing
 * behaviour — so an outage never empties the picker.
 */
function getPickerModelList(): Array<{ id: string; name: string }> {
  // The length check matters: an empty proxy list (fetched fine, but the
  // gate filtered every model — e.g. all scaled to zero) is truthy in JS and
  // would otherwise win over the settings list, crashing buildConfigOptions
  // at `availableModels[0].id`.
  if (isProxyModelsEnabled() && proxyPickerModels && proxyPickerModels.length > 0) {
    return proxyPickerModels;
  }
  return getAvailableModelsFromSettings();
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
 * Interaction with the override:
 *  - When the model's real window is UNKNOWN, the override is used verbatim
 *    (it's the only signal we have).
 *  - When the model's real window is SMALLER than the override, the real
 *    window wins — the override is a documented *maximum*, so applying a
 *    196608 override to glm-5.1 (131072) must clamp down to 131072.
 *    Without this, the threshold would sit above the real limit and the
 *    conversation would overflow without ever compacting.
 *  - When the model's real window is LARGER than the override, the real
 *    window wins — the override was written for a different (smaller)
 *    model and must not hide capacity from a larger one. Otherwise
 *    deepseek-v4-flash (262144) would display as 196608 and compact far
 *    earlier than necessary.
 *
 * Net effect: the real window always wins when known; the override only
 * applies when no real window is available.
 */
export function getContextWindowForModel(modelId: string | undefined): number {
  const proxyKnown = modelId ? modelContextWindows.get(modelId) : undefined;
  const cliKnown = modelId ? cliReportedContextWindows.get(modelId) : undefined;
  const known = proxyKnown ?? cliKnown;
  if (known !== undefined) return known;
  // The model is in neither map. If the proxy load hasn't completed (or failed
  // at spawn and is now retryable), kick off a background refresh so the NEXT
  // lookup sees the real window instead of falling through to the override —
  // which is typically pinned to one specific model's window and would
  // mis-report every other model's denominator and compaction threshold. This
  // is fire-and-forget: the current call still falls back below; the refresh
  // populates the map for subsequent calls (the next usage_update, which fires
  // many times per turn).
  if (modelId && !modelContextWindowsLoaded) {
    void loadModelContextWindows();
  }
  const override = parsePositiveIntEnv(getClaudeEnv("CLAUDE_CODE_MAX_CONTEXT_TOKENS"));
  if (override !== undefined) return override;
  return DEFAULT_CONTEXT_WINDOW;
}

// ─── Claude Code CLI auto-compact arithmetic ────────────────────────────────
// These mirror the CLI's own constants (services/compact/autoCompact.ts) so the
// ACP layer computes the same threshold the CLI does. Keeping the two in sync
// matters because both can compact: if they disagree, whichever fires first
// wins and the other becomes unreachable, making the configured behaviour a
// coin flip that depends on undocumented internals.

/** Headroom kept below the effective window before compacting. */
const AUTOCOMPACT_BUFFER_TOKENS = 13_000;
/** The CLI's max-output default for model ids it does not recognize. */
const DEFAULT_MAX_OUTPUT_TOKENS = 32_000;

// ─── Replay prompt framing strip ──────────────────────────────────────────
// The steer drain and the goal turn wrap the user's text in framing prefixes
// before sending it to the CLI: "[Follow-up instruction from user]\n…" and
// "[Goal: …]\n\n…" (plus the goal preamble "Please start working toward this
// goal now:\n\n…"). The Claude CLI persists the *wrapped* text into the JSONL
// transcript, so on resume/loadSession the wrapper — the framing, the goal
// annotation, and nested combinations — is replayed into the chat panel as if
// the user had typed it, duplicating the same line once per steered round.
// It also accumulates: a steer of a steer stacks wrappers. Strip the framing
// on replay so the user's own words come back clean. Iterated to a fixed
// point because the wrappers nest.
const REPLAY_PROMPT_FRAMING_RE =
  /^(?:\[Follow-up instruction from user\]\n|\[Goal:[^\]]*\]\n\n|Please start working toward this goal now:\n\n)+/;

// The <collaboration_mode> developer block is prepended to effectivePrompt in
// executePrompt (plan-mode instruction injection) and is therefore SENT to the
// Claude CLI, which persists the wrapped text into the JSONL transcript as a
// `type:"user"` entry. On resume/loadSession that block would otherwise replay
// into the chat panel as a user bubble — the user's own message arrives with a
// wall of "Collaboration Mode: Default/Plan … EnterPlanMode …" instructions
// they never typed. It is authoritative for the CLI's context, so it must NOT
// be stripped from what is sent to the model — only from the REPLAYED display.
// Same wrap-sent-to-CLI / strip-on-replay pattern as the framing above.
//
// Non-greedy up to the FIRST closing tag: the user's text (and the memory /
// persona / skill blocks that follow it in the transcript) never contains a
// well-formed <collaboration_mode>…</collaboration_mode> pair, so the first
// closer delimits the injected block. Written with escaped angle brackets
// (backslash + angle) so the tag is a plain regex literal — not a character
// class.
const REPLAY_COLLAB_MODE_BLOCK_RE =
  /^\<collaboration_mode>[\s\S]*?\<\/collaboration_mode>\n*/i;

export function stripReplayPromptFraming(text: string): string {
  let prev: string;
  let cur = text;
  do {
    prev = cur;
    cur = cur.replace(REPLAY_PROMPT_FRAMING_RE, "");
    // Strip a leading <collaboration_mode> block, then re-apply the framing
    // strip — the mode block and the steer/goal framing can nest (a steered
    // prompt carries BOTH: <collaboration_mode>…</collaboration_mode>\n\n
    // [Follow-up instruction from user]…). Iterate both to a fixed point.
    cur = cur.replace(REPLAY_COLLAB_MODE_BLOCK_RE, "");
  } while (cur !== prev);
  return cur;
}

/**
 * Scan Claude CLI's on-disk transcripts for a working directory and surface
 * them as SessionInfo entries — the same way kiro-cli enumerates
 * `~/.kiro/sessions/cli/*.json`. The Claude CLI spawns a fresh process per
 * turn and keeps no app-server RPC (unlike `codex app-server`), so the only
 * way to list CLI-started sessions is to read the transcript files it writes
 * at `~/.claude/projects/<encoded-cwd>/<claude-session-id>.jsonl`.
 *
 * Without this, `listSessions` returned only ACP-created sessions from
 * sessions.json; sessions started from the `claude` CLI in the same workspace
 * were invisible in the ACP session list even though `loadSession` could
 * already replay their transcripts.
 */
export async function listCliTranscriptSessions(
  cwd: string | undefined
): Promise<SessionInfo[]> {
  if (!cwd) return [];
  const { homedir } = await import("node:os");
  const fs = await import("node:fs");
  const projectDir = `${homedir()}/.claude/projects/${cwd.replace(/\//g, "-")}`;
  // existsSync is present in every fs mock; the heavier readdir/stat/read
  // helpers are accessed lazily below so partial fs mocks (used across the
  // test suite) don't break on missing exports when no project dir exists.
  if (!fs.existsSync(projectDir)) return [];

  let files: string[];
  try {
    files = (fs.readdirSync ?? (() => [] as string[]))(projectDir).filter(
      (f: string) => f.endsWith(".jsonl")
    );
  } catch {
    return [];
  }

  const results: SessionInfo[] = [];
  for (const file of files) {
    const filePath = `${projectDir}/${file}`;
    try {
      const raw = (fs.readFileSync as (p: string, enc: string) => string)(
        filePath,
        "utf-8"
      );
      const lines = raw.split("\n").filter((l) => l.trim());
      if (lines.length === 0) continue;

      let title: string | undefined;
      let updatedAt: string | undefined;
      // First pass: prefer a `summary` entry for the title (Claude writes
      // these as a short label), and take the last entry's timestamp as
      // updatedAt — mirroring how the store orders sessions.
      for (const line of lines) {
        let entry: any;
        try {
          entry = JSON.parse(line);
        } catch {
          continue;
        }
        if (entry.timestamp) updatedAt = entry.timestamp;
        if (!title && entry.type === "summary" && entry.summary) {
          title = String(entry.summary);
        }
      }
      // Fall back to the first user message text when no summary exists.
      if (!title) {
        for (const line of lines) {
          let entry: any;
          try {
            entry = JSON.parse(line);
          } catch {
            continue;
          }
          if (entry.type === "user" && entry.message) {
            const content = entry.message.content;
            if (typeof content === "string" && content.trim()) {
              title = content.trim().slice(0, 80);
              break;
            } else if (Array.isArray(content)) {
              const text = content
                .find((b: any) => b?.type === "text" && b.text)?.text;
              if (text) {
                title = text.trim().slice(0, 80);
                break;
              }
            }
          }
        }
      }
      if (!updatedAt) {
        const stat = fs.statSync as ((p: string) => { mtimeMs: number }) | undefined;
        updatedAt = stat
          ? new Date(stat(filePath).mtimeMs).toISOString()
          : undefined;
      }

      const claudeSessionId = file.replace(/\.jsonl$/, "");
      results.push({
        sessionId: claudeSessionId,
        cwd,
        title: title ?? "Claude CLI session",
        updatedAt,
      });
    } catch {
      // Skip unreadable / corrupted transcripts.
      continue;
    }
  }
  return results;
}

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
 * Effective window used for compaction maths.
 *
 * Deliberately NOT reduced by the summary output reserve: we compact as late
 * as possible (matching codex-acp's ~90%-of-window behavior) instead of
 * reserving `min(maxOutput, 20000)` tokens up front. Trade-off (accepted):
 * a turn whose prompt sits at the threshold AND whose response uses the full
 * maxOutput budget can push `prompt + max_tokens` past the real window, which
 * the backend rejects. The reserve existed to guarantee `threshold + maxOutput
 * <= window`; removing it trades that safety for later compaction.
 */
export function getEffectiveContextWindowSize(modelId: string | undefined): number {
  return getAutoCompactWindow(modelId);
}

/**
 * Auto-compact threshold.
 *
 *   threshold = effectiveWindow - 13000
 *   with CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=N:
 *     threshold = min(floor(effectiveWindow * N/100), threshold)
 *
 * `effectiveWindow` is the (possibly capped) model window with NO summary
 * reserve subtracted (see getEffectiveContextWindowSize), so the threshold
 * sits as high as possible. The percentage applies to the effective window
 * and can only lower the threshold further — values above the default
 * (N >= ~93) have no effect because floor(effective*N/100) exceeds
 * effective - 13000.
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

/**
 * Re-emit a usage_update when the session's model is switched mid-conversation.
 *
 * The context ring's `size` (denominator) is only ever set by a usage_update
 * frame, and those fire inside the per-turn result-event handler. A model
 * switch (setSessionConfigOption with configId "model") does NOT start a turn,
 * so without this the ring keeps the OLD model's window — e.g. the user picks
 * Qwen3.8-27B-FP8-1M but the ring still reads 262144 (Qwen3.8-27B's window)
 * until the next prompt completes.
 *
 * Carries over the last known used tokens (sessionLastUsage) so the ring does
 * not falsely reset to 0%; when no turn has run yet there is nothing to show,
 * so we emit nothing (the ring stays hidden until the first real usage).
 * The new size comes from getContextWindowForModel (proxy-authoritative), not
 * a re-estimate from the model name.
 */
export function reEmitUsageOnModelSwitch(
  sessionId: string,
  connection: {
    sessionUpdate: (p: {
      sessionId: string;
      update: { sessionUpdate: "usage_update"; used: number; size: number };
    }) => Promise<unknown>;
  },
  _store: SessionStore,
  prevModel: string | undefined,
  newModel: string,
): void {
  const newWindow = getContextWindowForModel(newModel);
  const last = sessionLastUsage.get(sessionId);
  const used = last ? Math.min(last.input_tokens, newWindow) : 0;
  if (used <= 0) return;
  void connection
    .sessionUpdate({
      sessionId,
      update: { sessionUpdate: "usage_update" as const, used, size: newWindow },
    })
    .catch(() => {
      // Best effort — a failed usage_update must not break the model switch.
    });
  logger.info(
    `Session ${sessionId} model ${prevModel} → ${newModel}: re-emitted usage_update size=${newWindow} used=${used}`,
  );
}

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

  /** Characters accumulated since the last real per-call count (for logging). */
  getPendingChars(): number {
    return this.bytesSinceReal;
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

  /**
   * Context size INCLUDING content accumulated since the last real per-call
   * count (the new user prompt + any trailing tool results added between API
   * calls).
   *
   * `getEstimatedTokens` returns `realTokens` verbatim and ignores
   * `bytesSinceReal` — correct mid-turn (the real count refreshes on every
   * API call) but stale for the pre-turn compaction check. Between turns the
   * next prompt's text has already been `addText`-ed into `bytesSinceReal`,
   * so the verbatim figure under-counts what vLLM is about to receive by
   * exactly the new prompt's token contribution. That gap is what let a turn
   * sit at 174,239 (under the 176,947 threshold) yet arrive at vLLM as
   * 176,609 input + 20,000 maxOutput → 1-token overflow.
   *
   * Used ONLY by the pre-turn compaction guard, not the live usage UI (the
   * char-based increment over-inflates when subagent/tool traffic is in
   * `bytesSinceReal` — see the getEstimatedTokens comment).
   */
  getEstimatedTokensWithPending(): number {
    if (this.realTokens > 0) {
      return this.realTokens + estimateTokensFromChars(this.bytesSinceReal);
    }
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
  // Windows paths use backslashes — split on both separators or the whole
  // path becomes the "basename" and renders as a truncated full path.
  const base = p.split(/[\\/]/).pop() || p;
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
 * Relative path when `target` is a directory INSIDE `cwd`, else null.
 *
 * `cwd` itself returns null — running in the session directory is the default.
 * A relative target resolves against `cwd`, never process.cwd(), so applying
 * this to an already-normalized command is a no-op.
 */
function subdirOfCwd(target: string, cwd?: string): string | null {
  if (!cwd) {
    return null;
  }
  try {
    const base = resolvePath(cwd);
    const abs = isAbsolutePath(target) ? resolvePath(target) : resolvePath(base, target);
    const rel = relativePath(base, abs);
    if (!rel || rel === "." || rel.startsWith("..") || isAbsolutePath(rel)) {
      return null;
    }
    return rel;
  } catch {
    return null;
  }
}

function isSameDir(a: string, b: string): boolean {
  const normalize = (p: string) => p.replace(/\/+$/, "") || "/";
  return normalize(resolvePath(a)) === normalize(resolvePath(b));
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
export function splitCdPrefix(command: string, cwd?: string): { command: string; cwd?: string } {
  const nl = command.indexOf("\n");
  const firstLine = (nl === -1 ? command : command.slice(0, nl)).trim();
  const rest = nl === -1 ? "" : command.slice(nl + 1);

  // `cd <dir> && <cmd>` / `cd <dir>; <cmd>` — cd and the command share the
  // first line. Claude Code emits the `&&` form; other agents use `;`.
  const cdAnd = firstLine.match(/^cd\s+(\S[^;&]*?)\s*(&&|;)\s*(.+)$/);
  if (cdAnd) {
    const target = cdAnd[1].trim();
    const inline = cdAnd[3].trim();
    const remainder = rest ? `${inline}\n${rest}` : inline;
    const sub = subdirOfCwd(target, cwd);
    if (sub) {
      // Inside the session cwd: the location carries information, so keep it in
      // the title — but relative, not as a long absolute path.
      const joined = `cd ${sub}${cdAnd[2] === ";" ? ";" : " &&"} ${inline}`;
      return { command: rest ? `${joined}\n${rest}` : joined, cwd: target };
    }
    if (cwd && isSameDir(target, cwd)) {
      // The session already runs there — drop the cd entirely and do not report
      // a redundant cwd.
      return { command: remainder };
    }
    return { command: remainder, cwd: target };
  }

  // Bare `cd <dir>` on its own line — the command starts on the next line.
  const cdOnly = firstLine.match(/^cd\s+(\S.*)$/);
  if (cdOnly && rest.trim()) {
    const target = cdOnly[1].trim();
    const body = rest.replace(/^\n+/, "");
    // The cd is always removed from the newline form — clients render only the
    // first line, so keeping it would hide the command. The directory still
    // travels out-of-band in rawInput.cwd (except when it IS the session cwd).
    if (cwd && isSameDir(target, cwd)) {
      return { command: body };
    }
    return { command: body, cwd: target };
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
function titleForCommand(command: string, description?: string, cwd?: string): string {
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
    // `cd <dir> && <rest>` / `cd <dir>; <rest>` on a single line.
    //  - target is the session cwd      → drop the cd, keep <rest>
    //  - target is INSIDE the cwd       → keep it, rewritten relative
    //  - target is anywhere else        → drop the cd, keep <rest>
    const cdAnd = t.match(/^cd\s+(\S[^;&]*?)\s*(&&|;)\s*(.*)$/);
    if (cdAnd) {
      const sub = subdirOfCwd(cdAnd[1].trim(), cwd);
      if (sub) {
        const joined = `cd ${sub}${cdAnd[2] === ";" ? ";" : " &&"} ${cdAnd[3].trim()}`.trim();
        return joined.length <= 120 ? joined : joined.slice(0, 117) + "…";
      }
      t = cdAnd[3].trim();
      if (!t) continue; // `cd <dir> && ` with nothing after — next line.
    } else if (/^cd\s+(\S.*)$/.test(t)) {
      const cdOnly = t.match(/^cd\s+(\S.*)$/);
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
  cwd?: string,
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
      const split = splitCdPrefix(command, cwd);
      const rawInput: Record<string, unknown> = {
        ...(typeof toolInput === "object" && toolInput !== null ? input : {}),
        command: split.command,
      };
      if (split.cwd !== undefined) rawInput.cwd = split.cwd;
      return {
        sessionUpdate: "tool_call",
        toolCallId,
        title: titleForCommand(command, description, cwd),
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

/** Read-only /goal subcommands that need no LLM turn (matches the exactArgs
 *  declared in BUILTIN_COMMANDS for /goal's turnSafe meta). */
const GOAL_READ_ONLY_SUBS = new Set([
  "", "clear", "stop", "off", "reset", "none", "cancel", "pause", "resume",
]);

/**
 * Native Claude CLI slash commands that the CLI services itself as a `local`
 * command WITH a non-interactive (headless / `claude -p`) text result — no LLM
 * turn, no API call. The CLI runs the command and returns the output as the
 * `result` field of a single `result` event with `stop_reason: null` (see
 * processSlashCommand's `case 'local'` → `{ shouldQuery: false, resultText }`,
 * then QueryEngine yielding `{ type: 'result', result: resultText,
 * stop_reason: null }`).
 *
 * Verified against CLI v2.1.88 source (claude-code-source-code):
 *   - /heapdump  — type 'local', supportsNonInteractive: true, isHidden: true.
 *                  Writes the JS heap to ~/Desktop and returns the path(s) as
 *                  text. Headless text result → surfaced here.
 *   - /context   — has TWO definitions: an interactive 'local-jsx' colored-grid
 *                  modal AND a 'local' non-interactive variant
 *                  (supportsNonInteractive: true) that returns a markdown token
 *                  breakdown. In a headless ACP session only the 'local'
 *                  variant is enabled (isEnabled: getIsNonInteractiveSession),
 *                  so it produces headless text → surfaced here.
 *
 * Deliberately NOT in this set (verified against CLI source):
 *   - /usage     — type 'local-jsx', availability: ['claude-ai']. It renders a
 *                  React <Settings> Usage modal — a VISUAL screen, not text. In
 *                  a non-interactive session processSlashCommand's
 *                  'local-jsx' case hits `if (isNonInteractiveSession) →
 *                  resolve({ messages: [], shouldQuery: false })` and emits NO
 *                  result text. There is nothing to surface, and it is not
 *                  available to API-key (proxy) users anyway.
 *   - /insights  — type 'prompt'. It CALLS THE MODEL to analyze sessions and
 *                  write an HTML report; it is not a 'local' command and does
 *                  not return via the result field with stop_reason null. It
 *                  runs as a normal turn.
 *   - /recap     — not present in CLI 2.1.88.
 *
 * These reach the adapter through the SAME queued-and-drained path as any
 * prompt (they are NOT in isReadOnlyCommand, so vscode-acp does not fire them
 * mid-turn as a side-channel; they run serially and are drained like a normal
 * turn). The adapter used to DROP their output (the `result` handler ignored
 * `event.text`), so /heapdump and /context showed nothing in the ACP chat even
 * though the CLI had produced a result. This set names the natives whose
 * result text is surfaced to the client as a transient commandOutput row when
 * it arrives — giving the CLI's own (precise) data without fabricating it on
 * the adapter side.
 *
 * Kept as an explicit allowlist (not "any result with no stop reason") so a
 * genuine end_turn that happens to carry no text is never mis-rendered, and a
 * future native command can be added deliberately.
 */
const CLI_LOCAL_COMMAND_NAMES = new Set(["heapdump", "context"]);

/** True when the prompt being forwarded to the CLI is a native `local`
 *  slash command (e.g. /heapdump, /context) that the CLI services itself with
 *  a headless text result. Used to recognize its result event so the CLI's
 *  output can be surfaced as a commandOutput row instead of being dropped. */
function isCliLocalCommandPrompt(text: string): boolean {
  const cmd = parseSlashCommand(text);
  return !!cmd && CLI_LOCAL_COMMAND_NAMES.has(cmd.name);
}

/** Check if a slash command is read-only (no LLM turn needed). Used both for
 *  the per-session prompt-lock bypass (read-only commands run concurrently
 *  with an in-flight turn) and for dispatchSlashCommand's fast path. */
function isReadOnlyCommand(name: string, rest?: string): boolean {
  if (name === "ps" || name === "status" || name === "skills" || name === "list-agents") {
    return true;
  }
  if (name === "goal" && GOAL_READ_ONLY_SUBS.has((rest ?? "").trim().toLowerCase())) {
    return true;
  }
  return false;
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
  // Dynamic proxy-filtered list when enabled, else the static settings list.
  const availableModels = getPickerModelList();
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

export interface ClaudeCodeAgentOptions {
  /**
   * How long to wait (ms) at turn end for pending subagent
   * <task-notification> messages to arrive before marking them completed as
   * a fallback. Real Claude awaits subagents before ending the turn;
   * GLM/EricAI ends the turn early and the notifications land within 1-5s.
   * Overridable so tests can avoid the full wait.
   * @default 15000
   */
  pendingAgentWaitMs?: number;
}

export function createClaudeCodeAgent(
  connection: AgentSideConnection,
  options: ClaudeCodeAgentOptions = {}
): Agent {
  const pendingAgentWaitMs = options.pendingAgentWaitMs ?? 15000;
  const store = new SessionStore();
  const runner = new ClaudeRunner();
  const cancelledSessions = new Set<string>();
  const steerQueues = new Map<string, string[]>();
  // Per-session prompt serialization + cancel scoping.
  //
  // The bug: cancel() latches cancelledSessions for the in-flight prompt, but
  // prompts serialize per session (sessionPromptLocks), so a NEW prompt queued
  // behind the cancelled one reaches executePrompt right after the in-flight one
  // settles — while the latch is still set — and was wrongly cancelled before it
  // started ("stop mid-stream, then the queued 'Send Now' message vanishes
  // instead of running as its own turn").
  //
  // Fix: tag each prompt with a monotonically-increasing generation
  // (promptGenCounter, assigned at queue time). executePrompt records the
  // generation it's running as the in-flight one (inFlightGen). cancel() latches
  // the session AND records the in-flight generation as the cancel's target
  // (cancelledGen). A prompt is treated as "cancelled before start" only if the
  // cancel's target generation matches its OWN, so a fresh post-cancel prompt
  // (a later generation) is not hit by the stale latch.
  const promptGenCounter = new Map<string, number>();
  // sessionId -> generation of the prompt currently executing.
  const inFlightGen = new Map<string, number>();
  // sessionId -> the generation a cancel was aimed at (the in-flight prompt).
  const cancelledGen = new Map<string, number>();
  /**
   * True if the session's current cancel latch targets `gen` — i.e., a cancel()
   * was aimed at the prompt with this generation AND the session is still
   * latched. A prompt should only treat itself as cancelled when the cancel
   * was aimed at ITS generation; a stale latch from an earlier (already-settled)
   * prompt must not cancel a fresh prompt that merely shares the session.
   */
  const isCancelledForGen = (sessionId: string, gen: number): boolean =>
    cancelledSessions.has(sessionId) && cancelledGen.get(sessionId) === gen;
  // Per-session auto-follow-up counter. When a turn ends with subagent(s)
  // still pending (premature end_turn), we enqueue a synthetic steer that
  // starts a new turn so the Claude CLI injects the pending
  // <task-notification> results. This is bounded per session to prevent an
  // infinite cascade (an auto-follow-up turn that itself spawns subagents
  // could re-trigger). See `ericli-subagent-late-arrival-turn-trigger`.
  const autoFollowUpCount = new Map<string, number>();
  const AUTO_FOLLOW_UP_CAP = 3;
  // Sessions whose steer queue is currently being drained. drainSteerQueue()
  // awaits LLM turns, so without this a second call (pre-turn vs post-turn vs
  // post-compaction) could run concurrently and interleave two consumers over
  // the same queue, delivering messages out of order.
  const steerDraining = new Set<string>();

  // Background drain promises started by executePrompt's post-turn
  // non-blocking drainSteerQueue(). The next prompt() chains on these so it
  // waits for the drain's LLM turns to finish before starting its own turn —
  // otherwise a new prompt would race with the drain's continueSessionStreaming
  // (same onEvent closure, same --resume session id). Cleared on settle.
  const pendingDrainPromises = new Map<string, Promise<unknown>>();

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

  // True for the entire duration of a executePrompt() call — from entry through
  // the LLM turn, steer drain, auto-compact, and the end_turn return. Unlike
  // runner.isRunning() (which flips false the moment the Claude subprocess
  // closes, before the drain runs) this spans the whole turn tail. Used by
  // handleSteer() to decide whether a steer should be queued vs. dropped: a
  // steer typed in the last moments of streaming arrives after the subprocess
  // closes but while a turn is still in flight, and must be queued (then
  // drained at line ~2056) rather than silently dropped. Cleared in a finally
  // so it can't get stuck true if executePrompt throws.
  const sessionPromptInFlight = new Set<string>();

  // ─── Subagent (Agent tool) lifecycle tracking ─────────────────────────────
  // The Agent tool spawns a subagent that runs ASYNCHRONOUSLY: the tool_result
  // it emits immediately ("Async agent launched successfully...") is a launch
  // acknowledgment, NOT a completion signal. The subagent keeps running for
  // seconds to minutes afterwards.
  //
  // Completion is signaled by a <task-notification> that Claude injects back
  // into the main conversation as a `user` text message once the subagent
  // finishes (or is stopped). The notification carries a task-id (NOT the
  // tool_use_id) and a <summary> that names the agent by its description
  // (e.g. `Agent "Trace interrupt..." completed` / `background agent
  // "Trace interrupt..."`). We link spawn→completion by description, since
  // the tool_use_id is never reused by the notification.
  //
  // We emit status:"in_progress" at the tool_use, flip to "completed" when the
  // matching <task-notification> arrives mid-turn, and fall back to a turn-end
  // flush for any still pending (the premature-end_turn case — see below).
  // Maps Claude's native tool_use id → { acpCallId, description }.
  const sessionPendingAgents = new Map<
    string,
    Map<string, { acpCallId: string; description: string }>
  >();

  // Non-subagent tools (Bash/Read/Edit/Grep/…) use the same in_progress →
  // completed lifecycle, but resolve on the tool_result message (not a
  // <task-notification>). The runner collects the tool_use_ids a user message
  // resolves (claude-runner.ts `toolResultIds`); we flip each matching pending
  // tool to "completed" when that user event arrives. Without this, every tool
  // is emitted as status:"completed" at the tool_use and the chat UI never
  // shows a running phase — the command text appears only after the tool ends.
  // Maps sessionId → (Claude tool_use_id → ACP toolCallId).
  const sessionPendingTools = new Map<string, Map<string, string>>();

  // Parallel to sessionPendingTools: maps toolUseId → toolName so the
  // permission_request handler can correlate by name (the permission event
  // carries toolName but NOT the original tool_use's toolUseId).
  const sessionPendingToolNames = new Map<string, Map<string, string>>();

  // Background post-turn drain promises. executePrompt returns its JSON-RPC
  // response immediately after the LLM turn; the drain's follow-up turns run
  // in the background. The next prompt() chains on this so it waits for the
  // drain's LLM turns before starting.
  const pendingDrainPromises = new Map<string, Promise<void>>();

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
  // Optional receiver for the plan snapshot handleTaskTool builds. The live
  // path leaves this undefined (handleTaskTool emits nothing itself); the
  // session/load replay path passes one so the snapshot is buffered with the
  // rest of the replayed updates instead of emitted inline.
  function handleTaskTool(
    sessionId: string,
    toolName: string,
    toolInput: unknown,
    out?: { snapshot?: any },
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
    // Where the built plan snapshot goes. When `out` is provided (replay path),
    // write it there for the caller to buffer; otherwise emit it directly on
    // the connection (live path).
    const deliver = (snapshot: any) => {
      if (!snapshot) return;
      if (out) {
        out.snapshot = snapshot;
      } else {
        connection.sessionUpdate({ sessionId, update: snapshot } as any).catch(() => {
          // Best effort.
        });
      }
    };

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
      let deduped = 0;
      for (const t of tasks) {
        const content =
          (typeof t.subject === "string" && t.subject) ||
          (typeof t.description === "string" && t.description) ||
          (typeof t.activeForm === "string" && t.activeForm) ||
          "task";
        // Dedup by content: the model (notably GLM-5.2) sometimes re-emits a
        // TaskCreate for a task it already created earlier in the session
        // (re-planning). Appending a second `task-N` with the same content
        // produced duplicate plan entries that never cleared. If an entry with
        // the same content already exists, skip it rather than adding a dupe.
        let alreadyTracked = false;
        for (const existing of todos.values()) {
          if (existing.content === content) {
            alreadyTracked = true;
            break;
          }
        }
        if (alreadyTracked) {
          deduped++;
          continue;
        }
        // Assign a synthetic id in creation order. Claude's real task ids
        // only surface in the tool_result (not the tool_use input), so we
        // can't key on them; we match TaskUpdate by derived id/subject below.
        const id = `task-${todos.size + 1}`;
        todos.set(id, { content, status: "pending" });
      }
      logger.debug(
        `TaskCreate: +${tasks.length - deduped} task(s)${deduped ? ` (${deduped} dup skipped)` : ""}, list now ${todos.size}`,
      );
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
      // Final fallback: a status-only TaskUpdate (taskId + status, no subject)
      // whose taskId is a Claude-internal id we never tracked. The model emits
      // these to mark the task it just finished as completed/in_progress. When
      // there is exactly ONE task currently in_progress, that is almost
      // certainly the one being updated (the active task transitions to
      // completed when done; a pending task flips to in_progress when started).
      // Match it so the plan panel reflects the transition instead of leaving
      // every entry stuck at "pending". If there are zero or multiple
      // in_progress tasks, don't guess — leave the list unchanged.
      if (!matchedId && input.status !== undefined && !subject) {
        let inProgressId: string | null = null;
        let inProgressCount = 0;
        for (const [id, t] of todos) {
          if (t.status === "in_progress") {
            inProgressId = id;
            inProgressCount++;
          }
        }
        if (inProgressCount === 1 && inProgressId) {
          matchedId = inProgressId;
          logger.debug(
            `TaskUpdate: taskId=${rawTaskId} unmatched by id/subject; ` +
              `applying status=${status} to the sole in_progress task`,
          );
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

    // Build a full plan snapshot — but ONLY when there is at least one tracked
    // todo. An empty snapshot would render as "Plan (no content)" in the UI,
    // which happens when the model calls TaskUpdate on tasks it never created
    // (e.g. GLM-5.2 replaying task ids it saw in memory/prior context across a
    // new session where sessionTodos is empty). In that case there is genuinely
    // nothing to plan, so we return null (no emission) rather than flash an
    // empty bar. Returning the snapshot (instead of emitting inline) lets the
    // caller decide delivery: the live path emits immediately, the session/load
    // replay path buffers it with the rest of the replayed updates.
    if (todos.size === 0) {
      return true;
    }
    const entries = Array.from(todos.entries()).map(([_id, t]) => ({
      content: t.content,
      priority: "medium" as const,
      status: t.status as "pending" | "in_progress" | "completed",
    }));
    const snapshot = {
      sessionUpdate: "plan",
      entries,
      // These come from Claude Code's TaskCreate/TaskUpdate/TaskList tools — a
      // TODO list, as the note at the top of this section says. ACP has one
      // channel for both a plan and a task list, and the client titles the bar
      // "Plan" by default, so a checklist read as a strategy.
      //
      // Adapter-neutral hint, shared with the other adapters: a client that ignores
      // _meta renders exactly as before.
      _meta: { acp: { plan: { kind: "todo", title: "Todos" } } },
    } as any;
    // When the caller supplies a receiver (session/load replay, or the
    // post-compaction re-push below), buffer the snapshot there instead of
    // emitting inline — the caller decides delivery timing.
    if (out) {
      out.snapshot = snapshot;
    } else {
      deliver(snapshot);
    }
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
    // streaming OR a executePrompt() is still in flight. runner.isRunning() flips
    // false the moment the subprocess closes (claude-runner.ts proc.on('close'))
    // — but executePrompt() still runs the steer drain + auto-compact + return
    // end_turn after that. Without the in-flight check, a steer typed in the
    // last moments of streaming (subprocess closed, executePrompt in its drain
    // tail) would hit this guard, log "no active turn", and be silently
    // dropped instead of queued — so it never appeared in the chat before the
    // next turn.
    if (!runner.isRunning(sessionId) && !sessionPromptInFlight.has(sessionId)) {
      logger.info(`Steer request for idle session ${sessionId} — ignoring (no active turn)`);
      return;
    }
    // The turn this steer was aimed at is being cancelled. Queuing it now would
    // reintroduce exactly what the cancel discards: the steer would sit here
    // until the NEXT turn's drain and be answered after a message the user sent
    // later. The client still holds it queued, so it drains as its own prompt.
    if (cancelledSessions.has(sessionId)) {
      logger.info(`Steer request for cancelled session ${sessionId} — ignoring (turn is stopping)`);
      return;
    }

    // Mid-turn steer: write directly into the running process's stdin. The
    // Claude CLI absorbs a second `user` line into the in-flight turn
    // (verified against the live CLI) instead of queueing it as a follow-up,
    // so the model folds this into its current work immediately.
    const midTurnWritten = runner.writeSteerToMainTurn(sessionId, text);
    if (midTurnWritten) {
      logger.info(`Steer delivered mid-turn for session ${sessionId}: ${text.length} chars`);
      // Stage 1 (visual bubble).
      await connection
        .sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "user_message_accepted",
            content: { type: "text", text },
          } as any,
        })
        .catch(() => { /* non-fatal: visual-only ack */ });
      // Stage 2 (delivered echo) — the message IS delivered, so emit the
      // delivered chunk now. This splices the client's queue entry and
      // releases the next queued message.
      await connection
        .sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "user_message_chunk",
            content: { type: "text", text },
          } as any,
        })
        .catch(() => { /* non-fatal */ });
      return;
    }

    // Fallback: no active turn (process not running). Queue the message for
    // delivery after the current turn finishes.
    if (!steerQueues.has(sessionId)) {
      steerQueues.set(sessionId, []);
    }
    steerQueues.get(sessionId)!.push(text);
    logger.info(`Steer queued for session ${sessionId}: ${text.length} chars`);
    // Stage 1 of the two-stage steer receipt (mirrors ericli's Option A):
    // emit a DISTINCT `user_message_accepted` notification so vscode-acp can
    // show the user's mid-turn bubble immediately (inserted before the running
    // turn) instead of leaving a queue chip tagged "in turn" for the whole
    // tool call.
    //
    // This is deliberately NOT a delivered-echo chunk. A plain delivered chunk
    // on receipt would be treated by the client as DELIVERED — it would splice
    // the queue entry and release the next queued message, so several
    // interjections during one long tool call all arrive at the drain at once
    // and the model answers only the last (the pile-up bug ericli hit). The
    // `user_message_accepted` type is visually-only: the client shows the
    // bubble but keeps the entry in the queue. The stage-2 delivered echo
    // (emitted in drainSteerQueue when the message is actually sent to the
    // model) is the only signal that splices the entry and releases the next
    // message — so the queue semantics stay intact.
    await connection
      .sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "user_message_accepted",
          content: { type: "text", text },
        } as any,
      })
      .catch(() => { /* non-fatal: visual-only ack */ });
  }

  /**
   * Collapse all whitespace runs to a single space and trim, for robust steer
   * matching. The client (vscode-acp) sends and echoes steer text through the
   * same collapsed rule (its `collapseWhitespaceChars` / `steerEchoMatchesEntry`),
   * so comparing collapsed forms tolerates the line-wrap / trailing-space drift
   * between what the user typed, what was queued, and what is echoed back.
   */
  const collapseSteerText = (s: string): string => s.replace(/\s+/g, " ").trim();

  /**
   * Retract a steer that the client queued (via `session/steer`) but then removed
   * from its own queue (the user deleted the queue chip). `session/steer` is a
   * notification with no retract, so without this the message would sit in
   * `steerQueues` and be delivered as a follow-up turn at the end of the current
   * one — i.e. the model would answer a message the user deleted.
   *
   * Symmetric with `handleSteer`: match on collapsed text, remove the first
   * queued entry that matches, and emit a `user_message_removed` sessionUpdate so
   * the client can drop the bubble it showed at stage-1 accept. If the matching
   * steer was already delivered to the model (drainSteerQueue ran), there is
   * nothing left in the queue to remove and this is a documented no-op — a steer
   * the model already saw cannot be unsent.
   */
  async function handleSteerRemove(params: { sessionId: string; text: string }): Promise<void> {
    const { sessionId, text } = params;
    if (!store.has(sessionId)) {
      logger.warn(`Steer-remove for unknown session ${sessionId}`);
      return;
    }
    const queue = steerQueues.get(sessionId);
    const target = collapseSteerText(text);
    if (!queue || !target) {
      // Nothing queued (or empty text) — the steer was either already drained
      // into the model or never accepted. No-op.
      logger.info(`Steer-remove for session ${sessionId}: no queued steer to retract (queueLen=${queue?.length ?? 0})`);
      return;
    }
    const idx = queue.findIndex((q) => collapseSteerText(q) === target);
    if (idx === -1) {
      logger.info(`Steer-remove for session ${sessionId}: no matching queued steer (already delivered or never queued)`);
      return;
    }
    queue.splice(idx, 1);
    logger.info(`Steer removed for session ${sessionId}: ${text.length} chars (remaining=${queue.length})`);
    await connection
      .sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "user_message_removed",
          content: { type: "text", text },
        } as any,
      })
      .catch(() => { /* non-fatal: visual-only retract ack */ });
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
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
          // Without this marker a client cannot tell command output from the
          // assistant's own text, so vscode-acp's compact mode folded it into the
          // running turn's collapsed block instead of showing it beside the command.
          _meta: { commandOutput: true },
        } as any,
      });
      return { handled: true };
    }

    if (name === "list-agents") {
      const { readdirSync, readFileSync } = await import("node:fs");
      const { join: joinPath } = await import("node:path");
      const { homedir } = await import("node:os");

      const sessionsDir = joinPath(homedir(), ".claude", "sessions");
      interface SessionEntry {
        pid?: number;
        sessionId?: string;
        name?: string;
        cwd?: string;
        startedAt?: number;
        status?: string;
      }
      const entries: SessionEntry[] = [];
      let dirEntries: import("node:fs").Dirent[];
      try {
        dirEntries = readdirSync(sessionsDir, { withFileTypes: true });
      } catch {
        dirEntries = [];
      }
      for (const d of dirEntries) {
        if (!d.isFile() || !d.name.endsWith(".json")) continue;
        try {
          const raw = readFileSync(joinPath(sessionsDir, d.name), "utf-8");
          entries.push(JSON.parse(raw) as SessionEntry);
        } catch {
          // skip unreadable
        }
      }

      const myPid = process.pid;
      const claudeSid = store.getClaudeSessionId(sessionId);
      const isSelf = (e: SessionEntry) =>
        e.pid === myPid || (claudeSid && e.sessionId === claudeSid);

      const self = entries.find(isSelf);
      const others = entries.filter((e) => !isSelf(e));

      const fmtElapsed = (ms: number): string => {
        const s = Math.floor((Date.now() - ms) / 1000);
        if (s < 60) return `${s}s ago`;
        const m = Math.floor(s / 60);
        if (m < 60) return `${m}m ago`;
        const h = Math.floor(m / 60);
        if (h < 24) return `${h}h ago`;
        return `${Math.floor(h / 24)}d ago`;
      };

      const lines: string[] = [];
      if (self?.name) {
        const shortId = self.sessionId ? self.sessionId.slice(0, 6) : "";
        lines.push(`This session: ${self.name}${shortId ? ` [${shortId}]` : ""} (the name other sessions use to message it)`);
      }
      if (others.length > 0) {
        lines.push(`\nOther Claude sessions (${others.length}):`);
        for (const o of others) {
          const nm = o.name ?? "unknown";
          const cwd = o.cwd ?? "";
          const started = o.startedAt ? fmtElapsed(o.startedAt) : "";
          const status = o.status ?? "";
          lines.push(`  [${status}] · ${nm} · ${cwd}${started ? ` · started ${started}` : ""}`);
        }
      } else {
        lines.push("\nNo other Claude sessions running.");
      }

      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: lines.join("\n") },
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
          // Without this marker a client cannot tell command output from the
          // assistant's own text, so vscode-acp's compact mode folded it into the
          // running turn's collapsed block instead of showing it beside the command.
          _meta: { commandOutput: true },
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
         messageId: nextCommandMessageId(),
         // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
         // Without this marker a client cannot tell command output from the
         // assistant's own text, so vscode-acp's compact mode folded it into the
         // running turn's collapsed block instead of showing it beside the command.
         _meta: { commandOutput: true },
       } as any,
     });
      await publishGoalSnapshot(sessionId, connection);
     return { handled: true };
    }

    if (name === "review-branch") {
      if (!rest) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: 'Command "/review-branch" requires branch name.' },
            messageId: nextCommandMessageId(),
            // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
            // Without this marker a client cannot tell command output from the
            // assistant's own text, so vscode-acp's compact mode folded it into the
            // running turn's collapsed block instead of showing it beside the command.
            _meta: { commandOutput: true },
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
            messageId: nextCommandMessageId(),
            // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
            // Without this marker a client cannot tell command output from the
            // assistant's own text, so vscode-acp's compact mode folded it into the
            // running turn's collapsed block instead of showing it beside the command.
            _meta: { commandOutput: true },
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
      // Plain text, not markdown: this is delivered as a transient SNAPSHOT row
      // (_meta.commandOutput) and vscode-acp renders command-output rows via
      // textContent (addCommandOutputRow → addMessageDOM 'system'), which shows
      // markdown source verbatim — the literal "**Model:**" asterisks. Plain
      // "Label: value" lines render cleanly and match the /ps panel's look.
      const lines = [
        `Model: ${currentModel}`,
        `Mode: ${currentMode}`,
        `Directory: ${cwd}`,
        `Session: ${sessionId}`,
        goal ? `Goal: ${goal.paused ? "(paused) " : ""}${goal.text.slice(0, 100)}` : "Goal: none",
        `Context: ${usedTokens.toLocaleString()} / ${contextWindow.toLocaleString()} tokens (${usedPct.toFixed(1)}%)`,
        `Max output: ${getMaxOutputTokens().toLocaleString()} tokens`,
        `Auto-compact: ${compactState}`,
      ];
      await connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: lines.join("\n") },
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
          // Without this marker a client cannot tell command output from the
          // assistant's own text, so vscode-acp's compact mode folded it into the
          // running turn's collapsed block instead of showing it beside the command.
          _meta: { commandOutput: true },
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
      scanSkillsDir(joinPath(homedir(), ".agents", "skills"), "global");
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
        skillsList = "No skills configured.\n\nSkills are loaded from:\n- ~/.claude/skills/\n- ~/.agents/skills/\n- <project>/.agents/skills/\n- <workspace>/.agents/skills/";
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
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
          // Without this marker a client cannot tell command output from the
          // assistant's own text, so vscode-acp's compact mode folded it into the
          // running turn's collapsed block instead of showing it beside the command.
          _meta: { commandOutput: true },
        } as any,
      });
      return { handled: true };
    }

   if (name === "goal") {
     const argument = rest.trim();
     if (!argument) {
       // /goal with no argument — show current goal status (matches Claude Code CLI behavior).
       const g = sessionGoals.get(sessionId);
       // Plain text (not markdown) — this is a transient command-output row that
       // the client renders via textContent, so markdown source would show verbatim.
       const statusText = g
         ? `Goal: ${g.paused ? "(paused) " : ""}${g.text.slice(0, 200)}`
         : "Goal: none";
       await connection.sessionUpdate({
         sessionId,
         update: {
           sessionUpdate: "agent_message_chunk",
           content: { type: "text", text: statusText },
           messageId: nextCommandMessageId(),
           // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
           // Without this marker a client cannot tell command output from the
           // assistant's own text, so vscode-acp's compact mode folded it into the
           // running turn's collapsed block instead of showing it beside the command.
           _meta: { commandOutput: true },
         } as any,
       });
       await publishGoalSnapshot(sessionId, connection);
       return { handled: true };
     }

      const lowerArg = argument.toLowerCase();
      // Claude Code CLI accepts: clear, stop, off, reset, none, cancel
      if (["clear", "stop", "off", "reset", "none", "cancel"].includes(lowerArg)) {
       sessionGoals.delete(sessionId);
       await connection.sessionUpdate({
         sessionId,
         update: {
           sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: "Goal cleared." },
           messageId: nextCommandMessageId(),
           // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
           // Without this marker a client cannot tell command output from the
           // assistant's own text, so vscode-acp's compact mode folded it into the
           // running turn's collapsed block instead of showing it beside the command.
           _meta: { commandOutput: true },
         } as any,
       });
        await publishGoalSnapshot(sessionId, connection);
       return { handled: true };
      }
      if (lowerArg === "pause") {
        const goal = sessionGoals.get(sessionId);
       if (goal) {
         goal.paused = true;
         goal.pausedAt = Date.now();
         await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
             content: { type: "text", text: "Goal paused. Use /goal resume to continue." },
             messageId: nextCommandMessageId(),
             // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
             // Without this marker a client cannot tell command output from the
             // assistant's own text, so vscode-acp's compact mode folded it into the
             // running turn's collapsed block instead of showing it beside the command.
             _meta: { commandOutput: true },
           } as any,
         });
          await publishGoalSnapshot(sessionId, connection);
       } else {
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "No goal to pause. Set one with /goal <objective>." },
              messageId: nextCommandMessageId(),
              // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
              // Without this marker a client cannot tell command output from the
              // assistant's own text, so vscode-acp's compact mode folded it into the
              // running turn's collapsed block instead of showing it beside the command.
              _meta: { commandOutput: true },
            } as any,
          });
        }
        return { handled: true };
      }
      if (lowerArg === "resume") {
        const goal = sessionGoals.get(sessionId);
       if (goal) {
         goal.paused = false;
         if (goal.pausedAt) {
           // Shift createdAt forward by the paused duration so timeUsedSeconds
           // excludes the pause interval.
           goal.createdAt += Date.now() - goal.pausedAt;
           goal.pausedAt = null;
         }
         await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
             content: { type: "text", text: `Goal resumed: ${goal.text.slice(0, 100)}` },
             messageId: nextCommandMessageId(),
             // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
             // Without this marker a client cannot tell command output from the
             // assistant's own text, so vscode-acp's compact mode folded it into the
             // running turn's collapsed block instead of showing it beside the command.
             _meta: { commandOutput: true },
           } as any,
         });
          await publishGoalSnapshot(sessionId, connection);
       } else {
          await connection.sessionUpdate({
            sessionId,
            update: {
              sessionUpdate: "agent_message_chunk",
              content: { type: "text", text: "No goal to resume. Set one with /goal <objective>." },
              messageId: nextCommandMessageId(),
              // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
              // Without this marker a client cannot tell command output from the
              // assistant's own text, so vscode-acp's compact mode folded it into the
              // running turn's collapsed block instead of showing it beside the command.
              _meta: { commandOutput: true },
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
            messageId: nextCommandMessageId(),
            // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
            // Without this marker a client cannot tell command output from the
            // assistant's own text, so vscode-acp's compact mode folded it into the
            // running turn's collapsed block instead of showing it beside the command.
            _meta: { commandOutput: true },
          } as any,
        });
        return { handled: true };
      }
     sessionGoals.set(sessionId, { text: argument, paused: false, createdAt: Date.now(), pausedAt: null, tokensUsed: 0, timeUsedSeconds: 0 });
     // Send user_message_chunk so the IDE shows the /goal command as a user
     // message bubble before the agent's "Goal set:" response. Without this,
     // the "Goal set:" text and the LLM's first response appear as two agent
     // messages with no user message separating them, looking concatenated.
     await connection.sessionUpdate({
       sessionId,
       update: {
         sessionUpdate: "user_message_chunk",
         content: { type: "text", text },
       } as any,
     });
     await connection.sessionUpdate({
       sessionId,
       update: {
         sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: `Goal set: ${argument.slice(0, 200)}` },
         messageId: nextCommandMessageId(),
         // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
         // Without this marker a client cannot tell command output from the
         // assistant's own text, so vscode-acp's compact mode folded it into the
         // running turn's collapsed block instead of showing it beside the command.
         _meta: { commandOutput: true },
       } as any,
     });
      await publishGoalSnapshot(sessionId, connection);
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
    skipCompleted: boolean = false,
  ): Promise<{ compactionId: string; summary: string } | boolean> {
    const claudeSid = store.getClaudeSessionId(sessionId);
    if (!claudeSid) {
      if (announce) {
        await connection.sessionUpdate({
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "No conversation to compact yet." },
            messageId: nextCommandMessageId(),
            // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
            // Without this marker a client cannot tell command output from the
            // assistant's own text, so vscode-acp's compact mode folded it into the
            // running turn's collapsed block instead of showing it beside the command.
            _meta: { commandOutput: true },
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
    // Ask the LLM to produce a concise summary of the conversation. The summary
    // seeds a FRESH session (see startSessionStreaming below), so anything the
    // ongoing work depends on that is NOT in the compacted transcript must be
    // named here — most importantly the current task board, which the model
    // manages with the TaskCreate/TaskUpdate tools. We inject the LIVE board
    // (sessionTodos, maintained from every TaskCreate/TaskUpdate — it survives
    // compaction on purpose, see the note after the success block) so the
    // summary reproduces it exactly (ids + statuses) rather than re-deriving
    // it from prose. Without this the fresh session starts with a board the
    // model has to rebuild from the summary's free-text "pending tasks", and
    // the client's plan bar (wiped by the compaction flush) is never re-pushed.
    const preCompactTodos = sessionTodos.get(sessionId);
    const todoBlock =
      preCompactTodos && preCompactTodos.size > 0
        ? `\n\nCurrent task board (preserve it verbatim in your summary — every task with its status):\n${Array.from(preCompactTodos.entries())
            .map(([_id, t]) => `- [${t.status}] ${t.content}`)
            .join("\n")}`
        : "";
    const compactPrompt =
      "Please provide a concise summary of our conversation so far, including key decisions, context, and any pending tasks. This summary will be used to continue the conversation with reduced context." +
      todoBlock;
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
      // Intentionally NOT clearing sessionTodos here. Compaction summarizes the
      // conversation but does not invalidate the tracked task list — the tasks
      // are still pending/in-progress/completed. Wiping the mirror here (while
      // the upstream CLI's task files survive compaction) left the plan panel
      // empty until the model happened to re-emit TaskCreate, which it usually
      // doesn't. codex and ericli both correctly avoid clearing plan state on
      // compaction; this matches them. The compaction summary already asks the
      // model for "pending tasks", so the surviving entries stay accurate.
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
      if (!skipCompleted) {
        await emitCompactionLifecycle(connection, sessionId, compactionId, "completed");
      }
      // Re-push the plan snapshot. The client wipes its plan bar when it
      // processes the compaction "completed" lifecycle frame (its history
      // flush), so without this the board the model still tracks — and that we
      // deliberately keep in sessionTodos — vanishes from the IDE until the
      // model happens to call TaskCreate/TaskUpdate again (which it usually
      // doesn't, since the surviving entries are accurate). Passing `out`
      // makes handleTaskTool buffer the snapshot instead of emitting inline;
      // we then emit it on the connection now that "completed" has flushed.
      // Only on the non-deferred path: with skipCompleted the client is still
      // mid-flush, so a plan frame here would race the wipe (the auto-compact
      // continuation re-pushes from the "completed" it defers to).
      if (!skipCompleted) {
        const out: { snapshot?: any } = {};
        handleTaskTool(sessionId, "TaskList", {}, out);
        if (out.snapshot) {
          connection
            .sessionUpdate({ sessionId, update: out.snapshot } as any)
            .catch(() => {
              // Best effort — a dropped re-push just means the bar stays
              // empty until the next task update, not a failure.
            });
          logger.debug(`Re-pushed plan snapshot after compaction for session ${sessionId}`);
        }
      }
      logger.info(`Compaction succeeded for session ${sessionId}`);
      return { compactionId, summary: result.text };
    } catch (compactErr) {
      const detail = compactErr instanceof Error ? compactErr.message : String(compactErr);
      logger.error(`Compaction failed for session ${sessionId}: ${detail}`);
      // `failed` makes the client hide the progress indicator and surface an
      // error instead of leaving a stuck "Compacting conversation..." banner.
      await emitCompactionLifecycle(connection, sessionId, compactionId, "failed", detail);
      return false;
    }
  }




  /**
   * Interrupt a running turn and re-prompt with the fresh <collaboration_mode>
   * block. Shared by the two mid-turn plan-mode flip paths so their behavior
   * can never drift apart:
   *
   *   - the USER flipping the collaboration picker (setSessionConfigOption,
   *     configId "collaboration_mode") — a deliberate mode change mid-turn;
   *   - the MODEL calling EnterPlanMode (tool_use event) — a self-initiated
   *     entry into Plan mode mid-turn.
   *
   * Why an interrupt is required at all: the Claude CLI is spawned PER PROMPT
   * (`claude -p --resume <id>`), and the per-turn spawn flags are fixed at spawn
   * time. A turn started in Default mode runs with `--permission-mode
   * acceptEdits` + `--dangerously-skip-permissions` (ACP needs the latter to
   * auto-approve). `--dangerously-skip-permissions` OVERRIDES `--permission-mode
   * plan` at the CLI layer — verified empirically: a plan-mode spawn WITHOUT it
   * hard-blocks writes (Write tool → permission_denials), but the SAME spawn WITH
   * it lets the write through. So merely flipping the runner's permission mode on
   * a running turn does nothing to the already-spawned process: the model keeps
   * full edit/Bash access for the rest of the turn. The only way to enforce
   * read-only (or default) mid-turn is to SIGINT the running CLI and re-spawn it
   * with the new mode's flags, carrying the fresh <collaboration_mode> block.
   *
   * @param planActive the mode the re-prompted turn should run in.
   * @param repromptModel the model override to reuse (so the flip doesn't change model).
   * @param repromptEffort the effort to reuse (so the flip doesn't change effort).
   */
  const interruptForPlanMode = (
    sessionId: string,
    planActive: boolean,
    repromptModel: string | undefined,
    repromptEffort: string | undefined,
  ): Promise<unknown> => {
    const repromptText = buildMidTurnModeChangeNotice(planActive);
    // SIGINT the running Claude subprocess (keeps the session id valid).
    // The current executePrompt settles with stopReason:"cancelled"; the
    // re-prompt chains after it on the per-session lock.
    runner.cancel(sessionId);
    // Chain the re-prompt on the lock: it runs once the cancelled turn
    // settles. .then(onFulfilled, onRejected) means we start the re-prompt
    // regardless of how the cancelled turn resolved — a cancelled turn that
    // threw (rare) must not block the mode switch.
    const prevLock = sessionPromptLocks.get(sessionId) ?? Promise.resolve();
    // A fresh generation so the cancel-latch scoping tracks this re-prompt
    // (a later queued prompt is not cancelled by the stale latch; runner.cancel
    // does not latch cancelledSessions, so the reprompt is not treated as a
    // cancelled-before-start prompt).
    const repromptGen = (promptGenCounter.get(sessionId) ?? 0) + 1;
    promptGenCounter.set(sessionId, repromptGen);
    const repromptLock = prevLock.then(
      () => executePrompt(sessionId, repromptText, repromptModel, repromptEffort, [], repromptGen),
      () => executePrompt(sessionId, repromptText, repromptModel, repromptEffort, [], repromptGen),
    );
    // Keep the lock chain alive for any further prompts queued behind.
    sessionPromptLocks.set(sessionId, repromptLock.catch(() => {}));
    return repromptLock;
  };

  /**
   * The full prompt-turn body (slash dispatch, LLM turn, steer drain,
   * auto-compact, usage/plan/goal bookkeeping). Hoisted to closure scope so
   * BOTH prompt() (the normal ACP entry point) AND setSessionConfigOption()
   * (the plan→exit interrupt, which cancels the running turn and re-prompts
   * with the fresh <collaboration_mode> block) can chain a turn on the
   * per-session lock.
   */
  const executePrompt = async (sessionId: string, bodyText: string, modelOverride: string | undefined, effectiveEffort: string | undefined, promptImages: import("./claude-runner.js").StdinContentBlock[] = [], promptGen: number = 0): Promise<PromptResponse> => {
    // Mark this session as having a prompt in flight for the entire executePrompt
    // body (LLM turn + steer drain + auto-compact + return). Cleared when the
    // lock chain settles (below). See sessionPromptInFlight.
    sessionPromptInFlight.add(sessionId);
    // Record which generation is now the in-flight one, so a concurrent cancel()
    // targets precisely this prompt (not a later queued one).
    inFlightGen.set(sessionId, promptGen);
    // If THIS prompt was cancelled while waiting in the serialization queue
    // behind a prior turn, don't start a new turn — drain-and-return.
    //
    // The latch is generation-scoped: cancel() records the generation of the
    // prompt that was in flight at cancel time (cancelledGen). Only a prompt
    // whose OWN generation matches that recorded cancel is treated as
    // cancelled-before-start. A fresh prompt queued behind a cancelled turn has
    // a LATER generation, so it is NOT hit by the stale latch — this is the
    // fix for "stop mid-stream, then the queued 'Send Now' message vanishes
    // instead of running as its own turn."
    if (isCancelledForGen(sessionId, promptGen)) {
      cancelledSessions.delete(sessionId);
      cancelledGen.delete(sessionId);
      logger.info(`Queued prompt (gen ${promptGen}) cancelled before start for session ${sessionId}`);
      return { stopReason: "cancelled" };
    }
    // ── Slash command handling ──────────────────────────────────
    // Intercept built-in commands (/ps, /goal, /skills, /status, /clear,
    // /compact) before starting an LLM turn. Shared dispatch logic lives in
    // dispatchSlashCommand() so prompt() and the steer-drain loop behave
    // identically. Read-only commands and /clear|/compact|/goal clear|pause
    // |resume are fully serviced and return immediately. /goal <objective>
    // returns a goalPrompt to start an LLM turn with.
    const dispatchResult = await dispatchSlashCommand(sessionId, connection, runner, store, bodyText);
    // When /goal sets a new objective, this holds the prompt to send to the
    // LLM to start working on it immediately.
    const goalPrompt = dispatchResult.startTurn ?? null;
    // A fully-serviced command (read-only, /clear, /compact, /goal subcmds)
    // needs no main LLM turn — but we still drain any steers queued during it
    // (e.g. steers queued while /compact was summarizing) and run the
    // auto-compact check, so we skip the main turn rather than returning early.
    const skipMainTurn = dispatchResult.handled && !goalPrompt;
    const command = dispatchResult.handled ? parseSlashCommand(bodyText) : null;

    // If there's an active (non-paused) goal, prepend it to the prompt.
   // goalPrompt is set when /goal just set a new objective — use it directly.
   // For subsequent prompts, prepend the goal as context.
   const activeGoal = sessionGoals.get(sessionId);
   let effectivePrompt = bodyText;
   if (goalPrompt) {
     effectivePrompt = goalPrompt;
   } else if (activeGoal && !activeGoal.paused && !command) {
     effectivePrompt = `[Goal: ${activeGoal.text}]\n\n${bodyText}`;
   }

    // Inject collaboration mode (plan mode) instructions
    const collabMode = sessionCollaborationModes.get(sessionId) ?? DEFAULT_COLLABORATION_MODE;
    if (collabMode === PLAN_COLLABORATION_MODE) {
      effectivePrompt = `<collaboration_mode># Collaboration Mode: Plan\n\nYou are now in Plan mode. Any previous instructions for other modes (e.g. Default mode) are no longer active.\n\nYour active mode changes only when new developer instructions with a different \`<collaboration_mode>...</collaboration_mode>\` change it; user requests or tool descriptions do not change mode by themselves.\n\nIn Plan mode, you MUST NOT make any changes to the codebase. Instead, analyze the request, investigate the codebase, and produce a detailed plan. Do not execute commands that modify files. Read-only investigation (searching, reading files, running read-only commands) is allowed.\n</collaboration_mode>\n\n${effectivePrompt}`;
    } else {
      // Default mode. The Claude CLI does NOT expose EnterPlanMode/ExitPlanMode
      // in headless -p mode (verified 2026-09-23: tools are interactive-only),
      // so the model cannot self-enter Plan mode. Plan mode is entered by the
      // user flipping the Collaboration Mode picker in the webview, which
      // triggers applyModeChange → --permission-mode plan on the next spawn.
      effectivePrompt = `<collaboration_mode># Collaboration Mode: Default\n\nYou are in Default mode: you may read and edit the codebase and execute commands.\n\nWhen a task requires exploration, architectural decisions, or multi-file changes — i.e. it is complex enough that planning first would avoid rework — present a plan and ask the user to switch to Plan mode via the Collaboration Mode picker before making changes. For simple, well-scoped tasks, proceed directly without planning.\n</collaboration_mode>\n\n${effectivePrompt}`;
    }

    const claudeSessionId = store.getClaudeSessionId(sessionId);
    const cwd = store.getCwd(sessionId)!;
   // Tool-call ids must be unique for the whole SESSION: clients resolve tool
   // elements by id, so a per-turn counter made a new turn adopt the previous
   // turn's rows. Kept on the store; the local reset below is gone too, since a
   // steer drain used to restart numbering mid-turn.
   const nextToolCallId = () => `call_${store.nextToolCallSeq(sessionId)}`;
   // Monotonic counter for ACP `messageId` on agent_message_chunk. The ACP
   // spec (ContentChunk.messageId) groups chunks into one logical message:
   // "All chunks belonging to the same message share the same messageId. A
   // change in messageId indicates a new message has started." Without it,
   // vscode-acp cannot tell that text emitted before and after a tool call
   // belong to the SAME assistant message, so it renders them as separate
   // bubbles (text→tool→text = 2 bubbles). We rotate the id at each
   // `assistant` event (one per LLM API call) so all text_delta chunks in
   // one API call share it; the next API call (after a tool result) gets a
   // new id → correctly starts a new bubble.
   let messageCounter = 0;
   let currentMessageId: string | null = null;

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
    const runModel = configOverrides["model"] as string | undefined;
    // Best-effort: trigger a context-window map load (one-shot, idempotent) so
    // the first usage_update of the turn uses the proxy's real per-model
    // window instead of falling through to the CLAUDE_CODE_MAX_CONTEXT_TOKENS
    // override (which is pinned to one model's window and would mis-report
    // every other model's denominator). This is NON-BLOCKING: a blocking await
    // here delays prompt() past the point where an early cancel() can be
    // observed, which loses the cancel signal (the test for cancel-during-prompt
    // times out). The startup fire-and-forget in index.ts and the refresh-on-miss
    // in getContextWindowForModel already populate the map; this just nudges an
    // in-flight load along if the proxy wasn't reachable at spawn.
    void loadModelContextWindows();
    void loadProxyPickerModels();
    // Only pass effort if the selected model actually supports it; otherwise
    // fall back to the model's default effort. Using resolveEffort() keeps
    // the actually-sent effort consistent with what the picker shows and
    // what /compact, steer drains, and auto-compact use (previously normal
    // turns fell back to undefined for reasoning models, while /compact via
    // prompt() used the default "high").
    const runModelResolved = resolveSessionModel(store, sessionId, runModel);
    const runEffort = resolveEffort(store, sessionId);

    logger.info(`Prompt for session ${sessionId}: ${bodyText.length} chars`);
    logger.debug(
      `Prompt content: ${bodyText.slice(0, 100)}${bodyText.length > 100 ? "..." : ""}`
    );

    const permissionPromises: Promise<void>[] = [];

   let lastUsage: { input_tokens?: number; output_tokens?: number } | null = null;
   // Track turn start time for goal budget accounting.
   const _turnStartTime = Date.now();
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
    // True when this turn forwards a native Claude CLI `local` slash command
    // (/usage, /heapdump, /context, /insights, /recap) that the CLI services
    // itself without an LLM turn. The CLI's output arrives as the `result`
    // field of the single `result` event (stop_reason null); when this flag
    // is set we surface that text as a commandOutput row so it is not
    // silently dropped. See isCliLocalCommandPrompt / CLI_LOCAL_COMMAND_NAMES.
    const isCliLocalTurn = isCliLocalCommandPrompt(bodyText);
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
        // The CLI's slash_commands list is the UNION of its builtins, skills,
        // and agents (plus plugin commands). Two of those categories are not
        // "commands" in the ACP sense, so exclude them before merging:
        //   - skills: invocable as /<name> but they're skill invocations, not
        //     session commands; showing them as "Claude Code: /<skill>" is
        //     misleading and they're already surfaced by /skills.
        //   - agents: subagent types (Explore, Plan, ...) — you never type
        //     /Explore as a slash command; the CLI lists them for the Task
        //     tool. Listing them here made the picker full of bogus entries.
        const skillSet = new Set(event.skillNames ?? []);
        const agentSet = new Set(event.agentNames ?? []);
        const builtinNames = new Set(BUILTIN_COMMANDS.map((b) => b.name));
        // Native command name → is it read-only/turn-safe? A native is
        // turn-safe ONLY if it's in the static read-only allowlist. We do NOT
        // trust the CLI's terminal_slash_commands (doctor/color/reload-plugins)
        // as a turnSafe signal: that list means "the CLI handles it in the
        // terminal layer", which is true for cosmetic or state-mutating
        // commands (color = terminal theme, reload-plugins = re-read config)
        // that are not read-only introspection useful to fire mid-turn.
        // Everything not in the allowlist is left unsafe so vscode-acp queues
        // it behind the running turn.
        const nativeCommands = event.slashCommands
          .filter((name) => !builtinNames.has(name) && !skillSet.has(name) && !agentSet.has(name))
          .map((name) => {
            const turnSafe = NATIVE_TURN_SAFE_COMMANDS.has(name);
            return {
              name,
              description: `Claude Code: /${name}`,
              input: null,
              ...(turnSafe ? { _meta: { turnSafe: true } } : {}),
            };
          });
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
      // Rotate the messageId at each API call boundary so all text_delta
      // chunks in THIS call share one id (→ one bubble in the client), and
      // the next call (after a tool result) starts a fresh id (→ new bubble).
      currentMessageId = `msg_${++messageCounter}`;
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
      sendEstimatedUsage(ctxEstimator, connection, sessionId, runModelResolved);
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
        sendEstimatedUsage(ctxEstimator, connection, sessionId, runModelResolved);
      }
    }
    // Handle user event (tool results / injected text) — accumulate for
    // context estimation.
    if (event.type === "user" && event.text) {
      ctxEstimator.addText(event.text);
    }
    // Subagent completion detection via <task-notification>.
    // Claude Code delivers a finished subagent's result as a `user` text
    // message containing a <task-notification> block — NOT as a tool_result
    // reusing the tool_use_id (the only tool_result for a subagent is the
    // immediate launch ack "Async agent launched successfully..."). The
    // notification carries a task-id (distinct from the tool_use_id) and a
    // <summary> naming the agent by its description, e.g.:
    //   Agent "Trace interrupt message wiring" completed
    //   background agent "Trace interrupt message wiring" ... [stopped]
    // So we link spawn→completion by DESCRIPTION: if a pending subagent's
    // description appears inside a <task-notification> in this user message,
    // flip its tool_call to "completed" and drop it from the pending map.
    // The post-turn flush then only catches subagents whose notification
    // never arrived before end_turn (the premature-end_turn case).
    if (event.type === "user" && event.text) {
      const text = event.text;
      const notifIdx = text.indexOf("<task-notification>");
      if (notifIdx >= 0) {
        const notif = text.slice(notifIdx);
        const pending = sessionPendingAgents.get(sessionId);
        if (pending && pending.size > 0) {
          // A single user message can carry multiple <task-notification>
          // blocks (several subagents finishing at once). Match each pending
          // subagent whose description the notification text references.
          for (const [tuId, info] of pending) {
            if (info.description && notif.includes(info.description)) {
              pending.delete(tuId);
              connection
                .sessionUpdate({
                  sessionId,
                  update: {
                    sessionUpdate: "tool_call_update",
                    toolCallId: info.acpCallId,
                    status: "completed",
                  } as any,
                })
                .catch(() => {});
              // Report what the subagent actually FOUND, not just that it
              // stopped.
              //
              // This notification is the only place its result exists on this
              // side: Claude Code delivers a finished subagent as this user
              // message, its content never enters the main conversation's
              // prompt, and the client's own memory capture records tool_call
              // titles plus the main assistant text — never tool results. So
              // without this the work is unrecoverable the moment the turn
              // ends, and could never become memory or a skill.
              //
              // Fire-and-forget: a client that does not implement the method,
              // or does not want the data, must not disturb a subagent that has
              // already finished.
              const findings = extractTaskNotificationSummary(notif, info.description);
              if (findings) {
                connection
                  .extNotification(SUBAGENT_FINDINGS_METHOD, {
                    sessionId,
                    id: tuId,
                    type: "subagent",
                    label: info.description,
                    task: info.description,
                    status: findings.stopped ? "stopped" : "completed",
                    summary: findings.summary,
                  })
                  .catch(() => {});
              }
              logger.info(
                `Subagent "${info.description}" (${tuId}) completion ` +
                  `notified via <task-notification>; marked completed.`,
              );
            }
          }
          if (pending.size === 0) {
            sessionPendingAgents.delete(sessionId);
          }
        }
      }
    }
    // A subagent's tool results live only in the subagent's own stream —
    // the main conversation sees the launch ack ("Async agent launched
    // successfully…") and the eventual <task-notification>, never a
    // tool_result carrying the subagent's child tool_use_ids. If any
    // tool_progress heartbeat ever mapped a sidechain id into the parent's
    // pending-tools map, it could never flip to completed here; drop such
    // stale entries so they don't sit in_progress until the post-turn flush.
    if (
      event.type === "user" &&
      event.isSidechain &&
      event.toolResultIds?.length
    ) {
      const staleTools = sessionPendingTools.get(sessionId);
      if (staleTools && staleTools.size > 0) {
        for (const tuId of event.toolResultIds) {
          staleTools.delete(tuId);
        }
        if (staleTools.size === 0) {
          sessionPendingTools.delete(sessionId);
        }
      }
    }
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
    // Surface the CLI's own output for a native `local` command (/usage,
    // /heapdump, /context, /insights, /recap). These are queued-and-drained
    // like a normal prompt (NOT bypassed mid-turn) and the CLI services them
    // itself — no LLM turn, no API call — returning the result as the `text`
    // of a single `result` event with no stop reason. Without this the text
    // was dropped here and /usage and /heapdump showed nothing in the ACP
    // chat. Emit it as a transient commandOutput SNAPSHOT row (like /ps and
    // /status) so it appears beside the command, not as a persistent
    // assistant bubble. Only when this turn was actually a CLI local command
    // (isCliLocalTurn) and the result carries text, so a normal end_turn that
    // happens to carry no/extra text is never mis-rendered.
    if (isCliLocalTurn && !event.isError && typeof event.text === "string" && event.text.trim()) {
      connection.sessionUpdate({
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: { type: "text", text: event.text },
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId
          // above. Without this marker a client cannot tell command output
          // from the assistant's own text, so vscode-acp's compact mode
          // folded it into the running turn's collapsed block instead of
          // showing it beside the command.
          _meta: { commandOutput: true },
        } as any,
      }).catch(() => {
        // Best effort — a missing command-output row must not fail the turn.
      });
    }
    // Record the context window Claude Code resolved for this model, so the
    // usage denominator can fall back to the CLI's own figure instead of a
    // hardcoded 200k when the proxy advertises nothing. NOTE: the CLI's value
    // is only a FALLBACK — getContextWindowForModel prefers the proxy's
    // /v1/models figure (authoritative) via `??`. The CLI often reports a
    // stale/default window for non-Claude models routed via ANTHROPIC_BASE_URL
    // (e.g. it reports 196608 for deepseek-v4-flash, whose real window is
    // 262144). If the proxy map doesn't have this model, refresh it — the
    // proxy may have added the model after our one-shot load.
    if (event.contextWindow && event.contextWindow > 0) {
      const cwModel = resolveSessionModel(store, sessionId, runModel);
      if (cliReportedContextWindows.get(cwModel) !== event.contextWindow) {
        cliReportedContextWindows.set(cwModel, event.contextWindow);
        logger.info(
          `Claude Code reports contextWindow=${event.contextWindow} for ${cwModel}`
        );
      }
      if (!modelContextWindows.has(cwModel)) {
        void loadModelContextWindows();
        void loadProxyPickerModels();
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
    const runModelResolved = resolveSessionModel(store, sessionId, runModel);
    const contextWindow = getContextWindowForModel(runModelResolved);
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
             // Group all chunks of this assistant API call into one logical
             // message so the client renders one bubble (not one per delta).
             messageId: currentMessageId ?? undefined,
           },
         });
       }
     } else if (event.type === "tool_use" && event.toolName) {
       ctxEstimator.addToolCall(event.toolName, event.toolInput);
       const toolCallId = nextToolCallId();
       logger.debug(`Tool call: ${event.toolName}`);
       // TaskCreate/TaskUpdate/TaskList maintain a todo list — mirror it
       // into an ACP `plan` update so the chat UI shows a plan bar with
       // the current task count (like codex-acp's update_plan). The plan
       // is a full snapshot, so the count reflects the live list size
       // rather than a running total of updates.
       handleTaskTool(sessionId, event.toolName, event.toolInput);
       // Claude Code owns a native ExitPlanMode tool: the model calls it once
       // the plan is ready and expects the host to leave Plan mode. Without
       // mapping it the session stayed in Plan forever — the picker still read
       // "Plan", the injected <collaboration_mode> block kept telling the model
       // that an execution request is a request to PLAN the execution, and a
       // follow-up "implement it" came back as code in the chat instead of
       // edits. Clear the flag and tell the client so the picker follows.
       if (event.toolName === "ExitPlanMode") {
         void leavePlanMode(sessionId, connection, store, runner);
       }
       // The symmetric enter path: the model calls EnterPlanMode when it decides
       // a task needs planning before acting. Map it so the picker flips to
       // "Plan" live (current_mode_update) and the next prompt injects the
       // read-only <collaboration_mode> Plan block. Without this the picker
       // only showed "Plan" on a user-initiated switch, so a self-entered plan
       // was invisible until the user manually flipped the mode.
       //
       // ROOT CAUSE FIX (plan-mode enforcement lag): EnterPlanMode fires MID-TURN,
       // but the running claude -p process was spawned at turn start in Default
       // mode with --dangerously-skip-permissions, which OVERRIDES the CLI's
       // --permission-mode plan read-only gate (verified empirically). Updating
       // the picker state alone therefore does NOT stop the model editing for
       // the rest of the turn — the in-flight process keeps its old spawn flags.
       // So we interrupt the running turn and re-prompt with the Plan
       // <collaboration_mode> block, exactly like the user-initiated picker
       // flip. enterPlanMode still does the picker/state writes; the interrupt
       // enforces it on the live process. We are already inside executePrompt's
       // onEvent (a turn in flight), so interruptForPlanMode chains the re-prompt
       // on the per-session lock after the cancelled turn settles — fire-and-forget
       // (no await: we're mid-stream and must keep the turn's event flow going).
       if (event.toolName === "EnterPlanMode") {
         void enterPlanMode(sessionId, connection, store, runner).then((entered) => {
           // Only interrupt on a GENUINE entry into plan (enterPlanMode returns
           // false for a redundant re-call when already in plan). newModeIsPlan
           // = true: the re-prompted turn runs under Plan (read-only).
           if (entered && (runner.isRunning(sessionId) || sessionPromptInFlight.has(sessionId))) {
             const repromptModel = store.getConfigOverrides(sessionId)["model"] as string | undefined;
             const repromptEffort = resolveEffort(store, sessionId);
             logger.info(
               `Session ${sessionId} model called EnterPlanMode mid-turn — cancelling ` +
                 `the running (full-access) turn and re-prompting with the Plan ` +
                 `<collaboration_mode> block so read-only takes effect NOW.`,
             );
             void interruptForPlanMode(sessionId, true, repromptModel, repromptEffort);
           }
         });
       }
       // Build a descriptive title + kind + (for edits) diff content +
       // locations, mirroring codex-acp, so the chat UI shows the action
       // verb and a click-to-open diff instead of a bare filename.
       const isSubagent =
         event.toolName === "Task" || event.toolName === "Agent";
       const update = buildToolCallUpdate(
         toolCallId,
         event.toolName,
         event.toolInput,
         store.getCwd(sessionId),
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
         const subInput = (event.toolInput ?? {}) as Record<string, unknown>;
         const subDesc =
           typeof subInput.description === "string"
             ? subInput.description
             : typeof subInput.prompt === "string"
               ? subInput.prompt
               : typeof subInput.subagent_type === "string"
                 ? subInput.subagent_type
                 : "";
         let pending = sessionPendingAgents.get(sessionId);
         if (!pending) {
           pending = new Map();
           sessionPendingAgents.set(sessionId, pending);
         }
         pending.set(event.toolUseId, {
           acpCallId: toolCallId,
           description: subDesc,
         });
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
         // Track toolName in parallel so the permission_request handler can
         // correlate by name (the permission event lacks toolUseId).
         let pendingNames = sessionPendingToolNames.get(sessionId);
         if (!pendingNames) {
           pendingNames = new Map();
           sessionPendingToolNames.set(sessionId, pendingNames);
         }
         pendingNames.set(event.toolUseId, event.toolName);
         // Server-side / network-blocked web tool watchdog. WebSearch
         // executes on Anthropic's API, so on a proxy route (non-anthropic
         // ANTHROPIC_BASE_URL) with a non-Claude model nobody executes it
         // and the tool hangs with no tool_result ever arriving. WebFetch is
         // fetched client-side by the CLI — normally fine through a proxy —
         // but on a host whose network blocks it the fetch stalls, so the
         // operator names it in CLAUDE_ACP_DISALLOWED_WEB_TOOLS and it gets
         // the same treatment. The CLI is told not to advertise them
         // (claude-runner buildExtraArgs), but the adapter must never depend
         // on that alone: flip the row to failed after a bounded wait so the
         // user sees a named cause instead of a frozen spinner. Cancel is
         // handled centrally at the prompt return (the cancelled flag +
         // "Turn ended with N tool(s) still pending" flush), so the timer
         // only needs to skip when the tool completes normally — hence the
         // per-session map plus a clear when the turn settles.
         if (
           watchedWebTools().has(event.toolName ?? "") &&
           isProxyBaseUrl(process.env.ANTHROPIC_BASE_URL) &&
           !/^claude/i.test(resolveSessionModel(store, sessionId, runModel))
         ) {
           const tuId = event.toolUseId;
           const timer = setTimeout(() => {
             webToolWatchdogs.get(sessionId)?.delete(timer);
             const pend = sessionPendingTools.get(sessionId);
             if (!pend?.has(tuId)) return; // completed / flushed already
             pend.delete(tuId);
             if (pend.size === 0) sessionPendingTools.delete(sessionId);
             logger.info(
               `Web tool ${event.toolName} (${tuId}) timed out after ${WEB_TOOL_TIMEOUT_MS}ms ` +
                 `on a proxy route — no server-side backend executed it.`,
             );
             connection
               .sessionUpdate({
                 sessionId,
                 update: {
                   sessionUpdate: "tool_call_update",
                   toolCallId,
                   status: "failed",
                   _meta: {
                     claudeCode: {
                       error:
                         "Web tools are executed by the Anthropic API and are not available on this proxy route.",
                     },
                   },
                 } as any,
               })
               .catch(() => {});
           }, WEB_TOOL_TIMEOUT_MS);
           let timers = webToolWatchdogs.get(sessionId);
           if (!timers) {
             timers = new Set();
             webToolWatchdogs.set(sessionId, timers);
           }
           timers.add(timer);
         }
       }
       connection.sessionUpdate({
         sessionId,
         update: emitted,
       });
     } else if (event.type === "tool_progress" && event.toolUseId) {
       // Claude CLI's periodic heartbeat while a tool runs. A long-running
       // tool (a Bash poll with sleep/ssh loops) emits nothing else for
       // minutes — no deltas, no tool_result — so without forwarding these
       // heartbeats the chat UI shows a frozen in_progress tool that looks
       // exactly like a hung session (the "session hangs, Stop fixes it"
       // symptom). Refresh the existing tool_call with the elapsed time so
       // the row visibly ticks. Sidechain (subagent) heartbeats are skipped
       // — subagent tools are tracked via sessionPendingAgents, and their
       // tool_use_ids are not in this session's pending-tools map.
       if (!event.toolSidechain) {
         const pendingTools = sessionPendingTools.get(sessionId);
         const acpCallId = pendingTools?.get(event.toolUseId);
         if (acpCallId) {
           connection
             .sessionUpdate({
               sessionId,
               update: {
                 sessionUpdate: "tool_call_update",
                 toolCallId: acpCallId,
                 status: "in_progress",
                 _meta: {
                   claudeCode: {
                     toolProgress: {
                       elapsedSeconds: event.toolElapsedSeconds,
                     },
                   },
                 },
               } as any,
             })
             .catch(() => {});
         }
       }
     } else if (event.type === "permission_request" && event.toolName) {
       ctxEstimator.addToolCall(event.toolName, event.toolInput);
       const toolCallId = nextToolCallId();
       const toolName = event.toolName;
       const toolInput = event.toolInput ?? {};
       logger.info(`Permission request: ${toolName}`);

        // Correlate the permission event back to the original in-progress
        // tool_use row by toolName (the permission event lacks toolUseId).
        // On approval, flip the original row to completed so it doesn't keep
        // spinning while the separate permission row is the one that resolves.
        const _matchNames = sessionPendingToolNames.get(sessionId);
        const _matchTools = sessionPendingTools.get(sessionId);
        let _matchedToolUseId: string | null = null;
        if (_matchNames && _matchTools) {
          for (const [tuId, name] of _matchNames) {
            if (name === toolName && _matchTools.has(tuId)) {
              _matchedToolUseId = tuId;
              break;
            }
          }
        }

        // Build the descriptive title + kind + diff content once, then reuse
        // across the pending → completed/failed lifecycle.
        const baseUpdate = buildToolCallUpdate(toolCallId, toolName, toolInput, store.getCwd(sessionId));

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

            // Flip the original in-progress tool_use row to completed and
            // clean up both pending maps so the tool_result handler doesn't
            // double-flip.
            if (isApproved && _matchedToolUseId && _matchTools && _matchNames) {
              const acpId = _matchTools.get(_matchedToolUseId);
              if (acpId) {
                connection.sessionUpdate({
                  sessionId,
                  update: {
                    sessionUpdate: "tool_call_update",
                    toolCallId: acpId,
                    status: "completed",
                  },
                }).catch(() => {});
                _matchTools.delete(_matchedToolUseId);
                _matchNames.delete(_matchedToolUseId);
                if (_matchTools.size === 0) sessionPendingTools.delete(sessionId);
                if (_matchNames.size === 0) sessionPendingToolNames.delete(sessionId);
              }
            }
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
      // Echo now that this message is actually going to the model, so the client's
      // release signal means delivered rather than received.
      await connection
        .sessionUpdate({
          sessionId,
          update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: steerText } } as any,
        })
        .catch(() => { /* non-fatal */ });
     // NOTE: the tool-call counter is deliberately NOT reset here. Restarting
     // it mid-turn on a steer drain made the steered work reuse the ids of the
     // tools that ran before it, so the client updated the wrong rows.
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
              await runner.continueSessionStreaming(claudeSid, steerDispatch.startTurn, onEvent, sessionId, runModel, runEffort);
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
             runModel,
              runEffort
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
      // Ensure the proxy /v1/models fetch has settled before the first
      // getContextWindowForModel call. startAgent() fires it as a
      // fire-and-forget; if the first prompt arrives before the ~200ms
      // fetch resolves, the map is empty and the lookup falls through to
      // CLAUDE_CODE_MAX_CONTEXT_TOKENS (pinned to one model's window),
      // mis-computing the compaction threshold for every other model.
      // After the first await the map is populated for the rest of the
      // process lifetime — subsequent prompts skip this (the promise is
      // already resolved).
      if (modelContextWindowsLoadPromise) {
        await modelContextWindowsLoadPromise;
      }
      // ─── Pre-turn compaction guard ────────────────────────────────
      // The post-turn auto-compact check (below, after "Prompt completed")
      // runs against `sessionLastUsage` — the REAL prompt count from the
      // previous turn's last API call. By the time THIS turn forwards its
      // prompt to the CLI, that figure is stale: the new user prompt (added
      // to ctxEstimator at addText(effectivePrompt) above) has grown the
      // request by the new prompt's token contribution. The between-turn
      // check can't see that growth, so a turn that measured 174,239 (under
      // the 176,947 threshold) arrives at vLLM as 176,609 input + 20,000
      // maxOutput = 196,609 > 196,608 → HTTP 500 context overflow.
      //
      // This guard mirrors the Claude Code CLI's pre-send `isAtBlockingLimit`
      // check (query.ts): project the about-to-be-sent prompt size using the
      // estimator's pending-inclusive count, and if it would cross the
      // threshold, compact BEFORE forwarding. getEstimatedTokensWithPending
      // adds the new prompt's char-estimated tokens to the last real count,
      // closing the measurement gap so the 90% threshold actually means 90%.
      if (isAutoCompactEnabled() && claudeSessionIdAfterDrain) {
        const projectedInput = ctxEstimator.getEstimatedTokensWithPending();
        const preTurnThreshold = getAutoCompactThreshold(runModelResolved);
        if (projectedInput >= preTurnThreshold) {
          logger.info(
            `Pre-turn compact for session ${sessionId}: projected ${projectedInput} >= ${preTurnThreshold} ` +
              `(lastReal=${ctxEstimator.getEstimatedTokens()}, pendingChars added ${ctxEstimator.getPendingChars()} chars, ` +
              `effectiveWindow=${getEffectiveContextWindowSize(runModelResolved)}, contextWindow=${getContextWindowForModel(runModelResolved)})`,
          );
          await doCompact(
            sessionId,
            connection,
            runner,
            store,
            runModel,
            runEffort,
            false,
          );
          // doCompact reset the estimator (conversation replaced by a short
          // summary). Re-add this turn's prompt so the estimator reflects
          // the post-compact conversation + the new prompt before the turn.
          ctxEstimator.addText(effectivePrompt);
          // Re-read the claude session id: doCompact starts a fresh session.
          // The variable captured below must point at the compacted session,
          // not the pre-compact one (which the summary replaced).
        }
      }
      // doCompact above may have replaced the claude session id; re-read so
      // the resume below targets the compacted conversation when compaction
      // ran, and the original when it didn't.
      //
      // Cancel re-check right before the spawn. The entry check (isCancelledForGen
      // above) runs once at executePrompt start, but the setup between that check
      // and the CLI spawn is full of blocking awaits (model-context-window load,
      // pre-turn compaction guard). A cancel landing in that window makes
      // runner.cancel a no-op — there is no process to SIGINT yet — and without
      // this re-check the spawn proceeds to a CLI that will never be cancelled,
      // so the prompt hangs until the client's fallback timer fires.
      if (isCancelledForGen(sessionId, promptGen)) {
        cancelledSessions.delete(sessionId);
        cancelledGen.delete(sessionId);
        logger.info(`Prompt (gen ${promptGen}) cancelled during setup for session ${sessionId} — not spawning the CLI`);
        return { stopReason: "cancelled" };
      }
      const claudeSessionIdForTurn = store.getClaudeSessionId(sessionId) ?? claudeSessionIdAfterDrain;
      // Build the stream-json stdin content for this turn. Always a text block
      // (effectivePrompt) plus any images. Routing the MAIN turn through the
      // stream-json stdin path (instead of the argv-prompt path) keeps the
      // CLI's stdin OPEN for the duration of the turn, which is what lets
      // handleSteer push a mid-turn steer into the in-flight turn (the Claude
      // CLI absorbs a second `user` line into the running turn). The argv
      // paths (startSessionStreaming / continueSessionStreaming) use
      // stdio ["ignore",...] and have no open stdin to steer into.
      const turnContent: import("./claude-runner.js").StdinContentBlock[] = [
        { type: "text", text: effectivePrompt },
        ...promptImages,
      ];
      if (claudeSessionIdForTurn) {
        try {
          result = await runner.continueSessionStreamingWithContent(
            claudeSessionIdForTurn,
            turnContent,
            onEvent,
            sessionId,
            runModel,
            runEffort
          );
        } catch (resumeErr) {
          const resumeMsg = resumeErr instanceof Error ? resumeErr.message : String(resumeErr);
          // A dead upstream must NOT cost the conversation. Starting fresh
          // here clears the persisted Claude session id, and nothing in the UI
          // can put it back — the user resumes, says "go on", and the agent
          // has no prior context while the transcript is still on disk.
          // Only a failure that actually implicates the session id may fall
          // back; anything transient is surfaced so a retry can resume.
          if (isTransientTurnFailure(resumeMsg, turnErrored)) {
            logger.warn(
              `Resume failed for ${claudeSessionIdForTurn} with a transient failure — ` +
                `keeping the session id so a retry can resume: ${resumeMsg}`,
            );
            throw resumeErr;
          }
          // Resume failed (expired session, etc.) — fall back to new session
          logger.warn(
            `Resume failed for ${claudeSessionIdAfterDrain}, starting fresh: ${resumeMsg}`
          );
          store.clearPersistedSession(cwd);
          const mcpServers = store.getMcpServers(sessionId);
          result = await runner.startSessionStreamingWithContent(
            cwd,
            turnContent,
            onEvent,
            sessionId,
            runModel,
            runEffort,
            mcpServers
          );
          store.setClaudeSessionId(sessionId, result.sessionId);
        }
      } else {
        const mcpServers = store.getMcpServers(sessionId);
        result = await runner.startSessionStreamingWithContent(
          cwd,
          turnContent,
          onEvent,
          sessionId,
          runModel,
          runEffort,
          mcpServers
        );
        // Only persist the Claude session id if the CLI actually emitted one.
        // On an early cancel (user hit "Send Now" before the CLI's init event
        // arrived) result.sessionId can be empty — storing "" would overwrite
        // nothing useful now, but guarding keeps the invariant that a stored
        // id is always a real one.
        if (result.sessionId) {
          store.setClaudeSessionId(sessionId, result.sessionId);
        }
      }
      }

      // Acknowledge a cancel IMMEDIATELY, before the permission-await and the
      // bounded subagent wait below. The user clicked Stop; those waits are
      // irrelevant once the turn is abandoned. Returning here promptly is what
      // keeps the host's 5s cancel-fallback from firing first: without it, a
      // turn with a settling subagent (wait up to pendingAgentWaitMs, default
      // 15s) or a slow permission resolve acknowledges cancel well past 5s, the
      // host force-aborts and posts its own promptEnd(cancelled), and the late
      // real resolution races a second promptEnd over the queue — the "stop
      // shows the queued message but the turn never ends" strand.
      if (isCancelledForGen(sessionId, promptGen)) {
        cancelledSessions.delete(sessionId);
        cancelledGen.delete(sessionId);
        if (inFlightGen.get(sessionId) === promptGen) inFlightGen.delete(sessionId);
        logger.info(`Prompt cancelled for session ${sessionId} (early, before post-turn waits)`);
        const _earlyCancelledGoal = sessionGoals.get(sessionId);
        const _earlyLu = lastUsage as any;
        if (_earlyCancelledGoal && !_earlyCancelledGoal.paused && _earlyLu) {
          const _earlyTurnTokens = (_earlyLu.input_tokens ?? 0) + (_earlyLu.output_tokens ?? 0);
          const _earlyTurnElapsed = Math.floor((Date.now() - _turnStartTime) / 1000);
          _earlyCancelledGoal.tokensUsed += _earlyTurnTokens;
          _earlyCancelledGoal.timeUsedSeconds += _earlyTurnElapsed;
          void publishGoalSnapshot(sessionId, connection).catch(() => {});
        }
        // Persist the partial turn's KV prefix so the next resume skips the
        // cold prefill (mirrors the cancel path in the catch below).
        saveSessionSlot(sessionId);
        return { stopReason: "cancelled" };
      }

      // Wait for all pending permission requests to resolve
      await Promise.all(permissionPromises);

      // Bounded wait for pending subagents: the CLI ended the turn before all
      // subagent <task-notification> messages arrived. Real Claude awaits
      // subagents before ending the turn; GLM/EricAI does not. Wait up to
      // `pendingAgentWaitMs` (default 15s) for the remaining notifications —
      // they often arrive within 1‑5s of the turn end. Only fall back to
      // marking completed if the wait times out.
      {
        const pendingAgentsTmp = sessionPendingAgents.get(sessionId);
        if (pendingAgentsTmp && pendingAgentsTmp.size > 0) {
          const waitDeadline = Date.now() + pendingAgentWaitMs;
          while (Date.now() < waitDeadline) {
            const stillPending = sessionPendingAgents.get(sessionId);
            if (!stillPending || stillPending.size === 0) break;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 500));
          }
        }
      }

      // Post-turn subagent flush. Subagents whose <task-notification>
      // arrived mid-turn were already flipped to "completed" and removed
      // from sessionPendingAgents by the user-event handler above. Anything
      // STILL pending here means the CLI ended the turn (end_turn) before
      // the subagent's result was injected back into the conversation —
      // i.e. a premature end_turn (observed with GLM/EricAI; real Claude
      // awaits subagents before ending the turn). Mark this "completed" as
      // a fallback so the UI spinner doesn't hang forever, and log the
      // anomaly so the lost result is visible: the main agent lacks the
      // subagent's output and may need a follow-up ("go on") to retrieve it
      // — at which point the <task-notification> lands and is matched there.
      const pendingAgents = sessionPendingAgents.get(sessionId);
      if (pendingAgents && pendingAgents.size > 0) {
        const pendingCount = pendingAgents.size;
        logger.info(
          `Turn ended with ${pendingCount} subagent(s) still pending ` +
            `(no <task-notification> before end_turn) for session ${sessionId} ` +
            `— marking completed as fallback; result was likely lost.`,
        );
        for (const info of pendingAgents.values()) {
          connection
            .sessionUpdate({
              sessionId,
              update: {
                sessionUpdate: "tool_call_update",
                toolCallId: info.acpCallId,
                status: "completed",
              } as any,
            })
            .catch(() => {});
        }
        pendingAgents.clear();
        sessionPendingAgents.delete(sessionId);

        // ── Auto follow-up for late subagent results ────────────────────
        // The turn ended before the subagent(s) reported back. The Claude
        // CLI only injects a finished subagent's <task-notification> into
        // the NEXT turn, so enqueue a synthetic steer that starts one — the
        // main agent then sees the pending results and can incorporate them,
        // instead of the results being silently lost until the user types
        // "go on". Bounded by AUTO_FOLLOW_UP_CAP so an auto-follow-up turn
        // that itself spawns subagents cannot cascade forever.
        const count = autoFollowUpCount.get(sessionId) ?? 0;
        if (count < AUTO_FOLLOW_UP_CAP) {
          autoFollowUpCount.set(sessionId, count + 1);
          const followUpText =
            `[Auto] ${pendingCount} background subagent(s) finished after the turn ended. ` +
            `Please incorporate their results (the <task-notification> blocks) and continue.`;

          // Deliver the synthetic follow-up as its OWN turn, NOT via
          // drainSteerQueue() on the shared queue. drainSteerQueue() drains
          // the ENTIRE steer queue — and a user steer typed mid-turn (e.g.
          // "deploy a2a" while the subagent was still pending) sits in that
          // same queue, accepted by handleSteer because a prompt was still in
          // flight. Consuming it here would fold the user's instruction into
          // the synthetic "[Auto] incorporate results" turn, so the model
          // answers the follow-up's framing first and the user's steer gets
          // deprioritized / batched rather than running as its own clean turn.
          //
          // Instead: deliver ONLY followUpText directly (resume the session,
          // the same as drainSteerQueue's non-slash path but scoped to one
          // message). Any user steers already in the queue are left in place;
          // the post-turn drain (below, at the end of executePrompt) delivers
          // them as their own turns.
          const claudeSidForFollowUp = store.getClaudeSessionId(sessionId);
          if (claudeSidForFollowUp) {
            // Echo the synthetic turn as a user chunk so the client shows it.
            await connection
              .sessionUpdate({
                sessionId,
                update: { sessionUpdate: 'user_message_chunk', content: { type: 'text', text: followUpText } } as any,
              })
              .catch(() => { /* non-fatal */ });
            logger.info(
              `Auto follow-up ${count + 1}/${AUTO_FOLLOW_UP_CAP} delivering as its own turn ` +
                `for session ${sessionId} (user steers left for the post-turn drain).`,
            );
            permissionPromises.length = 0;
            try {
              ctxEstimator.addText(followUpText);
              await runner.continueSessionStreaming(
                claudeSidForFollowUp,
                followUpText,
                onEvent,
                sessionId,
                runModel,
                runEffort,
              );
              await Promise.all(permissionPromises);
            } catch (followUpErr) {
              logger.error(
                `Auto follow-up turn failed for session ${sessionId}: ${followUpErr instanceof Error ? followUpErr.message : String(followUpErr)}`,
              );
            }
          } else {
            // No Claude session to resume into (e.g. straight after /clear).
            // Queue it so a later drain delivers it once a session exists.
            if (!steerQueues.has(sessionId)) steerQueues.set(sessionId, []);
            steerQueues.get(sessionId)!.push(followUpText);
            logger.info(
              `Auto follow-up ${count + 1}/${AUTO_FOLLOW_UP_CAP} queued (no active session yet) for session ${sessionId}.`,
            );
          }
        } else {
          logger.info(
            `Auto follow-up skipped for session ${sessionId}: reached cap ${AUTO_FOLLOW_UP_CAP}.`,
          );
        }
      }

      // Post-turn normal-tool flush. Tools whose tool_result never arrived
      // before end_turn (e.g. a tool_use emitted in the final assistant
      // message with no following user/tool_result) would otherwise hang in
      // in_progress forever. Mark them completed as a fallback.
      // The turn has settled — its web-tool watchdogs can no longer fire
      // meaningfully (the pending-tools map is about to be flushed), so
      // clear them to keep no timer alive past the turn.
      for (const t of webToolWatchdogs.get(sessionId) ?? []) clearTimeout(t);
      webToolWatchdogs.delete(sessionId);
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
      if (isCancelledForGen(sessionId, promptGen)) {
        cancelledSessions.delete(sessionId);
        cancelledGen.delete(sessionId);
        if (inFlightGen.get(sessionId) === promptGen) inFlightGen.delete(sessionId);
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
      // Non-blocking: the drain's follow-up turns run in the background so
      // executePrompt can return its JSON-RPC response immediately. A new
      // session/prompt chains on sessionPromptLocks + pendingDrainPromises
      // and waits for the drain's turn to finish before starting.
      const drainPromise = drainSteerQueue().catch((e) => {
        logger.error(`Post-turn drain failed for session ${sessionId}: ${e instanceof Error ? e.message : String(e)}`);
      });
      pendingDrainPromises.set(sessionId, drainPromise);
      void drainPromise.finally(() => pendingDrainPromises.delete(sessionId));

      // ─── Auto-compact check ───────────────────────────────────────────
      // Uses the same threshold Claude Code CLI computes (context window,
      // output-token reserve, 13k buffer and CLAUDE_AUTOCOMPACT_PCT_OVERRIDE),
      // and honors the CLI's DISABLE_AUTO_COMPACT / DISABLE_COMPACT switches,
      // so enabling or tuning compaction in one place affects both layers.
      const lastUsageEntry = sessionLastUsage.get(sessionId);
      if (lastUsageEntry && isAutoCompactEnabled()) {
        const threshold = getAutoCompactThreshold(runModelResolved);
        const used = lastUsageEntry.input_tokens + lastUsageEntry.output_tokens;
        if (used >= threshold) {
          logger.info(
            `Auto-compact triggered for session ${sessionId}: ${used} >= ${threshold} (effectiveWindow=${getEffectiveContextWindowSize(runModelResolved)}, contextWindow=${getContextWindowForModel(runModelResolved)})`,
          );
          // The compaction notice is emitted by doCompact as a dedicated
          // `_meta.contextCompaction` lifecycle, not as assistant text. An
          // agent_message_chunk here would be appended to the reply that just
          // streamed (vscode-acp only treats compaction text as standalone
          // when nothing is mid-stream), which is exactly the concatenated,
          // never-dismissed notice this replaces.
         // Emit "completed" IMMEDIATELY once the CLI backend finishes
         // compacting (skipCompleted=false), so the client clears the panel
         // and shows the "Context compacted" notice right away — not after
         // the continuation turn finishes. The continuation (re-sent original
         // prompt) then streams into the fresh panel as a new first line.
         // The client's pendingPostCompactChunks buffer absorbs any
         // continuation chunks that arrive inside the 400ms flush window, so
         // the immediate "completed" does not race the continuation.
         const compactResult = await doCompact(
           sessionId,
           connection,
           runner,
           store,
           runModel,
           runEffort,
           false, // no "No conversation to compact yet." text for auto-compact
           false, // skipCompleted=false — emit "completed" now, not after continuation
         ) as { compactionId: string; summary: string } | false;
         // doCompact already emitted "completed" + re-pushed the plan snapshot
         // (skipCompleted=false path). Drop the pre-compaction usage so the
         // ring doesn't show the stale count; the continuation's own
         // usage_update (below) repaints it with the fresh session's size.
         if (compactResult && typeof compactResult === 'object') {
           sessionLastUsage.delete(sessionId);
           const _postCompactEstimate = ctxEstimator.getEstimatedTokens();
           const _postCompactWindow = getContextWindowForModel(runModelResolved);
           await connection
             .sessionUpdate({
               sessionId,
               update: {
                 sessionUpdate: "usage_update",
                 used: Math.min(_postCompactEstimate, _postCompactWindow),
                 size: _postCompactWindow,
               } as any,
             })
             .catch(() => {});
         }
         // try/finally: the continuation below can throw (Claude CLI error,
         // aborted turn). "completed" was already emitted by doCompact, so a
         // throw here no longer leaves the lifecycle stuck at "start" — the
         // client already flushed. The finally block only re-pushes the plan
         // snapshot if the continuation succeeded (doCompact's re-push ran
         // before the continuation; a successful continuation may have
         // updated the board, so refresh it).
         try {
           // After compaction, re-send the original user prompt so the LLM
           // continues the task in the fresh (compacted) session. This matches
           // Codex behavior where compaction is transparent to the user — the
           // task continues automatically after context is compressed.
           logger.info(`Re-sending original prompt after auto-compact for session ${sessionId}`);
           const claudeSid = store.getClaudeSessionId(sessionId);
           if (claudeSid) {
             // Frame the continuation prompt so the model knows it is resuming
             // after a context compaction. Without this framing, the fresh
             // session (which only has the summary) can hallucinate tool
             // results as text ("ReadResult(file_path=...placeholder)...")
             // because it lost the conversation context and tries to simulate
             // prior tool calls it doesn't actually have results for. Preserve
             // accompanying images (if any) when re-sending the prompt.
             const continueFraming = `[Context was compacted to fit the model's context window. You are continuing the user's original request. Do not re-read files or re-execute tools that were already used — their results are summarized above. Continue working on the task.]\n\n${effectivePrompt}`;
             const continueContent = [
               { type: "text" as const, text: continueFraming },
               ...(promptImages.length > 0 ? promptImages : []),
             ];
             const continueResult = promptImages.length > 0
               ? await runner.continueSessionStreamingWithContent(
                   claudeSid,
                   continueContent,
                   onEvent,
                   sessionId,
                   runModel,
                   runEffort,
                 )
               : await runner.continueSessionStreaming(
                   claudeSid,
                   continueFraming,
                   onEvent,
                   sessionId,
                   runModel,
                   runEffort,
                 );
             store.setClaudeSessionId(sessionId, continueResult.sessionId);
             await Promise.all(permissionPromises);
             // Update usage after continuation
             const contEstimate = ctxEstimator.getEstimatedTokens();
             const contContextWindow = getContextWindowForModel(runModelResolved);
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
         } finally {
           // "completed" was already emitted by doCompact (skipCompleted=false),
           // so the client has already flushed the panel and shown the notice.
           // The continuation has now finished (successfully or not). Re-push
           // the plan snapshot: the continuation's TaskCreate/TaskUpdate calls
           // may have changed the board since doCompact's initial re-push, so
           // refresh it to the final state. (No "completed" re-emit here —
           // that would double-fire the lifecycle and re-wipe the panel.)
           if (compactResult && typeof compactResult === 'object') {
             const out: { snapshot?: any } = {};
             handleTaskTool(sessionId, "TaskList", {}, out);
             if (out.snapshot) {
               connection
                 .sessionUpdate({ sessionId, update: out.snapshot } as any)
                 .catch(() => {
                   // Best effort.
                 });
             }
           }
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
      //
      // Non-blocking: the drain's follow-up turns (continueSessionStreaming)
      // run in the background so executePrompt can return its JSON-RPC
      // response immediately. Without this, a mid-turn steer (absorbed via
      // writeSteerToMainTurn) keeps executePrompt alive through the drain's
      // LLM turns — the session/prompt response is never sent, vscode-acp's
      // connection.prompt() never resolves, and the Stop button never
      // switches to Send. The drain's turns still stream to the client; a
      // new session/prompt chains on sessionPromptLocks and waits for the
      // drain's turn to finish before starting.
      const drainPromise = drainSteerQueue().catch((e) => {
        logger.error(`Post-turn drain failed for session ${sessionId}: ${e instanceof Error ? e.message : String(e)}`);
      });
      pendingDrainPromises.set(sessionId, drainPromise);
      void drainPromise.finally(() => pendingDrainPromises.delete(sessionId));

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
            messageId: nextCommandMessageId(),
            // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
            // Without this marker a client cannot tell command output from the
            // assistant's own text, so vscode-acp's compact mode folded it into the
            // running turn's collapsed block instead of showing it beside the command.
            _meta: { commandOutput: true },
          },
       }).catch(() => {});
     }
    // Account token usage and elapsed time toward the active goal.
    const _activeGoal = sessionGoals.get(sessionId);
    if (_activeGoal && !_activeGoal.paused) {
      const _lu = lastUsage as any;
      const _turnTokens = _lu
        ? (_lu.input_tokens ?? 0) + (_lu.output_tokens ?? 0)
        : 0;
      const _turnElapsed = Math.floor((Date.now() - _turnStartTime) / 1000);
      _activeGoal.tokensUsed += _turnTokens;
      _activeGoal.timeUsedSeconds += _turnElapsed;
      await publishGoalSnapshot(sessionId, connection);
    }
    saveSessionSlot(sessionId);
    return { stopReason: "end_turn", usage: (lastUsage as any) ? { inputTokens: (lastUsage as any).input_tokens ?? 0, outputTokens: (lastUsage as any).output_tokens ?? 0, totalTokens: ((lastUsage as any).input_tokens ?? 0) + ((lastUsage as any).output_tokens ?? 0) } : undefined };
  } catch (err) {
    // The turn is unwinding (cancel or failure) — no watchdog may fire
    // against tools the post-turn flush is about to settle.
    for (const t of webToolWatchdogs.get(sessionId) ?? []) clearTimeout(t);
    webToolWatchdogs.delete(sessionId);
    // Check if cancelled
    if (isCancelledForGen(sessionId, promptGen)) {
      cancelledSessions.delete(sessionId);
      cancelledGen.delete(sessionId);
      if (inFlightGen.get(sessionId) === promptGen) inFlightGen.delete(sessionId);
      logger.info(`Prompt cancelled for session ${sessionId}`);
      // Account partial turn usage toward goal budget.
      const _cancelledGoal = sessionGoals.get(sessionId);
      const _clu = lastUsage as any;
      if (_cancelledGoal && !_cancelledGoal.paused && _clu) {
        const _turnTokens = (_clu.input_tokens ?? 0) + (_clu.output_tokens ?? 0);
        const _turnElapsed = Math.floor((Date.now() - _turnStartTime) / 1000);
        _cancelledGoal.tokensUsed += _turnTokens;
        _cancelledGoal.timeUsedSeconds += _turnElapsed;
        void publishGoalSnapshot(sessionId, connection).catch(() => {});
      }
      // A cancelled turn still leaves the full prompt prefix in the slot's
      // KV — persisting it lets the next resume skip the cold prefill even
      // though this turn never completed.
      saveSessionSlot(sessionId);
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
          messageId: nextCommandMessageId(),
          // Transient SNAPSHOT, not conversation — see nextCommandMessageId above.
          // Without this marker a client cannot tell command output from the
          // assistant's own text, so vscode-acp's compact mode folded it into the
          // running turn's collapsed block instead of showing it beside the command.
          _meta: { commandOutput: true },
        },
      });

      return { stopReason: "end_turn" };
    }
  };
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
            // Image support is enabled: `image` ACP content blocks are
            // forwarded to the CLI via --input-format=stream-json (see
            // acpImageBlocksToStdinBlocks + the WithContent runner methods).
            image: true,
            audio: false,
            embeddedContext: false,
          },
          mcpCapabilities: {
            http: false,
            sse: false,
          },
          sessionCapabilities: {
            list: {},
            fork: {},
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
      const mcpServers = normalizeMcpServers(params.mcpServers ?? []);
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

    // Best-effort KV cache restore before any replay — see slot-cache.ts.
    await tryRestoreSlot(sessionId);

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
         store.ensure(sessionId, params.cwd, normalizeMcpServers((params as any).mcpServers ?? []));
       } else {
         throw RequestError.resourceNotFound(
           `Session ${sessionId} not found`
         );
       }
     }

     // If the sessionId being loaded is itself a Claude CLI transcript UUID
     // (a session started from the `claude` CLI and surfaced via listSessions'
     // transcript scan), bind it as the Claude session id for this ACP
     // session. Without this, getClaudeSessionId returns undefined and the
     // replay fallback below picks the NEWEST transcript for the directory —
     // replaying the wrong conversation when the user picked a specific one.
     if (!store.getClaudeSessionId(sessionId) && typeof params.cwd === "string") {
       const { homedir } = await import("node:os");
       const { existsSync } = await import("node:fs");
       const transcriptPath = `${homedir()}/.claude/projects/${params.cwd.replace(/\//g, "-")}/${sessionId}.jsonl`;
       if (existsSync(transcriptPath)) {
         store.setClaudeSessionId(sessionId, sessionId);
         logger.info(
           `Bound listed CLI transcript ${sessionId} as the Claude session for replay`
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
            // Buffer replayed updates and only emit the surviving set after the
            // walk. The Claude CLI writes a `summary` entry at each compaction
            // (the post-compact context that replaces the conversation); turns
            // BEFORE the last compaction are no longer part of the live context
            // and must NOT be re-rendered into the chat panel. Dropping them on
            // a `summary` marker mirrors codex-acp's streamThreadHistory
            // (`threadUpdates.length = 0` on a contextCompaction item). We also
            // reset sessionTodos so stale pre-compaction plan state doesn't
            // leak into the post-compaction plan snapshot.
            const replayedUpdates: any[] = [];
            let compactedAtLeastOnce = false;
            let replayMessageCounter = 0;
            for (const line of lines) {
              try {
                const entry = JSON.parse(line);
                // Subagent (sidechain) entries are a separate conversation —
                // replaying them would interleave subagent chatter into the
                // main history.
                if (entry.isSidechain) continue;
                // Compact-summary continuation: when the CLI's context overflows
                // mid-session it rewrites the conversation and records the new
                // leading turn as a `user`-type entry whose content is the
                // machine-generated "This session is being continued from a
                // previous conversation… Summary: …" banner (14-19 KB). It is
                // flagged isCompactSummary / isVisibleInTranscriptOnly — metadata
                // for the CLI, not a user prompt. The type==="user" branch below
                // would otherwise replay that wall of summary text as a user
                // message, and on resume it renders as a user bubble the user
                // never typed.
                //
                // It also marks the post-compaction boundary. The CLI does NOT
                // write a separate `type === "summary"` entry (that marker is
                // dead in every real transcript), so this is the ONLY signal
                // that the conversation was rewritten. Treat it the same as a
                // summary marker: drop everything replayed so far (pre-compact
                // turns are no longer live context — the CLI replaced them with
                // the summary) and reset derived plan state. Only turns after
                // the LAST isCompactSummary entry are replayed, so a resumed
                // session shows the post-compact conversation, not the full
                // pre-compact history.
                if (entry.isCompactSummary) {
                  replayedUpdates.length = 0;
                  sessionTodos.delete(sessionId);
                  compactedAtLeastOnce = true;
                  continue;
                }
                // Compaction marker: the CLI replaces the conversation with
                // this summary. Drop everything replayed so far (it is no
                // longer live context) and reset derived plan state.
                if (entry.type === "summary") {
                  replayedUpdates.length = 0;
                  sessionTodos.delete(sessionId);
                  compactedAtLeastOnce = true;
                  continue;
                }
                if (entry.type === "user") {
                  const msg = entry.message;
                  if (!msg) continue;
                  let content = msg.content;
                  if (typeof content === "string") {
                    content = stripReplayPromptFraming(content);
                    // Interrupt notices are machine artifacts, not user prompts.
                    // Emitting them as user_message_chunk forces the webview to
                    // finalizeCurrentAssistantTurn() before its own skip check,
                    // splitting the last assistant turn and pushing the notice
                    // after the summary. Skip them here so they never reach the
                    // replay buffer.
                    if (/^\[Request interrupted by user\]/i.test(content.trim())) {
                      continue;
                    }
                  }
                  if (typeof content === "string" && content.trim()) {
                    replayedUpdates.push({
                      sessionUpdate: "user_message_chunk",
                      content: { type: "text", text: content },
                    });
                  } else if (Array.isArray(content)) {
                    for (const block of content) {
                      if (block.type === "text" && block.text) {
                        const clean = stripReplayPromptFraming(block.text);
                        if (!clean.trim()) continue;
                        if (/^\[Request interrupted by user\]/i.test(clean.trim())) continue;
                        replayedUpdates.push({
                          sessionUpdate: "user_message_chunk",
                          content: { type: "text", text: clean },
                        });
                      }
                    }
                  }
                } else if (entry.type === "assistant") {
                  const msg = entry.message;
                  if (!msg) continue;
                  const content = msg.content;
                  // One messageId per replayed assistant message, shared across
                  // its text blocks — mirrors the live path's per-API-call id
                  // so the client groups a resumed message into one bubble.
                  const replayMessageId = `replay_msg_${++replayMessageCounter}`;
                  if (Array.isArray(content)) {
                    for (const block of content) {
                      if (block.type === "text" && block.text) {
                        replayedUpdates.push({
                          sessionUpdate: "agent_message_chunk",
                          content: { type: "text", text: block.text },
                          messageId: replayMessageId,
                        });
                      } else if (block.type === "tool_use") {
                        // Build the update with the SAME mapper the live path
                        // uses, so a resumed session shows "Read file 'x.ts'"
                        // and the right `kind`/diff content instead of a bare
                        // tool name like "Read".
                        replayedUpdates.push(
                          buildToolCallUpdate(
                            block.id ?? `replay-${Math.random().toString(36).slice(2, 8)}`,
                            block.name ?? "tool",
                            block.input ?? {},
                            store.getCwd(sessionId),
                          ) as any,
                        );
                        // Reconstruct plan state from replayed TaskCreate/Update
                        // tool_use blocks so a reloaded session shows its todo
                        // list (the live path calls handleTaskTool at L2663; the
                        // replay path previously did not, so sessionTodos stayed
                        // empty and no `plan` snapshot reached the client). The
                        // `out` receiver buffers the snapshot instead of emitting
                        // it inline, so it is truncated/ordered with the rest of
                        // the replayed updates (incl. at a compaction marker).
                        const planOut: { snapshot?: any } = {};
                        if (
                          handleTaskTool(
                            sessionId,
                            block.name ?? "",
                            block.input ?? {},
                            planOut,
                          ) &&
                          planOut.snapshot
                        ) {
                          replayedUpdates.push(planOut.snapshot);
                        }
                      }
                    }
                  }
                }
              } catch {
                // Skip unparseable lines
              }
            }
            // Emit only the surviving (post-last-compaction) updates in order.
            for (const update of replayedUpdates) {
              await connection.sessionUpdate({ sessionId, update } as any);
            }
            logger.info(
              `Replay complete for session ${sessionId}` +
                (compactedAtLeastOnce
                  ? ` (truncated at compaction marker; ${replayedUpdates.length} post-compact update(s) emitted)`
                  : ` (${replayedUpdates.length} update(s) emitted)`),
            );
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

    /**
     * Fork a session (ACP `session/fork` — experimental).
     *
     * The conversation history lives in the Claude CLI JSONL transcript at
     * ~/.claude/projects/<encoded-cwd>/<claude-session-id>.jsonl, not in the
     * SessionStore (the store only maps ACP session id → claude session id).
     * Forking therefore means: copy the source transcript to a fresh Claude
     * session UUID and register a new ACP session pointing at the copy. The
     * fork's first prompt resumes (`--resume <new-claude-id>`) from the copy,
     * so subsequent turns append to the branch and leave the original
     * conversation untouched — true branch semantics.
     */
    async unstable_forkSession(
      params: ForkSessionRequest
    ): Promise<ForkSessionResponse> {
      const { sessionId: sourceSessionId } = params;
      logger.info(`Fork request for session ${sourceSessionId}`);

      // Same rehydration leniency as loadSession: the agent is respawned per
      // client session, so the source may not be in this process's map yet.
      if (!store.has(sourceSessionId)) {
        if (typeof params.cwd === "string" && params.cwd) {
          logger.info(
            `Fork source ${sourceSessionId} unknown to this process — rehydrating from cwd ${params.cwd}`
          );
          store.ensure(sourceSessionId, params.cwd, normalizeMcpServers((params as any).mcpServers ?? []));
        } else {
          throw RequestError.resourceNotFound(
            `Session ${sourceSessionId} not found`
          );
        }
      }

      const sourceClaudeId = store.getClaudeSessionId(sourceSessionId);
      const sourceCwd = store.getCwd(sourceSessionId)!;

      // No Claude session mapped yet (the source has had no turns). There is
      // nothing to branch from, but we still produce a valid empty fork so the
      // client gets a new session id rather than an error — matching how a
      // fresh session/new behaves.
      const forkClaudeId = randomUUID();
      if (sourceClaudeId) {
        const { homedir } = await import("node:os");
        const projectDir = `${homedir()}/.claude/projects/${sourceCwd.replace(/\//g, "-")}`;
        const sourcePath = `${projectDir}/${sourceClaudeId}.jsonl`;
        const forkPath = `${projectDir}/${forkClaudeId}.jsonl`;
        try {
          const { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } = await import("node:fs");
          if (existsSync(sourcePath)) {
            if (!existsSync(projectDir)) {
              mkdirSync(projectDir, { recursive: true });
            }
            // Copy and rewrite the internal session_id so the CLI accepts
            // --resume <forkClaudeId> (it cross-checks the file name against
            // the session_id field in each line). Without this, the forked
            // transcript still carries the source's UUID internally and the
            // CLI rejects the resume: "not a UUID and does not match any
            // session title" (32-hex forkClaudeId) or "session not found"
            // (dashed UUID mismatch).
            copyFileSync(sourcePath, forkPath);
            const raw = readFileSync(forkPath, "utf-8");
            const lines = raw.split("\n").filter((l) => l.trim());
            const rewritten = lines.map((line) => {
              try {
                const d = JSON.parse(line);
                if (d.session_id === sourceClaudeId) d.session_id = forkClaudeId;
                if (d.sessionId === sourceClaudeId) d.sessionId = forkClaudeId;
                return JSON.stringify(d);
              } catch {
                return line;
              }
            });
            writeFileSync(forkPath, rewritten.join("\n"));
            logger.info(
              `Forked transcript: ${sourceClaudeId}.jsonl → ${forkClaudeId}.jsonl (${sourceCwd})`
            );
          } else {
            // Source transcript not on disk (e.g. compaction wiped it, or it
            // lives on a different host). The fork starts empty; the mapping
            // below still lets the fork diverge from the source's future turns.
            logger.warn(
              `Fork source transcript not found at ${sourcePath}; fork starts empty`
            );
          }
        } catch (e) {
          logger.error(
            `Failed to copy transcript for fork: ${e instanceof Error ? e.message : String(e)}`
          );
        }
      }

      // Register a new ACP session bound to the fork's Claude session id. Use
      // the source's cwd unless the client explicitly overrode it.
      const forkCwd =
        typeof params.cwd === "string" && params.cwd ? params.cwd : sourceCwd;
      const forkAcpId = generateSessionId();
      store.create(forkAcpId, forkCwd, normalizeMcpServers((params as any).mcpServers ?? []));
      store.setClaudeSessionId(forkAcpId, forkClaudeId);

      logger.info(
        `Forked session ${sourceSessionId} → ${forkAcpId} (claude ${forkClaudeId})`
      );

      return {
        sessionId: forkAcpId,
        modes: buildModeState(store.getMode(sourceSessionId) ?? DEFAULT_MODE),
        configOptions: buildConfigOptions(store, forkAcpId),
      } as ForkSessionResponse;
    },

    async listSessions(
      params: ListSessionsRequest
    ): Promise<ListSessionsResponse> {
      const cwd = params.cwd ?? undefined;
      const storeSessions = store.listAll(cwd);

      // Merge in CLI-spawned transcripts (sessions started from the `claude`
      // CLI, not through ACP). The Claude CLI has no app-server RPC, so these
      // are enumerated from ~/.claude/projects/<encoded-cwd>/*.jsonl — the
      // same transcripts loadSession already replays. Dedupe by Claude
      // session UUID: an ACP session that already maps to a Claude UUID
      // suppresses the raw-transcript copy so the conversation is listed once.
      const cliSessions = await listCliTranscriptSessions(cwd);
      const coveredClaudeIds = new Set(
        storeSessions
          .map((s) => s.claudeSessionId)
          .filter((id): id is string => Boolean(id))
      );
      const merged = [
        ...storeSessions,
        ...cliSessions.filter((s) => !coveredClaudeIds.has(s.sessionId)),
      ];
      // Sort by updatedAt descending (most recent first), matching the store's
      // ordering, so merged CLI sessions interleave by activity rather than
      // appearing only after every ACP session.
      merged.sort((a, b) => {
        const at = a.updatedAt ? Date.parse(a.updatedAt) : 0;
        const bt = b.updatedAt ? Date.parse(b.updatedAt) : 0;
        return bt - at;
      });
      // Strip the internal claudeSessionId field before returning — it is not
      // part of the ACP wire type.
      const sessions = merged.map(({ claudeSessionId: _omit, ...rest }) => rest);
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

      // Route through applyModeChange so the legacy set_mode path also syncs the
      // runner's --permission-mode and flips the collaboration picker in
      // lockstep — previously it only wrote store.mode + emitted
      // current_mode_update, leaving the runner on the previous turn's flags.
      await applyModeChange(sessionId, connection, store, runner, modeId);

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

      // The mode picker (configId "mode") and the collaboration picker
      // (configId "collaboration_mode") are the two halves of plan mode and must
      // flip in lockstep, AND both must harden the runner's spawn flag so the
      // next claude -p process picks up --permission-mode plan. Routing both
      // through applyModeChange keeps a single source of truth (store.mode) that
      // persists across restarts (setMode persists to disk; setConfigOverride
      // would write configOverrides["mode"], which nothing reads back, so the
      // picker would reset to the default after restart), syncs the runner
      // immediately, and emits both current_mode_update + config_option_update.
      if (configId === "mode") {
        await applyModeChange(sessionId, connection, store, runner, String(value));
      } else if (configId === COLLABORATION_MODE_CONFIG_ID) {
        // Collaboration mode is the prompt-level half of plan mode; map it to
        // the permission mode the runner enforces (plan → read-only, default →
        // the session's previous non-read-only mode so we don't silently downgrade
        // a full-access session to plain agent).
        const nextCollab = String(value);
        const targetMode =
          nextCollab === PLAN_COLLABORATION_MODE
            ? "read-only"
            : (store.getMode(sessionId) ?? DEFAULT_MODE) === "read-only"
              ? DEFAULT_MODE
              : (store.getMode(sessionId) ?? DEFAULT_MODE);
        // Capture the previous collaboration mode BEFORE applyModeChange overwrites
        // it, so the mid-turn announcement only fires on a genuine flip.
        const previousCollabMode = sessionCollaborationModes.get(sessionId);
        await applyModeChange(sessionId, connection, store, runner, targetMode);
        // The <collaboration_mode> block is built once, at the START of each
        // turn, from sessionCollaborationModes. A flip made DURING a turn is
        // therefore invisible to the running turn: the model keeps acting under
        // the OLD block already in the transcript, and a bare user-level
        // "you're in Default now" notice cannot override that strong prior.
        //
        // On a genuine flip while a turn is in flight, CANCEL the running turn
        // and re-prompt with the fresh mode's <collaboration_mode> block so the
        // model immediately perceives the switch and continues in THIS turn.
        // applyModeChange has already updated sessionCollaborationModes to
        // `nextCollab`, so executePrompt builds the correct block for the new
        // mode. We pass buildMidTurnModeChangeNotice(prevWasPlan) as bodyText —
        // executePrompt prepends the (now-correct) <collaboration_mode> block
        // on top of it, so the model sees both the authoritative mode
        // declaration AND an explicit "just switched, do not wait" nudge.
        //
        // This mirrors codex-acp's plan-entry interrupt
        // (CodexAcpServer.applyCollaborationModeChange → interruptSessionTurn):
        // codex interrupts on plan ENTRY; we interrupt on EITHER genuine flip
        // (plan→default AND default→plan) so the model always acts on the
        // current mode without waiting for the user to open a new turn.
        const inFlight =
          runner.isRunning(sessionId) || sessionPromptInFlight.has(sessionId);
        if (shouldAnnounceModeChange(previousCollabMode, nextCollab) && inFlight) {
          // The re-prompt's notice describes the NEW mode (the one the turn is
          // about to run in), NOT the previous one. buildMidTurnModeChangeNotice
          // takes "is the new mode plan?" — so a Default→Plan flip passes true
          // ("no more edits") and a Plan→Default flip passes false ("implement
          // now"). (An earlier revision passed prevWasPlan, the PREVIOUS mode,
          // which inverted both directions.)
          const newModeIsPlan = nextCollab === PLAN_COLLABORATION_MODE;
          // Resolve the same model/effort overrides the running turn used, so
          // the re-prompt does not silently change model or effort.
          const repromptModel = store.getConfigOverrides(sessionId)["model"] as string | undefined;
          const repromptEffort = resolveEffort(store, sessionId);
          logger.info(
            `Session ${sessionId} mode flip ${previousCollabMode ?? "(none)"} → ${nextCollab} ` +
              `with a turn in flight — cancelling and re-prompting with the fresh ` +
              `${newModeIsPlan ? "Plan" : "Default"} <collaboration_mode> block.`,
          );
          // SIGINT the running turn and re-prompt with the fresh block (shared
          // core with the model-initiated EnterPlanMode path).
          //
          // Wait for the re-prompt to actually START the LLM turn (not just be
          // enqueued) before returning, so the client's config_option_update
          // reflects a turn that is genuinely running in the new mode.
          await interruptForPlanMode(sessionId, newModeIsPlan, repromptModel, repromptEffort);
          logger.info(`Session ${sessionId} mode-change re-prompt completed`);
        }
      } else {
        // Capture the previous model BEFORE the override is set, so a model
        // switch can re-emit a usage_update with the NEW window. Without this,
        // the webview's context ring keeps showing the old model's size
        // (e.g. 262144 for Qwen3.8-27B) until the next LLM turn fires a fresh
        // usage_update — the user switches to the 1M variant and the ring
        // still reads 262k. usage_update is only sent inside the result-event
        // handler (per turn), so a config change that doesn't start a turn
        // leaves the ring stale.
        const prevModel = store.getConfigOverrides(sessionId)[configId] as string | undefined;
        store.setConfigOverride(sessionId, configId, value);
        logger.info(`Session ${sessionId} config ${configId} set to ${value}`);
        if (configId === "model" && prevModel !== value) {
          reEmitUsageOnModelSwitch(sessionId, connection, store, prevModel, String(value));
        }
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

      // Sync the runner's permission mode from the stored session mode BEFORE
      // any spawn in this prompt. enterPlanMode/leavePlanMode set it during the
      // previous turn's tool events, but a user-initiated picker switch
      // (setSessionMode / setSessionConfigOption) only writes store.mode — it
      // does NOT touch the runner. Without this sync, a manual "Read-only"
      // picker flip would not propagate to the next claude -p spawn (the CLI
      // would still run with the previous turn's --permission-mode).
      runner.setSessionPermissionMode(store.getMode(sessionId) ?? DEFAULT_MODE);

      // Extract text from content blocks
      let text = prompt
        .filter((block): block is { type: "text"; text: string } =>
          block.type === "text"
        )
        .map((block) => block.text)
        .join("\n");

      if (!text.trim()) {
        throw RequestError.invalidParams("Empty prompt text");
      }

      // Extract image content blocks (base64 media) for the multimodal stdin
      // path. When present, the LLM turn below is sent via the Claude CLI's
      // `--input-format=stream-json` protocol so vision-capable models
      // actually receive the pictures. Text-only prompts keep using the
      // `-p <string>` path unchanged (zero behavior change).
      const promptImages = acpImageBlocksToStdinBlocks(prompt);
      const hasImages = promptImages.length > 0;

      // Graceful degradation: if the serving model is text-only (e.g. a proxy
      // base-url override routes "Claude" to a non-multimodal backend), persist
      // the images to temp files and replace them with a text note instead of
      // forwarding image blocks the upstream will reject. Native Claude (no
      // proxy) is vision-capable, so this is a no-op there — zero behavior
      // change for the common case. Mirrors codex-acp / pi-acp / ericli.
      if (hasImages) {
        const currentModel = store.getConfigOverrides(sessionId)["model"] as string | undefined;
        if (!modelSupportsVision(currentModel)) {
          logger.info(
            `Image prompt on text-only model (${currentModel || "unknown"}): degrading ${promptImages.length} image(s) to text notes`,
          );
          const { note, tempPaths } = degradeImageBlocksForTextOnlyModel(
            promptImages.filter((b): b is { type: "image"; source: { media_type: string; data: string } } => b.type === "image"),
            sessionId,
          );
          if (tempPaths.length > 0) {
            logger.info(`Persisted attached images to temp files: ${tempPaths.join(", ")}`);
          }
          // Append the note to the prompt text so the model sees it; clear the
          // image blocks so the turn uses the text-only `-p <string>` path.
          promptImages.length = 0;
          text = text + "\n" + note;
        }
      }

      // Read-only slash commands (/ps, /status, /skills, /list-agents, and
      // the no-turn subcommands of /goal) are fully serviced by
      // dispatchSlashCommand without touching shared per-session state
      // (sessionTodos, sessionEstimators, the Claude session id, the onEvent
      // closure). They bypass the per-session prompt lock so the result
      // appears immediately — not after the in-flight turn finishes.
      //
      // CRITICAL: when the command is handled we must RETURN here, not fall
      // through to the serialized path. dispatchSlashCommand is NOT idempotent
      // (e.g. /goal clear|pause mutates sessionGoals and the read-only commands
      // emit a snapshot), and executePrompt re-dispatches the same command a second
      // time. Falling through would double-emit the snapshot (and double-mutate
      // goal state) — and the queued second dispatch is what made a mid-turn
      // command "drain and trigger another turn" after end_turn.
      //
      // When dispatchSlashCommand does NOT handle it (defensive: a /goal
      // subcommand that actually starts a turn, e.g. /goal <objective>), fall
      // through to the serialized path so that turn runs normally.
      const cmd = parseSlashCommand(text);
      if (cmd && isReadOnlyCommand(cmd.name, cmd.rest)) {
        const dispatchResult = await dispatchSlashCommand(sessionId, connection, runner, store, text);
        if (dispatchResult.handled) {
          return { stopReason: "end_turn" };
        }
      }

      // Compute the model/effort overrides here (closure scope) so the
      // plan->exit interrupt in setSessionConfigOption can re-prompt with the
      // same settings.
      const configOverrides = store.getConfigOverrides(sessionId);
      const modelOverride = configOverrides["model"] as string | undefined;
      const effectiveEffort = resolveEffort(store, sessionId);

      // Serialize against any in-flight prompt on this session. See
      // sessionPromptLocks above: the SDK runs concurrent prompt() handlers,
      // so without this a mid-turn /skills (or any second prompt) would race
      // with the active turn. We chain on the lock and run the whole prompt
      // body — slash dispatch, LLM turn, steer drain, auto-compact — once the
      // prior prompt finishes. The lock is keyed by session and always
      // replaced, so it never holds a stale promise across /clear (which
      // resolves quickly and clears per-session state under the lock too).
      const prev = (sessionPromptLocks.get(sessionId) ?? Promise.resolve()).then(
        () => pendingDrainPromises.get(sessionId) ?? Promise.resolve(),
        () => pendingDrainPromises.get(sessionId) ?? Promise.resolve()
      );
      // Assign this prompt a monotonically-increasing generation for the
      // session. A cancel() latches by generation (the one in flight at cancel
      // time), so a NEW prompt queued behind a cancelled one (a later
      // generation) is not hit by the stale latch in executePrompt's
      // before-start check. See `cancelledGen`.
      const myGen = (promptGenCounter.get(sessionId) ?? 0) + 1;
      promptGenCounter.set(sessionId, myGen);
      const next = prev.then(
        () => executePrompt(sessionId, text, modelOverride, effectiveEffort, promptImages, myGen),
        () => executePrompt(sessionId, text, modelOverride, effectiveEffort, promptImages, myGen)
      );
      // Clear the in-flight flag once executePrompt settles (success or failure).
      // If a newer prompt queued behind this one has already re-added the flag
      // by the time this fires, the delete is a harmless no-op on a re-added
      // entry — but ordering (this finally registered before the newer prompt's
      // executePrompt runs) means this clears first, then the newer
      // executePrompt re-adds, so the flag stays accurate. Same guard for
      // inFlightGen: only clear it if it still points at THIS prompt's
      // generation (a newer prompt will have overwritten it).
      next.finally(() => {
        sessionPromptInFlight.delete(sessionId);
        if (inFlightGen.get(sessionId) === myGen) {
          inFlightGen.delete(sessionId);
          // A cancel() that arrived AFTER this prompt's turn already ended
          // (end_turn) latches cancelledSessions/cancelledGen but this prompt
          // never cleared them (it took the normal path, not a cancel site).
          // Clear them now if no newer prompt is in flight, so a stale latch
          // doesn't linger and wrongly ignore a later steer or cancel a much
          // later prompt. The generation-scoped before-start check already
          // prevents the stale latch from cancelling a fresh queued prompt;
          // this just prevents the Set/Map entries from sticking around.
          if (!sessionPromptInFlight.has(sessionId)) {
            cancelledSessions.delete(sessionId);
            cancelledGen.delete(sessionId);
          }
        }
      });
      sessionPromptLocks.set(sessionId, next.catch(() => {}));
      return next;
    },

    async cancel(params: CancelNotification): Promise<void> {
      logger.info(`Cancel request for session ${params.sessionId}`);
      cancelledSessions.add(params.sessionId);
      // Record which prompt generation this cancel targets — the one currently
      // executing (inFlightGen). A fresh prompt queued behind the cancelled one
      // has a LATER generation, so executePrompt's before-start check (which
      // matches the cancel's target generation against the prompt's own) will
      // not cancel it.
      const targetGen = inFlightGen.get(params.sessionId);
      if (targetGen !== undefined) {
        cancelledGen.set(params.sessionId, targetGen);
      }
      // Discard steers that never reached the model.
      //
      // Claude CLI cannot inject mid-turn, so a steer waits in this queue until
      // the main turn ends (drainSteerQueue). A cancel exits executePrompt BEFORE
      // that drain, so the steer used to survive and was delivered at the end of
      // the NEXT turn — reported as: queue two messages, "Send Now" the second,
      // both bubbles appear, and the last thing the model works on is the first
      // message. drainSteerQueue already states the policy ("Cancellation
      // discards the rest — the user asked to stop"); it only ever applied when a
      // drain was already running.
      //
      // Nothing is lost: vscode-acp keeps the message in its own queue until the
      // agent echoes it, so an undelivered steer drains as an ordinary prompt in
      // its own turn, after the one the user asked for.
      const pendingSteers = steerQueues.get(params.sessionId);
      if (pendingSteers && pendingSteers.length > 0) {
        logger.info(
          `Cancel: discarding ${pendingSteers.length} undelivered steer(s) for session ${params.sessionId}`,
        );
        pendingSteers.length = 0;
      }
      runner.cancel(params.sessionId);
    },

    /**
     * Handle custom (non-standard) ACP notifications.
     * The SDK's AgentSideConnection dispatches unknown notification methods here.
     * Used for session/steer and session/stop.
     */
    async extNotification(method: string, params: Record<string, unknown>): Promise<void> {
      if (method === "session/steer") {
        return handleSteer(params as { sessionId: string; text: string });
      }
      if (method === "session/steer-remove") {
        return handleSteerRemove(params as { sessionId: string; text: string });
      }
      // `session/stop`: the host's "kill background processes" notification
      // (Codex/Codex-ACP handle it to terminate spawned long-running processes
      // WITHOUT interrupting the active turn). The Claude CLI has no separate
      // background-process pool — its subprocess IS the turn, and stopping the
      // turn is `session/cancel`'s job. Acknowledge as a no-op so the host's
      // Stop path (which sends session/stop + session/cancel) doesn't log a
      // spurious "Unknown notification method: session/stop" warning.
      if (method === "session/stop") {
        const sessionId = (params as { sessionId?: string })?.sessionId;
        logger.info(`session/stop for session ${sessionId ?? "(none)"} — no separate background processes to stop (Claude CLI subprocess is the turn); no-op`);
        return;
      }
      logger.warn(`Unknown notification method: ${method}`);
    },
  };
}
