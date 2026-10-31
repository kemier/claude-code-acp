/**
 * Proxy models emit Claude Code's plan-mode tools as chat-template XML when
 * those tools are absent from the headless tool list:
 *
 *   <tool_call>
 *   <function=ExitPlanMode>
 *   <parameter=plan>
 *   ...
 *   </parameter>
 *   </function>
 *   </tool_call>
 *
 * Under `claude -p` the CLI never shows the ExitPlanMode approval dialog, and
 * this adapter disallows the tool in plan mode, so the call lands in the text
 * stream. This filter pulls those blocks out of a token stream and leaves
 * every other `<tool_call>` untouched.
 */

export type PlanModeToolName = "ExitPlanMode" | "EnterPlanMode";

export interface PlanModeToolCall {
  name: PlanModeToolName;
  /** Plan body from `<parameter=plan>` or a JSON `arguments.plan`. */
  plan?: string;
}

const OPEN = "<tool_call>";
const CLOSE = "</tool_call>";

function isPlanModeTool(name: string): name is PlanModeToolName {
  return name === "ExitPlanMode" || name === "EnterPlanMode";
}

/** Longest suffix of `buf` that is a proper prefix of `marker`. */
function heldSuffix(buf: string, marker: string): number {
  const max = Math.min(buf.length, marker.length - 1);
  for (let n = max; n > 0; n--) {
    if (marker.startsWith(buf.slice(buf.length - n))) return n;
  }
  return 0;
}

function planFromArgs(args: unknown): string | undefined {
  if (typeof args === "string") {
    try {
      return planFromArgs(JSON.parse(args));
    } catch {
      return undefined;
    }
  }
  if (!args || typeof args !== "object") return undefined;
  const plan = (args as { plan?: unknown }).plan;
  return typeof plan === "string" ? plan.trim() : undefined;
}

/**
 * Parse one `<tool_call>...</tool_call>` block, or an unclosed ExitPlanMode /
 * EnterPlanMode block held until end of turn. Returns null when the block is
 * some other tool — the caller keeps the raw text.
 */
export function parsePlanModeToolBlock(block: string): PlanModeToolCall | null {
  const inner = block.replace(/<\/?tool_call>/g, "").trim();
  if (inner.startsWith("{")) {
    try {
      const obj = JSON.parse(inner) as {
        name?: unknown;
        function?: { name?: unknown };
        arguments?: unknown;
        parameters?: unknown;
      };
      const name = typeof obj.name === "string" ? obj.name : obj.function?.name;
      if (typeof name !== "string" || !isPlanModeTool(name)) return null;
      const plan = planFromArgs(obj.arguments ?? obj.parameters);
      return plan === undefined ? { name } : { name, plan };
    } catch {
      // Not JSON. Fall through to the <function=...> form.
    }
  }

  const fn = /<function\s*=\s*([A-Za-z0-9_]+)\s*>/.exec(block);
  if (!fn || !isPlanModeTool(fn[1])) return null;
  const name = fn[1];
  const params = new Map<string, string>();
  const closed = /<parameter\s*=\s*([A-Za-z0-9_]+)\s*>([\s\S]*?)<\/parameter>/g;
  for (let m = closed.exec(block); m; m = closed.exec(block)) {
    params.set(m[1], m[2].trim());
  }
  if (!params.has("plan")) {
    const open = /<parameter\s*=\s*plan\s*>([\s\S]*)$/.exec(block);
    if (open) {
      const rest = open[1]
        .replace(/\s*<\/parameter>[\s\S]*$/, "")
        .replace(/\s*<\/function>[\s\S]*$/, "")
        .replace(/\s*<\/tool_call>\s*$/, "")
        .trim();
      if (rest) params.set("plan", rest);
    }
  }
  const plan = params.get("plan");
  return plan === undefined ? { name } : { name, plan };
}

export interface PlanModeFilterResult {
  text: string;
  calls: PlanModeToolCall[];
}

/**
 * Streaming splitter. Holds a trailing prefix of `<tool_call>` and an
 * unclosed plan-mode block so tags never reach the client. `flush()` releases
 * whatever remains at end of turn.
 */
export class PlanModeToolXmlFilter {
  private buf = "";

  push(delta: string): PlanModeFilterResult {
    this.buf += delta;
    return this.drain(false);
  }

  flush(): PlanModeFilterResult {
    return this.drain(true);
  }

  private drain(flush: boolean): PlanModeFilterResult {
    let text = "";
    const calls: PlanModeToolCall[] = [];
    while (true) {
      const start = this.buf.indexOf(OPEN);
      if (start === -1) {
        if (flush) {
          text += this.buf;
          this.buf = "";
        } else {
          const hold = heldSuffix(this.buf, OPEN);
          text += this.buf.slice(0, this.buf.length - hold);
          this.buf = this.buf.slice(this.buf.length - hold);
        }
        break;
      }
      const end = this.buf.indexOf(CLOSE, start);
      if (end === -1) {
        text += this.buf.slice(0, start);
        this.buf = this.buf.slice(start);
        if (flush) {
          const parsed = parsePlanModeToolBlock(this.buf);
          if (parsed) calls.push(parsed);
          else text += this.buf;
          this.buf = "";
        }
        break;
      }
      text += this.buf.slice(0, start);
      const block = this.buf.slice(start, end + CLOSE.length);
      this.buf = this.buf.slice(end + CLOSE.length);
      const parsed = parsePlanModeToolBlock(block);
      if (parsed) calls.push(parsed);
      else text += block;
    }
    return { text, calls };
  }
}
