import { describe, it, expect } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import type { StreamEvent } from "../src/claude-runner.js";


/** Drive processStreamLine (private) with a raw stream-json object. */
function feed(lines: unknown[]): StreamEvent[] {
  const runner = new ClaudeRunner();
  const events: StreamEvent[] = [];
  for (const line of lines) {
    (runner as any).processStreamLine(line, (e: StreamEvent) => events.push(e));
  }
  return events;
}

describe("assistant events carry real per-call usage", () => {
  it("forwards input_tokens and cache tokens from message.usage", () => {
    const events = feed([
      {
        type: "assistant",
        parent_tool_use_id: null,
        message: {
          content: [{ type: "text", text: "hi" }],
          usage: {
            input_tokens: 118035,
            output_tokens: 1271,
            cache_read_input_tokens: 200,
            cache_creation_input_tokens: 100,
          },
        },
      },
    ]);
    const assistant = events.find((e) => e.type === "assistant");
    expect(assistant?.usage?.input_tokens).toBe(118035);
    expect(assistant?.usage?.output_tokens).toBe(1271);
    expect(assistant?.usage?.cache_read_input_tokens).toBe(200);
    expect(assistant?.usage?.cache_creation_input_tokens).toBe(100);
    expect(assistant?.isSidechain).toBe(false);
  });

  it("flags subagent messages so their usage is not billed to the parent", () => {
    const events = feed([
      {
        type: "assistant",
        parent_tool_use_id: "toolu_abc",
        message: {
          content: [{ type: "text", text: "subagent work" }],
          usage: { input_tokens: 90000, output_tokens: 10 },
        },
      },
    ]);
    expect(events.find((e) => e.type === "assistant")?.isSidechain).toBe(true);
  });

  it("still emits an assistant event when usage is absent", () => {
    const events = feed([
      { type: "assistant", message: { content: [{ type: "text", text: "x" }] } },
    ]);
    const assistant = events.find((e) => e.type === "assistant");
    expect(assistant).toBeDefined();
    expect(assistant?.usage).toBeUndefined();
  });
});

