import { describe, it, expect } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import type { StreamEvent } from "../src/claude-runner.js";

/**
 * The runner must tag sidechain (subagent) text/thinking/result events with
 * isSidechain so the agent can drop them. Claude CLI nests content_block_delta
 * in a stream_event wrapper whose parent_tool_use_id is non-null for a
 * subagent's own API calls; the flag must be captured from the WRAPPER before
 * the unwrap, and attached to every emitted text/thinking event.
 */

describe("runner tags sidechain stream events", () => {
  function feed(lines: unknown[]): StreamEvent[] {
    const runner = new ClaudeRunner();
    const events: StreamEvent[] = [];
    for (const line of lines) {
      (runner as any).processStreamLine(line, (e: StreamEvent) => events.push(e));
    }
    return events;
  }

  it("marks a wrapped content_block_delta from a subagent as sidechain", () => {
    const events = feed([
      {
        type: "stream_event",
        parent_tool_use_id: "toolu_parent",
        event: {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "subagent thinking out loud" },
        },
      },
    ]);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("text_delta");
    expect(events[0].isSidechain).toBe(true);
  });

  it("main-conversation deltas are not sidechain", () => {
    const events = feed([
      {
        type: "stream_event",
        parent_tool_use_id: null,
        event: { type: "content_block_delta", delta: { type: "text_delta", text: "mine" } },
      },
    ]);
    expect(events[0].isSidechain).toBe(false);
  });

  it("marks a subagent thinking_delta as sidechain", () => {
    const events = feed([
      {
        type: "stream_event",
        parent_tool_use_id: "toolu_parent",
        event: { type: "content_block_delta", delta: { type: "thinking_delta", thinking: "hmm" } },
      },
    ]);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("thinking");
    expect(events[0].isSidechain).toBe(true);
  });

  it("marks a subagent final-block text as sidechain", () => {
    const events = feed([
      {
        type: "assistant",
        parent_tool_use_id: "toolu_parent",
        message: { content: [{ type: "text", text: "subagent final" }] },
      },
    ]);
    const text = events.find((e) => e.type === "text_delta" && e.fromAssistantBlock);
    expect(text?.isSidechain).toBe(true);
  });
});
