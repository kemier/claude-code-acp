import { describe, it, expect, vi } from "vitest";
import { ClaudeRunner } from "../src/claude-runner.js";
import type { StreamEvent } from "../src/claude-runner.js";

/** Drive the private processStreamLine with raw stream-json objects. */
function feed(lines: unknown[]): StreamEvent[] {
  const runner = new ClaudeRunner();
  const events: StreamEvent[] = [];
  for (const line of lines) {
    (runner as any).processStreamLine(line, (e: StreamEvent) => events.push(e));
  }
  return events;
}

describe("stream_event envelope (--include-partial-messages)", () => {
  // Verified against claude v2.1.220: with --include-partial-messages the CLI
  // emits {"type":"stream_event","event":{...}} wrappers, never bare
  // content_block_delta lines. Without the unwrap the delta path is dead code.
  it("unwraps a wrapped content_block_delta into a streaming text_delta", () => {
    const events = feed([
      {
        type: "stream_event",
        event: {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text: "hello" },
        },
      },
    ]);
    expect(events).toHaveLength(1);
    expect(events[0].type).toBe("text_delta");
    expect(events[0].text).toBe("hello");
    // Streaming deltas are NOT from the final assistant block — this is what
    // lets the agent drop the duplicate full-text re-emission.
    expect(events[0].fromAssistantBlock).toBeUndefined();
  });

  it("preserves delta order so text renders progressively", () => {
    const mk = (text: string) => ({
      type: "stream_event",
      event: { type: "content_block_delta", delta: { type: "text_delta", text } },
    });
    const events = feed([mk("Let me "), mk("check "), mk("that.")]);
    expect(events.map((e) => e.text)).toEqual(["Let me ", "check ", "that."]);
  });

  it("ignores non-text stream_event envelopes", () => {
    const events = feed([
      { type: "stream_event", event: { type: "message_start", message: {} } },
      { type: "stream_event", event: { type: "content_block_start", index: 0 } },
      { type: "stream_event", event: { type: "content_block_stop", index: 0 } },
      { type: "stream_event", event: { type: "message_stop" } },
    ]);
    expect(events).toHaveLength(0);
  });

  it("marks the completed assistant block so the duplicate can be dropped", () => {
    const events = feed([
      {
        type: "assistant",
        parent_tool_use_id: null,
        message: { content: [{ type: "text", text: "hello" }] },
      },
    ]);
    const textEvent = events.find((e) => e.type === "text_delta");
    expect(textEvent?.text).toBe("hello");
    expect(textEvent?.fromAssistantBlock).toBe(true);
  });
});

describe("streaming CLI args", () => {
  it("requests partial messages on every streaming invocation", async () => {
    const runner = new ClaudeRunner();
    const captured: string[][] = [];
    (runner as any).runStreaming = vi.fn(async (args: string[]) => {
      captured.push(args);
      return { text: "", sessionId: "sid" };
    });

    await runner.startSessionStreaming("/tmp", "p", () => {});
    await runner.continueSessionStreaming("sid", "p", () => {});
    await runner.startSessionWithMcp("/tmp", "p", [], () => {});

    expect(captured).toHaveLength(3);
    for (const args of captured) {
      expect(args).toContain("--include-partial-messages");
      expect(args).toContain("stream-json");
      expect(args).toContain("--verbose");
    }
  });
});

describe("exact usage from message_delta", () => {
  // Verified against claude v2.1.220 via the CCR/proxy path:
  //   completed `assistant` message → usage.input_tokens: 0
  //   `result`                      → 44193 (SUM of both calls)
  //   `message_delta`               → 20702 then 23491 (true per-call prompts)
  // Only message_delta is usable as the live context size.
  it("forwards per-call prompt tokens with cache fields", () => {
    const events = feed([
      {
        type: "stream_event",
        parent_tool_use_id: null,
        event: {
          type: "message_delta",
          usage: {
            input_tokens: 20694,
            output_tokens: 2,
            cache_read_input_tokens: 100,
            cache_creation_input_tokens: 50,
          },
        },
      },
    ]);
    const md = events.find((e) => e.type === "message_delta");
    expect(md?.usage?.input_tokens).toBe(20694);
    expect(md?.usage?.cache_read_input_tokens).toBe(100);
    expect(md?.usage?.cache_creation_input_tokens).toBe(50);
    expect(md?.isSidechain).toBe(false);
  });

  it("flags subagent calls from the envelope's parent_tool_use_id", () => {
    // The marker lives on the WRAPPER, so it must be captured before unwrapping.
    const events = feed([
      {
        type: "stream_event",
        parent_tool_use_id: "toolu_sub",
        event: { type: "message_delta", usage: { input_tokens: 90000, output_tokens: 5 } },
      },
    ]);
    expect(events.find((e) => e.type === "message_delta")?.isSidechain).toBe(true);
  });

  it("surfaces the context window Claude Code resolved for the model", () => {
    const events = feed([
      {
        type: "result",
        subtype: "success",
        result: "ok",
        session_id: "sid",
        usage: { input_tokens: 44193, output_tokens: 27 },
        modelUsage: {
          "glm-5.2": {
            inputTokens: 44193,
            outputTokens: 27,
            contextWindow: 200000,
            maxOutputTokens: 32000,
          },
        },
      },
    ]);
    const res = events.find((e) => e.type === "result");
    expect(res?.contextWindow).toBe(200000);
    expect(res?.maxOutputTokens).toBe(32000);
  });

  it("omits contextWindow when modelUsage is absent", () => {
    const events = feed([
      { type: "result", subtype: "success", result: "ok", session_id: "sid", usage: {} },
    ]);
    const res = events.find((e) => e.type === "result");
    expect(res?.contextWindow).toBeUndefined();
  });
});
