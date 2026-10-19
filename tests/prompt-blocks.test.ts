import { describe, expect, it } from "vitest";
import { promptBlocksToText } from "../src/agent.js";

describe("promptBlocksToText", () => {
  it("inlines a file selection resource ahead of the user sentence", () => {
    const text = promptBlocksToText([
      {
        type: "resource",
        resource: {
          uri: "file:///workspace/chapter11/cta_scheduler.cu#L8:20",
          text: "int cta = blockIdx.x;",
        },
      },
      { type: "text", text: "讲一下这段的原理" },
    ]);
    expect(text).toContain(
      '<ide_selection path="/workspace/chapter11/cta_scheduler.cu" lines="8-20">',
    );
    expect(text).toContain("int cta = blockIdx.x;");
    expect(text.indexOf("<ide_selection")).toBeLessThan(text.indexOf("讲一下这段的原理"));
  });

  it("keeps a path-only resource_link so the model can Read it", () => {
    const text = promptBlocksToText([
      {
        type: "resource_link",
        name: "cta_scheduler.cu (8:8)",
        uri: "file:///workspace/cta_scheduler.cu#L8:8",
      },
      { type: "text", text: "explain this" },
    ]);
    expect(text).toContain('<ide_selection path="/workspace/cta_scheduler.cu" lines="8" />');
    expect(text).toContain("explain this");
  });

  it("passes clipboard and memory bodies through unchanged", () => {
    const text = promptBlocksToText([
      {
        type: "resource",
        resource: { uri: "zed:///agent/clipboard", text: "<clipboard>\nhello\n</clipboard>" },
      },
      {
        type: "resource",
        resource: { uri: "vscode-acp:///memory/claude", text: "<user-persona>\narchitect\n</user-persona>" },
      },
      { type: "text", text: "go on" },
    ]);
    expect(text).toContain("<clipboard>\nhello\n</clipboard>");
    expect(text).toContain("<user-persona>\narchitect\n</user-persona>");
    expect(text).not.toContain("<ide_selection");
  });

  it("marks a directory link without pretending it is a file", () => {
    const text = promptBlocksToText([
      { type: "resource_link", name: "src/", uri: "file:///workspace/src/" },
      { type: "text", text: "what is in here" },
    ]);
    expect(text).toContain('<directory path="/workspace/src" />');
  });
});
