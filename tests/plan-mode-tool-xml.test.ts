import { describe, it, expect } from "vitest";
import { PlanModeToolXmlFilter } from "../src/plan-mode-tool-xml.js";

const PLAN = "Implement all feasible deep-agent capability gaps.";

const XML = `<tool_call>
<function=ExitPlanMode>
<parameter=plan>
${PLAN}
</parameter>
</function>
</tool_call>`;

describe("PlanModeToolXmlFilter", () => {
  it("extracts a complete ExitPlanMode block and keeps the surrounding prose", () => {
    const filter = new PlanModeToolXmlFilter();
    const { text, calls } = filter.push(`Ready.\n${XML}\nThanks.`);
    expect(text).toBe("Ready.\n\nThanks.");
    expect(calls).toEqual([{ name: "ExitPlanMode", plan: PLAN }]);
    expect(text).not.toContain("<tool_call>");
    expect(text).not.toContain("ExitPlanMode");
  });

  it("holds a block split across deltas until the close tag arrives", () => {
    const filter = new PlanModeToolXmlFilter();
    const mid = XML.indexOf("<parameter=plan>");
    const first = filter.push("Before " + XML.slice(0, mid));
    expect(first.calls).toEqual([]);
    expect(first.text).toBe("Before ");
    expect(first.text).not.toContain("<tool");
    const second = filter.push(XML.slice(mid));
    expect(second.calls).toEqual([{ name: "ExitPlanMode", plan: PLAN }]);
    expect(second.text).toBe("");
  });

  it("holds a split open tag so the client never sees a partial <tool_call", () => {
    const filter = new PlanModeToolXmlFilter();
    const first = filter.push("See <too");
    expect(first.text).toBe("See ");
    expect(first.calls).toEqual([]);
    const second = filter.push("l_call>\n<function=EnterPlanMode>\n</function>\n</tool_call>");
    expect(second.text).toBe("");
    expect(second.calls).toEqual([{ name: "EnterPlanMode" }]);
  });

  it("leaves a non-plan tool_call in the text", () => {
    const filter = new PlanModeToolXmlFilter();
    const raw = "<tool_call>\n<function=Read>\n<parameter=file_path>\n/tmp/x\n</parameter>\n</function>\n</tool_call>";
    const { text, calls } = filter.push(raw);
    expect(calls).toEqual([]);
    expect(text).toBe(raw);
  });

  it("reads a JSON tool_call body", () => {
    const filter = new PlanModeToolXmlFilter();
    const raw = `<tool_call>\n{"name":"ExitPlanMode","arguments":{"plan":${JSON.stringify(PLAN)}}}\n</tool_call>`;
    const { text, calls } = filter.push(raw);
    expect(text).toBe("");
    expect(calls).toEqual([{ name: "ExitPlanMode", plan: PLAN }]);
  });

  it("on flush, extracts an ExitPlanMode block that never closed", () => {
    const filter = new PlanModeToolXmlFilter();
    const open = `<tool_call>\n<function=ExitPlanMode>\n<parameter=plan>\n${PLAN}`;
    const held = filter.push(open);
    expect(held.text).toBe("");
    expect(held.calls).toEqual([]);
    const flushed = filter.flush();
    expect(flushed.calls).toEqual([{ name: "ExitPlanMode", plan: PLAN }]);
    expect(flushed.text).toBe("");
  });

  it("on flush, releases a held prefix that is not a plan-mode call", () => {
    const filter = new PlanModeToolXmlFilter();
    const held = filter.push("tail <too");
    expect(held.text).toBe("tail ");
    expect(held.calls).toEqual([]);
    const flushed = filter.flush();
    expect(flushed.calls).toEqual([]);
    expect(flushed.text).toBe("<too");
  });
});
