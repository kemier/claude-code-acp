import { describe, it, expect } from "vitest";
import { splitCdPrefix } from "../src/agent.js";

describe("splitCdPrefix", () => {
  it("extracts a bare `cd <dir>` line and keeps the real command first", () => {
    const r = splitCdPrefix("cd /workspace/git/ezennin/claude-code-acp\nnpm test");
    expect(r.cwd).toBe("/workspace/git/ezennin/claude-code-acp");
    expect(r.command).toBe("npm test");
    // The client derives the displayed command from the FIRST line, so this
    // is what stops every tool call rendering as `cd /workspace/...`.
    expect(r.command.split("\n")[0].startsWith("cd ")).toBe(false);
  });

  it("keeps every subsequent line of a multi-line command", () => {
    const r = splitCdPrefix("cd /tmp/p\necho one\ncat tsconfig.json\nls");
    expect(r.cwd).toBe("/tmp/p");
    expect(r.command).toBe("echo one\ncat tsconfig.json\nls");
  });

  it("extracts an inline `cd <dir> && <cmd>` prefix", () => {
    const r = splitCdPrefix("cd /tmp/p && npm run build");
    expect(r.cwd).toBe("/tmp/p");
    expect(r.command).toBe("npm run build");
  });

  it("keeps trailing lines after an inline `cd ... && ...` prefix", () => {
    const r = splitCdPrefix("cd /tmp/p && npm run build\nnpm test");
    expect(r.cwd).toBe("/tmp/p");
    expect(r.command).toBe("npm run build\nnpm test");
  });

  it("leaves a command without a cd prefix untouched", () => {
    const r = splitCdPrefix("git status --short");
    expect(r.cwd).toBeUndefined();
    expect(r.command).toBe("git status --short");
  });

  it("does not strip a lone `cd <dir>` with no following command", () => {
    // Nothing would remain to display, so keep the cd as the command.
    const r = splitCdPrefix("cd /tmp/p");
    expect(r.cwd).toBeUndefined();
    expect(r.command).toBe("cd /tmp/p");
  });

  it("does not treat `cdk deploy` as a cd prefix", () => {
    const r = splitCdPrefix("cdk deploy\nnpm test");
    expect(r.cwd).toBeUndefined();
    expect(r.command).toBe("cdk deploy\nnpm test");
  });
});

