import { describe, it, expect } from "vitest";
import { ClaudeRunner, isProxyBaseUrl } from "../src/claude-runner.js";

// buildExtraArgs is private on ClaudeRunner; this targeted suite exercises it
// on the real module WITHOUT mocking claude-runner (the watchdog suite mocks
// that module and must stay separate).
function extraArgsFor(base: string | undefined, keep?: string, drop?: string): string[] {
  const savedBase = process.env.ANTHROPIC_BASE_URL;
  const savedKeep = process.env.CLAUDE_ACP_KEEP_WEB_TOOLS;
  const savedDrop = process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS;
  try {
    if (base === undefined) delete process.env.ANTHROPIC_BASE_URL;
    else process.env.ANTHROPIC_BASE_URL = base;
    if (keep === undefined) delete process.env.CLAUDE_ACP_KEEP_WEB_TOOLS;
    else process.env.CLAUDE_ACP_KEEP_WEB_TOOLS = keep;
    if (drop === undefined) delete process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS;
    else process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS = drop;
    const runner = new ClaudeRunner();
    return (runner as any).buildExtraArgs(undefined, undefined);
  } finally {
    if (savedBase === undefined) delete process.env.ANTHROPIC_BASE_URL;
    else process.env.ANTHROPIC_BASE_URL = savedBase;
    if (savedKeep === undefined) delete process.env.CLAUDE_ACP_KEEP_WEB_TOOLS;
    else process.env.CLAUDE_ACP_KEEP_WEB_TOOLS = savedKeep;
    if (savedDrop === undefined) delete process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS;
    else process.env.CLAUDE_ACP_DISALLOWED_WEB_TOOLS = savedDrop;
  }
}

describe("isProxyBaseUrl", () => {
  it("treats unset as direct (keeps web tools)", () => {
    expect(isProxyBaseUrl(undefined)).toBe(false);
  });
  it("treats anthropic.com and subdomains as direct", () => {
    expect(isProxyBaseUrl("https://api.anthropic.com")).toBe(false);
    expect(isProxyBaseUrl("https://my-gateway.anthropic.com/v1")).toBe(false);
  });
  it("treats anything else as a proxy (drops web tools)", () => {
    expect(isProxyBaseUrl("http://127.0.0.1:8317")).toBe(true);
    expect(isProxyBaseUrl("https://proxy.corp.example.com/v1")).toBe(true);
    expect(isProxyBaseUrl("anthropic.com.evil.example")).toBe(true);
    expect(isProxyBaseUrl("not a url")).toBe(true);
  });
});

describe("buildExtraArgs drops server-side web tools in ACP mode (every route)", () => {
  it("adds --disallowedTools WebSearch for a proxy base URL", () => {
    const args = extraArgsFor("http://127.0.0.1:8317");
    const i = args.indexOf("--disallowedTools");
    expect(i).toBeGreaterThanOrEqual(0);
    expect(args[i + 1]).toBe("WebSearch");
    // WebFetch is executed client-side by the CLI and returns real results
    // through the proxy — it must stay available.
    expect(args[i + 1]).not.toContain("WebFetch");
  });

  it("drops WebSearch on a direct Anthropic route too (ACP mode never exposes it)", () => {
    for (const base of ["https://api.anthropic.com", undefined]) {
      const args = extraArgsFor(base);
      const i = args.indexOf("--disallowedTools");
      expect(i).toBeGreaterThanOrEqual(0);
      expect(args[i + 1]).toBe("WebSearch");
    }
  });

  it("honors the CLAUDE_ACP_KEEP_WEB_TOOLS=1 escape hatch on every route", () => {
    expect(extraArgsFor("http://127.0.0.1:8317", "1")).not.toContain("--disallowedTools");
    expect(extraArgsFor("https://api.anthropic.com", "1")).not.toContain("--disallowedTools");
  });

  it("drops WebFetch too when CLAUDE_ACP_DISALLOWED_WEB_TOOLS names it (network-blocked hosts)", () => {
    const args = extraArgsFor("http://127.0.0.1:8317", undefined, "WebSearch,WebFetch");
    const i = args.indexOf("--disallowedTools");
    expect(i).toBeGreaterThanOrEqual(0);
    expect(args[i + 1]).toBe("WebSearch,WebFetch");
  });
});
