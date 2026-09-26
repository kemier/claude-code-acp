import { defineConfig } from "tsup";

export default defineConfig({
  // index.ts is the ACP agent entry. jev-mcp-stdio.ts is a standalone stdio MCP
  // server exposing the jev_judge tool, spawned by vscode-acp (not the agent)
  // as a per-session MCP server when acp.jev.enabled — see the central Jev
  // wiring. It has its own shebang and is run directly.
  entry: ["src/index.ts", "src/jev-mcp-stdio.ts"],
  format: ["esm"],
  target: "node20",
  outDir: "dist",
  clean: true,
  sourcemap: true,
  dts: true,
});
