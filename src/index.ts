import { Readable, Writable } from "node:stream";
import {
  AgentSideConnection,
  ndJsonStream,
} from "@agentclientprotocol/sdk";
import { createClaudeCodeAgent, loadModelContextWindows } from "./agent.js";
import { logger } from "./logger.js";

export { createClaudeCodeAgent } from "./agent.js";
export { loadModelContextWindows } from "./agent.js";
export { ClaudeRunner } from "./claude-runner.js";
export { SessionStore } from "./session-store.js";
export type { SessionInfo } from "./session-store.js";
export { loadConfig } from "./config.js";
export type { AgentConfig } from "./config.js";
export type { ClaudeResult, StreamEvent, McpServerConfig } from "./claude-runner.js";

export function startAgent(): AgentSideConnection {
  const input = Writable.toWeb(process.stdout);
  const output = Readable.toWeb(
    process.stdin
  ) as ReadableStream<Uint8Array>;
  const stream = ndJsonStream(input, output);

  const connection = new AgentSideConnection(
    (conn) => createClaudeCodeAgent(conn),
    stream
  );

  logger.info("Agent started, waiting for connections on stdio...");

  // Fetch per-model context windows from ericai-proxy /v1/models.
  // Non-blocking — falls back to 200k default if the proxy is unreachable.
  loadModelContextWindows().catch(() => {});

  connection.closed.then(() => {
    logger.info("Connection closed.");
    process.exit(0);
  });

  process.on("SIGINT", () => {
    logger.info("Received SIGINT, shutting down...");
    process.exit(0);
  });

  process.on("SIGTERM", () => {
    logger.info("Received SIGTERM, shutting down...");
    process.exit(0);
  });

  return connection;
}

// Auto-start when run directly
startAgent();
