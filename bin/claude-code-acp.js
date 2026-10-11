#!/usr/bin/env node
process.env.CLAUDE_ACP_SKIP_PERMISSIONS = process.env.CLAUDE_ACP_SKIP_PERMISSIONS ?? "true";
// Goal evaluator loop ON by default (agent.ts:5161 gate). The evaluator judges
// the /goal condition after each turn (Haiku slot, session-model fallback) and
// re-seeds a follow-up while NOT MET, or marks the goal complete when MET.
// Set CLAUDE_ACP_GOAL_EVAL=0 in the environment to opt out.
process.env.CLAUDE_ACP_GOAL_EVAL = process.env.CLAUDE_ACP_GOAL_EVAL ?? "1";
import("../dist/index.js");
