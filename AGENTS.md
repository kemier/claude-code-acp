# AGENTS.md — claude-code-acp

## What This Repo Is

`claude-code-acp` is an ACP (Agent Client Protocol) bridge for Claude Code CLI. It exposes Claude Code as an ACP agent over stdio JSON-RPC, enabling ACP-compatible clients (vscode-acp, Zed, JetBrains) to use Claude Code.

The upstream repo (harukitosa/claude-code-acp) is designed for Pro/Max subscription auth. This fork adds **EricAI proxy integration** — routing Claude Code through CCR (Claude Code Router) → ericai-proxy → EricAI vLLM backend (GLM-5.2, Qwen3, DeepSeek, etc.) instead of Anthropic's API.

## Architecture

```
vscode-acp / Zed / JetBrains
  └─ stdio JSON-RPC (ACP)
      └─ claude-code-acp (this repo)
          └─ spawns: claude -p --output-format stream-json --verbose
              └─ Claude Code CLI
                  └─ ANTHROPIC_BASE_URL → CCR gateway (127.0.0.1:3457)
                      └─ ericai-proxy (127.0.0.1:8317)
                          └─ EricAI vLLM (GLM-5.2-FP8)
```

### Key Difference from Upstream

| Aspect | Upstream | This Fork |
|--------|----------|-----------|
| Auth | Pro/Max subscription (deletes `ANTHROPIC_API_KEY`) | API key via CCR gateway (keeps env vars) |
| Models | Claude models (sonnet, opus) | EricAI models (glm-5.2, Qwen3-Coder, etc.) |
| Thinking | Claude's native thinking | GLM-5.2 thinking via `enable_thinking` + `reasoning_effort` |
| Config pickers | Mode only | Mode + Model + Effort (model-aware) |
| Slash commands | None | `/ps`, `/goal`, `/skills`, `/status`, `/clear`, `/compact` |
| Steering | Not supported | `session/steer` custom notification |

## Branch Strategy

```
main (origin/main)              ← tracks upstream, never push directly
  └── local/dev                 ← non-EricAI development (steering, general fixes)
       └── ericai-proxy (HEAD)  ← EricAI-specific changes (proxy integration, pickers, commands)
```

- **`main`**: Clean sync with upstream. Use for pulling updates.
- **`local/dev`**: General improvements that work without EricAI (steering support, slash commands, bug fixes). Backport-able to upstream.
- **`ericai-proxy`**: EricAI-specific changes (model pickers, effort levels, proxy env vars, CCR integration). Built on top of `local/dev`.

### Commit Workflow

1. **Non-EricAI changes** (general ACP improvements, bug fixes):
   - Commit on `local/dev`
   - Then `git rebase local/dev` on `ericai-proxy` to port them
2. **EricAI changes** (model picker, proxy integration, effort levels):
   - Commit directly on `ericai-proxy`
3. **Pulling upstream updates**:
   - `git checkout main && git pull origin main`
   - `git checkout local/dev && git rebase main`
   - `git checkout ericai-proxy && git rebase local/dev`

## Key Files

| File | Purpose |
|------|---------|
| `src/agent.ts` | ACP agent implementation — session lifecycle, slash commands, config pickers, steering, goal management |
| `src/claude-runner.ts` | Spawns and manages `claude` CLI subprocess, parses stream-json output |
| `src/session-store.ts` | In-memory + persisted session state (cwd, mode, config overrides, Claude session IDs) |
| `src/config.ts` | Environment-based config (model, max turns, timeout, allowed tools) |
| `src/index.ts` | Entry point — starts stdio JSON-RPC connection |
| `src/logger.ts` | Logging utility |
| `src/validation.ts` | Input validation (cwd, MCP commands) |
| `bin/claude-code-acp.js` | Binary entry point — sets `CLAUDE_ACP_SKIP_PERMISSIONS=true` by default |
| `tsup.config.ts` | Build config (ESM output to `dist/index.js`) |

## Development

### Build & Run
```bash
npm install
npm run build          # tsup → dist/index.js
npm test               # vitest
npm run test:watch     # watch mode
```

### Local Binary
The binary at `~/.local/bin/claude-code-acp` imports directly from this repo's `dist/index.js`:
```javascript
import("/workspace/git/ezennin/claude-code-acp/dist/index.js");
```
So rebuilding (`npm run build`) immediately updates the installed binary — no copy step needed.

### CCR Wrapper
`~/.local/bin/claude-code-acp-ccr` is a wrapper that:
1. Reads the CCR gateway key from `~/.claude-code-router/gateway.config.json`
2. Sets `ANTHROPIC_API_KEY`, `ANTHROPIC_BASE_URL=http://127.0.0.1:3457`, `ANTHROPIC_MODEL=glm-5.2`
3. Disables all telemetry (`CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`, etc.)
4. Execs `claude-code-acp`

## EricAI Integration Details

### Model Picker
- `PROXY_MODELS` array in `agent.ts` lists all 9 EricAI models
- Must match models in `ericai-proxy/config.yaml` and `~/.codex/model_catalog.json`
- Selected model is passed as `--model <id>` to Claude CLI

### Effort Picker (Model-Aware)
- `MODEL_EFFORTS` map in `agent.ts` defines per-model supported effort levels
- GLM-5.2/GLM-5.1/Qwen3-Coder: low, medium, high
- DeepSeek-V4-Pro: low, medium, high, xhigh
- Kimi/MiMo: low, high
- Qwen2.5-14B-1M: [] (non-reasoning — effort picker hidden)
- Effort is passed as `--effort <level>` to Claude CLI
- Claude CLI sends `reasoning_effort` in request body → CCR → proxy
- Proxy reads `reasoning_effort` and sets `enable_thinking: true/false` for the model
- When thinking is enabled, proxy rewrites `delta.reasoning` → `delta.content` in SSE stream (GLM-5.2 puts output in `reasoning` field)

### Slash Commands
| Command | Behavior |
|---------|----------|
| `/ps` | Lists background processes (always empty — claude-code-acp doesn't track them) |
| `/status` | Shows model, mode, cwd, session ID, current goal |
| `/skills` | Scans `~/.claude/skills/`, `<cwd>/.agents/skills/`, parent `.agents/skills/` |
| `/goal <text>` | Sets persistent goal, starts LLM turn immediately to pursue it |
| `/goal clear` | Clears the goal |
| `/goal pause` | Pauses goal injection (goal kept but not prepended) |
| `/goal resume` | Resumes goal injection |
| `/clear` | Drops Claude session ID, starts fresh conversation |
| `/compact` | Asks LLM to summarize, then starts fresh session with summary |

Commands are advertised via `available_commands_update` notification (vscode-acp ignores the `availableCommands` field in `newSession` response).

### Steering
Custom ACP notification `session/steer` allows mid-turn message injection. Since Claude CLI doesn't support mid-turn injection, steer messages are queued and delivered as follow-up `--resume` prompts after the current turn finishes. A `user_message_chunk` is emitted immediately for visual feedback.

### Goal System
When a goal is set and active, every subsequent prompt is prepended with `[Goal: <objective>]`. The `/goal <text>` command sets the goal and immediately starts an LLM turn (mirrors codex-acp's `runGoalSet()`).

### sanitizeEnv()
`claude-runner.ts:sanitizeEnv()` keeps `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` in the subprocess env. The upstream repo deletes these to force subscription auth. In our setup, they're needed to route through CCR → ericai-proxy.

## Code Style

- TypeScript ESM (`"type": "module"`)
- Build with tsup
- Test with vitest (TDD preferred)
- No `println!` equivalent — use `logger.info()`, `logger.debug()`, etc.
- `any` casts are used for ACP notification types not yet in the SDK

## Related Repositories

| Repo | Path | Description |
|---|---|---|
| **ericai-proxy** | `../ericai-proxy` | Rust proxy translating OpenAI Responses/Chat Completions API ↔ EricAI vLLM. The `handle_chat_completions` passthrough injects `enable_thinking` and rewrites `reasoning` → `content`. |
| **vscode-acp** | `../vscode-acp` | VS Code extension implementing ACP client. Renders chat UI, spawns agent binaries. Agent config in VS Code `settings.json` under `acp.agents`. |
| **codex-acp** | `../codex-acp` | ACP adapter for Codex CLI. Reference implementation for slash commands, goal system, skill discovery. |
| **ericli** | `../ericli` | EricAI CLI — alternative ACP agent. Has its own skill system. |

## Configuration

### Environment Variables

| Variable | Description | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | CCR gateway key (set by `claude-code-acp-ccr` wrapper) | — |
| `ANTHROPIC_BASE_URL` | CCR gateway URL (`http://127.0.0.1:3457`) | — |
| `ANTHROPIC_MODEL` | Default model (`glm-5.2`) | — |
| `CLAUDE_ACP_MODEL` | Override default model | — |
| `CLAUDE_ACP_ALLOWED_TOOLS` | Comma-separated allowed tools | — |
| `CLAUDE_ACP_MAX_TURNS` | Max agentic turns per prompt | unlimited |
| `CLAUDE_ACP_TIMEOUT` | Timeout in ms | 300000 (5 min) |
| `CLAUDE_ACP_SKIP_PERMISSIONS` | Skip permission prompts | true (set in bin entrypoint) |
| `LOG_LEVEL` | Log verbosity | info |
| `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` | Disable telemetry | 1 (set by wrapper) |

### vscode-acp Agent Config

In VS Code `settings.json`:
```json
{
  "acp.agents": {
    "Claude Code": {
      "command": "/home/ezennin/.local/bin/claude-code-acp-ccr",
      "args": [],
      "env": {},
      "displayName": "Claude Code"
    }
  }
}
```

## Commit Convention

Use conventional commits with gitemojis (Ericsson internal guideline):
- `feat:` new feature (e.g. `feat: add model-aware effort picker`)
- `fix:` bug fix (e.g. `fix: /skills reads Claude Code skills`)
- `refactor:` code restructuring
- `docs:` documentation
- Lowercase, imperative mood, no trailing period

### Security
- This is a public repository fork — never commit secrets, API keys, or internal TODO lists
- Security fix details stay local only (commit message: "fix security issue in X")
- `.env` files must never be committed

## Build and Deploy

```bash
# Local Mac
./scripts/install.sh
# or: npm run install:local

# Remote workstation (rsync + build + install)
ACP_SSH_HOST=workstation ./scripts/install-remote.sh
# or: npm run install:remote

# Both
npm run deploy
```

Installs `claude-code-acp` and `claude-agent-acp` into `~/.local/bin` (symlinks to `bin/claude-code-acp.js`).
