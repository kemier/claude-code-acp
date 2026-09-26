#!/usr/bin/env node
/**
 * jev-mcp-stdio — a minimal stdio MCP server exposing one `jev_judge` tool.
 *
 * This is the CENTRAL-WIRING vehicle for giving Claude Code (claude-code-acp)
 * access to Jev without per-agent code duplication. vscode-acp, when
 * `acp.jev.enabled` is true, injects this server into the child's `mcpServers`
 * list and passes JEV_BASE_URL / JEV_API_KEY / JEV_MODEL / JEV_TIMEOUT_MS via
 * the child env. The Claude CLI then sees a real `jev_judge` tool and can call
 * it mid-turn; the answer returns as a normal tool result.
 *
 * Why a hand-rolled server (no @modelcontextprotocol/sdk): adding the SDK as a
 * dependency for a single small tool is disproportionate. The MCP stdio
 * transport is plain newline-delimited JSON-RPC 2.0 — only `initialize`,
 * `tools/list`, and `tools/call` are needed. Everything else is a -32601
 * MethodNotFound, which is spec-correct.
 *
 * The judgment logic (validation + wire request + verdict rendering) is a
 * byte-for-byte port of vscode-acp's src/jev/JevJudgeTool.ts + the request
 * building from JevClient.ts, so the tool behaves identically whether reached
 * via the ACP elicitation path (other agents) or via this MCP server
 * (Claude Code). Fail-open: no key / timeout / non-2xx / bad body → an
 * isError tool result telling the model to fall back to its own reasoning.
 */

// --- Config (env only — the child is spawned with these by vscode-acp) -------

function readJevEnvConfig() {
  const env = process.env;
  return {
    baseUrl: (env.JEV_BASE_URL || 'https://api.typesafe.ai').replace(/\/+$/, ''),
    apiKey: env.JEV_API_KEY || env.TYPESAFE_API_KEY || '',
    model: env.JEV_MODEL || 'jev-latest',
    timeoutMs: parseInt(env.JEV_TIMEOUT_MS || '5000', 10) || 5000,
  };
}

const RETRY_STATUSES = new Set([429, 529]);

// --- Wire request / response -------------------------------------------------

function buildJevRequestBody(state: any, questions: any, model: string) {
  return { state, model, questions };
}

function parseJevResponse(json: any) {
  if (!json || typeof json !== 'object') return null;
  if (typeof json.model !== 'string' || !json.answers || typeof json.answers !== 'object') return null;
  return json;
}

async function decide(state: any, questions: any, cfg: any) {
  if (!cfg.apiKey) return null;
  const body = JSON.stringify(buildJevRequestBody(state, questions, cfg.model));
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const res = await fetch(`${cfg.baseUrl}/v1/systemone`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${cfg.apiKey}`,
        },
        body,
        signal: AbortSignal.timeout(cfg.timeoutMs),
      });
      if (res.ok) return parseJevResponse(await res.json());
      if (RETRY_STATUSES.has(res.status) && attempt === 0) {
        await new Promise((r) => setTimeout(r, 200));
        continue;
      }
      return null;
    } catch {
      return null; // timeout / network — fail open
    }
  }
  return null;
}

// --- Validation (mirrors vscode-acp JevJudgeTool.validateJudgeRequest) -------

function validateJudgeRequest(req: any) {
  const issues = [];
  if (!req || typeof req !== 'object') {
    return [{ path: '', message: 'jev_judge expects { state, judgements }' }];
  }
  if (req.state === undefined || req.state === null || req.state === '') {
    issues.push({ path: 'state', message: 'state is required (the material to judge)' });
  }
  if (!Array.isArray(req.judgements) || req.judgements.length === 0) {
    issues.push({ path: 'judgements', message: 'judgements must be a non-empty array' });
    return issues;
  }
  const seen = new Set();
  req.judgements.forEach((j: any, i: number) => {
    const at = `judgements[${i}]`;
    if (!j || typeof j !== 'object') {
      issues.push({ path: at, message: 'not an object' });
      return;
    }
    if (typeof j.id !== 'string' || !j.id) {
      issues.push({ path: `${at}.id`, message: 'id is required' });
    } else if (seen.has(j.id)) {
      issues.push({ path: `${at}.id`, message: `duplicate id "${j.id}"` });
    } else {
      seen.add(j.id);
    }
    if (!['choice', 'score', 'noul'].includes(j.kind)) {
      issues.push({ path: `${at}.kind`, message: 'kind must be choice | score | noul' });
    }
    if (typeof j.instructions !== 'string' || !j.instructions.trim()) {
      issues.push({ path: `${at}.instructions`, message: 'instructions is required' });
    }
    if (j.kind === 'choice') {
      const n = j.criteria && typeof j.criteria === 'object' && !Array.isArray(j.criteria)
        ? Object.keys(j.criteria).length
        : 0;
      if (n < 2) issues.push({ path: `${at}.criteria`, message: 'choice needs >= 2 options' });
      if (n > 255) issues.push({ path: `${at}.criteria`, message: 'choice supports at most 255 options' });
    }
    if (j.kind === 'score') {
      const n = Array.isArray(j.criteria) ? j.criteria.length : 0;
      if (n < 2 || n > 10) issues.push({ path: `${at}.criteria`, message: 'score needs 2–10 ordered levels' });
    }
  });
  return issues;
}

function toQuestionMap(judgements: any[]) {
  const map: Record<string, any> = {};
  for (const j of judgements) {
    if (j.kind === 'noul') {
      map[j.id] = { type: 'noul', instructions: j.instructions, criteria: j.criteria };
    } else if (j.kind === 'choice') {
      map[j.id] = { type: 'choice', instructions: j.instructions, criteria: j.criteria ?? {} };
    } else {
      map[j.id] = { type: 'score', instructions: j.instructions, criteria: j.criteria ?? [] };
    }
  }
  return map;
}

// --- Verdict rendering (mirrors vscode-acp JevJudgeTool.formatVerdicts) ------

function extractVerdicts(judgements: any[], resp: any) {
  const out = [];
  for (const j of judgements) {
    const a = resp.answers[j.id];
    if (!a) continue;
    if (a.type === 'noul') {
      out.push({ id: j.id, kind: 'noul', value: a.noul, confidence: null });
    } else if (a.type === 'choice') {
      out.push({ id: j.id, kind: 'choice', value: a.choice, confidence: a.confidence, probabilities: a.probabilities });
    } else if (a.type === 'score') {
      out.push({ id: j.id, kind: 'score', value: a.score, confidence: a.confidence, probabilities: a.probabilities });
    }
  }
  return out;
}

function formatVerdicts(verdicts: any[], opts: any = {}) {
  const low = opts.lowConfidence ?? 0.5;
  const lines = [];
  lines.push(`jev_judge (${opts.model ?? 'jev'}): ${verdicts.length} judgement(s)`);
  for (const v of verdicts) {
    const conf = v.confidence === null
      ? ''
      : ` confidence=${v.confidence.toFixed(2)}${v.confidence < low ? ' LOW-CONFIDENCE — escalate' : ''}`;
    if (v.kind === 'noul') {
      const p = v.value;
      lines.push(`- ${v.id}: P(yes)=${p.toFixed(2)}${p >= 1 - low ? ' → yes' : p <= low ? ' → no' : ' → uncertain — escalate'}`);
    } else if (v.kind === 'choice') {
      lines.push(`- ${v.id}: ${v.value}${conf}`);
      if (v.probabilities) {
        const dist = Object.entries(v.probabilities as Record<string, number>)
          .sort((a, b) => b[1] - a[1])
          .map(([k, p]) => `${k}=${p.toFixed(2)}`)
          .join(' ');
        lines.push(`    dist: ${dist}`);
      }
    } else {
      lines.push(`- ${v.id}: score=${typeof v.value === 'number' ? v.value.toFixed(2) : v.value}${conf}`);
    }
  }
  if (opts.inputTokens !== undefined) {
    lines.push(`usage: ${opts.inputTokens} input tokens (output free)`);
  }
  return lines.join('\n');
}

// --- Tool execution ----------------------------------------------------------

async function executeJudgeTool(params: any) {
  const req = params || {};
  const issues = validateJudgeRequest(req);
  if (issues.length > 0) {
    return `jev_judge: invalid request\n${issues.map((i) => `- ${i.path || '(root)'}: ${i.message}`).join('\n')}`;
  }
  const cfg = readJevEnvConfig();
  if (!cfg.apiKey) {
    return 'jev_judge: no API key — set JEV_API_KEY (or TYPESAFE_API_KEY). Fall back to your own reasoning.';
  }
  const resp = await decide(req.state, toQuestionMap(req.judgements), cfg);
  if (!resp) {
    return 'jev_judge: Jev unavailable (no API key, timeout, or upstream error). Fall back to your own reasoning.';
  }
  const verdicts = extractVerdicts(req.judgements, resp);
  if (verdicts.length === 0) {
    return 'jev_judge: Jev returned no answers.';
  }
  return formatVerdicts(verdicts, { model: resp.model, inputTokens: resp.usage?.input_tokens });
}

// --- Tool definition (what the Claude CLI sees) ------------------------------

const JEV_JUDGE_TOOL = {
  name: 'jev_judge',
  description:
    'Get a calibrated, closed-form semantic judgement from Jev (a non-LLM decision ' +
    'model, 70–500 ms) instead of reasoning it out by hand. Batch one or more ' +
    'judgements — each is a choice (which of N options), score (a level on an ' +
    'ordered 2–10 scale), or noul (probability that a yes/no question is true). ' +
    'Pass `state` (the material to judge: a diff, a file, a README pair, …) plus ' +
    '`judgements` (an array of {id, kind, instructions, criteria}). Returns a ' +
    'compact verdict per judgement with confidence; low-confidence results are ' +
    'flagged so you can escalate to your own reasoning. Use for risk ' +
    'classification, relevance, covered-vs-gap, and similar closed semantic calls.',
  inputSchema: {
    type: 'object',
    properties: {
      state: {
        type: 'string',
        description: 'The material to judge: file contents, a diff, a URL batch, a README/lecture pair, …',
      },
      judgements: {
        type: 'array',
        minItems: 1,
        description: 'What to decide. One call fans out all judgements in parallel.',
        items: {
          type: 'object',
          properties: {
            id: { type: 'string', description: 'Stable id for the answer map.' },
            kind: { type: 'string', enum: ['choice', 'score', 'noul'] },
            instructions: { type: 'string', description: 'The question ("Does this command destroy data?").' },
            criteria: {
              oneOf: [
                { type: 'object', description: 'choice: option → rubric (>= 2, <= 255 options).' },
                { type: 'array', items: { type: 'string' }, description: 'score: ordered level descriptions (2–10).' },
                { type: 'object', properties: { true: { type: 'string' }, false: { type: 'string' } }, description: 'noul: optional {true,false} rubric.' },
              ],
            },
          },
          required: ['id', 'kind', 'instructions'],
        },
      },
    },
    required: ['state', 'judgements'],
  },
};

// --- stdio JSON-RPC transport (newline-delimited, one message per line) -------

function writeMsg(obj: any) {
  process.stdout.write(JSON.stringify(obj) + '\n');
}

function handleRequest(msg: any) {
  const { id, method, params } = msg;
  switch (method) {
    case 'initialize':
      return {
        protocolVersion: params?.protocolVersion ?? '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'jev-mcp', version: '1.0.0' },
      };
    case 'notifications/initialized':
    case 'initialized':
      return null; // notification — no response
    case 'tools/list':
      return { tools: [JEV_JUDGE_TOOL] };
    case 'tools/call':
      return null; // handled async below
    default:
      return { __error: { code: -32601, message: `Method not found: ${method}` } };
  }
}

async function main() {
  let buffer = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', async (chunk) => {
    buffer += chunk;
    let idx;
    while ((idx = buffer.indexOf('\n')) >= 0) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      let msg;
      try {
        msg = JSON.parse(line);
      } catch {
        writeMsg({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } });
        continue;
      }
      try {
        if (msg.method === 'tools/call') {
          const text = await executeJudgeTool(msg.params?.arguments ?? {});
          writeMsg({ jsonrpc: '2.0', id: msg.id, result: { content: [{ type: 'text', text }], isError: false } });
        } else {
          const result = handleRequest(msg);
          if (result === null) continue; // notification
          if (result.__error) {
            writeMsg({ jsonrpc: '2.0', id: msg.id, error: result.__error });
          } else {
            writeMsg({ jsonrpc: '2.0', id: msg.id, result });
          }
        }
      } catch (e: any) {
        writeMsg({ jsonrpc: '2.0', id: msg.id ?? null, error: { code: -32603, message: `Internal error: ${e?.message ?? e}` } });
      }
    }
  });
  process.stdin.on('end', () => process.exit(0));
}

main();
