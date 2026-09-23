// Un-mocked file reader. This test file (and its agent import) apply
// vi.mock("node:fs"), which stubs readFileSync for everything loaded through
// the transformed module graph — including any named readFileSync import here.
// Loading node:fs via createRequire in an untransformed .cjs bypasses that,
// so source-string assertions against src/agent.ts read the REAL file.
// (agent-enter-plan-mode.test.ts avoids this by simply not mocking node:fs.)
const { createRequire } = require("node:module");
module.exports = {
  readTextFile: (p) => createRequire(__filename)("node:fs").readFileSync(p, "utf8"),
};
