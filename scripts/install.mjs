#!/usr/bin/env node
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { installLocal } from "./install-lib.mjs";

const resolvedRepo = dirname(dirname(fileURLToPath(import.meta.url)));
console.log(`Building claude-code-acp from ${resolvedRepo}...`);
const result = await installLocal({ repoDir: resolvedRepo });
const version = JSON.parse(readFileSync(join(resolvedRepo, "package.json"), "utf8")).version;
console.log(`Installed claude-code-acp v${version}`);
for (const path of result.installed) {
  console.log(`  ${path} -> ${result.wrapperPath}`);
}
for (const path of result.extra) {
  console.log(`  Also relinked: ${path}`);
}
console.log("Reload VS Code / Cursor to use the new claude-agent-acp.");
