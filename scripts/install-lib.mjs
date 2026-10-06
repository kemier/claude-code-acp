/**
 * Cross-platform local install helpers for claude-code-acp.
 * Windows: .cmd shims. Linux / macOS: symlinks.
 */
import { spawnSync } from "node:child_process";
import {
  chmodSync,
  existsSync,
  lstatSync,
  mkdirSync,
  rmSync,
  symlinkSync,
  writeFileSync,
} from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const BIN_NAMES = ["claude-code-acp", "claude-agent-acp"];

export const DIST_BINS = ["dist/index.js", "dist/jev-mcp-stdio.js"];

export function windowsCmdShim(wrapperPath) {
  return [
    "@echo off",
    "set CLAUDE_ACP_SKIP_PERMISSIONS=true",
    `node "${wrapperPath}" %*`,
    "",
  ].join("\r\n");
}

export function markExecutable(filePath) {
  chmodSync(filePath, 0o755);
}

export function writeWindowsShim(localBin, name, wrapperPath) {
  const dest = join(localBin, `${name}.cmd`);
  writeFileSync(dest, windowsCmdShim(wrapperPath), "utf8");
  return dest;
}

export function linkUnix(localBin, name, wrapperPath) {
  const dest = join(localBin, name);
  rmSync(dest, { force: true });
  symlinkSync(wrapperPath, dest);
  return dest;
}

export function installLaunchers({ platform, localBin, wrapperPath }) {
  mkdirSync(localBin, { recursive: true });
  const installed = [];
  for (const name of BIN_NAMES) {
    if (platform === "win32") {
      installed.push(writeWindowsShim(localBin, name, wrapperPath));
    } else {
      installed.push(linkUnix(localBin, name, wrapperPath));
    }
  }
  return installed;
}

function runNpm(args, cwd) {
  // `npm run` sets npm_execpath to npm's JS entry. Spawn that with node so
  // Windows does not have to launch npm.cmd (Node 20+ returns EINVAL, and
  // shell:true warns DEP0190).
  const npmExec = process.env.npm_execpath;
  const command = npmExec
    ? process.execPath
    : process.platform === "win32"
      ? "npm.cmd"
      : "npm";
  const commandArgs = npmExec ? [npmExec, ...args] : args;
  const result = spawnSync(command, commandArgs, {
    cwd,
    stdio: "inherit",
    windowsHide: true,
    shell: !npmExec && process.platform === "win32",
  });
  if (result.error) {
    throw result.error;
  }
  if (result.status !== 0) {
    throw new Error(`npm ${args.join(" ")} exited ${result.status}`);
  }
}

function lexists(path) {
  try {
    lstatSync(path);
    return true;
  } catch (err) {
    if (err && err.code === "ENOENT") return false;
    throw err;
  }
}

function relinkPackagedInstalls(home, builtJs) {
  const brewGlobal =
    "/opt/homebrew/lib/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js";
  const npmGlobal = join(
    home,
    ".npm-global/lib/node_modules/@agentclientprotocol/claude-agent-acp/dist/index.js",
  );
  const extra = [];
  for (const target of [brewGlobal, npmGlobal]) {
    if (!lexists(target)) continue;
    rmSync(target, { force: true });
    symlinkSync(builtJs, target);
    extra.push(target);
  }
  return extra;
}

export async function installLocal({
  repoDir,
  platform = process.platform,
  home = homedir(),
  localBin = process.env.LOCAL_BIN ?? join(homedir(), ".local", "bin"),
  skipBuild = false,
} = {}) {
  if (!repoDir) {
    throw new Error("repoDir is required");
  }
  const wrapperPath = join(repoDir, "bin", "claude-code-acp.js");
  const builtJs = join(repoDir, "dist", "index.js");

  if (!skipBuild) {
    if (!existsSync(join(repoDir, "node_modules"))) {
      runNpm(["install"], repoDir);
    }
    runNpm(["run", "build"], repoDir);
  }

  if (!existsSync(builtJs)) {
    throw new Error(`Build output not found: ${builtJs}`);
  }
  if (!existsSync(wrapperPath)) {
    throw new Error(`Bin wrapper not found: ${wrapperPath}`);
  }

  markExecutable(wrapperPath);
  for (const rel of DIST_BINS) {
    const abs = join(repoDir, rel);
    if (!existsSync(abs)) {
      throw new Error(`Build output not found: ${abs}`);
    }
    markExecutable(abs);
  }

  const installed = installLaunchers({ platform, localBin, wrapperPath });
  const extra = platform === "win32" ? [] : relinkPackagedInstalls(home, builtJs);

  return { wrapperPath, builtJs, installed, extra };
}
