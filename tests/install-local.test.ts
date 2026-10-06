import { existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  BIN_NAMES,
  installLaunchers,
  markExecutable,
  windowsCmdShim,
  writeWindowsShim,
} from "../scripts/install-lib.mjs";

describe("windowsCmdShim", () => {
  it("invokes node with the repo wrapper and forwards argv", () => {
    const wrapper = "E:\\projects\\claude-code-acp\\bin\\claude-code-acp.js";
    const shim = windowsCmdShim(wrapper);
    expect(shim).toContain("@echo off");
    expect(shim).toContain(`node "${wrapper}" %*`);
    expect(shim).not.toContain("%%*");
  });
});

describe("installLaunchers", () => {
  it("writes .cmd shims when platform is win32", () => {
    const dir = mkdtempSync(join(tmpdir(), "cca-win-"));
    const wrapper = join(dir, "claude-code-acp.js");
    writeFileSync(wrapper, "#!/usr/bin/env node\n", "utf8");
    try {
      const installed = installLaunchers({
        platform: "win32",
        localBin: dir,
        wrapperPath: wrapper,
      });
      expect(installed).toEqual(BIN_NAMES.map((name) => join(dir, `${name}.cmd`)));
      for (const name of BIN_NAMES) {
        expect(readFileSync(join(dir, `${name}.cmd`), "utf8")).toBe(windowsCmdShim(wrapper));
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it.skipIf(process.platform === "win32")("symlinks the wrapper when platform is not win32", () => {
    const dir = mkdtempSync(join(tmpdir(), "cca-unix-"));
    const wrapper = join(dir, "claude-code-acp.js");
    writeFileSync(wrapper, "#!/usr/bin/env node\n", "utf8");
    try {
      const installed = installLaunchers({
        platform: "linux",
        localBin: dir,
        wrapperPath: wrapper,
      });
      expect(installed).toEqual(BIN_NAMES.map((name) => join(dir, name)));
      expect(existsSync(join(dir, "claude-code-acp"))).toBe(true);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("writeWindowsShim", () => {
  it("creates a named .cmd next to local bin", () => {
    const dir = mkdtempSync(join(tmpdir(), "cca-shim-"));
    const wrapper = join(dir, "wrapper.js");
    writeFileSync(wrapper, "", "utf8");
    try {
      const dest = writeWindowsShim(dir, "claude-code-acp", wrapper);
      expect(dest).toBe(join(dir, "claude-code-acp.cmd"));
      expect(readFileSync(dest, "utf8")).toBe(windowsCmdShim(wrapper));
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("markExecutable", () => {
  it("applies 0o755 through Node so build does not call chmod(1)", () => {
    const dir = mkdtempSync(join(tmpdir(), "cca-chmod-"));
    const file = join(dir, "index.js");
    writeFileSync(file, "export {}\n", "utf8");
    try {
      markExecutable(file);
      expect(existsSync(file)).toBe(true);
      if (process.platform !== "win32") {
        expect(statSync(file).mode & 0o111).toBeTruthy();
      }
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});
