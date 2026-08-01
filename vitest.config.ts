import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    // Scope test discovery to this repo's own tests. Without this, vitest
    // also picks up vendored skill tests under .codex/skills/**, which are
    // unrelated to claude-code-acp and fail noisily.
    include: ["tests/**/*.test.ts"],
    setupFiles: ["tests/setup.ts"],
  },
});
