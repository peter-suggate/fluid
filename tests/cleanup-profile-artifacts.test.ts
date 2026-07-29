import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  assertSafeScratchPath,
  cleanupRootsForTemporaryDirectory,
} from "../tools/cleanup-profile-artifacts";

test("artifact cleanup accepts only Instruments names in the two private scratch roots", () => {
  const base = mkdtempSync(join(tmpdir(), "fluid-cleanup-roots-"));
  try {
    const temporary = join(base, "T");
    const cache = join(base, "C/com.apple.dt.InstrumentsCLI/path_manager");
    mkdirSync(temporary, { recursive: true });
    mkdirSync(cache, { recursive: true });
    const roots = cleanupRootsForTemporaryDirectory(temporary);

    assert.doesNotThrow(() => assertSafeScratchPath(
      join(temporary, "instrumentsABC123.ktrace"), roots));
    assert.doesNotThrow(() => assertSafeScratchPath(
      join(cache, "xrtmp__12345.opaque"), roots));
    assert.throws(() => assertSafeScratchPath(join(temporary, "notes.txt"), roots), /refusing/);
    assert.throws(() => assertSafeScratchPath(join(base, "outside.ktrace"), roots), /refusing/);
  } finally {
    rmSync(base, { recursive: true, force: true });
  }
});
