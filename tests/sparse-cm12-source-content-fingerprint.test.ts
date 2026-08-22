import assert from "node:assert/strict";
import test from "node:test";
import {
  fingerprintSparseCM12SourceEntries,
} from "../tools/sparse-cm12-source-content-fingerprint";

const bytes = (value: string): Uint8Array => Buffer.from(value, "utf8");

test("source-content fingerprint is ordered, path-sensitive, and byte-sensitive", () => {
  const first = fingerprintSparseCM12SourceEntries([
    { path: "tools/probe.ts", bytes: bytes("probe") },
    { path: "lib/shader.wgsl.ts", bytes: bytes("shader") },
  ]);
  const reordered = fingerprintSparseCM12SourceEntries([
    { path: "lib/shader.wgsl.ts", bytes: bytes("shader") },
    { path: "tools/probe.ts", bytes: bytes("probe") },
  ]);
  const changedBytes = fingerprintSparseCM12SourceEntries([
    { path: "tools/probe.ts", bytes: bytes("probe changed") },
    { path: "lib/shader.wgsl.ts", bytes: bytes("shader") },
  ]);
  const changedPath = fingerprintSparseCM12SourceEntries([
    { path: "tools/renamed-probe.ts", bytes: bytes("probe") },
    { path: "lib/shader.wgsl.ts", bytes: bytes("shader") },
  ]);

  assert.equal(first.sha256, reordered.sha256);
  assert.notEqual(first.sha256, changedBytes.sha256);
  assert.notEqual(first.sha256, changedPath.sha256);
  assert.equal(first.fileCount, 2);
  assert.equal(first.sha256.length, 64);
});
