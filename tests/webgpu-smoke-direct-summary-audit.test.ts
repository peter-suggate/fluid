import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
const readbacks = readFileSync(new URL("../tools/webgpu-smoke-readbacks.ts", import.meta.url), "utf8");
const compactField = readFileSync(new URL("../tools/webgpu-smoke-compact-field.ts", import.meta.url), "utf8");

test("initial fine-summary audit reads the resident direct-summary buffers", () => {
  const start = smoke.indexOf("if (powerGenerationAuditRequested && method.id === \"octree\")");
  const end = smoke.indexOf("// Construction and t=0 publication have separate costs", start);
  assert.ok(start >= 0 && end > start, "initial generation audit must have a bounded source region");
  const audit = smoke.slice(start, end);

  for (const name of ["fineEntries", "fineReferences", "coarseRows", "rankKeys", "workState"] as const) {
    assert.match(audit, new RegExp(`globalFineSummaryDebug\\.${name}`),
      `audit must read the direct-summary ${name} buffer exposed by the solver`);
  }
  for (const retired of ["candidateDirectory", "fineCandidateDirectory", "records", "recordScratch"] as const) {
    assert.doesNotMatch(audit, new RegExp(`globalFineSummaryDebug\\.${retired}`),
      `audit must not wrap retired ${retired} state as a GPUBuffer`);
  }
  assert.match(audit, /phase: "initial-fine-summary-rank-audit"[\s\S]*firstKeyMismatch/,
    "the direct rank-to-entry invariant replaces the retired sort-order audit");
  assert.match(audit, /const highRank = Math\.min\(workState\[7\],[\s\S]*rankKeys\.size \/ 4[\s\S]*fineEntries\.size \/ 32/,
    "diagnostic readback must be bounded by both GPU-published high rank and resident buffer capacity");
});

test("compact field QA reads and validates the complete resident worklist ABI", () => {
  const start = readbacks.indexOf("async function readCubicVolumeField(");
  const end = readbacks.indexOf("\nexport async function dumpFineRedistancePageDeltaForensics", start);
  assert.ok(start >= 0 && end > start, "compact field readback must have a bounded source region");
  const readback = readbacks.slice(start, end);
  assert.match(readback,
    /readBufferBinding\(device, \{ buffer: source\.worklist \}, source\.worklist\.size\)/,
    "QA must copy the complete current seven-word-header/direct-directory/halo buffer");
  assert.doesNotMatch(readback, /\(5 \+ source\.plan\.maximumResidentBricks\) \* 4/,
    "QA must not truncate the worklist to the retired five-word prefix");
  assert.match(compactField,
    /const expectedWorklistWords = 7 \+ plan\.maximumResidentBricks \+ plan\.logicalBrickCount[\s\S]*plan\.includeHalo27 \? 27 \* plan\.maximumResidentBricks : 0/,
    "the decoder must continue to require the exact production worklist layout");
  assert.match(compactField, /if \(worklist\.length !== expectedWorklistWords\) throw new Error/,
    "a truncated or oversized QA snapshot must continue to fail closed");
});

test("fine-generation probes read the direct page directory after the current header", () => {
  const start = readbacks.indexOf("async function readGlobalFineGenerationDiagnostics(");
  const end = readbacks.indexOf("\nexport function decodeFloat16", start);
  assert.ok(start >= 0 && end > start, "fine-generation diagnostics must have a bounded source region");
  const diagnostics = readbacks.slice(start, end);
  assert.match(diagnostics,
    /readBufferBinding\(device, \{ buffer: source\.worklist \}, source\.worklist\.size\)/,
    "fine-generation probes must copy the resident direct page directory");
  assert.doesNotMatch(diagnostics, /\(5 \+ pageCapacity\) \* 4/,
    "fine-generation probes must not truncate the current seven-word worklist ABI");
  assert.match(diagnostics, /const directoryBase = 7 \+ pageCapacity/,
    "probe lookup must use the current direct-directory offset");
});
