#!/usr/bin/env node
/** CPU fixtures, source manifest, and standalone Naga gate for FPE1. */
import assert from "node:assert/strict";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";

import {
  SPARSE_CM12_FPA_EXACT_PRODUCER_CAUSE,
  SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY,
  SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER,
  SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC,
  SPARSE_CM12_FPA_EXACT_PRODUCER_PHASE,
  SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST,
  createSparseCM12FPAExactProducerAdapterInitialWords,
  createSparseCM12FPAExactProducerAdapterLayout,
  expandSparseCM12FPAExactProducerEvents,
  type SparseCM12FPAExactProducerEvent,
} from "../lib/methods/adaptive-mass/sparse-cm12-fpa-exact-producer-adapter";
import { createSparseCM12FPAExactProducerAdapterWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-fpa-exact-producer-adapter.wgsl";

const layout = createSparseCM12FPAExactProducerAdapterLayout({
  baseWords: 3, cellCapacity: 32, rowCapacity: 64,
});
assert.equal(layout.baseWords % 64, 0);
const initial = createSparseCM12FPAExactProducerAdapterInitialWords(layout);
const h = SPARSE_CM12_FPA_EXACT_PRODUCER_HEADER;
assert.equal(initial[h.magic], SPARSE_CM12_FPA_EXACT_PRODUCER_MAGIC);
assert.equal(initial[h.phase], SPARSE_CM12_FPA_EXACT_PRODUCER_PHASE.accepted);

const generation = 4;
const events: readonly SparseCM12FPAExactProducerEvent[] = Object.freeze([
  { family: "liquidPhaseCell", stableId: 1, generation,
    beforeBits: [0], afterBits: [1] },
  { family: "vexCell", stableId: 2, generation,
    beforeBits: [1, 2, 3, 0], afterBits: [1, 2, 4, 1] },
  { family: "sourceFaceRow", stableId: 9, generation,
    beforeBits: [0x3f800000], afterBits: [0x40000000] },
  { family: "topologyCell", stableId: 3, generation,
    beforeBits: [7, 8], afterBits: [7, 16] },
  { family: "movingSolidCell", stableId: 4, generation,
    beforeBits: [1, 2, 3, 4], afterBits: [1, 2, 3, 5] },
  { family: "movingSolidRow", stableId: 10, generation,
    beforeBits: [1, 1, 1, 1], afterBits: [1, 1, 2, 1] },
  { family: "policyRow", stableId: 11, generation,
    beforeBits: [8], afterBits: [9] },
]);
const expansion = new Map<string, readonly number[]>([
  ["liquidPhaseCell:1", [5, 3]], ["vexCell:2", [7, 5]],
  ["sourceFaceRow:9", [9]], ["topologyCell:3", [4, 3]],
  ["movingSolidCell:4", [8, 4]], ["movingSolidRow:10", [10]],
  ["policyRow:11", [11]],
]);
const receipt = expandSparseCM12FPAExactProducerEvents({ layout, generation, events,
  rowsForEvent: (event) => expansion.get(`${event.family}:${event.stableId}`) ?? [],
});
assert.deepEqual(receipt.preparationRows, [3, 4, 5, 7, 8, 9, 10, 11]);
for (const family of Object.keys(SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY) as
  Array<keyof typeof SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY>) {
  assert.equal(receipt.expectedReceipts[family], 1);
  assert.equal(receipt.coveredReceipts[family], 1);
  assert((receipt.causeMask & SPARSE_CM12_FPA_EXACT_PRODUCER_CAUSE[family]) !== 0);
}
assert.throws(() => expandSparseCM12FPAExactProducerEvents({ layout, generation,
  events: [{ ...events[0]!, beforeBits: [1], afterBits: [1] }], rowsForEvent: () => [1],
}), /does not describe an exact bit change/);
assert.throws(() => expandSparseCM12FPAExactProducerEvents({ layout, generation,
  events: [{ ...events[0]!, generation: generation + 1 }], rowsForEvent: () => [1],
}), /generation gap/);
assert.throws(() => expandSparseCM12FPAExactProducerEvents({ layout, generation,
  events: [events[0]!], rowsForEvent: () => [layout.rowCapacity],
}), /expanded row is invalid/);
assert.throws(() => expandSparseCM12FPAExactProducerEvents({ layout, generation,
  events: [events[0]!], rowsForEvent: () => [1],
  expectedReceipts: { liquidPhaseCell: 2 },
}), /coverage gap/);
assert.throws(() => expandSparseCM12FPAExactProducerEvents({ layout, generation,
  events: [events[0]!], rowsForEvent: () => [1],
  expectedReceipts: { liquidPhaseCell: 0 },
}), /receipt overflow/);

assert.equal(SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST.authority, "FPA1");
assert.equal(SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST.role,
  "producer-ingress-only");
assert(SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST.invariants.includes(
  "no accepted-domain scan"));
assert(SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST.requiredHooks.includes(
  "fpaPreparationRowLive"));
assert(SPARSE_CM12_FPA_EXACT_PRODUCER_SOURCE_MANIFEST.requiredHooks.includes(
  "fpeaVexReverseProvenanceValid"));

const library = createSparseCM12FPAExactProducerAdapterWGSL({
  layout, arenaName: "arena", prefix: "fpe",
});
for (const producer of ["fpeRecordLiquidPhaseCell", "fpeRecordVexCell",
  "fpeRecordSourceFaceRow", "fpeRecordTopologyCell", "fpeRecordMovingSolidCell",
  "fpeRecordMovingSolidRow", "fpeRecordPolicyRow"]) {
  assert(library.includes(`fn ${producer}`), `missing ${producer}`);
}
assert.match(library, /if\(!fpaPreparationRowLive\(row\)\)\{return true;\}/,
  "row membership must use the exact FPA live-domain predicate");
assert.match(library, /fpaMarkPreparationRow\(row,cause,0u,false\)/,
  "adapter must reuse FPA1 marking without fabricating per-row receipts");
assert.match(library, /fpaCoverPreparationReceipt\(\)/,
  "each completed exact event must cover one FPA1 receipt");
const closeEvent = library.slice(library.indexOf("fn fpeCloseEvent"),
  library.indexOf("fn fpeExpandIncidence"));
assert.doesNotMatch(closeEvent, /fpaCoverPreparationReceipt/,
  "producer events must not publish FPA receipts before clean adapter seal");
const seal = library.slice(library.indexOf("fn sealSparseCM12FPAExactProducerAdapter"));
assert.match(seal, /fpaCoverPreparationReceipt\(\)/,
  "clean parallel seal must publish the validated receipt count to FPA1");
assert.match(library, /if\(atomicLoad\([^\n]+\)!=0u\)\{[\s\S]*fpeFail\(9u/,
  "seal must fail closed on writes from an uncovered event");
assert.match(library, /old>=expected[\s\S]*fpeFail\(7u/,
  "per-family receipt overflow must fail closed");
assert.match(library,
  /if\(!fpeaVexReverseProvenanceValid\(\)\)[\s\S]*fpeFail\(11u/,
  "missing reverse dependency provenance must fault, never look like empty work");
assert.match(library,
  /begin==fpeInvalid\|\|end==fpeInvalid\|\|begin>end[\s\S]*fpeFail\(11u/,
  "invalid or reversed dependency ranges must fault before iteration");
assert.match(library,
  /row==fpeInvalid\|\|row>=fpeRowCapacity[\s\S]*fpeFail\(11u/,
  "malformed reverse rows must fault before FPA marking");
assert.doesNotMatch(library, /acceptedTemplate(?:Cell|Row)Invocation|dispatchWorkgroups|fallback/i,
  "adapter must not contain a global accepted scan or fallback path");

const source = /* wgsl */ `
@group(0)@binding(0)var<storage,read_write>arena:array<atomic<u32>>;
fn fpeaFrameGeneration()->u32{return 1u;}
fn fpeaExpectedProducerReceipts(family:u32)->u32{_=family;return 0u;}
fn fpeaIncidenceBegin(cell:u32)->u32{_=cell;return 0u;}
fn fpeaIncidenceEnd(cell:u32)->u32{_=cell;return 0u;}
fn fpeaIncidenceRow(at:u32)->u32{return at;}
fn fpeaVexReverseProvenanceValid()->bool{return true;}
fn fpeaVexReverseBegin(cell:u32)->u32{_=cell;return 0u;}
fn fpeaVexReverseEnd(cell:u32)->u32{_=cell;return 0u;}
fn fpeaVexReverseRow(at:u32)->u32{return at;}
fn fpaPreparationRowLive(row:u32)->bool{return row<${layout.rowCapacity}u;}
fn fpaMarkPreparationRow(row:u32,cause:u32,depth:u32,receipt:bool)->bool{
  _=cause;_=depth;_=receipt;return fpaPreparationRowLive(row);}
fn fpaCoverPreparationReceipt(){}
${library}
`;
const directory = mkdtempSync(join(tmpdir(), "fluid-fpe1-"));
try {
  const path = join(directory, "fpe1.wgsl"); writeFileSync(path, source);
  const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
} finally {
  rmSync(directory, { recursive: true, force: true });
}

console.log(JSON.stringify({ passed: true,
  contract: "sparse-cm12-fpa-exact-producer-adapter",
  families: Object.keys(SPARSE_CM12_FPA_EXACT_PRODUCER_FAMILY),
  cpuFixtureRows: receipt.preparationRows,
  naga: "PASS", authority: "FPA1", globalScan: false, fallback: false,
}, null, 2));
