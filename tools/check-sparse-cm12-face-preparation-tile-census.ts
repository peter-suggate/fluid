#!/usr/bin/env node
/** Static contract for the construction-only FTC1 observational census. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS,
  createSparseCM12FacePreparationTileCensusInitialWords,
  createSparseCM12FacePreparationTileCensusLayout,
  inspectSparseCM12FacePreparationTileCensusQA,
} from "../lib/methods/adaptive-mass/sparse-cm12-face-preparation-tile-census";

const root = resolve(import.meta.dirname, "..");
const read = (path: string) => readFileSync(resolve(root, path), "utf8");
const resident = read("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts");
const wgsl = read(
  "lib/methods/adaptive-mass/sparse-cm12-face-preparation-tile-census.wgsl.ts");
const sharedWGSL = read("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts");
const solver = read("lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts");
const probe = read("tools/probe-sparse-cm12-temporal-seed-ab.ts");

const layout = createSparseCM12FacePreparationTileCensusLayout({
  baseWords: 3, rowCapacity: 2051,
});
assert.equal(layout.baseWords % 64, 0);
assert.equal(layout.rowBitWordCount, Math.ceil(2051 / 32));
assert(layout.fullRowBitsBaseWords >= layout.headerBaseWords
  + SPARSE_CM12_FACE_PREPARATION_TILE_CENSUS_HEADER_WORDS);
assert(layout.tileRowBitsBaseWords >= layout.fullRowBitsBaseWords
  + layout.rowBitWordCount);
assert(layout.fullSourceCellBaseWords >= layout.tileRowBitsBaseWords
  + layout.rowBitWordCount);
assert(layout.priorAuthorityBaseWords >= layout.fullSourceCellBaseWords
  + layout.rowCapacity);
const initial = createSparseCM12FacePreparationTileCensusInitialWords(layout);
const receipt = inspectSparseCM12FacePreparationTileCensusQA(initial);
assert.equal(receipt.omittedChangedRowCount, 0);
assert.equal(receipt.firstWitness, undefined);
assert.equal(receipt.maskPopcountBySpanRung.length, 25);

for (const entry of ["beginSparseCM12FacePreparationTileCensus",
  "clearSparseCM12FacePreparationTileCensus",
  "markSparseCM12FacePreparationTileCensus",
  "finalizeSparseCM12FacePreparationTileCensus",
  "finalizeSparseCM12FacePreparationTileCensusWitness"]) {
  assert(wgsl.includes(`fn ${entry}`), `FTC1 lacks ${entry}`);
  assert(resident.includes(`"${entry}"`), `resident lacks ${entry} pipeline`);
}
assert.match(wgsl, /FTC_PRIOR_AUTHORITY\+id\],fpaPreparedAuthorityBits\(id\)/,
  "prior prepared authority must be captured before unchanged FPA execution");
assert.match(wgsl,
  /let full=.*FTC_FULL_BITS[\s\S]*let tile=.*FTC_TILE_BITS[\s\S]*if\(!full\|\|tile\)\{return;\}/,
  "omission must mean full=1 and tile=0");
assert.match(wgsl,
  /let current=fpaPreparedAuthorityBits\(row\);[\s\S]*let prior=atomicLoad/,
  "full-only rows must compare current full output bits with the prior snapshot");
assert.match(wgsl, /atomicMin\(&topologyArena\[FTC_FULL_SOURCE\+row\],cell\)/,
  "source-cell witness must use a deterministic minimum");
assert.match(wgsl,
  /if\(row>=FTC_ROW_CAPACITY\)\{[\s\S]*continue;[\s\S]*FTC_FULL_SOURCE\+row/,
  "an invalid incidence row must fault and continue before source indexing");
assert.match(wgsl,
  /if\(row>=FTC_ROW_CAPACITY\)\{[\s\S]*continue;\s*\}\s*if\(!fpaPreparationRowLive\(row\)\)\{continue;\}\s*_=ftcInsertRow\(FTC_FULL_BITS/,
  "row bitsets and witnesses must use the exact production FPA live domain");
assert.doesNotMatch(wgsl, /FTC_FULL_SOURCE\+row\],cell\);[\s\S]{0,40}else/,
  "source-cell publication must not race a first-writer store against atomicMin");
assert.match(wgsl,
  /finalizeSparseCM12FacePreparationTileCensusWitness[\s\S]*firstWitnessRow/,
  "witness fields require a singleton pass after parallel minimum selection");
assert.match(wgsl, /aggregate cause of the marked tiles in the source brick/,
  "witness cause must be labeled as aggregate brick cause");
assert.match(wgsl, /let fallback=span>1u\|\|partial\|\|topologyFallback;/,
  "macro, partial, and topology changes must stay on the full shadow set");

const censusConstruction = resident.indexOf(
  "const facePreparationTileCensusLayout = temporalSeedModeForQA === undefined");
assert(censusConstruction >= 0,
  "ordinary construction must omit FTC1 resources");
assert.match(resident,
  /size: Math\.max\(4, facePreparationTileCensusLayout\?\.totalBytes[\s\S]*faceProjectionAuthorityLayout\.totalBytes\)/,
  "only immutable QA construction may extend the topology arena");
const snapshotAt = resident.indexOf('dispatch("clearSparseCM12FacePreparationTileCensus"');
const productionBeginAt = resident.indexOf('dispatch("beginSparseCM12FacePreparationAuthority"');
const productionFinalizeAt = resident.indexOf(
  'dispatch("finalizeSparseCM12FacePreparationExecution"');
const compareAt = resident.indexOf(
  'dispatch("finalizeSparseCM12FacePreparationTileCensus"');
const witnessAt = resident.indexOf(
  'dispatch("finalizeSparseCM12FacePreparationTileCensusWitness"');
assert(snapshotAt >= 0 && productionBeginAt > snapshotAt
  && productionFinalizeAt > productionBeginAt && compareAt > productionFinalizeAt
  && witnessAt > compareAt,
"FTC1 must snapshot before and compare after unchanged full FPA execution");
assert(sharedWGSL.includes("facePreparationTileCensusLayout?:"),
  "shared WGSL constructor must make FTC1 optional");
assert(solver.includes("readFacePreparationTileCensusQA"));
assert(probe.includes("readFacePreparationTileCensusQA"));
assert.match(probe, /omittedChangedRowCount === 0/,
  "bounded probe must stop on an omitted changed row");

// Complete B16 dyadic ladder: every cell minimum maps to exactly one 4^3 tile.
for (const resolution of [1, 2, 4, 8, 16]) {
  const scale = 16 / resolution;
  const minima = new Set<number>();
  for (let z = 0; z < resolution; z += 1)
    for (let y = 0; y < resolution; y += 1)
      for (let x = 0; x < resolution; x += 1) {
        const tile = Math.floor(x * scale / 4)
          + 4 * (Math.floor(y * scale / 4) + 4 * Math.floor(z * scale / 4));
        assert(tile >= 0 && tile < 64);
        minima.add(tile * 4096 + x + resolution * (y + resolution * z));
      }
  assert.equal(minima.size, resolution ** 3);
}

// Duplicate raw incidences and inactive template rows remain part of the visit
// census, but only the exact production-live domain may enter row authority.
const liveRowFixture = [
  { row: 3, live: true, selected: false },
  { row: 3, live: true, selected: true },
  { row: 5, live: true, selected: false },
  { row: 7, live: false, selected: true },
] as const;
const fixtureFull = new Set(liveRowFixture.filter(({ live }) => live)
  .map(({ row }) => row));
const fixtureTile = new Set(liveRowFixture.filter(({ live, selected }) =>
  live && selected).map(({ row }) => row));
assert.deepEqual([...fixtureFull].sort((a, b) => a - b), [3, 5]);
assert.deepEqual([...fixtureTile], [3]);
assert.deepEqual([...fixtureFull].filter((row) => !fixtureTile.has(row)), [5]);
assert.equal(liveRowFixture.length, 4, "raw incidence visits remain unfiltered");

console.log(JSON.stringify({ passed: true,
  contract: "sparse-cm12-face-preparation-tile-census",
  constructionOnly: true, productionExecutionChanged: false,
  counters: ["dirtyBricks", "maskPopcountBySpanRung", "fullCells",
    "selectedCells", "incidenceVisits", "uniqueRows", "fallbacks",
    "omittedChangedRowCount", "firstWitness"],
}, null, 2));
