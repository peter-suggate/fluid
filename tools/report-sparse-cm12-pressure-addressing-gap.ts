/**
 * CPU-only source audit for the Sparse CM12 pressure-addressing hot path.
 *
 * This report proves that the former PCM rank-select hot path has been cut to
 * one immutable, generation-stamped materialized list. The load estimate is
 * the avoided source-level atomic work; it is not a GPU transaction counter.
 *
 * Run:
 *   node --import tsx tools/report-sparse-cm12-pressure-addressing-gap.ts
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const residentPath = fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", import.meta.url,
));
const wgslPath = fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url,
));
const pcmPath = fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership.wgsl.ts", import.meta.url,
));
const pabPath = fileURLToPath(new URL(
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-addressing-ab.wgsl.ts",
  import.meta.url,
));
const resident = readFileSync(residentPath, "utf8");
const wgsl = readFileSync(wgslPath, "utf8");
const pcm = readFileSync(pcmPath, "utf8");
const pab = readFileSync(pabPath, "utf8");

function functionBody(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}`);
  assert.notEqual(start, -1, `${name} is missing`);
  const opening = source.indexOf("{", start);
  assert.notEqual(opening, -1, `${name} has no body`);
  let depth = 0;
  for (let at = opening; at < source.length; at += 1) {
    if (source[at] === "{") depth += 1;
    if (source[at] === "}") depth -= 1;
    if (depth === 0) return source.slice(opening + 1, at);
  }
  throw new Error(`${name} has an unterminated body`);
}

const addressHelper = functionBody(wgsl, "pressureCellInvocation");
assert.match(addressHelper, /pabPressureCellAddress\(invocation\)/,
  "pressure-cell addressing no longer enters the construction-fixed PAB wrapper");
assert.match(addressHelper, /pcmCellContains\(cell\)/,
  "expected post-rank-select PCM membership validation is missing");
const productionPABWrapper = functionBody(wgsl, "pabPressureCellAddress");
assert.equal(productionPABWrapper.replace(/\s/g, ""),
  "returnpcmCellRankSelect(rank);",
  "layout-absent defensive wrapper must remain canonical rank-select");
assert.doesNotMatch(productionPABWrapper, /PAB_|activity|list|fallback/i,
  "layout-absent defensive wrapper unexpectedly references list state");
assert.match(resident, /pressureAddressingModeForQA \?\? "materializedList"/,
  "ordinary production is not fixed to materialized pressure addressing");
assert.match(resident, /createSparseCM12ProductionPressureAddressingLayout/,
  "ordinary production does not allocate the canonical pressure list");
assert.match(pab, /return pabLoad\(PAB_LIST_BASE\+rank\);/,
  "materialized address helper does not directly load the canonical list");
assert.match(pab, /return PAB_INVALID;/,
  "stale or faulted materialized addressing must fail closed");
assert.doesNotMatch(pab,
  /PAB_MODE_LIST[\s\S]{0,500}return pcmCellRankSelect\(rank\)/,
  "materialized production addressing must never fall back to rank select");
assert.match(wgsl,
  /fn psaPressureAddressingReady\(\)->bool\{return pabPressureAddressingReady\(\);\}/,
  "PSA does not consume the materialized-address receipt");

const directHotFunctions = [
  "preparePressure",
  "initializePCG",
  "initializeBrickAggregateDirection",
  "initializePipelinedImage",
  "updatePipelinedState",
  "applyBrickAggregatePreconditioner",
  "applyPipelinedImage",
  "restartPCGAfterCurvatureLoss",
  "initializeBrickAggregateRecoveryDirection",
  "applyPipelinedRecovery",
] as const;
for (const name of directHotFunctions) {
  assert.match(functionBody(wgsl, name), /pressureCellInvocation\(gid\.x\)/,
    `${name} no longer resolves its rank through pressureCellInvocation`);
}
assert.match(functionBody(wgsl, "measureTrueResidualWork"),
  /pressureCellInvocation\(gid\.x\)/,
  "true-residual kernels no longer resolve their rank through pressureCellInvocation");

for (const entry of [
  "countPressureCells",
  "finalizePressureCellWorklist",
  "compactPressureCells",
  "countPressureRows",
  "finalizePressureRowWorklist",
  "compactPressureRows",
] as const) {
  assert.ok(!wgsl.includes(`fn ${entry}`), `${entry} retired source returned`);
}

assert.match(pcm, /fn pcm\$\{label\}RankSelect\(rank:u32\)/,
  "PCM rank-select generator is missing");
assert.match(pcm, /let count=atomicLoad\([^\n]+TREE_\$\{level\}/,
  "PCM tree descent no longer performs the expected atomic child-count load");
assert.match(pcm, /let word=atomicLoad\([^\n]+ACTIVE_BITS/,
  "PCM leaf scan no longer performs the expected atomic active-word load");

const integerConstant = (name: string): number => {
  const match = resident.match(new RegExp(`(?:export )?const ${name} = (\\d+);`));
  assert.ok(match, `${name} is missing`);
  return Number(match[1]);
};
const iterations = integerConstant("SPARSE_CM12_PRESSURE_ITERATIONS");
const cadence = integerConstant("SPARSE_CM12_PRESSURE_TRUE_RESIDUAL_CADENCE");
const interiorCadences = Math.floor((iterations - 1) / cadence);
const addressResolutionsPerCell = 1 // pressure topology: preparePressure
  + 4 // pressure RHS
  + 3 * iterations // ordinary pipelined iteration
  + 4 * interiorCadences // guarded residual/restart/recovery
  + 1; // final true residual

function treeLevelCounts(leafCount: number): number[] {
  assert.ok(Number.isSafeInteger(leafCount) && leafCount > 0);
  const levels = [leafCount];
  while (levels.at(-1)! > 1) levels.push(Math.ceil(levels.at(-1)! / 32));
  return levels;
}

function sourceAtomicLoadEstimate(leafCount: number): {
  readonly treeLevelCounts: readonly number[];
  readonly uniformLeafMean: number;
  readonly maximum: number;
} {
  const levels = treeLevelCounts(leafCount);
  let treeLoads = 0;
  // The generated loop loads every child count in the selected parent, even
  // after `selected` is non-invalid. Weight parents uniformly by leaf here;
  // live member distribution can differ and must be measured on the GPU.
  for (let leaf = 0; leaf < leafCount; leaf += 1) {
    for (let level = 0; level < levels.length - 1; level += 1) {
      const target = Math.floor(leaf / 32 ** level);
      const parent = Math.floor(target / 32);
      treeLoads += Math.min(32, levels[level]! - parent * 32);
    }
  }
  const headerAndContainsLoads = 4; // phase+total, then phase+active word
  const uniformLeafWordLoads = 4.5; // eight 32-bit words per 256-id leaf
  return {
    treeLevelCounts: levels,
    uniformLeafMean: headerAndContainsLoads + uniformLeafWordLoads
      + treeLoads / leafCount,
    maximum: headerAndContainsLoads + 8
      + levels.slice(0, -1).reduce((sum, count) => sum + Math.min(32, count), 0),
  };
}

const oceanCellLeafCount = 2_099;
const oceanPressureCells = 273_142;
const oceanLoads = sourceAtomicLoadEstimate(oceanCellLeafCount);
const oceanAddressResolutions = oceanPressureCells * addressResolutionsPerCell;

console.log(JSON.stringify({
  status: "pass",
  productionAddressing: "immutable-generation-stamped-materialized-pcm-list",
  productionListFallback: false,
  retiredCompactorSourceDeleted: true,
  canonicalListMaterializedByResident: true,
  iterations,
  trueResidualCadence: cadence,
  addressResolutionsPerCell,
  oceanB16P16Reference: {
    pressureCells: oceanPressureCells,
    pcmCellLeafCount: oceanCellLeafCount,
    addressResolutionsPerFrame: oceanAddressResolutions,
    sourceAtomicLoadsPerResolution: oceanLoads,
    sourceAtomicLoadsPerFrameUniformLeafEstimate:
      oceanAddressResolutions * oceanLoads.uniformLeafMean,
    sourceAtomicLoadsPerFrameMaximum: oceanAddressResolutions * oceanLoads.maximum,
  },
  caveat: "source-level dependent atomic loads, not measured GPU memory transactions",
  requiredABConstructionGap: [],
}, null, 2));
