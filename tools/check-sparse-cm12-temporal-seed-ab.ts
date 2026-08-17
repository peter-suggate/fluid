#!/usr/bin/env node
/** Static construction-authority contract for the exact temporal seed A/B. */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const read = (path: string) => readFileSync(resolve(root, path), "utf8");
const wgsl = read("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts");
const resident = read("lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts");
const solver = read("lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts");
const harness = read("tools/probe-sparse-cm12-temporal-seed-ab.ts");

assert.match(wgsl, /override TEMPORAL_CHANGE_SEED_QA:bool=true;/,
  "paired exact-change seed must be the shared resident default");
assert.match(wgsl, /override TEMPORAL_SEED_COUNT_QA:bool=false;/,
  "seed telemetry must default off in production");
assert.match(wgsl, /let membershipChanged=pressureCellMembershipPredicate\(cell\)!=pcmCellContains\(cell\);/);
assert.match(wgsl, /bitcast<u32>\(state\[sourceDensity\(\)\+cell\]\)[\s\S]*bitcast<u32>\(state\[destinationDensity\(\)\+cell\]\)/,
  "change arm must compare exact density bits");
assert.match(wgsl, /bitcast<u32>\(state\[sourceGamma\(\)\+cell\]\)[\s\S]*bitcast<u32>\(state\[destinationGamma\(\)\+cell\]\)/,
  "change arm must compare exact gamma bits");
assert.match(wgsl, /let conservative=hasRigidBodies\(\)[\s\S]*cellOpenFraction\(cell\)/,
  "rigid/cut/open-fraction cells must retain conservative handling");
assert.match(wgsl, /slow=membershipChanged\|\|scalarChanged\|\|conservative;/);
assert.match(wgsl, /slow=membershipChanged[\s\S]*!temporalScalarCellCanonicalAt\(cell,destinationDensity\(\),destinationGamma\(\)\);/,
  "current arm must retain the production classifier exactly");

const firstDilation = resident.indexOf('dispatchAccepted("dilateTemporalScalarAtoB", "cell")');
const secondDilation = resident.indexOf('dispatchAccepted("dilateTemporalScalarBtoA", "cell")');
const compact = resident.indexOf('dispatchAccepted("compactTemporalScalarCells", "cell")');
assert(firstDilation >= 0 && secondDilation > firstDilation && compact > secondDilation,
  "both conservative incidence dilations must precede compaction");
for (const factory of ["createTemporalCurrentSeedOracleForQA",
  "createTemporalChangeSeedOracleForQA"]) {
  assert(resident.includes(`static ${factory}(`), `resident lacks ${factory}`);
  assert(solver.includes(`static ${factory}(`), `solver lacks ${factory}`);
  assert(harness.includes(factory), `paired harness does not require ${factory}`);
}
assert.match(resident, /TEMPORAL_CHANGE_SEED_QA: temporalSeedModeForQA === "change" \? 1 : 0/);
assert.match(resident, /TEMPORAL_SEED_COUNT_QA: 1/);
assert.match(resident, /seedCells: words\[8\]/);
assert.match(resident, /firstDilationCells: words\[9\]/);
assert.match(resident, /finalCells: words\[0\]/);
assert.match(resident, /finalRows: words\[4\]/);
assert.doesNotMatch(solver + resident, /URLSearchParams[\s\S]{0,160}TEMPORAL_(?:CHANGE|SEED)/,
  "temporal specialization must not be URL/runtime selectable");
for (const field of ["density", "gamma", "velocity", "pressure", "divergence"]) {
  assert(harness.includes(`"${field}"`) || harness.includes(`.${field}`),
    `paired harness lacks exact ${field} coverage`);
}
for (const receipt of ["fca", "srr", "vex", "pcm", "ptr", "pressure"]) {
  assert(harness.includes(`"${receipt}"`) || harness.includes(` ${receipt}`),
    `paired harness lacks ${receipt} receipt comparison`);
}
assert.match(harness, /stableAuthorityIdentity/,
  "paired gate must separate stable authority identity from work observations");
assert.match(harness,
  /const OMIT_AUTHORITY_BRANCHES = new Set\(\[[\s\S]*"cellClassificationScratchQA"[\s\S]*\]\);/,
  "local cell classification scratch must be excluded from paired identity");
assert.match(harness,
  /const \{ classificationBitCount, classificationBitsSha256,[\s\S]*matchesClassification, \.\.\.cellAuthority \} = pcm\.cell;/,
  "paired PCM cell identity must retain authority while separating scratch classification");
assert.match(harness, /cell: cellAuthority,/,
  "paired PCM identity must retain the accepted cell authority fields");
assert.match(harness,
  /pcm\[domain\]\.phase === SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE\.accepted/,
  "each PCM domain must be accepted");
assert.match(harness, /pcm\[domain\]\.firstFault === PCM_INVALID/,
  "each PCM domain must retain no first fault");
assert.match(harness,
  /pcm\[domain\]\.candidateGeneration === pcm\[domain\]\.acceptedGeneration/,
  "each PCM candidate must be committed");
assert.match(harness, /pcm\[domain\]\.totalCount === pcm\[domain\]\.activeBitCount/,
  "each PCM authority count must match its active bitset");
assert.match(harness,
  /if \(domain === "row" \|\| pcm\.mode === "full-refresh-oracle"\)/,
  "local aliased cell scratch must not be treated as a classification oracle");
assert.match(harness, /WORK_COUNTER_KEYS/);
for (const workField of ["temporalScalarCellCount", "temporalScalarRowCount",
  "pcmCellDirtyLeafCount", "pcmRowDirtyLeafCount"]) {
  assert.match(harness, new RegExp(`"${workField}"`),
    `${workField} must be reported as work, not paired authority identity`);
}
assert.match(harness,
  /const acceptedTopologyIdentity = \(snapshot:[\s\S]*acceptedSteps: snapshot\.acceptedSteps,[\s\S]*snapshot\.bricks\.map\(\(\{ key, active, acceptedResolution \}\) =>[\s\S]*\(\{ key, active, acceptedResolution \}\)\)/,
  "topology identity must contain only accepted step/key/active/resolution state");
assert.match(harness, /hashJSON\(acceptedTopologyIdentity\(topology\)\)/,
  "paired topology hash must use the accepted-only projection");
assert.doesNotMatch(harness, /topologySha256: hashJSON\(topology\)/,
  "full activity-policy work state must not enter paired topology identity");
assert.match(harness, /deltaChangeMinusCurrent/,
  "paired artifact must report change-minus-current work deltas");
assert.doesNotMatch(harness, /assert\.deepEqual\(right\[receipt\], left\[receipt\]/,
  "paired gate must not require intended work counters to remain equal");
assert.match(harness,
  /temporal\.seedCells <= temporal\.firstDilationCells[\s\S]*temporal\.firstDilationCells <= temporal\.finalCells/,
  "seed census must retain monotonic two-dilation evidence");
assert.match(harness, /partialFrames/,
  "failure artifacts must retain every completed per-arm frame");
assert.match(harness, /gateErrors/);
assert.match(harness, /validationErrors/);
assert.match(harness, /lifecycleErrors/);
const artifactWrite = harness.lastIndexOf("await writeFile(out");
const failureRethrow = harness.lastIndexOf("if (!passed)");
assert(artifactWrite >= 0 && failureRethrow > artifactWrite,
  "paired failure evidence must be written before rethrow");
assert.match(harness, /--scene=symmetric5\|symmetric20\|symmetric60\|dam5\|ocean8/,
  "construction-only probe help must document its fixed scene ladder");
assert.match(harness,
  /createSymmetricExpansionScene[\s\S]*horizontalGrid = 32[\s\S]*brickGrid\[2\] \/ 4[\s\S]*brickGrid\[1\] \/ 2[\s\S]*brickGrid\[0\] \/ 4/,
  "symmetric5 must use the exact weakened 32x16x32 middle-half/lower-half construction");
assert.match(harness, /scene\.numerics\.fixedDt_s = scene\.numerics\.maxDt_s = CM12_PAPER_DT_S/,
  "symmetric construction must use the canonical paper step");
assert.match(harness, /createProcessRetainedDawnGPU/,
  "Dawn GPU must remain process-reachable through final native callbacks");
assert.match(harness, /await new Promise<void>\(\(resolve\) => setImmediate\(resolve\)\)/,
  "teardown must yield one final event-loop turn before dropping local Dawn ownership");
assert.match(harness,
  /capacityStatus: "provisional-census-cap-not-a-production-capacity-claim"/,
  "K=32 must be identified as a provisional census cap, not production capacity");
assert.match(harness, /import \{ CM12_PAPER_DT_S \} from "\.\.\/lib\/core\/cm12-numerics";/,
  "paper-step probes must use the canonical CM12 time step");
assert.match(harness,
  /const dt_s = options\.timeStep === "paper"[\s\S]*\? CM12_PAPER_DT_S : timingScene\.numerics\.maxDt_s;/,
  "probe targets must match the solver's paper-vs-scene step convention");
assert.doesNotMatch(harness,
  /scene\.numerics\.fixedDt_s \?\? scene\.numerics\.maxDt_s/,
  "authored fixedDt cannot drive a paper-step solver target");
assert.match(harness, /dt_s, simulatedDuration_s: steps \* dt_s/,
  "artifacts must record their effective step and simulated duration");
assert.match(harness,
  /await gpuCompilationManagerFor\(device\)\.whenIdle\(\)[\s\S]*await device\.queue\.onSubmittedWorkDone\(\)/,
  "probe teardown must drain compiler and queue");
assert.match(harness, /solver\.destroy\(\)/,
  "each immutable arm must destroy its solver");
assert.match(harness, /if \(lockAcquired\)[\s\S]*await releaseWebGPUExclusiveLock\(\)/,
  "exclusive lock release must be attempted independently of device cleanup");
assert.match(harness, /invalidateGPUCompilationManager/,
  "harness must retire the managed compiler before device destruction");
assert.doesNotMatch(harness, /createAsync fallback/,
  "harness must not document or install a production fallback");

console.log(JSON.stringify({ passed: true,
  contract: "sparse-cm12-temporal-seed-ab",
  productionDefaults: { changeSeed: true, telemetry: false },
  factories: ["current", "change"], incidenceDilations: 2,
  fields: ["density", "gamma", "velocity", "pressure", "divergence"],
  receipts: ["FCA", "SRR", "VEX", "PCM", "PTR", "pressure"],
}, null, 2));
