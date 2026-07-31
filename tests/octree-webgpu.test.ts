import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { octreeMethod, octreeSolverOptions } from "../lib/methods/octree";
import { POWER_VALIDATION_METHOD_PROFILE, scenePresets } from "../lib/scenes";
import { resolveMethodValues } from "../lib/methods/types";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { defaultScene } from "../lib/model";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { enumerateOctreeFrontierCandidateLattice, expandOctreePressureRowAffectedOneRing, fineLevelSetWarmStartRequested, mergeOctreePowerRowIdentities, OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES, OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT, octreeDensePhiReleaseReady, octreeDiagnosticShader, octreePersistentMGPCGEnabled, octreeProjectionPipelineRequired, octreeProjectionShader, planOctreeCompactionAllocation, planOctreeLeafFrontierAllocation, planOctreePressureCapacity, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import {
  octreePipelinedMGPCGShader,
  WebGPUOctreePipelinedMGPCG,
} from "../lib/webgpu-octree-pipelined-mgpcg";
import { quadtreeSurfaceShader, WebGPUQuadtreeSurfaceState } from "../lib/webgpu-quadtree-builder";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const structuredVelocitySource = readFileSync(
  new URL("../lib/webgpu-octree-structured-velocity-gpu.ts", import.meta.url), "utf8");
const structuredBoundarySource = readFileSync(
  new URL("../lib/webgpu-octree-structured-boundary.ts", import.meta.url), "utf8");
const spgridVCycleSource = readFileSync(
  new URL("../lib/webgpu-octree-spgrid-vcycle.ts", import.meta.url), "utf8");
const uniformSolverSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const smokeSource = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
const packageManifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
  scripts: Record<string, string>;
};

test("row-indexed octree startup skips the unreachable dense Jacobi specialization", () => {
  const product = {
    solidRasterization: false,
  } as const;
  assert.equal(octreeProjectionPipelineRequired("rasterizeSolids", product), false);
  assert.equal(octreeProjectionPipelineRequired("buildDirtyTileDelta", product), true,
    "paged power authority always compiles its exact compact delta consumer");
  assert.equal(octreeProjectionPipelineRequired("rasterizeSolidsDelta", {
    ...product, solidRasterization: true,
  }), true, "terrain and rigid scenes retain solid rasterization");
  assert.equal(octreeProjectionPipelineRequired("jacobi", product), true,
    "unknown entry points are not part of the production list");
  const productionEntryPoints = octreeSource.slice(
    octreeSource.indexOf("private static readonly pipelineEntryPoints"),
    octreeSource.indexOf("private assignPipelines"),
  );
  for (const retired of [
    "jacobi", "project", "reconstructSmallGradients", "reconstructGradients",
    "projectLeaves", "passThroughPressureOverflow", "extrapolate",
    "assembleSystem", "assembleCoarseSystem",
  ]) {
    assert.doesNotMatch(productionEntryPoints, new RegExp(`"${retired}"`),
      `${retired} must not return to the production pipeline tuple`);
  }
  assert.match(octreeSource, /if \(!this\.basePipelineRequired\(entryPoint\)\) return/);
  assert.doesNotMatch(octreeSource, /compiled\.find|compiled\[index\]\s*\?\?=/,
    "body-free startup must leave unreachable slots unpublished instead of aliasing an unrelated kernel");
  assert.match(octreeSource, /for \(let size = Math\.min\(8, this\.maxLeafSize\); size >= 2; size >>= 1\)/,
    "fine refinement startup must not compile coarse or unreachable leaf sizes");
  assert.match(octreeSource, /for \(let size = this\.maxLeafSize; size >= 16; size >>= 1\)/,
    "coarse refinement startup must stop at this solver's immutable maximum");
  assert.doesNotMatch(octreeSource, /private (?:jacobi|iterateChebyshev|solve|pressureOverflow)Pipeline/,
    "retired compatibility solver and dense projection pipelines must be deleted");
});

test("octree pipeline cache keys include stable constants and reachability", () => {
  const cacheKey = octreeSource.slice(octreeSource.indexOf("private pipelineCacheKey()"),
    octreeSource.indexOf("private applyPipelineCache"));
  assert.match(cacheKey, /stableEntries\(this\.pipelineConstants\(\)\)/,
    "specializations with different sparse-layout constants must not alias");
  assert.match(cacheKey, /reachability: stableEntries\(reachability\)/,
    "immutable solver reachability must participate in the cache identity");
  assert.match(cacheKey, /requiredEntryPoints:[\s\S]*octreeProjectionPipelineRequired\(entryPoint, reachability\)/,
    "the cache identity must retain the exact reachable base-program set");
  assert.match(cacheKey, /\.sort\(\(\[left\], \[right\]\) => left < right \? -1 : left > right \? 1 : 0\)/,
    "cache-key object entries must be serialized in a stable key order");
});

test("paper grading preserves catalog-valid co-spherical masks", () => {
  assert.doesNotMatch(octreeProjectionShader,
    /paperStrictlyObtuseCoarseMask|repairPaperAcuteNeighbors|repairPaperAcuteTopology/,
    "row-local catalog triangulation must replace topology mutation for co-spherical links");
  assert.doesNotMatch(octreeSource, /paperAcuteColor|repairPaperAcutePipeline/,
    "the host must not schedule the removed refinement cascade");
  assert.match(octreeProjectionShader, /fn repairPaperMixedNeighbors/,
    "the paper's exclusive same\/coarser or same\/finer grading rule remains enforced");
  assert.match(octreeProjectionShader,
    /fn repairPaperRatioNeighbors[\s\S]*neighbor\.size > 2u \* size[\s\S]*for \(var parity = 0u; parity < 8u;/,
    "unit leaves at a sparse tile seam must repair a 4:1 lookup neighbor without publishing that tile as pressure work");
  assert.match(octreeSource,
    /this\.balanceRounds = octreeBalanceRounds\(this\.maxLeafSize\)/,
    "strong grading must use the immutable maximum-leaf round plan");
});

test("octree is a registered GPU method with dam-break defaults", () => {
  assert.ok(simulationMethods.includes(octreeMethod));
  assert.equal(octreeMethod.id, "octree");
  assert.equal(octreeMethod.backend, "webgpu");
  assert.equal(octreeMethod.showQualityControl, false,
    "legacy quality tiers must not obscure the explicit power-method controls");
  assert.deepEqual(octreeMethod.params.map((spec) => spec.key), [
    "globalFineLevelSetFactor", "maximumLeafSize", "interfaceRefinementBandCells",
    "surfaceRefinementGradingLayers",
  ], "the product UI must not expose a retired pressure path");
  assert.equal(octreeMethod.params.some((spec) => spec.key === "surfaceColumns"), false,
    "scene voxelDomain is the sole spatial-resolution authority");
  const layout = createTallCellLayout(defaultScene, "balanced", 2_048);
  assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [24, 18, 16]);
  assert.deepEqual([layout.nx * 4, layout.fineNy * 4, layout.nz * 4], [96, 72, 64],
    "the factor-4 authoritative lattice must remain an integer cubic refinement");
  assert.ok(Math.abs(defaultScene.container.width_m / layout.nx - defaultScene.container.height_m / layout.fineNy) < 1e-12);
  assert.ok(Math.abs(defaultScene.container.width_m / layout.nx - defaultScene.container.depth_m / layout.nz) < 1e-12);
  assert.equal(octreeMethod.presetFor("balanced").maximumLeafSize, "16");
  assert.equal(octreeMethod.presetFor("high").maximumLeafSize, "16");
  assert.equal(octreeMethod.params.some((spec) => spec.key === "secondaryParticles"), false,
    "unsupported spray must not be exposed by compact power-face authority");
  assert.match(octreeMethod.detail, /no topology readbacks/);
  assert.match(octreeMethod.detail, /Section 4\.3 hybrid MGPCG/);
  assert.match(octreeMethod.detail, /rigid-body coupling/);
  assert.match(octreeMethod.description, /sparse-pyramid hybrid MGPCG pressure solve/);
  const maximumLeaf = octreeMethod.params.find((spec) => spec.key === "maximumLeafSize");
  assert.ok(maximumLeaf && maximumLeaf.kind === "select");
  assert.equal(maximumLeaf.default, "16");
  assert.deepEqual(maximumLeaf.options.map((option) => option.value), ["2", "4", "8", "16", "32"]);
  assert.match(octreeSource, /function octreeLeafSize\(value: number\): 2 \| 4 \| 8 \| 16 \| 32/);
  assert.match(octreeSource, /rounded >= 32/);
  const interfaceBand = octreeMethod.params.find((spec) => spec.key === "interfaceRefinementBandCells");
  assert.ok(interfaceBand && interfaceBand.kind === "number" && interfaceBand.tier === "fine"
    && interfaceBand.label === "Band reach" && interfaceBand.default === 4);
  assert.deepEqual(interfaceBand && interfaceBand.kind === "number"
    ? [interfaceBand.min, interfaceBand.max, interfaceBand.step] : undefined, [0, 4, 1],
  "the product control exposes the five coupled experimental reach levels");
  assert.equal(octreeMethod.params.some((spec) => spec.key === "fineLevelSetBandCells"), false,
    "the surface width is derived from the single product knob");
  const surfaceGrading = octreeMethod.params.find((spec) => spec.key === "surfaceRefinementGradingLayers");
  assert.ok(surfaceGrading && surfaceGrading.kind === "number"
    && surfaceGrading.default === 1 && surfaceGrading.min === 1 && surfaceGrading.max === 4,
  "progressive grading must remain an explicit, legacy-preserving experiment");
  const globalFine = octreeMethod.params.find((spec) => spec.key === "globalFineLevelSetFactor");
  assert.ok(globalFine && globalFine.kind === "select" && globalFine.default === "4" && globalFine.tier === "coarse");
  assert.deepEqual(globalFine.options.map((option) => option.value), ["1", "4", "8"]);
  for (const quality of ["balanced", "high", "ultra"] as const) {
    const preset = octreeMethod.presetFor(quality);
    assert.equal(preset.globalFineLevelSetFactor, "4", `${quality} must default to the paper's factor-4 band`);
  }
  assert.match(smokeSource, /FLUID_OCTREE_GLOBAL_FINE_FACTOR/);
  assert.doesNotMatch(octreeSource,
    /surfaceRefinementFactor|surfacePageResolution|WebGPUOctreeSurfacePages/,
    "the product path must not retain the duplicate row-attached surface-page authority");
  assert.doesNotMatch(octreeProjectionShader,
    /surfacePageResolution|pagedSurfaceAuthority|surfacePagePhi/,
    "projection consumes compact coarse phi plus the factor-m global-fine band");
  assert.doesNotMatch(`${octreeSource}\n${uniformSolverSource}`, /airRefinementBandCells/);
  assert.match(uniformSolverSource, /interfaceRefinementBandCells: options\.octree\.interfaceRefinementBandCells \?\? 4/);
  assert.match(uniformSolverSource, /surfaceRefinementGradingLayers: options\.octree\.surfaceRefinementGradingLayers \?\? 1/);
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.doesNotMatch(methodSource, /faceVelocityTransport|powerDiagramProjection|leafSolver/);
  assert.match(methodSource, /globalFineLevelSetFactor: globalFineLevelSetFactor/);
  assert.doesNotMatch(octreeSource, /extrapolationSweeps|requestedExtrapolationSweeps/);
  assert.equal(octreeMethod.params.some((spec) => spec.key === "pressureWarmStart"), false,
    "pressure warm-start policy is an internal solver detail, not a product setting");
  assert.doesNotMatch(octreeSource, /pressureWarmStart|jacobiRelaxation|surfaceDetailStrength/);
  assert.doesNotMatch(uniformSolverSource,
    /options\.octree\.(?:pressureWarmStart|jacobiRelaxation|surfaceDetailStrength|brickPreActivation)/);
  assert.match(octreeSource, /bulkResidency: true,[\s\S]*brickPreActivation: true/,
    "production power-octree sparse residency always uses preactivation");
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  const candidate = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  assert.match(candidate, /this\.encodeFrontierRows\(/,
    "row compaction belongs exclusively to the inactive next-epoch transaction");
  assert.doesNotMatch(encode, /this\.encodeFrontierRows\(/,
    "the active solve must consume the beginning-of-substep committed rows without remapping them");
  assert.doesNotMatch(encode.slice(encode.indexOf("else")), /clearBuffer\(this\.pressure[AB]\)/,
    "compact cold starts are initialized row-by-row during emission, not by clearing capacity-sized buffers");
});

test("bounded power-vs-tall Dawn comparison uses one exact active-tall grid", () => {
  const command = packageManifest.scripts["test:webgpu:dam-power-fine-compare-one-step"];
  assert.ok(command);
  assert.match(command, /FLUID_METHOD=octree,tall-cell/);
  assert.match(command, /FLUID_TARGET_S=0\.004/);
  assert.match(command, /FLUID_ORACLE_STEPS=1/);
  assert.match(command, /FLUID_VOXEL_CELL_SIZE=0\.05/);
  assert.match(command, /FLUID_EXPECT_GRID=24,18,16/);
  assert.match(command, /FLUID_REGULAR_LAYERS=12/);
  assert.match(command, /FLUID_OCTREE_GLOBAL_FINE_FACTOR=4/);
  assert.doesNotMatch(packageManifest.scripts["test:webgpu:dam-power-fine-parity"], /FLUID_VOXEL_CELL_SIZE=0\.02/,
    "the named comparison path must not retain the former 2400-column allocation");

  const layout = createTallCellLayout(defaultScene, "balanced", 2_048, {
    regularLayers: 12,
  });
  assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [24, 18, 16]);
  assert.equal(layout.planning.ordinaryGridFallback, false);
  assert.ok(layout.columnBases.some((base) => base >= 2),
    "the comparison must exercise restricted tall cells, not the ordinary-grid fallback");
  assert.match(smokeSource, /scene\.voxelDomain\.finestCellSize_m = voxelCellSizeOverride/,
    "the smoke comparison must change the shared scene lattice directly");
  assert.match(smokeSource, /refusing to step a mismatched comparison/);
});

test("regular dam Dawn regression matches the UI cadence through two seconds", () => {
  const command = packageManifest.scripts["test:webgpu:dam-octree"];
  assert.ok(command);
  assert.match(command, /FLUID_SCENE=dam-break-ui/);
  assert.match(command, /FLUID_METHOD=octree/);
  assert.match(command, /FLUID_TARGET_S=2/);
  assert.match(command, /FLUID_MAX_DT=0\.008/);
  assert.match(command, /FLUID_ORACLE_STEPS=250/);
  assert.match(command, /FLUID_EXPECT_EXACT_STEPS=250/,
    "the default regression must run the browser cadence through the full requested interval");
});

test("native power pressure capacity reserves rows without a variable-entry arena", () => {
  const dims = { nx: 288, ny: 96, nz: 64 };
  const count = dims.nx * dims.ny * dims.nz;
  const plan = planOctreePressureCapacity(dims, 16, 4);
  assert.equal(plan.rowCapacity, 537_856);
  assert.deepEqual(Object.keys(plan).sort(), ["headerBytes", "pressureBytes", "rowCapacity"],
    "resolved case-local handles replace the retired variable-entry capacity");
  assert.ok(plan.rowCapacity < count / 2, "the widened ocean must not reserve one pressure slot per finest cell");
  assert.equal(planOctreePressureCapacity(dims, 16, 4, 1024).rowCapacity, 1024);
  assert.equal(planOctreePressureCapacity(dims, 16, 4, undefined, false, 1, 184_832).rowCapacity,
    184_832, "device binding limits cap every row-indexed authority before allocation");
});

test("compact frontier retains only the immutable public row-delta ABI", () => {
  const cellCount = 320 * 96 * 80;
  const rowCapacity = planOctreePressureCapacity({ nx: 320, ny: 96, nz: 80 }, 16, 4).rowCapacity;
  assert.equal(rowCapacity, 626_176);
  const compact = planOctreeLeafFrontierAllocation(cellCount, rowCapacity);
  assert.deepEqual(compact, {
    cellCount: 2_457_600,
    listCapacity: 626_176,
    candidateOffsetWords: 1_252_362,
    rowDeltaControlOffsetWords: 1_878_538,
    rowDeltaNewToOldOffsetWords: 1_878_554,
    rowDeltaOldToNewOffsetWords: 2_504_730,
    rowDeltaDirtyRowsOffsetWords: 3_130_906,
    rowDeltaAffectedRowsOffsetWords: 3_757_082,
    candidateBytes: 2_504_704,
    allocatedBytes: 17_533_032,
  });
  assert.throws(() => planOctreeLeafFrontierAllocation(0, rowCapacity), /positive integer/);
  assert.throws(() => planOctreeLeafFrontierAllocation(cellCount, 0), /positive integer/);
});

test("cold dam-UI frontier candidates cover every finest cell exactly once", () => {
  const [nx, ny, nz] = [24, 18, 16] as const;
  const cells = enumerateOctreeFrontierCandidateLattice([nx, ny, nz]);
  assert.equal(cells.length, nx * ny * nz);
  assert.equal(new Set(cells).size, cells.length);
  assert.deepEqual([...cells].sort((a, b) => a - b),
    Array.from({ length: nx * ny * nz }, (_, index) => index));
  assert.match(octreeProjectionShader,
    /candidateLaneBase[\s\S]*for\(var octant=0u;octant<8u;octant\+=1u\)/,
    "each 4³ candidate lane must enumerate its complete 2³ sub-block");
  assert.match(octreeProjectionShader, /if\(all\(base<dims\(\)\)\)\{cell=frontierCandidateAt\(base\+offset,deltaMode\);\}/,
    "rounded indirect workgroups must reject the invalid lane base before octant addition can wrap");
});

test("the band-reach control couples pressure and surface support", () => {
  const values = resolveMethodValues(octreeMethod, "balanced",
    { ...POWER_VALIDATION_METHOD_PROFILE.overrides });
  const pinned = octreeSolverOptions(defaultScene, "balanced", values).octree;
  assert.equal(pinned.interfaceRefinementBandCells, 3);
  assert.equal(pinned.fineLevelSetBandCells, 3,
    "one product level must reach both pressure and fine-surface consumers");
  for (const level of [0, 1, 2, 3, 4]) {
    const options = octreeSolverOptions(defaultScene, "balanced",
      { ...values, interfaceRefinementBandCells: level }).octree;
    assert.equal(options.interfaceRefinementBandCells, level);
    assert.equal(options.fineLevelSetBandCells, level);
  }
  assert.deepEqual(octreeSolverOptions(defaultScene, "balanced",
    { ...values, interfaceRefinementBandCells: 99 }).octree,
  { ...pinned, interfaceRefinementBandCells: 4, fineLevelSetBandCells: 4 },
  "out-of-range authored values clamp to the five supported product levels");
  // Fine-only fault injection remains available to the Dawn harness, but is
  // intentionally absent from the product schema.
  const swept = octreeSolverOptions(defaultScene, "balanced",
    { ...values, fineLevelSetBandCells: 6 }).octree;
  assert.equal(swept.interfaceRefinementBandCells, 3);
  assert.equal(swept.fineLevelSetBandCells, 6);
  for (const preset of scenePresets) {
    const overrides = preset.methodProfile?.methodId === "octree"
      ? preset.methodProfile.overrides : undefined;
    assert.equal(overrides?.fineLevelSetBandCells, undefined,
      `${preset.id} bypasses the coupled product reach with a fine-only override`);
  }
});

test("compact scan scratch feeds the class-specialized structured authority", () => {
  const dims = { nx: 640, ny: 192, nz: 160 };
  const pressure = planOctreePressureCapacity(dims, 16, 4);
  const activeTileCapacity = 4_800;
  const activeTileWorklistBytes = (16 + 2 * activeTileCapacity) * 4;
  const compact = planOctreeCompactionAllocation(
    dims, pressure.rowCapacity, activeTileWorklistBytes, activeTileCapacity, 16,
  );
  assert.deepEqual(compact, {
    scanBlockCapacity: 10_003,
    candidateBlockCapacity: 76_800,
    scanAndTaskBytes: 734_496,
    activeTileBytes: 269_508,
    changeStateBaseWords: 5_335_170,
    tileChangeFlagsOffsetWords: 5_335_170,
    tileRefinementSignaturesOffsetWords: 5_344_770,
    tileFrontierSignaturesOffsetWords: 5_368_770,
    tileSignatureChangedOffsetWords: 5_392_770,
    tileFrontierChangeFlagsOffsetWords: 5_397_570,
    frontierTopologyReuseWord: 5_402_530,
    dirtyFailureOffsetWords: 5_402_531,
    rowDeltaScratchBaseWords: 183_624,
    rowDeltaScratchWords: 5_151_546,
    allocatedBytes: 21_610_188,
  });
  assert.throws(() => planOctreeCompactionAllocation(dims, 0, 0, 0, 16), /positive integer/);
  assert.throws(() => planOctreeCompactionAllocation(dims, 1, -1, 0, 16), /active-tile bounds/);
  assert.match(octreeSource,
    /this\.resolvedLinearOperator\s*=\s*this\.firstOrderVCycle\.accurateOperator/,
    "production must cut directly from compact rows to rediscretized A2");
  assert.match(structuredVelocitySource,
    /for\(var family=0u;family<6u;family\+=1u\)/,
    "the direct publication must materialize all six velocity families");
  assert.match(structuredVelocitySource,
    /section63Coefficients:\s*array<f32>[\s\S]*fn publishSection63Rows/,
    "the same transaction must publish the fixed 19-channel Section 6.3 operator");
  assert.match(spgridVCycleSource,
    /SPGrid accurate A2 convergence-gated class records[\s\S]*dispatchWorkgroupsIndirect\(this\.accurateClassDispatch/,
    "the A2 gather must consume exact class records without host-sized row dispatches");
  assert.match(octreeProjectionShader,
    /fn classifyTopologyTileSignature[\s\S]*currentPressureOwnerWet\(owner\)[\s\S]*TILE_SIGNATURE_VALID_MAGIC/,
    "structural dirtiness must compare persistent discrete owner/wet decisions");
  assert.match(octreeProjectionShader,
    /fn topologyTileActive\(key: u32\)[\s\S]*sparseTopologyTileStates[\s\S]*tileStateHash/,
    "active membership must query the persistent dense or sparse state publication directly");
  assert.doesNotMatch(octreeProjectionShader,
    /authorityMaskWordsPerTile|summaryGenerationBase|markChangedTiles|buildDirtyWorklist/,
    "the full tile/cell scan and its retained phase masks must stay deleted");
  assert.match(octreeProjectionShader,
    /if \(activeCount > capacity \|\| compaction\[4\] > capacity\)[\s\S]*compaction\[1\] = 0u[\s\S]*compaction\[8\] = 0u/,
    "an invalid structural publication must fail closed with zero schedules");
  assert.doesNotMatch(octreeProjectionShader,
    /fineDeltaAuthoritative|fineDeltaChangedKey|finePageTopologyTile|deltaWord\(/,
    "fine value/page refresh must remain independent of structural dirtiness");
  assert.doesNotMatch(octreeSource, /changeDrivenEligible/,
    "recurring topology publication must not retain a whole-active-list selector");
  assert.match(octreeSource, /if \(active && !analyticColdBootstrap\)/,
    "only the one-time bounded analytic bootstrap may precede exact delta compaction");
  assert.match(octreeProjectionShader,
    /fn appendDirtyTileRing[\s\S]*for \(var z = -1; z <= 1[\s\S]*appendDirtyTile/,
    "changed structural decisions expand through the complete topology-tile 1-ring");
  assert.doesNotMatch(octreeSource, /FLUID_OCTREE_(?:INCREMENTAL|CHANGE_DRIVEN)/,
    "the exact paged-authority delta is the sole production topology path");
  assert.match(octreeProjectionShader,
    /required\s*==\s*carried\s*\+\s*added[\s\S]*required\s*==\s*previous(?:Count)?\s*\+\s*added\s*-\s*retired/,
    "frontier publication validates the exact carried + added - retired identity before swapping");
  assert.match(octreeProjectionShader,
    /let cellMatches=old\.cell==cell;[\s\S]*let sizeMatches=old\.size==owner\.size;[\s\S]*let originMatches=isOrigin[\s\S]*let exact=cellMatches&&sizeMatches&&originMatches/,
    "carried identity must be exact (cell, level), never positional or origin-only");
  assert.match(octreeProjectionShader,
    /fn previousFrontierHasExactIdentity[\s\S]*previousLowerBound[\s\S]*leafHeaders\[old\]\.cell==cell[\s\S]*fn frontierCandidateAt\(gid:vec3u,additionsOnly:bool\)[\s\S]*additionsOnly&&previousFrontierHasExactIdentity/,
    "recurring candidate generation must filter exact previous identities before sorting");
  assert.match(octreeProjectionShader,
    /let keep=exact&&wet;[\s\S]*select\(0u,32u,dirty\)/,
    "dirty exact identities must retain previous sort order while remaining affected");
  assert.match(octreeProjectionShader,
    /let dirty=output!=slot\|\|\(compaction\[rowDeltaFlagsBase\(\)\+slot\]&32u\)!=0u;[\s\S]*ROW_DELTA_AFFECTED/,
    "temporally carried dirty rows must still enter downstream affected-row work");
  assert.match(octreeProjectionShader,
    /let carried=frontier\[5\];[\s\S]*let added=candidateCount;[\s\S]*rowDeltaReduce\[0\]\.x==0u/,
    "the recurring merge must count every sorted candidate as a genuine addition");
  assert.match(octreeProjectionShader,
    /fn sortFrontierCandidates\([\s\S]*frontierSortStageCount\(count\)[\s\S]*candidateSortStore\(/,
    "large dirty candidate sets retain deterministic parallel merge sorting");
  assert.match(octreeProjectionShader,
    /var<workgroup> frontierLocalSortCells:array<u32,4096>[\s\S]*fn sortFrontierCandidatesLocal[\s\S]*workgroupUniformLoad\(&frontierLocalSortCells\[0u\]\)[\s\S]*frontierLocalCellLess/,
    "the complete mini-dam frontier must sort cooperatively in the portable 16 KiB workgroup floor");
  assert.match(octreeProjectionShader, /override frontierSortStage: u32 = 0u/);
  assert.match(octreeSource,
    /if \(this\.useLocalFrontierCandidateSort\) \{[\s\S]*sortFrontierCandidatesLocalPipeline[\s\S]*dispatchWorkgroups\(1\)[\s\S]*\} else \{[\s\S]*for \(const pipeline of this\.frontierCandidateSortPipelines\)[\s\S]*dispatchWorkgroupsIndirect\(this\.topologyCandidateDispatch, 0\)/,
    "construction-time capacity selection must replace thirteen mini-dam launches while retaining the large-domain fallback");
  assert.match(octreeSource,
    /this\.useLocalFrontierCandidateSort = this\.frontierAllocation\.listCapacity <= 4096/,
    "the shared-sort lane must be selected once from immutable capacity");
  assert.match(octreeSource,
    /if \(!cached && !this\.useLocalFrontierCandidateSort\) \{[\s\S]*octree\.pipeline\.frontier-sort/,
    "mini-dam construction must not compile the unused global sort specializations");
  assert.doesNotMatch(octreeSource, /advanceFrontierCandidateSort/,
    "the deleted mutable stage-counter dispatch must not return");
  assert.match(octreeProjectionShader,
    /fn classifyFrontierCarry[\s\S]*rowAuthorityDirtyGeneration\(cell,frontier\[8u\]\)/,
    "carried identity is exact and a carried row remains dirty when its fine-summary or mask tile ring changed");
  assert.match(octreeProjectionShader,
    /@compute @workgroup_size\(256\)\s*fn scanDirtyRowDeltaBlocks[\s\S]*fn prefixDirtyRowDeltaBlocks/,
    "dirty and affected row lists use deterministic fixed-record cooperative scans");
  assert.match(octreeProjectionShader,
    /fn rowDeltaExclusiveScan\([\s\S]*stride=1u;stride<256u;stride\*=2u[\s\S]*stride=128u;stride>0u;stride\/=2u/,
    "row-delta ranks use a logarithmic workgroup up-sweep/down-sweep");
  assert.doesNotMatch(octreeProjectionShader,
    /fn (?:rowDeltaExclusiveScan|scanRowDeltaBlock|prefixRowDeltaBlocks)[\s\S]*for\(var prior=0u;prior<lid/,
    "cooperative row-delta scans must not retain per-lane linear prefix loops");
  assert.match(octreeProjectionShader,
    /fn rowDeltaFlagsBase\(\)->u32\{return changeStateBase\(\)-rowDeltaScratchWords\(\);\}/,
    "temporary row-delta state belongs to the plain-u32 compaction tail");
  assert.doesNotMatch(octreeProjectionShader,
    /frontier\[rowDelta(?:Flags|Prefix|BlockTotals|CarriedBlocks)Base\(\)/,
    "scan scratch must never inherit the frontier binding's atomic ABI");
  assert.match(octreeProjectionShader,
    /fn scatterDirtyRowDelta[\s\S]*let encoded=frontier\[rowDeltaNewToOldBase\(\)\+row\][\s\S]*compaction\[rowDeltaFlagsBase\(\)\+row\]=[\s\S]*encoded&ROW_DELTA_AFFECTED/,
    "the complete initial affected image must be snapshotted before propagation");
  assert.match(octreeProjectionShader,
    /const ROW_DELTA_RING_DIRECTION_COUNT:u32=18u[\s\S]*const ROW_DELTA_RING_DIRECTIONS=array<vec3i,18>/,
    "the cooperative row kernel must cover the exact face-and-edge direction set");
  assert.match(octreeProjectionShader,
    /var<workgroup> rowDeltaRingVotes:array<u32,32>[\s\S]*@compute @workgroup_size\(32\)[\s\S]*fn markRowDeltaRing[\s\S]*lane<ROW_DELTA_RING_DIRECTION_COUNT[\s\S]*for\(var stride=16u;stride>0u;stride\/=2u\)[\s\S]*if\(lane==0u&&live&&rowDeltaRingVotes\[0u\]!=0u\)/,
    "one 32-lane workgroup per row must reduce the eighteen direction votes deterministically");
  const ring = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn rowDeltaRingDirectionAffected"),
    octreeProjectionShader.indexOf("fn scanAffectedRowDeltaBlocks"),
  );
  assert.match(ring, /compaction\[rowDeltaFlagsBase\(\)\+neighbor\]/,
    "both row-ring lanes must read the immutable affected snapshot");
  assert.doesNotMatch(ring,
    /frontier\[rowDeltaNewToOldBase\(\)\+neighbor\][^\n]*ROW_DELTA_AFFECTED/,
    "same-dispatch publications must never become propagation inputs");
  assert.match(ring,
    /@compute @workgroup_size\(256\)[\s\S]*fn markRowDeltaRingBlocks[\s\S]*rowDeltaRingDirectionAffected/,
    "capacities beyond a one-dimensional exact extent retain a snapshot-correct fallback");
  assert.doesNotMatch(octreeProjectionShader,
    /@compute @workgroup_size\(1\)\s*fn (?:sortFrontierRows|prepareRowDelta|classifyRowDelta|markRowDeltaRing|compactRowDelta)/,
    "no row-sized frontier or row-delta stage may execute as a singleton loop");
  const rowDeltaWGSL = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn prepareRowDelta"),
    octreeProjectionShader.indexOf("fn leafInfo"),
  );
  assert.doesNotMatch(rowDeltaWGSL,
    /atomic(?:Add|Sub|Min|Max|And|Or|Xor|Exchange|CompareExchange(?:Weak)?)\(/,
    "row-delta mapping and worklist publication must not use synchronization or RMW atomics");
});

test("unchanged octree generations zero every bulk topology and frontier schedule", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  const rows = octreeSource.slice(
    octreeSource.indexOf("private encodeFrontierRows("),
    octreeSource.indexOf("\n  encode(", octreeSource.indexOf("private encodeFrontierRows(")),
  );
  assert.match(rebuild,
    /pass\.setPipeline\(active\s*\?\s*pipelines\.delta\s*:\s*pipelines\.full\)[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,\s*48\)/,
    "coarse refinement and balance must consume the exact dirty-tile count");
  assert.match(rebuild,
    /prepareFrontierDispatchPipeline[\s\S]*copyBufferToBuffer\(this\.compaction,\s*4,\s*this\.topologyCandidateDispatch,\s*0,\s*36\)/,
    "candidate emission must publish sort, carry, and merge schedules in one batch");
  assert.match(rebuild,
    /classifyFrontierCarryPipeline[\s\S]*topologyCandidateDispatch,\s*12[\s\S]*mergeFrontierRowsPipeline[\s\S]*topologyCandidateDispatch,\s*24/);
  assert.doesNotMatch(rebuild, /dispatchWorkgroups\(this\.linearBlocks\)/,
    "the topology stage must not retain capacity-sized direct launches");
  assert.equal((rows.match(/dispatchWorkgroupsIndirect\(this\.topologyCandidateDispatch, 12\)/g) ?? []).length, 5,
    "block-shaped scans and identity reuse retain the live-row block schedule");
  assert.match(rows,
    /setPipeline\(this\.useCooperativeRowDeltaRing[\s\S]*this\.markRowDeltaRingPipeline\s*:\s*this\.markRowDeltaRingBlocksPipeline\)[\s\S]*dispatchWorkgroupsIndirect\(this\.topologyCandidateDispatch,\s*this\.useCooperativeRowDeltaRing\s*\?\s*36\s*:\s*12\)/,
    "the mini lane consumes the exact row-count record while large capacities retain the block fallback");
  assert.match(rebuild,
    /copyBufferToBuffer\(this\.compaction,\s*16,\s*this\.topologyCandidateDispatch,\s*36,\s*12\)/,
    "the fourth immutable indirect record must be staged from the exact row-count triple");
  assert.match(octreeSource,
    /this\.useCooperativeRowDeltaRing\s*=\s*this\.frontierAllocation\.listCapacity[\s\S]*device\.limits\.maxComputeWorkgroupsPerDimension/,
    "construction must select cooperative publication only when every row fits the exact 1-D dispatch");
  assert.match(octreeProjectionShader,
    /compaction\[4\]=0u;compaction\[5\]=1u;compaction\[6\]=1u;[\s\S]*FRONTIER_FAILED_MAGIC[\s\S]*compaction\[4\]=0u;compaction\[5\]=1u;compaction\[6\]=1u;/,
    "reuse and rejected candidates must both zero the cooperative row schedule");
  assert.doesNotMatch(rows, /prepareRowDeltaPipeline|classifyRowDeltaPipeline|finalizeRowDeltaClassificationPipeline/,
    "the canonical sorted frontier merge must publish exact row maps without a second binary-search join");
  assert.doesNotMatch(rows, /dispatchWorkgroups\(this\.linearBlocks\)/);
  assert.match(octreeProjectionShader,
    /let reuse = compaction\[dirtyAuthorityBase\(\)\] == DIRTY_TILE_VALID_MAGIC[\s\S]*compaction\[0\] == 0u && compaction\[4\] == 0u/);
  assert.match(octreeProjectionShader,
    /fn frontierGenerationReused\(\) -> bool[\s\S]*frontierTopologyReuseBase\(\)/,
    "row-delta finalizers must retain reuse after pressure controls restore the last publication");
  assert.match(octreeProjectionShader,
    /if\(frontierGenerationReused\(\)\)\{[\s\S]*rowDeltaNewToOldBase\(\)\+row\]=row\+1u;[\s\S]*rowDeltaOldToNewBase\(\)\+row\]=row\+1u;/,
    "topology reuse must still publish exact identity maps for value-refresh consumers");
  const dirty = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn buildDirtyTileDelta"),
    octreeProjectionShader.indexOf("// -----------------------------------------------------------------------------",
      octreeProjectionShader.indexOf("fn buildDirtyTileDelta")),
  );
  assert.match(dirty,
    /if \(activeCount > capacity \|\| compaction\[4\] > capacity\) \{[\s\S]*compaction\[1\] = 0u[\s\S]*compaction\[5\] = 0u[\s\S]*compaction\[8\] = 0u/,
    "malformed structural authority must fail closed with zero indirect work");
});

test("topology attempt generations are stamped in GPU invocation order", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  assert.match(rebuild,
    /stampFrontierAttemptPipeline[\s\S]*dispatchWorkgroups\(1\)[\s\S]*topology attempt generation stamped/,
    "every candidate must stamp its generation before any candidate publisher consumes it");
  assert.doesNotMatch(rebuild,
    /queue\.writeBuffer\(this\.(?:leafFrontier\s*,\s*32|params\s*,\s*136)/,
    "host writes to shared generation words collapse multiple encoded substeps onto the last stamp");
  assert.match(octreeProjectionShader,
    /fn stampFrontierAttempt\(\)[\s\S]*frontier\[8\] \+ 1u[\s\S]*frontier\[8\] = next/);
  assert.doesNotMatch(octreeProjectionShader, /frontier\[8\]\s*=\s*params\.pressureCapacity\.z/,
    "begin/finalize must preserve the invocation-stable GPU attempt stamp");
  const dirtyTransactions = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn buildDirtyTileDelta"),
    octreeProjectionShader.indexOf("// -----------------------------------------------------------------------------",
      octreeProjectionShader.indexOf("fn buildDirtyTileDelta")),
  );
  assert.equal((dirtyTransactions.match(/let generation = frontier\[8u\]/g) ?? []).length, 2,
    "both structural and wet-frontier dirty sets must use the candidate attempt generation");
  assert.doesNotMatch(dirtyTransactions, /frontier\[3\]\s*\+\s*1u/,
    "a rejected attempt must not leave subsequent dirty stamps on the accepted-generation clock");
});

test("cold pressure-row emission dereferences the inactive owner table", () => {
  const candidateView = (WebGPUOctreeProjection.prototype as unknown as {
    topologyCandidateEntryPoint: (entryPoint: string) => boolean;
  }).topologyCandidateEntryPoint;
  for (const entryPoint of ["planLeaves", "emitLeaves", "markRowDeltaRing"]) {
    assert.equal(candidateView.call(WebGPUOctreeProjection.prototype, entryPoint), true,
      `${entryPoint} must see generation-N owner pages before the coupled commit`);
  }
  assert.equal(candidateView.call(WebGPUOctreeProjection.prototype, "scanLeafBlocks"), false,
    "owner-independent scans must not acquire an unnecessary candidate-table specialization");
});

test("owner-page misses reject instead of synthesizing topology", () => {
  assert.doesNotMatch(octreeProjectionShader, /canonicalOwner|fallback|legacy|compatibility/,
    "production topology WGSL must contain no synthesized-owner or retired-path branch");
  assert.match(octreeProjectionShader,
    /fn rejectOwnerAuthority\(\) -> Owner \{[\s\S]*atomicStore\(&owners\[2\], 1u\);[\s\S]*return invalidOwner\(\);/,
    "every malformed recurring owner lookup must propagate candidate/solve rejection");
  assert.match(octreeProjectionShader,
    /if \(word == 0xffffffffu \|\| word == 0u\) \{ return rejectOwnerAuthority\(\); \}/,
    "a missing owner payload is an invalid authority, never an inferred leaf");
  assert.match(octreeProjectionShader,
    /fn storeOwnerRequired[\s\S]*atomicMin\(&owners\[[\s\S]*encodePagedOwner\(cell, origin, size\)\)/,
    "overlapping balance writers must deterministically retain the finer dyadic owner");
  assert.doesNotMatch(octreeDiagnosticShader, /canonicalOwner|fallback|legacy|compatibility/);
  assert.match(octreeDiagnosticShader,
    /if\(!ownerValid\(owner\)\)\{[\s\S]*textureStore\(topologyOut,vec3i\(gid\),vec4u\(0xffffffffu\)\)/,
    "diagnostic materialization must expose an invalid owner instead of fabricating one");
});

test("exact power-row merge retires a same-origin size change and dirties carried authority changes", () => {
  const previous = [
    { cell: 0, size: 2, morton: 0 },
    { cell: 8, size: 2, morton: 8 },
  ];
  const sizeChanged = mergeOctreePowerRowIdentities(previous, [
    { cell: 8, size: 2, morton: 8 },
    { cell: 0, size: 4, morton: 0 },
  ]);
  assert.deepEqual(sizeChanged.newToOld, [1, -1],
    "same origin with a different level is an addition, never a carried row");
  assert.deepEqual(sizeChanged.oldToNew, [-1, 0]);
  assert.deepEqual([sizeChanged.carried, sizeChanged.added, sizeChanged.retired], [1, 1, 1]);
  assert.deepEqual(sizeChanged.dirtyRows, [1]);

  const carriedDirty = mergeOctreePowerRowIdentities(previous, previous, new Set([1]));
  assert.deepEqual([carriedDirty.carried, carriedDirty.added, carriedDirty.retired], [2, 0, 0]);
  assert.deepEqual(carriedDirty.dirtyRows, [1],
    "fine-summary generation or phi-band mask change dirties an otherwise exact carried row");
});

test("pressure-row affected publication expands exactly one snapshotted ring", () => {
  assert.equal(OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT, 18);
  const neighbours = Array.from({ length: 4 }, () =>
    new Array<number>(OCTREE_PRESSURE_ROW_ONE_RING_DIRECTION_COUNT).fill(-1));
  neighbours[0]![0] = 1;
  neighbours[1]![0] = 0; neighbours[1]![1] = 2;
  neighbours[2]![0] = 1; neighbours[2]![1] = 3;
  neighbours[3]![0] = 2;
  const first = expandOctreePressureRowAffectedOneRing(
    [true, false, false, false], neighbours,
  );
  assert.deepEqual(first, [true, true, false, false],
    "a row reached during this publication must not cascade through the same dispatch");
  assert.deepEqual(expandOctreePressureRowAffectedOneRing(first, neighbours),
    [true, true, true, false],
    "a second explicit publication advances the chain by exactly one more ring");
  assert.deepEqual(expandOctreePressureRowAffectedOneRing(
    [false, false, false, false], neighbours, true,
  ), [true, true, true, true], "membership changes retain the conservative full-row rebuild");
  assert.throws(() => expandOctreePressureRowAffectedOneRing(
    [true], [[-1]],
  ), /exactly eighteen/);
});

test("retired fixed-reference hydrostatic A/B and its backing kernels are deleted", () => {
  for (const method of simulationMethods) {
    assert.equal(method.params.some((spec) => spec.key === "hydrostaticSplit"), false);
  }
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  for (const source of [methodSource, uniformSolverSource, octreeSource, legacyUniformComputeShader, octreeProjectionShader]) {
    assert.doesNotMatch(source,
      /hydrostaticSplit|fixedHydrostatic|hydrostaticAirPressure|setHydrostaticTimestep/);
  }
  assert.match(legacyUniformComputeShader,
    /for\(var y:i32=d\.y-1;y>=0;y-=1\)[\s\S]*vec4f\(highest,-1\.0,0\.0,0\.0\)/,
    "uniform occupancy retains one canonical top-down scan");
});

test("octree owns two-way immersed-body coupling without the dense compatibility graph", () => {
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(methodSource, /values, onRigidLoads\) => new WebGPUUniformEulerianSolver\(device, scene, quality, onRigidLoads/);
  assert.match(methodSource, /values, onRigidLoads, onProgress, signal\) => WebGPUUniformEulerianSolver\.createAsync/);
  assert.match(uniformSolverSource, /const activeBodies = bodies\.slice\(0, 12\)/);
  assert.doesNotMatch(uniformSolverSource, /this\.octreeProjection \? \[\] : bodies/);
  assert.match(uniformSolverSource, /this\.octreeProjection\?\.setCouplingBodies\(activeBodies\.length/);
  assert.match(uniformSolverSource, /rigidBodies: this\.rigidBuffer, rigidExchange: this\.rigidExchangeBuffer/,
    "the compact power-solid pipeline writes the resident rigid exchange directly");
  assert.match(uniformSolverSource, /if \(activeBodies\.length > 0 && !this\.octreeProjection\)/,
    "the dense Brinkman and phi-s passes are unavailable to the octree");
  assert.match(uniformSolverSource, /if \(this\.quadtreeProjection\) this\.solidPhiGroup/,
    "the retained dense phi-s group belongs only to the quadtree method");
  assert.match(uniformSolverSource, /this\.rigidSystem\.encode\(encoder, delta/);
  assert.doesNotMatch(uniformSolverSource.slice(uniformSolverSource.indexOf("advanceTo(time_s"), uniformSolverSource.indexOf("async readStats()")), /mapAsync|encodeBodyImpulseReadback/);
});

test("octree voxelizes partial solid volume and reports liquid-displaced volume", () => {
  assert.match(octreeProjectionShader, /fn bodySolidFraction/);
  assert.match(octreeProjectionShader, /for \(var corner = 0u; corner < 8u/);
  assert.match(octreeProjectionShader, /return inside \/ 8\.0/);
  assert.match(octreeProjectionShader, /fn rasterizeSolids/);
  assert.match(octreeProjectionShader, /if \(candidate > fraction\) \{ fraction = candidate; owner = i32\(bodyIndex\); \}/,
    "overlapping terrain/body samples must have one maximum-coverage owner");
  assert.match(legacyUniformComputeShader, /let displacedWeight=wetFraction\*solidFraction/,
    "buoyancy volume must integrate liquid occupancy times sub-cell solid volume");
  assert.match(legacyUniformComputeShader, /if\(candidate>solidFraction\)\{solidFraction=candidate;coupledBody=bodyIndex;\}/,
    "overlapping bodies must not double-count displaced volume");
});

test("structured boundary publication owns the variational solid constraint", () => {
  assert.match(structuredBoundarySource,
    /aperture=clamp\(\.5\+sdf\/max\(p\.physical\.x,1e-20\),0\.,1\.\)/,
    "solid SDF samples must produce the open-area fraction in the structured slot transaction");
  assert.match(structuredBoundarySource,
    /let coefficient=bitcast<f32>\(a\[abase\(\)\+p\.offset0\.z\+h\]\)\*bitcast<f32>\(a\[abase\(\)\+p\.offset0\.w\+h\]\)\*aperture\*scale/,
    "resolved rows must use area times inverse distance times aperture and free-surface scale");
  assert.match(structuredBoundarySource,
    /candidateSolidVelocity\[h\]=normalVelocity[\s\S]*solidVelocity\[sbase\(\)\+h\]=candidateSolidVelocity\[h\]/,
    "the sole variational moving-solid channel must commit u_solid with aperture and pressure scale");
  assert.doesNotMatch(structuredBoundarySource,
    /constrainedFaceVelocity|fullVector|divergenceRhs/,
    "the boundary transaction must not introduce a second full-vector or RHS authority");
  assert.doesNotMatch(octreeProjectionShader, /fn assembleSystem|fn assembleCoarseSystem/,
    "the deleted variable-row assembly shader must remain absent");
  assert.match(octreeProjectionShader,
    /crossesBoundary = crossesClosedWall \|\| \(denseSolidField && crossesSolidBoundary\)[\s\S]*if \(!fluidGatedBoundaryRefinement\) \{ return true; \}[\s\S]*boundaryLiquidMinimumPhi[\s\S]*minimumPhi <= params\.solve\.w \* params\.cellRelax\.x/,
    "the control must refine every solid interface while the experiment gates dry crossings by the liquid look-ahead band");
});

test("octree topology is genuinely three-dimensional and 2:1 balanced", () => {
  assert.match(octreeProjectionShader, /packOrigin\(p: vec3u\)/);
  assert.match(octreeProjectionShader, /fn packOrigin\(p: vec3u\) -> u32 \{ return index\(p\); \}/,
    "owner identity is the exact full-domain linear cell index");
  assert.doesNotMatch(octreeProjectionShader, /p\.z << 20u/,
    "owner identity must not retain the legacy 10:10:10 coordinate cap");
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  assert.match(rebuild, /this\.balanceRounds/);
  assert.match(rebuild, /this\.refinementSizes/);
  assert.doesNotMatch(rebuild, /Math\.ceil\(Math\.log2\(this\.maxLeafSize\)\)/,
    "the immutable dyadic level schedule must not be rebuilt per transaction");
  assert.match(octreeProjectionShader, /for \(var z = 0u; z < size/);
  assert.match(octreeProjectionShader, /for \(var y = 0u; y < size/);
  assert.match(octreeProjectionShader, /for \(var x = 0u; x < size/);
  assert.match(octreeProjectionShader, /neighborTooFine/);
  assert.match(octreeProjectionShader, /\.size \* 2u < size/);
});

test("coarse topology loops remain runtime-bounded at maximum leaf 16 and 32", () => {
  const refine = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn refineCoarseBlock"),
    octreeProjectionShader.indexOf("fn neighborTooFine"),
  );
  const balance = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn balanceCoarseBlock"),
    octreeProjectionShader.indexOf("fn pressureOf"),
  );
  assert.match(refine, /let size = workgroupUniformLoad\(&refineRuntimeSize\)/);
  assert.match(refine, /for \(var flat = lid; flat < solidCellsInLeaf; flat \+= 128u\)/,
    "the cubic solid predicate must be distributed across the coarse workgroup");
  assert.match(refine, /refineBoundaryRange\[lid\] = boundaryRange[\s\S]*stride = 64u/,
    "coarse lanes must reduce solid range and liquid phi cooperatively");
  assert.match(octreeProjectionShader,
    /if \(!denseSolidField && !crossesClosedWall\) \{ return false; \}/,
    "solid-free interior scenes must specialize away the entire cubic storage traversal");
  assert.match(balance, /let size = workgroupUniformLoad\(&balanceRuntimeSize\)/);
  assert.doesNotMatch(refine, /let size = targetRefinementSize/,
    "pipeline specialization must not unroll the size-cubed scan at 16/32");
  assert.doesNotMatch(balance, /let size = targetRefinementSize/,
    "pipeline specialization must not unroll the coarse balance loops at 16/32");
});

test("power topology has no surface-driven compatibility refinement", () => {
  assert.doesNotMatch(octreeProjectionShader, /levelSetIn: texture_3d<f32>/,
    "the sparse octree topology must not bind a box-sized level-set texture");
  assert.match(octreeProjectionShader, /fn correctedCoarsePhi\(point:vec3f\)->CorrectedCoarsePhi/,
    "topology classification must consume the compact corrected coarse-phi authority");
  assert.doesNotMatch(octreeProjectionShader, /volumeIn/, "the octree solve must not bind the diagnostic VOF field");
  assert.match(octreeProjectionShader, /if \(minimumSolid >= 1\.0 - 1e-5\) \{ return false; \}/,
    "fully solid bulk leaves should be allowed to stay coarse");
  assert.doesNotMatch(octreeProjectionShader,
    /surfaceDrivenRefinement|authoritativePowerTopology|closestSurface|crossesSurface/,
    "the separate Section 5 narrow band must be the only free-surface refinement owner");
  assert.doesNotMatch(octreeProjectionShader,
    /surfaceDetailStrengthValue|detailActivity|maximumCurvatureProxy|strainActivity/,
    "the retired detail-refinement branch must not survive the clean cut");
  assert.doesNotMatch(octreeProjectionShader, /closestSurface \* adaptivity < f32\(size\) \* finestWidth/);
  const refinement = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn leafNeedsRefinement"), octreeProjectionShader.indexOf("fn splitLeaf"));
  assert.doesNotMatch(refinement, /wet != liquidCell|a > 0\.001/);
});

test("post-bootstrap topology cuts directly to global-fine summaries and compact coarse phi", () => {
  assert.doesNotMatch(octreeProjectionShader, /pagedSurfaceAuthority|findFineSeedLeaf|surfacePagePhi/);
  assert.match(octreeProjectionShader, /let coarse=correctedCoarsePhi/);
  assert.doesNotMatch(octreeSource, /WebGPUOctreeSurfacePages|pagedGroups|encodeDensePublication/);
  assert.match(octreeSource, /releaseDenseBootstrapPhi\(\)[\s\S]*this\.surfaceState\.releasePresentationTexture\(\)/,
    "the box-sized bootstrap phi must be releasable after its submission");
  assert.match(octreeSource, /fineSeedAdapter\?\.setCoarsePhiSource/,
    "compact leaves must classify from coarse octree phi after cold bootstrap");
  assert.match(uniformSolverSource, /this\.device\.queue\.submit\(\[encoder\.finish\(\)\]\);[\s\S]*releaseDenseBootstrapPhi\(\)/,
    "release happens only after the bootstrap commands have been submitted");
});

test("dense phi lifetime gate requires every recurring compact consumer handoff", () => {
  const ready = {
    globalFineBootstrapped: true,
    coarseProjectionGroupsActive: true,
    fineSeedCoarseNative: true,
    topologyUsesFineSeedCandidates: true,
    compactRendererSourceReady: true,
    incompatibleDenseConsumer: false,
  };
  assert.equal(octreeDensePhiReleaseReady(ready), true);
  for (const key of [
    "globalFineBootstrapped", "coarseProjectionGroupsActive",
    "fineSeedCoarseNative", "topologyUsesFineSeedCandidates",
    "compactRendererSourceReady",
  ] as const) assert.equal(octreeDensePhiReleaseReady({ ...ready, [key]: false }), false, key);
  assert.equal(octreeDensePhiReleaseReady({ ...ready, incompatibleDenseConsumer: true }), false);
  assert.match(octreeSource, /fineSeedCoarseNative: this\.fineSeedAdapter\?\.hasCoarsePhiBindings === true/);
  assert.match(octreeSource, /topologyUsesFineSeedCandidates: this\.topologyWorklistReady/);
});

test("octree preserves advected level-set volume with GPU-only feedback", () => {
  assert.match(quadtreeSurfaceShader, /fn correctLevelSetVolume/);
  assert.match(quadtreeSurfaceShader, /let desiredVolume = params\.inflowTiming\.y/);
  assert.match(quadtreeSurfaceShader, /atomicLoad\(&reductions\[0\]\)/);
  assert.match(quadtreeSurfaceShader, /u32\(occupied \* open \* 256\.0 \+ 0\.5\)/, "mixed open-cell volume reduction must be unbiased");
  assert.match(quadtreeSurfaceShader, /abs\(value\) < 2\.0 \* params\.cellAndDt\.y/, "controller derivative must use the Heaviside support");
  assert.match(quadtreeSurfaceShader, /occupied \* open \* 256\.0/, "volume control must conserve liquid only in the open fraction of each cell");
  assert.match(quadtreeSurfaceShader, /surfaceSolids\[index3\(gid\)\]\.fraction/, "the controller must consume the octree's current VOS field");
  assert.match(octreeSource, /true, true, this\.hasDenseSolidCells \? this\.solidCells : undefined/,
    "octree must bind freshly rasterized solid fractions only when the dense solid field exists");
  assert.match(octreeSource, /async readSurfaceDiagnostics\(\) \{\s*return this\.surfaceState\.readVolumeDiagnostics\(\);/,
    "legacy page-volume telemetry must not survive the global-fine cutover");
  assert.match(legacyUniformComputeShader, /let represented=surfaceOccupancy\(id\)\*open/, "reported liquid volume must exclude the same displaced solid fraction");
  assert.match(legacyUniformComputeShader, /for\(var step=1;step<=64;step\+=1\)/, "phi-s must reach open liquid across a newly submerged large solid in one pass");
  assert.match(legacyUniformComputeShader, /exteriorSum\+=exteriorWeight\*textureLoad\(pressureIn,exterior,0\)\.x/, "solid-interior phase must be extended from exterior fluid samples");
  assert.match(uniformSolverSource, /gridKind: "octree",[\s\S]*?volumeControl: true/, "octree diagnostics must report that GPU volume control is active");
  assert.doesNotMatch(quadtreeSurfaceShader, /params\.control\.y > 1\.5/);
  assert.match(octreeSource, /undefined, false, false, true, true/, "octree enables GPU phi-volume correction and topology-preserving transport");
  const encode = WebGPUQuadtreeSurfaceState.prototype.encode.toString();
  assert.match(encode, /monotoneLevelSetTransport[\s\S]*advectLevelSet[\s\S]*advectPredict/, "octree can select monotone phi transport without changing other surface users");
});

test("octree shared fine-grid dynamics use the resident level set as their sole liquid authority", () => {
  const projectionConstruction = uniformSolverSource.indexOf("else if (options.octree)");
  const authorityBinding = uniformSolverSource.indexOf("const surfaceAuthority = this.adaptiveProjection?.levelSetTexture ?? this.volumeA");
  assert.ok(projectionConstruction >= 0 && authorityBinding > projectionConstruction, "surface bind groups must be created after the octree level set exists");
  assert.match(uniformSolverSource, /surfaceIn: GPUTexture = this\.adaptiveProjection\?\.levelSetTexture \?\? volumeIn/);
  assert.doesNotMatch(uniformSolverSource, /surfaceIn: GPUTexture = this\.quadtreeProjection\?\.levelSetTexture/);
  assert.match(uniformSolverSource, /this\.transportConservativeVolume = !this\.octreeProjection/);
  assert.match(uniformSolverSource, /if \(this\.transportConservativeVolume\) \{\s+prep\.setPipeline\(this\.buildFluxScalesPipeline\)/);
  assert.match(uniformSolverSource, /this\.densitySharpening && this\.transportConservativeVolume/);
  assert.match(uniformSolverSource, /this\.adaptiveProjection \? 1 : 0, sigma/, "octree must interpret the shared authority texture as signed distance");
  assert.match(legacyUniformComputeShader, /fn transportConservativeVolume\(\) -> bool/);
  assert.match(legacyUniformComputeShader, /if\(transportConservativeVolume\(\)\)\{advected=advectedVolume\(id,dt\);\}/);
});

test("octree rebuild and solve stay resident on the GPU", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString();
  const coupledCandidate = (WebGPUOctreeProjection.prototype as unknown as {
    encodeInactiveCoupledPowerCandidate: () => void;
  }).encodeInactiveCoupledPowerCandidate.toString();
  const solve = WebGPUOctreeProjection.prototype.encode.toString();
  const frontierRows = (WebGPUOctreeProjection.prototype as unknown as {
    encodeFrontierRows: () => void;
  }).encodeFrontierRows.toString();
  assert.match(rebuild, /new PassBroker\(encoder\)/);
  assert.doesNotMatch(rebuild, /beginComputePass/,
    "the rebuild must cut over completely to broker-owned compute passes");
  assert.doesNotMatch(solve, /projectionParity|clearBuffer/,
    "the retired face-projection parity reset must not survive the clean cut");
  assert.doesNotMatch(`${rebuild}\n${coupledCandidate}\n${solve}`, /mapAsync|getMappedRange/);
  // Device-local copies publish row-parallel indirect args and stage the
  // residual/count telemetry before the next rebuild can reuse compaction;
  // none of these copies is a CPU readback or dense-field transfer.
  assert.match(rebuild, /this\.encodeFrontierRows\(/,
    "the inactive candidate must publish compact indirect arguments before its coupled validation");
  assert.doesNotMatch(solve, /this\.encodeFrontierRows\(/,
    "the solve graph must not mutate or recompact its accepted epoch");
  const copies = (`${frontierRows}\n${solve}`.match(/copyBufferToBuffer\([^)]*\)/g) ?? [])
    .map((copy) => copy.replace(/\s+/g, ""));
  assert.deepEqual(copies, [
    "copyBufferToBuffer(this.compaction,8,this.solveDispatch,0,24)",
    "copyBufferToBuffer(this.compaction,this.compactionByteLength-32,this.solveStats,0,32)",
  ]);
  assert.doesNotMatch(`${frontierRows}\n${solve}`, /pressureOverflowDispatch/,
    "the retired dense overflow dispatch must not survive as device-local backing");
  assert.match(octreeProjectionShader, /var<storage, read_write> owners/);
  assert.match(octreeProjectionShader, /pressureIn: array<f32>/);
  assert.match(uniformSolverSource, /encodeReadyTopologyFlip\(encoder\)[\s\S]*this\.octreeProjection\.encodeSurface\(encoder, dt[\s\S]*this\.octreeProjection\.encode\([\s\S]*this\.octreeProjection\.encodeInactiveTopologyCandidate\(encoder\)/,
    "every substep must consume one immutable epoch and build its successor only at the tail");
});

test("octree telemetry samples the live compacted pressure-row count", () => {
  const diagnostics = WebGPUOctreeProjection.prototype.readSolveDiagnostics.toString();
  assert.match(diagnostics, /copyBufferToBuffer\(this\.solveStats,\s*0,\s*readback,\s*0,\s*32\)/,
    "diagnostics must read the solve-ordered staging buffer instead of racing the reused compaction arena");
  assert.match(diagnostics, /const liquidRows\s*=\s*words\[1\]/);
  assert.match(diagnostics, /pressureCapacityOverflow\s*=\s*overflow/);
  assert.match(diagnostics, /this\.info\.liquidDofCount\s*=\s*liquidRows/);
  assert.match(diagnostics, /this\.info\.pressureSampleCount\s*=\s*liquidRows/);
  assert.match(diagnostics, /this\.residualRms\s*=\s*Math\.sqrt\(rr\s*\/\s*liquidRows\)/,
    "octree diagnostics must publish the measured PCG residual instead of retaining the solver-info zero default");
  assert.match(diagnostics, /this\.relativeResidual\s*=\s*Math\.sqrt\(rr\s*\/\s*Math\.max\(bb,\s*1e-30\)\)/);
  assert.match(diagnostics,
    /const solverControl\s*=\s*this\.pipelinedMGPCG\.control[\s\S]*copyBufferToBuffer\(solverControl,\s*0,\s*readback,\s*32,\s*64\)/,
    "diagnostics must sample MGPCG authority in the same readback");
  assert.match(diagnostics, /this\.applyMGPCGDiagnostics\(new Uint32Array\(mapped,\s*32,\s*16\)\)/,
    "iteration and convergence telemetry must come from MGPCG control");
  assert.doesNotMatch(diagnostics, /updateSolveBudget/,
    "CPU residual feedback must not alter the immutable recorded solve");
});

test("octree has no executable variable-row assembly before the structured operator", () => {
  assert.doesNotMatch(octreeProjectionShader, /LeafEntry|leafEntries|assembleSystem|assembleCoarseSystem/);
  const rebuild = WebGPUOctreeProjection.prototype.encodeInactiveTopologyCandidate.toString().replace(/\s+/g, "");
  assert.match(rebuild, /candidates\.setBindGroup\(0,active\?this\.fineSummarySizingGroup:this\.groups\.ab\)/,
    "recurring candidate publication must bind the current fine-summary directory");
  const encode = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.doesNotMatch(encode, /assemblePipeline|assembleCoarsePipeline|leafEntries/);
  assert.match(encode, /encodeNativePowerAssembly/);
});

test("accepted SPGrid coefficients are reused from the committed tail candidate", () => {
  const assembly = octreeSource.slice(
    octreeSource.indexOf("  private encodeNativePowerAssembly("),
    octreeSource.indexOf("\n  private encodeStructuredProjection(",
      octreeSource.indexOf("  private encodeNativePowerAssembly(")),
  );
  assert.doesNotMatch(assembly, /firstOrderVCycle\.encodeCapture/,
    "assembly must not rebuild the inactive hierarchy that the prior tail already committed");
  assert.match(assembly, /preceding tail[\s\S]*committed at[\s\S]*substep's head/,
    "the omitted recapture is guarded by the explicit tail-to-head epoch contract");
});

test("recurring fine redistance keeps its warm start as the production default", () => {
  assert.equal(fineLevelSetWarmStartRequested({}), true);
  assert.equal(fineLevelSetWarmStartRequested({ FLUID_FINE_JFA_WARM_START: "1" }), true);
  assert.equal(fineLevelSetWarmStartRequested({ FLUID_FINE_JFA_WARM_START: "0" }), false,
    "the cold path remains available only as an explicit process-local oracle");
});

test("eligible octrees select persistent MGPCG by default with a hierarchical oracle", () => {
  assert.equal(octreePersistentMGPCGEnabled({}), true);
  assert.equal(octreePersistentMGPCGEnabled({ FLUID_OCTREE_PERSISTENT_MGPCG: "1" }), true);
  assert.equal(octreePersistentMGPCGEnabled({ FLUID_OCTREE_PERSISTENT_MGPCG: "0" }), false);
  assert.match(octreeSource,
    /globalFineLevelSet\?\.plan\.fineFactor !== 1\s*&& WebGPUOctreePersistentMGPCG\.selects\(rowCapacity\)/,
    "factor 1 must not construct the persistent experiment");
  assert.match(octreeSource,
    /const persistent = this\.globalFineLevelSet\?\.plan\.fineFactor !== 1\s*&& octreePersistentMGPCGEnabled\(\) \? this\.persistentMGPCG : undefined;/,
    "factor 1 must not select a dormant persistent executor at encode time");
});

test("octree delegates every capacity to the direct-curvature PCG authority", () => {
  const encode = WebGPUOctreePipelinedMGPCG.prototype.encode.toString();
  assert.doesNotMatch(encode, /executionMode|persistentMGPCG|WebGPUOctreeMGPCG/);
  assert.match(encode, /iteration\s*<\s*encodedIterationBudget/);
  assert.match(encode,
    /encodedIterationBudget[\s\S]*this\.device\.queue\.writeBuffer\([\s\S]*this\.params/,
    "the GPU exhaustion gate must receive the exact host-encoded prefix");
  assert.match(encode, /"reduceMergedPartials"/);
  assert.match(encode, /"finishMergedReduction"/);
  assert.match(encode, /"updateDirections"/);
  assert.doesNotMatch(encode, /Chebyshev|Spectrum|beginSpectrum|fallback/i,
    "direct-curvature PCG is the sole outer solver");
  assert.doesNotMatch(WebGPUOctreeProjection.prototype.encode.toString(),
    /iterateChebyshevPipeline|leafSolver|Math\.ceil\(this\.iterations\s*\/\s*4\)/,
    "the projection host must not retain a second solver schedule");
});

test("compact face authority shapes only a stale-safe factor-one encode prefix", () => {
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode,
    /const fullSolveBudget\s*=\s*this\.pipelinedMGPCG\.iterationBudget/);
  assert.match(encode,
    /selectOctreeFactorOneEncodedSolveTail[\s\S]*factorOne:[\s\S]*fineFactor\s*===\s*1/);
  assert.match(encode,
    /this\.info\.pressureIterationBudget\s*=\s*solveBudget[\s\S]*this\.info\.pressureIterationHardBudget\s*=\s*this\.solveTailPolicy\.hardOuterIterationCeiling/);
  assert.match(encode,
    /encodedIterationBudget:\s*solveBudget/,
    "recorded accounting and the GPU exhaustion gate consume the same selected prefix");
  assert.doesNotMatch(encode, /faceTransport\)\s*return|pressureIterationsUsed/,
    "live transport state and racing solver telemetry must not shape the schedule");
  assert.match(uniformSolverSource,
    /structuredStepWorkObservation\(record\)[\s\S]*observeFactorOneSolveTail\(\{[\s\S]*publishedIterationCount:\s*observation\.executedSolveIterations[\s\S]*acceptedTopologyEpoch:\s*observation\.acceptedEpoch/,
    "the selector may consume only the step-coherent snapshot observation");
  assert.match(uniformSolverSource,
    /this\.octreeProjection\.encode\([\s\S]*step:\s*this\.info\.encodedSteps/,
    "the encoded solve and its snapshot must share one exact step stamp");
});

test("projection WGSL contains only topology and compact-row publication authority", () => {
  assert.doesNotMatch(octreeProjectionShader,
    /fn (?:iterateLeaves|iterateChebyshev|solveLeaves|reduceResidualPartials|projectSmallLeaves|projectLeaves|passThroughPressureOverflow|extrapolateSeed|extrapolateSparse)/,
    "retired solver, dense projector, overflow, and extrapolation entry points must stay deleted");
  assert.match(octreePipelinedMGPCGShader, /fn reduceMergedPartials/);
  assert.match(octreePipelinedMGPCGShader, /fn finishMergedReduction/);
  assert.doesNotMatch(octreePipelinedMGPCGShader,
    /fn (?:iterateChebyshev|updateChebyshevState|beginChebyshevSpectrum)/);
});

test("rediscretized A2 owns coarse-fine contacts through spatial pages", () => {
  assert.match(spgridVCycleSource,
    /fn rebuildCandidateGhostsFor[\s\S]*catalogCoefficients\[base\+1u\+channel\][\s\S]*cLookup\(l\+1u,vec3u\(targetQ\)\/2u\)/,
    "catalog directions must spawn fine aliases only when the physical target resolves to a coarse active owner");
  assert.match(spgridVCycleSource,
    /fn finerAdjoint[\s\S]*pageSlot\(fine,ghostPage,ghostQ,vec3u\(activeQ\),row\)[\s\S]*coefficientForDirection\(other,otherMetric,delta\)/,
    "destination-owned E^T must gather coarse-fine contacts through physical page adjacency");
  assert.match(spgridVCycleSource,
    /fn applyRow[\s\S]*section63Coefficients\[base\+1u\+channel\][\s\S]*value\+=c\*\(x-inputVector\[encoded-1u\]\)/,
    "each destination row must own its dynamically published Section 6.3 coefficient gather");
  assert.match(spgridVCycleSource,
    /applyRegularInterior[\s\S]*applyTransitionInterior[\s\S]*applyPhysicalBoundary[\s\S]*applyTransitionBoundary/,
    "accurate A2 must execute the four topology-published row classes separately");
});

test("structured boundary rows use exact fine-over-coarse level-set crossings", () => {
  assert.match(structuredBoundarySource,
    /let loPoint=rowCenter\(lo\);let world=worldBoundaryBit\(h\)!=0u;var hiPoint=handleCenter\(h\);if\(hi!=INVALID\)\{hiPoint=rowCenter\(hi\);\}else if\(!world\)[\s\S]*let plo=fineSample\(loPoint,clo\.x\);let phi=fineSample\(hiPoint,chi\.x\)/,
    "each structured contact must sample both actual power-cell centres from fine phi with coarse phi as its explicit authority");
  assert.match(structuredBoundarySource,
    /if\(!world&&\(hi==INVALID\|\|\(\(plo\.x<0\.\)!=\(phi\.x<0\.\)\)\)\)/,
    "a crossing must be resolved for an open face whose ghost point flips sign");
  // theta is measured from whichever side is wet and floored at 1e-2; it is
  // deliberately not the uncapped crossing. A sliver drives the 1/theta face
  // coefficient without bound, and hard-failing the legitimately dry-centre
  // geometry (the frontier admits rows the interface has not reached yet)
  // poisoned the whole topology epoch, permanently freezing the accepted
  // velocity authority. The recordTheta histogram buckets are chosen around
  // this same floor, so restoring an uncapped crossing silently invalidates
  // bucket 0 as well.
  assert.match(structuredBoundarySource,
    /let theta=max\(-plo\.x\/\(phi\.x-plo\.x\),1e-2\);[\s\S]*let theta=max\(-phi\.x\/\(plo\.x-phi\.x\),1e-2\);/,
    "the crossing must be measured from whichever side is wet, floored at 1e-2");
  assert.match(structuredBoundarySource,
    /if\(!finite\(theta\)\|\|theta>1\.\)\{fail\(h,8u\|1024u\);return;\}[\s\S]*scale=1\.\/theta;/,
    "a crossing outside the face must fail the slot closed rather than clamp into range");
  assert.match(structuredBoundarySource,
    /\}else\{\n?\s*\/\/ No wet side[\s\S]*scale=0\.;/,
    "an open face with no wet side must decouple instead of coupling through air");
  assert.doesNotMatch(octreeProjectionShader,
    /fn reconstructGradients|fn storeReconstructedGradient|fn projectedComponentCached/,
    "affine dense projection was superseded by the compact power-face operator");
  assert.doesNotMatch(octreeProjectionShader, /fn assembleSystem|fn assembleCoarseSystem/,
    "the topology shader must not retain variable-row pressure assembly");
  assert.doesNotMatch(structuredBoundarySource, /ownerPhiGradient/,
    "the structured boundary transaction must use direct crossing samples, not a reconstructed phi gradient");
  assert.match(octreeDiagnosticShader,
    /textureStore\(pressureOut, vec3i\(gid\), vec4f\(select\(0\.0, centrePressure, wet\)\)\)/,
    "the pressure overlay must expose the compact row value without dead affine-gradient storage");
  assert.doesNotMatch(octreeDiagnosticShader, /leafGradient|@binding\(9\).*gradients/,
    "diagnostics must not retain the deleted dense affine-gradient ABI");
  assert.match(octreeDiagnosticShader, /vec4u\(row, 0u, vertical, horizontal\)/,
    "generic pressure ownership stays available without reviving dense Projection Δu");
  assert.doesNotMatch(octreeDiagnosticShader,
    /velocityBeforeProjection|var velocity: texture_3d|pressureUpdate|divergenceOut/,
    "compact velocity diagnostics belong to the native technique overlay");
});

test("retired dense affine-gradient projection remains deleted", () => {
  assert.doesNotMatch(octreeProjectionShader,
    /gradientPartials|coarseGradientContribution|reconstructSmallGradients|reconstructedAxisGradient/,
    "compact generalized faces are the only projection authority");
  assert.doesNotMatch(octreeSource,
    /reconstructSmallGradientsPipeline|reconstructGradientsPipeline|projectSmallLeavesPipeline|projectLeavesPipeline/);
});

test("octree materializes adaptive overlay fields without a readback", () => {
  assert.match(octreeDiagnosticShader, /texture_storage_3d<rg32uint, write>/);
  assert.match(octreeDiagnosticShader, /origin\.x \| \(origin\.z << 10u\) \| \(owner\.size << 20u\)/);
  assert.match(octreeDiagnosticShader, /origin\.y \| \(\(origin\.y \+ owner\.size\) << 10u\)/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureSamplesOut/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureOut/);
  assert.doesNotMatch(octreeDiagnosticShader,
    /var velocity(?:BeforeProjection)?: texture_3d|textureLoad\(velocity|divergenceOut|pressureUpdate/);
  assert.doesNotMatch(octreeDiagnosticShader, /mapAsync|getMappedRange/);
  assert.match(rendererSource, /gridKind === "quadtree-tall-cell" \|\| gpuInfo\?\.gridKind === "octree" \? 1 : 0/);
});

test("octree dense diagnostic textures are allocated only on overlay demand", () => {
  const constructor = octreeSource.slice(octreeSource.indexOf("constructor("), octreeSource.indexOf("private descriptor("));
  assert.doesNotMatch(constructor, /createTexture\(\{ label: "Octree overlay/);
  assert.match(octreeSource, /ensureDiagnosticTextures\(\): boolean/);
  assert.match(uniformSolverSource, /ensureGridDiagnosticTextures\(\)/);
  assert.match(rendererSource, /gridOverlay\?\.axis !== "off"[\s\S]{0,100}ensureGridDiagnosticTextures/);
});

test("structured solve dispatches are class-exact and convergence gated", () => {
  assert.match(structuredVelocitySource,
    /if\(clean&&cls<=4u\)\{let dispatch=6u\+3u\*cls;publicationDispatch\[dispatch\]=groupsX;publicationDispatch\[dispatch\+1u\]=groupsY/,
    "the accepted structured epoch must publish one exact indirect record per resolved row class");
  // The published record is two-dimensional. A class whose block count passes
  // maxComputeWorkgroupsPerDimension is silently zeroed by the indirect-args
  // validator, so a saturating class must pin X and carry the remainder in Y.
  assert.match(structuredVelocitySource,
    /let groupsX=max\(1u,min\(65535u,groups\)\);let groupsY=\(groups\+groupsX-1u\)\/groupsX;/,
    "class dispatch records must fold their block count into two dimensions");
  assert.match(structuredVelocitySource, /fn foldedItem\(g:vec3u\)->u32\{return g\.x\+g\.y\*65535u\*64u;\}/,
    "slot-shaped kernels must recover their linear item from the pinned X extent");
  assert.match(spgridVCycleSource,
    /classDispatch\[destination\]=select\(0u,worksets\[source\],solveLive&&valid\)/,
    "the convergence gate must zero future class work without host intervention");
  // The apply is no longer a per-class loop over records 0..3; those four
  // gated records now size the aggregate passes rather than being dispatched
  // one at a time, and the aggregates live directly above them in the same
  // buffer. The invariant the loop protected is unchanged: every accurate A2
  // gather reaches the GPU through a record the convergence gate can zero, so
  // a converged solve costs no host round trip to stop.
  assert.match(spgridVCycleSource,
    /const ACCURATE_UNION_DISPATCH_OFFSET_BYTES = 4 \* 12;\s*const ACCURATE_TERM_DISPATCH_OFFSET_BYTES = 5 \* 12;\s*const ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES = 6 \* 12;/,
    "the aggregate records must follow the four gated class records in one buffer");
  for (const offset of ["ACCURATE_TERM_DISPATCH_OFFSET_BYTES",
    "ACCURATE_ADJOINT_DISPATCH_OFFSET_BYTES", "ACCURATE_UNION_DISPATCH_OFFSET_BYTES"]) {
    assert.match(spgridVCycleSource,
      new RegExp(`pass\\.dispatchWorkgroupsIndirect\\(this\\.accurateClassDispatch, ${offset}\\)`),
      `the accurate A2 apply must consume its gated indirect record at ${offset}`);
  }
  const accurateApply = spgridVCycleSource.slice(
    spgridVCycleSource.indexOf("private encodeAccurateWorksets("),
    spgridVCycleSource.indexOf("private encodeAccurateMergedBandWorkset("));
  assert.ok(accurateApply.length > 0, "the accurate A2 apply must remain one readable method");
  assert.equal((accurateApply.match(/pass\.dispatchWorkgroups\(/g) ?? []).length, 1,
    "inside the apply only the record publication may dispatch directly");
  assert.equal((accurateApply.match(/pass\.dispatchWorkgroupsIndirect\(/g) ?? []).length, 3,
    "every other accurate A2 pass must reach the GPU through a gated record");
  assert.doesNotMatch(octreeProjectionShader, /fn compactRowIndex\(gid: vec3u\)/,
    "the deleted variable-row executor must not return to the topology shader");
});

test("pressure topology follows current fine-interface support without tile-wide retention", () => {
  assert.match(octreeProjectionShader,
    /fn pressureRefinementEvidence\(origin: vec3u, size: u32\)[\s\S]*inflowProtectionIntersects\(origin, size\)/,
    "authored pressure apertures must retain local topology protection");
  const refinementEvidence = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("fn pressureRefinementEvidence"),
    octreeProjectionShader.indexOf("fn pressureRefinementProtected"),
  );
  assert.match(refinementEvidence,
    /fineLeafSummary\(origin, size\)[\s\S]*gradingLayers \* max\(0\.0, f32\(size\) - 2\.0\)[\s\S]*if \(crossesInterface\) \{ return true; \}[\s\S]*if \(size <= 2u\) \{[\s\S]*return false;[\s\S]*observedNearInterface && \(summary\.complete \|\| fineSummaryFactor == 1u\)/,
    "size-two power cells must split only on an actual crossing while coarser candidates retain progressive grading");
  assert.match(octreeProjectionShader,
    /classifyTopologyTileSignature[\s\S]*pressureRefinementEvidence\(unpackOrigin\(owner\.packedOrigin\), owner\.size\)[\s\S]*compaction\[base \+ 4u\] = TILE_SIGNATURE_VALID_MAGIC/,
    "the structural signature must hash current spatial pressure evidence");
  assert.doesNotMatch(octreeProjectionShader,
    /PRESSURE_RETENTION_GENERATIONS|fn pressureRetentionAt|priorRetention|retention << 24u/,
    "tile activity must not retain unrelated dry cells");
  assert.match(octreeProjectionShader,
    /fn pressureRefinementProtected\(origin: vec3u, size: u32\) -> bool \{[\s\S]*return pressureRefinementEvidence\(origin, size\);/,
    "refinement protection must be a current spatial decision");
  assert.match(octreeProjectionShader,
    /fn leafNeedsRefinement[\s\S]*pressureRefinementProtected\(origin, size\)[\s\S]*if \(!denseSolidField && !crossesClosedWall\) \{ return false; \}/,
    "solid-free dam scenes must refine interface support before the cheap solid-free exit");
  assert.match(octreeProjectionShader,
    /fn boundaryLiquidMinimumPhi[\s\S]*if \(summary\.found\) \{ return summary\.minimumPhi; \}[\s\S]*3\.402823e38[\s\S]*coarse\.leafSize == 0u/,
    "the authoritative positive-air complement must not masquerade as a near-interface distance");
  assert.match(octreeProjectionShader,
    /fn refineCoarseBlock[\s\S]*pressureRefinementProtected\(origin, size\)/,
    "large dyadic leaves must use the same protection predicate as fine leaves");
});

test("retired topology tiles invalidate their old frontier identities", () => {
  assert.match(octreeProjectionShader,
    /for \(var slot = 0u; slot < compaction\[4\]; slot \+= 1u\)[\s\S]*compaction\[tileChangeFlagsBase\(\) \+ tileIndex\] = generation;[\s\S]*appendDirtyTileRing\(tileIndex, generation, &dirtyCount\)/,
    "a retired tile must stamp its own old rows even though it is absent from the active dirty list");
  assert.match(octreeProjectionShader,
    /fn rowAuthorityDirtyGeneration[\s\S]*compaction\[tileChangeFlagsBase\(\)\+index\]==generation/,
    "frontier carry must consume the retired-tile structural stamp");
});
