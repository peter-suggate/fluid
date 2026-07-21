import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { simulationMethods } from "../lib/methods";
import { octreeMethod } from "../lib/methods/octree";
import { OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS } from "../lib/octree-face-fragments";
import { legacyUniformComputeShader } from "../lib/webgpu-eulerian";
import { OCTREE_GENERATED_POWER_CATALOG_MANIFEST } from "../lib/generated/octree-power-catalog";
import { defaultScene } from "../lib/model";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES, octreeDensePhiReleaseReady, octreeDiagnosticShader, octreePressureCouplingShader, octreeProjectionPipelineRequired, octreeProjectionShader, planOctreeCompactionAllocation, planOctreeLeafFrontierAllocation, planOctreePressureCapacity, WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { quadtreeSurfaceShader, WebGPUQuadtreeSurfaceState } from "../lib/webgpu-quadtree-builder";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const uniformSolverSource = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
const smokeSource = readFileSync(new URL("../tools/run-webgpu-smoke.ts", import.meta.url), "utf8");
const packageManifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
  scripts: Record<string, string>;
};

test("row-indexed octree startup skips the unreachable dense Jacobi specialization", () => {
  const product = {
    leafSolver: "mgpcg",
    segmentedProjection: true,
    denseVelocityProjection: false,
    solidRasterization: false,
    extrapolationSweeps: 0,
    sparseExtrapolation: false,
    hasDensePhiSnapshot: false,
  } as const;
  assert.equal(octreeProjectionPipelineRequired("jacobi", product), false);
  assert.equal(octreeProjectionPipelineRequired("project", product), false);
  assert.equal(octreeProjectionPipelineRequired("reconstructSmallGradients", product), false);
  assert.equal(octreeProjectionPipelineRequired("reconstructGradients", product), false);
  assert.equal(octreeProjectionPipelineRequired("projectLeaves", product), false);
  assert.equal(octreeProjectionPipelineRequired("passThroughPressureOverflow", product), false);
  assert.equal(octreeProjectionPipelineRequired("rasterizeSolids", product), false);
  assert.equal(octreeProjectionPipelineRequired("extrapolate", product), false);
  assert.equal(octreeProjectionPipelineRequired("refreshSnapshotDense", product), false);
  assert.equal(octreeProjectionPipelineRequired("projectLeaves", {
    ...product, denseVelocityProjection: true,
  }), true, "dense compatibility retains its segmented leaf projector");
  assert.equal(octreeProjectionPipelineRequired("rasterizeSolidsActive", {
    ...product, solidRasterization: true,
  }), true, "terrain and rigid scenes retain solid rasterization");
  assert.equal(octreeProjectionPipelineRequired("jacobi", { ...product, leafSolver: "dense" }), true,
    "the explicit dense validation solver must retain its Jacobi pipeline");
  assert.equal(octreeProjectionPipelineRequired("assembleSystem", product), true);
  assert.match(octreeSource, /if \(!this\.basePipelineRequired\(entryPoint\)\) return/);
  assert.match(octreeSource, /compiled\.find\(\(pipeline\) => pipeline !== undefined\)/,
    "body-free fresh-device startup must fill unreachable slots from the first reachable pipeline, not slot zero");
  assert.match(octreeSource, /for \(let size = Math\.min\(8, this\.maxLeafSize\); size >= 2; size >>= 1\)/,
    "fine refinement startup must not compile coarse or unreachable leaf sizes");
  assert.match(octreeSource, /for \(let size = this\.maxLeafSize; size >= 16; size >>= 1\)/,
    "coarse refinement startup must stop at this solver's immutable maximum");
  assert.match(octreeSource, /if \(!this\.faceTransport\) \{[\s\S]*this\.reconstructSmallGradientsPipeline[\s\S]*this\.pressureOverflowPipeline/,
    "compact-face authority must not encode the retired dense velocity projection chain");
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

test("octree is a registered GPU method with dam-break defaults", () => {
  assert.ok(simulationMethods.includes(octreeMethod));
  assert.equal(octreeMethod.id, "octree");
  assert.equal(octreeMethod.backend, "webgpu");
  assert.equal(octreeMethod.presetFor("balanced").pressureIterations, 128);
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
  assert.equal(octreeMethod.presetFor("balanced").adaptivity, 1);
  assert.equal(octreeMethod.params.find((spec) => spec.key === "secondaryParticles")?.default, "off");
  assert.equal(octreeMethod.presetFor("balanced").secondaryParticles, "off");
  assert.match(octreeMethod.detail, /no topology readbacks/);
  assert.match(octreeMethod.detail, /Section 4\.3 hybrid PCG/);
  assert.match(octreeMethod.detail, /fail-closed paper authority/);
  assert.match(octreeMethod.detail, /rigid-body coupling/);
  assert.match(octreeMethod.description, /signed-distance level set/);
  const maximumLeaf = octreeMethod.params.find((spec) => spec.key === "maximumLeafSize");
  assert.ok(maximumLeaf && maximumLeaf.kind === "select");
  assert.equal(maximumLeaf.default, "16");
  assert.deepEqual(maximumLeaf.options.map((option) => option.value), ["2", "4", "8", "16", "32"]);
  assert.match(octreeSource, /function octreeLeafSize\(value: number\): 2 \| 4 \| 8 \| 16 \| 32/);
  assert.match(octreeSource, /rounded >= 32/);
  const interfaceBand = octreeMethod.params.find((spec) => spec.key === "interfaceRefinementBandCells");
  assert.ok(interfaceBand && interfaceBand.kind === "number" && interfaceBand.tier === "fine" && interfaceBand.default === 4);
  const surfaceRefinement = octreeMethod.params.find((spec) => spec.key === "surfaceRefinementFactor");
  assert.ok(surfaceRefinement && surfaceRefinement.kind === "select" && surfaceRefinement.default === "2");
  assert.deepEqual(surfaceRefinement.options.map((option) => option.value), ["1", "2", "4"]);
  const globalFine = octreeMethod.params.find((spec) => spec.key === "globalFineLevelSetFactor");
  assert.ok(globalFine && globalFine.kind === "select" && globalFine.default === "4");
  assert.deepEqual(globalFine.options.map((option) => option.value), ["off", "4", "8"]);
  const powerProjection = octreeMethod.params.find((spec) => spec.key === "powerDiagramProjection");
  assert.ok(powerProjection && powerProjection.kind === "select" && powerProjection.default === "authoritative");
  assert.deepEqual(powerProjection.options.map((option) => option.value), ["off", "mirror", "authoritative"]);
  assert.equal(octreeMethod.presetFor("balanced").globalFineLevelSetFactor, "4",
    "the balanced product path must request the factor-4 global fine lattice");
  assert.equal(octreeMethod.presetFor("balanced").powerDiagramProjection, "authoritative",
    "the balanced product path must request power authority");
  assert.equal(octreeMethod.presetFor("balanced").leafSolver, "auto",
    "auto admits Section 4.3 only after power authority passes its fail-closed policy");
  for (const quality of ["high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).globalFineLevelSetFactor, "off",
      `${quality} must remain an explicit compatibility preset until its memory/endurance gate passes`);
    assert.equal(octreeMethod.presetFor(quality).powerDiagramProjection, "off");
  }
  assert.match(smokeSource, /FLUID_OCTREE_POWER_PROJECTION/);
  assert.match(smokeSource, /FLUID_OCTREE_GLOBAL_FINE_FACTOR/);
  assert.match(octreeSource, /pageResolution: options\.surfaceRefinementFactor === 4 \? 4 : 2/,
    "surface consumers must default to the bandwidth-oriented 2-cubed page ABI");
  assert.match(octreeSource, /surfacePageResolution: plan\?\.pageResolution \?\? 2/);
  assert.match(octreeProjectionShader, /override surfacePageResolution: u32 = 2u/);
  assert.doesNotMatch(`${octreeSource}\n${uniformSolverSource}`, /airRefinementBandCells/);
  assert.match(uniformSolverSource, /interfaceRefinementBandCells: options\.octree\.interfaceRefinementBandCells \?\? 4/);
  const surfaceDetail = octreeMethod.params.find((spec) => spec.key === "surfaceDetailStrength");
  assert.ok(surfaceDetail && surfaceDetail.kind === "number" && surfaceDetail.default === 0);
  for (const quality of ["balanced", "high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).surfaceDetailStrength, 0, "dynamic refinement must be uniformly opt-in across quality presets");
  }
  const particleCorrection = octreeMethod.params.find((spec) => spec.key === "secondaryParticleSurfaceCorrection");
  assert.ok(particleCorrection && particleCorrection.kind === "number" && particleCorrection.default === 0);
  for (const quality of ["balanced", "high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).secondaryParticleSurfaceCorrection, 0, "particle feedback must be uniformly opt-in across quality presets");
  }
  const adaptiveVelocity = octreeMethod.params.find((spec) => spec.key === "faceVelocityTransport");
  assert.ok(adaptiveVelocity && adaptiveVelocity.kind === "select" && adaptiveVelocity.default === "on");
  for (const quality of ["balanced", "high", "ultra"] as const) {
    assert.equal(octreeMethod.presetFor(quality).faceVelocityTransport, "on",
      "compact octree-face velocity must be enabled across production quality presets");
  }
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(methodSource, /faceVelocityTransport: values\.faceVelocityTransport !== false && values\.faceVelocityTransport !== "off"/,
    "missing saved values must migrate to the compact default");
  assert.match(methodSource, /requiresCompatibilityGeometry\(scene\)[\s\S]*\? undefined : globalFineLevelSetFactor/,
    "terrain and imported\/seeded geometry must not inherit the balanced factor-4 allocation");
  assert.match(uniformSolverSource, /adaptiveFaceRhsIsSupported\([\s\S]*sceneHasTerrain\(scene\)[\s\S]*scene\.rigidBodies\.length[\s\S]*this\.hydrostaticSplit/,
    "the dense host allocation must remain fail-closed for unsupported scenes");
  assert.match(octreeSource, /this\.extrapolationSweeps = faceTransportEnabled \? 0 : requestedExtrapolationSweeps/,
    "compact face authority must not retain the full-domain texture extrapolation ladder");
  const warmStart = octreeMethod.params.find((spec) => spec.key === "pressureWarmStart");
  assert.ok(warmStart && warmStart.kind === "select" && warmStart.tier === "fine" && warmStart.default === "on");
  // Options are copied field-by-field into the solver; a dropped key would
  // silently revert the UI toggle to its default.
  assert.match(octreeSource, /pressureWarmStart\?: boolean/);
  assert.match(uniformSolverSource, /pressureWarmStart: options\.octree\.pressureWarmStart/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode.replace(/\s+/g, ""), /constremapGroup=this\.latestPressureInA\?this\.groups\.ab:this\.groups\.ba/);
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
  assert.match(command, /FLUID_OCTREE_POWER_PROJECTION=authoritative/);
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

test("compact octree pressure capacity scales with domain surface area", () => {
  const dims = { nx: 288, ny: 96, nz: 64 };
  const count = dims.nx * dims.ny * dims.nz;
  const plan = planOctreePressureCapacity(dims, 16, 4);
  assert.equal(plan.rowCapacity, 316_928);
  assert.equal(plan.entryCapacity, plan.rowCapacity * 6 * OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS);
  const powerPlan = planOctreePressureCapacity(dims, 16, 4, undefined, true);
  assert.equal(powerPlan.entryCapacity, powerPlan.rowCapacity * Math.max(
    6 * OCTREE_FACE_FRAGMENT_MAX_FINE_NEIGHBORS, OCTREE_GENERATED_POWER_CATALOG_MANIFEST.maximumNeighborRows));
  assert.ok(plan.rowCapacity < count / 5, "the widened ocean must not reserve one pressure slot per finest cell");
  const oldBytes = count * (2 * 4 + 32 + 48);
  assert.ok(plan.pressureBytes + plan.headerBytes + plan.entryBytes < oldBytes * 0.55,
    "compact pressure/header/entry arenas should remove at least 45% of the former dense-sized allocation");
  assert.equal(planOctreePressureCapacity(dims, 16, 4, 1024).rowCapacity, 1024);
});

test("compact authority replaces the dense frontier row map with an exact-key hash", () => {
  const cellCount = 320 * 96 * 80;
  const rowCapacity = planOctreePressureCapacity({ nx: 320, ny: 96, nz: 80 }, 16, 4).rowCapacity;
  assert.equal(rowCapacity, 388_864);
  const compact = planOctreeLeafFrontierAllocation(cellCount, rowCapacity, true);
  assert.deepEqual(compact, {
    cellCount: 2_457_600,
    listCapacity: 388_864,
    hashCapacity: 524_288,
    denseOriginMapBytes: 9_830_400,
    rowMapBytes: 4_194_304,
    allocatedBytes: 7_305_232,
    denseCompatibilityBytes: 29_491_216,
    savedBytes: 22_185_984,
  });
  const compatibility = planOctreeLeafFrontierAllocation(cellCount, rowCapacity, false);
  assert.equal(compatibility.listCapacity, cellCount);
  assert.equal(compatibility.hashCapacity, 0);
  assert.equal(compatibility.rowMapBytes, compatibility.denseOriginMapBytes);
  assert.equal(compatibility.allocatedBytes, compatibility.denseCompatibilityBytes);
  assert.equal(compatibility.savedBytes, 0);
  assert.throws(() => planOctreeLeafFrontierAllocation(0, rowCapacity, true), /positive integer/);
  assert.throws(() => planOctreeLeafFrontierAllocation(cellCount, 0, true), /positive integer/);
});

test("Dawn smoke never reads compact face authority as a dense velocity texture", () => {
  assert.match(smokeSource, /const compactFaceVelocity = stagedSolver\.adaptiveFaceVelocitySource !== undefined/);
  assert.match(smokeSource, /if \(!texture \|\| compactFaceVelocity\) return Promise\.resolve\(undefined\)/);
  assert.match(smokeSource, /velocityTexture && final && !finalSolver\.adaptiveFaceVelocitySource/);
  assert.match(smokeSource, /adaptiveFaceTransportedCount === faces\?\.faceCount/);
  assert.match(smokeSource, /compact transport reported invalid or zero CFL/);
});

test("compact scan and coarse-task scratch follows pressure and active-tile bounds", () => {
  const dims = { nx: 640, ny: 192, nz: 160 };
  const pressure = planOctreePressureCapacity(dims, 16, 4);
  const activeTileCapacity = 4_800;
  const activeTileWorklistBytes = (16 + 2 * activeTileCapacity) * 4;
  const compact = planOctreeCompactionAllocation(
    dims, pressure.rowCapacity, activeTileWorklistBytes, activeTileCapacity, 16, true,
  );
  assert.deepEqual(compact, {
    scanBlockCapacity: 6_150,
    coarseTaskCapacity: 38_400,
    scanAndTaskBytes: 381_444,
    activeTileBytes: 57_632,
    allocatedBytes: 439_076,
  });
  const compatibility = planOctreeCompactionAllocation(
    dims, pressure.rowCapacity, activeTileWorklistBytes, activeTileCapacity, 16, false,
  );
  assert.deepEqual(compatibility, {
    scanBlockCapacity: 76_800,
    coarseTaskCapacity: 38_400,
    scanAndTaskBytes: 1_229_244,
    activeTileBytes: 57_632,
    allocatedBytes: 1_286_876,
  });
  assert.ok(compact.allocatedBytes < compatibility.allocatedBytes * 0.35);
  assert.throws(() => planOctreeCompactionAllocation(dims, 0, 0, 0, 16, true), /positive integer/);
  assert.throws(() => planOctreeCompactionAllocation(dims, 1, -1, 0, 16, true), /active-tile bounds/);
  assert.match(octreeProjectionShader, /return params\.pressureCapacity\.z/,
    "the publication guard must consume the exact host-planned task capacity");
  assert.match(octreeProjectionShader, /total\.z > coarseTaskCapacity\(\)/,
    "a corrupted or oversized coarse task publication must fail closed before indirect dispatch");
  assert.match(octreeProjectionShader, /coarseTasks = select\(tiles \* tiles \* tiles, 1u, rowIndexedPressure\)/,
    "compact authority must publish one cooperative task per coarse pressure row");
  assert.match(octreeProjectionShader, /coarseTaskIndex\(wid\)/,
    "coarse indirect work must retain every row through a two-dimensional dispatch");
});

test("octree hydrostatic reference is a default-off fixed-rest-surface A/B", () => {
  const control = octreeMethod.params.find((spec) => spec.key === "hydrostaticSplit");
  assert.ok(control && control.kind === "select" && control.default === "off");
  assert.equal(octreeMethod.presetFor("balanced").hydrostaticSplit, "off");
  for (const method of simulationMethods.filter((candidate) => candidate.id !== "octree")) {
    assert.equal(method.params.some((spec) => spec.key === "hydrostaticSplit"), false,
      `${method.id} must not expose the octree hydrostatic experiment`);
  }
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(methodSource, /hydrostaticSplit: values\.hydrostaticSplit === "on"/);
  assert.match(uniformSolverSource, /this\.hydrostaticSplit = options\.hydrostaticSplit === true[\s\S]*scene\.fluid\.initialCondition === "tank-fill"[\s\S]*scene\.fluid\.inflow === undefined[\s\S]*scene\.rigidBodies\.length === 0/);
  assert.match(smokeSource, /FLUID_HYDROSTATIC_SPLIT/);
  assert.match(smokeSource, /method\.id === "octree" && hydrostaticSplitOverride/);

  assert.match(legacyUniformComputeShader, /texture_storage_2d<rg32float, write>/);
  assert.match(legacyUniformComputeShader, /fn hydrostaticSplit\(\) -> bool/);
  assert.match(legacyUniformComputeShader, /if\(!hydrostaticSplit\(\)\)[\s\S]*vec4f\(highest,-1\.0,0\.0,0\.0\)/,
    "the disabled path must retain the historical top-down occupancy scan and publish no reference");
  assert.match(legacyUniformComputeShader, /!referenceEnded&&!cellInsideTerrain\(p\)/);
  assert.match(legacyUniformComputeShader, /let wet=surfaceLiquid\(p\)/);
  assert.match(legacyUniformComputeShader, /eta=f32\(referenceTop\)\+0\.5\+crossing/,
    "eta must use the signed-distance zero crossing rather than an integer cell top");
  assert.match(uniformSolverSource, /hydrostaticSplit: this\.hydrostaticSplit/);
  assert.match(uniformSolverSource, /setHydrostaticTimestep\(dt\)/);
  assert.match(uniformSolverSource, /c\.fillFraction\*this\.info\.ny/);
  assert.match(octreeSource, /scene\.fluid\.initialCondition === "tank-fill"/);
  assert.match(octreeSource, /scene\.fluid\.inflow === undefined/);
  assert.match(octreeSource, /scene\.rigidBodies\.length === 0/,
    "the fixed datum must fail closed for dam breaks and pressure-coupled bodies");
  assert.match(octreeProjectionShader, /fn fixedHydrostaticPotential\(y_m: f32\)/);
  assert.match(octreeProjectionShader, /return -fixedHydrostaticPotential\(surfaceY\)/,
    "the perturbation Dirichlet value must make total pressure zero at the actual surface");
  assert.match(octreeProjectionShader, /header\.rhs = flux - boundarySum/,
    "assembly must move the known perturbation boundary pressure to the RHS");
  assert.match(octreeProjectionShader, /let airPressure = hydrostaticAirPressure\(left, right, leftPhi, rightPhi\)/,
    "projection must restore the same nonzero perturbation boundary value");
  assert.match(octreeProjectionShader, /pressureDistanceFromPhi\(left, right, axis, leftPhi, rightPhi\)/,
    "the perturbation boundary must use the projection's ghost-fluid distance");
  assert.match(uniformSolverSource, /!this\.hydrostaticSplit && this\.sparseOccupancyFluxPreparationRequested/,
    "the maximum-only sparse occupancy path cannot author connected eta");
});

test("compact pressure rows publish origin ranks and fail closed on capacity overflow", () => {
  assert.match(octreeProjectionShader, /override rowIndexedPressure: bool = true/);
  assert.match(octreeProjectionShader, /return select\(0xffffffffu, word - 2u, word >= 2u\)/);
  assert.match(octreeProjectionShader, /fn frontierSetRow\(cell:u32,word:u32\)/);
  assert.match(octreeProjectionShader, /let key=cell\+1u/);
  assert.match(octreeProjectionShader, /probe<32u/);
  assert.match(octreeProjectionShader, /firstTombstone=0xffffffffu/,
    "frontier insertion must remember, rather than immediately reuse, a tombstone");
  assert.match(octreeProjectionShader, /if\(stored==key\)\{return frontierClaimValue\(at\);\}/,
    "frontier insertion must find a same-key slot later in a tombstone-bearing probe cluster");
  assert.match(octreeProjectionShader, /if\(stored==0u\)\{emptySlot=slot;break;\}/,
    "only an actual empty slot may terminate the same-key search");
  assert.match(octreeProjectionShader, /claim\.exchanged\|\|claim\.old_value==key\)\{return frontierClaimValue\(at\);\}/,
    "the key winner and same-key contenders must share one value-word append arbitration");
  assert.match(octreeProjectionShader, /fn frontierListCapacity\(\) -> u32/);
  assert.match(octreeProjectionShader, /required > frontierListCapacity\(\)/);
  assert.match(octreeProjectionShader, /compaction\[control\] = 2u/,
    "frontier-list overflow must carry a distinct fail-closed diagnostic bit");
  assert.match(octreeProjectionShader, /pressureOut\[row\] = select\(0\.0, warm/);
  assert.match(octreeProjectionShader, /LeafEntry \{ row: u32, coefficient: f32 \}/);
  assert.match(octreeProjectionShader, /total\.x > params\.pressureCapacity\.x \|\| total\.y > params\.pressureCapacity\.y/);
  assert.match(octreeProjectionShader, /fn passThroughPressureOverflow/);
  assert.match(WebGPUOctreeProjection.prototype.encode.toString(), /dispatchWorkgroupsIndirect\(this\.pressureOverflowDispatch,\s*0\)/);
  assert.match(octreePressureCouplingShader, /rowIndexedPressure/);
  assert.match(octreeDiagnosticShader, /fn pressureRow\(owner: Owner\)/);
});

test("octree participates in the shared two-way immersed-body coupling path", () => {
  const methodSource = readFileSync(new URL("../lib/methods/octree.ts", import.meta.url), "utf8");
  assert.match(methodSource, /values, onRigidLoads\) => new WebGPUUniformEulerianSolver\(device, scene, quality, onRigidLoads/);
  assert.match(methodSource, /values, onRigidLoads, onProgress, signal\) => WebGPUUniformEulerianSolver\.createAsync/);
  assert.match(uniformSolverSource, /const activeBodies = bodies\.slice\(0, 12\)/);
  assert.doesNotMatch(uniformSolverSource, /this\.octreeProjection \? \[\] : bodies/);
  assert.match(uniformSolverSource, /if \(this\.adaptiveProjection\) this\.solidPhiGroup/);
  assert.match(uniformSolverSource, /texture: this\.adaptiveProjection\.levelSetTexture/);
  assert.match(uniformSolverSource, /this\.dispatch\(pass, this\.rigidPipeline, this\.rigidGroup\)/);
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

test("octree pressure solve uses the variational solid face constraint", () => {
  const assemble = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn assembleSystem"), octreeProjectionShader.indexOf("fn iterateLeaves"));
  assert.match(assemble, /let open = 1\.0 - clamp\(solid\.fraction/);
  assert.match(assemble, /let coefficient = open \* area \/ max\(distance/);
  assert.match(assemble, /constrainedFaceVelocity\(faceCell, axis, solid\)/);
  assert.match(octreeProjectionShader, /open \* component\(velocityAt\(faceCell\), axis\) \+ solid\.fraction \* component\(solidVelocity/);
  assert.match(octreeProjectionShader, /if \(crossesSurface \|\| crossesSolidBoundary\) \{ return true; \}/,
    "solid interfaces must force finest octree leaves");
});

test("octree retains exact rank-six coupling as an A/B path and defaults to lagged feedback", () => {
  assert.match(octreeProjectionShader, /K M\^-1 K\^T/);
  assert.match(octreeProjectionShader, /fn gatherBodyCoupling/);
  assert.match(octreeProjectionShader, /faceArea\(axis\) \* solid\.fraction \* \(p1 - p0\)/);
  assert.match(octreeProjectionShader, /let coupling = leafBodyCoupling/);
  assert.match(octreeProjectionShader, /effectiveDiagonal = header\.diagonal \+ coupling\.y/);
  assert.match(octreeProjectionShader, /fn applyBodyCoupling/);
  assert.match(octreeProjectionShader, /body\.linearVelocity\.xyz - linear/);
  assert.match(octreeProjectionShader, /body\.angularVelocity\.xyz - angular/,
    "the exact compact ladder must retain its same-step pressure-updated body velocity");
  assert.match(octreePressureCouplingShader, /params\.physical\.x \* faceArea\(axis\) \* solid\.fraction \* \(p0 - p1\)/);
  assert.match(octreePressureCouplingShader, /0\.5 \* f32\(owner\.size - 1u\)/,
    "pressure-to-body coupling must classify leaves at the same geometric centre as projection");
  assert.match(octreePressureCouplingShader, /atomicAdd\(&rigidExchange\[base/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /gatherBodyCoupling\(pressure,group\)/);
  assert.match(encode, /constuseChebyshev=this\.leafSolver==="chebyshev"/);
  assert.match(encode, /constuseLaggedRigidCoupling=useChebyshev&&this\.couplingHasDynamicBodies/);
  assert.match(encode, /if\(!useLaggedRigidCoupling\)gatherBodyCoupling\(project,finalGroup\)/);
  assert.match(encode, /this\.pressureImpulsePipeline/,
    "lagged coupling must still publish the current pressure impulse for the next batch");
});

test("octree topology is genuinely three-dimensional and 2:1 balanced", () => {
  assert.match(octreeProjectionShader, /packOrigin\(p: vec3u\)/);
  assert.match(octreeProjectionShader, /fn packOrigin\(p: vec3u\) -> u32 \{ return index\(p\); \}/,
    "owner identity is the exact full-domain linear cell index");
  assert.doesNotMatch(octreeProjectionShader, /p\.z << 20u/,
    "owner identity must not retain the legacy 10:10:10 coordinate cap");
  assert.match(WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString(), /Math\.ceil\(Math\.log2\(this\.maxLeafSize\)\)/);
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
    octreeProjectionShader.indexOf("fn hydrostaticSplit"),
  );
  assert.match(refine, /leafNeedsRefinement\(origin, owner\.size\)/,
    "the cubic sizing scan must use the storage-loaded leaf size");
  assert.match(refine, /let size = workgroupUniformLoad\(&refineRuntimeSize\)/);
  assert.match(balance, /let size = workgroupUniformLoad\(&balanceRuntimeSize\)/);
  assert.doesNotMatch(refine, /let size = targetRefinementSize/,
    "pipeline specialization must not unroll the size-cubed scan at 16/32");
  assert.doesNotMatch(balance, /let size = targetRefinementSize/,
    "pipeline specialization must not unroll the coarse balance loops at 16/32");
});

test("octree refinement is graded by resident signed distance rather than bulk VOF occupancy", () => {
  assert.match(octreeProjectionShader, /levelSetIn: texture_3d<f32>/);
  assert.doesNotMatch(octreeProjectionShader, /volumeIn/, "the octree solve must not bind the diagnostic VOF field");
  assert.match(octreeProjectionShader,
    /if \(!fineSummary\.complete\) \{\s*samplePhi=legacyPhi\(vec3i\(q\)\);closestSurface=min\(closestSurface,abs\(samplePhi\)\)/,
    "an incomplete indexed summary must scan the selected page/analytic source rather than infer a missing sparse sample");
  assert.match(octreeProjectionShader, /if \(minimumSolid >= 1\.0 - 1e-5\) \{ return false; \}/,
    "fully solid bulk leaves should be allowed to stay coarse");
  assert.match(octreeProjectionShader, /let effectiveBand = baseBand \+ 8\.0 \* detailActivity/);
  assert.match(octreeProjectionShader, /return closestSurface < effectiveBand \* finestWidth;/,
    "pure air and liquid leaves should use the explicit band plus bounded local detail support");
  assert.match(octreeProjectionShader, /surfaceDetailStrengthValue\(\) \* clamp\(max\(strainActivity, 2\.0 \* maximumCurvatureProxy\)/);
  assert.match(octreeProjectionShader, /surfaceDetailStrengthValue\(\) > 0\.0/,
    "zero decoded detail strength must skip activity sampling while analytic sparse sentinels retain baseline sizing");
  assert.doesNotMatch(octreeProjectionShader, /closestSurface \* adaptivity < f32\(size\) \* finestWidth/);
  const refinement = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn leafNeedsRefinement"), octreeProjectionShader.indexOf("fn splitLeaf"));
  assert.doesNotMatch(refinement, /wet != liquidCell|a > 0\.001/);
  assert.match(octreeProjectionShader, /fn liquidCell\(p: vec3i\) -> bool \{ return valid\(p\) && phi\(p\) < 0\.0; \}/);
});

test("post-bootstrap topology samples compact surface pages and affine leaf fallbacks directly", () => {
  assert.match(octreeProjectionShader, /fn pagedSurfaceAuthority\(\)/);
  assert.match(octreeProjectionShader, /fn findSurfaceLeaf\(p: vec3u\)/);
  assert.match(octreeProjectionShader, /surfaceHashCoord\(p \/ size\)/);
  assert.match(octreeProjectionShader, /fn surfacePagePhi\(row: u32, point: vec3f\)/);
  assert.match(octreeProjectionShader, /return surfaceLeafFallback\(row, point\)/);
  assert.match(octreeSource, /FLUID_OCTREE_DIRECT_PAGED_PHI !== "0"[\s\S]*if \(this\.directPagedTopology && !this\.surfacePagesBootstrapped && this\.pagedGroups\)[\s\S]*this\.groups = this\.pagedGroups/,
    "the direct sampler is the default authority after the one-time bootstrap");
  assert.match(octreeSource, /releaseDenseBootstrapPhi\(\)[\s\S]*this\.surfaceState\.releasePresentationTexture\(\)/,
    "the box-sized bootstrap phi must be releasable after its submission");
  assert.match(octreeSource, /this\.groups = this\.pagedGroups[\s\S]*faceMirror\?\.setSurfacePageSource\(this\.adaptiveSurfacePages\.source, this\.levelSetFallbackTexture!\)[\s\S]*adaptiveSurfaceAdapter\.setSurfacePageSource\(this\.adaptiveSurfacePages\.source, this\.levelSetFallbackTexture!\)/,
    "projection, face, and surface-adapter groups must hand off before dense phi can be destroyed");
  assert.match(uniformSolverSource, /this\.device\.queue\.submit\(\[encoder\.finish\(\)\]\);[\s\S]*releaseDenseBootstrapPhi\(\)/,
    "release happens only after the bootstrap commands have been submitted");
});

test("dense phi lifetime gate requires every recurring compact consumer handoff", () => {
  const ready = {
    directPagedTopology: true,
    surfacePagesBootstrapped: true,
    pagedProjectionGroupsActive: true,
    faceGroupsPageNative: true,
    surfaceAdapterPageNative: true,
    topologyUsesSurfaceCandidates: true,
    compactRendererSourceReady: true,
    incompatibleDenseConsumer: false,
  };
  assert.equal(octreeDensePhiReleaseReady(ready), true);
  for (const key of [
    "directPagedTopology", "surfacePagesBootstrapped", "pagedProjectionGroupsActive",
    "faceGroupsPageNative", "surfaceAdapterPageNative", "topologyUsesSurfaceCandidates",
    "compactRendererSourceReady",
  ] as const) assert.equal(octreeDensePhiReleaseReady({ ...ready, [key]: false }), false, key);
  assert.equal(octreeDensePhiReleaseReady({ ...ready, incompatibleDenseConsumer: true }), false);
  assert.match(octreeSource, /faceGroupsPageNative: this\.faceMirror\?\.hasPageNativePhiBindings === true/);
  assert.match(octreeSource, /surfaceAdapterPageNative: this\.adaptiveSurfaceAdapter\?\.hasPageNativePhiBindings === true/);
  assert.match(octreeSource, /topologyUsesSurfaceCandidates: this\.topologyWorklistReady/);
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
  assert.match(octreeSource, /async readSurfaceDiagnostics\(\) \{[\s\S]*?if \(!this\.adaptiveSurfacePages\) return this\.surfaceState\.readVolumeDiagnostics\(\);[\s\S]*?await this\.adaptiveSurfacePages\.readDiagnostics\(\);[\s\S]*?return this\.surfaceDiagnostics;/,
    "octree telemetry must report the adaptive surface authority, with dense level-set diagnostics only as its compatibility fallback");
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
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  const solve = WebGPUOctreeProjection.prototype.encode.toString();
  const frontierRows = (WebGPUOctreeProjection.prototype as unknown as {
    encodeFrontierRows: () => void;
  }).encodeFrontierRows.toString();
  assert.match(rebuild, /beginComputePass/);
  assert.match(solve, /clearBuffer/);
  assert.doesNotMatch(`${rebuild}\n${solve}`, /mapAsync|getMappedRange/);
  // Device-local copies publish row-parallel indirect args and stage the
  // residual/count telemetry before the next rebuild can reuse compaction;
  // none of these copies is a CPU readback or dense-field transfer.
  assert.match(solve, /this\.encodeFrontierRows\(/,
    "the solve must refresh compact indirect dispatch arguments before assembly");
  const copies = (`${frontierRows}\n${solve}`.match(/copyBufferToBuffer\([^)]*\)/g) ?? [])
    .map((copy) => copy.replace(/\s+/g, ""));
  assert.deepEqual(copies, [
    "copyBufferToBuffer(this.compaction,8,this.solveDispatch,0,24)",
    "copyBufferToBuffer(this.compaction,this.compactionByteLength-20,this.pressureOverflowDispatch,0,12)",
    "copyBufferToBuffer(this.compaction,this.compactionByteLength-32,this.solveStats,0,32)",
  ]);
  assert.match(octreeProjectionShader, /var<storage, read_write> owners/);
  assert.match(octreeProjectionShader, /pressureIn: array<f32>/);
  assert.match(uniformSolverSource, /if \(substep > 0 && this\.octreeProjection\) \{[\s\S]*Octree substep topology timing start[\s\S]*this\.octreeProjection\.encodeInlineRebuild\(encoder\)[\s\S]*Octree substep topology timing end[\s\S]*\}/,
    "CFL subdivision must rebuild from the level set transported by the preceding substep and include that work in topology timing");
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
    "octree diagnostics must publish the measured Chebyshev residual instead of retaining the solver-info zero default");
  assert.match(diagnostics, /this\.relativeResidual\s*=\s*Math\.sqrt\(rr\s*\/\s*Math\.max\(bb,\s*1e-30\)\)/);
  assert.match(diagnostics, /this\.updateSolveBudget\(rr,\s*bb,\s*liquidRows\)/);
});

test("octree compacted leaf solve scans, assembles once, and iterates over rows only", () => {
  // The persistent frontier uses atomics only while evolving active topology
  // tiles. Pressure-row rank/scatter remains a prefix scan over that frontier.
  assert.match(octreeProjectionShader, /fn beginFrontier/);
  assert.match(octreeProjectionShader, /fn filterFrontier/);
  assert.match(octreeProjectionShader, /fn appendFrontierActive/);
  assert.match(octreeProjectionShader, /fn appendFrontierRetired/);
  assert.match(octreeProjectionShader, /fn finalizeFrontier/);
  assert.match(octreeProjectionShader, /fn planLeaves/);
  assert.match(octreeProjectionShader, /fn scanLeafBlocks/);
  assert.match(octreeProjectionShader, /fn emitLeaves/);
  assert.match(octreeProjectionShader, /var running = scanPairs\[lid\] - sum;/);
  const emit = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn emitLeaves"), octreeProjectionShader.indexOf("fn compactRowIndex"));
  assert.match(emit, /frontierCell\(current, slot\)/);
  assert.doesNotMatch(emit, /liquidOwner|ownerPhi|textureLoad/,
    "row emission must not reclassify the finest lattice or resample phi");
  // Assembly caches diagonal, RHS flux, and a merged neighbor table, so the
  // per-iteration kernels never touch the velocity texture or owner map.
  const assemble = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn assembleSystem"), octreeProjectionShader.indexOf("fn iterateLeaves"));
  assert.match(assemble, /neighborCoefficients\[j\] \+= coefficient/);
  assert.match(assemble, /flux \+= f32\(side\) \* area \* /);
  const iterate = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn iterateLeaves"), octreeProjectionShader.indexOf("fn leafPressureLoad"));
  assert.doesNotMatch(iterate, /velocityAt|ownerAt|textureLoad/);
  assert.match(iterate, /entry\.coefficient \* pressureIn\[entry\.row\]/);
});

test("octree assembles coarse leaf faces cooperatively with deterministic quadrants", () => {
  const start = octreeProjectionShader.indexOf("fn assembleCoarseSystem");
  const end = octreeProjectionShader.indexOf("fn iterateLeaves", start);
  const coarse = octreeProjectionShader.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(octreeProjectionShader, /if \(owner\.size >= 8u\)[\s\S]*coarseTasks = select\(tiles \* tiles \* tiles, 1u, rowIndexedPressure\)/);
  assert.match(coarse, /@builtin\(local_invocation_index\) lid/);
  assert.match(coarse, /workgroupUniformLoad\(&coarseTaskEligible\) == 0u/);
  assert.match(coarse, /sample = lid; sample < faceSamples; sample \+= 64u/);
  assert.match(coarse, /quadrant \* 64u \+ lid/);
  assert.match(coarse, /header\.entryStart \+ face \* 4u \+ quadrant/);
  assert.match(coarse, /header\.entryCount = 24u/);
  assert.doesNotMatch(coarse, /atomicAdd/,
    "coarse face coefficients must reduce deterministically rather than accumulate atomically");
  const encode = WebGPUOctreeProjection.prototype.encode.toString().replace(/\s+/g, "");
  assert.match(encode, /this\.assemblePipeline[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,0\)[\s\S]*this\.assembleCoarsePipeline[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,12\)/);
});

test("octree Chebyshev solve removes three quarters of global iteration boundaries", () => {
  const start = octreeProjectionShader.indexOf("fn iterateChebyshev");
  const end = octreeProjectionShader.indexOf("// Relative-residual feedback", start);
  const chebyshev = octreeProjectionShader.slice(start, end);
  assert.ok(start >= 0 && end > start);
  assert.match(chebyshev, /params\.solve\.y/);
  assert.match(chebyshev, /params\.solve\.z/);
  assert.match(chebyshev, /leafHeaders\[row\]\.pad0 = bitcast<u32>\(search\)/);
  assert.match(chebyshev, /leafHeaders\[row\]\.pad1 = bitcast<u32>\(rho\)/);
  assert.doesNotMatch(chebyshev, /workgroupBarrier|storageBarrier|ownerAt|textureLoad/, "each pass stays row-parallel and reduction-free");
  const feedback = octreeProjectionShader.slice(end, octreeProjectionShader.indexOf("fn leafPressureLoad", end));
  assert.match(feedback, /fn reduceResidualPartials/);
  assert.match(feedback, /fn reduceResidualTotal/);
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /Math\.ceil\(this\.iterations\s*\/\s*4\)/);
  assert.match(encode, /this\.iterateChebyshevPipeline/);
  assert.doesNotMatch(encode, /this\.leafSolver === "chebyshev" && !this\.couplingHasDynamicBodies/,
    "dynamic bodies must not force the accelerated solve back onto the dispatch ladder");
});

test("compact face authority keeps a deterministic full Chebyshev budget", () => {
  assert.match(octreeSource, /if \(this\.faceTransport\) return;/);
});

test("octree megakernel keeps barriers in uniform control flow and folds parity back", () => {
  const solve = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn solveLeaves"));
  assert.match(solve, /workgroupUniformLoad\(&convergedShared\)/);
  assert.match(solve, /workgroupUniformLoad\(&rowCountShared\)/);
  assert.match(solve, /storageBarrier\(\)/);
  assert.match(solve, /pressureIn\[pressureRow\] = pressureOut\[pressureRow\]/);
  assert.match(solve, /tolerance2 \* normB/);
});

test("octree pressure traverses coarse-fine faces by finest subfaces", () => {
  const face = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn accumulateFace"), octreeProjectionShader.indexOf("@compute @workgroup_size(4,4,4)\nfn jacobi"));
  assert.match(face, /for \(var b = 0u; b < size/);
  assert.match(face, /for \(var a = 0u; a < size/);
  assert.match(face, /ownerAt\(outside\)/);
  assert.match(face, /let distance = (?:pressureDistance\(ownerAt\(vec3i\(origin\)\), neighbor, axis\)|0\.5 \* f32\(size \+ neighbor\.size\) \* h)/);
  if (octreeProjectionShader.includes("fn pressureDistance")) {
    assert.match(octreeProjectionShader, /let full = 0\.5 \* f32\(a\.size \+ b\.size\) \* cellWidth\(axis\)/);
  }
  assert.match(face, /area \/ max\(distance/);
});

test("octree uses the level-set crossing and prolongates pressure inside coarse leaves", () => {
  assert.match(octreeProjectionShader, /fn ownerPhi\(owner: Owner\)/);
  assert.match(octreeProjectionShader, /0\.5 \* f32\(owner\.size - 1u\)/,
    "even-sized leaf pressure samples must use the geometric centre rather than an upper fine cell");
  assert.match(octreeProjectionShader, /fn pressureDistance\(a: Owner, b: Owner, axis: u32\)/);
  assert.match(octreeProjectionShader, /abs\(liquidPhi\) \/ max\(abs\(liquidPhi\) \+ abs\(airPhi\)/,
    "the free-surface pressure boundary must lie at phi=0");
  assert.match(octreeProjectionShader, /fn reconstructGradients/);
  assert.match(octreeProjectionShader, /fn storeReconstructedGradient/);
  assert.match(octreeProjectionShader, /header\.gradient = vec4f\(gradient, 0\.0\)/,
    "post-solve gradient reconstruction must overwrite xyz while keeping the unused component inert");
  assert.doesNotMatch(octreeProjectionShader, /gradient\.w/,
    "the dead compact gradient padding component must not retain a hidden consumer");
  const assembly = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn assembleSystem"),
    octreeProjectionShader.indexOf("fn iterateLeaves"));
  assert.equal(assembly.match(/header\.gradient = vec4f\(0\.0\)/g)?.length, 2,
    "both serial and cooperative assembly must leave gradients for the post-solve reconstruction stage");
  assert.doesNotMatch(assembly, /ownerPhiGradient\(owner\)/,
    "assembly must not compile the expensive level-set gradient sampler into either entry point");
  const project = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectedComponentCached"), octreeProjectionShader.indexOf("fn extrapolate"));
  assert.match(project, /left\.packedOrigin != right\.packedOrigin[\s\S]*pressureDistanceFromPhi\(left, right, axis, leftPhi, rightPhi\)/,
    "leaf-boundary faces must retain the exact assembled variational gradient");
  assert.match(project, /else if \(leftWet && rightWet\) \{\s+fluid -= reconstructedGradient\(left, axis\)/,
    "dense faces inside a coarse leaf must no longer be invisible to pressure");
  assert.match(octreeDiagnosticShader, /mappedPressure = centrePressure \+ dot\(leafGradient\(owner\), offset\)/,
    "the pressure overlay must expose the same affine field used by projection");
  assert.match(octreeDiagnosticShader, /bitcast<u32>\(pressureUpdate\)/,
    "the pressure-ownership texture must expose the applied velocity update without a readback");
});

test("power descriptors bind the shared octree topology rather than the phase-row index", () => {
  const assembly = octreeSource.slice(
    octreeSource.indexOf("private encodePowerAssemblyMirror"),
    octreeSource.indexOf("private encodePowerProjection"),
  );
  assert.match(assembly, /descriptor\.encode\(encoder, this\.leafHeaders, this\.topology,/,
    "descriptor bits must describe the one shared balanced octree");
  assert.match(assembly, /ownerMode: this\.ownerPages \? "paged" : "dense"/);
  assert.doesNotMatch(assembly, /descriptor\.encode\([^;]*faces\.siteIndex/,
    "the liquid-row index cannot author geometry for absent air leaves");
});

test("octree reconstructs coarse affine gradients with a cooperative face reduction", () => {
  const reconstruct = octreeProjectionShader.slice(
    octreeProjectionShader.indexOf("var<workgroup> gradientPartials"),
    octreeProjectionShader.indexOf("fn projectedComponent"),
  );
  assert.match(reconstruct, /@builtin\(local_invocation_index\) lid: u32/);
  assert.match(reconstruct, /let sampleCount = 6u \* coarseOwner\.size \* coarseOwner\.size/);
  assert.match(reconstruct, /for \(var sample = lid; sample < sampleCount; sample \+= 64u\)/,
    "a 32^3 leaf must distribute its 6,144 boundary visits across the 64-lane workgroup");
  assert.match(reconstruct, /coarseGradientContribution\(coarseOwner, sample\)/);
  assert.match(reconstruct, /pressureDistance\(owner, neighbor, axis\)/,
    "coarse/fine and ghost-fluid pressure distances must survive reconstruction");
  assert.match(reconstruct, /let open = 1\.0 - clamp\(solid\.fraction/,
    "the affine fit must retain the same VOS open-face weighting as projection");
  assert.match(reconstruct, /workgroupUniformLoad\(&gradientOwnerEligible\)/,
    "the workgroup reduction branch must be uniform before reaching barriers");
  assert.match(reconstruct, /for \(var stride = 32u; stride > 0u; stride >>= 1u\)/);
  assert.match(reconstruct, /coarseOwner\.size >= 4u/);
  assert.match(reconstruct, /fn reconstructSmallGradients[\s\S]*header\.size >= 8u \|\| header\.size <= 1u[\s\S]*reconstructedAxisGradient\(owner, 0u\)/,
    "size-2 and size-4 leaves should retain the row-parallel scalar path");
});

test("octree projection is driven by the persistent leaf frontier", () => {
  const small = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectSmallLeaves"), octreeProjectionShader.indexOf("fn projectLeaves"));
  const project = octreeProjectionShader.slice(octreeProjectionShader.indexOf("fn projectLeaves"), octreeProjectionShader.indexOf("fn extrapolateSeed"));
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(256\)\s+fn projectSmallLeaves/);
  assert.match(small, /let row = compactRowIndex\(gid\)/);
  assert.match(small, /header\.size >= 8u/);
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(256\)\s+fn projectLeaves/);
  assert.match(project, /let row = coarseTaskRow\(task\)/);
  assert.match(project, /let tile = coarseTaskTile\(task\)/);
  assert.match(project, /tileCoord \* 8u/);
  assert.match(project, /sample = lid; sample < samples; sample \+= 256u/);
  assert.match(octreeProjectionShader, /projectionControllerCached\(id, loaded\) != owner\.packedOrigin/);
  assert.match(octreeProjectionShader, /fn projectionNeighborhood[\s\S]*loaded\.live\.y != loaded\.live\.x[\s\S]*xPhi = ownerPhi\(x\)/,
    "projection must reconstruct phi only at wet\/dry leaf boundaries instead of once per face use");
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(encode, /this\.projectSmallLeavesPipeline/);
  assert.match(encode, /this\.projectLeavesPipeline/);
  assert.match(encode, /this\.projectSmallLeavesPipeline\)[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,\s*0\)/);
  assert.match(encode, /this\.projectLeavesPipeline\)[\s\S]*dispatchWorkgroupsIndirect\(this\.solveDispatch,\s*12\)/);
  assert.match(octreeProjectionShader, /fn extrapolateSeed/,
    "the first dense compatibility sweep must ignore stale air validity");
});

test("octree velocity extrapolation can consume the bulk resident cell64 stream", () => {
  const encode = WebGPUOctreeProjection.prototype.encode.toString();
  assert.match(octreeProjectionShader, /@group\(0\) @binding\(15\) var<storage, read> bulkWorklist/);
  assert.match(octreeProjectionShader, /let stream = \(workgroup\.x \+ workgroup\.y \* dispatchWidth\) \* 64u \+ localIndex/,
    "the sparse kernel must linearize the producer's tiled 2D dispatch");
  assert.match(octreeProjectionShader, /let entry = 16u \+ 2u \* activeIndex/);
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(64\)\s+fn extrapolateSeedSparse/);
  assert.match(octreeProjectionShader, /@compute @workgroup_size\(64\)\s+fn extrapolateSparse/);
  assert.match(octreeProjectionShader, /fn extrapolateSeedSparse[\s\S]*extrapolateSeedAt\(bulkResidentCell\(wid, lid\)\)/);
  assert.match(octreeProjectionShader, /fn extrapolateSparse[\s\S]*extrapolateAt\(bulkResidentCell\(wid, lid\)\)/);
  assert.match(encode, /dispatchWorkgroupsIndirect\(bulkWorklist,\s*FLUID_BRICK_ACTIVE_CELL64_DISPATCH_OFFSET_BYTES\)/);
  assert.match(encode, /copyExtrapolatedSparsePipeline[\s\S]*dispatchWorkgroupsIndirect/,
    "an odd sweep count must not reintroduce a logical-volume texture copy");
});

test("octree materializes adaptive overlay fields without a readback", () => {
  assert.match(octreeDiagnosticShader, /texture_storage_3d<rg32uint, write>/);
  assert.match(octreeDiagnosticShader, /origin\.x \| \(origin\.z << 10u\) \| \(owner\.size << 20u\)/);
  assert.match(octreeDiagnosticShader, /origin\.y \| \(\(origin\.y \+ owner\.size\) << 10u\)/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureSamplesOut/);
  assert.match(octreeDiagnosticShader, /textureStore\(pressureOut/);
  assert.match(octreeDiagnosticShader, /textureStore\(divergenceOut/);
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

test("octree compact solve dispatches cover rows with two-dimensional tiles", () => {
  assert.match(octreeProjectionShader, /compaction\[2\] = x; compaction\[3\] = y/);
  assert.match(octreeProjectionShader, /fn compactRowIndex\(gid: vec3u\)/);
  assert.match(octreeProjectionShader, /gid\.x \+ gid\.y \* compaction\[2\] \* 256u/);
  assert.doesNotMatch(octreeProjectionShader, /compaction\[2\] = min\(/);
});

test("octree materializes its t=0 authority through internally fenced Section 5 submissions", () => {
  assert.deepEqual(OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES.map(({ id }) => id), [
    "cold-topology", "power-operator-authority", "surface-global-fine",
    "section5-face-band-topology", "section5-face-band-transitions",
    "section5-face-band-fast-march", "section5-face-band-power-publication",
    "sparse-render-world",
  ]);
  const phaseEncoder = WebGPUOctreeProjection.prototype.encodeInitialSparseAuthorityPhase.toString().replace(/\s+/g, "");
  assert.match(phaseEncoder,
    /case"cold-topology":this\.encodeColdBootstrapRebuild\(encoder\)[\s\S]*case"power-operator-authority":this\.encode\(encoder[\s\S]*case"surface-global-fine":this\.encodeSurface\(encoder,0\)[\s\S]*case"section5-face-band-topology":this\.encodeGlobalFineFaceBandPhase\(encoder,"topology-build"\)[\s\S]*case"section5-face-band-transitions":this\.encodeGlobalFineFaceBandPhase\(encoder,"transition-adjacency"\)[\s\S]*case"section5-face-band-fast-march":this\.encodeGlobalFineFaceBandPhase\(encoder,"fast-march"\)[\s\S]*case"section5-face-band-power-publication":this\.encodeGlobalFineFaceBandPhase\(encoder,"power-publication"\)[\s\S]*case"sparse-render-world":this\.encodeSparseBrickWorld\(encoder\)/,
    "phase encoder must retain topology -> power/operator -> fine redistance -> Section 5 rows -> Delaunay adjacency -> face march -> power publication -> render-world order");
  const combined = WebGPUOctreeProjection.prototype.encodeInitialSparseAuthority.toString().replace(/\s+/g, "");
  assert.match(combined, /for\(constphaseofOCTREE_INITIAL_SPARSE_AUTHORITY_PHASES\)/,
    "synchronous callers must retain a combined ordered encoder contract");
  const warmupTasks = uniformSolverSource.slice(
    uniformSolverSource.indexOf("private initializationTasks"),
    uniformSolverSource.indexOf("private async publishInitialSparseScenePhase"),
  );
  assert.match(warmupTasks, /OCTREE_INITIAL_SPARSE_AUTHORITY_PHASES\.forEach/);
  assert.match(warmupTasks, /const id = index === 0 \? "solver\.warmup" : `solver\.warmup\.\$\{authorityPhase\.id\}`/,
    "the first warmup task must remain the safe pre-submit resource boundary");
  assert.match(warmupTasks, /dependencies: \[previousTaskId\]/,
    "each later warmup phase must depend on its fenced predecessor");
  const phaseWarmup = uniformSolverSource.slice(
    uniformSolverSource.indexOf("private async publishInitialSparseScenePhase"),
    uniformSolverSource.indexOf("/** Publish a complete t=0 scene"),
  );
  assert.match(phaseWarmup, /Initial sparse authority: \$\{descriptor\.label\}/,
    "each bounded command buffer needs a log-friendly phase label");
  const submit = phaseWarmup.indexOf("this.device.queue.submit([initialSparseScene.finish()])");
  const fence = phaseWarmup.indexOf("await this.device.queue.onSubmittedWorkDone()", submit);
  const ready = phaseWarmup.indexOf("this.initialSparseAuthorityPublished = true", fence);
  assert.ok(submit >= 0 && fence > submit && ready > fence,
    "each task must submit then fence, and readiness must follow the final task fence");
  assert.match(phaseWarmup,
    /if \(phase === "sparse-render-world"\) \{[\s\S]*this\.initialSparseAuthorityPublished = true/,
    "only the sparse-render-world task may publish initial readiness");
  assert.match(octreeSource, /reset-time grid[\s\S]{0,120}zero-initialized topology storage as finest 1\^3/);
});
