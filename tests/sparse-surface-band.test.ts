import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { octreeMethod } from "../lib/methods/octree";
import {
  planSparseSurfaceBand,
  requiredSparseSurfaceBandCells,
  sparseSurfaceNeedsRefinement,
  sparseSurfaceSizingSignal,
  sparseSurfaceDynamicsShader,
  sparseSurfaceFieldShader,
  sparseSurfaceResidencyShader,
} from "../lib/webgpu-sparse-surface-band";

test("sparse surface indirect work tiles into two dimensions", () => {
  assert.match(sparseSurfaceResidencyShader, /let x = min\(blocks, 65535u\)/);
  assert.match(sparseSurfaceResidencyShader, /atomicStore\(&activePages\[2\], y\)/);
  assert.match(sparseSurfaceFieldShader, /gid\.x \+ gid\.y \* activePages\[1\] \* 256u/);
  assert.match(sparseSurfaceDynamicsShader, /gid\.x \+ gid\.y \* activePages\[1\] \* 256u/);
});

test("surface-band planning refines virtual samples without allocating the full fine volume", () => {
  const plan = planSparseSurfaceBand([61, 46, 41], { refinementFactor: 2, brickSize: 8, maximumResidentFraction: 0.35 });
  assert.deepEqual(plan.fineDimensions, [122, 92, 82]);
  assert.deepEqual(plan.brickDimensions, [16, 12, 11]);
  assert.equal(plan.logicalPageCount, 2_112);
  assert.equal(plan.physicalPageCapacity, 740);
  assert.equal(plan.voxelsPerPage, 512);
  assert.equal(plan.bytesPerPage, 512 * 8);
  assert.ok(plan.physicalPageCapacity < plan.logicalPageCount);
  assert.ok(plan.allocatedPayloadBytes < 122 * 92 * 82 * 8);
});

test("surface-band planning remains bounded for factor four and explicit page ceilings", () => {
  const plan = planSparseSurfaceBand([61, 46, 41], { refinementFactor: 4, maximumResidentFraction: 0.5, maximumPages: 512 });
  assert.deepEqual(plan.fineDimensions, [244, 184, 164]);
  assert.equal(plan.physicalPageCapacity, 512);
  assert.throws(() => planSparseSurfaceBand([0, 2, 3]), /positive integer/);
  assert.throws(() => planSparseSurfaceBand([2, 2, 2], { refinementFactor: 3 as 2 }), /1, 2, or 4/);
});

test("experimental local dynamics is opt-in and included in page accounting", () => {
  const geometric = planSparseSurfaceBand([16, 16, 16], { maximumResidentFraction: 0.5 });
  const coupled = planSparseSurfaceBand([16, 16, 16], { maximumResidentFraction: 0.5, fineDynamics: true });
  assert.equal(geometric.bytesPerPage, geometric.voxelsPerPage * 8);
  assert.equal(coupled.bytesPerPage, coupled.voxelsPerPage * 48);
});

test("surface support includes swept backtrace and every stencil radius", () => {
  assert.equal(requiredSparseSurfaceBandCells(0, 0.01, 0.01), 5);
  assert.equal(requiredSparseSurfaceBandCells(3, 0.02, 0.01), 11);
  assert.equal(requiredSparseSurfaceBandCells(1, 0.01, 0.01, 3, 4, 2), 10);
});

test("paper sizing rejects calm translation but selects curvature and non-translation motion", () => {
  const calm = sparseSurfaceSizingSignal(0, [0, 0, 0]);
  const translated = sparseSurfaceSizingSignal(0, [0, 0, 0]);
  const ripple = sparseSurfaceSizingSignal(4, [0, 0, 0]);
  const splash = sparseSurfaceSizingSignal(0, [3, -2, 1]);
  assert.equal(calm, 0);
  assert.equal(translated, 0, "uniform velocity has no derivative and must not refine a flat surface");
  assert.equal(sparseSurfaceNeedsRefinement(calm, 0.1), false);
  assert.equal(sparseSurfaceNeedsRefinement(ripple, 0.1), true);
  assert.equal(sparseSurfaceNeedsRefinement(splash, 0.1), true);
});

test("GPU residency follows the paper sizing history and keeps support separate from detail seeds", () => {
  assert.match(sparseSurfaceResidencyShader, /fn evaluateSizing/);
  assert.match(sparseSurfaceResidencyShader, /fn surfaceNormal/);
  assert.match(sparseSurfaceResidencyShader, /proposed=4\.0\*abs\(curvature\)\+3\.0\*sqrt/);
  assert.match(sparseSurfaceResidencyShader, /if\(interfaceAdjacent\(q\)\)/,
    "activity must be evaluated at the same interface sample it can refine");
  assert.match(sparseSurfaceResidencyShader, /pow\(0\.9,params\.cellAndDt\.w\/0\.01\)/);
  assert.match(sparseSurfaceResidencyShader, /fn propagateBToA/);
  assert.match(sparseSurfaceResidencyShader, /fn propagateAToB/);
  assert.match(sparseSurfaceResidencyShader, /sampleSizing\*coarseH>1\.0/);
  assert.match(sparseSurfaceResidencyShader, /var desired=core/);
  assert.match(sparseSurfaceResidencyShader, /haloBegin/);
  assert.ok(sparseSurfaceResidencyShader.indexOf("fn allocateCore3") < sparseSurfaceResidencyShader.indexOf("fn allocateHalo"),
    "detail cores have explicit priority classes ahead of stencil halos");
  assert.match(sparseSurfaceResidencyShader, /atomicSub\(&control\[0\], 1u\)/);
  assert.match(sparseSurfaceResidencyShader, /if \(oldFree == 0u\)/);
  assert.match(sparseSurfaceResidencyShader, /atomicStore\(&control\[2\], 1u\)/);
  assert.match(sparseSurfaceResidencyShader, /slot >= u32\(params\.sizing\.w\)/);
  assert.doesNotMatch(sparseSurfaceResidencyShader, /mapAsync|getMappedRange/);
});

test("fine phi sampling validates page addresses and falls back to coarse trilinear phi", () => {
  assert.match(sparseSurfaceFieldShader, /slot == INVALID \|\| slot >= u32\(params\.sizing\.w\)/);
  assert.match(sparseSurfaceFieldShader, /return densePhiAtFine\(vec3f\(q\)\)/);
  assert.match(sparseSurfaceFieldShader, /fn trilinearPhi/);
  assert.match(sparseSurfaceFieldShader, /fn advectAToB/);
  assert.match(sparseSurfaceFieldShader, /fn copyBToA/);
  assert.match(sparseSurfaceFieldShader, /let maximumDetail = 2\.0/,
    "fine transport is a bounded hierarchical correction to coarse phi");
});

test("optional fine dynamics is bounded but does not replace the global pressure solve", () => {
  const octree = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  const sparse = readFileSync(new URL("../lib/webgpu-sparse-surface-band.ts", import.meta.url), "utf8");
  const uniform = readFileSync(new URL("../lib/webgpu-uniform-eulerian.ts", import.meta.url), "utf8");
  assert.doesNotMatch(octree, /fineDynamics\s*:/,
    "the octree production path must not opt into a second page-local pressure ladder");
  assert.match(sparse, /if \(this\.options\.fineDynamics\)/,
    "experimental dynamics pipelines are not even compiled by default");
  assert.match(uniform, /requiresFineSurfaceTimestep \? this\.octreeProjection\.sparseSurfaceRefinementFactor : 1/,
    "geometry-only fine phi must not multiply the global Chebyshev substeps");
  assert.match(sparseSurfaceDynamicsShader, /fn advectResidualAToB/);
  assert.match(sparseSurfaceDynamicsShader, /params\.physical\.y/);
  assert.match(sparseSurfaceDynamicsShader, /let kappa=/);
  assert.match(sparseSurfaceDynamicsShader, /maximumDv/);
  assert.match(sparseSurfaceDynamicsShader, /fn jacobiAToB/);
  assert.match(sparseSurfaceDynamicsShader, /fn jacobiBToA/);
  assert.match(sparseSurfaceDynamicsShader, /fn projectResidual/);
  assert.match(sparseSurfaceDynamicsShader, /capillarySpeed/);
  assert.match(sparseSurfaceDynamicsShader, /smoothstep\(inner,outer,abs\(phiCell\(q\)\)\)/);
  assert.match(sparseSurfaceDynamicsShader, /return vec3f\(0\.0\)/, "missing pages impose the coarse-velocity boundary condition");
});

test("superseded adaptive surface patches stay disabled and out of the product UI", () => {
  for (const quality of ["balanced", "high", "ultra"] as const) {
    const preset = octreeMethod.presetFor(quality);
    assert.equal(preset.sparseSurfaceBand, "off");
    assert.equal(preset.surfaceRefinementFactor, "2");
    assert.equal(preset.sparseSurfaceBandCells, 4);
    assert.equal(preset.sparseSurfacePageFraction, 0.75);
  }
  assert.equal(octreeMethod.params.some((candidate) => candidate.key === "sparseSurfaceBand"), false);
  assert.equal(octreeMethod.params.some((candidate) => candidate.key === "surfaceRefinementFactor"), false);
});

test("raster extraction and solver-grid inspection consume the same live sparse surface", () => {
  const water = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  const grid = readFileSync(new URL("../lib/webgpu-grid-overlay.ts", import.meta.url), "utf8");
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  assert.match(water, /fn extractSparseMain/);
  assert.match(water, /dispatchWorkgroupsIndirect\(this\.sparseSurface\.activePages\.buffer/);
  assert.match(water, /sparsePayloadIndex/);
  assert.match(water, /let xBases = array<i32, 2>\(i32\(q\.x \+ 1u\), 0\)/);
  assert.match(water, /let yBases = array<i32, 2>\(i32\(q\.y \+ 1u\), 0\)/);
  assert.match(water, /let zBases = array<i32, 2>\(i32\(q\.z \+ 1u\), 0\)/);
  assert.match(water, /classifyCube\(vec3i\(xBases\[xIndex\], yBases\[yIndex\], zBases\[zIndex\]\)\)/,
    "sparse extraction must include face, edge, floor-strip, and triple-corner cubes without duplicates");
  assert.match(grid, /fn sparseSurfaceCoreSample/);
  assert.match(grid, /abs\(sparseSurfacePhi\[payload\]\) <= 1\.5\*fineH/,
    "pink must be confined to the phi=0 shell rather than filling support pages");
  assert.match(grid, /(?:sparseSurfaceStates\[pageIndex\]&2u|sparseSurfacePageState\(fineCell\) & 2u)/,
    "pink must identify a detail core rather than its allocation-only halo");
  assert.match(grid, /sparseSurfaceControl\[2\] == 0u/,
    "a partial fine hierarchy must not be presented after allocator overflow");
  assert.match(grid, /fineColor=vec3f\(1\.0,0\.08,0\.55\)/);
  assert.match(water, /fn extractHybridCoarseMain/);
  assert.match(water, /fn sparseCorePageAt/);
  assert.match(water, /sparseStates\[pageIndex\] & SPARSE_CORE/,
    "hybrid ownership must distinguish detail cores from their transition halos");
  assert.match(water, /if \(sparseCorePageAt\(fineCenter\)\) \{ return; \}/,
    "coarse cubes are replaced only inside detail cores, leaving a watertight overlap through support halos");
  assert.match(water, /this\.resetSurfaceWorklistPipeline/);
  assert.ok(water.indexOf("compute.setPipeline(this.polygonisePipeline)") < water.indexOf("compute.setPipeline(this.polygoniseSparsePipeline)"),
    "hybrid extraction emits the complete coarse level before fine patches");
  assert.match(renderer, /setSparseSurface\(this\.gpuFluid\?\.sparseSurfaceBand\)/);
});
