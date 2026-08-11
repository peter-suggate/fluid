import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  planFluidFootprintFineBandBrickFloor,
  planFluidFootprintFineNarrowBandBrickCapacity,
  POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY,
} from "../lib/webgpu-octree";
import { planFineLevelSetBandFineCells } from "../lib/webgpu-octree-fine-levelset-topology";

test("fine-band capacity includes compact-footprint edge and corner dilation", () => {
  assert.equal(planFluidFootprintFineBandBrickFloor(
    [24, 16, 24], [8, 8, 8], [16, 16, 16], 8,
  ), 9_216, "the ceiling drop needs one deformation ring beyond rigid translation");
  assert.equal(planFluidFootprintFineBandBrickFloor(
    [24, 24, 24], [8, 8, 8], [16, 16, 16], 8,
  ), 13_824, "the mid-air control needs the complete centred recurring envelope");
  assert.equal(planFluidFootprintFineBandBrickFloor(
    [24, 16, 24], [0, 8, 0], [8, 16, 8], 8,
  ), 9_216, "a wall-flush seed must retain headroom to separate from every wall");
});

test("fine-band capacity subtracts the footprint interior beyond the band", () => {
  assert.equal(planFluidFootprintFineBandBrickFloor(
    [64, 64, 64], [12, 12, 12], [52, 52, 52], 4,
  ), 48 ** 3 - 32 ** 3);
  assert.throws(() => planFluidFootprintFineBandBrickFloor(
    [24, 16, 24], [8, 8, 8], [25, 16, 16], 6,
  ), /bounds are invalid/);
});

test("Power factor-4 reserve admits the complete compact symmetric-expansion band", () => {
  const compact = planFluidFootprintFineNarrowBandBrickCapacity(
    [32, 16, 32], [16, 8, 16], 8, 0,
    POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY,
  );
  assert.equal(compact.maximumResidentBricks, 16_384);

  const large = planFluidFootprintFineNarrowBandBrickCapacity(
    [256, 128, 256], [128, 64, 128], 8, 0,
    POWER2017_FINE_BAND_SURFACE_GROWTH_SAFETY,
  );
  assert.ok(large.maximumResidentBricks < large.logicalBrickCount,
    "the factor-4 reference must retain sparse area scaling on production-sized domains");
});

test("the hybrid LoSasso lane uses the shared donor contract for its single trace", () => {
  assert.deepEqual(planFineLevelSetBandFineCells(1, 4), {
    transportBandFineCells: 8,
    redistanceBandFineCells: 17,
    maximumBacktraceFineCells: 8,
  });
  const transport = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-fine-transport.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(transport, /for\(var stage=0u;stage<stages;stage\+=1u\)/);
  assert.match(transport, /let subDt=p\.dt\/f32\(stages\)/);
  const host = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-fine-transport.ts", import.meta.url,
  ), "utf8");
  assert.match(host, /words\[30\] = 1;/,
    "the donor bound must not silently turn into a fixed transport-stage count");
  assert.match(host, /dispatchWorkgroupsIndirect\([\s\S]*this\.liveDispatch, 16\)/,
    "advect and commit must consume the GPU-compacted awake page count");
  assert.doesNotMatch(host, /dispatchWorkgroups\(this\.plan\.pageCapacity\)/);
  assert.match(transport,
    /atomicStore\(&liveDispatch\[4\],select\(0u,activeCount,acceptedStep\(\)\)\)/);
});

test("warm redistance counts validated carried closest points as seed authority", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-fine-levelset-redistance.ts", import.meta.url,
  ), "utf8");
  assert.match(source,
    /seedCount=select\(0u,1u,carried!=INVALID\);if\(hasCachedClosestPoint\(index\)\)/,
  "a warm transform with only remapped closest points must not publish a zero-seed receipt");
});

test("Losasso wall transport adds the closed-boundary characteristic exit distance", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-fine-transport.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(source, /exitCells\+=distance\(traced,interior\)/);
  assert.match(source,
    /nextPhi\[index\]=applyInflowPhi\(transported\+exitCells\*p\.fineCellWidth,world\)/);
});

test("Losasso keeps closed wall faces for fine-phi separation conditioning", () => {
  const topology = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-backend.wgsl.ts", import.meta.url,
  ), "utf8");
  const coarsePhi = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(topology, /FACE_CLOSED_BOUNDARY/);
  assert.match(coarsePhi, /face\.openFraction=0\.;flags=2u/);
  assert.match(coarsePhi, /face\.inverseDistance=1\.\/distance;face\.openFraction=1\.;flags=3u/);
});

test("Losasso velocity sampling extends only wall-separating normal velocity", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-velocity-sampler.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(source, /let away=select\(\(interior<0\.\),\(interior>0\.\),lowWall\)/);
  assert.match(source, /if\(away\)\{value=interior;\}/);
});

test("Losasso couples the 2017 fine band to a unilateral coarse wall active set", () => {
  const projection = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-projection.ts", import.meta.url,
  ), "utf8");
  const coarsePhi = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts", import.meta.url,
  ), "utf8");
  const migration = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-velocity-migration.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.match(projection, /contactPressure = 0\.25 \* this\.physical\.density \* weight/);
  assert.match(projection, /let releasePressure = select\(0\.0, params\.contactPressure, overhead\)/,
    "all closed walls release under tension while only the ceiling receives a hydrostatic bias");
  assert.match(projection, /let wasSeparated = \(face\.reserved & FACE_SEPARATED\) != 0u/);
  assert.match(projection, /solved < renewalPressure, wasSeparated/,
    "a released wall face must not re-weld on a p approximately zero duty cycle");
  assert.match(projection, /face\.reserved \| FACE_SEPARATED/);
  assert.match(coarsePhi,
    /let separated=\(face\.reserved&FACE_SEPARATED\)!=0u/);
  assert.match(coarsePhi, /if\(separated\|\|\(\(rowFlags&\(VALID\|FINITE\)\)/,
    "either the active set or a resolved fine-phi air gap must open a p=0 ghost");
  assert.match(migration, /oldGeometry\[old\]\.x&FACE_SEPARATED/);
  assert.match(migration, /newFaces\[face\]\.reserved\|=FACE_SEPARATED/,
    "refinement and coarsening must not re-weld a released wall face");
});

test("Losasso closed-wall ghost distances retain the Power/Gibou one-percent theta floor", () => {
  const source = readFileSync(new URL(
    "../lib/webgpu-octree-losasso-coarse-phi.wgsl.ts", import.meta.url,
  ), "utf8");
  assert.equal((source.match(/clamp\(-liquid\/\(airPhi-liquid\),1e-2,1\.\)/g) ?? []).length, 1);
  assert.equal((source.match(/clamp\(-liquid\/\(airPhi-liquid\),1e-4,1\.\)/g) ?? []).length, 1,
    "the ordinary free-surface path keeps its idempotence-protected floor");
});
