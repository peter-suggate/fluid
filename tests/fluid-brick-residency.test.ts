import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { cloneScene } from "../lib/model";
import { getScenePreset } from "../lib/scenes";
import { initialFluidBrickContainsCell, initialFluidBrickSignedDistance } from "../lib/initial-fluid";
import {
  FLUID_BRICK_ACTIVATED,
  FLUID_BRICK_CORE,
  FLUID_BRICK_HALO,
  FLUID_BRICK_RESIDENT,
  classifyCPUFluidBrick,
  fluidBrickResidencyShader,
} from "../lib/webgpu-fluid-brick-residency";
import { sparseBrickDenseFieldShader } from "../lib/sparse-brick-octree";
import { WebGPUOctreeProjection, octreeProjectionShader } from "../lib/webgpu-octree";
import { OctreeSparseBrickWorld } from "../lib/webgpu-octree-sparse-bricks";

test("disconnected interface regions independently create multiple resident fluid bricks", () => {
  const options = { haloPhi: 0.2, retireAfterFrames: 2 };
  const ranges = [[-0.1, 0.3], [0.4, 1], [-0.4, 0.1]] as const;
  const states = ranges.map(([minimum, maximum]) => classifyCPUFluidBrick(minimum, undefined, options, maximum));
  assert.equal(states.filter((state) => (state.flags & FLUID_BRICK_CORE) !== 0).length, 2);
  assert.ok((states[0].flags & (FLUID_BRICK_RESIDENT | FLUID_BRICK_ACTIVATED)) !== 0);
  assert.equal(states[1].flags, 0);
  assert.ok((states[2].flags & FLUID_BRICK_CORE) !== 0);
});

test("two-sided surface-band bricks form halos and vacated bricks retire with hysteresis", () => {
  const options = { haloPhi: 0.2, retireAfterFrames: 2 };
  const halo = classifyCPUFluidBrick(0.1, undefined, options, 0.5);
  assert.ok((halo.flags & FLUID_BRICK_HALO) !== 0);
  const liquidHalo = classifyCPUFluidBrick(-0.5, undefined, options, -0.1);
  assert.ok((liquidHalo.flags & FLUID_BRICK_HALO) !== 0, "the band must retain liquid-side stencil support");
  const deepLiquid = classifyCPUFluidBrick(-2, undefined, options, -0.5);
  assert.equal(deepLiquid.flags, 0, "deep liquid must not turn a surface band into a full liquid volume");
  let source = classifyCPUFluidBrick(-0.1, undefined, options, 0.1);
  source = classifyCPUFluidBrick(1, source, options, 2);
  assert.ok((source.flags & FLUID_BRICK_RESIDENT) !== 0);
  source = classifyCPUFluidBrick(1, source, options, 2);
  assert.ok((source.flags & FLUID_BRICK_RESIDENT) !== 0);
  source = classifyCPUFluidBrick(1, source, options, 2);
  assert.equal(source.flags & FLUID_BRICK_RESIDENT, 0, "the vacated source brick must return to the free pool");
});

test("bulk residency retains deep liquid without widening surface-only residency", () => {
  const deepLiquid = [-4, -2] as const;
  const surface = classifyCPUFluidBrick(deepLiquid[0], undefined, { haloPhi: 1, retireAfterFrames: 0 }, deepLiquid[1]);
  assert.equal(surface.flags & FLUID_BRICK_RESIDENT, 0, "surface scheduling excludes deep liquid");
  const bulk = classifyCPUFluidBrick(
    deepLiquid[0], undefined,
    { haloPhi: 1, retireAfterFrames: 0, includeLiquidInterior: true },
    deepLiquid[1],
  );
  assert.ok((bulk.flags & FLUID_BRICK_RESIDENT) !== 0, "bulk storage retains every wet brick");
  assert.equal(bulk.flags & FLUID_BRICK_CORE, 0, "deep liquid is not mislabeled as an interface brick");
  const source = readFileSync(new URL("../lib/webgpu-fluid-brick-residency.ts", import.meta.url), "utf8");
  assert.match(source, /writeBuffer\(this\.params, 40, new Float32Array\(\[Math\.max\(0, options\.dt_s \?\? 0\)\]\)\)/,
    "per-step swept dt updates must preserve the constructor-owned includeLiquidInterior word");
  assert.doesNotMatch(source, /Math\.max\(0, options\.dt_s \?\? 0\), preActivation/);
});

test("garden dam-break seeds exactly one resolution-independent 8-cubed brick", () => {
  const scene = getScenePreset("garden-dam-break").create();
  assert.equal(scene.fluid.initialBrickSeeds_m?.length, 1);
  const dimensions = [61, 46, 41] as const;
  const wetBricks = new Set<string>();
  let wetCells = 0;
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    if (!initialFluidBrickContainsCell(scene, x, y, z, dimensions)) continue;
    wetCells += 1;
    wetBricks.add(`${Math.floor(x / 8)},${Math.floor(y / 8)},${Math.floor(z / 8)}`);
  }
  assert.equal(wetBricks.size, 1);
  assert.equal(wetCells, 8 ** 3);
  const seed = scene.fluid.initialBrickSeeds_m![0];
  assert.ok((initialFluidBrickSignedDistance(scene, seed, dimensions) ?? 1) < 0);
});

test("multiple seed points create independent initial fluid bricks", () => {
  const scene = cloneScene(getScenePreset("garden-dam-break").create());
  scene.fluid.initialBrickSeeds_m = [
    { x: -0.9, y: 0.61, z: -0.3 },
    { x: 0.9, y: 0.61, z: 0.7 },
  ];
  const dimensions = [61, 46, 41] as const;
  const wetBricks = new Set<string>();
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    if (initialFluidBrickContainsCell(scene, x, y, z, dimensions)) wetBricks.add(`${Math.floor(x / 8)},${Math.floor(y / 8)},${Math.floor(z / 8)}`);
  }
  assert.equal(wetBricks.size, 2);
});

test("GPU residency builds active and retired indirect worklists without readback", () => {
  assert.match(fluidBrickResidencyShader, /atomicAdd\(&worklist\[0\], 1u\)/);
  assert.match(fluidBrickResidencyShader, /atomicAdd\(&worklist\[4\], 1u\)/);
  assert.match(fluidBrickResidencyShader, /minimumPhi <= 0\.0 && maximumPhi >= 0\.0/);
  assert.match(fluidBrickResidencyShader, /min\(abs\(minimumPhi\), abs\(maximumPhi\)\)/);
  assert.match(fluidBrickResidencyShader, /resident \* voxelsPerBrick/);
  assert.match(fluidBrickResidencyShader, /let activeDispatch = tiledDispatch/);
  assert.match(fluidBrickResidencyShader, /atomicStore\(&worklist\[2\], activeDispatch\.y\)/);
  assert.match(fluidBrickResidencyShader, /surfaceDispatch = tiledDispatch\(\(resident \* voxelsPerBrick \+ 63u\) \/ 64u\)/);
  assert.match(fluidBrickResidencyShader, /atomicStore\(&worklist\[13\], surfaceDispatch\.y\)/);
  assert.doesNotMatch(fluidBrickResidencyShader, /resident \* topologyGroups, 65535u/);
  assert.doesNotMatch(fluidBrickResidencyShader, /mapAsync|getMappedRange/);
  assert.match(sparseBrickDenseFieldShader, /usesActiveWorklist\(\)/);
  assert.match(sparseBrickDenseFieldShader, /localIndex = index - streamBrick \* voxelsPerBrick/);
  assert.match(sparseBrickDenseFieldShader, /clearRetiredDenseFields/);
});

test("GPU residency derives a topology-tile worklist so leaves never straddle a rebuild boundary", () => {
  assert.match(fluidBrickResidencyShader, /fn emitTopologyTiles/);
  assert.match(fluidBrickResidencyShader, /fn tileHasResident/);
  assert.match(fluidBrickResidencyShader, /vec3i\(tile\) \+ vec3i\(dx, dy, dz\)/, "grading support dilates by one maximum-leaf tile");
  assert.match(fluidBrickResidencyShader, /atomicExchange\(&tileStates\[tileIndex\]/, "dilation-only tiles retain retirement state");
  assert.match(fluidBrickResidencyShader, /atomicAdd\(&tileWorklist\[0\], 1u\)/);
  assert.match(fluidBrickResidencyShader, /atomicAdd\(&tileWorklist\[4\], 1u\)/);
  assert.match(fluidBrickResidencyShader, /let tileDispatch = tiledDispatch\(activeTiles \* groupsPerTile\)/);
  assert.match(fluidBrickResidencyShader, /let retiredTileDispatch = tiledDispatch\(retiredTiles \* groupsPerTile\)/);
  assert.match(fluidBrickResidencyShader, /candidateGroupsPerTile = max\(1u, groupsPerTile \/ 8u\)/);
});

test("pressure topology rebuild consumes the shared topology-tile worklist indirectly", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  assert.match(rebuild, /if\(active\|\|!this\.directPagedTopology\)this\.ownerPages\?\.encode\(encoder\)/,
    "an unpublished direct-paged worklist must not retire the coarse owner map before full-domain rebuild");
  assert.match(rebuild, /topologyResidency\.tileWorklist/);
  assert.match(rebuild, /dispatchWorkgroupsIndirect/);
  // The former leaf-size gate is gone: after initialization the tile path is
  // the only rebuild domain at every legal (brickSize, maximumLeafSize) pair.
  assert.doesNotMatch(rebuild, /maxLeafSize <= /);
  assert.match(octreeProjectionShader, /fn topologyTileSize\(\) -> u32 \{ return max\(8u, params\.dimsMax\.w\); \}/);
  assert.match(octreeProjectionShader, /fn residentTopologyCell/);
  assert.match(octreeProjectionShader, /fn residentTopologyCandidate/);
  assert.match(octreeProjectionShader, /subCoord \* 8u \+ local \* 2u/);
  assert.match(octreeProjectionShader, /workgroup\.x \+ workgroup\.y \* compaction\[widthWord\]/);
  assert.match(octreeProjectionShader, /fn rasterizeSolidsActive/);
  assert.match(octreeProjectionShader, /fn resetTopologyActive/);
  assert.match(octreeProjectionShader, /fn refineTopologyActive/);
  assert.match(octreeProjectionShader, /fn balanceTopologyActive/);
});

test("pressure topology residency covers refinement and 2:1 grading support", () => {
  const source = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(source, /const topologyHaloCells = this\.interfaceRefinementBandCells/);
  assert.match(source, /\+ 8 \* this\.surfaceDetailStrength/);
  assert.doesNotMatch(source, /\+ \(this\.maxLeafSize - 1\)/, "grading support no longer inflates brick residency");
  assert.match(fluidBrickResidencyShader, /2:1 grading chain travels less than one maximum-leaf tile/);
  assert.match(source, /haloCells: topologyHaloCells/);
});

test("retired topology tiles rebuild before leaving the active domain", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  assert.match(rebuild, /FLUID_TILE_RETIRED_DISPATCH_OFFSET_BYTES/);
  assert.match(rebuild, /dispatchRetired\(this\.resetRetiredPipeline\)/);
  assert.match(rebuild, /dispatchRetiredCandidates\(level\.retired/);
  assert.match(rebuild, /this\.refineCoarsePipelines\.get\(size\)/,
    "coarse cooperative refinement covers the full domain, including retired tiles");
  assert.match(rebuild, /dispatchRetiredCandidates\(this\.balanceRetiredPipeline\)/);
  assert.match(octreeProjectionShader, /fn retiredTopologyCell/);
  // Retired tile indices follow the active tile capacity in the copied list.
  assert.match(octreeProjectionShader, /topologyTileCell\(workgroup, local, 4u, 5u, 16u \+ tx \* ty \* tz\)/);
  assert.match(octreeProjectionShader, /fn resetTopologyRetired/);
  assert.match(octreeProjectionShader, /fn refineTopologyRetired/);
  assert.match(octreeProjectionShader, /fn balanceTopologyRetired/);
});

test("sparse fluid tail exposes separate residency and publication timestamp ranges", () => {
  const encode = OctreeSparseBrickWorld.prototype.encode.toString();
  assert.match(encode, /Fluid brick residency/);
  assert.match(encode, /Sparse brick publication/);
  const publish = WebGPUOctreeProjection.prototype.encodeSparseBrickWorld.toString();
  assert.match(publish, /timings/);
  assert.match(publish, /this\.topologyWorklistReady\s*=\s*true/,
    "direct-paged analytic topology remains ready because rejected GPU generations retain its last-good worklist");
});
