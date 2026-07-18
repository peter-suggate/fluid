import assert from "node:assert/strict";
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

test("disconnected wet regions independently create multiple resident fluid bricks", () => {
  const options = { haloPhi: 0.2, retireAfterFrames: 2 };
  const states = [-0.1, 1, -0.4].map((phi) => classifyCPUFluidBrick(phi, undefined, options));
  assert.equal(states.filter((state) => (state.flags & FLUID_BRICK_CORE) !== 0).length, 2);
  assert.ok((states[0].flags & (FLUID_BRICK_RESIDENT | FLUID_BRICK_ACTIVATED)) !== 0);
  assert.equal(states[1].flags, 0);
  assert.ok((states[2].flags & FLUID_BRICK_CORE) !== 0);
});

test("air-band bricks form halos and dry source bricks retire with hysteresis", () => {
  const options = { haloPhi: 0.2, retireAfterFrames: 2 };
  const halo = classifyCPUFluidBrick(0.1, undefined, options);
  assert.ok((halo.flags & FLUID_BRICK_HALO) !== 0);
  let source = classifyCPUFluidBrick(-0.1, undefined, options);
  source = classifyCPUFluidBrick(1, source, options);
  assert.ok((source.flags & FLUID_BRICK_RESIDENT) !== 0);
  source = classifyCPUFluidBrick(1, source, options);
  assert.ok((source.flags & FLUID_BRICK_RESIDENT) !== 0);
  source = classifyCPUFluidBrick(1, source, options);
  assert.equal(source.flags & FLUID_BRICK_RESIDENT, 0, "the vacated source brick must return to the free pool");
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
  assert.match(fluidBrickResidencyShader, /resident \* voxelsPerBrick/);
  assert.doesNotMatch(fluidBrickResidencyShader, /mapAsync|getMappedRange/);
  assert.match(sparseBrickDenseFieldShader, /usesActiveWorklist\(\)/);
  assert.match(sparseBrickDenseFieldShader, /localIndex = index - streamBrick \* voxelsPerBrick/);
  assert.match(sparseBrickDenseFieldShader, /clearRetiredDenseFields/);
});

test("pressure topology rebuild consumes the shared resident-brick worklist indirectly", () => {
  const rebuild = WebGPUOctreeProjection.prototype.encodeInlineRebuild.toString();
  assert.match(rebuild, /residency\.worklist/);
  assert.match(rebuild, /dispatchWorkgroupsIndirect/);
  assert.match(octreeProjectionShader, /fn residentTopologyCell/);
  assert.match(octreeProjectionShader, /fn rasterizeSolidsActive/);
  assert.match(octreeProjectionShader, /fn resetTopologyActive/);
  assert.match(octreeProjectionShader, /fn refineTopologyActive/);
  assert.match(octreeProjectionShader, /fn balanceTopologyActive/);
});

test("sparse fluid tail exposes separate residency and publication timestamp ranges", () => {
  const encode = OctreeSparseBrickWorld.prototype.encode.toString();
  assert.match(encode, /Fluid brick residency/);
  assert.match(encode, /Sparse brick publication/);
  const publish = WebGPUOctreeProjection.prototype.encodeSparseBrickWorld.toString();
  assert.match(publish, /timings/);
});
