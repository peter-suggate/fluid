import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  GPUSceneScaleFluidBrickArena,
  SCENE_FLUID_MISSING_AIR_PHI,
  SceneScaleFluidBrickLifecycle,
  addressSceneFluidBrick,
  planSceneScaleFluidBricks,
  sceneScaleFluidBrickLifecycleShader,
  splitSceneFluidCell,
} from "../lib/scene-scale-fluid-bricks";

test("scene-scale addressing preserves signed brick-relative identity at large coordinates", () => {
  assert.deepEqual(splitSceneFluidCell([-8_000_000_001, 8_000_000_007, -1]), {
    brick: [-1_000_000_001, 1_000_000_000, -1], localCell: [7, 7, 7],
  });
  assert.deepEqual(addressSceneFluidBrick([-9, 8, -1]), {
    brick: [-9, 8, -1], block: [-2, 1, -1], localBrick: [7, 0, 7], localBrickIndex: 455,
  });
  assert.deepEqual(addressSceneFluidBrick([-2_147_483_648, 0, 0]), {
    brick: [-2_147_483_648, 0, 0], block: [-268_435_456, 0, 0], localBrick: [0, 0, 0], localBrickIndex: 0,
  });
});

test("two-level storage is capacity-bounded and independent of scene span", () => {
  const tinyWorld = planSceneScaleFluidBricks(4, 16);
  const planetWorld = planSceneScaleFluidBricks(4, 16);
  assert.equal(tinyWorld.allocatedBytes, planetWorld.allocatedBytes);
  assert.equal(tinyWorld.rootCapacity, 8);
  assert.equal(tinyWorld.allocatedWords, 16 + 8 * 4 + 4 + 16 + 4 + 4 * 512);
  const doubledBlocks = planSceneScaleFluidBricks(8, 16);
  assert.equal(doubledBlocks.allocatedWords - tinyWorld.allocatedWords, (16 * 4 - 8 * 4) + 4 + 4 + 4 * 512);
  assert.equal("sceneDimensions" in tinyWorld, false);
});

test("CPU lifecycle crosses signed blocks, retires empty blocks, and returns deterministic air", () => {
  const space = new SceneScaleFluidBrickLifecycle(2, 3);
  let stats = space.publish([[-1, 0, 0], [8, 0, 0], [-9, 0, 0]], []);
  assert.deepEqual({ blocks: stats.residentBlocks, bricks: stats.residentBricks, blockOverflow: stats.blockOverflow }, {
    blocks: 2, bricks: 2, blockOverflow: 1,
  });
  assert.equal(space.slot([-1, 0, 0]), 0);
  assert.equal(space.slot([8, 0, 0]), 1);
  assert.equal(space.slot([-9, 0, 0]), undefined);
  const missingA = space.resolveCell([-72, 3, 3]);
  const missingB = space.resolveCell([-72, 3, 3]);
  assert.deepEqual(missingA, missingB);
  assert.equal(missingA.missing, true);
  assert.equal(missingA.airPhi, SCENE_FLUID_MISSING_AIR_PHI);

  stats = space.publish([[-9, 0, 0]], [[-1, 0, 0]]);
  assert.equal(stats.blockOverflow, 1, "activation cannot steal a block retired later in the publication");
  assert.equal(stats.retired, 1);
  assert.equal(stats.residentBlocks, 1);
  stats = space.publish([[-9, 0, 0]], []);
  assert.equal(stats.blockOverflow, 0);
  assert.equal(space.slot([-9, 0, 0]), 0, "the next generation reuses the retired brick slot");
  assert.equal(stats.peakBlocks, 2);
  assert.throws(() => space.publish([[1, 2, 3]], [[1, 2, 3]]), /cannot be active and retired/);
});

test("brick capacity overflow is local and does not depend on coordinate magnitude", () => {
  const space = new SceneScaleFluidBrickLifecycle(4, 2);
  const stats = space.publish([[0, 0, 0], [1_000_000_000, 0, 0], [-1_000_000_000, 0, 0]], []);
  assert.equal(stats.residentBricks, 2);
  assert.equal(stats.brickOverflow, 1);
  assert.equal(space.resolveCell([8_000_000_000, 0, 0]).missing, false);
  assert.equal(space.resolveCell([-8_000_000_000, 0, 0]).missing, true);
});

test("GPU lifecycle shader is sequential, signed, bounded, and has no scene-volume loop", () => {
  assert.match(sceneScaleFluidBrickLifecycleShader, /@workgroup_size\(1\)/);
  assert.match(sceneScaleFluidBrickLifecycleShader, /bitcast<i32>/);
  assert.match(sceneScaleFluidBrickLifecycleShader, /remainder < 0/);
  assert.doesNotMatch(sceneScaleFluidBrickLifecycleShader, /v - 7/);
  assert.match(sceneScaleFluidBrickLifecycleShader, /BLOCK_ENTRIES: u32 = 512u/);
  assert.match(sceneScaleFluidBrickLifecycleShader, /probe < params\.capacities\.z/);
  assert.doesNotMatch(sceneScaleFluidBrickLifecycleShader, /sceneDims|cellCount|dispatchWorkgroupsIndirect/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("Dawn GPU arena matches bounded activation-overflow-retirement order", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU scene-address checks" }, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(); const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const arena = new GPUSceneScaleFluidBrickArena(device, 2, 2);
  const publish = async (active: readonly (readonly [number, number, number])[], retired: readonly (readonly [number, number, number])[]) => {
    const encoder = device.createCommandEncoder(); arena.encodeLifecycle(encoder, active, retired); device.queue.submit([encoder.finish()]); return arena.readStats();
  };
  try {
    let stats = await publish([[-1, 0, 0], [8, 0, 0]], []);
    assert.deepEqual([stats.residentBlocks, stats.residentBricks, stats.blockOverflow, stats.brickOverflow], [2, 2, 0, 0]);
    stats = await publish([[-9, 0, 0]], [[-1, 0, 0]]);
    assert.equal(stats.blockOverflow, 1); assert.equal(stats.retired, 1); assert.equal(stats.residentBlocks, 1);
    stats = await publish([[-9, 0, 0]], []);
    assert.deepEqual([stats.residentBlocks, stats.residentBricks, stats.blockOverflow], [2, 2, 0]);
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
  } finally { arena.destroy(); device.destroy(); }
});
