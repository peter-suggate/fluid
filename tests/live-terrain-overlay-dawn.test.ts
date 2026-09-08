import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo/pipeline/svo-render-tuning";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { SVO_BRICK_OCCUPANCY, SVO_BRICK_LIFECYCLE } from "../lib/svo/features/construction/svo-brick-occupancy";
import { traversePackedSvo } from "../lib/svo/features/primary-visibility/webgpu-svo-traversal";

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("terrain overlay fill, clear and Undo preserve the live SVO and local occupancy", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for the default editor native regression", timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "live-terrain-overlay");
  let device: GPUDevice | undefined, display: WebGPULiveSvoScene | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.pushErrorScope("validation");
    const scene = createEmptyScene({ extents_m: { x: .8, y: .8, z: .8 }, finestCellSize_m: .05 });
    scene.terrain = { baseHeight_m: .3, features: [] };
    display = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {}, undefined, {
      environmentRefinementDepth: DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth,
      environmentBrickRefinementLevels: DEFAULT_SVO_RENDER_TUNING.environmentBrickRefinementLevels,
      environmentPlanarRefinementExemption: DEFAULT_SVO_RENDER_TUNING.environmentPlanarRefinementExemption,
    });
    assert.equal(display.builtRefinementDepth, DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth);
    const source = display.sparseVoxelSceneSource;
    const structural = source.structural!;
    assert.ok(structural);
    const edited = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [8, 7, 8], maximumExclusive: [10, 10, 10], materialId: 2 }]);
    async function publish(next: typeof scene) {
      display!.validateLiveSolidEdit(next);
      display!.stageSceneUpdate(next);
      for (let frame = 0; frame < 8; frame++) {
        const encoder = device!.createCommandEncoder();
        display!.encodeSceneMaintenance(encoder);
        device!.queue.submit([encoder.finish()]);
      }
      await device!.queue.onSubmittedWorkDone();
      assert.equal(display!.sparseVoxelSceneSource, source);
    }
    async function readStructure() {
      const buffer = device!.createBuffer({ size: structural.structure.buffer.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device!.createCommandEncoder();
        encoder.copyBufferToBuffer(structural.structure.buffer, 0, buffer, 0, buffer.size);
        device!.queue.submit([encoder.finish()]); await buffer.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(buffer.getMappedRange().slice(0)); buffer.unmap(); return words;
      } finally { buffer.destroy(); }
    }
    function localFlags(words: Uint32Array, y = .425) {
      const offsets = structural.structureOffsetsWords;
      const nodes = words.slice(offsets.nodes, offsets.nodes + words[offsets.control]! * 8);
      const leaves = words.slice(offsets.leaves, offsets.leaves + words[offsets.control + 1]! * 4);
      const domain = structural.domain;
      const result = traversePackedSvo({ origin: [.025, y, .025], direction: [0, 1, 0] }, { nodes, leaves }, {
        origin: domain.worldOrigin_m, cellSize: domain.cellSize_m,
        brickSize: domain.brickSize, maximumDepth: domain.maximumDepth,
      });
      assert.equal(result.status, "hit");
      if (result.status !== "hit") throw new Error("edited point has no presentation leaf");
      assert.equal(result.hit.terminalKind, 0);
      return nodes[result.hit.nodeIndex * 8 + 7]!;
    }
    await publish(edited);
    const filled = await readStructure();
    assert.equal(filled[structural.structureOffsetsWords.control + 12], 0);
    const flags = localFlags(filled);
    assert.ok(flags & SVO_BRICK_OCCUPANCY.occupiedBit, "the authored Box must have visible local voxel occupancy");
    assert.ok(flags & SVO_BRICK_OCCUPANCY.readyBit);
    assert.equal(flags & (SVO_BRICK_LIFECYCLE.dirtyBit | SVO_BRICK_LIFECYCLE.queuedBit), 0);
    await publish(scene);
    const undone = await readStructure();
    assert.equal(undone[structural.structureOffsetsWords.control + 23], filled[structural.structureOffsetsWords.control + 23], "Undo allocates no added topology");
    assert.equal(localFlags(undone) & SVO_BRICK_OCCUPANCY.occupiedBit, 0);
    // Carve below the immutable heightfield, then Undo the ordered clear.
    const carved = sceneWithSolidStroke(scene, [{ operation: "clear", minimum: [8, 2, 8], maximumExclusive: [9, 4, 9] }]);
    await publish(carved);
    const cut = await readStructure();
    assert.equal(localFlags(cut, .175) & SVO_BRICK_OCCUPANCY.occupiedBit, 0,
      "a one-voxel-wide clear must remove terrain below its original height without a rebuild");
    await publish(scene);
    const restored = await readStructure();
    assert.ok(localFlags(restored, .175) & SVO_BRICK_OCCUPANCY.occupiedBit,
      "Undo must restore the immutable terrain under the removed overlay");
    assert.equal(await device.popErrorScope(), null);
  } finally {
    display?.destroy(); if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
