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
test("default New scene accepts a depth-eight Box at renderer refinement and Undo clears its local occupancy", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for the default editor native regression", timeout: 180_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "voxel-editor-new-room");
  let device: GPUDevice | undefined, display: WebGPULiveSvoScene | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.pushErrorScope("validation");
    const scene = createEmptyScene();
    for (const [actual, expected] of [[scene.container.width_m, 1.6], [scene.container.height_m, 1.2], [scene.container.depth_m, 1.6]]) {
      assert.ok(Math.abs(actual! - expected!) < 1e-12);
    }
    display = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {}, undefined, {
      environmentRefinementDepth: DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth,
      environmentBrickRefinementLevels: DEFAULT_SVO_RENDER_TUNING.environmentBrickRefinementLevels,
      environmentPlanarRefinementExemption: DEFAULT_SVO_RENDER_TUNING.environmentPlanarRefinementExemption,
    });
    assert.equal(display.builtRefinementDepth, DEFAULT_SVO_RENDER_TUNING.environmentRefinementDepth);
    const source = display.sparseVoxelSceneSource;
    const structural = source.structural!;
    assert.ok(structural);
    const edited = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [16, 1, 16], maximumExclusive: [17, 9, 17], materialId: 2 }]);
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
    function localFlags(words: Uint32Array) {
      const offsets = structural.structureOffsetsWords;
      const nodes = words.slice(offsets.nodes, offsets.nodes + words[offsets.control]! * 8);
      const leaves = words.slice(offsets.leaves, offsets.leaves + words[offsets.control + 1]! * 4);
      const domain = structural.domain;
      const result = traversePackedSvo({ origin: [.025, .175, .025], direction: [0, 1, 0] }, { nodes, leaves }, {
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
    assert.equal(await device.popErrorScope(), null);
  } finally {
    display?.destroy(); if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
