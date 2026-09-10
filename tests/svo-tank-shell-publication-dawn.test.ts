import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { getScenePreset } from "../lib/core/scenes";
import { sceneWithSolidStroke, solidWorldForScene, sampleSolidWorld } from "../lib/core/solid-world";
import { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import { sparseBrickScenePayloadIdentityAt } from "../lib/svo/features/construction/sparse-brick-octree";

test("production SVO excludes tank glass at reset and after live edit/undo", {
  skip: !process.env.WEBGPU_NODE_MODULE, timeout: 120_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "svo-tank-shell-publication");
  let device: GPUDevice | undefined, display: WebGPULiveSvoScene | undefined;
  try {
    const initialized = await createDawnRenderDevice(); device = initialized.device;
    const scene = getScenePreset("dam-break").create();
    scene.rigidBodies = [];
    display = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {});
    const source = display.sparseVoxelSceneSource;
    const structural = source.structural!;
    const read = async (binding: GPUBufferBinding) => {
      const size = Number(binding.size ?? binding.buffer.size);
      const output = device!.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      try {
        const encoder = device!.createCommandEncoder();
        encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, output, 0, size);
        device!.queue.submit([encoder.finish()]); await output.mapAsync(GPUMapMode.READ);
        const words = new Uint32Array(output.getMappedRange().slice(0)); output.unmap(); return words;
      } finally { output.destroy(); }
    };
    const census = async () => {
      const structure = await read(structural.structure), payload = await read(structural.scenePayload);
      const counts = new Map<number, number>();
      const { control, leaves } = structural.structureOffsetsWords;
      for (let leaf = 0; leaf < structure[control + 1]!; leaf++) {
        const offset = leaves + leaf * 4;
        if (structure[offset + 2] !== 0) continue;
        const base = structure[offset + 1]!;
        for (let i = 0; i < structural.domain.brickSize ** 3; i++) {
          const material = sparseBrickScenePayloadIdentityAt(payload, structural.scenePayloadLanes, base + i) & 0xffff;
          if (material) counts.set(material, (counts.get(material) ?? 0) + 1);
        }
      }
      return counts;
    };
    const publish = async (next: typeof scene) => {
      display!.validateLiveSolidEdit(next); display!.stageSceneUpdate(next);
      for (let i = 0; i < 8; i++) {
        const encoder = device!.createCommandEncoder(); display!.encodeSceneMaintenance(encoder);
        device!.queue.submit([encoder.finish()]);
      }
      await device!.queue.onSubmittedWorkDone(); assert.equal(display!.sparseVoxelSceneSource, source);
    };
    assert.equal((await census()).get(1) ?? 0, 0, "startup must not voxelize the physics-only glass shell");
    assert.equal(sampleSolidWorld(solidWorldForScene(scene), [4, -1, 4]).materialId, 1,
      "the solver still owns a solid tank floor");
    const plate = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [4, 3, 4],
      maximumExclusive: [12, 4, 12], materialId: 2 }]);
    await publish(plate);
    assert.ok(((await census()).get(2) ?? 0) > 0, "editable thin solids remain voxelized");
    assert.equal((await census()).get(1) ?? 0, 0, "live publication must not reintroduce the shell");
    const cut = sceneWithSolidStroke(plate, [{ operation: "clear", minimum: [4, -1, 4], maximumExclusive: [5, 0, 5] }]);
    await publish(cut);
    assert.ok(((await census()).get(1) ?? 0) > 0, "a cut exposes the actual remaining wall geometry");
    await publish(scene);
    const undone = await census();
    assert.equal(undone.get(1) ?? 0, 0, "Undo clears the entire re-hidden slab, including cells outside the cut");
    assert.equal(undone.get(2) ?? 0, 0, "Undo removes the editable plate");
    assert.deepEqual(initialized.validationErrors, []);
  } finally {
    display?.destroy(); if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
    await releaseWebGPUExclusiveLock();
  }
});
