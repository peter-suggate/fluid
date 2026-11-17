import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { OGC_STATUS } from '../lib/cloth/contact/ogc-wgsl';
import { ClothSystem, CLOTH_SCENE_STEP_S } from '../lib/cloth/system';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { publishSceneClothSurfaces, snapshotSceneClothPose } from '../lib/cloth/scene-geometry';
import { cloneScene } from '../lib/core/model';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('subgroups') ? ['subgroups'] : [], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = []; device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); errors.push((e as GPUUncapturedErrorEvent).error.message); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); } finally { device.destroy(); }
}

const MEMORY = 3, TICKS = 90;
interface Tick { positions: number[]; entered: boolean; held: boolean }

/**
 * Fail-safe memory: after a mandatory round's least shift ends in the fail-safe, the next MEMORY steps start there. The
 * run is the reference's up to the first held start, held starts follow exactly the schedule, and no step intersects.
 */
test('fail-safe memory starts the scheduled steps held, matches the reference until then, and keeps every step free of intersections', { skip: !gpu }, async () => withDevice(async device => {
  const run = async (contactFailSafeMemory: number) => {
    const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 32 }));
    const system = await ClothSystem.create(device, scene, { liveOptimizations: true, contactFailSafeMemory, statusInterval: 1 });
    const ticks: Tick[] = [];
    try {
      for (let tick = 1; tick <= TICKS; tick++) {
        const pose = (await system.capture([tick * CLOTH_SCENE_STEP_S]))[0]![0]!;
        assert.equal(system.failure, undefined, JSON.stringify(system.failure));
        const c = await system.contactFor(scene.cloth![0]!.id)!.readCounters();
        assert.equal(c.intersections, 0, `tick ${tick}: cloth intersections`);
        assert.equal(c.solidIntersections, 0, `tick ${tick}: solid intersections`);
        assert.equal(c.status & OGC_STATUS.heldConflict, 0, `tick ${tick}: held conflict`);
        assert.ok(c.heldStarts <= 1 && c.failSafeEntries <= 1, `tick ${tick}: one step a tick`);
        const held = c.heldStarts > 0;
        if (held) assert.equal(c.failSafeEntries, 0, `tick ${tick}: a held start has no least shift to end`);
        ticks.push({ positions: Array.from(pose.positions_m), entered: c.failSafeEntries > 0, held });
      }
      return ticks;
    } finally { system.destroy(); }
  };
  const reference = await run(0), memory = await run(MEMORY);
  assert.ok(reference.every(t => !t.held), 'memory 0 never starts held');
  let left = 0, first = -1;
  for (let i = 0; i < memory.length; i++) {
    const expected = left > 0;
    assert.equal(memory[i]!.held, expected, `tick ${i + 1}: held start ${memory[i]!.held}, schedule ${expected}`);
    if (expected) { left--; if (first < 0) first = i; } else if (memory[i]!.entered) left = MEMORY;
  }
  assert.ok(first > 0, 'the run reaches the fail-safe and starts held steps');
  for (let i = 0; i < first; i++) assert.deepEqual(memory[i]!.positions, reference[i]!.positions, `tick ${i + 1}: identical before the first held start`);
}));

test('fail-safe memory is carried with a saved pose: the history endpoint restores the steps left that start held', { skip: !gpu }, async () => withDevice(async device => {
  const scene = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 32 }));
  const cloth = await ClothSystem.create(device, scene, { liveOptimizations: true, contactFailSafeMemory: 1024, statusInterval: 1 });
  try {
    const id = scene.cloth![0]!.id;
    let left = 0;
    for (let tick = 1; tick <= TICKS && left === 0; tick++) {
      await cloth.capture([tick * CLOTH_SCENE_STEP_S]);
      assert.equal(cloth.failure, undefined, cloth.failure?.message);
      left = (await cloth.contactFor(id)!.readCarriedState()).heldStart ?? 0;
    }
    assert.ok(left > 0, 'the run reaches the fail-safe');
    publishSceneClothSurfaces(scene, cloth.surfaces);
    const endpoint = snapshotSceneClothPose(scene), carried = endpoint.cloth![0]!.geometry.data.solverState!.contact!;
    assert.equal(carried.heldStart, left);
    assert.deepEqual(JSON.parse(JSON.stringify(carried)), carried);
    cloth.adopt(cloneScene(endpoint));
    await cloth.settled();
    assert.equal((await cloth.contactFor(id)!.readCarriedState()).heldStart, left, 'restored with the pose');
  } finally { cloth.destroy(); }
}));
