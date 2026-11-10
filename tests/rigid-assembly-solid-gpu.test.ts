import assert from 'node:assert/strict';
import test from 'node:test';
import { GPU_RIGID_BODY_CAPACITY } from '../lib/core/rigid-capacity';
import { createPaddleboatsScene } from '../lib/core/paddleboats-scene';
import { initializeRigidBodies } from '../lib/core/rigid-body';
import { stepRigidMirror } from '../lib/core/rigid-assembly-contact';
import { SCULPT_COLLISION_MAGIC } from '../lib/core/sculpt-collision.wgsl';
import { sceneWithSolidStroke } from '../lib/core/solid-world';
import { WebGPURigidBodySystem } from '../lib/core/webgpu-rigid-body';
import { SCULPT_ARENA_WORDS } from '../lib/svo/features/scene-publication/svo-sculpt-substrate';
import { WebgpuSculptSubstrate } from '../lib/svo/features/scene-publication/webgpu-sculpt-substrate';
import { rigidMirrorSolid, rigidMirrorWorld } from './helpers/rigid-mirror-world';
import { withUniformDevice } from './helpers/uniform-geometric';

type Scene = ReturnType<typeof createPaddleboatsScene>;
/** The first boat, zero g, driven +x at 4 m/s from x = 3 at depth z toward the end wall at x = 6.4. */
function driven(scene: Scene, z: number): Scene {
  const s = { ...scene, rigidBodies: scene.rigidBodies.slice(0, 9).map(b => structuredClone(b)), fluid: { ...scene.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } } };
  const hull = s.rigidBodies[0]!, dx = 3 - hull.position_m.x, dz = z - hull.position_m.z;
  for (const b of s.rigidBodies) b.position_m = { ...b.position_m, x: b.position_m.x + dx, z: b.position_m.z + dz };
  hull.linearVelocity_m_s = { x: 4, y: 0, z: 0 };
  return s;
}

test('the GPU kernel contacts the edited voxel shell and pool blocks as its mirror does', { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const base = createPaddleboatsScene();
  const hole = sceneWithSolidStroke(base, [{ operation: 'clear', affectsScenery: true, minimum: [496, 0, 66], maximumExclusive: [528, 34, 126] }]);
  const block = sceneWithSolidStroke(base, [{ operation: 'fill', minimum: [440, 0, 80], maximumExclusive: [456, 32, 112], materialId: 1 }]);
  const cases = [
    { label: 'through the hole', scene: driven(hole, 0), check: (x: number, vx: number) => x > 6.9 && Math.abs(vx - 4) < 1e-2 },
    { label: 'beside the hole', scene: driven(hole, .75), check: (x: number, vx: number) => x < 5.7 && vx < 2 },
    { label: 'at a pool block', scene: driven(block, 0), check: (x: number, vx: number) => x < 4 && vx < 2 },
  ];
  await withUniformDevice('GPU assembly voxel contact', async device => {
    for (const { label, scene, check } of cases) {
      const solid = rigidMirrorSolid(scene), mirror = rigidMirrorWorld(scene, { solid });
      // The uniform reference's scratch: the mask image at word 0, the sculpt arena, then its pointer words.
      const words = solid.mask.length + SCULPT_ARENA_WORDS + 4, sculptWord = words - 4 - SCULPT_ARENA_WORDS;
      const scratch = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(scratch, 0, Uint32Array.from(solid.mask));
      device.queue.writeBuffer(scratch, 4 * (words - 4), new Uint32Array([sculptWord, SCULPT_COLLISION_MAGIC]));
      const sculpt = new WebgpuSculptSubstrate(device);
      const exchange = device.createBuffer({ size: GPU_RIGID_BODY_CAPACITY * 12 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const terrain = device.createTexture({ size: [1, 1], format: 'r32float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture: terrain }, new Float32Array([0]), { bytesPerRow: 4 }, [1, 1]);
      const rigid = new WebGPURigidBodySystem(device, scene, exchange, terrain);
      const { cells, origin_m, cellSize_m, walls } = solid.layout;
      rigid.setProceduralCollisionSource({ buffer: scratch, pointerWord: words - 2, sculptCollision: true, origin_m, cellSize_m, assemblySolid: { cells, walls } });
      try {
        await sculpt.initialize();
        sculpt.publish(scratch, 4 * sculptWord, solid.plan.words);
        for (const task of rigid.initializationTasks()) await task.run(new AbortController().signal);
        rigid.syncBodies(initializeRigidBodies(scene.rigidBodies));
        for (let i = 0; i < 60; i++) {
          const e = device.createCommandEncoder(); rigid.encode(e, 1 / 60, .025 ** 3, 1, .025); device.queue.submit([e.finish()]);
          stepRigidMirror(mirror, 1 / 60);
        }
        const pose = (await rigid.readPoses())![0]!, hull = mirror.bodies[0]!;
        const off = Math.hypot(...(['x', 'y', 'z'] as const).map((a, i) => pose.position_m[a] - hull.positionShape[i]!));
        assert.ok(off < 5e-3, `${label}: GPU hull ${JSON.stringify(pose.position_m)} vs mirror ${hull.positionShape.join(',')}`);
        assert.ok(check(hull.positionShape[0]!, hull.linearVelocity[0]!), `${label}: mirror x=${hull.positionShape[0]}, vx=${hull.linearVelocity[0]}`);
      } finally { rigid.destroy(); sculpt.destroy(); scratch.destroy(); exchange.destroy(); terrain.destroy(); }
    }
  });
});
