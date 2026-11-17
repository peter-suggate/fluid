// LANE P OWNS THIS FILE. Motor assemblies in a tank with no static voxel: the brick host's rigid system (no procedural
// source; bodies-host.ts) contacts the container planes as the old host's (the fluid's solid mask and the tank's
// shell, rigidAssemblySolidWGSL) contacts the shell (lane P, 2026-11-17; M5 step 3, paddleboats).
//
// brickRigidBodyGaps refuses motor assemblies only with static voxels: with none the old host's mask is empty and its
// solid is the shell, the walls and floor (no lid: paddleboats' top is open), which are the rigid system's container
// planes. The planes solve an assembly piece by piece (staticPlaneContact -> assemblyPlaneContact), as the shell does.
// Each case runs both systems for 60 steps with no fluid and compares the first hull's pose and velocity:
// - driven +x at 4 m/s into the end wall, zero g;
// - driven +z at 4 m/s into the side wall, zero g;
// - dropped under gravity onto the floor (no fluid: it lands on its hull and paddles).
// An edited shell (a cut, a block) has static voxels, so the brick refuses it (tests/rigid-assembly-solid-gpu covers
// the old host's contact with them).
import assert from "node:assert/strict";
import test from "node:test";
import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";
import { createPaddleboatsScene } from "../lib/core/paddleboats-scene";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { SCULPT_COLLISION_MAGIC } from "../lib/core/sculpt-collision.wgsl";
import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";
import { SCULPT_ARENA_WORDS } from "../lib/svo/features/scene-publication/svo-sculpt-substrate";
import { WebgpuSculptSubstrate } from "../lib/svo/features/scene-publication/webgpu-sculpt-substrate";
import { brickRigidBodyGaps, BrickRigidBodies } from "../lib/methods/narrow-band/bodies-host";
import { rigidMirrorSolid } from "./helpers/rigid-mirror-world";
import { withUniformDevice } from "./helpers/uniform-geometric";

type Scene = ReturnType<typeof createPaddleboatsScene>;
const STEPS = 60, DT = 1 / 60, H_M = 0.025;

/** The first boat alone, its hull driven at `v` (m/s) from (3, z) at gravity `g`. */
function boat(scene: Scene, z: number, v: { x: number; z: number }, g: number): Scene {
  const s = { ...scene, rigidBodies: scene.rigidBodies.slice(0, 9).map(b => structuredClone(b)), fluid: { ...scene.fluid, gravity_m_s2: { x: 0, y: -g, z: 0 } } };
  const hull = s.rigidBodies[0]!, dx = 3 - hull.position_m.x, dz = z - hull.position_m.z;
  for (const b of s.rigidBodies) b.position_m = { ...b.position_m, x: b.position_m.x + dx, z: b.position_m.z + dz };
  hull.linearVelocity_m_s = { x: v.x, y: 0, z: v.z };
  return s;
}

test("brick bodies: motor assemblies contact the container planes as the old host's shell (no static voxel), 60 steps", { skip: !process.env.WEBGPU_NODE_MODULE }, async t => {
  const base = createPaddleboatsScene();
  assert.deepEqual(brickRigidBodyGaps(base), [], "paddleboats' motor assemblies run on the brick host");
  const cases = [
    { label: "into the end wall", scene: boat(base, 0, { x: 4, z: 0 }, 0), check: (x: number[], v: number[]) => x[0]! < 6.4 && v[0]! < 2 },
    { label: "into the side wall", scene: boat(base, 1, { x: 0, z: 4 }, 0), check: (x: number[], v: number[]) => x[2]! < 2.4 && v[2]! < 2 },
    { label: "onto the floor", scene: boat(base, 0, { x: 0, z: 0 }, 9.81), check: (x: number[], v: number[]) => x[1]! > 0 && Math.abs(v[1]!) < 0.5 },
  ];
  await withUniformDevice("brick motor shell", async device => {
    for (const { label, scene, check } of cases) {
      // The old host's: the mask image at word 0, the sculpt arena, its pointer words (as rigid-assembly-solid-gpu).
      const solid = rigidMirrorSolid(scene);
      const words = solid.mask.length + SCULPT_ARENA_WORDS + 4, sculptWord = words - 4 - SCULPT_ARENA_WORDS;
      const scratch = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(scratch, 0, Uint32Array.from(solid.mask));
      device.queue.writeBuffer(scratch, 4 * (words - 4), new Uint32Array([sculptWord, SCULPT_COLLISION_MAGIC]));
      const sculpt = new WebgpuSculptSubstrate(device);
      const exchange = device.createBuffer({ size: GPU_RIGID_BODY_CAPACITY * 12 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const terrain = device.createTexture({ size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
      device.queue.writeTexture({ texture: terrain }, new Float32Array([0]), { bytesPerRow: 4 }, [1, 1]);
      const old = new WebGPURigidBodySystem(device, scene, exchange, terrain);
      const { cells, origin_m, cellSize_m, walls } = solid.layout;
      old.setProceduralCollisionSource({ buffer: scratch, pointerWord: words - 2, sculptCollision: true, origin_m, cellSize_m, assemblySolid: { cells, walls } });
      // The brick host's, as it builds it.
      const brick = await BrickRigidBodies.create(device, scene, { frame_m: [-scene.container.width_m / 2, 0, -scene.container.depth_m / 2], h_m: H_M, dt_s: DT, density_kg_m3: scene.fluid.density_kg_m3 });
      try {
        await sculpt.initialize();
        sculpt.publish(scratch, 4 * sculptWord, solid.plan.words);
        for (const task of old.initializationTasks()) await task.run(new AbortController().signal);
        const roster = initializeRigidBodies(scene.rigidBodies);
        old.syncBodies(roster); brick.system.syncBodies(roster);
        // Poses at steps 55 and 60: the velocity is their difference (the system reads back poses only).
        const poses: number[][][] = [];
        for (let i = 1; i <= STEPS; i++) {
          const e = device.createCommandEncoder();
          old.encode(e, DT, H_M ** 3, 1, H_M); brick.system.encode(e, DT, H_M ** 3, 1, H_M);
          device.queue.submit([e.finish()]);
          if (i === STEPS - 5 || i === STEPS) poses.push(await Promise.all([old, brick.system].map(async s => { const q = (await s.readPoses())![0]!.position_m; return [q.x, q.y, q.z]; })));
        }
        const pa = poses[1]![0]!, pb = poses[1]![1]!, va = pa.map((x, i) => (x - poses[0]![0]![i]!) / (5 * DT)), vb = pb.map((x, i) => (x - poses[0]![1]![i]!) / (5 * DT));
        const off = Math.hypot(...pa.map((x, i) => x - pb[i]!)), dv = Math.hypot(...va.map((x, i) => x - vb[i]!));
        t.diagnostic(`${label}: shell ${pa.map(x => x.toFixed(4))} v ${va.map(x => x.toFixed(3))}; planes ${pb.map(x => x.toFixed(4))} v ${vb.map(x => x.toFixed(3))}; |dx| ${off.toExponential(2)} |dv| ${dv.toExponential(2)}`);
        assert.ok(check(pa, va), `${label}: the shell stops the boat (${pa}, ${va})`);
        assert.ok(off < 5e-3 && dv < 5e-2, `${label}: planes ${pb} v ${vb} vs shell ${pa} v ${va}`);
      } finally { old.destroy(); brick.destroy(); sculpt.destroy(); scratch.destroy(); exchange.destroy(); terrain.destroy(); }
    }
  });
});
