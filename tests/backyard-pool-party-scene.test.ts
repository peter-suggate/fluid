import assert from "node:assert/strict";
import test from "node:test";
import { createBackyardPoolPartyScene, POOL_PARTY_BOARD, POOL_PARTY_WATERLINE_M, POOL_PARTY_CELL_M, POOL_PARTY_GRID } from "../lib/core/backyard-pool-party-scene";
import { validateScene } from "../lib/core/model";
import { initializeRigidBodies, quaternionInverseRotate, quaternionRotate } from "../lib/core/rigid-body";
import { proceduralSolidField } from "../lib/core/procedural-solid";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { WebGPURigidBodySystem } from "../lib/core/webgpu-rigid-body";
import { withUniformDevice } from "./helpers/uniform-geometric";

test("pool party is discoverable with a smooth voxel board and real water settings", () => {
  const definition = getSceneDefinition("backyard-pool-party");
  const scene = sceneDocument(definition);
  assert.equal(definition.shelf, "Garden");
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(scene.solidVoxels, createBackyardPoolPartyScene().solidVoxels);
  assert.equal(scene.fluid.gravity_m_s2.y, -9.80665);
  assert.equal(scene.fluid.density_kg_m3, 998.2);
  assert.equal(scene.rigidBodies.length, 1, "board is a voxel solid, not a rigid body");
  assert.equal(scene.rigidBodies[0]!.shape, "sphere");
  assert.equal(scene.surfaceStyle, "smooth");
  assert.equal(scene.fluid.initialLiquidVolumes?.length, 1);
  assert.equal(scene.container.top, "open");
  assert.equal(scene.voxelDomain.finestCellSize_m, .05);
  assert.deepEqual(scene.solidBoundary?.dimensions, POOL_PARTY_GRID);
  const field = proceduralSolidField(scene)!;
  assert.ok(field, "catalog finishing retains the smooth boundary provenance");
  for (const x of [-1.5, 0, 1.5]) {
    const p = quaternionRotate(POOL_PARTY_BOARD.orientation, { x, y: .08, z: 0 });
    const sample = field.sample([(p.x + POOL_PARTY_BOARD.position.x + 8) / POOL_PARTY_CELL_M,
      (p.y + POOL_PARTY_BOARD.position.y) / POOL_PARTY_CELL_M, 6 / POOL_PARTY_CELL_M]);
    assert.ok(sample);
    assert.ok(Math.abs(sample[3]) < 1e-5, "smooth surface follows the sloped plane");
    assert.ok(sample[0] > 0 && sample[1] > .9, "normal points uphill in the voxel field");
  }
});

test("GPU ball rolls from rest on the voxelized board and reaches the pool", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("pool party voxel board", async device => {
    const scene = createBackyardPoolPartyScene();
    const field = proceduralSolidField(scene)!, image = field.pack();
    const packed = new Uint32Array(image.length + 1); packed[0] = 1; packed.set(image, 1);
    const boundary = device.createBuffer({ size: packed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(boundary, 0, packed);
    const terrain = device.createTexture({ size: [1, 1], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING });
    const exchange = device.createBuffer({ size: 12 * 12 * 4, usage: GPUBufferUsage.STORAGE });
    const rigid = new WebGPURigidBodySystem(device, scene, exchange, terrain);
    rigid.setProceduralCollisionSource({ buffer: boundary, pointerWord: 0, origin_m: [-8, 0, -6], cellSize_m: [POOL_PARTY_CELL_M, POOL_PARTY_CELL_M, POOL_PARTY_CELL_M] });
    rigid.syncBodies(initializeRigidBodies(scene.rigidBodies));
    try {
      for (const task of rigid.initializationTasks()) await task.run(new AbortController().signal);
      let rolling = false, cleared = false, landed = false;
      for (let step = 1; step <= 600; step++) {
        const encoder = device.createCommandEncoder();
        rigid.encode(encoder, 1 / 60, POOL_PARTY_CELL_M ** 3, 1, POOL_PARTY_CELL_M);
        device.queue.submit([encoder.finish()]);
        if (step % 15 !== 0) continue;
        const pose = (await rigid.readPoses())![0]!;
        const local = quaternionInverseRotate(POOL_PARTY_BOARD.orientation, {
          x: pose.position_m.x - POOL_PARTY_BOARD.position.x,
          y: pose.position_m.y - POOL_PARTY_BOARD.position.y, z: pose.position_m.z,
        });
        if (step === 120) {
          assert.ok(local.x > -.7, `gravity starts the ball rolling: ${JSON.stringify(pose)}`);
          assert.ok(Math.abs(local.y - .32) < .02, "ball remains on the sloped voxel surface");
          assert.ok(Math.abs(pose.orientation.z) > .1, "contact creates spin");
          rolling = true;
        }
        if (local.x > 2.24) cleared = true;
        if (pose.position_m.y - .24 <= POOL_PARTY_WATERLINE_M) {
          assert.ok(cleared, "ball clears the tip before entering the water");
          assert.ok(pose.position_m.x > -3.76 && pose.position_m.x < 3.76, "whole ball lands inside basin");
          assert.ok(Math.abs(pose.position_m.z) < 1.76);
          landed = true; break;
        }
      }
      assert.ok(rolling && cleared && landed);
      // The same pointer is invalidated when procedural geometry is removed.
      device.queue.writeBuffer(boundary, 0, new Uint32Array([0]));
      rigid.syncBodies(initializeRigidBodies(scene.rigidBodies.map(body => ({ ...body, id: `${body.id}/reset` }))));
      assert.ok((await rigid.readPoses())![0]!.position_m.y > 5.5, "reset really restores the release pose");
      for (let i = 0; i < 30; i++) {
        const encoder = device.createCommandEncoder(); rigid.encode(encoder, 1 / 60, POOL_PARTY_CELL_M ** 3, 1, POOL_PARTY_CELL_M);
        device.queue.submit([encoder.finish()]);
      }
      assert.ok((await rigid.readPoses())![0]!.position_m.y < 4.6, "removing the boundary removes contact");
    } finally { rigid.destroy(); boundary.destroy(); terrain.destroy(); exchange.destroy(); }
  });
});
