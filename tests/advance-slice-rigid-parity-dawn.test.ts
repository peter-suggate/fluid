import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import type { RigidBodyDescription, SceneDescription } from "../lib/core/model";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { GPU_RIGID_EXCHANGE_BYTES } from "../lib/core/webgpu-eulerian";
import { GPU_RIGID_STATE_BYTES, WebGPURigidBodySystem } from
  "../lib/core/webgpu-rigid-body";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { packSliceRigidBody, stepSliceRigidBodies } from
  "../lib/methods/adaptive-volume/advance-slice/slice-rigid-dynamics";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

function fixture(shape: "sphere" | "box"): {
  scene: SceneDescription; description: RigidBodyDescription;
} {
  const source = sceneDocument(getSceneDefinition("rigid-float"));
  const original = source.rigidBodies[0]!;
  const description: RigidBodyDescription = {
    ...original,
    id: `slice-rigid-${shape}`,
    name: `Slice rigid ${shape}`,
    shape,
    dimensions_m: shape === "sphere"
      ? { x: 0.11, y: 0.11, z: 0.11 }
      : { x: 0.10, y: 0.08, z: 0.06 },
    position_m: { x: 0, y: 0.42, z: 0 },
    orientation: { w: 0.98472655, x: 0.09164329, y: -0.12146645, z: 0.08512791 },
    linearVelocity_m_s: { x: 0.13, y: -0.07, z: 0.09 },
    angularVelocity_rad_s: { x: -0.31, y: 0.27, z: 0.16 },
    motion: "dynamic",
  };
  return { description, scene: {
    ...source,
    rigidBodies: [description],
    fluid: { ...source.fluid, gravity_m_s2: { x: 0, y: 0, z: 0 } },
  } };
}

async function gpuStep(device: GPUDevice, scene: SceneDescription,
  exchangeValues: Int32Array, dt: number, cellVolume: number): Promise<Float32Array> {
  const exchange = device.createBuffer({ label: "Slice rigid parity exchange",
    size: GPU_RIGID_EXCHANGE_BYTES,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
  const terrain = device.createTexture({ label: "Slice rigid parity flat terrain",
    size: [1, 1], format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const system = new WebGPURigidBodySystem(device, scene, exchange, terrain);
  const readback = device.createBuffer({ label: "Slice rigid parity state readback",
    size: GPU_RIGID_STATE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    system.syncBodies(initializeRigidBodies(scene.rigidBodies));
    for (const task of system.initializationTasks()) await task.run();
    device.queue.writeBuffer(exchange, 0, exchangeValues);
    const encoder = device.createCommandEncoder({ label: "Slice rigid parity step" });
    system.encode(encoder, dt, cellVolume, 1, scene.voxelDomain.finestCellSize_m);
    encoder.copyBufferToBuffer(system.stateBuffer, 0, readback, 0, GPU_RIGID_STATE_BYTES);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return Float32Array.from(new Float32Array(readback.getMappedRange()).subarray(0, 32));
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy(); system.destroy(); terrain.destroy(); exchange.destroy();
  }
}

function comparePacked(actual: Float32Array, expected: Float32Array): void {
  const differences: string[] = [];
  const bits = new Int32Array(1), value = new Float32Array(bits.buffer);
  const ordered = (source: number): number => {
    value[0] = source;
    const word = bits[0]!;
    return word < 0 ? 0x80000000 - word : word;
  };
  for (let lane = 0; lane < 32; lane += 1) {
    // Generation is an integer bitcast in the GPU record and does not
    // participate in integration. Selection is presentation-only.
    if (lane === 29 || lane === 31) continue;
    const ulps = Math.abs(ordered(actual[lane]!) - ordered(expected[lane]!));
    // Dawn/Metal may contract the quaternion multiply/normalize expression;
    // four ulps is the observed portable f32 envelope while every packed
    // scalar that does not cross that expression remains bit-identical.
    if (ulps > 4) differences.push(
      `${lane}: GPU=${actual[lane]} CPU=${expected[lane]} ulps=${ulps}`);
  }
  assert.deepEqual(differences, []);
}

dawnTest("2-D packed rigid integration matches the production GPU state record",
  { timeout: 30_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/advance-slice-rigid-parity-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU; globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice();

      for (const shape of ["sphere", "box"] as const) {
        const { scene, description } = fixture(shape);
        const state = initializeRigidBodies([description])[0]!;
        const packed = packSliceRigidBody(state, scene.fluid.density_kg_m3);
        // syncBodies publishes the motion generation as a u32 bitcast.
        new Uint32Array(packed.buffer)[29] = 1;
        const exchange = new Int32Array(12);
        if (shape === "box") {
          exchange.set([17000, -9000, 13000, 4200, -7100, 2900]);
          exchange[6] = 19_661;
          exchange[7] = 33_000;
          exchange[8] = -12_000;
          exchange[9] = 9_500;
          exchange[10] = 1;
          exchange[11] = 14_000;
        }
        const dt = 1 / 120, cellVolume = 0.05 ** 3;
        const expected = stepSliceRigidBodies({ bodies: [packed], descriptions: [description],
          exchange: [exchange], dt, densityKgM3: scene.fluid.density_kg_m3,
          gravity: scene.fluid.gravity_m_s2, cellVolumeM3: cellVolume });
        const actual = await gpuStep(device, scene, exchange, dt, cellVolume);
        comparePacked(actual, expected[0]!);
      }
    } finally {
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
