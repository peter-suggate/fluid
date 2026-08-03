import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  FineLevelSetBrickOracle,
  packFineLevelSetBrickKey,
  planFineLevelSetBricks,
} from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks } from "../lib/webgpu-octree-fine-levelset-bricks";
import { PassBroker } from "../lib/webgpu-pass-broker";
import {
  FINE_LEVELSET_TOPOLOGY_ERROR,
  WebGPUFineLevelSetTopology,
  unpackFineLevelSetGPUTopologyControl,
} from "../lib/webgpu-octree-fine-levelset-topology";

// The Dawn Node binding schedules a native instance event pump beyond the
// final device callback. Retain the instance for the test-process lifetime so
// V8 cannot collect its mutex while that pump is still armed.
const retainedDawnInstances: GPU[] = [];

test("Dawn atomically rolls back a downstream-rejected fine generation and accepts a retry", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU fine-publication checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([
    `backend=${process.env.WEBGPU_BACKEND ?? "metal"}`,
  ]);
  retainedDawnInstances.push(gpu);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  assert.ok(adapter.limits.maxStorageBuffersPerShaderStage >= 10);
  const device = await adapter.requestDevice({
    requiredLimits: { maxStorageBuffersPerShaderStage: 10 },
  });
  device.pushErrorScope("validation");

  const plan = planFineLevelSetBricks({
    domainOrigin: [0, 0, 0],
    finestCellDimensions: [2, 1, 1],
    finestCellWidth: 1,
    fineFactor: 4,
    brickResolution: 4,
    maximumResidentBricks: 2,
  });
  const oracle = new FineLevelSetBrickOracle(plan);
  oracle.publishInterfaceAndRing(
    [packFineLevelSetBrickKey(plan, [0, 0, 0])],
    ([x]) => x - 0.5,
  );
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const current = owner.uploadGeneration(oracle.exportGPUGeneration());
  const target = owner.prepareGPUGeneration(2);
  const topology = new WebGPUFineLevelSetTopology(
    device,
    current,
    target,
    "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-0.5;}",
  );
  const redistanceControl = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const run = async (redistanceWords: Uint32Array): Promise<ArrayBuffer> => {
    const redistanceUpload = new ArrayBuffer(redistanceWords.byteLength);
    new Uint32Array(redistanceUpload).set(redistanceWords);
    device.queue.writeBuffer(redistanceControl, 0, redistanceUpload);
    const readback = device.createBuffer({
      size: 48,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const broker = new PassBroker(device.createCommandEncoder());
    topology.encode(broker, undefined, [], undefined, true);
    topology.encodeFinalizePublication(broker, { redistance: redistanceControl });
    broker.copyBufferToBuffer(topology.control, 0, readback, 0, 36);
    broker.copyBufferToBuffer(target.worklist, 0, readback, 36, 8);
    broker.copyBufferToBuffer(target.samples, 0, readback, 44, 4);
    const commands = broker.finish();
    device.queue.submit([commands]);
    await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return bytes;
  };

  const rejected = await run(new Uint32Array(4));
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  const rejectedControl = unpackFineLevelSetGPUTopologyControl(
    new Uint32Array(rejected, 0, 9),
  );
  assert.notEqual(
    rejectedControl.flags & FINE_LEVELSET_TOPOLOGY_ERROR.downstreamPublication,
    0,
    `rejected publication control ${JSON.stringify(rejectedControl)}`,
  );
  assert.equal(rejectedControl.published, true);
  assert.equal(rejectedControl.rolledBack, true);
  assert.deepEqual([...new Uint32Array(rejected, 36, 2)], [2, 2]);
  assert.ok(new Float32Array(rejected, 44, 1)[0] < 0);

  const retried = await run(new Uint32Array([0, 0, 1, 1]));
  const retriedControl = unpackFineLevelSetGPUTopologyControl(
    new Uint32Array(retried, 0, 9),
  );
  assert.equal(retriedControl.flags, 0);
  assert.equal(retriedControl.published, true);
  assert.equal(retriedControl.rolledBack, false);
  assert.deepEqual([...new Uint32Array(retried, 36, 2)], [2, 2]);

  redistanceControl.destroy();
  topology.destroy();
  owner.destroy();
});
