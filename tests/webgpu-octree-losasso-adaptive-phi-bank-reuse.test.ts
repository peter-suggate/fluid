import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS }
  from "../lib/webgpu-octree-losasso-adaptive-phi";
import { octreeLosassoAdaptivePhiCommitWGSL }
  from "../lib/webgpu-octree-losasso-adaptive-phi.wgsl";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const MAGIC = 0x4150_4849;

test("accepted topology reuse preserves the active adaptive phi bank", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for adaptive phi bank validation",
}, async () => {
  const dawn = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    | GPUBufferUsage.COPY_DST;
  const graph = device.createBuffer({ size: 128, usage });
  const state = device.createBuffer({ size: 80, usage });
  const receiptBytes = 4 * OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS;
  const receipts = device.createBuffer({ size: receiptBytes, usage });
  const readback = device.createBuffer({ size: 80 + receiptBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const shaderModule = device.createShaderModule({ code: octreeLosassoAdaptivePhiCommitWGSL });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto",
    compute: { module: shaderModule, entryPoint: "syncAcceptedCommit" } });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 1, resource: { buffer: graph } },
    { binding: 5, resource: { buffer: state } },
    { binding: 6, resource: { buffer: receipts } },
  ] });

  const run = async (graphWords: Uint32Array, stateWords: Uint32Array,
    receiptWords = new Uint32Array(OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS)) => {
    device.queue.writeBuffer(graph, 0, graphWords.buffer as ArrayBuffer,
      graphWords.byteOffset, graphWords.byteLength);
    device.queue.writeBuffer(state, 0, stateWords.buffer as ArrayBuffer,
      stateWords.byteOffset, stateWords.byteLength);
    device.queue.writeBuffer(receipts, 0, receiptWords.buffer as ArrayBuffer,
      receiptWords.byteOffset, receiptWords.byteLength);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(state, 0, readback, 0, 80);
    encoder.copyBufferToBuffer(receipts, 0, readback, 80, receiptBytes);
    device.pushErrorScope("validation"); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    assert.equal((await device.popErrorScope())?.message, undefined);
    await readback.mapAsync(GPUMapMode.READ);
    const result = Uint32Array.from(new Uint32Array(readback.getMappedRange()));
    readback.unmap(); return result;
  };

  const accepted = new Uint32Array(32);
  accepted.set([2, 2_509, 3_394, 2, 0, 5, 5], 0);
  const current = new Uint32Array(20);
  current.set([MAGIC, 2, 5, 5, 3_394, 2_509, 1, 1], 0);
  const reused = await run(accepted, current);
  assert.equal(reused[6], 1,
    "a rejected/identical topology attempt must not replay stale phi bank zero");
  assert.equal(reused[1], 2); assert.equal(reused[2], 5);
  assert.equal(reused[7], 1); assert.equal(reused[12], 0);
  assert.equal(reused[20 + 20], 1);

  const advancedGraph = accepted.slice();
  advancedGraph[5] = 6; advancedGraph[6] = 6;
  const advancedState = current.slice();
  advancedState[2] = 6; advancedState[3] = 5; advancedState[6] = 1;
  const fieldClockSync = await run(advancedGraph, advancedState);
  assert.equal(fieldClockSync[1], 2, "same-topology field sync retains the graph epoch");
  assert.equal(fieldClockSync[2], 6, "same-topology field sync retains the scalar generation");
  assert.equal(fieldClockSync[3], 6,
    "same-topology field sync adopts the completed nodal-velocity generation");
  assert.equal(fieldClockSync[6], 1,
    "same-topology field sync must preserve the active adaptive phi bank");
  assert.equal(fieldClockSync[7], 1); assert.equal(fieldClockSync[12], 0);

  const changedGraph = accepted.slice();
  changedGraph[0] = 3; changedGraph[3] = 3;
  const migrationReceipts = new Uint32Array(OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS);
  migrationReceipts[21] = 1; migrationReceipts[31] = 1; migrationReceipts[51] = 1;
  const changed = await run(changedGraph, current, migrationReceipts);
  assert.equal(changed[6], 0,
    "a committed graph epoch copies canonical candidate phi into bank zero");
  assert.equal(changed[1], 3); assert.equal(changed[2], 5);
  assert.equal(changed[7], 1); assert.equal(changed[12], 0);
  assert.equal(changed[20 + 20], 1,
    "a changed topology still publishes the accepted commit receipt");
  assert.equal(changed[20 + 51], 1,
    "the candidate volume transaction bank must survive the accepted commit sync");

  for (const buffer of [graph, state, receipts, readback]) buffer.destroy();
  device.destroy();
});
