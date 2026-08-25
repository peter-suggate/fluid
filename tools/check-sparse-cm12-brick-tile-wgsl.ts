#!/usr/bin/env node
/** Compile and execute the BTI1 service ABI against its CPU mirrors. */

import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  compileSparseCM12BrickTileImage,
  sparseCM12BrickTileCell,
  sparseCM12BrickTileCellAtFine,
  sparseCM12BrickTileRows,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import { createSparseCM12BrickTileImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image.wgsl";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const lattice = [2, 1, 1] as const;
const brick = (x: number, resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
  key: sparseBrickKey([x, 0, 0], lattice), coordinate: [x, 0, 0], resolution,
  density: new Float64Array(resolution ** 3),
  gamma: new Float64Array(resolution ** 3).fill(1),
});
const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [brick(0, 8), brick(1, 4)],
  1, 8);
const grid = buildSparseAtlasCompositeGrid(atlas);
const image = compileSparseCM12BrickTileImage(grid);
const outputWords = Math.max(image.layout.tileCapacity * 64,
  atlas.dimensions[0] * atlas.dimensions[1] * atlas.dimensions[2],
  image.layout.tileCapacity * 6 * 64 * 2);
const service = createSparseCM12BrickTileImageWGSL({ layout: image.layout,
  arenaName: "acceptedTopology" });
const shader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> acceptedTopology:array<u32>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
${service}
@compute @workgroup_size(64)
fn checkCells(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let tile=wid.x;if(tile<BTI1_TILE_CAPACITY){output[64u*tile+lane]=bti1Cell(tile,lane);}
}
@compute @workgroup_size(64)
fn checkPoints(@builtin(global_invocation_id)gid:vec3u){
  let dims=vec3u(${atlas.dimensions[0]}u,${atlas.dimensions[1]}u,${atlas.dimensions[2]}u);
  let count=dims.x*dims.y*dims.z;if(gid.x>=count){return;}let z=gid.x/(dims.x*dims.y);
  let remain=gid.x-z*dims.x*dims.y;let y=remain/dims.x;let x=remain-y*dims.x;
  output[gid.x]=bti1PointOwner(vec3u(x,y,z));
}
@compute @workgroup_size(64)
fn checkFaces(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
  let tile=wid.x;if(tile>=BTI1_TILE_CAPACITY){return;}for(var family=0u;family<6u;family+=1u){
    let address=384u*tile+64u*family+lane;let count=bti1FaceRowCount(tile,family,lane);
    var sum=0u;for(var ordinal=0u;ordinal<count;ordinal+=1u){sum+=bti1FaceRow(tile,family,lane,ordinal);}
    output[2u*address]=count;output[2u*address+1u]=sum;}
}
`;

await acquireWebGPUExclusiveLock("dawn-check", "tools/check-sparse-cm12-brick-tile-wgsl.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const backend = process.env.WEBGPU_BACKEND ?? "metal";
  const adapter = await dawn.create([`backend=${backend}`]).requestAdapter();
  if (!adapter) throw new Error(`No Dawn adapter for ${backend}`);
  const device = await adapter.requestDevice();
  const topology = device.createBuffer({ size: image.words.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: outputWords * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  const readback = device.createBuffer({ size: outputWords * 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(topology, 0, image.words.buffer as ArrayBuffer,
      image.words.byteOffset, image.words.byteLength);
    const module = device.createShaderModule({ label: "BTI1 service check", code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
    const pipelines = await Promise.all(["checkCells", "checkPoints", "checkFaces"].map(
      (entryPoint) => device.createComputePipelineAsync({ label: `BTI1 ${entryPoint}`,
        layout: "auto", compute: { module, entryPoint } }),
    ));
    for (let check = 0; check < pipelines.length; check += 1) {
      device.queue.writeBuffer(output, 0, new Uint32Array(outputWords));
      const pipeline = pipelines[check]!;
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: topology } },
        { binding: 1, resource: { buffer: output } },
      ] });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0, group);
      if (check === 1) pass.dispatchWorkgroups(Math.ceil(
        atlas.dimensions[0] * atlas.dimensions[1] * atlas.dimensions[2] / 64));
      else pass.dispatchWorkgroups(image.layout.tileCapacity);
      pass.end();encoder.copyBufferToBuffer(output, 0, readback, 0, outputWords * 4);
      device.queue.submit([encoder.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const actual = new Uint32Array(readback.getMappedRange()).slice();readback.unmap();
      if (check === 0) for (let tile = 0; tile < image.layout.tileCapacity; tile += 1)
        for (let lane = 0; lane < 64; lane += 1) assert.equal(actual[64 * tile + lane],
          sparseCM12BrickTileCell(image, tile, lane) ?? 0xffff_ffff);
      if (check === 1) for (let z = 0; z < atlas.dimensions[2]; z += 1)
        for (let y = 0; y < atlas.dimensions[1]; y += 1)
          for (let x = 0; x < atlas.dimensions[0]; x += 1) {
            const at = x + atlas.dimensions[0] * (y + atlas.dimensions[1] * z);
            assert.equal(actual[at], sparseCM12BrickTileCellAtFine(image, [x, y, z])
              ?? 0xffff_ffff);
          }
      if (check === 2) for (let tile = 0; tile < image.layout.tileCapacity; tile += 1)
        for (let family = 0; family < 6; family += 1)
          for (let lane = 0; lane < 64; lane += 1) {
            const rows = sparseCM12BrickTileRows(image, tile, family, lane);
            const at = 2 * (384 * tile + 64 * family + lane);
            assert.equal(actual[at], rows.length);
            assert.equal(actual[at + 1], rows.reduce((sum, row) => (sum + row) >>> 0, 0));
          }
    }
    process.stdout.write(`${JSON.stringify({ passed: true, backend,
      cells: grid.cells.length, rows: grid.gradientRows.length,
      shaderBytes: shader.length })}\n`);
  } finally {
    topology.destroy();output.destroy();readback.destroy();device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
