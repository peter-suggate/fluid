#!/usr/bin/env node
/** Compile and execute the BTI1 service ABI against its CPU mirrors. */

import assert from "node:assert/strict";
import { writeSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
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
  compileSparseCM12BrickTileFaceProgram,
  validateSparseCM12BrickTileFaceProgram,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program";
import {
  sparseBrickKey,
  sparseBrickLadder,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock,
  releaseWebGPUExclusiveLockSync } from
  "../lib/harness/webgpu-smoke-isolation";

// Keep >2:1 grading bypass local to this topology-service adversary. The tail
// is the ordinary four-rung row; the prefix adds both unsupported-jump probes.
const resolutions = [8, 2, 8, 1, 8, 4, 2, 1] as const;
const lattice = [resolutions.length, 1, 1] as const;
const brick = (x: number, resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
  key: sparseBrickKey([x, 0, 0], lattice), coordinate: [x, 0, 0], resolution,
  density: new Float64Array(resolution ** 3),
  gamma: new Float64Array(resolution ** 3).fill(1),
});
const bricks = resolutions.map((resolution, x) => brick(x, resolution));
const directory = new Map(bricks.map((candidate) => [candidate.key, candidate] as const));
const atlas: SparseAdaptiveMassAtlas = {
  dimensions: [8 * resolutions.length, 8, 8],
  brickFineResolution: 8,
  brickCellCapacity: 8 ** 3,
  ladder: sparseBrickLadder(8),
  brickDimensions: lattice,
  bricks,
  directory,
  directoriesBySpan: new Map([[1, directory]]),
  maximumSpanBricks: 1,
  generation: 1,
};
const grid = buildSparseAtlasCompositeGrid(atlas);
const image = compileSparseCM12BrickTileImage(grid);
const faceProgram = compileSparseCM12BrickTileFaceProgram(image, grid);
validateSparseCM12BrickTileFaceProgram(faceProgram, image, grid);
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

let lockReleased = false;
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
  // The native instance owns asynchronous compile/map callbacks. Retain it
  // through the isolated process, as the other Dawn harnesses do, so GC cannot
  // destroy the instance while this gate is validating mixed-rung queries.
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${backend}`]);
  const adapter = await gpu.requestAdapter();
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
    const shaderModule = device.createShaderModule({ label: "BTI1 service check", code: shader });
    const info = await shaderModule.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length > 0) throw new Error(errors.map((message) =>
      `${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
    const entryPoints = ["checkCells", "checkPoints", "checkFaces"] as const;
    const pipelines = await Promise.all(entryPoints.map((entryPoint) =>
      device.createComputePipelineAsync({ label: `BTI1 ${entryPoint}`,
        layout: "auto", compute: { module: shaderModule, entryPoint } })));
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
    const receipt = `${JSON.stringify({ passed: true, backend,
      resolutions,
      cells: grid.cells.length, rows: grid.gradientRows.length,
      mixedSeamRows: grid.mixedSeamRowCount,
      seamPorts: faceProgram.layout.seamPortCount, shaderBytes: shader.length })}\n`;
    // Dawn's Node wrapper faults while finalizing an intentionally ungraded
    // B1 image on Metal. This gate is process-isolated, so after all readbacks
    // pass, return the repository lease synchronously and let the OS retire
    // the child process's native resources without running wrapper finalizers.
    releaseWebGPUExclusiveLockSync();
    lockReleased = true;
    writeSync(1, receipt);
    process.exit(0);
  } finally {
    // The standalone process owns these buffers; Dawn releases them with the
    // device after the repository lease is returned below.
  }
} finally {
  if (!lockReleased) await releaseWebGPUExclusiveLock();
}
