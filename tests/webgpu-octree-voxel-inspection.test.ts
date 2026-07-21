import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { getScenePreset } from "../lib/scenes";
import { CompactOctreeVoxelInspection, compactOctreeVoxelInspectionShader } from "../lib/webgpu-octree-voxel-inspection";

const source = readFileSync(new URL("../lib/webgpu-octree-voxel-inspection.ts", import.meta.url), "utf8");
const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const modulePath = process.env.WEBGPU_NODE_MODULE;

test("page-native raw voxels materialize directly from live pressure-grid rows", () => {
  assert.match(compactOctreeVoxelInspectionShader, /struct LeafHeader/);
  assert.match(compactOctreeVoxelInspectionShader, /let live=row<min\(rowCount\[0\],params\.shape\.x\)&&header\.size>0u/);
  assert.match(compactOctreeVoxelInspectionShader, /let header=headers\[row\]/);
  assert.match(compactOctreeVoxelInspectionShader, /header\.cell%nx/,
    "the pressure header's finest-cell index must determine the rendered voxel origin");
  assert.match(compactOctreeVoxelInspectionShader, /f32\(header\.size\)\*params\.cellSize\.xyz/,
    "the pressure header's dyadic size must determine the rendered voxel extent");
  assert.match(compactOctreeVoxelInspectionShader, /if\(!live\).*voxelRecords\[row\]=inactiveRecord\(\)/s,
    "retired pressure rows must be cleared when the compact frontier shrinks");
  assert.match(compactOctreeVoxelInspectionShader, /voxelRecords\[row\]=record/);
  assert.match(compactOctreeVoxelInspectionShader, /brickRecords\[row\]=record/);
  assert.doesNotMatch(compactOctreeVoxelInspectionShader, /textureLoad|textureStore|mapAsync/,
    "inspection must not recreate or read back the retired dense level-set volume");
});

test("compact inspection stays lazy and republishes with page-native simulation frames", () => {
  assert.match(source, /createSparseVoxelInspectionPublicationController\(false/,
    "smooth rendering allocates no expanded records");
  assert.match(source, /drawContainerGlass: false/,
    "page-native fluid inspection is not hidden behind filled tank panes");
  assert.match(source, /pass\.dispatchWorkgroups\(Math\.ceil\(this\.rowCapacity \/ 64\)\)/);
  assert.match(projection, /leafHeaders: \{ buffer: this\.leafHeaders \}/);
  assert.match(projection, /rowCount: \{ buffer: this\.compaction \}/);
  assert.match(projection, /this\.topologyWorklistReady = false;[\s\S]*this\.encodeInlineRebuild\(encoder\);[\s\S]*this\.encodeFrontierRows\(encoder, "Octree inspection frontier rows"[\s\S]*this\.compactVoxelInspection\.encode\(encoder\);/,
    "the first paused inspection captures transient pressure rows in the same rebuild submission");
  const pagedBranch = projection.slice(
    projection.indexOf("if (this.surfacePagesBootstrapped && this.adaptiveSurfaceAdapter)"),
    projection.indexOf("if (!this.sparseBrickWorld)", projection.indexOf("if (this.surfacePagesBootstrapped && this.adaptiveSurfaceAdapter)")),
  );
  assert.match(pagedBranch, /this\.compactVoxelInspection\?\.encode\(encoder\)/,
    "running fluid updates the raw view from the same compact generation");
});

test("GPU materialization decodes compact pressure headers and clears retired rows", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU compact-inspection checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const capacity = 4;
  const headers = device.createBuffer({ size: capacity * 48, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const count = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const headerData = new ArrayBuffer(capacity * 48);
  // Finest-cell origin (3,2,1) in an 8x6x5 grid, represented by a 2^3 leaf.
  new Uint32Array(headerData, 0, 4).set([3 + 8 * (2 + 6 * 1), 0, 0, 2]);
  device.queue.writeBuffer(headers, 0, headerData);
  device.queue.writeBuffer(count, 0, new Uint32Array([1]));
  const scene = getScenePreset("water-box-dam-break").create();
  const inspection = new CompactOctreeVoxelInspection(device, scene, [8, 6, 5], {
    leafHeaders: { buffer: headers }, rowCount: { buffer: count }, rowCapacity: capacity,
  });
  inspection.source.inspectionPublication!.setEnabled(true);
  const readback = device.createBuffer({ size: capacity * 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  inspection.encode(encoder);
  encoder.copyBufferToBuffer(inspection.source.voxelRecords.buffer, 0, readback, 0, capacity * 48);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const bytes = readback.getMappedRange().slice(0); readback.unmap();
  const floats = new Float32Array(bytes), uints = new Uint32Array(bytes);
  const expectedOrigin = [-scene.container.width_m / 2 + 3 * scene.container.width_m / 8,
    2 * scene.container.height_m / 6, -scene.container.depth_m / 2 + scene.container.depth_m / 5];
  const expectedExtent = [2 * scene.container.width_m / 8, 2 * scene.container.height_m / 6, 2 * scene.container.depth_m / 5];
  Array.from(floats.slice(0, 3)).forEach((value, axis) => assert.ok(Math.abs(value - expectedOrigin[axis]) < 1e-6));
  Array.from(floats.slice(4, 7)).forEach((value, axis) => assert.ok(Math.abs(value - expectedExtent[axis]) < 1e-6));
  assert.deepEqual(Array.from(uints.slice(8, 12)), [3, 1, 1, 0xffff_ffff]);
  assert.equal(uints[12 + 9], 0, "the first unused row is explicitly inactive");
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  inspection.destroy(); headers.destroy(); count.destroy(); readback.destroy(); device.destroy();
});
