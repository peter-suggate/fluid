#!/usr/bin/env node
/** Matched GPU service microbenchmark for BTI1, TEI2/LOD1, and dense arithmetic. */

import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12BrickTileImage } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import { createSparseCM12BrickTileImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image.wgsl";
import { compileSparseCM12BrickTileFaceProgram } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program";
import { createSparseCM12BrickTileFaceProgramWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program.wgsl";
import { createSparseCM12LogicalOwnerDirectory,
  type SparseCM12LogicalOwnerRuntime } from
  "../lib/methods/adaptive-mass/sparse-cm12-logical-owner-directory";
import { createSparseCM12TransportExecutionImage } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image";
import { createSparseAdaptiveMassAtlas, sparseBrickKey,
  type SparseAdaptiveMassBrick, type SparseBrickResolution } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const args = new Map(process.argv.slice(2).map((argument) => {
  const [key, value = "true"] = argument.replace(/^--/, "").split("=", 2);
  return [key, value] as const;
}));
const lattice = (args.get("lattice") ?? "8,4,4").split(",").map(Number) as
  [number, number, number];
const repeats = Math.max(1, Number(args.get("repeats") ?? 64));
const sampleCount = Math.max(5, Number(args.get("samples") ?? 15));
const outputPath = args.get("out");
const dimensions = lattice.map((value) => 8 * value) as [number, number, number];
const bricks: SparseAdaptiveMassBrick[] = [];
for (let z = 0; z < lattice[2]; z += 1)
  for (let y = 0; y < lattice[1]; y += 1)
    for (let x = 0; x < lattice[0]; x += 1) {
      const resolution: SparseBrickResolution = (x + 2 * y + 3 * z) % 3 === 0 ? 4 : 8;
      const count = resolution ** 3;
      bricks.push({ key: sparseBrickKey([x, y, z], lattice), coordinate: [x, y, z],
        resolution, density: new Float64Array(count),
        gamma: new Float64Array(count).fill(1) });
    }
const atlas = createSparseAdaptiveMassAtlas(dimensions, bricks, 1, undefined, 8);
const grid = buildSparseAtlasCompositeGrid(atlas);
const bti = compileSparseCM12BrickTileImage(grid);
const faceProgram = compileSparseCM12BrickTileFaceProgram(bti, grid);
const logical = createSparseCM12LogicalOwnerDirectory(atlas);
const runtime: SparseCM12LogicalOwnerRuntime = {
  brickActive: () => true,
  acceptedBrickResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  templateBrickCellRange: (leaf) => {
    const key = atlas.bricks[leaf]!.key;
    const first = grid.cellBaseByBrick.get(key)!;
    let count = 0;
    while (first + count < grid.cells.length && grid.cells[first + count]!.brickKey === key) {
      count += 1;
    }
    return [first, count];
  },
};
const tei = createSparseCM12TransportExecutionImage(atlas, logical, runtime);
const descriptors = new Uint32Array(8 * atlas.bricks.length);
for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1) {
  const brick = atlas.bricks[leaf]!, first = grid.cellBaseByBrick.get(brick.key)!;
  let count = 0;while (first + count < grid.cells.length
    && grid.cells[first + count]!.brickKey === brick.key) count += 1;
  const cells = grid.cells.slice(first, first + count);
  const valid = [0, 0, 0];for (const cell of cells) for (let axis = 0; axis < 3; axis += 1) {
    valid[axis] = Math.max(valid[axis]!, cell.local[axis]! + 1);
  }
  const scale = 8 / brick.resolution, at = 8 * leaf;
  descriptors.set([first, valid[0]! | (valid[1]! << 8) | (valid[2]! << 16), scale,
    8 * brick.coordinate[0], 8 * brick.coordinate[1], 8 * brick.coordinate[2], count, 0], at);
}
const pointCount = dimensions[0] * dimensions[1] * dimensions[2];
const denseOwners = new Uint32Array(pointCount);
for (const cell of grid.cells)
  for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1)
    for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
      for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
        denseOwners[x + dimensions[0] * (y + dimensions[1] * z)] = cell.id;
      }

const btiWGSL = createSparseCM12BrickTileImageWGSL({ layout: bti.layout,
  arenaName: "btiImage" });
const faceProgramWGSL = createSparseCM12BrickTileFaceProgramWGSL({
  layout: faceProgram.layout, arenaName: "faceProgram" });
const shader = /* wgsl */ `
@group(0) @binding(0) var<storage,read> btiImage:array<u32>;
@group(0) @binding(1) var<storage,read> teiImage:array<u32>;
@group(0) @binding(2) var<storage,read> lodImage:array<u32>;
@group(0) @binding(3) var<storage,read> leafDescriptors:array<u32>;
@group(0) @binding(4) var<storage,read> denseOwners:array<u32>;
@group(0) @binding(5) var<storage,read> faceProgram:array<u32>;
@group(0) @binding(6) var<storage,read_write> output:array<u32>;
${btiWGSL}
${faceProgramWGSL}
const POINT_DIMS:vec3u=vec3u(${dimensions[0]}u,${dimensions[1]}u,${dimensions[2]}u);
fn pointCoordinate(id:u32)->vec3u{let z=id/(POINT_DIMS.x*POINT_DIMS.y);
  let remain=id-z*POINT_DIMS.x*POINT_DIMS.y;let y=remain/POINT_DIMS.x;
  return vec3u(remain-y*POINT_DIMS.x,y,z);}
@compute @workgroup_size(64) fn enumerateDense(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x<${grid.cells.length}u){output[gid.x]=gid.x;}}
@compute @workgroup_size(64) fn enumerateBTI1(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){if(wid.x<BTI1_TILE_CAPACITY){
  output[64u*wid.x+lane]=bti1Cell(wid.x,lane);}}
@compute @workgroup_size(64) fn enumerateTEI2(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){if(wid.x>=BTI1_TILE_CAPACITY){return;}
  let leaf=wid.x/8u;let localPacket=wid.x-leaf*8u;let packet=leaf*64u+localPacket;
  let at=${tei.layout.slotPacketBaseOffsets[0]}u+4u*packet;let first=teiImage[at+1u];
  let packed=teiImage[at+2u];let counts=vec3u(packed&31u,(packed>>5u)&31u,(packed>>10u)&31u);
  let local=bti1Local(lane);if(first==BTI1_INVALID||any(local>=counts)){
    output[64u*wid.x+lane]=BTI1_INVALID;return;}let strides=teiImage[at+3u];
  output[64u*wid.x+lane]=first+local.x+(strides&0xffffu)*local.y+(strides>>16u)*local.z;}
@compute @workgroup_size(64) fn pointDense(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x<${pointCount}u){output[gid.x]=denseOwners[gid.x];}}
@compute @workgroup_size(64) fn pointBTI1(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x<${pointCount}u){output[gid.x]=bti1PointOwner(pointCoordinate(gid.x));}}
@compute @workgroup_size(64) fn pointLOD1(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=${pointCount}u){return;}let position=pointCoordinate(gid.x);let logical=position/vec3u(8u);
  let key=logical.x+${lattice[0]}u*(logical.y+${lattice[1]}u*logical.z);
  let packed=lodImage[${logical.layout.recordBaseWords}u+2u*key];let leaf=packed>>5u;
  if(leaf>=${atlas.bricks.length}u){output[gid.x]=BTI1_INVALID;return;}let at=8u*leaf;
  let dimsPacked=leafDescriptors[at+1u];let dims=vec3u(dimsPacked&255u,
    (dimsPacked>>8u)&255u,(dimsPacked>>16u)&255u);let scale=leafDescriptors[at+2u];
  let origin=vec3u(leafDescriptors[at+3u],leafDescriptors[at+4u],leafDescriptors[at+5u]);
  let local=(position-origin)/vec3u(scale);output[gid.x]=leafDescriptors[at]
    +local.x+dims.x*(local.y+dims.y*local.z);}
@compute @workgroup_size(64) fn faceDense(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x<${grid.gradientRows.length}u){output[gid.x]=gid.x;}}
@compute @workgroup_size(64) fn faceBTI1(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){if(wid.x>=BTI1_TILE_CAPACITY){return;}
  for(var family=0u;family<6u;family+=1u){let count=bti1FaceRowCount(wid.x,family,lane);
    var sum=0u;for(var ordinal=0u;ordinal<count;ordinal+=1u){
      sum+=bti1FaceRow(wid.x,family,lane,ordinal);}
    output[384u*wid.x+64u*family+lane]=sum;}}
@compute @workgroup_size(64) fn faceSplitInterior(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){let tile=bfp1InteriorTile(wid.x);
  if(tile==BTI1_INVALID||bti1Cell(tile,lane)==BTI1_INVALID){return;}
  for(var axis=0u;axis<3u;axis+=1u){let row=bti1ImplicitRow(tile,axis,lane);
    output[192u*wid.x+64u*axis+lane]=row;}}
@compute @workgroup_size(64) fn faceSplitSeams(@builtin(global_invocation_id)gid:vec3u){
  if(gid.x>=BFP1_SEAM_PORT_COUNT){return;}let port=bfp1SeamPort(gid.x);
  output[gid.x]=port.z;}
`;

type Variant = { readonly name: string;
  readonly steps: readonly { readonly entryPoint: string; readonly workgroups: number }[];
  readonly operationCount: number;
  readonly selectedCount: number };
const variants: Variant[] = [
  { name: "cell-dense", steps: [{ entryPoint: "enumerateDense",
    workgroups: Math.ceil(grid.cells.length / 64) }], operationCount: grid.cells.length,
    selectedCount: grid.cells.length },
  { name: "cell-current-tei2", steps: [{ entryPoint: "enumerateTEI2",
    workgroups: bti.layout.tileCapacity }], operationCount: bti.layout.tileCapacity * 64,
    selectedCount: grid.cells.length },
  { name: "cell-bti1", steps: [{ entryPoint: "enumerateBTI1",
    workgroups: bti.layout.tileCapacity }], operationCount: bti.layout.tileCapacity * 64,
    selectedCount: grid.cells.length },
  { name: "point-dense", steps: [{ entryPoint: "pointDense",
    workgroups: Math.ceil(pointCount / 64) }],
    operationCount: pointCount, selectedCount: pointCount },
  { name: "point-current-lod1", steps: [{ entryPoint: "pointLOD1",
    workgroups: Math.ceil(pointCount / 64) }], operationCount: pointCount, selectedCount: pointCount },
  { name: "point-bti1", steps: [{ entryPoint: "pointBTI1",
    workgroups: Math.ceil(pointCount / 64) }],
    operationCount: pointCount, selectedCount: pointCount },
  { name: "face-dense", steps: [{ entryPoint: "faceDense",
    workgroups: Math.ceil(grid.gradientRows.length / 64) }],
    operationCount: grid.gradientRows.length, selectedCount: grid.gradientRows.length },
  { name: "face-bti1", steps: [{ entryPoint: "faceBTI1",
    workgroups: bti.layout.tileCapacity }],
    operationCount: bti.layout.tileCapacity * 384, selectedCount: grid.gradientRows.length },
  { name: "face-split-bfp1", steps: [
    { entryPoint: "faceSplitInterior", workgroups: faceProgram.layout.interiorTileCount },
    { entryPoint: "faceSplitSeams", workgroups: faceProgram.layout.seamPacketCount },
  ], operationCount: 64 * (faceProgram.layout.interiorTileCount
      + faceProgram.layout.seamPacketCount), selectedCount: grid.gradientRows.length },
];

await acquireWebGPUExclusiveLock("dawn-benchmark",
  "tools/benchmark-sparse-cm12-brick-tile-services.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const backend = process.env.WEBGPU_BACKEND ?? "metal";
  const adapter = await dawn.create([`backend=${backend}`]).requestAdapter({
    powerPreference: "high-performance" });
  if (!adapter || !adapter.features.has("timestamp-query")) {
    throw new Error("BTI1 service benchmark requires timestamp-query");
  }
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const storage = (values: Uint32Array, label: string) => {
    const buffer = device.createBuffer({ label, size: Math.max(4, values.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, values.buffer as ArrayBuffer,
      values.byteOffset, values.byteLength);return buffer;
  };
  const buffers = [storage(bti.words, "BTI1"), storage(tei.words, "TEI2"),
    storage(logical.words, "LOD1"), storage(descriptors, "leaf descriptors"),
    storage(denseOwners, "dense owners"), storage(faceProgram.words, "BFP1")];
  const maximumOutputWords = Math.max(bti.layout.tileCapacity * 384,
    grid.gradientRows.length, pointCount);
  const output = device.createBuffer({ label: "BTI1 benchmark output",
    size: maximumOutputWords * 4, usage: GPUBufferUsage.STORAGE });
  try {
    const module = device.createShaderModule({ label: "BTI1 service benchmark", code: shader });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    if (errors.length) throw new Error(errors.map((message) => message.message).join("\n"));
    const layoutEntries: GPUBindGroupLayoutEntry[] = [0, 1, 2, 3, 4, 5].map(
      (binding) => ({ binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" } }),
    );
    layoutEntries.push({ binding: 6, visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" } });
    const bindGroupLayout = device.createBindGroupLayout({ label: "BTI1 service benchmark",
      entries: layoutEntries });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const group = device.createBindGroup({ layout: bindGroupLayout,
      entries: [...buffers.map((buffer, binding) =>
        ({ binding, resource: { buffer } })),
      { binding: 6, resource: { buffer: output } }] });
    const compiled = await Promise.all(variants.map(async (variant) => {
      const pipelines = await Promise.all(variant.steps.map((step) =>
        device.createComputePipelineAsync({ label: `${variant.name}/${step.entryPoint}`,
          layout: pipelineLayout, compute: { module, entryPoint: step.entryPoint } })));
      return { ...variant, pipelines, group };
    }));
    const measureOnce = async (variant: typeof compiled[number]): Promise<number> => {
      const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
      const resolveBuffer = device.createBuffer({ size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const readBuffer = device.createBuffer({ size: 16,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass({ timestampWrites: { querySet,
        beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
      pass.setBindGroup(0, variant.group);
      for (let repeat = 0; repeat < repeats; repeat += 1) {
        for (let step = 0; step < variant.steps.length; step += 1) {
          pass.setPipeline(variant.pipelines[step]!);
          pass.dispatchWorkgroups(variant.steps[step]!.workgroups);
        }
      }
      pass.end();encoder.resolveQuerySet(querySet, 0, 2, resolveBuffer, 0);
      encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, 16);
      device.queue.submit([encoder.finish()]);
      try {
        await readBuffer.mapAsync(GPUMapMode.READ);
        const timestamps = new BigUint64Array(readBuffer.getMappedRange().slice(0));
        if (timestamps[0] === 0n || timestamps[1] <= timestamps[0]!) {
          return Number.NaN;
        }
        return Number(timestamps[1]! - timestamps[0]!) / 1e6 / repeats;
      } finally {
        if (readBuffer.mapState === "mapped") readBuffer.unmap();
        querySet.destroy();resolveBuffer.destroy();readBuffer.destroy();
      }
    };
    const measure = async (variant: typeof compiled[number]): Promise<number> => {
      for (let attempt = 0; attempt < 5; attempt += 1) {
        const value = await measureOnce(variant);
        if (Number.isFinite(value)) return value;
      }
      throw new Error(`five invalid GPU timestamp samples for ${variant.name}`);
    };
    for (let warmup = 0; warmup < 3; warmup += 1)
      for (const variant of compiled) await measure(variant);
    const timings = new Map<string, number[]>(variants.map((variant) => [variant.name, []]));
    for (let sample = 0; sample < sampleCount; sample += 1) {
      const order = sample % 2 === 0 ? compiled : [...compiled].reverse();
      for (const variant of order) timings.get(variant.name)!.push(await measure(variant));
    }
    const median = (values: readonly number[]) => [...values]
      .sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
    const results = variants.map((variant) => {
      const samplesMs = timings.get(variant.name)!;const medianMs = median(samplesMs);
      return { name: variant.name,
        workgroups: variant.steps.reduce((sum, step) => sum + step.workgroups, 0),
        invokedLanes: variant.operationCount, selectedItems: variant.selectedCount,
        selectionRatio: variant.selectedCount / variant.operationCount,
        medianMilliseconds: medianMs,
        medianNanosecondsPerInvokedLane: medianMs * 1e6 / variant.operationCount,
        medianNanosecondsPerSelectedItem: medianMs * 1e6 / variant.selectedCount,
        samplesMilliseconds: samplesMs };
    });
    const byName = new Map(results.map((result) => [result.name, result]));
    const report = { schema: "sparse-cm12-brick-tile-gpu-services/v1",
      generatedAt: new Date().toISOString(), backend,
      adapter: (adapter as GPUAdapter & { readonly info?: GPUAdapterInfo }).info,
      fixture: { lattice, dimensions, leaves: atlas.bricks.length,
        cells: grid.cells.length, rows: grid.gradientRows.length,
        mixedSeamRows: grid.mixedSeamRowCount, points: pointCount,
        btiBytes: bti.layout.totalBytes, teiBytes: tei.layout.totalBytes,
        lodBytes: logical.layout.totalBytes }, repeats, sampleCount, results,
      ratios: {
        btiVsTEICell: byName.get("cell-bti1")!.medianMilliseconds
          / byName.get("cell-current-tei2")!.medianMilliseconds,
        btiVsLODPoint: byName.get("point-bti1")!.medianMilliseconds
          / byName.get("point-current-lod1")!.medianMilliseconds,
        btiVsDenseCell: byName.get("cell-bti1")!.medianMilliseconds
          / byName.get("cell-dense")!.medianMilliseconds,
        btiVsDensePoint: byName.get("point-bti1")!.medianMilliseconds
          / byName.get("point-dense")!.medianMilliseconds,
        btiVsDenseFace: byName.get("face-bti1")!.medianMilliseconds
          / byName.get("face-dense")!.medianMilliseconds,
        splitVsBroadBTIFace: byName.get("face-split-bfp1")!.medianMilliseconds
          / byName.get("face-bti1")!.medianMilliseconds,
        splitVsDenseFace: byName.get("face-split-bfp1")!.medianMilliseconds
          / byName.get("face-dense")!.medianMilliseconds,
      }, caveat: "Face-dense is a lower bound, not current ITR1. Production migration requires a matched live DFRM/ITR1 consumer A/B." };
    const json = `${JSON.stringify(report, null, 2)}\n`;
    if (outputPath) await writeFile(resolve(outputPath), json);process.stdout.write(json);
  } finally {
    for (const buffer of buffers) buffer.destroy();output.destroy();device.destroy();
  }
} finally {
  await releaseWebGPUExclusiveLock();
}
