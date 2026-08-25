#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { buildSparseAtlasCompositeGrid, type SparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  SPARSE_CM12_HOT_TOPOLOGY_CELL,
  SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_HEADER,
  SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS,
  SPARSE_CM12_HOT_TOPOLOGY_INVALID,
  SPARSE_CM12_HOT_TOPOLOGY_ROW,
  SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS,
  corruptSparseCM12HotTopologyWord,
  createSparseCM12HotTopology,
  sparseCM12HotTopologyByteMap,
  sparseCM12HotTopologyHeaderValid,
  validateSparseCM12HotTopology,
  type SparseCM12HotTopology,
} from "../lib/methods/adaptive-mass/sparse-cm12-hot-topology";
import { createSparseCM12HotTopologyWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-hot-topology.wgsl";

const OUTPUT_WORDS = 40;

function fixture(): {
  readonly topology: SparseCM12HotTopology;
  readonly commonRow: number;
  readonly variableRow: number;
  readonly probeCell: number;
  readonly grid: SparseAtlasCompositeGrid;
} {
  const dimensions = [32, 16, 16] as const;
  const logical = [2, 1, 1] as const;
  const brick = (coordinate: readonly [number, number, number],
    resolution: SparseBrickResolution): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey(coordinate, logical), coordinate, resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas(dimensions, [
    brick([0, 0, 0], 8), brick([1, 0, 0], 16),
  ], 19, 16);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const commonRow = grid.gradientRows.findIndex((row) => row.terms.length === 2);
  const variableRow = grid.gradientRows.findIndex((row) =>
    row.kind === "mixed-seam" && row.terms.length !== 2);
  if (commonRow < 0 || variableRow < 0) {
    throw new Error("fixture did not expose both common and mixed variable rows");
  }
  const topology = createSparseCM12HotTopology(grid);
  validateSparseCM12HotTopology(topology, grid);
  const probeCell = grid.gradientRows[variableRow]!.terms[0]!.cellId;
  const map = sparseCM12HotTopologyByteMap(topology.layout);
  if (map.find((entry) => entry.name === "literal cells")?.bytesPerRecord !== 32
    || map.find((entry) => entry.name === "tagged rows")?.bytesPerRecord !== 64
    || map.find((entry) => entry.name === "cell incidences")?.bytesPerRecord !== 8
    || map.find((entry) => entry.name === "pressure/extrapolation edges")?.bytesPerRecord !== 16) {
    throw new Error("HTP1 byte map drifted from the physical ABI");
  }
  return { topology, commonRow, variableRow, probeCell, grid };
}

function shaderSource(
  topology: SparseCM12HotTopology,
  commonRow: number,
  variableRow: number,
  probeCell: number,
): string {
  const helpers = createSparseCM12HotTopologyWGSL({
    layout: topology.layout, arenaName: "arena",
    brickActiveFunction: "fixtureBrickActive",
    acceptedBrickResolutionFunction: "fixtureAcceptedResolution",
    templateBrickCellRangeFunction: "fixtureCellRange",
    cellResolutionFunction: "fixtureCellResolution",
    cellOpenVolumeFunction: "fixtureCellOpenVolume",
  });
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read> arena:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;

fn fixtureBrickActive(brick:u32)->bool{return brick<2u;}
fn fixtureAcceptedResolution(brick:u32)->u32{return select(8u,16u,brick==1u);}
fn fixtureCellRange(brick:u32,resolution:u32)->vec2u{
  _=resolution;return select(vec2u(0u,512u),vec2u(512u,4096u),brick==1u);
}
fn fixtureCellResolution(cell:u32)->u32{
  return select(8u,16u,cell>=512u);
}
fn fixtureCellOpenVolume(cell:u32)->f32{_=cell;return 1.0;}

${helpers}

@compute @workgroup_size(1)
fn checkHotTopology(){
  let cell=cm12HotCellAt(${probeCell}u);
  let commonRowId=${commonRow}u;let variable=${variableRow}u;
  let variableCount=cm12HotRowTermCount(variable);
  let incidenceRange=cm12HotIncidenceRange(${probeCell}u);
  let incidence=cm12HotIncidence(incidenceRange.x);
  let edgeRange=cm12HotDirectedEdgeRange(${probeCell}u);
  let edge=cm12HotDirectedEdge(edgeRange.x);
  let owner=cm12LogicalOwnerRecordAtKey(0u);
  output[0]=select(0u,1u,cm12HotHeaderValid());
  output[1]=select(0u,1u,cm12LogicalOwnerHeaderValid());
  output[2]=cell.valid;output[3]=bitcast<u32>(cell.center.x);
  output[4]=bitcast<u32>(cell.volume);output[5]=bitcast<u32>(cell.widths.x);
  output[6]=cell.brick;output[7]=cell.resolution;
  output[8]=select(0u,1u,cm12HotRowValid(commonRowId));
  output[9]=cm12HotRowTag(commonRowId);output[10]=cm12HotRowTermCount(commonRowId);
  output[11]=cm12HotRowTermCell(commonRowId,0u);
  output[12]=bitcast<u32>(cm12HotRowTermCoefficient(commonRowId,0u));
  output[13]=cm12HotRowTermCell(commonRowId,1u);
  output[14]=bitcast<u32>(cm12HotRowTermCoefficient(commonRowId,1u));
  output[15]=select(0u,1u,cm12HotRowRequirementsMatch(commonRowId));
  output[16]=select(0u,1u,cm12HotRowValid(variable));
  output[17]=cm12HotRowTag(variable);output[18]=variableCount;
  output[19]=cm12HotRowTermCell(variable,0u);
  output[20]=bitcast<u32>(cm12HotRowTermCoefficient(variable,0u));
  output[21]=cm12HotRowTermCell(variable,variableCount-1u);
  output[22]=bitcast<u32>(cm12HotRowTermCoefficient(variable,variableCount-1u));
  output[23]=cm12HotRowRequirement(variable,0u);
  output[24]=incidenceRange.x;output[25]=incidenceRange.y;
  output[26]=incidence.x;output[27]=incidence.y;
  output[28]=bitcast<u32>(cm12HotRowTermCoefficient(incidence.x,incidence.y));
  output[29]=cm12HotRowTag(incidence.x);
  output[30]=edgeRange.x;output[31]=edgeRange.y;
  output[32]=edge.x;output[33]=edge.y;output[34]=edge.z;output[35]=edge.w;
  output[36]=owner.brick;output[37]=owner.spanBricks;
  output[38]=cm12HotTotalWords;output[39]=select(0u,1u,cm12HotRowIsCommonTwoTerm(variable));
}
`;
}

async function evaluate(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  topologyWords: Uint32Array,
): Promise<Uint32Array> {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const arenaBuffer = device.createBuffer({ size: topologyWords.byteLength, usage: storage });
  const arenaUpload = new Uint32Array(topologyWords);
  device.queue.writeBuffer(arenaBuffer, 0, arenaUpload.buffer as ArrayBuffer,
    arenaUpload.byteOffset, arenaUpload.byteLength);
  const outputBytes = OUTPUT_WORDS * 4;
  const output = device.createBuffer({ size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: arenaBuffer } },
      { binding: 2, resource: { buffer: output } }],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0, bindGroup);pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();arenaBuffer.destroy();output.destroy();readback.destroy();
  return result;
}

function expectedValid(
  topology: SparseCM12HotTopology,
  commonRow: number,
  variableRow: number,
  probeCell: number,
): Uint32Array {
  const { layout: l, words } = topology;
  const cellAt = l.cellBaseWords + probeCell * SPARSE_CM12_HOT_TOPOLOGY_CELL_WORDS;
  const commonAt = l.rowBaseWords + commonRow * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS;
  const variableAt = l.rowBaseWords + variableRow * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS;
  const commonTag = words[commonAt]!, variableTag = words[variableAt]!;
  const variableCount = variableTag & 0xffff;
  const variableFirst = words[variableAt + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm]!;
  const variableTermAt = l.variableTermBaseWords + 2 * variableFirst;
  const variableLastAt = variableTermAt + 2 * (variableCount - 1);
  const incidenceFirst = words[l.incidenceOffsetBaseWords + probeCell]!;
  const incidenceEnd = words[l.incidenceOffsetBaseWords + probeCell + 1]!;
  const incidenceAt = l.incidenceBaseWords
    + incidenceFirst * SPARSE_CM12_HOT_TOPOLOGY_INCIDENCE_WORDS;
  const edgeFirst = words[l.directedEdgeOffsetBaseWords + probeCell]!;
  const edgeEnd = words[l.directedEdgeOffsetBaseWords + probeCell + 1]!;
  const edgeAt = l.directedEdgeBaseWords + edgeFirst * 4;
  const metadata = words[cellAt + SPARSE_CM12_HOT_TOPOLOGY_CELL.brickAndResolution]!;
  return Uint32Array.from([
    1, 1, 1, words[cellAt + SPARSE_CM12_HOT_TOPOLOGY_CELL.centerX]!,
    words[cellAt + SPARSE_CM12_HOT_TOPOLOGY_CELL.volume]!,
    words[cellAt + SPARSE_CM12_HOT_TOPOLOGY_CELL.widthX]!, metadata >>> 5, metadata & 31,
    1, commonTag, 2, words[commonAt + 1]!, words[commonAt + 2]!,
    words[commonAt + 3]!, words[commonAt + 4]!, 1,
    1, variableTag, variableCount, words[variableTermAt]!, words[variableTermAt + 1]!,
    words[variableLastAt]!, words[variableLastAt + 1]!,
    words[l.requirementBaseWords
      + words[variableAt + SPARSE_CM12_HOT_TOPOLOGY_ROW.firstRequirement]!]!,
    incidenceFirst, incidenceEnd - incidenceFirst,
    words[incidenceAt]!, words[incidenceAt + 1]!,
    bitcastF32Word(topology, words[incidenceAt]!, words[incidenceAt + 1]!),
    words[l.rowBaseWords + words[incidenceAt]! * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS]!,
    edgeFirst, edgeEnd - edgeFirst,
    words[edgeAt]!, words[edgeAt + 1]!, words[edgeAt + 2]!, words[edgeAt + 3]!,
    0, 1, l.totalWords, 0,
  ]);
}

function bitcastF32Word(
  topology: SparseCM12HotTopology,
  row: number,
  ordinal: number,
): number {
  const { layout: l, words } = topology;
  const at = l.rowBaseWords + row * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS;
  const tag = words[at]!;
  if ((tag & (1 << 20)) !== 0) return words[at + (ordinal === 0 ? 2 : 4)]!;
  const first = words[at + SPARSE_CM12_HOT_TOPOLOGY_ROW.cell0OrFirstTerm]!;
  return words[l.variableTermBaseWords + 2 * (first + ordinal) + 1]!;
}

function assertWords(actual: Uint32Array, expected: Uint32Array, label: string): void {
  if (actual.length !== expected.length) throw new Error(`${label} length mismatch`);
  actual.forEach((value, index) => {
    if (value !== expected[index]) throw new Error(`${label} word ${index}: ${value} != ${expected[index]}`);
  });
}

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const { topology, commonRow, variableRow, probeCell, grid } = fixture();
  const source = shaderSource(topology, commonRow, variableRow, probeCell);
  if (process.argv.includes("--emit-wgsl")) { process.stdout.write(source); return; }
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-hot-topology");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU; globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "HTP1 hot-topology checker", code: source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    errors.forEach((error) => console.error(`${error.lineNum}:${error.linePos} ${error.message}`));
    if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "checkHotTopology" },
    });
    const scope = await device.popErrorScope();
    if (scope) throw new Error(scope.message);
    const valid = await evaluate(device, pipeline, topology.words);
    assertWords(valid, expectedValid(topology, commonRow, variableRow, probeCell), "valid HTP1");

    const corruptHeader = corruptSparseCM12HotTopologyWord(topology,
      topology.layout.headerBaseWords + SPARSE_CM12_HOT_TOPOLOGY_HEADER.magic, 0);
    if (sparseCM12HotTopologyHeaderValid(corruptHeader)) throw new Error("CPU header corruption accepted");
    const globalRejected = await evaluate(device, pipeline, corruptHeader.words);
    if (globalRejected[0] !== 0 || globalRejected[2] !== 0
      || globalRejected[8] !== 0 || globalRejected[16] !== 0
      || globalRejected[1] !== 1) throw new Error("global HTP1 corruption did not fail closed");

    const corruptVariable = corruptSparseCM12HotTopologyWord(topology,
      topology.layout.rowBaseWords + variableRow * SPARSE_CM12_HOT_TOPOLOGY_ROW_WORDS
        + SPARSE_CM12_HOT_TOPOLOGY_ROW.identity, SPARSE_CM12_HOT_TOPOLOGY_INVALID);
    const localRejected = await evaluate(device, pipeline, corruptVariable.words);
    if (localRejected[0] !== 1 || localRejected[8] !== 1 || localRejected[16] !== 0
      || localRejected[19] !== SPARSE_CM12_HOT_TOPOLOGY_INVALID
      || localRejected[20] !== 0) throw new Error("row-local HTP1 corruption escaped its blast radius");
    let validatorRejected = false;
    try { validateSparseCM12HotTopology(corruptVariable, grid); }
    catch { validatorRejected = true; }
    if (!validatorRejected) throw new Error("exhaustive validator accepted corrupt topology");

    console.log(`Sparse CM12 HTP1: B16/P16 ${topology.layout.totalBytes} bytes; ${topology.layout.cellCount} cells, ${topology.layout.rowCount} rows, ${topology.layout.directedEdgeCount} directed edges; common/mixed and fail-closed receipts passed`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

void main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
