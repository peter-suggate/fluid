#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickContainingCoordinate,
  sparseBrickKey,
  sparseBrickSpan,
  type SparseAdaptiveMassAtlas,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  SPARSE_CM12_LOGICAL_OWNER_HEADER,
  SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS,
  SPARSE_CM12_LOGICAL_OWNER_INVALID,
  SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS,
  createSparseCM12LogicalOwnerDirectory,
  sparseCM12LogicalOwnerAtKey,
  sparseCM12LogicalOwnerCellAtFine,
  sparseCM12LogicalOwnerHeaderValid,
  type SparseCM12LogicalOwnerDirectory,
  type SparseCM12LogicalOwnerRuntime,
} from "../lib/methods/adaptive-mass/sparse-cm12-logical-owner-directory";
import { createSparseCM12LogicalOwnerDirectoryWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-logical-owner-directory.wgsl";

const OUTPUT_WORDS = 10;
const CELL_STRIDE = 4096;

function fixture(): {
  readonly atlas: SparseAdaptiveMassAtlas;
  readonly directory: SparseCM12LogicalOwnerDirectory;
  readonly runtime: SparseCM12LogicalOwnerRuntime;
  readonly finestDimensions: readonly [number, number, number];
} {
  const finestDimensions = [64, 32, 16] as const;
  const logicalDimensions = [4, 2, 1] as const;
  const brick = (
    coordinate: readonly [number, number, number],
    resolution: SparseBrickResolution,
    spanBricks = 1,
  ): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey(coordinate, logicalDimensions),
    coordinate,
    spanBricks,
    resolution,
    density: new Float64Array(resolution ** 3),
    gamma: new Float64Array(resolution ** 3).fill(1),
  });
  // The ordinary brick at [1,0,0] lies inside the span-two macro. The old
  // lookup finds span one first there and the macro in its other three logical
  // coordinates; the dense directory must reproduce that exact behavior.
  const atlas = createSparseAdaptiveMassAtlas(finestDimensions, [
    brick([0, 0, 0], 4, 2),
    brick([1, 0, 0], 8),
    brick([2, 0, 0], 16),
    brick([3, 1, 0], 8),
  ], 7, 16);
  const active = [true, true, false, true] as const;
  const runtime: SparseCM12LogicalOwnerRuntime = {
    brickActive: (id) => active[id] ?? false,
    acceptedBrickResolution: (id) => atlas.bricks[id]!.resolution,
    templateBrickCellRange: (id, resolution) =>
      [CELL_STRIDE * id, resolution ** 3],
    cellResolution: (cell) => atlas.bricks[Math.floor(cell / CELL_STRIDE)]!.resolution,
    cellOpenVolume: () => 1,
  };
  return {
    atlas,
    directory: createSparseCM12LogicalOwnerDirectory(atlas),
    runtime,
    finestDimensions,
  };
}

function legacyCellAtFine(
  atlas: SparseAdaptiveMassAtlas,
  position: readonly [number, number, number],
  runtime: SparseCM12LogicalOwnerRuntime,
  requireActiveAndOpen: boolean,
): number | undefined {
  if (position.some((value, axis) => value < 0 || value >= atlas.dimensions[axis])) {
    return undefined;
  }
  const logical = position.map((value) => Math.floor(value / atlas.brickFineResolution)) as
    [number, number, number];
  const source = sparseBrickContainingCoordinate(atlas, logical);
  if (!source) return undefined;
  const brick = atlas.bricks.indexOf(source);
  if (requireActiveAndOpen && !runtime.brickActive(brick)) return undefined;
  const resolution = runtime.acceptedBrickResolution(brick);
  const range = runtime.templateBrickCellRange(brick, resolution);
  const span = sparseBrickSpan(source);
  const scale = atlas.brickFineResolution * span / resolution;
  const origin = source.coordinate.map((value) => value * atlas.brickFineResolution) as
    [number, number, number];
  const local = position.map((value, axis) =>
    Math.floor((value - origin[axis]!) / scale)) as [number, number, number];
  const valid = atlas.dimensions.map((value, axis) => Math.floor(Math.min(
    value - origin[axis]! + scale - 1,
    atlas.brickFineResolution * span,
  ) / scale)) as [number, number, number];
  const offset = local[0] + valid[0] * (local[1] + valid[1] * local[2]);
  if (offset < 0 || offset >= range[1]) return undefined;
  const cell = range[0] + offset;
  if (requireActiveAndOpen
    && (runtime.cellResolution?.(cell) !== resolution
      || !(runtime.cellOpenVolume?.(cell) ?? 0 > 1e-8))) return undefined;
  return cell;
}

function shaderSource(directory: SparseCM12LogicalOwnerDirectory): string {
  const helpers = createSparseCM12LogicalOwnerDirectoryWGSL({
    layout: directory.layout,
    directoryName: "ownerDirectory",
    brickActiveFunction: "fixtureBrickActive",
    acceptedBrickResolutionFunction: "fixtureAcceptedResolution",
    templateBrickCellRangeFunction: "fixtureCellRange",
    cellResolutionFunction: "fixtureCellResolution",
    cellOpenVolumeFunction: "fixtureCellOpenVolume",
  });
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read> ownerDirectory:array<u32>;
@group(0) @binding(1) var<storage,read> fixtureRuntime:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;

fn fixtureBrickActive(brick:u32)->bool{
  return brick<${directory.layout.residentBrickCount}u&&fixtureRuntime[brick]!=0u;
}
fn fixtureAcceptedResolution(brick:u32)->u32{
  if(brick>=${directory.layout.residentBrickCount}u){return 0u;}
  return fixtureRuntime[${directory.layout.residentBrickCount}u+brick];
}
fn fixtureCellRange(brick:u32,resolution:u32)->vec2u{
  return vec2u(${CELL_STRIDE}u*brick,resolution*resolution*resolution);
}
fn fixtureCellResolution(cell:u32)->u32{
  return fixtureAcceptedResolution(cell/${CELL_STRIDE}u);
}
fn fixtureCellOpenVolume(cell:u32)->f32{_=cell;return 1.0;}

${helpers}

@compute @workgroup_size(64)
fn checkLogicalOwners(@builtin(global_invocation_id) gid:vec3u){
  let key=gid.x;if(key>=cm12LogicalOwnerCount){return;}
  let owner=cm12LogicalOwnerRecordAtKey(key);
  let range=cm12LogicalOwnerCellRangeAtKey(key);
  let logical=cm12LogicalOwnerCoordinate(key);
  let q=vec3i(logical*cm12LogicalOwnerBrickFine+vec3u(3u,2u,1u));
  let compact=cm12LogicalOwnerCellAtFine(q,vec3u(64u,32u,16u));
  let at=${OUTPUT_WORDS}u*key;
  output[at]=select(0u,1u,cm12LogicalOwnerHeaderValid());
  output[at+1u]=owner.brick;
  output[at+2u]=owner.spanBricks;
  output[at+3u]=owner.originKey;
  output[at+4u]=select(0u,1u,cm12LogicalOwnerActiveAtKey(key));
  output[at+5u]=cm12LogicalOwnerRungAtKey(key);
  output[at+6u]=range.x;output[at+7u]=range.y;
  output[at+8u]=compact.x;
  output[at+9u]=cm12LogicalActiveCellAtFine(q,vec3u(64u,32u,16u));
}
`;
}

function assertCPUFixture(
  atlas: SparseAdaptiveMassAtlas,
  directory: SparseCM12LogicalOwnerDirectory,
  runtime: SparseCM12LogicalOwnerRuntime,
): void {
  if (directory.words.length !== SPARSE_CM12_LOGICAL_OWNER_HEADER_WORDS
    + SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS * directory.layout.logicalBrickCount) {
    throw new Error("LOD1 is not exactly an eight-byte logical-brick record");
  }
  const expected = [0, 1, 2, undefined, 0, 0, undefined, 3] as const;
  expected.forEach((brick, key) => {
    const owner = sparseCM12LogicalOwnerAtKey(directory, key);
    if (owner?.brick !== brick) {
      throw new Error(`CPU logical owner ${key}: ${owner?.brick} != ${brick}`);
    }
    const x = key % directory.layout.logicalBrickDimensions[0];
    const y = Math.floor(key / directory.layout.logicalBrickDimensions[0])
      % directory.layout.logicalBrickDimensions[1];
    const z = Math.floor(key
      / (directory.layout.logicalBrickDimensions[0]
        * directory.layout.logicalBrickDimensions[1]));
    const q = [16 * x + 3, 16 * y + 2, 16 * z + 1] as const;
    for (const active of [false, true]) {
      const direct = sparseCM12LogicalOwnerCellAtFine(
        directory, q, atlas.dimensions, runtime, active,
      )?.cell;
      const legacy = legacyCellAtFine(atlas, q, runtime, active);
      if (direct !== legacy) {
        throw new Error(`CPU fine owner ${key}/${active}: ${direct} != legacy ${legacy}`);
      }
    }
  });
  const corruptWords = directory.words.slice();
  corruptWords[SPARSE_CM12_LOGICAL_OWNER_HEADER.magic] = 0;
  const corrupt = { layout: directory.layout, words: corruptWords };
  if (sparseCM12LogicalOwnerHeaderValid(corrupt)
    || sparseCM12LogicalOwnerAtKey(corrupt, 0) !== undefined) {
    throw new Error("corrupt CPU directory did not fail closed");
  }
}

async function evaluate(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  directory: SparseCM12LogicalOwnerDirectory,
  runtime: SparseCM12LogicalOwnerRuntime,
  words: Uint32Array,
): Promise<Uint32Array> {
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const ownerBuffer = device.createBuffer({
    size: words.byteLength,
    usage: storage,
  });
  device.queue.writeBuffer(ownerBuffer, 0, words.buffer as ArrayBuffer,
    words.byteOffset, words.byteLength);
  const runtimeWords = new Uint32Array(2 * directory.layout.residentBrickCount);
  for (let brick = 0; brick < directory.layout.residentBrickCount; brick += 1) {
    runtimeWords[brick] = runtime.brickActive(brick) ? 1 : 0;
    runtimeWords[directory.layout.residentBrickCount + brick] =
      runtime.acceptedBrickResolution(brick);
  }
  const runtimeBuffer = device.createBuffer({ size: runtimeWords.byteLength, usage: storage });
  device.queue.writeBuffer(runtimeBuffer, 0, runtimeWords.buffer as ArrayBuffer,
    runtimeWords.byteOffset, runtimeWords.byteLength);
  const outputBytes = 4 * OUTPUT_WORDS * directory.layout.logicalBrickCount;
  const output = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: ownerBuffer } },
      { binding: 1, resource: { buffer: runtimeBuffer } },
      { binding: 2, resource: { buffer: output } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);pass.setBindGroup(0, bindGroup);pass.dispatchWorkgroups(1);pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  ownerBuffer.destroy();runtimeBuffer.destroy();output.destroy();readback.destroy();
  return result;
}

function assertValidGPU(
  result: Uint32Array,
  atlas: SparseAdaptiveMassAtlas,
  directory: SparseCM12LogicalOwnerDirectory,
  runtime: SparseCM12LogicalOwnerRuntime,
  finestDimensions: readonly [number, number, number],
): void {
  for (let key = 0; key < directory.layout.logicalBrickCount; key += 1) {
    const at = OUTPUT_WORDS * key;
    const owner = sparseCM12LogicalOwnerAtKey(directory, key);
    const logicalX = key % directory.layout.logicalBrickDimensions[0];
    const logicalY = Math.floor(key / directory.layout.logicalBrickDimensions[0])
      % directory.layout.logicalBrickDimensions[1];
    const logicalZ = Math.floor(key
      / (directory.layout.logicalBrickDimensions[0]
        * directory.layout.logicalBrickDimensions[1]));
    const q = [16 * logicalX + 3, 16 * logicalY + 2, 16 * logicalZ + 1] as const;
    const compact = legacyCellAtFine(atlas, q, runtime, false);
    const active = legacyCellAtFine(atlas, q, runtime, true);
    const expected = owner ? [
      1,
      owner.brick,
      owner.spanBricks,
      owner.originKey,
      runtime.brickActive(owner.brick) ? 1 : 0,
      runtime.acceptedBrickResolution(owner.brick),
      CELL_STRIDE * owner.brick,
      runtime.acceptedBrickResolution(owner.brick) ** 3,
      compact ?? SPARSE_CM12_LOGICAL_OWNER_INVALID,
      active ?? SPARSE_CM12_LOGICAL_OWNER_INVALID,
    ] : [
      1, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID,
      0, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0,
      SPARSE_CM12_LOGICAL_OWNER_INVALID, SPARSE_CM12_LOGICAL_OWNER_INVALID,
    ];
    for (let word = 0; word < OUTPUT_WORDS; word += 1) {
      if (result[at + word] !== expected[word]) {
        throw new Error(`GPU logical owner ${key} word ${word}: ${result[at + word]} != ${expected[word]}`);
      }
    }
  }
}

function assertFailClosedGPU(result: Uint32Array, logicalCount: number): void {
  for (let key = 0; key < logicalCount; key += 1) {
    const at = OUTPUT_WORDS * key;
    const expected = [
      0, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID,
      0, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0,
      SPARSE_CM12_LOGICAL_OWNER_INVALID, SPARSE_CM12_LOGICAL_OWNER_INVALID,
    ];
    for (let word = 0; word < OUTPUT_WORDS; word += 1) {
      if (result[at + word] !== expected[word]) {
        throw new Error(`corrupt GPU directory did not fail closed at ${key}/${word}`);
      }
    }
  }
}

function assertRecordFailClosedGPU(
  result: Uint32Array,
  valid: Uint32Array,
  logicalCount: number,
  rejectedKey: number,
): void {
  const rejected = [
    1, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID,
    0, 0, SPARSE_CM12_LOGICAL_OWNER_INVALID, 0,
    SPARSE_CM12_LOGICAL_OWNER_INVALID, SPARSE_CM12_LOGICAL_OWNER_INVALID,
  ];
  for (let key = 0; key < logicalCount; key += 1) {
    const at = OUTPUT_WORDS * key;
    for (let word = 0; word < OUTPUT_WORDS; word += 1) {
      const expected = key === rejectedKey ? rejected[word] : valid[at + word];
      if (result[at + word] !== expected) {
        throw new Error(`record-local fail-closed mismatch at ${key}/${word}`);
      }
    }
  }
}

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const { atlas, directory, runtime, finestDimensions } = fixture();
  assertCPUFixture(atlas, directory, runtime);
  const source = shaderSource(directory);
  if (process.argv.includes("--emit-wgsl")) {
    process.stdout.write(source);return;
  }
  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-logical-owner-directory");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    device.pushErrorScope("validation");
    const module = device.createShaderModule({ label: "LOD1 logical-owner checker", code: source });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((message) => message.type === "error");
    for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
    if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
    const pipeline = await device.createComputePipelineAsync({
      layout: "auto", compute: { module, entryPoint: "checkLogicalOwners" },
    });
    const scope = await device.popErrorScope();
    if (scope) throw new Error(scope.message);
    const valid = await evaluate(device, pipeline, directory, runtime, directory.words);
    assertValidGPU(valid, atlas, directory, runtime, finestDimensions);
    const corrupt = directory.words.slice();
    corrupt[SPARSE_CM12_LOGICAL_OWNER_HEADER.magic] = 0;
    const rejected = await evaluate(device, pipeline, directory, runtime, corrupt);
    assertFailClosedGPU(rejected, directory.layout.logicalBrickCount);
    const corruptRecord = directory.words.slice();
    const rejectedKey = 4;
    corruptRecord[directory.layout.recordBaseWords
      + SPARSE_CM12_LOGICAL_OWNER_RECORD_WORDS * rejectedKey] =
      (directory.layout.residentBrickCount << 5) | 1;
    const locallyRejected = await evaluate(
      device, pipeline, directory, runtime, corruptRecord,
    );
    assertRecordFailClosedGPU(locallyRejected, valid,
      directory.layout.logicalBrickCount, rejectedKey);
    console.log(`Sparse CM12 LOD1: valid B16/P16 eight-byte directory (${directory.layout.logicalBrickCount} logical bricks; macro override and fail-closed receipts passed)`);
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

void main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
