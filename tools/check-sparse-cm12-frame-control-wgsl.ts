#!/usr/bin/env node
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  SPARSE_CM12_FRAME_CONTROL_COVERAGE,
  SPARSE_CM12_FRAME_CONTROL_FAULT,
  SPARSE_CM12_FRAME_CONTROL_FAMILY,
  SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT,
  SPARSE_CM12_FRAME_CONTROL_HEADER,
  SPARSE_CM12_FRAME_CONTROL_PHASE,
  createSparseCM12FrameControl,
  corruptSparseCM12FrameControlWord,
  sparseCM12FrameControlByteMap,
  sparseCM12FrameControlHeaderValid,
  type SparseCM12FrameControl,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12FrameControlWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control.wgsl";

const SNAPSHOT_WORDS = 20 + 3 * SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT;
const MODE_NORMAL = 0;
const MODE_BODY_OVERFLOW = 1;
const MODE_INVALIDATE_BEFORE_SEAL = 2;
const MODE_INVALIDATE_AFTER_SEAL = 3;
const MODE_MISSING_BODY_EVIDENCE = 4;

type Commands = readonly [
  mode: number, bodyCount: number, reserved: number,
  scalarD4: number, faceD4: number, cause: number, owner: number, outputMask: number,
];

function shaderSource(control: SparseCM12FrameControl): string {
  const helpers = createSparseCM12FrameControlWGSL({
    layout: control.layout,
    controlName: "frameControl",
  });
  return /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> frameControl:array<atomic<u32>>;
@group(0) @binding(1) var<storage,read> commands:array<u32>;
@group(0) @binding(2) var<storage,read_write> output:array<u32>;

${helpers}

fn checkFCHeader(word:u32)->u32{
  return cm12FCLoad(${control.layout.baseWords}u+word);
}

fn snapshot(){
  output[0]=select(0u,1u,cm12FCHeaderValid());
  output[1]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.phase}u);
  output[2]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.acceptedGeneration}u);
  output[3]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.candidateGeneration}u);
  output[4]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.scalarParity}u);
  output[5]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.faceParity}u);
  output[6]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.bodyGeneration}u);
  output[7]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.bodyCount}u);
  output[8]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.reserved20}u);
  output[9]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.reserved21}u);
  output[10]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.d4Generation}u);
  output[11]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.scalarD4Authority}u);
  output[12]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.faceD4Authority}u);
  output[13]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationCause}u);
  output[14]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.d4InvalidationOwner}u);
  output[15]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.coverage}u);
  output[16]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.fault}u);
  output[17]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.firstFaultOwner}u);
  output[18]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.sealedGeneration}u);
  output[19]=checkFCHeader(${SPARSE_CM12_FRAME_CONTROL_HEADER.committedFrames}u);
  for(var family=0u;family<${SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT}u;family+=1u){
    let source=cm12FCFamilyBase(family);let destination=20u+3u*family;
    output[destination]=cm12FCLoad(source);
    output[destination+1u]=cm12FCLoad(source+1u);
    output[destination+2u]=cm12FCLoad(source+2u);
  }
}

@compute @workgroup_size(1)
fn checkFrameControl(){
  if(cm12FCBegin()){
    if(commands[0]==${MODE_BODY_OVERFLOW}u){
      _=cm12FCPublishBody(commands[1]);
    }else{
      var bodyOK=true;
      if(commands[0]!=${MODE_MISSING_BODY_EVIDENCE}u){
        bodyOK=cm12FCPublishBody(commands[1]);
      }
      let boundaryOK=true;
      var d4OK=false;
      if(commands[0]==${MODE_INVALIDATE_BEFORE_SEAL}u){
        d4OK=cm12FCInvalidateD4(commands[5],commands[6]);
      }else{
        d4OK=cm12FCPublishD4(commands[3]!=0u,commands[4]!=0u);
      }
      if(bodyOK&&boundaryOK&&d4OK&&cm12FCSeal()){
        if(commands[0]==${MODE_INVALIDATE_AFTER_SEAL}u){
          _=cm12FCInvalidateD4(commands[5],commands[6]);
        }
        _=cm12FCPublishOutput(commands[7]);
        _=cm12FCCommit();
      }
    }
  }
  snapshot();
}
`;
}

function familyX(snapshot: Uint32Array, family: number): number {
  return snapshot[20 + 3 * family]!;
}

function assertEqual(actual: number, expected: number, label: string): void {
  if (actual !== expected) throw new Error(`${label}: ${actual} != ${expected}`);
}

function assertTripletShape(snapshot: Uint32Array, label: string): void {
  for (let family = 0; family < SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT; family += 1) {
    assertEqual(snapshot[20 + 3 * family + 1]!, 1, `${label} family ${family} y`);
    assertEqual(snapshot[20 + 3 * family + 2]!, 1, `${label} family ${family} z`);
  }
}

function assertCommitted(
  snapshot: Uint32Array,
  label: string,
  expected: { readonly scalarD4: number; readonly faceD4: number;
    readonly solid: number; readonly bodyLive: number },
): void {
  assertEqual(snapshot[0]!, 1, `${label} header`);
  assertEqual(snapshot[1]!, SPARSE_CM12_FRAME_CONTROL_PHASE.accepted, `${label} phase`);
  assertEqual(snapshot[2]!, 6, `${label} accepted generation`);
  assertEqual(snapshot[3]!, 6, `${label} candidate generation`);
  assertEqual(snapshot[4]!, 1, `${label} scalar parity`);
  assertEqual(snapshot[5]!, 1, `${label} face parity`);
  assertEqual(snapshot[16]!, SPARSE_CM12_FRAME_CONTROL_FAULT.none, `${label} fault`);
  assertEqual(snapshot[19]!, 1, `${label} committed frames`);
  const cellGroups = 7;
  const rowGroups = 11;
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Work),
    expected.scalarD4 ? cellGroups : 0, `${label} scalar D4 work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.scalarD4Bypass),
    expected.scalarD4 ? 0 : 1, `${label} scalar D4 bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Work),
    expected.faceD4 ? cellGroups : 0, `${label} face D4 work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.faceD4Bypass),
    expected.faceD4 ? 0 : 1, `${label} face D4 bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellWork),
    expected.solid ? cellGroups : 0, `${label} solid cell work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.solidCellBypass),
    expected.solid ? 0 : 1, `${label} solid cell bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.solidRowWork),
    expected.solid ? rowGroups : 0, `${label} solid row work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.solidRowBypass),
    expected.solid ? 0 : 1, `${label} solid row bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyWork),
    expected.bodyLive ? cellGroups : 0, `${label} body work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyBypass),
    expected.bodyLive ? 0 : 1, `${label} body bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowWork),
    expected.bodyLive ? rowGroups : 0, `${label} body row work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.bodyRowBypass),
    expected.bodyLive ? 0 : 1, `${label} body row bypass`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.frameWork), 1,
    `${label} frame work`);
  assertEqual(familyX(snapshot, SPARSE_CM12_FRAME_CONTROL_FAMILY.frameBlocked), 0,
    `${label} frame blocked`);
  assertTripletShape(snapshot, label);
}

function assertFault(
  snapshot: Uint32Array,
  label: string,
  fault: number,
  headerValid = true,
): void {
  assertEqual(snapshot[0]!, headerValid ? 1 : 0, `${label} header`);
  assertEqual(snapshot[1]!, SPARSE_CM12_FRAME_CONTROL_PHASE.fault, `${label} phase`);
  assertEqual(snapshot[2]!, 5, `${label} accepted generation`);
  assertEqual(snapshot[4]!, 0, `${label} scalar parity`);
  assertEqual(snapshot[5]!, 0, `${label} face parity`);
  assertEqual(snapshot[16]!, fault, `${label} fault`);
  assertEqual(snapshot[19]!, 0, `${label} committed frames`);
  for (let family = 0; family < SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT; family += 1) {
    const expected = family === SPARSE_CM12_FRAME_CONTROL_FAMILY.frameBlocked ? 1 : 0;
    assertEqual(familyX(snapshot, family), expected, `${label} family ${family} x`);
  }
  assertTripletShape(snapshot, label);
}

async function evaluate(
  device: GPUDevice,
  pipeline: GPUComputePipeline,
  control: SparseCM12FrameControl,
  commands: Commands,
): Promise<Uint32Array> {
  const controlBuffer = device.createBuffer({
    size: control.words.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const commandWords = new Uint32Array(commands);
  const commandBuffer = device.createBuffer({
    size: commandWords.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  const outputBytes = 4 * SNAPSHOT_WORDS;
  const outputBuffer = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    size: outputBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  device.queue.writeBuffer(controlBuffer, 0, control.words.buffer as ArrayBuffer,
    control.words.byteOffset, control.words.byteLength);
  device.queue.writeBuffer(commandBuffer, 0, commandWords.buffer as ArrayBuffer,
    commandWords.byteOffset, commandWords.byteLength);
  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: controlBuffer } },
      { binding: 1, resource: { buffer: commandBuffer } },
      { binding: 2, resource: { buffer: outputBuffer } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outputBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const result = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  controlBuffer.destroy(); commandBuffer.destroy(); outputBuffer.destroy(); readback.destroy();
  return result;
}

async function createPipeline(
  device: GPUDevice,
  control: SparseCM12FrameControl,
): Promise<GPUComputePipeline> {
  const source = shaderSource(control);
  if (process.argv.includes("--emit-wgsl")) process.stdout.write(source);
  const module = device.createShaderModule({ label: "FCA1 frame-control checker", code: source });
  const info = await module.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  for (const error of errors) console.error(`${error.lineNum}:${error.linePos} ${error.message}`);
  if (errors.length > 0) throw new Error(`${errors.length} WGSL compilation error(s)`);
  return device.createComputePipelineAsync({
    layout: "auto", compute: { module, entryPoint: "checkFrameControl" },
  });
}

async function main(): Promise<void> {
  const dawnModule = process.env.WEBGPU_NODE_MODULE;
  if (!dawnModule) throw new Error("WEBGPU_NODE_MODULE is required");
  const control = createSparseCM12FrameControl({
    cellWorkgroups: 7, rowWorkgroups: 11, bodyCapacity: 8, initialGeneration: 5,
    d4Capable: true, rigidCapable: true, baseWords: 64,
  });
  const byteMap = sparseCM12FrameControlByteMap(control);
  if (byteMap.length !== 1 + SPARSE_CM12_FRAME_CONTROL_FAMILY_COUNT
    || byteMap.reduce((bytes, entry) => bytes + entry.sizeBytes, 0)
      !== control.layout.controlBytes
    || byteMap[0]!.offsetBytes !== 4 * control.layout.baseWords) {
    throw new Error("FCA1 byte map is not an exact arena partition");
  }
  const corruptCPU = corruptSparseCM12FrameControlWord(
    control, SPARSE_CM12_FRAME_CONTROL_HEADER.magic, 0,
  );
  if (sparseCM12FrameControlHeaderValid(corruptCPU)) {
    throw new Error("corrupt FCA1 CPU header did not fail closed");
  }

  await acquireWebGPUExclusiveLock("wgsl-check", "sparse-cm12-frame-control");
  try {
    const { create, globals } = await import(dawnModule) as {
      create: (flags: string[]) => GPU; globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    const pipeline = await createPipeline(device, control);
    const output = SPARSE_CM12_FRAME_CONTROL_COVERAGE.output;
    assertCommitted(await evaluate(device, pipeline, control,
      [MODE_NORMAL, 0, 0, 1, 1, 0, 0, output]), "D4-only", {
      scalarD4: 1, faceD4: 1, solid: 0, bodyLive: 0,
    });
    assertCommitted(await evaluate(device, pipeline, control,
      [MODE_NORMAL, 3, 0, 1, 1, 0, 0, output]), "body-live", {
      scalarD4: 0, faceD4: 0, solid: 1, bodyLive: 1,
    });
    assertCommitted(await evaluate(device, pipeline, control,
      [MODE_NORMAL, 0, 1, 1, 1, 0, 0, output]), "boundary-live", {
      scalarD4: 0, faceD4: 0, solid: 1, bodyLive: 0,
    });
    assertCommitted(await evaluate(device, pipeline, control,
      [MODE_NORMAL, 0, 0, 1, 0, 0, 0, output]), "split-D4", {
      scalarD4: 1, faceD4: 0, solid: 0, bodyLive: 0,
    });
    for (const [mode, label] of [[MODE_INVALIDATE_BEFORE_SEAL, "pre-seal invalidate"],
      [MODE_INVALIDATE_AFTER_SEAL, "post-seal invalidate"]] as const) {
      const result = await evaluate(device, pipeline, control,
        [mode, 0, 0, 1, 1, 37, 91, output]);
      assertCommitted(result, label, {
        scalarD4: 0, faceD4: 0, solid: 0, bodyLive: 0,
      });
      assertEqual(result[13]!, 37, `${label} cause`);
      assertEqual(result[14]!, 91, `${label} owner`);
    }
    assertFault(await evaluate(device, pipeline, control,
      [MODE_BODY_OVERFLOW, 9, 0, 0, 0, 0, 0, output]), "body overflow",
    SPARSE_CM12_FRAME_CONTROL_FAULT.bodyCapacity);
    assertFault(await evaluate(device, pipeline, control,
      [MODE_NORMAL, 0, 0, 1, 1, 0, 0, SPARSE_CM12_FRAME_CONTROL_COVERAGE.scalarOutput]),
    "incomplete output", SPARSE_CM12_FRAME_CONTROL_FAULT.incompleteOutput);
    assertFault(await evaluate(device, pipeline, control,
      [MODE_MISSING_BODY_EVIDENCE, 0, 0, 0, 0, 0, 0, output]),
    "missing generation-stamped authority", SPARSE_CM12_FRAME_CONTROL_FAULT.missingEvidence);
    assertFault(await evaluate(device, pipeline, corruptCPU,
      [MODE_NORMAL, 0, 0, 1, 1, 0, 0, output]), "corrupt header",
    SPARSE_CM12_FRAME_CONTROL_FAULT.invalidHeader, false);

    const staticOnly = createSparseCM12FrameControl({
      cellWorkgroups: 7, rowWorkgroups: 11, initialGeneration: 5,
    });
    const staticPipeline = await createPipeline(device, staticOnly);
    assertCommitted(await evaluate(device, staticPipeline, staticOnly,
      [MODE_NORMAL, 0, 0, 0, 0, 0, 0, output]), "static capabilities off", {
      scalarD4: 0, faceD4: 0, solid: 0, bodyLive: 0,
    });
    assertFault(await evaluate(device, staticPipeline, staticOnly,
      [MODE_NORMAL, 0, 0, 1, 0, 0, 0, output]), "D4 capability violation",
    SPARSE_CM12_FRAME_CONTROL_FAULT.capability);
    console.log("Sparse CM12 FCA1: B16/P16 GPU authority, complementary indirect triplets, local D4 invalidation, parity commit, and fail-closed cases passed");
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

void main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exit(1);
});
