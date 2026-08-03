import assert from "node:assert/strict";
import test, { after } from "node:test";
import { pathToFileURL } from "node:url";

import {
  SVO_PIXEL_TRACE_FLAGS,
  SVO_PIXEL_TRACE_HEADER,
  SVO_PIXEL_TRACE_HEADER_WORDS,
  SVO_PIXEL_TRACE_KINDS,
  SVO_PIXEL_TRACE_MAGIC,
  SVO_PIXEL_TRACE_PRIMARY_MODE,
  SVO_PIXEL_TRACE_RECORD_WORDS,
  SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS,
  decodeSvoPixelTrace,
} from "../lib/svo-pixel-trace";
import {
  SVO_BRICK_RASTER_PROBE_CONTRACT,
  createSvoBrickRasterProbeWGSL,
  svoBrickRasterProbeBindGroupLayoutEntries,
  svoBrickRasterProbeBufferBytes,
  svoBrickRasterProbeTextureRows,
  svoBrickRasterProbeWordCount,
} from "../lib/webgpu-svo-brick-raster-probe";

/**
 * The raster-primary probe is a compute module the host never hand-writes WGSL
 * for, so a syntax or binding error in it is invisible to every string
 * assertion. Only a compiler and a pipeline layout catch those.
 */
const modulePath = process.env.WEBGPU_NODE_MODULE;

// Layout builders read the WebGPU stage flags, which only exist once a device
// module has installed its globals.
if (typeof globalThis.GPUShaderStage === "undefined") {
  Object.defineProperty(globalThis, "GPUShaderStage", {
    configurable: true, writable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 },
  });
}

let shared: Promise<GPUDevice> | undefined;
let sharedGpu: GPU | undefined;

function device(): Promise<GPUDevice> {
  shared ??= (async () => {
    const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    sharedGpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await sharedGpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "an adapter is required for shader validation");
    return adapter.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage },
    });
  })();
  return shared;
}

after(async () => {
  const gpuDevice = await shared;
  gpuDevice?.destroy();
});

const skip = !modulePath && "set WEBGPU_NODE_MODULE for WGSL validation";

test("the probe's record texture is whole rows, so the readback needs no padding", () => {
  const words = svoBrickRasterProbeWordCount();
  assert.equal(words, SVO_PIXEL_TRACE_HEADER_WORDS
    + SVO_BRICK_RASTER_PROBE_CONTRACT.recordCapacity * SVO_PIXEL_TRACE_RECORD_WORDS);
  const rows = svoBrickRasterProbeTextureRows();
  assert.ok(rows * SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS >= words);
  // One row is 1024 bytes: a legal bytesPerRow that needs no per-row padding, so
  // the copied buffer is the flat word array the shared decode expects.
  assert.equal(SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4, 1024);
  assert.equal(svoBrickRasterProbeBufferBytes(), rows * 1024);
});

test("the probe declares one binding per resource it reads", () => {
  const entries = svoBrickRasterProbeBindGroupLayoutEntries();
  const bindings = Object.values(SVO_BRICK_RASTER_PROBE_CONTRACT.bindings);
  assert.deepEqual(entries.map((entry) => entry.binding).sort((a, b) => a - b), [...bindings].sort((a, b) => a - b));
  // The instance list and the sort state are read-only here on purpose: this
  // probe observes the frame's output and must never be able to perturb it.
  const writable = entries.filter((entry) => entry.buffer?.type === "storage");
  assert.deepEqual(writable, [], "the probe may not bind any writable storage buffer");
});

test("the covering-proxy count is exact only while the fragment writes its own depth", () => {
  const withDepth = createSvoBrickRasterProbeWGSL({ fragmentDepthWritten: true });
  const withoutDepth = createSvoBrickRasterProbeWGSL({ fragmentDepthWritten: false });
  // A fragment that writes frag_depth cannot be rejected before it runs, so no
  // proxy is ever flagged as merely possibly-shaded. Match the guard itself
  // rather than the bare flag value, which also spells the sort-bucket count.
  const guard = `recordFlags|=${SVO_PIXEL_TRACE_FLAGS.hsrEligible}u;hsrEligible+=1u;`;
  assert.ok(!withDepth.includes(guard),
    "the shipping composition must not claim any fragment might have been killed");
  assert.ok(withoutDepth.includes(guard),
    "without a written depth the count is an upper bound and must say so");
});

test("a dropped record never inflates the count of bricks drawn at this pixel", () => {
  const code = createSvoBrickRasterProbeWGSL({ fragmentDepthWritten: true });
  // Overflowing the record buffer (a long DDA) and overflowing the proxy array
  // are different failures. Only the latter makes the covering count a floor;
  // adding record drops into it would report bricks the frame never drew.
  const covering = `probeWriteWord(${SVO_PIXEL_TRACE_HEADER.coveringProxies}u,`;
  const line = code.split("\n").find((entry) => entry.includes(covering));
  assert.ok(line, "the probe must publish a covering-proxy count");
  assert.ok(line.includes("atomicLoad(&probeOverflow)"),
    "the covering count may only be short by proxies that never reached the array");
  assert.ok(!line.includes("probeRecordDrops"),
    "record drops must never be added to the covering count");
  // Both still reach droppedRecords, because either kind means the drawing is a
  // prefix and the HUD says so.
  assert.match(code, /let recordDrops=atomicLoad\(&probeOverflow\)\+atomicLoad\(&probeRecordDrops\);/);
});

// The probe used to take the aperture as a build-time option and reject a
// non-positive one. It now reads the scene's lens out of the shared view
// uniform block, which is the only way its ray can stay the production ray
// once the aperture became something a document authors.
test("the probe takes its aperture from the camera uniform", () => {
  const code = createSvoBrickRasterProbeWGSL({ fragmentDepthWritten: true });
  assert.match(code, /fn cameraTanHalfFov\(\)->f32\{let authored=uniforms\.cameraPosition\.w;/);
  assert.match(code, /forward\+right\*ndc\.x\*viewport\.x\/viewport\.y\*cameraTanHalfFov\(\)\+up\*ndc\.y\*cameraTanHalfFov\(\)/);
  assert.doesNotMatch(code, /PROBE_TAN_HALF/);
});

test("the raster probe compiles and builds its compute pipeline", { skip }, async () => {
  const gpuDevice = await device();
  for (const fragmentDepthWritten of [true, false]) {
    const code = createSvoBrickRasterProbeWGSL({ fragmentDepthWritten });
    const module = gpuDevice.createShaderModule({ label: `raster probe (depth=${fragmentDepthWritten})`, code });
    const info = await module.getCompilationInfo();
    assert.deepEqual(
      info.messages.filter((message) => message.type === "error")
        .map((error) => `${error.lineNum}:${error.linePos} ${error.message}`),
      [],
      `the raster probe must compile (fragmentDepthWritten=${fragmentDepthWritten})`,
    );
    const layout = gpuDevice.createBindGroupLayout({ entries: svoBrickRasterProbeBindGroupLayoutEntries() });
    const pipeline = await gpuDevice.createComputePipelineAsync({
      layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: SVO_BRICK_RASTER_PROBE_CONTRACT.entryPoint },
    }).catch((error: Error) => {
      assert.fail(`the raster probe pipeline must validate: ${error.message}`);
    });
    assert.ok(pipeline);
  }
});

test("the probe's storage budget fits a compute stage", { skip }, async () => {
  const gpuDevice = await device();
  const entries = svoBrickRasterProbeBindGroupLayoutEntries();
  const storage = entries.filter((entry) => entry.buffer?.type === "read-only-storage").length;
  assert.equal(storage, 4, "the raster trace contract must never grow past four storage buffers");
  assert.ok(storage <= gpuDevice.limits.maxStorageBuffersPerShaderStage,
    `the probe binds ${storage} storage buffers; the device allows ${gpuDevice.limits.maxStorageBuffersPerShaderStage}`);
  assert.ok(gpuDevice.limits.maxStorageTexturesPerShaderStage >= 1);
});

/**
 * The renderer requests default limits, so a probe needing a raised
 * `maxComputeWorkgroupStorageSize` would be unavailable on exactly the devices
 * the diagnostic exists for. Building against a device pinned to the guaranteed
 * minimum is what keeps the per-proxy record small on purpose.
 */
test("the probe's workgroup array fits the guaranteed minimum storage size", { skip }, async () => {
  const gpuDevice = await device();
  const module = gpuDevice.createShaderModule({
    label: "raster probe (default limits)",
    code: createSvoBrickRasterProbeWGSL({ fragmentDepthWritten: true }),
  });
  const layout = gpuDevice.createBindGroupLayout({ entries: svoBrickRasterProbeBindGroupLayoutEntries() });
  // 16384 is the WebGPU-guaranteed minimum; Dawn reports the overrun against
  // whatever the device was granted, so pin the expectation rather than the
  // adapter's generosity.
  assert.ok(gpuDevice.limits.maxComputeWorkgroupStorageSize >= 16384);
  const pipeline = await gpuDevice.createComputePipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout] }),
    compute: { module, entryPoint: SVO_BRICK_RASTER_PROBE_CONTRACT.entryPoint },
  }).catch((error: Error) => {
    assert.fail(`the probe must fit the guaranteed workgroup budget: ${error.message}`);
  });
  assert.ok(pipeline);
});

/** Assemble a probe buffer exactly as the compute shader lays one out. */
function encodeRasterBuffer(proxies: readonly {
  instanceIndex: number; sortBucket: number; cells: number; flags: number;
  tEnter: number; tExit: number;
}[]): { words: Uint32Array; floats: Float32Array } {
  const bytes = new ArrayBuffer(svoBrickRasterProbeBufferBytes());
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  const header = SVO_PIXEL_TRACE_HEADER;
  words[header.magic] = SVO_PIXEL_TRACE_MAGIC;
  words[header.status] = proxies.some((proxy) => (proxy.flags & SVO_PIXEL_TRACE_FLAGS.depthWinner) !== 0) ? 1 : 2;
  words[header.recordCount] = proxies.length;
  words[header.primaryMode] = SVO_PIXEL_TRACE_PRIMARY_MODE.raster;
  words[header.coveringProxies] = proxies.length;
  proxies.forEach((proxy, index) => {
    const base = SVO_PIXEL_TRACE_HEADER_WORDS + index * SVO_PIXEL_TRACE_RECORD_WORDS;
    words[base] = SVO_PIXEL_TRACE_KINDS.brickProxy;
    words[base + 1] = (proxy.sortBucket & 0xffff) | ((proxy.cells & 0xffff) << 16);
    words[base + 2] = proxy.instanceIndex;
    words[base + 3] = proxy.flags;
    floats[base + 10] = proxy.tEnter;
    floats[base + 11] = proxy.tExit;
  });
  return { words, floats };
}

test("a decoded raster trace reports the tournament, not a traversal", () => {
  const { words, floats } = encodeRasterBuffer([
    { instanceIndex: 3, sortBucket: 12, cells: 7, flags: SVO_PIXEL_TRACE_FLAGS.discarded, tEnter: 1, tExit: 2 },
    { instanceIndex: 9, sortBucket: 40, cells: 4, flags: SVO_PIXEL_TRACE_FLAGS.depthWinner | SVO_PIXEL_TRACE_FLAGS.hit, tEnter: 2, tExit: 3 },
    { instanceIndex: 14, sortBucket: 61, cells: 6, flags: SVO_PIXEL_TRACE_FLAGS.depthLoser, tEnter: 3, tExit: 4 },
  ]);
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  assert.equal(trace.primaryMode, "raster");
  assert.equal(trace.raster?.coveringProxies, 3);
  assert.equal(trace.records.length, 3);
  assert.equal(trace.records.filter((record) => record.kind === SVO_PIXEL_TRACE_KINDS.brickProxy).length, 3);
});
