import assert from "node:assert/strict";
import test, { after } from "node:test";
import { pathToFileURL } from "node:url";

import { createSvoDrySceneFragmentWGSL, sparseVoxelDrySceneBindGroupLayoutEntries } from "../lib/webgpu-svo-dry-scene";
import { svoPixelTraceOverlayShader } from "../lib/webgpu-svo-pixel-trace-overlay";

/**
 * Real-device validation for the two shaders the live pixel-trace diagnostic
 * adds. The probe is composed into the dry-scene module, so a WGSL error there
 * is caught by no string assertion: only a compiler sees it.
 */
const modulePath = process.env.WEBGPU_NODE_MODULE;

const PROBE_VERTEX_WGSL = /* wgsl */ `
struct VertexOut{@builtin(position) position:vec4f,@location(0) uv:vec2f}
@vertex fn vertexMain(@builtin(vertex_index) index:u32)->VertexOut {
  var points=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var output:VertexOut;output.position=vec4f(points[index],0,1);output.uv=points[index]*.5+.5;return output;
}
`;

/**
 * One adapter and one device for the whole file. Dawn's Node binding does not
 * survive repeated create/destroy cycles in a single process, and every test
 * here only compiles shaders, so sharing is both safer and faster.
 */
let shared: Promise<GPUDevice> | undefined;

function device(): Promise<GPUDevice> {
  shared ??= (async () => {
    const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, globals);
    const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "an adapter is required for shader validation");
    return adapter.requestDevice({
      requiredLimits: { maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage },
    });
  })();
  return shared;
}

async function compile(gpuDevice: GPUDevice, label: string, code: string): Promise<GPUShaderModule> {
  const shaderModule = gpuDevice.createShaderModule({ label, code });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(
    errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`),
    [],
    `${label} must compile`,
  );
  return shaderModule;
}

// Release the device before the process exits: Dawn's Metal backend faults in
// its own teardown if a live device outlives the runtime.
after(async () => {
  const gpuDevice = await shared;
  gpuDevice?.destroy();
});

const skip = !modulePath && "set WEBGPU_NODE_MODULE for WGSL validation";

test("the trace overlay shader compiles and builds its instanced pipeline", { skip }, async () => {
  const gpuDevice = await device();
  const module = await compile(gpuDevice, "pixel-trace overlay", svoPixelTraceOverlayShader);
  const layout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
    ],
  });
  gpuDevice.pushErrorScope("validation");
  const pipeline = await gpuDevice.createRenderPipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [layout] }),
    vertex: {
      module,
      entryPoint: "vertexMain",
      buffers: [{
        arrayStride: 64,
        stepMode: "instance",
        attributes: [0, 1, 2, 3].map((location) => ({
          shaderLocation: location, offset: location * 16, format: "float32x4" as const,
        })),
      }],
    },
    fragment: { module, entryPoint: "fragmentMain", targets: [{ format: "bgra8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const error = await gpuDevice.popErrorScope();
  assert.equal(error?.message, undefined, "the overlay pipeline must validate");
  assert.ok(pipeline);
});

test("the pixel-trace probe compiles inside every dry-scene traversal composition", { skip }, async () => {
  const gpuDevice = await device();
  for (const traversal of ["hybrid", "canonical", "canonical-parametric", "compact", "wide"] as const) {
    await compile(gpuDevice, `pixel-trace probe (${traversal})`,
      createSvoDrySceneFragmentWGSL(1, traversal, "off", "inline", 0, true));
  }
});

test("the probe pipeline is valid against the production bindings plus one record buffer", { skip }, async () => {
  const gpuDevice = await device();
  // The dry pass spends the whole storage-buffer budget, which is why records go
  // to a storage texture; assert that the production group really is at the cap.
  const storageBuffers = sparseVoxelDrySceneBindGroupLayoutEntries()
    .filter((entry) => entry.buffer?.type === "read-only-storage").length;
  assert.ok(storageBuffers <= gpuDevice.limits.maxStorageBuffersPerShaderStage);
  if (gpuDevice.limits.maxStorageTexturesPerShaderStage < 1) return;
  const [fragment, vertex] = await Promise.all([
    compile(gpuDevice, "pixel-trace probe (hybrid)", createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "inline", 0, true)),
    compile(gpuDevice, "pixel-trace probe vertex", PROBE_VERTEX_WGSL),
  ]);
  const dryLayout = gpuDevice.createBindGroupLayout({ entries: sparseVoxelDrySceneBindGroupLayoutEntries() });
  const probeLayout = gpuDevice.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: GPUShaderStage.FRAGMENT, storageTexture: { access: "write-only", format: "r32uint" } },
      { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    ],
  });
  gpuDevice.pushErrorScope("validation");
  const pipeline = await gpuDevice.createRenderPipelineAsync({
    layout: gpuDevice.createPipelineLayout({ bindGroupLayouts: [dryLayout, probeLayout] }),
    vertex: { module: vertex, entryPoint: "vertexMain" },
    fragment: { module: fragment, entryPoint: "dryProbeMain", targets: [{ format: "rgba8unorm" }] },
    primitive: { topology: "triangle-list" },
  });
  const error = await gpuDevice.popErrorScope();
  assert.equal(error?.message, undefined, "the probe pipeline must validate against the production group");
  assert.ok(pipeline);
});
