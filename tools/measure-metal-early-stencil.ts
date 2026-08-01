/**
 * Does this GPU reject fragments on the stencil test *before* running a
 * fragment shader that writes `frag_depth`?
 *
 * The brick raster's overdraw problem is that writing `frag_depth` and calling
 * `discard` both leave a fragment unresolved until it has been shaded, which
 * switches off tile-based hidden-surface removal — every overlapping brick
 * proxy shades, not just the winner. The stencil test is the one rejection that
 * does not depend on the fragment's depth or coverage, so in principle it can
 * still run early. If it does, sorting brick proxies into depth-disjoint slabs
 * and letting each slab stencil-mark the pixels it resolved would cull later
 * slabs before they shade, exactly and without changing the image.
 *
 * That redesign means moving the shared G-buffer depth attachment to
 * `depth32float-stencil8`, so this measures the premise in isolation first.
 * Three arms over the same expensive shader and the same geometry:
 *
 *   none    no stencil attachment  — every fragment shades (the baseline)
 *   pass    stencil test that always passes — isolates stencil-test overhead
 *   reject  stencil test that always fails  — shades nothing IF rejection is early
 *
 * `reject` collapsing towards zero is the positive result. `reject` staying near
 * `none` means the shader runs before the stencil test and the slab redesign
 * cannot work on this hardware.
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/measure-metal-early-stencil.ts
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const flag = (name: string): string | undefined =>
  process.argv.find((argument) => argument.startsWith(`--${name}=`))?.split("=").slice(1).join("=");

const width = Number(flag("width") ?? 1500);
const height = Number(flag("height") ?? 1500);
/** Overlapping full-screen layers, standing in for brick-proxy overdraw. */
const layers = Number(flag("layers") ?? 8);
/** Inner-loop trips, standing in for the in-brick DDA's cost. */
const workPerFragment = Number(flag("work") ?? 64);
const cycles = Number(flag("cycles") ?? 40);
const outPath = flag("out") ?? "artifacts/render-raster-primary/early-stencil.json";

const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "WEBGPU_NODE_MODULE must point at the Dawn node binding");
const { create, globals } = (await import(pathToFileURL(modulePath).href)) as {
  create(options: string[]): GPU; globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create(["backend=metal", "enable-dawn-features=use_user_defined_labels_in_backend"]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "no Metal adapter");
assert.ok(adapter.features.has("depth32float-stencil8"),
  "adapter cannot pair stencil with 32-bit float depth, which the slab redesign would require");
const device = await adapter.requestDevice({ requiredFeatures: ["depth32float-stencil8"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => {
  validationErrors.push(String((event as GPUUncapturedErrorEvent).error.message));
});

const DEPTH_FORMAT = "depth32float-stencil8" as GPUTextureFormat;
const COLOR_FORMAT = "rgba8unorm" as GPUTextureFormat;

// The shader deliberately mirrors the brick fragment's two HSR blockers: it
// writes frag_depth, and it discards on a data-dependent test. The loop stands
// in for the DDA so a skipped fragment is unmistakable in the timing.
const module = device.createShaderModule({
  label: "Early stencil probe",
  code: /* wgsl */ `
struct Push { layer: f32, work: f32 }
@group(0) @binding(0) var<uniform> push: Push;
struct Out { @location(0) colour: vec4f, @builtin(frag_depth) depth: f32 }
@vertex fn vertexMain(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
  var corner = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(-1.0, 3.0), vec2f(3.0, -1.0));
  return vec4f(corner[index], 0.5, 1.0);
}
@fragment fn fragmentMain(@builtin(position) position: vec4f) -> Out {
  var accumulator = position.x * 0.001 + position.y * 0.002 + push.layer;
  let trips = i32(push.work);
  for (var step = 0; step < trips; step = step + 1) {
    accumulator = fract(sin(accumulator * 12.9898 + f32(step)) * 43758.5453);
    accumulator = accumulator * 0.75 + sqrt(abs(accumulator)) * 0.25;
  }
  // Data-dependent discard, exactly like a brick whose ray finds no surface.
  if (accumulator > 2.0) { discard; }
  return Out(vec4f(accumulator, 0.0, 0.0, 1.0), clamp(accumulator, 0.0, 1.0));
}`,
});

const uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
const layout = device.createBindGroupLayout({
  entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
});
const bindGroup = device.createBindGroup({ layout, entries: [{ binding: 0, resource: { buffer: uniform } }] });
const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });

type Arm = "none" | "pass" | "reject";
const stencilFace = (arm: Arm): GPUStencilFaceState =>
  arm === "reject" ? { compare: "never", failOp: "keep", depthFailOp: "keep", passOp: "keep" }
    : { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "keep" };

const pipelineFor = (arm: Arm): GPURenderPipeline => device.createRenderPipeline({
  label: `Early stencil probe (${arm})`,
  layout: pipelineLayout,
  vertex: { module, entryPoint: "vertexMain" },
  fragment: { module, entryPoint: "fragmentMain", targets: [{ format: COLOR_FORMAT }] },
  primitive: { topology: "triangle-list" },
  depthStencil: {
    format: DEPTH_FORMAT,
    depthWriteEnabled: true,
    depthCompare: "greater",
    ...(arm === "none" ? {} : {
      stencilFront: stencilFace(arm), stencilBack: stencilFace(arm),
      stencilReadMask: 0xff, stencilWriteMask: 0,
    }),
  },
});

const colour = device.createTexture({
  size: { width, height }, format: COLOR_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const depth = device.createTexture({
  size: { width, height }, format: DEPTH_FORMAT, usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const colourView = colour.createView();
const depthView = depth.createView();

const encodeArm = (encoder: GPUCommandEncoder, pipeline: GPURenderPipeline, arm: Arm): void => {
  const pass = encoder.beginRenderPass({
    label: `Early stencil probe ${arm}`,
    colorAttachments: [{ view: colourView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }],
    depthStencilAttachment: {
      view: depthView,
      depthClearValue: 0, depthLoadOp: "clear", depthStoreOp: "store",
      // Cleared to zero so a "never" test rejects and an "always" test passes;
      // neither arm writes stencil, so the buffer stays constant across layers.
      stencilClearValue: 0, stencilLoadOp: "clear", stencilStoreOp: "store",
    },
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  if (arm !== "none") pass.setStencilReference(1);
  for (let layer = 0; layer < layers; layer += 1) pass.draw(3);
  pass.end();
};

const median = (values: readonly number[]): number => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
};

const timeArm = async (arm: Arm): Promise<{ median_ms: number; minimum_ms: number; samples: number }> => {
  const pipeline = pipelineFor(arm);
  device.queue.writeBuffer(uniform, 0, new Float32Array([1, workPerFragment, 0, 0]));
  for (let warmup = 0; warmup < 4; warmup += 1) {
    const encoder = device.createCommandEncoder();
    encodeArm(encoder, pipeline, arm);
    device.queue.submit([encoder.finish()]);
  }
  await device.queue.onSubmittedWorkDone();
  const samples: number[] = [];
  for (let cycle = 0; cycle < cycles; cycle += 1) {
    const encoder = device.createCommandEncoder();
    encodeArm(encoder, pipeline, arm);
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    samples.push(performance.now() - started);
  }
  return { median_ms: median(samples), minimum_ms: Math.min(...samples), samples: samples.length };
};

const arms: Record<string, Awaited<ReturnType<typeof timeArm>>> = {};
for (const arm of ["none", "pass", "reject"] as const) arms[arm] = await timeArm(arm);
assert.equal(validationErrors.length, 0, `device errors: ${validationErrors.join(" | ")}`);

const rejectedFraction = 1 - arms.reject.median_ms / arms.none.median_ms;
const report = {
  tool: "measure-metal-early-stencil",
  adapter: { vendor: adapter.info?.vendor, device: adapter.info?.device },
  configuration: { width, height, layers, workPerFragment, cycles },
  arms,
  rejectedFraction,
  // A shader that runs and is then thrown away costs the same as one that is
  // kept, so "reject" tracking "none" is the negative result.
  earlyStencilObserved: rejectedFraction > 0.5,
  interpretation: rejectedFraction > 0.5
    ? "stencil rejection happens before the fragment shader, so depth-disjoint brick slabs "
      + "with stencil marking would cull overlapping proxies before they shade"
    : "stencil rejection happens after the fragment shader, so the slab redesign cannot "
      + "recover the overdraw and the brick raster is bound by fragments launched",
};
mkdirSync(path.dirname(outPath), { recursive: true });
writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify(report, null, 2));
device.destroy();
