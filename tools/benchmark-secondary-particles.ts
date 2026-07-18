import { pathToFileURL } from "node:url";
import {
  SECONDARY_PARTICLE_STRIDE_BYTES,
  SecondaryParticleRenderPipeline,
  type GPUSecondaryParticleSource
} from "../lib/webgpu-secondary-particles";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter || !adapter.features.has("timestamp-query")) throw new Error("The spray benchmark requires timestamp-query support");
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });

const width = Math.max(64, Number(process.env.FLUID_SPRAY_WIDTH ?? 1280));
const height = Math.max(64, Number(process.env.FLUID_SPRAY_HEIGHT ?? 720));
const capacity = Math.max(1, Math.min(65_536, Number(process.env.FLUID_SPRAY_CAPACITY ?? 16_384)));
const activeFraction = Math.max(0, Math.min(1, Number(process.env.FLUID_SPRAY_ACTIVE_FRACTION ?? 1)));
const iterations = Math.max(10, Number(process.env.FLUID_SPRAY_ITERATIONS ?? 120));
const shapeMode = process.env.FLUID_SPRAY_SHAPE === "sphere" ? "sphere" : "ellipsoid";
const activeCount = Math.floor(capacity * activeFraction);

const uniformBuffer = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(uniformBuffer, 0, new Float32Array([
  width, height, 0, 0,
  0, 0, 4, 0,
  0, 0, 0, 0,
  10, 10, 10, 0
]));

const particleFloats = new Float32Array(capacity * SECONDARY_PARTICLE_STRIDE_BYTES / 4);
const columns = Math.max(1, Math.ceil(Math.sqrt(activeCount * width / height)));
const rows = Math.max(1, Math.ceil(activeCount / columns));
for (let index = 0; index < activeCount; index += 1) {
  const offset = index * 16;
  const x = index % columns, y = Math.floor(index / columns);
  const u = columns > 1 ? x / (columns - 1) : 0.5;
  const v = rows > 1 ? y / (rows - 1) : 0.5;
  const phase = index * 2.399963229728653;
  particleFloats.set([
    (u - 0.5) * 8.8 + 0.018 * Math.sin(phase),
    (v - 0.5) * 4.9 + 0.018 * Math.cos(phase),
    0.12 * Math.sin(phase * 0.37),
    0.032 + 0.012 * ((index * 17) % 23) / 22,
    1.3 + 0.5 * Math.sin(phase), 0.7 + 0.3 * Math.cos(phase), 0.2 * Math.sin(phase * 0.7), 0.025,
    0, 1, 0, 1.8,
    2.2 + 1.0 * ((index * 13) % 29) / 28, index % 3 === 0 ? 2 : 1, 1, 0.12
  ], offset);
}
const particleBuffer = device.createBuffer({
  size: particleFloats.byteLength,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
});
device.queue.writeBuffer(particleBuffer, 0, particleFloats);
const source: GPUSecondaryParticleSource = { buffer: particleBuffer, capacity, strideBytes: SECONDARY_PARTICLE_STRIDE_BYTES };
const pipeline = new SecondaryParticleRenderPipeline(device, "bgra8unorm", uniformBuffer);
await pipeline.initialize();
pipeline.setSource(source);

// Keep the former analytic sphere path in the benchmark, not in production,
// so performance comparisons use the same buffer, targets, and draw count.
const sphereModule = device.createShaderModule({ code: /* wgsl */ `
struct ViewUniforms { viewport:vec4f, cameraPosition:vec4f, cameraTarget:vec4f, container:vec4f }
struct Particle { positionRadius:vec4f, velocityAge:vec4f, birthNormalLifetime:vec4f, shape:vec4f }
@group(0) @binding(0) var<uniform> view:ViewUniforms;
@group(0) @binding(1) var<storage,read> particles:array<Particle>;
struct VertexOut { @builtin(position) clip:vec4f, @location(0) local:vec2f, @location(1) @interpolate(flat) center:vec3f, @location(2) @interpolate(flat) radius:f32, @location(3) @interpolate(flat) enabled:f32 }
struct FragmentOut { @location(0) position:vec4f, @location(1) normal:vec4f, @builtin(frag_depth) depth:f32 }
fn forward()->vec3f{return normalize(view.cameraTarget.xyz-view.cameraPosition.xyz);}
fn right()->vec3f{return normalize(cross(forward(),vec3f(0,1,0)));}
fn up()->vec3f{return normalize(cross(right(),forward()));}
fn project(world:vec3f)->vec4f{let relative=world-view.cameraPosition.xyz;let eyeDepth=dot(relative,forward());let aspect=view.viewport.x/max(view.viewport.y,1.0);let ndc=vec2f(dot(relative,right())/(max(eyeDepth,.001)*aspect*.72),dot(relative,up())/(max(eyeDepth,.001)*.72));return vec4f(ndc*eyeDepth,clamp(eyeDepth/50.0,0.0,1.0)*eyeDepth,eyeDepth);}
@vertex fn sphereVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->VertexOut{var corners=array<vec2f,6>(vec2f(-1.05,-1.05),vec2f(1.05,-1.05),vec2f(-1.05,1.05),vec2f(-1.05,1.05),vec2f(1.05,-1.05),vec2f(1.05,1.05));let particle=particles[instance];var out:VertexOut;out.local=corners[vertex];out.center=particle.positionRadius.xyz;out.radius=particle.positionRadius.w;out.enabled=particle.shape.z;if(out.enabled<.5||out.radius<=0){out.clip=vec4f(2,2,2,1);out.enabled=0;return out;}out.clip=project(out.center+(right()*out.local.x+up()*out.local.y)*out.radius);return out;}
fn sphereInterface(input:VertexOut,back:bool)->FragmentOut{let radius2=dot(input.local,input.local);if(input.enabled<.5||radius2>1){discard;}let z=sqrt(max(0.0,1.0-radius2));let facing=select(-1.0,1.0,back);let normal=normalize(right()*input.local.x+up()*input.local.y+forward()*(facing*z));let world=input.center+normal*input.radius;let clip=project(world);return FragmentOut(vec4f(world,1),vec4f(normal,1),clamp(clip.z/max(clip.w,.001),0.0,1.0));}
@fragment fn sphereFront(input:VertexOut)->FragmentOut{return sphereInterface(input,false);}
@fragment fn sphereBack(input:VertexOut)->FragmentOut{return sphereInterface(input,true);}
` });
const sphereLayout = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
  { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } }
] });
const spherePipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [sphereLayout] });
const sphereDescriptor = (entryPoint: "sphereFront" | "sphereBack"): GPURenderPipelineDescriptor => ({
  layout: spherePipelineLayout,
  vertex: { module: sphereModule, entryPoint: "sphereVertex" },
  fragment: { module: sphereModule, entryPoint, targets: [{ format: "rgba16float" }, { format: "rgba16float" }] },
  primitive: { topology: "triangle-list" },
  depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" }
});
const [sphereFrontPipeline, sphereBackPipeline] = await Promise.all([
  device.createRenderPipelineAsync(sphereDescriptor("sphereFront")),
  device.createRenderPipelineAsync(sphereDescriptor("sphereBack"))
]);
const sphereBindGroup = device.createBindGroup({ layout: sphereLayout, entries: [
  { binding: 0, resource: { buffer: uniformBuffer } },
  { binding: 1, resource: { buffer: particleBuffer } }
] });

const position = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
const normal = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT });
const depth = device.createTexture({ size: [width, height], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
const positionView = position.createView(), normalView = normal.createView(), depthView = depth.createView();

function encodeSide(encoder: GPUCommandEncoder, side: "front" | "back", timestampWrites?: GPURenderPassTimestampWrites) {
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      { view: positionView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      { view: normalView, clearValue: { r: 0, g: 1, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" }
    ],
    depthStencilAttachment: { view: depthView, depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store" },
    ...(timestampWrites ? { timestampWrites } : {})
  });
  if (shapeMode === "ellipsoid") pipeline.encodeOpticalInterface(pass, side);
  else {
    pass.setPipeline(side === "front" ? sphereFrontPipeline : sphereBackPipeline);
    pass.setBindGroup(0, sphereBindGroup);
    pass.draw(6, capacity);
  }
  pass.end();
}

for (let warmup = 0; warmup < 12; warmup += 1) {
  const encoder = device.createCommandEncoder();
  encodeSide(encoder, "front");
  encodeSide(encoder, "back");
  device.queue.submit([encoder.finish()]);
}
await device.queue.onSubmittedWorkDone();

const querySet = device.createQuerySet({ type: "timestamp", count: iterations * 4 });
const resolve = device.createBuffer({ size: iterations * 4 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const readback = device.createBuffer({ size: iterations * 4 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const encoder = device.createCommandEncoder();
for (let iteration = 0; iteration < iterations; iteration += 1) {
  const query = iteration * 4;
  encodeSide(encoder, "front", { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 });
  encodeSide(encoder, "back", { querySet, beginningOfPassWriteIndex: query + 2, endOfPassWriteIndex: query + 3 });
}
encoder.resolveQuerySet(querySet, 0, iterations * 4, resolve, 0);
encoder.copyBufferToBuffer(resolve, 0, readback, 0, iterations * 4 * 8);
device.queue.submit([encoder.finish()]);
await readback.mapAsync(GPUMapMode.READ);
const timestamps = new BigUint64Array(readback.getMappedRange());
const front: number[] = [], back: number[] = [], total: number[] = [];
for (let iteration = 0; iteration < iterations; iteration += 1) {
  const query = iteration * 4;
  const front_ms = Number(timestamps[query + 1] - timestamps[query]) / 1e6;
  const back_ms = Number(timestamps[query + 3] - timestamps[query + 2]) / 1e6;
  front.push(front_ms); back.push(back_ms); total.push(front_ms + back_ms);
}
const summary = (samples: number[]) => {
  const sorted = [...samples].sort((a, b) => a - b);
  const percentile = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))];
  return { median_ms: percentile(0.5), p95_ms: percentile(0.95), maximum_ms: sorted.at(-1) ?? 0 };
};
console.log(JSON.stringify({
  backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
  resolution: [width, height], capacity, activeCount, iterations, shapeMode,
  particleStrideBytes: SECONDARY_PARTICLE_STRIDE_BYTES,
  particleStorageMiB: particleFloats.byteLength / (1024 * 1024),
  front: summary(front), back: summary(back), combined: summary(total)
}, null, 2));

readback.unmap();
for (const resource of [readback, resolve, querySet, position, normal, depth, particleBuffer, uniformBuffer]) resource.destroy();
device.destroy();
