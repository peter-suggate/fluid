import { pathToFileURL } from "node:url";
const { create, globals } = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as any;
Object.assign(globalThis, globals);
const gpu = create(["backend=metal"]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
device.addEventListener("uncapturederror", (e: any) => console.error("ERR:", e.error.message));
const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3u) { if (id.x < 256) { data[id.x] = f32(id.x) * 2.0; } }
` });
const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
const buffer = device.createBuffer({ size: 1024, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] });
const encoder = device.createCommandEncoder();
const pass = encoder.beginComputePass();
pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(4); pass.end();
const read = device.createBuffer({ size: 1024, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
encoder.copyBufferToBuffer(buffer, 0, read, 0, 1024);
device.queue.submit([encoder.finish()]);
await read.mapAsync(GPUMapMode.READ);
const values = new Float32Array(read.getMappedRange());
console.log("compute result [0..4]:", values[0], values[1], values[2], values[3], "sum:", values.reduce((a, b) => a + b, 0));
device.destroy();
