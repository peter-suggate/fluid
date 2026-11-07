// GPU canary for fluid-gpu-run.sh. It dispatches one small kernel and checks
// every word it reads back. After a Metal out-of-memory fault, Metal can abort
// every command buffer while Dawn reports success, so readbacks come back
// stale or zero and runs "pass" while computing nothing (2026-11-07). The
// canary fails fast in that state instead of after a long run.
//   node tools/gpu/gpu-canary.mjs   exit 0 healthy, 1 wrong or missing readback
import {resolve, dirname} from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const fail = message => { console.error(`gpu-canary: ${message}`); process.exit(1); };
const timer = setTimeout(() => fail('timed out after 60 s'), 60_000);
try {
  const dawn = await import(pathToFileURL(resolve(root, 'node_modules/webgpu/index.js')).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create(['backend=metal']);
  const adapter = await gpu.requestAdapter();
  if (!adapter) fail('no adapter');
  const device = await adapter.requestDevice();
  let deviceError;
  device.onuncapturederror = event => { deviceError = event.error?.message ?? String(event.error); };
  // A seed the kernel must transform, so neither zeros nor the untouched
  // input pass, and 4096 words so a partial write fails too.
  const N = 4096, seed = (Date.now() ^ (process.pid << 8)) >>> 0;
  const module = device.createShaderModule({code: `
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> seed: vec4u;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x; if (i >= ${N}u) { return; }
  data[i] = (i * 2654435761u) ^ seed.x ^ (data[i] + 1u);
}`});
  const pipeline = await device.createComputePipelineAsync({layout: 'auto', compute: {module, entryPoint: 'main'}});
  const data = device.createBuffer({size: 4*N, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
  const uniform = device.createBuffer({size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
  const read = device.createBuffer({size: 4*N, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
  const input = new Uint32Array(N).map((_, i) => (i * 40503 + seed) >>> 0);
  device.queue.writeBuffer(data, 0, input);
  device.queue.writeBuffer(uniform, 0, new Uint32Array([seed, 0, 0, 0]));
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({layout: pipeline.getBindGroupLayout(0), entries: [{binding: 0, resource: {buffer: data}}, {binding: 1, resource: {buffer: uniform}}]}));
  pass.dispatchWorkgroups(N/64);
  pass.end();
  encoder.copyBufferToBuffer(data, 0, read, 0, 4*N);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const out = new Uint32Array(read.getMappedRange().slice(0));
  read.unmap();
  if (deviceError) fail(`device error: ${deviceError}`);
  let bad = 0, first = -1, wanted = 0;
  for (let i = 0; i < N; i++) {
    const want = (Math.imul(i, 2654435761) ^ seed ^ ((input[i] + 1) >>> 0)) >>> 0;
    if (out[i] !== want) { bad++; if (first < 0) {first = i; wanted = want;} }
  }
  if (bad) fail(`${bad}/${N} words wrong (first ${first}: got ${out[first]}, want ${wanted}); the GPU is not executing work`);
  device.destroy();
  clearTimeout(timer);
  process.exit(0);
} catch (error) {
  fail(String(error?.stack ?? error));
}
