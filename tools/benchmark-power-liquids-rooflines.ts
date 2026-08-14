import { pathToFileURL } from "node:url";
import { GPUPerformanceTraceRecorder } from "../lib/core/performance-trace";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// The captured large-lane band spans roughly 14–18K pages. Use its lower
// observed resident point: Dawn/Metal terminates a single 8-tap dispatch at
// the artificial 16,384-page power-of-two boundary on this M1 Max, while the
// real 14K population is stable and remains representative.
const pages = Number(process.env.FLUID_SOL_ACTIVE_PAGES ?? 14_000);
const samplesPerPage = 64;
const samples = pages * samplesPerPage;
// Nearly one million voxels already amortize submit/fence overhead. Repeating
// the hot gather inside one command buffer can trip Metal's watchdog, so
// stability comes from interleaved rounds rather than a longer submission.
const repeats = Number(process.env.FLUID_SOL_REPEATS ?? 1);
const rounds = Number(process.env.FLUID_SOL_ROUNDS ?? 5);
const substeps = Number(process.env.FLUID_SOL_SUBSTEPS ?? 4);
const timestampRequested = process.env.FLUID_SOL_TIMING === "timestamp";
const debug = process.env.FLUID_SOL_DEBUG === "1";
const debugLog = (message: string) => { if (debug) console.error(`[X-5] ${message}`); };
if (![pages, repeats, rounds, substeps].every((value) => Number.isSafeInteger(value) && value > 0)) {
  throw new RangeError("page, repeat, round and substep counts must be positive integers");
}

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-power-liquids-rooflines.ts");
// Dawn's native promises are not libuv handles. Keep the standalone process
// alive until its explicit queue/map fences and cleanup finish.
const nodeKeepAlive = setInterval(() => { /* native GPU completion owns exit */ }, 1_000);
try {
  debugLog("lock acquired");
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  debugLog("adapter acquired");
  if (!adapter) throw new Error("rooflines require a WebGPU adapter");
  if (timestampRequested && !adapter.features.has("timestamp-query")) {
    throw new Error("FLUID_SOL_TIMING=timestamp requires timestamp-query");
  }
  const device = await adapter.requestDevice(timestampRequested
    ? { requiredFeatures: ["timestamp-query"] } : {});
  debugLog("device acquired");
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, size: number) => device.createBuffer({ label, size, usage: storage });
  const params = device.createBuffer({ label: "SOL parameters", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([pages, samplesPerPage, substeps, samples]));
  const input = make("SOL packed phi input", samples * 4);
  const output = make("SOL packed phi output", samples * 4);
  const velocity = make("SOL pre-gathered velocity", samples * 16);
  const owner = make("SOL flat air-owner records", samples * 16);
  const halo = make("SOL pre-resolved 27-page halo", pages * 27 * 4);
  const values = Float32Array.from({ length: samples }, (_, index) => Math.fround((index & 1023) / 1024));
  device.queue.writeBuffer(input, 0, values);
  device.queue.writeBuffer(velocity, 0, Float32Array.from({ length: samples * 4 }, (_, index) =>
    Math.fround(((index * 17) & 255) / 4096)));
  device.queue.writeBuffer(owner, 0, Float32Array.from({ length: samples * 4 }, (_, index) =>
    Math.fround(((index * 29) & 255) / 2048)));
  const halos = new Uint32Array(pages * 27);
  for (let page = 0; page < pages; page += 1) for (let slot = 0; slot < 27; slot += 1) {
    halos[page * 27 + slot] = (page + slot + pages - 13) % pages;
  }
  device.queue.writeBuffer(halo, 0, halos);
  debugLog("buffers initialized");

  const shader = device.createShaderModule({ label: "Power-liquids speed-of-light kernels", code: `
struct Params{pages:u32,samplesPerPage:u32,substeps:u32,samples:u32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>input:array<f32>;
@group(0)@binding(2)var<storage,read_write>output:array<f32>;
@group(0)@binding(3)var<storage,read>halo:array<u32>;
@group(0)@binding(4)var<storage,read>velocity:array<vec4f>;
@group(0)@binding(5)var<storage,read>owner:array<vec4f>;
fn b4Index(page:u32,local:u32)->u32{return (page<<6u)|local;}
fn gather(page:u32,local:u32,corner:u32)->f32{
 let x=local&3u;let y=(local>>2u)&3u;let z=local>>4u;
 let nx=x+(corner&1u);let ny=y+((corner>>1u)&1u);let nz=z+((corner>>2u)&1u);
 let sx=select(1u,2u,nx>3u);let sy=select(1u,2u,ny>3u);let sz=select(1u,2u,nz>3u);
 let targetPage=halo[page*27u+sx+3u*sy+9u*sz];
 return input[b4Index(targetPage,(nx&3u)|((ny&3u)<<2u)|((nz&3u)<<4u))];}
@compute @workgroup_size(256)fn sol0Stream(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i<p.samples){output[i]=input[i];}}
@compute @workgroup_size(256)fn sol1Gather(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=p.samples){return;}let page=i>>6u;let local=i&63u;var sum=0.0;
 for(var corner=0u;corner<8u;corner+=1u){sum+=gather(page,local,corner);}output[i]=sum*0.125;}
@compute @workgroup_size(256)fn sol2Flood(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=p.samples){return;}let page=i>>6u;let local=i&63u;var best=abs(input[i]);
 for(var candidate=0u;candidate<28u;candidate+=1u){let slot=(candidate*11u+local)&63u;
  let targetPage=halo[page*27u+(candidate%27u)];best=min(best,abs(input[b4Index(targetPage,slot)]));}
 output[i]=best;}
@compute @workgroup_size(256)fn sol3Backtrace(@builtin(global_invocation_id)gid:vec3u){
 let i=gid.x;if(i>=p.samples){return;}let page=i>>6u;let local=i&63u;var value=input[i];
 for(var step=0u;step<p.substeps;step+=1u){let v=velocity[i]+owner[i];var sum=0.0;
  for(var corner=0u;corner<8u;corner+=1u){sum+=gather(page,local,corner);}
  value=0.5*value+0.0625*sum+1e-5*(v.x+v.y+v.z);}
 output[i]=value;}` });
  const entries = ["sol0Stream", "sol1Gather", "sol2Flood", "sol3Backtrace"] as const;
  const traffic = { sol0Stream: 8, sol1Gather: 40, sol2Flood: 116,
    sol3Backtrace: substeps * (40 + 32) + 4 } as const;
  const variants = entries.map((entryPoint) => {
    const pipeline = device.createComputePipeline({ label: entryPoint, layout: "auto",
      compute: { module: shader, entryPoint } });
    const buffers: readonly (readonly [number, GPUBuffer])[] = entryPoint === "sol0Stream"
      ? [[0, params], [1, input], [2, output]]
      : entryPoint === "sol3Backtrace"
        ? [[0, params], [1, input], [2, output], [3, halo], [4, velocity], [5, owner]]
        : [[0, params], [1, input], [2, output], [3, halo]];
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries:
      buffers.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
    return { entryPoint, pipeline, group };
  });
  debugLog("pipelines compiled");
  let wallFallbacks = 0;
  const measure = async (variant: typeof variants[number]) => {
    const encoder = device.createCommandEncoder();
    const trace = timestampRequested
      ? new GPUPerformanceTraceRecorder(device, 1, "physics", variant.entryPoint,
        [{ id: "fine-sdf-redistance", label: variant.entryPoint }]) : undefined;
    trace?.boundary(encoder, "begin"); const pass = encoder.beginComputePass();
    pass.setPipeline(variant.pipeline); pass.setBindGroup(0, variant.group);
    for (let repeat = 0; repeat < repeats; repeat += 1) pass.dispatchWorkgroups(Math.ceil(samples / 256));
    pass.end(); trace?.boundary(encoder, "end"); trace?.resolve(encoder);
    const started = performance.now();
    device.queue.submit([encoder.finish()]);
    debugLog(`${variant.entryPoint} submitted`);
    // Dawn's map callback alone does not keep Node's event loop alive. The
    // queue fence makes this standalone probe own a live asynchronous handle
    // until timestamp bytes are ready, so it cannot silently exit and strand
    // the process-wide GPU lock.
    await device.queue.onSubmittedWorkDone();
    debugLog(`${variant.entryPoint} queue complete`);
    const wall_ms = performance.now() - started;
    if (trace) {
      const result = await trace.read();
      debugLog(`${variant.entryPoint} timestamps mapped`);
      if (result) return result.total_ms / repeats;
    }
    wallFallbacks += 1;
    return wall_ms / repeats;
  };
  for (const variant of variants) await measure(variant);
  debugLog("warmup complete");
  const measurements = Object.fromEntries(variants.map((variant) => [variant.entryPoint, [] as number[]]));
  for (let round = 0; round < rounds; round += 1) {
    const order = round % 2 === 0 ? variants : [...variants].reverse();
    for (const variant of order) measurements[variant.entryPoint]!.push(await measure(variant));
  }
  const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
  const result = Object.fromEntries(entries.map((entryPoint) => {
    const ms = median(measurements[entryPoint]!);
    return [entryPoint, { ms, usPerMvoxel: ms * 1000 / (samples / 1e6),
      effectiveGB_s: samples * traffic[entryPoint] / (ms / 1000) / 1e9,
      assumedBytesPerVoxel: traffic[entryPoint], samples_ms: measurements[entryPoint] }];
  }));
  console.log(JSON.stringify({ schemaVersion: 1, experiment: "X-5-speed-of-light",
    pages, samplesPerPage, samples, substeps, repeats, rounds,
    timing: timestampRequested && wallFallbacks === 0
      ? "gpu-timestamp-query" : "serialized-submit-to-fence-wall",
    wallFallbacks, result }, null, 2));
  for (const buffer of [params, input, output, velocity, owner, halo]) buffer.destroy();
  device.destroy();
} finally {
  clearInterval(nodeKeepAlive);
  await releaseWebGPUExclusiveLock();
}
