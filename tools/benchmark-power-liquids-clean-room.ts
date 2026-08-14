import { pathToFileURL } from "node:url";
import { GPUPerformanceTraceRecorder } from "../lib/core/performance-trace";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Match the captured large-lane resident range without landing on Dawn/Metal's
// unstable synthetic 16,384-page boundary (the same boundary that kills the
// standalone X-5 gather). Fourteen thousand pages is a real observed band
// population and leaves the lower-bound arithmetic unchanged.
const pages = Number(process.env.FLUID_SPIKE_ACTIVE_PAGES ?? 14_000);
const steps = Number(process.env.FLUID_SPIKE_STEPS ?? 60);
const samplesPerPage = 64, samples = pages * samplesPerPage;
if (!Number.isSafeInteger(pages) || pages < 1 || !Number.isSafeInteger(steps) || steps < 1) {
  throw new RangeError("clean-room pages and steps must be positive integers");
}

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-power-liquids-clean-room.ts");
const nodeKeepAlive = setInterval(() => { /* native GPU completion owns exit */ }, 1_000);
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
  const dawn = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter();
  if (!adapter?.features.has("timestamp-query")) throw new Error("clean-room spike requires timestamp-query");
  const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const make = (label: string, bytes: number) => device.createBuffer({ label, size: bytes, usage });
  const params = device.createBuffer({ label: "Clean-room params", size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(params, 0, new Uint32Array([pages, samples, 4, 0]));
  const phiA = make("Clean-room phi A", samples * 4), phiB = make("Clean-room phi B", samples * 4);
  const flags = make("Clean-room flags", samples * 4), velocity = make("Clean-room frozen velocity", samples * 16);
  const halo = make("Clean-room resolved halo", pages * 27 * 4), classes = make("Clean-room page classes", pages * 4);
  const summary = make("Clean-room summaries", pages * 16);
  device.queue.writeBuffer(phiA, 0, Float32Array.from({ length: samples }, (_, i) => Math.fround((i % 257) / 32 - 4)));
  device.queue.writeBuffer(phiB, 0, new Float32Array(samples));
  device.queue.writeBuffer(flags, 0, new Uint32Array(samples).fill(1));
  device.queue.writeBuffer(velocity, 0, Float32Array.from({ length: samples * 4 }, (_, i) =>
    Math.fround(((i * 13) & 127) / 4096)));
  const halos = new Uint32Array(pages * 27);
  for (let page = 0; page < pages; page += 1) for (let slot = 0; slot < 27; slot += 1) {
    halos[page * 27 + slot] = (page + slot + pages - 13) % pages;
  }
  device.queue.writeBuffer(halo, 0, halos);
  const module = device.createShaderModule({ label: "Clean-room six-kernel fine band", code: `
struct Params{pages:u32,samples:u32,substeps:u32,pad:u32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>a:array<f32>;
@group(0)@binding(2)var<storage,read_write>b:array<f32>;
@group(0)@binding(3)var<storage,read_write>flags:array<u32>;
@group(0)@binding(4)var<storage,read>velocity:array<vec4f>;
@group(0)@binding(5)var<storage,read>halo:array<u32>;
@group(0)@binding(6)var<storage,read_write>classes:array<u32>;
@group(0)@binding(7)var<storage,read_write>summary:array<vec4f>;
fn at(page:u32,local:u32)->u32{return (page<<6u)|local;}
fn tap(sourceA:bool,page:u32,local:u32,corner:u32)->f32{let targetPage=halo[page*27u+corner*3u+1u];
 let q=at(targetPage,(local+corner*9u)&63u);return select(b[q],a[q],sourceA);}
@compute @workgroup_size(64)fn classify(@builtin(global_invocation_id)g:vec3u){let page=g.x;if(page>=p.pages){return;}
 var mask=0u;for(var local=0u;local<64u;local+=1u){mask|=select(0u,1u,a[at(page,local)]<0.0);}classes[page]=mask;}
@compute @workgroup_size(64)fn carryWorksets(@builtin(global_invocation_id)g:vec3u){let page=g.x;if(page<p.pages){classes[page]|=2u;}}
@compute @workgroup_size(256)fn advect(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=p.samples){return;}
 let page=i>>6u;let local=i&63u;var value=a[i];for(var step=0u;step<p.substeps;step+=1u){var sum=0.0;
  for(var c=0u;c<8u;c+=1u){sum+=tap(true,page,local,c);}value=.5*value+.0625*sum+.00001*velocity[i].x;}
 b[i]=value;flags[i]=select(1u,17u,value<0.0);}
@compute @workgroup_size(256)fn floodBToA(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=p.samples){return;}
 let page=i>>6u;let local=i&63u;var best=abs(b[i]);for(var c=0u;c<28u;c+=1u){best=min(best,abs(tap(false,page,local,c%8u)));}a[i]=best;}
@compute @workgroup_size(256)fn floodAToB(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=p.samples){return;}
 let page=i>>6u;let local=i&63u;var best=abs(a[i]);for(var c=0u;c<28u;c+=1u){best=min(best,abs(tap(true,page,local,c%8u)));}b[i]=best;}
@compute @workgroup_size(64)fn summarize(@builtin(global_invocation_id)g:vec3u){let page=g.x;if(page>=p.pages){return;}
 var lo=3.4e38;var hi=-3.4e38;var wet=0.0;for(var local=0u;local<64u;local+=1u){let value=b[at(page,local)];
 lo=min(lo,value);hi=max(hi,value);wet+=select(0.0,1.0,(flags[at(page,local)]&16u)!=0u);}summary[page]=vec4f(lo,hi,wet,1.0);}` });
  const names = ["classify", "carryWorksets", "advect", "floodBToA", "floodAToB", "summarize"] as const;
  const pipelines = names.map((entryPoint) => device.createComputePipeline({ label: entryPoint, layout: "auto",
    compute: { module, entryPoint } }));
  const buffers = [[0, params], [1, phiA], [2, phiB], [3, flags], [4, velocity], [5, halo],
    [6, classes], [7, summary]] as const;
  const groups = pipelines.map((pipeline, index) => {
    // Auto layout retains only entry-point-reachable globals.
    const reachable = index < 2 ? (index === 0 ? [0, 1, 6] : [0, 6])
      : index === 2 ? [0, 1, 2, 3, 4, 5]
      : index < 5 ? [0, 1, 2, 5] : [0, 2, 3, 7];
    return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: buffers
      .filter(([binding]) => reachable.includes(binding))
      .map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
  });
  const encodePipeline = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: "Clean-room band pipeline" });
    for (let step = 0; step < steps; step += 1) for (let index = 0; index < pipelines.length; index += 1) {
      pass.setPipeline(pipelines[index]!); pass.setBindGroup(0, groups[index]!);
      pass.dispatchWorkgroups(index < 2 || index === 5 ? Math.ceil(pages / 64) : Math.ceil(samples / 256));
    }
    pass.end();
  };
  const measureTimestamp = async (): Promise<number | undefined> => {
    const encoder = device.createCommandEncoder();
    const trace = new GPUPerformanceTraceRecorder(device, 1, "physics", "clean-room-band",
      [{ id: "fine-sdf-redistance", label: "clean-room-band" }]);
    trace.boundary(encoder, "begin"); encodePipeline(encoder);
    trace.boundary(encoder, "end"); trace.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone();
    const result = await trace.read();
    return result ? result.total_ms / steps : undefined;
  };
  let wallFallbacks = 0;
  const timestampRequested = process.env.FLUID_SPIKE_TIMING === "timestamp";
  const measure = async () => {
    if (timestampRequested) {
      for (let attempt = 0; attempt < 5; attempt += 1) {
        const value = await measureTimestamp(); if (value !== undefined) return value;
      }
    }
    // Some Metal/Dawn combinations expose timestamp-query but intermittently
    // resolve a zero marker boundary. A serialized queue wall is conservative
    // (it includes submission/fence latency) and keeps the experiment usable.
    const encoder = device.createCommandEncoder(); encodePipeline(encoder);
    const started = performance.now(); device.queue.submit([encoder.finish()]);
    await device.queue.onSubmittedWorkDone(); wallFallbacks += 1;
    return (performance.now() - started) / steps;
  };
  await measure(); const trials = [await measure(), await measure(), await measure()];
  const ordered = [...trials].sort((a, b) => a - b), ms = ordered[1]!;
  console.log(JSON.stringify({ schemaVersion: 1, experiment: "X-9-clean-room-band-pipeline",
    correctness: "quality-invalid frozen-velocity lower bound", pages, samples, steps,
    timing: timestampRequested && wallFallbacks === 0
      ? "gpu-timestamp-query" : "serialized-submit-to-fence-wall",
    wallFallbacks,
    kernelsPerAdvance: names.length, trials_msPerAdvance: trials, median_msPerAdvance: ms,
    decision: ms <= 2 ? "ten-x-exists" : ms >= 6 ? "ten-x-does-not-exist-on-slice" : "inconclusive" }, null, 2));
  for (const buffer of [params, phiA, phiB, flags, velocity, halo, classes, summary]) buffer.destroy();
  device.destroy();
} finally {
  clearInterval(nodeKeepAlive);
  await releaseWebGPUExclusiveLock();
}
