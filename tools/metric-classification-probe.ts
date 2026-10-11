/** Frozen metric comparison at all quarter-cell seed sites and cell centres.
 * Uses the particleDepth trilinear arithmetic on both captured vertex fields.
 * Includes every reached cell, conservatively including closed solid cells. */
import assert from "node:assert/strict";
import { NARROW_BAND_SEED_DEPTH, NARROW_BAND_SURFACE_RADIUS } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";

export async function metricClassificationProbe(device: GPUDevice, dims: readonly number[], reach: Uint32Array, baseline: Uint32Array, candidate: Uint32Array): Promise<unknown> {
  assert.equal(dims.length, 3); assert.equal(baseline.length, candidate.length);
  const thresholds = [-4.5, -4, -1, -NARROW_BAND_SURFACE_RADIUS, -NARROW_BAND_SEED_DEPTH, 0, 0.9].map(Math.fround);
  const buffers: GPUBuffer[] = [];
  const upload = (data: Uint32Array) => {
    const buffer = device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    // The native Dawn bridge rejects very large single typed-array uploads.
    // Bound each upload while retaining the original captured u32 words.
    for (let first = 0; first < data.length; first += 262144) device.queue.writeBuffer(buffer, 4 * first, Uint32Array.from(data.subarray(first, first + 262144)));
    buffers.push(buffer); return buffer;
  };
  const original = upload(baseline), changed = upload(candidate), mask = upload(reach);
  const stats = device.createBuffer({ size: 4 * (6 + thresholds.length), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }); buffers.push(stats);
  const read = device.createBuffer({ size: stats.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }); buffers.push(read);
  const sample = (name: string, bank: string) => `fn ${name}(c:vec3u,f:vec3f)->f32{
 var value=0.0;
 for(var k=0u;k<8u;k++){
  let corner=vec3u(k&1u,(k>>1u)&1u,k>>2u);let w=select(1.0-f,f,corner!=vec3u(0));let p=c+corner;
  value+=w.x*w.y*w.z*bitcast<f32>(${bank}[p.x+(D.x+1u)*(p.y+(D.y+1u)*p.z)]);
 }
 return value;
}`;
  try {
    const code = `const D=vec3u(${dims.map(n => `${n}u`).join(",")});
@group(0) @binding(0) var<storage,read> baseline:array<u32>;
@group(0) @binding(1) var<storage,read> candidate:array<u32>;
@group(0) @binding(2) var<storage,read> reach:array<u32>;
@group(0) @binding(3) var<storage,read_write> stats:array<atomic<u32>>;
${sample("baselineDepth", "baseline")}
${sample("candidateDepth", "candidate")}
@compute @workgroup_size(64) fn compare(@builtin(global_invocation_id) gid:vec3u){
 let cells=D.x*D.y*D.z;
 for(var i=gid.x;i<cells;i+=65536u){
  let c=vec3u(i%D.x,(i/D.x)%D.y,i/(D.x*D.y));let t=c/4u;let tile=t.x+(D.x/4u)*(t.y+(D.y/4u)*t.z);
  if((reach[tile]&1u)==0u){continue;}
  var dirty=false;
  for(var k=0u;k<8u;k++){
   let p=c+vec3u(k&1u,(k>>1u)&1u,k>>2u);let at=p.x+(D.x+1u)*(p.y+(D.y+1u)*p.z);
   dirty=dirty||baseline[at]!=candidate[at];
  }
  atomicAdd(&stats[0],9u);if(!dirty){continue;}atomicAdd(&stats[1],9u);
  for(var site=0u;site<9u;site++){
   var f=vec3f(0.5);if(site<8u){f=vec3f(0.25)+0.5*vec3f(vec3u(site&1u,(site>>1u)&1u,site>>2u));}
   let a=baselineDepth(c,f);let b=candidateDepth(c,f);
   if(bitcast<u32>(a)==bitcast<u32>(b)){continue;}
   atomicAdd(&stats[2],1u);if(sign(a)!=sign(b)){atomicAdd(&stats[3],1u);}
   atomicMax(&stats[4],bitcast<u32>(abs(a-b)));
   if(abs(a)>=1e-3){atomicMax(&stats[5],bitcast<u32>(abs(a-b)/abs(a)));}
   ${thresholds.map((threshold, i) => `if((a<=${threshold.toExponential(9)})!=(b<=${threshold.toExponential(9)})){atomicAdd(&stats[${6 + i}],1u);}`).join("\n   ")}
  }
 }
}`;
    const module = device.createShaderModule({ label: "Frozen metric particle-depth classifications", code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error"); assert.deepEqual(errors, []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "compare" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [original, changed, mask, stats].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const e = device.createCommandEncoder(); e.clearBuffer(stats); const pass = e.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1024); pass.end();
    e.copyBufferToBuffer(stats, 0, read, 0, read.size); device.queue.submit([e.finish()]); await read.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(read.getMappedRange()).slice(); read.unmap(); const floats = new Float32Array(words.buffer);
    return { sites: words[0], sitesTouchingChangedVertex: words[1], changedDepths: words[2], signChanges: words[3], maxAbsolute: floats[4], maxRelativeAwayZero: floats[5], relativeMinimumMagnitude: 1e-3, thresholds, classificationChanges: [...words.slice(6)] };
  } finally { for (const buffer of buffers) buffer.destroy(); }
}
