/** Focused GPU proof of the integrated integer crossing sampler. The source
 * under test is extracted from production; the reference is the original
 * resolved sampler. Exercise every closed-domain vertex of a non-cubic grid,
 * mixed positive-side ownership, distinct fine/base banks and signed zeros. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { narrowBandMembershipWGSL } from "../lib/methods/uniform/uniform-narrow-band-membership.wgsl";
import { uniformMixedVertexSamplingSource } from "../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl";
import { pruneWGSLToEntry } from "./prune-wgsl-entry";

const dims = [12, 16, 20], patterns = 8;
const vertices = dims.reduce((n, d) => n * (d + 1), 1), cells = dims.reduce((n, d) => n * d, 1), tiles = cells / 64;
const start = narrowBandMembershipWGSL.indexOf("fn nbCrossingVertex("), end = narrowBandMembershipWGSL.indexOf("\n}", start) + 2;
assert.ok(start >= 0 && end > start, "Integrated crossing sampler must be present");
const source = pruneWGSLToEntry(/* wgsl */ `
const UM_D=vec3u(${dims.join(",")});const UM_T=UM_D/4u;
const umCounts=vec4u(0,0,0,8);override umRegularFine:bool=false;
var<private> fixture:u32;
@group(0) @binding(0) var<storage,read_write> counters:array<atomic<u32>,8>;
fn umCorner(i:u32,n:u32)->vec3u{return vec3u(i%n,(i/n)%n,i/(n*n));}
fn umTileAt(t:vec3u)->u32{return t.x+UM_T.x*(t.y+UM_T.y*t.z);}
fn umTileWidth(t:u32)->u32{
 if(fixture==0u){return 4u;}if(fixture==1u){return 1u;}
 return select(4u,1u,((t*13u+fixture*7u)%(fixture+2u))<fixture);
}
fn fixtureValue(p:vec3u,salt:u32)->f32{
 let i=p.x+13u*p.y+37u*p.z+17u*fixture+salt;
 if(i%11u==0u){return bitcast<f32>((i&1u)<<31u);}
 if(i%17u==0u){return select(-1e-30,1e-30,(i&1u)!=0u);}
 if(i%23u==0u){return select(-1e20,1e20,(i&1u)!=0u);}
 return f32(i32(i%31u)-15)/16.0;
}
fn umLoadVertex(p:vec3u)->f32{return fixtureValue(p,0u);}
fn umLoadCorner(p:vec3u)->f32{return fixtureValue(p,101u);}
${uniformMixedVertexSamplingSource("", true, undefined, "umLoadCorner")}
${narrowBandMembershipWGSL.slice(start, end)}
fn sampleClass(v:f32)->u32{return u32(v<=0.0)|(u32(v>=0.0)<<1u)|(u32(v==0.0)<<2u);}
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) gid:vec3u){
 let index=gid.x;
 if(index<${vertices * patterns}u){
  fixture=index/${vertices}u;let i=index%${vertices}u;
  let p=vec3u(i%(UM_D.x+1u),(i/(UM_D.x+1u))%(UM_D.y+1u),i/((UM_D.x+1u)*(UM_D.y+1u)));
  let a=umSampleVertex(vec3f(p));let b=nbCrossingVertex(p);
  atomicAdd(&counters[0],1u);
  if(bitcast<u32>(a)!=bitcast<u32>(b)){atomicAdd(&counters[1],1u);}
  if(a!=b){atomicAdd(&counters[2],1u);}
  if(sampleClass(a)!=sampleClass(b)){atomicAdd(&counters[3],1u);}
 }
 if(index<${cells * patterns}u){
  fixture=index/${cells}u;let i=index%${cells}u;
  let p=vec3u(i%UM_D.x,(i/UM_D.x)%UM_D.y,i/(UM_D.x*UM_D.y));
  var low=vec2f(3e38);var high=vec2f(-3e38);
  for(var k=0u;k<8u;k++){let q=p+umCorner(k,2u);let v=vec2f(umSampleVertex(vec3f(q)),nbCrossingVertex(q));low=min(low,v);high=max(high,v);}
  let crossing=(low<=vec2f(0))&(high>=vec2f(0))&(low<high);
  atomicAdd(&counters[4],1u);if(crossing.x!=crossing.y){atomicAdd(&counters[5],1u);}
 }
 if(index<${tiles * patterns}u){
  fixture=index/${tiles}u;let i=index%${tiles}u;
  let p=4u*vec3u(i%UM_T.x,(i/UM_T.x)%UM_T.y,i/(UM_T.x*UM_T.y));
  var low=vec2f(3e38);var high=vec2f(-3e38);
  for(var k=0u;k<8u;k++){let q=p+4u*umCorner(k,2u);let v=vec2f(umSampleVertex(vec3f(q)),nbCrossingVertex(q));low=min(low,v);high=max(high,v);}
  let candidate=(low<=vec2f(0))&(high>=vec2f(0));
  atomicAdd(&counters[6],1u);if(candidate.x!=candidate.y){atomicAdd(&counters[7],1u);}
 }
}
`, "check").code;

async function run() {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href); Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]), adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(), errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const counters = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const module = device.createShaderModule({ code: source }), info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(message => message.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "check" } });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: counters } }] });
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(Math.ceil(vertices * patterns / 64)); pass.end();
    encoder.copyBufferToBuffer(counters, 0, read, 0, 32); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const counts = [...new Uint32Array(read.getMappedRange())]; read.unmap();
    const result = { date: new Date().toISOString(), dims, patterns, vertices: counts[0], signedZeroWordDifferences: counts[1], nonzeroValueDifferences: counts[2], vertexClassDifferences: counts[3], cells: counts[4], crossingDifferences: counts[5], tiles: counts[6], candidateDifferences: counts[7], errors };
    const out = process.argv.find(arg => arg.startsWith("--out="))?.slice(6) ?? "artifacts/nbflip-ui-perf/integer-crossing-endpoints.json";
    mkdirSync(dirname(out), { recursive: true }); writeFileSync(out, JSON.stringify(result, null, 2) + "\n"); console.log(JSON.stringify(result));
    assert.equal(counts[0], vertices * patterns); assert.equal(counts[4], cells * patterns); assert.equal(counts[6], tiles * patterns);
    for (const at of [2, 3, 5, 7]) assert.equal(counts[at], 0); assert.deepEqual(errors, []);
  } finally { read.destroy(); counters.destroy(); device.destroy(); }
}
if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await run();
