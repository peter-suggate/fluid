import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
const production = (name: string) => {
  const code = source.match(new RegExp(`fn ${name}\\([\\s\\S]*?\\n}`))?.[0];
  assert.ok(code, name); return code;
};
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("candidate face remap preserves signed-world flux and advances by donor rectangles", {
  timeout: 30_000,
}, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "candidate-face-world");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", event => errors.push(event.error.message));
    const code = `
const INVALID=0xffffffffu;
const FACE_COUNT=24576u;
const BRICK_FINE_RESOLUTION=8u;
struct Params { dimensions:vec4u }
const p=Params(vec4u(8));
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
var<private>query:u32;
var<private>calls:u32;
fn brickHasUnclippedWorldGeometry(brick:u32)->bool{return brick!=0u;}
fn cm12WorldOwnerAt(q:vec3i)->u32{
  if(any(q<vec3i(-1))||any(q>=vec3i(3))){return INVALID;}
  return select(1u,0u,all(q==vec3i(0)));
}
fn cm12WorldFloorToSpan(q:i32,span:i32)->i32{return i32(floor(f32(q)/f32(span)))*span;}
${production("presentationCanonicalCoarseCoordinate")}
fn ownerCellAt(q:vec3i)->u32{
  if(any(q<vec3i(-8))||any(q>=vec3i(24))){return INVALID;}
  let local=vec3u(q+vec3i(8))/2u;
  return local.x+16u*(local.y+16u*local.z);
}
fn cellCenter(cell:u32)->vec3f{
  return vec3f(-7)+2.0*vec3f(f32(cell%16u),f32((cell/16u)%16u),f32(cell/256u));
}
fn cellWidths(cell:u32)->vec3f{return vec3f(select(2.0,4.0,cell==4096u));}
fn incidenceBegin(cell:u32)->u32{return 6u*cell;}
fn incidenceEnd(cell:u32)->u32{return 6u*cell+6u;}
fn incidenceRow(at:u32)->u32{return at;}
fn incidenceTerm(at:u32)->u32{return at;}
fn acceptedRowMember(row:u32)->bool{return row<FACE_COUNT;}
fn rowAxis(row:u32)->u32{return select((row%6u)/2u,query%3u,row>=FACE_COUNT);}
fn rowCenter(row:u32)->vec3f{
  if(row<FACE_COUNT){var q=cellCenter(row/6u);q[rowAxis(row)]+=select(-1.0,1.0,(row&1u)!=0u);return q;}
  var q=vec3f(17);let axis=query%3u;
  if(query<6u){q[axis]=select(20.0,-4.0,query>=3u);}
  else if(query<12u){q=vec3f(1);q[axis]=select(24.0,-8.0,query>=9u);}
  else{q=vec3f(30);}
  return q;
}
fn termCoefficient(term:u32)->f32{_=term;return 1.0;}
fn rowStaticDualWeight(row:u32)->f32{_=row;return 1.0;}
fn destinationFaceVelocity()->u32{return 0u;}
fn rowTermOffset(row:u32)->u32{_=row;return 0u;}
fn rowTermCount(row:u32)->u32{_=row;return 1u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{_=term;return 4096u;}
${production("candidateAcceptedFaceSample").replace("fn candidateAcceptedFaceSample", "fn sampleProduction")}
fn candidateAcceptedFaceSample(point:vec3f,axis:u32)->vec4f{
  calls+=1u;return sampleProduction(point,axis);
}
${production("candidateRemappedFaceVelocity")}
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index)lane:u32){
  for(var row=lane;row<FACE_COUNT;row+=32u){state[row]=3.25;}
  storageBarrier();if(lane>=15u){return;}query=lane;calls=0u;
  let row=FACE_COUNT+lane;let center=rowCenter(row);let axis=rowAxis(row);
  let sample=sampleProduction(center,axis);
  let velocity=candidateRemappedFaceVelocity(row);
  let at=FACE_COUNT+12u*lane;
  state[at]=velocity;state[at+1u]=f32(calls);state[at+2u]=sample.x;
  state[at+3u]=sample.y;state[at+4u]=sample.z;state[at+5u]=sample.w;
  state[at+6u]=center[(axis+1u)%3u];state[at+7u]=center[(axis+2u)%3u];
  var coordinate=vec3i(1);coordinate[axis]=select(-2,10,query>=3u);
  if(query>=6u){coordinate=vec3i(-32,32,-32);}
  let canonical=presentationCanonicalCoarseCoordinate(coordinate,1u,select(0u,1u,query>=9u));
  state[at+8u]=f32(canonical.x);state[at+9u]=f32(canonical.y);state[at+10u]=f32(canonical.z);
}`;
    const shader = device.createShaderModule({ code });
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
    const output = device.createBuffer({ size: 4 * (24576 + 12 * 15),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const copy = device.createBuffer({ size: 4 * 12 * 15, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: output } }] });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
    encoder.copyBufferToBuffer(output, 4 * 24576, copy, 0, copy.size); device.queue.submit([encoder.finish()]);
    await copy.mapAsync(GPUMapMode.READ); const values = new Float32Array(copy.getMappedRange()).slice();
    copy.unmap(); copy.destroy(); output.destroy();
    for (let query = 0; query < 15; query++) {
      const at = query * 12, axis = query % 3;
      assert.equal(values[at], query < 12 ? 3.25 : 0, `query ${query}: constant normal flux or missing air`);
      assert.ok(values[at + 1]! <= 25, `query ${query}: ${values[at + 1]} samples must follow actual donor patches`);
      assert.ok(values[at + 3 + (axis + 1) % 3]! > values[at + 6]!, `query ${query}: first tangent advances geometrically`);
      assert.ok(values[at + 3 + (axis + 2) % 3]! > values[at + 7]!, `query ${query}: second tangent advances geometrically`);
      const expected = query < 6 ? [1, 1, 1] : query < 9 ? [0, 7, 0] : [-32, 32, -32];
      if (query < 6) expected[axis] = query < 3 ? -2 : 10;
      assert.deepEqual(Array.from(values.subarray(at + 8, at + 11)), expected,
        `query ${query}: presentation continues clipped boundaries while respecting world geometry`);
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu); }
});
