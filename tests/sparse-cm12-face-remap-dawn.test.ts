import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { runInNewContext } from "node:vm";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(process.env.CM12_FACE_SOURCE ?? new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
function production(name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  assert.ok(start >= 0, name);
  let depth = 0;
  for (let at = source.indexOf("{", start); at < source.length; at++) {
    if (source[at] === "{") depth++;
    if (source[at] === "}" && --depth === 0) return source.slice(start, at + 1);
  }
  throw new Error(`Unclosed production function ${name}`);
}
// Evaluate the production string transformation, so this fixture exercises
// the same relative-coordinate basis as the real generated shader.
const relativeExpression = source.match(
  /const relativeTransportStencil = ([\s\S]*?);\n/)?.[1];
assert.ok(relativeExpression);
const relativeStencil = runInNewContext(relativeExpression, {
  geometricTransportStencil: production("effectiveTransportStencilAtSpansMode"),
});
async function execute(device: GPUDevice, code: string, floats: number): Promise<Float32Array> {
  const shader = device.createShaderModule({ code });
  assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shader, entryPoint: "main" } });
  const bytes = floats * 4;
  const output = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const copy = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
  const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(output, 0, copy, 0, bytes); device.queue.submit([encoder.finish()]);
  await copy.mapAsync(GPUMapMode.READ); const values = new Float32Array(copy.getMappedRange()).slice();
  copy.unmap(); copy.destroy(); output.destroy();
  return values;
}
const live = new Set<GPU>();
const dawnTest = process.env.WEBGPU_NODE_MODULE ? test : test.skip;

dawnTest("a zero-length face characteristic preserves staggered modes at every width", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "face-remap-identity");
  let gpu: GPU | undefined, device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => errors.push(e.error.message));
    const fixture = `
struct Params { dimensions:vec4u, frame:vec4f, refinementRegionControl:vec4u, surfaceProof:vec4u }
const p=Params(vec4u(128),vec4f(0,.05,0,0),vec4u(0),vec4u(0));
const CM12_LIQUID_ISOVALUE=0.5;
struct TransportStencil { cells:array<u32,8>, weights:array<f32,8> }
struct Owner { cell:u32 }
struct TransportFaceSupport { width:f32, extended:bool, liquid:bool }
struct FaceVelocitySupport { velocity:vec3f, spans:vec3f, owner:bool, extended:bool, liquid:bool }
var<private>width:f32;
var<private>front:bool;
var<private>oneWet:bool;
var<private>currentRow:u32;
const INVALID=0xffffffffu;
const FACE_VELOCITY_SUPPORT=48u;
const FIXTURE_CELLS_PER_ROW=4096u;
fn cm12RecordFailure(code:u32,cell:u32,data:vec4u){_=code;_=cell;_=data;}
fn cm12ClampToResidentWorld(q:vec3f,margin:vec3f)->vec3f{
  return clamp(q,margin,vec3f(p.dimensions.xyz)-margin);
}
fn cm12WorldFineLower()->vec3f{return vec3f(0.0);}
fn cm12WorldFineUpper()->vec3f{return vec3f(p.dimensions.xyz);}
fn cellWidths(cell:u32)->vec3f{_=cell;return vec3f(width);}
fn cellTransportActive(cell:u32)->bool{return cell<12u*FIXTURE_CELLS_PER_ROW;}
fn fixtureCellOrigin()->vec3i{return vec3i(floor(rowCenter(currentRow)/width))-vec3i(8);}
fn cellCenter(cell:u32)->vec3f{
  let local=cell%FIXTURE_CELLS_PER_ROW;
  let q=vec3i(i32(local%16u),i32((local/16u)%16u),i32(local/256u));
  return (vec3f(fixtureCellOrigin()+q)+vec3f(.5))*width;
}
fn cm12TransportOwnerAtFine(q:vec3i,direct:bool)->Owner{
  _=direct;let coordinate=vec3i(floor(vec3f(q)/width));
  let local=coordinate-fixtureCellOrigin();
  if(any(local<vec3i(0))||any(local>=vec3i(16))){return Owner(INVALID);}
  let cell=currentRow*FIXTURE_CELLS_PER_ROW+u32(local.x+16*(local.y+16*local.z));
  // Seed the frozen cell-indexed support cache lazily from this fixture's
  // analytic field. Each invocation owns a disjoint cache, so the production
  // geometric sampler reads exactly the field used by its face-support query.
  let support=faceVelocitySupportAt(vec3i(floor(cellCenter(cell))));
  let at=FACE_VELOCITY_SUPPORT+4u*cell;
  state[at]=support.velocity.x;state[at+1u]=support.velocity.y;state[at+2u]=support.velocity.z;
  return Owner(cell);
}
fn ownerCellAt(q:vec3i)->u32{_=q;return 0u;}
// This analytic fixture has no IBO image. The full solver A/B covers that
// address path; retain the identical incidence fallback here.
const FIXTURE_USE_IBO=false;
fn ta(at:u32)->u32{_=at;return select(0u,1u,FIXTURE_USE_IBO);}
fn cm12IBOAcceptedSlot()->u32{return 0u;}
fn cm12IBOLeafActive(slot:u32,brick:u32)->bool{_=slot;_=brick;return FIXTURE_USE_IBO;}
fn cm12IBOLeafCellFirst(slot:u32,brick:u32)->u32{_=slot;_=brick;return 0u;}
fn cm12IBOLeafDimensions(slot:u32,brick:u32)->vec3u{_=slot;_=brick;return vec3u(1);}
fn cm12IBOTRAPacketForLocal(brick:u32,local:vec3u,slot:u32)->vec2u{
  _=brick;_=local;_=slot;return vec2u(0u);
}
fn itr1StableRowForOwner(packet:u32,axis:u32,lane:u32)->u32{
  _=packet;_=axis;_=lane;return currentRow;
}
fn incidenceBegin(cell:u32)->u32{_=cell;return 0u;}
fn incidenceEnd(cell:u32)->u32{_=cell;return 1u;}
fn incidenceRow(at:u32)->u32{_=at;return currentRow;}
fn rowAccepted(row:u32)->bool{_=row;return true;}
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
fn rowCenter(row:u32)->vec3f{_=row;return vec3f(24,32+.5*width,32+.5*width);}
fn rowArea(row:u32)->f32{_=row;return width*width;}
fn rowDistance(row:u32)->f32{_=row;return width;}
fn rowAxis(row:u32)->u32{_=row;return 0u;}
fn rowKind(row:u32)->u32{_=row;return 0u;}
fn rowOpenFraction(row:u32)->f32{_=row;return 1.0;}
fn rowSolidVelocity(row:u32)->f32{_=row;return 0.0;}
fn hasSolidBoundaries()->bool{return false;}
fn clipBoundarySegment(start:vec3f,end:vec3f)->vec3f{_=start;return end;}
fn sourceFaceVelocity()->u32{return 0u;}
fn destinationFaceVelocity()->u32{return 12u;}
fn sourceDensity()->u32{return 24u;}
fn cm12ExtendedCellSelected(cell:u32)->bool{_=cell;return true;}
fn transportSourceSamplingSpans(cell:u32,direct:bool)->vec3f{_=direct;return vec3f(cellMinimumWidth(cell));}
fn rowTermOffset(row:u32)->u32{return 2u*row;}
fn rowTermCount(row:u32)->u32{_=row;return 2u;}
fn termCell(term:u32)->u32{return term;}
fn cellBrick(cell:u32)->u32{_=cell;return 0u;}
fn cachedRefinementPolicyTileScale(brick:u32)->u32{_=brick;return 1u;}
fn cellMinimumWidth(cell:u32)->f32{_=cell;return width;}
fn velocity(q:vec3f)->vec3f{
  let k=6.28318530718/32.0;
  return vec3f(sin(k*q.x)*cos(k*q.z),0.0,-cos(k*q.x)*sin(k*q.z));
}
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
  let center=(floor(vec3f(q)/width)+.5)*width;
  // Exact collocation of the divergence-free staggered Taylor-Green mode.
  let collocated=velocity(center)*cos(3.14159265359*width/32.0);
  return FaceVelocitySupport(select(collocated,vec3f(3,0,0),front),vec3f(width),true,true,!front&&(!oneWet||q.x>=24));
}
${relativeStencil}
${["clampRelativeTransportPosition", "clipRelativeTransportSegment",
  "orderedScalarPair", "orderedVectorPair", "transportScalarSum", "transportVectorSum",
  "sampleRelativeFaceVelocity", "traceRelativeFaceDisplacement", "sampleRelativeNativeTransportFace",
  "transferLocalCoordinate", "nativeTransportFaceValue", "nativeTransportFaceAt",
  "finishTransportFaceRow", "transportFaceSupport", "transportFaceSamplingSpans", "prepareTransportFaceRow"].map(production).join("\n")}
@compute @workgroup_size(12)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;width=f32(1u<<(row%4u));front=row>=8u;oneWet=row>=4u;currentRow=row;
  state[row]=select(velocity(rowCenter(row)).x,0.0,front);
  state[24u+2u*row]=select(1.0,0.0,front||oneWet);
  state[25u+2u*row]=select(1.0,0.0,front);
  prepareTransportFaceRow(row);
}`;
    const fixtureFloats=48+4*12*4096;
    const values = await execute(device, fixture, fixtureFloats);
    console.log(JSON.stringify({ widths: [1, 2, 4, 8], before: [...values.slice(0, 4)], after: [...values.slice(12, 16)] }));
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(values[i]! - values[12 + i]!) < 1e-6,
      `width ${1 << (i % 4)}: a zero-length characteristic must not dissipate the face mode`);
    for (let i = 8; i < 12; i++) assert.equal(values[12 + i], 3,
      "new dry receiver faces must acquire extended jet velocity, not retain their old zero");
    const addressed = await execute(device, fixture.replace(
      "const FIXTURE_USE_IBO=false;", "const FIXTURE_USE_IBO=true;"), fixtureFloats);
    assert.deepEqual(addressed.slice(12, 24), values.slice(12, 24),
      "accepted ITR face addresses preserve the incidence-path staggered authority");
    const brickFaces = await execute(device, fixture.replace(
      "fn rowKind(row:u32)->u32{_=row;return 0u;}",
      "fn rowKind(row:u32)->u32{_=row;return 1u;}"), fixtureFloats);
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(brickFaces[i]! - brickFaces[12 + i]!) < 1e-6,
      `equal-width brick face ${i}: B1 must receive the same zero-time identity`);
    // A mixed port is located at the coarse patch center. Its finer positive
    // owner still lists that authoritative face in its incidence rows.
    const mixedFixture = fixture.replace(
      "fn rowKind(row:u32)->u32{_=row;return 0u;}",
      "fn rowKind(row:u32)->u32{_=row;return 2u;}")
      .replace("fn cellMinimumWidth(cell:u32)->f32{_=cell;return width;}",
        "fn cellMinimumWidth(cell:u32)->f32{return select(width,.5*width,cell%2u==0u);}")
      .replace("vec3f(width),true,true,!front", "vec3f(select(width,.5*width,q.x>=24)),true,true,!front");
    const mixedFaces = await execute(device, mixedFixture, fixtureFloats);
    for (let i = 0; i < 8; i++) assert.ok(Math.abs(mixedFaces[i]! - mixedFaces[12 + i]!) < 1e-6,
      `mixed face ${i}: a zero-time coarse/fine port must retain its staggered value`);
    // Finite characteristics: a divergence-free transverse wave translated
    // by a constant x velocity. A one-cell translation must move the wave,
    // while a half-cell translation is bounded linear interpolation.
    let moving = fixture.replace("const p=Params(vec4u(128),vec4f(0,.05,0,0),vec4u(0),vec4u(0));",
      "var<private>p:Params;");
    moving = moving.replace(production("nativeTransportFaceAt"), `
fn nativeTransportFaceAt(position:vec3f,axis:u32,width:f32)->vec2f{
  _=axis;_=width;return vec2f(sin(6.28318530718*position.x/32.0),1.0);
}`);
    moving = moving.replace("return 0u;}\nfn rowKind", "return 2u;}\nfn rowKind");
    moving = moving.replace("vec3f(24,32+.5*width,32+.5*width)", "vec3f(24+.5*width,32+.5*width,32)");
    const support = moving.match(/fn faceVelocitySupportAt\([\s\S]*?\n}/)![0];
    moving = moving.replace(support, `
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
  let x=(floor(f32(q.x)/width)+.5)*width;
  return FaceVelocitySupport(vec3f(3,0,sin(6.28318530718*x/32.0)),vec3f(width),true,true,true);
}`);
    const entry = moving.indexOf("@compute @workgroup_size(12)");
    moving = moving.slice(0, entry) + `
@compute @workgroup_size(12)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;width=f32(1u<<(row%4u));currentRow=row;
  let dt=f32(row/4u)*width/6.0;
  p=Params(vec4u(128),vec4f(dt,.05,0,0),vec4u(0),vec4u(0));
  state[24u+2u*row]=1.0;state[25u+2u*row]=1.0;
  prepareTransportFaceRow(row);
}`;
    const transported = await execute(device, moving, fixtureFloats);
    for (let i = 0; i < 12; i++) {
      const width = 1 << (i % 4), travel = Math.floor(i / 4) / 2;
      const here = Math.sin(2 * Math.PI * (24 + .5 * width) / 32);
      const upstream = Math.sin(2 * Math.PI * (24 - .5 * width) / 32);
      const expected = (1 - travel) * here + travel * upstream;
      assert.ok(Math.abs(transported[12 + i]! - expected) < 2e-6,
        `width ${width}, travel ${travel}: native wave translation`);
    }
    assert.deepEqual(errors, []);
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
