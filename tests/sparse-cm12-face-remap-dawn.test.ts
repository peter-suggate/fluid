import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";

const source = readFileSync(process.env.CM12_FACE_SOURCE ?? new URL(
  "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
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

// September 6 collocated remapping deliberately smooths staggered modes.
// Test its actual interpolation/characteristic contract, rather than the
// native-face identity contract of the removed reconstruction algorithm.
dawnTest("collocated face transport preserves constants, advects waves, and supplies dry receivers", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "face-remap-collocated");
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
struct Params { dimensions:vec4u, frame:vec4f, refinementRegionControl:vec4u }
const FACE_VELOCITY_SUPPORT=24u;
var<private>p:Params;
struct FaceVelocitySupport { velocity:vec3f, spans:vec3f, owner:bool, extended:bool, liquid:bool }
var<private>width:f32;
var<private>mode:u32;
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
fn rowCenter(row:u32)->vec3f{_=row;return vec3f(24+.5*width,32+.5*width,32);}
fn rowArea(row:u32)->f32{_=row;return select(width*width,0.0,mode==5u);}
fn rowDistance(row:u32)->f32{_=row;return width;}
fn rowAxis(row:u32)->u32{_=row;return 2u;}
fn rowOpenFraction(row:u32)->f32{_=row;return 0.25;}
fn rowSolidVelocity(row:u32)->f32{_=row;return -2.0;}
fn hasSolidBoundaries()->bool{return mode>=4u;}
fn clipBoundarySegment(start:vec3f,end:vec3f)->vec3f{_=start;return end;}
fn destinationFaceVelocity()->u32{return 0u;}
fn rowTermOffset(row:u32)->u32{return 2u*row;}
fn rowTermCount(row:u32)->u32{_=row;return 2u;}
fn rowTermRange(row:u32)->vec2u{let first=rowTermOffset(row);return vec2u(first,first+rowTermCount(row));}
fn termCell(term:u32)->u32{return term;}
fn cellBrick(cell:u32)->u32{_=cell;return 0u;}
fn cachedRefinementPolicyTileScale(brick:u32)->u32{_=brick;return u32(width);}
fn cellMinimumWidth(cell:u32)->f32{_=cell;return width;}
fn faceVelocitySupportAt(q:vec3i)->FaceVelocitySupport{
  let x=(floor(f32(q.x)/width)+.5)*width;
  var v=vec3f(3,0,sin(6.28318530718*x/32.0));
  if(mode>0u){v=vec3f(3,0,7);}
  return FaceVelocitySupport(v,vec3f(width),true,mode!=3u,mode!=2u);
}
${["sampleFaceVelocitySupport", "sampleFaceVelocitySupportAtSpans", "traceFaceDeparture",
  "traceFaceDepartureAtSpans", "finishTransportFaceRow", "prepareTransportFaceRow"].map(production).join("\n")}
@compute @workgroup_size(24)
fn main(@builtin(global_invocation_id)gid:vec3u){
  let row=gid.x;width=f32(1u<<(row%4u));mode=row/4u;
  p=Params(vec4u(128),vec4f(width/6.0,0,0,0),vec4u(1));
  let flags=1u|select(0u,2u,mode!=3u)|select(0u,4u,mode!=2u);
  state[24u+8u*row+3u]=width+f32(flags)/8.0;
  state[24u+8u*row+7u]=width+f32(flags)/8.0;
  prepareTransportFaceRow(row);
}`;
    const values = await execute(device, fixture, 24+8*24);
    for (let i = 0; i < 4; i++) {
      const width = 1 << i;
      const expected = .5 * (Math.sin(2*Math.PI*(24+.5*width)/32)
        + Math.sin(2*Math.PI*(24-.5*width)/32));
      assert.ok(Math.abs(values[i]! - expected) < 2e-6, `width ${width}: half-cell wave translation`);
      assert.equal(values[4+i], 7, "constant velocity");
      assert.equal(values[8+i], 7, "dry receiver uses extended velocity");
      assert.equal(values[12+i], 0, "unsupported row stays zero");
      assert.equal(values[16+i], .25, "open fraction blends with solid velocity");
      assert.equal(values[20+i], -2, "closed face follows solid velocity");
    }
    const ordinary = fixture.replace("vec4f(width/6.0,0,0,0),vec4u(1)", "vec4f(width/12.0,0,0,0),vec4u(0)");
    const quarter = await execute(device, ordinary, 24+8*24);
    for (let i=0;i<4;i++) {
      const width=1<<i;
      const expected=.75*Math.sin(2*Math.PI*(24+.5*width)/32)
        +.25*Math.sin(2*Math.PI*(24-.5*width)/32);
      assert.ok(Math.abs(quarter[i]!-expected)<2e-6, `width ${width}: ordinary quarter-cell translation`);
    }
    assert.deepEqual(errors, []);
  } finally {
    device?.destroy(); await releaseWebGPUExclusiveLock(); if (gpu) live.delete(gpu);
  }
});
