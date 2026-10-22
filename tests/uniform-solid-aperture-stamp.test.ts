import assert from "node:assert/strict";
import test from "node:test";
import { uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

// Exercise the production address and both pressure-cache readers with a
// tiny record, independently of the full solver's compilation and page pool.
function functionSource(source: string, name: string): string {
  const start = source.indexOf(`fn ${name}(`);
  assert.ok(start >= 0);
  let depth = 0;
  for (let i = source.indexOf("{", start); i < source.length; i++) {
    if (source[i] === "{") depth++;
    if (source[i] === "}" && --depth === 0) return source.slice(start, i + 1);
  }
  throw new Error(`Unclosed WGSL function ${name}`);
}

test("solid aperture records survive window recentering but reject reused tiles and old epochs", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("world-space solid aperture stamps", async device => {
    const source = uniformMixedSolidWGSL(0, 1792);
    const readers = ["umSolidApertureCoord", "umSolidAperturesServe", "umPressureFaceVRecorded"]
      .map(name => functionSource(source, name)).join("\n");
    const output = device.createBuffer({ size: 6 * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    try {
      const module = device.createShaderModule({ code: `
var<private> frame:vec3i;var<private> tile:vec3u;
var<private> umSolidRecord:array<vec4f,62>;
struct Params{cellGravity:vec4f};const umSolidParams=Params(vec4f(1));
fn umFramePage()->vec3i{return frame;}
fn umTileCoord(t:u32)->vec3u{return tile;}
fn umTileKey(p:vec3i)->u32{return 0u;}
fn umKeyStored(t:u32)->bool{return true;}
fn umSolidEnabled()->bool{return true;}
fn umBodyCount()->u32{return 0u;}
fn umSolidBodyBuilt(t:u32)->bool{return false;}
fn umBodyPoseNearTile(t:u32,margin:u32)->bool{return false;}
fn umAnyBodyNear(p:vec3f,margin:f32)->bool{return false;}
fn umSolidFaceWorld(p:vec3i,axis:u32)->vec3f{return vec3f(p);}
fn umSolidApertureRow(t:u32)->u32{return 0u;}
fn umSolidApertureEpochRow()->u32{return 61u;}
fn umSolidApertureCell(p:vec3i,axis:u32)->vec3i{return p;}
fn umSolidApertureAt(rows:u32,p:vec3i,axis:u32)->f32{return .25;}
fn umPressureFaceV(p:vec3i,axis:u32)->f32{return .75;}
${readers}
@group(0) @binding(0) var<storage,read_write> result:array<vec2f>;
@compute @workgroup_size(1) fn probe(){
 for(var c=0u;c<6u;c++){
  frame=vec3i(-3,-2,-1);tile=vec3u(8,16,24);
  umSolidRecord[61]=vec4f(1,0,0,0);
  umSolidRecord[48]=vec4f(vec3f(umSolidApertureCoord(0u)),1);
  if(c==1u||c==4u||c==5u){frame-=vec3i(27,30,28);tile+=vec3u(216,240,224);}
  if(c==2u){frame+=vec3i(1,1,1);tile-=vec3u(8);}
  if(c==3u||c==4u){tile.x+=8u;}
  if(c==5u){umSolidRecord[61].x=2.0;}
  result[c]=vec2f(f32(umSolidAperturesServe(vec3i(tile*4u))),umPressureFaceVRecorded(vec3i(tile*4u),0u));
 }
}` });
      const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "probe" } });
      const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
      const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end(); device.queue.submit([encoder.finish()]);
      assert.deepEqual(Array.from(await readMixedBuffer(device, output)), [0, .25, 0, .25, 0, .25, 2, -1, 2, -1, 2, -1]);
    } finally { output.destroy(); }
  });
});
