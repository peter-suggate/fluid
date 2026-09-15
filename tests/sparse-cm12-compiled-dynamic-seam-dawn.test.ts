import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { SPARSE_CM12_DYNAMIC_SEAM_BINDING_WGSL } from "../lib/methods/adaptive-volume/sparse-cm12-dynamic-seam-binding.wgsl";
import { dynamicRungLayout, dynamicRowTermOffset, dynamicRungLayoutWGSL, packDynamicSeamCatalogue, preparedDynamicSeamCatalogue } from "../lib/methods/adaptive-volume/sparse-cm12-dynamic-rung-catalog";

const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("prepared dynamic seams bind every rung and face; absent backing fails closed", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "compiled-dynamic-seam");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice(); assert.ok(device);
    const queries: number[] = [], expected: number[][] = [];
    const bits = (v: number) => new Uint32Array(new Float32Array([v]).buffer)[0]!;
    for (const seam of preparedDynamicSeamCatalogue().values()) {
      for (const row of seam.rows) {
        queries.push(seam.own, seam.neighbor, seam.side, 0, ...row.center.map(bits), 0);
        const rung = dynamicRungLayout(seam.own);
        expected.push([1000 + rung.rowOffset + row.row,
          2000 + dynamicRowTermOffset(seam.own, row.row), row.terms.length,
          ...row.terms.map(t => (t.neighbor ? 0 : 100) + dynamicRungLayout(t.neighbor ? seam.neighbor : seam.own).cellOffset + t.local)]);
      }
    }
    for (const [own, other, corrupt] of [[8, 1, 0], [2, 8, 0], [4, 4, 1], [3, 4, 0]]) {
      queries.push(own!, other!, 1, corrupt!, bits(8), bits(1), bits(1), 0);
      expected.push([0xffff_ffff]);
    }
    const helper = SPARSE_CM12_DYNAMIC_SEAM_BINDING_WGSL.split("fn compiledHostDynamicSeam")[0];
    const module = device.createShaderModule({ code: `
const INVALID=0xffffffffu; const CM12_WDR_INITIAL_LEAVES=1u; const CM12_DYNAMIC_SEAM_CATALOGUE=64u;
struct Params { dispatch:vec4u }; var<private>p:Params;
@group(0)@binding(0)var<storage,read>words:array<u32>;
@group(0)@binding(1)var<storage,read>queries:array<u32>;
@group(0)@binding(2)var<storage,read_write>result:array<u32>;
var<private>query:u32;
fn q(i:u32)->u32{return queries[8u*query+i];}
fn ta(i:u32)->u32{if(i==0u){return 1u;}if(i==3u){return 1000u;}if(i==4u){return 2000u;}
 if(i==19u&&q(3u)!=0u){return 0u;}return words[i];}
${dynamicRungLayoutWGSL()}
fn validBrickResolution(r:u32)->bool{return r>0u&&r<=8u&&(r&(r-1u))==0u;}
fn brickSpan(leaf:u32)->u32{return 1u;}
fn candidateTopologyPageBase(page:u32)->u32{return 16u;}
fn scheduledBrickActive(leaf:u32)->bool{return q(1u)!=0u;}
fn scheduledBrickResolution(leaf:u32)->u32{return select(q(1u),q(0u),leaf==1u);}
fn cm12WorldLeafCoordinate(leaf:u32)->vec3i{var d=vec3i(0);if(leaf==0u){d[q(2u)/2u]=select(-1,1,(q(2u)&1u)!=0u);}return d;}
fn cm12WorldOwnerAt(coord:vec3i)->u32{return select(INVALID,0u,q(1u)!=0u&&all(coord==cm12WorldLeafCoordinate(0u)));}
fn templateBrickCellRange(leaf:u32,r:u32)->vec2u{return vec2u(select(0u,100u,leaf==1u)+cm12DynamicCellOffset(r),r*r*r);}
${helper}
@compute @workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3u){
 query=gid.x;if(query>=${expected.length}u){return;}p.dispatch=vec4u(0u,0u,0u,2u);
 let point=bitcast<vec3f>(vec3u(q(4u),q(5u),q(6u)));
 let binding=cm12PreparedDynamicFace(1u,q(0u),q(2u),point);let base=8u*query;
 result[base]=binding.x;if(binding.x==INVALID){return;}
 result[base+1u]=binding.y;let count=ta(binding.z+6u);result[base+2u]=count;
 for(var t=0u;t<count;t+=1u){result[base+3u+t]=cm12PreparedDynamicTerm(binding,t);}
}` });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const catalogue = packDynamicSeamCatalogue();
    const words = new Uint32Array(64 + catalogue.length); words.set(catalogue, 64);
    words[16] = 1; words[19] = 0x8000001f; words[20] = 2010; words[21] = 3510;
    const upload = (data: Uint32Array) => { const b = device!.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST }); device!.queue.writeBuffer(b, 0, new Uint32Array(data)); return b; };
    const topology = upload(words), input = upload(new Uint32Array(queries));
    const output = device.createBuffer({ size: expected.length * 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readback = device.createBuffer({ size: output.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
    const bindings = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [topology, input, output].map((buffer, binding) => ({ binding, resource: { buffer } })) });
    const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(expected.length / 64)); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, output.size);
    device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const results = new Uint32Array(readback.getMappedRange());
    expected.forEach((wanted, query) => assert.deepEqual([...results.slice(8 * query, 8 * query + wanted.length)], wanted, `prepared face ${query}`));
    readback.unmap(); for (const buffer of [topology, input, output, readback]) buffer.destroy();
  } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
