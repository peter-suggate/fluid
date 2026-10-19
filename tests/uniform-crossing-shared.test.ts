import assert from 'node:assert/strict';
import test from 'node:test';
import { narrowBandMembershipWGSL } from '../lib/methods/uniform/uniform-narrow-band-membership.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';

test('shared crossing samples match scalar sampling across mixed owners, cold fields and page walls',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('shared crossing',async device=>{
  const original=narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf('fn nbCrossingVertex'),narrowBandMembershipWGSL.indexOf('// A 4h tile among'));
  const shared=narrowBandMembershipWGSL.slice(narrowBandMembershipWGSL.indexOf('var<workgroup> nbSeedOwner'),narrowBandMembershipWGSL.indexOf('fn nbMetricPlaneMissing'));
  const code=`
@group(0) @binding(0) var<storage,read> values:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<u32>;
override nbColdCurrentPhi:bool=false;
var<private> testCase:u32;
var<workgroup> nbBandVertices:array<f32,125>;
var<workgroup> nbCrossingFinite:atomic<u32>;
fn umCorner(k:u32,n:u32)->vec3u{return vec3u(k%n,(k/n)%n,k/(n*n));}
fn umCellNearFromPage(b:u32,page:vec3i,p:vec3i)->vec3i{
 let wall=vec3<bool>((testCase&1u)!=0u,(testCase&2u)!=0u,(testCase&4u)!=0u);
 return min(p,select(vec3i(39),vec3i(31),wall));
}
fn umTileAt(t:vec3u)->u32{return t.x+10u*(t.y+10u*t.z);}
fn umTileWidth(t:u32)->u32{
 if(testCase<8u){return 4u;}if(testCase<16u){return 1u;}
 return select(1u,4u,((t*17u+testCase*31u)%(1u+testCase%7u))==0u);
}
fn nbTileCold(t:u32)->bool{return ((t*3u+testCase)%5u)==0u;}
fn at(p:vec3u)->u32{return p.x+40u*(p.y+40u*p.z);}
fn umLoadVertex(p:vec3u)->f32{return values[at(p)];}
fn umLoadCorner(p:vec3u)->f32{return values[64000u+at(p)];}
fn testCurrent(p:vec3u)->f32{return values[128000u+at(p)];}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{var a:array<f32,8>;for(var k=0u;k<8u;k++){a[k]=umLoadCorner(p+s*umCorner(k,2u));}return a;}
fn umVertexSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
`+(original+shared).replaceAll('textureLoad(outputPhi,vec3i(p)).x','testCurrent(p)')+`
@compute @workgroup_size(128) fn compare(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 testCase=group.x;let origin=select(vec3u(12),vec3u(28),testCase%3u!=0u);let tile=umTileAt(origin/4u);
 if(lane==0u){atomicStore(&nbCrossingFinite,1u);}workgroupBarrier();
 nbGatherCrossing(tile,origin,lane);
 if(lane<125u){let old=nbCrossingVertex(tile,origin,origin+umCorner(lane,5u));let out=250u*group.x+2u*lane;result[out]=bitcast<u32>(old);result[out+1u]=bitcast<u32>(nbBandVertices[lane]);}
}
`;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error'),[]);
  const pipelines=await Promise.all([false,true].map(cold=>device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'compare',constants:{nbColdCurrentPhi:+cold}}})));
  const data=new Float32Array(192000);let rng=123;
  for(let i=0;i<data.length;i++){rng=(Math.imul(rng,1664525)+1013904223)>>>0;data[i]=(rng%100001-50000)/703;}
  const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,data);
  const output=device.createBuffer({size:64*250*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{
   for(const pipeline of pipelines){
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]});
    const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(64);p.end();e.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);
    const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
    for(let i=0;i<actual.length;i+=2)assert.equal(actual[i],actual[i+1],`case=${Math.floor(i/250)}, vertex=${(i%250)/2}`);
   }
  }finally{input.destroy();output.destroy();read.destroy();}
 });
});
