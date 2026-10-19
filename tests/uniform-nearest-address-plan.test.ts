import assert from 'node:assert/strict';
import test from 'node:test';
import { narrowBandNearestWGSL } from '../lib/methods/uniform/uniform-narrow-band-nearest.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';
// The previous scalar row lookup and nearest selection, retained literally.
const reference=`@compute @workgroup_size(64) fn referenceY(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));
 if(nbNearestReusable(tile)){return;}
 var best=vec4u(255u);var distance=vec4i(1000);
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,o,0),0u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code)-5;let d=dx*dx+o*o;let closer=(code!=vec4u(255u))&(d<distance);
  best=select(best,code|vec4u(u32(o+5)<<4u),closer);distance=select(distance,d,closer);
 }
 let packed=best.x|(best.y<<8u)|(best.z<<16u)|(best.w<<24u);
 atomicStore(&bins[NB_DEPTH_A+UM_CELL_KEYS/4u+16u*tile+lane%16u],packed);
 if(nbReuseCacheEnabled()&&nbReuseRegion(tile)<NB_REUSE_CAPACITY){atomicStore(&bins[nbNearestCacheAt(tile,1u,lane%16u)],packed);}
}
@compute @workgroup_size(64) fn referenceZ(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 let job=4u*group.x+lane/16u;if(job>=atomicLoad(&bins[NB_BAND+1u])){return;}
 let tile=atomicLoad(&bins[NB_BAND_SEARCH+job]);let coord=umTileCoord(tile);
 let p=vec3i(4u*coord+vec3u(0u,lane%4u,(lane%16u)/4u));var nearest=vec4u(NB_NO_SURFACE);var distance=vec4i(1000);
 if(nbNearestReusable(tile)){
  let destination=NB_DEPTH_B+umCellAt(vec3u(p));let cached=nbNearestFinalAt(tile,lane%16u);
  for(var k=0u;k<4u;k++){atomicStore(&bins[destination+k],atomicLoad(&bins[cached+k]));}return;
 }
 for(var o=-5;o<=5;o++){
  let row=nbGlobalNearestRow(p+vec3i(0,0,o),1u);let code=(vec4u(row)>>vec4u(0u,8u,16u,24u))&vec4u(255u);
  let dx=vec4i(code&vec4u(15u))-5;let dy=vec4i(code>>vec4u(4u))-5;let d=dx*dx+dy*dy+o*o;
  let closer=(code!=vec4u(255u))&(d<distance);
  // A missing or losing row cannot change any nearest key. In particular,
  // avoid four page-table lookups for its discarded coordinates.
  if(!any(closer)){continue;}
  let x=vec4u(vec4i(p.x)+vec4i(0,1,2,3)+dx);let y=vec4u(vec4i(p.y)+dy);let z=u32(p.z+o);
  let at=vec4u(umCellAt(vec3u(x.x,y.x,z)),umCellAt(vec3u(x.y,y.y,z)),umCellAt(vec3u(x.z,y.z,z)),umCellAt(vec3u(x.w,y.w,z)));
  nearest=select(nearest,at,closer);distance=select(distance,d,closer);
 }
 let at=NB_DEPTH_B+umCellAt(vec3u(p));
 if(nbReuseCacheEnabled()&&nbReuseRegion(tile)<NB_REUSE_CAPACITY){
  let cached=nbNearestFinalAt(tile,lane%16u);for(var k=0u;k<4u;k++){atomicStore(&bins[cached+k],nearest[k]);}
 }
 atomicStore(&bins[at],nearest.x);atomicStore(&bins[at+1u],nearest.y);atomicStore(&bins[at+2u],nearest.z);atomicStore(&bins[at+3u],nearest.w);
}`;
test('compiled nearest row plans preserve missing tiles, retained rows, ties and partial groups',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('nearest row address plan',async device=>{
  const functions=narrowBandNearestWGSL.slice(narrowBandNearestWGSL.indexOf('fn nbGlobalNearestRow'));
  const code=`
@group(0) @binding(0) var<storage,read_write> bins:array<atomic<u32>>;
const UM_TILES=216u;const UM_CELL_KEYS=13824u;
const NB_BAND=0u;const NB_BAND_SEARCH=16u;const NB_BAND_REACH=256u;
const NB_DEPTH_A=1024u;const NB_DEPTH_B=8192u;const NB_NO_SURFACE=0xffffffffu;const NB_REUSE_CAPACITY=216u;
fn umTileCoord(tile:u32)->vec3u{let i=(tile*173u)%216u;return vec3u(i%6u,(i/6u)%6u,i/36u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(6))){return UM_TILES;}let i=u32(p.x+6*(p.y+6*p.z));if(i%17u==0u){return UM_TILES;}return (i*5u)%216u;}
fn umCellAt(p:vec3u)->u32{let l=p&vec3u(3);return 64u*umTileKey(vec3i(p>>vec3u(2)))+l.x+4u*(l.y+4u*l.z);}
fn nbNearestReusable(tile:u32)->bool{return tile%7u==0u;}
fn nbReuseRegion(tile:u32)->u32{return tile;}
fn nbReuseCacheEnabled()->bool{return true;}
fn nbNearestCacheAt(tile:u32,axis:u32,row:u32)->u32{return 32768u+32u*tile+16u*axis+row;}
fn nbNearestFinalAt(tile:u32,row:u32)->u32{return 65536u+64u*tile+4u*row;}
fn nbBandRow(coord:vec3u,y:i32,z:i32)->u32{return 0u;}
`+functions+reference;
  const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
  const pipelines=await Promise.all(['referenceY','depthNearestY','referenceZ','depthNearestZ'].map(entryPoint=>device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}})));
  const input=new Uint32Array(80000);let rng=871;const random=()=>rng=(Math.imul(rng,1664525)+1013904223)>>>0;
  const jobs:number[]=[];for(let i=0;i<216;i++)if(i%17!==0)jobs.push(i*5%216);input[1]=jobs.length;input.set(jobs,16);
  for(let tile=0;tile<216;tile++){
   input[256+tile]=+(tile%11!==0);
   for(let row=0;row<16;row++)for(let axis=0;axis<2;axis++){
    let packed=0;for(let k=0;k<4;k++){const v=random();const code=v%5===0?255:(v%11)|(axis?((v>>>8)%11)<<4:0);packed|=code<<(8*k);}
    input[1024+axis*3456+16*tile+row]=packed;input[32768+32*tile+16*axis+row]=packed;
   }
   for(let k=0;k<64;k++)input[65536+64*tile+k]=random();
  }
  const buffer=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  try{
   for(let axis=0;axis<2;axis++){
    let expected:Uint32Array|undefined;
    for(let version=0;version<2;version++){
     const pipeline=pipelines[2*axis+version]!;device.queue.writeBuffer(buffer,0,input);
     const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer}}]});
     const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(jobs.length/4));p.end();e.copyBufferToBuffer(buffer,0,read,0,read.size);device.queue.submit([e.finish()]);await read.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     if(expected){for(let i=0;i<actual.length;i++)assert.equal(actual[i],expected[i],`axis ${axis}, word ${i}`);}else expected=actual;
    }
   }
  }finally{buffer.destroy();read.destroy();}
 });
});
