import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { uniformLatticeAddress } from '../lib/methods/uniform/uniform-lattice-address';
import { createUniformMixedLayoutFromWidths } from '../lib/methods/uniform/uniform-mixed-layout';
import { uniformMixedTopologyWGSL, uniformMixedTopologyNeighboursWGSL } from '../lib/methods/uniform/uniform-mixed-topology.wgsl';
import { narrowBandTiledSurfaceWGSL } from '../lib/methods/uniform/uniform-narrow-band-surface.wgsl';
import { withUniformDevice } from './helpers/uniform-geometric';

// Reference is the previous 125-lane ownership traversal. Both kernels use
// production paged addressing, compiled stencils and authority resolution.
const reference=`
@compute @workgroup_size(128) fn reference(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
 testJob=group.x;if(lane>=125u){return;}
 let owner=umAllOwner(vec3u(group.x*64u,0,0));let origin=vec3i(umTileCoord(owner.tile)*4u);
 let local=umCorner(lane,5u);let q=origin+vec3i(local);
 if(umTileMaximumWidth(owner.tile)==1u&&umTileMinimumWidth(owner.tile)==1u){if(any(local==vec3u(0))&&!all((local>vec3u(0))|umLowOpen(origin))){return;}}
 else{let authority=umVertexAuthority(vec3u(q));if(authority.width!=1u||authority.tile!=owner.tile){return;}}
 nbStoreReconstructed(q,0.0,1u);
}`;

test('reconstruction enumerates the same vertices across mixed seams and sparse page corners', {skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice('reconstruction ownership',async device=>{
  const dimensions=[96,96,96] as const;
  const address=uniformLatticeAddress(dimensions,true,{permute:871,air:[[0,0,0],[1,0,1],[0,1,1],[2,2,1],[1,2,2]]});
  const lattice={dimensions,cellSize_m:[1,1,1] as const,origin_m:{x:0,y:0,z:0}};
  const source=readFileSync(new URL('../lib/methods/uniform/uniform-mixed-vertex-sampling.wgsl.ts',import.meta.url),'utf8');
  const authority=source.slice(source.indexOf('fn umVertexAuthority('),source.indexOf('fn umVertexIsCanonical('));
  for(const period of [1,3,17]){
   const widths=Uint8Array.from({length:address.boxTiles},(_,i)=>period===1||((i*177+Math.floor(i/24)*31)%period)!==0?1:4);
   const layout=createUniformMixedLayoutFromWidths(lattice,widths,[],address),n=layout.tiles.length;
   const words=new Uint32Array(5*n+address.seed.tail.length);words.set(layout.tiles);words.set(layout.fineTiles,n);words.set(layout.coarseTiles,n+layout.fineTiles.length);words.set(layout.stencils,2*n);words.set(layout.blendMasks,4*n);words.set(address.seed.tail,5*n);
   const topology=device.createBuffer({size:words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(topology,0,words);
   const counts=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(counts,0,new Uint32Array([layout.fineTiles.length,layout.coarseTiles.length,0,8]));
   const output=device.createBuffer({size:4*125*layout.fineTiles.length,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
   const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
   try{
    const code=uniformMixedTopologyWGSL(layout,0)+uniformMixedTopologyNeighboursWGSL()+authority+`
@group(1) @binding(0) var<storage,read_write> result:array<atomic<u32>>;
var<private> testJob:u32;
struct Params{hDt:vec4f} const params=Params(vec4f(1));
fn umSolidSkip()->bool{return false;}
fn nbTileCold(tile:u32)->bool{return false;}
fn nbVertexCold(p:vec3u)->bool{return false;}
fn particleSurfaceBulk(tile:u32,origin:vec3i,q:vec3i)->vec3f{return vec3f(0);}
fn particleSurfaceFinish(home:vec3i,q:vec3i,bulk:f32,nearest:f32,heat:f32)->f32{return 0.0;}
fn nbCachedSurfaceNearest(q:vec3i)->f32{return 0.0;}
fn nbStoreReconstructed(q:vec3i,value:f32,width:u32){
 let owner=umAllOwner(vec3u(testJob*64u,0,0));let local=vec3u(q)-4u*umTileCoord(owner.tile);
 atomicAdd(&result[125u*testJob+local.x+5u*(local.y+5u*local.z)],1u);
}
`+narrowBandTiledSurfaceWGSL.replace('if(umSolidSkip())','testJob=group.x;if(umSolidSkip())')+reference;
    const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);
    let expected:Uint32Array|undefined;
    for(const entryPoint of ['reference','coupleFine']){
     const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint}});
     const groups=[device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:topology}},{binding:1,resource:{buffer:counts}}]}),device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:output}}]})];
     const encoder=device.createCommandEncoder();encoder.clearBuffer(output);const pass=encoder.beginComputePass();pass.setPipeline(pipeline);groups.forEach((group,i)=>pass.setBindGroup(i,group));pass.dispatchWorkgroups(layout.fineTiles.length);pass.end();encoder.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const actual=new Uint32Array(read.getMappedRange()).slice();read.unmap();
     if(expected){for(let i=0;i<actual.length;i++)assert.equal(actual[i],expected[i],`period ${period}, fine tile ${Math.floor(i/125)}, local vertex ${i%125}`);}else expected=actual;
    }
   }finally{topology.destroy();counts.destroy();output.destroy();read.destroy();}
  }
 });
});
