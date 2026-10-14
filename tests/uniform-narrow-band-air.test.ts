import assert from "node:assert/strict";
import test from "node:test";
import { narrowBandAirWGSL } from "../lib/methods/uniform/uniform-narrow-band-air.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

// Exercise the production cleanup kernels with an ordinary 3D field. Only
// addressing/solid bindings are replaced: the field and mask logic are real.
test("empty-air retirement clears buried phi and preserves bulk, particles and shared support",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("empty-air retirement",async device=>{
  for(const width of [1,4]){
   const nx=128,ny=8,nz=8,tiles=32*2*2,vertices=(nx+1)*(ny+1)*(nz+1),h=0.02,far=16*0.04;
   const at=(x:number,y:number,z:number)=>x+(nx+1)*(y+(ny+1)*z);
   const values=new Float32Array(vertices);
   for(let z=0;z<=nz;z++)for(let y=0;y<=ny;y++)for(let x=0;x<=nx;x++){
    // A physical Eulerian body on the right, with no particles. The buried
    // planar ghost on the left has negative corners in partially open cells
    // too, but is positive everywhere above terrain y=2.5.
    values[at(x,y,z)]=Math.min(6*h,(Math.max(Math.abs(x-104)-4,Math.abs(y-4)-2,Math.abs(z-4)-2))*h,
      x<=16?(y-2)*h:6*h);
   }
   const input=device.createTexture({size:[nx+1,ny+1,nz+1],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
   const output=device.createTexture({size:[nx+1,ny+1,nz+1],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST});
   const bins=device.createBuffer({size:4*6*tiles,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:4*vertices,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   // An isolated ballistic sample in positive air at x=48, across a page
   // boundary from its protected halo. It must retain support without phi.
   const particleTile=12+32*(1+2*1),mask=new Uint32Array(6*tiles);mask[particleTile]=1;
   device.queue.writeBuffer(bins,0,mask);
   for(const texture of [input,output])device.queue.writeTexture({texture},values,{bytesPerRow:4*(nx+1),rowsPerImage:ny+1},[nx+1,ny+1,nz+1]);
   try{
    const module=device.createShaderModule({code:`
@group(0) @binding(0) var phi:texture_3d<f32>;
@group(0) @binding(1) var outputPhi:texture_storage_3d<r32float,read_write>;
@group(0) @binding(2) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> result:array<f32>;
const UM_TILES=${tiles}u;const NB_SURFACE_TILES=0u;const NB_ACTIVITY_SURFACE_HEAT=UM_TILES;
struct Params{hDt:vec4f} const params=Params(vec4f(0.02,0.04,0.01,0.016666667));
fn umCorner(k:u32,side:u32)->vec3u{return vec3u(k%side,(k/side)%side,k/(side*side));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%32u,(t/32u)%2u,t/64u);}
fn umTileKey(p:vec3i)->u32{if(any(p<vec3i(0))||any(p>=vec3i(32,2,2))){return 0xffffffffu;}return u32(p.x+32*(p.y+2*p.z));}
fn umKeyStored(t:u32)->bool{return t<UM_TILES;}
fn umTileLive(t:u32)->bool{return t<UM_TILES;}
fn umTileWidth(t:u32)->u32{return ${width}u;}
fn umTileMinimumWidth(t:u32)->u32{return ${width}u;}
fn umTileMaximumWidth(t:u32)->u32{return ${width}u;}
fn umLowOpen(p:vec3i)->vec3<bool>{return p==vec3i(0);}
struct Authority{width:u32,tile:u32}
fn umVertexAuthority(p:vec3u)->Authority{return Authority(${width}u,umTileKey(vec3i(p/4u)));}
fn umLoadVertex8(p:vec3u,s:u32)->array<f32,8>{var v:array<f32,8>;for(var k=0u;k<8u;k++){v[k]=textureLoad(phi,vec3i(p+s*umCorner(k,2u)),0).x;}return v;}
fn umLoadCorner8(p:vec3u,s:u32)->array<f32,8>{return umLoadVertex8(p,s);}
fn umCellOpen(c:vec3i)->f32{return clamp(f32(c.y+1)-2.5,0.0,1.0);}
fn umSolidHasTerrain()->bool{return true;}
fn umTerrainSurface(p:vec3f)->vec3f{return vec3f(2.5,0,0);}
fn nbTrilinear(v:array<f32,8>,q:vec3f)->vec4f{return vec4f(0,0,0,mix(mix(mix(v[0],v[1],q.x),mix(v[2],v[3],q.x),q.y),mix(mix(v[4],v[5],q.x),mix(v[6],v[7],q.x),q.y),q.z));}
${narrowBandAirWGSL}
@compute @workgroup_size(64) fn read(@builtin(global_invocation_id) id:vec3u){let i=id.x;if(i<${vertices}u){result[i]=textureLoad(outputPhi,vec3i(vec3u(i%129u,(i/129u)%9u,i/(129u*9u)))).x;}}
`});
    const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}: ${m.message}`),[]);
    const layout=device.createBindGroupLayout({entries:[
     {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},
     {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"read-write",format:"r32float",viewDimension:"3d"}},
     ...[2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage" as const}})),
    ]});
    const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
    const entries=["airSeeds","airSpread0","airSpread1","airSpread2","airClearFine","airClearCoarse","read"];
    const pipelines=await Promise.all(entries.map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
    const group=device.createBindGroup({layout,entries:[{binding:0,resource:input.createView()},{binding:1,resource:output.createView()},{binding:2,resource:{buffer:bins}},{binding:3,resource:{buffer:read}}]});
    const e=device.createCommandEncoder();
    for(const [i,pipeline] of pipelines.entries()){const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(i===entries.length-1?Math.ceil(vertices/64):1024);p.end();}
    device.queue.submit([e.finish()]);
    const after=await readMixedBuffer(device,read),stored=await readMixedBuffer(device,bins),words=new Uint32Array(stored.buffer);
    for(let z=0;z<=nz;z+=width)for(let y=0;y<=ny;y+=width)for(let x=0;x<=nx;x+=width){
     const index=at(x,y,z);
     if(x<=16)assert.ok(after[index]!>=far-1e-7,`buried ghost/far air at ${[x,y,z]} width ${width}`);
     if(x>=96&&x<=112)assert.equal(after[index],values[index],`Eulerian bulk/halo changed at ${[x,y,z]} width ${width}`);
     if(x>=32&&x<=68)assert.equal(after[index],values[index],`ballistic support changed at ${[x,y,z]} width ${width}`);
    }
    assert.equal(words[2*tiles+particleTile],1,"ballistic tile seeds support");
    assert.equal(words[2*tiles],0,"buried liquid must not seed support");
    assert.equal(words[2*tiles+26+32*(1+2*1)],1,"Eulerian liquid must seed without particles");
   }finally{input.destroy();output.destroy();bins.destroy();read.destroy();}
  }
 });
});

test("live pages retire on the first unrequested commit and retain requested support",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {uniformLatticeAddress}=await import("../lib/methods/uniform/uniform-lattice-address");
 const {UniformPagePool,UNIFORM_PAGE_POOL_STATUS:S,UNIFORM_PAGE_POOL_RECORD:R,UNIFORM_PAGE_POOL_BLOCK:B}=await import("../lib/methods/uniform/uniform-page-pool");
 await withUniformDevice("prompt page retirement",async device=>{
  const address=uniformLatticeAddress([128,32,32],false,{blocks:6});
  const pool=new UniformPagePool(device,address.pool);
  try{
   pool.seed(address.seed);await pool.initialize();
   const commit=async(requested:number[])=>{
    for(const x of requested)pool.request(x,0,0);
    const e=device.createCommandEncoder();pool.encodeCommit(e);device.queue.submit([e.finish()]);
    const read=await readMixedBuffer(device,pool.state);return new Uint32Array(read.buffer);
   };
   const first=await commit([2,3]);
   assert.equal(first[S.fatal],0);assert.equal(first[S.resident],2);assert.equal(first[S.pending],2);
   const pages=[];
   for(let b=1;b<5;b++){const r=pool.layout.state.records+R.words*b;if(first[r+R.state]===B.resident)pages.push(first[r]);}
   assert.deepEqual(pages.sort(),[2,3],"fluid and its neighbouring support page stay resident");
   const second=await commit([2,3]);
   assert.equal(second[S.fatal],0);assert.equal(second[S.resident],2);assert.equal(second[S.free],2);assert.equal(second[S.pending],0);
   const reclaimed=await commit([0,2,3]);
   assert.equal(reclaimed[S.fatal],0);assert.equal(reclaimed[S.resident],3);assert.equal(reclaimed[S.claimed],1);
  }finally{pool.destroy();}
 });
});
