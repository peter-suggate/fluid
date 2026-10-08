import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { narrowBandParticleSurfaceWGSL, narrowBandFineTransferWGSL, narrowBandTiledSurfaceWGSL } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";
import { UniformNarrowBandOrder } from "../lib/methods/uniform/uniform-narrow-band-order";
import { withUniformDevice, advanceUniform } from "./helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("FLIP spatial ordering preserves crowded bins, sample identity and compact neighbor positions",async()=>{
 await withUniformDevice("FLIP spatial order",async device=>{
  const n=257,cells=8**3,data=new Float32Array(n*12),heads=new Uint32Array(cells*2),chain=new Uint32Array(n*9);
  for(let i=0;i<n;i++){
   const cell=i<150?19:(i*137)%cells;
   data.set([cell%8+0.25,Math.floor(cell/8)%8+0.25,Math.floor(cell/64)+0.25,i+1,i,-i,0.5*i,1,2,3,4,5],12*i);
   chain[i]=heads[2*cell]!;heads[2*cell]=i+1;heads[2*cell+1]++;
  }
  const buffer=(size:number)=>device.createBuffer({size,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const a=buffer(data.byteLength),b=buffer(data.byteLength),bins=buffer(heads.byteLength),links=buffer(chain.byteLength);
  device.queue.writeBuffer(b,0,data);device.queue.writeBuffer(bins,0,heads);device.queue.writeBuffer(links,0,chain);
  const order=new UniformNarrowBandOrder(device,[8,8,8],[a,b],bins,links);
  try{
   await order.initialize();const encoder=device.createCommandEncoder();order.encode(encoder,0);device.queue.submit([encoder.finish()]);
   const sorted=await readMixedBuffer(device,a),newHeads=new Uint32Array((await readMixedBuffer(device,bins)).buffer),arena=await readMixedBuffer(device,links),newLinks=new Uint32Array(arena.buffer),seen=new Set<number>();
   for(let cell=0;cell<cells;cell++){
    let link=newHeads[2*cell]!,count=0;
    while(link){const i=link-1;assert.ok(i<n&&!seen.has(i),"every sample appears in exactly one acyclic bin");seen.add(i);count++;
     const id=sorted[12*i+3]!-1;assert.deepEqual(sorted.slice(12*i,12*i+12),data.slice(12*id,12*id+12));
     assert.equal(Math.floor(sorted[12*i]!)+8*(Math.floor(sorted[12*i+1]!)+8*Math.floor(sorted[12*i+2]!)),cell);
     assert.deepEqual(arena.slice(n+4*i,n+4*i+4),sorted.slice(12*i,12*i+4));link=newLinks[i]!;
    }
    assert.equal(count,heads[2*cell+1]);
   }
   assert.equal(seen.size,n);
  }finally{order.destroy();for(const resource of [a,b,bins,links])resource.destroy();}
 });
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("FLIP preserves crowded surface samples and defers reseeding at the particle budget",{timeout:120_000},async()=>{
 await withUniformDevice("FLIP particle budget",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
   gravity_m_s2:{x:0,y:0,z:0},initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"paper",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   await advanceUniform(solver,1/30);
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   // A compressed outer layer uses the entire allocation. All of these are
   // persistent surface samples, not optional inner-band reseeding.
   const samples=new Float32Array(stage.capacity*12);
   for(let i=0;i<stage.capacity;i++){
    const at=i*12;samples[at]=0.25+0.5*(i%64);samples[at+1]=15.75;samples[at+2]=0.25+0.5*(Math.floor(i/64)%64);samples[at+3]=1;samples[at+7]=-0.25;
   }
   device.queue.writeBuffer(stage.activeParticles,0,samples);device.queue.writeBuffer(stage.state,0,new Uint32Array([stage.capacity,stage.capacity,0,0]));
   await advanceUniform(solver,2/30);
   assert.equal(stage.count,stage.capacity,"budget pressure does not discard the outer surface layer");
   assert.ok(stage.reseedClipped>0,"optional inner samples are deferred and reported instead of overflowing");
   const result=await readMixedBuffer(device,stage.activeParticles);
   assert.ok(result.every(Number.isFinite));
   let moved=false;for(let i=0;i<stage.count;i++)if(Math.abs(result[12*i+1]!-15.75)>1e-4){moved=true;break;}
   assert.equal(moved,false,"resampling never displaces the protected surface layer");
  }finally{solver.destroy();}
});
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("unresolved detached FLIP particles fall under gravity instead of following empty grid velocities",{timeout:120_000},async()=>{
 await withUniformDevice("FLIP detached gravity",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
   gravity_m_s2:{x:0,y:-9.81,z:0},initialCondition:"tank-fill",initialLiquidVolumes:[],initialBrickSeeds_m:undefined,initialHeightField:undefined});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const dt=0.017;await advanceUniform(solver,dt);
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   // Like the grid, a resting particle starts with its half-step kick.
   const poolCount=stage.count;
   device.queue.writeBuffer(stage.activeParticles,poolCount*48,new Float32Array([16,24,16,1,0,-0.5*9.81*dt,0,8,0,0,0,0]));
   device.queue.writeBuffer(stage.state,0,new Uint32Array([poolCount+1,poolCount+1,0,0]));
   for(let step=1;step<=3;step++){
    await advanceUniform(solver,(step+1)*dt);
    const samples=await readMixedBuffer(device,stage.activeParticles);let detached:Float32Array|undefined;
    for(let i=0;i<stage.count;i++)if(samples[12*i+1]!>20){assert.equal(detached,undefined,"no new particles seed the unsupported droplet");detached=samples.slice(12*i,12*i+12);}
    assert.ok(detached,"a real detached particle is retained");
    assert.ok(Math.abs(detached[1]!-(24-0.5*9.81*(step*dt)**2*32))<1e-3,"unresolved particle follows ballistic position");
    assert.ok(Math.abs(detached[5]!+9.81*(step+0.5)*dt)<1e-4,"gravity enters particle momentum once, at the grid's half-step time");
   }
  }finally{solver.destroy();}
});
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("FLIP surface and interior samples share the same free-fall acceleration",{timeout:120_000},async()=>{
 await withUniformDevice("FLIP free-fall transfer",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0,top:"closed",fluidWallMode:"free-slip"});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  Object.assign(scene.fluid,{initialVelocity_m_s:{x:0,y:0,z:0},inflow:undefined,surfaceTension_N_m:0,dynamicViscosity_Pa_s:0,
   gravity_m_s2:{x:0,y:-9.81,z:0},initialCondition:"tank-fill",initialLiquidVolumes:[{shape:"sphere",center_m:{x:0,y:0.65,z:0},radius_m:0.18}],initialBrickSeeds_m:undefined,initialHeightField:undefined});
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{timeStep:"scene",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   for(let step=1;step<=6;step++)await advanceUniform(solver,step*0.017);
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   const samples=await readMixedBuffer(device,stage.activeParticles);let outer=0,inner=0,no=0,ni=0;
   for(let i=0;i<stage.count;i++){
    if(samples[12*i+7]!>=-1){outer+=samples[12*i+5]!;no++;}else{inner+=samples[12*i+5]!;ni++;}
   }
   assert.ok(no>100&&ni>100);outer/=no;inner/=ni;
   // Uniform kicks the seeded velocity half a step before the first drift.
   const expected=-9.81*6.5*0.017;
   assert.ok(Math.abs(outer-expected)<1e-3,"air-side grid clearing must not slow the outer samples");
   assert.ok(Math.abs(outer-inner)<1e-3,"surface samples fall with the interior instead of leaving a trailing skin");
  }finally{solver.destroy();}
 });
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("cooperative FLIP transfer matches independent quadratic MAC gathers with crowded and ballistic samples",async()=>{
 await withUniformDevice("FLIP cooperative transfer",async device=>{
  const n=521,dims=8,cells=dims**3,data=new Float32Array(n*8),heads=new Uint32Array(cells*2),chain=new Uint32Array(n);
  for(let i=0;i<n;i++){
   const x=i<180?3.1+(i%7)*0.11:0.1+((i*137)%780)/100;
   const y=i<180?3.2+(i%5)*0.12:0.1+((i*239)%780)/100;
   const z=i<180?3.1+(i%9)*0.09:0.1+((i*317)%780)/100;
   data.set([x,y,z,1,Math.sin(i)*2,Math.cos(i)*3,(i%13-6)/3,i%11===0?1:0],i*8);
   const cell=Math.floor(data[8*i]!)+dims*(Math.floor(data[8*i+1]!)+dims*Math.floor(data[8*i+2]!));
   chain[i]=heads[2*cell]!;heads[2*cell]=i+1;
  }
  const buffer=(data:Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>)=>{const b=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,data);return b;};
  const buffers=[buffer(data),buffer(heads),buffer(chain)];
  const texture=()=>device.createTexture({size:[dims,dims,dims],dimension:"3d",format:"rgba32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});
  const velocity=texture(),output=texture(),initial=new Float32Array(cells*4);
  for(let i=0;i<cells;i++)initial.set([0.3,-0.2,0.7,9],4*i);
  device.queue.writeTexture({texture:velocity},initial,{bytesPerRow:dims*16,rowsPerImage:dims},[dims,dims,dims]);
  try{
   const module=device.createShaderModule({code:/* wgsl */`
const UM_D=vec3u(8);struct Particle{position:vec4f,motion:vec4f} struct UMOwner{index:u32,width:u32}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> links:array<u32>;
@group(0) @binding(3) var velocity:texture_3d<f32>;
@group(0) @binding(4) var output:texture_storage_3d<rgba32float,write>;
fn umAllOwner(g:vec3u)->UMOwner{return UMOwner(g.x,select(0u,1u,g.x<512u));}
fn umOrigin(o:UMOwner)->vec3u{return vec3u(o.index%8u,(o.index/8u)%8u,o.index/64u);}
fn particleDepth(p:vec3f)->f32{return p.y-4.0;}
fn cellIndex(p:vec3i)->u32{return u32(p.x+8*(p.y+8*p.z));}
fn nbPosition(i:u32)->vec3f{return particles[i].position.xyz;}
fn nbMotion(i:u32)->vec4f{return particles[i].motion;}
fn transferFallback(o:UMOwner){}
fn weight(x:f32)->f32{let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;}
${narrowBandFineTransferWGSL}
`});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"transfer"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[...buffers.map((buffer,binding)=>({binding,resource:{buffer}})),{binding:3,resource:velocity.createView()},{binding:4,resource:output.createView()}]});
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(cells*8/64);pass.end();device.queue.submit([encoder.finish()]);
   const actual=await readMixedTexture(device,output);
   const weight=(v:number)=>{const a=Math.abs(v);return a<0.5?0.75-a*a:0.5*Math.max(0,1.5-a)**2;};
   let maxError=0;
   for(let cell=0;cell<cells;cell++){
    const origin=[cell%8,Math.floor(cell/8)%8,Math.floor(cell/64)];
    for(let axis=0;axis<3;axis++){
     const q=origin.map(v=>v+0.5);q[axis]!+=0.5;const depth=q[1]!-4;
     let mass=0,momentum=0;
     for(let i=0;i<n;i++)if(data[8*i+7]!==1){const w=weight(q[0]!-data[8*i]!)*weight(q[1]!-data[8*i+1]!)*weight(q[2]!-data[8*i+2]!);mass+=w;momentum+=w*data[8*i+4+axis]!;}
     const blend=origin[axis]!+1>=dims||depth>1.5||mass<1e-5?0:(depth>=-2?1:0);
     const expected=initial[4*cell+axis]!*(1-blend)+(mass?momentum/mass:0)*blend;
     maxError=Math.max(maxError,Math.abs(actual[4*cell+axis]!-expected));
    }
    assert.equal(actual[4*cell+3],9,"transfer preserves the fourth channel");
   }
   assert.ok(maxError<2e-5,`quadratic transfer error ${maxError}`);
  }finally{buffers.forEach(b=>b.destroy());velocity.destroy();output.destroy();}
 });
});

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("tiled FLIP reconstruction matches vertex gathers across crowded bins and closed walls",async()=>{
 await withUniformDevice("FLIP shared reconstruction",async device=>{
  const n=1129,data=new Float32Array(n*4),heads=new Uint32Array(2*512+2*8),chain=new Uint32Array(n);
  for(let i=0;i<n;i++){
   const p=i<700?[3.1+(i%7)*0.11,3.2+(i%5)*0.12,3.1+(i%9)*0.09]:[0.1+((i*137)%780)/100,0.1+((i*239)%780)/100,0.1+((i*317)%780)/100];
   data.set([...p,1+i%3],4*i);const cell=Math.floor(data[4*i]!)+8*(Math.floor(data[4*i+1]!)+8*Math.floor(data[4*i+2]!));
   chain[i]=heads[2*cell]!;heads[2*cell]=i+1;
  }
  heads.fill(1,1024);
  const buffer=(data:Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>)=>{const b=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});device.queue.writeBuffer(b,0,data);return b;};
  const buffers=[buffer(data),buffer(heads),buffer(chain),buffer(new Float32Array(9**3))];
  const output=device.createTexture({size:[9,9,9],dimension:"3d",format:"r32float",usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});
  device.queue.writeTexture({texture:output},new Float32Array(9**3).fill(99),{bytesPerRow:36,rowsPerImage:9},[9,9,9]);
  try{
   const module=device.createShaderModule({code:/* wgsl */`
const UM_D=vec3u(8);const UM_T=vec3u(2);const NB_COVERAGE=1024u;const NB_SURFACE_TILES=1032u;
struct Params{hDt:vec4f,settings:vec4f} const params=Params(vec4f(1),vec4f(0));
struct Particle{position:vec4f} struct UMOwner{tile:u32,index:u32,width:u32}
@group(0) @binding(0) var<storage,read> particles:array<Particle>;
@group(0) @binding(1) var<storage,read_write> bins:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read> links:array<u32>;
@group(0) @binding(3) var<storage,read_write> reference:array<f32>;
@group(0) @binding(4) var outputPhi:texture_storage_3d<r32float,write>;
fn umAllOwner(g:vec3u)->UMOwner{return UMOwner(g.x/64u,g.x,1u);}
fn umTileCoord(t:u32)->vec3u{return vec3u(t%2u,(t/2u)%2u,t/4u);}
fn umOrigin(o:UMOwner)->vec3u{let l=o.index%64u;return 4u*umTileCoord(o.tile)+vec3u(l%4u,(l/4u)%4u,l/16u);}
fn umTileMinimumWidth(t:u32)->u32{return 1u;} fn umTileMaximumWidth(t:u32)->u32{return 1u;}
fn umVertexAuthority(q:vec3u)->UMOwner{return UMOwner();}
fn umTileAt(p:vec3u)->u32{return p.x+2u*(p.y+2u*p.z);}
fn cellIndex(p:vec3i)->u32{return u32(p.x+8*(p.y+8*p.z));}
fn nbPosition(i:u32)->vec3f{return particles[i].position.xyz;}
fn nbSourcePhi(p:vec3f,value:f32)->f32{return value;}
fn bandPhi(p:vec3f)->f32{return select(3.0,-2.0,p.x<1.0);}
fn bulkDepth(p:vec3f)->f32{return bandPhi(p);}
${narrowBandParticleSurfaceWGSL}
${narrowBandTiledSurfaceWGSL}
@compute @workgroup_size(64) fn serial(@builtin(global_invocation_id) gid:vec3u){
 let i=gid.x;if(i>=729u){return;}let q=vec3i(i32(i%9u),i32((i/9u)%9u),i32(i/81u));reference[i]=particleSurface(q);
}
`});
   const layout=device.createBindGroupLayout({entries:[...[0,1,2,3].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0||binding===2?"read-only-storage" as const:"storage" as const}})),{binding:4,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"r32float",viewDimension:"3d"}}]});
   const pipelineLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
   const pipelines=await Promise.all(["coupleFine","serial"].map(entryPoint=>device.createComputePipelineAsync({layout:pipelineLayout,compute:{module,entryPoint}})));
   const group=device.createBindGroup({layout,entries:[...buffers.map((buffer,binding)=>({binding,resource:{buffer}})),{binding:4,resource:output.createView()}]});
   const encoder=device.createCommandEncoder();
   for(let i=0;i<2;i++){const pass=encoder.beginComputePass();pass.setPipeline(pipelines[i]!);pass.setBindGroup(0,group);pass.dispatchWorkgroups(i===0?8:12);pass.end();}
   device.queue.submit([encoder.finish()]);
   const actual=await readMixedTexture(device,output),expected=await readMixedBuffer(device,buffers[3]!);
   let error=0;for(let i=0;i<actual.length;i++){assert.ok(Number.isFinite(actual[i]!));error=Math.max(error,Math.abs(actual[i]!-expected[i]!));}
   assert.ok(error<3e-5,`shared reconstruction error ${error}`);
  }finally{buffers.forEach(b=>b.destroy());output.destroy();}
 });
});
