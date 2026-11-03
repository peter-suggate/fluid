import assert from "node:assert/strict";
import test from "node:test";
import { sceneShapeWgsl, sceneShapeCode } from "../lib/core/scene-shape";
import { fluidRigidBoundaryWGSL } from "../lib/core/fluid-rigid-boundary.wgsl";
import { boundingRadius } from "../lib/core/rigid-body";
import { uniformMixedSolidWGSL, UniformMixedSolid } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import {pairedComputeTimings} from "./helpers/gpu-paired-timing";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";
import { GPU_RIGID_BODY_CAPACITY } from "../lib/core/rigid-capacity";

function fn(source:string,name:string):string {
 const start=source.indexOf(`fn ${name}(`);assert.ok(start>=0,name);let depth=0;
 for(let i=source.indexOf("{",start);i<source.length;i++){
  if(source[i]==="{")depth++;if(source[i]==="}"&&--depth===0)return source.slice(start,i+1);
 }
 throw new Error(name);
}
// Independent pre-optimization oracle: all samples search the whole roster,
// and nearest-wall extrapolation evaluates every body's distance in order.
const reference=/* wgsl */`
fn referenceAt(world:vec3f)->i32{for(var i=0u;i<umBodyCount();i++){if(umInsideBody(i,world)){return i32(i);}}return -1;}
fn referenceNear(world:vec3f,margin:f32)->bool{for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,world,margin)){return true;}}return false;}
fn reference(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()||umBodyCount()==0u){return 0.0;}
 var neighbor=id;neighbor[axis]+=1;
 if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;let reach=max(h.x,max(h.y,h.z));
 var near=false;for(var i=0u;i<umBodyCount();i++){near=near||umBodyNear(i,world,2.0*reach);}
 if(!near){return 0.0;}
 var solid=0.0;var velocity=0.0;
 for(var j=0u;j<8u;j++){
  let p=world+vec3f(select(-0.4,0.4,(j&1u)!=0u)*h.x,select(-0.4,0.4,(j&2u)!=0u)*h.y,select(-0.4,0.4,(j&4u)!=0u)*h.z);
  if(umSolidVoxelAtWorld(p)||umSolidTerrainAtWorld(p)){solid+=1.0;continue;}
  let body=referenceAt(p);if(body>=0){solid+=1.0;velocity+=umBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 var nearest=reach;var result=0.0;
 for(var i=0u;i<umBodyCount();i++){let d=abs(umBodyDistance(i,world));if(d<=nearest){nearest=d;result=umBodyVelocity(i,world)[axis];}}
 return result;
}`;

const referenceFace=/* wgsl */`
fn referenceFace(id:vec3i,axis:u32)->f32{
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;
 let tangentA=(axis+1u)%3u;let tangentB=(axis+2u)%3u;var solid=0.0;
 for(var j=0u;j<4u;j++){
  var p=world;p[tangentA]+=select(-0.35,0.35,(j&1u)!=0u)*h[tangentA];p[tangentB]+=select(-0.35,0.35,(j&2u)!=0u)*h[tangentB];
  solid+=select(0.0,1.0,umSolidVoxelAtWorld(p)||umSolidTerrainAtWorld(p)||referenceAt(p)>=0);
 }
 return 1.0-solid*.25;
}
fn baselineFace(p:vec3i,axis:u32)->f32{
 if(!referenceNear(umSolidFaceWorld(p,axis),length(.5*umSolidParams.cellGravity.xyz))){return 1.0;}
 return referenceFace(p,axis);
}
fn candidateFace(p:vec3i,axis:u32)->f32{
 let mask=umBodyCandidates(umSolidFaceWorld(p,axis),length(.5*umSolidParams.cellGravity.xyz));
 if(mask==0u){return 1.0;}return umFaceOpen(p,axis,mask,false);
}
`;

test("face candidate reuse preserves all-roster velocities, overlaps, nearest ties and interiors",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("exact rigid query reuse",async device=>{
  const count=32768,source=uniformMixedSolidWGSL(0);
  const functions=["umQuatRotate","umBodyLocal","umBodyNear","umInsideBody","umBodyDistance","umBodyVelocity","umBodyAt","umBodyAtCandidates","umBodyCandidates","umAnyBodyNear","umBodyCellMask","umSolidFaceWorld","umSolidFaceVelocity","umFaceOpen"].map(n=>fn(source,n)).join("\n");
  const body=device.createBuffer({size:12*128,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const settings=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const output=device.createBuffer({size:count*48,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const code=`
${sceneShapeWgsl()}\n${fluidRigidBoundaryWGSL}
struct Body{positionShape:vec4f,dimensions:vec4f,orientation:vec4f,linearVelocity:vec4f,angularVelocity:vec4f,pad0:vec4f,pad1:vec4f,pad2:vec4f};
@group(0) @binding(0) var<uniform> umSolidBodies:array<Body,12>;
@group(0) @binding(1) var<uniform> settings:vec4u;
@group(0) @binding(2) var<storage,read_write> result:array<u32>;
struct Params{cellGravity:vec4f};var<private> umSolidParams=Params(vec4f(.025,.035,.02,0));
fn umSolidEnabled()->bool{return true;}
fn umBodyCount()->u32{return settings.x;}
fn umSolidWorldCell(p:vec3i)->vec3f{return (vec3f(p)+.5)*umSolidParams.cellGravity.xyz;}
fn umSolidVoxel(p:vec3i)->bool{return settings.y!=0u&&p.y<0;}
fn umBoundaryRecord(p:vec3i)->u32{return 0u;}
fn umSolidVoxelAtWorld(p:vec3f)->bool{return settings.y!=0u&&p.y<0.0;}
fn umSolidTerrainAtWorld(p:vec3f)->bool{return settings.y!=0u&&p.y<.07*p.x;}
fn umSolidHasTerrain()->bool{return settings.y!=0u;}
// The static flags: everything clear without the static walls, nothing known with them.
const UM_STATIC_NO_VOXEL:u32=1u;const UM_STATIC_NO_TERRAIN:u32=2u;
fn umSolidStaticAt(p:vec3i)->u32{return select(0u,UM_STATIC_NO_VOXEL|UM_STATIC_NO_TERRAIN,settings.y==0u);}
${functions}\n${reference}\n${referenceFace}
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3u){
 let p=vec3i(i32(id.x&127u)-64,i32((id.x>>7u)&15u)-4,i32(id.x>>11u)-8);
 for(var axis=0u;axis<3u;axis++){
  result[12u*id.x+4u*axis]=bitcast<u32>(reference(p,axis));
  result[12u*id.x+4u*axis+1u]=bitcast<u32>(umSolidFaceVelocity(p,axis));
  result[12u*id.x+4u*axis+2u]=bitcast<u32>(baselineFace(p,axis));
  result[12u*id.x+4u*axis+3u]=bitcast<u32>(candidateFace(p,axis));
 }
}
${["baseline","candidate","apertureBaseline","apertureCandidate"].map(name=>`@compute @workgroup_size(64) fn ${name}(@builtin(global_invocation_id) g:vec3u){
 umSolidParams.cellGravity=vec4f(.025,.025,.025,0);
 let p=vec3i(i32(g.x%512u)-256,i32((g.x/512u)%8u)+16,4*i32(g.x/4096u)-16);
 for(var axis=0u;axis<3u;axis++){result[3u*g.x+axis]=bitcast<u32>(${name==="apertureBaseline"?"baselineFace":name==="apertureCandidate"?"candidateFace":name==="baseline"?"reference":"umSolidFaceVelocity"}(p,axis));}
}`).join("\n")}`;
   const module=device.createShaderModule({code});
   const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
   const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:body}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:output}}]});
   const writable=uniformMixedSolidWGSL(0,undefined,true,true);let tiledCode=code;
   for(const name of ["umBodyCandidates","umAnyBodyNear","umBodyAt"]){tiledCode=tiledCode.replace(fn(source,name),fn(writable,name));}
   tiledCode+="\nvar<private> umSolidQueryBodies:u32=0xfffu;\n"+fn(writable,"umBodyTileCandidates");
   const point="let p=vec3i(i32(id.x&127u)-64,i32((id.x>>7u)&15u)-4,i32(id.x>>11u)-8);";
   tiledCode=tiledCode.replace(point,point+"umSolidQueryBodies=umBodyTileCandidates(vec3u(p>>vec3u(2u)));");
   const tiledPipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:tiledCode}),entryPoint:"probe"}});
   const tiledGroup=device.createBindGroup({layout:tiledPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:body}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:output}}]});
   const words=new Float32Array(12*32);
   const shapes=["box","sphere","capsule","cylinder","cup"] as const;
   for(let i=0;i<12;i++){
    const shape=shapes[i%shapes.length]!,d=shape==="box"?{x:.8,y:.23,z:.14}:{x:.13,y:.31,z:.025};
    const at=32*i,angle=.19*i;
    words.set([i<2?0:(i-6)*.22,.15,i<2?0:(i%3-1)*.17,sceneShapeCode(shape)],at);
    words.set([d.x,d.y,d.z,boundingRadius({shape,dimensions_m:d} as Parameters<typeof boundingRadius>[0])],at+4);
    words.set([Math.cos(angle),0,Math.sin(angle),0],at+8);
    words.set([i*.13,-i*.07,.21-i*.02,0],at+12);words.set([.12*i,.23,.17,0],at+16);
   }
   // Identical overlapping bodies with different velocities exercise first
   // ownership and the nearest-distance <= tie (last body wins).
   words.set(words.subarray(0,12),32);
   for(const [bodies,staticWalls,interior] of [[0,0,0],[1,0,0],[2,0,0],[12,0,0],[12,1,0],[12,1,1]]){
    words[3]=interior?-sceneShapeCode("box")-1:sceneShapeCode("box");device.queue.writeBuffer(body,0,words);
    device.queue.writeBuffer(settings,0,new Uint32Array([bodies!,staticWalls!,0,0]));
    for(const [queryPipeline,queryGroup] of [[pipeline,group],[tiledPipeline,tiledGroup]] as const){
     const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(queryPipeline);p.setBindGroup(0,queryGroup);p.dispatchWorkgroups(count/64);p.end();device.queue.submit([e.finish()]);
     const values=new Uint32Array((await readMixedBuffer(device,output)).buffer);
     for(let i=0;i<values.length;i+=2)assert.equal(values[i+1],values[i],`tiled ${queryPipeline===tiledPipeline}, roster ${bodies}, walls ${staticWalls}, interior ${interior}, query ${i/2}`);
    }
   }
   if(process.env.FLUID_EXACT_ALGORITHM_TIMING){
    const {getSceneDefinition}=await import("../lib/core/scenes");
    const {sceneDocument}=await import("../lib/core/scene-definition");
    const scene=sceneDocument(getSceneDefinition("paddleboats"));
    words.fill(0);
    for(const [i,b] of scene.rigidBodies.entries()){
     const at=32*i,p=b.position_m,q=b.orientation,d=b.dimensions_m;
     words.set([p.x,p.y,p.z,sceneShapeCode(b.shape)],at);words.set([d.x,d.y,d.z,boundingRadius(b)],at+4);
     words.set([q.w,q.x,q.y,q.z],at+8);words.set([.1,.02,-.01,0],at+12);words.set([0,0,i%5===0?.02:-3.8,0],at+16);
    }
    device.queue.writeBuffer(body,0,words);device.queue.writeBuffer(settings,0,new Uint32Array([scene.rigidBodies.length,0,0,0]));
    const variants=await Promise.all(["baseline","candidate"].map(entryPoint=>device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}})));
    const groups=variants.map(p=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:body}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:output}}]}));
    console.log(JSON.stringify({algorithm:"paddleboats-body-candidates",...await pairedComputeTimings(device,variants,groups,count/64,64)}));
    const apertureVariants=await Promise.all(["apertureBaseline","apertureCandidate"].map(entryPoint=>device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint}})));
    const apertureGroups=apertureVariants.map(p=>device.createBindGroup({layout:p.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:body}},{binding:1,resource:{buffer:settings}},{binding:2,resource:{buffer:output}}]}));
    console.log(JSON.stringify({algorithm:"paddleboats-aperture-candidates",...await pairedComputeTimings(device,apertureVariants,apertureGroups,count/64,64)}));
   }
  }finally{body.destroy();settings.destroy();output.destroy();}
 },process.env.FLUID_EXACT_ALGORITHM_TIMING?["timestamp-query"]:[]);
});

test("body publication invalidates cached geometry in command order without clearing the edit epoch",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("body record lifetime",async device=>{
  const source=device.createBuffer({size:128*GPU_RIGID_BODY_CAPACITY,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
  const mirror=device.createBuffer({size:128*GPU_RIGID_BODY_CAPACITY,usage:GPUBufferUsage.COPY_DST});
  const record=device.createBuffer({size:32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const fresh=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  const output=device.createBuffer({size:48,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
  try{
   device.queue.writeBuffer(record,0,new Float32Array([1,2,3,4,17,1,0,0]));device.queue.writeBuffer(fresh,0,new Float32Array([1]));
   const solid=Object.assign(Object.create(UniformMixedSolid.prototype),{bodySource:source,bodies:mirror,coarse:{record},epochByte:16,bodyPendingByte:0}) as UniformMixedSolid;
   const e=device.createCommandEncoder();solid.encodeBodies(e);e.copyBufferToBuffer(record,16,output,0,16);
   e.copyBufferToBuffer(fresh,0,record,20,4);solid.consumeBodyEvidence(e);e.copyBufferToBuffer(record,16,output,16,16);
   solid.encodeBodies(e);e.copyBufferToBuffer(record,16,output,32,16);device.queue.submit([e.finish()]);
   assert.deepEqual(Array.from(await readMixedBuffer(device,output)),[17,0,0,0,17,1,0,0,17,0,0,0]);
  }finally{source.destroy();mirror.destroy();record.destroy();fresh.destroy();output.destroy();}
 });
});

test("rebuilt body records equal direct cell and face geometry before and after pose publication",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {createRigidCouplingOracleScene}=await import("../lib/core/scenes");
 const {initializeRigidBodies}=await import("../lib/core/rigid-body");
 const {createUniformSolver}=await import("./helpers/uniform-geometric");
 const {uniformMixedTopologyWGSL}=await import("../lib/methods/uniform/uniform-mixed-topology.wgsl");
 const {uniformDetailModule,uniformDetailPipeline,uniformDetailPick}=await import("../lib/methods/uniform/uniform-detail-fields");
 await withUniformDevice("body geometry record parity",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  Object.assign(scene.rigidBodies[0]!,{motion:"dynamic",density_kg_m3:500,position_m:{x:0,y:.4,z:0},linearVelocity_m_s:{x:.3,y:0,z:0}});
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:import("../lib/methods/uniform/uniform-mixed-frame").UniformMixedFrame}).mixedFrame;
  const solid=frame.solid!,ownership=frame.ownership;
  const {nx,ny,nz}=solver.info,count=nx*ny*nz;
  const a=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const b=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  try{
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const layout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,solid.tileLayout,resultLayout]});
   const groups=[a,b].map(buffer=>device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer}}]}));
   const pipelines:GPUComputePipeline[]=[];let cutPipeline:GPUComputePipeline|undefined;
   for(const cached of [false,true]){
    const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedSolidWGSL(1,solid.coarse!.count,cached)+`
@group(2) @binding(0) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=${count}u){return;}let c=vec3i(i32(g.x%${nx}u),i32((g.x/${nx}u)%${ny}u),i32(g.x/${nx*ny}u));
 output[g.x]=vec4f(umCellOpen(c),umPressureFaceVRecorded(c,0u),umPressureFaceVRecorded(c,1u),umPressureFaceVRecorded(c,2u));
}
@compute @workgroup_size(64) fn cutProbe(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=${count}u){return;}let c=vec3i(i32(g.x%${nx}u),i32((g.x/${nx}u)%${ny}u),i32(g.x/${nx*ny}u));
 output[g.x]=vec4f(0);let tile=umTileKey(c>>vec3u(2u));if(any((c&vec3i(3))!=vec3i(0))||!umKeyStored(tile)){return;}
 var cut=false;for(var i=0u;i<216u;i++){let q=c-1+vec3i(i32(i%6u),i32((i/6u)%6u),i32(i/36u));if(umSolidValid(q)&&umCellOpenGeometry(q)<1.0){cut=true;break;}}
 output[g.x]=vec4f(f32(cut),f32(umSolidStaticCut(tile)),0,0);
}`});
    pipelines.push(await uniformDetailPipeline(device,ownership,{layout,compute:{module,entryPoint:"probe"}}));
    if(!cached)cutPipeline=await uniformDetailPipeline(device,ownership,{layout,compute:{module,entryPoint:"cutProbe"}});
   }
   for(let step=1;step<=3;step++){
    assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    for(const rebuild of [false,true]){
     const e=device.createCommandEncoder();
     if(rebuild){solid.encodeBodies(e);solid.invalidate(true);solid.encodeCoarse(e,ownership.presentation);}
     for(let variant=0;variant<2;variant++){
      const p=e.beginComputePass();p.setPipeline(uniformDetailPick(pipelines[variant]!));p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,solid.tileGroup);p.setBindGroup(2,groups[variant]!);p.dispatchWorkgroups(Math.ceil(count/64));p.end();
     }
     device.queue.submit([e.finish()]);
     const expected=new Uint32Array((await readMixedBuffer(device,a)).buffer),actual=new Uint32Array((await readMixedBuffer(device,b)).buffer);
     assert.deepEqual(actual,expected,`step ${step}, rebuilt ${rebuild}: cached geometry must be bit-identical`);
     if(rebuild){
      const ce=device.createCommandEncoder(),cp=ce.beginComputePass();cp.setPipeline(uniformDetailPick(cutPipeline!));cp.setBindGroup(0,ownership.bindGroup);cp.setBindGroup(1,solid.tileGroup);cp.setBindGroup(2,groups[0]!);cp.dispatchWorkgroups(Math.ceil(count/64));cp.end();device.queue.submit([ce.finish()]);
      const cut=await readMixedBuffer(device,a);for(let i=0;i<cut.length;i+=4)assert.equal(cut[i],cut[i+1],`step ${step}, cut cell ${i/4}`);
     }
    }
   }
  }finally{a.destroy();b.destroy();solver.destroy();}
 });
});
