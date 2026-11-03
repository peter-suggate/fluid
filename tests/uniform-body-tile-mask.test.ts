import assert from "node:assert/strict";
import test from "node:test";
import { uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

// Independent all-roster oracles of the queries that now visit only their
// tile's candidate bodies (umBodyTileMask): the forms before the masks.
const reference=/* wgsl */`
fn refBodyAt(world:vec3f)->i32{for(var i=0u;i<umBodyCount();i++){if(umInsideBody(i,world)){return i32(i);}}return -1;}
fn refTileClear(p:vec3i)->bool{
 let t=umTileKey(p>>vec3u(2u));
 if(!umKeyStored(t)||umSolidScratch[umSolidCutMap()+t]!=0u){return false;}
 if(umSolidParams.boundary.z<0.5){return true;}
 let world=umSolidWorldCell(p);let h=umSolidParams.cellGravity.xyz;
 for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,world,3.0*max(h.x,max(h.y,h.z)))){return false;}}
 return true;
}
fn refInside(p:vec3i)->bool{
 if(!umSolidEnabled()){return false;}
 if(refTileClear(p)){return false;}
 if(umBoundaryRecord(p)!=0u){return umBoundarySample(vec3f(p)+0.5).w<0.0||umCellInsideTerrain(p)||refBodyAt(umSolidWorldCell(p))>=0;}
 if(umSolidVoxel(p)){return true;}
 return umCellInsideTerrain(p)||refBodyAt(umSolidWorldCell(p))>=0;
}
fn refPoseNear(t:u32,margin:u32)->bool{
 if(!umSolidEnabled()){return false;}if(!umKeyStored(t)){return true;}
 if(umBodyPoseReach(umBodyRegionOfTile(t))>margin){return false;}
 let h=umSolidParams.cellGravity.xyz;let lo=umSolidWorldCell(vec3i(4u*umTileCoord(t)))-.5*h;let hi=lo+4.0*h;
 for(var i=0u;i<umBodyCount();i++){let b=umSolidBodies[i];if(b.dimensions.w<=0.0){continue;}
  if(distance(clamp(b.positionShape.xyz,lo,hi),b.positionShape.xyz)<=b.dimensions.w+f32(margin)*max(h.x,max(h.y,h.z))){return true;}}
 return false;
}
fn refFaceVelocity(id:vec3i,axis:u32)->f32{
 if(!umSolidEnabled()){return 0.0;}
 if(umBodyCount()==0u){return 0.0;}
 var neighbor=id;neighbor[axis]+=1;
 if((umSolidVoxel(id)&&umBoundaryRecord(id)==0u)||(umSolidVoxel(neighbor)&&umBoundaryRecord(neighbor)==0u)){return 0.0;}
 let world=umSolidFaceWorld(id,axis);let h=umSolidParams.cellGravity.xyz;let reach=max(h.x,max(h.y,h.z));
 var candidates=0u;for(var i=0u;i<umBodyCount();i++){if(umBodyNear(i,world,2.0*reach)){candidates|=1u<<i;}}
 if(candidates==0u){return 0.0;}
 var solid=0.0;var velocity=0.0;
 for(var j=0u;j<8u;j++){
  let p=world+vec3f(select(-0.4,0.4,(j&1u)!=0u)*h.x,select(-0.4,0.4,(j&2u)!=0u)*h.y,select(-0.4,0.4,(j&4u)!=0u)*h.z);
  if(umSolidVoxelAtWorld(p)||umSolidTerrainAtWorld(p)){solid+=1.0;continue;}
  let body=umBodyAtCandidates(p,candidates);if(body>=0){solid+=1.0;velocity+=umBodyVelocity(u32(body),world)[axis];}
 }
 if(solid>0.0){return velocity/solid;}
 var nearest=reach;var result=0.0;
 for(var i=0u;i<umBodyCount();i++){if(((candidates>>i)&1u)==0u){continue;}let d=abs(umBodyDistance(i,world));if(d<=nearest){nearest=d;result=umBodyVelocity(i,world)[axis];}}
 return result;
}`;

test("tile body masks cover every reaching body and leave each masked query equal to the all-roster one",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {createRigidCouplingOracleScene}=await import("../lib/core/scenes");
 const {initializeRigidBodies}=await import("../lib/core/rigid-body");
 const {createUniformSolver}=await import("./helpers/uniform-geometric");
 const {uniformMixedTopologyWGSL}=await import("../lib/methods/uniform/uniform-mixed-topology.wgsl");
 const {uniformDetailModule,uniformDetailPipeline,uniformDetailPick}=await import("../lib/methods/uniform/uniform-detail-fields");
 await withUniformDevice("body tile masks",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-float");
  // Terrain beside the bodies: a face a body's sphere reaches weighs the
  // terrain by samples, one it does not by the plane fraction.
  scene.terrain={baseHeight_m:0,features:[{kind:"mound",center_m:{x:.15,z:.15},radius_m:{x:.3,z:.3},amount_m:.34,flat:0}]};
  const first=scene.rigidBodies[0]!;
  Object.assign(first,{position_m:{x:0,y:.4,z:0},linearVelocity_m_s:{x:.3,y:0,z:0}});
  const extra:[string,{x:number,y:number,z:number},{x:number,y:number,z:number}][]=[
   ["box",{x:.12,y:.06,z:.2},{x:-.22,y:.35,z:.2}],["capsule",{x:.05,y:.15,z:.05},{x:.22,y:.3,z:-.2}],
   ["cylinder",{x:.07,y:.1,z:.07},{x:-.2,y:.45,z:-.22}],["cup",{x:.09,y:.12,z:.02},{x:.2,y:.42,z:.22}],
   ["sphere",{x:.04,y:.04,z:.04},{x:.05,y:.2,z:.25}],
  ];
  for(const [i,[shape,d,p]] of extra.entries()){
   scene.rigidBodies.push({...first,id:`mask-${i}`,name:`mask-${i}`,shape:shape as typeof first.shape,dimensions_m:d,position_m:p,
    orientation:{w:Math.cos(.3*i),x:Math.sin(.3*i),y:0,z:0},linearVelocity_m_s:{x:.2*(i%2?1:-1),y:.1,z:-.15},angularVelocity_rad_s:{x:1+i,y:.5,z:0}});
  }
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:import("../lib/methods/uniform/uniform-mixed-frame").UniformMixedFrame}).mixedFrame;
  const solid=frame.solid!,ownership=frame.ownership;
  const {nx,ny,nz}=solver.info,count=nx*ny*nz;
  const out=device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const geometry=[0,1].map(()=>device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}));
  try{
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const layout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,solid.tileLayout,resultLayout]});
   const group=device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:out}}]});
   const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedSolidWGSL(1,solid.coarse!.count,true)+reference+`
@group(2) @binding(0) var<storage,read_write> output:array<u32>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=${count}u){return;}let c=vec3i(i32(g.x%${nx}u),i32((g.x/${nx}u)%${ny}u),i32(g.x/${nx*ny}u));
 var flags=0u;
 if(umCellInsideSolid(c)!=refInside(c)){flags|=1u;}
 if(umSolidTileClear(c)!=refTileClear(c)){flags|=2u;}
 for(var axis=0u;axis<3u;axis++){if(bitcast<u32>(umSolidFaceVelocity(c,axis))!=bitcast<u32>(refFaceVelocity(c,axis))){flags|=4u<<axis;}}
 let t=umTileKey(c>>vec3u(2u));
 for(var m=0u;m<5u;m++){if(umBodyPoseNearTile(t,m)!=refPoseNear(t,m)){flags|=32u<<m;}}
 // Every body whose sphere comes within the mask reach of a point of the
 // cell (centre and corners) is a candidate of its tile.
 let h=umSolidParams.cellGravity.xyz;let reach=f32(UM_BODY_MASK_REACH)*max(h.x,max(h.y,h.z));
 let mask=umBodyCellMask(c);
 for(var i=0u;i<umBodyCount();i++){
  for(var k=0u;k<9u;k++){
   var x=umSolidWorldCell(c);if(k<8u){x+=(vec3f(vec3u(k&1u,(k>>1u)&1u,k>>2u))-0.5)*h;}
   if(umBodyNear(i,x,reach)&&((mask>>i)&1u)==0u){flags|=1024u;}
  }
 }
 if(umKeyStored(t)&&mask!=(1u<<umBodyCount())-1u){flags|=65536u;}
 output[g.x]=flags;
}`});
   const pipeline=await uniformDetailPipeline(device,ownership,{layout,compute:{module,entryPoint:"probe"}});
   // The record after a bodies-only rebuild against the solids' direct evaluation.
   const geometryLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const geometryGroups=geometry.map(buffer=>device.createBindGroup({layout:geometryLayout,entries:[{binding:0,resource:{buffer}}]}));
   const geometryPipelines=await Promise.all([false,true].map(cached=>uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,solid.tileLayout,geometryLayout]}),compute:{module:uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedSolidWGSL(1,solid.coarse!.count,cached)+`
@group(2) @binding(0) var<storage,read_write> output:array<vec4f>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=${count}u){return;}let c=vec3i(i32(g.x%${nx}u),i32((g.x/${nx}u)%${ny}u),i32(g.x/${nx*ny}u));
 output[g.x]=vec4f(umCellOpen(c),umPressureFaceVRecorded(c,0u),umPressureFaceVRecorded(c,1u),umPressureFaceVRecorded(c,2u));
}`}),entryPoint:"probe"}})));
   let active=0;
   for(let step=1;step<=3;step++){
    assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    for(const rebuild of [false,true]){
     const e=device.createCommandEncoder();
     if(rebuild){solid.encodeBodies(e);solid.invalidate(true);solid.encodeCoarse(e,ownership.presentation);}
     const p=e.beginComputePass();p.setPipeline(uniformDetailPick(pipeline));p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,solid.tileGroup);p.setBindGroup(2,group);p.dispatchWorkgroups(Math.ceil(count/64));
     for(let variant=0;variant<2;variant++){p.setPipeline(uniformDetailPick(geometryPipelines[variant]!));p.setBindGroup(2,geometryGroups[variant]!);p.dispatchWorkgroups(Math.ceil(count/64));}
     p.end();device.queue.submit([e.finish()]);
     const direct=new Uint32Array((await readMixedBuffer(device,geometry[0]!)).buffer),recorded=new Uint32Array((await readMixedBuffer(device,geometry[1]!)).buffer);
     assert.deepEqual(recorded,direct,`step ${step}, rebuilt ${rebuild}: the record must equal the direct geometry`);
     const flags=new Uint32Array((await readMixedBuffer(device,out)).buffer);
     for(let i=0;i<count;i++)assert.equal(flags[i]!&0xffff,0,`step ${step}, rebuilt ${rebuild}, cell ${i}: flags ${flags[i]!.toString(2)}`);
     const masked=flags.reduce((n,f)=>n+(f>>16&1),0);
     // After publication the pose is not certified: every query takes the whole roster.
     if(!rebuild)assert.equal(masked,0,`step ${step}: published pose`);else active+=masked;
    }
   }
   assert.ok(active>count/4,`masks narrowed ${active} cell queries`);
  }finally{out.destroy();for(const buffer of geometry)buffer.destroy();solver.destroy();}
 });
});
