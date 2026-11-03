import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke } from "../lib/core/solid-world";
import type { SceneDescription } from "../lib/core/model";
import { uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

// The solid record's per-tile static flags (umSolidStaticAt) follow every
// static edit: a voxel fill, a terrain carve, their undo and redo, and a
// save and load of the edited document. At each state no flag claims a
// static solid absent where one is (a voxel or boundary record in the tile's
// ring, or terrain at a face sample position there).
test("static tile flags follow voxel fills, terrain carves, undo, redo and save/load",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {createUniformSolver,withUniformDevice}=await import("./helpers/uniform-geometric");
 const {uniformMixedTopologyWGSL}=await import("../lib/methods/uniform/uniform-mixed-topology.wgsl");
 const {uniformDetailModule,uniformDetailPipeline,uniformDetailPick}=await import("../lib/methods/uniform/uniform-detail-fields");
 type Frame=import("../lib/methods/uniform/uniform-mixed-frame").UniformMixedFrame;
 await withUniformDevice("static tile flags",async device=>{
  const scene=createEmptyScene({extents_m:{x:3.2,y:3.2,z:3.2},finestCellSize_m:.1});
  scene.systems={...scene.systems,fluid:true};
  scene.scenery=undefined;scene.solidVoxels=[];scene.solidBoundary=undefined;scene.rigidBodies=[];
  scene.terrain={baseHeight_m:1.6,features:[]};scene.numerics.fixedDt_s=scene.numerics.maxDt_s=1/60;
  scene.fluid.initialLiquidVolumes=[{shape:"box",min_m:{x:-.2,y:1.7,z:-.2},max_m:{x:.2,y:2.1,z:.2}}];
  // A state: per cell, its tile's flags, whether a page stores the tile, and
  // the premise's violations (bit 0 voxel, bit 1 terrain).
  type State={flags:Uint32Array;stored:Uint32Array;bad:Uint32Array};
  const sample=async(solver:Awaited<ReturnType<typeof createUniformSolver>>,step:number):Promise<State>=>{
   assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,[]));await solver.awaitFrameCompletion();
   assert.equal(solver.info.simulationPipelineError,undefined);
   const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame;const solid=frame.solid!,ownership=frame.ownership;
   const {nx,ny,nz}=solver.info,count=nx*ny*nz;
   const out=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   try{
    const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
    const module=uniformDetailModule(device,{code:uniformMixedTopologyWGSL(ownership.capacity,0)+uniformMixedSolidWGSL(1,solid.coarse!.count,true)+`
@group(2) @binding(0) var<storage,read_write> output:array<vec4u>;
@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) g:vec3u){
 if(g.x>=${count}u){return;}let p=vec3i(i32(g.x%${nx}u),i32((g.x/${nx}u)%${ny}u),i32(g.x/${nx*ny}u));
 var flags=0u;for(var k=0u;k<27u;k++){flags|=umSolidStaticAt(p+vec3i(vec3u(k%3u,(k/3u)%3u,k/9u))-vec3i(1));}
 var bad=0u;
 if((flags&UM_STATIC_NO_VOXEL)!=0u&&(umSolidVoxel(p)||umBoundaryRecord(p)!=0u)){bad|=1u;}
 if((flags&UM_STATIC_NO_TERRAIN)!=0u){
  let o=array<f32,5>(0.1,0.15,0.5,0.85,0.9);
  for(var k=0u;k<125u;k++){if(umSolidTerrainAtWorld(umSolidTraceWorld(vec3f(p)+vec3f(o[k%5u],o[(k/5u)%5u],o[k/25u])))){bad|=2u;}}
 }
 output[g.x]=vec4u(umSolidStaticAt(p),select(0u,1u,umKeyStored(umTileKey(p>>vec3u(2u)))),bad,0u);
}`});
    const pipeline=await uniformDetailPipeline(device,ownership,{layout:device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,solid.tileLayout,resultLayout]}),compute:{module,entryPoint:"probe"}});
    const e=device.createCommandEncoder(),pass=e.beginComputePass();
    pass.setPipeline(uniformDetailPick(pipeline));pass.setBindGroup(0,ownership.bindGroup);pass.setBindGroup(1,solid.tileGroup);
    pass.setBindGroup(2,device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer:out}}]}));
    pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();device.queue.submit([e.finish()]);
    const words=new Uint32Array((await readMixedBuffer(device,out)).buffer);
    const state={flags:new Uint32Array(count),stored:new Uint32Array(count),bad:new Uint32Array(count)};
    for(let i=0;i<count;i++){state.flags[i]=words[4*i]!;state.stored[i]=words[4*i+1]!;state.bad[i]=words[4*i+2]!;}
    for(let i=0;i<count;i++)assert.equal(state.bad[i],0,`step ${step}, cell ${i}: a flag claims no static solid where one is (${state.bad[i]})`);
    return state;
   }finally{out.destroy();}
  };
  const at=(x:number,y:number,z:number)=>x+32*(y+32*z);
  // Two states agree on every cell both store.
  const same=(a:State,b:State,label:string)=>{let shared=0;
   for(let i=0;i<a.flags.length;i++)if(a.stored[i]&&b.stored[i]){shared++;assert.equal(a.flags[i],b.flags[i],`${label}: cell ${i}`);}
   assert.ok(shared>1000,`${label}: ${shared} cells stored by both`);};
  const VOXEL=1,TERRAIN=2;
  const solver=await createUniformSolver(device,scene);
  let loaded:Awaited<ReturnType<typeof createUniformSolver>>|undefined;
  try{
   const before=await sample(solver,1);
   // Above the terrain (16 cells) and its ring: both flags. Below it the
   // heightfield is the solid (the ground is not voxels): the voxel flag only.
   assert.equal(before.flags[at(13,24,13)],VOXEL|TERRAIN,"clear air tile");
   assert.equal(before.flags[at(13,13,13)],VOXEL,"buried tile");
   // A voxel fill clears the voxel flag of every tile whose ring it enters, and no other.
   const filled=sceneWithSolidStroke(scene,[{operation:"fill",minimum:[12,24,12],maximumExclusive:[14,26,14],materialId:1}]);
   solver.validateLiveSolidEdit(filled);solver.applySceneUniforms(filled);
   const afterFill=await sample(solver,2);
   assert.equal(afterFill.flags[at(13,24,13)],TERRAIN,"filled tile");
   assert.equal(afterFill.flags[at(11,24,13)],TERRAIN,"tile whose ring the fill enters");
   assert.equal(afterFill.flags[at(20,24,20)],VOXEL|TERRAIN,"distant tile");
   // A carve through the terrain and the fill removes the fill's voxels: the
   // tile is clear again. The cut only opens terrain, so a carved tile keeps
   // its terrain claim (the probe above checks no tile's flags overclaim).
   const carved=sceneWithSolidStroke(filled,[{operation:"clear",affectsScenery:true,minimum:[8,4,8],maximumExclusive:[24,28,24]}]);
   solver.validateLiveSolidEdit(carved);solver.applySceneUniforms(carved);
   const afterCarve=await sample(solver,3);
   assert.equal(afterCarve.flags[at(13,24,13)],VOXEL|TERRAIN,"the carve clears the fill");
   assert.equal(afterCarve.flags[at(11,24,13)],VOXEL|TERRAIN,"and the ring it entered");
   assert.equal(afterCarve.flags[at(16,9,16)],VOXEL,"carved terrain tile");
   // Undo, redo, undo of both: each state's flags again.
   solver.applySceneUniforms(filled);same(await sample(solver,4),afterFill,"undo the carve");
   solver.applySceneUniforms(scene);same(await sample(solver,5),before,"undo the fill");
   solver.applySceneUniforms(filled);same(await sample(solver,6),afterFill,"redo the fill");
   solver.applySceneUniforms(carved);same(await sample(solver,7),afterCarve,"redo the carve");
   // Save and load: the edited document, serialized and parsed, builds the same flags.
   loaded=await createUniformSolver(device,JSON.parse(JSON.stringify(carved)) as SceneDescription);
   same(await sample(loaded,1),afterCarve,"saved and loaded");
  }finally{solver.destroy();loaded?.destroy();}
 });
});
