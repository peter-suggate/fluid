import assert from "node:assert/strict";
import test from "node:test";
import { uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

// The solid record's split: static openness built once and on an edit (the
// full build), body cuts refreshed only for the tiles the swept-body mask
// marks (the bodies-only build). After random motion of several bodies, and
// again after a local voxel edit, the record the frames left (built
// incrementally), and the record a forced full build makes, both answer
// every cell's openness and face apertures bit for bit as the solids read
// directly do (no record); and every tile's cut flag is a cell of it cut.
test("the incremental solid record equals a full rebuild and the direct geometry under random bodies and a local edit",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {createRigidCouplingOracleScene}=await import("../lib/core/scenes");
 const {initializeRigidBodies}=await import("../lib/core/rigid-body");
 const {sceneWithSolidStroke}=await import("../lib/core/solid-world");
 const {createUniformSolver,withUniformDevice}=await import("./helpers/uniform-geometric");
 const {uniformMixedTopologyWGSL}=await import("../lib/methods/uniform/uniform-mixed-topology.wgsl");
 const {uniformDetailModule,uniformDetailPipeline,uniformDetailPick}=await import("../lib/methods/uniform/uniform-detail-fields");
 await withUniformDevice("solid record split",async device=>{
  const scene=createRigidCouplingOracleScene("rigid-hydrostatic");
  // Three free bodies of two shapes, seeded random poses, velocities and spins.
  let seed=0x9e3779b9;const random=()=>{seed^=seed<<13;seed^=seed>>>17;seed^=seed<<5;return (seed>>>0)/2**32;};
  const span=(a:number,b:number)=>a+(b-a)*random();
  const base=scene.rigidBodies[0]!;
  scene.rigidBodies=[0,1,2].map(k=>({...base,id:`split-${k}`,name:`split ${k}`,shape:k===1?"box":"sphere",motion:"dynamic" as const,density_kg_m3:span(400,900),
   dimensions_m:k===1?{x:.08,y:.06,z:.1}:{x:.07,y:.07,z:.07},position_m:{x:span(-.22,.22),y:span(.32,.5),z:span(-.22,.22)},
   linearVelocity_m_s:{x:span(-.6,.6),y:span(-.3,.3),z:span(-.6,.6)},angularVelocity_rad_s:{x:span(-4,4),y:span(-4,4),z:span(-4,4)}}));
  const bodies=initializeRigidBodies(scene.rigidBodies),solver=await createUniformSolver(device,scene);
  const frame=(solver as unknown as {mixedFrame:import("../lib/methods/uniform/uniform-mixed-frame").UniformMixedFrame}).mixedFrame;
  const solid=frame.solid!,ownership=frame.ownership;
  const {nx,ny,nz}=solver.info,count=nx*ny*nz;
  const buffers=[0,1,2,3,4].map(()=>device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}));
  try{
   const resultLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const layout=device.createPipelineLayout({bindGroupLayouts:[ownership.bindLayout,solid.tileLayout,resultLayout]});
   const groups=buffers.map(buffer=>device.createBindGroup({layout:resultLayout,entries:[{binding:0,resource:{buffer}}]}));
   // probe: openness and the three face apertures, direct (cached false) or from the record.
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
   const run=(e:GPUCommandEncoder,pipeline:GPUComputePipeline,into:number)=>{
    const p=e.beginComputePass();p.setPipeline(uniformDetailPick(pipeline));p.setBindGroup(0,ownership.bindGroup);p.setBindGroup(1,solid.tileGroup);p.setBindGroup(2,groups[into]!);p.dispatchWorkgroups(Math.ceil(count/64));p.end();};
   const words=async(into:number)=>new Uint32Array((await readMixedBuffer(device,buffers[into]!)).buffer).slice();
   const floats=(w:Uint32Array)=>new Float32Array(w.buffer);
   // Word for word, failing at the first difference (a whole-array diff of
   // these buffers in the assertion message takes the process's memory).
   const same=(actual:Uint32Array,expected:Uint32Array,label:string)=>{
    assert.equal(actual.length,expected.length,label);let bad=-1,n=0;
    for(let i=0;i<actual.length;i++)if(actual[i]!==expected[i]){if(bad<0)bad=i;n++;}
    if(bad>=0)assert.fail(`${label}: ${n} words differ, the first at cell ${bad>>2} component ${bad&3} (${floats(actual.slice(bad,bad+1))[0]} against ${floats(expected.slice(bad,bad+1))[0]})`);
   };
   // Closed cells in the lowest three layers: the bodies float near the surface, so only the fill changes them.
   const closedLow=(open:Float32Array)=>{let n=0;for(let z=0;z<nz;z++)for(let y=0;y<Math.min(3,ny);y++)for(let x=0;x<nx;x++)if(open[4*(x+nx*(y+ny*z))]===0)n++;return n;};
   let cutTiles=0,bodyCells=0,lowBefore=-1;
   for(let step=1;step<=6;step++){
    if(step===4){
     // A local voxel fill on the floor, away from the bodies: a live edit (the full build next frame).
     const edited=sceneWithSolidStroke(scene,[{operation:"fill",minimum:[1,1,1],maximumExclusive:[4,3,4],materialId:1}]);
     solver.validateLiveSolidEdit(edited);solver.applySceneUniforms(edited);
    }
    assert.ok(solver.advanceTo(step*scene.numerics.fixedDt_s,bodies));await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError,undefined);
    // The frame left the record of the pose it advanced from and published
    // the next. A bodies-only build to that pose, chained from the frame's
    // record (itself bodies-only after steps 1 and 4, which build in full);
    // at steps 3 and 6 then a forced full build of the same pose.
    const rebuild=step===3||step===6;
    const e=device.createCommandEncoder();
    solid.encodeBodies(e);solid.invalidate(true);solid.encodeCoarse(e,ownership.presentation);
    run(e,pipelines[0]!,0);run(e,pipelines[1]!,1);run(e,cutPipeline!,2);
    device.queue.submit([e.finish()]);
    // The full build in a later submission: invalidate() advances the
    // aperture epoch with queue.writeBuffer, which runs ahead of every
    // command buffer submitted after it. In the same submission as the
    // probes above, they would read the new epoch against the stamps of the
    // build before it, and every stored aperture would answer the stale
    // sentinel (-1, "rebuild first"), as a consumer between an edit and its
    // build would see it.
    if(rebuild){const f=device.createCommandEncoder();solid.encodeBodies(f);solid.invalidate();solid.encodeCoarse(f,ownership.presentation);run(f,pipelines[1]!,3);run(f,cutPipeline!,4);device.queue.submit([f.finish()]);}
    const direct=await words(0),incremental=await words(1),cut=floats(await words(2));
    same(incremental,direct,`step ${step}: the bodies-only record must answer the direct geometry bit for bit`);
    for(let i=0;i<cut.length;i+=4){assert.equal(cut[i+1],cut[i],`step ${step}, the bodies-only record's cut flag at cell ${i/4}`);if(cut[i])cutTiles++;}
    if(rebuild){
     same(await words(3),incremental,`step ${step}: a full rebuild must answer the bodies-only record bit for bit`);
     const full=floats(await words(4));for(let i=0;i<full.length;i+=4)assert.equal(full[i+1],cut[i+1],`step ${step}, full and bodies-only cut flags at cell ${i/4}`);
    }
    const open=floats(direct);for(let i=0;i<count;i++)if(open[4*i]!<1)bodyCells++;
    if(step===3)lowBefore=closedLow(open);
    if(step===4)assert.ok(closedLow(open)>lowBefore,`the fill closes floor cells (${lowBefore} -> ${closedLow(open)})`);
   }
   assert.ok(bodyCells>0&&cutTiles>0,`${bodyCells} cut cells, ${cutTiles} cut tiles`);
  }finally{for(const b of buffers)b.destroy();solver.destroy();}
 });
});
