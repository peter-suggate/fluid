import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipAnalyticScene, NBFLIP_ANALYTIC_CELL_M, NBFLIP_ANALYTIC_DT_S, NBFLIP_ANALYTIC_GRID } from "../lib/core/nbflip-analytic-scenes";
import { cellPoint } from "../lib/core/nbflip-paper-scenes";
import type { UniformFrameBricks } from "../lib/methods/uniform/uniform-frame-bricks";
import { advanceUniform, createUniformSolver, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture, readMixedTileWords } from "./helpers/uniform-mixed-native-fields";
import { uniformTileKey, uniformTileLive } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformDetailBindLayout, uniformDetailGroup, uniformDetailModule, uniformDetailPick, uniformDetailPipeline } from "../lib/methods/uniform/uniform-detail-fields";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { uniformMixedTopologyWGSL } from "../lib/methods/uniform/uniform-mixed-topology.wgsl";

interface Frame{bricks:UniformFrameBricks;fields:{velocity:GPUTexture;velocityScratch:GPUTexture;departure:GPUTexture;phase:GPUTexture};encodeExtension(encoder:GPUCommandEncoder):void;copyWhole(encoder:GPUCommandEncoder,from:GPUTexture,to:GPUTexture):void}

// The frame's velocity extension (UniformFrameExtension: 4h, then h, on the
// frame brick store) on a dam break after a few steps, with an h box
// straddling the column's free surface and the surge front so both levels
// and their seam run. The extension runs twice, its output first holding the
// departure field and then the stored one, so a face the pass owns but never
// wrote differs between the runs (stale) wherever those two differ. Checked:
// - every source face (liquid on either side) keeps its value exactly;
// - every face of a live tile is written (none stale);
// - a side wall face keeps its value (a closed wall is never swept);
// - every extended face lies within its axis's source values: fast sweeping
//   takes each face as a positively weighted mean of upwind neighbours
//   (Zhao 2005; Adalsteinsson & Sethian 1999), so the extension obeys a
//   maximum principle;
// - both levels reach the band.
test("frame extension: sources exact, every face written, a maximum principle at h and 4h",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("frame extension",async device=>{
  const scene=createNbflipAnalyticScene("dam-break"),g=NBFLIP_ANALYTIC_GRID,h=NBFLIP_ANALYTIC_CELL_M;
  scene.fluid.refinementRegions=[{id:"frame-extension-h",min_m:cellPoint(g,h,[8,8,8]),max_m:cellPoint(g,h,[40,40,24]),rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene);
  try{
   for(let step=1;step<=3;step++)await advanceUniform(solver,step*NBFLIP_ANALYTIC_DT_S);
   await solver.awaitFrameCompletion();
   const frame=(solver as unknown as {mixedFrame:Frame}).mixedFrame,f=frame.fields;
   const address=await readUniformLiveAddress(device,solver);
   let e=device.createCommandEncoder();frame.copyWhole(e,f.departure,f.velocityScratch);frame.encodeExtension(e);device.queue.submit([e.finish()]);
   const [fresh,velocity,departure,phase,tiles]=await Promise.all([readMixedTexture(device,f.velocityScratch,address),readMixedTexture(device,f.velocity,address),readMixedTexture(device,f.departure,address),readMixedTexture(device,f.phase,address),readMixedTileWords(device,solver)]);
   e=device.createCommandEncoder();frame.copyWhole(e,f.velocity,f.velocityScratch);frame.encodeExtension(e);device.queue.submit([e.finish()]);
   const again=await readMixedTexture(device,f.velocityScratch,address);
   const fatal=device.createBuffer({size:4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   e=device.createCommandEncoder();e.copyBufferToBuffer(frame.bricks.table,frame.bricks.fatalOffset,fatal,0,4);device.queue.submit([e.finish()]);
   await fatal.mapAsync(GPUMapMode.READ);assert.equal(new Uint32Array(fatal.getMappedRange())[0],0,"no slot refused");fatal.unmap();fatal.destroy();
   const {nx,ny,nz}=solver.info,at=(x:number,y:number,z:number)=>x+nx*(y+ny*z);
   const key=(x:number,y:number,z:number)=>uniformTileKey(address,x>>2,y>>2,z>>2);
   const width=(x:number,y:number,z:number)=>(tiles[key(x,y,z)]!&0x80000000)?1:4;
   const owner=(x:number,y:number,z:number)=>{if(x<0||y<0||z<0||x>=nx||y>=ny||z>=nz)return -1;const w=width(x,y,z);return phase[at(x-x%w,y-y%w,z-z%w)]!;};
   // The faces each level owns: an h cell's +faces, a 4h tile's canonical +face texels (origin + 3 on the axis).
   const faces:{x:number;y:number;z:number;a:number;w:number;i:number;source:boolean}[]=[];
   const low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];let hCells=0;
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    if(!uniformTileLive(address,key(x,y,z)))continue;
    const w=width(x,y,z),l=[x%4,y%4,z%4];if(w===1)hCells++;
    for(let a=0;a<3;a++){
     if(w!==1&&!l.every((v,b)=>v===(b===a?3:0)))continue;
     const i=4*at(x,y,z)+a,q=[x,y,z];q[a]!++;
     const source=owner(x,y,z)>0.5||owner(q[0]!,q[1]!,q[2]!)>0.5;
     if(source){low[a]=Math.min(low[a]!,velocity[i]!);high[a]=Math.max(high[a]!,velocity[i]!);}
     faces.push({x,y,z,a,w,i,source});
    }
   }
   let sources=0,distinct=0,stale=0,walls=0,reached=[0,0],outside=0,worst=0;
   for(const {x,y,z,a,w,i,source} of faces){
    if(source){sources++;assert.equal(fresh[i],velocity[i],`source face ${x},${y},${z}/${a} (width ${w}) keeps its value`);continue;}
    if(departure[i]!==velocity[i]){distinct++;if(fresh[i]!==again[i])stale++;}
    // The side walls (the top may be open).
    if(a!==1&&[x,y,z][a]===[nx,ny,nz][a]!-1){walls++;assert.equal(fresh[i],velocity[i],`wall face ${x},${y},${z}/${a} keeps its value`);continue;}
    if(fresh[i]===velocity[i])continue;
    reached[w===1?0:1]!++;
    const span=high[a]!-low[a]!,slack=1e-6*Math.max(1,Math.abs(high[a]!),Math.abs(low[a]!));
    const off=Math.max(low[a]!-fresh[i]!,fresh[i]!-high[a]!,0);
    if(off>slack){outside++;worst=Math.max(worst,off/Math.max(span,1e-30));}
   }
   console.log(`frame extension: ${nx}x${ny}x${nz}, ${hCells} h cells; ${sources} source faces, ${walls} wall faces, ${reached[0]} h and ${reached[1]} 4h faces extended, ${stale} of ${distinct} distinguishable faces stale, ${outside} outside the source range (worst ${worst.toExponential(3)} of the span)`);
   assert.ok(hCells>1000,`the h box is resident (${hCells} h cells)`);
   assert.ok(sources>500,`the dam break has liquid faces (${sources})`);
   assert.ok(distinct>100,`the departure poison is distinguishable (${distinct})`);
   assert.equal(stale,0,"every face the pass owns is written");
   assert.ok(reached[0]!>20&&reached[1]!>20,`both levels reach the band (h ${reached[0]}, 4h ${reached[1]})`);
   assert.equal(outside,0,"every extended face lies within its axis's source values");
  }finally{solver.destroy();}
 },["subgroups"]);
});

// The pair tail's FLIP delta (projected minus kept, UniformFrameExtension
// "pair") on the same dam break, its kept field (the departure) the
// velocity less a linear shear, D = k (y, z, x) on the x, y, z faces
// (written by a detail-rewritten pass: an h tile's every face, a 4h tile's
// anchors and its subfaces against an h tile, the texels the passes read). The h
// window sweeps the projected field alone; an h face it extends takes the
// 4h band's delta prolonged (linear along the axis, constant across it).
// Checked, at the h tiles' +faces (their delta rows):
// - a source face's delta is exact: its own values' difference;
// - an extended face's delta is the shear's to the 4h band's order: within
//   k times 12 cells (the 4h band's two 4h cells, 8, and 1.5 each for the
//   prolongation's constant across a 4h face and the liquid restriction's
//   offset in it: 11, rounded up), and within a 4h cell on the whole;
// - few extended faces are unknown (the h sweep reached, the 4h band not).
test("pair tail: an h face's delta is exact at sources, a linear shear's to the 4h order where extended",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("frame extension pair",async device=>{
  const scene=createNbflipAnalyticScene("dam-break"),g=NBFLIP_ANALYTIC_GRID,h=NBFLIP_ANALYTIC_CELL_M;
  scene.fluid.refinementRegions=[{id:"frame-extension-h",min_m:cellPoint(g,h,[8,8,8]),max_m:cellPoint(g,h,[40,40,24]),rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1}];
  const solver=await createUniformSolver(device,scene,{particleTransfer:"flip"});
  try{
   for(let step=1;step<=3;step++)await advanceUniform(solver,step*NBFLIP_ANALYTIC_DT_S);
   await solver.awaitFrameCompletion();
   const frame=(solver as unknown as {mixedFrame:Frame&{flipTailGroup:unknown;extension:{delta:GPUBuffer;device:GPUDevice};ownership:UniformMixedOwnership;encodeVelocityExtension(e:GPUCommandEncoder,group:unknown):void}}).mixedFrame,f=frame.fields,o=frame.ownership;
   const address=await readUniformLiveAddress(device,solver);
   const {nx,ny,nz}=solver.info,at=(x:number,y:number,z:number)=>x+nx*(y+ny*z);
   const velocity=await readMixedTexture(device,f.velocity,address);
   const k=0.002,shear=(x:number,y:number,z:number,a:number)=>k*[y+0.5,z+0.5,x+0.5][a]!;
   const resources=uniformDetailBindLayout(frame.extension.device,{entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:"unfilterable-float",viewDimension:"3d"}},{binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:"write-only",format:"rgba32float",viewDimension:"3d"}}]});
   // The solver's device, which the detail rewrite knows.
   const solverDevice=frame.extension.device;
   const module=uniformDetailModule(solverDevice,{code:uniformMixedTopologyWGSL(o.capacity,0)+`
@group(1) @binding(0) var velocity:texture_3d<f32>;
@group(1) @binding(1) var output:texture_storage_3d<rgba32float,write>;
@compute @workgroup_size(64) fn shear(@builtin(global_invocation_id) id:vec3u){
 let n=vec3u(${nx}u,${ny}u,${nz}u);if(id.x>=n.x*n.y*n.z){return;}
 let q=vec3u(id.x%n.x,(id.x/n.x)%n.y,id.x/(n.x*n.y));let p=vec3i(q);
 if(!umCellPresent(p)){return;}let t=umTileAt(q>>vec3u(2u));if(!umTileLive(t)){return;}
 if(umTileWidth(t)!=1u){
  let l=q&vec3u(3u);var own=false;
  for(var a=0u;a<3u;a++){
   var e=vec3u(0u);e[a]=3u;if(all(l==e)){own=true;}
   var r=q;r[a]+=1u;
   if(l[a]==3u&&r[a]<n[a]&&umCellPresent(vec3i(r))){let u=umTileAt(r>>vec3u(2u));if(umTileLive(u)&&umTileWidth(u)==1u){own=true;}}
  }
  if(!own){return;}
 }
 let c=vec3f(q)+0.5;textureStore(output,p,textureLoad(velocity,p,0)-vec4f(${k}*c.y,${k}*c.z,${k}*c.x,0.0));
}`});
   assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   const pipeline=await uniformDetailPipeline(solverDevice,o,{layout:solverDevice.createPipelineLayout({bindGroupLayouts:[o.bindLayout,resources]}),compute:{module,entryPoint:"shear"}});
   const group=uniformDetailGroup(solverDevice,{layout:resources,entries:[{binding:0,resource:f.velocity},{binding:1,resource:f.departure}]});
   const e=device.createCommandEncoder(),pass=e.beginComputePass();pass.setPipeline(uniformDetailPick(pipeline));pass.setBindGroup(0,o.bindGroup);pass.setBindGroup(1,group.group);pass.dispatchWorkgroups(Math.ceil(nx*ny*nz/64));pass.end();
   frame.copyWhole(e,f.velocity,f.velocityScratch);frame.encodeVelocityExtension(e,frame.flipTailGroup);device.queue.submit([e.finish()]);
   const [fresh,departure,phase,tiles,pool]=await Promise.all([readMixedTexture(device,f.velocityScratch,address),readMixedTexture(device,f.departure,address),readMixedTexture(device,f.phase,address),readMixedTileWords(device,solver),readMixedBuffer(device,frame.extension.delta)]);
   const rows=9*512*frame.ownership.capacity.address.blocks;
   const key=(x:number,y:number,z:number)=>uniformTileKey(address,x>>2,y>>2,z>>2);
   const width=(x:number,y:number,z:number)=>(tiles[key(x,y,z)]!&0x80000000)?1:4;
   const owner=(x:number,y:number,z:number)=>{if(x<0||y<0||z<0||x>=nx||y>=ny||z>=nz)return -1;const w=width(x,y,z);return phase[at(x-x%w,y-y%w,z-z%w)]!;};
   let sources=0,extended=0,unknown=0,worst=0,sum=0,bad=0,hf=0;
   for(let z=0;z<nz;z++)for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){
    const t=key(x,y,z);if(!uniformTileLive(address,t)||width(x,y,z)!==1)continue;
    const rank=(tiles[t]!&0x3fffffff)>>>6;
    for(let a=0;a<3;a++){
     const i=4*at(x,y,z)+a,q=[x,y,z];q[a]!++;
     if(a!==1&&[x,y,z][a]===[nx,ny,nz][a]!-1)continue;hf++;
     if(!(Math.abs(velocity[i]!-departure[i]!-shear(x,y,z,a))<=1e-6)){bad++;if(bad<6)console.log("BAD",x,y,z,a,velocity[i],departure[i],shear(x,y,z,a),(tiles[t]!>>>0).toString(16));}
     const d=pool[rows+240*rank+3*((x&3)+4*(y&3)+16*(z&3))+a]!;
     if(owner(x,y,z)>0.5||owner(q[0]!,q[1]!,q[2]!)>0.5){sources++;assert.equal(d,Math.fround(velocity[i]!-departure[i]!),`source face ${x},${y},${z}/${a}: its own values' delta`);continue;}
     if(fresh[i]===velocity[i])continue;
     if(d>=1e30){unknown++;continue;}
     extended++;const off=Math.abs(d-shear(x,y,z,a));worst=Math.max(worst,off);sum+=off;
    }
   }
   console.log('BADCOUNT',bad,'of',hf);const mean=sum/Math.max(extended,1);
   console.log(`pair tail: ${sources} source and ${extended} extended h faces (${unknown} unknown); shear delta error worst ${(worst/k).toFixed(2)} cells, mean ${(mean/k).toFixed(2)} cells (of k)`);
   assert.ok(sources>500&&extended>100,`the h tiles hold sources (${sources}) and extended faces (${extended})`);
   assert.ok(unknown<=0.05*extended,`few extended faces are unknown (${unknown} of ${extended})`);
   assert.ok(worst<=12*k,`every extended face's delta is the shear's within 12 cells of k (worst ${(worst/k).toFixed(2)})`);
   assert.ok(mean<=4*k,`on the whole within a 4h cell of k (mean ${(mean/k).toFixed(2)})`);
  }finally{solver.destroy();}
 },["subgroups"]);
});
