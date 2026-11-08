import assert from "node:assert/strict";
import test from "node:test";
import { createNbflipAnalyticScene, NBFLIP_ANALYTIC_CELL_M, NBFLIP_ANALYTIC_DT_S, NBFLIP_ANALYTIC_GRID } from "../lib/core/nbflip-analytic-scenes";
import { cellPoint } from "../lib/core/nbflip-paper-scenes";
import type { UniformFrameBricks } from "../lib/methods/uniform/uniform-frame-bricks";
import { advanceUniform, createUniformSolver, readUniformLiveAddress, withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedTexture, readMixedTileWords } from "./helpers/uniform-mixed-native-fields";
import { uniformTileKey, uniformTileLive } from "../lib/methods/uniform/uniform-lattice-address";

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
