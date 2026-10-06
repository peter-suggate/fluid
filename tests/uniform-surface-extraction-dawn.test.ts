import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import type {FluidRefinementRegion,SceneDescription} from "../lib/core/model";
import {RasterWaterPipeline,surfaceExtractionShader,type WaterSurfaceClassify} from "../lib/core/webgpu-water-pipeline";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {mixedExtent,readMixedTexture,readMixedTileWords} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;

/** The renderer's water surface from the solver's 4h vertex base, on one
 * running solver whose ownership changes live (Requested with no region,
 * a Fine region, Full):
 *  - the base is current: texel g is the stored phi at lattice vertex 4g;
 *  - the window scan (a window's vertices and cells formed once in workgroup
 *    memory, by the window's stencil-word class) finds exactly the cubes the
 *    full lattice scan finds (every cube's corners through the owner-aware
 *    sampler), and the mesh is the same triangles;
 *  - the shipped polygonise (a cube classified once, its nodal values formed
 *    once, a crossing evaluated once per cube) emits the reference
 *    polygonise's mesh (every corner through latticeValue, every normal
 *    sample through umVertexValue, a crossing per triangle corner):
 *    the same position bits per triangle and the same normal bits per vertex,
 *    asserted separately, with cubes of every class in the worklist.
 * The window test and the classified samples are exact, so every comparison
 * is equality: count-only totals, the sorted worklists, the sorted triangles'
 * bits. Compute only: no raster pass is encoded. */
const ARMS:readonly WaterSurfaceClassify[]=["full","windows"];

const base32=():SceneDescription=>{
 const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
 s.container.width_m=s.container.height_m=s.container.depth_m=.8;s.voxelDomain.finestCellSize_m=.025;
 s.rigidBodies=[];s.fluid.inflow=undefined;s.fluid.initialDamBreakDimensions_m={x:.3,y:.4,z:.8};s.solidVoxels=[];s.fluid.refinementRegions=[];
 return s;
};
const base64=():SceneDescription=>{const s=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));s.fluid.refinementRegions=[];return s;};
/** The +x lower quarter of the container, all of z. */
const fineRegion=(s:SceneDescription):FluidRefinementRegion=>({id:"fine",rule:"minimum-cell-size",minimumCellSize_cells:1,maximumCellSize_cells:1,
 min_m:{x:0,y:0,z:-s.container.depth_m/2},max_m:{x:s.container.width_m/2,y:s.container.height_m/2,z:s.container.depth_m/2}});

/** Rows of `width` words, sorted lexicographically: scan order is a race. */
function sortedRows(words:Uint32Array,width:number):Uint32Array[]{
 const rows:Uint32Array[]=[];for(let i=0;i+width<=words.length;i+=width)rows.push(words.subarray(i,i+width));
 return rows.sort((a,b)=>{for(let i=0;i<width;i++){if(a[i]!==b[i])return a[i]!<b[i]!?-1:1;}return 0;});
}
function firstDifference(a:Uint32Array[],b:Uint32Array[]):number{
 if(a.length!==b.length)return Math.min(a.length,b.length);
 for(let r=0;r<a.length;r++){const x=a[r]!,y=b[r]!;for(let i=0;i<x.length;i++)if(x[i]!==y[i])return r;}
 return -1;
}

/** Position words, then normal words, of a 24-word triangle row. */
const POSITION_WORDS=[0,1,2,3,8,9,10,11,16,17,18,19],NORMAL_WORDS=[4,5,6,7,12,13,14,15,20,21,22,23];
/** Triangles ordered by position, then normal: rows pair by position even where normals differ. */
function byPosition(rows:Uint32Array[]):Uint32Array[]{
 return [...rows].sort((a,b)=>{for(const i of POSITION_WORDS)if(a[i]!==b[i])return a[i]!<b[i]!?-1:1;for(const i of NORMAL_WORDS)if(a[i]!==b[i])return a[i]!<b[i]!?-1:1;return 0;});
}
/** Triangles whose position bits differ, vertices of the others whose normal
 * bits differ, and the largest angle between such normals. */
function meshDifference(a:Uint32Array[],b:Uint32Array[]):{positions:number;normals:number;angle_deg:number}{
 let positions=Math.abs(a.length-b.length),normals=0,angle_deg=0;
 for(let r=0;r<Math.min(a.length,b.length);r++){
  const x=a[r]!,y=b[r]!;
  if(POSITION_WORDS.some(i=>x[i]!==y[i])){positions++;continue;}
  if(!NORMAL_WORDS.some(i=>x[i]!==y[i]))continue;
  const p=new Float32Array(x.buffer,x.byteOffset,24),q=new Float32Array(y.buffer,y.byteOffset,24);
  for(const v of [4,12,20]){
   if(x[v]===y[v]&&x[v+1]===y[v+1]&&x[v+2]===y[v+2]&&x[v+3]===y[v+3])continue;
   normals++;angle_deg=Math.max(angle_deg,Math.acos(Math.min(1,Math.max(-1,p[v]!*q[v]!+p[v+1]!*q[v+1]!+p[v+2]!*q[v+2]!)))*180/Math.PI);
  }
 }
 return {positions,normals,angle_deg};
}
/** Vertices whose normal is not a finite unit direction (w = 0) or whose
 * position is not a point (w = 1). Unit to 1e-4: normalize's own rounding is
 * of order 1e-7; this is a test for unnormalised or non-finite output. */
function malformedVertices(rows:Uint32Array[]):number{
 let bad=0;
 for(const row of rows){const f=new Float32Array(row.buffer,row.byteOffset,24);
  for(const v of [0,8,16]){
   const length=Math.hypot(f[v+4]!,f[v+5]!,f[v+6]!);
   if(!(Number.isFinite(f[v]!)&&Number.isFinite(f[v+1]!)&&Number.isFinite(f[v+2]!)&&f[v+3]===1&&Math.abs(length-1)<=1e-4&&f[v+7]===0))bad++;
  }
 }
 return bad;
}
/** The shader's cube classes on the host, over a worklist: cubes whose home
 * stencil is all h, whose base block (the tiles holding lattice vertices
 * b-2..b+2) is all 4h, and the rest (mixed). Reported, and each required
 * in the phase that must hold it (all 4h at zero detail, mixed beside 4h
 * with a region drawn, all h at Full): it shows the lane exercises each
 * class; the mesh equality is the gate. */
function cubeClasses(cubes:Uint32Array[],tiles:Uint32Array,t:readonly[number,number,number],n:readonly[number,number,number]):{fine:number;coarse:number;mixed:number}{
 const fineTile=(g:number[])=>(tiles[g[0]!+t[0]*(g[1]!+t[1]*g[2]!)]!&0x80000000)!==0,classes={fine:0,coarse:0,mixed:0};
 for(const row of cubes){
  const b=[row[0]!&0xffff,row[1]!&0xffff,row[0]!>>>16],home=b.map((v,a)=>Math.min(v,n[a]!-1)>>2);
  let allFine=true;
  for(let k=0;k<27&&allFine;k++){const g=[home[0]!+k%3-1,home[1]!+Math.floor(k/3)%3-1,home[2]!+Math.floor(k/9)-1];
   if(g.some((v,a)=>v<0||v>=t[a]!))continue;
   allFine=fineTile(g);}
  if(allFine){classes.fine++;continue;}
  const g0=b.map(v=>Math.max(v-2,0)>>2),g1=b.map((v,a)=>((Math.min(v+2,n[a]!)+3)>>2)-1);
  let allCoarse=true;
  for(let z=g0[2]!;z<=g1[2]!;z++)for(let y=g0[1]!;y<=g1[1]!;y++)for(let x=g0[0]!;x<=g1[0]!;x++)if(fineTile([x,y,z]))allCoarse=false;
  if(allCoarse)classes.coarse++;else classes.mixed++;
 }
 return classes;
}

/** The shader's window test on the host, from the read-back base and tile
 * words: how many windows the window scan skips (reported; the equality of the
 * scans is the gate, this only shows the skip is exercised). */
function quietWindows(coarse:Float32Array,tiles:Uint32Array,t:readonly[number,number,number],h:number):number{
 const margin=Math.fround(1e-6*h);let quiet=0;
 for(let z=0;z<=t[2];z++)for(let y=0;y<=t[1];y++)for(let x=0;x<=t[0];x++){
  const g=[x,y,z];let coarseOnly=true;
  for(let k=0;k<8&&coarseOnly;k++){const p=[x-(k&1),y-((k>>1)&1),z-(k>>2)];
   if(p.some((v,a)=>v<0||v>=t[a]!))continue;
   if(tiles[p[0]!+t[0]*(p[1]!+t[1]*p[2]!)]!&0x80000000)coarseOnly=false;}
  if(!coarseOnly)continue;
  let air=true,liquid=x>0&&z>0&&g.every((v,a)=>v<t[a]!);
  for(let k=0;k<27;k++){const p=[x+k%3-1,y+Math.floor(k/3)%3-1,z+Math.floor(k/9)-1].map((v,a)=>Math.min(Math.max(v,0),t[a]!));
   const phi=coarse[p[0]!+(t[0]+1)*(p[1]!+(t[1]+1)*p[2]!)]!;air=air&&phi>margin;liquid=liquid&&phi<=0;}
  if(air||liquid)quiet++;
 }
 return quiet;
}

async function lifecycle(device:GPUDevice,scene:SceneDescription,framesPerPhase:number,failures:string[],report:Record<string,unknown>[]){
 const solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,uniformGeometricSolverOptions({detailPolicy:"requested"},scene),()=>{});
 const uniform=device.createBuffer({size:400,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
 const bodies=device.createBuffer({size:768,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const column=device.createTexture({size:[1,1],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});
 const water=new RasterWaterPipeline(device,"rgba8unorm",uniform,bodies);
 try{
  await water.initialize();await water.prepareSurfaceCountForQA();
  // The reference polygonise: the shipped shader with its shortcuts off.
  const referencePolygoniser=await device.createComputePipelineAsync({label:"Polygonise (reference)",layout:water.denseExtractionLayoutForQA,
   compute:{module:device.createShaderModule({label:"Extraction (reference polygonise)",code:surfaceExtractionShader}),entryPoint:"polygoniseMain",constants:{uniformNormalGather:0,shareCubeVertices:0,uniformCubeMemo:0,uniformNormalLoops:0}}});
  const [nx,ny,nz]=mixedExtent(solver.volumeTexture),h=scene.container.height_m/ny;
  const packed=new Float32Array(100);
  packed.set([scene.container.width_m,scene.container.height_m,scene.container.depth_m,scene.container.height_m*scene.container.fillFraction],12);
  packed.set([0,scene.voxelDomain.finestCellSize_m,0,0],16);packed.set([nx,ny,nz,1],20);
  device.queue.writeBuffer(uniform,0,packed);
  water.setVolume(solver.surfaceFieldTexture??solver.volumeTexture,solver.columnBaseTexture??column,solver.denseLevelSetVolumeSource);
  let frame=0;
  const step=async(n:number)=>{for(let i=0;i<n;i++){frame++;assert.ok(solver.advanceTo(frame/30),`advance ${frame}`);await solver.awaitFrameCompletion();}};
  const check=async(label:string,expectFine:(fineTiles:number,tiles:number)=>boolean,expectClasses:(classes:{fine:number;coarse:number;mixed:number},cubes:number)=>boolean)=>{
   const source=solver.denseLevelSetVolumeSource;assert.ok(source?.coarseVertexPhi&&source.mixedOwnership,`${label}: the solver publishes the 4h vertex base`);
   // As the renderer does every frame.
   water.setDenseLevelSetVolumeSource(source);
   const capacity=(solver as unknown as {mixedFrame:{ownership:{capacity:{fineTiles:number}}}}).mixedFrame.ownership.capacity.fineTiles;
   if(Boolean(source.detailVertexPhi)!==capacity>0)failures.push(`${label}: detail field ${source.detailVertexPhi?"named":"absent"} at h-tile capacity ${capacity}`);
   // The base is current: texel g is phi at vertex 4g. Lattice extents come
   // from the field's own extent, never from a texture's dimensions.
   const phiTexture=solver.vertexPhiTexture!,[vx,vy,vz]=mixedExtent(phiTexture);
   assert.deepEqual([vx,vy,vz],[nx+1,ny+1,nz+1],`${label}: vertex extent`);
   const t=[nx>>2,ny>>2,nz>>2] as const,phi=await readMixedTexture(device,phiTexture),coarse=await readMixedTexture(device,source.coarseVertexPhi);
   assert.equal(coarse.length,(t[0]+1)*(t[1]+1)*(t[2]+1),`${label}: the base is one texel per tile corner`);
   const phiBits=new Uint32Array(phi.buffer,phi.byteOffset,phi.length),coarseBits=new Uint32Array(coarse.buffer,coarse.byteOffset,coarse.length);
   let stale=0,first="";
   for(let z=0;z<=t[2];z++)for(let y=0;y<=t[1];y++)for(let x=0;x<=t[0];x++){
    const c=x+(t[0]+1)*(y+(t[1]+1)*z),p=4*x+vx*(4*y+vy*4*z);
    if(coarseBits[c]!==phiBits[p]){if(!stale)first=`${x},${y},${z}: ${coarse[c]} vs ${phi[p]}`;stale++;}
   }
   if(stale)failures.push(`${label}: ${stale} base texels differ from phi at their tile corner (first ${first})`);
   const tiles=await readMixedTileWords(device,solver);let fineTiles=0;for(const word of tiles)if(word&0x80000000)fineTiles++;
   if(!expectFine(fineTiles,tiles.length))failures.push(`${label}: ${fineTiles}/${tiles.length} h tiles`);
   const windows=(t[0]+1)*(t[1]+1)*(t[2]+1),quiet=quietWindows(coarse,tiles,t,h);
   const arms:Record<string,{count:number;cubes:Uint32Array[];triangles:Uint32Array[]}>={};
   for(const arm of ARMS){
    let encoder=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(encoder,nx,ny,nz,arm,true);device.queue.submit([encoder.finish()]);
    const count=(await water.readDenseSurfaceExtractionForQA()).vertexCount;
    encoder=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(encoder,nx,ny,nz,arm,false);device.queue.submit([encoder.finish()]);
    const mesh=await water.readDenseSurfaceExtractionForQA();
    if(mesh.activeCubeCount*8!==mesh.activeCubes.byteLength)failures.push(`${label} ${arm}: the worklist clipped ${mesh.activeCubeCount} cubes`);
    if(mesh.vertexAllocator!==mesh.vertexCount||mesh.vertexCount!==count)failures.push(`${label} ${arm}: mesh ${mesh.vertexCount} vertices (allocated ${mesh.vertexAllocator}), count-only ${count}`);
    arms[arm]={count,cubes:sortedRows(mesh.activeCubes,2),triangles:sortedRows(new Uint32Array(mesh.vertices.buffer,mesh.vertices.byteOffset,mesh.vertices.length),24)};
   }
   const reference=arms.full!;
   if(!(reference.count>0))failures.push(`${label}: the full scan found no surface`);
   for(const arm of ARMS.slice(1)){const a=arms[arm]!;
    if(a.count!==reference.count)failures.push(`${label} ${arm}: count-only ${a.count} vertices, full scan ${reference.count}`);
    const cube=firstDifference(a.cubes,reference.cubes);if(cube>=0)failures.push(`${label} ${arm}: worklist differs (${a.cubes.length} vs ${reference.cubes.length} cubes, first at sorted row ${cube})`);
    const triangle=firstDifference(a.triangles,reference.triangles);if(triangle>=0)failures.push(`${label} ${arm}: mesh differs (${a.triangles.length} vs ${reference.triangles.length} triangles, first at sorted row ${triangle})`);
   }
   // The shipped polygonise against the reference one, on the same worklist.
   const encoder=device.createCommandEncoder();water.encodeDenseSurfaceExtractionForQA(encoder,nx,ny,nz,"windows",false,undefined,referencePolygoniser);device.queue.submit([encoder.finish()]);
   const referenceMesh=await water.readDenseSurfaceExtractionForQA();
   const shipped=byPosition(arms.windows!.triangles),wanted=byPosition(sortedRows(new Uint32Array(referenceMesh.vertices.buffer,referenceMesh.vertices.byteOffset,referenceMesh.vertices.length),24));
   if(referenceMesh.vertexCount!==reference.count)failures.push(`${label}: the reference polygonise emitted ${referenceMesh.vertexCount} vertices, count-only ${reference.count}`);
   const difference=meshDifference(shipped,wanted);
   if(difference.positions)failures.push(`${label}: ${difference.positions} of ${wanted.length} triangles differ from the reference polygonise in position`);
   if(difference.normals)failures.push(`${label}: ${difference.normals} vertex normals differ from the reference normal (largest angle ${difference.angle_deg} deg)`);
   const malformed=malformedVertices(shipped);
   if(malformed)failures.push(`${label}: ${malformed} vertices are not a point with a finite unit normal`);
   const classes=cubeClasses(arms.windows!.cubes,tiles,t,[nx,ny,nz]);
   if(!expectClasses(classes,arms.windows!.cubes.length))failures.push(`${label}: cube classes ${JSON.stringify(classes)} of ${arms.windows!.cubes.length}`);
   report.push({label,frame,dimensions:[nx,ny,nz],fineTiles,tiles:tiles.length,capacity,detailBound:Boolean(source.detailVertexPhi),windows,quiet,vertices:reference.count,cubes:reference.cubes.length,classes,normalDifference:difference});
   return {quiet,windows};
  };
  await step(framesPerPhase);
  const zero=await check("requested, no region",fine=>fine===0,(classes,cubes)=>classes.coarse===cubes);
  if(!(zero.quiet>0&&zero.quiet<zero.windows))failures.push(`zero detail: ${zero.quiet}/${zero.windows} windows skipped; the skip is not exercised`);
  if(solver.denseLevelSetVolumeSource?.detailVertexPhi)failures.push("zero detail: Requested with no region names a detail field");
  const drawn=structuredClone(scene);drawn.fluid.refinementRegions=[fineRegion(scene)];solver.applySceneUniforms(drawn);await solver.pipelinesPrepared();
  await step(framesPerPhase);
  await check("requested, fine region",(fine,tiles)=>fine>0&&fine<tiles,classes=>classes.coarse>0&&classes.mixed>0);
  solver.applyRuntimeValues({detailPolicy:"full"});await solver.pipelinesPrepared();
  await step(framesPerPhase);
  await check("full",(fine,tiles)=>fine===tiles,(classes,cubes)=>classes.fine===cubes);
  solver.applyRuntimeValues({detailPolicy:"requested"});solver.applySceneUniforms(scene);await solver.pipelinesPrepared();
  await step(framesPerPhase);
  await check("requested again, region removed",fine=>fine===0,(classes,cubes)=>classes.coarse===cubes);
 }finally{water.destroy();uniform.destroy();bodies.destroy();column.destroy();solver.destroy();}
}

for(const [name,scene,frames] of [["32³",base32,6],["64³",base64,10]] as const){
 (modulePath?test:test.skip)(`water surface from the 4h vertex base, ${name}: current base, window scan equals full scan and the shipped polygonise equals its reference at zero, partial and full detail`,{timeout:1800000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test",`Uniform surface extraction ${name}`);let device:GPUDevice|undefined;
  try{
   const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
   const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
   device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
   const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
   const failures:string[]=[],report:Record<string,unknown>[]=[];
   await lifecycle(device,scene(),frames,failures,report);
   console.log(JSON.stringify(report));
   assert.deepEqual(errors,[]);assert.deepEqual(failures,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
 });
}
