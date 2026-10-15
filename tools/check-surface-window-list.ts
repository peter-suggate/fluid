/** Full scan/window-list oracle. Diagnostic GPU reads only; run under lockf.
 * <scene> <steps> [--outside] [--overlay-only] [--compare-cache]. Default trough/40, or open/120
 * with --outside. Every failure prints FAIL and exits nonzero. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { findSceneDefinition } from '../lib/core/scenes';
import { sceneFluidFrame } from '../lib/core/scene-lattice-dimensions';
import { uniformNarrowBandMethod as method } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { UNIFORM_PAGE_POOL_STATUS as S, UNIFORM_PAGE_POOL_PUBLISHED as P, uniformPagePoolFatalMessage, type UniformPagePool } from '../lib/methods/uniform/uniform-page-pool';
import { RasterWaterPipeline, surfaceExtractionShader, surfaceVertexCapacity } from '../lib/core/webgpu-water-pipeline';
const args=process.argv.slice(2),outside=args.includes('--outside'),overlayOnly=args.includes('--overlay-only'),pos=args.filter(a=>!a.startsWith('--'));
const compareCache=args.includes('--compare-cache');
const sceneId=pos[0]??(outside||overlayOnly?'nbflip-open-ground':'uniform-trough-dam-break'),steps=Number(pos[1]??(outside||overlayOnly?120:40)),dt=1/60;
assert.ok(Number.isInteger(steps)&&steps>=0);
const definition=findSceneDefinition(sceneId);assert.ok(definition);const scene=structuredClone(sceneDocument(definition));
scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined,water:RasterWaterPipeline|undefined;
const owned:GPUBuffer[]=[],errors:string[]=[];
const timeoutMs=Number(args.find(a=>a.startsWith('--timeout-ms='))?.slice(13)??115000);
assert.ok(Number.isFinite(timeoutMs)&&timeoutMs>0);
const timeout=setTimeout(()=>{console.error(`FAIL timeout: smoke exceeded ${timeoutMs/1000} seconds`);process.exit(1);},timeoutMs);
try{
 const dawn=await import(pathToFileURL(resolve(process.env.WEBGPU_NODE_MODULE??'node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']);const adapter=await gpu.requestAdapter();assert.ok(adapter);
 const d=device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits),requiredFeatures:compareCache?["timestamp-query"]:[]}),{requireWorkerRealm:false});
 d.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
 const buffer=(size:number,usage:number)=>{const b=d.createBuffer({size,usage});owned.push(b);return b;};
 const read=async(b:GPUBuffer,offset=0,size=b.size-offset)=>{const r=d.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});try{const e=d.createCommandEncoder();e.copyBufferToBuffer(b,offset,r,0,size);d.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);return r.getMappedRange().slice(0);}finally{r.unmap();r.destroy();}};
 const u=buffer(416,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
 if(!overlayOnly){water=new RasterWaterPipeline(d,'bgra8unorm',u,buffer(12*64,GPUBufferUsage.STORAGE|GPUBufferUsage.UNIFORM));await water.initialize();}
 console.log('water ready');
 solver=await method.createSolverAsync!(d,scene,'balanced',{...method.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const pool=(solver as unknown as {mixedFrame:{pagePool:UniformPagePool}}).mixedFrame.pagePool;
 console.log(`solver ready ${sceneId} steps=${steps}`);
 for(let step=1;step<=steps;step++){solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();assert.ok(!solver.info.simulationPipelineError,solver.info.simulationPipelineError);}
 const state=new Uint32Array(await read(pool.state));const fatal=uniformPagePoolFatalMessage(pool.layout,state.subarray(0,S.words));assert.equal(fatal,null);
 const published=new Uint32Array(await read(pool.published!));let drawn=0,missing=0;
 const signed=new Int32Array(published.buffer),low=[Infinity,Infinity,Infinity],high=[-Infinity,-Infinity,-Infinity];
 for(let b=1;b<pool.layout.blocks-1;b++){
  const r=P.records+4*b,live=state[pool.layout.state.records+8*b+3]===1;
  assert.equal(published[r+3]===1,live,`block ${b} published state`);if(!live)continue;
  drawn++;const page=[signed[r]!,signed[r+1]!,signed[r+2]!];
  const mask=pool.layout.table-1,at=(page[0]!&mask)+pool.layout.table*((page[1]!&mask)+pool.layout.table*(page[2]!&mask));
  if(state[pool.layout.state.table+at]!==b)missing++;
  for(let a=0;a<3;a++){assert.equal(signed[r+a],state[pool.layout.state.records+8*b+a]!|0);low[a]=Math.min(low[a]!,32*page[a]!);high[a]=Math.max(high[a]!,32*(page[a]!+1));}
 }
 assert.equal(drawn,state[S.resident]);assert.equal(missing,0);
 // Records are signed world pages; the extraction scans lattice cells (world - 32 F).
 const F=[0,1,2].map(a=>state[pool.layout.state.window+a]!|0);for(let a=0;a<3;a++){low[a]=low[a]!-32*F[a]!;high[a]=high[a]!-32*F[a]!;}
 for(let a=0;a<3;a++){assert.equal(signed[P.low+a],Math.min(...[...Array(pool.layout.blocks).keys()].filter(b=>published[P.records+4*b+3]===1).map(b=>signed[P.records+4*b+a]!)),'published low bound');}
 console.log(`PASS overlay step=${steps} drawn=${drawn} resident=${state[S.resident]} missing=${missing} F=${F.join(',')} fatal=none`);
 if(!overlayOnly){
  const source=solver.denseLevelSetVolumeSource;assert.ok(source?.mixedFrame&&source.mixedOwnership);water!.setVolume(solver.surfaceFieldTexture,source);
  const info=solver.info,tank=info.tankCells??[info.nx,info.ny,info.nz],view=new Float32Array(104);
  view.set([scene.container.width_m,scene.container.height_m,scene.container.depth_m,0],12);view.set([...tank,1],20);d.queue.writeBuffer(u,0,view);
  const size=high.map((v,a)=>v-low[a]!) as [number,number,number];
  // Full reference scans the resident bounding rectangle, including AIR gaps.
  // Its only specialization is launch origin/bounds; it still uses latticeValue.
  const anchor='  classifyCube(vec3i(gid));';assert.ok(surfaceExtractionShader.includes(anchor));
  const fullSource=surfaceExtractionShader.replace(anchor,`  if(any(gid>vec3u(${size.map(v=>v+'u').join(',')}))){return;}\n  classifyCube(vec3i(gid)+vec3i(${low.join(',')}));`);
  const full=outside?undefined:await d.createComputePipelineAsync({label:'QA full resident bounding scan',layout:water!.denseExtractionLayoutForQA,compute:{module:d.createShaderModule({code:fullSource}),entryPoint:'extractMain'}});
  type Mesh=Awaited<ReturnType<RasterWaterPipeline['readDenseSurfaceExtractionForQA']>>;
  const extract=async(mode:'full'|'windows')=>{const e=d.createCommandEncoder();water!.encodeDenseSurfaceExtractionForQA(e,...size,mode,false,mode==='full'?full:undefined);d.queue.submit([e.finish()]);const mesh=await water!.readDenseSurfaceExtractionForQA();assert.equal(mesh.vertexAllocator,mesh.vertexCount,'mesh capacity overflow');assert.ok(mesh.vertexCount<=surfaceVertexCapacity(tank[0]!,tank[1]!,tank[2]!), 'production mesh capacity overflow');return mesh;};
  // Canonical position multiset removes append-order nondeterminism. Quantize
  // at 1e-5 cells, then retain full multiplicity in a cryptographic digest.
  const digest=(mesh:Mesh)=>{const rows:string[]=[];const h=Math.min(...source.cellSize_m);for(let i=0;i<mesh.vertices.length;i+=24){const tri:string[]=[];for(let v=0;v<3;v++)tri.push([0,1,2].map(a=>Math.round(mesh.vertices[i+8*v+a]!/h*1e5)).join(','));rows.push(tri.sort().join('/'));}return createHash('sha256').update(rows.sort().join('\n')).digest('hex');};
  if(compareCache){
   // Restore only the collector's original sample loads. Every downstream
   // classifier and polygoniser is identical and reads the same frozen state.
   const cacheRead="let phi = collectedCoarsePhi[cached.x + 7u * (cached.y + 7u * cached.z)];";
   assert.ok(surfaceExtractionShader.includes(cacheRead));
   let referenceSource=surfaceExtractionShader.replace(cacheRead,"let phi = textureLoad(coarseVertexPhi, umWideVertexTexel(vertex), 0).x;");
   const cacheStart=referenceSource.indexOf("  // Load raw texels, retaining the original per-window wall clamp below.");
   const cacheEnd=referenceSource.indexOf("  if (record == UM_FREE) { return; }",cacheStart);
   assert.ok(cacheStart>=0&&cacheEnd>cacheStart);
   referenceSource=referenceSource.slice(0,cacheStart)+referenceSource.slice(cacheEnd);
   const collector=await d.createComputePipelineAsync({label:'QA original uncached window collector',layout:water!.denseExtractionLayoutForQA,compute:{module:d.createShaderModule({code:referenceSource}),entryPoint:'collectWindowsMain'}});
   const querySet=d.createQuerySet({type:'timestamp',count:4});
   const resolved=buffer(32,GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC);
   const timed=async(reference:boolean)=>{
    const e=d.createCommandEncoder();
    water!.encodeDenseSurfaceExtractionForQA(e,...size,'windows',false,undefined,undefined,{collector:reference?collector:undefined,timestampQuerySet:querySet});
    e.resolveQuerySet(querySet,0,4,resolved,0);d.queue.submit([e.finish()]);
    const times=new BigUint64Array(await read(resolved));
    return {classify_ms:Number(times[1]!-times[0]!)/1e6,polygonise_ms:Number(times[3]!-times[2]!)/1e6};
   };
   try{
    await timed(true);const reference=await water!.readDenseSurfaceExtractionForQA();
    await timed(false);const cached=await water!.readDenseSurfaceExtractionForQA();
    assert.equal(reference.vertexAllocator,reference.vertexCount,'reference mesh capacity overflow');
    assert.equal(cached.vertexAllocator,cached.vertexCount,'cached mesh capacity overflow');
    assert.equal(reference.vertexCount,cached.vertexCount);assert.equal(digest(reference),digest(cached));
    console.log(`PASS cached/reference triangles=${cached.vertexCount/3} hash=${digest(cached)}`);
    const rows:{reference:boolean;classify_ms:number;polygonise_ms:number}[]=[];
    for(let iteration=0;iteration<20;iteration++)for(const reference of iteration%2?[false,true]:[true,false]){
     const result=await timed(reference);if(iteration>=4)rows.push({reference,...result});
    }
    const median=(values:number[])=>{values.sort((a,b)=>a-b);return .5*(values[(values.length-1)>>1]!+values[values.length>>1]!);};
    console.log(JSON.stringify({sceneId,steps,resident:drawn,triangles:cached.vertexCount/3,renderExtraction: [true,false].map(reference=>({collector:reference?'reference':'cached',samples:rows.filter(r=>r.reference===reference).length,classify_ms:median(rows.filter(r=>r.reference===reference).map(r=>r.classify_ms)),polygonise_ms:median(rows.filter(r=>r.reference===reference).map(r=>r.polygonise_ms))})),rows}));
   }finally{querySet.destroy();}
  }
  if(outside){
   const mesh=await extract('windows'),frame=sceneFluidFrame(scene);let count=0;
   for(let i=0;i<mesh.vertices.length;i+=24){let beyond=false;for(let v=0;v<3;v++)for(let a=0;a<3;a++){
    const cell=Math.floor((mesh.vertices[i+8*v+a]!-frame.origin_m[a]!)/frame.cellSize_m[a]!);
    const page=Math.floor(cell/32),seedLow=pool.layout.box.origin[a]!;
    beyond ||= page<seedLow||page>=seedLow+pool.layout.box.pages[a]!;
   }if(beyond)count++;}
   assert.ok(count>0,'no triangles outside the seed box');console.log(`PASS outside step=${steps} triangles=${mesh.vertexCount/3} outside=${count} fatal=none`);
  }else{
   const a=await extract('full'),b=await extract('windows'),ha=digest(a),hb=digest(b);
   console.log(`full triangles=${a.vertexCount/3} hash=${ha}`);console.log(`windows triangles=${b.vertexCount/3} hash=${hb}`);
   assert.equal(a.vertexCount,b.vertexCount);assert.equal(ha,hb);console.log(`PASS full/windows ${sceneId} step=${steps} triangles=${a.vertexCount/3}/${b.vertexCount/3}`);
  }
 }
 assert.deepEqual(errors,[]);
}catch(e){console.error('FAIL',e);process.exitCode=1;}
finally{clearTimeout(timeout);water?.destroy();solver?.destroy();for(const b of owned)b.destroy();await device?.queue.onSubmittedWorkDone();device?.destroy();}
