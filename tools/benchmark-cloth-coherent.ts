/** Off-live sequence benchmark. Run under fluid-gpu-run.sh timing. Measures
 * complete refit/cache update, historical-pair rejection, CCD and reduction.
 * Upload, compilation, CPU hierarchy construction and validation are excluded
 * from GPU timestamps and setup is reported. Not a solver/frame benchmark. */
import assert from 'node:assert/strict';
import {readFileSync,writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {join,resolve} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createAffineBroadphase} from './cloth-affine-bvh';
import {createAffineCCDShadow} from './cloth-affine-ccd-gpu';
import {clothSweptCandidates} from './cloth-swept-candidates';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD,type AffineVertex,type Vec3} from './cloth-affine-ccd';
const args=process.argv.slice(2),opt=(k:string,d:string)=>{const i=args.indexOf('--'+k);return i<0?d:args[i+1]!;};
const compact=args.includes('--compact'),changedQueries=args.includes('--changed-queries');
const cacheStart=Number(opt('cache-start','0'));assert.ok(Number.isInteger(cacheStart)&&cacheStart>=0&&cacheStart<8);
const kinds=opt('kinds','vt,ee').split(',') as ('vt'|'ee')[];assert.ok(kinds.every(k=>k==='vt'||k==='ee'));
const input=opt('input','/tmp/cloth-directions-forward'),ticks=opt('ticks','231,340').split(',').map(Number),warmup=Number(opt('warmup','8')),samples=Number(opt('samples','7')),skin=Number(opt('skin','.0005')),thickness=.000002;
assert.ok(warmup>=3&&samples>=3);
const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);const device=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];
device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});const results:unknown[]=[],provenance:unknown[]=[];
const sourceHashes=Object.fromEntries(['tools/benchmark-cloth-coherent.ts','tools/cloth-affine-bvh.ts','tools/cloth-affine-ccd-gpu.ts','tools/cloth-affine-ccd-wgsl.ts','tools/cloth-swept-candidates.ts','tools/cloth-affine-ccd.ts','lib/cloth/contact/bvh.ts'].map(path=>[path,createHash('sha256').update(readFileSync(resolve(path))).digest('hex')]));
try{for(const tick of ticks){
 const manifest=JSON.parse(readFileSync(join(input,`tick${tick}.json`),'utf8'));
 provenance.push({tick,input:resolve(input),manifestSha256:createHash('sha256').update(readFileSync(join(input,`tick${tick}.json`))).digest('hex'),checkpointSha256:manifest.checkpointSha256,clothSourceSha256:manifest.clothSourceSha256,dt:manifest.dt,documentSolver:manifest.documentSolver,liveOptimizations:manifest.liveOptimizations,slots:manifest.slots,vertices:manifest.vertices});
 assert.equal(manifest.vertices,150*150,'Benchmark must preserve full 150x150 cloth');
 const read=(name:string)=>{const f=manifest.files.find((f:{name:string})=>f.name===name);assert.ok(f);const b=readFileSync(join(input,f.filename));assert.equal(createHash('sha256').update(b).digest('hex'),f.sha256);return Uint8Array.from(b).buffer;};
 const t=new Uint32Array(read('triangles')),e=new Uint32Array(read('edges')),triangles=Array.from({length:manifest.triangles},(_,i)=>Array.from(t.subarray(4*i,4*i+3))).flat(),edges=Array.from({length:manifest.edges},(_,i)=>Array.from(e.subarray(4*i,4*i+2))).flat();
 const motions=Array.from({length:manifest.iterations},(_,slot)=>({slot,x:new Float32Array(read(`${slot}-x`)),d:new Float32Array(read(`${slot}-d`)),active:new Float32Array(read(`${slot}-control`))[0]===0})).filter(m=>m.active);assert.equal(motions.length,8);assert.ok(motions.every((m,i)=>m.slot===i));
 // Both variants receive the same current-pose CPU-built hierarchy. Its setup
 // is not a proposed live GPU rebuild and its cost is reported separately.
 for(const kind of kinds){
 const variants:{run(rep:number):Promise<void>;validate():Promise<void>;destroy():void}[]=[];try{for(const coherent of (args.includes('--reverse')?[true,false]:[false,true])){
  const start=performance.now(),first=motions[0]!,broad=await createAffineBroadphase(device,first.x,first.d,first.x,kind==='vt'?t:e,kind,thickness,1_000_000,manifest.vertices,false,'off',true,coherent?skin:undefined,coherent&&compact,coherent&&kind==='ee'&&changedQueries),narrow=await createAffineCCDShadow(device,first.x,first.d,new Uint32Array(),kind,thickness,64,256,true,broad,true,false,true),primarySetupMs=performance.now()-start;
  const earlyBroad=coherent&&cacheStart>0?await createAffineBroadphase(device,first.x,first.d,first.x,kind==='vt'?t:e,kind,thickness,1_000_000,manifest.vertices,false,'off',true):undefined;
  const earlyNarrow=earlyBroad?await createAffineCCDShadow(device,first.x,first.d,new Uint32Array(),kind,thickness,64,256,true,earlyBroad,true,false,true):undefined;
  const setupMs=performance.now()-start;
  const motionBuffers=motions.map(m=>{const upload=(data:Float32Array<ArrayBuffer>)=>{const buffer=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(buffer,0,data);return buffer;};return {x:upload(m.x),d:upload(m.d)};});
  const queryCount=4*motions.length,queryBytes=8*queryCount,queries=device.createQuerySet({type:'timestamp',count:queryCount}),resolved=device.createBuffer({size:queryBytes,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readback=device.createBuffer({size:queryBytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  const sequenceMs:number[]=[];
  const rows=motions.map(m=>({tick,kind,coherent,compact,changedQueries,cacheStart,cachedActive:coherent&&m.slot>=cacheStart,slot:m.slot,skin,pipelineHierarchySetupMs:setupMs,primaryPipelineHierarchySetupMs:primarySetupMs,batched:true,interleaved:true,sequenceMs,ms:[] as number[],broadMs:[] as number[],narrowMs:[] as number[],count:0,alpha:1,culled:0,maxAhead:0}));
  variants.push({run:async(rep:number)=>{
   broad.reset();const enc=device.createCommandEncoder();
   for(let slot=0;slot<motions.length;slot++){
    const buffers=motionBuffers[slot]!,activeBroad=slot<cacheStart&&earlyBroad?earlyBroad:broad,activeNarrow=slot<cacheStart&&earlyNarrow?earlyNarrow:narrow;
    activeBroad.encodeMotion(enc,buffers.x,buffers.d);activeNarrow.encodeMotion(enc,buffers.x,buffers.d);
    activeBroad.encode(enc,{querySet:queries,beginningOfPassWriteIndex:4*slot,endOfPassWriteIndex:4*slot+1});activeNarrow.encode(enc,{querySet:queries,beginningOfPassWriteIndex:4*slot+2,endOfPassWriteIndex:4*slot+3});
   }
   enc.resolveQuerySet(queries,0,queryCount,resolved,0);enc.copyBufferToBuffer(resolved,0,readback,0,queryBytes);device.queue.submit([enc.finish()]);await readback.mapAsync(GPUMapMode.READ);const times=new BigUint64Array(readback.getMappedRange().slice(0));readback.unmap();
   if(rep>=0){sequenceMs.push(Number(times[queryCount-1]!-times[0]!)/1e6);for(let slot=0;slot<motions.length;slot++){const at=4*slot,row=rows[slot]!;row.ms.push(Number(times[at+3]!-times[at]!)/1e6);row.broadMs.push(Number(times[at+1]!-times[at]!)/1e6);row.narrowMs.push(Number(times[at+3]!-times[at+2]!)/1e6);}}
  }
  ,validate:async()=>{
  // Independent exhaustive validation is a separate replay; no CPU work or
  // per-direction readback intervenes in the timing sequences above.
  broad.reset();for(let slot=0;slot<motions.length;slot++){
   const m=motions[slot]!,row=rows[slot]!,activeBroad=slot<cacheStart&&earlyBroad?earlyBroad:broad,activeNarrow=slot<cacheStart&&earlyNarrow?earlyNarrow:narrow;activeBroad.updateMotion(m.x,m.d);activeNarrow.updateMotion(m.x,m.d);
   const enc=device.createCommandEncoder();activeBroad.encode(enc);activeNarrow.encode(enc);device.queue.submit([enc.finish()]);
     const ids=await activeBroad.read(),out=await activeNarrow.read(ids.length/4),keys=new Map<string,number>();for(let i=0;i<ids.length;i+=4)keys.set(Array.from(ids.subarray(i,i+4)).join(','),i);
     const vertices:AffineVertex[]=Array.from({length:manifest.vertices},(_,i)=>({x:Array.from(m.x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(m.d.subarray(4*i,4*i+3)) as unknown as Vec3}));
     for(const c of clothSweptCandidates(vertices,triangles,edges,thickness))if(c.kind===kind){const at=keys.get(c.ids.join(','));assert.ok(at!==undefined,'Incremental cache omitted actual pair');const p=c.ids.map(i=>vertices[i]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex],ref=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(thickness),sameTimeEndpoints:true});row.maxAhead=Math.max(row.maxAhead,out[at]!-ref.alpha);}
     for(let i=0;i<out.length;i+=4){assert.notEqual(out[i+2],3);row.alpha=Math.min(row.alpha,out[i]!);if(out[i+1]===0)row.culled++;}assert.equal(activeBroad.minimumAlpha,row.alpha);assert.ok(row.maxAhead<=2e-6);row.count=ids.length/4;
  }
  for(const row of rows){const median=(a:number[])=>[...a].sort((a,b)=>a-b)[a.length>>>1]!;const result={...row,medianMs:median(row.ms),medianBroadMs:median(row.broadMs),medianNarrowMs:median(row.narrowMs)};results.push(result);console.log(JSON.stringify(result));}
  },destroy:()=>{for(const b of motionBuffers){b.x.destroy();b.d.destroy();}earlyBroad?.destroy();earlyNarrow?.destroy();broad.destroy();narrow.destroy();queries.destroy();resolved.destroy();readback.destroy();}});
 }
 for(let rep=-warmup;rep<samples;rep++)for(const variant of rep%2===0?variants:[...variants].reverse())await variant.run(rep);
 for(const variant of variants)await variant.validate();
 }finally{for(const variant of variants)variant.destroy();}
 }
}assert.deepEqual(errors,[]);writeFileSync(opt('out','/tmp/cloth-coherent-benchmark.json'),JSON.stringify({contract:'Off-live affine cloth sequence only; not complete solver or rendered frames; current-pose CPU hierarchy construction excluded and reported',adapter:{vendor:adapter.info.vendor,architecture:adapter.info.architecture,device:adapter.info.device,description:adapter.info.description},provenance,sourceHashes,batched:true,interleaved:true,sequenceTiming:'Includes inter-direction GPU motion copies; excludes initial queue reset, first motion copy, CPU upload/setup and validation',warmup,samples,results},null,2),{flag:'wx'});
}finally{device.destroy();}
