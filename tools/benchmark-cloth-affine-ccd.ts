/** Off-live M1 narrowphase experiment. Use fluid-gpu-run.sh timing.
 * CPU broadphase/upload/setup are excluded from kernel timings and reported
 * separately. These numbers are NOT full solver or rendered frame timings.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFileSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {homedir} from 'node:os';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createAffineCCDShadow} from './cloth-affine-ccd-gpu';
import {createAffineBroadphase} from './cloth-affine-bvh';
import {clothSweptCandidates} from './cloth-swept-candidates';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD,type AffineVertex,type Vec3} from './cloth-affine-ccd';
const args=process.argv.slice(2),opt=(k:string,d:string)=>{const i=args.indexOf('--'+k);return i<0?d:args[i+1]!;};
const input=opt('input','/tmp/cloth-directions-forward'),ticks=opt('ticks','231,340').split(',').map(Number),slot=Number(opt('slot','0')),thickness=Number(opt('thickness','.000002'));
const sameTimeEndpoints=args.includes('--same-time-endpoints');
const gpuBroadphase=args.includes('--gpu-broadphase');
const packetBroadphase=args.includes('--packet-broadphase');if(packetBroadphase&&!gpuBroadphase)throw new Error('Packet traversal needs GPU broadphase');
const broadphaseFilter=opt('broadphase-filter','off') as 'off'|'face'|'full';if(broadphaseFilter!=='off'&&!gpuBroadphase)throw new Error('Broadphase filter needs GPU broadphase');
const projectedSupports=args.includes('--projected-supports');
const advancement=args.includes('--advancement');
const hierarchy=opt('hierarchy','rest');if(!['rest','current'].includes(hierarchy))throw new Error('Unknown hierarchy construction pose');
const warmup=Number(opt('warmup','64')),samples=Number(opt('samples','31'));
assert.ok(Number.isInteger(warmup)&&warmup>=3&&Number.isInteger(samples)&&samples>=3);
const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE??resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredFeatures:['timestamp-query',...(packetBroadphase?['subgroups' as GPUFeatureName]:[])],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];
device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
const results=[];
try{
 for(const tick of ticks){
  const manifest=JSON.parse(readFileSync(join(input,`tick${tick}.json`),'utf8'));
  const read=(name:string)=>{const f=manifest.files.find((f:{name:string})=>f.name===name);assert.ok(f);const b=readFileSync(join(input,f.filename));assert.equal(createHash('sha256').update(b).digest('hex'),f.sha256);return Uint8Array.from(b).buffer;};
  assert.equal(new Float32Array(read(`${slot}-control`))[0],0,'Inactive direction');
  const x=new Float32Array(read(`${slot}-x`)),d=new Float32Array(read(`${slot}-d`)),t=new Uint32Array(read('triangles')),e=new Uint32Array(read('edges'));
  const rest=new Float32Array(x.length);
  if(gpuBroadphase){const saved=readFileSync(join(opt('cache',join(homedir(),'code/me/fluid-scratch/cloth-bench')),`sphere-150-tick${tick}.json`),'utf8');assert.equal(createHash('sha256').update(saved).digest('hex'),manifest.checkpointSha256);const positions=JSON.parse(saved).document.geometry.data.restPositions_m;assert.equal(positions.length,manifest.vertices*3);for(let i=0;i<manifest.vertices;i++)rest.set(positions.slice(3*i,3*i+3),4*i);}
  const triangles=Array.from({length:manifest.triangles},(_,i)=>Array.from(t.subarray(4*i,4*i+3))).flat(),edges=Array.from({length:manifest.edges},(_,i)=>Array.from(e.subarray(4*i,4*i+2))).flat();
  const vertices:AffineVertex[]=Array.from({length:manifest.vertices},(_,i)=>({x:Array.from(x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(d.subarray(4*i,4*i+3)) as unknown as Vec3}));
  const begin=performance.now(),candidates={vt:[] as number[],ee:[] as number[]};
  for(const c of clothSweptCandidates(vertices,triangles,edges,thickness))candidates[c.kind].push(...c.ids);
  const cpuBroadphaseMs=performance.now()-begin;
  for(const kind of ['vt','ee'] as const){
   let ids=new Uint32Array(candidates[kind]);
   const inputCandidates=ids.length/4;
   const broadSetupStart=performance.now();
   const broad=gpuBroadphase?await createAffineBroadphase(device,x,d,hierarchy==='current'?x:rest,kind==='vt'?t:e,kind,thickness,1_000_000,manifest.vertices,packetBroadphase,broadphaseFilter,projectedSupports):undefined;
   const broadphaseSetupMs=performance.now()-broadSetupStart;
   const pairKey=(a:Uint32Array,i:number)=>`${a[i]},${a[i+1]},${a[i+2]},${a[i+3]}`;
   if(broad){const enc=device.createCommandEncoder();broad.encode(enc);device.queue.submit([enc.finish()]);const generated=await broad.read(),set=new Set<string>();for(let i=0;i<generated.length;i+=4)set.add(pairKey(generated,i));assert.equal(set.size,generated.length/4,'Duplicate GPU pairs');for(let i=0;i<ids.length;i+=4)if(!set.has(pairKey(ids,i))){assert.notEqual(broadphaseFilter,'off','GPU broadphase omitted a swept candidate');const p=Array.from(ids.subarray(i,i+4),j=>vertices[j]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];const proof=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(thickness),sameTimeEndpoints:true,exactEndpoints:true});assert.equal(proof.reason,'separated','Dropped candidate failed exact full-motion proof');assert.equal(proof.alpha,1);}ids=generated;}
   const count=ids.length/4,cpu=new Float64Array(count),cpuByPair=new Map<string,number>();
   for(let i=0;i<count;i++){const p=Array.from(ids.subarray(4*i,4*i+4),j=>vertices[j]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];cpu[i]=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(thickness),sameTimeEndpoints}).alpha;if(broad)cpuByPair.set(pairKey(ids,4*i),cpu[i]!);}
   for(const wg of (args.includes('--reverse')?[128,64,32]:[32,64,128]) as (32|64|128)[]){
    broad?.setNarrowWorkgroupSize(wg);
    const setup=performance.now(),runner=await createAffineCCDShadow(device,x,d,ids,kind,thickness,wg,256,sameTimeEndpoints,broad,projectedSupports,advancement),setupMs=performance.now()-setup;
    const queryCount=broad?4:2,queries=device.createQuerySet({type:'timestamp',count:queryCount}),resolved=device.createBuffer({size:8*queryCount,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC}),readback=device.createBuffer({size:8*queryCount,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
     const ms=[],broadphaseMs=[],narrowphaseMs=[];
     for(let rep=-warmup;rep<samples;rep++){
      const enc=device.createCommandEncoder();broad?.encode(enc,{querySet:queries,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1});runner.encode(enc,{querySet:queries,beginningOfPassWriteIndex:broad?2:0,endOfPassWriteIndex:broad?3:1});enc.resolveQuerySet(queries,0,queryCount,resolved,0);enc.copyBufferToBuffer(resolved,0,readback,0,8*queryCount);device.queue.submit([enc.finish()]);await readback.mapAsync(GPUMapMode.READ);
      const stamps=new BigUint64Array(readback.getMappedRange().slice(0));readback.unmap();if(rep>=0){ms.push(Number(stamps[queryCount-1]!-stamps[0]!)/1e6);narrowphaseMs.push(Number(stamps[queryCount-1]!-stamps[queryCount-2]!)/1e6);if(broad)broadphaseMs.push(Number(stamps[1]!-stamps[0]!)/1e6);}
     }
     if(broad){ids=await broad.read();assert.equal(ids.length,4*count,'GPU candidate set changed on identical motion');}
     const output=await runner.read(count);let alpha=1,zeroPrefix=0,unresolved=0,budget=0,intervals=0,maxAhead=0;
     for(let i=0;i<count;i++){const a=output[4*i]!,reason=output[4*i+2]!,reference=broad?cpuByPair.get(pairKey(ids,4*i)):cpu[i];assert.ok(reference!==undefined);assert.ok(Number.isFinite(a)&&a>=0&&a<=1);assert.notEqual(reason,3,'Nonfinite GPU geometry');alpha=Math.min(alpha,a);if(a===0)zeroPrefix++;if(reason===1)unresolved++;if(reason===2)budget++;intervals+=output[4*i+1]!;maxAhead=Math.max(maxAhead,a-reference);}
     assert.ok(maxAhead<=2e-6,`GPU prefix exceeds Float64 reference: ${maxAhead}`);
     if(broad)assert.equal(broad.minimumAlpha,alpha,'GPU minimum reduction differs');
     const row={tick,slot,kind,count,inputCandidates,wg,thickness,sameTimeEndpoints,gpuBroadphase,packetBroadphase,broadphaseFilter,projectedSupports,advancement,hierarchy,broadphaseSetupMs,warmup,samples,alpha,zeroPrefix,unresolved,budget,intervals,maxAhead,ms,broadphaseMs,narrowphaseMs,medianGpuMs:[...ms].sort((a,b)=>a-b)[Math.floor(samples/2)],cpuBroadphaseMs,setupMs};results.push(row);console.log(JSON.stringify(row));
    }finally{runner.destroy();queries.destroy();resolved.destroy();readback.destroy();}
   }
   broad?.destroy();
  }
 }
 assert.deepEqual(errors,[]);writeFileSync(opt('out','/tmp/cloth-affine-gpu-benchmark.json'),JSON.stringify({contract:gpuBroadphase?'Off-live affine CLOTH: GPU BVH refit, queue construction, indirect CCD and per-role alpha reduction; excludes solids, cross-role reduction, solver and rendering':'Off-live affine cloth narrowphase only; excludes GPU broadphase, solids, solver and rendering',results},null,2),{flag:'wx'});
}finally{device.destroy();}
