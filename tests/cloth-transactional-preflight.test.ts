/** Independent allocation-contract regression. Does not modify/promote live policy.
 * A proven-doomed levelled round is compared to an eager reference with enough storage.
 * Eager tiny-cap overflow is retained as evidence of the deliberate contract difference.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {OgcContactDetector} from '../lib/cloth/contact/ogc-detector';
import {OGC_PARAM} from '../lib/cloth/contact/ogc-detector-wgsl';
import {OGC_COUNTER as C,OGC_COUNTER_WORDS,OGC_STATUS,OGC_LEVEL_FLAG} from '../lib/cloth/contact/ogc-wgsl';
import {buildOgcEdges} from '../lib/cloth/contact/ogc-topology';
const gpu=!!process.env.WEBGPU_NODE_MODULE;
test('transactional preflight avoids discarded allocation only; accepted overflow and coincidence still fail',{skip:!gpu},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 const pack=(a:number[][])=>new Float32Array(a.flatMap(v=>[...v,0]));
 const storage=(a:Float32Array<ArrayBuffer>)=>{const b=device.createBuffer({size:a.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,a);return b;};
 const pass=(fn:(p:GPUComputePassEncoder)=>void)=>{const e=device.createCommandEncoder(),p=e.beginComputePass();fn(p);p.end();device.queue.submit([e.finish()]);};
 const read=async(b:GPUBuffer,offset:number,size:number)=>{const r=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=device.createCommandEncoder();e.copyBufferToBuffer(b,offset,r,0,size);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const out=r.getMappedRange().slice(0);r.destroy();return out;};
 async function run(gap:number,pairCapacity:number,witness:boolean,nextStep=false){
  const x=pack([[0,0,0],[.01,0,0],[0,.01,0],[0,0,gap],[.01,0,gap],[0,.01,gap]]);
  const positions=storage(x),zero=storage(new Float32Array(x.length));const triangles=new Uint32Array([0,1,2,0,3,4,5,0]);
  const det=await OgcContactDetector.create(device,{vertexCount:6,triangles,edges:buildOgcEdges(triangles)},x,{
   r:.002,kc:1e5,mu:.3,epsV:.01,dt:1/60,gravity:[0,0,0],rqBase:.003,rqMax:.003,envelopeScale:0,contactCapacity:64,pairCapacity,gridCapacity:8192,bucketCount:4096,roundCap:8,
   truncation:'planar',prediction:'inertial',newton:true,conflictWitnesses:true,locality:'off',
  },{positions,velocities:zero,direction:zero});
  // Harness-only parameter offsets; all production kernels and actual witness capture are used.
  const P=(det as unknown as {paramWords:Uint32Array}).paramWords;
  const setup=(pool?:Uint32Array<ArrayBuffer>)=>{
   device.queue.writeBuffer(det.counters,0,new Uint32Array(OGC_COUNTER_WORDS));device.queue.writeBuffer(det.work,0,new Uint32Array(det.work.size/4));device.queue.writeBuffer(det.pred,0,new Uint32Array(det.pred.size/4));
   pass(p=>{det.encodeFrameBegin(p);det.encodeStepBegin(p);det.encodeRoundKernel(p,'prep');});
   const pred=new Float32Array(24);for(let i=0;i<3;i++)pred[4*i+2]=.04;
   device.queue.writeBuffer(det.pred,0,pred);device.queue.writeBuffer(det.work,4*P[OGC_PARAM.envOff]!,new Float32Array(6).fill(.003));
   device.queue.writeBuffer(det.contacts,4*P[OGC_PARAM.levelWord]!,new Uint32Array([0x3f800000,OGC_LEVEL_FLAG.levelled]));device.queue.writeBuffer(det.counters,4*C.rounds,new Uint32Array([1]));
   if(pool)device.queue.writeBuffer(det.counters,4*C.witnessCount,pool);
  };
  const post=(p:GPUComputePassEncoder,probe:boolean)=>{for(const k of det.roundKernels){if(k==='prep'||(!probe&&(k==='witnessProbe'||k==='witnessGate')))continue;det.encodeRoundKernel(p,k);}};
  const snapshot=async()=>{
   const c=await det.readContactEntries(),pairs=await det.readPairs(),bits=new Uint32Array(pairs.planes.buffer);
   const contacts=Array.from(c.counts,(n,v)=>Array.from({length:n},(_,k)=>Array.from(c.entries.slice(4*(v*c.header[1]!+k),4*(v*c.header[1]!+k+1))).join(',')).sort());
   const dat=Array.from({length:pairs.slots},(_,i)=>i).filter(i=>pairs.ids[4*i]!==0xffffffff).map(i=>[...pairs.ids.slice(4*i,4*i+4),...bits.slice(4*i,4*i+4)].join(',')).sort();
   return {contacts,dat,diagnosis:await det.readDiagnosis(),carried:await det.readCarriedState()};
  };
  try{
   setup();pass(p=>post(p,false));const seed=await det.readCounters();
   const pool=new Uint32Array(await read(det.counters,4*C.witnessCount,258*4));assert.ok(pool[0]!>0,'must capture a real unresolved conflict, not manufacture IDs');
   setup(pool);pass(p=>{post(p,witness);det.encodeRoundKernel(p,'redo');for(const k of det.roundKernels)det.encodeRoundKernel(p,k);});
   const counters=await det.readCounters(),state=await snapshot();assert.equal(counters.levelFallback,1);assert.equal(counters.witnessSkips,witness?1:0);
   let next:Awaited<ReturnType<typeof snapshot>>|undefined;
   if(nextStep){pass(p=>{det.encodeFrameBegin(p);det.encodeStepBegin(p);det.encodeDetection(p);det.encodeTruncate(p);det.encodeContactState(p);});next=await snapshot();}
   return{seed,counters,state,next};
  }finally{det.destroy();positions.destroy();zero.destroy();}
 }
 try{
  const reference=await run(.02,128,false,true),speculative=await run(.02,1,true,true),eagerSmall=await run(.02,1,false);
  assert.equal(reference.counters.status,0);assert.equal(speculative.counters.status,0);
  assert.deepEqual(speculative.state,reference.state,'accepted arrays, held/reference/carry state equal sufficiently provisioned eager reference');
  assert.deepEqual(speculative.next,reference.next,'following mandatory step preserves carried/reference state');
  assert.ok(eagerSmall.counters.status&OGC_STATUS.pairOverflow,'document eager discarded-round allocation failure');
  assert.deepEqual(eagerSmall.state,reference.state,'overflow here belonged only to discarded allocation');
  assert.equal(reference.state.dat.length,0);assert.equal(speculative.state.dat.length,0);
  const acceptedOverflow=await run(.001,1,true);
  assert.ok(acceptedOverflow.counters.status&OGC_STATUS.pairOverflow,'full accepted fallback MUST still report actual capacity loss');
  assert.ok(acceptedOverflow.state.dat.length>0,'fallback actually emits DAT constraints');
  const coincident=await run(0,128,true);
  assert.ok(coincident.counters.status&OGC_STATUS.coincident,'actual coincident current geometry MUST still fail after fallback');
  console.log({referenceStatus:reference.counters.status,speculativeStatus:speculative.counters.status,eagerSmallStatus:eagerSmall.counters.status,acceptedOverflowStatus:acceptedOverflow.counters.status,coincidentStatus:coincident.counters.status});
  await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
 }finally{device.destroy();}
});
