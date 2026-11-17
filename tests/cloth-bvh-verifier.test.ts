/** Independent GPU coverage checks for grid and BVH verification.
 * Each probe calls production verification role in isolation, so an omitted role
 * cannot hide behind another role's nonzero aggregate missing counter. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {OgcContactDetector} from '../lib/cloth/contact/ogc-detector';
import {ogcVerifyWGSL,OGC_PARAM as P} from '../lib/cloth/contact/ogc-detector-wgsl';
import {OGC_COUNTER as C} from '../lib/cloth/contact/ogc-wgsl';
import {buildOgcEdges} from '../lib/cloth/contact/ogc-topology';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';

test('grid and BVH verifier find deliberately missing VT/EE/sample/facet contacts and uncovered vertices',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];
 device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 const pack=(x:number[][])=>new Float32Array(x.flatMap(v=>[...v,0]));
 const pass=(fn:(p:GPUComputePassEncoder)=>void)=>{const e=device.createCommandEncoder(),p=e.beginComputePass();fn(p);p.end();device.queue.submit([e.finish()]);};
 const read=async(b:GPUBuffer,offset:number,size:number)=>{const r=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),e=device.createCommandEncoder();e.copyBufferToBuffer(b,offset,r,0,size);device.queue.submit([e.finish()]);await r.mapAsync(GPUMapMode.READ);const out=r.getMappedRange().slice(0);r.destroy();return out;};
 // Fixture geometry is chosen so target nearest features have elementary independent oracles.
 // Large face z=0; detached vertex at its strict interior with gap1mm; other vertices far.
 const vt=[[0,0,0],[.1,0,0],[0,.1,0],[.02,.02,.001],[.2,.2,.04],[.3,.2,.04]];
 // Interior segments perpendicular in XY, separated1mm in Z. Both off-axis vertices are far.
 const ee=[[-.05,0,0],[.05,0,0],[0,-.1,0],[0,-.05,.001],[0,.05,.001],[.1,0,.1]];
 async function fixture(x:number[][],useBVH:boolean){
  const a=pack(x),make=(v:Float32Array<ArrayBuffer>)=>{const b=device.createBuffer({size:v.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(b,0,v);return b;};
  const positions=make(a),zero=make(new Float32Array(a.length)),triangles=new Uint32Array([0,1,2,0,3,4,5,0]),edges=buildOgcEdges(triangles);
  const det=await OgcContactDetector.create(device,{vertexCount:6,triangles,edges},a,{r:.002,kc:1e5,mu:.3,epsV:.01,dt:1/60,gravity:[0,0,0],rqBase:.004,rqMax:.004,envelopeScale:0,contactCapacity:64,pairCapacity:4096,gridCapacity:32768,bucketCount:4096,roundCap:8,truncation:'planar',prediction:'none',broadphase:'bvh',bvhVerify:useBVH,locality:'off',solidCapacityWords:20,solidSampleCapacity:3},{positions,velocities:zero});
  const words=(det as any).paramWords as Uint32Array<ArrayBuffer>;
  // Reserve sample/hull storage but disable field math: these are verifier-only geometry fixtures.
  words[P.solidBase]=0;device.queue.writeBuffer(det.params,0,words);
  pass(p=>{det.encodeFrameBegin(p);det.encodeStepBegin(p);det.encodeDetection(p);});
  // All actual contact entries removed: verifier must independently rediscover them.
  device.queue.writeBuffer(det.contacts,4*words[P.countsOff]!,new Uint32Array(6));
  const layout=device.createBindGroupLayout({entries:[0,1,2,4,5,7,8,9,14].map(binding=>({binding,visibility:GPUShaderStage.COMPUTE,buffer:{type:binding===0?'uniform':binding===1||binding===2||binding===14?'read-only-storage':'storage'}})) as GPUBindGroupLayoutEntry[]});
  const resources:Record<number,GPUBuffer>={0:det.params,1:det.topo,2:positions,4:det.det,5:det.work,7:det.grid,8:det.contacts,9:det.counters,14:det.pred};
  const group=device.createBindGroup({layout,entries:Object.entries(resources).map(([binding,buffer])=>({binding:Number(binding),resource:{buffer}}))}),pl=device.createPipelineLayout({bindGroupLayouts:[layout]});
  async function probe(expression:string){
   const code=ogcVerifyWGSL(useBVH?(det as any).bvhLayout:undefined)+`\n@compute @workgroup_size(1) fn independentProbe(){atomicStore(&counters[OGC_VERIFY_MISSING],${expression});}`;
   const module=device.createShaderModule({code}),info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error'),[]);
   const pipeline=await device.createComputePipelineAsync({layout:pl,compute:{module,entryPoint:'independentProbe'}});
   pass(p=>{p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);});
   return new Uint32Array(await read(det.counters,4*C.verifyMissing,4))[0]!;
  }
  const sample=(points:number[][],facet=false)=>{
   words[P.S]=points.length;words[P.SF]=facet?1:0;device.queue.writeBuffer(det.params,0,words);device.queue.writeBuffer(det.det,6*16,pack(points));
   if(facet)device.queue.writeBuffer(det.contacts,4*4*words[P.facetBase]!,new Uint32Array([0,1,2,0]));
  };
  return {det,positions,words,probe,sample,async destroy(){await device.queue.onSubmittedWorkDone();det.destroy();positions.destroy();zero.destroy();}};
 }
 const edgeResults:number[]=[];
 try{
  for(const useBVH of [false,true]){
   const f=await fixture(vt,useBVH);
   try{
    // Brute geometry oracle: (.02,.02) strictly inside right triangle, distance=.001<r.
    assert.equal(await f.probe('pointFaces(3u,3u)'),1,'missing strict interior vertex-face');
    device.queue.writeBuffer(f.det.contacts,4*(f.words[P.countsOff]!+3),new Uint32Array([1]));
    device.queue.writeBuffer(f.det.contacts,4*(f.words[P.entriesBase]!+4*3*64),new Uint32Array([3,6<<28,0,0]));
    assert.equal(await f.probe('pointFaces(3u,3u)'),0,'matching VT record covers missing feature');
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.countsOff]!,new Uint32Array(6));
    // Same snapshot, static sample above face interior. No solid field approximation involved.
    f.sample([[.025,.01,.001]]);assert.equal(await f.probe('pointFaces(6u,OGC_NONE)'),1,'missing sample-face');
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.countsOff]!,new Uint32Array([1]));
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.entriesBase]!,new Uint32Array([6,6<<28,0,1]));
    assert.equal(await f.probe('pointFaces(6u,OGC_NONE)'),0,'matching sample record covers feature');
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.countsOff]!,new Uint32Array(6));
    // Large horizontal hull facet spans cloth vertex0 at gap1mm; vertices1/2 outside it.
    f.sample([[-.02,-.02,.001],[.04,-.02,.001],[-.02,.04,.001]],true);
    assert.equal(await f.probe('facetVertices(0u)'),1,'missing facet-vertex');
    // An unrelated entry cannot satisfy hull verification.
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.countsOff]!,new Uint32Array([1]));
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.entriesBase]!,new Uint32Array([0,0,0,0]));
    assert.equal(await f.probe('facetVertices(0u)'),1,'unrelated zero entry must not cover hull contact');
    device.queue.writeBuffer(f.det.contacts,4*f.words[P.entriesBase]!,new Uint32Array([0,6<<28,0,2]));
    assert.equal(await f.probe('facetVertices(0u)'),0,'correct hull representation covers vertex');
    // Escape detection envelope without changing the detection snapshot/BVH.
    const moved=pack(vt);moved[0]=-.02;device.queue.writeBuffer(f.positions,0,moved);
    assert.equal(await f.probe('select(0u,1u,uncovered(0u))'),1);
    assert.equal(await f.probe('select(0u,1u,uncovered(1u))'),0);
    // Intersection test still builds its independent current-position grid.
    const hit=await f.det.countIntersections();assert.equal(hit.intersections,0);
    const crossed=pack(vt);crossed[4*3+2]=-.001;device.queue.writeBuffer(f.positions,0,crossed);
    const crossing=await f.det.countIntersections();assert.equal(crossing.intersections,1,"independent intersection-grid positive control");
   }finally{await f.destroy();}
   const f2=await fixture(ee,useBVH);
   try{
    // Find edge0-1 from topology records instead of assuming edge enumeration order.
    const topo=new Uint32Array(await read(f2.det.topo,0,f2.det.topo.size)),base=f2.words[P.edgesOff]!;
    let edge=-1;for(let e=0;e<f2.words[P.E]!;e++)if(topo[base+4*e]===0&&topo[base+4*e+1]===1)edge=e;
    assert.ok(edge>=0);const missing=await f2.probe(`edgeEdges(${edge}u)`);edgeResults.push(missing);assert.equal(missing,1,'exactly one perpendicular interior edge pair is within r');
    let partner=-1;for(let e=0;e<f2.words[P.E]!;e++)if(topo[base+4*e]===3&&topo[base+4*e+1]===4)partner=e;assert.ok(partner>edge);
    device.queue.writeBuffer(f2.det.contacts,4*f2.words[P.countsOff]!,new Uint32Array([1]));
    device.queue.writeBuffer(f2.det.contacts,4*f2.words[P.entriesBase]!,new Uint32Array([edge,(partner|0x80000000)>>>0,0,0]));
    assert.equal(await f2.probe(`edgeEdges(${edge}u)`),0,'matching EE record covers missing pair');
   }finally{await f2.destroy();}
  }
  assert.equal(edgeResults[0],edgeResults[1],"edge verifier A/B exact count");
  await device.queue.onSubmittedWorkDone();assert.deepEqual(errors,[]);
 }finally{device.destroy();}
});
