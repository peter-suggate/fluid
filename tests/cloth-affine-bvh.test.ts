import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {createAffineBroadphase} from '../tools/cloth-affine-bvh';
import {createAffineCCDShadow} from '../tools/cloth-affine-ccd-gpu';
import {affineVertexTriangleCCD,affineEdgeEdgeCCD} from '../tools/cloth-affine-ccd';
import {clothSweptCandidates} from '../tools/cloth-swept-candidates';
import type {AffineVertex,Vec3} from '../tools/cloth-affine-ccd';
test('GPU affine BVH covers exhaustive swept candidates, refits deformations, and reports overflow',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
 try{
  let seed=1729;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return (seed%21-10)*.01;};
  const N=96,x=new Float32Array(4*(N+16)),d=new Float32Array(x.length),rest=new Float32Array(x.length),triangles:number[]=[],edges:number[]=[];
  for(let i=0;i<N;i++){x.set([rnd(),rnd(),rnd()],4*i);d.set([rnd(),rnd(),rnd()],4*i);rest.set([i*.01,0,0],4*i);}
  for(let i=0;i<N;i+=3)triangles.push(i,i+1,i+2);for(let i=0;i<N;i+=2)edges.push(i,i+1);
  const vertices:AffineVertex[]=Array.from({length:N},(_,i)=>({x:Array.from(x.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(d.subarray(4*i,4*i+3)) as unknown as Vec3}));
  const cpu=[...clothSweptCandidates(vertices,triangles,edges,.001)];
  const invalid=x.slice();invalid[0]=NaN;await assert.rejects(createAffineBroadphase(device,invalid,d,rest,new Uint32Array([0,1,2,0]),'vt',.001),/Nonfinite/);
  for(const kind of ['vt','ee'] as const){
   const indices=kind==='vt'?triangles:edges,arity=kind==='vt'?3:2,primitives=new Uint32Array(indices.length/arity*4);for(let i=0;i<indices.length/arity;i++)primitives.set(indices.slice(arity*i,arity*i+arity),4*i);
   for(const filter of ['off','face','full'] as const)for(const packet of (device.features.has('subgroups')?[false,true]:[false]))for(const capacity of [10000,1]){
    const runner=await createAffineBroadphase(device,x,d,rest,primitives,kind,.001,capacity,N,packet,filter);
    try{const enc=device.createCommandEncoder();runner.encode(enc);device.queue.submit([enc.finish()]);
     if(capacity===1){await assert.rejects(runner.read(),/overflow/);continue;}
     const ids=await runner.read(),set=new Set<string>();for(let i=0;i<ids.length;i+=4)set.add(Array.from(ids.subarray(i,i+4)).join(','));assert.equal(set.size,ids.length/4,'duplicate pairs');
     for(const c of cpu.filter(c=>c.kind===kind))if(!set.has(c.ids.join(','))){assert.notEqual(filter,'off',JSON.stringify(c));const p=c.ids.map(j=>vertices[j]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];assert.equal((kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(.001),sameTimeEndpoints:true,exactEndpoints:true}).alpha,1,'Dropped pair must separate over the entire motion');}
     for(const key of set)assert.ok(key.split(',').map(Number).every(i=>i<N),'padded vertices must not be queried');
     // GPU-generated count and queue feed CCD without an intervening readback.
     const narrow=await createAffineCCDShadow(device,x,d,new Uint32Array(),kind,.001,64,256,true,runner);
     try{const joint=device.createCommandEncoder();runner.encode(joint);narrow.encode(joint);device.queue.submit([joint.finish()]);const latest=await runner.read(),result=await narrow.read(latest.length/4);
      for(let i=0;i<latest.length;i+=4){const p=Array.from(latest.subarray(i,i+4),j=>vertices[j]!) as [AffineVertex,AffineVertex,AffineVertex,AffineVertex];const reference=(kind==='vt'?affineVertexTriangleCCD:affineEdgeEdgeCCD)(...p,{thickness:Math.fround(.001),sameTimeEndpoints:true});assert.ok(result[i]!<=reference.alpha+2e-6);assert.notEqual(result[i+2],3);}
      let minimum=1;for(let i=0;i<result.length;i+=4)minimum=Math.min(minimum,result[i]!);assert.equal(runner.minimumAlpha,minimum,'GPU reduction must include every pair');
     }finally{narrow.destroy();}
    }finally{runner.destroy();}
   }
  }
  assert.deepEqual(errors,[]);
 }finally{device.destroy();}
});

test('GPU coherent BVH retains complete current candidates over local escapes and resets',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),errors:string[]=[];device.addEventListener('uncapturederror',(e:GPUUncapturedErrorEvent)=>{e.preventDefault();errors.push(e.error.message);});
 try{
  const N=69,x=new Float32Array(4*N),d=new Float32Array(4*N),triangles=Array.from({length:N},(_,i)=>i),edges=Array.from({length:68},(_,i)=>i);
  for(let i=0;i<N;i++)x.set([(i%5)*.1,Math.floor(i/5)*.1,0],4*i);
  for(const kind of ['vt','ee'] as const){
   const arity=kind==='vt'?3:2,indices=kind==='vt'?triangles:edges,primitives=new Uint32Array(indices.length/arity*4);for(let i=0;i<indices.length/arity;i++)primitives.set(indices.slice(arity*i,arity*i+arity),4*i);
   for(const changedQueries of (kind==='ee'?[false,true]:[false]))for(const compact of [false,true])for(const capacity of [10000,1]){
    const runner=await createAffineBroadphase(device,x,d,x,primitives,kind,.001,capacity,N,false,'off',true,.025,compact,changedQueries);
    try{for(let epoch=0;epoch<12;epoch++){
     const xx=x.slice(),dd=d.slice();for(let i=0;i<N;i++){xx[4*i+2]=Math.fround(.002*epoch*(i%3));if(epoch%3===1&&i%7===0)dd[4*i]=-.15;}
     if(epoch===8)runner.reset();runner.updateMotion(xx,dd);const enc=device.createCommandEncoder();runner.encode(enc);device.queue.submit([enc.finish()]);
     if(capacity===1){await assert.rejects(runner.read(),/overflow/);assert.equal(runner.minimumAlpha,0);continue;}
     const output=await runner.read(),keys=new Set<string>();for(let i=0;i<output.length;i+=4)keys.add(Array.from(output.subarray(i,i+4)).join(','));if(compact)assert.equal(keys.size,output.length/4,'Compacted padded overlap set must stay unique');
     const vs:AffineVertex[]=Array.from({length:N},(_,i)=>({x:Array.from(xx.subarray(4*i,4*i+3)) as unknown as Vec3,d:Array.from(dd.subarray(4*i,4*i+3)) as unknown as Vec3}));
     for(const c of clothSweptCandidates(vs,triangles,edges,.001))if(c.kind===kind)assert.ok(keys.has(c.ids.join(',')),`Missing ${kind} epoch ${epoch}: ${c.ids}`);
    }}finally{runner.destroy();}
   }
  }
  // Historical union exceeds capacity while the fresh complete set still fits.
  const base=new Float32Array([-1,-1,0,0,1,-1,0,0,0,1,0,0,0,0,0,0,0,0,10,0]),zero=new Float32Array(20);
  const tiny=await createAffineBroadphase(device,base,zero,base,new Uint32Array([0,1,2,0]),'vt',0,1,5,false,'off',true,0);
  try{for(const mode of [0,1,2,1]){const xx=base.slice();if(mode!==0){xx[14]=mode===2?0:10;xx[18]=0;}tiny.updateMotion(xx,zero);const enc=device.createCommandEncoder();tiny.encode(enc);device.queue.submit([enc.finish()]);if(mode===2){await assert.rejects(tiny.read(),/overflow/);assert.equal(tiny.minimumAlpha,0);}else{const ids=await tiny.read();assert.equal(ids.length,4);assert.equal(ids[0],mode===0?3:4);}}}finally{tiny.destroy();}
  for(const compact of [false,true]){const cycleX=base.slice(0,16),cycleD=new Float32Array(16),cycle=await createAffineBroadphase(device,cycleX,cycleD,cycleX,new Uint32Array([0,1,2,0]),'vt',0,3,4,false,'off',true,0,compact);
  try{for(let epoch=0;epoch<7;epoch++){const xx=cycleX.slice();xx[14]=epoch%2?10:0;cycle.updateMotion(xx,cycleD);const enc=device.createCommandEncoder();cycle.encode(enc);device.queue.submit([enc.finish()]);const ids=await cycle.read();assert.equal(ids.length/4,compact?(epoch%2?0:1):epoch===6?1:1+Math.floor(epoch/2),'Reappearance duplicates are bounded by complete overflow rebuild');for(let i=0;i<ids.length;i+=4)assert.deepEqual(Array.from(ids.subarray(i,i+4)),[3,0,1,2]);}}finally{cycle.destroy();}}
  for(const scale of [1e-35,1e-10,1,1e10,1e30]){
   const xx=new Float32Array([-scale,-scale,0,0,scale,-scale,0,0,0,scale,0,0,0,0,2*scale,0,0,0,-scale,0]),dd=new Float32Array(20);dd[14]=-4*scale;dd[18]=2*scale;
   const bounds=await createAffineBroadphase(device,xx,dd,xx,new Uint32Array([0,1,2,0]),'vt',0,8,5,false,'off',true,0);
   try{for(let epoch=0;epoch<3;epoch++){const moved=xx.slice();if(epoch===1)moved[12]=Math.fround(scale);bounds.updateMotion(moved,dd);const enc=device.createCommandEncoder();bounds.encode(enc);device.queue.submit([enc.finish()]);const ids=await bounds.read(),keys=new Set<string>();for(let at=0;at<ids.length;at+=4)keys.add(Array.from(ids.subarray(at,at+4)).join(','));const vertices:AffineVertex[]=Array.from({length:5},(_,j)=>({x:Array.from(moved.subarray(4*j,4*j+3)) as unknown as Vec3,d:Array.from(dd.subarray(4*j,4*j+3)) as unknown as Vec3}));for(const c of clothSweptCandidates(vertices,[0,1,2],[],0))assert.ok(keys.has(c.ids.join(',')),`Outward interval coverage at scale ${scale}, epoch ${epoch}`);}}finally{bounds.destroy();}
  }
  // Explicit ownership frontier cases, including a higher-index-only escape.
  for(const changing of ['higher','lower','both'] as const){
   const points=new Float32Array([-1,0,0,0,1,0,0,0,0,-1,0,0,0,1,0,0,-1,-1,20,0,1,1,20,0]),motion=new Float32Array(24);
   if(changing!=='higher'){points[2]=10;points[6]=10;}if(changing!=='lower'){points[10]=-10;points[14]=-10;}
   const frontier=await createAffineBroadphase(device,points,motion,points,new Uint32Array([0,1,0,0,2,3,0,0,4,5,0,0]),'ee',0,1,6,false,'off',true,.01,true,true);
   try{for(const phase of ['empty','contact','overflow','recover'] as const){const xx=points.slice();if(phase!=='empty'){for(const at of [2,6,10,14])xx[at]=0;}if(phase==='overflow'){xx[18]=0;xx[22]=0;}frontier.updateMotion(xx,motion);const enc=device.createCommandEncoder();frontier.encode(enc);device.queue.submit([enc.finish()]);if(phase==='overflow'){await assert.rejects(frontier.read(),/overflow/);assert.equal(frontier.minimumAlpha,0);}else{assert.deepEqual(Array.from(await frontier.read()),phase==='empty'?[]:[0,1,2,3],`${changing} ${phase}`);}}}finally{frontier.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device.destroy();}
});
