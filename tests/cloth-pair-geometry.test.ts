import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {pairGeometryLayout,pairGeometryResetWGSL,pairGeometryBoundsBodyWGSL,pairGeometryQueryWGSL} from '../lib/cloth/contact/pair-geometry-cache';
import {ogcQueryWGSL,parseOgcLocality} from '../lib/cloth/contact/ogc-detector-wgsl';
import {buildOgcTopology,buildOgcEdges} from '../lib/cloth/contact/ogc-topology';
import {buildOgcBVH} from '../lib/cloth/contact/bvh';
import {OGC_COMMON_WGSL,OGC_COUNTER as C,OGC_COUNTER_WORDS} from '../lib/cloth/contact/ogc-wgsl';
const bucket=(v:number,t:number)=>{let h=(Math.imul(v,0x9e3779b9)^Math.imul(t,0x85ebca6b))>>>0;return (h^(h>>>16))&1023;};
test('pair geometry layout rejects aliasing and out-of-range allocations',()=>{
 const m=pairGeometryLayout(13,4,1024);assert.equal(m.xyz,14);assert.equal(m.entries,26);assert.equal(m.words,1+12+7*1024);
 assert.equal(m.base+m.words,m.entries+7*m.buckets);
 const d=pairGeometryLayout(13,4,1024,true);assert.equal(d.recordWords,8);assert.equal(d.decisions!.state,d.entries+8*1024);assert.equal(d.decisions!.changed,d.decisions!.state+20);assert.equal(d.decisions!.globals,d.decisions!.changed+4);assert.equal(d.base+d.words,d.decisions!.globals+3);
 for(const args of [[-1,4,1024],[0,0,1024],[0,4,1000],[0,4,512],[0,4,4194304],[0xffffffff,4,1024],[.5,4,1024]])assert.throws(()=>pairGeometryLayout(...args as [number,number,number]));
 assert.equal(pairGeometryLayout(0,4,2097152).buckets/64,32768);
});
test('pair geometry publication, collisions and exact geometry invalidation',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 const m=pairGeometryLayout(8,4,1024),buffers=new Map<number,GPUBuffer>();
 const make=(b:number,size:number,uniform=false)=>{const x=device.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.set(b,x);};
 make(0,16,true);make(4,64);make(5,4*(m.base+m.words));make(9,4*OGC_COUNTER_WORDS);make(10,16);make(20,128*8);make(21,128*32);
 const put=(b:number,data:Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>,offset=0)=>device.queue.writeBuffer(buffers.get(b)!,offset,data);
 put(0,new Uint32Array([4,128,0,0]));put(10,new Uint32Array([1,0,0,0]));
 const initial=new Float32Array([.2,.3,.4,0, 0,0,0,0, 1,0,0,0, 0,1,0,0]);put(4,initial);
 const common=OGC_COMMON_WGSL+`
 struct Params {N:u32,count:u32,a:u32,b:u32}
 @group(0) @binding(0)var<uniform>P:Params;
 @group(0) @binding(4)var<storage,read_write>xdet:array<vec4f>;
 @group(0) @binding(5)var<storage,read_write>work:array<atomic<u32>>;
 @group(0) @binding(9)var<storage,read_write>counters:array<atomic<u32>>;
 @group(0) @binding(10)var<storage,read_write>args:array<u32>;
 const OGC_ROUNDS=${C.rounds}u;const OGC_FORCE_HELD=${C.forceHeld}u;
 `;
 const pipelines=new Map<string,{p:GPUComputePipeline,g:GPUBindGroup}>();
 const compile=async(name:string,code:string,bindings:number[])=>{const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);const p=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});const g=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:bindings.map(binding=>({binding,resource:{buffer:buffers.get(binding)!}}))});pipelines.set(name,{p,g});};
 try{
 await compile('reset',pairGeometryResetWGSL(m),[5]);await compile('bounds',common+pairGeometryBoundsBodyWGSL(m),[0,4,5,9,10]);
 await compile('query',common+pairGeometryQueryWGSL(m)+`
 @group(0) @binding(20)var<storage,read>keys:array<vec2u>;
 @group(0) @binding(21)var<storage,read_write>out:array<u32>;
 @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)li:u32){
 if(li==0u){pairMemoRound=atomicLoad(&counters[OGC_FORCE_HELD])+1u;pairMemoEnabled=select(0u,1u,atomicLoad(&counters[OGC_ROUNDS])==0u&&atomicLoad(&work[${m.base}u])==0u);}workgroupBarrier();
 if(gid.x<P.count){let key=keys[gid.x];let x=xdet[0].xyz;let a=xdet[1].xyz;let b=xdet[2].xyz;let c=xdet[3].xyz;
 let memo=memoTriangle(key.x,key.y,x,a,b,c);let direct=ogcClosestTriangle(x,a,b,c);
 let cp=memo.bary.x*a+memo.bary.y*b+memo.bary.z*c;let dp=direct.bary.x*a+direct.bary.y*b+direct.bary.z*c;
 out[8u*gid.x]=select(0u,1u,any(bitcast<vec3u>(memo.bary)!=bitcast<vec3u>(direct.bary))||memo.feature!=direct.feature||any(bitcast<vec3u>(cp)!=bitcast<vec3u>(dp))||bitcast<u32>(length(x-cp))!=bitcast<u32>(length(x-dp)));
 }workgroupBarrier();if(li==0u){out[8u*gid.x+1u]=atomicLoad(&wgPairGeometryHits);out[8u*gid.x+2u]=atomicLoad(&wgPairGeometryMisses);out[8u*gid.x+3u]=atomicLoad(&wgPairGeometryCollisions);}}
 `,[0,4,5,9,20,21]);
 const dispatch=(names:string[])=>{const e=device.createCommandEncoder(),p=e.beginComputePass();for(const name of names){const k=pipelines.get(name)!;p.setPipeline(k.p);p.setBindGroup(0,k.g);p.dispatchWorkgroups(name==='reset'?16:name==='query'?2:1);}p.end();device.queue.submit([e.finish()]);};
 const read=async(b:number,offset:number,size:number)=>{const dst=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffers.get(b)!,offset,dst,0,size);device.queue.submit([e.finish()]);await dst.mapAsync(GPUMapMode.READ);const result=new Uint32Array(dst.getMappedRange().slice(0));dst.destroy();return result;};
 const counter=(at:number,v:number)=>put(9,new Uint32Array([v]),4*at);
 let other=1;while(bucket(0,other)!==bucket(0,0))other++;assert.notEqual(other,0);
 const keys=new Uint32Array(256);const setKeys=(collision=false)=>{for(let i=0;i<128;i++){keys[2*i]=0;keys[2*i+1]=collision&&i%2?other:0;}put(20,keys);};setKeys();
 const query=async(label:string)=>{dispatch(['bounds','query']);const a=await read(21,0,128*32);for(let i=0;i<128;i++)assert.equal(a[8*i],0,label+' geometry '+i);return{hits:a[1]!+a[513]!,misses:a[2]!+a[514]!};};
 dispatch(['reset']);let q=await query('concurrent first publication');assert.equal(q.hits,0);assert.equal(q.misses,128);
 q=await query('same wave never consumes');assert.equal(q.hits,0);
 counter(C.forceHeld,1);q=await query('next wave published payload');assert.equal(q.hits,128);
 setKeys(true);q=await query('colliding exact keys fall back');assert.equal(q.hits,64);assert.equal(q.misses,64);
 // Poison payload of an unfinished claim: same-wave readers must not touch it.
 const at=m.entries+7*bucket(0,0);put(5,new Uint32Array([2,0,0,0x7fc00000,0x7fc00000,0x7fc00000,0xffffffff]),4*at);
 setKeys();q=await query('unfinished same-wave poison');assert.equal(q.hits,0);
 // Reset invalidates all old stamps and takes a new exact XYZ snapshot.
 dispatch(['reset']);counter(C.forceHeld,0);await query('reset');counter(C.forceHeld,1);q=await query('reset publication');assert.equal(q.hits,128);
 const changed=initial.slice();changed[0]=.25;put(4,changed);q=await query('XYZ changes');assert.equal(q.hits,0);assert.equal((await read(5,4*m.base,4))[0],1);
 put(4,initial);q=await query('sticky invalidation');assert.equal(q.hits,0);
 dispatch(['reset']);counter(C.forceHeld,0);await query('signed zero seed');counter(C.forceHeld,1);changed.set(initial);changed[4]=-0;put(4,changed);q=await query('signed zero mismatch');assert.equal(q.hits,0);assert.equal((await read(5,4*m.base,4))[0],1);
 put(4,initial);dispatch(['reset']);counter(C.forceHeld,0);await query('w seed');counter(C.forceHeld,1);changed.set(initial);changed[3]=987;put(4,changed);q=await query('w-only allowed');assert.equal(q.hits,128);
 counter(C.rounds,1);q=await query('nonmandatory bypass');assert.equal(q.hits,0);assert.equal(q.misses,128);
 // Inactive bounds must not accidentally refresh or invalidate the snapshot.
 counter(C.rounds,0);put(10,new Uint32Array([0,0,0,0]));changed[0]=.8;put(4,changed);dispatch(['bounds']);assert.equal((await read(5,4*m.base,4))[0],0);
 put(10,new Uint32Array([1,0,0,0]));q=await query('reactivated guard');assert.equal(q.hits,0);
 assert.deepEqual(errors,[]);
 }finally{await device.queue.onSubmittedWorkDone();for(const b of buffers.values())b.destroy();device.destroy();}
});

function extractPairFunction(source:string,name:string){const start=source.indexOf(`fn ${name}(`);assert.ok(start>=0,name);const open=source.indexOf('{',start);let depth=1;for(let i=open+1;i<source.length;i++){if(source[i]==='{')depth++;if(source[i]==='}'&&!--depth)return source.slice(start,i+1);}throw new Error(name);}
test('packed DAT decisions track exact wave inputs and retain original predicate',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 const m=pairGeometryLayout(64,4,1024,true),buffers=new Map<number,GPUBuffer>();
 const make=(b:number,size:number,uniform=false)=>{const x=device.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.set(b,x);};
 make(0,48,true);make(4,64);make(5,4*(m.base+m.words));make(9,4*OGC_COUNTER_WORDS);make(10,16);make(14,64);make(20,128*8);make(21,128*32);
 const put=(b:number,data:Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>,offset=0)=>device.queue.writeBuffer(buffers.get(b)!,offset,data);
 const params=new Uint32Array([4,128,0,0,0,4,8,0,0,0,0,0]);const floats=new Float32Array(params.buffer);floats[8]=.5;floats[9]=.0001;put(0,params);
 put(14,new Float32Array(16));put(5,new Float32Array(4).fill(.001));put(10,new Uint32Array([1,0,0,0]));
 const initial=new Float32Array([.2,.3,.4,0, 0,0,0,0, 1,0,0,0, 0,1,0,0]);put(4,initial);
 const common=OGC_COMMON_WGSL+`
 struct Params {N:u32,count:u32,a:u32,b:u32,envOff:u32,heldOff:u32,bodyOff:u32,bodyBase:u32,gammaR:f32,sepEps:f32,pad0:u32,pad1:u32}
 @group(0) @binding(0)var<uniform>P:Params;
 @group(0) @binding(4)var<storage,read_write>xdet:array<vec4f>;
 @group(0) @binding(5)var<storage,read_write>work:array<atomic<u32>>;
 @group(0) @binding(9)var<storage,read_write>counters:array<atomic<u32>>;
 @group(0) @binding(14)var<storage,read>pred:array<vec4f>;
 @group(0) @binding(10)var<storage,read_write>args:array<u32>;
 const OGC_ROUNDS=${C.rounds}u;const OGC_FORCE_HELD=${C.forceHeld}u;
 `;
 const pipelines=new Map<string,{p:GPUComputePipeline,g:GPUBindGroup}>();
 const compile=async(name:string,code:string,bindings:number[])=>{const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==='error').map(m=>m.message),[]);const p=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});const g=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:bindings.map(binding=>({binding,resource:{buffer:buffers.get(binding)!}}))});pipelines.set(name,{p,g});};
 try{
 const triangles=new Uint32Array([1,2,3,0]);const top=buildOgcTopology({vertexCount:4,triangles,edges:buildOgcEdges(triangles)},initial);const bvh=buildOgcBVH(top,initial,4,1);
 const generated=ogcQueryWGSL('queryVertex',parseOgcLocality('off'),undefined,false,{...bvh,pairGeometry:m});
 const fragments=['spreadVT','datCaps','datRound','absMax','datPointSeg','datPointTri','datLevels','datPathVT','datPairVTUncached','datPairVT'].map(n=>extractPairFunction(generated,n)).join('\n');
 const access=`fn predOf(i:u32)->vec3f{return pred[i].xyz;}fn envOf(i:u32)->f32{return bitcast<f32>(atomicLoad(&work[P.envOff+i]));}fn triEnv(t:vec3u)->f32{return max(envOf(t.x),max(envOf(t.y),envOf(t.z)));}fn pathStray(i:u32)->f32{if(P.bodyBase==0u||(atomicLoad(&work[P.heldOff+i])&64u)==0u){return 0.0;}return bitcast<f32>(atomicLoad(&work[P.bodyOff+8u*i+7u]));}`;
 await compile('reset' ,pairGeometryResetWGSL(m),[5]);await compile('bounds',common+pairGeometryBoundsBodyWGSL(m),[0,4,5,9,10,14]);
 await compile('query',common+pairGeometryQueryWGSL(m)+access+fragments+`
 @group(0) @binding(20)var<storage,read>keys:array<vec2u>;
 @group(0) @binding(21)var<storage,read_write>out:array<u32>;
 @compute @workgroup_size(64)fn main(@builtin(global_invocation_id)gid:vec3u,@builtin(local_invocation_index)li:u32){
 if(li==0u){pairMemoRound=atomicLoad(&counters[OGC_FORCE_HELD])+1u;pairMemoEnabled=select(0u,1u,atomicLoad(&counters[OGC_ROUNDS])==0u&&atomicLoad(&work[${m.base}u])==0u);}workgroupBarrier();
 if(gid.x<P.count){let key=keys[gid.x];let x=xdet[0].xyz;let a=xdet[1].xyz;let b=xdet[2].xyz;let c=xdet[3].xyz;
 let memo=memoTriangle(key.x,key.y,x,a,b,c);let direct=ogcClosestTriangle(x,a,b,c);
 let cp=memo.bary.x*a+memo.bary.y*b+memo.bary.z*c;let dp=direct.bary.x*a+direct.bary.y*b+direct.bary.z*c;
 let value=datPairVT(x,length(x-cp),0u,vec3u(1,2,3),a,b,c,envOf(0u));
 let directValue=datPairVTUncached(x,length(x-dp),0u,vec3u(1,2,3),a,b,c,envOf(0u));
 out[8u*gid.x]=select(0u,1u,value!=directValue);out[8u*gid.x+4u]=select(0u,1u,value);
 }workgroupBarrier();if(li==0u){out[8u*gid.x+1u]=atomicLoad(&wgPairDecisionHits);out[8u*gid.x+2u]=atomicLoad(&wgPairDecisionMisses);out[8u*gid.x+3u]=atomicLoad(&wgPairGeometryHits);}}
 `,[0,4,5,9,14,20,21]);
 const dispatch=(names:string[])=>{const e=device.createCommandEncoder(),p=e.beginComputePass();for(const name of names){const k=pipelines.get(name)!;p.setPipeline(k.p);p.setBindGroup(0,k.g);p.dispatchWorkgroups(name==='reset'?16:name==='query'?2:1);}p.end();device.queue.submit([e.finish()]);};
 const read=async(b:number,offset:number,size:number)=>{const dst=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffers.get(b)!,offset,dst,0,size);device.queue.submit([e.finish()]);await dst.mapAsync(GPUMapMode.READ);const result=new Uint32Array(dst.getMappedRange().slice(0));dst.destroy();return result;};
 const counter=(at:number,v:number)=>put(9,new Uint32Array([v]),4*at);
 let other=1;while(bucket(0,other)!==bucket(0,0))other++;assert.notEqual(other,0);
 const keys=new Uint32Array(256);const setKeys=(collision=false)=>{for(let i=0;i<128;i++){keys[2*i]=0;keys[2*i+1]=collision&&i%2?other:0;}put(20,keys);};setKeys();
 const query=async(label:string)=>{dispatch(['bounds','query']);const a=await read(21,0,128*32);for(let i=0;i<128;i++)assert.equal(a[8*i],0,label+' geometry '+i);return{hits:a[1]!+a[513]!,misses:a[2]!+a[514]!,value:a[4]};};
 let wave=0;const advance=()=>counter(C.forceHeld,++wave);
 const prediction=new Float32Array(16);
 dispatch(['reset']);let q=await query('first evaluation');assert.equal(q.hits,0);assert.equal(q.value,0);
 advance();q=await query('later immutable geometry and decision');assert.equal(q.hits,128);
 // Change prediction so the previously separated point crosses the triangle.
 prediction[2]=-.8;put(14,prediction);advance();q=await query('changed prediction flips result');assert.equal(q.value,1);assert.ok(q.misses>0);
 q=await query('same-wave packed result');assert.equal(q.hits,128);
 prediction[2]=0;put(14,prediction);advance();q=await query('opposite Boolean new stamp');assert.equal(q.value,0);assert.ok(q.misses>0);
 // A negative zero changes the exact state stamp even though geometry is equal.
 prediction[0]=-0;put(14,prediction);advance();q=await query('negative zero prediction');assert.ok(q.misses>0);assert.equal((await read(5,4*(m.decisions!.changed),4))[0],wave+1);
 // A high envelope forces retention; then return to separation near the threshold.
 put(5,new Float32Array(4).fill(1));advance();q=await query('envelope transition');assert.equal(q.value,1);assert.ok(q.misses>0);
 for(const env of [.7997,.7998,.7999,.8,.8001]){put(5,new Float32Array(4).fill(env));advance();await query('threshold '+env);}
 // Bitwise globals invalidate all entries, including field absent/present changes.
 for(const field of [8,9,7]){
  dispatch(['reset']);wave=0;counter(C.forceHeld,0);await query('global seed');advance();q=await query('global seed hit');assert.equal(q.hits,128);
  if(field===7)params[field]=params[field]===0?1:0;else floats[field]=floats[field]!+.00001;put(0,params);advance();q=await query('global invalidation '+field);assert.equal(q.hits,0);assert.equal((await read(5,4*m.base,4))[0],1);
 }
 // Body stray changes are per-vertex state changes, not geometry changes.
 params[7]=1;put(0,params);put(5,new Float32Array(4).fill(.001));dispatch(['reset']);wave=0;counter(C.forceHeld,0);await query('stray seed');advance();await query('stray hit');
 put(5,new Uint32Array([64]),16);put(5,new Float32Array([.8]),4*(8+7));advance();q=await query('body stray becomes active');assert.equal(q.value,1);assert.ok(q.misses>0);
 put(5,new Uint32Array([0]),16);advance();q=await query('body stray disabled');assert.equal(q.value,0);assert.ok(q.misses>0);
 // Reset must clear decision words independently of old geometry payloads.
 dispatch(['reset']);const at=m.entries+m.recordWords*bucket(0,0);assert.equal((await read(5,4*(at+7),4))[0],0);
 wave=0;counter(C.forceHeld,0);q=await query('reset decision publication');assert.equal(q.hits,0);
 advance();q=await query('reset next wave');assert.equal(q.hits,128);
 setKeys(true);q=await query('hash collisions never use other decision');assert.equal(q.hits,64);assert.equal(q.misses,64);
 setKeys();counter(C.rounds,1);q=await query('nonmandatory decision bypass');assert.equal(q.hits,0);
 assert.deepEqual(errors,[]);
 }finally{await device.queue.onSubmittedWorkDone();for(const b of buffers.values())b.destroy();device.destroy();}
});
