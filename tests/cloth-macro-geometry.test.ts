/** Actual cache kernels versus an independent brute-force current-face coverage oracle.
 * Synthetic rigid translation isolates containment; full sphere/hull integration is a separate gate. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
import {buildOgcTopology,buildOgcEdges} from '../lib/cloth/contact/ogc-topology';
import {buildOgcBVH,ogcBVHCommon} from '../lib/cloth/contact/bvh';
import {macroCacheLayout,macroBoundsBodyWGSL,macroRefitNode,macroCandidateQueryWGSL,macroCacheResetWGSL,macroSampleGeometryWGSL} from '../lib/cloth/contact/macro-cache';
import {OGC_COUNTER as C,OGC_COUNTER_WORDS,OGC_COMMON_WGSL} from '../lib/cloth/contact/ogc-wgsl';
const gpu=!!process.env.WEBGPU_NODE_MODULE;
test('sample geometry memo: exact math, lazy slots, invalidation and current material',{skip:!gpu},async()=>{
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
 const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
 const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 const N=6,T=2,Q=2,gridWords=4,faceIds=[[0,1,2],[3,4,5]];
 const initial=[[0,0,0],[.02,0,0],[0,.02,0],[0,0,.01],[.02,0,.01],[0,.02,.01],[.005,.005,.05],[.505,.005,.05]];
 const pack=(a:number[][])=>new Float32Array(a.flatMap(p=>[...p,0]));
 const triangles=new Uint32Array(faceIds.flatMap(t=>[...t,0]));
 const topology=buildOgcTopology({vertexCount:N,triangles,edges:buildOgcEdges(triangles)},pack(initial.slice(0,N)));
 const base=buildOgcBVH(topology,pack(initial.slice(0,N)),gridWords,1);
 const common=OGC_COMMON_WGSL+`struct PType {N:u32,E:u32,S:u32,SB:u32,SE:u32,SEB:u32,SF:u32,SFB:u32,envOff:u32,bodyBase:u32,rqTop:f32,pad:u32}
 @group(0) @binding(0) var<uniform>P:PType;
 @group(0) @binding(4) var<storage,read_write>xdet:array<vec4f>;
 @group(0) @binding(5) var<storage,read_write>work:array<atomic<u32>>;
 @group(0) @binding(9) var<storage,read_write>counters:array<atomic<u32>>;
 @group(0) @binding(10) var<storage,read_write>args:array<u32>;
 @group(0) @binding(14) var<storage,read>pred:array<vec4f>;
 struct OgcBox {lo:vec3f,hi:vec3f}
 const OGC_ROUNDS=${C.rounds}u;const OGC_FORCE_HELD=${C.forceHeld}u;const OGC_PLANAR=1u;
 fn boxesOverlap(a:OgcBox,b:OgcBox)->bool{return all(a.lo<=b.hi)&&all(b.lo<=a.hi);}
 fn predOf(i:u32)->vec3f{if(i<P.N){return pred[i].xyz;}return pred[2u*P.N+i-P.N].xyz;}
 fn triVerts(t:u32)->vec3u{return vec3u(3u*t,3u*t+1u,3u*t+2u);}
 fn ogcPose(i:u32)->vec3f{return vec3f(0.0,0.0,.1*f32(i));}
 fn ogcToBody(p:vec3f,x:vec3f)->vec3f{return x-p;}
 fn ogcToWorld(p:vec3f,x:vec3f)->vec3f{return x+p;}
 fn hullEdgeAt(k:u32)->vec2u{return vec2u(P.N,P.N+1u);}
 fn hullFacetAt(k:u32)->vec4u{return vec4u(P.N,P.N+1u,P.N,0u);}
 `;
 try{for(const wholeStep of [false,true])for(const capacity of [8,1]){
  const m=macroCacheLayout(base.words.length,N+Q,Q,base.nodes,T,capacity,wholeStep,Q),b={...base,macroCache:m};
  const buffers=new Map<number,GPUBuffer>();const make=(binding:number,size:number,uniform=false)=>{const x=device.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.set(binding,x);return x;};
  make(0,48,true);make(4,16*(N+Q));make(5,4*N);make(7,4*(gridWords+base.words.length+m.words));make(9,4*OGC_COUNTER_WORDS);make(10,16);make(14,16*(2*N+Q));make(22,4*Q*(T+3));make(23,16*Q);
  const params=new Uint32Array([N,0,Q,0,0,0,0,0,0,1,0,0]);new Float32Array(params.buffer)[10]=.1;device.queue.writeBuffer(buffers.get(0)!,0,params);
  device.queue.writeBuffer(buffers.get(7)!,4*gridWords,new Uint32Array(base.words));
  const pipelines=new Map<string,{p:GPUComputePipeline,g:GPUBindGroup}>();
  const compile=async(name:string,source:string,bindings:number[])=>{const module=device.createShaderModule({code:source,label:name});const info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error').map(m=>`${name} ${m.lineNum}:${m.linePos} ${m.message}`),[]);const p=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});const g=device.createBindGroup({layout:p.getBindGroupLayout(0),entries:bindings.map(binding=>({binding,resource:{buffer:buffers.get(binding)!}}))});pipelines.set(name,{p,g});};
  await compile('bounds',common+macroBoundsBodyWGSL(b),[0,4,5,7,9,10,14]);
  await compile('refit',common+ogcBVHCommon(b)+macroRefitNode(b)+`@compute @workgroup_size(1) fn main(){for(var n=${base.nodes}u;n>0u;n--){macroRefitNode(n-1u);}}`,[0,7,9]);
  await compile('reset',macroCacheResetWGSL(b),[7,9]);
  await compile('query',common+macroCandidateQueryWGSL(b)+macroSampleGeometryWGSL(b)+`
 @group(0) @binding(22) var<storage,read_write>out:array<u32>;
 @group(0) @binding(23) var<storage,read_write>stats:array<u32>;
 @compute @workgroup_size(1)fn main(@builtin(global_invocation_id) gid:vec3u){let i=gid.x;if(i>=P.S){return;}
 let x=xdet[P.N+i].xyz;let y=x+predOf(P.N+i);var s=candidateFaces(OgcBox(min(x,y),max(x,y)),P.N+P.E+i);var n=0u;var mismatch=0u;var traction=0.0;
 loop{let t=candidateFaceNext(&s);if(t==OGC_NONE){break;}out[i*${T+3}u+3u+n]=t;n++;
 let ids=triVerts(t);let a=xdet[ids.x].xyz;let b=xdet[ids.y].xyz;let c=xdet[ids.z].xyz;
 let memo=sampleClosest(&s,x,a,b,c);let direct=ogcClosestTriangle(x,a,b,c);
 let cp=direct.bary.x*a+direct.bary.y*b+direct.bary.z*c;let d=length(x-cp);
 let memoCp=memo.bary.x*a+memo.bary.y*b+memo.bary.z*c;let memoD=length(x-memoCp);
 if(any(bitcast<vec3u>(memo.bary)!=bitcast<vec3u>(direct.bary))||any(bitcast<vec3u>(memoCp)!=bitcast<vec3u>(cp))||bitcast<u32>(memoD)!=bitcast<u32>(d)||memo.feature!=direct.feature){mismatch++;}
 traction+=-ogcActivation(memoD,.1,bitcast<f32>(P.pad)).y*xdet[P.N+i].w;
 }
 stats[4u*i]=atomicLoad(&wgGeometryHits);stats[4u*i+1u]=atomicLoad(&wgGeometryMisses);stats[4u*i+2u]=mismatch;stats[4u*i+3u]=bitcast<u32>(traction);
 out[i*${T+3}u]=n;out[i*${T+3}u+1u]=select(0u,1u,s.cached);out[i*${T+3}u+2u]=s.nodes;}`, [0,4,7,9,14,22,23]);
  const dispatch=(names:string[])=>{const enc=device.createCommandEncoder(),p=enc.beginComputePass();for(const name of names){const k=pipelines.get(name)!;p.setPipeline(k.p);p.setBindGroup(0,k.g);p.dispatchWorkgroups(name==='query'?Q:1);}p.end();device.queue.submit([enc.finish()]);};
  const read=async(binding:number,offset:number,size:number)=>{const dst=device.createBuffer({size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});const e=device.createCommandEncoder();e.copyBufferToBuffer(buffers.get(binding)!,offset,dst,0,size);device.queue.submit([e.finish()]);await dst.mapAsync(GPUMapMode.READ);const a=new Uint32Array(dst.getMappedRange().slice(0));dst.destroy();return a;};
  let x=initial.map(p=>p.slice()),w=initial.map(()=>[0,0,0]),env=.001,friction=.3,clothW=0,stiffness=1;
  const upload=()=>{
   new Float32Array(params.buffer)[11]=stiffness;device.queue.writeBuffer(buffers.get(0)!,0,params);
   const x4=pack(x);for(let i=0;i<N+Q;i++)x4[4*i+3]=i<N?clothW:friction;device.queue.writeBuffer(buffers.get(4)!,0,x4);const prediction=new Float32Array(4*(2*N+Q));prediction.set(pack(w.slice(0,N)));prediction.set(pack(w.slice(N)),8*N);for(let i=0;i<N;i++)prediction.set([...x[i]!,0],4*(N+i));device.queue.writeBuffer(buffers.get(14)!,0,prediction);device.queue.writeBuffer(buffers.get(5)!,0,new Float32Array(N).fill(env));
   // Independent current BVH refit from f32-uploaded geometry; oracle below scans faces directly.
   const X=new Float32Array(pack(x)),W=new Float32Array(pack(w));const ef=Math.fround(env);
   const boxes=faceIds.map(ids=>{const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];for(const v of ids)for(let c=0;c<3;c++){const a=X[4*v+c]!,z=Math.fround(a+W[4*v+c]!);lo[c]=Math.min(lo[c]!,a,z);hi[c]=Math.max(hi[c]!,a,z);}return{lo:lo.map(v=>Math.fround(v-ef)),hi:hi.map(v=>Math.fround(v+ef))};});
   const nodes:Array<{lo:number[],hi:number[]}>=[];for(let n=base.nodes-1;n>=0;n--){const at=12*n,count=base.words[at+8]!;let list;if(count)list=Array.from(base.words.slice(base.words[at+7]!,base.words[at+7]!+count)).map(t=>boxes[t]!);else list=[nodes[n+1]!,nodes[base.words[at+9]!]!];nodes[n]={lo:[0,1,2].map(c=>Math.min(...list.map(v=>v.lo[c]!))),hi:[0,1,2].map(c=>Math.max(...list.map(v=>v.hi[c]!)))};device.queue.writeBuffer(buffers.get(7)!,4*(gridWords+12*n),new Float32Array([...nodes[n]!.lo,...nodes[n]!.hi]));}
   boxes.forEach((a,t)=>device.queue.writeBuffer(buffers.get(7)!,4*(gridWords+base.faceBoxes+6*t),new Float32Array([...a.lo,...a.hi])));
   return boxes;
  };
  const query=async(label:string,cached?:boolean)=>{const boxes=upload();dispatch(['bounds','refit','query']);const a=await read(22,0,4*Q*(T+3));for(let i=0;i<Q;i++){const p=x[N+i]!.map(Math.fround),end=p.map((v,c)=>Math.fround(v+Math.fround(w[N+i]![c]!)));const lo=p.map((v,c)=>Math.min(v,end[c]!)),hi=p.map((v,c)=>Math.max(v,end[c]!));const wanted=boxes.flatMap((b,t)=>[0,1,2].every(c=>lo[c]!<=b.hi[c]!&&b.lo[c]!<=hi[c]!)?[t]:[]);const got=Array.from(a.slice(i*(T+3)+3,i*(T+3)+3+a[i*(T+3)]!));assert.deepEqual(got.sort(),wanted,label+' sample'+i);if(cached!==undefined)assert.equal(a[i*(T+3)+1],Number(cached),label+' cached');}const st=await read(23,0,16*Q);assert.equal(st[2],0,label+' exact geometry');assert.equal(st[6],0,label+' far exact geometry');return {a,st};};
  try{
   const counter=(at:number,v:number)=>device.queue.writeBuffer(buffers.get(9)!,4*at,new Uint32Array([v]));
   device.queue.writeBuffer(buffers.get(10)!,0,new Uint32Array([1,1,1,0]));dispatch(['reset']);
   await query('initial no admitted faces');counter(C.forceHeld,1);
   env=.045;let q=await query('only later macro slot admitted');assert.equal(q.a[0],1);assert.equal(q.st[0],0);assert.equal(q.st[1],1);
   env=.06;q=await query('earlier filtered slot becomes relevant');assert.equal(q.a[0],2);assert.equal(q.st[0],capacity===1?0:1);assert.equal(q.st[1],capacity===1?2:1);
   q=await query('unchanged geometry replay');assert.equal(q.st[0],capacity===1?0:2);assert.equal(q.st[1],capacity===1?2:0);
   const oldTraction=new Float32Array(q.st.buffer)[3]!;assert.ok(oldTraction>0);
   friction=.6;clothW=123;q=await query('w-only and friction change');assert.equal(q.st[0],capacity===1?0:2);assert.equal((await read(9,4*C.macroGeometryInvalid,4))[0],0);
   const newTraction=new Float32Array(q.st.buffer)[3]!;assert.ok(Math.abs(newTraction/oldTraction-2)<1e-6,'friction recomputed from current w');
   stiffness=3;q=await query('material stiffness changes with cached geometry');assert.equal(q.st[0],capacity===1?0:2);assert.ok(Math.abs(new Float32Array(q.st.buffer)[3]!/newTraction-3)<1e-5,'activation uses current material stiffness');
   // XYZ moves inside original body-motion macro hull: containment remains valid,
   // but closest geometry must be recomputed and the geometry invalidation sticks.
   x[0]![2]=.005;q=await query('geometry changes within containment');assert.equal(q.st[0],0);assert.equal((await read(9,4*C.macroInvalid,4))[0],0);assert.equal((await read(9,4*C.macroGeometryInvalid,4))[0],1);
   x[0]![2]=0;q=await query('geometry returns but invalidation sticks');assert.equal(q.st[0],0);
   // Fresh step resets geometry tags, even when XYZ happens to match the old step.
   dispatch(['reset']);counter(C.forceHeld,0);q=await query('new step invalid tags');assert.equal(q.st[0],0);assert.equal(q.st[1],2);
   counter(C.forceHeld,1);q=await query('new step memo hit');assert.equal(q.st[0],capacity===1?0:2);
   counter(C.rounds,1);counter(C.forceHeld,0);q=await query('nonmandatory geometry bypass even with whole-step faces');assert.equal(q.st[0],0);assert.equal(q.st[1],2);assert.equal(q.a[1],Number(wholeStep&&capacity!==1));
   // Validity must include signed-zero XYZ bit changes, not numeric equality.
   counter(C.rounds,0);counter(C.forceHeld,0);dispatch(['reset']);await query('signed-zero seed');counter(C.forceHeld,1);x[0]![0]=-0;
   q=await query('signed-zero bit mismatch');assert.equal(q.st[0],0);assert.equal((await read(9,4*C.macroGeometryInvalid,4))[0],1);
  }finally{for(const buffer of buffers.values())buffer.destroy();}
 }}finally{await device.queue.onSubmittedWorkDone();device.destroy();}
 assert.deepEqual(errors,[]);
});
