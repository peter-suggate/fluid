/** Actual generated DAT functions: full scan vs early predicate, including sampled
 * collapse and degeneracy. Out-of-domain nonfinite/overflow cases are reported
 * separately and never used to assert a universal NaN arithmetic contract. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import {ogcQueryWGSL,parseOgcLocality} from '../lib/cloth/contact/ogc-detector-wgsl';
import {OGC_COMMON_WGSL} from '../lib/cloth/contact/ogc-wgsl';
import {buildOgcTopology,buildOgcEdges} from '../lib/cloth/contact/ogc-topology';
import {buildOgcBVH} from '../lib/cloth/contact/bvh';
import {createProcessRetainedDawnGPU} from '../lib/harness/node-dawn-provider';
import {requiredFluidDeviceLimits} from '../lib/core/webgpu-device-limits';
function extract(source:string,name:string){const a=source.indexOf(`fn ${name}(`);assert.ok(a>=0,name);let p=source.indexOf('{',a),depth=1;for(let i=p+1;i<source.length;i++){if(source[i]==='{')depth++;if(source[i]==='}'&&!--depth)return source.slice(a,i+1);}throw new Error('unclosed '+name);}
type V=[number,number,number];
type Case={x:V[],w:V[],L:number,thr:number,kind:number,label:string,domain:boolean};
test('GPU DAT early exit preserves valid-geometry predicates and saves closest evaluations',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const tris=new Uint32Array([0,1,2,0]),rest=new Float32Array([0,0,0,0,1,0,0,0,0,1,0,0]),top=buildOgcTopology({vertexCount:3,triangles:tris,edges:buildOgcEdges(tris)},rest),bvh=buildOgcBVH(top,rest,4,1);
 const names=['datPointSeg','datSegSeg','datPointTri','datLevels','datPathVT','datPathEE'];
 const fragment=(early:boolean)=>{const source=ogcQueryWGSL('queryVertex',parseOgcLocality('off'),undefined,false,{...bvh,datPathEarlyExit:early});let s=names.map(n=>extract(source,n)).join('\n');s=s.replace(/(fn dat(?:PointTri|SegSeg)\([^]*?\{)/g,'$1 evaluations++;');for(const n of names)s=s.replace(new RegExp('\\b'+n+'\\b','g'),(early?'fast_':'full_')+n);return s;};
 let seed=0x12345678;const rnd=()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/2**32;};
 const cases:Case[]=[];const zero=():V[]=>Array.from({length:4},()=>[0,0,0] as V);
 for(let K=1;K<=32;K++)for(const kind of [0,1]){
  const thr=.002,L=(K-.25)*thr/5;
  for(let k=0;k<32;k++)cases.push({x:Array.from({length:4},()=>[(rnd()-.5)*.03,(rnd()-.5)*.03,(rnd()-.5)*.03] as V),w:Array.from({length:4},()=>[(rnd()-.5)*L,(rnd()-.5)*L,(rnd()-.5)*L] as V),L,thr,kind,label:`random K${K}`,domain:true});
  for(const epsilon of [-1e-7,-1e-9,0,1e-9,1e-7]){
   const gap=thr+L/(2*K)+epsilon;
   const x:V[]=kind===0?[[0,0,gap],[-.02,-.02,0],[.02,-.02,0],[0,.02,0]]:[[-.02,0,gap],[.02,0,gap],[0,-.02,0],[0,.02,0]];
   cases.push({x,w:zero(),L,thr,kind,label:`threshold K${K} eps${epsilon}`,domain:true});
  }
  const collapse:V[]=kind===0?[[0,0,.003],[-.02,-.02,0],[.02,-.02,0],[0,.02,0]]:[[-.02,0,.003],[.02,0,.003],[0,-.02,0],[0,.02,0]];
  const w=collapse.map((p,i)=>kind===0&&i===0?[0,0,0] as V:[-2*p[0],-2*p[1],0] as V);
  cases.push({x:collapse,w,L:.2,thr:1/K,kind,label:`sample collapse ${K}`,domain:true});
  cases.push({x:[[0,0,.001],[0,0,0],[0,0,0],[0,0,0]],w:zero(),L,thr,kind,label:`point degeneracy K${K}`,domain:true});
  cases.push({x:[[0,0,.001],[.02,0,.001],[0,0,0],[.02,0,0]],w:zero(),L,thr,kind,label:`parallel/collinear K${K}`,domain:true});
 }
 for(const kind of [0,1])for(const scale of [1e-18,1e-12,1e-6,1e3])for(const moving of [false,true]){
  const x:V[]=[[0,0,.003],[-.02,-.02,0],[.02,-.02,0],[0,.02,0]].map(p=>p.map(v=>v*scale) as V);
  const w=moving?x.map((p,i)=>i===0?[0,0,0] as V:[-2*p[0],-2*p[1],0] as V):zero();
  cases.push({x,w,L:.2*scale,thr:.002*scale,kind,label:`scaled collapse ${scale} moving${moving}`,domain:true});
 }
 for(const kind of [0,1])for(const value of [NaN,Infinity,1e20])cases.push({x:[[value,0,.001],[1,0,0],[0,1,0],[0,0,0]],w:zero(),L:.1,thr:.002,kind,label:`out-of-domain ${value}`,domain:false});
 const input=new Float32Array(cases.length*36);cases.forEach((c,i)=>{c.x.forEach((v,k)=>input.set(v,36*i+4*k));c.w.forEach((v,k)=>input.set(v,36*i+16+4*k));input.set([c.L,c.thr,c.kind,0],36*i+32);});
 const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
 const errors:string[]=[];device.addEventListener('uncapturederror',(e:any)=>{e.preventDefault();errors.push(e.error.message);});
 try{
  const source=OGC_COMMON_WGSL+`struct Case {x:array<vec4f,4>,w:array<vec4f,4>,control:vec4f}
  @group(0) @binding(0) var<storage,read> cases:array<Case>;
  @group(0) @binding(1) var<storage,read_write> out:array<u32>;
  var<private> caseIndex:u32;var<private> evaluations:u32;
  fn predOf(i:u32)->vec3f{return cases[caseIndex].w[i].xyz;}
  `+fragment(false)+fragment(true)+`
  @compute @workgroup_size(64)fn main(@builtin(global_invocation_id) id:vec3u){if(id.x>=${cases.length}u){return;}caseIndex=id.x;let c=cases[caseIndex];let L=c.control.x;let thr=c.control.y;
  evaluations=0u;var a:f32;var b:f32;
  if(c.control.z==0.0){a=full_datPathVT(c.x[0].xyz,0u,vec3u(1,2,3),c.x[1].xyz,c.x[2].xyz,c.x[3].xyz,L,thr);}else{a=full_datPathEE(vec4u(0,1,0,0),vec4u(2,3,0,0),c.x[0].xyz,c.x[1].xyz,c.x[2].xyz,c.x[3].xyz,L,thr);}let fullCalls=evaluations;
  evaluations=0u;
  if(c.control.z==0.0){b=fast_datPathVT(c.x[0].xyz,0u,vec3u(1,2,3),c.x[1].xyz,c.x[2].xyz,c.x[3].xyz,L,thr);}else{b=fast_datPathEE(vec4u(0,1,0,0),vec4u(2,3,0,0),c.x[0].xyz,c.x[1].xyz,c.x[2].xyz,c.x[3].xyz,L,thr);}
  let at=8u*id.x;out[at]=select(0u,1u,a>thr);out[at+1u]=select(0u,1u,b>thr);out[at+2u]=fullCalls;out[at+3u]=evaluations;out[at+4u]=full_datLevels(L,thr);out[at+5u]=bitcast<u32>(a);out[at+6u]=bitcast<u32>(b);out[at+7u]=u32(c.control.z);
  }`;
  const module=device.createShaderModule({code:source}),info=await module.getCompilationInfo();assert.deepEqual(info.messages.filter(m=>m.type==='error').map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}}),ib=device.createBuffer({size:input.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),ob=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:cases.length*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});device.queue.writeBuffer(ib,0,input);
  const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ib}},{binding:1,resource:{buffer:ob}}]}),enc=device.createCommandEncoder(),p=enc.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(cases.length/64));p.end();enc.copyBufferToBuffer(ob,0,read,0,cases.length*32);device.queue.submit([enc.finish()]);await read.mapAsync(GPUMapMode.READ);const out=new Uint32Array(read.getMappedRange().slice(0));
  const levels=new Set<number>();let saved=0,total=0,separated=0,retained=0;const diagnostics:any[]=[];
  cases.forEach((c,i)=>{const a=out.subarray(8*i,8*i+8);if(!c.domain){diagnostics.push({label:c.label,full:a[0],early:a[1],fullBoundBits:a[5],earlyBoundBits:a[6]});return;}assert.equal(a[0],a[1],c.label);assert.equal(a[2],a[4]!+1,c.label+'full evaluations');assert.ok(a[3]!<=a[2]!);if(a[0]){separated++;assert.equal(a[5],a[6],c.label+' unchanged successful bound');}else retained++;levels.add(a[4]!);saved+=a[2]!-a[3]!;total+=a[2]!;});
  assert.equal(levels.size,32);assert.ok(separated>0&&retained>0&&saved>0);assert.deepEqual(errors,[]);console.log({validCases:cases.filter(c=>c.domain).length,separated,retained,totalClosestEvaluations:total,savedClosestEvaluations:saved,diagnostics});ib.destroy();ob.destroy();read.destroy();
 }finally{device.destroy();}
});
