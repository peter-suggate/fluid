// CPU-only Uniform Geometric shader preflight. Constructs the solver against a
// recording mock GPUDevice (no Dawn, no GPU), dumps every WGSL module, prints
// compiled pipelines per module, encodes frame 1 to record which pipelines are
// dispatched, and parses every module with naga. Run it before any Dawn run
// after a WGSL edit: a parse error costs nothing here.
// Usage: node --import tsx tools/capture-uniform-wgsl.mts [scene] [outDir] [default|fine]
//   default = dynamic coarsening; fine = all-h simulation (regions). Pressure is always band.
//   env FRAMES=n (frames to encode; the mock's zero readbacks stop after frame 1's encode,
//   reported as "did not converge ... 0 cycles" — expected).
// Naga quirks that Dawn accepts: "already in scope", "read-write storage textures prior to MSL 1.2".
import {mkdirSync,writeFileSync,readdirSync,readFileSync} from "node:fs";
import {spawnSync} from "node:child_process";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformVolumeMethod} from "../lib/methods/uniform/uniform-volume-method";

const [sceneId="sparse-cm12-long-dam-break",out="wgsl",arm="default"]=process.argv.slice(2);
mkdirSync(out,{recursive:true});
const flags=(names:string[])=>Object.fromEntries(names.map((n,i)=>[n,1<<i]));
Object.assign(globalThis,{
 GPUBufferUsage:flags(["MAP_READ","MAP_WRITE","COPY_SRC","COPY_DST","INDEX","VERTEX","UNIFORM","STORAGE","INDIRECT","QUERY_RESOLVE"]),
 GPUTextureUsage:flags(["COPY_SRC","COPY_DST","TEXTURE_BINDING","STORAGE_BINDING","RENDER_ATTACHMENT"]),
 GPUShaderStage:flags(["VERTEX","FRAGMENT","COMPUTE"]),GPUMapMode:flags(["READ","WRITE"]),GPUColorWrite:{ALL:15},
});
let modules=0;const used=new Set<number>();
const pass=():any=>obj({setPipeline:(p:any)=>{if(p&&p.__pipe!==undefined)used.add(p.__pipe);},end(){}});
const encoder=():any=>obj({beginComputePass:()=>pass(),beginRenderPass:()=>pass(),finish:()=>obj()});const pipes:any[]=[];
const obj=(extra:Record<string,unknown>={}):any=>new Proxy(extra,{get(t,p){
 if(p in t)return (t as any)[p];if(p==="then")return undefined;
 return (..._a:unknown[])=>obj();
}});
const buffer=(d:GPUBufferDescriptor)=>{const bytes=new ArrayBuffer(Math.max(d.size,4));return obj({size:d.size,usage:d.usage,label:d.label,
 getMappedRange:(o=0,s?:number)=>bytes.slice(o,s===undefined?undefined:o+s),mapAsync:async()=>{},unmap(){},destroy(){},mapState:"unmapped"});};
const texture=(d:GPUTextureDescriptor)=>{const s=d.size as number[]|{width:number;height?:number;depthOrArrayLayers?:number};
 const [w,h=1,z=1]=Array.isArray(s)?s:[s.width,s.height,s.depthOrArrayLayers];
 return obj({width:w,height:h,depthOrArrayLayers:z,format:d.format,dimension:d.dimension??"2d",usage:d.usage,mipLevelCount:d.mipLevelCount??1,sampleCount:1,label:d.label,createView:()=>obj(),destroy(){}});};
const limits=new Proxy({} as Record<string,number>,{get:(_t,p)=>typeof p==="string"&&/Count|Size|Dimension|Bindings|Groups|Attributes|Buffers|Components|Invocations|Layers|PerStage|PerPipeline|PerBindGroup/.test(p)?2**30:2**30});
const device:any=obj({
 limits,features:new Set(["timestamp-query","float32-filterable","shader-f16"]),lost:new Promise(()=>{}),
 createBuffer:buffer,createTexture:texture,
 createShaderModule:(d:GPUShaderModuleDescriptor)=>{const i=modules++;writeFileSync(`${out}/${String(i).padStart(3,"0")}.wgsl`,`// label: ${d.label??""}\n${d.code}`);
  const m=obj({getCompilationInfo:async()=>({messages:[]}),__id:i,__label:d.label??"",__bytes:d.code.length});return m;},
 createComputePipelineAsync:async(d:any)=>{pipes.push([d.compute.module.__id,d.compute.module.__label,d.compute.module.__bytes,d.compute.entryPoint]);return obj({__pipe:pipes.length-1});},
 createComputePipeline:(d:any)=>{pipes.push([d.compute.module.__id,d.compute.module.__label,d.compute.module.__bytes,d.compute.entryPoint]);return obj({__pipe:pipes.length-1});},
 createCommandEncoder:()=>encoder(),createRenderPipelineAsync:async()=>obj(),
 popErrorScope:async()=>null,queue:obj({onSubmittedWorkDone:async()=>{}}),
});
const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));scene.fluid.refinementRegions=[];
if(arm!=="default"&&arm!=="fine")throw new Error(`Unknown arm ${arm}`);
const dynamic=arm==="default";
const t0=performance.now();
const solver:any=await uniformVolumeMethod.createSolverAsync!(device,scene,"balanced",{coarsening:dynamic?"dynamic":"regions",surfaceDeficitBalancing:"off"},undefined,()=>{});
console.log(JSON.stringify({modules,pipelines:pipes.length,ms:performance.now()-t0}));
const by=new Map<string,{n:number,bytes:number,label:string}>();for(const [id,label,bytes] of pipes){const k=String(id);const e=by.get(k)??{n:0,bytes,label};e.n++;by.set(k,e);}
const rows=[...by].map(([id,e])=>({id,...e,cost:e.n*e.bytes})).sort((a,b)=>b.cost-a.cost);let tot=0;for(const r of rows)tot+=r.cost;
for(const r of rows.slice(0,20))console.log(r.id.padStart(3),String(r.n).padStart(4),String(r.bytes).padStart(7),(100*r.cost/tot).toFixed(1).padStart(5)+"%",r.label);
writeFileSync(out+"/pipelines.json",JSON.stringify(pipes));
const init=new Set(used);let frames=0;
try{for(let step=1;step<=Number(process.env.FRAMES??4);step++){solver.advanceTo(step/30,[]);await solver.awaitFrameCompletion?.();frames++;}}catch(e){console.log("advance stopped:",String(e).slice(0,300));}
console.log(JSON.stringify({frames,usedInit:init.size,used:used.size}));
writeFileSync(out+"/used.json",JSON.stringify({init:[...init],all:[...used]}));
let parseErrors=0;
for(const file of readdirSync(out).filter(f=>f.endsWith(".wgsl")).sort()){
 const run=spawnSync("naga",[`${out}/${file}`],{encoding:"utf8"});
 if(run.error){console.log("naga unavailable:",run.error.message);break;}
 const text=(run.stdout??"")+(run.stderr??"");
 if(/error/i.test(text)&&!/already in scope|prior to MSL 1\.2/.test(text)){parseErrors++;
  console.log(`NAGA ${file} ${readFileSync(`${out}/${file}`,"utf8").split("\n",1)[0]}\n${text.split("\n").slice(0,6).join("\n")}`);}
}
console.log(parseErrors?`${parseErrors} module(s) fail naga`:"naga: all modules parse");
if(parseErrors)process.exit(1);
process.exit(0);
