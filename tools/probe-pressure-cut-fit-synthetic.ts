/** Small numerical/geometry stress test for both cooperative cut fits. Uses
 * the original production cut functions, all 64 single-open-cell masks,
 * complementary masks, planar/curved data and near-interface values. */
import assert from "node:assert/strict";
import { mkdirSync,readFileSync,writeFileSync } from "node:fs";
import { dirname,resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { uniformMixedPressureCutWGSL } from "../lib/methods/uniform/uniform-mixed-pressure-cut.wgsl";
import { groupedCutFitWGSL,hybridCutFitWGSL,maskCutFitWGSL,parallelCutFitWGSL,referenceCutFitWGSL,streamedCutFitWGSL } from "./parallel-cut-fit";

const maskCount=256,cases=8,tiles=maskCount*cases;
const source=referenceCutFitWGSL(readFileSync(resolve("lib/methods/uniform/uniform-mixed-pressure-authority.ts"),"utf8").replace("${uniformMixedPressureCutWGSL}",uniformMixedPressureCutWGSL));
const begin=source.indexOf("@compute @workgroup_size(64) fn cut("),end=source.indexOf("fn umCutPhi(",begin);assert.ok(begin>=0&&end>begin);
let cut=referenceCutFitWGSL(source.slice(begin,end)).replace("textureLoad(vertexPhi,origin+vec3i(vec3u(v%5u,(v/5u)%5u,v/25u)),0).x","fixturePhi(t,v)");
assert.ok(!cut.includes("textureLoad(vertexPhi"));
cut=cut.replaceAll("umTopology[t]&0x3fffffffu","t");
// The transform's original output expression names topology, so leave an
// identity array in the fixture for the transformed paths too.
const code=/* wgsl */`
const UM_TILES=${tiles}u;const UM_CUT_BASE=0u;const UM_HMIN=1.0;
@group(0) @binding(0) var<storage,read> masks:array<vec2u>;
@group(0) @binding(1) var<storage,read> umTopology:array<u32>;
@group(0) @binding(2) var<storage,read_write> balance:array<vec2f>;
fn umCorner(i:u32,n:u32)->vec3u{return vec3u(i%n,(i/n)%n,i/(n*n));}
fn umTileCoord(t:u32)->vec3u{return vec3u(t,0,0);}
fn umSolidCut(t:u32)->bool{return true;}
fn umSolidOpenCells(t:u32)->vec2u{return masks[t%${maskCount}u];}
fn fixturePhi(t:u32,v:u32)->f32{
 let q=vec3f(vec3u(v%5u,(v/5u)%5u,v/25u))-2.0;let mode=t/${maskCount}u;
 var level=f32(i32(t%7u)-3)/16.0;
 if(mode==1u){level=0.0;}if(mode==2u){level=-0.004;}if(mode==3u){level=1e-7;}
 let plane=level+0.125*q.x-0.25*q.y+0.5*q.z;
 if(mode<4u){return plane;}
 if(mode==4u){return plane+0.0625*(q.x*q.y+q.z*q.z);}
 if(mode==5u){return f32(i32((v*173u+t*59u)%1024u)-512)/128.0;}
 if(mode==6u){return 1e-4*plane;}
 return 16.0+plane;
}
${cut}
`;
const maskWords=new Uint32Array(2*maskCount);
for(let mask=0;mask<maskCount;mask++){
 if(mask<64)maskWords[2*mask+(mask>>>5)]=1<<(mask&31);
 else if(mask<128){maskWords[2*mask]=0xffffffff;maskWords[2*mask+1]=0xffffffff;maskWords[2*mask+((mask-64)>>>5)]!^=1<<((mask-64)&31);}
 else if(mask===128){/* no open cell, default continuation */}
 else if(mask===129){maskWords[2*mask]=0xffffffff;maskWords[2*mask+1]=0xffffffff;}
 else for(let c=0;c<64;c++)if(((c*73+mask*19)%(mask%13+2))===0)maskWords[2*mask+(c>>>5)]!|=1<<(c&31);
}
async function run(){
 const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
 const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);const device=await adapter.requestDevice(),errors:string[]=[];
 device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
 const inputs=device.createBuffer({size:maskWords.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST}),topology=device.createBuffer({size:4*tiles,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
 const output=device.createBuffer({size:8*tiles,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:8*tiles,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
 try{
  device.queue.writeBuffer(inputs,0,maskWords);device.queue.writeBuffer(topology,0,Uint32Array.from({length:tiles},(_,i)=>i));
  let baseline:Uint32Array|undefined;const results=[];
  for(const variant of ["reference","parallel","grouped","empty","masks","hybrid","streamed"]){
   const source=variant==="parallel"?parallelCutFitWGSL(code):variant==="grouped"?groupedCutFitWGSL(code):variant==="hybrid"?hybridCutFitWGSL(code):variant==="streamed"?streamedCutFitWGSL(code):variant==="empty"||variant==="masks"?maskCutFitWGSL(code,variant==="masks"):code;
   const module=device.createShaderModule({label:`Synthetic ${variant} cut fit`,code:source});
   const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[]);
   const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}]});
   const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[layout]}),compute:{module,entryPoint:"cut"}}),group=device.createBindGroup({layout,entries:[{binding:0,resource:{buffer:inputs}},{binding:1,resource:{buffer:topology}},{binding:2,resource:{buffer:output}}]});
   const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(Math.ceil(tiles/64));p.end();e.copyBufferToBuffer(output,0,read,0,read.size);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const words=new Uint32Array(read.getMappedRange().slice(0));read.unmap();
   if(!baseline){baseline=words;continue;}
   const a=new Float32Array(baseline.buffer),b=new Float32Array(words.buffer);let differences=0,maxAbsolute=0,signChanges=0,liquidChanges=0,nonfinite=0,planeError=0,baselinePlaneError=0;
   const byCase:Array<{case:number;differences:number;maxAbsolute:number;signChanges:number;liquidChanges:number}>=Array.from({length:cases},(_,mode)=>({case:mode,differences:0,maxAbsolute:0,signChanges:0,liquidChanges:0}));
   for(let t=0;t<tiles;t++){
    const mode=Math.floor(t/maskCount),item=byCase[mode]!,x=a[2*t]!,y=b[2*t]!,absolute=Math.abs(x-y),sign=Number((x<0)!==(y<0)),liquid=Number((x<Math.fround(-.004))!==(y<Math.fround(-.004)));
    if(baseline[2*t]!==words[2*t]){differences++;item.differences++;}maxAbsolute=Math.max(maxAbsolute,absolute);item.maxAbsolute=Math.max(item.maxAbsolute,absolute);signChanges+=sign;item.signChanges+=sign;liquidChanges+=liquid;item.liquidChanges+=liquid;
    if(!Number.isFinite(y))nonfinite++;
    if(mode<4&&t%maskCount!==128){const level=mode===1?0:mode===2?-.004:mode===3?1e-7:((t%7)-3)/16;planeError=Math.max(planeError,Math.abs(y-level));baselinePlaneError=Math.max(baselinePlaneError,Math.abs(x-level));}
   }
   const result={variant,tiles,differences,maxAbsolute_cells:maxAbsolute,signChanges,liquidChanges,nonfinite,maxPlaneError_cells:planeError,baselinePlaneError_cells:baselinePlaneError,byCase};results.push(result);console.log(JSON.stringify(result));assert.equal(nonfinite,0);
  }
  assert.deepEqual(errors,[]);const out="artifacts/nbflip-ui-perf/parallel-pressure-cut-fit-synthetic.json";mkdirSync(dirname(out),{recursive:true});writeFileSync(out,JSON.stringify({date:new Date().toISOString(),maskCount,cases,results,errors},null,2)+"\n");
 }finally{for(const b of [inputs,topology,output,read])b.destroy();device.destroy();}
}
if(process.argv[1]&&import.meta.url===pathToFileURL(resolve(process.argv[1])).href)await run();
