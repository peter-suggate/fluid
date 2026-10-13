/** Exact cached solid predicate versus geometric evaluation, including a
 * one-cell domain halo, terrain at the cutoff, voxels, every rigid shape,
 * incremental body motion and body removal. Small synthetic correctness
 * cases; this does not modify any benchmark scene or simulation setting.
 * node --import tsx tools/probe-solid-open-mask.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { SCENE_SHAPES_BY_CODE } from "../lib/core/scene-shape";
import { UNIFORM_PAGE_TABLE_WORDS,uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedSolid,uniformMixedSolidWGSL } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

const dims=[16,12,8] as const,extent=dims.map(n=>n+2),cells=extent.reduce((n,d)=>n*d,1);
const tiles=dims.reduce((n,d)=>n*d/4,1),cutWord=Math.ceil((4+Math.ceil(cells/32))/64)*64;
const errors:string[]=[],results:{name:string;cells:number;open:number;partial:number;mismatches:number}[]=[];
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
const params=device.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const scratch=device.createBuffer({size:4*(cutWord+tiles),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
const bodies=device.createBuffer({size:12*128,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const terrain=device.createTexture({size:[dims[0],dims[2]],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
const output=device.createBuffer({size:8*cells,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const words=new Float32Array(68);words.set([...dims,1/60]);words.set([1,1,1,9.8],4);words.set([...dims,0],8);
const layout=createUniformMixedLayout({dimensions:dims,cellSize_m:[1,1,1],origin_m:{x:-8,y:0,z:-4}},[],4);
const solid=new UniformMixedSolid(device,{params,scratch,terrain,bodies,coupledTiles:0,cutMapOffsetWords:cutWord,paramsWritten:()=>true},layout);
// A topology of the layout's address: its tile words, then the page table and block words.
const address=layout.address,keys=address.tileKeys,topology=device.createBuffer({size:20*keys+address.seed.tail.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
device.queue.writeBuffer(topology,0,layout.tiles);device.queue.writeBuffer(topology,20*keys,address.seed.tail);
try{
 await solid.initialize();
 const module=device.createShaderModule({label:"Cached solid predicate parity",code:`const UM_D:vec3u=vec3u(${dims.join("u,")}u);
@group(1) @binding(1) var<storage,read> topology:array<u32>;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return topology[${5*keys}u+(e.x|(e.y<<6u)|(e.z<<12u))];}
fn umBlockWord(b:u32)->u32{return topology[${5*keys+UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({cells:"UM_D",tiles:"(UM_D/4u)",constant:true,paged:{blocks:`${address.blocks}u`,whole:address.whole}})}
${uniformMixedSolidWGSL(0,solid.coarse!.count)}
@group(1) @binding(0) var<storage,read_write> result:array<vec2u>;
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${cells}u){return;}
 let q=vec3i(i32(id.x%${extent[0]}u)-1,i32((id.x/${extent[0]}u)%${extent[1]}u)-1,i32(id.x/${extent[0]! * extent[1]!}u)-1);
 let fraction=umCellOpen(q);result[id.x]=vec2u(u32(fraction>1e-5)|(u32(umCellIsOpen(q))<<1u),bitcast<u32>(fraction));
}`});
 const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages,[]);
 const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"check"}});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:topology}}]});
 // Auto-layout includes only the resources reached by these two helpers.
 const resourceMap:Record<number,GPUBindingResource>={0:{buffer:params,size:272},1:{buffer:scratch},2:terrain.createView(),3:{buffer:solid.coarse!.record},4:{buffer:(solid as unknown as {bodies:GPUBuffer}).bodies}};
 const solids=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:Object.entries(resourceMap).map(([binding,resource])=>({binding:+binding,resource}))});
 const bodyWords=new Float32Array(12*32),mask=new Uint32Array(cutWord+tiles),heights=new Float32Array(dims[0]*dims[2]);
 const run=async(name:string,bodyOnly=false)=>{
  device.queue.writeBuffer(params,0,words);device.queue.writeBuffer(bodies,0,bodyWords);
  if(!bodyOnly){device.queue.writeBuffer(scratch,0,mask);device.queue.writeTexture({texture:terrain},heights,{bytesPerRow:4*dims[0]},[dims[0],dims[2]]);}
  solid.invalidate(bodyOnly);const e=device.createCommandEncoder();solid.encodeBodies(e);solid.encodeCoarse(e,{buffer:topology});
  const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,solids);p.setBindGroup(1,group);p.dispatchWorkgroups(Math.ceil(cells/64));p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);const bits=new Uint32Array(read.getMappedRange()),fractions=new Float32Array(bits.buffer);let mismatches=0,open=0,partial=0;
  for(let i=0;i<cells;i++){if((bits[2*i]!&1)!==(bits[2*i]!>>>1))mismatches++;open+=bits[2*i]!&1;if(fractions[2*i+1]!>0&&fractions[2*i+1]!<1)partial++;}
  read.unmap();const result={name,cells,open,partial,mismatches};results.push(result);console.log(JSON.stringify(result));assert.equal(mismatches,0,name);assert.deepEqual(errors,[]);return result;
 };
 words[52]=1;await run("empty, solid features disabled");words[52]=0;
 for(let z=0;z<dims[2];z++)for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++)if((3*x+5*y+7*z)%19===0){const i=x+1+extent[0]!*(y+1+extent[1]!*(z+1));mask[4+(i>>>5)]!|=1<<(i&31);}
 await run("static voxel edits");
 for(let i=0;i<heights.length;i++)heights[i]=3-[0,1e-6,1e-5,2e-5,0.125,0.5,0.875,1][i%8]!;
 words[11]=1;const terrainResult=await run("voxel and terrain threshold fractions");
 const dimensions={x:2.5,y:4.5,z:3.5};
 for(const shape of SCENE_SHAPES_BY_CODE){
  words[18]=1;bodyWords.fill(0);bodyWords.set([-3,5,-1,shape.code]);bodyWords.set([dimensions.x,dimensions.y,dimensions.z,shape.boundingRadius_m(dimensions)],4);
  bodyWords.set([Math.cos(0.2),0,Math.sin(0.2),0],8);const initial=await run(`${shape.name} initial`);assert.ok(initial.open<terrainResult.open,`${shape.name} must close cells`);
  bodyWords[0]=3;bodyWords[1]=7;bodyWords[2]=1;const moved=await run(`${shape.name} incremental motion`,true);assert.ok(moved.open<terrainResult.open,`${shape.name} must close cells after moving`);
  bodyWords.fill(0);words[18]=0;const removed=await run(`${shape.name} removed`,true);assert.equal(removed.open,terrainResult.open,`${shape.name} removal must restore every open cell`);
 }
 assert.ok(results.some(r=>r.partial>0),"fractional solid coverage exercised");
 console.log(JSON.stringify({cases:results.length,cells:results.reduce((n,r)=>n+r.cells,0),mismatches:0,errors}));
}finally{solid.destroy();for(const b of [params,scratch,bodies,output,read])b.destroy();terrain.destroy();await device.queue.onSubmittedWorkDone();device.destroy();}
