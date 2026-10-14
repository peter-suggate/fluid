/** Exact cached solid predicate versus geometric evaluation, and the voxel
 * read of the paged mask image versus its host twin, including a one-cell
 * halo of wall pages, terrain at the cutoff, voxels, every rigid shape,
 * incremental body motion and body removal. Small synthetic correctness
 * cases; this does not modify any benchmark scene or simulation setting.
 * node --import tsx tools/probe-solid-open-mask.ts
 */
import assert from "node:assert/strict";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneLatticeDimensions,solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { SCENE_SHAPES_BY_CODE } from "../lib/core/scene-shape";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { sceneWithSolidStroke,solidWorldForScene,type SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { UNIFORM_PAGE_TABLE_WORDS,uniformLatticeAddress,uniformLatticeAddressWGSL } from "../lib/methods/uniform/uniform-lattice-address";
import { createUniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedSolid,uniformMixedSolidWGSL,uniformSolidMaskPages } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";

const dims=[64,32,32] as const,extent=dims.map(n=>n+2),cells=extent.reduce((n,d)=>n*d,1),OPEN_TOP=false;
const errors:string[]=[],results:{name:string;cells:number;open:number;partial:number;voxels:number;mismatches:number}[]=[];
const dawn=await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
const device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
const params=device.createBuffer({size:272,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
const bodies=device.createBuffer({size:12*128,usage:GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
const terrain=device.createTexture({size:[dims[0],dims[2]],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
const output=device.createBuffer({size:8*cells,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
const read=device.createBuffer({size:output.size,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
const words=new Float32Array(68);words.set([...dims,1/60]);words.set([1,1,1,9.8],4);words.set([...dims,0],8);
const layout=createUniformMixedLayout({dimensions:dims,cellSize_m:[1,1,1],origin_m:{x:-dims[0]/2,y:0,z:-dims[2]/2}},[],4,undefined,uniformLatticeAddress(dims,OPEN_TOP));
// The mask image of the layout's address, a slot for every page of the box: the solver's solid scratch.
const address=layout.address,keys=address.tileKeys,image=new SolidOccupancyMask(uniformSolidMaskPages(address));
const scratch=device.createBuffer({size:image.words.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
const solid=new UniformMixedSolid(device,{params,scratch,terrain,bodies,coupledTiles:0,paramsWritten:()=>true},layout);
// A topology of that address: its tile words, then the page table and block words.
const topology=device.createBuffer({size:20*keys+address.seed.tail.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
device.queue.writeBuffer(topology,0,layout.tiles);device.queue.writeBuffer(topology,20*keys,address.seed.tail);
try{
 await solid.initialize();
 const module=device.createShaderModule({label:"Cached solid predicate parity",code:`const UM_D:vec3u=vec3u(${dims.join("u,")}u);
@group(1) @binding(1) var<storage,read> topology:array<u32>;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(63u);return topology[${5*keys}u+(e.x|(e.y<<6u)|(e.z<<12u))];}
fn umBlockWord(b:u32)->u32{return topology[${5*keys+UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({cells:"UM_D",tiles:"(UM_D/4u)",constant:true,paged:{blocks:`${address.blocks}u`}})}
${uniformMixedSolidWGSL(0,solid.coarse!.count)}
@group(1) @binding(0) var<storage,read_write> result:array<vec2u>;
@compute @workgroup_size(64) fn check(@builtin(global_invocation_id) id:vec3u){
 if(id.x>=${cells}u){return;}
 let q=vec3i(i32(id.x%${extent[0]}u)-1,i32((id.x/${extent[0]}u)%${extent[1]}u)-1,i32(id.x/${extent[0]! * extent[1]!}u)-1);
 let fraction=umCellOpen(q);result[id.x]=vec2u(u32(fraction>1e-5)|(u32(umCellIsOpen(q))<<1u)|(u32(umSolidVoxel(q))<<2u),bitcast<u32>(fraction));
}`});
 const messages=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(messages,[]);
 const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"check"}});
 const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries:[{binding:0,resource:{buffer:output}},{binding:1,resource:{buffer:topology}}]});
 // Auto-layout includes only the resources reached by these two helpers.
 const resourceMap:Record<number,GPUBindingResource>={0:{buffer:params,size:272},1:{buffer:scratch},2:terrain.createView(),3:{buffer:solid.coarse!.record},4:{buffer:(solid as unknown as {bodies:GPUBuffer}).bodies}};
 const solids=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:Object.entries(resourceMap).map(([binding,resource])=>({binding:+binding,resource}))});
 const bodyWords=new Float32Array(12*32),heights=new Float32Array(dims[0]*dims[2]);
 const run=async(name:string,bodyOnly=false)=>{
  device.queue.writeBuffer(params,0,words);device.queue.writeBuffer(bodies,0,bodyWords);
  if(!bodyOnly){device.queue.writeBuffer(scratch,0,image.words);device.queue.writeTexture({texture:terrain},heights,{bytesPerRow:4*dims[0]},[dims[0],dims[2]]);}
  solid.invalidate(bodyOnly);const e=device.createCommandEncoder();solid.encodeBodies(e);solid.encodeCoarse(e,{buffer:topology});
  const p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,solids);p.setBindGroup(1,group);p.dispatchWorkgroups(Math.ceil(cells/64));p.end();e.copyBufferToBuffer(output,0,read,0,output.size);device.queue.submit([e.finish()]);
  await read.mapAsync(GPUMapMode.READ);const bits=new Uint32Array(read.getMappedRange()),fractions=new Float32Array(bits.buffer);let mismatches=0,open=0,partial=0,voxels=0;
  for(let i=0;i<cells;i++){
   const word=bits[2*i]!,held=image.solid(i%extent[0]!-1,Math.floor(i/extent[0]!)%extent[1]!-1,Math.floor(i/(extent[0]!*extent[1]!))-1);
   if((word&1)!==((word>>>1)&1)||((word>>>2)&1)!==+held)mismatches++;open+=word&1;voxels+=+held;if(fractions[2*i+1]!>0&&fractions[2*i+1]!<1)partial++;
  }
  read.unmap();const result={name,cells,open,partial,voxels,mismatches};results.push(result);console.log(JSON.stringify(result));assert.equal(mismatches,0,name);assert.deepEqual(errors,[]);return result;
 };
 words[52]=1;await run("empty, solid features disabled");words[52]=0;
 // Scattered voxels in the first page, a block in the second: both reach the image as the solver's do, through a SolidWorld.
 const strokes:SolidWorldVoxelPatch[]=[{operation:"fill",minimum:[40,0,8],maximumExclusive:[64,6,20]}];
 for(let z=0;z<24;z++)for(let y=0;y<24;y++)for(let x=0;x<24;x++)if((3*x+5*y+7*z)%19===0)strokes.push({operation:"fill",minimum:[x,y,z],maximumExclusive:[x+1,y+1,z+1]});
 // The scene of the probe's lattice, in the empty scene's own 0.05 m cells, with the shell of its top: a shell
 // plane over an open top would lie in the pages above the box, which are the image's to hold.
 const scene=createEmptyScene({extents_m:{x:dims[0]/20,y:dims[1]/20,z:dims[2]/20}});scene.container.top=OPEN_TOP?"open":"closed";scene.solidVoxels=[...solidVoxelShellForScene(scene)];
 assert.deepEqual([...sceneLatticeDimensions(scene)],[...dims],"the probe's scene has the probe's lattice");
 image.update(solidWorldForScene(sceneWithSolidStroke(scene,strokes)));
 assert.ok(image.slotsHeld===2&&image.solid(0,0,0)&&image.solid(63,5,19)&&!image.solid(63,6,19),"the probe's voxels are in the image");
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
