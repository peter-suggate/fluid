import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSharpening} from "../lib/methods/uniform/uniform-mixed-sharpening";
import {sharpeningReference} from "./helpers/uniform-sharpening-reference";
import {UniformMixedSolid} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("packed sharpening seams preserve every sweep across changing h/4h ownership",{timeout:240_000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","mixed sharpening seam parity");
 let device:GPUDevice|undefined,ownership:UniformMixedOwnership|undefined,solid:UniformMixedSolid|undefined;
 const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  const create=raw.createShaderModule.bind(raw);let reference=false,replaced=0;
  Object.defineProperty(raw,"createShaderModule",{configurable:true,writable:true,value:(descriptor:GPUShaderModuleDescriptor)=>{
   let code=descriptor.code;if(reference&&code.includes("fn shSweepJobs(")){const next=sharpeningReference(code);if(next!==code)replaced++;code=next;}
   return create({...descriptor,code});
  }});
  device=managedGPUDevice(raw,{requireWorkerRealm:false});const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const dims=[32,24,16] as const,vertices=dims.map(n=>n+1),h=[.04,.05,.06] as const,n=dims.reduce((a,b)=>a*b,1);
  const lattice={dimensions:dims,cellSize_m:h,origin_m:{x:0,y:0,z:0}};
  const fine=createUniformMixedLayout(lattice,[]),coarse=createUniformMixedLayout(lattice,[],4);
  // A single coarse island exercises an odd tail group. A larger island
  // includes both seam and regular coarse owners beside fine owners.
  const island=(size:number)=>createUniformMixedLayout(lattice,[{id:"island",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,
   min_m:{x:.32,y:.2,z:.24},max_m:{x:.32+size*4*h[0],y:.2+size*4*h[1],z:.24+size*4*h[2]}}]);
  const odd=island(1),mixed=island(3);
  const scattered=createUniformMixedLayout(lattice,[],4,Uint8Array.from({length:fine.tiles.length},(_,i)=>((i*37)^(i>>1))%11<3?1:0));
  ownership=new UniformMixedOwnership(d,fine,false);
  const texture=(size:readonly number[])=>{const t=d.createTexture({size:[...size],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
  const buffer=(size:number,uniform=false)=>{const b=d.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});buffers.push(b);return b;};
  const write=(t:GPUTexture,a:Float32Array<ArrayBuffer>)=>d.queue.writeTexture({texture:t},a,{bytesPerRow:t.width*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  const read=async(b:GPUBuffer,size=b.size)=>{const staging=d.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});const e=d.createCommandEncoder();e.copyBufferToBuffer(b,0,staging,0,size);d.queue.submit([e.finish()]);await staging.mapAsync(GPUMapMode.READ);const words=new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();return words;};
  const phi=texture(vertices),target=texture(dims),center=texture(dims),params=buffer(32,true);
  const solidParams=buffer(272,true),words=Math.ceil((dims[0]+2)*(dims[1]+2)*(dims[2]+2)/32),cutMapOffsetWords=Math.ceil((4+words)/64)*64;
  const solidScratch=buffer(4*(cutMapOffsetWords+ownership.capacity.tiles));
  const solidData=new Uint32Array(solidScratch.size/4);solidData.fill(1,cutMapOffsetWords);
  // An embedded block in the fine side of the mixed layout; coarse cubic
  // stencils can still reach it. Also exercise the solid-present fallback.
  for(let z=1;z<8;z++)for(let y=1;y<8;y++)for(let x=1;x<8;x++){
   const i=x+1+(dims[0]+2)*(y+1+(dims[1]+2)*(z+1));solidData[4+(i>>>5)]!|=1<<(i&31);
  }
  d.queue.writeBuffer(solidScratch,0,solidData);d.queue.writeBuffer(solidParams,0,new Float32Array([...dims,.037,...h,0,dims[0]*h[0],dims[1]*h[1],dims[2]*h[2],0]));
  const terrain=d.createTexture({size:[dims[0],dims[2]],format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING});textures.push(terrain);
  solid=new UniformMixedSolid(d,{params:solidParams,scratch:solidScratch,terrain,bodies:buffer(12*128,true),coupledTiles:0,cutMapOffsetWords});

  const variants=[];
  for(const old of [true,false]){
   reference=old;const work=buffer(UniformMixedSharpening.workBytes(ownership.capacity.tiles));
   const stage=new UniformMixedSharpening(d,ownership,solid,{list:work});await stage.initialize();
   const a=texture(dims),b=texture(dims),scratch=buffer(40*n),reductions=buffer(64);
   const groups=[stage.bind(a,b,phi,target,center,{buffer:scratch},params,reductions),stage.bind(b,a,phi,target,center,{buffer:scratch},params,reductions)] as const;
   variants.push({stage,a,b,scratch,reductions,groups,work});
  }
  reference=false;assert.equal(replaced,1);let changed=0,oddCovered=false;
  for(const present of [false,true]){solid.present=present;
  for(const [name,layout] of [["coarse",coarse],["odd",odd],["mixed",mixed],["scattered",scattered],["fine",fine],["mixed-again",mixed]] as const){
   ownership.update(layout);
   for(const policy of [[0,0],[1,1],[0,2]]){
    d.queue.writeBuffer(params,0,new Float32Array([.8,2.1,.001,0,...policy,0,0]));
    const values=new Float32Array(n),desired=new Float32Array(n),distances=new Float32Array(n);
    const field=(x:number,y:number,z:number)=>name==="scattered"?Math.hypot((x-16)*h[0],(y-12)*h[1],(z-8)*h[2])-.4:(x-16)*h[0]*.31+(y-10)*h[1]*.83+(z-8)*h[2]*.46;
    for(let z=0;z<dims[2];z++)for(let y=0;y<dims[1];y++)for(let x=0;x<dims[0];x++){
     const i=x+dims[0]*(y+dims[1]*z),distance=field(x+.5,y+.5,z+.5);
     distances[i]=distance;desired[i]=Math.max(0,Math.min(1,.5-distance/.12));
     values[i]=i%17===0?-.0003:i%13===0?.0004:i%7===0?desired[i]!:.15+.7*(.5+.5*Math.sin(x*.61+y*.79+z*.37));
    }
    const vertex=new Float32Array(vertices.reduce((a,b)=>a*b,1));
    for(let z=0;z<vertices[2]!;z++)for(let y=0;y<vertices[1]!;y++)for(let x=0;x<vertices[0]!;x++)vertex[x+vertices[0]!*(y+vertices[1]!*z)]=field(x,y,z);
    write(phi,vertex);write(target,desired);write(center,distances);
    for(const v of variants){write(v.a,values);write(v.b,new Float32Array(n).fill(123));d.queue.writeBuffer(v.scratch,0,new Float32Array(10*n).fill(71));const e=d.createCommandEncoder();e.clearBuffer(v.reductions);v.stage.encodeGeometry(e,v.groups[0]);d.queue.submit([e.finish()]);}
    const headers:Uint32Array[]=await Promise.all(variants.map(v=>read(v.work,32)));
    assert.deepEqual(headers[1],headers[0]);if(name==="odd"&&headers[0]![5]===1)oddCovered=true;
    // Observe all four pairs of sweeps, including each scratch output, so
    // stale writes and errors that cancel by sweep eight cannot hide.
    for(let pair=0;pair<4;pair++){
     const results:{a:Float32Array;b:Float32Array;dust:Uint32Array}[]=[];
     for(const v of variants){const e=d.createCommandEncoder();v.stage.encodeSweeps(e,v.groups,2);d.queue.submit([e.finish()]);results.push({a:await readMixedTexture(d,v.a),b:await readMixedTexture(d,v.b),dust:await read(v.reductions)});}
     assert.deepEqual(results[1],results[0],`${present?"solid":"clear"} ${name} policy ${policy} sweep ${2*pair+2}`);
     assert.ok(results[1]!.a.every(Number.isFinite)&&results[1]!.b.every(Number.isFinite),"both sweep outputs must remain finite");
     if(results[1]!.a.some((v,i)=>Math.abs(v-values[i]!)>.01))changed++;
    }
   }
  }}
  assert.ok(oddCovered,"must exercise a single active seam tile and an unused packed lane block");assert.ok(changed>0,"sharpening must move more volume than dust cleanup alone");assert.deepEqual(errors,[]);
 }finally{ownership?.destroy();solid?.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());device?.destroy();await releaseWebGPUExclusiveLock();}
});
