import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import {UniformMixedSolid} from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import {UniformMixedPhiResolve} from "../lib/methods/uniform/uniform-mixed-phi-resolve";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("solid-free cubic sampling preserves advection and redistance across changing h/4h layouts",{timeout:240_000},async()=>{
 let device:GPUDevice|undefined;
 const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[],surfaces:UniformMixedSurface[]=[];
 let ownership:UniformMixedOwnership|undefined,solid:UniformMixedSolid|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const adapter=await createProcessRetainedDawnGPU(dawn,["backend=metal"]).requestAdapter();assert.ok(adapter);
  const raw=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  // Compile the previous cubic solid-tap checks as a GPU reference.
  // Both variants execute the production owner dispatch and seam handling.
  const create=raw.createShaderModule.bind(raw);let reference=false,replaced=0;
  Object.defineProperty(raw,"createShaderModule",{configurable:true,writable:true,value:(descriptor:GPUShaderModuleDescriptor)=>{
   let code=descriptor.code;
   if(reference&&code.includes("fn umCubicPhi(")){
    const before=code;code=code.replace("!umSolidEnabled()||((umRegularFine||cell.width==1u)&&umSolidClear(base))","(umRegularFine||cell.width==1u)&&umSolidClear(base)");
    assert.notEqual(code,before,"reference must restore the old tap checks");replaced++;
   }
   return create({...descriptor,code});
  }});
  device=managedGPUDevice(raw,{requireWorkerRealm:false});const d=device;
  const errors:string[]=[];d.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const dims=[32,24,16] as const,vertices=dims.map(n=>n+1),h=[.04,.05,.06] as const;
  const lattice={dimensions:dims,cellSize_m:h,origin_m:{x:0,y:0,z:0}};
  const fine=createUniformMixedLayout(lattice,[]),coarse=createUniformMixedLayout(lattice,[],4);
  const mixed=createUniformMixedLayout(lattice,[{id:"interior-coarse",rule:"minimum-cell-size",minimumCellSize_cells:4,maximumCellSize_cells:4,
   min_m:{x:.32,y:.2,z:.24},max_m:{x:.96,y:1,z:.72}}]);
  assert.ok(mixed.fineTiles.length&&mixed.coarseTiles.length);
  ownership=new UniformMixedOwnership(d,fine,false);
  const texture=(size:readonly number[],format:GPUTextureFormat)=>{
   const t=d.createTexture({size:[...size],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;
  };
  const buffer=(size:number,uniform=false)=>{
   const b=d.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_DST});buffers.push(b);return b;
  };
  const phi=texture(vertices,"r32float"),outputPhi=texture(vertices,"r32float"),velocity=texture(dims,"rgba32float"),volume=texture(dims,"r32float");
  const coarseVelocity=texture(dims.map(n=>n/4+2),"rgba32float"),departures=texture(dims,"rgba32float");
  const negative=buffer(4*(dims[0]*dims[1]+dims[1]*dims[2]+dims[2]*dims[0])),params=buffer(32,true);
  const count=vertices.reduce((a,n)=>a*n,1),evidence=buffer(Math.ceil((ownership.capacity.tiles*12+32)/256)*256+16+4*count);
  const write=(t:GPUTexture,a:Float32Array<ArrayBuffer>,components=1)=>d.queue.writeTexture({texture:t},a,{bytesPerRow:t.width*components*4,rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
  for(const t of [velocity,coarseVelocity]){
   const a=new Float32Array(t.width*t.height*t.depthOrArrayLayers*4);
   for(let i=0;i<a.length;i+=4)a.set([.17,-.23,.11,0],i);write(t,a,4);
  }
  write(volume,new Float32Array(dims.reduce((a,n)=>a*n,1)).fill(1));
  d.queue.writeBuffer(params,0,new Float32Array([...h,.037]));d.queue.writeBuffer(params,16,new Uint32Array([1,1,0,4]));
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
  const resolve=new UniformMixedPhiResolve(d,ownership);await resolve.initialize();const resolveGroup=resolve.bind(phi);
  for(const old of [true,false]){
   reference=old;const surface=new UniformMixedSurface(d,ownership,undefined,solid,false,true);surfaces.push(surface);await surface.initialize();
  }
  reference=false;assert.equal(replaced,1);
  const groups=surfaces.map(s=>s.bind({phi,outputPhi,velocity,coarseVelocity,volume,negative,departures,params,evidence:{buffer:evidence}}));
  // Reuse the same pipelines across ownership changes, including an enclosed
  // coarse island with face/edge/corner seams. Curved and oblique fields also
  // exercise limiter extrema, clamped domain taps and nonzero cubic weights.
  for(const present of [false,true]){solid.present=present;
  for(const [layoutName,layout] of [["coarse",coarse],["mixed",mixed],["fine",fine],["mixed-again",mixed]] as const){
   ownership.update(layout);
   for(const shape of ["sphere","oblique"]){
    const input=new Float32Array(count);
    for(let z=0;z<vertices[2]!;z++)for(let y=0;y<vertices[1]!;y++)for(let x=0;x<vertices[0]!;x++){
     input[x+vertices[0]!*(y+vertices[1]!*z)]=shape==="sphere"?Math.hypot((x-16)*h[0],(y-12)*h[1],(z-8)*h[2])-.43
      :(x-15)*h[0]*.31+(y-10)*h[1]*.83+(z-8)*h[2]*.46+.025*Math.sin(x*.6+z*.9);
    }
    write(phi,input);const e=d.createCommandEncoder();resolve.encode(e,resolveGroup);d.queue.submit([e.finish()]);
    for(const stage of ["advect","redistance"] as const){
     const results:Float32Array[]=[];
     for(let i=0;i<surfaces.length;i++){
      write(outputPhi,new Float32Array(count).fill(123));const encoder=d.createCommandEncoder();surfaces[i]!.encode(encoder,stage,groups[i]!);d.queue.submit([encoder.finish()]);results.push(await readMixedTexture(d,outputPhi));
     }
     assert.ok(results[1]!.some(v=>v!==123),"stage must actually write vertices");
     // Solid-free twin: production folds tapsClear to a constant, the
     // reference keeps the per-tap branch, and Metal contracts the two tap
     // reductions differently. Same arithmetic, so compare to rounding there.
     const tolerance=present?0:1e-6;
     const first=results[1]!.findIndex((v,i)=>!(Math.abs(v-results[0]![i]!)<=tolerance));
     assert.equal(first,-1,`${present?"solid":"clear"} ${layoutName} ${shape} ${stage}: first difference ${first}: ${results[1]![first]} vs ${results[0]![first]}`);
    }
   }
  }
  }
  assert.deepEqual(errors,[]);
 }finally{surfaces.forEach(s=>s.destroy());ownership?.destroy();solid?.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());device?.destroy();}
});
