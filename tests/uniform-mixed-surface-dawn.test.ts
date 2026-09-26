import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedSurface} from "../lib/methods/uniform/uniform-mixed-surface";
import {mixedPressureFixture,mixedPressureLayouts} from "./helpers/uniform-mixed-pressure";
import {seamLayout} from "./helpers/uniform-geometric-seam";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed surface advances canonical vertices and cell traces without expanding fine fields",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed surface");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!]){
   const d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,size=d.map(n=>n+1),n=d.reduce((a,b)=>a*b),nv=size.reduce((a,b)=>a*b);
   const cells=mixedPressureFixture(layout).cells.map(c=>({width:c.width,origin:c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))}));
   const canonical=(p:number[])=>{const c=cells.filter(c=>c.origin.every((v,a)=>p[a]!>=v&&p[a]!<=v+c.width)).sort((a,b)=>b.width-a.width)[0]!;return c.origin.every((v,a)=>(p[a]!-v)%c.width===0);};
   const points:number[][]=[];for(let z=0;z<=d[2];z++)for(let y=0;y<=d[1];y++)for(let x=0;x<=d[0];x++)points.push([x,y,z]);
   const ownership:UniformMixedOwnership=new UniformMixedOwnership(device,layout);const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=(format:GPUTextureFormat,extent:readonly number[])=>{const t=device!.createTexture({size:[...extent],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});textures.push(t);return t;};
   const buffer=(bytes:number,uniform=false)=>{const b=device!.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)});buffers.push(b);return b;};
   const phi=texture("r32float",size),outputPhi=texture("r32float",size),velocity=texture("rgba32float",d),coarseVelocity=texture("rgba32float",d.map(n=>n/4+2)),volume=texture("r32float",d),departures=texture("rgba32float",d);
   const negative=buffer(4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2])),params=buffer(32,true);
   const upload=(t:GPUTexture,data:Float32Array<ArrayBuffer>)=>device!.queue.writeTexture({texture:t},data,{bytesPerRow:t.width*4*(t.format==="rgba32float"?4:1),rowsPerImage:t.height},[t.width,t.height,t.depthOrArrayLayers]);
   try{
    const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("surface allocated a field");};const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
    const stage:UniformMixedSurface=new UniformMixedSurface(borrowed,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const group=stage.bind({phi,outputPhi,velocity,coarseVelocity,volume,negative,params,departures,evidence:{buffer:buffer(layout.tiles.length*4)}});
    const speed=.03,dt=.1;
    upload(velocity,Float32Array.from({length:n*4},(_,i)=>i%4===1?speed:0));
    upload(coarseVelocity,Float32Array.from({length:coarseVelocity.width*coarseVelocity.height*coarseVelocity.depthOrArrayLayers*4},(_,i)=>i%4===1?speed:0));
    const boundaries=new Float32Array(negative.size/4);boundaries.fill(speed,d[1]*d[2],d[1]*d[2]+d[0]*d[2]);device.queue.writeBuffer(negative,0,boundaries);
    device.queue.writeBuffer(params,0,new Float32Array([...h,dt]));device.queue.writeBuffer(params,16,new Uint32Array([0,1,1,4]));
    const canonicalFlags=points.map(canonical);
    const values=Float32Array.from(points,(p,i)=>canonicalFlags[i]?(p[1]!-d[1]/2)*h[1]:NaN);
    upload(phi,values);upload(outputPhi,new Float32Array(nv).fill(NaN));upload(volume,new Float32Array(n).fill(1));
    const encoder=device.createCommandEncoder();stage.encode(encoder,"advect",group);stage.encode(encoder,"traceCells",group);device.queue.submit([encoder.finish()]);
    const result=await readMixedTexture(device,outputPhi),trace=await readMixedTexture(device,departures);
    if(layout.fineTiles.length===layout.tiles.length){
     const count=layout.tiles.length,header=new Uint32Array(16);header[1]=count;
     header.set([count,1,1],4);
     device.queue.writeBuffer(ownership.support,count*16,header);
     device.queue.writeBuffer(ownership.support,(5*count+16)*4,layout.fineTiles);
     device.queue.writeBuffer(ownership.certifiedDispatch,0,header.subarray(4,12));
     const fast=device.createCommandEncoder();stage.encode(fast,"advect",group);stage.encode(fast,"traceCells",group);device.queue.submit([fast.finish()]);
     const fastPhi=await readMixedTexture(device,outputPhi),fastTrace=await readMixedTexture(device,departures);
     for(let i=0;i<result.length;i++)assert.ok(Math.abs(result[i]!-fastPhi[i]!)<1e-6,"certified interpolation changed the surface");
     for(let i=0;i<trace.length;i++)assert.ok(Math.abs(trace[i]!-fastTrace[i]!)<1e-6,"certified interpolation changed the characteristic");
     ownership.update(layout);
    }
    for(let i=0;i<points.length;i++){
     if(!canonicalFlags[i]){assert.ok(Number.isNaN(result[i]),"surface expanded an inactive vertex");continue;}
     assert.ok(Number.isFinite(result[i]),`invalid canonical phi ${points[i]}`);
     const y=points[i]![1]!;
     if(y>=4&&y<=d[1]-4)assert.ok(Math.abs(result[i]!-(values[i]!-speed*dt))<2e-5,`translated plane ${result[i]} != ${values[i]!-speed*dt}`);
    }
    for(const cell of cells){const at=cell.origin[0]!+d[0]*(cell.origin[1]!+d[1]*cell.origin[2]!);
     for(let a=0;a<3;a++){const expected=Math.max(0,Math.min(d[a]!,cell.origin[a]!+cell.width/2-(a===1?speed*dt/h[a]!:0)));assert.ok(Math.abs(trace[4*at+a]!-expected)<2e-5,"cell RK2 characteristic");}}
    // The once-per-frame speed bound may exclude wall probes, but must
    // preserve their actual continuation, including released negative walls.
    const releasedVelocity=Float32Array.from({length:n*4},(_,i)=>i%4===1?speed:i%4===3?16:0);
    upload(velocity,releasedVelocity);
    const withoutBound=device.createCommandEncoder();stage.encode(withoutBound,"advect",group);device.queue.submit([withoutBound.finish()]);
    const unbounded=await readMixedTexture(device,outputPhi);
    const boundHeader=new Uint32Array([new Uint32Array(new Float32Array([speed]).buffer)[0]!,0,layout.fineTiles.length,1]);
    device.queue.writeBuffer(ownership.support,layout.tiles.length*16,boundHeader);
    const withBound=device.createCommandEncoder();stage.encode(withBound,"advect",group);device.queue.submit([withBound.finish()]);
    const bounded=await readMixedTexture(device,outputPhi);
    for(let i=0;i<points.length;i++)if(canonicalFlags[i])assert.equal(bounded[i],unbounded[i],"frame speed bound changed a wall continuation");
    ownership.update(layout);
    // Zero-velocity rebuild of an affine signed distance is an identity.
    upload(phi,values);const e=device.createCommandEncoder();stage.encode(e,"redistance",group);device.queue.submit([e.finish()]);
    const rebuilt=await readMixedTexture(device,outputPhi);
    for(let i=0;i<points.length;i++)if(canonicalFlags[i])assert.ok(Math.abs(rebuilt[i]!-values[i]!)<2e-5,"planar redistance changed the contour");
    // A drained positive plateau has no contour for Newton to find. It
    // retires to the physical band only with draining enabled.
    const plateau=Float32Array.from(points,(_,i)=>canonicalFlags[i]?.5*Math.min(...h):NaN);
    upload(phi,plateau);
    const retire=device.createCommandEncoder();stage.encode(retire,"redistance",group);device.queue.submit([retire.finish()]);
    const retired=await readMixedTexture(device,outputPhi);
    for(let i=0;i<points.length;i++)if(canonicalFlags[i]){
     const p=points[i]!;const owner=cells.filter(c=>c.origin.every((v,a)=>p[a]!>=v&&p[a]!<=v+c.width)).sort((a,b)=>b.width-a.width)[0]!;
     assert.ok(Math.abs(retired[i]!-4*owner.width*Math.max(...h))<2e-5,"drained plateau did not retire");
    }
    device.queue.writeBuffer(params,16,new Uint32Array([0,1,0,4]));
    const disabled=device.createCommandEncoder();stage.encode(disabled,"redistance",group);device.queue.submit([disabled.finish()]);
    const retained=await readMixedTexture(device,outputPhi);
    for(let i=0;i<points.length;i++)if(canonicalFlags[i])assert.equal(retained[i],plateau[i],"disabled drain retired phi");
    // A zero on the outer wall is real surface evidence, including when it
    // lies beyond the local Newton stencil of a flat positive vertex.
    plateau[0]=0;upload(phi,plateau);device.queue.writeBuffer(params,16,new Uint32Array([0,1,1,4]));
    const contact=device.createCommandEncoder();stage.encode(contact,"redistance",group);device.queue.submit([contact.finish()]);
    const protectedPhi=await readMixedTexture(device,outputPhi);
    assert.ok(protectedPhi[4]!<4*Math.max(...h),"nearby wall contour was retired");
    assert.deepEqual(errors,[]);
   }finally{ownership.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());}
  }
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
