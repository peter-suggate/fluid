import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedPressureAuthority } from "../lib/methods/uniform/uniform-mixed-pressure-authority";
import { mixedPressureFixture, mixedPressureLayouts } from "./helpers/uniform-mixed-pressure";
import { seamLayout } from "./helpers/uniform-geometric-seam";
import { readMixedBuffer, readMixedTexture } from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("mixed pressure authority preserves phase policies and physical excess/deficit balance",{timeout:180000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed pressure authority");let device:GPUDevice|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  for(const layout of [seamLayout(0,"fine"),seamLayout(0,"coarse"),mixedPressureLayouts()[0]!]){
   const fixture=mixedPressureFixture(layout),d=layout.lattice.dimensions,h=layout.lattice.cellSize_m,n=layout.cellCount,total=d[0]*d[1]*d[2],ownership=new UniformMixedOwnership(device,layout);
   const textures:GPUTexture[]=[],buffers:GPUBuffer[]=[];
   const texture=()=>{const t=device!.createTexture({size:[...d],dimension:"3d",format:"r32float",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(t);return t;};
   const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});buffers.push(b);return b;};
   try{
    const centerPhi=texture(),volume=texture(),targetFill=texture(),phase=texture(),correction=texture(),phi=buffer(n*4),params=buffer(16,true);
    const stage:UniformMixedPressureAuthority=new UniformMixedPressureAuthority(device,ownership);await stage.initialize();assert.equal(stage.allocatedBytes,0);
    const scratch=buffer(stage.scratchBytes),group=stage.bind({centerPhi,volume,targetFill,phase,correction,phi:{buffer:phi},params,scratch:{buffer:scratch}});
    const origins=fixture.cells.map(c=>c.center.map((v,a)=>Math.round(v/h[a]!-c.width/2))),index=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
    const raw=Float32Array.from(fixture.cells,(_,i)=>i%5===0?-.1:1),v=Float32Array.from(fixture.cells,(_,i)=>i%7===0?1.3:i%3===0?.7:.1),fill=Float32Array.from(fixture.cells,(_,i)=>i%2===0?.9:.4);
    for(const [t,source] of [[centerPhi,raw],[volume,v],[targetFill,fill]] as const){const field=new Float32Array(total).fill(NaN);origins.forEach((p,i)=>field[index(p)]=source[i]!);device.queue.writeTexture({texture:t},field,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},[...d]);}
    for(const fallback of [0,1,2])for(const airborne of [0,1])for(const dt of [1/30,1e-5]){
     const dust=.01;device.queue.writeBuffer(params,0,new Float32Array([dt,fallback,airborne,dust]));
     const expected=Float64Array.from(raw,(p,i)=>{
      const width=fixture.cells[i]!.width,localH=Math.min(...h)*width;if(airborne)p=Math.min(p,Math.max(localH*(1-v[i]!),-.5*localH));
      const vp=localH*(.5-v[i]!);if(fallback===0)return p;if(fallback===2)return Math.min(p,vp);if(p<0||vp>=0)return p;
      const neighbor=fixture.faces.some(f=>f.left===i&&raw[f.right]!<0||f.right===i&&raw[f.left]!<0);return neighbor?p:Math.max(vp,-.5*localH);
     });
     let positive=0,deficit=0;const excess=Float64Array.from(v,value=>Math.min(-Math.expm1(-dt*30*Math.LN2)*Math.max(0,value-1),1));
     const deficits=Float64Array.from(v,(value,i)=>value>1||expected[i]!>=0?0:Math.max(0,fill[i]!-value));
     fixture.cells.forEach((c,i)=>{positive+=excess[i]!*c.width**3;deficit+=deficits[i]!*c.width**3;});const rate=deficit?Math.min(1,positive/deficit):0;
     const encoder=device.createCommandEncoder();stage.encode(encoder,group);device.queue.submit([encoder.finish()]);
     const actual=await readMixedBuffer(device,phi),support=await readMixedTexture(device,phase),adjustment=await readMixedTexture(device,correction),balance=await readMixedBuffer(device,scratch);
     assert.ok(Math.abs(balance[0]!-rate)<2e-6);
     fixture.cells.forEach((c,i)=>{
      const at=index(origins[i]!);assert.ok(Math.abs(actual[i]!-expected[i]!)<1e-6,`phase ${fallback}/${airborne}/${i}`);
      const flying=airborne&&v[i]!>Math.max(dust,.05)&&raw[i]!>1.5*Math.min(...h)*c.width&&origins[i]!.every((p,a)=>p>=2*c.width&&p+3*c.width<=d[a]!);
      assert.equal(support[at],expected[i]!<0||flying?1:0);
      const correction=(excess[i]!-rate*deficits[i]!)/dt;assert.ok(Math.abs(adjustment[at]!-correction)<2e-5,`volume correction ${adjustment[at]} != ${correction}`);
     });
    }
   }finally{textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());ownership.destroy();}
  }
  assert.deepEqual(errors,[]);
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
