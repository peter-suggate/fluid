import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { UniformSurfaceVolumeCorrection } from "../lib/methods/uniform/webgpu-uniform-surface-volume";
import { uniformSurfaceVolumeWGSL } from "../lib/methods/uniform/uniform-surface-volume.wgsl";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("3D total surface volume constraint",{timeout:120000},async t=>{
  await acquireWebGPUExclusiveLock("dawn-test","3D total surface volume");
  let device:GPUDevice|undefined;let correction:UniformSurfaceVolumeCorrection|undefined;let denseCorrection:UniformSurfaceVolumeCorrection|undefined;
  const textures:GPUTexture[]=[];
  try {
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=createProcessRetainedDawnGPU(dawn as NodeDawnProvider,["backend=metal"]);
    const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=managedGPUDevice(await adapter.requestDevice(),{requireWorkerRealm:false});
    const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
    const shader=device.createShaderModule({code:uniformSurfaceVolumeWGSL});
    assert.deepEqual((await shader.getCompilationInfo()).messages.filter(m=>m.type==="error").map(m=>`${m.lineNum}: ${m.message}`),[]);
    const dims=[20,18,16] as const,h=[.05,.06,.07] as const;
    const n=dims[0]*dims[1]*dims[2],vd=dims.map(x=>x+1),nv=vd[0]!*vd[1]!*vd[2]!;
    const texture=(size:readonly number[])=>{const texture=device!.createTexture({size:[...size],format:"r32float",dimension:"3d",usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});textures.push(texture);return texture;};
    const densePhi=texture(vd);
    const phi=texture(vd),volume=texture(dims),capacity=texture(dims);
    const write=(texture:GPUTexture,data:Float32Array)=>device!.queue.writeTexture({texture},data as Float32Array<ArrayBuffer>,{bytesPerRow:texture.width*4,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);
    const read=async(texture:GPUTexture)=>{
      const row=Math.ceil(texture.width*4/256)*256;
      const b=device!.createBuffer({size:row*texture.height*texture.depthOrArrayLayers,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
      try {const e=device!.createCommandEncoder();e.copyTextureToBuffer({texture},{buffer:b,bytesPerRow:row,rowsPerImage:texture.height},[texture.width,texture.height,texture.depthOrArrayLayers]);device!.queue.submit([e.finish()]);await b.mapAsync(GPUMapMode.READ);
        const source=new Float32Array(b.getMappedRange()),r=new Float32Array(texture.width*texture.height*texture.depthOrArrayLayers);
        for(let z=0;z<texture.depthOrArrayLayers;z++)for(let y=0;y<texture.height;y++)r.set(source.subarray((z*texture.height+y)*row/4,(z*texture.height+y)*row/4+texture.width),(z*texture.height+y)*texture.width);return r;
      } finally {b.unmap();b.destroy();}
    };
    correction=new UniformSurfaceVolumeCorrection(device,dims,h,phi,volume,capacity);await correction.initialize();
    denseCorrection=new UniformSurfaceVolumeCorrection(device,dims,h,densePhi,volume,capacity,undefined,false);
    await denseCorrection.initialize();
    const run=async()=>{
      const e=device!.createCommandEncoder();
      e.copyTextureToTexture({texture:phi},{texture:densePhi},vd);
      correction!.encode(e);denseCorrection!.encode(e);
      device!.queue.submit([e.finish()]);await device!.queue.onSubmittedWorkDone();
      const result=await read(phi),control=await read(densePhi);
      // Compact reduction changes summation order; compare physical phi in m.
      for(let i=0;i<result.length;i++)assert.ok(Number.isFinite(result[i])&&Math.abs(result[i]!-control[i]!)<2e-5*Math.min(...h),
        `bounded correction at ${i}: ${result[i]} vs dense ${control[i]}`);
      return result;
    };
    const coordinate=(i:number,d:readonly number[])=>[i%d[0]!,Math.floor(i/d[0]!)%d[1]!,Math.floor(i/(d[0]!*d[1]!))];
    write(capacity,new Float32Array(n).fill(1));
    for(const axis of [0,1,2]) await t.test(`fractional plane on axis ${axis}, stretched phi`,async()=>{
      const target=7.3;
      const values=Float32Array.from({length:n},(_,i)=>Math.min(1,Math.max(0,target-coordinate(i,dims)[axis]!)));
      const original=Float32Array.from({length:nv},(_,i)=>(coordinate(i,vd)[axis]!-target+.4)*h[axis]!*6);
      write(phi,original);write(volume,values);const fixed=await run();
      const stride=[1,vd[0]!,vd[0]!*vd[1]!][axis]!;
      for(let i=0;i<nv;i++) if(coordinate(i,vd)[axis]===7) {
        const position=7-fixed[i]!/(fixed[i+stride]!-fixed[i]!);
        assert.ok(Math.abs(position-target)<2e-4, `axis ${axis}: surface ${position}, expected ${target}`);
      }
      assert.deepEqual(await read(volume),values,"V is never changed");
      assert.equal(fixed[nv-1],original[nv-1],"far air is untouched");
      const again=await run();for(let i=0;i<nv;i++)assert.ok(Math.abs(again[i]!-fixed[i]!)<2e-5,"matching volume stays still");
    });
    // Sphere of radius .28 about the lattice centre: its x extent is cells 4.4..15.6.
    const sphere=Float32Array.from({length:nv},(_,i)=>{
      const q=coordinate(i,vd);return Math.hypot(...q.map((x,a)=>(x-dims[a]!/2)*h[a]!))-.28;
    });
    const occupancy=(field:Float32Array,caps:Float32Array)=>Float32Array.from({length:n},(_,i)=>{
      if(!caps[i])return 0;const q=coordinate(i,dims);
      const c=Array.from({length:8},(_,k)=>field[q[0]!+(k&1)+vd[0]!*(q[1]!+((k>>1)&1)+vd[1]!*(q[2]!+(k>>2)))]!);
      if(c.every(v=>v<0))return 1;if(c.every(v=>v>=0))return 0;
      let count=0;
      for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
        const f=[(x+.5)/8,(y+.5)/8,(z+.5)/8];let v=0;
        for(let k=0;k<8;k++)v+=c[k]!*((k&1)?f[0]!:1-f[0]!)*((k&2)?f[1]!:1-f[1]!)*((k&4)?f[2]!:1-f[2]!);
        if(v<0)count++;
      }return count/512;
    });
    const shrunkenSphere=async(caps:Float32Array)=>{
      const values=occupancy(sphere,caps),desired=values.reduce((a,b)=>a+b,0);
      const shrunken=Float32Array.from(sphere,v=>(v+.015)*4);
      write(phi,shrunken);write(volume,values);write(capacity,caps);
      const fixed=await run();const measured=occupancy(fixed,caps).reduce((a,b)=>a+b,0);
      assert.ok(Math.abs(measured-desired)/desired<.02,`${measured} vs ${desired}`);
      assert.deepEqual(await read(volume),values);
      return {shrunken,fixed};
    };
    for (const solid of [false,true]) await t.test(`curved surface matches independent quadrature${solid ? " with solid capacity" : ""}`,async()=>{
      await shrunkenSphere(Float32Array.from({length:n},(_,i)=>solid && coordinate(i,dims)[0]!<10 ? 0 : 1));
    });
    await t.test("the band never crosses a one-cell solid slab",async()=>{
      // The sphere ends at x=15.6, inside the open cells next to a slab at x=16.
      // The shift may reach the slab's near face; nothing beyond it may move.
      const {shrunken,fixed}=await shrunkenSphere(Float32Array.from({length:n},(_,i)=>coordinate(i,dims)[0]===16 ? 0 : 1));
      for(let i=0;i<nv;i++) if(coordinate(i,vd)[0]!>=17) assert.equal(fixed[i],shrunken[i],`vertex ${coordinate(i,vd)} beyond the slab moved`);
    });
    await t.test("empty surface cannot seed liquid from V",async()=>{
      const original=new Float32Array(nv).fill(.5);write(phi,original);write(volume,new Float32Array(n).fill(.2));
      assert.deepEqual(await run(),original);
    });
    await t.test("zero work after a nonempty surface clears reused band scratch",async()=>{
      const original=new Float32Array(nv).fill(.5);
      write(phi,original);write(volume,new Float32Array(n));write(capacity,new Float32Array(n).fill(1));
      assert.deepEqual(await run(),original);
    });
    await t.test("full liquid with no interface retains bulk contributions",async()=>{
      const original=new Float32Array(nv).fill(-.5);
      write(phi,original);write(volume,new Float32Array(n).fill(1));
      assert.deepEqual(await run(),original);
    });
    await t.test("remote volume and disconnected boundary surfaces enter the global constraint",async()=>{
      const original=Float32Array.from({length:nv},(_,i)=>{
        const q=coordinate(i,vd);
        return Math.min(Math.hypot(...q.map((v,a)=>(v-2)*h[a]!))-.07,
          Math.hypot(...q.map((v,a)=>(v-(dims[a]!-2))*h[a]!))-.08);
      });
      const values=new Float32Array(n);values[n/2]=.25;values[0]=.3;values[n-1]=.4;
      write(phi,original);write(volume,values);await run();
      assert.deepEqual(await read(volume),values);
    });
    assert.deepEqual(errors,[]);
  } finally {denseCorrection?.destroy();correction?.destroy();for(const texture of textures)texture.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
