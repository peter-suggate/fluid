import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
import { WebGPUUniformVelocityExtrapolator } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("nearest extension preserves reflected MAC fields and excludes closed pressure dual faces", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform extension source and wall symmetry");
  let device: GPUDevice | undefined;
  let extension: WebGPUUniformVelocityExtrapolator | undefined;
  const textures: GPUTexture[] = [], buffers: GPUBuffer[] = [];
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    const d = device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    d.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const dims = [32,16,32] as const, padded = [34,18,34] as const;
    const texture = (size: readonly number[], format: GPUTextureFormat) => {
      const t = d.createTexture({ size: [...size], dimension: "3d", format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
      textures.push(t); return t;
    };
    const buffer = (size: number) => {
      const b = d.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      buffers.push(b); return b;
    };
    const density = texture(dims,"r32float"), open = texture(dims,"rgba32float");
    const velocity = texture(dims,"rgba32float"), predicted = texture(dims,"rgba32float");
    const transport = texture(padded,"rgba32float"), predictedTransport = texture(padded,"rgba32float");
    const params = buffer(208), active = buffer(64), scratch = buffer(4);
    const parameters = new Float32Array(52); parameters.set([32,16,32,1/30,.05,.05,.05,0]);
    d.queue.writeBuffer(params,0,parameters);
    const region = new Uint32Array(16);region.set(dims,10);region.set([8,4,8],13);d.queue.writeBuffer(active,0,region);
    extension = new WebGPUUniformVelocityExtrapolator(d,dims,[.05,.05,.05],params,density,open,velocity,predicted,transport,predictedTransport,active,scratch,undefined,true,true);
    extension.setFrontPasses(2);await extension.initialize();
    const count = 32*16*32;
    const rho = new Float32Array(count), faces = new Float32Array(4*count), speeds = new Float32Array(4*count);
    const at = (x:number,y:number,z:number) => x+32*(y+16*z);
    const write = (t:GPUTexture,data:Float32Array,c:number) => d.queue.writeTexture({texture:t},data as Float32Array<ArrayBuffer>,{bytesPerRow:32*c*4,rowsPerImage:16},dims);
    for (const inset of [8,1]) for (const sweeps of [1,2,4]) {
      extension.setFrontPasses(sweeps);
      for(let z=0;z<32;z++)for(let y=0;y<16;y++)for(let x=0;x<32;x++) {
        const i=at(x,y,z);
        rho[i]=x>=inset&&x<32-inset&&z>=inset&&z<32-inset&&y<4?1:0;
        // Pressure uses half a dual cell at closed walls. This is not an
        // open transport face and must never seed the extension hierarchy.
        faces[4*i]=x===31?.5:1;faces[4*i+1]=y===15?.5:1;faces[4*i+2]=z===31?.5:1;
        speeds[4*i]=(x+1-16)*(16-y)/256;
        speeds[4*i+1]=-((x-15.5)**2+(z-15.5)**2+8*y)/1024;
        speeds[4*i+2]=(z+1-16)*(16-y)/256;
      }
      write(density,rho,1);write(open,faces,4);write(velocity,speeds,4);
      const encoder=d.createCommandEncoder();extension.encode(encoder,false);d.queue.submit([encoder.finish()]);
      const field=await readRgbaTexture3D(d,transport,...padded);
      const face=(x:number,y:number,z:number,c:number)=>field[4*((x+1)+34*((y+1)+18*(z+1)))+c]!;
      let worst=0;
      for(let z=0;z<32;z++)for(let y=0;y<16;y++)for(let x=0;x<32;x++)for(let c=0;c<3;c++) {
        const value=face(x,y,z,c);
        const rx=31-x-(c===0?1:0),rz=31-z-(c===2?1:0);
        if(rx>=0)worst=Math.max(worst,Math.abs(value-(c===0?-1:1)*face(rx,y,z,c)));
        if(rz>=0)worst=Math.max(worst,Math.abs(value-(c===2?-1:1)*face(x,y,rz,c)));
        worst=Math.max(worst,Math.abs(value-face(z,y,x,c===1?1:2-c)));
        if([x,y,z][c]===dims[c]-1) {
          assert.equal(value,0,`closed wall component ${c} at ${x},${y},${z}`);
          assert.equal((Math.round(face(x,y,z,3))>>(c+3))&1,0,"closed wall has no open transport bit");
        }
      }
      assert.ok(worst<1e-6,`inset ${inset}: reflection/transpose error ${worst}`);
      assert.equal(extension.encodedPassCount,13+3*sweeps,"each extra sweep includes a shared convergence pass");
    }
    assert.deepEqual(errors,[]);
  } finally {
    extension?.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy());device?.destroy();await releaseWebGPUExclusiveLock();
  }
});
