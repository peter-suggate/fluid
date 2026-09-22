import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readRgbaTexture3D } from "../lib/harness/webgpu-smoke-readbacks";
import { WebGPUUniformVelocityExtrapolator } from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";

function equalFields(actual: Float32Array, expected: Float32Array, message: string): void {
  assert.equal(actual.length, expected.length);
  const first=actual.findIndex((value,i)=>value!==expected[i]);
  assert.equal(first,-1,`${message}: first difference at ${first}: ${actual[first]} vs ${expected[first]}`);
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("nearest source extension isolates a falling drop from a remote pool with two sweeps", { timeout: 120_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "nearest source extension");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const d = device;
    const dims = [64, 64, 8] as const, padded = [66, 66, 10] as const;
    async function run(pool: boolean, nearest: boolean, fused: boolean, predicted = false, varying = false, shell: "dense" | "full" | "empty" = "dense") {
      const textures: GPUTexture[] = [], buffers: GPUBuffer[] = [];
      const texture = (size: readonly number[], format: GPUTextureFormat) => {
        const t = d.createTexture({ size: [...size], dimension: "3d", format,
          usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
        textures.push(t); return t;
      };
      const buffer = (size: number, usage: number) => { const b = d.createBuffer({ size, usage }); buffers.push(b); return b; };
      const density = texture(dims, "r32float"), open = texture(dims, "rgba32float");
      const velocity = texture(dims, "rgba32float"), predictedVelocity = texture(dims, "rgba32float");
      const transport = texture(padded, "rgba32float"), predictedTransport = texture(padded, "rgba32float");
      const rho = new Float32Array(64*64*8), faces = new Float32Array(rho.length*4), speeds = new Float32Array(faces.length);
      for (let z=0;z<8;z++) for (let y=0;y<64;y++) for (let x=0;x<64;x++) {
        const i=x+64*(y+64*z), drop=x>=24&&x<40&&y>=40&&y<48;
        rho[i] = drop || (pool&&y<8) ? 1 : 0;
        faces[4*i]=x<63?1:0; faces[4*i+1]=y<63?1:0; faces[4*i+2]=z<7?1:0;
        if (y>=32) speeds[4*i+1]=-4;
        if (varying) {
          speeds[4*i] = Math.sin(x*.21)*Math.cos(z*.37);
          speeds[4*i+1] *= 1+.15*Math.sin(x*.17+z*.31);
          speeds[4*i+2] = Math.cos(x*.13)*Math.sin(z*.27);
        }
      }
      const write = (t: GPUTexture, data: Float32Array, components: number) => d.queue.writeTexture({texture:t}, data as Float32Array<ArrayBuffer>, {bytesPerRow:64*components*4,rowsPerImage:64},dims);
      write(density,rho,1);write(open,faces,4);write(velocity,speeds,4);
      write(predictedVelocity,Float32Array.from(speeds,v=>v*2),4);
      const params = buffer(208,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST), values = new Float32Array(52);
      values.set([64,64,8,1/30, .05,.05,.05,0]);/* twoLevel.y */ values[45]=shell==="dense"?0:1;d.queue.writeBuffer(params,0,values);
      const active = buffer(64,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST), region = new Uint32Array(16);
      region.set(dims,10);region.set([16,16,2],13);d.queue.writeBuffer(active,0,region);
      const tileCount=16*16*2;
      const scratch = buffer(shell==="dense"?4:4*(rho.length+4*tileCount),GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
      if(shell!=="dense") {
        const classes=new Uint32Array(4*tileCount);
        if(shell==="full")for(let i=0;i<tileCount;i++)classes[4*i+3]=2;
        d.queue.writeBuffer(scratch,4*rho.length,classes);
      }
      const extension = new WebGPUUniformVelocityExtrapolator(d,dims,[.05,.05,.05],params,density,open,velocity,predictedVelocity,transport,predictedTransport,active,scratch,undefined,nearest,fused);
      try {
        extension.setFrontPasses(2);await extension.initialize();
        const encoder=d.createCommandEncoder();extension.encode(encoder,predicted,undefined,false,shell!=="dense");d.queue.submit([encoder.finish()]);
        const field=await readRgbaTexture3D(d,predicted?predictedTransport:transport,...padded);
        return {field,passes:extension.encodedPassCount};
      } finally { extension.destroy();textures.forEach(t=>t.destroy());buffers.forEach(b=>b.destroy()); }
    }
    const prior=await run(true,false,false), fixed=await run(true,true,true), unfused=await run(true,true,false), alone=await run(false,true,true);
    assert.deepEqual(errors,[]);
    const at=(field:Float32Array,y:number)=>field[4*(33+66*(y+1+66*5))+1]!;
    assert.ok(at(prior.field,36)>-3.9,"control must reproduce remote-pool dilution");
    for(let y=34;y<=46;y++) {
      assert.ok(Math.abs(at(fixed.field,y)+4)<1e-6,`drop extension y=${y}: ${at(fixed.field,y)}`);
      assert.equal(at(fixed.field,y),at(alone.field,y));
    }
    assert.equal(at(fixed.field,4),0,"pool source stays stationary");
    equalFields(fixed.field,unfused.field,"fused packing preserves every value and known/open bit");
    assert.equal(fixed.passes,unfused.passes-1);
    const predicted=await run(true,true,true,true), predictedUnfused=await run(true,true,false,true,false,"full");
    equalFields(predicted.field,predictedUnfused.field,"predicted compact shell");assert.equal(at(predicted.field,36),-8);
    const varying=await run(true,true,true,false,true), varyingUnfused=await run(true,true,false,false,true,"full");
    equalFields(varying.field,varyingUnfused.field,"packing also agrees for varying three-component velocities");
    const empty=await run(false,true,true,false,false,"empty");
    assert.ok(empty.field.every(v=>v===0),"empty shell issues zero finest work without publishing stale transport");
    assert.deepEqual(errors,[]);
    console.log(JSON.stringify({prior:at(prior.field,36),fixed:at(fixed.field,36),passes:fixed.passes}));
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
