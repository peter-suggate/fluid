import assert from "node:assert/strict";
import test from "node:test";
import { createCm12NumericsWGSL } from "../lib/core/cm12-numerics";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { uniformCoarseSolverWGSL, uniformPressureStateWGSL, UNIFORM_CM11A_COARSE_HEADER_BYTES, UNIFORM_CM11A_COARSE_ROW_BYTES } from "../lib/methods/uniform/uniform-coarse-solver.wgsl";
import { createHeroGardenHoseScene } from "../lib/core/hero-garden-scene";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

// Isolate the production coarse kernel. Unit spacing and phi = +/- 0.5 make
// each open liquid/air face coefficient 2 and each liquid/liquid coefficient 1.
const fixtureWGSL = /* wgsl */ `
struct MG { fineDims:vec4u, levelDims:vec4u, coarseDims:vec4u, spacing:vec4f, control:vec4u };
@group(1) @binding(0) var<uniform> mg:MG;
@group(1) @binding(1) var mgPressureIn:texture_3d<f32>;
@group(1) @binding(2) var mgPressureOut:texture_storage_3d<r32float,write>;
@group(1) @binding(3) var mgRhsIn:texture_3d<f32>;
@group(1) @binding(5) var mgPhiIn:texture_3d<f32>;
@group(1) @binding(7) var mgVolumeIn:texture_3d<f32>;
@group(1) @binding(11) var mgMinimumIn:texture_3d<f32>;
${uniformPressureStateWGSL}
struct Params { dimsDt:vec4f, physical:vec4f, boundary:vec4f };
const params=Params(vec4f(0,0,0,1),vec4f(1,0,0,0),vec4f(0));
var<workgroup> mgCycleStopped:u32;
fn mgSkipCycle()->bool{return false;}
fn mgP(p:vec3i)->f32{return textureLoad(mgPressureIn,p,0).x;}
fn mgPhi(p:vec3i)->f32{return textureLoad(mgPhiIn,p,0).x;}
fn mgTopology(p:vec3i)->vec4f{return textureLoad(mgVolumeIn,p,0);}
fn mgInterior(p:vec3i,d:vec3u)->bool{return all(p>=vec3i(1))&&all(p<vec3i(d)-vec3i(1));}
fn mgD4Sum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
${createCm12NumericsWGSL()}
`;

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("coarse pressure solves grids larger than one lane per cell", { timeout: 180_000 }, async t => {
  await acquireWebGPUExclusiveLock("dawn-test", "uniform strided coarse solve");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href) as NodeDawnProvider;
    Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
    const errors: string[] = [];
    device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    const module = device.createShaderModule({ code: fixtureWGSL + uniformCoarseSolverWGSL });
    const compilation = await module.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(m => m.type === "error"), []);
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "mgSolveCoarsest" } });

    async function solve(dims: [number,number,number], cap = 4096) {
      const gpuDevice = device!;
      const [dx,dy,dz]=dims, count=dx*dy*dz;
      const index=(x:number,y:number,z:number)=>x+dx*(y+dy*z);
      const interior=(x:number,y:number,z:number)=>x>0&&x<dx-1&&y>0&&y<dy-1&&z>0&&z<dz-1;
      const expected=new Float32Array(count), phi=new Float32Array(count).fill(0.5), rhs=new Float32Array(count);
      const minimum=new Float32Array(count), topology=new Float32Array(count*4).fill(1);
      // Mixed free and lower-bound-active rows, deliberately crossing lane 256.
      for(let z=1;z<dz-1;z++)for(let y=1;y<dy-1;y++)for(let x=1;x<dx-1;x++) {
        const i=index(x,y,z);phi[i]=-0.5;
        minimum[i]=(x+y+z)%7===0 ? 0.25 : 0;
        expected[i]=minimum[i]+((x+2*y+z)%4===0 ? 0 : 0.5+0.125*((x+y+z)%3));
      }
      const offsets=[[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]];
      for(let z=1;z<dz-1;z++)for(let y=1;y<dy-1;y++)for(let x=1;x<dx-1;x++) {
        const i=index(x,y,z);
        let b=0;
        for(const [ox,oy,oz] of offsets) {
          const liquid=interior(x+ox,y+oy,z+oz);
          b+=(liquid?1:2)*(expected[i]-(liquid?expected[index(x+ox,y+oy,z+oz)]:0));
        }
        rhs[i]=b-(expected[i]===minimum[i]?0.75:0);
      }
      const owned: (GPUBuffer|GPUTexture)[]=[];
      const buffer=(size:number,usage:GPUBufferUsageFlags)=>{
        const b=gpuDevice.createBuffer({size,usage});owned.push(b);return b;
      };
      const texture=(data:Float32Array<ArrayBuffer>, components=1)=>{
        const tex=gpuDevice.createTexture({size:dims,dimension:"3d",format:components===4?"rgba32float":"r32float",
          usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.COPY_SRC});
        owned.push(tex);gpuDevice.queue.writeTexture({texture:tex},data,{bytesPerRow:dx*4*components,rowsPerImage:dy},dims);return tex;
      };
      try {
        const params=buffer(80,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
        const words=new Uint32Array(20);words.set([...dims,0],4);words.set([0,0,1,cap],16);
        new Float32Array(words.buffer).set([1,1,1,0],12);gpuDevice.queue.writeBuffer(params,0,words);
        const diagnostic=buffer(UNIFORM_CM11A_COARSE_HEADER_BYTES + count*UNIFORM_CM11A_COARSE_ROW_BYTES,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
        const output=texture(new Float32Array(count));
        const entries: GPUBindGroupEntry[]=[{binding:0,resource:{buffer:params}},
          {binding:1,resource:texture(new Float32Array(count)).createView()},
          {binding:2,resource:output.createView()}, {binding:3,resource:texture(rhs).createView()},
          {binding:5,resource:texture(phi).createView()}, {binding:7,resource:texture(topology,4).createView()},
          {binding:11,resource:texture(minimum).createView()}, {binding:13,resource:{buffer:diagnostic}}];
        const group=gpuDevice.createBindGroup({layout:pipeline.getBindGroupLayout(1),entries});
        const empty=gpuDevice.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[]});
        const stride=Math.ceil(dx*4/256)*256;
        const read=buffer(stride*dy*dz,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
        const diagRead=buffer(104,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);
        const encoder=gpuDevice.createCommandEncoder();const pass=encoder.beginComputePass();
        pass.setPipeline(pipeline);pass.setBindGroup(0,empty);pass.setBindGroup(1,group);pass.dispatchWorkgroups(1);pass.end();
        encoder.copyTextureToBuffer({texture:output},{buffer:read,bytesPerRow:stride,rowsPerImage:dy},dims);
        encoder.copyBufferToBuffer(diagnostic,0,diagRead,0,104);gpuDevice.queue.submit([encoder.finish()]);
        await Promise.all([read.mapAsync(GPUMapMode.READ),diagRead.mapAsync(GPUMapMode.READ)]);
        const pressure=new Float32Array(count);
        const mapped=read.getMappedRange();
        for(let z=0;z<dz;z++)for(let y=0;y<dy;y++)pressure.set(new Float32Array(mapped,stride*(y+dy*z),dx),dx*(y+dy*z));
        const diagnostics=new Uint32Array(diagRead.getMappedRange().slice(0));
        read.unmap();diagRead.unmap();
        if(cap===4096) {
          assert.equal(diagnostics[3],0,"coarse solve must converge without exhausting its cap");
          let error=0;for(let i=0;i<count;i++) {assert.ok(pressure[i]>=minimum[i]);error=Math.max(error,Math.abs(pressure[i]-expected[i]));}
          assert.ok(error<1e-4,`manufactured pressure error ${error}`);
        } else {
          assert.equal(diagnostics[3],1,"a one-sweep budget must report nonconvergence");
          assert.ok(diagnostics[12]>0&&diagnostics[13]>0,"both constrained and free rows must be diagnosed");
          assert.ok((diagnostics[14]&0x3fffffff)<count,"worst-row index must decode within the grid");
        }
        return {pressure,diagnostics};
      } finally {owned.forEach(resource=>resource.destroy());}
    }

    for(const dims of [[4,4,4],[5,5,5],[11,5,5],[21,9,7],[81,5,5]] as [number,number,number][]) {
      await t.test(`manufactured constrained pressure ${dims.join("x")}`,async()=>{
        await solve(dims);await solve(dims,1);
      });
    }
    await t.test("authored wet garden constructs and advances with default geometric windows",async()=>{
      const scene=createHeroGardenHoseScene({water:true});
      const solver=await WebGPUUniformReferenceSolver.createAsync(device!,scene,"balanced",undefined,
        uniformGeometricSolverOptions({},scene),()=>{});
      try {
        assert.deepEqual([solver.info.nx,solver.info.ny,solver.info.nz],[72,48,48]);
        solver.enableCM11aCoarsestCapture(1);
        for(let frame=1;frame<=3;frame++) {
          while(!solver.advanceTo(frame/30,[]))await new Promise(setImmediate);
          const stats=await solver.readStats() as unknown as Record<string,number|boolean>;
          assert.ok(Number.isFinite(stats.uniformCM11aFineResidualInfinity));
          assert.ok(Number(stats.uniformPressureAcceptedResidual)<=Number(stats.uniformPressureInitialResidual));
          assert.ok(Number.isFinite(stats.maxSpeed_m_s));
        }
        const capture=await solver.readCM11aCoarsestCapture();assert.ok(capture);
        assert.equal(capture.pressure.length,capture.dimensions.reduce((a,b)=>a*b,1));
        assert.ok(capture.pressure.every(Number.isFinite));
      } finally {solver.destroy();}
    });
    assert.deepEqual(errors,[]);
  } finally {device?.destroy();await releaseWebGPUExclusiveLock();}
});
