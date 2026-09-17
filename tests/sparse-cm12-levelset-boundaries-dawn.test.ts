import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { sparseCM12LevelSetBoundariesWGSL } from "../lib/methods/adaptive-volume/sparse-cm12-levelset-boundaries.wgsl";

const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("3D wall continuation and ambient inflow use physical faces, diagonal old-phi support, and source precedence",{timeout:60_000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","3D scalar boundary contracts");let device:GPUDevice|undefined;
  try{
    const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
    const gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);const adapter=await gpu.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice();assert.ok(device);
    const code=`
struct Params {dimensions:vec4u,frame:vec4f,acceleration:vec4f,position:vec4f,control:vec4f}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read_write>state:array<f32>;
const INVALID=0xffffffffu;
fn compactOwnerCellAt(q:vec3i)->vec2u{return vec2u(select(INVALID,0u,all(q>=vec3i(0))&&all(q<vec3i(p.dimensions.xyz))),0u);}
fn cellOpenVolume(c:u32)->f32{_=c;return p.control.w;}
fn cnxCellIncidenceRangeUnchecked(c:u32)->vec2u{_=c;return vec2u(0u,6u);}
fn cnxIncidenceRowOrdinalUnchecked(i:u32)->u32{return i;}
fn cnxStableRowUnchecked(i:u32)->u32{return i;}
fn rowKind(r:u32)->u32{_=r;return 3u;}
fn rowAxis(r:u32)->u32{return r/2u;}
fn rowOpenFraction(r:u32)->f32{return select(0.0,p.control.x,r==u32(p.position.w));}
fn rowSolidVelocity(r:u32)->f32{_=r;return .3;}
fn rowSeparatingFromClosedWorld(r:u32)->bool{return (u32(p.control.z)&(1u<<r))!=0u;}
fn sparseCM12InflowFaceCoverage(r:u32)->f32{return select(0.0,p.control.y,r==u32(p.position.w));}
fn cnxRowTermRangeByOrdinalUnchecked(r:u32)->vec2u{return vec2u(r,r+1u);}
fn cnxRowTermCellUnchecked(t:u32)->u32{_=t;return 0u;}
fn cnxRowTermCoefficientUnchecked(t:u32)->f32{return select(1.0,-1.0,(t&1u)!=0u);}
fn rowCenter(r:u32)->vec3f{var q=vec3f(p.dimensions.xyz)*.5;q[r/2u]=select(0.0,f32(p.dimensions[r/2u]),(r&1u)!=0u);return q;}
fn cellWidths(c:u32)->vec3f{_=c;return vec3f(p.dimensions.xyz);}
fn destinationFaceVelocity()->u32{return 0u;}
fn airBoundaryFaceVelocity(row:u32)->f32{return state[row];}
fn acceptedPointInsideSolid(q:vec3f)->bool{_=q;return false;}
fn lsvAcceptedVelocitySample(q:vec3f)->vec4f{_=q;return vec4f(0.0,0.0,0.0,1.0);}
fn lsvClipCharacteristic(a:vec3f,b:vec3f)->vec3f{_=a;return b;}
fn lsvAcceptedSlot()->u32{return 0u;}
struct Sample {phi:f32,valid:bool}
fn lsvSampleAtSlot(slot:u32,q:vec3f)->Sample{_=slot;return Sample(.25-length(q-p.position.xyz),true);}
${sparseCM12LevelSetBoundariesWGSL}
@compute @workgroup_size(1)fn probe(){let contact=cm12ClosedWallPhi(p.position.xyz);let air=cm12ReleasedWallPhi(p.position.xyz);
 state[6]=contact.x;state[7]=contact.y;state[8]=air.x;state[9]=air.y;}`;
    const module=device.createShaderModule({code});assert.deepEqual((await module.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
    const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"probe"}});
    const params=device.createBuffer({size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const state=device.createBuffer({size:40,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});
    const read=device.createBuffer({size:16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:{buffer:state}}]});
    const run=async(face:number,position:number[],open=0,source=0,release=0,dt=.25,capacity=1,speed=2,enabled=1)=>{
      const data=new ArrayBuffer(80);const u=new Uint32Array(data),f=new Float32Array(data);
      u.set([8,8,8,0]);f.set([dt,1,1,enabled],4);const gravity=[0,0,0,0];gravity[Math.floor(face/2)]=face&1?-1:1;f.set(gravity,8);
      f.set([...position,face],12);f.set([open,source,release,capacity],16);device!.queue.writeBuffer(params,0,data);
      const velocities=new Float32Array(10);velocities.fill(.3);velocities[face]=open>0?open*(face&1?-speed:speed)+(1-open)*.3:.3+(face&1?-speed:speed);
      device!.queue.writeBuffer(state,0,velocities);const encoder=device!.createCommandEncoder();const pass=encoder.beginComputePass();
      pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(state,24,read,0,16);
      device!.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=Array.from(new Float32Array(read.getMappedRange()));read.unmap();return result;
    };
    for(let face=0;face<6;face++){
      const q=[4,4,4];q[Math.floor(face/2)]=face&1?8:0;
      let r=await run(face,q);assert.equal(r[1],1);assert.ok(Math.abs(r[0]!+.75)<1e-6);assert.equal(r[3],0);
      for(const open of [.4,1]){
        r=await run(face,q,open);assert.equal(r[1],0);assert.equal(r[3],1);assert.ok(Math.abs(r[2]!-.5)<1e-6);
        r=await run(face,q,open,1);assert.equal(r[3],0,"prescribed liquid inflow excludes ambient air");
        r=await run(face,q,open,0,0,.25,1,-2);assert.equal(r[3],0,"outflow excludes ambient air");
        r=await run(face,q,open,0,0,.25,1,2,0);assert.equal(r[3],0,"off arm retains its scalar boundary policy");
      }
      r=await run(face,q,0,0,1<<face);assert.equal(r[1],0);assert.equal(r[3],1);assert.ok(Math.abs(r[2]!-.5)<1e-6);
      r=await run(face,q,0,0,0,0);assert.equal(r[1],0,"zero dt preserves wall phi");
      r=await run(face,q,0,0,0,.25,0);assert.equal(r[1],0,"zero capacity excludes contact");
    }
    for(const q of [[0,0,0],[8,8,8],[0,8,0]]){
      const r=await run(0,q);assert.equal(r[1],1);assert.ok(Math.abs(r[0]!-(.25-Math.sqrt(3)))<1e-6,"corner uses one diagonal trace");
    }
    params.destroy();state.destroy();read.destroy();
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
