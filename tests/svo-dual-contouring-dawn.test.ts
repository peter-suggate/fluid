import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { svoDualContouringFitWGSL } from "../lib/svo/features/meshing/dual-contouring";
import { svoDualContouringMeshWGSL } from "../lib/svo/features/meshing/dual-contouring-mesh";
import { sparseSceneProxyVoxelizationShaderFor } from "../lib/core/webgpu-sparse-scene-proxies";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("GPU Hermite DC reconstructs closed spheres, sharp rotated boxes and bounded wavy terrain",async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","hermite-dc");let device:GPUDevice|undefined;
 try{
  device=(await createDawnRenderDevice()).device;
  for(const mode of ["dense","occupancy","banded"] as const){
   const m: GPUShaderModule=device.createShaderModule({code:sparseSceneProxyVoxelizationShaderFor("dry","f16-unorm8",mode,undefined,undefined,false,128)});
   assert.deepEqual((await m.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   await device.createComputePipelineAsync({layout:"auto",compute:{module:m,entryPoint:"rebuildDirtyBrickPayload"}});
  }
  const code=`
  @group(0) @binding(0) var<storage,read_write> vertices:array<vec4u>;
  @group(0) @binding(1) var<storage,read_write> meshState:array<atomic<u32>>;
  @group(0) @binding(2) var<storage,read_write> meshMaintenance:array<u32>;
  @group(0) @binding(3) var<storage,read_write> triangles:array<vec4u>;
  @group(0) @binding(4) var<uniform> shape:vec4f;
  const SVO_INVALID=0xffffffffu;
  fn dcField(p:vec3f,dirty:u32,count:u32)->f32{
   var q=p-vec3f(8.13,8.27,8.19);
   if(shape.x==0.){return length(q)-4.7;}
   if(shape.x==2.){let d=abs(q)-vec3f(4.8,2.8,4.8);let box=length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);
    return max(box,q.y-.4*sin(.6*q.x)*cos(.5*q.z));}
   q=vec3f(.8660254*q.x+.5*q.z,q.y,-.5*q.x+.8660254*q.z);
   let d=abs(q)-vec3f(3.3,2.7,3.1);return length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);
  }
  ${svoDualContouringFitWGSL}
  @compute @workgroup_size(64) fn fit(@builtin(global_invocation_id) id:vec3u){
   let i=id.x;if(i>=4096u){return;}let base=vec3f(f32(i%16u),f32((i/16u)%16u),f32(i/256u));
   let fitted=dcFit(base,vec3f(1),0u,0u);let p=base+fitted.point;
   vertices[i]=vec4u(bitcast<vec3u>(p),fitted.signs|(fitted.sharp<<24u));
   for(var j=0u;j<4u;j+=1u){meshMaintenance[1u+4u*i+j]=vertices[i][j];}
  }
  struct MeshRegion{size:f32,voxel:u32,identity:u32}
  fn meshRegionAt(p:vec3f)->MeshRegion{
   if(any(p<vec3f(0))||any(p>=vec3f(16))){return MeshRegion(1.,SVO_INVALID,0u);}
   let q=vec3u(floor(p));return MeshRegion(1.,q.x+16u*q.y+256u*q.z,1u);
  }
  fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return 0u;}
  fn meshAppend(base:vec3u,extent:vec3u,face:u32,identity:u32){let i=atomicAdd(&meshState[0],1u);triangles[2u*i]=vec4u(base,face);triangles[2u*i+1u]=vec4u(extent,identity);}
  ${svoDualContouringMeshWGSL}
  @compute @workgroup_size(1) fn emit(@builtin(global_invocation_id) id:vec3u){meshDcExtract(vec3u(0),1u,0u,id.x,16u);}
  `;
  const module=device.createShaderModule({code});const fit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"fit"}});
  const emit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"emit"}});
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const vertices=device.createBuffer({size:4096*16,usage}),state=device.createBuffer({size:256,usage}),maintenance=device.createBuffer({size:4096*16+4,usage}),triangles=device.createBuffer({size:20000*32,usage}),shape=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const buffers=[vertices,state,maintenance,triangles,shape];
  const group=(p:GPUComputePipeline,ids:number[])=>device!.createBindGroup({layout:p.getBindGroupLayout(0),entries:ids.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
  const fg=group(fit,[0,2,4]),eg=group(emit,[1,2,3]);
  const read=device.createBuffer({size:256+20000*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  for(const kind of [0,1,2]){
   device.queue.writeBuffer(shape,0,new Float32Array([kind,0,0,0]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([1]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups(48);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];assert.ok(count>100&&count<20000);
   const edges=new Map<string,number>();let error=0,sharp=0,volume=0;
   for(let i=0;i<count;i++){
    const at=64+i*8,base=Array.from(data.slice(at,at+3));sharp+=Number((data[at+3]&0x20000000)!==0);
    const pts=[0,1,2].map(j=>[0,10,20].map((shift,a)=>base[a]+((data[at+4+j]>>>shift)&1023)/256));
    for(const pos of pts){let q=pos.map((v,a)=>v-[8.13,8.27,8.19][a]);let d=0;
     if(kind===0)d=Math.hypot(...q)-4.7;
     else if(kind===2){const r=q.map((v,a)=>Math.abs(v)-[4.8,2.8,4.8][a]);const box=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);d=Math.max(box,q[1]-.4*Math.sin(.6*q[0])*Math.cos(.5*q[2]));}
     else{q=[.8660254*q[0]+.5*q[2],q[1],-.5*q[0]+.8660254*q[2]];const r=q.map((v,a)=>Math.abs(v)-[3.3,2.7,3.1][a]);d=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);}
     error=Math.max(error,Math.abs(d));
    }
    for(let j=0;j<3;j++){const key=[pts[j].join(","),pts[(j+1)%3].join(",")].sort().join("|");edges.set(key,(edges.get(key)??0)+1);}
    const [a,b,c]=pts;volume+=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
   }
   console.log(JSON.stringify({kind,triangles:count,maxVertexErrorCells:error,sharpTriangles:sharp,volume:volume/6}));
   assert.ok([...edges.values()].every(v=>v===2),"shared vertices produce a closed two-owner-edge mesh");assert.ok(volume>0,"outward winding");
   assert.ok(error<(kind===1?.02:.15),"analytic surface/edge fidelity");if(kind===1)assert.ok(sharp>0,"crease metadata survives meshing");
  }
  read.destroy();for(const b of buffers)b.destroy();
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
