import { selectSvoBrickOccupancyGpu } from "../lib/svo/features/primary-visibility/webgpu-svo-brick-selection";
import { buildSvoScenePrimitives } from "../lib/svo/features/scene-publication/svo-scene-primitives";
import { getScenePreset } from "../lib/core/scenes";
import assert from "node:assert/strict";
import test from "node:test";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createDawnRenderDevice } from "../tools/svo-dry-frame-harness";
import { svoDualMarchingCubesCachedFitWGSL } from "../lib/svo/features/meshing/dual-marching-cubes";
import { svoDualMarchingCubesMeshWGSL } from "../lib/svo/features/meshing/dual-marching-cubes-mesh";
import { svoDualContouringMeshWGSL } from "../lib/svo/features/meshing/dual-contouring-mesh";
import { sparseSceneProxyVoxelizationShaderFor } from "../lib/core/webgpu-sparse-scene-proxies";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("GPU Dual Marching Cubes reconstructs closed surfaces and a wall missed by primal corner signs",async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","dual-marching-cubes");let device:GPUDevice|undefined;
 try{
  device=(await createDawnRenderDevice()).device;
  const built=buildSvoScenePrimitives(getScenePreset("garden-svo-lighting").create());
  const bounds={minimum:[4.1,4.1,4.1] as const,maximum:[7.9,7.9,7.9] as const};
  const selection={regions:[bounds],retainedRegions:[bounds],worldOrigin:[0,0,0] as const,cellSize:[1,1,1] as const,
    brickSize:4,brickDimensions:[4,4,4] as const,maximumDepth:2};
  const sparseBuild={...built,metadata:[]};
  const exact=await selectSvoBrickOccupancyGpu(device,sparseBuild,selection);
  const halo=await selectSvoBrickOccupancyGpu(device,sparseBuild,{...selection,haloCells:1});
  assert.equal([...exact.keys(2)].length,1);
  assert.equal([...halo.keys(2)].length,27,"positive dual samples retain face, edge and corner neighbour bricks");
  for(const mode of ["dense","occupancy","banded"] as const){
   for(const terrain of [undefined,{baseWords:64,heightsBaseWords:128,width:16,depth:16,patchBaseWords:512,patchCapacity:4}]){
   const m: GPUShaderModule=device.createShaderModule({code:sparseSceneProxyVoxelizationShaderFor("dry","f16-unorm8",mode,undefined,terrain,false,1024,true)});
   assert.deepEqual((await m.getCompilationInfo()).messages.filter(m=>m.type==="error"),[]);
   await device.createComputePipelineAsync({layout:"auto",compute:{module:m,entryPoint:"rebuildDirtyBrickPayload"}});
   }
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
   if(shape.x==3.){let d=abs(q)-vec3f(.08,3.1,3.4);return length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);}
   if(shape.x==2.||shape.x==5.){let d=abs(q)-vec3f(4.8,2.8,4.8);let box=length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);
    return max(box,q.y-select(.4,.004,shape.x==5.)*sin(.6*q.x)*cos(.5*q.z));}
   q=vec3f(.8660254*q.x+.5*q.z,q.y,-.5*q.x+.8660254*q.z);
   let d=abs(q)-vec3f(3.3,2.7,3.1);return length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);
  }
  ${svoDualMarchingCubesCachedFitWGSL}
  @compute @workgroup_size(64) fn fit(@builtin(global_invocation_id) id:vec3u){
   let tile=id.x/64u;let lane=id.x%64u;
   let tileBase=vec3u(tile%4u,(tile/4u)%4u,tile/16u)*4u;
   let cell=tileBase+vec3u(lane%4u,(lane/4u)%4u,lane/16u);
   let i=payloadIndex(cell);let base=vec3f(cell);
   dmcCacheSamples(vec3f(tileBase),vec3f(1),0u,0u,lane,true);
   let fitted=dmcFit(base,vec3f(1),0u,0u);var p=base+fitted.point;var value=fitted.value;
   // Direct dual-grid checkerboard stresses shared-face ambiguity independently of fitting.
   if(shape.x==4.){p=base+vec3f(.5);value=1.;if(all(base>=vec3f(6))&&all(base<=vec3f(9))&&((u32(base.x+base.y+base.z)&1u)==0u)){value=-1.;}}
   vertices[i]=vec4u(bitcast<vec3u>(p),bitcast<u32>(value));
   for(var j=0u;j<4u;j+=1u){meshMaintenance[1u+4u*i+j]=vertices[i][j];}
  }
  fn payloadIndex(q:vec3u)->u32{
   let n=u32(shape.y);let b=q/n;let local=q%n;let side=16u/n;
   return (b.x+side*b.y+side*side*b.z)*n*n*n+local.x+n*local.y+n*n*local.z;
  }
  fn sceneIdentityAt(voxel:u32)->u32{return voxel+1u;}
  fn sceneIdentitySolid(identity:u32)->bool{return identity!=0u;}
  fn sceneIdentityHasNormal(identity:u32)->bool{return identity!=0u;}
  fn sceneIdentityNormal(identity:u32)->vec3f{
   let point=bitcast<vec3f>(vertices[identity-1u].xyz);
   let g=dmcGradient(point,vec3f(1),0u,0u);
   return normalize(g+vec3f(1e-20));
  }
  struct MeshRegion{size:f32,voxel:u32,identity:u32}
  fn meshRegionAt(p:vec3f)->MeshRegion{
   if(any(p<vec3f(0))||any(p>=vec3f(16))){return MeshRegion(1.,SVO_INVALID,0u);}
   let q=vec3u(floor(p));return MeshRegion(1.,payloadIndex(q),1u);
  }
  fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return 0u;}
  fn meshAppend(base:vec3u,extent:vec3u,face:u32,identity:u32){let i=atomicAdd(&meshState[0],1u);triangles[2u*i]=vec4u(base,face);triangles[2u*i+1u]=vec4u(extent,identity);}
  ${svoDualContouringMeshWGSL}
  ${svoDualMarchingCubesMeshWGSL.replace("if(count==4u){", "if(count==4u){atomicAdd(&meshState[1],1u);")}
  @compute @workgroup_size(1) fn emit(@builtin(global_invocation_id) id:vec3u){let n=u32(shape.y);let b=id.x/n;let side=16u/n;if(b>=side*side*side){return;}meshDmcExtract(vec3u(b%side,(b/side)%side,b/(side*side))*n,1u,0u,id.x%n,n);}
  `;
  const module=device.createShaderModule({code});const fit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"fit"}});
  const emit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"emit"}});
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const vertices=device.createBuffer({size:4096*16,usage}),state=device.createBuffer({size:256,usage}),maintenance=device.createBuffer({size:4096*16+4,usage}),triangles=device.createBuffer({size:20000*32,usage}),shape=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const buffers=[vertices,state,maintenance,triangles,shape];
  const group=(p:GPUComputePipeline,ids:number[])=>device!.createBindGroup({layout:p.getBindGroupLayout(0),entries:ids.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
  const fg=group(fit,[0,2,4]),eg=group(emit,[0,1,2,3,4]);
  const read=device.createBuffer({size:256+20000*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  for(const n of [8,16]) for(const kind of [0,1,2,3,4,5]){
   device.queue.writeBuffer(shape,0,new Float32Array([kind,n,0,0]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([0x80000001]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups(64);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];assert.ok(count>100&&count<20000);
   const edges=new Map<string,number>();const winding=new Map<string,number>();let error=0,sharp=0,volume=0,smoothTop=0,sharpTop=0;
   for(let i=0;i<count;i++){
    const at=64+i*8,base=Array.from(data.slice(at,at+3));sharp+=Number((data[at+3]&0x20000000)!==0);
    const pts=[0,1,2].map(j=>[0,10,20].map((shift,a)=>base[a]+((data[at+4+j]>>>shift)&1023)/256));
    if(kind===5 && pts.every(p=>Math.abs(p[1]-8.27)<.05&&Math.abs(p[0]-8.13)<3&&Math.abs(p[2]-8.19)<3)){
      smoothTop++;sharpTop+=Number((data[at+3]&0x20000000)!==0);
    }
    for(const pos of pts){let q=pos.map((v,a)=>v-[8.13,8.27,8.19][a]);let d=0;
     if(kind===4)d=0;
     else if(kind===0)d=Math.hypot(...q)-4.7;
     else if(kind===3){const r=q.map((v,a)=>Math.abs(v)-[.08,3.1,3.4][a]);d=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);}
     else if(kind===2||kind===5){const r=q.map((v,a)=>Math.abs(v)-[4.8,2.8,4.8][a]);const box=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);d=Math.max(box,q[1]-(kind===5?.004:.4)*Math.sin(.6*q[0])*Math.cos(.5*q[2]));}
     else{q=[.8660254*q[0]+.5*q[2],q[1],-.5*q[0]+.8660254*q[2]];const r=q.map((v,a)=>Math.abs(v)-[3.3,2.7,3.1][a]);d=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);}
     error=Math.max(error,Math.abs(d));
    }
    for(let j=0;j<3;j++){const key=[pts[j].join(","),pts[(j+1)%3].join(",")].sort().join("|");edges.set(key,(edges.get(key)??0)+1);winding.set(key,(winding.get(key)??0)+(pts[j].join(",")<pts[(j+1)%3].join(",")?1:-1));}
    const [a,b,c]=pts;volume+=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
   }
   console.log(JSON.stringify({kind,triangles:count,maxVertexErrorCells:error,sharpTriangles:sharp,volume:volume/6}));
   assert.ok([...edges.values()].every(v=>v===2),"shared vertices produce a closed two-owner-edge mesh");assert.ok(volume>0,"outward winding");
   assert.ok([...winding.values()].every(n=>n===0),"adjacent triangles have consistent directed winding");
   if(kind===5){assert.ok(smoothTop>10,"gently curved snapped patch is exercised");assert.equal(sharpTop,0,"zero-snapped smooth patches must not switch to flat shading");}
   if(kind===0)assert.equal(sharp,0,"a smooth sphere must not be classified as creased");
   if(kind===1)assert.ok(sharp>0 && sharp<count,"box edges retain sharp shading while face interiors stay smooth");
   if(kind===4)assert.ok(data[1]>0,"checkerboard exercised ambiguous faces");
   assert.ok(error<.06,"analytic surface error stays below 0.06 cells");
   if(kind===3)assert.ok(volume/6>.16*6.2*6.8*.7,"sub-cell wall retains both sheets and most enclosed volume");
  }
  read.destroy();for(const b of buffers)b.destroy();
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
