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
import { HERO_GARDEN_BACKDROP } from "../lib/core/hero-garden-scene";
import { backdropFieldHeight, compileBackdropField } from "../lib/svo/features/backdrop/backdrop-field";
import { backdropDetailCentreLattice, backdropDetailFromPlan, backdropDetailVoxelizerWGSL } from "../lib/svo/features/backdrop/backdrop-detail";
import { backdropTerrainWGSL, packBackdropTerrainTable, planBackdropTiles } from "../lib/svo/features/backdrop/backdrop-terrain-tiles";

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
  // A backdrop on the set's lattice: two stored rings, then the walked far ground.
  const cell0=.0125;
  const field=compileBackdropField(HERO_GARDEN_BACKDROP,{height_m:.1,footprint_m:[-.9,-.6,.9,.6]});
  const plan=planBackdropTiles(field,{origin_m:[-.9,0,-.6],cellSize_m:cell0,firstLevel:2,centreLattice_m:backdropDetailCentreLattice(cell0,2)});
  const detail=backdropDetailFromPlan(field,plan,2);
  const terrainTable=packBackdropTerrainTable(field,3,plan);
  const backdropWGSL=backdropDetailVoxelizerWGSL(terrainTable,detail,true);
  // The fixture window sits on ring 0's corner: the octants with x or z past
  // it are ring 1's, one level coarser.
  const corner=[plan.centre[0]+plan.halfWidth0_m,plan.centre[1]+plan.halfWidth0_m];
  const window=[corner[0]-8*cell0,Math.round(backdropFieldHeight(field,corner[0],corner[1])/cell0)*cell0-8*cell0,corner[1]-8*cell0];
  // A second window straddles the set footprint's -z edge, where the field is
  // clipped to the outside half-space.
  const edge=[.3-8*cell0,Math.round(backdropFieldHeight(field,.3,-.6-4*cell0)/cell0)*cell0-8*cell0,-.6-8.37*cell0];
  {
   const m: GPUShaderModule=device.createShaderModule({code:sparseSceneProxyVoxelizationShaderFor("dry","f16-unorm8","dense",undefined,
    {baseWords:64,heightsBaseWords:128,width:16,depth:16,patchBaseWords:512,patchCapacity:4},false,1024,true,backdropWGSL)});
   {const errors=(await m.getCompilationInfo()).messages.filter(m=>m.type==="error");assert.deepEqual(errors.map(m=>`${m.lineNum}:${m.linePos} ${m.message}`),[],"the fit compiles with the continuous backdrop field");}
   await device.createComputePipelineAsync({layout:"auto",compute:{module:m,entryPoint:"rebuildDirtyBrickPayload"}});
  }
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
  struct SolidWorldSample{fraction:f32,distance:f32,material:u32,normal:vec3f}
  ${backdropWGSL}
  fn dcField(p:vec3f,dirty:u32,count:u32)->f32{
   // The production backdrop field, in fixture cells.
   if(shape.x==8.){return backdropDetailField(vec3f(${window.join(",")})+p*${cell0},${.25*cell0})/${cell0};}
   // Unioned, as the voxelizer does, with a set slab that ends at the edge.
   if(shape.x==9.){let w=vec3f(${edge.join(",")})+p*${cell0};
    return min(max(w.y-${edge[1]+7.6*cell0},-.6-w.z),backdropDetailField(w,${.25*cell0}))/${cell0};}
   var q=p-vec3f(8.13,8.27,8.19);
   if(shape.x==0.){return length(q)-4.7;}
   // Small enough that the 4-wide leaves at the world's edge stay outside.
   if(shape.x==6.){return length(q)-3.;}
   if(shape.x==7.){q=vec3f(.8660254*q.x+.5*q.z,q.y,-.5*q.x+.8660254*q.z);let d=abs(q)-vec3f(2.3,1.9,2.1);return length(max(d,vec3f(0)))+min(max(d.x,max(d.y,d.z)),0.);}
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
  // Mixed fixtures: shape.z is a mask of the 8-cell octants stored as one brick
  // of cells shape.w wide; their payload follows the fine lattice's.
  fn coarseOctant(q:vec3u)->u32{let o=q/8u;return o.x|(o.y<<1u)|(o.z<<2u);}
  fn coarseStored(o:u32)->bool{return ((u32(shape.z)>>o)&1u)!=0u;}
  @compute @workgroup_size(64) fn fitCoarse(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
   dmcCacheSamples(vec3f(0),vec3f(1),0u,0u,lane,false);
   let S=u32(shape.w);if(S<2u){return;}let c=8u/S;let o=id.x/64u;let k=id.x%64u;if(o>=8u||k>=c*c*c){return;}
   let base=vec3f(vec3u(o&1u,(o>>1u)&1u,o>>2u)*8u+vec3u(k%c,(k/c)%c,k/(c*c))*S);
   let fitted=dmcFit(base,vec3f(f32(S)),0u,0u);let p=base+fitted.point*f32(S);let i=4096u+o*512u+k;
   vertices[i]=vec4u(bitcast<vec3u>(p),bitcast<u32>(fitted.value));
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
   if(any(p<vec3f(0))||any(p>=vec3f(16))){return MeshRegion(48.,SVO_INVALID,0u);}
   let q=vec3u(floor(p));let o=coarseOctant(q);
   if(coarseStored(o)){let S=u32(shape.w);let c=8u/S;let cell=(q%8u)/S;return MeshRegion(f32(S),4096u+o*512u+cell.x+c*cell.y+c*c*cell.z,1u);}
   return MeshRegion(1.,payloadIndex(q),1u);
  }
  // The fixture's depth lane carries the owner's cell width.
  fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return depth<<6u;}
  fn meshDmcUnsupported(){atomicAdd(&meshState[2],1u);}
  fn meshDmcPackNormal(normal:vec3f)->u32{return 0u;}
  fn meshAppend(base:vec3u,extent:vec3u,face:u32,identity:u32){let i=atomicAdd(&meshState[0],1u);triangles[2u*i]=vec4u(base,face);triangles[2u*i+1u]=vec4u(extent,identity);}
  ${svoDualContouringMeshWGSL}
  ${svoDualMarchingCubesMeshWGSL.replace("if(count==4u){", "if(count==4u){atomicAdd(&meshState[1],1u);")}
  @compute @workgroup_size(1) fn emit(@builtin(global_invocation_id) id:vec3u){let n=u32(shape.y);let b=id.x/n;let side=16u/n;if(b>=side*side*side){return;}
   let brick=vec3u(b%side,(b/side)%side,b/(side*side))*n;
   if(coarseStored(coarseOctant(brick))){if(all(brick%8u==vec3u(0))){meshDmcExtract(brick,u32(shape.w),u32(shape.w),id.x%n,n);}return;}
   meshDmcExtract(brick,1u,1u,id.x%n,n);}
  `;
  const module=device.createShaderModule({code});const fit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"fit"}});
  const emit=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"emit"}});
  const fitCoarse=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"fitCoarse"}});
  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC;
  const vertices=device.createBuffer({size:8192*16,usage}),state=device.createBuffer({size:256,usage}),maintenance=device.createBuffer({size:8192*16+4,usage}),triangles=device.createBuffer({size:20000*32,usage}),shape=device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const buffers=[vertices,state,maintenance,triangles,shape];
  const group=(p:GPUComputePipeline,ids:number[])=>device!.createBindGroup({layout:p.getBindGroupLayout(0),entries:ids.map(binding=>({binding,resource:{buffer:buffers[binding]}}))});
  const fg=group(fit,[0,2,4]),eg=group(emit,[0,1,2,3,4]),cg=group(fitCoarse,[0,2,4]);
  // One triangle record's three vertices, in fine cells: 12 bits per axis in
  // 1/256 of the owner's cell, the high two bits of each in the face word.
  const points=(data:Uint32Array,at:number)=>{const face=data[at+3],scale=(face>>>6)&31;
   return [0,1,2].map(j=>{const high=(face>>>(11+6*j))&63;
    return [0,1,2].map(a=>(data[at+a]&0xffff)+((((data[at+4+j]>>>(10*a))&1023)|(((high>>>(2*a))&3)<<10))/256)*scale);});};
  const read=device.createBuffer({size:256+20000*32,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
  for(const n of [4,8,16]) for(const kind of [0,1,2,3,4,5]){
   device.queue.writeBuffer(shape,0,new Float32Array([kind,n,0,0]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([0x80000001]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups((16/n)**3*n);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];assert.ok(count>100&&count<20000);
   const edges=new Map<string,number>();const winding=new Map<string,number>();let error=0,sharp=0,volume=0,smoothTop=0,sharpTop=0;
   for(let i=0;i<count;i++){
    const at=64+i*8;sharp+=Number((data[at+3]&0x20000000)!==0);
    const pts=points(data,at);
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
  // Refinement boundaries: octants stored at a coarser cell beside fine
  // ones, across faces, edges and corners, at 2:1 and 4:1.
  for(const [n,coarse,kinds] of [[4,2,[0,1,2]],[2,4,[6,7]]] as const) for(const mask of [0xaa,0xfe,0x96,0x01]) for(const kind of kinds){
   device.queue.writeBuffer(shape,0,new Float32Array([kind,n,mask,coarse]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([0x80000001]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();
   p=e.beginComputePass();p.setPipeline(fitCoarse);p.setBindGroup(0,cg);p.dispatchWorkgroups(8);p.end();
   p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups((16/n)**3*n);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];assert.ok(count>50&&count<20000);
   const edges=new Map<string,number>();const winding=new Map<string,number>();let error=0,volume=0,wide=0,straddling=0,coarseOwned=0;
   const stored=(pos:number[])=>{const o=pos.map(v=>Math.min(1,Math.floor(v/8)));return (mask>>(o[0]|(o[1]<<1)|(o[2]<<2)))&1;};
   for(let i=0;i<count;i++){
    const at=64+i*8,pts=points(data,at);wide+=Number((data[at+3]&0x1ffff800)!==0);
    straddling+=Number(new Set(pts.map(stored)).size===2);
    if(((data[at+3]>>>6)&31)>1)coarseOwned++;
    for(const pos of pts){let q=pos.map((v,a)=>v-[8.13,8.27,8.19][a]);let d=0;
     if(kind===0||kind===6)d=Math.hypot(...q)-(kind===0?4.7:3);
     else if(kind===2){const r=q.map((v,a)=>Math.abs(v)-[4.8,2.8,4.8][a]);const box=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);d=Math.max(box,q[1]-.4*Math.sin(.6*q[0])*Math.cos(.5*q[2]));}
     else{q=[.8660254*q[0]+.5*q[2],q[1],-.5*q[0]+.8660254*q[2]];const r=q.map((v,a)=>Math.abs(v)-(kind===1?[3.3,2.7,3.1]:[2.3,1.9,2.1])[a]);d=Math.hypot(...r.map(v=>Math.max(v,0)))+Math.min(Math.max(...r),0);}
     error=Math.max(error,Math.abs(d));
    }
    for(let j=0;j<3;j++){const key=[pts[j].join(","),pts[(j+1)%3].join(",")].sort().join("|");edges.set(key,(edges.get(key)??0)+1);winding.set(key,(winding.get(key)??0)+(pts[j].join(",")<pts[(j+1)%3].join(",")?1:-1));}
    const [a,b,c]=pts;volume+=a[0]*(b[1]*c[2]-b[2]*c[1])+a[1]*(b[2]*c[0]-b[0]*c[2])+a[2]*(b[0]*c[1]-b[1]*c[0]);
   }
   const open=[...edges.values()].filter(v=>v!==2).length;
   console.log(JSON.stringify({mixed:{n,coarse,mask},kind,triangles:count,wideTriangles:wide,coarseOwned,unsupported:data[2],straddlingTriangles:straddling,openEdges:open,maxVertexErrorCells:error,volume:volume/6}));
   assert.equal(data[2],0,"every dual cell fits the 12-bit triangle record");
   assert.equal(open,0,"dual cells spanning two cell sizes close the mesh across the refinement boundary");
   assert.ok([...winding.values()].every(v=>v===0),"mixed cells keep consistent directed winding");
   assert.ok(volume>0,"outward winding");assert.ok(straddling>0,"triangles span the refinement boundary");
  }
  // The backdrop's own field across a ring corner: ground and scatter fitted
  // at two cell sizes. The window's edge is open; everything inside closes.
  {
   const mask=0xfa;
   device.queue.writeBuffer(shape,0,new Float32Array([8,4,mask,2]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([0x80000001]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();
   p=e.beginComputePass();p.setPipeline(fitCoarse);p.setBindGroup(0,cg);p.dispatchWorkgroups(8);p.end();
   p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups(64*4);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];
   const edges=new Map<string,number>();const winding=new Map<string,number>();let below=0,onGround=0,straddling=0,vertices=0,up=0;
   const inside=(pos:number[])=>pos[0]>2.5&&pos[0]<13.5&&pos[2]>2.5&&pos[2]<13.5;
   const coarse=(pos:number[])=>pos[0]>=8||pos[2]>=8;
   for(let i=0;i<count;i++){
    const at=64+i*8,pts=points(data,at);
    straddling+=Number(new Set(pts.map(coarse)).size===2);
    for(const pos of pts){
     const clearance=(window[1]+pos[1]*cell0-backdropFieldHeight(field,window[0]+pos[0]*cell0,window[2]+pos[2]*cell0))/cell0;
     below=Math.min(below,clearance);vertices++;onGround+=Number(Math.abs(clearance)<.15);
    }
    const [a,b,c]=pts;up+=Number((b[2]-a[2])*(c[0]-a[0])-(b[0]-a[0])*(c[2]-a[2])>0);
    for(let j=0;j<3;j++){if(!inside(pts[j])||!inside(pts[(j+1)%3]))continue;
     const key=[pts[j].join(","),pts[(j+1)%3].join(",")].sort().join("|");edges.set(key,(edges.get(key)??0)+1);winding.set(key,(winding.get(key)??0)+(pts[j].join(",")<pts[(j+1)%3].join(",")?1:-1));}
   }
   const open=[...edges.values()].filter(v=>v!==2).length;
   console.log(JSON.stringify({backdropRingCorner:{triangles:count,unsupported:data[2],straddlingTriangles:straddling,interiorEdges:edges.size,openInteriorEdges:open,lowestVertexCells:below,groundVertexShare:onGround/vertices,upwardShare:up/count}}));
   assert.ok(count>100,"the ring corner is meshed");assert.equal(data[2],0,"ring grading fits the triangle record");
   assert.ok(straddling>0,"triangles span the ring boundary");
   assert.equal(open,0,"the backdrop mesh is closed across a ring corner");
   assert.ok([...winding.values()].every(v=>v===0),"ring-corner triangles keep consistent directed winding");
   assert.ok(below>-.3,"no vertex sinks below the continuous ground");
   assert.ok(onGround/vertices>.5,"the ground is the fitted height field, not its voxel columns");
  }
  // The footprint edge: the clip plane is a wall only below the ground. (This
  // fit alone does not reproduce the upright plates the x10 frame showed with
  // the clip's early return one overlap inside; it pins the edge's shape.)
  {const kind=9;
   device.queue.writeBuffer(shape,0,new Float32Array([kind,4,0,1]));device.queue.writeBuffer(state,0,new Uint32Array(64));device.queue.writeBuffer(state,47*4,new Uint32Array([0x80000001]));
   const e=device.createCommandEncoder();let p=e.beginComputePass();p.setPipeline(fit);p.setBindGroup(0,fg);p.dispatchWorkgroups(64);p.end();
   p=e.beginComputePass();p.setPipeline(emit);p.setBindGroup(0,eg);p.dispatchWorkgroups(64*4);p.end();
   e.copyBufferToBuffer(state,0,read,0,256);e.copyBufferToBuffer(triangles,0,read,256,20000*32);device.queue.submit([e.finish()]);
   await read.mapAsync(GPUMapMode.READ);const data=new Uint32Array(read.getMappedRange().slice(0));read.unmap();const count=data[0];
   let above=0,outsideGround=0;
   for(let i=0;i<count;i++) for(const pos of points(data,64+i*8)){
    const x=edge[0]+pos[0]*cell0,z=edge[2]+pos[2]*cell0;
    const clearance=(edge[1]+pos[1]*cell0-backdropFieldHeight(field,x,z))/cell0;
    // Scatter never reaches the footprint, so anything raised here is the clip.
    if(z>-.6-cell0)above=Math.max(above,clearance);outsideGround+=Number(z<-.6&&Math.abs(clearance)<.15);
   }
   console.log(JSON.stringify({backdropFootprintEdge:{kind,triangles:count,unsupported:data[2],highestVertexCells:above,outsideGroundVertices:outsideGround}}));
   assert.equal(data[2],0);
   if(kind===9)assert.ok(outsideGround>50,"the ground outside the footprint is meshed");
   assert.ok(above<(kind===9?.3:1),"nothing stands above the ground on the footprint's clip plane");
  }
  // The far ground as a continuous height field: every hit lies on h, and no
  // ray passes more than half a column under the ground before it.
  {
   const outer=plan.halfWidth0_m*2**(plan.levels-1),stored=plan.halfWidth0_m*2;
   const walked=(x:number,z:number)=>{const m=Math.max(Math.abs(x-plan.centre[0]),Math.abs(z-plan.centre[1]));return m>stored&&m<outer;};
   const list:number[][]=[];
   for(let a=0;a<24;a++) for(const pitch of [-.5,-.2,-.08,-.03,-.01,.01]){
    const azimuth=a*Math.PI/12+.1,c=Math.cos(pitch);
    list.push([.3,1.1,-.2],[c*Math.cos(azimuth),Math.sin(pitch),c*Math.sin(azimuth)]);
   }
   // Shadow rays: from the ground toward a low sun.
   for(let a=0;a<24;a++){
    const x=plan.centre[0]+(6+2*a)*Math.cos(a),z=plan.centre[1]+(6+2*a)*Math.sin(a);
    list.push([x,backdropFieldHeight(field,x,z)+.02,z],[Math.cos(.2)*.6,Math.sin(.2),Math.cos(.2)*.8]);
   }
   const count=list.length/2;
   const rayData=new Float32Array(count*8);list.forEach((v,i)=>rayData.set(v,i*4));
   const rays=device.createBuffer({size:rayData.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   const hits=device.createBuffer({size:count*16,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const hitRead=device.createBuffer({size:count*16,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   device.queue.writeBuffer(rays,0,rayData);
   const table=device.createBuffer({size:terrainTable.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
   device.queue.writeBuffer(table,0,terrainTable);
   const trace=await device.createComputePipelineAsync({layout:"auto",compute:{module:device.createShaderModule({code:`
  @group(0) @binding(0) var<storage,read> table:array<u32>;
  @group(0) @binding(1) var<storage,read> rays:array<vec4f>;
  @group(0) @binding(2) var<storage,read_write> hits:array<vec4f>;
  ${backdropTerrainWGSL({load:(index)=>`table[${index}]`,tableBase:"0u"})}
  @compute @workgroup_size(64) fn trace(@builtin(global_invocation_id) id:vec3u){
   if(id.x>=arrayLength(&hits)){return;}
   let ro=rays[2u*id.x].xyz;let rd=rays[2u*id.x+1u].xyz;
   let smoothHit=backdropTerrainSmoothTrace(ro,rd,0.,1e4);let voxelHit=backdropTerrainTrace(ro,rd,0.,1e4);
   hits[id.x]=vec4f(smoothHit.t,voxelHit.t,f32(smoothHit.exhausted),f32(smoothHit.tiles+smoothHit.steps));
  }`}),entryPoint:"trace"}});
   const tg=device.createBindGroup({layout:trace.getBindGroupLayout(0),entries:[table,rays,hits].map((buffer,binding)=>({binding,resource:{buffer}}))});
   const e=device.createCommandEncoder();const p=e.beginComputePass();p.setPipeline(trace);p.setBindGroup(0,tg);p.dispatchWorkgroups(Math.ceil(count/64));p.end();
   e.copyBufferToBuffer(hits,0,hitRead,0,count*16);device.queue.submit([e.finish()]);
   await hitRead.mapAsync(GPUMapMode.READ);const result=new Float32Array(hitRead.getMappedRange().slice(0));hitRead.unmap();
   let hit=0,miss=0,offSurface=0,deepest=0,work=0;
   for(let i=0;i<count;i++){
    const ro=list[2*i],rd=list[2*i+1],t=result[4*i];
    assert.equal(result[4*i+2],0,`ray ${i} stays within the walk's budget`);work=Math.max(work,result[4*i+3]);
    // A ray that meets the ground inside the stored rings is the mesh's: it
    // reaches the walked ground already under it.
    let meshed=false;
    for(let s=0;s<20&&!meshed;s+=.004){const x=ro[0]+rd[0]*s,z=ro[2]+rd[2]*s;if(walked(x,z))break;meshed=ro[1]+rd[1]*s<backdropFieldHeight(field,x,z);}
    if(meshed){continue;}
    if(t>=0){
     hit++;const x=ro[0]+rd[0]*t,z=ro[2]+rd[2]*t;
     const off=Math.abs(ro[1]+rd[1]*t-backdropFieldHeight(field,x,z))/(1+t);
     offSurface=Math.max(offSurface,off);
    }else miss++;
    // Dense march up to the reported hit: depth under the ground in local columns.
    for(let s=0;s<(t>=0?t:400);s+=Math.max(.004,.002*s)){
     const x=ro[0]+rd[0]*s,y=ro[1]+rd[1]*s,z=ro[2]+rd[2]*s;if(!walked(x,z))continue;
     const column=cell0*2**Math.ceil(Math.log2(Math.max(Math.abs(x-plan.centre[0]),Math.abs(z-plan.centre[1]))/plan.halfWidth0_m));
     if(s<t-2*column||t<0)deepest=Math.max(deepest,(backdropFieldHeight(field,x,z)-y)/column);
    }
   }
   console.log(JSON.stringify({smoothGroundWalk:{rays:count,hit,miss,offSurfacePerMetre:offSurface,deepestPassColumns:deepest,mostWork:work}}));
   assert.ok(hit>30&&miss>10,"the fan both meets and clears the ground");
   assert.ok(offSurface<2e-3,"hits lie on the continuous ground");
   assert.ok(deepest<.5,"no ray passes under the ground before its hit");
   rays.destroy();hits.destroy();hitRead.destroy();table.destroy();
  }
  read.destroy();for(const b of buffers)b.destroy();
 }finally{device?.destroy();await releaseWebGPUExclusiveLock();}
});
