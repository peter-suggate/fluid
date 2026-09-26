import assert from "node:assert/strict";
import test from "node:test";
import {pathToFileURL} from "node:url";
import {managedGPUDevice} from "../lib/core/gpu-compilation-manager";
import {requiredFluidDeviceLimits} from "../lib/core/webgpu-device-limits";
import {createProcessRetainedDawnGPU} from "../lib/harness/node-dawn-provider";
import {acquireWebGPUExclusiveLock,releaseWebGPUExclusiveLock} from "../lib/harness/webgpu-smoke-isolation";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {refinementRegionLattice} from "../lib/core/refinement-regions";
import {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import type {WebGPUUniformVelocityExtrapolator} from "../lib/methods/uniform/webgpu-uniform-velocity-extrapolation";
import {uniformGeometricSolverOptions} from "../lib/methods/uniform/uniform-geometric-options";
import {UniformMixedOwnership} from "../lib/methods/uniform/uniform-mixed-ownership";
import {UniformMixedExtension} from "../lib/methods/uniform/uniform-mixed-extension";
import {createUniformMixedLayout,mixedCellWidth,MIXED_CELL_MASK} from "../lib/methods/uniform/uniform-mixed-layout";
import {readMixedTexture} from "./helpers/uniform-mixed-native-fields";
const modulePath=process.env.WEBGPU_NODE_MODULE;
(modulePath?test:test.skip)("unified extension follows live ownership and retains supported MAC velocity",{timeout:240000},async()=>{
 await acquireWebGPUExclusiveLock("dawn-test","Uniform mixed extension");let device:GPUDevice|undefined,solver:WebGPUUniformReferenceSolver|undefined;
 const owned:(GPUTexture|GPUBuffer)[]=[];let ownership:UniformMixedOwnership|undefined;
 try{
  const dawn=await import(pathToFileURL(modulePath!).href);Object.assign(globalThis,dawn.globals);
  const gpu=createProcessRetainedDawnGPU(dawn,["backend=metal"]),adapter=await gpu.requestAdapter();assert.ok(adapter);
  device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
  const errors:string[]=[];device.addEventListener("uncapturederror",e=>{e.preventDefault();errors.push(e.error.message);});
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-64")));
  scene.container.width_m=scene.container.height_m=scene.container.depth_m=.8;scene.voxelDomain.finestCellSize_m=.025;scene.solidVoxels=[];scene.rigidBodies=[];
  scene.fluid.initialDamBreakDimensions_m={x:.2,y:.4,z:.4};
  solver=await WebGPUUniformReferenceSolver.createAsync(device,scene,"balanced",undefined,{...uniformGeometricSolverOptions({},scene),volumePages:16},()=>{});
  const internal=solver as unknown as {velocityExtrapolator:WebGPUUniformVelocityExtrapolator;scratchArena:{buffer:GPUBuffer};};
  const native=internal.velocityExtrapolator.prepareMixedContinuation();const lattice=refinementRegionLattice(scene),d=lattice.dimensions,n=d.reduce((a,b)=>a*b);
  ownership=new UniformMixedOwnership(device,createUniformMixedLayout(lattice,[]));
  const buffer=(size:number,uniform=false)=>{const b=device!.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC|(uniform?GPUBufferUsage.UNIFORM:GPUBufferUsage.STORAGE)});owned.push(b);return b;};
  const texture=(format:GPUTextureFormat)=>{const t=device!.createTexture({size:[...d],dimension:"3d",format,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC|GPUTextureUsage.COPY_DST});owned.push(t);return t;};
  const physical=texture("rgba32float"),phase=texture("r32float"),output=texture("rgba32float");
  const bytes=4*(d[0]*d[1]+d[0]*d[2]+d[1]*d[2]),negative=buffer(bytes),outputNegative=buffer(bytes),params=buffer(16,true);
  device.queue.writeBuffer(params,0,new Float32Array([...lattice.cellSize_m,0]));
  const borrowed:GPUDevice=new Proxy(device,{get(target,key){if(key==="createBuffer"||key==="createTexture")return()=>{throw new Error("extension allocated fields");};const value=Reflect.get(target,key,target);return typeof value==="function"?value.bind(target):value;}});
  const stage=new UniformMixedExtension(borrowed,ownership,native);await stage.initialize();assert.equal(stage.allocatedBytes,0);
  const groups=stage.bind({physical,phase,negative,output,outputNegative,params,scratch:{buffer:internal.scratchArena.buffer}});
  const coarse={id:"coarse",rule:"minimum-cell-size" as const,minimumCellSize_cells:4 as const,maximumCellSize_cells:4 as const,min_m:{x:0,y:0,z:-.4},max_m:{x:.4,y:.8,z:.4}};
  const layouts=[ownership.layout,createUniformMixedLayout(lattice,[coarse]),createUniformMixedLayout(lattice,[{...coarse,min_m:{x:-.4,y:0,z:-.4}}]),ownership.layout];
  for(const layout of layouts){
   ownership.update(layout);
   const cells:{min:number[];width:number}[]=new Array(layout.cellCount);
   layout.tiles.forEach((word,tile)=>{const width=mixedCellWidth(word),t=layout.tileDimensions,origin=[tile%t[0]*4,Math.floor(tile/t[0])%t[1]*4,Math.floor(tile/(t[0]*t[1]))*4];let i=word&MIXED_CELL_MASK;
    for(let z=0;z<4;z+=width)for(let y=0;y<4;y+=width)for(let x=0;x<4;x+=width)cells[i++]={min:[origin[0]!+x,origin[1]!+y,origin[2]!+z],width};});
   const voxelOwners=new Uint32Array(n);
   cells.forEach((c,i)=>{for(let z=0;z<c.width;z++)for(let y=0;y<c.width;y++)for(let x=0;x<c.width;x++)voxelOwners[c.min[0]!+x+d[0]*(c.min[1]!+y+d[1]*(c.min[2]!+z))]=i;});
   const velocity=new Float32Array(n*4).fill(NaN),support=new Float32Array(n).fill(NaN),boundary=new Float32Array(bytes/4).fill(NaN);
   const value=[.7,.3,-.2];const at=(p:readonly number[])=>p[0]!+d[0]*(p[1]!+d[1]*p[2]!);
   const faces:{anchor:number[];axis:number;source:boolean}[]=[];
   for(const c of cells){support[at(c.min)]=c.min[0]<8?1:0;for(let axis=0;axis<3;axis++){
    const u=(axis+1)%3,v=(axis+2)%3,plane=c.min[axis]!+c.width;
    const neighborIds=new Set<number>();
    if(plane<d[axis]!)for(let y=0;y<c.width;y++)for(let x=0;x<c.width;x++){const q=[...c.min];q[axis]=plane;q[u]!+=x;q[v]!+=y;neighborIds.add(voxelOwners[at(q)]!);}
    const neighbors=[...neighborIds].map(i=>cells[i]!);
    for(const o of neighbors.length?neighbors:[undefined]){
     const anchor=[...c.min];anchor[axis]=plane-1;if(o){anchor[u]=Math.max(c.min[u]!,o.min[u]!);anchor[v]=Math.max(c.min[v]!,o.min[v]!);}
     const source=c.min[0]<8||!!o&&o.min[0]<8;velocity[4*at(anchor)+axis]=source?value[axis]!:0;velocity[4*at(anchor)+3]=0;
     faces.push({anchor,axis,source});
    }
    if(c.min[axis]===0){const index=axis===0?c.min[1]+d[1]*c.min[2]:axis===1?d[1]*d[2]+c.min[0]+d[0]*c.min[2]:d[1]*d[2]+d[0]*d[2]+c.min[0]+d[0]*c.min[1];boundary[index]=value[axis]!;}
   }}
   device.queue.writeTexture({texture:physical},velocity,{bytesPerRow:d[0]*16,rowsPerImage:d[1]},[...d]);device.queue.writeTexture({texture:phase},support,{bytesPerRow:d[0]*4,rowsPerImage:d[1]},[...d]);device.queue.writeBuffer(negative,0,boundary);
   const e=device.createCommandEncoder();stage.encode(e,groups);device.queue.submit([e.finish()]);const actual=await readMixedTexture(device,output);assert.deepEqual(errors,[]);
   for(const face of faces){const observed=actual[4*at(face.anchor)+face.axis]!;assert.ok(Number.isFinite(observed),`invalid ${face.anchor}/${face.axis}`);if(face.source||face.axis===1)assert.ok(Math.abs(observed-value[face.axis]!)<2e-5,`supported/vertical extension ${observed} != ${value[face.axis]} at ${face.anchor}/${face.axis}`);
    else assert.ok(observed>=Math.min(0,value[face.axis]!)-2e-5&&observed<=Math.max(0,value[face.axis]!)+2e-5,"far extension introduced an extremum");}
   assert.deepEqual(errors,[]);
  }
 }finally{ownership?.destroy();owned.forEach(r=>r.destroy());solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();}
});
