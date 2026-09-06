import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { packFineLevelSetSample } from "../lib/core/fine-levelset-packed-sample";
import { globalFineSurfaceClassificationShader } from "../lib/core/webgpu-water-global-fine-classify";
import { GLOBAL_FINE_SURFACE_EMIT_LANES, globalFineClassifiedEmitShader,
  globalFineClassifiedIndirectScanShader } from "../lib/core/webgpu-water-global-fine-tetra";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { sceneAtContainerExtents } from "../lib/core/scene-scale";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import type { WebGPUFineLevelSetBrickSource } from "../lib/core/levelset-consumer-abi";
import { rasterMeshSymmetryMetrics } from "../lib/harness/raster-mesh-symmetry";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
const INVALID=0xffffffff;
const dimensions=[32,24,24] as const;
const sampleCount=32*24*24;
const cubeCapacity=sampleCount*6;
const vertexCapacity=cubeCapacity*12;
const generation=7;
// dawn-node's GPU owns the native instance. Large mesh readbacks trigger GC;
// devices alone do not retain that instance while map callbacks are pending.
const liveDawnInstances=new Set<GPU>();
type Point=readonly [number,number,number];
function buffer(device: GPUDevice, label: string, size: number, usage: GPUBufferUsageFlags,
  contents?: ArrayBufferView) {
  const result = device.createBuffer({ label, size: Math.max(4, Math.ceil(size / 4) * 4),
    usage, mappedAtCreation: contents !== undefined });
  if (contents) {
    const bytes = new Uint8Array(contents.buffer, contents.byteOffset, contents.byteLength);
    new Uint8Array(result.getMappedRange()).set(bytes); result.unmap();
  }
  return result;
}

async function read(device: GPUDevice, source: GPUBuffer, bytes: number) {
  const target = device.createBuffer({ label: "mixed raster readback", size: bytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, target, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await target.mapAsync(GPUMapMode.READ);
  const copy = target.getMappedRange().slice(0); target.unmap(); target.destroy();
  return copy;
}

/** Count connected components after welding exact emitted positions. */
function components(mesh: Float32Array, vertexCount: number): number {
  const ids = new Map<string, number>(); const parents: number[] = [];
  const root = (id: number): number => {
    while (parents[id] !== id) { parents[id] = parents[parents[id]!]!; id = parents[id]!; }
    return id;
  };
  for (let triangle = 0; triangle < vertexCount; triangle += 3) {
    const vertices: number[] = [];
    for (let corner = 0; corner < 3; corner++) {
      const at = 8 * (triangle + corner);
      const key = `${mesh[at]},${mesh[at + 1]},${mesh[at + 2]}`;
      let id = ids.get(key);
      if (id === undefined) { id = parents.length; parents.push(id); ids.set(key, id); }
      vertices.push(id);
    }
    parents[root(vertices[1]!)] = root(vertices[0]!);
    parents[root(vertices[2]!)] = root(vertices[0]!);
  }
  return new Set(parents.map((_, id) => root(id))).size;
}

function topology(mesh: Float32Array, vertexCount: number) {
  const vertices=new Set<string>();const edges=new Map<string,{count:number;orientation:number}>();
  let faces=0;
  for(let triangle=0;triangle<vertexCount;triangle+=3){
    const points=Array.from({length:3},(_,corner)=>{
      const at=8*(triangle+corner);return `${mesh[at]},${mesh[at+1]},${mesh[at+2]}`;
    });
    if(new Set(points).size<3)continue;
    faces++;points.forEach(point=>vertices.add(point));
    for(let corner=0;corner<3;corner++){
      const a=points[corner]!,b=points[(corner+1)%3]!;
      const key=a<b?`${a}|${b}`:`${b}|${a}`;
      const edge=edges.get(key)??{count:0,orientation:0};
      edge.count++;edge.orientation+=a<b?1:-1;edges.set(key,edge);
    }
  }
  return {euler:vertices.size-edges.size+faces,
    windingDefects:[...edges.values()].filter(edge=>edge.count===2&&edge.orientation!==0).length};
}

function assertSphereWinding(mesh: Float32Array) {
  for(let i=0;i<mesh.length;i+=24){
    const a=[mesh[i]!,mesh[i+1]!,mesh[i+2]!];
    const ab=[mesh[i+8]!-a[0]!,mesh[i+9]!-a[1]!,mesh[i+10]!-a[2]!];
    const ac=[mesh[i+16]!-a[0]!,mesh[i+17]!-a[1]!,mesh[i+18]!-a[2]!];
    const n=[ab[1]!*ac[2]!-ab[2]!*ac[1]!,ab[2]!*ac[0]!-ab[0]!*ac[2]!,ab[0]!*ac[1]!-ab[1]!*ac[0]!];
    const outward=n.reduce((sum,value,axis)=>sum+value*(a[axis]!+(ab[axis]!+ac[axis]!)/3-[16,12,12][axis]!),0);
    assert.ok(outward>0,`sphere triangle ${i/24} must face outward`);
  }
}

async function runField(device: GPUDevice, name: string, ratio: 0|1|2|4,
  analytic:(q:Point)=>number, mixed=false, source?:WebGPUFineLevelSetBrickSource,
  widthAxis?: 0|1|2) {
  const dimensions=source?.plan.sampleDimensions ?? [32,24,24];
  const pages: { key:number; q:Point }[]=[];
  for(let z=0;z<3;z++)for(let y=0;y<3;y++)for(let x=0;x<4;x++){
    pages.push({key:((x+1024)|((y+512)<<11)|((z+1024)<<21))>>>0,q:[x,y,z]});
  }
  pages.sort((a,b)=>a.key-b.key);
  let metadataWords=new Uint32Array(pages.length*4);
  let sampleWords=new Uint32Array(sampleCount);
  pages.forEach((page,id)=>{
    metadataWords.set([id,page.key,generation,id],id*4);
    for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
      const q:Point=[page.q[0]*8+x,page.q[1]*8+y,page.q[2]*8+z];
      // Exercise each adjacent 2:1 rung, including the 2h/h handoff.
      const widthLog=!mixed?3:widthAxis===undefined
        ? (q[0]<16?3:q[0]<24?2:0)
        : Math.max(0,3-Math.floor(q[widthAxis]/8));
      sampleWords[id*512+x+8*(y+8*z)]=(packFineLevelSetSample(analytic(q),1 | ((name==="pool" || name==="film" || name==="rising-film") && q[1]===0 ? 2 : 0))|(widthLog<<24))>>>0;
    }
  });
  let worklistWords=new Uint32Array(7+pages.length);
  worklistWords.set([generation,pages.length,pages.length,0xc0080003,Math.ceil(pages.length/64),1,1]);
  pages.forEach((_,id)=>worklistWords[7+id]=id);
  const directoryWords=new Uint32Array(16);
  const uniformWords=new Uint32Array(28);const uf=new Float32Array(uniformWords.buffer);
  uf.set([...dimensions,1],12);
  const paramsWords=new Uint32Array(28);const pf=new Float32Array(paramsWords.buffer);
  paramsWords.set([...dimensions,8],0);paramsWords.set([4,3,3,512],4);
  paramsWords.set([pages.length,7,pages.length,generation],8);
  pf.set([0,0,0,1],12);pf[16]=1;pf.set([1,1,1,0],20);pf[27]=ratio;
  if(source){
    paramsWords.set([...dimensions,8],0);
    paramsWords.set([...dimensions.map(n=>Math.ceil(n/8)),512],4);
    metadataWords=new Uint32Array(await read(device,source.metadata,source.metadata.size));
    worklistWords=new Uint32Array(await read(device,source.worklist,source.worklist.size));
    sampleWords=new Uint32Array(await read(device,source.samples,source.plan.maximumResidentBricks*512*4));
    paramsWords.set([source.plan.maximumResidentBricks,7,source.plan.maximumResidentBricks,worklistWords[0]!],8);
    // Source distances are metres; geometry stays in the unit-cell test frame.
    pf[15]=source.plan.fineCellWidth;
    pf.set([source.plan.fineCellWidth,source.plan.fineCellWidth,source.plan.fineCellWidth,0],20);

  }
  const uniforms = buffer(device, `${name} uniforms`, uniformWords.byteLength,
    GPUBufferUsage.UNIFORM, uniformWords);
  const params = buffer(device, `${name} params`, paramsWords.byteLength,
    GPUBufferUsage.UNIFORM, paramsWords);
  const argsInitial = Uint32Array.from([0, 1, 0, 0, 0, INVALID, 0, INVALID]);
  const args = buffer(device, `${name} args`, argsInitial.byteLength,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, argsInitial);
  const cubes = buffer(device, `${name} cubes`, cubeCapacity * 8,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const values = buffer(device, `${name} values`, cubeCapacity * 32, GPUBufferUsage.STORAGE);
  const offsets = buffer(device, `${name} offsets`,
    cubeCapacity * GLOBAL_FINE_SURFACE_EMIT_LANES * 4, GPUBufferUsage.STORAGE);
  const vertices = buffer(device, `${name} vertices`, vertexCapacity * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  const metadata = buffer(device, "metadata", metadataWords.byteLength, GPUBufferUsage.STORAGE, metadataWords);
  const worklist = buffer(device, "worklist", worklistWords.byteLength, GPUBufferUsage.STORAGE, worklistWords);
  const samples = buffer(device, "samples", sampleWords.byteLength, GPUBufferUsage.STORAGE, sampleWords);
  const empty = buffer(device, `${name} empty`, 256,
    GPUBufferUsage.STORAGE, new Uint32Array(64));
  const directory = buffer(device, `${name} directory`, directoryWords.byteLength,
    GPUBufferUsage.STORAGE, directoryWords);

  const classify = device.createComputePipeline({ label: `${name} production classify`,
    layout: "auto", compute: { module: device.createShaderModule({ code: globalFineSurfaceClassificationShader }),
      entryPoint: "extractGlobalFineMain" } });
  const scan = device.createComputePipeline({ label: `${name} production scan`, layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedIndirectScanShader }),
      entryPoint: "scanGlobalFineTriangles" } });
  const emit = device.createComputePipeline({ label: `${name} production emit`, layout: "auto",
    compute: { module: device.createShaderModule({ code: globalFineClassifiedEmitShader }),
      entryPoint: "emitGlobalFineTetrahedra" } });
  const classifyGroup = device.createBindGroup({ layout: classify.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 8, resource: { buffer: worklist } }, { binding: 9, resource: { buffer: samples } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: metadata } },
    { binding: 16, resource: { buffer: directory } }, { binding: 17, resource: { buffer: empty } },
  ] });
  const scanGroup = device.createBindGroup({ layout: scan.getBindGroupLayout(0), entries: [
    { binding: 3, resource: { buffer: vertices } }, { binding: 4, resource: { buffer: args } },
    { binding: 5, resource: { buffer: cubes } }, { binding: 6, resource: { buffer: values } },
    { binding: 7, resource: { buffer: offsets } }, { binding: 10, resource: { buffer: params } },
    { binding: 11, resource: { buffer: empty } },
  ] });
  const emitGroup = device.createBindGroup({ layout: emit.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: uniforms } }, { binding: 3, resource: { buffer: vertices } },
    { binding: 4, resource: { buffer: args } }, { binding: 5, resource: { buffer: cubes } },
    { binding: 6, resource: { buffer: values } }, { binding: 7, resource: { buffer: offsets } },
    { binding: 8, resource: { buffer: worklist } }, { binding: 9, resource: { buffer: samples } },
    { binding: 10, resource: { buffer: params } }, { binding: 12, resource: { buffer: metadata } },
    { binding: 16, resource: { buffer: directory } },
  ] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(classify); pass.setBindGroup(0, classifyGroup);
  pass.dispatchWorkgroups(Math.ceil(worklistWords[1]!*512 / 256));
  pass.setPipeline(scan); pass.setBindGroup(0, scanGroup); pass.dispatchWorkgroups(1);
  pass.setPipeline(emit); pass.setBindGroup(0, emitGroup);
  pass.dispatchWorkgroups(Math.ceil(cubeCapacity / 64), GLOBAL_FINE_SURFACE_EMIT_LANES, 1);
  pass.end(); device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();

  const argsOut = new Uint32Array(await read(device, args, argsInitial.byteLength));
  const vertexCount = argsOut[0]!;
  assert.ok(vertexCount <= vertexCapacity, `${name}: vertex allocation overflow`);
  const mesh = new Float32Array(await read(device, vertices, Math.max(4, vertexCount * 32)));
  const metrics=rasterMeshSymmetryMetrics(mesh,vertexCount,
    {minimum:[0,0,0],maximum:dimensions.map(n=>n*(source?.plan.fineCellWidth ?? 1)) as [number,number,number],tolerance:1e-4});
  const cubeWords=new Uint32Array(await read(device,cubes,argsOut[4]!*8));
  let adaptive=0;for(let i=0;i<argsOut[4]!;i++)adaptive+=Number(((cubeWords[i*2+1]!>>>16)&255)===193);
  for(const resource of [uniforms,params,args,cubes,values,offsets,vertices,empty,directory,metadata,worklist,samples])resource.destroy();
  return {metrics,adaptive,mesh};
}

const fs=await import("node:fs/promises");
await acquireWebGPUExclusiveLock("dawn-probe","mini32-mesh-holes");
const {create,globals}=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis,globals);const gpu=create(["backend=metal"]);liveDawnInstances.add(gpu);
const adapter=await gpu.requestAdapter();const device=await adapter!.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter!.limits)});
try {
 for(const step of [3,8,12,16,20,24,30,40,52,60]){
  const path=`/tmp/cm12-holes/step-${step}/`;const plan=JSON.parse(await fs.readFile(path+"source.json","utf8"));
  const source:any={plan};for(const key of ["metadata","worklist","samples"]){const bytes=await fs.readFile(path+key+".bin");source[key]=buffer(device,key,bytes.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC,bytes);}
  for(const ratio of [0,1] as const){const result=await runField(device,`step-${step}`,ratio,()=>0,false,source);
   console.log(JSON.stringify({step,ratio,adaptive:result.adaptive,triangles:result.metrics.triangleCount,open:result.metrics.interiorOpenEdgeCount,firstOpen:result.metrics.firstInteriorOpenEdge,topology:topology(result.mesh,result.metrics.vertexCount)}));
   await fs.writeFile(`/tmp/cm12-holes/step-${step}/mesh-${ratio}.bin`,new Uint8Array(result.mesh.buffer));
  }
  for(const key of ["metadata","worklist","samples"])source[key].destroy();
 }
}finally{device.destroy();await releaseWebGPUExclusiveLock();liveDawnInstances.delete(gpu);}
