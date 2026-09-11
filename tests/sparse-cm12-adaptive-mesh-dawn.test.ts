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
const dawnTest=process.env.WEBGPU_NODE_MODULE?test:test.skip;
for(const fullPool of (process.env.FLUID_FULL_POOL ? [true] : [false,true])) dawnTest(
  fullPool ? "adaptive surface mesh covers the default coarse-first pool"
    : "adaptive surface mesh ratios preserve closed mixed-resolution contours",
  {timeout:600000},async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","adaptive-surface-mesh");
  let device:GPUDevice|undefined;
  let gpu:GPU|undefined;
  try {
    const {create,globals}=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis,globals);gpu=create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    liveDawnInstances.add(gpu!);
    const adapter=await gpu!.requestAdapter();assert.ok(adapter);
    device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    const errors:string[]=[];device!.addEventListener("uncapturederror",event=>errors.push(event.error.message));
    const fields:Record<string,(q:Point)=>number>={
      pool:([,y])=>y-7.25,
      "clamped-pool":([,y])=>y<7.5?-2:2,
      film:([,y])=>y+0.25,
      "rising-film":([x,y])=>y+0.25-0.25*x,
      torus:([x,y,z])=>Math.hypot(Math.hypot(x-15.5,z-11.5)-6,y-11.5)-2.25,
      sphere:([x,y,z])=>Math.hypot(x-15.5,y-11.5,z-11.5)-8.25,
      ellipsoid:([x,y,z])=>Math.hypot((x-15.5)/10,(y-11.5)/7,(z-11.5)/8)-1,
      box:([x,y,z])=>Math.max(Math.abs(x-15.5)-10.25,Math.abs(y-11.5)-6.25,Math.abs(z-11.5)-8.25),
      disconnected:([x,y,z])=>Math.min(Math.hypot(x-10,y-10,z-10)-2.25,Math.hypot(x-14,y-12,z-12)-1.25),
    };
    for(const [name,field] of Object.entries(fullPool ? {} : fields)){
      const fine=await runField(device!,name,0,field,true);
      for(const ratio of [1,2,4] as const){
        const result=await runField(device!,name,ratio,field,true);
        console.log(JSON.stringify({name,ratio,fine:fine.metrics.triangleCount,adaptive:result.adaptive,triangles:result.metrics.triangleCount,open:result.metrics.interiorOpenEdgeCount,degenerate:result.metrics.degenerateTriangleCount,fineDegenerate:fine.metrics.degenerateTriangleCount}));
        assert.equal(components(result.mesh,result.metrics.vertexCount),
          components(fine.mesh,fine.metrics.vertexCount),"simplification must preserve disconnected components");
        assert.deepEqual(topology(result.mesh,result.metrics.vertexCount),
          topology(fine.mesh,fine.metrics.vertexCount),"simplification must preserve holes and winding");
        assert.equal(result.metrics.nonFiniteCount,0);
        if(name==="sphere")assertSphereWinding(result.mesh);
        // Existing floor/wall clipping contains collapsed boundary triangles;
        // simplification must add none. Closed interior fixtures have zero.
        assert.equal(result.metrics.degenerateTriangleCount,fine.metrics.degenerateTriangleCount);
        assert.equal(result.metrics.interiorOpenEdgeCount,0,`${name} x${ratio}: cracks ${result.metrics.firstInteriorOpenEdge}`);
        assert.equal(result.metrics.nonManifoldEdgeCount,0);
        if(name!=="disconnected" && name!=="film"){
          assert.ok(result.adaptive>0,"adaptive mesh path must execute");
          assert.ok(result.metrics.triangleCount<fine.metrics.triangleCount,"adaptive mesh must reduce triangles");
        }
      }
    }
    if(!fullPool)for(const axis of [0,1,2] as const){
      const field=fields.sphere!;
      const fine=await runField(device!,`sphere-axis-${axis}`,0,field,true,undefined,axis);
      for(const ratio of [1,2,4] as const){
        const result=await runField(device!,`sphere-axis-${axis}`,ratio,field,true,undefined,axis);
        assert.ok(result.adaptive>0,`axis ${axis}: adaptive mesh must execute`);
        assertSphereWinding(result.mesh);
        assert.equal(result.metrics.interiorOpenEdgeCount,0,`axis ${axis} x${ratio}: 2:1 seams must close`);
        assert.equal(result.metrics.nonManifoldEdgeCount,0);
        assert.equal(result.metrics.degenerateTriangleCount,0);
        assert.equal(result.metrics.nonFiniteCount,0);
        assert.deepEqual(topology(result.mesh,result.metrics.vertexCount),topology(fine.mesh,fine.metrics.vertexCount));
      }
    }
    let solver:WebGPUAdaptiveMassSolver|undefined;
    try {
      const scene=fullPool ? sceneDocument(getSceneDefinition("coarse-first-pool-impact")) : sceneAtContainerExtents(sceneDocument(getSceneDefinition("coarse-first-pool-impact")),
        {width_m:1.6,height_m:1.2,depth_m:1.2});
      if(!fullPool){scene.voxelDomain.finestCellSize_m=.05;scene.container.fillFraction=.5;
      delete scene.fluid.initialLiquidVolumes;}
      const values=resolveMethodValues(adaptiveMassMethod,"balanced",{selectorMode:"coarse-first",timeStep:"scene"});
      solver=await adaptiveMassMethod.createSolverAsync!(device!,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();
      assert.equal(solver.globalFineLevelSetSource.surfaceMeshRefinement,2);
      for(const step of (fullPool ? [0,1,3,30,55] : [0,1])){
        while ((solver.info.encodedSteps ?? 0) < step) {
          const nextStep = (solver.info.encodedSteps ?? 0) + 1;
          while (!solver.advanceTo(nextStep / 60, [])) await new Promise(setImmediate);
          await solver.waitForTopologyReady();
        }
        assert.equal(solver.info.encodedSteps ?? 0, step, "capture the actual solver checkpoint");
        if(fullPool&&step<=30){
          // Inspect accepted mass and velocity independently of presentation.
          // The falling ball is still above 2 m at 0.5 s; its changing support
          // rungs must not disturb the disconnected, initially still pool.
          const fields=await solver.readDiagnosticFields(true);
          let minimumHeight=Infinity,maximumHeight=-Infinity,maximumSpeed=0;
          for(let z=0;z<128;z++)for(let x=0;x<128;x++){
            let height=0;
            for(let y=0;y<40;y++){
              const at=x+128*(y+96*z);
              height+=Math.max(0,Math.min(1,fields.density[at]!/
                Math.max(1e-6,fields.solidOpenFraction[at]!)))*.05;
              if(y<31)maximumSpeed=Math.max(maximumSpeed,
                Math.hypot(...fields.velocity.subarray(at*4,at*4+3)));
            }
            minimumHeight=Math.min(minimumHeight,height);
            maximumHeight=Math.max(maximumHeight,height);
          }
          assert.ok(minimumHeight>1.599&&maximumHeight<1.601,
            `step ${step}: accepted pool height ${minimumHeight}..${maximumHeight}`);
          assert.ok(maximumSpeed<.003,`step ${step}: still pool speed ${maximumSpeed} m/s`);
          console.log(JSON.stringify({fullPool,step,minimumHeight,maximumHeight,maximumSpeed}));
        }
        const source:WebGPUFineLevelSetBrickSource=solver.globalFineLevelSetSource;
        const fine=await runField(device!,"real-pool",0,()=>0,false,source);
        for(const ratio of [1,2,4] as const){
          const adaptive=await runField(device!,"real-pool",ratio,()=>0,false,source);
          assert.ok(adaptive.adaptive>0,"accepted solver cell sizes must reach mesh classification");
          assert.equal(adaptive.metrics.nonFiniteCount,0);
          assert.equal(adaptive.metrics.nonManifoldEdgeCount,fine.metrics.nonManifoldEdgeCount);
          assert.equal(adaptive.metrics.degenerateTriangleCount,fine.metrics.degenerateTriangleCount);
          assert.equal(adaptive.metrics.openEdgeCount,fine.metrics.openEdgeCount,"solver publication must remain equally closed");
          assert.ok(adaptive.metrics.triangleCount<fine.metrics.triangleCount);
          if(fullPool&&step===55){
            // The published surface crosses moving 2:1 interfaces.
            // Inspect emitted edges after impact, including all fallback patches.
            // Floor clipping already leaves edges on the first sample plane
            // (also present in the full-resolution reference). No open edge
            // may escape that exact plane into the reconstructed free surface.
            const floorY=source.plan.fineCellWidth*.5;
            const surfaceEdges=adaptive.metrics.interiorOpenEdges!.filter(edge=>
              edge.endpoints.some(point=>Math.abs(point[1]-floorY)>1e-5));
            assert.equal(surfaceEdges.length,0,
              `impact x${ratio}: cracks ${JSON.stringify(surfaceEdges[0])}`);
            let internalTriangles=0;
            const m=adaptive.mesh;
            for(let i=0;i<m.length;i+=24){
              const x=(m[i]!+m[i+8]!+m[i+16]!)/3;
              const y=(m[i+1]!+m[i+9]!+m[i+17]!)/3;
              const z=(m[i+2]!+m[i+10]!+m[i+18]!)/3;
              if(x>.1&&x<6.3&&z>.1&&z<6.3&&y>.1&&y<1.5
                &&Math.hypot(x-3.2,z-3.2)>1.2)internalTriangles++;
            }
            assert.equal(internalTriangles,0,"impact must not expose internal pool sheets");
            console.log(JSON.stringify({fullPool,step,ratio,internalTriangles,
              surfaceOpenEdges:surfaceEdges.length,
              floorClippingEdges:adaptive.metrics.interiorOpenEdgeCount,
              adaptiveTriangles:adaptive.adaptive}));
          }else if(fullPool){
            // Count actual oriented geometry: closure alone accepts an empty top.
            // The falling ball has not reached the 1.6 m pool in these checkpoints.
            let upwardArea=0,downwardArea=0,internalTriangles=0;
            let minimumTop=Infinity,maximumTop=-Infinity;
            const m=adaptive.mesh;
            for(let i=0;i<m.length;i+=24){
              const center=[0,1,2].map(axis=>(m[i+axis]!+m[i+8+axis]!+m[i+16+axis]!)/3);
              if(center[0]!>.1&&center[0]!<6.3&&center[2]!>.1&&center[2]!<6.3
                &&center[1]!>.1&&center[1]!<1.5)internalTriangles++;
              if(![1,9,17].every(k=>Math.abs(m[i+k]!-1.6)<.1))continue;
              const area=((m[i+10]!-m[i+2]!)*(m[i+16]!-m[i]!)-(m[i+8]!-m[i]!)*(m[i+18]!-m[i+2]!))*.5;
              upwardArea+=Math.max(0,area);downwardArea+=Math.max(0,-area);
              if(area>0)for(const k of [1,9,17]){
                minimumTop=Math.min(minimumTop,m[i+k]!);maximumTop=Math.max(maximumTop,m[i+k]!);
              }
            }
            assert.ok(Math.abs(upwardArea-40.96)<1e-4,`step ${step} x${ratio}: pool top area ${upwardArea}`);
            assert.equal(downwardArea,0,"pool top triangles must face upward");
            assert.equal(internalTriangles,0,"a filled pool must not publish interior interfaces");
            assert.ok(minimumTop>1.599&&maximumTop<1.601,
              `step ${step} x${ratio}: calm pool height ${minimumTop}..${maximumTop}`);
            console.log(JSON.stringify({fullPool,step,ratio,upwardArea,downwardArea,
              minimumTop,maximumTop,internalTriangles,adaptiveTriangles:adaptive.adaptive}));
          }else{
            assert.equal(adaptive.metrics.nonManifoldEdgeCount,0);
          }
        }
      }

      const generation=solver.globalFineLevelSetSource.generation;
      for(const ratio of [1,4,2] as const){
        solver.applyRuntimeValues({...values,surfaceMeshRefinement:String(ratio)});
        assert.equal(solver.globalFineLevelSetSource.surfaceMeshRefinement,ratio);
        assert.equal(solver.globalFineLevelSetSource.generation,generation,"changing mesh quality must preserve physics");
      }
    }finally{solver?.destroy();}
    assert.deepEqual(errors,[]);
  }finally{device?.destroy();await releaseWebGPUExclusiveLock();if(gpu)liveDawnInstances.delete(gpu);}
});

// The early mini64 tower crosses native macro/fine pages. Counting a far-front
// sample alone misses the open surface left behind at those shared faces.
dawnTest("mini64 moving native macro contours remain closed", {timeout:180000}, async()=>{
  await acquireWebGPUExclusiveLock("dawn-test","mini64-macro-mesh");
  let device:GPUDevice|undefined, solver:WebGPUAdaptiveMassSolver|undefined, gpu:GPU|undefined;
  try {
    const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis,dawn.globals); gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);
    liveDawnInstances.add(gpu!);
    const adapter=await gpu!.requestAdapter(); assert.ok(adapter);
    device=await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
    const errors:string[]=[];
    device.addEventListener("uncapturederror",event=>{event.preventDefault();errors.push(event.error.message);});
    const scene=sceneDocument(getSceneDefinition("minimal-power-dam-break-64")); scene.duration_s=1;
    const values=resolveMethodValues(adaptiveMassMethod,"balanced",{});
    solver=await adaptiveMassMethod.createSolverAsync!(device,scene,"balanced",values,undefined,()=>{}) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    for(let step=1;step<=10;step++){
      while(!solver.advanceTo(step/30,[]))await new Promise(setImmediate);
      if(step%2===0)await device.queue.onSubmittedWorkDone();
    }
    await solver.assertSimulationHealthy();
    const source=solver.globalFineLevelSetSource;
    const floorY=source.plan.fineCellWidth*.5;
    for(const ratio of [0,2,4] as const){
      const result=await runField(device,"mini64",ratio,()=>0,false,source);
      const surfaceEdges=result.metrics.interiorOpenEdges!.filter(edge=>
        edge.endpoints.some(point=>Math.abs(point[1]-floorY)>1e-5));
      console.log(JSON.stringify({scene:"mini64",step:10,ratio,
        triangles:result.metrics.triangleCount,surfaceOpenEdges:surfaceEdges.length,
        nonManifold:result.metrics.nonManifoldEdgeCount}));
      assert.ok(result.metrics.triangleCount>0);
      assert.equal(result.metrics.nonFiniteCount,0);
      assert.equal(surfaceEdges.length,0,`macro/fine surface crack: ${JSON.stringify(surfaceEdges[0])}`);
      assert.equal(result.metrics.nonManifoldEdgeCount,0);
    }
    assert.deepEqual(errors,[]);
  }finally{solver?.destroy();device?.destroy();if(gpu)liveDawnInstances.delete(gpu);await releaseWebGPUExclusiveLock();}
});
