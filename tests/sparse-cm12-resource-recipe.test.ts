import assert from "node:assert/strict";
import test from "node:test";
import { createCM12ResourceRecorder, realizeCM12ResourceRecipe } from "../lib/methods/adaptive-mass/sparse-cm12-resource-recipe";
import { createSolidWorld, SolidWorldDirectory } from "../lib/core/solid-world";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { PreparedSparseCM12GenerationTransfer, sparseCM12TransferFaceGeometry } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";
import { WebGPUSparseCM12RigidCoupling } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-rigid-coupling";

async function withWebGPUConstants(run: () => Promise<void>) {
 const constants = {
  GPUBufferUsage: {MAP_READ:1,MAP_WRITE:2,COPY_SRC:4,COPY_DST:8,INDEX:16,VERTEX:32,
   UNIFORM:64,STORAGE:128,INDIRECT:256,QUERY_RESOLVE:512},
  GPUShaderStage: {VERTEX:1,FRAGMENT:2,COMPUTE:4},
  GPUMapMode: {READ:1,WRITE:2},
 };
 const previous = new Map(Object.keys(constants).map(key =>
  [key,Object.getOwnPropertyDescriptor(globalThis,key)]));
 try {
  for(const [key,value] of Object.entries(constants))
   Object.defineProperty(globalThis,key,{configurable:true,value});
  await run();
 } finally {
  for(const [key,descriptor] of previous) {
   if(descriptor) Object.defineProperty(globalThis,key,descriptor);
   else Reflect.deleteProperty(globalThis,key);
  }
 }
}

test("CPU resource recipes snapshot uploads and preserve shared resource references", async () => {
 const recorder = createCM12ResourceRecorder({} as GPUSupportedLimits);
 const buffer = recorder.device.createBuffer({size:16,usage:128,mappedAtCreation:true});
 new Uint32Array(buffer.getMappedRange()).set([1,2,3,4]); buffer.unmap();
 const words = new Uint32Array([8,9]); recorder.device.queue.writeBuffer(buffer,4,words); words.fill(0);
 const recipe = structuredClone(recorder.finish({buffer,again:buffer,words:new Uint32Array([7])}));
 const allocated: {data:Uint8Array;destroyed:boolean}[] = [];
 const device = { createBuffer(descriptor:GPUBufferDescriptor) {
   const result={data:new Uint8Array(descriptor.size),destroyed:false,destroy(){this.destroyed=true;}};
   allocated.push(result); return result;
  }, queue:{writeBuffer(buffer:typeof allocated[number],offset:number,data:ArrayBuffer,start=0,size?:number){
   buffer.data.set(new Uint8Array(data,start,size),offset);
  }}, lost:new Promise(()=>{}), addEventListener(){},
 } as unknown as GPUDevice;
 const realized = await realizeCM12ResourceRecipe(device,recipe,[],{uploadChunkBytes:4});
 const state=realized.state as {buffer:typeof allocated[number];again:object;words:Uint32Array};
 assert.equal(state.buffer,state.again);
 assert.deepEqual([...new Uint32Array(state.buffer.data.buffer)],[1,8,9,4]);
 assert.deepEqual([...state.words],[7]);
 realized.destroy(); assert.ok(allocated.every(buffer=>buffer.destroyed));
});

test("recipes release temporary shader modules after their final pipeline dependency", () => {
 const recorder = createCM12ResourceRecorder({} as GPUSupportedLimits);
 const module = recorder.device.createShaderModule({code:"@compute @workgroup_size(1) fn main() {}"});
 const first = recorder.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
 void first;
 const recipe = recorder.finish({});
 const moduleId = recipe.operations[0]!.result!;
 assert.ok(!recipe.releaseAfter![0]!.includes(moduleId));
 assert.ok(recipe.releaseAfter![1]!.includes(moduleId));
 const retained = recorder.finish({module});
 assert.ok(retained.releaseAfter!.every(ids => !ids.includes(moduleId)), "resident-owned handles must survive hydration");
});

test("resource hydration preserves cyclic maps, sets, views, and external identities", async () => {
 const recorder=createCM12ResourceRecorder({} as GPUSupportedLimits,[{size:16,usage:128}]);
 const external=recorder.externalResources[0]!;
 const backing=new ArrayBuffer(16);
 const values=new Uint32Array(backing,4,2); values.set([11,12]);
 const graph: {self?:unknown;map:Map<unknown,unknown>;set:Set<unknown>;values:Uint32Array;view:DataView} = {
  map:new Map(),set:new Set(),values,view:new DataView(backing,4,8),
 };
 graph.self=graph; graph.map.set(external,graph); graph.set.add(external);
 const recipe=structuredClone(recorder.finish(graph));
 const liveExternal={identity:"live buffer"};
 const realized=await realizeCM12ResourceRecipe({lost:new Promise(()=>{}),addEventListener(){}} as unknown as GPUDevice,
  recipe,[liveExternal]);
 const result=realized.state as typeof graph;
 assert.equal(result.self,result);
 assert.equal(result.map.get(liveExternal),result);
 assert.ok(result.set.has(liveExternal));
 assert.equal(result.values.buffer,result.view.buffer);
 assert.equal(result.view.getUint32(0,true),11);
 assert.equal(result.values.byteOffset,4);
 realized.destroy();
});

test("a failed recipe destroys owned buffers without destroying external resources", async () => {
 const recorder=createCM12ResourceRecorder({} as GPUSupportedLimits,[{size:16,usage:128}]);
 recorder.device.createBuffer({size:16,usage:128});
 const recipe=structuredClone(recorder.finish({}));
 const broken={...recipe,operations:[...recipe.operations,
  {target:"device" as const,method:"unsupportedMethod",args:[]}]};
 let destroyed=0;
 let externalDestroyed=0;
 const device={createBuffer(){return {destroy(){destroyed++;}}},
  lost:new Promise(()=>{}),addEventListener(){}} as unknown as GPUDevice;
 await assert.rejects(realizeCM12ResourceRecipe(device,broken,[{destroy(){externalDestroyed++;}}]),/unsupportedMethod is unavailable/);
 assert.equal(destroyed,1);
 assert.equal(externalDestroyed,0);
});

test("rigid and transfer recipes hydrate prototypes and retain live external bindings", async () => {
 await withWebGPUConstants(async () => {
  const atlas=createSparseAdaptiveMassAtlas([8,8,8],[{
   key:0,coordinate:[0,0,0],resolution:1,
   density:new Float64Array([1]),gamma:new Float64Array([1]),
  }],0,8);
  const grid=buildSparseAtlasCompositeGrid(atlas);
  const descriptor={size:1024,usage:128|4|8};
  const input: Parameters<typeof WebGPUSparseCM12Resident.recordPreparedGeneration>[0]={
   atlas,active:new Set([0]),finestCellSize_m:0.05,solidWorld:createSolidWorld(),
   maximumBytes:64*1024*1024,topologyPageCapacityMaximum:0,
   symmetry:{scalar:false,face:false},limits:{maxComputeWorkgroupsPerDimension:65535} as GPUSupportedLimits,
   rigid:{bodies:descriptor,exchange:descriptor,worldDimensions_m:[0.4,0.4,0.4]},
   source:{geometry:{dimensions:[8,8,8],cells:[{id:0,lower:[0,0,0],widths:[8,8,8],span:8}],
    faces:grid.gradientRows.map(row=>sparseCM12TransferFaceGeometry(row.id,row.axis,row.centerFine,row.area,8))},
    cellIds:new Uint32Array([0]),rowIds:Uint32Array.from(grid.gradientRows,row=>row.id),
    densityOffset:0,gammaOffset:2,velocityOffset:4,pressureOffset:48,faceOffset:16,
    stateDescriptor:descriptor,controlDescriptor:descriptor,
    liveControl:{scalarParityWord:0,faceParityWord:1,densityOffsets:[0,1],gammaOffsets:[2,3],
     velocityOffsets:[4,8],faceOffsets:[16,32]}},
  };
  const recipe=structuredClone(await WebGPUSparseCM12Resident.recordPreparedGeneration(input));
  const replay=createCM12ResourceRecorder(input.limits);
  const external=Array.from({length:4},()=>replay.device.createBuffer(descriptor));
  const realized=await realizeCM12ResourceRecipe(replay.device,recipe,external);
  const data=realized.state as {resident:Record<string,unknown>;transfer:object};
  assert.ok(data.transfer);
  const resident=Object.assign(Object.create(WebGPUSparseCM12Resident.prototype),data.resident,
   {device:replay.device,currentSolidWorld:input.solidWorld}) as WebGPUSparseCM12Resident;
  const coupling=data.resident.rigidCoupling as object;
  assert.ok(coupling);
  Object.setPrototypeOf(coupling,WebGPUSparseCM12RigidCoupling.prototype);
  const transfer=Object.assign(Object.create(PreparedSparseCM12GenerationTransfer.prototype),
   data.transfer,{device:replay.device}) as PreparedSparseCM12GenerationTransfer;
  const configuration=data.resident.replacementConfiguration as {rigid:{bodies:GPUBuffer;exchange:GPUBuffer}};
  assert.equal(configuration.rigid.bodies,external[0]);
  assert.equal(configuration.rigid.exchange,external[1]);
  const encoder=replay.device.createCommandEncoder();
  transfer.encode(encoder);
  resident.encode(encoder,0.01,0.05,1,[0,-9.81,0],undefined,undefined,undefined,
   undefined,1,[0.4,0.4,0.4]);
  replay.device.queue.submit([encoder.finish()]);
  assert.doesNotThrow(()=>structuredClone(replay.finish({resident,transfer})));
  transfer.destroy(); resident.destroy(); realized.destroy();
 });
});

test("a tiny B8 resident records and clones its complete construction without a GPU", async () => {
 await withWebGPUConstants(async () => {
  const atlas = createSparseAdaptiveMassAtlas([8,8,8],[{
   key:0,coordinate:[0,0,0],resolution:1,
   density:new Float64Array([1]),gamma:new Float64Array([1]),
  }],0,8);
  const input = structuredClone({atlas,active:new Set([0]),finestCellSize_m:0.05,
   solidWorld:createSolidWorld([{operation:"fill",minimum:[0,0,0],maximumExclusive:[1,1,1]}]),maximumBytes:64*1024*1024,topologyPageCapacityMaximum:0,
   symmetry:{scalar:false,face:false},
   limits:{maxComputeWorkgroupsPerDimension:65535} as GPUSupportedLimits});
  // This is the explicit hydration performed by the CPU preparation worker.
  Object.setPrototypeOf(input.solidWorld.directory,SolidWorldDirectory.prototype);
  const recipe = structuredClone(await WebGPUSparseCM12Resident.recordPreparedGeneration(input));
  assert.ok(recipe.operations.some(operation=>operation.method==="createBuffer"));
  assert.ok(recipe.operations.some(operation=>operation.method==="createComputePipelineAsync"));
  assert.ok(recipe.operations.some(operation=>operation.method==="writeBuffer"));
  for(const method of ["createCommandEncoder","beginComputePass","setPipeline",
   "setBindGroup","dispatchWorkgroups","end","finish","submit"])
   assert.ok(recipe.operations.some(operation=>operation.method===method),
    `aperture initialization must record ${method}`);
  const state=(recipe.state as {resident:Record<string,unknown>}).resident;
  assert.equal(state.cellCount, 1 + 8 + 64 + 512,
    "recorded capacity includes the B1/B2/B4/B8 mutation catalogue, not just the one accepted cell");
  assert.equal(state.simulationPipelinesReady,true);
  assert.ok(state.templateWords instanceof Uint32Array);
  assert.ok(state.currentSolidWorld);
  assert.doesNotThrow(()=>structuredClone(recipe));
  // Replay onto a second CPU recorder: this exercises every recorded API and
  // symbolic dependency without obtaining or executing on a GPU device.
  const replay = createCM12ResourceRecorder(input.limits);
  const realized = await realizeCM12ResourceRecipe(replay.device,recipe,[],
   {uploadChunkBytes:4096});
  const resident = Object.assign(Object.create(WebGPUSparseCM12Resident.prototype),
   (realized.state as {resident:object}).resident,{device:replay.device,currentSolidWorld:input.solidWorld,
    simulationCompilationSnapshot:()=>({state:"idle",generation:0,queued:0,active:0,cached:0})}) as WebGPUSparseCM12Resident;
  assert.equal(resident.simulationReady,true);
  await resident.waitForSimulationPipelines();
  const encoder=replay.device.createCommandEncoder();
  resident.encodeInitialPresentation(encoder,0.05);
  resident.encode(encoder,0.01,0.05,1,[0,-9.81,0]);
  replay.device.queue.submit([encoder.finish()]);
  const replayed=structuredClone(replay.finish({resident}));
  assert.ok(replayed.operations.some(operation=>operation.method==="dispatchWorkgroupsIndirect"));
  assert.ok(replayed.operations.some(operation=>operation.method==="submit"));
  resident.destroy();
  realized.destroy();
 });
});
