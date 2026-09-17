import assert from "node:assert/strict";
import test from "node:test";
import { createCM12ResourceRecorder, realizeCM12ResourceRecipe } from "../lib/methods/adaptive-volume/sparse-cm12-resource-recipe";
import { createSolidWorld, SolidWorldDirectory } from "../lib/core/solid-world";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { PreparedSparseCM12GenerationTransfer, sparseCM12TransferFaceGeometry } from "../lib/methods/adaptive-volume/sparse-cm12-generation-transfer";
import { WebGPUSparseCM12RigidCoupling } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-rigid-coupling";

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

test("recipes preserve auto pipeline bind-group layout queries through hydration", async () => {
 const recorder=createCM12ResourceRecorder({} as GPUSupportedLimits);
 const module=recorder.device.createShaderModule({code:"@compute @workgroup_size(1) fn main() {}"});
 const pipeline=await recorder.device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
 const layout=pipeline.getBindGroupLayout(0);
 const group=recorder.device.createBindGroup({layout,entries:[]});
 const recipe=structuredClone(recorder.finish({group}));
 const query=recipe.operations.find(operation=>operation.method==="getBindGroupLayout");
 assert.ok(query?.result !== undefined);
 const creation=recipe.operations.find(operation=>operation.method==="createBindGroup");
 assert.deepEqual((creation?.args[0] as {layout:unknown}).layout,{cm12Resource:query.result});

 const replay=createCM12ResourceRecorder({} as GPUSupportedLimits);
 const realized=await realizeCM12ResourceRecipe(replay.device,recipe);
 const replayed=replay.finish(realized.state);
 assert.ok(replayed.operations.some(operation=>operation.method==="getBindGroupLayout"));
 assert.ok(replayed.operations.some(operation=>operation.method==="createBindGroup"));
 realized.destroy();
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


