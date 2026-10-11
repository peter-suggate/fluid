import assert from "node:assert/strict";
import test from "node:test";
import { UniformMixedExtension } from "../lib/methods/uniform/uniform-mixed-extension";
import { pairedExtensionSource, pairedSnapshotSource, replayedPublicationSource } from "../lib/methods/uniform/uniform-mixed-extension-pair.wgsl";
import { uniformMixedExtensionSnapshotWGSL } from "../lib/methods/uniform/uniform-mixed-extension-slots.wgsl";
import type { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import type { UniformDetailGroup } from "../lib/methods/uniform/uniform-detail-fields";

async function fixture(storageBuffers = 9, fineTiles = 8, regularBulk = false) {
 Object.assign(globalThis, { GPUShaderStage: { COMPUTE: 4 }, GPUTextureUsage: { STORAGE_BINDING: 8 } });
 const codes: string[] = [], pipelines: GPUComputePipelineDescriptor[] = [];
 const device = {
  limits: { maxStorageBuffersPerShaderStage: storageBuffers },
  createBindGroupLayout: (d: unknown) => d, createBindGroup: (d: unknown) => d,
  createPipelineLayout: (d: unknown) => d,
  createTexture: () => ({ createView: () => ({}) }),
  createShaderModule: (d: GPUShaderModuleDescriptor) => { codes.push(d.code); return { getCompilationInfo: async () => ({ messages: [] }) }; },
  createComputePipelineAsync: async (d: GPUComputePipelineDescriptor) => { pipelines.push(d); return d; },
 } as unknown as GPUDevice;
 const dimensions = [8, 8, 8] as const;
 const events: { entry: string; group: unknown; pipeline:GPUComputePipeline }[] = [], hierarchyReplay: boolean[] = [];
 let group: unknown;
 const dispatch = (_pass: unknown, pipeline: GPUComputePipeline) => {
  events.push({ entry: (pipeline as unknown as GPUComputePipelineDescriptor).compute.entryPoint!, group, pipeline });
 };
 const ownership = {
  capacity: { lattice: { dimensions }, tileDimensions: [2, 2, 2], tiles: 8, fineTiles },
  bindLayout: {}, bindGroup: {}, dispatchX: 65535, coarseOnly: fineTiles === 0,
  dispatchTierCounted: dispatch, dispatchCounted: dispatch, dispatchRegularCoarseCounted: dispatch,
 } as unknown as UniformMixedOwnership;
 const texture = (format = "rgba32float") => ({ width: 8, height: 8, depthOrArrayLayers: 8, format, createView: () => ({}) }) as unknown as GPUTexture;
 const hierarchy = { output: texture(), outputOrigins: texture("rgba32uint"), input: texture(), inputOrigins: texture("rgba32uint"),
  encode: (_pass: unknown, replay: boolean) => hierarchyReplay.push(replay) };
 const extension = new UniformMixedExtension(device, ownership, hierarchy as never, regularBulk, true);
 await extension.initialize();
 const pass = { setBindGroup: (index: number, value: unknown) => { if (index === 1) group = value; },
  setPipeline: (pipeline: GPUComputePipeline) => dispatch(undefined, pipeline), dispatchWorkgroups() {}, end() {} };
 const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
 const groups = [0, 1].map(epoch => ({ group: { epoch }, base: { epoch } })) as unknown as readonly [UniformDetailGroup, UniformDetailGroup];
 const fields = { physical: texture(), phase: texture("r32float"), negative: {} as GPUBuffer,
  output: texture(), outputNegative: {} as GPUBuffer, params: {} as GPUBuffer,
  scratch: { buffer: { size: extension.scratchBytesAt(fineTiles) } as GPUBuffer } };
 return { extension, codes, pipelines, events, hierarchyReplay, encoder, groups, fields, texture };
}

test("paired generation rejects missing, duplicated and changed source operations", async () => {
 const { codes } = await fixture();
 // The fake device has no detail storage. Undo runtime constant accessors so
 // these mutations exercise the generator's input contract independently.
 const scalar = codes[0]!.replace(/\b([A-Z][A-Z_0-9]*)\(\)/g, "$1");
 const pair = pairedExtensionSource(scalar);
 assert.match(pair, /fn pair_ueCombine\(old:UEPair/);
 assert.match(pair, /pairIn\[slot\]=value.x.y;pairOut\[slot\]=value.x.y/);
 assert.match(pair, /let state=ueFinishState\(umSlot\(/);
 const retained=pairedExtensionSource(scalar,true);
 const seed=retained.slice(retained.indexOf("fn pair_ueSeedStore("),retained.indexOf("fn pair_umSeedState("));
 assert.doesNotMatch(seed,/pairIn\[slot\]=|pairOut\[slot\]=/);
 assert.match(retained,/fn retainFine/);
 assert.match(retained,/fn retainedStore\(slot:u32,value:f32\)\{pairIn\[slot\]=value;pairOut\[slot\]=value;/);
 assert.match(pairedSnapshotSource(retained,true),/return pairIn\[umSlot\(face.anchor,face.axis,face.width\)\];/);
 assert.throws(() => pairedExtensionSource(scalar.replace("fn umSlotState(", "fn renamedSlotState(")), /expected one definition, found 0/);
 assert.throws(() => pairedExtensionSource(scalar + "\nfn umSlotState(){}"), /expected one definition, found 2/);
 assert.throws(() => pairedExtensionSource(scalar.replace("var weighted=0.0", "var weighted=1.0")), /expected 1 occurrences.*var weighted/);
 assert.throws(() => pairedExtensionSource(scalar.replace("stateOut[slot]=value;stateIn[slot]=value;", "stateOut[slot]=value;")), /ueSeedStore changed/);
 assert.throws(() => pairedExtensionSource(scalar.replace("return stateIn[umSlot(p,axis,1u)].x;", "return 0.0;")), /expected 1 occurrences.*return stateIn/);
});

test("paired encoding honors zero, odd and even sweeps for fine and coarse-only capacities", async () => {
 for (const fineTiles of [0, 8]) {
  const f = await fixture(9, fineTiles);
  f.events.length=0;f.extension.retainPair(f.encoder,f.groups);
  assert.deepEqual(f.events.map(e=>e.entry),[...(fineTiles?["retainFine"]:[]),"retainCoarse"]);
  assert.ok(f.events.every(e=>(e.group as {epoch:number}).epoch===0));
  for (const sweeps of [0, 1, 2, 3, 8]) {
   f.events.length = 0;
   f.extension.preparePair(f.encoder, f.groups, sweeps);
   const seeds = f.events.filter(e => e.entry.startsWith("pair_seed"));
   const regularSweeps = f.events.filter(e => e.entry === "pair_sweep");
   assert.ok(seeds.length > 0);
   assert.ok(seeds.every(e => (e.group as { epoch: number }).epoch === 1));
   assert.deepEqual(regularSweeps.map(e => (e.group as { epoch: number }).epoch), Array.from({ length: sweeps }, (_, i) => i % 2));
   f.events.length=0;f.extension.finishSnapshot(f.encoder,f.groups,sweeps);
   assert.equal(f.hierarchyReplay.at(-1),false);
   assert.deepEqual(f.events.map(e=>e.entry),["restrictBand",...(fineTiles?["snapshotFine"]:[]),"snapshotCoarse"]);
   assert.ok(f.events.every(e=>(e.group as {epoch:number}).epoch===sweeps%2));
   for (const companion of [true, false]) {
    f.events.length = 0;
    f.extension.finishPair(f.encoder, f.groups, sweeps, companion);
    assert.equal(f.hierarchyReplay.at(-1), !companion);
    assert.ok(f.events.every(e => !e.entry.includes("seed") && !e.entry.includes("sweep")));
    assert.ok(f.events.every(e => (e.group as { epoch: number }).epoch === sweeps % 2));
   }
  }
  for (const bad of [-1, 0.5, NaN, Infinity]) assert.throws(() => f.extension.preparePair(f.encoder, f.groups, bad), /Invalid mixed extension sweep count/);
 }
});

test("paired scratch bounds and device limit preserve the scalar fallback", async () => {
 for (const [limit, bulk] of [[8, false], [9, true]] as const) {
  const f = await fixture(limit, 8, bulk);
  assert.equal(f.extension.supportsPair, false);
  assert.equal(f.codes.length, 1);
  assert.doesNotThrow(() => f.extension.bind(f.fields));
  assert.doesNotThrow(() => f.extension.encode(f.encoder, f.groups, 3));
  assert.throws(() => f.extension.bind({ ...f.fields, companion: { physical: f.texture(), negative: {} as GPUBuffer, scratch: { size: 1e6 } as GPUBuffer } }), /unsupported/);
 }
 const f = await fixture();
 const bytes = f.extension.companionBytesAt(8);
 const companion = { physical: f.texture(), negative: {} as GPUBuffer, scratch: { size: bytes - 1 } as GPUBuffer };
 assert.equal(bytes % 512, 0, "both companion epochs are aligned independently");
 assert.throws(() => f.extension.bind({ ...f.fields, companion }), /companion scratch is too small/);
 companion.scratch = { size: bytes } as GPUBuffer;
 assert.doesNotThrow(() => f.extension.bind({ ...f.fields, companion }));
});


test("slot snapshot publisher is guarded and consumers read finalized values only",async()=>{
 const f=await fixture();
 const scalar=f.codes[0]!.replace(/\b([A-Z][A-Z_0-9]*)\(\)/g,"$1");
 const pair=pairedExtensionSource(scalar),snapshot=pairedSnapshotSource(pair);
 assert.match(snapshot,/fn ueSnapshotStore/);
 assert.throws(()=>pairedSnapshotSource(pair.replace("value.w=textureLoad(physical,vec3i(origin),0).w;textureStore(output,vec3i(origin),value);","")),/expected 2 occurrences/);
 const reader=uniformMixedExtensionSnapshotWGSL(1,17);
 assert.doesNotMatch(reader,/stateIn|pairIn|origins|umFar|textureLoad/);
 assert.match(reader,/var<storage,read> ueSnapshotSlots/);
 const bytes=f.extension.companionBytesAt(8),buffer={size:bytes} as GPUBuffer;
 assert.deepEqual(f.extension.companionBindings(buffer).map(b=>[b.offset,b.size]),[[0,bytes/2],[bytes/2,bytes/2]]);
 assert.throws(()=>f.extension.companionBindings({size:bytes-1} as GPUBuffer),/too small/);
 assert.throws(()=>f.extension.bind({...f.fields,snapshot:f.texture()}),/requires paired extension/);
});


test("far publication replay requires a fresh matching snapshot and preserves scalar fallback",async()=>{
 for(const fineTiles of [0,8]){
  const f=await fixture(9,fineTiles),extension=f.extension as unknown as {
   regularPipelines:Map<string,GPUComputePipeline>;replayPublishers:{regular:Map<string,GPUComputePipeline>};
  };
  const publication=()=>f.events.filter(e=>e.entry==="publishFine").map(e=>e.pipeline);
  const expect=(replay:boolean)=>assert.ok(publication().every(p=>p===(replay?extension.replayPublishers.regular:extension.regularPipelines).get("publish")));
  for(const sweeps of [0,1,2,3]){
   f.extension.preparePair(f.encoder,f.groups,sweeps);f.extension.finishSnapshot(f.encoder,f.groups,sweeps);
   f.events.length=0;f.extension.finishPair(f.encoder,f.groups,sweeps);expect(true);
   f.events.length=0;f.extension.finishPair(f.encoder,f.groups,sweeps);expect(false);
   f.extension.finishSnapshot(f.encoder,f.groups,sweeps);
   f.events.length=0;f.extension.finishPair({...f.encoder},f.groups,sweeps);expect(false);
   f.extension.finishSnapshot(f.encoder,f.groups,sweeps);
   f.events.length=0;f.extension.finishPair(f.encoder,f.groups,sweeps+1);expect(false);
   f.extension.finishSnapshot(f.encoder,f.groups,sweeps);f.extension.preparePair(f.encoder,f.groups,sweeps);
   f.events.length=0;f.extension.finishPair(f.encoder,f.groups,sweeps);expect(false);
  }
 }
});

test("publication generators retain tagged selections and reject changed sampling paths",async()=>{
 const f=await fixture(),scalar=f.codes[0]!.replace(/\b([A-Z][A-Z_0-9]*)\(\)/g,"$1");
 const snapshot=pairedSnapshotSource(pairedExtensionSource(scalar,true),true),replay=replayedPublicationSource(scalar);
 assert.match(snapshot,/stateOut\[umSlot\(face.anchor,face.axis,face.width\)\]=vec2f\(f32\(mask\),-1.0\)/);
 assert.match(snapshot,/publicationRemember\(face,publicationSelection\)/);
 assert.match(replay,/if\(retained.y!=-1.0\)\{return umFarValue\(face\);\}/);
 assert.doesNotMatch(replay,/publicationStrict|publicationForceFallback/);
 assert.throws(()=>replayedPublicationSource(scalar.replace(" for(var item=lane;item<81u;item+=64u){"," for(var item=lane;item<80u;item+=64u){")),/far staging changed/);
 assert.throws(()=>pairedSnapshotSource(pairedExtensionSource(scalar.replace("sum+=values[k];count+=1.0;","sum=values[k];count+=1.0;"),true),true),/expected 1 occurrences/);
});
