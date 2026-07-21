import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { FineLevelSetBrickOracle, packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks, type WebGPUFineLevelSetBrickSource } from
  "../lib/webgpu-octree-fine-levelset-bricks";
import { WebGPUFineLevelSetRedistance,
  unpackFineLevelSetGPURedistanceControl } from "../lib/webgpu-octree-fine-levelset-redistance";
import { WebGPUFineLevelSetTopology } from "../lib/webgpu-octree-fine-levelset-topology";
import { WebGPUFineLevelSetTransport,
  unpackFineLevelSetGPUTransportControl } from "../lib/webgpu-octree-fine-levelset-transport";
import { WebGPUOctreePowerFaces } from "../lib/webgpu-octree-power-faces";
import { OCTREE_POWER_TOPOLOGY_VALID,
  WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import { WebGPUOctreePowerVelocity } from "../lib/webgpu-octree-power-velocity";
import { WebGPUOctreePowerVelocityPrepass } from "../lib/webgpu-octree-power-velocity-prepass";

const topologyEnduranceWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params { expectedGeneration:u32,pageCapacity:u32,hashCapacity:u32,maximumHashProbes:u32,logicalBricks:u32,pad0:u32,pad1:u32,pad2:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> metadata:array<u32>;
@group(0) @binding(2) var<storage,read> worklist:array<u32>;
@group(0) @binding(3) var<storage,read> hashTable:array<u32>;
@group(0) @binding(4) var<storage,read> topologyControl:array<u32>;
@group(0) @binding(5) var<storage,read_write> endurance:array<u32>;
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(params.hashCapacity-1u);}
fn lookup(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){
 if(probe>=params.maximumHashProbes){break;}let slot=(start+probe)&(params.hashCapacity-1u);let stored=hashTable[slot*2u];
 if(stored==key){return hashTable[slot*2u+1u];}if(stored==INVALID){return INVALID;}}return INVALID;}
@compute @workgroup_size(1) fn validateGeneration(){
 var errors=topologyControl[0];if(topologyControl[4]!=1u){errors|=16u;}
 let count=worklist[0];if(count>params.pageCapacity){errors|=32u;}
 if(worklist[1]!=params.expectedGeneration){errors|=64u;}
 var occupied=0u;
 for(var slot=0u;slot<params.hashCapacity;slot+=1u){let key=hashTable[slot*2u];let id=hashTable[slot*2u+1u];
  if(key==INVALID){if(id!=INVALID){errors|=128u;}continue;}occupied+=1u;
  if(key>=params.logicalBricks||id>=params.pageCapacity){errors|=256u;continue;}
  let base=id*10u;if(metadata[base]!=id||metadata[base+1u]!=key||metadata[base+2u]!=params.expectedGeneration||metadata[base+3u]!=1u){errors|=512u;}}
 if(occupied!=count){errors|=1024u;}
 for(var i=0u;i<count;i+=1u){let id=worklist[5u+i];if(id>=params.pageCapacity){errors|=2048u;continue;}
  let base=id*10u;let key=metadata[base+1u];if(metadata[base+2u]!=params.expectedGeneration||lookup(key)!=id){errors|=4096u;}
  for(var j=i+1u;j<count;j+=1u){if(worklist[5u+j]==id){errors|=8192u;}}}
 if(errors!=0u&&endurance[5]==0u){endurance[5]=params.expectedGeneration;endurance[6]=topologyControl[0];endurance[7]=(worklist[1]<<16u)|min(count,65535u);}
 endurance[0]|=errors;endurance[1]+=1u;endurance[2]=min(endurance[2],count);endurance[3]=max(endurance[3],count);endurance[4]=params.expectedGeneration;
}
`;

function generationBindGroup(device: GPUDevice, pipeline: GPUComputePipeline, params: GPUBuffer,
  target: WebGPUFineLevelSetBrickSource, topology: WebGPUFineLevelSetTopology, endurance: GPUBuffer): GPUBindGroup {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: target.metadata } },
    { binding: 2, resource: { buffer: target.worklist } }, { binding: 3, resource: { buffer: target.hash } },
    { binding: 4, resource: { buffer: topology.control } }, { binding: 5, resource: { buffer: endurance } },
  ] });
}

// This is deliberately a topology-lifetime gate: analytic slab fallback seeds
// phi, but no transport or redistance operator runs here. The separately gated
// test below owns transported-payload endurance.
test("Dawn factor-4/factor-8 topology-only A/B generations grow, shrink, and churn 300 frames", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter); const device = await adapter.requestDevice();
  const shaderModule = device.createShaderModule({ label: "fine topology endurance validator", code: topologyEnduranceWGSL });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const validationPipeline = device.createComputePipeline({ label: "validate fine topology generation", layout: "auto",
    compute: { module: shaderModule, entryPoint: "validateGeneration" } });

  for (const factor of [4, 8] as const) {
    const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [8, 2, 2],
      finestCellWidth: 1, fineFactor: factor, brickResolution: 4,
      maximumResidentBricks: factor === 4 ? 32 : 256 });
    const owner = new WebGPUFineLevelSetBricks(device, plan);
    const sourceA = owner.initializeEmptyGPUGeneration(1); const sourceB = owner.prepareGPUGeneration(2);
    const plane = device.createBuffer({ label: "moving constant-volume slab", size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const coarseWGSL = `@group(0) @binding(9) var<uniform> slab:vec4f;
fn sampleCoarseOctreePhi(position:vec3f)->f32{return max(slab.x-position.x,position.x-slab.y);}`;
    const topologyAB = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, coarseWGSL, true);
    const topologyBA = new WebGPUFineLevelSetTopology(device, sourceB, sourceA, coarseWGSL, true);
    const seeds = device.createBuffer({ label: "moving fine interface seeds", size: (2 + plan.maximumResidentBricks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const validationParams = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const endurance = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(endurance, 0, new Uint32Array([0, 0, 0xffff_ffff, 0, 0, 0, 0, 0]));
    const groupAB = generationBindGroup(device, validationPipeline, validationParams, sourceB, topologyAB, endurance);
    const groupBA = generationBindGroup(device, validationPipeline, validationParams, sourceA, topologyBA, endurance);
    const stableBuffers = [owner.flags, owner.phi, owner.workA, owner.workB, ...owner.hashes, ...owner.metadata,
      ...owner.worklists, ...owner.params];

    let currentIsA = true; let generation = 1;
    for (let frame = 0; frame < 300; frame += 1) {
      generation += 1;
      const target = currentIsA ? sourceB : sourceA;
      if (frame > 0) owner.repurposeGPUGeneration(target, generation);
      const left = 2 + 2 * ((frame % 60) / 59); const right = left + 2;
      device.queue.writeBuffer(plane, 0, new Float32Array([left, right, 0, 0]));
      const brickWidth = plan.brickResolution * plan.fineCellWidth;
      const seedKeys: number[] = [];
      for (let z = 0; z < plan.brickDimensions[2]; z += 1) for (let y = 0; y < plan.brickDimensions[1]; y += 1) {
        for (const boundary of [left, right]) {
          const x = Math.min(plan.brickDimensions[0] - 1, Math.floor(boundary / brickWidth));
          seedKeys.push(packFineLevelSetBrickKey(plan, [x, y, z]));
        }
        // Every third 20-frame epoch grows a remote safety patch. Omitting it
        // in the following epoch forces actual old-only page retirement.
        if (Math.floor(frame / 20) % 3 === 1) {
          seedKeys.push(packFineLevelSetBrickKey(plan, [plan.brickDimensions[0] - 1, y, z]));
        }
      }
      const uniqueSeeds = [...new Set(seedKeys)];
      assert.ok(uniqueSeeds.length <= plan.maximumResidentBricks);
      device.queue.writeBuffer(seeds, 0, new Uint32Array([uniqueSeeds.length, 0, ...uniqueSeeds]));
      device.queue.writeBuffer(validationParams, 0, new Uint32Array([generation, plan.maximumResidentBricks,
        plan.hashCapacity, plan.maximumHashProbes, plan.logicalBrickCount, 0, 0, 0]));
      const topology = currentIsA ? topologyAB : topologyBA; const group = currentIsA ? groupAB : groupBA;
      const encoder = device.createCommandEncoder({ label: `fine topology endurance factor ${factor} frame ${frame}` });
      topology.encode(encoder, { buffer: seeds }, [{ binding: 9, resource: { buffer: plane } }]);
      const pass = encoder.beginComputePass({ label: "Latch fine topology invariants" });
      pass.setPipeline(validationPipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
      device.queue.submit([encoder.finish()]); currentIsA = !currentIsA;
    }
    await device.queue.onSubmittedWorkDone();
    const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(endurance, 0, readback, 0, 32);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await readback.mapAsync(GPUMapMode.READ); const result = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
    assert.equal(result[0], 0, `factor ${factor} latched topology error mask 0x${result[0].toString(16)}; first generation ${result[5]}, topology 0x${result[6].toString(16)}, worklist 0x${result[7].toString(16)}`);
    assert.equal(result[1], 300); assert.equal(result[4], 301);
    assert.ok(result[2] > 0 && result[3] > result[2],
      `factor ${factor} topology did not both shrink and grow (${result[2]}..${result[3]})`);
    assert.deepEqual([owner.flags, owner.phi, owner.workA, owner.workB, ...owner.hashes, ...owner.metadata,
      ...owner.worklists, ...owner.params], stableBuffers, "resident pool buffers changed during churn");
    readback.destroy(); endurance.destroy(); validationParams.destroy(); seeds.destroy();
    topologyBA.destroy(); topologyAB.destroy(); plane.destroy(); owner.destroy();
  }
  device.destroy();
});

test("Dawn factor-4 transported phi survives 300 A/B topology, Stage-B transport, and redistance generations", {
  skip: !process.env.WEBGPU_NODE_MODULE
    ? "set WEBGPU_NODE_MODULE"
    : process.env.FLUID_FINE_TRANSPORT_ENDURANCE !== "1"
      && "set FLUID_FINE_TRANSPORT_ENDURANCE=1 after the Section-5 velocity coverage gate passes",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice();

  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [2, 2, 2],
    finestCellWidth: 1, fineFactor: 4, brickResolution: 4, maximumResidentBricks: 8 });
  const oracle = new FineLevelSetBrickOracle(plan);
  const allKeys = Array.from({ length: plan.logicalBrickCount }, (_, key) => key);
  oracle.publishInterfaceAndRing(allKeys, ([x]) => x - 1);
  const owner = new WebGPUFineLevelSetBricks(device, plan);
  const sourceA = owner.uploadGeneration(oracle.exportGPUGeneration());
  const sourceB = owner.prepareGPUGeneration(2);
  const coarseWGSL = "fn sampleCoarseOctreePhi(position:vec3f)->f32{return position.x-1.0;}";
  const topologyAB = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, coarseWGSL);
  const topologyBA = new WebGPUFineLevelSetTopology(device, sourceB, sourceA, coarseWGSL);

  const raw = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  const powerTopology = new WebGPUOctreePowerTopology(device, 8, catalog);
  const powerFaces = new WebGPUOctreePowerFaces(device, 8, 64, powerTopology.source);
  const powerVelocity = new WebGPUOctreePowerVelocity(device, 8);
  const prepass = new WebGPUOctreePowerVelocityPrepass(device,
    plan.maximumResidentBricks * plan.samplesPerBrick, powerTopology.source, powerFaces.source);
  const transportA = new WebGPUFineLevelSetTransport(device, sourceA, prepass);
  const transportB = new WebGPUFineLevelSetTransport(device, sourceB, prepass);
  const redistanceA = new WebGPUFineLevelSetRedistance(device, sourceA);
  const redistanceB = new WebGPUFineLevelSetRedistance(device, sourceB);

  const entry = catalog.sameOrFinerDirect[0x3ffff] & 0xffff;
  const metrics = new Uint32Array(8 * 4), headers = new Uint32Array(8 * 12);
  const velocities = new Float32Array(8 * 4), siteIndex = new Uint32Array(powerFaces.plan.hashCapacity * 4);
  const hash = (cell: number, size: number) => {
    let value = (cell ^ Math.imul(size, 0x9e3779b9)) >>> 0;
    value = Math.imul((value ^ (value >>> 16)) >>> 0, 0x7feb352d) >>> 0;
    value = Math.imul((value ^ (value >>> 15)) >>> 0, 0x846ca68b) >>> 0;
    return (value ^ (value >>> 16)) >>> 0;
  };
  for (let row = 0; row < 8; row += 1) {
    metrics.set([entry, OCTREE_POWER_TOPOLOGY_VALID, 0, 0], row * 4);
    headers[row * 12] = row; headers[row * 12 + 3] = 1;
    // The x=0 layer is a stationary extrapolation guard. The x=1 layer
    // translates the interface slowly enough to remain inside the complete
    // resident support throughout all 300 production transport steps.
    velocities.set([(row & 1) === 0 ? 0 : 0.25, 0, 0, 1], row * 4);
    let slot = hash(row, 1) & (powerFaces.plan.hashCapacity - 1);
    while (siteIndex[slot * 4] !== 0) slot = (slot + 1) & (powerFaces.plan.hashCapacity - 1);
    siteIndex.set([row + 1, 1, row, 0], slot * 4);
  }
  device.queue.writeBuffer(powerTopology.metrics, 0, metrics);
  device.queue.writeBuffer(powerFaces.siteIndex, 0, siteIndex);
  device.queue.writeBuffer(powerVelocity.velocities, 0, velocities);
  const headerBuffer = device.createBuffer({ size: headers.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(headerBuffer, 0, headers);

  let currentIsA = true; let generation = 1; let latest = sourceA;
  let latestTransport = transportA; let latestRedistance = redistanceA;
  for (let frame = 0; frame < 300; frame += 1) {
    generation += 1;
    const target = currentIsA ? sourceB : sourceA;
    if (frame > 0) owner.repurposeGPUGeneration(target, generation);
    const topology = currentIsA ? topologyAB : topologyBA;
    const transport = currentIsA ? transportB : transportA;
    const redistance = currentIsA ? redistanceB : redistanceA;
    const encoder = device.createCommandEncoder({ label: `factor-4 transported-phi endurance ${frame}` });
    topology.encode(encoder);
    transport.encode(encoder, { timestep: 0.002, headers: headerBuffer,
      rowVelocities: powerVelocity.velocities, dimensions: [2, 2, 2], physicalCellSize: 1,
      maximumLeafSize: 1, generation });
    redistance.encode(encoder, { bandCells: 4, residualTolerance: 1 });
    device.queue.submit([encoder.finish()]);
    currentIsA = !currentIsA; latest = target; latestTransport = transport; latestRedistance = redistance;
  }
  await device.queue.onSubmittedWorkDone();

  const transportBytes = 32, redistanceBytes = 16;
  const readback = device.createBuffer({ size: transportBytes + redistanceBytes + plan.payloadCapacityBytes / 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(latestTransport.control, 0, readback, 0, transportBytes);
  encoder.copyBufferToBuffer(latestRedistance.control, 0, readback, transportBytes, redistanceBytes);
  encoder.copyBufferToBuffer(latest.phi, 0, readback, transportBytes + redistanceBytes, plan.payloadCapacityBytes / 4);
  device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ); const bytes = readback.getMappedRange().slice(0); readback.unmap();
  const transport = unpackFineLevelSetGPUTransportControl(new Uint32Array(bytes, 0, transportBytes / 4));
  const redistance = unpackFineLevelSetGPURedistanceControl(new Uint32Array(bytes, transportBytes, redistanceBytes / 4));
  assert.equal(transport.committed, true); assert.equal(transport.departureOutsideBand, 0);
  assert.equal(transport.nonfiniteVelocity, 0); assert.ok(transport.processed > 0);
  assert.equal(redistance.committed, true); assert.equal(redistance.unresolvedCells, 0);
  const phi = new Float32Array(bytes, transportBytes + redistanceBytes, plan.payloadCapacityBytes / 16);
  assert.ok([...phi].every(Number.isFinite)); assert.ok([...phi].some((value) => value < 0));
  assert.ok([...phi].some((value) => value > 0)); assert.equal(generation, 301);

  readback.destroy(); headerBuffer.destroy(); redistanceB.destroy(); redistanceA.destroy();
  transportB.destroy(); transportA.destroy(); prepass.destroy(); powerVelocity.destroy();
  powerFaces.destroy(); powerTopology.destroy(); topologyBA.destroy(); topologyAB.destroy(); owner.destroy(); device.destroy();
});
