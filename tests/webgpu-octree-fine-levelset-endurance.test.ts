import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { packFineLevelSetBrickKey,
  planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { WebGPUFineLevelSetBricks, type WebGPUFineLevelSetBrickSource } from
  "../lib/webgpu-octree-fine-levelset-bricks";
import { WebGPUFineLevelSetTopology } from "../lib/webgpu-octree-fine-levelset-topology";
import { PassBroker } from "../lib/webgpu-pass-broker";

const topologyEnduranceWGSL = /* wgsl */ `
const INVALID:u32=0xffffffffu;
struct Params { expectedGeneration:u32,pageCapacity:u32,logicalBricks:u32,pad0:u32 }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> metadata:array<u32>;
@group(0) @binding(2) var<storage,read> worklist:array<u32>;
@group(0) @binding(3) var<storage,read> topologyControl:array<u32>;
@group(0) @binding(4) var<storage,read_write> endurance:array<u32>;
fn lookup(key:u32)->u32{
 let count=min(worklist[0],params.pageCapacity);var low=0u;var high=count;
 for(var step=0u;step<32u&&low<high;step+=1u){let middle=low+(high-low)/2u;let id=worklist[5u+middle];let base=id*10u;
  if(id>=params.pageCapacity||base+2u>=arrayLength(&metadata)||metadata[base]!=id||metadata[base+2u]!=params.expectedGeneration){return INVALID;}
  if(metadata[base+1u]<key){low=middle+1u;}else{high=middle;}}
 if(low>=count){return INVALID;}let id=worklist[5u+low];let base=id*10u;
 return select(INVALID,id,metadata[base+1u]==key);
}
@compute @workgroup_size(1) fn validateGeneration(){
 var errors=topologyControl[0];if(topologyControl[4]!=1u){errors|=16u;}
 let count=worklist[0];if(count>params.pageCapacity){errors|=32u;}
 if(worklist[1]!=params.expectedGeneration||worklist[3]!=1u||worklist[4]!=1u
   ||worklist[2]!=(count+63u)/64u||5u+count>arrayLength(&worklist)){errors|=64u;}
 var previous=INVALID;
 for(var i=0u;i<count;i+=1u){let id=worklist[5u+i];if(id>=params.pageCapacity){errors|=2048u;continue;}
  let base=id*10u;let key=metadata[base+1u];
  if(key>=params.logicalBricks||metadata[base]!=id||metadata[base+2u]!=params.expectedGeneration
    ||metadata[base+3u]!=1u||lookup(key)!=id){errors|=4096u;}
  if(i>0u&&key<=previous){errors|=1024u;}previous=key;
  for(var j=i+1u;j<count;j+=1u){if(worklist[5u+j]==id){errors|=8192u;}}}
 if(errors!=0u&&endurance[5]==0u){endurance[5]=params.expectedGeneration;endurance[6]=topologyControl[0];endurance[7]=(worklist[1]<<16u)|min(count,65535u);}
 endurance[0]|=errors;endurance[1]+=1u;endurance[2]=min(endurance[2],count);endurance[3]=max(endurance[3],count);endurance[4]=params.expectedGeneration;
}
`;

function generationBindGroup(device: GPUDevice, pipeline: GPUComputePipeline, params: GPUBuffer,
  target: WebGPUFineLevelSetBrickSource, topology: WebGPUFineLevelSetTopology, endurance: GPUBuffer): GPUBindGroup {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: params } }, { binding: 1, resource: { buffer: target.metadata } },
    { binding: 2, resource: { buffer: target.worklist } },
    { binding: 3, resource: { buffer: topology.control } }, { binding: 4, resource: { buffer: endurance } },
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
    const topologyAB = new WebGPUFineLevelSetTopology(device, sourceA, sourceB, coarseWGSL);
    const topologyBA = new WebGPUFineLevelSetTopology(device, sourceB, sourceA, coarseWGSL);
    const seeds = device.createBuffer({ label: "moving fine interface seeds", size: (2 + plan.maximumResidentBricks) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const validationParams = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const endurance = device.createBuffer({ size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(endurance, 0, new Uint32Array([0, 0, 0xffff_ffff, 0, 0, 0, 0, 0]));
    const groupAB = generationBindGroup(device, validationPipeline, validationParams, sourceB, topologyAB, endurance);
    const groupBA = generationBindGroup(device, validationPipeline, validationParams, sourceA, topologyBA, endurance);
    const stableBuffers = [owner.flags, owner.phi, owner.workA, owner.workB, ...owner.metadata,
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
      device.queue.writeBuffer(validationParams, 0, new Uint32Array([
        generation, plan.maximumResidentBricks, plan.logicalBrickCount, 0,
      ]));
      const topology = currentIsA ? topologyAB : topologyBA; const group = currentIsA ? groupAB : groupBA;
      const encoder = device.createCommandEncoder({ label: `fine topology endurance factor ${factor} frame ${frame}` });
      topology.encode(new PassBroker(encoder), { buffer: seeds }, [{ binding: 9, resource: { buffer: plane } }]);
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
    assert.deepEqual([owner.flags, owner.phi, owner.workA, owner.workB, ...owner.metadata,
      ...owner.worklists, ...owner.params], stableBuffers, "resident pool buffers changed during churn");
    readback.destroy(); endurance.destroy(); validationParams.destroy(); seeds.destroy();
    topologyBA.destroy(); topologyAB.destroy(); plane.destroy(); owner.destroy();
  }
  device.destroy();
});
