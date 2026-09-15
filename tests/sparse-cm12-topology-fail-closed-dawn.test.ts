import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { CM12_FAILURE_WORDS, decodeCM12SimulationFailure } from "../lib/methods/adaptive-volume/sparse-cm12-simulation-failure";
import { cm12SimulationFailureWGSL, guardCM12SimulationDispatches } from "../lib/methods/adaptive-volume/sparse-cm12-simulation-failure.wgsl";

const live = new Set<GPU>();
(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("missing compiled faces halt publication without requesting repair", async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "cm12-topology-fail-closed");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
    Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    live.add(gpu);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice();
    assert.ok(device);
    const source = readFileSync(new URL(
      "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url), "utf8");
    // Execute the production rejection and sealing kernels. The fixture supplies
    // a single material cell with either complete coverage or a missing Y face.
    const kernel = (name: string) => source.match(new RegExp(
      `@compute @workgroup_size\\(64\\)\\nfn ${name}\\([\\s\\S]*?\\n}`))![0];
    const code = guardCM12SimulationDispatches(`
@group(0)@binding(0)var<storage,read_write>topologyArena:array<atomic<u32>>;
@group(0)@binding(1)var<storage,read_write>activity:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read_write>conditioning:array<atomic<i32>>;
struct Params { failure:vec4u, dispatch:vec4u }
@group(0)@binding(3)var<uniform>p:Params;
const GEOMETRIC_TOPOLOGY_BACKING_MISSING=34u;
const CM12_WDR_INITIAL_LEAVES=1u;const INVALID=0xffffffffu;
fn cm12WorldLeafCoordinate(brick:u32)->vec3i{return vec3i(0);}
fn cm12PreparedDynamicFace(brick:u32,r:u32,side:u32,point:vec3f)->vec4u{return vec4u(0u);}
var<workgroup>geometricBrickBackingMissing:atomic<u32>;
var<workgroup>geometricBrickCertificationEnabled:u32;
fn cm12FCCandidateGeneration()->u32{return 17u;}
${cm12SimulationFailureWGSL}
fn scheduledBrickActive(brick:u32)->bool{return atomicLoad(&activity[1])!=0u;}
fn brickActive(brick:u32)->bool{return true;}
fn scheduledBrickResolution(brick:u32)->u32{return 8u;}
fn acceptedBrickResolution(brick:u32)->u32{return 4u;}
fn templateBrickCellRange(brick:u32,resolution:u32)->vec2u{return vec2u(165060u,1u);}
fn geometricScheduledCellHasMaterial(cell:u32,brick:u32)->bool{return true;}
fn geometricScheduledCellMissingFace(cell:u32)->u32{return atomicLoad(&activity[2]);}
fn activityRecord(brick:u32)->u32{return 0u;}
fn setTopologyPreparationScheduled(record:u32,enabled:bool){atomicStore(&activity[1],select(0u,1u,enabled));}
${kernel("certifyGeometricTopologyFaces")}
${kernel("sealGeometricTopologyFaces")}
@compute @workgroup_size(1) fn publish(){atomicAdd(&topologyArena[0],1u);}
`);
    const module = device.createShaderModule({ code });
    assert.deepEqual((await module.getCompilationInfo()).messages.filter(m => m.type === "error"), []);
    const storage = (size: number) => device!.createBuffer({ size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const topology = storage(4 * (CM12_FAILURE_WORDS + 1));
    const activity = storage(4 * 48);
    const conditioning = storage(4 * 36);
    const params = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const pipelines = new Map(["certifyGeometricTopologyFaces", "sealGeometricTopologyFaces", "publish"].map(entryPoint => {
      const pipeline = device!.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
      const buffers = entryPoint === "publish" ? [[0, topology], [3, params]] as const
        : [[0, topology], [1, activity], [2, conditioning], [3, params]] as const;
      // The sealing entry does not use the failure recorder's topology storage.
      const used = entryPoint === "sealGeometricTopologyFaces" ? buffers.filter(([binding]) => binding !== 0) : buffers;
      const bindings = device!.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: used.map(([binding, buffer]) => ({ binding, resource: { buffer } })) });
      return [entryPoint, { pipeline, bindings }] as const;
    }));
    const readback = device.createBuffer({ size: topology.size + conditioning.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    for (const missingAxis of [0, 2]) {
      device.queue.writeBuffer(params, 0, new Uint32Array([0, 0, 0, 0, 0, 0, 0, 1]));
      const activityWords = new Uint32Array(48);
      activityWords[0] = 6; activityWords[1] = 1; activityWords[2] = missingAxis; activityWords[16] = 1;
      device.queue.writeBuffer(activity, 0, activityWords);
      const encoder = device.createCommandEncoder();
      encoder.clearBuffer(topology); encoder.clearBuffer(conditioning);
      for (const entry of ["certifyGeometricTopologyFaces", "sealGeometricTopologyFaces", "publish", "publish"]) {
        // Certification and sealing run together; the publication tail and
        // subsequent frame must both observe the sticky failure gate.
        if (entry === "publish") encoder.copyBufferToBuffer(topology, 4, params, 0, 4);
        const { pipeline, bindings } = pipelines.get(entry)!;
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(1); pass.end();
      }
      encoder.copyBufferToBuffer(topology, 0, readback, 0, topology.size);
      encoder.copyBufferToBuffer(conditioning, 0, readback, topology.size, conditioning.size);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(readback.getMappedRange()).slice(); readback.unmap();
      const failure = decodeCM12SimulationFailure(words.slice(1, 1 + CM12_FAILURE_WORDS));
      assert.equal(words[1 + CM12_FAILURE_WORDS + 33], 0, "no host repair request");
      if (missingAxis === 0) { assert.equal(words[0], 2); assert.equal(failure, undefined); }
      else {
        assert.equal(words[0], 0, "accepted publication remains untouched");
        assert.equal(failure?.code, "MISSING_COMPILED_TOPOLOGY_FACE");
        assert.equal(failure?.ownerId, 165060);
        assert.deepEqual(failure?.operands, [0, 1, 4, 8]);
      }
    }
    for (const buffer of [topology, activity, conditioning, params, readback]) buffer.destroy();
  } finally { device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock(); }
});
