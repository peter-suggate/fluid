import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createGeometricVolumeResidentWGSL, LIQUID_CAPACITY_BALANCING_ROUNDS,
  LIQUID_CAPACITY_RELATIVE_TOLERANCE, type SparseGeometricVolumeLayout } from "../lib/methods/adaptive-volume/resident-volume.wgsl";

type Edge = { receiver: number; donor: number; weight: number };
type Fixture = { name: string; capacity: number[]; receivers: number[]; volume: number[]; edges: Edge[] };
const INVALID = 0xffffffff;

// Independent f64 oracle over a dense list, including explicit sink edges.
function reference(f: Fixture) {
  const weights = f.edges.map(e => e.weight);
  const amounts = () => f.receivers.map((_, r) => f.edges.reduce((v, e, i) =>
    v + (e.receiver === r && f.capacity[e.donor]! > 0 ? weights[i]! * f.volume[e.donor]! / f.capacity[e.donor]! : 0), 0));
  for (let round = 0; round < LIQUID_CAPACITY_BALANCING_ROUNDS; round++) {
    const mass = amounts();
    if (mass.every((v, r) => v <= f.receivers[r]! * (1 + LIQUID_CAPACITY_RELATIVE_TOLERANCE))) break;
    f.edges.forEach((e, i) => {
      if (e.receiver !== INVALID && mass[e.receiver]! > f.receivers[e.receiver]! * (1 + LIQUID_CAPACITY_RELATIVE_TOLERANCE))
        weights[i]! *= f.receivers[e.receiver]! / mass[e.receiver]!;
    });
    const sums = f.capacity.map((_, d) => f.edges.reduce((v, e, i) => v + (e.donor === d ? weights[i]! : 0), 0));
    f.edges.forEach((e, i) => { if (sums[e.donor]! > 0) weights[i]! *= f.capacity[e.donor]! / sums[e.donor]!; });
  }
  return { weights, amounts: amounts() };
}

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("liquid remap balancing preserves donor mass, sinks and stencil; capped excess stays observable", { timeout: 60_000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "liquid receiver balancing");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
    const gpu = dawn.create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice(); assert.ok(device);
    const production = createGeometricVolumeResidentWGSL({} as SparseGeometricVolumeLayout);
    const kernels = production.slice(production.indexOf("fn gvNormalizeReceiver"), production.indexOf("@compute @workgroup_size(64)\nfn validateWholeFrameVolume"));
    const module = device.createShaderModule({ code: `
@group(0)@binding(0)var<storage,read_write>state:array<f32>;
@group(0)@binding(1)var<storage,read_write>conditioning:array<atomic<i32>>;
const INVALID=0xffffffffu;const GV_CURRENT=0u;const GV_LOW=16u;
const GV_EDGE_A=128u;const GV_EDGE_B=256u;const GV_EDGE_CAPACITY=128u;
const GV_WHOLE_FRAME_CONTROL=0u;const GV_RECEIVER_HEADS=64u;const GV_DONOR_HEADS=80u;
const GV_LIQUID_BALANCE_TOLERANCE:f32=${LIQUID_CAPACITY_RELATIVE_TOLERANCE};
fn gvFailed()->bool{return atomicLoad(&conditioning[63])!=0;}
fn gvFault(a:u32,b:u32,c:f32,d:f32,e:f32){_=a;_=b;_=c;_=d;_=e;atomicStore(&conditioning[63],1);}
fn acceptedTemplateCellInvocation(i:u32)->u32{return select(INVALID,i,i<bitcast<u32>(state[64]));}
fn gvDonorCapacity(c:u32)->f32{return state[32u+c];}
fn gvReceiverCapacity(c:u32)->f32{return state[48u+c];}
fn gvEdgeReceiver(e:u32)->u32{return bitcast<u32>(state[512u+4u*e]);}
fn gvEdgeDonor(e:u32)->u32{return bitcast<u32>(state[513u+4u*e]);}
fn gvEdgeReceiverNext(e:u32)->u32{return bitcast<u32>(state[514u+4u*e]);}
fn gvEdgeDonorNext(e:u32)->u32{return bitcast<u32>(state[515u+4u*e]);}
fn gvRecordOutflow(v:f32){state[65]+=v;}
` + kernels });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(m => m.type === "error"), []);
    const layout = device.createBindGroupLayout({ entries: [0, 1].map(binding => ({ binding,
      visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" as const } })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const entries = ["balanceWholeFrameLiquidReceivers", "balanceWholeFrameLiquidDonors", "finishWholeFrameLiquidBalanceRound",
      "auditWholeFrameVolumeMarginals", "gatherWholeFrameVolume"];
    const pipelines = entries.map(entryPoint => device!.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } }));
    const pair = [{ receiver: 0, donor: 0, weight: .9 }, { receiver: 1, donor: 0, weight: .1 },
      { receiver: 0, donor: 1, weight: .9 }, { receiver: 1, donor: 1, weight: .1 }];
    const fixtures: Fixture[] = [
      { name: "feasible crowding", capacity: [1, 1], receivers: [1, 1], volume: [1, 1], edges: pair },
      { name: "moving capacities", capacity: [1, 1], receivers: [.4, 1.6], volume: [1, 1], edges: pair },
      { name: "infeasible stencil", capacity: [1, 1], receivers: [.4, .4], volume: [1, 1], edges: pair },
      { name: "underfilled identity", capacity: [1, 1], receivers: [1, 1], volume: [.2, 0], edges: pair },
      { name: "zero-capacity dry donor", capacity: [1, 0], receivers: [1, 0], volume: [.2, 0], edges: [{ receiver: 0, donor: 0, weight: 1 }] },
      { name: "open sink", capacity: [1], receivers: [.25], volume: [1], edges: [
        { receiver: 0, donor: 0, weight: .9 }, { receiver: INVALID, donor: 0, weight: .1 }] },
    ];
    for (const fixture of fixtures) for (const reverse of [false, true]) {
      const f = { ...fixture, edges: reverse ? [...fixture.edges].reverse() : fixture.edges };
      const values = new Float32Array(1024), words = new Uint32Array(values.buffer), controls = new Int32Array(96);
      values.set(f.volume, 0); values.set(f.capacity, 32); values.set(f.receivers, 48); words[64] = f.capacity.length;
      controls.fill(-1, 64); controls[34] = 1;
      f.edges.forEach((e, i) => {
        words.set([e.receiver, e.donor, e.receiver === INVALID ? INVALID : controls[64 + e.receiver]!, controls[80 + e.donor]!], 512 + 4 * i);
        if (e.receiver !== INVALID) controls[64 + e.receiver] = i;
        controls[80 + e.donor] = i; values[128 + i] = e.weight;
      });
      const state = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const control = device.createBuffer({ size: controls.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const readback: GPUBuffer = device.createBuffer({ size: values.byteLength + controls.byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      device.queue.writeBuffer(state, 0, values); device.queue.writeBuffer(control, 0, controls);
      const bindings = device.createBindGroup({ layout, entries: [state, control].map((buffer, binding) => ({ binding, resource: { buffer } })) });
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass(); pass.setBindGroup(0, bindings);
      const dispatch = (index: number) => { pass.setPipeline(pipelines[index]!); pass.dispatchWorkgroups(1); };
      for (let round = 0; round < LIQUID_CAPACITY_BALANCING_ROUNDS; round++) { dispatch(0); dispatch(1); dispatch(2); }
      dispatch(3); dispatch(4); pass.end();
      encoder.copyBufferToBuffer(state, 0, readback, 0, values.byteLength);
      encoder.copyBufferToBuffer(control, 0, readback, values.byteLength, controls.byteLength);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const result: Float32Array = new Float32Array(readback.getMappedRange());
      const receipt: Int32Array = new Int32Array(result.buffer, values.byteLength);
      assert.equal(receipt[63], 0, f.name);
      const oracle = reference(f);
      f.edges.forEach((_, i) => assert.ok(Math.abs(result[128 + i]! - oracle.weights[i]!) < 5e-6, `${f.name} weight ${i}`));
      f.capacity.forEach((capacity, d) => {
        const total = f.edges.reduce((v, e, i) => v + (e.donor === d ? result[128 + i]! : 0), 0);
        assert.ok(Math.abs(total - capacity) < 2e-6, `${f.name} donor ${d} conservation`);
      });
      const amounts = Array.from(result.slice(16, 16 + f.receivers.length));
      const maxExcess = Math.max(...amounts.map((v, r) => Math.max(0, v - f.receivers[r]!) / Math.max(f.receivers[r]!, 1e-30)));
      assert.ok(Math.abs(result[1024 + 38]! - maxExcess) < 1e-6, `${f.name} fresh final receipt`);
      if (f.name === "infeasible stencil") { assert.equal(receipt[36], 64); assert.ok(maxExcess > 1); }
      else assert.ok(maxExcess < 2e-6, `${f.name} capacity`);
      if (f.name === "underfilled identity") {
        assert.equal(receipt[36], 0);
        f.edges.forEach((e, i) => assert.equal(result[128 + i], Math.fround(e.weight)));
      }
      // Sink volume is included in the conservation sum, not discarded.
      const outflow = f.edges.reduce((v, e, i) => v + (e.receiver === INVALID ? result[128 + i]! * f.volume[e.donor]! / f.capacity[e.donor]! : 0), 0);
      assert.ok(Math.abs(amounts.reduce((a, b) => a + b, outflow) - f.volume.reduce((a, b) => a + b, 0)) < 3e-6, `${f.name} total material`);
      readback.unmap(); readback.destroy(); state.destroy(); control.destroy();
    }
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
