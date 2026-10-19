import assert from "node:assert/strict";
import test from "node:test";
import { createRegionTierScene } from "../tools/region-tier-scenes";
import { createRigidCouplingOracleScene } from "../lib/core/scenes";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { UNIFORM_PAGE_TABLE_WORDS } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformMixedPressureStorage } from "../lib/methods/uniform/uniform-mixed-pressure-boundary.wgsl";
import type { UniformMixedPressureCycles } from "../lib/methods/uniform/uniform-mixed-pressure-cycles";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

function pressure(solver: WebGPUUniformReferenceSolver) {
  return (solver as unknown as { mixedFrame: { cycles: UniformMixedPressureCycles } }).mixedFrame.cycles;
}
async function read(device: GPUDevice, range: GPUBufferBinding) {
  const size = range.size ?? range.buffer.size - (range.offset ?? 0);
  const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const e = device.createCommandEncoder(); e.copyBufferToBuffer(range.buffer, range.offset ?? 0, staging, 0, size);
    device.queue.submit([e.finish()]); await staging.mapAsync(GPUMapMode.READ);
    const words = new Uint32Array(staging.getMappedRange()).slice(); staging.unmap(); return words;
  } finally { staging.destroy(); }
}
/** Compare every hierarchy word with an unconditional rebuild from the same
 * inputs, not just a convergence statistic or a few sampled coefficients. */
async function audit(device: GPUDevice, cycles: UniformMixedPressureCycles, label: string, geometryHeld = false) {
  cycles.retainRegionalSetup = true;
  const native = (cycles as unknown as { resourceEntries: GPUBindGroupEntry[] }).resourceEntries.find(e => e.binding === 1)!.resource as GPUBufferBinding;
  const run = (retain: boolean) => {
    cycles.batchSetup = retain; cycles.retainSetup = retain; const e = device.createCommandEncoder(); cycles.encodeSetup(e, geometryHeld); device.queue.submit([e.finish()]);
  };
  run(true); const candidate = await read(device, native), work = await read(device, cycles.setupWork);
  run(false); const reference = await read(device, native); cycles.retainSetup = true;
  assert.deepEqual(candidate, reference, label);
  return [...work];
}

test("retained pressure hierarchy propagates exact geometry, neighbour and page changes", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("retained pressure setup", async device => {
    const scene = createRegionTierScene("rest", 128);
    // Odd capacity exercises partial workgroups and cell/wall boundaries on
    // the small levels, plus a free block alongside the stored pages.
    scene.fluid.pagePool = { blocks: 11 };
    const solver = await createUniformSolver(device, scene);
    try {
      assert.ok(solver.advanceTo(1 / 60, [])); await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
      const cycles = pressure(solver), root = cycles.level;
      await audit(device, cycles, "initial geometry");
      assert.deepEqual(await audit(device, cycles, "unchanged geometry"), [0, 0, 0]);
      assert.deepEqual(await audit(device, cycles, "explicitly held producer inputs", true), [0, 0, 0]);
      // A represented zero can move while the fitted centre phi stays fixed.
      // The retained hierarchy must notice the directional crossing itself.
      if(root.topology){
        const record=root.topology.buffer, saved=await read(device,record);
        const base=4*uniformMixedPressureStorage(root.ownership.layout).count;
        const floats=new Float32Array(saved.buffer);
        const face=floats.findIndex((v,i)=>i>=base&&(i-base)%4<3&&v>0&&v<1&&floats[4*Math.floor((i-base)/4)+(i-base)%4+1]!>1e-6);
        assert.ok(face>=base,"fixture has a subcell free-surface crossing");
        // The live topology is GPU-written storage, without COPY_DST.
        const setCrossing=async(bits:number)=>{
          const module=device.createShaderModule({code:`@group(0) @binding(0) var<storage,read_write> words:array<u32>;
@compute @workgroup_size(1) fn main(){words[${face}u]=${bits}u;}`});
          const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
          const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:record}]});
          const e=device.createCommandEncoder(),p=e.beginComputePass();p.setPipeline(pipeline);p.setBindGroup(0,group);p.dispatchWorkgroups(1);p.end();device.queue.submit([e.finish()]);
        };
        await setCrossing(new Uint32Array(new Float32Array([floats[face]!*0.8]).buffer)[0]!);
        assert.ok((await audit(device,cycles,"changed nodal crossing with unchanged centre phi")).some(n=>n>0));
        await setCrossing(saved[face]!);
        await audit(device,cycles,"restored nodal crossing");
      }
      const topology = root.ownership.presentation, words = await read(device, topology);
      const blocks = root.ownership.layout.address.blocks, records = 512 * blocks * 5 + UNIFORM_PAGE_TABLE_WORDS;
      const block = Array.from({ length: blocks - 2 }, (_, i) => i + 1).find(b => words[records + b] !== 0xffffffff)!;
      assert.ok(block);
      const phi = await read(device, root.phi), edge = 512 * block + 7;
      // A root cell on a block face changes, then returns to its exact bits.
      device.queue.writeBuffer(root.phi.buffer, (root.phi.offset ?? 0) + 4 * edge, new Float32Array([-.12345]));
      assert.ok((await audit(device, cycles, "changed face phi")).some(n => n > 0));
      device.queue.writeBuffer(root.phi.buffer, (root.phi.offset ?? 0) + 4 * edge, phi.subarray(edge, edge + 1));
      await audit(device, cycles, "restored phi");
      // Page identity changes even with numerically identical geometry.
      device.queue.writeBuffer(topology.buffer, 4 * (records + block), new Uint32Array([words[records + block]! ^ 1]));
      assert.ok((await audit(device, cycles, "page identity change")).some(n => n > 0));
      device.queue.writeBuffer(topology.buffer, 4 * (records + block), words.subarray(records + block, records + block + 1));
      await audit(device, cycles, "restored page identity");
      // Release a page and remove inbound references, then reclaim it.
      const released = words.slice(); released[records + block] = 0xffffffff;
      for (let i = records + blocks; i < records + blocks + 27 * blocks; i++) if (released[i] === block) released[i] = 0;
      device.queue.writeBuffer(topology.buffer, 0, released);
      await audit(device, cycles, "released page and changed neighbours");
      device.queue.writeBuffer(topology.buffer, 0, words);
      await audit(device, cycles, "reclaimed page");
      assert.deepEqual(await audit(device, cycles, "settled after reclaim"), [0, 0, 0]);
    } finally { solver.destroy(); }
  });
});

test("retained pressure hierarchy stays exact across moving solid geometry", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("retained pressure moving solids", async device => {
    const scene = createRigidCouplingOracleScene("rigid-hydrostatic");
    scene.voxelDomain.finestCellSize_m = .025;
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
    const bodies = initializeRigidBodies(scene.rigidBodies), solver = await createUniformSolver(device, scene);
    try {
      for (let step = 1; step <= 3; step++) {
        if (step > 1) bodies[0]!.position_m.x += .025;
        assert.ok(solver.advanceTo(step / 60, bodies)); await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined);
        await audit(device, pressure(solver), `body pose ${step}`);
      }
    } finally { solver.destroy(); }
  });
});
