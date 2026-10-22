import assert from "node:assert/strict";
import test from "node:test";
import { particleDepositStorage } from "../lib/methods/uniform/uniform-particle-deposit-storage";
import { narrowBandTransferPropertiesSource } from "../lib/methods/uniform/uniform-narrow-band-surface.wgsl";

test("forest deposit reuses the consumed particle bank with all fine tiles retained", () => {
  const poolTiles = 514 * 512;
  const plan = particleDepositStorage(160 * 1024 ** 2, poolTiles, 2 ** 31 - 256, 514 * 32 ** 3 * 48);
  assert.equal(plan.tiles, poolTiles);
  assert.equal(plan.shared, true);
  assert.equal(plan.offsetWords, 0);
  assert.equal(plan.binsBytes, 640 * 1024 ** 2);
  assert.equal(plan.depositBytes, poolTiles * 385 * 4);
});

test("GPU deposit clear replaces poisoned scratch only inside the current fine-owner range", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  const { withUniformDevice } = await import("./helpers/uniform-geometric");
  const { readMixedBuffer } = await import("./helpers/uniform-mixed-native-fields");
  await withUniformDevice("Shared particle deposit clear", async device => {
    // Compile the production clear entry, including its capacity guard. The
    // other transfer entries require live grid fields and run in the scene gate.
    const clear = narrowBandTransferPropertiesSource(false, "depositScratch").split("var<workgroup> nbTransferFlag")[0]!;
    const module = device.createShaderModule({ code: `
@group(0) @binding(0) var<storage,read_write> source:array<vec4f>;
@group(0) @binding(0) var<storage,read_write> depositScratch:array<atomic<u32>>;
@group(0) @binding(1) var<uniform> umCounts:vec4u;
const NB_DEPOSIT=0u;const NB_DEPOSIT_TILES=3u;
${clear}
@compute @workgroup_size(1) fn sourceAccess(){source[0]=vec4f(1);}
` });
    const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "transferClear" } });
    // WebGPU permits alternate typed views of the same binding when a single
    // entry point uses only one of them. Compile the particle view as well.
    await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "sourceAccess" } });
    const words = new Uint32Array(4 * 385 + 16).fill(0xdeadbeef);
    const scratch = device.createBuffer({ size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const counts = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: scratch } }, { binding: 1, resource: { buffer: counts } },
    ] });
    try {
      for (const fineTiles of [0, 1, 3, 4]) {
        device.queue.writeBuffer(scratch, 0, words);
        device.queue.writeBuffer(counts, 0, new Uint32Array([fineTiles, 0, 0, 0]));
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(2); pass.end();
        device.queue.submit([encoder.finish()]);
        const result = new Uint32Array((await readMixedBuffer(device, scratch)).buffer);
        const cleared = fineTiles <= 3 ? fineTiles * 385 : 0;
        assert.ok(result.subarray(0, cleared).every(v => v === 0));
        assert.ok(result.subarray(cleared).every(v => v === 0xdeadbeef), "Reserved tail/guard words must remain untouched");
      }
    } finally { scratch.destroy(); counts.destroy(); }
  });
});

test("deposit preserves adapter-limited capacity and the separate-storage fallback", () => {
  for (const maxBytes of [64 * 1024 ** 2, 128 * 1024 ** 2, 2 ** 30]) {
    const prefix = 123456, poolTiles = 262144;
    const oldTiles = Math.max(0, Math.min(poolTiles, Math.floor((maxBytes / 4 - prefix) / 385)));
    for (const bankBytes of [16, 16 * 1024 ** 2, 2 ** 30]) {
      const plan = particleDepositStorage(prefix, poolTiles, maxBytes, bankBytes);
      assert.equal(plan.tiles, oldTiles);
      assert.ok(plan.binsBytes <= maxBytes);
      if (plan.shared) {
        assert.ok(plan.depositBytes <= bankBytes);
        assert.equal(plan.offsetWords, 0);
      } else {
        assert.equal(plan.offsetWords, prefix);
        assert.equal(plan.binsBytes, 4 * (prefix + 385 * oldTiles));
      }
    }
  }
});
