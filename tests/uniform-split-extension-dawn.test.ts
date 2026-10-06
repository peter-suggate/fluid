import assert from "node:assert/strict";
import test from "node:test";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { createUniformMixedLayoutFromWidths } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedOwnership } from "../lib/methods/uniform/uniform-mixed-ownership";
import { UniformMixedExtension } from "../lib/methods/uniform/uniform-mixed-extension";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("split extension sweeps match merged state across changing layouts and dispatch widths", async () => {
  await withUniformDevice("Split mixed extension parity", async device => {
    const lattice = { dimensions: [12, 12, 12] as const, cellSize_m: [.5, 1, 2] as const, origin_m: { x: 0, y: 0, z: 0 } };
    const ownership = new UniformMixedOwnership(device, createUniformMixedLayoutFromWidths(lattice, new Uint8Array(27).fill(1), []));
    const textures: GPUTexture[] = [], buffers: GPUBuffer[] = [];
    const texture = (size: number[], format: GPUTextureFormat) => {
      const t = device.createTexture({ size, dimension: "3d", format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST });
      textures.push(t); return t;
    };
    const buffer = (size: number, usage: GPUBufferUsageFlags) => {
      const b = device.createBuffer({ size, usage }); buffers.push(b); return b;
    };
    let mergedDescriptor: GPUComputePipelineDescriptor | undefined;
    const wrapped = new Proxy(device, { get(target, key) {
      if (key === "createComputePipelineAsync") return (d: GPUComputePipelineDescriptor) => {
        if (d.compute.entryPoint === "sweepSeams" && d.compute.constants?.ueSweepKind === 2) mergedDescriptor = d;
        return target.createComputePipelineAsync(d);
      };
      const value = Reflect.get(target, key, target);
      return typeof value === "function" ? value.bind(target) : value;
    } });
    const hierarchy = {
      input: texture([3, 3, 3], "rgba32float"), inputOrigins: texture([3, 3, 3], "rgba32uint"),
      output: texture([3, 3, 3], "rgba32float"), outputOrigins: texture([3, 3, 3], "rgba32uint"),
      encode() {},
    } as unknown as ConstructorParameters<typeof UniformMixedExtension>[2];
    const extension = new UniformMixedExtension(wrapped, ownership, hierarchy, false, true);
    try {
      await extension.initialize(); assert.ok(mergedDescriptor);
      const merged = await device.createComputePipelineAsync({ ...mergedDescriptor, compute: { ...mergedDescriptor.compute,
        constants: { ueStagedTiles: 1, ueMixedJobs: 1, ueSweepKind: 0, ueStagePack: 2 } } });
      const pipelines = extension as unknown as { regularPipelines: Map<string, GPUComputePipeline>; seamPipelines: Map<string, GPUComputePipeline>; fineSeamSweep: GPUComputePipeline };
      const physical = texture([12, 12, 12], "rgba32float"), phase = texture([12, 12, 12], "r32float"), output = texture([12, 12, 12], "rgba32float");
      const field = new Float32Array(12 ** 3 * 4), liquid = new Float32Array(12 ** 3);
      for (let i = 0; i < liquid.length; i++) {
        liquid[i] = Number((i * 13 + Math.floor(i / 12)) % 17 < 5);
        for (let a = 0; a < 3; a++) field[4 * i + a] = Math.sin(i * .17 + a);
      }
      device.queue.writeTexture({ texture: physical }, field, { bytesPerRow: 12 * 16, rowsPerImage: 12 }, [12, 12, 12]);
      device.queue.writeTexture({ texture: phase }, liquid, { bytesPerRow: 12 * 4, rowsPerImage: 12 }, [12, 12, 12]);
      const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
      const negative = buffer(3 * 12 * 12 * 4, usage), outputNegative = buffer(negative.size, usage), params = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
      device.queue.writeBuffer(negative, 0, Float32Array.from({ length: 3 * 12 * 12 }, (_, i) => Math.cos(i * .13)));
      const size = extension.scratchBytesAt(27), left = buffer(size, usage), right = buffer(size, usage);
      const read = buffer(2 * size, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ);
      const bind = (scratch: GPUBuffer) => extension.bind({ physical, phase, output, negative, outputNegative, params, scratch: { buffer: scratch } });
      const a = bind(left), b = bind(right);
      let random = 1234567;
      for (let sample = 0; sample < 66; sample++) {
        const widths = Uint8Array.from({ length: 27 }, () => { random = (Math.imul(random, 1664525) + 1013904223) >>> 0; return random >>> 31 ? 1 : 4; });
        if (sample < 2) widths.fill(sample === 0 ? 1 : 4);
        // Coexisting regular and seam tiers, with a single exceptional corner.
        if (sample === 2) { widths.fill(4); widths[0] = 1; }
        if (sample === 3) { widths.fill(1); widths[0] = 4; }
        ownership.update(createUniformMixedLayoutFromWidths(lattice, widths, []));
        const support = Uint32Array.from({ length: 27 }, (_, i) => sample === 65 || sample % 3 === 0 && i % 5 === 0 ? 0 : 3);
        device.queue.writeBuffer(ownership.support, 3 * 27 * 4, support);
        device.queue.writeBuffer(params, 0, new Float32Array([.5, 1, 2, sample & 1]));
        const seed = device.createCommandEncoder(); seed.clearBuffer(left); seed.clearBuffer(right);
        // Seed once; the hierarchy/publication do not alter extension slots.
        extension.encode(seed, a, 0); seed.copyBufferToBuffer(left, 0, right, 0, size); device.queue.submit([seed.finish()]);
        for (let sweep = 0; sweep < 3; sweep++) {
          const encoder = device.createCommandEncoder();
          for (const [split, groups] of [[true, a], [false, b]] as const) {
            const pass = encoder.beginComputePass(); pass.setBindGroup(0, ownership.bindGroup); pass.setBindGroup(1, groups[sweep % 2]!.group);
            ownership.dispatchTierCounted(pass, pipelines.regularPipelines.get("sweep")!, 0);
            if (split) {
              pass.setPipeline(pipelines.fineSeamSweep); pass.dispatchWorkgroups(1 + sample % 5);
              pass.setPipeline(pipelines.seamPipelines.get("sweep")!); pass.dispatchWorkgroups(1 + sample % 3);
            } else { pass.setPipeline(merged); pass.dispatchWorkgroups(1 + sample % 7); }
            pass.end();
          }
          encoder.copyBufferToBuffer(left, 0, read, 0, size); encoder.copyBufferToBuffer(right, 0, read, size, size);
          device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
          const data = new Uint32Array(read.getMappedRange());
          assert.deepEqual(data.subarray(0, size / 4), data.subarray(size / 4), `layout ${sample}, sweep ${sweep}`);
          read.unmap();
        }
      }
    } finally { for (const b of buffers) { if (b.mapState === "mapped") b.unmap(); b.destroy(); } textures.forEach(t => t.destroy()); ownership.destroy(); }
  });
});
