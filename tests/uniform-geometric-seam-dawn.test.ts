import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { geometricSeamRows, transportSeamReference, seamLayout, gradedSeamLayout, type Triple } from "./helpers/uniform-geometric-seam";
import type { UniformMixedLayout } from "../lib/methods/uniform/uniform-mixed-layout";
import { UniformMixedTransportStage } from "../lib/methods/uniform/uniform-mixed-transport";
import { UniformScratchArena } from "../lib/methods/uniform/uniform-scratch-arena";
import { readMixedTexture } from "./helpers/uniform-mixed-native-fields";
import { uniformVolumeStencilBytes } from "../lib/methods/uniform/uniform-volume-stencil";

/** All row geometry, owner lookup, exact donor sums and balancing happen on GPU.
 * The fixture supplies only ownership, extensive mass and prescribed departure.
 * It does not supply a per-edge donor array or a CPU-built incidence structure.
 */
async function run(device: GPUDevice, layout: UniformMixedLayout, input: Float32Array, displacements: readonly Triple[]) {
  const [nx, ny, nz] = layout.lattice.dimensions, n = nx * ny * nz;
  const arena = new UniformScratchArena(device, [nx, ny, nz], n * uniformVolumeStencilBytes(1));
  const textures: GPUTexture[] = [];
  const texture = (format: GPUTextureFormat) => {
    const t = device.createTexture({ size: [nx, ny, nz], dimension: "3d", format,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    textures.push(t); return t;
  };
  const a = texture("r32float"), b = texture("r32float"), departures = texture("rgba32float");
  const stage = new UniformMixedTransportStage(device, layout, arena, a, b, departures);
  try {
    await stage.initialize();
    assert.equal(stage.allocatedBytes, layout.tiles.length * 12 + 16);
    const cells = geometricSeamRows(layout, () => [0, 0, 0]).cells;
    const dense = new Float32Array(n), traces = new Float32Array(n * 4);
    const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
    for (const [i, c] of cells.entries()) for (let z = 0; z < c.width; z++) for (let y = 0; y < c.width; y++) for (let x = 0; x < c.width; x++)
      dense[index(c.min[0] + x, c.min[1] + y, c.min[2] + z)] = input[i]! / c.capacity;
    const results: Float32Array[] = [];
    for (const delta of displacements) {
      for (const c of cells) traces.set([...c.min.map((v, a) => v + c.width / 2 - delta[a]!), c.width], index(...c.min) * 4);
      device.queue.writeTexture({ texture: a }, dense, { bytesPerRow: nx * 4, rowsPerImage: ny }, [nx, ny, nz]);
      device.queue.writeTexture({ texture: departures }, traces, { bytesPerRow: nx * 16, rowsPerImage: ny }, [nx, ny, nz]);
      const encoder = device.createCommandEncoder(); stage.encodeRestriction(encoder); stage.encodeTransport(encoder);
      device.queue.submit([encoder.finish()]);
      const output = await readMixedTexture(device, a);
      results.push(Float32Array.from(cells, c => output[index(...c.min)]! * c.capacity));
    }
    return results;
  } finally { stage.destroy(); arena.destroy(); textures.forEach(t => t.destroy()); }
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
(modulePath ? test : test.skip)("GPU fixed geometric stencils on graded and ungraded seams match the independent volume oracle", { timeout: 180000 }, async () => {
  await acquireWebGPUExclusiveLock("dawn-test", "Uniform fixed geometric seam transport");
  let device: GPUDevice | undefined;
  try {
    const dawn = await import(pathToFileURL(modulePath!).href); Object.assign(globalThis, dawn.globals);
    const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
    const adapter = await gpu.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    const errors: string[] = []; device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
    for (const mode of ["fine", "coarse", "mixed", "graded"] as const) for (let axis = 0; axis < 3; axis++) {
      const displacements: Triple[] = [-6.25, -.25, 0, .25, 6.25].map(distance => {
        const delta = distance === 0 ? [0, 0, 0] : [.25, -.5, .75]; delta[axis] = distance;
        return delta as unknown as Triple;
      });
      const layout = mode === "graded" ? gradedSeamLayout(axis) : seamLayout(axis, mode);
      const cells = geometricSeamRows(layout, () => [0, 0, 0]).cells;
      const input = Float32Array.from(cells, (c, i) => c.capacity * (i % 7) / 3);
      const results = await run(device, layout, input, displacements);
      for (const [step, delta] of displacements.entries()) {
        const oracleRows = geometricSeamRows(layout, () => delta);
        const expected = transportSeamReference(oracleRows, input), actual = results[step]!;
        for (let i = 0; i < actual.length; i++) assert.ok(Number.isFinite(actual[i]) && actual[i]! >= 0 && Math.abs(actual[i]! - expected[i]!) < 3e-5 * Math.max(1, expected[i]!), `${mode} axis ${axis} displacement ${delta} cell ${i}: ${actual[i]} vs ${expected[i]}`);
        const sum = (a: ArrayLike<number>) => Array.from(a).reduce((s, v) => s + v, 0);
        assert.ok(Math.abs(sum(actual) - sum(input)) < 2e-6 * sum(input));
      }
    }
    assert.deepEqual(errors, []);
  } finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
});
