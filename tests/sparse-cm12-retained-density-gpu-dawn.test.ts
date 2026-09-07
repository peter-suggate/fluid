import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { compileBernsteinSupport, positiveBernsteinField, type Point3 } from "../lib/methods/adaptive-mass/sparse-cm12-positive-density-field";
import { compileDensityNativeGeometry } from "../lib/methods/adaptive-mass/sparse-cm12-density-native-geometry";
import { compileDensitySupportCoupling } from "../lib/methods/adaptive-mass/sparse-cm12-density-support-coupling";
import { WebGPURetainedDensityField, type WebGPURetainedDensityOperation } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-retained-density";
import type { SparseAtlasCompositeCell } from "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
const near = (a: number, b: number) => assert.ok(Math.abs(a - b) < 3e-6 * Math.max(1, Math.abs(b)), `${a} != ${b}`);
for (const fixture of ["quadratic", "crease"] as const) (dawnModule ? test : test.skip)(
  `${fixture}: native GPU retained queries, integrals and generation lifetimes`, { timeout: 90_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", `retained-density-${fixture}`);
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    const fields: WebGPURetainedDensityField[] = [], operations: WebGPURetainedDensityOperation[] = [];
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const support = compileBernsteinSupport([{ lower: [0, 0, 0], width: 2 }, { lower: [2, 0, 0], width: 2 }], 11);
      const value = ([x, y, z]: Point3) => fixture === "crease" ? .6 + .1 * Math.abs(x - 2) + .02 * y
        : .6 + .04 * x + .02 * y + .03 * z + .02 * x * x - .01 * y * y + .015 * x * y;
      const gradient = ([x, y]: Point3, cell: number) => fixture === "crease" ? [cell === 0 ? -.1 : .1, .02, 0]
        : [.04 + .04 * x + .015 * y, .02 - .02 * y + .015 * x, .03];
      const source = positiveBernsteinField(support, support.positions.map(p => value(p)
        + (fixture === "quadratic" ? (p[0] % 2 === 1 ? -.02 : 0) + (p[1] === 1 ? .01 : 0) : 0)), 17);
      const retained = await WebGPURetainedDensityField.create(device, source); fields.push(retained);
      const queries = [0, 1].flatMap(cell => [0, .17, .5, .93, 1].map(t => ({ cell, point: [2 * cell + 2 * t, .71, 1.23] as Point3 })));
      const query = retained.compileQueries(queries); operations.push(query);
      const pendingOutput = device.createBuffer({ size: query.outputBytes, usage: GPUBufferUsage.STORAGE });
      try { assert.throws(() => query.encode(device!.createCommandEncoder(), pendingOutput), /not ready/); }
      finally { pendingOutput.destroy(); }
      const run = async (op: WebGPURetainedDensityOperation, epoch?: { topologyGeneration: number; boundaryGeneration: number }) => {
        await op.ready();
        const output = device!.createBuffer({ size: op.outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const readback = device!.createBuffer({ size: op.outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        try {
          const encoder = device!.createCommandEncoder(); op.encode(encoder, output, epoch);
          encoder.copyBufferToBuffer(output, 0, readback, 0, op.outputBytes); device!.queue.submit([encoder.finish()]);
          await readback.mapAsync(GPUMapMode.READ); return new Float32Array(readback.getMappedRange()).slice();
        } finally { if (readback.mapState === "mapped") readback.unmap(); readback.destroy(); output.destroy(); }
      };
      const checkQueries = (result: Float32Array, increment = 0) => queries.forEach((q, i) => {
        near(result[4 * i]!, value(q.point) + increment);
        gradient(q.point, q.cell).forEach((g, axis) => near(result[4 * i + 1 + axis]!, g));
      });
      checkQueries(await run(query));
      const stamp = { topologyGeneration: 3, boundaryGeneration: 7, supportGeneration: support.generation };
      const nativeBoxes = [
        { lower: [0, 0, 0] as Point3, upper: [.7, 2, 2] as Point3 },
        { lower: [.7, 0, 0] as Point3, upper: [2.8, 2, 2] as Point3 },
        { lower: [2.8, 0, 0] as Point3, upper: [4, 2, 2] as Point3 },
      ];
      const geometry = compileDensityNativeGeometry({ ...stamp, fineCellWidth: 1, rows: [],
        cells: nativeBoxes.map((box, i) => ({ id: 10 + i * 19, stableLeafId: i, minimumFine: box.lower, maximumFine: box.upper } as SparseAtlasCompositeCell)) });
      const coupling = compileDensitySupportCoupling({ ...stamp, geometry,
        supports: support.boxes.map((box, id) => ({ id, lower: box.lower, upper: box.lower.map(v => v + box.width) as unknown as Point3 })) });
      const integrate = retained.compileCoupling(coupling); operations.push(integrate);
      for (const epoch of [undefined, { topologyGeneration: 4, boundaryGeneration: 7 }, { topologyGeneration: 3, boundaryGeneration: 8 }])
        await assert.rejects(run(integrate, epoch), /Stale/);
      const results = await run(integrate, stamp);
      // Independent tensor Gauss oracle; cut at x=2 so a crease is never
      // mistaken for one smooth polynomial by the oracle itself.
      const gaussAmount = (lower: Point3, upper: Point3) => {
        let sum = 0;
        for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
          const p = [sx, sy, sz].map((sign, a) => (lower[a] + upper[a]) / 2 + sign * (upper[a] - lower[a]) / (2 * Math.sqrt(3))) as unknown as Point3;
          sum += value(p);
        }
        return sum / 8 * upper.reduce((v, hi, a) => v * (hi - lower[a]), 1);
      };
      nativeBoxes.forEach((box, i) => {
        let amount = 0;
        for (const [lo, hi] of [[box.lower[0], Math.min(2, box.upper[0])], [Math.max(2, box.lower[0]), box.upper[0]]])
          if (hi > lo) amount += gaussAmount([lo, 0, 0], [hi, 2, 2]);
        near(results[4 * i + 1]!, amount); near(results[4 * i]!, amount / ((box.upper[0] - box.lower[0]) * 4));
      });
      const nextSource = positiveBernsteinField(support, source.controls.map(v => v + .2), 18);
      const next = retained.next(nextSource); fields.push(next);
      await next.ready();
      const nextQuery = next.compileQueries(queries); operations.push(nextQuery);
      retained.release(); next.release();
      assert.throws(() => retained.compileQueries(queries), /Released/);
      checkQueries(await run(query)); checkQueries(await run(nextQuery), .2);
      // Couplings also retain their source generation after owner release.
      const afterRelease = await run(integrate, stamp);
      assert.deepEqual(afterRelease, results);
      query.release();
      const retiredOutput = device.createBuffer({ size: query.outputBytes, usage: GPUBufferUsage.STORAGE });
      try { assert.throws(() => query.encode(device!.createCommandEncoder(), retiredOutput), /Released/); }
      finally { retiredOutput.destroy(); }
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    } finally {
      for (const op of operations) op.release(); for (const field of fields) field.release();
      device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock();
    }
  });
