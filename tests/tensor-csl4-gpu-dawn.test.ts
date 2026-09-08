import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { GPUTensorCSL4, type TensorCSL4GPUQuery } from "../tools/implicit-density/tensor-csl4-gpu";
import { initializeTensorCSL4, sampleTensorCSL4, tensorAmount, tensorFunctional, tensorRangeAdmission,
  translateTensorCSL4, type AxisFunctional, type SeparableFactor, type TensorCSL4Field, type Triple } from
  "../tools/implicit-density/tensor-csl4-oracle";
import { initializeHermiteLine, sampleHermiteLine, translateHermiteLine } from
  "../tools/implicit-density/conservative-hermite-oracle";

const tau = 2 * Math.PI;
const one: SeparableFactor = { value: () => 1, derivative: () => 0, integral: (a, b) => b - a };
const sin: SeparableFactor = { value: x => Math.sin(tau * x), derivative: x => tau * Math.cos(tau * x),
  integral: (a, b) => (Math.cos(tau * a) - Math.cos(tau * b)) / tau };
const cos: SeparableFactor = { value: x => Math.cos(tau * x), derivative: x => -tau * Math.sin(tau * x),
  integral: (a, b) => (Math.sin(tau * b) - Math.sin(tau * a)) / tau };
const wave = (n: number) => initializeTensorCSL4([n, n, n], [1, 1, 1], [
  { scale: .5, factors: [one, one, one] }, { scale: .1, factors: [sin, sin, sin] },
  { scale: .07, factors: [cos, cos, cos] },
]);
function exactWave(point: Triple): readonly number[] {
  const s = point.map(sin.value), c = point.map(cos.value);
  return [.5 + .1 * s[0]! * s[1]! * s[2]! + .07 * c[0]! * c[1]! * c[2]!,
    tau * (.1 * c[0]! * s[1]! * s[2]! - .07 * s[0]! * c[1]! * c[2]!),
    tau * (.1 * s[0]! * c[1]! * s[2]! - .07 * c[0]! * s[1]! * c[2]!),
    tau * (.1 * s[0]! * s[1]! * c[2]! - .07 * c[0]! * c[1]! * s[2]!)];
}
const near = (actual: number, expected: number, tolerance: number, label: string) =>
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual} versus ${expected}, error=${Math.abs(actual - expected)}, tolerance=${tolerance}`);
async function admitted(field: GPUTensorCSL4, generation: number) {
  const receipt = await field.readReceiptForQA();
  assert.equal(receipt.fault, 0, JSON.stringify(receipt));
  assert.equal(receipt.committed, true); assert.equal(receipt.initialized, true); assert.equal(receipt.generation, generation);
}
function maximumDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
  assert.equal(a.length, b.length); let maximum = 0;
  for (let i = 0; i < a.length; i++) maximum = Math.max(maximum, Math.abs(a[i]! - b[i]!));
  return maximum;
}
const quantized = (field: TensorCSL4Field): TensorCSL4Field => ({ ...field, data: Float64Array.from(Float32Array.from(field.data)) });
function integerPermutation(field: TensorCSL4Field, displacementCells: Triple): Float32Array {
  const [nx, ny, nz] = field.dimensions, result = new Float32Array(field.data.length);
  const wrap = (i: number, n: number) => ((i % n) + n) % n;
  for (let z = 0; z < 3 * nz; z++) for (let y = 0; y < 3 * ny; y++) for (let x = 0; x < 3 * nx; x++) {
    const q = [x, y, z].map((slot, axis) => 3 * wrap(Math.floor(slot / 3) - displacementCells[axis]!, field.dimensions[axis]!) + slot % 3);
    result[x + 3 * nx * (y + 3 * ny * z)] = field.data[q[0]! + 3 * nx * (q[1]! + 3 * ny * q[2]!)]!;
  }
  return result;
}

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const liveDawn = new Set<GPU>();
(dawnModule ? test : test.skip)("periodic GPU tensor CSL4 current-field translation and atomic range admission",
  { timeout: 120_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "tensor-csl4-prescribed-translation");
    let device: GPUDevice | undefined, gpu: GPU | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); liveDawn.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });

      await t.test("anisotropic dimensional reduction matches independent 1D values, derivatives and amounts", async () => {
        const initial = initializeTensorCSL4([8, 3, 4], [1, 2, 3], [
          { scale: .5, factors: [one, one, one] }, { scale: .3, factors: [sin, one, one] },
        ]);
        let line = initializeHermiteLine(8, 1, x => .5 + .3 * sin.value(x), x => .3 * sin.derivative(x),
          (a, b) => .5 * (b - a) + .3 * sin.integral(a, b));
        const field = await GPUTensorCSL4.create(device!, initial);
        try {
          await admitted(field, 1); let generation = 1;
          for (const dx of [0, .137, .25, -.43]) {
            await field.advance([dx, .271, -.137]); line = translateHermiteLine(line, dx); await admitted(field, ++generation);
            const records = await field.readCurrentRecordsForQA(); let maximumRecordError = 0;
            for (let z = 0; z < 12; z++) for (let y = 0; y < 9; y++) for (let x = 0; x < 24; x++) {
              const i = Math.floor(x / 3), kind = x % 3;
              const expected = y % 3 === 1 || z % 3 === 1 ? 0
                : kind === 0 ? line.value[i]! : kind === 1 ? line.derivative[i]! / 8 : 8 * line.amount[i]!;
              maximumRecordError = Math.max(maximumRecordError, Math.abs(records[x + 24 * (y + 9 * z)]! - expected));
            }
            assert.ok(maximumRecordError < 2e-5, `normalized mixed moment error ${maximumRecordError}`);
            const points = Array.from({ length: 23 }, (_, i) => ({ point: [(i + .37) / 23, .13 + i / 32, 1.31 + i / 64] as Triple }));
            const values = await field.queryForQA([...points, { lower: [0, 0, 0], upper: [1, 2, 3] }]);
            for (let i = 0; i < points.length; i++) {
              const expected = sampleHermiteLine(line, points[i]!.point[0]);
              near(values[4 * i]!, expected[0], 2e-5, "dimensional value"); near(values[4 * i + 1]!, expected[1], 1e-4, "physical x derivative");
              near(values[4 * i + 2]!, 0, 2e-5, "transverse y derivative"); near(values[4 * i + 3]!, 0, 2e-5, "transverse z derivative");
            }
            near(values[4 * points.length]!, 3, 1e-5, "physical volume amount");
          }
        } finally { field.destroy(); }
      });

      await t.test("fractional translation preserves an arbitrary constant and its zero mixed derivatives word-exactly", async () => {
        const initial = initializeTensorCSL4([4, 4, 4], [1, 1, 1], [{ scale: .37, factors: [one, one, one] }]);
        const field = await GPUTensorCSL4.create(device!, initial);
        try {
          await admitted(field, 1); const before = await field.readCurrentRecordsForQA();
          for (let step = 1; step <= 12; step++) { await field.advance([1 / 12, 1 / 6, -1 / 12]); await admitted(field, step + 1); }
          assert.deepEqual(await field.readCurrentRecordsForQA(), before);
        } finally { field.destroy(); }
      });

      await t.test("nonquadratic 3D orbits match current CPU moments and converge to independent analytic motion", async () => {
        const metrics: { n: number; qError: number; gradientError: number; momentError: number; faceError: number }[] = [];
        for (const n of [4, 8]) {
          const initial = wave(n), field = await GPUTensorCSL4.create(device!, initial);
          let expected = quantized(initial);
          try {
            await admitted(field, 1);
            const shift: Triple = [1 / (3 * n), 2 / (3 * n), -1 / (3 * n)];
            for (let step = 1; step <= 3 * n; step++) {
              await field.advance(shift); expected = translateTensorCSL4(expected, shift); await admitted(field, step + 1);
            }
            const records = await field.readCurrentRecordsForQA(), momentError = maximumDifference(records, expected.data);
            assert.ok(momentError < 3e-5, `n=${n} current moment error ${momentError}`);
            const points: { point: Triple }[] = [];
            for (let z = 0; z < 7; z++) for (let y = 0; y < 7; y++) for (let x = 0; x < 7; x++) {
              points.push({ point: [(x + .137) / 7, (y + .271) / 7, (z + .413) / 7] });
            }
            const values = await field.queryForQA(points); let qError = 0, gradientError = 0;
            for (let i = 0; i < points.length; i++) {
              const exact = exactWave(points[i]!.point), cpu = sampleTensorCSL4(expected, points[i]!.point);
              qError = Math.max(qError, Math.abs(values[4 * i]! - exact[0]!));
              near(values[4 * i]!, cpu[0], 3e-5, "GPU current polynomial versus CPU");
              for (let axis = 1; axis < 4; axis++) {
                gradientError = Math.max(gradientError, Math.abs(values[4 * i + axis]! - exact[axis]!));
                near(values[4 * i + axis]!, cpu[axis]!, 3e-4, "GPU physical gradient versus CPU");
              }
            }
            const faces: TensorCSL4GPUQuery[] = [];
            for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let axis = 0; axis < 3; axis++) {
              const cell = [x, y, z] as [number, number, number], neighbor = [...cell] as [number, number, number]; neighbor[axis]!++;
              const point = cell.map((c, a) => (c + [.271, .413, .617][a]!) / n) as [number, number, number]; point[axis] = (cell[axis]! + 1) / n;
              faces.push({ point, cell }, { point, cell: neighbor });
            }
            const traces = await field.queryForQA(faces); let faceError = 0;
            for (let i = 0; i < faces.length; i += 2) for (let j = 0; j < 4; j++) {
              faceError = Math.max(faceError, Math.abs(traces[4 * i + j]! - traces[4 * (i + 1) + j]!));
            }
            assert.ok(faceError < 2e-6, `n=${n} shared C1 trace error ${faceError}`);
            const lower: Triple = [-.125, .0625, .3125], upper: Triple = [.6875, .875, 1.0625];
            const boxes = await field.queryForQA([{ lower: [0, 0, 0], upper: [1, 1, 1] }, { lower, upper }]);
            near(boxes[0]!, tensorAmount(initial), 2e-6, "same-field conserved total amount");
            const functionals = lower.map((lo, axis) => ({ kind: "integral", lower: lo, upper: upper[axis]! })) as [AxisFunctional, AxisFunctional, AxisFunctional];
            near(boxes[4]!, tensorFunctional(expected, functionals), 3e-6, "cross-cell periodic physical box integral");
            const ranges = await field.readCandidateRangesForQA();
            for (let i = 0; i < ranges.length; i += 4) assert.ok(ranges[i]! >= 0 && ranges[i + 1]! <= 1 && ranges[i + 3] === 1);
            metrics.push({ n, qError, gradientError, momentError, faceError });
          } finally { field.destroy(); }
        }
        assert.ok(metrics[0]!.qError > 8 * metrics[1]!.qError, JSON.stringify(metrics));
        assert.ok(metrics[0]!.gradientError > 6 * metrics[1]!.gradientError, JSON.stringify(metrics));
        assert.ok(metrics[1]!.qError < 8e-5 && metrics[1]!.gradientError < 2e-3, JSON.stringify(metrics));
        console.log(JSON.stringify({ fixture: "GPU tensor CSL4 smooth orbit", metrics }));
      });

      await t.test("range rejection is an atomic generation failure and a later integer shift is an exact permutation", async () => {
        const initial = initializeTensorCSL4([4, 2, 2], [4, 2, 2], []);
        for (let z = 0; z < 6; z++) for (let y = 0; y < 6; y++) for (let x = 0; x < 12; x++) {
          const i = Math.floor(x / 3), kind = x % 3;
          initial.data[x + 12 * (y + 6 * z)] = y % 3 === 1 || z % 3 === 1 ? 0
            : kind === 0 ? [0, 0, 1, 1][i]! : kind === 1 ? 0 : [0, .5, 1, .5][i]!;
        }
        assert.equal(tensorRangeAdmission(initial).admitted, true);
        const field = await GPUTensorCSL4.create(device!, initial);
        try {
          await admitted(field, 1); const before = await field.readCurrentRecordsForQA();
          await field.advance([.5, 0, 0]); const failure = await field.readReceiptForQA();
          assert.equal(failure.fault, 1); assert.equal(failure.generation, 1); assert.equal(failure.committed, false); assert.ok(failure.rejectedCells > 0);
          assert.deepEqual(await field.readCurrentRecordsForQA(), before, "failed candidate must preserve every accepted word");
          const bounds = await field.readCandidateRangesForQA(); let lower = Infinity, upper = -Infinity;
          for (let i = 0; i < bounds.length; i += 4) { lower = Math.min(lower, bounds[i]!); upper = Math.max(upper, bounds[i + 1]!); }
          near(lower, -.15625, 2e-6, "candidate negative Bernstein bound"); near(upper, 1.15625, 2e-6, "candidate overshoot bound");
          await field.advance([1, -1, 2]); await admitted(field, 2);
          assert.deepEqual(await field.readCurrentRecordsForQA(), integerPermutation(initial, [1, -1, 2]), "rejected scratch data must not survive a successful retry");
        } finally { field.destroy(); }
      });

      await t.test("unresolved positive sharp moments fail initial polynomial admission", async () => {
        const a = .015, b = .035;
        const primitive = (x: number) => x <= a ? x : x >= b ? (a + b) / 2 : a + (x - a) - (x - a) ** 2 / (2 * (b - a));
        const plane: SeparableFactor = { value: x => Math.max(0, Math.min(1, (b - x) / (b - a))),
          derivative: x => x > a && x < b ? -1 / (b - a) : 0, integral: (lo, hi) => primitive(hi) - primitive(lo) };
        const initial = initializeTensorCSL4([4, 3, 3], [1, 1, 1], [{ scale: 1, factors: [plane, one, one] }]);
        const field = await GPUTensorCSL4.create(device!, initial);
        try {
          const failure = await field.readReceiptForQA(); assert.equal(failure.fault, 1); assert.equal(failure.initialized, false); assert.equal(failure.generation, 0);
          await assert.rejects(field.queryForQA([{ point: [.125, .37, .61] }]), /fault=8/);
          assert.ok((await field.readCurrentRecordsForQA()).every(value => value === 0), "initial rejection must not publish clipped coefficients");
        } finally { field.destroy(); }
      });
      assert.deepEqual(errors, []);
    } finally {
      if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
      await new Promise<void>(resolve => setImmediate(resolve)); if (gpu) liveDawn.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
