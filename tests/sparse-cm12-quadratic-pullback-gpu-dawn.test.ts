import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { compileQuadraticSupports, GPUQuadraticPullback, IDENTITY_MATRIX, quadraticGradient, quadraticValue,
  supportBoxWorklist, supportOrigin, type AffineDeparture, type Quadratic, type QuadraticSupportGrid, type V3,
} from "../tools/implicit-density/sparse-quadratic-pullback";
import { integrateBoxDensity, mappedPoint, mappedSphereGradient, sphereQuadratic,
  sphereRampMass, sphereValue } from "./helpers/quadratic-pullback-oracle";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const liveDawn = new Set<GPU>();
const grid: QuadraticSupportGrid = { origin: [-.4, -.4, -.4], dimensions: [32, 32, 32], h: .025 };
const h = grid.h, width = .025, radius = .1;
const nativeBox = supportBoxWorklist(grid, [8, 8, 8], [24, 24, 24]);
const sum = (values: Float32Array) => values.reduce((total, value) => total + value, 0);
const at = (x: number, y: number, z: number) => x + 32 * (y + 32 * z);
const near = (actual: number, expected: number, tolerance: number, label: string) =>
  assert.ok(Number.isFinite(actual) && Math.abs(actual - expected) <= tolerance,
    `${label}: ${actual} versus ${expected}; tolerance ${tolerance}`);

function checkCompleteField(records: Float32Array, generation: number,
  value: (point: V3) => number, gradient: (point: V3) => V3) {
  const words = new Uint32Array(records.buffer, records.byteOffset, records.length);
  let cells = 0, maximumValueError = 0, maximumGradientError = 0;
  let maximumFaceValueJump = 0, maximumFaceGradientJump = 0, maximumFaceHessianJump = 0;
  for (let id = 0; id < records.length / 16; id++) {
    if (words[16 * id + 10] !== generation) continue;
    cells++; const lower = supportOrigin(grid, id);
    const q = Array.from(records.subarray(16 * id, 16 * id + 10)) as unknown as Quadratic;
    const local: V3 = [.37 * h, .63 * h, .29 * h];
    const point = local.map((coordinate, axis) => coordinate + lower[axis]!) as unknown as V3;
    maximumValueError = Math.max(maximumValueError, Math.abs(quadraticValue(q, local) - value(point)));
    const expectedGradient = gradient(point), actualGradient = quadraticGradient(q, local);
    for (let axis = 0; axis < 3; axis++) maximumGradientError = Math.max(maximumGradientError,
      Math.abs(actualGradient[axis]! - expectedGradient[axis]!));
    const coordinate = [id % 32, Math.floor(id / 32) % 32, Math.floor(id / 1024)];
    for (let axis = 0; axis < 3; axis++) {
      if (coordinate[axis] === 31) continue;
      const neighbor = id + [1, 32, 1024][axis]!;
      if (words[16 * neighbor + 10] !== generation) continue;
      const other = Array.from(records.subarray(16 * neighbor, 16 * neighbor + 10)) as unknown as Quadratic;
      for (const fraction of [.13, .5, .87]) {
        const a = [fraction * h, (1 - fraction) * h, .43 * h]; a[axis] = h;
        const b = a.slice(); b[axis] = 0;
        maximumFaceValueJump = Math.max(maximumFaceValueJump,
          Math.abs(quadraticValue(q, a as unknown as V3) - quadraticValue(other, b as unknown as V3)));
        const ga = quadraticGradient(q, a as unknown as V3), gb = quadraticGradient(other, b as unknown as V3);
        for (let c = 0; c < 3; c++) maximumFaceGradientJump = Math.max(maximumFaceGradientJump, Math.abs(ga[c]! - gb[c]!));
        for (let c = 4; c < 10; c++) maximumFaceHessianJump = Math.max(maximumFaceHessianJump, Math.abs(q[c]! - other[c]!));
      }
    }
  }
  assert.ok(cells > 0);
  assert.ok(maximumValueError < 3e-6, `value error ${maximumValueError}`);
  assert.ok(maximumGradientError < 2e-5, `gradient error ${maximumGradientError}`);
  assert.ok(maximumFaceValueJump < 3e-6, `face value jump ${maximumFaceValueJump}`);
  assert.ok(maximumFaceGradientJump < 2e-5, `face gradient jump ${maximumFaceGradientJump}`);
  assert.ok(maximumFaceHessianJump < 2e-5, `face Hessian jump ${maximumFaceHessianJump}`);
  return { cells, maximumValueError, maximumGradientError, maximumFaceValueJump,
    maximumFaceGradientJump, maximumFaceHessianJump };
}

(dawnModule ? test : test.skip)("coherent current quadratic GPU full-fine affine transport",
  { timeout: 120_000 }, async t => {
    await acquireWebGPUExclusiveLock("dawn-test", "current-quadratic-affine-pullback");
    let gpu: GPU | undefined, device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href); Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); liveDawn.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });

      await t.test("eight sub-support translations preserve enclosed sphere mass, roots, normals, and face coherence", async () => {
        const initialCenter: V3 = [-.075, 0, 0];
        const field = await GPUQuadraticPullback.create(device!, grid, width,
          compileQuadraticSupports(grid, sphereQuadratic(initialCenter, radius)));
        try {
          const exactMass = sphereRampMass(radius, width), initialMass = sum(await field.integrate(nativeBox));
          near(initialMass, exactMass, exactMass * 2e-5, "initial quadrature mass versus radial oracle");
          const initiallyDry = await field.sample([{ point: [.11, 0, 0] }]); assert.equal(initiallyDry[1], 0);
          let maximumMassError = Math.abs(initialMass - exactMass), donorVisits = 0;
          for (let step = 1; step <= 8; step++) {
            const receipt = await field.advance({ matrix: IDENTITY_MATRIX, translation: [-.0125, 0, 0] },
              supportBoxWorklist(grid, [step, 0, 0], [32, 32, 32]));
            donorVisits += receipt.donorVisits;
            assert.ok(receipt.donorVisits >= 2 * receipt.supports, "half-support displacement validates both donor columns");
            const center: V3 = [initialCenter[0] + step * .0125, 0, 0];
            const sample = await field.sample([{ point: center, direction: [1, 0, 0], maximum: .15 }]);
            near(sample[5]!, radius, 2e-6, "translated positive-x sphere root");
            near(sample[6]!, 1, 2e-6, "translated outward normal x"); near(sample[7]!, 0, 2e-6, "normal y");
            const mass = sum(await field.integrate(nativeBox));
            maximumMassError = Math.max(maximumMassError, Math.abs(mass - exactMass));
            near(mass, exactMass, exactMass * 2e-5, "same-native-cell M0 after sub-support movement");
          }
          const finalCenter: V3 = [.025, 0, 0];
          const migrated = await field.sample([{ point: [.11, 0, 0] }]); assert.ok(migrated[1]! > .5);
          const metrics = checkCompleteField(await field.readCurrentRecordsForQA(), field.generation,
            point => sphereValue(point, finalCenter, radius),
            point => point.map((coordinate, axis) => (coordinate - finalCenter[axis]!) / radius) as unknown as V3);
          console.log(JSON.stringify({ fixture: "sphere-eight-current-generation-translations", generation: field.generation,
            exactMass, initialMass, maximumMassError, relativeQuadratureError: maximumMassError / exactMass,
            displacement: .1, donorVisits, integration: field.lastIntegrationStats, ...metrics }));
        } finally { field.destroy(); }
      });

      await t.test("plane crosses previously dry supports with the analytic root and independent per-support integrals", async () => {
        const plane: Quadratic = [.06, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        const field = await GPUQuadraticPullback.create(device!, grid, .01, compileQuadraticSupports(grid, plane));
        try {
          const old = await field.sample([{ point: [.04, 0, 0] }]); assert.equal(old[1], 0);
          await field.advance({ matrix: IDENTITY_MATRIX, translation: [-.1, 0, 0] },
            supportBoxWorklist(grid, [5, 1, 1], [31, 31, 31]));
          const sample = await field.sample([{ point: [.025, 0, 0], direction: [1, 0, 0], maximum: .024 }]);
          near(sample[5]!, .015, 2e-7, "new plane root inside formerly dry support");
          near(sample[6]!, 1, 1e-7, "plane normal");
          const ids = Uint32Array.from([at(15, 16, 16), at(16, 16, 16), at(17, 16, 16), at(18, 16, 16)]);
          const masses = await field.integrate(ids);
          for (let index = 0; index < ids.length; index++) {
            const lower = supportOrigin(grid, ids[index]!);
            const upper = lower.map(value => value + h) as unknown as V3;
            const oracle = integrateBoxDensity(point => point[0] - .04, lower, upper, .01, 64);
            near(masses[index]!, oracle, h ** 3 * 1e-4, `plane support ${ids[index]} integral`);
          }
          const metrics = checkCompleteField(await field.readCurrentRecordsForQA(), field.generation,
            point => point[0] - .04, () => [1, 0, 0]);
          console.log(JSON.stringify({ fixture: "plane-dry-support-migration", ...metrics }));
        } finally { field.destroy(); }
      });

      for (const [name, map] of [
        ["rotation", { matrix: [Math.cos(.35), Math.sin(.35), 0, -Math.sin(.35), Math.cos(.35), 0, 0, 0, 1], translation: [0, 0, 0] }],
        ["volume-preserving shear", { matrix: [1, -.3, 0, 0, 1, -.2, 0, 0, 1], translation: [0, 0, 0] }],
      ] as readonly (readonly [string, AffineDeparture])[]) await t.test(`${name}: transformed quadric, curvature, and global mass`, async () => {
        const center: V3 = [-.035, .025, 0];
        const field = await GPUQuadraticPullback.create(device!, grid, width,
          compileQuadraticSupports(grid, sphereQuadratic(center, radius)));
        try {
          await field.advance(map, supportBoxWorklist(grid, [5, 5, 5], [27, 27, 27]));
          const metrics = checkCompleteField(await field.readCurrentRecordsForQA(), field.generation,
            point => sphereValue(mappedPoint(map, point), center, radius),
            point => mappedSphereGradient(map, point, center, radius));
          const exactMass = sphereRampMass(radius, width), mass = sum(await field.integrate(nativeBox));
          near(mass, exactMass, exactMass * 2e-5, `${name} determinant-one global mass`);
          // Independent numerical oracle on a curved partially filled support;
          // convergence is measured at two orders before comparing the GPU.
          const ids = Uint32Array.from([at(18, 16, 16), at(16, 20, 16), at(14, 16, 20)]);
          const masses = await field.integrate(ids);
          for (let index = 0; index < ids.length; index++) {
            const lower = supportOrigin(grid, ids[index]!); const upper = lower.map(value => value + h) as unknown as V3;
            const value = (point: V3) => sphereValue(mappedPoint(map, point), center, radius);
            const a = integrateBoxDensity(value, lower, upper, width, 48), b = integrateBoxDensity(value, lower, upper, width, 80);
            near(a, b, h ** 3 * 2e-5, "independent partial-support quadrature convergence");
            near(masses[index]!, b, h ** 3 * 3e-5, "GPU partial-support amount");
          }
          console.log(JSON.stringify({ fixture: name, exactMass, mass, relativeQuadratureError: Math.abs(mass - exactMass) / exactMass,
            integration: field.lastIntegrationStats, ...metrics }));
        } finally { field.destroy(); }
      });

      for (const [name, mutate, fault] of [
        ["missing corner donor outside center sample", (records: Float32Array) => { new Uint32Array(records.buffer)[16 * at(15, 16, 16) + 10] = 0; }, 1],
        ["inconsistent noncenter corner coefficient", (records: Float32Array) => { records[16 * at(15, 16, 16)]! += .01; }, 2],
        ["finite face values cannot conceal inconsistent Hessians", (records: Float32Array) => { records[16 * at(15, 16, 16) + 4]! += 1; }, 2],
        ["nonfinite donor coefficient", (records: Float32Array) => { records[16 * at(15, 16, 16) + 1] = NaN; }, 8],
      ] as const) await t.test(`${name} rejects without publishing the candidate bank`, async () => {
        const records = compileQuadraticSupports(grid, sphereQuadratic([0, 0, 0], radius)); mutate(records);
        const field = await GPUQuadraticPullback.create(device!, grid, width, records);
        try {
          const before = await field.readCurrentRecordsForQA();
          await assert.rejects(field.advance({ matrix: IDENTITY_MATRIX, translation: [-.0125, -.0125, -.0125] },
            Uint32Array.of(at(16, 16, 16))), error => error instanceof Error
              && /generation rejected/.test(error.message) && ((Number(error.message.match(/fault=(\d+)/)?.[1]) & fault) !== 0));
          assert.equal(field.generation, 1);
          const after = await field.readCurrentRecordsForQA();
          assert.deepEqual(new Uint32Array(after.buffer), new Uint32Array(before.buffer), "accepted source remains word-exact after failure");
        } finally { field.destroy(); }
      });

      await t.test("ray root query rejects an unsupported interior even with valid endpoints", async () => {
        const center: V3 = [-.075, 0, 0], records = compileQuadraticSupports(grid, sphereQuadratic(center, radius));
        new Uint32Array(records.buffer)[16 * at(15, 16, 16) + 10] = 0;
        const field = await GPUQuadraticPullback.create(device!, grid, width, records);
        try {
          await assert.rejects(field.sample([{ point: center, direction: [1, 0, 0], maximum: .15 }]), /fault=16 /);
        } finally { field.destroy(); }
      });

      await t.test("failed partial writes cannot become valid during a disjoint retry", async () => {
        const field = await GPUQuadraticPullback.create(device!, grid, width,
          compileQuadraticSupports(grid, sphereQuadratic([0, 0, 0], radius)));
        try {
          const failedTarget = at(12, 12, 12), successfulTarget = at(20, 20, 20);
          await assert.rejects(field.advance({ matrix: IDENTITY_MATRIX, translation: [-.0125, 0, 0] },
            Uint32Array.of(failedTarget, 0)), /fault=1 /);
          assert.equal(field.generation, 1);
          await field.advance({ matrix: IDENTITY_MATRIX, translation: [.0125, 0, 0] }, Uint32Array.of(successfulTarget));
          const records = await field.readCurrentRecordsForQA(), words = new Uint32Array(records.buffer);
          assert.equal(field.generation, 2); assert.equal(words[16 * successfulTarget + 10], 2);
          assert.equal(words[16 * failedTarget + 10], 0, "failed candidate record is absent from accepted retry");
          assert.equal(Array.from({ length: records.length / 16 }, (_, id) => words[16 * id + 10]).filter(tag => tag === 2).length, 1);
        } finally { field.destroy(); }
      });

      await t.test("uncovered boundary, excessive footprint, singular/nonfinite/non-volume-preserving map reject", async () => {
        const field = await GPUQuadraticPullback.create(device!, grid, width,
          compileQuadraticSupports(grid, sphereQuadratic([0, 0, 0], radius)));
        try {
          await assert.rejects(field.advance({ matrix: IDENTITY_MATRIX, translation: [-h, 0, 0] }, Uint32Array.of(0)), /fault=1 /);
          await assert.rejects(field.advance({ matrix: IDENTITY_MATRIX, translation: [-.0125, -.0125, -.0125] },
            Uint32Array.of(at(16, 16, 16)), 1), /fault=4 /);
          await assert.rejects(field.advance({ matrix: [1, 0, 0, 0, 0, 0, 0, 0, 1], translation: [0, 0, 0] }, Uint32Array.of(0)), /unsupported/);
          await assert.rejects(field.advance({ matrix: [2, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] }, Uint32Array.of(0)), /unsupported/);
          await assert.rejects(field.advance({ matrix: IDENTITY_MATRIX, translation: [NaN, 0, 0] }, Uint32Array.of(0)), /unsupported/);
          assert.equal(field.generation, 1);
        } finally { field.destroy(); }
      });
      await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    } finally { device?.destroy(); if (gpu) liveDawn.delete(gpu); await releaseWebGPUExclusiveLock(); }
  });
