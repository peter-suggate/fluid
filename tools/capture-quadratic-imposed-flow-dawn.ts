/** Matched isolated current-quadric prototype capture. No shipping solver. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { compileQuadraticSupports, GPUQuadraticPullback, IDENTITY_MATRIX, supportBoxWorklist,
  type Quadratic, type QuadraticSupportGrid, type V3 } from "./implicit-density/sparse-quadratic-pullback";
import { sphereBoxAmount, spherePhi, sphereQ, sphereTotalAmount, type Point, type Sphere } from "./retained-imposed-flow-oracle";

const option = (name: string, fallback: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const grid: QuadraticSupportGrid = { origin: [-.8, 0, -.8], dimensions: [32, 32, 32], h: .05 };
const sphere: Sphere = { center: [-.15, .8, 0], radius: .25, width: .05 };
const velocity: Point = [.75, 0, 0], dt = 1 / 30, count = 32 ** 3;
// Independent expansion of (|x-c|²-R²)/(2R), used only for initial compilation.
const [cx, cy, cz] = sphere.center, R = sphere.radius;
const global: Quadratic = [(cx * cx + cy * cy + cz * cz - R * R) / (2 * R),
  -cx / R, -cy / R, -cz / R, 1 / R, 1 / R, 1 / R, 0, 0, 0];
const pointFor = (id: number, f: Point = [.5, .5, .5]): Point => [
  grid.origin[0] + (id % 32 + f[0]) * grid.h,
  grid.origin[1] + (Math.floor(id / 32) % 32 + f[1]) * grid.h,
  grid.origin[2] + (Math.floor(id / 1024) + f[2]) * grid.h];
const currentSphere = (step: number): Sphere => ({ ...sphere,
  center: [cx + step * velocity[0] * dt, cy, cz] });
const targetsFor = (step: number) => supportBoxWorklist(grid, [step, 0, 0], [32, 32, 32]);
const coverage = [0, 1, 2].map(step => {
  const s = currentSphere(step), outer = Math.sqrt(R * R + R * sphere.width);
  const lower: Point = [grid.origin[0] + step * grid.h, grid.origin[1], grid.origin[2]];
  const upper = grid.origin.map((v, axis) => v + grid.dimensions[axis]! * grid.h) as unknown as Point;
  assert.ok(s.center.every((v, axis) => v - outer > lower[axis]! && v + outer < upper[axis]!),
    "the complete positive-density sphere must lie inside the current valid support union");
  if (step) {
    // x_old=x_new-.025: the requested first destination support overlaps
    // exactly previous columns step-1 and step, both previously valid.
    const firstDonor = Math.floor(step - .5), lastDonor = Math.ceil(32 - .5) - 1;
    assert.ok(firstDonor >= step - 1 && lastDonor < 32);
  }
  return { step, supports: targetsFor(step).length, validLower_m: lower, validUpper_m: upper,
    fullPositiveDensityBounds_m: [s.center.map(v => v - outer), s.center.map(v => v + outer)],
    discardedColumnsContainNoPositiveDensity: true, completeDestinationDonorCoverage: true };
});
const configuration = { grid, sphere, dt, velocity_m_s: velocity, steps: [0, 1, 2], coverage,
  scope: "Isolated prescribed affine-flow prototype. Current GPU quadratics and their GPU integrals/queries; not adopted shipping physics, not a pressure/adaptivity/rigid validation.",
  queryStride: "16 floats: phi,q,gradient xyz,ray root t,normal xyz,Hxx,Hyy,Hzz,Hxy,Hxz,Hyz,valid",
  recordStride: "16 floats: c,gxyz,Hxx,Hyy,Hzz,Hxy,Hxz,Hyz,bitcast generation,donor count,reserved4",
  baseline: "artifacts/retained-imposed-flow/sphere-full-fine",
};
if (process.argv.includes("--list")) console.log(JSON.stringify(configuration, null, 2));
else {
  const gpuOnly = process.argv.includes("--gpu-only"), analyzeOnly = process.argv.includes("--analyze-only");
  assert.ok(!(gpuOnly && analyzeOnly), "Choose at most one phase");
  const modulePath = process.env.WEBGPU_NODE_MODULE;
  if (!analyzeOnly) assert.ok(modulePath, "Set WEBGPU_NODE_MODULE or use --list for CPU preparation");
  const output = option("out", "artifacts/retained-imposed-flow/current-quadratic"); await mkdir(output, { recursive: true });
  const json = async (name: string, value: unknown) => writeFile(join(output, name), JSON.stringify(value, null, 2));
  const binary = async (name: string, value: Float32Array | Uint32Array) => writeFile(join(output, name),
    new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
  if (!analyzeOnly) {
  const paths = ["tools/implicit-density/sparse-quadratic-pullback.ts", "tools/capture-quadratic-imposed-flow-dawn.ts",
    "tools/retained-imposed-flow-oracle.ts"];
  const hashes: Record<string, string> = {};
  for (const path of paths) {
    const contents = await readFile(path); hashes[path] = createHash("sha256").update(contents).digest("hex");
    await writeFile(join(output, path.replaceAll("/", "__") + ".snapshot"), contents);
  }
  await json("configuration.json", configuration);
  await json("provenance.json", { runStart: new Date().toISOString(), sha256: hashes,
    gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    workingTree: execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim().split("\n"),
    note: "Run-start source snapshots; prototype is isolated and has not replaced shipping transport." });
  }
  // Dawn lives only in a short-lived child. The default parent does the CPU
  // oracle after that child has EXITED, not merely after device.destroy().
  if (!gpuOnly && !analyzeOnly) execFileSync(process.execPath,
    ["--import", "tsx", fileURLToPath(import.meta.url), "--gpu-only", `--out=${output}`], { stdio: "inherit" });
  const start = performance.now(), live = new Set<GPU>(), errors: string[] = [], trace: unknown[] = [];
  let gpu: GPU | undefined, device: GPUDevice | undefined, field: GPUQuadraticPullback | undefined;
  let gpuReleased = true;
  const captures: { step: number; generation: number; targets: Uint32Array; records: Float32Array;
    amounts: Float32Array; sampled: Float32Array; offcenter: Float32Array; roots: Float32Array;
    advance: Awaited<ReturnType<GPUQuadraticPullback["advance"]>> | undefined;
    integration: GPUQuadraticPullback["lastIntegrationStats"] }[] = [];
  const releaseGPU = async () => {
    if (gpuReleased) return;
    field?.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
    await releaseWebGPUExclusiveLock(); gpuReleased = true;
  };
  const evaluateRecord = (records: Float32Array, id: number, local: Point) => {
    const at = 16 * id, [x, y, z] = local;
    return records[at]! + records[at + 1]! * x + records[at + 2]! * y + records[at + 3]! * z
      + .5 * (records[at + 4]! * x * x + records[at + 5]! * y * y + records[at + 6]! * z * z)
      + records[at + 7]! * x * y + records[at + 8]! * x * z + records[at + 9]! * y * z;
  };
  try {
    if (gpuOnly) {
    await acquireWebGPUExclusiveLock("dawn-probe", "matched-current-quadratic-imposed-flow"); gpuReleased = false;
    assert.ok(modulePath);
    const dawn = await import(pathToFileURL(modulePath).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter); device = await adapter.requestDevice();
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    console.log(JSON.stringify({ phase: "compile", elapsed_ms: performance.now() - start }));
    field = await GPUQuadraticPullback.create(device, grid, sphere.width, compileQuadraticSupports(grid, global));
    console.log(JSON.stringify({ phase: "compiled", elapsed_ms: performance.now() - start }));
    for (let step = 0; step <= 2; step++) {
      const targets = targetsFor(step), analytic = currentSphere(step);
      const advance = step ? await field.advance({ matrix: IDENTITY_MATRIX, translation: [-.025, 0, 0] }, targets) : undefined;
      const records: Float32Array = await field.readCurrentRecordsForQA();
      const words: Uint32Array = new Uint32Array(records.buffer);
      let currentSupports = 0;
      for (let id = 0; id < count; id++) {
        const valid: boolean = words[16 * id + 10] === field.generation;
        assert.equal(valid, id % 32 >= step, `current record coverage at support ${id}`); if (valid) currentSupports++;
      }
      await binary(`step-${step}-records.bin`, records); await binary(`step-${step}-ids.bin`, targets);
      const amounts = await field.integrate(targets), integration = field.lastIntegrationStats;
      await binary(`step-${step}-amounts.bin`, amounts);
      const sampled = await field.sample(Array.from(targets, id => ({ point: pointFor(id) })));
      await binary(`step-${step}-samples.bin`, sampled);
      const offcenter = await field.sample(Array.from(targets, id => ({ point: pointFor(id, [.37, .63, .29]) })));
      await binary(`step-${step}-offcenter-samples.bin`, offcenter);
      const roots = await field.sample(([0, 1, 2] as const).flatMap(axis => [-1, 1].map(sign => {
        const direction = [0, 0, 0]; direction[axis] = sign;
        return { point: analytic.center as V3, direction: direction as unknown as V3, maximum: .3 };
      })));
      await binary(`step-${step}-axis-roots.bin`, roots);
      captures.push({ step, generation: field.generation, targets, records, amounts, sampled, offcenter,
        roots, advance, integration });
      assert.deepEqual(errors, []);
      console.log(JSON.stringify({ phase: "gpu-snapshot", step, generation: field.generation,
        supports: targets.length, elapsed_ms: performance.now() - start }));
    }
    await releaseGPU();
    await json("gpu-captures.json", { completed: true, elapsed_ms: performance.now() - start,
      captures: captures.map(({ step, generation, advance, integration }) => ({ step, generation, advance, integration })) });
    console.log(JSON.stringify({ phase: "gpu-released", elapsed_ms: performance.now() - start }));
    } else {
    const metadata = JSON.parse(await readFile(join(output, "gpu-captures.json"), "utf8"));
    assert.equal(metadata.completed, true); assert.equal(metadata.captures.length, 3);
    const floats = async (name: string) => {
      const bytes = await readFile(join(output, name));
      return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
    };
    for (const entry of metadata.captures) {
      const step = entry.step; assert.equal(step, captures.length);
      const [records, ids, amounts, sampled, offcenter, roots] = await Promise.all([
        floats(`step-${step}-records.bin`), floats(`step-${step}-ids.bin`), floats(`step-${step}-amounts.bin`),
        floats(`step-${step}-samples.bin`), floats(`step-${step}-offcenter-samples.bin`), floats(`step-${step}-axis-roots.bin`)]);
      const targets = new Uint32Array(ids.buffer);
      assert.deepEqual(targets, targetsFor(step)); assert.equal(records.length, 16 * count);
      assert.equal(amounts.length, targets.length); assert.equal(sampled.length, 16 * targets.length);
      assert.equal(offcenter.length, sampled.length); assert.equal(roots.length, 16 * 6);
      captures.push({ ...entry, targets, records, amounts, sampled, offcenter, roots });
    }
    console.log(JSON.stringify({ phase: "cpu-analysis", dawnChildExited: true, gpuCapture_ms: metadata.elapsed_ms }));
    for (const capture of captures) {
      const { step, generation, targets, records, amounts, sampled, offcenter, roots, advance, integration } = capture;
      const analytic = currentSphere(step), words = new Uint32Array(records.buffer);
      const currentSupports = targets.length;
      let maximumPhiError = 0, maximumDensityError = 0, maximumGradientError = 0, coefficientQueryMismatch = 0;
      let maximumHessianError = 0, maximumGPUScalarDensityMismatch = 0;
      let maximumPhiFaceJump = 0, maximumDensityFaceJump = 0, maximumMeanError = 0, nativeL1AmountError = 0;
      let quadratureEstimatedError = 0;
      for (let rank = 0; rank < targets.length; rank++) {
        const id = targets[rank]!, p = pointFor(id);
        for (const [values, fraction] of [[sampled, [.5, .5, .5]], [offcenter, [.37, .63, .29]]] as const) {
          const q = pointFor(id, fraction), at = 16 * rank;
          assert.equal(values[at + 15], 1, "GPU query validity");
          maximumPhiError = Math.max(maximumPhiError, Math.abs(values[at]! - spherePhi(analytic, q)));
          maximumDensityError = Math.max(maximumDensityError, Math.abs(values[at + 1]! - sphereQ(analytic, q)));
          maximumGPUScalarDensityMismatch = Math.max(maximumGPUScalarDensityMismatch,
            Math.abs(values[at + 1]! - Math.max(0, Math.min(1, .5 - values[at]! / sphere.width))));
          for (let axis = 0; axis < 3; axis++) maximumGradientError = Math.max(maximumGradientError,
            Math.abs(values[at + 2 + axis]! - (q[axis]! - analytic.center[axis]!) / R));
          for (let component = 0; component < 6; component++) maximumHessianError = Math.max(maximumHessianError,
            Math.abs(values[at + 9 + component]! - (component < 3 ? 1 / R : 0)));
          coefficientQueryMismatch = Math.max(coefficientQueryMismatch, Math.abs(values[at]!
            - evaluateRecord(records, id, fraction.map(v => v * grid.h) as unknown as Point)));
        }
        for (let axis = 0; axis < 3; axis++) {
          const coordinate = [id % 32, Math.floor(id / 32) % 32, Math.floor(id / 1024)];
          if (coordinate[axis] === 31) continue;
          const neighbor = id + [1, 32, 1024][axis]!;
          if (words[16 * neighbor + 10] !== generation) continue;
          for (const u of [.125, .5, .875]) for (const v of [.125, .5, .875]) {
            const left = [0, 0, 0], right = [0, 0, 0]; left[axis] = grid.h;
            left[(axis + 1) % 3] = right[(axis + 1) % 3] = u * grid.h;
            left[(axis + 2) % 3] = right[(axis + 2) % 3] = v * grid.h;
            const a = evaluateRecord(records, id, left as unknown as Point), b = evaluateRecord(records, neighbor, right as unknown as Point);
            maximumPhiFaceJump = Math.max(maximumPhiFaceJump, Math.abs(a - b));
            maximumDensityFaceJump = Math.max(maximumDensityFaceJump,
              Math.abs(Math.max(0, Math.min(1, .5 - a / sphere.width)) - Math.max(0, Math.min(1, .5 - b / sphere.width))));
          }
        }
        // A radial bound proves constant full/dry supports. Only the curved
        // shell needs the independent exact-y/adaptive-xz integration oracle.
        const distance = Math.hypot(...p.map((v, axis) => v - analytic.center[axis]!));
        const halfDiagonal = Math.sqrt(3) * grid.h / 2;
        let expected = 0;
        if (distance + halfDiagonal <= Math.sqrt(R * R - R * sphere.width)) expected = grid.h ** 3;
        else if (distance - halfDiagonal < Math.sqrt(R * R + R * sphere.width)) {
          const reference = sphereBoxAmount(analytic, p.map(v => v - grid.h / 2) as unknown as Point,
            p.map(v => v + grid.h / 2) as unknown as Point);
          expected = reference.amount; quadratureEstimatedError += reference.estimatedError;
        }
        const error = Math.abs(amounts[rank]! - expected);
        maximumMeanError = Math.max(maximumMeanError, error / grid.h ** 3); nativeL1AmountError += error;
      }
      let maximumRootError = 0, maximumAxisNormalError = 0;
      for (let ray = 0; ray < 6; ray++) {
        maximumRootError = Math.max(maximumRootError, Math.abs(roots[16 * ray + 5]! - R));
        for (let axis = 0; axis < 3; axis++) maximumAxisNormalError = Math.max(maximumAxisNormalError,
          Math.abs(roots[16 * ray + 6 + axis]! - (axis === Math.floor(ray / 2) ? (ray % 2 ? 1 : -1) : 0)));
      }
      const amount = amounts.reduce((a, b) => a + b, 0), exactAmount = sphereTotalAmount(sphere);
      const receipt = { step, generation, time_s: step * dt, displacement_m: step * .025,
        elapsed_ms: performance.now() - start, currentSupports, advance, coverage: coverage[step], integration,
        amount_m3: amount, analyticAmount_m3: exactAmount, relativeAmountError: Math.abs(amount - exactAmount) / exactAmount,
        maximumNativeMeanError: maximumMeanError, nativeL1AmountError_m3: nativeL1AmountError, quadratureEstimatedError_m3: quadratureEstimatedError,
        maximumPhiError_m: maximumPhiError, maximumDensityError, maximumGradientError, maximumHessianError,
        maximumGPUScalarDensityMismatch, coefficientQueryMismatch_m: coefficientQueryMismatch,
        maximumPhiFaceJump_m: maximumPhiFaceJump, maximumDensityFaceJump,
        maximumAxisRootError_m: maximumRootError, maximumAxisNormalError };
      await json(`step-${step}-receipt.json`, receipt); trace.push(receipt); await json("trace.json", trace); console.log(JSON.stringify(receipt));
      assert.ok(maximumPhiError < 3e-6 && maximumGradientError < 2e-5 && maximumPhiFaceJump < 3e-6);
      assert.ok(maximumDensityError < 1e-4 && maximumDensityFaceJump < 1e-4 && maximumRootError < 3e-6);
      assert.ok(maximumHessianError < 2e-5 && maximumAxisNormalError < 2e-5 && maximumGPUScalarDensityMismatch < 1e-6);
      assert.ok(Math.abs(amount - exactAmount) / exactAmount < 2e-5, "complete retained-field mass against radial oracle");
      assert.deepEqual(errors, []);
    }
    await json("completed.json", { completed: true, elapsed_ms: performance.now() - start,
      gpuCapture_ms: metadata.elapsed_ms, scope: configuration.scope, snapshots: trace.length });
    }
  } catch (error) {
    await json("failure.json", { message: error instanceof Error ? error.message : String(error), errors,
      elapsed_ms: performance.now() - start, completedSnapshots: trace.length }); throw error;
  } finally { await releaseGPU(); }
}
