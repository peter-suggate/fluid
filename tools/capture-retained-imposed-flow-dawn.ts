/** Imposed-flow QA: production scalar stages, prescribed velocity, no mesh. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";
import { affineSupportPhi, affineSupportQ, sphereBoxAmount, spherePhi, sphereQ, sphereTotalAmount,
  type Point, type Sphere } from "./retained-imposed-flow-oracle";

const option = (name: string, fallback: string) => process.argv.find(arg => arg.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const h = .05, dt = 1 / 30, dimensions = [32, 32, 32] as const, origin: Point = [-.8, 0, -.8];
const velocity: Point = [.75, 0, 0], seed: Sphere = { center: [-.15, .8, 0], radius: .25, width: h };
const nf = dimensions.reduce((a, b) => a * b, 1);
const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-quarter"));
scene.sceneId = "qa-retained-imposed-uniform-sphere";
scene.container.height_m = 1.6; scene.container.fillFraction = 0;
scene.solidVoxels = []; scene.rigidBodies = [];
scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0;
scene.fluid.initialVelocity_m_s = { x: velocity[0], y: velocity[1], z: velocity[2] };
scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: seed.center[0], y: seed.center[1], z: seed.center[2] }, radius_m: seed.radius }];
scene.fluid.refinementRegions = [{ id: "imposed-all-fine", rule: "minimum-cell-size",
  minimumCellSize_cells: 1, maximumCellSize_cells: 1,
  min_m: { x: -.8, y: 0, z: -.8 }, max_m: { x: .8, y: 1.6, z: .8 } }];
const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
  selectorMode: "coarse-first", timeStep: "paper", gammaDiffusion: "off", surfaceSharpening: "off",
});
const config = { scene, values, h, dt, dimensions, origin_m: origin, seed, velocity_m_s: velocity,
  steps: [0, 1, 2], displacement_m: [0, .025, .05],
  scope: "Imposed-flow QA, not shipping freefall. Before each normal advance, prescribe both cell/face velocity banks, effective velocity, gamma=1 and pressure=0. Never overwrite density or retained coefficients. Capture before body forces/pressure and before meshing.",
  oracle: "Independent quadratic diffuse sphere, exact y integration and adaptive Gauss x/z. Native transport discretization errors are reported separately from continuity failure. Retained point values are derived from raw captured coefficients; published phi is an actual GPU payload readback.",
  assertion: "Every donor in the padded sphere sweep must be full-fine, gamma=1 and effective u=(.75,0,0). --assert-continuity additionally rejects a retained q jump >1e-4 across a support face.",
};
if (process.argv.includes("--list")) {
  console.log(JSON.stringify(config, null, 2));
} else {
  const modulePath = process.env.WEBGPU_NODE_MODULE; assert.ok(modulePath, "Set WEBGPU_NODE_MODULE or use CPU-only --list");
  const output = option("out", "artifacts/retained-imposed-flow/sphere-full-fine");
  await mkdir(output, { recursive: true });
  const json = async (name: string, value: unknown) => writeFile(join(output, name), JSON.stringify(value, null, 2));
  const provenancePaths = ["lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-retained-scene-density.ts",
    "tools/capture-retained-imposed-flow-dawn.ts", "tools/retained-imposed-flow-oracle.ts"];
  await json("configuration.json", config);
  await json("provenance.json", { gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    workingDiffNames: execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim().split("\n"),
    sha256: Object.fromEntries(await Promise.all(provenancePaths.map(async path =>
      [path, createHash("sha256").update(await readFile(path)).digest("hex")]))), capturedAt: new Date().toISOString() });
  await acquireWebGPUExclusiveLock("dawn-probe", "retained-imposed-flow-full-fine");
  const start = performance.now(), errors: string[] = [], trace: Record<string, unknown>[] = [];
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined, gpu: GPU | undefined;
  const live = new Set<GPU>();
  type Source = WebGPUAdaptiveMassSolver["fieldSnapshotSourceForQA"];
  type RetainedLayout = { integralBaseWords: number; controlBaseWords: number;
    support: { dimensions: readonly number[]; seedMeanBaseWords: number; openFractionBaseWords: number;
      coefficientBaseWords: readonly [number, number] } };
  const layoutFor = () => {
    const resident = (solver!.sparseWorld as unknown as { resident: { retainedDensityLayout?: RetainedLayout } }).resident;
    assert.ok(resident.retainedDensityLayout); return resident.retainedDensityLayout;
  };
  const pointFor = (index: number): Point => [origin[0] + (index % 32 + .5) * h,
    (Math.floor(index / 32) % 32 + .5) * h, origin[2] + (Math.floor(index / 1024) + .5) * h];
  // The swept transition sphere plus a full fine-cell diagonal contains every
  // trilinear donor of every intersected fine-cell center and its departure.
  // Exclude remote corners of an enclosing box: those are not flow donors and
  // legitimately need not have active native storage. Walls are farther away.
  const donorRadius = Math.sqrt(seed.radius ** 2 + seed.radius * h) + Math.sqrt(3) * h;
  const donorROI = (p: Point) => Math.hypot(p[0] - Math.max(seed.center[0], Math.min(seed.center[0] + .05, p[0])),
    p[1] - seed.center[1], p[2] - seed.center[2]) <= donorRadius;
  type Cell = { id: number; index: number; point: Point; volume: number };
  type Snapshot = { source: Source; retained: RetainedLayout; cells: Cell[]; buffer: GPUBuffer;
    ranges: Record<string, { offset: number; count: number }>; stage: string; step: number };
  const snapshots: Snapshot[] = [];
  const cellsFor = async (source: Source) => {
    const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const activity = await solver!.readGPUActivityPolicy();
    const cells: Cell[] = [], seen = new Set<number>();
    for (const brick of activity.bricks) if (brick.active) {
      assert.equal(8 * brick.spanBricks / brick.acceptedResolution, 1, "actual accepted native support must be full-fine");
      assert.ok(brick.leafId < words[13]!, "bounded probe unexpectedly needs dynamic cells");
      const range = words[11]! + 2 * (4 * brick.leafId + Math.log2(brick.acceptedResolution));
      for (let id = words[range]!; id < words[range]! + words[range + 1]!; id++) {
        const at = words[6]! + 8 * id;
        const q = [floats[at]!, floats[at + 1]!, floats[at + 2]!].map(Math.floor);
        if (q.some(x => x < 0 || x >= 32)) continue;
        const index = q[0]! + 32 * (q[1]! + 32 * q[2]!);
        assert.ok(!seen.has(index), "full-fine native ownership must be unique"); seen.add(index);
        assert.equal(floats[at + 3], 1, "native cell volume is one finest voxel");
        cells.push({ id, index, point: pointFor(index), volume: floats[at + 3]! });
      }
    }
    for (let i = 0; i < nf; i++) if (donorROI(pointFor(i))) assert.ok(seen.has(i), `missing prescribed donor support ${i}`);
    return cells;
  };
  const encodeSnapshot = (source: Source, retained: RetainedLayout, cells: Cell[], encoder: GPUCommandEncoder, stage: string, step: number) => {
    const parts: { name: string; source: GPUBuffer; base: number; count: number }[] = [];
    const add = (name: string, base: number, count: number, buffer = source.state) => parts.push({ name, source: buffer, base, count });
    for (const name of ["densityA", "densityB", "gammaA", "gammaB", "pressure"] as const) add(name, source.layout[name], source.cellCapacity);
    add("integral", retained.integralBaseWords, source.cellCapacity);
    add("control", retained.controlBaseWords, 4);
    add("seedMean", retained.support.seedMeanBaseWords, nf); add("open", retained.support.openFractionBaseWords, nf);
    add("coeffA", retained.support.coefficientBaseWords[0], 2 * nf); add("coeffB", retained.support.coefficientBaseWords[1], 2 * nf);
    assert.ok(source.effectiveTransportVelocity);
    add("effective", 0, 4 * source.cellCapacity, source.effectiveTransportVelocity);
    add("scalarParity", source.frameControlBaseWords + source.scalarParityWord, 1, source.topologyArena);
    add("faceParity", source.frameControlBaseWords + source.faceParityWord, 1, source.topologyArena);
    const ranges: Snapshot["ranges"] = {}, buffer = device!.createBuffer({ size: 4 * parts.reduce((sum, p) => sum + p.count, 0),
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label: `imposed-flow ${step} ${stage}` });
    let offset = 0;
    for (const part of parts) {
      encoder.copyBufferToBuffer(part.source, 4 * part.base, buffer, 4 * offset, 4 * part.count);
      ranges[part.name] = { offset, count: part.count }; offset += part.count;
    }
    snapshots.push({ source, retained, cells, buffer, ranges, stage, step });
  };
  const means = new Map<string, ReturnType<typeof sphereBoxAmount>>();
  const analyze = async (snapshot: Snapshot) => {
    const { stage, step, ranges, cells, buffer } = snapshot;
    let data: Float32Array;
    try { await buffer.mapAsync(GPUMapMode.READ); data = new Float32Array(buffer.getMappedRange()).slice(); }
    finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
    const plane = (name: string) => data.subarray(ranges[name]!.offset, ranges[name]!.offset + ranges[name]!.count);
    const scalarParity = new Uint32Array(data.buffer)[ranges.scalarParity!.offset]!;
    const candidate = stage !== "initial" && stage !== "transport-velocity-extension";
    const rho = plane((scalarParity ^ Number(candidate)) === 0 ? "densityA" : "densityB");
    const gamma = plane((scalarParity ^ Number(candidate)) === 0 ? "gammaA" : "gammaB");
    const control = plane("control"), coeff = plane(control[1]! === 0 ? "coeffA" : "coeffB");
    const coefficient = (i: number) => [coeff[2 * i]!, coeff[2 * i + 1]!] as const;
    const atStep = stage === "transport-velocity-extension" ? step - 1 : step;
    const exact: Sphere = { ...seed, center: [seed.center[0] + atStep * dt * velocity[0], seed.center[1], seed.center[2]] };
    let amount = 0, nativeMaxError = 0, nativeL1 = 0, integralMismatch = 0, coefficientIntegralMismatch = 0;
    let velocityError = 0, gammaError = 0, pressureMax = 0, qPointMaxError = 0, continuityMaxJump = 0;
    let worstPoint: unknown, worstJump: unknown, roiCells = 0, quadratureError = 0;
    for (const cell of cells) {
      amount += rho[cell.id]! * h ** 3;
      if (!donorROI(cell.point)) continue;
      roiCells++;
      const [a, b] = coefficient(cell.index);
      for (let axis = 0; axis < 3; axis++) velocityError = Math.max(velocityError,
        Math.abs(plane("effective")[4 * cell.id + axis]! * h - velocity[axis]!));
      gammaError = Math.max(gammaError, Math.abs(gamma[cell.id]! - 1));
      pressureMax = Math.max(pressureMax, Math.abs(plane("pressure")[cell.id]!));
      const key = `${atStep}:${cell.index}`;
      if (!means.has(key)) means.set(key, sphereBoxAmount(exact,
        cell.point.map(x => x - h / 2) as unknown as Point, cell.point.map(x => x + h / 2) as unknown as Point));
      const reference = means.get(key)!;
      quadratureError += reference.estimatedError;
      const error = Math.abs(rho[cell.id]! - reference.amount / h ** 3);
      nativeMaxError = Math.max(nativeMaxError, error); nativeL1 += error * h ** 3;
      integralMismatch = Math.max(integralMismatch, Math.abs(plane("integral")[cell.id]! - rho[cell.id]!));
      coefficientIntegralMismatch = Math.max(coefficientIntegralMismatch,
        Math.abs(a * plane("seedMean")[cell.index]! + b * plane("open")[cell.index]! - rho[cell.id]!));
      for (const dx of [-.375, 0, .375]) for (const dy of [-.375, 0, .375]) for (const dz of [-.375, 0, .375]) {
        const p: Point = [cell.point[0] + dx * h, cell.point[1] + dy * h, cell.point[2] + dz * h];
        const actual = affineSupportQ(seed, p, a, b), expected = sphereQ(exact, p), e = Math.abs(actual - expected);
        if (e > qPointMaxError) { qPointMaxError = e; worstPoint = { point_m: p, actual, expected, coefficient: [a, b], cell }; }
      }
      for (let axis = 0; axis < 3; axis++) {
        const neighbor = cell.index + [1, 32, 1024][axis]!;
        const nextPoint = pointFor(neighbor); if (!donorROI(nextPoint)) continue;
        const [na, nb] = coefficient(neighbor);
        for (const u of [-.375, 0, .375]) for (const v of [-.375, 0, .375]) {
          const p = [...cell.point] as [number, number, number];
          p[axis]! += h / 2; p[(axis + 1) % 3]! += u * h; p[(axis + 2) % 3]! += v * h;
          const left = affineSupportQ(seed, p, a, b), right = affineSupportQ(seed, p, na, nb);
          const jump = Math.abs(left - right);
          if (jump > continuityMaxJump) { continuityMaxJump = jump; worstJump = { point_m: p, left, right, axis,
            index: cell.index, neighbor, leftCoefficient: [a, b], rightCoefficient: [na, nb] }; }
        }
      }
    }
    const receipt = { stage, step, referenceStep: atStep, displacement_m: atStep * dt * velocity[0],
      elapsed_ms: performance.now() - start, nativeCells: cells.length, roiCells, scalarParity, retainedControl: Array.from(control),
      amount_m3: amount, analyticAmount_m3: sphereTotalAmount(seed), nativeMaxMeanError: nativeMaxError,
      nativeL1AmountError_m3: nativeL1, quadratureEstimatedError_m3: quadratureError,
      retainedNativeMeanMismatch: integralMismatch, coefficientNativeMeanMismatch: coefficientIntegralMismatch,
      effectiveVelocityMaxError_m_s: velocityError, gammaMaxError: gammaError, pressureMax,
      retainedQPointMaxError: qPointMaxError, retainedQContinuityMaxJump: continuityMaxJump, worstPoint, worstJump };
    const stem = `step-${step}-${stage}`;
    await writeFile(join(output, `${stem}.bin`), new Uint8Array(data.buffer));
    await json(`${stem}.json`, { ...receipt, ranges, cells });
    trace.push(receipt); await json("trace.json", trace); console.log(JSON.stringify(receipt));
    // An invalid imposed velocity must never masquerade as a scalar failure.
    if (stage === "transport-velocity-extension") {
      assert.ok(velocityError <= 2e-6, `prescribed donor velocity violated: ${velocityError}`);
      assert.ok(gammaError <= 2e-6, `prescribed donor gamma violated: ${gammaError}`);
      assert.equal(pressureMax, 0, "pressure prescription before scalar transport");
    }
    if (stage === "initial" || stage === "scalar-publication") {
      assert.ok(integralMismatch <= 2e-6, `native retained integral receipt mismatch ${integralMismatch}`);
      assert.ok(coefficientIntegralMismatch <= 2e-6, `coefficient/native mean receipt mismatch ${coefficientIntegralMismatch}`);
    }
    return { receipt, coeff, control };
  };
  try {
    const dawn = await import(pathToFileURL(modulePath).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined,
      progress => console.log(JSON.stringify({ phase: "initialization", elapsed_ms: performance.now() - start, progress }))) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    for (let step = 0; step <= 2; step++) {
      const source = solver.fieldSnapshotSourceForQA, retained = layoutFor(), cells = await cellsFor(source);
      assert.deepEqual(retained.support.dimensions, dimensions);
      if (step === 0) {
        const encoder = device.createCommandEncoder(); encodeSnapshot(source, retained, cells, encoder, "initial", 0);
        device.queue.submit([encoder.finish()]);
      } else {
        const cellVelocity = new Float32Array(4 * source.cellCapacity), faces = new Float32Array(source.rowCapacity);
        for (let i = 0; i < source.cellCapacity; i++) cellVelocity.set([velocity[0] / h, 0, 0, 1], 4 * i);
        for (let row = 0; row < source.rowCapacity; row++) {
          const axis = source.templateWords[source.templateWords[7]! + source.rowCapacity + row]! >>> 30;
          assert.ok(axis < 3); faces[row] = velocity[axis]! / h;
        }
        for (const base of [source.layout.cellVelocityA, source.layout.cellVelocityB]) device.queue.writeBuffer(source.state, 4 * base, cellVelocity);
        for (const base of [source.layout.faceA, source.layout.faceB]) device.queue.writeBuffer(source.state, 4 * base, faces);
        for (const base of [source.layout.gammaA, source.layout.gammaB]) device.queue.writeBuffer(source.state, 4 * base, new Float32Array(source.cellCapacity).fill(1));
        device.queue.writeBuffer(source.state, 4 * source.layout.pressure, new Float32Array(source.cellCapacity));
        assert.ok(source.effectiveTransportVelocity); device.queue.writeBuffer(source.effectiveTransportVelocity, 0, cellVelocity);
        solver.setStageCaptureForQA((stage, encoder) => {
          if (["transport-velocity-extension", "conservative-transport", "scalar-publication"].includes(stage))
            encodeSnapshot(source, retained, cells, encoder, stage, step);
        });
        while (!solver.advanceTo(step * dt, [])) {
          assert.ok(performance.now() - start < 240000, "imposed-flow probe exceeded 240s"); await new Promise(setImmediate);
        }
        solver.setStageCaptureForQA(undefined);
        await solver.waitForTopologyReady(); assert.equal(solver.info.encodedSteps, step);
        await solver.assertSimulationHealthy();
      }
      let latest: Awaited<ReturnType<typeof analyze>> | undefined;
      while (snapshots.length) latest = await analyze(snapshots.shift()!);
      assert.ok(latest);
      const published = (await readPublishedCM12Field(device, solver)).values;
      await writeFile(join(output, `step-${step}-published-phi.bin`), new Uint8Array(published.buffer));
      const target: Sphere = { ...seed, center: [seed.center[0] + step * dt * velocity[0], .8, 0] };
      let missing = 0, maximumOracleError = 0, maximumCoefficientQueryError = 0, samples = 0;
      for (let i = 0; i < nf; i++) {
        const p = pointFor(i); if (Math.abs(spherePhi(target, p)) > 2 * h) continue;
        samples++; if (!Number.isFinite(published[i])) { missing++; continue; }
        maximumOracleError = Math.max(maximumOracleError, Math.abs(published[i]! - spherePhi(target, p)));
        const derived = affineSupportPhi(seed, p, latest.coeff[2 * i]!, latest.coeff[2 * i + 1]!, latest.control[0]! >= 1.5);
        maximumCoefficientQueryError = Math.max(maximumCoefficientQueryError, Math.abs(published[i]! - derived));
      }
      await json(`step-${step}-publication.json`, { samples, missing, maximumOracleError_m: maximumOracleError,
        maximumCoefficientQueryError_m: maximumCoefficientQueryError, scope: "actual packed GPU sample phi compared with independent sphere and separately with captured coefficient interpretation" });
      assert.equal(missing, 0, "complete publication around the translated interface");
      assert.deepEqual(errors, []);
    }
    const failures = trace.filter(r => (r.stage === "initial" || r.stage === "scalar-publication")
      && Number(r.retainedQContinuityMaxJump) > 1e-4);
    await json("completed.json", { completed: true, elapsed_ms: performance.now() - start, snapshots: trace.length,
      continuityPass: failures.length === 0, continuityFailures: failures.map(r => ({ step: r.step, jump: r.retainedQContinuityMaxJump })) });
    if (process.argv.includes("--assert-continuity")) assert.equal(failures.length, 0, "retained density must remain continuous under uniform imposed transport");
  } catch (error) {
    await json("failure.json", { elapsed_ms: performance.now() - start,
      message: error instanceof Error ? error.message : String(error), errors }); throw error;
  } finally {
    solver?.setStageCaptureForQA(undefined);
    for (const snapshot of snapshots) snapshot.buffer.destroy();
    solver?.destroy(); device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock();
  }
}
