/** Controlled free-surface waves. All arms use the production Sparse CM12 solver. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath, pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { solidVoxelShellForScene } from "../lib/core/scene-lattice";
import { CM12_GHOST_FLUID_THETA_MIN } from "../lib/core/cm12-numerics";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, readWebGPUExclusiveLockHolder, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { createSparseWaveStageAudit } from "./sparse-wave-stage-audit";

const arg = (name: string, fallback: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const arm = arg("arm", "fixed4"), direction = arg("direction", "x");
const width = Number(arm.replace(/^(fixed|split)/, ""));
assert.ok(["fixed1", "fixed2", "fixed4", "fixed8", "split4", "mixed", "adaptive", "oscillate"].includes(arm));
assert.ok(["x", "z", "diagonal"].includes(direction));
const h = Number(arg("h", ".05")), dt = Number(arg("dt", String(1 / 60)));
const nx = Number(arg("nx", direction === "z" ? "8" : "64"));
const ny = Number(arg("ny", "32"));
const nz = Number(arg("nz", direction === "x" ? "8" : direction === "z" ? "64" : "56"));
const height = Number(arg("height", "1.025")), amplitude = Number(arg("amplitude", ".01"));
const kx = direction === "z" ? 0 : Math.PI / (nx * h);
const kz = direction === "x" ? 0 : Math.PI / (nz * h);
const k = Math.hypot(kx, kz), omega = Math.sqrt(9.81 * k * Math.tanh(k * height));
const period = 2 * Math.PI / omega;
const steps = Number(arg("steps", String(Math.ceil(3 * period / dt))));
const sampleEvery = Number(arg("sample-every", "2"));
const conditioning = arg("conditioning", "on");
const pressureBoundary = arg("pressure-boundary", "production");
const thetaMin = Number(arg("theta-min", String(CM12_GHOST_FLUID_THETA_MIN)));
assert.ok(thetaMin > 0 && thetaMin <= 1);
assert.ok(["production", "column"].includes(pressureBoundary));
const auditSteps = new Set(arg("audit", "").split(",").filter(Boolean).map(Number));
const splitAt = Number(arg("split-at", "0"));
const forkStep = Number(arg("fork-step", "0"));
assert.ok(!forkStep || (arm.startsWith("fixed") && forkStep > 0 && forkStep < steps));
const output = arg("output", `artifacts/sparse-gravity-wave/${arm}-${direction}`);
assert.ok(nx !== nz, "the manufactured seed must not inherit square-tank D4 authority");
assert.ok([nx, ny, nz].every(n => Number.isInteger(n) && n >= 8 && n % 8 === 0));
assert.ok(height > amplitude && height + amplitude < ny * h);
assert.ok(Number.isInteger(steps) && steps > 0 && dt > 0 && h > 0);
if (process.argv.includes("--wait")) {
  for (let attempt = 0; attempt < 600 && await readWebGPUExclusiveLockHolder(); attempt++)
    await new Promise(resolve => setTimeout(resolve, 1000));
}
const liveGPU = new Set<GPU>();
await acquireWebGPUExclusiveLock("dawn-probe", `gravity-wave:${arm}:${direction}`);
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  liveGPU.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  // Record the code actually compiled, including any concurrently edited source
  // loaded by this process. Comparisons require matching shader hashes.
  const shaderHashes: Record<string, string> = {};
  const createShaderModule = device.createShaderModule.bind(device);
  let pressureShaderEdits = 0;
  let thetaShaderEdits = 0;
  device.createShaderModule = descriptor => {
    let code = descriptor.code;
    if (thetaMin !== CM12_GHOST_FLUID_THETA_MIN && code.includes("const CM12_GHOST_FLUID_THETA_MIN:f32=")) {
      code = code.replace(/const CM12_GHOST_FLUID_THETA_MIN:f32=[^;]+;/,
        `const CM12_GHOST_FLUID_THETA_MIN:f32=${thetaMin};`);
      thetaShaderEdits++;
    }
    if (pressureBoundary === "column" && code.includes("fn pressureHasPartialRefinementRegion()->bool{")) {
      // Diagnostic only: test a geometric pressure boundary on a monotone wave.
      // This does not claim a general replacement for overturning interfaces.
      code = code.replace(/fn pressureHasPartialRefinementRegion\(\)->bool\{[\s\S]*?\n\}/,
        "fn pressureHasPartialRefinementRegion()->bool{return true;}");
      code = code.replace(/fn pressurePlanarColumnHeight\(row:u32\)->vec2f\{[\s\S]*?\n\}/,
        "fn pressurePlanarColumnHeight(row:u32)->vec2f{let c=rowCenter(row);return pressureIntegratedColumnHeight(clamp(i32(floor(c.x)),0,i32(p.dimensions.x)-1),clamp(i32(floor(c.z)),0,i32(p.dimensions.z)-1));}");
      assert.notEqual(code, descriptor.code); pressureShaderEdits++;
    }
    shaderHashes[descriptor.label ?? `shader-${Object.keys(shaderHashes).length}`] = createHash("sha256").update(code).digest("hex");
    return createShaderModule({ ...descriptor, code });
  };
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = []; scene.solidVoxels = []; delete scene.terrain;
  scene.container = { ...scene.container, width_m: nx * h, height_m: ny * h, depth_m: nz * h,
    fillFraction: height / (ny * h), top: "closed", fluidWallMode: "free-slip",
    depthBoundary: direction === "x" ? "symmetry" : "closed" };
  scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.surfaceTension_N_m = 0; scene.fluid.dynamicViscosity_Pa_s = 0;
  scene.fluid.gravity_m_s2 = { x: 0, y: -9.81, z: 0 };
  delete scene.fluid.initialBrickSeeds_m; delete scene.fluid.initialLiquidVolumes;
  delete scene.fluid.initialDamBreakDimensions_m; delete scene.fluid.inflow;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  // Direct constructors do not run the scene catalog's shell compilation.
  scene.solidVoxels = [...solidVoxelShellForScene(scene)];
  const region = (min: number, max: number, left = -nx * h / 2, right = nx * h / 2) => ({
    id: `wave-${left}-${right}`, rule: "minimum-cell-size" as const,
    minimumCellSize_cells: min, maximumCellSize_cells: max,
    min_m: { x: left, y: 0, z: -nz * h / 2 }, max_m: { x: right, y: ny * h, z: nz * h / 2 },
  });
  scene.fluid.refinementRegions = arm.startsWith("fixed") ? [region(forkStep ? 1 : width, forkStep ? 1 : width)]
    : arm === "split4" ? [region(4, 4, -nx * h / 2, 0), region(4, 4, 0, nx * h / 2)]
    : arm === "mixed" ? [region(2, 2, -nx * h / 2, 0), region(4, 4, 0, nx * h / 2)]
      : arm === "oscillate" ? [region(2, 2)] : [];
  const options = adaptiveMassSolverOptions({ selectorMode: "coarse-first", timeStep: "scene" });
  solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
    ...options, maximumMacroSpanBricks: 1, topologyPageBudget: 0,
    pressureIterations: Number(arg("pressure-iterations", String(options.pressureIterations))),
    pressureRelativeTolerance: Number(arg("pressure-tolerance", String(options.pressureRelativeTolerance))),
    ...(conditioning === "off" ? { gammaDiffusionEnabled: false, surfaceSharpeningEnabled: false } : {}),
    ...(arm === "oscillate" || forkStep ? { activityPolicy: { ...options.activityPolicy!, topologyCadenceSteps: 1, demoteEpochs: 1 } } : {}),
  }, () => {});
  await solver.waitForSimulationReady();
  if (pressureBoundary !== "production") assert.ok(pressureShaderEdits > 0);
  if (thetaMin !== CM12_GHOST_FLUID_THETA_MIN) assert.ok(thetaShaderEdits > 0);
  assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], [nx, ny, nz]);
  const source = solver.fieldSnapshotSourceForQA, words = source.templateWords;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  // All levels are volume restrictions of one shared fine-volume quadrature.
  // This removes differences in shape rasterization from the comparison.
  const fine = new Float64Array(nx * ny * nz);
  const basis = new Float64Array(nx * nz);
  const sinc = (v: number) => v === 0 ? 1 : Math.sin(v) / v;
  for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
    basis[x + nx * z] = Math.cos(kx * (x + .5) * h) * sinc(.5 * kx * h)
      * Math.cos(kz * (z + .5) * h) * sinc(.5 * kz * h);
    const sx = kx ? 8 : 1, sz = kz ? 8 : 1;
    for (let iz = 0; iz < sz; iz++) for (let ix = 0; ix < sx; ix++) {
      const waterline = (height + amplitude * Math.cos(kx * (x + (ix + .5) / sx) * h)
        * Math.cos(kz * (z + (iz + .5) / sz) * h)) / h;
      for (let y = 0; y < ny; y++) fine[x + nx * (y + ny * z)]! += Math.max(0, Math.min(1, waterline - y)) / (sx * sz);
    }
  }
  const density = new Float32Array(source.cellCapacity), gamma = new Float32Array(source.cellCapacity).fill(1);
  for (let id = 0; id < words[2]!; id++) {
    const base = words[6]! + 8 * id;
    const span = [floats[base + 4]!, floats[base + 5]!, floats[base + 6]!];
    const low = [0, 1, 2].map(a => Math.round(floats[base + a]! - span[a]! / 2));
    let amount = 0;
    for (let z = low[2]!; z < low[2]! + span[2]!; z++) for (let y = low[1]!; y < low[1]! + span[1]!; y++) for (let x = low[0]!; x < low[0]! + span[0]!; x++) {
      if (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz) amount += fine[x + nx * (y + ny * z)]!;
    }
    density[id] = amount / floats[base + 3]!;
  }
  for (const offset of [source.layout.densityA, source.layout.densityB]) device.queue.writeBuffer(source.state, 4 * offset, density);
  for (const offset of [source.layout.gammaA, source.layout.gammaB]) device.queue.writeBuffer(source.state, 4 * offset, gamma);
  const frozen = arm.startsWith("fixed") || arm === "split4" || arm === "mixed";
  if (frozen) solver.setTopologyFrozen(true);
  const topology = (a: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>) => ({
    generation: a.acceptedTopologyGeneration,
    leaves: a.bricks.map(b => [b.leafId, b.coordinate, b.spanBricks, b.active, b.acceptedResolution]),
  });
  const initialActivity = await solver.readGPUActivityPolicy(), initialTopology = topology(initialActivity);
  let expectedFrozenTopology = initialTopology;
  if (arm.startsWith("fixed") || arm === "split4") assert.ok(initialActivity.bricks.every(b => !b.active || b.acceptedResolution === (forkStep ? 8 : 8 / width)), "actual enforced cell width");
  await mkdir(output, { recursive: true });
  await writeFile(`${output}/config.json`, JSON.stringify({ arm, direction, nx, ny, nz, h, dt, height, amplitude,
    kx, kz, k, omega, period, steps, conditioning, pressureBoundary, pressureShaderEdits, thetaMin, thetaShaderEdits, splitAt, forkStep, scene, initialTopology, shaderHashes,
    arguments: process.argv.slice(2),
    pressureIterations: Number(arg("pressure-iterations", String(options.pressureIterations))),
    pressureRelativeTolerance: Number(arg("pressure-tolerance", String(options.pressureRelativeTolerance))),
    seedMass: fine.reduce((a, b) => a + b, 0) * h ** 3,
    probeHash: createHash("sha256").update(await readFile(fileURLToPath(import.meta.url))).digest("hex") }, null, 2));
  const trace: unknown[] = [], timings: number[] = [];
  let previousTopology = JSON.stringify(initialTopology);
  const norm = basis.reduce((a, b) => a + b * b, 0);
  for (let step = 0; step <= steps; step++) {
    const audit = auditSteps.has(step) ? await createSparseWaveStageAudit(device, solver, h, kx, kz, nx * nz * h * h, conditioning !== "off") : undefined;
    audit?.arm();
    if (step) {
      if (step === forkStep) {
        solver.setTopologyFrozen(false);
        scene.fluid.refinementRegions = [region(width, width)];
        solver.applySceneUniforms(structuredClone(scene));
      }
      if (step === splitAt) {
        assert.equal(arm, "fixed4");
        scene.fluid.refinementRegions = [region(4, 4, -nx * h / 2, 0), region(4, 4, 0, nx * h / 2)];
        solver.applySceneUniforms(structuredClone(scene));
      }
      if (arm === "oscillate" && step > 2) {
        scene.fluid.refinementRegions = [region(step % 2 ? 1 : 2, step % 2 ? 1 : 2)];
        solver.applySceneUniforms(structuredClone(scene));
      }
      const start = performance.now();
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady(); await device.queue.onSubmittedWorkDone();
      timings.push(performance.now() - start);
      assert.equal(solver.info.encodedSteps, step);
      if (step === forkStep) {
        const accepted = await solver.readGPUActivityPolicy();
        await writeFile(`${output}/${step}-branch-activity.json`, JSON.stringify(accepted, null, 2));
        await writeFile(`${output}/${step}-branch-stats.json`, JSON.stringify(await solver.readStats(), null, 2));
        assert.ok(accepted.bricks.every(b => !b.active || b.acceptedResolution === 8 / width), "checkpoint branch commits requested width");
        solver.setTopologyFrozen(true);
        expectedFrozenTopology = topology(accepted);
      }
    }
    if (audit) await writeFile(`${output}/${step}-stages.json`, JSON.stringify(await audit.read(), null, 2));
    if (step % sampleEvery && step !== steps && step !== forkStep && arm !== "oscillate") continue;
    const fields = await solver.readDiagnosticFields(true), activity = await solver.readGPUActivityPolicy();
    const stats = await solver.readStats();
    assert.ok(fields.density.every(Number.isFinite) && fields.velocity.every(Number.isFinite));
    assert.equal(activity.faultFlags, 0); assert.equal(activity.commitFailed, false);
    if (frozen) assert.deepEqual(topology(activity), expectedFrozenTopology, "fixed topology remains unchanged");
    const nextTopology = JSON.stringify(topology(activity));
    const changed = previousTopology !== nextTopology;
    if (arm === "oscillate" && step > 2) assert.ok(changed, "each requested topology change commits");
    previousTopology = nextTopology;
    const columns = new Float64Array(nx * nz);
    let mass = 0, kinetic = 0, potential = 0, maxSpeed = 0, maxDivergence = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z), rho = fields.density[at]!;
      const m = rho * h ** 3;
      columns[x + nx * z]! += rho * h; mass += m; potential += m * 9.81 * (y + .5) * h;
      const speed = Math.hypot(fields.velocity[4 * at]!, fields.velocity[4 * at + 1]!, fields.velocity[4 * at + 2]!);
      kinetic += .5 * m * speed ** 2;
      if (rho > 1e-6) { maxSpeed = Math.max(maxSpeed, speed); maxDivergence = Math.max(maxDivergence, Math.abs(fields.divergence[at]!)); }
    }
    if (step === 0) assert.ok(Math.abs(mass - fine.reduce((a, b) => a + b, 0) * h ** 3) < 1e-6, "shared seed mass survives native restriction");
    const meanHeight = columns.reduce((a, b) => a + b, 0) / columns.length;
    const mode = columns.reduce((a, b, i) => a + (b - meanHeight) * basis[i]!, 0) / norm;
    const residualRms = Math.sqrt(columns.reduce((a, b, i) => a + (b - meanHeight - mode * basis[i]!) ** 2, 0) / columns.length);
    const row = { step, time: step * dt, mode, meanHeight, residualRms, mass, kinetic, potential, maxSpeed, maxDivergence,
      cells: activity.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0), changed,
      histogram: Object.fromEntries([1, 2, 4, 8].map(r => [8 / r, activity.bricks.filter(b => b.active && b.acceptedResolution === r).length])),
      iterations: stats.pressureIterationsExecuted, pressureResidual: stats.pressureRelativeResidual };
    trace.push(row);
    if (step % 30 === 0 || step === steps) { console.log(JSON.stringify(row)); await writeFile(`${output}/trace.json`, JSON.stringify(trace)); }
    if (step === 0 || step === steps || step === forkStep || process.argv.includes("--capture-fields")) {
      for (const name of ["density", "velocity", "gamma", "pressure", "divergence"] as const)
        await writeFile(`${output}/${step}-${name}.bin`, new Uint8Array(fields[name].buffer, fields[name].byteOffset, fields[name].byteLength));
      await writeFile(`${output}/${step}-columns.bin`, new Uint8Array(columns.buffer));
    }
    assert.deepEqual(errors, []);
  }
  await writeFile(`${output}/timings.json`, JSON.stringify({ kind: "serialized simulation wall time, excluding diagnostics", timings }));
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  if (gpu) liveGPU.delete(gpu);
}
