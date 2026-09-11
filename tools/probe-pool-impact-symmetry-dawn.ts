import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from "../lib/methods/adaptive-volume/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";

// Run one arm per process, under the same GPU lease as the regression suite.
// Compare the accepted volume averages, independently of the surface renderer.
const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
const maxCell = Number(process.env.POOL_SYMMETRY_MAX_CELL ?? 0);
const steps = Number(process.env.POOL_SYMMETRY_STEPS ?? 120);
const dt = Number(process.env.POOL_SYMMETRY_DT ?? 1 / 30);
const freezeTopology = process.env.POOL_SYMMETRY_FREEZE_TOPOLOGY === "1";
const initialAtlasResident = process.env.POOL_SYMMETRY_INITIAL_ATLAS_RESIDENT === "1";
const verify = process.env.POOL_SYMMETRY_VERIFY === "1";
const output = process.env.POOL_SYMMETRY_OUTPUT ?? "artifacts/pool-impact-symmetry/coarse";
const sceneId = process.env.POOL_SYMMETRY_SCENE ?? "coarse-first-pool-impact-quarter";
assert.ok(["coarse-first-pool-impact-quarter", "minimal-power-dam-break-32"].includes(sceneId));
const capturePresentation = process.env.POOL_SYMMETRY_CAPTURE_PRESENTATION === "1";
const methodDirectory = new URL("../lib/methods/adaptive-volume/", import.meta.url);
const sourceNames = (await readdir(methodDirectory)).filter(name => name.endsWith(".ts")).sort();
const sourceHashes = async () => Object.fromEntries(await Promise.all([
  ...sourceNames.map(name => [name, new URL(name, methodDirectory)] as const),
  ["core/cm12-numerics.ts", new URL("../lib/core/cm12-numerics.ts", import.meta.url)] as const,
].map(async ([name, url]) => [name, createHash("sha256").update(await readFile(url)).digest("hex")])));
const residentSourceHashes = await sourceHashes();
assert.ok([0, 1, 2, 4, 8].includes(maxCell));
assert.ok(Number.isSafeInteger(steps) && steps > 0);
assert.ok(Number.isFinite(dt) && dt > 0);
await acquireWebGPUExclusiveLock("dawn-probe", "pool-impact-symmetry");
const live = new Set<GPU>();
Object.assign(globalThis, { poolImpactSymmetryGPU: live });
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  live.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const scene = sceneDocument(getSceneDefinition(sceneId));
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  if (maxCell) scene.fluid.refinementRegions = [{
    id: "ab-whole-domain", rule: "minimum-cell-size", minimumCellSize_cells: 1,
    maximumCellSize_cells: maxCell,
    min_m: { x: -scene.container.width_m/2, y: 0, z: -scene.container.depth_m/2 },
    max_m: { x: scene.container.width_m/2, y: scene.container.height_m, z: scene.container.depth_m/2 },
  }];
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    selectorMode: "coarse-first",
    timeStep: "scene", ...JSON.parse(process.env.POOL_SYMMETRY_OVERRIDES ?? "{}"),
  });
  solver = initialAtlasResident
    ? await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device, scene,
      "balanced", undefined, { ...adaptiveMassSolverOptions(values), initialAtlasResidentForQA: true }, () => {})
    : await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
      undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  assert.deepEqual(await sourceHashes(), residentSourceHashes, "solver sources changed during initialization");
  if (freezeTopology) solver.setTopologyFrozen(true);
  if (process.env.POOL_SYMMETRY_LEGACY_FACE === "1") solver.setLegacyFaceTransportForQA(true);
  const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
  assert.deepEqual([nx, ny, nz], sceneId === "minimal-power-dam-break-32" ? [32,32,32] : [32,24,32]);
  await mkdir(output, { recursive: true });
  if (capturePresentation) {
    const source=solver.fieldSnapshotSourceForQA;
    await writeFile(`${output}/template.bin`,new Uint8Array(source.templateWords.buffer,
      source.templateWords.byteOffset,source.templateWords.byteLength));
  }
  const residentWGSLHash = createHash("sha256").update(await readFile(new URL(
    "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url))).digest("hex");
  await writeFile(`${output}/configuration.json`, JSON.stringify({ scene, values, grid: [nx, ny, nz],
    steps, dt, maxCell, freezeTopology, initialAtlasResident, capturePresentation, legacyFace: process.env.POOL_SYMMETRY_LEGACY_FACE === "1", residentWGSLHash,
    residentSourceHashes }, null, 2));
  const symmetry = (field: ArrayLike<number>, vector = false) => {
    return ["reflectX", "reflectZ", "swapXZ"].flatMap((name, transform) => {
      // The corner dam is invariant under x/z exchange, not tank reflection.
      if (sceneId === "minimal-power-dam-break-32" && transform !== 2) return [];
      let maximum = 0, sum = 0, count = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const tx = transform === 0 ? nx - 1 - x : transform === 2 ? z : x;
        const tz = transform === 1 ? nz - 1 - z : transform === 2 ? x : z;
        const a = x + nx * (y + ny * z), b = tx + nx * (y + ny * tz);
        for (let c = 0; c < (vector ? 3 : 1); c++) {
          const tc = vector && transform === 2 ? 2 - c : c;
          const sign = vector && ((transform === 0 && c === 0) || (transform === 1 && c === 2)) ? -1 : 1;
          const error = Math.abs(field[(vector ? 4 : 1) * a + c]! - sign * field[(vector ? 4 : 1) * b + tc]!);
          maximum = Math.max(maximum, error); sum += error; count++;
        }
      }
      return [{ name, maximum, mean: sum / count }];
    });
  };
  const trace: Array<{ step: number; mass: number;
    symmetry: Record<"density" | "velocity" | "pressure" | "topology", ReturnType<typeof symmetry>>;
    [key: string]: unknown }> = [];
  let frozenRoster: string | undefined;
  for (let step = 0; step <= steps; step++) {
    if (step) {
      const captures = new Map<string, GPUBuffer>();
      const faceCaptures = new Map<string, GPUBuffer>();
      const pressureCaptures = new Map<string, GPUBuffer>();
      const audit = (process.env.POOL_SYMMETRY_AUDIT_STEPS ?? "").split(",").includes(String(step));
      const source = solver.fieldSnapshotSourceForQA;
      const records = audit ? (await solver.readGPUActivityPolicy()).bricks : [];
      if (audit) solver.setStageCaptureForQA((stage, encoder) => {
        if (stage === "pressure-solve") {
          for (const [name, base, count] of [
            ["liquid", source.layout.liquid, source.cellCapacity],
            ["theta", source.layout.theta, source.rowCapacity],
          ] as const) {
            const buffer = device!.createBuffer({ size: 4 * count,
              usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
            encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 0, 4 * count);
            pressureCaptures.set(name, buffer);
          }
        }
        if (["transport-velocity-extension", "face-preparation", "body-forces", "pressure-solve", "velocity-projection"].includes(stage)) {
          const nr = source.rowCapacity;
          const buffer = device!.createBuffer({ size: 4 * (2 * nr + 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          for (const [i, base] of [source.layout.faceA, source.layout.faceB].entries())
            encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 4 * i * nr, 4 * nr);
          encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.faceParityWord), buffer, 8 * nr, 4);
          faceCaptures.set(stage, buffer);
        }
        if (!["transport-velocity-extension", "conservative-transport", "gamma-diffusion", "surface-sharpening", "scalar-publication"].includes(stage)) return;
        const nc = source.cellCapacity;
        const buffer = device!.createBuffer({size: 4 * (6 * nc + 1), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ});
        for (const [i, base] of [source.layout.densityA, source.layout.densityB, source.layout.pressure,
          source.layout.gammaA, source.layout.gammaB, source.layout.rhs].entries())
          encoder.copyBufferToBuffer(source.state, 4 * base, buffer, 4 * i * nc, 4 * nc);
        encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.scalarParityWord), buffer, 24 * nc, 4);
        captures.set(stage, buffer);
      });
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      if (audit) {
        solver.setStageCaptureForQA(undefined);
        await writeFile(`${output}/${step}-template.bin`, new Uint8Array(source.templateWords.buffer, source.templateWords.byteOffset, source.templateWords.byteLength));
        await writeFile(`${output}/${step}-audit.json`, JSON.stringify({ cells: source.cellCapacity, rows: source.rowCapacity, records }));
        for (const [name, buffer] of pressureCaptures) {
          await buffer.mapAsync(GPUMapMode.READ);
          await writeFile(`${output}/${step}-pressure-${name}.bin`, new Uint8Array(buffer.getMappedRange()));
          buffer.unmap(); buffer.destroy();
        }
        for (const [stage, buffer] of faceCaptures) {
          await buffer.mapAsync(GPUMapMode.READ);
          await writeFile(`${output}/${step}-${stage}-faces.bin`, new Uint8Array(buffer.getMappedRange()));
          buffer.unmap(); buffer.destroy();
        }
        for (const [stage, buffer] of captures) {
          await buffer.mapAsync(GPUMapMode.READ);
          const data = new Float32Array(buffer.getMappedRange()), nc = source.cellCapacity;
          const parity = new Uint32Array(data.buffer, 24 * nc, 1)[0]! ^ Number(stage !== "transport-velocity-extension");
          const offset = stage === "gamma-diffusion" ? 2 * nc : parity * nc;
          const dense = new Float32Array(nx * ny * nz);
          const gamma = new Float32Array(dense.length);
          const gammaOffset = stage === "gamma-diffusion" ? 5 * nc : (3 + parity) * nc;
          for (const b of records.filter(b => b.active)) {
            const r = b.acceptedResolution, width = 8 * b.spanBricks / r;
            const first = source.templateWords[source.templateWords[11]! + 2 * (4 * b.leafId + Math.log2(r))]!;
            for (let z = 0; z < 8 * b.spanBricks; z++) for (let y = 0; y < 8 * b.spanBricks; y++) for (let x = 0; x < 8 * b.spanBricks; x++) {
              const [qx,qy,qz] = [8*b.coordinate[0]+x,8*b.coordinate[1]+y,8*b.coordinate[2]+z];
              const cell = first+Math.floor(x/width)+r*(Math.floor(y/width)+r*Math.floor(z/width));
              const at = qx+nx*(qy+ny*qz);
              dense[at] = data[offset+cell]!;
              gamma[at] = data[gammaOffset+cell]!;
            }
          }
          console.log(JSON.stringify({ step, stage, density: symmetry(dense) }));
          await writeFile(`${output}/${step}-${stage}.bin`, new Uint8Array(dense.buffer));
          await writeFile(`${output}/${step}-${stage}-gamma.bin`, new Uint8Array(gamma.buffer));
          buffer.unmap(); buffer.destroy();
        }
      }
      assert.equal(solver.info.encodedSteps, step, "deferred preparation must not drop a step");
    }
    try { await solver.assertSimulationHealthy(); }
    catch (error) {
      await writeFile(`${output}/failure.json`, JSON.stringify({ step, message: String(error),
        failure: error && typeof error === "object" && "failure" in error ? error.failure : undefined },null,2));
      await writeFile(`${output}/trace.json`, JSON.stringify(trace,null,2));
      throw error;
    }
    const fields = await solver.readDiagnosticFields(true);
    if (capturePresentation) {
      const published = await readPublishedCM12Field(device,solver);
      await writeFile(`${output}/${step}-published-phi.bin`, new Uint8Array(published.values.buffer));
      await writeFile(`${output}/${step}-floor-continuation.bin`, published.floorContinuation);
    }
    for (const name of ["density", "solidOpenFraction", "velocity", "pressure", "divergence"] as const) {
      const data = fields[name];
      assert.ok(data.every(Number.isFinite), `${name} finite at ${step}`);
      await writeFile(`${output}/${step}-${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }
    const activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> = await solver.readGPUActivityPolicy();
    if (freezeTopology) {
      const roster: string = JSON.stringify(activity.bricks.filter(b => b.active).map(b =>
        [b.leafId, b.coordinate, b.spanBricks, b.acceptedResolution]));
      frozenRoster ??= roster;
      assert.equal(roster, frozenRoster, `frozen membership and resolution at step ${step}`);
    }
    assert.equal(activity.faultFlags, 0);
    assert.equal(activity.commitFailed, false);
    await writeFile(`${output}/${step}-activity.json`, JSON.stringify(activity));
    const topology = new Uint8Array(nx * ny * nz);
    for (const b of activity.bricks.filter(b => b.active)) {
      const width = 8 * b.spanBricks;
      for (let z = 8 * b.coordinate[2]; z < Math.min(nz, 8 * b.coordinate[2] + width); z++)
        for (let y = 8 * b.coordinate[1]; y < Math.min(ny, 8 * b.coordinate[1] + width); y++)
          for (let x = 8 * b.coordinate[0]; x < Math.min(nx, 8 * b.coordinate[0] + width); x++)
            topology[x + nx * (y + ny * z)] = width / b.acceptedResolution;
    }
    if (initialAtlasResident && (step === 0 || freezeTopology)) {
      assert.ok(topology.every(width => width > 0), "the initial atlas must cover the entire diagnostic domain");
    }
    const stats = await solver.readStats();
    await writeFile(`${output}/${step}-stats.json`, JSON.stringify(stats));
    const heights = new Float32Array(nx * nz);
    let mass = 0, momentY = 0, momentR2 = 0, kinetic = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z), rho = fields.density[at]!;
      mass += rho; heights[x + nx * z] += rho;
      momentY += rho * (y + 0.5); momentR2 += rho * ((x + 0.5 - nx / 2) ** 2 + (z + 0.5 - nz / 2) ** 2);
      kinetic += 0.5 * rho * (fields.velocity[4 * at]! ** 2 + fields.velocity[4 * at + 1]! ** 2 + fields.velocity[4 * at + 2]! ** 2);
    }
    await writeFile(`${output}/${step}-height.bin`, new Uint8Array(heights.buffer));
    const row = { symmetry: { density: symmetry(fields.density), velocity: symmetry(fields.velocity, true), pressure: symmetry(fields.pressure), topology: symmetry(topology) }, step, time: step * dt, mass, centerOfMassY: momentY / mass,
      rmsRadius: Math.sqrt(momentR2 / mass), kinetic,
      cells: activity.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0),
      histogram: Object.fromEntries([1, 2, 4, 8].map(r => [r, activity.bricks.filter(b => b.active && b.acceptedResolution === r).length])),
      pressureIterations: stats.pressureIterationsExecuted, pressureResidual: stats.pressureRelativeResidual };
    trace.push(row); console.log(JSON.stringify(row));
    assert.deepEqual(errors, []);
  }
  assert.deepEqual(await sourceHashes(), residentSourceHashes, "solver sources changed during the capture");
  await writeFile(`${output}/trace.json`, JSON.stringify(trace, null, 2));
  if (verify) {
    const failures = trace.flatMap(row => {
      const errors: string[] = [];
      for (const metric of row.symmetry.density) {
        if (metric.maximum > 0.01 || metric.mean > 0.001)
          errors.push(`step ${row.step} density ${metric.name}: max=${metric.maximum}, mean=${metric.mean}`);
      }
      for (const metric of row.symmetry.velocity) {
        if (metric.maximum > 0.02 || metric.mean > 0.001)
          errors.push(`step ${row.step} velocity ${metric.name}: max=${metric.maximum}, mean=${metric.mean}`);
      }
      if (Math.abs(row.mass / trace[0]!.mass - 1) > 0.005) errors.push(`step ${row.step} mass retention`);
      return errors;
    });
    await writeFile(`${output}/symmetry-verdict.json`, JSON.stringify({ passed: failures.length === 0,
      limits: { densityMaximum: 0.01, densityMean: 0.001, velocityMaximum_m_s: 0.02, velocityMean_m_s: 0.001,
        relativeMassError: 0.005 }, failures }, null, 2));
    assert.deepEqual(failures, [], `Symmetry gate failed; full trajectory saved to ${output}`);
  }
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  if (gpu) live.delete(gpu);
}
