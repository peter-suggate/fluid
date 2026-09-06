import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const mode = process.env.TOPOLOGY_ARM ?? "fixed";
assert.ok(["fixed", "oscillate", "allow"].includes(mode));
const transferOnly = process.env.TOPOLOGY_TRANSFER_ONLY === "1";
const steps = Number(process.env.TOPOLOGY_STEPS ?? 60), dt = transferOnly ? 1e-6 : 1 / 60;
const captureFields = process.env.TOPOLOGY_CAPTURE_FIELDS === "1";
const output = process.env.TOPOLOGY_OUTPUT ?? `artifacts/topology-oscillation/${mode}`;
await acquireWebGPUExclusiveLock("dawn-probe", "topology-oscillation");
const live = new Set<GPU>();
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device!.addEventListener("uncapturederror", e => errors.push(e.error.message));
  const scene = cloneScene(defaultScene);
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: .8, height_m: .8, depth_m: .8, fillFraction: 1 };
  scene.voxelDomain.finestCellSize_m = .05;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  scene.fluid.initialCondition = "dam-break";
  scene.fluid.initialDamBreakDimensions_m = transferOnly ? { x: .8, y: .8, z: .8 } : { x: .4, y: .4, z: .8 };
  if (transferOnly) scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  const region = (min: number, max: number) => [{ id: "whole-domain", rule: "minimum-cell-size" as const,
    minimumCellSize_cells: min, maximumCellSize_cells: max,
    min_m: { x: -2, y: -2, z: -2 }, max_m: { x: 2, y: 2, z: 2 } }];
  scene.fluid.refinementRegions = region(2, 2);
  const defaults = adaptiveMassSolverOptions({ timeStep: "scene", selectorMode: "coarse-first" });
  solver = await WebGPUAdaptiveMassSolver.createAsync(device!, scene, "balanced", undefined, {
    ...defaults, timeStep: "scene", initialResolutionForQA: 4, maximumMacroSpanBricks: 1,
    topologyPageBudget: 0, pressureIterations: 80,
    ...(transferOnly ? { gammaDiffusionEnabled: false, surfaceSharpeningEnabled: false } : {}),
    activityPolicy: { ...defaults.activityPolicy!, topologyCadenceSteps: 1, demoteEpochs: 1, prepareBricksPerFrame: 256 },
  }, () => {});
  await solver.waitForSimulationReady();
  await mkdir(output, { recursive: true });
  const source = solver.fieldSnapshotSourceForQA;
  if (captureFields) {
    await writeFile(`${output}/template.bin`, new Uint8Array(source.templateWords.buffer, source.templateWords.byteOffset, source.templateWords.byteLength));
    await writeFile(`${output}/layout.json`, JSON.stringify({ layout: source.layout, scene, dimensions: [solver.info.nx, solver.info.ny, solver.info.nz] }, null, 2));
  }
  if (transferOnly) {
    const w = source.templateWords, f = new Float32Array(w.buffer, w.byteOffset, w.length);
    const initial = await solver.readDiagnosticFields();
    const lower = [Infinity, Infinity, Infinity], upper = [-Infinity, -Infinity, -Infinity];
    const dims = [solver.info.nx, solver.info.ny, solver.info.nz];
    for (let z = 0; z < dims[2]!; z++) for (let y = 0; y < dims[1]!; y++) for (let x = 0; x < dims[0]!; x++) {
      if (initial.density[x + dims[0]! * (y + dims[1]! * z)]! < .5) continue;
      for (const [axis, q] of [x, y, z].entries()) { lower[axis] = Math.min(lower[axis]!, q); upper[axis] = Math.max(upper[axis]!, q + 1); }
    }
    const kx = 2 * Math.PI / (upper[0]! - lower[0]!), kz = 2 * Math.PI / (upper[2]! - lower[2]!);
    // D4-compatible divergence-free pair, so the authored symmetry authority
    // preserves this mode instead of projecting away a seeded vortex.
    const velocity = (q: number[], axis: number) => {
      const x = kx * (q[0]! - lower[0]!), z = kz * (q[2]! - lower[2]!);
      return axis === 0 ? Math.sin(2 * kz) * Math.sin(x) * Math.cos(2 * z) - Math.sin(kz) * Math.sin(2 * x) * Math.cos(z)
        : axis === 2 ? -Math.sin(kx) * Math.cos(x) * Math.sin(2 * z) + Math.sin(2 * kx) * Math.cos(2 * x) * Math.sin(z) : 0;
    };
    const faces = new Float32Array(source.rowCapacity), cells = new Float32Array(4 * source.cellCapacity);
    for (let row = 0; row < w[3]!; row++) {
      const axis = w[w[7]! + w[3]! + row]! >>> 30;
      const base = w[7]! + 6 * w[3]! + row;
      faces[row] = velocity([f[base]!, f[base + w[3]!]!, f[base + 2 * w[3]!]!], axis);
    }
    for (let id = 0; id < w[2]!; id++) {
      const base = w[6]! + 8 * id, q = [f[base]!, f[base + 1]!, f[base + 2]!];
      for (let axis = 0; axis < 3; axis++) {
        const left = [...q], right = [...q];
        left[axis]! -= .5 * f[base + 4 + axis]!; right[axis]! += .5 * f[base + 4 + axis]!;
        cells[4 * id + axis] = .5 * (velocity(left, axis) + velocity(right, axis));
      }
    }
    for (const offset of [source.layout.faceA, source.layout.faceB]) device!.queue.writeBuffer(source.state, 4 * offset, faces);
    for (const offset of [source.layout.cellVelocityA, source.layout.cellVelocityB]) device!.queue.writeBuffer(source.state, 4 * offset, cells);
  }
  const captures = new Map<string, GPUBuffer>();
  solver.setStageCaptureForQA((stage, encoder) => {
    if (!["velocity-projection", "candidate-transfer"].includes(stage)) return;
    const copy = device!.createBuffer({ size: source.state.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(source.state, 0, copy, 0, source.state.size);
    captures.set(stage, copy);
  });
  type Activity = Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>;
  const metrics = (state: Float32Array, activity: Activity, step: number) => {
    const w = source.templateWords, f = new Float32Array(w.buffer, w.byteOffset, w.length);
    // End-of-step fields occupy the destination parity. Candidate publication writes both.
    const density = step % 2 ? source.layout.densityB : source.layout.densityA;
    const velocity = step % 2 ? source.layout.cellVelocityB : source.layout.cellVelocityA;
    const faceOffset = step % 2 ? source.layout.faceB : source.layout.faceA;
    const nr = w[3]!, rowBase = w[7]!, enabled = new Uint8Array(nr);
    for (let row = 0; row < nr; row++) {
      const req = w[rowBase + nr + row]! & 0x0fffffff;
      enabled[row] = Number(Array.from({ length: w[req]! }, (_, i) => {
        const meta = w[req + 1 + i]!, b = activity.bricks[meta >>> 5];
        return b?.active && b.acceptedResolution === (meta & 31);
      }).every(Boolean));
    }
    let faceKinetic = 0;
    let mass = 0, kinetic = 0; const momentum = [0, 0, 0];
    for (const b of activity.bricks) if (b.active) {
      const range = w[11]! + 2 * (4 * b.leafId + Math.log2(b.acceptedResolution));
      for (let id = w[range]!; id < w[range]! + w[range + 1]!; id++) {
        const m = state[density + id]! * f[w[6]! + 8 * id + 3]!;
        mass += m;
        const square = [0, 0, 0], weight = [0, 0, 0];
        for (let i = w[w[9]! + id]!; i < w[w[9]! + id + 1]!; i++) {
          const row = w[w[10]! + 2 * i]!, term = w[w[10]! + 2 * i + 1]!;
          if (!enabled[row]) continue;
          const axis = w[rowBase + nr + row]! >>> 30;
          const wt = Math.abs(f[w[8]! + 2 * term + 1]!) * f[rowBase + 2 * nr + row]!;
          square[axis]! += wt * state[faceOffset + row]! ** 2; weight[axis]! += wt;
        }
        for (let a = 0; a < 3; a++) if (weight[a]! > 0) faceKinetic += .5 * m * square[a]! / weight[a]!;
        for (let a = 0; a < 3; a++) {
          const v = state[velocity + 4 * id + a]!;
          momentum[a]! += m * v; kinetic += .5 * m * v * v;
        }
      }
    }
    return { mass, momentum, kinetic, faceKinetic };
  };
  const trace = [];
  let previous = await solver.readGPUActivityPolicy();
  for (let step = 0; step <= steps; step++) {
    if (step) {
      // Both arms have an identical two-step warm-up at width two.
      if (step > 2 && mode !== "fixed") {
        const fine = step % 2 === 1;
        scene.fluid.refinementRegions = region(fine ? 1 : 2, fine && mode === "oscillate" ? 1 : 2);
        solver.applySceneUniforms(structuredClone(scene));
      }
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps, step);
    }
    const activity = await solver.readGPUActivityPolicy();
    const fields = await solver.readDiagnosticFields();
    if (captureFields) await writeFile(`${output}/${step}-density.bin`, new Uint8Array(fields.density.buffer, fields.density.byteOffset, fields.density.byteLength));
    assert.ok(fields.density.every(Number.isFinite));
    const stageMetrics: Record<string, ReturnType<typeof metrics>> = {};
    for (const [stage, buffer] of captures) {
      await buffer.mapAsync(GPUMapMode.READ);
      stageMetrics[stage] = metrics(new Float32Array(buffer.getMappedRange()), stage === "velocity-projection" ? previous : activity, step);
      if (captureFields) {
        await writeFile(`${output}/${step}-${stage}-state.bin`, new Uint8Array(buffer.getMappedRange()));
        await writeFile(`${output}/${step}-${stage}-activity.json`, JSON.stringify(stage === "velocity-projection" ? previous : activity));
      }
      buffer.unmap(); buffer.destroy();
    }
    captures.clear();
    const mass = fields.density.reduce((s, x) => s + x, 0);
    const nx = solver.info.nx, ny = solver.info.ny, nz = solver.info.nz;
    let symmetry = 0, front = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const r = fields.density[x + nx * (y + ny * z)]!;
      symmetry = Math.max(symmetry, Math.abs(r - fields.density[x + nx * (y + ny * (nz - 1 - z))]!));
      if (x >= nx - 2) front += r;
    }
    const changed = activity.bricks.filter((b, i) => b.acceptedResolution !== previous.bricks[i]?.acceptedResolution || b.active !== previous.bricks[i]?.active).length;
    const row = { step, mass, symmetry, front, changed, generation: activity.acceptedTopologyGeneration,
      cells: activity.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0), histogram: activity.bricks.filter(b => b.active).map(b => b.acceptedResolution),
      fault: activity.faultFlags, commitFailed: activity.commitFailed, stages: stageMetrics };
    trace.push(row); console.log(JSON.stringify(row));
    previous = activity;
  }
  await writeFile(`${output}/trace.json`, JSON.stringify(trace, null, 2));
  assert.deepEqual(errors, []);
  assert.ok(trace.every(x => !x.fault && !x.commitFailed));
  if (mode === "oscillate") assert.ok(trace.slice(3).every(x => x.changed > 0), "every requested oscillation must actually commit");
} finally {
  solver?.destroy(); device?.destroy(); live.clear(); await releaseWebGPUExclusiveLock();
}
