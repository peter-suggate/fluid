/** Actual production simulation and freshly emitted surface snapshots.
 * Run one arm per process. Offline QA shading consumes these triangles without
 * rebuilding, projecting, smoothing, or substituting an analytic surface.
 */
import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { withRefinementRegionsFromQuery } from "../lib/core/editor-refinement-region";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { CM12_FAILURE_BYTES } from "../lib/methods/adaptive-mass/sparse-cm12-simulation-failure";
import { captureCurrentMapSnapshot } from "./current-map-snapshot-dawn";
import { readPublishedCM12Mesh } from "./sparse-cm12-published-mesh";

const sceneIds = {
  quarter: "coarse-first-pool-impact-quarter",
  half: "coarse-first-pool-impact-half",
  mini32: "minimal-power-dam-break-32",
  rigid: "water-box-tank-fill",
} as const;
const option = (name: string, fallback: string) => process.argv.find(arg => arg.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const sceneKey = option("scene", "quarter") as keyof typeof sceneIds;
assert.ok(sceneKey in sceneIds, `Unknown scene ${sceneKey}`);
const arm = option("arm", "coarse"); assert.ok(arm === "coarse" || arm === "fine");
const densityTransport = option("transport", "native-cm12");
assert.ok(densityTransport === "native-cm12" || densityTransport === "current-map");
const steps = option("steps", "0,6,15,30").split(",").map(Number);
assert.ok(steps[0] === 0 && steps.every((step, i) => Number.isSafeInteger(step) && step >= 0 && (!i || step > steps[i - 1]!)));
const definition = getSceneDefinition(sceneIds[sceneKey]);
let scene = sceneDocument(definition);
const regionMode = option("regions", "authored");
assert.ok(regionMode === "authored" || regionMode === "handoff");
const regionQuery = regionMode === "handoff" && (sceneKey === "quarter" || sceneKey === "half")
  ? "0_0_0_25_66.6667_100_8_8" : undefined;
if (regionQuery) scene = withRefinementRegionsFromQuery(scene, regionQuery);
const h = scene.voxelDomain.finestCellSize_m;
const { width_m: width, height_m: height, depth_m: depth } = scene.container;
const origin = [-width / 2, 0, -depth / 2] as const;
const dimensions = [width, height, depth].map(length => Math.round(length / h));
if (arm === "fine") scene.fluid.refinementRegions = [{
  id: "visual-ab-all-fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
  min_m: { x: origin[0], y: origin[1], z: origin[2] },
  max_m: { x: width / 2, y: height, z: depth / 2 },
}];
const dt = 1 / 30;
assert.equal(scene.numerics.fixedDt_s, dt, "capture keeps the catalog's authored paper step");
const values = resolveMethodValues(adaptiveMassMethod, "balanced", { selectorMode: "coarse-first", timeStep: "paper", densityTransport });
const config = { sceneKey, sceneId: sceneIds[sceneKey], arm, scene, values, h, dimensions, origin,
  steps, times_s: steps.map(step => step * dt), dt, regionMode, originalRegionQuery: regionQuery,
  fieldDomain: "Diagnostic arrays are clipped to the authored domain; signed world leaves are additionally checked in activity receipts.",
  camera: { projection: "orthographic QA mesh view", elevation_deg: 24, azimuth_deg: -55,
    bounds_m: [origin, [width / 2, Math.max(height, ...scene.rigidBodies.map(body => body.position_m.y + body.dimensions_m.y)), depth / 2]] },
  displayContract: "Unmodified shipping GPU-emitted triangles; QA shading is not the app's water optics. Rigid overlays use GPU poses.",
};
if (process.argv.includes("--list")) {
  console.log(JSON.stringify(config, null, 2));
} else {
  const modulePath = process.env.WEBGPU_NODE_MODULE;
  assert.ok(modulePath, "Set WEBGPU_NODE_MODULE; --list performs CPU-only preparation");
  const output = option("out", join("artifacts/retained-visual-ab", sceneKey, arm));
  await mkdir(output, { recursive: true });
  await writeFile(join(output, "configuration.json"), JSON.stringify(config, null, 2));
  const sources = ["lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-current-map-measure.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-current-map-velocity.wgsl.ts",
    "tools/capture-retained-visual-ab-dawn.ts"];
  await writeFile(join(output, "provenance.json"), JSON.stringify({
    gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    capturedAt: new Date().toISOString(),
    sha256: Object.fromEntries(await Promise.all(sources.map(async path =>
      [path, createHash("sha256").update(await readFile(path)).digest("hex")]))),
  }, null, 2));
  await acquireWebGPUExclusiveLock("dawn-probe", `retained-visual-${sceneKey}-${arm}`);
  const startedAt = performance.now(), budgetMs = Number(option("budget-ms", "240000"));
  let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
  const live = new Set<GPU>(), errors: string[] = [], trace: unknown[] = [];
  let step = 0;
  const json = async (name: string, value: unknown) => writeFile(join(output, name), JSON.stringify(value, null, 2));
  try {
    const dawn = await import(pathToFileURL(modulePath).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    const bodies = initializeRigidBodies(scene.rigidBodies);
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined,
      progress => console.log(JSON.stringify({ phase: "initialization", elapsed_ms: performance.now() - startedAt, progress }))) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady();
    assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    const stageReceipts: {stage:string, buffer:GPUBuffer}[] = [];
    if (densityTransport === "current-map") solver.setStageCaptureForQA((stage, encoder) => {
      const source = solver!.fieldSnapshotSourceForQA;
      const buffer = device!.createBuffer({ size: 20 + CM12_FAILURE_BYTES + 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.copyBufferToBuffer(source.state, 4 * source.retainedControlBaseWords!, buffer, 0, 16);
      encoder.copyBufferToBuffer(source.state, 4 * source.currentMap!.chainCountBaseWords, buffer, 16, 4);
      encoder.copyBufferToBuffer(source.topologyArena, source.topologyArena.size - CM12_FAILURE_BYTES, buffer, 20, CM12_FAILURE_BYTES);
      encoder.copyBufferToBuffer(source.topologyArena, source.topologyArena.size - CM12_FAILURE_BYTES - 128, buffer, 20 + CM12_FAILURE_BYTES, 128);
      stageReceipts.push({stage, buffer});
    });
    const capture = async () => {
      const directory = join(output, `step-${step}`); await mkdir(directory, { recursive: true });
      await captureCurrentMapSnapshot(device!, solver!.fieldSnapshotSourceForQA, directory);
      const fields = await solver!.readDiagnosticFields(true);
      const activity = await solver!.readGPUActivityPolicy();
      const poses = await solver!.readRigidBodyPoses();
      const stats = await solver!.readStats();
      const widths: Record<string, number> = {}, violations: unknown[] = [], outsideAuthoredLeaves: unknown[] = [];
      for (const brick of activity.bricks) if (brick.active) {
        const nativeWidth = 8 * brick.spanBricks / brick.acceptedResolution;
        widths[nativeWidth] = (widths[nativeWidth] ?? 0) + 1;
        if (arm === "fine" && nativeWidth !== 1) violations.push({ leafId: brick.leafId,
          coordinate: brick.coordinate, spanBricks: brick.spanBricks, resolution: brick.acceptedResolution, nativeWidth });
        if (brick.coordinate.some((q, axis) => q < 0 || (q + brick.spanBricks) * 8 > dimensions[axis]!))
          outsideAuthoredLeaves.push({ leafId: brick.leafId, coordinate: brick.coordinate,
            spanBricks: brick.spanBricks, meanDensity: brick.meanDensity });
      }
      let amount = 0, kinetic = 0, maximumSpeed = 0, nonFinite = 0, minimumDensity = Infinity, maximumDensity = -Infinity;
      const center = [0, 0, 0], wetMin = [Infinity, Infinity, Infinity], wetMax = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < fields.density.length; i++) {
        const rho = fields.density[i]!, v = [fields.velocity[4 * i]!, fields.velocity[4 * i + 1]!, fields.velocity[4 * i + 2]!];
        if (![rho, ...v, fields.pressure[i]!, fields.divergence[i]!].every(Number.isFinite)) { nonFinite++; continue; }
        amount += rho; minimumDensity = Math.min(minimumDensity, rho); maximumDensity = Math.max(maximumDensity, rho);
        const q = [i % dimensions[0]!, Math.floor(i / dimensions[0]!) % dimensions[1]!, Math.floor(i / (dimensions[0]! * dimensions[1]!))];
        const speed2 = v.reduce((sum, component) => sum + component * component, 0);
        kinetic += Math.max(0, rho) * speed2;
        if (rho > .01) maximumSpeed = Math.max(maximumSpeed, Math.sqrt(speed2));
        for (let axis = 0; axis < 3; axis++) {
          const p = origin[axis]! + (q[axis]! + .5) * h;
          center[axis]! += rho * p;
          if (rho >= .5) { wetMin[axis] = Math.min(wetMin[axis]!, p - h / 2); wetMax[axis] = Math.max(wetMax[axis]!, p + h / 2); }
        }
      }
      for (const name of ["density", "solidOpenFraction", "velocity", "pressure", "divergence"] as const) await writeFile(join(directory, `${name}.bin`),
        new Uint8Array(fields[name].buffer, fields[name].byteOffset, fields[name].byteLength));
      await writeFile(join(directory, "activity.json"), JSON.stringify(activity));
      const receipt: Record<string, unknown> = { step, time_s: step * dt, elapsed_ms: performance.now() - startedAt,
        fieldDomain: { dimensions, origin_m: origin, scope: "authored-domain crop" }, outsideAuthoredLeaves,
        amount_m3: amount * h ** 3, centerOfMass_m: center.map(value => value / amount),
        kineticEnergy_J: .5 * scene.fluid.density_kg_m3 * h ** 3 * kinetic,
        maximumLiquidSpeed_m_s: maximumSpeed, wetBoundsAtHalfDensity_m: [wetMin, wetMax],
        densityRange: [minimumDensity, maximumDensity], nonFinite,
        nativeWidthActiveLeafHistogram: widths, allFineVerified: arm === "fine" ? violations.length === 0 : null,
        allFineViolations: violations, activeLeaves: activity.bricks.filter(brick => brick.active).length,
        acceptedTopologyGeneration: activity.acceptedTopologyGeneration, stats,
        bodies: scene.rigidBodies.map((description, index) => ({ description, pose: poses?.[index] ?? bodies[index] })),
      };
      // Write physical evidence first: a broken mesh still leaves diagnosable
      // fluid motion, native topology and rigid poses in this checkpoint.
      await writeFile(join(directory, "receipt.json"), JSON.stringify(receipt, null, 2));
      const mesh = await readPublishedCM12Mesh(device!, solver!.globalFineLevelSetSource, origin);
      await writeFile(join(directory, "mesh.bin"), new Uint8Array(mesh.mesh.buffer));
      Object.assign(receipt, { mesh: { vertices: mesh.mesh.length / 8, triangles: mesh.mesh.length / 24,
        generation: mesh.generation, activeCubes: mesh.activeCubes,
        surfaceMeshRefinement: solver!.globalFineLevelSetSource.surfaceMeshRefinement } });
      await writeFile(join(directory, "receipt.json"), JSON.stringify(receipt, null, 2));
      trace.push(receipt); await json("trace.json", trace);
      console.log(JSON.stringify({ phase: "capture", scene: sceneKey, arm, step, time_s: step * dt,
        amount_m3: receipt.amount_m3, widths, allFineVerified: receipt.allFineVerified,
        vertices: mesh.mesh.length / 8, elapsed_ms: performance.now() - startedAt }));
      assert.equal(nonFinite, 0, "physical fields must remain finite"); assert.deepEqual(errors, []);
    };
    await capture();
    for (step = 1; step <= steps[steps.length - 1]!; step++) {
      while (!solver.advanceTo(step * dt, bodies)) {
        if (performance.now() - startedAt > budgetMs) throw new Error(`Arm exceeded ${budgetMs}ms at step ${step}`);
        await new Promise(setImmediate);
      }
      await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps, step, "deferred preparation must not drop a step");
      if (stageReceipts.length) {
        const receipts = [];
        for (const {stage, buffer} of stageReceipts.splice(0)) {
          await buffer.mapAsync(GPUMapMode.READ);
          const f = new Float32Array(buffer.getMappedRange()).slice(), u = new Uint32Array(f.buffer);
          receipts.push({stage, retainedControl:[...f.subarray(0,4)], chainCount:f[4], failure:[...u.subarray(5, 5 + CM12_FAILURE_BYTES / 4)], completions:[...u.subarray(5 + CM12_FAILURE_BYTES / 4)]});
          buffer.unmap(); buffer.destroy();
        }
        await json(`publication-step-${step}.json`, receipts);
        const last = receipts.at(-1)!;
        await solver.assertSimulationHealthy();
        assert.equal(last.chainCount, step, `Current field must publish exactly once per encoded step ${step}`);
      }
      if (steps.includes(step)) await capture();
      await solver.assertSimulationHealthy();
      if (performance.now() - startedAt > budgetMs) throw new Error(`Arm exceeded ${budgetMs}ms at step ${step}`);
    }
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
    await json("completed.json", { completed: true, snapshots: trace.length, elapsed_ms: performance.now() - startedAt });
  } catch (error) {
    if (device && solver) await captureCurrentMapSnapshot(device, solver.fieldSnapshotSourceForQA, output, "current-map-failure");
    await json("failure.json", { completed: false, step, elapsed_ms: performance.now() - startedAt,
      message: error instanceof Error ? error.message : String(error), errors, completedSnapshots: trace.length });
    throw error;
  } finally {
    solver?.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
    await releaseWebGPUExclusiveLock();
  }
}
