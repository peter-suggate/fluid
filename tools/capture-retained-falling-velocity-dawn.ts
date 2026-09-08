/** Real production gravity/VEX QA. Never writes any physical GPU state. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { withRefinementRegionsFromQuery } from "../lib/core/editor-refinement-region";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { captureCurrentMapSnapshot } from "./current-map-snapshot-dawn";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { analyzeNativeVelocity, nativeLocalGradient, type NativeVelocitySample, type Point3, selectSnapshotVelocity } from "./retained-falling-velocity-analysis";

const option = (name: string, fallback: string) => process.argv.find(arg => arg.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const arm = option("arm", "fine"); assert.ok(arm === "fine" || arm === "coarse");
const stepCount = Number(option("steps", "6"));
assert.ok(Number.isSafeInteger(stepCount) && stepCount > 0 && stepCount <= 30);
const densityTransport = option("transport", "native-cm12");
assert.ok(densityTransport === "native-cm12" || densityTransport === "current-map");
const regionQuery = "0_0_0_25_66.6667_100_8_8";
let scene = withRefinementRegionsFromQuery(sceneDocument(getSceneDefinition("coarse-first-pool-impact-quarter")), regionQuery);
const h = scene.voxelDomain.finestCellSize_m, dt = scene.numerics.fixedDt_s;
const { width_m: width, height_m: height, depth_m: depth } = scene.container;
const dimensions = [width, height, depth].map(length => Math.round(length / h));
const origin: Point3 = [-width / 2, 0, -depth / 2];
if (arm === "fine") scene.fluid.refinementRegions = [{ id: "falling-velocity-all-fine", rule: "minimum-cell-size",
  minimumCellSize_cells: 1, maximumCellSize_cells: 1,
  min_m: { x: origin[0], y: 0, z: origin[2] }, max_m: { x: width / 2, y: height, z: depth / 2 } }];
const sphere = scene.fluid.initialLiquidVolumes![0]!; assert.equal(sphere.shape, "sphere");
assert.ok(sphere.shape === "sphere");
const seedCenter: Point3 = [sphere.center_m.x, sphere.center_m.y, sphere.center_m.z];
const gravity: Point3 = [scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.z];
const poolHeight = height * scene.container.fillFraction;
const values = resolveMethodValues(adaptiveMassMethod, "balanced", { selectorMode: "coarse-first", timeStep: "paper", densityTransport });
const config = { scene, values, arm, h, dt, dimensions, origin_m: origin, seedCenter_m: seedCenter,
  radius_m: sphere.radius_m, poolHeight_m: poolHeight, gravity_m_s2: gravity, originalRegionQuery: regionQuery,
  steps: Array.from({ length: stepCount }, (_, i) => i + 1), scope: "Unmodified production velocity/density under authored gravity. Readbacks after VEX and after velocity projection; no prescribed velocity, force, pressure, gamma, scalar, or retained coefficient writes.",
  expectedSplit: "VEX in step n uses velocity from n-1 completed gravity/projection updates. Exact kick-at-end translation has u=g*(n-1)*dt and position c0+g*dt^2*(n-1)*(n-2)/2 before its gather. This is a QA reference only.",
  velocityValidity: "VEX uses accepted depth 0..8 and effective w>0. Projection uses the destination collocated velocity bank, restricted to destination density>0.5; it does not measure an extended projected halo. Both captures precede frame commit, so projection reads source parity XOR 1.",
  selections: "sphere-wet: positive native mean with centre above midpoint between pool top and expected sphere bottom; sphere-halo: centre within diffuse outer radius + sqrt(3)*h, above pool top+h/2; pool: centre below pool top+h/2. Velocity statistics exclude samples outside the stage validity mask; group mass includes every geometrically selected native cell. Native-cell volumes weight fits. Local derivatives are physical face-neighbour least squares, separately from stored pressure divergence." };
if (process.argv.includes("--list")) console.log(JSON.stringify(config, null, 2));
else {
  const modulePath = process.env.WEBGPU_NODE_MODULE; assert.ok(modulePath, "Set WEBGPU_NODE_MODULE");
  const output = option("out", `artifacts/retained-falling-velocity/quarter/${arm}`);
  await mkdir(output, { recursive: true });
  const json = async (name: string, value: unknown) => writeFile(join(output, name), JSON.stringify(value, null, 2));
  const provenancePaths = ["lib/core/scenes.ts", "lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts",
    "tools/capture-retained-falling-velocity-dawn.ts", "tools/retained-falling-velocity-analysis.ts"];
  await json("configuration.json", config);
  await json("provenance.json", { gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    workingDiffNames: execFileSync("git", ["status", "--short"], { encoding: "utf8" }).trim().split("\n"),
    sha256: Object.fromEntries(await Promise.all(provenancePaths.map(async path =>
      [path, createHash("sha256").update(await readFile(path)).digest("hex")]))), capturedAt: new Date().toISOString() });
  await acquireWebGPUExclusiveLock("dawn-probe", `retained-falling-velocity-${arm}`);
  const start = performance.now(), errors: string[] = [], trace: unknown[] = [];
  let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined, gpu: GPU | undefined;
  const live = new Set<GPU>();
  type Source = WebGPUAdaptiveMassSolver["fieldSnapshotSourceForQA"];
  type Cell = { id: number; leaf: number; point: Point3; lower: readonly number[]; span: readonly number[]; volume: number };
  type Snapshot = { stage: string; step: number; cells: Cell[]; buffer: GPUBuffer;
    ranges: Record<string, { offset: number; count: number }>; topologyGeneration: number };
  const snapshots: Snapshot[] = [];
  const cellsFor = async (source: Source) => {
    const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
    const activity = await solver!.readGPUActivityPolicy();
    const resident = (solver!.sparseWorld as unknown as { resident: { topologyPageCapacity: number; initialWorldLeafCount: number } }).resident;
    const cells: Cell[] = [];
    for (const brick of activity.bricks) if (brick.active) {
      if (arm === "fine") assert.equal(8 * brick.spanBricks / brick.acceptedResolution, 1, "accepted all-fine native width");
      if (brick.leafId >= resident.initialWorldLeafCount) {
        assert.ok(brick.topologyPage !== undefined && brick.coordinate);
        const first = source.cellCapacity - 512 * resident.topologyPageCapacity + 512 * brick.topologyPage;
        for (let local = 0; local < 512; local++) {
          const lower = [local % 8, Math.floor(local / 8) % 8, Math.floor(local / 64)].map((q, axis) => q + 8 * brick.coordinate[axis]!);
          cells.push({ id: first + local, leaf: brick.leafId, lower, span: [1, 1, 1], volume: h ** 3,
            point: lower.map((q, axis) => origin[axis]! + h * (q + .5)) as unknown as Point3 });
        }
      } else {
        const range = words[11]! + 2 * (4 * brick.leafId + Math.log2(brick.acceptedResolution));
        for (let id = words[range]!; id < words[range]! + words[range + 1]!; id++) {
          const at = words[6]! + 8 * id;
          const span = [floats[at + 4]!, floats[at + 5]!, floats[at + 6]!];
          const lower = span.map((size, axis) => Math.round(floats[at + axis]! - size / 2));
          cells.push({ id, leaf: brick.leafId, lower, span, volume: floats[at + 3]! * h ** 3,
            point: [0, 1, 2].map(axis => origin[axis]! + h * floats[at + axis]!) as unknown as Point3 });
        }
      }
    }
    cells.sort((a, b) => a.id - b.id);
    assert.equal(new Set(cells.map(cell => cell.id)).size, cells.length, "unique accepted native IDs");
    return { cells, topologyGeneration: activity.acceptedTopologyGeneration };
  };
  const encodeSnapshot = (source: Source, cells: Cell[], encoder: GPUCommandEncoder, stage: string, step: number, topologyGeneration: number) => {
    const parts = [
      ...(["densityA", "densityB", "gammaA", "gammaB", "pressure", "divergence"] as const).map(name => ({ name, buffer: source.state, base: source.layout[name], stride: 1 })),
      ...(["cellVelocityA", "cellVelocityB"] as const).map(name => ({ name, buffer: source.state, base: source.layout[name], stride: 4 })),
      { name: "effective", buffer: source.effectiveTransportVelocity!, base: 0, stride: 4 },
      { name: "extensionDepth", buffer: source.activity, base: source.velocityExtensionLayout.acceptedDepthBaseWords, stride: 1 },
    ];
    assert.ok(source.effectiveTransportVelocity);
    const count = parts.reduce((sum, part) => sum + part.stride * cells.length, 0);
    const buffer = device!.createBuffer({ size: 4 * (count + 2), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const ranges: Snapshot["ranges"] = {}; let offset = 0;
    for (const part of parts) {
      ranges[part.name] = { offset, count: part.stride * cells.length };
      for (let first = 0; first < cells.length;) {
        let end = first + 1; while (end < cells.length && cells[end]!.id === cells[end - 1]!.id + 1) end++;
        encoder.copyBufferToBuffer(part.buffer, 4 * (part.base + part.stride * cells[first]!.id), buffer,
          4 * (offset + part.stride * first), 4 * part.stride * (end - first)); first = end;
      }
      offset += part.stride * cells.length;
    }
    ranges.scalarParity = { offset, count: 1 };
    encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.scalarParityWord), buffer, 4 * offset, 4);
    ranges.faceParity = { offset: offset + 1, count: 1 };
    encoder.copyBufferToBuffer(source.topologyArena, 4 * (source.frameControlBaseWords + source.faceParityWord), buffer, 4 * (offset + 1), 4);
    snapshots.push({ stage, step, cells, buffer, ranges, topologyGeneration });
  };
  const analyze = async (snapshot: Snapshot) => {
    const { buffer, stage, step, cells, ranges } = snapshot;
    let data: Float32Array;
    try { await buffer.mapAsync(GPUMapMode.READ); data = new Float32Array(buffer.getMappedRange()).slice(); }
    finally { if (buffer.mapState === "mapped") buffer.unmap(); buffer.destroy(); }
    const plane = (name: string) => data.subarray(ranges[name]!.offset, ranges[name]!.offset + ranges[name]!.count);
    const words = new Uint32Array(data.buffer);
    const sourceScalarParity = words[ranges.scalarParity!.offset]!;
    const sourceFaceParity = words[ranges.faceParity!.offset]!;
    const selection = selectSnapshotVelocity(stage, sourceScalarParity, sourceFaceParity);
    const parity = selection.densityParity;
    const density = plane(parity ? "densityB" : "densityA"), effective = plane(selection.velocityPlane);
    const depths = words.subarray(ranges.extensionDepth!.offset, ranges.extensionDepth!.offset + cells.length);
    const valid = cells.map((_, i) => selection.extended
      ? depths[i]! <= 8 && effective[4 * i + 3]! > 0
      : density[i]! > .5);
    const velocityUpdates = stage === "transport-velocity-extension" ? step - 1 : step;
    const positionUpdates = stage === "transport-velocity-extension" ? step - 1 : step;
    const expected: Point3 = gravity.map(value => value * velocityUpdates * dt) as unknown as Point3;
    const center = seedCenter.map((value, axis) => value + gravity[axis]! * dt ** 2 * positionUpdates * (positionUpdates - 1) / 2);
    const dividingY = .5 * (poolHeight + center[1]! - sphere.radius_m);
    const haloRadius = Math.sqrt(sphere.radius_m ** 2 + sphere.radius_m * h) + Math.sqrt(3) * h;
    const samples: NativeVelocitySample[] = cells.map((cell, i) => ({ id: cell.id, point: cell.point, volume: cell.volume,
      velocity: [effective[4 * i]! * h, effective[4 * i + 1]! * h, effective[4 * i + 2]! * h] }));
    assert.ok(samples.every(cell => [...cell.velocity, cell.volume].every(Number.isFinite)), "finite native physical samples");
    const owner = new Map<string, number>();
    cells.forEach((cell, i) => {
      for (let z = 0; z < cell.span[2]!; z++) for (let y = 0; y < cell.span[1]!; y++) for (let x = 0; x < cell.span[0]!; x++) {
        const key = `${cell.lower[0]! + x},${cell.lower[1]! + y},${cell.lower[2]! + z}`;
        assert.ok(!owner.has(key), `duplicate physical native support ${key}`); owner.set(key, i);
      }
    });
    const localGradients = cells.map((cell, i) => {
      if (!valid[i]) return undefined;
      const areas = new Map<number, number>();
      for (let axis = 0; axis < 3; axis++) for (const side of [-1, 1]) {
        const u = (axis + 1) % 3, v = (axis + 2) % 3;
        for (let a = 0; a < cell.span[u]!; a++) for (let b = 0; b < cell.span[v]!; b++) {
          const q = [...cell.lower]; q[axis]! += side < 0 ? -1 : cell.span[axis]!; q[u]! += a; q[v]! += b;
          const neighbor = owner.get(q.join(","));
          if (neighbor !== undefined && valid[neighbor]) areas.set(neighbor, (areas.get(neighbor) ?? 0) + h * h);
        }
      }
      return nativeLocalGradient(samples[i]!, [...areas].map(([index, area]) => ({ sample: samples[index]!, area })));
    });
    const selections = {
      sphereWet: (i: number) => density[i]! > 1e-6 && cells[i]!.point[1] > dividingY,
      sphereBulk: (i: number) => density[i]! >= .99 && cells[i]!.point[1] > dividingY,
      sphereHalo: (i: number) => Math.hypot(...cells[i]!.point.map((value, axis) => value - center[axis]!)) <= haloRadius
        && cells[i]!.point[1] > poolHeight + h / 2,
      pool: (i: number) => density[i]! > 1e-6 && cells[i]!.point[1] <= poolHeight + h / 2,
    };
    const groups = Object.fromEntries(Object.entries(selections).map(([name, include]) => {
      const indices = cells.flatMap((_, i) => include(i) && valid[i] ? [i] : []), selected = indices.map(i => samples[i]!);
      const gradients = indices.flatMap(i => localGradients[i] ? [{ id: cells[i]!.id, ...localGradients[i]! }] : []);
      let maxStrain = 0, maxDiv = 0, worstStrain: unknown;
      for (const value of gradients) {
        if (value.strainFrobenius_per_s >= maxStrain) { maxStrain = value.strainFrobenius_per_s; worstStrain = value; }
        maxDiv = Math.max(maxDiv, Math.abs(value.divergence_per_s));
      }
      return [name, { ...analyzeNativeVelocity(selected, name === "pool" ? [0, 0, 0] : expected),
        geometricCellCount: cells.filter((_, i) => include(i)).length,
        excludedInvalidCellCount: cells.filter((_, i) => include(i) && !valid[i]).length,
        extensionDepthHistogram: selection.extended ? indices.reduce<Record<string, number>>((out, i) => { const depth = String(depths[i]); out[depth] = (out[depth] ?? 0) + 1; return out; }, {}) : null,
        nativeWidthHistogram: indices.reduce<Record<string, number>>((out, i) => { const width = cells[i]!.span.join("x"); out[width] = (out[width] ?? 0) + 1; return out; }, {}),
        mass_m3: cells.reduce((sum, cell, i) => sum + (include(i) ? density[i]! * cell.volume : 0), 0),
        velocitySampleMass_m3: indices.reduce((sum, i) => sum + density[i]! * cells[i]!.volume, 0),
        localGradientCount: gradients.length, maximumLocalStrain_per_s: maxStrain,
        maximumLocalDivergence_per_s: maxDiv, worstLocalStrain: worstStrain,
        maximumStoredPressureDivergence: indices.reduce((max, i) => Math.max(max, Math.abs(plane("divergence")[i]!)), 0) }];
    }));
    const receipt = { stage, step, arm, sourceScalarParity, sourceFaceParity, scalarParity: parity, velocitySelection: selection, topologyGeneration: snapshot.topologyGeneration,
      expectedSphereCenter_m: center, expectedTranslationVelocity_m_s: expected, groups,
      totalNativeLiquidVolume_m3: cells.reduce((sum, cell, i) => sum + density[i]! * cell.volume, 0),
      elapsed_ms: performance.now() - start, nativeCells: cells.length };
    const stem = `step-${step}-${stage}`;
    await writeFile(join(output, `${stem}.bin`), new Uint8Array(data.buffer));
    await json(`${stem}.json`, { ...receipt, ranges, cells });
    trace.push(receipt); await json("trace.json", trace);
    console.log(JSON.stringify(receipt));
  };
  try {
    const dawn = await import(pathToFileURL(modulePath).href); Object.assign(globalThis, dawn.globals);
    gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
    const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
    device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
    device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined,
      progress => console.log(JSON.stringify({ phase: "initialization", elapsed_ms: performance.now() - start, progress }))) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady(); assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], dimensions);
    for (let step = 1; step <= stepCount; step++) {
      for (;;) {
        assert.ok(performance.now() - start < 240000, "capture exceeded 240s");
        // advanceTo(false) can initiate required frontier preparation without
        // encoding physics. Refresh ownership and buffers after that boundary;
        // retaining the previous callback would read a retired generation.
        await solver.waitForTopologyReady();
        assert.equal(solver.info.encodedSteps ?? 0, step - 1, "one accepted step per capture");
        assert.equal(snapshots.length, 0, "no partially encoded capture on retry");
        const source: Source = solver.fieldSnapshotSourceForQA;
        const { cells, topologyGeneration } = await cellsFor(source);
        const current: Source = solver.fieldSnapshotSourceForQA;
        assert.equal(current.state, source.state, "state stable during native ownership readback");
        assert.equal(current.topologyArena, source.topologyArena, "topology stable during native ownership readback");
        assert.equal(current.templateWords, source.templateWords, "template stable during native ownership readback");
        solver.setStageCaptureForQA((stage, encoder) => {
          if (stage === "transport-velocity-extension" || stage === "velocity-projection")
            encodeSnapshot(source, cells, encoder, stage, step, topologyGeneration);
        });
        const advanced = solver.advanceTo(step * dt, []);
        solver.setStageCaptureForQA(undefined);
        if (advanced) {
          assert.deepEqual(snapshots.map(snapshot => snapshot.stage),
            ["transport-velocity-extension", "velocity-projection"], "exactly two ordered stage captures per step");
          break;
        }
        assert.equal(solver.info.encodedSteps ?? 0, step - 1, "false advance must not encode a step");
        assert.equal(snapshots.length, 0, "false advance must not encode stage snapshots");
        await new Promise(setImmediate);
      }
      await solver.waitForTopologyReady(); assert.equal(solver.info.encodedSteps, step);
      await solver.assertSimulationHealthy();
      while (snapshots.length) await analyze(snapshots.shift()!);
      assert.deepEqual(errors, []);
    }
    await json("completed.json", { completed: true, snapshots: trace.length, elapsed_ms: performance.now() - start });
  } catch (error) {
    if (device && solver) await captureCurrentMapSnapshot(device, solver.fieldSnapshotSourceForQA, output, "current-map-failure");
    await json("failure.json", { elapsed_ms: performance.now() - start, errors, message: error instanceof Error ? error.message : String(error) });
    throw error;
  } finally {
    solver?.setStageCaptureForQA(undefined); for (const snapshot of snapshots) snapshot.buffer.destroy();
    solver?.destroy(); device?.destroy(); if (gpu) live.delete(gpu); await releaseWebGPUExclusiveLock();
  }
}
