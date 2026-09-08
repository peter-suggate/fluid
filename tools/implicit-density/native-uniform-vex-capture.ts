import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { GPUUniformVexMapCompiler, type UniformVexMapAttempt, type UniformVexSnapshotSource } from "./uniform-vex-map-gpu";
import type { V3 } from "./sparse-quadratic-pullback";
import type { WebGPUAdaptiveMassSolver } from "../../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

export const NATIVE_UNIFORM_VEX_FIXTURE = Object.freeze({ dimensions: [32, 32, 32] as V3,
  origin_m: [-.8, 0, -.8] as V3, h_m: .05, timeStep_s: 1 / 30,
  sphere: { center_m: [-.15, .8, 0] as V3, radius_m: .25, width_m: .05 },
  velocity_m_s: [.75, 0, 0] as V3, steps: 2 });

export interface SweptNativeCoverage {
  requiredCenterIds: Uint32Array;
  wetSupportIds: Uint32Array;
  sweptMinimumFine: V3;
  sweptMaximumFine: V3;
}
/** Fixture-only geometric oracle, never a recognizer input. Every fine cube
 * intersecting the analytic q>0 sphere is swept along the uniform trajectory.
 * The union of ALL trilinear native-center donors of those full swept cubes is
 * enumerated. This includes off-center roots, intermediate times, and the
 * source/destination interpolation halo, rather than checking a probe grid.
 * It proves this known imposed fixture's coverage, not arbitrary native rho. */
export function requiredSweptSphereNativeCenters(options: {
  dimensions: V3; origin: V3; h: number; center: V3; wetRadius: number; displacement: V3;
}): SweptNativeCoverage {
  const { dimensions: n, origin, h, center, wetRadius, displacement } = options;
  if (!(h > 0 && wetRadius > 0) || ![...origin, ...center, ...displacement, h, wetRadius].every(Number.isFinite)
    || n.some(value => !Number.isSafeInteger(value) || value < 1) || n[0] * n[1] * n[2] > 1_048_576) throw new Error("Invalid fixture coverage geometry");
  const required = new Set<number>(), wet: number[] = [], minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
  for (let z = 0; z < n[2]; z++) for (let y = 0; y < n[1]; y++) for (let x = 0; x < n[0]; x++) {
    const coordinate = [x, y, z];
    let distanceSquared = 0;
    for (let axis = 0; axis < 3; axis++) {
      const lo = origin[axis]! + coordinate[axis]! * h, hi = origin[axis]! + (coordinate[axis]! + 1) * h;
      const distance = Math.max(lo - center[axis]!, center[axis]! - hi, 0); distanceSquared += distance ** 2;
    }
    // Include the tangency envelope instead of dropping a rounded boundary.
    if (distanceSquared > wetRadius ** 2 + 64 * Number.EPSILON * Math.max(1, distanceSquared, wetRadius ** 2)) continue;
    wet.push(x + n[0] * (y + n[1] * z));
    const first = [0, 0, 0], last = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      const shift = displacement[axis]! / h;
      const lo = coordinate[axis]! + Math.min(0, shift), hi = coordinate[axis]! + 1 + Math.max(0, shift);
      minimum[axis] = Math.min(minimum[axis]!, lo); maximum[axis] = Math.max(maximum[axis]!, hi);
      // Native values live at j+1/2. Their hat basis is positive for
      // |x-(j+1/2)|<1. These closed interval bounds contain every such j.
      first[axis] = Math.floor(lo - .5); last[axis] = Math.ceil(hi - .5);
      if (first[axis]! < 0 || last[axis]! >= n[axis]!) throw new Error("Swept wet fixture needs native donors outside the physical grid");
    }
    for (let dz = first[2]!; dz <= last[2]!; dz++) for (let dy = first[1]!; dy <= last[1]!; dy++) for (let dx = first[0]!; dx <= last[0]!; dx++) {
      required.add(dx + n[0] * (dy + n[1] * dz));
    }
  }
  if (!wet.length) throw new Error("Empty fixture wet support");
  return { requiredCenterIds: Uint32Array.from([...required].sort((a, b) => a - b)), wetSupportIds: Uint32Array.from(wet),
    sweptMinimumFine: minimum as unknown as V3, sweptMaximumFine: maximum as unknown as V3 };
}
export function assertSweptNativeCoverage(mask: Uint32Array, proof: SweptNativeCoverage): void {
  for (const id of proof.requiredCenterIds) if (id >= mask.length || mask[id] === 0) {
    throw new Error(`Missing actual VEX native-center coverage for swept wet donor ${id}`);
  }
}

/** Narrow access to existing private QA sources. There are no runtime edits.
 * The compiler independently validates their copied headers and generations. */
function nativeSource(solver: WebGPUAdaptiveMassSolver): UniformVexSnapshotSource {
  const publicSource = solver.fieldSnapshotSourceForQA;
  const resident = (solver.sparseWorld as unknown as { resident: {
    activity: GPUBuffer; parameters: GPUBuffer;
    velocityExtensionLayout: { headerBaseWords: number; acceptedDepthBaseWords: number; cellCapacity: number; packetCapacity: number };
  } }).resident;
  assert.ok(publicSource.effectiveTransportVelocity, "actual resident has no VEX plane");
  assert.equal(resident.velocityExtensionLayout.cellCapacity, publicSource.cellCapacity, "VEX/native capacity agreement");
  return { topologyArena: publicSource.topologyArena, activity: resident.activity, parameters: resident.parameters,
    state: publicSource.state, effectiveTransportVelocity: publicSource.effectiveTransportVelocity,
    cellCapacity: publicSource.cellCapacity, templateCellCount: publicSource.templateWords[2]!, templateCellBaseWords: publicSource.templateWords[6]!,
    topologyWorklistBaseWords: publicSource.topologyWorklistBaseWords, frameControlBaseWords: publicSource.frameControlBaseWords,
    velocityExtension: resident.velocityExtensionLayout, solidCellOpenBaseWords: publicSource.layout.solidCellOpen,
    solidVoxelCellOpenBaseWords: publicSource.layout.solidVoxelCellOpen };
}

/** Caller owns the Dawn/browser lease and GPU device. This prescribed QA run
 * does not drive a new scalar field or assert that production density moves
 * correctly. It obtains the first map/coverage receipt from real native VEX. */
export async function captureNativeUniformVexMaps(device: GPUDevice, outputDirectory: string) {
  const [{ sceneDocument }, { getSceneDefinition }, { resolveMethodValues }, { adaptiveMassMethod }] = await Promise.all([
    import("../../lib/core/scene-definition"), import("../../lib/core/scenes"),
    import("../../lib/core/method-contract"), import("../../lib/methods/adaptive-mass/method"),
  ]);
  const config = NATIVE_UNIFORM_VEX_FIXTURE, scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-quarter"));
  scene.sceneId = "qa-native-uniform-vex-map";
  scene.container.height_m = 1.6; scene.container.fillFraction = 0; scene.solidVoxels = []; scene.rigidBodies = [];
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 }; scene.fluid.dynamicViscosity_Pa_s = 0; scene.fluid.surfaceTension_N_m = 0;
  scene.fluid.initialVelocity_m_s = { x: .75, y: 0, z: 0 };
  scene.fluid.initialLiquidVolumes = [{ shape: "sphere", center_m: { x: -.15, y: .8, z: 0 }, radius_m: .25 }];
  scene.fluid.refinementRegions = [{ id: "imposed-all-fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
    min_m: { x: -.8, y: 0, z: -.8 }, max_m: { x: .8, y: 1.6, z: .8 } }];
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    selectorMode: "coarse-first", timeStep: "paper", gammaDiffusion: "off", surfaceSharpening: "off",
  });
  await mkdir(outputDirectory, { recursive: true });
  const json = (name: string, value: unknown) => writeFile(join(outputDirectory, name), JSON.stringify(value, null, 2));
  const paths = ["tools/implicit-density/native-uniform-vex-capture.ts", "tools/implicit-density/uniform-vex-map-gpu.ts",
    "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts", "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    "lib/methods/adaptive-mass/sparse-cm12-velocity-extension.wgsl.ts"];
  await json("configuration.json", { ...config, scene, values,
    scope: "Two actual production steps with prescribed cell/face velocity, gamma1 and pressure0. Only new map/coverage readbacks are QA. No new scalar field is advanced." });
  await json("provenance.json", { gitHead: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8" }).trim(),
    capturedAt: new Date().toISOString(), sourceSha256: Object.fromEntries(await Promise.all(paths.map(async path =>
      [path, createHash("sha256").update(await readFile(path)).digest("hex")]))),
    captureStage: "transport-velocity-extension", captureOrder: "snapshot and compiler encoded before conservative transport/gather" });
  let solver: WebGPUAdaptiveMassSolver | undefined;
  const pending: UniformVexMapAttempt[] = [], receipts: Record<string, unknown>[] = [], started = performance.now();
  try {
    const compiler = await GPUUniformVexMapCompiler.create(device);
    solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values, undefined,
      progress => console.log(JSON.stringify({ phase: "native-uniform-vex-initialization", elapsed_ms: performance.now() - started, progress }))) as WebGPUAdaptiveMassSolver;
    await solver.waitForSimulationReady(); assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], config.dimensions);
    for (let step = 1; step <= config.steps; step++) {
      const source = solver.fieldSnapshotSourceForQA, cellVelocity = new Float32Array(4 * source.cellCapacity), faces = new Float32Array(source.rowCapacity);
      for (let id = 0; id < source.cellCapacity; id++) cellVelocity.set([config.velocity_m_s[0] / config.h_m, 0, 0, 1], 4 * id);
      for (let row = 0; row < source.rowCapacity; row++) {
        const axis = source.templateWords[source.templateWords[7]! + source.rowCapacity + row]! >>> 30;
        assert.ok(axis < 3, "bounded fixture has only static authored face rows"); faces[row] = config.velocity_m_s[axis]! / config.h_m;
      }
      for (const base of [source.layout.cellVelocityA, source.layout.cellVelocityB]) device.queue.writeBuffer(source.state, 4 * base, cellVelocity);
      for (const base of [source.layout.faceA, source.layout.faceB]) device.queue.writeBuffer(source.state, 4 * base, faces);
      const unitGamma = new Float32Array(source.cellCapacity).fill(1);
      for (const base of [source.layout.gammaA, source.layout.gammaB]) device.queue.writeBuffer(source.state, 4 * base, unitGamma);
      device.queue.writeBuffer(source.state, 4 * source.layout.pressure, new Float32Array(source.cellCapacity));
      assert.ok(source.effectiveTransportVelocity); device.queue.writeBuffer(source.effectiveTransportVelocity, 0, cellVelocity);
      solver.setStageCaptureForQA((stage, encoder) => {
        if (stage === "transport-velocity-extension") pending.push(compiler.encodeSnapshotAndCompile(encoder, nativeSource(solver!), config.dimensions));
      });
      while (!solver.advanceTo(step * config.timeStep_s, [])) {
        assert.ok(performance.now() - started < 240000, "native imposed-flow capture exceeded its existing240s fixture budget");
        await new Promise(setImmediate);
      }
      solver.setStageCaptureForQA(undefined); await solver.waitForTopologyReady();
      assert.equal(solver.info.encodedSteps, step); await solver.assertSimulationHealthy();
      assert.equal(pending.length, 1, "exactly one pre-gather snapshot per native step");
      const attempt = pending.pop()!;
      try {
        const raw = await attempt.readRawReceiptForQA(), mask = await attempt.readCoverageForQA();
        await writeFile(join(outputDirectory, `step-${step}-map.bin`), new Uint8Array(raw.buffer));
        await writeFile(join(outputDirectory, `step-${step}-coverage.bin`), new Uint8Array(mask.buffer));
        await json(`step-${step}-copied-headers.json`, await attempt.readCopiedHeadersForQA());
        const receipt = await attempt.readReceiptForQA();
        const center = config.sphere.center_m.map((value, axis) => value + (step - 1) * config.timeStep_s * config.velocity_m_s[axis]!) as unknown as V3;
        const expectedDisplacement = config.velocity_m_s.map(value => value * config.timeStep_s) as unknown as V3;
        const compiledDisplacement = receipt.map.translation.map(value => -value) as unknown as V3;
        const proof = (displacement: V3) => requiredSweptSphereNativeCenters({ dimensions: config.dimensions, origin: config.origin_m,
          h: config.h_m, center, wetRadius: Math.sqrt(config.sphere.radius_m ** 2 + config.sphere.radius_m * config.sphere.width_m), displacement });
        const expectedCoverage = proof(expectedDisplacement), compiledCoverage = proof(compiledDisplacement);
        // Coverage is a prerequisite to the velocity/map comparison. No
        // center-only or hardcoded sphere ROI is used in the map compiler.
        assertSweptNativeCoverage(mask, expectedCoverage); assertSweptNativeCoverage(mask, compiledCoverage);
        for (let axis = 0; axis < 3; axis++) {
          assert.ok(Math.abs(receipt.velocity_m_s[axis]! - config.velocity_m_s[axis]!) <= 2e-6, "actual native VEX physical velocity");
          assert.ok(Math.abs(compiledDisplacement[axis]! - expectedDisplacement[axis]!) <= 2e-7, "compiled physical departure map");
        }
        assert.equal(receipt.timeStep_s, Math.fround(config.timeStep_s)); assert.equal(receipt.finestCellSize_m, Math.fround(config.h_m));
        const row = { step, elapsed_ms: performance.now() - started, ...receipt,
          expectedWetSourceCenter_m: center, expectedDisplacement_m: expectedDisplacement,
          expectedWetSupports: expectedCoverage.wetSupportIds.length, expectedRequiredNativeCenters: expectedCoverage.requiredCenterIds.length,
          compiledRequiredNativeCenters: compiledCoverage.requiredCenterIds.length,
          coverageSha256: createHash("sha256").update(new Uint8Array(mask.buffer)).digest("hex"),
          coverageProof: "All native-center trilinear donors of complete fine cubes intersecting the analytic wet sphere, swept over the whole uniform trajectory; expected and compiled maps both covered.",
          boundaryScope: "The proof throws if any requested donor leaves the physical grid; all recognized VEX cells have unit open fractions. No solid/world clipping is inferred." };
        receipts.push(row); await json(`step-${step}-receipt.json`, row); console.log(JSON.stringify(row));
      } finally { attempt.destroy(); }
    }
    await json("completed.json", { completed: true, steps: receipts.length, elapsed_ms: performance.now() - started, receipts });
    return receipts;
  } catch (error) {
    await json("failure.json", { message: error instanceof Error ? error.message : String(error), elapsed_ms: performance.now() - started }); throw error;
  } finally { solver?.setStageCaptureForQA(undefined); pending.forEach(attempt => attempt.destroy()); solver?.destroy(); }
}
