import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { rasterMeshSymmetryMetrics } from "../lib/harness/raster-mesh-symmetry";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";
import { readPublishedCM12Mesh } from "../tools/sparse-cm12-published-mesh";
import { measurePoolImpactMesh, measurePublishedPoolImpact, poolImpactBudgets,
  poolImpactOracle, POOL_IMPACT_REGION_QUERY, POOL_IMPACT_SCENES } from "../tools/implicit-density/pool-impact-oracle";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
const requestedScene = process.env.POOL_ANALYTIC_SCENE;
if (requestedScene && !POOL_IMPACT_SCENES.some(id => id === requestedScene)) throw new Error(`Unknown catalog scene ${requestedScene}`);

for (const id of POOL_IMPACT_SCENES.filter(id => !requestedScene || id === requestedScene)) (dawnModule ? test : test.skip)(
  `${id}: analytic pool and complete sphere survive actual minmax8 region edits with native integrals`,
  { timeout: 600_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", `analytic-${id}`);
    let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]); live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const oracle = poolImpactOracle(id), { scene, h } = oracle, budgets = poolImpactBudgets(oracle);
      const [nx, ny, nz] = oracle.dimensions;
      solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined,
        { ...adaptiveMassSolverOptions({ selectorMode: "coarse-first" }), pressureIterations: 8 }, () => {});
      await solver.waitForSimulationReady();
      assert.deepEqual([solver.info.nx, solver.info.ny, solver.info.nz], oracle.dimensions);
      assert.equal(solver.globalFineLevelSetSource.surfaceMeshRefinement, 2, "use shipping balanced mesh quality");
      const reports: unknown[] = [];
      let baselinePhi: Float32Array | undefined, nativeFineDensity: Float32Array | undefined;
      let originalMass: number | undefined;
      const originalRegion = scene.fluid.refinementRegions![0]!;
      const fullRegion = (width: number) => [{ id: "all-cells", rule: "minimum-cell-size" as const,
        minimumCellSize_cells: width, maximumCellSize_cells: width,
        min_m: { x: oracle.origin[0], y: 0, z: oracle.origin[2] },
        max_m: { x: -oracle.origin[0], y: ny * h, z: -oracle.origin[2] } }];
      // Start from the actual URL scene, then visit an independently fine native
      // partition to retain its integral receipt before repeated paused edits.
      const partitions = ["original", 1, "original", 4, "original", 2, "original", 1,
        "original", 4, "original", 2, "original"] as const;
      for (let edit = 0; edit < partitions.length; edit++) {
        const partition = partitions[edit]!;
        if (edit) {
          const next = structuredClone(scene);
          next.fluid.refinementRegions = partition === "original" ? [originalRegion] : fullRegion(partition);
          solver.applySceneUniforms(next); await solver.refreshSceneTopology();
        }
        const label = `${id} edit=${edit} partition=${partition}`;
        const activity = await solver.readGPUActivityPolicy();
        const fields = await solver.readDiagnosticFields(true);
        const phi = (await readPublishedCM12Field(device, solver)).values;
        const fieldMetrics = measurePublishedPoolImpact(phi, oracle);
        const mass = fields.density.reduce((sum, rho) => sum + rho, 0);
        originalMass ??= mass;
        assert.equal(solver.info.encodedSteps ?? 0, 0, `${label}: topology changes must not advance time`);
        assert.equal(activity.acceptedSteps, 0, label); assert.equal(activity.faultFlags, 0, label);
        assert.ok(fields.velocity.every(v => v === 0), `${label}: zero initial velocity`);
        assert.ok(Math.abs(mass - originalMass) < 1e-3, `${label}: native mass changed ${mass - originalMass}`);
        if (partition === 1 && !nativeFineDensity) nativeFineDensity = fields.density.slice();
        let checkedNativeCells = 0, region8Cells = 0, maximumNativeMeanError = 0;
        const nativeWidths = new Set<number>();
        for (const brick of activity.bricks.filter(b => b.active)) {
          const span = 8 * brick.spanBricks, width = span / brick.acceptedResolution;
          const origin = brick.coordinate.map(q => 8 * q);
          if (origin.some((q, axis) => q < 0 || q + span > oracle.dimensions[axis]!)) continue;
          nativeWidths.add(width);
          if (partition !== "original") assert.equal(width, partition, `${label}: native partition must actually change`);
          for (let z = origin[2]!; z < origin[2]! + span; z += width)
            for (let y = origin[1]!; y < origin[1]! + span; y += width)
              for (let x = origin[0]!; x < origin[0]! + span; x += width) {
                const cellLo = [oracle.origin[0] + x * h, y * h, oracle.origin[2] + z * h];
                if (partition === "original" && cellLo[0]! >= originalRegion.min_m.x - 1e-7
                  && cellLo[1]! >= originalRegion.min_m.y - 1e-7 && cellLo[2]! >= originalRegion.min_m.z - 1e-7
                  && cellLo[0]! + width * h <= originalRegion.max_m.x + 1e-7
                  && cellLo[1]! + width * h <= originalRegion.max_m.y + 1e-7
                  && cellLo[2]! + width * h <= originalRegion.max_m.z + 1e-7) {
                  assert.equal(width, 8, `${label}: original minmax8 box must reach native cells`); region8Cells++;
                }
                if (nativeFineDensity) {
                  let reference = 0;
                  for (let dz = 0; dz < width; dz++) for (let dy = 0; dy < width; dy++) for (let dx = 0; dx < width; dx++)
                    reference += nativeFineDensity[x + dx + nx * (y + dy + ny * (z + dz))]!;
                  const actual = fields.density[x + nx * (y + ny * z)]!;
                  maximumNativeMeanError = Math.max(maximumNativeMeanError, Math.abs(actual - reference / width ** 3));
                }
                checkedNativeCells++;
              }
        }
        assert.ok(checkedNativeCells > 0, `${label}: missing physical native cells`);
        if (partition === "original") assert.ok(region8Cells > 0 && nativeWidths.size > 1,
          `${label}: original local region must leave a mixed physical topology`);
        assert.ok(maximumNativeMeanError <= budgets.nativeMean, `${label}: native means lost retained integrals ${maximumNativeMeanError}`);
        let maximumPausedSampleChange_m = 0;
        if (!baselinePhi) baselinePhi = phi;
        else for (let q = 0; q < phi.length; q++) if (Number.isFinite(baselinePhi[q]) && Math.abs(baselinePhi[q]!) < 1.5 * h) {
          assert.ok(Number.isFinite(phi[q]), `${label}: published interface sample ${q} disappeared`);
          maximumPausedSampleChange_m = Math.max(maximumPausedSampleChange_m, Math.abs(phi[q]! - baselinePhi[q]!));
        }
        const report = { id, regionQuery: POOL_IMPACT_REGION_QUERY, edit, partition, budgets,
          fieldMetrics, mass, checkedNativeCells, region8Cells, nativeWidths: [...nativeWidths].sort((a, b) => a - b),
          maximumNativeMeanError, maximumPausedSampleChange_m };
        reports.push(report); console.log(JSON.stringify(report));
        assert.equal(fieldMetrics.missingOrExtraCrossingColumns, 0, `${label}: every pool/upper/lower sphere crossing: ${JSON.stringify(fieldMetrics.firstBadColumn)}`);
        assert.equal(fieldMetrics.missingAnalyticSamples, 0, `${label}: continuous interface band must be published`);
        assert.ok(fieldMetrics.maximumSamplePrecisionBudgetRatio <= 1,
          `${label}: published values must equal the analytic defining field within f16/f32 precision: ${fieldMetrics.maximumAnalyticSampleError_m} m`);
        assert.ok(fieldMetrics.sphereCrossings > 0, `${label}: ball must be present`);
        assert.ok(fieldMetrics.maximumPoolHeightError_m <= budgets.poolPlanarity_m, `${label}: pool displaced ${fieldMetrics.maximumPoolHeightError_m} m`);
        assert.ok(fieldMetrics.maximumSphereDistanceError_m <= budgets.spherePublishedRadial_m, `${label}: published ball radial error ${fieldMetrics.maximumSphereDistanceError_m} m`);
        assert.ok(maximumPausedSampleChange_m <= budgets.pausedPublication_m, `${label}: paused field depends on native partition`);
        // Fresh production extraction at startup and after both edit cycles.
        // Native integrals above keep a cached or decorative mesh from passing.
        if (edit === 0 || edit === 6 || edit === 12) {
          const capture = await readPublishedCM12Mesh(device, solver.globalFineLevelSetSource, oracle.origin);
          const meshMetrics = measurePoolImpactMesh(capture.mesh, oracle);
          const topology = rasterMeshSymmetryMetrics(capture.mesh, capture.mesh.length / 8, {
            minimum: oracle.origin, maximum: [-oracle.origin[0], ny * h, -oracle.origin[2]], tolerance: 1e-5 });
          console.log(JSON.stringify({ label, meshMetrics, generation: capture.generation,
            nonManifoldEdges: topology.nonManifoldEdgeCount, interiorOpenEdges: topology.interiorOpenEdgeCount }));
          reports.push({ edit, meshMetrics, generation: capture.generation });
          const output = process.env.POOL_ANALYTIC_OUTPUT;
          if (output) {
            const directory = join(output, id, `edit-${edit}`); await mkdir(directory, { recursive: true });
            await writeFile(join(directory, "mesh.bin"), new Uint8Array(capture.mesh.buffer));
            await writeFile(join(directory, "phi.bin"), new Uint8Array(phi.buffer));
            await writeFile(join(directory, "receipt.json"), JSON.stringify({ oracle, budgets, report, meshMetrics }, null, 2));
          }
          assert.ok(meshMetrics.poolVertices > 0 && meshMetrics.sphereVertices > 0, `${label}: both physical components must be emitted`);
          assert.equal(meshMetrics.unexpectedInteriorVertices, 0, `${label}: no invented interior sheet`);
          assert.equal(topology.nonFiniteCount, 0, label); assert.equal(topology.nonManifoldEdgeCount, 0, label);
          assert.ok(meshMetrics.maximumPoolHeightError_m <= budgets.poolPlanarity_m, `${label}: mesh pool is not planar`);
          assert.ok(meshMetrics.maximumPoolNormalError < .001, `${label}: calm pool normal changed`);
          assert.ok(Math.abs(meshMetrics.upwardPoolArea_m2 - scene.container.width_m * scene.container.depth_m) < 1e-4,
            `${label}: pool must cover the full authored top area ${meshMetrics.upwardPoolArea_m2}`);
          assert.equal(meshMetrics.downwardPoolArea_m2, 0, `${label}: pool has reversed triangles`);
          assert.ok(meshMetrics.maximumSphereVertexError_m <= budgets.sphereMeshVertexRadial_m, `${label}: mesh sphere vertex distance ${meshMetrics.maximumSphereVertexError_m}`);
          assert.ok(meshMetrics.maximumSphereInteriorError_m <= budgets.sphereMeshInteriorRadial_m, `${label}: mesh sphere triangle distance ${meshMetrics.maximumSphereInteriorError_m}`);
          assert.ok(meshMetrics.maximumSphereNormalError <= budgets.sphereMeshNormalVector, `${label}: mesh sphere normal ${meshMetrics.maximumSphereNormalError}`);
        }
      }
      // A frozen t=0 renderer or field cannot satisfy the acceptance: accepted
      // transport must move both native state and newly published sphere data.
      while (!solver.advanceTo(1 / 30, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      const evolved = await solver.readDiagnosticFields(true);
      const evolvedPhi = (await readPublishedCM12Field(device, solver)).values;
      assert.equal(solver.info.encodedSteps, 1, "resume must accept a physical step");
      assert.ok(evolved.velocity.some(v => Math.abs(v) > 1e-5), "gravity must change physical velocity");
      assert.ok(evolvedPhi.some((v, q) => Number.isFinite(v) && Number.isFinite(baselinePhi![q])
        && Math.abs(v - baselinePhi![q]!) > 1e-5), "the published field must evolve after resuming");
      await solver.assertSimulationHealthy(); assert.deepEqual(errors, []);
      if (process.env.POOL_ANALYTIC_OUTPUT) {
        await mkdir(join(process.env.POOL_ANALYTIC_OUTPUT, id), { recursive: true });
        await writeFile(join(process.env.POOL_ANALYTIC_OUTPUT, id, "report.json"), JSON.stringify({ id, reports, resumedSteps: solver.info.encodedSteps }, null, 2));
      }
    } finally {
      solver?.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
