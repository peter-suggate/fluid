import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { cloneScene, defaultScene } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { readPublishedCM12Field } from "../tools/sparse-cm12-published-field";
import { measurePartitionAnalyticField } from "../tools/implicit-density/partition-oracle";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const live = new Set<GPU>();
const fixtures = ["flat", "quadratic", "sphere-pool", "sharp-box"] as const;

/** Every interior vertical ray, including both sides of suspended components.
 * Samples live at finest-cell centers. Only adjacent finite samples define
 * an interpolated root; missing publication therefore changes root counts.
 */
function verticalRoots(phi: Float32Array, nx: number, ny: number, nz: number, h: number) {
  const columns: number[][] = [];
  for (let z = 1; z < nz - 1; z++) for (let x = 1; x < nx - 1; x++) {
    const roots: number[] = [];
    for (let y = 0; y < ny - 1; y++) {
      const a = phi[x + nx * (y + ny * z)]!, b = phi[x + nx * (y + 1 + ny * z)]!;
      if (!Number.isFinite(a) || !Number.isFinite(b) || (a > 0 && b > 0) || (a < 0 && b < 0) || a === b) continue;
      const root = (y + .5 - a / (b - a)) * h;
      if (!roots.length || Math.abs(root - roots[roots.length - 1]!) > 1e-12) roots.push(root);
    }
    columns.push(roots);
  }
  return columns;
}

for (const fixture of fixtures) (dawnModule ? test : test.skip)(
  `${fixture}: retained field preserves native integrals and published interface through repeated paused partitions`,
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test", `retained-field-${fixture}`);
    let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href);
      Object.assign(globalThis, dawn.globals);
      gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      live.add(gpu!);
      const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
      device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
      const errors: string[] = [];
      device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
      const scene = cloneScene(defaultScene);
      scene.rigidBodies = []; scene.solidVoxels = [];
      scene.container = { ...scene.container, width_m: .8, height_m: .8, depth_m: .8,
        fillFraction: fixture === "sphere-pool" || fixture === "sharp-box" ? .25 : .43 };
      scene.voxelDomain.finestCellSize_m = .05;
      scene.fluid.initialCondition = "tank-fill";
      scene.fluid.initialHeightField = undefined;
      scene.fluid.initialLiquidVolumes = [];
      // An empty seed array overrides the base fill with empty occupancy.
      // Remove the optional seed authority to retain the authored pool.
      delete scene.fluid.initialBrickSeeds_m;
      delete scene.fluid.initialBrickSeedsAdditive;
      if (fixture === "quadratic") scene.fluid.initialHeightField = {
        kind: "quadratic", baseHeight_m: .32, center_m: { x: 0, z: 0 },
        curvatureX_mInv: .6, curvatureZ_mInv: .35,
      };
      if (fixture === "sphere-pool") scene.fluid.initialLiquidVolumes = [
        { shape: "sphere", center_m: { x: .03, y: .51, z: -.02 }, radius_m: .14 },
      ];
      if (fixture === "sharp-box") scene.fluid.initialLiquidVolumes = [
        { shape: "box", min_m: { x: -.17, y: .16, z: -.13 }, max_m: { x: .13, y: .57, z: .18 } },
      ];
      const regions = (width: number, mixed = false) => [{ id: "partition", rule: "minimum-cell-size" as const,
        minimumCellSize_cells: width, maximumCellSize_cells: width,
        min_m: { x: -.4, y: 0, z: -.4 }, max_m: { x: mixed ? 0 : .4, y: .8, z: .4 } },
      ...(mixed ? [{ id: "other-half", rule: "minimum-cell-size" as const,
        // Adjacent width-one/width-four constraints have no 2:1-graded
        // solution. Exercise the finest admissible mixed seam explicitly.
        minimumCellSize_cells: 2, maximumCellSize_cells: 2,
        min_m: { x: 0, y: 0, z: -.4 }, max_m: { x: .4, y: .8, z: .4 } }] : [])];
      scene.fluid.refinementRegions = regions(1);
      solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined,
        { ...adaptiveMassSolverOptions({ selectorMode: "coarse-first" }), pressureIterations: 8 }, () => {});
      await solver.waitForSimulationReady();
      let resetPhi: Float32Array | undefined;
      for (const phase of ["reset", "resumed"] as const) {
        if (phase === "resumed") {
          while (!solver.advanceTo(1 / 30, [])) await new Promise(setImmediate);
          await solver.waitForTopologyReady();
          // Establish the post-transport fine partition before comparing its
          // subsequent zero-time edits. This also verifies the field evolves.
          const fine = structuredClone(scene); fine.fluid.refinementRegions = regions(1);
          solver.applySceneUniforms(fine); await solver.refreshSceneTopology();
        }
        const expectedSteps = phase === "reset" ? 0 : 1;
        const initial = await solver.readDiagnosticFields(true);
        const initialPhi = (await readPublishedCM12Field(device, solver)).values;
        const baselineActivity = await solver.readGPUActivityPolicy();
        const baselineBricks = baselineActivity.bricks.filter(b => b.active && b.coordinate.every(q => q >= 0 && q < 2));
        assert.ok(baselineBricks.length > 0);
        assert.ok(baselineBricks.every(b => 8 * b.spanBricks / b.acceptedResolution === 1),
          "restriction oracle must start from native finest-cell integrals");
        if (phase === "reset") resetPhi = initialPhi;
        else if (fixture === "sphere-pool" || fixture === "sharp-box") assert.ok(initialPhi.some((phi, i) =>
          Number.isFinite(phi) && Number.isFinite(resetPhi![i]) && Math.abs(phi - resetPhi![i]!) > 1e-5),
          "the production field must evolve when gravity resumes");
        const [nx, ny, nz]: [number, number, number] = [solver.info.nx, solver.info.ny, solver.info.nz];
        const h = scene.voxelDomain.finestCellSize_m;
        if (phase === "reset") {
          const analytic = measurePartitionAnalyticField(fixture, initialPhi, [nx, ny, nz], h);
          console.log(JSON.stringify({ fixture, phase, analytic }));
          assert.ok(analytic.sampleCount > 100, "independent analytic interface is nontrivial");
          assert.equal(analytic.missingSamples, 0, "every analytic interface sample must publish");
          assert.ok(analytic.maximumSamplePrecisionRatio <= 1,
            `${fixture}: initial scalar differs from authored analytic field: ${analytic.maximumSamplePrecisionRatio}`);
          assert.equal(analytic.unresolvedAnalyticColumns, 0, "the finest lattice resolves every authored crossing");
          assert.equal(analytic.changedCrossingColumns, 0, "all analytic component crossings must publish");
          assert.equal(analytic.observedCrossings, analytic.analyticCrossings);
          assert.ok(analytic.maximumRootPrecisionRatio <= 1,
            `${fixture}: zero crossings differ from independently sampled analytic field`);
          assert.ok(analytic.maximumZeroDistance_m <= analytic.surfaceBudget_m,
            `${fixture}: continuous analytic surface distance ${analytic.maximumZeroDistance_m} m`);
        }
        const baselineRoots = verticalRoots(initialPhi, nx, ny, nz, h);
        assert.ok(baselineRoots.some(roots => roots.length > 0), "fixture has visible vertical crossings");
        const at = (x: number, y: number, z: number) => x + nx * (y + ny * z);
        const initialMass = initial.density.reduce((sum, rho) => sum + rho, 0);
        assert.ok(initialMass > 0);
        // Compare actual published scalar samples near every component of the
        // interface; a retained renderer mesh cannot satisfy the native checks.
        const probes = [...initialPhi.keys()].filter(i => Number.isFinite(initialPhi[i]) && Math.abs(initialPhi[i]!) < .075);
        assert.ok(probes.length > 30, "fixture must publish a nontrivial interface");
        for (let cycle = 0; cycle < (phase === "reset" ? 3 : 1); cycle++) for (const [width, mixed] of [[4, false], [2, false], [1, true], [1, false]] as const) {
          const next = structuredClone(scene); next.fluid.refinementRegions = regions(width, mixed);
          solver.applySceneUniforms(next); await solver.refreshSceneTopology();
          const activity = await solver.readGPUActivityPolicy();
          const fields = await solver.readDiagnosticFields(true);
          const phi = (await readPublishedCM12Field(device, solver)).values;
          const label = `${fixture} phase=${phase} cycle=${cycle} width=${width} mixed=${mixed}`;
          assert.equal(solver.info.encodedSteps ?? 0, expectedSteps, label);
          assert.equal(activity.acceptedSteps, expectedSteps, label);
          assert.equal(activity.faultFlags, 0, label);
          assert.ok(fields.velocity.every(Number.isFinite), `${label}: finite velocity`);
          if (phase === "reset") assert.ok(fields.velocity.every(v => v === 0), `${label}: no physics advance`);
          assert.ok(Math.abs(fields.density.reduce((sum, rho) => sum + rho, 0) - initialMass) < 1e-3,
            `${label}: physical mass changed`);
          let checkedCells = 0;
          for (const brick of activity.bricks.filter(b => b.active)) {
            const span = 8 * brick.spanBricks;
            const nativeWidth = span / brick.acceptedResolution;
            const origin = brick.coordinate.map(q => 8 * q);
            if (origin.some((q, axis) => q < 0 || q + span > [nx, ny, nz][axis]!)) continue;
            const expectedWidth: number = mixed && origin[0]! >= nx / 2 ? 2 : width;
            assert.equal(nativeWidth, expectedWidth, `${label}: topology must actually change`);
            for (let z = origin[2]!; z < origin[2]! + span; z += nativeWidth)
              for (let y = origin[1]!; y < origin[1]! + span; y += nativeWidth)
                for (let x = origin[0]!; x < origin[0]! + span; x += nativeWidth) {
                  let expectedMass = 0;
                  for (let dz = 0; dz < nativeWidth; dz++) for (let dy = 0; dy < nativeWidth; dy++) for (let dx = 0; dx < nativeWidth; dx++)
                    expectedMass += initial.density[at(x + dx, y + dy, z + dz)]!;
                  const expectedMean = expectedMass / nativeWidth ** 3;
                  // Match the resident integral guard. A whole CM12 mass
                  // quantum must not hide in a zero-time native remap.
                  assert.ok(Math.abs(fields.density[at(x, y, z)]! - expectedMean) < 2e-6,
                    `${label}: native mean at ${x},${y},${z} is ${fields.density[at(x, y, z)]}, expected ${expectedMean}`);
                  checkedCells++;
                }
          }
          assert.ok(checkedCells > 0, `${label}: inspect accepted native cells`);
          const currentRoots = verticalRoots(phi, nx, ny, nz, h);
          let changedRootCountColumns = 0, maximumRootDisplacement_m = 0;
          currentRoots.forEach((roots, column) => {
            const expected = baselineRoots[column]!;
            if (roots.length !== expected.length) changedRootCountColumns++;
            else roots.forEach((root, i) => { maximumRootDisplacement_m = Math.max(maximumRootDisplacement_m, Math.abs(root - expected[i]!)); });
          });
          console.log(JSON.stringify({ fixture, phase, cycle, width, mixed,
            baselineCrossings: baselineRoots.reduce((sum, roots) => sum + roots.length, 0),
            currentCrossings: currentRoots.reduce((sum, roots) => sum + roots.length, 0),
            changedRootCountColumns, maximumRootDisplacement_m,
            maximumRootDisplacement_finestCells: maximumRootDisplacement_m / h }));
          assert.equal(changedRootCountColumns, 0, `${label}: visible component crossing counts changed`);
          assert.ok(maximumRootDisplacement_m < 1e-5, `${label}: vertical interface moved ${maximumRootDisplacement_m} m`);
          for (const i of probes) assert.ok(Number.isFinite(phi[i]) && Math.abs(phi[i]! - initialPhi[i]!) < 1e-5,
            `${label}: published field sample ${i} changed from ${initialPhi[i]} to ${phi[i]}`);
        }
      }
      await solver.assertSimulationHealthy();
      assert.deepEqual(errors, []);
    } finally {
      solver?.destroy(); device?.destroy(); if (gpu) live.delete(gpu);
      await releaseWebGPUExclusiveLock();
    }
  });
