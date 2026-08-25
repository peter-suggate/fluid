import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createSparseCM12LongDamBreakScene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

dawnTest("a 4-cell floor retires diffuse bricks away from the represented surface",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-min4-surface-dawn.test.ts");
    let device: GPUDevice | undefined;
    let solver: WebGPUAdaptiveMassSolver | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter);
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      const scene = createSparseCM12LongDamBreakScene();
      scene.fluid.refinementRegions = [{
        id: "whole-domain-floor",
        rule: "minimum-cell-size",
        minimumCellSize_cells: 4,
        min_m: { x: -0.5 * scene.container.width_m, y: 0,
          z: -0.5 * scene.container.depth_m },
        max_m: { x: 0.5 * scene.container.width_m, y: scene.container.height_m,
          z: 0.5 * scene.container.depth_m },
      }];
      solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined, {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          timeStep: "paper",
          pressureIterations: 64,
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: false,
            topologyCadenceSteps: 1,
            prepareBricksPerFrame: 256,
          },
        }, () => {});

      for (let step = 1; step <= 96; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        if (step % 16 === 0) await device.queue.onSubmittedWorkDone();
      }
      await device.queue.onSubmittedWorkDone();

      const snapshot = await solver.readGPUActivityPolicy();
      const fields = await solver.readDiagnosticFields();
      const active = snapshot.bricks.filter((brick) => brick.active);
      const occupied = active.filter((brick) => (brick.reasons & 64) !== 0);
      const occupiedCoordinates = new Set(occupied.map((brick) => brick.coordinate.join("/")));
      const requestedAsDestination = (brick: (typeof active)[number]): boolean => {
        for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const neighbour = active.find((candidate) => candidate.coordinate[0]
              === brick.coordinate[0] + dx && candidate.coordinate[1]
              === brick.coordinate[1] + dy && candidate.coordinate[2]
              === brick.coordinate[2] + dz);
            if (!neighbour) continue;
            const bit = 1 - dx + 3 * (1 - dy) + 9 * (1 - dz);
            if ((neighbour.supportMask & 2 ** bit) !== 0) return true;
          }
        }
        return false;
      };
      const unsupportedDry = active.filter((brick) => !occupiedCoordinates.has(
        brick.coordinate.join("/")) && !requestedAsDestination(brick));
      const coarseMist = occupied.filter((brick) => brick.acceptedResolution === 1
        && brick.meanDensity <= SPARSE_CM12_ACTIVITY_POLICY.surfaceDensityMinimum
        && (brick.reasons & 256) === 0);
      const surface = occupied.filter((brick) => (brick.reasons & 1) !== 0);
      const coarseSurface = surface.filter((brick) => brick.acceptedResolution !== 2);
      const byCoordinate = new Map(active.map((brick) =>
        [brick.coordinate.join("/"), brick] as const));
      const crossingBrickKeys = new Set<string>();
      const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
      const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const wet = fields.density[index(x, y, z)]! >= 0.5;
          for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
            const qx = x + dx!, qy = y + dy!, qz = z + dz!;
            if (qx >= nx || qy >= ny || qz >= nz
              || (fields.density[index(qx, qy, qz)]! >= 0.5) === wet) continue;
            crossingBrickKeys.add([Math.floor(x / 8), Math.floor(y / 8),
              Math.floor(z / 8)].join("/"));
            crossingBrickKeys.add([Math.floor(qx / 8), Math.floor(qy / 8),
              Math.floor(qz / 8)].join("/"));
          }
        }
      }
      const coarseFieldCrossings = [...crossingBrickKeys].map((key) => byCoordinate.get(key))
        .filter((brick): brick is (typeof active)[number] =>
          brick !== undefined && brick.acceptedResolution !== 2);
      const census = Object.fromEntries(Array.from({ length: 12 }, (_, y) => [y, {
        active: active.filter((brick) => brick.coordinate[1] === y).length,
        occupied: active.filter((brick) => brick.coordinate[1] === y
          && (brick.reasons & 64) !== 0).length,
        surface: active.filter((brick) => brick.coordinate[1] === y
          && (brick.reasons & 1) !== 0).length,
        coarseMist: coarseMist.filter((brick) => brick.coordinate[1] === y).length,
      }]));
      console.log(JSON.stringify({ census, unsupportedDry: unsupportedDry.length,
        coarseMist: coarseMist.map((brick) => ({ coordinate: brick.coordinate,
          meanDensity: brick.meanDensity })), coarseSurface: coarseSurface.map((brick) => ({
          coordinate: brick.coordinate,
          acceptedResolution: brick.acceptedResolution,
          plannedResolution: brick.plannedResolution,
          planReasons: brick.planReasons,
          candidateResolution: brick.candidateResolution,
          candidateStatus: brick.candidateStatus,
        })), coarseFieldCrossings: coarseFieldCrossings.map((brick) => ({
          coordinate: brick.coordinate,
          meanDensity: brick.meanDensity,
          reasons: brick.reasons,
          acceptedResolution: brick.acceptedResolution,
          plannedResolution: brick.plannedResolution,
          planReasons: brick.planReasons,
        })) }, null, 2));
      assert.equal(unsupportedDry.length, 0,
        `dry bricks remained without current surface support: ${
          unsupportedDry.map((brick) => brick.coordinate.join(",")).join("; ")}`);
      assert.equal(coarseMist.length, 0,
        `coarse haze was classified as liquid: ${coarseMist.map((brick) =>
          `${brick.coordinate.join(",")}=${brick.meanDensity}`).join("; ")}`);
      assert.equal(active.filter((brick) => brick.coordinate[1] >= 5).length, 0,
        "the settled surface must not retain a second unsupported brick row above it");
      assert.ok(surface.length > 0, "the long dam must retain a measured free surface");
      assert.equal(coarseSurface.length, 0,
        `surface bricks escaped the finest 4-cell rung allowed by the box: ${
          coarseSurface.map((brick) => `${brick.coordinate.join(",")} accepted=${
            brick.acceptedResolution} planned=${brick.plannedResolution} reasons=${
            brick.planReasons} candidate=${brick.candidateResolution}/${
            brick.candidateStatus}`).join("; ")}`);
      assert.equal(coarseFieldCrossings.length, 0,
        `rendered isovalue crossings escaped the finest 4-cell rung: ${
          coarseFieldCrossings.map((brick) => `${brick.coordinate.join(",")} rho=${
            brick.meanDensity} reasons=${brick.reasons} accepted=${
            brick.acceptedResolution} planned=${brick.plannedResolution}/${
            brick.planReasons}`).join("; ")}`);
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
