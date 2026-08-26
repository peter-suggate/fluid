import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod } from "../lib/methods/adaptive-mass/method";
import type { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

const key = (coordinate: readonly number[]) => coordinate.join("/");

dawnTest("mini32 retires vacant bricks and refines every represented surface crossing",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-mini32-surface-retirement-dawn.test.ts");
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
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const validationErrors: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        validationErrors.push(event.error.message);
      });

      // Reproduce the actual library/UI document, including the final-lattice
      // SolidWorld tank shell. Calling the raw factory here omits the document
      // finalizer and creates an unbounded world with no floor.
      const scene = sceneDocument(getSceneDefinition("minimal-power-dam-break-32"));
      const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
        brickFineResolution: "8",
        resolutionMode: "adaptive",
        selectorMode: "surface",
        surfaceFineRings: 1,
        timeStep: "paper",
      });
      solver = await adaptiveMassMethod.createSolverAsync!(
        device, scene, "balanced", values, undefined, () => {},
      ) as WebGPUAdaptiveMassSolver;
      await solver.waitForSimulationReady();

      const initial = await solver.readGPUActivityPolicy();
      const initialByCoordinate = new Map(initial.bricks.map((brick) =>
        [key(brick.coordinate), brick] as const));
      assert.ok(initial.bricks.some((brick) => brick.active
        && brick.acceptedResolution < 8),
      "the fixture must begin with coarsened resident bricks");
      const initiallyCoarseKeys = new Set(initial.bricks.filter((brick) => brick.active
        && brick.acceptedResolution < 8).map((brick) => brick.key));
      const initiallyCoarseHistory: Array<Readonly<{
        step: number; coordinate: readonly number[]; active: boolean;
        accepted: number; reasons: number; planned: number; planReasons: number;
      }>> = [];

      const steps = 14;
      for (let step = 1; step <= steps; step += 1) {
        assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const snapshot = await solver.readGPUActivityPolicy();
        if (snapshot.commitFailed) {
          const rejectionReceipts: unknown[] = await Promise.all([
            solver.readCandidateEffectsTransactionQA(),
            solver.readFrameControlQA(),
            solver.readPresentationPageAllocatorReceiptQA(),
            solver.readWorldGrowthReceiptQA(),
          ]);
          const involved = snapshot.bricks.filter((brick) =>
            brick.topologyPreparationScheduled || brick.candidateStatus !== 0
            || brick.transferStatus !== 0 || brick.faceTransferStatus !== 0
            || brick.plannedResolution !== brick.acceptedResolution);
          assert.fail(`topology transaction rejected at step ${step} (${(
            step * CM12_PAPER_DT_S).toFixed(6)} s): ${JSON.stringify({
            activity: {
              acceptedTopologyGeneration: snapshot.acceptedTopologyGeneration,
              faultFlags: snapshot.faultFlags,
              preparedBrickCount: snapshot.preparedBrickCount,
              committedBrickCount: snapshot.committedBrickCount,
            }, effects: rejectionReceipts[0], frameControl: rejectionReceipts[1],
            pages: rejectionReceipts[2], world: rejectionReceipts[3],
            involved: involved.map((brick) => ({
              key: brick.key, coordinate: brick.coordinate, active: brick.active,
              accepted: brick.acceptedResolution, candidate: brick.candidateResolution,
              candidateStatus: brick.candidateStatus, planned: brick.plannedResolution,
              planReasons: brick.planReasons, reasons: brick.reasons,
              transferStatus: brick.transferStatus,
              faceTransferStatus: brick.faceTransferStatus,
              topologyPreparationScheduled: brick.topologyPreparationScheduled,
              topologyPreparationEpoch: brick.topologyPreparationEpoch,
              topologyPage: brick.topologyPage,
            })),
          })}`);
        }
        for (const brick of snapshot.bricks) if (initiallyCoarseKeys.has(brick.key)) {
          initiallyCoarseHistory.push({ step, coordinate: brick.coordinate,
            active: brick.active, accepted: brick.acceptedResolution,
            reasons: brick.reasons, planned: brick.plannedResolution,
            planReasons: brick.planReasons });
        }
      }
      await device.queue.onSubmittedWorkDone();

      const [activity, fields, pages, world, effects, frameControl] = await Promise.all([
        solver.readGPUActivityPolicy(),
        solver.readDiagnosticFields(),
        solver.readPresentationPageAllocatorReceiptQA(),
        solver.readWorldGrowthReceiptQA(),
        solver.readCandidateEffectsTransactionQA(),
        solver.readFrameControlQA(),
      ]);
      const active = activity.bricks.filter((brick) => brick.active);
      const activeByCoordinate = new Map(active.map((brick) =>
        [key(brick.coordinate), brick] as const));
      const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
      const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
      const brickMaximumDensity = new Map<string, number>();
      for (const brick of activity.bricks) {
        let maximum = 0;
        for (let z = 8 * brick.coordinate[2]; z < 8 * (brick.coordinate[2] + 1); z += 1) {
          for (let y = 8 * brick.coordinate[1]; y < 8 * (brick.coordinate[1] + 1); y += 1) {
            for (let x = 8 * brick.coordinate[0]; x < 8 * (brick.coordinate[0] + 1); x += 1) {
              if (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz) {
                maximum = Math.max(maximum, fields.density[index(x, y, z)]!);
              }
            }
          }
        }
        brickMaximumDensity.set(key(brick.coordinate), maximum);
      }

      const crossingKeys = new Set<string>();
      for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
        for (let x = 0; x < nx; x += 1) {
          const wet = fields.density[index(x, y, z)]! >= 0.5;
          for (const [dx, dy, dz] of [[1, 0, 0], [0, 1, 0], [0, 0, 1]]) {
            const qx = x + dx!, qy = y + dy!, qz = z + dz!;
            if (qx >= nx || qy >= ny || qz >= nz
              || (fields.density[index(qx, qy, qz)]! >= 0.5) === wet) continue;
            crossingKeys.add(key([Math.floor(x / 8), Math.floor(y / 8),
              Math.floor(z / 8)]));
            crossingKeys.add(key([Math.floor(qx / 8), Math.floor(qy / 8),
              Math.floor(qz / 8)]));
          }
        }
      }

      const crossingBricks = [...crossingKeys]
        .map((coordinate) => activeByCoordinate.get(coordinate))
        .filter((brick): brick is (typeof active)[number] => brick !== undefined);
      const initiallyCoarseCrossings = crossingBricks.filter((brick) =>
        (initialByCoordinate.get(key(brick.coordinate))?.acceptedResolution ?? 8) < 8);
      const coarseCrossings = crossingBricks.filter((brick) =>
        brick.acceptedResolution !== 8);
      // Mirror retireUnsupportedEmptyBricks: a dry leaf remains resident only
      // while an active neighbour's immutable sweep mask requests it.
      const requestedAsDestination = (target: (typeof active)[number]): boolean => {
        for (let dz = -1; dz <= 1; dz += 1) for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0 && dz === 0) continue;
            const neighbour = activeByCoordinate.get(key([
              target.coordinate[0] + dx, target.coordinate[1] + dy,
              target.coordinate[2] + dz,
            ]));
            if (!neighbour) continue;
            const bit = 1 - dx + 3 * (1 - dy) + 9 * (1 - dz);
            if ((neighbour.supportMask & 2 ** bit) !== 0) return true;
          }
        }
        return false;
      };
      const unsupportedVacant = active.filter((brick) =>
        (brickMaximumDensity.get(key(brick.coordinate)) ?? 0)
          <= SPARSE_CM12_ACTIVITY_POLICY.residencyDensity
        && (brick.reasons & 64) === 0 && !requestedAsDestination(brick));
      const initiallyCoarseSurface = initiallyCoarseHistory.filter((brick) =>
        brick.active && (brick.reasons & 1) !== 0);
      const initiallyCoarseSurfaceViolations = initiallyCoarseSurface.filter((brick) =>
        brick.accepted !== 8);
      const topActive = active.filter((brick) => brick.coordinate[1] === 3);

      if (process.env.FLUID_MINI32_SURFACE_TRACE === "1") {
        process.stderr.write(`[mini32-surface-retirement] ${JSON.stringify({
          time_s: steps * CM12_PAPER_DT_S,
          faultFlags: activity.faultFlags,
          prepared: activity.preparedBrickCount,
          committed: activity.committedBrickCount,
          commitFailed: activity.commitFailed,
          topologyGeneration: activity.acceptedTopologyGeneration,
          residentBrickCount: activity.residentBrickCount,
          presentationPages: pages.residentPages,
          world,
          effects,
          frameControl,
          initialActive: initial.bricks.filter((brick) => brick.active).length,
          active: active.length,
          topActive: topActive.map((brick) => ({
            coordinate: brick.coordinate,
            maximumDensity: brickMaximumDensity.get(key(brick.coordinate)),
            reasons: brick.reasons,
            supportMask: brick.supportMask,
            accepted: brick.acceptedResolution,
            planned: brick.plannedResolution,
            planReasons: brick.planReasons,
            candidate: brick.candidateResolution,
            candidateStatus: brick.candidateStatus,
            transferStatus: brick.transferStatus,
            faceTransferStatus: brick.faceTransferStatus,
          })),
          initialRungs: Object.fromEntries([1, 2, 4, 8].map((resolution) =>
            [resolution, initial.bricks.filter((brick) => brick.active
              && brick.acceptedResolution === resolution).map((brick) => brick.coordinate)])),
          crossingBricks: crossingBricks.map((brick) => ({
            coordinate: brick.coordinate,
            initial: initialByCoordinate.get(key(brick.coordinate))?.acceptedResolution,
            accepted: brick.acceptedResolution,
            reasons: brick.reasons,
          })),
          initiallyCoarseCrossings: initiallyCoarseCrossings.map((brick) => ({
            coordinate: brick.coordinate,
            initial: initialByCoordinate.get(key(brick.coordinate))?.acceptedResolution,
            accepted: brick.acceptedResolution,
            planned: brick.plannedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
          })),
          initiallyCoarseHistory,
          coarseCrossings: coarseCrossings.map((brick) => ({
            coordinate: brick.coordinate,
            accepted: brick.acceptedResolution,
            planned: brick.plannedResolution,
            reasons: brick.reasons,
            planReasons: brick.planReasons,
          })),
          unsupportedVacant: unsupportedVacant.map((brick) => brick.coordinate),
        })}\n`);
      }

      assert.ok(initiallyCoarseSurface.length > 0,
        "the moving surface must enter at least one initially coarsened brick");
      assert.equal(initiallyCoarseSurfaceViolations.length, 0,
        `initially coarse surface bricks were not refined: ${
          initiallyCoarseSurfaceViolations.map((brick) =>
            `step ${brick.step} ${brick.coordinate.join(",")}=${brick.accepted}`).join("; ")}`);
      assert.equal(coarseCrossings.length, 0,
        `represented surface crossings remained coarse: ${coarseCrossings.map((brick) =>
          `${brick.coordinate.join(",")}=${brick.acceptedResolution}`).join("; ")}`);
      assert.equal(topActive.length, 0,
        `vacant top bricks remained resident: ${topActive.map((brick) =>
          brick.coordinate.join(",")).join("; ")}`);
      assert.equal(unsupportedVacant.length, 0,
        `unsupported vacant bricks remained resident: ${unsupportedVacant.map((brick) =>
          brick.coordinate.join(",")).join("; ")}`);
      assert.equal(activity.commitFailed, false);
      assert.ok(activity.acceptedTopologyGeneration
        > initial.acceptedTopologyGeneration,
        "retirement/refinement must publish through accepted topology generations");
      assert.deepEqual(validationErrors, []);
    } finally {
      solver?.destroy();
      device?.destroy();
      await releaseWebGPUExclusiveLock();
    }
  });
