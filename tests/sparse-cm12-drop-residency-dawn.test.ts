import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { cloneScene, defaultScene } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { initializeSparseBrickAtlasFromScene } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { residentSupportAtlas, WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_ACTIVITY_POLICY } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

/**
 * A ball dropped where no liquid has ever been must land, exactly as a ball
 * dropped into the water does.
 *
 * The default tank seeds its reservoir in the low-x half, so the whole +x
 * quarter is dormant apron: bricks that are packed and dry but whose
 * `brickActive` bit is clear. Every writer in the resident kernel is gated on
 * that bit, so a drop there used to dispatch over the drop's cells, find their
 * bricks inactive, and return having written nothing — the gesture reported
 * success and the water simply did not exist. Injection now enters residency
 * through `activateSweptReceivers`, the same path a front sweeping into a
 * dormant receiver already takes.
 */
dawnTest("Dawn makes a dormant apron brick resident for a dropped ball",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-drop-residency-dawn.test.ts");
    let device: GPUDevice | undefined;
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
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = defaultScene;
      const dimensions = sceneLatticeDimensions(scene);
      // The host layers answer which brick is dormant without asking the
      // device: seeded liquid plus its face receivers is exactly the initially
      // active set, so anything the apron adds beyond it starts inactive.
      const seeded = residentSupportAtlas(
        initializeSparseBrickAtlasFromScene(scene, { finestDimensions: dimensions }),
        "adaptive",
      );
      const activeAtConstruction = new Set(seeded.bricks.map((brick) => brick.key));
      const container = scene.container;
      const brickCentre = (coordinate: readonly [number, number, number]) => ({
        x: (coordinate[0] + 0.5) * 8 * container.width_m / dimensions[0]
          - 0.5 * container.width_m,
        y: (coordinate[1] + 0.5) * 8 * container.height_m / dimensions[1],
        z: (coordinate[2] + 0.5) * 8 * container.depth_m / dimensions[2]
          - 0.5 * container.depth_m,
      });

      device.pushErrorScope("validation");
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          timeStep: "paper",
        },
        () => {},
      );
      try {
        const before = await solver.readGPUActivityPolicy();
        const dormant = before.bricks.filter((brick) =>
          !activeAtConstruction.has(brick.key));
        assert.ok(dormant.length > 0,
          "the default tank must reserve dormant apron capacity to drop into");
        for (const brick of dormant) {
          assert.equal(brick.active, false,
            `apron brick ${brick.coordinate.join(",")} must start dormant`);
        }

        const target = dormant[dormant.length - 1]!;
        const centre_m = brickCentre(target.coordinate);
        // Three finest cells, so a 4^3 receiver's two-cell-wide cells are
        // covered rather than straddled, and the ball still clears the brick's
        // faces: residency must follow the water, not a generous bound.
        const radius_m = 3 * container.width_m / dimensions[0];
        const brickMass = (density: Float32Array,
          coordinate: readonly [number, number, number]) => {
          let mass = 0;
          for (let z = 8 * coordinate[2]; z < Math.min(8 * coordinate[2] + 8, dimensions[2]); z += 1)
            for (let y = 8 * coordinate[1]; y < Math.min(8 * coordinate[1] + 8, dimensions[1]); y += 1)
              for (let x = 8 * coordinate[0]; x < Math.min(8 * coordinate[0] + 8, dimensions[0]); x += 1) {
                mass += density[x + dimensions[0] * (y + dimensions[1] * z)]!;
              }
          return mass;
        };

        assert.equal(brickMass((await solver.readDiagnosticFields()).density,
          target.coordinate), 0, "the target brick must be dry before the drop");
        solver.injectLiquidBall({ centre_m, radius_m });
        await device.queue.onSubmittedWorkDone();

        const afterDrop = await solver.readGPUActivityPolicy();
        const activated = afterDrop.bricks.find((brick) => brick.key === target.key)!;
        assert.equal(activated.active, true,
          `dropping into brick ${target.coordinate.join(",")} must make it resident`);
        assert.equal(activated.acceptedResolution, 8,
          "a dropped ball must be promoted before it is wetted, not one simulation step later");
        // Read the water itself, not only the activity pass's summary of it:
        // a drop that activates a brick and then writes nothing into it is the
        // same lost gesture wearing a resident bit.
        const droppedMass = brickMass((await solver.readDiagnosticFields()).density,
          target.coordinate);
        assert.ok(droppedMass > 1,
          `the drop must wet its brick; measured ${droppedMass} finest cells`);

        // Residency must not be generous. A ball wholly inside one brick may
        // wake exactly that brick, or the apron stops being sparse the first
        // time anyone drops water into it.
        const activeCount = (snapshot: typeof afterDrop) =>
          snapshot.bricks.filter((brick) => brick.active).length;
        assert.equal(activeCount(afterDrop), activeCount(before) + 1,
          "a drop inside one brick must wake exactly that brick");

        // The water has to be simulated from the brick the drop created, not
        // merely stored in it.
        assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const afterStep = await solver.readGPUActivityPolicy();
        const stepped = afterStep.bricks.find((brick) => brick.key === target.key)!;
        assert.equal(stepped.active, true, "an activated receiver must stay resident");
        assert.ok(stepped.meanDensity > 0,
          `the dropped ball must carry mass; measured ${stepped.meanDensity}`);
        assert.equal(stepped.acceptedResolution, 8,
          "a dropped ball is a free surface, so its brick must remain 8^3");
      } finally {
        solver.destroy();
      }

      // Isolate the already-active path in a full, zero-gravity cube which
      // deterministically walks 8 -> 4 -> 2 -> 1 before the injection.
      const calm = cloneScene(defaultScene);
      calm.rigidBodies = [];
      calm.container = { ...calm.container,
        width_m: 0.8, height_m: 0.8, depth_m: 0.8, fillFraction: 1 };
      calm.voxelDomain.finestCellSize_m = 0.05;
      calm.fluid.initialCondition = "dam-break";
      calm.fluid.initialDamBreakDimensions_m = { x: 0.8, y: 0.8, z: 0.8 };
      calm.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
      const activeSolver = await WebGPUAdaptiveMassSolver.createAsync(
        device, calm, "balanced", undefined,
        {
          resolutionMode: "all-fine",
          brickFineResolution: 8,
          surfaceFineRings: 1,
          receiverSupportRings: 1,
          receiverFloor: 1,
          timeStep: "paper",
          activityPolicy: {
            ...SPARSE_CM12_ACTIVITY_POLICY,
            activitySignals: true,
            topologyCadenceSteps: 1,
            demoteEpochs: 1,
            prepareBricksPerFrame: 256,
          },
          pressureIterations: 8,
        },
        () => {},
      );
      try {
        for (let step = 1; step <= 3; step += 1) {
          assert.equal(activeSolver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();
        const beforeActiveDrop = await activeSolver.readGPUActivityPolicy();
        const intersected = beforeActiveDrop.bricks.filter((brick) => brick.active
          && brick.coordinate[1] === 0);
        assert.equal(intersected.length, 4,
          "the calm cube must expose four active bricks around its horizontal centre");
        assert.ok(intersected.every((brick) => brick.acceptedResolution === 1),
          "the calm injection targets must first reach the 1-cubed rung");
        const beforeByKey = new Map(beforeActiveDrop.bricks
          .map((brick) => [brick.key, brick.acceptedResolution] as const));
        activeSolver.injectLiquidBall({
          centre_m: { x: 0, y: 0.2, z: 0 }, radius_m: 0.1,
        });
        await device.queue.onSubmittedWorkDone();
        const afterActiveDrop = await activeSolver.readGPUActivityPolicy();
        assert.ok(intersected.every((targetBrick) => afterActiveDrop.bricks.find(
          (brick) => brick.key === targetBrick.key)?.acceptedResolution === 8),
        "all four already-active coarse injection targets must promote before wetting");
        assert.ok(afterActiveDrop.bricks.every((brick) => {
          const previous = beforeByKey.get(brick.key);
          return previous === undefined || brick.acceptedResolution >= previous;
        }), "dropping liquid must never coarsen accepted topology");
      } finally {
        activeSolver.destroy();
      }
      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
