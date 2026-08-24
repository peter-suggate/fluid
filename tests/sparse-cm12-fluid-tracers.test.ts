import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { defaultScene } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import {
  SPARSE_CM12_RESIDENT_STAGES,
  SPARSE_CM12_TRACER_BUDGET,
  SPARSE_CM12_TRACER_DENSITY,
  sparseCM12TracerLattice,
} from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

/**
 * The seed lattice is the whole economy of the marker view.
 *
 * Both the compute pass and the vertex stage rebuild a marker's seed position
 * from its index through this arithmetic, which is what lets the colour be
 * stored nowhere and therefore never be numerically diffused. If the planner
 * can return a lattice whose product exceeds the budget, the state region sized
 * from it and the range the kernels address stop agreeing.
 */
test("the marker lattice never exceeds its budget, on any domain aspect", () => {
  const domains: readonly (readonly [number, number, number])[] = [
    [64, 64, 64], [256, 128, 256], [320, 96, 80], [8, 8, 8], [512, 512, 512],
    [1, 1, 1], [1000, 3, 7], [97, 61, 43],
  ];
  for (const domain of domains) {
    const lattice = sparseCM12TracerLattice(domain);
    const product = lattice.dimensions[0] * lattice.dimensions[1] * lattice.dimensions[2];
    assert.equal(product, lattice.count, `${domain.join("x")} count must be the product`);
    assert.ok(lattice.count <= SPARSE_CM12_TRACER_BUDGET,
      `${domain.join("x")} seeded ${lattice.count} markers over budget`);
    // Seeding is allowed inside a cell, but only up to the declared density: a
    // marker cloud far finer than the grid is drawing detail the solve does not
    // have, and it costs memory per marker to do it.
    assert.ok(lattice.spacingFine >= Math.cbrt(1 / SPARSE_CM12_TRACER_DENSITY) - 1e-9,
      `${domain.join("x")} seeded finer than the declared marker density`);
    assert.ok(lattice.count
      <= SPARSE_CM12_TRACER_DENSITY * domain[0] * domain[1] * domain[2],
      `${domain.join("x")} seeded ${lattice.count} markers past its cell count`);
    // Every seed must land inside the domain, or a marker begins outside the
    // lattice its colour is normalised against.
    for (let axis = 0; axis < 3; axis += 1) {
      const first = lattice.originFine[axis]! + 0.5 * lattice.spacingFine;
      const last = lattice.originFine[axis]!
        + (lattice.dimensions[axis]! - 0.5) * lattice.spacingFine;
      assert.ok(first >= 0 && last <= domain[axis]!,
        `${domain.join("x")} axis ${axis} seeds ${first}..${last} outside 0..${domain[axis]}`);
    }
  }
});

/** A budget nobody can satisfy must seed nothing rather than one marker. */
test("an unusable marker budget seeds nothing", () => {
  for (const budget of [0, -1, Number.NaN]) {
    assert.equal(sparseCM12TracerLattice([64, 64, 64], budget).count, 0,
      `budget ${budget} must seed no markers`);
  }
});

/**
 * The markers are advected inside the advance, so they are on the stage
 * partition and the SIM panel can price them. A view whose cost is invisible is
 * a view that quietly becomes expensive.
 */
test("marker advection is its own timed stage, inside the transport window", () => {
  const stages = [...SPARSE_CM12_RESIDENT_STAGES];
  const tracers = stages.indexOf("tracer-advection");
  assert.ok(tracers >= 0, "marker advection must be a declared stage");
  // Between the transport and the projection: the markers read the extrapolated
  // transport velocity, which nothing writes between the extension sweeps and
  // `collocateAndDiagnose`, and the accepted density, which the conservative
  // transport leaves alone because it writes the destination bank.
  assert.ok(tracers > stages.indexOf("conservative-transport"),
    "markers must follow the transport whose characteristic they ride");
  assert.ok(tracers < stages.indexOf("velocity-projection"),
    "markers must be advected before the projection rewrites the velocity");
});

/**
 * Markers must seed inside the liquid and then move with it.
 *
 * The three failures this is against, in order of how quietly they would ship:
 * seeding against the wrong density bank (markers scattered through air),
 * addressing the wrong state range (markers at the origin, or physics fields
 * corrupted by marker writes), and advecting with a velocity field that is zero
 * at this point in the frame (a perfectly still, perfectly plausible cloud).
 */
dawnTest("Dawn seeds markers in liquid and carries them with the flow",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-fluid-tracers.test.ts");
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
        const source = solver.sparseWorldUI.overlays.tracers!;
        assert.ok(source.count > 0, "the default tank must plan a marker lattice");

        // Off is free, not cheap: an advance with the view off must encode no
        // marker dispatch at all, so the range stays exactly as allocated.
        assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const untouched = await solver.sparseWorldUI.diagnostics.readTracers();
        assert.ok(untouched.every((value) => value === 0),
          "advancing with the marker view off must not write the marker range");

        solver.sparseWorldUI.control.tracers!.setEnabled(true);
        assert.equal(solver.advanceTo(2 * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const seeded = await solver.sparseWorldUI.diagnostics.readTracers();

        const live: number[] = [];
        for (let marker = 0; marker < source.count; marker += 1) {
          if (seeded[4 * marker + 3]! > 0.5) live.push(marker);
        }
        // The default tank seeds its reservoir in the low-x half, so a correct
        // seed lights up a substantial minority of the lattice. All of it would
        // mean the liquid test never ran; none of it means it ran on the wrong
        // density bank.
        assert.ok(live.length > 0.05 * source.count && live.length < 0.9 * source.count,
          `${live.length} of ${source.count} markers seeded live — expected a wet minority`);

        // Every live marker must sit inside the domain it was seeded into.
        for (const marker of live) {
          for (let axis = 0; axis < 3; axis += 1) {
            const value = seeded[4 * marker + axis]!;
            assert.ok(value >= 0 && value <= dimensions[axis]!,
              `marker ${marker} axis ${axis} seeded at ${value}, outside the domain`);
          }
        }

        // Seeding is a pure function of the index, so a freshly seeded marker
        // has to be exactly where the shared lattice arithmetic says it is.
        const sample = live[Math.floor(live.length / 2)]!;
        const extent = source.latticeDimensions;
        const cell = [
          sample % extent[0],
          Math.floor(sample / extent[0]) % extent[1],
          Math.floor(sample / (extent[0] * extent[1])),
        ];
        for (let axis = 0; axis < 3; axis += 1) {
          const expected = source.originFine[axis]!
            + (cell[axis]! + 0.5) * source.spacingFine;
          // One advance has already moved it, so this is a locality check on the
          // addressing, not an equality check on the position.
          assert.ok(Math.abs(seeded[4 * sample + axis]! - expected) < 4 * source.spacingFine,
            `marker ${sample} axis ${axis} is nowhere near its seed ${expected}`);
        }

        // ...and then it has to move. A dam break under gravity cannot leave a
        // seeded cloud where it was.
        for (let step = 0; step < 12; step += 1) {
          assert.equal(solver.advanceTo((3 + step) * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();
        const carried = await solver.sparseWorldUI.diagnostics.readTracers();

        let moved = 0, fell = 0;
        for (const marker of live) {
          if (carried[4 * marker + 3]! <= 0.5) continue;
          const delta = [0, 1, 2].map((axis) =>
            carried[4 * marker + axis]! - seeded[4 * marker + axis]!);
          if (Math.hypot(...delta) > 0.05) moved += 1;
          if (delta[1]! < 0) fell += 1;
        }
        assert.ok(moved > 0.5 * live.length,
          `only ${moved} of ${live.length} markers moved — the velocity read is dead`);
        assert.ok(fell > 0.3 * live.length,
          `only ${fell} of ${live.length} markers fell — gravity is not reaching them`);

        // The whole contract of a presentation-only marker: it may read the
        // physics and must never write it. Markers sit past every physics field
        // in the state buffer, so a kernel addressing outside its range would
        // land on a pressure or velocity row — which shows up here as a density
        // field that has stopped being finite and positive.
        const { density } = await solver.readDiagnosticFields();
        assert.ok(density.every((value) => Number.isFinite(value) && value >= 0),
          "an advance with markers on must leave the density field intact");
        assert.ok(density.some((value) => value > 0.5),
          "an advance with markers on must leave liquid in the tank");
      } finally {
        solver.destroy();
      }
      assert.deepEqual(uncaptured, [], "marker work must raise no device errors");
    } finally {
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
