import assert from "node:assert/strict";
import test from "node:test";

import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning } from "../lib/svo/svo-render-tuning";
import { svoPrimaryReuseEligible } from "../lib/core/webgpu-renderer";

/**
 * The gate this file holds used to read "a fluid solver is running", and that
 * cost every fluid scene the whole primary band — 46% of the SVO frame — for
 * an input the water does not actually reach. The water is composited over
 * this G-buffer by the raster pipeline and reaches the dry renderer only as
 * the cone-shadow coverage volume, so a running solver moves nothing in it.
 *
 * What a running solver does move is rigid pose, and only when it owns the
 * poses: the roster the coherence key is built from is a readback of the
 * solver's buffer rather than its source, so it can be a frame behind. Both
 * halves are pinned here because each one alone is a plausible-looking gate
 * and the wrong one in either direction is silent — too loose freezes bodies
 * in the G-buffer, too tight just costs.
 */
const frame = {
  reuseEnabled: true,
  residentRigidPoses: false,
  bodyCount: 0,
  simulationRunning: true,
};

test("running water does not block primary reuse", () => {
  assert.equal(svoPrimaryReuseEligible(frame), true);
  // A solver that owns rigid poses but has no bodies moves nothing here.
  assert.equal(svoPrimaryReuseEligible({ ...frame, residentRigidPoses: true }), true);
});

test("solver-owned rigid poses block primary reuse while the simulation runs", () => {
  assert.equal(
    svoPrimaryReuseEligible({ ...frame, residentRigidPoses: true, bodyCount: 3 }),
    false,
  );
  // Paused, the resident buffer is not moving, so the roster cannot be stale.
  assert.equal(
    svoPrimaryReuseEligible({
      ...frame, residentRigidPoses: true, bodyCount: 3, simulationRunning: false,
    }),
    true,
  );
});

test("bodies the renderer itself uploads are covered by the key, not by the gate", () => {
  // Without a resident buffer the CPU roster *is* the source of the body
  // buffer, and the roster is in the coherence key, so moving bodies
  // invalidate through the key and need no gate of their own.
  assert.equal(svoPrimaryReuseEligible({ ...frame, bodyCount: 3 }), true);
});

test("the tuning switch still refuses reuse outright", () => {
  assert.equal(svoPrimaryReuseEligible({ ...frame, reuseEnabled: false }), false);
});

test("stationary primary reuse is on by default and survives a tuning written without it", () => {
  assert.equal(DEFAULT_SVO_RENDER_TUNING.stationaryPrimaryReuseEnabled, true);
  const legacy = { ...DEFAULT_SVO_RENDER_TUNING } as Record<string, unknown>;
  delete legacy["stationaryPrimaryReuseEnabled"];
  assert.equal(
    normalizeSvoRenderTuning(legacy as never).stationaryPrimaryReuseEnabled,
    true,
    "a stored tuning predating the field must fall back to the default, not to false",
  );
  assert.equal(
    normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, stationaryPrimaryReuseEnabled: false })
      .stationaryPrimaryReuseEnabled,
    false,
    "an explicit off must survive normalization",
  );
});
