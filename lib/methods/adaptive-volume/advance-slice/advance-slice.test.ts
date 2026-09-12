import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES,
  SPARSE_CM12_RESIDENT_STAGES,
} from "../webgpu-sparse-cm12-resident";
import {
  ADVANCE_STAGE_ORDER, ADVANCE_WORK, ADVANCE_WORK_SCENES,
  advanceCosts, advanceStageCost, advanceWorkModel,
} from "./advance-work";
import {
  advanceSlice, createAdvanceSlice, planMicrosteps,
  SLICE_BX, SLICE_BY, SLICE_NX, SLICE_NY, sliceCell,
} from "./slice-solver";

const inputs = (over: Partial<Parameters<typeof advanceWorkModel>[0]> = {}) =>
  advanceWorkModel({
    scene: ADVANCE_WORK_SCENES.mini32, pressureIterations: 64, cfl: 1,
    limiterPasses: 8, churn: 0.06, markers: 65_536, ...over,
  });

/**
 * The lab's stage strip is only worth reading if it partitions the advance the
 * encoder actually writes. The stage ids and sub-seam ids are typed against
 * the resident ABI; their order, and the fact that every kernel named here is
 * a kernel the encoder dispatches, are pinned against the encoder's source.
 */
test("the work table covers the resident stage ABI, in encode order", () => {
  assert.deepEqual([...ADVANCE_STAGE_ORDER], [...SPARSE_CM12_RESIDENT_STAGES]);
  for (const stage of ADVANCE_STAGE_ORDER) {
    const declared = SPARSE_CM12_RESIDENT_STAGE_SUBSTAGES[stage] as readonly string[];
    const modelled = ADVANCE_WORK[stage].seams
      .flatMap(seam => (seam.id === null ? [] : [seam.id as string]));
    assert.deepEqual(modelled, [...declared],
      `${stage} must model exactly the sub-seams the encoder closes`);
    for (const seam of ADVANCE_WORK[stage].seams) {
      assert.ok(seam.id !== null || typeof seam.label === "string",
        `${stage} names an interval that is not an ABI sub-seam, so it needs a label`);
    }
  }
});

test("every kernel the work table prices is one the encoder dispatches", () => {
  const source = readFileSync(new URL(
    "../webgpu-sparse-cm12-resident.ts", import.meta.url), "utf8");
  const missing: string[] = [];
  for (const stage of ADVANCE_STAGE_ORDER) {
    for (const seam of ADVANCE_WORK[stage].seams) {
      for (const kernel of seam.kernels) {
        if (kernel.isCopy) continue;
        /* a host helper is called by name, a shader is dispatched by string */
        const found = kernel.host
          ? source.includes(kernel.name)
          : source.includes(`"${kernel.name}"`);
        if (!found) missing.push(`${stage}/${kernel.name}`);
      }
    }
  }
  assert.deepEqual(missing, []);
});

test("a capability a scene lacks encodes no dispatches for it", () => {
  const withInflow = advanceStageCost(
    ADVANCE_WORK["body-forces"], inputs({ scene: ADVANCE_WORK_SCENES.dam }));
  const without = advanceStageCost(ADVANCE_WORK["body-forces"], inputs());
  assert.ok(withInflow.dispatches > without.dispatches,
    "the inflow source chain is gated, so it must move the encoded count");
  assert.equal(without.dispatches, 3, "gravity plus the two unconditional source seals");
});

test("the pressure solve keeps its tail encoded past the residual guard", () => {
  const budget = 64;
  const cost = advanceStageCost(
    ADVANCE_WORK["pressure-solve"], inputs({ pressureIterations: budget }));
  /* three per iteration, an eight-dispatch guard on every eighth but the
     last, and a three-dispatch close */
  assert.equal(cost.dispatches, budget * 3 + (budget / 8 - 1) * 8 + 3);
});

test("transport is the frame's largest stage at mini32 scale", () => {
  const costs = advanceCosts(inputs({ cfl: 1.5 }));
  const largest = costs.indexOf(
    costs.reduce((a, b) => (b.workgroups > a.workgroups ? b : a)));
  assert.equal(ADVANCE_STAGE_ORDER[largest], "conservative-transport");
});

/**
 * The slice exists to show that the transport is conservative by construction.
 * If it ever is not, the lab is drawing a lie, so this is the one number the
 * whole page rests on.
 */
test("the slice conserves volume exactly across an advance", () => {
  const slice = createAdvanceSlice();
  const seeded = slice.seededVolume;
  assert.ok(seeded > 100, "the scene must seed a dam worth reading");
  for (let frame = 0; frame < 60; frame++) {
    advanceSlice(slice, 24);
    assert.ok(Math.abs(slice.drift) < 1e-6,
      `volume drifted by ${(slice.drift * 100).toFixed(6)}% at frame ${frame}`);
  }
});

test("no cell ever holds more liquid than it has capacity for", () => {
  const slice = createAdvanceSlice();
  for (let frame = 0; frame < 40; frame++) advanceSlice(slice, 24);
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const i = sliceCell(x, y);
    assert.ok(slice.V[i] >= -1e-9 && slice.V[i] <= slice.K[i] + 1e-9,
      `cell ${x},${y} holds ${slice.V[i]} of ${slice.K[i]}`);
  }
});

test("the microstep plan is the shader's own rule", () => {
  assert.equal(planMicrosteps(0), 1);
  assert.equal(planMicrosteps(0.4), 1);
  assert.equal(planMicrosteps(0.6), 2);
  assert.equal(planMicrosteps(1.7), 4);
});

test("bricks stay within one rung of every neighbour", () => {
  const slice = createAdvanceSlice();
  for (let frame = 0; frame < 40; frame++) {
    advanceSlice(slice, 24);
    for (let by = 0; by < SLICE_BY; by++) for (let bx = 0; bx < SLICE_BX; bx++) {
      const here = slice.rung[by * SLICE_BX + bx];
      if (bx + 1 < SLICE_BX) {
        assert.ok(Math.abs(here - slice.rung[by * SLICE_BX + bx + 1]) <= 1);
      }
      if (by + 1 < SLICE_BY) {
        assert.ok(Math.abs(here - slice.rung[(by + 1) * SLICE_BX + bx]) <= 1);
      }
    }
  }
});
