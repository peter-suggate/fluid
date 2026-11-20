import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { NBFLIP_POUR_SCENE_ID } from "../lib/core/nbflip-paper-scenes";
import { integratedInflowVolume } from "../lib/core/inflow-boundary";
import { brickStaticMask } from "../lib/methods/narrow-band/brick-static.wgsl";
import {
  BRICKS, CLASS_DEAD, DT, NX, NY, NZ, assertPourScene, bakePour, brickId, cellIndex, glassSolid,
} from "../lib/methods/pour/pour-layout";
import { pourKernels } from "../lib/methods/pour/pour-kernels";

const scene = () => sceneDocument(getSceneDefinition(NBFLIP_POUR_SCENE_ID));

test("pour bake: the glass rule is the brick engine's voxel mask, cell for cell", () => {
  const s = scene(), mask = brickStaticMask(s)!;
  assert.ok(mask);
  assert.doesNotThrow(() => assertPourScene(s, (x, y, z) => mask.solid(x, y, z)));
  // A one-cell difference fails explicitly.
  assert.throws(() => assertPourScene(s, (x, y, z) => (x === 5 && y === 7 && z === 9) !== mask.solid(x, y, z)), /differs from the baked rule at cell \(5, 7, 9\)/);
  let solid = 0;
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) if (glassSolid(x, y, z)) solid++;
  assert.ok(solid > 100_000, `the glass has ${solid} solid cells`);
  // The rim row is the last solid one; the floor is the box's.
  assert.ok(glassSolid(64 + 52, 191, 64) && !glassSolid(64 + 52, 192, 64));
  assert.ok(!glassSolid(64, 0, 64));
});

test("pour bake: other scenes and changed pours are refused", () => {
  const s = scene(), mask = brickStaticMask(s)!, solid = (x: number, y: number, z: number) => mask.solid(x, y, z);
  assert.throws(() => assertPourScene({ ...s, sceneId: "reservoir-seep" }, solid), /supports only/);
  assert.throws(() => assertPourScene({ ...s, container: { ...s.container, top: "open" } }, solid), /closed/);
  assert.throws(() => assertPourScene({ ...s, fluid: { ...s.fluid, inflow: { ...s.fluid.inflow!, ramp_s: 0.5 } } }, solid), /no ramp/);
  assert.throws(() => assertPourScene({ ...s, fluid: { ...s.fluid, inflow: undefined } }, solid), /inflow is missing/);
});

test("pour bake: brick-major indexing is a bijection", () => {
  const seen = new Uint8Array(NX * NY * NZ);
  for (let z = 0; z < NZ; z++) for (let y = 0; y < NY; y++) for (let x = 0; x < NX; x++) {
    const i = cellIndex(x, y, z);
    assert.equal(i >> 9, brickId(x >> 3, y >> 3, z >> 3));
    assert.equal(seen[i], 0); seen[i] = 1;
  }
  assert.equal(BRICKS, 8192);
});

test("pour bake: the nozzle's statics and schedule are the brick engine's", () => {
  const s = scene(), bake = bakePour(s), n = bake.inflow, inflow = s.fluid.inflow!;
  // No pour brick is wholly solid.
  assert.equal(bake.brickClass.filter(c => c & CLASS_DEAD).length, 0);
  assert.equal(n.axis, 0);
  assert.equal(n.face, 36);
  assert.ok(Math.abs(n.outlet[0] - 37.46) < 0.01 && Math.abs(n.outlet[1] - 210.62) < 0.01 && Math.abs(n.outlet[2] - 64) < 1e-9, `outlet ${n.outlet}`);
  assert.ok(Math.abs(n.radius - 19.2) < 0.01);
  // The schedule: on through the inflow's end, off after it, continuing after the first step.
  const flags = (k: number) => n.scheduleWords[3 * k + 2]!;
  assert.equal(flags(0), 1);
  assert.equal(flags(1), 3);
  const last = Math.ceil(inflow.end_s / DT) - 1;
  assert.equal(flags(last) & 1, 1);
  assert.equal(flags(last + 1), 0);
  // averageInflowStrength's trapezoid (the brick engine's) gives the last step half strength (zero at end_s).
  assert.equal(n.schedule[3 * last]!, 0.5);
  // The swept length a step is speed x strength x dt (cells).
  let sweep = 0, strength = 0;
  for (let k = 0; k < n.steps; k++) { sweep += n.schedule[3 * k + 1]!; strength += n.schedule[3 * k]!; }
  assert.equal(strength, last + 0.5);
  assert.ok(Math.abs(sweep - n.speed * DT * strength) < 1e-4 * sweep);
  assert.ok(integratedInflowVolume(inflow, 0, inflow.end_s) > 0);
  // The plug box holds the outlet.
  for (let a = 0; a < 3; a++) assert.ok(n.box[0][a]! <= n.outlet[a]! && n.outlet[a]! < n.box[1][a]!);
});

test("pour kernels: every kernel is generated for the baked nozzle, labelled pour.*", () => {
  const k = pourKernels(bakePour(scene()).inflow);
  for (const [name, src] of Object.entries(k)) {
    assert.match(src.label, /^pour\./, name);
    assert.ok(!/\bPX0\b|\bPBX0\b|CLASS_CHAMBER|THROAT_/.test(src.code), `${name} keeps a seep-only symbol`);
  }
});
