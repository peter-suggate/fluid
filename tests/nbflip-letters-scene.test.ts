import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { validateScene } from "../lib/core/model";
import { scaleScene } from "../lib/core/scene-scale";
import {
  LIQUID_EXTRUSION_MAX_EDGES, liquidExtrusionBounds, liquidExtrusionDistance, liquidExtrusionEdges,
  liquidExtrusionRefusal, liquidExtrusionVolume_m3, type LiquidExtrusion,
} from "../lib/core/liquid-extrusion";
import { NBFLIP_LETTERS_SCENE_ID } from "../lib/core/nbflip-paper-scenes";

/** A 4 x 4 square frame around a 2 x 2 hole, extruded 2 deep about z = 1. */
const frame = (offset_m = 0, edgeRadius_m = 0): LiquidExtrusion => ({
  shape: "extrusion",
  contours_m: [[-2, -2, 2, -2, 2, 2, -2, 2], [-1, -1, 1, -1, 1, 1, -1, 1]],
  centerZ_m: 1, halfDepth_m: 1, offset_m, edgeRadius_m,
});

test("an extrusion's distance is exact for an outline with a hole", () => {
  const sharp = frame();
  const d = (x: number, y: number, z: number, shape = sharp) => liquidExtrusionDistance(shape, { x, y, z });
  assert.ok(Math.abs(d(1.5, 0, 1) + 0.5) < 1e-6, "mid-stroke, half a unit from either edge");
  assert.ok(Math.abs(d(0, 0, 1) - 1) < 1e-6, "the hole is outside, one unit from the stroke");
  assert.ok(Math.abs(d(3, 0, 1) - 1) < 1e-6, "beside the outline");
  assert.ok(Math.abs(d(1.5, 0, 2.75) - 0.75) < 1e-6, "in front of the flat face");
  assert.ok(Math.abs(d(3, 0, 3) - Math.SQRT2) < 1e-6, "off an edge of the face");
  assert.ok(Math.abs(d(3, 3, 1) - Math.SQRT2) < 1e-6, "off a corner of the outline");
  // Growing the outline moves every planar distance by the offset and closes the hole by it.
  const grown = frame(0.25);
  assert.ok(Math.abs(d(3, 0, 1, grown) - 0.75) < 1e-6);
  assert.ok(Math.abs(d(0, 0, 1, grown) - 0.75) < 1e-6);
  // A rounded edge takes the corner of the face off and nothing else.
  const round = frame(0, 0.5);
  assert.ok(Math.abs(d(3, 0, 1, round) - 1) < 1e-6 && Math.abs(d(1.5, 0, 2.75, round) - 0.75) < 1e-6);
  assert.ok(d(2, 0, 2, round) > 0.2 && d(2, 0, 2, sharp) === 0, "the sharp edge is on the surface; the rounded one is not");
});

test("an extrusion's volume and bounds follow its outline", () => {
  assert.ok(Math.abs(liquidExtrusionVolume_m3(frame(), 0.01) - 24) < 1e-6, "(16 - 4) x 2");
  // Rounding removes (1 - pi/4) r^2 of cross-section along both perimeters (16 + 8) of both faces;
  // the eight corners where two of those cuts meet differ from that by a term the bound covers.
  const rounded = liquidExtrusionVolume_m3(frame(0, 0.5), 0.005), expected = 24 - 2 * (16 + 8) * (1 - Math.PI / 4) * 0.25;
  assert.ok(Math.abs(rounded - expected) < 0.05, `${rounded} vs ${expected}`);
  assert.deepEqual(liquidExtrusionBounds(frame(0.25)), { min_m: { x: -2.25, y: -2.25, z: 0 }, max_m: { x: 2.25, y: 2.25, z: 2 } });
  assert.equal(liquidExtrusionEdges(frame()).length, 4 * 8);
});

test("an extrusion the solver cannot take is refused with its reason", () => {
  assert.equal(liquidExtrusionRefusal(frame(0.25, 0.5)), undefined);
  assert.match(liquidExtrusionRefusal({ ...frame(), contours_m: [] })!, /at least one contour/);
  assert.match(liquidExtrusionRefusal({ ...frame(), contours_m: [[0, 0, 1, 0]] })!, /three x\/y points/);
  assert.match(liquidExtrusionRefusal({ ...frame(), contours_m: [[0, 0, 1, 0, 1, 0, 0, 1]] })!, /must differ/);
  assert.match(liquidExtrusionRefusal({ ...frame(), edgeRadius_m: 2 })!, /edge radius/);
  const circle = Array.from({ length: LIQUID_EXTRUSION_MAX_EDGES + 1 }, (_, index) => {
    const angle = 2 * Math.PI * index / (LIQUID_EXTRUSION_MAX_EDGES + 1);
    return [Math.cos(angle), Math.sin(angle)];
  }).flat();
  assert.match(liquidExtrusionRefusal({ ...frame(), contours_m: [circle] })!, /65 outline edges/);
});

test("world scale carries scheduled drops with the container", () => {
  const scene = sceneDocument(getSceneDefinition(NBFLIP_LETTERS_SCENE_ID)), doubled = scaleScene(scene, "world", 2)!;
  assert.deepEqual(validateScene(doubled), []);
  const before = scene.fluid.scheduledDrops![1]!, after = doubled.fluid.scheduledDrops![1]!;
  assert.equal(after.time_s, before.time_s);
  const a = liquidExtrusionBounds(before.volume as LiquidExtrusion), b = liquidExtrusionBounds(after.volume as LiquidExtrusion);
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(b.min_m[axis] - 2 * a.min_m[axis]) < 1e-9 && Math.abs(b.max_m[axis] - 2 * a.max_m[axis]) < 1e-9);
  }
});
