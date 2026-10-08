import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { scaleScene } from "../lib/core/scene-scale";
import {
  LIQUID_EXTRUSION_MAX_EDGES, liquidExtrusionBounds, liquidExtrusionDistance, liquidExtrusionEdges,
  liquidExtrusionRefusal, liquidExtrusionVolume_m3, type LiquidExtrusion,
} from "../lib/core/liquid-extrusion";
import {
  NBFLIP_LETTERS_CELL_SIZE_M, NBFLIP_LETTERS_GRID, NBFLIP_LETTERS_SCENE_ID, NBFLIP_LETTERS_TIME_STEP_S,
} from "../lib/core/nbflip-paper-scenes";

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

test("NB-FLIP Figure 8 is the published grid with nine letters on a ten-frame schedule", () => {
  const definition = getSceneDefinition(NBFLIP_LETTERS_SCENE_ID), scene = sceneDocument(definition);
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).fluid, scene.fluid);
  assert.deepEqual(definition.methodProfile, { methodId: "uniform-narrow-band-flip", quality: "balanced", overrides: { timeStep: "scene" } });
  const h = scene.voxelDomain.finestCellSize_m, c = scene.container;
  assert.equal(h, NBFLIP_LETTERS_CELL_SIZE_M);
  assert.deepEqual([c.width_m, c.height_m, c.depth_m].map((length) => length / h), [...NBFLIP_LETTERS_GRID]);
  assert.deepEqual(NBFLIP_LETTERS_GRID, [256, 192, 128]);
  assert.equal(c.fillFraction * 192, 37);
  assert.equal(scene.numerics.fixedDt_s, 1 / 24);
  // The fall the footage shows: 0.8 cells per frame squared.
  assert.ok(Math.abs(-scene.fluid.gravity_m_s2.y * NBFLIP_LETTERS_TIME_STEP_S ** 2 / h - 0.8) < 1e-12);
  const drops = scene.fluid.scheduledDrops!;
  assert.deepEqual(drops.map((drop) => Math.round(drop.time_s * 24)), [1, 11, 21, 31, 41, 51, 61, 71, 81]);
  for (const [index, drop] of drops.entries()) {
    assert.equal(drop.volume.shape, "extrusion");
    const letter = drop.volume as LiquidExtrusion, bounds = liquidExtrusionBounds(letter);
    assert.equal(liquidExtrusionRefusal(letter), undefined);
    const height = (bounds.max_m.y - bounds.min_m.y) / h, width = (bounds.max_m.x - bounds.min_m.x) / h;
    assert.ok(height > 43 && height < 45.5 && width < 43, `letter ${index} is ${width} x ${height} cells`);
    assert.ok(bounds.min_m.y / h > 37 + 40, "spawns clear of the pool");
    // The centre of the stroke is liquid: every letter is solid one cell inside its outline somewhere.
    const edges = liquidExtrusionEdges(letter);
    const inside = liquidExtrusionDistance(letter, { x: 0.5 * (edges[0]! + edges[2]!), y: 0.5 * (edges[1]! + edges[3]!), z: letter.centerZ_m }, edges);
    assert.ok(Math.abs(inside + letter.offset_m) < 1e-5, "an outline edge lies one offset inside the liquid");
  }
  // Letters carry between 9 and 13 thousand cells of liquid each.
  const cells = drops.map((drop) => liquidExtrusionVolume_m3(drop.volume as LiquidExtrusion, h / 2) / h ** 3);
  assert.ok(cells.every((volume) => volume > 5000 && volume < 16000), cells.map(Math.round).join(" "));
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
