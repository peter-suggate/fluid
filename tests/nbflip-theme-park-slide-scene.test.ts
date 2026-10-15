import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { parseScene, serializeScene, validateScene } from "../lib/core/model";
import { NBFLIP_METHOD_PROFILE } from "../lib/core/nbflip-paper-scenes";
import { presentationModeForScene, sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { buildEnvironmentProxyCatalog } from "../lib/core/voxel-environments";
import { VOXEL_MATERIAL_IDS } from "../lib/core/voxel-scene";
import { sweptFlumeDistance, sweptFlumeSurfaceErrors } from "../lib/core/swept-flume-surface";
import {
  NBFLIP_THEME_PARK_HALF_WIDTH, NBFLIP_THEME_PARK_GRID, NBFLIP_THEME_PARK_LENGTH, NBFLIP_THEME_PARK_POOL, NBFLIP_THEME_PARK_SCENE_ID,
  nbflipThemeParkCentreline,
} from "../lib/core/nbflip-theme-park-slide-scene";

const definition = getSceneDefinition(NBFLIP_THEME_PARK_SCENE_ID), scene = sceneDocument(definition);
const [nx, ny, nz] = NBFLIP_THEME_PARK_GRID;
const patches = scene.solidVoxels.filter(({ minimum, maximumExclusive }) =>
  minimum.every((cell, axis) => cell >= 0 && maximumExclusive[axis]! <= NBFLIP_THEME_PARK_GRID[axis]!));
// Actual emitted voxel geometry, including supports, pool walls and every deck.
const solid = new Uint8Array(nx * ny * nz);
for (const { minimum: min, maximumExclusive: max } of patches) {
  for (let z = min[2]; z < max[2]; z++) for (let y = min[1]; y < max[1]; y++) {
    solid.fill(1, min[0] + nx * (y + ny * z), max[0] + nx * (y + ny * z));
  }
}
const occupied = (x: number, y: number, z: number) => solid[Math.floor(x) + nx * (Math.floor(y) + ny * Math.floor(z))];

test("theme park slide is selectable, serializable and starts empty on sparse NB-FLIP pages", () => {
  assert.deepEqual(validateScene(scene), []);
  assert.deepEqual(parseScene(serializeScene(scene)).solidVoxels, scene.solidVoxels);
  assert.deepEqual(definition.methodProfile, NBFLIP_METHOD_PROFILE);
  assert.equal(presentationModeForScene(definition, scene), "full-scene", "the flume must be visible alongside the water");
  assert.equal(scene.container.fillFraction, 0);
  assert.equal(scene.fluid.initialLiquidVolumes, undefined);
  assert.equal(scene.container.top, "closed");
  assert.ok(NBFLIP_THEME_PARK_GRID.every(cells => cells % 32 === 0));
  assert.ok(scene.fluid.pagePool!.blocks! < nx * ny * nz / 32 ** 3 / 2);
  assert.ok(scene.duration_s - scene.fluid.inflow!.end_s >= 20);
  for (const { minimum: min, maximumExclusive: max } of patches) {
    assert.ok(min.every((value, axis) => Number.isInteger(value) && value >= 0 && value < max[axis]!));
    assert.ok(max.every((value, axis) => Number.isInteger(value) && value <= NBFLIP_THEME_PARK_GRID[axis]! - 4));
    assert.ok(min[0] >= 4 && min[2] >= 4);
  }
});

test("the entire theme park course has a solid bed, an open water corridor and continuous walls", () => {
  let previous = Infinity;
  for (let distance = 10; distance < NBFLIP_THEME_PARK_LENGTH - 1; distance += 0.5) {
    const [x, z, heading, floor] = nbflipThemeParkCentreline(distance);
    assert.ok(floor <= previous, `uphill at ${distance}`); previous = floor;
    for (const across of [-14, -8, 0, 8, 14]) {
      const px = x - across * Math.sin(heading), pz = z + across * Math.cos(heading);
      const bed = distance >= NBFLIP_THEME_PARK_LENGTH - 24 ? floor : floor + NBFLIP_THEME_PARK_HALF_WIDTH - Math.sqrt(NBFLIP_THEME_PARK_HALF_WIDTH ** 2 - across ** 2);
      assert.equal(occupied(px, bed - 3, pz), 1, `missing bed at ${distance}, across ${across}`);
      assert.equal(occupied(px, bed + 3, pz), 0, `blocked channel at ${distance}, across ${across}`);
    }
    if (distance < NBFLIP_THEME_PARK_LENGTH - 21) for (const across of [-18, 18]) {
      assert.equal(occupied(x - across * Math.sin(heading), floor + 24, z + across * Math.cos(heading)), 1,
        `missing wall at ${distance}, across ${across}`);
    }
  }
  assert.ok(nbflipThemeParkCentreline(10)[3] - previous >= 175);
  assert.ok(NBFLIP_THEME_PARK_LENGTH > 1400);
});

test("stacked spiral decks and the launch underpass retain vertical clearance", () => {
  const lap = 2 * Math.PI * 72;
  for (let distance = 16; distance < lap - 16; distance += 2) {
    const [x, z, , upper] = nbflipThemeParkCentreline(distance);
    const [lowerX, lowerZ, , lower] = nbflipThemeParkCentreline(distance + lap);
    assert.ok(Math.hypot(x - lowerX, z - lowerZ) < 1e-9);
    assert.equal(upper - lower, 60);
    assert.equal(occupied(x, upper - 2, z), 1);
    assert.equal(occupied(x, lower - 2, z), 1);
    for (let y = lower + 3; y < upper - 7; y++) assert.equal(occupied(x, y, z), 0, `sealed decks at ${distance}, y ${y}`);
  }
  const [x, z, , floor] = nbflipThemeParkCentreline(2 * lap + 10);
  assert.equal(occupied(x, floor + 3, z), 0, "the exit passes below the launch");
  assert.equal(occupied(x, 209, z), 1, "the launch deck crosses overhead");
});

test("the hood vents to a sealed splash pool with room for the emitted water", () => {
  const pool = NBFLIP_THEME_PARK_POOL;
  let capacity = 0;
  for (let z = pool.minZ + pool.wall; z < pool.maxZ - pool.wall; z++) {
    for (let x = pool.minX + pool.wall; x < pool.maxX - pool.wall; x++) {
      assert.equal(occupied(x, pool.floor - 1, z), 1);
      for (let y = pool.floor; y < 32; y++) if (!occupied(x, y, z)) capacity++;
    }
  }
  const inflow = scene.fluid.inflow!, h = scene.voxelDomain.finestCellSize_m;
  const emitted = Math.PI * inflow.radius_m ** 2 * Math.hypot(...Object.values(inflow.velocity_m_s)) * inflow.end_s / h ** 3;
  assert.ok(capacity > 1.3 * emitted && emitted > 0.5 * capacity, `${capacity} cells for ${emitted}`);
  for (const side of [-1, 1]) {
    for (let across = 0; across <= 20; across++) assert.equal(occupied(300, 35, 280 + side * across), 0, "hood vents sideways");
  }
  assert.equal(occupied(308, 35, 280), 1, "impact wall stops the stream");
  assert.equal(occupied(300, 65, 280), 1, "hood contains the impact");
});

test("physical pages cover the full wet course and pool with an eight-cell migration halo", () => {
  const pages = new Set<string>();
  const include = (min: readonly number[], max: readonly number[]) => {
    for (let z = Math.max(0, Math.floor((min[2]! - 8) / 32)); z <= Math.min(nz / 32 - 1, Math.floor((max[2]! + 8) / 32)); z++) {
      for (let y = Math.max(0, Math.floor((min[1]! - 8) / 32)); y <= Math.min(ny / 32 - 1, Math.floor((max[1]! + 8) / 32)); y++) {
        for (let x = Math.max(0, Math.floor((min[0]! - 8) / 32)); x <= Math.min(nx / 32 - 1, Math.floor((max[0]! + 8) / 32)); x++) pages.add(`${x},${y},${z}`);
      }
    }
  };
  for (let distance = 0; distance <= NBFLIP_THEME_PARK_LENGTH; distance++) {
    const [x, z, , floor] = nbflipThemeParkCentreline(distance);
    include([x - 20, floor, z - 20], [x + 20, floor + 40, z + 20]);
  }
  const pool = NBFLIP_THEME_PARK_POOL;
  include([pool.minX, pool.floor, pool.minZ], [pool.maxX, pool.rim, pool.maxZ]);
  assert.ok(pages.size + 2 <= scene.fluid.pagePool!.blocks!, `${pages.size} pages plus two sentinels`);
});

test("the white park survives serialization and fits its scenery budget", () => {
  const restored = parseScene(serializeScene(scene));
  // JSON normalizes signed zero in rotation components.
  assert.equal(JSON.stringify(restored.scenery), JSON.stringify(scene.scenery));
  const catalog = buildEnvironmentProxyCatalog(restored, restored.environment!);
  assert.ok(catalog.primitives.length > 2000, "retain the detailed branching trees and shrubs");
  assert.ok(catalog.primitives.length < 4096, "keep the detailed park within its scenery budget");
  assert.equal(new Set(catalog.primitives.map(p => p.key)).size, catalog.primitives.length);
  for (const primitive of catalog.primitives) {
    assert.deepEqual(primitive.material.colorLinear, [.9, .9, .9], `${primitive.key} must stay white`);
    for (const axis of ["x", "y", "z"] as const) {
      assert.ok(Number.isFinite(primitive.aabb_m.min[axis]) && Number.isFinite(primitive.aabb_m.max[axis]));
      assert.ok(primitive.aabb_m.min[axis] < primitive.aabb_m.max[axis]);
    }
  }
  assert.ok(restored.solidVoxels.filter(p => p.materialId !== VOXEL_MATERIAL_IDS.container)
    .every(p => p.materialId === VOXEL_MATERIAL_IDS.white), "the flume and pool share the white material");
  assert.equal(presentationModeForScene(definition, restored), "full-scene", "the complete park must remain visible");
});

test("the flume's continuous surface has open channels and consistent curved wall normals", () => {
  const surface = scene.solidSurface;
  assert.equal(surface?.kind, "swept-flume");
  if (surface?.kind !== "swept-flume") return;
  assert.deepEqual(sweptFlumeSurfaceErrors(surface), []);
  assert.deepEqual(parseScene(serializeScene(scene)).solidSurface, surface);
  const sample = (p: readonly number[]) => sweptFlumeDistance(surface, p[0]!, p[1]!, p[2]!);
  for (const p of surface.pieces) for (const t of [.13, .35, .65, .85]) {
    const d = p.length * t, a = p.heading + p.curvature * d;
    const x = p.x + (p.curvature ? (Math.sin(a) - Math.sin(p.heading)) / p.curvature : d * Math.cos(a));
    const z = p.z + (p.curvature ? (Math.cos(p.heading) - Math.cos(a)) / p.curvature : d * Math.sin(a));
    const floor = p.top - p.fall * t;
    assert.ok(sample([x, floor - .04, z]) < 0, "continuous bed");
    assert.ok(sample([x, floor + .08, z]) > 0, "open channel");
    if (d >= p.wallEnd) continue;
    for (const side of [-1, 1]) {
      const normal = [-Math.sin(a) * side, 0, Math.cos(a) * side];
      const q = [x + normal[0]! * (surface.halfWidth + surface.wallThickness), floor + .45,
        z + normal[2]! * (surface.halfWidth + surface.wallThickness)];
      assert.ok(Math.abs(sample(q)) < 1e-9, "wall lies on the authored curve");
      const e = 1e-5;
      const g = q.map((_, axis) => (sample(q.map((v, i) => v + (i === axis ? e : 0)))
        - sample(q.map((v, i) => v - (i === axis ? e : 0)))) / (2 * e));
      const alignment = g.reduce((sum, v, axis) => sum + v * normal[axis]!, 0) / Math.hypot(...g);
      assert.ok(alignment > .9999, `normal must follow the curve, not a voxel face: ${alignment}`);
    }
  }
  // The floor and wall are one shell: no zero-distance internal seam along
  // their join, which otherwise appears as a dark horizontal crack.
  const p = surface.pieces[2]!, x = p.x + p.length * .5, floor = p.top - p.fall * .5;
  assert.ok(sample([x, floor + surface.halfWidth, p.z + surface.halfWidth + surface.wallThickness * .5]) < -.02);
});


test("round flume has a semicircular bed and higher outside bend lips", () => {
  const surface = scene.solidSurface;
  assert.equal(surface?.kind, "swept-flume");
  if (surface?.kind !== "swept-flume") return;
  assert.equal(surface.crossSection, "semicircular");
  const r = surface.halfWidth;
  const isolated = { ...surface, boxes: [], pieces: [{ x: 0, z: 0, heading: 0, curvature: 0,
    length: 2, top: 1, fall: 0, wallEnd: 2, hood: false }] };
  for (const angle of [-1.4, -.9, -.4, 0, .4, .9, 1.4]) {
    const across = r * Math.sin(angle), y = 1 + r - r * Math.cos(angle);
    assert.ok(Math.abs(sweptFlumeDistance(isolated, 1, y, across)) < 1e-9, "inner bed is a circle");
    assert.ok(sweptFlumeDistance(isolated, 1, y + .02, across) > 0, "water lies above the bowl");
    assert.ok(sweptFlumeDistance(isolated, 1, y - .02, across) < 0, "shell supports the bowl");
  }
  const p = surface.pieces[0]!, d = p.length / 2, a = p.heading + p.curvature * d;
  const x = p.x + (Math.sin(a) - Math.sin(p.heading)) / p.curvature;
  const z = p.z + (Math.cos(p.heading) - Math.cos(a)) / p.curvature;
  const y = p.top - p.fall / 2 + surface.wallHeight + .06;
  for (const side of [-1, 1]) {
    const across: number = side * (r + surface.wallThickness / 2);
    const value = sweptFlumeDistance(surface, x - across * Math.sin(a), y, z + across * Math.cos(a));
    assert.equal(value < 0, side < 0, "only the outside wall rises above the standard lip");
    const cx: number = (x - across * Math.sin(a)) / .02 + nx / 2;
    const cz: number = (z + across * Math.cos(a)) / .02 + nz / 2;
    assert.equal(occupied(cx, y / .02, cz), side < 0 ? 1 : 0, "collider matches the raised lip");
  }
});
