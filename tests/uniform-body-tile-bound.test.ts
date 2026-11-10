import assert from "node:assert/strict";
import test from "node:test";
import { createPaddleboatsScene } from "../lib/core/paddleboats-scene";
import { rigidMotorPose } from "../lib/core/rigid-motor";
import { quaternionRotate } from "../lib/core/rigid-body";
import { bodyBoxTileGap, bodyMarksTile, boxDistance, quatRotate } from "../lib/methods/uniform/uniform-body-tile-bound";

// markTiles' box gap (uniform-body-tile-bound.ts) on the CPU. The WGSL is the
// same arithmetic; naga and the GPU stencil tests hold the kernel.
type V = readonly [number, number, number];
type Q = readonly [number, number, number, number];
type Body = { centre: V; orientation: Q; dimensions: V; velocity: V; spin: V };
const h = .025, horizon = 1 / 60, cell: V = [h, h, h];
const HULL: V = [1.4, .28, .54], BLADE: V = [.13, .05, .5];

function random(seed: number) { let s = seed >>> 0; return () => ((s = Math.imul(s ^ (s >>> 15), 0x2c1b3c6d) + 0x9e3779b9 >>> 0) / 2 ** 32); }
function unitQuat(r: () => number): Q {
  const q = [r() - .5, r() - .5, r() - .5, r() - .5], l = Math.hypot(...q);
  return [q[0]! / l, q[1]! / l, q[2]! / l, q[3]! / l];
}
function multiply(a: Q, b: Q): Q {
  return [a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
}
/** The body's pose t seconds on: centre c + v t, orientation turned by the world spin w over t. */
function posed(b: Body, t: number): { centre: V; orientation: Q } {
  const w = Math.hypot(...b.spin), s = Math.sin(w * t / 2) / (w || 1);
  return { centre: [b.centre[0] + b.velocity[0] * t, b.centre[1] + b.velocity[1] * t, b.centre[2] + b.velocity[2] * t],
    orientation: multiply([Math.cos(w * t / 2), s * b.spin[0], s * b.spin[1], s * b.spin[2]], b.orientation) };
}
const conj = (q: Q): Q => [q[0], -q[1], -q[2], -q[3]];
/** Dense samples of the tile (spacing h/2): their distance to a pose, the
 * least of which is at least the true distance. */
function sampledDistance(pose: { centre: V; orientation: Q }, dims: V, low: V): number {
  let best = Infinity; const q = conj(pose.orientation);
  for (let i = 0; i <= 8; i++) for (let j = 0; j <= 8; j++) for (let k = 0; k <= 8; k++) {
    const p: V = [low[0] + i * h / 2 - pose.centre[0], low[1] + j * h / 2 - pose.centre[1], low[2] + k * h / 2 - pose.centre[2]];
    best = Math.min(best, Math.max(0, boxDistance(dims, quatRotate(q, p))));
    if (best === 0) return 0;
  }
  return best;
}
/** Tiles (4h, on the h grid from the origin) whose boxes come within reach of the body's ball. */
function* candidateTiles(b: Body): Generator<V> {
  const reach = Math.hypot(...b.dimensions) / 2 + Math.hypot(...b.velocity) * horizon + 2 * h;
  const lo = b.centre.map(c => Math.floor((c - reach) / (4 * h))), hi = b.centre.map(c => Math.floor((c + reach) / (4 * h)));
  for (let x = lo[0]!; x <= hi[0]!; x++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let z = lo[2]!; z <= hi[2]!; z++) yield [4 * h * x, 4 * h * y, 4 * h * z];
}

test("the box gap never exceeds the boxes' distance and is exact beside a face", () => {
  const r = random(7);
  for (let trial = 0; trial < 12; trial++) {
    const dims = trial % 2 ? HULL : BLADE, body = { centre: [r() - .5, r() - .5, r() - .5] as V, orientation: unitQuat(r), dimensions: dims };
    for (const low of candidateTiles({ ...body, velocity: [0, 0, 0], spin: [0, 0, 0] })) {
      const gap = bodyBoxTileGap(body, [low[0] + 2 * h, low[1] + 2 * h, low[2] + 2 * h], [2 * h, 2 * h, 2 * h]);
      assert.ok(gap <= sampledDistance(body, dims, low) + 1e-12, `gap ${gap} above the distance at tile ${low}`);
    }
  }
  // An axis-aligned hull with its top face 3h below a tile row: the gap is the 3h, the old bound 1.54 h.
  const hull = { centre: [0, 0, 0] as V, orientation: [1, 0, 0, 0] as Q, dimensions: HULL }, top = HULL[1] / 2 + 3 * h;
  const middle: V = [.05, top + 2 * h, .05];
  assert.ok(Math.abs(bodyBoxTileGap(hull, middle, [2 * h, 2 * h, 2 * h]) - 3 * h) < 1e-12);
  assert.ok(Math.abs(boxDistance(HULL, middle) - Math.hypot(2 * h, 2 * h, 2 * h) - (5 - Math.sqrt(12)) * h) < 1e-12);
});

test("markTiles' new rule marks every tile a box body's swept shape reaches within the horizon and a cell", () => {
  const r = random(11); let reached = 0, oldMarks = 0, newMarks = 0;
  for (let trial = 0; trial < 8; trial++) {
    const dims = trial % 2 ? HULL : BLADE, speed = 2 * r(), spin = 15 * r(), v = unitQuat(r), w = unitQuat(r);
    const lv = Math.hypot(v[1], v[2], v[3]), lw = Math.hypot(w[1], w[2], w[3]);
    const body: Body = { centre: [r() - .5, r() - .5, r() - .5], orientation: unitQuat(r), dimensions: dims,
      velocity: [speed * v[1] / lv, speed * v[2] / lv, speed * v[3] / lv], spin: [spin * w[1] / lw, spin * w[2] / lw, spin * w[3] / lw] };
    const poses = Array.from({ length: 9 }, (_, i) => posed(body, horizon * i / 8));
    for (const low of candidateTiles(body)) {
      const marked = bodyMarksTile(body, low, cell, horizon, "new"), old = bodyMarksTile(body, low, cell, horizon, "old");
      assert.ok(!marked || old, `the new rule marked tile ${low} the old one did not`);
      oldMarks += +old; newMarks += +marked;
      if (marked) { reached += +poses.some(p => sampledDistance(p, dims, low) <= h); continue; }
      // Unmarked: no sampled point of the tile may come within a cell of the body at any sampled time.
      for (const pose of poses) {
        const d = sampledDistance(pose, dims, low);
        if (d <= h) assert.fail(`trial ${trial}: tile ${low} unmarked, the swept body within ${d} of it`);
      }
    }
  }
  assert.ok(reached > 0 && newMarks < oldMarks, `reached ${reached}, old ${oldMarks}, new ${newMarks}`);
  console.log(`random box bodies: ${reached} tiles reached by sampling, old rule marks ${oldMarks}, new ${newMarks}`);
});

test("paddleboats at t = 0.2 s: the new rule marks fewer tiles, a subset of the old", () => {
  const scene = createPaddleboatsScene(), time = .2, c = scene.container;
  const hulls = new Map(scene.rigidBodies.filter(b => !b.motorAttachment).map(b => [b.id, b]));
  const bodies: Body[] = scene.rigidBodies.map(b => {
    const m = b.motorAttachment, dims: V = [b.dimensions_m.x, b.dimensions_m.y, b.dimensions_m.z];
    if (!m) return { centre: [b.position_m.x, b.position_m.y, b.position_m.z], orientation: [b.orientation.w, b.orientation.x, b.orientation.y, b.orientation.z], dimensions: dims, velocity: [0, 0, 0], spin: [0, 0, 0] };
    const hull = hulls.get(m.parentId)!, strength = hull.motorStrengths?.[m.channel!] ?? 1;
    const drive = { ...m, angularSpeed_rad_s: m.angularSpeed_rad_s * strength }, pose = rigidMotorPose(hull, drive, time);
    const u = Math.min(1, time / m.rampUp_s), rate = drive.angularSpeed_rad_s * (3 * u * u - 2 * u * u * u);
    const axis = quaternionRotate(hull.orientation, m.axis), omega: V = [axis.x * rate, axis.y * rate, axis.z * rate];
    const pivot = quaternionRotate(hull.orientation, m.pivot_m);
    const arm: V = [pose.position_m.x - hull.position_m.x - pivot.x, pose.position_m.y - hull.position_m.y - pivot.y, pose.position_m.z - hull.position_m.z - pivot.z];
    return { centre: [pose.position_m.x, pose.position_m.y, pose.position_m.z], orientation: [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z],
      dimensions: dims, velocity: [omega[1] * arm[2] - omega[2] * arm[1], omega[2] * arm[0] - omega[0] * arm[2], omega[0] * arm[1] - omega[1] * arm[0]], spin: omega };
  });
  assert.equal(bodies.length, 18);
  const count = { old: 0, new: 0, hullOld: 0, hullNew: 0 }, tiles = [c.width_m, c.height_m, c.depth_m].map(l => Math.ceil(l / (4 * h)));
  for (let x = 0; x < tiles[0]!; x++) for (let y = 0; y < tiles[1]!; y++) for (let z = 0; z < tiles[2]!; z++) {
    const low: V = [-c.width_m / 2 + 4 * h * x, 4 * h * y, -c.depth_m / 2 + 4 * h * z];
    const old = bodies.some(b => bodyMarksTile(b, low, cell, horizon, "old")), marked = bodies.some(b => bodyMarksTile(b, low, cell, horizon, "new"));
    assert.ok(!marked || old);
    count.old += +old; count.new += +marked;
    count.hullOld += +bodies.some((b, i) => !scene.rigidBodies[i]!.motorAttachment && bodyMarksTile(b, low, cell, horizon, "old"));
    count.hullNew += +bodies.some((b, i) => !scene.rigidBodies[i]!.motorAttachment && bodyMarksTile(b, low, cell, horizon, "new"));
  }
  assert.ok(count.new < count.old, JSON.stringify(count));
  console.log(`paddleboats t=0.2 s marked tiles: old ${count.old}, new ${count.new} (hulls alone ${count.hullOld} -> ${count.hullNew})`);
});
