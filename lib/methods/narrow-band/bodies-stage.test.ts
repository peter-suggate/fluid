// Lane P's body stages in the engine (A2-A6; bodies-stage.wgsl.ts, setMark's body criterion): a sphere in a sheet of
// liquid on E's bring-up stages.
// - Boot: the entry path stamps the boot bricks a body reaches; setMark's NEAR_BODY is exactly the CPU criterion
//   (pbBodyMarkTile for a sphere: |middle - c| - r - length(2h) <= cell, after the ball prefilter); every NEAR_BODY brick
//   holds a K slot with BMASK = the body, FRAC 0 well inside, 1 well outside, OPEN = FRAC > 1e-5, W = 0 at rest.
// - Moving: the body's W on its faces is its velocity; the bricks it left are BODY_WAS for a step (cleaned to the
//   static values, BMASK 0), then give their K slots back.
// - Statics: a body passing a CUT_STATIC brick (E's mask bound) leaves solidStamp's words bitwise as they were.
// GPU only.
import assert from "node:assert/strict";
import test from "node:test";
import { BODY_FLAG, BRICK_CELLS, BRICK_STRIDE, FLAG, F_CLASS, H, NONE, REC, brickKeyCoord } from "./brick-abi";

type Vec3 = [number, number, number];
import { BrickEngine, type BrickEngineOptions } from "./brick-engine";
import { compositeMGPCG } from "./brick-pressure-solver";
import { bringUpStages } from "./brick-stages";
import { snapshot } from "./brick-readback";
import { SCENE_SHAPE_TABLE } from "../../core/scene-shape";
import { RIGID_BODY_CUT_ATLAS_WIDTH } from "../../core/solid-cuts";
import { GPU_RIGID_STATE_BYTES } from "../../core/webgpu-rigid-body";
import { GPU_RIGID_EXCHANGE_BYTES } from "../../core/webgpu-eulerian";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const H_M = 1 / 32, DT = 1 / 60, G = 9.81, N = 32;
const asF32 = (u: Uint32Array) => new Float32Array(u.buffer, u.byteOffset, u.length);

interface Sphere { c: Vec3; r: number; v: Vec3 }
function record(s: Sphere): number[] {
  return [...s.c, SCENE_SHAPE_TABLE.sphere.code, s.r, s.r, s.r, s.r, 1, 0, 0, 0, ...s.v, 1, 0, 0, 0, 1000, 1, 1, 1, 1, 0, 0, 0, 0.05, 0.3, 0, 0, 0];
}

async function engine(device: GPUDevice, sphere: Sphere, staticMask?: BrickEngineOptions["staticMask"], count = 1) {
  const bodies = device.createBuffer({ size: GPU_RIGID_STATE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const cuts = device.createBuffer({ size: 16 * RIGID_BODY_CUT_ATLAS_WIDTH, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const exchange = device.createBuffer({ size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  let pose: Sphere | undefined = sphere; const syncs: number[] = [];
  const o: BrickEngineOptions = {
    lattice: [N, N, N], h_m: H_M, dt_s: DT, gravity_mps2: [0, -G, 0], capacities: { f: 128, c: 8, b: 0, k: 96, particles: 1024 },
    liquid: [{ lo: [-1, -1, -1], hi: [N + 1, 16.3, N + 1] }], solver: compositeMGPCG(), stages: bringUpStages,
    ...(staticMask ? { staticMask } : {}),
    bodies: {
      frame_m: [0, 0, 0], h_m: H_M, openTop: false, bodies, cuts, exchange, density_kg_m3: 1000, integrate: () => undefined,
      sync(step) { syncs.push(step); if (pose) { device.queue.writeBuffer(bodies, 0, new Float32Array(record(pose))); pose = undefined; } return { count, flags: BODY_FLAG.COUPLING, generation: 1 }; },
    },
  };
  const e = await BrickEngine.create(device, o);
  return { e, move(s: Sphere) { pose = s; }, syncs, destroy() { e.destroy(); bodies.destroy(); cuts.destroy(); exchange.destroy(); } };
}

/** The CPU criterion: the bricks with a tile pbBodyMarkTile marks for a sphere (horizon dt). */
function marked(s: Sphere): Set<number> {
  const out = new Set<number>(), cell = H_M, half = Math.hypot(2 * H_M, 2 * H_M, 2 * H_M), speed = Math.hypot(...s.v);
  for (let tz = 0; tz < N / 4; tz++) for (let ty = 0; ty < N / 4; ty++) for (let tx = 0; tx < N / 4; tx++) {
    const lo = [tx, ty, tz].map(t => 4 * t * H_M), hi = lo.map(v => v + 4 * H_M), mid = lo.map((v, i) => 0.5 * (v + hi[i]!));
    const clamped = s.c.map((v, i) => Math.min(Math.max(v, lo[i]!), hi[i]!));
    if (Math.hypot(...clamped.map((v, i) => v - s.c[i]!)) > s.r + speed * DT + cell) continue;
    const margin = speed * DT + cell, lower = Math.hypot(...mid.map((v, i) => v - s.c[i]!)) - s.r - half;
    // f32 rounding in the shader: skip tiles within 1e-5 m of the threshold.
    if (Math.abs(lower - margin) < 1e-5) throw new Error(`tile ${tx},${ty},${tz} sits on the criterion's threshold`);
    if (lower <= margin) out.add((tx >> 1) | ((ty >> 1) << 10) | ((tz >> 1) << 20));
  }
  return out;
}

test("bodies stage (GPU): the boot's entry path and setMark mark and stamp exactly the bricks the sphere criterion names", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("bodies stage", async device => {
    const sphere: Sphere = { c: [0.51, 0.3, 0.47], r: 0.1, v: [0, 0, 0] };
    const run = await engine(device, sphere), { e } = run;
    await e.assertHealthy();
    let s = await snapshot(e);
    const L = e.layout, CLASS = REC.CLASS, CUT = REC.CUT;
    const cls = (slot: number) => s.dir[L.dir.frec + 32 * slot + CLASS]!, kslot = (slot: number) => s.dir[L.dir.frec + 32 * slot + CUT]!;
    const want = marked(sphere);
    const near = new Set([...s.bricks].filter(([, slot]) => (cls(slot) & F_CLASS.NEAR_BODY) !== 0).map(([key]) => key));
    t.diagnostic(`boot: ${s.bricks.size} F bricks, ${near.size} NEAR_BODY (criterion ${want.size}), LIVE_K ${s.header[H.LIVE_K]}`);
    assert.deepEqual([...near].sort((a, b) => a - b), [...want].filter(k => s.bricks.has(k)).sort((a, b) => a - b));
    assert.ok(near.size > 0);
    const cut = asF32(await e.read("cut")), phi = await e.read("phi"), base = (ch: string) => L.base("cut", "K", ch);
    let inside = 0, outside = 0;
    for (const key of near) {
      const slot = s.bricks.get(key)!, k = kslot(slot);
      assert.notEqual(k, NONE, `NEAR_BODY brick ${key} holds a K slot`);
      assert.equal(new Uint32Array(cut.buffer, 4 * (L.cutBmask + k), 1)[0], 1, "BMASK");
      const b = brickKeyCoord(key);
      for (let l = 0; l < BRICK_CELLS; l++) {
        const c = [8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)], d = Math.hypot(...c.map((v, i) => (v + 0.5) * H_M - sphere.c[i]!));
        const frac = cut[base("FRAC") + BRICK_STRIDE * k + l]!, open = (phi[L.base("phi", "F", "FLAGS") + BRICK_STRIDE * slot + l]! & FLAG.OPEN) !== 0;
        assert.equal(open, frac > 1e-5, `OPEN at ${c}`);
        if (d < sphere.r - H_M) { assert.equal(frac, 0, `inside ${c}`); inside++; }
        if (d > sphere.r + H_M) { assert.equal(frac, 1, `outside ${c}`); outside++; }
        for (const ch of ["W0", "W1", "W2"]) assert.equal(cut[base(ch) + BRICK_STRIDE * k + l], 0, `${ch} at rest`);
      }
    }
    assert.ok(inside > 0 && outside > 0, `${inside} inside, ${outside} outside`);
    // One step at rest: the same NEAR_BODY set, no FAIL; the entry pass's extra bricks (the prefilter's wider reach)
    // were BODY_WAS for the boot's class pass and are released by step 2's head.
    e.advance(2);
    await e.assertHealthy();
    s = await snapshot(e);
    const held = [...s.bricks].filter(([, slot]) => kslot(slot) !== NONE).map(([key]) => key).sort((a, b) => a - b);
    assert.deepEqual(held, [...near].sort((a, b) => a - b), "after two steps only the NEAR_BODY bricks hold K slots");
    assert.equal(s.header[H.LIVE_K], near.size);
    run.destroy();
  });
});

test("bodies stage (GPU): a moving sphere's W is its velocity, and the bricks it leaves clean up and release their slots", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("bodies stage", async device => {
    const a: Sphere = { c: [0.27, 0.3, 0.47], r: 0.1, v: [0.6, 0, 0] };
    const run = await engine(device, a), { e } = run;
    await e.assertHealthy();
    const L = e.layout, CLASS = REC.CLASS, CUT = REC.CUT, base = (ch: string) => L.base("cut", "K", ch);
    let s = await snapshot(e);
    const cls = (slot: number) => s.dir[L.dir.frec + 32 * slot + CLASS]!, kslot = (slot: number) => s.dir[L.dir.frec + 32 * slot + CUT]!;
    // W: every lower x face of a stamped brick whose 8 dual samples all lie inside the sphere carries v.x, y and z 0.
    let cut = asF32(await e.read("cut")), full = 0;
    for (const [key, slot] of s.bricks) {
      const k = kslot(slot); if (k === NONE || (cls(slot) & F_CLASS.NEAR_BODY) === 0) continue;
      const b = brickKeyCoord(key);
      for (let l = 0; l < BRICK_CELLS; l++) {
        const c = [8 * b[0] + (l & 7), 8 * b[1] + ((l >> 3) & 7), 8 * b[2] + (l >> 6)];
        const face = [c[0]! * H_M, (c[1]! + 0.5) * H_M, (c[2]! + 0.5) * H_M];
        if (Math.hypot(...face.map((v, i) => v - a.c[i]!)) + Math.hypot(0.4, 0.4, 0.4) * H_M < a.r) {
          full++;
          assert.ok(Math.abs(cut[base("W0") + BRICK_STRIDE * k + l]! - 0.6) <= 1e-6, `W0 at ${c}: ${cut[base("W0") + BRICK_STRIDE * k + l]}`);
        }
        assert.equal(cut[base("W1") + BRICK_STRIDE * k + l], 0); assert.equal(cut[base("W2") + BRICK_STRIDE * k + l], 0);
      }
    }
    assert.ok(full > 0, "faces inside the sphere");
    // Jump the sphere 10 cells in +x (at rest): step 1 marks the new bricks and leaves the old ones BODY_WAS; step 1's
    // head had stamped with the new pose already (the sync precedes part 1).
    const b: Sphere = { c: [0.27 + 10 * H_M, 0.3, 0.47], r: 0.1, v: [0, 0, 0] };
    const before = marked(a), after = marked(b), left = [...before].filter(k => !after.has(k) && s.bricks.has(k));
    assert.ok(left.length > 0, "the jump leaves bricks");
    run.move(b); e.advance(1); await e.assertHealthy();
    s = await snapshot(e); cut = asF32(await e.read("cut"));
    for (const key of left) {
      const slot = s.bricks.get(key)!;
      assert.equal(cls(slot) & (F_CLASS.NEAR_BODY | F_CLASS.BODY_WAS), F_CLASS.BODY_WAS, `brick ${key} BODY_WAS after the jump`);
      const k = kslot(slot); assert.notEqual(k, NONE, "BODY_WAS holds its slot");
      // Cleaned at this step's head? No: this step's head ran before setMark saw the jump; the brick was NEAR_BODY
      // there and stamped with the new pose, which no longer reaches it: FRAC 1, W 0 everywhere.
      for (let l = 0; l < BRICK_CELLS; l++) { assert.equal(cut[base("FRAC") + BRICK_STRIDE * k + l], 1); assert.equal(cut[base("W0") + BRICK_STRIDE * k + l], 0); }
    }
    e.advance(1); await e.assertHealthy(); s = await snapshot(e); cut = asF32(await e.read("cut"));
    for (const key of left) {
      const slot = s.bricks.get(key)!;
      assert.equal(cls(slot) & (F_CLASS.NEAR_BODY | F_CLASS.BODY_WAS), 0, `brick ${key} clear`);
    }
    e.advance(1); await e.assertHealthy(); s = await snapshot(e);
    for (const key of left) assert.equal(kslot(s.bricks.get(key)!), NONE, `brick ${key} released its slot`);
    const near = [...s.bricks].filter(([, slot]) => (cls(slot) & F_CLASS.NEAR_BODY) !== 0).map(([key]) => key).sort((x, y) => x - y);
    assert.deepEqual(near, [...after].filter(k => s.bricks.has(k)).sort((x, y) => x - y));
    t.diagnostic(`moving: ${full} full faces at W0 = 0.6; ${left.length} bricks left, LIVE_K ${s.header[H.LIVE_K]}, syncs ${run.syncs.slice(0, 6).join(",")}`);
    run.destroy();
  });
});

test("bodies stage (GPU): a body passing a CUT_STATIC brick leaves solidStamp's words bitwise as they were", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { SolidOccupancyMask } = await import("../../core/solid-occupancy-mask");
  const { createSolidWorld } = await import("../../core/solid-world");
  const { brickStaticMaskWalls } = await import("./brick-static.wgsl");
  const walls = brickStaticMaskWalls([N, N, N], false);
  const mask = new SolidOccupancyMask({ walls, tileKeys: 0, slots: 64, separateTerrain: true });
  // Above the liquid (y > 16.3: a static solid cell holding liquid latches 0xa2, its rows not ported yet): a voxel
  // block in brick (1, 2, 1) and a ball across bricks (2, 2, 1) and (2, 2, 2).
  mask.update(createSolidWorld([
    { operation: "fill", minimum: [12, 18, 12], maximumExclusive: [16, 22, 16] },
    { operation: "fill", kind: "ellipsoid", minimum: [17, 18, 13], maximumExclusive: [22, 22, 18] },
  ]));
  mask.seal(mask.slotsHeld);
  await withUniformDevice("bodies stage", async device => {
    const far: Sphere = { c: [0.8, 0.3, 0.8], r: 0.1, v: [0, 0, 0] }, over: Sphere = { c: [14 * H_M, 21 * H_M, 14 * H_M], r: 0.1, v: [0, 0, 0] };
    // The reference: the same scene with no body present (FLAGS.BODIES off), solidStamp's words alone.
    const ref = await engine(device, far, { words: mask.words, walls }, 0);
    await ref.e.assertHealthy();
    const run = await engine(device, far, { words: mask.words, walls }), { e } = run;
    await e.assertHealthy();
    const L = e.layout, CHANNELS = ["FRAC", "AP0", "AP1", "AP2", "W0", "W1", "W2"];
    // A CUT_STATIC brick's slot words: the 7 channels and BMASK (as u32).
    const words = async (e: BrickEngine) => {
      const s = await snapshot(e), cut = await e.read("cut"), out = new Map<number, number[]>();
      for (const [key, slot] of s.bricks) {
        const r = L.dir.frec + 32 * slot, cls = s.dir[r + REC.CLASS]!, k = s.dir[r + REC.CUT]!;
        if ((cls & F_CLASS.CUT_STATIC) === 0) continue;
        assert.notEqual(k, NONE);
        out.set(key, [...CHANNELS.flatMap(ch => Array.from(cut.subarray(L.base("cut", "K", ch) + BRICK_STRIDE * k, L.base("cut", "K", ch) + BRICK_STRIDE * k + BRICK_CELLS))), cut[L.cutBmask + k]!]);
      }
      return { out, s };
    };
    const before = await words(ref.e);
    // The far sphere's reach crosses the x = N wall into the skirt bricks (the vertex row): none of them is a body brick.
    assert.ok(far.c[0] + far.r + Math.hypot(2 * H_M, 2 * H_M, 2 * H_M) + H_M > 1, "the far sphere's reach crosses the wall");
    const boot = await snapshot(e);
    for (const [key, slot] of boot.bricks) if ((boot.dir[L.dir.frec + 32 * slot + REC.CLASS]! & F_CLASS.NEAR_BODY) !== 0) assert.ok(brickKeyCoord(key).every(v => 8 * v < N), `skirt brick ${brickKeyCoord(key)} NEAR_BODY`);
    assert.ok(before.out.size >= 2, `${before.out.size} CUT_STATIC bricks`);
    // Over: the static bricks go NEAR_BODY (setMark, step 1) and are stamped with the body (step 2's head).
    run.move(over); e.advance(2); await e.assertHealthy();
    const during = await words(e);
    const changed = [...before.out].filter(([key, w]) => during.out.get(key)!.some((v, i) => v !== w[i])).length;
    assert.ok(changed >= 1, "the body changes a static brick's words");
    // Away: one step stamped with the far pose (its candidates in BMASK), then BODY_WAS's clean-up pass (BMASK 0), then clear.
    run.move(far); e.advance(3); await e.assertHealthy();
    const after = await words(e);
    assert.deepEqual([...after.out.keys()].sort(), [...before.out.keys()].sort());
    let differ = 0;
    for (const [key, w] of before.out) { const v = after.out.get(key)!; for (let i = 0; i < w.length; i++) if (v[i] !== w[i] && differ++ < 3) console.error(key, i >> 9, i & 511, w[i], v[i]); }
    assert.equal(differ, 0, "solidStamp's words, bitwise");
    t.diagnostic(`${before.out.size} CUT_STATIC bricks, ${changed} changed by the body, restored bitwise`);
    run.destroy(); ref.destroy();
  });
});

test("bodies stage: bodyStamp's static hooks see the container's box on a padded lattice (L.box, as solidStamp's)", async () => {
  const { BrickLayout } = await import("./brick-abi"), { brickBodyKernels } = await import("./bodies-stage.wgsl");
  const frame = { frame_m: [0, 0, 0] as Vec3, h_m: H_M };
  for (const [box, top] of [[[16, 16, 16], false], [[24, 16, 24], true], [undefined, false]] as const) {
    const L = new BrickLayout({ lattice: [32, 32, 32], ...(box ? { box } : {}), f: 125, c: 8, b: 16, k: 8, particles: 1024 });
    const code = brickBodyKernels(L, frame, { openTop: top }).bodyStamp.code, b = box ?? [32, 32, 32];
    assert.ok(code.includes(`const NB_STATIC_BOX:vec3i=vec3i(${b.join(",")});`), `box ${b}: the hooks' box`);
    assert.ok(code.includes(`const NB_STATIC_OPEN_TOP:bool=${top};`));
  }
});
