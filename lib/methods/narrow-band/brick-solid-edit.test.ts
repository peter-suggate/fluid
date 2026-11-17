// LANE E OWNS THIS FILE. The live voxel edit on the brick engine (M3 stage 2c): the AGENTS.md scene authoring
// regressions for the static mask's representation on the running engine.
//
// The solver takes a voxel stroke in place (BrickNBFlipSolver.validateLiveSolidEdit, applySceneUniforms): the setup's
// mask folds it, the rewritten words go up, and solidRestamp restamps the bricks the changed cells reach.
// - Hit testing: the engine's bound image is, cell for cell, the SolidWorld the editor's voxel reader answers from,
//   after every edit, undo and redo.
// - Local edits: an edit changes FLAGS.OPEN, the class word and the cut slot of the bricks within the ring of its
//   changed cells, and no other word: every other brick, the velocity pool and every φ value keep every bit.
// - The edited cells: a fill closes its cells (FLAGS.OPEN clear, the cut's FRAC 0, CUT_STATIC), a clear opens them,
//   and a brick the clear leaves uncut gives its K slot back.
// - Undo and redo: each state again, brick for brick; the restamp equals the stamp a solver built from the saved and
//   loaded document makes, on every brick both hold.
// - Refusals are explicit and change nothing: an unknown patch kind, a change beyond the voxels.
// - The run goes on: steps after the edits latch nothing, and no live record lies in a closed cell.
import assert from "node:assert/strict";
import test from "node:test";
import type { SceneDescription } from "../../core/model";
import { sceneDiffersInSolidsAlone } from "./brick-solver";
import "../index";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V3 = [number, number, number];

test("brick solid edit: a scene differs in its solids alone only when every other field has the same content", () => {
  const base = { sceneId: "a", fluid: { density_kg_m3: 1000 }, solidVoxels: [] } as unknown as SceneDescription;
  assert.ok(sceneDiffersInSolidsAlone({ ...base, solidVoxels: [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }] } as SceneDescription, base));
  assert.ok(sceneDiffersInSolidsAlone(JSON.parse(JSON.stringify(base)) as SceneDescription, base), "a rebuilt field with the same content");
  assert.ok(!sceneDiffersInSolidsAlone({ ...base, fluid: { density_kg_m3: 998 } } as unknown as SceneDescription, base), "a fluid change");
  assert.ok(!sceneDiffersInSolidsAlone({ ...base, terrain: {} } as unknown as SceneDescription, base), "an added field");
});

test("brick solid edit (GPU): fill, undo, redo, clear and reload on the running dam, local and explicit", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { getMethod } = await import("../../core/method-registry");
  const { resolveMethodValues } = await import("../../core/method-contract");
  const { sceneDocument } = await import("../../core/scene-definition");
  const { getSceneDefinition } = await import("../../core/scenes");
  const { sceneWithSolidStroke } = await import("../../core/solid-world");
  const { solidOccupancyMaskVoxel } = await import("../../core/solid-occupancy-mask");
  const { voxelReaderForScene } = await import("../../core/voxel-editor/voxel-reader");
  const { BRICK_STRIDE, F_CLASS, FLAG, H, NONE, REC, brickKeyCoord } = await import("./brick-abi");
  const { snapshot } = await import("./brick-readback");
  const { createBrickSolver } = await import("./brick-method");
  type Solver = Awaited<ReturnType<typeof createBrickSolver>>;

  // The 64^3 dam (its column x, z < 40) with P's obstacle (x 48..54, y 0..12, z 20..44) and authored pools (the rule's dam64 f 729, c 27) with K slots.
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition("minimal-power-dam-break-64");
  const dam = sceneDocument(definition);
  const scene0 = sceneWithSolidStroke(dam, [{ operation: "fill", minimum: [48, 0, 20], maximumExclusive: [54, 12, 44], materialId: 1 }]);
  scene0.fluid.brickPools = { f: 729, c: 27, b: 16, k: 160, particles: 600000 };
  scene0.duration_s = 3;
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" }) as Record<string, unknown>;
  // The fill lies on the floor deep inside the column (4h there); the clear empties its low corner, bricks (1, 0, 1)
  // and (1, 1, 1).
  const B = { lo: [8, 0, 8] as V3, hi: [20, 10, 20] as V3 }, C = { lo: [8, 0, 8] as V3, hi: [16, 10, 16] as V3 };
  const filled = sceneWithSolidStroke(scene0, [{ operation: "fill", minimum: B.lo, maximumExclusive: B.hi, materialId: 2 }]);
  const cleared = sceneWithSolidStroke(filled, [{ operation: "clear", minimum: C.lo, maximumExclusive: C.hi }]);
  const N: V3 = [64, 64, 64], RING = 8;
  const inBox = (b: { lo: V3; hi: V3 }, c: readonly number[]) => c.every((v, a) => v >= b.lo[a]! && v < b.hi[a]!);

  // A state: per resident brick, its class bits, its cut (FRAC and the AP channels, or none) and its OPEN bits; and the
  // pools whole, for the bitwise checks.
  interface Brick { cls: number; cut?: Float32Array; open: Uint8Array }
  interface State { bricks: Map<number, Brick>; slots: Map<number, number>; phi: Uint32Array; vel: Uint32Array; solid: Uint32Array; topK: number }
  const STATIC = F_CLASS.NEAR_SOLID | F_CLASS.CUT_STATIC;
  const state = async (s: Solver): Promise<State> => {
    const e = s.brickEngine, L = e.layout, snap = await snapshot(e);
    const phi = await e.read("phi"), vel = await e.read("vel"), cut = new Float32Array((await e.read("cut")).buffer), solid = await e.read("solid");
    const flags = L.base("phi", "F", "FLAGS"), bricks = new Map<number, Brick>();
    const channels = ["FRAC", "AP0", "AP1", "AP2"].map(ch => L.base("cut", "K", ch));
    for (const [key, slot] of snap.bricks) {
      const r = L.dir.frec + 32 * slot, ks = snap.dir[r + REC.CUT]!;
      const open = new Uint8Array(512);
      for (let l = 0; l < 512; l++) open[l] = phi[flags + BRICK_STRIDE * slot + l]! & FLAG.OPEN;
      const brick: Brick = { cls: snap.dir[r + REC.CLASS]! & STATIC, open };
      if (ks !== NONE) { const c = new Float32Array(4 * 512); channels.forEach((b, i) => c.set(cut.subarray(b + BRICK_STRIDE * ks, b + BRICK_STRIDE * ks + 512), 512 * i)); brick.cut = c; }
      bricks.set(key, brick);
    }
    return { bricks, slots: snap.bricks, phi, vel, solid, topK: snap.header[H.TOP_K]! };
  };
  // Two states agree on every brick both hold: class bits, cut presence and contents, OPEN bits.
  const same = (a: State, b: State, label: string) => {
    let shared = 0;
    for (const [key, x] of a.bricks) {
      const y = b.bricks.get(key); if (!y) continue; shared++;
      const at = brickKeyCoord(key).join(",");
      assert.equal(y.cls, x.cls, `${label}: brick ${at} class`);
      assert.equal(y.cut === undefined, x.cut === undefined, `${label}: brick ${at} cut held`);
      if (x.cut) assert.deepEqual(Array.from(y.cut!), Array.from(x.cut), `${label}: brick ${at} cut`);
      assert.deepEqual(Array.from(y.open), Array.from(x.open), `${label}: brick ${at} OPEN`);
    }
    assert.ok(shared > 20, `${label}: ${shared} bricks held by both`);
    return shared;
  };
  // Hit testing: the bound image is the editor's SolidWorld, cell for cell.
  const hits = (s: State, scene: SceneDescription, label: string) => {
    const reader = voxelReaderForScene(scene);
    let bad = 0, solid = 0;
    for (let z = 0; z < N[2]; z++) for (let y = 0; y < N[1]; y++) for (let x = 0; x < N[0]; x++) {
      const want = reader.solid([x, y, z]), have = solidOccupancyMaskVoxel(s.solid, x, y, z);
      if (want) solid++;
      if (want !== have && bad++ < 3) t.diagnostic(`${label}: cell ${x},${y},${z} reader ${want} image ${have}`);
    }
    assert.equal(bad, 0, `${label}: the engine's image and the editor's reader disagree at ${bad} cells`);
    return solid;
  };
  // In the edit's reach: a brick within RING cells of the changed cells [lo, hi).
  const reach = (key: number, lo: V3, hi: V3) => { const o = brickKeyCoord(key).map(v => 8 * v); return o.every((v, a) => v + 8 > lo[a]! - RING && v < hi[a]! + RING); };
  // Local: the bricks out of reach keep every word the state reads; the velocity pool and every φ word but FLAGS too.
  const local = (a: State, b: State, lo: V3, hi: V3, label: string, phiBase: number, capF: number) => {
    assert.deepEqual(Array.from(b.slots), Array.from(a.slots), `${label}: residency unchanged`);
    assert.ok(Buffer.from(b.vel.buffer).equals(Buffer.from(a.vel.buffer)), `${label}: the velocity pool keeps every bit`);
    const keyOfSlot = new Map([...a.slots].map(([key, slot]) => [slot, key]));
    let differ = 0, touched = 0;
    for (let i = 0; i < a.phi.length; i++) {
      if (a.phi[i] === b.phi[i]) continue;
      const rel = i - phiBase, slot = Math.floor(rel / BRICK_STRIDE), key = keyOfSlot.get(slot);
      if (rel < 0 || rel % BRICK_STRIDE >= 512 || slot >= capF || key === undefined || !reach(key, lo, hi)) differ++;
    }
    assert.equal(differ, 0, `${label}: ${differ} φ-pool words changed outside the reach's FLAGS`);
    for (const [key, x] of a.bricks) {
      if (reach(key, lo, hi)) { touched++; continue; }
      const y = b.bricks.get(key)!;
      assert.equal(y.cls, x.cls, `${label}: an untouched brick's class`);
      assert.deepEqual(y.cut && Array.from(y.cut), x.cut && Array.from(x.cut), `${label}: an untouched brick's cut`);
    }
    return touched;
  };
  const voxel = (s: State) => (c: readonly number[]) => c.every((v, a) => v >= 0 && v < N[a]!) && solidOccupancyMaskVoxel(s.solid, c[0]!, c[1]!, c[2]!);
  // The run goes on: k steps, each latching nothing, with no live record in a closed cell.
  const run = async (s: Solver, from: number, k: number, label: string) => {
    const e = s.brickEngine, L = e.layout, P = L.cap.particles, S = L.particleStride, dt = e.options.dt_s;
    for (let step = from + 1; step <= from + k; step++) {
      assert.ok(s.advanceTo(step * dt, []), `${label}: step ${step} admitted`);
      await s.assertSimulationHealthy();
      const snap = await snapshot(e), solid = voxel({ solid: await e.read("solid") } as State);
      const parts = new Float32Array((await e.read("parts")).buffer), bank = snap.header[H.STEP]! & 1, count = Math.min(snap.header[bank ? H.PCOUNT1 : H.PCOUNT0]!, P);
      let inside = 0, live = 0;
      for (let i = 0; i < count; i++) {
        const r = S * (bank * P + i); if (!(parts[r + 3]! > 0)) continue;
        live++; if (solid([0, 1, 2].map(a => Math.floor(parts[r + a]!)))) inside++;
      }
      assert.equal(inside, 0, `${label}: step ${step}: ${inside} of ${live} live records in closed cells`);
      assert.equal(snap.header[H.FAIL], 0, `${label}: step ${step}: FAIL ${snap.header[H.FAIL]}`);
    }
  };

  await withUniformDevice("brick solid edit", async device => {
    const solver = await createBrickSolver(device, scene0, values, quality);
    let loaded: Solver | undefined;
    try {
      const e = solver.brickEngine, mask = solver.brickEngine.options.staticMask!, held0 = (mask as unknown as { slotsHeld: number }).slotsHeld;
      const phiBase = e.layout.base("phi", "F", "FLAGS");
      await run(solver, 0, 3, "before");
      const s0 = await state(solver);
      hits(s0, scene0, "base");

      // The fill: validated, taken, and nothing but the reach's static words change.
      solver.validateLiveSolidEdit(filled);
      assert.notEqual(solver.applySceneUniforms(filled), false, "the fill is taken live");
      const s1 = await state(solver);
      const solidCells = hits(s1, filled, "fill");
      assert.ok(Buffer.from(s1.solid.buffer).equals(Buffer.from(mask.words.buffer, mask.words.byteOffset, mask.words.byteLength)), "the GPU image is the host's");
      const touched = local(s0, s1, B.lo, B.hi, "fill", phiBase, e.layout.cap.f);
      // Every resident brick in the fill's reach is NEAR_SOLID, and every fill cell a resident brick holds is closed.
      const fillCells = (s: State, label: string) => {
        let closed = 0, near = 0;
        for (const [key, b] of s.bricks) {
          const o = brickKeyCoord(key).map(v => 8 * v);
          if (reach(key, B.lo, B.hi)) { assert.ok(b.cls & F_CLASS.NEAR_SOLID, `${label}: brick ${o} near the fill is NEAR_SOLID`); near++; }
          for (let l = 0; l < 512; l++) {
            const c = [o[0]! + (l & 7), o[1]! + ((l >> 3) & 7), o[2]! + (l >> 6)];
            if (!inBox(B, c)) continue;
            closed++;
            assert.equal(b.open[l], 0, `${label}: cell ${c} of the fill is closed`);
            assert.ok(b.cls & F_CLASS.CUT_STATIC, `${label}: brick ${o} holding the fill is CUT_STATIC`);
            assert.equal(b.cut![l], 0, `${label}: cell ${c} of the fill has FRAC 0`);
          }
        }
        return { closed, near };
      };
      // The fill lies in the column's deep interior, which the census holds 4h (C): the restamp reaches the F bricks
      // resident now; the next step's census promotes the rest (the static need rule reads the new mask).
      const f1 = fillCells(s1, "fill");
      t.diagnostic(`fill: ${solidCells} solid cells, ${touched} bricks in reach, ${f1.near} NEAR_SOLID, ${f1.closed} fill cells resident and closed, K free ${s0.topK} -> ${s1.topK}`);
      assert.ok(f1.near > 0, "the fill's reach holds resident bricks");

      // Undo, then redo: each state again, and the mask back to its slots.
      assert.notEqual(solver.applySceneUniforms(scene0), false, "the undo is taken live");
      const s2 = await state(solver);
      hits(s2, scene0, "undo");
      same(s0, s2, "undo");
      assert.ok(Buffer.from(s2.phi.buffer).equals(Buffer.from(s0.phi.buffer)), "undo: the φ pool keeps every bit of the base");
      assert.equal(s2.topK, s0.topK, "undo: every K slot the fill took is back");
      assert.equal((mask as unknown as { slotsHeld: number }).slotsHeld, held0, "undo: the mask's slots are back");
      assert.notEqual(solver.applySceneUniforms(filled), false, "the redo is taken live");
      const s3 = await state(solver);
      hits(s3, filled, "redo"); same(s1, s3, "redo");

      // The next step: its head's census promotes the cut tiles to h and solidStamp cuts them, before the step's
      // physics; every fill cell is then a resident, closed, FRAC 0 cell of a CUT_STATIC brick.
      await run(solver, 3, 1, "the step after the fill");
      const s3s = await state(solver), f3 = fillCells(s3s, "after a step");
      const volume = [0, 1, 2].reduce((v, a) => v * (B.hi[a]! - B.lo[a]!), 1);
      t.diagnostic(`after a step: ${f3.closed} of ${volume} fill cells resident and closed, ${f3.near} bricks NEAR_SOLID, K free ${s3s.topK}`);
      assert.equal(f3.closed, volume, "the census took every fill cell into F");

      // The clear: its cells open, and a brick it leaves uncut gives its slot back.
      assert.notEqual(solver.applySceneUniforms(cleared), false, "the clear is taken live");
      const s4 = await state(solver);
      hits(s4, cleared, "clear"); local(s3s, s4, C.lo, C.hi, "clear", phiBase, e.layout.cap.f);
      let freed = 0;
      for (const [key, b] of s4.bricks) {
        const k = brickKeyCoord(key);
        if (k[0] !== 1 || k[2] !== 1 || k[1]! > 1) continue;
        assert.equal(b.cls & F_CLASS.CUT_STATIC, 0, `brick ${k} the clear empties is no longer CUT_STATIC`);
        assert.equal(b.cut, undefined, `brick ${k} gave its K slot back`);
        assert.ok(b.open.every(v => v === FLAG.OPEN), `brick ${k}: every cell open`);
        if (s3s.bricks.get(key)?.cut) freed++;
      }
      assert.equal(s4.topK, s3s.topK + freed, `the clear gave back ${freed} slots`);
      assert.ok(freed > 0, "the clear emptied a cut brick");

      // The refill, on resident bricks: the restamp takes K slots and cuts them now, the post-step state again; and the
      // clear once more.
      assert.notEqual(solver.applySceneUniforms(filled), false, "the refill is taken live");
      const s5 = await state(solver), f5 = fillCells(s5, "refill");
      hits(s5, filled, "refill"); same(s3s, s5, "refill");
      assert.equal(f5.closed, volume, "the refill closes every fill cell at once");
      assert.equal(s5.topK, s3s.topK, "the refill takes back the slots the clear gave");
      assert.notEqual(solver.applySceneUniforms(cleared), false, "the second clear is taken live");
      same(s4, await state(solver), "the second clear");

      // Refusals: explicit, and nothing changes.
      const unknown = { ...cleared, solidVoxels: [...cleared.solidVoxels, { operation: "fill", kind: "no-such-kind", minimum: [30, 0, 30], maximumExclusive: [34, 4, 34] }] } as SceneDescription;
      assert.throws(() => solver.validateLiveSolidEdit(unknown), /Unknown solid patch kind/);
      const heavier = { ...cleared, fluid: { ...cleared.fluid, gravity_m_s2: { x: 0, y: -12, z: 0 } } };
      assert.throws(() => solver.validateLiveSolidEdit(heavier), /nothing else in the scene changes/);
      assert.equal(solver.applySceneUniforms(heavier), false, "a gravity change is the renderer's rebuild");
      same(s4, await state(solver), "after the refusals");

      // Save and load: the edited document builds the same image, and its stamp (solidStamp, at the first step's head)
      // is the restamp on every brick both hold.
      loaded = await createBrickSolver(device, JSON.parse(JSON.stringify(cleared)) as SceneDescription, values, quality);
      await run(loaded, 0, 1, "loaded");
      const l0 = await state(loaded);
      hits(l0, cleared, "loaded");
      t.diagnostic(`loaded: ${same(s4, l0, "the loaded document's stamp")} bricks agree with the restamp`);

      // The run goes on after the edits.
      await run(solver, 4, 10, "after the edits");
    } finally { solver.destroy(); loaded?.destroy(); }
  });
});

/**
 * The first voxel stroke on a scene built with none (the interactive scenes on auto): the engine runs the plain
 * schedule until it (BrickEngineOptions.staticsDeferred), and the stroke is a live edit like any other. Then it arms
 * the statics schedule, uploads the mask, and the next step's census and solidStamp close the fill. A stroke past the
 * K pool is refused at the preflight, and nothing changes.
 */
test("brick solid edit (GPU): the first stroke on a voxel-free dam arms the static stages live", GPU, async t => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  const { getMethod } = await import("../../core/method-registry");
  const { resolveMethodValues } = await import("../../core/method-contract");
  const { sceneDocument } = await import("../../core/scene-definition");
  const { getSceneDefinition } = await import("../../core/scenes");
  const { sceneWithSolidStroke } = await import("../../core/solid-world");
  const { solidOccupancyMaskVoxel } = await import("../../core/solid-occupancy-mask");
  const { voxelReaderForScene } = await import("../../core/voxel-editor/voxel-reader");
  const { BRICK_STRIDE, F_CLASS, FLAG, H, NONE, REC, brickKeyCoord } = await import("./brick-abi");
  const { snapshot } = await import("./brick-readback");
  const { createBrickSolver } = await import("./brick-method");
  const METHOD = "uniform-narrow-band-flip", method = getMethod(METHOD), definition = getSceneDefinition("minimal-power-dam-break-64");
  const scene0 = sceneDocument(definition);
  scene0.fluid.brickPools = { f: 729, c: 27, b: 16, k: 0, particles: 600000 };
  scene0.duration_s = 3;
  const profile = definition.methodProfile, quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" }) as Record<string, unknown>;
  const B = { lo: [8, 0, 8] as V3, hi: [20, 10, 20] as V3 }, N = [64, 64, 64];
  const filled = sceneWithSolidStroke(scene0, [{ operation: "fill", minimum: B.lo, maximumExclusive: B.hi, materialId: 2 }]);
  const huge = sceneWithSolidStroke(scene0, [{ operation: "fill", minimum: [0, 0, 0], maximumExclusive: [64, 40, 64], materialId: 2 }]);
  const inBox = (c: readonly number[]) => c.every((v, a) => v >= B.lo[a]! && v < B.hi[a]!);
  await withUniformDevice("brick first stroke", async device => {
    const solver = await createBrickSolver(device, scene0, values, quality);
    try {
      const e = solver.brickEngine, L = e.layout, dt = e.options.dt_s, P = L.cap.particles, S = L.particleStride;
      const run = async (from: number, k: number, label: string) => {
        for (let step = from + 1; step <= from + k; step++) {
          assert.ok(solver.advanceTo(step * dt, []), `${label}: step ${step} admitted`);
          await solver.assertSimulationHealthy();
          const snap = await snapshot(e), words = await e.read("solid");
          const parts = new Float32Array((await e.read("parts")).buffer), bank = snap.header[H.STEP]! & 1, count = Math.min(snap.header[bank ? H.PCOUNT1 : H.PCOUNT0]!, P);
          let inside = 0;
          for (let i = 0; i < count; i++) {
            const r = S * (bank * P + i); if (!(parts[r + 3]! > 0)) continue;
            const c = [0, 1, 2].map(a => Math.floor(parts[r + a]!));
            if (c.every((v, a) => v >= 0 && v < N[a]!) && solidOccupancyMaskVoxel(words, c[0]!, c[1]!, c[2]!)) inside++;
          }
          assert.equal(inside, 0, `${label}: step ${step}: ${inside} live records in closed cells`);
          assert.equal(snap.header[H.FAIL], 0, `${label}: step ${step}: FAIL`);
        }
      };
      assert.equal(e.options.staticsDeferred, true, "built deferred");
      assert.equal(e.staticsArmed, false);
      assert.equal(L.cap.k, 64, "K: the edit headroom");
      const plainLabels = new Set([...e.schedule.part1, ...e.schedule.part2].map(p => p.kernel.label));
      for (const label of ["solidStamp", "pSolid", "phiSolid", "setKRelease"]) assert.ok(!plainLabels.has(label), `plain: no ${label}`);
      await run(0, 3, "before");
      const vel0 = await e.read("vel");

      // Past the K pool: refused at the preflight, nothing changes.
      assert.throws(() => solver.validateLiveSolidEdit(huge), /K slots/);
      assert.equal(e.staticsArmed, false, "a refused stroke arms nothing");

      // The first stroke: live, armed.
      solver.validateLiveSolidEdit(filled);
      assert.notEqual(solver.applySceneUniforms(filled), false, "the first stroke is taken live");
      assert.equal(e.staticsArmed, true);
      const armedLabels = new Set([...e.schedule.part1, ...e.schedule.part2].map(p => p.kernel.label));
      for (const label of ["solidStamp", "pSolid", "phiSolid", "setKRelease"]) assert.ok(armedLabels.has(label), `armed: ${label}`);
      const words = await e.read("solid"), reader = voxelReaderForScene(filled);
      let bad = 0, solid = 0;
      for (let z = 0; z < 64; z++) for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
        const want = reader.solid([x, y, z]); if (want) solid++;
        if (want !== solidOccupancyMaskVoxel(words, x, y, z)) bad++;
      }
      assert.equal(bad, 0, "the engine's image is the editor's reader"); assert.equal(solid, 1440);
      assert.ok(Buffer.from((await e.read("vel")).buffer).equals(Buffer.from(vel0.buffer)), "the edit keeps every velocity bit");

      // The next step: every fill cell resident, closed, FRAC 0, in a CUT_STATIC brick.
      await run(3, 1, "the step after the stroke");
      const snap = await snapshot(e), phi = await e.read("phi"), cut = new Float32Array((await e.read("cut")).buffer);
      const flags = L.base("phi", "F", "FLAGS"), frac = L.base("cut", "K", "FRAC");
      let closed = 0;
      for (const [key, slot] of snap.bricks) {
        const o = brickKeyCoord(key).map(v => 8 * v), r = L.dir.frec + 32 * slot, ks = snap.dir[r + REC.CUT]!, cls = snap.dir[r + REC.CLASS]!;
        for (let l = 0; l < 512; l++) {
          const c = [o[0]! + (l & 7), o[1]! + ((l >> 3) & 7), o[2]! + (l >> 6)];
          if (!inBox(c)) continue;
          closed++;
          assert.equal(phi[flags + BRICK_STRIDE * slot + l]! & FLAG.OPEN, 0, `cell ${c} closed`);
          assert.ok(cls & F_CLASS.CUT_STATIC, `brick ${o} CUT_STATIC`); assert.notEqual(ks, NONE);
          assert.equal(cut[frac + BRICK_STRIDE * ks + l], 0, `cell ${c} FRAC 0`);
        }
      }
      t.diagnostic(`first stroke: ${closed} of 1440 fill cells resident and closed after a step, K free ${snap.header[H.TOP_K]} of ${L.cap.k}`);
      assert.equal(closed, 1440);
      await run(4, 10, "after the first stroke");
    } finally { solver.destroy(); }
  });
});
