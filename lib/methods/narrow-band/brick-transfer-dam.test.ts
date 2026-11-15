// setTransfer (brick-transfer.wgsl.ts) on a frozen state from the old NB-FLIP
// engine itself: frame 22 of high-resolution-dam-break, its frame-head remap's
// pre and post state over the 64^3 window at the lattice's low corner
// (parity/dam-remap-22.{bin,json}, tools/nbflip-remap-capture.ts). The
// formulas are F's bitwise nbfRemap* ports; this pins the listing and the
// tile flow: which tiles are remapped, which faces and vertices each writes,
// and where it stores them, against the old engine's real output rather than
// a transcription of it (brick-transfer.test.ts).
//
// - The pre state goes in by brick-abi.md §2.5: old owners to CMASK (and the
//   F records' word 28), new owners to WANT, phi to F (h vertices) or C
//   (4h corners), U and US to F (h faces) or C (4h faces), walls by their
//   in-window cell. Every word that holds no state holds a poison value, so a
//   read of a non-owned copy shows.
// - φ is the old engine's (metres); the remap is linear in it.
// - The window's low planes are the lattice's walls. Its upper planes are
//   walls of the brick lattice but not of the old one, so a tile whose remap
//   stencil (the 3^3 tiles around it, which hold every slope tap) reaches the
//   last tile layer is left out: the layer itself, its vertices and the faces
//   its cells anchor.
// - The legitimate storage deviations (brick-abi.md, M2 without cuts) do not
//   enter the comparison, and nothing here encodes them away:
//   - there are no released bits or cut weights until solids land. The window
//     holds no solid cut, and the old engine's released bits live in the
//     velocity texel's w, which is not a face velocity, so they are not
//     compared;
//   - V = 0.5 into the exterior is the pressure face weight, which setTransfer
//     neither reads nor writes. The wall faces it stores (planes 0) are
//     compared like any other, bitwise.
//   Every window page is resident in both engines: the tool refuses AIR,
//   SOLID and BULK16 pages there.
// GPU only.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import test from "node:test";
import { ARGS, BRICK_STRIDE, CMASK_WORDS, H, LIST, REC } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { snapshot } from "./brick-readback";
import { bringUpStages } from "./brick-stages";
import { brickTransferSchedule } from "./brick-transfer.wgsl";
import { decodeFixture, ownerOf, storedFaces, storedVertices, type RemapFixtureMeta, type V3 } from "./parity/remap-fixture";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
const PARITY = resolve(import.meta.dirname, "parity"), NAME = "dam-remap-22";
const POISON_PHI = 4321.5, POISON_U = -987.25;
// U and US are bitwise (copies and F's bitwise nbfRemap* restrictions and
// prolongations). φ is bitwise but for a few last-ulp words, which TOL
// admits: see the note at the φ check.
const TOL = 2e-6;
const key = (b: readonly number[]) => ((b[0]! & 1023) | ((b[1]! & 1023) << 10) | ((b[2]! & 1023) << 20)) >>> 0;
const local8 = (c: readonly number[]) => (c[0]! & 7) | ((c[1]! & 7) << 3) | ((c[2]! & 7) << 6);
const tileIndex = local8;
const shr = (c: readonly number[], s: number): V3 => [c[0]! >> s, c[1]! >> s, c[2]! >> s];
const bits = (x: number) => new Uint32Array(new Float32Array([x]).buffer)[0]!;

test("setTransfer (GPU): the old NB-FLIP engine's own dam remap, listing, φ, U and US, storage and CMASK", GPU, async () => {
  const meta = JSON.parse(readFileSync(resolve(PARITY, `${NAME}.json`), "utf8")) as RemapFixtureMeta;
  const fx = decodeFixture(readFileSync(resolve(PARITY, `${NAME}.bin`)), meta);
  const W = meta.window, T = W / 4, hOld = ownerOf(fx.oldOwner, T), hNew = ownerOf(fx.newOwner, T);
  const pre = { verts: storedVertices(hOld, W), faces: storedFaces(hOld, W) }, post = { verts: storedVertices(hNew, W), faces: storedFaces(hNew, W) };
  assert.equal(pre.verts.length, fx.pre.phi.length); assert.equal(pre.faces.length, fx.pre.U.length);
  assert.equal(post.verts.length, fx.post.phi.length); assert.equal(post.faces.length, fx.post.U.length);
  // The comparison region: tiles whose 3^3 stencil stays inside the window.
  const tileOut = (t: readonly number[]) => t.some(v => v >= T - 1);
  const vertexOut = (p: V3) => p.some(v => v >= 4 * (T - 1));
  const faceOut = (c: V3, a: number) => { const anchor: V3 = [...c]; if (c[a] > 0) anchor[a]!--; return tileOut(shr(anchor, 2)); };
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("brick transfer dam", async device => {
    // Every brick and page of the window resident (boot liquid over all of it), every brick h at boot.
    const nb = W / 8 + 1;
    const e = await BrickEngine.create(device, { lattice: [W, W, W], h_m: 0.8 / 128, dt_s: 1 / 60, gravity_mps2: [0, -9.81, 0],
      capacities: { f: nb ** 3 + 8, c: 32, b: 0, k: 0, particles: 1024, t: 2048 }, liquid: [{ lo: [-1, -1, -1], hi: [W + 1, W + 1, W + 1] }], solver: L => bringUpCG(L), stages: bringUpStages });
    const s = await snapshot(e), L = e.layout;
    assert.equal(s.bricks.size, nb ** 3, "every brick of the window resident");
    assert.equal(s.pages.size, Math.ceil((W + 1) / 32) ** 3, "every page of the window resident");
    // ---- Masks: CMASK and word 28 the old owners, WANT the new.
    const mask = (h: (t: V3) => boolean, page: V3) => { const m = new Uint32Array(CMASK_WORDS); for (let i = 0; i < 512; i++) if (h([8 * page[0] + (i & 7), 8 * page[1] + ((i >> 3) & 7), 8 * page[2] + (i >> 6)])) m[i >> 5]! |= 1 << (i & 31); return m; };
    const pageOf = new Map<number, V3>();
    for (const [k, cs] of s.pages) {
      const page: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023]; pageOf.set(cs, page);
      e.write("dir", L.dir.cmask + CMASK_WORDS * cs, mask(hOld, page));
      e.write("set", L.set.want + CMASK_WORDS * cs, mask(hNew, page));
    }
    const brickBits = (h: (t: V3) => boolean, b: V3) => { let m = 0; for (let j = 0; j < 8; j++) if (h([2 * b[0] + (j & 1), 2 * b[1] + ((j >> 1) & 1), 2 * b[2] + (j >> 2)])) m |= 1 << j; return m; };
    const recs = await e.read("dir", L.dir.frec, 32 * L.cap.f);
    for (const [k, fs] of s.bricks) {
      const b: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023], r = 32 * fs + REC.MASK;
      e.write("dir", L.dir.frec + r, new Uint32Array([(recs[r]! & ~0xff) | brickBits(hOld, b)]));
    }
    // ---- Fields: each stored value at its storage (F at the cell or vertex, C at the tile), poison elsewhere.
    const fStore = new Map<string, Float32Array>(), cStore = new Map<string, Float32Array>();
    const store = (name: string, kind: "F" | "C", at: V3, value: number) => {
      if (kind === "F") { const fs = s.bricks.get(key(shr(at, 3))); assert.ok(fs !== undefined, `F storage for ${at}`); fStore.get(name)![BRICK_STRIDE * fs + local8(at)] = value; }
      else { const T4 = shr(at, 2), cs = s.pages.get(key(shr(T4, 3))); assert.ok(cs !== undefined, `C storage for ${at}`); cStore.get(name)![BRICK_STRIDE * cs + tileIndex(T4)] = value; }
    };
    const fields = ["PHI", "U0", "U1", "U2", "US0", "US1", "US2"];
    for (const n of fields) { const poison = n === "PHI" ? POISON_PHI : POISON_U; fStore.set(n, new Float32Array(BRICK_STRIDE * L.cap.f).fill(poison)); cStore.set(n, new Float32Array(BRICK_STRIDE * L.cap.c).fill(poison)); }
    pre.verts.forEach(({ p, kind }, i) => store("PHI", kind, p, fx.pre.phi[i]!));
    pre.faces.forEach(({ c, a, kind }, i) => { store(`U${a}`, kind, c, fx.pre.U[i]!); store(`US${a}`, kind, c, fx.pre.US[i]!); });
    for (const n of fields) {
      const pool = n === "PHI" ? "phi" : "vel";
      // 512 words a slot: the record's tail words are not the channel's.
      for (const fs of s.bricks.values()) e.write(pool, L.base(pool, "F", n) + BRICK_STRIDE * fs, fStore.get(n)!.slice(BRICK_STRIDE * fs, BRICK_STRIDE * fs + 512));
      for (const cs of s.pages.values()) e.write(pool, L.base(pool, "C", n) + BRICK_STRIDE * cs, cStore.get(n)!.slice(BRICK_STRIDE * cs, BRICK_STRIDE * cs + 512));
    }
    // ---- Run setTransfer.
    await e.run(brickTransferSchedule(L, k => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 }), LIST.C_ALL));
    const set = await e.read("set"), dir = await e.read("dir");
    assert.equal(set[H.FAIL], 0, `FAIL ${set[H.FAIL]} info ${set[H.FAIL_INFO]?.toString(16)}`);
    const report: Record<string, unknown> = { step: meta.step, window: W };

    // ---- Listing: TLIST against the old remap's worklist, on the compared tiles.
    const tlist = new Map<number, number>();
    for (let g = 0; g < set[H.N_TLIST_RUN]!; g++) {
      const word = set[L.set.tlist + 2 * g]!, cs = word >>> 9, i = word & 511, page = pageOf.get(cs)!;
      const t = [8 * page[0] + (i & 7), 8 * page[1] + ((i >> 3) & 7), 8 * page[2] + (i >> 6)];
      tlist.set(t[0]! + T * (t[1]! + T * t[2]!), set[L.set.tlist + 2 * g + 1]!);
    }
    const old = new Map<number, number>(); for (let i = 0; i < fx.listed.length; i += 2) old.set(fx.listed[i]!, fx.listed[i + 1]!);
    const coord = (i: number) => [i % T, Math.floor(i / T) % T, Math.floor(i / (T * T))];
    let excluded = 0, refine = 0, coarsen = 0, neighbour = 0;
    const onlyOld: string[] = [], onlyNew: string[] = [], flagDiff: string[] = [];
    for (const [i, f] of old) {
      if (tileOut(coord(i))) { excluded++; continue; }
      if ((f >>> 30) & 1) refine++; else if (f >>> 31) coarsen++; else neighbour++;
      const g = tlist.get(i);
      if (g === undefined) onlyOld.push(`${coord(i)} 0x${f.toString(16)}`); else if (g !== f) flagDiff.push(`${coord(i)} old 0x${f.toString(16)} new 0x${g.toString(16)}`);
    }
    for (const [i, g] of tlist) if (!tileOut(coord(i)) && !old.has(i)) onlyNew.push(`${coord(i)} 0x${g.toString(16)}`);
    const excludedNew = [...tlist.keys()].filter(i => tileOut(coord(i))).length;
    report.listing = { old: old.size, tlist: tlist.size, compared: old.size - excluded, refine, coarsen, neighbour, excluded: { old: excluded, tlist: excludedNew }, onlyOld: onlyOld.length, onlyTlist: onlyNew.length, flagDiff: flagDiff.length };

    // ---- CMASK = new owners; word 28 = new owners.
    let maskDiff = 0, recDiff = 0;
    for (const [cs, page] of pageOf) { const want = mask(hNew, page); for (let w = 0; w < CMASK_WORDS; w++) if (dir[L.dir.cmask + CMASK_WORDS * cs + w] !== want[w]) maskDiff++; }
    for (const [k, fs] of s.bricks) { const b: V3 = [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023]; if ((dir[L.dir.frec + 32 * fs + REC.MASK]! & 0xff) !== brickBits(hNew, b)) recDiff++; }
    report.cmask = { wordDiff: maskDiff, recordDiff: recDiff };

    // ---- Every stored word of the new layout against the old post state.
    const after = new Map<string, { f: Float32Array; c: Float32Array }>();
    for (const n of fields) {
      const pool = n === "PHI" ? "phi" : "vel";
      after.set(n, { f: new Float32Array((await e.read(pool, L.base(pool, "F", n), BRICK_STRIDE * L.cap.f)).buffer), c: new Float32Array((await e.read(pool, L.base(pool, "C", n), BRICK_STRIDE * L.cap.c)).buffer) });
    }
    const load = (name: string, kind: "F" | "C", at: V3) => {
      const a = after.get(name)!;
      if (kind === "F") return a.f[BRICK_STRIDE * s.bricks.get(key(shr(at, 3)))! + local8(at)]!;
      const T4 = shr(at, 2); return a.c[BRICK_STRIDE * s.pages.get(key(shr(T4, 3)))! + tileIndex(T4)]!;
    };
    type Tally = { compared: number; bitwise: number; mismatched: number; maxAbs: number; maxRel: number; poison: number; out: number; examples: string[]; rounded: string[] };
    const tallies = new Map<string, Tally>();
    const tally = (k: string) => { let t = tallies.get(k); if (!t) tallies.set(k, t = { compared: 0, bitwise: 0, mismatched: 0, maxAbs: 0, maxRel: 0, poison: 0, out: 0, examples: [], rounded: [] }); return t; };
    // Old pre values by location, for the diagnosis of a mismatch.
    const preAt = new Map<string, number>();
    pre.verts.forEach(({ p }, i) => preAt.set(`PHI|${p}`, fx.pre.phi[i]!));
    pre.faces.forEach(({ c, a }, i) => { preAt.set(`U${a}|${c}`, fx.pre.U[i]!); preAt.set(`US${a}|${c}`, fx.pre.US[i]!); });
    // The owners of the tiles around a location (old->new, h or 4h), for a diagnosis.
    const tileState = (at: V3) => { const t = shr(at, 2), w = (h: (t: V3) => boolean) => h(t) ? "h" : "4h"; return `tile ${t} ${w(hOld)}->${w(hNew)}`; };
    const check = (kindName: string, name: string, kind: "F" | "C", at: V3, want: number, out: boolean, where: string) => {
      const t = tally(kindName);
      if (out) { t.out++; return; }
      const got = load(name, kind, at), err = Math.abs(got - want);
      t.compared++;
      if (bits(got) === bits(want)) { t.bitwise++; return; }
      if (got === POISON_PHI || got === POISON_U) t.poison++;
      t.maxAbs = Math.max(t.maxAbs, err); t.maxRel = Math.max(t.maxRel, err / Math.max(Math.abs(want), 1e-6));
      const pre = preAt.get(`${name}|${at}`), line = `${where} ${kind} got ${got} want ${want} (pre ${pre ?? "not stored"}, ${tileState(at)})`;
      if (err > TOL * Math.max(1, Math.abs(want))) { t.mismatched++; if (t.examples.length < 6) t.examples.push(line); }
      else if (t.rounded.length < 6) t.rounded.push(line);
    };
    // φ: on the step-22 capture, 3 of 17317 F words differ in the last ulp
    // (rel 1.0e-7), all new h vertices of one refined tile (5,14,9), on its
    // x = 20 face where nothing was stored before. They are interpolated from
    // that face's coarse corners, where the two shaders' compilers may round
    // or contract (fma) the same sum differently. Where each word goes and
    // which input it reads match exactly; TOL admits only the rounding.
    post.verts.forEach(({ p, kind }, i) => check(`phi ${kind}`, "PHI", kind, p, fx.post.phi[i]!, vertexOut(p), `vertex ${p}`));
    for (const [ch, values] of [["U", fx.post.U], ["US", fx.post.US]] as const)
      post.faces.forEach(({ c, a, kind }, i) => check(`${ch} ${c[a] === 0 ? "wall" : kind}`, `${ch}${a}`, kind, c, values[i]!, faceOut(c, a), `face ${c} axis ${a}`));
    report.values = Object.fromEntries([...tallies].map(([k, t]) => [k, { ...t, examples: t.examples.length ? t.examples : undefined, rounded: t.rounded.length ? t.rounded : undefined }]));
    console.log(JSON.stringify(report, null, 1));
    if (onlyOld.length || onlyNew.length || flagDiff.length) console.log({ onlyOld: onlyOld.slice(0, 10), onlyTlist: onlyNew.slice(0, 10), flagDiff: flagDiff.slice(0, 10) });

    // ---- Verdicts.
    assert.ok(refine > 0 && coarsen > 0 && neighbour > 0, "the compared listing refines, coarsens and lists neighbours");
    assert.deepEqual({ onlyOld, onlyNew, flagDiff }, { onlyOld: [], onlyNew: [], flagDiff: [] }, "listed tiles and flags");
    assert.equal(maskDiff + recDiff, 0, "CMASK and word 28 = the new owners");
    for (const [k, t] of tallies) {
      assert.equal(t.poison, 0, `${k}: a non-owned copy was read or a stored word left unwritten`);
      assert.equal(t.mismatched, 0, `${k}: ${t.mismatched} of ${t.compared} words differ: ${t.examples.join("; ")}`);
    }
    e.destroy();
  });
});
