/**
 * The static mask's exact 4h tiles (brick-static-tiles.ts, 2026-11-19). CPU only: the bits against a cell-by-cell
 * oracle on an aligned and a misaligned block, a live edit's recompute against a full one (and against a save/load
 * of the edited scene), and the kernel text: a closed tile is the box rule's solid only when the layout binds the tiles,
 * and without them the text is unchanged.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { solidVoxelsOnMovedLattice } from "../../core/scene-lattice";
import { sceneWithSolidStroke, solidWorldForFluidLattice } from "../../core/solid-world";
import { solidOccupancyMaskVoxel } from "../../core/solid-occupancy-mask";
import type { SceneDescription } from "../../core/model";
import { BrickLayout, brickAbiWGSL, resolveBox } from "./brick-abi";
import { brickStaticMask, brickStaticMaskWalls } from "./brick-static.wgsl";
import { brickStaticNeedKernels } from "./brick-static-need.wgsl";
import { brickStaticTiles, brickStaticTilesEdit, type BrickStaticTiles } from "./brick-static-tiles";

type V3 = [number, number, number];
const LATTICE: V3 = [32, 64, 32];
// An aligned block (whole tiles: 2 x 2 x 2 closed) and a misaligned one (y 19..22: no closed tile, inexact both sides).
const ALIGNED = { lo: [8, 40, 8] as V3, hi: [16, 48, 16] as V3 }, MISALIGNED = { lo: [8, 19, 8] as V3, hi: [16, 23, 16] as V3 };

function tank(blocks: { lo: V3; hi: V3 }[]): SceneDescription {
  const definition = getSceneDefinition("minimal-power-dam-break-32"), base = structuredClone(sceneDocument(definition)), scene = structuredClone(base);
  const h = 0.025;
  scene.container = { ...scene.container, width_m: 32 * h, height_m: 64 * h, depth_m: 32 * h, fillFraction: 0 };
  scene.voxelDomain.finestCellSize_m = h; scene.solidVoxels = solidVoxelsOnMovedLattice(base, scene);
  scene.solidVoxels = [...scene.solidVoxels, ...blocks.map(b => ({ operation: "fill" as const, minimum: b.lo, maximumExclusive: b.hi }))];
  return scene;
}

/** The rule cell by cell, no page skip: the module's doc as written. */
function oracle(words: ArrayLike<number>, box: V3) {
  const [TX, TY, TZ] = LATTICE.map(n => n / 4) as V3;
  const inBox = (c: V3) => c.every((v, a) => v >= 0 && v < box[a]!);
  const voxel = (c: V3) => inBox(c) && solidOccupancyMaskVoxel(words, ...c);
  const shut = (c: V3) => !inBox(c) || solidOccupancyMaskVoxel(words, ...c);
  const cells = (t: V3) => Array.from({ length: 64 }, (_, k) => [4 * t[0] + (k & 3), 4 * t[1] + ((k >> 2) & 3), 4 * t[2] + (k >> 4)] as V3);
  const tileIn = (t: V3) => t.every((v, a) => v >= 0 && 4 * v + 4 <= box[a]!);
  const closed = (t: V3) => !tileIn(t) || cells(t).every(voxel);
  const exact = (t: V3) => closed(t) || (cells(t).every(c => !shut(c)) && [0, 1, 2, 3, 4, 5].every(f => {
    const a = f >> 1, s = f & 1, n = [...t] as V3; n[a] += s ? 1 : -1;
    if (closed(n)) return true;
    return cells(t).filter(c => c[a] === 4 * t[a] + (s ? 3 : 0)).every(c => { const d = [...c] as V3; d[a] += s ? 1 : -1; return !shut(d); });
  }));
  const near = (t: V3) => { for (let l = 0; l < 216; l++) if (voxel([4 * t[0] - 1 + (l % 6), 4 * t[1] - 1 + (Math.floor(l / 6) % 6), 4 * t[2] - 1 + Math.floor(l / 36)])) return true; return false; };
  const out = { closed: new Set<string>(), inexact: new Set<string>() };
  for (let z = 0; z < TZ; z++) for (let y = 0; y < TY; y++) for (let x = 0; x < TX; x++) {
    const t: V3 = [x, y, z];
    if (closed(t)) out.closed.add(t.join());
    else if (near(t) && !exact(t)) out.inexact.add(t.join());
  }
  return out;
}

function sets(t: BrickStaticTiles) {
  const [TX, TY] = t.tiles, out = { closed: new Set<string>(), inexact: new Set<string>() };
  for (let i = 0; i < t.tiles[0] * t.tiles[1] * t.tiles[2]; i++) {
    const key = [i % TX, Math.floor(i / TX) % TY, Math.floor(i / (TX * TY))].join();
    if ((t.bits[i >>> 5]! >>> (i & 31)) & 1) out.closed.add(key);
    if ((t.bits[t.words + (i >>> 5)]! >>> (i & 31)) & 1) out.inexact.add(key);
  }
  return out;
}

test("static tiles: the bits are the rule, cell by cell, on an aligned and a misaligned block", () => {
  const scene = tank([ALIGNED, MISALIGNED]), mask = brickStaticMask(scene)!;
  const t = brickStaticTiles(LATTICE, LATTICE, mask.words), got = sets(t), want = oracle(mask.words, LATTICE);
  assert.deepEqual([...got.closed].sort(), [...want.closed].sort());
  assert.deepEqual([...got.inexact].sort(), [...want.inexact].sort());
  assert.equal(t.closed, got.closed.size);
  // The aligned block: its 8 tiles closed, and every tile about it exact (its faces whole or walls): none promoted.
  for (const x of [2, 3]) for (const y of [10, 11]) for (const z of [2, 3]) assert.ok(got.closed.has([x, y, z].join()), `closed ${x},${y},${z}`);
  for (const k of got.inexact) assert.ok(Number(k.split(",")[1]) < 8, `only the misaligned block's tiles are inexact (${k})`);
  // The misaligned block: its two tile layers (y 16..19, 20..23) and no closed tile among them.
  for (const y of [4, 5]) assert.ok(got.inexact.has([2, y, 2].join()) && !got.closed.has([2, y, 2].join()), `inexact 2,${y},2`);
  assert.ok(!got.inexact.has("2,6,2") && !got.inexact.has("2,3,2"), "a tile with no voxel in its shell is not support");
});

test("static tiles: a full page is closed without a cell read, and agrees with the rule at its rim", () => {
  // The upper page (y 32..63) solid, a misaligned shelf below it: the full page's tiles, its rim and the shelf.
  const scene = tank([{ lo: [0, 32, 0], hi: [32, 64, 32] }, { lo: [0, 29, 0], hi: [10, 32, 32] }]), mask = brickStaticMask(scene)!;
  const t = brickStaticTiles(LATTICE, LATTICE, mask.words), got = sets(t), want = oracle(mask.words, LATTICE);
  assert.deepEqual([...got.closed].sort(), [...want.closed].sort());
  assert.deepEqual([...got.inexact].sort(), [...want.inexact].sort());
  assert.ok(got.closed.size >= 8 * 8 * 8, "the page's 512 tiles");
});

test("static tiles: a live edit's recompute is the full recompute, and a saved and loaded edit builds the same bits", () => {
  const scene = tank([ALIGNED]), live = brickStaticMask(scene)!, t = brickStaticTiles(LATTICE, LATTICE, live.words);
  // Open a cell of the closed block (a closed tile opens, its neighbours turn inexact) and fill an aligned tile.
  const edited = sceneWithSolidStroke(scene, [
    { operation: "clear", minimum: [9, 45, 9], maximumExclusive: [10, 46, 10] },
    { operation: "fill", minimum: [20, 0, 20], maximumExclusive: [24, 4, 24] },
  ]);
  assert.deepEqual(edited.solidVoxels.slice(0, scene.solidVoxels.length), scene.solidVoxels, "the edit keeps the authored source");
  const lo: V3 = [Infinity, Infinity, Infinity], hi: V3 = [-Infinity, -Infinity, -Infinity];
  const ranges = live.update(solidWorldForFluidLattice(edited), (x, y, z) => {
    lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
    hi[0] = Math.max(hi[0], x + 1); hi[1] = Math.max(hi[1], y + 1); hi[2] = Math.max(hi[2], z + 1);
  });
  assert.ok(ranges && ranges.length > 0);
  const before = sets(t), touched = brickStaticTilesEdit(t, LATTICE, live.words, lo, hi), after = sets(t);
  const full = brickStaticTiles(LATTICE, LATTICE, brickStaticMask(edited)!.words);
  assert.deepEqual([...t.bits], [...full.bits], "the edit's tiles equal a rebuild's");
  assert.equal(t.closed, full.closed);
  assert.ok(touched.last >= touched.first);
  assert.ok(before.closed.has("2,11,2") && !after.closed.has("2,11,2"), "the opened tile is no longer closed");
  assert.ok(after.inexact.has("2,11,2"), "and is support");
  assert.ok(after.closed.has("5,0,5"), "the filled tile is closed");
  const loaded = JSON.parse(JSON.stringify(edited)) as SceneDescription;
  assert.deepEqual([...brickStaticTiles(LATTICE, LATTICE, brickStaticMask(loaded)!.words).bits], [...full.bits], "save/load");
  // Undo: the base scene again.
  live.update(solidWorldForFluidLattice(scene), () => {});
  brickStaticTilesEdit(t, LATTICE, live.words, lo, hi);
  assert.deepEqual([...t.bits], [...brickStaticTiles(LATTICE, LATTICE, brickStaticMask(scene)!.words).bits], "undo");
});

test("static tiles: the 4h tier reads a closed tile only when the layout binds the tiles; otherwise the text is unchanged", () => {
  const cap = { lattice: LATTICE, f: 64, c: 8, b: 16, k: 8, particles: 1024 };
  const plain = new BrickLayout(cap), tiled = new BrickLayout({ ...cap, staticTiles: true });
  const code = "nbInWallBox(c) nbInWallC(c) nbTileInWallBox(t)";
  assert.equal(resolveBox(plain, code), "nbInLattice(c) nbInLattice(c) nbTileInLattice(t)", "without the tiles: the box rule");
  assert.equal(resolveBox(tiled, code), "nbInLattice(c) nbInWallS(c) nbTileInWallS(t)", "the 4h tests read a closed tile; the F cell test keeps the box");
  assert.equal(plain.dir.words, tiled.dir.words - 2 * Math.ceil((8 * 16 * 8) / 32), "two bit planes a lattice tile");
  const dir = { dir: { name: "dir", atomic: false } };
  assert.doesNotMatch(brickAbiWGSL(plain, dir), /nbTileClosed|NB_D_CLOSED/);
  assert.match(brickAbiWGSL(tiled, dir), /fn nbTileInexact\(t:vec3i\)->bool/);
  const walls = brickStaticMaskWalls(LATTICE, false);
  assert.match(brickStaticNeedKernels(plain, walls).censusStaticActive.code, /for\(var l=0u;l<216u;l\+\+\)/, "without the tiles: the 216-cell scan");
  const need = brickStaticNeedKernels(tiled, walls).censusStaticActive.code;
  assert.match(need, /return nbTileInexact\(T\);/); assert.doesNotMatch(need, /l<216u/);
});
