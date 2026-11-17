/** Brick ABI invariants (docs/plans/nbflip-port/brick-abi.md): the layout's
 * regions are disjoint and aligned, channel bases follow §2.2, the window is a
 * power of two covering the lattice plus the skirt, and the WGSL prelude
 * carries the same numbers. */
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, BrickLayout, FAIL, FAIL_POOL, H, LIST, LIST_SLOTS, POOLS, RING_SLOTS, RING_WORDS, brickAbiWGSL, brickKey, brickKeyCoord, foldArgs, neighbourIndex, windowPages } from "./brick-abi";

const dam = () => new BrickLayout({ lattice: [128, 128, 128], f: 4608, c: 72, b: 0, k: 0, particles: 1 << 20 });

test("brick ABI: the window is a power of two an axis covering the lattice's pages and the skirt", () => {
  assert.deepEqual(windowPages([128, 128, 128]), [8, 8, 8]);
  assert.deepEqual(windowPages([512, 256, 64]), [32, 16, 4]);
  assert.deepEqual(windowPages([100, 20, 33]), [4, 1, 2]);
  assert.throws(() => windowPages([40000, 8, 8]), /over 1024/);
});

test("brick ABI: set and dir regions are disjoint, 64-word aligned, in order", () => {
  const L = dam(), s = L.set;
  const starts = [s.ring, s.freeF, s.freeC, s.freeB, s.freeK, s.markF, s.markC, s.cand, s.boundF, s.boundC, s.binStart, s.binCount, s.binCursor, s.pLiquid, ...s.lists];
  for (const at of starts) assert.equal(at % 64, 0);
  for (let i = 1; i < starts.length; i++) assert.ok(starts[i]! > starts[i - 1]!, `region ${i} after ${i - 1}`);
  assert.equal(s.ring, 256);
  assert.equal(s.freeF - s.ring, RING_SLOTS * RING_WORDS);
  assert.ok(s.words > s.lists[LIST_SLOTS - 1]!);
  assert.equal(s.listCap[LIST.F_ALL], 4608);
  assert.equal(s.listCap[LIST.C_ALL], 72);
  const d = L.dir;
  assert.equal(d.page, 0);
  assert.ok(d.child >= 512 && d.frec >= d.child + 64 * 72 && d.crec >= d.frec + 32 * 4608 && d.cmask >= d.crec + 32 * 72);
  // Header words stay inside the header.
  assert.ok(Object.values(H).every(w => w < 256));
  assert.ok(H.LIST_COUNT + LIST_SLOTS <= H.LANE_F);
});

test("brick ABI: channel-major bases, F then C, at stride 520", () => {
  const L = dam();
  assert.equal(L.base("vel", "F", "U0"), 0);
  assert.equal(L.base("vel", "F", "U1"), 4608 * BRICK_STRIDE);
  assert.equal(L.base("vel", "C", "U0"), POOLS.vel.length * 4608 * BRICK_STRIDE);
  assert.equal(L.base("vel", "C", "US2"), POOLS.vel.length * 4608 * BRICK_STRIDE + 8 * 72 * BRICK_STRIDE);
  assert.equal(L.poolWords.vel, POOLS.vel.length * BRICK_STRIDE * (4608 + 72));
  assert.throws(() => L.base("cut", "F", "FRAC"), /K-indexed/);
  assert.throws(() => L.base("phi", "F", "U0"), /no channel/);
});

test("brick ABI: keys, neighbour indices and the fold rule", () => {
  assert.deepEqual(brickKeyCoord(brickKey(15, 1023, 7)), [15, 1023, 7]);
  assert.equal(neighbourIndex(0, 0, 0), 13);
  assert.equal(neighbourIndex(-1, -1, -1), 0);
  assert.equal(neighbourIndex(1, 1, 1), 26);
  assert.deepEqual(foldArgs(0), [0, 1, 1]);
  assert.deepEqual(foldArgs(5), [5, 1, 1]);
  assert.deepEqual(foldArgs(70000), [32768, 3, 1]);
  assert.equal(ARGS.FC_G64 + 3 <= ARGS.LANE_P, true);
  assert.equal(FAIL_POOL, FAIL.POOL_F | FAIL.POOL_C | FAIL.POOL_B | FAIL.POOL_K);
});

test("brick ABI: the WGSL prelude carries the layout's numbers", () => {
  const L = dam(), w = brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "nbDir", atomic: false } });
  assert.match(w, new RegExp(`const NB_FCAP:u32=4608u;`));
  assert.match(w, new RegExp(`const NB_VEL_C_U0:u32=${L.base("vel", "C", "U0")}u;`));
  assert.match(w, new RegExp(`const NB_S_CAND:u32=${L.set.cand}u;`));
  assert.match(w, /fn nbFSlotOfCell/);
  assert.match(w, /fn nbFail/);
  assert.doesNotMatch(brickAbiWGSL(L, { set: { name: "nbSet", atomic: false } }), /fn nbFail/);
});

test("brick ABI: capacities are validated", () => {
  assert.throws(() => new BrickLayout({ lattice: [64, 64, 64], f: 0, c: 1, b: 0, k: 0, particles: 1 }), /capacity f/);
  assert.throws(() => new BrickLayout({ lattice: [64, 64, 64], f: 1.5, c: 1, b: 0, k: 0, particles: 1 }), /capacity f/);
});

test("brick ABI: the box hook (M3 padding): walls at the box, an unpadded text unchanged, my kernels resolved", async () => {
  const { BOX_TOKEN, boxPadded, boxWGSL, resolveBox } = await import("./brick-abi");
  const { brickSetKernels } = await import("./brick-set.wgsl");
  const { brickCensusKernels } = await import("./brick-census.wgsl");
  const { brickTransferKernels } = await import("./brick-transfer.wgsl");
  const L = dam(), P = new BrickLayout({ lattice: [32, 32, 32], box: [24, 16, 24], f: 125, c: 8, b: 0, k: 0, particles: 1024 });
  assert.deepEqual(L.box, L.lattice); assert.equal(boxPadded(L), false); assert.equal(boxWGSL(L), "NB_LATTICE");
  assert.deepEqual(P.box, [24, 16, 24]); assert.equal(boxPadded(P), true); assert.equal(boxWGSL(P), "NB_BOX");
  assert.throws(() => new BrickLayout({ lattice: [32, 32, 32], box: [24, 40, 24], f: 1, c: 1, b: 0, k: 0, particles: 1 }), /box 24x40x24 is not inside the lattice/);
  // Unpadded: no NB_BOX in the prelude, and the token resolves to the text it replaced.
  assert.ok(!brickAbiWGSL(L).includes("NB_BOX"));
  assert.equal(resolveBox(L, `${BOX_TOKEN}[a] nbInWallBox(c) nbTileInWallBox(t)`), "NB_LATTICE[a] nbInLattice(c) nbTileInLattice(t)");
  assert.match(brickAbiWGSL(P), /const NB_BOX:vec3i=vec3i\(24,16,24\);\nfn nbInBox\(c:vec3i\)->bool/);
  assert.equal(resolveBox(P, `${BOX_TOKEN}[a] nbInWallBox(c) nbTileInWallBox(t)`), "NB_BOX[a] nbInBox(c) nbTileInBox(t)");
  assert.match(brickAbiWGSL(P), /\nfn nbTileInBox\(t:vec3i\)->bool\{return all\(t>=vec3i\(0\)\)&&all\(4\*t<NB_BOX\);\}/);
  // My kernels on the padded layout: every token resolved; the owner walls, setInit's OPEN, the boot's inner faces,
  // the transfer's wall faces and the census's cell kind read the box.
  const kernels = ([...Object.values(brickSetKernels(P, { tiered: true })), ...Object.values(brickCensusKernels(P)), ...Object.values(brickTransferKernels(P))] as unknown[])
    .filter((k): k is { label: string; code: string } => typeof (k as { code?: unknown })?.code === "string");
  assert.ok(kernels.length > 10);
  for (const k of kernels) {
    assert.ok(!k.code.includes(BOX_TOKEN) && !k.code.includes("nbInWallBox("), `${k.label}: token unresolved`);
    assert.ok(!/OnWall\(c:vec3i,a:u32\)->bool\{return c\[a\]<=0\|\|c\[a\]>=NB_LATTICE\[a\];\}/.test(k.code), `${k.label}: an owner wall at the storage edge`);
  }
  const code = (label: string) => kernels.find(k => k.label === label)!.code;
  assert.match(code("setInit"), /NB_FLAG_OPEN,nbInBox\(o\+nbLocalCoord\(l\)\)\)/);
  assert.match(code("censusFine"), /fn ncCellKind\(c:vec3i\)->u32\{if\(!nbInBox\(c\)\)/);
  assert.match(code("setTransferFaces"), /let wall=c\[a\]\+1==NB_BOX\[a\];/);
  // The C level set (levelset-coarse, lc*): both owner instances and the kernels' own walls, holds and domains read
  // the box (the padded tank's creation guard named lcClear, lcAdvect, lcCouple, lcRetire and lcPublish); storage
  // (the whole-pages latch, the page lookups) stays at the lattice.
  const { levelSetCoarseKernels } = await import("./levelset-coarse.wgsl");
  const lc = Object.values(levelSetCoarseKernels(P)), lcUnpadded = Object.values(levelSetCoarseKernels(L));
  assert.equal(lc.length, 5);
  for (const k of lc) {
    assert.ok(!k.code.includes(BOX_TOKEN) && !k.code.includes("InWallBox("), `${k.label}: token unresolved`);
    assert.ok(!/OnWall\(c:vec3i,a:u32\)->bool\{return c\[a\]<=0\|\|c\[a\]>=NB_LATTICE\[a\];\}/.test(k.code), `${k.label}: an owner wall at the storage edge`);
    assert.match(k.code, /fn lcCheckLattice\(lid:u32\)\{if\(lid==0u&&any\(\(NB_LATTICE&vec3i\(31\)\)/, k.label);
    // In the C kernels' own text (not the lc1 owner's), only the lattice's storage reads remain at NB_LATTICE.
    const own = k.code.split("\n").filter((l: string) => /^fn lc(?!1)/.test(l) && /NB_LATTICE/.test(l)).map((l: string) => l.slice(0, l.indexOf("(")));
    assert.deepEqual(own, ["fn lcCheckLattice"], k.label);
  }
  const all = lc.map(k => k.code).join("\n");
  assert.match(all, /fn lcOnWall\(v:vec3i\)->bool\{return any\(v==vec3i\(0\)\)\|\|any\(v==NB_BOX\);\}/);
  assert.match(all, /fn lcTileLive\(t:vec3i\)->bool\{return nbTileInBox\(t\)/);
  assert.match(all, /fn lcClosed\(q:vec3i\)->bool\{for\(var k=0u;k<8u;k\+\+\)\{if\(nbInBox\(/);
  for (const k of lcUnpadded) assert.ok(!k.code.includes("NB_BOX") && !k.code.includes("nbTileInBox"), `${k.label}: unpadded names the box`);
});
