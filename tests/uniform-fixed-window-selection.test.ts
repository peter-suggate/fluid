import assert from "node:assert/strict";
import test from "node:test";
import { uniformFixedWindowSelection } from "../lib/methods/uniform/uniform-detail-fields";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { UNIFORM_DIRECTORY_CHILD_WORDS, UNIFORM_DIRECTORY_FLAT_EDGE, UNIFORM_DIRECTORY_TOP_WORDS, UNIFORM_DIRECTORY_WORDS, UNIFORM_WINDOW_HEADER, uniformDirectoryFlat, uniformDirectoryWord, uniformWindowPageWord } from "../lib/methods/uniform/uniform-window-address";

// Closed tanks 62, 63 and 64 pages long on one axis: the widths either side
// of the flat window's edge (lattice pages -1..62) and past the narrow
// pool's 60. Each takes the path uniformDetailPipeline compiles, and pages
// -1, 62 and 63 of the long axis read what the two-level directory and the
// world box give (uniformWindowPageWord).
type Path = "flat" | "two-level";
/** The host twin of umPageTableWord and umPageWord, as a fixed-geometry module compiles them. */
function compiledPageWord(tail: ArrayLike<number>, flatWindow: boolean, last: number, page: readonly number[]): { path: Path; word: number } {
  const h = (i: number) => tail[UNIFORM_DIRECTORY_WORDS + i]! | 0, frame = [0, 1, 2].map(a => h(UNIFORM_WINDOW_HEADER.frame + a));
  const world = page.map((p, a) => p + frame[a]!), f = page.map(p => (p + 1) >>> 0), E = UNIFORM_DIRECTORY_FLAT_EDGE;
  const inFlat = f.every(v => v < E);
  let raw: number, path: Path;
  if (flatWindow || inFlat) {
    const g = f.map(v => Math.min(v, E - 1));
    raw = inFlat ? tail[UNIFORM_DIRECTORY_TOP_WORDS + UNIFORM_DIRECTORY_CHILD_WORDS + (g[0]! | (g[1]! << 6) | (g[2]! << 12))]! : 0; path = "flat";
  } else { raw = uniformDirectoryWord(tail, world[0]!, world[1]!, world[2]!); path = "two-level"; }
  const low = [0, 1, 2].map(a => h(UNIFORM_WINDOW_HEADER.solidLow + a)), high = [0, 1, 2].map(a => h(UNIFORM_WINDOW_HEADER.solidHigh + a));
  if (world.some((w, a) => w < low[a]! || w >= high[a]!)) return { path, word: 0xffffffff };
  const keep = page.every(p => (p >>> 0) <= last) && raw !== (h(UNIFORM_WINDOW_HEADER.stale) >>> 0);
  return { path, word: keep ? raw : 0 };
}

/** Every lattice page of `pages` reads, through the path the selection compiles, the directory's word of its world page under the world box. */
function readsAsDirectory(tail: Uint32Array, flatWindow: boolean, last: number, pages: readonly (readonly number[])[], extent: readonly number[]): void {
  const frame = [0, 1, 2].map(a => tail[UNIFORM_DIRECTORY_WORDS + UNIFORM_WINDOW_HEADER.frame + a]! | 0);
  for (const page of pages) {
    const { path, word } = compiledPageWord(tail, flatWindow, last, page), world = page.map((p, a) => p + frame[a]!);
    // The flat window holds lattice pages -1..62; past them a flat module reads AIR, then the world box.
    assert.equal(path, flatWindow || page.every(v => v >= -1 && v <= 62) ? "flat" : "two-level", `page ${page}`);
    assert.equal(word, uniformWindowPageWord(tail, world[0]!, world[1]!, world[2]!), `lattice page ${page}, world page ${world}`);
    const stored = world.every((v, a) => v >= 0 && v < extent[a]!);
    assert.equal(word !== 0 && word !== 0xffffffff, stored, `lattice page ${page} (world ${world}) is ${stored ? "a stored box page" : "off the box"}`);
  }
}
const across = (axis: number, along: readonly number[], others: readonly number[]) => along.flatMap(p => others.map(o => { const page = [o, o, o]; page[axis] = p; return page; }));

for (const axis of [0, 2]) for (const [pages, want] of [
  [62, { narrow: false, flat: true }],
  [63, { narrow: false, flat: true }],
  [64, { narrow: false, flat: false }],
] as const) {
  test(`a closed tank ${pages} pages long on axis ${axis} takes the ${want.flat ? "flat" : "two-level"} window and reads its edge pages as the directory holds them`, () => {
    const extent = [2, 2, 2]; extent[axis] = pages;
    const seed = uniformLatticeAddress(extent.map(n => 32 * n), false).seed;
    const selection = uniformFixedWindowSelection(seed);
    assert.deepEqual(selection, { frame: true, inside: true, ...want });
    // windowLast is compiled to all ones where the box lies inside the window.
    readsAsDirectory(seed.tail, want.flat, 0xffffffff, across(axis, [-1, 0, 61, 62, 63, 64], [-1, 0, 1, 2]), extent);
  });
}

// An open-top column (the seep scene's rest variant, --closed: the throat
// filled, the top open) never compiles its frame, and the frame moves: the
// commit places the resident pages in the middle of the inner window, so
// box page 0 can sit at lattice page 16 and the box's far pages past 63.
for (const [pages, narrow] of [[56, true], [60, true], [61, false], [63, false], [64, false]] as const) {
  test(`an open-top column ${pages} pages long reads every lattice page as the directory holds it, at a moved frame`, () => {
    const extent = [pages, 4, 1], seed = uniformLatticeAddress(extent.map(n => 32 * n), true).seed;
    assert.deepEqual(uniformFixedWindowSelection(seed), { frame: false, inside: false, narrow, flat: false });
    const tail = new Uint32Array(seed.tail), frame = [-16, -30, -31];
    tail.set(new Uint32Array(new Int32Array(frame).buffer), UNIFORM_DIRECTORY_WORDS + UNIFORM_WINDOW_HEADER.frame);
    uniformDirectoryFlat(tail, frame);
    const last = tail[UNIFORM_DIRECTORY_WORDS + UNIFORM_WINDOW_HEADER.windowLast]! >>> 0;
    // Lattice pages: the halo below the box, its first and last pages and the flat window's edge. A
    // narrow pool's commit keeps every stored page in lattice pages 2..61 (an edge fatal otherwise), so
    // its modules read the flat window alone and are held to it only there.
    const xs = (narrow ? [15, 16, 47, 61, 62] : [15, 16, 47, 62, 63, 64, 16 + pages - 1, 16 + pages]);
    readsAsDirectory(tail, narrow, last, xs.map(x => [x, 31, 31]), extent);
  });
}
