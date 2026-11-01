import assert from "node:assert/strict";
import test from "node:test";
import { uniformCensusCrossingTiles, uniformCensusInexactSolidTiles, uniformCensusSeedTiles, uniformCensusSurfacePages } from "../lib/methods/uniform/uniform-particle-census";
import { fluidLatticePagedCellVolumes, fluidLatticePagedVertexPhi } from "../lib/core/initial-liquid-lattice";
import { solidWorldForFluidLattice } from "../lib/core/solid-world";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { uniformSolidMaskPages } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import { uniformMixedSolidTiles } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformFineBlocks } from "../lib/methods/uniform/uniform-fine-slots";
import { validateScene } from "../lib/core/model";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";

// The census start's t=0 h tiles (docs/plans/bulk16-plan-2026-10-28.md, V2a):
// frame 1's census rule on the initial liquid by page, with no dense h field.

/** A 64³ lattice (two pages an axis) of vertex phi `phi` and the cells it holds liquid at (phi at
 * the cell's centre below zero, outside `solid`), with its crossing tiles. */
function lattice(phi: (x: number, y: number, z: number) => number, solid: (x: number, y: number, z: number) => boolean = () => false) {
  const edge = 32, at = new Map<number, Float32Array>(), vertices = new Map<number, Float32Array>();
  for (let pz = 0; pz < 2; pz++) for (let py = 0; py < 2; py++) for (let px = 0; px < 2; px++) {
    const page = new Float32Array(edge ** 3), v = new Float32Array((edge + 1) ** 3);
    for (let z = 0; z < edge; z++) for (let y = 0; y < edge; y++) for (let x = 0; x < edge; x++) {
      const X = edge * px + x, Y = edge * py + y, Z = edge * pz + z;
      if (!solid(X, Y, Z) && phi(X + 0.5, Y + 0.5, Z + 0.5) < 0) page[x + edge * (y + edge * z)] = 1;
    }
    for (let z = 0; z <= edge; z++) for (let y = 0; y <= edge; y++) for (let x = 0; x <= edge; x++)
      v[x + (edge + 1) * (y + (edge + 1) * z)] = phi(edge * px + x, edge * py + y, edge * pz + z);
    const i = px + 2 * (py + 2 * pz);
    if (page.some(c => c > 0)) at.set(i, page);
    vertices.set(i, v);
  }
  const volume = { edge, cells: [64, 64, 64] as const, pages: [2, 2, 2] as const, at };
  const crossing = uniformCensusCrossingTiles({ ...volume, at: vertices }, [T, T, T], (x, y, z) => !solid(x, y, z));
  return { volume, crossing };
}
const T = 16, key = (x: number, y: number, z: number) => x + T * (y + T * z);
/** A set of the box's tiles as a byte a tile (this small lattice's), and back. */
const dense = (tiles: ReadonlySet<number>) => { const out = new Uint8Array(T ** 3); for (const k of tiles) out[k] = 1; return out; };
const sparse = (tiles: Uint8Array) => { const out = new Set<number>(); tiles.forEach((v, k) => { if (v) out.add(k); }); return out; };
const layers = (seed: Uint8Array | ReadonlySet<number>) => { const out = new Set<number>(); for (const k of seed instanceof Uint8Array ? sparse(seed) : seed) out.add(Math.floor(k / T) % T); return [...out].sort((a, b) => a - b); };
const seedOf = (l: ReturnType<typeof lattice>, solid?: Uint8Array, bodies: Parameters<typeof uniformCensusSeedTiles>[4] = []) => dense(uniformCensusSeedTiles(l.volume, l.crossing, [T, T, T], solid && sparse(solid), bodies));
const flat = (_x: number, y: number) => y - 30;

test("a census start holds the free surface's crossing tiles and their closure at h, and the deep water and air at 4h", () => {
  // Liquid below y = 30: vertex phi changes sign on the edges from y = 29 to 30, in tile layer 7; the closure adds 6 and 8.
  const l = lattice(flat), seed = seedOf(l);
  assert.deepEqual(layers(l.crossing), [7]);
  assert.deepEqual(layers(seed), [6, 7, 8]);
  assert.equal(seed.reduce((n, s) => n + s, 0), 3 * T * T);
  // The walls are past the lattice: no crossing along them, so the deep water is 4h to the walls.
  assert.equal(seed[key(0, 0, 0)], 0);
  assert.equal(seed[key(0, 7, 0)], 1);
});

test("a census start holds a solid contact tile, and its stencil, at h only beside liquid", () => {
  const solid = new Uint8Array(T ** 3);
  solid[key(8, 2, 8)] = 1;   // under the water
  solid[key(8, 14, 8)] = 1;  // in the air, far from it
  const seed = seedOf(lattice(flat), solid);
  assert.equal(seed[key(8, 2, 8)], 1);
  assert.equal(seed[key(9, 3, 9)], 1, "solidPromote's stencil around the contact tile");
  assert.equal(seed[key(10, 2, 8)], 1, "the closure of the stencil");
  assert.equal(seed[key(11, 2, 8)], 0, "three tiles from it, deep: 4h");
  assert.equal(seed[key(8, 14, 8)], 0);
});

test("a census start's solid contact is the coupled tiles 4h cannot hold exactly", () => {
  // A wall across x = 32..35 lies on tile 8 exactly: every tile about it is one kind of cell, so 4h
  // holds it (umSolidGeometryExact) and none is a contact tile. Moved a cell, to x = 33..36, it cuts
  // tiles 8 and 9, which are.
  const coupledOf = (wall: (x: number) => boolean) => {
    const c = new Uint8Array(T ** 3);
    for (let z = 0; z < T; z++) for (let y = 0; y < T; y++) for (let x = 0; x < T; x++) {
      let near = false;
      for (let cx = 4 * x - 1; cx <= 4 * x + 4 && !near; cx++) if (wall(cx)) near = true;
      if (near) c[key(x, y, z)] = 1;
    }
    return c;
  };
  const inside = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < 4 * T && y < 4 * T && z < 4 * T;
  const aligned = (x: number) => x >= 32 && x < 36, moved = (x: number) => x >= 33 && x < 37;
  const exact = dense(uniformCensusInexactSolidTiles([T, T, T], sparse(coupledOf(aligned)), (x, y, z) => inside(x, y, z) && !aligned(x)));
  assert.equal(exact.reduce((n, s) => n + s, 0), 0, "a tile-aligned wall: every coupled tile 4h-exact");
  const cut = dense(uniformCensusInexactSolidTiles([T, T, T], sparse(coupledOf(moved)), (x, y, z) => inside(x, y, z) && !moved(x)));
  assert.equal(cut[key(8, 2, 8)], 1);
  assert.equal(cut[key(9, 2, 8)], 1);
  assert.equal(cut.reduce((n, s) => n + s, 0), 2 * T * T, "tiles 8 and 9 in every row");
});

test("a census start holds a body's ball at h where it meets liquid", () => {
  // A ball of radius 6 cells centred in the deep water at (32, 12, 32): tiles 6..9 across, 1..4 up.
  const seed = seedOf(lattice(flat), undefined, [{ centre: [32, 12, 32], radius: 6 }]);
  assert.equal(seed[key(8, 3, 8)], 1);
  assert.equal(seed[key(7, 2, 7)], 1);
  assert.equal(seed[key(2, 2, 2)], 0);
  // A ball in the air, two tiles past the closure of the surface, holds nothing.
  const dry = seedOf(lattice(flat), undefined, [{ centre: [32, 56, 32], radius: 4 }]);
  assert.deepEqual(layers(dry), [6, 7, 8]);
});

test("a census start keeps an uncut wall's wet face under the water at 4h, and its contact line at h", () => {
  // A tile-aligned solid wall across x = 32..35 (tile 8), full height; liquid either side below y = 30.
  // Frame 1's census holds the closed reservoir's 54 h tiles at every depth (docs/plans/bulk16-plan-2026-10-28.md,
  // V2a gates): a solid is not air, so the wall's wet face holds no crossing (umCensusEdgeOpen). Vertex phi
  // inside the wall is the buried vertices' far value, as fluidLatticePagedVertexPhi gives it.
  const wall = (x: number) => x >= 32 && x < 36, buried = (x: number) => x > 32 && x < 36;
  const l = lattice((x, y) => buried(x) ? 64 : y - 30, (x) => wall(x)), seed = seedOf(l);
  assert.equal(l.crossing.has(key(8, 2, 8)), false, "the wall's own tile: no open edge changes sign");
  assert.equal(seed[key(7, 2, 8)], 0, "the wall's wet face, deep: 4h");
  assert.equal(seed[key(9, 2, 8)], 0, "the far face, deep: 4h");
  assert.equal(seed[key(7, 7, 8)], 1, "the contact line at the free surface: h");
  assert.deepEqual(layers(seed), [6, 7, 8], "only the free surface and its closure");
});

test("a census start's h tiles on the closed reservoir grow with its surface area and not with its depth", () => {
  // The closed reservoir (10 mm) at heightScale 1 and 2: the free surface is the same, the middle wall's
  // wet face twice as tall. Frame 1's census holds the same h tiles at every depth; so does the seed,
  // through the solver's own solid path: the wall lies on tiles, which 4h holds exactly.
  // At widthScale 4 and depthScale 2 the surface is wider and deeper (in z): its crossing tiles are the
  // surface's own tile columns, and the seed grows no faster than they do.
  const count = (scale: { heightScale?: number; widthScale?: number; depthScale?: number }) => {
    const scene = createReservoirSeepScene(0.01, scale);
    scene.solidVoxels = scene.solidVoxels!.filter(v => v.operation === "fill");
    scene.fluid.initialLiquidVolumes = scene.fluid.initialLiquidVolumes!.slice(0, 1);
    const { volume } = fluidLatticePagedCellVolumes(scene, 32), solids = solidWorldForFluidLattice(scene);
    const tiles = volume.cells.map(n => Math.ceil(n / 4)) as [number, number, number];
    const phi = fluidLatticePagedVertexPhi(scene, 32, uniformCensusSurfacePages(volume));
    // The solver's solid path (censusSeed): the solid mask, its coupled tiles, the inexact ones.
    const mask = new SolidOccupancyMask(uniformSolidMaskPages(uniformLatticeAddress(volume.cells)));
    mask.update(solids);
    const [nx, ny, nz] = volume.cells, open = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && !mask.solid(x, y, z);
    const solid = uniformCensusInexactSolidTiles(tiles, uniformMixedSolidTiles(volume.cells as [number, number, number], mask).coupled, open);
    const crossing = uniformCensusCrossingTiles(phi, tiles, open);
    // The surface's tile columns: a tile column (x, z) with liquid in any cell under it.
    const [px, py] = volume.pages, E = volume.edge, wet = new Set<number>();
    volume.at.forEach((page, i) => {
      const ox = E * (i % px), oz = E * Math.floor(i / (px * py));
      for (let z = 0, j = 0; z < E; z++) for (let y = 0; y < E; y++) for (let x = 0; x < E; x++, j++) if (page[j]! > 0) wet.add(((ox + x) >> 2) + tiles[0] * ((oz + z) >> 2));
    });
    return { area: wet.size, crossing: crossing.size, seed: uniformCensusSeedTiles(volume, crossing, tiles, solid).size };
  };
  const shallow = count({ heightScale: 1 }), deep = count({ heightScale: 2 }), wide = count({ widthScale: 4, depthScale: 2 });
  assert.ok(shallow.seed > 0);
  assert.equal(deep.seed, shallow.seed, `seed h tiles: ${shallow.seed} at depth 1, ${deep.seed} at depth 2`);
  for (const c of [shallow, wide]) assert.equal(c.crossing, c.area, `crossing tiles ${c.crossing} against the surface's ${c.area} tile columns`);
  assert.ok(wide.area > 6 * shallow.area, `surface tile columns: ${shallow.area}, wide ${wide.area}`);
  assert.ok(wide.seed / wide.area <= shallow.seed / shallow.area, `seed per surface tile: ${shallow.seed / shallow.area}, wide ${wide.seed / wide.area}`);
});

test("a census start refuses a lattice that is not the box's tiles", () => {
  const l = lattice(() => 1);
  assert.throws(() => uniformCensusSeedTiles(l.volume, l.crossing, [T, T, 8]), /not the box's tiles/);
});

test("fluid.tiers.start \"census\" is valid and compacts the h stores at \"auto\"", () => {
  const scene = createReservoirSeepScene(0.02);
  scene.fluid.tiers = { ...scene.fluid.tiers, start: "census" };
  assert.deepEqual(validateScene(scene).filter(e => /tiers/.test(e)), []);
  assert.equal(uniformFineBlocks({ start: "census" }), "auto");
  assert.equal(uniformFineBlocks({ start: "census", fineBlocks: 40 }), 40);
});
