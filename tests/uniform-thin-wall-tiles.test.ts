import assert from "node:assert/strict";
import test from "node:test";
import { uniformThinWallTiles } from "../lib/methods/uniform/uniform-mixed-layout";
import { uniformBoxTileIndex, uniformLatticeBox } from "../lib/methods/uniform/uniform-lattice-address";

// Thin-wall tiles (docs/plans/bulk16-plan-2026-10-28.md, M4b): a cut 4h
// owner whose open cells, with its corners' open incident cells, lie on two
// sides of a solid cannot be one trilinear field. A 2-cell wall at x = 9..10
// (inside tile x 2) and an 8-cell wall at x = 20..27 (tiles x 5 and 6) in a
// 32³ lattice. Only the thin wall's tiles are marked, and only by the solids.
test("thin-wall tiles: a wall thinner than a tile, and not a thick one", () => {
  const dims = [32, 32, 32] as const, box = uniformLatticeBox(dims);
  const solid = (x: number) => (x >= 9 && x <= 10) || (x >= 20 && x <= 27);
  const coupled = new Set(Array.from({ length: box.boxTiles }, (_, t) => t));
  const thin = uniformThinWallTiles(dims, { solid: (x, _y, _z) => solid(x) }, coupled);
  const at = (tx: number) => thin.has(uniformBoxTileIndex(box, tx, 3, 3)) ? 1 : 0;
  // Tile 2 holds the wall (cells 8 and 11 open on its two sides).
  assert.equal(at(2), 1);
  // Tiles 1 and 3 are uncut (all 64 open): never thin-wall, though a corner of each is the shared plane x = 8 or 12.
  assert.equal(at(1), 0);assert.equal(at(3), 0);
  // The thick wall's cut tiles (cells 20..23 of tile 5 are solid: tile 5 is closed throughout; tile 6 likewise) and their
  // open neighbours see one side each.
  for (const tx of [4, 5, 6, 7]) assert.equal(at(tx), 0, `tile ${tx}`);
  // A wall at x = 8..9 on tile 2's low face: its open cells (10, 11) lie on one side, its low corners' open incident
  // cell (7, in tile 1) on the other: the corner tile 1 makes liquid fills tile 2's dry cells (the trough's shell).
  const face = uniformThinWallTiles(dims, { solid: (x, _y, _z) => x >= 8 && x <= 9 }, coupled);
  assert.equal((face.has(uniformBoxTileIndex(box, 2, 3, 3)) ? 1 : 0), 1);
  assert.equal((face.has(uniformBoxTileIndex(box, 1, 3, 3)) ? 1 : 0), 0);
  // A wall at x = 7..8 across tiles 1 and 2: each tile's open cells lie on one side, but their shared corners
  // (x = 8) are buried, one value for the open cells on both sides: both tiles are thin-wall (the dam's residual).
  const across = uniformThinWallTiles(dims, { solid: (x, _y, _z) => x >= 7 && x <= 8 }, coupled);
  assert.equal((across.has(uniformBoxTileIndex(box, 1, 3, 3)) ? 1 : 0), 1);
  assert.equal((across.has(uniformBoxTileIndex(box, 2, 3, 3)) ? 1 : 0), 1);
  assert.equal((across.has(uniformBoxTileIndex(box, 0, 3, 3)) ? 1 : 0), 0);
  assert.equal((across.has(uniformBoxTileIndex(box, 3, 3, 3)) ? 1 : 0), 0);
  // A wall at x = 7..11: the buried corner x = 8 has open cells on one side only (tile 2 is solid): none.
  const thick = uniformThinWallTiles(dims, { solid: (x, _y, _z) => x >= 7 && x <= 11 }, coupled);
  for (const tx of [0, 1, 2, 3, 4]) assert.equal((thick.has(uniformBoxTileIndex(box, tx, 3, 3)) ? 1 : 0), 0, `tile ${tx}`);
});
