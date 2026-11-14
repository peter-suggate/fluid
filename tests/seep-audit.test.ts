import assert from "node:assert/strict";
import test from "node:test";
import { auditCells, throatOpen, type CellRecord } from "../tools/seep-audit-cells";
import type { SeepGeometry } from "../tools/seep-observables";

// The real seep's wall, throat and lattice; records placed by hand.
const g: SeepGeometry = { nx: 512, ny: 256, nz: 64, cell_m: 0.005, wall: { low: 240, high: 256 }, throat: [248, 64, 32], throatSide: 2, gravity_m_s2: 9.80665 };
const L = 1.25e-4;

test("audit splits the wall zone into open throat, solid and 4h owners", () => {
  assert.ok(throatOpen(g, 240, 64, 32) && throatOpen(g, 255, 65, 33));
  assert.ok(!throatOpen(g, 248, 66, 32) && !throatOpen(g, 256, 64, 32) && !throatOpen(g, 239, 64, 32));
  const records: CellRecord[] = [
    { x: 250, y: 64, z: 33, w: 1, liquid: 1, u: 2.6, open: true },
    { x: 250, y: 63, z: 33, w: 1, liquid: 0.5, u: 0, open: false },
    { x: 240, y: 60, z: 32, w: 4, liquid: 8, u: 0 },
    { x: 236, y: 60, z: 32, w: 4, liquid: 64, u: 0 }, // left of the wall: not counted
    { x: 251, y: 66, z: 33, w: 1, liquid: 0.25, u: 0, open: true },
  ];
  const a = auditCells(g, 0, records, 2.6);
  assert.ok(Math.abs(a.wall_L.open - L) < 1e-12 && Math.abs(a.wall_L.solid - 0.75 * L) < 1e-12 && Math.abs(a.wall_L.coarse - 8 * L) < 1e-12);
  assert.equal(a.openMismatch, 1);
});

test("audit jet slices: y >= 8 only, 4h owners spread over their columns, ballistic reference", () => {
  const records: CellRecord[] = [
    { x: 256, y: 64, z: 32, w: 1, liquid: 1, u: 2, open: true },
    { x: 257, y: 62, z: 32, w: 1, liquid: 1, u: 4, open: true },
    { x: 260, y: 2, z: 10, w: 1, liquid: 1, u: 0, open: true }, // floor band: all_L only
    { x: 264, y: 8, z: 32, w: 4, liquid: 32, u: 1 },          // slice 1, centre y 10
  ];
  const a = auditCells(g, 300, records, 2.6);
  const [s0, s1] = a.slices;
  assert.equal(a.slices.length, 24); assert.equal(s0!.x0, 256); assert.equal(a.slices.at(-1)!.x1, 448);
  assert.ok(Math.abs(s0!.jet_L - 2 * L) < 1e-12 && Math.abs(s0!.all_L - 3 * L) < 1e-12);
  assert.ok(Math.abs(s0!.meanY! - 63.5) < 1e-12 && Math.abs(s0!.meanU! - 3) < 1e-12);
  assert.ok(Math.abs(s1!.jet_L - 32 * L) < 1e-12 && s1!.meanY === 10);
  assert.ok(Math.abs(a.jetColumn_L - 34 * L) < 1e-12);
  // Ballistic: y0 = 65; at the first slice centre (x = 4 cells) the drop is g (0.02 m)^2 / (2 u0^2) / h.
  assert.ok(Math.abs(s0!.ballisticY - (65 - 9.80665 * 0.02 ** 2 / (2 * 2.6 ** 2) / 0.005)) < 1e-12);
  assert.ok(Math.abs(a.ballisticImpactX - (256 + 2.6 * Math.sqrt(2 * 57 * 0.005 / 9.80665) / 0.005)) < 1e-9);
  // Constant horizontal u: the column holds the throat's 2x2 section over 64 cells.
  assert.ok(Math.abs(a.ballisticColumn_L - 256 * L) < 1e-12);
  assert.equal(a.measuredImpactX, null);
});
