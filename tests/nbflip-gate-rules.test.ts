/** The NB-FLIP port referee's amended rules (plan §4.5, tools/nbflip-gate-rules.ts) against hand values and a
 * Float64 oracle: one-sided drift, the restored band (scalar and profile), and restoreBox against the observer's own
 * oracle (tools/nbflip-observables.ts boxOracle) on pools whose depth is raised by exactly the lost volume. */
import { test } from "node:test";
import assert from "node:assert/strict";
import { bandDistanceGate, bandScalarGate, binDepths, driftGate, driftMeanGate, l1, restoreBox, scalarGate, upperGate } from "../tools/nbflip-gate-rules";
import { boxObservation, boxOracle, type BoxGeometry } from "../tools/nbflip-observables";

const close = (a: number, b: number, eps = 1e-9, what = "") => assert.ok(Math.abs(a - b) <= eps * Math.max(1, Math.abs(b)), `${what}: ${a} vs ${b}`);
const allClose = (a: readonly number[], b: readonly number[], eps = 1e-9, what = "") => { assert.equal(a.length, b.length, what); a.forEach((v, i) => close(v, b[i]!, eps, `${what}[${i}]`)); };

test("an upper bound passes any count under the references and fails c - ref > max(floor, 3S)", () => {
  // refs 100 and 104: mean 102, S 4, floor 10% = 10.2 over 3S = 12, so c <= 114.
  const gate = (c: number) => upperGate("D8", "", { rel: 0.10 }, "", [{ at: "16", refs: [100, 104], candidate: c }]);
  for (const c of [0, 50, 102, 113.9]) assert.ok(gate(c).pass, `${c} passes`);
  for (const c of [114.1, 200]) assert.ok(!gate(c).pass, `${c} fails`);
  close(gate(108).rows[0]!.margin, 0.5); assert.equal(gate(60).rows[0]!.margin, 0);
  // The two-sided original fails a count well under the references; the bound does not.
  assert.ok(!scalarGate("D8", "", { rel: 0.10 }, "", [{ at: "16", refs: [100, 104], candidate: 60 }]).pass);
  assert.ok(!upperGate("D8", "", { rel: 0.10 }, "", [{ at: "16", refs: [100, 104], candidate: null }]).pass);
});

test("drift is one-sided against NB-FLIP's envelope: |c| <= max |ref_i| + floor", () => {
  // refs -0.010 and -0.012: envelope 0.012, floor 0.001, so |c| <= 0.013 whatever the spread.
  const gate = (c: number) => driftGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [-0.010, -0.012], candidate: c }]);
  for (const c of [0, -0.003, 0.003, -0.0129, 0.0129]) assert.ok(gate(c).pass, `${c} passes`);
  for (const c of [-0.0131, 0.0131, -0.05]) assert.ok(!gate(c).pass, `${c} fails`);
  const row = gate(-0.0125).rows[0]!;
  close(row.ref!, 0.012); close(row.tol, 0.001); close(row.margin, 0.5, 1e-9, "margin is the excess over the envelope / floor");
  assert.equal(gate(-0.002).rows[0]!.margin, 0);
  // A reference that gained volume widens S but not the bound: refs -0.010 and +0.008 (S 0.018).
  const mixed = (c: number) => driftGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [-0.010, 0.008], candidate: c }]);
  assert.ok(mixed(-0.0109).pass); assert.ok(!mixed(-0.0111).pass);
  // The first ruling (|ref mean| + max(floor, 3S)) would pass that loss of 5.5%; the envelope does not.
  assert.ok(driftMeanGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [-0.010, 0.008], candidate: -0.055 }]).pass);
  assert.ok(!mixed(-0.055).pass);
  // The two-sided original fails a candidate that simply drifts less.
  assert.ok(!scalarGate("D", "", { abs: 0.001 }, "", [{ at: "1", refs: [-0.010, -0.012], candidate: 0.003 }]).pass);
  // Missing values: both sides missing passes, one side missing fails.
  assert.ok(driftGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [null, null], candidate: null }]).pass);
  assert.ok(!driftGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [-0.01, -0.01], candidate: null }]).pass);
});

test("the first ruling's drift bound (printed, not gated): |c| <= |mean ref| + max(floor, 3S)", () => {
  const gate = (c: number) => driftMeanGate("D", "", { abs: 0.001 }, [{ at: "1", refs: [-0.010, -0.012], candidate: c }]);
  assert.ok(gate(-0.0169).pass); assert.ok(!gate(-0.0171).pass);
  close(gate(-0.0145).rows[0]!.margin, 0.0035 / 0.006, 1e-12);
});

test("a volume-coupled scalar passes anywhere between the reference band and the restored band", () => {
  // refs 100, 101 (mean 100.5, tol max(1%, 3) = 3); restored 110, 111 (mean 110.5, tol 3).
  const gate = (c: number) => bandScalarGate("B", "", { rel: 0.01 }, "", [{ at: "1", refs: [100, 101], restored: [110, 111], candidate: c }]);
  for (const c of [97.6, 100.5, 105, 110.5, 113.4]) assert.ok(gate(c).pass, `${c} passes`);
  for (const c of [97.4, 113.6, 120]) assert.ok(!gate(c).pass, `${c} fails`);
  assert.equal(gate(105).rows[0]!.margin, 0);
  close(gate(105).rows[0]!.unionMargin!, 4.5 / 3, 1e-12, "between the bands, outside both: the strict union fails it");
  close(gate(112).rows[0]!.unionMargin!, 0.5, 1e-12);
  const row = gate(105).rows[0]!;
  close(row.ref!, 100.5); close(row.restored!, 110.5); close(row.S, 1); close(row.restoredS!, 1); close(row.tol, 3); close(row.restoredTol!, 3);
  // The tolerance is interpolated along the segment: ref 10 (tol 1), restored 20 (tol 2, the 10% floor).
  const lin = (c: number) => bandScalarGate("B", "", { rel: 0.1 }, "", [{ at: "1", refs: [10, 10], restored: [20, 20], candidate: c }]);
  for (const c of [9.05, 15, 21.95]) assert.ok(lin(c).pass, `${c} passes`);
  for (const c of [8.95, 22.05]) assert.ok(!lin(c).pass, `${c} fails`);
  close(lin(8.5).rows[0]!.margin, 1.5, 1e-12, "below the ref end: d / tol_ref");
  close(lin(23).rows[0]!.margin, 1.5, 1e-12, "above the restored end: d / tol_restored");
  // A restored end below the references (a run that gained) works the same way.
  assert.ok(bandScalarGate("B", "", { abs: 1 }, "", [{ at: "1", refs: [20, 20], restored: [10, 10], candidate: 12 }]).pass);
  // With nothing lost (restored = refs) the band is the reference band.
  assert.ok(!bandScalarGate("B", "", { rel: 0.01 }, "", [{ at: "1", refs: [100, 101], restored: [100, 101], candidate: 103.6 }]).pass);
  assert.ok(bandScalarGate("B", "", { rel: 0.01 }, "", [{ at: "1", refs: [100, 101], restored: [100, 101], candidate: 103.4 }]).pass);
});

test("a volume-coupled profile passes within tolerance of the segment between the two mean profiles", () => {
  const gate = (c: number[]) => bandDistanceGate("P", "", 0.1, "", l1, [{ at: "1", refs: [[1, 0], [1, 0]], restored: [[0, 1], [0, 1]], candidate: c }]);
  assert.ok(gate([0.5, 0.5]).pass);
  assert.equal(gate([0.5, 0.5]).rows[0]!.margin, 0);
  assert.ok(gate([1, 0.05]).pass);
  assert.ok(gate([0.03, 1.05]).pass);
  assert.ok(!gate([1, 0.2]).pass, "0.2 from the nearest segment point");
  close(gate([1, 0.2]).rows[0]!.margin, 2, 1e-9);
  const row = gate([1, 0.2]).rows[0]!;
  close(row.candidate!, 0.2); close(row.restored!, 1.8); close(row.S, 0); close(row.tol, 0.1);
  // The references' own pairwise spread widens the tolerance at the reference end only.
  const wide = bandDistanceGate("P", "", 0.1, "", l1, [{ at: "1", refs: [[1, 0], [0.9, 0.1]], restored: [[0, 1], [0, 1]], candidate: [1.2, -0.2] }]);
  close(wide.rows[0]!.tol, 0.6); close(wide.rows[0]!.restoredTol!, 0.1);
  assert.ok(wide.pass);
});

/** A pool over a 48 x 16 x 16 box, 1 cm cells, three 16-cell x bins, the dam's back-wall column in bin 0. */
const g: BoxGeometry = { nx: 48, ny: 16, nz: 16, cell_m: 0.01, gravity_m_s2: 9.81, density_kg_m3: 1000, binCells: [16, 16], bins: [3, 1], column: { x: [0, 4], z: [0, 8] } };
/** The observer's own oracle over dense h cells: liquid below `depth(x)` (cells), at rest. */
function pool(depth: (x: number) => number, expected_cells: number) {
  const owners = [];
  for (let z = 0; z < g.nz; z++) for (let y = 0; y < g.ny; y++) for (let x = 0; x < g.nx; x++)
    owners.push({ q: [x, y, z] as [number, number, number], w: 1, phi: (_x: number, py: number) => py - depth(x), u: [0, 0, 0] as [number, number, number] });
  return boxObservation(g, "dam", 0, 1 / 60, boxOracle(g, owners), expected_cells);
}
const fill = (depth: (x: number) => number) => { let v = 0; for (let x = 0; x < g.nx; x++) v += depth(x) * g.nz; return v; };

test("restoreBox on a level pool equals the oracle's pool raised by the lost volume over the footprint", () => {
  const measured = () => 10, raised = () => 10.375;
  const s = pool(measured, fill(raised)), oracle = pool(raised, fill(raised));
  const b = restoreBox(g, s);
  close(b.lost_L, (fill(raised) - fill(measured)) * 1e-3, 1e-12, "lost litres");
  close(b.depth_cells, 0.375, 1e-12, "delta"); assert.equal(b.wetBins, 3);
  allClose(b.bins_L, oracle.bins_L, 1e-9, "bins"); allClose(b.binDepth_cells, binDepths(g, oracle.bins_L), 1e-9, "bin depths");
  allClose(b.profileDepth_cells, oracle.profileDepth_cells, 1e-9, "profile depth"); allClose(b.binTops_cells, oracle.binTops_cells, 1e-9, "tops");
  allClose(b.profileTop_cells, oracle.profileTop_cells, 1e-9, "profile tops");
  close(b.columnDepth_cells!, oracle.columnDepth_cells!, 1e-9, "column depth"); close(b.columnTop_cells!, oracle.columnTop_cells!, 1e-9, "column top");
  close(b.pe_J, oracle.pe_J, 1e-9, "PE"); close(b.energy_J, oracle.energy_J, 1e-9, "E"); close(b.centroidY_cells!, oracle.centroid![1], 1e-9, "centroid y");
  // PE in closed form: rho g h^4 nx nz D^2 / 2 at D = 10.375.
  close(b.pe_J, 1000 * 9.81 * 1e-8 * 48 * 16 * 10.375 ** 2 / 2, 1e-9, "PE closed form");
});

test("restoreBox raises only the wet footprint, by one uniform depth, and leaves a run that gained untouched", () => {
  // Bins 0 and 1 wet at 10 and 6 cells; bin 2 a 0.5-cell film, below WET_DEPTH_CELLS, so it is not raised.
  const measured = (x: number) => x < 16 ? 10 : x < 32 ? 6 : 0.5, raised = (x: number) => x < 16 ? 10.25 : x < 32 ? 6.25 : 0.5;
  const s = pool(measured, fill(raised)), oracle = pool(raised, fill(raised)), b = restoreBox(g, s);
  close(b.depth_cells, 0.25, 1e-12, "delta over the two wet bins"); assert.equal(b.wetBins, 2);
  allClose(b.bins_L, oracle.bins_L, 1e-9, "bins"); allClose(b.profileDepth_cells, oracle.profileDepth_cells, 1e-9, "profile");
  allClose(b.binTops_cells, oracle.binTops_cells, 1e-9, "tops"); close(b.columnDepth_cells!, 10.25, 1e-9, "column");
  close(b.pe_J, oracle.pe_J, 1e-9, "PE"); close(b.centroidY_cells!, oracle.centroid![1], 1e-9, "centroid y");
  // A run that gained volume (expected < total) is not restored.
  const gained = restoreBox(g, pool(raised, fill(measured)));
  assert.equal(gained.lost_L, 0); assert.equal(gained.depth_cells, 0);
  allClose(gained.bins_L, oracle.bins_L, 1e-12, "unchanged bins"); close(gained.pe_J, oracle.pe_J, 1e-12, "unchanged PE");
});
