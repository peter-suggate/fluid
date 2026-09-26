import assert from "node:assert/strict";
import test from "node:test";
import { createUniformMixedLayout, mixedCellWidth, uniformMixedPressureLevel } from "../lib/methods/uniform/uniform-mixed-layout";
import { DEFAULT_UNIFORM_CM11A_SCHEDULE } from "../lib/methods/uniform/pressure-policy";
import { mixedPressureLayouts, geometricDivergence } from "./helpers/uniform-mixed-pressure";
import { MixedMultigridOracle } from "./helpers/uniform-mixed-multigrid";

const max = (values: Iterable<number>) => Math.max(...Array.from(values, Math.abs));
test("mixed pressure levels retain coarse owners and conserve restricted physical residual", () => {
  for (const layout of mixedPressureLayouts()) {
    const oracle = new MixedMultigridOracle(layout);
    for (const level of [0, 1]) {
      const fine = oracle.fixtures[level]!, coarse = oracle.fixtures[level + 1]!;
      const volumes = new Float64Array(coarse.cells.length);
      fine.cells.forEach((cell, i) => { volumes[oracle.parents[level]![i]!]! += cell.volume; });
      assert.deepEqual([...volumes], coarse.cells.map(c => c.volume));
      const values = Float64Array.from(fine.cells, (_, i) => Math.sin(i * 7));
      const restricted = oracle.restrict(level, values);
      assert.ok(Math.abs(restricted.reduce((s, v) => s + v, 0) - values.reduce((s, v) => s + v, 0)) < 1e-11);
      assert.ok(oracle.prolong(level, new Float64Array(coarse.cells.length).fill(7)).every(v => Math.abs(v - 7) < 1e-12));
      fine.cells.forEach((cell, i) => {
        const parent = oracle.parents[level]![i]!;
        if (cell.width === coarse.cells[parent]!.width) assert.deepEqual([...oracle.interpolation[level]![i]!], [[parent, 1]]);
      });
      // All direct core dependencies cross tier/parity colours. Reconstruction
      // dependencies may not, which is why those values are frozen first.
      const colour = (i: number) => {
        const c = fine.cells[i]!, h = layout.lattice.cellSize_m;
        return 2 * Math.log2(c.width) + (c.center.reduce((s, v, a) => s + Math.floor(v / h[a]! / c.width), 0) & 1);
      };
      for (const face of fine.faces) assert.notEqual(colour(face.left), colour(face.right));
    }
    const bottom = oracle.layouts[2]!;
    assert.ok(bottom.tiles.every(word => mixedCellWidth(word) === 4));
    assert.equal(uniformMixedPressureLevel(bottom, 4), bottom);
    assert.equal(bottom.cellCount, layout.tiles.length);
  }
});

test("coupled mixed cycles converge with the unchanged native cycle budget, including the thin-domain counterexample", () => {
  assert.deepEqual([DEFAULT_UNIFORM_CM11A_SCHEDULE.fullCycles, DEFAULT_UNIFORM_CM11A_SCHEDULE.vCycles,
    DEFAULT_UNIFORM_CM11A_SCHEDULE.preSweeps, DEFAULT_UNIFORM_CM11A_SCHEDULE.postSweeps], [3, 4, 6, 6]);
  for (const [index, layout] of mixedPressureLayouts().entries()) {
    const oracle = new MixedMultigridOracle(layout), fixture = oracle.fixtures[0]!;
    for (const mode of ["hydrostatic", "random"] as const) {
      const velocity = Float64Array.from(fixture.faces, (f, k) => mode === "hydrostatic" ? (f.axis === 1 ? -9.81 : 0) : Math.sin(k * 13));
      const rhs = geometricDivergence(fixture, velocity).map((v, i) => -v * fixture.cells[i]!.volume);
      const solved = oracle.solve(rhs);
      assert.equal(solved.residuals.length, 7);
      assert.ok(solved.residuals.every((r, i, a) => Number.isFinite(r) && (i === 0 || r <= a[i - 1]!)), `${index}/${mode}: ${solved.residuals}`);
      // Coupled-cycle acceptance is separate from the exact-core outer-loop
      // experiment. This is a physical divergence bound under native budgets.
      assert.ok(solved.residuals.at(-1)! < 1e-3, `${index}/${mode}: ${solved.residuals.at(-1)}`);
    }
    // Compare hydrostatic convergence with the same CPU cycle oracle entirely
    // fine. This is not parity with a live GPU/free-surface simulation.
    const h = layout.lattice.cellSize_m, d = layout.lattice.dimensions;
    const fine = createUniformMixedLayout(layout.lattice, [{ id: "all-fine", rule: "minimum-cell-size", minimumCellSize_cells: 1,
      maximumCellSize_cells: 1, min_m: { x: 0, y: 0, z: 0 }, max_m: { x: d[0] * h[0], y: d[1] * h[1], z: d[2] * h[2] } }], true, 4);
    const baseline = new MixedMultigridOracle(fine), bf = baseline.fixtures[0]!;
    const baseRhs = geometricDivergence(bf, Float64Array.from(bf.faces, f => f.axis === 1 ? -9.81 : 0)).map((v, i) => -v * bf.cells[i]!.volume);
    const mixedRhs = geometricDivergence(fixture, Float64Array.from(fixture.faces, f => f.axis === 1 ? -9.81 : 0)).map((v, i) => -v * fixture.cells[i]!.volume);
    const mixedError = oracle.solve(mixedRhs).residuals.at(-1)!, fineError = baseline.solve(baseRhs).residuals.at(-1)!;
    assert.ok(mixedError <= Math.max(1e-8, 1.05 * fineError), `fixture ${index}: mixed ${mixedError}, fine ${fineError}`);
    assert.ok(max(oracle.apply(0, new Float64Array(fixture.cells.length).fill(3))) < 1e-12);
  }
});


test("mixed correction prolongation preserves affine pressure across interior resolution seams",()=>{
 for(const layout of mixedPressureLayouts().slice(0,3)){
  const oracle=new MixedMultigridOracle(layout),h=layout.lattice.cellSize_m;
  for(const level of [0,1]){
   const fine=oracle.fixtures[level]!,coarse=oracle.fixtures[level+1]!;
   const affine=(p:readonly number[])=>3+p[0]!*0.7-p[1]!*1.3+p[2]!*0.2;
   const actual=oracle.prolong(level,Float64Array.from(coarse.cells,c=>affine(c.center)));
   let checked=0;
   fine.cells.forEach((c,i)=>{
    const width=coarse.cells[oracle.parents[level]![i]!]!.width;
    if(c.center.some((v,a)=>v/h[a]!<width/2||v/h[a]!>layout.lattice.dimensions[a]!-width/2))return;
    assert.ok(Math.abs(actual[i]!-affine(c.center))<1e-11,`level ${level}, cell ${i}`);checked++;
   });
   assert.ok(checked>0);
  }
 }
});
