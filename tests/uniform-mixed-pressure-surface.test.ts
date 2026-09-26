import assert from "node:assert/strict";
import test from "node:test";
import { cm12GhostFluidTheta, CM12_GHOST_FLUID_THETA_MIN } from "../lib/core/cm12-numerics";
import { mixedPressureFixture, mixedPressureLayouts, type PressureCell, type PressureFace } from "./helpers/uniform-mixed-pressure";

/** Algebraic seam candidate, not a surface solver. Slopes are supplied exactly
 * here: reconstruction, multigrid and evolving surfaces require separate gates.
 * Keep native theta along the center-to-center segment, then remove its
 * tangential directional derivative to obtain the MAC normal derivative. */
function ghostGradient(face: PressureFace, cells: readonly PressureCell[], phi: ArrayLike<number>, pressure: ArrayLike<number>, slope: readonly number[]): number {
  const {left,right,axis,distance} = face;
  assert.notEqual(phi[left]! < 0, phi[right]! < 0);
  const liquid = phi[left]! < 0 ? left : right, air = liquid === left ? right : left;
  const theta = cm12GhostFluidTheta(phi[liquid]!,phi[air]!,1e-9);
  const base = (liquid === left ? -pressure[left]! : pressure[right]!) / (theta*distance);
  let sideways = 0;
  for(let a=0;a<3;a++)if(a!==axis)sideways += slope[a]!*(cells[right]!.center[a]!-cells[left]!.center[a]!);
  return base-sideways/distance;
}

test("ghost-fluid seam correction preserves planar hydrostatics with supplied affine liquid slopes", t => {
  let mixedCrossings=0, falseHorizontalFaces=0, maxUncorrected=0;
  for(const layout of mixedPressureLayouts()) {
    const fixture=mixedPressureFixture(layout);
    for(let vertical=0;vertical<3;vertical++)for(const fraction of [.1875,.34375,.59375,.8125]) {
      const height=layout.lattice.dimensions[vertical]!*layout.lattice.cellSize_m[vertical]!*fraction;
      const phi=fixture.cells.map(c=>c.center[vertical]!-height), pressure=phi.map(v=>Math.max(0,-v));
      const slope=[0,0,0];slope[vertical]=-1;
      for(const face of fixture.faces) {
        const {left,right}=face;
        if((phi[left]!<0)===(phi[right]!<0))continue;
        const liquid=phi[left]!<0?left:right,air=liquid===left?right:left;
        const rawTheta=Math.abs(phi[liquid]!)/(Math.abs(phi[liquid]!)+Math.abs(phi[air]!));
        // The native 0.05 clamp intentionally changes the boundary distance.
        // Test its separate parity below, not exact hydrostatics at a moved BC.
        if(rawTheta<CM12_GHOST_FLUID_THETA_MIN)continue;
        const actual=ghostGradient(face,fixture.cells,phi,pressure,slope);
        assert.ok(Math.abs(actual-slope[face.axis]!)<1e-12,`axis ${vertical}, face ${face.axis}, gradient ${actual}`);
        if(fixture.cells[left]!.width!==fixture.cells[right]!.width) {
          mixedCrossings++;
          if(face.axis!==vertical) {
            falseHorizontalFaces++;
            maxUncorrected=Math.max(maxUncorrected,pressure[liquid]!/(rawTheta*face.distance));
          }
        }
      }
    }
  }
  assert.ok(mixedCrossings>0 && falseHorizontalFaces>0 && maxUncorrected>.1);
  t.diagnostic(`${mixedCrossings} mixed surface crossings; ${falseHorizontalFaces} tangential faces; naive peak=${maxUncorrected}`);
});

test("aligned ghost-fluid faces retain native theta including its minimum clamp", () => {
  for(const liquidOnLeft of [true,false])for(const liquidPhi of [-1e-8,-.01,-.5,-2]) {
    const cells:PressureCell[]=[{center:[0,0,0],width:1,volume:1},{center:[1,0,0],width:1,volume:1}];
    const face:PressureFace={left:0,right:1,axis:0,center:[.5,0,0],area:1,distance:1};
    const phi=liquidOnLeft?[liquidPhi,1]:[1,liquidPhi], pressure=liquidOnLeft?[2,0]:[0,2];
    const expected=(liquidOnLeft?-2:2)/cm12GhostFluidTheta(liquidPhi,1,1e-9);
    assert.equal(ghostGradient(face,cells,phi,pressure,[3,5,7]),expected);
  }
});
