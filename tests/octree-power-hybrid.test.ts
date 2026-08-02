import assert from "node:assert/strict";
import test from "node:test";
import {
  OCTREE_POWER_HYBRID_FACE_CLASS,
  OCTREE_POWER_HYBRID_ROW_CLASS,
  applyOctreePowerHybridMatrix,
  buildOctreePowerHybridPlan,
} from "../lib/octree-power-hybrid";
import { applyOctreePowerMatrix } from "../lib/octree-power-operator";
import { OCTREE_CUBE_TRANSFORMS } from "../lib/octree-power-topology";
import { buildCartesianPowerFixture } from "../tools/octree-power-hybrid-cartesian-fixture";

const bits64 = (value: number) => new BigUint64Array(Float64Array.of(value).buffer)[0];

test("hybrid proof keeps GFM/cut and boundary rows in the power band and power-owns every seam", () => {
  const dimensions = [8, 8, 8] as const;
  const cut = ([x, y, z]: readonly number[], [bx, by, bz]: readonly number[]) =>
    y === 4 && by === 4 && z === 4 && bz === 4 && Math.min(x, bx) === 3 ? 0.375 : 1;
  const operator = buildCartesianPowerFixture(dimensions, { openFraction: cut });
  const forced = 4 + 8 * (3 + 8 * 4);
  const plan = buildOctreePowerHybridPlan(operator, { forcePowerRows: [forced] });
  const cutRows = [3 + 8 * (4 + 8 * 4), 4 + 8 * (4 + 8 * 4)];
  for (const row of cutRows) assert.equal(plan.rowClasses[row], OCTREE_POWER_HYBRID_ROW_CLASS.powerBand);
  assert.equal(plan.rowClasses[forced], OCTREE_POWER_HYBRID_ROW_CLASS.powerBand);
  assert.ok(plan.regularRows.length > 0);
  assert.ok(plan.seamFaces > 0);
  for (const face of operator.faces) {
    const negativePower = plan.rowClasses[face.negativeRow] === OCTREE_POWER_HYBRID_ROW_CLASS.powerBand;
    const positivePower = face.positiveRow === 0xffff_ffff
      || plan.rowClasses[face.positiveRow] === OCTREE_POWER_HYBRID_ROW_CLASS.powerBand;
    assert.equal(plan.faceClasses[face.id] === OCTREE_POWER_HYBRID_FACE_CLASS.power,
      negativePower || positivePower, "a face incident to power-band must use one shared power coefficient");
  }
  assert.throws(() => buildOctreePowerHybridPlan(operator, { forcePowerRows: [operator.rows.length] }),
    /outside the published operator/);
  const brokenIncidence = operator.incidence.map((incidence) => [...incidence]);
  brokenIncidence[0] = [];
  assert.throws(() => buildOctreePowerHybridPlan({ ...operator, incidence: brokenIncidence }),
    /unique physical faces/, "an under-published seam must reject rather than silently lose symmetry");
});

test("f64 hybrid differential is bit-identical and the seam remains symmetric with matching energy", () => {
  const operator = buildCartesianPowerFixture([9, 8, 7]);
  const forced = Array.from({ length: operator.rows.length }, (_, row) => row)
    .filter((row) => operator.sites[row]!.origin[0] === 4 && operator.sites[row]!.origin[1] >= 2);
  const plan = buildOctreePowerHybridPlan(operator, { forcePowerRows: forced });
  let state = 0x8f3a_19d5;
  const random = () => {
    state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
    return (state >>> 0) / 0x1_0000_0000;
  };
  const r = Array.from({ length: operator.rows.length }, () => random() * 2 - 1);
  const s = Array.from({ length: operator.rows.length }, () => random() * 2 - 1);
  const fullR = applyOctreePowerMatrix(operator, r), hybridR = applyOctreePowerHybridMatrix(operator, plan, r);
  const fullS = applyOctreePowerMatrix(operator, s), hybridS = applyOctreePowerHybridMatrix(operator, plan, s);
  assert.deepEqual(hybridR.map(bits64), fullR.map(bits64));
  assert.deepEqual(hybridS.map(bits64), fullS.map(bits64));
  const rAs = r.reduce((sum, value, row) => sum + value * hybridS[row]!, 0);
  const sAr = s.reduce((sum, value, row) => sum + value * hybridR[row]!, 0);
  const scale = Math.max(1, Math.abs(rAs), Math.abs(sAr));
  assert.ok(Math.abs(rAs - sAr) <= 32 * Number.EPSILON * scale, `${rAs} != ${sAr}`);
  const fullEnergy = r.reduce((sum, value, row) => sum + value * fullR[row]!, 0);
  const hybridEnergy = r.reduce((sum, value, row) => sum + value * hybridR[row]!, 0);
  assert.equal(bits64(hybridEnergy), bits64(fullEnergy));
  assert.ok(hybridEnergy > 0);
  assert.throws(() => applyOctreePowerHybridMatrix(
    buildCartesianPowerFixture([9, 8, 7]), plan, r,
  ), /epoch mismatch/);
});

test("regular-interior classification and operator are equivariant under all 48 D4/cube transforms", () => {
  const n = 8;
  const operator = buildCartesianPowerFixture([n, n, n]);
  const plan = buildOctreePowerHybridPlan(operator);
  const row = (x: number, y: number, z: number) => x + n * (y + n * z);
  const source = Array.from({ length: n ** 3 }, (_, index) => ((index * 37) % 251) - 125);
  const sourceImage = applyOctreePowerHybridMatrix(operator, plan, source);
  for (const transform of OCTREE_CUBE_TRANSFORMS) {
    const transformed = new Array<number>(source.length);
    const destinationOf = (coordinate: readonly [number, number, number]) => {
      const centered = coordinate.map((value) => 2 * value - (n - 1));
      const out = [0, 1, 2].map((axis) => transform.sign[axis] * centered[transform.permutation[axis]]!);
      return out.map((value) => (value + n - 1) / 2) as [number, number, number];
    };
    for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
      const destination = destinationOf([x, y, z]);
      const from = row(x, y, z), to = row(...destination);
      transformed[to] = source[from]!;
      assert.equal(plan.rowClasses[to], plan.rowClasses[from], `class changed under transform ${transform.code}`);
    }
    const transformedImage = applyOctreePowerHybridMatrix(operator, plan, transformed);
    for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
      const destination = destinationOf([x, y, z]);
      assert.equal(bits64(transformedImage[row(...destination)]!), bits64(sourceImage[row(x, y, z)]!),
        `operator changed under transform ${transform.code}`);
    }
  }
});

test("32-cubed interior removes more than 2x of descriptor/catalog/pageSlot machinery", () => {
  const operator = buildCartesianPowerFixture([32, 32, 32]);
  const plan = buildOctreePowerHybridPlan(operator);
  assert.equal(plan.regularRows.length, 30 ** 3);
  assert.equal(plan.powerRows.length, 32 ** 3 - 30 ** 3);
  assert.ok(plan.workReduction >= 2, `only ${plan.workReduction.toFixed(3)}x machinery reduction`);
  assert.ok(plan.fullPowerWork.descriptorRows / plan.hybridWork.descriptorRows >= 2);
  assert.ok(plan.fullPowerWork.pageSlotTraversals / plan.hybridWork.pageSlotTraversals >= 2);
});
