import { buildOctreePowerHybridPlan } from "../lib/octree-power-hybrid";
import { buildCartesianPowerFixture } from "./octree-power-hybrid-cartesian-fixture";

const dimensions = [32, 32, 32] as const;
const operator = buildCartesianPowerFixture(dimensions);
const plan = buildOctreePowerHybridPlan(operator);
const report = {
  dimensions,
  rows: operator.rows.length,
  regularInteriorRows: plan.regularRows.length,
  powerBandRows: plan.powerRows.length,
  seamFaces: plan.seamFaces,
  fullPowerWork: plan.fullPowerWork,
  hybridWork: plan.hybridWork,
  workReduction: plan.workReduction,
  accepted: plan.workReduction >= 2,
};
if (!report.accepted) throw new Error(`Bet 4 Stage-1 work reduction ${report.workReduction.toFixed(3)}x is below 2x`);
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
