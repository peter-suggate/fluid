/**
 * Per-level, per-stage census of the CM11a pressure plan, on the CPU.
 *
 * `UniformCM11aPressureMultigrid.buildPlan()` decides a dispatch's extent from
 * two things only: the level dimensions the hierarchy planner produced, and the
 * fixed cycle schedule. Neither needs a device, so the plan's *shape* -- entry
 * point, schedule stage, active level, workgroup counts -- can be reproduced
 * exactly without spending a GPU run on it. This file mirrors that walk, and
 * `--validate` checks the mirror against the plan censuses the Dawn probe
 * captured from the real solver, so the numbers below are not a model.
 *
 * Usage:
 *   node --import tsx docs/research/uniform-geometric-tall-air-2026-09-19/pressure-plan-census.mts
 *   node --import tsx .../pressure-plan-census.mts --validate=<probe-output.json>[,<more.json>]
 */
import { readFileSync } from "node:fs";
import { planUniformCM11aHierarchy, planUniformCM11aWindow,
  UNIFORM_CM11A_FULL_CYCLES, UNIFORM_CM11A_V_CYCLES,
  UNIFORM_CM11A_RECOVERY_BATCHES, UNIFORM_CM11A_RECOVERY_SWEEPS,
  UNIFORM_CM11A_PRE_SWEEPS, UNIFORM_CM11A_POST_SWEEPS } from "../../../lib/methods/uniform/webgpu-uniform-pressure-multigrid";
import { tallAirScene, TALL_AIR_RESERVOIR_M } from "./tall-air-scene.mjs";
import { sceneLatticeDimensions } from "../../../lib/core/scene-lattice-dimensions";

type Stage = "setup" | "full-cycle" | "v-cycle" | "finish";
interface Emit { entryPoint: string; stage: Stage; level: number; workgroups: number }

const SCHEDULE = { fullCycles: UNIFORM_CM11A_FULL_CYCLES, vCycles: UNIFORM_CM11A_V_CYCLES,
  preSweeps: UNIFORM_CM11A_PRE_SWEEPS, postSweeps: UNIFORM_CM11A_POST_SWEEPS };

/**
 * Mirror of buildPlan(). Only the shape is reproduced; every texture argument
 * is dropped, and the pressure/minimum parity flips are kept because the two
 * checkpoint copies are conditional on them.
 */
function mirrorPlan(physical: readonly (readonly [number, number, number])[]): {
  emits: Emit[]; cycleBoundaries: number[]; finishStart: number;
} {
  // Levels carry a one-cell halo on every axis (multigrid :339).
  const dims = physical.map((d) => [d[0] + 2, d[1] + 2, d[2] + 2] as [number, number, number]);
  const M = dims.length;
  const emits: Emit[] = [];
  const p = new Array<number>(M).fill(0);
  let stage: Stage = "setup";
  const emit = (entryPoint: string, _source: number, destination = _source,
    dispatch: readonly [number, number, number] = dims[destination]!) => {
    emits.push({ entryPoint, stage, level: destination,
      workgroups: Math.ceil(dispatch[0] / 4) * Math.ceil(dispatch[1] / 4) * Math.ceil(dispatch[2] / 4) });
  };
  const sweep = (level: number) => { for (let c = 0; c < 2; c += 1) { emit("mgSmoothColour", level); p[level]! ^= 1; } };
  const coarseSolve = () => { emit("mgSolveCoarsest", M - 1, M - 1, [1, 1, 1]); p[M - 1]! ^= 1; };
  const vCycle = (level: number): void => {
    if (level === M - 1) { coarseSolve(); return; }
    for (let i = 0; i < SCHEDULE.preSweeps; i += 1) sweep(level);
    emit("mgResidual", level);
    emit("mgRestrictResidual", level, level + 1);
    emit("mgClearPressure", level + 1); p[level + 1]! ^= 1;
    emit("mgDownsampleSubtract", level, level + 1);
    vCycle(level + 1);
    emit("mgProlongateAdd", level + 1, level); p[level]! ^= 1;
    for (let i = 0; i < SCHEDULE.postSweeps; i += 1) sweep(level);
  };
  const fullCycle = () => {
    emit("mgCopyPressure", 0); emit("mgShiftMinimum", 0); emit("mgResidual", 0);
    for (let level = 0; level + 1 < M; level += 1) {
      emit("mgRestrictResidual", level, level + 1);
      emit("mgDownsampleMinimum", level, level + 1);
    }
    emit("mgClearPressure", M - 1); p[M - 1]! ^= 1;
    coarseSolve();
    for (let level = M - 2; level >= 0; level -= 1) {
      emit("mgProlongateAssign", level + 1, level); p[level]! ^= 1;
      vCycle(level);
    }
    emit("mgAddPressure", 0); p[0]! ^= 1;
  };
  const checkpoint = () => {
    if (p[0] !== 0) { emit("mgCopyPressure", 0); p[0] = 0; }
    emit("mgMeasureFineResidual", 0);
    emit("mgCheckCycleConvergence", 0, 0, [1, 1, 1]);
    emit("mgSaveAccepted", 0); emit("mgRestoreRejected", 0);
  };
  emit("mgBuildFinestTopology", 0); emit("mgBuildFinestRhs", 0);
  for (let level = 0; level + 1 < M; level += 1) emit("mgDownsampleTopology", level, level + 1);
  for (let level = 0; level < M; level += 1) { emit("mgExtrapolatePhiOneCell", level); emit("mgBakeCoefficients", level); }
  emit("mgMeasureFineResidual", 0);
  emit("mgCheckCycleConvergence", 0, 0, [1, 1, 1]); emit("mgCopyPressure", 0);
  const cycleBoundaries: number[] = [emits.length];
  stage = "full-cycle";
  for (let c = 0; c < SCHEDULE.fullCycles; c += 1) { fullCycle(); checkpoint(); cycleBoundaries.push(emits.length); }
  stage = "v-cycle";
  for (let c = 0; c < SCHEDULE.vCycles; c += 1) { vCycle(0); checkpoint(); cycleBoundaries.push(emits.length); }
  stage = "finish";
  const finishStart = emits.length;
  for (let batch = 0; batch < UNIFORM_CM11A_RECOVERY_BATCHES; batch++) {
    for (let i = 0; i < UNIFORM_CM11A_RECOVERY_SWEEPS; i++) sweep(0);
    checkpoint();
  }
  emit("mgRestoreRejected", 0); emit("mgFinishSafety", 0, 0, [1, 1, 1]);
  if (p[0] !== 0) { emit("mgCopyPressure", 0); p[0] = 0; }
  emit("mgMeasureFineResidual", 0);
  return { emits, cycleBoundaries, finishStart };
}

const summarize = (emits: readonly Emit[]) => {
  const byLevel: Record<number, { passes: number; workgroups: number }> = {};
  const byStage: Record<string, number> = {};
  const byStageLevel: Record<string, Record<number, { passes: number; workgroups: number }>> = {};
  const byEntry: Record<string, { passes: number; workgroups: number }> = {};
  for (const e of emits) {
    (byLevel[e.level] ??= { passes: 0, workgroups: 0 });
    byLevel[e.level]!.passes += 1; byLevel[e.level]!.workgroups += e.workgroups;
    byStage[e.stage] = (byStage[e.stage] ?? 0) + 1;
    (byStageLevel[e.stage] ??= {});
    (byStageLevel[e.stage]![e.level] ??= { passes: 0, workgroups: 0 });
    byStageLevel[e.stage]![e.level]!.passes += 1;
    byStageLevel[e.stage]![e.level]!.workgroups += e.workgroups;
    (byEntry[e.entryPoint] ??= { passes: 0, workgroups: 0 });
    byEntry[e.entryPoint]!.passes += 1; byEntry[e.entryPoint]!.workgroups += e.workgroups;
  }
  return { passes: emits.length, workgroups: emits.reduce((t, e) => t + e.workgroups, 0),
    byLevel, byStage, byStageLevel, byEntry };
};

/** The prefix a `cycles`-cycle lagged budget actually encodes, plus the finish. */
const encodedSubset = (plan: ReturnType<typeof mirrorPlan>, cycles: number): Emit[] => {
  const prefixEnd = plan.cycleBoundaries[Math.min(cycles, plan.cycleBoundaries.length - 1)] ?? plan.emits.length;
  return plan.emits.filter((_, index) => index < prefixEnd || index >= plan.finishStart);
};

const MULTIPLES = (process.env.FLUID_TALL_MULTIPLES ?? "1,2,4,8").split(",").map(Number);
/** Host slacks worth pricing: the startup seed pad, and a settled step's pad. */
const WINDOW_PADS = (process.env.FLUID_WINDOW_PADS ?? "32,12").split(",").map(Number);
const report: Record<string, unknown> = {};
for (const multiple of MULTIPLES) {
  const scene = tallAirScene(multiple);
  const lattice = sceneLatticeDimensions(scene, Number.MAX_SAFE_INTEGER) as unknown as [number, number, number];
  const hierarchy = planUniformCM11aHierarchy(lattice);
  const plan = mirrorPlan(hierarchy.levelDimensions);
  const full = summarize(plan.emits);
  const encoded: Record<number, ReturnType<typeof summarize>> = {};
  for (const cycles of [1, 2, 3]) encoded[cycles] = summarize(encodedSubset(plan, cycles));
  // The window lattice, planned the way the host plans it: the CPU-known
  // starting wet box, padded by the seed slack, clipped to the domain. The
  // reservoir is authored in absolute metres and the cell size is fixed, so
  // this box is the SAME in cells for every multiple -- which is the whole
  // claim: a capacity sized to it must give every arm arm A's hierarchy.
  const cell = scene.container.width_m / lattice[0];
  const reservoir = [TALL_AIR_RESERVOIR_M.x, TALL_AIR_RESERVOIR_M.y, TALL_AIR_RESERVOIR_M.z]
    .map((metres) => Math.round(metres / cell));
  const windowPlans: Record<string, unknown> = {};
  for (const pad of WINDOW_PADS) {
    // The reservoir is seated in the corner, so its box is [0, cells) per axis.
    const low: [number, number, number] = [0, 0, 0];
    const high = reservoir.map((cells, axis) =>
      Math.min(lattice[axis]!, cells + pad)) as [number, number, number];
    const window = planUniformCM11aWindow(lattice, low, high);
    const windowPlan = mirrorPlan(window.hierarchy.levelDimensions);
    const windowFull = summarize(windowPlan.emits);
    const windowEncoded: Record<number, ReturnType<typeof summarize>> = {};
    for (const cycles of [1, 2, 3]) windowEncoded[cycles] = summarize(encodedSubset(windowPlan, cycles));
    windowPlans[`pad${pad}`] = { box: { low, high }, capacity: window.capacity, origin: window.origin,
      alignment: window.alignment, levelCount: window.hierarchy.levelCount,
      semiCoarsened: window.hierarchy.semiCoarsened, coarsestCells: window.hierarchy.coarsestCells,
      levelDimensionsHaloed: window.hierarchy.levelDimensions.map((d) => [d[0] + 2, d[1] + 2, d[2] + 2]),
      cells: window.capacity.reduce((product, value) => product * value, 1),
      domainCells: lattice.reduce((product, value) => product * value, 1),
      full: windowFull, encoded: windowEncoded };
  }
  report[`${multiple}x`] = { lattice, levelCount: hierarchy.levelCount,
    semiCoarsened: hierarchy.semiCoarsened, coarsestCells: hierarchy.coarsestCells,
    levelDimensionsHaloed: hierarchy.levelDimensions.map((d) => [d[0] + 2, d[1] + 2, d[2] + 2]),
    cycleBoundaries: plan.cycleBoundaries, finishStart: plan.finishStart, full, encoded,
    window: windowPlans };
}

const validate = process.argv.find((a) => a.startsWith("--validate="));
if (validate) {
  const problems: string[] = [];
  for (const file of validate.slice("--validate=".length).split(",")) {
    const captured = JSON.parse(readFileSync(file, "utf8")) as { arms: Record<string, any>[] };
    for (const arm of captured.arms ?? []) {
      const key = `${arm.multiple}x`;
      const mine = report[key] as any;
      if (!mine || arm.dense) continue;
      const theirs = arm.structure?.pressurePlanByLevel ?? arm.pressurePlanByLevel;
      if (!theirs) continue;
      for (const [level, value] of Object.entries(theirs) as [string, any][]) {
        const got = mine.full.byLevel[Number(level)];
        if (!got || got.passes !== value.passes || got.workgroups !== value.workgroups) {
          problems.push(`${file} ${key} L${level}: mirror ${JSON.stringify(got)} vs captured ${JSON.stringify(value)}`);
        }
      }
      const theirStage = arm.structure?.pressurePlanByStage ?? arm.pressurePlanByStage;
      for (const [stage, passes] of Object.entries(theirStage ?? {}) as [string, number][]) {
        if (mine.full.byStage[stage] !== passes) {
          problems.push(`${file} ${key} stage ${stage}: mirror ${mine.full.byStage[stage]} vs captured ${passes}`);
        }
      }
    }
  }
  report.validation = problems.length === 0 ? "mirror matches every captured plan census" : problems;
}

console.log(JSON.stringify(report, null, 2));
