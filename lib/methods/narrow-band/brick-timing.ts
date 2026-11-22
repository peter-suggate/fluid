/**
 * The brick engine's timing seams: the phase labels a sampled step closes (BrickEngine's timed encode) and the SIM
 * panel's brick graph reads (brick-pipeline.ts). One table, so the graph's labels are the encoder's by construction.
 *
 * A seam is a run of placements of one part (BrickStageSchedule.seams): the engine encodes each run as its own compute
 * pass on a sampled step and closes its phase at that pass's end, so the phases partition the step's GPU time exactly.
 */
import type { GPUTimestampPhase } from "../../core/performance-trace";

export const BRICK_PHASE = {
  // Part 1, the head (lane E).
  census: { id: "coarse-grid", label: "brick:census" },
  residency: { id: "coarse-grid", label: "brick:residency" },
  transfer: { id: "coarse-grid", label: "brick:transfer" },
  stamp: { id: "coarse-grid", label: "brick:stamp" },
  // Part 1, F's move and rebin.
  moveCaches: { id: "fine-sdf-advection", label: "brick:move-caches" },
  move: { id: "fine-sdf-advection", label: "brick:move" },
  rebin: { id: "fine-sdf-advection", label: "brick:rebin" },
  // Part 2, E's brick set.
  set: { id: "coarse-grid", label: "brick:set" },
  // Part 2, F's level set.
  levelAdvect: { id: "fine-sdf-advection", label: "brick:level-advect" },
  levelUnion: { id: "fine-sdf-advection", label: "brick:level-union" },
  levelAnalysis: { id: "fine-sdf-redistance", label: "brick:level-analysis" },
  levelPublish: { id: "fine-sdf-redistance", label: "brick:level-publish" },
  // Part 2, F's velocity.
  velocityAdvect: { id: "velocity-advection", label: "brick:velocity-advect" },
  velocityDeposit: { id: "velocity-advection", label: "brick:velocity-deposit" },
  velocityViscous: { id: "velocity-advection", label: "brick:velocity-viscous" },
  velocityTransfer: { id: "velocity-advection", label: "brick:velocity-transfer" },
  // Part 2, P's pressure.
  pressureOperator: { id: "pressure-system", label: "brick:pressure-operator" },
  pressureSolve: { id: "pressure-solve", label: "brick:pressure-solve" },
  pressureProject: { id: "velocity-projection", label: "brick:pressure-project" },
  // Part 2, F's tail, P's couple and E's step end.
  tailExtend: { id: "velocity-extrapolation", label: "brick:tail-extend" },
  tailParticles: { id: "velocity-advection", label: "brick:tail-particles" },
  couple: { id: "svo-rigid", label: "brick:couple" },
  stepEnd: { id: "frame-control", label: "brick:step-end" },
  // The rigid integrator's own command buffer, after part 2 (bodies-host.ts integrate).
  rigid: { id: "svo-rigid", label: "brick:rigid" },
} as const satisfies Record<string, GPUTimestampPhase>;

export type BrickPhase = typeof BRICK_PHASE[keyof typeof BRICK_PHASE];

/** A run of a part's placements under one phase: placements [previous end, end). */
export interface BrickSeam { readonly phase: BrickPhase; readonly end: number }

/** A stage list's own seams (BrickStages.seams): consecutive runs of `count` placements, in order. */
export interface BrickStageSeam { readonly phase: BrickPhase; readonly count: number }

/** Concatenate phased runs into a placement list and its seams. Empty runs close no seam. */
export function seamed<T>(runs: readonly (readonly [BrickPhase, readonly T[]])[]): { readonly list: T[]; readonly seams: BrickSeam[] } {
  const list: T[] = [], seams: BrickSeam[] = [];
  for (const [phase, items] of runs) {
    if (items.length === 0) continue;
    list.push(...items);
    const last = seams.at(-1);
    if (last && last.phase === phase) seams[seams.length - 1] = { phase, end: list.length };
    else seams.push({ phase, end: list.length });
  }
  return { list, seams };
}

/**
 * A stage list split by its declared seams, or the whole list under `fallback`. The counts must cover the list
 * exactly: a stage set whose seams drift from its lists fails here, at the schedule, not as a mislabelled figure.
 */
export function stageRuns<T>(name: string, list: readonly T[], fallback: BrickPhase, seams?: readonly BrickStageSeam[]): (readonly [BrickPhase, readonly T[]])[] {
  if (!seams) return [[fallback, list]];
  const total = seams.reduce((sum, seam) => sum + seam.count, 0);
  if (total !== list.length || seams.some(seam => !(Number.isInteger(seam.count) && seam.count >= 0)))
    throw new Error(`Brick engine: the ${name} seams cover ${total} placements of ${list.length}`);
  let at = 0;
  return seams.map(seam => { const run = list.slice(at, at + seam.count); at += seam.count; return [seam.phase, run] as const; });
}
