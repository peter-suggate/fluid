/**
 * Wire format of the structured-authority receipts the engine reads back.
 *
 * The structured velocity/dynamics passes belong to the power backend, but the
 * words they publish are read by the shared engine: its per-step diagnostics
 * name the error bits, decode the projection-energy reduction, and report the
 * reject carry as a named stage. A published buffer's layout is not solver
 * physics, so it lives here where both the producing method and the engine
 * that reads it can name it — and the engine no longer has to import the
 * backend to understand a word the GPU already wrote.
 *
 * The producers import these back and re-export them, so the power package and
 * its tests still name each format in one place.
 */

/** Error bits `rejectSample`/`rejectVector` raise into the accepted authority. */
export const OCTREE_STRUCTURED_GPU_ERROR = Object.freeze({
  source: 1 << 0,
  capacity: 1 << 1,
  catalog: 1 << 2,
  neighbor: 1 << 3,
  reciprocity: 1 << 4,
  geometry: 1 << 5,
  carry: 1 << 6,
} as const);

export const STRUCTURED_PROJECTION_ENERGY_WORDS = 32;

export interface StructuredProjectionEnergySample {
  readonly epoch: number;
  readonly activeBank: 0 | 1;
  readonly familySampleCount: number;
  readonly startKineticEnergyProxy: number;
  readonly postAdvectionKineticEnergyProxy: number;
  readonly preProjectionKineticEnergyProxy: number;
  readonly postProjectionKineticEnergyProxy: number;
  readonly wetStartKineticEnergyProxy: number;
  readonly wetPostAdvectionKineticEnergyProxy: number;
  readonly wetPreProjectionKineticEnergyProxy: number;
  readonly wetPostProjectionKineticEnergyProxy: number;
  readonly wetFaceCount: number;
  /** Wet energies re-weighted by the face liquid fraction theta (1/pressure
   * scale): the variational face mass, the closest face-based analogue of the
   * physical rho/2 integral |u|^2 over liquid. The unweighted proxies above
   * charge a barely-wet surface face its full dual volume. */
  readonly wetStartThetaEnergyProxy: number;
  readonly wetPostAdvectionThetaEnergyProxy: number;
  readonly wetPreProjectionThetaEnergyProxy: number;
  readonly wetPostProjectionThetaEnergyProxy: number;
  readonly staggeredPathCount: number;
  readonly projectionEnergyRatio: number;
}

export interface StructuredProjectionEnergyDecode {
  readonly sample: StructuredProjectionEnergySample | null;
  readonly blocker: string | null;
}

/** Decode the one-generation four-stage reduction (start-of-step, post-
 * advection, post-force/pre-projection, post-projection). An initialized or
 * partial buffer never becomes a zero-energy observation. */
export function decodeStructuredProjectionEnergy(
  words: ArrayLike<number>,
): StructuredProjectionEnergyDecode {
  if (words.length < STRUCTURED_PROJECTION_ENERGY_WORDS) {
    return Object.freeze({ sample: null, blocker: "structured projection-energy report is truncated" });
  }
  const stages = [0, 1, 2, 3].map((stage) => ({
    flags: Number(words[8 * stage]) >>> 0,
    epochAndBank: Number(words[8 * stage + 1]) >>> 0,
    count: Number(words[8 * stage + 2]) >>> 0,
    energyBits: Number(words[8 * stage + 3]) >>> 0,
    wetCount: Number(words[8 * stage + 4]) >>> 0,
    wetEnergyBits: Number(words[8 * stage + 5]) >>> 0,
    wetThetaEnergyBits: Number(words[8 * stage + 6]) >>> 0,
    staggeredPathCount: Number(words[8 * stage + 7]) >>> 0,
  }));
  const failed = stages.find((stage) => stage.flags !== 0);
  if (failed) return Object.freeze({ sample: null,
    blocker: `structured projection-energy reduction failed with flags ${failed.flags}` });
  const epoch = stages[0]!.epochAndBank >>> 1;
  const activeBank = stages[0]!.epochAndBank & 1;
  if (epoch === 0 || stages.some((stage) => stage.epochAndBank !== stages[0]!.epochAndBank)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages are unpublished or generation-incoherent" });
  }
  if (stages[0]!.count === 0 || stages.some((stage) => stage.count !== stages[0]!.count)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages have incomplete family coverage" });
  }
  const bits = new Uint32Array(stages.flatMap((stage) =>
    [stage.energyBits, stage.wetEnergyBits, stage.wetThetaEnergyBits]));
  const energy = new Float32Array(bits.buffer);
  if (Array.from(energy).some((value) => !Number.isFinite(value) || value < 0)) {
    return Object.freeze({ sample: null,
      blocker: "structured projection-energy stages contain invalid energy" });
  }
  const pre = energy[6]!;
  const post = energy[9]!;
  return Object.freeze({ sample: Object.freeze({
    epoch, activeBank: activeBank as 0 | 1, familySampleCount: stages[0]!.count,
    startKineticEnergyProxy: energy[0]!,
    postAdvectionKineticEnergyProxy: energy[3]!,
    preProjectionKineticEnergyProxy: pre,
    postProjectionKineticEnergyProxy: post,
    wetStartKineticEnergyProxy: energy[1]!,
    wetPostAdvectionKineticEnergyProxy: energy[4]!,
    wetPreProjectionKineticEnergyProxy: energy[7]!,
    wetPostProjectionKineticEnergyProxy: energy[10]!,
    wetStartThetaEnergyProxy: energy[2]!,
    wetPostAdvectionThetaEnergyProxy: energy[5]!,
    wetPreProjectionThetaEnergyProxy: energy[8]!,
    wetPostProjectionThetaEnergyProxy: energy[11]!,
    wetFaceCount: stages[1]!.wetCount,
    staggeredPathCount: stages[1]!.staggeredPathCount,
    projectionEnergyRatio: pre === 0 ? 1 : post / pre,
  }), blocker: null });
}
