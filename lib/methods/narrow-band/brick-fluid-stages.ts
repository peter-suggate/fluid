/**
 * LANE F OWNS THIS FILE. Lane F's stages on the brick engine (brick-stages.ts, brick-abi.md §6.2).
 *
 * `fluidStages` is the stage set: the particles (particles-kernels.wgsl.ts), the level set
 * (levelset-kernels.wgsl.ts) and the velocity (velocity-kernels.wgsl.ts), placed in the old frame's order.
 * `brickFluidStages` (the registration brick-method.ts reads) stays undefined until the dam gates pass on it
 * (no migration before the gates: the stage set runs from the gate runners only).
 *
 * Schedule:
 * - part 1, move: pMove (a record a lane); rebin: pScan, pScatter, pTiles.
 * - part 2, levelSet: vGauge (the volume control's gauge of the opening φ), lInit, lAdvect, lSplat, lUnion, lTiles, the round-1 dilation (air seed, occupancy), lClear
 *   (clearAir with the crossing cells), the round-2 dilation (crossings, activity), lNear (x, y, z), lDistance,
 *   lPublish, then pSeed (the seeds of this step, after the bins, outside every run).
 * - part 2, velocity: uAdvect (and the gauge's rate into FH.RATE), pDeposit, uViscous (a viscous fluid),
 *   uTransfer (US for the operator).
 * - part 2, tail (after project): uExtend, pJoin (the census's join bits, before pDelta overwrites `before`), pDelta.
 * - boot: the level set's analysis of the boot φ (no transport), then vGauge's capture of the budget.
 * - kick (E's boot slot, after boot): force uKick (US = U + g dt/2); E's operator, solve and project; after: uExtend,
 *   the initial seeding (from the kicked US), then the rebin of the seeds into bank 0 (cfg.z = 1: the move files
 *   each record unchanged).
 *
 * Banks: a step of parity y moves bank y into bank y ^ 1 (part 1); part 2 reads bank y ^ 1. The boot seeds into
 * bank 1 and its rebin (cfg.z = 1) files bank 1 into bank 0, the bank step 0 (parity 0) moves.
 */
import { ARGS, foldArgs, LIST } from "./brick-abi";
import type { BrickDispatch, BrickKernel, BrickPlacement } from "./brick-kernel";
import type { BrickStages, BrickStagesFactory } from "./brick-stages";
import { fluidLayout } from "./fluid-grid.wgsl";
import { levelSetKernels } from "./levelset-kernels.wgsl";
import { nbfParticleSupported } from "./particles-formulas.wgsl";
import { particleKernels } from "./particles-kernels.wgsl";
import { nbfVelocitySupported } from "./velocity-formulas.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

/** The stage set at a tier mode: all h (tiered false), or reading through the tier owners and writing only owned
 * storage (tiered: the engine then runs the census and the transfer, brick-abi.md §2.5). */
export const fluidStagesAt = (tiered: boolean): BrickStagesFactory => context => {
  const L = context.layout, F = fluidLayout(L, { tiered });
  nbfParticleSupported(context.values);
  // The stage context carries no wall mode yet (lane E): the brick scenes are free-slip boxes.
  // Tiered viscosity (the 4h Laplacian on the 4h cache, umLaplacian at width 4) is not ported.
  if (tiered && (context.fluid?.viscosity_Pa_s ?? 0) > 0) throw new Error("NB-FLIP brick engine: tiered viscosity is not ported (the 4h Laplacian)");
  if (context.fluid) nbfVelocitySupported({ viscosity: context.fluid.viscosity_Pa_s, surfaceTension: context.fluid.surfaceTension_N_m, noSlip: false });
  // Volume control (Kim et al. 2007): the method's control time; 0 turns it off.
  const seconds = context.values.volumeControlSeconds;
  if (typeof seconds !== "number" || !(seconds >= 0)) throw new Error(`NB-FLIP brick engine: volumeControlSeconds ${String(seconds)} is not a control time`);
  const physics = { viscosity_Pa_s: context.fluid?.viscosity_Pa_s ?? 0, density_kg_m3: context.fluid?.density_kg_m3 ?? 1000, volumeControl_1_s: seconds > 0 ? 1 / seconds : 0 };
  const p = particleKernels(L, F), l = levelSetKernels(L, F), v = velocityKernels(L, F, physics);
  const list = (k: number, g: "G1" | "G8" | "G64"): BrickDispatch => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
  const all = list(LIST.F_ALL, "G1"), all8 = list(LIST.F_ALL, "G8");
  const records: BrickDispatch = { direct: foldArgs(Math.ceil(L.cap.particles / 128)) };
  const at = (kernel: BrickKernel, dispatch: BrickDispatch, z = 0, w = 0): BrickPlacement => ({ kernel, dispatch, z, w });
  const dilate = (round: 1 | 2) => [0, 1, 2].map(pass => at(l.lDilate, all8, pass, round));
  const near = [0, 1, 2].map(pass => at(l.lNear, all, pass));
  const rebin = (z: number) => [at(p.pScan, all, z), at(p.pScatter, records, z), at(p.pTiles, all, z)];
  const analysis = [at(l.lTiles, all), ...dilate(1), at(l.lClear, all), ...dilate(2), ...near, at(l.lDistance, all)];
  const stages: BrickStages = {
    label: tiered ? "lane F (NB-FLIP tiered)" : "lane F (NB-FLIP all-h)", bringUp: false, tiered,
    move: [at(p.pMove, records)],
    rebin: rebin(0),
    levelSet: [at(v.vGauge, all), at(l.lInit, all), at(l.lAdvect, all), at(l.lSplat, all), at(l.lUnion, all), ...analysis, at(l.lPublish, all), at(p.pSeed, all)],
    velocity: [at(v.uAdvect, all), at(v.pDeposit, all), ...(v.viscous ? [at(v.uViscous, all)] : []), at(v.uTransfer, all)],
    tail: [at(v.uExtend, all), at(p.pJoin, records), at(p.pDelta, records)],
    boot: [at(l.lInit, all), ...analysis, at(v.vGauge, all, 0, 1)],
    // The old host's projected half kick (kickMixedFrame: U <- P(U + g dt/2) before the first frame), then the
    // extension and the initial seeding from the kicked, extended U (the old engine seeds in its first frame from
    // the kicked grid), filed into bank 0.
    kick: { force: [at(v.uKick, all)], after: [at(v.uExtend, all), at(p.pSeed, all, 0, 1), at(p.pMove, records, 1), ...rebin(1)] },
    buffers: { fcell: F.fcell.words, fdep: F.fdep.words, fpart: F.fpart.words, fmoved: F.fmoved.words },
  };
  return stages;
};

/** The all-h stage set. */
export const fluidStages: BrickStagesFactory = fluidStagesAt(false);

/** The registration brick-method.ts reads: undefined until the dam gates pass on fluidStages. */
export const brickFluidStages: BrickStagesFactory | undefined = undefined;
