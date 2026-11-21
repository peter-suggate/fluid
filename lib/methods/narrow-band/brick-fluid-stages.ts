/**
 * LANE F OWNS THIS FILE. Lane F's stages on the brick engine (brick-stages.ts, brick-abi.md §6.2).
 *
 * `fluidStages` is the stage set: the particles (particles-kernels.wgsl.ts), the level set
 * (levelset-kernels.wgsl.ts) and the velocity (velocity-kernels.wgsl.ts), placed in the old frame's order.
 * `brickFluidStages` (the registration brick-method.ts reads) stays undefined until the dam gates pass on it
 * (no migration before the gates: the stage set runs from the gate runners only).
 *
 * Schedule:
 * - part 1, move: pLiquid (the liquid cells of the opening φ, pSupported's), pMove (a record a lane); rebin: pScan, pScatter, pTiles.
 * - part 2, levelSet: vGauge (the volume control's gauge of the opening φ), lInit, lAdvect, lSplat, lUnion, lTiles, the round-1 dilation (air seed, occupancy; tiered, lDilateC then lDilate's three passes), lClear
 *   (clearAir with the crossing cells), the round-2 dilation (crossings, activity), lNear (x, y, z), lDistance,
 *   lPublish, then pSeed (the seeds of this step, after the bins, outside every run).
 * - part 2, velocity: uAdvect (and the gauge's rate into FH.RATE), pDeposit, uViscous (a viscous fluid),
 *   uTransfer (US for the operator).
 * - part 2, tail (after project): uExtend, pArgs (the live records' arguments, after pSeed: the tail's and the next
 *   step's move and scatter), pJoin (the census's join bits, before pDelta overwrites `before`), pDelta.
 * - boot: the level set's analysis of the boot φ (no transport), then vGauge's capture of the budget.
 * - kick (E's boot slot, after boot): force uKick (US = U + g dt/2); E's operator, solve and project; after: uExtend
 *   (and tiered, the caches); E's TRIG-only census of the kicked state; seed: the initial seeding (from the kicked
 *   US), then the rebin of the seeds into bank 0 (cfg.z = 1: the move files each record unchanged).
 *
 * Tiered (fluidStagesAt(true)): uAdvect, uViscous, uTransfer, uKick and vGauge run over FC_G1 (F_ALL's bricks, then
 * C_ALL's pages), their 4h faces or tiles in the h pass; lane E's 4h corner kernels (levelset-coarse.wgsl.ts) and the
 * 4h extension (uExt4Seed, uExt4, before uExtend) run on C_ALL.
 *
 * Banks: a step of parity y moves bank y into bank y ^ 1 (part 1); part 2 reads bank y ^ 1. The boot seeds into
 * bank 1 and its rebin (cfg.z = 1) files bank 1 into bank 0, the bank step 0 (parity 0) moves.
 */
import { ARGS, LIST } from "./brick-abi";
import type { BrickDispatch, BrickKernel, BrickPlacement } from "./brick-kernel";
import type { BrickStages, BrickStagesFactory } from "./brick-stages";
import { fluidLayout } from "./fluid-grid.wgsl";
import { LC_GROUPS, levelSetCoarseKernels } from "./levelset-coarse.wgsl";
import { levelSetKernels } from "./levelset-kernels.wgsl";
import { NBF_PARTICLE_SETTINGS, nbfParticleSupported } from "./particles-formulas.wgsl";
import { P_ARGS_RECORDS, P_ENERGY_BUFFER, particleKernels } from "./particles-kernels.wgsl";
import { nbfVelocitySupported } from "./velocity-formulas.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

/** The stage set at a tier mode: all h (tiered false), or reading through the tier owners and writing only owned
 * storage (tiered: the engine then runs the census and the transfer, brick-abi.md §2.5). */
export const fluidStagesAt = (tiered: boolean): BrickStagesFactory => context => {
  const L = context.layout, F = fluidLayout(L, { tiered });
  nbfParticleSupported(context.values);
  // The stage context carries no wall mode yet (lane E): the brick scenes are free-slip boxes.
  if (context.fluid) nbfVelocitySupported({ viscosity: context.fluid.viscosity_Pa_s, surfaceTension: context.fluid.surfaceTension_N_m, noSlip: false });
  // Volume control (Kim et al. 2007): the method's control time; 0 turns it off.
  const seconds = context.values.volumeControlSeconds;
  if (typeof seconds !== "number" || !(seconds >= 0)) throw new Error(`NB-FLIP brick engine: volumeControlSeconds ${String(seconds)} is not a control time`);
  const physics = { viscosity_Pa_s: context.fluid?.viscosity_Pa_s ?? 0, density_kg_m3: context.fluid?.density_kg_m3 ?? 1000, surfaceTension_N_m: context.fluid?.surfaceTension_N_m ?? 0, statics: context.statics === true, volumeControl_1_s: seconds > 0 ? 1 / seconds : 0 };
  // Statics (lane E): the scene binds a static voxel mask, so pSolid and phiSolid run (brick-solid-walk.wgsl.ts).
  const statics = context.statics === true;
  const p = particleKernels(L, F, { bandDepth: context.particles?.bandDepth ?? NBF_PARTICLE_SETTINGS.particleBandDepth, statics, ...(context.staticWalls ? { staticWalls: context.staticWalls } : {}) }), l = levelSetKernels(L, F, { statics, ...(context.staticWalls ? { staticWalls: context.staticWalls } : {}) }), v = velocityKernels(L, F, physics);
  const list = (k: number, g: "G1" | "G8" | "G64" | "X8"): BrickDispatch => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
  const all = list(LIST.F_ALL, "G1"), all8 = list(LIST.F_ALL, "G8"), pages = list(LIST.C_ALL, "G1");
  // Tiered, a pass over both tiers (the FC list: F_ALL's bricks, then C_ALL's pages, a workgroup each): the kernels
  // that hold a C page body (uAdvect, uTransfer, uKick, vGauge) run their 4h faces or tiles in their h pass.
  const both: BrickDispatch = tiered ? { indirect: ARGS.FC_G1 } : all;
  // Still-tank A: the passes that skip the STILL pages (F_ALL's bricks, then C_ACT's pages; C_ACT's pages).
  const bothAct: BrickDispatch = tiered ? { indirect: ARGS.FCA_G1 } : all, pagesAct = list(LIST.C_ACT, "G1");
  const c = tiered ? levelSetCoarseKernels(L, F) : undefined;
  const coarse = (kernel: keyof NonNullable<typeof c>): BrickPlacement[] => c ? [{ kernel: c[kernel], dispatch: list(LIST.C_ACT, LC_GROUPS[kernel]), z: 0, w: 0 }] : [];
  // The record passes run over the live records (pArgs's arguments), not the bank's capacity.
  const records: BrickDispatch = { indirect: P_ARGS_RECORDS };
  const at = (kernel: BrickKernel, dispatch: BrickDispatch, z = 0, w = 0): BrickPlacement => ({ kernel, dispatch, z, w });
  const args = (z: number, w: number) => at(p.pArgs, { direct: [1, 1, 1] }, z, w);
  const ext4 = tiered ? [at(v.uExt4Seed, pagesAct), at(v.uExt4, pagesAct)] : [];
  // Tiered, the samplers' caches of this step's opening state (after the head's transfer, before pMove; at the kick,
  // after the extension, before the seeding).
  const cache = v.uCache ? [at(v.uCache, both)] : [];
  // Tiered, lDilateC first: the round's sources and x and y passes at the tiles no F slot holds (DIL4).
  const dilate = (round: 1 | 2) => [...(l.lDilateC ? [at(l.lDilateC, pagesAct, 0, round)] : []), ...[0, 1, 2].map(pass => at(l.lDilate, all8, pass, round))];
  const near = [0, 1, 2].map(pass => at(l.lNear, all, pass));
  const rebin = (z: number) => [at(p.pScan, all, z), at(p.pScatter, records, z), at(p.pTiles, all, z)];
  // Tiered, lResolve after each writer of a non-owned vertex's value (fluid-grid.wgsl.ts FG_PHI_WGSL): the set
  // update (ownership, the transfer's C writes) before each schedule, lcCouple, lcClear and lcPublish (C PHI).
  const resolve = l.lResolve ? [at(l.lResolve, all)] : [];
  // The step's analysis re-resolves only what its clear changed (lResolveD); boot's follows lInit, which rewrites every
  // F word, so it resolves every slot.
  const analysisOf = (after: typeof resolve) => [at(l.lTiles, all), ...dilate(1), at(l.lClear, all), ...coarse("lcClear"), ...after, ...dilate(2), ...near, at(l.lDistance, all)];
  const analysis = analysisOf(l.lResolveD ? [at(l.lResolveD, all)] : resolve), bootAnalysis = analysisOf(resolve);
  const stages: BrickStages = {
    label: tiered ? "lane F (NB-FLIP tiered)" : "lane F (NB-FLIP all-h)", bringUp: false, tiered,
    // Tiered, the re-entry seed (pSeed cfg.w = 2) into the source bank (z = 1: pSrc ^ 1 at z = 1 is part 1's source).
    // Then the move's arguments (pArgs w = 0: the source's records, so the head's refined tiles' seeds move and bin
    // this step), which also empty the destination bank, every move (brick-particle-banks.test.ts).
    move: [...resolve, ...cache, at(p.pLiquid, all), ...(tiered ? [at(p.pSeed, all, 1, 2)] : []), args(0, 0), at(p.pSeparate, records), at(p.pMove, records), ...(p.pSolid ? [at(p.pSolid, records)] : [])],
    rebin: rebin(0),
    levelSet: [at(v.vGauge, both), at(l.lInit, all), at(l.lAdvect, all), ...(l.phiSolid ? [at(l.phiSolid, all)] : []), ...coarse("lcAdvect"), at(l.lSplat, all), at(l.lUnion, all), ...coarse("lcCouple"), ...resolve,
      ...analysis, ...coarse("lcRetire"), at(l.lPublish, all), ...coarse("lcPublish"), ...resolve, at(p.pSeed, all), ...(l.lWall ? [at(l.lWall, all)] : [])],
    velocity: [at(v.uAdvect, bothAct), at(v.pDeposit, all), ...(v.viscous ? [at(v.uViscous, bothAct)] : []), at(v.uTransfer, bothAct)],
    tail: [...ext4, at(v.uExtend, all), args(0, 1), at(p.pJoin, records), at(p.pDelta, records)],
    boot: [...resolve, at(l.lInit, all), ...bootAnalysis, at(v.vGauge, both, 0, 1)],
    // The old host's projected half kick (kickMixedFrame: U <- P(U + g dt/2) before the first frame), then the
    // extension and the initial seeding from the kicked, extended U (the old engine seeds in its first frame from
    // the kicked grid), filed into bank 0. The seeding is kick.seed: E's TRIG-only census of the kicked state runs
    // between after and seed, so the first seeding heats from the triggers of the state it seeds.
    kick: { force: [...resolve, at(v.uKick, both)], after: [...ext4, at(v.uExtend, all), ...cache], seed: [...resolve, at(p.pSeed, all, 0, 1), args(1, 0), at(p.pMove, records, 1), ...rebin(1), args(1, 1)] },
    buffers: { fcell: F.fcell.words, fdep: F.fdep.words, fpart: F.fpart.words, fmoved: F.fmoved.words, [P_ENERGY_BUFFER]: L.cap.particles, ...(tiered ? { fext4: F.fext4.words } : {}) },
  };
  return stages;
};

/** The all-h stage set. */
export const fluidStages: BrickStagesFactory = fluidStagesAt(false);

/** The registration brick-method.ts reads: the tiered set, opt-in only (engine "brick"; the default stays
 * "uniform" until the M2 gates pass), so the dam can be tried in the app. */
export const brickFluidStages: BrickStagesFactory | undefined = fluidStagesAt(true);
