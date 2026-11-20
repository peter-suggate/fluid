/**
 * Lane P's pool MGPCG: the pressure solver at the slot of brick-abi.md §6.1.
 *
 * Conjugate gradients (Saad 2003, Iterative Methods for Sparse Linear
 * Systems, Alg. 9.1) preconditioned by one symmetric multigrid V-cycle
 * (McAdams, Sifakis & Teran 2010), over an aggregation hierarchy on the
 * engine's own slots. Its Float64 reference is pressure-model.ts with
 * `levels: 8`; tests/pressure-mgpcg.test.ts checks one GPU V-cycle and the
 * converged X against it.
 *
 * Hierarchy (level l aggregates c >> l; Galerkin A_c = P^T A P / OC, OC 1.6,
 * Braess 1995, as the seep):
 * - L0: the F bricks' cells, `pres` (X R Z D Q RHS DIAG K0-2), with two
 *   smoothing planes W0 and W1 in `pmgF`;
 * - L1 (2h): 64 rows an F slot, in `pmgF`;
 * - L2 (4h, a tile), L3 (8h, a brick), L4 (16h) and L5 (32h, a page): 512,
 *   64, 8 and 1 rows a C slot, in `pmgC`. Level 2 holds every tile of a
 *   page, so C rows merge there when the tiers land (design §2.7);
 * - L6 (64h) and L7 (128h): compact rows in `pmgTop`, rebuilt each step by
 *   `pmgCoarsenTop`. L7 is the coarsest, at most 64 rows, inverted once a
 *   step by Gauss-Jordan without pivoting (Trottenberg, Oosterlee & Schuller
 *   2001, sec. 2.5; Golub & Van Loan 2013, sec. 4.2): a pivot <= 0 latches
 *   FAIL.PIVOT, more than 64 rows FAIL.UNSUPPORTED. The level count is fixed
 *   when the solver is built, never at run time.
 *
 * Smoothers (pressure-model.ts):
 * - L0: the staggered block Schwarz of Smith, Bjorstad & Gropp 1996 (ch. 1-2),
 *   as the seep's shiftSmoothKernel: red-black Gauss-Seidel over the bricks,
 *   then over the 8^3 blocks shifted by (4, 0, 4), then the bricks, each block
 *   staged as a 10^3 tile with its halo frozen at the pass's entry. A shifted
 *   block is run by the first liquid brick of its four (x fastest).
 * - L1-L6: block red-black Gauss-Seidel over blocks of side max(2, 8 >> l).
 * - Post-smoothing applies the adjoint (the same palindromic passes, each
 *   block's colours reversed), so the V-cycle is symmetric positive definite.
 *
 * PCG, one iteration a slot:
 * - The check runs before the V-cycle: `pmgCoarsenTop` opens slot 0 only
 *   when max|r0| is above tolerance, and each slot's `pmgCheck` opens the
 *   next one the same way.
 * - Every residual is the true one, r = b - A (x + alpha p), computed from
 *   x and p rather than by the recurrence r - alpha q (residual replacement
 *   at every iteration, van der Vorst & Ye 2000). The recurrence drifted on
 *   the arena: lane P1's finding, 2026-11-15. The x update itself is applied
 *   by the next slot's `pmgDirection` (before p changes) or by `pmgFinish`.
 * - q = A z + beta q (= A p), so p never needs a halo in its own pass.
 * - Dot products are fixed-order sums: each producer workgroup writes its
 *   partial to its own word and every reader folds the words in index order
 *   (pFixed), so they all see the same value and no float atomic orders it.
 *
 * Gates (P's args region, MG_GATE): F at 512 (the row bricks' G1 or 0), C at 515 (C_ALL's
 * G1 or 0), ONE at 518, the walls' WF and WC at 521 and 524, S at 527 (the shifted blocks,
 * pmgTop SB, for L0's S passes: a workgroup a block, listed once a step by pmgCoarsenTop) and FC at
 * 530 (C_ALL's pages then the row bricks: pmgDirection and pmgUpdate run their h and native halves
 * in one launch). Header words: MG_HEADER.
 *
 * The tolerance is a divergence bound in 1/s: the solve stops when
 * max|r| <= tol h^2, since r = -h^2 div (the bring-up CG's rule). A native
 * row's residual counts over 64, its band equivalent.
 *
 * Composite rows (pressure-composite.wgsl.ts; the Float64 reference is
 * pressure-composite-model.ts). The unknowns are the h rows (pres F) and the
 * native 4h rows (pres C, DIAG > 0). An h row's K[a] couples to the h row
 * c - e_a or to the native row of its tile; its KU[a] to the native row of
 * c + e_a's tile. A native row's K[a] couples to the native row T - e_a, and
 * its seams are its h neighbours' K and KU, gathered (pSeam*), so the matrix
 * is symmetric. A cell of a 4h-owned tile inside an F brick has DIAG 0, and
 * every reader takes its value from pres C (pNatAt), never from its F slot.
 * The hierarchy is the composite model's, A_c = W^-1 P^T A P W^-1 with W
 * sqrt(OC) on an aggregated row and 1 on a carried native row:
 * - L0 and L1 smooth the h rows and their 2h aggregates only. A native row is
 *   carried: 0 in pre-smoothing, its L2 EPOST in post-smoothing. An F1 row's
 *   link to a native row is the children's seams over sqrt(OC) (L1 KU holds
 *   the upper one).
 * - L2 is every tile: an h tile's aggregate, and a native row unscaled (its
 *   DIAG and native K; the link to an F2 tile below is that tile's L2 KU).
 *   A native row's L2 right side is its residual after the h pre-smoothing,
 *   r + sum_m ch(m) (e0(m) + e1(m/2)/sqrt(OC)). L2 smooths in a page kernel
 *   over the page's 2^3 tile blocks, and L2 -> L3 aggregates there.
 * - A native row's z is its L2 EPOST. Its PCG vectors (X R Z D Q) are pres C's,
 *   run by page kernels beside the brick kernels; their dots share pmgRed.
 */
import { ARGS, BRICK_STRIDE, BrickLayout, FOLD, F_CLASS, LIST, brickAbiWGSL } from "./brick-abi";
import { CFG, type BrickBinding, type BrickKernel, type BrickPlacement, declareBindings } from "./brick-kernel";
import type { PressureSolver } from "./brick-pressure-slot";
import { FINE_ORDER, OC } from "./pressure-model";
import { P_FLAG, P_WALL_C, pressureCompositeKernels, pressureResolveBox } from "./pressure-composite.wgsl";
import { NP_K } from "./pressure-rows.wgsl";
import { BODIES_RO, BODY_CUTS_RO, pbBrickBodiesWGSL, type BrickBodiesFrame } from "./bodies-binding.wgsl";
import { pbBodyCoupleWGSL } from "./bodies-shapes.wgsl";
import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";

const P_WALL0 = P_FLAG.WALL0, WALL_F = P_WALL_C.F.toFixed(1), WALL_C = P_WALL_C.C.toFixed(1);

export const MG_GATE = { F: ARGS.LANE_P, C: ARGS.LANE_P + 3, ONE: ARGS.LANE_P + 6, WF: ARGS.LANE_P + 9, WC: ARGS.LANE_P + 12, S: ARGS.LANE_P + 15, FC: ARGS.LANE_P + 18,
  /** The live lists' indirect arguments (pmgListArgs, from pressureRowsList's counts): HB, HP, ZL (pmgF HBL, HPL, ZLL). */
  HB: ARGS.LANE_P + 21, HP: ARGS.LANE_P + 24, ZL: ARGS.LANE_P + 27 } as const;
/** P's header words (set 128..191). */
export const MG_HEADER = {
  OPEN: 128, RMAX: 129, RZ: 130, ALPHA: 131, ITERATIONS: 132, CAPPED: 133, RMAX0: 134, TOTAL: 135,
  N5: 136, N6: 137, N7: 138, RLAST: 139, FLIPS: 140, WALLS: 141, DECIDES: 142, RMAXW: 143,
  /** The step's residual history in 1/s (max|r|/h^2): HIST the initial residual, HIST+1+k after slot k (up to 42 slots). */
  HIST: 144,
  /** The native rows' stop scale (f32, tol / nBound; the old root's nStopScale): the native rows' initial max|r| during
   * the setup (pmgCoarsenC's reduction), then the scale pmgCoarsenTop's gate writes for the slots. */
  NSCALE: 187,
  /** MGPCGOptions.audit: the solve's end against its wall state, 4 words: h wall faces whose state is not their final
   * ghost's complementarity (x + lift >= 0 attached), h wall faces, the same two for native rows. */
  AUDIT: 188,
} as const;
/** Slots the history words cover (144..186). */
export const MG_HIST_SLOTS = 42;
/**
 * The stopping tolerance, 1/s: max|div| <= 0.05 over the rows (a native row's over 64), the seep's
 * PRESSURE_TOLERANCE and the old engine's h band target (UNIFORM_PRESSURE_ARENA_RETURNS_RESIDUAL), an
 * absolute max-norm bound as TallCells' (Chentanez & Mueller 2011). The earlier 1e-3 sat below the float32
 * PCG's attainable accuracy on the dam (its true residual stalls at 2e-3 - 1e-2 1/s; Greenbaum 1997), so every
 * dam step ran to the cap.
 */
export const MG_TOLERANCE = 0.05;
/**
 * The walls' default policy: the build's decision at the warm start, a decision at the Jacobi guess (pmgWallsG, before
 * the setup, so it costs no restart), then one re-decision at the first slot's iterate, after which the system is
 * fixed and PCG converges on it. An inexact primal-dual active set (Hintermueller, Ito & Kunisch 2002, the semismooth
 * Newton step taken at an inexact iterate; Chentanez & Mueller 2012 decide between converged solves): each
 * re-decision restarts the Krylov space, and the restarts were the iteration driver. Tiered dam, 30 steps, steps
 * 16-30 (lane P, 2026-11-15; tools/probe-pressure-active-set.ts for the old engine), before the guess: every
 * converged solve deciding, 12.6 its a step and 0% of h wall faces inconsistent with the final x; the build alone,
 * 7.67 its and 15.5%; one re-decision within 10 / 100 / 1000 x tolerance, 9.67 / 8.27 / 7.87 its and 1.5 / 1.6 /
 * 1.5%. The old engine leaves 9-11% of its band wall faces (and up to 36% of native ones) inconsistent.
 * With the guess (24 steps, steps 13-24, the inconsistent share summed over the steps; 2026-11-16): within 1000 x
 * tolerance, 7.50 its (2 runs) and 1.1-1.5%, as the guess's first slot is within 50 1/s only in calm steps; the
 * build alone 6.17 its and 12.2%; the guess's decision alone 6.08 and 8.3%; the guess's decision then one at the
 * first slot (this policy, MG_WALL_LOOSE Infinity) 7.08 and 0.80%; the first slot without the guess's 7.08 (2 runs)
 * and 1.4-1.5%. Without the guess: 8.00 its with 3 steps capped at 8, and 1.26%.
 */
export const MG_WALL_DECISIONS = 1, MG_WALL_LOOSE = Infinity;
/** L0's Schwarz passes by default (MGPCGOptions.fineOrder). */
export const MG_FINE_ORDER: "BS" | "BSB" = FINE_ORDER;
/** Levels of the hierarchy (L0..L7) and the compact top's row caps. */
export const MG_LEVELS = 8;
export const MG_ROWS6 = 512, MG_ROWS7 = 64;
/** pmgTopCycle's workgroup vectors' budget, bytes (of WebGPU's 16 KB minimum maxComputeWorkgroupStorageSize). */
const MG_TOP_WG_BYTES = 16384;
const HS6 = 1024, HS7 = 256;
/**
 * Level channels: D, K0..2 (the lower couplings), B, EPRE, EPOST, KU0..2 (L1 and L2: an aggregate's
 * upper link to a native row). L5's KU0 holds the page's native row count. L2 alone has NC channels: the
 * native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeGS) keep N0 (L0 pre), R1 (the
 * row's L1 RHS), N1 (L1 pre), N1P (L1 post) and N0P (L0 post, the row's z) at the row's L2 index.
 */
const CH = { D: 0, K: 1, B: 4, EPRE: 5, EPOST: 6, KU: 7, N: 10, N0: 10, R1: 11, N1: 12, N1P: 13, N0P: 14, NC: 15 } as const;

/**
 * pmgTop's words. RBN, RB and RBF are the row bricks: the F slots holding an h row (RBF[slot] 1, set by pmgSetupF
 * over F_ALL), listed in RB (RBN of them) by pmgCoarsenTop. P owns this list and reads no F_LIQ: on the tiered
 * dam 613 of 3492 F bricks hold an h row (the rest are the native tiles' bricks).
 */
export function mgTopLayout(C: number, F: number) {
  let at = 0;
  const take = (n: number) => { const o = at; at += n; return o; };
  const r6 = MG_ROWS6, r7 = MG_ROWS7;
  const t = {
    SLOT5: take(C), IDX5: take(C), PAR5: take(C), NBR5: take(6 * C), C5: take(6 * C), D5: take(C), B5: take(C), E5: take(C), F5: take(C), P5: take(C), R5: take(C),
    KEY6: take(r6), PAR6: take(r6), NBR6: take(6 * r6), C6: take(6 * r6), D6: take(r6), B6: take(r6), E6: take(r6), F6: take(r6), P6: take(r6), R6: take(r6), CH6: take(8 * r6),
    KEY7: take(r7), CH7: take(8 * r7), B7: take(r7), E7: take(r7), INV: take(r7 * r7),
    RBN: take(1), RB: take(F), RBF: take(F), SBN: take(1), SB: take(4 * F),
    /** The live lists' counts HB, HP, ZL, RL (pressureRowsList appends, pmgListArgs moves them on and zeroes them). */
    HBN: take(4),
    /**
     * The live lists' certificates, a slot each (pressureRowsList): the brick's key and parent C slot at its last
     * pass, and step + 2 of the last step it left with its zero rows written (ZS) or its FLAGS bits 7-10 clear (FS),
     * 0 when its rows or flags may since be written (it was on HB / HP); the next step (step + 1) finds them so.
     */
    ZK: take(F), ZC: take(F), ZS: take(F), FS: take(F),
    /** pmgPressureDelta's history, a C slot each step parity: the page key and step + 1 of its last publish. */
    PHW: take(4 * C),
  };
  return { ...t, words: at };
}

/** pmgF's and pmgC's words (brick-abi.md §2.3: "MG levels above 4h are P's own"). */
export function mgLayout(L: BrickLayout) {
  const F = L.cap.f, C = L.cap.c;
  const W0 = 0, W1 = BRICK_STRIDE * F, L1 = 2 * BRICK_STRIDE * F;
  const L2 = 0, L3 = L2 + CH.NC * 512 * C, L4 = L3 + CH.N * 64 * C, L5 = L4 + CH.N * 8 * C;
  // PDELTA (lane E's region heat reads it; brick-abi.md "PDELTA"): 8 words a C slot, the f32 region pressure change of
  // pmgPressureDelta, Delta(p / rho) in m^2/s^2, +Inf unknown. PH: its history, a tile's p / rho a C slot each step parity.
  const PDELTA = L5 + CH.N * C, PH = PDELTA + 8 * C;
  // The seam list (pmgCoarsenC, once a step; every native pass's seam sum reads it), exact f32 integers: a page's
  // native rows' seam faces in tile then face order, SEAML (tile << 3 | face) and SEAMF (the F slot holding the h tile's
  // seam layer across the face), up to 6 a tile; SEAMT a tile's first entry << 3 | its count; SEAMN the page's count.
  const SEAMT = PH + 2 * 512 * C, SEAML = SEAMT + 512 * C, SEAMF = SEAML + 3072 * C, SEAMN = SEAMF + 3072 * C;
  // STG (pmgF) and QSTG (pmgC): a brick's 27-neighbour stage (pStage: F slots, their pages' C slots, native-tile masks)
  // and a page's 27 neighbour pages (qStage), as exact f32 integers (-1 none), written once a step by pmgSetupF over
  // F_ALL and pmgCoarsenC over C_ALL; every slot pass then stages in one load instead of the directory chain.
  // RKU (pmgF): an h cell's upper couplings as pRow reads them (KU[a] to a native row, else the h neighbour's K[a]),
  // 3 x 512 a slot, written once a step by pmgSetupF over F_ALL's live bricks; every L0 pass's pRow then loads its
  // row in one round of independent loads instead of the neighbour slot's chain.
  // FC (pmgF): the face cache of pressure-composite.wgsl.ts (rowsF writes, projectF reads), 7 x BRICK_STRIDE a slot.
  // The live lists (pressureRowsList over F_ALL, once a step, before the solve's setup; mgLive), F slots as exact f32
  // integers in append order: HBL the bricks with an h-owned tile (the h rows' bricks: tileRoots, rowsF, setupF and
  // finish run over it), HPL those with an h-owned tile in the brick or across its lower faces (an h face stored
  // there: projectF runs over it), ZLL the bricks off HB whose zero rows are not certified (pressureRowsZero), RLL
  // those off HP whose FLAGS bits are not (pressureProjectRest), and RLN RL's count. DRY, a word an F slot:
  // projectF's flag of an HP brick with a dry seam face (projectDry's work, pressure-composite.wgsl.ts).
  const STG = L1 + CH.N * 64 * F, RKU = STG + 81 * F, FC = RKU + 3 * BRICK_STRIDE * F, QSTG = SEAMN + C;
  const HBL = FC + 7 * BRICK_STRIDE * F, HPL = HBL + F, ZLL = HPL + F, RLL = ZLL + F, RLN = RLL + F, DRY = RLN + 1;
  return { W0, W1, L1, STG, RKU, FC, HBL, HPL, ZLL, RLL, RLN, DRY, fWords: DRY + F, L2, L3, L4, L5, PDELTA, PH, SEAMT, SEAML, SEAMF, SEAMN, QSTG, cWords: QSTG + 27 * C, top: mgTopLayout(C, F) };
}

const setRW: BrickBinding = { name: "set", kind: "rw", type: "array<atomic<u32>>" };
const dirRO: BrickBinding = { name: "dir", kind: "ro", type: "array<u32>" };
const presRW: BrickBinding = { name: "pres", kind: "rw", type: "array<f32>" };
const fRW: BrickBinding = { name: "pmgF", kind: "rw", type: "array<f32>" };
const cRW: BrickBinding = { name: "pmgC", kind: "rw", type: "array<f32>" };
const topRW: BrickBinding = { name: "pmgTop", kind: "rw", type: "array<u32>" };
const redRW: BrickBinding = { name: "pmgRed", kind: "rw", type: "array<f32>" };
const argsRW: BrickBinding = { name: "args", kind: "rw", type: "array<u32>" };
const phiRW: BrickBinding = { name: "phi", kind: "rw", type: "array<u32>" };
const velRO: BrickBinding = { name: "vel", kind: "ro", type: "array<f32>" };

export interface MGPCGOptions {
  /** Encoded slots (PCG iterations) at most a step; brick-abi.md §6.1 says 8. */
  readonly slots?: number;
  /** Divergence tolerance, 1/s. */
  readonly tolerance?: number;
  /** Red-black sweeps inside a level 0 Schwarz block (pressure-model.ts fineSweeps). */
  readonly fineSweeps?: number;
  /**
   * L0's Schwarz passes (pressure-model.ts fineOrder): "BS" the bricks then the shifted blocks, the adjoint (shifted
   * blocks, bricks) after the coarse correction, as the seep's pcgUpdate/pcgShiftDown | pcgShiftUp/pcgPrecond; "BSB"
   * the palindrome bricks, shifted blocks, bricks on both sides.
   */
  readonly fineOrder?: "BS" | "BSB";
  /** Red-black sweeps inside a coarse block (pressure-model.ts coarseSweeps). */
  readonly coarseSweeps?: number;
  /** Red-black sweeps of the native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeSweeps). */
  readonly nativeSweeps?: number;
  /**
   * The separating walls' outer loop (the composite rows' lattice walls, FLAGS bits 8-10): each converged solve
   * re-decides every wall face from its iterate and restarts PCG when a state changes. Off for rows without the
   * wall state (the pool tests' synthetic systems).
   */
  readonly walls?: boolean;
  /**
   * The walls' re-decisions a solve may take after the build's (MG_WALL_DECISIONS, default 1; Infinity decides at
   * every converged solve until no state changes, the exact complementarity; 0 keeps the build's state, start + lift
   * >= 0, as the old engine freezes its state at the Krylov start).
   */
  readonly wallDecisions?: number;
  /** A re-decision is taken at the first slot whose max|r| is within wallLoose x tolerance (MG_WALL_LOOSE; Infinity, the first slot). */
  readonly wallLoose?: number;
  /**
   * The Jacobi guess (default true): before the setup, every h row's start takes one Jacobi step, x0 + D^-1 (b - A x0)
   * (Saad 2003 §4.1; the seep's pcgGuess). Native rows keep their start.
   */
  readonly guess?: boolean;
  /**
   * With the guess, the walls and a re-decision left (wallDecisions > 0): the build's wall state is decided again at the
   * guess, before the setup (pmgWallsG; default true). It does not count as a re-decision.
   */
  readonly guessWalls?: boolean;
  /** Count the end's inconsistent wall faces into MG_HEADER.AUDIT (a diagnostic pass in the finish). */
  readonly audit?: boolean;
  /**
   * The live lists (default mgLive(L)): the h rows' kernels over HB / HP instead of F_ALL. The engine pairs a solver
   * with a face cache with pressureOperatorKernels' live operator and project, so false is for tests (the F_ALL path).
   */
  readonly live?: boolean;
  /**
   * The cuts (default off; M5 design §5): the h rows read the K slots' FRAC / AP / W (pressure-composite.wgsl.ts), and
   * the walls' re-decision takes a wall face's coupling (its aperture V) and kind from the face cache, skipping a
   * closed one. Off, every kernel's text is the cut-free solver's.
   */
  readonly cuts?: boolean;
  /**
   * The monolithic rigid coupling (lane P, 2026-11-17; Batty, Bertails & Bridson 2007, "A fast variational framework
   * for accurate solid-fluid coupling", the rigid bodies' rows): the scene's bodies' frame, or off. The explicit
   * coupling (couple's impulse rho J^T X from a solve that saw the bodies at V_n) has the added-mass mode
   * lambda = -m_a / m, unstable for a light flat hull at any dt (the paddleboat's -1.55 a step). On, every dynamic body
   * (material.z 1, inverseMassInertia.x > 0) is implicit in the solve: A' = A + (1/h) J M~ J^T, J's row j_c =
   * [a_c; (r_c - X) x a_c] couple's face areas a_c (pbCoupleBody, a liquid row with FRAC > 1e-5), M~ =
   * diag(imi.x I, R diag(imi.yzw) R^T) the integrator's (rigidKicked: dV = imi.x / rho I, dL = I, w = I^-1 L), and the
   * RHS's V_n becomes V* = V_n + g dt (RHS -= (1/h) j_c.(g dt, 0)). Couple's impulse from the final X then gives the
   * integrator V_n+1 = V* + M^-1 rho J^T X, the velocity the solve's divergence assumed. The multigrid stays on A (a
   * valid SPD preconditioner of A'), and the rank-6-a-body term costs one J^T y reduction a slot (pmgBodyZ, pmgBodyFold).
   * Fixed, held and motor-attached bodies stay explicit (the stamp's W). Needs the cuts (a body brick's K slot).
   * Undefined, every kernel's text is the solver's without bodies.
   */
  readonly bodies?: BrickBodiesFrame;
}

/**
 * The live lists need one indirect dimension (an append's count is the x word): F's capacity within the fold.
 * Above it the h rows' kernels stay on F_ALL.
 */
export const mgLive = (L: BrickLayout) => L.cap.f <= FOLD;

/** The live path's parts, for the composite (brick-pressure-solver.ts) to place around its native rows. */
export interface MGPCGLive {
  /** The lists' arguments from rowsList's counts and the zero rows off HB (the setup's head). */
  readonly args: readonly BrickPlacement[];
  /** tileRoots over HB (the composite's: before rowsF). */
  readonly tileRoots: readonly BrickPlacement[];
  /** rowsF over HB (the setup's head). */
  readonly rows: readonly BrickPlacement[];
  /** projectF over HP (the finish's tail). */
  readonly project: readonly BrickPlacement[];
  /** pressureRowsList: the engine's operator (pressureOperatorKernels), F_ALL G1. */
  readonly rowsList: BrickKernel;
}

/** The kick's pressure in a step's dt units (boot only, after the kick's solves): X is in its solve's dt units
 * (U = US - grad X / h), and the kick solves at dt/2 (the old host's kickMixedFrame), so step 1's warm starts (poStart's
 * X, poTileStartR's C X) and its seam heads read twice the kick's X: the old arena's starts are dt-free pressures. Every
 * F and C X word of the pools, zero rows staying zero; a lane a word, the groups in rows of 65535 (any pool size). */
export function pressureKickScaleKernel(L: BrickLayout): BrickKernel {
  const f = L.cap.f * BRICK_STRIDE, c = L.cap.c * BRICK_STRIDE;
  return { label: "pmgKickScale", entry: "pmgKickScale", bindings: [CFG, setRW, presRW], code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true } }), declareBindings([CFG, setRW, presRW]), /* wgsl */`
@compute @workgroup_size(256) fn pmgKickScale(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let i=256u*(wid.x+65535u*wid.y)+lid;
  if(i<${f}u){pres[NB_PRES_F_X+i]*=2.0;}
  else if(i<${f + c}u){pres[NB_PRES_C_X+i-${f}u]*=2.0;}
}`].join("\n") };
}
export const pressureKickScaleGroups = (L: BrickLayout): [number, number, number] => {
  const g = Math.ceil((L.cap.f + L.cap.c) * BRICK_STRIDE / 256);
  return [Math.min(g, 65535), Math.ceil(g / 65535), 1];
};

export function poolMGPCG(L: BrickLayout, options: MGPCGOptions = {}): PressureSolver & { readonly publish: readonly BrickPlacement[]; readonly live?: MGPCGLive; readonly core: { readonly setup: readonly BrickPlacement[]; readonly finish: readonly BrickPlacement[] } } {
  const slots = options.slots ?? 8, tol = options.tolerance ?? MG_TOLERANCE, bsb = (options.fineOrder ?? MG_FINE_ORDER) === "BSB";
  if (options.fineOrder !== undefined && options.fineOrder !== "BS" && options.fineOrder !== "BSB") throw new Error(`Pool MGPCG: fine order ${String(options.fineOrder)}`);
  // The h rows' pre-smoothed plane: W0 after BSB's last brick pass (pmgPreU), W1 after BS's shifted pass (pmgPreS).
  const PRE = bsb ? "P_W0" : "P_W1";
  const fineSweeps = options.fineSweeps ?? 3, coarseSweeps = options.coarseSweeps ?? 2, nativeSweeps = options.nativeSweeps ?? 2, walls = options.walls ?? false, guessOn = options.guess ?? true, guessWalls0 = options.guessWalls ?? true;
  if (!Number.isInteger(slots) || slots < 1) throw new Error(`Pool MGPCG: ${slots} slots`);
  const tolBits = new Uint32Array(new Float32Array([tol]).buffer)[0]!;
  const wallDecisions = Math.min(options.wallDecisions ?? MG_WALL_DECISIONS, 0xffffffff), wallLoose = options.wallLoose ?? MG_WALL_LOOSE, audit = options.audit ?? false, cuts = options.cuts ?? L.cap.k > 0;
  if (!(wallDecisions >= 0) || !(wallLoose >= 1)) throw new Error(`Pool MGPCG: walls policy ${wallDecisions}/${wallLoose}`);
  const guessWalls = guessOn && walls && wallDecisions > 0 && guessWalls0;
  const M = mgLayout(L), T = M.top;
  const u = (n: number) => `${n >>> 0}u`;
  const live = options.live ?? mgLive(L);
  const bodies = options.bodies;
  if (bodies && !cuts) throw new Error("Pool MGPCG: the monolithic rigid coupling needs the cuts (a body brick's K slot)");
  if (bodies && !(bodies.h_m > 0)) throw new Error(`Pool MGPCG: body frame h ${bodies.h_m}`);
  if (live && !mgLive(L)) throw new Error(`Pool MGPCG: live lists with ${L.cap.f} F slots (fold ${FOLD})`);
  // A brick kernel's head over F_ALL, or (live) over HB's exact arguments: no count test.
  const fHead = (wcount = "") => live
    ? `${wcount ? `if(lid==0u){${wcount}}` : ""}let g=nbGroup(wid);let slot=u32(pmgF[P_HBL+g]);`
    : `if(lid==0u){wCount=nbListCount(NB_LIST_F_ALL);${wcount}}\n  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}\n  let slot=nbListSlot(NB_LIST_F_ALL,g);`;

  const COMMON = [
    ...Object.entries(MG_HEADER).map(([n, v]) => `const PH_${n}:u32=${u(v)};`),
    ...Object.entries(MG_GATE).map(([n, v]) => `const P_GATE_${n}:u32=${u(v)};`),
    ...Object.entries(CH).map(([n, v]) => `const CH_${n}:u32=${u(v)};`),
    ...Object.entries(T).filter(([n]) => n !== "words").map(([n, v]) => `const T_${n}:u32=${u(v)};`),
    `const P_OC:f32=${OC.toFixed(6)};const P_SOC:f32=${Math.sqrt(OC).toFixed(7)};`,
    `const P_W0:u32=${u(M.W0)};const P_W1:u32=${u(M.W1)};const P_L1:u32=${u(M.L1)};`,
    `const P_SEAMT:u32=${u(M.SEAMT)};const P_SEAML:u32=${u(M.SEAML)};const P_SEAMF:u32=${u(M.SEAMF)};const P_SEAMN:u32=${u(M.SEAMN)};`,
    `const P_STG:u32=${u(M.STG)};const P_RKU:u32=${u(M.RKU)};const P_QSTG:u32=${u(M.QSTG)};`,
    `${cuts ? `const P_FC:u32=${u(M.FC)};` : ""}const P_HBL:u32=${u(M.HBL)};const P_HPL:u32=${u(M.HPL)};const P_ZLL:u32=${u(M.ZLL)};const P_RLL:u32=${u(M.RLL)};const P_RLN:u32=${u(M.RLN)};`,
    `const P_L2:u32=${u(M.L2)};const P_L3:u32=${u(M.L3)};const P_L4:u32=${u(M.L4)};const P_L5:u32=${u(M.L5)};`,
    `const P_FS2:u32=${u(2 * fineSweeps)};const P_CS2:u32=${u(2 * coarseSweeps)};const P_NS2:u32=${u(2 * nativeSweeps)};const P_SLOTS:u32=${u(slots)};`,
    `const P_ROWS6:u32=${u(MG_ROWS6)};const P_ROWS7:u32=${u(MG_ROWS7)};`,
    /* wgsl */`
fn pUnit(a:u32)->vec3i{var e=vec3i(0);e[a]=1;return e;}
fn pPar(c:vec3i)->u32{return u32((c.x+c.y+c.z)&1);}
// Local indices: 4 a side (L1 in a brick, L3 in a page), 2 a side (L4 in a page, children).
fn pI1(c:vec3i)->u32{return u32((c.x&3)|((c.y&3)<<2u)|((c.z&3)<<4u));}
fn pC1(i:u32)->vec3i{return vec3i(i32(i&3u),i32((i>>2u)&3u),i32(i>>4u));}
fn pI4(c:vec3i)->u32{return u32((c.x&1)|((c.y&1)<<1u)|((c.z&1)<<2u));}
fn pC4(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32(i>>2u));}
fn pL1(ch:u32,s:u32,i:u32)->u32{return P_L1+(ch*NB_FCAP+s)*64u+i;}
fn pL2(ch:u32,s:u32,i:u32)->u32{return P_L2+(ch*NB_CCAP+s)*512u+i;}
fn pL3(ch:u32,s:u32,i:u32)->u32{return P_L3+(ch*NB_CCAP+s)*64u+i;}
fn pL4(ch:u32,s:u32,i:u32)->u32{return P_L4+(ch*NB_CCAP+s)*8u+i;}
// L4's spare channel (L4 has no KU): pmgDown3's parts of the page's L4 residual, pmgTopCycle's L5 RHS.
const CH4R:u32=${u(CH.KU)};
fn pL5(ch:u32,s:u32)->u32{return P_L5+ch*NB_CCAP+s;}
`].join("\n");

  // Helpers by what they bind (WGSL resolves every function, used or not).
  const PRES = /* wgsl */`
fn pDiag(s:u32,l:u32)->f32{return pres[NB_PRES_F_DIAG+NB_STRIDE*s+l];}
fn pK(a:u32,s:u32,l:u32)->f32{return pres[NB_PRES_F_K0+NB_STRIDE*(s+a*NB_FCAP)+l];}
fn pKU(a:u32,s:u32,l:u32)->f32{return pres[NB_PRES_F_KU0+NB_STRIDE*(s+a*NB_FCAP)+l];}
fn pCK(a:u32,cs:u32,i:u32)->f32{return pres[NB_PRES_C_K0+NB_STRIDE*(cs+a*NB_CCAP)+i];}
fn pP(ch:u32,s:u32,l:u32)->f32{return pres[ch+NB_STRIDE*s+l];}
`;
  const PAGE = /* wgsl */`
  // The C slot of page p, or NONE: a page outside the lattice's bricks, or the
  // window's other page at the same index (the key must match). C records'
  // neighbour words are not maintained in M1, so pages are found by the directory.
  fn pPageSlot(p:vec3i)->u32{
    if(any(p<vec3i(0))||any(4*p>=NB_BRICKS)){return NB_NONE;}
    let cs=nbCSlotOfPage(p);if(cs==NB_NONE){return NB_NONE;}
    if(nbCRec(cs,NB_REC_KEY)!=nbKey(p)){return NB_NONE;}
    return cs;
  }
  fn pTileIn(t:vec3i)->bool{return all(t>=vec3i(0))&&all(4*t<NB_WALL_BOX);}
  // The C slot of tile t when it is 4h-owned (a native tile of a resident page), or NONE.
  fn pNativeTile(t:vec3i)->u32{
    if(!pTileIn(t)){return NB_NONE;}
    let cs=pPageSlot(t>>vec3u(3u));if(cs==NB_NONE){return NB_NONE;}
    return select(cs,NB_NONE,nbTileH(cs,nbLocal(t)));
  }
  `;
  // A brick's 27 neighbours, NONE unless a live F slot (every live slot holds rows: the operator's zero rows off the
  // liquid, so P reads no F_LIQ or LIQUID class); and each one's native tiles
  // (8 bits, pI4 order) with its page's C slot.
  const STAGE = /* wgsl */`
var<workgroup> pN:array<u32,27>;
var<workgroup> pNC:array<u32,27>;
var<workgroup> pNM:array<u32,27>;
fn pStage(slot:u32,lid:u32){
  if(lid<27u){
    let n=nbFRec(slot,lid);pN[lid]=select(NB_NONE,n,n<NB_SEAM);
    let bn=nbKeyCoord(nbFRec(slot,NB_REC_KEY))+nbNeighbourOffset(lid);
    var cs=NB_NONE;var m=0u;
    if(all(bn>=vec3i(0))){cs=pPageSlot(bn>>vec3u(2u));}
    if(cs!=NB_NONE){for(var j=0u;j<8u;j++){let t=2*bn+pC4(j);if(pTileIn(t)&&!nbTileH(cs,nbLocal(t))){m|=1u<<j;}}}
    pNC[lid]=cs;pNM[lid]=m;
  }
}
fn pSlotAt(b:vec3i,c:vec3i)->u32{let d=(c>>vec3u(3u))-b;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}return pN[nbNeighbourIndex(d)];}
// The C slot holding h cell c's native row (c in a 4h-owned tile), or NONE; the row is pres C's at nbLocal(c >> 2).
fn pNatAt(b:vec3i,c:vec3i)->u32{
  let d=(c>>vec3u(3u))-b;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}
  let i=nbNeighbourIndex(d);if(((pNM[i]>>pI4(c>>vec3u(2u)))&1u)==0u){return NB_NONE;}
  return pNC[i];
}
`;
  // A page's 27 neighbour pages (C slots), NONE otherwise; self at 13.
  // The stage cache (P_STG, P_QSTG): written by the setup's staging kernels, read by every later pass of the step.
  const STAGEK = /* wgsl */`
fn pStageW(slot:u32,lid:u32){
  if(lid<27u){let at=P_STG+81u*slot+lid;pmgF[at]=select(f32(pN[lid]),-1.0,pN[lid]==NB_NONE);pmgF[at+27u]=select(f32(pNC[lid]),-1.0,pNC[lid]==NB_NONE);pmgF[at+54u]=f32(pNM[lid]);}
}
fn pStageK(slot:u32,lid:u32){
  if(lid<27u){let at=P_STG+81u*slot+lid;let n=pmgF[at];let c=pmgF[at+27u];pN[lid]=select(u32(n),NB_NONE,n<0.0);pNC[lid]=select(u32(c),NB_NONE,c<0.0);pNM[lid]=u32(pmgF[at+54u]);}
}`;
  const CSTAGEK = /* wgsl */`
fn qStageW(cs:u32,lid:u32){if(lid<27u){pmgC[P_QSTG+27u*cs+lid]=select(f32(qN[lid]),-1.0,qN[lid]==NB_NONE);}}
fn qStageK(cs:u32,lid:u32){if(lid<27u){let v=pmgC[P_QSTG+27u*cs+lid];qN[lid]=select(u32(v),NB_NONE,v<0.0);}}`;
  const CSTAGE = /* wgsl */`
var<workgroup> qN:array<u32,27>;
var<workgroup> wKids:u32;
fn qStage(cs:u32,lid:u32){if(lid<27u){qN[lid]=select(pPageSlot(nbKeyCoord(nbCRec(cs,NB_REC_KEY))+nbNeighbourOffset(lid)),cs,lid==13u);}}
// The C slot of the page holding a level cell c (cells a page side = 1 << sh), seen from page p.
fn qPage(p:vec3i,c:vec3i,sh:u32)->u32{let d=(c>>vec3u(sh))-p;if(any(d<vec3i(-1))||any(d>vec3i(1))){return NB_NONE;}return qN[nbNeighbourIndex(d)];}
`;
  // The CG scalars, summed in a fixed order (no float atomics, so the sum is the code's and never the arrivals'): a
  // producer workgroup g writes its pSum512 partial to its own word (R_ZF + g for pmgPostZ's row brick g, R_ZC + g for
  // pmgNativePost0's page g, R_Q + g for pmgDirection's workgroup g), and a consumer workgroup folds the words in index
  // order (pFixed). Every consumer workgroup runs the same fold on the same words, so it reads the same value; its
  // first row brick's workgroup stores it (pFixedWriter: R_RZ by pmgDirection, R_PQ by pmgUpdate) for the later kernels' reads. Lane E traced the
  // dam's run-to-run spread to the old CAS adds into 64 bins (alpha and beta moved in the last bits at step 0).
  const RED = /* wgsl */`
const R_RZ:u32=0u;const R_PQ:u32=1u;const R_ZF:u32=8u;const R_ZC:u32=${u(8 + L.cap.f)};const R_Q:u32=${u(8 + L.cap.f + L.cap.c)};
`;
  // Every lane: the sum of words base..base+n then base2..base2+n2, lane j folding j, j + 512, ... in order and
  // pSum512 folding the lanes. A 512-lane workgroup in uniform control flow (four barriers).
  const FIXED = /* wgsl */`
var<workgroup> wFix:f32;
fn pFixed(lid:u32,base:u32,n:u32,base2:u32,n2:u32)->f32{
  var s=0.0;for(var i=lid;i<n;i+=512u){s+=pmgRed[base+i];}for(var i=lid;i<n2;i+=512u){s+=pmgRed[base2+i];}
  let t=pSum512(lid,s);if(lid==0u){wFix=t;}return workgroupUniformLoad(&wFix);
}
// The workgroup of a C | F launch (pmgDirection, pmgUpdate) that stores the fold: the first row brick's, which always
// runs it (a page without native rows returns before it); with no row bricks, every page's (the same value).
fn pFixedWriter(g:u32,nC:u32,nF:u32)->bool{return select(true,g==nC,nF>0u);}
`;
  const REDUCE512 = /* wgsl */`
// Lane 0's sum of the workgroup's 512 values (other lanes get 0): 64 lanes fold 8 each, 8 lanes fold those 8 each,
// then lane 0 folds the 8: a chain of 24 loads behind three barriers, where lane 0's fold of 32 behind 32 lanes' of 16
// (48 loads) held the workgroup resident (a halving tree takes ten barriers). A next call writes wRed after this
// call's second barrier (the 64 lanes' reads before it), wRed2 after its first and wRed3 after its second.
var<workgroup> wRed:array<f32,512>;var<workgroup> wRed2:array<f32,64>;var<workgroup> wRed3:array<f32,8>;
fn pSum512(lid:u32,v:f32)->f32{
  wRed[lid]=v;workgroupBarrier();
  if(lid<64u){var s=0.0;for(var j=0u;j<8u;j++){s+=wRed[64u*j+lid];}wRed2[lid]=s;}
  workgroupBarrier();
  if(lid<8u){var s=0.0;for(var j=0u;j<8u;j++){s+=wRed2[8u*j+lid];}wRed3[lid]=s;}
  workgroupBarrier();
  var r=0.0;if(lid==0u){for(var j=0u;j<8u;j++){r+=wRed3[j];}}return r;
}
// Lane 0's max of the workgroup's 512 values, each >= 0 (other lanes get 0): a workgroup atomic max of the f32 bits
// (order-preserving for non-negative floats, so the value is the fold's exactly), two barriers and no serial fold
// (lane 0's fold of 32 behind 32 lanes' of 16 cost setupF 85 of its 455 us on the tiered dam, three calls a brick).
var<workgroup> wMaxA:atomic<u32>;
fn pMax512(lid:u32,v:f32)->f32{
  if(lid==0u){atomicStore(&wMaxA,0u);}workgroupBarrier();
  if(v>0.0){atomicMax(&wMaxA,bitcast<u32>(v));}
  workgroupBarrier();
  var r=0.0;if(lid==0u){r=bitcast<f32>(atomicLoad(&wMaxA));}return r;
}
`;
  const GATE = /* wgsl */`
// The F gate runs the row bricks (pmgTop RB, RBN of them) under the fold rule.
fn pGate(open:bool){
  let c=NB_ARGS_PER_LIST*NB_LIST_C_ALL+NB_ARGS_G1;let n=min(pmgTop[T_RBN],NB_FCAP);
  args[P_GATE_F]=select(0u,min(n,NB_FOLD),open&&n>0u);args[P_GATE_F+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[P_GATE_F+2u]=1u;
  args[P_GATE_C]=select(0u,args[c],open);args[P_GATE_C+1u]=args[c+1u];args[P_GATE_C+2u]=1u;
  let m=min(pmgTop[T_SBN],4u*NB_FCAP);let fc=n+nbListCount(NB_LIST_C_ALL);
  args[P_GATE_FC]=select(0u,min(fc,NB_FOLD),open&&fc>0u);args[P_GATE_FC+1u]=max(1u,(fc+NB_FOLD-1u)/NB_FOLD);args[P_GATE_FC+2u]=1u;
  args[P_GATE_S]=select(0u,min(m,NB_FOLD),open&&m>0u);args[P_GATE_S+1u]=max(1u,(m+NB_FOLD-1u)/NB_FOLD);args[P_GATE_S+2u]=1u;
  args[P_GATE_ONE]=select(0u,1u,open);args[P_GATE_ONE+1u]=1u;args[P_GATE_ONE+2u]=1u;
  atomicStore(&nbSet[PH_OPEN],select(0u,1u,open));
}
fn pTolR()->f32{let h=nbRingF(cfg.x,NB_RING_H);return bitcast<f32>(cfg.w)*h*h;}
/** History word k (0 the initial residual, k after slot k-1) in 1/s; the rest of the step's words cleared at k 0. */
fn pHist(k:u32,rmax:u32){
  let h=nbRingF(cfg.x,NB_RING_H);
  if(k==0u){for(var j=1u;j<=${MG_HIST_SLOTS}u;j++){atomicStore(&nbSet[PH_HIST+j],0u);}}
  if(k<=${MG_HIST_SLOTS}u){atomicStore(&nbSet[PH_HIST+k],bitcast<u32>(bitcast<f32>(rmax)/(h*h)));}
}
`;
  // (A v)(c) at a fine row, v given by an expression of (slot, local); a native neighbour's value
  // by nat(C slot, tile index), or 0 when nat is omitted (pre-smoothing: native rows are carried at 0).
  // (A v)(c) at h row c (slot s, local l). eager (pmgUpdate): every coupling (K, and RKU, pmgSetupF's upper
  // couplings) and every neighbour value is loaded in one round, before any test on a loaded value, so a lane's chain
  // is one load deep rather than D, then K, then the value (Update 218 -> 183 us a call on the tiered dam). The other
  // callers keep the tested form: eager in pmgSetupF and pmgDown0, heavier kernels, measured 852 -> 1453 and 90 -> 128
  // us. Both forms sum the same terms in the same order (a zero coupling or a missing neighbour adds nothing).
  const fineApply = (name: string, v: (s: string, l: string) => string, nat?: (cs: string, i: string) => string, eager = false) => eager ? /* wgsl */`
fn ${name}(b:vec3i,c:vec3i,s:u32,l:u32)->f32{
  let d=pDiag(s,l);let v0=${v("s", "l")};let ra=P_RKU+3u*NB_STRIDE*s+l;
  var kl=vec3f(0.0);var ku=vec3f(0.0);var vl=vec3f(0.0);var vu=vec3f(0.0);var hl=vec3<bool>(false);var hu=vec3<bool>(false);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    kl[a]=pK(a,s,l);
    let cn=c-e;let nn=pNatAt(b,cn);
    if(nn!=NB_NONE){${nat ? `hl[a]=true;vl[a]=${nat("nn", "nbLocal(cn>>vec3u(2u))")};` : ""}}
    else{let sn=pSlotAt(b,cn);if(sn!=NB_NONE){let ln=nbLocal(cn);hl[a]=true;vl[a]=${v("sn", "ln")};}}
    let cu=c+e;let nu=pNatAt(b,cu);
    if(nu!=NB_NONE){${nat ? `hu[a]=true;ku[a]=pmgF[ra+a*NB_STRIDE];vu[a]=${nat("nu", "nbLocal(cu>>vec3u(2u))")};` : ""}}
    else{let su=pSlotAt(b,cu);if(su!=NB_NONE){let lu=nbLocal(cu);hu[a]=true;ku[a]=pmgF[ra+a*NB_STRIDE];vu[a]=${v("su", "lu")};}}
  }
  if(d<=0.0){return 0.0;}
  var r=d*v0;
  for(var a=0u;a<3u;a++){
    if(kl[a]!=0.0&&hl[a]){r-=kl[a]*vl[a];}
    if(hu[a]&&ku[a]!=0.0){r-=ku[a]*vu[a];}
  }
  return r;
}` : /* wgsl */`
fn ${name}(b:vec3i,c:vec3i,s:u32,l:u32)->f32{
  let d=pDiag(s,l);if(d<=0.0){return 0.0;}
  var r=d*(${v("s", "l")});
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    let kl=pK(a,s,l);
    if(kl!=0.0){
      let cn=c-e;let nn=pNatAt(b,cn);
      if(nn!=NB_NONE){${nat ? `r-=kl*(${nat("nn", "nbLocal(cn>>vec3u(2u))")});` : ""}}
      else{let sn=pSlotAt(b,cn);if(sn!=NB_NONE){let ln=nbLocal(cn);r-=kl*(${v("sn", "ln")});}}
    }
    let cu=c+e;let nu=pNatAt(b,cu);
    if(nu!=NB_NONE){${nat ? `let ku=pKU(a,s,l);if(ku!=0.0){r-=ku*(${nat("nu", "nbLocal(cu>>vec3u(2u))")});}` : ""}}
    else{let su=pSlotAt(b,cu);if(su!=NB_NONE){let lu=nbLocal(cu);let ku=pK(a,su,lu);if(ku!=0.0){r-=ku*(${v("su", "lu")});}}}
  }
  return r;
}`;
  // The seam sums of a page's native rows: over row T's seam subfaces m (the 16 h cells of an h tile beside each of its
  // seam faces), sum ch(m) v(m), ch the h row's own coupling to T (its K[a] when T is below m, its KU[a] when T is
  // above). The page's seam list (pmgCoarsenC) is spread over the workgroup: 32 faces a round, a lane a subface (its
  // coupling and value loads independent of every other lane's), each face's 16 products summed in subface order, then
  // each row's lane adds its faces' sums in list (face) order. A lane walked its row's faces' 16 subfaces alone before,
  // the slowest row's chain the whole pass. v2 gives a second sum over the same couplings (vec2f).
  // Called by every lane in uniform control flow (n = pSeamN); lane lid gets tile lid's sum (0 off the native rows).
  const SEAM = /* wgsl */`
var<workgroup> wSV:array<f32,512>;var<workgroup> wSF:array<f32,32>;var<workgroup> wSV2:array<vec2f,512>;var<workgroup> wSF2:array<vec2f,32>;
var<workgroup> wSN:u32;
fn pSeamN(cs:u32,lid:u32)->u32{if(lid==0u){wSN=min(u32(pmgC[P_SEAMN+cs]),3072u);}return workgroupUniformLoad(&wSN);}`;
  const seamSum = (name: string, v: (s: string, l: string, m: string) => string, v2?: (s: string, l: string, m: string) => string) => {
    const ty = v2 ? "vec2f" : "f32", zero = `${ty}(0.0)`, ws = v2 ? "wSV2" : "wSV", wf = v2 ? "wSF2" : "wSF";
    const val = v2 ? `vec2f(${v("fs", "l", "m")},${v2("fs", "l", "m")})` : `(${v("fs", "l", "m")})`;
    return /* wgsl */`
fn ${name}(p:vec3i,cs:u32,lid:u32,n:u32)->${ty}{
  let rng=u32(pmgC[P_SEAMT+512u*cs+lid]);let st=rng>>3u;let ct=rng&7u;
  var mine=${zero};
  for(var c0=0u;c0<n;c0+=32u){
    let e=c0+(lid>>4u);var x=${zero};
    if(e<n){
      let ent=u32(pmgC[P_SEAML+3072u*cs+e]);let fs=u32(pmgC[P_SEAMF+3072u*cs+e]);
      let f=ent&7u;let a=f/2u;let up=(f&1u)==1u;let ea=pUnit(a);let T=8*p+nbLocalCoord(ent>>3u);
      var layer=4*select(T-ea,T+ea,up);if(!up){layer[a]+=3;}
      let q=lid&15u;let m=layer+i32(q&3u)*pUnit((a+1u)%3u)+i32(q>>2u)*pUnit((a+2u)%3u);let l=nbLocal(m);
      let ch=select(pKU(a,fs,l),pK(a,fs,l),up);
      x=select(${zero},ch*${val},ch!=0.0);
    }
    ${ws}[lid]=x;
    workgroupBarrier();
    if(lid<32u){var t=${zero};for(var j=0u;j<16u;j++){t+=${ws}[16u*lid+j];}${wf}[lid]=t;}
    workgroupBarrier();
    for(var k=max(st,c0);k<min(st+ct,c0+32u);k++){mine+=${wf}[k-c0];}
  }
  return mine;
}`;
  };
  // pmgCoarsenC's seam list. Across face f of tile T (a native row): the F slot of the h tile's seam layer, else NONE
  // (no tile, no page, a native neighbour, or no F brick).
  const SEAM_BUILD = /* wgsl */`
fn pSeamBrick(T:vec3i,f:u32)->u32{
  let a=f/2u;let up=(f&1u)==1u;let e=pUnit(a);let Tn=select(T-e,T+e,up);
  if(!pTileIn(Tn)){return NB_NONE;}
  let cn=pPageSlot(Tn>>vec3u(3u));
  if(cn==NB_NONE||!nbTileH(cn,nbLocal(Tn))){return NB_NONE;}
  var layer=4*Tn;if(!up){layer[a]+=3;}
  let c=nbChild(cn,nbBrickLocal(layer>>vec3u(3u)));return select(NB_NONE,c,c<NB_SEAM);
}
// Exclusive prefix sum over the workgroup's 512 lanes (wRed; every lane in uniform control flow); wSN the total.
fn pScan512(lid:u32,v:u32)->u32{
  wRed[lid]=f32(v);workgroupBarrier();
  for(var off=1u;off<512u;off*=2u){var t=0.0;if(lid>=off){t=wRed[lid-off];}workgroupBarrier();wRed[lid]+=t;workgroupBarrier();}
  if(lid==0u){wSN=u32(wRed[511u]);}
  return u32(wRed[lid])-v;
}`;
  // (A v)(T) at a native row: v native by vn(C slot, tile index); sm the row's seam sum of v on the h rows (seamSum).
  const nativeApplyFn = (name: string, vn: (cs: string, i: string) => string) => /* wgsl */`
fn ${name}(p:vec3i,cs:u32,i:u32,T:vec3i,sm:f32)->f32{
  var r=pP(NB_PRES_C_DIAG,cs,i)*(${vn("cs", "i")});
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    let kl=pCK(a,cs,i);if(kl!=0.0){let Tl=T-e;let sl=qPage(p,Tl,3u);if(sl!=NB_NONE){let il=nbLocal(Tl);r-=kl*(${vn("sl", "il")});}}
    let Tu=T+e;let su=qPage(p,Tu,3u);if(su!=NB_NONE){let iu=nbLocal(Tu);let ku=pCK(a,su,iu);if(ku!=0.0){r-=ku*(${vn("su", "iu")});}}
  }
  return r-sm;
}`;
  const nativeApply = (name: string, seam: string, vn: (cs: string, i: string) => string, vf: (s: string, l: string, m: string) => string) => seamSum(seam, vf) + nativeApplyFn(name, vn);
  // A native row's A x in face-difference form, the old engine's (uniform-pressure-arena.ts pNative: rhs += c (x_n -
  // start)): (D - sum k) x_i + sum k (x_i - x_n), the seam's faces in the same form. x is the total pressure, so
  // D x - sum k x_n cancels the hydrostatic part in f32 and leaves eps |D| |x| of noise in every residual (at rest
  // 1e-3 1/s band-eq on the 1 cm tank, ten times the native floor nBound reaches for); a face difference is the
  // neighbour step, exact to eps of itself. The seam sum carries (sum ch (x_f - x_i), sum ch).
  const nativeApplyDiff = (name: string, seam: string, vn: (cs: string, i: string) => string, vf: (s: string, l: string, m: string) => string) =>
    seamSum(seam, (s, l, m) => `(${vf(s, l, m)})-(${vn("cs", "(ent>>3u)")})`, () => "1.0") + /* wgsl */`
fn ${name}(p:vec3i,cs:u32,i:u32,T:vec3i,sm:vec2f)->f32{
  let xi=${vn("cs", "i")};var ks=sm.y;var r=0.0;
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);
    let kl=pCK(a,cs,i);if(kl!=0.0){let Tl=T-e;let sl=qPage(p,Tl,3u);if(sl!=NB_NONE){let il=nbLocal(Tl);r+=kl*(xi-(${vn("sl", "il")}));ks+=kl;}}
    let Tu=T+e;let su=qPage(p,Tu,3u);if(su!=NB_NONE){let iu=nbLocal(Tu);let ku=pCK(a,su,iu);if(ku!=0.0){r+=ku*(xi-(${vn("su", "iu")}));ks+=ku;}}
  }
  return (pP(NB_PRES_C_DIAG,cs,i)-ks)*xi+r-sm.x;
}`;
  // Level 0 tiles: a block of 8^3 rows with a one-cell halo, smoothed in place.
  const TILE = /* wgsl */`
var<workgroup> tp:array<f32,1000>;
fn pTi(lc:vec3i)->u32{return u32(lc.x+1)+10u*(u32(lc.y+1)+10u*u32(lc.z+1));}
fn pTileQ(o:vec3i,t:u32)->vec3i{return o-vec3i(1)+vec3i(i32(t%10u),i32((t/10u)%10u),i32(t/100u));}
struct PRow{d:f32,kl:vec3f,ku:vec3f}
// The upper couplings of h cell c (slot s, local l): KU[a] to a native row, else the h neighbour's K[a] (pmgSetupF
// writes them to RKU, which pRow reads).
fn pRowKU(b:vec3i,c:vec3i,s:u32,l:u32)->vec3f{
  var ku=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let cu=c+pUnit(a);
    if(pNatAt(b,cu)!=NB_NONE){ku[a]=pKU(a,s,l);}
    else{let su=pSlotAt(b,cu);if(su!=NB_NONE){ku[a]=pK(a,su,nbLocal(cu));}}
  }
  return ku;
}
fn pRow(b:vec3i,c:vec3i)->PRow{
  var r=PRow(0.0,vec3f(0.0),vec3f(0.0));let s=pSlotAt(b,c);if(s==NB_NONE){return r;}
  let l=nbLocal(c);let d=pDiag(s,l);
  let kl=vec3f(pK(0u,s,l),pK(1u,s,l),pK(2u,s,l));
  let at=P_RKU+3u*NB_STRIDE*s+l;let ku=vec3f(pmgF[at],pmgF[at+NB_STRIDE],pmgF[at+2u*NB_STRIDE]);
  if(d<=0.0){return r;}
  r.d=d;r.kl=kl;r.ku=ku;
  return r;
}
// Red-black Gauss-Seidel on tp, the halo frozen; black first when rev (the adjoint).
fn pSmooth(row:PRow,c:vec3i,ti:u32,rhs:f32,rev:bool){
  let red=pPar(c);
  for(var p=0u;p<P_FS2;p++){
    let colour=select(p&1u,1u-(p&1u),rev);
    if(row.d>0.0&&red==colour){
      tp[ti]=(rhs+row.kl.x*tp[ti-1u]+row.ku.x*tp[ti+1u]+row.kl.y*tp[ti-10u]+row.ku.y*tp[ti+10u]+row.kl.z*tp[ti-100u]+row.ku.z*tp[ti+100u])/row.d;
    }
    workgroupBarrier();
  }
}
fn pTileZero(lid:u32){for(var t=lid;t<1000u;t+=512u){tp[t]=0.0;}workgroupBarrier();}
`;
  // A row brick (pmgTop RB): the slot's brick kernels.
  const rowHead = /* wgsl */`
  if(lid==0u){wCount=min(pmgTop[T_RBN],NB_FCAP);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let slot=pmgTop[T_RB+g];pStageK(slot,lid);workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);`;
  // A shifted block (pmgTop SB, SBN of them): the L0 S passes; k picks the block from its owner brick.
  const shiftHead = /* wgsl */`
  if(lid==0u){wCount=min(pmgTop[T_SBN],4u*NB_FCAP);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let e=pmgTop[T_SB+g];let slot=e>>2u;let k=e&3u;pStageK(slot,lid);workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);`;
  // A page of C_ALL. "all" keeps every page; "rows" returns when the page has no rows (L5 D 0);
  // "native" when it has no native rows (L5 KU0, the count coarsenC writes).
  const pageHead = (keep: "all" | "rows" | "native" = "rows", skip = "") => /* wgsl */`
  if(lid==0u){wCount=nbListCount(NB_LIST_C_ALL);}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  ${keep === "all" ? "" : `if(lid==0u){wKids=select(0u,1u,pmgC[pL5(${keep === "rows" ? "CH_D" : "CH_KU"},cs)]>0.0);}if(workgroupUniformLoad(&wKids)==0u){${skip}return;}`}
  ${keep === "all" ? "qStage" : "qStageK"}(cs,lid);workgroupBarrier();
  let p=nbKeyCoord(nbCRec(cs,NB_REC_KEY));`;

  type Part = "pres" | "stage" | "cstage" | "red" | "reduce" | "gate" | "tile" | "seam";
  const kernel = (label: string, bindings: readonly BrickBinding[], parts: readonly Part[], body: string): BrickKernel => {
    const has = (n: string) => bindings.some(b => b.name === n);
    const abi = brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, ...(has("dir") ? { dir: { name: "dir", atomic: false } } : {}) });
    const src: Record<Part, string> = { pres: PRES, stage: STAGE + (has("pmgF") ? STAGEK : ""), cstage: CSTAGE + (has("pmgC") ? CSTAGEK : ""), red: RED + (parts.includes("reduce") ? FIXED : ""), reduce: REDUCE512, gate: GATE, tile: TILE, seam: SEAM };
    const page = parts.includes("stage") || parts.includes("cstage") ? [PAGE, "var<workgroup> wCount:u32;"] : [];
    return { label, entry: label, bindings, code: pressureResolveBox(L, [abi, declareBindings(bindings), COMMON, ...page, ...parts.map(p => src[p]), body].join("\n")) };
  };
  const FB = [CFG, setRW, dirRO, presRW, fRW, cRW]; // brick kernels
  const FBT = [...FB, topRW]; // brick kernels over the row bricks
  const CB = [CFG, setRW, dirRO, cRW]; // page kernels
  const CBP = [CFG, setRW, dirRO, presRW, cRW]; // page kernels on the native rows

  // ---- the monolithic rigid coupling (MGPCGOptions.bodies) -------------------------------------------------------------
  // pmgB (u32 words, floats as bits): MB_Q a body's arm origin Q (pbWorldCell(c) - X = h c + Q, couple's arm); MB_W its
  // W = M~ J^T y vectors (Wz this slot's z, Wx the iterate x, Wd the direction d; linear then angular); MB_MK a K slot's
  // marker (step + 1 when its brick holds a dynamic body's rows this step: pmgBodyJ's); MB_P a K slot's J^T z partials
  // (6 a body); MB_J a K slot's cells, 2 words a cell, each a body's face areas a_c / h^2 in quarters (pbBodyFaceFraction
  // is a multiple of 0.25, so the word is exact) and body + 1 (0 none). The words a_c, Q and M~ are fixed in a step, so
  // B = (1/h) J M~ J^T is one symmetric operator for every product the solve takes.
  const NB = GPU_RIGID_BODY_CAPACITY, KCAP = L.cap.k;
  const MB_W = 4 * NB, MB_MK = MB_W + 18 * NB, MB_P = MB_MK + KCAP, MB_J = MB_P + 6 * NB * KCAP, MB_WORDS = MB_J + 1024 * KCAP;
  const bRW: BrickBinding = { name: "pmgB", kind: "rw", type: "array<u32>" };
  const cutRO: BrickBinding = { name: "cut", kind: "ro", type: "array<f32>" };
  const MB_B = bodies ? [bRW] : [];
  const MB = bodies ? /* wgsl */`
const MB_N:u32=${u(NB)};const MB_Q:u32=0u;const MB_W:u32=${u(MB_W)};const MB_MK:u32=${u(MB_MK)};const MB_P:u32=${u(MB_P)};const MB_J:u32=${u(MB_J)};
const MB_WZ:u32=0u;const MB_WX:u32=6u;const MB_WD:u32=12u;
fn mbF(i:u32)->f32{return bitcast<f32>(pmgB[i]);}
fn mbV(i:u32)->vec3f{return vec3f(mbF(i),mbF(i+1u),mbF(i+2u));}
// Brick slot's K slot when it holds this step's body rows (pmgBodyJ's marker), else NONE.
fn mbBrick(slot:u32)->u32{
  let k=nbFRec(slot,NB_REC_CUT);if(k==NB_NONE){return NB_NONE;}
  return select(NB_NONE,k,pmgB[MB_MK+k]==nbRing(cfg.x,NB_RING_STEP)+1u);
}
fn mbArea(w:u32)->vec3f{return vec3f(f32(i32(w&15u)-4),f32(i32((w>>4u)&15u)-4),f32(i32((w>>8u)&15u)-4))*0.25;}
fn mbWord(k:u32,l:u32,t:u32)->u32{return pmgB[MB_J+2u*(512u*k+l)+t];}
// (B y)_c: (1/h) sum over cell (k, l)'s bodies of j_c . (W_o + alpha W_o2) (o2 = o: W_o alone); 0 off the body rows.
fn mbApply(k:u32,l:u32,c:vec3i,o:u32,alpha:f32,o2:u32)->f32{
  if(k==NB_NONE){return 0.0;}
  let h=nbRingF(cfg.x,NB_RING_H);var s=0.0;
  for(var t=0u;t<2u;t++){
    let w=mbWord(k,l,t);let ib=(w>>12u)&31u;if(ib==0u){continue;}
    let i=ib-1u;let a=mbArea(w)*(h*h);let arm=h*vec3f(c)+mbV(MB_Q+4u*i);let base=MB_W+18u*i;
    var lin=mbV(base+o);var ang=mbV(base+o+3u);
    if(o2!=o){lin+=alpha*mbV(base+o2);ang+=alpha*mbV(base+o2+3u);}
    s+=dot(a,lin)+dot(cross(arm,a),ang);
  }
  return s/h;
}
// Lanes 0-5: the workgroup's sums of (lin, ang) in a fixed order (pSum512's tree, six at once); other lanes 0. A
// 512-lane workgroup in uniform control flow (three barriers); a next call's writes follow this call's reads.
var<workgroup> mbS6:array<f32,3072>;var<workgroup> mbS2:array<f32,384>;var<workgroup> mbS3:array<f32,48>;
fn mbSum6(lid:u32,lin:vec3f,ang:vec3f)->f32{
  mbS6[lid]=lin.x;mbS6[512u+lid]=lin.y;mbS6[1024u+lid]=lin.z;mbS6[1536u+lid]=ang.x;mbS6[2048u+lid]=ang.y;mbS6[2560u+lid]=ang.z;
  workgroupBarrier();
  if(lid<384u){let cc=lid/64u;let j=lid%64u;var s=0.0;for(var m=0u;m<8u;m++){s+=mbS6[512u*cc+64u*m+j];}mbS2[lid]=s;}
  workgroupBarrier();
  if(lid<48u){let cc=lid/8u;let j=lid%8u;var s=0.0;for(var m=0u;m<8u;m++){s+=mbS2[64u*cc+8u*m+j];}mbS3[lid]=s;}
  workgroupBarrier();
  var r=0.0;if(lid<6u){for(var m=0u;m<8u;m++){r+=mbS3[8u*lid+m];}}
  return r;
}
` : "";
  // The body kernels' rigid state: the bodies' poses (pbBrickBodiesWGSL: the ring's gate, couple's shape queries).
  const mbBodies = bodies ? pbBrickBodiesWGSL(bodies) + "\n" + /* wgsl */`
// A dynamic body with mass: the integrator's (rigidIsDynamic; a held or fixed body integrates nothing).
fn mbImplicit(i:u32)->bool{let b=bodies[i];return b.material.z>0.5&&b.material.z<1.5&&b.inverseMassInertia.x>0.0;}
// The solve's bodies: the ring's gate with the coupling on, past the boot (the kick has no couple; step 0 is the boot's ring).
fn mbOn()->bool{return pbBrickBodiesOn()&&(nbRing(cfg.x,NB_RING_BODY_FLAGS)&NB_BODY_FLAG_COUPLING)!=0u&&nbRing(cfg.x,NB_RING_STEP)>0u;}
` : "";
  // pmgBodyJ (the setup's head, after the rows; F_BODY, a workgroup a brick, as couple): a body brick's (K slot,
  // BMASK != 0) cells' face areas against its dynamic bodies, the RHS's V* (RHS -= dt h (a_c / h^2) . g) and the K
  // slot's marker; workgroup 0 writes the bodies' arm origins. A third dynamic body at one cell latches (0xb4).
  const bodyJ = bodies && kernel("pmgBodyJ", [CFG, setRW, dirRO, presRW, cutRO, BODIES_RO, BODY_CUTS_RO, bRW], [], MB + mbBodies + pbBodyCoupleWGSL + /* wgsl */`
var<workgroup> mbHead:vec4u;var<workgroup> mbAny:atomic<u32>;
@compute @workgroup_size(512) fn pmgBodyJ(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=nbGroup(wid);
  if(lid==0u){
    var head=vec4u(0u,0u,NB_NONE,0u);
    if(mbOn()&&e<nbListCount(NB_LIST_F_BODY)){
      let s=nbListSlot(NB_LIST_F_BODY,e);let k=nbFRec(s,NB_REC_CUT);
      if(k!=NB_NONE){
        let mask=bitcast<u32>(cut[NB_CUT_BMASK+k])&pbBodyMaskAll();var dyn=0u;
        var rem=mask;while(rem!=0u){let i=firstTrailingBit(rem);rem&=rem-1u;if(mbImplicit(i)){dyn|=1u<<i;}}
        head=vec4u(1u,s,k,dyn);
      }
    }
    mbHead=head;atomicStore(&mbAny,0u);
  }
  if(e==0u&&lid<pbBodyCount()){let q=pbWorldCell(vec3i(0))-pbBodyPositionOf(lid);pmgB[MB_Q+4u*lid]=bitcast<u32>(q.x);pmgB[MB_Q+4u*lid+1u]=bitcast<u32>(q.y);pmgB[MB_Q+4u*lid+2u]=bitcast<u32>(q.z);}
  let head=workgroupUniformLoad(&mbHead);
  if(head.x==0u){return;}
  let s=head.y;let k=head.z;let c=nbFOrigin(s)+nbLocalCoord(lid);let at=NB_STRIDE*s+lid;
  var w0=0u;var w1=0u;var n=0u;var dRhs=0.0;
  if(head.w!=0u&&pres[NB_PRES_F_DIAG+at]>0.0&&cut[NB_CUT_FRAC+NB_STRIDE*k+lid]>1e-5){
    let h=nbRingF(cfg.x,NB_RING_H);let dt=nbRingF(cfg.x,NB_RING_DT);
    let g=vec3f(nbRingF(cfg.x,NB_RING_GRAVITY),nbRingF(cfg.x,NB_RING_GRAVITY+1u),nbRingF(cfg.x,NB_RING_GRAVITY+2u));
    var rem=head.w;
    while(rem!=0u){
      let i=firstTrailingBit(rem);rem&=rem-1u;
      let r=pbCoupleBody(i,c,1.0,0.0,1.0);if(!r.near){continue;}
      let q=vec3i(round(r.reaction*(4.0/(h*h))));
      if(all(q==vec3i(0))){continue;}
      let word=u32(q.x+4)|(u32(q.y+4)<<4u)|(u32(q.z+4)<<8u)|((i+1u)<<12u);
      if(n==0u){w0=word;}else if(n==1u){w1=word;}else{nbFail(NB_FAIL_UNSUPPORTED,0xb4000000u|nbFRec(s,NB_REC_KEY));}
      n+=1u;dRhs-=dt*h*dot(0.25*vec3f(q),g);
    }
  }
  pmgB[MB_J+2u*(512u*k+lid)]=w0;pmgB[MB_J+2u*(512u*k+lid)+1u]=w1;
  if(w0!=0u){atomicOr(&mbAny,1u);}
  if(dRhs!=0.0){pres[NB_PRES_F_RHS+at]+=dRhs;}
  workgroupBarrier();
  if(lid==0u){pmgB[MB_MK+k]=select(0u,nbRing(cfg.x,NB_RING_STEP)+1u,atomicLoad(&mbAny)!=0u);}
}`);
  // pmgBodyZ (F_BODY): a marked brick's J^T y partials (y = pres F Z: the guess in the setup, a slot's z after
  // pmgPostZ), 6 a body, into its K slot's MB_P words (0 for a body without rows there). open: a slot's (PH_OPEN).
  const bodyZKernel = (open: boolean) => bodies && kernel(open ? "pmgBodyZ" : "pmgBodyZ0", [CFG, setRW, dirRO, presRW, bRW], [], MB + /* wgsl */`
var<workgroup> mbHead2:vec2u;var<workgroup> mbMaskA:atomic<u32>;var<workgroup> mbMask:u32;var<workgroup> mbCount:u32;
@compute @workgroup_size(512) fn ${open ? "pmgBodyZ" : "pmgBodyZ0"}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let e=nbGroup(wid);
  if(lid==0u){
    var k=NB_NONE;var s=0u;
    if(${open ? "atomicLoad(&nbSet[PH_OPEN])!=0u&&" : ""}e<nbListCount(NB_LIST_F_BODY)){s=nbListSlot(NB_LIST_F_BODY,e);k=mbBrick(s);}
    mbHead2=vec2u(k,s);atomicStore(&mbMaskA,0u);
  }
  let hd=workgroupUniformLoad(&mbHead2);let k=hd.x;
  if(k==NB_NONE){return;}
  let s=hd.y;let c=nbFOrigin(s)+nbLocalCoord(lid);let y=pres[NB_PRES_F_Z+NB_STRIDE*s+lid];let h=nbRingF(cfg.x,NB_RING_H);
  let w0=mbWord(k,lid,0u);let w1=mbWord(k,lid,1u);let i0=(w0>>12u)&31u;let i1=(w1>>12u)&31u;
  if(i0!=0u){atomicOr(&mbMaskA,1u<<(i0-1u));}
  if(i1!=0u){atomicOr(&mbMaskA,1u<<(i1-1u));}
  let a0=mbArea(w0)*(h*h*y);let a1=mbArea(w1)*(h*h*y);
  let t0=select(vec3f(0.0),cross(h*vec3f(c)+mbV(MB_Q+4u*(max(i0,1u)-1u)),a0),i0!=0u);
  let t1=select(vec3f(0.0),cross(h*vec3f(c)+mbV(MB_Q+4u*(max(i1,1u)-1u)),a1),i1!=0u);
  workgroupBarrier();
  if(lid==0u){mbMask=atomicLoad(&mbMaskA);mbCount=min(nbRing(cfg.x,NB_RING_BODY_COUNT),MB_N);}
  let mask=workgroupUniformLoad(&mbMask);
  let n=workgroupUniformLoad(&mbCount);
  for(var i=0u;i<n;i++){
    let base=MB_P+6u*(MB_N*k+i);
    if(((mask>>i)&1u)==0u){if(lid<6u){pmgB[base+lid]=0u;}continue;}
    var lin=vec3f(0.0);var ang=vec3f(0.0);
    if(i0==i+1u){lin+=a0;ang+=t0;}
    if(i1==i+1u){lin+=a1;ang+=t1;}
    let v=mbSum6(lid,lin,ang);
    if(lid<6u){pmgB[base+lid]=bitcast<u32>(v);}
  }
}`);
  // pmgBodyFold (a workgroup a body): J^T y over F_BODY's marked bricks in list order, then W = M~ J^T y. The setup's
  // (init): Wx from the guess, Wz = Wd = 0. A slot's: Wz; x += alpha_old d and d = z + beta d (pmgDirection's, beta
  // from the same fold of r.z and PH_RZ) carried on the W vectors, so pmgDirection and pmgUpdate take B z and
  // B (x + alpha d) without another reduction.
  const bodyFoldKernel = (init: boolean) => bodies && kernel(init ? "pmgBodyFold0" : "pmgBodyFold", [CFG, setRW, dirRO, topRW, redRW, bRW, BODIES_RO, BODY_CUTS_RO], init ? [] : ["red", "reduce"], MB + mbBodies + /* wgsl */`
var<workgroup> mbG:array<f32,6>;var<workgroup> mbCount:u32;${init ? "" : "var<workgroup> wOpenB:u32;var<workgroup> wNF:u32;var<workgroup> wNC:u32;"}
@compute @workgroup_size(512) fn ${init ? "pmgBodyFold0" : "pmgBodyFold"}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let i=wid.x;${init ? "" : `
  if(lid==0u){wOpenB=atomicLoad(&nbSet[PH_OPEN]);wNF=min(pmgTop[T_RBN],NB_FCAP);wNC=nbListCount(NB_LIST_C_ALL);}
  if(workgroupUniformLoad(&wOpenB)==0u){return;}
  let rz=pFixed(lid,R_ZF,workgroupUniformLoad(&wNF),R_ZC,workgroupUniformLoad(&wNC));`}
  if(lid==0u){mbCount=pbBodyCount();}
  if(i>=workgroupUniformLoad(&mbCount)){return;}
  let n=nbListCount(NB_LIST_F_BODY);
  var lin=vec3f(0.0);var ang=vec3f(0.0);
  for(var e=lid;e<n;e+=512u){let k=mbBrick(nbListSlot(NB_LIST_F_BODY,e));if(k!=NB_NONE){let b=MB_P+6u*(MB_N*k+i);lin+=mbV(b);ang+=mbV(b+3u);}}
  let v=mbSum6(lid,lin,ang);
  if(lid<6u){mbG[lid]=v;}
  workgroupBarrier();
  if(lid==0u){
    var wl=vec3f(0.0);var wa=vec3f(0.0);
    if(mbImplicit(i)){
      let b=bodies[i];let q=b.orientation;
      wl=b.inverseMassInertia.x*vec3f(mbG[0],mbG[1],mbG[2]);
      wa=pbQuatRotate(q,b.inverseMassInertia.yzw*pbQuatRotate(vec4f(q.x,-q.yzw),vec3f(mbG[3],mbG[4],mbG[5])));
    }
    let base=MB_W+18u*i;${init ? `
    for(var j=0u;j<18u;j++){pmgB[base+j]=0u;}
    pmgB[base+MB_WX]=bitcast<u32>(wl.x);pmgB[base+MB_WX+1u]=bitcast<u32>(wl.y);pmgB[base+MB_WX+2u]=bitcast<u32>(wl.z);
    pmgB[base+MB_WX+3u]=bitcast<u32>(wa.x);pmgB[base+MB_WX+4u]=bitcast<u32>(wa.y);pmgB[base+MB_WX+5u]=bitcast<u32>(wa.z);` : `
    var beta=0.0;
    if(cfg.z>0u){let old=bitcast<f32>(atomicLoad(&nbSet[PH_RZ]));beta=select(0.0,rz/old,old!=0.0);}
    let alpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));
    var wz=array<f32,6>(wl.x,wl.y,wl.z,wa.x,wa.y,wa.z);
    for(var j=0u;j<6u;j++){
      let d=mbF(base+MB_WD+j);
      pmgB[base+MB_WZ+j]=bitcast<u32>(wz[j]);
      pmgB[base+MB_WX+j]=bitcast<u32>(mbF(base+MB_WX+j)+alpha*d);
      pmgB[base+MB_WD+j]=bitcast<u32>(wz[j]+beta*d);
    }`}
  }
}`);

  // ---- setup ---------------------------------------------------------------
  const setupClear = kernel("pmgSetupClear", [CFG, setRW, redRW], [], /* wgsl */`
@compute @workgroup_size(128) fn pmgSetupClear(@builtin(local_invocation_index) lid:u32){
  if(lid<8u){pmgRed[lid]=0.0;}
  if(lid==0u){
    atomicStore(&nbSet[PH_RMAX],0u);atomicStore(&nbSet[PH_ALPHA],0u);atomicStore(&nbSet[PH_RZ],0u);
    atomicStore(&nbSet[PH_ITERATIONS],0u);atomicStore(&nbSet[PH_OPEN],0u);
    atomicStore(&nbSet[PH_FLIPS],0u);atomicStore(&nbSet[PH_WALLS],0u);atomicStore(&nbSet[PH_DECIDES],0u);atomicStore(&nbSet[PH_RMAXW],0u);atomicStore(&nbSet[PH_NSCALE],0u);
    for(var j=0u;j<4u;j++){atomicStore(&nbSet[PH_AUDIT+j],0u);}
    atomicAnd(&nbSet[NB_H_STATE],~NB_STATE_PRESSURE_CAP);
  }
}`);

  // setupF (a brick of F_ALL): r = b - A x, max|r|, D and Q cleared, the first brick pass from zero into W0, the brick's
  // L1 and L2 rows aggregated, and the stage cache (P_STG). A brick without a row and without a coupling (on the tiered
  // dam ~5 in 6 of F_ALL: the native tiles' bricks) has all of these 0, so it writes its zeros and returns before the
  // stage, the apply, the pass and the aggregation (bitwise the full path's values: every sum is of zeros).
  // pmgGuess (a brick of the setup's rows, before setupF): Z = x0 + D^-1 (b - A x0) on the h rows, x0 elsewhere (Saad
  // 2003 §4.1, the seep's pcgGuess). A row's ghost-fluid diagonal changes with the surface, and a warm row whose
  // diagonal grew starts with a residual of about D x0: on the tiered dam (steps 16-22) the start's max|r| is 1e5-3e5
  // 1/s on the surface rows, 1.2e3-1.5e3 after the step, and the composite model took 6.2 PCG iterations from it where
  // it took 8.0 from x0. Native rows keep x0 (their apply needs the seam list, which coarsenC builds after the setup;
  // the model's iterations were the same as with them stepped). setupF reads Z for its residual and writes X = Z, and
  // the stage it would build (pStageW) is this kernel's.
  const guess = kernel("pmgGuess", FBT, ["pres", "stage"],
    fineApply("pApplyG", (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]`, (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]`) + /* wgsl */`
var<workgroup> wGLiveA:atomic<u32>;var<workgroup> wGLive:u32;
@compute @workgroup_size(512) fn pmgGuess(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${fHead("atomicStore(&wGLiveA,0u);")}let o=nbFOrigin(slot);let b=o>>vec3u(3u);
  let l=lid;let at=NB_STRIDE*slot+l;let d=pDiag(slot,l);
  // setupF's live test (a row or a coupling): a brick without either is setupF's early return, which reads neither Z
  // nor the stage, so it writes neither (the F_ALL path then leaves the bricks off HB as the live path does).
  workgroupBarrier();
  if(d>0.0||any(vec3f(pK(0u,slot,l),pK(1u,slot,l),pK(2u,slot,l))!=vec3f(0.0))||any(vec3f(pKU(0u,slot,l),pKU(1u,slot,l),pKU(2u,slot,l))!=vec3f(0.0))){atomicOr(&wGLiveA,1u);}
  workgroupBarrier();
  if(lid==0u){wGLive=atomicLoad(&wGLiveA);}
  if(workgroupUniformLoad(&wGLive)==0u){return;}
  pStage(slot,lid);workgroupBarrier();
  pStageW(slot,lid);
  var x=pres[NB_PRES_F_X+at];
  if(d>0.0){x+=(pP(NB_PRES_F_RHS,slot,l)-pApplyG(b,o+nbLocalCoord(l),slot,l))/d;}
  pres[NB_PRES_F_Z+at]=x;
}`);
  const X0 = guessOn ? "NB_PRES_F_Z" : "NB_PRES_F_X";
  const setupF = kernel("pmgSetupF", [...FBT, ...MB_B], ["pres", "stage", "reduce", "tile"],
    fineApply("pApplyR", (s, l) => `pres[${X0}+NB_STRIDE*${s}+${l}]`, (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]`) + MB + /* wgsl */`
var<workgroup> wD:array<f32,512>;var<workgroup> wK:array<vec3f,512>;var<workgroup> wKU:array<vec3f,512>;var<workgroup> w1D:array<f32,64>;var<workgroup> w1K:array<vec3f,64>;var<workgroup> w1KU:array<vec3f,64>;
var<workgroup> wLive:u32;
@compute @workgroup_size(512) fn pmgSetupF(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${fHead()}let o=nbFOrigin(slot);let b=o>>vec3u(3u);
  let l=lid;let d=pDiag(slot,l);
  let kl=vec3f(pK(0u,slot,l),pK(1u,slot,l),pK(2u,slot,l));let kh=vec3f(pKU(0u,slot,l),pKU(1u,slot,l),pKU(2u,slot,l));
  let rows=pMax512(lid,select(0.0,1.0,d>0.0));
  let links=pMax512(lid,select(0.0,1.0,any(kl!=vec3f(0.0))||any(kh!=vec3f(0.0))));
  if(lid==0u){pmgTop[T_RBF+slot]=select(0u,1u,rows>0.0);wLive=select(0u,1u,rows>0.0||links>0.0);}
  let at=NB_STRIDE*slot+l;
  pres[NB_PRES_F_D+at]=0.0;pres[NB_PRES_F_Q+at]=0.0;
  let cs=nbFRec(slot,NB_REC_PARENT);
  if(workgroupUniformLoad(&wLive)==0u){
    pres[NB_PRES_F_R+at]=0.0;
    if(lid<64u){pmgF[pL1(CH_D,slot,lid)]=0.0;for(var a=0u;a<3u;a++){pmgF[pL1(CH_K+a,slot,lid)]=0.0;pmgF[pL1(CH_KU+a,slot,lid)]=0.0;}}
    if(lid<8u){let i2=nbLocal((o>>vec3u(2u))+pC4(lid));pmgC[pL2(CH_D,cs,i2)]=0.0;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i2)]=0.0;pmgC[pL2(CH_KU+a,cs,i2)]=0.0;}}
    return;
  }
  ${guessOn ? "pStageK(slot,lid);workgroupBarrier();pres[NB_PRES_F_X+at]=pres[NB_PRES_F_Z+at];" : "pStage(slot,lid);workgroupBarrier();\n  pStageW(slot,lid);"}
  let lc=nbLocalCoord(l);let c=o+lc;let ti=pTi(lc);
  {let ku=pRowKU(b,c,slot,l);let ra=P_RKU+3u*NB_STRIDE*slot+l;pmgF[ra]=ku.x;pmgF[ra+NB_STRIDE]=ku.y;pmgF[ra+2u*NB_STRIDE]=ku.z;}
  var r=0.0;
  if(d>0.0){r=pP(NB_PRES_F_RHS,slot,l)-pApplyR(b,c,slot,l)${bodies ? "-mbApply(mbBrick(slot),l,c,MB_WX,0.0,MB_WX)" : ""};}
  pres[NB_PRES_F_R+at]=r;
  let rmax=pMax512(lid,abs(r));
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}
  // The first brick pass of the next V-cycle: from zero, so its halo is zero.
  pTileZero(lid);
  let row=pRow(b,c);
  pSmooth(row,c,ti,r,false);
  if(row.d>0.0){pmgF[P_W0+NB_STRIDE*slot+l]=tp[ti];}
  // Galerkin aggregation over 2^3 children, A_c = W^-1 P^T A P W^-1 (W sqrt(OC) on an aggregate,
  // 1 on a native row; the composite model): a child's lower coupling is internal (subtracted twice
  // from D) when its coordinate along a is odd, else it crosses to the parent's lower neighbour, an
  // aggregate (over OC) or a native row (over sqrt(OC)). Upper links (KU) reach native rows only.
  // L2 -> L3 runs in pmgCoarsenC, beside the native rows.
  wD[l]=max(d,0.0);wK[l]=kl;wKU[l]=kh;
  workgroupBarrier();
  if(lid<64u){
    let c1=pC1(lid);var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
    for(var m=0u;m<8u;m++){
      let q=2*c1+pC4(m);let i=nbLocal(q);D+=wD[i];
      for(var a=0u;a<3u;a++){let w=wK[i][a];if((q[a]&1)==1){D-=2.0*w;KU[a]+=wKU[i][a];}else{K[a]+=w;}}
    }
    let g1=(o>>vec3u(1u))+c1;
    for(var a=0u;a<3u;a++){K[a]/=select(P_OC,P_SOC,pNatAt(b,2*(g1-pUnit(a)))!=NB_NONE);}
    D/=P_OC;KU/=P_SOC;w1D[lid]=D;w1K[lid]=K;w1KU[lid]=KU;
    pmgF[pL1(CH_D,slot,lid)]=D;for(var a=0u;a<3u;a++){pmgF[pL1(CH_K+a,slot,lid)]=K[a];pmgF[pL1(CH_KU+a,slot,lid)]=KU[a];}
  }
  workgroupBarrier();
  if(lid<8u){
    let c2=pC4(lid);var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
    for(var m=0u;m<8u;m++){
      let q=2*c2+pC4(m);let i=pI1(q);D+=w1D[i];
      for(var a=0u;a<3u;a++){let w=w1K[i][a];if((q[a]&1)==1){D-=2.0*w;KU[a]+=w1KU[i][a];}else{K[a]+=w;}}
    }
    let T=(o>>vec3u(2u))+c2;
    for(var a=0u;a<3u;a++){K[a]/=select(P_OC,P_SOC,pNatAt(b,4*(T-pUnit(a)))!=NB_NONE);}
    D/=P_OC;KU/=P_SOC;
    let i2=nbLocal(T);
    pmgC[pL2(CH_D,cs,i2)]=D;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i2)]=K[a];pmgC[pL2(CH_KU+a,cs,i2)]=KU[a];}
  }
}`);

  // ---- the live lists (mgLive) -------------------------------------------------------------------------------------
  // On the tiered dam 2879 of F_ALL's 3492 bricks hold no h-owned tile (the native interior), yet each launched a
  // workgroup of rowsF (512 lanes, 30 KB of shared memory), setupF and projectF, which returned at once: those
  // kernels' residency held a core to a few workgroups, so the empty ones cost 296 / 171 / 202 us a step (measured,
  // the kernels returning at their heads over F_ALL), and their zero rows were rewritten every step.
  // pressureRowsList (the engine's operator over F_ALL G1: a lane a brick, so 1 in 64 workgroups works) appends
  // - HB: a brick with an h-owned tile (the h rows' kernels' own test, nbTileOwnerH);
  // - HP: one with an h-owned tile among the 3^3 from its origin tile - 1 (projectF's: HB is a subset);
  // - ZL: off HB without a certificate that its zero rows hold since the last step: rowsF's B X RHS DIAG PPHI K KU,
  //   setupF's D Q R, the L1 rows and the 8 L2 rows (pressureRowsZero writes them, the values the F_ALL path's heads
  //   wrote). Only lane P writes these rows of a brick (setInit zeroes an entering one's pres), and off HB none of P's
  //   kernels does, so a brick that was off HB the step before (same key, same parent C slot) still holds them;
  // - RL: off HP without a certificate that its FLAGS bits 7-10 are clear since the last step (pressureProjectRest
  //   clears them, the engine's project; only projectF on HP and the rows on HB write them);
  // and writes the CUT latch and RBF 0 off HB. HB's bricks run the h rows' kernels unchanged, so the rows, the solve
  // and the projection are the F_ALL path's bit for bit (the composite live lists' GPU test). The append order varies;
  // no reduction reads it (RB is coarsenTop's).
  // The counts go through pmgTop, not the arguments: rowsList is dispatched from the arguments buffer (F_ALL G1), which
  // a dispatch may not also bind writable. pmgListArgs (one lane, after rowsList) moves them on and zeroes the counts
  // for the next step's appends (a step's operator and setup always run together, so the counts are 0 between steps).
  const topA: BrickBinding = { name: "pmgTop", kind: "rw", type: "array<atomic<u32>>" };
  const listArgs = kernel("pmgListArgs", [CFG, setRW, fRW, argsRW, topA], [], /* wgsl */`
@compute @workgroup_size(1) fn pmgListArgs(){
  for(var k=0u;k<4u;k++){
    // Each list holds at most F_ALL's bricks (NB_FCAP words): a larger count is a fault, latched, and the dispatch is
    // held to the list (the engine's indirect bound, lane E, 2026-11-17).
    var n=atomicLoad(&pmgTop[T_HBN+k]);atomicStore(&pmgTop[T_HBN+k],0u);
    if(n>NB_FCAP){nbFail(NB_FAIL_LIST,0x50000000u|(k<<24u)|min(n,0xffffffu));n=NB_FCAP;}
    if(k==3u){pmgF[P_RLN]=f32(n);continue;}
    let a=P_GATE_HB+3u*k;args[a]=min(n,NB_FOLD);args[a+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[a+2u]=1u;
  }
}`);
  const rowsList = kernel("pressureRowsList", [CFG, setRW, dirRO, fRW, topA], [], /* wgsl */`
var<workgroup> wCount:u32;var<workgroup> wN:array<atomic<u32>,4>;var<workgroup> wB:array<u32,4>;
// nbTileOwnerH (brick-owner.wgsl.ts), the h rows' kernels' test: a lattice tile of a resident page, h-owned.
fn pOwnerT(t:vec3i)->bool{
  if(!(all(t>=vec3i(0))&&all(4*t<NB_LATTICE))){return false;}
  let cs=nbCSlotOfPage(t>>vec3u(3u));if(cs==NB_NONE){return false;}
  let q=vec3u(t&vec3i(7));return nbTileH(cs,q.x|(q.y<<3u)|(q.z<<6u));
}
@compute @workgroup_size(64) fn pressureRowsList(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wCount=nbListCount(NB_LIST_F_ALL);}
  if(lid<4u){atomicStore(&wN[lid],0u);}
  let n=workgroupUniformLoad(&wCount);let e0=64u*nbGroup(wid);if(e0>=n){return;}
  let e=e0+lid;var bits=0u;var slot=0u;
  if(e<n){
    slot=nbListSlot(NB_LIST_F_ALL,e);let T0=nbFOrigin(slot)>>vec3u(2u);
    // Bit 0 HB (the brick's own tiles, d in 1..2), bit 1 HP.
    for(var i=0u;i<27u;i++){
      let d=vec3i(i32(i%3u),i32((i/3u)%3u),i32(i/9u));
      if(pOwnerT(T0-vec3i(1)+d)){bits|=2u;if(all(d>=vec3i(1))){bits|=1u;}}
    }
    let key=nbFRec(slot,NB_REC_KEY);let cs=nbFRec(slot,NB_REC_PARENT);let step=nbRing(cfg.x,NB_RING_STEP);
    let same=atomicLoad(&pmgTop[T_ZK+slot])==key&&atomicLoad(&pmgTop[T_ZC+slot])==cs;
    atomicStore(&pmgTop[T_ZK+slot],key);atomicStore(&pmgTop[T_ZC+slot],cs);
    if((bits&1u)==0u){
      ${cuts ? `// A brick of 4h tiles only: a body's latches (§4 keeps a body's tiles h). A static cut's may stay at 4h (lane E's
      // static rule leaves dry cut bricks there); a liquid native row in one latches in rowsC.
      if(nbFRec(slot,NB_REC_CUT)!=NB_NONE&&(nbFRec(slot,NB_REC_CLASS)&(${F_CLASS.NEAR_BODY | F_CLASS.BODY_WAS}u))!=0u){nbFail(NB_FAIL_UNSUPPORTED,key);}` : "if(nbFRec(slot,NB_REC_CUT)!=NB_NONE){nbFail(NB_FAIL_UNSUPPORTED,key);}"}
      atomicStore(&pmgTop[T_RBF+slot],0u);
      if(!(same&&atomicLoad(&pmgTop[T_ZS+slot])==step+1u)){bits|=4u;}
      atomicStore(&pmgTop[T_ZS+slot],step+2u);
    } else {atomicStore(&pmgTop[T_ZS+slot],0u);}
    if((bits&2u)==0u){
      if(!(same&&atomicLoad(&pmgTop[T_FS+slot])==step+1u)){bits|=8u;}
      atomicStore(&pmgTop[T_FS+slot],step+2u);
    } else {atomicStore(&pmgTop[T_FS+slot],0u);}
  }
  var at=vec4u(0u);
  for(var k=0u;k<4u;k++){if(((bits>>k)&1u)!=0u){at[k]=atomicAdd(&wN[k],1u);}}
  workgroupBarrier();
  if(lid<4u){wB[lid]=atomicAdd(&pmgTop[T_HBN+lid],atomicLoad(&wN[lid]));}
  workgroupBarrier();
  var lists=array<u32,4>(P_HBL,P_HPL,P_ZLL,P_RLL);
  for(var k=0u;k<4u;k++){if(((bits>>k)&1u)!=0u&&wB[k]+at[k]<NB_FCAP){pmgF[lists[k]+wB[k]+at[k]]=f32(slot);}}
}`);
  // pressureRowsZero: a workgroup a brick of ZL (setup, before the rows): the zero rows off HB.
  const rowsZero = kernel("pressureRowsZero", [CFG, setRW, dirRO, presRW, fRW, cRW], [], /* wgsl */`
@compute @workgroup_size(64) fn pressureRowsZero(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  let slot=u32(pmgF[P_ZLL+nbGroup(wid)]);
  for(var j=0u;j<8u;j++){
    let at=NB_STRIDE*slot+lid+64u*j;
    pres[NB_PRES_F_B+at]=0.0;pres[NB_PRES_F_X+at]=0.0;pres[NB_PRES_F_RHS+at]=0.0;pres[NB_PRES_F_DIAG+at]=0.0;pres[NB_PRES_F_PPHI+at]=0.0;
    pres[NB_PRES_F_K0+at]=0.0;pres[NB_PRES_F_K1+at]=0.0;pres[NB_PRES_F_K2+at]=0.0;pres[NB_PRES_F_KU0+at]=0.0;pres[NB_PRES_F_KU1+at]=0.0;pres[NB_PRES_F_KU2+at]=0.0;
    pres[NB_PRES_F_D+at]=0.0;pres[NB_PRES_F_Q+at]=0.0;pres[NB_PRES_F_R+at]=0.0;
  }
  pmgF[pL1(CH_D,slot,lid)]=0.0;for(var a=0u;a<3u;a++){pmgF[pL1(CH_K+a,slot,lid)]=0.0;pmgF[pL1(CH_KU+a,slot,lid)]=0.0;}
  if(lid<8u){
    let cs=nbFRec(slot,NB_REC_PARENT);let i2=nbLocal((nbFOrigin(slot)>>vec3u(2u))+pC4(lid));
    pmgC[pL2(CH_D,cs,i2)]=0.0;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i2)]=0.0;pmgC[pL2(CH_KU+a,cs,i2)]=0.0;}
  }
}`);
  // The h rows' kernels over the lists (labels "operator" and "project", the F_ALL path's areas): rowsF and
  // tileRoots on HB, projectF on HP.
  const KL = live ? pressureCompositeKernels(L, { rowsF: ["operator", "nbOperator"], projectF: ["project", "project"] }, { faces: M.FC, live: { hb: M.HBL, hp: M.HPL, rl: M.RLL, rln: M.RLN, dry: M.DRY }, cuts }) : undefined;

  // A live F slot holds the brick of tile T (seen from page p), so setupF (over F_ALL) wrote T's L2 row.
  const LIQUID_TILE = /* wgsl */`
fn pLiquidTile(p:vec3i,T:vec3i)->bool{
  let s=qPage(p,T,3u);if(s==NB_NONE){return false;}
  return nbChild(s,nbBrickLocal(T>>vec3u(1u)))<NB_SEAM;
}`;
  // Per page, a lane a tile: a native row's setup residual r = b - A x (max |r|/64), D = Q = 0, and its
  // L2 row (DIAG and the native K, or the F2 tile below's KU); an h tile without a liquid F brick is
  // cleared. Then L2 -> L3 -> L4 -> L5, and the page's native row count in L5 KU0.
  const coarsenC = kernel("pmgCoarsenC", CBP, ["pres", "cstage", "reduce", "seam"], LIQUID_TILE + SEAM_BUILD +
    nativeApplyDiff("pApplyNX", "pSeamX", (cs, i) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]`, (s, l) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]`) + /* wgsl */`
var<workgroup> w2D:array<f32,512>;var<workgroup> w2K:array<vec3f,512>;var<workgroup> wNat:atomic<u32>;
var<workgroup> w3D:array<f32,64>;var<workgroup> w3K:array<vec3f,64>;var<workgroup> w4D:array<f32,8>;var<workgroup> w4K:array<vec3f,8>;
@compute @workgroup_size(512) fn pmgCoarsenC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("all")}
  qStageW(cs,lid);
  if(lid==0u){atomicStore(&wNat,0u);}
  workgroupBarrier();
  let i=lid;let tl=nbLocalCoord(i);let T=8*p+tl;
  // The seam list: a native row's seam faces, listed in tile then face order (an exclusive scan of the counts).
  let nat=pTileIn(T)&&!nbTileH(cs,i);
  var D0=0.0;if(nat){D0=pP(NB_PRES_C_DIAG,cs,i);}
  var fsv:array<u32,6>;var cnt=0u;
  for(var f=0u;f<6u;f++){fsv[f]=NB_NONE;if(nat&&D0>0.0){fsv[f]=pSeamBrick(T,f);if(fsv[f]!=NB_NONE){cnt++;}}}
  let st=pScan512(lid,cnt);
  var k=st;
  for(var f=0u;f<6u;f++){if(fsv[f]!=NB_NONE){pmgC[P_SEAML+3072u*cs+k]=f32((i<<3u)|f);pmgC[P_SEAMF+3072u*cs+k]=f32(fsv[f]);k++;}}
  pmgC[P_SEAMT+512u*cs+i]=f32((st<<3u)|cnt);
  let n=workgroupUniformLoad(&wSN);
  if(lid==0u){pmgC[P_SEAMN+cs]=f32(n);}
  storageBarrier();workgroupBarrier();
  let sm=pSeamX(p,cs,lid,n);
  var D=0.0;var K=vec3f(0.0);var r=0.0;
  if(nat){
    let at=NB_STRIDE*cs+i;
    D=D0;
    if(D>0.0){
      atomicAdd(&wNat,1u);
      for(var a=0u;a<3u;a++){
        let Tl=T-pUnit(a);
        if(pNativeTile(Tl)!=NB_NONE){K[a]=pCK(a,cs,i);}
        else if(pTileIn(Tl)&&pLiquidTile(p,Tl)){K[a]=pmgC[pL2(CH_KU+a,qPage(p,Tl,3u),nbLocal(Tl))];}
      }
      r=pP(NB_PRES_C_RHS,cs,i)-pApplyNX(p,cs,i,T,sm);
    }
    pres[NB_PRES_C_R+at]=r;pres[NB_PRES_C_D+at]=0.0;pres[NB_PRES_C_Q+at]=0.0;
    pmgC[pL2(CH_D,cs,i)]=D;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i)]=K[a];}
  }
  else if(pLiquidTile(p,T)){D=pmgC[pL2(CH_D,cs,i)];K=vec3f(pmgC[pL2(CH_K,cs,i)],pmgC[pL2(CH_K+1u,cs,i)],pmgC[pL2(CH_K+2u,cs,i)]);}
  else{pmgC[pL2(CH_D,cs,i)]=0.0;for(var a=0u;a<3u;a++){pmgC[pL2(CH_K+a,cs,i)]=0.0;}}
  w2D[i]=max(D,0.0);w2K[i]=K;
  let rmax=pMax512(lid,abs(r)/64.0);
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));atomicMax(&nbSet[PH_NSCALE],bitcast<u32>(rmax));}
  if(lid<64u){
    let c3=pC1(lid);var D3=0.0;var K3=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=2*c3+pC4(m);let k=nbLocal(q);D3+=w2D[k];for(var a=0u;a<3u;a++){let w=w2K[k][a];if((q[a]&1)==1){D3-=2.0*w;}else{K3[a]+=w;}}}
    D3/=P_OC;K3/=P_OC;w3D[lid]=D3;w3K[lid]=K3;
    pmgC[pL3(CH_D,cs,lid)]=D3;for(var a=0u;a<3u;a++){pmgC[pL3(CH_K+a,cs,lid)]=K3[a];}
  }
  workgroupBarrier();
  if(lid<8u){
    let c4=pC4(lid);var D4=0.0;var K4=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=2*c4+pC4(m);let i=pI1(q);D4+=w3D[i];for(var a=0u;a<3u;a++){let w=w3K[i][a];if((q[a]&1)==1){D4-=2.0*w;}else{K4[a]+=w;}}}
    D4/=P_OC;K4/=P_OC;w4D[lid]=D4;w4K[lid]=K4;
    pmgC[pL4(CH_D,cs,lid)]=D4;for(var a=0u;a<3u;a++){pmgC[pL4(CH_K+a,cs,lid)]=K4[a];}
  }
  workgroupBarrier();
  if(lid==0u){
    var D5=0.0;var K5=vec3f(0.0);
    for(var m=0u;m<8u;m++){let q=pC4(m);D5+=w4D[m];for(var a=0u;a<3u;a++){let w=w4K[m][a];if((q[a]&1)==1){D5-=2.0*w;}else{K5[a]+=w;}}}
    D5/=P_OC;K5/=P_OC;
    pmgC[pL5(CH_D,cs)]=D5;for(var a=0u;a<3u;a++){pmgC[pL5(CH_K+a,cs)]=K5[a];}
    pmgC[pL5(CH_KU,cs)]=f32(atomicLoad(&wNat));
  }
}`);

  const TOPCOMMON = PAGE + /* wgsl */`
fn tF(i:u32)->f32{return bitcast<f32>(pmgTop[i]);}
fn tS(i:u32,v:f32){pmgTop[i]=bitcast<u32>(v);}
fn sbwb(){storageBarrier();workgroupBarrier();}
fn tPage(cs:u32)->vec3i{return nbKeyCoord(nbCRec(cs,NB_REC_KEY));}
// The C slot across face (a, +-1) of page cs, or NONE.
fn tNbr(cs:u32,a:u32,up:bool)->u32{return pPageSlot(tPage(cs)+select(-pUnit(a),pUnit(a),up));}
`;
  const hashInsert = (name: string, keys: string, size: number) => /* wgsl */`
// Linear probing; returns the position (bit 31 set: inserted here), or NONE when full.
fn ${name}(key:u32)->u32{
  var h=(key*2654435761u)>>${32 - Math.log2(size)}u;var probes=0u;
  loop{
    if(probes>=${size}u){return NB_NONE;}
    let r=atomicCompareExchangeWeak(&${keys}[h],NB_NONE,key);
    if(r.exchanged){return h|0x80000000u;}
    if(r.old_value==key){return h;}
    if(r.old_value!=NB_NONE){h=(h+1u)&${size - 1}u;probes++;}
  }
}`;

  // One workgroup: L5 rows (one a page), L6 and L7 by hashing keys, their Galerkin
  // operators, L7's inverse, and slot 0's gate (the check before the first V-cycle).
  const coarsenTop = kernel("pmgCoarsenTop", [CFG, setRW, dirRO, cRW, topRW, argsRW], ["gate"], TOPCOMMON + /* wgsl */`
var<workgroup> hk6:array<atomic<u32>,${HS6}>;var<workgroup> hv6:array<u32,${HS6}>;
var<workgroup> hk7:array<atomic<u32>,${HS7}>;var<workgroup> hv7:array<u32,${HS7}>;
var<workgroup> cN:array<atomic<u32>,4>;
var<workgroup> gA:array<f32,${MG_ROWS7 * MG_ROWS7}>;
var<workgroup> wNC:u32;var<workgroup> wN5:u32;var<workgroup> wN6:u32;var<workgroup> wN7:u32;var<workgroup> wRB:atomic<u32>;var<workgroup> wSB:atomic<u32>;
${hashInsert("hIns6", "hk6", HS6)}
${hashInsert("hIns7", "hk7", HS7)}
@compute @workgroup_size(1024) fn pmgCoarsenTop(@builtin(local_invocation_index) t:u32){
  // The row bricks: F_ALL's slots with an h row (RBF, pmgSetupF's), listed for the slot's brick kernels.
  if(t==0u){atomicStore(&wRB,0u);atomicStore(&wSB,0u);}
  workgroupBarrier();
  let nF=nbListCount(NB_LIST_F_ALL);
  for(var e=t;e<nF;e+=1024u){
    let sl=nbListSlot(NB_LIST_F_ALL,e);if(pmgTop[T_RBF+sl]==0u){continue;}
    pmgTop[T_RB+atomicAdd(&wRB,1u)]=sl;
    // The shifted blocks (L0's S passes): block q, origin 8q - (4, 0, 4), holds bricks (qx-1..qx, b.y, qz-1..qz) and is
    // run by its first row brick (x fastest), listed as (slot << 2) | k for q = (b.x + (k & 1), b.y, b.z + (k >> 1)).
    for(var k=0u;k<4u;k++){
      var own=false;
      for(var m=0u;m<4u;m++){
        let d=vec3i(i32(k&1u)-1+i32(m&1u),0,i32(k>>1u)-1+i32(m>>1u));let n=nbFRec(sl,nbNeighbourIndex(d));
        if(n<NB_SEAM&&pmgTop[T_RBF+n]!=0u){own=all(d==vec3i(0));break;}
      }
      if(own){pmgTop[T_SB+atomicAdd(&wSB,1u)]=(sl<<2u)|k;}
    }
  }
  workgroupBarrier();
  if(t==0u){pmgTop[T_RBN]=atomicLoad(&wRB);pmgTop[T_SBN]=atomicLoad(&wSB);}
  for(var j=t;j<${HS6}u;j+=1024u){atomicStore(&hk6[j],NB_NONE);}
  if(t<${HS7}u){atomicStore(&hk7[t],NB_NONE);}
  if(t<4u){atomicStore(&cN[t],0u);}
  if(t==0u){wNC=nbListCount(NB_LIST_C_ALL);}
  let nC=workgroupUniformLoad(&wNC);
  for(var e=t;e<nC;e+=1024u){
    let cs=nbListSlot(NB_LIST_C_ALL,e);
    if(pmgC[pL5(CH_D,cs)]>0.0){let i=atomicAdd(&cN[0],1u);pmgTop[T_SLOT5+i]=cs;pmgTop[T_IDX5+cs]=i;}else{pmgTop[T_IDX5+cs]=NB_NONE;}
  }
  sbwb();
  if(t==0u){wN5=atomicLoad(&cN[0]);}
  let n5=workgroupUniformLoad(&wN5);
  for(var i=t;i<n5;i+=1024u){
    let k6=nbKey(tPage(pmgTop[T_SLOT5+i])>>vec3u(1u));let h=hIns6(k6);
    if(h==NB_NONE){atomicOr(&cN[3],1u);pmgTop[T_PAR5+i]=0u;}
    else{let hp=h&0x7fffffffu;if(h!=hp){let r=atomicAdd(&cN[1],1u);hv6[hp]=r;if(r<P_ROWS6){pmgTop[T_KEY6+r]=k6;}}pmgTop[T_PAR5+i]=hp;}
  }
  sbwb();
  if(t==0u){wN6=select(atomicLoad(&cN[1]),P_ROWS6+1u,atomicLoad(&cN[3])!=0u);}
  let n6=workgroupUniformLoad(&wN6);
  if(n6>P_ROWS6){if(t==0u){nbFail(NB_FAIL_UNSUPPORTED,0x6000000u|n6);pGate(false);}return;}
  for(var i=t;i<n5;i+=1024u){
    pmgTop[T_PAR5+i]=hv6[pmgTop[T_PAR5+i]];
    let cs=pmgTop[T_SLOT5+i];tS(T_D5+i,pmgC[pL5(CH_D,cs)]);
    for(var a=0u;a<3u;a++){
      let wl=tNbr(cs,a,false);var jl=NB_NONE;if(wl!=NB_NONE){jl=pmgTop[T_IDX5+wl];}
      pmgTop[T_NBR5+6u*i+2u*a]=jl;tS(T_C5+6u*i+2u*a,select(0.0,pmgC[pL5(CH_K+a,cs)],jl!=NB_NONE));
      let wu=tNbr(cs,a,true);var ju=NB_NONE;if(wu!=NB_NONE){ju=pmgTop[T_IDX5+wu];}
      pmgTop[T_NBR5+6u*i+2u*a+1u]=ju;var ku=0.0;if(ju!=NB_NONE){ku=pmgC[pL5(CH_K+a,wu)];}tS(T_C5+6u*i+2u*a+1u,ku);
    }
  }
  for(var j=t;j<8u*n6;j+=1024u){pmgTop[T_CH6+j]=NB_NONE;}
  sbwb();
  for(var i=t;i<n5;i+=1024u){pmgTop[T_CH6+8u*pmgTop[T_PAR5+i]+pI4(tPage(pmgTop[T_SLOT5+i]))]=i;}
  sbwb();
  // L6 rows: D, and the six couplings (lower faces crossing out of the row, then upper).
  for(var r=t;r<n6;r+=1024u){
    var D=0.0;var K=array<f32,6>(0.0,0.0,0.0,0.0,0.0,0.0);var nb=array<u32,6>(NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE);
    for(var k=0u;k<8u;k++){
      let i=pmgTop[T_CH6+8u*r+k];if(i==NB_NONE){continue;}
      D+=tF(T_D5+i);
      for(var f=0u;f<6u;f++){
        let j=pmgTop[T_NBR5+6u*i+f];let w=tF(T_C5+6u*i+f);if(j==NB_NONE||w==0.0){continue;}
        let pj=pmgTop[T_PAR5+j];
        if(pj==r){if((f&1u)==0u){D-=2.0*w;}}else{K[f]+=w;nb[f]=pj;}
      }
    }
    tS(T_D6+r,D/P_OC);
    for(var f=0u;f<6u;f++){pmgTop[T_NBR6+6u*r+f]=select(NB_NONE,nb[f],K[f]!=0.0);tS(T_C6+6u*r+f,K[f]/P_OC);}
    let k7=nbKey(nbKeyCoord(pmgTop[T_KEY6+r])>>vec3u(1u));let h=hIns7(k7);
    if(h==NB_NONE){atomicOr(&cN[3],2u);pmgTop[T_PAR6+r]=0u;}
    else{let hp=h&0x7fffffffu;if(h!=hp){let q=atomicAdd(&cN[2],1u);hv7[hp]=q;if(q<P_ROWS7){pmgTop[T_KEY7+q]=k7;}}pmgTop[T_PAR6+r]=hp;}
  }
  sbwb();
  if(t==0u){wN7=select(atomicLoad(&cN[2]),P_ROWS7+1u,atomicLoad(&cN[3])!=0u);}
  let n7=workgroupUniformLoad(&wN7);
  if(n7>P_ROWS7){if(t==0u){nbFail(NB_FAIL_UNSUPPORTED,0x7000000u|n7);pGate(false);}return;}
  for(var r=t;r<n6;r+=1024u){pmgTop[T_PAR6+r]=hv7[pmgTop[T_PAR6+r]];}
  for(var j=t;j<8u*n7;j+=1024u){pmgTop[T_CH7+j]=NB_NONE;}
  for(var e=t;e<${MG_ROWS7 * MG_ROWS7}u;e+=1024u){let i=e/P_ROWS7;let j=e%P_ROWS7;gA[e]=select(0.0,1.0,i==j&&i>=n7);}
  sbwb();
  for(var r=t;r<n6;r+=1024u){pmgTop[T_CH7+8u*pmgTop[T_PAR6+r]+pI4(nbKeyCoord(pmgTop[T_KEY6+r]))]=r;}
  sbwb();
  // L7: the dense operator, row R by lane R.
  if(t<n7){
    var D=0.0;
    for(var k=0u;k<8u;k++){
      let r=pmgTop[T_CH7+8u*t+k];if(r==NB_NONE){continue;}
      D+=tF(T_D6+r);
      for(var f=0u;f<6u;f++){
        let j=pmgTop[T_NBR6+6u*r+f];let w=tF(T_C6+6u*r+f);if(j==NB_NONE||w==0.0){continue;}
        let pj=pmgTop[T_PAR6+j];
        if(pj==t){if((f&1u)==0u){D-=2.0*w;}}else{gA[t*P_ROWS7+pj]-=w/P_OC;}
      }
    }
    gA[t*P_ROWS7+t]=D/P_OC;
  }
  workgroupBarrier();
  // Gauss-Jordan without pivoting (SPD), four entries a lane.
  for(var pv=0u;pv<n7;pv++){
    let app=gA[pv*P_ROWS7+pv];
    var a:array<f32,4>;var aip:array<f32,4>;var apj:array<f32,4>;
    for(var m=0u;m<4u;m++){let e=t+1024u*m;let i=e/P_ROWS7;let j=e%P_ROWS7;a[m]=gA[e];aip[m]=gA[i*P_ROWS7+pv];apj[m]=gA[pv*P_ROWS7+j];}
    workgroupBarrier();
    if(t==0u&&!(app>0.0)){nbFail(NB_FAIL_PIVOT,pv);}
    let rp=1.0/app;
    for(var m=0u;m<4u;m++){
      let e=t+1024u*m;let i=e/P_ROWS7;let j=e%P_ROWS7;
      var v=a[m]-aip[m]*apj[m]*rp;
      if(i==pv&&j==pv){v=rp;}else if(i==pv){v=apj[m]*rp;}else if(j==pv){v=-aip[m]*rp;}
      gA[e]=v;
    }
    workgroupBarrier();
  }
  for(var e=t;e<${MG_ROWS7 * MG_ROWS7}u;e+=1024u){tS(T_INV+e,gA[e]);}
  if(t==0u){
    let rmax=atomicExchange(&nbSet[PH_RMAX],0u);
    atomicStore(&nbSet[PH_RMAX0],rmax);atomicStore(&nbSet[PH_RLAST],rmax);pHist(0u,rmax);
    atomicStore(&nbSet[PH_N5],n5);atomicStore(&nbSet[PH_N6],n6);atomicStore(&nbSet[PH_N7],n7);
    // The native rows' bound (the old root's nBound, uniform-pressure-arena.ts): the tolerance, or a tenth of their
    // initial max|r| down to the floor nFloor (1e-4 1/s, band-equivalent), whichever is smaller. The slots count a
    // native row's residual times tol / nBound (nStopScale), so one threshold stops both; a native start above the
    // floor always runs (it is met only at a tenth of itself), so the stored pressure's imbalance cannot accumulate
    // under the tolerance at rest (the old tank's natives sit at f32 round-off).
    let tolR=pTolR();let hh=nbRingF(cfg.x,NB_RING_H);let floorR=1e-4*hh*hh;let nr=bitcast<f32>(atomicLoad(&nbSet[PH_NSCALE]));
    let nb=min(tolR,max(0.1*nr,floorR));atomicStore(&nbSet[PH_NSCALE],bitcast<u32>(select(1.0,tolR/nb,nb>0.0&&nb<tolR)));
    pGate(bitcast<f32>(rmax)>tolR||nr>floorR);
  }
}`);

  // ---- the slot ------------------------------------------------------------
  // Level 0 passes: the bricks (U) or the shifted blocks (S), from plane src to dst.
  const fineKernel = (label: string, o: { shift: boolean; src: "W0" | "W1"; prolong?: boolean; dst: "W0" | "W1" | "Z"; post: boolean; dot?: boolean }) => {
    const plane = (p: "W0" | "W1") => p === "W0" ? "P_W0" : "P_W1";
    const read = `pmgF[${plane(o.src)}+NB_STRIDE*s+l]` + (o.prolong ? `+pmgF[pL1(CH_EPOST,s,pI1(q>>vec3u(1u)))]` : "");
    const write = o.dst === "Z" ? `pres[NB_PRES_F_Z+NB_STRIDE*s+nbLocal(c)]` : `pmgF[${plane(o.dst)}+NB_STRIDE*s+nbLocal(c)]`;
    return kernel(label, o.dot ? [...FBT, redRW] : FBT, ["pres", "stage", "tile", ...(o.dot ? ["reduce" as const, "red" as const] : [])], /* wgsl */`
@compute @workgroup_size(512) fn ${label}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${o.shift ? shiftHead : rowHead}
  let lc=nbLocalCoord(lid);let ti=pTi(lc);var dotv=0.0;
  {
    ${o.shift ? /* wgsl */`
    // Block q (origin 8q - (4,0,4)) holds bricks (qx-1..qx, b.y, qz-1..qz); its first row brick runs it (pmgTop SB).
    let ob=8*vec3i(b.x+i32(k&1u),b.y,b.z+i32(k>>1u))-vec3i(4,0,4);` : "let ob=o;"}
    // A native cell holds its row's carried value: 0 before its own pass (pmgNativePre0), N0P after (pmgNativePost0).
    for(var t=lid;t<1000u;t+=512u){
      let q=pTileQ(ob,t);var v=0.0;let nq=pNatAt(b,q);
      if(nq!=NB_NONE){${o.post ? "v=pmgC[pL2(CH_N0P,nq,nbLocal(q>>vec3u(2u)))];" : ""}}
      else{let s=pSlotAt(b,q);if(s!=NB_NONE){let l=nbLocal(q);v=${read};}}
      tp[t]=v;
    }
    workgroupBarrier();
    let c=ob+lc;let row=pRow(b,c);let s=pSlotAt(b,c);
    var rhs=0.0;if(row.d>0.0){rhs=pP(NB_PRES_F_R,s,nbLocal(c));}
    pSmooth(row,c,ti,rhs,${o.post});
    if(row.d>0.0){${write}=tp[ti];${o.dot ? "dotv+=rhs*tp[ti];" : ""}}
    ${o.dst === "Z" ? `else if(s!=NB_NONE){${write}=0.0;}` : ""}
  }
  ${o.dot ? "let sum=pSum512(lid,dotv);if(lid==0u){pmgRed[R_ZF+g]=sum;}" : ""}
}`);
  };
  const preS = fineKernel("pmgPreS", { shift: true, src: "W0", dst: "W1", post: false });
  const preU = fineKernel("pmgPreU", { shift: false, src: "W1", dst: "W0", post: false });
  const postU = fineKernel("pmgPostU", { shift: false, src: "W0", prolong: true, dst: "W1", post: true });
  // BS: post-smoothing opens with the shifted pass, from the pre-smoothed W1 plus the prolonged L1 correction.
  const postS = fineKernel("pmgPostS", { shift: true, src: "W1", prolong: !bsb, dst: "W0", post: true });
  const postZ = fineKernel("pmgPostZ", { shift: false, src: "W0", dst: "Z", post: true, dot: true });

  // Red-black passes inside a coarse block. Caller loops; `cur` is the workgroup vector.
  const coarsePasses = (rev: boolean, cond: string, update: string) => /* wgsl */`
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    if(${cond}&&par==colour){${update}}
    workgroupBarrier();
  }`;

  // D0: the level 0 residual restricted to L1 (the native rows at their pass's N0), and L1's pre-smoothing (the
  // brick, zero halo; the native rows carried at 0).
  const down0 = kernel("pmgDown0", FBT, ["pres", "stage"], fineApply("pApplyW0", (s, l) => `pmgF[${PRE}+NB_STRIDE*${s}+${l}]`, (cs, i) => `pmgC[pL2(CH_N0,${cs},${i})]`) + /* wgsl */`
var<workgroup> wD:array<f32,512>;var<workgroup> wE:array<f32,64>;
@compute @workgroup_size(512) fn pmgDown0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let l=lid;let c=o+nbLocalCoord(l);
  var res=0.0;if(pDiag(slot,l)>0.0){res=pP(NB_PRES_F_R,slot,l)-pApplyW0(b,c,slot,l);}
  wD[l]=res;
  if(lid<64u){wE[lid]=0.0;}
  workgroupBarrier();
  let c1=pC1(lid&63u);let par=pPar((o>>vec3u(1u))+c1);
  var b1=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<64u){
    for(var m=0u;m<8u;m++){b1+=wD[nbLocal(2*c1+pC4(m))];}
    D=pmgF[pL1(CH_D,slot,lid)];
    for(var a=0u;a<3u;a++){
      if(c1[a]>0){K[a]=pmgF[pL1(CH_K+a,slot,lid)];}
      if(c1[a]<3){KU[a]=pmgF[pL1(CH_K+a,slot,pI1(c1+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<64u&&D>0.0", `var s=b1;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI1(c1-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI1(c1+e)];}}wE[lid]=s/D;`)}
  if(lid<64u){pmgF[pL1(CH_B,slot,lid)]=b1;pmgF[pL1(CH_EPRE,slot,lid)]=wE[lid];}
}`);

  // D1: the L1 residual restricted to the brick's 8 tiles' L2 B (a native row's is pmgDown1C's); a native
  // neighbour at its L1 pass's N1, across the aggregate's K (below) or KU (above).
  const down1 = kernel("pmgDown1", FBT, ["stage"], /* wgsl */`
var<workgroup> wR:array<f32,64>;
// The L1 row c (global 2h) lives in the brick holding fine cell 2c.
fn pV1(b:vec3i,c:vec3i,ch:u32)->f32{let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(ch,s,pI1(c))];}
@compute @workgroup_size(64) fn pmgDown1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let c1=(o>>vec3u(1u))+pC1(lid);let D1=pmgF[pL1(CH_D,slot,lid)];
  var r=0.0;
  if(D1>0.0){
    r=pmgF[pL1(CH_B,slot,lid)]-D1*pmgF[pL1(CH_EPRE,slot,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgF[pL1(CH_K+a,slot,lid)];
      if(kl!=0.0){let cn=2*(c1-e);let nn=pNatAt(b,cn);if(nn!=NB_NONE){r+=kl*pmgC[pL2(CH_N1,nn,nbLocal(cn>>vec3u(2u)))];}else{r+=kl*pV1(b,c1-e,CH_EPRE);}}
      let cu=2*(c1+e);let nu=pNatAt(b,cu);
      if(nu!=NB_NONE){let ku=pmgF[pL1(CH_KU+a,slot,lid)];if(ku!=0.0){r+=ku*pmgC[pL2(CH_N1,nu,nbLocal(cu>>vec3u(2u)))];}}
      else{let ku=pV1(b,c1+e,CH_K+a);if(ku!=0.0){r+=ku*pV1(b,c1+e,CH_EPRE);}}
    }
  }
  wR[lid]=r;
  workgroupBarrier();
  if(lid<8u){
    let c2l=pC4(lid);var b2=0.0;
    for(var m=0u;m<8u;m++){b2+=wR[pI1(2*c2l+pC4(m))];}
    pmgC[pL2(CH_B,nbFRec(slot,NB_REC_PARENT),nbLocal((o>>vec3u(2u))+c2l))]=b2;
  }
}`);

  // D1C (a page): a native row's L2 B, its L1 residual R1 - A1 e1 (e1 its N1 and the h aggregates' EPRE; the L1
  // seam link is the h rows' seams over sqrt(OC)), then L2's pre-smoothing over every tile, 2^3 tile blocks (the
  // bricks), zero halo.
  const down1C = kernel("pmgDown1C", FB, ["pres", "cstage", "seam"],
    nativeApply("pApplyN1", "pSeamE1", (cs, i) => `pmgC[pL2(CH_N1,${cs},${i})]`, (s, l, m) => `pmgF[pL1(CH_EPRE,${s},pI1(${m}>>vec3u(1u)))]/P_SOC`) + /* wgsl */`
var<workgroup> wE:array<f32,512>;
@compute @workgroup_size(512) fn pmgDown1C(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2l=nbLocalCoord(lid);let T=8*p+c2l;let par=pPar(T);
  let sm=pSeamE1(p,cs,lid,pSeamN(cs,lid));
  let D=pmgC[pL2(CH_D,cs,lid)];var b2=0.0;
  if(D>0.0){
    if(pTileIn(T)&&!nbTileH(cs,lid)){b2=pmgC[pL2(CH_R1,cs,lid)]-pApplyN1(p,cs,lid,T,sm);pmgC[pL2(CH_B,cs,lid)]=b2;}
    else{b2=pmgC[pL2(CH_B,cs,lid)];}
  }
  var K=vec3f(0.0);var KU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    if((c2l[a]&1)==1){K[a]=pmgC[pL2(CH_K+a,cs,lid)];}
    else{KU[a]=pmgC[pL2(CH_K+a,cs,nbLocal(c2l+pUnit(a)))];}
  }
  wE[lid]=0.0;
  workgroupBarrier();
  ${coarsePasses(false, "D>0.0", `var s=b2;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[nbLocal(c2l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[nbLocal(c2l+e)];}}wE[lid]=s/D;`)}
  pmgC[pL2(CH_EPRE,cs,lid)]=wE[lid];
}`);

  // Values of a page level by global level coordinate (cells a page side 1 << sh).
  const pageValue = (fn: string, at: string, sh: number, local: string) => /* wgsl */`
fn ${fn}(p:vec3i,c:vec3i,ch:u32)->f32{let s=qPage(p,c,${sh}u);if(s==NB_NONE){return 0.0;}return pmgC[${at}(ch,s,${local})];}`;
  const V2 = pageValue("pV2", "pL2", 3, "nbLocal(c)"), V3 = pageValue("pV3", "pL3", 2, "pI1(c)"), V4 = pageValue("pV4", "pL4", 1, "pI4(c)");

  // D2 (a page): the L2 residual restricted to L3, and L3's pre-smoothing (16h blocks, zero halo).
  const down2 = kernel("pmgDown2", CB, ["cstage"], V2 + /* wgsl */`
var<workgroup> wR:array<f32,512>;var<workgroup> wE:array<f32,64>;
@compute @workgroup_size(512) fn pmgDown2(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2=8*p+nbLocalCoord(lid);let D2=pmgC[pL2(CH_D,cs,lid)];
  var r=0.0;
  if(D2>0.0){
    r=pmgC[pL2(CH_B,cs,lid)]-D2*pmgC[pL2(CH_EPRE,cs,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgC[pL2(CH_K+a,cs,lid)];
      if(kl!=0.0){r+=kl*pV2(p,c2-e,CH_EPRE);}
      let ku=pV2(p,c2+e,CH_K+a);if(ku!=0.0){r+=ku*pV2(p,c2+e,CH_EPRE);}
    }
  }
  wR[lid]=r;if(lid<64u){wE[lid]=0.0;}
  workgroupBarrier();
  let c3l=pC1(lid&63u);let par=pPar(4*p+c3l);
  var b3=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<64u){
    for(var m=0u;m<8u;m++){b3+=wR[nbLocal(2*c3l+pC4(m))];}
    D=pmgC[pL3(CH_D,cs,lid)];
    for(var a=0u;a<3u;a++){
      if((c3l[a]&1)==1){K[a]=pmgC[pL3(CH_K+a,cs,lid)];}
      else{KU[a]=pmgC[pL3(CH_K+a,cs,pI1(c3l+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<64u&&D>0.0", `var s=b3;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI1(c3l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI1(c3l+e)];}}wE[lid]=s/D;`)}
  if(lid<64u){pmgC[pL3(CH_B,cs,lid)]=b3;pmgC[pL3(CH_EPRE,cs,lid)]=wE[lid];}
}`);

  // D3 (a page): the L3 residual restricted to L4, L4's pre-smoothing (the page, zero halo), and the parts of the L4
  // residual (CH4R) that the top sums into the page's L5 row: at 0 the page's own part (its rows' residuals with the
  // couplings inside the page, in cell order), and at 1 + 2a + s (s 1 the upper side) the couplings across its face
  // (a, s) times its E there, which the page across adds. The single-workgroup top then reads 13 words a page, not
  // the 8 cells' 170-odd scattered loads (fill-the-GPU: the gather moves onto the pages' workgroups).
  const down3 = kernel("pmgDown3", CB, ["cstage"], V3 + V4 + /* wgsl */`
var<workgroup> wR:array<f32,64>;var<workgroup> wE:array<f32,8>;
@compute @workgroup_size(64) fn pmgDown3(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c3=4*p+pC1(lid);let D3=pmgC[pL3(CH_D,cs,lid)];
  var r=0.0;
  if(D3>0.0){
    r=pmgC[pL3(CH_B,cs,lid)]-D3*pmgC[pL3(CH_EPRE,cs,lid)];
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let kl=pmgC[pL3(CH_K+a,cs,lid)];
      if(kl!=0.0){r+=kl*pV3(p,c3-e,CH_EPRE);}
      let ku=pV3(p,c3+e,CH_K+a);if(ku!=0.0){r+=ku*pV3(p,c3+e,CH_EPRE);}
    }
  }
  wR[lid]=r;if(lid<8u){wE[lid]=0.0;}
  workgroupBarrier();
  let c4l=pC4(lid&7u);let par=pPar(2*p+c4l);
  var b4=0.0;var D=0.0;var K=vec3f(0.0);var KU=vec3f(0.0);
  if(lid<8u){
    for(var m=0u;m<8u;m++){b4+=wR[pI1(2*c4l+pC4(m))];}
    D=pmgC[pL4(CH_D,cs,lid)];
    for(var a=0u;a<3u;a++){
      if(c4l[a]==1){K[a]=pmgC[pL4(CH_K+a,cs,lid)];}
      else{KU[a]=pmgC[pL4(CH_K+a,cs,pI4(c4l+pUnit(a)))];}
    }
  }
  ${coarsePasses(false, "lid<8u&&D>0.0", `var s=b4;for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){s+=K[a]*wE[pI4(c4l-e)];}if(KU[a]!=0.0){s+=KU[a]*wE[pI4(c4l+e)];}}wE[lid]=s/D;`)}
  if(lid<8u){pmgC[pL4(CH_B,cs,lid)]=b4;pmgC[pL4(CH_EPRE,cs,lid)]=wE[lid];}
  if(lid<8u){
    var r=0.0;
    if(D>0.0){r=b4-D*wE[lid];for(var a=0u;a<3u;a++){let e=pUnit(a);if(K[a]!=0.0){r+=K[a]*wE[pI4(c4l-e)];}if(KU[a]!=0.0){r+=KU[a]*wE[pI4(c4l+e)];}}}
    wR[lid]=r;
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);var w=0.0;
      if(c4l[a]==0){w=pmgC[pL4(CH_K+a,cs,lid)];}else{w=pV4(p,2*p+c4l+e,CH_K+a);}
      wR[8u+3u*lid+a]=w*wE[lid];
    }
  }
  workgroupBarrier();
  if(lid<7u){
    var s=0.0;
    if(lid==0u){for(var k=0u;k<8u;k++){s+=wR[k];}}
    else{let a=(lid-1u)>>1u;let side=i32((lid-1u)&1u);for(var k=0u;k<8u;k++){if(pC4(k)[a]==side){s+=wR[8u+3u*k+a];}}}
    pmgC[pL4(CH4R,cs,lid)]=s;
  }
}`);

  // The compact levels' red-black passes (one workgroup; frozen halo = F array, or 0 when pre). When every row of
  // L5 and L6 has its own lane (C and MG_ROWS6 <= 1024), the lane holds its row's links (D, colour, the six
  // neighbours, couplings and same-parent bits; topLinks) in registers for the whole cycle, so a pass reads only the
  // vectors; the sums are the uncached loop's, term for term.
  const topCached = L.cap.c <= 1024 && MG_ROWS6 <= 1024;
  const topLinks = (lv: 5 | 6, n: string, parOf: string) => !topCached ? "" : /* wgsl */`
  var l${lv}D=0.0;var l${lv}c=2u;var l${lv}j=array<u32,6>(NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE);var l${lv}w=array<f32,6>();var l${lv}m=0u;
  if(t<${n}){
    let i=t;l${lv}D=tF(T_D${lv}+i);l${lv}c=${parOf};let pi=pmgTop[T_PAR${lv}+i];
    for(var f=0u;f<6u;f++){
      let j=pmgTop[T_NBR${lv}+6u*i+f];l${lv}j[f]=j;
      if(j!=NB_NONE){l${lv}w[f]=tF(T_C${lv}+6u*i+f);if(pmgTop[T_PAR${lv}+j]==pi){l${lv}m|=1u<<f;}}
    }
  }`;
  const topPasses = (lv: 5 | 6, rev: boolean, n: string, parOf: string) => topCached ? /* wgsl */`
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    if(l${lv}D>0.0&&l${lv}c==colour){
      var s=tF(T_B${lv}+t);
      for(var f=0u;f<6u;f++){
        let j=l${lv}j[f];if(j==NB_NONE){continue;}
        if((l${lv}m&(1u<<f))!=0u){s+=l${lv}w[f]*tF(T_E${lv}+j);}${rev ? `else{s+=l${lv}w[f]*tF(T_F${lv}+j);}` : ""}
      }
      tS(T_E${lv}+t,s/l${lv}D);
    }
    tbar();
  }` : /* wgsl */`
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    for(var i=t;i<${n};i+=1024u){
      let D=tF(T_D${lv}+i);
      if(D>0.0&&${parOf}==colour){
        var s=tF(T_B${lv}+i);let pi=pmgTop[T_PAR${lv}+i];
        for(var f=0u;f<6u;f++){
          let j=pmgTop[T_NBR${lv}+6u*i+f];if(j==NB_NONE){continue;}
          let w=tF(T_C${lv}+6u*i+f);
          if(pmgTop[T_PAR${lv}+j]==pi){s+=w*tF(T_E${lv}+j);}${rev ? `else{s+=w*tF(T_F${lv}+j);}` : ""}
        }
        tS(T_E${lv}+i,s/D);
      }
    }
    tbar();
  }`;
  const topResidual = (lv: 5 | 6, n: string) => topCached ? /* wgsl */`
  if(t<${n}){
    let e=tF(T_E${lv}+t);tS(T_P${lv}+t,e);
    var r=tF(T_B${lv}+t)-l${lv}D*e;
    for(var f=0u;f<6u;f++){let j=l${lv}j[f];if(j!=NB_NONE){r+=l${lv}w[f]*tF(T_E${lv}+j);}}
    tS(T_R${lv}+t,r);
  }
  tbar();` : /* wgsl */`
  for(var i=t;i<${n};i+=1024u){
    let e=tF(T_E${lv}+i);tS(T_P${lv}+i,e);
    var r=tF(T_B${lv}+i)-tF(T_D${lv}+i)*e;
    for(var f=0u;f<6u;f++){let j=pmgTop[T_NBR${lv}+6u*i+f];if(j!=NB_NONE){r+=tF(T_C${lv}+6u*i+f)*tF(T_E${lv}+j);}}
    tS(T_R${lv}+i,r);
  }
  tbar();`;

  // Top (one workgroup): L4's residual into L5, L5..L7 (exact), and L4's post-smoothing. The compact levels' vectors
  // (B, E, F, P, R of L5 and L6; B, E of L7) live only inside this kernel, so they are held in workgroup memory
  // (wv; 4 (5 C + 5 ROWS6 + 2 ROWS7) bytes, 14.6 KB at C 192) instead of pmgTop when they fit: every pass of the one
  // workgroup then reads them at threadgroup latency, not device memory's. pmgTop keeps their words (the larger pools').
  const C = L.cap.c, wvWords = 5 * C + 5 * MG_ROWS6 + 2 * MG_ROWS7, wvFits = 4 * wvWords <= MG_TOP_WG_BYTES;
  const wvAt: Record<string, number> = {};
  ["B", "E", "F", "P", "R"].forEach((x, k) => { wvAt[`${x}5`] = k * C; wvAt[`${x}6`] = 5 * C + k * MG_ROWS6; });
  wvAt.B7 = 5 * C + 5 * MG_ROWS6; wvAt.E7 = wvAt.B7 + MG_ROWS7;
  const topVectors = (code: string) => !wvFits ? "fn tbar(){storageBarrier();workgroupBarrier();}" + code : /* wgsl */`
var<workgroup> wv:array<f32,${wvWords}>;
fn tbar(){workgroupBarrier();}
fn vF(i:u32)->f32{return wv[i];}
fn vS(i:u32,v:f32){wv[i]=v;}
${Object.entries(wvAt).map(([n, v]) => `const V_${n}:u32=${u(v)};`).join("")}
` + code.replace(/t([FS])\(T_([BEFPR][567])\+/g, (_m, fs: string, n: string) => {
    if (wvAt[n] === undefined) throw new Error(`pmgTopCycle: no workgroup vector ${n}`);
    return `v${fs}(V_${n}+`;
  });
  // The L4 post-smoothing's cells a lane each, their E in wv's L6 words, when they fit (8 C lanes and words).
  const l4Cached = wvFits && 8 * C <= 1024 && 8 * C <= 5 * MG_ROWS6;
  const top = kernel("pmgTopCycle", [CFG, setRW, dirRO, cRW, topRW], [], TOPCOMMON + topVectors(/* wgsl */`
var<workgroup> wN5:u32;var<workgroup> wN6:u32;var<workgroup> wN7:u32;
fn par5(i:u32)->u32{return pPar(tPage(pmgTop[T_SLOT5+i]));}
fn par6(r:u32)->u32{return pPar(nbKeyCoord(pmgTop[T_KEY6+r]));}
// L4 cell k of page cs: the page holding the neighbour across (a, up) (from the page's stage cache, QSTG, whose
// index steps 1, 3, 9 along x, y, z are nbNeighbourOffset's), and its local index.
fn l4Nbr(cs:u32,k:u32,a:u32,up:bool)->vec2u{
  let c=pC4(k);var d=c;if(up){d[a]+=1;}else{d[a]-=1;}
  if(d[a]>=0&&d[a]<=1){return vec2u(cs,pI4(d));}
  let st=select(select(9u,3u,a==1u),1u,a==0u);let v=pmgC[P_QSTG+27u*cs+select(13u-st,13u+st,up)];
  return vec2u(select(u32(v),NB_NONE,v<0.0),pI4(d));
}
@compute @workgroup_size(1024) fn pmgTopCycle(@builtin(local_invocation_index) t:u32){
  if(t==0u){wN5=min(atomicLoad(&nbSet[PH_N5]),NB_CCAP);wN6=min(atomicLoad(&nbSet[PH_N6]),P_ROWS6);wN7=min(atomicLoad(&nbSet[PH_N7]),P_ROWS7);}
  let n5=workgroupUniformLoad(&wN5);let n6=workgroupUniformLoad(&wN6);let n7=workgroupUniformLoad(&wN7);
${topLinks(5, "n5", "par5(i)")}
${topLinks(6, "n6", "par6(i)")}
  // L4's residual into each L5 row: the page's own part and its face neighbours' (pmgDown3's CH4R; a page across
  // with no rows, IDX5 NONE, has no part this step).
  for(var i=t;i<n5;i+=1024u){
    let cs=pmgTop[T_SLOT5+i];var sum=pmgC[pL4(CH4R,cs,0u)];
    for(var a=0u;a<3u;a++){
      let st=select(select(9u,3u,a==1u),1u,a==0u);
      for(var s=0u;s<2u;s++){
        let v=pmgC[P_QSTG+27u*cs+select(13u-st,13u+st,s==1u)];
        if(v>=0.0){let q=u32(v);if(pmgTop[T_IDX5+q]!=NB_NONE){sum+=pmgC[pL4(CH4R,q,2u+2u*a-s)];}}
      }
    }
    tS(T_B5+i,sum);tS(T_E5+i,0.0);
  }
  for(var r=t;r<n6;r+=1024u){tS(T_E6+r,0.0);}
  sbwb();
  ${topPasses(5, false, "n5", "par5(i)")}
  ${topResidual(5, "n5")}
  for(var r=t;r<n6;r+=1024u){var s=0.0;for(var k=0u;k<8u;k++){let i=pmgTop[T_CH6+8u*r+k];if(i!=NB_NONE){s+=tF(T_R5+i);}}tS(T_B6+r,s);}
  sbwb();
  ${topPasses(6, false, "n6", "par6(i)")}
  ${topResidual(6, "n6")}
  for(var q=t;q<n7;q+=1024u){var s=0.0;for(var k=0u;k<8u;k++){let r=pmgTop[T_CH7+8u*q+k];if(r!=NB_NONE){s+=tF(T_R6+r);}}tS(T_B7+q,s);}
  sbwb();
  // L7: one product with the inverse.
  for(var q=t;q<n7;q+=1024u){var s=0.0;for(var j=0u;j<n7;j++){s+=tF(T_INV+q*P_ROWS7+j)*tF(T_B7+j);}tS(T_E7+q,s);}
  sbwb();
  for(var r=t;r<n6;r+=1024u){let e=tF(T_P6+r)+tF(T_E7+pmgTop[T_PAR6+r]);tS(T_E6+r,e);tS(T_F6+r,e);}
  sbwb();
  ${topPasses(6, true, "n6", "par6(i)")}
  for(var i=t;i<n5;i+=1024u){let e=tF(T_P5+i)+tF(T_E6+pmgTop[T_PAR5+i]);tS(T_E5+i,e);tS(T_F5+i,e);}
  sbwb();
  ${topPasses(5, true, "n5", "par5(i)")}
${l4Cached ? /* wgsl */`
  // L4 post-smoothing, a page a block, a lane a cell: the lane keeps its row's couplings and frozen halo (EPRE + the
  // page across's L5 value) in registers and the page's E in workgroup memory (wv's L6 words, dead by now), so the
  // passes read no device memory; the sums are the device loop's, term for term.
  var cD=0.0;var cB=0.0;var cw=array<f32,6>();var cv=array<f32,6>();var cj=array<u32,6>(NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE,NB_NONE);
  var cs=0u;var k=0u;
  if(t<8u*n5){
    let i=t/8u;k=t%8u;cs=pmgTop[T_SLOT5+i];let e5=tF(T_E5+i);
    if(k==0u){pmgC[pL5(CH_EPOST,cs)]=e5;}
    wv[V_B6+t]=pmgC[pL4(CH_EPRE,cs,k)]+e5;
    cD=pmgC[pL4(CH_D,cs,k)];cB=pmgC[pL4(CH_B,cs,k)];
    for(var a=0u;a<3u;a++){
      for(var up=0u;up<2u;up++){
        let f=2u*a+up;let n=l4Nbr(cs,k,a,up==1u);if(n.x==NB_NONE){continue;}
        var w4=0.0;if(up==1u){w4=pmgC[pL4(CH_K+a,n.x,n.y)];}else{w4=pmgC[pL4(CH_K+a,cs,k)];}
        if(w4==0.0){continue;}
        cw[f]=w4;
        if(n.x==cs){cj[f]=8u*i+n.y;}else{let jn=pmgTop[T_IDX5+n.x];if(jn!=NB_NONE){cv[f]=pmgC[pL4(CH_EPRE,n.x,n.y)]+tF(T_E5+jn);}}
      }
    }
  }
  workgroupBarrier();
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=1u-(ps&1u);
    if(cD>0.0&&pPar(pC4(k))==colour){
      var s=cB;
      for(var f=0u;f<6u;f++){if(cw[f]==0.0){continue;}var v=cv[f];if(cj[f]!=NB_NONE){v=wv[V_B6+cj[f]];}s+=cw[f]*v;}
      wv[V_B6+t]=s/cD;
    }
    workgroupBarrier();
  }
  if(t<8u*n5){pmgC[pL4(CH_EPOST,cs,k)]=wv[V_B6+t];}
` : `
  // L4 post-smoothing, a page a block: the entry is EPRE + the page's L5 value.
  for(var w=t;w<8u*n5;w+=1024u){let i=w/8u;let k=w%8u;let cs=pmgTop[T_SLOT5+i];pmgC[pL5(CH_EPOST,cs)]=tF(T_E5+i);pmgC[pL4(CH_EPOST,cs,k)]=pmgC[pL4(CH_EPRE,cs,k)]+tF(T_E5+i);}
  sbwb();
  for(var ps=0u;ps<P_CS2;ps++){
    let colour=1u-(ps&1u);
    for(var w=t;w<8u*n5;w+=1024u){
      let i=w/8u;let k=w%8u;let cs=pmgTop[T_SLOT5+i];
      let D4=pmgC[pL4(CH_D,cs,k)];
      if(D4>0.0&&pPar(pC4(k))==colour){
        var s=pmgC[pL4(CH_B,cs,k)];
        for(var a=0u;a<3u;a++){
          for(var up=0u;up<2u;up++){
            let n=l4Nbr(cs,k,a,up==1u);if(n.x==NB_NONE){continue;}
            var w4=0.0;if(up==1u){w4=pmgC[pL4(CH_K+a,n.x,n.y)];}else{w4=pmgC[pL4(CH_K+a,cs,k)];}
            if(w4==0.0){continue;}
            var v=0.0;
            if(n.x==cs){v=pmgC[pL4(CH_EPOST,cs,n.y)];}
            else{let jn=pmgTop[T_IDX5+n.x];if(jn!=NB_NONE){v=pmgC[pL4(CH_EPRE,n.x,n.y)]+tF(T_E5+jn);}}
            s+=w4*v;
          }
        }
        pmgC[pL4(CH_EPOST,cs,k)]=s/D4;
      }
    }
    sbwb();
  }
`}
}`));

  // U3 (a page): L3 post-smoothing, 16h blocks; the frozen halo is EPRE + the parent's L4 EPOST.
  const up3 = kernel("pmgUp3", CB, ["cstage"], V3 + V4 + /* wgsl */`
var<workgroup> wE:array<f32,64>;
fn pEntry3(p:vec3i,c:vec3i)->f32{return pV3(p,c,CH_EPRE)+pV4(p,c>>vec3u(1u),CH_EPOST);}
@compute @workgroup_size(64) fn pmgUp3(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c3l=pC1(lid);let c3=4*p+c3l;let par=pPar(c3);
  let D=pmgC[pL3(CH_D,cs,lid)];let b3=pmgC[pL3(CH_B,cs,lid)];
  wE[lid]=pmgC[pL3(CH_EPRE,cs,lid)]+pmgC[pL4(CH_EPOST,cs,pI4(c3l>>vec3u(1u)))];
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgC[pL3(CH_K+a,cs,lid)];KU[a]=pV3(p,c3+e,CH_K+a);
    if((c3l[a]&1)==0&&K[a]!=0.0){H[a]=pEntry3(p,c3-e);}
    if((c3l[a]&1)==1&&KU[a]!=0.0){HU[a]=pEntry3(p,c3+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b3;for(var a=0u;a<3u;a++){let e=pUnit(a);if((c3l[a]&1)==1){s+=K[a]*wE[pI1(c3l-e)];}else{s+=K[a]*H[a];}if((c3l[a]&1)==0){s+=KU[a]*wE[pI1(c3l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgC[pL3(CH_EPOST,cs,lid)]=wE[lid];
}`);

  // U2 (a page): L2 post-smoothing, brick blocks (2^3); the frozen halo is EPRE + the parent's L3 EPOST.
  const up2 = kernel("pmgUp2", CB, ["cstage"], V2 + V3 + /* wgsl */`
var<workgroup> wE:array<f32,512>;
fn pEntry2(p:vec3i,c:vec3i)->f32{return pV2(p,c,CH_EPRE)+pV3(p,c>>vec3u(1u),CH_EPOST);}
@compute @workgroup_size(512) fn pmgUp2(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead()}
  let c2l=nbLocalCoord(lid);let c2=8*p+c2l;let par=pPar(c2);
  let D=pmgC[pL2(CH_D,cs,lid)];let b2=pmgC[pL2(CH_B,cs,lid)];
  wE[lid]=pmgC[pL2(CH_EPRE,cs,lid)]+pmgC[pL3(CH_EPOST,cs,pI1(c2l>>vec3u(1u)))];
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgC[pL2(CH_K+a,cs,lid)];KU[a]=pV2(p,c2+e,CH_K+a);
    if((c2l[a]&1)==0&&K[a]!=0.0){H[a]=pEntry2(p,c2-e);}
    if((c2l[a]&1)==1&&KU[a]!=0.0){HU[a]=pEntry2(p,c2+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b2;for(var a=0u;a<3u;a++){let e=pUnit(a);if((c2l[a]&1)==1){s+=K[a]*wE[nbLocal(c2l-e)];}else{s+=K[a]*H[a];}if((c2l[a]&1)==0){s+=KU[a]*wE[nbLocal(c2l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgC[pL2(CH_EPOST,cs,lid)]=wE[lid];
}`);

  // U1 (a brick): L1 post-smoothing, the brick a block; the frozen halo is EPRE + the parent's L2 EPOST.
  const up1 = kernel("pmgUp1", FBT, ["stage"], /* wgsl */`
var<workgroup> wE:array<f32,64>;
fn pV1(b:vec3i,c:vec3i,ch:u32)->f32{let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(ch,s,pI1(c))];}
// An L1 row's entry value: EPRE + the parent's L2 EPOST; a native row's carried value, its L1 post pass's N1P.
fn pEntry1(b:vec3i,c:vec3i)->f32{
  let n=pNatAt(b,2*c);if(n!=NB_NONE){return pmgC[pL2(CH_N1P,n,nbLocal(c>>vec3u(1u)))];}
  let s=pSlotAt(b,2*c);if(s==NB_NONE){return 0.0;}return pmgF[pL1(CH_EPRE,s,pI1(c))]+pmgC[pL2(CH_EPOST,nbFRec(s,NB_REC_PARENT),nbLocal(c>>vec3u(1u)))];
}
@compute @workgroup_size(64) fn pmgUp1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${rowHead}
  let c1l=pC1(lid);let c1=(o>>vec3u(1u))+c1l;let par=pPar(c1);
  let D=pmgF[pL1(CH_D,slot,lid)];let b1=pmgF[pL1(CH_B,slot,lid)];
  wE[lid]=pEntry1(b,c1);
  var K=vec3f(0.0);var KU=vec3f(0.0);var H=vec3f(0.0);var HU=vec3f(0.0);
  for(var a=0u;a<3u;a++){
    let e=pUnit(a);K[a]=pmgF[pL1(CH_K+a,slot,lid)];
    if(pNatAt(b,2*(c1+e))!=NB_NONE){KU[a]=pmgF[pL1(CH_KU+a,slot,lid)];}else{KU[a]=pV1(b,c1+e,CH_K+a);}
    if(c1l[a]==0&&K[a]!=0.0){H[a]=pEntry1(b,c1-e);}
    if(c1l[a]==3&&KU[a]!=0.0){HU[a]=pEntry1(b,c1+e);}
  }
  workgroupBarrier();
  ${coarsePasses(true, "D>0.0", `var s=b1;for(var a=0u;a<3u;a++){let e=pUnit(a);if(c1l[a]>0){s+=K[a]*wE[pI1(c1l-e)];}else{s+=K[a]*H[a];}if(c1l[a]<3){s+=KU[a]*wE[pI1(c1l+e)];}else{s+=KU[a]*HU[a];}}wE[lid]=s/D;`)}
  pmgF[pL1(CH_EPOST,slot,lid)]=wE[lid];
}`);

  // The native rows' own passes at L0 and L1 (pressure-composite-model.ts nativeGS: FAC's composite-grid smoother,
  // McCormick & Thomas 1986). The h passes carry the native rows; each pass then relaxes them on their own level:
  // block red-black GS in 2^3-tile blocks (a page a workgroup, a lane a tile), the h rows and the other blocks'
  // native rows frozen at the pass's entry (frozen, by C slot and tile). rhs is the row's RHS plus its frozen
  // seam sum; colours reversed after the coarse correction (the adjoint, so the V-cycle stays SPD).
  const NATIVE = /* wgsl */`
var<workgroup> wN:array<f32,512>;
// The C slot of tile T when native, seen from page p (as pNativeTile, by the staged pages), or NONE.
fn pNatQ(p:vec3i,T:vec3i)->u32{if(!pTileIn(T)){return NB_NONE;}let s=qPage(p,T,3u);if(s==NB_NONE){return NB_NONE;}return select(s,NB_NONE,nbTileH(s,nbLocal(T)));}`;
  const nativePass = (rev: boolean, entry: string, frozen: (cs: string, i: string) => string) => /* wgsl */`
  var s0=rhs;var K=vec3f(0.0);
  if(row){
    for(var a=0u;a<3u;a++){
      let e=pUnit(a);let lowIn=(c2l[a]&1)==1;
      let Tl=T-e;let nl=pNatQ(p,Tl);
      if(nl!=NB_NONE){let kl=pCK(a,cs,lid);if(kl!=0.0){if(lowIn){K[a]=kl;}else{s0+=kl*(${frozen("nl", "nbLocal(Tl)")});}}}
      let Tu=T+e;let nu=pNatQ(p,Tu);
      if(nu!=NB_NONE){let iu=nbLocal(Tu);let ku=pCK(a,nu,iu);if(ku!=0.0){if(!lowIn){K[a]=ku;}else{s0+=ku*(${frozen("nu", "iu")});}}}
    }
  }
  wN[lid]=${entry};
  workgroupBarrier();
  for(var ps=0u;ps<P_NS2;ps++){
    let colour=${rev ? "1u-(ps&1u)" : "ps&1u"};
    if(row&&par==colour){var s=s0;for(var a=0u;a<3u;a++){if(K[a]!=0.0){let e=pUnit(a);s+=K[a]*wN[nbLocal(select(c2l+e,c2l-e,(c2l[a]&1)==1))];}}wN[lid]=s/D;}
    workgroupBarrier();
  }
  let v=select(0.0,wN[lid],row);`;
  const nativeHead = (skip = "") => /* wgsl */`${pageHead("native", skip)}
  let c2l=nbLocalCoord(lid);let T=8*p+c2l;let par=pPar(T);
  let D=pP(NB_PRES_C_DIAG,cs,lid);let row=pTileIn(T)&&!nbTileH(cs,lid)&&D>0.0;let ns=pSeamN(cs,lid);`;
  const cN = (ch: string) => (cs: string, i: string) => `pmgC[pL2(${ch},${cs},${i})]`;
  const w0 = (s: string, l: string) => `pmgF[${PRE}+NB_STRIDE*${s}+${l}]`;
  const e1 = (ch: string, s: string, m: string) => `pmgF[pL1(${ch},${s},pI1(${m}>>vec3u(1u)))]`;
  // L0 pre (after the h rows' last pre pass): from 0, the h rows at their pre-smoothed plane (PRE). Writes N0.
  const nativePre0 = kernel("pmgNativePre0", FB, ["pres", "cstage", "seam"], NATIVE + seamSum("pSeamW0", (s, l) => w0(s, l)) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePre0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead()}
  let sm=pSeamW0(p,cs,lid,ns);
  var rhs=0.0;if(row){rhs=pP(NB_PRES_C_R,cs,lid)+sm;}
  ${nativePass(false, "0.0", () => "0.0")}
  pmgC[pL2(CH_N0,cs,lid)]=v;
}`);
  // L1 pre (after pmgDown0): the row's L1 RHS R1 = r - A0 e0 (e0 its N0 and the h rows' W0), then its pass from 0, the
  // h aggregates at their L1 EPRE (the seams over sqrt(OC)). Writes R1 and N1.
  const nativePre1 = kernel("pmgNativePre1", FB, ["pres", "cstage", "seam"], NATIVE +
    nativeApplyFn("pApplyN0", cN("CH_N0")) + seamSum("pSeamPre1", (s, l) => w0(s, l), (s, l, m) => `${e1("CH_EPRE", s, m)}/P_SOC`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePre1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead()}
  let sm=pSeamPre1(p,cs,lid,ns);
  var r1=0.0;var rhs=0.0;if(row){r1=pP(NB_PRES_C_R,cs,lid)-pApplyN0(p,cs,lid,T,sm.x);rhs=r1+sm.y;}
  ${nativePass(false, "0.0", () => "0.0")}
  pmgC[pL2(CH_R1,cs,lid)]=r1;pmgC[pL2(CH_N1,cs,lid)]=v;
}`);
  // L1 post (after pmgUp2): entry N1 + the row's L2 EPOST, the h aggregates at EPRE + their parent's L2 EPOST. Writes N1P.
  const nativePost1 = kernel("pmgNativePost1", FB, ["pres", "cstage", "seam"], NATIVE +
    seamSum("pSeamP1", (s, l, m) => `(${e1("CH_EPRE", s, m)}+pmgC[pL2(CH_EPOST,nbFRec(${s},NB_REC_PARENT),nbLocal(${m}>>vec3u(2u)))])/P_SOC`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePost1(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead()}
  let sm=pSeamP1(p,cs,lid,ns);
  var rhs=0.0;if(row){rhs=pmgC[pL2(CH_R1,cs,lid)]+sm;}
  ${nativePass(true, "pmgC[pL2(CH_N1,cs,lid)]+pmgC[pL2(CH_EPOST,cs,lid)]", (c, i) => `pmgC[pL2(CH_N1,${c},${i})]+pmgC[pL2(CH_EPOST,${c},${i})]`)}
  pmgC[pL2(CH_N1P,cs,lid)]=v;
}`);
  // L0 post (after pmgUp1): entry N0 + N1P, the h rows at PRE + their L1 EPOST. Writes N0P, the row's z, and r.z into
  // the dot's slots.
  const nativePost0 = kernel("pmgNativePost0", [...FB, redRW], ["pres", "cstage", "reduce", "red", "seam"], NATIVE +
    seamSum("pSeamP0", (s, l, m) => `${w0(s, l)}+${e1("CH_EPOST", s, m)}`) + /* wgsl */`
@compute @workgroup_size(512) fn pmgNativePost0(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${nativeHead("if(lid==0u){pmgRed[R_ZC+g]=0.0;}")}
  let sm=pSeamP0(p,cs,lid,ns);
  var rhs=0.0;if(row){rhs=pP(NB_PRES_C_R,cs,lid)+sm;}
  ${nativePass(true, "pmgC[pL2(CH_N0,cs,lid)]+pmgC[pL2(CH_N1P,cs,lid)]", (c, i) => `pmgC[pL2(CH_N0,${c},${i})]+pmgC[pL2(CH_N1P,${c},${i})]`)}
  pmgC[pL2(CH_N0P,cs,lid)]=v;
  var dotv=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)){pres[NB_PRES_C_Z+NB_STRIDE*cs+lid]=v;if(row){dotv=pP(NB_PRES_C_R,cs,lid)*v;}}
  let sum=pSum512(lid,dotv);if(lid==0u){pmgRed[R_ZC+g]=sum;}
}`);

  // C | F in one dispatch (MG_GATE.FC: C_ALL's pages, then the row bricks): workgroup g < C_ALL's count runs page g's
  // native rows (a page without native rows returns), the rest row brick g - count's h rows. The halves are
  // independent (each writes only its own rows and its own dot partial), so one launch replaces two; the
  // pages, few and each long (the seam sums), go first so the bricks' waves run beside them rather than after.
  // head: the CG scalar's fold (pFixed), once in the entry in uniform control flow, after a C workgroup without native
  // rows has returned (skipC, its partial's zero); its F and C halves read the result from private variables.
  const fusedFC = (label: string, decl: string, head: string, fBody: string, cBody: string, skipC: string, tile = false) =>
    kernel(label, [...FBT, redRW, ...MB_B], ["pres", "stage", "cstage", "reduce", "red", "seam", ...(tile ? ["tile" as const] : [])], decl + /* wgsl */`
var<workgroup> wCountC:u32;
fn ${label}F(g:u32,lid:u32){
  let slot=pmgTop[T_RB+g];pStageK(slot,lid);workgroupBarrier();
  let o=nbFOrigin(slot);let b=o>>vec3u(3u);
${fBody}
}
fn ${label}C(g:u32,lid:u32){
  let cs=nbListSlot(NB_LIST_C_ALL,g);
  qStageK(cs,lid);workgroupBarrier();
  let p=nbKeyCoord(nbCRec(cs,NB_REC_KEY));
${cBody}
}
@compute @workgroup_size(512) fn ${label}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wCount=min(pmgTop[T_RBN],NB_FCAP);wCountC=nbListCount(NB_LIST_C_ALL);}
  let g=nbGroup(wid);let nF=workgroupUniformLoad(&wCount);let nC=workgroupUniformLoad(&wCountC);
  if(g>=nC+nF){return;}
  if(g<nC){
    if(lid==0u){wKids=select(0u,1u,pmgC[pL5(CH_KU,nbListSlot(NB_LIST_C_ALL,g))]>0.0);}
    if(workgroupUniformLoad(&wKids)==0u){${skipC}return;}
  }
${head}
  if(g<nC){${label}C(g,lid);}else{${label}F(g-nC,lid);}
}`);

  // Direction: x += alpha_(k-1) p (deferred from the last update), p = z + beta p, q = A z + beta q, p.q; the native
  // rows' PCG vectors a page a workgroup, a lane a tile.
  const zN = (cs: string, i: string) => `pres[NB_PRES_C_Z+NB_STRIDE*${cs}+${i}]`, zF = (s: string, l: string) => `pres[NB_PRES_F_Z+NB_STRIDE*${s}+${l}]`;
  // r.z over pmgPostZ's row bricks then pmgNativePost0's pages; beta against the last slot's (PH_RZ, the check's).
  const betaHead = /* wgsl */`
  let rz=pFixed(lid,R_ZF,nF,R_ZC,nC);
  if(lid==0u){
    var beta=0.0;
    if(cfg.z>0u){let old=bitcast<f32>(atomicLoad(&nbSet[PH_RZ]));beta=select(0.0,rz/old,old!=0.0);}
    wBeta=beta;wAlphaOld=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));
    if(pFixedWriter(g,nC,nF)){pmgRed[R_RZ]=rz;}
  }
  pBeta=workgroupUniformLoad(&wBeta);pAlphaOld=workgroupUniformLoad(&wAlphaOld);`;
  const betaLets = /* wgsl */`
  let beta=pBeta;let alpha=pAlphaOld;`;
  const direction = fusedFC("pmgDirection",
    fineApply("pApplyZ", zF, zN) + nativeApply("pApplyNZ", "pSeamZ", zN, zF) + MB + `\nvar<workgroup> wBeta:f32;var<workgroup> wAlphaOld:f32;var<private> pBeta:f32;var<private> pAlphaOld:f32;`, betaHead, betaLets + /* wgsl */`
  let l=lid;let c=o+nbLocalCoord(l);let at=NB_STRIDE*slot+l;
  var pq=0.0;
  if(pDiag(slot,l)>0.0){
    let dOld=pres[NB_PRES_F_D+at];
    pres[NB_PRES_F_X+at]+=alpha*dOld;
    let dn=pres[NB_PRES_F_Z+at]+beta*dOld;
    let q=pApplyZ(b,c,slot,l)+beta*pres[NB_PRES_F_Q+at]${bodies ? "+mbApply(mbBrick(slot),l,c,MB_WZ,0.0,MB_WZ)" : ""};
    pres[NB_PRES_F_D+at]=dn;pres[NB_PRES_F_Q+at]=q;pq=dn*q;
  }else{pres[NB_PRES_F_D+at]=0.0;pres[NB_PRES_F_Q+at]=0.0;}
  let sum=pSum512(lid,pq);if(lid==0u){pmgRed[R_Q+wCountC+g]=sum;}`, betaLets + /* wgsl */`
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;
  let sm=pSeamZ(p,cs,lid,pSeamN(cs,lid));
  var pq=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    let dOld=pres[NB_PRES_C_D+at];
    pres[NB_PRES_C_X+at]+=alpha*dOld;
    let dn=pres[NB_PRES_C_Z+at]+beta*dOld;
    let q=pApplyNZ(p,cs,lid,T,sm)+beta*pres[NB_PRES_C_Q+at];
    pres[NB_PRES_C_D+at]=dn;pres[NB_PRES_C_Q+at]=q;pq=dn*q;
  }
  let sum=pSum512(lid,pq);if(lid==0u){pmgRed[R_Q+g]=sum;}`, "if(lid==0u){pmgRed[R_Q+g]=0.0;}");

  // Update: r = b - A (x + alpha p) and max|r| (a native row's over 64); the h rows' first V-cycle pass from zero into W0.
  // p.q over pmgDirection's workgroups; r.z the direction's (R_RZ).
  const alphaHead = /* wgsl */`
  let pq=pFixed(lid,R_Q,nC+nF,0u,0u);
  if(lid==0u){let rz=pmgRed[R_RZ];wAlpha=select(0.0,rz/pq,pq!=0.0);if(pFixedWriter(g,nC,nF)){pmgRed[R_PQ]=pq;}}
  pAlpha=workgroupUniformLoad(&wAlpha);`;
  const xF = (s: string, l: string) => `pres[NB_PRES_F_X+NB_STRIDE*${s}+${l}]+pAlpha*pres[NB_PRES_F_D+NB_STRIDE*${s}+${l}]`;
  const xN = (cs: string, i: string) => `pres[NB_PRES_C_X+NB_STRIDE*${cs}+${i}]+pAlpha*pres[NB_PRES_C_D+NB_STRIDE*${cs}+${i}]`;
  const update = fusedFC("pmgUpdate",
    `var<private> pAlpha:f32;var<workgroup> wAlpha:f32;\n` + fineApply("pApplyR", xF, xN, true) + MB, alphaHead, /* wgsl */`
  let l=lid;let lc=nbLocalCoord(l);let c=o+lc;let ti=pTi(lc);
  let d=pDiag(slot,l);var r=0.0;
  if(d>0.0){r=pP(NB_PRES_F_RHS,slot,l)-pApplyR(b,c,slot,l)${bodies ? "-mbApply(mbBrick(slot),l,c,MB_WX,pAlpha,MB_WD)" : ""};}
  pres[NB_PRES_F_R+NB_STRIDE*slot+l]=r;
  let rmax=pMax512(lid,abs(r));
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}
  pTileZero(lid);
  let row=pRow(b,c);
  pSmooth(row,c,ti,r,false);
  if(row.d>0.0){pmgF[P_W0+NB_STRIDE*slot+l]=tp[ti];}`, /* wgsl */`
  let T=8*p+nbLocalCoord(lid);
  // A native row solves for its correction from the step's start (the old engine's pNative: PN_X 0, the start in
  // PN_B): its residual is the start's, formed once in face-difference form (pmgCoarsenC), less alpha A d (Q). Recomputing
  // b - A x from the total pressure reads the f32 quantisation of x (ulp(x) D, 1e-4 1/s band-eq on the 1 cm tank's
  // 2.5 m column), the floor nBound asks the natives to reach.
  var r=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    let at=NB_STRIDE*cs+lid;r=pres[NB_PRES_C_R+at]-pAlpha*pres[NB_PRES_C_Q+at];pres[NB_PRES_C_R+at]=r;
  }
  let rmax=pMax512(lid,abs(r)/64.0*bitcast<f32>(atomicLoad(&nbSet[PH_NSCALE])));
  if(lid==0u){atomicMax(&nbSet[PH_RMAX],bitcast<u32>(rmax));}`, "", true);
  const finishC = kernel("pmgFinishC", CBP, ["pres", "cstage"], /* wgsl */`
var<workgroup> wAlpha:f32;
@compute @workgroup_size(512) fn pmgFinishC(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));}
  let alpha=workgroupUniformLoad(&wAlpha);
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){pres[NB_PRES_C_X+at]+=alpha*pres[NB_PRES_C_D+at];}
}`);

  // The separating walls' outer loop (Chentanez & Mueller 2012, "A multigrid fluid pressure solver handling
  // separating solid boundary conditions"; the decision rule is the primal-dual active set of Hintermueller, Ito &
  // Kunisch 2002, a semismooth Newton step on the wall's ghost max(x + lift, 0)). After a slot's update, while the solve
  // has re-decisions left (wallDecisions, MG_WALL_DECISIONS), when its max|r| is within wallLoose x tolerance or at the
  // last slot (a capped solve ends on its final iterate's state, as the old projection's max(x + lift, 0) at the final
  // x), every lattice wall face of a liquid row is re-decided at the iterate x + alpha p:
  // attached when x + lift >= 0, else separated. A face that changes state moves its coupling c between the RHS
  // (c lift) and the diagonal (c), and its row's residual becomes r + dRHS - dDIAG (x + alpha p): the coupling is
  // diagonal only, so no other row's residual moves. The state is FLAGS bit 8+a at the face's storage (the cell or
  // tile itself on plane 0, the skirt cell or tile N/4 on plane N), which the rows' build set. pmgCheck then
  // restarts PCG (beta 0) when any face changed and the new max|r| (PH_RMAXW) is above tolerance.
  // Where the old engine differs: it takes max(x + lift, -base) at every residual and smoothing evaluation of the
  // V-cycles and freezes the state at the Krylov start; here a solve's system is fixed until it converges (a
  // Krylov method on a fixed SPD matrix), and the state changes only between converged solves. The coarse levels
  // keep the setup's aggregates of the build's diagonal: the V-cycle stays symmetric positive definite (a symmetric
  // smoother on the current L0 rows and any SPD coarse correction), so it is still a valid preconditioner.
  // A changed brick's first V-cycle pass (W0, written by pmgUpdate from r) is recomputed from the new r and rows.
  // pmgWallsG (guessAt) takes the same decision at the Jacobi guess, before the setup forms r and the aggregates, so it
  // restarts nothing.
  // A row's wall side along a (-1 plane 0, +1 plane N, 0 none) on n rows an axis; the tolerance (cfg.w, as pTolR).
  const wallSign = /* wgsl */`fn pWallFace(c:vec3i,a:u32,n:i32)->i32{if(c[a]==0){return -1;}if(c[a]==n-1){return 1;}return 0;}
fn pWallTol()->f32{let h=nbRingF(cfg.x,NB_RING_H);return bitcast<f32>(cfg.w)*h*h;}`;
  // audit: the finish's diagnostic count at the final iterate x + alpha p (PH_ALPHA), over F_ALL's row bricks; no state changes.
  // guess: the decision at the Jacobi guess (pmgGuess's Z, before setupF, which forms r from the decided rows), over the
  // setup's rows; it moves the rows' DIAG and RHS only, and counts its flips into PH_WALLS.
  const wallsFKernel = (audit: boolean, guessAt = false) => kernel(audit ? "pmgAuditF" : guessAt ? "pmgWallsG" : "pmgWallsF", [CFG, setRW, dirRO, presRW, fRW, topRW, redRW, phiRW, velRO], /* 8 storage: no pmgC */ ["pres", "stage", "reduce", "tile", "red"], wallSign + /* wgsl */`
var<workgroup> wGo:u32;var<workgroup> wAlpha:f32;var<workgroup> wFlips:atomic<u32>;var<workgroup> wAud:array<atomic<u32>,2>;
@compute @workgroup_size(512) fn ${audit ? "pmgAuditF" : guessAt ? "pmgWallsG" : "pmgWallsF"}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  ${guessAt ? fHead() : `if(lid==0u){wCount=${audit ? "nbListCount(NB_LIST_F_ALL)" : "min(pmgTop[T_RBN],NB_FCAP)"};}
  let g=nbGroup(wid);if(g>=workgroupUniformLoad(&wCount)){return;}
  let slot=${audit ? "nbListSlot(NB_LIST_F_ALL,g)" : "pmgTop[T_RB+g]"};`}let o=nbFOrigin(slot);let b=o>>vec3u(3u);
  if(lid==0u){
    let edge=any(o==vec3i(0))||any(o+vec3i(8)>=NB_WALL_BOX);
    wGo=select(0u,1u,${cuts ? "(edge||(nbFRec(slot,NB_REC_CLASS)&NB_CLASS_CUT_STATIC)!=0u)" : "edge"}${audit ? "&&pmgTop[T_RBF+slot]!=0u" : ""});
    ${audit ? "wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));atomicStore(&wAud[0],0u);atomicStore(&wAud[1],0u);" : guessAt ? "wAlpha=0.0;" : "let rz=pmgRed[R_RZ];let pq=pmgRed[R_PQ];wAlpha=select(0.0,rz/pq,pq!=0.0);"}
    atomicStore(&wFlips,0u);
  }
  if(workgroupUniformLoad(&wGo)==0u){return;}
  ${guessAt ? "" : "pStageK(slot,lid);workgroupBarrier();"}
  let alpha=workgroupUniformLoad(&wAlpha);
  let l=lid;let lc=nbLocalCoord(l);let c=o+lc;let at=NB_STRIDE*slot+l;let h=nbRingF(cfg.x,NB_RING_H);
  var r=${guessAt ? "0.0" : "pres[NB_PRES_F_R+at]"};
  if(pDiag(slot,l)>0.0){
    let x=${guessAt ? "pres[NB_PRES_F_Z+at]" : "pres[NB_PRES_F_X+at]+alpha*pres[NB_PRES_F_D+at]"};
    var dd=0.0;var db=0.0;var flips=0u;
    ${cuts ? `// The cuts: every face of the row, the box's and a static solid's (option (b): the static cut bricks), by its
    // kind and coupling from rowsF's face cache (a wall face's word is its V); a closed one (A <= 1e-6) has no state.
    for(var fk=0u;fk<6u;fk++){
      let a=fk/2u;let side=select(-1,1,(fk&1u)==1u);
      if(` : `for(var a=0u;a<3u;a++){
      let side=pWallFace(c,a,NB_WALL_BOX[a]);if(side==0){continue;}`}${cuts ? `((u32(pmgF[P_FC+NB_STRIDE*(7u*slot+6u)+l])>>(4u*fk))&15u)!=${NP_K.WALL + 1}u){continue;}
      let wc=pmgF[P_FC+NB_STRIDE*(7u*slot+fk)+l];` : ""}
      var sc=c;if(side>0){sc[a]+=1;}
      let s=nbFSlotOfCell(sc);if(s>=NB_SEAM){continue;}
      let w=NB_PHI_F_FLAGS+NB_STRIDE*s+nbLocal(sc);let bit=${P_WALL0}u<<a;
      // npLift(f, 1, US, h, 1): the face's forced velocity as a pressure in the slot units.
      let lift=f32(side)*vel[NB_VEL_F_US0+NB_STRIDE*(s+a*NB_FCAP)+nbLocal(sc)]*h;
      let attached=x+lift>=0.0;
      ${audit ? "atomicAdd(&wAud[1],1u);if(attached==((phi[w]&bit)!=0u)){atomicAdd(&wAud[0],1u);}" : `if(attached==((phi[w]&bit)!=0u)){
        if(attached){dd-=${cuts ? "wc" : WALL_F};db+=${cuts ? "wc" : WALL_F}*lift;}else{dd+=${cuts ? "wc" : WALL_F};db-=${cuts ? "wc" : WALL_F}*lift;}
        phi[w]=(phi[w]&~bit)|select(bit,0u,attached);flips++;
      }`}
    }
    if(flips>0u){
      pres[NB_PRES_F_DIAG+at]+=dd;pres[NB_PRES_F_RHS+at]+=db;${guessAt ? "" : "r=r+db-dd*x;pres[NB_PRES_F_R+at]=r;"}
      atomicAdd(&wFlips,flips);
    }
  }
  workgroupBarrier();
  ${audit ? "if(lid==0u){atomicAdd(&nbSet[PH_AUDIT],atomicLoad(&wAud[0]));atomicAdd(&nbSet[PH_AUDIT+1u],atomicLoad(&wAud[1]));}" : ""}${guessAt ? `
  if(lid==0u){let n=atomicLoad(&wFlips);if(n>0u){atomicAdd(&nbSet[PH_WALLS],n);}}
}` : ""}
${guessAt ? "" : `
  if(lid==0u){let n=atomicLoad(&wFlips);if(n>0u){atomicAdd(&nbSet[PH_FLIPS],n);}wGo=n;}
  if(workgroupUniformLoad(&wGo)==0u){return;}
  storageBarrier();
  let rmax=pMax512(lid,abs(r));
  if(lid==0u){atomicMax(&nbSet[PH_RMAXW],bitcast<u32>(rmax));}
  // pmgUpdate's first brick pass, again from the new r and rows.
  let ti=pTi(lc);
  pTileZero(lid);
  let row=pRow(b,c);
  pSmooth(row,c,ti,r,false);
  if(row.d>0.0){pmgF[P_W0+at]=tp[ti];}
}`}`);
  const wallsF = wallsFKernel(false);
  // The native rows' walls (rowsC's root wall slot, c = 4 0.5, lift = sign 4h US; N_HELD, rootPhi < -4, never
  // separates), a page a workgroup, a lane a tile.
  const wallsCKernel = (audit: boolean) => kernel(audit ? "pmgAuditC" : "pmgWallsC", [...CBP, redRW, phiRW, velRO], ["pres", "cstage", "reduce", "red"], wallSign + /* wgsl */`
var<workgroup> wGo:u32;var<workgroup> wAlpha:f32;var<workgroup> wFlips:atomic<u32>;var<workgroup> wAud:array<atomic<u32>,2>;
@compute @workgroup_size(512) fn ${audit ? "pmgAuditC" : "pmgWallsC"}(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("native")}
  if(lid==0u){
    let edge=any(p==vec3i(0))||any(32*(p+vec3i(1))>=NB_WALL_BOX);
    wGo=select(0u,1u,edge);
    ${audit ? "wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));atomicStore(&wAud[0],0u);atomicStore(&wAud[1],0u);" : "let rz=pmgRed[R_RZ];let pq=pmgRed[R_PQ];wAlpha=select(0.0,rz/pq,pq!=0.0);"}
    atomicStore(&wFlips,0u);
  }
  if(workgroupUniformLoad(&wGo)==0u){return;}
  let alpha=workgroupUniformLoad(&wAlpha);
  let T=8*p+nbLocalCoord(lid);let at=NB_STRIDE*cs+lid;let h=nbRingF(cfg.x,NB_RING_H);
  var r=0.0;
  if(pTileIn(T)&&!nbTileH(cs,lid)&&pP(NB_PRES_C_DIAG,cs,lid)>0.0){
    r=pres[NB_PRES_C_R+at];
    if(!(pres[NB_PRES_C_PPHI+at]< -4.0)){
      let x=pres[NB_PRES_C_X+at]+alpha*pres[NB_PRES_C_D+at];
      var dd=0.0;var db=0.0;var flips=0u;
      for(var a=0u;a<3u;a++){
        let side=pWallFace(T,a,(NB_WALL_BOX[a]+3)/4);if(side==0){continue;}
        var Ts=T;if(side>0){Ts[a]+=1;}
        let ws=nbCSlotOfPage(Ts>>vec3u(3u));if(ws==NB_NONE){continue;}
        let w=NB_PHI_C_FLAGS+NB_STRIDE*ws+nbLocal(Ts);let bit=${P_WALL0}u<<a;
        let lift=f32(side)*4.0*h*vel[NB_VEL_C_US0+NB_STRIDE*(ws+a*NB_CCAP)+nbLocal(Ts)];
        let attached=x+lift>=0.0;
        ${audit ? "atomicAdd(&wAud[1],1u);if(attached==((phi[w]&bit)!=0u)){atomicAdd(&wAud[0],1u);}" : `if(attached==((phi[w]&bit)!=0u)){
          if(attached){dd-=${WALL_C};db+=${WALL_C}*lift;}else{dd+=${WALL_C};db-=${WALL_C}*lift;}
          phi[w]=(phi[w]&~bit)|select(bit,0u,attached);flips++;
        }`}
      }
      if(flips>0u){
        pres[NB_PRES_C_DIAG+at]+=dd;pres[NB_PRES_C_RHS+at]+=db;r=r+db-dd*x;pres[NB_PRES_C_R+at]=r;
        atomicAdd(&wFlips,flips);
      }
    }
  }
  workgroupBarrier();
  ${audit ? "if(lid==0u){atomicAdd(&nbSet[PH_AUDIT+2u],atomicLoad(&wAud[0]));atomicAdd(&nbSet[PH_AUDIT+3u],atomicLoad(&wAud[1]));}" : ""}
  if(lid==0u){let n=atomicLoad(&wFlips);if(n>0u){atomicAdd(&nbSet[PH_FLIPS],n);}wGo=n;}
  if(workgroupUniformLoad(&wGo)==0u){return;}
  let rmax=pMax512(lid,abs(r)/64.0*bitcast<f32>(atomicLoad(&nbSet[PH_NSCALE])));
  if(lid==0u){atomicMax(&nbSet[PH_RMAXW],bitcast<u32>(rmax));}
}`);

  const wallsC = wallsCKernel(false);
  // The walls' gate (one workgroup, after pmgUpdate): WF and WC carry the row bricks' and C_ALL's G1 when this slot decides
  // (open, a re-decision left, and max|r| within wallLoose x tolerance or the last slot), else 0, so a slot that doesn't
  // decide launches nothing.
  const wallGate = kernel("pmgWallGate", [CFG, setRW, argsRW, topRW], [], wallSign + /* wgsl */`
@compute @workgroup_size(1) fn pmgWallGate(){
  let decide=atomicLoad(&nbSet[PH_OPEN])!=0u&&atomicLoad(&nbSet[PH_DECIDES])<${u(wallDecisions)}
    &&(${Number.isFinite(wallLoose) ? `bitcast<f32>(atomicLoad(&nbSet[PH_RMAX]))<=${wallLoose.toFixed(6)}*pWallTol()` : "true"}||cfg.z+1u>=P_SLOTS);
  if(decide){atomicAdd(&nbSet[PH_DECIDES],1u);}
  let c=NB_ARGS_PER_LIST*NB_LIST_C_ALL+NB_ARGS_G1;let n=min(pmgTop[T_RBN],NB_FCAP);
  args[P_GATE_WF]=select(0u,min(n,NB_FOLD),decide&&n>0u);args[P_GATE_WF+1u]=max(1u,(n+NB_FOLD-1u)/NB_FOLD);args[P_GATE_WF+2u]=1u;
  args[P_GATE_WC]=select(0u,args[c],decide);args[P_GATE_WC+1u]=args[c+1u];args[P_GATE_WC+2u]=1u;
}`);

  // Check: one workgroup, every slot; a closed slot's writes nothing.
  const check = kernel("pmgCheck", [CFG, setRW, redRW, argsRW, topRW], ["red", "gate"], /* wgsl */`
var<workgroup> wOpen:u32;
@compute @workgroup_size(64) fn pmgCheck(@builtin(local_invocation_index) lid:u32){
  if(lid==0u){wOpen=atomicLoad(&nbSet[PH_OPEN]);}
  if(workgroupUniformLoad(&wOpen)==0u){return;}
  if(lid==0u){
    let rz=pmgRed[R_RZ];let pq=pmgRed[R_PQ];let alpha=select(0.0,rz/pq,pq!=0.0);
    atomicStore(&nbSet[PH_RZ],bitcast<u32>(rz));atomicStore(&nbSet[PH_ALPHA],bitcast<u32>(alpha));
    // max|r| over the update's rows and, after a wall change, the changed bricks' and pages' new residuals
    // (PH_RMAXW, kept apart so every walls workgroup decides on the update's max). Non-negative floats order as u32.
    let rmax=max(atomicExchange(&nbSet[PH_RMAX],0u),atomicExchange(&nbSet[PH_RMAXW],0u));atomicStore(&nbSet[PH_RLAST],rmax);pHist(cfg.z+1u,rmax);
    atomicAdd(&nbSet[PH_ITERATIONS],1u);atomicAdd(&nbSet[PH_TOTAL],1u);
    // The walls' outer loop (pmgWallsF/C) changed the system at a converged iterate: PCG restarts from it
    // (beta 0 at the next slot); alpha stays, since the deferred x += alpha p is the iterate the walls decided at.
    let flips=atomicExchange(&nbSet[PH_FLIPS],0u);
    if(flips>0u){atomicStore(&nbSet[PH_RZ],0u);atomicAdd(&nbSet[PH_WALLS],flips);}
    if(bitcast<f32>(rmax)<=pTolR()){pGate(false);}
    else if(cfg.z+1u>=P_SLOTS){atomicOr(&nbSet[NB_H_STATE],NB_STATE_PRESSURE_CAP);atomicAdd(&nbSet[PH_CAPPED],1u);pGate(false);}
    else{pGate(true);}
  }
}`);

  // Finish: the last update's x += alpha p.
  const finish = kernel("pmgFinish", live ? [CFG, setRW, dirRO, presRW, fRW] : [CFG, setRW, dirRO, presRW], ["pres", "stage"], /* wgsl */`
var<workgroup> wAlpha:f32;
@compute @workgroup_size(512) fn pmgFinish(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
  // No neighbour is read: no stage.
  ${fHead("wAlpha=bitcast<f32>(atomicLoad(&nbSet[PH_ALPHA]));")}
  let alpha=workgroupUniformLoad(&wAlpha);
  let at=NB_STRIDE*slot+lid;
  if(pDiag(slot,lid)>0.0){pres[NB_PRES_F_X+at]+=alpha*pres[NB_PRES_F_D+at];}
}`);

  const list = (k: number) => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 });
  const gF = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.F }, z });
  const gC = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.C }, z });
  const gFC = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.FC }, z });
  const gS = (kernel: BrickKernel, z: number): BrickPlacement => ({ kernel, dispatch: { indirect: MG_GATE.S }, z });
  // ---- the region pressure change (lane E's region heat, brick-abi.md "PDELTA") ----------------------------------
  // The old engine's per-region pressureDelta (uniform-mixed-region-dynamics.ts publishPressure), on the composite's
  // root grid: a tile's value v = X / dt = p / rho (pres C X after pmgProjectC: a native row's X, an h tile's mean of its
  // liquid h rows' X, 0 off the liquid) and its change since the last publish dv; per axis the larger over the two
  // stored neighbours n of |dv - dv(n)| / 4 (the 4h root spacing), summed over the axes; a region (an octant of the
  // page, r = (x >> 2) | ((y >> 2) << 1) | ((z >> 2) << 2) of the tile's C local) takes the max over its tiles, as f32
  // bits (non-negative floats order as u32). A tile without history (no publish the step before at this C slot for this
  // page, its own or a neighbour's) makes its region +Inf: unknown fails hot. A region with no tile in the lattice is
  // +Inf (not stored). The history is double-buffered by step parity (pmgC PH: v a tile; pmgTop PHW: the page key and step + 1 a slot), so
  // the pass reads the previous step's half while it writes its own; one workgroup a page writes its 8 words.
  const pressureDelta = kernel("pmgPressureDelta", [CFG, setRW, dirRO, presRW, cRW, topRW], ["pres", "cstage"], /* wgsl */`
const P_PDELTA:u32=${u(M.PDELTA)};const P_PH:u32=${u(M.PH)};
var<workgroup> wReg:array<atomic<u32>,8>;var<workgroup> wIn:array<atomic<u32>,8>;
fn pHistV(par:u32,cs:u32,i:u32)->u32{return P_PH+(par*NB_CCAP+cs)*512u+i;}
fn pHistW(par:u32,cs:u32,k:u32)->u32{return T_PHW+(par*NB_CCAP+cs)*2u+k;}
fn pKnown(par:u32,cs:u32,step:u32)->bool{
  return pmgTop[pHistW(par,cs,0u)]==nbCRec(cs,NB_REC_KEY)&&pmgTop[pHistW(par,cs,1u)]==step;
}
@compute @workgroup_size(512) fn pmgPressureDelta(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){${pageHead("all")}
  if(lid<8u){atomicStore(&wReg[lid],0u);atomicStore(&wIn[lid],0u);}
  workgroupBarrier();
  let step=nbRing(cfg.x,NB_RING_STEP);let dt=nbRingF(cfg.x,NB_RING_DT);
  let par=step&1u;let prev=1u-par;
  let lc=nbLocalCoord(lid);let T=8*p+lc;let r=u32((lc.x>>2u)|((lc.y>>2u)<<1u)|((lc.z>>2u)<<2u));
  let v=pres[NB_PRES_C_X+NB_STRIDE*cs+lid]/dt;
  if(pTileIn(T)){
    atomicStore(&wIn[r],1u);
    // A slot's history is its page's from the step before: the publish writes step + 1 (0 never published).
    var known=pKnown(prev,cs,step);
    let dv=v-pmgC[pHistV(prev,cs,lid)];
    var delta=0.0;
    for(var a=0u;a<3u;a++){
      var edge=0.0;
      for(var sd=0u;sd<2u;sd++){
        var n=T;if(sd==0u){n[a]-=1;}else{n[a]+=1;}
        if(!pTileIn(n)){continue;}
        let ns=qPage(p,n,3u);if(ns==NB_NONE){continue;}
        let ni=nbLocal(n);
        known=known&&pKnown(prev,ns,step);
        let dn=pres[NB_PRES_C_X+NB_STRIDE*ns+ni]/dt-pmgC[pHistV(prev,ns,ni)];
        edge=max(edge,abs(dv-dn)*0.25);
      }
      delta+=edge;
    }
    let bits=select(0x7f800000u,bitcast<u32>(delta),known&&delta<=3.4e38);
    atomicMax(&wReg[r],bits);
  }
  workgroupBarrier();
  // Every reader of this step's half reads the previous step's, so the writes below race with none of them.
  pmgC[pHistV(par,cs,lid)]=v;
  if(lid<8u){pmgC[P_PDELTA+8u*cs+lid]=bitcast<f32>(select(0x7f800000u,atomicLoad(&wReg[lid]),atomicLoad(&wIn[lid])!=0u));}
  if(lid==0u){pmgTop[pHistW(par,cs,0u)]=nbCRec(cs,NB_REC_KEY);pmgTop[pHistW(par,cs,1u)]=step+1u;}
}`);

  const hb = { indirect: MG_GATE.HB }, fRows = live ? hb : list(LIST.F_ALL);
  const bodyZ0 = bodyZKernel(false), bodyZ = bodyZKernel(true), bodyFold0 = bodyFoldKernel(true), bodyFold = bodyFoldKernel(false);
  const lv: MGPCGLive | undefined = KL && {
    args: [{ kernel: listArgs, dispatch: { direct: [1, 1, 1] } }, { kernel: rowsZero, dispatch: { indirect: MG_GATE.ZL } }],
    tileRoots: [{ kernel: KL.tileRoots, dispatch: hb }],
    rows: [{ kernel: KL.rowsF, dispatch: hb }],
    project: [{ kernel: KL.projectF, dispatch: { indirect: MG_GATE.HP } }, { kernel: KL.projectDry!, dispatch: { indirect: MG_GATE.HP } }],
    rowsList,
  };
  const core = {
    setup: [
      { kernel: setupClear, dispatch: { direct: [1, 1, 1] } },
      // The bodies' rows and the RHS's V* before the guess reads the RHS; J^T of the guess before setupF's r.
      ...(bodyJ ? [{ kernel: bodyJ, dispatch: list(LIST.F_BODY) }] : []),
      ...(guessOn ? [{ kernel: guess, dispatch: fRows }] : []),
      ...(guessWalls ? [{ kernel: wallsFKernel(false, true), dispatch: fRows }] : []),
      ...(bodyZ0 && bodyFold0 ? [{ kernel: bodyZ0, dispatch: list(LIST.F_BODY) }, { kernel: bodyFold0, dispatch: { direct: [NB, 1, 1] as [number, number, number] } }] : []),
      { kernel: setupF, dispatch: fRows },
      { kernel: coarsenC, dispatch: list(LIST.C_ALL) },
      { kernel: coarsenTop, dispatch: { direct: [1, 1, 1] }, w: tolBits },
    ] as BrickPlacement[],
    finish: [...(audit && walls ? [{ kernel: wallsFKernel(true), dispatch: list(LIST.F_ALL) }, { kernel: wallsCKernel(true), dispatch: list(LIST.C_ALL) }] : []),
      { kernel: finish, dispatch: fRows }, { kernel: finishC, dispatch: list(LIST.C_ALL) }] as BrickPlacement[],
  };
  return {
    label: "pool MGPCG (lane P)",
    bringUp: false,
    slots,
    // Live: after the engine's operator (pressureRowsList) the lists' arguments and rowsF open the setup, and projectF
    // closes the finish (before the engine's project, pressureProjectRest: the F_ALL path's order).
    setup: lv ? [...lv.args, ...lv.rows, ...core.setup] : core.setup,
    slot: (k: number) => [
      gS(preS, k), ...(bsb ? [gF(preU, k)] : []), gC(nativePre0, k), gF(down0, k), gC(nativePre1, k), gF(down1, k), gC(down1C, k), gC(down2, k), gC(down3, k),
      { kernel: top, dispatch: { indirect: MG_GATE.ONE }, z: k },
      gC(up3, k), gC(up2, k), gC(nativePost1, k), gF(up1, k), gC(nativePost0, k), ...(bsb ? [gF(postU, k)] : []), gS(postS, k), gF(postZ, k),
      ...(bodyZ && bodyFold ? [{ kernel: bodyZ, dispatch: list(LIST.F_BODY), z: k }, { kernel: bodyFold, dispatch: { direct: [NB, 1, 1] as [number, number, number] }, z: k }] : []),
      gFC(direction, k), gFC(update, k),
      // A slot that can never decide encodes no walls: with wallLoose Infinity, slot k decides exactly when k < wallDecisions
      // (every open slot before it decided; a closed slot k closes every later one).
      ...(walls && (Number.isFinite(wallLoose) || k < wallDecisions) ? [{ kernel: wallGate, dispatch: { direct: [1, 1, 1] }, z: k, w: tolBits } as BrickPlacement,
        { kernel: wallsF, dispatch: { indirect: MG_GATE.WF }, z: k }, { kernel: wallsC, dispatch: { indirect: MG_GATE.WC }, z: k }] : []),
      { kernel: check, dispatch: { direct: [1, 1, 1] }, z: k, w: tolBits },
    ],
    finish: lv ? [...core.finish, ...lv.project] : core.finish,
    live: lv,
    core,
    // After the composite's native projection (pmgProjectC writes an h tile's C X): compositeMGPCG places it.
    publish: [{ kernel: pressureDelta, dispatch: list(LIST.C_ALL) }],
    buffers: { pmgF: M.fWords, pmgC: M.cWords, pmgTop: T.words, pmgRed: 8 + 2 * (L.cap.f + L.cap.c), ...(bodies ? { pmgB: MB_WORDS } : {}) },
    receipt: { iterations: MG_HEADER.ITERATIONS, capped: MG_HEADER.CAPPED },
  };
}
