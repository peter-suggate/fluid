/**
 * Every number the analytic cloth scenes use, each with where it came from.
 * The builders in `scenes.ts` read only this table.
 *
 * These scenes are acceptance gates for the solver (user, 2026-11-09:
 * "correctness first ... simulation physics results matching the analytical /
 * paper"). Each has a closed-form or textbook answer, given in `references.ts`
 * and on its library card. The physical parameters are chosen for a clean,
 * visible configuration, never to make a gate pass; when the solver misses a
 * gate, the miss is the finding.
 *
 * Sources:
 * - bridson:    the Bridson 2002 cloth model (../bridson-2002/parameters.ts), reused so a
 *               gate here speaks for the paper scenes;
 * - c1:         lane C1's hanging-sheet equilibrium (tests/cloth-coarse.test.ts);
 * - textbook:   a standard result (named in `ref`);
 * - astm:       ASTM D1388 / Peirce (1930) cantilever bending test;
 * - choice:     a configuration choice made for this scene; the reason is in `ref`.
 */
import { CLOTH } from "../bridson-2002/parameters";

export type ClothAnalyticSource = "bridson" | "c1" | "textbook" | "astm" | "choice";

export interface ClothAnalyticParameter {
  readonly value: number;
  readonly unit: string;
  readonly source: ClothAnalyticSource;
  /** Short pointer into the source, or the reason for a choice. */
  readonly ref: string;
}

const P = (value: number, unit: string, source: ClothAnalyticSource, ref: string): ClothAnalyticParameter => ({ value, unit, source, ref });

/** One implicit cloth step per 1/60 s frame. Never substepped. */
export const CLOTH_ANALYTIC_STEP_S = 1 / 60;

export const GRAVITY = P(9.81, "m/s^2", "textbook", "every analytic scene sets SceneDescription.fluid.gravity_m_s2 to (0, -g, 0); the references use the same g");

/** The shared cloth: the Bridson 2002 material, so these gates speak for the paper scenes. */
export const SHEET = {
  arealDensity: P(CLOTH.arealDensity.value, "kg/m^2", "bridson", "CLOTH.arealDensity"),
  stretch: P(CLOTH.stretch.value, "N/m", "bridson", "CLOTH.stretch"),
  shear: P(CLOTH.shear.value, "N/m", "bridson", "CLOTH.shear"),
  poisson: P(CLOTH.poisson.value, "1", "bridson", "CLOTH.poisson: zero keeps the hanging sheet uniaxial"),
  compressionRatio: P(CLOTH.compressionRatio.value, "1", "bridson", "CLOTH.compressionRatio"),
  bending: P(CLOTH.bending.value, "N m", "bridson", "CLOTH.bending"),
  damping: P(CLOTH.damping.value, "s", "bridson", "CLOTH.damping"),
  thickness: P(CLOTH.thickness.value, "m", "bridson", "CLOTH.thickness"),
  strainLimit: P(CLOTH.strainLimit.value, "1", "bridson", "CLOTH.strainLimit"),
  newtonIterations: P(CLOTH.newtonIterations.value, "iterations", "bridson", "CLOTH.newtonIterations: the Newton-PCG caps of the Bridson scenes, so a gate here speaks for them"),
  newtonPcgIterations: P(CLOTH.newtonPcgIterations.value, "iterations", "bridson", "CLOTH.newtonPcgIterations"),
  newtonPcgTolerance: P(CLOTH.newtonPcgTolerance.value, "1", "bridson", "CLOTH.newtonPcgTolerance"),
  newtonResidualTolerance: P(CLOTH.newtonResidualTolerance.value, "1", "bridson", "CLOTH.newtonResidualTolerance"),
  cell: P(0.01, "m", "choice", "1 cm cloth spacing, near the Bridson sheets' 6.7-12.8 mm: the resolution at which K 8 under-converged the Fig 2 drape"),
} as const;

/** Scene 1: an unpinned flat sheet falling with nothing to touch until it lands. */
export const FREE_FALL = {
  side: P(1, "m", "choice", "a 1 m square, 101 x 101 nodes at the shared 1 cm spacing"),
  height: P(1.2, "m", "choice", "the sheet starts 1.2 m above the floor: 0.49 s of free fall, 29 steps"),
  gateSteps: P(24, "steps", "choice", "gated over the first 0.4 s, while the sheet is still 0.41 m above the floor (far outside any contact band)"),
  duration: P(2, "s", "choice", "falls, lands flat on the frictionless floor and rests"),
} as const;

/** Scene 2: lane C1's hanging Bridson sheet, pinned along its whole top edge. */
export const CURTAIN = {
  width: P(1, "m", "c1", "tests/cloth-coarse.test.ts: 1 m wide"),
  aspect: P(1.7, "1", "c1", "1.7 m hanging length, as the Fig 1 curtain"),
  nodesAcross: P(100, "nodes", "choice",
    "C1's test runs 150 nodes across (38 100 vertices); 100 keeps the 1 cm spacing of the other scenes at 16 900 vertices, so the scene settles in a reasonable wall time"),
  top: P(2, "m", "choice", "pinned edge 2 m above the floor; the stretched sheet hangs to about 0.23 m"),
  settle: P(10, "s", "c1", "C1: 'At K = 8 with the two-level correction the sheet reaches it in 10 s'"),
  duration: P(12, "s", "choice", "settles by 10 s, then holds still"),
} as const;

/** Scene 3: a sheet pinned along one edge, released from horizontal. */
export const SWING = {
  length: P(1, "m", "choice", "1 m from the pinned edge to the free edge"),
  width: P(1, "m", "choice", "1 m along the pinned edge; 101 x 101 nodes at 1 cm"),
  height: P(1.3, "m", "choice", "pinned edge 1.3 m above the floor: the swung sheet (about 1.08 m at its stretch) clears it"),
  gateSteps: P(20, "steps", "choice",
    "the outer part of the sheet stays flat for about 22 steps (Float64 oracle at 41 x 41 nodes, K 8-256: free edge on free fall to 0.01 mm through step 22); gated over 20"),
  convergedFactor: P(4, "1", "choice", "the semi-analytic reference is the same scene at 4x the Newton and PCG caps and tolerances / 4^2"),
  duration: P(4, "s", "choice", "first swing and the following swings under the pinned edge"),
} as const;

/**
 * Scene 4: Peirce cantilever strips clamped on a ledge. A card-weight strip so
 * the droop is centimetres, not millimetres (the Bridson cloth's bending length
 * is 2.5 cm).
 */
export const PEIRCE = {
  arealDensity: P(0.2, "kg/m^2", "choice", "a 200 g/m^2 card-weight strip"),
  bending: P(1e-3, "N m", "choice", "bending length c = (B / rho g)^(1/3) = 79.9 mm: overhangs of 12-20 cm"),
  stretch: P(200, "N/m", "bridson", "CLOTH.stretch; under its own weight the strip stretches 1e-4: inextensible to the gate"),
  shear: P(100, "N/m", "bridson", "CLOTH.shear"),
  damping: P(0.05, "s", "choice",
    "Rayleigh damping of the first bending mode (about 10 rad/s) at a quarter of critical, so the strips come to rest within the scene; the static answer does not depend on it"),
  ratios: [
    P(1.5, "1", "choice", "overhang / bending length: exact chord angle 22.2 deg"),
    P(2, "1", "astm", "ASTM D1388: the overhang at which the tip falls to the 41.5 deg line is twice the bending length"),
    P(2.5, "1", "choice", "overhang / bending length: exact chord angle 56.9 deg"),
  ],
  stripWidth: P(0.04, "m", "choice", "4 cm strips, 9 nodes across"),
  cell: P(0.005, "m", "choice", "5 mm spacing: 24-40 hinges along each overhang"),
  clamp: P(0.06, "m", "astm", "the strip's on-ledge part, held flat as under D1388's weighted slide (pinned vertices)"),
  ledgeHeight: P(0.5, "m", "choice", "the ledge top 0.5 m above the floor"),
  gap: P(0.004, "m", "choice", "the clamped part rests 4 mm above the ledge top, outside the 1 mm contact radius plus the field band"),
  settle: P(6, "s", "choice", "measured at 6 s, after the damped droop has rung down"),
  duration: P(8, "s", "choice", "droops and holds"),
} as const;

/** Scene 5: two square patches on voxel inclines either side of the friction angle. */
export const INCLINE = {
  friction: P(0.5, "1", "choice", "one Coulomb mu for static and kinetic (paper section 7.3): friction angle atan 0.5 = 26.6 deg"),
  stickAngle: P(20, "deg", "choice", "tan 20 = 0.364 < mu: the patch sticks"),
  slideAngle: P(35, "deg", "choice", "tan 35 = 0.700 > mu: a = g (sin 35 - mu cos 35) = 1.609 m/s^2"),
  patch: P(0.1, "m", "choice", "10 cm square patches, 11 x 11 nodes at 1 cm"),
  slabLength: P(1.2, "m", "choice", "1.2 m of slope: the sliding patch travels 0.51 m in the 0.8 s gate"),
  slabWidth: P(0.3, "m", "choice", "30 cm wide slabs"),
  slabThickness: P(0.06, "m", "choice", "6 cm thick, six voxel cells"),
  gap: P(0.003, "m", "choice", "the patch starts 3 mm above the slope, settling onto it in the first steps"),
  gateStart: P(0.2, "s", "choice", "acceleration fitted from 0.2 s (after the patch has settled onto the slope) to 1.0 s"),
  gateEnd: P(1, "s", "choice", "before the sliding patch reaches the foot of its slope"),
  duration: P(3, "s", "choice", "the sliding patch runs off its slope onto the floor"),
  cell: P(0.01, "m", "choice", "1 cm voxels, as the Bridson solids"),
} as const;

export const CLOTH_ANALYTIC_PARAMETERS = { GRAVITY: { g: GRAVITY }, SHEET, FREE_FALL, CURTAIN, SWING, PEIRCE, INCLINE } as const;
