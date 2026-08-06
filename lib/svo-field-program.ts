/**
 * The shape language — H2 of `docs/hero-fidelity-1000x-handoff.md`.
 *
 * A record's geometry stops being a primitive and becomes a **program**: a
 * short tape of composed field ops, evaluated as one SDF. The point is stated
 * in §1.2 of the handoff and is worth restating, because it is the whole reason
 * this file exists rather than a twelfth entry in `svo-primitive-kinds.ts`:
 *
 * > 501 records must stay ~501 records. A floret canopy authored as records is
 * > ~50 000 of them and hits the cluster arena, the material table and the BVH
 * > in that order.
 *
 * Effective surface features become unbounded while record count does not move.
 * Cost is paid per voxel near the surface at voxelization time, amortised over
 * frames.
 *
 * ---------------------------------------------------------------------------
 * What is here, and what is deliberately not yet
 * ---------------------------------------------------------------------------
 * The framework, plus the ops §2.2 names as the ones that make stone read as
 * stone:
 *
 *   - the tape encoding, register machine and arena packing — the ABI, which is
 *     the part that is expensive to change later;
 *   - **Lipschitz through the graph** (§2.4), carried from day one because
 *     retrofitting it is a rewrite;
 *   - `domain-warp` (op 1), `anisotropic-frame` (op 2), `fracture` (op 3), the
 *     variable-k `smooth-union` / `smooth-subtract` / `smooth-intersect` family
 *     (op 4), `worley-subtract` and `ridged-worley-add` (op 5), `erosion-mask`
 *     (op 6) and the recursive `scatter` (op 7), over the three source solids;
 *   - the TypeScript evaluator and its generated WGSL mirror, sharing the
 *     tree's existing `svoProceduralNoise` and `svoProceduralHashCell` so the
 *     two sides agree by construction rather than by transcription.
 *
 * Op 8 of §2.2 — sparse hash-placed chip and crack defects — is *not* here. The
 * encoding has room for it and the register machine is general enough to run
 * it; what is missing is its op-table row and evaluator arm.
 *
 * **Interval pruning (§2.5) is not here either, and that is a scoping decision
 * rather than the omission the handoff warns about.** The warning is that a
 * *deep* tape evaluated per sample is unaffordable and that a naive-first
 * implementation will produce a "field programs are too slow" measurement that
 * is an artifact. That warning now bites: a fully dressed stone is a dozen-odd
 * ops and the two Worley ops are eight cell hashes apiece, so pruning is the
 * next thing this module needs and the first thing to reach for if a
 * voxelization measurement disappoints. What *is* needed now, and is here, is a
 * sound conservative **extent** bound: a warp moves the surface, a scatter
 * replicates it, and a proxy box that did not account for either would clip the
 * shape.
 *
 * ---------------------------------------------------------------------------
 * The one rule every op in here obeys
 * ---------------------------------------------------------------------------
 * Every op returns a distance **and** the Lipschitz constant of the field that
 * produced it, and that constant is derived analytically from the op's own
 * parameters and its operands' constants — never guessed, never a round number
 * chosen because it looked safe. `tests/svo-field-program.test.ts` measures the
 * achieved gradient by finite difference for every op individually and for a
 * tape that uses all of them at once, because §2.4 is blunt about the failure
 * mode: a tape that overstates its clearance punches holes in every rock, worst
 * at exactly the silhouettes the eye reads first.
 *
 * The second rule, which the first depends on, is **continuity**. Two of the
 * ops here — Worley cellular noise and domain repetition — are written in the
 * literature in forms that are quietly discontinuous at cell boundaries, and a
 * discontinuity has no finite Lipschitz constant at all. Both are built here so
 * that the truncated neighbourhood search is provably *exact* over the range
 * that matters and provably *clamped* outside it; the derivations are beside
 * the ops and the clamps are the reason the tests pass rather than a detail.
 *
 * ---------------------------------------------------------------------------
 * The band limit — why every evaluation carries a sampling length
 * ---------------------------------------------------------------------------
 * Shading is voxels only. The field program is not evaluated at shade time at
 * all: the **voxelizer** evaluates it, to decide a cell's occupancy and its
 * material, and nothing analytic reaches the screen. That single fact settles
 * what the "detail band" of §2.6 is measured against — it is the voxel cell
 * size, a world-space length that is uniform for a voxelization pass and known
 * by the caller. There is no ray cone, no camera and no per-pixel anything in
 * this file, and there must not be.
 *
 * So `evaluateSvoFieldProgram` takes a `samplingLength_m`, and every op that
 * introduces a spatial period fades **analytically to zero** as that period
 * approaches twice the sampling length. This is textbook antialiasing before a
 * resample rather than a level-of-detail approximation: an octave whose
 * wavelength is under two cells cannot change an occupancy decision, it can
 * only alias, so removing it is the correct answer and not a cheaper one.
 *
 * The fade lowers the **Lipschitz contribution as well as the amplitude**, and
 * that is the half that makes the design affordable rather than merely correct.
 * Every faded op is evaluated as `mix(input, output, fade)` with `fade` a
 * constant over the pass, so its bound is `mix(L_input, L_output, fade)` by the
 * same convex combination — an octave that has faded out costs exactly nothing
 * in march steps. A faded-out octave that still paid its worst-case bound would
 * make coarse voxelization march at fine-detail cost for detail it cannot show,
 * which is the difference between this approach being affordable and not.
 *
 * One corollary worth stating because it constrains authoring: with lacunarity
 * 2, octave `i` of a warp chain contributes a factor `1 + A·f·J·(2g)^i` to the
 * bound, where `g` is the amplitude gain. The product over octaves converges if
 * and only if `Σ (2g)^i` does, that is if and only if **`g < 0.5`**. At exactly
 * one half every octave contributes the same factor, the bound grows without
 * limit in octave count, and so does every march step through the stone. See
 * `SVO_FIELD_RECOMMENDED_OCTAVE_GAIN`.
 *
 * ---------------------------------------------------------------------------
 * Why the point is a register and not an argument
 * ---------------------------------------------------------------------------
 * A domain warp is not a transform applied to a returned distance — it changes
 * *where the subtree is evaluated*: `d(p) = base(p + A·w(p)) / L`. A tape whose
 * registers held only distances could not express that, and the natural-looking
 * fix (let each source op carry its own warp parameters) forecloses §2.3's
 * recursive scatter, which is a domain op whose occupant is another domain op.
 *
 * So the machine has two register files, points and field values, and ops are
 * typed by which they write. A scatter is then a point op like any other, and
 * "recursion depth = octave count = detail band count" falls out as *how many
 * domain ops the tape has* — a number the LOD can choose per brick without the
 * shape language knowing anything about cameras.
 */

import {
  sampleSvoProceduralNoise,
  svoProceduralHashCell,
} from "./svo-procedural-material";
import type { Vec3 } from "./model";

const f32 = Math.fround;

// ---------------------------------------------------------------------------
// Machine limits
// ---------------------------------------------------------------------------

/**
 * Registers of each kind.
 *
 * Four rather than "as many as the tape is long" because the shader indexes
 * these dynamically, and a dynamically indexed array in WGSL is scratch memory
 * on this backend, not registers. Four points and four field values is 20
 * floats of state — enough for a chain with a couple of live branches, which is
 * every shape in §2.2 once the tape is scheduled in post-order.
 *
 * Raising this is a measurement, never a convenience: it costs occupancy on a
 * path that is already primary-visibility bound.
 *
 * Both stayed at four through the op set of §2.2, which is the evidence that
 * four is enough: the worked tape in the tests runs every op in the language —
 * frame, two warps, a scatter, three sources, three smooth combinations, a
 * fracture, both Worley ops and an erosion mask — inside four points and four
 * fields. What did grow is the *per-register* state, from one scalar to three
 * on each side: a point now carries its Lipschitz factor and its fold
 * clearance, a field its Lipschitz constant, its characteristic radius and its
 * saturation value. That is 48 bytes more dynamically-indexed scratch per
 * invocation, and it is the price of three separate things the op set cannot do
 * without — a smooth-min whose blend radius knows how big its operands are, a
 * domain repetition that stays continuous at its cell walls, and a fracture
 * plane that can cut a scattered occupant without tearing it.
 */
export const SVO_FIELD_PROGRAM_POINT_REGISTERS = 4;
export const SVO_FIELD_PROGRAM_FIELD_REGISTERS = 4;

/**
 * Ops per tape.
 *
 * The handoff budgets 20-80 nodes for a full boulder or canopy. Sixteen was
 * what op 1 alone needed several times over, and it is *still* the limit after
 * the whole op set of §2.2 landed, because the fully dressed stone that
 * `tests/svo-field-program.test.ts` builds — anisotropic frame, two warp
 * octaves, a scatter, three sources, three smooth combinations, a fracture
 * plane, both Worley ops and an erosion mask — is fourteen ops. Raising it is
 * not free: the block is `4 + 8·ops` words, so the arena cost per tape is
 * linear in this number whether a tape uses the room or not.
 *
 * Where it will have to rise is a full 3-6 plane fracture set on a scattered,
 * self-similar canopy, and interval pruning (§2.5) has to arrive with that
 * rise, because that is the point at which per-sample tape depth starts to
 * matter more than tape capacity.
 */
export const SVO_FIELD_PROGRAM_MAXIMUM_OPS = 16;

/**
 * Words per packed op. Five scalar parameters, two operands, a seed and a code.
 *
 * The fifth parameter costs nothing: it occupies the word that was reserved and
 * written as zero from the start, which sits *before* the other four in the
 * block rather than after them. Leaving it where the spare word already was
 * keeps the byte layout of every previously packed tape identical — a zero word
 * decodes as a zero parameter either way — and moving it for tidiness would
 * have been an ABI change in exchange for nothing.
 */
export const SVO_FIELD_PROGRAM_OP_WORDS = 8;

/**
 * The packed byte meaning "this operand is not read".
 *
 * `SVO_PRIMITIVE_INVALID_REFERENCE`'s discipline, one level down: an unused
 * operand must not be indistinguishable from a read of register 0, or a stale
 * word decodes as a shape nobody authored.
 */
export const SVO_FIELD_PROGRAM_UNUSED_OPERAND = 0xff;

/** Words per tape block in the arena: a header word pair plus the ops. */
export const SVO_FIELD_PROGRAM_HEADER_WORDS = 4;
export const SVO_FIELD_PROGRAM_BLOCK_WORDS =
  SVO_FIELD_PROGRAM_HEADER_WORDS + SVO_FIELD_PROGRAM_MAXIMUM_OPS * SVO_FIELD_PROGRAM_OP_WORDS;

// ---------------------------------------------------------------------------
// The op table
// ---------------------------------------------------------------------------

/** Which register file an op writes. Points and fields are not interchangeable. */
export type SvoFieldOpResult = "point" | "field";

export interface SvoFieldOpPolicy {
  readonly name: string;
  /** Stable ABI code. Never reused, never reordered. */
  readonly code: number;
  readonly result: SvoFieldOpResult;
  /**
   * How many field registers this op consumes.
   *
   * Declared rather than inferred, because "unused" and "register 0" are the
   * same byte once an op is packed, and a validator that could not tell them
   * apart would either reject every source op or accept a combining op whose
   * operand was never written. The op is the only thing that knows.
   */
  readonly readsFields: 0 | 1 | 2;
  /** WGSL constant emitted for this op, so the shader never spells a literal. */
  readonly wgslConstant: string;
  /** One line on what the op buys, from §2.2's table. */
  readonly buys: string;
  /**
   * The coarsest spatial period this op introduces, as a function of its own
   * parameters, in metres — or `null` for ops that add no new scale.
   *
   * This is the input to the band limit, and it is no longer only advisory: an
   * op that declares a period is faded out by `svoFieldBandLimitFade` once that
   * period falls under twice the sampling length, in amplitude *and* in
   * Lipschitz contribution. An op that declares `null` introduces no scale of
   * its own — a plane, a smooth blend, a mask — and is never faded, because
   * there is nothing about it that a coarser voxel grid cannot represent.
   *
   * The three-band law of §2.6 reads the same number: a period above twice the
   * leaf is geometry, which is precisely the range this fade leaves alone.
   */
  readonly coarsestPeriod_m: ((parameters: SvoFieldOpParameters) => number) | null;
}

export interface SvoFieldOpParameters {
  readonly a: number;
  readonly b: number;
  readonly c: number;
  readonly d: number;
  /** The fifth scalar; see `SVO_FIELD_PROGRAM_OP_WORDS` for where it lives. */
  readonly e: number;
}

/**
 * Every op, in one frozen table, in the shape `svo-procedural-material.ts`
 * established: a typed row set that the CPU evaluator reads and the WGSL is
 * generated from, so the two cannot drift into disagreement silently.
 */
export const SVO_FIELD_OP_TABLE = Object.freeze([
  Object.freeze({
    name: "sphere",
    readsFields: 0,
    code: 1,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_SPHERE",
    buys: "source solid; the exact SDF, Lipschitz 1 by construction",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "ellipsoid",
    readsFields: 0,
    code: 2,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_ELLIPSOID",
    buys: "source solid with three radii; the mass before any warp",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "box",
    readsFields: 0,
    code: 3,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_BOX",
    buys: "source solid with hard features; the bedding plane a rock broke on",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "domain-warp",
    readsFields: 0,
    code: 4,
    result: "point",
    wgslConstant: "SVO_FIELD_OP_DOMAIN_WARP",
    buys: "organic mass — §2.2 op 1, the single highest-value op in the language",
    // One lattice period of the noise, in metres: the frequency is in 1/m.
    coarsestPeriod_m: (parameters: SvoFieldOpParameters) => (parameters.b > 0 ? 1 / parameters.b : Infinity),
  }),
  Object.freeze({
    name: "fracture",
    readsFields: 1,
    code: 5,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_FRACTURE",
    buys: "flat faces with tight arrises — §2.2 op 3, the strongest single 'this is stone' cue",
    // A halfspace has no period. Repeating it is the author's job, not the op's.
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "smooth-union",
    readsFields: 2,
    code: 6,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_SMOOTH_UNION",
    buys: "unions that read at both ends — §2.2 op 4, blend radius scaled by the smaller operand",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "smooth-subtract",
    readsFields: 2,
    code: 7,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_SMOOTH_SUBTRACT",
    buys: "bites and hollows taken out of a mass, at the bitten feature's own blend radius",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "smooth-intersect",
    readsFields: 2,
    code: 8,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_SMOOTH_INTERSECT",
    buys: "a mass trimmed to a second solid; the slab a bedded stone was cut from",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "anisotropic-frame",
    readsFields: 0,
    code: 9,
    result: "point",
    wgslConstant: "SVO_FIELD_OP_ANISOTROPIC_FRAME",
    buys: "bedding and flattening — §2.2 op 2, a stone lying the way it fell rather than isotropic",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "worley-subtract",
    readsFields: 1,
    code: 10,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_WORLEY_SUBTRACT",
    buys: "the grain band — dense subtractive pitting, measured off the plate at about three millimetres",
    // The cellular lattice period, in metres, and therefore the band limit this
    // op fades against: at a 25 mm voxel a 3 mm pit is pure alias and fades out.
    coarsestPeriod_m: (parameters: SvoFieldOpParameters) => (parameters.a > 0 ? parameters.a : Infinity),
  }),
  Object.freeze({
    name: "ridged-worley-add",
    readsFields: 1,
    code: 11,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_RIDGED_WORLEY_ADD",
    buys: "hard inclusions standing proud of a receded matrix; the lower-value twin of the pitting",
    coarsestPeriod_m: (parameters: SvoFieldOpParameters) => (parameters.a > 0 ? parameters.a : Infinity),
  }),
  Object.freeze({
    name: "erosion-mask",
    readsFields: 2,
    code: 12,
    result: "field",
    wgslConstant: "SVO_FIELD_OP_EROSION_MASK",
    buys: "weathering that collects where water and grit collect — §2.2 op 6, fabric becomes stone",
    coarsestPeriod_m: null,
  }),
  Object.freeze({
    name: "scatter",
    readsFields: 0,
    code: 13,
    result: "point",
    wgslConstant: "SVO_FIELD_OP_SCATTER",
    buys: "the florets — §2.2 op 7; an occupant that is itself a scatter is self-similarity for one op",
    // The replication period. Declared for §2.6's band assignment, but see the
    // evaluator: a scatter is deliberately *not* faded, because fading a domain
    // repetition does not attenuate a feature, it deletes instances.
    coarsestPeriod_m: (parameters: SvoFieldOpParameters) => (parameters.a > 0 ? parameters.a : Infinity),
  }),
] as const satisfies readonly SvoFieldOpPolicy[]);

export type SvoFieldOpName = (typeof SVO_FIELD_OP_TABLE)[number]["name"];

const opByName = new Map<string, SvoFieldOpPolicy>(SVO_FIELD_OP_TABLE.map((op) => [op.name, op]));
const opByCode = new Map<number, SvoFieldOpPolicy>(SVO_FIELD_OP_TABLE.map((op) => [op.code, op]));

export function svoFieldOp(name: SvoFieldOpName): SvoFieldOpPolicy {
  const op = opByName.get(name);
  if (!op) throw new RangeError(`unknown field op ${name}`);
  return op;
}

// ---------------------------------------------------------------------------
// Lipschitz
// ---------------------------------------------------------------------------

/**
 * Bound on the operator norm of the noise displacement's Jacobian, per unit of
 * *scaled* domain (that is, before multiplying by the warp's frequency).
 *
 * `svoProceduralNoise` is trilinear interpolation of per-cell hashes under the
 * smoothstep blend `s(t) = t²(3 − 2t)`. Along one axis the derivative is
 * `s'(fx) · (difference of two bilinear interpolations)`; `s'` peaks at 1.5 and
 * the hash values live in [0, 1], so that difference lies in [−1, 1] and
 * `|∂n/∂x| ≤ 1.5`. The warp maps the noise to [−1, 1], doubling it to 3. A
 * three-component displacement then has Frobenius norm at most `3·√3`, which
 * bounds the operator norm.
 *
 * `tests/svo-field-program.test.ts` measures the achieved maximum on a dense
 * grid and asserts it is *under* this — a bound that is not checked is a
 * decoration, and the tracer divides by it.
 */
export const SVO_FIELD_NOISE_JACOBIAN_BOUND = 3 * Math.sqrt(3);

/**
 * Bound on the gradient magnitude of the Worley cellular distance this module
 * samples. It is exactly **one**, and that is a construction rather than an
 * observation.
 *
 * `sampleSvoFieldWorley_m` returns `min(cell/2, min over eight cells of the
 * distance to that cell's jittered feature point)`. Every term of the inner
 * minimum is the distance from the sample to a point, which is 1-Lipschitz; the
 * outer term is a constant; and a pointwise minimum of 1-Lipschitz functions is
 * 1-Lipschitz. The only way the argument fails is if the *set* being minimised
 * over changes discontinuously as the sample crosses a lattice boundary — which
 * it does, and which the `cell/2` clamp is there to neutralise. The derivation
 * is on `sampleSvoFieldWorley_m`.
 *
 * Measured against a dense grid in `tests/svo-field-program.test.ts`, including
 * samples deliberately walked across lattice boundaries, because a truncated
 * cellular search that had a discontinuity anywhere would have *no* finite
 * bound and the tracer would step straight through the surface there.
 */
export const SVO_FIELD_WORLEY_JACOBIAN_BOUND = 1;

/**
 * The sampling factor the band limit is written against: a feature needs its
 * period to exceed this many sampling lengths before it survives.
 *
 * Two is Nyquist and nothing more interesting than that. A period of exactly
 * two voxels is the finest thing an occupancy decision can even in principle
 * distinguish from its neighbour, so the fade reaches zero there and holds full
 * amplitude one octave above it, at four.
 */
export const SVO_FIELD_BAND_LIMIT_SAMPLES = 2;

/**
 * The amplitude gain a multi-octave warp chain should be authored at.
 *
 * With lacunarity 2, octave `i` has amplitude `A·g^i` and frequency `f·2^i`, so
 * it multiplies the tape's Lipschitz constant by `1 + A·f·J·(2g)^i`. The
 * product over octaves is bounded exactly when `Σ (2g)^i` converges, that is
 * when `g < 1/2`. At `g = 1/2` every octave contributes the same factor and the
 * bound — and therefore the number of march steps the voxelizer takes through
 * the stone — grows geometrically with octave count for detail that is by
 * construction getting finer and less visible.
 *
 * At `0.42` the series sums to `1/(1 − 0.84) = 6.25`, so a chain of any depth
 * has a bound within a small constant of a three-octave one. The useful range
 * is 0.40 to 0.45; below that the fine octaves stop contributing shape, above
 * it the march cost stops converging.
 */
export const SVO_FIELD_RECOMMENDED_OCTAVE_GAIN = 0.42;

/**
 * Slack on the anisotropic frame's rotation, whose matrix is orthonormal only
 * to single-precision.
 *
 * The rotation is built by normalising a hash-derived quaternion, so its
 * operator norm is one up to a few units in the last place. One part in ten
 * thousand is orders of magnitude more than that costs and it keeps the finite
 * difference assertion from failing on a rounding tie, which would be a false
 * alarm about the one thing in this file that must never produce false alarms.
 */
const FRAME_ORTHONORMALITY_MARGIN = 1.0001;

/**
 * The value a field saturates at when nothing has clamped it.
 *
 * A literal rather than `Infinity` because WGSL has no infinity literal and the
 * two evaluators are the same table: a sentinel both sides can spell is worth
 * more than a mathematically tidier value one side would have to approximate.
 * It is far above any authored record extent, so `min(distance, this)` is the
 * identity everywhere it is not deliberately lowered by a scatter.
 */
const UNBOUNDED_DISTANCE_M = 1e20;

/** Salts for the three components of the warp displacement. Distinct, stable ABI values. */
const WARP_SALT_X = 0x51ed_2701;
const WARP_SALT_Y = 0x9c1e_7f3b;
const WARP_SALT_Z = 0x2f6d_c5a9;

/** Salts for the four quaternion components of the anisotropic frame's rotation. */
const FRAME_SALT_X = 0x1d4b_9f07;
const FRAME_SALT_Y = 0x77a3_e15d;
const FRAME_SALT_Z = 0x3b8c_2fd1;
const FRAME_SALT_W = 0xa1f5_60b3;

/** Salts for the three components of a Worley cell's feature-point jitter. */
const WORLEY_SALT_X = 0x6d2b_79f5;
const WORLEY_SALT_Y = 0xb5c7_1e2d;
const WORLEY_SALT_Z = 0x0f93_a48b;

/** Salts for the three components of a scatter cell's placement jitter. */
const SCATTER_SALT_X = 0x4e7d_1c35;
const SCATTER_SALT_Y = 0x9b21_f6a7;
const SCATTER_SALT_Z = 0xd3a8_5e19;

/**
 * The band-limit fade: what fraction of an op's effect survives being sampled
 * on a grid of the given spacing.
 *
 * One when the feature's period is at least twice the Nyquist limit — four
 * sampling lengths — and zero at the limit itself, with the tree's own
 * smoothstep between them so the fade is C¹ in the sampling length and a
 * voxelization that changes resolution does not pop.
 *
 * A sampling length of zero means "do not band-limit", which is what the CPU
 * oracles and the authoring-time extent bound want: they must see the field the
 * finest voxelization would see, not a blurred one.
 */
export function svoFieldBandLimitFade(period_m: number, samplingLength_m: number): number {
  if (!(samplingLength_m > 0)) return 1;
  if (!(period_m > 0)) return 1;
  const octaves = period_m / (SVO_FIELD_BAND_LIMIT_SAMPLES * samplingLength_m);
  const t = Math.min(1, Math.max(0, octaves - 1));
  return f32(t * t * (3 - 2 * t));
}

/**
 * The quadratic smooth minimum, in the form whose gradient is a *convex
 * combination* of its operands' gradients.
 *
 * That form is not a stylistic choice. Writing `h` as a clamped linear blend
 * and subtracting `k·h·(1−h)` makes the two derivative terms — the one from
 * moving the blend weight and the one from the polynomial correction — cancel
 * exactly, leaving `∇smin = (1−h)·∇b + h·∇a`. So `|∇smin| ≤ max(|∇a|, |∇b|)`,
 * which is precisely the rule §2.4 states for smooth-min and the reason a
 * blended union costs nothing in march steps. The popular `−log(exp(−a/k) +
 * exp(−b/k))·k` form does not have this property.
 *
 * Two consequences the extent bound depends on, both from `h·(1−h) ≤ 1/4`:
 * `min(a,b) − k/4 ≤ smin(a,b,k) ≤ min(a,b)`.
 */
function smoothMinimum(a: number, b: number, k: number): number {
  if (k > 0) {
    const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (b - a)) / k));
    return b + (a - b) * h - k * h * (1 - h);
  }
  return Math.min(a, b);
}

/**
 * The smooth maximum, `−smin(−a, −b, k)` written out.
 *
 * Same convex-combination gradient, and the mirrored envelope
 * `max(a,b) ≤ smax(a,b,k) ≤ max(a,b) + k/4`. The upper half of that is what
 * makes every op built on `smax` — fracture, subtraction, intersection, the
 * Worley carve — free for the extent bound: they can only *raise* the field,
 * which can only shrink the solid.
 */
function smoothMaximum(a: number, b: number, k: number): number {
  if (k > 0) {
    const h = Math.min(1, Math.max(0, 0.5 + (0.5 * (a - b)) / k));
    return b + (a - b) * h + k * h * (1 - h);
  }
  return Math.max(a, b);
}

/** The largest deviation `smin`/`smax` can put between its result and the hard min/max. */
function smoothOvershoot_m(k: number): number {
  return k > 0 ? k / 4 : 0;
}

/**
 * The blend radius of a smooth combination, scaled by the **smaller** operand's
 * characteristic radius.
 *
 * §2.2 op 4 in one function, and the reason the current stone set reads as a
 * lava lamp: a single authored `smoothRadius_m` serving a whole aggregate
 * blends a two centimetre lobe into a twenty centimetre mass with the same
 * fillet, which dissolves the lobe and barely touches the mass. That is the
 * blobbiness, precisely. Scaling by the smaller operand makes both joints read
 * — the lobe keeps its own silhouette and the mass keeps its own shoulder — and
 * the fraction is authored once for the whole aggregate instead of once per
 * pair.
 *
 * The floor and ceiling are what stop the scaling running away at the two ends:
 * a source with no radius yet would otherwise blend at zero, and a very large
 * mass would blend at a radius wider than the features it is joining.
 *
 * One function rather than two because the extent bound has to know the same
 * number the evaluator will use. A second copy of this rule that drifted would
 * put the proxy box a quarter of a blend radius inside the surface, which is
 * exactly the silhouette clip the box exists to prevent.
 */
function svoFieldBlendRadius_m(parameters: SvoFieldOpParameters, radiusA_m: number, radiusB_m: number): number {
  const smaller_m = Math.min(radiusA_m, radiusB_m);
  const floor_m = Math.max(0, parameters.b);
  const ceiling_m = parameters.c > 0 ? parameters.c : UNBOUNDED_DISTANCE_M;
  return Math.min(Math.max(Math.max(0, parameters.a) * smaller_m, floor_m), ceiling_m);
}

/**
 * A field value: a distance, and the Lipschitz constant of the field that
 * produced it.
 *
 * These travel together and are never separated. The primary ray and every
 * visibility cone sphere-trace `distance`, and a tracer that steps the returned
 * value through a warped field overshoots and punches holes in every rock —
 * §2.4 names this as the constraint that kills naive versions of the design.
 * The correct step is `distance / lipschitz`, and the only way to make that
 * impossible to forget is to make the pair the type the evaluator returns.
 */
export interface SvoFieldValue {
  readonly distance_m: number;
  readonly lipschitz: number;
}

/** The safe sphere-trace step for a field value: never larger than the true clearance. */
export function svoFieldStep_m(value: SvoFieldValue): number {
  return value.distance_m / Math.max(1, value.lipschitz);
}

// ---------------------------------------------------------------------------
// Authoring
// ---------------------------------------------------------------------------

export interface SvoFieldOpDescriptor {
  readonly op: SvoFieldOpName;
  /** Register this op writes, in its own file. */
  readonly out: number;
  /** Point register read by source ops and by domain ops. */
  readonly point?: number;
  /** Field registers read by combining ops. Unused by everything in this landing. */
  readonly fieldA?: number;
  readonly fieldB?: number;
  readonly seed?: number;
  readonly parameters?: Partial<SvoFieldOpParameters>;
}

export interface SvoFieldProgram {
  readonly ops: readonly SvoFieldOpDescriptor[];
  /** Field register holding the result. */
  readonly result: number;
}

const ZERO_PARAMETERS: SvoFieldOpParameters = Object.freeze({ a: 0, b: 0, c: 0, d: 0, e: 0 });

function parametersOf(descriptor: SvoFieldOpDescriptor): SvoFieldOpParameters {
  return { ...ZERO_PARAMETERS, ...descriptor.parameters };
}

/**
 * The per-axis scale an `anisotropic-frame` applies, with non-positive
 * components read as one.
 *
 * A scale of zero is not a flattened solid, it is a collapsed domain with an
 * infinite Lipschitz constant — which is to say it is an unwritten word, not a
 * shape. The honest decode of an unwritten word here is "this axis is not
 * scaled", and it makes the all-zero parameter block the identity frame rather
 * than a singularity, which is the same discipline the rest of the file applies
 * to a stale operand byte.
 */
function frameScale(parameters: SvoFieldOpParameters): [number, number, number] {
  return [
    parameters.a > 0 ? parameters.a : 1,
    parameters.b > 0 ? parameters.b : 1,
    parameters.c > 0 ? parameters.c : 1,
  ];
}

/**
 * How far below the true clearance a source op's distance bound can read, as a
 * multiplicative factor.
 *
 * Only `scatter` cares. Its cell-wall clamp has to sit at or below the value
 * the occupant's own distance function *reports* on the wall, not at the
 * geometric gap, and the ellipsoid's gradient-normalised bound is the one
 * source here that reports less than the truth. For radii `r`, far from the
 * solid that bound is at least `(|p| − max r)·(min r)² / (max r)²`, so it
 * understates by at most the squared radius ratio. Sphere and box are exact
 * outside, so they are one.
 *
 * Shrinking a fold clamp is always safe — it can only make the field flatter
 * and more conservative — so a factor that is too large costs a little march
 * speed near a scattered occupant and can never cost correctness.
 */
function sourceClearanceFactor(op: SvoFieldOpPolicy, parameters: SvoFieldOpParameters): number {
  if (op.name !== "ellipsoid") return 1;
  const radii = [Math.abs(parameters.a), Math.abs(parameters.b), Math.abs(parameters.c)].map((value) =>
    Math.max(1e-6, value),
  );
  const ratio = Math.max(...radii) / Math.min(...radii);
  return ratio * ratio;
}

/** The radius of the smallest origin-centred ball containing a source op's solid. */
function sourceBoundingRadius_m(op: SvoFieldOpPolicy, parameters: SvoFieldOpParameters): number {
  if (op.name === "sphere") return Math.abs(parameters.a);
  // A box's corner is further out than its largest half-extent; an ellipsoid's
  // is not. Getting this the wrong way round would let a scattered box poke
  // through its own cell wall, which is the one thing the clearance forbids.
  if (op.name === "box") return Math.hypot(parameters.a, parameters.b, parameters.c);
  return Math.max(Math.abs(parameters.a), Math.abs(parameters.b), Math.abs(parameters.c));
}

/** The characteristic feature radius a source writes into its field register; see `smooth-union`. */
function sourceCharacteristicRadius_m(op: SvoFieldOpPolicy, parameters: SvoFieldOpParameters): number {
  if (op.name === "sphere") return Math.abs(parameters.a);
  // The *smallest* semi-axis, not the largest: a blend wider than the tightest
  // curvature on a solid swallows the feature it was supposed to join.
  return Math.min(Math.abs(parameters.a), Math.abs(parameters.b), Math.abs(parameters.c));
}

/**
 * A `scatter`'s geometry, resolved from its raw parameters once so the
 * evaluator, the validator and the extent bound cannot disagree about it.
 */
interface ScatterGeometry {
  readonly cell_m: number;
  readonly repetitions: number;
  readonly jitter_m: number;
  readonly occupantRadius_m: number;
  /** Half-extent of the whole replicated array, per axis, in the parent frame. */
  readonly arrayHalf_m: number;
  /**
   * The gap between a cell wall and the furthest an occupant may reach.
   *
   * This is the single number the whole construction turns on. It is the value
   * the folded field saturates at, it is a lower bound on the distance from a
   * cell wall to *any* instance, and it is what makes a jittered domain
   * repetition continuous rather than merely usually-continuous.
   */
  readonly clearance_m: number;
}

function scatterGeometry(parameters: SvoFieldOpParameters): ScatterGeometry {
  const cell_m = Math.max(1e-6, parameters.a);
  const repetitions = Math.max(0, Math.floor(parameters.b));
  const jitter_m = Math.min(cell_m, Math.max(0, parameters.c));
  const occupantRadius_m = Math.max(0, parameters.d);
  return {
    cell_m,
    repetitions,
    jitter_m,
    occupantRadius_m,
    arrayHalf_m: repetitions * cell_m + 0.5 * cell_m,
    clearance_m: 0.5 * cell_m - 0.5 * jitter_m - occupantRadius_m,
  };
}

/**
 * Reject a program the machine cannot run, at authoring time.
 *
 * The discipline is `SVO_CLUSTER_ARENA_BLOCK`'s, verbatim in spirit: a slot past
 * capacity decodes as a **miss, never as the envelope**, because "an aggregate
 * that quietly became an ellipsoid is a shape nobody authored". The same
 * applies one level up — a tape that silently dropped its warp would render a
 * smooth ellipsoid, which is precisely the frame this whole program exists to
 * stop producing, and it would do it without a single error.
 */
export function validateSvoFieldProgram(program: SvoFieldProgram): void {
  if (program.ops.length === 0) throw new RangeError("a field program must have at least one op");
  if (program.ops.length > SVO_FIELD_PROGRAM_MAXIMUM_OPS) {
    throw new RangeError(
      `field program has ${program.ops.length} ops, over the ${SVO_FIELD_PROGRAM_MAXIMUM_OPS}-op machine limit`,
    );
  }
  // Point register 0 is the incoming local point and is live before op 0.
  const pointWritten = new Set<number>([0]);
  const fieldWritten = new Set<number>();
  // The two facts a forward pass has to carry to police `scatter`. `budget_m`
  // is the largest bounding radius an occupant may still have *in this
  // register's own frame*; `folded` records whether a domain repetition is
  // anywhere above this register. Both start unconstrained, so a tape with no
  // scatter in it can never fail either of the new rules.
  const budget_m = new Array<number>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(Infinity);
  const folded = new Array<boolean>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(false);
  program.ops.forEach((descriptor, index) => {
    const op = svoFieldOp(descriptor.op);
    const limit = op.result === "point" ? SVO_FIELD_PROGRAM_POINT_REGISTERS : SVO_FIELD_PROGRAM_FIELD_REGISTERS;
    if (!Number.isInteger(descriptor.out) || descriptor.out < 0 || descriptor.out >= limit) {
      throw new RangeError(`op ${index} (${op.name}) writes register ${descriptor.out}, outside 0..${limit - 1}`);
    }
    const point = descriptor.point ?? 0;
    if (!pointWritten.has(point)) {
      throw new RangeError(`op ${index} (${op.name}) reads point register ${point} before anything wrote it`);
    }
    const fieldOperands = [descriptor.fieldA, descriptor.fieldB].slice(0, op.readsFields);
    fieldOperands.forEach((field, operand) => {
      if (field === undefined) {
        throw new RangeError(`op ${index} (${op.name}) reads ${op.readsFields} field operands but operand ${operand} is absent`);
      }
      if (!fieldWritten.has(field)) {
        throw new RangeError(`op ${index} (${op.name}) reads field register ${field} before anything wrote it`);
      }
    });

    const parameters = parametersOf(descriptor);
    switch (op.name) {
      case "domain-warp": {
        // A warp displaces the point by at most its amplitude per component, so
        // it grows whatever is downstream of it by that much in radius.
        budget_m[descriptor.out] = budget_m[point] - Math.sqrt(3) * Math.abs(parameters.a);
        folded[descriptor.out] = folded[point];
        break;
      }
      case "anisotropic-frame": {
        // A length in the child frame is worth up to `max(scale)` in the
        // parent's, so the budget the child inherits is the parent's divided by
        // it. Rotation is a norm-preserving map and does not enter.
        budget_m[descriptor.out] = budget_m[point] / Math.max(...frameScale(parameters));
        folded[descriptor.out] = folded[point];
        break;
      }
      case "scatter": {
        const geometry = scatterGeometry(parameters);
        if (!(geometry.clearance_m > 0)) {
          throw new RangeError(
            `op ${index} (scatter) declares an occupant of radius ${geometry.occupantRadius_m} m with ` +
              `${geometry.jitter_m} m of jitter in a ${geometry.cell_m} m cell, which leaves no clearance at the ` +
              "cell wall. A domain repetition whose occupant touches its own cell wall has no continuous " +
              "distance bound at all, and the tracer walks through the surface there",
          );
        }
        const arrayRadius_m = Math.sqrt(3) * (geometry.repetitions * geometry.cell_m + 0.5 * geometry.jitter_m) +
          geometry.occupantRadius_m;
        if (arrayRadius_m > budget_m[point]) {
          throw new RangeError(
            `op ${index} (scatter) replicates to a bounding radius of ${arrayRadius_m.toFixed(4)} m inside a ` +
              `${budget_m[point].toFixed(4)} m budget — the array does not fit the cell it is itself scattered into`,
          );
        }
        budget_m[descriptor.out] = geometry.occupantRadius_m;
        folded[descriptor.out] = true;
        break;
      }
      case "sphere":
      case "ellipsoid":
      case "box": {
        const radius_m = sourceBoundingRadius_m(op, parameters);
        if (radius_m > budget_m[point]) {
          throw new RangeError(
            `op ${index} (${op.name}) has a bounding radius of ${radius_m.toFixed(4)} m but the scatter above it ` +
              `declared an occupant of at most ${budget_m[point].toFixed(4)} m. Raise the scatter's occupant ` +
              "radius (parameter d) or shrink the source: an occupant wider than its declaration reaches past " +
              "its cell wall, and every instance of it tears along that wall",
          );
        }
        break;
      }
      case "ridged-worley-add":
      case "erosion-mask": {
        // Both read the point register for something other than a source
        // solid — a cellular lattice and an up-facing proxy — and both can
        // *lower* the field. The fold clamp squeezes an op that can only raise
        // the field back to the wall value from both sides of a cell wall, so
        // fracture and the Worley carve survive being scattered; these two are
        // not squeezed, and their discontinuity at the wall would be the whole
        // shell depth or the whole erosion depth rather than a vanishing one.
        //
        // Refused rather than approximated, because the alternative is a tear
        // in every cell of a scattered aggregate. Both belong on the assembled
        // surface anyway, which is where weathering and exposed inclusions
        // physically are.
        if (folded[point]) {
          throw new RangeError(
            `op ${index} (${op.name}) reads point register ${point}, which a scatter has folded. Apply it to the ` +
              "assembled aggregate instead: neither op is continuous across a repetition's cell walls",
          );
        }
        break;
      }
      default:
        break;
    }

    (op.result === "point" ? pointWritten : fieldWritten).add(descriptor.out);
  });
  if (!fieldWritten.has(program.result)) {
    throw new RangeError(`field program result register ${program.result} is never written`);
  }
}

// ---------------------------------------------------------------------------
// The CPU evaluator
// ---------------------------------------------------------------------------

/** `2n − 1` of the shared value noise: a signed unit displacement component. */
function signedNoise(point: Vec3, frequency_mInv: number, seed: number): number {
  const frequency = [frequency_mInv, frequency_mInv, frequency_mInv] as const;
  return f32(2 * sampleSvoProceduralNoise(point, frequency, seed) - 1);
}

function warpPoint(point: Vec3, amplitude_m: number, frequency_mInv: number, seed: number): Vec3 {
  return {
    x: f32(point.x + amplitude_m * signedNoise(point, frequency_mInv, (seed ^ WARP_SALT_X) >>> 0)),
    y: f32(point.y + amplitude_m * signedNoise(point, frequency_mInv, (seed ^ WARP_SALT_Y) >>> 0)),
    z: f32(point.z + amplitude_m * signedNoise(point, frequency_mInv, (seed ^ WARP_SALT_Z) >>> 0)),
  };
}

/**
 * Ellipsoid distance bound.
 *
 * The exact distance to an ellipsoid needs a root solve
 * (`lib/svo-implicit-reference.ts` bisects it in 128 iterations, which is the
 * ground truth, not a tracer step). This is the standard gradient-normalised
 * bound, which **under**estimates outside the solid — the direction that is
 * safe for a sphere trace. Its Lipschitz constant is declared as 1 and checked
 * by finite difference in the tests rather than asserted here.
 */
function ellipsoidDistance_m(point: Vec3, radii: readonly [number, number, number]): number {
  const rx = Math.max(1e-6, radii[0]);
  const ry = Math.max(1e-6, radii[1]);
  const rz = Math.max(1e-6, radii[2]);
  const k0 = Math.hypot(point.x / rx, point.y / ry, point.z / rz);
  const k1 = Math.hypot(point.x / (rx * rx), point.y / (ry * ry), point.z / (rz * rz));
  if (k1 <= 1e-12) return -Math.min(rx, ry, rz);
  return (k0 * (k0 - 1)) / k1;
}

function boxDistance_m(point: Vec3, half: readonly [number, number, number]): number {
  const qx = Math.abs(point.x) - half[0];
  const qy = Math.abs(point.y) - half[1];
  const qz = Math.abs(point.z) - half[2];
  const outside = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0));
  return outside + Math.min(0, Math.max(qx, qy, qz));
}

/**
 * Distance to the nearest point of a jittered cellular lattice — Worley's F1 —
 * clamped at half a cell.
 *
 * ---------------------------------------------------------------------------
 * Why eight cells and why the clamp
 * ---------------------------------------------------------------------------
 * The textbook version searches the 3×3×3 block around the sample's own cell
 * and returns the nearest feature point found. That version is **not
 * continuous**, and this is worth being precise about because a discontinuity
 * has no Lipschitz constant at all — not a large one, none — and the tracer
 * steps straight through the surface wherever one is.
 *
 * The problem is that the *set* being minimised over changes as the sample
 * crosses a lattice boundary. A feature point that was in the searched block on
 * one side drops out on the other, and if it was the minimiser the returned
 * distance jumps.
 *
 * This version searches the eight cells adjacent to the nearest lattice corner
 * — `floor(p/cell + 1/2)` — and clamps the result at half a cell. The claim is
 * that the clamp makes the truncation invisible:
 *
 *   Let the sample sit in cell `C`, so the searched x-indices are
 *   `{round(p_x/cell) − 1, round(p_x/cell)}`. Any cell excluded from that pair
 *   differs from the sample's own coordinate by at least half a cell along x —
 *   the excluded index is either below `floor(p_x/cell) − 1`, whose cell ends
 *   before `p_x − cell/2`, or above `floor(p_x/cell) + 1`, whose cell begins
 *   after `p_x + cell/2`. So **every feature point the search skips is at least
 *   `cell/2` away**, and the outer `min(cell/2, …)` overrides it.
 *
 *   The same statement evaluated on the boundary where the index set flips
 *   gives the same value from both sides, because every term that differs
 *   between the two sets is at least `cell/2` and is therefore clamped away.
 *
 * The function is therefore exactly `min(cell/2, true F1)` — continuous, and
 * 1-Lipschitz as a pointwise minimum of 1-Lipschitz distances and a constant.
 * The clamp is not a fudge factor: it is the value at which the truncated and
 * the exhaustive searches provably agree.
 *
 * Cost is 8 cells × 3 salted hashes = 24 hash evaluations per sample. The
 * 3×3×3 alternative is 81 for a clamp of `cell` instead of `cell/2`, which buys
 * nothing: a pit is a fraction of a cell across, so the field only matters well
 * inside the clamp either way.
 *
 * The shared `svoProceduralHashCell` is used rather than a local hash for the
 * same reason `domain-warp` shares the tree's value noise — one hash, mirrored
 * once and already pinned by its own tests, is the difference between two
 * evaluators that agree by construction and two that agree until somebody edits
 * one.
 */
export function sampleSvoFieldWorley_m(point: Vec3, cell_m: number, seed: number): number {
  const cell = Math.max(1e-6, cell_m);
  const qx = f32(point.x / cell);
  const qy = f32(point.y / cell);
  const qz = f32(point.z / cell);
  const baseX = Math.floor(qx + 0.5);
  const baseY = Math.floor(qy + 0.5);
  const baseZ = Math.floor(qz + 0.5);
  let nearest = 0.5;
  for (let dz = -1; dz <= 0; dz += 1) {
    for (let dy = -1; dy <= 0; dy += 1) {
      for (let dx = -1; dx <= 0; dx += 1) {
        const cellIndex: [number, number, number] = [baseX + dx, baseY + dy, baseZ + dz];
        const featureX = cellIndex[0] + svoProceduralHashCell(cellIndex, (seed ^ WORLEY_SALT_X) >>> 0);
        const featureY = cellIndex[1] + svoProceduralHashCell(cellIndex, (seed ^ WORLEY_SALT_Y) >>> 0);
        const featureZ = cellIndex[2] + svoProceduralHashCell(cellIndex, (seed ^ WORLEY_SALT_Z) >>> 0);
        nearest = Math.min(nearest, f32(Math.hypot(qx - featureX, qy - featureY, qz - featureZ)));
      }
    }
  }
  return f32(nearest * cell);
}

/**
 * The inverse of a hash-derived rotation, applied to a point.
 *
 * Built from a normalised quaternion rather than from Euler angles because the
 * two evaluators have to agree numerically and WGSL's `sin`/`cos` are permitted
 * an error orders of magnitude larger than its arithmetic. Four hashes mapped
 * to [−1, 1] give the quaternion; the `amount` blends it linearly toward the
 * identity before normalising, so zero is exactly no rotation and one is the
 * full hash-derived orientation. Nothing here is transcendental.
 *
 * The distribution over SO(3) is not perfectly uniform — it is the uniform
 * measure on a 4-cube projected to the 3-sphere, which is mildly biased toward
 * the diagonals. That is irrelevant for scattering the orientations of stones
 * and it costs no trigonometry, which is not.
 *
 * Applying the *inverse* rotation is what a point op wants: the tape evaluates
 * the occupant in its own frame, so the point has to be brought into that frame
 * rather than the solid pushed out of it.
 */
function frameInverseRotate(point: Vec3, seed: number, amount: number): Vec3 {
  const blend = Math.min(1, Math.max(0, amount));
  if (!(blend > 0)) return { ...point };
  const hash = (salt: number) => f32(2 * svoProceduralHashCell([0, 0, 0], (seed ^ salt) >>> 0) - 1);
  const ux = f32(blend * hash(FRAME_SALT_X));
  const uy = f32(blend * hash(FRAME_SALT_Y));
  const uz = f32(blend * hash(FRAME_SALT_Z));
  const w = f32(1 - blend + blend * hash(FRAME_SALT_W));
  const norm = f32(Math.sqrt(w * w + ux * ux + uy * uy + uz * uz));
  if (!(norm > 1e-6)) return { ...point };
  const nw = f32(w / norm);
  const nx = f32(ux / norm);
  const ny = f32(uy / norm);
  const nz = f32(uz / norm);
  // v − 2w(u × v) + 2u × (u × v) is the rotation by the conjugate quaternion.
  const cx = f32(ny * point.z - nz * point.y);
  const cy = f32(nz * point.x - nx * point.z);
  const cz = f32(nx * point.y - ny * point.x);
  const ccx = f32(ny * cz - nz * cy);
  const ccy = f32(nz * cx - nx * cz);
  const ccz = f32(nx * cy - ny * cx);
  return {
    x: f32(point.x - 2 * nw * cx + 2 * ccx),
    y: f32(point.y - 2 * nw * cy + 2 * ccy),
    z: f32(point.z - 2 * nw * cz + 2 * ccz),
  };
}

/**
 * The up-facing proxy the erosion mask modulates by.
 *
 * §2.2 asks for `dot(n, up)`, and the honest reason this is `p̂ · up` instead is
 * that a flat register machine cannot re-enter a subtree to take a gradient,
 * and a mask keyed on a true surface normal has a gradient that scales with
 * *curvature* — unbounded at exactly the tight arrises `fracture` exists to
 * create. That is not a bound that can be declared.
 *
 * The radial proxy is exact for a sphere centred on the record origin, which is
 * what every stone in the set is before its warp, and its gradient is bounded
 * by `2/reference` everywhere including the origin, where the `max` switches it
 * to a linear ramp instead of letting it blow up. A weathering mask does not
 * need to know the normal of a three-millimetre pit; it needs to know which end
 * of the stone is the top, and it needs to say so with a bounded gradient.
 */
function upFacingProxy(point: Vec3, reference_m: number): number {
  const reference = Math.max(1e-4, reference_m);
  return f32(point.y / Math.max(reference, Math.hypot(point.x, point.y, point.z)));
}

/**
 * Evaluate a program at a local point, sampled on a grid of the given spacing.
 *
 * This is the reference the WGSL mirror is held to, and — per H2's gate — the
 * answer the voxelizer's occupancy test and its normal tap must agree on.
 *
 * `samplingLength_m` is the voxel cell size the field is being sampled at, and
 * every op that declares a spatial period fades out as that period approaches
 * twice it. Zero means "do not band-limit": that is what the authoring-time
 * extent bound and the CPU oracles want, because they have to see the finest
 * form of the shape rather than one particular resolution's view of it.
 */
export function evaluateSvoFieldProgram(
  program: SvoFieldProgram,
  localPoint: Vec3,
  samplingLength_m = 0,
): SvoFieldValue {
  validateSvoFieldProgram(program);
  return evaluateValidatedSvoFieldProgram(program, localPoint, samplingLength_m);
}

/**
 * The same evaluation, for a caller that has already validated this exact tape.
 *
 * Validation is a forward pass over every op with its own `Set`s and arrays, and
 * it does not depend on the point — so a consumer holding one tape fixed and
 * evaluating it a million times pays for a million identical proofs. That is not
 * a micro-optimisation here: a CPU sphere-trace of the bonsai's canopy spends
 * more time proving the tape legal than evaluating it, and the interactive shape
 * lab exists precisely to evaluate one tape at every pixel.
 *
 * Validate once, against the tape object you will keep handing in, and use this.
 * Anything else — a tape built per call, a tape mutated between calls — must go
 * through {@link evaluateSvoFieldProgram}, which is the same function with the
 * proof restored.
 */
export function evaluateValidatedSvoFieldProgram(
  program: SvoFieldProgram,
  localPoint: Vec3,
  samplingLength_m = 0,
): SvoFieldValue {
  const points: Vec3[] = new Array(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(null).map(() => ({ x: 0, y: 0, z: 0 }));
  points[0] = { ...localPoint };
  // Lipschitz travels with the *point*: a warp inflates the true distance of
  // everything evaluated downstream of it, whatever that turns out to be.
  const pointLipschitz = new Array<number>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(1);
  // And so does the fold clearance, which is the value a field evaluated at
  // this point is allowed to saturate at. Unbounded until a `scatter` lowers it.
  const pointClearance_m = new Array<number>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(UNBOUNDED_DISTANCE_M);
  const fields: SvoFieldValue[] = new Array(SVO_FIELD_PROGRAM_FIELD_REGISTERS)
    .fill(null)
    .map(() => ({ distance_m: Infinity, lipschitz: 1 }));
  // The characteristic feature radius of each field, which is what makes a
  // smooth-min's blend radius vary with what it is blending (§2.2 op 4).
  const fieldRadius_m = new Array<number>(SVO_FIELD_PROGRAM_FIELD_REGISTERS).fill(0);
  // And the value each field saturates at, carried forward so that an op which
  // can only raise the field stays continuous across a fold's cell walls.
  const fieldClearance_m = new Array<number>(SVO_FIELD_PROGRAM_FIELD_REGISTERS).fill(UNBOUNDED_DISTANCE_M);

  /** Write a source solid's result, clamped so a fold above it stays continuous. */
  const writeSource = (
    out: number,
    distance_m: number,
    pointIndex: number,
    op: SvoFieldOpPolicy,
    parameters: SvoFieldOpParameters,
  ) => {
    const clearance_m = pointClearance_m[pointIndex] / sourceClearanceFactor(op, parameters);
    fields[out] = { distance_m: Math.min(distance_m, clearance_m), lipschitz: pointLipschitz[pointIndex] };
    fieldRadius_m[out] = sourceCharacteristicRadius_m(op, parameters);
    fieldClearance_m[out] = clearance_m;
  };

  for (const descriptor of program.ops) {
    const op = svoFieldOp(descriptor.op);
    const parameters = parametersOf(descriptor);
    const pointIndex = descriptor.point ?? 0;
    const point = points[pointIndex];
    const seed = descriptor.seed ?? 0;
    // One number per op, and it is a constant over the whole evaluation because
    // the sampling length is: an op whose feature period cannot survive this
    // voxel grid is mixed out of the tape, in amplitude and in bound alike.
    const fade = op.coarsestPeriod_m ? svoFieldBandLimitFade(op.coarsestPeriod_m(parameters), samplingLength_m) : 1;
    const inputA = descriptor.fieldA !== undefined ? fields[descriptor.fieldA] : undefined;
    const inputB = descriptor.fieldB !== undefined ? fields[descriptor.fieldB] : undefined;
    switch (op.name) {
      case "domain-warp": {
        const amplitude_m = parameters.a * fade;
        points[descriptor.out] = warpPoint(point, amplitude_m, parameters.b, seed);
        // A warp of displacement-gradient magnitude g inflates distance by up
        // to 1 + g (§2.4). The bound composes multiplicatively down the chain,
        // and the faded amplitude enters it, so an octave the voxel grid cannot
        // resolve costs no march steps rather than merely no shape.
        pointLipschitz[descriptor.out] =
          pointLipschitz[pointIndex] *
          (1 + Math.abs(amplitude_m) * Math.abs(parameters.b) * SVO_FIELD_NOISE_JACOBIAN_BOUND);
        pointClearance_m[descriptor.out] = pointClearance_m[pointIndex];
        break;
      }
      case "anisotropic-frame": {
        const scale = frameScale(parameters);
        const rotated = frameInverseRotate(point, seed, parameters.d);
        points[descriptor.out] = {
          x: f32(rotated.x / scale[0]),
          y: f32(rotated.y / scale[1]),
          z: f32(rotated.z / scale[2]),
        };
        // The point map is `diag(1/scale) · Rᵀ`, whose operator norm is exactly
        // `1/min(scale)` because a rotation contributes none. Squashing an axis
        // to a tenth makes every distance downstream read ten times too far.
        pointLipschitz[descriptor.out] =
          (pointLipschitz[pointIndex] * FRAME_ORTHONORMALITY_MARGIN) / Math.min(...scale);
        // A parent-frame clearance is worth at least this much in the child's.
        pointClearance_m[descriptor.out] = pointClearance_m[pointIndex] / Math.max(...scale);
        break;
      }
      case "scatter": {
        const geometry = scatterGeometry(parameters);
        const foldAxis = (coordinate: number, salt: number, index: [number, number, number]) => {
          const jitter = f32((svoProceduralHashCell(index, (seed ^ salt) >>> 0) - 0.5) * geometry.jitter_m);
          return f32(coordinate - jitter);
        };
        // The repetition index is *clamped*, which is what makes a scatter a
        // finite cluster of instances rather than an infinite lattice. An
        // infinite lattice has no bounded extent, so a record carrying one
        // could not be culled, binned or bracketed; and clamping the index is
        // continuous, because beyond the last cell the fold degenerates into a
        // plain translation and the field is the true distance to the outermost
        // instance.
        const foldIndex = (coordinate: number) =>
          Math.min(
            geometry.repetitions,
            Math.max(-geometry.repetitions, Math.floor(f32(coordinate / geometry.cell_m) + 0.5)),
          );
        const index: [number, number, number] = [foldIndex(point.x), foldIndex(point.y), foldIndex(point.z)];
        points[descriptor.out] = {
          x: foldAxis(f32(point.x - index[0] * geometry.cell_m), SCATTER_SALT_X, index),
          y: foldAxis(f32(point.y - index[1] * geometry.cell_m), SCATTER_SALT_Y, index),
          z: foldAxis(f32(point.z - index[2] * geometry.cell_m), SCATTER_SALT_Z, index),
        };
        // A fold is a translation, so it changes no distances.
        pointLipschitz[descriptor.out] = pointLipschitz[pointIndex];
        // Outside the array the clearance grows with the distance to the array
        // box, which is what stops the saturated field from throttling a ray
        // that is still metres away from the cluster. It is sound because every
        // instance sits at least `clearance` inside the box's own walls.
        const outside_m = Math.max(
          0,
          boxDistance_m(point, [geometry.arrayHalf_m, geometry.arrayHalf_m, geometry.arrayHalf_m]),
        );
        pointClearance_m[descriptor.out] = Math.min(
          pointClearance_m[pointIndex],
          geometry.clearance_m + outside_m,
        );
        break;
      }
      case "sphere": {
        writeSource(descriptor.out, Math.hypot(point.x, point.y, point.z) - parameters.a, pointIndex, op, parameters);
        break;
      }
      case "ellipsoid": {
        const distance_m = ellipsoidDistance_m(point, [parameters.a, parameters.b, parameters.c]);
        writeSource(descriptor.out, distance_m, pointIndex, op, parameters);
        break;
      }
      case "box": {
        const distance_m = boxDistance_m(point, [parameters.a, parameters.b, parameters.c]);
        writeSource(descriptor.out, distance_m, pointIndex, op, parameters);
        break;
      }
      case "fracture": {
        const base = inputA!;
        const length = Math.hypot(parameters.a, parameters.b, parameters.c);
        if (!(length > 1e-9)) {
          // A plane with no normal is not a plane. Passing the field through is
          // the only decode that cannot invent a cut nobody authored.
          fields[descriptor.out] = base;
          fieldRadius_m[descriptor.out] = fieldRadius_m[descriptor.fieldA!];
          fieldClearance_m[descriptor.out] = fieldClearance_m[descriptor.fieldA!];
          break;
        }
        const halfspace_m =
          (parameters.a * point.x + parameters.b * point.y + parameters.c * point.z) / length - parameters.d;
        const clearance_m = fieldClearance_m[descriptor.fieldA!];
        // `smax` can only raise the field, so clamping the result back at the
        // operand's own saturation value is what lets a fracture plane cut a
        // *scattered* occupant: on a cell wall the operand has already
        // saturated, the result is the same saturated value from both sides,
        // and the plane's own jump across the wall is squeezed out.
        fields[descriptor.out] = {
          distance_m: Math.min(smoothMaximum(base.distance_m, halfspace_m, Math.max(0, parameters.e)), clearance_m),
          lipschitz: Math.max(base.lipschitz, pointLipschitz[pointIndex]),
        };
        fieldRadius_m[descriptor.out] = fieldRadius_m[descriptor.fieldA!];
        fieldClearance_m[descriptor.out] = clearance_m;
        break;
      }
      case "smooth-union":
      case "smooth-subtract":
      case "smooth-intersect": {
        const a = inputA!;
        const b = inputB!;
        const radiusA = fieldRadius_m[descriptor.fieldA!];
        const radiusB = fieldRadius_m[descriptor.fieldB!];
        const k_m = svoFieldBlendRadius_m(parameters, radiusA, radiusB);
        // §2.4's rule, and it is exact rather than conservative: the gradient of
        // this smooth-min form is a convex combination of its operands', so the
        // blended field is no steeper than the steeper of the two.
        const lipschitz = Math.max(a.lipschitz, b.lipschitz);
        if (op.name === "smooth-union") {
          fields[descriptor.out] = { distance_m: smoothMinimum(a.distance_m, b.distance_m, k_m), lipschitz };
          fieldRadius_m[descriptor.out] = Math.max(radiusA, radiusB);
          fieldClearance_m[descriptor.out] = Math.min(
            fieldClearance_m[descriptor.fieldA!],
            fieldClearance_m[descriptor.fieldB!],
          );
        } else if (op.name === "smooth-subtract") {
          fields[descriptor.out] = { distance_m: smoothMaximum(a.distance_m, -b.distance_m, k_m), lipschitz };
          fieldRadius_m[descriptor.out] = radiusA;
          fieldClearance_m[descriptor.out] = UNBOUNDED_DISTANCE_M;
        } else {
          fields[descriptor.out] = { distance_m: smoothMaximum(a.distance_m, b.distance_m, k_m), lipschitz };
          fieldRadius_m[descriptor.out] = Math.min(radiusA, radiusB);
          fieldClearance_m[descriptor.out] = UNBOUNDED_DISTANCE_M;
        }
        break;
      }
      case "worley-subtract": {
        const base = inputA!;
        const clearance_m = fieldClearance_m[descriptor.fieldA!];
        // The union of balls of radius `b` on the cellular lattice has the exact
        // signed distance `F1 − b`; subtracting it from the solid is
        // `max(d, b − F1)`. Pits, not bumps — the plate's grain band is a pocked
        // surface, and additive noise on the same statistics reads as plaster.
        const carved_m = smoothMaximum(
          base.distance_m,
          Math.max(0, parameters.b) - sampleSvoFieldWorley_m(point, parameters.a, seed),
          Math.max(0, parameters.c),
        );
        // Faded by mixing toward the untouched field rather than by shrinking
        // the pits alone, so that the bound below is the same convex
        // combination and a faded-out grain costs nothing to march through.
        const distance_m = base.distance_m + fade * (Math.min(carved_m, clearance_m) - base.distance_m);
        fields[descriptor.out] = {
          distance_m,
          lipschitz:
            base.lipschitz +
            fade * (Math.max(base.lipschitz, pointLipschitz[pointIndex] * SVO_FIELD_WORLEY_JACOBIAN_BOUND) - base.lipschitz),
        };
        fieldRadius_m[descriptor.out] = fieldRadius_m[descriptor.fieldA!];
        fieldClearance_m[descriptor.out] = clearance_m;
        break;
      }
      case "ridged-worley-add": {
        const base = inputA!;
        const shell_m = Math.max(0, parameters.d);
        // A union of lattice balls is solid *everywhere in space*, which has no
        // bounded extent at all and would fill the scene. Gating it against
        // `base − shell` restricts the inclusions to a shell of the surface,
        // which is both where a proud nodule in a receded matrix physically is
        // and the only form of this op whose extent can be bounded.
        //
        // The gate is a plain `max`, which is exact for the solid and free in
        // Lipschitz terms — a max of two fields is no steeper than the steeper
        // of them. What it is not is tight far from the surface: out there the
        // gate is what the union sees, so the field reads a constant `shell`
        // short of the truth. That is an under-estimate, the safe direction for
        // a sphere trace, and it costs a few millimetres of step on a march the
        // proxy box has already bracketed.
        //
        // The alternative — fading the whole op out on a smoothstep of the base
        // field — removes the offset and costs about 4.6x in the declared
        // Lipschitz constant, because the fade's own gradient scales with
        // `1/shell`. Measured, and pinned by a test, so the offset is a known
        // quantity rather than a surprise in a march profile.
        const inclusion_m = Math.max(
          sampleSvoFieldWorley_m(point, parameters.a, seed) - Math.max(0, parameters.b),
          base.distance_m - shell_m,
        );
        const joined_m = smoothMinimum(base.distance_m, inclusion_m, Math.max(0, parameters.c));
        fields[descriptor.out] = {
          distance_m: base.distance_m + fade * (joined_m - base.distance_m),
          lipschitz:
            base.lipschitz +
            fade * (Math.max(base.lipschitz, pointLipschitz[pointIndex] * SVO_FIELD_WORLEY_JACOBIAN_BOUND) - base.lipschitz),
        };
        fieldRadius_m[descriptor.out] = fieldRadius_m[descriptor.fieldA!];
        fieldClearance_m[descriptor.out] = fieldClearance_m[descriptor.fieldA!];
        break;
      }
      case "erosion-mask": {
        const base = inputA!;
        const eroded = inputB!;
        const depth_m = Math.max(0, parameters.c);
        // The modifier may only *remove*, which is what "erosion" means and
        // also what keeps the extent bound trivial: the masked result is never
        // below the unmasked base, so the solid can only shrink.
        const removal_m = Math.min(depth_m, Math.max(0, eroded.distance_m - base.distance_m));
        const cavity = depth_m > 0 ? (2 * removal_m) / depth_m - 1 : -1;
        const facing = upFacingProxy(point, parameters.d);
        const weight = Math.min(
          1,
          Math.max(0, 0.5 + 0.5 * (Math.max(0, parameters.a) * facing + Math.max(0, parameters.b) * cavity)),
        );
        const operandLipschitz = base.lipschitz + eroded.lipschitz;
        // `∇(A + m·e) = ∇A + m∇e + e∇m`, term by term: the base, the clamped
        // difference at weight at most one, and the mask's own gradient scaled
        // by a removal of at most `depth`. The up proxy's gradient is bounded by
        // `2/reference` and the cavity term's by `2·(L_A + L_B)/depth`, and the
        // depth cancels out of the second — which is why a deeper erosion does
        // not cost proportionally more march steps.
        const referenceRadius_m = Math.max(1e-4, parameters.d);
        fields[descriptor.out] = {
          distance_m: base.distance_m + weight * removal_m,
          lipschitz:
            base.lipschitz +
            operandLipschitz * (1 + Math.max(0, parameters.b)) +
            (depth_m * Math.max(0, parameters.a) * pointLipschitz[pointIndex]) / referenceRadius_m,
        };
        fieldRadius_m[descriptor.out] = fieldRadius_m[descriptor.fieldA!];
        fieldClearance_m[descriptor.out] = fieldClearance_m[descriptor.fieldA!];
        break;
      }
      default: {
        // Unreachable while the table and this switch agree; the table is the
        // authority and an op it declares but this cannot run is a build error
        // waiting in `tests/svo-field-program.test.ts`, not a silent miss.
        throw new RangeError(`field op ${op.name} has no evaluator arm`);
      }
    }
  }
  return fields[program.result];
}

// ---------------------------------------------------------------------------
// Conservative extent
// ---------------------------------------------------------------------------

/**
 * A local half-extent that provably contains the program's zero set.
 *
 * Load-bearing rather than advisory: the proxy box drives brick binning and the
 * march bracket, and a box that clipped the warped surface would delete the
 * parts of the rock the warp exists to create — silently, and worst at exactly
 * the silhouettes the eye reads first.
 *
 * ---------------------------------------------------------------------------
 * How the bound is built
 * ---------------------------------------------------------------------------
 * A **per-field-register** box, propagated forward with the tape, rather than
 * one box that takes the maximum over every source op. The distinction only
 * started to matter with the combining ops: `ridged-worley-add` grows a field
 * by its shell depth and `smooth-union` by a quarter of its blend radius, and
 * attributing those to the right register is the difference between a proxy
 * that covers the shape and one that covers a shape the tape never assembled.
 *
 * Two quantities travel alongside the box.
 *
 *   - **Dilation.** `{d ≤ τ} ⊆ box ⊕ dilation·τ`, which is what any op that can
 *     *lower* a field needs in order to say how far the zero set can move. It
 *     is one for the exact sources, the squared radius ratio for the
 *     ellipsoid's gradient-normalised bound, and it picks up the frame's scale.
 *   - **Saturation.** A folded field stops growing at its cell clearance, so
 *     `{d ≤ τ}` is the whole of space for any `τ` past it. An op that asked for
 *     a dilation past a saturation is refused rather than answered, because the
 *     honest answer is "unbounded" and an unbounded proxy box is not a box.
 *
 * The band limit is deliberately **not** applied. A record's extent is authored
 * once and has to contain the shape at every voxel resolution it will ever be
 * sampled at, and the unfaded field is the largest of those. A proxy that
 * shrank with the sampling length would clip the stone the moment somebody
 * voxelized it finer.
 *
 * The result is not tight — that wants the interval arithmetic of §2.5 — but it
 * never under-covers, and an untight proxy costs traversal while a loose
 * surface costs the image.
 */
export function svoFieldProgramExtent_m(program: SvoFieldProgram): [number, number, number] {
  validateSvoFieldProgram(program);
  const axes = [0, 1, 2] as const;
  // Per point register: how a length in this register's frame converts to
  // parent metres, how much the chain has already displaced things, and whether
  // a rotation has made the per-axis accounting meaningless.
  const scale = new Array<[number, number, number]>(SVO_FIELD_PROGRAM_POINT_REGISTERS)
    .fill([1, 1, 1])
    .map((value) => [...value] as [number, number, number]);
  const growth_m = new Array<[number, number, number]>(SVO_FIELD_PROGRAM_POINT_REGISTERS)
    .fill([0, 0, 0])
    .map((value) => [...value] as [number, number, number]);
  const rotated = new Array<boolean>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(false);
  const pointClearance_m = new Array<number>(SVO_FIELD_PROGRAM_POINT_REGISTERS).fill(Infinity);

  const extent_m = new Array<[number, number, number]>(SVO_FIELD_PROGRAM_FIELD_REGISTERS)
    .fill([0, 0, 0])
    .map((value) => [...value] as [number, number, number]);
  const dilation = new Array<number>(SVO_FIELD_PROGRAM_FIELD_REGISTERS).fill(1);
  const fieldClearance_m = new Array<number>(SVO_FIELD_PROGRAM_FIELD_REGISTERS).fill(Infinity);

  // The same characteristic radius the evaluator carries, propagated by the
  // same rules, because the blend radius it feeds is what the box has to bound.
  const radius_m = new Array<number>(SVO_FIELD_PROGRAM_FIELD_REGISTERS).fill(0);

  /** The box that contains `{d ≤ τ}` for a field, in parent metres. */
  const dilated_m = (register: number, tolerance_m: number): [number, number, number] => {
    if (tolerance_m >= fieldClearance_m[register]) {
      throw new RangeError(
        `field register ${register} saturates at ${fieldClearance_m[register].toFixed(4)} m but an op asks the ` +
          `extent bound to dilate it by ${tolerance_m.toFixed(4)} m. Past its saturation the field's sub-level ` +
          "set is the whole of space, so there is no proxy box to compute — shrink the blend radius or the " +
          "shell depth, or take the op off the folded operand",
      );
    }
    return axes.map((axis) => extent_m[register][axis] + dilation[register] * tolerance_m) as [
      number,
      number,
      number,
    ];
  };

  for (const descriptor of program.ops) {
    const op = svoFieldOp(descriptor.op);
    const parameters = parametersOf(descriptor);
    const pointIndex = descriptor.point ?? 0;
    const out = descriptor.out;
    const a = descriptor.fieldA ?? 0;
    const b = descriptor.fieldB ?? 0;
    switch (op.name) {
      case "domain-warp": {
        // A warp displaces the evaluation point by at most its amplitude per
        // component, in the frame it is applied in.
        for (const axis of axes) {
          growth_m[out][axis] = growth_m[pointIndex][axis] + scale[pointIndex][axis] * Math.abs(parameters.a);
          scale[out][axis] = scale[pointIndex][axis];
        }
        rotated[out] = rotated[pointIndex];
        pointClearance_m[out] = pointClearance_m[pointIndex];
        break;
      }
      case "anisotropic-frame": {
        const frame = frameScale(parameters);
        const turns = Math.min(1, Math.max(0, parameters.d)) > 0;
        if (turns) {
          // A rotation mixes the axes, so the child's per-axis half-extents stop
          // being per-axis in the parent and only its bounding radius survives.
          // Recorded as a flag rather than silently inflating every axis by √3,
          // which would loosen the common unrotated case for nothing.
          const uniform = Math.max(...axes.map((axis) => scale[pointIndex][axis] * frame[axis]));
          for (const axis of axes) scale[out][axis] = uniform;
        } else {
          for (const axis of axes) scale[out][axis] = scale[pointIndex][axis] * frame[axis];
        }
        for (const axis of axes) growth_m[out][axis] = growth_m[pointIndex][axis];
        rotated[out] = rotated[pointIndex] || turns;
        pointClearance_m[out] = pointClearance_m[pointIndex] / Math.max(...frame);
        break;
      }
      case "scatter": {
        const geometry = scatterGeometry(parameters);
        // Instances reach `repetitions` cells out plus half the jitter, per
        // axis. A lattice is exactly per-axis, so nothing here needs isotropy.
        const reach_m = geometry.repetitions * geometry.cell_m + 0.5 * geometry.jitter_m;
        for (const axis of axes) {
          growth_m[out][axis] = growth_m[pointIndex][axis] + scale[pointIndex][axis] * reach_m;
          scale[out][axis] = scale[pointIndex][axis];
        }
        rotated[out] = rotated[pointIndex];
        pointClearance_m[out] = Math.min(pointClearance_m[pointIndex], geometry.clearance_m);
        break;
      }
      case "sphere":
      case "ellipsoid":
      case "box": {
        const radii: [number, number, number] =
          op.name === "sphere"
            ? [Math.abs(parameters.a), Math.abs(parameters.a), Math.abs(parameters.a)]
            : [Math.abs(parameters.a), Math.abs(parameters.b), Math.abs(parameters.c)];
        const bounding_m = sourceBoundingRadius_m(op, parameters);
        for (const axis of axes) {
          // Under a rotation neither half of this is per-axis any more. The
          // solid becomes its own bounding radius, and the accumulated
          // displacement box becomes the radius of *that* box, which is √3 of
          // its half-extent — a cube turned 45° in two planes reaches its
          // corner along every axis.
          extent_m[out][axis] = rotated[pointIndex]
            ? scale[pointIndex][axis] * bounding_m + Math.sqrt(3) * growth_m[pointIndex][axis]
            : scale[pointIndex][axis] * radii[axis] + growth_m[pointIndex][axis];
        }
        dilation[out] = Math.max(...scale[pointIndex]) * sourceClearanceFactor(op, parameters);
        fieldClearance_m[out] = pointClearance_m[pointIndex] / sourceClearanceFactor(op, parameters);
        radius_m[out] = sourceCharacteristicRadius_m(op, parameters);
        break;
      }
      case "fracture":
      case "worley-subtract":
      case "erosion-mask": {
        // All three can only raise the field — a halfspace intersection, a
        // carve, a masked removal — so the solid only shrinks and the operand's
        // own box still contains it.
        for (const axis of axes) extent_m[out][axis] = extent_m[a][axis];
        dilation[out] = dilation[a];
        fieldClearance_m[out] = fieldClearance_m[a];
        radius_m[out] = radius_m[a];
        break;
      }
      case "smooth-union": {
        // `smin` dips below the hard minimum by at most a quarter of the blend
        // radius, and either operand's zero set can move outward by that much.
        const tolerance_m = smoothOvershoot_m(svoFieldBlendRadius_m(parameters, radius_m[a], radius_m[b]));
        const first = dilated_m(a, tolerance_m);
        const second = dilated_m(b, tolerance_m);
        for (const axis of axes) extent_m[out][axis] = Math.max(first[axis], second[axis]);
        dilation[out] = Math.max(dilation[a], dilation[b]);
        fieldClearance_m[out] = Math.min(fieldClearance_m[a], fieldClearance_m[b]);
        radius_m[out] = Math.max(radius_m[a], radius_m[b]);
        break;
      }
      case "smooth-subtract": {
        for (const axis of axes) extent_m[out][axis] = extent_m[a][axis];
        dilation[out] = dilation[a];
        fieldClearance_m[out] = Infinity;
        radius_m[out] = radius_m[a];
        break;
      }
      case "smooth-intersect": {
        // The intersection is inside *both* operands, so either box would do,
        // and the smaller one is tempting. It is not taken: the box has to hold
        // for every dilation, and `min(box) + min(dilation)·τ` can fall inside
        // `min(box + dilation·τ)` when the tighter box has the steeper field.
        // Keeping the first operand's pair keeps the two consistent.
        for (const axis of axes) extent_m[out][axis] = extent_m[a][axis];
        dilation[out] = dilation[a];
        fieldClearance_m[out] = Infinity;
        radius_m[out] = Math.min(radius_m[a], radius_m[b]);
        break;
      }
      case "ridged-worley-add": {
        // The only op here that adds material. Inclusions live inside the shell
        // it gates against, and the smooth union can dip a further quarter
        // blend radius below that.
        const tolerance_m = Math.max(0, parameters.d) + smoothOvershoot_m(Math.max(0, parameters.c));
        extent_m[out] = dilated_m(a, tolerance_m);
        dilation[out] = dilation[a];
        fieldClearance_m[out] = fieldClearance_m[a];
        radius_m[out] = radius_m[a];
        break;
      }
      default:
        throw new RangeError(`field op ${op.name} has no extent arm`);
    }
  }
  return extent_m[program.result];
}

/**
 * The smallest solid feature the tape draws, as a radius, in metres.
 *
 * `svoClusterFeatureRadius_m`'s counterpart one level up, and it answers the
 * same two questions: is this record thin-featured enough that cell-centre
 * voxelization could drop it, and how far apart should a gradient's taps be.
 *
 * A tape says so directly rather than through an envelope. Every op that
 * introduces a spatial period declares it — `coarsestPeriod_m` is the same
 * function the band limit fades against — so the finest of those halved is the
 * finest lobe the tape can carry. A tape that declares no period at all is a
 * plain composition of source solids, and then the smallest source's own
 * characteristic radius is the answer.
 *
 * Deliberately **not** band-limited, exactly as the extent bound is not: this is
 * a property of the authored shape, and one that shrank with the sampling length
 * would call a stone thick at a coarse voxel and thin at a fine one.
 */
export function svoFieldProgramFeatureRadius_m(program: SvoFieldProgram): number {
  validateSvoFieldProgram(program);
  let finest_m = Infinity;
  for (const descriptor of program.ops) {
    const op = svoFieldOp(descriptor.op);
    const parameters = parametersOf(descriptor);
    if (op.coarsestPeriod_m) {
      const period_m = op.coarsestPeriod_m(parameters);
      // A lobe of a lattice with period `p` is at most `p/2` across before its
      // neighbours meet it, so half the period is the radius bound. `scatter`
      // is included: a replicated occupant is at most half a cell.
      if (Number.isFinite(period_m)) finest_m = Math.min(finest_m, 0.5 * period_m);
      continue;
    }
    if (op.readsFields === 0 && op.result === "field") {
      finest_m = Math.min(finest_m, sourceCharacteristicRadius_m(op, parameters));
    }
  }
  // A tape with neither a period nor a source is not a shape the validator would
  // have accepted, so the fallback is unreachable rather than a default.
  return Number.isFinite(finest_m) ? Math.max(finest_m, 0) : 0;
}

// ---------------------------------------------------------------------------
// Arena packing
// ---------------------------------------------------------------------------

/**
 * Pack tapes into a dense arena; block `index` starts at `index * SVO_FIELD_PROGRAM_BLOCK_WORDS`.
 *
 * Shaped after `packSvoClusterArena`, including the capacity guard: the record
 * carries a reference into this arena and never the tape itself, because the
 * 64-byte record stride is shared with BVH nodes and widening it is refused
 * outright by §5 of the handoff.
 */
export function packSvoFieldProgramArena(
  programs: readonly SvoFieldProgram[],
  capacity = programs.length,
): Uint32Array<ArrayBuffer> {
  if (!Number.isSafeInteger(capacity) || capacity < programs.length || capacity < 0) {
    throw new RangeError("Field program arena capacity must hold every program");
  }
  const buffer = new ArrayBuffer(Math.max(1, capacity) * SVO_FIELD_PROGRAM_BLOCK_WORDS * 4);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  programs.forEach((program, index) => {
    validateSvoFieldProgram(program);
    const base = index * SVO_FIELD_PROGRAM_BLOCK_WORDS;
    words[base] = program.ops.length >>> 0;
    words[base + 1] = program.result >>> 0;
    program.ops.forEach((descriptor, opIndex) => {
      const op = svoFieldOp(descriptor.op);
      const parameters = parametersOf(descriptor);
      const at = base + SVO_FIELD_PROGRAM_HEADER_WORDS + opIndex * SVO_FIELD_PROGRAM_OP_WORDS;
      words[at] = op.code >>> 0;
      // Operands in one word: point in the low byte, then the two field reads.
      const operand = (value: number | undefined, index: number) =>
        (index < op.readsFields && value !== undefined ? value : SVO_FIELD_PROGRAM_UNUSED_OPERAND) & 0xff;
      words[at + 1] =
        ((descriptor.point ?? 0) & 0xff) |
        (operand(descriptor.fieldA, 0) << 8) |
        (operand(descriptor.fieldB, 1) << 16) |
        ((descriptor.out & 0xff) << 24);
      words[at + 2] = (descriptor.seed ?? 0) >>> 0;
      floats[at + 3] = parameters.e;
      floats[at + 4] = parameters.a;
      floats[at + 5] = parameters.b;
      floats[at + 6] = parameters.c;
      floats[at + 7] = parameters.d;
    });
  });
  return words;
}

/** Read a packed block back. The inverse of `packSvoFieldProgramArena`, for tests and tools. */
export function unpackSvoFieldProgram(words: Uint32Array, index: number): SvoFieldProgram {
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const base = index * SVO_FIELD_PROGRAM_BLOCK_WORDS;
  const count = words[base];
  if (count > SVO_FIELD_PROGRAM_MAXIMUM_OPS) {
    throw new RangeError(`packed field program at ${index} claims ${count} ops`);
  }
  const ops: SvoFieldOpDescriptor[] = [];
  for (let opIndex = 0; opIndex < count; opIndex += 1) {
    const at = base + SVO_FIELD_PROGRAM_HEADER_WORDS + opIndex * SVO_FIELD_PROGRAM_OP_WORDS;
    const op = opByCode.get(words[at]);
    if (!op) throw new RangeError(`packed field program at ${index} uses unknown op code ${words[at]}`);
    const operands = words[at + 1];
    const operandAt = (shift: number) => {
      const value = (operands >>> shift) & 0xff;
      return value === SVO_FIELD_PROGRAM_UNUSED_OPERAND ? undefined : value;
    };
    ops.push({
      op: op.name as SvoFieldOpName,
      point: operands & 0xff,
      fieldA: operandAt(8),
      fieldB: operandAt(16),
      out: (operands >>> 24) & 0xff,
      seed: words[at + 2],
      parameters: {
        a: floats[at + 4],
        b: floats[at + 5],
        c: floats[at + 6],
        d: floats[at + 7],
        e: floats[at + 3],
      },
    });
  }
  return { ops, result: words[base + 1] };
}

// ---------------------------------------------------------------------------
// The WGSL mirror
// ---------------------------------------------------------------------------

const wgslOpConstants = SVO_FIELD_OP_TABLE.map((op) => `const ${op.wgslConstant}:u32=${op.code}u;`).join("\n");

/**
 * WGSL evaluator, generated from the same table the CPU evaluator reads.
 *
 * Requires `svoProceduralNoise` **and** `svoProceduralHashCell` to already be in
 * the module — this deliberately does not carry its own copy of either. One
 * hash and one noise, mirrored once and already pinned by their own tests, is
 * the difference between two evaluators that agree by construction and two that
 * agree until somebody edits one. Both come together in `svoProceduralNoiseWGSL`
 * and in the material WGSL that includes it, so any module that already had the
 * warp has the cellular ops as well.
 *
 * Note the two entry points: `<functionName>Banded` takes the voxel cell size
 * the field is about to be resampled onto and applies the band limit; the plain
 * `<functionName>` forwards to it with zero, which is the unbanded field. The
 * voxelizer wants the first, and it is a one-argument change at the call site
 * to move over.
 *
 * The caller supplies `loadWord`, as `svoClusterArenaDecodeWGSL` does, because
 * the live voxelizer keeps its arenas inside a shared maintenance buffer and
 * holds itself to four storage bindings.
 */
export interface SvoFieldProgramWGSLOptions {
  /**
   * Name of the generated `fn(blockIndex:u32, localPoint:vec3f)->SvoFieldValue`.
   *
   * Two entry points are emitted under this name. The plain one takes no
   * sampling length and evaluates the field unbanded, which is what every
   * existing call site already spells and what a CPU-oracle comparison wants.
   * `<name>Banded` takes the voxel cell size as a third argument and is the one
   * the voxelizer should call, because that is the only place that knows the
   * grid the field is about to be resampled onto.
   */
  readonly functionName: string;
  /** An existing `fn(word:u32)->u32` over whatever buffer holds the arena. */
  readonly loadWord: string;
  /** WGSL expression for the arena's first word. */
  readonly baseWordExpression: string;
  /** WGSL expression for the number of blocks the arena holds. */
  readonly capacityExpression: string;
}

export function svoFieldProgramWGSL(options: SvoFieldProgramWGSLOptions): string {
  const { functionName, loadWord, baseWordExpression, capacityExpression } = options;
  return /* wgsl */ `
${wgslOpConstants}
const SVO_FIELD_PROGRAM_BLOCK_WORDS:u32=${SVO_FIELD_PROGRAM_BLOCK_WORDS}u;
const SVO_FIELD_PROGRAM_HEADER_WORDS:u32=${SVO_FIELD_PROGRAM_HEADER_WORDS}u;
const SVO_FIELD_PROGRAM_OP_WORDS:u32=${SVO_FIELD_PROGRAM_OP_WORDS}u;
const SVO_FIELD_NOISE_JACOBIAN_BOUND:f32=${SVO_FIELD_NOISE_JACOBIAN_BOUND.toFixed(8)};
const SVO_FIELD_WORLEY_JACOBIAN_BOUND:f32=${SVO_FIELD_WORLEY_JACOBIAN_BOUND.toFixed(8)};
const SVO_FIELD_BAND_LIMIT_SAMPLES:f32=${SVO_FIELD_BAND_LIMIT_SAMPLES.toFixed(8)};
const SVO_FIELD_FRAME_ORTHONORMALITY_MARGIN:f32=${FRAME_ORTHONORMALITY_MARGIN.toFixed(8)};
const SVO_FIELD_UNBOUNDED_DISTANCE_M:f32=${UNBOUNDED_DISTANCE_M.toExponential()};

// A distance and the Lipschitz constant of the field that produced it. The
// tracer steps distance/lipschitz; see §2.4 of the fidelity handoff.
struct SvoFieldValue{distance_m:f32,lipschitz:f32}

fn svoFieldSignedNoise(point:vec3f,frequency_mInv:f32,seed:u32)->f32{
  return 2.0*svoProceduralNoise(point,vec3f(frequency_mInv),seed)-1.0;
}

fn svoFieldWarpPoint(point:vec3f,amplitude_m:f32,frequency_mInv:f32,seed:u32)->vec3f{
  return point+amplitude_m*vec3f(
    svoFieldSignedNoise(point,frequency_mInv,seed^${WARP_SALT_X}u),
    svoFieldSignedNoise(point,frequency_mInv,seed^${WARP_SALT_Y}u),
    svoFieldSignedNoise(point,frequency_mInv,seed^${WARP_SALT_Z}u));
}

fn svoFieldEllipsoid_m(point:vec3f,radii:vec3f)->f32{
  let r=max(radii,vec3f(1e-6));
  let k0=length(point/r);
  let k1=length(point/(r*r));
  if(k1<=1e-12){return -min(r.x,min(r.y,r.z));}
  return k0*(k0-1.0)/k1;
}

fn svoFieldBox_m(point:vec3f,half:vec3f)->f32{
  let q=abs(point)-half;
  return length(max(q,vec3f(0.0)))+min(0.0,max(q.x,max(q.y,q.z)));
}

// The quadratic smooth min/max in the form whose gradient is a convex
// combination of its operands' — the reason §2.4 can say "smooth-min takes
// max(L)" and mean it exactly rather than conservatively.
fn svoFieldSmoothMinimum(a:f32,b:f32,k:f32)->f32{
  if(k>0.0){let h=clamp(0.5+0.5*(b-a)/k,0.0,1.0);return b+(a-b)*h-k*h*(1.0-h);}
  return min(a,b);
}
fn svoFieldSmoothMaximum(a:f32,b:f32,k:f32)->f32{
  if(k>0.0){let h=clamp(0.5+0.5*(a-b)/k,0.0,1.0);return b+(a-b)*h+k*h*(1.0-h);}
  return max(a,b);
}

// The band limit: one when the period is at least four sampling lengths, zero
// at two, the tree's own smoothstep between. Constant over an evaluation.
fn svoFieldBandLimitFade(period_m:f32,samplingLength_m:f32)->f32{
  if(!(samplingLength_m>0.0)||!(period_m>0.0)){return 1.0;}
  let t=clamp(period_m/(SVO_FIELD_BAND_LIMIT_SAMPLES*samplingLength_m)-1.0,0.0,1.0);
  return t*t*(3.0-2.0*t);
}

// Worley F1 over the eight cells around the nearest lattice corner, clamped at
// half a cell. Every feature point the search skips is at least that far away,
// so the clamp makes the truncation exact and the field continuous; see the
// derivation on sampleSvoFieldWorley_m.
fn svoFieldWorley_m(point:vec3f,cell_m:f32,seed:u32)->f32{
  let cell=max(cell_m,1e-6);
  let q=point/cell;
  let base=vec3i(floor(q+vec3f(0.5)));
  var nearest=0.5;
  for(var dz:i32=-1;dz<=0;dz+=1){
    for(var dy:i32=-1;dy<=0;dy+=1){
      for(var dx:i32=-1;dx<=0;dx+=1){
        let cellIndex=base+vec3i(dx,dy,dz);
        let feature=vec3f(cellIndex)+vec3f(
          svoProceduralHashCell(cellIndex,seed^${WORLEY_SALT_X}u),
          svoProceduralHashCell(cellIndex,seed^${WORLEY_SALT_Y}u),
          svoProceduralHashCell(cellIndex,seed^${WORLEY_SALT_Z}u));
        nearest=min(nearest,length(q-feature));
      }
    }
  }
  return nearest*cell;
}

// The inverse of a hash-derived rotation. Quaternion rather than Euler angles
// because WGSL's sin/cos are permitted an error orders of magnitude larger than
// its arithmetic, and this has to match the CPU evaluator.
fn svoFieldFrameRotateInverse(point:vec3f,seed:u32,amount:f32)->vec3f{
  let blend=clamp(amount,0.0,1.0);
  if(!(blend>0.0)){return point;}
  let u=blend*vec3f(
    2.0*svoProceduralHashCell(vec3i(0,0,0),seed^${FRAME_SALT_X}u)-1.0,
    2.0*svoProceduralHashCell(vec3i(0,0,0),seed^${FRAME_SALT_Y}u)-1.0,
    2.0*svoProceduralHashCell(vec3i(0,0,0),seed^${FRAME_SALT_Z}u)-1.0);
  let w=1.0-blend+blend*(2.0*svoProceduralHashCell(vec3i(0,0,0),seed^${FRAME_SALT_W}u)-1.0);
  let norm=sqrt(w*w+dot(u,u));
  if(!(norm>1e-6)){return point;}
  let nw=w/norm;
  let nu=u/norm;
  let first=cross(nu,point);
  return point-2.0*nw*first+2.0*cross(nu,first);
}

// p̂·up rather than n·up, with a bounded gradient everywhere; see upFacingProxy.
fn svoFieldUpFacing(point:vec3f,reference_m:f32)->f32{
  let reference=max(reference_m,1e-4);
  return point.y/max(reference,length(point));
}

// A block past capacity evaluates as a miss, never as the envelope — the same
// refusal svoClusterArenaDecodeWGSL makes, for the same reason: a tape that
// quietly became an ellipsoid is a shape nobody authored.
fn ${functionName}Banded(blockIndex:u32,localPoint:vec3f,samplingLength_m:f32)->SvoFieldValue{
  if(blockIndex>=(${capacityExpression})){return SvoFieldValue(1e20,1.0);}
  let base=(${baseWordExpression})+blockIndex*SVO_FIELD_PROGRAM_BLOCK_WORDS;
  let opCount=min(${loadWord}(base),${SVO_FIELD_PROGRAM_MAXIMUM_OPS}u);
  let resultRegister=${loadWord}(base+1u);

  var points=array<vec3f,${SVO_FIELD_PROGRAM_POINT_REGISTERS}>();
  var pointLipschitz=array<f32,${SVO_FIELD_PROGRAM_POINT_REGISTERS}>();
  var pointClearance=array<f32,${SVO_FIELD_PROGRAM_POINT_REGISTERS}>();
  var fieldDistance=array<f32,${SVO_FIELD_PROGRAM_FIELD_REGISTERS}>();
  var fieldLipschitz=array<f32,${SVO_FIELD_PROGRAM_FIELD_REGISTERS}>();
  var fieldRadius=array<f32,${SVO_FIELD_PROGRAM_FIELD_REGISTERS}>();
  var fieldClearance=array<f32,${SVO_FIELD_PROGRAM_FIELD_REGISTERS}>();
  points[0]=localPoint;
  for(var index=0u;index<${SVO_FIELD_PROGRAM_POINT_REGISTERS}u;index+=1u){pointLipschitz[index]=1.0;pointClearance[index]=SVO_FIELD_UNBOUNDED_DISTANCE_M;}
  for(var index=0u;index<${SVO_FIELD_PROGRAM_FIELD_REGISTERS}u;index+=1u){fieldDistance[index]=1e20;fieldLipschitz[index]=1.0;fieldRadius[index]=0.0;fieldClearance[index]=SVO_FIELD_UNBOUNDED_DISTANCE_M;}

  for(var opIndex=0u;opIndex<opCount;opIndex+=1u){
    let at=base+SVO_FIELD_PROGRAM_HEADER_WORDS+opIndex*SVO_FIELD_PROGRAM_OP_WORDS;
    let code=${loadWord}(at);
    let operands=${loadWord}(at+1u);
    let seed=${loadWord}(at+2u);
    let pointIndex=operands&0xffu;
    let fieldA=(operands>>8u)&0xffu;
    let fieldB=(operands>>16u)&0xffu;
    let out=(operands>>24u)&0xffu;
    let e=bitcast<f32>(${loadWord}(at+3u));
    let a=bitcast<f32>(${loadWord}(at+4u));
    let b=bitcast<f32>(${loadWord}(at+5u));
    let c=bitcast<f32>(${loadWord}(at+6u));
    let d=bitcast<f32>(${loadWord}(at+7u));
    let point=points[pointIndex];
    let carriedLipschitz=pointLipschitz[pointIndex];
    let carriedClearance=pointClearance[pointIndex];
    if(code==${svoFieldOp("domain-warp").wgslConstant}){
      let fade=svoFieldBandLimitFade(select(1e20,1.0/b,b>0.0),samplingLength_m);
      let amplitude=a*fade;
      points[out]=svoFieldWarpPoint(point,amplitude,b,seed);
      pointLipschitz[out]=carriedLipschitz*(1.0+abs(amplitude)*abs(b)*SVO_FIELD_NOISE_JACOBIAN_BOUND);
      pointClearance[out]=carriedClearance;
    }else if(code==${svoFieldOp("anisotropic-frame").wgslConstant}){
      let scale=select(vec3f(1.0),vec3f(a,b,c),vec3f(a,b,c)>vec3f(0.0));
      points[out]=svoFieldFrameRotateInverse(point,seed,d)/scale;
      pointLipschitz[out]=carriedLipschitz*SVO_FIELD_FRAME_ORTHONORMALITY_MARGIN/min(scale.x,min(scale.y,scale.z));
      pointClearance[out]=carriedClearance/max(scale.x,max(scale.y,scale.z));
    }else if(code==${svoFieldOp("scatter").wgslConstant}){
      let cell=max(a,1e-6);
      let repetitions=max(0.0,floor(b));
      let jitterAmplitude=clamp(c,0.0,cell);
      let occupant=max(0.0,d);
      let index=clamp(floor(point/cell+vec3f(0.5)),vec3f(-repetitions),vec3f(repetitions));
      let cellIndex=vec3i(index);
      let jitter=(vec3f(
        svoProceduralHashCell(cellIndex,seed^${SCATTER_SALT_X}u),
        svoProceduralHashCell(cellIndex,seed^${SCATTER_SALT_Y}u),
        svoProceduralHashCell(cellIndex,seed^${SCATTER_SALT_Z}u))-vec3f(0.5))*jitterAmplitude;
      points[out]=point-index*cell-jitter;
      pointLipschitz[out]=carriedLipschitz;
      let arrayHalf=vec3f(repetitions*cell+0.5*cell);
      let outside=max(0.0,svoFieldBox_m(point,arrayHalf));
      pointClearance[out]=min(carriedClearance,0.5*cell-0.5*jitterAmplitude-occupant+outside);
    }else if(code==${svoFieldOp("sphere").wgslConstant}){
      fieldDistance[out]=min(length(point)-a,carriedClearance);
      fieldLipschitz[out]=carriedLipschitz;
      fieldRadius[out]=abs(a);
      fieldClearance[out]=carriedClearance;
    }else if(code==${svoFieldOp("ellipsoid").wgslConstant}){
      let semiAxes=abs(vec3f(a,b,c));
      let radii=max(semiAxes,vec3f(1e-6));
      let ratio=max(radii.x,max(radii.y,radii.z))/min(radii.x,min(radii.y,radii.z));
      let clearance=carriedClearance/(ratio*ratio);
      fieldDistance[out]=min(svoFieldEllipsoid_m(point,vec3f(a,b,c)),clearance);
      fieldLipschitz[out]=carriedLipschitz;
      fieldRadius[out]=min(semiAxes.x,min(semiAxes.y,semiAxes.z));
      fieldClearance[out]=clearance;
    }else if(code==${svoFieldOp("box").wgslConstant}){
      let half=abs(vec3f(a,b,c));
      fieldDistance[out]=min(svoFieldBox_m(point,vec3f(a,b,c)),carriedClearance);
      fieldLipschitz[out]=carriedLipschitz;
      fieldRadius[out]=min(half.x,min(half.y,half.z));
      fieldClearance[out]=carriedClearance;
    }else if(code==${svoFieldOp("fracture").wgslConstant}){
      let normalLength=length(vec3f(a,b,c));
      let clearance=fieldClearance[fieldA];
      if(normalLength>1e-9){
        let halfspace=dot(vec3f(a,b,c),point)/normalLength-d;
        fieldDistance[out]=min(svoFieldSmoothMaximum(fieldDistance[fieldA],halfspace,max(0.0,e)),clearance);
        fieldLipschitz[out]=max(fieldLipschitz[fieldA],carriedLipschitz);
      }else{
        fieldDistance[out]=fieldDistance[fieldA];
        fieldLipschitz[out]=fieldLipschitz[fieldA];
      }
      fieldRadius[out]=fieldRadius[fieldA];
      fieldClearance[out]=clearance;
    }else if(code==${svoFieldOp("smooth-union").wgslConstant}||code==${svoFieldOp("smooth-subtract").wgslConstant}||code==${svoFieldOp("smooth-intersect").wgslConstant}){
      let radiusA=fieldRadius[fieldA];
      let radiusB=fieldRadius[fieldB];
      let ceiling=select(SVO_FIELD_UNBOUNDED_DISTANCE_M,c,c>0.0);
      let k=min(max(max(0.0,a)*min(radiusA,radiusB),max(0.0,b)),ceiling);
      let lipschitz=max(fieldLipschitz[fieldA],fieldLipschitz[fieldB]);
      if(code==${svoFieldOp("smooth-union").wgslConstant}){
        fieldDistance[out]=svoFieldSmoothMinimum(fieldDistance[fieldA],fieldDistance[fieldB],k);
        fieldRadius[out]=max(radiusA,radiusB);
        fieldClearance[out]=min(fieldClearance[fieldA],fieldClearance[fieldB]);
      }else if(code==${svoFieldOp("smooth-subtract").wgslConstant}){
        fieldDistance[out]=svoFieldSmoothMaximum(fieldDistance[fieldA],-fieldDistance[fieldB],k);
        fieldRadius[out]=radiusA;
        fieldClearance[out]=SVO_FIELD_UNBOUNDED_DISTANCE_M;
      }else{
        fieldDistance[out]=svoFieldSmoothMaximum(fieldDistance[fieldA],fieldDistance[fieldB],k);
        fieldRadius[out]=min(radiusA,radiusB);
        fieldClearance[out]=SVO_FIELD_UNBOUNDED_DISTANCE_M;
      }
      fieldLipschitz[out]=lipschitz;
    }else if(code==${svoFieldOp("worley-subtract").wgslConstant}){
      let fade=svoFieldBandLimitFade(select(1e20,a,a>0.0),samplingLength_m);
      let baseDistance=fieldDistance[fieldA];
      let clearance=fieldClearance[fieldA];
      let carved=svoFieldSmoothMaximum(baseDistance,max(0.0,b)-svoFieldWorley_m(point,a,seed),max(0.0,c));
      fieldDistance[out]=baseDistance+fade*(min(carved,clearance)-baseDistance);
      fieldLipschitz[out]=fieldLipschitz[fieldA]+fade*(max(fieldLipschitz[fieldA],carriedLipschitz*SVO_FIELD_WORLEY_JACOBIAN_BOUND)-fieldLipschitz[fieldA]);
      fieldRadius[out]=fieldRadius[fieldA];
      fieldClearance[out]=clearance;
    }else if(code==${svoFieldOp("ridged-worley-add").wgslConstant}){
      let fade=svoFieldBandLimitFade(select(1e20,a,a>0.0),samplingLength_m);
      let baseDistance=fieldDistance[fieldA];
      let inclusion=max(svoFieldWorley_m(point,a,seed)-max(0.0,b),baseDistance-max(0.0,d));
      let joined=svoFieldSmoothMinimum(baseDistance,inclusion,max(0.0,c));
      fieldDistance[out]=baseDistance+fade*(joined-baseDistance);
      fieldLipschitz[out]=fieldLipschitz[fieldA]+fade*(max(fieldLipschitz[fieldA],carriedLipschitz*SVO_FIELD_WORLEY_JACOBIAN_BOUND)-fieldLipschitz[fieldA]);
      fieldRadius[out]=fieldRadius[fieldA];
      fieldClearance[out]=fieldClearance[fieldA];
    }else if(code==${svoFieldOp("erosion-mask").wgslConstant}){
      let baseDistance=fieldDistance[fieldA];
      let depth=max(0.0,c);
      let removal=min(depth,max(0.0,fieldDistance[fieldB]-baseDistance));
      let cavity=select(-1.0,2.0*removal/depth-1.0,depth>0.0);
      let facing=svoFieldUpFacing(point,d);
      let weight=clamp(0.5+0.5*(max(0.0,a)*facing+max(0.0,b)*cavity),0.0,1.0);
      let operandLipschitz=fieldLipschitz[fieldA]+fieldLipschitz[fieldB];
      fieldDistance[out]=baseDistance+weight*removal;
      fieldLipschitz[out]=fieldLipschitz[fieldA]+operandLipschitz*(1.0+max(0.0,b))+depth*max(0.0,a)*carriedLipschitz/max(1e-4,d);
      fieldRadius[out]=fieldRadius[fieldA];
      fieldClearance[out]=fieldClearance[fieldA];
    }
  }
  return SvoFieldValue(fieldDistance[resultRegister],fieldLipschitz[resultRegister]);
}

// The unbanded entry point every existing call site spells. Threading a real
// voxel cell size into it is a one-argument change at the caller, and until
// that happens a zero sampling length means "no band limit", which is exactly
// the field a CPU-oracle comparison has to be held against.
fn ${functionName}(blockIndex:u32,localPoint:vec3f)->SvoFieldValue{
  return ${functionName}Banded(blockIndex,localPoint,0.0);
}
`;
}

/** Re-exported so a consumer needing the shared noise does not reach past this module. */
export { sampleSvoProceduralNoise, svoProceduralHashCell };
