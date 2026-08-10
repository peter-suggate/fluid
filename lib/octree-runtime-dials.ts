/**
 * Live accuracy-for-frame-time dials on the Losasso coarse-dynamics lane.
 *
 * Every dial declared here is applied to a LIVE solver: no rebuild, no
 * re-seed, no change to the simulation clock. That is what makes them usable
 * as an experiment surface — you can sweep one while the water is moving and
 * watch the physics lane's measured time respond in the same session. The
 * topology dials change the next accepted lattice epoch inside allocations
 * that construction already sized; they never reallocate or reset t=0. That
 * is why every dial is listed in the method's `runtimeParamKeys` and excluded
 * from the structural fingerprint that would otherwise reset on every drag.
 *
 * The table below is what picked these dials and set their order. It is ONE
 * capture — `docs/losasso-coarse-band-10x-handoff.md`, symmetric-expansion at
 * factor 1, leaf 16, ~11.2 ms/advance steady after B1/B3b landed — and the
 * split moves with scene, domain, and how much of the surface is active. It is
 * deliberately not shown in the panel for that reason: a number a user cannot
 * reproduce on their own scene is worse than no number, and the executed
 * iteration count beside the strip is the live signal that does generalise.
 *
 * | subsystem                              | ms   | share |
 * |----------------------------------------|-----:|------:|
 * | resident MGPCG pressure solve          | 5.2  |  46%  |
 * |   · bottom sweeps                      | 1.9  |  17%  |
 * |   · L0 sweeps                          | 1.3  |  12%  |
 * |   · operator                           | 1.0  |   9%  |
 * | transport/dynamics (exact sampler)     | 3.2  |  28%  |
 * | topology + hierarchy maintenance       | 1.9  |  17%  |
 * | Section 5 axis-face extension          | 0.3  |   3%  |
 *
 * Two things that table understates. First, it is a two-level hierarchy: the
 * "L0 sweeps" line is the whole of what `vcycleSmoothingSweeps` controls here,
 * but on a domain deep enough to need the per-level V-cycle that dial spends a
 * dispatch per sweep at EVERY level, and it becomes the largest term left once
 * the iteration count is already small. Second, capping iterations does not
 * remove the fixed prelude — one operator apply, one V-cycle, the reductions —
 * so a cap of 1 still pays roughly two V-cycles. That is why the smoothing
 * dials, not the iteration dials, are what a large scene has left.
 *
 * The transport/dynamics term is the largest thing NOT dialled here: its cost
 * is the 36-limb exact superaccumulator that the D4 equivariance gate depends
 * on, and it is compiled into the shader rather than parameterised, so it
 * cannot move without a pipeline rebuild.
 *
 * A dial whose value equals its `auto` sentinel means "keep whatever
 * construction chose", so the untouched panel is bit-identical to the
 * production path.
 */

export interface OctreeRuntimeDials {
  /**
   * Target half-thickness, in finest cells, of the refined band the pressure
   * ladder keeps on each side of the free surface. An upper bound on what the
   * scene authored. Zero means AUTO: retain the authored pair.
   */
  readonly interfaceBandCells: number;
  /**
   * Smallest pressure-cell edge admitted at an ordinary factor-one surface
   * cut. One preserves the uniformly-unit shell; two lets band one represent
   * the cut directly on a size-two Losasso row. Sizing evidence still wins.
   */
  readonly finestSurfaceCellSize: number;
  /** Closed-container look-ahead width, in finest cells. */
  readonly wallBandCells: number;
  /**
   * Decades of slack added to the authored pressure residual tolerance.
   * 0 keeps `scene.numerics.pressureRelativeTolerance`; +3 solves to a
   * thousand times looser residual and stops that many CG iterations earlier.
   */
  readonly pressureToleranceDecades: number;
  /** Hard cap on executed CG iterations; 0 keeps the compiled envelope. */
  readonly pressureIterationCap: number;
  /** Bottom-level smoother sweeps per V-cycle; always even (ping-pong parity). */
  readonly vcycleBottomSweeps: number;
  /** Pre- and post-smoothing sweeps at the finest and intermediate levels. */
  readonly vcycleSmoothingSweeps: number;
  /** Section 5 fixed-K axis-face Jacobi sweeps per advance. */
  readonly velocityExtensionSweeps: number;
  /** Advances per topology epoch; 0 keeps the structural cadence. */
  readonly topologyRebuildCadence: number;
}

export interface OctreeRuntimeDialSpec {
  readonly key: keyof OctreeRuntimeDials;
  readonly label: string;
  /** Upper-case strip label; the panel has one line to say what this is. */
  readonly short: string;
  readonly unit: string;
  readonly min: number;
  readonly max: number;
  readonly step: number;
  readonly digits: number;
  readonly default: number;
  /** Value meaning "construction decides"; rendered as AUTO. Absent = none. */
  readonly auto?: number;
  /**
   * What the knob controls, in one line, in terms of the simulation rather
   * than the machinery. This is the line that has to answer "what am I even
   * moving" without the reader knowing what a V-cycle is.
   */
  readonly does: string;
  /**
   * What the cheap direction costs, naming the direction explicitly — cheaper
   * is *down* for three of these and *up* for the cadence, so "lower" is not a
   * safe shorthand. Concrete failure, not "reduced accuracy": the reason to
   * print this beside the slider is so a strange-looking frame is recognisable
   * as a dial you moved rather than a bug you introduced.
   */
  readonly cost: string;
  readonly hint: string;
}

/** Built-in Losasso constants the dials override, kept here for the readouts. */
export const OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS = 8;
export const OCTREE_RUNTIME_DIAL_BUILT_EXTENSION_SWEEPS = 8;
/** `OCTREE_LOSASSO_VCYCLE_SCHEDULE`'s authored preSweeps/postSweeps. */
export const OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS = 2;

/**
 * What a pressure solver is told, once the scene's authored numerics and the
 * compiled envelope have been folded in. Both Losasso executors accept this
 * shape; `undefined` on a field means "keep what was compiled", so a solver
 * that receives an empty object is bit-identical to the untouched path.
 */
export interface OctreeLosassoSolveTuning {
  /** Absolute relative-residual target, not a decade offset. */
  readonly relativeTolerance?: number;
  /** Executed-iteration ceiling; never above the compiled envelope. */
  readonly maximumIterations?: number;
  /** Even bottom-level smoother sweeps per V-cycle application. */
  readonly bottomSweeps?: number;
  /**
   * Pre- and post-smoothing sweeps at every level above the bottom. Pre and
   * post always take the SAME count: an asymmetric schedule makes the
   * preconditioner non-symmetric, and CG's convergence proof needs M to be SPD.
   */
  readonly smoothingSweeps?: number;
}

export const OCTREE_RUNTIME_DIALS: readonly OctreeRuntimeDialSpec[] = Object.freeze([
  {
    key: "interfaceBandCells",
    label: "Surface band thickness",
    short: "SURFACE BAND",
    unit: "cells",
    // AUTO is the paper/authored reach. Explicit values are experiments that
    // may thin it, never the shipped default.
    min: 0, max: 8, step: 1, digits: 0, default: 0, auto: 0,
    does: "How far from the water surface the grid stays at its finest cells."
      + " The readout is the resulting half-thickness in finest cells, which is"
      + " what an octree slice actually shows.",
    cost: "Thinner is the cheap direction, and it is the one dial here that"
      + " makes the grid itself smaller rather than the work on it cheaper."
      + " Too thin and the surface moves out of its own refined region between"
      + " rebuilds: detail pops, thin sheets and spray lose their cells first,"
      + " and a fast front simulates on cells coarser than the feature it"
      + " carries.",
    hint: "AUTO retains the authored band. An explicit value IS the half-thickness, in finest cells, and it"
      + " drives BOTH terms of the protection width the ladder compares against"
      + " -- interfaceRefinementBandCells and surfaceRefinementGradingLayers."
      + " Grading layer 1 is the mandatory sharp 2:1 transition, enforced by"
      + " the topology balance closure, so it adds no second refinement halo."
      + " Layers above 1 are optional progressive padding and are removed"
      + " before the direct interface band is trimmed. Both are clamped to what"
      + " the scene authored, because the row capacity, the residency halo and"
      + " the redistance reach were all sized from those values -- this dial"
      + " thins the grid, it never widens it past what was allocated.",
  },
  {
    key: "finestSurfaceCellSize",
    label: "Finest surface cell",
    short: "SURFACE CELL",
    unit: "cells",
    min: 1, max: 2, step: 1, digits: 0, default: 1,
    does: "The smallest pressure cell used where the free surface actually cuts"
      + " the factor-one octree. Set Surface band to 1 and this to 2 to keep an"
      + " ordinary cut on a 2×2×2 pressure cell.",
    cost: "2 is much cheaper and removes the unit-cell skin visible on an"
      + " octree slice, but it is an aggressive diagnostic setting: the compact"
      + " dam-break A/B currently stalls when every initial cut is admitted at"
      + " that size. Thin sheets and one-cell droplets also become vulnerable.",
    hint: "Live factor-one experiment. At Surface band 1, value 2 stops the"
      + " topology ladder from hard-splitting every size-two sign crossing to"
      + " unit cells. The Losasso ghost pressure, topology migration and face"
      + " operator admit that coarse representation, but trajectory parity is"
      + " not implied: the compact dam-break currently exposes a stationary"
      + " coarse-cut failure. Direct transported sizing evidence overrides the"
      + " dial and retains unit cells. At wider surface bands the"
      + " uniformly-fine shell remains authoritative, so this control has no"
      + " effect.",
  },
  {
    key: "wallBandCells",
    label: "Wall refinement band",
    short: "WALL BAND",
    unit: "cells",
    min: 1, max: 4, step: 1, digits: 0, default: 1,
    does: "How many finest cells ahead of a closed container wall are checked"
      + " for nearby liquid before boundary cells are split.",
    cost: "Thinner is cheaper, especially on a plane against the tank wall. A"
      + " fast front can reach a coarse wall row before the next topology epoch,"
      + " shifting impact timing and reflected run-up.",
    hint: "Live closed-wall look-ahead. The Ando-style default keeps one finest"
      + " cell beside a wet closed wall; the free surface and ordinary 2:1"
      + " balance, rather than a domain-wide wall slab, drive all additional"
      + " refinement. Larger values are conservative wall-impact experiments"
      + " and must be justified by contact-time and reflected-run-up evidence."
      + " This does not weaken solid cuts or free-surface crossings.",
  },
  {
    key: "pressureToleranceDecades",
    label: "Pressure residual tolerance",
    short: "SOLVE TOLERANCE",
    unit: "decades looser",
    min: -2, max: 6, step: 0.5, digits: 1, default: 0, auto: 0,
    does: "How completely the pressure solve has to cancel divergence before it"
      + " is allowed to stop. The readout is the resulting residual target.",
    cost: "Looser leaves real divergence in the velocity field: water gains or"
      + " loses volume over time, jets lose their push, and the surface creeps"
      + " through walls where the error piles up.",
    hint: "Multiplies the authored relative residual tolerance by 10^n. The GPU"
      + " residual gate stops the CG loop the moment it is met, so a looser"
      + " target is fewer executed iterations and a proportionally cheaper"
      + " solve. Costs divergence: |∇·u| after projection rises by the same"
      + " factor, which shows up first as volume drift and softened jets.",
  },
  {
    key: "pressureIterationCap",
    label: "Pressure iteration cap",
    short: "ITERATION CAP",
    unit: "iterations",
    min: 0, max: 48, step: 1, digits: 0, default: 0, auto: 0,
    does: "Stops the pressure solve after this many iterations even if it has"
      + " not converged. Calm steps finish early anyway, so this only bites on"
      + " the hard ones — impacts, a dropped body, the first advance.",
    cost: "Below what a transient needs, that step publishes a half-solved"
      + " pressure: a visible bulge or squirt at the moment of impact, which"
      + " then propagates. It is a frame-time ceiling bought with the worst"
      + " frames, not the average one.",
    hint: "Hard ceiling on executed CG iterations, clamped to the compiled"
      + " envelope. Warm advances already converge in ~12, so this is free"
      + " until a transient (impact, drop, first advance) asks for more — it"
      + " buys a predictable frame at the cost of an unconverged projection on"
      + " exactly those steps.",
  },
  {
    key: "vcycleSmoothingSweeps",
    label: "V-cycle smoothing sweeps",
    short: "SMOOTHING",
    unit: "sweeps",
    min: 1, max: 4, step: 1, digits: 0,
    default: OCTREE_RUNTIME_DIAL_BUILT_SMOOTHING_SWEEPS,
    does: "Relaxation passes over every grid the solver visits on the way down"
      + " and back up, at the resolution the water actually lives at. This is"
      + " the bulk of what one solver step costs, and it is what still runs"
      + " when the iteration cap is 1.",
    cost: "Fewer passes leave short-wavelength pressure error uncorrected, so"
      + " small features go soft — ripples and thin sheets flatten, and splash"
      + " detail rounds off — while the large-scale flow stays right. Like the"
      + " bottom sweeps, it can cost more iterations than it saves.",
    hint: "Damped-Jacobi pre- and post-smoothing sweeps at level 0 and every"
      + " intermediate level; the coarsest level has its own dial. Pre and post"
      + " always match, because an asymmetric schedule breaks the SPD property"
      + " CG requires of its preconditioner. Odd counts are legal: the smoother"
      + " picks its starting ping-pong bank so the chain still ends where"
      + " restriction and prolongation expect it. Sweeps happen at the widest"
      + " levels, so on a large domain this is the biggest single lever left"
      + " once the iteration count is already small.",
  },
  {
    key: "vcycleBottomSweeps",
    label: "V-cycle bottom sweeps",
    short: "BOTTOM SWEEPS",
    unit: "sweeps",
    min: 2, max: 8, step: 2, digits: 0, default: OCTREE_RUNTIME_DIAL_BUILT_BOTTOM_SWEEPS,
    does: "Work spent on the coarsest grid smoothing out large-scale pressure"
      + " error. That is what lets the solve converge in a dozen iterations"
      + " instead of hundreds.",
    cost: "Fewer sweeps is a weaker head start, so the solve often needs MORE"
      + " iterations than the sweeps saved and the step gets slower, not"
      + " faster. Judge this one by the iteration readout above, never by the"
      + " sweep count.",
    hint: "Damped-Jacobi sweeps on the coarsest multigrid level per V-cycle"
      + " application. The measured cost is barrier latency per sweep, not"
      + " arithmetic, so this scales almost linearly. Weakening the"
      + " preconditioner can hand the saving back as extra CG iterations —"
      + " watch the executed-iteration readout, not just the sweep count."
      + " Even values only: the smoother ping-pongs between two vectors.",
  },
  {
    key: "velocityExtensionSweeps",
    label: "Velocity extension sweeps",
    short: "EXTENSION K",
    unit: "sweeps",
    min: 8, max: 8, step: 1, digits: 0, default: OCTREE_RUNTIME_DIAL_BUILT_EXTENSION_SWEEPS,
    does: "How far the water's velocity is carried out into the empty air ahead"
      + " of the surface, so the next step has something to advect the surface"
      + " through. 8 covers the whole band the solver tracks.",
    cost: "The W7 graph requires all 8 sweeps (seven propagation layers plus"
      + " one settling sweep), so this reach is coupled and cannot be thinned"
      + " independently without shrinking the published graph too.",
    hint: "Section 5 fixed-K Jacobi extension of the projected velocity into"
      + " the air band, for both the published field and the MacCormack"
      + " predictor. The default 8 covers the W7 adjacency graph plus one"
      + " settling sweep; below 7 the outer air faces keep a stale velocity"
      + " and the next advance's semi-Lagrangian backtrace samples it.",
  },
  {
    key: "topologyRebuildCadence",
    label: "Topology rebuild cadence",
    short: "TOPOLOGY CADENCE",
    unit: "advances",
    min: 0, max: 8, step: 1, digits: 0, default: 0, auto: 0,
    does: "How many steps reuse one octree before the fine cells are rebuilt"
      + " around wherever the surface has moved to. 1 rebuilds every step.",
    cost: "Raising it is the cheap direction here. Fast water outruns its own"
      + " refined region before the grid follows, so the surface temporarily"
      + " simulates on coarse cells and detail pops back when the rebuild"
      + " catches up. The padding for this was sized at build time from the"
      + " Advanced setting, not from this dial.",
    hint: "Advances per topology epoch: k skips the candidate build on k-1 of"
      + " every k advances and reuses the accepted octree. The construction"
      + " time cadence is what sized the candidate's extra dilation rings, so"
      + " raising this dial above the structural setting spends the band-4"
      + " slack instead of purpose-built padding — the surface can outrun its"
      + " refined region. That is the experiment; it degrades into a stale"
      + " pressure topology, not a failure.",
  },
] as const);

export const OCTREE_RUNTIME_DIAL_DEFAULTS: OctreeRuntimeDials = Object.freeze(
  Object.fromEntries(OCTREE_RUNTIME_DIALS.map((dial) => [dial.key, dial.default])),
) as unknown as OctreeRuntimeDials;

export const OCTREE_RUNTIME_DIAL_KEYS: readonly (keyof OctreeRuntimeDials)[] =
  Object.freeze(OCTREE_RUNTIME_DIALS.map((dial) => dial.key));

type DialValues = Readonly<Record<string, number | string | boolean | undefined>>;

function dialNumber(values: DialValues, spec: OctreeRuntimeDialSpec): number {
  const raw = values[spec.key];
  const numeric = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(numeric)) return spec.default;
  const clamped = Math.min(spec.max, Math.max(spec.min, numeric));
  // Snapping to the declared step is what keeps an odd bottom-sweep count out
  // of the ping-pong smoother; a URL or a profile can supply any number.
  const snapped = spec.min + Math.round((clamped - spec.min) / spec.step) * spec.step;
  return Math.min(spec.max, Math.max(spec.min, snapped));
}

/** Resolve a method-parameter bag into clamped, step-snapped dial values. */
export function resolveOctreeRuntimeDials(values: DialValues): OctreeRuntimeDials {
  return Object.freeze(Object.fromEntries(
    OCTREE_RUNTIME_DIALS.map((spec) => [spec.key, dialNumber(values, spec)]),
  )) as unknown as OctreeRuntimeDials;
}

/** True when both bags select the same GPU behaviour. */
export function octreeRuntimeDialsEqual(
  left: OctreeRuntimeDials, right: OctreeRuntimeDials,
): boolean {
  return OCTREE_RUNTIME_DIAL_KEYS.every((key) => left[key] === right[key]);
}

/** True when every dial is at its construction-time meaning. */
export function octreeRuntimeDialsAreDefault(dials: OctreeRuntimeDials): boolean {
  return octreeRuntimeDialsEqual(dials, OCTREE_RUNTIME_DIAL_DEFAULTS);
}

/**
 * The half-thickness the ladder actually keeps refined, in finest cells.
 *
 * This mirrors `pressureRefinementEvidence`'s protection width, in the units an
 * octree slice is read in, and it is what the panel shows -- because "band 4"
 * is not a thickness. The two branches differ, so the factor is not optional.
 * Grading layer one names the mandatory sharp 2:1 transition. The independent
 * topology balance closure supplies that transition, so charging it here too
 * would create a second, leaf-width halo at every level. Only explicitly
 * authored layers beyond one widen this distance predicate. Factor one takes
 * the RETAINED width, `(grading - 1) * max(2, size)`; factors four and eight
 * take the COMPACT width, `(grading - 1) * max(0, size - 2)`.
 */
export function octreeSurfaceProtectionWidthCells(
  bandCells: number, gradingLayers: number, leafSizeCells: number,
  fineLevelSetFactor: number,
): number {
  const extraGrading = Math.max(0, gradingLayers - 1);
  const reach = fineLevelSetFactor === 1
    ? extraGrading * Math.max(2, leafSizeCells)
    : extraGrading * Math.max(0, leafSizeCells - 2);
  return Math.max(1, bandCells) + reach;
}

/** The band and grading a dialled thickness resolves to, and what it achieves. */
export interface OctreeSurfaceBandSelection {
  readonly bandCells: number;
  readonly gradingLayers: number;
  /** Half-thickness actually achieved at the finest merge candidate, in cells. */
  readonly widthCells: number;
}

/**
 * Resolve a dialled half-thickness into the two authored terms that produce it.
 *
 * The protection width `pressureRefinementEvidence` compares against is
 * `max(1, band) + max(0, grading - 1) * max(2, size)` at coarse band. The first
 * grading layer is not distance padding: it names the mandatory 2:1 balance
 * that the separate topology closure already enforces. Only layers above one
 * are an authored progressive-refinement experiment.
 *
 * Prefer a direct interface band over optional grading when more than one pair
 * reaches the requested finest-cell thickness. This minimizes the amplified
 * reach at coarse candidates while preserving the exact dialled width.
 *
 * Both are clamped to what the scene authored. `planOctreePressureCapacity`,
 * the residency halo, the candidate dilation rings and
 * `planOctreeRedistanceSweeps` were all sized from the authored pair, and every
 * one of them is an upper bound that thinning stays inside.
 */
export function octreeDialledSurfaceBand(
  authoredBandCells: number,
  authoredGradingLayers: number,
  fineLevelSetFactor: number,
  dials: OctreeRuntimeDials,
): OctreeSurfaceBandSelection {
  if (!Number.isFinite(authoredBandCells) || authoredBandCells < 0) {
    throw new RangeError("Authored interface band must be finite and non-negative");
  }
  if (!Number.isFinite(authoredGradingLayers) || authoredGradingLayers < 1) {
    throw new RangeError("Authored grading layers must be finite and at least one");
  }
  const authoredBand = Math.max(1, Math.round(authoredBandCells));
  const authoredGrading = Math.max(1, Math.round(authoredGradingLayers));
  if (dials.interfaceBandCells === 0) {
    return { bandCells: authoredBand, gradingLayers: authoredGrading,
      widthCells: octreeSurfaceProtectionWidthCells(
        authoredBand, authoredGrading, 2, fineLevelSetFactor) };
  }
  const target = Math.max(1, Math.round(dials.interfaceBandCells));
  const width = (band: number, grading: number) =>
    octreeSurfaceProtectionWidthCells(band, grading, 2, fineLevelSetFactor);
  // Factor 4/8 takes the compact width, whose grading term vanishes at the
  // finest candidate: there the band alone is the thickness and grading has
  // nothing to contribute.
  if (fineLevelSetFactor !== 1) {
    const bandCells = Math.min(authoredBand, target);
    return { bandCells, gradingLayers: authoredGrading, widthCells: width(bandCells, authoredGrading) };
  }
  let best = { bandCells: 1, gradingLayers: 1, widthCells: width(1, 1) };
  for (let gradingLayers = 1; gradingLayers <= authoredGrading; gradingLayers += 1) {
    for (let bandCells = 1; bandCells <= authoredBand; bandCells += 1) {
      const widthCells = width(bandCells, gradingLayers);
      if (widthCells > target) continue;
      if (widthCells > best.widthCells
        || (widthCells === best.widthCells && gradingLayers < best.gradingLayers)) {
        best = { bandCells, gradingLayers, widthCells };
      }
    }
  }
  return best;
}

/**
 * Effective relative residual tolerance for an authored scene tolerance.
 * Floored well above f32 epsilon so a stricter dial cannot ask for a residual
 * the accumulator cannot represent.
 */
export function octreeDialledRelativeTolerance(
  authored: number, dials: OctreeRuntimeDials,
): number {
  if (!Number.isFinite(authored) || authored <= 0) {
    throw new RangeError("Authored pressure tolerance must be finite and positive");
  }
  const scaled = authored * 10 ** dials.pressureToleranceDecades;
  return Math.min(1e-1, Math.max(1e-12, scaled));
}

/**
 * Effective iteration ceiling given the compiled envelope. The dial can only
 * ever lower it: the encoded graph and the resident kernel's accounting were
 * both sized for `built`, and nothing here may widen them.
 */
export function octreeDialledIterationCap(
  built: number, dials: OctreeRuntimeDials,
): number {
  if (!Number.isSafeInteger(built) || built < 1) {
    throw new RangeError("Built pressure iteration envelope must be a positive integer");
  }
  const requested = dials.pressureIterationCap;
  if (requested <= 0) return built;
  return Math.max(1, Math.min(built, Math.round(requested)));
}
