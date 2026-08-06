/**
 * What one cloud pad of the hero bonsai's crown is made of.
 *
 * `bonsai-canopy-pads.ts` says where the pads go; this says what is inside one.
 * The answer is a `field-program` tape whose florets are **instanced geometry**,
 * not surface relief.
 *
 * ## Why the previous tape drew flat circles
 *
 * It built the florets out of `ridged-worley-add`, which is a displacement, and
 * it did it with the one parameter pairing that op cannot survive. From the
 * evaluator:
 *
 *     inclusion = max(F1(p, cell) - radius, base - shell)
 *     result    = smin(base, inclusion, blend)
 *
 * The second branch of that `max` is not a floor on how far the surface may
 * travel. It is a **cutting plane**: intersecting the ball with `base <= shell`
 * slices every ball off flat at exactly `shell` above the surface it sits on. A
 * ball of radius `r` centred on the surface comes back as a disc of radius
 * `sqrt(r^2 - shell^2)`, and it is flat, and its rim is sharp because `blend` is
 * far smaller than the cut.
 *
 * Every rung of the old ladder had `shell < radius`, so every rung was sliced:
 *
 *   rung     cell    radius   shell    flat disc across   lattice pitch
 *   clump    80 mm   36.0 mm  19.8 mm       60.1 mm           80 mm
 *   floret   32 mm   14.4 mm  13.0 mm       12.5 mm           32 mm
 *
 * Discs 60 mm across on an 80 mm pitch, merging into capsules wherever two
 * landed within a blend of each other. That is exactly the frame that came back,
 * and no amount of retuning the ball share reaches a round nodule, because the
 * cut is imposed by `shell` and `shell` is also the reach budget.
 *
 * ## The insight that was missing
 *
 * Not "the shell was mistuned". The construction was wrong:
 *
 * > A displacement can move a surface in and out. It cannot put a void between
 * > two masses, because there is only ever one surface.
 *
 * Which is the same sentence `bonsai-canopy-pads.ts` already carries about the
 * crown — a displacement could not turn one crown into seven pads, so the pads
 * became records. Foliage is separated masses with sky and shadow between them,
 * at *every* scale, so the same argument applies one scale down and the florets
 * have to be instanced too. Bumps on a closed blob read as a raspberry however
 * they are tuned; they read as a raspberry because that is what they are.
 *
 * ## `scatter`, which the machine already had
 *
 * `svo-field-program.ts` carries op 7, `scatter` — domain repetition with
 * per-cell jitter and a clamped index — and its own row says what it is for:
 * "the florets; an occupant that is itself a scatter is self-similarity for one
 * op". The long comment at the head of `bonsai.ts` had already worked out that
 * this was the construction the crown wanted, and recorded that it could not be
 * reached because `SceneryNode` had no `field-program` member and no generator
 * could author a tape.
 *
 * That wall is gone — the pads publish `field-program` nodes now — so the tape
 * that comment describes is the tape below. A scatter folds the evaluation point
 * into one cell, so a single ellipsoid op becomes an instance in every cell of
 * the lattice, and the space between instances is genuinely empty. It is also
 * far cheaper than the op it replaces: a fold is three hashes against Worley's
 * twenty-four.
 *
 * The one thing a scatter cannot do is close up. `clearance_m` — the gap between
 * an occupant and its own cell wall — has to stay positive or the field has no
 * continuous bound and the tracer walks through the surface, and the validator
 * refuses the tape outright if it does not. So an occupant is always under half
 * a cell and the array is always beads, never a solid. That is the reason for
 * the core below rather than a detail of it: the florets supply the outline and
 * the shadow, and the core supplies the mass they hang on.
 */
import type { SvoFieldProgram } from "../svo-field-program";
import type { Vec3 } from "../model";

/**
 * The three lattices the pad's surface is made of, coarsest first, as cell
 * pitches in metres.
 *
 * **One lattice cannot pave.** `scatter` refuses an occupant that touches its
 * own cell wall — without a positive clearance the folded field has no
 * continuous bound at all — so an occupant is always under half a cell and one
 * level covers at most `pi/4` of the plan, in practice 41 %. The first trace of
 * this tape used a single level and came back as a smooth ellipsoid with sparse
 * warts on it, which is the old failure wearing different clothes: 25 % plan
 * coverage means three quarters of what the eye sees is still the core.
 *
 * Three interleaved lattices at independent seeds leave `(1 - 0.407)^3` of the
 * plan bare — **79 % covered** — and because their pitches differ the lumps they
 * make are three sizes rather than one. That is the self-similarity, spelled as
 * three levels of instanced mass rather than as octaves of displacement.
 *
 * The middle pitch is set so its occupant is the plate's own 30.8 mm floret. The
 * coarse one is the clump above it and the fine one the sub-floret below, and
 * the ladder stops there because `scatter` writes a point register and the
 * machine has three free.
 *
 * The floor under all three is the march, not the op budget. Clearance is
 * `0.10 * cell`, and it is the sphere-trace step for every ray inside the pad —
 * so the finest level sets what a canopy pixel costs. At 30 mm that is a 3 mm
 * step across a 300 mm pad; halving the pitch again would double the cost of the
 * largest object in the frame to add a scale the leaf can barely hold.
 */
export const CANOPY_FLORET_CELLS_M: readonly number[] = Object.freeze([0.068, 0.043, 0.030]);

/**
 * The sub-floret grain, across — the plate's fourth scale.
 *
 * Five to eight plate pixels, about 7 mm on this crown, and it is carried as
 * *pitting* rather than as another instanced level. Two reasons, and the second
 * is the binding one:
 *
 *  - A carve costs no reach. It only ever removes solid, so the pad's authored
 *    radius stays the pad's drawn radius.
 *  - A nested `scatter` would cost the march everything. Clearance is `min`ed
 *    down the chain, and an inner lattice fine enough for a 7 mm grain has a
 *    clearance under a millimetre — so every ray crossing a pad would step in
 *    sub-millimetre increments across 340 mm of it. The outer lattice's 7.7 mm
 *    is already the number that decides what a canopy pixel costs.
 *
 * Gated on the leaf, because a rung under about four voxels of pitch is not a
 * feature, it is aliasing. At production's depth-3 leaf of 0.78125 mm this is
 * nine voxels and draws; at the 6.25 mm default it is one and does not.
 */
export const CANOPY_GRAIN_ACROSS_M = 0.007;

/** Voxels of pitch a rung needs before it is geometry rather than noise. */
export const CANOPY_LEGIBLE_PITCH_LEAVES = 4;

/**
 * Occupant radius as a share of its lattice's cell, and the number the whole
 * construction trades against.
 *
 * The validator caps it strictly under a half. Everything below that cap is a
 * three-way trade:
 *
 *  - **Coverage** goes as `pi * share^2` — 41 % at 0.36 against 25 % at 0.28,
 *    which over three levels is the difference between 79 % of the pad's plan
 *    being floret and 62 %.
 *  - **Clearance** is `(0.5 - jitter/2 - this) * cell`, the sphere-trace step
 *    inside the pad. Going to 0.44 for another six points of coverage would cut
 *    it from a tenth of a cell to a fiftieth and cost the march five times over.
 *  - **Separation.** At 0.36 a floret is 72 % of its pitch across, so the gaps
 *    within one level are still real voids and not blend seams. The levels
 *    overlap each other, which is the point; a level overlapping *itself* would
 *    just be a lumpy solid.
 */
const FLORET_OCCUPANT_SHARE = 0.40;

/**
 * How deep the solid core sits below the pad's envelope, in coarse clusters.
 *
 * **A canopy pad is about 60 % full, not solid.** This is the number that decides
 * that, and it was wrong: the core was inset by one *finest* leaf radius — 6 % of
 * the pad — so the pad was a 90 %-solid ball with leaves stuck to the outside of
 * it, and every trace of it came back reading as a textured ball, because that is
 * what it was. No amount of work on the leaves fixes a shape whose volume is
 * already committed.
 *
 * At two coarse clusters the core holds about a third of the envelope and the
 * remaining two thirds is a shell the leaves populate to roughly a seventh of
 * itself, which puts the pad near 40 % solid and — more to the point — makes the
 * voids between clusters *deep*. A void that bottoms out 6 mm down is a dimple; a
 * void that bottoms out 54 mm down is shadow, and shadow between clusters is what
 * separates a canopy into leaves rather than into bumps.
 *
 * The cost is that a fine cluster can now sit in the outer shell touching nothing
 * and hang in mid-air. That is accepted rather than prevented: at this fill a
 * cluster almost always overlaps one of its coarser neighbours, and the
 * alternative — an inset small enough to guarantee contact — is the solid ball
 * this constant exists to stop.
 */
const CANOPY_CORE_INSET_CLUSTERS = 2;

/**
 * Jitter as a share of the cell.
 *
 * Zero is a crystal and reads as one. It comes out of the clearance budget at
 * half rate, so it is cheap, but it buys less here than it would with a single
 * lattice: three interleaved levels at independent seeds already decorrelate
 * the surface far more than jitter within one of them does.
 */
const FLORET_JITTER_SHARE = 0.04;

/**
 * A leaf's half-thickness, as a share of its half-width.
 *
 * Round lumps at three sizes are still round lumps — the first two traces of
 * this tape came back as cauliflower, and thickness is most of the reason. A
 * leaf is a *plate*: thin enough that its rim reads as an edge rather than as a
 * curve, and thin enough that the light on its face and the light on its
 * neighbour's face differ. At 0.15 the rim's radius of curvature is 2 % of the
 * plate's width, which is a hard edge for anything the voxel grid can hold.
 *
 * Anisotropy is free here — `scatter` polices the occupant's *bounding* radius,
 * which the plan axes already set — so this costs neither clearance nor
 * coverage. It is the cheapest knob on the tape and it was the one set wrong.
 *
 * It is a floor rather than a fixed value, because a plate thinner than about
 * two voxels is not a plate, it is speckle. See `canopyLeafRadii_m`.
 */
const FLORET_FLATTEN = 0.28;

/**
 * How far each level's leaves are turned out of axis alignment, in `[0, 1]`.
 *
 * A `scatter` folds by translation alone, so **every instance of one level has
 * the same orientation** — which is the second half of why lumps at three sizes
 * read as cauliflower rather than as foliage. Real foliage is legible precisely
 * because its plates face every which way: some catch the key light flat-on and
 * some are edge-on and nearly black, and that variance across a few pixels is
 * the signal the eye reads as "leaves".
 *
 * `anisotropic-frame` rotates from a hash of its seed, once, not per cell — so
 * one rotation per level is the most this machine can give, and three levels
 * give three. Not many, but three sets of interleaved plates at unrelated
 * angles, each at its own size, is enough for the variance; one set is not.
 *
 * Full turn rather than a partial blend toward the identity: a leaf has no
 * preferred axis, and a partial turn keeps the lattice's own axes visible in the
 * result, which is the artefact this exists to remove.
 */
const FLORET_TURN = 1;

/** Carve depth as a share of the grain cell. */
const GRAIN_DEPTH_SHARE = 0.32;

/**
 * Fillet joining a floret to what is under it, as a share of the floret's
 * radius.
 *
 * Small, and it wants to stay small. The gap between florets is the whole point
 * of instancing them, and a blend is exactly the operation that fills a gap:
 * `smin`'s `k` is its own maximum inflation, so every millimetre here is a
 * millimetre of shadow traded for a millimetre of fillet. It is above zero only
 * because a hard union leaves a crease at every junction and a crease shades as
 * a black line.
 */
const FLORET_UNION_BLEND_SHARE = 0.16;

/**
 * Rounding on the clip that holds the array to the pad's authored silhouette, as
 * a share of the floret's radius.
 *
 * A floret straddling the pad's envelope is cut, and a hard cut is the flat disc
 * this module exists to stop drawing — one scale up and on the outline, which is
 * the worst place for it. This rounds the cut edge instead. It cannot remove the
 * cut: the authored radius is a contract with the layout, and something has to
 * enforce it.
 */
const FLORET_CLIP_BLEND_SHARE = 0.40;

/** A jittered lattice of instances, resolved into the four words `scatter` takes. */
export interface CanopyScatter {
  readonly cell_m: number;
  readonly jitter_m: number;
  readonly occupantRadius_m: number;
  /**
   * Gap from an occupant to its own cell wall — the value the folded field
   * saturates at, and therefore the sphere-trace step inside the array.
   */
  readonly clearance_m: number;
}

/** A Worley pitting pass: pits of `depth_m` on a `cell_m` lattice. */
export interface CanopyCarve {
  readonly cell_m: number;
  readonly depth_m: number;
  readonly blend_m: number;
}

export interface BonsaiCanopyField {
  /** The interleaved lattices, coarsest first. */
  readonly levels: readonly CanopyScatter[];
  /** The voxel the pads will be built at, which sets how thin a leaf may be. */
  readonly leafSize_m: number;
  /** The sub-floret grain — present only when the leaf can draw it. */
  readonly grain?: CanopyCarve;
}

/**
 * One level's leaf, as the half-extents of a **box** in its own rotated frame,
 * sized so its corner lands exactly on `occupantRadius_m` — which is the radius
 * `scatter` polices and the radius the clearance budget was spent on.
 *
 * A box rather than an ellipsoid, and the reason is not aesthetic. An ellipsoid's
 * distance is gradient-normalised rather than exact, so `sourceClearanceFactor`
 * charges it the **square** of its radius ratio: a 6.7 : 1 plate divides its
 * clearance by 44, and the first trace of this leaf was refused outright by the
 * extent bound — "field register 0 saturates at 0.0002 m but an op asks the
 * extent bound to dilate it by 0.0010 m". A box is exact outside, so its factor
 * is one and a plate of any aspect keeps the whole 3 to 7 mm the fold left it.
 *
 * That it is also the better leaf is luck worth taking. A box has flat faces and
 * straight edges; an oblate ellipsoid is a lens, and a canopy of lenses is the
 * cauliflower this tape has already drawn twice.
 *
 * Thin, but never thinner than one voxel each side. A plate under the leaf size
 * does not render as a thin plate; it renders as a dotted line wherever the grid
 * happens to catch it, and a canopy of those is worse than a canopy of lumps. So
 * the coarse-leaf fallback is a fatter leaf rather than a broken one, and the
 * width shrinks to keep the corner on the declared radius.
 *
 * "Exactly on" is a tenth of a percent inside. The validator's test is a strict
 * inequality against a radius it recomputes from the packed half-extents, so a
 * corner solved to land *on* the bound fails it about half the time on rounding
 * alone — "a bounding radius of 0.0272 m but the scatter above it declared an
 * occupant of at most 0.0272 m", at depth 1 and finer and not at all at depth 0.
 * The margin is far below anything the picture can show and removes the tie.
 */
const FLORET_CORNER_MARGIN = 0.999;

export const canopyLeafRadii_m = (level: CanopyScatter, leafSize_m: number): [number, number, number] => {
  const radius_m = FLORET_CORNER_MARGIN * level.occupantRadius_m;
  // `2w^2 + t^2 = radius^2` with `t = FLORET_FLATTEN * w` is the box whose
  // corner is the declared bound.
  const nominalWidth_m = radius_m / Math.sqrt(2 + FLORET_FLATTEN * FLORET_FLATTEN);
  const halfThickness_m = Math.min(radius_m, Math.max(FLORET_FLATTEN * nominalWidth_m, leafSize_m));
  const halfWidth_m = Math.sqrt(Math.max(1e-12, radius_m * radius_m - halfThickness_m * halfThickness_m) / 2);
  return [halfWidth_m, halfThickness_m, halfWidth_m];
};

/**
 * The ladder, resolved against the leaf the pads will actually be voxelized
 * into.
 *
 * `svoSceneryDetailCellSize_m` is that leaf: 6.25 mm at refinement depth 0 and
 * **0.78125 mm at the depth 3 production runs at**. Every length here is the
 * plate's own measurement in absolute metres; the leaf only chooses where to
 * stop.
 */
export function bonsaiCanopyField(leafSize_m = 0.00625): BonsaiCanopyField {
  const levels: CanopyScatter[] = CANOPY_FLORET_CELLS_M.map((cell_m) => {
    const occupantRadius_m = FLORET_OCCUPANT_SHARE * cell_m;
    const jitter_m = FLORET_JITTER_SHARE * cell_m;
    return {
      cell_m, jitter_m, occupantRadius_m,
      clearance_m: 0.5 * cell_m - 0.5 * jitter_m - occupantRadius_m,
    };
  });
  const grain: CanopyCarve = {
    cell_m: CANOPY_GRAIN_ACROSS_M,
    depth_m: GRAIN_DEPTH_SHARE * CANOPY_GRAIN_ACROSS_M,
    blend_m: 0.2 * GRAIN_DEPTH_SHARE * CANOPY_GRAIN_ACROSS_M,
  };
  // The carve runs on the *assembled* pad, where nothing clamps it back to a
  // fold's saturation value, so it is only sound while it acts strictly inside
  // the tightest of them: a pit `depth_m` deep can only raise the field where
  // the field is already under `depth_m`, and the finest level saturates at
  // 3.0 mm. Checked rather than assumed, because the failure it guards against
  // is a tracer stepping through the canopy rather than a wrong shape.
  const tightest_m = Math.min(...levels.map((level) => level.clearance_m));
  const affordable = CANOPY_GRAIN_ACROSS_M >= CANOPY_LEGIBLE_PITCH_LEAVES * leafSize_m
    && grain.depth_m + 0.25 * grain.blend_m < tightest_m;
  return { levels, leafSize_m, ...(affordable ? { grain } : {}) };
}

/**
 * One pad, as a tape: a core, three interleaved lattices of rotated leaf plates
 * unioned onto it, a clip to the pad's authored envelope, and — at a fine enough
 * leaf — a pitting pass over the whole thing.
 *
 * `radius_m` is the **finished** pad. The clip enforces that rather than the
 * relief budget the old tape had to reserve, so a caller sizes the shape it
 * wants to see and gets it.
 *
 * Sixteen ops of the machine's sixteen at production's leaf; fifteen where the
 * grain is gated off. The tape is full, and what fills it is one lattice, one
 * rotation and one leaf per level. A fourth level would need a fourth of each
 * and there is no room, which is the honest reason the ladder is three deep.
 */
export function bonsaiCanopyPadProgram(
  radius_m: Vec3,
  field: BonsaiCanopyField,
  seed: number,
): SvoFieldProgram {
  const coarsest = field.levels[0];
  // See `CANOPY_CORE_INSET_CLUSTERS`. This is what makes the pad a canopy rather
  // than a ball with a pattern on it, and it is a floor of a tenth of the pad on
  // each axis so that a small pad in the set is hollowed too rather than being
  // inset away to nothing.
  const inset_m = CANOPY_CORE_INSET_CLUSTERS * coarsest.occupantRadius_m;
  const core: [number, number, number] = [
    Math.max(0.10 * radius_m.x, radius_m.x - inset_m),
    Math.max(0.10 * radius_m.y, radius_m.y - inset_m),
    Math.max(0.10 * radius_m.z, radius_m.z - inset_m),
  ];
  const pad: [number, number, number] = [radius_m.x, radius_m.y, radius_m.z];
  const span_m = Math.max(radius_m.x, radius_m.y, radius_m.z);
  // Field registers alternate rather than accumulating in place. Reading and
  // writing one register in a single op happens to work in both evaluators, but
  // it is not a property either of them states, and a tape is not the place to
  // rely on one.
  const accumulator = (index: number): number => (index % 2 === 0 ? 2 : 1);
  return {
    ops: [
      // The mass. An ellipsoid's extent is its own half-axes per axis, which is
      // exact — the `sourceClearanceFactor` blow-up that made the old tape
      // borrow a box's bound was a *clearance* correction, and nothing on this
      // tape clamps against this field's clearance.
      { op: "ellipsoid", out: accumulator(0), point: 0, parameters: { a: core[0], b: core[1], c: core[2] } },
      ...field.levels.flatMap((level, index) => {
        const radii = canopyLeafRadii_m(level, field.leafSize_m);
        // Enough cells to cover the pad's longest axis. The array is a cube and
        // the pad is not, so this over-covers the thin one; that costs nothing,
        // because the clip removes the surplus instances and the record's extent
        // comes from the clip's operand rather than from the array.
        const repetitions = Math.max(1, Math.ceil(span_m / level.cell_m));
        // Each level gets its own seed, and that is what makes three lattices
        // cover 79 % of the plan instead of one covering 41 % three times.
        const levelSeed = (seed ^ Math.imul(index + 1, 0x9e37_79b1)) >>> 0;
        return [
          // The lattice. Everything reading the point register it writes is
          // evaluated once and drawn in every cell.
          {
            op: "scatter" as const, out: 1, point: 0, seed: levelSeed,
            parameters: {
              a: level.cell_m, b: repetitions, c: level.jitter_m, d: level.occupantRadius_m,
            },
          },
          // This level's leaf angle. Rotation only — the scale words are left at
          // zero, which reads as one, because squashing an axis in a frame costs
          // `1/min(scale)` on the Lipschitz constant and therefore that factor on
          // every march step. The plate's thinness is in the source's own radii,
          // where it is free.
          //
          // Point registers are reused across levels rather than one per level:
          // a `scatter` and a frame each write one, the machine has four, and
          // three levels would need seven. Nothing reads a level's registers
          // after its union, so they are scratch.
          {
            op: "anisotropic-frame" as const, out: 2, point: 1,
            seed: (levelSeed ^ 0x2545_f491) >>> 0, parameters: { d: FLORET_TURN },
          },
          // One leaf — which is to say, every leaf of this level.
          {
            op: "box" as const, out: 0, point: 2,
            parameters: { a: radii[0], b: radii[1], c: radii[2] },
          },
          // Onto whatever is already there. `b` is an absolute floor on the
          // blend rather than `a`'s share of the smaller operand's radius,
          // because the operands differ in scale by an order of magnitude and
          // only the floret's own size means anything here.
          {
            op: "smooth-union" as const,
            out: accumulator(index + 1), fieldA: accumulator(index), fieldB: 0,
            parameters: { b: FLORET_UNION_BLEND_SHARE * level.occupantRadius_m },
          },
        ];
      }),
      // The pad's authored silhouette, on the unfolded point.
      { op: "ellipsoid", out: 0, point: 0, parameters: { a: pad[0], b: pad[1], c: pad[2] } },
      // Operand order is the extent contract: `smooth-intersect` carries operand
      // A's box forward, and A is the pad. Handing it the array instead would
      // declare a record the size of the lattice's bounding cube.
      {
        op: "smooth-intersect", out: 3, fieldA: 0, fieldB: accumulator(field.levels.length),
        parameters: { b: FLORET_CLIP_BLEND_SHARE * coarsest.occupantRadius_m },
      },
      // The fourth scale, on the assembled surface so that it is coherent across
      // the whole pad. Running it on a folded point instead would repeat the
      // identical pit pattern inside every floret of that level.
      ...(field.grain ? [
        {
          op: "worley-subtract" as const, out: 0, fieldA: 3, point: 0, seed: (seed ^ 0x632b_e59b) >>> 0,
          parameters: { a: field.grain.cell_m, b: field.grain.depth_m, c: field.grain.blend_m },
        },
      ] : []),
    ],
    result: field.grain ? 0 : 3,
  };
}
