/**
 * What one foliage mass of the procedural oak is made of.
 *
 * `oak-branching.ts` says where the masses go; this says what is inside one. The
 * split is the one `bonsai-canopy-pads.ts` argues for and it is not a filing
 * convenience:
 *
 * > A displacement can roughen a surface; it cannot turn one mass into six.
 *
 * The first field canopy in this codebase put a whole crown in one ellipsoid and
 * roughened it, and every trace came back a smooth disc on a stem, because the
 * silhouette was the ellipsoid's and there was only ever one ellipsoid. So the
 * scale at which the crown stops being connected is **layout** — records, placed
 * by the branching module — and everything below it is **relief**: this tape.
 *
 * ## Instanced, not displaced — and why that is settled rather than re-argued
 *
 * `bonsai-canopy-field.ts` carries the whole derivation and it is worth reading
 * before touching anything here. In one paragraph: `ridged-worley-add` is a
 * displacement, its `base - shell` branch is a *cutting plane* rather than a
 * floor, and every ball it draws comes back sliced flat at `shell` above the
 * surface it sits on. Rounder tuning does not reach separated masses because
 * there is only one surface to move. Foliage is separated masses with sky and
 * shadow between them at *every* scale, so the leaves have to be instanced —
 * `scatter`, op 7, a domain fold that evaluates one occupant and draws it in
 * every cell of a jittered lattice, for three hashes against Worley's
 * twenty-four.
 *
 * Two consequences of that op are load-bearing and are repeated here because
 * every constant below is priced against them.
 *
 *  - **A scatter cannot close up.** `clearance_m` — the gap from an occupant to
 *    its own cell wall — is the value the folded field saturates at, and the
 *    validator refuses a tape whose clearance is not strictly positive, because
 *    a repetition whose occupant touches its cell wall has no continuous
 *    distance bound and the tracer walks through the surface there. So an
 *    occupant is always under half a cell, one lattice is always beads, and the
 *    core below is the mass the beads hang on rather than a detail of them.
 *  - **One lattice cannot pave.** At an occupant share `s` the *bounding ball*
 *    of an instance covers `pi·s²` of its cell's plan — 40.7 % at the 0.36 used
 *    here — so a single level leaves most of the pad showing whatever is under
 *    it. Three interleaved lattices at independent seeds, at three pitches, are
 *    what turn that into a covered surface with three sizes of mass in it. That
 *    is the self-similarity, spelled as instanced geometry rather than as
 *    octaves of displacement.
 *
 * ## What is different about an oak
 *
 * The bonsai's crown is cauliflower: round clumps at three sizes, packed. Its
 * own file records the two corrections that got it there — `FLORET_FLATTEN`,
 * because round lumps at three sizes are still round lumps, and `FLORET_TURN`,
 * because a fold is a translation and every instance of one level therefore
 * shares an orientation. An oak leans harder on both, plus a third thing the
 * bonsai did not want: **void**.
 *
 * The void does not come from a lower occupant share. It comes from the plate
 * being thin, and the arithmetic is worth writing out because it is the whole
 * argument for `OAK_LEAF_FLATTEN` and it is exact rather than measured.
 *
 * By Cauchy's projection formula the mean projected area of a convex body over
 * uniform orientations is a quarter of its surface area. A leaf here is a box
 * `2w x 2t x 2w` with `t = f·w` and a corner on the declared bounding radius
 * `r = s·cell`, so `w² = r²/(2 + f²)` and
 *
 *     mean projected area = (8w² + 16wt)/4 = w²(2 + 4f)
 *     plan coverage       = s²·(2 + 4f)/(2 + f²)
 *
 * At `s = 0.36`:
 *
 *     f          plan coverage per level     three levels covered
 *     ball       40.7 %                      79.1 %
 *     0.28       24.0 %   (the bonsai's)     56.1 %
 *     0.16       16.9 %   (this file's)      42.6 %
 *
 * So the same lattice at the same share, with a plate at 0.16 instead of a ball,
 * covers **41 % of what a ball would** and leaves more than half the pad's plan
 * open to whatever is behind it. That is the sky and the shadow, and it is also
 * why the skeleton stays visible through the crown — which `docs/oak-tree-
 * species-plan.md` takes from the reference article as an explicit requirement
 * ("branches remain visible, not hidden by excessive leaves"), and which the
 * bonsai's flat plate had no need of.
 *
 * The coverage figures above are the *unfloored* plate. At any real leaf the two
 * fine levels are floored to one voxel of half-thickness (see `oakLeafRadii_m`)
 * and are therefore fatter than authored, so the achieved coverage at the
 * production leaf is nearer 48 % and it *falls toward 42.6 %* as the leaf
 * shrinks. That direction is the right one and it is not an accident: a finer
 * voxel buys a thinner plate, and a thinner plate is more oak.
 *
 * ## The ladder, and what admitting a level by the leaf buys
 *
 * The specimen is a tree about 0.80 m tall. A real oak leaf blade is ~100 mm,
 * which at that scale is **4 mm**; a spray is 8-12 mm; a whole clump at the end
 * of a secondary limb is 25-30 mm. `OAK_FOLIAGE_CELLS_M` is exactly those three,
 * coarsest first, and the occupant share turns each pitch back into the thing it
 * is named for: a 30 mm cell holds a 21.6 mm clump, a 13 mm cell a 9.4 mm spray,
 * a 6 mm cell a 4.3 mm leaf blade. The ladder is not three arbitrary octaves; it
 * is clump / spray / leaf, and the finest rung is one leaf.
 *
 * A level is admitted only when its cell clears **three leaves** — the same band
 * law the branch orders use, from `bonsai.ts`: under about two voxels of period a
 * feature does not render as that feature, it renders as aliasing, and above
 * about three it renders as geometry. `oakCanopyField` therefore returns
 *
 *     rung      leaf        floor      levels admitted
 *     depth 0   6.25 mm     18.75 mm   1   (30 mm)
 *     depth 1   3.125 mm     9.375 mm  2   (30, 13 mm)
 *     depth 2   1.5625 mm    4.69 mm   3   (30, 13, 6 mm)
 *     depth 3   0.78125 mm   2.34 mm   3   (30, 13, 6 mm)
 *
 * always keeping the coarsest, because a foliage mass with no foliage in it is a
 * ball and a ball is a worse answer than a coarse one.
 *
 * This buys two things at once and both are the point of the design.
 *
 *  - **The foliage gains scales as the leaf shrinks rather than aliasing into
 *    it.** Every length here is the plant's own measurement in absolute metres;
 *    the leaf only chooses where to stop. A finer voxelization adds a rung of
 *    real structure it can actually resolve, and a coarser one drops the rung it
 *    could only have speckled. Nothing is retuned per rung, so the silhouette is
 *    leaf-invariant — which is the crown-envelope invariant the plan doc asks
 *    to be asserted.
 *  - **The march cost is paid only at the rungs that can show it.** Which is the
 *    next section, and it is the one number here with a real performance
 *    consequence.
 *
 * ## The cost: clearance is the sphere-trace step
 *
 * `clearance_m` is not only the continuity condition. It is the value the folded
 * field saturates at, which means it is the **largest step any ray may take
 * inside the array** — a ray crossing a pad steps at most `clearance_m` at a
 * time however far it actually is from a leaf. Clearance is `min`ed down the
 * chain, so the *finest admitted level* sets what a canopy pixel costs, and the
 * canopy is the largest object in the frame.
 *
 * At `share = 0.36` and `jitter = 0.06` the clearance is `0.11 · cell`:
 *
 *     cell      clearance     admitted from
 *     30 mm     3.30 mm       depth 0
 *     13 mm     1.43 mm       depth 1
 *      6 mm     0.66 mm       depth 2
 *
 * The bonsai's tightest level saturates at 2.4 mm — its own header quotes 3.0 mm
 * from a `0.10 · cell` that predates its current 0.40 share, but the constants as
 * they stand give 2.4. So the oak's steps near the surface are about **3.6x
 * shorter** than the bonsai's at depth 2 and finer, and a canopy pixel is
 * correspondingly dearer there. How much dearer in *frame* terms is not
 * derivable here: the crossings a ray makes are `padRadius / clearance` and this
 * file does not choose the pad radius. **Measure it** on the depth-2 and depth-3
 * rungs before deciding the ladder is affordable; do not assume the ratio above
 * is the frame ratio, and do not assume it is free because the bonsai's was.
 *
 * If it is not affordable, the lever is `OAK_FOLIAGE_CELLS_M[2]`, not the share:
 * dropping the finest level costs one scale and returns 2.2x of step, whereas
 * buying clearance out of the share costs coverage quadratically.
 *
 * ## The clip is not the silhouette
 *
 * Everything above is *relief*, and the first render of this tape at depth 3
 * showed all of it working — plates draw, crisply, individually — over a crown
 * that still read as thirty pillows. The reason is one op and it is the last
 * one: the mass ends in `smooth-intersect` against `ellipsoid(pad)`, so no
 * arrangement of leaves can exceed the envelope and wherever they reach it the
 * drawn surface *is* the envelope. A perfect ellipse, at the one scale the hero
 * framing can actually resolve.
 *
 * So the tape carries a **lobe** — a `worley-subtract` at a good fraction of the
 * mass across, applied to the clip solid before it clips anything, which is the
 * one place on this tape where a carve of that depth is sound. `OakLobeShape`
 * carries the derivation. It replaces the sub-leaf grain carve that occupied
 * the same op and never once drew.
 *
 * ## Every number here is a starting point
 *
 * The bonsai's constants carry measured percentile tables off
 * `artifacts/plate-crops/canopy.png` and two failed art passes; these carry
 * arithmetic, an argument, and one look at a render. `/shape-lab` draws one mass
 * on the CPU at a chosen leaf with these on sliders (see
 * `lib/hero-garden-overrides.ts` for how a tuned result comes back), and the
 * shares below — flatten, jitter, the core inset, the lobe, all three blends —
 * are exactly the knobs that loop exists to settle. The *structure* is not a
 * starting point: three lattices, a box occupant, a rotation per level, the clip
 * last and the carve on the clip rather than on the pad are each forced by
 * something in `svo-field-program.ts`, and the reasons are on the constants.
 */
import { SVO_FIELD_PROGRAM_MAXIMUM_OPS, type SvoFieldProgram } from "../svo-field-program";
import type { Vec3 } from "../model";

/**
 * The three lattices a foliage mass is made of, coarsest first, as cell pitches
 * in metres: **clump, spray, leaf**.
 *
 * These are the plant's measurements at the specimen's 0.80 m, not octaves of
 * each other — 30 / 13 / 6 is a ratio of 2.31 then 2.17, near enough to an
 * octave to interleave cleanly and far enough off to stop the three lattices
 * sharing every eighth node. Each cell's occupant is `2 · 0.36 = 0.72` of the
 * pitch across, so the ladder reads 21.6 mm / 9.4 mm / 4.3 mm, which is a clump
 * at the end of a secondary limb, a spray, and one leaf blade.
 *
 * Three and not four: `scatter` writes a point register and `anisotropic-frame`
 * writes another, the machine has four points and three of them are scratch, and
 * the sixteen-op tape is already full at three levels plus the carve. A fourth
 * scale has to come from *layout* — more, smaller masses from `oak-branching.ts`
 * — which is the same trade `bonsai-canopy-pads.ts` makes one scale up.
 *
 * Moving the coarsest changes the crown's granulation and the clip blend with
 * it (both derive from `levels[0]`); moving the finest changes what a canopy
 * pixel costs and nothing else the eye can see at depth 2.
 */
export const OAK_FOLIAGE_CELLS_M: readonly number[] = Object.freeze([0.030, 0.013, 0.006]);

/**
 * Voxels of pitch a lattice needs before it is geometry rather than noise.
 *
 * Three, from the band law in `bonsai.ts` and restated in the plan doc: under
 * about two voxels a feature is aliasing, above about three it is geometry.
 * Three is the floor and this file sits on it deliberately — the finest level at
 * depth 2 is 3.84 voxels of pitch and its occupant is 2.8 voxels across, which
 * is the least a lattice can be and still be a lattice.
 *
 * Raising this to four would drop the 6 mm level at depth 2 and buy back 2.2x of
 * march step there. That is the honest lever if the measurement in the header
 * comes back badly, and it is a one-constant change.
 */
export const OAK_LEGIBLE_PITCH_LEAVES = 3;

/**
 * The lobing of a foliage mass's own outline — the scale between one clump and
 * the whole mass, and **the reason a crown stops reading as a bag of balls**.
 *
 * ## What this replaces, and why the thing it replaces could never have worked
 *
 * This op used to be a 2.5 mm sub-leaf grain carved into the *assembled* pad,
 * copied from the bonsai. It never once drew: it wanted four voxels of pitch and
 * 2.5 mm is 3.2 voxels at production's 0.78125 mm leaf, so `oakCanopyField`
 * omitted it at every rung the scene runs. Reviving it at its own scale was
 * never the fix, because a grain on a blade is not what was wrong with the
 * picture. What was wrong is one order of magnitude up.
 *
 * A rendered crown at depth 3 shows the three lattices working — plates draw,
 * crisply, individually. It still reads as thirty pillows, because every mass
 * ends in `smooth-intersect` against `ellipsoid(pad)` and **nothing on this tape
 * can exceed that ellipsoid**. Wherever the lattices reach the envelope — which
 * is most of it, at 48 % plan coverage over three levels blended together — the
 * drawn surface *is* the envelope, so each mass has a mathematically perfect
 * elliptical outline. At the hero framing a mass is about fifty pixels and the
 * leaf lattice is one or two, so the outline is the entire read. The relief was
 * never the problem. The clip was.
 *
 * So the carve moves two things at once: its **scale**, from a 2.5 mm blade
 * grain to a lobe a good fraction of the mass across, and its **operand**, from
 * the assembled pad to the clip solid.
 *
 * ## Carving the clip is sound where carving the pad is not
 *
 * The old placement is exactly why the old carve had to be shallow. On the
 * assembled pad the *declared* clearance has been released to unbounded by the
 * intersect, but the field's **value** there is still saturated at the tightest
 * fold's clearance — 0.66 mm — and a pit deeper than that lifts the field above
 * the true clearance and authorises a march step that is a lie. That is a
 * tracer walking through the canopy, not a shape that looks wrong, which is why
 * `oakCanopyField` checked it rather than assuming it. A 19 mm lobe on the pad
 * would violate it by a factor of thirty.
 *
 * On the clip solid there is nothing to violate. The clip is an `ellipsoid` on
 * point register 0, which is never folded, so `pointClearance_m[0]` is unbounded
 * and `svo-field-program.ts` gives the source `fieldClearance_m = Infinity`; the
 * field's value is the exact ellipsoid distance everywhere, with no saturation
 * anywhere to lift. `worley-subtract` inherits its operand's clearance, so the
 * carved clip is still unbounded, and its extent rule — a carve can only raise
 * the field, so the solid only shrinks — hands the intersect the same box the
 * bare ellipsoid would have. **The depth is therefore free**, and the only bound
 * on it is geometric rather than about soundness: see `OAK_LOBE_DEPTH_SHARE`.
 *
 * ## Relative to the mass rather than absolute
 *
 * Every other length in this file is the plant's own measurement in metres, and
 * this one deliberately is not. Its job is to break *this mass's* silhouette, the
 * masses are dealt at ±34 % of swell about a radius that is itself solved from a
 * volume share, and a fixed pitch would put nine lobes on the big ones and none
 * on the small. A share keeps the lobe count per mass constant, which is what
 * makes the crown read at one scale.
 *
 * That costs nothing in leaf-invariance, which is the contract that matters
 * here: a mass's radius comes from a share that sums to one at every rung with a
 * floor on the count, so it does not move with the leaf, so neither does the
 * lobe. And the lobe clears the band law at *every* rung — 54 mm against a
 * 18.75 mm floor at depth 0 — so unlike everything else on the ladder it draws
 * from the coarsest voxel up. Which is the point: the ball is worst exactly
 * where the leaves cannot draw.
 */
export interface OakLobeShape {
  /** Lobe pitch as a share of the mass's plan radius. */
  readonly across: number;
  /** Bite depth as a share of the mass's plan radius. */
  readonly depth: number;
}

/**
 * Lobe pitch as a share of the mass's plan radius.
 *
 * 0.85, which on the hero's 64 mm mass is a 54 mm lattice over a 128 mm mass —
 * about three cells across, so a handful of bites on the outline rather than a
 * scallop shell. Below about 0.5 the lobes are numerous enough to read as
 * texture on an ellipse instead of as a change to its shape, which is the
 * failure this whole op exists to correct and is easy to walk back into.
 */
export const OAK_LOBE_ACROSS_SHARE = 0.85;

/**
 * Bite depth as a share of the mass's plan radius, and the one number here with
 * a hard ceiling rather than a taste.
 *
 * `worley-subtract` removes the union of balls of radius `b` on the lattice, as
 * `max(d, b − F1)`. `sampleSvoFieldWorley_m` returns `min(cell/2, …)` — the
 * clamp that makes it 1-Lipschitz across a cell boundary — so **F1 never exceeds
 * half the cell**, and at `b >= cell/2` the term `b − F1` is positive
 * *everywhere* and the mass is eaten in its entirety. The ceiling is therefore
 * `0.5 · across`, exactly: 0.425 of the plan radius at the pitch above.
 *
 * 0.30 sits at 71 % of that, which is a 19 mm bite out of a 64 mm radius — deep
 * enough that the outline is genuinely re-entrant rather than dimpled, with
 * enough headroom that a slider does not fall off the cliff. `oakCanopyPadProgram`
 * refuses anything at or above the ceiling rather than publishing an empty
 * record, because a mass that vanishes is a hole in a crown and nothing in the
 * pipeline downstream of here would name the cause.
 */
export const OAK_LOBE_DEPTH_SHARE = 0.30;

/** The lobe as the hero form takes it, and what a caller gets by omitting one. */
export const OAK_LOBE_DEFAULT: OakLobeShape = Object.freeze({
  across: OAK_LOBE_ACROSS_SHARE,
  depth: OAK_LOBE_DEPTH_SHARE,
});

/**
 * Occupant radius as a share of its lattice's cell, and the number the whole
 * construction trades against.
 *
 * The validator caps it strictly under a half. Everything below that cap is a
 * three-way trade:
 *
 *  - **Coverage** goes as `pi · share²` for a ball and `share² · (2 + 4f)/(2 + f²)`
 *    for the plate this actually places — 16.9 % per level at 0.36 against
 *    12.6 % at 0.31, which over three levels is 42.6 % of the pad's plan covered
 *    against 33.3 %.
 *  - **Clearance** is `(0.5 - jitter/2 - this) · cell`, the sphere-trace step
 *    inside the pad. Every point of share comes straight off it, linearly, and
 *    at the 6 mm level a point of share is 60 µm of step.
 *  - **Separation.** At 0.36 an occupant's bounding ball is 72 % of its pitch
 *    across, so the gaps within one level are real voids rather than blend
 *    seams. The levels overlap each other, which is the point; a level
 *    overlapping *itself* is a lumpy solid.
 *
 * 0.36 against the bonsai's 0.40, and the four points are bought for clearance,
 * not for coverage: these cells are between 2.3x and 11x finer than the
 * bonsai's, so the same share is a much shorter step in metres. The coverage the
 * four points cost is more than returned by the plate — see the header's table —
 * which is the one place in this file where two constants are genuinely trading
 * against each other rather than against the march.
 */
const OAK_LEAF_OCCUPANT_SHARE = 0.36;

/**
 * Jitter as a share of the cell.
 *
 * Zero is a crystal and reads as one. Half of it comes out of the clearance
 * budget, so at 0.06 it costs 0.03 of a cell — 180 µm at the finest level, which
 * is a quarter of that level's clearance and the single most expensive thing on
 * this tape that is not the share.
 *
 * Above the bonsai's 0.04 nonetheless. A `scatter` folds by translation and
 * `anisotropic-frame` rotates once per level, not per cell, so **position jitter
 * is the only per-instance variation any level has** — every leaf of a level is
 * the same plate at the same angle, and the lattice is visible in the result to
 * exactly the extent that jitter does not hide it. A clipped cauliflower is
 * genuinely regular and can afford 0.04; an oak spray is not and cannot.
 *
 * If the march measurement comes back badly this is the first thing to spend,
 * before the share and before the finest level: it is worth less than either.
 */
const OAK_LEAF_JITTER_SHARE = 0.06;

/**
 * A leaf's half-thickness, as a share of its half-width.
 *
 * The bonsai's 0.28 was itself a correction — round lumps at three sizes read as
 * cauliflower — and an oak is the same argument taken further. 0.16 puts the
 * rim's radius of curvature at about 1 % of the plate's width, which is a hard
 * edge for anything the voxel grid can hold, and it is what makes the light on
 * one plate's face and the light on its neighbour's differ enough that a few
 * pixels of variance read as "leaves" rather than as noise on a solid.
 *
 * It is also, per the header, where the oak's void comes from: coverage goes as
 * `(2 + 4f)/(2 + f²)`, so dropping f from 0.28 to 0.16 removes 30 % of the plan
 * a level covers without touching the share, the clearance or the march.
 *
 * Anisotropy is free here — `scatter` polices the occupant's *bounding* radius,
 * which the corner constraint already pins — so this costs nothing at all. It is
 * the cheapest knob on the tape and it is the one most worth looking at first.
 *
 * A floor rather than a fixed value: a plate thinner than about one voxel each
 * side is not a plate, it is speckle. See `oakLeafRadii_m`.
 */
const OAK_LEAF_FLATTEN = 0.16;

/**
 * How far each level's leaves are turned out of axis alignment, in `[0, 1]`.
 *
 * A `scatter` folds by translation alone, so every instance of one level shares
 * an orientation, and `anisotropic-frame` rotates from a hash of its seed once
 * rather than per cell. One rotation per level is therefore the most this machine
 * can give and three levels give three — not many, but three sets of interleaved
 * plates at unrelated angles, each at its own size, is enough for the variance
 * that reads as foliage. One set is not.
 *
 * A full turn rather than a partial blend toward the identity: a leaf has no
 * preferred axis, and a partial turn leaves the lattice's own axes visible in
 * the result, which is the artefact the rotation exists to remove. An oak leaf
 * *does* have a preferred axis in nature — leaves present their faces to the sky
 * — but that is a per-instance fact and this op is per-level, so imposing it
 * here would tilt a whole third of the crown one way rather than orienting
 * anything.
 */
const OAK_LEAF_TURN = 1;

/**
 * How deep the solid core sits below the mass's envelope, in coarse clumps.
 *
 * **This is the number that decides how full a foliage mass is**, and the
 * bonsai's file records getting it wrong: its core was once inset by one
 * *finest* leaf radius — 6 % of the pad — so the pad was a 90 %-solid ball with
 * leaves stuck on the outside, and every trace read as a textured ball, because
 * that is what it was. No amount of work on the leaves fixes a shape whose
 * volume is already committed.
 *
 * Three coarse clumps rather than the bonsai's two, because an oak wants *more*
 * void than a cauliflower. At three the inset is 32.4 mm, so a 90 mm mass keeps
 * a 57.6 mm core — 26 % of the envelope by volume — and the outer 74 % is a
 * shell the three lattices populate to well under half of itself. The voids
 * between clumps then bottom out 32 mm down rather than 6 mm down, and that is
 * the whole difference between shadow and a dimple: shadow between clumps is
 * what separates a canopy into leaves.
 *
 * The cost is that a fine leaf can sit in the outer shell touching nothing and
 * hang in mid-air. Accepted rather than prevented, and more readily than the
 * bonsai accepts it: an oak's leaves *do* hang off twigs that this ladder never
 * draws (the plan doc's table puts a twig at 0.4 mm, half a voxel at production),
 * so a spray with no visible support is the correct picture of a thing whose
 * support is below the band limit. The alternative — an inset small enough to
 * guarantee contact — is the solid ball this constant exists to stop.
 */
const OAK_CORE_INSET_CLUMPS = 3;

/**
 * Floor under the core's half-axes, as a share of the mass's own radius.
 *
 * The inset is absolute — three clumps is 32.4 mm whatever the mass is — so a
 * small mass in the set would be inset away to nothing and publish a shell with
 * no mass inside it. Eight percent rather than the bonsai's ten, for the same
 * reason the inset is three clumps and not two, and it binds on any mass under
 * 35 mm of radius, which is the point at which a foliage mass is barely more
 * than one clump anyway.
 */
const OAK_CORE_FLOOR_SHARE = 0.08;

/**
 * Fillet joining a leaf to what is under it, as a share of its own radius.
 *
 * Small, and smaller than the bonsai's 0.16. The gap between leaves is the whole
 * point of instancing them and a blend is exactly the operation that fills a
 * gap: `smin`'s `k` is its own maximum inflation, so every micron here is a
 * micron of shadow traded for a micron of fillet. It is above zero only because
 * a hard union leaves a crease at every junction and a crease shades as a black
 * line.
 *
 * Passed as the blend's absolute floor (`b`) rather than as its share of the
 * smaller operand (`a`), because the operands differ in scale by an order of
 * magnitude — a 90 mm core against a 2 mm leaf — and only the leaf's own size
 * means anything at the joint.
 */
const OAK_LEAF_UNION_BLEND_SHARE = 0.10;

/**
 * Rounding on the clip that holds the array to the mass's authored silhouette,
 * as a share of the coarsest occupant's radius.
 *
 * A leaf straddling the envelope is cut, and a hard cut on a fat floret is a
 * flat disc on the outline — the worst place for the artefact this whole module
 * exists to stop drawing. The clip's blend rounds that cut instead. It cannot
 * remove it: the authored radius is a contract with the layout and something has
 * to enforce it.
 *
 * 0.25 against the bonsai's 0.40, and the direction is deliberate. `smooth-
 * intersect` is a `smax`, which overshoots *upward* by `k/4` — it erodes near
 * the boundary — so a generous blend does not just round the cut, it dissolves
 * the outermost leaves into a smooth shoulder. The bonsai wants that, because a
 * sliced cauliflower floret shows a broad flat face. An oak does not: a slice
 * through a 2.4 mm-thick plate is a sliver nobody reads, and a ragged outline of
 * half-cut leaves is what the silhouette should be.
 */
const OAK_LEAF_CLIP_BLEND_SHARE = 0.25;

/**
 * Rounding on the lobe's own cut, as a share of its depth.
 *
 * The bite is a union of balls and a ball meeting an ellipsoid leaves a crease;
 * a crease shades as a black line, which is the same reason the leaf union and
 * the clip both carry a blend. Small, because `smoothMaximum` overshoots upward
 * by `k/4` and every micron of that is the lobe eroding back toward the ellipse
 * it exists to break.
 */
const OAK_LOBE_BLEND_SHARE = 0.15;

/**
 * How far inside the declared occupant radius the leaf's corner is placed.
 *
 * A tenth of a percent, and it is a rounding fix rather than a shape decision.
 * The validator's test is a strict inequality against a bounding radius it
 * recomputes from the packed half-extents, so a corner solved to land exactly
 * *on* the bound fails about half the time on float rounding alone — the bonsai
 * saw it as "a bounding radius of 0.0272 m but the scatter above it declared an
 * occupant of at most 0.0272 m", at depth 1 and finer and not at depth 0. The
 * margin is orders of magnitude below anything the picture can show and it
 * removes the tie.
 */
const OAK_LEAF_CORNER_MARGIN = 0.999;

/** Ops the tape spends per admitted lattice: `scatter`, frame, `box`, `smooth-union`. */
const OPS_PER_LEVEL = 4;

/** Ops the tape spends regardless: the core, the clip solid, and the intersect. */
const OPS_FIXED = 3;

/** The lobe's `worley-subtract`, when the mass is large enough to carry one. */
const OPS_LOBE = 1;

/** One jittered lattice of instanced leaf plates. */
export interface OakScatter {
  readonly cell_m: number;
  readonly jitter_m: number;
  readonly occupantRadius_m: number;
  /** Gap from an occupant to its own cell wall: the sphere-trace step inside the array. */
  readonly clearance_m: number;
}

/** The lobe carve, resolved against one mass. */
export interface OakLobe {
  readonly cell_m: number;
  readonly depth_m: number;
  readonly blend_m: number;
}

export interface OakCanopyField {
  /** The admitted lattices, coarsest first. Never empty. */
  readonly levels: readonly OakScatter[];
  readonly leafSize_m: number;
}

/**
 * The ladder, resolved against the leaf the pads will be voxelized into.
 *
 * `svoSceneryDetailCellSize_m` is that leaf: 6.25 mm at refinement depth 0 and
 * **0.78125 mm at the depth 3 production runs at**. Every length in this module
 * is the plant's own measurement in absolute metres; the leaf only chooses where
 * the ladder stops, which is what makes "refines gracefully" a property of the
 * derivation rather than a set of per-rung numbers somebody has to keep in
 * agreement.
 *
 * The coarsest level is kept unconditionally. At a leaf coarse enough to refuse
 * even 30 mm the honest output is one lattice of fat plates — a granulated mass
 * — and the alternative is a bare ellipsoid, which is the shape this whole
 * module exists to stop publishing.
 */
export function oakCanopyField(leafSize_m = 0.00625): OakCanopyField {
  const floor_m = OAK_LEGIBLE_PITCH_LEAVES * leafSize_m;
  const admitted = OAK_FOLIAGE_CELLS_M.filter((cell_m) => cell_m >= floor_m);
  const cells = admitted.length > 0 ? admitted : [OAK_FOLIAGE_CELLS_M[0]];
  const levels: OakScatter[] = cells.map((cell_m) => {
    const occupantRadius_m = OAK_LEAF_OCCUPANT_SHARE * cell_m;
    const jitter_m = OAK_LEAF_JITTER_SHARE * cell_m;
    return {
      cell_m,
      jitter_m,
      occupantRadius_m,
      clearance_m: 0.5 * cell_m - 0.5 * jitter_m - occupantRadius_m,
    };
  });
  return { levels, leafSize_m };
}

/**
 * The lobe for one mass, or `undefined` where it would be noise.
 *
 * Only the band law gates it — a lobe under `OAK_LEGIBLE_PITCH_LEAVES` voxels of
 * pitch is the surface's own quantisation rather than a change to its shape.
 * There is no clearance gate, because the operand is the unfolded clip solid and
 * nothing about its field saturates; see `OakLobeShape` for why that is the
 * whole reason this op moved off the assembled pad.
 *
 * In practice the gate never binds on a hero mass — 54 mm against 18.75 mm at
 * the coarsest rung the lab offers — and it exists for the small end of the
 * swell distribution and for a caller who slides `across` down to nothing.
 */
export function oakCanopyLobe(
  planRadius_m: number,
  leafSize_m: number,
  shape: OakLobeShape = OAK_LOBE_DEFAULT,
): OakLobe | undefined {
  const cell_m = shape.across * planRadius_m;
  const depth_m = shape.depth * planRadius_m;
  if (!(cell_m >= OAK_LEGIBLE_PITCH_LEAVES * leafSize_m) || !(depth_m > 0)) return undefined;
  // The hard ceiling from `OAK_LOBE_DEPTH_SHARE`: `F1` is clamped to half the
  // cell, so a bite of half the cell or more removes the mass entirely. Refused
  // rather than clamped, because a caller who asks for this has a number wrong
  // and a silently shallower lobe is the kind of thing that gets tuned around
  // for an afternoon.
  if (!(depth_m < 0.5 * cell_m)) {
    throw new RangeError(
      `an oak foliage lobe of depth ${(shape.depth).toFixed(3)} of the plan radius needs a pitch above ` +
        `${(2 * shape.depth).toFixed(3)}, but OAK_LOBE_ACROSS_SHARE is ${shape.across.toFixed(3)}. ` +
        "worley-subtract removes balls of the depth's radius and its distance saturates at half the cell, " +
        "so at or above half the pitch the bite covers every point of the mass and the record publishes empty",
    );
  }
  return { cell_m, depth_m, blend_m: OAK_LOBE_BLEND_SHARE * depth_m };
}

/**
 * One level's leaf plate, as the half-extents of a **box** in its own rotated
 * frame, sized so its corner lands just inside `occupantRadius_m` — the radius
 * `scatter` polices and the radius the clearance budget was spent on.
 *
 * A box rather than an ellipsoid, and the reason is mechanical rather than
 * aesthetic. An ellipsoid's distance is gradient-normalised rather than exact,
 * so `sourceClearanceFactor` charges it the **square** of its radius ratio: the
 * 6.25 : 1 plate this file authors at the coarse level would divide its
 * clearance by 39, and the bonsai's first thin leaf was refused outright by the
 * extent bound — "field register 0 saturates at 0.0002 m but an op asks the
 * extent bound to dilate it by 0.0010 m". A box is exact outside, so its factor
 * is one and a plate of any aspect keeps the whole clearance the fold left it.
 *
 * That it is also the better oak leaf is luck worth taking twice over. A box has
 * flat faces and straight edges; an oblate ellipsoid is a lens, and a canopy of
 * lenses is the cauliflower the bonsai's tape drew twice before it stopped.
 *
 * Thin, but never thinner than one voxel each side. A plate under the leaf size
 * does not render as a thin plate; it renders as a dotted line wherever the grid
 * happens to catch it, and a canopy of those is worse than a canopy of lumps. So
 * the coarse-leaf fallback is a fatter leaf rather than a broken one, and the
 * width shrinks to keep the corner on the declared radius — which is why the
 * achieved plan coverage falls toward the authored figure as the leaf shrinks
 * rather than rising.
 */
export function oakLeafRadii_m(level: OakScatter, leafSize_m: number): [number, number, number] {
  const radius_m = OAK_LEAF_CORNER_MARGIN * level.occupantRadius_m;
  // `2w² + t² = radius²` with `t = OAK_LEAF_FLATTEN · w` is the box whose corner
  // is the declared bound.
  const nominalWidth_m = radius_m / Math.sqrt(2 + OAK_LEAF_FLATTEN * OAK_LEAF_FLATTEN);
  const halfThickness_m = Math.min(radius_m, Math.max(OAK_LEAF_FLATTEN * nominalWidth_m, leafSize_m));
  const halfWidth_m = Math.sqrt(Math.max(1e-12, radius_m * radius_m - halfThickness_m * halfThickness_m) / 2);
  return [halfWidth_m, halfThickness_m, halfWidth_m];
}

/**
 * One foliage mass, as a tape: a deeply inset core, the admitted lattices of
 * rotated leaf plates unioned onto it, and a clip to the mass's authored
 * envelope — the clip **lobed** first, so that the envelope is not an ellipsoid.
 *
 * `radius_m` is the **finished** mass. The clip enforces that rather than the
 * relief budget a displacement-based tape would have to reserve, so a caller
 * sizes the shape it wants to see and gets it. The lobe does not change that
 * contract: a carve only ever raises the field, so the mass shrinks inside the
 * authored envelope and never past it, and `smooth-intersect` still carries the
 * ellipsoid's own box forward as the record's extent.
 *
 * The op budget, and it is exactly full at the top of the ladder:
 *
 *     core ellipsoid                                    1
 *     per level: scatter, frame, box, smooth-union    4·L
 *     clip ellipsoid                                    1
 *     lobe carve, when admitted                       0/1
 *     smooth-intersect                                  1
 *
 * which is 8 ops at one level, 12 at two and 16 at three with the lobe — the
 * machine's whole sixteen. Identical to what the never-emitted grain carve
 * budgeted for, so this costs no rung of the ladder. A fourth lattice would need
 * four more ops and four more registers and there is room for neither, which is
 * the honest reason `OAK_FOLIAGE_CELLS_M` is three long.
 */
export function oakCanopyPadProgram(
  radius_m: Vec3,
  field: OakCanopyField,
  seed: number,
  lobeShape: OakLobeShape = OAK_LOBE_DEFAULT,
): SvoFieldProgram {
  const coarsest = field.levels[0];
  if (!coarsest) throw new RangeError("an oak foliage mass needs at least one lattice; oakCanopyField never returns none");
  // The lobe is sized off the plan radius rather than off the mean, so a mass
  // squashed by `foliageFlatten` gets the lobe count its *outline* asks for —
  // the outline being the thing this op exists to break.
  const lobe = oakCanopyLobe(Math.max(radius_m.x, radius_m.z), field.leafSize_m, lobeShape);
  const opCount = OPS_FIXED + OPS_PER_LEVEL * field.levels.length + (lobe ? OPS_LOBE : 0);
  // Checked here rather than left to the validator so that the failure names the
  // ladder rather than the tape. The budget is what caps `OAK_FOLIAGE_CELLS_M`
  // at three, and a fourth pitch added there would otherwise surface as an
  // arithmetic error inside `packSvoFieldProgramArena` with nothing pointing
  // back at the cause.
  if (opCount > SVO_FIELD_PROGRAM_MAXIMUM_OPS) {
    throw new RangeError(
      `an oak foliage mass at ${field.levels.length} lattices${lobe ? " plus the lobe carve" : ""} is ` +
        `${opCount} ops, over the ${SVO_FIELD_PROGRAM_MAXIMUM_OPS}-op machine limit. Each lattice costs ` +
        `${OPS_PER_LEVEL} ops (scatter, frame, box, union); shorten OAK_FOLIAGE_CELLS_M rather than raising the ` +
        "machine limit, which is per-tape arena words for every tape in the scene",
    );
  }
  // See `OAK_CORE_INSET_CLUMPS`. This is what makes the mass a canopy rather
  // than a ball with a pattern on it.
  const inset_m = OAK_CORE_INSET_CLUMPS * coarsest.occupantRadius_m;
  const core: [number, number, number] = [
    Math.max(OAK_CORE_FLOOR_SHARE * radius_m.x, radius_m.x - inset_m),
    Math.max(OAK_CORE_FLOOR_SHARE * radius_m.y, radius_m.y - inset_m),
    Math.max(OAK_CORE_FLOOR_SHARE * radius_m.z, radius_m.z - inset_m),
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
      // exact — the `sourceClearanceFactor` blow-up that forces a box on the
      // leaf is a *clearance* correction, and nothing on this tape clamps
      // against this field's clearance because point register 0 is never folded.
      { op: "ellipsoid", out: accumulator(0), point: 0, parameters: { a: core[0], b: core[1], c: core[2] } },
      ...field.levels.flatMap((level, index) => {
        const radii = oakLeafRadii_m(level, field.leafSize_m);
        // Enough cells to cover the mass's longest axis. The array is a cube and
        // the mass is not, so this over-covers the thin one; that costs nothing,
        // because the clip removes the surplus instances and the record's extent
        // comes from the clip's operand rather than from the array.
        const repetitions = Math.max(1, Math.ceil(span_m / level.cell_m));
        // Each level gets its own seed, and that is what makes three lattices
        // cover about 48 % of the plan at the production leaf instead of one
        // covering 17 % three times over the same cells. The salts differ from
        // the bonsai's so an oak and a bonsai handed the same seed do not grow
        // the same lattice.
        const levelSeed = (seed ^ Math.imul(index + 1, 0x85eb_ca6b)) >>> 0;
        return [
          // The lattice. Everything reading the point register it writes is
          // authored once and drawn in every cell.
          {
            op: "scatter" as const,
            out: 1,
            point: 0,
            seed: levelSeed,
            parameters: { a: level.cell_m, b: repetitions, c: level.jitter_m, d: level.occupantRadius_m },
          },
          // This level's leaf angle. Rotation only — the scale words are left at
          // zero, which reads as one, because squashing an axis in a frame costs
          // `1/min(scale)` on the Lipschitz constant and therefore that factor
          // on every march step through the crown. The plate's thinness lives in
          // the source's own radii, where it is free.
          //
          // Point registers are reused across levels rather than one per level:
          // a `scatter` and a frame each write one, the machine has four, and
          // three levels would need seven. Nothing reads a level's registers
          // after its union, so they are scratch.
          {
            op: "anisotropic-frame" as const,
            out: 2,
            point: 1,
            seed: (levelSeed ^ 0xc2b2_ae35) >>> 0,
            parameters: { d: OAK_LEAF_TURN },
          },
          // One leaf — which is to say, every leaf of this level.
          { op: "box" as const, out: 0, point: 2, parameters: { a: radii[0], b: radii[1], c: radii[2] } },
          // Onto whatever is already there. `b` is an absolute floor on the
          // blend rather than `a`'s share of the smaller operand's radius,
          // because the operands differ in scale by an order of magnitude and
          // only the leaf's own size means anything at the joint.
          {
            op: "smooth-union" as const,
            out: accumulator(index + 1),
            fieldA: accumulator(index),
            fieldB: 0,
            parameters: { b: OAK_LEAF_UNION_BLEND_SHARE * level.occupantRadius_m },
          },
        ];
      }),
      // The mass's authored silhouette, on the unfolded point. Written to
      // register 3 when a lobe follows and to 0 when none does, so that the
      // carve never reads and writes one register in a single op — which happens
      // to work in both evaluators but is not a property either of them states.
      { op: "ellipsoid", out: lobe ? 3 : 0, point: 0, parameters: { a: pad[0], b: pad[1], c: pad[2] } },
      // The lobing, on the clip **before** it clips anything.
      //
      // On the unfolded point, so the pattern is coherent across the whole mass
      // and neighbouring masses at different centres bite differently. Sound at
      // any depth here for the reason `OakLobeShape` sets out: this operand is
      // an exact ellipsoid with an unbounded clearance, so there is no saturated
      // value to lift and no march step to falsify. The same op on the assembled
      // pad — where it used to sit — is capped at 0.66 mm.
      ...(lobe
        ? [
            {
              op: "worley-subtract" as const,
              out: 0,
              fieldA: 3,
              point: 0,
              seed: (seed ^ 0x27d4_eb2f) >>> 0,
              parameters: { a: lobe.cell_m, b: lobe.depth_m, c: lobe.blend_m },
            },
          ]
        : []),
      // Operand order is the extent contract: `smooth-intersect` carries operand
      // A's box forward, and A is the pad. Handing it the array instead would
      // declare a record the size of the lattice's bounding cube — for the
      // finest level on a 90 mm mass, a 15-cell array, which is a proxy box the
      // brick binner would then have to reconcile against a shape a fifth its
      // size. The carve preserves that box exactly (it can only shrink the
      // solid), so A is still the pad whether or not the lobe ran.
      {
        op: "smooth-intersect",
        out: 3,
        fieldA: 0,
        fieldB: accumulator(field.levels.length),
        parameters: { b: OAK_LEAF_CLIP_BLEND_SHARE * coarsest.occupantRadius_m },
      },
    ],
    result: 3,
  };
}
