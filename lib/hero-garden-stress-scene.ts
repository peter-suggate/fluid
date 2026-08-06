import type { SceneDescription } from "./model";
import type { SceneryNode } from "./scenery-graph";
import {
  createHeroGardenHoseScene,
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_SET_SEED,
  HERO_GARDEN_VESSEL,
  type HeroGardenHoseOptions,
} from "./hero-garden-scene";
import { BONSAI_SHELF_MINIATURE } from "./voxel-scenery/bonsai";
import { ROSETTE_AIR_PLANT, ROSETTE_GRASS_TUFT, type RosetteForm } from "./voxel-scenery/rosette";
import { heroGardenPlanDistance_m, withHeroLayout } from "./voxel-scenery/hero-layout";

/**
 * The hero garden at a chosen multiple of its own authored record count.
 *
 * This is the acceptance scene for the raster-visibility program
 * (`docs/svo-raster-visibility-handoff.md`, W0). Its whole reason to exist is
 * one number: the hero frame spends **85 % of 405 ms** in a single pass that
 * rasterizes one proxy box per authored SDF record and sphere-traces the record
 * in the fragment, so its cost is proportional to *record count* rather than to
 * *visible surface*. The program's thesis is that visibility must stop scaling
 * with authored record count, and that claim is only falsifiable against a scene
 * where the record count actually moves. `hero-garden-hose` publishes 501
 * records; at `recordMultiplier: 10` this publishes ~5 000 of them.
 *
 * ---------------------------------------------------------------------------
 * Why it densifies the hero rather than inventing a scene
 * ---------------------------------------------------------------------------
 * W3's gate is "frame cost < 1.3x when authored record count is scaled 10x
 * behind an **unchanged camera**". Two scenes with different cameras cannot
 * discharge that, and neither can two scenes made of different species: a
 * comparison whose 10x set is a cloud of spheres would measure sphere marching,
 * not the pass. So this document is the hero document — the same vessel, the
 * same baked terrain, the same coping, stone set, bonsai, hose, plantings and
 * layout, at the same `heroGardenCamera` — with *more of the set it already
 * has* standing on the banks around the pond:
 *
 *   - `rosette` air plants and grass tufts, the same two authored forms the
 *     hero plants three of. 46 and 61 cone records each.
 *   - `BONSAI_SHELF_MINIATURE`, the catalogue's small specimen: 91 records, of
 *     which the crown is smooth-union aggregates. 0.22 m across, so it stands on
 *     a 0.2 m bank without becoming the subject the way the hero's own 0.66 m
 *     specimen would.
 *
 * Both are already in the shipped species catalog, so the 10x set evaluates the
 * same distance functions in the same proportions as the 1x set. Nothing is
 * scaled, displaced or given a new material: the only thing that changes between
 * `recordMultiplier: 1` and `recordMultiplier: 10` is how many records the
 * camera is looking at.
 *
 * ---------------------------------------------------------------------------
 * Where the extra set stands
 * ---------------------------------------------------------------------------
 * On the banks, on a golden-angle spiral seeded from the pond's own centre, with
 * every stand that would land on the coping crest or in the water rejected. The
 * spiral fills outward from the middle, so the accepted stands ring the pond —
 * which is where a garden's planting is, and, more to the point, where this
 * camera is looking. A stress scene whose extra records sat behind the eye would
 * flatter every measurement taken on it.
 *
 * The sequence is a *prefix*: stand `k` is at the same place whatever the
 * multiplier, so 2x is a subset of 10x rather than a different arrangement, and
 * a sweep 1x -> 10x moves one variable.
 *
 * ---------------------------------------------------------------------------
 * The ceiling this scene is authored against
 * ---------------------------------------------------------------------------
 * Aggregate (`smooth-union-cluster`) records are the binding constraint, not the
 * record count. `SVO_DRY_SCENE_CLUSTER_CAPACITY` is 1 024 blocks
 * (`lib/webgpu-svo-dry-scene.ts:292`) and the hero already spends 208 of them on
 * its pebble beds, boulder caps and bonsai. A proportional 10x would want 2 080
 * and would not draw at all, so the mix below is deliberately cluster-light:
 * one miniature bonsai per nine stands, eight rosettes beside it. At 10x that is
 * ~4 500 extra records for well under half the cluster arena. Raising the
 * cluster ceiling is W4's row, not this scene's business.
 *
 * ---------------------------------------------------------------------------
 * What still stops it at 10x, as of W0
 * ---------------------------------------------------------------------------
 * The document builds at every multiplier up to 10 and the record publication
 * is clean. What refuses is the *live sparse world*, at construction, before a
 * frame is attempted: `OCTREE_LIVE_SCENE_MATERIAL_CAPACITY` is
 * `ENVIRONMENT_VOXEL_MATERIAL_BASE + OCTREE_LIVE_SCENE_PRIMITIVE_CAPACITY`
 * = 32 + 4 096 (`lib/webgpu-octree-sparse-bricks.ts:225`, `:229`), and 5 039
 * records ask for 5 071 —
 *
 *   Write range (size: 486816) does not fit in
 *   [Buffer "Sparse voxel PBR material table"] size (396288)
 *
 * followed by `RangeError: offset is out of bounds` out of
 * `packLiveSvoMaterialEmission` (`:353`), from `new OctreeSparseBrickWorld`
 * (`:969`). Behind it waits the record throw at `:1473`. Both are W2's file, so
 * both are handoffs rather than edits here. **8x (4 026 records) is the largest
 * rung that draws today**, and `FLUID_SVO_DRY_SMOKE_RECORD_MULTIPLIER` is how a
 * lane asks for it.
 */

/** Stands laid per whole step of the multiplier: one bonsai and eight rosettes. */
const STANDS_PER_MULTIPLIER_STEP = 9;

/**
 * Records one whole step publishes, measured rather than predicted.
 *
 * 91 (shelf miniature) + 3 x 61 (grass tuft) + 5 x 46 (air plant), which comes
 * out at ~504 against the hero's own 501. That near-identity is what makes the
 * multiplier mean what it says, and it is why the mix is a function of the stand
 * index rather than a table. Measured through `buildSvoScenePrimitives`:
 *
 *   x1   501 records, 208 aggregates      x8   4 026 records, 779 aggregates
 *   x2  1 006, 291                        x10  5 039 records, 948 aggregates
 *   x5  2 510, 529
 *
 * so every rung is within a percent of its own label (10x is 10.06x).
 */
export const HERO_GARDEN_STRESS_RECORDS_PER_STEP = 504;

/**
 * Ceiling on the sweep, and it is the cluster arena's rather than this scene's.
 *
 * Measured: 10x publishes 5 039 records and **948** of the 1 024 aggregate
 * blocks `SVO_DRY_SCENE_CLUSTER_CAPACITY` holds. 12x throws
 * `Cluster index 1024 is outside the arena's 1024-block capacity`
 * (`lib/webgpu-svo-dry-scene.ts:581`) while building the records, before any
 * device is involved. So the sweep stops at the acceptance point: going past it
 * is W4's cluster row, not an edit here.
 */
export const HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER = 10;

/** Every emitted node is under this prefix, so the hero's own ids cannot collide. */
export const HERO_GARDEN_STRESS_NODE_PREFIX = "stress/";

/** The golden angle, which is what stops a spiral from growing arms. */
const GOLDEN_ANGLE_RAD = Math.PI * (3 - Math.sqrt(5));

/**
 * How many stands the spiral is *parameterized* over, which is not how many are
 * taken.
 *
 * Fixing it makes stand `k` independent of the count requested — the prefix
 * property the whole sweep rests on. 384 is enough that the accepted subset
 * (roughly half; the pond covers the rest of the ground plane) comfortably
 * exceeds the 81 stands a 10x scene asks for.
 */
const SPIRAL_STANDS = 384;

/** Plan half-extent of the spiral, inside the container's own walls. */
const SPIRAL_HALF_EXTENT_M = [0.86, 0.56] as const;

/**
 * Clear of the coping's crest centreline, per species.
 *
 * `heroGardenPlanDistance_m` is signed from the crest, so `rimHalfWidth_m` is
 * the outer foot. A rosette is 60-76 mm across and may bank against that foot; a
 * 0.22 m bonsai standing there would have its crown over the water and its roots
 * in the rim, which is a different scene rather than a denser one.
 */
const ROSETTE_BANK_CLEARANCE_M = 0.035;
const BONSAI_BANK_CLEARANCE_M = 0.135;

/** Margin from the container walls, so no stand is half outside the domain. */
const CONTAINER_MARGIN_M = 0.02;

interface StressStand {
  readonly index: number;
  readonly at_m: readonly [number, number];
  readonly species: "bonsai" | "air-plant" | "grass-tuft";
}

/** Integer avalanche hash in [0, 1). Same index in, same jitter out. */
function hash01(n: number): number {
  let h = Math.imul(n ^ 0x9e3779b9, 0x85ebca6b) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35) >>> 0;
  return ((h ^ (h >>> 16)) >>> 0) / 0x1_0000_0000;
}

/**
 * Which species stand `k` is.
 *
 * One bonsai every nine, then grass every third of the rest. Written as a
 * function of the index rather than as a table so the mix is the same at every
 * multiplier — a step that changed its own composition would make a sweep
 * measure two things at once.
 */
function speciesForStand(index: number): StressStand["species"] {
  if (index % STANDS_PER_MULTIPLIER_STEP === 0) return "bonsai";
  return index % 3 === 1 ? "grass-tuft" : "air-plant";
}

/**
 * The stands, in spiral order, with the pond and the container walls removed.
 *
 * `u = sqrt((k + 1/2) / N)` is the radius that spreads a golden-angle spiral
 * uniformly over a disc; mapping it onto the container's own aspect keeps the
 * spread uniform over the ground plane rather than over a circle inscribed in
 * it, which would leave the long ends of an 1.8 x 1.2 m garden empty.
 */
export function heroGardenStressStands(count: number): readonly StressStand[] {
  if (!Number.isSafeInteger(count) || count < 0) throw new RangeError("Stress stand count must be a non-negative integer");
  const stands: StressStand[] = [];
  const halfWidth = 0.5 * HERO_GARDEN_CONTAINER.width_m - CONTAINER_MARGIN_M;
  const halfDepth = 0.5 * HERO_GARDEN_CONTAINER.depth_m - CONTAINER_MARGIN_M;
  for (let k = 0; k < SPIRAL_STANDS && stands.length < count; k += 1) {
    const u = Math.sqrt((k + 0.5) / SPIRAL_STANDS);
    const angle = k * GOLDEN_ANGLE_RAD;
    const x = SPIRAL_HALF_EXTENT_M[0] * u * Math.cos(angle);
    const z = SPIRAL_HALF_EXTENT_M[1] * u * Math.sin(angle);
    const species = speciesForStand(stands.length);
    // Half the specimen's own plan extent, so "clear of the rim" and "inside the
    // container" are both about the object rather than about its anchor.
    const reach_m = species === "bonsai" ? 0.5 * 0.224 : 0.5 * 0.076;
    const clearance_m = species === "bonsai" ? BONSAI_BANK_CLEARANCE_M : ROSETTE_BANK_CLEARANCE_M;
    if (heroGardenPlanDistance_m(x, z) < HERO_GARDEN_VESSEL.rimHalfWidth_m + clearance_m) continue;
    if (Math.abs(x) + reach_m > halfWidth || Math.abs(z) + reach_m > halfDepth) continue;
    stands.push({ index: stands.length, at_m: [x, z], species });
  }
  if (stands.length < count) {
    throw new RangeError(`The hero garden's banks hold ${stands.length} stress stands; ${count} were asked for`);
  }
  return stands;
}

/** The scenery nodes one stand publishes. Seeded off its index, so no two repeat. */
function standNodes(stand: StressStand): SceneryNode {
  const seed = HERO_GARDEN_SET_SEED ^ (0x0057_0000 + 0x0101 * (stand.index + 1));
  const id = `${HERO_GARDEN_STRESS_NODE_PREFIX}${stand.species}-${stand.index}`;
  const at_m = stand.at_m;
  if (stand.species === "bonsai") {
    return {
      kind: "generator", id, generator: "bonsai", seed,
      // Leaning away from the pond's centre, which is what a specimen on a bank
      // does and, incidentally, what keeps a ring of them from all reaching for
      // the same point over the water.
      params: { ...BONSAI_SHELF_MINIATURE, at_m, lean: [at_m[0], at_m[1]] },
    };
  }
  const form: RosetteForm = stand.species === "grass-tuft" ? ROSETTE_GRASS_TUFT : ROSETTE_AIR_PLANT;
  return {
    kind: "generator", id, generator: "rosette", seed,
    // A little of the ground disturbed under each, jittered off the same hash
    // the stand is placed by, so a bed is never identical twice running.
    params: { ...form, at_m, bed_m: 0.020 + 0.012 * hash01(stand.index * 7 + 11) },
  };
}

export interface HeroGardenStressOptions extends HeroGardenHoseOptions {
  /**
   * How many times the hero's own authored record count this document publishes.
   *
   * 1 is the hero set exactly — the control the paired benchmark lane measures
   * against — and 10 is the program's acceptance point. Fractions are allowed
   * and resolve to a whole number of stands, so a sweep can walk 1 -> 10 in any
   * step it likes.
   */
  readonly recordMultiplier?: number;
}

/** Stands a multiplier asks for. Exported so a lane can report the sweep it ran. */
export function heroGardenStressStandCount(recordMultiplier: number): number {
  if (!Number.isFinite(recordMultiplier) || recordMultiplier < 1
    || recordMultiplier > HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER) {
    throw new RangeError(`Hero garden record multiplier must be between 1 and ${HERO_GARDEN_STRESS_MAXIMUM_MULTIPLIER}`);
  }
  return Math.round((recordMultiplier - 1) * STANDS_PER_MULTIPLIER_STEP);
}

/**
 * The hero garden, densified.
 *
 * `systems.fluid` stays off exactly as the hero's does: this is the dry render
 * path's acceptance scene, and turning the solver on would put a t=0 pressure
 * gate between a measurement and the thing being measured.
 */
export function createHeroGardenHoseStressScene(options: HeroGardenStressOptions = {}): SceneDescription {
  const recordMultiplier = options.recordMultiplier ?? 10;
  const standCount = heroGardenStressStandCount(recordMultiplier);
  // Spread rather than pick: `HeroGardenStressOptions` extends the hero's own,
  // and a densified scene that ignored the lattice would bake its ground and
  // grow its stands at a resolution the render tree is not using.
  const { recordMultiplier: _ignored, ...heroOptions } = options;
  const scene = createHeroGardenHoseScene(heroOptions);
  scene.sceneId = `hero-garden-hose-x${Number(recordMultiplier.toFixed(2))}`;
  const base = scene.scenery;
  if (!base) throw new Error("The hero garden must carry a scenery graph to be densified");
  scene.scenery = {
    ...base,
    nodes: [...base.nodes, ...heroGardenStressStands(standCount).map(standNodes)],
  };
  // The same blocking layout the catalog composes into the hero document, applied
  // after the extra set so both live in one graph with one id space. It drops and
  // re-adds only its own `layout/` nodes, so nothing above is disturbed.
  scene.scenery = withHeroLayout(scene);
  return scene;
}
