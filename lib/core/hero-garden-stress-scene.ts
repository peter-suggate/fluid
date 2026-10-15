import type { SceneDescription } from "./model";
import type { SceneryNode } from "./scenery-graph";
import {
  createHeroGardenHoseScene,
  HERO_GARDEN_CONTAINER,
  HERO_GARDEN_SET_SEED,
  HERO_GARDEN_VESSEL,
  type HeroGardenHoseOptions,
} from "./hero-garden-scene";
import { planWoodyPlant, WOODY_PLANT_SPECIES, type WoodyPlantSpecies } from "./voxel-scenery/woody-plants";
import { ROSETTE_AIR_PLANT, ROSETTE_GRASS_TUFT, type RosetteForm } from "./voxel-scenery/rosette";
import { heroGardenPlanDistance_m, withHeroLayout } from "./voxel-scenery/hero-layout";

/**
 * The densified hero garden. Each nine-stand step adds one seeded woody
 * specimen and eight rosettes. Specimens cycle through birch, pine, willow,
 * azalea, juniper, boxwood and hebe; the original oak remains the focal tree.
 *
 * The historical x10 name denotes nine additional planting steps. Species
 * have different branching costs, so it is no longer an exact record-count
 * ratio. Benchmark callers should measure the published catalog, not infer
 * primitive counts from the scene name. Stand placement remains a stable
 * prefix at every multiplier, with the original camera and pond layout.
 */
const STANDS_PER_MULTIPLIER_STEP = 9;
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
 * 0.24 m woody specimen needs room for its crown and roots outside the rim.
 */
const ROSETTE_BANK_CLEARANCE_M = 0.035;
const WOODY_BANK_CLEARANCE_M = 0.135;

/** Margin from the container walls, so no stand is half outside the domain. */
const CONTAINER_MARGIN_M = 0.02;

interface StressStand {
  readonly index: number;
  readonly at_m: readonly [number, number];
  readonly species: WoodyPlantSpecies | "air-plant" | "grass-tuft";
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
 * One woody specimen every nine, then grass every third of the rest. Cycling
 * by stand index keeps each specimen's species stable as density increases.
 */
function speciesForStand(index: number): StressStand["species"] {
  if (index % STANDS_PER_MULTIPLIER_STEP === 0)
    return WOODY_PLANT_SPECIES[Math.floor(index / STANDS_PER_MULTIPLIER_STEP) % WOODY_PLANT_SPECIES.length]!;
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
    const woody = species !== "air-plant" && species !== "grass-tuft";
    const reach_m = woody ? 0.12 : 0.5 * 0.076;
    const clearance_m = woody ? WOODY_BANK_CLEARANCE_M : ROSETTE_BANK_CLEARANCE_M;
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
  if (stand.species !== "air-plant" && stand.species !== "grass-tuft") {
    const scale = { birch: .23, pine: .22, willow: .19, azalea: .20, juniper: .19, boxwood: .22, hebe: .20 };
    const { node } = planWoodyPlant({ key: id, species: stand.species, seed,
      scale_m: scale[stand.species], detail: "miniature",
      bark: { palette: "clay", value: .955, surface: "architectural" },
      foliage: { palette: "clay", value: .975, surface: "foliage" } });
    const angle = Math.atan2(at_m[0], at_m[1]);
    return { ...node, place: { units: "metres", anchor: "terrain", ground: at_m,
      position: { x: at_m[0], y: 0, z: at_m[1] },
      orientation: { w: Math.cos(angle / 2), x: 0, y: Math.sin(angle / 2), z: 0 } } };
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
   * Planting density step (the historical scene multiplier).
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
 * The factory defaults to a dry render scene for benchmark lanes. The catalog
 * explicitly requests water so its pond starts filled and visible.
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
