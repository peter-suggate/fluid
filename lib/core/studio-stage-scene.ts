import type { CameraState, SceneDescription } from "./model";
import type { SceneryGraph } from "./scenery-graph";

/**
 * The house set: a floor, one lamp over it, and nothing else at all.
 *
 * This is the `stage` environment, and it is the set the catalog's tanks and
 * oracles stand on. It is what those scenes moved to from the 2.9 m white
 * enclosure with a softbox — a room which was never seen, because the presets
 * that seeded it all presented `fluid-only`, and which could not have been seen
 * on most of them anyway: its walls were a fixed 1.45 m from the origin and
 * half the catalog is a container wider than that. A fixed room is not a
 * stage; it is a room that happens to fit one scene. The `default` environment
 * remains that white room, bright and unopinionated, for scenes that never
 * asked to be staged — a dark set is an art direction, not a default.
 *
 * What replaced it is the other end of that trade taken as cheaply as it can
 * be. A dam break in a flat void has no ground to break across, nothing to cast
 * onto, and no second surface to be brighter than — the fluid is legible and
 * the picture is inert. A floor and a practical fix all three for two
 * primitives, and both of them scale, so the same set frames a 0.8 m tank and a
 * 12.8 m paper figure without being re-authored for either.
 *
 * ---------------------------------------------------------------------------
 * What each of the three pieces is for
 * ---------------------------------------------------------------------------
 * **No enclosure.** The shell builds no faces (see `SceneryRoomShellNode.faces`)
 * and the stage is an ordinary box node instead. Two things follow. A wall three
 * metres from a tank is a large bright bounce surface, and a set lit by one
 * practical wants that bounce gone rather than dimmed — paint dark enough to
 * stop bouncing stops reading as paint. And the floor's top face can then be put
 * at exactly `y = 0`, which is where the container's base is, so the set meets
 * the water at the plane the solver already agrees on.
 *
 * **One lamp, and its beam is its shape.** The fixture is a truncated cone —
 * a reflector, narrow at the throat and open at the mouth — tagged `spot-light`,
 * and `svo-light-abi.ts` reads the beam angle off that taper. There is no second
 * number to disagree with the lamp standing in frame: widening the mouth widens
 * the pool. It is placed so the pool covers the container and falls off inside
 * the stage, which is what puts an edge in the picture.
 *
 * **A near-black ambient, and a key held down to a fill.** A spot is only a spot
 * against something darker; the rig is `STUDIO_STAGE_DRY_SCENE_LIGHTING` in
 * `lib/svo/svo-dry-scene-lighting.ts`, and it belongs to the environment rather
 * than to any one scene for the same reason the set does.
 *
 * ---------------------------------------------------------------------------
 * Why every number below is a ratio
 * ---------------------------------------------------------------------------
 * The set was solved on the 1.2 x 0.8 x 0.8 water box, and those solved values
 * are the constants' reference point: at that container every ratio here
 * reproduces the authored number exactly (pool 1.05 m, floor half 1.9 m, mouth
 * 2.45 m, reflector 0.22/0.05 m). Everything is expressed against the pool
 * radius rather than as an absolute so the whole fixture is one similarity
 * transform of itself, which is what keeps the *picture* — the ratio of lit
 * pool to dark stage, the place the falloff ring lands — invariant across a
 * sixteen-fold range of container sizes.
 */

/**
 * The pool covers the container's plan diagonal, with margin.
 *
 * A pool sized to the container's width leaves its corners in the dark, and a
 * tank whose corners are unlit reads as badly lit rather than as dramatically
 * lit. The diagonal is the smallest circle that contains the footprint, and the
 * margin puts the falloff ring outside the glass instead of on it.
 *
 * Two rungs, not one. The numerators are the pool radius at the reference
 * container: 1.05 m is the solved composition minimum — tighter than that and
 * the falloff ring lands on the glass — and 1.25 m is the preferred pool, with
 * lit boards left around the subject. The whole set scales with the pool, and
 * the set is paid for in domain bricks, so the wide pool is taken only where
 * the budget can hold it and the solved pool is the fallback: a scene that
 * could afford yesterday's composition keeps it rather than losing the set to
 * a preference. See `studioStageDimensions`, where the rung is chosen. The
 * per-pool ratios below still denominate by the 1.05 solve, so either rung is
 * one similarity transform of the same fixture.
 */
const POOL_RADIUS_PER_FOOTPRINT_SOLVED = 1.05 / (0.5 * Math.hypot(1.2, 0.8));
const POOL_RADIUS_PER_FOOTPRINT_PREFERRED = 1.25 / (0.5 * Math.hypot(1.2, 0.8));

/**
 * Where the lamp's mouth hangs, and how long the reflector therefore is.
 *
 * The beam half-angle is `atan((mouth - throat) / 2h)`, so choosing the pool
 * radius chooses `h`: raising the lamp does not widen the pool, it lengthens
 * the reflector. That is what lets the fixture hang out of frame without
 * redrawing the light it casts, and it is why the hang can take a maximum
 * below without any other number moving.
 *
 * The maximum is the second term. A container twenty metres tall is a column,
 * not a tank, and a lamp hung at 2.3 pool radii would be *inside* the water it
 * is meant to light. Clearing the container's own height is the floor under the
 * hang; on every ordinary tank the first term wins and the second never binds.
 */
const LAMP_HANG_PER_POOL_RADIUS = 2.45 / 1.05;
const LAMP_HEADROOM_PER_POOL_RADIUS = 1;

/** Reflector mouth and throat radii, and the stem that holds it up. */
const LAMP_MOUTH_RADIUS_PER_POOL = 0.22 / 1.05;
const LAMP_THROAT_RADIUS_PER_POOL = 0.05 / 1.05;
const LAMP_STEM_RADIUS_PER_POOL = 0.018 / 1.05;
const LAMP_STEM_HALF_HEIGHT_PER_POOL = 0.45 / 1.05;

/**
 * Emission, solved against the hang.
 *
 * A spot's irradiance in the dry shader is `intensity / max(1, d^2)` — see the
 * `SVO_LIGHT_SPOT` branch of `dryLightSample` — so it is point-like rather than
 * area-scaled, and a fixture that grew with the set without growing its
 * emission would light a large stage exactly as dimly as the distance made it.
 * Squaring the hang ratio holds the pool at one brightness across every scene,
 * which is what makes a single exposure correct for all of them.
 */
const LAMP_EMISSION_AT_REFERENCE_HANG = 75;
const LAMP_REFERENCE_HANG_M = 2.45;

/**
 * Half extent of the stage floor, and half its thickness.
 *
 * Wide enough that the falloff ring lands on the boards rather than running off
 * their edge, which is the whole composition: the lit pool needs a dark stage
 * around it to be a pool of anything.
 *
 * The slab's top lands on `y = 0` exactly, and that is load-bearing. The
 * voxeliser resolves a face flush with a lattice plane cleanly; offsetting it
 * by a fraction of a cell does not. A 4 mm drop — under a tenth of the 50 mm
 * lattice the reference tank is built on — put 49 isolated black pixels into
 * the lit pool, shading points whose baked normal disagreed with the neighbours
 * they sit between. They are a primary-surface artifact rather than a shadow
 * one: they survive `FLUID_SVO_DRY_SMOKE_CONE_TRACING=off`, which withholds
 * both secondary terms. Keep the face on the plane; only the buried underside
 * moves when the thickness scales.
 */
const FLOOR_HALF_PER_POOL_RADIUS = 1.9 / 1.05;
const FLOOR_HALF_THICKNESS_PER_POOL = 0.06 / 1.05;

/** The stage a container stands on, in metres. */
export interface StudioStageDimensions {
  readonly poolRadius_m: number;
  readonly floorHalf_m: number;
  readonly floorHalfThickness_m: number;
  readonly lampMouthHeight_m: number;
  readonly lampMouthRadius_m: number;
  readonly lampThroatRadius_m: number;
  readonly lampHalfHeight_m: number;
  readonly lampEmission: number;
  readonly stemRadius_m: number;
  readonly stemHalfHeight_m: number;
}

/**
 * The set, solved for one container.
 *
 * Exported because the framing depends on it: a camera that wants the falloff
 * ring in shot has to know where the ring is, and reading it from here is the
 * only way that stays true when a scene changes size.
 */
export function studioStageDimensions(scene: SceneDescription): StudioStageDimensions {
  // The wide pool where the budget can hold it, the solved pool where only
  // that fits. Both rungs honour the composition; only the bare fallback in
  // `SCENERY_GRAPHS` gives the set up entirely.
  const preferred = stageDimensionsAt(scene, POOL_RADIUS_PER_FOOTPRINT_PREFERRED);
  if (stageDomainBricks(scene, preferred) <= STUDIO_STAGE_DOMAIN_BRICK_BUDGET) return preferred;
  return stageDimensionsAt(scene, POOL_RADIUS_PER_FOOTPRINT_SOLVED);
}

function stageDimensionsAt(
  scene: SceneDescription,
  poolRadiusPerFootprint: number,
): StudioStageDimensions {
  const container = scene.container;
  const footprintRadius_m = 0.5 * Math.hypot(container.width_m, container.depth_m);
  const poolRadius_m = poolRadiusPerFootprint * footprintRadius_m;
  const lampMouthHeight_m = Math.max(
    LAMP_HANG_PER_POOL_RADIUS * poolRadius_m,
    container.height_m + LAMP_HEADROOM_PER_POOL_RADIUS * poolRadius_m,
  );
  const lampMouthRadius_m = LAMP_MOUTH_RADIUS_PER_POOL * poolRadius_m;
  const lampThroatRadius_m = LAMP_THROAT_RADIUS_PER_POOL * poolRadius_m;
  return {
    poolRadius_m,
    floorHalf_m: FLOOR_HALF_PER_POOL_RADIUS * poolRadius_m,
    floorHalfThickness_m: FLOOR_HALF_THICKNESS_PER_POOL * poolRadius_m,
    lampMouthHeight_m,
    lampMouthRadius_m,
    lampThroatRadius_m,
    // The taper that puts the beam's edge on `poolRadius_m` at the floor.
    lampHalfHeight_m: (lampMouthRadius_m - lampThroatRadius_m) * lampMouthHeight_m / (2 * poolRadius_m),
    lampEmission: LAMP_EMISSION_AT_REFERENCE_HANG * (lampMouthHeight_m / LAMP_REFERENCE_HANG_M) ** 2,
    stemRadius_m: LAMP_STEM_RADIUS_PER_POOL * poolRadius_m,
    stemHalfHeight_m: LAMP_STEM_HALF_HEIGHT_PER_POOL * poolRadius_m,
  };
}

/**
 * The set: a stage, a lamp over it, and the stem the lamp hangs from.
 *
 * The stem is not decoration. The fixture is emissive and therefore visible, and
 * a glowing can floating unsupported over a tank is the kind of detail that
 * makes a frame read as a render; one dark rod running up out of shot costs a
 * single primitive and answers the question.
 */
export function studioStageSceneryGraph(scene: SceneDescription): SceneryGraph {
  const stage = studioStageDimensions(scene);
  return {
    palettes: {
      stage: { tint: [1, 0.995, 0.985] },
      fixture: { tint: [0.94, 0.95, 1] },
      lamp: { tint: [1, 0.955, 0.88] },
    },
    nodes: [
      // No faces: this set has no room. See the note above, and
      // `SceneryRoomShellNode.faces` for why the empty list is meaningful.
      {
        kind: "room-shell", id: "shell", materialModel: "room", faces: [],
        floor: { palette: "stage", value: 0.62 },
        wall: { palette: "stage", value: 0.62 },
        ceiling: { palette: "stage", value: 0.62 },
      },
      {
        kind: "box", id: "stage/floor", group: "stage", tags: ["stage", "floor"],
        place: { position: { x: 0, y: -stage.floorHalfThickness_m, z: 0 }, units: "metres" },
        halfSize: { x: stage.floorHalf_m, y: stage.floorHalfThickness_m, z: stage.floorHalf_m },
        // Charcoal, not the mid grey a floor usually wants. The pool has to be
        // able to run forty times brighter than the stage around it without the
        // lit part clipping, and every stop of albedo the floor keeps is a stop
        // the lamp has to spend before the picture has any contrast in it.
        material: { palette: "stage", value: 0.12, surface: "plaster" },
      },
      {
        kind: "cylinder", id: "lamp/stem", group: "lamp", tags: ["lamp", "fixture"],
        place: {
          position: {
            x: 0,
            y: stage.lampMouthHeight_m + 2 * stage.lampHalfHeight_m + stage.stemHalfHeight_m,
            z: 0,
          },
          units: "metres",
        },
        radius: stage.stemRadius_m, halfHeight: stage.stemHalfHeight_m,
        material: { palette: "fixture", value: 0.16, surface: "brushed-metal" },
      },
      {
        // The beam *is* this taper: mouth at -Y, throat at +Y, so the light runs
        // down the local axis. `baseRadius` is the -Y cap by the primitive ABI's
        // own convention, which is why the wide radius is the base here.
        kind: "cone", id: "lamp/reflector", group: "lamp",
        tags: ["lamp", "fixture", "light", "spot-light"],
        place: {
          position: { x: 0, y: stage.lampMouthHeight_m + stage.lampHalfHeight_m, z: 0 },
          units: "metres",
        },
        baseRadius: stage.lampMouthRadius_m,
        topRadius: stage.lampThroatRadius_m,
        halfHeight: stage.lampHalfHeight_m,
        material: { colorLinear: [1, 0.955, 0.88], emission: stage.lampEmission },
      },
    ],
  };
}

/**
 * The framing the set was solved on: the reference 1.2 x 0.8 x 0.8 water box.
 *
 * High enough to see the water's surface, close enough that the tank fills the
 * frame and the falloff ring is the frame's only other subject. The lamp itself
 * is above the top edge at this distance, which is where a practical belongs —
 * present in the light, absent from the composition.
 *
 * The elevation is the framing decision. From 15 degrees the water is a slab
 * seen edge-on and the pool is a thin ellipse behind it; from 25 the surface
 * opens up, the pool wraps the tank's base on three sides, and settled water has
 * a top rather than only a front. Held down from there because the glass box is
 * most of the tank's height and looking further down turns it into a lid.
 *
 * A constant rather than a function of the container: a scene at another size
 * has its own subject and its own reason for a camera, and inheriting a framing
 * solved for a tank is how a paper figure ends up composed like a still life.
 */
export const studioStageCamera: Partial<CameraState> = {
  azimuth_rad: 0.62,
  elevation_rad: 0.44,
  distance_m: 1.95,
  target_m: { x: 0, y: 0.22, z: 0 },
};

/**
 * The domain, in solver bricks, that standing the set around a container costs.
 *
 * The set is authored in metres but paid for in *cells*, and on a wet scene
 * those are the solver's cells: `planSparseSceneDomain` unions the proxy AABBs
 * onto the solver lattice, and `buildSteps` notes that while a solver is
 * present every container brick is pinned at the solver level, so there is no
 * coarser lattice for environment geometry to descend into. A 0.8 m tank at
 * 6.25 mm and a 6.4 m tank at 50 mm therefore cost the set exactly the same,
 * and the physical size of the scene is not the variable — its lattice is.
 *
 * Counted as the whole spanned box rather than the occupied shell because that
 * is what the structures sized off the domain charge for: the node-mip opacity
 * pyramid and the unified payload arena are both cut from the span, and the air
 * column under a lamp hung high is as expensive to them as the boards are.
 */
export function studioStageDomainBricks(scene: SceneDescription): number {
  return stageDomainBricks(scene, studioStageDimensions(scene));
}

function stageDomainBricks(scene: SceneDescription, stage: StudioStageDimensions): number {
  const cellSize_m = scene.voxelDomain.finestCellSize_m;
  const brickSize = scene.voxelDomain.brickSize_cells;
  const spanXZ_m = 2 * stage.floorHalf_m;
  const spanY_m = stage.floorHalfThickness_m + stage.lampMouthHeight_m
    + 2 * stage.lampHalfHeight_m + 2 * stage.stemHalfHeight_m;
  const bricks = (span_m: number) => Math.max(1, Math.ceil(span_m / cellSize_m / brickSize));
  return bricks(spanXZ_m) ** 2 * bricks(spanY_m);
}

/**
 * How much domain a set may claim before it stops being worth having.
 *
 * Measured on the Dawn dry lane rather than reasoned about, because the failure
 * is not gradual. The reference water box spans ~1e3 bricks and renders in
 * 4.8 ms with a 1,426-page opacity pyramid. `ocean-seiche` — the same set
 * around an 8 m tank at 25 mm — asks for 79,507 pages, which the live radiance
 * atlas refuses; the pyramid is withdrawn, cone visibility falls back to exact
 * rays, and the frame goes to 31 ms. `cm12-figure-1` at 12.8 m asks for a
 * 12.7 GB payload arena against a 4 GB buffer limit and does not render at all.
 *
 * So the set is not something to scale until it breaks and then apologise for.
 * A scene above this line keeps the bare presentation it had before, which is
 * a worse picture and a working one; see `studioStageFits`.
 *
 * The number is deliberately below the first *measured* failure rather than
 * at it — the interval between 1e3 (renders well) and 8e4 (withdrawn) has not
 * been swept, and the cost of guessing high is a scene that does not open.
 */
export const STUDIO_STAGE_DOMAIN_BRICK_BUDGET = 32_768;

/**
 * Whether this container can afford to be staged.
 *
 * The composition is not negotiable downward: the pool has to cover the tank
 * corner to corner or the tank reads as badly lit, and the floor has to be
 * about 1.8 pool radii or the falloff ring runs off its edge and there is no
 * pool to speak of. A set trimmed until it fits a fine lattice is a floor
 * smaller than the tank standing on it. So the choice is the whole set or none
 * of it, and this is where that is decided.
 */
export function studioStageFits(scene: SceneDescription): boolean {
  return studioStageDomainBricks(scene) <= STUDIO_STAGE_DOMAIN_BRICK_BUDGET;
}
