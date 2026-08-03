import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { terrainHeightAt } from "./terrain";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import {
  bakePondVesselTerrain,
  pondVesselPlanCurve,
  pondVesselWaterline,
  type PondVesselSpec,
} from "./voxel-scenery/pond-vessel";
import { bonsaiNodes, BONSAI_POND_CANOPY } from "./voxel-scenery/bonsai";
import { sweptCopingNodes, sweptCopingSection, SWEPT_COPING_POND_BULLNOSE } from "./voxel-scenery/swept-coping";
import { stoneSet } from "./voxel-scenery/stone-set";

/**
 * The hero scene: a porcelain pond with a hose filling it.
 *
 * Reference: `output/imagegen/garden-pond-hose-fill-simplified.png`, and the
 * plan in `docs/HERO_GARDEN_HOSE_SCENE_PLAN.md`.
 *
 * This is the phase-0/1 body — the vessel, the water and the framing. The dry
 * set (pebble beds, mushroom-cap boulders, bonsai, air plants, hose geometry)
 * arrives in phase 2 and deliberately does not exist yet: the coping-as-terrain
 * bet is the one thing every later phase stands on, and it wants to be judged in
 * an empty frame rather than through a set that could flatter or hide it.
 *
 * Why the lattice is 25 mm, and why it wants to be 7.5 mm. The reference's two
 * most recognisable water features are the concentric ripple train and the
 * plunge column; the ripples read as ~4 % of the pond's width, so ~5 cm, and the
 * stream is ~4 cm across. At 25 mm both are within a cell or two of unresolvable
 * — this lattice can show that water is moving, not what shape the motion has.
 * 7.5 mm is where they land at ~6.7 and ~5 cells and the reference becomes
 * reachable rather than merely suggested.
 *
 * The scene therefore opens **dry**, and the water is opted into — see
 * `HeroGardenHoseOptions`. None of the walls below stand between opening this
 * scene and looking at it, because none of them are on the renderer's path.
 *
 * 25 mm is where the lattice sits because that is the finest measured to get this
 * scene through solver bring-up. Two walls were found above it, in this order:
 *
 *  - **7.5 mm** overruns a hard device limit. "Seed global fine bricks from
 *    every interface leaf" dispatches one workgroup per interface leaf on a
 *    single axis, and this scene asks for 65 536 against WebGPU's 65 535 ceiling.
 *    That is an engine shape, not a scene budget: no amount of art direction
 *    moves it, and a two-dimensional dispatch would.
 *  - **12.5 mm and 15 mm** refused the t = 0 pressure gate. **Fixed, and it was
 *    the engine, not the vessel.** The inner face was the wrong suspect: the
 *    SPGrid logical-page directory left every never-occupied page at its
 *    zero-fill, which is the legal index of physical page 0, so the coarse
 *    rows over this basin's empty regions resolved a stranger's page and the
 *    MGPCG rejected the publication (ERR_ROW stage 22). This pond has such
 *    rows and the tank scenes do not, which is the whole of why it was
 *    scene-specific. 25/15/12.5 mm all publish now.
 *
 * Every container dimension is a whole number of 8-cell bricks at every spacing
 * named here (0.2 m at 25 mm), so the sparse domain has no partial brick at any
 * wall.
 */

export const HERO_GARDEN_CONTAINER = { width_m: 1.8, height_m: 0.6, depth_m: 1.2 } as const;
export const HERO_GARDEN_CELL_M = 0.025;
export const HERO_GARDEN_BRICK_CELLS = 8 as const;

/**
 * How finely the vessel is baked, which is *not* the lattice the solver runs on.
 *
 * A quarter of a cell, so every solver column centre lands exactly on a grid
 * node: (k + 1/2) * 25 mm is 2 + 4k nodes out, and a bilinear fetch at a node is
 * the node. The ground the water meets is therefore the ground that was
 * generated, with no resampling error between them.
 *
 * The renderer gets nothing from it *yet*. `TerrainGrid` is currently a
 * CPU-and-solver feature: `terrainColumnHeights` bakes it into the octree's
 * height texture, but `cloneTerrain` (`lib/voxel-scene.ts`) drops the grid on
 * the way to the render path, and the dry scene's WGSL evaluates terrain from
 * `terrainMeta` plus eight analytic features in a uniform array with no grid
 * sampler anywhere. A sculpted vessel therefore draws as a flat plane at
 * `baseHeight_m`. The `terrainHeightfield` primitive kind exists in the SVO ABI,
 * with an `externalTerrain` flag and a `terrainReference` word — and no
 * producer. Closing that is the prerequisite for looking at this scene at all.
 */
export const HERO_GARDEN_TERRAIN_SAMPLE_M = HERO_GARDEN_CELL_M / 4;

/**
 * How far the still-water level sits below the plaster *outside* the pond.
 *
 * This is the number that keeps the water in the basin. A tank fill wets every
 * column the level clears, so a waterline above the outer ground floods the
 * whole 1.8 x 1.2 m floor with a two-cell film — which is not what the reference
 * shows, and which the sparse solver refuses outright: it publishes no
 * liquid-row frontier for a domain that is all film and no body.
 *
 * Expressed in cells rather than millimetres, because that is what the property
 * is actually about: the fill must not reach the outer ground *through rounding*
 * either, so the clearance has to survive the lattice it is sampled on. Under
 * one cell it does not, whatever the metre value says.
 */
export const HERO_GARDEN_WATER_BELOW_GROUND_M = Math.max(0.02, 1.6 * HERO_GARDEN_CELL_M);

/**
 * The vessel.
 *
 * `radius_m` is the coping's crest centreline, not the waterline: the water
 * meets the rim partway down its inner flank, so the wetted plan comes out a
 * little inside these numbers. Seven lobes at an 8 % wobble is enough wander
 * that no two quadrants of the rim repeat, and little enough that the outline
 * still reads as one deliberate curve rather than as a puddle.
 */
export const HERO_GARDEN_VESSEL: PondVesselSpec = Object.freeze({
  center_m: [0, 0] as const,
  radius_m: [0.52, 0.38] as const,
  groundHeight_m: 0.30,
  basinDepth_m: 0.155,
  rimHeight_m: 0.055,
  /**
   * The coping has left the heightfield, so this is no longer a coping width.
   *
   * With `crest: "flat"` the plaster runs level to the plan curve and the inner
   * face starts falling there, and the solid rim is set into that. The number
   * that belongs here is therefore the *solid's own footprint* — where its
   * flanks meet the ground — not the width of a crest that no longer exists.
   * Leaving it at 55 mm lays a 23 mm shelf of dry plaster inside the rim, which
   * the reference does not have and which the pebble courses would then bed
   * against, since they offset from this same number.
   */
  rimHalfWidth_m: sweptCopingSection(SWEPT_COPING_POND_BULLNOSE).groundHalfWidth_m,
  /**
   * The crest is a swept solid now, not a swelling of the ground.
   *
   * The reason is the *meeting*, not the crown. `pondVesselCrestProfile` is
   * tangent at its foot by construction, so a heightfield rim leaves the plaster
   * with a continuous normal — measured at 0.24 degrees from vertical at the
   * foot — and there is simply no line where the two meet. The reference has a
   * hard one, and a union of a solid with the ground gives it for free, along
   * with the roll-back under the lip that no function of (x, z) can express.
   *
   * What this does *not* change is containment, which is the property that made
   * the vessel terrain in the first place: the still waterline sits 40.9 mm below
   * the lowest plaster outside the rim, and omitting the crest moves that number
   * by less than a nanometre. The wall the water rests against was always the
   * outer ground, never the crest — a coping is by definition the part above the
   * waterline. That is pinned by a test rather than left as an argument.
   */
  crest: "flat",
  // The inner face is a wall, not a beach. At 0.16 m of run for 0.155 m of drop
  // the basin was a 45-degree dish, and a third of the pond's 1.04 m width was
  // spent on the slope — which is why the first renders read as a crater in
  // plaster rather than as a vessel with sides. The reference drops almost
  // vertically from the coping's inner foot: the water meets the rim and the
  // wall goes straight down behind it. Three centimetres of run is as near to
  // vertical as a heightfield gets before the bilinear sample spacing, not the
  // authored profile, is what sets the slope.
  innerFace_m: 0.035,
  // …except on the left, where the reference has a shore. The stepping discs
  // wade in over this sector and the pebble courses run down into the water on
  // it; everywhere else the pond stays a wall. Centred on the bearing of the
  // near-left terrace below, so the boulder group, the beach and the disc path
  // are three consequences of one decision about where the near bank is rather
  // than three separately placed things that have to be kept in agreement.
  beach: { turn: 0.452, width: 0.17, innerFace_m: 0.30 },
  lobes: 7,
  wobble: 0.08,
  // The coping swells and narrows as it runs, which is most of what separates a
  // formed rim from an extruded one.
  sectionHeightVariation: 0.16,
  sectionWidthVariation: 0.22,
  relief_m: 0.0025,
  seed: 0x9a7de11,
  terraces: [
    // The plateau the bonsai stands on, at the back right.
    { center_m: [0.62, 0.30] as const, radius_m: [0.44, 0.34] as const, height_m: 0.075, rotation_rad: -0.35, flat: 0.30 },
    // A lower step at the near left, where the boulder group beds in.
    { center_m: [-0.64, -0.20] as const, radius_m: [0.36, 0.30] as const, height_m: 0.042, rotation_rad: 0.25, flat: 0.28 },
  ],
});

export const HERO_GARDEN_WATERLINE_M = pondVesselWaterline(HERO_GARDEN_VESSEL, HERO_GARDEN_WATER_BELOW_GROUND_M);

/**
 * Close, and looking down at about twenty-five degrees, so the pond fills the
 * frame the way it does in the reference.
 *
 * The first framing sat at 0.30 rad and 1.62 m, which put the eye 78 cm up and
 * more than a metre back: from there the basin was a slot seen edge-on, the
 * coping's far arc lay flat against its own near arc, and a third of the image
 * was horizon. None of that is in the reference, which is a close
 * three-quarter view with no horizon at all — the pond spans nearly the whole
 * width and the ground behind it is soft and out of focus.
 *
 * The distance is set against the aperture, and the aperture is the awkward
 * part. The dry shader builds its primary rays with a fixed vertical half-tangent
 * of 0.72 — about 72 degrees vertical, and past 100 degrees horizontal at this
 * frame's aspect. That is a much wider lens than the reference, which shows
 * little enough perspective to read as something like a 50 mm. Filling the frame
 * the way the reference does therefore means standing under a metre away, and
 * the near coping grows against the far one as the price. Just under a metre is
 * where the pond commands the frame without the near arc bowing.
 *
 * Getting the rest of the way is a renderer change, not a scene one: the camera
 * has no aperture to author, so `0.72` would have to become one. Worth doing —
 * but it is a shared constant in both the dry shader and the pixel-trace mirror,
 * and it is the sort of change that wants the frame otherwise settled first.
 */
/**
 * Where the camera stands, and — because the light is derived from it — where
 * the sun is.
 *
 * Solved against the set rather than chosen by eye. `cameraPosition` puts the
 * eye at `target + d(sin az, ·, cos az)`, so screen right is `(cos az, -sin az)`
 * and depth is `-(x sin az + z cos az)`. Requiring the bonsai and the hose to
 * land far-right while the boulder group, the shore and the stepping stones stay
 * near-left — which is the reference's whole arrangement — is four inequalities
 * in one unknown, and they hold together only between about 5.1 and 5.7.
 *
 * The middle of that window is the part that matters: it means the framing
 * survives the set being moved rather than balancing on one number. An earlier
 * pass put this at 2.4 by deriving screen right as `(sin az, -cos az)` — the
 * other common convention, and a half-turn out — which stood the tree between
 * the camera and its own pond.
 */
export const HERO_GARDEN_AZIMUTH_RAD = 5.4;

/** Screen right in the XZ plane at that azimuth: `(cos az, -sin az)`. */
const HERO_GARDEN_SCREEN_RIGHT = [Math.cos(HERO_GARDEN_AZIMUTH_RAD), -Math.sin(HERO_GARDEN_AZIMUTH_RAD)] as const;

export const heroGardenCamera: Partial<CameraState> = {
  azimuth_rad: HERO_GARDEN_AZIMUTH_RAD,
  elevation_rad: 0.40,
  distance_m: 1.05,
  target_m: { x: 0.02, y: 0.30, z: 0.0 },
};

/**
 * The sun, as a bearing in the camera's frame rather than in the world's.
 *
 * Three quarters of screen right, lifted to `y = 0.62` — about forty degrees —
 * is a sun over the camera's right shoulder, which is where the reference's is.
 * Deriving it from the same azimuth the camera uses means re-aiming the camera
 * re-aims the light with it, and the set cannot be relit into silhouettes by a
 * framing change alone.
 *
 * The default key is `[-0.45, 0.86, 0.28]`: nearly overhead, and from the left
 * of this camera. That is the one arrangement that flattens a set of round pale
 * objects into silhouettes, which is most of why the early frames read as grey.
 */
const HERO_GARDEN_KEY_DIRECTION: readonly [number, number, number] = [
  0.75 * HERO_GARDEN_SCREEN_RIGHT[0],
  0.62,
  0.75 * HERO_GARDEN_SCREEN_RIGHT[1],
];

/**
 * Where the water leaves the hose, and how fast.
 *
 * One mouth and one aim, shared by the jet the solver injects and the tube the
 * renderer draws. They are the same three numbers on purpose: a hose whose
 * nozzle is anywhere but where the water actually appears is the single most
 * obvious way this scene could look wrong, and deriving both from here means it
 * cannot happen by drift.
 */
export const HERO_GARDEN_HOSE_MOUTH_M = Object.freeze({ x: 0.28, y: 0.44, z: 0.06 });
const HOSE_AIM = Object.freeze({ x: -0.252, y: -0.966, z: -0.050 });
const HOSE_SPEED_M_S = 1.19;
const HOSE_BORE_M = 0.013;

/**
 * The hose's run, from the nozzle backwards, in world metres.
 *
 * The first point off the mouth is placed along the aim, so the last hand's
 * breadth of tube points where the water goes. The rest lifts over the back
 * terrace and leaves frame to the right.
 *
 * Authored as the polyline it is rather than as placed beads: a capsule already
 * takes the two ends of the run it follows, which is what a hose is described
 * by. When phase 2 brings in the `swept-tube` generator this becomes its input
 * unchanged.
 */
const HOSE_PATH_M: readonly (readonly [number, number, number])[] = Object.freeze([
  [HERO_GARDEN_HOSE_MOUTH_M.x, HERO_GARDEN_HOSE_MOUTH_M.y, HERO_GARDEN_HOSE_MOUTH_M.z],
  [HERO_GARDEN_HOSE_MOUTH_M.x + HOSE_AIM.x * -0.09, HERO_GARDEN_HOSE_MOUTH_M.y + HOSE_AIM.y * -0.09, HERO_GARDEN_HOSE_MOUTH_M.z + HOSE_AIM.z * -0.09],
  [0.40, 0.585, 0.09],
  [0.55, 0.600, 0.135],
  [0.72, 0.585, 0.195],
  [0.95, 0.545, 0.27],
]);

/**
 * The set, which for now is the ground and the hose.
 *
 * A terrain shell publishes no boxes — the heightfield it stands for is already
 * the authority — so a pond whose every surface is terrain publishes nothing at
 * all, and the SVO's candidate index requires a scene to contain at least one
 * primitive. The hose is the right thing to arrive first regardless: it is the
 * scene's subject, and it is what makes the jet legible as something a person
 * is doing rather than water appearing out of the air.
 *
 * The hose is also the one deliberate departure from the house monochrome rule.
 * Every other surface in this set reads its form from shading alone; the
 * reference's dark slate-teal tube is the single object carrying hue, and it is
 * carrying it on purpose — the same licence the sets already grant a lantern
 * ember, spent on the one object here that is manufactured rather than grown.
 */
const HOSE_MATERIAL = { colorLinear: [0.045, 0.085, 0.082] as const };
const FERRULE_MATERIAL = { colorLinear: [0.29, 0.275, 0.235] as const };

function hoseNodes(): SceneryNode[] {
  const nodes: SceneryNode[] = HOSE_PATH_M.slice(0, -1).map((from, index) => ({
    kind: "capsule",
    id: `hose/run-${index}`,
    group: "hose-tube",
    tags: ["hose"],
    place: { units: "metres" },
    from: { x: from[0], y: from[1], z: from[2] },
    to: { x: HOSE_PATH_M[index + 1][0], y: HOSE_PATH_M[index + 1][1], z: HOSE_PATH_M[index + 1][2] },
    radius: HOSE_BORE_M,
    material: HOSE_MATERIAL,
  }));
  // The collar, as a short fat capsule rather than a cone, so it is authored by
  // the run it sits on and needs no orientation solved for it.
  nodes.push({
    kind: "capsule",
    id: "hose/ferrule",
    group: "metal-ferrule",
    tags: ["hose", "ferrule"],
    place: { units: "metres" },
    from: HERO_GARDEN_HOSE_MOUTH_M,
    to: {
      x: HERO_GARDEN_HOSE_MOUTH_M.x - HOSE_AIM.x * 0.035,
      y: HERO_GARDEN_HOSE_MOUTH_M.y - HOSE_AIM.y * 0.035,
      z: HERO_GARDEN_HOSE_MOUTH_M.z - HOSE_AIM.z * 0.035,
    },
    radius: 0.017,
    material: FERRULE_MATERIAL,
  });
  return nodes;
}

/**
 * The seed the whole dry set is grown from.
 *
 * One number, because the point of a generated set is that "give me another
 * garden" is a re-seed rather than a re-author. Changing it moves every stone
 * and re-packs every bed; it does not move the pond, which has a seed of its
 * own so that the vessel can be pinned while the set is still being explored.
 */
export const HERO_GARDEN_SET_SEED = 0x5701_e5;

/**
 * Where the bonsai stands, and which way it leans.
 *
 * On the back-right terrace, and leaning back toward the pond's centre so the
 * crown overhangs the water the way the reference's does. The lean is the
 * negated stand position for exactly that reason — the tree is told to reach
 * for the middle, not given a bearing that has to be kept in agreement with one.
 */
const BONSAI_AT_M = [0.60, 0.26] as const;

/**
 * The set, grown against the ground it stands on.
 *
 * A function rather than a constant because the species take a ground query, not
 * a heightfield of their own — see `lib/voxel-scenery/README.md`. Passing the
 * baked grid's own sampler is what stops a root or a pebble from being seated
 * against a surface subtly different from the one the renderer draws and the
 * solver collides with.
 */
function heroGardenScenery(groundHeightAt: (x_m: number, z_m: number) => number): SceneryGraph {
  return {
    palettes: {
      clay: { tint: [1, 0.985, 0.955] },
      stone: { tint: [0.972, 0.984, 1] },
    },
    nodes: [
      // Porcelain, not lawn. The vessel and the plaster it is set into are one
      // fired surface in the reference; the garden closure would band them by
      // height into liner, soil and grass and speckle the lot with daisies.
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
      ...hoseNodes(),
      // The stones place themselves off the pond's own plan curve rather than off
      // world coordinates, so moving a control point in the vessel spec carries
      // the boulders, the beds and the disc path with it. That is the whole reason
      // the curve is an export.
      // The rim, as a solid swept along the same plan curve the basin was cut
      // from. Forty-eight samples per lobe rather than the plan's own sixteen,
      // and that number is load-bearing: a cone chain's *outline* is settled at
      // sixteen (the chord sags 0.18 mm), but its shading normal steps by the
      // whole turn angle at every joint, so at sixteen the rim comes back banded
      // like a caterpillar. Three degrees per joint bands, one degree does not.
      // The rail's resolution is set by the highlight, not by the silhouette —
      // which is also the honest argument for a marched swept-arc primitive,
      // where one record spans an arc and there are no joints to shade across.
      ...sweptCopingNodes({
        ...SWEPT_COPING_POND_BULLNOSE,
        key: "coping",
        rail: pondVesselPlanCurve(HERO_GARDEN_VESSEL, 48),
        groundHeightAt,
        material: { palette: "stone", value: 0.92 },
        seed: HERO_GARDEN_SET_SEED ^ 0x00c0_9179,
      }),
      ...stoneSet({
        vessel: HERO_GARDEN_VESSEL,
        waterline_m: HERO_GARDEN_WATERLINE_M,
        seed: HERO_GARDEN_SET_SEED,
      }),
      ...bonsaiNodes({
        ...BONSAI_POND_CANOPY,
        key: "bonsai",
        at_m: BONSAI_AT_M,
        groundHeightAt,
        lean: [-BONSAI_AT_M[0], -BONSAI_AT_M[1]],
        seed: HERO_GARDEN_SET_SEED ^ 0x8017a1,
      }),
    ],
  };
}

/** The hose, as the solver sees it: a round jet leaving the nozzle along its aim. */
function heroGardenInflow(): SceneDescription["fluid"]["inflow"] {
  return {
    center_m: { ...HERO_GARDEN_HOSE_MOUTH_M },
    radius_m: 0.021,
    length_m: 0.05,
    velocity_m_s: {
      x: HOSE_AIM.x * HOSE_SPEED_M_S,
      y: HOSE_AIM.y * HOSE_SPEED_M_S,
      z: HOSE_AIM.z * HOSE_SPEED_M_S,
    },
    start_s: 0,
    end_s: 120,
    ramp_s: 0.35,
  };
}

export interface HeroGardenHoseOptions {
  /**
   * Whether the fluid solver owns this document. Off by default.
   *
   * The set is the part of this scene that is ready to be looked at, and the
   * water is the part still in bring-up — so the default is the one that always
   * opens. With `systems.fluid === false` the renderer attaches the live sparse
   * scene directly: no solver, no t=0 authority fence, no transport gate, and
   * none of the three walls in the header above are on the path to a frame.
   *
   * The water is *authored* either way. The fill, the initial condition and the
   * jet stay on the document because they describe a pond that exists whether or
   * not it is being solved this second; the flag alone decides. That is what
   * makes this one scene with a switch rather than two scenes that will drift.
   */
  readonly water?: boolean;
}

export function createHeroGardenHoseScene(options: HeroGardenHoseOptions = {}): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "hero-garden-hose";
  scene.systems = { ...scene.systems, fluid: options.water === true };
  scene.container.width_m = HERO_GARDEN_CONTAINER.width_m;
  scene.container.height_m = HERO_GARDEN_CONTAINER.height_m;
  scene.container.depth_m = HERO_GARDEN_CONTAINER.depth_m;
  scene.container.top = "open";
  scene.container.vessel = "none";
  scene.voxelDomain = { finestCellSize_m: HERO_GARDEN_CELL_M, brickSize_cells: HERO_GARDEN_BRICK_CELLS };
  const terrain = {
    baseHeight_m: HERO_GARDEN_VESSEL.groundHeight_m,
    features: [],
    grid: bakePondVesselTerrain(HERO_GARDEN_VESSEL, HERO_GARDEN_CONTAINER, HERO_GARDEN_TERRAIN_SAMPLE_M),
  };
  scene.terrain = terrain;
  scene.container.fillFraction = HERO_GARDEN_WATERLINE_M / HERO_GARDEN_CONTAINER.height_m;
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.inflow = heroGardenInflow();
  scene.rigidBodies = [];
  /**
   * The light, aimed at the frame rather than at the world.
   *
   * The reference is one hard warm sun from the upper right with a very bright
   * bounce under it — the shadows have direction and softness but almost no
   * depth, because everything they fall on is white and throws the light back.
   * The default key sits at [-0.45, 0.86, 0.28]: nearly overhead, and from the
   * *left* of this camera, which is the one arrangement that flattens a set of
   * round pale objects into silhouettes.
   *
   * So the direction is derived from the camera's own basis instead of authored
   * as three numbers. Screen right at azimuth `az` is `(sin az, 0, -cos az)`;
   * this is that vector at three-quarters strength, lifted to about 40 degrees.
   * The sun therefore rakes across the set from the upper right of frame, which
   * is what puts a lit edge on the near side of every boulder and a soft shadow
   * running to the left of it.
   */
  scene.lighting = {
    ...scene.lighting,
    directional: {
      direction: HERO_GARDEN_KEY_DIRECTION,
      // Warm, and strong enough that the plaster runs close to white without
      // clipping — the reference is a high-key image, and a mid-grey pond reads
      // as concrete however correct its geometry is.
      colorLinear: [1, 0.965, 0.90],
      intensity: 1.55,
    },
    // The bounce. White ground under white objects returns a great deal, and
    // the reference's shadow interiors are barely a stop below its highlights.
    environment: { diffuseScale: 1.35, specularScale: 1.0 },
  };
  // The set is seated on the grid that was just baked, not on the generator that
  // produced it: the two agree at every grid node by construction and differ
  // between them by the bilinear fetch, and the ground a root meets should be
  // the one the ray actually hits.
  scene.scenery = heroGardenScenery((x_m, z_m) => terrainHeightAt(terrain, x_m, z_m));
  return scene;
}
