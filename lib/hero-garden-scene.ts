import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import {
  bakePondVesselTerrain,
  pondVesselWaterline,
  type PondVesselSpec,
} from "./voxel-scenery/pond-vessel";
import { BONSAI_POND_CANOPY } from "./voxel-scenery/bonsai";
import { sweptCopingSection, SWEPT_COPING_POND_BULLNOSE } from "./voxel-scenery/swept-coping";
import { stoneSetBoulderStations, stoneSetSteppingBodies } from "./voxel-scenery/stone-set";
import { ROSETTE_AIR_PLANT, ROSETTE_GRASS_TUFT, type RosetteForm } from "./voxel-scenery/rosette";
import { sweptTubeNodes } from "./voxel-scenery/swept-tube";
import { tanHalfFovFor35mmFocalLength } from "./webgpu-camera";

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
 * The renderer sees it too, which it did not always. `TerrainGrid` was once a
 * CPU-and-solver feature — the dry scene's WGSL evaluated terrain from
 * `terrainMeta` plus eight analytic features with no grid sampler anywhere, so
 * a sculpted vessel drew as a flat plane at `baseHeight_m` and this scene could
 * not be looked at at all. `packSvoDrySceneTerrainHeightfield` is the producer
 * the `terrainHeightfield` primitive kind was waiting for, and the ray now hits
 * the surface that was generated here. Sample spacing is therefore a *visual*
 * decision as well as a solver one.
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

/**
 * The lens: a 50 mm, which on the 36 x 24 mm frame is a vertical half-tangent
 * of 0.24 — 27 degrees vertical and 46 horizontal at the studio's 16:9.
 *
 * The renderer's default is 0.72: 71 degrees vertical, 104 horizontal, an
 * ultra-wide by any reckoning and about a 17 mm. The reference is not that. It
 * shows a pond a metre across with the far coping only a little smaller than
 * the near one, which is a normal lens at a couple of metres and nothing else.
 *
 * Fifty rather than something longer because the reference does carry visible
 * perspective — the basin is read as a bowl, not as a disc — and because a
 * longer lens would put the eye far enough back that the set's own terraces
 * start occluding the shore.
 */
const HERO_GARDEN_TAN_HALF_FOV = tanHalfFovFor35mmFocalLength(50);

/**
 * Where the camera stands.
 *
 * Solved against the set, not against the pond. Three quantities move together
 * as the distance changes, and only one of them is a free choice:
 *
 * - The pond's outer plan is 1.10 x 0.82 m, so at this azimuth its silhouette
 *   is 0.473 m of half-extent across screen right. At 1.40 m it reaches both
 *   side crops instead of sitting inside a field of empty plaster.
 * - The bonsai's crown is the highest thing in the set, and it is deliberately
 *   cut by the top edge. The reference is a crop, not a catalog
 *   view; keeping every generated floret inside the picture made the subject
 *   too small.
 * - The near-left boulder terrace approaches the left edge while the near
 *   coping approaches the bottom. Those two near cuts are what make the pond
 *   read as a place the camera is standing over rather than a miniature placed
 *   at the centre of a ground plane.
 *
 * What the longer lens buys is controlled perspective. Closing from 2.70 m to
 * 1.40 m spends the safety margin the old framing reserved around the set; the
 * 50 mm aperture keeps the near coping from acquiring an ultra-wide bow.
 */
export const heroGardenCamera: Partial<CameraState> = {
  azimuth_rad: HERO_GARDEN_AZIMUTH_RAD,
  elevation_rad: 0.40,
  distance_m: 1.40,
  tanHalfFov: HERO_GARDEN_TAN_HALF_FOV,
  target_m: { x: -0.04, y: 0.35, z: 0.0 },
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
export const HERO_GARDEN_HOSE_MOUTH_M = Object.freeze({ x: 0.27, y: 0.40, z: 0.22 });
/** The visible plunge point, so nozzle placement and jet composition share one authority. */
export const HERO_GARDEN_HOSE_IMPACT_M = Object.freeze({ x: 0.20, y: HERO_GARDEN_WATERLINE_M, z: 0.07 });
const HOSE_AIM_LENGTH = Math.hypot(
  HERO_GARDEN_HOSE_IMPACT_M.x - HERO_GARDEN_HOSE_MOUTH_M.x,
  HERO_GARDEN_HOSE_IMPACT_M.y - HERO_GARDEN_HOSE_MOUTH_M.y,
  HERO_GARDEN_HOSE_IMPACT_M.z - HERO_GARDEN_HOSE_MOUTH_M.z,
);
const HOSE_AIM = Object.freeze({
  x: (HERO_GARDEN_HOSE_IMPACT_M.x - HERO_GARDEN_HOSE_MOUTH_M.x) / HOSE_AIM_LENGTH,
  y: (HERO_GARDEN_HOSE_IMPACT_M.y - HERO_GARDEN_HOSE_MOUTH_M.y) / HOSE_AIM_LENGTH,
  z: (HERO_GARDEN_HOSE_IMPACT_M.z - HERO_GARDEN_HOSE_MOUTH_M.z) / HOSE_AIM_LENGTH,
});
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
  [0.44, 0.470, 0.34],
  [0.60, 0.460, 0.40],
  [0.80, 0.460, 0.50],
  [1.02, 0.460, 0.60],
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
const HOSE_MATERIAL = { colorLinear: [0.045, 0.085, 0.082] as const, surface: "organic" as const };
const FERRULE_MATERIAL = { colorLinear: [0.29, 0.275, 0.235] as const, surface: "brushed-metal" as const };

function hoseNodes(): SceneryNode[] {
  // One swept-tube record family over the path as authored, rather than a
  // capsule per segment: the reference's hose is a continuous curve and a chain
  // of capsules creases its shading normal at every bend. `sweptTubeNodes`
  // fits the envelopes, so the only thing decided here is the path and the bore.
  const nodes: SceneryNode[] = sweptTubeNodes({
    id: "hose/tube",
    group: "hose-tube",
    tags: ["hose"],
    stations: HOSE_PATH_M.map(([x, y, z]) => ({ at_m: { x, y, z }, radius_m: HOSE_BORE_M })),
    material: HOSE_MATERIAL,
  });
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
 * On the far-right bank, and leaning back toward the pond's centre so the
 * crown overhangs the water the way the reference's does. The lean is the
 * negated stand position for exactly that reason — the tree is told to reach
 * for the middle, not given a bearing that has to be kept in agreement with one.
 */
const BONSAI_AT_M = [0.55, -0.15] as const;

/**
 * This specimen's crown, which is flatter and slightly narrower than the
 * species'.
 *
 * The reference's canopy is a *cloud layer*: it reads as a broad horizontal
 * plate hanging over the water, and the single number that decides whether it
 * does is the crown's width against its own thickness. The species stands at
 * 5.5:1 and this asks for 5.8:1 on a marginally smaller crown — broad enough to
 * overhang the basin, thin enough not to read as a dome on a stick.
 *
 * It is an override rather than a species edit because the species is also the
 * catalogue specimen, which is looked at from a wider frame where a plate this
 * thin reads as a disc. Kept in step with the species by hand: the aspect is the
 * thing to preserve if either moves, not the absolute numbers.
 */
const HERO_BONSAI_FORM = Object.freeze({
  ...BONSAI_POND_CANOPY,
  crownRadius_m: [0.36, 0.31] as const,
  crownThickness_m: 0.125,
});

/**
 * The set, as a description of itself.
 *
 * This used to be a *composition step*: a function that took a ground query,
 * called `sweptCopingNodes`, `stoneSet` and `bonsaiNodes`, and splatted their
 * output into the node list. What the document then held was 684 baked nodes and
 * 884 kB of ellipsoid centres — the seeds and the parameters gone, "give me
 * another garden" a re-run of this factory that discards every edit the user had
 * made, and every `cloneScene` a copy of the lot. The three nodes below say the
 * same thing in 1.7 kB and publish the same 2 197 primitives in the same order.
 * See lib/scenery-generators.ts.
 *
 * The vessel is named rather than repeated. All three generators lay their runs
 * against the same plan curve — that is why moving a control point carries the
 * rim, the beds and the disc path with it — and a curve that lived on each node
 * would be three copies with three chances to disagree.
 *
 * No ground query is passed. The species still need one and still get one; it
 * arrives from the expansion context, which is the scene's own baked grid, so a
 * root or a pebble is still seated against exactly the surface the renderer
 * draws and the solver collides with.
 */

/**
 * Where the boulder-family plant stands: derived, not authored.
 *
 * The near plant belongs *to the boulder group* — it is the tuft that has taken
 * hold in the gravel heaped at their feet, and that reading is the whole reason
 * it carries a wider `bed_m` than the other two. So it is solved from the
 * family's own stations rather than given a world coordinate beside them.
 *
 * That is not tidiness. This constant used to hold `[-0.60, -0.26]`, and when
 * the stone set re-solved the family's position against the camera the plant
 * stayed behind and stood alone on an empty bank — a coordinate restating
 * another module's answer is a copy that drifts the first time the original
 * moves. `stoneSetSteppingStations` is read the same way by the layout for the
 * same reason.
 *
 * It stands off the outboard end of the family rather than among it, clear by
 * half the outermost stone's own width, so a three-millimetre rosette is never
 * inside a boulder it cannot be seen against.
 */
const HERO_BOULDER_PLANT_CLEARANCE = 0.5;
function heroBoulderPlantStand(): readonly [number, number] {
  const family = stoneSetBoulderStations({
    vessel: HERO_GARDEN_VESSEL,
    waterline_m: HERO_GARDEN_WATERLINE_M,
    seed: HERO_GARDEN_SET_SEED,
  });
  // The outboard end is the station furthest from the pond's centre in x, which
  // is the direction the family runs; standing beyond it keeps the plant on the
  // open bank instead of between two stones.
  const outboard = family.reduce((a, b) => (a.at_m[0] <= b.at_m[0] ? a : b));
  return [
    outboard.at_m[0] - HERO_BOULDER_PLANT_CLEARANCE * outboard.width_m,
    outboard.at_m[1] + HERO_BOULDER_PLANT_CLEARANCE * outboard.width_m,
  ];
}

/**
 * Where the plants stand.
 *
 * All three sit *off* the coping rather than on it: a rosette's blades are
 * three millimetres across and the rim is the one surface in this scene the eye
 * follows as a continuous line, so a tuft on the crest reads as damage to the
 * casting. `bed_m` is the little of the ground each one has disturbed under
 * itself, and the boulder-group plant has more of it because it is bedded into
 * that group's own gravel.
 */
const HERO_PLANT_STANDS: readonly {
  readonly id: string;
  readonly form: RosetteForm;
  readonly at_m: readonly [number, number];
  readonly bed_m?: number;
  readonly salt: number;
}[] = Object.freeze([
  { id: "plant/boulder-group", form: ROSETTE_AIR_PLANT, at_m: heroBoulderPlantStand(), bed_m: 0.035, salt: 0x00a1_7e01 },
  { id: "plant/terrace-back", form: ROSETTE_AIR_PLANT, at_m: [0.44, 0.42] as const, bed_m: 0.028, salt: 0x00a1_7e02 },
  { id: "plant/rim-right", form: ROSETTE_GRASS_TUFT, at_m: [0.72, -0.32] as const, bed_m: 0.024, salt: 0x00a1_7e03 },
]);

function heroGardenScenery(): SceneryGraph {
  return {
    palettes: {
      clay: { tint: [1, 0.985, 0.955] },
      stone: { tint: [0.972, 0.984, 1] },
    },
    vessels: { pond: HERO_GARDEN_VESSEL },
    nodes: [
      /**
       * Porcelain, not lawn — and it is the whole set that is porcelain, not
       * only the ground.
       *
       * The immediate reason is the ground: the vessel and the plaster it is set
       * into are one fired surface in the reference, and the garden closure would
       * band them by height into liner, soil and grass and speckle the lot with
       * daisies. But this node is also the scene's only declaration of what it is
       * *made of*, so `svoMaterialFunctionIdForEnvironmentProxy` reads it for
       * every prop as well — see `lib/svo-material-abi.ts`. A boulder here is not
       * granite that happens to be pale; it is the same fired white clay as the
       * rim it is bedded against, and it would be granite by default only because
       * its group carries the word "stone". One word decides for the whole set,
       * which is the only way a monochrome set can be kept monochrome as
       * generators come and go.
       */
      { kind: "terrain-shell", id: "shell", materialModel: "porcelain" },
      ...hoseNodes(),
      // The rim, as a solid swept along the same plan curve the basin was cut
      // from.
      {
        kind: "generator", id: "coping", generator: "swept-coping", vessel: "pond",
        seed: HERO_GARDEN_SET_SEED ^ 0x00c0_9179,
        params: { ...SWEPT_COPING_POND_BULLNOSE, material: { palette: "stone", value: 0.92, surface: "stone" } },
      },
      // Boulders, beds and the wading path, all placed in the vessel's own
      // coordinates — a fraction of a turn round the plan curve and a plan
      // distance either side of it — rather than in world metres that could
      // drift from the coping they hug.
      {
        kind: "generator", id: "stone", generator: "pond-stone-set", vessel: "pond",
        seed: HERO_GARDEN_SET_SEED,
        params: { waterline_m: HERO_GARDEN_WATERLINE_M },
      },
      {
        kind: "generator", id: "bonsai", generator: "bonsai",
        seed: HERO_GARDEN_SET_SEED ^ 0x8017a1,
        params: { ...HERO_BONSAI_FORM, at_m: BONSAI_AT_M, lean: [-BONSAI_AT_M[0], -BONSAI_AT_M[1]] },
      },
      // The plants. Built and tested for this scene and then never called by
      // it, which is the failure mode a generator catalog exists to make
      // visible: a species with no node in any document is a species nobody can
      // tell is missing.
      //
      // Two air plants where the reference has them — one wedged against the
      // boulder group on the near-left terrace, one on the back-right plateau
      // behind the tree's stand — and one grass tuft at the near-right rim,
      // which is the only place in frame with a run of plaster wide enough to
      // stand something on without crowding the coping.
      ...HERO_PLANT_STANDS.map(({ id, form, at_m, bed_m, salt }): SceneryNode => ({
        kind: "generator", id, generator: "rosette",
        seed: HERO_GARDEN_SET_SEED ^ salt,
        params: { ...form, at_m, ...(bed_m === undefined ? {} : { bed_m }) },
      })),
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
  /**
   * The teal, as a property of *this* pond.
   *
   * Absorption is a rate per metre, and the frozen clean-water table it
   * defaults to is calibrated on tanks: at this pond's 115 mm of still depth it
   * transmits (0.950, 0.990, 0.993), so red is 4.6 % down on green and the
   * water carries no hue a viewer could name. The reference's pond is
   * unmistakably teal at the same depth, and no amount of shader tuning can
   * produce that from a coefficient an order of magnitude too small — Beer's
   * law is the only thing between the two, and it has one input.
   *
   * So these are pond coefficients, not water ones. Red goes up 5.8x, green
   * 4.7x and blue 10x: the last is what makes it *teal* rather than *blue*,
   * because dissolved organics and algae absorb in the blue where pure water
   * barely does, and a pond that attenuates blue least ends up the colour of a
   * swimming pool. What the numbers buy, in transmission:
   *
   *   still depth, 0.115 m   (0.742, 0.953, 0.931)  — red 22 % down on green,
   *                                                   against 4.6 % before
   *   eye path,    0.296 m   (0.463, 0.883, 0.832)  — the slant through the
   *                                                   basin at this camera
   *   sun path,    0.141 m   (0.693, 0.942, 0.916)  — refracted to 35 degrees
   *                                                   off vertical
   *
   * Round trip, the basin floor keeps 32 % of its red against 83 % of its
   * green. That is the whole of the pond's colour, and it comes from the medium
   * rather than from a tint added at the end.
   *
   * The in-scatter goes up with it, and by the same factor rather than by an
   * independently chosen one: `unifiedAbsorbingTransmission` blends toward the
   * scatter colour as the transmission falls, so a body that now extinguishes
   * three times as much light needs three times the scatter to stay luminous
   * instead of going to ink. Three times the default, to within 5 % per channel.
   */
  scene.fluid.optics = {
    absorption_mInv: [2.6, 0.42, 0.62],
    scatter: [0.038, 0.165, 0.145],
  };
  /**
   * The stepping discs, as solids the water has to part around.
   *
   * The path is scenery too — the drawn plate is a tapered footing, a tread and
   * a crown that softens its edge, and that silhouette is what stops five discs
   * reading as five drums — but scenery has no solver term, so a disc that is
   * *only* scenery is a disc the jet pours straight through. These are the same
   * five stones, solved once by `heroSteppingPath` and published twice: the
   * generator node draws them, and this collides them. Neither derives the
   * contour again, which is the only way the two can be guaranteed to agree.
   *
   * The collider is a single static cylinder inscribed in the drawn solid — see
   * `steppingStoneBodies` for why one, and why inscribed rather than at the
   * tread's own radius.
   */
  scene.rigidBodies = stoneSetSteppingBodies({
    vessel: HERO_GARDEN_VESSEL,
    waterline_m: HERO_GARDEN_WATERLINE_M,
    seed: HERO_GARDEN_SET_SEED,
  });
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
      /**
       * Warm daylight. An eighth of a stop of blue out of the red is what turns
       * a neutral lamp into a sun, and on a set with no hue of its own it is the
       * only thing separating the lit side of a stone from its shaded side by
       * *colour* rather than only by value — the sky fill below is cool, so the
       * two halves of every boulder land on opposite sides of neutral. That
       * split is most of what reads as daylight in the reference.
       */
      colorLinear: [1, 0.955, 0.875],
      /**
       * Three and a bit, and the number is a Lambert denominator rather than an
       * exposure in disguise.
       *
       * A diffuse surface returns `albedo · I · cos / π`, so porcelain at 0.90
       * albedo facing this sun square-on comes back at `0.275 · I`. Unit
       * intensity therefore puts a fully lit white surface at 0.28 radiance,
       * which ACES places on 0.68 display: a mid grey, which is exactly what the
       * frame looked like. `I = 3.2` is where that same surface returns roughly
       * unit radiance and lands at 0.95 — near white with the curve's shoulder
       * still under it, so the coping's crest and its flank are different whites
       * rather than the same clipped one.
       *
       * It is also where the key stops being a rounding error against the fill.
       * At 1.05 the sun carried under a quarter of a lit surface's radiance and
       * the set had no light direction at all; here it carries about 40 %, which
       * is a lit side and a shaded side that differ by roughly two thirds of a
       * stop — soft, directional, and nowhere near the silhouettes a hard key
       * would cut into a set of round pale objects.
       */
      intensity: 3.2,
      direction: HERO_GARDEN_KEY_DIRECTION,
    },
    /**
     * The sky, which on this set is doing two jobs and therefore needs its own
     * balance rather than inheriting the environment preset unchanged.
     *
     * `diffuseScale` is the fill *and* the bounce: the analytic environment
     * irradiance is the sky a surface sees, and the derived GI gathers radiance
     * from surfaces that are themselves lit by it, so raising it raises the
     * light coming back off the ground as well as the light coming down. That
     * compounding is the reference's whole shadow character — its shadow
     * interiors sit a little over half a stop below its highlights because
     * everything they fall on is white and throws the light straight back up.
     * At 1.05 the shaded flanks remain open without whitening away the contact
     * shadows. The key can then carry the highlights and direction instead of
     * the whole set converging on the same display white.
     *
     * The garden palette's sky is cool (0.52, 0.60, 0.72 at the zenith) against
     * the warm key above it, so scaling it does not merely brighten the shadows,
     * it keeps them blue.
     *
     * `specularScale` is the same sky seen as a mirror, and it is also the
     * *background*: a primary ray that misses returns the prefiltered
     * environment at roughness zero. 0.75 keeps the backdrop bright but below
     * the sunlit porcelain. Surface reflection also passes
     * through the roughness-aware integrated environment BRDF, so the same
     * scale produces a broad porcelain sheen without outlining every matte
     * coping and boulder with the mirror-bright background.
     */
    environment: { diffuseScale: 1.05, specularScale: 0.75 },
    /**
     * The grade, which is where the high-key look belongs.
     *
     * ACES rather than Reinhard because the reference has a shoulder: its
     * plaster runs to within a few percent of white across a whole coping and
     * still has form in it, and `x / (x + 1)` cannot do that without the scene
     * pushing radiance far enough up that the shadows go with it. Exposure
     * stays at 1 — the set is lit correctly in physical terms, and the curve is
     * the thing that was wrong.
     */
    grade: { toneCurve: "aces", exposure: 1 },
  };
  // The set is seated on the grid that was just baked, not on the generator that
  // produced it: the two agree at every grid node by construction and differ
  // between them by the bilinear fetch, and the ground a root meets should be
  // the one the ray actually hits. That is now the expander's doing rather than
  // a closure captured here — it samples `scene.terrain`, which is the grid
  // assigned above.
  scene.scenery = heroGardenScenery();
  return scene;
}
