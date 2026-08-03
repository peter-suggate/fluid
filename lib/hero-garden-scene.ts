import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import type { SceneryGraph, SceneryNode } from "./scenery-graph";
import {
  bakePondVesselTerrain,
  pondVesselWaterline,
  type PondVesselSpec,
} from "./voxel-scenery/pond-vessel";

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
 * Why the lattice is 7.5 mm. The reference's two most recognisable water
 * features are the concentric ripple train and the plunge column, and both are
 * sub-cell at the 25 mm the existing garden runs: the ripples read as ~4 % of
 * the pond's width, so ~5 cm, and the stream is ~4 cm across. At 7.5 mm they
 * land at ~6.7 and ~5 cells respectively, which is the coarsest lattice on which
 * the reference is reachable at all rather than merely suggested.
 *
 * Every container dimension is a whole number of 8-cell bricks at that spacing
 * (0.06 m), so the sparse domain has no partial brick at any wall.
 */

export const HERO_GARDEN_CONTAINER = { width_m: 1.8, height_m: 0.6, depth_m: 1.2 } as const;
export const HERO_GARDEN_CELL_M = 0.0075;
export const HERO_GARDEN_BRICK_CELLS = 8 as const;

/**
 * Dry coping left standing above the still-water level.
 *
 * The reference's water sits high — it laps the rounded inner flank of the rim
 * rather than sitting down in a well — and that is most of why the pond reads as
 * full rather than as a drained basin. It also leaves the hose somewhere to put
 * its water before the rim spills, which is the honest reason it is not smaller.
 */
export const HERO_GARDEN_FREEBOARD_M = 0.04;

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
  basinDepth_m: 0.13,
  rimHeight_m: 0.055,
  rimHalfWidth_m: 0.055,
  innerFace_m: 0.16,
  lobes: 7,
  wobble: 0.08,
  seed: 0x9a7de11,
  terraces: [
    // The plateau the bonsai stands on, at the back right.
    { center_m: [0.62, 0.30] as const, radius_m: [0.44, 0.34] as const, height_m: 0.075, rotation_rad: -0.35, flat: 0.30 },
    // A lower step at the near left, where the boulder group beds in.
    { center_m: [-0.64, -0.20] as const, radius_m: [0.36, 0.30] as const, height_m: 0.042, rotation_rad: 0.25, flat: 0.28 },
  ],
});

export const HERO_GARDEN_WATERLINE_M = pondVesselWaterline(HERO_GARDEN_VESSEL, HERO_GARDEN_FREEBOARD_M);

/**
 * Low, close and off the long axis, so the coping's near arc crosses the bottom
 * of the frame and the far rim still occludes its own waterline — which is what
 * makes the pond read as set into the ground instead of sitting on it.
 */
export const heroGardenCamera: Partial<CameraState> = {
  azimuth_rad: 0.95,
  elevation_rad: 0.30,
  distance_m: 1.62,
  target_m: { x: 0.02, y: 0.30, z: 0.0 },
};

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

const HERO_GARDEN_SCENERY: SceneryGraph = {
  palettes: {
    clay: { tint: [1, 0.985, 0.955] },
    stone: { tint: [0.972, 0.984, 1] },
  },
  nodes: [
    { kind: "terrain-shell", id: "shell", materialModel: "garden-terrain" },
    ...hoseNodes(),
  ],
};

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

export function createHeroGardenHoseScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "hero-garden-hose";
  scene.container.width_m = HERO_GARDEN_CONTAINER.width_m;
  scene.container.height_m = HERO_GARDEN_CONTAINER.height_m;
  scene.container.depth_m = HERO_GARDEN_CONTAINER.depth_m;
  scene.container.top = "open";
  scene.container.vessel = "none";
  scene.voxelDomain = { finestCellSize_m: HERO_GARDEN_CELL_M, brickSize_cells: HERO_GARDEN_BRICK_CELLS };
  scene.terrain = {
    baseHeight_m: HERO_GARDEN_VESSEL.groundHeight_m,
    features: [],
    // The bake samples on the lattice the solver reads columns at, so the ground
    // the water meets is the ground that was generated rather than a resample of it.
    grid: bakePondVesselTerrain(HERO_GARDEN_VESSEL, HERO_GARDEN_CONTAINER, HERO_GARDEN_CELL_M),
  };
  scene.container.fillFraction = HERO_GARDEN_WATERLINE_M / HERO_GARDEN_CONTAINER.height_m;
  scene.fluid.initialCondition = "tank-fill";
  scene.fluid.inflow = heroGardenInflow();
  scene.rigidBodies = [];
  scene.scenery = HERO_GARDEN_SCENERY;
  return scene;
}
