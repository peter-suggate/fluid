/**
 * The planting, as a still.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/rosette.ts ...
 *
 * `ROSETTE_PREVIEW` picks what is in frame:
 *
 *   `bank` (default)  four plants on the pond's bank, at the reference's own
 *                     stations, from close enough to judge one blade.
 *   `width`           the same air plant at five blade half-widths, on the flat
 *                     ground outside the coping — the shot that finds where a
 *                     blade stops being drawn. Run it a second time with
 *                     `FLUID_SVO_DRY_FRAME_TRAVERSAL=raster-primary`: the two
 *                     paths resolve a primitive completely differently, one
 *                     through the per-voxel material owner and one through a
 *                     conservative box per record, so a difference between them
 *                     is the 25 mm lattice and an agreement is not.
 *   `segments`        the same air plant at one to six cone segments per blade,
 *                     which is the shot that prices the recurve.
 *
 * A ladder is deliberately one variable wide. Two swept at once cannot tell you
 * which of them moved the picture.
 */
import { HERO_GARDEN_VESSEL } from "../../lib/core/hero-garden-scene";
import type { CameraState, SceneDescription } from "../../lib/core/model";
import type { SceneryNode } from "../../lib/core/scenery-graph";
import { terrainHeightAt } from "../../lib/core/terrain";
import {
  ROSETTE_AIR_PLANT,
  ROSETTE_GRASS_TUFT,
  rosetteNodes,
  type RosetteForm,
} from "../../lib/core/voxel-scenery/rosette";
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

type PreviewMode = "bank" | "width" | "segments";
const mode = (process.env.ROSETTE_PREVIEW ?? "bank") as PreviewMode;

/** Blade half-widths the width ladder sweeps, in metres. A 25 mm cell for scale. */
const WIDTH_LADDER_M = [0.0012, 0.0018, 0.0025, 0.0033, 0.0045] as const;
const SEGMENT_LADDER = [1, 2, 3, 4, 5, 6] as const;
/** Ladder rank spacing and the strip of flat ground outside the coping it runs on. */
const LADDER_PITCH_M = 0.095;
const LADDER_Z_M = 0.52;

/**
 * A point on the bank, given as the pond's own polar angle and how far outside
 * the crest centreline it sits. Every one of the reference's plants is wedged
 * against the coping rather than standing anywhere in particular, so stations
 * are authored against the vessel and not as coordinates that would drift the
 * moment a control point moved.
 */
function bankStation(angle_rad: number, outward_m: number): readonly [number, number] {
  const [cx, cz] = HERO_GARDEN_VESSEL.center_m;
  const [rx, rz] = HERO_GARDEN_VESSEL.radius_m;
  const x = rx * Math.cos(angle_rad);
  const z = rz * Math.sin(angle_rad);
  const reach = Math.hypot(x, z);
  return [cx + x * (1 + outward_m / reach), cz + z * (1 + outward_m / reach)];
}

interface Planting {
  readonly key: string;
  readonly form: RosetteForm;
  readonly at_m: readonly [number, number];
  readonly seed: number;
  readonly overrides?: Partial<RosetteForm>;
}

/** The reference's own stations: grass at the far left, air plants round the rim. */
const BANK_PLANTING: readonly Planting[] = [
  { key: "planting/tuft-left", form: ROSETTE_GRASS_TUFT, at_m: bankStation(3.02, 0.048), seed: 0x51e1 },
  { key: "planting/air-back", form: ROSETTE_AIR_PLANT, at_m: bankStation(-2.20, 0.052), seed: 0x2b73 },
  { key: "planting/air-right", form: ROSETTE_AIR_PLANT, at_m: bankStation(0.15, 0.050), seed: 0x9c4d },
  { key: "planting/air-near", form: ROSETTE_AIR_PLANT, at_m: bankStation(1.05, 0.046), seed: 0x7fa1 },
];

function ladder(overrides: readonly Partial<RosetteForm>[]): readonly Planting[] {
  const first = -0.5 * LADDER_PITCH_M * (overrides.length - 1);
  return overrides.map((override, index) => ({
    key: `planting/rank-${index}`,
    form: ROSETTE_AIR_PLANT,
    at_m: [first + index * LADDER_PITCH_M, LADDER_Z_M] as const,
    // One seed across the rank, so the only difference between two plants in
    // frame is the number being swept.
    seed: 0x4d2,
    overrides: override,
  }));
}

const PLANTING: readonly Planting[] = mode === "width"
  ? ladder(WIDTH_LADDER_M.map((halfWidth_m) => ({ halfWidth_m })))
  : mode === "segments"
    ? ladder(SEGMENT_LADDER.map((segments) => ({ segments })))
    : BANK_PLANTING;

export const createScene = (): SceneDescription => {
  // The bank the plants are bedded into is the hero document's own terrain, so
  // the ground query has to come from a built scene rather than from the vessel
  // spec: the bake is what the renderer and the solver both read.
  const ground = heroPreviewScene();
  const groundHeightAt = (x: number, z: number): number => terrainHeightAt(ground.terrain, x, z);
  const nodes: SceneryNode[] = PLANTING.flatMap(({ key, form, at_m, seed, overrides }) =>
    rosetteNodes({ ...form, ...overrides, key, at_m, groundHeightAt, seed }));
  return heroPreviewScene(nodes);
};

/**
 * Close, and framed on a plant rather than on the pond.
 *
 * The distance is to the *target*, so a camera aimed at the pond's centre sits a
 * metre off a rosette on its far rim and the plant is thirty pixels again. The
 * bank target is therefore a station, not a composition — and 0.30 m puts a
 * 9 mm blade at eleven pixels, which is roughly four times what it gets in the
 * hero frame and is the only range at which a question about one blade has an
 * answer.
 */
export const camera: Partial<CameraState> = {
  ...heroPreviewCamera(),
  ...(mode === "bank"
    ? { azimuth_rad: 0.95, elevation_rad: 0.28, distance_m: 0.30, target_m: { x: 0.564, y: 0.340, z: 0.062 } }
    : { azimuth_rad: 0.10, elevation_rad: 0.16, distance_m: 0.21, target_m: { x: 0, y: 0.345, z: LADDER_Z_M } }),
};
