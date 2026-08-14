/**
 * The hero pond's bonsai, on its own bank, at a camera that fills the frame
 * with it.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/bonsai.ts
 *
 * The specimen is spliced onto the hero document the catalog builds, so it is
 * lit, shaded and voxelized exactly as it will be once `bonsai` is wired into
 * `lib/hero-garden-scene.ts` — see `tools/preview/README.md` for why this must
 * go through `heroPreviewScene` and not through the scene factory.
 *
 * The ground query is the document's own baked terrain rather than a second
 * evaluation of `pondVesselHeightAt`. The roots have to meet the bank the
 * renderer draws, and the bake is a quarter-cell resampling of the generator, so
 * asking the generator directly would put the toes a fraction of a millimetre
 * off the surface they are supposed to be gripping — and would silently stop
 * tracking the vessel the day the hero scene bakes it differently.
 *
 * Two overrides exist because the canopy's whole open question is what the
 * lattice does to it (see the header of `lib/voxel-scenery/bonsai.ts`), and
 * answering it means rendering one geometry on several lattices:
 *
 *   FLUID_BONSAI_CELL_M       voxel size, default the hero scene's own
 *   FLUID_BONSAI_FLORET_M     floret half-width, default the tuned one
 *   FLUID_BONSAI_FLORETS      florets cast per lobe
 *   FLUID_BONSAI_MAX_LEAVES   the leaf budget, so a measurement can exceed the
 *                             shipping one on purpose rather than by mistake
 *
 * A finer lattice costs nothing here: this scene opens dry, so none of the fluid
 * walls the hero scene's header records are on the path to a frame.
 */
import type { CameraState, SceneDescription } from "../../lib/core/model";
import { terrainHeightAt } from "../../lib/core/terrain";
import { BONSAI_POND_CANOPY, bonsaiNodes, type BonsaiSpec } from "../../lib/core/voxel-scenery/bonsai";
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

/**
 * The hero pond's back-right terrace, whose plateau centre is (0.62, 0.30) and
 * whose flat runs to about a third of its radius — so this stands on level
 * ground with the fall of the bank still inside the root fingers' reach.
 */
const HERO_BONSAI_STAND_M = [0.60, 0.26] as const;

const number = (name: string, fallback: number): number => {
  const raw = process.env[name];
  if (raw === undefined || raw === "") return fallback;
  const value = Number(raw);
  if (!(value > 0)) throw new RangeError(`${name} must be a positive number`);
  return value;
};

export const createScene = (): SceneDescription => {
  const bank = heroPreviewScene();
  const ground = (x: number, z: number) => terrainHeightAt(bank.terrain, x, z);
  const spec: BonsaiSpec = {
    ...BONSAI_POND_CANOPY,
    key: "bonsai",
    at_m: HERO_BONSAI_STAND_M,
    groundHeightAt: ground,
    // Toward the pond's centre from the stand: the crown overhangs the water.
    lean: [-HERO_BONSAI_STAND_M[0], -HERO_BONSAI_STAND_M[1]],
    seed: 0x8017a1,
    floretRadius_m: number("FLUID_BONSAI_FLORET_M", BONSAI_POND_CANOPY.floretRadius_m),
    floretsPerLobe: Math.round(number("FLUID_BONSAI_FLORETS", BONSAI_POND_CANOPY.floretsPerLobe)),
    maximumLeaves: Math.round(number("FLUID_BONSAI_MAX_LEAVES", BONSAI_POND_CANOPY.maximumLeaves)),
  };
  const scene = heroPreviewScene(bonsaiNodes(spec));
  const cell = number("FLUID_BONSAI_CELL_M", scene.voxelDomain.finestCellSize_m);
  return { ...scene, voxelDomain: { ...scene.voxelDomain, finestCellSize_m: cell } };
};

/**
 * Three framings, on the hero camera's own azimuth so the specimen is always lit
 * from where it will be lit in the finished frame. `FLUID_BONSAI_VIEW` picks:
 *
 *   whole  the specimen filling the frame — the framing every silhouette and
 *          proportion judgement is made in
 *   base   the root fingers and the trunks alone, which is the only way to tell
 *          a shape from a shading artifact at this scale
 *   crown  the canopy alone, for judging the floret grain against the lattice
 *
 * The dry path's vertical half-angle is 0.72 in tangent, so a 0.7 m tree needs
 * about 0.56 m of distance to fill a 520-line frame.
 */
const VIEWS: Readonly<Record<string, Partial<CameraState>>> = {
  whole: { elevation_rad: 0.19, distance_m: 0.56, target_m: { x: 0.52, y: 0.52, z: 0.21 } },
  base: { elevation_rad: 0.16, distance_m: 0.26, target_m: { x: 0.58, y: 0.41, z: 0.25 } },
  crown: { elevation_rad: 0.30, distance_m: 0.32, target_m: { x: 0.48, y: 0.66, z: 0.19 } },
};

export const camera: Partial<CameraState> = {
  ...heroPreviewCamera(),
  ...(VIEWS[process.env.FLUID_BONSAI_VIEW ?? "whole"] ?? VIEWS.whole),
};
