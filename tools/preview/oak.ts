/**
 * The hero garden's oak, on its own bank, at a camera that fills the frame with
 * it.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/oak.ts
 *
 * The specimen is spliced onto the hero document the catalog builds, so it is
 * lit, shaded and voxelized exactly as it is in the finished scene — see
 * `tools/preview/README.md` for why this must go through `heroPreviewScene` and
 * never through the scene factory, which would render the porcelain garden under
 * the default set's dark teal sky.
 *
 * The ground query is the document's own baked terrain rather than a second
 * evaluation of the vessel's height function: the root flare has to meet the
 * bank the renderer draws, and the bake is a resampling of the generator.
 *
 * `/shape-lab` is the loop for *proportion* — CPU, sliders, any leaf, no GPU
 * lock. This is the loop for **light**: the leaf-invariance contract is about a
 * silhouette, and a silhouette is a thing you judge in a lit frame. One
 * override, because the whole open question about this species is what the
 * lattice does to it:
 *
 *   FLUID_OAK_CELL_M       voxel size, default the hero scene's own
 */
import type { CameraState, SceneDescription } from "../../lib/model";
import { terrainHeightAt } from "../../lib/terrain";
import { OAK_HERO_SPREADING, oakNodes, type OakSpec } from "../../lib/voxel-scenery/oak";
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

/**
 * The hero pond's back-right terrace, whose plateau centre is (0.62, 0.30) and
 * whose flat runs to about a third of its radius — so this stands on level
 * ground with the fall of the bank still inside the root flare's reach. The
 * bonsai this replaces stood at (0.55, -0.15) in the document and (0.60, 0.26)
 * in its own preview; the tree is twice as tall now, so it is worth looking at
 * both and this preview takes the terrace's own centre.
 */
const HERO_OAK_STAND_M = [0.60, 0.26] as const;

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
  const spec: OakSpec = {
    ...OAK_HERO_SPREADING,
    key: "oak",
    at_m: HERO_OAK_STAND_M,
    groundHeightAt: ground,
    // Toward the pond's centre from the stand: the crown reaches out over the
    // water. A bearing derived from the stand rather than authored beside it,
    // so the two cannot drift apart.
    lean: [-HERO_OAK_STAND_M[0], -HERO_OAK_STAND_M[1]],
    seed: 0x0a_c0_de,
    // The leaf the specimen resolves its branch orders and foliage lattices
    // against, taken from the document it is being spliced into rather than
    // defaulted — otherwise the preview draws a coarser tree than the scene.
    leafSize_m: bank.voxelDomain.detailCellSize_m ?? bank.voxelDomain.finestCellSize_m,
  };
  const scene = heroPreviewScene(oakNodes(spec));
  const cell = number("FLUID_OAK_CELL_M", scene.voxelDomain.finestCellSize_m);
  return { ...scene, voxelDomain: { ...scene.voxelDomain, finestCellSize_m: cell } };
};

/**
 * Three framings, on the hero camera's own azimuth so the specimen is always lit
 * from where it will be lit in the finished frame. `FLUID_OAK_VIEW` picks:
 *
 *   whole  the specimen filling the frame — the framing every silhouette and
 *          proportion judgement is made in
 *   base   the root flare and the bole alone, which is the only way to tell a
 *          shape from a shading artifact at this scale
 *   crown  the foliage alone, for judging the leaf grain against the lattice and
 *          — the thing this species is judged on — whether the limbs are still
 *          visible through it
 *
 * The dry path's vertical half-angle is 0.72 in tangent, so a 0.8 m tree standing
 * on a 0.21 m bank needs about 0.75 m of distance to fill a 520-line frame.
 */
const VIEWS: Readonly<Record<string, Partial<CameraState>>> = {
  whole: { elevation_rad: 0.22, distance_m: 0.95, target_m: { x: 0.52, y: 0.62, z: 0.21 } },
  base: { elevation_rad: 0.14, distance_m: 0.30, target_m: { x: 0.59, y: 0.28, z: 0.25 } },
  crown: { elevation_rad: 0.28, distance_m: 0.55, target_m: { x: 0.46, y: 0.82, z: 0.18 } },
};

export const camera: Partial<CameraState> = {
  ...heroPreviewCamera(),
  ...(VIEWS[process.env.FLUID_OAK_VIEW ?? "whole"] ?? VIEWS.whole),
};
