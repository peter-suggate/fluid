/**
 * The detail engine's own scene (brick-detail.ts): the parent's container refined by its factor, with no liquid.
 *
 * The fine engine is an ordinary brick engine, built by the ordinary scene adapter (brickSceneSetup) from this scene,
 * so every stage, the static mask and the container's walls are the parent's code at h / factor:
 * - the container is the parent's (its metres and walls), cells h / factor; the fine lattice's origin is the parent's;
 * - no initial liquid: the fine engine's liquid is the parent's, brought in where its mask turns on (a fresh tile
 *   starts from the parent's state, brick-detail.wgsl.ts);
 * - the solid voxels are the parent's authored edits, scaled by the factor, a kind that carries cells resampled
 *   (SolidPatchKind.resized). They are the same document patches, so a live voxel edit reaches both engines through
 *   the same SolidWorld fold;
 * - sources, drops, bodies, regions and the parent's authored pools are the parent's alone.
 * Anything the fine engine cannot carry is a gap, explicit (brickDetailSceneGaps).
 */
import type { SceneDescription } from "../../core/model";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { solidPatchKinds } from "../../core/solid-patch/registry";
import type { SolidWorldVoxelPatch } from "../../core/solid-world";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";

/** What the detail engine cannot carry yet, every reason (empty: none). */
export function brickDetailSceneGaps(scene: SceneDescription): string[] {
  const gaps: string[] = [], where = "the detail engine";
  if ((scene.container.shape ?? "box") !== "box") gaps.push(`${where}: a ${scene.container.shape} container`);
  if (scene.container.walls === "none") gaps.push(`${where}: an open world`);
  if (scene.terrain) gaps.push(`${where}: a terrain`);
  if (scene.solidBoundary && scene.solidBoundary.collision !== "voxels") gaps.push(`${where}: a procedural boundary`);
  if ((scene.rigidBodies?.length ?? 0) > 0) gaps.push(`${where}: rigid bodies`);
  return gaps;
}

/** The fine engine's world frame: the container's low corner (m), the parent's and the fine lattice's origin. */
export function brickDetailFrame(scene: SceneDescription): { readonly h_m: number; readonly low_m: readonly [number, number, number] } {
  const c = scene.container, cells = sceneLatticeDimensions(scene), h = c.width_m / cells[0]!;
  return { h_m: h, low_m: [-0.5 * c.width_m, 0, -0.5 * c.depth_m] };
}

/** The detail engine's scene (call brickDetailSceneGaps first: a gap here throws): the container at h / factor, empty. */
export function brickDetailSubScene(scene: SceneDescription, factor: 2 | 4): SceneDescription {
  const gaps = brickDetailSceneGaps(scene);
  if (gaps.length) throw new Error(`NB-FLIP detail engine: ${gaps.join("; ")}`);
  const r = factor, c = scene.container, f = scene.fluid, h = brickDetailFrame(scene).h_m;
  const solidVoxels: SolidWorldVoxelPatch[] = solidVoxelEditsForScene(scene).map(patch => {
    const minimum = [0, 1, 2].map(a => patch.minimum[a]! * r) as unknown as SolidWorldVoxelPatch["minimum"];
    const maximumExclusive = [0, 1, 2].map(a => patch.maximumExclusive[a]! * r) as unknown as SolidWorldVoxelPatch["maximumExclusive"];
    const kind = solidPatchKinds.get(patch);
    return kind.resized ? kind.resized(patch, minimum, maximumExclusive) : { ...patch, minimum, maximumExclusive };
  });
  const {
    refinementRegions: _regions, refinementKeyframes: _frames, automaticDetail: _automatic, brickPools: _pools,
    scheduledDrops: _drops, inflow: _inflow, initialHeightField: _height, initialDamBreakDimensions_m: _dam, initialBrickSeeds_m: _seeds,
    initialVelocity_m_s: _velocity, ...fluid
  } = f;
  return {
    ...scene,
    sceneId: `${scene.sceneId ?? "(unnamed)"}#detail`,
    container: { ...c, fillFraction: 0 },
    voxelDomain: { ...scene.voxelDomain, finestCellSize_m: h / r },
    nominalResolution: { ...scene.nominalResolution, length_m: h / r },
    fluid: { ...fluid, initialLiquidVolumes: [] },
    solidVoxels,
    rigidBodies: [],
  };
}
