/**
 * A detail engine's own scene (brick-detail.ts): the parent scene cut to one detail box and refined by its factor.
 *
 * The fine engine is an ordinary brick engine, built by the ordinary scene adapter (brickSceneSetup) from this scene,
 * so every stage, the static mask, the initial liquid and the pool rule are the parent's code at h / factor:
 * - the container is the box (its metres; x and z centred, its floor at y = 0), cells h / factor;
 * - the initial liquid is the parent's, translated: its volumes, and a fill fraction as the box of liquid up to the
 *   fill's height (the box's own fill would be a different height);
 * - the solid voxels are the parent's authored edits (never its derived shell: the box's walls are the coupling's)
 *   translated and scaled by the factor, a kind that carries cells resampled (SolidPatchKind.resized). They are the
 *   same document patches, so a live voxel edit reaches both engines through the same SolidWorld fold;
 * - sources, bodies, regions and the parent's authored pools are the parent's alone: liquid reaches the fine engine
 *   through its coupled boundary (brick-detail.wgsl.ts).
 * Anything the box cannot carry is a gap, explicit (brickDetailSceneGaps).
 */
import type { InitialLiquidVolume, SceneDescription, Vec3 as ModelVec3 } from "../../core/model";
import { solidVoxelEditsForScene } from "../../core/scene-lattice";
import { solidPatchKinds } from "../../core/solid-patch/registry";
import type { SolidWorldVoxelPatch } from "../../core/solid-world";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import type { DetailBox } from "./brick-detail-regions";

/** What a detail box cannot carry yet, every reason (empty: none). */
export function brickDetailSceneGaps(scene: SceneDescription, box: DetailBox): string[] {
  const gaps: string[] = [], f = scene.fluid, where = `detail box ${box.origins.join("+")}`;
  if ((scene.container.shape ?? "box") !== "box") gaps.push(`${where}: a ${scene.container.shape} container`);
  if (scene.container.walls === "none") gaps.push(`${where}: an open world`);
  if (scene.terrain) gaps.push(`${where}: a terrain`);
  if (scene.solidBoundary && scene.solidBoundary.collision !== "voxels") gaps.push(`${where}: a procedural boundary`);
  if ((scene.rigidBodies?.length ?? 0) > 0) gaps.push(`${where}: rigid bodies`);
  if (f.initialHeightField || f.initialDamBreakDimensions_m || f.initialBrickSeeds_m?.length) gaps.push(`${where}: an initial liquid that is not volumes and a fill`);
  if (f.initialVelocity_m_s && (f.initialVelocity_m_s.x !== 0 || f.initialVelocity_m_s.y !== 0 || f.initialVelocity_m_s.z !== 0)) gaps.push(`${where}: an initial velocity`);
  return gaps;
}

/** The box's world frame: its low corner (m) in the parent's frame, and the shift that centres it (parent − sub). */
export function brickDetailFrame(scene: SceneDescription, box: DetailBox): { readonly h_m: number; readonly low_m: readonly [number, number, number]; readonly shift_m: readonly [number, number, number] } {
  const c = scene.container, cells = sceneLatticeDimensions(scene), h = c.width_m / cells[0]!;
  const low = [-0.5 * c.width_m + box.lo[0] * h, box.lo[1] * h, -0.5 * c.depth_m + box.lo[2] * h] as const;
  const shift = [low[0] + 0.5 * (box.hi[0] - box.lo[0]) * h, low[1], low[2] + 0.5 * (box.hi[2] - box.lo[2]) * h] as const;
  return { h_m: h, low_m: low, shift_m: shift };
}

/** The detail box's scene (call brickDetailSceneGaps first: a gap here throws). */
export function brickDetailSubScene(scene: SceneDescription, box: DetailBox): SceneDescription {
  const gaps = brickDetailSceneGaps(scene, box);
  if (gaps.length) throw new Error(`NB-FLIP detail engine: ${gaps.join("; ")}`);
  const r = box.factor, c = scene.container, f = scene.fluid, { h_m: h, shift_m: s } = brickDetailFrame(scene, box);
  const ext = [0, 1, 2].map(a => box.hi[a]! - box.lo[a]!);
  const move = (p: ModelVec3): ModelVec3 => ({ x: p.x - s[0], y: p.y - s[1], z: p.z - s[2] });
  const volumes: InitialLiquidVolume[] = (f.initialLiquidVolumes ?? []).map(v => v.shape === "box" ? { ...v, min_m: move(v.min_m), max_m: move(v.max_m) } : { ...v, center_m: move(v.center_m) });
  // The parent's fill (its container's fraction of height) as the box of liquid it is.
  if (c.fillFraction > 0) {
    const top = c.fillFraction * c.height_m - s[1];
    if (top > 0) volumes.push({ shape: "box", min_m: { x: -0.5 * ext[0]! * h - h, y: 0, z: -0.5 * ext[2]! * h - h }, max_m: { x: 0.5 * ext[0]! * h + h, y: top, z: 0.5 * ext[2]! * h + h } });
  }
  const solidVoxels: SolidWorldVoxelPatch[] = solidVoxelEditsForScene(scene).map(patch => {
    const minimum = [0, 1, 2].map(a => (patch.minimum[a]! - box.lo[a]!) * r) as unknown as SolidWorldVoxelPatch["minimum"];
    const maximumExclusive = [0, 1, 2].map(a => (patch.maximumExclusive[a]! - box.lo[a]!) * r) as unknown as SolidWorldVoxelPatch["maximumExclusive"];
    const kind = solidPatchKinds.get(patch);
    return kind.resized ? kind.resized(patch, minimum, maximumExclusive) : { ...patch, minimum, maximumExclusive };
  }).filter(p => [0, 1, 2].every(a => p.maximumExclusive[a]! > 0 && p.minimum[a]! < ext[a]! * r));
  const top = box.hi[1] >= sceneLatticeDimensions(scene)[1]! ? c.top : "closed";
  const {
    refinementRegions: _regions, refinementKeyframes: _frames, automaticDetail: _automatic, brickPools: _pools,
    scheduledDrops: _drops, inflow: _inflow, ...fluid
  } = f;
  return {
    ...scene,
    sceneId: `${scene.sceneId ?? "(unnamed)"}#detail-${box.origins.join("+")}`,
    container: { ...c, width_m: ext[0]! * h, height_m: ext[1]! * h, depth_m: ext[2]! * h, fillFraction: 0, top },
    voxelDomain: { ...scene.voxelDomain, finestCellSize_m: h / r },
    nominalResolution: { ...scene.nominalResolution, length_m: h / r },
    fluid: { ...fluid, initialLiquidVolumes: volumes },
    solidVoxels,
    rigidBodies: [],
  };
}
