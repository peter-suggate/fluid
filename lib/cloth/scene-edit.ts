import type { SceneDescription } from '../core/model';
import { sceneWithSourceSolidCuts } from './source-solid-edit';
import type { ClothStrokeCut } from './edit';
import { cutClothShellBox } from './edit';
import { cachedClothTopology } from './source-assets';
import { publishSceneClothSurfaces, sceneClothSurface } from './scene-geometry';
import type { ClothSurfacePublication, ClothTopologyData } from './types';

/** One immutable edit of authoritative cloth geometry. The same resulting
 * topology is serialized, rendered, picked and used for collision. */
export function sceneWithClothStroke(base: SceneDescription, cuts: readonly ClothStrokeCut[]): SceneDescription {
  const sourceEdited = sceneWithSourceSolidCuts(base, cuts);
  if (!base.cloth?.length || !cuts.length) return sourceEdited;
  let changed = sourceEdited !== base;
  const publications: ClothSurfacePublication[] = [];
  const cloth = base.cloth.map(document => {
    const live = sceneClothSurface(base, document.id);
    let data: ClothTopologyData | undefined = live?.topology ?? cachedClothTopology(document.geometry);
    if (!data) return document; // Unloaded assets have no visible surface.
    let current: ArrayLike<number> = live?.positions_m ?? data.initialPositions_m ?? data.restPositions_m;
    if (live) data = {...data, initialWaterVolume_m3: Array.from(live.waterVolume_m3),
      initialVelocities_m_s: live.velocities_m_s ? Array.from(live.velocities_m_s) : data.initialVelocities_m_s,
      initialTwist_rad:live.rodTwist_rad?Array.from(live.rodTwist_rad):data.initialTwist_rad,
      initialReferenceTwist_rad:live.rodReferenceTwist_rad?Array.from(live.rodReferenceTwist_rad):data.initialReferenceTwist_rad,
      initialAngularVelocity_rad_s:live.rodAngularVelocity_rad_s?Array.from(live.rodAngularVelocity_rad_s):data.initialAngularVelocity_rad_s,
      initialSupportPositions_m:live.supportPositions_m?Array.from(live.supportPositions_m):data.initialSupportPositions_m,
      initialAttachmentTwist_rad:live.attachmentTwistTarget_rad?Array.from(live.attachmentTwistTarget_rad):data.initialAttachmentTwist_rad,
      initialRodDirectors:live.rodDirectors?Array.from(live.rodDirectors):data.initialRodDirectors,
      initialRodTangents:live.rodTangents?Array.from(live.rodTangents):data.initialRodTangents};
    let next = document, revision = live?.topologyRevision ?? 0;
    for (const {box, kind, halfSpaces} of cuts) {
      const intersects = [0, 1, 2].every(axis => {
        let lo = Infinity, hi = -Infinity;
        for (let i = axis; i < current.length; i += 3) { lo = Math.min(lo, current[i]!); hi = Math.max(hi, current[i]!); }
        // Yarn is a capsule of its radius and a shell a solid of its thickness (see cutClothShellBox).
        const radius=Math.max(data!.rodIndices.length?Math.max(...document.materials.map(m=>Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m))):0,
          data!.triangles.length?Math.max(...document.materials.map(m=>m.thickness_m/2)):0);
        return lo-radius < box.maximum_m[axis]! && hi+radius > box.minimum_m[axis]!;
      });
      if (!intersects) continue;
      const cut = cutClothShellBox(next, data, current, box, halfSpaces);
      if (!cut.changed) continue;
      if (kind !== 'box' && !halfSpaces?.length) throw new Error(`Cloth ${kind} cuts need a matching surface operation; refusing a bounding-box substitute`);
      next = cut.document; data = cut.topology; current = cut.positions_m;
      changed = true; revision++;
    }
    if (live) publications.push({...live, topology: data, positions_m: Float64Array.from(current),
      velocities_m_s: data.initialVelocities_m_s ? Float64Array.from(data.initialVelocities_m_s) : undefined,
      waterVolume_m3: Float64Array.from(data.initialWaterVolume_m3 ?? new Array(data.restPositions_m.length / 3).fill(0)),
      rodTwist_rad:data.initialTwist_rad?Float64Array.from(data.initialTwist_rad):undefined,
      rodReferenceTwist_rad:data.initialReferenceTwist_rad?Float64Array.from(data.initialReferenceTwist_rad):undefined,
      rodAngularVelocity_rad_s:data.initialAngularVelocity_rad_s?Float64Array.from(data.initialAngularVelocity_rad_s):undefined,
      rodDirectors:data.initialRodDirectors?Float64Array.from(data.initialRodDirectors):undefined,
      rodTangents:data.initialRodTangents?Float64Array.from(data.initialRodTangents):undefined,
      // Derived capacities and saturation must be rebuilt from the edited topology.
      controlVolume_m3: undefined, solidFraction: undefined, saturation: undefined, topologyRevision: revision});
    return next;
  });
  if (!changed) return base;
  const result = {...sourceEdited, cloth};
  if (publications.length) publishSceneClothSurfaces(result, publications);
  return result;
}
