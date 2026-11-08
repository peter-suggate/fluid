import type { SceneDescription } from '../core/model';
import type { ClothStrokeCut, ClothVertexOrigin } from './edit';
import { cutClothShellBox } from './edit';
import { publishSceneClothSurfaces, sceneClothSurface } from './scene-geometry';
import type { ClothDocument, ClothSurfacePublication, ClothTopologyData } from './types';

/** How an edited document was derived from the live one it replaced: the
 * cloth system remaps its current GPU state through these origins and
 * rebuilds colouring and patches for the touched vertices only. */
export interface ClothEditProvenance {
  readonly source: ClothDocument;
  readonly vertexOrigins: readonly ClothVertexOrigin[];
  readonly touchedVertices: ReadonlySet<number>;
}
const provenance = new WeakMap<ClothDocument, ClothEditProvenance>();
export function clothEditProvenance(document: ClothDocument): ClothEditProvenance | undefined { return provenance.get(document); }

function compose(first: ClothEditProvenance | undefined, origins: readonly ClothVertexOrigin[], touched: ReadonlySet<number>, source: ClothDocument): ClothEditProvenance {
  if (!first) return {source, vertexOrigins: origins, touchedVertices: touched};
  const vertexOrigins = origins.map(o => {
    const weights = new Map<number, number>();
    o.vertices.forEach((v, k) => first.vertexOrigins[v]!.vertices.forEach((u, j) => weights.set(u, (weights.get(u) ?? 0) + o.weights[k]! * first.vertexOrigins[v]!.weights[j]!)));
    return {vertices: [...weights.keys()], weights: [...weights.values()]};
  });
  const all = new Set(touched);
  origins.forEach((o, i) => { if (o.vertices.some(v => first.touchedVertices.has(v))) all.add(i); });
  return {source: first.source, vertexOrigins, touchedVertices: all};
}

/** One immutable edit of authoritative cloth geometry. The same resulting
 * topology is serialized, rendered, picked and simulated. */
export function sceneWithClothStroke(base: SceneDescription, cuts: readonly ClothStrokeCut[]): SceneDescription {
  if (!base.cloth?.length || !cuts.length) return base;
  let changed = false;
  const publications: ClothSurfacePublication[] = [];
  const cloth = base.cloth.map(document => {
    const live = sceneClothSurface(base, document.id);
    let data: ClothTopologyData = live ? {...live.topology, solverState: live.solverState} : document.geometry.data;
    let current: ArrayLike<number> = live?.positions_m ?? data.initialPositions_m ?? data.restPositions_m;
    if (live?.velocities_m_s) data = {...data, initialVelocities_m_s: Array.from(live.velocities_m_s)};
    let next = document, revision = live?.topologyRevision ?? 0, trail: ClothEditProvenance | undefined;
    for (const {box, kind, halfSpaces} of cuts) {
      const radius = Math.max(...document.materials.map(m => m.thickness_m / 2));
      const intersects = [0, 1, 2].every(axis => {
        let lo = Infinity, hi = -Infinity;
        for (let i = axis; i < current.length; i += 3) { lo = Math.min(lo, current[i]!); hi = Math.max(hi, current[i]!); }
        return lo - radius < box.maximum_m[axis]! && hi + radius > box.minimum_m[axis]!;
      });
      if (!intersects) continue;
      if (kind !== 'box' && !halfSpaces?.length) throw new Error(`Cloth ${kind} cuts need a matching surface operation; refusing a bounding-box substitute`);
      const cut = cutClothShellBox(next, data, current, box, halfSpaces);
      if (!cut.changed) continue;
      trail = compose(trail, cut.vertexOrigins, cut.touchedVertices, document);
      next = cut.document; data = cut.topology; current = cut.positions_m;
      changed = true; revision++;
    }
    if (next === document) return document;
    provenance.set(next, trail!);
    if (live) publications.push({...live, topology: data, positions_m: Float64Array.from(current),
      velocities_m_s: data.initialVelocities_m_s ? Float64Array.from(data.initialVelocities_m_s) : undefined,
      solverState: data.solverState, topologyRevision: revision});
    return next;
  });
  if (!changed) return base;
  const result = {...base, cloth};
  if (publications.length) publishSceneClothSurfaces(result, publications);
  return result;
}
