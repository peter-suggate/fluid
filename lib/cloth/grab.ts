/**
 * The region of cloth a press grabs (the viewport's cloth-grab gesture): the
 * vertices around the picked point, found by walking the mesh from the hit
 * triangle, so a fold's other layer within reach is not taken unless the
 * cloth itself joins the two inside the region. Each vertex gets a smooth
 * falloff weight, so the grab has no hard edge to kink the sheet. The solver
 * side is `ClothGPUSolver.setGrab`; this is CPU work at the press only.
 */
import type { ClothGrab, ClothSurfacePublication, ClothTopologyData, ClothVector } from './types';
import type { ClothSurfaceHit } from './geometry';

/** Grab radius in mean rest edge lengths (a coarse sheet still grabs a patch, not a vertex). */
export const CLOTH_GRAB_EDGE_RADII = 2.5;
/** Grab radius as a fraction of the sheet's rest diagonal (a fine sheet still grabs a hand-sized patch). */
export const CLOTH_GRAB_SIZE_FRACTION = 0.04;
/** Least weight of the hit triangle's own corners, so a press always holds the triangle it landed on. */
export const CLOTH_GRAB_CORNER_WEIGHT = 0.25;

export interface ClothGrabRegion {
  readonly vertices: number[];
  readonly weights: number[];
  readonly radius_m: number;
}

/** The default grab radius of a sheet: the larger of a few edges and a fraction of its size. */
export function clothGrabRadius(topology: ClothTopologyData): number {
  const x = topology.restPositions_m, t = topology.triangles;
  let edges = 0, length = 0;
  for (let f = 0; f < t.length; f += 3) for (let k = 0; k < 3; k++) {
    const a = t[f + k]!, b = t[f + (k + 1) % 3]!;
    length += Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!); edges++;
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < x.length; i++) { lo[i % 3] = Math.min(lo[i % 3]!, x[i]!); hi[i % 3] = Math.max(hi[i % 3]!, x[i]!); }
  const diagonal = Math.hypot(hi[0]! - lo[0]!, hi[1]! - lo[1]!, hi[2]! - lo[2]!);
  const radius = Math.max(CLOTH_GRAB_EDGE_RADII * (edges ? length / edges : 0), CLOTH_GRAB_SIZE_FRACTION * diagonal);
  if (!(radius > 0 && Number.isFinite(radius))) throw new Error('Cloth grab radius needs a sheet with non-degenerate edges');
  return radius;
}

/**
 * The vertices within `radius_m` of `point_m` (in the pose `positions`) that the mesh joins to the hit triangle
 * through vertices also within it, with weight (1 - (d/R)^2)^2 (the hit triangle's corners at least
 * CLOTH_GRAB_CORNER_WEIGHT).
 */
export function clothGrabRegion(topology: ClothTopologyData, positions: ArrayLike<number>, element: number, point_m: ClothVector, radius_m = clothGrabRadius(topology)): ClothGrabRegion {
  const t = topology.triangles, N = topology.restPositions_m.length / 3;
  if (positions.length !== 3 * N) throw new Error('Cloth grab pose and topology disagree');
  if (!(Number.isInteger(element) && element >= 0 && 3 * element < t.length)) throw new Error(`Cloth grab triangle ${element} is not in the topology`);
  if (!(radius_m > 0 && Number.isFinite(radius_m))) throw new Error('Cloth grab radius must be positive and finite');
  // Vertex adjacency through triangle edges (CSR), built per press.
  const degree = new Uint32Array(N + 1);
  for (let f = 0; f < t.length; f += 3) for (let k = 0; k < 3; k++) { degree[t[f + k]! + 1] += 2; }
  for (let v = 0; v < N; v++) degree[v + 1] += degree[v]!;
  const fill = degree.slice(0, N), neighbours = new Uint32Array(degree[N]!);
  for (let f = 0; f < t.length; f += 3) for (let k = 0; k < 3; k++) {
    const a = t[f + k]!, b = t[f + (k + 1) % 3]!, c = t[f + (k + 2) % 3]!;
    neighbours[fill[a]!++] = b; neighbours[fill[a]!++] = c;
  }
  const distance = (v: number) => Math.hypot(positions[3 * v]! - point_m[0], positions[3 * v + 1]! - point_m[1], positions[3 * v + 2]! - point_m[2]);
  const weight = (d: number) => { const s = 1 - (d / radius_m) ** 2; return s > 0 ? s * s : 0; };
  const seen = new Uint8Array(N), vertices: number[] = [], weights: number[] = [], queue: number[] = [];
  for (let k = 0; k < 3; k++) {
    const v = t[3 * element + k]!;
    if (seen[v]) continue;
    seen[v] = 1; queue.push(v);
    vertices.push(v); weights.push(Math.max(weight(distance(v)), CLOTH_GRAB_CORNER_WEIGHT));
  }
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head]!;
    for (let j = degree[u]!; j < degree[u + 1]!; j++) {
      const v = neighbours[j]!;
      if (seen[v]) continue;
      seen[v] = 1;
      const w = weight(distance(v));
      if (!(w > 0)) continue;
      queue.push(v); vertices.push(v); weights.push(w);
    }
  }
  return { vertices, weights, radius_m };
}

/** A displacement range per axis, world metres. */
export interface ClothGrabReach {
  readonly lo: ClothVector;
  readonly hi: ClothVector;
}

/**
 * How far a grab may pull its region: the displacements that keep every grabbed vertex's target inside `box` (the
 * container, which is also the box cloth contact sees solids in; a vertex outside it halts the solver), with
 * `margin_m` to spare for the motion between the press's pose and the anchors the GPU takes at the grab's first step.
 * Zero is always in range, so a press never jumps.
 */
export function clothGrabReach(grab: Pick<ClothGrab, 'vertices'>, positions: ArrayLike<number>, box: { readonly lo: ClothVector; readonly hi: ClothVector }, margin_m: number): ClothGrabReach {
  const lo = [0, 0, 0], hi = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    let min = Infinity, max = -Infinity;
    for (const v of grab.vertices) { const x = positions[3 * v + c]!; min = Math.min(min, x); max = Math.max(max, x); }
    lo[c] = Math.min(0, box.lo[c]! + margin_m - min);
    hi[c] = Math.max(0, box.hi[c]! - margin_m - max);
  }
  return { lo: lo as unknown as ClothVector, hi: hi as unknown as ClothVector };
}

/** `displacement_m` held within `reach`. */
export function clothGrabDisplacement(displacement_m: ClothVector, reach: ClothGrabReach): ClothVector {
  return [0, 1, 2].map(c => Math.min(reach.hi[c]!, Math.max(reach.lo[c]!, displacement_m[c]!))) as unknown as ClothVector;
}

/** The grab a press on `hit` opens on the published surface it picked, at zero displacement. */
export function clothGrabForHit(id: number, surface: ClothSurfacePublication, hit: Pick<ClothSurfaceHit, 'documentId' | 'element' | 'point_m'>): ClothGrab {
  if (surface.documentId !== hit.documentId) throw new Error('Cloth grab hit and surface name different documents');
  const region = clothGrabRegion(surface.topology, surface.positions_m, hit.element, hit.point_m);
  return { id, documentId: surface.documentId, topologyRevision: surface.topologyRevision, vertices: region.vertices, weights: region.weights, displacement_m: [0, 0, 0] };
}
