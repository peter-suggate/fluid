import { encodeOctreeFaceKey, type OctreeFaceOrigin } from "./webgpu-octree-face-transport";

export interface OctreeFlowRegion {
  readonly id: string;
}

/** A canonical physical face fragment; velocity is positive negative -> positive. */
export interface OctreeFlowFaceFragment {
  readonly negative: string | null;
  readonly positive: string | null;
  readonly origin: OctreeFaceOrigin;
  readonly axis: 0 | 1 | 2;
  readonly span: number;
  readonly area: number;
  readonly normalVelocity: number;
}

export interface OctreeConnectedFlowAudit {
  readonly connected: boolean;
  readonly visited: ReadonlySet<string>;
  readonly netOutwardFlux: ReadonlyMap<string, number>;
  readonly totalBoundaryFlux: number;
  readonly faceKeys: readonly string[];
}

/**
 * Deterministic sparse contract oracle for a connected adaptive MAC graph.
 * It allocates only by live region/fragment count and fails closed on duplicate
 * physical faces, malformed dyadic spans, invalid incidence, or non-finite
 * flux.  This is intentionally independent of a finest-box allocation.
 */
export function auditOctreeConnectedFlow(
  regions: readonly OctreeFlowRegion[],
  fragments: readonly OctreeFlowFaceFragment[],
  source: string,
  sink: string,
): OctreeConnectedFlowAudit {
  const ids = new Set<string>();
  for (const region of regions) {
    if (!region.id || ids.has(region.id)) throw new RangeError("Flow regions require unique non-empty IDs");
    ids.add(region.id);
  }
  if (!ids.has(source) || !ids.has(sink)) throw new RangeError("Flow source and sink must be live regions");

  const adjacency = new Map([...ids].map((id) => [id, new Set<string>()]));
  const netOutwardFlux = new Map([...ids].map((id) => [id, 0]));
  const keys = new Set<string>();
  for (const face of fragments) {
    if (face.axis < 0 || face.axis > 2 || !Number.isSafeInteger(face.span)
      || face.span < 1 || (face.span & (face.span - 1)) !== 0) {
      throw new RangeError("Flow face fragments require a valid axis and dyadic span");
    }
    if (!Number.isFinite(face.area) || face.area <= 0 || !Number.isFinite(face.normalVelocity)) {
      throw new RangeError("Flow face fragments require finite positive area and finite velocity");
    }
    if (face.negative === null && face.positive === null) throw new RangeError("Flow face fragment has no incident region");
    if (face.negative !== null && (!ids.has(face.negative) || face.negative === face.positive)) {
      throw new RangeError("Flow face fragment has invalid negative incidence");
    }
    if (face.positive !== null && !ids.has(face.positive)) throw new RangeError("Flow face fragment has invalid positive incidence");
    const axisSpan = face.axis | (face.span << 2);
    const key = [...encodeOctreeFaceKey(face.origin, axisSpan)].join(":");
    if (keys.has(key)) throw new RangeError("Flow topology contains a duplicate canonical face fragment");
    keys.add(key);

    const flux = face.area * face.normalVelocity;
    if (face.negative !== null) netOutwardFlux.set(face.negative, netOutwardFlux.get(face.negative)! + flux);
    if (face.positive !== null) netOutwardFlux.set(face.positive, netOutwardFlux.get(face.positive)! - flux);
    if (face.negative !== null && face.positive !== null) {
      adjacency.get(face.negative)!.add(face.positive);
      adjacency.get(face.positive)!.add(face.negative);
    }
  }

  const visited = new Set<string>([source]);
  const queue = [source];
  for (let cursor = 0; cursor < queue.length; cursor += 1) {
    for (const neighbor of adjacency.get(queue[cursor])!) {
      if (!visited.has(neighbor)) { visited.add(neighbor); queue.push(neighbor); }
    }
  }
  return {
    connected: visited.has(sink),
    visited,
    netOutwardFlux,
    totalBoundaryFlux: [...netOutwardFlux.values()].reduce((sum, flux) => sum + flux, 0),
    faceKeys: [...keys].sort(),
  };
}
