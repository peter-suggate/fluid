import type { Vec3 } from "./model";

/**
 * Most outline edges one extrusion may carry.
 *
 * The solver evaluates an extrusion on the GPU from a fixed tail of its
 * parameter block, so this is a storage size and not a quality setting: an
 * outline over it is refused when the scene is validated rather than being
 * thinned to fit.
 */
export const LIQUID_EXTRUSION_MAX_EDGES = 64;

/**
 * A flat outline, thickened and extruded along world z.
 *
 * This is the shape a bevelled text object makes: a closed figure in the
 * x/y plane, grown outward by `offset_m`, given depth, with the front and back
 * edges rounded. A letter is the motivating case, but nothing here knows about
 * letters; any closed polygon set is an outline.
 */
export interface LiquidExtrusion {
  shape: "extrusion";
  /**
   * Closed contours in the world x/y plane, each a flat `[x0, y0, x1, y1, ...]`
   * in metres with the closing edge implied. The fill is even-odd, so a
   * contour inside another is a hole (the counter of an A, B or D).
   */
  contours_m: number[][];
  /** World z of the mid-plane. */
  centerZ_m: number;
  /** Half the extruded depth, measured to the flat faces. */
  halfDepth_m: number;
  /** How far the outline is grown outward in the x/y plane before extruding. */
  offset_m: number;
  /** Radius of the rounded front and back edges; at most the half depth. */
  edgeRadius_m: number;
}

/** Every outline edge as `[ax, ay, bx, by]`, the layout the solver uploads. */
export function liquidExtrusionEdges(extrusion: LiquidExtrusion): Float32Array<ArrayBuffer> {
  const count = extrusion.contours_m.reduce((sum, contour) => sum + contour.length / 2, 0);
  const edges = new Float32Array(4 * count);
  let offset = 0;
  for (const contour of extrusion.contours_m) {
    const points = contour.length / 2;
    for (let index = 0; index < points; index++) {
      const next = (index + 1) % points;
      edges[offset++] = contour[2 * index]!;
      edges[offset++] = contour[2 * index + 1]!;
      edges[offset++] = contour[2 * next]!;
      edges[offset++] = contour[2 * next + 1]!;
    }
  }
  return edges;
}

/** Why this is not an extrusion the solver can take, or undefined when it is. */
export function liquidExtrusionRefusal(extrusion: LiquidExtrusion): string | undefined {
  if (!extrusion || !Array.isArray(extrusion.contours_m) || extrusion.contours_m.length === 0) return "needs at least one contour";
  let edges = 0;
  for (const contour of extrusion.contours_m) {
    if (!Array.isArray(contour) || contour.length < 6 || contour.length % 2 !== 0) return "contours need at least three x/y points";
    if (!contour.every(Number.isFinite)) return "contour points must be finite";
    // A repeated point is an edge of no length, which has no nearest point.
    for (let index = 0; index < contour.length; index += 2) {
      const next = (index + 2) % contour.length;
      if (contour[index] === contour[next] && contour[index + 1] === contour[next + 1]) return "consecutive contour points must differ";
    }
    edges += contour.length / 2;
  }
  if (edges > LIQUID_EXTRUSION_MAX_EDGES) return `has ${edges} outline edges, over the ${LIQUID_EXTRUSION_MAX_EDGES}-edge limit`;
  if (![extrusion.centerZ_m, extrusion.halfDepth_m, extrusion.offset_m, extrusion.edgeRadius_m].every(Number.isFinite)) return "depth, offset and edge radius must be finite";
  if (!(extrusion.halfDepth_m > 0)) return "half depth must be positive";
  if (extrusion.offset_m < 0) return "offset must not be negative";
  if (extrusion.edgeRadius_m < 0 || extrusion.edgeRadius_m > extrusion.halfDepth_m) return "edge radius must lie between zero and the half depth";
  return undefined;
}

/** Signed distance from an x/y point to the raw outline; negative inside. */
function outlineDistance(edges: Float32Array, x: number, y: number): number {
  let nearest = Infinity, inside = false;
  for (let offset = 0; offset < edges.length; offset += 4) {
    const ax = edges[offset]!, ay = edges[offset + 1]!, bx = edges[offset + 2]!, by = edges[offset + 3]!;
    const ex = bx - ax, ey = by - ay, px = x - ax, py = y - ay;
    const along = Math.min(1, Math.max(0, (px * ex + py * ey) / (ex * ex + ey * ey)));
    nearest = Math.min(nearest, (px - along * ex) ** 2 + (py - along * ey) ** 2);
    // Even-odd crossing of the ray toward +x; the half-open test counts a
    // shared vertex once.
    if ((ay > y) !== (by > y) && x < ax + (y - ay) * ex / ey) inside = !inside;
  }
  return inside ? -Math.sqrt(nearest) : Math.sqrt(nearest);
}

/** Rounded-extrusion distance from the offset outline distance and |z|. */
function extrudedDistance(extrusion: LiquidExtrusion, planar: number, depth: number): number {
  const r = extrusion.edgeRadius_m;
  const wx = planar - extrusion.offset_m + r, wy = depth - extrusion.halfDepth_m + r;
  return Math.min(Math.max(wx, wy), 0) + Math.hypot(Math.max(wx, 0), Math.max(wy, 0)) - r;
}

/** Signed distance in metres, the same convention as the GPU source: negative is liquid. */
export function liquidExtrusionDistance(extrusion: LiquidExtrusion, point: Vec3, edges = liquidExtrusionEdges(extrusion)): number {
  return extrudedDistance(extrusion, outlineDistance(edges, point.x, point.y), Math.abs(point.z - extrusion.centerZ_m));
}

/** The box that contains the liquid, offset included. */
export function liquidExtrusionBounds(extrusion: LiquidExtrusion): { min_m: Vec3; max_m: Vec3 } {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const contour of extrusion.contours_m) for (let index = 0; index < contour.length; index += 2) {
    minX = Math.min(minX, contour[index]!); maxX = Math.max(maxX, contour[index]!);
    minY = Math.min(minY, contour[index + 1]!); maxY = Math.max(maxY, contour[index + 1]!);
  }
  const o = extrusion.offset_m;
  return {
    min_m: { x: minX - o, y: minY - o, z: extrusion.centerZ_m - extrusion.halfDepth_m },
    max_m: { x: maxX + o, y: maxY + o, z: extrusion.centerZ_m + extrusion.halfDepth_m },
  };
}

/**
 * The liquid's volume in cubic metres.
 *
 * Offsetting a non-convex outline has no closed form, so the x/y plane is
 * sampled at `resolution_m`; the depth each sample carries is exact, because
 * through the thickness the shape is the flat faces plus a quarter-round.
 */
export function liquidExtrusionVolume_m3(extrusion: LiquidExtrusion, resolution_m: number): number {
  const edges = liquidExtrusionEdges(extrusion), { min_m, max_m } = liquidExtrusionBounds(extrusion);
  const r = extrusion.edgeRadius_m, half = extrusion.halfDepth_m;
  const nx = Math.max(1, Math.ceil((max_m.x - min_m.x) / resolution_m)), ny = Math.max(1, Math.ceil((max_m.y - min_m.y) / resolution_m));
  const dx = (max_m.x - min_m.x) / nx, dy = (max_m.y - min_m.y) / ny;
  let volume = 0;
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const planar = outlineDistance(edges, min_m.x + (i + 0.5) * dx, min_m.y + (j + 0.5) * dy) - extrusion.offset_m;
    if (planar >= 0) continue;
    const inset = planar + r;
    volume += 2 * (inset <= 0 ? half : half - r + Math.sqrt(r * r - inset * inset)) * dx * dy;
  }
  return volume;
}
