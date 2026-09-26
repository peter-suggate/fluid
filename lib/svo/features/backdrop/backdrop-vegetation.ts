/**
 * The backdrop's vegetation, and the combined signed-distance queries over
 * ground and plants that the octree builder refines against.
 *
 * Every plant is a handful of analytic primitives in the porcelain garden's
 * idiom — soft, rounded, unglazed — and nothing else:
 *
 *   - a **cloud tree** is a trunk capsule plus 3-5 ellipsoid puffs, the puffs
 *     smooth-unioned into one crown;
 *   - a **shrub** is 1-3 smooth-unioned, slightly flattened puffs;
 *   - a **tuft** is 2-4 thin vertical cones;
 *   - a **pebble** is a half-sunk ellipsoid.
 *
 * Each is seated on `backdropFieldHeight`, never on `terrainHeightAt`: the set's
 * terrain sampler clamps to the pond's edge, so anything outside the footprint
 * seated through it would float or sink by the slab's slope.
 *
 * All distance functions here are Lipschitz-1 *lower bounds* on the true
 * distance (exact for capsules and cones; `(|p/r| - 1) * min(r)` for
 * ellipsoids, whose gradient is at most 1). The polynomial smooth minimum of
 * Lipschitz-1 functions is Lipschitz-1 as well. That is the property the octree
 * needs: `|f(centre)| > halfDiagonal` proves the surface misses the box.
 *
 * No DOM or GPU imports: this runs in the render worker.
 */
import {
  backdropFieldHeight,
  backdropFootprintDistance,
  backdropHeightBounds,
  backdropRandom,
  type BackdropField,
} from "./backdrop-field";
import { BACKDROP_MAXIMUM_ITEMS_PER_CLASS, type BackdropVegetationClass } from "../../../core/backdrop";

/** Per-voxel material codes (two bits in the leaf payload). Air is an occupancy bit, not a code. */
export const BACKDROP_MATERIAL = Object.freeze({ hill: 0, trunk: 1, foliage: 2, pebble: 3 });
export type BackdropMaterialCode = 0 | 1 | 2 | 3;
/** Returned by occupancy queries for empty space. */
export const BACKDROP_AIR = -1;

export type BackdropPrimitive =
  | { kind: "capsule"; material: BackdropMaterialCode; blend: false; a: Vec3; b: Vec3; radius: number }
  | { kind: "ellipsoid"; material: BackdropMaterialCode; blend: boolean; centre: Vec3; radii: Vec3 }
  /** Vertical capped cone: `base` is the centre of its bottom disc. */
  | { kind: "cone"; material: BackdropMaterialCode; blend: false; base: Vec3; height: number; baseRadius: number; tipRadius: number };

type Vec3 = readonly [number, number, number];

export interface BackdropBox { minimum: Vec3; maximum: Vec3 }

export type BackdropItemKind = "tree" | "shrub" | "tuft" | "pebble";

export interface BackdropItem {
  kind: BackdropItemKind;
  /** Primitive range `[primitiveStart, primitiveStart + primitiveCount)`. */
  primitiveStart: number;
  primitiveCount: number;
  /** Smooth-union radius for the item's `blend` primitives (the crown). */
  smooth_m: number;
  /** Conservative bound on the item's solid, smooth-union bulge included. */
  bounds: BackdropBox;
}

export interface BackdropExpansion {
  field: BackdropField;
  items: readonly BackdropItem[];
  primitives: readonly BackdropPrimitive[];
  counts: Readonly<Record<BackdropItemKind, number>>;
}

const CLASS_STREAM: Record<BackdropItemKind, number> = { tree: 11, shrub: 12, tuft: 13, pebble: 14 };

// ---------------------------------------------------------------------------
// Primitive distances

function capsuleDistance(p: Vec3, a: Vec3, b: Vec3, radius: number): number {
  const pax = p[0] - a[0], pay = p[1] - a[1], paz = p[2] - a[2];
  const bax = b[0] - a[0], bay = b[1] - a[1], baz = b[2] - a[2];
  const t = Math.min(1, Math.max(0, (pax * bax + pay * bay + paz * baz) / (bax * bax + bay * bay + baz * baz)));
  return Math.hypot(pax - bax * t, pay - bay * t, paz - baz * t) - radius;
}

function ellipsoidDistance(p: Vec3, centre: Vec3, radii: Vec3): number {
  const k = Math.hypot((p[0] - centre[0]) / radii[0], (p[1] - centre[1]) / radii[1], (p[2] - centre[2]) / radii[2]);
  return (k - 1) * Math.min(radii[0], radii[1], radii[2]);
}

/** Exact capped cone (Quilez), vertical, base disc at `base`. */
function coneDistance(p: Vec3, base: Vec3, height: number, r1: number, r2: number): number {
  const h = 0.5 * height;
  const qx = Math.hypot(p[0] - base[0], p[2] - base[2]);
  const qy = p[1] - (base[1] + h);
  const k2x = r2 - r1, k2y = 2 * h;
  const cax = qx - Math.min(qx, qy < 0 ? r1 : r2), cay = Math.abs(qy) - h;
  const t = Math.min(1, Math.max(0, ((r2 - qx) * k2x + (h - qy) * k2y) / (k2x * k2x + k2y * k2y)));
  const cbx = qx - r2 + k2x * t, cby = qy - h + k2y * t;
  const sign = cbx < 0 && cay < 0 ? -1 : 1;
  return sign * Math.sqrt(Math.min(cax * cax + cay * cay, cbx * cbx + cby * cby));
}

export function backdropPrimitiveDistance(primitive: BackdropPrimitive, p: Vec3): number {
  switch (primitive.kind) {
    case "capsule": return capsuleDistance(p, primitive.a, primitive.b, primitive.radius);
    case "ellipsoid": return ellipsoidDistance(p, primitive.centre, primitive.radii);
    case "cone": return coneDistance(p, primitive.base, primitive.height, primitive.baseRadius, primitive.tipRadius);
  }
}

function primitiveBounds(primitive: BackdropPrimitive): BackdropBox {
  switch (primitive.kind) {
    case "capsule": {
      const { a, b, radius } = primitive;
      return {
        minimum: [Math.min(a[0], b[0]) - radius, Math.min(a[1], b[1]) - radius, Math.min(a[2], b[2]) - radius],
        maximum: [Math.max(a[0], b[0]) + radius, Math.max(a[1], b[1]) + radius, Math.max(a[2], b[2]) + radius],
      };
    }
    case "ellipsoid": {
      const { centre: c, radii: r } = primitive;
      return { minimum: [c[0] - r[0], c[1] - r[1], c[2] - r[2]], maximum: [c[0] + r[0], c[1] + r[1], c[2] + r[2]] };
    }
    case "cone": {
      const { base: b, height, baseRadius, tipRadius } = primitive;
      const r = Math.max(baseRadius, tipRadius);
      return { minimum: [b[0] - r, b[1], b[2] - r], maximum: [b[0] + r, b[1] + height, b[2] + r] };
    }
  }
}

/** Polynomial smooth minimum; Lipschitz-1 when both inputs are. */
function smoothMin(a: number, b: number, k: number): number {
  const h = Math.max(k - Math.abs(a - b), 0) / k;
  return Math.min(a, b) - h * h * k * 0.25;
}

/** Signed distance bound and owning material of one item at `p`. */
export function backdropItemDistance(
  expansion: Pick<BackdropExpansion, "primitives">,
  item: BackdropItem,
  p: Vec3,
): { distance: number; material: BackdropMaterialCode } {
  let solid = Number.POSITIVE_INFINITY, solidMaterial: BackdropMaterialCode = BACKDROP_MATERIAL.hill;
  let blended = Number.POSITIVE_INFINITY, blendMaterial: BackdropMaterialCode = BACKDROP_MATERIAL.foliage;
  const end = item.primitiveStart + item.primitiveCount;
  for (let index = item.primitiveStart; index < end; index += 1) {
    const primitive = expansion.primitives[index];
    const d = backdropPrimitiveDistance(primitive, p);
    if (primitive.blend) {
      blended = blended === Number.POSITIVE_INFINITY ? d : smoothMin(blended, d, item.smooth_m);
      blendMaterial = primitive.material;
    } else if (d < solid) {
      solid = d;
      solidMaterial = primitive.material;
    }
  }
  return blended < solid ? { distance: blended, material: blendMaterial } : { distance: solid, material: solidMaterial };
}

function unionBounds(boxes: readonly BackdropBox[], pad: number): BackdropBox {
  const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
  for (const box of boxes) for (let axis = 0; axis < 3; axis += 1) {
    minimum[axis] = Math.min(minimum[axis], box.minimum[axis] - pad);
    maximum[axis] = Math.max(maximum[axis], box.maximum[axis] + pad);
  }
  return { minimum: minimum as unknown as Vec3, maximum: maximum as unknown as Vec3 };
}

// ---------------------------------------------------------------------------
// Placement

interface Placement { x: number; z: number; size: number; radial: number; random: (stream: number) => number }

/**
 * Jittered-grid placement over an annulus: one candidate per cell of area
 * `1 / density`, uniform within its cell, kept if it lands in the annulus and
 * clear of the footprint. Deterministic in (seed, class, cell) alone.
 *
 * Size is biased outward — half uniform draw, half radial fraction — so the
 * largest trees stand far away, where they fill the horizon, and nothing tall
 * crowds the set.
 */
function place(
  field: BackdropField,
  kind: BackdropItemKind,
  spec: BackdropVegetationClass,
  clearance: (size: number) => number,
): Placement[] {
  if (spec.density_m2 === 0) return [];
  const { seed } = field.description;
  const stream = CLASS_STREAM[kind];
  const [inner, outer] = spec.radius_m;
  const cell = 1 / Math.sqrt(spec.density_m2);
  const cells = Math.ceil(outer / cell);
  const placements: Placement[] = [];
  for (let j = -cells; j < cells; j += 1) {
    for (let i = -cells; i < cells; i += 1) {
      const x = (i + backdropRandom(seed, stream, i, j, 0)) * cell;
      const z = (j + backdropRandom(seed, stream, i, j, 1)) * cell;
      const radial = Math.hypot(x, z);
      if (radial < inner || radial > outer) continue;
      const t = (radial - inner) / (outer - inner);
      const size = spec.size_m[0] + (spec.size_m[1] - spec.size_m[0]) * (0.5 * backdropRandom(seed, stream, i, j, 2) + 0.5 * t);
      if (backdropFootprintDistance(field.seam, x, z) < clearance(size)) continue;
      placements.push({ x, z, size, radial, random: (salt) => backdropRandom(seed, stream, i, j, 16 + salt) });
      if (placements.length > BACKDROP_MAXIMUM_ITEMS_PER_CLASS) {
        throw new RangeError(`Backdrop ${kind} placement exceeded ${BACKDROP_MAXIMUM_ITEMS_PER_CLASS} items`);
      }
    }
  }
  return placements;
}

/**
 * The lowest ground under a footprint of radius `r`: five samples, so a plant
 * on a slope sinks its uphill side rather than floating on its downhill one.
 */
function seatHeight(field: BackdropField, x: number, z: number, r: number): number {
  return Math.min(
    backdropFieldHeight(field, x, z),
    backdropFieldHeight(field, x + r, z), backdropFieldHeight(field, x - r, z),
    backdropFieldHeight(field, x, z + r), backdropFieldHeight(field, x, z - r),
  );
}

/** Expand a compiled field's vegetation into primitives. Pure and deterministic. */
export function expandBackdrop(field: BackdropField): BackdropExpansion {
  const primitives: BackdropPrimitive[] = [];
  const items: BackdropItem[] = [];
  const counts: Record<BackdropItemKind, number> = { tree: 0, shrub: 0, tuft: 0, pebble: 0 };
  const { vegetation } = field.description;
  const push = (kind: BackdropItemKind, parts: BackdropPrimitive[], smooth_m: number) => {
    items.push({
      kind,
      primitiveStart: primitives.length,
      primitiveCount: parts.length,
      smooth_m,
      bounds: unionBounds(parts.map(primitiveBounds), 0.25 * smooth_m),
    });
    primitives.push(...parts);
    counts[kind] += 1;
  };

  for (const at of place(field, "tree", vegetation.trees, (size) => 0.4 * size)) {
    const H = at.size;
    const y0 = seatHeight(field, at.x, at.z, 0.06 * H);
    const leanX = (at.random(0) - 0.5) * 0.1 * H, leanZ = (at.random(1) - 0.5) * 0.1 * H;
    const top: Vec3 = [at.x + leanX, y0 + 0.55 * H, at.z + leanZ];
    const parts: BackdropPrimitive[] = [{
      kind: "capsule", material: BACKDROP_MATERIAL.trunk, blend: false,
      a: [at.x, y0 - 0.05 * H, at.z], b: top, radius: 0.045 * H,
    }];
    const crown: Vec3 = [top[0], top[1] + 0.15 * H, top[2]];
    parts.push({ kind: "ellipsoid", material: BACKDROP_MATERIAL.foliage, blend: true, centre: crown, radii: [0.26 * H, 0.21 * H, 0.26 * H] });
    const puffs = 2 + Math.floor(at.random(2) * 3);
    const phase = 2 * Math.PI * at.random(3);
    for (let index = 0; index < puffs; index += 1) {
      const angle = phase + 2 * Math.PI * index / puffs;
      const r = (0.16 + 0.06 * at.random(4 + index)) * H;
      parts.push({
        kind: "ellipsoid", material: BACKDROP_MATERIAL.foliage, blend: true,
        centre: [crown[0] + 0.2 * H * Math.cos(angle), crown[1] + (at.random(8 + index) * 0.12 - 0.05) * H, crown[2] + 0.2 * H * Math.sin(angle)],
        radii: [r, 0.8 * r, r],
      });
    }
    push("tree", parts, 0.1 * H);
  }

  for (const at of place(field, "shrub", vegetation.shrubs, (size) => 0.6 * size)) {
    const H = at.size;
    const y0 = seatHeight(field, at.x, at.z, 0.3 * H);
    const parts: BackdropPrimitive[] = [];
    const puffs = 1 + Math.floor(at.random(0) * 3);
    const phase = 2 * Math.PI * at.random(1);
    for (let index = 0; index < puffs; index += 1) {
      const angle = phase + 2 * Math.PI * index / puffs;
      const offset = puffs === 1 ? 0 : 0.35 * H;
      const r = (0.45 + 0.15 * at.random(2 + index)) * H;
      parts.push({
        kind: "ellipsoid", material: BACKDROP_MATERIAL.foliage, blend: true,
        centre: [at.x + offset * Math.cos(angle), y0 + 0.4 * H, at.z + offset * Math.sin(angle)],
        radii: [r, 0.5 * H, r],
      });
    }
    push("shrub", parts, 0.2 * H);
  }

  for (const at of place(field, "tuft", vegetation.tufts, (size) => 0.3 * size)) {
    const H = at.size;
    const y0 = seatHeight(field, at.x, at.z, 0.2 * H) - 0.05 * H;
    const parts: BackdropPrimitive[] = [];
    const blades = 2 + Math.floor(at.random(0) * 3);
    for (let index = 0; index < blades; index += 1) {
      const angle = 2 * Math.PI * at.random(1 + index);
      const offset = 0.2 * H * at.random(5 + index);
      parts.push({
        kind: "cone", material: BACKDROP_MATERIAL.foliage, blend: false,
        base: [at.x + offset * Math.cos(angle), y0, at.z + offset * Math.sin(angle)],
        height: (0.6 + 0.4 * at.random(9 + index)) * H * 1.05,
        baseRadius: 0.15 * H, tipRadius: 0.01 * H,
      });
    }
    push("tuft", parts, 0);
  }

  for (const at of place(field, "pebble", vegetation.pebbles, (size) => 0.5 * size)) {
    const D = at.size;
    const y0 = seatHeight(field, at.x, at.z, 0.4 * D);
    push("pebble", [{
      kind: "ellipsoid", material: BACKDROP_MATERIAL.pebble, blend: false,
      centre: [at.x, y0 + 0.1 * D, at.z],
      radii: [0.5 * D * (0.8 + 0.4 * at.random(0)), 0.3 * D, 0.5 * D * (0.7 + 0.4 * at.random(1))],
    }], 0);
  }

  return { field, items, primitives, counts };
}

// ---------------------------------------------------------------------------
// Queries over ground and plants together

function outsideContent(field: BackdropField, x: number, z: number): boolean {
  const [minX, minZ, maxX, maxZ] = field.seam.footprint_m;
  // The footprint is a hole: the set owns it. Beyond the outer radius the
  // world simply ends (the phase-2 pass hazes the analytic plane out there).
  return (x > minX && x < maxX && z > minZ && z < maxZ) || Math.hypot(x, z) > field.description.outerRadius_m;
}

/**
 * Material at a point, or `BACKDROP_AIR`. Vegetation wins over ground where
 * both are solid, so a sunk trunk base stays trunk-coloured.
 */
export function backdropOccupancy(expansion: BackdropExpansion, x: number, y: number, z: number): number {
  const { field } = expansion;
  if (outsideContent(field, x, z)) return BACKDROP_AIR;
  const p: Vec3 = [x, y, z];
  for (const item of expansion.items) {
    const b = item.bounds;
    if (x < b.minimum[0] || x > b.maximum[0] || y < b.minimum[1] || y > b.maximum[1] || z < b.minimum[2] || z > b.maximum[2]) continue;
    const hit = backdropItemDistance(expansion, item, p);
    if (hit.distance < 0) return hit.material;
  }
  return y < backdropFieldHeight(field, x, z) ? BACKDROP_MATERIAL.hill : BACKDROP_AIR;
}

/**
 * A Lipschitz-1 lower bound on the distance to the backdrop's surface (signed:
 * negative inside). The ground term is the vertical gap divided by
 * `sqrt(1 + L^2)`, which cannot exceed the true distance to a heightfield whose
 * slope is bounded by L. The footprint hole and outer radius are not modelled —
 * this is a bound on the solid, used for refinement and stepping.
 */
export function backdropSignedDistanceBound(expansion: BackdropExpansion, x: number, y: number, z: number): number {
  const { field } = expansion;
  let d = (y - backdropFieldHeight(field, x, z)) / Math.sqrt(1 + field.lipschitz * field.lipschitz);
  const p: Vec3 = [x, y, z];
  for (const item of expansion.items) d = Math.min(d, backdropItemDistance(expansion, item, p).distance);
  return d;
}

function boxesOverlap(a: BackdropBox, b: BackdropBox, pad: number): boolean {
  for (let axis = 0; axis < 3; axis += 1) {
    if (a.minimum[axis] - pad > b.maximum[axis] || b.minimum[axis] - pad > a.maximum[axis]) return false;
  }
  return true;
}

/** Can the ground's surface band (`+-margin`) pass through this box? */
export function backdropGroundMayCross(field: BackdropField, box: BackdropBox, margin: number): boolean {
  const [low, high] = backdropHeightBounds(field, box.minimum[0], box.minimum[2], box.maximum[0], box.maximum[2]);
  return low - margin <= box.maximum[1] && high + margin >= box.minimum[1];
}

/**
 * Filter `candidates` (item indices) to those whose surface band can pass
 * through the box: bounds first, then the centre distance against the half
 * diagonal. The octree passes each node's survivors down to its children, so
 * no spatial index is needed — the root starts with every item.
 */
export function backdropItemsCrossing(
  expansion: BackdropExpansion,
  box: BackdropBox,
  margin: number,
  candidates: readonly number[],
): number[] {
  const centre: Vec3 = [
    0.5 * (box.minimum[0] + box.maximum[0]),
    0.5 * (box.minimum[1] + box.maximum[1]),
    0.5 * (box.minimum[2] + box.maximum[2]),
  ];
  const halfDiagonal = 0.5 * Math.hypot(box.maximum[0] - box.minimum[0], box.maximum[1] - box.minimum[1], box.maximum[2] - box.minimum[2]);
  const survivors: number[] = [];
  for (const index of candidates) {
    const item = expansion.items[index];
    if (!boxesOverlap(item.bounds, box, margin)) continue;
    if (Math.abs(backdropItemDistance(expansion, item, centre).distance) > halfDiagonal + margin) continue;
    survivors.push(index);
  }
  return survivors;
}

/** Is the box wholly inside the footprint hole or wholly beyond the outer radius? */
export function backdropBoxOutsideContent(field: BackdropField, box: BackdropBox): boolean {
  const [minX, minZ, maxX, maxZ] = field.seam.footprint_m;
  if (box.minimum[0] >= minX && box.maximum[0] <= maxX && box.minimum[2] >= minZ && box.maximum[2] <= maxZ) return true;
  const nearX = Math.max(box.minimum[0], 0, -box.maximum[0]);
  const nearZ = Math.max(box.minimum[2], 0, -box.maximum[2]);
  return Math.hypot(nearX, nearZ) > field.description.outerRadius_m;
}

/**
 * "Can the surface band intersect this box?" — the octree refinement test,
 * standalone. The builder uses the same pieces with candidate lists inherited
 * from the parent; this form scans every item and suits one-off probes.
 */
export function backdropBoxMayContainSurface(expansion: BackdropExpansion, box: BackdropBox, margin: number): boolean {
  if (backdropBoxOutsideContent(expansion.field, box)) return false;
  if (backdropGroundMayCross(expansion.field, box, margin)) return true;
  return backdropItemsCrossing(expansion, box, margin, expansion.items.map((_, index) => index)).length > 0;
}
