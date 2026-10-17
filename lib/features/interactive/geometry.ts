import type { Quaternion, Vec3 } from "../../core/model";
import { add, cross, dot, length, normalize, scale, sub } from "../../core/math";
import type { EntityRecord, InteractionCatalog, InteractionView, PreviewPrimitive, Transform } from "../../interaction";

/**
 * Placement and presentation geometry for interactive entities.
 *
 * Everything here is in terms of a kind's `PreviewPrimitive` and an authored
 * `Transform`, never a kind ID: the ghost, the outline, the click test and the
 * overlap bounds are the same arithmetic for a character as for a crate. The
 * primitive is origin-centred in local space, so "feet on the surface" is half
 * its height along the local up axis.
 */

export const IDENTITY_ORIENTATION: Quaternion = Object.freeze({ w: 1, x: 0, y: 0, z: 0 });
const UP: Vec3 = { x: 0, y: 1, z: 0 };

export function multiplyQuaternions(a: Quaternion, b: Quaternion): Quaternion {
  return {
    w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
  };
}

export function normalizeQuaternion(q: Quaternion): Quaternion {
  const l = Math.hypot(q.w, q.x, q.y, q.z);
  return l > 0 ? { w: q.w / l, x: q.x / l, y: q.y / l, z: q.z / l } : IDENTITY_ORIENTATION;
}

/** Rotation about +Y. A positive yaw carries +Z toward +X. */
export function yawQuaternion(yaw_rad: number): Quaternion {
  return { w: Math.cos(yaw_rad / 2), x: 0, y: Math.sin(yaw_rad / 2), z: 0 };
}

export function rotateVector(q: Quaternion, v: Vec3): Vec3 {
  const u = { x: q.x, y: q.y, z: q.z };
  const t = scale(cross(u, v), 2);
  return add(add(v, scale(t, q.w)), cross(u, t));
}

export function inverseRotate(q: Quaternion, v: Vec3): Vec3 {
  return rotateVector({ w: q.w, x: -q.x, y: -q.y, z: -q.z }, v);
}

/** The heading a transform faces, read off its local +Z. */
export function quaternionYaw(q: Quaternion): number {
  const forward = rotateVector(q, { x: 0, y: 0, z: 1 });
  return Math.atan2(forward.x, forward.z);
}

/** The shortest rotation carrying `from` onto `to`; both unit. */
export function quaternionBetween(from: Vec3, to: Vec3): Quaternion {
  const d = dot(from, to);
  if (d > 1 - 1e-9) return IDENTITY_ORIENTATION;
  if (d < -1 + 1e-9) {
    const axis = normalize(Math.abs(from.x) < 0.9 ? cross(from, { x: 1, y: 0, z: 0 }) : cross(from, { x: 0, y: 0, z: 1 }));
    return { w: 0, ...axis };
  }
  const axis = cross(from, to);
  return normalizeQuaternion({ w: 1 + d, x: axis.x, y: axis.y, z: axis.z });
}

/** Half of the primitive's extent along its own up axis. */
export function previewHalfHeight(preview: PreviewPrimitive): number {
  return preview.shape === "box" ? preview.dimensions_m.y / 2
    : preview.shape === "sphere" ? preview.radius_m : preview.height_m / 2;
}

export interface SurfaceHit {
  readonly point_m: Vec3;
  readonly normal: Vec3;
  /** True when the ray met no surface and the ground plane answered instead. */
  readonly fallback: boolean;
}

/**
 * Where a draft stands for a surface hit.
 *
 * `upright` keeps the local up axis on world up, whatever the slope — a
 * character stands, it does not lie along a hillside. `surface` turns local up
 * onto the surface normal. Either way the primitive's base sits on the point,
 * and `yaw_rad` turns it about its own up axis.
 */
export function placementTransform(
  alignment: "upright" | "surface",
  preview: PreviewPrimitive,
  hit: SurfaceHit,
  yaw_rad: number,
): Transform {
  const surface = normalize(hit.normal);
  const normal = alignment === "upright" || length(surface) === 0 ? UP : surface;
  const tilt = quaternionBetween(UP, normal);
  const orientation = normalizeQuaternion(multiplyQuaternions(tilt, yawQuaternion(yaw_rad)));
  return {
    position_m: add(hit.point_m, scale(normal, previewHalfHeight(preview))),
    orientation,
  };
}

/** Snap a horizontal position to a grid, leaving height to the surface. */
export function snapPosition(point_m: Vec3, step_m: number): Vec3 {
  if (!(step_m > 0)) return point_m;
  return { x: Math.round(point_m.x / step_m) * step_m, y: point_m.y, z: Math.round(point_m.z / step_m) * step_m };
}

/** Snap within the hit's tangent plane, so sloped surfaces keep their height. */
export function snapSurfaceHit(hit: SurfaceHit, step_m: number): SurfaceHit {
  const p=snapPosition(hit.point_m,step_m),n=hit.normal;
  if(step_m<=0)return hit;
  if(Math.abs(n.y)>.1)p.y=hit.point_m.y-(n.x*(p.x-hit.point_m.x)+n.z*(p.z-hit.point_m.z))/n.y;
  else {const d=dot(sub(p,hit.point_m),n)/Math.max(1e-12,dot(n,n));return {...hit,point_m:sub(p,scale(n,d))};}
  return {...hit,point_m:p};
}

export function rotationDegrees(q: Quaternion): Vec3 {
  return {x:Math.atan2(2*(q.w*q.x+q.y*q.z),1-2*(q.x*q.x+q.y*q.y))*180/Math.PI,
    y:Math.asin(Math.max(-1,Math.min(1,2*(q.w*q.y-q.z*q.x))))*180/Math.PI,
    z:Math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))*180/Math.PI};
}
export function rotationFromDegrees(v: Vec3): Quaternion {
  const [x,y,z]=[v.x,v.y,v.z].map(a=>a*Math.PI/360);
  return normalizeQuaternion(multiplyQuaternions({x:0,y:0,z:Math.sin(z),w:Math.cos(z)},multiplyQuaternions({x:0,y:Math.sin(y),z:0,w:Math.cos(y)},{x:Math.sin(x),y:0,z:0,w:Math.cos(x)})));
}

function circle(centre: Vec3, u: Vec3, v: Vec3, radius: number, segments = 28, from = 0, to = Math.PI * 2): Vec3[] {
  const points: Vec3[] = [];
  for (let i = 0; i <= segments; i += 1) {
    const a = from + ((to - from) * i) / segments;
    points.push(add(centre, add(scale(u, Math.cos(a) * radius), scale(v, Math.sin(a) * radius))));
  }
  return points;
}

/**
 * Wireframe paths for a primitive at a transform, in world metres.
 *
 * Paths rather than a mesh because the overlay is SVG strokes: a box is its
 * twelve edges, a sphere three great circles, a capsule its two rings, four
 * side lines and the hemisphere arcs that close it.
 */
export function previewOutline(preview: PreviewPrimitive, transform: Transform): Vec3[][] {
  const local = (p: Vec3) => add(transform.position_m, rotateVector(transform.orientation, p));
  const X: Vec3 = { x: 1, y: 0, z: 0 }, Y: Vec3 = UP, Z: Vec3 = { x: 0, y: 0, z: 1 };
  if (preview.shape === "box") {
    const h = scale(preview.dimensions_m, 0.5);
    const c = (sx: number, sy: number, sz: number) => local({ x: sx * h.x, y: sy * h.y, z: sz * h.z });
    const bottom = [c(-1, -1, -1), c(1, -1, -1), c(1, -1, 1), c(-1, -1, 1), c(-1, -1, -1)];
    const top = [c(-1, 1, -1), c(1, 1, -1), c(1, 1, 1), c(-1, 1, 1), c(-1, 1, -1)];
    const posts = ([[-1, -1], [1, -1], [1, 1], [-1, 1]] as const).map(([sx, sz]) => [c(sx, -1, sz), c(sx, 1, sz)]);
    // The facing tick: a short line on the +Z face, so a placed thing says which way it looks.
    const facing = [local({ x: 0, y: 0, z: h.z }), local({ x: 0, y: 0, z: h.z + Math.min(0.25, h.x) })];
    return [bottom, top, ...posts, facing];
  }
  if (preview.shape === "sphere") {
    const r = preview.radius_m;
    return [[X, Y], [X, Z], [Y, Z]].map(([u, v]) => circle({ x: 0, y: 0, z: 0 }, u, v, r).map(local));
  }
  const r = preview.radius_m, half = preview.height_m / 2 - r;
  const top: Vec3 = { x: 0, y: half, z: 0 }, bottom: Vec3 = { x: 0, y: -half, z: 0 };
  const paths = [
    circle(top, X, Z, r), circle(bottom, X, Z, r),
    ...([[X, 1], [X, -1], [Z, 1], [Z, -1]] as const).map(([axis, sign]) => [add(top, scale(axis, sign * r)), add(bottom, scale(axis, sign * r))]),
    circle(top, X, Y, r, 14, 0, Math.PI), circle(top, Z, Y, r, 14, 0, Math.PI),
    circle(bottom, X, Y, r, 14, Math.PI, Math.PI * 2), circle(bottom, Z, Y, r, 14, Math.PI, Math.PI * 2),
    [add(top, { x: 0, y: r * 0.3, z: 0 }), add(top, { x: 0, y: r * 0.3, z: r * 1.4 })],
  ];
  return paths.map((path) => path.map(local));
}

/** World-space bounds of a primitive at a transform. */
export function previewBounds(preview: PreviewPrimitive, transform: Transform): { min: Vec3; max: Vec3 } {
  const points = previewOutline(preview, transform).flat();
  const min = { x: Infinity, y: Infinity, z: Infinity }, max = { x: -Infinity, y: -Infinity, z: -Infinity };
  for (const p of points) {
    min.x = Math.min(min.x, p.x); min.y = Math.min(min.y, p.y); min.z = Math.min(min.z, p.z);
    max.x = Math.max(max.x, p.x); max.y = Math.max(max.y, p.y); max.z = Math.max(max.z, p.z);
  }
  return { min, max };
}

export function boundsOverlap(a: { min: Vec3; max: Vec3 }, b: { min: Vec3; max: Vec3 }, slack_m = 1e-3): boolean {
  return a.min.x < b.max.x - slack_m && b.min.x < a.max.x - slack_m
    && a.min.y < b.max.y - slack_m && b.min.y < a.max.y - slack_m
    && a.min.z < b.max.z - slack_m && b.min.z < a.max.z - slack_m;
}

export interface Ray { readonly origin: Vec3; readonly direction: Vec3 }

function raySphere(origin: Vec3, direction: Vec3, centre: Vec3, r: number): number | undefined {
  const oc = sub(origin, centre);
  const b = dot(oc, direction), c = dot(oc, oc) - r * r;
  const disc = b * b - c;
  if (disc < 0) return undefined;
  const s = Math.sqrt(disc);
  const t = -b - s >= 0 ? -b - s : -b + s;
  return t >= 0 ? t : undefined;
}

/**
 * Distance along a ray to a primitive at a transform, or undefined for a miss.
 *
 * The ray is taken into the primitive's local frame, where the box is a slab
 * test and the capsule a vertical cylinder closed by two spheres.
 */
export function rayPreviewDistance(ray: Ray, preview: PreviewPrimitive, transform: Transform): number | undefined {
  const origin = inverseRotate(transform.orientation, sub(ray.origin, transform.position_m));
  const direction = inverseRotate(transform.orientation, ray.direction);
  if (preview.shape === "sphere") return raySphere(origin, direction, { x: 0, y: 0, z: 0 }, preview.radius_m);
  if (preview.shape === "box") {
    const h = scale(preview.dimensions_m, 0.5);
    let near = -Infinity, far = Infinity;
    for (const axis of ["x", "y", "z"] as const) {
      if (Math.abs(direction[axis]) < 1e-12) {
        if (Math.abs(origin[axis]) > h[axis]) return undefined;
        continue;
      }
      const a = (-h[axis] - origin[axis]) / direction[axis], b = (h[axis] - origin[axis]) / direction[axis];
      near = Math.max(near, Math.min(a, b)); far = Math.min(far, Math.max(a, b));
    }
    if (near > far || far < 0) return undefined;
    return near >= 0 ? near : far;
  }
  const r = preview.radius_m, half = preview.height_m / 2 - r;
  let best: number | undefined;
  const consider = (t: number | undefined) => { if (t !== undefined && t >= 0 && (best === undefined || t < best)) best = t; };
  const a = direction.x * direction.x + direction.z * direction.z;
  if (a > 1e-12) {
    const b = origin.x * direction.x + origin.z * direction.z;
    const c = origin.x * origin.x + origin.z * origin.z - r * r;
    const disc = b * b - a * c;
    if (disc >= 0) {
      for (const t of [(-b - Math.sqrt(disc)) / a, (-b + Math.sqrt(disc)) / a]) {
        const y = origin.y + direction.y * t;
        if (Math.abs(y) <= half) consider(t);
      }
    }
  }
  consider(raySphere(origin, direction, { x: 0, y: half, z: 0 }, r));
  consider(raySphere(origin, direction, { x: 0, y: -half, z: 0 }, r));
  return best;
}

/** Normal at an actual primitive hit, in world space (including rotated boxes). */
export function previewNormalAt(preview: PreviewPrimitive, transform: Transform, point: Vec3): Vec3 {
  const local = inverseRotate(transform.orientation, sub(point, transform.position_m));
  let normal: Vec3;
  if (preview.shape === 'box') {
    const half = scale(preview.dimensions_m, .5);
    const axis = (['x', 'y', 'z'] as const).reduce((best, axis) =>
      Math.abs(Math.abs(local[axis]) - half[axis]) < Math.abs(Math.abs(local[best]) - half[best]) ? axis : best, 'x');
    normal = {x:0,y:0,z:0}; normal[axis] = local[axis] < 0 ? -1 : 1;
  } else {
    const half = preview.shape === 'capsule' ? Math.max(0, preview.height_m / 2 - preview.radius_m) : 0;
    normal = normalize(sub(local, {x:0,y:Math.max(-half,Math.min(half,local.y)),z:0}));
  }
  return rotateVector(transform.orientation, normal);
}

/** A ray's meeting with a horizontal plane at `height_m`, ahead of its origin. */
export function rayHorizontalPlane(ray: Ray, height_m: number): Vec3 | undefined {
  if (Math.abs(ray.direction.y) < 1e-9) return undefined;
  const t = (height_m - ray.origin.y) / ray.direction.y;
  return t > 0 ? add(ray.origin, scale(ray.direction, t)) : undefined;
}

export function isUnitVector(v: Vec3): boolean {
  return Math.abs(length(v) - 1) < 1e-4;
}

/** Pick the published pose; authored spawn poses are only a pre-snapshot fallback. */
export function pickInteractiveEntity(view: InteractionView, catalog: InteractionCatalog, ray: Ray, occluderDistance = Infinity): {entity: EntityRecord;distance:number} | undefined {
  let best: {entity:EntityRecord;distance:number}|undefined;
  for(const entity of view.document.entities){
    const kind=catalog.entityKind(entity.kind);if(!kind)continue;
    const pose=view.snapshot?.bodies.find(b=>b.entityId===entity.id)?.transform??entity.transform;
    const distance=rayPreviewDistance(ray,kind.preview(entity.config),pose);
    if(distance!==undefined&&distance<=occluderDistance+1e-4&&(!best||distance<best.distance))best={entity,distance};
  }
  return best;
}
