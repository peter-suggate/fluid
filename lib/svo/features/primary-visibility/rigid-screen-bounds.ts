import type { Vec3 } from "../../../core/model";

export interface ScreenRect { x: number; y: number; width: number; height: number }
interface ViewBasis { position: Vec3; forward: Vec3; right: Vec3; up: Vec3 }

/** Bound the exact CPU display poses in pixel space. The camera-aligned box
 * encloses the rotated body, including custom render skins. A near-plane
 * crossing takes the full frame. */
export function rigidScreenBounds(records: Float32Array, count: number, view: ViewBasis,
  tanHalfFov: number, width: number, height: number): ScreenRect {
  let left = width, top = height, right = 0, bottom = 0;
  const focal = .5 * height / tanHalfFov;
  for (let i = 0; i < count; i++) {
    const at = i * 16;
    const x = records[at]! - view.position.x, y = records[at + 1]! - view.position.y, z = records[at + 2]! - view.position.z;
    const dot = (axis: Vec3) => x * axis.x + y * axis.y + z * axis.z;
    const radius = Math.max(records[at + 3]!, Math.hypot(records[at + 4]!, records[at + 5]!, records[at + 6]!));
    const depth = dot(view.forward);
    if (depth + radius <= 0) continue;
    if (!(depth - radius > .001)) return { x: 0, y: 0, width, height };
    for (const d of [depth - radius, depth + radius]) {
      const scale = focal / d, cx = width / 2 + dot(view.right) * scale, cy = height / 2 - dot(view.up) * scale;
      left = Math.min(left, Math.floor(cx - radius * scale) - 2);
      right = Math.max(right, Math.ceil(cx + radius * scale) + 2);
      top = Math.min(top, Math.floor(cy - radius * scale) - 2);
      bottom = Math.max(bottom, Math.ceil(cy + radius * scale) + 2);
    }
  }
  left = Math.max(0, left); top = Math.max(0, top);
  return { x: left, y: top, width: Math.max(0, Math.min(width, right) - left), height: Math.max(0, Math.min(height, bottom) - top) };
}

export function enclosingScreenRect(a: ScreenRect, b: ScreenRect): ScreenRect {
  if (!a.width || !a.height) return b;
  if (!b.width || !b.height) return a;
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  return { x, y, width: Math.max(a.x + a.width, b.x + b.width) - x, height: Math.max(a.y + a.height, b.y + b.height) - y };
}
