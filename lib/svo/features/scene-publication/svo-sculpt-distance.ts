import type { Vec3 } from "../../../core/model";
import { SVO_SCULPT_CUT_HEADER_WORDS, SVO_SCULPT_CUT_RECORD_WORDS } from "./svo-sculpt-cuts";

interface Cut {
  x: number; y: number; z: number;
  rx: number; ry: number; rz: number;
  round: boolean;
}
interface Node {
  lo: number[]; hi: number[];
  scale: number; depth: number;
  begin: number; end: number;
  left?: Node; right?: Node;
}

/** Immutable cut queries shared by all ray-march steps for a picked object.
 * The hierarchy only rejects provably farther cuts; it preserves the exact
 * box/ellipsoid union field, including overlapping and anisotropic stamps. */
export function prepareSvoSculptCutDistance(words: Uint32Array): (point: Vec3) => number {
  const f = new Float32Array(words.buffer, words.byteOffset, words.length);
  const cuts: Cut[] = Array.from({ length: words[0] ?? 0 }, (_, i) => {
    const at = SVO_SCULPT_CUT_HEADER_WORDS + i * SVO_SCULPT_CUT_RECORD_WORDS;
    return { x: .5 * (f[at]! + f[at + 4]!), y: .5 * (f[at + 1]! + f[at + 5]!), z: .5 * (f[at + 2]! + f[at + 6]!),
      rx: .5 * (f[at + 4]! - f[at]!), ry: .5 * (f[at + 5]! - f[at + 1]!), rz: .5 * (f[at + 6]! - f[at + 2]!),
      round: words[at + 3] === 2 };
  });
  if (!cuts.length) return () => 1e20;
  function build(begin: number, end: number): Node {
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    let scale = 1, depth = 0;
    for (let i = begin; i < end; i++) {
      const c = cuts[i]!, centre = [c.x, c.y, c.z], radius = [c.rx, c.ry, c.rz];
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a]!, centre[a]! - radius[a]!); hi[a] = Math.max(hi[a]!, centre[a]! + radius[a]!); }
      const small = Math.min(...radius);
      depth = Math.max(depth, small);
      if (c.round) scale = Math.min(scale, small / Math.max(...radius));
    }
    const node: Node = { lo, hi, scale, depth, begin, end };
    if (end - begin > 8) {
      const span = hi.map((v, a) => v - lo[a]!);
      const axis = span[0]! >= span[1]! && span[0]! >= span[2]! ? "x" : span[1]! >= span[2]! ? "y" : "z";
      const sorted = cuts.slice(begin, end).sort((a, b) => a[axis] - b[axis]);
      for (let i = 0; i < sorted.length; i++) cuts[begin + i] = sorted[i]!;
      const middle = (begin + end) >>> 1;
      node.left = build(begin, middle); node.right = build(middle, end);
    }
    return node;
  }
  const root = build(0, cuts.length);
  return point => {
    let nearest = 1e20;
    const bound = (node: Node) => {
      const x = Math.max(node.lo[0]! - point.x, point.x - node.hi[0]!, 0);
      const y = Math.max(node.lo[1]! - point.y, point.y - node.hi[1]!, 0);
      const z = Math.max(node.lo[2]! - point.z, point.z - node.hi[2]!, 0);
      // Outside the box an ellipsoid's conservative field is at least the
      // box distance times minRadius/maxRadius. Inside, no cut is deeper
      // than its smallest radius. A plain world-space AABB distance alone
      // would incorrectly reject stretched ellipsoids.
      return x > 0 || y > 0 || z > 0 ? Math.hypot(x, y, z) * node.scale : -node.depth;
    };
    const visit = (node: Node): void => {
      if (node.left && node.right) {
        const left = bound(node.left), right = bound(node.right);
        const first = left <= right ? node.left : node.right, second = left <= right ? node.right : node.left;
        if (Math.min(left, right) <= nearest + 1e-12) visit(first);
        if (Math.max(left, right) <= nearest + 1e-12) visit(second);
        return;
      }
      for (let i = node.begin; i < node.end; i++) {
        const c = cuts[i]!, x = point.x - c.x, y = point.y - c.y, z = point.z - c.z;
        let distance: number;
        if (c.round) distance = (Math.hypot(x / c.rx, y / c.ry, z / c.rz) - 1) * Math.min(c.rx, c.ry, c.rz);
        else {
          const qx = Math.abs(x) - c.rx, qy = Math.abs(y) - c.ry, qz = Math.abs(z) - c.rz;
          distance = Math.hypot(Math.max(qx, 0), Math.max(qy, 0), Math.max(qz, 0)) + Math.min(Math.max(qx, qy, qz), 0);
        }
        nearest = Math.min(nearest, distance);
      }
    };
    visit(root);
    return nearest;
  };
}
