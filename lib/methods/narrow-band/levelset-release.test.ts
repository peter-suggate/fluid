// Lane E's release port (levelset-release.wgsl.ts): kept deep air and the released walls, with their producers, each
// against a Float64 CPU oracle of the old formulas, on a frozen tiered state.
// - The state: the floor-slab boot with h tiles committed at the floor corner and beside the x = 0 wall (some of
//   them with no F brick), an analytic φ on every F and C copy, US zero except in two moving blobs, and U with
//   lane P's release bits (FLAGS bit 8 + a) at wall faces: released, toward the wall, below the speed threshold, and
//   unreleased.
// - The producers: nrReach + nrSpread x3 give the reach words (both halves, the step stamp and the plan speed), and
//   nrWallReach gives the wall tiles and the plane reach. Every other set word, and the vel, phi, dir and fcell
//   buffers, must stay bitwise unchanged.
// - The readers on the GPU (a probe over every F and C vertex, h and 4h widths) against the oracle.
// - lcAdvect before the producers (no words: nothing fires) and after: each written C vertex is kept deep air
//   (bitwise its φ), raised by the released walls (the oracle's umReleasedWalls of the before value), or bitwise
//   the before value.
import assert from "node:assert/strict";
import test from "node:test";
import { ARGS, BRICK_STRIDE, BrickLayout, CMASK_WORDS, H, LIST, NONE, REC, RING, RING_WORDS, SEAM, brickAbiWGSL, brickKeyCoord } from "./brick-abi";
import { bringUpCG } from "./brick-bringup-cg";
import { BrickEngine } from "./brick-engine";
import { type BrickBinding, CFG, declareBindings } from "./brick-kernel";
import { snapshot } from "./brick-readback";
import { brickSetKernels } from "./brick-set.wgsl";
import { bringUpStages } from "./brick-stages";
import { brickTransferKernels } from "./brick-transfer.wgsl";
import { fluidLayout } from "./fluid-grid.wgsl";
import { LC_ADVECT_GROUPS, levelSetCoarseKernels } from "./levelset-coarse.wgsl";
import { levelSetReleaseKernels, levelSetReleaseSchedule, levelSetReleaseWGSL } from "./levelset-release.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

const GPU = { skip: !process.env.WEBGPU_NODE_MODULE };
type V = [number, number, number];
const N = 64, NL: V = [N, N, N], DT = 1 / 60, HM = 1 / N;
const POISON = 0x7fc0beef;

const hash = (...k: number[]) => { let h = 0x9e3779b9; for (const x of k) { h = Math.imul(h ^ (x | 0), 0x85ebca6b); h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35); h ^= h >>> 16; } return (h >>> 0) / 4294967296; };
const fbits = (x: number) => { const b = new Float32Array(1); b[0] = x; return new Uint32Array(b.buffer)[0]!; };
const bitsf = (u: number) => { const b = new Uint32Array(1); b[0] = u; return new Float32Array(b.buffer)[0]!; };
const add = (a: V, b: V): V => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const mul = (a: V, s: number): V => [a[0] * s, a[1] * s, a[2] * s];
const shr = (a: V, n: number): V => [a[0] >> n, a[1] >> n, a[2] >> n];
const unit = (a: number): V => { const e: V = [0, 0, 0]; e[a] = 1; return e; };
const coord = (l: number): V => [l & 7, (l >> 3) & 7, l >> 6];

// ---- the frozen state (cells; velocities m/s)
// Steep, so the C vertices at y = 16 (tile 4: within the spread radius of neither the blobs nor the non-resident
// page above, which saturates tiles 5..7) clear deep air's 6 + 3w = 18 about half the time.
const phiAt = (v: V) => 5 * (v[1] - 12.4) + 0.25 * Math.sin(0.3 * v[0] + 0.2 * v[2]);
/** US on the lower face of cell c along a: two moving blobs, zero elsewhere. */
const usAt = (c: V, a: number) => {
  if (c[0] < 24 && c[1] < 12 && c[2] < 24) return 4 * hash(c[0], c[1], c[2], a) - 2;
  if (c[0] >= 40 && c[0] < 56 && c[1] >= 20 && c[1] < 28 && c[2] >= 8 && c[2] < 24) return hash(c[0], c[1], c[2], 10 + a) - 0.5;
  return 0;
};
/** U and the release bit at a wall face's storage s (s[a] = 0 or N) along a, or undefined off the walls. */
const wallAt = (s: V, a: number): { u: number; released: boolean } | undefined => {
  const [x, y, z] = s;
  if (a === 1 && y === 0) {
    if (x < 40 && z < 40) return { u: 1.0, released: hash(x >> 2, z >> 2, 7) < 0.75 };
    if (x >= 40) return { u: -1.0, released: true };
    return { u: 1e-5, released: true };
  }
  if (a === 0 && x === 0) return y >= 16 && y < 32 && (z < 16 || z >= 40) ? { u: 25, released: true } : { u: 0, released: false };
  if (a === 0 && x === N) return y < 16 ? { u: -2.5, released: hash(y >> 2, z >> 2, 9) < 0.8 } : { u: 0, released: false };
  return undefined;
};
// h tiles: the floor corner, a patch beside the x = 0 wall across the surface, and a patch in the air whose F bricks
// the fixture retires (h tiles with no F slot: the producer's absent-brick reduction).
const want = (T: V) => (T[0] <= 5 && T[1] <= 2 && T[2] <= 5) || (T[0] <= 1 && T[1] >= 5 && T[1] <= 6 && T[2] >= 2 && T[2] <= 5)
  || (T[0] >= 8 && T[0] <= 11 && T[1] >= 6 && T[1] <= 7 && T[2] >= 8 && T[2] <= 11);
/** The F bricks the fixture retires (their tiles h-owned): bx, bz in 4..5, by 3. */
const retired = (b: V) => b[0] >= 4 && b[0] <= 5 && b[1] === 3 && b[2] >= 4 && b[2] <= 5;

// ---- the CPU world: the GPU's buffers, read back, and the ABI's readers over them
class World {
  constructor(readonly L: BrickLayout, readonly dir: Uint32Array, readonly vel: Float32Array, readonly phi: Uint32Array, readonly set: Uint32Array) {}
  inLattice = (c: V) => c.every((x, a) => x >= 0 && x < NL[a]!);
  pageWord(P: V) { return this.dir[this.L.dir.page + this.L.pageIndex(P[0], P[1], P[2])]!; }
  cslot(P: V) { if (P.some(x => x < 0)) return NONE; const w = this.pageWord(P); return w < SEAM ? w : NONE; }
  tileIndex = (T: V) => (T[0] & 7) | ((T[1] & 7) << 3) | ((T[2] & 7) << 6);
  local = (c: V) => (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 7) << 6);
  tileInLattice = (T: V) => T.every((x, a) => x >= 0 && 4 * x < NL[a]!);
  tileOwnerH(T: V) {
    if (!this.tileInLattice(T)) return false;
    const cs = this.cslot(shr(T, 3)); if (cs === NONE) return false;
    const i = this.tileIndex(T); return ((this.dir[this.L.dir.cmask + CMASK_WORDS * cs + (i >> 5)]! >>> (i & 31)) & 1) === 1;
  }
  cellOwnerH = (c: V) => this.tileOwnerH(shr(c, 2));
  fslot(c: V) {
    const cs = this.cslot(shr(c, 5)); if (cs === NONE) return NONE;
    const b = shr(c, 3), w = this.dir[this.L.dir.child + 64 * cs + ((b[0] & 3) | ((b[1] & 3) << 2) | ((b[2] & 3) << 4))]!;
    return w < SEAM ? w : NONE;
  }
  fFace(ch: string, c: V, a: number) { const fs = this.fslot(c); return fs === NONE ? 0 : this.vel[this.L.base("vel", "F", `${ch}${a}`) + BRICK_STRIDE * fs + this.local(c)]!; }
  cFace(ch: string, T: V, a: number) {
    if (T.some((x, i) => x < 0 || 4 * x > NL[i]!)) return 0;
    const cs = this.cslot(shr(T, 3)); return cs === NONE ? 0 : this.vel[this.L.base("vel", "C", `${ch}${a}`) + BRICK_STRIDE * cs + this.tileIndex(T)]!;
  }
  facePlane(ch: string, c: V, a: number) {
    if (!c.every((x, i) => x >= 0 && x < NL[i]! + (i === a ? 1 : 0))) return 0;
    if (this.cellOwnerH(c) || this.cellOwnerH(add(c, mul(unit(a), -1)))) return this.fFace(ch, c, a);
    return this.cFace(ch, shr(c, 2), a);
  }
  present = (c: V) => this.inLattice(c) && this.cslot(shr(c, 5)) !== NONE;
  /** umPageKind in the box: 0 stored, 1 air, 2 solid. */
  pageKind(Q: V) {
    if (Q.some((x, a) => x < 0 || 32 * x >= NL[a]!)) return 2;
    const w = this.pageWord(Q); if (w < SEAM) return 0;
    assert.equal(w, NONE, "no BULK16 or CLAIM page in the fixture");
    return 1;
  }
}

// ---- the producers' oracle: umReach, localSpeed, the spread and wallReach
const reachOf = (travel: number) => { const t = Math.ceil(travel * 1.00001 / 4) + 2; return t < 31 ? t : 31; };
const pack = (r: number[]) => (r[0]! | (r[1]! << 5) | (r[2]! << 10) | (r[3]! << 15) | (r[4]! << 20) | (r[5]! << 25)) >>> 0;
const fields = (w: number) => [w & 31, (w >>> 5) & 31, (w >>> 10) & 31, (w >>> 15) & 31, (w >>> 20) & 31, (w >>> 25) & 31];
function reachOracle(W: World, slots: number[]) {
  const scale = DT / HM, local = new Map<number, number>(); let fastest = 0;
  const tally: Record<string, number> = {};
  const bump = (k: string) => { tally[k] = (tally[k] ?? 0) + 1; };
  const take = (faces: number[][]) => {
    const r = [0, 0, 0, 0, 0, 0];
    for (let a = 0; a < 3; a++) {
      const lo = Math.min(...faces[a]!), hi = Math.max(...faces[a]!);
      r[a] = reachOf(Math.max(0, -lo) * scale); r[3 + a] = reachOf(Math.max(0, hi) * scale);
      fastest = Math.max(fastest, Math.abs(lo), Math.abs(hi));
    }
    return r;
  };
  for (const cs of slots) {
    const O = mul(brickKeyCoord(W.dir[W.L.dir.crec + 32 * cs + REC.KEY]!) as V, 8);
    for (let j = 0; j < 512; j++) {
      const T = add(O, coord(j)), at = 512 * cs + j;
      if (!W.tileInLattice(T)) { local.set(at, 0); continue; }
      const r = [0, 0, 0, 0, 0, 0];
      if (W.tileOwnerH(T)) {
        bump(W.fslot(mul(T, 4)) === NONE ? "h tile, no F slot" : "h tile");
        for (let m = 0; m < 64; m++) {
          const c = add(mul(T, 4), [m & 3, (m >> 2) & 3, m >> 4]);
          const faces = [0, 1, 2].map(a => {
            const f = [W.facePlane("US", add(c, unit(a)), a)];
            if ((c[a]! & 31) === 0 && !W.present(add(c, mul(unit(a), -1)))) f.push(W.facePlane("US", c, a));
            return f;
          });
          take(faces).forEach((x, k) => { r[k] = Math.max(r[k]!, x); });
        }
      } else {
        bump("4h tile");
        const faces = [0, 1, 2].map(a => {
          const u = (a + 1) % 3, v = (a + 2) % 3, up = mul(add(T, unit(a)), 4), f: number[] = [];
          if (W.cellOwnerH(up)) for (let i = 0; i < 16; i++) { const c: V = [...up]; c[u] += i & 3; c[v] += i >> 2; f.push(W.fFace("US", c, a)); }
          else f.push(W.cFace("US", add(T, unit(a)), a));
          if (((4 * T[a]!) & 31) === 0 && !W.present(add(mul(T, 4), mul(unit(a), -1)))) f.push(W.cFace("US", T, a));
          return f;
        });
        take(faces).forEach((x, k) => { r[k] = Math.max(r[k]!, x); });
      }
      local.set(at, pack(r));
    }
  }
  // umCertificateRadius, then spread0..2 (the plan's speed is an f32 maximum: the same faces' f32 values).
  const R = Math.min(Math.ceil(fastest * DT / HM * 1.00001 / 4) + 2, 1e8);
  const spread = (src: Map<number, number>, axis: number) => {
    const out = new Map<number, number>();
    for (const cs of slots) {
      const O = mul(brickKeyCoord(W.dir[W.L.dir.crec + 32 * cs + REC.KEY]!) as V, 8);
      for (let j = 0; j < 512; j++) {
        const p = add(O, coord(j)), at = 512 * cs + j;
        if (!W.tileInLattice(p)) { out.set(at, 0); continue; }
        const r = [0, 0, 0, 0, 0, 0];
        for (let q = Math.max(p[axis]! - R, 0); q <= Math.min(p[axis]! + R, NL[axis]! / 4 - 1); q++) {
          const t: V = [...p]; t[axis] = q;
          const qs = W.cslot(shr(t, 3));
          if (qs === NONE) { r.fill(31); break; }
          fields(src.get(512 * qs + W.tileIndex(t))!).forEach((x, k) => { r[k] = Math.max(r[k]!, x); });
        }
        out.set(at, pack(r));
      }
    }
    return out;
  };
  const x = spread(local, 0), y = spread(x, 1), z = spread(y, 2);
  return { local, x, y, z, R, fastest, tally };
}
/** wallReach: each wall face tile's released dt·away / h (cells), and the plane maxima. */
function wallOracle(W: World, slots: number[]) {
  const tiles = new Map<number, number>(), plane = [0, 0, 0, 0, 0, 0], tally: Record<string, number> = {};
  const bump = (k: string) => { tally[k] = (tally[k] ?? 0) + 1; };
  for (const cs of slots) {
    const P = brickKeyCoord(W.dir[W.L.dir.crec + 32 * cs + REC.KEY]!) as V;
    if (W.pageKind(P) === 2) continue;
    for (let i = 0; i < 384; i++) {
      const f = i >> 6, lane = i & 63, axis = f >> 1, upper = (f & 1) === 1, inward = upper ? -1 : 1;
      const across: V = [...P]; across[axis] += upper ? 1 : -1;
      if (W.pageKind(across) !== 2) continue;
      const tile = mul(P, 8); tile[axis] += upper ? 7 : 0; tile[(axis + 1) % 3] += lane & 7; tile[(axis + 2) % 3] += lane >> 3;
      const width = W.tileOwnerH(tile) ? 1 : 4; let bits = 0;
      for (let j = 0; j < 16; j += width * width) {
        const cell = mul(tile, 4); cell[axis] += upper ? 3 : 0; cell[(axis + 1) % 3] += j & 3; cell[(axis + 2) % 3] += j >> 2;
        const o = wallOwner(W, cell, axis, upper), gap = DT * inward * o.speed / HM;
        if (o.released && gap > 1e-4 * o.width) { bits = Math.max(bits, gap); bump(width === 1 ? "released h" : "released 4h"); }
        else bump(o.released ? "released, not away" : "unreleased");
      }
      tiles.set(384 * cs + i, bits);
      plane[f] = Math.max(plane[f]!, bits);
    }
  }
  return { tiles, plane, tally };
}
function wallOwner(W: World, q: V, a: number, upper: boolean) {
  const width = W.cellOwnerH(q) ? 1 : 4, s: V = [...q]; if (upper) s[a] += 1;
  let speed = 0, flags = 0;
  if (width === 1) {
    const fs = W.fslot(s);
    if (fs !== NONE) { speed = W.vel[W.L.base("vel", "F", `U${a}`) + BRICK_STRIDE * fs + W.local(s)]!; flags = W.phi[W.L.base("phi", "F", "FLAGS") + BRICK_STRIDE * fs + W.local(s)]!; }
  } else {
    const T = shr(s, 2), cs = W.cslot(shr(T, 3));
    if (cs !== NONE) { speed = W.vel[W.L.base("vel", "C", `U${a}`) + BRICK_STRIDE * cs + W.tileIndex(T)]!; flags = W.phi[W.L.base("phi", "C", "FLAGS") + BRICK_STRIDE * cs + W.tileIndex(T)]!; }
  }
  return { width, speed, released: ((flags >>> (8 + a)) & 1) === 1 };
}

// ---- the readers' oracle (umWallAhead, umReleasedWalls, umReleasedMayChange, umDeepAirKept), in cells
interface Words { plane: number[]; wallTile: (i: number) => number; reach: (T: V) => number }
function wallAhead(W: World, p: V, k: number, reach: number, bound: number) {
  const axis = k >> 1, u = (axis + 1) % 3, v = (axis + 2) % 3, upper = (k & 1) === 1, inward = upper ? -1 : 1;
  const ahead = { plane: 0, blocks: [NONE, NONE, NONE, NONE] };
  const mask = ((p[u]! & 31) === 0 ? 1 : 0) | ((p[v]! & 31) === 0 ? 2 : 0);
  let plane = upper ? (p[axis]! + 31) & -32 : p[axis]! & -32, first = (p[axis]! & 31) === 0;
  for (let n = 0; n < 64; n++) {
    if (reach - inward * (p[axis]! - plane) + 1e-5 < bound) return ahead;
    const blocks = [NONE, NONE, NONE, NONE]; let open = false, wall = false;
    for (let c = 0; c < 4; c++) {
      if ((c & mask) !== c) { blocks[c] = blocks[c & mask]!; continue; }
      const q: V = [...p]; q[axis] = plane - (upper ? 1 : 0); q[u] -= 1 - (c & 1); q[v] -= 1 - (c >> 1);
      const Q = shr(q, 5), b = W.pageKind(Q), A: V = [...Q]; A[axis] += upper ? 1 : -1; const across = W.pageKind(A);
      if (across === 0) open = open || first || b === 0;
      else if (across !== 1 && b === 0) { blocks[c] = W.cslot(Q); wall = true; }
    }
    if (wall) return { plane, blocks };
    if (!open) return ahead;
    plane += upper ? 32 : -32; first = false;
  }
  return ahead;
}
function releasedWalls(W: World, w: Words, p: V, value: number, tally?: Record<string, number>) {
  let result = value;
  for (let k = 0; k < 6; k++) {
    const axis = k >> 1, upper = (k & 1) === 1, inward = upper ? -1 : 1, reach = w.plane[k]!;
    if (reach <= 0) continue;
    const wall = wallAhead(W, p, k, reach, result), depth = inward * (p[axis]! - wall.plane);
    let whole = 1e30, probes = 0;
    for (let c = 0; c < 4; c++) {
      if (wall.blocks[c] === NONE) continue;
      const cell: V = [...p]; cell[axis] = wall.plane - (upper ? 1 : 0); cell[(axis + 1) % 3] -= 1 - (c & 1); cell[(axis + 2) % 3] -= 1 - (c >> 1);
      const o = wallOwner(W, cell, axis, upper), g = DT * inward * o.speed / HM;
      const resolved = o.width === 1 || value > -0.5 * o.width, gap = o.released && g > 1e-4 * o.width ? g : 0;
      if (resolved && gap > 0 && gap - depth > result) { result = gap - depth; if (tally) tally[o.width === 1 ? "raised, h column" : "raised, 4h column"] = 1 + (tally[o.width === 1 ? "raised, h column" : "raised, 4h column"] ?? 0); }
      whole = Math.min(whole, gap); probes++;
    }
    if (probes > 0 && whole > 0 && whole - depth > result) { result = whole - depth; if (tally) tally["raised, whole"] = 1 + (tally["raised, whole"] ?? 0); }
  }
  return result;
}
function mayChange(W: World, w: Words, p: V, value: number) {
  for (let k = 0; k < 6; k++) {
    const reach = w.plane[k]!; if (!(reach > 0)) continue;
    const axis = k >> 1, inward = (k & 1) === 1 ? -1 : 1, wall = wallAhead(W, p, k, reach, value), depth = inward * (p[axis]! - wall.plane);
    let local = 0;
    for (let c = 0; c < 4; c++) {
      if (wall.blocks[c] === NONE) continue;
      const cell: V = [...p]; cell[(axis + 1) % 3] -= 1 - (c & 1); cell[(axis + 2) % 3] -= 1 - (c >> 1);
      const t = shr(cell, 2), uv = (t[(axis + 1) % 3]! & 7) + 8 * (t[(axis + 2) % 3]! & 7);
      local = Math.max(local, w.wallTile(384 * wall.blocks[c]! + 64 * k + uv));
    }
    if (local > 0 && !(local - depth + 1e-5 < value)) return true;
  }
  return false;
}
function deepAirKept(W: World, w: Words, v: V, width: number, value: number, tally?: Record<string, number>) {
  const bump = (k: string) => { if (tally) tally[k] = (tally[k] ?? 0) + 1; };
  if (!(value >= 6 + 3 * width)) { bump("deep: under 6 + 3w"); return -1; }
  const t = shr(v.map((x, a) => Math.min(Math.max(x, 0), NL[a]! - 1)) as V, 2);
  if (Math.max(...fields(w.reach(t))) > 2) { bump("deep: moving tile"); return -1; }
  if (mayChange(W, w, v, value)) { bump("deep: a released wall may change it"); return -1; }
  bump("deep: kept");
  return value;
}

// ---- a probe of the readers: each F vertex at width 1 and each C vertex at width 4, its φ the value. The deep-air
// result goes to PHI1 and the released walls' to UP0 (channels the readers do not read).
const PROBE_BINDINGS: readonly BrickBinding[] = [CFG, { name: "set", kind: "rw", type: "array<atomic<u32>>" }, { name: "dir", kind: "ro", type: "array<u32>" },
  { name: "vel", kind: "rw", type: "array<f32>" }, { name: "phi", kind: "rw", type: "array<u32>" }];
const probeKernel = (L: BrickLayout) => ({
  label: "nrProbe", entry: "nrProbe", bindings: PROBE_BINDINGS,
  code: [brickAbiWGSL(L, { set: { name: "nbSet", atomic: true }, dir: { name: "dir", atomic: false } }), declareBindings(PROBE_BINDINGS), levelSetReleaseWGSL(), /* wgsl */`
@compute @workgroup_size(256) fn nrProbe(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 let g=nbGroup(wid);let nF=nbListCount(NB_LIST_F_ALL);
 if(g<nF){
  let fs=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(fs);
  for(var r=0u;r<2u;r++){
   let l=lid+256u*r;let v=o+nbLocalCoord(l);if(any(v>NB_LATTICE)){continue;}
   let at=NB_STRIDE*fs+l;let x=bitcast<f32>(phi[NB_PHI_F_PHI+at]);
   phi[NB_PHI_F_PHI1+at]=bitcast<u32>(nbrDeepAirKept(v,1u,x));vel[NB_VEL_F_UP0+at]=nbrReleasedWalls(vec3f(v),x);
  }
  return;
 }
 if(g-nF>=nbListCount(NB_LIST_C_ALL)){return;}
 let cs=nbListSlot(NB_LIST_C_ALL,g-nF);let o=nbCOrigin(cs);
 for(var r=0u;r<2u;r++){
  let l=lid+256u*r;let v=o+4*nbLocalCoord(l);if(any(v>NB_LATTICE)){continue;}
  let at=NB_STRIDE*cs+l;let x=bitcast<f32>(phi[NB_PHI_C_PHI+at]);
  phi[NB_PHI_C_PHI1+at]=bitcast<u32>(nbrDeepAirKept(v,4u,x));vel[NB_VEL_C_UP0+at]=nbrReleasedWalls(vec3f(v),x);
 }
}`].join("\n"),
});

test("levelset-release: the readers are self-contained and the schedule places the producers", () => {
  const L = new BrickLayout({ lattice: NL, f: 600, c: 40, b: 0, k: 0, particles: 1024 });
  const ks = levelSetReleaseKernels(L);
  for (const [name, k] of Object.entries(ks)) {
    const fns = [...k.code.matchAll(/\bfn\s+(\w+)\s*\(/g)].map(m => m[1]!);
    assert.deepEqual(fns.filter((f, i) => fns.indexOf(f) !== i), [], `${name}: functions defined twice`);
  }
  const coarse = levelSetCoarseKernels(L).lcAdvect.code;
  for (const f of ["nbrDeepAirKept", "nbrReleasedWalls", "nbrReleasedMayChange"]) assert.ok(coarse.includes(`fn ${f}(`), `lcAdvect includes ${f}`);
  assert.ok(/nbrDeepAirKept\(v,4u,c\)/.test(coarse) && /nbrReleasedWalls\(p,lcWallContact\(/.test(coarse), "lcAdvect calls the readers at the old sites");
  const s = levelSetReleaseSchedule(L, ks);
  assert.deepEqual(s.part1.map(p => [p.kernel.label, p.z ?? 0]), [["nrReach", 0], ["nrSpread", 0], ["nrSpread", 1], ["nrSpread", 2]]);
  assert.deepEqual(s.part2.map(p => p.kernel.label), ["nrWallReach"]);
  // The set regions are additive (after AIR_GATE) and the header words are E's reserved range.
  assert.ok(L.set.reach0 > L.set.airGate && L.set.reach1 > L.set.reach0 && L.set.reachStep > L.set.reach1 && L.set.wallTile > L.set.reachStep && L.set.words >= L.set.wallTile + 384 * 40);
  assert.ok(H.REL_SPEED0 >= 232 && H.WALL_REACH1 + 6 <= 256);
});

test("levelset-release (GPU): reach words, wall reach, the readers and lcAdvect's kept deep air and released walls match the CPU oracle", GPU, async () => {
  const { withUniformDevice } = await import("../../../tests/helpers/uniform-geometric");
  await withUniformDevice("levelset release", async device => {
    const stages = (c: Parameters<typeof bringUpStages>[0]) => ({ ...bringUpStages(c), tiered: true, buffers: { fcell: fluidLayout(c.layout, { tiered: true }).fcell.words } });
    const e = await BrickEngine.create(device, { lattice: NL, h_m: HM, dt_s: DT, gravity_mps2: [0, -9.81, 0],
      capacities: { f: 600, c: 40, b: 0, k: 0, particles: 1024 }, liquid: [{ lo: [-1, -1, -1], hi: [N + 1, 20.3, N + 1] }], solver: L => bringUpCG(L), stages });
    const L = e.layout, F = fluidLayout(L, { tiered: true });
    {
      const boot = await snapshot(e);
      for (const cs of boot.pages.values()) {
        const P = brickKeyCoord(boot.dir[L.dir.crec + 32 * cs + REC.KEY]!) as V, w = new Uint32Array(CMASK_WORDS);
        for (let i = 0; i < 512; i++) { const T = add(mul(P, 8), coord(i)); if (T.every(x => 4 * x < N) && want(T)) w[i >> 5] = (w[i >> 5]! | (1 << (i & 31))) >>> 0; }
        e.write("set", L.set.want + CMASK_WORDS * cs, w);
      }
      await e.run([{ kernel: brickTransferKernels(L).setCommitMask, dispatch: { indirect: 12 * LIST.C_ALL } }]);
      // Retire the air patch's F bricks as setLists retires a leaving brick (key NONE, out of its lists), with the
      // page's child word and the neighbours' record words NONE, then setArgs's indirect arguments.
      const sb = await snapshot(e), dir = await e.read("dir"), set = await e.read("set"), gone = new Set<number>();
      for (const [bk, fs] of sb.bricks) if (retired(brickKeyCoord(bk) as V)) gone.add(fs);
      assert.equal(gone.size, 4, "the retired bricks are resident at boot");
      for (const fs of gone) dir[L.dir.frec + 32 * fs + REC.KEY] = NONE;
      for (const cs of sb.pages.values()) for (let j = 0; j < 64; j++) if (gone.has(dir[L.dir.child + 64 * cs + j]!)) dir[L.dir.child + 64 * cs + j] = NONE;
      for (const fs of sb.bricks.values()) for (let j = 0; j < 27; j++) if (gone.has(dir[L.dir.frec + 32 * fs + j]!)) dir[L.dir.frec + 32 * fs + j] = NONE;
      for (let k = LIST.F_ALL; k <= LIST.F_CHANGED; k++) {
        const kept = sb.list(k).filter(x => !gone.has(x));
        set.set(kept, L.set.lists[k]!); set[H.LIST_COUNT + k] = kept.length;
      }
      e.write("dir", 0, new Uint32Array(dir)); e.write("set", 0, new Uint32Array(set));
      await e.run([{ kernel: brickSetKernels(L, { tiered: true }).setArgs, dispatch: { direct: [1, 1, 1] } }]);
    }
    const s0 = await snapshot(e);
    assert.equal(s0.header[H.FAIL], 0);
    const slots = s0.list(LIST.C_ALL);
    assert.equal(s0.bricks.size, 320, "four F bricks retired");
    // ---- the frozen state: φ on every copy, US and U, the release bits, the far-air gate set everywhere (so far air
    // never keeps a vertex and deep air is reached).
    const phiWords = new Uint32Array(await e.read("phi")), phiF = new Float32Array(phiWords.buffer);
    const vel = new Float32Array((await e.read("vel")).buffer);
    vel.fill(0);
    const flagsF = L.base("phi", "F", "FLAGS"), flagsC = L.base("phi", "C", "FLAGS");
    let releasedBits = 0;
    const face = (kind: "F" | "C", slot: number, l: number, c: V) => {
      for (let a = 0; a < 3; a++) {
        vel[L.base("vel", kind, `US${a}`) + BRICK_STRIDE * slot + l] = usAt(c, a);
        const wall = c.every((x, i) => i === a ? x === 0 || x === N : x >= 0 && x < N) ? wallAt(c, a) : undefined;
        if (!wall) continue;
        vel[L.base("vel", kind, `U${a}`) + BRICK_STRIDE * slot + l] = wall.u;
        const fi = (kind === "F" ? flagsF : flagsC) + BRICK_STRIDE * slot + l;
        if (wall.released) { phiWords[fi] = (phiWords[fi]! | (1 << (8 + a))) >>> 0; releasedBits++; }
      }
    };
    for (const [bk, fs] of s0.bricks) {
      const o = mul(brickKeyCoord(bk) as V, 8);
      for (let l = 0; l < 512; l++) {
        const v = add(o, coord(l));
        phiF[L.base("phi", "F", "PHI") + BRICK_STRIDE * fs + l] = phiAt(v); phiWords[L.base("phi", "F", "PHI1") + BRICK_STRIDE * fs + l] = POISON;
        face("F", fs, l, v);
      }
    }
    for (const [pk, cs] of s0.pages) {
      const o = mul(brickKeyCoord(pk) as V, 32);
      for (let l = 0; l < 512; l++) {
        const v = add(o, mul(coord(l), 4));
        phiF[L.base("phi", "C", "PHI") + BRICK_STRIDE * cs + l] = phiAt(v); phiWords[L.base("phi", "C", "PHI1") + BRICK_STRIDE * cs + l] = POISON;
        face("C", cs, l, v);
      }
    }
    e.write("phi", 0, new Uint32Array(phiWords)); e.write("vel", 0, new Float32Array(vel));
    for (const cs of slots) e.write("set", L.set.airGate + CMASK_WORDS * cs, new Uint32Array(CMASK_WORDS).fill(0xffffffff));
    await e.run([{ kernel: velocityKernels(L, F).uCache!, dispatch: { indirect: ARGS.FC_G1 } }]);
    const fcell0 = await e.read("fcell"), dir0 = await e.read("dir"), phi0 = await e.read("phi"), vel0 = await e.read("vel");
    const W = new World(L, dir0, new Float32Array(vel0.buffer), phi0, await e.read("set"));
    console.log(`levelset-release: ${s0.bricks.size} F, ${s0.pages.size} C pages, ${releasedBits} release bits`);

    // ---- lcAdvect with no words (REACH_STEP clear: every tile saturated; no plane reach): nothing fires.
    // lcAdvect runs eight workgroups a C_ALL entry (its X8 arguments), as the step places it. The plain C_ALL
    // arguments (one workgroup an entry) advected only the first eighth of the list, and the list's atomic order
    // chose a different eighth each run.
    const kernels = levelSetCoarseKernels(L, F), listC = { indirect: 12 * LIST.C_ALL + ARGS[LC_ADVECT_GROUPS] };
    await e.run([{ kernel: kernels.lcAdvect, dispatch: listC }]);
    const before = new Float32Array((await e.read("phi")).buffer).slice();
    e.write("phi", 0, new Uint32Array(phi0));

    // ---- the producers.
    const setBefore = await e.read("set");
    const sched = levelSetReleaseSchedule(L);
    await e.run([...sched.part1, ...sched.part2]);
    const set = await e.read("set");
    assert.equal(set[H.FAIL], 0, `producers: FAIL ${set[H.FAIL]} info ${set[H.FAIL_INFO]}`);
    const R = reachOracle(W, slots), Wl = wallOracle(W, slots);
    console.log(`reach: radius ${R.R} tiles, plan speed ${R.fastest.toFixed(3)} m/s; ${JSON.stringify(R.tally)}; walls ${JSON.stringify(Wl.tally)}`);
    const mine = new Set<number>();
    const expectWords = (name: string, base: number, want: Map<number, number>) => {
      let bad = 0; const report: string[] = [];
      for (const [i, x] of want) { mine.add(base + i); if (set[base + i] !== x) { bad++; if (report.length < 6) report.push(`${name}[${i}]: gpu ${set[base + i]!.toString(16)} oracle ${x.toString(16)}`); } }
      assert.equal(bad, 0, `${name}: ${bad} words off the oracle\n${report.join("\n")}`);
    };
    expectWords("REACH1 (final, z)", L.set.reach1, R.z);
    expectWords("REACH0 (y)", L.set.reach0, R.y);
    const ringStep = set[L.set.ring + RING_WORDS * 0 + RING.STEP]!;
    for (const cs of slots) { mine.add(L.set.reachStep + cs); assert.equal(set[L.set.reachStep + cs], ringStep + 1, "REACH_STEP: the ring step + 1"); }
    assert.equal(bitsf(set[H.REL_SPEED0]!), Math.fround(R.fastest), "REL_SPEED0: the plan speed");
    let tileBad = 0;
    for (const [i, x] of Wl.tiles) { mine.add(L.set.wallTile + i); const g = bitsf(set[L.set.wallTile + i]!); if ((x === 0) !== (g === 0) || Math.abs(g - x) > 1e-6 * x) tileBad++; }
    assert.equal(tileBad, 0, "WALL_TILE against wallReach's oracle");
    for (let f = 0; f < 6; f++) { const g = bitsf(set[H.WALL_REACH0 + f]!), x = Wl.plane[f]!; assert.ok((x === 0) === (g === 0) && Math.abs(g - x) <= 1e-6 * x, `WALL_REACH0[${f}]: gpu ${g} oracle ${x}`); }
    for (let i = H.REL_SPEED0; i < H.WALL_REACH1 + 6; i++) mine.add(i);
    for (let f = 0; f < 6; f++) assert.equal(set[H.WALL_REACH1 + f], 0, "the other parity's plane words are zeroed");
    assert.equal(set[H.REL_SPEED1], 0);
    // Every other set word bitwise unchanged, and the other buffers untouched (REACH0's local half is the x pass's
    // input, overwritten by the y pass: checked through the final words; REACH1's x half likewise).
    let stray = 0; for (let i = 0; i < set.length; i++) if (!mine.has(i) && set[i] !== setBefore[i]) stray++;
    assert.equal(stray, 0, "producers: no set word outside their regions moved");
    const same = (a: Uint32Array, b: Uint32Array) => a.length === b.length && a.every((x, i) => x === b[i]);
    assert.ok(same(await e.read("vel"), vel0) && same(await e.read("phi"), phi0) && same(await e.read("dir"), dir0) && same(await e.read("fcell"), fcell0), "vel, phi, dir and fcell untouched");
    assert.ok((R.tally["h tile"] ?? 0) > 0 && (R.tally["4h tile"] ?? 0) > 0 && (R.tally["h tile, no F slot"] ?? 0) > 0, "every owner kind");
    for (const k of ["released h", "released 4h", "released, not away", "unreleased"]) assert.ok((Wl.tally[k] ?? 0) > 0, `wall path ${k}`);
    const words: Words = {
      plane: Array.from({ length: 6 }, (_, f) => bitsf(set[H.WALL_REACH0 + f]!)), wallTile: i => bitsf(set[L.set.wallTile + i]!),
      reach: T => { const cs = W.cslot(shr(T, 3)); return cs === NONE || set[L.set.reachStep + cs] !== ringStep + 1 ? 0x3fffffff : set[L.set.reach1 + 512 * cs + W.tileIndex(T)]! & 0x3fffffff; },
    };

    // ---- the readers on the GPU: every F vertex (width 1) and C vertex (width 4).
    await e.run([{ kernel: probeKernel(L), dispatch: { indirect: ARGS.FC_G1 } }]);
    {
      const ph = new Float32Array((await e.read("phi")).buffer), ve = new Float32Array((await e.read("vel")).buffer), tally: Record<string, number> = {};
      let bad = 0; const report: string[] = [];
      const check = (kind: "F" | "C", slot: number, l: number, v: V, width: number) => {
        const at = BRICK_STRIDE * slot + l, x = ph[L.base("phi", kind, "PHI") + at]!;
        const deep = deepAirKept(W, words, v, width, x, tally), gd = ph[L.base("phi", kind, "PHI1") + at]!;
        const rel = releasedWalls(W, words, v, x, tally), gr = ve[L.base("vel", kind, "UP0") + at]!;
        if (gd !== Math.fround(deep) || !(Math.abs(gr - rel) <= 1e-5 * Math.max(1, Math.abs(rel)))) { bad++; if (report.length < 8) report.push(`${kind} v ${v} x ${x}: deep gpu ${gd} oracle ${deep}; released gpu ${gr} oracle ${rel}`); }
      };
      for (const [bk, fs] of s0.bricks) { const o = mul(brickKeyCoord(bk) as V, 8); for (let l = 0; l < 512; l++) { const v = add(o, coord(l)); if (v.every((x, a) => x <= NL[a]!)) check("F", fs, l, v, 1); } }
      for (const [pk, cs] of s0.pages) { const o = mul(brickKeyCoord(pk) as V, 32); for (let l = 0; l < 512; l++) { const v = add(o, mul(coord(l), 4)); if (v.every((x, a) => x <= NL[a]!)) check("C", cs, l, v, 4); } }
      console.log(`readers: ${JSON.stringify(tally)}`);
      assert.equal(bad, 0, `readers: ${bad} vertices off the oracle\n${report.join("\n")}`);
      for (const k of ["deep: kept", "deep: under 6 + 3w", "deep: moving tile", "deep: a released wall may change it", "raised, h column", "raised, whole"]) assert.ok((tally[k] ?? 0) > 0, `reader path ${k} covered`);
    }
    e.write("phi", 0, new Uint32Array(phi0)); e.write("vel", 0, new Float32Array(new Uint32Array(vel0).buffer));

    // ---- lcAdvect with the words: kept deep air, raised, or bitwise the no-words value.
    await e.run([{ kernel: kernels.lcAdvect, dispatch: listC }]);
    const after = new Float32Array((await e.read("phi")).buffer), afterU = new Uint32Array(after.buffer);
    const setAfter = await e.read("set");
    assert.equal(setAfter[H.FAIL], 0, `lcAdvect: FAIL ${setAfter[H.FAIL]} info ${setAfter[H.FAIL_INFO]}`);
    const tally: Record<string, number> = {};
    const bump = (k: string) => { tally[k] = (tally[k] ?? 0) + 1; };
    let bad = 0; const report: string[] = [];
    const beforeU = new Uint32Array(before.buffer), phi1C = L.base("phi", "C", "PHI1"), phiC = L.base("phi", "C", "PHI");
    for (const [pk, cs] of s0.pages) {
      const o = mul(brickKeyCoord(pk) as V, 32);
      for (let l = 0; l < 512; l++) {
        const i = phi1C + BRICK_STRIDE * cs + l, v = add(o, mul(coord(l), 4));
        if (beforeU[i] === POISON) { if (afterU[i] !== POISON) { bad++; report.push(`v ${v}: written now, not before`); } continue; }
        const c = bitsf(phi0[phiC + BRICK_STRIDE * cs + l]!), wallV = v.some((x, a) => x === 0 || x === NL[a]);
        const deep = wallV ? -1 : deepAirKept(W, words, v, 4, c);
        if (wallV && c >= 18 && Math.max(...fields(words.reach(shr(v.map((x, a) => Math.min(x, NL[a]! - 1)) as V, 2)))) <= 2 && !mayChange(W, words, v, c)) bump("wall vertex: deep air not applied");
        if (deep >= 0) { bump("kept deep air"); if (afterU[i] !== fbits(c)) { bad++; if (report.length < 8) report.push(`v ${v}: kept deep air ${c}, gpu ${after[i]}`); } continue; }
        const a0 = before[i]!, x = releasedWalls(W, words, v, a0);
        if (x > a0 + 1e-6 * Math.max(1, Math.abs(a0))) {
          bump("raised");
          if (!(Math.abs(after[i]! - x) <= 1e-5 * Math.max(1, Math.abs(x)))) { bad++; if (report.length < 8) report.push(`v ${v}: raised ${a0} -> oracle ${x}, gpu ${after[i]}`); }
        } else {
          bump(c >= 18 ? "traced deep candidate, unchanged" : "unchanged");
          if (afterU[i] !== beforeU[i]) { bad++; if (report.length < 8) report.push(`v ${v}: unchanged ${a0}, gpu ${after[i]}`); }
        }
      }
    }
    console.log(`lcAdvect: ${JSON.stringify(tally)}`);
    assert.equal(bad, 0, `lcAdvect: ${bad} vertices off the oracle\n${report.join("\n")}`);
    for (const k of ["kept deep air", "raised", "unchanged", "traced deep candidate, unchanged", "wall vertex: deep air not applied"]) assert.ok((tally[k] ?? 0) > 0, `lcAdvect path ${k} covered`);
    // lcAdvect writes PHI1 only: the rest of the φ pool is the state's.
    const nowPhi = await e.read("phi"); let strayPhi = 0;
    for (let i = 0; i < nowPhi.length; i++) { const inC1 = i >= phi1C && i < phi1C + BRICK_STRIDE * L.cap.c; if (!inC1 && nowPhi[i] !== phi0[i]) strayPhi++; }
    assert.equal(strayPhi, 0, "lcAdvect writes C PHI1 only");
  });
});
