/**
 * The layout-neutral form of a frozen tier change (the old NB-FLIP remap's
 * pre and post state, tools/nbflip-remap-capture.ts), shared by the capture
 * tool and brick-transfer-dam.test.ts.
 *
 * A window of W^3 cells at the lattice's low corner. Per tile: the old and
 * new owner (1 = h, 4 = 4h). Per state (pre under the old owners, post under
 * the new): the value at every stored vertex and face, in the order
 * storedVertices / storedFaces enumerate them under that state's owners.
 * "Stored" is brick-abi.md §2.5 on the window as a lattice:
 * - a vertex with no 4h-owned incident tile is an h vertex (F); a tile corner
 *   with one is a C vertex; every other vertex hangs (not stored);
 * - a lower face with an h-owned cell on either side is an h face (F); a face
 *   between two 4h-owned tiles on a tile plane is the C face of the upper
 *   tile, enumerated once, at the tile's origin; walls (planes 0 and W) by
 *   their in-window cell.
 * The window's upper planes are not the old lattice's walls, so locations
 * there are enumerated but meaningless; the test leaves them out.
 */
export type V3 = [number, number, number];
export type Owner = (t: V3) => boolean;

/** Owners of a window: tile t is h when the byte is 1. */
export const ownerOf = (widths: Uint8Array, tiles: number): Owner => (t: V3) =>
  t.every(v => v >= 0 && v < tiles) && widths[t[0] + tiles * (t[1] + tiles * t[2])] === 1;

/** The 4h-owned tile holding vertex v's authority (nbVertexCoarse's order), or undefined. */
export function vertexCoarse(h: Owner, tiles: number, v: V3): V3 | undefined {
  const tA = v.map(x => x >> 2), inside = v.map((x, a) => x - 4 * tA[a]! !== 0);
  for (let k = 0; k < 8; k++) {
    const corner = [k & 1, (k >> 1) & 1, k >> 2];
    if (corner.some((c, a) => inside[a] && c === 0)) continue;
    const t = tA.map((x, a) => x + corner[a]! - 1) as V3;
    if (t.every(x => x >= 0 && x < tiles) && !h(t)) return t;
  }
  return undefined;
}

/** "F", "C" or undefined (hanging). */
export function vertexKind(h: Owner, tiles: number, p: V3): "F" | "C" | undefined {
  if (!vertexCoarse(h, tiles, p)) return "F";
  return p.every(x => (x & 3) === 0) ? "C" : undefined;
}

/** The lower face of cell c along a: "F", "C" (at the upper tile's origin only) or undefined. */
export function faceKind(h: Owner, W: number, c: V3, a: number): "F" | "C" | undefined {
  const inCell = (q: V3) => q.every(v => v >= 0 && v < W);
  const below: V3 = [...c]; below[a]!--;
  const hOf = (q: V3) => inCell(q) && h([q[0] >> 2, q[1] >> 2, q[2] >> 2]);
  if (hOf(c) || hOf(below)) return "F";
  return c.every(x => (x & 3) === 0) ? "C" : undefined;
}

/** Every vertex of the window, x fastest. */
export function* windowVertices(W: number): Generator<V3> {
  for (let z = 0; z <= W; z++) for (let y = 0; y <= W; y++) for (let x = 0; x <= W; x++) yield [x, y, z];
}
/** Every lower face of the window: axis, then cell (x fastest); planes 0..W along the axis. */
export function* windowFaces(W: number): Generator<[V3, number]> {
  for (let a = 0; a < 3; a++) {
    const hi = [W, W, W]; hi[a] = W + 1;
    for (let z = 0; z < hi[2]!; z++) for (let y = 0; y < hi[1]!; y++) for (let x = 0; x < hi[0]!; x++) yield [[x, y, z], a];
  }
}
export function storedVertices(h: Owner, W: number): { p: V3; kind: "F" | "C" }[] {
  const out: { p: V3; kind: "F" | "C" }[] = [];
  for (const p of windowVertices(W)) { const kind = vertexKind(h, W / 4, p); if (kind) out.push({ p, kind }); }
  return out;
}
export function storedFaces(h: Owner, W: number): { c: V3; a: number; kind: "F" | "C" }[] {
  const out: { c: V3; a: number; kind: "F" | "C" }[] = [];
  for (const [c, a] of windowFaces(W)) { const kind = faceKind(h, W, c, a); if (kind) out.push({ c, a, kind }); }
  return out;
}

export interface RemapFixtureMeta {
  readonly scene: string;
  /** The frame whose head remap was captured (1-based, as solver.advanceTo(step * dt)). */
  readonly step: number;
  readonly window: number;
  readonly lattice: readonly number[];
  /** Listed tiles in the whole lattice, and those in the window. */
  readonly listed: { readonly total: number; readonly refine: number; readonly coarsen: number; readonly neighbour: number; readonly window: number };
  /** Byte offsets and element counts of the sections of the .bin (little-endian). */
  readonly sections: Readonly<Record<string, { readonly offset: number; readonly count: number; readonly type: "u8" | "u32" | "f32" }>>;
  readonly notes: readonly string[];
}

/** Section names: owners, listing, then per state and field the stored values. */
export const SECTIONS = ["oldOwner", "newOwner", "listed", "prePhi", "preU", "preUS", "postPhi", "postU", "postUS"] as const;

export function encodeFixture(parts: Record<(typeof SECTIONS)[number], Uint8Array | Uint32Array | Float32Array>): { bin: Uint8Array; sections: RemapFixtureMeta["sections"] } {
  let at = 0; const sections: Record<string, { offset: number; count: number; type: "u8" | "u32" | "f32" }> = {};
  const chunks: Uint8Array[] = [];
  for (const name of SECTIONS) {
    const v = parts[name], type = v instanceof Uint8Array ? "u8" : v instanceof Uint32Array ? "u32" : "f32";
    const bytes = new Uint8Array(v.buffer, v.byteOffset, v.byteLength), pad = (4 - (bytes.length & 3)) & 3;
    sections[name] = { offset: at, count: v.length, type };
    chunks.push(bytes, new Uint8Array(pad)); at += bytes.length + pad;
  }
  const bin = new Uint8Array(at); let o = 0; for (const c of chunks) { bin.set(c, o); o += c.length; }
  return { bin, sections };
}

export function decodeFixture(bin: Uint8Array, meta: RemapFixtureMeta) {
  const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);
  const get = (name: (typeof SECTIONS)[number]) => {
    const s = meta.sections[name]!;
    return s.type === "u8" ? new Uint8Array(buf, s.offset, s.count) : s.type === "u32" ? new Uint32Array(buf, s.offset, s.count) : new Float32Array(buf, s.offset, s.count);
  };
  return {
    oldOwner: get("oldOwner") as Uint8Array, newOwner: get("newOwner") as Uint8Array,
    /** Pairs (tile index in the window, flags) of the old remap's worklist entries in the window. */
    listed: get("listed") as Uint32Array,
    pre: { phi: get("prePhi") as Float32Array, U: get("preU") as Float32Array, US: get("preUS") as Float32Array },
    post: { phi: get("postPhi") as Float32Array, U: get("postU") as Float32Array, US: get("postUS") as Float32Array },
  };
}
