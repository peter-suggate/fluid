/** Host emulation of the pressure root's cut-tile rules
 * (lib/methods/uniform/uniform-mixed-pressure-authority.ts): the side of a
 * wall a cut tile's phi is fit through (cutSide, cutLive, cutTile), an uncut h
 * tile's phi (pcHeldFit), and the h faces of a 4h face the liquid's rows hold
 * (umWetSide, umWetFace). Statement for statement the kernels' own, on the
 * same single-precision sums where a sign decides.
 *
 * A tile's open h cells are two words, bit x + 4y + 16z. A tile's vertices
 * are its 5^3, vertex v = x + 5y + 25z. */

export type OpenCells = readonly [number, number];

/** One tile as umWetFace reads it. */
export interface WetTile {
  /** The open h cells of the tile (umSolidOpenCells). */
  readonly open: OpenCells;
  /** Statically cut and held at h (umSolidCut). */
  readonly cut: boolean;
  /** Held at h (umSolidFineTile). */
  readonly fine: boolean;
  /** The owner is a row (umWetRow). */
  readonly row: boolean;
  /** The sides the owner's phi was fit through: the cut fit's second word. */
  readonly sides: number;
}

const bit = (open: OpenCells, i: number): boolean => ((open[i >>> 5]! >>> (i & 31)) & 1) !== 0;

/** One face-connected set of the open h cells, grown from the lowest. */
export function cutSide(open: OpenCells): [number, number] {
  let sx = 0, sy = 0;
  if (open[0] !== 0) sx = (open[0] & (~open[0] + 1)) >>> 0; else sy = (open[1] & (~open[1] + 1)) >>> 0;
  for (let k = 0; k < 64; k++) {
    const xx = (((sx << 1) & 0xeeeeeeee) | ((sx >>> 1) & 0x77777777)) >>> 0, xy = (((sy << 1) & 0xeeeeeeee) | ((sy >>> 1) & 0x77777777)) >>> 0;
    const yx = (((sx << 4) & 0xfff0fff0) | ((sx >>> 4) & 0x0fff0fff)) >>> 0, yy = (((sy << 4) & 0xfff0fff0) | ((sy >>> 4) & 0x0fff0fff)) >>> 0;
    const zx = ((sx << 16) | (sx >>> 16) | (sy << 16)) >>> 0, zy = ((sy << 16) | (sx >>> 16) | (sy >>> 16)) >>> 0;
    const gx = ((sx | xx | yx | zx) & open[0]) >>> 0, gy = ((sy | xy | yy | zy) & open[1]) >>> 0;
    if (gx === sx && gy === sy) break;
    sx = gx; sy = gy;
  }
  return [sx, sy];
}

/** Vertex v is live: one of the tile's h cells at it is open. */
export function cutLive(open: OpenCells, v: number): boolean {
  const g = [v % 5, Math.floor(v / 5) % 5, Math.floor(v / 25)];
  for (let k = 0; k < 8; k++) {
    const x = g[0]! - (k & 1), y = g[1]! - ((k >> 1) & 1), z = g[2]! - ((k >> 2) & 1);
    if (x >= 0 && y >= 0 && z >= 0 && x < 4 && y < 4 && z < 4 && bit(open, x + 4 * y + 16 * z)) return true;
  }
  return false;
}

/** The least-squares plane through the vertices `live` names, read at the
 * tile centre: centred normal equations in cell units, as cutTile's. */
export function planeAtCentre(live: (v: number) => boolean, vertex: (v: number) => number, hmin: number): number {
  const at = (v: number) => [v % 5 - 2, Math.floor(v / 5) % 5 - 2, Math.floor(v / 25) - 2];
  let count = 0, total = 0; const first = [0, 0, 0];
  for (let v = 0; v < 125; v++) if (live(v)) { const g = at(v); count++; total += vertex(v); for (let a = 0; a < 3; a++) first[a]! += g[a]!; }
  if (count === 0) return 2 * hmin;
  const mean = first.map(f => f / count), level = total / count, diagonal = [0, 0, 0], cross = [0, 0, 0], moment = [0, 0, 0];
  for (let v = 0; v < 125; v++) if (live(v)) {
    const g = at(v), r = [g[0]! - mean[0]!, g[1]! - mean[1]!, g[2]! - mean[2]!];
    for (let a = 0; a < 3; a++) { diagonal[a]! += r[a]! * r[a]!; moment[a]! += r[a]! * (vertex(v) - level); }
    cross[0]! += r[0]! * r[1]!; cross[1]! += r[0]! * r[2]!; cross[2]! += r[1]! * r[2]!;
  }
  const c0 = [diagonal[0]!, cross[0]!, cross[1]!], c1 = [cross[0]!, diagonal[1]!, cross[2]!], c2 = [cross[1]!, cross[2]!, diagonal[2]!];
  const x = (a: number[], b: number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
  const dot = (a: number[], b: number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  const determinant = dot(c0, x(c1, c2)), slope = [dot(moment, x(c1, c2)) / determinant, dot(c0, x(moment, c2)) / determinant, dot(c0, x(c1, moment)) / determinant];
  return level - dot(slope, mean);
}

/** A cut tile's fit (cutTile): its plane at the tile centre, and the sides
 * that plane is through: 1 the side cutSide grows, 2 the rest, 3 both.
 * `vertex` is phi at a vertex of the tile's 5^3; `hmin` the cell size. */
export function cutFit(open: OpenCells, vertex: (v: number) => number, hmin: number): { phi: number; sides: number } {
  if (open[0] === 0 && open[1] === 0) return { phi: 2 * hmin, sides: 0 };
  const side = cutSide(open), own = new Uint8Array(125), rest = new Uint8Array(125);
  let wet = false, wetRest = false, sum = 0, sumRest = 0, n = 0, nRest = 0;
  for (let v = 0; v < 125; v++) {
    if (!cutLive(open, v)) continue;
    const value = vertex(v);
    if (cutLive(side, v)) { own[v] = 1; wet ||= value < 0; sum += value; n++; } else { rest[v] = 1; wetRest ||= value < 0; sumRest += value; nRest++; }
  }
  // Liquid on one side of a wall alone: the fit is that side's. Liquid on
  // neither: the side nearer it, by its mean phi.
  const near = sum * nRest <= sumRest * n, keep = wet || (!wetRest && near), keepRest = wetRest || (!wet && !near);
  return { phi: planeAtCentre(v => (keep && own[v] === 1) || (keepRest && rest[v] === 1), vertex, hmin), sides: (keep ? 1 : 0) + (keepRest ? 2 : 0) };
}

/** An uncut tile the simulation holds at h (pcHeldFit): the mean of its h cells' centre phi. */
export function heldFit(vertex: (v: number) => number): number {
  let total = 0;
  for (let v = 0; v < 125; v++) {
    const g = [v % 5, Math.floor(v / 5) % 5, Math.floor(v / 25)];
    total += g.reduce((w, c) => w * (c === 0 || c === 4 ? 0.5 : 1), 1) * vertex(v);
  }
  return total / 64;
}

/** The open h cells on the sides a tile's phi was fit through (umWetSide). */
export function wetSide(tile: WetTile): [number, number] {
  if (!tile.cut) return [tile.open[0], tile.open[1]];
  const side = cutSide(tile.open), held = tile.sides;
  return [(((held & 1) !== 0 ? side[0] : 0) | ((held & 2) !== 0 ? tile.open[0] & ~side[0] : 0)) >>> 0,
    (((held & 1) !== 0 ? side[1] : 0) | ((held & 2) !== 0 ? tile.open[1] & ~side[1] : 0)) >>> 0];
}

/** The +axis face of tile `low`, toward tile `high` (umWetFace): the V and
 * the mask (bit i + 4j) of the h faces the liquid's rows hold.
 * `faceV(i, j)` is umPatchFaceV of the h face at (i, j) of the plane, i along
 * (axis + 1) % 3 and j along (axis + 2) % 3. `vertex(plane, i, j)` is phi at
 * vertex (i, j) of the plane 3 + plane cells from the low tile's origin along
 * the axis: read for plane 0 where the low tile is held at h, for plane 2
 * where the high one is, and for plane 1 where either is. */
export function wetFace(axis: number, low: WetTile, high: WetTile, faceV: (i: number, j: number) => number, vertex: (plane: number, i: number, j: number) => number): { v: number; mask: number } {
  const a = (axis + 1) % 3, b = (axis + 2) % 3, f = Math.fround;
  const sideLow = wetSide(low), sideHigh = wetSide(high), surfaceLow = low.row && !high.row, surfaceHigh = high.row && !low.row;
  const cell = (i: number, j: number, along: number) => { const l = [0, 0, 0]; l[a] = i; l[b] = j; l[axis] = along; return l[0]! + 4 * l[1]! + 16 * l[2]!; };
  const corners = (plane: number, i: number, j: number) => f(f(f(vertex(plane, i, j)) + f(vertex(plane, i + 1, j))) + f(f(vertex(plane, i, j + 1)) + f(vertex(plane, i + 1, j + 1))));
  let kept = 0, mask = 0;
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
    const v = faceV(i, j);
    if (v <= 0) continue;
    const isLow = bit(low.open, cell(i, j, 3)), isHigh = bit(high.open, cell(i, j, 0));
    let wetLow = isLow && low.row, wetHigh = isHigh && high.row;
    if (isLow && low.fine) wetLow = f(corners(1, i, j) + corners(0, i, j)) < 0;
    if (isHigh && high.fine) wetHigh = f(corners(1, i, j) + corners(2, i, j)) < 0;
    const surface = isLow && isHigh && ((surfaceLow && bit(sideLow, cell(i, j, 3))) || (surfaceHigh && bit(sideHigh, cell(i, j, 0))));
    if (((wetLow || wetHigh) && (isLow || low.row) && (isHigh || high.row)) || surface) { kept += v; mask |= 1 << (i + 4 * j); }
  }
  return { v: kept / 16, mask };
}
