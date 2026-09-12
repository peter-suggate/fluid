/**
 * A 2-D slice that runs the method, small enough to draw.
 *
 * The advance lab needs liquid on a grid that behaves the way Sparse CM12
 * behaves, at a size a canvas can show a cell of. This is that model, and it
 * is a model of the *algorithm*, not a picture of one: staggered face
 * velocities cut against exact cut-cell capacity, an exact PLIC line per
 * interface cell, swept-prism volume fluxes handed across one shared subface,
 * a bounded limiter on every donor, and the resident shader's own microstep
 * rule. Volume is conserved to the bit because every transfer is a paired
 * debit and credit, which is the property the lab exists to show.
 *
 * What it is not: it is not the solver. There is no brick atlas, no topology
 * transaction, no pressure cache — the lab reads those from the stage registry
 * and the work model beside this file. Nothing here may be imported by the
 * method itself.
 */

/** Fine cells across and down. 12 x 5 bricks of 8. */
export const SLICE_NX = 96;
export const SLICE_NY = 40;
export const SLICE_BRICK = 8;
export const SLICE_BX = SLICE_NX / SLICE_BRICK;
export const SLICE_BY = SLICE_NY / SLICE_BRICK;
/** The dyadic ladder, as cells per brick edge — B8's own rungs. */
export const SLICE_RUNGS = Object.freeze([1, 2, 4, 8]);

const GRAVITY = 0.115;
const DT = 1;
/** Face speeds are capped so the microstep count stays legible on screen. */
const VELOCITY_CEILING = 2.6;
const MAX_MICROSTEPS = 16;
const EXTENSION_SWEEPS = 8;

/** Cell index. */
export const sliceCell = (x: number, y: number): number => y * SLICE_NX + x;
/** X-row index: the face on the low-x side of cell `x`. */
export const sliceRowX = (x: number, y: number): number => y * (SLICE_NX + 1) + x;
/** Y-row index: the face on the low-y side of cell `y`. */
export const sliceRowY = (x: number, y: number): number => y * SLICE_NX + x;

export interface SliceMarker { x: number; y: number }

export interface AdvanceSlice {
  /** Liquid volume held per cell. The conserved quantity: 0 <= V <= K. */
  readonly V: Float32Array;
  /** Open capacity after solids, from clipped cut-cell geometry. */
  readonly K: Float32Array;
  /** Volume as it stood when this advance's transport began. */
  readonly Vp: Float32Array;
  readonly u: Float32Array;
  readonly v: Float32Array;
  readonly u0: Float32Array;
  readonly v0: Float32Array;
  /** Face field entering the projection, kept so the lens can show both. */
  readonly uPre: Float32Array;
  readonly vPre: Float32Array;
  /** Last advance's swept flux per row, signed along the axis. */
  readonly fx: Float32Array;
  readonly fy: Float32Array;
  /** Rows the bounded limiter had to cut back. */
  readonly cx: Uint8Array;
  readonly cy: Uint8Array;
  readonly p: Float32Array;
  readonly div: Float32Array;
  /** Rows whose velocity came from extension rather than from liquid. */
  readonly ext: Uint8Array;
  readonly plicX: Float32Array;
  readonly plicY: Float32Array;
  readonly plicOffset: Float32Array;
  readonly rung: Int8Array;
  readonly rungWas: Int8Array;
  readonly activity: Float32Array;
  markers: SliceMarker[];
  frame: number;
  /** Microsteps the CFL plan asked for on the last advance. */
  microsteps: number;
  maxVelocity: number;
  /** Relative volume drift since the scene was seeded. Expected: zero. */
  drift: number;
  /** Bricks whose rung moved on the last advance. */
  churn: number;
  iterations: number;
  residual: number;
  seededVolume: number;
}

export function createAdvanceSlice(): AdvanceSlice {
  const cells = SLICE_NX * SLICE_NY;
  const slice: AdvanceSlice = {
    V: new Float32Array(cells), K: new Float32Array(cells), Vp: new Float32Array(cells),
    u: new Float32Array((SLICE_NX + 1) * SLICE_NY),
    v: new Float32Array(SLICE_NX * (SLICE_NY + 1)),
    u0: new Float32Array((SLICE_NX + 1) * SLICE_NY),
    v0: new Float32Array(SLICE_NX * (SLICE_NY + 1)),
    uPre: new Float32Array((SLICE_NX + 1) * SLICE_NY),
    vPre: new Float32Array(SLICE_NX * (SLICE_NY + 1)),
    fx: new Float32Array((SLICE_NX + 1) * SLICE_NY),
    fy: new Float32Array(SLICE_NX * (SLICE_NY + 1)),
    cx: new Uint8Array((SLICE_NX + 1) * SLICE_NY),
    cy: new Uint8Array(SLICE_NX * (SLICE_NY + 1)),
    p: new Float32Array(cells), div: new Float32Array(cells),
    ext: new Uint8Array((SLICE_NX + 1) * SLICE_NY),
    plicX: new Float32Array(cells), plicY: new Float32Array(cells),
    plicOffset: new Float32Array(cells),
    rung: new Int8Array(SLICE_BX * SLICE_BY),
    rungWas: new Int8Array(SLICE_BX * SLICE_BY),
    activity: new Float32Array(SLICE_BX * SLICE_BY),
    markers: [], frame: 0, microsteps: 1, maxVelocity: 0, drift: 0, churn: 0,
    iterations: 0, residual: 0, seededVolume: 0,
  };
  resetAdvanceSlice(slice);
  return slice;
}

const wet = (s: AdvanceSlice, i: number): boolean => s.V[i] > 1e-5;
const liquid = (s: AdvanceSlice, i: number): boolean =>
  s.K[i] > 0.02 && s.V[i] > 0.5 * s.K[i];

/**
 * The scene's solids, as one signed test.
 *
 * Walls, a weir standing on the floor, and a floor that ramps up on the right.
 * The ramp is deliberately off-axis: it is what gives the slice cells with a
 * fractional open capacity, which is the case a level set cannot hold and this
 * method can.
 */
export function sliceSolidAt(px: number, py: number): boolean {
  if (px < 1.5 || px > SLICE_NX - 1.5 || py > SLICE_NY - 1.5) return true;
  const floor = SLICE_NY - 1.5 - Math.max(0, Math.min(16, 0.52 * (px - 60)));
  if (py > floor) return true;
  return px > 31 && px < 36 && py > SLICE_NY - 15;
}

/** Where the ramp reaches its cap, in fine cells — the renderer draws to it. */
export const SLICE_RAMP_START = 60;
export const SLICE_RAMP_RISE = 16;
export const SLICE_RAMP_SLOPE = 0.52;
export const SLICE_WEIR = Object.freeze({ x0: 31, x1: 36, top: SLICE_NY - 15 });

export function resetAdvanceSlice(s: AdvanceSlice): void {
  s.V.fill(0); s.u.fill(0); s.v.fill(0); s.p.fill(0);
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    let open = 0;
    for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
      if (!sliceSolidAt(x + (i + 0.5) / 4, y + (j + 0.5) / 4)) open += 1;
    }
    s.K[sliceCell(x, y)] = open / 16;
  }
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const capacity = s.K[sliceCell(x, y)];
    if (capacity > 0 && x > 1.5 && x < 21 && y > 8) s.V[sliceCell(x, y)] = capacity;
  }
  s.Vp.set(s.V);
  let total = 0;
  for (let i = 0; i < s.V.length; i++) total += s.V[i];
  s.seededVolume = total;
  s.rung.fill(2); s.rungWas.fill(2);
  s.markers = [];
  for (let t = 0; t < 220; t++) {
    const x = 2 + Math.random() * 19, y = 9 + Math.random() * (SLICE_NY - 11);
    if (s.V[sliceCell(x | 0, y | 0)] > 0.5) s.markers.push({ x, y });
  }
  s.frame = 0; s.drift = 0; s.churn = 0; s.microsteps = 1; s.maxVelocity = 0;
  reconstruct(s);
}

/* ---- polygon geometry, shared with the display lattice ------------- */

export const UNIT_SQUARE: readonly number[] = Object.freeze([0, 0, 1, 0, 1, 1, 0, 1]);

/** Sutherland-Hodgman against one half-plane, keeping `ax*x + ay*y <= d`. */
export function clipUnitSquare(
  polygon: readonly number[], ax: number, ay: number, d: number,
): number[] {
  const out: number[] = [];
  for (let i = 0; i < polygon.length; i += 2) {
    const x0 = polygon[i], y0 = polygon[i + 1];
    const j = (i + 2) % polygon.length;
    const x1 = polygon[j], y1 = polygon[j + 1];
    const s0 = ax * x0 + ay * y0 - d, s1 = ax * x1 + ay * y1 - d;
    if (s0 <= 0) out.push(x0, y0);
    if ((s0 < 0 && s1 > 0) || (s0 > 0 && s1 < 0)) {
      const t = s0 / (s0 - s1);
      out.push(x0 + t * (x1 - x0), y0 + t * (y1 - y0));
    }
  }
  return out;
}

export function polygonArea(polygon: readonly number[]): number {
  let twice = 0;
  for (let i = 0; i < polygon.length; i += 2) {
    const j = (i + 2) % polygon.length;
    twice += polygon[i] * polygon[j + 1] - polygon[j] * polygon[i + 1];
  }
  return Math.abs(twice) * 0.5;
}

/**
 * The exact 2-D inverse: the offset whose half-plane cuts `fill` out of the
 * unit square, for a normal already scaled so |nx| + |ny| = 1.
 *
 * Solved in the mirrored frame where both components are positive, then
 * shifted back — reflecting an axis moves the offset by that component.
 */
export function plicOffset(fill: number, nx: number, ny: number): number {
  const m1 = Math.min(Math.abs(nx), Math.abs(ny));
  const m2 = Math.max(Math.abs(nx), Math.abs(ny));
  const corner = m1 / (2 * m2);
  const mirrored = fill < corner ? Math.sqrt(2 * m1 * m2 * fill)
    : fill < 1 - corner ? fill * m2 + m1 / 2
      : 1 - Math.sqrt(2 * m1 * m2 * (1 - fill));
  return mirrored + (nx < 0 ? nx : 0) + (ny < 0 ? ny : 0);
}

/** Youngs' normal from a 3x3 of fill fractions, normalized to |nx|+|ny| = 1. */
export function youngsNormal(
  sample: (dx: number, dy: number) => number,
): { nx: number; ny: number } | null {
  const gx = (sample(1, -1) + 2 * sample(1, 0) + sample(1, 1))
    - (sample(-1, -1) + 2 * sample(-1, 0) + sample(-1, 1));
  const gy = (sample(-1, 1) + 2 * sample(0, 1) + sample(1, 1))
    - (sample(-1, -1) + 2 * sample(0, -1) + sample(1, -1));
  const scale = Math.abs(gx) + Math.abs(gy);
  if (scale < 1e-8) return null;
  /* the liquid body's outward normal runs down the fill gradient */
  return { nx: -gx / scale, ny: -gy / scale };
}

/** Refresh every cell's PLIC plane from the volume field. */
function reconstruct(s: AdvanceSlice): void {
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const i = sliceCell(x, y), capacity = s.K[i];
    s.plicX[i] = 0; s.plicY[i] = 0; s.plicOffset[i] = 0;
    if (capacity <= 0.02) continue;
    const fill = s.V[i] / capacity;
    if (fill <= 1e-4 || fill >= 1 - 1e-4) continue;
    const normal = youngsNormal((dx, dy) => {
      const xx = Math.max(0, Math.min(SLICE_NX - 1, x + dx));
      const yy = Math.max(0, Math.min(SLICE_NY - 1, y + dy));
      const j = sliceCell(xx, yy);
      /* a solid neighbour reads as whatever this cell already is, so the
         reconstruction does not bend the interface into the wall */
      return s.K[j] <= 0.02 ? (fill > 0.5 ? 1 : 0) : s.V[j] / s.K[j];
    });
    if (!normal) continue;
    s.plicX[i] = normal.nx; s.plicY[i] = normal.ny;
    s.plicOffset[i] = plicOffset(fill, normal.nx, normal.ny);
  }
}

/** One cell's liquid polygon, in unit coordinates. */
export function sliceLiquidPolygon(s: AdvanceSlice, i: number): number[] | null {
  const capacity = s.K[i];
  if (capacity <= 0.02) return null;
  const fill = s.V[i] / capacity;
  if (fill >= 1 - 1e-4) return UNIT_SQUARE.slice();
  if (fill <= 1e-4) return null;
  if (s.plicX[i] === 0 && s.plicY[i] === 0) return UNIT_SQUARE.slice();
  return clipUnitSquare(UNIT_SQUARE, s.plicX[i], s.plicY[i], s.plicOffset[i]);
}

/** Liquid volume inside an axis-aligned box of one cell: the swept prism. */
function liquidIn(
  s: AdvanceSlice, i: number, x0: number, x1: number, y0: number, y1: number,
): number {
  const capacity = s.K[i];
  if (capacity <= 0.02 || s.V[i] <= 1e-9) return 0;
  const box = (x1 - x0) * (y1 - y0);
  if (box <= 0) return 0;
  if (s.V[i] / capacity >= 1 - 1e-4) return box * capacity;
  let polygon = sliceLiquidPolygon(s, i);
  if (!polygon) return 0;
  polygon = clipUnitSquare(polygon, -1, 0, -x0); if (!polygon.length) return 0;
  polygon = clipUnitSquare(polygon, 1, 0, x1); if (!polygon.length) return 0;
  polygon = clipUnitSquare(polygon, 0, -1, -y0); if (!polygon.length) return 0;
  polygon = clipUnitSquare(polygon, 0, 1, y1); if (!polygon.length) return 0;
  return polygonArea(polygon) * capacity;
}

const closedX = (s: AdvanceSlice, x: number, y: number): boolean => Math.min(
  x > 0 ? s.K[sliceCell(x - 1, y)] : 0,
  x < SLICE_NX ? s.K[sliceCell(x, y)] : 0,
) <= 0.05;
const closedY = (s: AdvanceSlice, x: number, y: number): boolean => Math.min(
  y > 0 ? s.K[sliceCell(x, y - 1)] : 0,
  y < SLICE_NY ? s.K[sliceCell(x, y)] : 0,
) <= 0.05;

/** Eight sweeps push face velocity out of the liquid, as VEX2 does. */
function extendVelocity(s: AdvanceSlice): void {
  s.ext.fill(0);
  const validX = new Uint8Array(s.u.length);
  const validY = new Uint8Array(s.v.length);
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x <= SLICE_NX; x++) {
    const lo = x > 0 ? sliceCell(x - 1, y) : -1;
    const hi = x < SLICE_NX ? sliceCell(x, y) : -1;
    validX[sliceRowX(x, y)] =
      (lo >= 0 && wet(s, lo)) || (hi >= 0 && wet(s, hi)) ? 1 : 0;
  }
  for (let y = 0; y <= SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const lo = y > 0 ? sliceCell(x, y - 1) : -1;
    const hi = y < SLICE_NY ? sliceCell(x, y) : -1;
    validY[sliceRowY(x, y)] =
      (lo >= 0 && wet(s, lo)) || (hi >= 0 && wet(s, hi)) ? 1 : 0;
  }
  const steps: readonly (readonly [number, number])[] =
    [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let sweep = 0; sweep < EXTENSION_SWEEPS; sweep++) {
    const nextX = validX.slice(), nextY = validY.slice();
    for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x <= SLICE_NX; x++) {
      const row = sliceRowX(x, y);
      if (validX[row]) continue;
      let sum = 0, count = 0;
      for (const [dx, dy] of steps) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx > SLICE_NX || yy < 0 || yy >= SLICE_NY) continue;
        if (validX[sliceRowX(xx, yy)]) { sum += s.u[sliceRowX(xx, yy)]; count += 1; }
      }
      if (count) { s.u[row] = sum / count; nextX[row] = 1; s.ext[row] = 1; }
    }
    for (let y = 0; y <= SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
      const row = sliceRowY(x, y);
      if (validY[row]) continue;
      let sum = 0, count = 0;
      for (const [dx, dy] of steps) {
        const xx = x + dx, yy = y + dy;
        if (xx < 0 || xx >= SLICE_NX || yy < 0 || yy > SLICE_NY) continue;
        if (validY[sliceRowY(xx, yy)]) { sum += s.v[sliceRowY(xx, yy)]; count += 1; }
      }
      if (count) { s.v[row] = sum / count; nextY[row] = 1; }
    }
    validX.set(nextX); validY.set(nextY);
  }
}

/** Gravity lands on rows, never on cells. */
function bodyForces(s: AdvanceSlice): void {
  for (let y = 0; y <= SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const lo = y > 0 ? sliceCell(x, y - 1) : -1;
    const hi = y < SLICE_NY ? sliceCell(x, y) : -1;
    if ((lo >= 0 && wet(s, lo)) || (hi >= 0 && wet(s, hi))) {
      s.v[sliceRowY(x, y)] += GRAVITY * DT;
    }
  }
}

/**
 * Divergence, solve, projection.
 *
 * Gauss-Seidel rather than the solver's pipelined CG: the lab's point is that
 * the projection is a fixed iteration budget whose tail stays encoded, and a
 * relaxation makes the iteration count visible in the picture.
 */
function project(s: AdvanceSlice, iterations: number): void {
  s.uPre.set(s.u); s.vPre.set(s.v);
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    const i = sliceCell(x, y);
    if (closedX(s, x, y)) s.u[sliceRowX(x, y)] = 0;
    if (closedX(s, x + 1, y)) s.u[sliceRowX(x + 1, y)] = 0;
    if (closedY(s, x, y)) s.v[sliceRowY(x, y)] = 0;
    if (closedY(s, x, y + 1)) s.v[sliceRowY(x, y + 1)] = 0;
    if (!liquid(s, i)) { s.p[i] = 0; s.div[i] = 0; continue; }
    s.div[i] = s.u[sliceRowX(x + 1, y)] - s.u[sliceRowX(x, y)]
      + s.v[sliceRowY(x, y + 1)] - s.v[sliceRowY(x, y)];
  }
  let residual = 0;
  for (let it = 0; it < iterations; it++) {
    residual = 0;
    for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
      const i = sliceCell(x, y);
      if (!liquid(s, i)) continue;
      let sum = 0, rows = 0;
      /* an air neighbour contributes p = 0: the ghost-fluid free surface */
      if (!closedX(s, x, y)) {
        rows += 1;
        if (liquid(s, sliceCell(x - 1, y))) sum += s.p[sliceCell(x - 1, y)];
      }
      if (!closedX(s, x + 1, y)) {
        rows += 1;
        if (liquid(s, sliceCell(x + 1, y))) sum += s.p[sliceCell(x + 1, y)];
      }
      if (!closedY(s, x, y)) {
        rows += 1;
        if (liquid(s, sliceCell(x, y - 1))) sum += s.p[sliceCell(x, y - 1)];
      }
      if (!closedY(s, x, y + 1)) {
        rows += 1;
        if (liquid(s, sliceCell(x, y + 1))) sum += s.p[sliceCell(x, y + 1)];
      }
      if (!rows) { s.p[i] = 0; continue; }
      const next = (sum - s.div[i]) / rows;
      residual = Math.max(residual, Math.abs(next - s.p[i]));
      s.p[i] = next;
    }
  }
  s.iterations = iterations; s.residual = residual;
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x <= SLICE_NX; x++) {
    if (closedX(s, x, y)) { s.u[sliceRowX(x, y)] = 0; continue; }
    const lo = x > 0 ? sliceCell(x - 1, y) : -1;
    const hi = x < SLICE_NX ? sliceCell(x, y) : -1;
    if ((lo >= 0 && liquid(s, lo)) || (hi >= 0 && liquid(s, hi))) {
      s.u[sliceRowX(x, y)] -= (hi >= 0 ? s.p[hi] : 0) - (lo >= 0 ? s.p[lo] : 0);
    }
  }
  for (let y = 0; y <= SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    if (closedY(s, x, y)) { s.v[sliceRowY(x, y)] = 0; continue; }
    const lo = y > 0 ? sliceCell(x, y - 1) : -1;
    const hi = y < SLICE_NY ? sliceCell(x, y) : -1;
    if ((lo >= 0 && liquid(s, lo)) || (hi >= 0 && liquid(s, hi))) {
      s.v[sliceRowY(x, y)] -= (hi >= 0 ? s.p[hi] : 0) - (lo >= 0 ? s.p[lo] : 0);
    }
  }
  let peak = 0;
  for (let i = 0; i < s.u.length; i++) {
    s.u[i] = Math.max(-VELOCITY_CEILING, Math.min(VELOCITY_CEILING, s.u[i]));
    peak = Math.max(peak, Math.abs(s.u[i]));
  }
  for (let i = 0; i < s.v.length; i++) {
    s.v[i] = Math.max(-VELOCITY_CEILING, Math.min(VELOCITY_CEILING, s.v[i]));
    peak = Math.max(peak, Math.abs(s.v[i]));
  }
  s.maxVelocity = peak;
}

function bilinear(
  field: Float32Array, width: number, height: number, x: number, y: number,
): number {
  const cx = Math.max(0, Math.min(width - 1.001, x));
  const cy = Math.max(0, Math.min(height - 1.001, y));
  const x0 = cx | 0, y0 = cy | 0, tx = cx - x0, ty = cy - y0;
  const i = y0 * width + x0;
  return (field[i] * (1 - tx) + field[i + 1] * tx) * (1 - ty)
    + (field[i + width] * (1 - tx) + field[i + width + 1] * tx) * ty;
}

function advectVelocity(s: AdvanceSlice): void {
  s.u0.set(s.u); s.v0.set(s.v);
  for (let y = 0; y < SLICE_NY; y++) for (let x = 0; x <= SLICE_NX; x++) {
    if (closedX(s, x, y)) continue;
    const xr = Math.min(SLICE_NX - 1, x), xl = Math.max(0, x - 1);
    const vy = 0.25 * (s.v0[sliceRowY(xr, y)] + s.v0[sliceRowY(xl, y)]
      + s.v0[sliceRowY(xr, y + 1)] + s.v0[sliceRowY(xl, y + 1)]);
    const px = x - s.u0[sliceRowX(x, y)] * DT, py = y - vy * DT;
    s.u[sliceRowX(x, y)] = bilinear(s.u0, SLICE_NX + 1, SLICE_NY, px, py);
  }
  for (let y = 0; y <= SLICE_NY; y++) for (let x = 0; x < SLICE_NX; x++) {
    if (closedY(s, x, y)) continue;
    const yb = Math.min(SLICE_NY - 1, y), yt = Math.max(0, y - 1);
    const vx = 0.25 * (s.u0[sliceRowX(x, yb)] + s.u0[sliceRowX(x, yt)]
      + s.u0[sliceRowX(x + 1, yb)] + s.u0[sliceRowX(x + 1, yt)]);
    const px = x - vx * DT, py = y - s.v0[sliceRowY(x, y)] * DT;
    s.v[sliceRowY(x, y)] = bilinear(s.v0, SLICE_NX, SLICE_NY + 1, px, py);
  }
}

/**
 * The microstep count, by the resident shader's own rule.
 *
 * `sealGeometricVolumePlan` takes `ceil(2 * CFL)` and faults above 128; the
 * slice caps lower because it draws every microstep's flux.
 */
export function planMicrosteps(maxVelocity: number): number {
  return Math.min(MAX_MICROSTEPS, Math.max(1, Math.ceil(2 * maxVelocity * DT)));
}

/**
 * Geometric volume transport.
 *
 * Per microstep, one sweep per axis, order alternating. A sweep cuts the swept
 * prism out of the donor's PLIC polygon, scales every donor's outflow to what
 * it actually holds, and stops short of overfilling a receiver. Each transfer
 * is one subtraction and one addition on a shared subface, so the sweep is
 * conservative by construction rather than by correction.
 */
function transport(s: AdvanceSlice): void {
  const microsteps = planMicrosteps(s.maxVelocity);
  s.microsteps = microsteps;
  const dtm = DT / microsteps;
  s.fx.fill(0); s.fy.fill(0); s.cx.fill(0); s.cy.fill(0);
  for (let step = 0; step < microsteps; step++) {
    const horizontalFirst = (s.frame + step) % 2 === 0;
    reconstruct(s);
    sweep(s, dtm, horizontalFirst);
    reconstruct(s);
    sweep(s, dtm, !horizontalFirst);
  }
  let total = 0;
  for (let i = 0; i < s.V.length; i++) {
    s.V[i] = Math.max(0, Math.min(s.K[i], s.V[i]));
    total += s.V[i];
  }
  s.drift = s.seededVolume > 0 ? (total - s.seededVolume) / s.seededVolume : 0;
  reconstruct(s);
}

function sweep(s: AdvanceSlice, dtm: number, horizontal: boolean): void {
  const flux = horizontal ? s.fx : s.fy;
  const clipped = horizontal ? s.cx : s.cy;
  const raw = new Float32Array(flux.length);
  const outflow = new Float32Array(SLICE_NX * SLICE_NY);
  const width = horizontal ? SLICE_NX + 1 : SLICE_NX;
  const height = horizontal ? SLICE_NY : SLICE_NY + 1;
  const rowOf = horizontal ? sliceRowX : sliceRowY;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const row = rowOf(x, y);
    if (horizontal ? closedX(s, x, y) : closedY(s, x, y)) continue;
    const swept = (horizontal ? s.u[row] : s.v[row]) * dtm;
    if (Math.abs(swept) < 1e-7) continue;
    const donor = horizontal
      ? (swept > 0 ? (x > 0 ? sliceCell(x - 1, y) : -1)
        : (x < SLICE_NX ? sliceCell(x, y) : -1))
      : (swept > 0 ? (y > 0 ? sliceCell(x, y - 1) : -1)
        : (y < SLICE_NY ? sliceCell(x, y) : -1));
    const receiver = horizontal
      ? (swept > 0 ? (x < SLICE_NX ? sliceCell(x, y) : -1)
        : (x > 0 ? sliceCell(x - 1, y) : -1))
      : (swept > 0 ? (y < SLICE_NY ? sliceCell(x, y) : -1)
        : (y > 0 ? sliceCell(x, y - 1) : -1));
    if (donor < 0 || receiver < 0) continue;
    const w = Math.min(0.98, Math.abs(swept));
    const volume = horizontal
      ? (swept > 0 ? liquidIn(s, donor, 1 - w, 1, 0, 1)
        : liquidIn(s, donor, 0, w, 0, 1))
      : (swept > 0 ? liquidIn(s, donor, 0, 1, 1 - w, 1)
        : liquidIn(s, donor, 0, 1, 0, w));
    raw[row] = Math.sign(swept) * volume;
    outflow[donor] += volume;
  }
  const limit = new Float32Array(SLICE_NX * SLICE_NY).fill(1);
  for (let i = 0; i < outflow.length; i++) {
    if (outflow[i] > s.V[i] && outflow[i] > 1e-9) limit[i] = s.V[i] / outflow[i];
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const row = rowOf(x, y);
    if (raw[row] === 0) continue;
    const forward = raw[row] > 0;
    const donor = horizontal
      ? (forward ? sliceCell(x - 1, y) : sliceCell(x, y))
      : (forward ? sliceCell(x, y - 1) : sliceCell(x, y));
    const receiver = horizontal
      ? (forward ? sliceCell(x, y) : sliceCell(x - 1, y))
      : (forward ? sliceCell(x, y) : sliceCell(x, y - 1));
    let volume = Math.abs(raw[row]) * limit[donor];
    const room = s.K[receiver] - s.V[receiver];
    if (volume > room) { volume = Math.max(0, room); clipped[row] = 1; }
    if (limit[donor] < 0.999) clipped[row] = 1;
    s.V[donor] -= volume;
    s.V[receiver] += volume;
    flux[row] = (forward ? 1 : -1) * volume;
  }
}

/** Markers ride the published transport velocity and carry no mass. */
function advectMarkers(s: AdvanceSlice): void {
  for (const marker of s.markers) {
    const mu = bilinear(s.u, SLICE_NX + 1, SLICE_NY, marker.x, Math.max(0, marker.y - 0.5));
    const mv = bilinear(s.v, SLICE_NX, SLICE_NY + 1, Math.max(0, marker.x - 0.5), marker.y);
    marker.x = Math.max(1.6, Math.min(SLICE_NX - 1.6, marker.x + mu * DT));
    marker.y = Math.max(0.6, Math.min(SLICE_NY - 1.6, marker.y + mv * DT));
  }
}

/**
 * Activity census, resolution plan, 2:1 grading.
 *
 * The score reads interface presence and peak speed and nothing else — no
 * authored region, no distance to a camera. Grading then pulls any neighbour
 * that would sit more than one rung away, which is what makes the ladder
 * dyadic rather than merely per-brick.
 */
function adapt(s: AdvanceSlice): void {
  s.rungWas.set(s.rung);
  for (let by = 0; by < SLICE_BY; by++) for (let bx = 0; bx < SLICE_BX; bx++) {
    let interface_ = 0, speed = 0, fill = 0, cells = 0;
    for (let j = 0; j < SLICE_BRICK; j++) for (let i = 0; i < SLICE_BRICK; i++) {
      const x = bx * SLICE_BRICK + i, y = by * SLICE_BRICK + j;
      const cell = sliceCell(x, y);
      if (s.K[cell] <= 0.02) continue;
      cells += 1;
      const f = s.V[cell] / s.K[cell];
      fill += f;
      if (f > 0.02 && f < 0.98) interface_ += 1;
      speed = Math.max(speed,
        Math.abs(s.u[sliceRowX(x, y)]) + Math.abs(s.v[sliceRowY(x, y)]));
    }
    const brick = by * SLICE_BX + bx;
    s.activity[brick] = !cells || fill < 1e-3 ? 0
      : Math.min(1, interface_ / 6) * 0.62 + Math.min(1, speed / 1.4) * 0.38;
  }
  for (let brick = 0; brick < s.activity.length; brick++) {
    const a = s.activity[brick];
    s.rung[brick] = a <= 0.001 ? 0 : a < 0.16 ? 1 : a < 0.42 ? 2 : 3;
  }
  const steps: readonly (readonly [number, number])[] =
    [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let pass = 0; pass < 4; pass++) {
    let moved = false;
    for (let by = 0; by < SLICE_BY; by++) for (let bx = 0; bx < SLICE_BX; bx++) {
      const brick = by * SLICE_BX + bx;
      for (const [dx, dy] of steps) {
        const nx = bx + dx, ny = by + dy;
        if (nx < 0 || nx >= SLICE_BX || ny < 0 || ny >= SLICE_BY) continue;
        const neighbour = ny * SLICE_BX + nx;
        if (s.rung[neighbour] < s.rung[brick] - 1) {
          s.rung[neighbour] = s.rung[brick] - 1;
          moved = true;
        }
      }
    }
    if (!moved) break;
  }
  let churn = 0;
  for (let brick = 0; brick < s.rung.length; brick++) {
    if (s.rung[brick] !== s.rungWas[brick]) churn += 1;
  }
  s.churn = churn;
}

/**
 * One advance, in the resident encoder's own order.
 *
 * Extension, forces, projection, velocity self-advection, projection again,
 * geometric transport, markers, then the activity census and the candidate
 * rungs — whose commit lands at the tail and is the *next* advance's input.
 */
export function advanceSlice(s: AdvanceSlice, pressureIterations: number): void {
  extendVelocity(s);
  bodyForces(s);
  project(s, pressureIterations);
  advectVelocity(s);
  project(s, pressureIterations);
  s.Vp.set(s.V);
  transport(s);
  advectMarkers(s);
  adapt(s);
  s.frame += 1;
}
