/**
 * The slice, drawn, and one lens per stage of the advance.
 *
 * Every lens paints onto the same picture — the same water, the same bricks,
 * the same rungs — so switching stages moves the reading rather than the
 * subject. That is the whole design: a stage list that changes what you can
 * see about one scene, not fifteen separate diagrams.
 */
import {
  buildSliceLattice, type LatticeCell, latticeCellAt, type LatticePlane,
  latticePlane, type SliceLattice,
} from "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import {
  type AdvanceSlice, clipUnitSquare, SLICE_BRICK, SLICE_RUNGS,
  sliceCell, sliceRowX, sliceRowY, UNIT_SQUARE,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import type { AdvanceStageId } from "../lib/methods/adaptive-volume/advance-slice/advance-work";
import type { SliceSharedRdfIsocontour } from
  "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import { sliceRdfTriangles, type SliceRdfVertex } from
  "../lib/methods/adaptive-volume/advance-slice/slice-rdf-triangulation";

/**
 * The drawn palette, resolved from the page's own theme.
 *
 * These are the same tokens the chrome around the canvas is built from, which
 * is the point: a 2-D diagram of the solver is a drawing on the page, not a
 * photograph of something else, so it follows the reader's light or dark
 * setting exactly as every panel does. The values below are the dark reading,
 * kept as the fallback for a context with no host element to ask — a test, or
 * a server render — and `syncPalette` replaces them from the live custom
 * properties declared beside the viewport in `AdvanceLab.module.css`.
 */
export type PaletteTone =
  | "ground" | "grid" | "brick" | "liquid" | "solid" | "solidEdge"
  | "amber" | "muted" | "ink" | "alarm"
  | "transport" | "momentum" | "pressure" | "adaptivity" | "output";

export const PALETTE: Record<PaletteTone, string> = {
  ground: "#0b1420",
  grid: "#20364a",
  brick: "#33536b",
  liquid: "#2f7fd4",
  solid: "#3c4d5e",
  solidEdge: "#54697c",
  amber: "#d9a05b",
  muted: "#7d94a8",
  ink: "#dce8f0",
  alarm: "#e0705a",
  transport: "#e0aa62",
  momentum: "#5fd3b0",
  pressure: "#6aa9f0",
  adaptivity: "#b192e6",
  output: "#d9c55f",
};

/** Where each tone is authored. One name, used by the canvas and by the CSS. */
const PALETTE_VAR: Readonly<Record<PaletteTone, string>> = {
  ground: "--slice-ground",
  grid: "--slice-grid",
  brick: "--slice-brick",
  liquid: "--slice-liquid",
  solid: "--slice-solid",
  solidEdge: "--slice-solid-edge",
  amber: "--slice-amber",
  muted: "--slice-muted",
  ink: "--slice-ink",
  alarm: "--slice-alarm",
  transport: "--slice-transport",
  momentum: "--slice-momentum",
  pressure: "--slice-pressure",
  adaptivity: "--slice-adaptivity",
  output: "--slice-output",
};

/** The tone as the chrome states it, for a swatch React draws rather than the canvas. */
export const paletteVar = (tone: PaletteTone): string => `var(${PALETTE_VAR[tone]})`;

/**
 * Read the drawn palette back off the page, before painting.
 *
 * The canvas cannot hold a `var()`, so the tokens are resolved here each paint
 * — which is also what keeps the picture and the legend beside it in step when
 * the reader changes theme mid-frame. The properties are authored as plain
 * colours per theme rather than through `light-dark()`, because an unregistered
 * custom property comes back from `getComputedStyle` as the literal tokens it
 * was written with, and `light-dark(...)` is not something a canvas can fill.
 */
export function syncPalette(host: Element): void {
  const style = getComputedStyle(host);
  for (const tone of Object.keys(PALETTE_VAR) as PaletteTone[]) {
    const value = style.getPropertyValue(PALETTE_VAR[tone]).trim();
    if (value) PALETTE[tone] = value;
  }
}

/** Which tone stands for each band of the advance. */
export const BAND_TONE = {
  transport: "transport",
  momentum: "momentum",
  pressure: "pressure",
  adaptivity: "adaptivity",
  output: "output",
} as const satisfies Record<string, PaletteTone>;

/** A legend entry: which tone it is drawn in, and what it means in this lens. */
export type LensKey = readonly [tone: PaletteTone, label: string];

export interface Lens {
  /** What the reader is looking at, in the lens's own terms. */
  readonly caption: string;
  readonly keys: readonly LensKey[];
  readonly draw: (c: LensContext) => void;
}

export interface LensContext {
  readonly g: CanvasRenderingContext2D;
  readonly s: AdvanceSlice;
  readonly lattice: SliceLattice;
  /** Pixels per fine cell. */
  readonly scale: number;
}

/* ---- base picture -------------------------------------------------- */

function arrow(
  g: CanvasRenderingContext2D, x: number, y: number,
  dx: number, dy: number, width = 1.2,
): void {
  const length = Math.hypot(dx, dy);
  if (length < 1e-4) return;
  g.beginPath();
  g.moveTo(x, y);
  g.lineTo(x + dx, y + dy);
  const angle = Math.atan2(dy, dx), head = Math.min(5, 2 + length * 0.3);
  g.lineTo(x + dx - head * Math.cos(angle - 0.42), y + dy - head * Math.sin(angle - 0.42));
  g.moveTo(x + dx, y + dy);
  g.lineTo(x + dx - head * Math.cos(angle + 0.42), y + dy - head * Math.sin(angle + 0.42));
  g.lineWidth = width;
  g.stroke();
}

function tint(
  c: LensContext, x0: number, y0: number, width: number,
  color: string, alpha: number, height = width,
): void {
  c.g.fillStyle = color;
  c.g.globalAlpha = alpha;
  c.g.fillRect(x0 * c.scale, y0 * c.scale, width * c.scale, height * c.scale);
  c.g.globalAlpha = 1;
}

function label(
  c: LensContext, x: number, y: number, text: string, color: string = PALETTE.ink,
): void {
  c.g.font = "600 10px ui-monospace, monospace";
  c.g.fillStyle = color;
  c.g.textAlign = "center";
  c.g.textBaseline = "middle";
  c.g.fillText(text, x, y);
  c.g.textAlign = "left";
  c.g.textBaseline = "alphabetic";
}

/** Draw the selected production scene's sampled centre-plane capacity. */
function drawSolidRaster(c: LensContext): void {
  const { g, s, scale: S } = c;
  for (let y = 0; y < s.ny; y += 1) for (let x = 0; x < s.nx; x += 1) {
    const closed = 1 - s.K[sliceCell(s, x, y)]!;
    if (closed <= 1e-4) continue;
    g.globalAlpha = Math.max(0.18, closed);
    g.fillStyle = PALETTE.solid;
    g.fillRect(x * S, y * S, S, S);
  }
  g.globalAlpha = 1;
}

function clippedScalarTriangle(points: readonly SliceRdfVertex[]): number[] {
  const polygon: [number, number, number][] = [];
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i]!, b = points[(i + 1) % points.length]!;
    if (a[2] <= 0) polygon.push([...a]);
    if ((a[2] < 0) !== (b[2] < 0)) {
      const t = a[2] / (a[2] - b[2]);
      polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0]);
    }
  }
  return polygon.flatMap(point => [point[0], point[1]]);
}

function appendPolygon(g: CanvasRenderingContext2D, polygon: readonly number[], scale: number): void {
  if (polygon.length < 6) return;
  g.moveTo(polygon[0]! * scale, polygon[1]! * scale);
  for (let i = 2; i < polygon.length; i += 2) g.lineTo(polygon[i]! * scale, polygon[i + 1]! * scale);
  g.closePath();
}

function polygonArea(polygon: readonly number[]): number {
  let twice = 0;
  for (let index = 0; index < polygon.length; index += 2) {
    const next = (index + 2) % polygon.length;
    twice += polygon[index]! * polygon[next + 1]! - polygon[next]! * polygon[index + 1]!;
  }
  return Math.abs(twice) / 2;
}

export function rdfMinorityAreaDistorted(acceptedLiquid: number,
  representedLiquid: number, area: number): boolean {
  const accepted = acceptedLiquid <= area / 2 ? acceptedLiquid : area - acceptedLiquid;
  const represented = acceptedLiquid <= area / 2 ? representedLiquid : area - representedLiquid;
  const absoluteError = Math.abs(represented - accepted);
  if (!(absoluteError > 1e-3 * area)) return false;
  const smaller = Math.min(accepted, represented), larger = Math.max(accepted, represented);
  return smaller <= 0 ? larger > 0 : larger > 1.5 * smaller;
}

export function sliceRdfPlicFallbackCells(s: AdvanceSlice, lattice: SliceLattice,
  sharedRdf: SliceSharedRdfIsocontour): ReadonlySet<number> {
  const result = new Set<number>(), stride = s.nx + 1, phi = sharedRdf.vertexPhiFine;
  for (const cell of lattice.cells) {
    if (!cell.open || cell.capacity < 0.999999 * cell.width * cell.height
      || cell.fill <= 1e-6 || cell.fill >= 1 - 1e-6) continue;
    const topologyY0 = s.ny - cell.y0 - cell.height;
    let valid = true;
    for (let y = topologyY0; y < topologyY0 + cell.height; y += 1)
      for (let x = cell.x0; x < cell.x0 + cell.width; x += 1) {
        const values = [phi[x + stride * y], phi[x + 1 + stride * y],
          phi[x + stride * (y + 1)], phi[x + 1 + stride * (y + 1)]];
        if (!values.every(Number.isFinite)) valid = false;
      }
    if (!valid) result.add(cell.topologyCell);
  }
  return result;
}

export interface SliceRdfDisplayCell {
  readonly topologyCell: number;
  readonly mode: "rdf" | "plic";
  readonly reason: "shared-rdf" | "cut-cell" | "nonfinite-rdf";
  /** The canonical shared samples consumed by the canvas, in topology coordinates. */
  readonly samples: readonly { readonly x: number; readonly y: number; readonly value: number }[];
}

/** Exact per-partial-cell branch and scalar samples consumed by `drawSlice`. */
export function inspectSliceRdfDisplay(s: AdvanceSlice, lattice: SliceLattice,
  sharedRdf: SliceSharedRdfIsocontour): readonly SliceRdfDisplayCell[] {
  const fallback = sliceRdfPlicFallbackCells(s, lattice, sharedRdf);
  const stride = s.nx + 1, phi = sharedRdf.vertexPhiFine;
  return lattice.cells.filter(cell => cell.open && cell.fill > 1e-6 && cell.fill < 1 - 1e-6)
    .map(cell => {
      const cut = cell.capacity < 0.999999 * cell.width * cell.height;
      const topologyY0 = s.ny - cell.y0 - cell.height;
      const samples: { x: number; y: number; value: number }[] = [];
      for (let y = topologyY0; y <= topologyY0 + cell.height; y += 1)
        for (let x = cell.x0; x <= cell.x0 + cell.width; x += 1)
          samples.push({ x, y, value: phi[x + stride * y]! });
      return {
        topologyCell: cell.topologyCell,
        mode: cut || fallback.has(cell.topologyCell) ? "plic" as const : "rdf" as const,
        reason: cut ? "cut-cell" as const : fallback.has(cell.topologyCell)
          ? "nonfinite-rdf" as const : "shared-rdf" as const,
        samples,
      };
    });
}

/** Grid at each brick's rung, liquid cut by PLIC, bricks, then solids. */
export function drawSlice(c: LensContext, sharedRdf?: SliceSharedRdfIsocontour): void {
  const { g, s, lattice, scale: S } = c;
  buildSliceLattice(lattice, s);
  g.clearRect(0, 0, s.nx * S, s.ny * S);
  g.fillStyle = PALETTE.ground;
  g.fillRect(0, 0, s.nx * S, s.ny * S);

  g.fillStyle = PALETTE.liquid;
  g.beginPath();
  let plicFallback: ReadonlySet<number> = new Set<number>();
  if (sharedRdf) {
    const stride = s.nx + 1, phi = sharedRdf.vertexPhiFine;
    plicFallback = sliceRdfPlicFallbackCells(s, lattice, sharedRdf);
    for (let y = 0; y < s.ny; y += 1) for (let x = 0; x < s.nx; x += 1) {
      const canvasY = s.ny - 1 - y;
      // Immersed/cut-cell geometry does not yet expose the open polygon needed
      // by RDF. It is rendered by the explicit PLIC/fill fallback below and is
      // counted in the preview receipt rather than silently crossed.
      const dense = sliceCell(s, x, canvasY), capacity = s.K[dense]!;
      if (capacity < 0.999999) continue;
      const acceptedFill = s.V[dense]! / Math.max(capacity, 1e-8);
      // A pure accepted owner is stronger evidence than a render-only RDF.
      // Publishing it directly prevents a shared-vertex fit from carving an
      // enclosed opposite-phase cell out of homogeneous bulk. Mixed owners
      // retain the shared RDF, including genuine subcell sheets and droplets.
      if (acceptedFill >= 1 - 1e-6) {
        g.rect(x * S, canvasY * S, S, S);
        continue;
      }
      if (acceptedFill <= 1e-6) continue;
      const owner = latticeCellAt(lattice, s, x + 0.5, canvasY + 0.5);
      if (owner && plicFallback.has(owner.topologyCell)) continue;
      const a = phi[x + stride * y]!;
      const b = phi[x + 1 + stride * y]!;
      const d = phi[x + stride * (y + 1)]!;
      const e = phi[x + 1 + stride * (y + 1)]!;
      if (![a, b, d, e].every(Number.isFinite)) continue;
      for (const triangle of sliceRdfTriangles(x, canvasY, d, e, b, a))
        appendPolygon(g, clippedScalarTriangle(triangle), S);
    }
  }
  for (const cell of lattice.cells) {
    if (!cell.open || cell.fill <= 1e-3) continue;
    if (sharedRdf && cell.capacity >= 0.999999 * cell.width * cell.height
      && !plicFallback.has(cell.topologyCell)) continue;
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    if (cell.fill >= 1 - 1e-6) { g.rect(x, y, w, h); continue; }
    const plane = plicFallback.has(cell.topologyCell) ? cell.plane : latticePlane(lattice, cell);
    /* No published plane means the solver could not resolve this interface, so
     * the picture falls back to the monotone reading the solver itself falls
     * back to: the liquid held at the bottom of the cell. Gravity is +y here —
     * filling from the top edge down would draw every unresolved cell upside
     * down, and a row of them reads as a sheet of water floating over a gap. */
    if (!plane) { g.rect(x, y + h * (1 - cell.fill), w, h * cell.fill); continue; }
    const polygon = clipUnitSquare(UNIT_SQUARE, plane.nx, plane.ny, plane.offset);
    if (polygon.length < 6) continue;
    g.moveTo(x + polygon[0] * w, y + polygon[1] * h);
    for (let i = 2; i < polygon.length; i += 2) {
      g.lineTo(x + polygon[i] * w, y + polygon[i + 1] * h);
    }
    g.closePath();
  }
  g.globalAlpha = 0.9;
  g.fill();
  g.globalAlpha = 1;

  /* The lattice is drawn *over* the water, not under it.
   *
   * The rung a brick carries is the whole adaptivity story, and a filled cell
   * is exactly where it matters most — a solid wash of liquid with the cell
   * edges hidden beneath it says the interior is uniform, which is the one
   * thing this solver is not. So the cell grid goes on last of the three, at an
   * alpha that reads on the ground and on the liquid alike. */
  g.lineWidth = 1;
  g.strokeStyle = PALETTE.grid;
  g.globalAlpha = 0.75;
  g.beginPath();
  for (const cell of lattice.cells) {
    g.rect(cell.x0 * S, cell.y0 * S, cell.width * S, cell.height * S);
  }
  g.stroke();
  g.globalAlpha = 1;

  g.lineWidth = 1.4;
  g.strokeStyle = PALETTE.brick;
  for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
    g.globalAlpha = 0.3 + 0.17 * s.rung[by * s.bx + bx]!;
    g.strokeRect(bx * SLICE_BRICK * S, by * SLICE_BRICK * S,
      Math.min(SLICE_BRICK, s.nx - bx * SLICE_BRICK) * S,
      Math.min(SLICE_BRICK, s.ny - by * SLICE_BRICK) * S);
  }
  g.globalAlpha = 1;
  drawSolidRaster(c);
}

function velocityField(
  c: LensContext, stride: number, color: string, alpha: number, before = false,
): void {
  const { g, s, scale: S } = c;
  g.strokeStyle = color;
  g.globalAlpha = alpha;
  const u = before ? s.uPre : s.u;
  const v = before ? s.vPre : s.v;
  for (let y = 1; y < s.ny - 1; y += stride) {
    for (let x = 1; x < s.nx - 1; x += stride) {
      const i = sliceCell(s, x, y);
      if (s.K[i] <= 0.05 || s.V[i] <= 1e-5) continue;
      const ux = 0.5 * (u[sliceRowX(s, x, y)]! + u[sliceRowX(s, x + 1, y)]!);
      const uy = 0.5 * (v[sliceRowY(s, x, y)]! + v[sliceRowY(s, x, y + 1)]!);
      arrow(g, (x + 0.5) * S, (y + 0.5) * S, ux * S * 1.5, uy * S * 1.5, 1.1);
    }
  }
  g.globalAlpha = 1;
}

/** One interface chord inside a cell: `[ax, ay, bx, by]` in unit-cell terms. */
export type InterfaceSegment = readonly [number, number, number, number];

const onCellEdge = (value: number): boolean => value < 1e-6 || value > 1 - 1e-6;

/**
 * The interface inside one cell, in unit-cell coordinates.
 *
 * `clipUnitSquare` returns the *liquid polygon*, and most of that polygon's
 * boundary is cell edge. Only the chords that cross the box are the surface;
 * keeping the rest would outline every cut cell instead of drawing the water
 * line through it. Shared rather than inlined because both the interface stroke
 * and the normal overlay have to agree on where the surface is — an arrow
 * anchored by one rule to a line drawn by another is a picture of nothing.
 */
export function interfaceSegments(plane: LatticePlane): readonly InterfaceSegment[] {
  const polygon = clipUnitSquare(UNIT_SQUARE, plane.nx, plane.ny, plane.offset);
  const segments: InterfaceSegment[] = [];
  for (let i = 0; i < polygon.length; i += 2) {
    const j = (i + 2) % polygon.length;
    const ax = polygon[i]!, ay = polygon[i + 1]!;
    const bx = polygon[j]!, by = polygon[j + 1]!;
    if (onCellEdge(ax) && onCellEdge(bx) && Math.abs(ax - bx) < 1e-6) continue;
    if (onCellEdge(ay) && onCellEdge(by) && Math.abs(ay - by) < 1e-6) continue;
    segments.push([ax, ay, bx, by]);
  }
  return segments;
}

/** Every PLIC segment, which is what presentation actually publishes. */
function drawInterface(c: LensContext, color: string, width: number): void {
  const { g, lattice, scale: S } = c;
  g.strokeStyle = color;
  g.lineWidth = width;
  g.lineCap = "round";
  g.beginPath();
  for (const cell of lattice.cells) {
    if (!cell.open) continue;
    const plane = latticePlane(lattice, cell);
    if (!plane) continue;
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    for (const [ax, ay, bx, by] of interfaceSegments(plane)) {
      g.moveTo(x + ax * w, y + ay * h);
      g.lineTo(x + bx * w, y + by * h);
    }
  }
  g.stroke();
  g.lineCap = "butt";
}

const brickVolume = (s: AdvanceSlice, bx: number, by: number): number => {
  let total = 0;
  for (let j = 0; j < SLICE_BRICK; j++) for (let i = 0; i < SLICE_BRICK; i++) {
    const x = bx * SLICE_BRICK + i, y = by * SLICE_BRICK + j;
    if (x < s.nx && y < s.ny) total += s.V[sliceCell(s, x, y)]!;
  }
  return total;
};

const cellMean = (
  s: AdvanceSlice, cell: LatticeCell, read: (index: number) => number,
): number => {
  let total = 0, count = 0;
  for (let j = 0; j < cell.height; j++) for (let i = 0; i < cell.width; i++) {
    total += read(sliceCell(s, cell.x0 + i, cell.y0 + j));
    count += 1;
  }
  return count ? total / count : 0;
};

/* ---- optional overlays ---------------------------------------------- */

/**
 * Two readings the picture implies but never states, drawn over any lens.
 *
 * The lab's other views are *lenses*: one per stage, mutually exclusive,
 * chosen by the strip, and each one replaces the reading. These are not that.
 * Volume fraction and the interface normal are what a cell carries at every
 * stage, so they compose over whichever lens is selected instead of being two
 * more stages to choose between — the same split the 3-D catalog draws between
 * a field view and a decoration, and for the same reason.
 *
 * Both are already in the probe bubble, for one cell, under the pointer. These
 * are those two rows across the whole grid at once, which is the question the
 * probe cannot answer: not "what is this cell" but "where does this change".
 */
export type SliceOverlayId = "fraction" | "normal";

export interface SliceOverlay {
  /** The word on the toggle. */
  readonly label: string;
  /** What it draws, for the control's tooltip and the sidebar caption. */
  readonly caption: string;
  readonly keys: readonly LensKey[];
  readonly draw: (c: LensContext) => void;
}

/** Declaration order, which is also draw order: washes first, lines over them. */
export const SLICE_OVERLAY_ORDER = ["fraction", "normal"] as const;

/**
 * Below this a cell is vacuum, not dilute.
 *
 * Six decades under a full cell — the bottom of the residue band transport
 * actually produces, and the same floor `dense-grid/density` draws vacuum at.
 * A cell under it keeps its grid lines and nothing else: "is there any liquid
 * here at all" is the first question this overlay has to answer, and a floor of
 * tinted haze over empty cells is how that answer gets lost.
 */
export const FRACTION_FLOOR = 1e-6;

/**
 * Where a sub-half fraction sits on the residue ramp, in [0, 1].
 *
 * Volume fraction is not a linear quantity down here. Transport leaves residue
 * across every decade between the floor and about 10⁻², and a linear ramp over
 * [0, ½] buries all of it in the bottom two percent: every residue cell then
 * draws the same near-nothing at the same near-zero alpha, which is the one
 * failure this overlay exists to prevent. A unit of the ramp is a fixed number
 * of decades instead, so the low end separates from itself.
 */
export function fractionResidueRamp(fill: number): number {
  return Math.min(1, Math.max(0,
    Math.log2(Math.max(fill, FRACTION_FLOOR) / FRACTION_FLOOR)
    / Math.log2(0.5 / FRACTION_FLOOR)));
}

/**
 * The fraction as the fewest characters that keep it honest.
 *
 * A cell is a handful of pixels wide, so the readout is sized to the answer
 * rather than formatted uniformly: `.42` for anything the two decimals can
 * carry, a bare `1` for a full cell, and `1e-4` once two decimals would round
 * a resolved residue cell to `.00` and make it indistinguishable from vacuum.
 * Overfull keeps its whole value — `1.04` is a fault the lab reports, and the
 * digit that says how far past capacity the cell is, is the point of it.
 */
export function fractionReadout(fill: number): string {
  if (fill > 1 + 1e-4) return fill.toFixed(2);
  if (fill >= 0.995) return "1";
  if (fill >= 0.005) return fill.toFixed(2).slice(1);
  return `1e${Math.round(Math.log10(Math.max(fill, FRACTION_FLOOR)))}`;
}

/** Roughly the pixels `label` needs for a readout, at its 10px monospace. */
const readoutPixels = (text: string): number => text.length * 6 + 5;

export const SLICE_OVERLAYS: Readonly<Record<SliceOverlayId, SliceOverlay>> = {
  fraction: {
    label: "fraction",
    caption: "V/K per accepted cell — the conserved quantity itself, read off the compact record rather than resampled. The water already draws the liquid half of the range geometrically, so the tint is spent where the geometry cannot help: the dilute decades below a half, which a cut line renders as a sliver too thin to see, and the overfull cells past V = K that the projection has to drain.",
    keys: [["ink", "V/K"], ["transport", "dilute residue, by decade"],
      ["alarm", "overfull, V > K"]],
    draw(c) {
      const { g, lattice, scale: S } = c;
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= FRACTION_FLOOR) continue;
        const x = cell.x0 * S, y = cell.y0 * S;
        const w = cell.width * S, h = cell.height * S;
        const overfull = cell.fill > 1 + 1e-4;
        /* The liquid band gets no wash. Between a half and a full cell the
         * picture underneath is already the answer — a PLIC polygon covering
         * that share of the cell — and tinting it would only dim the one part
         * of this field the reader can already measure by eye. */
        if (overfull) {
          g.fillStyle = PALETTE.alarm;
          g.globalAlpha = 0.42;
          g.fillRect(x, y, w, h);
          g.globalAlpha = 1;
        } else if (cell.fill < 0.5) {
          g.fillStyle = PALETTE.transport;
          g.globalAlpha = 0.14 + 0.40 * fractionResidueRamp(cell.fill);
          g.fillRect(x, y, w, h);
          g.globalAlpha = 1;
        }
        const text = fractionReadout(cell.fill);
        if (Math.min(w, h) < readoutPixels(text)) continue;
        label(c, x + w / 2, y + h / 2, text, overfull ? PALETTE.alarm : PALETTE.ink);
      }
    },
  },
  normal: {
    label: "normals",
    caption: "The PLIC normal each cut cell carries, drawn from the middle of its own interface chord and pointing out of the liquid. This is the record transport and the pressure embedding both read, in the canvas frame the picture is drawn in — so a normal that disagrees with the line it sits on is a reconstruction fault, not a drawing one. A cut cell the reconstruction gave no normal at all is ringed rather than left blank.",
    keys: [["output", "interface normal, out of the liquid"],
      ["alarm", "cut cell with no reconstruction"]],
    draw(c) {
      const { g, lattice, scale: S } = c;
      /* The surface the arrows are normal to, drawn with them. An arrow with no
       * line under it is a direction attached to nothing, and this overlay is
       * about the relationship between the two. */
      drawInterface(c, PALETTE.output, 1.6);
      g.fillStyle = PALETTE.output;
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= 1e-3 || cell.fill >= 1 - 1e-3) continue;
        const x = cell.x0 * S, y = cell.y0 * S;
        const w = cell.width * S, h = cell.height * S;
        const plane = latticePlane(lattice, cell);
        if (!plane) {
          /* Cut, and unreconstructed. `drawSlice` falls back to holding the
           * liquid at the bottom of such a cell, which looks like an answer;
           * this is the mark that says it was not one. */
          g.strokeStyle = PALETTE.alarm;
          g.lineWidth = 1.4;
          g.beginPath();
          g.arc(x + w / 2, y + h / 2, Math.max(2.5, Math.min(w, h) * 0.17), 0, Math.PI * 2);
          g.stroke();
          continue;
        }
        const magnitude = Math.hypot(plane.nx, plane.ny);
        if (magnitude < 1e-6) continue;
        const reach = Math.max(6, 0.40 * Math.min(w, h));
        g.strokeStyle = PALETTE.output;
        for (const [ax, ay, bx, by] of interfaceSegments(plane)) {
          const mx = x + 0.5 * (ax + bx) * w, my = y + 0.5 * (ay + by) * h;
          arrow(g, mx, my, plane.nx / magnitude * reach, plane.ny / magnitude * reach, 1.3);
          g.beginPath();
          g.arc(mx, my, 1.7, 0, Math.PI * 2);
          g.fill();
        }
      }
    },
  },
};

/* ---- one lens per stage -------------------------------------------- */

export const ADVANCE_LENSES: Readonly<Record<AdvanceStageId, Lens>> = {
  "transport-velocity-extension": {
    caption: "Eight packet sweeps push face velocity out of the liquid into the empty band, so transport has a defined velocity everywhere it might sweep. A ghost row is one no liquid cell touches.",
    keys: [["transport", "extended ghost row"], ["amber", "carried velocity"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.transport;
      g.globalAlpha = 0.85;
      g.lineWidth = 2.4;
      g.beginPath();
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        if (!s.ext[sliceRowX(s, x, y)]) continue;
        g.moveTo(x * S, y * S + 1.5);
        g.lineTo(x * S, (y + 1) * S - 1.5);
      }
      g.stroke();
      g.globalAlpha = 1;
      velocityField(c, 3, PALETTE.amber, 0.85);
    },
  },
  "face-preparation": {
    caption: "Every row is re-cut against the solids. The stored face velocity already folds in the aperture as u = a·u_fluid + (1−a)·u_wall — flux code must never multiply by a twice.",
    keys: [["solidEdge", "closed row · a = 0"], ["momentum", "partly open row"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.lineWidth = 2.6;
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        const aperture = Math.min(
          x > 0 ? s.K[sliceCell(s, x - 1, y)] : 0,
          x < s.nx ? s.K[sliceCell(s, x, y)] : 0);
        if (aperture > 0.98) continue;
        g.strokeStyle = aperture <= 0.05 ? PALETTE.solidEdge : PALETTE.momentum;
        g.globalAlpha = aperture <= 0.05 ? 0.8 : 0.95;
        g.beginPath();
        g.moveTo(x * S, y * S + 1);
        g.lineTo(x * S, (y + 1) * S - 1);
        g.stroke();
      }
      g.globalAlpha = 1;
    },
  },
  "body-forces": {
    caption: "Gravity lands on the rows, not the cells — one add per row that touches liquid. Nothing else in the advance writes velocity without being projected afterwards.",
    keys: [["momentum", "row taking g·dt"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.momentum;
      g.globalAlpha = 0.9;
      for (let y = 0; y < s.ny; y += 2) for (let x = 1; x < s.nx - 1; x += 2) {
        const above = sliceCell(s, x, Math.max(0, y - 1)), here = sliceCell(s, x, y);
        if (s.V[above] <= 1e-5 && s.V[here] <= 1e-5) continue;
        arrow(g, (x + 0.5) * S, y * S - 5, 0, 11, 1.3);
      }
      g.globalAlpha = 1;
    },
  },
  "pressure-topology": {
    caption: "The compact leaf set this solve runs on. The repair is incremental — seeded from the previous generation, walked over dirty worklists — but the classify pass is still a full accepted-cell scan.",
    keys: [["pressure", "pressure cell"], ["adaptivity", "2:1 port"]],
    draw(c) {
      const { g, s, lattice, scale: S } = c;
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= 0.5) continue;
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.pressure, 0.3, cell.height);
      }
      g.strokeStyle = PALETTE.adaptivity;
      g.lineWidth = 2.2;
      g.globalAlpha = 0.9;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx - 1; bx++) {
        if (s.rung[by * s.bx + bx] === s.rung[by * s.bx + bx + 1]) continue;
        g.beginPath();
        g.moveTo((bx + 1) * SLICE_BRICK * S, by * SLICE_BRICK * S);
        g.lineTo((bx + 1) * SLICE_BRICK * S, (by + 1) * SLICE_BRICK * S);
        g.stroke();
      }
      g.globalAlpha = 1;
    },
  },
  "pressure-rhs": {
    caption: "Divergence of the extended face field, one row per canonical incidence. Blue is compressing, red expanding; a converged solve drives every one of them to zero.",
    keys: [["pressure", "negative divergence"], ["alarm", "positive divergence"]],
    draw(c) {
      const { s, lattice } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.div.length; i++) peak = Math.max(peak, Math.abs(s.div[i]));
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= 0.5) continue;
        const mean = cellMean(s, cell, i => s.div[i]);
        tint(c, cell.x0, cell.y0, cell.width,
          mean < 0 ? PALETTE.pressure : PALETTE.alarm,
          Math.min(0.85, (Math.abs(mean) / peak) * 1.6), cell.height);
      }
    },
  },
  "pressure-solve": {
    caption: "The solved pressure. One reduction per iteration, a single positive Jacobi diagonal as the preconditioner, and a true-residual guard every eighth iteration — the tail stays encoded whether or not it has converged.",
    keys: [["pressure", "high pressure"], ["ground", "free surface · p = 0"]],
    draw(c) {
      const { s, lattice } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.p.length; i++) peak = Math.max(peak, s.p[i]);
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= 0.5) continue;
        const mean = cellMean(s, cell, i => s.p[i]);
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.pressure,
          Math.min(0.9, (Math.max(0, mean) / peak) * 0.95), cell.height);
      }
    },
  },
  "velocity-projection": {
    caption: "Grey is the field entering the projection, amber the divergence-free field leaving it. The difference is the pressure gradient, applied one row at a time.",
    keys: [["muted", "before projection"], ["amber", "after projection"]],
    draw(c) {
      velocityField(c, 3, PALETTE.muted, 0.55, true);
      velocityField(c, 3, PALETTE.amber, 0.95, false);
    },
  },
  "conservative-transport": {
    caption: "Volume moves as swept prisms cut from the PLIC polygon and handed across one shared subface. Each arrow is a paired debit and credit; a marked row is one the bounded limiter had to cut back.",
    keys: [["transport", "swept flux"], ["alarm", "limiter clipped"]],
    draw(c) {
      const { g, s, scale: S } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.fx.length; i++) peak = Math.max(peak, Math.abs(s.fx[i]));
      for (let i = 0; i < s.fy.length; i++) peak = Math.max(peak, Math.abs(s.fy[i]));
      g.strokeStyle = PALETTE.transport;
      g.fillStyle = PALETTE.transport;
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        const flux = s.fx[sliceRowX(s, x, y)];
        if (Math.abs(flux) < peak * 0.05) continue;
        g.globalAlpha = Math.min(1, 0.3 + Math.abs(flux) / peak);
        const width = Math.max(2, Math.abs(flux) * S * 2.2);
        g.fillRect(x * S - (flux > 0 ? width : 0), y * S + 2, width, S - 4);
        arrow(g, x * S - (flux > 0 ? 4 : -4), (y + 0.5) * S, Math.sign(flux) * 9, 0);
      }
      for (let y = 0; y <= s.ny; y++) for (let x = 0; x < s.nx; x++) {
        const flux = s.fy[sliceRowY(s, x, y)];
        if (Math.abs(flux) < peak * 0.05) continue;
        g.globalAlpha = Math.min(1, 0.3 + Math.abs(flux) / peak);
        arrow(g, (x + 0.5) * S, y * S - (flux > 0 ? 4 : -4), 0, Math.sign(flux) * 9);
      }
      g.globalAlpha = 1;
      g.strokeStyle = PALETTE.alarm;
      g.lineWidth = 2.4;
      g.beginPath();
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        if (!s.cx[sliceRowX(s, x, y)]) continue;
        g.moveTo(x * S, y * S + 1);
        g.lineTo(x * S, (y + 1) * S - 1);
      }
      g.stroke();
    },
  },
  "tracer-advection": {
    caption: "Markers ride the same published transport velocity the volume does. They carry no mass — they exist so a colour or an age can be read back out of the flow.",
    keys: [["adaptivity", "marker"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.fillStyle = PALETTE.adaptivity;
      g.globalAlpha = 0.9;
      for (const marker of s.markers) {
        g.beginPath();
        g.arc(marker.x * S, marker.y * S, 2.2, 0, Math.PI * 2);
        g.fill();
      }
      g.globalAlpha = 1;
    },
  },
  "scalar-publication": {
    caption: "What this advance actually changed. Only these cells enter the dirty worklists the adaptivity band walks — everything unlit is carried forward untouched.",
    keys: [["output", "volume changed"]],
    draw(c) {
      const { s, lattice } = c;
      for (const cell of lattice.cells) {
        if (!cell.open) continue;
        let changed = 0;
        for (let j = 0; j < cell.height; j++) for (let i = 0; i < cell.width; i++) {
          const index = sliceCell(s, cell.x0 + i, cell.y0 + j);
          changed += Math.abs(s.V[index] - s.Vp[index]);
        }
        if (changed < 1e-4) continue;
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.output,
          Math.min(0.8, 0.18 + changed * 3), cell.height);
      }
    },
  },
  "activity-measurement": {
    caption: "One score per brick, from interface presence and peak speed. This is the only number the resolution policy reads — geometry and motion, never an authored region.",
    keys: [["adaptivity", "high activity"]],
    draw(c) {
      const { s, scale: S } = c;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const score = s.activity[by * s.bx + bx];
        if (score <= 0.001) continue;
        tint(c, bx * SLICE_BRICK, by * SLICE_BRICK, SLICE_BRICK,
          PALETTE.adaptivity, Math.min(0.7, score * 0.8));
        label(c, (bx + 0.5) * SLICE_BRICK * S, (by + 0.5) * SLICE_BRICK * S,
          score.toFixed(2));
      }
    },
  },
  "resolution-planning": {
    caption: "The activity score becomes a target rung on the dyadic ladder — 1, 2, 4 or 8 cells per brick edge — then 2:1 grading pulls in any neighbour sitting more than one rung away.",
    keys: [["adaptivity", "target rung"]],
    draw(c) {
      const { s, scale: S } = c;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const rung = s.rung[by * s.bx + bx];
        tint(c, bx * SLICE_BRICK, by * SLICE_BRICK, SLICE_BRICK,
          PALETTE.adaptivity, 0.08 + 0.13 * rung);
        label(c, (bx + 0.5) * SLICE_BRICK * S, (by + 0.5) * SLICE_BRICK * S,
          `${SLICE_RUNGS[rung]}²`);
      }
    },
  },
  "candidate-transfer": {
    caption: "Bricks whose rung moved this advance. The shadow topology is built beside the live one and committed as a single transaction at the frame tail — so this flip is the next advance's input, never this one's.",
    keys: [["output", "rung changed"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.lineWidth = 3;
      g.strokeStyle = PALETTE.output;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const brick = by * s.bx + bx;
        if (s.rung[brick] === s.rungWas[brick]) continue;
        tint(c, bx * SLICE_BRICK, by * SLICE_BRICK, SLICE_BRICK, PALETTE.output, 0.3);
        g.strokeRect(bx * SLICE_BRICK * S + 2, by * SLICE_BRICK * S + 2,
          SLICE_BRICK * S - 4, SLICE_BRICK * S - 4);
        label(c, (bx + 0.5) * SLICE_BRICK * S, (by + 0.5) * SLICE_BRICK * S,
          `${SLICE_RUNGS[s.rungWas[brick]!]} → ${SLICE_RUNGS[s.rung[brick]!]}`);
      }
    },
  },
  "brick-retirement": {
    caption: "A brick holding no liquid and no source is released back to the atlas. The hatched bricks pay nothing this frame — the sparse set is the lit region plus its band, and no more.",
    keys: [["muted", "retired brick"], ["liquid", "resident brick"]],
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.muted;
      g.globalAlpha = 0.45;
      g.lineWidth = 1;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        if (brickVolume(s, bx, by) > 1e-3) continue;
        g.save();
        g.beginPath();
        g.rect(bx * SLICE_BRICK * S, by * SLICE_BRICK * S,
          SLICE_BRICK * S, SLICE_BRICK * S);
        g.clip();
        g.beginPath();
        for (let d = -SLICE_BRICK; d < SLICE_BRICK; d += 1.6) {
          g.moveTo((bx * SLICE_BRICK + d) * S, by * SLICE_BRICK * S);
          g.lineTo((bx * SLICE_BRICK + d + SLICE_BRICK) * S, (by + 1) * SLICE_BRICK * S);
        }
        g.stroke();
        g.restore();
      }
      g.globalAlpha = 1;
    },
  },
  "presentation-publication": {
    caption: "The surface the renderer receives: the PLIC segments of every interface cell, stitched across brick boundaries at whatever rung each brick happens to be carrying.",
    keys: [["output", "published interface"]],
    draw(c) { drawInterface(c, PALETTE.output, 2.6); },
  },
};

/**
 * Step 1 of the loop is not a stage — it is the state the advance starts from,
 * so it gets a lens of its own rather than borrowing one.
 */
export const REPRESENT_LENS: Lens = {
  caption: "Before anything moves: a sparse set of bricks, each carrying its own rung on the dyadic ladder, and inside them the liquid volume held per cell with an exact PLIC line wherever a cell is cut. Nothing here is a level set — the conserved quantity is volume.",
  keys: [["adaptivity", "brick rung"], ["output", "reconstructed interface"]],
  draw(c) {
    const { s, scale: S } = c;
    for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
      if (brickVolume(s, bx, by) <= 1e-3) continue;
      label(c, (bx + 0.5) * SLICE_BRICK * S, by * SLICE_BRICK * S + 9,
        `${SLICE_RUNGS[s.rung[by * s.bx + bx]!]}²`, PALETTE.muted);
    }
    drawInterface(c, PALETTE.output, 2.6);
  },
};
