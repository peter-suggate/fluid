/**
 * The slice, drawn, and one lens per stage of the advance.
 *
 * Every lens paints onto the same picture — the same water, the same bricks,
 * the same rungs — so switching stages moves the reading rather than the
 * subject. That is the whole design: a stage list that changes what you can
 * see about one scene, not fifteen separate diagrams.
 *
 * What a lens *claims* is not in this file. The caption a stage reads under,
 * the marks its picture may put on a cell, what carrying one of them says and
 * the numbers each is cut at are all declarations about the encoder, and they
 * live beside it, in `lib/methods/adaptive-volume/features/advance-slice/` —
 * the per-stage half on `SPARSE_CM12_STAGES[stage].slice`, the thresholds as
 * `ADVANCE_SLICE_THRESHOLDS`. What stays here is the drawing: the palette the
 * page resolves per theme, the canvas primitives, one `draw(c)` per stage, and
 * the predicate that answers each declared mark for the cell under the pointer.
 * `ADVANCE_LENSES` is the two joined, which is why a mark declared with no
 * predicate is a failure rather than a blank row in the probe.
 */
import {
  ADVANCE_BRICK_FINE, ADVANCE_RUNGS, advanceCell, advanceCellAt, advanceCellPlane,
  advanceRdfTriangles, advanceRowX, advanceRowY, clipUnitSquare, UNIT_SQUARE,
  type AdvanceCellView, type AdvanceLattice, type AdvancePlane, type AdvanceRdfVertex,
  type AdvanceRdfView, type AdvanceView,
} from "../lib/physics-wasm/advance-view";
import {
  FRACTION_FLOOR, cellFillOpacity, fractionBand, fractionBandPaint, fractionReadout,
  fractionResidueRamp,
  type FractionBand, type FractionTone,
} from "../lib/core/fluid-fraction-view";
import {
  ADVANCE_STAGE_ORDER, type AdvanceStageId,
} from "../lib/methods/adaptive-volume/features/advance-slice/advance-work";
import {
  ADVANCE_REPRESENT_SLICE, ADVANCE_SLICE_THRESHOLDS,
  type AdvanceSliceDeclaration, type AdvanceSliceKey, type AdvanceSliceTone,
} from "../lib/methods/adaptive-volume/features/advance-slice/definition";
import {
  advanceStageSlice,
} from "../lib/methods/adaptive-volume/features/advance-slice/loop";

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
/**
 * The tone a mark or a band is drawn in.
 *
 * The roles are the method's — a declaration beside the encoder names one, and
 * this file is where a name becomes a colour — so the union is imported rather
 * than restated. Adding a role is a change to the declaration vocabulary, and
 * `PALETTE_VAR` below is what makes an unpainted one a type error.
 */
export type PaletteTone = AdvanceSliceTone;

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

/**
 * The shared fraction view's colour roles, in the lab's own tones.
 *
 * `lib/core/fluid-fraction-view` states which band a cell's V/K is in and what
 * that band is called; it also carries a literal colour, for a legend or a
 * shader. The canvas here cannot use that literal — the lab paints tones the
 * page resolves per theme — so this is the one table where the shared roles
 * become lab tones. One table, so a role that gains a renderer does not gain a
 * second opinion about what colour it is.
 */
const FRACTION_TONE: Readonly<Record<FractionTone, PaletteTone>> = {
  empty: "ground",
  residue: "transport",
  liquid: "liquid",
  excess: "alarm",
};

/** One shared fraction role, resolved to the colour the canvas can hold. */
const fractionInk = (tone: FractionTone): string => PALETTE[FRACTION_TONE[tone]];

/** Which tone stands for each band of the advance. */
export const BAND_TONE = {
  transport: "transport",
  momentum: "momentum",
  pressure: "pressure",
  adaptivity: "adaptivity",
  output: "output",
} as const satisfies Record<string, PaletteTone>;

/**
 * One mark the picture can put on a cell: the tone it is drawn in, the name it
 * goes by, what carrying it says about that cell, and how a cell earns it.
 *
 * A strip of these along the bottom of the picture states what *could* be
 * drawn — all of it, all the time — and leaves the reader to match a colour by
 * eye. `holds` is what turns the same declaration into an answer about the one
 * cell under the pointer, which is the question a reader actually has. Both
 * readings come out of this single record, so the probe cannot drift from the
 * ink on the canvas.
 */
export interface LensKey extends AdvanceSliceKey {
  /** True when the cell the query names carries this mark. */
  readonly holds: (q: MarkQuery) => boolean;
}

const key = (id: string, tone: PaletteTone, label: string, note: string,
  holds: (q: MarkQuery) => boolean): LensKey => ({ id, tone, label, note, holds });

/**
 * A key for one band of the shared fraction view.
 *
 * The name and the line the probe reads out come from
 * `lib/core/fluid-fraction-view`, so what the lab calls a band and what the
 * 3-D overlay's legend calls it cannot drift apart; only the tone is the lab's,
 * because only the lab has a theme.
 */
const fractionKey = (band: FractionBand,
  holds: (q: MarkQuery) => boolean): LensKey => {
  const paint = fractionBandPaint(band);
  return key(`fraction-${band}`, FRACTION_TONE[paint.tone], paint.label, paint.note, holds);
};

/**
 * What the pointer is over, for a mark to answer about.
 *
 * The lenses draw against three different addressings — a drawn block, a
 * finest cell, a brick — so the query carries all three rather than making
 * every predicate re-derive the two it was not handed.
 */
export interface MarkQuery {
  readonly s: AdvanceView;
  readonly lattice: AdvanceLattice;
  /** The drawn block the pointer is inside, at its brick's rung. */
  readonly cell: AdvanceCellView;
  /** The finest cell under the pointer, inside that block, in canvas terms. */
  readonly fx: number;
  readonly fy: number;
  /** The brick holding it, indexed the way every lens indexes bricks. */
  readonly brick: number;
  /** Peak |value| over a field — the same normaliser the lens draws against. */
  readonly peakOf: (field: Float32Array) => number;
}

export function markQuery(s: AdvanceView, cell: AdvanceCellView,
  fx: number, fy: number): MarkQuery {
  /* One scan per field per probe rather than one per mark: the divergence and
   * flux lenses normalise against a peak, and the flux lens asks for two. */
  const peaks = new Map<Float32Array, number>();
  return {
    s, lattice: s.lattice, cell, fx, fy,
    brick: Math.floor(fy / ADVANCE_BRICK_FINE) * s.bx + Math.floor(fx / ADVANCE_BRICK_FINE),
    peakOf(field) {
      const known = peaks.get(field);
      if (known !== undefined) return known;
      let peak = 1e-6;
      for (let i = 0; i < field.length; i += 1) peak = Math.max(peak, Math.abs(field[i]!));
      peaks.set(field, peak);
      return peak;
    },
  };
}

/**
 * The numbers every predicate below is cut at, named once beside the method.
 *
 * Destructured rather than reached through, because a threshold read inline
 * reads as a magic number in the drawing; these are claims about the solver,
 * and the picture, the probe and the overlays all have to be cut at the same
 * ones or they are three different readings of one cell.
 */
const {
  interfaceFill, pressureFill, velocityCapacity, velocityVolume,
  closedAperture, openAperture, residentBrickVolume, activeBrickScore,
  changedVolume, fluxShare, solidCapacity,
  rdfFullCapacity, rdfPartialFill,
  rdfMinorityAreaTolerance, rdfMinorityAmplification,
} = ADVANCE_SLICE_THRESHOLDS;

/** The aperture of a vertical row, cut exactly as `face-preparation` cuts it. */
const rowAperture = (s: AdvanceView, x: number, y: number): number => Math.min(
  x > 0 ? s.capacityFine[advanceCell(s, x - 1, y)]! : 0,
  x < s.nx ? s.capacityFine[advanceCell(s, x, y)]! : 0);

/** True when either vertical row bounding the probed cell answers the test. */
const eitherRow = (q: MarkQuery, test: (aperture: number) => boolean): boolean =>
  test(rowAperture(q.s, q.fx, q.fy)) || test(rowAperture(q.s, q.fx + 1, q.fy));

/** The rows bounding the probed cell, as indices into the staggered planes. */
const boundingRowsX = (q: MarkQuery): readonly number[] =>
  [advanceRowX(q.s, q.fx, q.fy), advanceRowX(q.s, q.fx + 1, q.fy)];
const boundingRowsY = (q: MarkQuery): readonly number[] =>
  [advanceRowY(q.s, q.fx, q.fy), advanceRowY(q.s, q.fx, q.fy + 1)];

/** `velocityField` draws an arrow only where there is liquid to carry one. */
const carriesVelocity = (q: MarkQuery): boolean => {
  const i = advanceCell(q.s, q.fx, q.fy);
  return q.s.capacityFine[i]! > velocityCapacity
    && q.s.liquidVolumeFine[i]! > velocityVolume;
};

/** Cut: the interface passes through this cell, so it carries a PLIC plane. */
const cutCell = (q: MarkQuery): boolean =>
  q.cell.open && q.cell.fill > interfaceFill && q.cell.fill < 1 - interfaceFill;

/** The half-full test every pressure lens tints against. */
const pressureCell = (q: MarkQuery): boolean =>
  q.cell.open && q.cell.fill > pressureFill;

export interface Lens {
  /** What the reader is looking at, in the lens's own terms. */
  readonly caption: string;
  readonly keys: readonly LensKey[];
  readonly draw: (c: LensContext) => void;
}

export interface LensContext {
  readonly g: CanvasRenderingContext2D;
  readonly s: AdvanceView;
  readonly lattice: AdvanceLattice;
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
    const closed = 1 - s.capacityFine[advanceCell(s, x, y)]!;
    if (closed <= 1e-4) continue;
    g.globalAlpha = Math.max(0.18, closed);
    g.fillStyle = PALETTE.solid;
    g.fillRect(x * S, y * S, S, S);
  }
  g.globalAlpha = 1;
}

function clippedScalarTriangle(points: readonly AdvanceRdfVertex[]): number[] {
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

export function rdfMinorityAreaDistorted(acceptedLiquid: number,
  representedLiquid: number, area: number): boolean {
  const accepted = acceptedLiquid <= area / 2 ? acceptedLiquid : area - acceptedLiquid;
  const represented = acceptedLiquid <= area / 2 ? representedLiquid : area - representedLiquid;
  const absoluteError = Math.abs(represented - accepted);
  if (!(absoluteError > rdfMinorityAreaTolerance * area)) return false;
  const smaller = Math.min(accepted, represented), larger = Math.max(accepted, represented);
  return smaller <= 0 ? larger > 0 : larger > rdfMinorityAmplification * smaller;
}

interface RdfDisplaySource { readonly nx: number; readonly ny: number }
interface RdfDisplaySurface { readonly vertexPhiFine: Float32Array }

export function sliceRdfPlicFallbackCells(s: RdfDisplaySource, lattice: AdvanceLattice,
  sharedRdf: RdfDisplaySurface): ReadonlySet<number> {
  const result = new Set<number>(), stride = s.nx + 1, phi = sharedRdf.vertexPhiFine;
  for (const cell of lattice.cells) {
    if (!cell.open || cell.capacity < rdfFullCapacity * cell.width * cell.height
      || cell.fill <= rdfPartialFill || cell.fill >= 1 - rdfPartialFill) continue;
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
export function inspectSliceRdfDisplay(s: RdfDisplaySource, lattice: AdvanceLattice,
  sharedRdf: RdfDisplaySurface): readonly SliceRdfDisplayCell[] {
  const fallback = sliceRdfPlicFallbackCells(s, lattice, sharedRdf);
  const stride = s.nx + 1, phi = sharedRdf.vertexPhiFine;
  return lattice.cells
    .filter(cell => cell.open && cell.fill > rdfPartialFill && cell.fill < 1 - rdfPartialFill)
    .map(cell => {
      const cut = cell.capacity < rdfFullCapacity * cell.width * cell.height;
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
export function drawSlice(c: LensContext, sharedRdf?: AdvanceRdfView): void {
  const { g, s, lattice, scale: S } = c;
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
      const dense = advanceCell(s, x, canvasY), capacity = s.capacityFine[dense]!;
      if (capacity < rdfFullCapacity) continue;
      const acceptedFill = s.liquidVolumeFine[dense]! / Math.max(capacity, 1e-8);
      // A pure accepted owner is stronger evidence than a render-only RDF.
      // Publishing it directly prevents a shared-vertex fit from carving an
      // enclosed opposite-phase cell out of homogeneous bulk. Mixed owners
      // retain the shared RDF, including genuine subcell sheets and droplets.
      if (acceptedFill >= 1 - rdfPartialFill) {
        g.rect(x * S, canvasY * S, S, S);
        continue;
      }
      if (acceptedFill <= rdfPartialFill) continue;
      const owner = advanceCellAt(lattice, s, x + 0.5, canvasY + 0.5);
      if (owner && plicFallback.has(owner.topologyCell)) continue;
      const a = phi[x + stride * y]!;
      const b = phi[x + 1 + stride * y]!;
      const d = phi[x + stride * (y + 1)]!;
      const e = phi[x + 1 + stride * (y + 1)]!;
      if (![a, b, d, e].every(Number.isFinite)) continue;
      for (const triangle of advanceRdfTriangles(x, canvasY, d, e, b, a))
        appendPolygon(g, clippedScalarTriangle(triangle), S);
    }
  }
  for (const cell of lattice.cells) {
    if (!cell.open || cell.fill <= interfaceFill) continue;
    if (sharedRdf && cell.capacity >= rdfFullCapacity * cell.width * cell.height
      && !plicFallback.has(cell.topologyCell)) continue;
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    if (cell.fill >= 1 - rdfPartialFill) { g.rect(x, y, w, h); continue; }
    const plane = plicFallback.has(cell.topologyCell) ? cell.plane : advanceCellPlane(lattice, cell);
    /* No published plane means the solver could not resolve this interface, so
     * the picture falls back to the monotone reading the solver itself falls
     * back to: the liquid held at the bottom of the cell. Gravity is +y here —
     * filling from the top edge down would draw every unresolved cell upside
     * down, and a row of them reads as a sheet of water floating over a gap. */
    if (!plane) { g.rect(x, y + h * (1 - cell.fill), w, h * cell.fill); continue; }
    const polygon = clipUnitSquare(UNIT_SQUARE, plane.clipNx, plane.clipNy, plane.offset);
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

  drawLatticeAndSolids(c);
}

function drawLatticeAndSolids(c: LensContext): void {
  const { g, s, lattice, scale: S } = c;
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
    g.globalAlpha = 0.3 + 0.17 * s.brickRung[by * s.bx + bx]!;
    g.strokeRect(bx * ADVANCE_BRICK_FINE * S, by * ADVANCE_BRICK_FINE * S,
      Math.min(ADVANCE_BRICK_FINE, s.nx - bx * ADVANCE_BRICK_FINE) * S,
      Math.min(ADVANCE_BRICK_FINE, s.ny - by * ADVANCE_BRICK_FINE) * S);
  }
  g.globalAlpha = 1;
  drawSolidRaster(c);
}

/** The liquid the base picture cuts into a cell, under every lens but transport. */
export const LIQUID_KEY: LensKey = key("liquid", "liquid", "liquid",
  "the published surface geometry places liquid in this cell",
  q => q.cell.open && q.cell.fill > interfaceFill);

const directPhiCorners = (q: MarkQuery): readonly number[] => {
  const y = q.s.ny - 1 - q.fy, stride = q.s.nx + 1;
  const phi = q.s.rdf.vertexPhiFine;
  return [phi[q.fx + stride * y]!, phi[q.fx + 1 + stride * y]!,
    phi[q.fx + 1 + stride * (y + 1)]!, phi[q.fx + stride * (y + 1)]!];
};

export const DIRECT_LEVEL_SET_KEY: LensKey = key("direct-level-set-liquid",
  "liquid", "direct level-set liquid",
  "the advected signed-distance field is negative in part of this finest cell",
  q => directPhiCorners(q).some(value => Number.isFinite(value) && value < 0));

export const DIRECT_LEVEL_SET_CONTOUR_KEY: LensKey = key("direct-zero-set",
  "output", "direct zero set",
  "the published signed-distance zero set crosses this finest cell",
  q => {
    const corners = directPhiCorners(q);
    return corners.every(Number.isFinite) && corners.some(value => value < 0)
      && corners.some(value => value >= 0);
  });

/** Solid, which every lens draws and no lens owns. */
export const SOLID_KEY: LensKey = key("solid", "solid", "solid",
  "solid takes part of this cell; K is the open capacity left to the water",
  q => q.s.capacityFine[advanceCell(q.s, q.fx, q.fy)]! < 1 - solidCapacity);

export const CELL_FILL_KEYS: readonly LensKey[] = [
  key("cell-fill", "liquid", "cell fill opacity · clamp(V/K, 0, 1)",
    "the blue wash is V/K itself, clamped — opacity, not geometry",
    q => q.cell.open && cellFillOpacity(q.cell.fill) > 0),
  key("accepted-surface-contour", "output", "accepted surface contour",
    "the published surface passes through this cell",
    q => q.cell.open && fractionBand(q.cell.fill) !== "vacuum"
      && q.cell.fill < 1 - FRACTION_FLOOR),
  key("over-capacity-cell", "amber", "over-capacity cell · V/K > 1",
    "V is past the open capacity K, and the projection has to drain it",
    q => q.cell.open && fractionBand(q.cell.fill) === "overfull"),
];

/** The transport stage defaults to the authoritative cell-fill reading. */
export function usesCellFillSlice(stage: AdvanceStageId, representing: boolean): boolean {
  return !representing && stage === "conservative-transport";
}

/** Draw authoritative V/K over complete cells, with the surface as reference. */
export function drawCellFillSlice(c: LensContext, sharedRdf?: AdvanceRdfView): void {
  const { g, s, lattice, scale: S } = c;
  g.clearRect(0, 0, s.nx * S, s.ny * S);
  g.globalAlpha = 1;
  g.fillStyle = PALETTE.ground;
  g.fillRect(0, 0, s.nx * S, s.ny * S);

  g.fillStyle = PALETTE.liquid;
  for (const cell of lattice.cells) {
    if (!cell.open) continue;
    const alpha = cellFillOpacity(cell.fill);
    if (alpha <= 0) continue;
    g.globalAlpha = alpha;
    g.fillRect(cell.x0 * S, cell.y0 * S, cell.width * S, cell.height * S);
  }
  g.globalAlpha = 1;

  if (sharedRdf) {
    g.strokeStyle = PALETTE.output;
    g.lineWidth = 1.25;
    g.globalAlpha = 0.9;
    g.beginPath();
    for (let i = 0; i + 3 < sharedRdf.segmentsFine.length; i += 4) {
      g.moveTo(sharedRdf.segmentsFine[i]! * S, (s.ny - sharedRdf.segmentsFine[i + 1]!) * S);
      g.lineTo(sharedRdf.segmentsFine[i + 2]! * S,
        (s.ny - sharedRdf.segmentsFine[i + 3]!) * S);
    }
    g.stroke();
  } else {
    g.globalAlpha = 0.9;
    drawInterface(c, PALETTE.output, 1.25);
  }
  g.globalAlpha = 1;

  /* Saturated blue cannot distinguish V/K = 1 from V/K > 1. */
  g.strokeStyle = PALETTE.amber;
  g.lineWidth = 1.5;
  for (const cell of lattice.cells) {
    if (!cell.open || fractionBand(cell.fill) !== "overfull") continue;
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    g.globalAlpha = 0.95;
    g.strokeRect(x + 0.75, y + 0.75, Math.max(0, w - 1.5), Math.max(0, h - 1.5));
    g.save();
    g.beginPath();
    g.rect(x, y, w, h);
    g.clip();
    g.beginPath();
    const step = Math.max(4, Math.min(w, h) * 0.28);
    for (let d = -h; d < w; d += step) {
      g.moveTo(x + d, y + h);
      g.lineTo(x + d + h, y);
    }
    g.stroke();
    g.restore();
  }
  g.globalAlpha = 1;
  drawLatticeAndSolids(c);
}

/** Draw only the published signed-distance field and its zero set. */
export function drawDirectLevelSetSlice(c: LensContext, surface: AdvanceRdfView): void {
  const { g, s, scale: S } = c;
  g.clearRect(0, 0, s.nx * S, s.ny * S);
  g.globalAlpha = 1;
  g.fillStyle = PALETTE.ground;
  g.fillRect(0, 0, s.nx * S, s.ny * S);

  const stride = s.nx + 1, phi = surface.vertexPhiFine;
  g.fillStyle = PALETTE.liquid;
  g.globalAlpha = 0.9;
  g.beginPath();
  for (let y = 0; y < s.ny; y += 1) for (let x = 0; x < s.nx; x += 1) {
    const canvasY = s.ny - 1 - y;
    const a = phi[x + stride * y]!, b = phi[x + 1 + stride * y]!;
    const d = phi[x + stride * (y + 1)]!, e = phi[x + 1 + stride * (y + 1)]!;
    if (![a, b, d, e].every(Number.isFinite)) continue;
    for (const triangle of advanceRdfTriangles(x, canvasY, d, e, b, a))
      appendPolygon(g, clippedScalarTriangle(triangle), S);
  }
  g.fill();
  g.globalAlpha = 1;

  g.strokeStyle = PALETTE.output;
  g.lineWidth = 2.2;
  g.beginPath();
  for (let i = 0; i + 3 < surface.segmentsFine.length; i += 4) {
    g.moveTo(surface.segmentsFine[i]! * S, (s.ny - surface.segmentsFine[i + 1]!) * S);
    g.lineTo(surface.segmentsFine[i + 2]! * S,
      (s.ny - surface.segmentsFine[i + 3]!) * S);
  }
  g.stroke();
  drawLatticeAndSolids(c);
}

function velocityField(
  c: LensContext, stride: number, color: string, alpha: number, before = false,
): void {
  const { g, s, scale: S } = c;
  g.strokeStyle = color;
  g.globalAlpha = alpha;
  const u = before ? s.faceVelocityXBeforePressure : s.faceVelocityXFine;
  const v = before ? s.faceVelocityYBeforePressure : s.faceVelocityYFine;
  for (let y = 1; y < s.ny - 1; y += stride) {
    for (let x = 1; x < s.nx - 1; x += stride) {
      const i = advanceCell(s, x, y);
      if (s.capacityFine[i] <= velocityCapacity
        || s.liquidVolumeFine[i] <= velocityVolume) continue;
      const ux = 0.5 * (u[advanceRowX(s, x, y)]! + u[advanceRowX(s, x + 1, y)]!);
      const uy = 0.5 * (v[advanceRowY(s, x, y)]! + v[advanceRowY(s, x, y + 1)]!);
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
export function interfaceSegments(plane: AdvancePlane): readonly InterfaceSegment[] {
  const polygon = clipUnitSquare(UNIT_SQUARE, plane.clipNx, plane.clipNy, plane.offset);
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
    const plane = advanceCellPlane(lattice, cell);
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

const brickVolume = (s: AdvanceView, bx: number, by: number): number => {
  let total = 0;
  for (let j = 0; j < ADVANCE_BRICK_FINE; j++) for (let i = 0; i < ADVANCE_BRICK_FINE; i++) {
    const x = bx * ADVANCE_BRICK_FINE + i, y = by * ADVANCE_BRICK_FINE + j;
    if (x < s.nx && y < s.ny) total += s.liquidVolumeFine[advanceCell(s, x, y)]!;
  }
  return total;
};

/** The same sum, addressed the way a probe knows a brick: by index. */
const brickVolumeAt = (s: AdvanceView, brick: number): number =>
  brickVolume(s, brick % s.bx, Math.floor(brick / s.bx));

const cellMean = (
  s: AdvanceView, cell: AdvanceCellView, read: (index: number) => number,
): number => {
  let total = 0, count = 0;
  for (let j = 0; j < cell.height; j++) for (let i = 0; i < cell.width; i++) {
    total += read(advanceCell(s, cell.x0 + i, cell.y0 + j));
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
  /** One line, for the toggle's own tooltip. */
  readonly hint: string;
  /** The whole of what it claims, for the sidebar, while it is on. */
  readonly caption: string;
  readonly keys: readonly LensKey[];
  readonly draw: (c: LensContext) => void;
}

/** Declaration order, which is also draw order: washes first, lines over them. */
export const SLICE_OVERLAY_ORDER = ["fraction", "normal"] as const;

/** Roughly the pixels `label` needs for a readout, at its 10px monospace. */
const readoutPixels = (text: string): number => text.length * 6 + 5;

export const SLICE_OVERLAYS: Readonly<Record<SliceOverlayId, SliceOverlay>> = {
  fraction: {
    label: "fraction",
    hint: "Write V/K into every cell that has room for it, and tint the dilute decades the water's own outline cannot show",
    caption: "V/K per accepted cell — the conserved quantity itself, read off the compact record rather than resampled. The water already draws the liquid half of the range geometrically, so the tint is spent where the geometry cannot help: the dilute decades below a half, which a cut line renders as a sliver too thin to see, and the overfull cells past V = K that the projection has to drain. The value is written into any cell with the pixels to hold it, so on a fine scene at a low zoom the tint is the whole reading and the numbers arrive as the cells grow.",
    keys: [
      /* Not a band: every cell with any liquid in it gets the number, and the
       * number is written in ink. Its note is the liquid band's, because the
       * quantity it states is the same one. */
      key("fraction-readout", "ink", "V/K", fractionBandPaint("liquid").note,
        q => q.cell.open && fractionBand(q.cell.fill) !== "vacuum"),
      fractionKey("dilute", q => q.cell.open && fractionBand(q.cell.fill) === "dilute"),
      fractionKey("overfull", q => q.cell.open && fractionBand(q.cell.fill) === "overfull"),
    ],
    draw(c) {
      const { g, lattice, scale: S } = c;
      for (const cell of lattice.cells) {
        const band = fractionBand(cell.fill);
        if (!cell.open || band === "vacuum") continue;
        const x = cell.x0 * S, y = cell.y0 * S;
        const w = cell.width * S, h = cell.height * S;
        /* The liquid band gets no wash. Between a half and a full cell the
         * picture underneath is already the answer — a PLIC polygon covering
         * that share of the cell — and tinting it would only dim the one part
         * of this field the reader can already measure by eye. */
        if (band === "overfull") {
          g.fillStyle = fractionInk("excess");
          g.globalAlpha = 0.42;
          g.fillRect(x, y, w, h);
          g.globalAlpha = 1;
        } else if (band === "dilute") {
          g.fillStyle = fractionInk("residue");
          g.globalAlpha = 0.14 + 0.40 * fractionResidueRamp(cell.fill);
          g.fillRect(x, y, w, h);
          g.globalAlpha = 1;
        }
        const text = fractionReadout(cell.fill);
        if (Math.min(w, h) < readoutPixels(text)) continue;
        label(c, x + w / 2, y + h / 2, text,
          band === "overfull" ? fractionInk("excess") : PALETTE.ink);
      }
    },
  },
  normal: {
    label: "normals",
    hint: "Draw the PLIC normal each cut cell carries, from its own interface line and out of the liquid",
    caption: "The PLIC normal each cut cell carries, drawn from the middle of its own interface chord and pointing out of the liquid. This is the record transport and the pressure embedding both read, in the canvas frame the picture is drawn in — so a normal that disagrees with the line it sits on is a reconstruction fault, not a drawing one. A cut cell the reconstruction gave no normal at all is ringed rather than left blank.",
    keys: [
      key("interface-normal", "output", "interface normal, out of the liquid",
        "the PLIC normal this cut cell carries, drawn from the middle of its own chord",
        q => cutCell(q) && advanceCellPlane(q.lattice, q.cell) !== null),
      key("unreconstructed-cut-cell", "alarm", "cut cell with no reconstruction",
        "cut, and given no normal — the liquid drawn here is the solver's fallback",
        q => cutCell(q) && advanceCellPlane(q.lattice, q.cell) === null),
    ],
    draw(c) {
      const { g, lattice, scale: S } = c;
      /* The surface the arrows are normal to, drawn with them. An arrow with no
       * line under it is a direction attached to nothing, and this overlay is
       * about the relationship between the two. */
      drawInterface(c, PALETTE.output, 1.6);
      g.fillStyle = PALETTE.output;
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= interfaceFill
          || cell.fill >= 1 - interfaceFill) continue;
        const x = cell.x0 * S, y = cell.y0 * S;
        const w = cell.width * S, h = cell.height * S;
        const plane = advanceCellPlane(lattice, cell);
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

/* ---- how each stage's lens is drawn --------------------------------- */

/**
 * The drawing half of every stage lens, and nothing else.
 *
 * What each mark is *called*, what it *means* and what colour role it takes
 * are declared beside the method, in `SPARSE_CM12_STAGES[stage].slice`. This
 * table answers the other half: how the stage is painted, and — per declared
 * mark id — the one predicate that decides whether the cell under the pointer
 * carries it. The two are joined below, so a mark declared with no predicate,
 * or a predicate answering for a mark nobody declared, fails at the join
 * rather than becoming a silent gap in the probe.
 */
interface StageDrawing {
  readonly draw: (c: LensContext) => void;
  /** One predicate per declared mark id, cut at the shared thresholds. */
  readonly holds: Readonly<Record<string, (q: MarkQuery) => boolean>>;
}

const ADVANCE_STAGE_DRAWING = {
  "transport-velocity-extension": {
    holds: {
      "extended-ghost-row": q => q.s.extensionFine[advanceCell(q.s, q.fx, q.fy)] !== 0,
      "carried-velocity": carriesVelocity,
    },
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.transport;
      g.globalAlpha = 0.85;
      g.lineWidth = 2.4;
      g.beginPath();
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        if (!s.extensionFine[advanceCell(s, Math.min(x, s.nx - 1), y)]) continue;
        g.moveTo(x * S, y * S + 1.5);
        g.lineTo(x * S, (y + 1) * S - 1.5);
      }
      g.stroke();
      g.globalAlpha = 1;
      velocityField(c, 3, PALETTE.amber, 0.85);
    },
  },
  "face-preparation": {
    holds: {
      "closed-row": q => eitherRow(q, aperture => aperture <= closedAperture),
      "partly-open-row": q =>
        eitherRow(q, aperture => aperture > closedAperture && aperture <= openAperture),
    },
    draw(c) {
      const { g, s, scale: S } = c;
      g.lineWidth = 2.6;
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        const aperture = Math.min(
          x > 0 ? s.capacityFine[advanceCell(s, x - 1, y)] : 0,
          x < s.nx ? s.capacityFine[advanceCell(s, x, y)] : 0);
        if (aperture > openAperture) continue;
        const shut = aperture <= closedAperture;
        g.strokeStyle = shut ? PALETTE.solidEdge : PALETTE.momentum;
        g.globalAlpha = shut ? 0.8 : 0.95;
        g.beginPath();
        g.moveTo(x * S, y * S + 1);
        g.lineTo(x * S, (y + 1) * S - 1);
        g.stroke();
      }
      g.globalAlpha = 1;
    },
  },
  "body-forces": {
    holds: {
      "row-taking-gravity": q =>
        q.s.liquidVolumeFine[advanceCell(q.s, q.fx, q.fy)]! > velocityVolume
        || q.s.liquidVolumeFine[advanceCell(q.s, q.fx, Math.max(0, q.fy - 1))]! > velocityVolume,
    },
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.momentum;
      g.globalAlpha = 0.9;
      for (let y = 0; y < s.ny; y += 2) for (let x = 1; x < s.nx - 1; x += 2) {
        const above = advanceCell(s, x, Math.max(0, y - 1)), here = advanceCell(s, x, y);
        if (s.liquidVolumeFine[above] <= velocityVolume
          && s.liquidVolumeFine[here] <= velocityVolume) continue;
        arrow(g, (x + 0.5) * S, y * S - 5, 0, 11, 1.3);
      }
      g.globalAlpha = 1;
    },
  },
  "pressure-topology": {
    holds: {
      "pressure-cell": pressureCell,
      "two-to-one-port": q => {
        const column = q.brick % q.s.bx, rung = q.s.brickRung[q.brick];
        return (column > 0 && q.s.brickRung[q.brick - 1] !== rung)
          || (column + 1 < q.s.bx && q.s.brickRung[q.brick + 1] !== rung);
      },
    },
    draw(c) {
      const { g, s, lattice, scale: S } = c;
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= pressureFill) continue;
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.pressure, 0.3, cell.height);
      }
      g.strokeStyle = PALETTE.adaptivity;
      g.lineWidth = 2.2;
      g.globalAlpha = 0.9;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx - 1; bx++) {
        if (s.brickRung[by * s.bx + bx] === s.brickRung[by * s.bx + bx + 1]) continue;
        g.beginPath();
        g.moveTo((bx + 1) * ADVANCE_BRICK_FINE * S, by * ADVANCE_BRICK_FINE * S);
        g.lineTo((bx + 1) * ADVANCE_BRICK_FINE * S, (by + 1) * ADVANCE_BRICK_FINE * S);
        g.stroke();
      }
      g.globalAlpha = 1;
    },
  },
  "pressure-rhs": {
    holds: {
      "negative-divergence": q => pressureCell(q) && cellMean(q.s, q.cell, i => q.s.divergenceFine[i]!) < 0,
      "positive-divergence": q => pressureCell(q) && cellMean(q.s, q.cell, i => q.s.divergenceFine[i]!) > 0,
    },
    draw(c) {
      const { s, lattice } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.divergenceFine.length; i++) peak = Math.max(peak, Math.abs(s.divergenceFine[i]));
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= pressureFill) continue;
        const mean = cellMean(s, cell, i => s.divergenceFine[i]);
        tint(c, cell.x0, cell.y0, cell.width,
          mean < 0 ? PALETTE.pressure : PALETTE.alarm,
          Math.min(0.85, (Math.abs(mean) / peak) * 1.6), cell.height);
      }
    },
  },
  "pressure-solve": {
    holds: {
      "high-pressure": q => pressureCell(q) && cellMean(q.s, q.cell, i => q.s.pressureFine[i]!) > 0,
      /* Restricted to cells that hold water. Every empty cell in the domain
       * is also at p = 0, and saying so about air three bricks from the
       * liquid would bury the one place the reading means something. */
      "free-surface": q => q.cell.open && q.cell.fill > interfaceFill
        && (!pressureCell(q) || cellMean(q.s, q.cell, i => q.s.pressureFine[i]!) <= 0),
    },
    draw(c) {
      const { s, lattice } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.pressureFine.length; i++) peak = Math.max(peak, s.pressureFine[i]);
      for (const cell of lattice.cells) {
        if (!cell.open || cell.fill <= pressureFill) continue;
        const mean = cellMean(s, cell, i => s.pressureFine[i]);
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.pressure,
          Math.min(0.9, (Math.max(0, mean) / peak) * 0.95), cell.height);
      }
    },
  },
  "velocity-projection": {
    holds: {
      "before-projection": carriesVelocity,
      "after-projection": carriesVelocity,
    },
    draw(c) {
      velocityField(c, 3, PALETTE.muted, 0.55, true);
      velocityField(c, 3, PALETTE.amber, 0.95, false);
    },
  },
  "conservative-transport": {
    holds: {
      "swept-flux": q => {
        const peak = Math.max(q.peakOf(q.s.limitedFluxXFine), q.peakOf(q.s.limitedFluxYFine));
        return boundingRowsX(q)
          .some(row => Math.abs(q.s.limitedFluxXFine[row]!) >= peak * fluxShare)
          || boundingRowsY(q)
            .some(row => Math.abs(q.s.limitedFluxYFine[row]!) >= peak * fluxShare);
      },
      "limiter-clipped": q => boundingRowsX(q).some(row => q.s.fluxLimitedXFine[row] !== 0)
        || boundingRowsY(q).some(row => q.s.fluxLimitedYFine[row] !== 0),
    },
    draw(c) {
      const { g, s, scale: S } = c;
      let peak = 1e-6;
      for (let i = 0; i < s.limitedFluxXFine.length; i++) peak = Math.max(peak, Math.abs(s.limitedFluxXFine[i]));
      for (let i = 0; i < s.limitedFluxYFine.length; i++) peak = Math.max(peak, Math.abs(s.limitedFluxYFine[i]));
      g.strokeStyle = PALETTE.transport;
      g.fillStyle = PALETTE.transport;
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        const flux = s.limitedFluxXFine[advanceRowX(s, x, y)];
        if (Math.abs(flux) < peak * fluxShare) continue;
        g.globalAlpha = Math.min(1, 0.3 + Math.abs(flux) / peak);
        const width = Math.max(2, Math.abs(flux) * S * 2.2);
        g.fillRect(x * S - (flux > 0 ? width : 0), y * S + 2, width, S - 4);
        arrow(g, x * S - (flux > 0 ? 4 : -4), (y + 0.5) * S, Math.sign(flux) * 9, 0);
      }
      for (let y = 0; y <= s.ny; y++) for (let x = 0; x < s.nx; x++) {
        const flux = s.limitedFluxYFine[advanceRowY(s, x, y)];
        if (Math.abs(flux) < peak * fluxShare) continue;
        g.globalAlpha = Math.min(1, 0.3 + Math.abs(flux) / peak);
        arrow(g, (x + 0.5) * S, y * S - (flux > 0 ? 4 : -4), 0, Math.sign(flux) * 9);
      }
      g.globalAlpha = 1;
      g.strokeStyle = PALETTE.alarm;
      g.lineWidth = 2.4;
      g.beginPath();
      for (let y = 0; y < s.ny; y++) for (let x = 0; x <= s.nx; x++) {
        if (!s.fluxLimitedXFine[advanceRowX(s, x, y)]) continue;
        g.moveTo(x * S, y * S + 1);
        g.lineTo(x * S, (y + 1) * S - 1);
      }
      g.stroke();
    },
  },
  "tracer-advection": {
    holds: {
      "marker": q => q.s.markers.some(marker =>
        marker.x >= q.cell.x0 && marker.x < q.cell.x0 + q.cell.width
        && marker.y >= q.cell.y0 && marker.y < q.cell.y0 + q.cell.height),
    },
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
    holds: {
      "volume-changed": q => {
        let changed = 0;
        for (let j = 0; j < q.cell.height; j += 1) for (let i = 0; i < q.cell.width; i += 1) {
          const index = advanceCell(q.s, q.cell.x0 + i, q.cell.y0 + j);
          changed += Math.abs(
            q.s.liquidVolumeFine[index]! - q.s.previousLiquidVolumeFine[index]!);
        }
        return changed >= changedVolume;
      },
    },
    draw(c) {
      const { s, lattice } = c;
      for (const cell of lattice.cells) {
        if (!cell.open) continue;
        let changed = 0;
        for (let j = 0; j < cell.height; j++) for (let i = 0; i < cell.width; i++) {
          const index = advanceCell(s, cell.x0 + i, cell.y0 + j);
          changed += Math.abs(s.liquidVolumeFine[index] - s.previousLiquidVolumeFine[index]);
        }
        if (changed < changedVolume) continue;
        tint(c, cell.x0, cell.y0, cell.width, PALETTE.output,
          Math.min(0.8, 0.18 + changed * 3), cell.height);
      }
    },
  },
  "activity-measurement": {
    holds: {
      "high-activity": q => q.s.brickActivity[q.brick]! > activeBrickScore,
    },
    draw(c) {
      const { s, scale: S } = c;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const score = s.brickActivity[by * s.bx + bx];
        if (score <= activeBrickScore) continue;
        tint(c, bx * ADVANCE_BRICK_FINE, by * ADVANCE_BRICK_FINE, ADVANCE_BRICK_FINE,
          PALETTE.adaptivity, Math.min(0.7, score * 0.8));
        label(c, (bx + 0.5) * ADVANCE_BRICK_FINE * S, (by + 0.5) * ADVANCE_BRICK_FINE * S,
          score.toFixed(2));
      }
    },
  },
  "resolution-planning": {
    holds: {
      "target-rung": () => true,
    },
    draw(c) {
      const { s, scale: S } = c;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const rung = s.brickRung[by * s.bx + bx];
        tint(c, bx * ADVANCE_BRICK_FINE, by * ADVANCE_BRICK_FINE, ADVANCE_BRICK_FINE,
          PALETTE.adaptivity, 0.08 + 0.13 * rung);
        label(c, (bx + 0.5) * ADVANCE_BRICK_FINE * S, (by + 0.5) * ADVANCE_BRICK_FINE * S,
          `${ADVANCE_RUNGS[rung]}²`);
      }
    },
  },
  "candidate-transfer": {
    holds: {
      "rung-changed": q => q.s.brickRung[q.brick] !== q.s.previousBrickRung[q.brick],
    },
    draw(c) {
      const { g, s, scale: S } = c;
      g.lineWidth = 3;
      g.strokeStyle = PALETTE.output;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        const brick = by * s.bx + bx;
        if (s.brickRung[brick] === s.previousBrickRung[brick]) continue;
        tint(c, bx * ADVANCE_BRICK_FINE, by * ADVANCE_BRICK_FINE, ADVANCE_BRICK_FINE, PALETTE.output, 0.3);
        g.strokeRect(bx * ADVANCE_BRICK_FINE * S + 2, by * ADVANCE_BRICK_FINE * S + 2,
          ADVANCE_BRICK_FINE * S - 4, ADVANCE_BRICK_FINE * S - 4);
        label(c, (bx + 0.5) * ADVANCE_BRICK_FINE * S, (by + 0.5) * ADVANCE_BRICK_FINE * S,
          `${ADVANCE_RUNGS[s.previousBrickRung[brick]!]} → ${ADVANCE_RUNGS[s.brickRung[brick]!]}`);
      }
    },
  },
  "brick-retirement": {
    holds: {
      "retired-brick": q => brickVolumeAt(q.s, q.brick) <= residentBrickVolume,
      "resident-brick": q => brickVolumeAt(q.s, q.brick) > residentBrickVolume,
    },
    draw(c) {
      const { g, s, scale: S } = c;
      g.strokeStyle = PALETTE.muted;
      g.globalAlpha = 0.45;
      g.lineWidth = 1;
      for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
        if (brickVolume(s, bx, by) > residentBrickVolume) continue;
        g.save();
        g.beginPath();
        g.rect(bx * ADVANCE_BRICK_FINE * S, by * ADVANCE_BRICK_FINE * S,
          ADVANCE_BRICK_FINE * S, ADVANCE_BRICK_FINE * S);
        g.clip();
        g.beginPath();
        for (let d = -ADVANCE_BRICK_FINE; d < ADVANCE_BRICK_FINE; d += 1.6) {
          g.moveTo((bx * ADVANCE_BRICK_FINE + d) * S, by * ADVANCE_BRICK_FINE * S);
          g.lineTo((bx * ADVANCE_BRICK_FINE + d + ADVANCE_BRICK_FINE) * S, (by + 1) * ADVANCE_BRICK_FINE * S);
        }
        g.stroke();
        g.restore();
      }
      g.globalAlpha = 1;
    },
  },
  "presentation-publication": {
    holds: {
      "published-interface": q => advanceCellPlane(q.lattice, q.cell) !== null,
    },
    draw(c) { drawInterface(c, PALETTE.output, 2.6); },
  },
} as const satisfies Readonly<Record<AdvanceStageId, StageDrawing>>;

/**
 * Join one declaration to its drawing.
 *
 * The declaration is the roster: a mark exists because the method says the
 * stage can put it on a cell. The drawing has to answer for every one of them
 * and for no others — an unanswered mark would sit in the strip and never
 * appear in the probe, and an unclaimed predicate is a reading nobody named.
 */
function joinLens(declaration: Omit<AdvanceSliceDeclaration, "loopStep">,
  drawing: StageDrawing): Lens {
  const answered = new Set(Object.keys(drawing.holds));
  const keys = declaration.keys.map(declared => {
    const holds = drawing.holds[declared.id];
    if (!holds) throw new Error(`the "${declared.id}" mark is declared but never answered`);
    answered.delete(declared.id);
    return { ...declared, holds };
  });
  if (answered.size) {
    throw new Error(`nothing declares the ${[...answered].join(", ")} mark`);
  }
  return { caption: declaration.caption, keys, draw: drawing.draw };
}

/* ---- one lens per stage -------------------------------------------- */

/**
 * The lens roster, assembled rather than written.
 *
 * `Record<AdvanceStageId, Lens>` is what makes a stage the encoder renames a
 * type error here as well as in the registry, and `advanceStageSlice` is what
 * makes a stage with no declaration a loud failure rather than a blank caption.
 */
export const ADVANCE_LENSES: Readonly<Record<AdvanceStageId, Lens>> = Object.freeze(
  Object.fromEntries(ADVANCE_STAGE_ORDER.map(stage =>
    [stage, joinLens(advanceStageSlice(stage), ADVANCE_STAGE_DRAWING[stage])],
  )) as Record<AdvanceStageId, Lens>);

/**
 * Step 1 of the loop is not a stage — it is the state the advance starts from,
 * so it gets a lens of its own rather than borrowing one.
 */
export const REPRESENT_LENS: Lens = joinLens(ADVANCE_REPRESENT_SLICE, {
  holds: {
    "brick-rung": () => true,
    "reconstructed-interface": q => advanceCellPlane(q.lattice, q.cell) !== null,
  },
  draw(c) {
    const { s, scale: S } = c;
    for (let by = 0; by < s.by; by++) for (let bx = 0; bx < s.bx; bx++) {
      if (brickVolume(s, bx, by) <= residentBrickVolume) continue;
      label(c, (bx + 0.5) * ADVANCE_BRICK_FINE * S, by * ADVANCE_BRICK_FINE * S + 9,
        `${ADVANCE_RUNGS[s.brickRung[by * s.bx + bx]!]}²`, PALETTE.muted);
    }
    drawInterface(c, PALETTE.output, 2.6);
  },
});
