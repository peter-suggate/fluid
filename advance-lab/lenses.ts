/**
 * The slice, drawn, and one lens per stage of the advance.
 *
 * Every lens paints onto the same picture — the same water, the same bricks,
 * the same rungs — so switching stages moves the reading rather than the
 * subject. That is the whole design: a stage list that changes what you can
 * see about one scene, not fifteen separate diagrams.
 */
import {
  buildSliceLattice, buildSliceSums, type LatticeCell, latticePlane,
  type SliceLattice,
} from "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import {
  type AdvanceSlice, clipUnitSquare, SLICE_BRICK, SLICE_RUNGS,
  sliceCell, sliceRowX, sliceRowY, UNIT_SQUARE,
} from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import type { AdvanceStageId } from "../lib/methods/adaptive-volume/advance-slice/advance-work";

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

/** Grid at each brick's rung, liquid cut by PLIC, bricks, then solids. */
export function drawSlice(c: LensContext): void {
  const { g, s, lattice, scale: S } = c;
  buildSliceSums(lattice, s);
  buildSliceLattice(lattice, s);
  g.clearRect(0, 0, s.nx * S, s.ny * S);
  g.fillStyle = PALETTE.ground;
  g.fillRect(0, 0, s.nx * S, s.ny * S);

  g.fillStyle = PALETTE.liquid;
  g.beginPath();
  for (const cell of lattice.cells) {
    if (!cell.open || cell.fill <= 1e-3) continue;
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    if (cell.fill >= 1 - 1e-3) { g.rect(x, y, w, h); continue; }
    const plane = latticePlane(lattice, cell);
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

/** Every PLIC segment, which is what presentation actually publishes. */
function drawInterface(c: LensContext, color: string, width: number): void {
  const { g, lattice, scale: S } = c;
  g.strokeStyle = color;
  g.lineWidth = width;
  g.lineCap = "round";
  g.beginPath();
  for (const cell of lattice.cells) {
    if (!cell.open || cell.fill <= 1e-3 || cell.fill >= 1 - 1e-3) continue;
    const plane = latticePlane(lattice, cell);
    if (!plane) continue;
    const polygon = clipUnitSquare(UNIT_SQUARE, plane.nx, plane.ny, plane.offset);
    const x = cell.x0 * S, y = cell.y0 * S;
    const w = cell.width * S, h = cell.height * S;
    const onEdge = (value: number): boolean => value < 1e-6 || value > 1 - 1e-6;
    for (let i = 0; i < polygon.length; i += 2) {
      const j = (i + 2) % polygon.length;
      const ax = polygon[i], ay = polygon[i + 1];
      const bx = polygon[j], by = polygon[j + 1];
      /* a segment lying along a cell edge is the box, not the interface */
      if (onEdge(ax) && onEdge(bx) && Math.abs(ax - bx) < 1e-6) continue;
      if (onEdge(ay) && onEdge(by) && Math.abs(ay - by) < 1e-6) continue;
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
