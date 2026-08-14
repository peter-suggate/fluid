#!/usr/bin/env node
/**
 * H0 of `docs/hero-fidelity-1000x-handoff.md` — the hero garden's fidelity gate.
 *
 *   npm run gate:hero-fidelity              — render four reps and score them
 *   npm run gate:hero-fidelity:overlay      — draw the scored regions on the plate
 *
 * Every change after this one is an opinion without a number, so this lane
 * exists before the work rather than after it. It produces two scores per
 * region — CIEDE2000 for tone and a gradient-magnitude histogram distance for
 * detail density — plus a contact sheet a person can look at, and it reports
 * the spread across reps so a later A/B can be read against the lane's own
 * noise rather than against zero.
 *
 * ---------------------------------------------------------------------------
 * Why it renders four reps by default
 * ---------------------------------------------------------------------------
 * It was written when the hero frame was believed not to be reproducible across
 * processes — `binDirtyBrickCandidates` dropping the losers of an atomic race, so
 * a second process keeps a different 64 primitives in each over-subscribed brick.
 * On that premise a single-run score would carry an unknown error bar, and the
 * program's measurement rule (`docs/hero-fidelity-1000x-handoff.md` §5) is that
 * arms are interleaved rather than run in blocks. So this lane reports the
 * **median** across reps as the score and the **half-range** as the noise floor,
 * and gates a candidate against a baseline *through* that floor.
 *
 * **The premise no longer holds, and this lane's own output is the evidence:**
 * `docs/hero-fidelity-baseline.json` records a `deltaE00HalfRange` and a
 * `gradientDistanceHalfRange` of exactly **0** in every region across four reps,
 * and two separate `hero-garden-hose` smoke processes now produce byte-identical
 * PNGs and the same settled hash. See the corrected note in
 * `tools/run-svo-dry-render-smoke.ts`.
 *
 * The default stays at four reps, deliberately: the reps are what *demonstrate*
 * the floor is zero, and a zero half-range is only informative when something
 * was actually repeated. But a score here does not need interleaving to be
 * trusted, and a difference is not obliged to clear a noise floor that is zero —
 * it has to clear the **measurement** floor instead, which the program puts at
 * 0.25 ΔE₀₀ and 0.02 octaves.
 *
 * ---------------------------------------------------------------------------
 * Why it drives the smoke lane rather than rendering itself
 * ---------------------------------------------------------------------------
 * Two PNG paths exist in this tree and they **disagree on grading**:
 * `tools/run-svo-dry-render-smoke.ts` applies the scene's own ACES grade via
 * `resolveDisplayGrade(scene.lighting?.grade)`, while
 * `tools/benchmark-svo-dry-frame-gpu.ts` applies a bare gamma 2.2 with no tone
 * curve and no exposure. Scoring the second against the plate would measure the
 * missing tonemap and nothing else — the ΔE₀₀ number would be dominated by a
 * difference that is not in the renderer.
 *
 * This lane therefore spawns the smoke lane, which also means it inherits the
 * `tools/run-webgpu-exclusive.ts` GPU mutex per rep instead of holding one lock
 * across all four.
 *
 * ---------------------------------------------------------------------------
 * Environment
 * ---------------------------------------------------------------------------
 *   FLUID_HERO_FIDELITY_MODE      `score` (default) | `overlay`
 *   FLUID_HERO_FIDELITY_REPS      reps to render and score (default 4)
 *   FLUID_HERO_FIDELITY_WIDTH     render width (default 836 — half the plate)
 *   FLUID_HERO_FIDELITY_HEIGHT    render height (default 470)
 *   FLUID_HERO_FIDELITY_SCENE     scene preset (default hero-garden-hose)
 *   FLUID_HERO_FIDELITY_FRAMES    pre-rendered frame PNGs, comma separated;
 *                                 when set, nothing is rendered
 *   FLUID_HERO_FIDELITY_PLATE     plate path (default the tracked reference)
 *   FLUID_HERO_FIDELITY_OUT       report directory (default artifacts/hero-fidelity)
 *   FLUID_HERO_FIDELITY_BASELINE  a previous report.json; regression fails the lane.
 *                                 Defaults to the committed docs/hero-fidelity-baseline.json;
 *                                 set it empty to report without gating.
 *   FLUID_HERO_FIDELITY_KEEP      keep the per-rep frame PNGs (default 1)
 *
 * Exits 0 when the lane produced both scores; non-zero when a baseline was
 * given and a region regressed beyond the measured noise floor.
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

import { encodeRgbPng, decodeRgbPng, type RgbImage } from "../lib/core/png-codec";
import {
  decodeLinearPng,
  deltaE2000,
  HERO_FIDELITY_REGIONS,
  labFromLinearRgb,
  linearFromSrgbImage,
  regionBounds,
  registerToScoringGrid,
  scoreFidelity,
  srgbImageFromLinear,
  type FidelityReport,
  type FidelityRegion,
  type LinearImage,
  type RegionFidelityScore,
} from "../lib/core/hero-fidelity-score";

const environment = process.env;
const mode = environment.FLUID_HERO_FIDELITY_MODE ?? "score";
const reps = Number(environment.FLUID_HERO_FIDELITY_REPS ?? 4);
const width = Number(environment.FLUID_HERO_FIDELITY_WIDTH ?? 836);
const height = Number(environment.FLUID_HERO_FIDELITY_HEIGHT ?? 470);
const scenePreset = environment.FLUID_HERO_FIDELITY_SCENE ?? "hero-garden-hose";
const platePath = environment.FLUID_HERO_FIDELITY_PLATE ?? "output/imagegen/garden-pond-hose-fill-simplified.png";
const outDirectory = environment.FLUID_HERO_FIDELITY_OUT ?? "artifacts/hero-fidelity";
/**
 * The committed baseline, used unless overridden.
 *
 * Without this the lane reports numbers and gates on nothing, which is one step
 * better than an opinion and not the thing H0 asked for ("Regression on either
 * score fails the lane"). It lives under `docs/` because `artifacts/` is
 * gitignored, so a report written there cannot be the thing a later run is
 * measured against.
 *
 * Set `FLUID_HERO_FIDELITY_BASELINE=` (empty) to report without gating — which
 * is what you want while deliberately changing the image, before re-blessing.
 */
const DEFAULT_BASELINE = "docs/hero-fidelity-baseline.json";
const baselinePath = environment.FLUID_HERO_FIDELITY_BASELINE !== undefined
  ? (environment.FLUID_HERO_FIDELITY_BASELINE || undefined)
  : (existsSync(DEFAULT_BASELINE) ? DEFAULT_BASELINE : undefined);

if (!Number.isInteger(reps) || reps < 1) throw new RangeError(`FLUID_HERO_FIDELITY_REPS must be >= 1, got ${reps}`);

// ---------------------------------------------------------------------------
// A 5x7 bitmap font, so the contact sheet needs no legend beside it
// ---------------------------------------------------------------------------
/**
 * A sheet whose labels live in a separate file is a sheet that gets read
 * against the wrong run. These glyphs are the alphabet the region ids and the
 * two score names actually use.
 */
const GLYPHS: Readonly<Record<string, readonly string[]>> = Object.freeze({
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  B: ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
  C: ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
  D: ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
  F: ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
  G: ["01110", "10001", "10000", "10111", "10001", "10001", "01111"],
  H: ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
  I: ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
  J: ["00111", "00010", "00010", "00010", "00010", "10010", "01100"],
  K: ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
  L: ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
  M: ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
  N: ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
  O: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
  P: ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
  Q: ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  S: ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
  T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
  U: ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
  V: ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
  W: ["10001", "10001", "10001", "10101", "10101", "11011", "10001"],
  X: ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
  Y: ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
  Z: ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
  "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
  "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
  "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
  "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
  "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
  "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
  "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
  "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
  "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
  "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
  ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
  ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
  "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
  "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
  "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
  " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
});

interface Canvas {
  readonly width: number;
  readonly height: number;
  readonly rgb: Uint8Array;
}

function createCanvas(canvasWidth: number, canvasHeight: number, fill: readonly [number, number, number]): Canvas {
  const rgb = new Uint8Array(canvasWidth * canvasHeight * 3);
  for (let pixel = 0; pixel < canvasWidth * canvasHeight; pixel += 1) {
    rgb[pixel * 3] = fill[0];
    rgb[pixel * 3 + 1] = fill[1];
    rgb[pixel * 3 + 2] = fill[2];
  }
  return { width: canvasWidth, height: canvasHeight, rgb };
}

function blit(canvas: Canvas, image: RgbImage, originX: number, originY: number): void {
  for (let y = 0; y < image.height; y += 1) {
    const destinationY = y + originY;
    if (destinationY < 0 || destinationY >= canvas.height) continue;
    for (let x = 0; x < image.width; x += 1) {
      const destinationX = x + originX;
      if (destinationX < 0 || destinationX >= canvas.width) continue;
      const source = (y * image.width + x) * 3;
      const destination = (destinationY * canvas.width + destinationX) * 3;
      canvas.rgb[destination] = image.rgb[source];
      canvas.rgb[destination + 1] = image.rgb[source + 1];
      canvas.rgb[destination + 2] = image.rgb[source + 2];
    }
  }
}

function setPixel(canvas: Canvas, x: number, y: number, colour: readonly [number, number, number]): void {
  if (x < 0 || y < 0 || x >= canvas.width || y >= canvas.height) return;
  const destination = (y * canvas.width + x) * 3;
  canvas.rgb[destination] = colour[0];
  canvas.rgb[destination + 1] = colour[1];
  canvas.rgb[destination + 2] = colour[2];
}

function strokeRect(
  canvas: Canvas,
  x: number,
  y: number,
  rectWidth: number,
  rectHeight: number,
  colour: readonly [number, number, number],
): void {
  for (let column = 0; column < rectWidth; column += 1) {
    setPixel(canvas, x + column, y, colour);
    setPixel(canvas, x + column, y + rectHeight - 1, colour);
  }
  for (let row = 0; row < rectHeight; row += 1) {
    setPixel(canvas, x, y + row, colour);
    setPixel(canvas, x + rectWidth - 1, y + row, colour);
  }
}

function fillRect(
  canvas: Canvas,
  x: number,
  y: number,
  rectWidth: number,
  rectHeight: number,
  colour: readonly [number, number, number],
): void {
  for (let row = 0; row < rectHeight; row += 1) {
    for (let column = 0; column < rectWidth; column += 1) setPixel(canvas, x + column, y + row, colour);
  }
}

function drawText(
  canvas: Canvas,
  text: string,
  x: number,
  y: number,
  colour: readonly [number, number, number],
  scale = 1,
): number {
  let cursor = x;
  for (const character of text.toUpperCase()) {
    const glyph = GLYPHS[character] ?? GLYPHS[" "];
    for (let row = 0; row < 7; row += 1) {
      for (let column = 0; column < 5; column += 1) {
        if (glyph[row][column] !== "1") continue;
        fillRect(canvas, cursor + column * scale, y + row * scale, scale, scale, colour);
      }
    }
    cursor += 6 * scale;
  }
  return cursor;
}

/** Distinct, and distinguishable against both the plate's white set and its teal pool. */
const REGION_COLOURS: Readonly<Record<string, readonly [number, number, number]>> = Object.freeze({
  pond: [255, 96, 32],
  coping: [40, 180, 255],
  stones: [255, 220, 0],
  canopy: [255, 40, 200],
  ground: [60, 255, 120],
});

/** A sign that is always present, so a column of biases stays aligned and readable. */
function signed(value: number): string {
  return `${value >= 0 ? "+" : "-"}${Math.abs(value).toFixed(3)}`;
}

function regionColour(id: string): readonly [number, number, number] {
  return REGION_COLOURS[id] ?? [255, 255, 255];
}

// ---------------------------------------------------------------------------
// Rendering reps
// ---------------------------------------------------------------------------

function renderRep(index: number): string {
  const framePath = path.resolve(outDirectory, `frame-rep${index}.png`);
  const reportPath = path.resolve(outDirectory, `frame-rep${index}.json`);
  const result = spawnSync(
    process.execPath,
    [
      "--import",
      "tsx",
      "tools/run-webgpu-exclusive.ts",
      "--import",
      "tsx",
      "tools/run-svo-dry-render-smoke.ts",
    ],
    {
      stdio: ["ignore", "pipe", "inherit"],
      encoding: "utf8",
      env: {
        ...environment,
        WEBGPU_NODE_MODULE: environment.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`,
        FLUID_SVO_DRY_SMOKE_SCENE: scenePreset,
        FLUID_SVO_DRY_SMOKE_WIDTH: String(width),
        FLUID_SVO_DRY_SMOKE_HEIGHT: String(height),
        FLUID_SVO_DRY_SMOKE_PNG: framePath,
        FLUID_SVO_DRY_SMOKE_OUT: reportPath,
      },
    },
  );
  if (result.status !== 0) {
    process.stdout.write(result.stdout ?? "");
    throw new Error(`render rep ${index} failed with status ${result.status}`);
  }
  return framePath;
}

// ---------------------------------------------------------------------------
// Contact sheet
// ---------------------------------------------------------------------------

/**
 * ΔE₀₀ as a heat map, on a fixed scale.
 *
 * The scale is pinned rather than normalised per frame: an auto-scaled heat map
 * looks the same for a frame that is 2 ΔE off and one that is 40 ΔE off, which
 * would make the sheet useless for exactly the comparison it exists to support.
 * 0 is black, `DELTA_E_HEAT_CEILING` and above is white-hot.
 */
const DELTA_E_HEAT_CEILING = 40;

function deltaEHeatMap(frame: LinearImage, plate: LinearImage): RgbImage {
  const rgb = new Uint8Array(frame.width * frame.height * 3);
  for (let pixel = 0; pixel < frame.width * frame.height; pixel += 1) {
    const source = pixel * 3;
    const difference = deltaE2000(
      labFromLinearRgb(frame.rgb[source], frame.rgb[source + 1], frame.rgb[source + 2]),
      labFromLinearRgb(plate.rgb[source], plate.rgb[source + 1], plate.rgb[source + 2]),
    );
    const t = Math.min(1, difference / DELTA_E_HEAT_CEILING);
    // Black → deep blue → magenta → orange → white: monotone in luminance, so
    // it reads correctly in a greyscale print and has no false banding.
    rgb[source] = Math.round(255 * Math.min(1, Math.max(0, t * 2.2 - 0.35)));
    rgb[source + 1] = Math.round(255 * Math.min(1, Math.max(0, t * 2.4 - 1.1)));
    rgb[source + 2] = Math.round(255 * Math.min(1, Math.max(0, t * 3.0 - 0.05)));
  }
  return { width: frame.width, height: frame.height, rgb };
}

const PANEL_LABEL_HEIGHT = 22;
const SHEET_MARGIN = 12;

function buildContactSheet(
  frame: LinearImage,
  plate: LinearImage,
  report: FidelityReport,
  noise: ReadonlyMap<string, RegionNoise>,
  regions: readonly FidelityRegion[],
): RgbImage {
  const panels: Array<{ readonly title: string; readonly image: RgbImage; readonly boxes: boolean }> = [
    { title: "plate registered", image: srgbImageFromLinear(plate), boxes: true },
    { title: "frame", image: srgbImageFromLinear(frame), boxes: true },
    { title: `delta e00 heat 0-${DELTA_E_HEAT_CEILING}`, image: deltaEHeatMap(frame, plate), boxes: false },
  ];
  const tableHeight = PANEL_LABEL_HEIGHT + (regions.length + 2) * 14;
  const sheetWidth = SHEET_MARGIN * 2 + panels.length * frame.width + (panels.length - 1) * SHEET_MARGIN;
  const sheetHeight = SHEET_MARGIN * 3 + PANEL_LABEL_HEIGHT + frame.height + tableHeight;
  const canvas = createCanvas(sheetWidth, sheetHeight, [18, 18, 20]);

  panels.forEach((panel, index) => {
    const originX = SHEET_MARGIN + index * (frame.width + SHEET_MARGIN);
    const originY = SHEET_MARGIN + PANEL_LABEL_HEIGHT;
    drawText(canvas, panel.title, originX, SHEET_MARGIN + 6, [210, 210, 215], 2);
    blit(canvas, panel.image, originX, originY);
    if (!panel.boxes) return;
    for (const region of regions) {
      const bounds = regionBounds(region, frame.width, frame.height);
      const colour = regionColour(region.id);
      strokeRect(canvas, originX + bounds.x, originY + bounds.y, bounds.width, bounds.height, colour);
      drawText(canvas, region.id, originX + bounds.x + 2, originY + bounds.y + 2, colour, 1);
    }
  });

  // The table: one row per region, both scores, and the rep spread beside each.
  let tableY = SHEET_MARGIN * 2 + PANEL_LABEL_HEIGHT + frame.height + 8;
  const columnX = SHEET_MARGIN;
  drawText(canvas, "region      delta e00  p95    grad oct   +-      bias    grad frame/plate", columnX + 14, tableY, [170, 170, 178], 1);
  tableY += 14;
  for (const score of report.regions) {
    const colour = regionColour(score.id);
    fillRect(canvas, columnX, tableY, 10, 8, colour);
    const spread = noise.get(score.id);
    const row = [
      score.id.padEnd(11),
      score.deltaE00Mean.toFixed(2).padStart(9),
      score.deltaE00P95.toFixed(2).padStart(7),
      score.gradientDistance.toFixed(3).padStart(10),
      (spread ? spread.gradientDistanceHalfRange.toFixed(3) : "-").padStart(8),
      signed(score.gradientBias).padStart(8),
      `${score.gradientMeanFrame.toFixed(3)}/${score.gradientMeanPlate.toFixed(3)}`.padStart(18),
    ].join("");
    drawText(canvas, row, columnX + 14, tableY, [225, 225, 230], 1);
    tableY += 14;
  }
  drawText(
    canvas,
    `mean       ${report.deltaE00Mean.toFixed(2).padStart(9)}${"".padStart(7)}${report.gradientDistanceMean.toFixed(3).padStart(10)}`,
    columnX + 14,
    tableY,
    [255, 255, 255],
    1,
  );
  return { width: canvas.width, height: canvas.height, rgb: canvas.rgb };
}

// ---------------------------------------------------------------------------
// Aggregation across reps
// ---------------------------------------------------------------------------

interface RegionNoise {
  readonly deltaE00HalfRange: number;
  readonly gradientDistanceHalfRange: number;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = sorted.length >> 1;
  return sorted.length % 2 === 1 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function halfRange(values: readonly number[]): number {
  return values.length < 2 ? 0 : (Math.max(...values) - Math.min(...values)) / 2;
}

function aggregate(reports: readonly FidelityReport[]): {
  readonly report: FidelityReport;
  readonly noise: Map<string, RegionNoise>;
} {
  const noise = new Map<string, RegionNoise>();
  const regions: RegionFidelityScore[] = [];
  for (let index = 0; index < reports[0].regions.length; index += 1) {
    const across = reports.map((report) => report.regions[index]);
    const first = across[0];
    regions.push({
      ...first,
      deltaE00Mean: median(across.map((score) => score.deltaE00Mean)),
      deltaE00P95: median(across.map((score) => score.deltaE00P95)),
      gradientDistance: median(across.map((score) => score.gradientDistance)),
      gradientBias: median(across.map((score) => score.gradientBias)),
      gradientMeanFrame: median(across.map((score) => score.gradientMeanFrame)),
      gradientMeanPlate: first.gradientMeanPlate,
    });
    noise.set(first.id, {
      deltaE00HalfRange: halfRange(across.map((score) => score.deltaE00Mean)),
      gradientDistanceHalfRange: halfRange(across.map((score) => score.gradientDistance)),
    });
  }
  return {
    report: {
      width: reports[0].width,
      height: reports[0].height,
      regions,
      deltaE00Mean: regions.reduce((sum, score) => sum + score.deltaE00Mean, 0) / regions.length,
      gradientDistanceMean: regions.reduce((sum, score) => sum + score.gradientDistance, 0) / regions.length,
    },
    noise,
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

mkdirSync(outDirectory, { recursive: true });
const plateImage = decodeLinearPng(readFileSync(platePath));

if (mode === "overlay") {
  // Registration check: the scored rectangles, drawn on the plate at plate
  // resolution. Region bounds that address the wrong subject are the one
  // failure this whole lane cannot detect from its own numbers.
  const registered = registerToScoringGrid(
    { width, height, rgb: new Float32Array(width * height * 3) },
    plateImage,
  ).plate;
  const sheet = srgbImageFromLinear(registered);
  const canvas: Canvas = { width: sheet.width, height: sheet.height, rgb: Uint8Array.from(sheet.rgb) };
  for (const region of HERO_FIDELITY_REGIONS) {
    const bounds = regionBounds(region, canvas.width, canvas.height);
    const colour = regionColour(region.id);
    strokeRect(canvas, bounds.x, bounds.y, bounds.width, bounds.height, colour);
    strokeRect(canvas, bounds.x + 1, bounds.y + 1, bounds.width - 2, bounds.height - 2, colour);
    drawText(canvas, region.id, bounds.x + 4, bounds.y + 4, colour, 2);
  }
  const overlayPath = path.join(outDirectory, "region-overlay.png");
  writeFileSync(overlayPath, encodeRgbPng(canvas));
  process.stdout.write(`[hero-fidelity] region overlay written to ${overlayPath}\n`);
  process.exit(0);
}

const preRendered = environment.FLUID_HERO_FIDELITY_FRAMES?.split(",").map((entry) => entry.trim()).filter(Boolean);
const framePaths = preRendered?.length
  ? preRendered
  : Array.from({ length: reps }, (_unused, index) => renderRep(index));

const reports: FidelityReport[] = [];
let lastRegistered: { frame: LinearImage; plate: LinearImage } | undefined;
for (const framePath of framePaths) {
  const frame = linearFromSrgbImage(decodeRgbPng(readFileSync(framePath)));
  const registered = registerToScoringGrid(frame, plateImage);
  lastRegistered = { frame: registered.frame, plate: registered.plate };
  reports.push(scoreFidelity(registered.frame, registered.plate));
}
if (!lastRegistered) throw new Error("no frames were scored");

const { report, noise } = aggregate(reports);
const sheetPath = path.join(outDirectory, "contact-sheet.png");
writeFileSync(
  sheetPath,
  encodeRgbPng(buildContactSheet(lastRegistered.frame, lastRegistered.plate, report, noise, HERO_FIDELITY_REGIONS)),
);

const payload = {
  scene: scenePreset,
  plate: platePath,
  grid: { width: report.width, height: report.height },
  reps: framePaths.length,
  frames: framePaths,
  deltaE00Mean: report.deltaE00Mean,
  gradientDistanceMean: report.gradientDistanceMean,
  regions: report.regions.map((score) => ({
    id: score.id,
    subject: score.subject,
    bounds: score.bounds,
    deltaE00Mean: score.deltaE00Mean,
    deltaE00P95: score.deltaE00P95,
    gradientDistance: score.gradientDistance,
    gradientBias: score.gradientBias,
    gradientMeanFrame: score.gradientMeanFrame,
    gradientMeanPlate: score.gradientMeanPlate,
    noise: noise.get(score.id),
  })),
  contactSheet: sheetPath,
};
const reportPath = path.join(outDirectory, "report.json");
writeFileSync(reportPath, `${JSON.stringify(payload, null, 2)}\n`);

process.stdout.write(`\n[hero-fidelity] ${scenePreset} vs ${path.basename(platePath)} at ${report.width}x${report.height}, ${framePaths.length} reps\n`);
process.stdout.write(
  "region      deltaE00   p95    gradOct       +-    bias   gradFrame  gradPlate\n" +
  "                                (octaves; bias > 0 = frame smoother than plate)\n",
);
for (const score of report.regions) {
  const spread = noise.get(score.id);
  process.stdout.write(
    `${score.id.padEnd(11)}${score.deltaE00Mean.toFixed(2).padStart(8)}${score.deltaE00P95.toFixed(2).padStart(7)}` +
      `${score.gradientDistance.toFixed(3).padStart(10)}${(spread?.gradientDistanceHalfRange ?? 0).toFixed(3).padStart(9)}` +
      `${signed(score.gradientBias).padStart(8)}${score.gradientMeanFrame.toFixed(4).padStart(11)}${score.gradientMeanPlate.toFixed(4).padStart(11)}\n`,
  );
}
process.stdout.write(
  `${"mean".padEnd(11)}${report.deltaE00Mean.toFixed(2).padStart(8)}${"".padStart(7)}${report.gradientDistanceMean.toFixed(3).padStart(10)}\n`,
);
process.stdout.write(`[hero-fidelity] contact sheet ${sheetPath}\n[hero-fidelity] report ${reportPath}\n`);

if (baselinePath) {
  const baseline = JSON.parse(readFileSync(baselinePath, "utf8")) as typeof payload;
  const failures: string[] = [];
  for (const score of report.regions) {
    const previous = baseline.regions.find((entry) => entry.id === score.id);
    if (!previous) continue;
    const spread = noise.get(score.id);
    // Regression is only claimed outside the measured noise; the lane's own
    // rule is that a difference inside its error bar has measured nothing.
    const deltaEFloor = Math.max(0.25, (spread?.deltaE00HalfRange ?? 0) + (previous.noise?.deltaE00HalfRange ?? 0));
    const gradientFloor = Math.max(
      0.02,
      (spread?.gradientDistanceHalfRange ?? 0) + (previous.noise?.gradientDistanceHalfRange ?? 0),
    );
    if (score.deltaE00Mean > previous.deltaE00Mean + deltaEFloor) {
      failures.push(
        `${score.id}: deltaE00 ${previous.deltaE00Mean.toFixed(2)} -> ${score.deltaE00Mean.toFixed(2)} (floor ${deltaEFloor.toFixed(2)})`,
      );
    }
    if (score.gradientDistance > previous.gradientDistance + gradientFloor) {
      failures.push(
        `${score.id}: gradOct ${previous.gradientDistance.toFixed(3)} -> ${score.gradientDistance.toFixed(3)} (floor ${gradientFloor.toFixed(3)})`,
      );
    }
  }
  if (failures.length > 0) {
    process.stdout.write(`\n[hero-fidelity] REGRESSED against ${baselinePath}\n`);
    for (const failure of failures) process.stdout.write(`  - ${failure}\n`);
    process.exit(1);
  }
  process.stdout.write(`\n[hero-fidelity] no regression against ${baselinePath}\n`);
}
