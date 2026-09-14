/** Render a CPU-only before/after SVG from gentle-moving-blob analysis artifacts. */
import assert from "node:assert/strict";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

type Arm = {
  arm: string;
  contours: { finalFine: number[] };
  grids: {
    dimensionsFine: [number, number];
    finalCapacityFine: number[];
    finalLiquidVolumeFine: number[];
  };
};
type Report = { arms: Arm[] };

const argument = (name: string, fallback?: string): string => {
  const prefix = `--${name}=`;
  const value = process.argv.find(item => item.startsWith(prefix))?.slice(prefix.length) ?? fallback;
  assert.ok(value, `missing --${name}=...`);
  return value;
};
const baselinePath = resolve(argument("before"));
const candidatePath = resolve(argument("after", baselinePath));
const outputPath = resolve(argument("output", "artifacts/level-set-volume/sharpening-contour-comparison.svg"));
const armName = argument("arm", "lsv-stationary");

const load = (path: string): Arm => {
  const report = JSON.parse(readFileSync(path, "utf8")) as Report;
  const arm = report.arms.find(value => value.arm === armName);
  assert.ok(arm, `${path} has no ${armName} arm`);
  assert.ok(arm.grids?.dimensionsFine, `${path} predates sharpening grid capture; rerun the analysis`);
  return arm;
};
const before = load(baselinePath), after = load(candidatePath);
assert.deepEqual(after.grids.dimensionsFine, before.grids.dimensionsFine,
  "before and after dimensions differ");
const [nx, ny] = before.grids.dimensionsFine;
const scale = Math.max(5, Math.min(12, Math.floor(620 / Math.max(nx, ny))));
const plotWidth = nx * scale, plotHeight = ny * scale;
const margin = 28, labelHeight = 42, width = 2 * plotWidth + 3 * margin;
const height = plotHeight + labelHeight + 2 * margin;
const escape = (value: string) => value.replace(/[&<>"']/g, c =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[c]!);

function contourPath(values: readonly number[], ox: number): string {
  const paths: string[] = [];
  for (let at = 0; at + 3 < values.length; at += 4) {
    const x0 = ox + values[at]! * scale, y0 = margin + (ny - values[at + 1]!) * scale;
    const x1 = ox + values[at + 2]! * scale, y1 = margin + (ny - values[at + 3]!) * scale;
    paths.push(`M${x0.toFixed(2)},${y0.toFixed(2)}L${x1.toFixed(2)},${y1.toFixed(2)}`);
  }
  return paths.join("");
}

function panel(arm: Arm, ox: number, label: string, overlay?: Arm): string {
  const parts = [`<g>`, `<rect x="${ox}" y="${margin}" width="${plotWidth}" height="${plotHeight}" fill="#fff" stroke="#64748b"/>`];
  for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const at = y * nx + x, capacity = arm.grids.finalCapacityFine[at] ?? 0;
    const volume = arm.grids.finalLiquidVolumeFine[at] ?? 0;
    const py = margin + (ny - y - 1) * scale;
    if (capacity < 0.999999) {
      const shade = Math.round(45 + 175 * Math.max(0, Math.min(1, capacity)));
      parts.push(`<rect x="${ox + x * scale}" y="${py}" width="${scale}" height="${scale}" fill="rgb(${shade},${shade},${shade})"/>`);
    }
    if (volume > 1e-8) {
      const alpha = (0.08 + 0.22 * Math.min(1, volume / Math.max(capacity, 1e-8))).toFixed(3);
      parts.push(`<rect x="${ox + x * scale}" y="${py}" width="${scale}" height="${scale}" fill="#38bdf8" fill-opacity="${alpha}"/>`);
    }
  }
  if (overlay) parts.push(`<path d="${contourPath(overlay.contours.finalFine, ox)}" fill="none" stroke="#f97316" stroke-width="1.5" stroke-dasharray="4 3"/>`);
  parts.push(`<path d="${contourPath(arm.contours.finalFine, ox)}" fill="none" stroke="#075985" stroke-width="2"/>`,
    `<text x="${ox}" y="${margin + plotHeight + 27}" font-family="ui-monospace,monospace" font-size="14" fill="#0f172a">${escape(label)}</text>`, `</g>`);
  return parts.join("");
}

const left = margin, right = 2 * margin + plotWidth;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<rect width="100%" height="100%" fill="#f8fafc"/>
${panel(before, left, "before: transported V fill + phi contour")}
${panel(after, right, "after: phi contour; dashed orange = before", before)}
<g font-family="ui-monospace,monospace" font-size="12" fill="#334155"><text x="${margin}" y="${height - 5}">grey = terrain capacity; pale blue = conservative V; solid = candidate phi zero contour</text></g>
</svg>\n`;
writeFileSync(outputPath, svg);
process.stdout.write(`${outputPath}\n`);
