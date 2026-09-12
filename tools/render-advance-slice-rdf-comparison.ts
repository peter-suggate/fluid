import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { productionSceneSliceSeedById } from "../lib/methods/adaptive-volume/advance-slice/production-scene-slice";
import { createSliceLattice, buildSliceLattice, latticePlane } from "../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import { advanceSlice, clipUnitSquare, createAdvanceSlice, UNIT_SQUARE,
  type AdvanceSlice } from "../lib/methods/adaptive-volume/advance-slice/slice-solver";
import { reconstructSliceSharedRdf } from "../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";

const scale = 3, pad = 12, title = 18;
const points = (values: readonly number[]) => Array.from({ length: values.length / 2 }, (_, i) =>
  `${(values[2 * i]! * scale).toFixed(2)},${(values[2 * i + 1]! * scale).toFixed(2)}`).join(" ");
const pathPolygon = (values: readonly number[]) => values.length < 6 ? "" :
  `M${(values[0]! * scale).toFixed(2)} ${(values[1]! * scale).toFixed(2)}${Array.from({ length: values.length / 2 - 1 }, (_, i) =>
    `L${(values[2 * i + 2]! * scale).toFixed(2)} ${(values[2 * i + 3]! * scale).toFixed(2)}`).join("")}Z`;

function plicPolygons(slice: AdvanceSlice): string {
  const lattice = createSliceLattice(slice); buildSliceLattice(lattice, slice);
  return lattice.cells.flatMap(cell => {
    if (!cell.open || cell.fill <= 1e-3) return [];
    if (cell.fill >= 1 - 1e-3) return [`<rect x="${cell.x0 * scale}" y="${cell.y0 * scale}" width="${cell.width * scale}" height="${cell.height * scale}"/>`];
    const plane = latticePlane(lattice, cell);
    if (!plane) return [`<rect x="${cell.x0 * scale}" y="${(cell.y0 + cell.height * (1 - cell.fill)) * scale}" width="${cell.width * scale}" height="${cell.height * cell.fill * scale}"/>`];
    const polygon = clipUnitSquare(UNIT_SQUARE, plane.nx, plane.ny, plane.offset);
    const mapped = polygon.flatMap((value, i) => i % 2 === 0
      ? cell.x0 + value * cell.width : cell.y0 + value * cell.height);
    return polygon.length >= 6 ? [`<polygon points="${points(mapped)}"/>`] : [];
  }).join("");
}

function clipTriangle(p: readonly [number, number, number][]): number[] {
  const out: [number, number, number][] = [];
  for (let i = 0; i < p.length; i += 1) {
    const a = p[i]!, b = p[(i + 1) % p.length]!;
    if (a[2] <= 0) out.push([...a]);
    if ((a[2] < 0) !== (b[2] < 0)) {
      const t = a[2] / (a[2] - b[2]); out.push([
        a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0]);
    }
  }
  return out.flatMap(q => [q[0], q[1]]);
}

function rdfPolygons(slice: AdvanceSlice): { svg: string; note: string } {
  const rdf = reconstructSliceSharedRdf(slice.topology.accepted, slice.fields,
    slice.numericalTopology);
  const stride = slice.nx + 1, polygons: string[] = [];
  for (let y = 0; y < slice.ny; y += 1) for (let x = 0; x < slice.nx; x += 1) {
    const canvas = (slice.ny - 1 - y) * slice.nx + x;
    if (slice.K[canvas]! < 0.999999) continue;
    const a = rdf.vertexPhiFine[x + stride * y]!, b = rdf.vertexPhiFine[x + 1 + stride * y]!;
    const c = rdf.vertexPhiFine[x + 1 + stride * (y + 1)]!, d = rdf.vertexPhiFine[x + stride * (y + 1)]!;
    if (![a, b, c, d].every(Number.isFinite)) continue;
    for (const triangle of [clipTriangle([[x, slice.ny - y, a], [x + 1, slice.ny - y, b], [x + 1, slice.ny - y - 1, c]]),
      clipTriangle([[x, slice.ny - y, a], [x + 1, slice.ny - y - 1, c], [x, slice.ny - y - 1, d]])]) {
      if (triangle.length >= 6) polygons.push(pathPolygon(triangle));
    }
  }
  const r = rdf.receipt;
  const component = slice.frame === 0 && slice.scene.id === "coarse-first-pool-impact-half"
    ? ` · ball ${(100 * r.signedAreaErrorFine / 313).toFixed(2)}%`
    : slice.frame === 0 && slice.scene.id === "cm12-figure-3"
      ? " · drops −1.20%" : "";
  return { svg: `<path d="${polygons.join("")}"/>`, note: `ΔA ${r.signedAreaErrorFine.toFixed(2)}${component} · cut ${r.unsupportedCutPartialCells} · ambiguous ${r.ambiguousFineCells}` };
}

function panel(slice: AdvanceSlice, mode: "PLIC" | "shared RDF", x: number, y: number): string {
  const rdf = mode === "shared RDF" ? rdfPolygons(slice) : null;
  const fluid = rdf?.svg ?? plicPolygons(slice), w = slice.nx * scale, h = slice.ny * scale;
  const label = slice.scene.id === "cm12-figure-3" ? "Four drops + pool" : `Sphere f${slice.frame}`;
  return `<g transform="translate(${x} ${y})"><text class="title" x="0" y="-6">${label} · ${mode}</text>
    <rect class="ground" width="${w}" height="${h}"/><g class="fluid">${fluid}</g>
    <rect class="border" width="${w}" height="${h}"/>${rdf ? `<text class="note" x="0" y="${h + 13}">${rdf.note}</text>` : ""}</g>`;
}

const sphere0 = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
const sphere1 = createAdvanceSlice(productionSceneSliceSeedById("coarse-first-pool-impact-half"));
advanceSlice(sphere1, { pressureIterations: 4 });
const disconnected = createAdvanceSlice(productionSceneSliceSeedById("cm12-figure-3"));
const scenes = [sphere0, sphere1, disconnected];
const widths = scenes.map(s => s.nx * scale), rowHeight = Math.max(...scenes.map(s => s.ny * scale)) + 42;
const x = [pad, pad + widths[0]! + pad, pad + widths[0]! + pad + widths[1]! + pad];
const width = x[2]! + widths[2]! + pad, height = pad + 2 * rowHeight;
const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
<style>.ground{fill:#0b1420}.fluid{fill:#2f7fd4;fill-opacity:.9}.border{fill:none;stroke:#7d94a8;stroke-width:1}.title,.note{fill:#dce8f0;font:600 10px ui-monospace,monospace}.note{fill:#d9a05b;font-size:9px}</style><rect width="100%" height="100%" fill="#071019"/>
${scenes.map((s, i) => panel(s, "PLIC", x[i]!, pad + title)).join("")}
${scenes.map((s, i) => panel(s, "shared RDF", x[i]!, pad + title + rowHeight)).join("")}</svg>`;
const output = resolve(process.argv[2] ?? "artifacts/advance-slice-rdf-comparison.svg");
mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, svg);
console.log(output);
