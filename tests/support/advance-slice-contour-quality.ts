import { inspectSliceRdfDisplay } from "../../advance-lab/lenses";
import {
  buildSliceLattice,
  createSliceLattice,
  type LatticeCell,
} from "../../lib/methods/adaptive-volume/advance-slice/slice-lattice";
import type { SliceSharedRdfIsocontour } from
  "../../lib/methods/adaptive-volume/advance-slice/slice-presentation-publication";
import type { AdvanceSlice } from
  "../../lib/methods/adaptive-volume/advance-slice/slice-solver";

export type ContourPoint = readonly [number, number];

export interface DisplayContourSegment {
  readonly a: ContourPoint;
  readonly b: ContourPoint;
  readonly mode: "rdf" | "plic";
  readonly ownerCell: number;
  readonly ownerBrick: number;
  readonly ownerWidth: number;
}

export interface SphereContourQuality {
  readonly frame: number;
  readonly topologyGeneration: number;
  readonly partialCellCount: number;
  readonly minimumAcceptedMinorityFraction: number;
  readonly supportedFallbackCellCount: number;
  readonly cutCellFallbackCount: number;
  readonly nonfiniteRdfFallbackCount: number;
  readonly sharedVertexOwnerUseCount: number;
  readonly sharedVertexConflictCount: number;
  readonly maximumSharedVertexConflictFine: number;
  readonly rawSegmentCount: number;
  readonly displaySegmentCount: number;
  readonly plicSegmentCount: number;
  readonly danglingEndpointCount: number;
  readonly maximumEndpointGapFine: number;
  readonly radialRmsFine: number;
  readonly radialMaximumFine: number;
  readonly tangentRmsDegrees: number;
  readonly tangentMaximumDegrees: number;
  readonly maximumNeighbourTangentJumpDegrees: number;
  readonly maximumBrickBoundaryTangentJumpDegrees: number;
  readonly longestNearCollinearRunFine: number;
  readonly worstTangentPointFine: ContourPoint | null;
  readonly worstTangentOwnerWidth: number | null;
  readonly worstTangentAtBrickBoundary: boolean;
  readonly rawSegments: readonly DisplayContourSegment[];
  readonly displaySegments: readonly DisplayContourSegment[];
}

const pointKey = (point: ContourPoint): string =>
  `${point[0].toFixed(5)}:${point[1].toFixed(5)}`;

function cellAtFinePoint(slice: AdvanceSlice, cells: readonly LatticeCell[],
  x: number, y: number): LatticeCell | undefined {
  const canvasY = slice.ny - y;
  return cells.find(cell => x >= cell.x0 && x <= cell.x0 + cell.width
    && canvasY >= cell.y0 && canvasY <= cell.y0 + cell.height);
}

function plicSegment(slice: AdvanceSlice, cell: LatticeCell):
DisplayContourSegment | undefined {
  const source = slice.topology.accepted.cells[cell.topologyCell];
  if (!source) return undefined;
  const nx = slice.fields.interfaceNormal[2 * source.id]!;
  const ny = slice.fields.interfaceNormal[2 * source.id + 1]!;
  const offset = slice.fields.interfaceOffset[source.id]!;
  if (!(nx * nx + ny * ny > 0.5)) return undefined;
  const [x0, y0] = source.minimumFine, [x1, y1] = source.maximumFine;
  const [cx, cy] = source.centerFine;
  const points: ContourPoint[] = [];
  const add = (x: number, y: number): void => {
    if (x < x0 - 1e-7 || x > x1 + 1e-7 || y < y0 - 1e-7 || y > y1 + 1e-7) return;
    if (!points.some(point => Math.hypot(point[0] - x, point[1] - y) < 1e-6)) points.push([x, y]);
  };
  if (Math.abs(ny) > 1e-12) {
    add(x0, cy + (offset - nx * (x0 - cx)) / ny);
    add(x1, cy + (offset - nx * (x1 - cx)) / ny);
  }
  if (Math.abs(nx) > 1e-12) {
    add(cx + (offset - ny * (y0 - cy)) / nx, y0);
    add(cx + (offset - ny * (y1 - cy)) / nx, y1);
  }
  if (points.length !== 2) return undefined;
  return { a: points[0]!, b: points[1]!, mode: "plic", ownerCell: source.id,
    ownerBrick: cell.brick, ownerWidth: cell.width };
}

function rawSegments(slice: AdvanceSlice, rdf: SliceSharedRdfIsocontour,
  cells: readonly LatticeCell[], minimumY: number): DisplayContourSegment[] {
  const result: DisplayContourSegment[] = [];
  for (let index = 0; index < rdf.segmentsFine.length; index += 4) {
    const a = [rdf.segmentsFine[index]!, rdf.segmentsFine[index + 1]!] as const;
    const b = [rdf.segmentsFine[index + 2]!, rdf.segmentsFine[index + 3]!] as const;
    const midpoint = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2] as const;
    if (midpoint[1] <= minimumY) continue;
    const owner = cellAtFinePoint(slice, cells, midpoint[0], midpoint[1]);
    result.push({ a, b, mode: "rdf", ownerCell: owner?.topologyCell ?? -1,
      ownerBrick: owner?.brick ?? -1, ownerWidth: owner?.width ?? 0 });
  }
  return result;
}

function angleDifference(a: number, b: number): number {
  return Math.atan2(Math.sin(a - b), Math.cos(a - b));
}

function metrics(segments: readonly DisplayContourSegment[], center: ContourPoint,
  radius: number, brickResolution: number): Omit<SphereContourQuality,
  "frame" | "topologyGeneration" | "partialCellCount" | "supportedFallbackCellCount"
  | "minimumAcceptedMinorityFraction"
  | "cutCellFallbackCount" | "nonfiniteRdfFallbackCount" | "sharedVertexOwnerUseCount"
  | "sharedVertexConflictCount" | "maximumSharedVertexConflictFine"
  | "rawSegmentCount" | "displaySegmentCount" | "plicSegmentCount" | "rawSegments"
  | "displaySegments"> {
  const samples = segments.map(segment => {
    const midpoint = [(segment.a[0] + segment.b[0]) / 2,
      (segment.a[1] + segment.b[1]) / 2] as const;
    let tangent = Math.atan2(segment.b[1] - segment.a[1], segment.b[0] - segment.a[0]);
    const polar = Math.atan2(midpoint[1] - center[1], midpoint[0] - center[0]);
    const expected = polar + Math.PI / 2;
    if (Math.abs(angleDifference(tangent + Math.PI, expected))
      < Math.abs(angleDifference(tangent, expected))) tangent += Math.PI;
    return { segment, midpoint, polar, tangent,
      length: Math.hypot(segment.b[0] - segment.a[0], segment.b[1] - segment.a[1]),
      radialError: Math.hypot(midpoint[0] - center[0], midpoint[1] - center[1]) - radius,
      tangentError: Math.abs(angleDifference(tangent, expected)) };
  }).sort((a, b) => a.polar - b.polar);
  const endpointCounts = new Map<string, number>();
  for (const { segment } of samples) for (const point of [segment.a, segment.b])
    endpointCounts.set(pointKey(point), (endpointCounts.get(pointKey(point)) ?? 0) + 1);
  const dangling = [...endpointCounts.values()].filter(count => count % 2 !== 0).length;
  const jumps = samples.map((sample, index) => {
    const next = samples[(index + 1) % samples.length]!;
    const endpointGap = Math.min(
      Math.hypot(sample.segment.a[0] - next.segment.a[0], sample.segment.a[1] - next.segment.a[1]),
      Math.hypot(sample.segment.a[0] - next.segment.b[0], sample.segment.a[1] - next.segment.b[1]),
      Math.hypot(sample.segment.b[0] - next.segment.a[0], sample.segment.b[1] - next.segment.a[1]),
      Math.hypot(sample.segment.b[0] - next.segment.b[0], sample.segment.b[1] - next.segment.b[1]));
    const tangentJump = Math.abs(angleDifference(next.tangent, sample.tangent));
    const point = endpointGap < 1e-4 ? [
      (sample.midpoint[0] + next.midpoint[0]) / 2,
      (sample.midpoint[1] + next.midpoint[1]) / 2,
    ] as const : sample.midpoint;
    const atBrickBoundary = [point[0], point[1]].some(coordinate =>
      Math.abs(coordinate / brickResolution - Math.round(coordinate / brickResolution)) < 1e-3);
    return { endpointGap, tangentJump, atBrickBoundary };
  });
  let longestRun = 0, run = 0;
  for (let iteration = 0; iteration < samples.length * 2; iteration += 1) {
    const index = iteration % samples.length;
    run += samples[index]!.length;
    if (jumps[index]!.tangentJump >= Math.PI / 360) run = 0;
    longestRun = Math.max(longestRun, run);
    if (iteration >= samples.length && run === 0) break;
  }
  const radial = samples.map(sample => sample.radialError);
  const tangent = samples.map(sample => sample.tangentError);
  const worst = samples.reduce<typeof samples[number] | undefined>((current, sample) =>
    !current || sample.tangentError > current.tangentError ? sample : current, undefined);
  const rms = (values: readonly number[]): number => Math.sqrt(values.reduce(
    (sum, value) => sum + value * value, 0) / Math.max(1, values.length));
  return {
    danglingEndpointCount: dangling,
    maximumEndpointGapFine: Math.max(0, ...jumps.map(jump => jump.endpointGap)),
    radialRmsFine: rms(radial), radialMaximumFine: Math.max(0, ...radial.map(Math.abs)),
    tangentRmsDegrees: rms(tangent) * 180 / Math.PI,
    tangentMaximumDegrees: Math.max(0, ...tangent) * 180 / Math.PI,
    maximumNeighbourTangentJumpDegrees: Math.max(0, ...jumps.map(jump => jump.tangentJump))
      * 180 / Math.PI,
    maximumBrickBoundaryTangentJumpDegrees: Math.max(0, ...jumps.filter(jump => jump.atBrickBoundary)
      .map(jump => jump.tangentJump)) * 180 / Math.PI,
    longestNearCollinearRunFine: longestRun,
    worstTangentPointFine: worst?.midpoint ?? null,
    worstTangentOwnerWidth: worst?.segment.ownerWidth ?? null,
    worstTangentAtBrickBoundary: worst ? [worst.midpoint[0], worst.midpoint[1]].some(coordinate =>
      Math.abs(coordinate / brickResolution - Math.round(coordinate / brickResolution)) < 1e-3) : false,
  };
}

export function measureSphereDisplayContour(slice: AdvanceSlice,
  rdf: SliceSharedRdfIsocontour, center: ContourPoint, radius: number,
  minimumY = center[1] - 1.5 * radius): SphereContourQuality {
  const lattice = createSliceLattice(slice);
  buildSliceLattice(lattice, slice);
  const decisions = inspectSliceRdfDisplay(slice, lattice, rdf);
  const decisionByCell = new Map(decisions.map(decision => [decision.topologyCell, decision]));
  const fallback = new Set(decisions.filter(decision => decision.mode === "plic")
    .map(decision => decision.topologyCell));
  const partial = lattice.cells.filter(cell => decisionByCell.has(cell.topologyCell));
  const valuesByVertex = new Map<string, number[]>();
  for (const decision of decisions) for (const sample of decision.samples) {
    const key = `${sample.x}:${sample.y}`, values = valuesByVertex.get(key) ?? [];
    values.push(sample.value); valuesByVertex.set(key, values);
  }
  const wordBuffer = new ArrayBuffer(4), wordFloat = new Float32Array(wordBuffer),
    wordUint = new Uint32Array(wordBuffer);
  const word = (value: number): number => { wordFloat[0] = value; return wordUint[0]!; };
  const conflictedValues = [...valuesByVertex.values()].filter(values =>
    values.some(value => word(value) !== word(values[0]!)));
  const conflicts = conflictedValues.map(values => Math.max(...values) - Math.min(...values));
  const raw = rawSegments(slice, rdf, lattice.cells, minimumY);
  const display = raw.filter(segment => !fallback.has(segment.ownerCell));
  for (const cell of partial) {
    if (!fallback.has(cell.topologyCell)) continue;
    const segment = plicSegment(slice, cell);
    if (segment && (segment.a[1] + segment.b[1]) / 2 > minimumY) display.push(segment);
  }
  return {
    frame: slice.frame, topologyGeneration: slice.topology.accepted.generation,
    partialCellCount: partial.length,
    minimumAcceptedMinorityFraction: Math.min(1, ...partial.map(cell =>
      Math.min(cell.fill, 1 - cell.fill))),
    supportedFallbackCellCount: decisions.filter(decision =>
      decision.mode === "plic" && decision.reason !== "cut-cell").length,
    cutCellFallbackCount: decisions.filter(decision => decision.reason === "cut-cell").length,
    nonfiniteRdfFallbackCount: decisions.filter(decision => decision.reason === "nonfinite-rdf").length,
    sharedVertexOwnerUseCount: [...valuesByVertex.values()].reduce((sum, values) =>
      sum + Math.max(0, values.length - 1), 0),
    sharedVertexConflictCount: conflictedValues.length,
    maximumSharedVertexConflictFine: Math.max(0, ...conflicts),
    rawSegmentCount: raw.length, displaySegmentCount: display.length,
    plicSegmentCount: display.filter(segment => segment.mode === "plic").length,
    ...metrics(display, center, radius, 8), rawSegments: raw, displaySegments: display,
  };
}

export function contourQualityReceipt(quality: SphereContourQuality):
Omit<SphereContourQuality, "rawSegments" | "displaySegments"> {
  const { rawSegments: _raw, displaySegments: _display, ...receipt } = quality;
  return receipt;
}

export function sphereContourComparisonSvg(
  states: readonly { readonly label: string; readonly quality: SphereContourQuality }[],
  dimensions: readonly [number, number],
): string {
  const scale = 6, panelWidth = dimensions[0] * scale, panelHeight = dimensions[1] * scale;
  const line = (segment: DisplayContourSegment, xOffset: number, colour: string, width: number): string =>
    `<line x1="${xOffset + segment.a[0] * scale}" y1="${panelHeight - segment.a[1] * scale}" `
    + `x2="${xOffset + segment.b[0] * scale}" y2="${panelHeight - segment.b[1] * scale}" `
    + `stroke="${colour}" stroke-width="${width}" stroke-linecap="round"/>`;
  const panels = states.map(({ label, quality }, index) => {
    const xOffset = index * panelWidth;
    const grid = Array.from({ length: Math.floor(dimensions[0] / 8) + 1 }, (_, x) =>
      `<line x1="${xOffset + x * 48}" y1="0" x2="${xOffset + x * 48}" y2="${panelHeight}"/>`)
      .concat(Array.from({ length: Math.floor(dimensions[1] / 8) + 1 }, (_, y) =>
        `<line x1="${xOffset}" y1="${panelHeight - y * 48}" x2="${xOffset + panelWidth}" y2="${panelHeight - y * 48}"/>`))
      .join("");
    return `<g><rect x="${xOffset}" width="${panelWidth}" height="${panelHeight}" fill="#0b0f14"/>`
      + `<g stroke="#303942" stroke-width="1">${grid}</g>`
      + quality.rawSegments.map(segment => line(segment, xOffset, "#55c9ff", 3)).join("")
      + quality.displaySegments.filter(segment => segment.mode === "plic")
        .map(segment => line(segment, xOffset, "#ff5d73", 5)).join("")
      + `<text x="${xOffset + 8}" y="18" fill="white" font-family="monospace" font-size="12">${label}: `
      + `${quality.supportedFallbackCellCount} PLIC fallbacks</text></g>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${panelWidth * states.length}" `
    + `height="${panelHeight}" viewBox="0 0 ${panelWidth * states.length} ${panelHeight}">${panels}</svg>\n`;
}
