import { DecodedPhysicsPublication, PhysicsPlane } from "./publication";
import type { PhysicsRevision } from "./protocol";
import { BRICK_FINE_CELLS } from "../core/sparse-brick-geometry";

/** The lab draws the same brick the 3-D solver binds on — see `BRICK_FINE_CELLS`. */
export const ADVANCE_BRICK_FINE = BRICK_FINE_CELLS;
export const ADVANCE_RUNGS = Object.freeze([1, 2, 4, 8] as const);

export interface AdvanceGraphCell {
  readonly id: number; readonly minimum: readonly number[]; readonly maximum: readonly number[];
  readonly center: readonly number[]; readonly widths: readonly number[]; readonly measure: number;
  readonly brickKey?: number;
}
export interface AdvanceGraphRow {
  readonly id: number; readonly axis: number; readonly kind: string; readonly center: readonly number[];
  readonly measure: number; readonly terms: readonly { readonly cellId: number; readonly coefficient: number }[];
}
export interface AdvanceGraphSubface {
  readonly id: number; readonly rowId: number; readonly axis: number; readonly center: readonly number[];
  readonly measure: number;
}
export interface AdvanceBrick {
  readonly id: number; readonly key: number; readonly coordinate: readonly number[];
  readonly spanBricks: number; readonly resolution: number; readonly active: boolean;
}
export interface AdvanceGraph {
  readonly schemaVersion: number; readonly dimension: 2; readonly dimensions: readonly number[];
  readonly topologyGeneration: number; readonly cells: readonly AdvanceGraphCell[];
  readonly rows: readonly AdvanceGraphRow[]; readonly subfaces: readonly AdvanceGraphSubface[];
  readonly bricks: readonly AdvanceBrick[];
}

export interface AdvancePlane {
  readonly nx: number; readonly ny: number; readonly clipNx: number;
  readonly clipNy: number; readonly offset: number;
}
export interface AdvanceCellView {
  readonly x0: number; readonly y0: number; readonly width: number; readonly height: number;
  readonly size: number; readonly brick: number; readonly topologyCell: number;
  readonly volume: number; readonly capacity: number; readonly fill: number;
  readonly open: boolean; readonly plane: AdvancePlane | null;
}
export interface AdvanceLattice {
  readonly nx: number; readonly ny: number; readonly cells: readonly AdvanceCellView[];
}
export interface AdvanceRdfView {
  readonly vertexPhiFine: Float32Array; readonly segmentsFine: Float32Array;
  readonly receipt?: Readonly<Record<string, number>>;
}
export interface AdvanceMarker { readonly x: number; readonly y: number; readonly alive: boolean }
export interface AdvanceSceneView {
  readonly id: string; readonly label: string; readonly dimensions: readonly number[];
  readonly cellSizeM: number; readonly hasStaticWorld: boolean; readonly hasInflow: boolean;
  readonly hasRigidBodies: boolean; readonly unfrozen: boolean;
  readonly limitations: readonly string[];
}
export interface AdvanceView {
  readonly revision: PhysicsRevision;
  readonly graph: AdvanceGraph;
  readonly scene: AdvanceSceneView;
  readonly nx: number; readonly ny: number; readonly bx: number; readonly by: number;
  readonly lattice: AdvanceLattice;
  readonly liquidVolumeFine: Float32Array;
  readonly capacityFine: Float32Array;
  readonly previousLiquidVolumeFine: Float32Array;
  readonly faceVelocityXFine: Float32Array;
  readonly faceVelocityYFine: Float32Array;
  readonly faceVelocityXBeforePressure: Float32Array;
  readonly faceVelocityYBeforePressure: Float32Array;
  readonly pressureFine: Float32Array;
  readonly divergenceFine: Float32Array;
  readonly extensionFine: Uint8Array;
  readonly limitedFluxXFine: Float32Array;
  readonly limitedFluxYFine: Float32Array;
  readonly fluxLimitedXFine: Uint8Array;
  readonly fluxLimitedYFine: Uint8Array;
  readonly materialFine: Float32Array;
  readonly brickRung: Int8Array;
  readonly previousBrickRung: Int8Array;
  readonly brickActivity: Float32Array;
  readonly markers: readonly AdvanceMarker[];
  readonly rdf: AdvanceRdfView;
  readonly receipt: Readonly<Record<string, unknown>>;
  readonly metadata: Readonly<Record<string, unknown>>;
}

const f32 = (publication: DecodedPhysicsPublication, id: number, required = true): Float32Array => {
  const value = publication.plane(id as never);
  if (value instanceof Float32Array) return value;
  if (required) throw new TypeError(`Advance publication is missing f32 plane ${id}`);
  return new Float32Array();
};
const u8 = (publication: DecodedPhysicsPublication, id: number, required = true): Uint8Array => {
  const value = publication.plane(id as never);
  if (value instanceof Uint8Array) return value;
  if (required) throw new TypeError(`Advance publication is missing u8 plane ${id}`);
  return new Uint8Array();
};
const record = (value: unknown, label: string): Record<string, unknown> => {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError(`${label} must be an object`);
  return value as Record<string, unknown>;
};
const finite = (value: unknown, label: string): number => {
  if (typeof value !== "number" || !Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
  return value;
};
const own = <T extends Float32Array | Uint8Array>(value: T): T => value.slice() as T;
const expectLength = <T extends Float32Array | Uint8Array>(value: T, length: number,
  label: string): T => {
  if (value.length !== length) throw new RangeError(`${label} plane has ${value.length} values; expected ${length}`);
  return value;
};

export function advanceCell(view: Pick<AdvanceView, "nx">, x: number, y: number): number {
  return y * view.nx + x;
}
export function advanceRowX(view: Pick<AdvanceView, "nx">, x: number, y: number): number {
  return y * (view.nx + 1) + x;
}
export function advanceRowY(view: Pick<AdvanceView, "nx">, x: number, y: number): number {
  return y * view.nx + x;
}
export function advanceCellAt(lattice: AdvanceLattice, view: Pick<AdvanceView, "nx" | "ny">,
  x: number, y: number): AdvanceCellView | null {
  if (x < 0 || y < 0 || x >= view.nx || y >= view.ny) return null;
  return lattice.cells.find(cell => x >= cell.x0 && x < cell.x0 + cell.width
    && y >= cell.y0 && y < cell.y0 + cell.height) ?? null;
}
export function advanceCellPlane(_lattice: AdvanceLattice, cell: AdvanceCellView): AdvancePlane | null {
  return cell.fill > 1e-3 && cell.fill < 1 - 1e-3 ? cell.plane : null;
}

export const UNIT_SQUARE: readonly number[] = Object.freeze([0, 0, 1, 0, 1, 1, 0, 1]);
export function clipUnitSquare(polygon: readonly number[], ax: number, ay: number, d: number): number[] {
  const output: number[] = [];
  for (let i = 0; i < polygon.length; i += 2) {
    const j = (i + 2) % polygon.length;
    const x0 = polygon[i]!, y0 = polygon[i + 1]!, x1 = polygon[j]!, y1 = polygon[j + 1]!;
    const s0 = ax * x0 + ay * y0 - d, s1 = ax * x1 + ay * y1 - d;
    if (s0 <= 0) output.push(x0, y0);
    if (s0 * s1 < 0) { const t = s0 / (s0 - s1); output.push(x0 + t * (x1 - x0), y0 + t * (y1 - y0)); }
  }
  return output;
}
export type AdvanceRdfVertex = readonly [number, number, number];
export function advanceRdfTriangles(x: number, y: number, a: number, b: number, c: number, d: number):
readonly (readonly AdvanceRdfVertex[])[] {
  const corners: readonly AdvanceRdfVertex[] = [[x, y, a], [x + 1, y, b], [x + 1, y + 1, c], [x, y + 1, d]];
  const centre: AdvanceRdfVertex = [x + 0.5, y + 0.5, ((a + c) + (b + d)) * 0.25];
  return corners.map((corner, index) => [corner, corners[(index + 1) % 4]!, centre]);
}

export function clippedScalarTriangle(points: readonly AdvanceRdfVertex[]): number[] {
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


function rasterCells(graph: AdvanceGraph, values: Float32Array, nx: number, ny: number): Float32Array {
  const output = new Float32Array(nx * ny);
  for (const cell of graph.cells) for (let y = Math.floor(cell.minimum[1]!); y < Math.ceil(cell.maximum[1]!); y++)
    for (let x = Math.floor(cell.minimum[0]!); x < Math.ceil(cell.maximum[0]!); x++)
      if (x >= 0 && x < nx && y >= 0 && y < ny) output[advanceCell({ nx }, x, ny - 1 - y)] = values[cell.id] ?? 0;
  return output;
}
function rasterRows(graph: AdvanceGraph, values: Float32Array, nx: number, ny: number) {
  const x = new Float32Array((nx + 1) * ny), y = new Float32Array(nx * (ny + 1));
  for (const row of graph.rows) {
    const value = values[row.id] ?? 0;
    if (row.axis === 0) { const fx = Math.round(row.center[0]!), cy = ny - 1 - Math.floor(row.center[1]!);
      if (fx >= 0 && fx <= nx && cy >= 0 && cy < ny) x[advanceRowX({ nx }, fx, cy)] = value; }
    else { const fx = Math.floor(row.center[0]!), cy = ny - Math.round(row.center[1]!);
      if (fx >= 0 && fx < nx && cy >= 0 && cy <= ny) y[advanceRowY({ nx }, fx, cy)] = -value; }
  }
  return { x, y };
}
function rasterSubfaces(graph: AdvanceGraph, values: Float32Array, nx: number, ny: number) {
  const x = new Float32Array((nx + 1) * ny), y = new Float32Array(nx * (ny + 1));
  for (const face of graph.subfaces) {
    const value = values[face.id] ?? 0;
    if (face.axis === 0) { const fx = Math.round(face.center[0]!), cy = ny - 1 - Math.floor(face.center[1]!);
      if (fx >= 0 && fx <= nx && cy >= 0 && cy < ny) x[advanceRowX({ nx }, fx, cy)] += value; }
    else { const fx = Math.floor(face.center[0]!), cy = ny - Math.round(face.center[1]!);
      if (fx >= 0 && fx < nx && cy >= 0 && cy <= ny) y[advanceRowY({ nx }, fx, cy)] -= value; }
  }
  return { x, y };
}

export function createAdvanceView(publication: DecodedPhysicsPublication,
  cachedGraph?: AdvanceGraph, authored?: { id: string; label: string; limitations?: readonly string[];
    document?: unknown }): AdvanceView {
  const metadata = publication.metadata, graphValue = publication.jsonPlane(PhysicsPlane.GraphJson);
  const graph = (graphValue ? record(graphValue, "Advance graph") : cachedGraph) as AdvanceGraph | undefined;
  if (!graph || graph.dimension !== 2 || !Array.isArray(graph.cells) || !Array.isArray(graph.rows)
    || !Array.isArray(graph.subfaces) || !Array.isArray(graph.bricks)) throw new TypeError("Advance publication has no complete 2D graph");
  const nx = Math.trunc(finite(graph.dimensions[0], "graph width"));
  const ny = Math.trunc(finite(graph.dimensions[1], "graph height"));
  const sceneMetadata = record(metadata.scene, "Advance scene metadata");
  const frame = record(sceneMetadata.frame, "Advance scene frame");
  const sourceDimensions = frame.sourceDimensions as unknown;
  const canvasDimensions = Array.isArray(sourceDimensions) ? sourceDimensions : graph.dimensions;
  const canvasNx = Math.trunc(finite(canvasDimensions[0], "source width"));
  const canvasNy = Math.trunc(finite(canvasDimensions[1], "source height"));
  if (canvasNx !== nx || canvasNy !== ny) throw new RangeError("Advance graph and source frame dimensions disagree");
  const receipt = record(metadata.receipt, "Advance receipt");
  const density = expectLength(own(f32(publication, PhysicsPlane.Density)), graph.cells.length, "density");
  const capacity = expectLength(own(f32(publication, PhysicsPlane.Capacity)), graph.cells.length, "capacity");
  const pressure = expectLength(own(f32(publication, PhysicsPlane.Pressure)), graph.cells.length, "pressure");
  const rhs = expectLength(own(f32(publication, PhysicsPlane.PressureRhs)), graph.cells.length, "pressure RHS");
  const extension = expectLength(own(u8(publication, PhysicsPlane.ExtensionDepth)), graph.cells.length, "extension");
  const normals = expectLength(own(f32(publication, PhysicsPlane.InterfaceNormal)), graph.cells.length * 2, "interface normal");
  const offsets = expectLength(own(f32(publication, PhysicsPlane.InterfaceOffset)), graph.cells.length, "interface offset");
  const currentRows = rasterRows(graph, expectLength(own(f32(publication, PhysicsPlane.FaceVelocity)),
    graph.rows.length, "face velocity"), nx, ny);
  const limited = rasterSubfaces(graph, expectLength(own(f32(publication, PhysicsPlane.LimitedFlux)),
    graph.subfaces.length, "limited flux"), nx, ny);
  const high = rasterSubfaces(graph, expectLength(own(f32(publication, PhysicsPlane.HighFlux)),
    graph.subfaces.length, "high flux"), nx, ny);
  const fluxLimitedXFine = Uint8Array.from(limited.x, (value, index) =>
    Math.abs(value - high.x[index]!) > 1e-7 ? 1 : 0);
  const fluxLimitedYFine = Uint8Array.from(limited.y, (value, index) =>
    Math.abs(value - high.y[index]!) > 1e-7 ? 1 : 0);
  const liquidVolumeFine = rasterCells(graph, density, nx, ny);
  const compactPrevious = own(f32(publication, PhysicsPlane.DensityBefore));
  const previousLiquidVolumeFine = compactPrevious.length === nx * ny
    ? compactPrevious : rasterCells(graph, compactPrevious, nx, ny);
  const capacityPublished = own(f32(publication, PhysicsPlane.CapacityFine, false));
  const capacityFine = capacityPublished.length === nx * ny ? capacityPublished : rasterCells(graph, capacity, nx, ny);
  const pressureFine = rasterCells(graph, pressure, nx, ny);
  const divergenceFine = rasterCells(graph, rhs, nx, ny);
  const extensionFine = Uint8Array.from(rasterCells(graph, Float32Array.from(extension,
    value => value === 255 ? 0 : value), nx, ny));
  const materialPublished = own(f32(publication, PhysicsPlane.MaterialFine));
  if (materialPublished.length !== nx * ny) throw new RangeError("Advance material plane has the wrong size");
  const bx = Math.ceil(nx / ADVANCE_BRICK_FINE), by = Math.ceil(ny / ADVANCE_BRICK_FINE);
  const brickRung = new Int8Array(bx * by);
  for (const brick of graph.bricks) {
    const x = brick.coordinate[0]!, cy = by - 1 - brick.coordinate[1]!;
    if (x >= 0 && x < bx && cy >= 0 && cy < by) brickRung[cy * bx + x] = brick.active === false
      ? 0 : Math.max(0, ADVANCE_RUNGS.indexOf(brick.resolution as never));
  }
  const previousBrickRung = Int8Array.from(own(u8(publication, PhysicsPlane.BrickResolutionBefore)));
  const brickActivity = own(f32(publication, PhysicsPlane.BrickActivity));
  if (previousBrickRung.length !== bx * by || brickActivity.length !== bx * by) {
    throw new RangeError("Advance brick history planes have the wrong size");
  }
  const beforeX = own(f32(publication, PhysicsPlane.VelocityXBeforePressure));
  const beforeY = own(f32(publication, PhysicsPlane.VelocityYBeforePressure));
  if (beforeX.length !== (nx + 1) * ny || beforeY.length !== nx * (ny + 1)) {
    throw new RangeError("Advance pre-pressure velocity planes have the wrong size");
  }
  const cells = graph.cells.map(cell => {
    const width = cell.widths[0]!, height = cell.widths[1]!, y0 = ny - cell.maximum[1]!;
    const area = cell.measure, cellCapacity = Math.fround(capacity[cell.id]! * area);
    const volume = Math.fround(density[cell.id]! * area);
    const fill = cellCapacity > 1e-8 ? Math.fround(volume / cellCapacity) : 0;
    const px = normals[2 * cell.id]!, py = normals[2 * cell.id + 1]!, normalY = -py;
    const clipNx = px * width, clipNy = normalY * height;
    const plane = px === 0 && py === 0 ? null : { nx: px, ny: normalY, clipNx, clipNy,
      offset: offsets[cell.id]! + 0.5 * (clipNx + clipNy) };
    return Object.freeze({ x0: cell.minimum[0]!, y0, width, height, size: width,
      brick: cell.brickKey ?? 0, topologyCell: cell.id, volume, capacity: cellCapacity,
      fill, open: cellCapacity > 1e-8, plane });
  });
  const tracer = own(f32(publication, PhysicsPlane.Tracers, false)), markers: AdvanceMarker[] = [];
  for (let at = 0; at + 3 < tracer.length; at += 4) markers.push(Object.freeze({
    x: tracer[at]!, y: ny - tracer[at + 1]!, alive: tracer[at + 3]! >= 0.5,
  }));
  const surface = record(metadata.surface, "Advance RDF receipt") as Readonly<Record<string, number>>;
  const rdf = Object.freeze({ vertexPhiFine: own(f32(publication, PhysicsPlane.RdfVertices)),
    segmentsFine: own(f32(publication, PhysicsPlane.RdfSegments)), receipt: surface });
  const document = authored?.document as { systems?: { fluid?: boolean } } | undefined;
  const scene: AdvanceSceneView = Object.freeze({ id: authored?.id ?? String(sceneMetadata.id ?? "scene"),
    label: authored?.label ?? String(sceneMetadata.label ?? "Scene"), dimensions: graph.dimensions,
    cellSizeM: finite(sceneMetadata.cellSizeM, "scene cell size"),
    hasStaticWorld: Boolean(sceneMetadata.hasStaticWorld), hasInflow: Boolean(sceneMetadata.hasInflow),
    hasRigidBodies: Boolean(sceneMetadata.hasRigidBodies), unfrozen: document?.systems?.fluid !== false,
    limitations: authored?.limitations ?? [] });
  return Object.freeze({ revision: publication.metadata.revision as unknown as PhysicsRevision,
    graph, scene, nx, ny, bx, by, lattice: Object.freeze({ nx, ny, cells }),
    liquidVolumeFine, capacityFine, previousLiquidVolumeFine,
    faceVelocityXFine: currentRows.x, faceVelocityYFine: currentRows.y,
    faceVelocityXBeforePressure: beforeX, faceVelocityYBeforePressure: beforeY,
    pressureFine, divergenceFine, extensionFine, limitedFluxXFine: limited.x,
    limitedFluxYFine: limited.y, fluxLimitedXFine, fluxLimitedYFine,
    materialFine: materialPublished,
    brickRung, previousBrickRung, brickActivity, markers, rdf, receipt, metadata });
}
