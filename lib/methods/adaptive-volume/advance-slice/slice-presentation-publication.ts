import { CM12_LIQUID_ISOVALUE } from "../../../core/cm12-numerics";
import type { SparseAdaptiveMassAtlas } from "../sparse-brick-atlas";
import type { SliceSceneSeed } from "./slice-scene-seed";
import type { SliceNumericalFields, SliceNumericalTopology } from "./slice-stage-numerics";
import type { SliceTopology, SliceTopologyBrick, SliceTopologyCell } from "./slice-topology";
import { sliceRdfTriangles } from "./slice-rdf-triangulation";

const f = Math.fround;
const PAGE_RESOLUTION = 8;

export interface SlicePresentationColumnAuthority {
  /** Production P+2 halo layout, reduced from x/z columns to centre-z x. */
  readonly axis: number;
  readonly heightFine: Float32Array;
  /** 1 when the floor-connected single-surface receipt is valid. */
  readonly valid: Uint8Array;
  readonly represented: Uint8Array;
}

export interface SlicePresentationPage {
  readonly id: number;
  readonly key: number;
  readonly sourceBrickId: number;
  readonly sourceBrickKey: number;
  /** Generation-zero atlas index, or -1 for a runtime-created WDR page. */
  readonly sourceAtlasBrick: number;
  readonly coordinate: readonly [number, number];
  readonly spanBricks: number;
  readonly acceptedResolution: number;
  readonly sampleScale: number;
  readonly generation: number;
  /** Production payload layout: binary16 phi in low bits, flags in high bits. */
  readonly samples: Uint32Array;
  readonly phi: Float32Array;
  readonly columns: SlicePresentationColumnAuthority;
  readonly wet: boolean;
}

export interface SlicePresentationBank {
  readonly generation: number;
  readonly topologyGeneration: number;
  readonly pages: readonly SlicePresentationPage[];
  readonly pageByKey: ReadonlyMap<number, SlicePresentationPage>;
}

export interface SlicePresentationDenseReadback {
  readonly dimensions: readonly [number, number];
  readonly phi: Float32Array;
  readonly flags: Uint16Array;
  readonly pageOwner: Int32Array;
}

export interface SlicePresentationFault {
  readonly code: "field-shape" | "duplicate-page" | "nonfinite-sample" | "source-provenance";
  readonly page: number;
  readonly sample: number;
}

export interface SlicePresentationReceipt {
  readonly accepted: boolean;
  readonly publicationGeneration: number;
  readonly topologyGeneration: number;
  readonly pageCount: number;
  readonly sampleCount: number;
  readonly wetPageCount: number;
  readonly columnPageCount: number;
  readonly sourceAtlasGeneration: number;
  /** Published extensive liquid amount in finest-cell squared units. */
  readonly liquidAreaFine: number;
  readonly liquidCentroidFine: readonly [number, number] | null;
  readonly liquidBoundsFine: readonly [number, number, number, number] | null;
  readonly rigidBodies: readonly SlicePublishedRigidBody[];
  readonly fault: SlicePresentationFault | null;
}

export interface SlicePublishedRigidBody {
  readonly id: string;
  readonly position_m: readonly [number, number, number];
  readonly orientation: readonly [number, number, number, number];
  readonly linearVelocity_m_s: readonly [number, number, number];
  readonly angularVelocity_rad_s: readonly [number, number, number];
  readonly load_N: readonly [number, number, number];
  readonly torque_Nm: readonly [number, number, number];
}

export interface SlicePresentationFrameInput {
  readonly rigidBodies?: readonly SlicePublishedRigidBody[];
}

export interface SlicePresentationState {
  readonly dimensions: readonly [number, number];
  readonly mode: "off" | "on" | "auto";
  readonly cellSize: number;
  readonly hasSource: boolean;
  readonly accepted: SlicePresentationBank;
  readonly candidate?: SlicePresentationBank;
  readonly publicationGeneration: number;
  readonly denseReadback: SlicePresentationDenseReadback;
  readonly receipt: SlicePresentationReceipt;
}

/**
 * A diagnostic view derived from the accepted VOF/PLIC authority using the
 * reconstructed-distance-function weights of Scheufler and Roenby. It is a
 * shared C0 isocontour, not plicRDF transport and not a volume authority.
 */
export interface SliceSharedRdfReceipt {
  readonly interfaceCells: number;
  readonly unsupportedCutPartialCells: number;
  /** Inactive allocated cells sampled only to complete Eq. 11's air stencil. */
  readonly inactiveAirGhostSamples: number;
  /** Reflected samples used where no allocated inactive cell describes the rung. */
  readonly reflectedAirGhostSamples: number;
  readonly ambiguousFineCells: number;
  readonly unresolvedFineCells: number;
  readonly exactAreaFine: number;
  readonly representedAreaFine: number;
  readonly signedAreaErrorFine: number;
  readonly meanAbsolutePartialCellErrorFine: number;
  readonly maximumAbsolutePartialCellErrorFine: number;
}

export interface SliceSharedRdfIsocontour {
  readonly dimensions: readonly [number, number];
  /** One scalar at every shared finest-lattice vertex, y-up like topology. */
  readonly vertexPhiFine: Float32Array;
  /** Segment endpoint tuples x0,y0,x1,y1 in finest-lattice coordinates. */
  readonly segmentsFine: Float32Array;
  readonly receipt: SliceSharedRdfReceipt;
}

export interface SlicePresentationPublication {
  readonly state: SlicePresentationState;
  readonly receipt: SlicePresentationReceipt;
  readonly denseReadback: SlicePresentationDenseReadback;
}

function span(brick: SliceTopologyBrick): number { return brick.spanBricks ?? 1; }

function pageKey(brick: SliceTopologyBrick, dimensions: readonly [number, number]): number {
  const bx = Math.ceil(dimensions[0] / PAGE_RESOLUTION);
  return brick.coordinate[0] + bx * brick.coordinate[1];
}

const OWNER_LOOKUP = new WeakMap<SliceTopology, Int32Array>();
function ownerAt(topology: SliceTopology, x: number, y: number): SliceTopologyCell | undefined {
  const [nx, ny] = topology.dimensions, ix = Math.floor(x), iy = Math.floor(y);
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return undefined;
  let lookup = OWNER_LOOKUP.get(topology);
  if (!lookup) {
    lookup = new Int32Array(nx * ny).fill(-1);
    for (const cell of topology.cells)
      for (let cy = cell.minimumFine[1]; cy < cell.maximumFine[1]; cy += 1)
        for (let cx = cell.minimumFine[0]; cx < cell.maximumFine[0]; cx += 1)
          lookup[cx + nx * cy] = cell.id;
    OWNER_LOOKUP.set(topology, lookup);
  }
  const id = lookup[ix + nx * iy]!;
  return id >= 0 ? topology.cells[id] : undefined;
}

function phiAt(cell: SliceTopologyCell | undefined, fields: SliceNumericalFields,
  x: number, y: number, cellSize: number): number {
  if (!cell) return f(4 * cellSize);
  const capacity = fields.capacity[cell.id]!;
  if (!(capacity > 1e-8)) return f(4 * cellSize);
  const fill = Math.max(0, Math.min(1, f(fields.density[cell.id]! / Math.max(capacity, 1e-6))));
  const fallback = f(f(CM12_LIQUID_ISOVALUE - fill) * f(4 * cellSize));
  const nx = fields.interfaceNormal[2 * cell.id] ?? 0;
  const ny = fields.interfaceNormal[2 * cell.id + 1] ?? 0;
  if (!(f(nx * nx + ny * ny) > 0.5)) return fallback;
  const signedFine = f(f(nx * f(x - cell.centerFine[0]))
    + f(ny * f(y - cell.centerFine[1])) - fields.interfaceOffset[cell.id]!);
  return f(signedFine * cellSize);
}

function f16(value: number): number {
  const buffer = new ArrayBuffer(4), float = new Float32Array(buffer), word = new Uint32Array(buffer);
  float[0] = value;
  const bits = word[0]!, sign = (bits >>> 16) & 0x8000;
  let exponent = ((bits >>> 23) & 0xff) - 127 + 15;
  let mantissa = bits & 0x7fffff;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    mantissa = (mantissa | 0x800000) >>> (1 - exponent);
    return sign | ((mantissa + 0xfff + ((mantissa >>> 13) & 1)) >>> 13);
  }
  if (exponent >= 31) return sign | (mantissa ? 0x7e00 : 0x7c00);
  mantissa += 0xfff + ((mantissa >>> 13) & 1);
  if (mantissa & 0x800000) { mantissa = 0; exponent += 1; }
  return sign | (exponent >= 31 ? 0x7c00 : exponent << 10) | (mantissa >>> 13);
}

function columnAuthority(topology: SliceTopology, fields: SliceNumericalFields,
  brick: SliceTopologyBrick, mode: SlicePresentationState["mode"], hasSource: boolean):
  SlicePresentationColumnAuthority {
  const axis = PAGE_RESOLUTION + 2;
  const heightFine = new Float32Array(axis).fill(-1);
  const valid = new Uint8Array(axis), represented = new Uint8Array(axis);
  const enabled = !hasSource && (mode === "on"
    || mode === "auto" && brick.resolution < PAGE_RESOLUTION * span(brick));
  if (!enabled) return { axis, heightFine, valid, represented };
  const originX = brick.coordinate[0] * PAGE_RESOLUTION;
  const sampleScale = span(brick);
  for (let column = 0; column < axis; column += 1) {
    const x = Math.max(0, Math.min(topology.dimensions[0] - 1,
      originX + (column - 1) * sampleScale));
    let height = 0, seenAir = false, monotone = true, hasLiquid = false;
    for (let y = 0; y < topology.dimensions[1]; y += 1) {
      const cell = ownerAt(topology, x + 0.5, y + 0.5);
      const capacity = cell ? fields.capacity[cell.id]! : 0;
      const fill = capacity > 1e-8
        ? Math.max(0, Math.min(1, fields.density[cell!.id]! / Math.max(capacity, 1e-6))) : 0;
      if (fill > 1e-6) {
        hasLiquid = true;
        if (seenAir) monotone = false;
        height = f(height + fill);
      } else if (hasLiquid) seenAir = true;
    }
    represented[column] = hasLiquid ? 1 : 0;
    valid[column] = hasLiquid && monotone ? 1 : 0;
    heightFine[column] = valid[column] ? f(height) : hasLiquid ? -2 : -1;
  }
  return { axis, heightFine, valid, represented };
}

function sourceAtlasIndex(atlas: SparseAdaptiveMassAtlas | undefined,
  brick: SliceTopologyBrick): number {
  if (!atlas) return brick.id;
  return atlas.bricks.findIndex(source => source.key === brick.key);
}

function buildBank(topology: SliceTopology, fields: SliceNumericalFields,
  generation: number, mode: SlicePresentationState["mode"], cellSize: number,
  hasSource: boolean, sourceAtlas?: SparseAdaptiveMassAtlas):
  { bank?: SlicePresentationBank; fault: SlicePresentationFault | null } {
  if (fields.density.length !== topology.cells.length
    || fields.capacity.length !== topology.cells.length) {
    return { fault: { code: "field-shape", page: -1, sample: -1 } };
  }
  const pages: SlicePresentationPage[] = [];
  const cellsByBrick = new Map<number, SliceTopologyCell[]>();
  for (const cell of topology.cells) {
    const list = cellsByBrick.get(cell.brickKey) ?? [];
    list.push(cell); cellsByBrick.set(cell.brickKey, list);
  }
  for (const brick of topology.bricks) {
    if (brick.active === false) continue;
    const key = pageKey(brick, topology.dimensions);
    if (pages.some(page => page.key === key)) {
      return { fault: { code: "duplicate-page", page: pages.length, sample: -1 } };
    }
    const source = sourceAtlasIndex(sourceAtlas, brick);
    const authored = source >= 0 ? sourceAtlas?.bricks[source] : undefined;
    if (authored && (authored.key !== brick.key
      || authored.coordinate[0] !== brick.coordinate[0]
      || authored.coordinate[1] !== brick.coordinate[1]
      || (authored.spanBricks ?? 1) !== span(brick))) {
      return { fault: { code: "source-provenance", page: pages.length, sample: -1 } };
    }
    const scale = PAGE_RESOLUTION * span(brick) / brick.resolution;
    const sampleScale = span(brick);
    const origin = [brick.coordinate[0] * PAGE_RESOLUTION,
      brick.coordinate[1] * PAGE_RESOLUTION] as const;
    const samples = new Uint32Array(PAGE_RESOLUTION ** 2);
    const phi = new Float32Array(PAGE_RESOLUTION ** 2);
    const columns = columnAuthority(topology, fields, brick, mode, hasSource);
    let mass = 0, wet = false;
    for (const cell of cellsByBrick.get(brick.key) ?? []) {
      const fill = fields.density[cell.id]! / Math.max(fields.capacity[cell.id]!, 1e-6);
      wet ||= fill > CM12_LIQUID_ISOVALUE;
      mass = f(mass + f(Math.max(0, fields.density[cell.id]!) * cell.volumeFineCells));
    }
    wet &&= mass >= 1e-6;
    for (let localY = 0; localY < PAGE_RESOLUTION; localY += 1)
      for (let localX = 0; localX < PAGE_RESOLUTION; localX += 1) {
        const index = localX + PAGE_RESOLUTION * localY;
        const qx = origin[0] + localX * sampleScale;
        const qy = origin[1] + localY * sampleScale;
        let value: number;
        const column = columns.valid[localX + 1] !== 0 ? columns.heightFine[localX + 1]! : -1;
        if (column >= 0 && sampleScale === 1) value = f(f(qy + 0.5 - column) * cellSize);
        else value = phiAt(ownerAt(topology, qx + 0.5, qy + 0.5), fields,
          qx + 0.5, qy + 0.5, cellSize);
        if (!Number.isFinite(value)) {
          return { fault: { code: "nonfinite-sample", page: pages.length, sample: index } };
        }
        phi[index] = value;
        const floorContinuation = qy === 0 && column > 1e-3 ? 2 : 0;
        const flags = 1 | floorContinuation | (value < 0 ? 16 : 0)
          | (Math.log2(Math.max(1, scale)) << 8);
        samples[index] = (f16(value) & 0xffff) | ((flags & 0xffff) << 16);
      }
    pages.push(Object.freeze({ id: pages.length, key, sourceBrickId: brick.id,
      sourceBrickKey: brick.key, sourceAtlasBrick: source,
      coordinate: brick.coordinate, spanBricks: span(brick),
      acceptedResolution: brick.resolution, sampleScale, generation,
      samples, phi, columns, wet }));
  }
  pages.sort((a, b) => a.key - b.key);
  const normalized = pages.map((page, id) => page.id === id ? page : Object.freeze({ ...page, id }));
  return { bank: Object.freeze({ generation, topologyGeneration: topology.generation,
    pages: Object.freeze(normalized), pageByKey: new Map(normalized.map(page => [page.key, page])) }),
    fault: null };
}

function denseReadback(bank: SlicePresentationBank, topology: SliceTopology,
  fields: SliceNumericalFields, cellSize: number): SlicePresentationDenseReadback {
  const [nx, ny] = topology.dimensions, phi = new Float32Array(nx * ny);
  const flags = new Uint16Array(nx * ny), pageOwner = new Int32Array(nx * ny).fill(-1);
  const pageByBrick = new Map(bank.pages.map(page => [page.sourceBrickKey, page]));
  for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const index = x + nx * y, cell = ownerAt(topology, x + 0.5, y + 0.5);
    const page = cell ? pageByBrick.get(cell.brickKey) : undefined;
    const value = phiAt(cell, fields,
      x + 0.5, y + 0.5, cellSize);
    phi[index] = value; flags[index] = 1 | (value < 0 ? 16 : 0);
    pageOwner[index] = page?.id ?? -1;
  }
  return Object.freeze({ dimensions: topology.dimensions, phi, flags, pageOwner });
}

function mode(seed: SliceSceneSeed): SlicePresentationState["mode"] {
  const value = seed.production?.values.presentationColumnHeight;
  return value === "on" || value === "off" ? value : "auto";
}

function frameMetrics(topology: SliceTopology, fields: SliceNumericalFields,
  input?: SlicePresentationFrameInput): Pick<SlicePresentationReceipt,
    "liquidAreaFine" | "liquidCentroidFine" | "liquidBoundsFine" | "rigidBodies"> {
  let area = 0, mx = 0, my = 0;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const cell of topology.cells) {
    const amount = f(fields.density[cell.id]! * cell.volumeFineCells);
    if (!(amount > 0)) continue;
    area = f(area + amount);
    mx = f(mx + f(amount * cell.centerFine[0]));
    my = f(my + f(amount * cell.centerFine[1]));
    x0 = Math.min(x0, cell.minimumFine[0]); y0 = Math.min(y0, cell.minimumFine[1]);
    x1 = Math.max(x1, cell.maximumFine[0]); y1 = Math.max(y1, cell.maximumFine[1]);
  }
  return Object.freeze({ liquidAreaFine: area,
    liquidCentroidFine: area > 0 ? [f(mx / area), f(my / area)] as const : null,
    liquidBoundsFine: area > 0 ? [x0, y0, x1, y1] as const : null,
    rigidBodies: Object.freeze([...(input?.rigidBodies ?? [])]) });
}

export function createSlicePresentation(seed: SliceSceneSeed, topology: SliceTopology,
  fields: SliceNumericalFields, input?: SlicePresentationFrameInput): SlicePresentationState {
  const selectedMode = mode(seed);
  const built = buildBank(topology, fields, 1, selectedMode,
    seed.viewport.sourceCellSize, Boolean(seed.production?.scene.fluid.inflow), seed.sourceAtlas);
  if (!built.bank || built.fault) throw new Error(`slice presentation bootstrap failed: ${built.fault?.code}`);
  const dense = denseReadback(built.bank, topology, fields, seed.viewport.sourceCellSize);
  const receipt: SlicePresentationReceipt = Object.freeze({ accepted: true,
    publicationGeneration: 1, topologyGeneration: topology.generation,
    pageCount: built.bank.pages.length, sampleCount: built.bank.pages.length * PAGE_RESOLUTION ** 2,
    wetPageCount: built.bank.pages.filter(page => page.wet).length,
    columnPageCount: built.bank.pages.filter(page => page.columns.valid.some(Boolean)).length,
    sourceAtlasGeneration: seed.sourceAtlas?.generation ?? 0,
    ...frameMetrics(topology, fields, input), fault: null });
  return Object.freeze({ dimensions: seed.dimensions, mode: selectedMode,
    cellSize: seed.viewport.sourceCellSize,
    hasSource: Boolean(seed.production?.scene.fluid.inflow),
    accepted: built.bank, publicationGeneration: 1, denseReadback: dense, receipt });
}

export function publishSlicePresentation(previous: SlicePresentationState,
  topology: SliceTopology, fields: SliceNumericalFields,
  sourceAtlas?: SparseAdaptiveMassAtlas,
  input?: SlicePresentationFrameInput): SlicePresentationPublication {
  const generation = previous.publicationGeneration + 1;
  const built = buildBank(topology, fields, generation, previous.mode,
    previous.cellSize, previous.hasSource, sourceAtlas);
  if (!built.bank || built.fault) {
    const receipt = Object.freeze({ ...previous.receipt, accepted: false,
      publicationGeneration: generation, topologyGeneration: topology.generation,
      fault: built.fault });
    return { state: previous, receipt, denseReadback: previous.denseReadback };
  }
  const dense = denseReadback(built.bank, topology, fields, previous.cellSize);
  const receipt: SlicePresentationReceipt = Object.freeze({ accepted: true,
    publicationGeneration: generation, topologyGeneration: topology.generation,
    pageCount: built.bank.pages.length, sampleCount: built.bank.pages.length * PAGE_RESOLUTION ** 2,
    wetPageCount: built.bank.pages.filter(page => page.wet).length,
    columnPageCount: built.bank.pages.filter(page => page.columns.valid.some(Boolean)).length,
    sourceAtlasGeneration: sourceAtlas?.generation ?? 0,
    ...frameMetrics(topology, fields, input), fault: null });
  const state: SlicePresentationState = Object.freeze({ dimensions: previous.dimensions,
    mode: previous.mode, cellSize: previous.cellSize, hasSource: previous.hasSource,
    accepted: built.bank, publicationGeneration: generation,
    denseReadback: dense, receipt });
  return { state, receipt, denseReadback: dense };
}

type RdfPoint = readonly [number, number];
interface RdfPlane { readonly cell: number; readonly normal: RdfPoint; readonly centre: RdfPoint }
type RdfPhase = -1 | 0 | 1;

function partialOpen(fields: SliceNumericalFields, cell: number): boolean {
  const capacity = fields.capacity[cell]!, fill = capacity > 1e-8
    ? fields.density[cell]! / capacity : 0;
  return capacity >= 0.999999 && fill > 1e-6 && fill < 1 - 1e-6;
}

function segmentForPlane(cell: SliceTopologyCell, normal: RdfPoint,
  offset: number): RdfPoint[] {
  const c = cell.centerFine, lo = cell.minimumFine, hi = cell.maximumFine;
  const out: RdfPoint[] = [];
  const addPoint = (x: number, y: number): void => {
    if (x < lo[0] - 1e-8 || x > hi[0] + 1e-8
      || y < lo[1] - 1e-8 || y > hi[1] + 1e-8) return;
    if (!out.some(point => Math.hypot(point[0] - x, point[1] - y) < 1e-7)) out.push([x, y]);
  };
  if (Math.abs(normal[1]) > 1e-12) {
    addPoint(lo[0], c[1] + (offset - normal[0] * (lo[0] - c[0])) / normal[1]);
    addPoint(hi[0], c[1] + (offset - normal[0] * (hi[0] - c[0])) / normal[1]);
  }
  if (Math.abs(normal[0]) > 1e-12) {
    addPoint(c[0] + (offset - normal[1] * (lo[1] - c[1])) / normal[0], lo[1]);
    addPoint(c[0] + (offset - normal[1] * (hi[1] - c[1])) / normal[0], hi[1]);
  }
  return out.slice(0, 2);
}

function rdfValueAtPoint(point: RdfPoint, planes: readonly RdfPlane[]): number | null {
  let weighted = 0, totalWeight = 0;
  for (const plane of planes) {
    const dx = point[0] - plane.centre[0], dy = point[1] - plane.centre[1];
    const distance = plane.normal[0] * dx + plane.normal[1] * dy;
    const weight = distance * distance / Math.max(dx * dx + dy * dy, 1e-12);
    weighted += weight * distance; totalWeight += weight;
  }
  return totalWeight > 1e-8 ? weighted / totalWeight : null;
}

export type SliceSharedRdfSupportTopology = Pick<SliceNumericalTopology,
  "solidVoxelAt" | "solidVoxelFractionAt">;

function provenOpenVoxel(topology: SliceSharedRdfSupportTopology | undefined,
  x: number, y: number): boolean {
  if (!topology) return false;
  if (topology.solidVoxelFractionAt) return topology.solidVoxelFractionAt(x, y) <= 0;
  if (topology.solidVoxelAt) return !topology.solidVoxelAt(x, y);
  return false;
}

interface RdfGhostGeometry {
  readonly point: RdfPoint;
  readonly minimum: RdfPoint;
  readonly maximum: RdfPoint;
}

function provenOpenBox(topology: SliceSharedRdfSupportTopology | undefined,
  geometry: RdfGhostGeometry): boolean {
  for (let y = Math.floor(geometry.minimum[1]); y < Math.ceil(geometry.maximum[1]); y += 1)
    for (let x = Math.floor(geometry.minimum[0]); x < Math.ceil(geometry.maximum[0]); x += 1)
      if (!provenOpenVoxel(topology, x, y)) return false;
  return true;
}

/** Presentation-only geometry for an allocated, inactive sparse cell. */
function inactiveCellGeometryAt(topology: SliceTopology,
  x: number, y: number): RdfGhostGeometry | null {
  const [nx, ny] = topology.dimensions, ix = Math.floor(x), iy = Math.floor(y);
  if (ix < 0 || iy < 0 || ix >= nx || iy >= ny) return null;
  for (const brick of topology.bricks) {
    if (brick.active !== false) continue;
    const spanFine = PAGE_RESOLUTION * span(brick);
    const ox = PAGE_RESOLUTION * brick.coordinate[0];
    const oy = PAGE_RESOLUTION * brick.coordinate[1];
    const upperX = Math.min(nx, ox + spanFine), upperY = Math.min(ny, oy + spanFine);
    if (ix < ox || iy < oy || ix >= upperX || iy >= upperY) continue;
    const scale = spanFine / brick.resolution;
    const lx = Math.floor((ix - ox) / scale), ly = Math.floor((iy - oy) / scale);
    const lowerX = ox + lx * scale, lowerY = oy + ly * scale;
    const maximum: RdfPoint = [Math.min(upperX, lowerX + scale),
      Math.min(upperY, lowerY + scale)];
    return { point: [0.5 * (lowerX + maximum[0]), 0.5 * (lowerY + maximum[1])],
      minimum: [lowerX, lowerY], maximum };
  }
  return null;
}

function reflectedAirGeometryAt(incident: readonly SliceTopologyCell[], vertex: RdfPoint,
  sx: number, sy: number): RdfGhostGeometry | null {
  const candidates = incident.filter(cell => {
    const ownX = cell.centerFine[0] < vertex[0] ? -1 : 1;
    const ownY = cell.centerFine[1] < vertex[1] ? -1 : 1;
    return Number(ownX !== sx) + Number(ownY !== sy) === 1;
  }).sort((a, b) => Math.hypot(a.centerFine[0] - vertex[0], a.centerFine[1] - vertex[1])
    - Math.hypot(b.centerFine[0] - vertex[0], b.centerFine[1] - vertex[1]) || a.id - b.id);
  const source = candidates[0];
  if (!source) return null;
  const point: RdfPoint = [vertex[0] + sx * Math.abs(source.centerFine[0] - vertex[0]),
    vertex[1] + sy * Math.abs(source.centerFine[1] - vertex[1])];
  const half: RdfPoint = [0.5 * source.widthsFine[0], 0.5 * source.widthsFine[1]];
  return { point, minimum: [point[0] - half[0], point[1] - half[1]],
    maximum: [point[0] + half[0], point[1] + half[1]] };
}

function rdfVertexCells(topology: SliceTopology): Map<string, number[]> {
  const result = new Map<string, number[]>();
  for (const cell of topology.cells) {
    // Composite coarse faces are partitioned at fine-side T-junctions. The
    // dense publication therefore gives the coarse affine support to every
    // finest vertex in its bounds; this is our sparse-lattice prolongation,
    // not a guarantee made by the plicRDF paper for nonconforming AMR.
    for (let x = cell.minimumFine[0]; x <= cell.maximumFine[0]; x += 1)
      for (let y = cell.minimumFine[1]; y <= cell.maximumFine[1]; y += 1) {
        const key = `${x}:${y}`, list = result.get(key) ?? [];
        list.push(cell.id); result.set(key, list);
      }
  }
  return result;
}

export function rdfAffineAt(samples: readonly { point: RdfPoint; value: number }[],
  origin: RdfPoint, preserveObservableSlope = true): number | null {
  if (samples.length === 0) return null;
  const inverseCount = 1 / samples.length;
  const mean: RdfPoint = [samples.reduce((sum, sample) => sum + sample.point[0], 0)
    * inverseCount, samples.reduce((sum, sample) => sum + sample.point[1], 0)
    * inverseCount];
  const meanValue = samples.reduce((sum, sample) => sum + sample.value, 0) * inverseCount;
  let mxx = 0, mxy = 0, myy = 0, rhsX = 0, rhsY = 0;
  for (const sample of samples) {
    const x = sample.point[0] - mean[0], y = sample.point[1] - mean[1];
    const difference = sample.value - meanValue;
    mxx += x * x; mxy += x * y; myy += y * y;
    rhsX += difference * x; rhsY += difference * y;
  }
  const determinant = mxx * myy - mxy * mxy, trace = mxx + myy;
  let gradient: RdfPoint = [0, 0];
  if (Math.abs(determinant) > 64 * 1.1920928955078125e-7 * trace * trace) gradient = [
    (rhsX * myy - mxy * rhsY) / determinant,
    (mxx * rhsY - mxy * rhsX) / determinant,
  ];
  else if (preserveObservableSlope && trace > 1e-12) {
    // The centred covariance is rank one. Its nonzero eigenvalue is the trace,
    // and rhs lies in that eigenspace, so this is the Moore-Penrose solution.
    gradient = [rhsX / trace, rhsY / trace];
  }
  return meanValue + gradient[0] * (origin[0] - mean[0])
    + gradient[1] * (origin[1] - mean[1]);
}

function rdfAffineRank(samples: readonly { point: RdfPoint; value: number }[]): 0 | 1 | 2 {
  if (samples.length < 2) return 0;
  const inverseCount = 1 / samples.length;
  const meanX = samples.reduce((sum, sample) => sum + sample.point[0], 0) * inverseCount;
  const meanY = samples.reduce((sum, sample) => sum + sample.point[1], 0) * inverseCount;
  let mxx = 0, mxy = 0, myy = 0;
  for (const sample of samples) {
    const x = sample.point[0] - meanX, y = sample.point[1] - meanY;
    mxx += x * x; mxy += x * y; myy += y * y;
  }
  const trace = mxx + myy;
  if (!(trace > 1e-12)) return 0;
  return Math.abs(mxx * myy - mxy * mxy)
    > 64 * 1.1920928955078125e-7 * trace * trace ? 2 : 1;
}

function clippedRdfTriangleArea(points: readonly (readonly [number, number, number])[]): number {
  const polygon: [number, number, number][] = [];
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i]!, b = points[(i + 1) % points.length]!;
    if (a[2] <= 0) polygon.push([...a]);
    if ((a[2] < 0) !== (b[2] < 0)) {
      const t = a[2] / (a[2] - b[2]);
      polygon.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), 0]);
    }
  }
  let twice = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i]!, b = polygon[(i + 1) % polygon.length]!;
    twice += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(twice) / 2;
}

/**
 * Construct the reversible shared-RDF preview from accepted VOF fractions and
 * normals. Equation 9/10 weights are evaluated at compact cell centres;
 * least-squares interpolation publishes one value per shared finest vertex.
 * See https://doi.org/10.1016/j.jcp.2019.01.009 sections 3.2.1 and 3.6.
 */
export function reconstructSliceSharedRdf(topology: SliceTopology,
  fields: SliceNumericalFields,
  numericalTopology?: SliceSharedRdfSupportTopology): SliceSharedRdfIsocontour {
  const planes: RdfPlane[] = [];
  let cutPartial = 0;
  for (const cell of topology.cells) {
    const capacity = fields.capacity[cell.id]!, fill = capacity > 1e-8
      ? fields.density[cell.id]! / capacity : 0;
    if (capacity < 0.999999 && fill > 1e-6 && fill < 1 - 1e-6) cutPartial += 1;
    if (!partialOpen(fields, cell.id)) continue;
    const normal: RdfPoint = [fields.interfaceNormal[2 * cell.id]!,
      fields.interfaceNormal[2 * cell.id + 1]!];
    if (!(Math.hypot(normal[0], normal[1]) > 0.5)) continue;
    const segment = segmentForPlane(cell, normal, fields.interfaceOffset[cell.id]!);
    if (segment.length !== 2) continue;
    planes.push({ cell: cell.id, normal,
      centre: [(segment[0]![0] + segment[1]![0]) / 2,
        (segment[0]![1] + segment[1]![1]) / 2] });
  }
  const byCell = new Map(planes.map(plane => [plane.cell, plane]));
  const attached = rdfVertexCells(topology);
  const cellPhi = new Float64Array(topology.cells.length).fill(Number.NaN);
  for (const cell of topology.cells) {
    const neighbours = new Set<number>();
    for (const x of [cell.minimumFine[0], cell.maximumFine[0]])
      for (const y of [cell.minimumFine[1], cell.maximumFine[1]])
        for (const other of attached.get(`${x}:${y}`) ?? []) neighbours.add(other);
    let weighted = 0, totalWeight = 0;
    for (const other of neighbours) {
      const plane = byCell.get(other); if (!plane) continue;
      const dx = cell.centerFine[0] - plane.centre[0], dy = cell.centerFine[1] - plane.centre[1];
      const squared = dx * dx + dy * dy;
      const distance = plane.normal[0] * dx + plane.normal[1] * dy;
      // Scheufler/Roenby A=2 orientation weight: cos(theta)^2.
      // Equation 14 also applies this expression to the cell's own segment;
      // a centred cut contributes zero rather than an invented unit weight.
      const weight = distance * distance / Math.max(squared, 1e-12);
      weighted += weight * distance; totalWeight += weight;
    }
    if (totalWeight > 0) cellPhi[cell.id] = weighted / totalWeight;
  }

  const [nx, ny] = topology.dimensions, stride = nx + 1;
  const vertices = new Float32Array((nx + 1) * (ny + 1)).fill(Number.NaN);
  let inactiveAirGhostSamples = 0, reflectedAirGhostSamples = 0;
  const actualVertices = new Set<string>();
  for (const cell of topology.cells) for (const x of [cell.minimumFine[0], cell.maximumFine[0]])
    for (const y of [cell.minimumFine[1], cell.maximumFine[1]]) actualVertices.add(`${x}:${y}`);
  for (let y = 0; y <= ny; y += 1) for (let x = 0; x <= nx; x += 1) {
    const key = `${x}:${y}`, ids = attached.get(key) ?? [];
    // A dense sample strictly owned by one coarse mixed cell is prolongated
    // with that cell's affine PLIC distance. Treating its centre RDF as a
    // constant over the whole B8 leaf shifts a planar pool by almost one fine
    // cell. Shared corners and T-junctions still take the common LS value.
    if (!actualVertices.has(key)) {
      const direct = ids.map(id => byCell.get(id))
        .filter((plane): plane is RdfPlane => Boolean(plane))
        .map(plane => plane.normal[0] * (x - plane.centre[0])
          + plane.normal[1] * (y - plane.centre[1]));
      if (direct.length) {
        vertices[x + stride * y] = f(direct.reduce((sum, value) => sum + value, 0) / direct.length);
        continue;
      }
    }
    let samples = ids.filter(id => Number.isFinite(cellPhi[id]!)).map(id => ({
      point: topology.cells[id]!.centerFine, value: cellPhi[id]!,
    }));
    // The production sparse publisher retains an air support layer around the
    // accepted surface. The dimensionally reduced topology omits empty cells,
    // so read the allocated inactive directory rung to complete Eq. 11 without
    // admitting that cell to numerical authority.
    const incident = ids.map(id => topology.cells[id]!).filter(Boolean);
    if (incident.length && rdfAffineRank(samples) < 2 && !fields.solidMotionActive) {
      const supportPlanes = ids.map(id => byCell.get(id))
        .filter((plane): plane is RdfPlane => Boolean(plane));
      for (const sx of [-1, 1]) for (const sy of [-1, 1]) {
        const probeX = x + sx * 0.25, probeY = y + sy * 0.25;
        if (probeX < 0 || probeY < 0 || probeX >= nx || probeY >= ny) continue;
        if (ownerAt(topology, probeX, probeY)) continue;
        let geometry = inactiveCellGeometryAt(topology, probeX, probeY);
        let reflected = false;
        if (!geometry) {
          geometry = reflectedAirGeometryAt(incident, [x, y], sx, sy);
          reflected = geometry !== null;
        }
        if (!geometry || geometry.minimum[0] < 0 || geometry.minimum[1] < 0
          || geometry.maximum[0] > nx || geometry.maximum[1] > ny
          || !provenOpenBox(numericalTopology, geometry)) continue;
        const point = geometry.point;
        const value = rdfValueAtPoint(point, supportPlanes);
        if (value === null) continue;
        if (!samples.some(sample => Math.hypot(sample.point[0] - point[0],
          sample.point[1] - point[1]) < 1e-7)) {
          samples.push({ point, value });
          if (reflected) reflectedAirGhostSamples += 1;
          else inactiveAirGhostSamples += 1;
        }
      }
    }
    // isoAlpha fallback for a perfectly face-aligned jump, which contains no
    // partial cell and therefore supplies no RDF interface segment.
    if (!samples.length) samples = ids.flatMap(id => {
      const cell = topology.cells[id]!, capacity = fields.capacity[id]!;
      if (!(capacity > 1e-8)) return [];
      const fill = Math.max(0, Math.min(1, fields.density[id]! / capacity));
      return [{ point: cell.centerFine,
        value: (0.5 - fill) * 4 * Math.max(...cell.widthsFine) }];
    });
    if (!samples.length) continue;
    const fitted = samples.length >= 3
      ? rdfAffineAt(samples, [x, y]) : null;
    vertices[x + stride * y] = f(fitted ?? samples.reduce((sum, q) => sum + q.value, 0) / samples.length);
  }
  const phaseAt = (cell: SliceTopologyCell | undefined, x: number, y: number): RdfPhase => {
    if (!cell) return 0;
    if (!(fields.capacity[cell.id]! > 1e-8)) return 0;
    const fill = fields.density[cell.id]! / fields.capacity[cell.id]!;
    if (fill >= 1 - 1e-6) return -1;
    if (fill <= 1e-6) return 1;
    const plane = byCell.get(cell.id);
    if (!plane) return 0;
    const distance = plane.normal[0] * (x - plane.centre[0])
      + plane.normal[1] * (y - plane.centre[1]);
    return distance <= 0 ? -1 : 1;
  };
  // Bound the free affine fit by the accepted phase in every incident owner.
  // A mixed owner contributes the side of its own PLIC, retaining small real
  // cuts while preventing an extrapolated neighbour from reversing them.
  for (let y = 0; y <= ny; y += 1) for (let x = 0; x <= nx; x += 1) {
    const phases = new Set<RdfPhase>();
    const incidentIds = attached.get(`${x}:${y}`) ?? [];
    // A valid interface plane already supplies the signed Eq. 9/10/11 value.
    // Phase certificates are only the no-interface fallback; applying them to
    // a fitted vertex clips exact oblique planes at sparse air boundaries.
    if (incidentIds.some(id => byCell.has(id))) continue;
    for (const id of incidentIds) {
      const accepted = phaseAt(topology.cells[id], x, y);
      if (accepted !== 0) phases.add(accepted);
    }
    for (const dx of [-0.25, 0.25]) for (const dy of [-0.25, 0.25]) {
      const qx = x + dx, qy = y + dy;
      if (qx >= 0 && qy >= 0 && qx < nx && qy < ny
        && !ownerAt(topology, qx, qy) && !fields.solidMotionActive
        && provenOpenVoxel(numericalTopology, Math.floor(qx), Math.floor(qy))) phases.add(1);
    }
    const index = x + stride * y, value = vertices[index]!;
    if (!Number.isFinite(value)) continue;
    const margin = 0.25 * Math.min(...incidentIds.map(id =>
      Math.min(...topology.cells[id]!.widthsFine)), 1);
    if (phases.size === 1 && phases.has(-1) && value >= 0) vertices[index] = f(-margin);
    else if (phases.size === 1 && phases.has(1) && value <= 0) vertices[index] = f(margin);
  }
  // A face-aligned full/empty jump has no mixed owner and hence no PLIC
  // segment. Keep the established exact zero only on that true interface.
  const purePhaseAt = (x: number, y: number): RdfPhase => {
    const cell = ownerAt(topology, x, y);
    if (!cell) {
      if (x < 0 || y < 0 || x >= nx || y >= ny || fields.solidMotionActive) return 0;
      return provenOpenVoxel(numericalTopology, Math.floor(x), Math.floor(y)) ? 1 : 0;
    }
    if (!(fields.capacity[cell.id]! > 1e-8)) return 0;
    const fill = fields.density[cell.id]! / fields.capacity[cell.id]!;
    return fill >= 1 - 1e-6 ? -1 : fill <= 1e-6 ? 1 : 0;
  };
  for (let y = 0; y <= ny; y += 1) for (let x = 0; x <= nx; x += 1) {
    if ((attached.get(`${x}:${y}`) ?? []).some(id => byCell.has(id))) continue;
    let aligned = false;
    for (const tangent of [-0.25, 0.25]) {
      const horizontal = purePhaseAt(x + tangent, y - 1e-4)
        * purePhaseAt(x + tangent, y + 1e-4);
      const vertical = purePhaseAt(x - 1e-4, y + tangent)
        * purePhaseAt(x + 1e-4, y + tangent);
      aligned ||= horizontal === -1 || vertical === -1;
    }
    if (aligned) vertices[x + stride * y] = 0;
  }

  const segments: number[] = [];
  let ambiguous = 0, unresolved = 0;
  for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const corners: readonly [number, number, number][] = [
      [x, y, vertices[x + stride * y]!], [x + 1, y, vertices[x + 1 + stride * y]!],
      [x + 1, y + 1, vertices[x + 1 + stride * (y + 1)]!],
      [x, y + 1, vertices[x + stride * (y + 1)]!],
    ];
    const cuts: RdfPoint[] = [];
    for (let edge = 0; edge < 4; edge += 1) {
      const a = corners[edge]!, b = corners[(edge + 1) % 4]!;
      if (!Number.isFinite(a[2]) || !Number.isFinite(b[2])) continue;
      if ((a[2] < 0) === (b[2] < 0) || a[2] === b[2]) continue;
      const t = a[2] / (a[2] - b[2]);
      cuts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
    }
    if (cuts.length === 4) ambiguous += 1;
    else if (cuts.length !== 0 && cuts.length !== 2) unresolved += 1;
    // Triangulate through the bilinear centre value. Besides resolving the
    // four-edge saddle deterministically, this commutes with either axis
    // reflection; pairing square edges in traversal order does not.
    for (const triangle of sliceRdfTriangles(x, y,
      corners[0]![2], corners[1]![2], corners[2]![2], corners[3]![2])) {
      const triangleCuts: RdfPoint[] = [];
      for (let edge = 0; edge < 3; edge += 1) {
        const a = triangle[edge]!, b = triangle[(edge + 1) % 3]!;
        if (!Number.isFinite(a[2]) || !Number.isFinite(b[2])) continue;
        if ((a[2] < 0) === (b[2] < 0) || a[2] === b[2]) continue;
        const t = a[2] / (a[2] - b[2]);
        triangleCuts.push([a[0] + t * (b[0] - a[0]),
          a[1] + t * (b[1] - a[1])]);
      }
      if (triangleCuts.length === 2) segments.push(...triangleCuts[0]!, ...triangleCuts[1]!);
      else if (triangleCuts.length !== 0) unresolved += 1;
    }
  }

  let exact = 0, represented = 0;
  const partialErrors: number[] = [];
  for (const cell of topology.cells) {
    const target = fields.density[cell.id]! * cell.volumeFineCells;
    exact += target;
    if (!partialOpen(fields, cell.id)) { represented += target; continue; }
    let area = 0, valid = true;
    for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
      for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
        const a = vertices[x + stride * y]!, b = vertices[x + 1 + stride * y]!;
        const c = vertices[x + 1 + stride * (y + 1)]!, d = vertices[x + stride * (y + 1)]!;
        if (![a, b, c, d].every(Number.isFinite)) { valid = false; continue; }
        for (const triangle of sliceRdfTriangles(x, y, a, b, c, d)) {
          area += clippedRdfTriangleArea(triangle);
        }
      }
    if (!valid) continue;
    represented += area; partialErrors.push(area - target);
  }
  const receipt: SliceSharedRdfReceipt = Object.freeze({ interfaceCells: planes.length,
    unsupportedCutPartialCells: cutPartial, inactiveAirGhostSamples,
    reflectedAirGhostSamples, ambiguousFineCells: ambiguous,
    unresolvedFineCells: unresolved, exactAreaFine: exact, representedAreaFine: represented,
    signedAreaErrorFine: represented - exact,
    meanAbsolutePartialCellErrorFine: partialErrors.reduce((sum, value) => sum + Math.abs(value), 0)
      / Math.max(1, partialErrors.length),
    maximumAbsolutePartialCellErrorFine: Math.max(0, ...partialErrors.map(Math.abs)) });
  return Object.freeze({ dimensions: topology.dimensions, vertexPhiFine: vertices,
    segmentsFine: Float32Array.from(segments, f), receipt });
}
