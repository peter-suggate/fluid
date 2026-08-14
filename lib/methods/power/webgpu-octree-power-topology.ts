/** Fail-closed GPU catalog lookup and PowerRowMetric publication. */

import {
  OCTREE_GENERATED_POWER_CATALOG_MANIFEST,
  type GeneratedOctreePowerCatalogViews,
} from "./generated/octree-power-catalog";
import {
  OCTREE_POWER_CATALOG_FACE_FLOATS,
  OCTREE_POWER_RECONSTRUCTION_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_FLAGS,
  OCTREE_POWER_ROW_TEMPLATE_FLOATS,
  OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS,
  OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR,
  OCTREE_POWER_ROW_TEMPLATE_VERSION,
  OCTREE_POWER_TRANSFER_RELATION,
  unpackOctreePowerRowTemplateSlot,
} from "./octree-power-catalog";
import {
  OCTREE_POWER_COMPILED_SAMPLER_FIXED_AXES,
  OCTREE_POWER_COMPILED_SAMPLER_HEADER_WORDS,
  OCTREE_POWER_COMPILED_SAMPLER_OCTANTS,
  OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS,
  OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS,
  compileOctreePowerSampler,
} from "./octree-power-compiled-sampler";
import { OCTREE_POWER_ROW_METRIC_BYTES } from "./octree-power-operator";
import {
  OCTREE_POWER_ROW_DELTA_VALID,
  assertOctreePowerRowDeltaLayout,
  type OctreePowerRowDeltaSource,
} from "./webgpu-octree-power-descriptor";
import { PassBroker } from "../../core/webgpu-pass-broker";
import { gpuCompilationManagerFor } from "../../core/gpu-compilation-manager";

export const OCTREE_POWER_TOPOLOGY_CONTROL_BYTES = 32;
// WGSL ControlArena: two 32-byte controls followed by two independently
// 16-byte-aligned vec3u dispatches and two scalar counters.
const OCTREE_POWER_TOPOLOGY_CONTROL_ARENA_BYTES = 112;
export const OCTREE_POWER_TOPOLOGY_VALID = 0x8000_0000;
export const OCTREE_POWER_TOPOLOGY_BOUNDARY_SHIFT = 8;
export const OCTREE_POWER_TOPOLOGY_BOUNDARY_MASK = 0x0000_3f00;
/** The 18 occupied same-level neighbours of an interior Cartesian bulk cell. */
export const OCTREE_POWER_REGULAR_DESCRIPTOR = 0x0003_ffff;
export const OCTREE_POWER_TOPOLOGY_ERROR = Object.freeze({
  anisotropicCell: 1 << 0,
  lookupMiss: 1 << 1,
  catalogVersion: 1 << 2,
  capacity: 1 << 3,
} as const);

export interface OctreePowerTopologyPlan {
  readonly rowCapacity: number;
  readonly entryCount: number;
  readonly lookupCount: number;
  readonly metricBytes: number;
  readonly catalogBytes: number;
  readonly rowTemplateBytes: number;
  readonly allocatedBytes: number;
}

export interface OctreePowerTopologySource {
  readonly plan: OctreePowerTopologyPlan;
  readonly metrics: GPUBuffer;
  readonly control: GPUBuffer;
  readonly catalogEntryHeaders: GPUBuffer;
  readonly catalogVolumes: GPUBuffer;
  readonly catalogFaces: GPUBuffer;
  /** Section 6.3 diagonal plus eighteen canonical face-slot coefficients. */
  readonly catalogCoefficients: GPUBuffer;
  readonly rowTemplateHeaders: GPUBuffer;
  readonly rowTemplateSlots: GPUBuffer;
  readonly rowTemplateData: GPUBuffer;
  readonly rowTemplateDiagonals: GPUBuffer;
  readonly reconstructionData: GPUBuffer;
  readonly rowTemplateHeaderOffsetBytes: number;
  readonly rowTemplateSlotOffsetBytes: number;
  readonly rowTemplateDataOffsetBytes: number;
  readonly rowTemplateDiagonalOffsetBytes: number;
  readonly reconstructionDataOffsetBytes: number;
  readonly catalogTetrahedronHeaders?: GPUBuffer;
  readonly catalogTetrahedra?: GPUBuffer;
  readonly catalogTetrahedronVertices?: GPUBuffer;
  readonly catalogTetrahedronVertexCount?: number;
  /** Immutable selector-transform and tetrahedron-walk point-location tables. */
  readonly catalogCompiledSampler?: GPUBuffer;
  readonly catalogLookup: GPUBuffer;
  readonly sameOrFinerDirect: GPUBuffer;
  readonly sameOrCoarserDirect: GPUBuffer;
}

export function powerCellSpacingIsotropic(spacing: readonly [number, number, number], tolerance = 1e-5): boolean {
  if (spacing.some((value) => !Number.isFinite(value) || value <= 0) || !Number.isFinite(tolerance) || tolerance < 0) return false;
  return Math.max(...spacing) / Math.min(...spacing) <= 1 + tolerance;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function topologyPublicationArenaBytes(rowCapacity: number): number {
  const blocks = Math.ceil(rowCapacity / 256);
  return (3 * rowCapacity + blocks + 1 + 6 * blocks) * 4;
}

export function planOctreePowerTopology(rowCapacityValue: number, catalog: GeneratedOctreePowerCatalogViews): OctreePowerTopologyPlan {
  const rowCapacity = positiveInteger(rowCapacityValue, "Power topology row capacity");
  if (catalog.lookup.length % 3 !== 0 || catalog.entryHeaders.length % 2 !== 0
    || catalog.faceData.length % OCTREE_POWER_CATALOG_FACE_FLOATS !== 0 || catalog.entryHeaders.length !== catalog.entryVolumes.length * 2
    || catalog.tetrahedronHeaders.length !== catalog.entryVolumes.length * 3
    || catalog.coefficientData.length !== catalog.entryVolumes.length * 19
    || catalog.rowTemplateHeaders.length
      !== catalog.entryVolumes.length * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS
    || catalog.rowTemplateSlots.length * OCTREE_POWER_CATALOG_FACE_FLOATS !== catalog.faceData.length
    || catalog.rowTemplateData.length
      !== catalog.rowTemplateSlots.length * OCTREE_POWER_ROW_TEMPLATE_FLOATS
    || catalog.rowTemplateDiagonals.length !== catalog.entryVolumes.length
    || catalog.reconstructionData.length
      !== catalog.rowTemplateSlots.length * OCTREE_POWER_RECONSTRUCTION_FLOATS
    || catalog.tetrahedronVertexData.length % 4 !== 0 || catalog.tetrahedronVertexData.length / 4 > 256
    || catalog.sameOrFinerDirect.length !== 1 << 18 || catalog.sameOrCoarserDirect.length !== 1 << 9) {
    throw new RangeError("Power catalog typed-array shape is invalid");
  }
  const entryCount = catalog.entryVolumes.length;
  const lookupCount = catalog.lookup.length / 3;
  const faceCount = catalog.faceData.length / OCTREE_POWER_CATALOG_FACE_FLOATS;
  if (entryCount === 0 || lookupCount === 0 || faceCount === 0) throw new RangeError("Power catalog must not be empty");
  for (let entry = 0; entry < entryCount; entry += 1) {
    const firstFace = catalog.entryHeaders[entry * 2];
    const count = catalog.entryHeaders[entry * 2 + 1];
    if (firstFace > faceCount || count > faceCount - firstFace
      || !(catalog.entryVolumes[entry] > 0) || !Number.isFinite(catalog.entryVolumes[entry])) {
      throw new RangeError("Power catalog entry is invalid");
    }
    const firstTetrahedron = catalog.tetrahedronHeaders[entry * 3];
    const tetrahedronCount = catalog.tetrahedronHeaders[entry * 3 + 1];
    if (firstTetrahedron > catalog.tetrahedronData.length
      || tetrahedronCount > catalog.tetrahedronData.length - firstTetrahedron) {
      throw new RangeError("Power catalog tetrahedron range is invalid");
    }
    const template = entry * OCTREE_POWER_ROW_TEMPLATE_HEADER_WORDS;
    const firstSlot = catalog.rowTemplateHeaders[template];
    const slotCount = catalog.rowTemplateHeaders[template + 1];
    const caseFlags = catalog.rowTemplateHeaders[template + 2];
    const dynamicSlotCount = catalog.rowTemplateHeaders[template + 3];
    const knownFlags = OCTREE_POWER_ROW_TEMPLATE_FLAGS.regularInterior
      | OCTREE_POWER_ROW_TEMPLATE_FLAGS.transition
      | OCTREE_POWER_ROW_TEMPLATE_FLAGS.physicalBoundary
      | OCTREE_POWER_ROW_TEMPLATE_FLAGS.transitionBoundary;
    if (firstSlot !== firstFace || slotCount !== count || dynamicSlotCount !== count
      || (caseFlags & ~knownFlags) !== 0 || (entry === 0)
        !== ((caseFlags & OCTREE_POWER_ROW_TEMPLATE_FLAGS.regularInterior) !== 0)) {
      throw new RangeError("Power catalog dense row-template header is invalid");
    }
    let diagonal = Math.fround(0);
    for (let local = 0; local < slotCount; local += 1) {
      const slot = firstSlot + local;
      let selector: ReturnType<typeof unpackOctreePowerRowTemplateSlot>;
      try {
        selector = unpackOctreePowerRowTemplateSlot(catalog.rowTemplateSlots[slot]);
      } catch {
        throw new RangeError("Power catalog packed row-template selector is invalid");
      }
      const data = slot * OCTREE_POWER_ROW_TEMPLATE_FLOATS;
      const coefficient = catalog.rowTemplateData[data];
      const area = catalog.rowTemplateData[data + 1];
      const inverseDistance = catalog.rowTemplateData[data + 2];
      if (selector.family >= 6 || selector.dynamicBoundarySlot !== local
        || selector.worldBoundary
          !== (selector.neighborSelector === OCTREE_POWER_ROW_TEMPLATE_INVALID_SELECTOR)
        || selector.worldBoundary
          !== (selector.transferRelation === OCTREE_POWER_TRANSFER_RELATION.boundary)
        || !(coefficient > 0) || !(area > 0) || !(inverseDistance > 0)
        || !Number.isFinite(coefficient) || !Number.isFinite(area) || !Number.isFinite(inverseDistance)) {
        throw new RangeError("Power catalog row-template slot is invalid");
      }
      diagonal = Math.fround(diagonal + coefficient);
      const reconstruction = slot * OCTREE_POWER_RECONSTRUCTION_FLOATS;
      for (let axis = 0; axis < OCTREE_POWER_RECONSTRUCTION_FLOATS; axis += 1) {
        if (!Number.isFinite(catalog.reconstructionData[reconstruction + axis])) {
          throw new RangeError("Power catalog reconstruction pseudoinverse is invalid");
        }
      }
    }
    if (!Number.isFinite(catalog.rowTemplateDiagonals[entry])
      || new Uint32Array(Float32Array.of(catalog.rowTemplateDiagonals[entry]).buffer)[0]
        !== new Uint32Array(Float32Array.of(diagonal).buffer)[0]) {
      throw new RangeError("Power catalog row-template diagonal is invalid");
    }
  }
  if ((catalog.sameOrFinerDirect[OCTREE_POWER_REGULAR_DESCRIPTOR] & 0xffff) !== 0) {
    throw new RangeError("Power catalog regular descriptor must resolve to dense case zero");
  }
  for (let index = 0; index < lookupCount; index += 1) {
    const offset = index * 3;
    if ((index > 0 && catalog.lookup[offset] <= catalog.lookup[offset - 3])
      || catalog.lookup[offset + 1] >= entryCount || catalog.lookup[offset + 2] >= 48) {
      throw new RangeError("Power catalog lookup is invalid");
    }
  }
  for (const packed of catalog.sameOrFinerDirect) if (packed === 0xffff_ffff
    || (packed & 0xffff) >= entryCount || (packed >>> 16) >= 48) {
    throw new RangeError("Power catalog same/finer direct lookup is invalid");
  }
  for (let descriptor = 0; descriptor < catalog.sameOrCoarserDirect.length; descriptor += 1) {
    const packed = catalog.sameOrCoarserDirect[descriptor];
    if (packed === 0xffff_ffff || (packed & 0xffff) >= entryCount || (packed >>> 16) >= 48) {
      throw new RangeError("Power catalog same/coarser direct lookup is invalid");
    }
  }
  const metricBytes = rowCapacity * OCTREE_POWER_ROW_METRIC_BYTES;
  const rowTemplateBytes = catalog.rowTemplateHeaders.byteLength + catalog.rowTemplateSlots.byteLength
    + catalog.rowTemplateData.byteLength + catalog.rowTemplateDiagonals.byteLength
    + catalog.reconstructionData.byteLength;
  const catalogBytes = catalog.entryHeaders.byteLength + catalog.entryVolumes.byteLength + catalog.faceData.byteLength
    + catalog.lookup.byteLength + catalog.sameOrFinerDirect.byteLength + catalog.sameOrCoarserDirect.byteLength
    + catalog.tetrahedronHeaders.byteLength + catalog.tetrahedronData.byteLength + catalog.tetrahedronVertexData.byteLength
    + (OCTREE_POWER_COMPILED_SAMPLER_HEADER_WORDS
      + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS * (catalog.tetrahedronVertexData.length / 4)
      + catalog.tetrahedronData.length + OCTREE_POWER_COMPILED_SAMPLER_OCTANTS * entryCount
      + 9 * catalog.tetrahedronData.length
      + OCTREE_POWER_COMPILED_SAMPLER_FIXED_AXES * OCTREE_POWER_COMPILED_SAMPLER_SYMMETRY_MASKS
        * OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS
      + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS * OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS
      + OCTREE_POWER_COMPILED_SAMPLER_TRANSFORMS) * 4
    + catalog.coefficientData.byteLength + rowTemplateBytes;
  return { rowCapacity, entryCount, lookupCount, metricBytes, catalogBytes, rowTemplateBytes,
    allocatedBytes: 2 * metricBytes + OCTREE_POWER_TOPOLOGY_CONTROL_ARENA_BYTES + 80 + 12 + catalogBytes
      + topologyPublicationArenaBytes(rowCapacity) };
}

interface OctreePowerTopologyPipelineBundle {
  readonly preparePipeline: GPUComputePipeline;
  readonly resolvePipeline: GPUComputePipeline;
  readonly stagePipeline: GPUComputePipeline;
  readonly prefixPipeline: GPUComputePipeline;
  readonly scatterPipeline: GPUComputePipeline;
  readonly commitPipeline: GPUComputePipeline;
  readonly finalizePipeline: GPUComputePipeline;
}

const octreePowerTopologyPipelineCache = new WeakMap<GPUDevice,
  Promise<OctreePowerTopologyPipelineBundle>>();
const octreePowerTopologyLayoutCache = new WeakMap<GPUDevice, {
  readonly layout: GPUBindGroupLayout;
  readonly pipelineLayout: GPUPipelineLayout;
}>();

export class WebGPUOctreePowerTopology {
  readonly plan: OctreePowerTopologyPlan;
  readonly metrics: GPUBuffer;
  /** Complete inactive row-metric candidate. */
  readonly candidateMetrics: GPUBuffer;
  readonly control: GPUBuffer;
  readonly catalogEntryHeaders: GPUBuffer;
  readonly catalogVolumes: GPUBuffer;
  readonly catalogFaces: GPUBuffer;
  readonly catalogCoefficients: GPUBuffer;
  readonly rowTemplateHeaders: GPUBuffer;
  readonly rowTemplateSlots: GPUBuffer;
  readonly rowTemplateData: GPUBuffer;
  readonly rowTemplateDiagonals: GPUBuffer;
  readonly reconstructionData: GPUBuffer;
  readonly rowTemplateHeaderOffsetBytes: number;
  readonly rowTemplateSlotOffsetBytes: number;
  readonly rowTemplateDataOffsetBytes: number;
  readonly rowTemplateDiagonalOffsetBytes: number;
  readonly reconstructionDataOffsetBytes: number;
  readonly catalogTetrahedronHeaders: GPUBuffer;
  readonly catalogTetrahedra: GPUBuffer;
  readonly catalogTetrahedronVertices: GPUBuffer;
  readonly catalogTetrahedronVertexCount: number;
  readonly catalogCompiledSampler: GPUBuffer;
  readonly catalogLookup: GPUBuffer;
  readonly sameOrFinerDirect: GPUBuffer;
  readonly sameOrCoarserDirect: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly workDispatch: GPUBuffer;
  private preparePipeline!: GPUComputePipeline;
  private resolvePipeline!: GPUComputePipeline;
  private stagePipeline!: GPUComputePipeline;
  private prefixPipeline!: GPUComputePipeline;
  private scatterPipeline!: GPUComputePipeline;
  private commitPipeline!: GPUComputePipeline;
  private finalizePipeline!: GPUComputePipeline;
  private readonly layout: GPUBindGroupLayout;
  private readonly pipelineLayout: GPUPipelineLayout;
  private readonly device: GPUDevice;
  private readonly regularPacked: number;
  private readonly regularVolume: number;
  private cachedGroup?: {
    descriptors: GPUBuffer;
    rowDelta: GPUBuffer;
    group: GPUBindGroup;
  };
  private destroyed = false;

  constructor(device: GPUDevice, rowCapacity: number, catalog: GeneratedOctreePowerCatalogViews) {
    this.device = device;
    this.plan = planOctreePowerTopology(rowCapacity, catalog);
    this.regularPacked = catalog.sameOrFinerDirect[OCTREE_POWER_REGULAR_DESCRIPTOR];
    this.regularVolume = catalog.entryVolumes[this.regularPacked & 0xffff];
    this.catalogTetrahedronVertexCount = catalog.tetrahedronVertexData.length / 4;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const upload = (label: string, data: ArrayBufferView) => {
      const buffer = device.createBuffer({
        label,
        size: Math.max(4, data.byteLength),
        // Failure-only Dawn diagnostics read exact selector payloads back to
        // distinguish catalog asymmetry from live topology corruption.
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        mappedAtCreation: true,
      });
      new Uint8Array(buffer.getMappedRange(), 0, data.byteLength)
        .set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      buffer.unmap();
      return buffer;
    };
    this.metrics = device.createBuffer({ label: "Octree power row metrics", size: this.plan.metricBytes, usage: storage });
    this.candidateMetrics = device.createBuffer({ label: "Octree power row metric candidates", size: this.plan.metricBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power topology control transaction",
      size: OCTREE_POWER_TOPOLOGY_CONTROL_ARENA_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power topology params", size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.workDispatch = device.createBuffer({ label: "Octree power topology work dispatch", size: 12,
      usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.catalogEntryHeaders = upload("Octree power catalog entry headers", catalog.entryHeaders);
    let denseOffsetBytes = catalog.entryVolumes.byteLength;
    this.rowTemplateHeaderOffsetBytes = denseOffsetBytes;
    denseOffsetBytes += catalog.rowTemplateHeaders.byteLength;
    this.rowTemplateSlotOffsetBytes = denseOffsetBytes;
    denseOffsetBytes += catalog.rowTemplateSlots.byteLength;
    this.rowTemplateDataOffsetBytes = denseOffsetBytes;
    denseOffsetBytes += catalog.rowTemplateData.byteLength;
    this.rowTemplateDiagonalOffsetBytes = denseOffsetBytes;
    denseOffsetBytes += catalog.rowTemplateDiagonals.byteLength;
    this.reconstructionDataOffsetBytes = denseOffsetBytes;
    denseOffsetBytes += catalog.reconstructionData.byteLength;
    const denseCatalog = new Uint8Array(denseOffsetBytes);
    const copyDense = (data: ArrayBufferView, offset: number) =>
      denseCatalog.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), offset);
    copyDense(catalog.entryVolumes, 0);
    copyDense(catalog.rowTemplateHeaders, this.rowTemplateHeaderOffsetBytes);
    copyDense(catalog.rowTemplateSlots, this.rowTemplateSlotOffsetBytes);
    copyDense(catalog.rowTemplateData, this.rowTemplateDataOffsetBytes);
    copyDense(catalog.rowTemplateDiagonals, this.rowTemplateDiagonalOffsetBytes);
    copyDense(catalog.reconstructionData, this.reconstructionDataOffsetBytes);
    this.catalogVolumes = upload("Octree power dense case and row-template arena", denseCatalog);
    this.catalogFaces = upload("Octree power catalog faces", catalog.faceData);
    this.catalogCoefficients = upload("Octree power Section 6.3 coefficients", catalog.coefficientData);
    this.rowTemplateHeaders = this.catalogVolumes;
    this.rowTemplateSlots = this.catalogVolumes;
    this.rowTemplateData = this.catalogVolumes;
    this.rowTemplateDiagonals = this.catalogVolumes;
    this.reconstructionData = this.catalogVolumes;
    this.catalogTetrahedronHeaders = upload("Octree power catalog tetrahedron headers", catalog.tetrahedronHeaders);
    this.catalogTetrahedra = upload("Octree power catalog tetrahedra", catalog.tetrahedronData);
    this.catalogTetrahedronVertices = upload("Octree power catalog tetrahedron vertices", catalog.tetrahedronVertexData);
    const compiledSampler = compileOctreePowerSampler(catalog);
    this.catalogCompiledSampler = upload("Octree power compiled interpolation sampler", compiledSampler.words);
    this.catalogLookup = device.createBuffer({
      label: "Octree power catalog lookup and exact publication arena",
      size: catalog.lookup.byteLength + topologyPublicationArenaBytes(rowCapacity),
      usage: storage,
      mappedAtCreation: true,
    });
    new Uint8Array(this.catalogLookup.getMappedRange(), 0, catalog.lookup.byteLength)
      .set(new Uint8Array(catalog.lookup.buffer, catalog.lookup.byteOffset, catalog.lookup.byteLength));
    this.catalogLookup.unmap();
    this.sameOrFinerDirect = upload("Octree power same/finer direct lookup", catalog.sameOrFinerDirect);
    this.sameOrCoarserDirect = upload("Octree power same/coarser direct lookup", catalog.sameOrCoarserDirect);
    let layouts = octreePowerTopologyLayoutCache.get(device);
    if (!layouts) {
      const layout = device.createBindGroupLayout({ label: "Octree power topology layout", entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 9, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 10, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ] });
      layouts = { layout, pipelineLayout: device.createPipelineLayout({ bindGroupLayouts: [layout] }) };
      octreePowerTopologyLayoutCache.set(device, layouts);
    }
    this.layout = layouts.layout;
    this.pipelineLayout = layouts.pipelineLayout;
  }

  async initializePipelines(): Promise<void> {
    let pipelines = octreePowerTopologyPipelineCache.get(this.device);
    if (!pipelines) {
      const compiler = gpuCompilationManagerFor(this.device);
      const shaderModule = compiler.createShaderModule({
        label: "Octree power topology", code: octreePowerTopologyShader });
      const compile = (label: string, entryPoint: string) => compiler.compileComputePipeline({
        label, layout: this.pipelineLayout, compute: { module: shaderModule, entryPoint },
      });
      pipelines = Promise.all([
        compile("Prepare octree power topology", "preparePowerTopology"),
        compile("Resolve octree power topology", "resolvePowerTopology"),
        compile("Stage exact octree power topology delta", "stagePowerTopologyDelta"),
        compile("Prefix exact octree power topology blocks", "prefixPowerTopologyDelta"),
        compile("Scatter exact octree power topology publication rows", "scatterPowerTopologyDelta"),
        compile("Commit exact octree power topology publication rows", "commitPowerTopologyDelta"),
        compile("Finalize octree power topology", "finalizePowerTopology"),
      ]).then(([preparePipeline, resolvePipeline, stagePipeline, prefixPipeline,
        scatterPipeline, commitPipeline, finalizePipeline]) => ({
        preparePipeline, resolvePipeline, stagePipeline, prefixPipeline,
        scatterPipeline, commitPipeline, finalizePipeline,
      }));
      octreePowerTopologyPipelineCache.set(this.device, pipelines);
    }
    const compiled = await pipelines;
    this.preparePipeline = compiled.preparePipeline;
    this.resolvePipeline = compiled.resolvePipeline;
    this.stagePipeline = compiled.stagePipeline;
    this.prefixPipeline = compiled.prefixPipeline;
    this.scatterPipeline = compiled.scatterPipeline;
    this.commitPipeline = compiled.commitPipeline;
    this.finalizePipeline = compiled.finalizePipeline;
  }

  private createGroup(descriptors: GPUBuffer, rowDelta: OctreePowerRowDeltaSource): GPUBindGroup {
    const cached = this.cachedGroup;
    if (cached && cached.descriptors === descriptors && cached.rowDelta === rowDelta.rows) return cached.group;
    const group = this.device.createBindGroup({ label: "Octree power topology bindings", layout: this.layout, entries: [
      { binding: 0, resource: { buffer: this.params } }, { binding: 1, resource: { buffer: descriptors } },
      { binding: 2, resource: { buffer: this.candidateMetrics } }, { binding: 3, resource: { buffer: this.control } },
      { binding: 4, resource: { buffer: this.catalogLookup } }, { binding: 5, resource: { buffer: this.catalogVolumes } },
      { binding: 6, resource: { buffer: this.sameOrFinerDirect } }, { binding: 7, resource: { buffer: this.sameOrCoarserDirect } },
      { binding: 9, resource: { buffer: rowDelta.rows } },
      { binding: 10, resource: { buffer: this.metrics } },
    ] });
    this.cachedGroup = { descriptors, rowDelta: rowDelta.rows, group };
    return group;
  }

  encodeCandidate(broker: PassBroker, descriptors: GPUBuffer,
    physicalSpacing: readonly [number, number, number], rowDelta: OctreePowerRowDeltaSource): void {
    if (this.destroyed) throw new Error("Octree power topology is destroyed");
    assertOctreePowerRowDeltaLayout(rowDelta, this.plan.rowCapacity, "Power topology");
    const anisotropic = powerCellSpacingIsotropic(physicalSpacing) ? 0 : 1;
    const parameterBytes = new ArrayBuffer(80);
    const parameterWords = new Uint32Array(parameterBytes);
    parameterWords.set([this.plan.rowCapacity, this.plan.lookupCount, anisotropic,
      OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version, this.regularPacked,
      0, OCTREE_POWER_ROW_TEMPLATE_VERSION, 0]);
    parameterWords.set([rowDelta.controlOffsetWords, rowDelta.newToOldOffsetWords,
      rowDelta.affectedRowsOffsetWords, 1], 8);
    parameterWords.set([
      this.rowTemplateHeaderOffsetBytes / 4,
      this.rowTemplateSlotOffsetBytes / 4,
      this.rowTemplateDataOffsetBytes / 4,
      this.rowTemplateDiagonalOffsetBytes / 4,
      this.reconstructionDataOffsetBytes / 4,
      this.plan.entryCount,
      (this.rowTemplateDataOffsetBytes - this.rowTemplateSlotOffsetBytes) / 4,
      0,
    ], 12);
    new Float32Array(parameterBytes)[5] = this.regularVolume;
    this.device.queue.writeBuffer(this.params, 0, parameterBytes);
    const group = this.createGroup(descriptors, rowDelta);
    let pass = broker.compute({ label: "Prepare power topology delta" });
    pass.setPipeline(this.preparePipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1);
    broker.copyBufferToBuffer(rowDelta.rows, (rowDelta.controlOffsetWords + 12) * 4,
      this.workDispatch, 0, 12);
    pass = broker.compute({ label: "Resolve affected power topology rows" });
    pass.setPipeline(this.resolvePipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    broker.copyBufferToBuffer(this.control, 64, this.workDispatch, 0, 12);
    pass = broker.compute({ label: "Stage exact power topology carry and affected rows" });
    pass.setPipeline(this.stagePipeline); pass.setBindGroup(0, group);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    pass.setPipeline(this.prefixPipeline); pass.dispatchWorkgroups(1);
    pass.setPipeline(this.scatterPipeline); pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
  }

  encodeReadyCommit(broker: PassBroker): void {
    if (this.destroyed) throw new Error("Octree power topology is destroyed");
    const group = this.cachedGroup?.group;
    if (!group) throw new Error("Power topology candidate has not been encoded");
    const pass = broker.compute({ label: "Commit power topology publication" });
    pass.setBindGroup(0, group); pass.setPipeline(this.commitPipeline);
    pass.dispatchWorkgroupsIndirect(this.workDispatch, 0);
    pass.setPipeline(this.finalizePipeline); pass.dispatchWorkgroups(1);
  }

  encode(broker: PassBroker, descriptors: GPUBuffer,
    physicalSpacing: readonly [number, number, number], rowDelta: OctreePowerRowDeltaSource): void {
    this.encodeCandidate(broker, descriptors, physicalSpacing, rowDelta);
    this.encodeReadyCommit(broker);
  }

  get source(): OctreePowerTopologySource {
    return { plan: this.plan, metrics: this.metrics, control: this.control, catalogEntryHeaders: this.catalogEntryHeaders,
      catalogVolumes: this.catalogVolumes, catalogFaces: this.catalogFaces, catalogLookup: this.catalogLookup,
      catalogCoefficients: this.catalogCoefficients,
      rowTemplateHeaders: this.rowTemplateHeaders, rowTemplateSlots: this.rowTemplateSlots,
      rowTemplateData: this.rowTemplateData, rowTemplateDiagonals: this.rowTemplateDiagonals,
      reconstructionData: this.reconstructionData,
      rowTemplateHeaderOffsetBytes: this.rowTemplateHeaderOffsetBytes,
      rowTemplateSlotOffsetBytes: this.rowTemplateSlotOffsetBytes,
      rowTemplateDataOffsetBytes: this.rowTemplateDataOffsetBytes,
      rowTemplateDiagonalOffsetBytes: this.rowTemplateDiagonalOffsetBytes,
      reconstructionDataOffsetBytes: this.reconstructionDataOffsetBytes,
      catalogTetrahedronHeaders: this.catalogTetrahedronHeaders, catalogTetrahedra: this.catalogTetrahedra,
      catalogTetrahedronVertices: this.catalogTetrahedronVertices,
      catalogTetrahedronVertexCount: this.catalogTetrahedronVertexCount,
      catalogCompiledSampler: this.catalogCompiledSampler,
      sameOrFinerDirect: this.sameOrFinerDirect, sameOrCoarserDirect: this.sameOrCoarserDirect };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.metrics.destroy(); this.candidateMetrics.destroy();
    this.control.destroy(); this.params.destroy(); this.workDispatch.destroy();
    this.catalogEntryHeaders.destroy(); this.catalogVolumes.destroy(); this.catalogFaces.destroy();
    this.catalogCoefficients.destroy(); this.catalogLookup.destroy();
    this.catalogTetrahedronHeaders.destroy(); this.catalogTetrahedra.destroy();
    this.catalogTetrahedronVertices.destroy();
    this.catalogCompiledSampler.destroy();
    this.sameOrFinerDirect.destroy(); this.sameOrCoarserDirect.destroy();
  }

}

export const octreePowerTopologyShader = /* wgsl */ `
struct Params {
  rowCount:u32, lookupCount:u32, anisotropic:u32, catalogVersion:u32,
  regularPacked:u32, regularVolume:f32, rowTemplateVersion:u32, pad0:u32,
  delta:vec4u,
  template0:vec4u,
  template1:vec4u,
}
struct PowerRowMetric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct Control { invalidCount:u32, firstInvalid:u32, flags:u32, resolvedCount:u32, version:u32, pad0:u32, pad1:u32, pad2:u32 }
// Candidate is first because it is the public current-attempt diagnostic.
// Authority is private retained state used only to validate immutable carry.
struct ControlArena {
  candidate:Control,
  authority:Control,
  blockDispatch:vec3u,
  commitDispatch:vec3u,
  publicationCount:u32,
  blockCount:u32,
}
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> descriptors:array<u32>;
@group(0) @binding(2) var<storage,read_write> metrics:array<PowerRowMetric>;
@group(0) @binding(3) var<storage,read_write> controlArena:ControlArena;
@group(0) @binding(4) var<storage,read_write> lookupArena:array<u32>;
@group(0) @binding(5) var<storage,read> denseCatalog:array<u32>;
@group(0) @binding(6) var<storage,read> sameOrFinerDirect:array<u32>;
@group(0) @binding(7) var<storage,read> sameOrCoarserDirect:array<u32>;
@group(0) @binding(9) var<storage,read> rowDelta:array<u32>;
@group(0) @binding(10) var<storage,read_write> committedMetrics:array<PowerRowMetric>;
const INVALID:u32=0xffffffffu;
const ROW_DELTA_VALID:u32=${OCTREE_POWER_ROW_DELTA_VALID}u;
const ROW_DELTA_AFFECTED:u32=0x80000000u;
const VALID:u32=0x80000000u;
const ANISOTROPIC:u32=1u;
const LOOKUP_MISS:u32=2u;
const CATALOG_VERSION:u32=4u;
const CAPACITY:u32=8u;
const EXPECTED_CATALOG_VERSION:u32=${OCTREE_GENERATED_POWER_CATALOG_MANIFEST.version}u;
const EXPECTED_ROW_TEMPLATE_VERSION:u32=${OCTREE_POWER_ROW_TEMPLATE_VERSION}u;
const REGULAR_DESCRIPTOR:u32=0x3ffffu;
const STATUS_VALID:u32=0x80000000u;
const STATUS_PUBLISH:u32=0x40000000u;
const STATUS_AFFECTED:u32=0x20000000u;
const STATUS_LISTED:u32=0x10000000u;
fn powerTransformVector(value:vec3i,code:u32)->vec3i{
  let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));let permutation=(code/8u)%6u;var q=value;
  if(permutation==1u){q=value.xzy;}else if(permutation==2u){q=value.yxz;}else if(permutation==3u){q=value.yzx;}else if(permutation==4u){q=value.zxy;}else if(permutation==5u){q=value.zyx;}return q*signs;
}
fn boundaryDirectionBit(direction:vec3i)->u32{if(direction.x<0){return 0u;}if(direction.y<0){return 1u;}if(direction.z<0){return 2u;}if(direction.z>0){return 3u;}if(direction.y>0){return 4u;}return 5u;}
fn transformBoundaryMask(mask:u32,transform:u32)->u32{let directions=array<vec3i,6>(vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1),vec3i(0,0,1),vec3i(0,1,0),vec3i(1,0,0));var result=0u;for(var bit=0u;bit<6u;bit+=1u){if((mask&(1u<<bit))!=0u){result|=1u<<boundaryDirectionBit(powerTransformVector(directions[bit],transform));}}return result;}
fn resolveBoundaryEntry(interiorEntry:u32,canonicalMask:u32)->u32{let key=interiorEntry*64u+canonicalMask;var low=0u;var high=min(params.lookupCount,arrayLength(&lookupArena)/3u);while(low<high){let middle=low+(high-low)/2u;let candidate=lookupArena[3u*middle];if(candidate<key){low=middle+1u;}else{high=middle;}}if(low>=min(params.lookupCount,arrayLength(&lookupArena)/3u)||lookupArena[3u*low]!=key){return INVALID;}return lookupArena[3u*low+1u];}
fn resolveDescriptor(descriptor:u32)->vec2u{let boundary=(descriptor>>24u)&63u;let geometry=descriptor&0xc0ffffffu;if(geometry==REGULAR_DESCRIPTOR&&boundary==0u){return vec2u(params.regularPacked&0xffffu,params.regularPacked>>16u);}var packed=INVALID;if((geometry&0x80000000u)!=0u){let index=geometry&0x1ffu;if((geometry&0x40fffe00u)==0u&&index<arrayLength(&sameOrCoarserDirect)){packed=sameOrCoarserDirect[index];}}else{let index=geometry&0x3ffffu;if((geometry&0x40fc0000u)==0u&&index<arrayLength(&sameOrFinerDirect)){packed=sameOrFinerDirect[index];}}if(packed==INVALID){return vec2u(INVALID);}let transform=packed>>16u;var entry=packed&0xffffu;if(boundary!=0u){entry=resolveBoundaryEntry(entry,transformBoundaryMask(boundary,transform));if(entry==INVALID){return vec2u(INVALID);}}return vec2u(entry,transform);}
fn finiteTemplate(value:f32)->bool{return value==value&&abs(value)<3.402823e38;}
fn rowTemplateValid(caseId:u32)->bool{
  if(caseId>=params.template1.y){return false;}
  let headerBase=params.template0.x+caseId*4u;
  if(headerBase>arrayLength(&denseCatalog)||4u>arrayLength(&denseCatalog)-headerBase){return false;}
  let header=vec4u(denseCatalog[headerBase],denseCatalog[headerBase+1u],
    denseCatalog[headerBase+2u],denseCatalog[headerBase+3u]);
  let first=header.x;let count=header.y;
  if(count==0u||count>30u||header.w!=count||first>params.template1.z
      ||count>params.template1.z-first){return false;}
  let finalSlot=first+count;let finalData=finalSlot*3u;
  if(params.template0.y+finalSlot>arrayLength(&denseCatalog)
      ||params.template0.z+finalData>arrayLength(&denseCatalog)
      ||params.template1.x+finalData>arrayLength(&denseCatalog)
      ||params.template0.w+caseId>=arrayLength(&denseCatalog)){return false;}
  if(caseId==0u&&(first!=0u||count!=6u||(header.z&1u)==0u)){return false;}
  var diagonal=0.0;
  for(var local=0u;local<count;local+=1u){
    let slot=first+local;let packed=denseCatalog[params.template0.y+slot];let data=slot*3u;
    let world=(packed&(1u<<21u))!=0u;let invalidSelector=(packed&255u)==255u;
    let boundaryTransfer=((packed>>14u)&3u)==3u;
    if((packed&0xffc00000u)!=0u||((packed>>8u)&7u)>=6u||((packed>>16u)&31u)!=local
        ||world!=invalidSelector||world!=boundaryTransfer){return false;}
    let coefficient=bitcast<f32>(denseCatalog[params.template0.z+data]);
    let area=bitcast<f32>(denseCatalog[params.template0.z+data+1u]);
    let inverseDistance=bitcast<f32>(denseCatalog[params.template0.z+data+2u]);
    if(!(coefficient>0.0)||!(area>0.0)||!(inverseDistance>0.0)
        ||!finiteTemplate(coefficient)||!finiteTemplate(area)||!finiteTemplate(inverseDistance)
        ||!finiteTemplate(bitcast<f32>(denseCatalog[params.template1.x+data]))
        ||!finiteTemplate(bitcast<f32>(denseCatalog[params.template1.x+data+1u]))
        ||!finiteTemplate(bitcast<f32>(denseCatalog[params.template1.x+data+2u]))){return false;}
    diagonal+=coefficient;
  }
  let packedDiagonal=denseCatalog[params.template0.w+caseId];
  return bitcast<u32>(diagonal)==packedDiagonal&&finiteTemplate(bitcast<f32>(packedDiagonal));
}
fn requestedRows()->u32{return rowDelta[params.delta.x];}
fn deltaAccepted(requested:u32)->bool{if(params.delta.x+15u>=arrayLength(&rowDelta)){return false;}let base=params.delta.x;let previous=rowDelta[base+1u];let carried=rowDelta[base+2u];let added=rowDelta[base+3u];let retired=rowDelta[base+4u];let affected=rowDelta[base+6u];return rowDelta[base]==requested&&rowDelta[base+8u]==ROW_DELTA_VALID&&carried<=previous&&affected<=requested&&params.delta.y<=arrayLength(&rowDelta)&&requested<=arrayLength(&rowDelta)-params.delta.y&&params.delta.z<=arrayLength(&rowDelta)&&affected<=arrayLength(&rowDelta)-params.delta.z&&requested==carried+added&&requested==previous+added-retired;}
fn affectedRow(item:u32)->u32{return rowDelta[params.delta.z+item];}
fn affectedCount()->u32{return rowDelta[params.delta.x+6u];}
fn topologyCandidateScratchReusable()->bool{
  let candidate=controlArena.candidate;let authority=controlArena.authority;
  return candidate.invalidCount==0u&&candidate.firstInvalid==INVALID&&candidate.flags==0u
    &&candidate.resolvedCount==authority.resolvedCount&&candidate.version==authority.version;
}
fn exactTopologyIdentity(requested:u32,reusableScratch:bool)->bool{
  if(!deltaAccepted(requested)||params.delta.x+15u>=arrayLength(&rowDelta)){return false;}
  let base=params.delta.x;
  return rowDelta[base+1u]==requested&&rowDelta[base+2u]==requested
    &&rowDelta[base+3u]==0u&&rowDelta[base+4u]==0u
    &&rowDelta[base+5u]==0u&&rowDelta[base+6u]==0u
    &&rowDelta[base+15u]==1u
    &&controlArena.authority.invalidCount==0u&&controlArena.authority.flags==0u
    &&controlArena.authority.resolvedCount==requested
    &&controlArena.authority.version==params.catalogVersion&&reusableScratch;
}
fn linearItem(wid:vec3u,lid:u32,workgroups:vec3u)->u32{return (wid.x+wid.y*workgroups.x)*64u+lid;}
fn fail(row:u32,flag:u32){if(row<arrayLength(&metrics)){metrics[row]=PowerRowMetric(INVALID,0u,0.0,flag);}}
fn metricValid(metric:PowerRowMetric)->bool{return metric.reserved==0u&&(metric.transformAndFlags&VALID)!=0u;}
fn metricFlags(metric:PowerRowMetric)->u32{return select(CAPACITY,metric.reserved,metric.reserved!=0u);}
fn dispatchFor(count:u32)->vec3u{let groups=(count+63u)/64u;let x=min(groups,65535u);return vec3u(x,select(1u,(groups+x-1u)/x,x>0u),1u);}
fn blockDispatchFor(count:u32)->vec3u{let groups=(count+255u)/256u;let x=min(groups,65535u);return vec3u(x,select(1u,(groups+x-1u)/x,x>0u),1u);}
fn arenaBase()->u32{return 3u*params.lookupCount;}
fn statusBase()->u32{return arenaBase();}
fn prefixBase()->u32{return statusBase()+params.rowCount;}
fn publicationBase()->u32{return prefixBase()+params.rowCount;}
fn blockOffsetBase()->u32{return publicationBase()+params.rowCount;}
fn blockSummaryBase()->u32{return blockOffsetBase()+controlArena.blockCount+1u;}
fn rejectCandidate(row:u32,flags:u32){controlArena.candidate.invalidCount+=1u;controlArena.candidate.firstInvalid=min(controlArena.candidate.firstInvalid,row);controlArena.candidate.flags|=flags;}
var<workgroup> publicationScan:array<u32,256>;
var<workgroup> publicationCounts:array<vec4u,256>;
var<workgroup> publicationAux:array<vec2u,256>;
@compute @workgroup_size(1) fn preparePowerTopology(){
  let requested=requestedRows();
  let reusableScratch=topologyCandidateScratchReusable();
  let available=min(requested,min(arrayLength(&descriptors),min(arrayLength(&metrics),arrayLength(&committedMetrics))));
  let anisotropic=params.anisotropic!=0u;
  controlArena.candidate=Control(select(0u,requested,anisotropic),
    select(INVALID,0u,anisotropic&&requested!=0u),select(0u,ANISOTROPIC,anisotropic),
    0u,params.catalogVersion,0u,0u,0u);
  controlArena.blockDispatch=vec3u(0u,1u,1u);controlArena.commitDispatch=vec3u(0u,1u,1u);
  controlArena.publicationCount=0u;controlArena.blockCount=0u;
  if(params.catalogVersion!=EXPECTED_CATALOG_VERSION
      ||params.rowTemplateVersion!=EXPECTED_ROW_TEMPLATE_VERSION){
    rejectCandidate(0u,CATALOG_VERSION);return;
  }
  // Case zero is the immutable regular-grid contract. Validate it once per
  // transaction; exact regular rows below can then publish their metric
  // without repeating six template-slot/catalog walks per row.
  if(!rowTemplateValid(0u)){rejectCandidate(0u,CATALOG_VERSION);return;}
  if(!deltaAccepted(requested)){rejectCandidate(0u,CAPACITY);return;}
  let previous=rowDelta[params.delta.x+1u];
  if(previous>arrayLength(&committedMetrics)
      ||(previous!=0u&&(controlArena.authority.invalidCount!=0u||controlArena.authority.flags!=0u
        ||controlArena.authority.resolvedCount!=previous||controlArena.authority.version!=params.catalogVersion))){
    rejectCandidate(0u,CAPACITY);return;
  }
  if(available<requested){controlArena.candidate.invalidCount=requested-available;controlArena.candidate.firstInvalid=available;controlArena.candidate.flags|=CAPACITY;}
  if(controlArena.candidate.invalidCount==0u&&exactTopologyIdentity(requested,reusableScratch)){
    controlArena.candidate=controlArena.authority;
    // See the descriptor transaction: INVALID is candidate-private and marks
    // a complete identity carry whose row-shaped schedules stay zero.
    controlArena.publicationCount=INVALID;
    return;
  }
  let blocks=(requested+255u)/256u;controlArena.blockCount=blocks;
  if(blockSummaryBase()+6u*blocks>arrayLength(&lookupArena)){rejectCandidate(0u,CAPACITY);return;}
  controlArena.blockDispatch=blockDispatchFor(requested);
}
@compute @workgroup_size(64) fn resolvePowerTopology(@builtin(local_invocation_index) lid:u32,@builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) workgroups:vec3u){
  let item=linearItem(wid,lid,workgroups);if(item>=affectedCount()||controlArena.candidate.flags!=0u){return;}
  let row=affectedRow(item);let requested=requestedRows();let available=min(requested,min(arrayLength(&descriptors),arrayLength(&metrics)));
  if(row>=available){return;}lookupArena[statusBase()+row]=STATUS_LISTED;
  if(params.anisotropic!=0u){fail(row,ANISOTROPIC);return;}
  if((item>0u&&affectedRow(item-1u)>=row)||(rowDelta[params.delta.y+row]&ROW_DELTA_AFFECTED)==0u){fail(row,CAPACITY);return;}
  let descriptor=descriptors[row];let boundary=(descriptor>>16u)&0x3f00u;let geometry=descriptor&0xc0ffffffu;
  if(geometry==REGULAR_DESCRIPTOR&&boundary==0u){
    metrics[row]=PowerRowMetric(params.regularPacked&0xffffu,
      (params.regularPacked>>16u)|VALID,params.regularVolume,0u);return;
  }
  let found=resolveDescriptor(descriptor);
  if(found.x==INVALID||found.x>=params.template1.y||found.x>=arrayLength(&denseCatalog)){fail(row,LOOKUP_MISS);return;}
  if(!rowTemplateValid(found.x)){fail(row,CATALOG_VERSION);return;}
  var volume=bitcast<f32>(denseCatalog[found.x]);
  metrics[row]=PowerRowMetric(found.x,found.y|boundary|VALID,volume,0u);
}
@compute @workgroup_size(256) fn stagePowerTopologyDelta(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;let requested=requestedRows();var status=0u;
  if(row<requested&&controlArena.candidate.flags==0u){
    let encoded=rowDelta[params.delta.y+row];let flagged=(encoded&ROW_DELTA_AFFECTED)!=0u;
    let listed=(lookupArena[statusBase()+row]&STATUS_LISTED)!=0u;
    let oldPlusOne=encoded&0x3fffffffu;var metric=PowerRowMetric(INVALID,0u,0.0,CAPACITY);var flags=0u;
    if(flagged!=listed){flags|=CAPACITY;}
    if(flagged){metric=metrics[row];status|=STATUS_AFFECTED|STATUS_PUBLISH;}
    else if(oldPlusOne==0u){flags|=CAPACITY;}
    else{let old=oldPlusOne-1u;if(old>=controlArena.authority.resolvedCount||old>=arrayLength(&committedMetrics)){flags|=CAPACITY;}
      else{
        metric=committedMetrics[old];
        // Section 4's pressure/normal-velocity discretization is defined on
        // the complete power-cell face geometry. Republish every carried
        // metric into candidate scratch; the sparse commit bit still records
        // only rows whose stable index changed.
        metrics[row]=metric;if(old!=row){status|=STATUS_PUBLISH;}
      }}
    if(!metricValid(metric)){flags|=metricFlags(metric);}else{status|=STATUS_VALID;}
    status|=flags&255u;lookupArena[statusBase()+row]=status;
  }
  let publish=select(0u,1u,(status&STATUS_PUBLISH)!=0u);publicationScan[lane]=publish;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){var add=0u;if(lane>=offset){add=publicationScan[lane-offset];}
    workgroupBarrier();publicationScan[lane]+=add;workgroupBarrier();}
  if(row<requested){lookupArena[prefixBase()+row]=publicationScan[lane]-publish;}
  let flags=status&255u;publicationCounts[lane]=vec4u(select(0u,1u,(status&STATUS_VALID)!=0u),
    select(0u,1u,flags!=0u),flags,select(0u,1u,(status&STATUS_AFFECTED)!=0u));
  publicationAux[lane]=vec2u(select(INVALID,row,flags!=0u),0u);workgroupBarrier();
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){publicationCounts[lane].x+=publicationCounts[lane+stride].x;
      publicationCounts[lane].y+=publicationCounts[lane+stride].y;publicationCounts[lane].z|=publicationCounts[lane+stride].z;
      publicationCounts[lane].w+=publicationCounts[lane+stride].w;
      publicationAux[lane].x=min(publicationAux[lane].x,publicationAux[lane+stride].x);}workgroupBarrier();}
  if(lane==0u){let at=blockSummaryBase()+6u*block;let counts=publicationCounts[0];
    lookupArena[at]=counts.x;lookupArena[at+1u]=counts.y;lookupArena[at+2u]=counts.z;
    lookupArena[at+3u]=publicationAux[0].x;lookupArena[at+4u]=lookupArena[prefixBase()+min(requested-1u,block*256u+255u)]+select(0u,1u,(lookupArena[statusBase()+min(requested-1u,block*256u+255u)]&STATUS_PUBLISH)!=0u);
    lookupArena[at+5u]=counts.w;}
}
@compute @workgroup_size(256) fn prefixPowerTopologyDelta(@builtin(local_invocation_index) lane:u32){
  let blocks=controlArena.blockCount;let chunk=(blocks+255u)/256u;let begin=min(blocks,lane*chunk);let end=min(blocks,begin+chunk);
  var counts=vec4u(0u);var changed=0u;var firstInvalid=INVALID;var affected=0u;
  for(var block=begin;block<end;block+=1u){let at=blockSummaryBase()+6u*block;counts.x+=lookupArena[at];
    counts.y+=lookupArena[at+1u];counts.z|=lookupArena[at+2u];counts.w=min(counts.w,lookupArena[at+3u]);
    firstInvalid=min(firstInvalid,lookupArena[at+3u]);changed+=lookupArena[at+4u];affected+=lookupArena[at+5u];}
  publicationCounts[lane]=counts;publicationAux[lane]=vec2u(firstInvalid,affected);
  publicationScan[lane]=changed;workgroupBarrier();
  for(var offset=1u;offset<256u;offset*=2u){var add=0u;if(lane>=offset){add=publicationScan[lane-offset];}
    workgroupBarrier();publicationScan[lane]+=add;workgroupBarrier();}
  var cursor=publicationScan[lane]-changed;
  for(var block=begin;block<end;block+=1u){let at=blockSummaryBase()+6u*block;lookupArena[blockOffsetBase()+block]=cursor;cursor+=lookupArena[at+4u];}
  for(var stride=128u;stride>0u;stride/=2u){if(lane<stride){publicationCounts[lane].x+=publicationCounts[lane+stride].x;
      publicationCounts[lane].y+=publicationCounts[lane+stride].y;publicationCounts[lane].z|=publicationCounts[lane+stride].z;
      publicationAux[lane].x=min(publicationAux[lane].x,publicationAux[lane+stride].x);
      publicationAux[lane].y+=publicationAux[lane+stride].y;}workgroupBarrier();}
  if(lane==0u&&controlArena.candidate.invalidCount==0u&&controlArena.candidate.flags==0u&&controlArena.publicationCount!=INVALID){
    let totals=publicationCounts[0];let aux=publicationAux[0];let published=publicationScan[255];lookupArena[blockOffsetBase()+blocks]=published;
    controlArena.publicationCount=published;controlArena.candidate.resolvedCount=totals.x;
    controlArena.candidate.invalidCount+=totals.y;controlArena.candidate.flags|=totals.z;
    controlArena.candidate.firstInvalid=min(controlArena.candidate.firstInvalid,aux.x);
    if(aux.y!=affectedCount()||totals.x+totals.y!=requestedRows()){rejectCandidate(0u,CAPACITY);}
    if(controlArena.candidate.invalidCount==0u&&controlArena.candidate.flags==0u){controlArena.commitDispatch=dispatchFor(published);}
  }
}
@compute @workgroup_size(256) fn scatterPowerTopologyDelta(
    @builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) wid:vec3u,
    @builtin(num_workgroups) workgroups:vec3u){
  if(controlArena.candidate.invalidCount!=0u||controlArena.candidate.flags!=0u){return;}
  let block=wid.x+wid.y*workgroups.x;let row=block*256u+lane;if(row>=requestedRows()){return;}
  if((lookupArena[statusBase()+row]&STATUS_PUBLISH)==0u){return;}
  let output=lookupArena[blockOffsetBase()+block]+lookupArena[prefixBase()+row];
  if(output<controlArena.publicationCount&&row<arrayLength(&metrics)){
    lookupArena[publicationBase()+output]=row;
  }
}
@compute @workgroup_size(256) fn commitPowerTopologyDelta(
    @builtin(workgroup_id) wid:vec3u,@builtin(num_workgroups) workgroups:vec3u,@builtin(local_invocation_index) lane:u32){
  if(controlArena.candidate.invalidCount!=0u||controlArena.candidate.flags!=0u){return;}
  let item=(wid.x+wid.y*workgroups.x)*256u+lane;if(item>=controlArena.publicationCount){return;}
  let row=lookupArena[publicationBase()+item];if(row>=arrayLength(&metrics)||row>=arrayLength(&committedMetrics)){return;}
  committedMetrics[row]=metrics[row];
}
@compute @workgroup_size(1) fn finalizePowerTopology(){
  let candidate=controlArena.candidate;
  if(candidate.invalidCount!=0u||candidate.flags!=0u||candidate.resolvedCount!=requestedRows()||candidate.version!=params.catalogVersion){return;}
  controlArena.authority=candidate;
}
`;
