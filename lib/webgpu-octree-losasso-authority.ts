import {
  OCTREE_LOSASSO_CONTROL_WORDS,
  OCTREE_LOSASSO_FACE_BYTES,
  type OctreeLosassoPackedAuthority,
} from "./octree-losasso-operator";
import type { PassBroker } from "./webgpu-pass-broker";
import type {
  OctreeLosassoVCycleLevelSource,
  OctreeLosassoVCycleTransferSource,
} from "./webgpu-octree-losasso-vcycle";
import type {
  WebGPUOctreeLosassoCandidateInput,
  WebGPUOctreeLosassoCandidatePublisher,
  WebGPUOctreeLosassoPublishedSources,
} from "./webgpu-octree-losasso-backend";
import { planOctreeLosassoAxisFaceDirectory } from "./webgpu-octree-losasso-velocity-sampler";
import type { WebGPUOctreeLosassoDynamicsSource } from "./webgpu-octree-losasso-dynamics";

export interface WebGPUOctreeLosassoAuthorityCapacities {
  readonly rows: number;
  readonly faces: number;
  readonly incidences?: number;
  readonly faceAdjacencies?: number;
}

export interface WebGPUOctreeLosassoWritableAuthority {
  readonly capacities: Readonly<Required<WebGPUOctreeLosassoAuthorityCapacities>>;
  readonly control: GPUBuffer;
  readonly rowFaceOffsets: GPUBuffer;
  readonly rowFaces: GPUBuffer;
  readonly faces: GPUBuffer;
  readonly rowDispatch: GPUBuffer;
  readonly faceDispatch: GPUBuffer;
  readonly rightHandSide: GPUBuffer;
  readonly faceMetrics: GPUBuffer;
  readonly faceAdjacencyOffsets: GPUBuffer;
  readonly faceAdjacency: GPUBuffer;
  readonly projectedVelocity: GPUBuffer;
  readonly extendedVelocity: GPUBuffer;
  readonly advectedVelocity: GPUBuffer;
  readonly predictedVelocity: GPUBuffer;
  /** vec4u(axis,x,y,z) in finest-cell staggered coordinates. */
  readonly faceGeometry: GPUBuffer;
  /** Open-addressed vec2u(facePlusOne,hash) sampler directory. */
  readonly axisFaceDirectory: GPUBuffer;
  readonly faceDirectoryCapacity: number;
  /** One closed-form matrix diagonal per row for the generic wide solver. */
  readonly diagonal: GPUBuffer;
  /** Generic seven-word accepted-row control consumed by wide MGPCG. */
  readonly solverAuthority: GPUBuffer;
}

export type WebGPUOctreeLosassoCandidateEncoder = (
  broker: PassBroker,
  input: WebGPUOctreeLosassoCandidateInput,
  output: WebGPUOctreeLosassoWritableAuthority,
) => void;

export interface WebGPUOctreeLosassoAuthorityOptions {
  readonly capacities: WebGPUOctreeLosassoAuthorityCapacities;
  /** Coarser closed-form levels after the owned finest authority. */
  readonly coarseLevels?: readonly OctreeLosassoVCycleLevelSource[];
  readonly transfers?: readonly OctreeLosassoVCycleTransferSource[];
  /** GPU candidate builder; receives only shared leaf/phi/solid sources. */
  readonly encodeCandidate?: WebGPUOctreeLosassoCandidateEncoder;
}

function capacity(value: number | undefined, fallback: number, label: string): number {
  const result = value ?? fallback;
  if (!Number.isSafeInteger(result) || result < 1) throw new RangeError(`Losasso ${label} capacity must be positive`);
  return result;
}

/**
 * Concrete reduced-buffer owner for the Losasso backend. It is deliberately
 * allocated without a descriptor, catalogue, LUT, edge channel, or power face
 * arena. A topology publisher can write these buffers directly on GPU through
 * `encodeCandidate`; `uploadAccepted` is the cold/bootstrap host route.
 */
export class WebGPUOctreeLosassoAuthority implements WebGPUOctreeLosassoCandidatePublisher {
  readonly initializationTasks = [] as const;
  /** Fixed buffers consumed by the live solver/dynamics graph. */
  readonly writable: WebGPUOctreeLosassoWritableAuthority;
  /** Isolated rebuild bank; never visible through `sources` before ready commit. */
  readonly candidate: WebGPUOctreeLosassoWritableAuthority;
  readonly sources: WebGPUOctreeLosassoPublishedSources;
  readonly allocatedBytes: number;
  private destroyed = false;

  constructor(private readonly device: GPUDevice,
    private readonly options: WebGPUOctreeLosassoAuthorityOptions) {
    const rows = capacity(options.capacities.rows, 0, "row");
    const faces = capacity(options.capacities.faces, 0, "face");
    const incidences = capacity(options.capacities.incidences, 2 * faces, "incidence");
    const faceAdjacencies = capacity(options.capacities.faceAdjacencies, 12 * faces,
      "face-adjacency");
    if (incidences > 2 * faces) throw new RangeError("Losasso incidence capacity exceeds unique-face bound");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const indirect = storage | GPUBufferUsage.INDIRECT;
    const faceDirectoryCapacity = planOctreeLosassoAxisFaceDirectory(faces).directoryCapacity;
    const capacities = Object.freeze({ rows, faces, incidences, faceAdjacencies });
    const allocateBank = (bank: "accepted" | "candidate") => {
      const make = (label: string, size: number, usage = storage) => device.createBuffer({
        label: `Losasso ${bank} ${label}`, size: Math.max(16, size), usage,
      });
      return Object.freeze({ capacities,
        control: make("coarse authority", OCTREE_LOSASSO_CONTROL_WORDS * 4),
        rowFaceOffsets: make("row-face CSR offsets", (rows + 1) * 4),
        rowFaces: make("row-face CSR incidence", incidences * 4),
        faces: make("unique axis faces", faces * OCTREE_LOSASSO_FACE_BYTES),
        rowDispatch: make("GPU-authored row dispatch", 12, indirect),
        // The second indirect record carries the epoch-local directory grid.
        faceDispatch: make("GPU-authored face and directory dispatch", 24, indirect),
        rightHandSide: make("integrated pressure RHS", rows * 4),
        faceMetrics: make("axis-face extension metrics", faces * 16),
        faceAdjacencyOffsets: make("axis-face adjacency offsets", (faces + 1) * 4),
        faceAdjacency: make("axis-face adjacency", faceAdjacencies * 4),
        projectedVelocity: make("projected axis-face velocity", faces * 4),
        extendedVelocity: make("lagged extended axis-face velocity", faces * 4),
        advectedVelocity: make("advected axis-face velocity", faces * 4),
        predictedVelocity: make("predicted axis-face velocity", faces * 4),
        faceGeometry: make("axis-face finest-grid geometry", faces * 16),
        axisFaceDirectory: make("axis-face sampling directory", faceDirectoryCapacity * 8),
        faceDirectoryCapacity,
        diagonal: make("first-order operator diagonal", rows * 4),
        solverAuthority: make("wide-solver authority", 7 * 4),
      }) satisfies WebGPUOctreeLosassoWritableAuthority;
    };
    this.writable = allocateBank("accepted");
    this.candidate = allocateBank("candidate");
    const { control, rowFaceOffsets, rowFaces, faces: faceBuffer, rowDispatch,
      faceDispatch, rightHandSide, faceMetrics, faceAdjacencyOffsets, faceAdjacency,
      projectedVelocity, extendedVelocity, advectedVelocity, predictedVelocity,
      faceGeometry, axisFaceDirectory } = this.writable;
    const finest: OctreeLosassoVCycleLevelSource = {
      rowCapacity: rows, control, rowFaceOffsets, rowFaces, faces: faceBuffer, rowDispatch,
    };
    const levels = [finest, ...(options.coarseLevels ?? [])];
    const transfers = options.transfers ?? [];
    this.sources = Object.freeze({
      operator: { rowCapacity: rows, control, rowFaceOffsets, rowFaces, faces: faceBuffer, rowDispatch },
      projection: { control, faces: faceBuffer, faceDispatch, predictedVelocity,
        projectedVelocity },
      dynamics: {
        faceCapacity: faces, rowCapacity: rows, control, faces: faceBuffer,
        faceGeometry, axisFaceDirectory, faceDirectoryCapacity, extendedVelocity,
        advectedVelocity, predictedVelocity, projectedVelocity, rowFaceOffsets, rowFaces,
        rightHandSide, faceDispatch, rowDispatch,
      } satisfies WebGPUOctreeLosassoDynamicsSource,
      extension: { faceCapacity: faces, control, faceMetrics,
        adjacencyOffsets: faceAdjacencyOffsets, adjacencyFaces: faceAdjacency,
        projectedVelocity, extendedVelocity },
      vcycle: { levels, transfers }, rightHandSide, rowCount: control,
    });
    this.allocatedBytes = [...Object.values(this.writable), ...Object.values(this.candidate)]
      .filter((value): value is GPUBuffer => typeof value === "object" && value !== null && "size" in value)
      .reduce((sum, buffer) => sum + buffer.size, 0);
  }

  encodeCandidatePublication(broker: PassBroker, input: WebGPUOctreeLosassoCandidateInput): void {
    this.assertLive();
    if (!this.options.encodeCandidate) {
      throw new Error("Losasso authority needs a GPU candidate encoder for dynamic topology publication");
    }
    this.options.encodeCandidate(broker, input, this.candidate);
  }

  /** Host bootstrap route. Dynamic advances should use the GPU candidate encoder. */
  uploadAccepted(authority: OctreeLosassoPackedAuthority, input: {
    readonly rightHandSide: Float32Array;
    readonly faceMetrics: Uint32Array;
    readonly faceAdjacencyOffsets: Uint32Array;
    readonly faceAdjacency: Uint32Array;
  }): void {
    this.assertLive();
    const counts = { rows: authority.control[1]!, faces: authority.control[2]!,
      incidences: authority.rowFaces.length, faceAdjacencies: input.faceAdjacency.length };
    for (const [key, value] of Object.entries(counts) as [keyof typeof counts, number][]) {
      if (value > this.writable.capacities[key]) throw new RangeError(`Losasso ${key} upload exceeds capacity`);
    }
    if (input.rightHandSide.length !== counts.rows || input.faceMetrics.length !== 4 * counts.faces
      || input.faceAdjacencyOffsets.length !== counts.faces + 1) {
      throw new RangeError("Losasso bootstrap arrays do not match the accepted authority");
    }
    const queue = this.device.queue, target = this.writable;
    // `from` also detaches the upload ABI from SharedArrayBuffer-backed views,
    // which WebGPU intentionally rejects as mutable queue input.
    queue.writeBuffer(target.control, 0, Uint32Array.from(authority.control));
    queue.writeBuffer(target.rowFaceOffsets, 0, Uint32Array.from(authority.rowFaceOffsets));
    queue.writeBuffer(target.rowFaces, 0, Uint32Array.from(authority.rowFaces));
    queue.writeBuffer(target.faces, 0, Uint32Array.from(authority.faces));
    queue.writeBuffer(target.rightHandSide, 0, Float32Array.from(input.rightHandSide));
    queue.writeBuffer(target.faceMetrics, 0, Uint32Array.from(input.faceMetrics));
    queue.writeBuffer(target.faceAdjacencyOffsets, 0, Uint32Array.from(input.faceAdjacencyOffsets));
    queue.writeBuffer(target.faceAdjacency, 0, Uint32Array.from(input.faceAdjacency));
    queue.writeBuffer(target.rowDispatch, 0, Uint32Array.of(Math.ceil(counts.rows / 64), 1, 1));
    queue.writeBuffer(target.faceDispatch, 0, Uint32Array.of(Math.ceil(counts.faces / 64), 1, 1));
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    for (const value of [...Object.values(this.writable), ...Object.values(this.candidate)]) {
      if (typeof value === "object" && value !== null && "destroy" in value) (value as GPUBuffer).destroy();
    }
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Losasso authority is destroyed"); }
}
