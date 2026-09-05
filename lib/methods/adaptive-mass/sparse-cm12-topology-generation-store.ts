import type { SparseCM12TopologyPreparation } from "./webgpu-sparse-cm12-resident";
import { SPARSE_CM12_PACKED_TEMPLATE_MAGIC } from "./sparse-cm12-factored-aei-packed-template";

type Packet = Extract<SparseCM12TopologyPreparation, { status: "ready" }>;

/** Topology storage only. The resident transaction must also prepare field,
 * pressure, transport and presentation consumers before committing this store. */
export interface SparseCM12TopologyGenerationBuffers {
  readonly generation: number;
  readonly topology: GPUBuffer;
  /** [generation, cell count, row count, row-list offset], then cell and row IDs. */
  readonly membership: GPUBuffer;
  readonly bytes: number;
}
interface Image {
  readonly buffers: SparseCM12TopologyGenerationBuffers;
  leases: number;
  retired: boolean;
  destroyed: boolean;
}
export interface SparseCM12TopologyGenerationLease {
  readonly buffers: SparseCM12TopologyGenerationBuffers;
  /** Call after submitting every command that uses this lease. Reclamation waits
   * for queue completion. An encoded but unsubmitted command still needs its lease. */
  releaseAfterSubmission(): Promise<void>;
}
export interface SparseCM12StagedTopologyGeneration {
  readonly buffers: SparseCM12TopologyGenerationBuffers;
}
export type SparseCM12TopologyGenerationReservation =
  | { readonly status: "deferred"; readonly requiredBytes: number; readonly availableBytes: number }
  | { readonly status: "ready"; readonly candidate: SparseCM12StagedTopologyGeneration };

function membership(packet: Packet, candidate: boolean): Uint32Array {
  const generation = candidate ? packet.candidateGeneration : packet.acceptedGeneration;
  const cells = candidate ? packet.candidateCellWorklist : packet.acceptedCellWorklist;
  const rows = candidate ? packet.candidateRowWorklist : packet.acceptedRowWorklist;
  if (!Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff
    || packet.words[0] !== SPARSE_CM12_PACKED_TEMPLATE_MAGIC || packet.words[2] !== packet.cellCount
    || packet.words[3] !== packet.rowCount) {
    throw new Error("Sparse CM12 generation packet header is invalid");
  }
  const selected = new Set<number>();
  for (const cell of cells) {
    if (cell >= packet.cellCount || selected.has(cell)) {
      throw new Error("Sparse CM12 generation has invalid or duplicate cell membership");
    }
    selected.add(cell);
  }
  const selectedRows = new Set<number>();
  for (const row of rows) {
    if (row >= packet.rowCount || selectedRows.has(row)) {
      throw new Error("Sparse CM12 generation has invalid or duplicate row membership");
    }
    selectedRows.add(row);
    const bits = packet.words[packet.words[7]! + row];
    if (bits === undefined) throw new Error("Sparse CM12 generation row is missing");
    const first = bits & 0x007f_ffff, count = bits >>> 23;
    if (count === 0 || first + count > packet.words[4]!) {
      throw new Error("Sparse CM12 generation row terms are invalid");
    }
    for (let term = first; term < first + count; term++) {
      if (!selected.has(packet.words[packet.words[8]! + 2 * term]!)) {
        throw new Error("Sparse CM12 generation row references an unselected cell");
      }
    }
  }
  const result = new Uint32Array(4 + cells.length + rows.length);
  result.set([generation, cells.length, rows.length, 4 + cells.length]);
  result.set(cells, 4); result.set(rows, 4 + cells.length);
  return result;
}

/** Owns bounded, immutable GPU topology generations. Accepted, staged and leased
 * retired images all count against the same byte budget. There is no scene policy
 * here and no implicit simulation publication or field transfer. */
export class SparseCM12TopologyGenerationStore {
  private readonly images = new Set<Image>();
  private accepted!: Image;
  private pending?: Image;
  private preparing = false;
  private disposed = false;
  private reserved = 0;
  private peakReserved = 0;
  private deferredRequests = 0;
  private allocationFailures = 0;

  private constructor(private readonly device: GPUDevice, readonly maximumBytes: number) {
    if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 0) {
      throw new RangeError("Sparse CM12 generation storage budget must be a nonnegative safe integer");
    }
  }

  static async create(device: GPUDevice, packet: Packet, maximumBytes: number) {
    const store = new SparseCM12TopologyGenerationStore(device, maximumBytes);
    const words = membership(packet, false);
    const image = await store.allocate(packet.words, words, packet.acceptedGeneration);
    if (!image) throw new RangeError("Sparse CM12 accepted generation exceeds storage budget");
    store.accepted = image;
    return store;
  }

  get receipt() {
    const acceptedBytes = this.accepted.destroyed ? 0 : this.accepted.buffers.bytes;
    const stagedBytes = this.pending?.buffers.bytes ?? 0;
    const retiredBytes = [...this.images].filter((image) => image.retired)
      .reduce((sum, image) => sum + image.buffers.bytes, 0);
    // A destroyed store's leased accepted image belongs to the retired total.
    const activeBytes = this.accepted.retired ? 0 : acceptedBytes;
    return Object.freeze({ generation: this.accepted.buffers.generation,
      maximumBytes: this.maximumBytes, reservedBytes: this.reserved,
      peakReservedBytes: this.peakReserved,
      deferredRequests: this.deferredRequests, allocationFailures: this.allocationFailures,
      acceptedBytes: activeBytes, stagedBytes, retiredBytes,
      preparingBytes: this.reserved - activeBytes - stagedBytes - retiredBytes });
  }

  private assertLive() {
    if (this.disposed) throw new Error("Sparse CM12 generation store is destroyed");
  }

  private async allocate(topologyWords: Uint32Array, memberWords: Uint32Array,
    generation: number): Promise<Image | undefined> {
    if ([topologyWords, memberWords].some((words) =>
      words.byteLength > this.device.limits.maxStorageBufferBindingSize)) {
      throw new RangeError("Sparse CM12 generation exceeds the device storage binding limit");
    }
    const bytes = topologyWords.byteLength + memberWords.byteLength;
    if (bytes > this.maximumBytes - this.reserved) return undefined;
    this.reserved += bytes;
    this.peakReserved = Math.max(this.peakReserved, this.reserved);
    const buffers: GPUBuffer[] = [];
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("validation");
    let thrown: unknown;
    try {
      for (const [label, words] of [["topology", topologyWords],
        ["membership", memberWords]] as const) {
        const buffer = this.device.createBuffer({
          label: `Sparse CM12 generation ${generation} ${label}`,
          size: words.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        buffers.push(buffer);
        this.device.queue.writeBuffer(buffer, 0, words.buffer as ArrayBuffer,
          words.byteOffset, words.byteLength);
      }
    } catch (error) { thrown = error; }
    // Pop synchronously before awaiting: another device user must not inherit
    // our scopes while its own work is being encoded.
    const validation = this.device.popErrorScope(), memory = this.device.popErrorScope();
    try {
      const errors = await Promise.all([validation, memory]);
      if (thrown) throw thrown;
      const error = errors.find((value) => value !== null);
      if (error) throw new Error(`Sparse CM12 generation allocation failed: ${error.message}`);
      this.assertLive();
      const image: Image = { buffers: Object.freeze({ generation, bytes,
        topology: buffers[0]!, membership: buffers[1]! }),
        leases: 0, retired: false, destroyed: false };
      this.images.add(image);
      return image;
    } catch (error) {
      for (const buffer of buffers) buffer.destroy();
      this.reserved -= bytes;
      if (!this.disposed) this.allocationFailures++;
      throw error;
    }
  }

  async prepare(packet: Packet): Promise<SparseCM12TopologyGenerationReservation> {
    this.assertLive();
    if (this.preparing || this.pending) throw new Error("Sparse CM12 already has a pending generation");
    const generation = this.accepted.buffers.generation;
    if (packet.acceptedGeneration !== generation || packet.candidateGeneration !== generation + 1) {
      throw new Error("Sparse CM12 candidate generation is stale or discontinuous");
    }
    const words = membership(packet, true);
    const requiredBytes = packet.words.byteLength + words.byteLength;
    const availableBytes = this.maximumBytes - this.reserved;
    this.preparing = true;
    try {
      const image = await this.allocate(packet.words, words, packet.candidateGeneration);
      if (!image) {
        this.deferredRequests++;
        return { status: "deferred", requiredBytes, availableBytes };
      }
      this.pending = image;
      return { status: "ready", candidate: image };
    } finally { this.preparing = false; }
  }

  private requirePending(candidate: SparseCM12StagedTopologyGeneration): Image {
    this.assertLive();
    if (candidate !== this.pending) throw new Error("Sparse CM12 candidate is not owned by this transaction");
    return this.pending;
  }

  cancel(candidate: SparseCM12StagedTopologyGeneration): void {
    const image = this.requirePending(candidate);
    this.pending = undefined;
    image.retired = true;
    this.reclaim(image);
  }

  /** Storage publication only. The caller must have validated all dependent
   * physics consumers and switch their bindings in the same resident transaction. */
  commit(candidate: SparseCM12StagedTopologyGeneration): void {
    const image = this.requirePending(candidate);
    const previous = this.accepted;
    this.accepted = image;
    this.pending = undefined;
    previous.retired = true;
    this.reclaim(previous);
  }

  /** Lease either accepted storage or a pending image for validation/transfer.
   * Holding the lease protects commands that have been encoded but not submitted. */
  acquire(candidate?: SparseCM12StagedTopologyGeneration): SparseCM12TopologyGenerationLease {
    this.assertLive();
    const image = candidate ? this.requirePending(candidate) : this.accepted;
    image.leases++;
    let completion: Promise<void> | undefined;
    return Object.freeze({ buffers: image.buffers,
      releaseAfterSubmission: () => completion ??= this.device.queue.onSubmittedWorkDone()
        .finally(() => { image.leases--; this.reclaim(image); }) });
  }

  private reclaim(image: Image) {
    if (!image.retired || image.leases !== 0 || image.destroyed) return;
    image.destroyed = true;
    image.buffers.topology.destroy(); image.buffers.membership.destroy();
    this.reserved -= image.buffers.bytes;
    this.images.delete(image);
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.pending = undefined;
    for (const image of this.images) { image.retired = true; this.reclaim(image); }
  }
}
