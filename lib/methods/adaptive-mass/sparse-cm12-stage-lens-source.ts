/**
 * The buffers, snapshots and counters behind Sparse CM12's stage lenses.
 *
 * This is the whole cost model. Unarmed it holds three references and encodes
 * nothing: no snapshot buffers exist, no `copyBufferToBuffer` runs at any seam,
 * and no header is read back. Arming one lens allocates exactly the snapshots
 * that lens's taps declare and starts one small readback a frame. Disarming
 * frees them again. A method can therefore declare a lens beside all eighteen
 * of its stages without a scene that never opens one paying for any of them.
 *
 * ## Why a tap is a copy and not a second buffer
 *
 * The interesting quantity in a stage is almost always a difference: the face
 * velocity before the projection minus the face velocity after it. The solver
 * has no reason to keep the "before", and giving it one would change the
 * arena's size and the frame's bandwidth for every scene, armed or not. Copying
 * the range aside at the seam costs nothing when nobody is looking and is
 * *exact* when somebody is — the value as it stood at that point inside the
 * stage, not a reconstruction of it.
 */
import type {
  AnyStageLens,
  StageLensReceipt,
  StageLensSource,
} from "../../core/stage-lens";
import type { SparseCM12AddressingSpec } from "./sparse-cm12-stage-contract";
import {
  SPARSE_CM12_ADDRESSING_BYTES,
  SPARSE_CM12_ADDRESSING_WORDS,
  writeSparseCM12Addressing,
} from "./sparse-cm12-stage-contract";

/** Readback slots. Three so a frame never waits on the frame before last. */
const HEADER_READBACK_SLOTS = 3;

export interface SparseCM12StageLensSpace {
  readonly dimensions: readonly [number, number, number];
  readonly origin_m: readonly [number, number, number];
  readonly extent_m: readonly [number, number, number];
}

export interface SparseCM12StageLensSourceOptions {
  readonly device: GPUDevice;
  readonly state: GPUBuffer;
  readonly arena: GPUBuffer;
  /** Every lens the method declares, in stage order. */
  readonly lenses: readonly AnyStageLens[];
  /**
   * The current frame's layout.
   *
   * A function rather than a value because row and cell counts move with the
   * topology, and a lens that cached last re-mesh's offsets would read a
   * pressure field where the divergence now is.
   */
  readonly addressing: () => SparseCM12AddressingSpec;
  readonly space: () => SparseCM12StageLensSpace;
  /** Byte range of a lens's stage header. Only the resident knows this. */
  readonly headerRange: (lens: AnyStageLens) => GPUBufferBinding | undefined;
}

interface HeaderSlot {
  readonly buffer: GPUBuffer;
  busy: boolean;
}

export class SparseCM12StageLensSource implements StageLensSource {
  private readonly snapshots = new Map<string, GPUBuffer>();
  private readonly headerSlots: HeaderSlot[] = [];
  private readonly capturedThisFrame = new Set<string>();
  private capturedLastFrame = new Set<string>();
  private addressingBuffer?: GPUBuffer;
  private readonly addressingWords = new Uint32Array(SPARSE_CM12_ADDRESSING_WORDS);
  private readonly addressingScratch = new Uint32Array(SPARSE_CM12_ADDRESSING_WORDS);
  private addressingUploaded = false;
  private armedLens?: string;
  private lastReceipt?: StageLensReceipt;
  private frames = 0;
  private destroyed = false;

  constructor(private readonly options: SparseCM12StageLensSourceOptions) {}

  get lenses(): readonly AnyStageLens[] {
    return this.options.lenses;
  }

  get armed(): string | undefined {
    return this.armedLens;
  }

  get space(): SparseCM12StageLensSpace {
    return this.options.space();
  }

  get capacities(): { readonly rows: number; readonly cells: number; readonly bricks: number } {
    const spec = this.options.addressing();
    return { rows: spec.rowCount, cells: spec.cellCount, bricks: spec.brickCount };
  }

  arm(lensId: string | undefined): boolean {
    if (this.destroyed) return false;
    if (lensId === this.armedLens) return true;
    if (lensId !== undefined && !this.options.lenses.some((lens) => lens.id === lensId)) {
      return false;
    }
    this.release();
    this.armedLens = lensId;
    this.lastReceipt = undefined;
    return true;
  }

  /**
   * Where a publication lives right now.
   *
   * Every range is recomputed from the current addressing rather than cached:
   * the cost is a handful of object literals per frame, and the alternative is
   * a stale offset that draws a confident wrong picture after a re-mesh.
   */
  publication(key: string): GPUBufferBinding | undefined {
    if (this.destroyed) return undefined;
    const spec = this.options.addressing();
    const { state, arena } = this.options;
    const cells = 4 * Math.max(1, spec.cellFieldStride);
    const field = (base: number): GPUBufferBinding =>
      ({ buffer: state, offset: 4 * base, size: cells });
    switch (key) {
      case "arena": return { buffer: arena, offset: 0, size: arena.size };
      case "state": return { buffer: state, offset: 0, size: state.size };
      case "addressing": return this.addressing(spec);
      case "faces": return { buffer: state, offset: 4 * spec.faceBase,
        size: 4 * 2 * Math.max(1, spec.faceBankStride) };
      case "pressure": return field(spec.pressure);
      case "rhs": return field(spec.rhs);
      case "divergence": return field(spec.divergence);
      case "liquid": return field(spec.liquid);
      default: return undefined;
    }
  }

  private addressing(spec: SparseCM12AddressingSpec): GPUBufferBinding | undefined {
    if (!this.addressingBuffer) {
      this.addressingBuffer = this.options.device.createBuffer({
        label: "Sparse CM12 stage lens addressing",
        size: SPARSE_CM12_ADDRESSING_BYTES,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    // Resolution runs once per program per frame, and the offsets move only
    // when the scene's counts do, so the upload is gated on the words actually
    // changing rather than on having been asked for them.
    writeSparseCM12Addressing(this.addressingScratch, spec);
    if (!this.addressingUploaded
      || this.addressingScratch.some((word, i) => word !== this.addressingWords[i])) {
      this.addressingWords.set(this.addressingScratch);
      this.options.device.queue.writeBuffer(this.addressingBuffer, 0, this.addressingWords);
      this.addressingUploaded = true;
    }
    return { buffer: this.addressingBuffer, offset: 0, size: SPARSE_CM12_ADDRESSING_BYTES };
  }

  snapshot(lensId: string, tap: string, publication: string): GPUBufferBinding | undefined {
    if (this.armedLens !== lensId) return undefined;
    // A tap whose `capture` did not run this frame has nothing truthful to
    // show, so the group is refused and the phase paints its fault class. The
    // last frame's copy is still in the buffer, and drawing it would be the one
    // failure this whole design exists to prevent.
    if (!this.capturedLastFrame.has(`${tap}:${publication}`)) return undefined;
    const buffer = this.snapshots.get(`${lensId}#${tap}#${publication}`);
    if (!buffer) return undefined;
    return { buffer, offset: 0, size: buffer.size };
  }

  /**
   * Copy the tap's publications aside, if this lens is the armed one.
   *
   * Called from inside a stage's encode body at the seam the tap names. The
   * guard is first and cheap, because this runs on every frame of every scene
   * whether or not anybody is looking.
   */
  capture(encoder: GPUCommandEncoder, lensId: string, tap: string): void {
    if (this.destroyed || this.armedLens !== lensId) return;
    const lens = this.options.lenses.find((candidate) => candidate.id === lensId);
    const publications = lens?.taps[tap];
    if (!lens || !publications) return;
    for (const publication of publications) {
      const source = this.publication(publication);
      if (!source) continue;
      const size = source.size ?? 0;
      if (size === 0) continue;
      const key = `${lensId}#${tap}#${publication}`;
      let target = this.snapshots.get(key);
      if (!target || target.size < size) {
        target?.destroy();
        target = this.options.device.createBuffer({
          label: `Sparse CM12 lens snapshot ${tap}/${publication}`,
          size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this.snapshots.set(key, target);
      }
      encoder.copyBufferToBuffer(source.buffer, source.offset ?? 0, target, 0, size);
      this.capturedThisFrame.add(`${tap}:${publication}`);
    }
  }

  /**
   * Close the frame: promote this frame's captures and queue the header read.
   *
   * Called once from the tail of the advance encode. The promotion is what
   * makes a missed tap visible — a tap in a branch the frame did not take
   * simply is not in the new set, and its phases go magenta next frame rather
   * than quietly redrawing the last frame that did take it.
   */
  endFrame(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.armedLens) return;
    this.frames += 1;
    this.capturedLastFrame = new Set(this.capturedThisFrame);
    this.capturedThisFrame.clear();
    const lens = this.options.lenses.find((candidate) => candidate.id === this.armedLens);
    if (!lens) return;
    const range = this.options.headerRange(lens);
    if (!range) return;
    const bytes = 4 * lens.header.wordCount;
    const slot = this.headerSlot(bytes);
    if (!slot) return;
    slot.busy = true;
    encoder.copyBufferToBuffer(range.buffer, range.offset ?? 0, slot.buffer, 0,
      Math.min(bytes, slot.buffer.size));
    // The caller submits this encoder synchronously after `encode` returns, so
    // a microtask is the first moment the copy is on the queue. Waiting on the
    // queue rather than on a timer means the map cannot race the submit.
    queueMicrotask(() => void this.drain(lens, slot));
  }

  private headerSlot(bytes: number): HeaderSlot | undefined {
    for (const slot of this.headerSlots) {
      if (!slot.busy && slot.buffer.size >= bytes) return slot;
    }
    if (this.headerSlots.length >= HEADER_READBACK_SLOTS) return undefined;
    const slot: HeaderSlot = {
      buffer: this.options.device.createBuffer({
        label: "Sparse CM12 stage lens header readback",
        size: Math.max(64, bytes),
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      busy: false,
    };
    this.headerSlots.push(slot);
    return slot;
  }

  private async drain(lens: AnyStageLens, slot: HeaderSlot): Promise<void> {
    try {
      await this.options.device.queue.onSubmittedWorkDone();
      if (this.destroyed) return;
      await slot.buffer.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(slot.buffer.getMappedRange()).slice();
      this.lastReceipt = decodeStageLensHeader(lens, words,
        this.capturedLastFrame.size, this.frames);
    } catch {
      // A device loss or a lens disarmed mid-flight. The panel keeps the last
      // receipt it had, which the step number in it dates.
    } finally {
      if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
      slot.busy = false;
    }
  }

  receipt(lensId: string): StageLensReceipt | undefined {
    return this.lastReceipt?.lensId === lensId ? this.lastReceipt : undefined;
  }

  private release(): void {
    for (const buffer of this.snapshots.values()) buffer.destroy();
    this.snapshots.clear();
    this.capturedThisFrame.clear();
    this.capturedLastFrame = new Set();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.armedLens = undefined;
    this.release();
    this.addressingBuffer?.destroy();
    this.addressingBuffer = undefined;
    for (const slot of this.headerSlots) {
      if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
      slot.buffer.destroy();
    }
    this.headerSlots.length = 0;
  }
}

/**
 * Turn a stage header's words into a receipt, using the lens's own table.
 *
 * The magic check is the runtime backstop types cannot reach: a header whose
 * base offset moved reads as plausible integers, and only the stamp says it is
 * the wrong region. `headerValid` false means every counter is untrustworthy,
 * which the panel draws rather than hides.
 */
export function decodeStageLensHeader(
  lens: AnyStageLens,
  words: Uint32Array,
  capturedTaps: number,
  step: number,
): StageLensReceipt {
  const magic = lens.header.magic;
  const headerValid = !magic || words[magic.word] === magic.value;
  const counters: Record<string, number> = {};
  for (const [name, offset] of Object.entries(lens.header.words)) {
    counters[name] = words[offset] ?? 0;
  }
  return Object.freeze({
    lensId: lens.id,
    armed: true,
    expectedTaps: Object.values(lens.taps)
      .reduce((total, publications) => total + publications.length, 0),
    capturedTaps,
    headerValid,
    counters: Object.freeze(counters),
    step,
  });
}
