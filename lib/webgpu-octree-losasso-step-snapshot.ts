/** End-of-step receipt for the production Losasso authority chain. */
export interface LosassoStepSnapshotSources {
  readonly authority: GPUBuffer;
  readonly solver: GPUBuffer;
  readonly fineWorklist?: GPUBuffer;
  readonly coarsePhi: GPUBuffer;
  readonly extension: GPUBuffer;
  readonly fineTransport?: GPUBuffer;
}

const LAYOUT = Object.freeze({
  authority: 0, solver: 32, fine: 96, coarsePhi: 124, extension: 184,
  fineTransport: 216, stride: 248,
});
const COPY_DST = 0x0008, MAP_READ = 0x0001, READ = 0x0001;

export interface LosassoStepSnapshotRecord {
  readonly step: number;
  readonly surfaceKind: "fine" | "coarse";
  readonly authority: Uint32Array;
  readonly solver: Uint32Array;
  readonly fine: Uint32Array;
  readonly coarsePhi: Uint32Array;
  readonly extension: Uint32Array;
  readonly fineTransport: Uint32Array;
}

export function losassoStepSnapshotFailures(
  record: LosassoStepSnapshotRecord,
): readonly string[] {
  const { authority, solver, fine, coarsePhi, extension, fineTransport, surfaceKind } = record;
  const failures: string[] = [];
  const epoch = authority[0] ?? 0, rows = authority[1] ?? 0, faces = authority[2] ?? 0;
  const generation = surfaceKind === "coarse" ? (coarsePhi[12] ?? 0) : (fine[0] ?? 0);
  if (epoch === 0 || rows === 0 || faces === 0 || authority[3] !== 1 || authority[4] !== 0) {
    failures.push("reduced authority is not valid and non-empty");
  }
  if (solver[0] !== 0 || solver[1] === 0 || solver[4] !== rows) {
    failures.push("MGPCG receipt is invalid, unconverged, or row-incoherent");
  }
  if (surfaceKind === "fine" && (generation === 0 || generation === 0xffff_ffff || fine[1] === 0
    || fine[1]! > fine[2]! || (fine[3]! & 3) !== 3 || fine[5] !== 1 || fine[6] !== 1)) {
    failures.push("fine-phi publication header is invalid");
  }
  if (coarsePhi[0] !== 0x4c50_4849 || coarsePhi[1] !== epoch
    || coarsePhi[2] !== rows || coarsePhi[3] !== faces
    || coarsePhi[12] !== generation || coarsePhi[13] !== 0 || coarsePhi[14] !== generation) {
    failures.push("coarse-phi receipt disagrees with authority or fine generation");
  }
  if (extension[0] !== epoch || extension[2] === 0 || extension[3] !== 1
    || extension[4] !== 0 || extension[5] !== generation) {
    failures.push("W7 extension receipt is invalid or generation-incoherent");
  }
  if (surfaceKind === "fine" && (fineTransport[1] !== 0 || fineTransport[3] !== 1
    || fineTransport[6] !== 0 || fineTransport[7] !== 0)) {
    failures.push("fine transport did not commit cleanly");
  }
  return Object.freeze(failures);
}

interface Slot {
  readonly buffer: GPUBuffer;
  state: "free" | "encoded" | "mapping";
  sequence: number;
  step: number;
  surfaceKind: "fine" | "coarse";
}

export class WebGPUOctreeLosassoStepSnapshotRing {
  private readonly slots: Slot[];
  private sequence = 0;
  private disposed = false;
  private skipped = 0;

  constructor(private readonly device: GPUDevice, slotCount: number) {
    this.slots = Array.from({ length: Math.max(2, slotCount) }, (_, index) => ({
      buffer: device.createBuffer({ label: `Losasso step snapshot slot ${index}`,
        size: LAYOUT.stride, usage: COPY_DST | MAP_READ }),
      state: "free", sequence: 0, step: 0, surfaceKind: "fine",
    }));
  }

  get skippedRecords() { return this.skipped; }
  get slotCount() { return this.slots.length; }

  encode(encoder: GPUCommandEncoder, sources: LosassoStepSnapshotSources, step: number,
    surfaceKind: "fine" | "coarse" = "fine"): boolean {
    if (this.disposed) return false;
    let slot: Slot | undefined;
    for (const candidate of this.slots) {
      if (candidate.state === "mapping") continue;
      if (!slot || candidate.sequence < slot.sequence) slot = candidate;
    }
    if (!slot) { this.skipped += 1; return false; }
    const copy = (source: GPUBuffer, targetOffset: number, bytes: number) =>
      encoder.copyBufferToBuffer(source, 0, slot!.buffer, targetOffset,
        Math.min(bytes, Math.floor(source.size / 4) * 4));
    copy(sources.authority, LAYOUT.authority, 32);
    copy(sources.solver, LAYOUT.solver, 64);
    if (sources.fineWorklist) copy(sources.fineWorklist, LAYOUT.fine, 28);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fine, 28);
    copy(sources.coarsePhi, LAYOUT.coarsePhi, 60);
    copy(sources.extension, LAYOUT.extension, 32);
    if (sources.fineTransport) copy(sources.fineTransport, LAYOUT.fineTransport, 32);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fineTransport, 32);
    slot.sequence = ++this.sequence; slot.step = step; slot.surfaceKind = surfaceKind;
    slot.state = "encoded";
    return true;
  }

  async readLatest(): Promise<LosassoStepSnapshotRecord | undefined> {
    if (this.disposed) return undefined;
    let slot: Slot | undefined;
    for (const candidate of this.slots) {
      if (candidate.state !== "encoded") continue;
      if (!slot || candidate.sequence > slot.sequence) slot = candidate;
    }
    if (!slot) return undefined;
    slot.state = "mapping";
    try {
      await slot.buffer.mapAsync(READ);
      const mapped = slot.buffer.getMappedRange();
      const words = (offset: number, count: number) =>
        Uint32Array.from(new Uint32Array(mapped, offset, count));
      return Object.freeze({ step: slot.step, surfaceKind: slot.surfaceKind,
        authority: words(LAYOUT.authority, 8), solver: words(LAYOUT.solver, 16),
        fine: words(LAYOUT.fine, 7), coarsePhi: words(LAYOUT.coarsePhi, 15),
        extension: words(LAYOUT.extension, 8), fineTransport: words(LAYOUT.fineTransport, 8) });
    } catch {
      return undefined;
    } finally {
      if (slot.buffer.mapState === "mapped") slot.buffer.unmap();
      slot.state = this.disposed ? "mapping" : "free";
      if (this.disposed) slot.buffer.destroy();
    }
  }

  destroy(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const slot of this.slots) if (slot.state !== "mapping") slot.buffer.destroy();
  }
}
