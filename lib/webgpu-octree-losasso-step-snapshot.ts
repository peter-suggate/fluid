import { OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS }
  from "./webgpu-octree-losasso-adaptive-phi";

/** End-of-step receipt for the production Losasso authority chain. */
export interface LosassoStepSnapshotSources {
  readonly authority: GPUBuffer;
  readonly solver: GPUBuffer;
  readonly fineWorklist?: GPUBuffer;
  readonly coarsePhi: GPUBuffer;
  readonly extension: GPUBuffer;
  readonly fineTransport?: GPUBuffer;
  /** Ando--Batty surface transaction paired with `fineWorklist`. A Losasso
   * fine publication is not accepted from the worklist header alone: its
   * remesh, redistance, and non-mutating volume measurement must all describe
   * that same generation and none may have retained an older publication. */
  readonly fineTopology?: GPUBuffer;
  readonly fineRedistance?: GPUBuffer;
  readonly fineVolume?: GPUBuffer;
  readonly fluidResidency?: GPUBuffer;
  readonly fluidBulkResidency?: GPUBuffer;
  /** Factor-one compact authority. These sources are copied together at the
   * step tail; none is interpreted as the retired dense/coarse-phi ABI. */
  readonly adaptive?: {
    readonly candidateAuthority: GPUBuffer;
    readonly acceptedGraph: GPUBuffer;
    readonly candidateGraph: GPUBuffer;
    readonly phiControl: GPUBuffer;
    readonly phiReceipts: GPUBuffer;
    readonly velocityReceipts: GPUBuffer;
    readonly rendererDirectory: GPUBuffer;
    readonly massControl: GPUBuffer;
    readonly massReceipts: GPUBuffer;
    /** Candidate wet-face migration receipt:
     * generation, faces, migrated, nodal, errors, valid-generation. */
    readonly velocityMigration: GPUBuffer;
  };
}

const ADAPTIVE_PHI_RECEIPT_BYTES = 4 * OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS;
const LAYOUT = Object.freeze({
  authority: 0, solver: 32, fine: 96, coarsePhi: 124, extension: 184,
  fineTransport: 216, fluidResidency: 248, fluidBulkResidency: 312,
  fineTopology: 376, fineRedistance: 440, fineVolume: 504,
  adaptiveAcceptedGraph: 568, adaptiveCandidateGraph: 696,
  adaptivePhiControl: 824, adaptivePhiReceipts: 904,
  adaptiveVelocityReceipts: 904 + ADAPTIVE_PHI_RECEIPT_BYTES,
  adaptiveRenderer: 1112 + ADAPTIVE_PHI_RECEIPT_BYTES,
  adaptiveCandidateAuthority: 1144 + ADAPTIVE_PHI_RECEIPT_BYTES,
  adaptiveMassControl: 1176 + ADAPTIVE_PHI_RECEIPT_BYTES,
  adaptiveMassReceipts: 1304 + ADAPTIVE_PHI_RECEIPT_BYTES,
  adaptiveVelocityMigration: 1432 + ADAPTIVE_PHI_RECEIPT_BYTES,
  stride: 1464 + ADAPTIVE_PHI_RECEIPT_BYTES,
});
const COPY_DST = 0x0008, MAP_READ = 0x0001, READ = 0x0001;

export interface LosassoStepSnapshotRecord {
  readonly step: number;
  readonly surfaceKind: "fine" | "coarse" | "adaptive";
  readonly authority: Uint32Array;
  readonly solver: Uint32Array;
  readonly fine: Uint32Array;
  readonly coarsePhi: Uint32Array;
  readonly extension: Uint32Array;
  readonly fineTransport: Uint32Array;
  readonly fineTopology: Uint32Array;
  readonly fineRedistance: Uint32Array;
  readonly fineVolume: Uint32Array;
  readonly fluidResidency: Uint32Array;
  readonly fluidBulkResidency: Uint32Array;
  readonly adaptive?: {
    readonly candidateAuthority: Uint32Array;
    readonly acceptedGraph: Uint32Array;
    readonly candidateGraph: Uint32Array;
    readonly phiControl: Uint32Array;
    readonly phiReceipts: Uint32Array;
    readonly velocityReceipts: Uint32Array;
    readonly renderer: Uint32Array;
    readonly massControl: Uint32Array;
    readonly massReceipts: Uint32Array;
    readonly velocityMigration: Uint32Array;
  };
}

const ADAPTIVE_PHI_MAGIC = 0x4150_4849;
const ADAPTIVE_TOPOLOGY_MAGIC = 0x4c50_4849;

export function losassoStepSnapshotFailures(
  record: LosassoStepSnapshotRecord,
): readonly string[] {
  const { authority, solver, fine, coarsePhi, extension, fineTransport,
    fineTopology, fineRedistance, fineVolume, surfaceKind } = record;
  const failures: string[] = [];
  const epoch = authority[0] ?? 0, rows = authority[1] ?? 0, faces = authority[2] ?? 0;
  const generation = surfaceKind === "coarse" ? (coarsePhi[12] ?? 0)
    : surfaceKind === "adaptive" ? (record.adaptive?.acceptedGraph[5] ?? 0)
    : (fine[0] ?? 0);
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
  if (surfaceKind === "fine") {
    const topologyStickyReasons = fineTopology[10] ?? 0;
    const topologyRejectCount = (fineTopology[11] ?? 0) & 0xffff;
    if (fineTopology.length < 16 || fineTopology[0] !== 0
      || fineTopology[3] !== fine[1] || fineTopology[4] !== 1
      || fineTopology[5] !== 0 || fineTopology[7] !== 0
      || topologyStickyReasons !== 0 || topologyRejectCount !== 0
      || fineTopology[15] !== 0) {
      failures.push("fine topology remesh rejected, retained, or disagrees with the published surface");
    }
    if (fineRedistance.length < 16 || fineRedistance[0] !== 0
      || fineRedistance[3] !== 1 || fineRedistance[4] !== 0
      || fineRedistance[9] !== 0) {
      failures.push("fine signed-distance publication is unresolved, stale, or uncommitted");
    }
    // The Losasso/Ando lane only measures volume; it never shifts phi to hit
    // a target. These words therefore attest owner coverage and generation,
    // not permission for a global repair.
    if (fineVolume.length < 16 || fineVolume[0] !== 0x8000_0000
      || fineVolume[1] !== 1 || fineVolume[2] === 0
      || fineVolume[13] !== generation || fineVolume[14] !== 0
      || fineVolume[15] !== 0) {
      failures.push("fine surface-volume measurement is invalid or owner-incoherent");
    }
  }
  if (surfaceKind === "adaptive") {
    const adaptive = record.adaptive;
    const graph = adaptive?.acceptedGraph ?? new Uint32Array();
    const candidateAuthority = adaptive?.candidateAuthority ?? new Uint32Array();
    const phi = adaptive?.phiControl ?? new Uint32Array();
    const phiReceipts = adaptive?.phiReceipts ?? new Uint32Array();
    const velocity = adaptive?.velocityReceipts ?? new Uint32Array();
    const renderer = adaptive?.renderer ?? new Uint32Array();
    if (!adaptive) {
      failures.push("adaptive graph/phi/velocity receipt tuple is absent");
    } else {
      const mass = adaptive.massControl;
      const massReceipts = adaptive.massReceipts;
      const migration = adaptive.velocityMigration;
      // Epoch zero is the publisher's explicit dormant-candidate state. The
      // candidate graph, mass, and migration buffers are scratch receipts and
      // may still contain rejection sentinels from the skipped transaction;
      // they have no authority until a nonzero candidate epoch is published.
      const candidateEpoch = candidateAuthority[0] ?? 0;
      const hardCandidateFailure = candidateEpoch !== 0
        && ((candidateAuthority[4] ?? 0) & 0x8000_0000) !== 0;
      if (hardCandidateFailure) {
        const massErrors = mass[12] ?? 0;
        const signMismatches = massReceipts[16] ?? 0;
        const firstItem = massReceipts[17] ?? 0xffff_ffff;
        const migrated = migration[2] ?? 0, nodal = migration[3] ?? 0;
        failures.push(`fatal adaptive candidate transaction: authority=${candidateAuthority[4] ?? 0}, mass=${massErrors}, reconstructionSignMismatches=${signMismatches}, firstItem=${firstItem}, velocityMigration=${migration[0] ?? 0}/${migration[1] ?? 0}/${migrated}+${nodal}/${migration[4] ?? 0}/${migration[5] ?? 0}/${migration[6] ?? 0}/${migration[7] ?? 0}`);
      }
      if (candidateEpoch !== 0 && (mass[0] !== 0x414d_4153
        || mass[1] !== candidateEpoch || mass[7] !== 1 || mass[12] !== 0
        || massReceipts[12] !== 0)) {
        failures.push("adaptive conserved-mass receipt is invalid");
      }
      const scheduledPhiNodes = (phiReceipts[32] ?? 0) + (phiReceipts[34] ?? 0);
      const retainedPhiNodes = (phiReceipts[33] ?? 0) + (phiReceipts[35] ?? 0);
      const identityAdvance = phiReceipts[1] === 0 && phiReceipts[12] === 0;
      if (graph[0] === 0 || graph[3] !== graph[0] || graph[4] !== 0
        || graph[1] === 0 || graph[2] === 0 || graph[1]! <= graph[28]!
        || graph[29] !== 0 || graph[5] === 0 || graph[6] !== graph[5]) {
        failures.push("accepted adaptive graph is invalid or field-generation-incoherent");
      }
      if (phi[0] !== ADAPTIVE_PHI_MAGIC || phi[1] !== graph[0] || phi[2] !== graph[5]
        || phi[3] !== graph[6] || phi[4] !== graph[2] || phi[5] !== graph[1]
        || phi[7] !== 1 || phi[12] !== 0) {
        failures.push("accepted adaptive phi state disagrees with the accepted graph");
      }
      if (phiReceipts[2] !== 0 || phiReceipts[3] !== 0
        || scheduledPhiNodes + retainedPhiNodes !== graph[2]
        || phiReceipts[4] !== scheduledPhiNodes || phiReceipts[5] !== 0
        || phiReceipts[7] !== (identityAdvance ? 0 : scheduledPhiNodes)
        || phiReceipts[14] !== 1
        || phiReceipts[15] !== graph[1] || phiReceipts[22] !== 1
        || phiReceipts[23] !== graph[5]) {
        failures.push("accepted adaptive phi advance/redistance/renderer receipt is incomplete or stale");
      }
      if (coarsePhi[0] !== ADAPTIVE_TOPOLOGY_MAGIC || coarsePhi[1] !== graph[0]
        || coarsePhi[2] !== graph[1] || coarsePhi[3] !== 0
        || coarsePhi[12] !== graph[5] || coarsePhi[13] !== 0
        || coarsePhi[14] !== graph[5]) {
        failures.push("adaptive topology evidence disagrees with graph/phi clocks");
      }
      for (const base of [0, 12]) {
        if (velocity[base] !== graph[5] || velocity[base + 2] !== 0
          || velocity[base + 6] === 0 || velocity[base + 7] !== 1) {
          failures.push(`${base === 0 ? "accepted" : "predictor"} adaptive velocity receipt is invalid`);
        }
      }
      if (renderer[0] !== 0x8000_0000 || renderer[1] !== graph[5]
        || renderer[2] !== graph[1]) {
        failures.push("renderer publication disagrees with accepted graph/phi clocks");
      }
    }
  } else {
    if (coarsePhi[0] !== ADAPTIVE_TOPOLOGY_MAGIC || coarsePhi[1] !== epoch
      || coarsePhi[2] !== rows || coarsePhi[3] !== faces
      || coarsePhi[12] !== generation || coarsePhi[13] !== 0 || coarsePhi[14] !== generation) {
      failures.push("coarse-phi receipt disagrees with authority or fine generation");
    }
    if (extension[0] !== epoch || extension[2] === 0 || extension[3] !== 1
      || extension[4] !== 0 || extension[5] !== generation) {
      failures.push("W7 extension receipt is invalid or generation-incoherent");
    }
  }
  // Generation three is the cold sparse publication (A=1, prepared B=2,
  // first accepted target=3). It is seeded analytically and therefore has no
  // preceding field to transport. Every recurring generation must carry a
  // clean committed transport receipt; this is a lifecycle distinction, not
  // permission to retain or repair a failed update.
  const coldFineBootstrap = surfaceKind === "fine" && generation === 3
    && fineTransport[3] === 0;
  if (surfaceKind === "fine" && !coldFineBootstrap
    && (fineTransport[1] !== 0 || fineTransport[3] !== 1
      || fineTransport[6] !== 0 || fineTransport[7] !== 0)) {
    failures.push("fine transport did not commit cleanly");
  }
  return Object.freeze(failures);
}

/** Compact one-line forensic summary suitable for the runtime diagnostics UI. */
export function losassoStepSnapshotDiagnosticSummary(
  record: LosassoStepSnapshotRecord,
  failures = losassoStepSnapshotFailures(record),
): string {
  const authority = record.authority, solver = record.solver;
  if (record.surfaceKind !== "adaptive" || !record.adaptive) {
    return `${failures.join("; ")} · pressure(epoch=${authority[0] ?? 0},rows=${authority[1] ?? 0},`
      + `faces=${authority[2] ?? 0},valid=${authority[3] ?? 0},errors=${authority[4] ?? 0}) `
      + `MGPCG(flags=${solver[0] ?? 0},converged=${solver[1] ?? 0},rows=${solver[4] ?? 0})`;
  }
  const { candidateAuthority, acceptedGraph: accepted, candidateGraph: candidate,
    phiControl: phi, phiReceipts, velocityReceipts: velocity, renderer,
    massControl: mass, massReceipts, velocityMigration: migration } = record.adaptive;
  const velocityReceipt = (base: number) =>
    `${velocity[base] ?? 0}/${velocity[base + 2] ?? 0}/${velocity[base + 7] ?? 0}`;
  const candidateEpoch = candidateAuthority[0] ?? 0;
  const candidateSummary = candidateEpoch === 0 ? "candidate=none"
    : `candidate=${candidate[0] ?? 0}/${candidate[3] ?? 0}/${candidate[4] ?? 0}`;
  const candidateTransactionSummary = candidateEpoch === 0 ? "candidateTransaction=none"
    : `candidateAuthority=${candidateEpoch}/${candidateAuthority[3] ?? 0}/${candidateAuthority[4] ?? 0}`
      + ` mass=${mass[1] ?? 0}/${mass[7] ?? 0}/${mass[12] ?? 0}`
      + ` reconstruction=${massReceipts[16] ?? 0}/${massReceipts[17] ?? 0xffff_ffff}`
      + ` migration=${migration[0] ?? 0}/${migration[1] ?? 0}/`
      + `${migration[2] ?? 0}+${migration[3] ?? 0}/${migration[4] ?? 0}/${migration[5] ?? 0}`
      + `/${migration[6] ?? 0}/${migration[7] ?? 0}`;
  return `${failures.join("; ")} · pressure(epoch=${authority[0] ?? 0},rows=${authority[1] ?? 0},`
    + `faces=${authority[2] ?? 0},valid=${authority[3] ?? 0},errors=${authority[4] ?? 0}) `
    + `MGPCG(flags=${solver[0] ?? 0},converged=${solver[1] ?? 0},rows=${solver[4] ?? 0}) · `
    + `graph accepted=${accepted[0] ?? 0}/${accepted[3] ?? 0}/${accepted[4] ?? 0}`
    + ` phi=${accepted[5] ?? 0} velocity=${accepted[6] ?? 0}; `
    + `${candidateSummary} · `
    + `phi=${phi[1] ?? 0}/${phi[2] ?? 0}/${phi[3] ?? 0}/${phi[7] ?? 0}/${phi[12] ?? 0}`
    + ` receipts(advance=${phiReceipts[22] ?? 0},transport=${phiReceipts[4] ?? 0},`
    + `redistance=${phiReceipts[7] ?? 0}/${phiReceipts[14] ?? 0},`
    + `band=${phiReceipts[32] ?? 0}+${phiReceipts[33] ?? 0}`
    + `+${phiReceipts[34] ?? 0}+${phiReceipts[35] ?? 0},`
    + `velocityMiss=${phiReceipts[2] ?? 0}/${phiReceipts[3] ?? 0},`
    + `phiMiss=${phiReceipts[5] ?? 0},commit=${phiReceipts[20] ?? 0},`
    + `repair=${phiReceipts[21] ?? 0}) · `
    + `velocity accepted=${velocityReceipt(0)} predictor=${velocityReceipt(12)}`
    + ` candidate=${velocityReceipt(24)} · renderer=${renderer[1] ?? 0}/${renderer[2] ?? 0}`
    + ` · ${candidateTransactionSummary}`;
}

interface Slot {
  readonly buffer: GPUBuffer;
  state: "free" | "encoded" | "mapping";
  sequence: number;
  step: number;
  surfaceKind: "fine" | "coarse" | "adaptive";
}

export class WebGPUOctreeLosassoStepSnapshotRing {
  private readonly slots: Slot[];
  private sequence = 0;
  private disposed = false;
  private skipped = 0;
  private due = true;

  constructor(private readonly device: GPUDevice, slotCount: number) {
    this.slots = Array.from({ length: Math.max(2, slotCount) }, (_, index) => ({
      buffer: device.createBuffer({ label: `Losasso step snapshot slot ${index}`,
        size: LAYOUT.stride, usage: COPY_DST | MAP_READ }),
      state: "free", sequence: 0, step: 0, surfaceKind: "fine",
    }));
  }

  get skippedRecords() { return this.skipped; }
  get slotCount() { return this.slots.length; }
  get hasUnreadRecord() { return this.slots.some((slot) => slot.state === "encoded"); }

  /**
   * Whether the next step should spend its tail copying a record.
   *
   * The destinations are MAP_READ, so every encoded record writes solver state
   * into host-visible memory and ends the step with a run of transfer commands.
   * Producing one per step served a consumer that takes one every 250 ms in the
   * browser, so seven of every eight were staged and discarded. Arming follows
   * the consumer instead: a record is encoded only once the previous one has
   * been taken. A harness that reads every step (the smoke lanes do, whenever
   * an exact step count is pinned) therefore still gets a record per step, and
   * the record a reader receives is still copied by exactly one step's own
   * encoder — the coherence the ring exists for is untouched.
   */
  get recordDue() { return this.due; }

  encode(encoder: GPUCommandEncoder, sources: LosassoStepSnapshotSources, step: number,
    surfaceKind: "fine" | "coarse" | "adaptive" = "fine"): boolean {
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
    if (sources.fluidResidency) copy(sources.fluidResidency, LAYOUT.fluidResidency, 64);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fluidResidency, 64);
    if (sources.fluidBulkResidency) copy(sources.fluidBulkResidency, LAYOUT.fluidBulkResidency, 64);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fluidBulkResidency, 64);
    if (sources.fineTopology) copy(sources.fineTopology, LAYOUT.fineTopology, 64);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fineTopology, 64);
    if (sources.fineRedistance) copy(sources.fineRedistance, LAYOUT.fineRedistance, 64);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fineRedistance, 64);
    if (sources.fineVolume) copy(sources.fineVolume, LAYOUT.fineVolume, 64);
    else encoder.clearBuffer(slot.buffer, LAYOUT.fineVolume, 64);
    if (sources.adaptive) {
      copy(sources.adaptive.candidateAuthority, LAYOUT.adaptiveCandidateAuthority, 32);
      copy(sources.adaptive.acceptedGraph, LAYOUT.adaptiveAcceptedGraph, 128);
      copy(sources.adaptive.candidateGraph, LAYOUT.adaptiveCandidateGraph, 128);
      copy(sources.adaptive.phiControl, LAYOUT.adaptivePhiControl, 80);
      copy(sources.adaptive.phiReceipts, LAYOUT.adaptivePhiReceipts,
        ADAPTIVE_PHI_RECEIPT_BYTES);
      copy(sources.adaptive.velocityReceipts, LAYOUT.adaptiveVelocityReceipts, 208);
      copy(sources.adaptive.rendererDirectory, LAYOUT.adaptiveRenderer, 32);
      copy(sources.adaptive.massControl, LAYOUT.adaptiveMassControl, 128);
      copy(sources.adaptive.massReceipts, LAYOUT.adaptiveMassReceipts, 128);
      copy(sources.adaptive.velocityMigration, LAYOUT.adaptiveVelocityMigration, 32);
    } else {
      encoder.clearBuffer(slot.buffer, LAYOUT.adaptiveAcceptedGraph,
        LAYOUT.stride - LAYOUT.adaptiveAcceptedGraph);
    }
    slot.sequence = ++this.sequence; slot.step = step; slot.surfaceKind = surfaceKind;
    slot.state = "encoded";
    this.due = false;
    return true;
  }

  async readLatest(): Promise<LosassoStepSnapshotRecord | undefined> {
    if (this.disposed) return undefined;
    let slot: Slot | undefined;
    for (const candidate of this.slots) {
      if (candidate.state !== "encoded") continue;
      if (!slot || candidate.sequence > slot.sequence) slot = candidate;
    }
    // Re-arm on every consumer visit, including one that found nothing: a
    // reader asking for a record is what makes the next step owe one.
    this.due = true;
    if (!slot) return undefined;
    slot.state = "mapping";
    try {
      await slot.buffer.mapAsync(READ);
      const mapped = slot.buffer.getMappedRange();
      const words = (offset: number, count: number) =>
        Uint32Array.from(new Uint32Array(mapped, offset, count));
      const adaptive = slot.surfaceKind === "adaptive" ? Object.freeze({
        candidateAuthority: words(LAYOUT.adaptiveCandidateAuthority, 8),
        acceptedGraph: words(LAYOUT.adaptiveAcceptedGraph, 32),
        candidateGraph: words(LAYOUT.adaptiveCandidateGraph, 32),
        phiControl: words(LAYOUT.adaptivePhiControl, 20),
        phiReceipts: words(LAYOUT.adaptivePhiReceipts,
          OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS),
        velocityReceipts: words(LAYOUT.adaptiveVelocityReceipts, 52),
        renderer: words(LAYOUT.adaptiveRenderer, 8),
        massControl: words(LAYOUT.adaptiveMassControl, 32),
        massReceipts: words(LAYOUT.adaptiveMassReceipts, 32),
        velocityMigration: words(LAYOUT.adaptiveVelocityMigration, 8),
      }) : undefined;
      return Object.freeze({ step: slot.step, surfaceKind: slot.surfaceKind,
        authority: words(LAYOUT.authority, 8), solver: words(LAYOUT.solver, 16),
        fine: words(LAYOUT.fine, 7), coarsePhi: words(LAYOUT.coarsePhi, 15),
        extension: words(LAYOUT.extension, 8), fineTransport: words(LAYOUT.fineTransport, 8),
        fineTopology: words(LAYOUT.fineTopology, 16),
        fineRedistance: words(LAYOUT.fineRedistance, 16),
        fineVolume: words(LAYOUT.fineVolume, 16),
        fluidResidency: words(LAYOUT.fluidResidency, 16),
        fluidBulkResidency: words(LAYOUT.fluidBulkResidency, 16),
        ...(adaptive ? { adaptive } : {}) });
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
