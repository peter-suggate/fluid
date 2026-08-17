import {
  SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER,
  SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY,
  SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY_COUNT,
  SPARSE_CM12_PRESSURE_SOLVE_TAIL_INDIRECT_WORDS,
  sparseCM12PressureSolveTailIndirectByteOffset,
  type SparseCM12PressureSolveAuthorityLayout,
  type SparseCM12PressureSolveTailFamilyName,
} from "./sparse-cm12-pressure-solve-authority";

/** Host-side, fixed-command-stream adapter for PSA1's GPU-authored tail banks. */
export const SPARSE_CM12_PRESSURE_TAIL_AUTHORITY_ABI = "PTL1/v1";
export const SPARSE_CM12_PRESSURE_TAIL_BANK_BYTES = 4
  * SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY_COUNT
  * SPARSE_CM12_PRESSURE_SOLVE_TAIL_INDIRECT_WORDS;

export interface SparseCM12PressureTailCopyPlan {
  readonly abi: typeof SPARSE_CM12_PRESSURE_TAIL_AUTHORITY_ABI;
  readonly sourceByteOffsets: readonly [number, number];
  readonly destinationByteLength: number;
  readonly familyByteOffsets: Readonly<Record<SparseCM12PressureSolveTailFamilyName, number>>;
  readonly publisherEntryPoints: readonly [
    "publishSparseCM12PressureTailA", "publishSparseCM12PressureTailB",
  ];
}

export function createSparseCM12PressureTailCopyPlan(
  layout: SparseCM12PressureSolveAuthorityLayout,
): SparseCM12PressureTailCopyPlan {
  const familyByteOffsets = Object.freeze(Object.fromEntries(
    Object.entries(SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY).map(([name, family]) => [
      name, 4 * SPARSE_CM12_PRESSURE_SOLVE_TAIL_INDIRECT_WORDS * family,
    ]),
  )) as Readonly<Record<SparseCM12PressureSolveTailFamilyName, number>>;
  return Object.freeze({
    abi: SPARSE_CM12_PRESSURE_TAIL_AUTHORITY_ABI,
    sourceByteOffsets: Object.freeze([
      sparseCM12PressureSolveTailIndirectByteOffset(layout, 0, "cell"),
      sparseCM12PressureSolveTailIndirectByteOffset(layout, 1, "cell"),
    ]) as readonly [number, number],
    destinationByteLength: SPARSE_CM12_PRESSURE_TAIL_BANK_BYTES,
    familyByteOffsets,
    publisherEntryPoints: Object.freeze([
      "publishSparseCM12PressureTailA", "publishSparseCM12PressureTailB",
    ]) as SparseCM12PressureTailCopyPlan["publisherEntryPoints"],
  });
}

/**
 * Two physically distinct INDIRECT buffers. The source arena is storage-only
 * during publication; callers close that compute pass before `encodeCopy`.
 * No CPU count, convergence result, or readback selects either bank.
 */
export class WebGPUSparseCM12PressureTailAuthority {
  readonly plan: SparseCM12PressureTailCopyPlan;
  readonly buffers: readonly [GPUBuffer, GPUBuffer];

  constructor(
    device: GPUDevice,
    private readonly sourceArena: GPUBuffer,
    layout: SparseCM12PressureSolveAuthorityLayout,
  ) {
    this.plan = createSparseCM12PressureTailCopyPlan(layout);
    const create = (bank: "A" | "B") => device.createBuffer({
      label: `Sparse CM12 PSA1 pressure tail ${bank} indirect snapshot`,
      size: this.plan.destinationByteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT,
    });
    this.buffers = Object.freeze([create("A"), create("B")]);
  }

  /** Must follow the matching GPU publisher and a compute-pass boundary. */
  encodeCopy(encoder: GPUCommandEncoder, bank: 0 | 1): void {
    encoder.copyBufferToBuffer(this.sourceArena, this.plan.sourceByteOffsets[bank],
      this.buffers[bank], 0, this.plan.destinationByteLength);
  }

  dispatch(pass: GPUComputePassEncoder, bank: 0 | 1,
    family: SparseCM12PressureSolveTailFamilyName): void {
    pass.dispatchWorkgroupsIndirect(this.buffers[bank], this.plan.familyByteOffsets[family]);
  }

  destroy(): void { this.buffers[0].destroy(); this.buffers[1].destroy(); }
}

export interface SparseCM12PressureTailSeam {
  readonly seam: number;
  readonly bank: 0 | 1;
  readonly publisher: "publishSparseCM12PressureTailA" | "publishSparseCM12PressureTailB";
}

/** Construction-fixed A/B alternation; runtime convergence never changes it. */
export function sparseCM12PressureTailSeams(count: number): readonly SparseCM12PressureTailSeam[] {
  if (!Number.isSafeInteger(count) || count < 1) {
    throw new RangeError("PTL1 seam count must be a positive integer");
  }
  return Object.freeze(Array.from({ length: count }, (_, seam) => {
    const bank = (seam & 1) as 0 | 1;
    return Object.freeze({ seam, bank,
      publisher: bank === 0 ? "publishSparseCM12PressureTailA"
        : "publishSparseCM12PressureTailB" });
  }));
}

export interface SparseCM12PressureTailQASnapshot {
  readonly acceptedGeneration: number;
  readonly publishedGeneration: number;
  readonly fault: number;
  readonly arithmeticActive: boolean;
  readonly familyWorkgroups: readonly [number, number, number, number];
  /** Exact externally visible u32 words after the seam, in canonical order. */
  readonly outputWords: readonly number[];
}

export interface SparseCM12PressureTailQAReceipt {
  readonly exact: boolean;
  readonly firstMismatchSeam: number | null;
  readonly firstMismatchFamily: SparseCM12PressureSolveTailFamilyName | null;
  readonly firstMismatchWord: number | null;
  readonly reason: "generation" | "fail-closed" | "work" | "output" | null;
}

/**
 * Telemetry-only acceptance seam for one published PSA tail bank.  The GPU
 * triplets remain the scheduling authority; this structure exists so static
 * and construction gates can reject a missing/stale publisher without
 * inventing a host convergence decision.
 */
export interface SparseCM12PressureTailGenerationReceipt {
  readonly phaseAccepted: boolean;
  readonly acceptedGeneration: number;
  readonly publishedGeneration: number;
  readonly activeBank: number;
  readonly fault: number;
  /** Construction/telemetry seam being certified; never a runtime selector. */
  readonly predicate: "ordinary" | "recovery";
  /** Exact GPU predicate sampled at the matching publication seam. */
  readonly predicateActive: boolean;
  /** PSA-authored work expected when that predicate is true. */
  readonly expectedFamilyWorkgroups: readonly [number, number, number, number];
  readonly familyWorkgroups: readonly [number, number, number, number];
}

export interface SparseCM12PressureTailGenerationAcceptance {
  readonly accepted: boolean;
  readonly reason: "psa-not-accepted" | "psa-fault" | "missing-publication"
    | "stale-publication" | "invalid-bank" | "invalid-work"
    | "predicate-work" | "fail-open-work" | null;
}

function validWorkgroups(values: readonly number[]): boolean {
  return values.length === SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY_COUNT
    && values.every((value) => Number.isSafeInteger(value) && value >= 0);
}

/**
 * Fail-closed generation receipt shared by bootstrap and recurring seams.
 * Generation zero is never a successful PSA publication.  A PSA fault is
 * reported before the generation comparison because the publisher correctly
 * leaves the last accepted generation intact while zeroing indirect work.
 */
export function acceptSparseCM12PressureTailGeneration(
  receipt: SparseCM12PressureTailGenerationReceipt,
): SparseCM12PressureTailGenerationAcceptance {
  if (!validWorkgroups(receipt.expectedFamilyWorkgroups)
    || !validWorkgroups(receipt.familyWorkgroups)) {
    return { accepted: false, reason: "invalid-work" };
  }
  const hasWork = receipt.familyWorkgroups.some((count) => count !== 0);
  if ((!receipt.phaseAccepted || receipt.fault !== 0) && hasWork) {
    return { accepted: false, reason: "fail-open-work" };
  }
  if (!receipt.phaseAccepted) return { accepted: false, reason: "psa-not-accepted" };
  if (receipt.fault !== 0) return { accepted: false, reason: "psa-fault" };
  if (receipt.acceptedGeneration === 0 || receipt.publishedGeneration === 0) {
    return { accepted: false, reason: "missing-publication" };
  }
  if (receipt.publishedGeneration !== receipt.acceptedGeneration) {
    return { accepted: false, reason: "stale-publication" };
  }
  if (receipt.activeBank !== 0 && receipt.activeBank !== 1) {
    return { accepted: false, reason: "invalid-bank" };
  }
  const expected = receipt.predicateActive
    ? receipt.expectedFamilyWorkgroups : [0, 0, 0, 0] as const;
  if (receipt.familyWorkgroups.some((count, family) => count !== expected[family])) {
    return { accepted: false, reason: "predicate-work" };
  }
  return { accepted: true, reason: null };
}

const tailFamilies = Object.keys(
  SPARSE_CM12_PRESSURE_SOLVE_TAIL_FAMILY,
) as SparseCM12PressureSolveTailFamilyName[];

/** Construction-only comparison against the fully encoded guarded path. */
export function compareSparseCM12PressureTailAuthorityQA(
  full: readonly SparseCM12PressureTailQASnapshot[],
  sparse: readonly SparseCM12PressureTailQASnapshot[],
): SparseCM12PressureTailQAReceipt {
  if (full.length !== sparse.length) throw new RangeError("PTL1 QA seam counts differ");
  for (let seam = 0; seam < full.length; seam += 1) {
    const expected = full[seam]!; const actual = sparse[seam]!;
    if (actual.publishedGeneration !== actual.acceptedGeneration) {
      return { exact: false, firstMismatchSeam: seam, firstMismatchFamily: null,
        firstMismatchWord: null, reason: "generation" };
    }
    for (let family = 0; family < tailFamilies.length; family += 1) {
      const expectedWork = actual.fault === 0 && actual.arithmeticActive
        ? expected.familyWorkgroups[family]! : 0;
      if (actual.familyWorkgroups[family] !== expectedWork) {
        return { exact: false, firstMismatchSeam: seam,
          firstMismatchFamily: tailFamilies[family]!, firstMismatchWord: null,
          reason: actual.fault === 0 ? "work" : "fail-closed" };
      }
    }
    const words = Math.max(expected.outputWords.length, actual.outputWords.length);
    for (let word = 0; word < words; word += 1) {
      if (expected.outputWords[word] !== actual.outputWords[word]) {
        return { exact: false, firstMismatchSeam: seam, firstMismatchFamily: null,
          firstMismatchWord: word, reason: "output" };
      }
    }
  }
  return { exact: true, firstMismatchSeam: null, firstMismatchFamily: null,
    firstMismatchWord: null, reason: null };
}

export interface SparseCM12PressureTailIntegrationStep {
  readonly kind: "publish" | "close-pass" | "copy" | "dispatch" | "gate";
  readonly bank?: 0 | 1;
  readonly family?: SparseCM12PressureSolveTailFamilyName;
}

/** Static manifest check: stale indirect banks cannot be consumed. */
export function assertSparseCM12PressureTailIntegration(
  steps: readonly SparseCM12PressureTailIntegrationStep[],
): void {
  let published: 0 | 1 | null = null; let closed = false; let copied: 0 | 1 | null = null;
  let priorBank: 0 | 1 | null = null;
  for (const step of steps) {
    if (step.kind === "publish") {
      if (step.bank === undefined || step.bank === priorBank) {
        throw new Error("PTL1 publishers must alternate A/B banks");
      }
      published = step.bank; priorBank = step.bank; closed = false; copied = null;
    } else if (step.kind === "close-pass") {
      if (published === null) throw new Error("PTL1 pass closed before publication");
      closed = true;
    } else if (step.kind === "copy") {
      if (!closed || step.bank !== published) {
        throw new Error("PTL1 copy must follow the matching publisher and pass boundary");
      }
      copied = step.bank!;
    } else if (step.kind === "dispatch") {
      if (copied === null || step.bank !== copied || step.family === undefined) {
        throw new Error("PTL1 dispatch consumed an unpublished or aliased bank");
      }
    } else if (step.kind === "gate") {
      if (copied === null) throw new Error("PTL1 gate lacks a published tail bank");
      published = null; copied = null; closed = false;
    }
  }
  if (published !== null || copied !== null) throw new Error("PTL1 manifest ends mid-seam");
}

/** Header byte offset used by telemetry-only generation receipts. */
export function sparseCM12PressureTailPublishedGenerationByteOffset(
  layout: SparseCM12PressureSolveAuthorityLayout,
): number {
  return 4 * (layout.baseWords
    + SPARSE_CM12_PRESSURE_SOLVE_AUTHORITY_HEADER.tailPublishedGeneration);
}
