import type { GPUEulerianInfo } from "../../core/webgpu-eulerian";
import { SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE } from
  "./sparse-cm12-canonical-membership";

export interface SparseCM12PressureTopologyTerminalReceipt {
  readonly encodedStep: number;
  readonly topologyGeneration: number;
  readonly committedBrickCount: number;
}

export interface SparseCM12PressureTopologyWorkReceipt {
  readonly acceptedCellCount: number;
  readonly acceptedRowCount: number;
  readonly pressureCellCount: number;
  readonly pressureActiveRowCount: number;
  readonly pcm: NonNullable<GPUEulerianInfo["adaptivePressureCanonicalMembership"]>;
  readonly authorities?: NonNullable<NonNullable<
    GPUEulerianInfo["adaptivePressureTopologyAttribution"]>["authorities"]>;
}

export type SparseCM12PressureTopologyAttribution = NonNullable<
  GPUEulerianInfo["adaptivePressureTopologyAttribution"]
>;

const nonnegativeInteger = (value: number): number =>
  Number.isSafeInteger(value) && value >= 0 ? value : 0;

function pcmReceiptMatched(
  pcm: SparseCM12PressureTopologyWorkReceipt["pcm"],
): boolean {
  const accepted = SPARSE_CM12_CANONICAL_MEMBERSHIP_PHASE.accepted;
  return pcm.cell.phase === accepted && pcm.row.phase === accepted
    && pcm.cell.fault === 0 && pcm.row.fault === 0
    && pcm.cell.acceptedGeneration === pcm.cell.candidateGeneration
    && pcm.row.acceptedGeneration === pcm.row.candidateGeneration;
}

/**
 * Associates one terminal diagnostics tuple with the topology repair that ran
 * earlier in that advance. Exact input attribution needs either the immediately
 * preceding terminal tuple or proof that no generation changed across a wider
 * telemetry gap. It never guesses through an unobserved generation change.
 */
export function sparseCM12PressureTopologyAttribution(options: {
  readonly current: SparseCM12PressureTopologyTerminalReceipt;
  readonly prior?: SparseCM12PressureTopologyTerminalReceipt;
  readonly work: SparseCM12PressureTopologyWorkReceipt;
}): SparseCM12PressureTopologyAttribution {
  const current = options.current;
  const prior = options.prior;
  const adjacent = prior !== undefined && current.encodedStep === prior.encodedStep + 1;
  const unchangedAcrossGap = prior !== undefined && current.encodedStep > prior.encodedStep + 1
    && current.topologyGeneration === prior.topologyGeneration;
  const initial = current.encodedStep === 0;
  const matched = adjacent || unchangedAcrossGap || initial;
  const inputTopologyGeneration = adjacent
    ? prior.topologyGeneration
    : matched ? current.topologyGeneration : undefined;
  const priorCommittedBrickCount = adjacent
    ? prior.committedBrickCount
    : matched ? 0 : undefined;
  const pcmMatched = pcmReceiptMatched(options.work.pcm);
  const gap = prior ? current.encodedStep - prior.encodedStep : undefined;
  const detail = matched
    ? adjacent
      ? "Matched to the immediately prior end-frame topology receipt."
      : initial
        ? "Construction receipt: no prior physics advance committed topology."
        : `No topology generation changed across the ${gap}-advance telemetry gap; prior commit is exactly zero.`
    : prior
      ? `Unavailable: topology generation changed across an unobserved ${gap}-advance telemetry gap.`
      : "Unavailable until a prior terminal topology receipt has been observed.";
  const authorityInputMatched = matched && inputTopologyGeneration !== undefined
    && options.work.authorities?.inputTopologyGeneration === inputTopologyGeneration;
  const authorities = options.work.authorities === undefined ? undefined : Object.freeze({
    ...options.work.authorities,
    status: authorityInputMatched ? options.work.authorities.status : "unavailable" as const,
  });
  return Object.freeze({
    status: matched ? "matched" : "unavailable",
    encodedStep: nonnegativeInteger(current.encodedStep),
    ...(inputTopologyGeneration === undefined ? {} : {
      inputTopologyGeneration: nonnegativeInteger(inputTopologyGeneration),
    }),
    ...(priorCommittedBrickCount === undefined ? {} : {
      priorCommittedBrickCount: nonnegativeInteger(priorCommittedBrickCount),
    }),
    currentEndFrameTopologyGeneration: nonnegativeInteger(current.topologyGeneration),
    currentEndFrameCommittedBrickCount: nonnegativeInteger(current.committedBrickCount),
    acceptedCellCount: nonnegativeInteger(options.work.acceptedCellCount),
    acceptedRowCount: nonnegativeInteger(options.work.acceptedRowCount),
    pressureCellCount: nonnegativeInteger(options.work.pressureCellCount),
    pressureActiveRowCount: nonnegativeInteger(options.work.pressureActiveRowCount),
    pcmCellDirtyLeafCount: nonnegativeInteger(options.work.pcm.cell.dirtyCount),
    pcmRowDirtyLeafCount: nonnegativeInteger(options.work.pcm.row.dirtyCount),
    pcmCellAcceptedGeneration: nonnegativeInteger(options.work.pcm.cell.acceptedGeneration),
    pcmRowAcceptedGeneration: nonnegativeInteger(options.work.pcm.row.acceptedGeneration),
    pcmMatched,
    ...(authorities === undefined ? {} : { authorities }),
    detail,
  });
}

/** Readback-side state only. It never participates in scheduling. */
export class SparseCM12PressureTopologyAttributionTracker {
  private terminal?: SparseCM12PressureTopologyTerminalReceipt;
  private latest?: SparseCM12PressureTopologyAttribution;

  observe(options: {
    readonly current: SparseCM12PressureTopologyTerminalReceipt;
    readonly work: SparseCM12PressureTopologyWorkReceipt;
  }): SparseCM12PressureTopologyAttribution {
    if (this.terminal?.encodedStep === options.current.encodedStep && this.latest) {
      return this.latest;
    }
    const result = sparseCM12PressureTopologyAttribution({
      current: options.current,
      prior: this.terminal,
      work: options.work,
    });
    this.terminal = Object.freeze({ ...options.current });
    this.latest = result;
    return result;
  }
}
