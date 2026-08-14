export interface FineLevelSetActivityCensus {
  readonly step: number;
  readonly generation: number;
  readonly pageDeltaGeneration: number;
  readonly receiptValid: boolean;
  readonly liveBandPages: number;
  /** Pages whose transport commit set PAGE_DIRTY before halo expansion. */
  readonly dirtyPages: number;
  readonly dirtyHaloPages: number;
  readonly supportHaloPages: number;
  readonly changedPages: number;
  readonly addedPages: number;
  readonly retiredPages: number;
  readonly maximumDisplacementFineCells: number | null;
  /** Pages dispatched through expensive Losasso characteristic transport. */
  readonly activeTransportPages: number;
  /** Live pages carried bit-identically by the Tier-A sleep path. */
  readonly sleepingTransportPages: number;
  /** Input generation classified by transport before topology publishes its output generation. */
  readonly transportInputPages: number;
  readonly transportReceiptValid: boolean;
  readonly executedSolveIterations: number;
  readonly transportActivityFraction: number;
  readonly dirtyFraction: number;
  readonly dirtyHaloFraction: number;
  readonly supportHaloFraction: number;
}

/**
 * Decode the compact, immutable receipts already published by one fine-band
 * advance. This is diagnostics-only: callers copy the two headers after the
 * advance and never feed the result back into scheduling.
 */
export function decodeFineLevelSetActivityCensus(
  step: number,
  worklistHeader: ArrayLike<number>,
  pageDeltaHeader: ArrayLike<number>,
  executedSolveIterations: number,
  transportControl?: ArrayLike<number>,
): FineLevelSetActivityCensus | undefined {
  if (!Number.isSafeInteger(step) || step < 1
    || worklistHeader.length < 4 || pageDeltaHeader.length < 16) return undefined;
  const generation = worklistHeader[0]! >>> 0;
  const liveBandPages = worklistHeader[1]! >>> 0;
  const capacity = worklistHeader[2]! >>> 0;
  if (generation === 0xffff_ffff || liveBandPages === 0xffff_ffff
    || liveBandPages > capacity) return undefined;

  const pageDeltaGeneration = pageDeltaHeader[1]! >>> 0;
  const dirtyPages = pageDeltaHeader[5]! >>> 0;
  const dirtyHaloPages = pageDeltaHeader[2]! >>> 0;
  const supportHaloPages = pageDeltaHeader[3]! >>> 0;
  const changedPages = pageDeltaHeader[0]! >>> 0;
  const addedPages = pageDeltaHeader[6]! >>> 0;
  const retiredPages = pageDeltaHeader[7]! >>> 0;
  const countsValid = ![dirtyPages, dirtyHaloPages, supportHaloPages, addedPages, retiredPages]
    .some((count) => count > capacity) && changedPages <= 2 * capacity;
  const activeTransportPages = transportControl && transportControl.length > 15
    ? transportControl[14]! >>> 0 : liveBandPages;
  const sleepingTransportPages = transportControl && transportControl.length > 15
    ? transportControl[15]! >>> 0 : 0;
  const transportInputPages = activeTransportPages + sleepingTransportPages;
  const transportReceiptValid = transportControl === undefined
    || (transportControl.length > 15 && (transportControl[3]! >>> 0) === 1
      && activeTransportPages <= capacity && sleepingTransportPages <= capacity
      && transportInputPages <= capacity);
  const receiptValid = (pageDeltaHeader[15]! >>> 0) === 1
    && pageDeltaGeneration === generation && countsValid;
  const fraction = (count: number) => !receiptValid || liveBandPages === 0
    ? 0 : count / liveBandPages;
  const displacement = pageDeltaHeader[9]! >>> 0;
  return Object.freeze({
    step,
    generation,
    pageDeltaGeneration,
    receiptValid,
    liveBandPages,
    dirtyPages,
    dirtyHaloPages,
    supportHaloPages,
    changedPages,
    addedPages,
    retiredPages,
    maximumDisplacementFineCells: !receiptValid || displacement === 0xffff_ffff
      ? null : displacement,
    activeTransportPages,
    sleepingTransportPages,
    transportInputPages,
    transportReceiptValid,
    executedSolveIterations: Number.isSafeInteger(executedSolveIterations)
      && executedSolveIterations >= 0 ? executedSolveIterations : 0,
    transportActivityFraction: !transportReceiptValid || transportInputPages === 0
      ? 0 : activeTransportPages / transportInputPages,
    dirtyFraction: fraction(dirtyPages),
    dirtyHaloFraction: fraction(dirtyHaloPages),
    supportHaloFraction: fraction(supportHaloPages),
  });
}
