/** Experimental air-band projection. The original effective cell plane stays
 * at offset zero; correction storage is private to the optional transport path. */
export const AIR_EXTENSION_SWEEPS = 8;
export const AIR_EXTENSION_ITERATIONS = 128;
export const AIR_EXTENSION_ENTRY_POINTS = [
  "airBegin", "airClassifyCells", "airSeedFaces", "airExtendFacesA", "airExtendFacesB", "airPrepareRows", "airConnect",
  "airAssemble", "airInitialize", "airReduceInitial", "airApply",
  "airReduceAlpha", "airUpdate", "airReduceBeta", "airDirection",
  "airMeasure", "airReduceFinal", "airCorrect",
] as const;

export function airExtensionLayout(cells: number, rows: number) {
  if (![cells, rows].every(n => Number.isSafeInteger(n) && n >= 0)) {
    throw new RangeError("Invalid air-extension capacity");
  }
  const cellBase = cells;
  const rowBase = cellBase + 3 * cells;
  const header = rowBase + rows;
  const reduction = header + 4;
  return { cells, rows, cellBase, rowBase, header, reduction,
    byteLength: 16 * (reduction + Math.max(1, Math.ceil(cells / 64))) };
}

export function decodeAirExtensionReceipt(values: Float32Array) {
  return {
    ready: values[0] === 1, iterations: values[1], converged: values[2] === 1,
    breakdown: values[3] !== 0, initialResidualSquared: values[4],
    finalResidualSquared: values[5], activeCells: values[6], isolatedCells: values[7],
    initialMaxDivergence: values[8], finalMaxDivergenceError: values[9],
    compatibleMaxDivergence: values[10], correctedFaces: values[11],
  };
}
