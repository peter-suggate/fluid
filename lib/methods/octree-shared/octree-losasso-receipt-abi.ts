/**
 * Wire format of the Losasso end-of-step receipts the engine reads back.
 *
 * The adaptive phi and mass passes belong to the Losasso backend, but the
 * words they publish are read by the shared engine: it copies the receipts
 * into its step-snapshot ring and decodes the mass receipt for the field
 * statistics it reports. A published buffer's layout is not solver physics, so
 * it lives here where both the producing method and the engine that reads it
 * can name it.
 *
 * The producers import these back and re-export them, so the Losasso package,
 * the harness and their tests still name each format in one place.
 */

/**
 * Words in the adaptive phi receipt.
 *
 * Only the count lives here: the engine's snapshot ring needs it to size its
 * slot and slice the mapped record, and nothing about what those words mean.
 */
export const OCTREE_LOSASSO_ADAPTIVE_PHI_RECEIPT_WORDS = 55;

export const OCTREE_LOSASSO_ADAPTIVE_MASS_MAGIC = 0x414d_4153;
export const OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS = 32;

const floatWord = (word: number): number => {
  const bits = new Uint32Array(1); bits[0] = word >>> 0;
  return new Float32Array(bits.buffer)[0]!;
};

export function unpackAdaptiveMassReceipt(words: ArrayLike<number>) {
  if (words.length < OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS) {
    throw new RangeError(`adaptive mass receipt requires ${OCTREE_LOSASSO_ADAPTIVE_MASS_RECEIPT_WORDS} words`);
  }
  return Object.freeze({
    acceptedMass_m3: floatWord(words[0]!), transportedMass_m3: floatWord(words[1]!),
    signedTransportDrift_m3: floatWord(words[2]!),
    maximumDonorWeightDefect: floatWord(words[4]!), donors: words[5]! >>> 0,
    transfers: words[6]! >>> 0, missingRecipients: words[7]! >>> 0,
    handoffSourceMass_m3: floatWord(words[8]!), handoffTargetMass_m3: floatWord(words[9]!),
    signedHandoffDrift_m3: floatWord(words[10]!), handoffLeafCount: words[11]! >>> 0,
    errors: words[12]! >>> 0,
    reconstructionThreshold: floatWord(words[13]!),
    reconstructionTargetUnits: words[14]! >>> 0,
    reconstructionMeasuredUnits: words[15]! >>> 0,
    reconstructionSignMismatches: words[16]! >>> 0,
    firstReconstructionSignMismatchItem: words[17]! >>> 0,
    handoffGraphErrors: words[18]! >>> 0,
    compressedExcessMass_m3: floatWord(words[19]!),
    subIsoMass_m3: floatWord(words[20]!),
    overfullLeafCount: words[21]! >>> 0,
    subIsoLeafCount: words[22]! >>> 0,
  });
}
