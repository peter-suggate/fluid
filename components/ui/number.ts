/**
 * How every control in the app prints and bounds a number.
 *
 * There were three answers to this — the studio's `formatNumber` (fixed digits,
 * exponent under a thousandth), the toolstrip's `printedNumber` (the step's own
 * precision) and the pipeline's bare `toFixed` — and a quantity shown in two
 * panels could read as two different figures. One module now, and the two
 * policies it keeps are named for what they are for.
 */

/** Keep a value inside its authored ends; either end may be open. */
export function clampNumber(value: number, min?: number, max?: number): number {
  let next = value;
  if (min !== undefined && next < min) next = min;
  if (max !== undefined && next > max) next = max;
  return next;
}

/**
 * The decimals a step is worth.
 *
 * Read off the step's own decimal text rather than its logarithm, because a
 * step of 0.0125 is four decimals even though it is under a tenth, and a field
 * that printed two would turn one press of the up arrow into a value the reader
 * never chose.
 */
export function stepDecimals(step: number): number {
  const text = String(step);
  if (text.includes("e")) return 6;
  const dot = text.indexOf(".");
  return dot < 0 ? 0 : Math.min(6, text.length - dot - 1);
}

/**
 * A readout: a fixed number of decimals, and an exponent for a magnitude the
 * decimals would print as zero.
 */
export function formatNumber(value: number, digits = 3): string {
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
  return value.toFixed(digits);
}

/**
 * An entry: the value at its step's precision, float noise dropped.
 *
 * A quantity snapped to a lattice arrives as 0.6000000000000001, and a box
 * sized for the number it means shows "0.60000000" and clips the rest. Rounded
 * for display only, so the document keeps holding the exact figure it computed.
 * `digits`, when given, wins over the step: a caller that authored a precision
 * wants that one.
 */
export function printNumber(value: number, { step, digits }: { step?: number; digits?: number } = {}): string {
  if (digits !== undefined) return value.toFixed(digits);
  if (step === undefined) return String(value);
  return String(Number(value.toFixed(stepDecimals(step))));
}
