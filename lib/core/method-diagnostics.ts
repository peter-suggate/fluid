/**
 * The shapes a method contributes to the panels that print published state.
 *
 * A panel that branches on a grid kind or a method id to decide which cards
 * to draw is asserting that it knows who produced a publication — and it is
 * wrong the moment two methods share a grid kind, or a new method publishes
 * none of the counters the branch assumed. These types let the producer hand
 * over finished rows instead: the panel places and styles them, and never
 * learns what a frontier capacity or a power row is.
 *
 * They are display data, not JSX, for the same reason `RuntimeDialReadouts`
 * is: a method package must not reach into `components/`, and the panel keeps
 * the one decision it can actually make — how a card looks.
 */

/** One card in the live diagnostics grid. */
export interface DiagnosticRow {
  /** Unique within one method's contribution; the panel keys the card by it. */
  readonly id: string;
  readonly label: string;
  readonly value: string;
  readonly unit?: string;
  readonly tone?: "neutral" | "good" | "warn";
  /** Carried only where a lane already selects this card by test id. */
  readonly testId?: string;
}

/**
 * One readout printed under the controls that chose it.
 *
 * Distinct from a diagnostics card because this row has no heading: it sits
 * directly beneath the parameter that produced it and names itself in the
 * detail line, with the tooltip carrying the comparison being made.
 */
export interface ConfigurationReadout {
  /** Unique within one method's contribution; the panel keys the row by it. */
  readonly id: string;
  /** What the two numbers are being compared against; the row has no label. */
  readonly title: string;
  readonly value: string;
  readonly detail: string;
}

/**
 * A cell witness as the panels print it.
 *
 * Shared rather than repeated per producer because the absent case is the one
 * that matters: a method that spelled the missing witness differently would
 * read as a different kind of gap than the one beside it.
 */
export function formatGridLocation(location?: { x: number; y: number; z: number }) {
  return location ? `[${location.x}, ${location.y}, ${location.z}]` : "location pending";
}
