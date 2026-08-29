"use client";

import type {
  PressureJournal,
} from "../lib/core/pressure-journal";
import { useSession } from "../lib/core/session/session-context";

/**
 * The captured pressure solve's convergence, beside the scrub that plays it.
 *
 * The overlay answers *where* the solve is still working; this answers *how
 * fast*, and the two belong together — a frame of the film means something
 * different at the third iteration than at the thirtieth, and the only way to
 * know which you are looking at is to see the curve with your position on it.
 *
 * Three things are drawn that a plain residual plot would not show, and each
 * one is a question the film exists to answer:
 *
 * - **The gated tail.** An encoded iteration is not an executed one: the solve
 *   stops early and the remaining iterations are dispatched but do nothing.
 *   Drawn as a shaded region rather than omitted, because "it converged at 19
 *   of 32" is the interesting fact and a curve that simply ended at 19 would
 *   hide the 13 that were paid for in the command buffer.
 * - **The snapshot stops.** The film has frames only where a snapshot was
 *   taken, and they are log-spaced. Marking them makes the scrub legible —
 *   the gaps are wide at the end because that is where nothing changes — and
 *   each is clickable, which is the fastest way to step the film.
 * - **Recursive against guarded.** The recursive residual is what the
 *   algorithm carries forward; the guarded one is a true ‖b − Ap‖ measured at
 *   the cadence checks. They drift apart, and the drift is a real property of
 *   the solve rather than a defect of the plot, so both are drawn.
 */

/** Decades of residual the plot spans below the seed. */
const FILM_STRIP_DECADES = 8;
const FILM_STRIP_WIDTH = 200;
const FILM_STRIP_HEIGHT = 44;

/** Log position of a relative residual, as a 0..1 fraction from the top. */
function strength(relative: number): number {
  if (!(relative > 0)) return 1;
  return Math.max(0, Math.min(1, -Math.log10(relative) / FILM_STRIP_DECADES));
}

function point(iteration: number, relative: number, encoded: number): string {
  const x = encoded > 0 ? (iteration / encoded) * FILM_STRIP_WIDTH : 0;
  return `${x.toFixed(2)},${(strength(relative) * FILM_STRIP_HEIGHT).toFixed(2)}`;
}

/** Two significant figures in exponent form, for numbers spanning ten decades. */
function residualLabel(value: number): string {
  if (!(value > 0)) return "0";
  if (value >= 0.01) return value.toFixed(3);
  return value.toExponential(1);
}

export function PressureFilmStrip({
  slot,
  onSelectSlot,
}: {
  /** Snapshot index the scrub currently sits on. */
  slot: number;
  onSelectSlot: (slot: number) => void;
}) {
  const session = useSession();
  const journal = session.diagnostics((state) => state.pressureJournal);
  const runState = session.runtime((state) => state.runState);
  if (!journal) {
    return <p className="pressure-film" data-testid="pressure-film-empty">
      <span>{runState === "running"
        // Reading the records maps a buffer the solver owns, which is only
        // allowed at the pause boundary. Saying so is better than an empty
        // panel that looks broken.
        ? "Pause to read the captured film."
        : "No film captured yet — open a Pressure lab view and step a frame."}</span>
    </p>;
  }
  return <FilmStrip journal={journal} slot={slot} onSelectSlot={onSelectSlot} />;
}

function FilmStrip({
  journal, slot, onSelectSlot,
}: {
  journal: PressureJournal;
  slot: number;
  onSelectSlot: (slot: number) => void;
}) {
  const { records, encodedIterations, executedIterations } = journal;
  const recursive = records
    .filter((record) => record.iteration > 0)
    .map((record) => point(record.iteration, record.recursiveRelativeL2, encodedIterations))
    .join(" ");
  // The guarded value only exists at the cadence checks, so its polyline is a
  // sparse one. Joining across the gaps is right: it is the same quantity
  // sampled less often, not a different one.
  const guarded = records
    .filter((record) => record.iteration > 0 && record.guardedRelativeL2 !== undefined)
    .map((record) => point(record.iteration, record.guardedRelativeL2!, encodedIterations))
    .join(" ");
  const gatedFrom = executedIterations < encodedIterations
    ? (executedIterations / encodedIterations) * FILM_STRIP_WIDTH
    : undefined;
  const last = records[executedIterations] ?? records.at(-1);
  const crossing = journal.firstCrossingIteration;

  return <div className="pressure-film" data-testid="pressure-film">
    <svg
      viewBox={`0 0 ${FILM_STRIP_WIDTH} ${FILM_STRIP_HEIGHT}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={`Residual over ${encodedIterations} iterations, ${
        executedIterations} executed`}
    >
      {gatedFrom !== undefined && <rect
        className="pressure-film-gated"
        x={gatedFrom} y={0}
        width={FILM_STRIP_WIDTH - gatedFrom} height={FILM_STRIP_HEIGHT}
      />}
      {guarded && <polyline className="pressure-film-guarded" points={guarded} />}
      {recursive && <polyline className="pressure-film-recursive" points={recursive} />}
      {crossing !== undefined && <line
        className="pressure-film-crossing"
        x1={(crossing / Math.max(1, encodedIterations)) * FILM_STRIP_WIDTH} y1={0}
        x2={(crossing / Math.max(1, encodedIterations)) * FILM_STRIP_WIDTH}
        y2={FILM_STRIP_HEIGHT}
      />}
      {journal.snapshotIterations.map((iteration, index) => {
        const x = (iteration / Math.max(1, encodedIterations)) * FILM_STRIP_WIDTH;
        return <line
          key={iteration}
          className={index === slot ? "pressure-film-stop active" : "pressure-film-stop"}
          x1={x} y1={FILM_STRIP_HEIGHT - 6} x2={x} y2={FILM_STRIP_HEIGHT}
        />;
      })}
    </svg>
    {/* One clickable target per captured frame, laid over the plot. Stepping
        the film by its own stops beats dragging a slider past positions that
        have no frame behind them. */}
    <div className="pressure-film-stops" role="group" aria-label="Captured iterations">
      {journal.snapshotIterations.map((iteration, index) => <button
        key={iteration}
        type="button"
        className={index === slot ? "active" : ""}
        aria-pressed={index === slot}
        title={`Iteration ${iteration}`}
        onClick={() => onSelectSlot(index)}
      >{iteration}</button>)}
    </div>
    <p className="pressure-film-caption">
      <span>{executedIterations} of {encodedIterations} ran</span>
      <code>{residualLabel(journal.initialRelativeL2)} → {
        residualLabel(last?.guardedRelativeL2 ?? last?.recursiveRelativeL2 ?? 0)}</code>
      {crossing !== undefined && <em>crossed at {crossing}</em>}
    </p>
  </div>;
}
