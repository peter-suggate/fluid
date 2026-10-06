"use client";

import { useEffect, useState } from "react";

/**
 * Everything the pane is waiting on, as one mark.
 *
 * This was a card and a stack of pills. The card spent a heading, an operation
 * line, a strip of phase tabs, a meter, a count, an elapsed clock and two
 * sentences on one piece of work; each further piece of work then hung its own
 * box under it with its own meter. Three boxes and three bars to say "not yet"
 * — and the two sentences said the same thing twice.
 *
 * One plate instead, the same one every chip in the studio stands on. A ring
 * carries the whole of the progress, so there is exactly one indicator however
 * many things are running; under it, one line per piece of work with its count,
 * and at most one sentence about what is waiting on them. The phase strip is
 * gone as a control and kept as arithmetic: a phased entry reports how far
 * through *all* its phases it is, so the ring advances across them instead of
 * running back to empty at each one.
 *
 * Nothing here estimates. A fraction is only ever counted work over total work,
 * and an entry that has no count turns the ring into a spinner rather than
 * guessing.
 */

export interface ActivityEntry {
  readonly id: string;
  readonly label: string;
  readonly state: "active" | "waiting" | "error";
  /** `completed / total unit`, already phrased; absent when the work has no count. */
  readonly count?: string;
  /** Counted progress through the whole of this entry, 0–1; absent when unknown. */
  readonly fraction?: number;
  /** Only said for an entry that needs attention: why it is not moving. */
  readonly detail?: string;
  readonly startedAt_ms?: number;
}

/** Long enough that a reader starts to wonder whether it is stuck. */
const ELAPSED_AFTER_S = 8;

function useElapsed_s(startedAt_ms: number | undefined): number | undefined {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (startedAt_ms === undefined) return;
    const tick = () => setNow(performance.now());
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [startedAt_ms]);
  return startedAt_ms === undefined || now === 0 ? undefined : Math.max(0, now - startedAt_ms) / 1000;
}

export function ActivityIndicator({ title, entries, note }: {
  /** What all of this is, in the pane's words. Absent, a lone entry names itself. */
  readonly title?: string;
  readonly entries: readonly ActivityEntry[];
  /** What is waiting on this work — one sentence, or nothing. */
  readonly note?: string;
}) {
  const active = entries.filter((entry) => entry.state === "active");
  const attention = entries.length > active.length;
  const counted = active.flatMap((entry) => entry.fraction === undefined ? [] : [entry.fraction]);
  // Every running entry has to be counted for the ring to be a fraction: one
  // uncounted piece of work makes the whole of it unknown.
  const fraction = active.length > 0 && counted.length === active.length
    ? Math.max(0, Math.min(1, counted.reduce((sum, value) => sum + value, 0) / counted.length))
    : undefined;
  const starts = active.flatMap((entry) => entry.startedAt_ms === undefined ? [] : [entry.startedAt_ms]);
  const elapsed_s = useElapsed_s(starts.length > 0 ? Math.min(...starts) : undefined);
  if (entries.length === 0) return null;
  // A lone entry under no title is its own headline; its count rides beside it.
  const lone = title === undefined && entries.length === 1 ? entries[0] : undefined;
  const heading = title ?? lone?.label ?? "Working in the background";
  const rows = lone ? [] : entries;
  const percent = fraction === undefined ? undefined : Math.round(fraction * 100);
  return <div className="activity" data-tone={attention ? "attention" : undefined}
    data-testid="activity-indicator" role="group" aria-label="Background work">
    <div className="activity-head">
      {active.length > 0
        ? <svg className="activity-ring" viewBox="0 0 16 16" width={16} height={16}
          data-indeterminate={percent === undefined || undefined}
          role="progressbar" aria-label={heading} aria-valuemin={0} aria-valuemax={100}
          {...(percent === undefined ? { "aria-valuetext": "Working; completion count unavailable" } : { "aria-valuenow": percent })}>
          <circle className="activity-ring-track" cx={8} cy={8} r={6} />
          <circle className="activity-ring-value" cx={8} cy={8} r={6} pathLength={100}
            strokeDasharray={`${percent ?? 26} 100`} transform="rotate(-90 8 8)" />
        </svg>
        : <i className="activity-flag" aria-hidden />}
      <strong role="status">{heading}</strong>
      {lone?.count && <span className="activity-meta">{lone.count}</span>}
      {elapsed_s !== undefined && elapsed_s >= ELAPSED_AFTER_S && <span className="activity-meta">{elapsed_s.toFixed(0)} s</span>}
    </div>
    {lone?.detail && <p className="activity-note">{lone.detail}</p>}
    {rows.length > 0 && <ul className="activity-rows">
      {rows.map((entry) => <li key={entry.id} data-state={entry.state}>
        <span>{entry.label}</span>
        {entry.count && <span className="activity-meta">{entry.count}</span>}
        {entry.detail && <p>{entry.detail}</p>}
      </li>)}
    </ul>}
    {note && <p className="activity-note">{note}</p>}
  </div>;
}
