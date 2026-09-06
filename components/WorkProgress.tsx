"use client";

import { useEffect, useState } from "react";
import type { WorkProgress as WorkProgressValue } from "../lib/core/work-progress";

/** One visual grammar for shader work, resource startup and background rebuilds. */
export function WorkProgress({ progress, compact = false }: { progress: WorkProgressValue; compact?: boolean }) {
  const [now, setNow] = useState(0);
  useEffect(() => {
    if (progress.startedAt_ms === undefined || progress.state === "complete") return;
    const tick = () => setNow(performance.now());
    tick();
    const timer = window.setInterval(tick, 1000);
    return () => window.clearInterval(timer);
  }, [progress.startedAt_ms, progress.state]);
  const determinate = Number.isFinite(progress.total) && progress.total! > 0 && Number.isFinite(progress.completed);
  const completed = Math.max(0, Math.min(progress.completed ?? 0, progress.total ?? Infinity));
  const elapsed = progress.startedAt_ms === undefined ? undefined : Math.max(0, now - progress.startedAt_ms) / 1000;
  const count = determinate ? `${completed.toLocaleString()} / ${progress.total!.toLocaleString()} ${progress.unit ?? "items"}` : undefined;
  // Compact is one line: label, then the one number worth a glance. The filler
  // words ("Working…") and the elapsed suffix are the full card's vocabulary.
  const brief = count ?? (elapsed !== undefined ? `${elapsed.toFixed(0)} s` : undefined);
  return <div className="work-progress" data-state={progress.state} data-compact={compact || undefined}>
    <div className="work-progress-heading" role="status" aria-live="polite">
      <strong>{progress.label}</strong>
      {compact
        ? brief && <span>{brief}</span>
        : progress.generation !== undefined && <span>Build {progress.generation}</span>}
    </div>
    {!compact && progress.phases && progress.phases.some((phase) => phase.id === progress.phase) && <ol className="work-progress-phases" aria-label="Work phases">
      {progress.phases.map((phase) => <li key={phase.id} aria-current={phase.id === progress.phase ? "step" : undefined}>{phase.label}</li>)}
    </ol>}
    {progress.state === "active" && <div className="work-progress-meter" data-indeterminate={!determinate || undefined}>
      <progress aria-label={progress.label} aria-valuetext={count ?? "Working; completion count unavailable"}
        max={determinate ? progress.total : 1} {...(determinate ? { value: completed } : {})} />
    </div>}
    {!compact && <div className="work-progress-summary" aria-live="off">
      <span>{count ?? (progress.state === "waiting" ? "Waiting" : progress.state === "error" ? "Needs attention" : progress.state === "complete" ? "Complete" : "Working…")}</span>
      {elapsed !== undefined && <span>{elapsed.toFixed(0)} s elapsed</span>}
    </div>}
    {!compact && progress.detail && <p>{progress.detail}</p>}
  </div>;
}
