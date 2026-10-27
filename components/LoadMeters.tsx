"use client";

import { useHostLoadMeters, useHostLoadStore } from "../lib/core/host-load";
import type { PaneSession } from "../lib/core/session/session";
import { LOAD_SAMPLE_INTERVAL_MS, type CPUPressureState } from "../lib/core/system-load";

/**
 * What the machine is doing, as a line per thing measured.
 *
 * Each line is named for what is actually measured (see `system-load.ts`): how
 * much of the time a thread could not answer, how much of the time this tab's
 * GPU queue held work, and the platform's own word for CPU pressure. None of
 * them is a utilisation percentage, because a page cannot read one; the hints
 * say so. A worker that has gone quiet is reported as such, since a report
 * that never arrives is the clearest reading of a blocked thread there is.
 */

/** A worker report this overdue means its thread has not been free to send one. */
const WORKER_SILENT_MS = LOAD_SAMPLE_INTERVAL_MS * 3;

const PRESSURE_TONE: Readonly<Record<CPUPressureState, string>> = {
  nominal: "good", fair: "good", serious: "warn", critical: "alert",
};

function Sparkline({ values }: { values: readonly number[] }) {
  const width = 64, height = 14;
  if (values.length < 2) return <svg className="load-spark" width={width} height={height} aria-hidden />;
  const step = width / (values.length - 1);
  const points = values.map((value, index) =>
    `${(index * step).toFixed(1)},${(height - 1 - Math.max(0, Math.min(1, value)) * (height - 2)).toFixed(1)}`);
  return <svg className="load-spark" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden>
    <polygon className="load-spark-area" points={`0,${height} ${points.join(" ")} ${width},${height}`} />
    <polyline className="load-spark-line" points={points.join(" ")} />
  </svg>;
}

const percent = (value: number) => `${Math.round(value * 100)}%`;

function LoadLine({ name, hint, values, readout, tone, testId }: {
  name: string;
  hint: string;
  values: readonly number[];
  readout: string;
  tone?: string;
  testId?: string;
}) {
  return <div className="load-line" title={hint} data-tone={tone} data-testid={testId}>
    <span className="load-name">{name}</span>
    <Sparkline values={values} />
    <output className="load-value">{readout}</output>
  </div>;
}

export function LoadMeters({ session }: { readonly session: PaneSession }) {
  useHostLoadMeters(true);
  const uiThread = useHostLoadStore((state) => state.uiThread);
  const pressure = useHostLoadStore((state) => state.cpuPressure);
  const worker = session.diagnostics((state) => state.workerLoad);
  const ui = uiThread.at(-1);
  const latest = worker.at(-1);
  // The window's own clock ticks with its meter, so "now" is its last sample.
  const silent_ms = ui && latest ? ui.at_ms - latest.at_ms : 0;
  const workerSilent = silent_ms > WORKER_SILENT_MS;
  const stalls = uiThread.reduce((sum, sample) => sum + sample.stalls, 0);
  const gpu = worker.flatMap((sample) => sample.gpu ? [sample.gpu.busy] : []);
  return <div className="load-meters" role="group" aria-label="Machine load" data-testid="load-meters">
    {pressure && <div className="load-line" data-tone={PRESSURE_TONE[pressure]}
      title="The platform's own verdict on the whole machine's CPU (Compute Pressure API): nominal, fair, serious or critical.">
      <span className="load-name">CPU</span>
      <i className="load-dot" aria-hidden />
      <output className="load-value">{pressure}</output>
    </div>}
    <LoadLine name="Worker" testId="load-worker"
      hint="Share of the time the render worker could not answer a timer: shader modules, uploads and encoding run here. Awaiting an async pipeline compile leaves it free."
      values={worker.map((sample) => sample.thread.blocked)}
      readout={workerSilent ? `silent ${(silent_ms / 1000).toFixed(0)} s` : latest ? percent(latest.thread.blocked) : "—"}
      tone={workerSilent ? "warn" : undefined} />
    <LoadLine name="GPU" testId="load-gpu"
      hint="Share of the time this tab's GPU queue held unfinished work, submit to completion. This tab only; includes completion latency, so it is an upper bound."
      values={gpu} readout={gpu.length ? percent(gpu.at(-1)!) : "—"} />
    <LoadLine name="UI" testId="load-ui"
      hint={`Share of the time the page's own thread could not answer a timer. Hitches are waits over 100 ms in the last ${(uiThread.length * LOAD_SAMPLE_INTERVAL_MS / 1000).toFixed(0)} s.`}
      values={uiThread.map((sample) => sample.blocked)}
      readout={ui ? `${percent(ui.blocked)}${stalls ? ` · ${stalls} hitch${stalls === 1 ? "" : "es"}` : ""}` : "—"}
      tone={stalls ? "warn" : undefined} />
  </div>;
}
