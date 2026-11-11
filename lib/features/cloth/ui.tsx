"use client";

import { useEffect, useState } from "react";
import { formatPipelineDuration, pipelineCostFigure } from "../../../components/PipelineCost";
import { RailChoice, RailLine, RailRange, RailReadout, RailRule } from "../../../components/rail";
import { Stepper } from "../../../components/ui";
import { CLOTH_NEWTON_SCENE_DEFAULTS } from "../../cloth/newton-defaults";
import { CLOTH_STAGES, type ClothPerformanceSample, type ClothStageId } from "../../cloth/stage-timing";
import type { ClothDocument, ClothNewtonSettings } from "../../cloth/types";
import { useSession } from "../../core/session/session-context";
import { usePerformanceInstrumentationStore } from "../../core/stores/performance-instrumentation-store";
import { clothFeature } from "./definition";

/** Sampled frames the stage figures average over (about two seconds at the sampling cadence). */
const WINDOW = 8;

interface Averaged {
  readonly samples: number;
  readonly steps: number;
  readonly span_ms: number;
  readonly sum_ms: number;
  readonly stages: ReadonlyMap<ClothStageId, { duration_ms: number; passes: number }>;
}

/**
 * Turn measurement on while the card is open, and hand it back as found: the
 * same guard the advance pipeline uses, so an explicit choice made elsewhere
 * (the Performance card) is never overridden.
 */
function useMeasurement() {
  useEffect(() => {
    const store = usePerformanceInstrumentationStore.getState();
    if (store.mode !== "off") return;
    store.setMode("timeline");
    return () => {
      const current = usePerformanceInstrumentationStore.getState();
      if (current.mode === "timeline") current.setMode("off");
    };
  }, []);
}

/** The newest samples, averaged; a sample is counted once however often the info is republished. */
function useAveraged(sample: ClothPerformanceSample | undefined): Averaged | undefined {
  const [samples, setSamples] = useState<readonly ClothPerformanceSample[]>([]);
  if (sample && !sample.unavailable && samples.at(-1)?.sampleId !== sample.sampleId) {
    // A different step count is a different frame shape (catch-up beside a fluid): start over rather than mix them.
    setSamples([...(samples.length && samples[0]!.steps !== sample.steps ? [] : samples), sample].slice(-WINDOW));
  }
  if (!samples.length) return undefined;
  const stages = new Map<ClothStageId, { duration_ms: number; passes: number }>();
  for (const s of samples) for (const stage of s.stages) {
    const at = stages.get(stage.id) ?? { duration_ms: 0, passes: 0 };
    at.duration_ms += stage.duration_ms / samples.length; at.passes = Math.max(at.passes, stage.passes);
    stages.set(stage.id, at);
  }
  const mean = (pick: (s: ClothPerformanceSample) => number) => samples.reduce((a, s) => a + pick(s), 0) / samples.length;
  return { samples: samples.length, steps: samples.at(-1)!.steps, span_ms: mean(s => s.span_ms), sum_ms: mean(s => s.sum_ms), stages };
}

/** What a cloth frame costs: its GPU span, then a line per stage. */
export function ClothCostLines() {
  useMeasurement();
  const session = useSession();
  const sample = session.diagnostics((state) => state.gpuInfo?.clothPerformance);
  const running = session.runtime((state) => state.runState === "running");
  const averaged = useAveraged(sample);
  const control = clothFeature.controls[0];
  const frame = averaged
    ? `${formatPipelineDuration(averaged.span_ms)}${averaged.steps > 1 ? ` · ${averaged.steps} steps` : ""}`
    : sample?.unavailable ? "unavailable" : running ? "sampling…" : "run to measure";
  const hint = sample?.unavailable
    ?? (averaged
      ? `GPU time from the first stage pass's start to the last one's end, mean of ${averaged.samples} sampled frames. `
        + `Stage figures are each stage's own pass time; they sum to ${formatPipelineDuration(averaged.sum_ms)}`
        + (averaged.sum_ms > averaged.span_ms * 1.05 ? " (passes overlapped)." : ".")
      : control.hint);
  return <>
    <RailReadout label={control.label} value={frame} hint={hint} testId="cloth-frame-cost" />
    {averaged && <div className="rail-cost" data-testid="cloth-pipeline">
      {CLOTH_STAGES.filter(stage => averaged.stages.has(stage.id)).map(stage => {
        const { duration_ms, passes } = averaged.stages.get(stage.id)!;
        const share = averaged.sum_ms > 0 ? duration_ms / averaged.sum_ms : 0;
        return <div key={stage.id} className="rail-cost-band">
          <div className="rail-cost-line rail-cost-head" title={`${stage.tip} ${passes} pass${passes === 1 ? "" : "es"} in the sampled frame · ${(share * 100).toFixed(1)}% of the stage sum.`}>
            <span className="rail-cost-name">{stage.label}</span>
            <span className="rail-cost-bar" aria-hidden><i style={{ width: `${Math.min(1, share) * 100}%` }} /></span>
            <output className="rail-cost-figure">{pipelineCostFigure({ kind: "measured", duration_ms })}</output>
          </div>
        </div>;
      })}
    </div>}
  </>;
}

const formatResidual = (value: number) => Number.isFinite(value) ? value.toExponential(1) : "—";

/** How the last sampled step converged, per document. */
export function ClothConvergenceLines() {
  const session = useSession();
  const documents = session.diagnostics((state) => state.gpuInfo?.clothPerformance?.documents);
  if (!documents?.length) return null;
  return <>
    <RailRule label="Last step" />
    {documents.map(document => {
      const c = document.convergence;
      const prefix = documents.length > 1 ? `${document.id} · ` : "";
      if (!c) return <RailReadout key={document.id} label={`${prefix}Newton`} value="—" hint="The Newton control record could not be read." />;
      return <div key={document.id}>
        <RailReadout label={`${prefix}Newton`} value={`${c.newtonIterations} / ${document.newtonIterations}`}
          hint="Newton iterations the last sampled step took, of its cap. At the cap, the step stopped on the budget rather than the residual test." />
        <RailReadout label="PCG" value={`${c.pcgIterations}`}
          hint={`PCG iterations over the step's Newton iterations (cap ${document.pcgIterations} per Newton iteration).`} />
        <RailReadout label="Residual" value={`${formatResidual(c.residual)}${document.residualTolerance !== undefined ? ` / ${document.residualTolerance.toExponential(0)}` : ""}`}
          hint="|g| / |m g| over free vertices at the step's last Newton iterate, against the residual tolerance." />
        <RailReadout label="Step length" value={Number.isFinite(c.alpha) ? c.alpha.toFixed(3) : "—"}
          hint="The last accepted line-search step length α (1 is the full Newton step)." />
      </div>;
    })}
  </>;
}

/**
 * The PCG cap spans three orders of magnitude, so it is a slider. The readout
 * follows the drag; the scene only takes the value it lands on, since each
 * change rebuilds the cloth (from its live pose).
 */
function PcgIterationsRange({ value, onChange }: { value: number; onChange: (value: number) => void }) {
  const [draft, setDraft] = useState<number>();
  return <RailRange label="PCG its" hint="PCG iterations per Newton iteration (cap)." ariaLabel="Cloth PCG iterations"
    value={draft ?? value} min={10} max={1000} step={10} digits={0}
    onInput={setDraft} onChange={next => { setDraft(undefined); if (next !== value) onChange(Math.round(next)); }}
    onGestureEnd={() => setDraft(undefined)} />;
}

/** A tolerance is a log-scale knob: tightened and loosened by decades, printed as one. */
function ToleranceStepper({ value, min, max, ariaLabel, onChange }: {
  value: number; min: number; max: number; ariaLabel: string; onChange: (value: number) => void;
}) {
  // Snap to the decade so repeated steps never drift into 9.999e-5.
  const decade = (v: number) => Number(v.toExponential(2));
  return <Stepper value={value} onChange={v => onChange(decade(v))} min={min} max={max} factor={10}
    readout={value.toExponential(0)} ariaLabel={ariaLabel}
    decreaseHint="Tighten tenfold" increaseHint="Loosen tenfold" />;
}

/** The Newton-PCG knobs of every cloth document, and what each document is. */
export function ClothSettingsLines() {
  const session = useSession();
  const cloth = session.scene((state) => state.scene.cloth);
  const patchScene = session.scene((state) => state.patchScene);
  const facts = session.diagnostics((state) => state.gpuInfo?.clothPerformance?.documents);
  if (!cloth?.length) return null;
  const patchNewton = (document: ClothDocument, patch: Partial<ClothNewtonSettings>) => {
    const newton = { ...CLOTH_NEWTON_SCENE_DEFAULTS, ...document.solver?.newton, ...patch };
    patchScene({ cloth: cloth.map(d => d.id === document.id ? { ...d, solver: { ...d.solver, newton } } : d) });
  };
  return <>
    {cloth.map(document => {
      const newton = { ...CLOTH_NEWTON_SCENE_DEFAULTS, ...document.solver?.newton };
      const fact = facts?.find(f => f.id === document.id);
      const name = cloth.length > 1 ? `Solver · ${document.id}` : "Solver";
      return <div key={document.id} data-testid={`cloth-settings-${document.id}`}>
        <RailRule label={name} />
        <RailLine label="Newton its" hint="Newton iterations per step (cap). Each one re-assembles, solves and line-searches.">
          <Stepper value={newton.iterations} min={1} max={32} step={1} digits={0} ariaLabel="Cloth Newton iterations"
            decreaseHint="One Newton iteration fewer" increaseHint="One Newton iteration more"
            onChange={value => patchNewton(document, { iterations: Math.round(value) })} />
        </RailLine>
        <PcgIterationsRange value={newton.pcgIterations} onChange={value => patchNewton(document, { pcgIterations: value })} />
        <RailLine label="PCG tol" hint="PCG stops when r·z ≤ tol² r₀·z₀. Steps by a factor of ten.">
          <ToleranceStepper value={newton.pcgTolerance ?? 1e-3} min={1e-8} max={1e-1} ariaLabel="Cloth PCG tolerance"
            onChange={value => patchNewton(document, { pcgTolerance: value })} />
        </RailLine>
        <RailLine label="Residual tol" hint="Newton stops when |g| ≤ tol |m g| over free vertices. Steps by a factor of ten.">
          <ToleranceStepper value={newton.residualTolerance ?? 1e-4} min={1e-9} max={1e-1} ariaLabel="Cloth residual tolerance"
            onChange={value => patchNewton(document, { residualTolerance: value })} />
        </RailLine>
        <RailChoice<"off" | "hybrid"> label="Trust region" value={newton.trustRegion ?? "off"}
          hint="'hybrid': the line search until two consecutive full steps, then Steihaug-Toint trust-region iterations for the rest of the step."
          options={[{ value: "off", label: "Off" }, { value: "hybrid", label: "Hybrid" }]}
          onChange={trustRegion => patchNewton(document, { trustRegion })} />
        <RailReadout label="Integrator" value={(document.solver?.integrator ?? "bdf2").toUpperCase()} />
        {fact && <RailReadout label="Mesh" value={`${fact.vertices.toLocaleString()} v · ${fact.triangles.toLocaleString()} t`}
          hint={`${fact.vertices} vertices, ${fact.triangles} triangles, ${fact.edges} edges; contact radius ${(fact.contactRadius_m * 1000).toFixed(2)} mm.`} />}
      </div>;
    })}
  </>;
}
