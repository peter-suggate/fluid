"use client";

import type { ReactNode } from "react";
import {
  fluidPipelineTipText,
  measureFluidPipelineBand,
  measureFluidPipelineStage,
  type FluidPhaseCost,
  type FluidPipelineContext,
  type FluidPipelineGraph,
  type FluidPipelineMeasurement,
  type FluidPipelineStage,
  type FluidPipelineStageState,
} from "@/lib/fluid-pipeline";

/**
 * The simulation sibling of `RenderPipeline`: same trunk-and-cards anatomy and
 * the same `.rp-*` grammar, so the two observatories read as one instrument.
 *
 * The differences are the honest ones. Lamps are readouts — a solver stage is
 * not optional, so the lamp reports what the advance encodes and takes no
 * click. There are no taps and no ablation deltas. And because the physics
 * trace is an exact boundary-chain partition, the figures on this trunk sum to
 * the advance rather than to a pass subset.
 */

const ANCHOR_Y = 12;
const TRUNK_WIDTH = 76;
const TRUNK_X = TRUNK_WIDTH / 2;

function Trunk({ side, state, cap, cost, costTitle }: {
  side?: "left" | "right";
  state?: FluidPipelineStageState;
  cap?: "start" | "end";
  cost?: FluidPipelineMeasurement;
  costTitle?: string;
}) {
  const dashed = state === "unavailable";
  const live = state === "on";
  return <div className="rp-trunk-cell">
    <svg className="rp-trunk" viewBox={`0 0 ${TRUNK_WIDTH} 100`} preserveAspectRatio="none"
      aria-hidden="true" focusable="false">
      <line className="rp-pipe" x1={TRUNK_X} y1={cap === "start" ? 50 : 0}
        x2={TRUNK_X} y2={cap === "end" ? 50 : 100} vectorEffect="non-scaling-stroke" />
    </svg>
    <svg className="rp-trunk-marks" width={TRUNK_WIDTH} height={ANCHOR_Y * 2} focusable="false"
      role={cost ? "img" : undefined} aria-hidden={cost ? undefined : true}
      aria-label={cost ? costTitle : undefined}>
      {side && <line
        className={`rp-link${dashed ? " is-dashed" : ""}${live ? " is-live" : ""}`}
        x1={side === "left" ? 0 : TRUNK_X} y1={ANCHOR_Y}
        x2={side === "left" ? TRUNK_X : TRUNK_WIDTH} y2={ANCHOR_Y} />}
      {side && <circle className={`rp-junction${live ? " is-live" : ""}`} cx={TRUNK_X} cy={ANCHOR_Y} r={dashed ? 2 : 3} />}
      {cap && <circle className="rp-cap" cx={TRUNK_X} cy={ANCHOR_Y} r={4} />}
      {cost && side && <text
        className={`rp-trunk-cost is-${cost.kind}`}
        x={side === "left" ? TRUNK_X + 6 : TRUNK_X - 6} y={ANCHOR_Y}
        textAnchor={side === "left" ? "start" : "end"} dominantBaseline="central">
        {costTitle && <title>{costTitle}</title>}
        {trunkCostText(cost)}
      </text>}
    </svg>
  </div>;
}

const formatDuration = (duration_ms: number) =>
  duration_ms >= 10 ? `${duration_ms.toFixed(1)} ms` : `${duration_ms.toFixed(2)} ms`;

/**
 * The cost as it reads on the pipe: a bare number, unit stated once per band.
 * `~` marks an intermittent stage — the figure is the expected per-advance
 * cost, diluted by how often the sampled advances encoded it.
 */
function trunkCostText(cost: FluidPipelineMeasurement): string {
  if (cost.duration_ms === undefined) return "—";
  const figure = cost.duration_ms >= 10 ? cost.duration_ms.toFixed(1) : cost.duration_ms.toFixed(2);
  if (cost.kind === "shared") return `⊂${figure}`;
  return cost.encodedFraction !== undefined ? `~${figure}` : figure;
}

/** Why the figure on the pipe is the kind of number it is. */
function costExplanation(cost: FluidPipelineMeasurement): string {
  switch (cost.kind) {
    case "withheld":
      return "0 ms — not encoded this advance. The stage's gate is closed, so this is a measured zero rather than a missing measurement.";
    case "shared":
      return `Not a seam of its own: this work runs inside ${cost.insideStage?.replace(/-/g, " ").toUpperCase()}, and ⊂ marks the figure as that stage's.`;
    case "idle":
      return "0 ms — the advance partition arrived, and this gated stage encoded nothing in the sampled advances.";
    case "structural":
      return "A decision, not a dispatch. It spends no advance time either way.";
    case "unmeasured":
      return "No phase in the advance partition names this stage yet — either no trace has arrived, or the fallback queue-wall observation is too coarse to split the advance.";
    default:
      return `${formatDuration(cost.duration_ms ?? 0)} GPU execution time between this stage's trace seams, averaged across the trace window.${cost.encodedFraction !== undefined
        ? `\n\nEncoded in ${Math.round(cost.encodedFraction * 100)}% of sampled advances; the figure is the expected cost per advance, not the per-encode mean.`
        : ""}`;
  }
}

function StageCard({ stage, state, chip, tip, controls, onToggle }: {
  stage: FluidPipelineStage;
  state: FluidPipelineStageState;
  chip: string;
  tip: string;
  controls?: ReactNode;
  onToggle?: (checked: boolean) => void;
}) {
  const toggleable = Boolean(stage.toggle);
  const statusLabel = toggleable
    ? state === "on" ? "encoding" : state === "off" ? "configured off" : "not applicable to this scene"
    : "always encoded";
  const lampTitle = toggleable
    ? stage.toggle?.hint ? `${stage.toggle.hint}\n\n${tip}` : tip
    : `Always encoded · fixed pipeline stage\n\n${tip}`;

  return <div className={`rp-node is-${state}`} data-node={stage.id}>
    {/* Optional stages expose the same compact lamp switch as the render
        pipeline; fixed stages use a smaller, neutral status marker. */}
    <div className="rp-node-head">
      <button type="button" className={`rp-lamp ${toggleable ? "is-toggleable" : "is-fixed"}`}
        disabled={!toggleable || state === "unavailable"}
        role={toggleable ? "switch" : undefined}
        aria-checked={toggleable ? state === "on" : undefined}
        onClick={toggleable ? () => onToggle?.(state !== "on") : undefined}
        aria-label={`${stage.label}: ${statusLabel}`}
        title={lampTitle} />
      <strong title={tip}>{stage.label}</strong>
    </div>
    {chip && <small className="rp-node-chip" title={tip}>{chip}</small>}
    {controls && <div className="rp-node-controls">{controls}</div>}
  </div>;
}

export interface FluidPipelineProps {
  graph: FluidPipelineGraph;
  context: FluidPipelineContext;
  costs: ReadonlyMap<string, FluidPhaseCost>;
  total_ms: number;
  /** True when an averaged physics partition has arrived. */
  measured: boolean;
  /** Declarative stage controls, rendered by the panel and slotted by stage id. */
  controls: Readonly<Record<string, ReactNode>>;
  onToggleStage?: (stageId: string, checked: boolean) => void;
}

export function FluidPipeline({ graph, context, costs, total_ms, measured, controls, onToggleStage }: FluidPipelineProps) {
  return <div className="rp-graph" data-testid="fluid-pipeline">
    <div className="rp-row rp-row-cap">
      <div className="rp-slot" /><Trunk cap="start" /><div className="rp-slot" />
    </div>
    {graph.bands.map((band) => {
      const bandCost = measureFluidPipelineBand(band.id, graph, costs, total_ms);
      const stages = graph.stages.filter((stage) => stage.band === band.id);
      return <div key={band.id} className="rp-band" role="group" aria-label={band.label}>
        <div className="rp-row rp-row-collar">
          <div className="rp-slot rp-collar-name"><h3>{band.label}</h3></div>
          <Trunk />
          <div className="rp-slot rp-collar-cost">
            <output>{measured && bandCost.duration_ms !== undefined
              ? formatDuration(bandCost.duration_ms) : "—"}</output>
            {measured && total_ms > 0 && bandCost.duration_ms !== undefined
              && <small>{(bandCost.share * 100).toFixed(1)}%</small>}
          </div>
        </div>
        {stages.map((stage) => {
          const state = stage.state(context);
          const chip = stage.chip(context);
          const cost = measured
            ? measureFluidPipelineStage(stage, graph.stages, costs, total_ms, state)
            : { kind: "unmeasured" as const, share: 0 };
          const card = <StageCard
            stage={stage} state={state} chip={chip}
            tip={fluidPipelineTipText(stage, chip)}
            controls={controls[stage.id]}
            onToggle={(checked) => onToggleStage?.(stage.id, checked)}
          />;
          return <div key={stage.id} className={`rp-row rp-row-node is-${stage.side}`}>
            {stage.side === "left" ? card : <div className="rp-slot" />}
            <Trunk side={stage.side} state={state} cost={cost}
              costTitle={`${stage.label}\n\n${costExplanation(cost)}`} />
            {stage.side === "right" ? card : <div className="rp-slot" />}
          </div>;
        })}
      </div>;
    })}
    <div className="rp-row rp-row-cap">
      <div className="rp-slot" />
      <Trunk cap="end" /><div className="rp-slot" />
    </div>
  </div>;
}
