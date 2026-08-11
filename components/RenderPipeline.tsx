"use client";

import type { ReactNode } from "react";
import {
  measureRenderPipelineBand,
  measureRenderPipelineNode,
  RENDER_PIPELINE_BANDS,
  RENDER_PIPELINE_COLLAPSE_GROUPS,
  RENDER_PIPELINE_NODES,
  renderPipelineTipText,
  type RenderPipelineContext,
  type RenderPipelineMeasurement,
  type RenderPipelineNode,
  type RenderPipelineNodeState,
} from "@/lib/render-pipeline-graph";
import { SVO_RENDER_STAGE_DEFINITIONS, type SvoRenderStageView } from "@/lib/svo-render-diagnostics";

/** Where a connector meets a card: the centre of its single head row. */
const ANCHOR_Y = 12;
// Wide enough that a five-character figure clears the pipe on either side —
// the trunk gutter carries the frame's costs now, not just the pipe.
const TRUNK_WIDTH = 76;
const TRUNK_X = TRUNK_WIDTH / 2;

/**
 * One row's worth of pipe.
 *
 * Each row draws its own segment, so the segments stack into a continuous pipe
 * without anybody measuring anything — an absolutely positioned pipe over the
 * whole graph would need live row geometry and would be one frame wrong on
 * every layout change.
 *
 * The wrapper is not decoration. An `<svg height="100%">` is a replaced element:
 * with no definite height on its containing block the percentage cannot resolve,
 * so it falls back to the 150px intrinsic default and every row inherits that
 * height. The wrapper is the definite box — stretched by the grid to the row's
 * real height — and the pipe fills it absolutely.
 */
function Trunk({ side, state, cap, cost, costTitle }: {
  side?: "left" | "right";
  state?: RenderPipelineNodeState;
  cap?: "start" | "end";
  cost?: RenderPipelineMeasurement;
  costTitle?: string;
}) {
  const dashed = state === "unavailable";
  const live = state === "on" || state === "armed";
  return <div className="rp-trunk-cell">
    <svg className="rp-trunk" viewBox={`0 0 ${TRUNK_WIDTH} 100`} preserveAspectRatio="none"
      aria-hidden="true" focusable="false">
      <line className="rp-pipe" x1={TRUNK_X} y1={cap === "start" ? 50 : 0}
        x2={TRUNK_X} y2={cap === "end" ? 50 : 100} vectorEffect="non-scaling-stroke" />
    </svg>
    {/* The connector and its junction live in their own unscaled overlay, so a
        tall node cannot stretch a round junction into an ellipse. */}
    <svg className="rp-trunk-marks" width={TRUNK_WIDTH} height={ANCHOR_Y * 2} focusable="false"
      role={cost ? "img" : undefined} aria-hidden={cost ? undefined : true}
      aria-label={cost ? costTitle : undefined}>
      {side && <line
        className={`rp-link${dashed ? " is-dashed" : ""}${live ? " is-live" : ""}`}
        x1={side === "left" ? 0 : TRUNK_X} y1={ANCHOR_Y}
        x2={side === "left" ? TRUNK_X : TRUNK_WIDTH} y2={ANCHOR_Y} />}
      {side && <circle className={`rp-junction${live ? " is-live" : ""}`} cx={TRUNK_X} cy={ANCHOR_Y} r={dashed ? 2 : 3} />}
      {cap && <circle className="rp-cap" cx={TRUNK_X} cy={ANCHOR_Y} r={4} />}
      {/* The cost reads at the tap. It sits just past the junction, on the side
          of the trunk the node is not on, so the number is against the pipe it
          is a share of and the row it belongs to is unambiguous. */}
      {cost && side && <text
        className={`rp-trunk-cost is-${cost.kind}`}
        x={side === "left" ? TRUNK_X + 6 : TRUNK_X - 6} y={ANCHOR_Y}
        textAnchor={side === "left" ? "start" : "end"} dominantBaseline="central">
        {costTitle && <title>{costTitle}</title>}
        {trunkCostText(cost)}
        {/* A wall figure is a different basis than the compute column around
            it — the tag keeps a fence-derived number from being scanned as a
            per-pass timestamp. */}
        {cost.kind === "wall" && <tspan className="rp-trunk-wall-tag" dx={3}>wall</tspan>}
      </text>}
    </svg>
  </div>;
}

const formatDuration = (duration_ms: number) =>
  duration_ms >= 10 ? `${duration_ms.toFixed(1)} ms` : `${duration_ms.toFixed(2)} ms`;

/**
 * The cost as it reads on the pipe: a bare number, no unit.
 *
 * The unit is stated once per band on the collar. Repeating it eighteen times
 * inside a 26px gutter would cost more width than the figure itself, and the
 * whole value of putting the number on the trunk is that a column of them can
 * be scanned against each other.
 *
 * A withheld pass reads `0.00`, not an em dash. The dash is reserved for a
 * measurement that is genuinely missing, because a stage the encoder skipped is
 * the one case where the cost is known exactly — and it is the case a user
 * reaches for the switch to see.
 */
function trunkCostText(cost: RenderPipelineMeasurement): string {
  if (cost.duration_ms === undefined) return "—";
  const figure = cost.duration_ms >= 10 ? cost.duration_ms.toFixed(1) : cost.duration_ms.toFixed(2);
  return cost.kind === "shared" ? `⊂${figure}` : figure;
}

/** Why the figure on the pipe is the kind of number it is. */
function costExplanation(cost: RenderPipelineMeasurement, exhaustive: boolean): string {
  const measuredSuffix = "GPU execution time, summed over the passes this node owns and averaged across the trace window.";
  switch (cost.kind) {
    case "withheld":
      return "0 ms — not encoded this frame. The switch withholds the pass, so this is a measured zero rather than a missing measurement.";
    case "shared":
      return `Not a dispatch of its own: this work runs inside ${cost.insideNode?.replace(/-/g, " ").toUpperCase()}, and ⊂ marks the figure as that pass's.`;
    case "idle":
      return "0 ms — the detailed GPU partition arrived, and this gated pass did not encode in the sampled frame.";
    case "structural":
      return "A gate, not a pass. It spends no frame time either way — its worth shows up as the row it lets the frame skip going to zero.";
    case "wall":
      return `${formatDuration(cost.duration_ms ?? 0)} wall-clock, derived: this is the band's only row without a trusted per-pass timestamp, so the band's fence-partitioned wall minus its measured compute lands here. Render passes included; a different basis than the compute figures around it.`;
    case "unmeasured":
      return exhaustive
        ? "No phase in this frame's partition names this node: either no trace has arrived yet, or this stage encoded no pass."
        : "No trustworthy per-pass cost is available. The stage either encoded no pass, or is a Metal render pass whose timestamp pair is a whole tiler window rather than this stage's execution cost.";
    default:
      return `${formatDuration(cost.duration_ms ?? 0)} ${measuredSuffix}`;
  }
}

interface NodeCardProps {
  node: RenderPipelineNode;
  state: RenderPipelineNodeState;
  chip: string;
  /** Frame milliseconds observed to move when this node was last switched. */
  delta_ms?: number;
  stageView: SvoRenderStageView;
  onToggle: () => void;
  onTap: (view: SvoRenderStageView) => void;
  controls?: ReactNode;
}

function NodeCard({ node, state, chip, delta_ms, stageView, onToggle, onTap, controls }: NodeCardProps) {
  const tip = renderPipelineTipText(node, chip);
  const unavailable = state === "unavailable";
  const active = node.taps.find((view) => view === stageView);
  const primaryTap = node.taps[0];
  return <div className={`rp-node is-${state}`} data-node={node.id}>
    {/* Nothing folds. The head is one line — lamp, name, saving, tap — and every
        control the node owns sits under it at all times. The cost is not here:
        it reads on the trunk at this node's junction, so the frame's costs form
        one scannable column against the pipe instead of eighteen figures
        right-aligned to eighteen different card edges. What a node *means* is in
        the hover tip, which is the only place prose lives here. */}
    <div className="rp-node-head">
      <button
        type="button"
        className="rp-lamp"
        role="switch"
        aria-checked={state !== "off"}
        aria-label={node.label}
        disabled={!node.toggleable || unavailable}
        title={tip}
        onClick={onToggle}
      />
      <strong title={tip}>{node.label}</strong>
      {/* The whole point of a switch: what the frame did when this last moved.
          Measured across the toggle rather than derived from the node's own
          phases, so it is the only figure that can price a node whose work is a
          term inside somebody else's pass. */}
      {delta_ms !== undefined && <output className="rp-node-delta" title={
        `Switching this ${state === "off" ? "off" : "on"} moved the frame total by ${Math.abs(delta_ms).toFixed(2)} ms.\n\n`
        + "Measured: the averaged frame total under each setting, differenced. It includes anything downstream that got cheaper or dearer with it, which is why it can differ from the row's own cost."
      }>{delta_ms >= 0 ? "−" : "+"}{Math.abs(delta_ms).toFixed(2)}</output>}
      {primaryTap && !unavailable && <button type="button"
        className={`rp-tap${active ? " is-active" : ""}`} aria-pressed={Boolean(active)}
        title={`Present a plane this pass wrote instead of the composite.\n\n${tip}`}
        onClick={() => onTap(active ? "off" : primaryTap)}>◨</button>}
    </div>
    {(controls || node.taps.length > 1) && <div className="rp-node-controls">
      {node.taps.length > 1 && <div className="rp-planes" role="group" aria-label={`${node.label} published planes`}>
        {node.taps.map((view) => {
          const definition = SVO_RENDER_STAGE_DEFINITIONS[view];
          return <button key={view} type="button" aria-pressed={stageView === view}
            className={stageView === view ? "active" : ""}
            title={`${definition.plane}\n\n${definition.description}`}
            onClick={() => onTap(stageView === view ? "off" : view)}>{definition.label}</button>;
        })}
      </div>}
      {controls}
    </div>}
  </div>;
}

export interface RenderPipelineProps {
  context: RenderPipelineContext;
  durations: ReadonlyMap<string, number>;
  total_ms: number;
  /** True when a hardware per-pass partition has arrived. */
  measured: boolean;
  /** False for trusted compute-pass sums, which intentionally omit render-pass tiler windows. */
  exhaustive: boolean;
  stageView: SvoRenderStageView;
  onToggle: (id: string) => void;
  onTap: (view: SvoRenderStageView) => void;
  controls: Readonly<Record<string, ReactNode>>;
  /** Per node: frame milliseconds saved by its current setting, when both have been seen. */
  deltas: ReadonlyMap<string, number>;
  /**
   * Per band: real wall-clock cost from fence-partitioned sampling frames.
   * The collar shows it beside the Σcompute figure, tagged as wall — it is the
   * only measurement that prices this band's render passes, and the two bases
   * never share a denominator.
   */
  bandWalls?: ReadonlyMap<string, number>;
}

export function RenderPipeline({
  context, durations, total_ms, measured, exhaustive, stageView, onToggle, onTap, controls, deltas, bandWalls,
}: RenderPipelineProps) {
  return <div className="rp-graph" data-testid="render-pipeline">
    <div className="rp-row rp-row-cap">
      <div className="rp-slot" /><Trunk cap="start" /><div className="rp-slot" />
    </div>
    {RENDER_PIPELINE_BANDS.map((band) => {
      const bandCost = measureRenderPipelineBand(band.id, durations, total_ms, exhaustive);
      const bandWall_ms = bandWalls?.get(band.id);
      const nodes = RENDER_PIPELINE_NODES.filter((node) => node.band === band.id);
      // The wall reads at a row's junction, not on the collar, whenever it can
      // be attributed: when exactly one reachable row in the band has no
      // trusted per-pass cost, that row IS what the wall measured beyond the
      // band's compute, so the junction that would read "—" carries the real
      // number. The collar keeps the wall only when the band has several
      // unmeasured rows and the split is unknowable.
      const entries = nodes.map((node) => {
        const state = node.state(context);
        const cost: RenderPipelineMeasurement = measured
          ? measureRenderPipelineNode(node, durations, total_ms, state, exhaustive)
          : { kind: "unmeasured", share: 0 };
        return { node, state, cost };
      });
      const unmeasuredRows = entries.filter((entry) => entry.cost.kind === "unmeasured" && entry.state !== "unavailable");
      const wallRow = bandWall_ms !== undefined && unmeasuredRows.length === 1 ? unmeasuredRows[0] : undefined;
      if (wallRow && bandWall_ms !== undefined) {
        const compute_ms = entries.reduce((sum, entry) =>
          sum + (entry.cost.kind === "measured" ? entry.cost.duration_ms ?? 0 : 0), 0);
        wallRow.cost = { kind: "wall", duration_ms: Math.max(0, bandWall_ms - compute_ms), share: 0 };
      }
      return <div key={band.id} className="rp-band" role="group" aria-label={band.label}>
        <div className="rp-row rp-row-collar">
          <div className="rp-slot rp-collar-name"><h3>{band.label}</h3></div>
          <Trunk />
          <div className="rp-slot rp-collar-cost">
            {/* Two figures, two bases, one junction. The wall is the band's
                real cost from a fence-partitioned sampling frame — the only
                measurement that includes this band's render passes — and the
                Σcompute figure is what the rows below it add up to. They are
                labelled apart and never share a denominator. */}
            {bandWall_ms !== undefined && !wallRow && <output className="rp-collar-wall" title={
              `${formatDuration(bandWall_ms)} wall-clock for this band's whole encode segment, render passes included.\n\nMeasured by splitting one frame in sixteen at band boundaries into separate submits and timing each queue fence. Sampling frames are excluded from the frame mean.`
            }>{formatDuration(bandWall_ms)}<small>wall</small></output>}
            <output>{measured && bandCost.duration_ms !== undefined
              ? formatDuration(bandCost.duration_ms) : "—"}</output>
            {measured && total_ms > 0 && bandCost.duration_ms !== undefined
              && <small>{(bandCost.share * 100).toFixed(1)}%</small>}
          </div>
        </div>
        {entries.map(({ node, state, cost }) => {
          // A run of rows on an unreachable arm reads as one collapsed row:
          // repeating `unavailable` per tier is diagram space spent on a path
          // the frame cannot take. The first member renders the placeholder;
          // the rest render nothing while every member is unreachable.
          if (node.collapseGroup) {
            const members = nodes.filter((other) => other.collapseGroup === node.collapseGroup);
            if (members.every((member) => member.state(context) === "unavailable")) {
              if (members.indexOf(node) !== 0) return undefined;
              const group = RENDER_PIPELINE_COLLAPSE_GROUPS[node.collapseGroup];
              const tip = `${group.label} · ${group.chip}\n\n${group.summary}`;
              return <div key={node.collapseGroup} className="rp-row rp-row-node is-left">
                <div className="rp-node is-unavailable" data-node={node.collapseGroup}>
                  <div className="rp-node-head">
                    <button type="button" className="rp-lamp" role="switch" aria-checked
                      aria-label={group.label} disabled title={tip} />
                    <strong title={tip}>{group.label}</strong>
                  </div>
                </div>
                <Trunk side="left" state="unavailable" />
                <div className="rp-slot" />
              </div>;
            }
          }
          const chip = node.chip(context);
          const card = <NodeCard
            node={node} state={state} chip={chip}
            delta_ms={measured ? deltas.get(node.id) : undefined}
            stageView={stageView}
            onToggle={() => onToggle(node.id)}
            onTap={onTap}
            controls={controls[node.id]}
          />;
          return <div key={node.id} className={`rp-row rp-row-node is-${node.side}`}>
            {node.side === "left" ? card : <div className="rp-slot" />}
            <Trunk side={node.side} state={state} cost={cost}
              costTitle={`${node.label}\n\n${costExplanation(cost, exhaustive)}`} />
            {node.side === "right" ? card : <div className="rp-slot" />}
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
