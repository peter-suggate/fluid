"use client";

import type { ReactNode } from "react";

/**
 * One instrument, two pipelines.
 *
 * The advance pipeline and the frame pipeline were two components with the same
 * anatomy — a spine, band collars, a card per stage tapping it, a lamp per card
 * and the cost read at the junction — and the same `.rp-*` grammar. Keeping them
 * apart meant every fix to the diagram was two edits and the two observatories
 * drifted apart between them.
 *
 * So the drawing lives here and the *domains* stay where they belong. This
 * module knows nothing about solvers, passes or method parameters: a
 * caller resolves its own graph, its own measurements and its own prose into the
 * view model below, and this draws it. That is what keeps the merge honest — the
 * differences between the two pipelines are real (a solver stage is not optional
 * and takes no click; a render stage publishes planes),
 * and they survive as *fields a caller may or may not set* rather than as a
 * second copy of the diagram.
 *
 * The timing colocation is the whole design, and it is enforced here: a cost
 * reads at the junction it was measured at, and a band's figure reads on its
 * collar. Nothing is a separate strip of numbers beside a picture, because a
 * column of figures against the pipe can be scanned against each other and a
 * column right-aligned to eighteen different card edges cannot.
 */

/** Where a connector meets a card: the centre of its single head row. */
const ANCHOR_Y = 13;
/**
 * One rail, not two.
 *
 * The diagram used to alternate cards left and right of a centred trunk, which
 * spent half the instrument's width on empty slots — a row never shares its
 * grid line with the row opposite, so the alternation bought no height back for
 * the width it cost. The spine now sits at the left edge with every card hung
 * off it in encode order: the same graph, a third narrower, and the costs land
 * in one right-aligned column against the pipe instead of straddling it. That
 * column is the whole reason the figure reads on the trunk at all — eighteen
 * numbers can only be scanned against each other when they share an edge.
 */
const TRUNK_WIDTH = 64;
const PIPE_X = 48;
/** The right edge of the cost column: figures set back from the pipe. */
const COST_X = 42;

export type PipelineRowState = "on" | "off" | "armed" | "unavailable";

/**
 * Why the figure on the pipe is the kind of number it is.
 *
 * Shared because the diagram draws each kind differently — `shared` carries
 * the ⊂ mark and missing measurements are dimmed — while the
 * *sentence* explaining it is the caller's, since only the caller knows whether
 * "not encoded" means a closed method gate or a withheld render pass.
 */
export type PipelineCostKind =
  | "measured" | "withheld" | "shared" | "idle" | "structural" | "unmeasured";

export interface PipelineCost {
  readonly kind: PipelineCostKind;
  /** Undefined reads as an em dash: a measurement genuinely missing. */
  readonly duration_ms?: number;
  /**
   * Fraction of sampled encodes that ran this row, when below one. The figure is
   * already the expected per-encode cost, so the `~` badge says so rather than
   * letting an intermittent row read as a steady one.
   */
  readonly encodedFraction?: number;
  /** The hover sentence. Composed by the caller, which owns the vocabulary. */
  readonly explanation?: string;
}

/**
 * The row's own switch.
 *
 * `fixed` is a readout: a solver stage the advance always encodes is not
 * optional, and drawing it as a dead switch would invite clicking it. It gets a
 * smaller neutral marker and no `role`, so a screen reader is told a status
 * rather than a control that refuses.
 */
export interface PipelineLamp {
  readonly kind: "switch" | "fixed";
  readonly checked: boolean;
  readonly disabled?: boolean;
  readonly title: string;
  readonly ariaLabel: string;
  readonly onToggle?: () => void;
}

/** A published plane this row wrote, presented instead of the composite. */
export interface PipelineTap {
  readonly label: string;
  readonly title: string;
  readonly active: boolean;
  readonly onToggle: () => void;
}

export interface PipelineRow {
  readonly id: string;
  readonly label: string;
  readonly state: PipelineRowState;
  /** Everything the row *means*, as a hover tip. The only prose on the card. */
  readonly tip: string;
  /** The short factual line under the label. Never a description. */
  readonly chip?: string;
  readonly cost?: PipelineCost;
  readonly lamp: PipelineLamp;
  /** The ◨ affordance beside the name: present this row's primary plane. */
  readonly tap?: PipelineTap;
  /** Its other planes, as a strip under the head. */
  readonly planes?: readonly PipelineTap[];
  /**
   * The row's own controls, drawn under its head.
   *
   * A caller that owns a large cluster folds it itself, behind a named
   * `<details class="rp-tune">`: only the caller knows how many fields there
   * are and what they are collectively *for*, and the name on the drawer is the
   * whole affordance — two related sliders belong on the card, twenty cone
   * budgets belong in a drawer somebody can predict the contents of.
   */
  readonly controls?: ReactNode;
}

export interface PipelineBand {
  readonly id: string;
  readonly label: string;
  /** Σ of the rows below, as the caller measured them. Undefined reads "—". */
  readonly cost_ms?: number;
  /** Share of the whole, when there is an honest denominator for one. */
  readonly share?: number;
  readonly rows: readonly PipelineRow[];
}

export const formatPipelineDuration = (duration_ms: number) =>
  duration_ms >= 10 ? `${duration_ms.toFixed(1)} ms` : `${duration_ms.toFixed(2)} ms`;

/**
 * The cost as it reads on the pipe: a bare number, unit stated once per band.
 *
 * Repeating the unit eighteen times inside a 26px gutter would cost more width
 * than the figure itself, and the whole value of putting the number on the trunk
 * is that a column of them can be scanned against each other.
 *
 * A withheld row reads `0.00`, not an em dash. The dash is reserved for a
 * measurement that is genuinely missing, because a stage the encoder skipped is
 * the one case where the cost is known exactly — and it is the case a reader
 * reaches for the switch to see.
 */
function trunkCostText(cost: PipelineCost): string {
  if (cost.duration_ms === undefined) return "—";
  const figure = cost.duration_ms >= 10 ? cost.duration_ms.toFixed(1) : cost.duration_ms.toFixed(2);
  if (cost.kind === "shared") return `⊂${figure}`;
  return cost.encodedFraction !== undefined ? `~${figure}` : figure;
}

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
function Trunk({ tapped, state, cap, cost }: {
  /** Whether a card hangs off this row and the junction is drawn for it. */
  tapped?: boolean;
  state?: PipelineRowState;
  cap?: "start" | "end";
  cost?: PipelineCost;
}) {
  const dashed = state === "unavailable";
  const live = state === "on" || state === "armed";
  return <div className="rp-trunk-cell">
    <svg className="rp-trunk" viewBox={`0 0 ${TRUNK_WIDTH} 100`} preserveAspectRatio="none"
      aria-hidden="true" focusable="false">
      <line className="rp-pipe" x1={PIPE_X} y1={cap === "start" ? 50 : 0}
        x2={PIPE_X} y2={cap === "end" ? 50 : 100} vectorEffect="non-scaling-stroke" />
    </svg>
    {/* The connector and its junction live in their own unscaled overlay, so a
        tall node cannot stretch a round junction into an ellipse. */}
    <svg className="rp-trunk-marks" width={TRUNK_WIDTH} height={ANCHOR_Y * 2} focusable="false"
      role={cost ? "img" : undefined} aria-hidden={cost ? undefined : true}
      aria-label={cost ? cost.explanation : undefined}>
      {tapped && <line
        className={`rp-link${dashed ? " is-dashed" : ""}${live ? " is-live" : ""}`}
        x1={PIPE_X} y1={ANCHOR_Y} x2={TRUNK_WIDTH} y2={ANCHOR_Y} />}
      {tapped && <circle className={`rp-junction${live ? " is-live" : ""}`}
        cx={PIPE_X} cy={ANCHOR_Y} r={dashed ? 2 : 2.5} />}
      {cap && <circle className="rp-cap" cx={PIPE_X} cy={ANCHOR_Y} r={3.5} />}
      {/* The cost reads at the tap, right-aligned into the gutter the pipe runs
          down. One column, one edge: the frame's shape is legible by scanning
          the figures against each other, which is the only thing putting them
          on the trunk was ever for. */}
      {cost && tapped && <text
        className={`rp-trunk-cost is-${cost.kind}`}
        x={COST_X} y={ANCHOR_Y} textAnchor="end" dominantBaseline="central">
        {cost.explanation && <title>{cost.explanation}</title>}
        {trunkCostText(cost)}
      </text>}
    </svg>
  </div>;
}

function TapButton({ tap, variant }: { tap: PipelineTap; variant: "head" | "plane" }) {
  const className = variant === "head"
    ? `rp-tap${tap.active ? " is-active" : ""}`
    : tap.active ? "active" : "";
  return <button type="button" className={className}
    aria-pressed={tap.active} title={tap.title} onClick={tap.onToggle}>{tap.label}</button>;
}

/** Everything under a row's head: its published planes, then its controls. */
function RowControls({ row }: { row: PipelineRow }) {
  return <div className="rp-node-controls">
    {(row.planes?.length ?? 0) > 0 && <div className="rp-planes" role="group"
      aria-label={`${row.label} published planes`}>
      {row.planes?.map((plane) => <TapButton key={plane.label} tap={plane} variant="plane" />)}
    </div>}
    {row.controls}
  </div>;
}

/**
 * One card.
 *
 * Nothing folds. The head is one line — lamp, name, tap — and every
 * control the row owns sits under it at all times. The cost is deliberately not
 * here: it reads on the trunk at this row's junction.
 */
function RowCard({ row }: { row: PipelineRow }) {
  const { lamp } = row;
  const toggleable = lamp.kind === "switch";
  return <div className={`rp-node is-${row.state}`} data-node={row.id}>
    <div className="rp-node-head">
      <button
        type="button"
        className={`rp-lamp ${toggleable ? "is-toggleable" : "is-fixed"}`}
        role={toggleable ? "switch" : undefined}
        aria-checked={toggleable ? lamp.checked : undefined}
        aria-label={lamp.ariaLabel}
        disabled={lamp.disabled ?? !toggleable}
        title={lamp.title}
        onClick={toggleable ? lamp.onToggle : undefined}
      />
      <strong title={row.tip}>{row.label}</strong>
      {row.tap && <TapButton tap={row.tap} variant="head" />}
    </div>
    {row.chip && <small className="rp-node-chip" title={row.tip}>{row.chip}</small>}
    {(row.controls || (row.planes?.length ?? 0) > 0) && <RowControls row={row} />}
  </div>;
}

/**
 * The whole diagram: a capped spine, one collar per band, one row per card.
 *
 * A row whose card is `undefined` is a caller's collapsed placeholder and is
 * simply not in `rows` — the view model has no notion of a hidden row, because a
 * diagram that decides for itself what to draw is a second place the pipeline is
 * described.
 */
export function PipelineGraph({ bands, testId }: {
  bands: readonly PipelineBand[];
  testId: string;
}) {
  return <div className="rp-graph" data-testid={testId}>
    <div className="rp-row rp-row-cap"><Trunk cap="start" /><div className="rp-slot" /></div>
    {bands.map((band) => <div key={band.id} className="rp-band" role="group" aria-label={band.label}>
      <div className="rp-row rp-row-collar">
        <Trunk />
        <div className="rp-collar">
          <h3>{band.label}</h3>
          <i aria-hidden="true" />
          <output>{band.cost_ms !== undefined ? formatPipelineDuration(band.cost_ms) : "—"}</output>
          {band.share !== undefined && <small>{(band.share * 100).toFixed(1)}%</small>}
        </div>
      </div>
      {band.rows.map((row) => <div key={row.id} className="rp-row rp-row-node">
        <Trunk tapped state={row.state} cost={row.cost} />
        <RowCard row={row} />
      </div>)}
    </div>)}
    <div className="rp-row rp-row-cap"><Trunk cap="end" /><div className="rp-slot" /></div>
  </div>;
}
