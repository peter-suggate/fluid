"use client";

import { useState, type ReactNode } from "react";
import { Switch } from "./ui";

/**
 * A pipeline's cost, in a rail card: one line per band, in encode order.
 *
 * Disclosed a level at a time — a band opens to its stages, a stage to its
 * breakdown and settings — so a closed card is five figures and the reader
 * opens only the one they are chasing.
 * Stages that cannot run in this scene stand behind a count; a stage with a
 * notice marks its band, so nothing broken hides behind a fold.
 */

export type PipelineRowState = "on" | "off" | "armed" | "unavailable";

/**
 * Why a figure is the kind of number it is. The sentence explaining it is the
 * caller's, since only the caller knows whether "not encoded" means a closed
 * method gate or a withheld render pass.
 */
export type PipelineCostKind =
  | "measured" | "withheld" | "shared" | "idle" | "structural" | "unmeasured";

export interface PipelineCostFigure {
  readonly kind: PipelineCostKind;
  /** Undefined reads as an em dash: a measurement genuinely missing. */
  readonly duration_ms?: number;
  /** Fraction of sampled encodes that ran this row, when below one: the `~` badge. */
  readonly encodedFraction?: number;
  /** The hover sentence. Composed by the caller, which owns the vocabulary. */
  readonly explanation?: string;
}

/**
 * The row's own switch. `fixed` is a stage the pipeline always encodes: it
 * reads as a status, not a control that refuses.
 */
export interface PipelineLamp {
  readonly kind: "switch" | "fixed";
  readonly checked: boolean;
  readonly disabled?: boolean;
  readonly title: string;
  readonly ariaLabel: string;
  readonly onToggle?: () => void;
}

/** A plane this row wrote, presented instead of the composite. */
export interface PipelinePlane {
  readonly label: string;
  readonly title: string;
  readonly active: boolean;
  readonly onToggle: () => void;
}

export interface PipelineRow {
  readonly id: string;
  readonly label: string;
  readonly state: PipelineRowState;
  /** Everything the row means, as a hover tip. */
  readonly tip: string;
  /** A short factual line, shown when the stage is open. */
  readonly chip?: string;
  readonly cost?: PipelineCostFigure;
  /** Exclusive child measurements, already included in the row's total. */
  readonly breakdown?: readonly { readonly label: string; readonly cost: PipelineCostFigure }[];
  readonly lamp: PipelineLamp;
  readonly planes?: readonly PipelinePlane[];
  /** The stage's own settings, shown when it is open. */
  readonly controls?: ReactNode;
  /** A status the reader must see without opening anything: always drawn. */
  readonly notice?: ReactNode;
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
 * The figure as it reads in a column: a bare number, so a column of them can
 * be scanned against each other. A withheld row reads `0.00`, not a dash —
 * the dash is reserved for a measurement that is genuinely missing.
 */
export function pipelineCostFigure(cost: PipelineCostFigure): string {
  if (cost.duration_ms === undefined) return "—";
  const figure = cost.duration_ms >= 10 ? cost.duration_ms.toFixed(1) : cost.duration_ms.toFixed(2);
  if (cost.kind === "shared") return `⊂${figure}`;
  return cost.encodedFraction !== undefined ? `~${figure}` : figure;
}

/** Which band and stage are open, per pipeline, for the life of the page. */
const opened = new Map<string, { band?: string; stage?: string; unused?: string }>();

function useOpened(id: string) {
  const [state, setState] = useState(() => opened.get(id) ?? {});
  const update = (next: typeof state) => { opened.set(id, next); setState(next); };
  return [state, update] as const;
}

function Figure({ row }: { row: PipelineRow }) {
  return <output className="rail-cost-figure" data-kind={row.cost?.kind} title={row.cost?.explanation}>
    {row.cost ? pipelineCostFigure(row.cost) : "—"}
  </output>;
}

function Stage({ row, open, onOpen }: { row: PipelineRow; open: boolean; onOpen: () => void }) {
  const expandable = Boolean(row.controls || row.breakdown?.length || row.chip || row.planes?.length);
  return <div className="rail-cost-stage" data-state={row.state} data-open={open || undefined}>
    <div className="rail-cost-line">
      <button type="button" className="rail-cost-name" title={row.tip} disabled={!expandable}
        aria-expanded={expandable ? open : undefined} onClick={onOpen}>{row.label}</button>
      {row.lamp.kind === "switch"
        ? <Switch checked={row.lamp.checked} disabled={row.lamp.disabled} ariaLabel={row.lamp.ariaLabel}
          hint={row.lamp.title} onChange={() => row.lamp.onToggle?.()} />
        : <span />}
      <Figure row={row} />
    </div>
    {row.notice}
    {open && <div className="rail-sub">
      {row.chip && <small className="rail-cost-chip">{row.chip}</small>}
      {(row.breakdown?.length ?? 0) > 0 && <dl className="rail-cost-breakdown">
        {row.breakdown!.map(part => <div key={part.label} title={part.cost.explanation}>
          <dt>{part.label}</dt>
          <dd>{part.cost.duration_ms === undefined ? "—" : formatPipelineDuration(part.cost.duration_ms)}</dd>
        </div>)}
      </dl>}
      {(row.planes?.length ?? 0) > 0 && <div className="rail-cost-planes" role="group" aria-label={`${row.label} planes`}>
        {row.planes!.map(plane => <button key={plane.label} type="button" className={plane.active ? "active" : ""}
          aria-pressed={plane.active} title={plane.title} onClick={plane.onToggle}>{plane.label}</button>)}
      </div>}
      {row.controls}
    </div>}
  </div>;
}

export function PipelineCost({ id, bands }: { readonly id: string; readonly bands: readonly PipelineBand[] }) {
  const [state, setState] = useOpened(id);
  return <div className="rail-cost" data-testid={id}>
    {bands.map((band) => {
      const open = state.band === band.id;
      const used = band.rows.filter(row => row.state !== "unavailable");
      const unused = band.rows.filter(row => row.state === "unavailable");
      const showUnused = state.unused === band.id;
      const attention = band.rows.some(row => row.notice);
      const stage = (row: PipelineRow) => <Stage key={row.id} row={row} open={state.stage === row.id}
        onOpen={() => setState({ ...state, stage: state.stage === row.id ? undefined : row.id })} />;
      return <div key={band.id} className="rail-cost-band" data-open={open || undefined}>
        <button type="button" className="rail-cost-line rail-cost-head" aria-expanded={open}
          title={`${band.label}${band.share === undefined ? "" : ` · ${(band.share * 100).toFixed(1)}% of the total`}`}
          onClick={() => setState({ ...state, band: open ? undefined : band.id })}>
          <span className="rail-cost-name">{band.label.split(" · ")[0]}{attention && <i className="rail-cost-alert" aria-label="needs attention" />}</span>
          <span className="rail-cost-bar" aria-hidden><i style={{ width: `${Math.min(1, band.share ?? 0) * 100}%` }} /></span>
          <output className="rail-cost-figure">{band.cost_ms === undefined ? "—" : pipelineCostFigure({ kind: "measured", duration_ms: band.cost_ms })}</output>
        </button>
        {open && <div className="rail-cost-stages">
          {used.map(stage)}
          {unused.length > 0 && <button type="button" className="rail-text-button" aria-expanded={showUnused}
            onClick={() => setState({ ...state, unused: showUnused ? undefined : band.id })}>
            {unused.length} not used
          </button>}
          {showUnused && unused.map(stage)}
        </div>}
      </div>;
    })}
  </div>;
}
