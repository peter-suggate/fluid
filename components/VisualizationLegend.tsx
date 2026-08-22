"use client";

import type { CSSProperties } from "react";
import { visualizationMark } from "../lib/core/visualization-catalog";
import type {
  Visualization,
  VisualizationLegendEntry,
  VisualizationLegendMark,
} from "../lib/core/visualization-registry";

/**
 * The mark a legend row draws, at legend size.
 *
 * Deliberately the same vocabulary the decorators build with — a coupling is an
 * arrow because it has a direction, an extent is a box, and anything derived
 * rather than measured is dashed. Colour alone cannot carry that, and a reader
 * who has to learn which of four blue things is which has not been given a
 * legend at all.
 */
export function LegendMark({
  mark, swatch, dim,
}: {
  readonly mark: VisualizationLegendMark;
  readonly swatch: string;
  readonly dim?: boolean;
}) {
  const stroke = swatch;
  const opacity = dim ? 0.32 : 1;
  return (
    <svg
      className="legend-mark"
      data-mark={mark}
      viewBox="0 0 18 12"
      width="18"
      height="12"
      aria-hidden="true"
      style={{ opacity }}
    >
      {mark === "line" && <path d="M1 6 H17" stroke={stroke} strokeWidth="2" fill="none" />}
      {mark === "arrow" && <>
        <path d="M1 6 H12" stroke={stroke} strokeWidth="2" fill="none" />
        <path d="M11 2.5 L17 6 L11 9.5 Z" fill={stroke} />
      </>}
      {mark === "box" && (
        <rect x="2" y="1.5" width="14" height="9" stroke={stroke} strokeWidth="1.6" fill="none" />
      )}
      {mark === "dashed-box" && (
        <rect
          x="2" y="1.5" width="14" height="9"
          stroke={stroke} strokeWidth="1.6" fill="none" strokeDasharray="3 2.4"
        />
      )}
      {mark === "ring" && (
        <ellipse cx="9" cy="6" rx="5" ry="4.2" stroke={stroke} strokeWidth="1.8" fill="none" />
      )}
      {mark === "cross" && (
        <path d="M4 1.5 L14 10.5 M14 1.5 L4 10.5" stroke={stroke} strokeWidth="1.8" fill="none" />
      )}
      {mark === "point" && <circle cx="9" cy="6" r="3" fill={stroke} />}
    </svg>
  );
}

/**
 * The cases one view distinguishes, as marks and their names.
 *
 * Shared with the stage lenses rather than reimplemented for them: a lens's
 * `legend` is the same `VisualizationLegendEntry[]` a catalog view's is, and two
 * renderings of it would be two chances for a lens colour to be drawn by a rule
 * the field views stopped following. The only difference is `leadMark`, because
 * a catalog row already draws its first swatch on the row itself and a lens has
 * no row above the list to carry one.
 */
export function LegendEntries({
  entries, dim, leadMark, style,
}: {
  readonly entries: readonly VisualizationLegendEntry[];
  readonly dim?: boolean;
  readonly leadMark?: boolean;
  readonly style?: CSSProperties;
}) {
  return (
    <ul style={style}>
      {entries.map((entry, index) => (
        <li
          key={`${entry.swatch}:${entry.label}`}
          data-primary={index === 0 ? "true" : "false"}
        >
          {(leadMark || index > 0) && (
            <LegendMark mark={entry.mark ?? "line"} swatch={entry.swatch} dim={dim} />
          )}
          <span>{entry.label}</span>
        </li>
      ))}
    </ul>
  );
}

interface VisualizationLegendProps {
  /** Catalog entries, in catalog order — which is also the order they draw. */
  readonly definitions: readonly Visualization[];
  /** Groups currently drawn. Expanded, a row that is off stays, dimmed. */
  readonly enabled: readonly string[];
  /** What each group contributed to this subject, by group tag. */
  readonly counts?: ReadonlyMap<string, number>;
  /** Show each view's own sub-legend — its second colour and what it means. */
  readonly expanded?: boolean;
  readonly onToggle?: (group: string) => void;
}

/** One row of the legend: a view, how it is drawn, and whether it is. */
export interface VisualizationLegendRow {
  readonly definition: Visualization;
  readonly group: string;
  readonly swatch: string;
  readonly mark: VisualizationLegendMark;
  readonly enabled: boolean;
  /** A published count of zero — the view is on and there is none of it here. */
  readonly empty: boolean;
  readonly count?: number;
  readonly entries: readonly VisualizationLegendEntry[];
}

/**
 * Every declared view, in catalog order, as rows.
 *
 * A row is never dropped. One that vanished when its view was switched off took
 * its own switch with it, and a list that reflows as the picture changes cannot
 * be clicked at speed. Off and empty are states a row is *in*, which is what
 * `enabled` and `empty` carry — the difference between them matters, because a
 * measured zero says the band had nothing here and an absent count says nobody
 * counted.
 */
export function legendRows(
  definitions: readonly Visualization[],
  enabled: readonly string[],
  counts?: ReadonlyMap<string, number>,
): readonly VisualizationLegendRow[] {
  const active = new Set(enabled);
  return definitions.map((definition) => {
    const group = definition.group ?? definition.id;
    const count = counts?.get(group);
    return {
      definition,
      group,
      swatch: definition.swatch ?? definition.legend?.[0]?.swatch ?? "#8895a4",
      mark: visualizationMark(definition),
      enabled: active.has(group),
      empty: count === 0,
      ...(count === undefined ? {} : { count }),
      entries: definition.legend ?? [],
    };
  });
}

/**
 * The legend, which is also the switchboard.
 *
 * One list rather than a strip of toggles plus a separate key: they were always
 * the same list, and splitting them made the reader match a colour in one place
 * against a name in another. Every glyph, colour, label and description here is
 * read from the pass that draws the line work, so a view cannot go on being
 * described by a colour it stopped using.
 *
 * **Every view is always listed**, on or off, drawn or empty. A row that
 * disappeared when it was switched off would take its own switch with it, and
 * flicking a layer off to see what is underneath and straight back on is the
 * gesture this panel exists for. Off and empty rows are dimmed, not hidden, so
 * the list still reads as what is on screen at a glance.
 *
 * What the disclosure adds is meaning, not entries: what each mark stands for,
 * and the extra colours a view distinguishes its cases by. That detail is worth
 * nothing until someone is looking closely enough to want it.
 */
export function VisualizationLegend({
  definitions, enabled, counts, expanded, onToggle,
}: VisualizationLegendProps) {
  return (
    <ul
      className="visualization-legend"
      data-testid="visualization-legend"
      data-expanded={expanded ? "true" : "false"}
    >
      {legendRows(definitions, enabled, counts).map((row) => (
        <li
          key={row.definition.id}
          data-group={row.group}
          data-enabled={row.enabled ? "true" : "false"}
          // Dimmed, never dropped: the row is the switch, and a list that
          // reflowed as the picture changed could not be clicked at speed.
          data-empty={row.empty ? "true" : "false"}
        >
          <button
            type="button"
            aria-pressed={row.enabled}
            onClick={onToggle ? () => onToggle(row.group) : undefined}
            disabled={!onToggle}
            title={`${row.definition.description}`
              + `${row.definition.source ? ` · ${row.definition.source}` : ""}`}
          >
            <LegendMark mark={row.mark} swatch={row.swatch} dim={!row.enabled} />
            <span>{row.definition.label}</span>
            {row.count !== undefined && <small>{row.count}</small>}
          </button>
          {expanded && row.entries.length > 0 && (
            <LegendEntries entries={row.entries} dim={!row.enabled} />
          )}
        </li>
      ))}
    </ul>
  );
}
