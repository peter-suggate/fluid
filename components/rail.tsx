"use client";

import type { ReactNode } from "react";
import { Choice, ResetButton, Select, Slider, Swatch, Switch, Value, type ControlOption } from "./ui";

/**
 * What a rail key draws: its glyph, its digit in the corner where it has one,
 * and its name.
 *
 * The name stands outside the rail's card, over the scene — right of the
 * sculpt rail, left of the view rail — so the card stays a column of marks and
 * the names read as captions to it, haloed against whatever the scene puts
 * behind them. A reader learns the rail by reading it rather than by hovering
 * each mark in turn; the tip, and the line a lit key opens, take the name's
 * place while they stand.
 */
export function RailKeyFace({ glyph, label, shortcut }: { glyph: ReactNode; label: string; shortcut?: string }) {
  return <>
    {glyph}
    {shortcut && <kbd aria-hidden>{shortcut}</kbd>}
    <span className="tool-rail-label" aria-hidden>{label}</span>
  </>;
}

/*
 * The lines a rail section's card is made of.
 *
 * One setting is one line: its name on the left, taking what is left and
 * clipping, and its control in a fixed column on the right, so every track on
 * every card starts and ends in the same place and every knob stands on the
 * same right edge. The card is the only disclosure — a line never opens a
 * menu of its own — and a line's explanation is its hover hint, not prose
 * under it; a note is for a status the reader cannot otherwise see, such as
 * why a line is dimmed.
 */

/** A line's name, with the swatch of the colour it draws in where it has one. */
function RailName({ label, swatch }: { label: ReactNode; swatch?: string }) {
  return <span className="rail-name">{swatch && <Swatch color={swatch} />}{label}</span>;
}

/**
 * A slider: name, track, readout, reset. The reset keeps its place when there
 * is nothing to reset, and when the setting has no reset at all, so a slider
 * that has one and a slider that does not still share their track.
 */
export function RailRange({ label, value, min, max, step, digits, unit, display, disabled, hint, ariaLabel, testId,
  onInput, onChange, onGestureEnd, modified = false, onReset }: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  digits?: number;
  unit?: string;
  /** A readout in place of the value, e.g. "Custom" for a dial off its curve. */
  display?: string;
  disabled?: boolean;
  hint?: string;
  ariaLabel?: string;
  testId?: string;
  onInput?: (value: number) => void;
  onChange?: (value: number) => void;
  onGestureEnd?: () => void;
  modified?: boolean;
  onReset?: () => void;
}) {
  const name = ariaLabel ?? (typeof label === "string" ? label : undefined);
  return <div className="view-rail-range" title={hint} data-testid={testId}>
    <RailName label={label} />
    <Slider value={value} min={min} max={max} step={step} disabled={disabled} ariaLabel={name}
      onInput={onInput} onChange={onChange ?? onInput} onGestureEnd={onGestureEnd} />
    <Value value={display ?? value} digits={display === undefined ? digits : undefined} step={step} unit={unit} />
    {onReset
      ? <ResetButton onReset={onReset} ariaLabel={`Reset ${name ?? "setting"}`} className={modified ? undefined : "is-idle"} />
      : <span aria-hidden />}
  </div>;
}

/** A switch: its name, and its knob where the sliders' readouts end. */
export function RailToggle({ label, checked, onChange, disabled, hint, ariaLabel, swatch, testId }: {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  hint?: string;
  ariaLabel?: string;
  swatch?: string;
  testId?: string;
}) {
  return <div className="view-rail-toggle" title={hint} data-testid={testId}>
    <RailName label={label} swatch={swatch} />
    <Switch checked={checked} onChange={onChange} disabled={disabled} hint={hint}
      ariaLabel={ariaLabel ?? (typeof label === "string" ? label : undefined)} />
  </div>;
}

/** One of a few: its name, and the row of marks ending where the knobs do. */
export function RailChoice<T extends string>({ label, value, options, onChange, disabled, hint, ariaLabel, testId }: {
  label: ReactNode;
  value: T;
  options: ReadonlyArray<ControlOption<T>>;
  onChange: (value: T) => void;
  disabled?: boolean;
  hint?: string;
  ariaLabel?: string;
  testId?: string;
}) {
  return <div className="view-rail-toggle" title={hint} data-testid={testId}>
    <RailName label={label} />
    <Choice value={value} options={options} onChange={onChange} disabled={disabled}
      ariaLabel={ariaLabel ?? (typeof label === "string" ? label : undefined)} />
  </div>;
}

/** One of many: its name, and a list ending where the knobs do. */
export function RailSelect<T extends string>({ label, value, options, onChange, disabled, hint, ariaLabel, testId }: {
  label: ReactNode;
  value: T;
  options: ReadonlyArray<{ readonly value: T; readonly label: string; readonly hint?: string }>;
  onChange: (value: T) => void;
  disabled?: boolean;
  hint?: string;
  ariaLabel?: string;
  testId?: string;
}) {
  return <div className="view-rail-toggle" title={hint}>
    <RailName label={label} />
    <Select value={value} options={options} onChange={onChange} disabled={disabled} hint={hint} testId={testId}
      ariaLabel={ariaLabel ?? (typeof label === "string" ? label : undefined)} />
  </div>;
}

/** A reading: its name, and the figure where the knobs end. */
export function RailReadout({ label, value, hint, testId }: { label: ReactNode; value: ReactNode; hint?: string; testId?: string }) {
  return <div className="view-rail-toggle" title={hint}>
    <RailName label={label} />
    <output className="ui-value" data-testid={testId}>{value}</output>
  </div>;
}

/** A control of its own kind: its name, and the control where the knobs end. */
export function RailLine({ label, hint, testId, children }: { label: ReactNode; hint?: string; testId?: string; children: ReactNode }) {
  return <div className="view-rail-toggle" title={hint} data-testid={testId}>
    <RailName label={label} />
    {children}
  </div>;
}

/** The lines that belong to the switch or choice above them, set in under it. */
export function RailSub({ children }: { children: ReactNode }) {
  return <div className="rail-sub">{children}</div>;
}

/** A seam between groups of lines, captioned where the group needs a name. */
export function RailRule({ label }: { label?: string }) {
  return <div className="rail-rule" role="separator">{label && <span>{label}</span>}</div>;
}

/** A way out of the card: a dialog that holds what the card does not. */
export function RailLink({ label, onClick, hint, testId }: { label: string; onClick: () => void; hint?: string; testId?: string }) {
  return <button type="button" className="rail-link" title={hint} data-testid={testId} onClick={onClick}>
    <span>{label}</span><span aria-hidden>→</span>
  </button>;
}

/** What a drawn colour means: a run of swatches, wrapped to the card. */
export function RailKey({ label, hint, entries }: {
  label: string;
  hint?: string;
  entries: ReadonlyArray<{ readonly color: string; readonly label: string; readonly title?: string; readonly wide?: boolean }>;
}) {
  return <div className="rail-key" aria-label={label} title={hint}>
    {entries.map(entry => <span key={entry.label} title={entry.title}>
      <i aria-hidden className={entry.wide ? "is-wide" : undefined} style={{ background: entry.color }} />{entry.label}
    </span>)}
  </div>;
}
