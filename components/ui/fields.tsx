"use client";

import { useId, type HTMLAttributes, type ReactNode } from "react";
import {
  Choice, FieldLabelContext, NumberInput, ResetButton, Select, Slider, Switch, Value, type ControlOption,
} from "./controls";

/**
 * Labelled rows, and the list that lines them up.
 *
 * A field is three cells — name, control, value — and a `FieldList` shares
 * those three columns across every field in it (subgrid), so every track,
 * choice and switch starts on one edge and every value ends on another.
 * Row by row, each field used to size its own columns, and a card's tracks
 * started wherever that row's value text left them.
 *
 * The name column is a fixed share of the list (`--ui-name`), not sized to its
 * content, so a list of readouts below a list of settings puts its values on
 * the same edge as the controls above it.
 */

const cx = (...names: ReadonlyArray<string | false | undefined>) => names.filter(Boolean).join(" ");

/**
 * One row: name, control, and — for a control with a figure — its value.
 *
 * A field without a value lets its control take the value column too, so a
 * choice or a select is as wide as a track and its value together.
 */
export function Field({ label, hint, disabled = false, value, modified = false, onReset, resetHint, className, testId, children }: {
  label: ReactNode;
  hint?: string;
  disabled?: boolean;
  value?: ReactNode;
  /** Shows ↺ beside the value while true; pressing it calls `onReset`. */
  modified?: boolean;
  onReset?: () => void;
  resetHint?: string;
  className?: string;
  testId?: string;
  children: ReactNode;
}) {
  const labelId = useId();
  const reset = modified && onReset ? <ResetButton onReset={onReset} hint={resetHint} /> : null;
  const hasValue = value !== undefined || reset !== null;
  return <div className={cx("ui-field", disabled && "is-disabled", className)} title={hint} data-testid={testId}>
    <span className="ui-field-label" id={labelId}>{label}</span>
    <FieldLabelContext.Provider value={labelId}>
      <span className={cx("ui-field-control", !hasValue && "is-wide")}>{children}</span>
    </FieldLabelContext.Provider>
    {hasValue && <span className="ui-field-value">{value}{reset}</span>}
  </div>;
}

/**
 * A column of fields on one rail.
 *
 * Anything in it that is not a field — a note, a drawer, a nested list, a row
 * of verbs — spans the whole width. `section` sets the list apart from what is
 * above it with a hairline, for a block that is about the card rather than in
 * it.
 */
export function FieldList({ section = false, className, ariaLabel, testId, children, ...rest }: HTMLAttributes<HTMLDivElement> & {
  section?: boolean;
  ariaLabel?: string;
  testId?: string;
  /** Any other attribute — `data-*` state a test reads, a role — lands on the list. */
  [data: `data-${string}`]: string | undefined;
}) {
  return <div {...rest} className={cx("ui-fields", section && "is-section", className)} aria-label={ariaLabel ?? rest["aria-label"]}
    data-testid={testId ?? rest["data-testid"]}>{children}</div>;
}

/** A row of verbs at the foot of a list, right-aligned. */
export function FieldActions({ children }: { children: ReactNode }) {
  return <div className="ui-actions">{children}</div>;
}

/** A line of prose inside a list. Kept short; the hover tip is where prose lives. */
export function FieldNote({ children }: { children: ReactNode }) {
  return <p className="ui-note">{children}</p>;
}

/** A row of pill switches or chips that *are* the row. */
export function ControlRow({ children, className, ariaLabel }: { children: ReactNode; className?: string; ariaLabel?: string }) {
  return <div className={cx("ui-row", className)} role={ariaLabel ? "group" : undefined} aria-label={ariaLabel}>{children}</div>;
}

/**
 * A slider with its value, and optionally a typed entry and a reset.
 *
 * `defaultValue` derives the modified mark and the reset from the one figure a
 * caller knows; `modified`/`onReset` override it for a caller whose "default"
 * is a preset rather than a number. `scale` shows the value multiplied — a
 * fraction as a percentage — without the caller converting both ways.
 */
export function RangeField({ label, value, min, max, step, digits, scale = 1, unit, onChange, onInput, hint, disabled = false,
  editable = false, overrange = false, defaultValue, modified, onReset, resetHint, className, testId }: {
  label: ReactNode;
  value: number;
  min: number;
  max: number;
  step: number;
  digits?: number;
  scale?: number;
  unit?: string;
  onChange: (value: number) => void;
  onInput?: (value: number) => void;
  hint?: string;
  disabled?: boolean;
  /** Make the value typable, for a figure the track's steps cannot land on. */
  editable?: boolean;
  /**
   * Let the typed value go past the track's ends. For a track that is a span
   * guessed around a figure (the shape lab derives one from each pristine
   * value) rather than the quantity's real limits, where clamping the entry
   * would make the guess a wall.
   */
  overrange?: boolean;
  defaultValue?: number;
  modified?: boolean;
  onReset?: () => void;
  resetHint?: string;
  className?: string;
  testId?: string;
}) {
  const isModified = modified ?? (defaultValue !== undefined && value !== defaultValue);
  const reset = onReset ?? (defaultValue !== undefined ? () => onChange(defaultValue) : undefined);
  return <Field label={label} hint={hint} disabled={disabled} className={className} testId={testId}
    modified={isModified} onReset={reset} resetHint={resetHint}
    value={editable
      ? <NumberInput value={value} onChange={onChange} min={overrange ? undefined : min}
          max={overrange ? undefined : max} step={step * scale}
          digits={digits} scale={scale} unit={unit} disabled={disabled} className="is-inline" />
      : <Value value={value} digits={digits} step={step * scale} scale={scale} unit={unit} />}>
    <Slider value={value} min={min} max={max} step={step} onChange={onChange} onInput={onInput} disabled={disabled} />
  </Field>;
}

/** A typed number with its name, for a quantity with no two ends to slide between. */
export function NumberField({ label, value, onChange, step, digits, min, max, scale, unit, hint, disabled, testId }: {
  label: ReactNode;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  digits?: number;
  min?: number;
  max?: number;
  scale?: number;
  unit?: string;
  hint?: string;
  disabled?: boolean;
  testId?: string;
}) {
  return <Field label={label} hint={hint} disabled={disabled} testId={testId}>
    <NumberInput value={value} onChange={onChange} step={step} digits={digits} min={min} max={max}
      scale={scale} unit={unit} disabled={disabled} />
  </Field>;
}

export function ChoiceField<T extends string>({ label, value, options, onChange, hint, disabled, testId }: {
  label: ReactNode;
  value: T;
  options: ReadonlyArray<ControlOption<T>>;
  onChange: (value: T) => void;
  hint?: string;
  disabled?: boolean;
  testId?: string;
}) {
  return <Field label={label} hint={hint} disabled={disabled} testId={testId}>
    <Choice value={value} options={options} onChange={onChange} disabled={disabled} />
  </Field>;
}

export function SelectField<T extends string>({ label, value, options, onChange, hint, disabled, customLabel, testId }: {
  label: ReactNode;
  value: T;
  options: ReadonlyArray<{ readonly value: T; readonly label: string; readonly hint?: string; readonly disabled?: boolean }>;
  onChange: (value: T) => void;
  hint?: string;
  disabled?: boolean;
  customLabel?: string;
  testId?: string;
}) {
  return <Field label={label} hint={hint} disabled={disabled} testId={testId}>
    <Select value={value} options={options} onChange={onChange} disabled={disabled} customLabel={customLabel} />
  </Field>;
}

export function SwitchField({ label, checked, onChange, hint, disabled, testId }: {
  label: ReactNode;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
  disabled?: boolean;
  testId?: string;
}) {
  return <Field label={label} hint={hint} disabled={disabled} testId={testId}>
    <Switch checked={checked} onChange={onChange} disabled={disabled} />
  </Field>;
}

/** A measured figure in the same row rhythm as the controls around it. */
export function Readout({ label, value, hint, testId }: { label: ReactNode; value: ReactNode; hint?: string; testId?: string }) {
  return <Field label={label} hint={hint} testId={testId} className="is-readout">
    <output className="ui-value">{value}</output>
  </Field>;
}

/**
 * Facts, not controls: a key/value list whose values are text.
 *
 * A fact is often a phrase ("Native rectangular fields; all-resident page
 * catalogue"), so its value gets the control and value columns together and
 * wraps as text; in a slider's value slot a phrase stacked one word per line.
 */
export function Facts({ items, className }: {
  items: ReadonlyArray<{ readonly label: ReactNode; readonly value: ReactNode; readonly hint?: string }>;
  className?: string;
}) {
  return <dl className={cx("ui-facts", className)}>
    {items.map((item, index) => <div key={index} title={item.hint}>
      <dt>{item.label}</dt>
      <dd>{item.value}</dd>
    </div>)}
  </dl>;
}
