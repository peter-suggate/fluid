"use client";

import { createContext, useContext, type ReactNode } from "react";
import { useNumberEntry, useSliderGesture } from "./gestures";
import { clampNumber, formatNumber, printNumber } from "./number";

/**
 * The app's control primitives: one of each, sized by where they are put.
 *
 * Every panel used to carry its own copy — the studio's `RangeControl`, the
 * pipeline's `PipeRange`, the toolstrip's `ToolstripScrub`, and a dozen raw
 * `<input>`s and `<select>`s — each with its own commit rule, number format,
 * disabled story and accessible name. These are the one copy.
 *
 * None of them sets its own size. Height, inline padding and type size come
 * from the `--control-*` tokens the surrounding surface states (the studio's
 * form floor, a pipeline card, the toolstrip, a toolstrip pane), so the same
 * slider is a form field in one place and a 20px strip control in another
 * without a variant prop.
 *
 * Every primitive takes `hint` (the vocabulary the feature declarations use)
 * and an accessible name — `ariaLabel`, or the enclosing `Field`'s label.
 */

/** The label a `Field` gives the control inside it, when it has no name of its own. */
export const FieldLabelContext = createContext<string | undefined>(undefined);

function useName(ariaLabel: string | undefined) {
  const labelledBy = useContext(FieldLabelContext);
  return ariaLabel !== undefined ? { "aria-label": ariaLabel } : labelledBy ? { "aria-labelledby": labelledBy } : {};
}

const cx = (...names: ReadonlyArray<string | false | undefined>) => names.filter(Boolean).join(" ");

export interface ControlOption<T extends string> {
  readonly value: T;
  readonly label: ReactNode;
  readonly hint?: string;
  readonly disabled?: boolean;
  /** The option's name when its label is a glyph with no words in it. */
  readonly ariaLabel?: string;
}

/**
 * A range, committed once per gesture.
 *
 * `onChange` hears the landed value; `onInput`, when given, hears every move of
 * the thumb, for an owner that previews live (see `useSliderGesture`);
 * `onGestureEnd` hears every gesture close, changed or not, for an owner that
 * brackets its live previews in one undo entry.
 */
export function Slider({ value, min, max, step, onChange, onInput, onGestureEnd, disabled = false, ariaLabel, hint, className }: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange?: (value: number) => void;
  onInput?: (value: number) => void;
  onGestureEnd?: () => void;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  className?: string;
}) {
  const { inputProps } = useSliderGesture({ value, min, max, onChange: onChange ?? (() => {}), onInput, onGestureEnd });
  return <input type="range" className={cx("ui-slider", className)} min={min} max={max} step={step}
    disabled={disabled} title={hint} {...useName(ariaLabel)} {...inputProps} />;
}

/**
 * A typed number, committed on Enter or on leaving the field; Escape abandons.
 *
 * `scale` shows the value multiplied (a fraction typed as a percentage) and
 * divides the entry back; `tag` is the one or two letters that say which
 * quantity a bare box in a row of siblings is (W/H/D).
 */
export function NumberInput({ value, step, digits, min, max, scale = 1, onChange, disabled = false, ariaLabel, hint, tag, unit, className }: {
  value: number;
  step?: number;
  digits?: number;
  min?: number;
  max?: number;
  scale?: number;
  onChange: (value: number) => void;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  tag?: string;
  unit?: string;
  className?: string;
}) {
  const { inputProps } = useNumberEntry({
    value: value * scale, step, digits,
    min: min === undefined ? undefined : min * scale,
    max: max === undefined ? undefined : max * scale,
    onCommit: (next) => onChange(scale === 1 ? next : next / scale),
  });
  return <span className={cx("ui-number", disabled && "is-disabled", className)} title={hint}>
    {tag !== undefined && <b aria-hidden="true">{tag}</b>}
    <input disabled={disabled} {...useName(ariaLabel)} {...inputProps} />
    {unit !== undefined && <small>{unit}</small>}
  </span>;
}

/**
 * A number printed as a readout, with its unit.
 *
 * `digits` fixes the decimals (exponent under a thousandth); `step` prints at
 * the step's grain instead. A string is printed as given.
 */
export function Value({ value, digits, step, scale = 1, unit, className }: {
  value: number | string;
  digits?: number;
  step?: number;
  scale?: number;
  unit?: string;
  className?: string;
}) {
  const text = typeof value === "string" ? value
    : digits !== undefined ? formatNumber(value * scale, digits)
    : printNumber(value * scale, { step });
  return <output className={cx("ui-value", className)}>{text}{unit ? <small>{unit}</small> : null}</output>;
}

/**
 * Minus, the value, plus.
 *
 * `factor` makes the steps multiplicative — halve and double a step size, a
 * scale — where an additive step would take a hundred presses to cross the
 * range. Each button says why it is disabled when it is, since a stepper at its
 * end is otherwise a button that silently refuses.
 *
 * A multiplicative stepper has no step to print at, so `grain` gives it one: a
 * halved 16.667 ms reads "8.3" rather than every digit of 8.3333, and a whole
 * value reads without a trailing ".0". The value itself is never rounded — the
 * next halving starts from the exact figure, not the printed one.
 *
 * `readout` is for a stepper whose value is not the figure worth reading: a
 * scale step taken relative to 1, whose ends are "the next step is available",
 * reads as the extents it produces rather than as "1".
 */
export function Stepper({ value, onChange, min, max, step = 1, factor, grain, digits, unit, editable = false, disabled = false,
  ariaLabel, hint, readout, decreaseLabel = "−", increaseLabel = "+", decreaseHint, increaseHint,
  decreaseTestId, increaseTestId, className }: {
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  factor?: number;
  /** With `factor`: the precision the value is printed and typed at. */
  grain?: number;
  digits?: number;
  unit?: string;
  editable?: boolean;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  /** Printed in place of the value; ignored when `editable`. */
  readout?: ReactNode;
  decreaseLabel?: ReactNode;
  increaseLabel?: ReactNode;
  decreaseHint?: string;
  increaseHint?: string;
  decreaseTestId?: string;
  increaseTestId?: string;
  className?: string;
}) {
  const name = useName(ariaLabel);
  const down = clampNumber(factor ? value / factor : value - step, min, max);
  const up = clampNumber(factor ? value * factor : value + step, min, max);
  const printStep = factor ? grain : step;
  return <span className={cx("ui-stepper", disabled && "is-disabled", className)} role="group" title={hint} {...name}>
    <button type="button" disabled={disabled || down === value} title={decreaseHint}
      aria-label={decreaseHint ?? "Decrease"} data-testid={decreaseTestId} onClick={() => onChange(down)}>{decreaseLabel}</button>
    {editable
      ? <NumberInput value={value} onChange={onChange} min={min} max={max} step={printStep}
          digits={digits} unit={unit} disabled={disabled} ariaLabel={ariaLabel ?? "Value"} />
      : readout !== undefined ? <output className="ui-value">{readout}</output>
      : <Value value={value} digits={digits} step={printStep} unit={unit} />}
    <button type="button" disabled={disabled || up === value} title={increaseHint}
      aria-label={increaseHint ?? "Increase"} data-testid={increaseTestId} onClick={() => onChange(up)}>{increaseLabel}</button>
  </span>;
}

/**
 * A dropdown.
 *
 * A value outside the options — a scene that set something the menu does not
 * offer — shows as a disabled `customLabel` entry rather than silently reading
 * as the first option.
 *
 * `id` is for a caller that names it with a sibling `<label htmlFor>` rather
 * than an enclosing `Field` — a header slot whose flex row owns the gap.
 */
export function Select<T extends string>({ value, options, onChange, disabled = false, ariaLabel, hint, customLabel = "Custom", className, id, testId }: {
  value: T;
  options: ReadonlyArray<{ readonly value: T; readonly label: string; readonly hint?: string; readonly disabled?: boolean }>;
  onChange: (value: T) => void;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  customLabel?: string;
  className?: string;
  id?: string;
  testId?: string;
}) {
  const known = options.some((option) => option.value === value);
  return <select className={cx("ui-select", className)} id={id} data-testid={testId} value={value} disabled={disabled} title={hint}
    {...useName(ariaLabel)} onChange={(event) => onChange(event.currentTarget.value as T)}>
    {!known && <option value={value} disabled>{customLabel}</option>}
    {options.map((option) => <option key={option.value} value={option.value} title={option.hint}
      disabled={option.disabled}>{option.label}</option>)}
  </select>;
}

/**
 * One of a few, as a row of buttons.
 *
 * Choosing the option already chosen is not a change and is not reported: an
 * owner whose `onChange` rebuilds or re-seeds should not do it for a no-op.
 */
export function Choice<T extends string>({ value, options, onChange, disabled = false, ariaLabel, className }: {
  value: T;
  options: ReadonlyArray<ControlOption<T>>;
  onChange: (value: T) => void;
  disabled?: boolean;
  ariaLabel?: string;
  className?: string;
}) {
  return <div className={cx("ui-choice", className)} role="group" {...useName(ariaLabel)}>
    {options.map((option) => <button key={option.value} type="button" title={option.hint}
      aria-label={option.ariaLabel} disabled={disabled || option.disabled} aria-pressed={option.value === value}
      className={option.value === value ? "active" : undefined}
      onClick={() => { if (option.value !== value) onChange(option.value); }}>{option.label}</button>)}
  </div>;
}

/**
 * A button that stays down: a pin, a lock, a layer, a mode that is on.
 *
 * For a setting that is simply on or off, `Switch` says so better; this is for
 * a verb-like state that reads as a chip — and for a set of them side by side,
 * where each is independent.
 */
export function ToggleButton({ pressed, onChange, children, disabled = false, ariaLabel, hint, className, testId }: {
  pressed: boolean;
  onChange: (pressed: boolean) => void;
  children: ReactNode;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  className?: string;
  testId?: string;
}) {
  return <button type="button" className={cx("ui-toggle", pressed && "active", className)} aria-pressed={pressed}
    aria-label={ariaLabel} disabled={disabled} title={hint} data-testid={testId}
    onClick={() => onChange(!pressed)}>{children}</button>;
}

/**
 * On or off.
 *
 * Bare, it is a knob on a field's control column; with `label` it is a pill
 * carrying its own name, for a row of switches that *are* the row.
 */
export function Switch({ checked, onChange, label, disabled = false, ariaLabel, hint, className }: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label?: ReactNode;
  disabled?: boolean;
  ariaLabel?: string;
  hint?: string;
  className?: string;
}) {
  // A pill is named by its own text; a bare knob by `ariaLabel` or its field.
  const name = useName(ariaLabel);
  return <button type="button" role="switch" aria-checked={checked}
    className={cx("ui-switch", label !== undefined && "is-pill", className)} disabled={disabled} title={hint}
    {...(label !== undefined && ariaLabel === undefined ? {} : name)} onClick={() => onChange(!checked)}>
    <i aria-hidden="true" />{label !== undefined && <span>{label}</span>}
  </button>;
}

/** A verb. `tone` marks the one that is the point of its row, or the one that destroys. */
export function Button({ children, onClick, disabled = false, hint, ariaLabel, tone = "default", className, testId }: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  hint?: string;
  ariaLabel?: string;
  tone?: "default" | "accent" | "danger";
  className?: string;
  testId?: string;
}) {
  return <button type="button" className={cx("ui-button", tone !== "default" && `is-${tone}`, className)}
    disabled={disabled} title={hint} aria-label={ariaLabel} data-testid={testId} onClick={onClick}>{children}</button>;
}

/**
 * ↺: back to the default. Shown only while a value differs from it.
 *
 * `ariaLabel` names *which* value, for a list of them where every hint is the
 * same sentence; alone, the hint is the name.
 */
export function ResetButton({ onReset, hint = "Reset to default", ariaLabel, className }: {
  onReset: () => void;
  hint?: string;
  ariaLabel?: string;
  className?: string;
}) {
  return <button type="button" className={cx("ui-reset", className)} title={hint} aria-label={ariaLabel ?? hint}
    onClick={(event) => { event.preventDefault(); onReset(); }}>↺</button>;
}

/** A figure in a card: the studio diagnostics' tile. */
export function Metric({ label, value, unit, tone = "neutral", testId }: {
  label: string;
  value: string;
  unit?: string;
  tone?: "neutral" | "good" | "warn";
  testId?: string;
}) {
  return <div className={`metric-card tone-${tone}`} data-testid={testId}><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</div>;
}
