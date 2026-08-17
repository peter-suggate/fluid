"use client";

import { useState } from "react";

/**
 * One-line controls for the frame pipeline.
 *
 * `RangeControl` stacks a heading, a readout and a track over about 44px, which
 * is right for a panel that shows a dozen fields and wrong for one that shows a
 * hundred with nothing folded away. These put the label, the track and the value
 * on a single 20px row and put every word of explanation in the hover tip, which
 * is the only place prose lives in this panel.
 */

export function PipeRange({ label, value, min, max, step, digits = 0, unit, onChange, hint, disabled = false, modified, onReset }: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits?: number;
  unit?: string;
  onChange: (value: number) => void;
  hint?: string;
  disabled?: boolean;
  modified?: boolean;
  onReset?: () => void;
}) {
  // Held while dragging so the readout tracks the thumb without committing a
  // value per pointer-move; the commit lands on release, as `RangeControl` does.
  const [draft, setDraft] = useState<{ base: number; value: number } | null>(null);
  const shown = draft?.base === value ? draft.value : value;
  const commit = (next: number) => {
    const clamped = Math.min(max, Math.max(min, next));
    setDraft(null);
    if (clamped !== value) onChange(clamped);
  };
  return <label className={`pipe-field${disabled ? " is-disabled" : ""}`} title={hint}>
    <span>{label}</span>
    <input type="range" disabled={disabled} min={min} max={max} step={step} value={shown}
      onChange={(event) => setDraft({ base: value, value: Number(event.currentTarget.value) })}
      onPointerUp={(event) => commit(Number(event.currentTarget.value))}
      onPointerCancel={() => setDraft(null)}
      onKeyUp={(event) => commit(Number(event.currentTarget.value))} />
    <output>
      {shown.toFixed(digits)}{unit ? <small>{unit}</small> : null}
      {modified && onReset && <button type="button" className="pipe-reset" title="Reset to the balanced value"
        onClick={(event) => { event.preventDefault(); onReset(); }}>↺</button>}
    </output>
  </label>;
}

export function PipeToggle({ label, checked, onChange, disabled = false, hint }: {
  label: string; checked: boolean; onChange: (checked: boolean) => void; disabled?: boolean; hint?: string;
}) {
  return <button type="button" className="pipe-toggle" role="switch" aria-checked={checked}
    disabled={disabled} title={hint} onClick={() => onChange(!checked)}>
    <i aria-hidden="true" /><span>{label}</span>
  </button>;
}

export function PipeChoice<T extends string>({ label, value, options, onChange, disabled = false }: {
  label?: string; value: T; disabled?: boolean;
  options: ReadonlyArray<{ value: T; label: string; hint?: string }>;
  onChange: (value: T) => void;
}) {
  return <div className="pipe-choice">
    {label && <span>{label}</span>}
    <div role="group" aria-label={label}>
      {options.map((option) => <button key={option.value} type="button" title={option.hint}
        disabled={disabled} aria-pressed={option.value === value}
        className={option.value === value ? "active" : ""}
        onClick={() => onChange(option.value)}>{option.label}</button>)}
    </div>
  </div>;
}
