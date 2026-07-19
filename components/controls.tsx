"use client";

import { useEffect, useRef, useState } from "react";
import type { MetricSample } from "@/lib/model";

export function formatNumber(value: number, digits = 3) {
  if (Math.abs(value) < 0.001 && value !== 0) return value.toExponential(2);
  return value.toFixed(digits);
}

interface RangeControlProps {
  label: string;
  unit: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  displayDigits?: number;
  hint?: string;
  /** Shown when the value differs from a preset baseline; click resets. */
  onReset?: () => void;
  modified?: boolean;
}

export function RangeControl({ label, unit, value, min, max, step, onChange, displayDigits = 3, hint, onReset, modified }: RangeControlProps) {
  const [draft, setDraft] = useState<string | null>(null);
  const [rangeDraft, setRangeDraft] = useState<{ base: number; value: number } | null>(null);
  const commit = () => {
    if (draft !== null) {
      const next = Number(draft);
      if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
    }
    setDraft(null);
  };
  const commitRange = (next: number) => {
    const normalized = Math.min(max, Math.max(min, next));
    setRangeDraft(null);
    if (normalized !== value) onChange(normalized);
  };
  const displayedValue = rangeDraft?.base === value ? rangeDraft.value : value;
  return (
    <label className="range-control" title={hint}>
      <span className="control-heading">
        <span>{label}{modified && onReset && <button type="button" className="reset-chip" onClick={(event) => { event.preventDefault(); onReset(); }} title="Reset to preset value">↺</button>}</span>
        {draft !== null
          ? <input
              className="value-edit"
              type="number"
              autoFocus
              value={draft}
              min={min} max={max} step={step}
              onChange={(event) => setDraft(event.target.value)}
              onBlur={commit}
              onKeyDown={(event) => { if (event.key === "Enter") commit(); else if (event.key === "Escape") setDraft(null); }}
            />
          : <output title="Double-click to type a value" onDoubleClick={(event) => { event.preventDefault(); setDraft(String(value)); }}>{formatNumber(displayedValue, displayDigits)} <small>{unit}</small></output>}
      </span>
      <input
        type="range" min={min} max={max} step={step} value={displayedValue}
        onChange={(event) => setRangeDraft({ base: value, value: Number(event.currentTarget.value) })}
        onPointerUp={(event) => commitRange(Number(event.currentTarget.value))}
        onPointerCancel={() => setRangeDraft(null)}
        onBlur={(event) => { if (rangeDraft?.base === value) commitRange(Number(event.currentTarget.value)); }}
        onKeyUp={(event) => { if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"].includes(event.key)) commitRange(Number(event.currentTarget.value)); }}
      />
    </label>
  );
}

export function NumberField({ label, unit, value, step, onChange, min, max }: { label: string; unit?: string; value: number; step?: number; min?: number; max?: number; onChange: (value: number) => void }) {
  // Hold the raw text while the field is being edited: intermediate states
  // like "-", "0.", or "" are not finite numbers, and snapping the controlled
  // value back on every keystroke would swallow the leading minus sign.
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <label className="number-field">
      <span>{label}{unit ? <small> {unit}</small> : null}</span>
      <input
        type="number"
        value={draft ?? value}
        step={step}
        min={min}
        max={max}
        onChange={(event) => {
          setDraft(event.target.value);
          const next = Number(event.target.value);
          if (event.target.value !== "" && Number.isFinite(next)) onChange(next);
        }}
        onBlur={() => setDraft(null)}
      />
    </label>
  );
}

export function Segmented<T extends string>({ value, options, onChange, ariaLabel }: { value: T; options: ReadonlyArray<{ value: T; label: string; title?: string }>; onChange: (value: T) => void; ariaLabel?: string }) {
  return (
    <div className="segmented compact" aria-label={ariaLabel}>
      {options.map((option) => (
        <button key={option.value} title={option.title} className={option.value === value ? "active" : ""} onClick={() => onChange(option.value)}>{option.label}</button>
      ))}
    </div>
  );
}

export function MetricCard({ label, value, unit, tone = "neutral" }: { label: string; value: string; unit?: string; tone?: "neutral" | "good" | "warn" }) {
  return <div className={`metric-card tone-${tone}`}><span>{label}</span><strong>{value}</strong>{unit && <small>{unit}</small>}</div>;
}

export function Sparkline({ samples }: { samples: MetricSample[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
    const height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, width, height);
    ctx.strokeStyle = "rgba(145, 242, 213, .13)";
    ctx.lineWidth = ratio;
    for (let i = 1; i < 4; i += 1) {
      const y = (height * i) / 4;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    if (samples.length < 2) return;
    const values = samples.map((sample) => sample.frame_ms);
    const max = Math.max(16.7, ...values);
    ctx.strokeStyle = "#83f1d1";
    ctx.lineWidth = 1.5 * ratio;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - (value / max) * (height - 4 * ratio) - 2 * ratio;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }, [samples]);
  return <canvas ref={canvasRef} className="sparkline" aria-label="Presentation frame time history" />;
}
