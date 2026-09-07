/** Common vocabulary; domain adapters retain their own commands and value types. */
export interface ControlMetadata {
  readonly label: string;
  readonly hint?: string;
  readonly unit?: string;
  readonly min?: number;
  readonly max?: number;
  readonly step?: number;
}

export function normalizeControlNumber(value: unknown, fallback: number,
  control: Pick<ControlMetadata, "min" | "max" | "step">, snap = false): number {
  const candidate = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  const min = control.min ?? -Infinity, max = control.max ?? Infinity;
  if (min > max || (control.step !== undefined && (!Number.isFinite(control.step) || control.step <= 0))) {
    throw new Error("Invalid numeric control bounds or step");
  }
  const clamped = Math.max(min, Math.min(max, candidate));
  const origin = Number.isFinite(min) ? min : 0;
  const stepped = snap && control.step !== undefined
    ? origin + Math.round((clamped - origin) / control.step) * control.step : clamped;
  return Math.max(min, Math.min(max, stepped));
}
