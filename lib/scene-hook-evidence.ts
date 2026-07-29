import type {
  DiagnosticSeverity,
  MethodDiagnosticEvidence,
  SceneDiagnosticEvidence,
  SceneDiagnosticHookFinding,
} from "./scene-diagnostics";

export type UnknownRecord = Readonly<Record<string, unknown>>;

export function recordValue(value: unknown): UnknownRecord | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as UnknownRecord
    : undefined;
}

export function recordPath(value: unknown, ...path: readonly string[]): UnknownRecord | undefined {
  let current = recordValue(value);
  for (const segment of path) {
    if (!current) return undefined;
    current = recordValue(current[segment]);
  }
  return current;
}

export function arrayPath(value: unknown, ...path: readonly string[]): readonly unknown[] | undefined {
  let current: unknown = value;
  for (const segment of path) {
    const record = recordValue(current);
    if (!record) return undefined;
    current = record[segment];
  }
  return Array.isArray(current) ? current : undefined;
}

export function numberValue(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

export function numberPath(value: unknown, ...path: readonly string[]): number | undefined {
  let current: unknown = value;
  for (const segment of path) {
    const record = recordValue(current);
    if (!record) return undefined;
    current = record[segment];
  }
  return numberValue(current);
}

export function booleanPath(value: unknown, ...path: readonly string[]): boolean | undefined {
  let current: unknown = value;
  for (const segment of path) {
    const record = recordValue(current);
    if (!record) return undefined;
    current = record[segment];
  }
  return typeof current === "boolean" ? current : undefined;
}

export function methodDiagnostics(
  evidence: SceneDiagnosticEvidence,
  method: string,
): UnknownRecord | undefined {
  return recordValue(evidence.methods[method]?.diagnostics);
}

export function selectedMethodDiagnostics(
  evidence: SceneDiagnosticEvidence,
  methods?: readonly string[],
): readonly (readonly [string, UnknownRecord])[] {
  const selected = methods ?? Object.keys(evidence.methods).sort();
  const result: Array<readonly [string, UnknownRecord]> = [];
  for (const method of selected) {
    const diagnostics = methodDiagnostics(evidence, method);
    if (diagnostics) result.push([method, diagnostics]);
  }
  return result;
}

export function hookFinding(input: {
  id: string;
  passed: boolean;
  message: string;
  method?: string;
  severity?: DiagnosticSeverity;
  actual?: unknown;
  expected?: unknown;
}): SceneDiagnosticHookFinding {
  return {
    id: input.id,
    passed: input.passed,
    severity: input.severity ?? "error",
    message: input.message,
    ...(input.method === undefined ? {} : { method: input.method }),
    ...(input.actual === undefined ? {} : { actual: input.actual }),
    ...(input.expected === undefined ? {} : { expected: input.expected }),
  };
}

export function gridFromDiagnostics(diagnostics: UnknownRecord): readonly [number, number, number] | undefined {
  const field = recordValue(diagnostics.field);
  const grid = field?.grid ?? diagnostics.grid;
  return Array.isArray(grid) && grid.length === 3
    && grid.every((dimension) => Number.isInteger(dimension) && dimension > 0)
    ? grid as unknown as readonly [number, number, number]
    : undefined;
}

export function runTime(diagnostics: UnknownRecord): number | undefined {
  return numberPath(diagnostics, "run", "simulatedTime_s")
    ?? numberPath(diagnostics, "solver", "simulatedTime_s")
    ?? numberValue(diagnostics.simulatedTime_s);
}

export function runSteps(diagnostics: UnknownRecord): number | undefined {
  return numberPath(diagnostics, "run", "steps") ?? numberValue(diagnostics.steps);
}

export function fieldCheckpoints(diagnostics: UnknownRecord): readonly unknown[] {
  return arrayPath(diagnostics, "field", "checkpoints")
    ?? (Array.isArray(diagnostics.checkpoints) ? diagnostics.checkpoints : []);
}

export function methodEvidence(
  diagnostics: UnknownRecord,
): MethodDiagnosticEvidence {
  return { diagnostics };
}
