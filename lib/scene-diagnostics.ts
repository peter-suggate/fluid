import type { SceneDescription } from "./model";

/** A stable identifier suitable for reports, baselines, and CI annotations. */
export type DiagnosticId = string;

export type DeepReadonly<Value> =
  Value extends (...args: never[]) => unknown ? Value
    : Value extends readonly (infer Item)[] ? readonly DeepReadonly<Item>[]
      : Value extends object ? { readonly [Key in keyof Value]: DeepReadonly<Value[Key]> }
        : Value;

/** Serializable options passed to a runner-owned diagnostic collector. */
export type DiagnosticOptionValue =
  | null
  | boolean
  | number
  | string
  | readonly DiagnosticOptionValue[]
  | { readonly [key: string]: DiagnosticOptionValue };

/**
 * A request for evidence. The scene describes the request; the runner decides
 * how the named diagnostic is collected.
 */
export interface DiagnosticRequest<
  Options extends Readonly<Record<string, DiagnosticOptionValue>> = Readonly<Record<string, DiagnosticOptionValue>>,
> {
  readonly id: DiagnosticId;
  readonly options?: Options;
}

export type DiagnosticPathSegment = string | number;
export type DiagnosticPath = string | readonly DiagnosticPathSegment[];

/** Evidence emitted by one solver/method run. */
export interface MethodDiagnosticEvidence {
  readonly diagnostics: Readonly<Record<string, unknown>>;
}

/** Evidence is deliberately keyed by method rather than by scenario. */
export interface SceneDiagnosticEvidence {
  readonly methods: Readonly<Record<string, MethodDiagnosticEvidence>>;
}

export type DiagnosticMethodSelector = string | "*";

export interface DiagnosticValueSelector {
  /** A concrete method ID, or `*` to apply the check independently to all methods. */
  readonly method: DiagnosticMethodSelector;
  /** Dot-separated shorthand or explicit path segments within `diagnostics`. */
  readonly path: DiagnosticPath;
}

export type DiagnosticSeverity = "error" | "warning";

interface SceneDiagnosticCheckBase {
  readonly id: DiagnosticId;
  readonly severity?: DiagnosticSeverity;
  readonly select: DiagnosticValueSelector;
  readonly description?: string;
}

export type ScalarDiagnosticOperator =
  | "equal"
  | "not-equal"
  | "less-than"
  | "less-than-or-equal"
  | "greater-than"
  | "greater-than-or-equal"
  | "finite";

export interface ScalarDiagnosticCheck extends SceneDiagnosticCheckBase {
  readonly kind: "scalar";
  readonly operator: ScalarDiagnosticOperator;
  /** Omit only for the `finite` operator. */
  readonly expected?: number;
  /** Absolute tolerance used by `equal` and `not-equal`. Defaults to zero. */
  readonly tolerance?: number;
}

export interface PathDiagnosticCheck extends SceneDiagnosticCheckBase {
  readonly kind: "path";
  readonly operator: "present" | "absent";
}

export type SceneDiagnosticCheck = ScalarDiagnosticCheck | PathDiagnosticCheck;

export interface DiagnosticFinding {
  /** Stable declarative check ID, or a hook ID qualified with a hook finding ID. */
  readonly checkId: DiagnosticId;
  readonly passed: boolean;
  readonly severity: DiagnosticSeverity;
  readonly message: string;
  readonly method?: string;
  readonly path?: readonly DiagnosticPathSegment[];
  readonly actual?: unknown;
  readonly expected?: unknown;
}

/** A hook-owned finding ID is qualified by the evaluator with the hook ID. */
export interface SceneDiagnosticHookFinding {
  readonly id: DiagnosticId;
  readonly passed: boolean;
  readonly severity?: DiagnosticSeverity;
  readonly message: string;
  readonly method?: string;
  readonly path?: DiagnosticPath;
  readonly actual?: unknown;
  readonly expected?: unknown;
}

export interface SceneDiagnosticHookContext {
  readonly scene: DeepReadonly<SceneDescription>;
  readonly evidence: SceneDiagnosticEvidence;
  readonly getValue: typeof getDiagnosticValue;
}

/**
 * Pure post-processing escape hatch for assertions that cannot be expressed as
 * scalar/path checks. Hooks have no solver, GPU, logger, or environment access.
 */
export interface SceneDiagnosticHook {
  readonly id: DiagnosticId;
  readonly evaluate: (
    context: SceneDiagnosticHookContext,
  ) => readonly SceneDiagnosticHookFinding[];
}

export interface SceneDiagnosticsContract {
  readonly diagnostics: readonly DiagnosticRequest[];
  readonly checks: readonly SceneDiagnosticCheck[];
  readonly hooks?: readonly SceneDiagnosticHook[];
}

export interface SceneDiagnosticEvaluation {
  readonly passed: boolean;
  readonly findings: readonly DiagnosticFinding[];
  readonly failedErrorCount: number;
  readonly failedWarningCount: number;
}

export type DeclarativeDiagnosticComparisonOperator =
  | "equal"
  | "at-least"
  | "at-most"
  | "between"
  | "present"
  | "finite";

export interface DeclarativeDiagnosticReference {
  readonly selector: string;
}

export interface DeclarativeDiagnosticPredicate {
  readonly metric: string;
  readonly operator: DeclarativeDiagnosticComparisonOperator;
  readonly expected?: unknown;
}

/**
 * A compact serializable rule for suites loaded from a scene catalog. Metrics
 * are rooted in normalized evidence and may use `methods.*` to evaluate every
 * selected method independently. A trailing `.abs` is a value transform.
 */
export interface DeclarativeDiagnosticRule extends DeclarativeDiagnosticPredicate {
  readonly id: DiagnosticId;
  readonly methods?: readonly string[];
  readonly when?: readonly DeclarativeDiagnosticPredicate[];
}

export interface ResolvedDiagnosticValue {
  readonly found: boolean;
  readonly value: unknown;
  readonly path: readonly DiagnosticPathSegment[];
}

export function diagnosticRequest<
  Options extends Readonly<Record<string, DiagnosticOptionValue>>,
>(id: DiagnosticId, options?: Options): DiagnosticRequest<Options> {
  assertStableDiagnosticId(id, "diagnostic request");
  return options === undefined ? { id } : { id, options };
}

export function scalarCheck(check: Omit<ScalarDiagnosticCheck, "kind">): ScalarDiagnosticCheck {
  return { kind: "scalar", ...check };
}

export function pathCheck(check: Omit<PathDiagnosticCheck, "kind">): PathDiagnosticCheck {
  return { kind: "path", ...check };
}

export function defineSceneDiagnosticHook(hook: SceneDiagnosticHook): SceneDiagnosticHook {
  assertStableDiagnosticId(hook.id, "diagnostic hook");
  return hook;
}

/**
 * Defines and validates a scene-owned diagnostics contract without attaching it
 * to the serializable SceneDescription.
 */
export function defineSceneDiagnostics<const Contract extends SceneDiagnosticsContract>(
  contract: Contract,
): Contract {
  const requestIds = new Set<string>();
  for (const request of contract.diagnostics) {
    assertStableDiagnosticId(request.id, "diagnostic request");
    if (requestIds.has(request.id)) {
      throw new Error(`Duplicate diagnostic request ID: ${request.id}`);
    }
    requestIds.add(request.id);
  }

  const checkIds = new Set<string>();
  for (const check of contract.checks) {
    assertStableDiagnosticId(check.id, "diagnostic check");
    if (checkIds.has(check.id)) {
      throw new Error(`Duplicate diagnostic check ID: ${check.id}`);
    }
    checkIds.add(check.id);
    validateSelector(check.select, check.id);
    if (check.kind === "scalar") validateScalarCheck(check);
  }

  const hookIds = new Set<string>();
  for (const hook of contract.hooks ?? []) {
    assertStableDiagnosticId(hook.id, "diagnostic hook");
    if (hookIds.has(hook.id)) {
      throw new Error(`Duplicate diagnostic hook ID: ${hook.id}`);
    }
    hookIds.add(hook.id);
  }

  return contract;
}

export function getDiagnosticValue(
  evidence: SceneDiagnosticEvidence,
  method: string,
  path: DiagnosticPath,
): ResolvedDiagnosticValue {
  const segments = normalizeDiagnosticPath(path);
  const methodEvidence = evidence.methods[method];
  if (methodEvidence === undefined) return { found: false, value: undefined, path: segments };

  let value: unknown = methodEvidence.diagnostics;
  for (const segment of segments) {
    if (typeof segment === "number") {
      if (!Array.isArray(value) || segment < 0 || !Number.isInteger(segment) || segment >= value.length) {
        return { found: false, value: undefined, path: segments };
      }
      value = value[segment];
      continue;
    }

    if (value === null || typeof value !== "object" || !Object.prototype.hasOwnProperty.call(value, segment)) {
      return { found: false, value: undefined, path: segments };
    }
    value = (value as Readonly<Record<string, unknown>>)[segment];
  }
  return { found: true, value, path: segments };
}

export function evaluateSceneDiagnostics(
  contract: SceneDiagnosticsContract,
  input: {
    readonly scene: DeepReadonly<SceneDescription>;
    readonly evidence: SceneDiagnosticEvidence;
  },
): SceneDiagnosticEvaluation {
  // Validate here as well as at definition time so loaded/generated contracts
  // cannot silently produce unstable reports.
  defineSceneDiagnostics(contract);

  const findings: DiagnosticFinding[] = [];
  const methodIds = Object.keys(input.evidence.methods).sort();

  for (const check of contract.checks) {
    const selectedMethods = check.select.method === "*"
      ? methodIds
      : [check.select.method];
    if (selectedMethods.length === 0) {
      findings.push({
        checkId: check.id,
        passed: false,
        severity: check.severity ?? "error",
        message: check.description ?? "No method evidence was available for the check.",
        path: normalizeDiagnosticPath(check.select.path),
      });
      continue;
    }
    for (const method of selectedMethods) {
      findings.push(evaluateCheckForMethod(check, method, input.evidence));
    }
  }

  for (const hook of contract.hooks ?? []) {
    let hookFindings: readonly SceneDiagnosticHookFinding[];
    try {
      hookFindings = hook.evaluate({
        scene: input.scene,
        evidence: input.evidence,
        getValue: getDiagnosticValue,
      });
    } catch (error) {
      findings.push({
        checkId: `${hook.id}.execution`,
        passed: false,
        severity: "error",
        message: `Diagnostic hook ${hook.id} threw: ${formatError(error)}`,
      });
      continue;
    }

    const findingIds = new Set<string>();
    for (const finding of hookFindings) {
      assertStableDiagnosticId(finding.id, `finding from hook ${hook.id}`);
      if (findingIds.has(finding.id)) {
        throw new Error(`Duplicate finding ID from diagnostic hook ${hook.id}: ${finding.id}`);
      }
      findingIds.add(finding.id);
      findings.push({
        checkId: `${hook.id}.${finding.id}`,
        passed: finding.passed,
        severity: finding.severity ?? "error",
        message: finding.message,
        ...(finding.method === undefined ? {} : { method: finding.method }),
        ...(finding.path === undefined ? {} : { path: normalizeDiagnosticPath(finding.path) }),
        ...(finding.actual === undefined ? {} : { actual: finding.actual }),
        ...(finding.expected === undefined ? {} : { expected: finding.expected }),
      });
    }
  }

  const failedErrorCount = findings.filter((finding) => !finding.passed && finding.severity === "error").length;
  const failedWarningCount = findings.filter((finding) => !finding.passed && finding.severity === "warning").length;
  return {
    passed: failedErrorCount === 0,
    findings,
    failedErrorCount,
    failedWarningCount,
  };
}

export function evaluateDeclarativeDiagnosticRules(
  rules: readonly DeclarativeDiagnosticRule[],
  evidence: Readonly<Record<string, unknown>>,
): SceneDiagnosticEvaluation {
  const findings: DiagnosticFinding[] = [];
  const methodEvidence = isRecord(evidence.methods) ? evidence.methods : {};

  for (const rule of rules) {
    assertStableDiagnosticId(rule.id, "diagnostic rule");
    const wildcard = rule.metric.startsWith("methods.*.");
    const selectedMethods = wildcard
      ? Object.keys(methodEvidence).filter((method) => rule.methods === undefined || rule.methods.includes(method)).sort()
      : [methodFromMetric(rule.metric)].filter((method): method is string => method !== undefined
        && (rule.methods === undefined || rule.methods.includes(method)));
    if (selectedMethods.length === 0) {
      findings.push({
        checkId: rule.id,
        passed: false,
        severity: "error",
        message: `${rule.id} selected no method evidence.`,
      });
      continue;
    }

    for (const method of selectedMethods) {
      if (!(rule.when ?? []).every((predicate) => predicatePasses(predicate, evidence, method))) continue;
      const metric = specializeMethodWildcard(rule.metric, method);
      const resolved = resolveRootValue(evidence, metric);
      const expected = isDiagnosticReference(rule.expected)
        ? resolveRootValue(evidence, specializeMethodWildcard(rule.expected.selector, method)).value
        : rule.expected;
      const passed = resolved.found && compareDeclarativeValue(resolved.value, rule.operator, expected);
      findings.push({
        checkId: rule.id,
        passed,
        severity: "error",
        message: `${metric} must satisfy ${rule.operator}${expected === undefined ? "" : ` ${JSON.stringify(expected)}`}.`,
        method,
        path: resolved.path,
        actual: resolved.value,
        expected,
      });
    }
  }

  const failedErrorCount = findings.filter((finding) => !finding.passed).length;
  return { passed: failedErrorCount === 0, findings, failedErrorCount, failedWarningCount: 0 };
}

function predicatePasses(
  predicate: DeclarativeDiagnosticPredicate,
  evidence: Readonly<Record<string, unknown>>,
  method: string,
): boolean {
  const resolved = resolveRootValue(evidence, specializeMethodWildcard(predicate.metric, method));
  const expected = isDiagnosticReference(predicate.expected)
    ? resolveRootValue(evidence, specializeMethodWildcard(predicate.expected.selector, method)).value
    : predicate.expected;
  return resolved.found && compareDeclarativeValue(resolved.value, predicate.operator, expected);
}

function compareDeclarativeValue(
  actual: unknown,
  operator: DeclarativeDiagnosticComparisonOperator,
  expected: unknown,
): boolean {
  switch (operator) {
    case "equal": return deepEqualDiagnosticValue(actual, expected);
    case "at-least": return typeof actual === "number" && typeof expected === "number" && actual >= expected;
    case "at-most": return typeof actual === "number" && typeof expected === "number" && actual <= expected;
    case "between": return typeof actual === "number" && Array.isArray(expected) && expected.length === 2
      && typeof expected[0] === "number" && typeof expected[1] === "number"
      && actual >= expected[0] && actual <= expected[1];
    case "present": return actual !== undefined;
    case "finite": return typeof actual === "number" && Number.isFinite(actual);
  }
}

function resolveRootValue(
  root: Readonly<Record<string, unknown>>,
  metric: string,
): ResolvedDiagnosticValue {
  const segments = normalizeDiagnosticPath(metric);
  const transformAbsolute = segments.at(-1) === "abs";
  const sourceSegments = transformAbsolute ? segments.slice(0, -1) : segments;
  let value: unknown = root;
  for (const segment of sourceSegments) {
    if (typeof segment === "number") {
      if (!Array.isArray(value) || segment >= value.length) return { found: false, value: undefined, path: segments };
      value = value[segment];
    } else {
      if (!isRecord(value) || !Object.prototype.hasOwnProperty.call(value, segment)) {
        return { found: false, value: undefined, path: segments };
      }
      value = value[segment];
    }
  }
  if (transformAbsolute) {
    if (typeof value !== "number") return { found: false, value: undefined, path: segments };
    value = Math.abs(value);
  }
  return { found: true, value, path: segments };
}

function methodFromMetric(metric: string): string | undefined {
  const match = /^methods\.([^.]+)\./.exec(metric);
  return match?.[1];
}

function specializeMethodWildcard(metric: string, method: string): string {
  return metric.replace(/^methods\.\*\./, `methods.${method}.`);
}

function isDiagnosticReference(value: unknown): value is DeclarativeDiagnosticReference {
  return isRecord(value) && typeof value.selector === "string";
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function deepEqualDiagnosticValue(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (Array.isArray(left) && Array.isArray(right)) {
    return left.length === right.length && left.every((value, index) => deepEqualDiagnosticValue(value, right[index]));
  }
  if (isRecord(left) && isRecord(right)) {
    const leftKeys = Object.keys(left).sort(), rightKeys = Object.keys(right).sort();
    return leftKeys.length === rightKeys.length
      && leftKeys.every((key, index) => key === rightKeys[index]
        && deepEqualDiagnosticValue(left[key], right[key]));
  }
  return false;
}

function evaluateCheckForMethod(
  check: SceneDiagnosticCheck,
  method: string,
  evidence: SceneDiagnosticEvidence,
): DiagnosticFinding {
  const resolved = getDiagnosticValue(evidence, method, check.select.path);
  const severity = check.severity ?? "error";
  if (check.kind === "path") {
    const passed = check.operator === "present" ? resolved.found : !resolved.found;
    return {
      checkId: check.id,
      passed,
      severity,
      message: check.description ?? `${formatSelector(method, resolved.path)} must be ${check.operator}.`,
      method,
      path: resolved.path,
      actual: resolved.found ? resolved.value : undefined,
      expected: check.operator,
    };
  }

  if (!resolved.found || typeof resolved.value !== "number") {
    return {
      checkId: check.id,
      passed: false,
      severity,
      message: check.description ?? `${formatSelector(method, resolved.path)} must be a numeric scalar.`,
      method,
      path: resolved.path,
      actual: resolved.value,
      expected: scalarExpected(check),
    };
  }

  return {
    checkId: check.id,
    passed: compareScalar(resolved.value, check),
    severity,
    message: check.description ?? `${formatSelector(method, resolved.path)} must be ${scalarExpected(check)}.`,
    method,
    path: resolved.path,
    actual: resolved.value,
    expected: scalarExpected(check),
  };
}

function compareScalar(actual: number, check: ScalarDiagnosticCheck): boolean {
  const expected = check.expected as number;
  const tolerance = check.tolerance ?? 0;
  switch (check.operator) {
    case "equal": return Math.abs(actual - expected) <= tolerance;
    case "not-equal": return Math.abs(actual - expected) > tolerance;
    case "less-than": return actual < expected;
    case "less-than-or-equal": return actual <= expected;
    case "greater-than": return actual > expected;
    case "greater-than-or-equal": return actual >= expected;
    case "finite": return Number.isFinite(actual);
  }
}

function validateScalarCheck(check: ScalarDiagnosticCheck): void {
  if (check.operator === "finite") {
    if (check.expected !== undefined) {
      throw new Error(`Scalar check ${check.id} cannot set expected for the finite operator.`);
    }
  } else if (typeof check.expected !== "number" || !Number.isFinite(check.expected)) {
    throw new Error(`Scalar check ${check.id} requires a finite expected value.`);
  }
  if (check.tolerance !== undefined && (!Number.isFinite(check.tolerance) || check.tolerance < 0)) {
    throw new Error(`Scalar check ${check.id} requires a non-negative finite tolerance.`);
  }
}

function validateSelector(selector: DiagnosticValueSelector, checkId: string): void {
  if (selector.method !== "*") assertStableDiagnosticId(selector.method, `method selector in ${checkId}`);
  normalizeDiagnosticPath(selector.path);
}

function normalizeDiagnosticPath(path: DiagnosticPath): readonly DiagnosticPathSegment[] {
  const segments = typeof path === "string"
    ? path.split(".")
    : [...path];
  if (segments.length === 0 || segments.some((segment) => (
    typeof segment === "string" ? segment.length === 0 : !Number.isInteger(segment) || segment < 0
  ))) {
    throw new Error(`Invalid diagnostic path: ${JSON.stringify(path)}`);
  }
  return segments;
}

function assertStableDiagnosticId(id: string, subject: string): void {
  if (!/^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(id)) {
    throw new Error(`Invalid ${subject} ID: ${JSON.stringify(id)}`);
  }
}

function scalarExpected(check: ScalarDiagnosticCheck): unknown {
  if (check.operator === "finite") return "finite";
  return {
    operator: check.operator,
    value: check.expected,
    ...(check.tolerance === undefined ? {} : { tolerance: check.tolerance }),
  };
}

function formatSelector(method: string, path: readonly DiagnosticPathSegment[]): string {
  return `${method}.${path.join(".")}`;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
