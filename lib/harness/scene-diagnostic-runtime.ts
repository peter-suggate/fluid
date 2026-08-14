import type { SceneDescription } from "../core/model";
import {
  evaluateDeclarativeDiagnosticRules,
  type DeepReadonly,
  type DiagnosticFinding,
  type DiagnosticPath,
  type DiagnosticSeverity,
  type SceneDiagnosticEvaluation,
} from "./scene-diagnostics";
import type {
  SceneWebGPUHookId,
  SceneWebGPUDiagnosticHook,
  SceneWebGPUDiagnosticPack,
  SceneWebGPUDiagnosticPackId,
  SceneWebGPUSmokeLane,
  WebGPUSmokeMethodId,
} from "./scene-webgpu-smoke";

/** Normalized evidence for one method, independent of its solver implementation. */
export interface NormalizedMethodDiagnosticEvidence {
  /** Explicit evidence capabilities collected for this run. */
  readonly available: readonly string[];
  /**
   * Normalized namespaces such as run, solver, field, stability, energy,
   * sparse, velocity, and raster.
   */
  readonly diagnostics: Readonly<Record<string, unknown>>;
}

export interface NormalizedSceneDiagnosticEvidence {
  readonly methods: Readonly<Record<string, NormalizedMethodDiagnosticEvidence>>;
  /** Optional scene-wide observations that do not belong to one method. */
  readonly shared?: Readonly<Record<string, unknown>>;
}

export interface RuntimeDiagnosticFinding {
  /** Stable local ID. The runtime prefixes this with the pack/hook declaration ID. */
  readonly id: string;
  readonly passed: boolean;
  readonly severity?: DiagnosticSeverity;
  readonly message: string;
  readonly method?: string;
  readonly path?: DiagnosticPath;
  readonly actual?: unknown;
  readonly expected?: unknown;
}

interface SceneDiagnosticImplementationContext<Declaration> {
  readonly scene: DeepReadonly<SceneDescription>;
  readonly lane: DeepReadonly<SceneWebGPUSmokeLane>;
  readonly declaration: DeepReadonly<Declaration>;
  readonly parameters: Readonly<Record<string, unknown>>;
  readonly selectedMethods: readonly WebGPUSmokeMethodId[];
  readonly evidence: NormalizedSceneDiagnosticEvidence;
  readonly getMethod: (method: string) => NormalizedMethodDiagnosticEvidence | undefined;
  readonly hasEvidence: (method: string, requirement: string) => boolean;
}

export type DiagnosticPackImplementationContext = SceneDiagnosticImplementationContext<SceneWebGPUDiagnosticPack>;
export type SceneHookImplementationContext = SceneDiagnosticImplementationContext<SceneWebGPUDiagnosticHook>;

export interface DiagnosticPackImplementation<Id extends SceneWebGPUDiagnosticPackId = SceneWebGPUDiagnosticPackId> {
  readonly id: Id;
  /** Evidence owned by the reusable pack implementation. */
  readonly requires?: readonly string[];
  readonly evaluate: (
    context: DiagnosticPackImplementationContext,
  ) => readonly RuntimeDiagnosticFinding[];
}

export interface SceneHookImplementation<Id extends SceneWebGPUHookId = SceneWebGPUHookId> {
  readonly id: Id;
  /**
   * Optional implementation audit. Every entry must also be declared by the
   * scene hook's `requires` list or execution fails closed.
   */
  readonly requires?: readonly string[];
  readonly evaluate: (
    context: SceneHookImplementationContext,
  ) => readonly RuntimeDiagnosticFinding[];
}

export type DiagnosticPackRegistry = Partial<{
  readonly [Id in SceneWebGPUDiagnosticPackId]: DiagnosticPackImplementation<Id>;
}>;

export type SceneHookRegistry = Partial<{
  readonly [Id in SceneWebGPUHookId]: SceneHookImplementation<Id>;
}>;

export interface SceneDiagnosticRuntimeRegistry {
  readonly packs: DiagnosticPackRegistry;
  readonly hooks: SceneHookRegistry;
}

export interface SceneDiagnosticRuntimeInput {
  readonly scene: DeepReadonly<SceneDescription>;
  readonly lane: DeepReadonly<SceneWebGPUSmokeLane>;
  readonly evidence: NormalizedSceneDiagnosticEvidence;
}

export interface SceneDiagnosticRuntimeEvaluation extends SceneDiagnosticEvaluation {
  readonly runtimeFindings: readonly DiagnosticFinding[];
  readonly acceptanceFindings: readonly DiagnosticFinding[];
  readonly packFindings: readonly DiagnosticFinding[];
  readonly hookFindings: readonly DiagnosticFinding[];
}

export interface SceneDiagnosticRuntime {
  readonly evaluate: (input: SceneDiagnosticRuntimeInput) => SceneDiagnosticRuntimeEvaluation;
}

/**
 * Creates a data-driven diagnostics runtime. Partial registries are accepted so
 * plugin/build composition remains possible; unresolved declarations become
 * ordinary failing findings when a lane executes.
 */
export function createSceneDiagnosticRuntime(
  registry: SceneDiagnosticRuntimeRegistry,
): SceneDiagnosticRuntime {
  validateRegistry(registry);
  const stableRegistry: SceneDiagnosticRuntimeRegistry = Object.freeze({
    packs: Object.freeze({ ...registry.packs }),
    hooks: Object.freeze({ ...registry.hooks }),
  });
  return Object.freeze({
    evaluate: (input: SceneDiagnosticRuntimeInput) => evaluateSceneDiagnosticLane(stableRegistry, input),
  });
}

export function defineDiagnosticPackImplementation<
  const Id extends SceneWebGPUDiagnosticPackId,
>(implementation: DiagnosticPackImplementation<Id>): DiagnosticPackImplementation<Id> {
  return implementation;
}

export function defineSceneHookImplementation<
  const Id extends SceneWebGPUHookId,
>(implementation: SceneHookImplementation<Id>): SceneHookImplementation<Id> {
  return implementation;
}

export function evaluateSceneDiagnosticLane(
  registry: SceneDiagnosticRuntimeRegistry,
  input: SceneDiagnosticRuntimeInput,
): SceneDiagnosticRuntimeEvaluation {
  const runtimeFindings = input.lane.methods.flatMap(({ id }) => (
    input.evidence.methods[id] === undefined
      ? [failure(
        "runtime.method-evidence",
        `The ${input.lane.id} lane has no normalized evidence for ${id}.`,
        { method: id, expected: "normalized method evidence" },
      )]
      : []
  ));
  const acceptance = evaluateDeclarativeDiagnosticRules(
    input.lane.acceptance,
    declarativeEvidence(input.evidence, input.lane.methods.map(({ id }) => id)),
  );
  const packFindings = input.lane.diagnostics.flatMap((declaration) => evaluateDeclaration(
    "pack",
    declaration,
    registry.packs[declaration.id],
    input,
  ));
  const hookFindings = input.lane.hooks.flatMap((declaration) => evaluateDeclaration(
    "hook",
    declaration,
    registry.hooks[declaration.id],
    input,
  ));
  const findings = [...runtimeFindings, ...acceptance.findings, ...packFindings, ...hookFindings];
  const failedErrorCount = findings.filter((finding) => !finding.passed && finding.severity === "error").length;
  const failedWarningCount = findings.filter((finding) => !finding.passed && finding.severity === "warning").length;
  return {
    passed: failedErrorCount === 0,
    findings,
    failedErrorCount,
    failedWarningCount,
    runtimeFindings,
    acceptanceFindings: acceptance.findings,
    packFindings,
    hookFindings,
  };
}

type RuntimeDeclaration = SceneWebGPUDiagnosticPack | SceneWebGPUDiagnosticHook;
type RuntimeImplementation = DiagnosticPackImplementation | SceneHookImplementation;

function evaluateDeclaration(
  kind: "pack" | "hook",
  declaration: RuntimeDeclaration,
  implementation: RuntimeImplementation | undefined,
  input: SceneDiagnosticRuntimeInput,
): DiagnosticFinding[] {
  const prefix = `${kind}.${declaration.id}`;
  if (implementation === undefined) {
    return [failure(`${prefix}.implementation`, `No implementation is registered for ${prefix}.`)];
  }

  const selectedMethods = selectMethods(declaration, input.lane);
  if (selectedMethods.length === 0) {
    return [failure(`${prefix}.methods`, `${prefix} selected no lane methods.`)];
  }

  const declaredRequirements = kind === "hook"
    ? (declaration as SceneWebGPUDiagnosticHook).requires
    : implementation.requires ?? [];
  const implementationRequirements = implementation.requires ?? [];
  if (kind === "hook") {
    const undeclared = implementationRequirements.filter((requirement) => !declaredRequirements.includes(requirement));
    if (undeclared.length > 0) {
      return [failure(
        `${prefix}.requirements-declaration`,
        `${prefix} implementation requires evidence not declared by the scene: ${undeclared.join(", ")}.`,
        { expected: implementationRequirements, actual: declaredRequirements },
      )];
    }
  }

  const requirementFindings: DiagnosticFinding[] = [];
  for (const method of selectedMethods) {
    const methodEvidence = input.evidence.methods[method];
    if (methodEvidence === undefined) {
      requirementFindings.push(failure(
        `${prefix}.method-evidence`,
        `${prefix} has no normalized evidence for ${method}.`,
        { method, expected: "normalized method evidence" },
      ));
      continue;
    }
    const missing = declaredRequirements.filter((requirement) => !methodEvidence.available.includes(requirement));
    if (missing.length > 0) {
      requirementFindings.push(failure(
        `${prefix}.requirements`,
        `${prefix} is missing required ${method} evidence: ${missing.join(", ")}.`,
        { method, expected: declaredRequirements, actual: methodEvidence.available },
      ));
    }
  }
  if (requirementFindings.length > 0) return requirementFindings;

  const context: SceneDiagnosticImplementationContext<RuntimeDeclaration> = {
    scene: input.scene,
    lane: input.lane,
    declaration,
    parameters: declaration.parameters ?? {},
    selectedMethods,
    evidence: input.evidence,
    getMethod: (method: string) => input.evidence.methods[method],
    hasEvidence: (method: string, requirement: string) => (
      input.evidence.methods[method]?.available.includes(requirement) ?? false
    ),
  };

  let localFindings: readonly RuntimeDiagnosticFinding[];
  try {
    // The declaration and implementation kinds are paired by their registry;
    // after runtime ID validation they share this common context shape.
    const evaluate = implementation.evaluate as (
      value: SceneDiagnosticImplementationContext<RuntimeDeclaration>,
    ) => readonly RuntimeDiagnosticFinding[];
    localFindings = evaluate(context);
  } catch (error) {
    return [failure(
      `${prefix}.execution`,
      `${prefix} implementation threw: ${error instanceof Error ? error.message : String(error)}`,
    )];
  }

  const identities = new Set<string>();
  const findings: DiagnosticFinding[] = [];
  for (const finding of localFindings) {
    if (!isStableId(finding.id)) {
      findings.push(failure(
        `${prefix}.finding-id`,
        `${prefix} returned an invalid local finding ID: ${JSON.stringify(finding.id)}.`,
      ));
      continue;
    }
    if (typeof finding.passed !== "boolean"
      || typeof finding.message !== "string"
      || (finding.severity !== undefined && finding.severity !== "error" && finding.severity !== "warning")
      || (finding.method !== undefined && !selectedMethods.includes(finding.method as WebGPUSmokeMethodId))) {
      findings.push(failure(
        `${prefix}.finding-contract`,
        `${prefix} returned malformed finding ${finding.id}.`,
        { actual: finding },
      ));
      continue;
    }
    let path: readonly (string | number)[] | undefined;
    try {
      path = finding.path === undefined ? undefined : normalizePath(finding.path);
    } catch (error) {
      findings.push(failure(
        `${prefix}.finding-contract`,
        `${prefix} returned an invalid path for finding ${finding.id}: ${error instanceof Error ? error.message : String(error)}`,
      ));
      continue;
    }
    const identity = `${finding.id}\0${finding.method ?? ""}`;
    if (identities.has(identity)) {
      findings.push(failure(
        `${prefix}.duplicate-finding`,
        `${prefix} returned duplicate finding ${finding.id}${finding.method === undefined ? "" : ` for ${finding.method}`}.`,
        { method: finding.method },
      ));
      continue;
    }
    identities.add(identity);
    findings.push({
      checkId: `${prefix}.${finding.id}`,
      passed: finding.passed,
      severity: finding.severity ?? "error",
      message: finding.message,
      ...(finding.method === undefined ? {} : { method: finding.method }),
      ...(path === undefined ? {} : { path }),
      ...(finding.actual === undefined ? {} : { actual: finding.actual }),
      ...(finding.expected === undefined ? {} : { expected: finding.expected }),
    });
  }
  return findings;
}

function selectMethods(
  declaration: RuntimeDeclaration,
  lane: DeepReadonly<SceneWebGPUSmokeLane>,
): WebGPUSmokeMethodId[] {
  const laneMethods = lane.methods.map(({ id }) => id);
  return (declaration.methods ?? laneMethods).filter((method) => laneMethods.includes(method));
}

function declarativeEvidence(
  evidence: NormalizedSceneDiagnosticEvidence,
  laneMethods: readonly WebGPUSmokeMethodId[],
): Readonly<Record<string, unknown>> {
  return {
    ...(evidence.shared ?? {}),
    methods: Object.fromEntries(
      laneMethods.flatMap((method) => {
        const entry = evidence.methods[method];
        return entry === undefined ? [] : [[method, entry.diagnostics] as const];
      }),
    ),
  };
}

function validateRegistry(registry: SceneDiagnosticRuntimeRegistry): void {
  validateImplementations("pack", registry.packs);
  validateImplementations("hook", registry.hooks);
}

function validateImplementations(
  kind: "pack" | "hook",
  implementations: Readonly<Record<string, RuntimeImplementation | undefined>>,
): void {
  for (const [key, implementation] of Object.entries(implementations)) {
    if (implementation === undefined) continue;
    if (implementation.id !== key) {
      throw new Error(`Registered ${kind} key ${key} differs from implementation ID ${implementation.id}.`);
    }
    if (!isStableId(key)) throw new Error(`Invalid registered ${kind} ID: ${JSON.stringify(key)}.`);
    const requirements = implementation.requires ?? [];
    if (new Set(requirements).size !== requirements.length || requirements.some((value) => value.trim().length === 0)) {
      throw new Error(`${kind}.${key} has invalid or duplicate evidence requirements.`);
    }
  }
}

function failure(
  checkId: string,
  message: string,
  details: { readonly method?: string; readonly actual?: unknown; readonly expected?: unknown } = {},
): DiagnosticFinding {
  return {
    checkId,
    passed: false,
    severity: "error",
    message,
    ...details,
  };
}

function normalizePath(path: DiagnosticPath): readonly (string | number)[] {
  const segments = typeof path === "string" ? path.split(".") : [...path];
  if (segments.length === 0 || segments.some((segment) => (
    typeof segment === "string" ? segment.length === 0 : !Number.isInteger(segment) || segment < 0
  ))) throw new Error(JSON.stringify(path));
  return segments;
}

function isStableId(id: string): boolean {
  return /^[A-Za-z0-9][A-Za-z0-9._:/-]*$/.test(id);
}
