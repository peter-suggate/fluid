#!/usr/bin/env node
/**
 * CPU-only production gate for Sparse CM12 pipeline compilation authority.
 *
 * The resident runtime receives a managed GPUDevice. A direct call to either
 * compute-pipeline creation method bypasses the source-level authority contract
 * (and synchronous creation is deliberately disabled by the managed proxy).
 * Follow runtime imports from the resident root so helpers cannot hide such a
 * call in a descriptor adapter or another transitive dependency.
 */
import assert from "node:assert/strict";
import { existsSync, readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import ts from "typescript";

const PROJECT_ROOT = path.resolve(fileURLToPath(new URL("../", import.meta.url)));
const RUNTIME_ROOTS = [
  "lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
] as const;
const COMPILATION_MANAGER = "lib/core/gpu-compilation-manager.ts";
const DIRECT_COMPUTE_PIPELINE_METHODS = new Set([
  "createComputePipeline",
  "createComputePipelineAsync",
]);

interface DirectPipelineCall {
  readonly file: string;
  readonly line: number;
  readonly column: number;
  readonly method: "createComputePipeline" | "createComputePipelineAsync";
}

interface ImportEdge {
  readonly importer: string;
  readonly specifier: string;
}

function projectPath(absolutePath: string): string {
  return path.relative(PROJECT_ROOT, absolutePath).split(path.sep).join("/");
}

function sourceKind(file: string): ts.ScriptKind {
  return file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
}

function unwrapExpression(expression: ts.Expression): ts.Expression {
  let current = expression;
  while (ts.isParenthesizedExpression(current)
    || ts.isAsExpression(current)
    || ts.isTypeAssertionExpression(current)
    || ts.isNonNullExpression(current)) {
    current = current.expression;
  }
  return current;
}

function calledMethod(expression: ts.Expression): string | undefined {
  const target = unwrapExpression(expression);
  if (ts.isPropertyAccessExpression(target)) return target.name.text;
  if (ts.isElementAccessExpression(target)
    && target.argumentExpression
    && ts.isStringLiteralLike(target.argumentExpression)) {
    return target.argumentExpression.text;
  }
  return undefined;
}

/** AST inspection deliberately ignores comments and descriptor documentation. */
export function inspectDirectComputePipelineCalls(
  file: string,
  source: string,
): readonly DirectPipelineCall[] {
  const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true,
    sourceKind(file));
  const calls: DirectPipelineCall[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isCallExpression(node)) {
      const method = calledMethod(node.expression);
      if (method && DIRECT_COMPUTE_PIPELINE_METHODS.has(method)) {
        const position = parsed.getLineAndCharacterOfPosition(node.getStart(parsed));
        calls.push({
          file,
          line: position.line + 1,
          column: position.character + 1,
          method: method as DirectPipelineCall["method"],
        });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(parsed);
  return calls;
}

function importClauseHasRuntimeBinding(clause: ts.ImportClause | undefined): boolean {
  if (!clause) return true; // Side-effect import.
  if (clause.isTypeOnly) return false;
  if (clause.name) return true;
  if (!clause.namedBindings) return false;
  if (ts.isNamespaceImport(clause.namedBindings)) return true;
  return clause.namedBindings.elements.some((element) => !element.isTypeOnly);
}

function exportDeclarationHasRuntimeBinding(declaration: ts.ExportDeclaration): boolean {
  if (declaration.isTypeOnly) return false;
  if (!declaration.exportClause) return true;
  if (ts.isNamespaceExport(declaration.exportClause)) return true;
  return declaration.exportClause.elements.some((element) => !element.isTypeOnly);
}

function runtimeRelativeSpecifiers(file: string, source: string): readonly string[] {
  const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true,
    sourceKind(file));
  const specifiers: string[] = [];
  const visit = (node: ts.Node): void => {
    if (ts.isImportDeclaration(node)
      && ts.isStringLiteralLike(node.moduleSpecifier)
      && node.moduleSpecifier.text.startsWith(".")
      && importClauseHasRuntimeBinding(node.importClause)) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (ts.isExportDeclaration(node)
      && node.moduleSpecifier
      && ts.isStringLiteralLike(node.moduleSpecifier)
      && node.moduleSpecifier.text.startsWith(".")
      && exportDeclarationHasRuntimeBinding(node)) {
      specifiers.push(node.moduleSpecifier.text);
    } else if (ts.isCallExpression(node)
      && node.expression.kind === ts.SyntaxKind.ImportKeyword
      && node.arguments.length === 1
      && ts.isStringLiteralLike(node.arguments[0]!)
      && node.arguments[0]!.text.startsWith(".")) {
      specifiers.push(node.arguments[0]!.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(parsed);
  return specifiers;
}

function resolveRuntimeModule(importer: string, specifier: string): string | undefined {
  const unresolved = path.resolve(path.dirname(importer), specifier);
  const extension = path.extname(unresolved);
  const extensionless = [".js", ".mjs", ".cjs"].includes(extension)
    ? unresolved.slice(0, -extension.length)
    : unresolved;
  const candidates = [
    unresolved,
    extensionless,
    `${extensionless}.ts`,
    `${extensionless}.tsx`,
    `${extensionless}.mts`,
    `${extensionless}.cts`,
    path.join(extensionless, "index.ts"),
    path.join(extensionless, "index.tsx"),
  ];
  return candidates.find((candidate) => existsSync(candidate) && statSync(candidate).isFile());
}

function isCompilationAuthority(file: string): boolean {
  // Test proxies are outside the production-root closure. Within that closure,
  // only the manager implementation may touch the raw WebGPU methods.
  return file === COMPILATION_MANAGER;
}

// Negative fixture: this is the exact defect class introduced when the VEX
// descriptor loop compiled its generated descriptors through the raw device.
const vexDescriptorRegression = `
  for (const descriptor of createSparseCM12VexActivityBatchPipelineDescriptors()) {
    pipelines.set(descriptor.key, await device.createComputePipelineAsync({
      layout: pipelineLayout,
      compute: { module, entryPoint: descriptor.entryPoint },
    }));
  }
`;
assert.deepEqual(
  inspectDirectComputePipelineCalls(
    "lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch-runtime.ts",
    vexDescriptorRegression,
  ).map(({ method }) => method),
  ["createComputePipelineAsync"],
  "the managed-pipeline gate must catch the VEX descriptor regression",
);
assert.deepEqual(
  inspectDirectComputePipelineCalls("fixture.ts", `
    await gpuCompilationManagerFor(device).compileComputePipeline(descriptor);
    const documentation = "device.createComputePipelineAsync(descriptor)";
  `),
  [],
  "manager acquisition and documentation must not be mistaken for direct compilation",
);

const compilationManagerSource = readFileSync(
  path.resolve(PROJECT_ROOT, COMPILATION_MANAGER), "utf8");
assert.match(compilationManagerSource,
  /process\.env\?\.FLUID_GPU_COMPILATION_CONCURRENCY \?\? 3\) : 3/,
  "interactive and native production compilation must default to bounded width three");
assert.match(compilationManagerSource,
  /maximumConcurrentBundles > 4/,
  "the compilation manager must retain a hard upper concurrency bound");
assert.doesNotMatch(compilationManagerSource,
  /Promise\.(?:all|allSettled)\([^;]*create(?:Compute|Render)PipelineAsync/s,
  "the compilation authority must not regain unbounded native pipeline fan-out");

const residentSource = readFileSync(path.resolve(PROJECT_ROOT, RUNTIME_ROOTS[0]), "utf8");
for (const retiredEntryPoint of [
  "classifyTemporalScalarCells",
  "countPressureCells", "finalizePressureCellWorklist", "compactPressureCells",
  "countPressureRows", "finalizePressureRowWorklist", "compactPressureRows",
  "resetPressureMembershipRepair",
  "classifyPersistentPressureCells", "classifyPressureTopologyCells",
  "repairPressureCellMembership", "classifyPersistentPressureRows",
  "classifyPressureTopologyRows", "repairPressureRowMembership",
  "finalizePressureMembershipBootstrap", "finalizePressureTopologyRepair",
  "bakeDirtyEffectivePressureEdges", "bakePressureTopologyEdges",
  "bakeEffectivePressureEdges",
  "prepareSharpeningFieldSparse", "scatterSharpeningMassSparse",
  "finalizeSharpeningSparse",
] as const) {
  assert.doesNotMatch(residentSource, new RegExp(`\\b${retiredEntryPoint}\\b`),
    `retired production pipeline ${retiredEntryPoint} must not return to eager compilation`);
}

// A weak compare-exchange may fail spuriously. Production retry sites must
// therefore be bounded and fail closed instead of leaving a command buffer
// permanently resident on the GPU.
for (const [file, required] of [
  ["lib/methods/adaptive-mass/sparse-cm12-canonical-membership.wgsl.ts",
    ["PCM_FAULT_ATOMIC_CONTENTION", "attempt<64u"]],
  ["lib/methods/adaptive-mass/sparse-cm12-srr1-resident-ingress.wgsl.ts",
    ["sirResidentFail(3u,tile)", "attempt<64u",
      "sirResidentResetCopiedEvents(tile:u32)"]],
  ["lib/methods/adaptive-mass/sparse-cm12-scalar-result-receipts.wgsl.ts",
    ["SPARSE_CM12_SCALAR_RESULT_FAULT.atomicContention", "attempt<64u"]],
  ["lib/methods/adaptive-mass/sparse-cm12-face-projection-authority.wgsl.ts",
    ["SPARSE_CM12_FACE_PROJECTION_FAULT.atomicContention", "attempt<64u"]],
  ["lib/methods/adaptive-mass/sparse-cm12-dirty-scheduler.wgsl.ts",
    ["SetPackedLaneMaximum(tile", "SetOriginStages(tile", "attempt<64u"]],
] as const) {
  const source = readFileSync(path.resolve(PROJECT_ROOT, file), "utf8");
  for (const token of required) assert.ok(source.includes(token),
    `${file} must retain bounded, fail-closed weak-CAS retries (${token})`);
}

const pending = RUNTIME_ROOTS.map((root) => path.resolve(PROJECT_ROOT, root));
const visited = new Set<string>();
const importedBy = new Map<string, ImportEdge>();
const unresolvedImports: ImportEdge[] = [];
const violations: DirectPipelineCall[] = [];

while (pending.length > 0) {
  const file = pending.shift()!;
  if (visited.has(file)) continue;
  visited.add(file);
  const relativeFile = projectPath(file);
  const source = readFileSync(file, "utf8");
  if (!isCompilationAuthority(relativeFile)) {
    violations.push(...inspectDirectComputePipelineCalls(relativeFile, source));
  }
  for (const specifier of runtimeRelativeSpecifiers(file, source)) {
    const dependency = resolveRuntimeModule(file, specifier);
    if (!dependency) {
      unresolvedImports.push({ importer: relativeFile, specifier });
      continue;
    }
    const relativeDependency = path.relative(PROJECT_ROOT, dependency);
    if (relativeDependency.startsWith(`..${path.sep}`) || path.isAbsolute(relativeDependency)) {
      continue;
    }
    if (!importedBy.has(dependency)) importedBy.set(dependency, { importer: relativeFile, specifier });
    pending.push(dependency);
  }
}

function reachabilityTrace(file: string): string {
  const trace = [file];
  let current = path.resolve(PROJECT_ROOT, file);
  const seen = new Set<string>();
  while (!RUNTIME_ROOTS.includes(projectPath(current) as typeof RUNTIME_ROOTS[number])) {
    if (seen.has(current)) break;
    seen.add(current);
    const edge = importedBy.get(current);
    if (!edge) break;
    trace.unshift(edge.importer);
    current = path.resolve(PROJECT_ROOT, edge.importer);
  }
  return trace.join(" -> ");
}

if (unresolvedImports.length > 0 || violations.length > 0) {
  console.error("Sparse CM12 managed pipeline compliance: FAIL");
  for (const edge of unresolvedImports) {
    console.error(`- unresolved runtime import ${edge.importer} -> ${edge.specifier}`);
  }
  for (const violation of violations) {
    console.error(`- ${violation.file}:${violation.line}:${violation.column} calls ${violation.method} directly`);
    console.error(`  reachable via ${reachabilityTrace(violation.file)}`);
  }
  console.error("All reachable production compute pipelines must be acquired through GPUCompilationManager.");
  process.exitCode = 1;
} else {
  console.log(`Sparse CM12 managed pipeline compliance: PASS (${visited.size} reachable modules)`);
}
