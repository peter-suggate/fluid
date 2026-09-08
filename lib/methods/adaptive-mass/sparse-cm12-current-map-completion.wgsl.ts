import type { SparseCM12CurrentMapLayout } from "./sparse-cm12-current-map.wgsl";
import { SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS } from "./sparse-cm12-current-map-velocity.wgsl";

/** Atomic receipt block immediately before the existing failure tail. The host
 * clears this block with GPUCommandEncoder.clearBuffer before each transaction.
 * A successful previous frame must never stand in for unfinished current work. */
export const SPARSE_CM12_CURRENT_MAP_COMPLETION_WORDS = 32;
/** Written by the final singleton publication, never by a worker census. */
export const SPARSE_CM12_CURRENT_MAP_COMMIT_COMPLETION_SLOT = 31;
type InvocationCount = number | "p.counts.x";

export interface SparseCM12CurrentMapCompletionSpec {
  readonly name: string;
  readonly slot: number;
  readonly invocations: InvocationCount;
  readonly dispatches: number;
  readonly workgroupSize: 1 | 64;
  readonly expectedCompletions: InvocationCount;
  /** Cooperative kernels report one completed support explicitly after all
   * workgroup lanes have finished. Their uniform barriers cannot be wrapped. */
  readonly instrumentation?: "worker" | "manual";
}

export function createSparseCM12CurrentMapCompletionSpecs(
  map: SparseCM12CurrentMapLayout,
  templateCellCount: InvocationCount = "p.counts.x",
  options: { readonly cooperativeMeasure?: boolean; readonly cachedMeasureRanges?: boolean } = {},
): readonly SparseCM12CurrentMapCompletionSpec[] {
  const fineCount = map.dimensions.reduce((a, b) => a * b, 1);
  const entries: readonly (readonly [string, InvocationCount, 1 | 64, number?])[] = [
    ["compileCurrentMapVelocity", map.nodeCount, 64],
    ["extendCurrentMapVelocityToScratch", map.nodeCount, 64,
      SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS / 2],
    ["extendCurrentMapVelocityFromScratch", map.nodeCount, 64,
      SPARSE_CM12_CURRENT_MAP_VELOCITY_EXTENSION_SWEEPS / 2],
    ["compileCurrentMapTraceSchedule", map.lineCounts[0], 64],
    ["sealCurrentMapTraceSchedule", 1, 1],
    ["advanceCurrentMapNodes", map.nodeCount, 64],
    ["filterCurrentMapX", map.lineCounts[0], 64],
    ["filterCurrentMapY", map.lineCounts[1], 64],
    ["filterCurrentMapZ", map.lineCounts[2], 64],
    ["certifyCurrentMap", map.cellCount, 64],
    ["compileCurrentMapPhysicalBoundary", map.boundarySampleCount, 64],
    ["boundCurrentMapPhysicalBoundary", 6, 1],
    ["validateCurrentMapMaterialCoverage", 1, 1],
    ...(options.cachedMeasureRanges ? [["compileCurrentMapFineMeasureRanges", fineCount, 64] as const] : []),
    [options.cooperativeMeasure ? "integrateCurrentMapFineMeasureCooperative" : "integrateCurrentMapFineMeasure", fineCount, 64],
    ["validateCurrentMapCoverage", fineCount, 64],
    ["compileRetainedDensityNativeIntegrals", templateCellCount, 64],
    ["publishCurrentMapIncrement", map.nodeCount, 64],
  ];
  const specs = entries.map(([name, invocations, workgroupSize, dispatches = 1], slot) =>
    Object.freeze({ name, slot, invocations, dispatches, workgroupSize,
      expectedCompletions: typeof invocations === "number" ? invocations * dispatches : invocations,
      instrumentation: name === "integrateCurrentMapFineMeasureCooperative" ? "manual" as const : "worker" as const }));
  validateSpecs(specs);
  return Object.freeze(specs);
}

function validateSpecs(specs: readonly SparseCM12CurrentMapCompletionSpec[]): void {
  const names = new Set<string>();
  const slots = new Set<number>();
  for (const spec of specs) {
    if (!/^[A-Za-z_][A-Za-z_0-9]*$/.test(spec.name) || names.has(spec.name)
      || !Number.isSafeInteger(spec.slot) || spec.slot < 0
      || spec.slot >= SPARSE_CM12_CURRENT_MAP_COMMIT_COMPLETION_SLOT || slots.has(spec.slot)
      || (typeof spec.invocations === "number"
        ? !Number.isSafeInteger(spec.invocations) || spec.invocations < 1
        : spec.invocations !== "p.counts.x" || spec.dispatches !== 1)
      || !Number.isSafeInteger(spec.dispatches) || spec.dispatches < 1
      || spec.expectedCompletions !== (typeof spec.invocations === "number"
        ? spec.invocations * spec.dispatches : spec.invocations)
      || (typeof spec.expectedCompletions === "number"
        && (!Number.isSafeInteger(spec.expectedCompletions) || spec.expectedCompletions > 0xffff_ffff))
      || (spec.instrumentation !== undefined && spec.instrumentation !== "manual" && spec.instrumentation !== "worker")
      || (spec.workgroupSize !== 1 && spec.workgroupSize !== 64)) {
      throw new RangeError(`Invalid current-map completion specification: ${spec.name}`);
    }
    names.add(spec.name); slots.add(spec.slot);
  }
}

function countWGSL(count: InvocationCount): string {
  return typeof count === "number" ? `${count}u` : count;
}

/** Requires the existing atomic topologyArena, cm12FailureBase(),
 * cm12CurrentMapFailed(), and cm12RecordFailure(). This validates actual
 * successful worker completions, not merely dispatch submission. Call before
 * changing either accepted density bank or the archived increment count. */
export function createSparseCM12CurrentMapCompletionWGSL(
  specs: readonly SparseCM12CurrentMapCompletionSpec[],
): string {
  validateSpecs(specs);
  const measure = specs.find(spec => spec.name === "integrateCurrentMapFineMeasure"
    || spec.name === "integrateCurrentMapFineMeasureCooperative");
  return /* wgsl */ `
var<workgroup> cm12CurrentMapCompletionLanes:array<u32,64>;
fn cm12CurrentMapCompletionBase()->u32{
  return cm12FailureBase()-${SPARSE_CM12_CURRENT_MAP_COMPLETION_WORDS}u;
}
fn cm12CurrentMapCompletionValidPrefix(limit:u32)->bool{
  if(cm12CurrentMapFailed()){return false;}
${specs.map(spec => /* wgsl */ `  if(${spec.slot}u<limit){
    let observed=atomicLoad(&topologyArena[cm12CurrentMapCompletionBase()+${spec.slot}u]);
    if(observed!=${countWGSL(spec.expectedCompletions)}){
      cm12RecordFailure(6u,${spec.slot}u,vec4u(124u,${spec.slot}u,observed,${countWGSL(spec.expectedCompletions)}));
      return false;
    }
  }`).join("\n")}
  return true;
}
fn cm12CurrentMapCompletionValid()->bool{
  return cm12CurrentMapCompletionValidPrefix(${SPARSE_CM12_CURRENT_MAP_COMPLETION_WORDS}u);
}
${measure ? /* wgsl */ `fn cm12CurrentMapMeasureCompleted(ordinal:u32){
  if(ordinal<${countWGSL(measure.invocations)}&&!cm12CurrentMapFailed()){
    atomicAdd(&topologyArena[cm12CurrentMapCompletionBase()+${measure.slot}u],1u);
  }
}` : ""}
`;
}

/** Find the matching function brace while ignoring both WGSL comment forms.
 * Block comments may be nested. A malformed generated kernel is an error,
 * never a reason to silently omit its completion receipt. */
function closingBrace(source: string, start: number): number {
  let depth = 0, blockComment = 0, lineComment = false;
  for (let at = start; at < source.length; at++) {
    const char = source[at], next = source[at + 1];
    if (lineComment) { if (char === "\n") lineComment = false; continue; }
    if (blockComment) {
      if (char === "/" && next === "*") { blockComment++; at++; }
      else if (char === "*" && next === "/") { blockComment--; at++; }
      continue;
    }
    if (char === "/" && next === "/") { lineComment = true; at++; continue; }
    if (char === "/" && next === "*") { blockComment = 1; at++; continue; }
    if (char === "{") depth++;
    else if (char === "}" && --depth === 0) return at;
  }
  throw new Error("Unterminated current-map completion worker");
}

/** Instrument the complete combined shader before generic failure guards and
 * entry-point splitting. The private worker retains all original early returns.
 * Every lane then reaches the wrapper's barrier, including out-of-range lanes.
 * One global atomic per workgroup reduces contention in million-node passes.
 * Determining that an in-range native cell is inactive is completed work;
 * recording a shared failure prevents publication independently of this count. */
export function instrumentSparseCM12CurrentMapCompletionWGSL(
  source: string,
  specs: readonly SparseCM12CurrentMapCompletionSpec[],
): string {
  validateSpecs(specs);
  const edits: { start: number; end: number; replacement: string }[] = [];
  for (const spec of specs) {
    const pattern = new RegExp(`@compute\\s+@workgroup_size\\(\\s*${spec.workgroupSize}\\s*\\)\\s*fn\\s+${spec.name}\\s*\\(`, "g");
    const matches = [...source.matchAll(pattern)];
    if (matches.length !== 1) throw new Error(`Expected one current-map completion entry ${spec.name}; found ${matches.length}`);
    if (spec.instrumentation === "manual") continue;
    const match = matches[0]!;
    const start = match.index!;
    const argumentStart = start + match[0].length;
    const bodyStart = source.indexOf("{", argumentStart);
    if (bodyStart < 0) throw new Error(`Missing current-map completion body: ${spec.name}`);
    const signature = source.slice(argumentStart, bodyStart).trim();
    if (!signature.endsWith(")")) throw new Error(`Unexpected current-map completion signature: ${spec.name}`);
    const argumentsText = signature.slice(0, -1).trim();
    const gid = argumentsText.match(/^@builtin\(\s*global_invocation_id\s*\)\s*([A-Za-z_][A-Za-z_0-9]*)\s*:\s*vec3u\s*,?$/);
    if (argumentsText && !gid) throw new Error(`Unsupported current-map completion arguments: ${spec.name}`);
    if (!gid && spec.invocations !== 1) throw new Error(`Non-singleton current-map entry has no invocation id: ${spec.name}`);
    const bodyEnd = closingBrace(source, bodyStart);
    const workerName = `cm12CurrentMapCompletionWorker_${spec.name}`;
    const body = source.slice(bodyStart, bodyEnd + 1);
    const replacement = /* wgsl */ `fn ${workerName}(${gid ? `${gid[1]}:vec3u` : ""})${body}
@compute @workgroup_size(${spec.workgroupSize})
fn ${spec.name}(@builtin(global_invocation_id)cm12CompletionGid:vec3u,
  @builtin(local_invocation_index)cm12CompletionLane:u32){
  ${workerName}(${gid ? "cm12CompletionGid" : ""});
  cm12CurrentMapCompletionLanes[cm12CompletionLane]=select(0u,1u,
    cm12CompletionGid.x<${countWGSL(spec.invocations)}&&!cm12CurrentMapFailed());
  workgroupBarrier();
  if(cm12CompletionLane==0u){
    var completed=0u;
    for(var lane=0u;lane<${spec.workgroupSize}u;lane++){
      completed+=cm12CurrentMapCompletionLanes[lane];
    }
    atomicAdd(&topologyArena[cm12CurrentMapCompletionBase()+${spec.slot}u],completed);
  }
}`;
    edits.push({ start, end: bodyEnd + 1, replacement });
  }
  for (const edit of edits.sort((a, b) => b.start - a.start)) {
    source = source.slice(0, edit.start) + edit.replacement + source.slice(edit.end);
  }
  return source;
}
