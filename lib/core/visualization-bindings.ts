/**
 * One declaration per binding, driving both the shader and the bind group.
 *
 * A field view is a fragment shader plus the publications it reads. Today those
 * are two hand-maintained lists that have to agree: `@group(0) @binding(4)` in
 * the WGSL and `{binding: 4, resource: source.catalogEntryHeaders}` on the host.
 * Nothing checks that they do, and a mismatch is a wrong picture rather than an
 * error. Declaring the binding once and generating both removes the class.
 *
 * It also makes the budget checkable. Browsers on Apple silicon report a
 * **ten storage buffer** ceiling per shader stage, which is why the technique
 * overlay is five programs rather than one: each holds a disjoint subset of the
 * publication bundle. That constraint is currently only in a comment, so the
 * eleventh binding would be found by a driver at run time. Here it is arithmetic
 * over a declaration, and a program that overruns it fails to build.
 */

/**
 * Storage buffers a fragment stage may bind.
 *
 * Not a WebGPU default — `maxStorageBuffersPerShaderStage` is 8 in the spec's
 * base limits and higher on most desktop adapters — but the number browsers
 * report on the Apple silicon this runs on, and therefore the one a program has
 * to fit inside to be useful. Storage *textures* are a separate budget, which is
 * why the pixel trace records into an r32uint texture rather than a buffer.
 */
export const VISUALIZATION_STORAGE_BUFFERS_PER_STAGE = 10;

export type VisualizationBindingKind =
  | "uniform"
  | "read-only-storage"
  | "storage"
  | "texture-3d-uint"
  | "texture-depth-2d";

/**
 * `Resource` is the set of keys this binding may name.
 *
 * `string` for a field program, whose publications are a hand-kept bundle. A
 * stage lens narrows it to a union of exactly the publications and tap
 * snapshots that lens declares, which is what turns a mistyped resource key
 * from a magenta frame into a `tsc` error.
 */
export interface VisualizationBinding<Resource extends string = string> {
  /** WGSL identifier the shader body refers to. */
  readonly name: string;
  readonly kind: VisualizationBindingKind;
  /** WGSL type, for everything but a texture. `array<LeafHeader>`, `Uniforms`. */
  readonly type?: string;
  /**
   * Key the host resolves to a `GPUBindingResource`. Named rather than passed
   * positionally so a reordered declaration cannot silently rebind a buffer.
   */
  readonly resource: Resource;
}

/**
 * A shader and the bindings it holds.
 *
 * Programs exist because of the storage ceiling: a field belongs to whichever
 * program already binds the publications it reads, and adding a field to a
 * program that is full is a decision someone has to make rather than a driver
 * failure to debug.
 */
export interface VisualizationProgram {
  readonly id: string;
  readonly label: string;
  readonly bindings: readonly VisualizationBinding[];
}

const STORAGE_KINDS: ReadonlySet<VisualizationBindingKind> =
  new Set<VisualizationBindingKind>(["read-only-storage", "storage"]);

export interface VisualizationBindingBudget {
  readonly storageBuffers: number;
  readonly uniformBuffers: number;
  readonly textures: number;
  readonly total: number;
  /** Storage buffers still available on the portable stage. */
  readonly headroom: number;
}

export function visualizationBindingBudget(
  bindings: readonly VisualizationBinding[],
): VisualizationBindingBudget {
  let storageBuffers = 0, uniformBuffers = 0, textures = 0;
  for (const binding of bindings) {
    if (STORAGE_KINDS.has(binding.kind)) storageBuffers += 1;
    else if (binding.kind === "uniform") uniformBuffers += 1;
    else textures += 1;
  }
  return {
    storageBuffers, uniformBuffers, textures,
    total: bindings.length,
    headroom: VISUALIZATION_STORAGE_BUFFERS_PER_STAGE - storageBuffers,
  };
}

/**
 * Why a program cannot be built, or nothing.
 *
 * Returned rather than thrown so a catalog test can report every problem at
 * once; `assertVisualizationProgram` is the throwing form for the build path.
 */
export function visualizationProgramProblems(
  program: VisualizationProgram,
): readonly string[] {
  const problems: string[] = [];
  const names = new Set<string>(), resources = new Set<string>();
  for (const binding of program.bindings) {
    if (names.has(binding.name)) problems.push(`${program.id}: duplicate binding name "${binding.name}"`);
    names.add(binding.name);
    if (resources.has(binding.resource)) {
      problems.push(`${program.id}: resource "${binding.resource}" is bound twice`);
    }
    resources.add(binding.resource);
    const needsType = binding.kind === "uniform" || STORAGE_KINDS.has(binding.kind);
    if (needsType && !binding.type) problems.push(`${program.id}: binding "${binding.name}" has no WGSL type`);
  }
  const budget = visualizationBindingBudget(program.bindings);
  if (budget.storageBuffers > VISUALIZATION_STORAGE_BUFFERS_PER_STAGE) {
    problems.push(
      `${program.id}: binds ${budget.storageBuffers} storage buffers, `
      + `${VISUALIZATION_STORAGE_BUFFERS_PER_STAGE} is the portable per-stage ceiling. `
      + `Split the program or move a field to one that already binds what it reads.`);
  }
  return problems;
}

export function assertVisualizationProgram(program: VisualizationProgram): void {
  const problems = visualizationProgramProblems(program);
  if (problems.length > 0) throw new RangeError(problems.join("\n"));
}

function declaration(binding: VisualizationBinding, index: number): string {
  const at = `@group(0) @binding(${index})`;
  switch (binding.kind) {
    case "uniform": return `${at} var<uniform> ${binding.name}:${binding.type};`;
    case "read-only-storage": return `${at} var<storage,read> ${binding.name}:${binding.type};`;
    case "storage": return `${at} var<storage,read_write> ${binding.name}:${binding.type};`;
    case "texture-3d-uint": return `${at} var ${binding.name}:texture_3d<u32>;`;
    case "texture-depth-2d": return `${at} var ${binding.name}:texture_depth_2d;`;
  }
}

/**
 * The WGSL preamble for a program's bindings.
 *
 * Slot numbers are the declaration order, which is the only place they are
 * decided. Nothing downstream writes a slot number by hand, so nothing
 * downstream can disagree about one.
 */
export function visualizationBindingPreambleWGSL(program: VisualizationProgram): string {
  assertVisualizationProgram(program);
  return program.bindings.map(declaration).join("\n");
}

/**
 * Bind-group entries for a program, from the same declaration as the preamble.
 *
 * `resolve` maps a declared resource key onto whatever the frame has bound. A
 * key it cannot supply returns undefined and the whole group is refused: a
 * partially bound program would either fail validation or, worse, read a stale
 * buffer left over from the last publication.
 */
export function visualizationBindGroupEntries<Resource>(
  program: VisualizationProgram,
  resolve: (resource: string) => Resource | undefined,
): readonly { binding: number; resource: Resource }[] | undefined {
  const entries: { binding: number; resource: Resource }[] = [];
  for (const [binding, declared] of program.bindings.entries()) {
    const resource = resolve(declared.resource);
    if (resource === undefined) return undefined;
    entries.push({ binding, resource });
  }
  return entries;
}
