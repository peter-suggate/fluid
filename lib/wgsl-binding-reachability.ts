export type WGSLBindingClass = "storage" | "uniform" | "other";

export interface WGSLBindingDeclaration {
  readonly group: number;
  readonly binding: number;
  readonly name: string;
  readonly bindingClass: WGSLBindingClass;
  readonly addressSpace?: string;
  readonly access?: string;
}

export interface WGSLComputeBindingReachabilityAudit {
  readonly entryPoint: string;
  readonly reachableFunctions: readonly string[];
  readonly bindings: readonly WGSLBindingDeclaration[];
  readonly storage: readonly WGSLBindingDeclaration[];
  readonly uniform: readonly WGSLBindingDeclaration[];
  readonly other: readonly WGSLBindingDeclaration[];
  readonly storageCount: number;
  readonly uniformCount: number;
}

export interface WGSLActivityBindingEligibility {
  readonly entryPoint: string;
  readonly productionStorageCount: number;
  readonly activityStorageCount: number;
  readonly totalStorageCount: number;
  readonly maximumStorageCount: number;
}

/** Enforce the portable activity contract before a shader variant acquires
 * the dedicated recorder binding. */
export function assertWGSLActivityBindingEligibility(
  audit: Pick<WGSLComputeBindingReachabilityAudit, "entryPoint" | "storageCount">,
  maximumStorageCount = 10,
  activityStorageCount = 1,
): WGSLActivityBindingEligibility {
  if (!Number.isSafeInteger(maximumStorageCount) || maximumStorageCount < 1
    || !Number.isSafeInteger(activityStorageCount) || activityStorageCount < 1) {
    throw new RangeError("WGSL activity binding limits must be positive integers");
  }
  const totalStorageCount = audit.storageCount + activityStorageCount;
  if (totalStorageCount > maximumStorageCount) {
    throw new Error(
      `${audit.entryPoint} uses ${audit.storageCount} production storage bindings; `
      + `${activityStorageCount} activity binding would exceed ${maximumStorageCount}`,
    );
  }
  return Object.freeze({
    entryPoint: audit.entryPoint,
    productionStorageCount: audit.storageCount,
    activityStorageCount,
    totalStorageCount,
    maximumStorageCount,
  });
}

interface WGSLFunctionDeclaration {
  readonly name: string;
  readonly body: string;
  readonly compute: boolean;
}

/** Remove comments without changing offsets or accidentally joining tokens. */
function codeOnlyWGSL(source: string): string {
  let output = "";
  for (let index = 0; index < source.length;) {
    if (source[index] === "/" && source[index + 1] === "/") {
      output += "  "; index += 2;
      while (index < source.length && source[index] !== "\n" && source[index] !== "\r") {
        output += " "; index += 1;
      }
      continue;
    }
    if (source[index] === "/" && source[index + 1] === "*") {
      output += "  "; index += 2;
      while (index < source.length) {
        if (source[index] === "*" && source[index + 1] === "/") {
          output += "  "; index += 2; break;
        }
        output += source[index] === "\n" || source[index] === "\r" ? source[index] : " ";
        index += 1;
      }
      continue;
    }
    output += source[index]; index += 1;
  }
  return output;
}

function checkedU32(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > 0xffff_ffff) {
    throw new RangeError(`WGSL ${label} must be a u32`);
  }
  return parsed;
}

function parseBindings(source: string): readonly WGSLBindingDeclaration[] {
  const bindings: WGSLBindingDeclaration[] = [];
  const slots = new Set<string>();
  const declaration = /((?:@[A-Za-z_]\w*\s*\([^)]*\)\s*)+)var(?:\s*<\s*([^>]*)>)?\s*([A-Za-z_]\w*)\s*:/g;
  for (const match of source.matchAll(declaration)) {
    const attributes = match[1]!;
    const groupMatch = /@group\s*\(\s*(\d+)\s*\)/.exec(attributes);
    const bindingMatch = /@binding\s*\(\s*(\d+)\s*\)/.exec(attributes);
    if (!groupMatch || !bindingMatch) continue;
    const group = checkedU32(groupMatch[1]!, "bind group");
    const binding = checkedU32(bindingMatch[1]!, "binding");
    const slot = `${group}:${binding}`;
    if (slots.has(slot)) throw new Error(`WGSL binding @group(${group}) @binding(${binding}) is duplicated`);
    slots.add(slot);
    const qualifiers = match[2]?.split(",").map((value) => value.trim()).filter(Boolean) ?? [];
    const addressSpace = qualifiers[0];
    const access = qualifiers[1];
    const bindingClass: WGSLBindingClass = addressSpace === "storage"
      ? "storage"
      : addressSpace === "uniform" ? "uniform" : "other";
    bindings.push({
      group, binding, name: match[3]!, bindingClass,
      ...(addressSpace ? { addressSpace } : {}),
      ...(access ? { access } : {}),
    });
  }
  return bindings.sort((left, right) => left.group - right.group
    || left.binding - right.binding || left.name.localeCompare(right.name));
}

function matchingBrace(source: string, open: number, label: string): number {
  if (open < 0) throw new Error(`WGSL ${label} has no body`);
  let depth = 0;
  for (let index = open; index < source.length; index += 1) {
    if (source[index] === "{") depth += 1;
    else if (source[index] === "}" && --depth === 0) return index;
  }
  throw new Error(`WGSL ${label} body is unterminated`);
}

function parseFunctions(source: string): ReadonlyMap<string, WGSLFunctionDeclaration> {
  const functions = new Map<string, WGSLFunctionDeclaration>();
  const signature = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  for (const match of source.matchAll(signature)) {
    const name = match[1]!;
    if (functions.has(name)) throw new Error(`WGSL function ${name} is duplicated`);
    const start = match.index!;
    const bodyStart = source.indexOf("{", start + match[0].length);
    const bodyEnd = matchingBrace(source, bodyStart, `function ${name}`);
    const previousBrace = source.lastIndexOf("}", start);
    const previousSemicolon = source.lastIndexOf(";", start);
    const attributes = source.slice(Math.max(previousBrace, previousSemicolon) + 1, start);
    functions.set(name, {
      name,
      body: source.slice(bodyStart + 1, bodyEnd),
      compute: /@compute\b/.test(attributes),
    });
  }
  return functions;
}

function identifierUsed(code: string, name: string, followedByCall = false): boolean {
  const suffix = followedByCall ? "\\s*\\(" : "\\b";
  // A field access such as `settings.control` must not make a module-scope
  // binding named `control` look reachable.
  return new RegExp(`(?<![.A-Za-z0-9_])${name}${suffix}`).test(code);
}

/**
 * Audit the resource interface statically reachable from one compute entry.
 *
 * This intentionally reports code reachability, not declarations present in
 * the module. It is a lightweight project audit rather than a full WGSL type
 * checker; callers should still rely on WebGPU validation for authoritative
 * pipeline-layout reflection.
 */
export function auditWGSLComputeBindingReachability(
  wgsl: string,
  entryPoint: string,
): WGSLComputeBindingReachabilityAudit {
  if (!/^[A-Za-z_]\w*$/.test(entryPoint)) throw new RangeError("WGSL compute entry point is invalid");
  const source = codeOnlyWGSL(wgsl);
  const declarations = parseBindings(source);
  const functions = parseFunctions(source);
  const entry = functions.get(entryPoint);
  if (!entry) throw new Error(`WGSL compute entry point ${entryPoint} is missing`);
  if (!entry.compute) throw new Error(`WGSL function ${entryPoint} is not a compute entry point`);

  const reachable = new Set<string>();
  const pending = [entryPoint];
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reachable.has(name)) continue;
    reachable.add(name);
    const body = functions.get(name)!.body;
    for (const candidate of functions.keys()) {
      if (!reachable.has(candidate) && identifierUsed(body, candidate, true)) pending.push(candidate);
    }
  }

  const bodies = [...reachable].map((name) => functions.get(name)!.body);
  const bindings = declarations.filter(({ name }) => bodies.some((body) => identifierUsed(body, name)));
  const storage = bindings.filter(({ bindingClass }) => bindingClass === "storage");
  const uniform = bindings.filter(({ bindingClass }) => bindingClass === "uniform");
  const other = bindings.filter(({ bindingClass }) => bindingClass === "other");
  return {
    entryPoint,
    reachableFunctions: [...reachable].sort(),
    bindings,
    storage,
    uniform,
    other,
    storageCount: storage.length,
    uniformCount: uniform.length,
  };
}

export interface ExplicitBindGroupLayoutEntryLike {
  readonly binding: number;
  readonly visibility: number;
  readonly buffer?: { readonly type?: "uniform" | "storage" | "read-only-storage" };
}

export interface ExplicitComputeBufferBindingAudit {
  readonly storageBindings: readonly number[];
  readonly uniformBindings: readonly number[];
  readonly storageCount: number;
  readonly uniformCount: number;
}

/** WebGPU's GPUShaderStage.COMPUTE bit, kept literal so this helper is runtime-pure. */
export const GPU_COMPUTE_STAGE_BIT = 0x4;

/** Count compute-visible explicit-layout buffer entries without requiring a GPU device. */
export function auditExplicitComputeBufferBindings(
  entries: readonly ExplicitBindGroupLayoutEntryLike[],
  computeStageBit = GPU_COMPUTE_STAGE_BIT,
): ExplicitComputeBufferBindingAudit {
  if (!Number.isSafeInteger(computeStageBit) || computeStageBit <= 0) {
    throw new RangeError("Compute shader-stage bit must be positive");
  }
  const seen = new Set<number>();
  const storageBindings: number[] = [];
  const uniformBindings: number[] = [];
  for (const entry of entries) {
    if (!Number.isSafeInteger(entry.binding) || entry.binding < 0) {
      throw new RangeError("Explicit bind-group-layout binding must be a non-negative integer");
    }
    if (seen.has(entry.binding)) throw new Error(`Explicit bind-group-layout binding ${entry.binding} is duplicated`);
    seen.add(entry.binding);
    if ((entry.visibility & computeStageBit) === 0 || !entry.buffer) continue;
    const type = entry.buffer.type ?? "uniform";
    if (type === "storage" || type === "read-only-storage") storageBindings.push(entry.binding);
    else uniformBindings.push(entry.binding);
  }
  storageBindings.sort((left, right) => left - right);
  uniformBindings.sort((left, right) => left - right);
  return {
    storageBindings,
    uniformBindings,
    storageCount: storageBindings.length,
    uniformCount: uniformBindings.length,
  };
}
