/**
 * Import-graph partition measurement for the method-decoupling refactor.
 * Walks static + dynamic imports from a set of roots and reports reach sets.
 *
 * Usage: node --import tsx tools/import-partition.ts [--json] [root ...]
 */
import { readFileSync, existsSync, statSync } from "node:fs";
import { dirname, join, resolve, relative } from "node:path";

const REPO = resolve(new URL("..", import.meta.url).pathname);

// The last alternative is the bare side-effect import (`import "./methods"`).
// Composition roots install plugins that way, so a walker blind to it reports
// an entry point as unwired when it is in fact the one doing the wiring.
const IMPORT_RE =
  /(?:^|\n)\s*import\s*["']([^"']+)["']|(?:^|\n)\s*(?:import|export)[\s\S]{0,400}?from\s*["']([^"']+)["']|import\s*\(\s*["']([^"']+)["']\s*\)|new\s+URL\s*\(\s*["']([^"']+)["']/g;

function resolveSpec(fromFile: string, spec: string): string | null {
  let base: string;
  if (spec.startsWith(".")) base = resolve(dirname(fromFile), spec);
  else if (spec.startsWith("@/")) base = resolve(REPO, spec.slice(2));
  else return null;
  const candidates = [
    base,
    `${base}.ts`,
    `${base}.tsx`,
    join(base, "index.ts"),
    join(base, "index.tsx"),
  ];
  for (const c of candidates) {
    if (existsSync(c) && statSync(c).isFile()) return c;
  }
  return null;
}

type Edge = { readonly file: string; readonly worker: boolean };

const cache = new Map<string, Edge[]>();
function importsOf(file: string): Edge[] {
  const hit = cache.get(file);
  if (hit) return hit;
  let src = "";
  try {
    src = readFileSync(file, "utf8");
  } catch {
    cache.set(file, []);
    return [];
  }
  const out: Edge[] = [];
  IMPORT_RE.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = IMPORT_RE.exec(src))) {
    const spec = m[1] ?? m[2] ?? m[3] ?? m[4];
    if (!spec) continue;
    const r = resolveSpec(file, spec);
    // A `new URL(...)` specifier is a worker entry, not an import: the file it
    // names is bundled, but it is evaluated on another thread with a module
    // graph of its own. Both readings are useful, so the edge is tagged rather
    // than dropped.
    if (r) out.push({ file: r, worker: m[4] !== undefined });
  }
  cache.set(file, out);
  return out;
}

/**
 * Every file reachable from `roots`.
 *
 * `evaluated` restricts the walk to what actually runs in the roots' own
 * module graph, stopping at worker entries. That is the reading any question
 * about module-scope side effects wants; the default reading is what ends up
 * in the bundle.
 */
export function reach(roots: string[], options: { evaluated?: boolean } = {}): Set<string> {
  const seen = new Set<string>();
  const stack = roots.map((r) => resolve(REPO, r)).filter((r) => existsSync(r));
  while (stack.length) {
    const f = stack.pop()!;
    if (seen.has(f)) continue;
    seen.add(f);
    for (const dep of importsOf(f)) {
      if (options.evaluated && dep.worker) continue;
      if (!seen.has(dep.file)) stack.push(dep.file);
    }
  }
  return seen;
}

/**
 * Whether `file` is a client module — the `"use client"` boundary that splits
 * a route's server graph from the one the browser and its SSR pass evaluate.
 */
export function isClientModule(file: string): boolean {
  try {
    return /^\uFEFF?\s*(?:\/\/[^\n]*\n|\/\*[\s\S]*?\*\/\s*)*["']use client["']/.test(readFileSync(resolve(REPO, file), "utf8"));
  } catch {
    return false;
  }
}

export function rel(f: string): string {
  return relative(REPO, f);
}

/**
 * Shortest import chain from a root to a target, or null when unreachable.
 *
 * A reach-set count says a boundary is violated; only the chain says which
 * edge to cut, which is the whole value of the measurement during a
 * decoupling.
 */
export function path(root: string, target: string): string[] | null {
  const from = resolve(REPO, root);
  const to = resolve(REPO, target);
  const parent = new Map<string, string | null>([[from, null]]);
  const queue = [from];
  while (queue.length) {
    const file = queue.shift()!;
    if (file === to) {
      const chain: string[] = [];
      for (let at: string | null = file; at; at = parent.get(at) ?? null) chain.push(rel(at));
      return chain.reverse();
    }
    for (const { file: dep } of importsOf(file)) {
      if (parent.has(dep)) continue;
      parent.set(dep, file);
      queue.push(dep);
    }
  }
  return null;
}

function main() {
  const args = process.argv.slice(2);
  const pathFlag = args.indexOf("--path");
  if (pathFlag >= 0) {
    const chain = path(args[pathFlag + 1]!, args[pathFlag + 2]!);
    console.log(chain ? chain.join("\n  -> ") : "unreachable");
    return;
  }
  const roots = args.filter((a) => !a.startsWith("--"));
  if (roots.length) {
    const set = [...reach(roots)].map(rel).sort();
    console.log(set.join("\n"));
    console.error(`# ${set.length} files`);
    return;
  }
  const lanes: Record<string, string[]> = {
    uniform: ["lib/methods/uniform/method.ts"],
    losasso: ["lib/methods/losasso/method.ts"],
    power: ["lib/methods/power/method.ts"],
    renderer: ["lib/core/webgpu-renderer.ts"],
  };
  const sets: Record<string, Set<string>> = {};
  for (const [k, v] of Object.entries(lanes)) sets[k] = reach(v);
  for (const [k, v] of Object.entries(sets)) console.log(`${k}: ${v.size} files`);
  const names = Object.keys(sets);
  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const a = sets[names[i]];
      const b = sets[names[j]];
      const inter = [...a].filter((x) => b.has(x));
      console.log(
        `${names[i]} ∩ ${names[j]}: ${inter.length}; ${names[i]}-only ${a.size - inter.length}; ${names[j]}-only ${b.size - inter.length}`,
      );
    }
  }
  // exclusive sets
  const uOnly = [...sets.uniform].filter((x) => !sets.octree.has(x));
  const oOnly = [...sets.octree].filter((x) => !sets.uniform.has(x));
  console.log(`\nuniform-exclusive (${uOnly.length}):`);
  console.log(uOnly.map(rel).sort().join("\n"));
  console.log(`\noctree-exclusive (${oOnly.length}) [first 200]:`);
  console.log(oOnly.map(rel).sort().slice(0, 200).join("\n"));
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split("/").pop()!)) main();
