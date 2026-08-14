/**
 * Every entry point that looks a method up must also install the catalog.
 *
 * `lib/core/method-registry.ts` is deliberately empty until something imports
 * `lib/methods` for its side effect. That inversion is what keeps core off the
 * method graph, and its one hazard is a new entry point that forgets to wire
 * itself: the failure surfaces at run time, at whatever moment the first
 * lookup happens. This walks the import graph from every command, page and
 * test, and reports any that reach the registry without reaching the catalog.
 *
 * An entry point is a *module graph*, not a product. The app has three of them
 * and they share no evaluation: the server components under `app/` never run
 * in the browser, the `use client` modules the router hands them never run on
 * the server, and the render worker is its own thread again. A check that
 * modelled one graph read the root layout's install as covering the browser
 * and reported green while every page 500'd.
 *
 * Run: node --import tsx tools/check-method-install.ts [--strict]
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { isClientModule, reach, rel } from "./import-partition";

const REPO = resolve(new URL("..", import.meta.url).pathname);
const REGISTRY = "lib/core/method-registry.ts";
const CATALOG = "lib/methods/index.ts";

const SPEC_RE = /(?:^|\n)\s*import\s*["']([^"']+)["']|(?:^|\n)\s*(?:import|export)[\s\S]{0,400}?from\s*["']([^"']+)["']|import\s*\(\s*(?:\/\*[^*]*\*\/\s*)?["']([^"']+)["']\s*\)/g;

function walkSources(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    if (name.startsWith(".") || name === "node_modules") continue;
    const path = join(dir, name);
    if (statSync(path).isDirectory()) walkSources(path, out);
    else if (/\.(ts|tsx)$/.test(name)) out.push(rel(path));
  }
  return out;
}

function resolveSpec(fromFile: string, spec: string): string | null {
  let base: string;
  if (spec.startsWith(".")) base = resolve(REPO, dirname(fromFile), spec);
  else if (spec.startsWith("@/")) base = resolve(REPO, spec.slice(2));
  else return null;
  for (const c of [base, `${base}.ts`, `${base}.tsx`, join(base, "index.ts"), join(base, "index.tsx")]) {
    if (existsSync(c) && statSync(c).isFile()) return rel(c);
  }
  return null;
}

/**
 * The `use client` modules the server graph hands to the router.
 *
 * Each is a browser entry in its own right — a client module reached only as
 * another component's `children` is a sibling of the shell rather than a
 * descendant of it, so it cannot rely on the shell's module having been
 * evaluated first and has to install the catalog itself.
 */
function clientRoots(): string[] {
  const roots = new Set<string>();
  const seen = new Set<string>();
  const stack = walkSources(join(REPO, "app"));
  while (stack.length) {
    const file = stack.pop()!;
    if (seen.has(file)) continue;
    seen.add(file);
    const src = (() => { try { return readFileSync(resolve(REPO, file), "utf8"); } catch { return ""; } })();
    SPEC_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = SPEC_RE.exec(src))) {
      const spec = m[1] ?? m[2] ?? m[3];
      const target = spec ? resolveSpec(file, spec) : null;
      if (!target) continue;
      if (isClientModule(target)) roots.add(target);
      else stack.push(target);
    }
  }
  return [...roots];
}

/**
 * What actually starts a module graph: a command, a test, a server worker, the
 * app's server routes, the client modules those routes mount, and the render
 * worker's own thread.
 */
function entryPoints(): string[] {
  const out: string[] = [];
  for (const dir of ["tools", "tests", "worker"]) walkSources(join(REPO, dir), out);
  out.push(...walkSources(join(REPO, "app")));
  out.push(...clientRoots());
  out.push("lib/core/webgpu-render-worker.ts");
  return out;
}

function main() {
  const unwired = entryPoints().filter((entry) => {
    // `evaluated` stops the walk at worker entries. Without it the render
    // worker's install counts for the page that spawns it, which is precisely
    // the false green this check exists to prevent.
    const set = new Set([...reach([entry], { evaluated: true })].map(rel));
    return set.has(REGISTRY) && !set.has(CATALOG);
  });
  if (!unwired.length) {
    console.log("method install: every entry point that resolves a method installs the catalog");
    return;
  }
  console.log(`method install: ${unwired.length} entry point(s) resolve a method without installing it`);
  for (const entry of unwired.sort()) console.log(`  ${entry}`);
  console.log(`\nAdd \`import "@/lib/methods";\` (or the relative equivalent) to each.`);
  if (process.argv.includes("--strict")) process.exit(1);
}

main();
