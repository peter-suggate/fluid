/**
 * Module boundary check for the fluid-core / method-plugin layout.
 *
 * Rules (docs/method-decoupling-handoff.md §2.1):
 *   1. lib/core imports nothing outside lib/core.
 *   2. lib/methods/<m> imports lib/core, itself, and (losasso/power only)
 *      lib/methods/octree-shared. Never another method.
 *   3. lib/methods/octree-shared imports only lib/core and itself.
 *   4. Nothing imports lib/harness except tools/ and tests/.
 *   5. components/, app/, worker/ import lib/core only.
 *   6. Nothing in lib/ imports shape-lab/.
 *   7. lib/svo is a rendering layer, not a method: it imports lib/core and
 *      itself, and no method may import it. The renderer composes it; nothing
 *      about a simulation method may reach it, and it may not reach a method.
 *   8. Only a composition root may import the method catalog. Everything else
 *      resolves a method through lib/core/method-registry.
 *   9. lib/sparse-world is a public semantic boundary shared by core and sparse
 *      methods. Its internal adapters may reach a method implementation; its
 *      public index may not.
 *
 * Run: node --import tsx tools/check-module-boundaries.ts [--strict]
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const REPO = resolve(new URL("..", import.meta.url).pathname);
// `\bimport\s+` catches the bare side-effect import a composition root uses
// to install a plugin package; without it those edges go unpoliced.
const SPEC_RE = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*|\bnew\s+URL\s*\(\s*)(["'])([^"']+)\1/g;

type Zone =
  | "core"
  | "method-uniform"
  | "method-losasso"
  | "method-power"
  | "method-adaptive-mass"
  | "octree-shared"
  | "sparse-world"
  | "svo"
  | "harness"
  | "shape-lab"
  | "lib-other"
  | "ui"
  | "tooling";

function zoneOf(relPath: string): Zone {
  if (relPath.startsWith("lib/core/")) return "core";
  if (relPath.startsWith("lib/methods/uniform/")) return "method-uniform";
  if (relPath.startsWith("lib/methods/losasso/")) return "method-losasso";
  if (relPath.startsWith("lib/methods/power/")) return "method-power";
  if (relPath.startsWith("lib/methods/adaptive-mass/")) return "method-adaptive-mass";
  if (relPath.startsWith("lib/methods/octree-shared/")) return "octree-shared";
  if (relPath.startsWith("lib/sparse-world/")) return "sparse-world";
  if (relPath.startsWith("lib/svo/")) return "svo";
  if (relPath.startsWith("lib/harness/")) return "harness";
  if (relPath.startsWith("shape-lab/")) return "shape-lab";
  if (relPath.startsWith("lib/")) return "lib-other";
  if (relPath.startsWith("components/") || relPath.startsWith("app/") || relPath.startsWith("worker/"))
    return "ui";
  return "tooling";
}

const ALLOWED: Record<Zone, ReadonlySet<Zone>> = {
  // Core composes the SVO layer: the production renderer draws through it.
  // The reverse direction is what carries the meaning — see "svo" below.
  core: new Set<Zone>(["core", "svo", "sparse-world"]),
  "method-uniform": new Set<Zone>(["core", "method-uniform"]),
  "method-losasso": new Set<Zone>(["core", "octree-shared", "method-losasso"]),
  "method-power": new Set<Zone>(["core", "octree-shared", "method-power"]),
  "method-adaptive-mass": new Set<Zone>([
    "core", "method-adaptive-mass", "sparse-world",
  ]),
  "octree-shared": new Set<Zone>(["core", "octree-shared"]),
  // The method edge exists only under sparse-world/internal during Phase A;
  // a dedicated public-index audit below keeps it out of the semantic API.
  "sparse-world": new Set<Zone>(["core", "sparse-world", "method-adaptive-mass"]),
  // The SVO stack encodes and traces voxels. Which solver filled them is not
  // its business, so no method zone appears here — that absence is the rule.
  svo: new Set<Zone>(["core", "svo"]),
  harness: new Set<Zone>([
    "core",
    "harness",
    "svo",
    "method-uniform",
    "method-losasso",
    "method-power",
    "method-adaptive-mass",
    "octree-shared",
    "sparse-world",
  ]),
  // The lab expands SVO primitives and field programs on CPU — that is what it
  // is for. It stays barred from every method, which is the rule that matters.
  "shape-lab": new Set<Zone>(["core", "svo", "shape-lab"]),
  "lib-other": new Set<Zone>([
    "core",
    "lib-other",
    "svo",
    "method-uniform",
    "method-losasso",
    "method-power",
    "method-adaptive-mass",
    "octree-shared",
    "sparse-world",
    "harness",
  ]),
  // The UI presents the render layer's own diagnostics (pixel traces, SVO
  // render receipts), which is the same composition relationship core has with
  // it. What the UI must never reach is a method — that stays absent here.
  ui: new Set<Zone>(["core", "ui", "svo", "shape-lab"]),
  tooling: new Set<Zone>([
    "core",
    "tooling",
    "harness",
    "svo",
    "method-uniform",
    "method-losasso",
    "method-power",
    "method-adaptive-mass",
    "octree-shared",
    "sparse-world",
    "lib-other",
    "shape-lab",
    "ui",
  ]),
};

function listSources(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name.startsWith(".") || name === "dist" || name === "build") continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) listSources(p, out);
    else if (/\.(ts|tsx)$/.test(name)) out.push(p);
  }
  return out;
}

function resolveSpec(fromFile: string, spec: string): string | null {
  let base: string;
  if (spec.startsWith(".")) base = resolve(dirname(fromFile), spec);
  else if (spec.startsWith("@/")) base = resolve(REPO, spec.slice(2));
  else return null;
  for (const c of [base, `${base}.ts`, `${base}.tsx`, join(base, "index.ts"), join(base, "index.tsx")]) {
    if (existsSync(c) && statSync(c).isFile()) return c;
  }
  return null;
}

/**
 * The files permitted to import `lib/methods` — the only module that names
 * every method.
 *
 * These are module-graph entry points, and there is one per graph rather than
 * one per product: the server components under `app/` never evaluate in the
 * browser, the two outermost `use client` modules never evaluate on the
 * server, and the render worker is its own thread again. Installing in each is
 * how the product gets wired, so the edge is the design rather than a leak.
 * The eight SVO commands are listed in `tools/check-method-install.ts`; tooling
 * may reach anything, so they need no exception here.
 */
const COMPOSITION_ROOTS = new Set<string>([
  "app/layout.tsx",
  "components/AppShell.tsx",
  "components/SceneLibrary.tsx",
  "lib/core/webgpu-render-worker.ts",
]);

const METHOD_CATALOG = "lib/methods/index.ts";

function main() {
  const strict = process.argv.includes("--strict");
  const violations: string[] = [];
  const roots = ["lib", "components", "app", "worker", "shape-lab", "tools", "tests"];
  for (const root of roots) {
    for (const file of listSources(join(REPO, root))) {
      const rel = relative(REPO, file);
      const from = zoneOf(rel);
      const src = readFileSync(file, "utf8");
      SPEC_RE.lastIndex = 0;
      let m: RegExpExecArray | null;
      while ((m = SPEC_RE.exec(src))) {
        const target = resolveSpec(file, m[2]);
        if (!target) continue;
        const targetRel = relative(REPO, target);
        if (targetRel === METHOD_CATALOG) {
          if (!COMPOSITION_ROOTS.has(rel) && from !== "tooling" && from !== "harness") {
            violations.push(`${rel} [${from}] → ${targetRel} [method-catalog]`);
          }
          continue;
        }
        const to = zoneOf(targetRel);
        if (from === "sparse-world" && to.startsWith("method-")
          && !rel.startsWith("lib/sparse-world/internal/")) {
          violations.push(`${rel} [${from}] → ${targetRel} [${to}]`);
          continue;
        }
        if (!ALLOWED[from].has(to)) {
          violations.push(`${rel} [${from}] → ${relative(REPO, target)} [${to}]`);
        }
      }
    }
  }
  if (!violations.length) {
    console.log("module boundaries: clean");
    return;
  }
  const byEdge = new Map<string, number>();
  for (const v of violations) {
    const key = v.replace(/^.*\[(.*?)\] → .*\[(.*?)\]$/, "$1 → $2");
    byEdge.set(key, (byEdge.get(key) ?? 0) + 1);
  }
  console.log(`module boundaries: ${violations.length} violation(s)`);
  for (const [k, n] of [...byEdge].sort((a, b) => b[1] - a[1])) console.log(`  ${n.toString().padStart(5)}  ${k}`);
  console.log("");
  for (const v of violations.slice(0, 200)) console.log(`  ${v}`);
  if (violations.length > 200) console.log(`  … ${violations.length - 200} more`);
  if (strict) process.exit(1);
}

main();
