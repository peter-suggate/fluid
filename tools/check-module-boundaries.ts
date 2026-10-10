/** Plugin boundaries for the retained solver, shared features and SVO renderer.
 * The advance and shape labs are independent hosts. Only composition roots
 * install the simulation catalog; other modules resolve the contract registry.
 * Run with --strict to fail on a forbidden edge.
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const REPO = resolve(new URL("..", import.meta.url).pathname);
// `\bimport\s+` catches the bare side-effect import a composition root uses
// to install a plugin package; without it those edges go unpoliced.
const SPEC_RE = /(?:\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*|\bnew\s+URL\s*\(\s*)(["'])([^"']+)\1/g;

type Zone =
  | "framework"
  | "feature"
  | "composition"
  | "core"
  | "method-uniform"
  | "svo"
  | "harness"
  | "shape-lab"
  | "advance-lab"
  | "lib-other"
  | "ui"
  | "tooling";

function zoneOf(relPath: string): Zone {
  if (/\.test\.tsx?$/.test(relPath) || /\/(?:probes|verification|tests)\//.test(relPath)) return "tooling";
  if (relPath.startsWith("lib/framework/")) return "framework";
  if (relPath.startsWith("lib/features/ui/")) return "composition";
  if (relPath.startsWith("lib/features/")) return "feature";
  if (relPath.startsWith("lib/core/")) return "core";
  if (relPath.startsWith("lib/methods/uniform/")) return "method-uniform";
  if (relPath.startsWith("lib/svo/")) return "svo";
  if (relPath.startsWith("lib/harness/")) return "harness";
  if (relPath.startsWith("shape-lab/")) return "shape-lab";
  if (relPath.startsWith("advance-lab/")) return "advance-lab";
  if (relPath.startsWith("lib/")) return "lib-other";
  if (relPath.startsWith("components/") || relPath.startsWith("app/") || relPath.startsWith("worker/"))
    return "ui";
  return "tooling";
}

const ALLOWED: Record<Zone, ReadonlySet<Zone>> = {
  framework: new Set<Zone>(["framework"]),
  feature: new Set<Zone>(["framework", "core", "feature", "ui"]),
  composition: new Set<Zone>(["framework", "core", "feature", "composition", "ui", "svo", "method-uniform"]),
  // Core composes the SVO layer: the production renderer draws through it.
  // The reverse direction is what carries the meaning — see "svo" below.
  core: new Set<Zone>(["core", "svo", "framework", "feature"]),
  "method-uniform": new Set<Zone>(["core", "method-uniform"]),
  // The SVO stack encodes and traces voxels. Which solver filled them is not
  // its business, so no method zone appears here — that absence is the rule.
  svo: new Set<Zone>(["core", "svo"]),
  harness: new Set<Zone>([
    "core",
    "harness",
    "svo",
    "method-uniform",
  ]),
  // The lab expands SVO primitives and field programs on CPU — that is what it
  // is for. It stays barred from every method, which is the rule that matters.
  "shape-lab": new Set<Zone>(["core", "svo", "shape-lab"]),
  // The 2-D advance lab is a second *host*, not a second method: it renders the
  // same capability modules the studio does over a Rust/Wasm world. So it may
  // reach core, the shared UI primitives, the feature packages and the framework
  // that composes them, plus `lib/physics-wasm` (its world). Solver packages
  // remain private to their composition roots.
  "advance-lab": new Set<Zone>([
    "core",
    "ui",
    "feature",
    "composition",
    "framework",
    "lib-other",
    "advance-lab",
    "tooling",
  ]),
  "lib-other": new Set<Zone>([
    "core",
    "lib-other",
    "svo",
    "method-uniform",
    "harness",
  ]),
  // The UI presents the render layer's own diagnostics (pixel traces, SVO
  // render receipts), which is the same composition relationship core has with
  // it. What the UI must never reach is a method — that stays absent here.
  ui: new Set<Zone>(["core", "ui", "svo", "shape-lab", "advance-lab", "framework", "feature", "composition"]),
  tooling: new Set<Zone>([
    "feature",
    "composition",
    "core",
    "tooling",
    "harness",
    "svo",
    "method-uniform",
    "lib-other",
    "shape-lab",
    "advance-lab",
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
    else if (/\.(ts|tsx|mts)$/.test(name)) out.push(p);
  }
  return out;
}

function resolveSpec(fromFile: string, spec: string): string | null {
  let base: string;
  if (spec.startsWith(".")) base = resolve(dirname(fromFile), spec);
  else if (spec.startsWith("@/")) base = resolve(REPO, spec.slice(2));
  else return null;
  for (const c of [base, `${base}.ts`, `${base}.tsx`, `${base}.mts`, join(base, "index.ts"), join(base, "index.tsx")]) {
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
  // The 2-D advance lab is its own route and its own client boundary:
  // `app/advance-lab/page.tsx` is a server component, so nothing the studio's
  // shell installs is evaluated for it. This is the outermost `use client`
  // module that route reaches, which makes it the graph's entry point by the
  // same rule as the four above.
  "advance-lab/AdvanceLab.tsx",
]);

const METHOD_CATALOG = "lib/methods/index.ts";

function main() {
  const strict = process.argv.includes("--strict");
  const violations: string[] = [];
  const roots = ["lib", "components", "app", "worker", "shape-lab", "advance-lab", "tools", "tests"];
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
        // Feature-owned React adapters consume the shared UI primitives. Numerical
        // modules retain the prohibition against reaching the UI layer.
        const featureView = /\/(?:ui|controls)\.tsx$/.test(rel) && (from === "svo" || from.startsWith("method-"));
        const persistenceCatalog = rel === "lib/features/persistence.ts" && to === "svo";
        const sharedFeatureDefinition = from.startsWith("method-") && to === "feature" && (targetRel.endsWith("/definition.ts"));
        const frameworkConsumer = to === "framework" && from !== "framework";
        if (!ALLOWED[from].has(to) && !frameworkConsumer && !persistenceCatalog && !sharedFeatureDefinition && !(featureView && to === "ui")) {
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
