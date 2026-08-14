/**
 * Emits the file move plan for the fluid-core / method-plugin layout
 * (docs/method-decoupling-handoff.md §2.1). Feed the output to
 * `tools/move-modules.ts`.
 *
 *   node --import tsx tools/plan-method-layout.ts <stage> > plan.json
 *
 * Stages are applied in order so each lands as one reviewable commit:
 *   svo       lib/svo-*, lib/webgpu-svo-*, lib/webgpu-live-svo-scene  -> lib/svo/
 *   uniform   the uniform solver family                               -> lib/methods/uniform/
 *   losasso   the losasso lane                                        -> lib/methods/losasso/
 *   power     the power-liquids lane                                  -> lib/methods/power/
 *   shared    the remaining octree modules                            -> lib/methods/octree-shared/
 *   harness   smoke catalog / diagnostic packs / evidence             -> lib/harness/
 *   shapelab  the CPU art-direction route                             -> shape-lab/
 *   core      everything left in lib/                                 -> lib/core/
 */
import { existsSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const REPO = resolve(new URL("..", import.meta.url).pathname);

const libFiles = () =>
  readdirSync(resolve(REPO, "lib")).filter((f) => /\.tsx?$/.test(f));

const POWER_NAMES = [
  /^octree-power-/, /^webgpu-octree-power-/, /^octree-spgrid-/, /^webgpu-octree-spgrid-/,
  /^octree-structured-/, /^webgpu-octree-structured-/, /^structured-/, /^power-liquids-/,
  /^webgpu-octree-section43-/, /^webgpu-octree-air-velocity-support/,
];
const LOSASSO_NAMES = [/^octree-losasso-/, /^webgpu-octree-losasso-/];
const UNIFORM_NAMES = [/^uniform-/, /^webgpu-uniform-/];
const SVO_NAMES = [/^svo-/, /^webgpu-svo-/, /^webgpu-live-svo-scene/];
const SHARED_NAMES = [/^octree-/, /^webgpu-octree/, /^sparse-brick-octree/];
const HARNESS_NAMES = [
  /^scene-webgpu-smoke/, /^scene-diagnostic-/, /^scene-custom-diagnostic-/,
  /^scene-evidence-/, /^scene-diagnostics\./, /^scene-.*-diagnostic\./,
];

/**
 * Harness modules that today sit under tools/.
 *
 * `webgpu-smoke-*` is the acceptance harness a lane runs through, and the two
 * probes are imported by the diagnostic implementations in lib/ — a lib module
 * reaching into tools/ is the boundary violation the move resolves. The
 * `run-*` entry points stay in tools/: those are commands, not library code.
 */
const HARNESS_TOOLS = [
  /^webgpu-smoke-/, /^brick-quad-coverage-probe/, /^ocean-wave-propagation-probe/,
  /^scene-diagnostic-probe/,
];

const match = (name: string, patterns: readonly RegExp[]) =>
  patterns.some((p) => p.test(name));

function stage(name: string): Record<string, string> {
  const moves: Record<string, string> = {};
  const add = (from: string, to: string) => {
    if (existsSync(resolve(REPO, from))) moves[from] = to;
  };
  const files = libFiles();
  switch (name) {
    case "svo":
      for (const f of files) if (match(f, SVO_NAMES)) add(`lib/${f}`, `lib/svo/${f}`);
      break;
    case "uniform":
      for (const f of files) if (match(f, UNIFORM_NAMES)) add(`lib/${f}`, `lib/methods/uniform/${f}`);
      add("lib/methods/uniform.ts", "lib/methods/uniform/method.ts");
      break;
    case "losasso":
      for (const f of files) if (match(f, LOSASSO_NAMES)) add(`lib/${f}`, `lib/methods/losasso/${f}`);
      break;
    case "power":
      for (const f of files) if (match(f, POWER_NAMES)) add(`lib/${f}`, `lib/methods/power/${f}`);
      // The catalog is generated data the power lane owns, so its module and
      // its binary travel together. Two callers reach the binary through a
      // plain string rather than a specifier and are fixed up separately.
      add("lib/generated/octree-power-catalog.ts", "lib/methods/power/generated/octree-power-catalog.ts");
      add("lib/generated/octree-power-catalog.bin", "lib/methods/power/generated/octree-power-catalog.bin");
      break;
    case "shared":
      for (const f of files) {
        if (match(f, POWER_NAMES) || match(f, LOSASSO_NAMES)) continue;
        if (match(f, SHARED_NAMES)) add(`lib/${f}`, `lib/methods/octree-shared/${f}`);
      }
      break;
    case "harness":
      for (const f of files) if (match(f, HARNESS_NAMES)) add(`lib/${f}`, `lib/harness/${f}`);
      for (const f of readdirSync(resolve(REPO, "tools"))) {
        if (/\.tsx?$/.test(f) && match(f, HARNESS_TOOLS)) add(`tools/${f}`, `lib/harness/${f}`);
      }
      break;
    case "shapelab":
      for (const f of readdirSync(resolve(REPO, "lib/shape-lab"))) {
        add(`lib/shape-lab/${f}`, `shape-lab/${f}`);
      }
      add("components/ShapeLab.tsx", "shape-lab/ShapeLab.tsx");
      add("components/shape-lab-styles.ts", "shape-lab/shape-lab-styles.ts");
      break;
    case "core":
      for (const f of files) add(`lib/${f}`, `lib/core/${f}`);
      for (const dir of ["stores", "simulation", "voxel-scenery", "generated"]) {
        const d = resolve(REPO, "lib", dir);
        if (!existsSync(d)) continue;
        for (const f of readdirSync(d)) {
          if (/\.tsx?$/.test(f)) add(`lib/${dir}/${f}`, `lib/core/${dir}/${f}`);
        }
      }
      break;
    default:
      throw new Error(`unknown stage ${name}`);
  }
  return moves;
}

const requested = process.argv[2];
if (!requested) throw new Error("usage: plan-method-layout.ts <stage>");
console.log(JSON.stringify({ moves: stage(requested) }, null, 2));
