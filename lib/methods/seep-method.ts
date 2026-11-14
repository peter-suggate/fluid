/** Seep · M1 Max: the experimental dense brick solver of the seep rewrite
 * (docs/plans/seep-rewrite-2026-11-14.md), wrapping lane S's createSeepSolver.
 * It is built for reservoir-seep alone and refuses any other scene by name:
 * it has no rigid coupling, no scheduled drops, no live solid edits and no
 * sources, so a scene that needs any of them must not open under it.
 * Opened with ?scene=reservoir-seep&method=seep-m1max; NB-FLIP stays the default. */
import type { GPUInitializationPhase } from "../core/gpu-initialization";
import type { GPUInitializationReporter, SimulationMethod } from "../core/method-contract";
import { composeFeatures } from "../framework/composition";
import { SEEP_METHOD_ID, createSeepSolver } from "./seep";

export { SEEP_METHOD_ID };
export const SEEP_METHOD_SCENES: readonly string[] = ["reservoir-seep"];

const resolveComposition = () => composeFeatures({
  features: [{ id: "simulation.seep-m1max.host", provides: ["simulation.vertex-phi"] }],
  selections: {},
});

/** Lane S's construction stages as the app's initialization phases. */
const SEEP_PHASES: Record<string, GPUInitializationPhase> = { bake: "planning", compile: "solver-pipelines", upload: "upload", ready: "attach" };
const seepProgress = (report: GPUInitializationReporter) => ({ stage, fraction }: { stage: string; fraction: number }) =>
  report({ phase: SEEP_PHASES[stage] ?? "warmup", taskId: `seep.${stage}`, label: `Seep: ${stage}`, completed: Math.round(100 * fraction), total: 100 });

export const seepMethod: SimulationMethod = {
  id: SEEP_METHOD_ID,
  label: "Seep · M1 Max (experimental)",
  shortLabel: "Seep M1 Max",
  badge: "SEEP · M1 MAX · EXPERIMENTAL",
  description: "A dense brick-resident level-set solver with FLIP particles downstream of the throat, built for the reservoir seep alone.",
  detail: "Vertex level set over 8^3 bricks with work on compacted active-brick lists, warm-started multigrid pressure, and FLIP particles right of the seep wall.",
  backend: "webgpu",
  composition: resolveComposition(),
  resolveComposition,
  resource: {
    id: "fluid.seep-m1max", label: "Seep M1 Max fluid", lane: "fluid",
    provides: ["fluid-authority", "water-presentation"], blocks: "transport",
    phaseCopy: { planning: "Baking the seep's solid and initial level set.", "solver-pipelines": "Compiling the seep's transport and pressure programs.", upload: "Uploading the initial seep state.", attach: "Attaching the seep solver." },
  },
  qualityLabels: { balanced: "Fixed numerics", high: "Fixed numerics", ultra: "Fixed numerics" },
  showQualityControl: false,
  params: [],
  pressureMapping: "Warm-started multigrid V-cycles until the divergence tolerance, capped.",
  presetFor: () => ({}),
  effectiveStep_s: () => 1 / 60,
  createSolverAsync: (device, scene, _quality, _values, _loads, onProgress, signal) => {
    if (!SEEP_METHOD_SCENES.includes(scene.sceneId ?? "")) {
      return Promise.reject(new Error(`${SEEP_METHOD_ID} runs only ${SEEP_METHOD_SCENES.join(", ")}; scene "${scene.sceneId ?? "(unnamed)"}" must use another method.`));
    }
    return createSeepSolver(device, scene, { onProgress: seepProgress(onProgress), signal });
  },
};
