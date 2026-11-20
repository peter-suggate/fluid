/** Pour · M1 Max: the pour rewrite's single-scene reference solver (docs/plans/pour-rewrite-2026-11-20.md), the
 * brick engine's NB-FLIP physics on nbflip-figure-7-pour with every stage built for that scene. It refuses any other
 * scene, and any change to the pour's geometry or nozzle. Picked from the toolstrip's "Fluid solver" select on the pour
 * scene (or ?scene=nbflip-figure-7-pour&method=pour-m1max). */
import type { GPUInitializationPhase } from "../core/gpu-initialization";
import type { GPUInitializationReporter, SimulationMethod } from "../core/method-contract";
import { NBFLIP_POUR_SCENE_ID } from "../core/nbflip-paper-scenes";
import { composeFeatures } from "../framework/composition";
import { POUR_METHOD_ID, createPourSolver } from "./pour";

export { POUR_METHOD_ID };
export const POUR_METHOD_SCENES: readonly string[] = [NBFLIP_POUR_SCENE_ID];

const resolveComposition = () => composeFeatures({
  features: [{ id: "simulation.pour-m1max.host", provides: ["simulation.vertex-phi"] }],
  selections: {},
});

const POUR_PHASES: Record<string, GPUInitializationPhase> = { bake: "planning", compile: "solver-pipelines", upload: "upload", ready: "attach" };
const pourProgress = (report: GPUInitializationReporter) => ({ stage, fraction }: { stage: string; fraction: number }) =>
  report({ phase: POUR_PHASES[stage] ?? "warmup", taskId: `pour.${stage}`, label: `Pour: ${stage}`, completed: Math.round(100 * fraction), total: 100 });

export const pourMethod: SimulationMethod = {
  id: POUR_METHOD_ID,
  label: "Pour · M1 Max (experimental)",
  shortLabel: "Pour M1 Max",
  badge: "POUR · M1 MAX · EXPERIMENTAL",
  description: "A dense brick-resident NB-FLIP solver built for the NB-FLIP paper's pour alone.",
  detail: "Vertex level set over 8^3 bricks with FLIP particles in a three-cell band, the nozzle's swept plug, warm-started multigrid pressure, all on compacted active-brick lists.",
  backend: "webgpu",
  composition: resolveComposition(),
  resolveComposition,
  resource: {
    id: "fluid.pour-m1max", label: "Pour M1 Max fluid", lane: "fluid",
    provides: ["fluid-authority", "water-presentation"], blocks: "transport",
    phaseCopy: { planning: "Baking the pour's glass and nozzle.", "solver-pipelines": "Compiling the pour's transport and pressure programs.", upload: "Uploading the empty box.", attach: "Attaching the pour solver." },
  },
  qualityLabels: { balanced: "Fixed numerics", high: "Fixed numerics", ultra: "Fixed numerics" },
  showQualityControl: false,
  scenes: POUR_METHOD_SCENES,
  params: [],
  pressureMapping: "Warm-started multigrid V-cycles until the divergence tolerance, capped.",
  presetFor: () => ({}),
  effectiveStep_s: () => 1 / 60,
  createSolverAsync: (device, scene, _quality, _values, _loads, onProgress, signal) => {
    if (!POUR_METHOD_SCENES.includes(scene.sceneId ?? "")) {
      return Promise.reject(new Error(`${POUR_METHOD_ID} runs only ${POUR_METHOD_SCENES.join(", ")}; scene "${scene.sceneId ?? "(unnamed)"}" must use another method.`));
    }
    return createPourSolver(device, scene, { onProgress: pourProgress(onProgress), signal });
  },
};
