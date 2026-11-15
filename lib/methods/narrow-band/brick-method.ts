/**
 * NB-FLIP's engine switch (lane E). The narrow-band-flip method id takes
 * `engine: "uniform" | "brick"`:
 * - "uniform" (the default) runs the old engine;
 * - "brick" runs the brick engine in lib/methods/narrow-band/.
 *
 * A scene the brick engine cannot run fails explicitly, naming every gap,
 * and its user runs it on engine "uniform". The gaps are:
 * - the scene's (brick-scene.ts: M2 enables the dam only, without solids,
 *   bodies or sources);
 * - the lanes': F's stages and P's solver register in their own files
 *   (brick-fluid-stages.ts, brick-pressure-solver.ts). The bring-up CG never
 *   registers.
 */
import type { SceneDescription } from "../../core/model";
import type { GPUQuality } from "../../core/gpu-quality";
import { fluidLatticeFarPhi } from "../../core/initial-liquid-lattice";
import { BrickLayout } from "./brick-abi";
import { brickFluidStages } from "./brick-fluid-stages";
import { brickPressureSolver } from "./brick-pressure-solver";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import { brickSceneSetup } from "./brick-scene";
import { BrickNBFlipSolver } from "./brick-solver";
import type { BrickStagesFactory } from "./brick-stages";

export const NB_ENGINES = ["uniform", "brick"] as const;
export type NBEngine = typeof NB_ENGINES[number];

export const nbEngine = (value: unknown): NBEngine => {
  if (value === undefined || value === "uniform") return "uniform";
  if (value === "brick") return "brick";
  throw new Error(`NB-FLIP: unknown engine ${JSON.stringify(value)} (expected ${NB_ENGINES.join(" or ")})`);
};

/** The registered lane stages and solver (lane-owned registration files), or what is missing. */
export function brickEngineLanes(fluid: BrickStagesFactory | undefined = brickFluidStages, pressure: PressureSolverFactory | undefined = brickPressureSolver): string[] {
  const gaps: string[] = [];
  if (!fluid) gaps.push("lane F's stages (particles, level set, velocity) are not registered (brick-fluid-stages.ts)");
  if (!pressure) gaps.push("lane P's pressure solver is not registered (brick-pressure-solver.ts)");
  return gaps;
}

/** Why the brick engine cannot run this scene yet; empty when it can. */
export function brickEngineGaps(scene: SceneDescription, values: Readonly<Record<string, unknown>> = {}): string[] {
  return [...brickSceneSetup(scene, values).gaps, ...brickEngineLanes()];
}

/** The refusal for a scene the brick engine cannot run, naming every gap. */
export function brickEngineRefusal(scene: SceneDescription, values: Readonly<Record<string, unknown>> = {}): Error {
  const gaps = brickEngineGaps(scene, values);
  return new Error(`NB-FLIP brick engine cannot run scene "${scene.sceneId ?? "(unnamed)"}": ${gaps.join("; ")}. Run it with engine "uniform".`);
}

/** The app's brick solver for a scene, or the explicit refusal. */
export async function createBrickSolver(device: GPUDevice, scene: SceneDescription, values: Readonly<Record<string, unknown>>, quality: GPUQuality): Promise<BrickNBFlipSolver> {
  const { setup, gaps } = brickSceneSetup(scene, values), lanes = brickEngineLanes();
  if (!setup || gaps.length || lanes.length || !brickFluidStages || !brickPressureSolver) throw brickEngineRefusal(scene, values);
  const stages = brickFluidStages, solver = brickPressureSolver;
  const probe = solver(new BrickLayout({ lattice: setup.lattice, ...setup.capacities }));
  if (probe.bringUp) throw new Error("NB-FLIP brick engine: the registered pressure solver is the bring-up CG, which no scene may run (coordinator ruling, 2026-11-15)");
  return BrickNBFlipSolver.create(device, { setup, stages, solver, values, quality, farPhi_m: fluidLatticeFarPhi(scene) });
}
