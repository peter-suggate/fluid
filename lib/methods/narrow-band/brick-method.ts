/**
 * NB-FLIP's engine switch (lane E). The narrow-band-flip method id takes
 * `engine: "uniform" | "brick"`:
 * - "uniform" (the default) runs the old engine;
 * - "brick" runs the brick engine in lib/methods/narrow-band/.
 *
 * Until each stage lands, a scene the brick engine cannot run fails
 * explicitly, naming what is missing, and the uniform engine runs it.
 *
 * M1 has the brick set, the lists, the schedule and the pressure slot (with
 * the bring-up solver). Particle transport, the level set and the velocity
 * stage are lane F's, and bodies are lane P's. So every authored NB-FLIP scene
 * fails here today, as it must; the engine itself runs under its tests.
 */
import type { SceneDescription } from "../../core/model";

export const NB_ENGINES = ["uniform", "brick"] as const;
export type NBEngine = typeof NB_ENGINES[number];

export const nbEngine = (value: unknown): NBEngine => {
  if (value === undefined || value === "uniform") return "uniform";
  if (value === "brick") return "brick";
  throw new Error(`NB-FLIP: unknown engine ${JSON.stringify(value)} (expected ${NB_ENGINES.join(" or ")})`);
};

/** The brick engine's stages that have landed (M1). */
export const BRICK_ENGINE_LANDED = { brickSet: true, pressureSlot: true, particles: false, levelSet: false, velocity: false, bodies: false, solids: false, sources: false } as const;

/** Why the brick engine cannot run this scene yet; empty when it can. */
export function brickEngineGaps(scene: SceneDescription): string[] {
  const gaps: string[] = [];
  if (!scene.fluid?.brickPools) gaps.push("scene.fluid.brickPools is not authored (brick-abi.md §9)");
  if (!BRICK_ENGINE_LANDED.particles) gaps.push("particle transport (lane F: pTrace, rebin) has not landed");
  if (!BRICK_ENGINE_LANDED.levelSet) gaps.push("the level set (lane F: phiAdvect, union, redistance) has not landed");
  if (!BRICK_ENGINE_LANDED.velocity) gaps.push("the velocity stage (lane F: p2g, uAdvect, extension) has not landed");
  if ((scene.rigidBodies?.length ?? 0) > 0 && !BRICK_ENGINE_LANDED.bodies) gaps.push("rigid bodies (lane P) have not landed");
  if (scene.voxelDomain && !BRICK_ENGINE_LANDED.solids) gaps.push("static solids (solidStamp, M3) have not landed");
  if ((scene.fluid?.inflow || scene.fluid?.scheduledDrops?.length) && !BRICK_ENGINE_LANDED.sources) gaps.push("sources (lane F) have not landed");
  return gaps;
}

/** The refusal for a scene the brick engine cannot run, naming every gap. M1 has no host adapter, so it refuses every scene. */
export function brickEngineRefusal(scene: SceneDescription): Error {
  const gaps = brickEngineGaps(scene);
  return new Error(`NB-FLIP brick engine cannot run scene "${scene.sceneId ?? "(unnamed)"}": ${[...gaps, "its host adapter (M1) has not landed"].join("; ")}. Run it with engine "uniform".`);
}
