/**
 * The hero scene's composition, in placeholders — and the A/B that measures it.
 *
 *     FLUID_SVO_DRY_FRAME_SCENE_MODULE=tools/preview/hero-layout.ts
 *
 * The blocking set is part of the `hero-garden-hose` document itself, so this
 * module and the product show the same thing — which is the point, and why at
 * the default fidelity it adds nothing of its own.
 *
 * `FLUID_HERO_LAYOUT_FIDELITY` picks which of three scenes to render, and the
 * three exist to be measured against one another at one engine revision:
 *
 *   blocking  (default) the crude stand-ins the composition is judged on
 *   budget    every row's *detailed* leaf allowance, as cheap primitives filling
 *             the same envelope. Not meant to be looked at; it is the load test,
 *             and it is how the cost of a fully populated hero scene gets
 *             measured while the art is still a table
 *   none      the vessel and the hose alone, which is the baseline the other two
 *             are read against. It belongs in this file rather than in a
 *             separate one: a baseline taken at a different revision of a scene
 *             four people are editing is not a baseline.
 */
import { withHeroLayout, type LayoutFidelity } from "../../lib/voxel-scenery/hero-layout";
import { heroPreviewCamera, heroPreviewScene } from "./hero-still";

const fidelity = process.env.FLUID_HERO_LAYOUT_FIDELITY ?? "blocking";

export const createScene = () => {
  const scene = heroPreviewScene();
  if (fidelity === "blocking" || !scene.scenery) return scene;
  // An empty `include` keeps the replace-then-append path and appends nothing,
  // so the baseline is the same document with the set lifted out rather than a
  // second document that might differ in some other way too.
  const options = fidelity === "none" ? { include: [] } : { fidelity: fidelity as LayoutFidelity };
  return { ...scene, scenery: withHeroLayout(scene, options) };
};
export const camera = heroPreviewCamera();
