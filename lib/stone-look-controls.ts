import type { SceneDescription } from "./model";
import { findSceneryNode, sceneSceneryGraph, withSceneryNode } from "./scenery-edit";
import type { CappedBoulderGeneratorParams } from "./scenery-generators";
import type { SceneryGeneratorNode, SceneryNode } from "./scenery-graph";

/**
 * Three dials over a capped boulder's form, phrased the way a stone reads.
 *
 * The raw `CappedBoulderForm` is fourteen coupled numbers whose legal band is
 * the four authored family forms — most joint settings are a lamp, not a stone.
 * These dials move along the family's own axes:
 *
 *  - **size** — the cap radius: the whole stone's scale. Log-spaced, because a
 *    stone reads by its ratio to its neighbours, not by millimetres.
 *  - **squash** — the mushroom-to-cobble axis the four authored forms sit on:
 *    it flattens the cap while shortening the stem together, so a squat stone
 *    settles onto its slab the way the rounded cobble does instead of becoming
 *    a flat plate on a tall post.
 *  - **lip** — how far the cap overhangs its stem. The stem's base keeps the
 *    family's constant taper (+0.14 cap radii) so the foot always widens under
 *    the waist and the two solids never meet in a re-entrant corner.
 *
 * The other eleven numbers keep the node's authored values — the dials sculpt
 * the silhouette, not the seating — and every reachable setting stays inside
 * the band the four shipped forms span, so a slider cannot author a shape the
 * family would disown.
 */
export interface StoneDials {
  /** Cap radius, in [0, 1]; log-spaced over the family's sizes. */
  readonly size: number;
  /** Mushroom (0) to rounded cobble (1): flatter cap on a shorter stem. */
  readonly squash: number;
  /** Cap overhang past the stem, in [0, 1]; 0 is nearly flush. */
  readonly lip: number;
}

// Cap radii in metres. The hero family spans 0.030-0.080, so all four stones
// read back interior and the dial still reaches "pebble" and "twice the anchor".
const SIZE_MIN_M = 0.02;
const SIZE_MAX_M = 0.12;
// The squash axis: capFlatten sweeps past both ends of the authored 0.40-0.74
// band so no shipped form reads back pinned, and the stem height co-varies the
// way the family does (the mushroom's 0.95-radius stem down to the cobble's
// stub), clamped where the line would leave the legal range.
const FLATTEN_MUSHROOM = 0.34;
const FLATTEN_COBBLE = 0.80;
const STEM_HEIGHT_MUSHROOM = 1.05;
const STEM_HEIGHT_COBBLE = 0.15;
// The lip axis is the stem's top width: a narrow stem under a wide cap is an
// overhang. The four forms sit at 0.54-0.80, again interior to the sweep.
const STEM_TOP_FLUSH = 0.90;
const STEM_TOP_DEEP = 0.48;
// Every authored form tapers its stem by exactly this much from waist to foot.
const STEM_TAPER = 0.14;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const unlerp = (a: number, b: number, value: number) => clamp01((value - a) / (b - a));

/** The capped-boulder generator node `nodeId` names, or undefined. */
export function sceneStoneNode(
  scene: SceneDescription,
  nodeId: string,
): (SceneryGeneratorNode & { generator: "capped-boulder" }) | undefined {
  const node = findSceneryNode(scene, nodeId);
  return node?.kind === "generator" && node.generator === "capped-boulder" ? node : undefined;
}

/** The dial positions a boulder's stored form reads back as. */
export function stoneDials(params: CappedBoulderGeneratorParams): StoneDials {
  return {
    size: clamp01(Math.log(params.capRadius_m / SIZE_MIN_M) / Math.log(SIZE_MAX_M / SIZE_MIN_M)),
    squash: unlerp(FLATTEN_MUSHROOM, FLATTEN_COBBLE, params.capFlatten),
    lip: unlerp(STEM_TOP_FLUSH, STEM_TOP_DEEP, params.stemTopShare),
  };
}

function paramsForDials(
  current: CappedBoulderGeneratorParams,
  dials: StoneDials,
): CappedBoulderGeneratorParams {
  const squash = clamp01(dials.squash);
  const stemTopShare = lerp(STEM_TOP_FLUSH, STEM_TOP_DEEP, clamp01(dials.lip));
  return {
    ...current,
    capRadius_m: SIZE_MIN_M * (SIZE_MAX_M / SIZE_MIN_M) ** clamp01(dials.size),
    capFlatten: lerp(FLATTEN_MUSHROOM, FLATTEN_COBBLE, squash),
    stemHeightShare: Math.min(1, Math.max(
      STEM_HEIGHT_COBBLE,
      lerp(STEM_HEIGHT_MUSHROOM, STEM_HEIGHT_COBBLE, squash),
    )),
    stemTopShare,
    stemBaseShare: Math.min(0.98, stemTopShare + STEM_TAPER),
  };
}

/** The scene with the boulder at `nodeId` set to `dials`; a no-op elsewhere. */
export function withStoneDials(
  scene: SceneDescription,
  nodeId: string,
  dials: StoneDials,
): SceneDescription {
  if (sceneStoneNode(scene, nodeId) === undefined) return scene;
  return withSceneryNode(scene, nodeId, (node) =>
    node.kind === "generator" && node.generator === "capped-boulder"
      ? { ...node, params: paramsForDials(node.params, dials) }
      : node);
}

/**
 * The scene with the boulder's seed re-rolled: the same stone, another
 * individual. Deterministic — each press walks the same golden-ratio sequence —
 * so a re-rolled stone survives a URL round-trip exactly.
 */
export function withStoneSeedRerolled(scene: SceneDescription, nodeId: string): SceneDescription {
  if (sceneStoneNode(scene, nodeId) === undefined) return scene;
  return withSceneryNode(scene, nodeId, (node) =>
    node.kind === "generator" && node.generator === "capped-boulder"
      ? { ...node, seed: (node.seed + 0x9e3779b9) >>> 0 }
      : node);
}

const isStoneNode = (node: SceneryNode): node is SceneryGeneratorNode & { generator: "capped-boulder" } =>
  node.kind === "generator" && node.generator === "capped-boulder";

/**
 * The look of every boulder in the scene, as one compact string:
 * `nodeId~size,squash,lip,seed` per capped-boulder top-level node, ";"-joined
 * in document order.
 *
 * Same contract as `sceneCanopyQuery`: the dials are the whole authoring
 * surface, three 3-decimal numbers and the seed reproduce anything the flyout
 * can reach, and a carried value re-reads as the same string so an applied
 * query never makes the scene look edited on the next diff.
 */
export function sceneStoneQuery(scene: SceneDescription): string {
  const round = (value: number) => String(Math.round(value * 1000) / 1000);
  return sceneSceneryGraph(scene).nodes
    .filter(isStoneNode)
    .map((node) => {
      const dials = stoneDials(node.params);
      return `${node.id}~${round(dials.size)},${round(dials.squash)},${round(dials.lip)},${node.seed}`;
    })
    .join(";");
}

/** Apply a `sceneStoneQuery` string; unknown nodes and malformed runs are skipped. */
export function withSceneStoneQuery(scene: SceneDescription, query: string): SceneDescription {
  let next = scene;
  for (const run of query.split(";")) {
    const separator = run.lastIndexOf("~");
    if (separator <= 0) continue;
    const nodeId = run.slice(0, separator);
    const values = run.slice(separator + 1).split(",").map(Number);
    if (values.length !== 4 || values.some((value) => !Number.isFinite(value))) continue;
    if (sceneStoneNode(next, nodeId) === undefined) continue;
    next = withStoneDials(next, nodeId, {
      size: values[0]!,
      squash: values[1]!,
      lip: values[2]!,
    });
    const seed = values[3]!;
    next = withSceneryNode(next, nodeId, (node) =>
      node.kind === "generator" ? { ...node, seed } : node);
  }
  return next;
}
