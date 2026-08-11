import type { SceneDescription } from "./model";
import { findSceneryNode, sceneSceneryGraph, withSceneryNode } from "./scenery-edit";
import type {
  FoliageDensityForm,
  SceneryNode,
  SceneryRecursiveShapeNode,
} from "./scenery-graph";

/**
 * Three dials over a canopy's density field, phrased the way a tree reads.
 *
 * The raw `FoliageDensityForm` is six coupled numbers, and most of their joint
 * settings look like static, not foliage. These dials move along the curves
 * that stay tree-like:
 *
 *  - **coverage** — how much of the crown envelope carries leaves. Lowering it
 *    raises the iso threshold *and* drains the interior fill together, so
 *    foliage retreats into distinct clumps around the branch tips with sky
 *    between them — the way a real crown thins — instead of dissolving into a
 *    uniform mist the way threshold alone does.
 *  - **clumpSize** — the period of the low-frequency cluster octave: the size
 *    of the leaf masses and, equally, of the voids between them. Log-spaced,
 *    because a mass reads by its ratio to the crown, not by metres.
 *  - **breakup** — the mix between the cluster octave and the fine detail
 *    octave: low is smooth cloud masses, high is a frizzy leaf-speckled edge.
 *
 * `dotSpacing_m` is deliberately not a dial: it is also the voxelizer's
 * conservative feature radius, so shrinking it silently multiplies rebuild
 * cost. All outputs stay inside `validateSceneryGraph`'s ranges by
 * construction.
 */
export interface CanopyDials {
  /** Foliage coverage of the envelope, in [0, 1]; 0 is nearly bare. */
  readonly coverage: number;
  /** Size of the leaf masses and their voids, in [0, 1]; log-spaced. */
  readonly clumpSize: number;
  /** Fine leaf speckle versus smooth cloud masses, in [0, 1]. */
  readonly breakup: number;
}

// Coverage sweeps the iso threshold down while filling the interior. The
// authored hero canopy (threshold .5002, bias .093) sits at coverage ≈ 0.67 on
// these curves, so selecting the tree changes nothing until a dial moves.
const THRESHOLD_BARE = 0.66;
const THRESHOLD_FULL = 0.42;
const INTERIOR_BIAS_FULL = 0.14;
// Clump periods in metres, spanning "many fist-sized tufts" to "three or four
// crown-scale boughs" on the hero's ~0.6 m crown.
const CLUMP_PERIOD_MIN_M = 0.09;
const CLUMP_PERIOD_MAX_M = 0.30;
// Detail share; the cluster octave takes the complement so the weights always
// sum to one and the validator's positive-weight rule holds at both ends.
const DETAIL_WEIGHT_MIN = 0.08;
const DETAIL_WEIGHT_MAX = 0.55;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const unlerp = (a: number, b: number, value: number) => clamp01((value - a) / (b - a));

/** The dial positions a pad's stored density reads back as. */
export function canopyDials(pad: SceneryRecursiveShapeNode): CanopyDials {
  const density = pad.form.density;
  return {
    coverage: unlerp(THRESHOLD_BARE, THRESHOLD_FULL, density.threshold),
    clumpSize: clamp01(
      Math.log(density.clusterPeriod_m / CLUMP_PERIOD_MIN_M)
      / Math.log(CLUMP_PERIOD_MAX_M / CLUMP_PERIOD_MIN_M),
    ),
    breakup: unlerp(DETAIL_WEIGHT_MIN, DETAIL_WEIGHT_MAX, density.detailWeight),
  };
}

function densityForDials(current: FoliageDensityForm, dials: CanopyDials): FoliageDensityForm {
  const coverage = clamp01(dials.coverage);
  const detailWeight = lerp(DETAIL_WEIGHT_MIN, DETAIL_WEIGHT_MAX, clamp01(dials.breakup));
  return {
    ...current,
    threshold: lerp(THRESHOLD_BARE, THRESHOLD_FULL, coverage),
    interiorBias: lerp(0, INTERIOR_BIAS_FULL, coverage),
    clusterPeriod_m: CLUMP_PERIOD_MIN_M
      * (CLUMP_PERIOD_MAX_M / CLUMP_PERIOD_MIN_M) ** clamp01(dials.clumpSize),
    detailWeight,
    clusterWeight: 1 - detailWeight,
  };
}

/**
 * Every *active* foliage pad under a node: the leaves of the recursive crown.
 * A pad with children has delegated its volume to them and publishes nothing,
 * so its density is not a canopy anyone can see.
 */
export function canopyPads(node: SceneryNode): readonly SceneryRecursiveShapeNode[] {
  if (node.kind === "group") return node.children.flatMap(canopyPads);
  if (node.kind !== "recursive-shape") return [];
  return node.children === undefined ? [node] : node.children.flatMap(canopyPads);
}

/** The active pads of one top-level scenery node, or none. */
export function sceneCanopyPads(
  scene: SceneDescription,
  nodeId: string,
): readonly SceneryRecursiveShapeNode[] {
  const node = findSceneryNode(scene, nodeId);
  return node ? canopyPads(node) : [];
}

function mapCanopyPads(
  node: SceneryNode,
  update: (pad: SceneryRecursiveShapeNode) => SceneryRecursiveShapeNode,
): SceneryNode {
  if (node.kind === "group") {
    return { ...node, children: node.children.map((child) => mapCanopyPads(child, update)) };
  }
  if (node.kind !== "recursive-shape") return node;
  if (node.children !== undefined) {
    return {
      ...node,
      children: node.children.map((child) =>
        mapCanopyPads(child, update) as SceneryRecursiveShapeNode),
    };
  }
  return update(node);
}

/**
 * The scene with every active canopy pad under `nodeId` set to `dials`.
 *
 * All of a tree's pads move together: the dials describe the specimen, and a
 * crown whose clumps disagree about their own scale reads as two trees
 * interleaved. Editing keeps each pad's envelope, seed and `dotSpacing_m`, so
 * the masses stay *where* they were — only how much foliage condenses there
 * changes.
 */
export function withCanopyDials(
  scene: SceneDescription,
  nodeId: string,
  dials: CanopyDials,
): SceneDescription {
  return withSceneryNode(scene, nodeId, (node) => mapCanopyPads(node, (pad) => ({
    ...pad,
    form: { ...pad.form, density: densityForDials(pad.form.density, dials) },
  })));
}

/**
 * The canopy state of every tree in the scene, as one compact string:
 * `nodeId~coverage,clumpSize,breakup` per canopy-bearing top-level node,
 * ";"-joined in document order.
 *
 * This is the round-trip form. The full density field never travels: the dials
 * are the whole authoring surface, three 3-decimal numbers reproduce any
 * setting a slider can reach, and re-reading a rounded value re-rounds to the
 * same string, so a carried canopy compares equal to itself on the next diff.
 */
export function sceneCanopyQuery(scene: SceneDescription): string {
  const round = (value: number) => String(Math.round(value * 1000) / 1000);
  return sceneSceneryGraph(scene).nodes
    .map((node) => ({ node, pads: canopyPads(node) }))
    .filter(({ pads }) => pads.length > 0)
    .map(({ node, pads }) => {
      const dials = canopyDials(pads[0]!);
      return `${node.id}~${round(dials.coverage)},${round(dials.clumpSize)},${round(dials.breakup)}`;
    })
    .join(";");
}

/** Apply a `sceneCanopyQuery` string; unknown nodes and malformed runs are skipped. */
export function withSceneCanopyQuery(scene: SceneDescription, query: string): SceneDescription {
  let next = scene;
  for (const run of query.split(";")) {
    const separator = run.lastIndexOf("~");
    if (separator <= 0) continue;
    const nodeId = run.slice(0, separator);
    const values = run.slice(separator + 1).split(",").map(Number);
    if (values.length !== 3 || values.some((value) => !Number.isFinite(value))) continue;
    if (sceneCanopyPads(next, nodeId).length === 0) continue;
    next = withCanopyDials(next, nodeId, {
      coverage: values[0]!,
      clumpSize: values[1]!,
      breakup: values[2]!,
    });
  }
  return next;
}
