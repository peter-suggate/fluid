import type { SceneDescription } from "./model";
import { findSceneryNode, sceneSceneryGraph, withSceneryNode } from "./scenery-edit";
import type { SceneryGraph, SceneryMaterial, SceneryNode } from "./scenery-graph";
import { lightChromaticity, type LinearRgb } from "./light-color";

/**
 * The lamps in the scenery, as things with a strength and a colour.
 *
 * A fixture is not a separate kind of object. It is any scenery part tagged
 * `light` — the renderer's light table reads the same tag, see
 * `proxyPhysicalLight` — and its emission and colour are its material's. So a
 * lantern moves, scales and deletes as the scenery it is, and these helpers
 * only add the two numbers a lamp has that a stone does not.
 *
 * A top-level node may hold several emitters (a lantern's ember and its glow):
 * the strength shown is the brightest, and a new strength scales them all by
 * the same ratio, so the fixture keeps its own internal balance.
 */

/** A part carrying a material, which is every leaf that can emit. */
type MaterialNode = Extract<SceneryNode, { readonly material: SceneryMaterial }>;

function hasMaterial(node: SceneryNode): node is MaterialNode {
  return "material" in node && node.material !== undefined;
}

function isEmitter(node: SceneryNode): node is MaterialNode {
  return hasMaterial(node) && (node.tags?.includes("light") ?? false);
}

function* partsOf(node: SceneryNode): Generator<SceneryNode> {
  yield node;
  const children = (node as { readonly children?: readonly SceneryNode[] }).children;
  for (const child of children ?? []) yield* partsOf(child);
}

function mapParts(node: SceneryNode, map: (part: SceneryNode) => SceneryNode): SceneryNode {
  const mapped = map(node);
  const children = (mapped as { readonly children?: readonly SceneryNode[] }).children;
  return children ? { ...mapped, children: children.map((child) => mapParts(child, map)) } as SceneryNode : mapped;
}

function resolvedColor(material: SceneryMaterial, graph: SceneryGraph): LinearRgb {
  if ("colorLinear" in material) return material.colorLinear;
  const tint = graph.palettes[material.palette]?.tint ?? [1, 1, 1];
  return [material.value * tint[0], material.value * tint[1], material.value * tint[2]];
}

export interface SceneryLightFixture {
  /** The brightest emitter's emission. */
  readonly intensity: number;
  /** The brightest emitter's colour, brightest channel at 1. */
  readonly colorLinear: LinearRgb;
}

export function sceneryLightFixture(scene: SceneDescription, nodeId: string): SceneryLightFixture | undefined {
  const node = findSceneryNode(scene, nodeId);
  if (!node) return undefined;
  const emitters = [...partsOf(node)].filter(isEmitter);
  if (emitters.length === 0) return undefined;
  const brightest = emitters.reduce((best, part) =>
    (part.material.emission ?? 0) > (best.material.emission ?? 0) ? part : best);
  return {
    intensity: brightest.material.emission ?? 0,
    colorLinear: lightChromaticity(resolvedColor(brightest.material, sceneSceneryGraph(scene))),
  };
}

/**
 * Every emitter scaled so the brightest reads `intensity`.
 *
 * From zero there is no ratio to keep, so a fixture switched fully off comes
 * back with every emitter at the new value.
 */
export function withSceneryLightStrength(scene: SceneDescription, nodeId: string, intensity: number): SceneDescription {
  const current = sceneryLightFixture(scene, nodeId)?.intensity ?? 0;
  const next = Math.max(0, intensity);
  return withSceneryNode(scene, nodeId, (node) => mapParts(node, (part) => {
    if (!isEmitter(part)) return part;
    const emission = current > 0 ? (part.material.emission ?? 0) * next / current : next;
    return { ...part, material: { ...part.material, emission } } as SceneryNode;
  }));
}

/**
 * Every emitter recoloured, each keeping its own brightness of surface.
 *
 * A palette material becomes an explicit colour: the palette is a shared ramp
 * the rest of the set may also use, and recolouring one lamp must not recolour
 * the stone that shares its palette.
 */
export function withSceneryLightColor(scene: SceneDescription, nodeId: string, colorLinear: LinearRgb): SceneDescription {
  const graph = sceneSceneryGraph(scene);
  const chroma = lightChromaticity(colorLinear);
  return withSceneryNode(scene, nodeId, (node) => mapParts(node, (part) => {
    if (!isEmitter(part)) return part;
    const previous = resolvedColor(part.material, graph);
    const peak = Math.max(previous[0], previous[1], previous[2], 1e-3);
    const { emission, surface } = part.material;
    const material: SceneryMaterial = {
      colorLinear: [chroma[0] * peak, chroma[1] * peak, chroma[2] * peak],
      ...(emission !== undefined ? { emission } : {}),
      ...(surface !== undefined ? { surface } : {}),
    };
    return { ...part, material } as SceneryNode;
  }));
}
