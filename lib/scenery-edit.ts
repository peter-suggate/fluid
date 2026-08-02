import type { EnvironmentId } from "./environments";
import type { SceneDescription } from "./model";
import { sceneryGraphForEnvironment } from "./scenery-presets";
import type { SceneryGraph, SceneryNode, SceneryPlacement } from "./scenery-graph";

/**
 * Document operations on a scene's scenery.
 *
 * Every edit is an ordinary change to `scene.scenery` — there is no override
 * layer, no patch list and no second authority. That is the whole reason the
 * description lives in the document: an object the editor moves and an object
 * the preset placed are the same kind of thing, and the second one is not
 * privileged.
 *
 * Only *top-level* nodes are addressed here. A node's children are its own
 * business: a lantern moves as a lantern, and its cap is not separately
 * placeable in the document even though it is separately expanded.
 */

/**
 * The graph a scene presents. A scene that has never been given one falls back
 * to its environment's seed, which is what `buildEnvironmentProxyCatalog` also
 * renders — so an edit to a fallback graph makes it concrete rather than
 * changing what is on screen.
 */
export function sceneSceneryGraph(scene: SceneDescription): SceneryGraph {
  return scene.scenery
    ?? sceneryGraphForEnvironment(scene, (scene.environment as EnvironmentId | undefined) ?? "default");
}

export function findSceneryNode(scene: SceneDescription, id: string): SceneryNode | undefined {
  return sceneSceneryGraph(scene).nodes.find((node) => node.id === id);
}

/** Append nodes, keeping document order — which is expansion and owner order. */
export function withSceneryNodes<T extends SceneDescription>(
  scene: T,
  nodes: readonly SceneryNode[],
): T {
  const graph = sceneSceneryGraph(scene);
  return { ...scene, scenery: { ...graph, nodes: [...graph.nodes, ...nodes] } };
}

/**
 * Replace one top-level node in place.
 *
 * In place, not appended: owner indices are the GPU material table's own
 * addresses, so moving an object must not renumber every object after it.
 */
export function withSceneryNode<T extends SceneDescription>(
  scene: T,
  id: string,
  replace: (node: SceneryNode) => SceneryNode,
): T {
  const graph = sceneSceneryGraph(scene);
  return {
    ...scene,
    scenery: { ...graph, nodes: graph.nodes.map((node) => node.id === id ? replace(node) : node) },
  };
}

/** Patch a node's placement, leaving the rest of it alone. */
export function withSceneryPlacement<T extends SceneDescription>(
  scene: T,
  id: string,
  patch: Partial<SceneryPlacement>,
): T {
  return withSceneryNode(scene, id, (node) => ({ ...node, place: { ...node.place, ...patch } }));
}

export function withoutSceneryNode<T extends SceneDescription>(scene: T, id: string): T {
  const graph = sceneSceneryGraph(scene);
  return { ...scene, scenery: { ...graph, nodes: graph.nodes.filter((node) => node.id !== id) } };
}

/** First free `<prefix>-<n>`, matching the rigid-body naming convention. */
export function nextSceneryNodeId(scene: SceneDescription, prefix: string): string {
  const taken = new Set(sceneSceneryGraph(scene).nodes.map((node) => node.id));
  let index = 1;
  while (taken.has(`${prefix}-${index}`)) index += 1;
  return `${prefix}-${index}`;
}
