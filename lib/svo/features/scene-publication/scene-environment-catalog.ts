import type { SceneDescription } from "../../../core/model";
import { classifySceneryChange } from "../../../core/scenery-construction-key";
import { expandSceneryGraph } from "../../../core/scenery-expand";
import { isSceneryShellNode, type SceneryGraph } from "../../../core/scenery-graph";
import { terrainContentStamp } from "../../../core/terrain";
import {
  adoptEnvironmentProxyCatalog,
  buildEnvironmentProxyCatalog,
  cachedEnvironmentProxyCatalog,
  environmentCatalogPaletteSources,
  environmentPaletteProbes,
  environmentPaletteProbeSource,
  environmentPaletteProbeValue,
  environmentSceneryContext,
  holdEnvironmentCatalogPaletteSources,
  type EnvironmentProxyCatalog,
  type EnvironmentProxyMaterial,
  type EnvironmentProxyPrimitive,
} from "../../../core/voxel-environments";
import { ProxyBuilder } from "../../../core/voxel-scenery/index";

/** A recoloured catalog's predecessor: the same primitives in the same places, some under other materials. */
const recolouredFrom = new WeakMap<EnvironmentProxyCatalog, EnvironmentProxyCatalog>();

/**
 * The catalog this one was recoloured from, if it was. The two agree on every
 * field of every primitive but `material`, and share each primitive whose
 * material did not move — so what was derived from the predecessor's geometry
 * holds for this one, and what moved is found by identity.
 */
export function environmentCatalogRecolouredFrom(catalog: EnvironmentProxyCatalog): EnvironmentProxyCatalog | undefined {
  return recolouredFrom.get(catalog);
}

function sameMaterial(a: EnvironmentProxyMaterial, b: EnvironmentProxyMaterial): boolean {
  return a.colorLinear[0] === b.colorLinear[0] && a.colorLinear[1] === b.colorLinear[1]
    && a.colorLinear[2] === b.colorLinear[2] && a.emission === b.emission && a.roughness === b.roughness
    && a.surface === b.surface;
}

/** The inputs to expansion that are not the graph: the frame scenery is placed in. */
function sceneryFrame(scene: SceneDescription): string {
  const c = scene.container;
  return `${scene.environment ?? "default"}|${c.width_m}:${c.height_m}:${c.depth_m}|${scene.voxelDomain.finestCellSize_m}`
    + `|${scene.voxelDomain.detailCellSize_m ?? "cell"}|${terrainContentStamp(scene.terrain)}`;
}

/**
 * The previous catalog with the surfaces an appearance change reached, and
 * nothing else, replaced.
 *
 * Only the changed top-level nodes are expanded again, alone, and only for
 * their materials: an appearance field cannot move a primitive, so the span a
 * node published keeps its count, order, keys and owner indices, and every
 * primitive whose material came out the same is the same object it was.
 * Undefined when the expansion did something to a colour that a palette entry
 * cannot describe; the caller then expands the document whole.
 */
function recolouredCatalog(
  previous: EnvironmentProxyCatalog,
  scene: SceneDescription,
  graph: SceneryGraph,
  changedNodes: readonly number[],
  changedPalettes: readonly string[],
  detailCellSize_m: number | undefined,
): EnvironmentProxyCatalog | undefined {
  const shellNode = graph.nodes.find(isSceneryShellNode);
  if (!shellNode) throw new Error("A scenery graph needs exactly one shell node");
  const shellCount = previous.shell.primitives.length;
  const total = shellCount + previous.primitives.length;
  const context = environmentSceneryContext(scene, previous.environmentId,
    detailCellSize_m === undefined ? {} : { detailCellSize_m });
  const sources = environmentCatalogPaletteSources(previous);
  // Which positions name a palette is the builder's to say; a catalog it could
  // not say it for is expanded from the document.
  if (!sources && changedPalettes.length > 0) return undefined;
  const names = sources ? [...sources.names] : Object.keys(graph.palettes);
  for (const name in graph.palettes) if (!names.includes(name)) names.push(name);
  const probes = environmentPaletteProbes(names);
  const palette = sources ? new Int16Array(sources.palette) : new Int16Array(total);
  const value = sources ? new Float64Array(sources.value) : new Float64Array(total);
  const materials = new Map<number, EnvironmentProxyMaterial>();
  let separable = true;
  // Position `first` onward takes the materials of `emitted`, which must be the
  // primitives already published there.
  const adopt = (first: number, emitted: readonly EnvironmentProxyPrimitive[], published: readonly EnvironmentProxyPrimitive[]): void => {
    if (emitted.length !== published.length) {
      throw new Error(`Scenery appearance change republished ${emitted.length} primitives where ${published.length} stood`);
    }
    emitted.forEach((primitive, offset) => {
      if (primitive.key !== published[offset]!.key || primitive.kind !== published[offset]!.kind) {
        throw new Error(`Scenery appearance change replaced ${published[offset]!.key} with ${primitive.key}`);
      }
      const source = environmentPaletteProbeSource(primitive.material.colorLinear, names.length);
      if (source === undefined) { separable = false; return; }
      palette[first + offset] = source;
      value[first + offset] = source < 0 ? 0 : environmentPaletteProbeValue(primitive.material.colorLinear);
      materials.set(first + offset, primitive.material);
    });
  };
  const expand = (nodes: SceneryGraph["nodes"]): ProxyBuilder => {
    const builder = new ProxyBuilder(previous.environmentId);
    expandSceneryGraph(builder, { ...graph, palettes: probes, nodes }, context);
    return builder;
  };
  let shell = previous.shell;
  let span = 0;
  const changed = new Set(changedNodes);
  graph.nodes.forEach((node, index) => {
    if (isSceneryShellNode(node)) {
      if (!changed.has(index)) return;
      const builder = expand([node]);
      adopt(0, builder.shell, previous.shell.primitives);
      return;
    }
    const { from, to } = previous.spans[span++]!;
    if (!changed.has(index)) return;
    adopt(shellCount + from, expand([shellNode, node]).props, previous.primitives.slice(from, to));
  });
  if (!separable) return undefined;
  if (shellNode.materialModel !== shell.materialModel) shell = { ...shell, materialModel: shellNode.materialModel };
  if (sources && changedPalettes.length > 0) {
    // A tint reaches every position naming its palette, expanded just now or not.
    const tinted = new Set(changedPalettes.map((name) => names.indexOf(name)));
    for (let position = 0; position < total; position += 1) {
      if (!tinted.has(palette[position]!) || materials.has(position)) continue;
      const published = position < shellCount ? previous.shell.primitives[position]! : previous.primitives[position - shellCount]!;
      materials.set(position, published.material);
    }
  }
  if (materials.size === 0 && shell === previous.shell) return previous;
  let shellPrimitives = previous.shell.primitives, primitives = previous.primitives;
  for (const [position, material] of materials) {
    const published = position < shellCount ? previous.shell.primitives[position]! : previous.primitives[position - shellCount]!;
    let colorLinear = material.colorLinear;
    if (palette[position]! >= 0) {
      const tint = graph.palettes[names[palette[position]!]!]?.tint;
      if (!tint) throw new Error(`Scenery names unknown palette ${names[palette[position]!]}`);
      const v = value[position]!;
      colorLinear = [v * tint[0], v * tint[1], v * tint[2]];
    }
    const next = { ...material, colorLinear };
    if (sameMaterial(published.material, next)) continue;
    if (position < shellCount) {
      if (shellPrimitives === previous.shell.primitives) shellPrimitives = [...shellPrimitives];
      (shellPrimitives as EnvironmentProxyPrimitive[])[position] = { ...published, material: next };
    } else {
      if (primitives === previous.primitives) primitives = [...primitives];
      (primitives as EnvironmentProxyPrimitive[])[position - shellCount] = { ...published, material: next };
    }
  }
  if (shellPrimitives !== previous.shell.primitives) shell = { ...shell, primitives: shellPrimitives };
  const catalog = shell === previous.shell && primitives === previous.primitives
    ? previous : { ...previous, shell, primitives };
  if (catalog !== previous) recolouredFrom.set(catalog, previous);
  if (sources) holdEnvironmentCatalogPaletteSources(catalog, { names, palette, value });
  return catalog;
}

/**
 * Catalogs drawn at a detail cell their document does not carry, by document.
 * The document's own are kept with it by `voxel-environments`.
 */
const refinedCatalogs = new WeakMap<SceneDescription, Map<number, EnvironmentProxyCatalog>>();

function heldCatalog(scene: SceneDescription, detailCellSize_m: number | undefined): EnvironmentProxyCatalog | undefined {
  return detailCellSize_m === undefined ? cachedEnvironmentProxyCatalog(scene) : refinedCatalogs.get(scene)?.get(detailCellSize_m);
}

function holdCatalog(scene: SceneDescription, detailCellSize_m: number | undefined, catalog: EnvironmentProxyCatalog): void {
  if (detailCellSize_m === undefined) { adoptEnvironmentProxyCatalog(scene, catalog); return; }
  let held = refinedCatalogs.get(scene);
  if (!held) refinedCatalogs.set(scene, held = new Map());
  held.set(detailCellSize_m, catalog);
}

/**
 * The catalog of a document that follows another, at a cost proportional to
 * what separates them.
 *
 * The same scenery is the same catalog. Scenery that differs only in
 * appearance is the previous catalog recoloured, so geometry, owner indices and
 * every untouched primitive object are shared and a consumer can find what
 * changed by identity. Anything else is expanded from the document.
 *
 * `detailCellSize_m` is the voxel the caller draws at, for a caller drawing
 * finer than the document says; its catalogs succeed one another the same way.
 */
export function successorEnvironmentCatalog(
  previous: SceneDescription | undefined,
  next: SceneDescription,
  detailCellSize_m?: number,
): EnvironmentProxyCatalog {
  const environmentId = next.environment ?? "default";
  const authored = next.voxelDomain.detailCellSize_m ?? next.voxelDomain.finestCellSize_m;
  const detail = detailCellSize_m === authored ? undefined : detailCellSize_m;
  const held = heldCatalog(next, detail);
  if (held) return held;
  const published = previous && previous !== next ? heldCatalog(previous, detail) : undefined;
  if (published && previous && (previous.scenery === next.scenery || (previous.scenery && next.scenery))
    && sceneryFrame(previous) === sceneryFrame(next)) {
    const change = classifySceneryChange(previous.scenery, next.scenery);
    const catalog = change.kind === "same" ? published
      : change.kind === "appearance"
        ? recolouredCatalog(published, next, next.scenery!, change.nodes, change.palettes, detail)
        : undefined;
    if (catalog) {
      holdCatalog(next, detail, catalog);
      return catalog;
    }
  }
  const catalog = buildEnvironmentProxyCatalog(next, environmentId, detail === undefined ? {} : { detailCellSize_m: detail });
  if (detail !== undefined) holdCatalog(next, detail, catalog);
  return catalog;
}
