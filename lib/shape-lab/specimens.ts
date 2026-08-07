/**
 * The hero scene's shapes, one at a time, at whatever leaf you want to see them
 * at.
 *
 * ## Why this exists
 *
 * Every art-direction pass on the garden has been run the same way: change a
 * number in a species, build a whole scene, take the exclusive GPU lock,
 * voxelize 2 200 records and render a frame — minutes per look, on a picture
 * where seven other objects and the grade are competing for the eye. Two passes
 * were spent tuning parameters inside functions that could not express the
 * target shape at all, and neither noticed, because the loop was too slow to ask
 * the only question that mattered: *what is this function actually drawing?*
 *
 * `tools/shape-lab.ts` answered that for one hard-coded specimen on the CPU. This
 * is the same idea with the two things that tool could not have: **every** shape
 * the hero scene publishes, and a parameter you can move while looking at it.
 *
 * ## What it is not
 *
 * It is not a second definition of any shape. Nothing here describes geometry.
 * The chain is exactly the product's own —
 *
 *     createHeroGardenHoseSceneWithSet(lattice)   the document, at that leaf
 *       -> buildEnvironmentProxyCatalog             the generators expand
 *         -> svoDescriptorForEnvironmentProxy       the record the SVO publishes
 *
 * — and the lab only chooses *which* node to expand and *what leaf* to expand it
 * at. A shape that draws wrong here draws wrong in the scene, because it is the
 * same call.
 *
 * ## The leaf is an input, not a viewer setting
 *
 * `environmentRefinementDepth` moves two separate things and the lab has to show
 * both. It sets the voxel a record is drawn into — the terracing — *and* it is
 * the number every generator's legibility floor is measured in, so a bonsai
 * expanded at 6.25 mm is a different tree from the same bonsai at 0.78 mm before
 * a single voxel is placed. That is why the depth selector rebuilds the document
 * rather than re-rendering the one it had: `SceneLattice` exists because writing
 * a cell size onto a finished scene moves the octree and nothing that feeds it.
 */
import { HERO_GARDEN_CELL_M, HERO_GARDEN_CONTAINER } from "../hero-garden-scene";
import type { SceneDescription } from "../model";
import { createHeroGardenHoseSceneWithSet } from "../scenes";
import type { SceneryGraph, SceneryNode } from "../scenery-graph";
import { svoDescriptorForEnvironmentProxy } from "../svo-scene-primitives";
import type { SvoPrimitiveDescriptor } from "../svo-primitive-abi";
import { terrainSampleGrid, type TerrainGrid } from "../terrain";

import { buildEnvironmentProxyCatalog } from "../voxel-environments";
import type { PondVesselSpec } from "../voxel-scenery/pond-vessel";

/** The lattice every hero document is authored on: 6.25 mm. */
export const SHAPE_LAB_CELL_M = HERO_GARDEN_CELL_M;

/** Refinement rungs the lab offers. Production renders at 3. */
export const SHAPE_LAB_DEPTHS: readonly number[] = Object.freeze([0, 1, 2, 3, 4]);

/** The leaf a rung draws into, which is also the leaf its generators expand at. */
export const shapeLabLeaf_m = (depth: number): number => SHAPE_LAB_CELL_M / 2 ** Math.max(0, Math.trunc(depth));

/**
 * How fine the lab bakes the ground, whatever the leaf is.
 *
 * The scene's own rule is half the detail cell — 0.39 mm at depth 3, which is a
 * 4608 x 3072 grid and fourteen million samples. That is sound for a render that
 * happens once and fatal for a control the user is dragging. The lab caps it at
 * 1.5625 mm (1153 x 769), and what it gives up is sub-voxel: the ground under a
 * specimen can sit up to about one depth-3 voxel from where the scene would put
 * it. Nothing about a *shape* is decided at that scale.
 */
export const SHAPE_LAB_GROUND_SPACING_M = 0.0015625;

export interface ShapeLabRecord {
  readonly descriptor: SvoPrimitiveDescriptor;
  /** The proxy's own linear colour, for the material shading mode. */
  readonly colorLinear: readonly [number, number, number];
  readonly group: string;
  readonly key: string;
}

/**
 * One selectable thing.
 *
 * `nodeIds` rather than one id because the hose is six top-level nodes — five
 * swept-tube runs and a ferrule — and "the hose" is what a person wants to look
 * at. Editing stays per node; only the framing is shared.
 */
export interface ShapeLabSpecimen {
  readonly id: string;
  readonly label: string;
  /** What kind of thing this is, for the list: `generator · bonsai`, `cluster`. */
  readonly detail: string;
  readonly nodeIds: readonly string[];
}

/** The pond itself, which is a heightfield rather than any record. */
export const SHAPE_LAB_VESSEL_ID = "vessel";

export interface ShapeLabWorld {
  readonly scene: SceneDescription;
  readonly graph: SceneryGraph;
  /** Top-level scenery nodes, shell excluded, in document order. */
  readonly nodes: readonly SceneryNode[];
  readonly specimens: readonly ShapeLabSpecimen[];
  readonly vessel: PondVesselSpec;
  readonly leaf_m: number;
  readonly buildMs: number;
}

/**
 * Keys the form hides.
 *
 * Two kinds, and the second is worth naming separately. Most are **identity**:
 * which one this is rather than what shape it is. `orders` is not — it is the
 * oak's branch table, four objects of eight fields, and it is hidden because the
 * lab renders it faithfully as thirty-two controls behind four disclosure
 * triangles. That is the right shape for a species *definition* and the wrong
 * shape for something dragged while looking at a tree, so the table is edited in
 * `oak-branching.ts` and the lab gets `branchDensity` and `branchFineness`,
 * which scale it. Delete the entry to get the raw table back.
 */
const IDENTITY_KEYS = new Set([
  "kind", "id", "generator", "group", "tags", "vessel", "material", "materialModel",
  "orders",
]);

export function shapeLabHiddenKey(key: string): boolean {
  return IDENTITY_KEYS.has(key);
}

/** The first path segment, which is how the hero graph already groups its nodes. */
function specimenGroupOf(nodeId: string): string {
  const cut = nodeId.indexOf("/");
  return cut < 0 ? nodeId : nodeId.slice(0, cut);
}

function specimenDetail(nodes: readonly SceneryNode[]): string {
  if (nodes.length === 1) {
    const node = nodes[0];
    return node.kind === "generator" ? `generator · ${node.generator}` : node.kind;
  }
  const kinds = [...new Set(nodes.map((node) => node.kind))];
  return `${nodes.length} nodes · ${kinds.join(", ")}`;
}

/**
 * Build the hero document at one rung, with a set of nodes replaced.
 *
 * Overrides are whole nodes rather than parameter patches. A generator's shape
 * lives in `params`, a capsule's lives in `from`/`to`/`radius`, and a group's
 * lives in its children — one rule that covers all three is a node swap, and it
 * is also the only form that survives a round trip through the editor's JSON.
 */
export function shapeLabWorld(options: {
  readonly depth: number;
  readonly nodeOverrides?: Readonly<Record<string, SceneryNode>>;
  readonly vessel?: PondVesselSpec;
}): ShapeLabWorld {
  const started = Date.now();
  const leaf_m = shapeLabLeaf_m(options.depth);
  const built = createHeroGardenHoseSceneWithSet({ cellSize_m: SHAPE_LAB_CELL_M, detailCellSize_m: leaf_m });
  const authored = built.scenery;
  if (!authored) throw new Error("The hero garden document has no scenery graph");
  const vessel = options.vessel ?? authored.vessels?.pond;
  if (!vessel) throw new Error("The hero garden graph declares no pond vessel");
  // The vessel is named in two places and both have to move together: the graph's
  // table is what a coping or a stone set lays its runs against, and the terrain
  // description is what every specimen is *seated* on. Patching one would put the
  // stones on a bank the ground no longer has.
  const graph: SceneryGraph = { ...authored, vessels: { ...authored.vessels, pond: vessel } };
  const authoredTerrain = built.terrain;
  const procedural = authoredTerrain?.procedural;
  const scene: SceneDescription = {
    ...built,
    scenery: graph,
    terrain: authoredTerrain && procedural
      ? {
        ...authoredTerrain,
        baseHeight_m: vessel.groundHeight_m,
        procedural: procedural.kind === "pond-vessel"
          ? { ...procedural, spec: vessel, spacing_m: Math.max(SHAPE_LAB_GROUND_SPACING_M, leaf_m / 2) }
          : { ...procedural, spacing_m: Math.max(SHAPE_LAB_GROUND_SPACING_M, leaf_m / 2) },
      }
      : authoredTerrain,
  };
  const overrides = options.nodeOverrides ?? {};
  const nodes = graph.nodes
    .filter((node) => node.kind !== "terrain-shell" && node.kind !== "room-shell")
    .map((node) => overrides[node.id] ?? node);
  const groups = new Map<string, SceneryNode[]>();
  for (const node of nodes) {
    const key = specimenGroupOf(node.id);
    const list = groups.get(key);
    if (list) list.push(node); else groups.set(key, [node]);
  }
  const specimens: ShapeLabSpecimen[] = [
    {
      id: SHAPE_LAB_VESSEL_ID,
      label: "Pond vessel",
      detail: "heightfield · ground, walls, basin",
      nodeIds: [],
    },
    ...[...groups].map(([key, members]) => ({
      id: key,
      label: key,
      detail: specimenDetail(members),
      nodeIds: members.map((node) => node.id),
    })),
  ];
  return { scene, graph, nodes, specimens, vessel, leaf_m, buildMs: Date.now() - started };
}

/**
 * Expand one specimen and hand back the records the SVO would publish for it.
 *
 * The graph is cut down to the shell and the wanted nodes before expansion, so a
 * bonsai costs a bonsai rather than the whole garden. Everything a species reads
 * beyond its own parameters — the vessel it rings, the ground it stands on — is
 * on the scene and the graph's tables, both of which survive the cut.
 */
export function shapeLabRecords(world: ShapeLabWorld, nodeIds: readonly string[]): {
  readonly records: readonly ShapeLabRecord[];
  readonly expandMs: number;
} {
  if (nodeIds.length === 0) return { records: [], expandMs: 0 };
  const started = Date.now();
  const wanted = new Set(nodeIds);
  const shell = world.graph.nodes.find((node) => node.kind === "terrain-shell" || node.kind === "room-shell");
  const nodes = [
    ...(shell ? [shell] : []),
    ...world.nodes.filter((node) => wanted.has(node.id)),
  ];
  const scene: SceneDescription = { ...world.scene, scenery: { ...world.graph, nodes } };
  const catalog = buildEnvironmentProxyCatalog(scene, "garden");
  const spans = catalog.spans.filter((span) => wanted.has(span.nodeId));
  const records: ShapeLabRecord[] = [];
  for (const span of spans) {
    for (let index = span.from; index < span.to; index += 1) {
      const proxy = catalog.primitives[index];
      records.push({
        descriptor: svoDescriptorForEnvironmentProxy(scene, proxy),
        colorLinear: proxy.material.colorLinear,
        group: proxy.group,
        key: proxy.key,
      });
    }
  }
  return { records, expandMs: Date.now() - started };
}

/** The ground, baked once per document, as the renderer's own bilinear grid. */
export function shapeLabGround(world: ShapeLabWorld): TerrainGrid | undefined {
  return terrainSampleGrid(world.scene.terrain);
}

/**
 * Does this specimen's *geometry* move when the leaf does?
 *
 * The question the depth selector cannot answer on its own, and the one that
 * makes it look broken when it is working. Two things move with the rung — the
 * voxel a record is drawn into, and whatever the generator derived from the leaf
 * — and for most of the hero set the second is nothing at all: the bonsai
 * publishes the same 154 records at every rung, the hose the same 6, and only
 * the rosettes change count. A depth control that looks dead on a specimen whose
 * shape is genuinely leaf-independent is a control that will be mistrusted on
 * the specimens where it is not.
 *
 * Compared with a relative tolerance rather than by identity, because a
 * generator seats itself against a heightfield baked at a spacing that follows
 * the leaf, so *every* record moves by a micron or two between rungs. That is
 * the ground being resampled, not the shape being re-derived, and reporting it
 * as a change would make the answer "yes" everywhere and mean nothing.
 */
const SHAPE_LAB_SAME_TOLERANCE = 1e-5;

function sameNumber(a: number, b: number): boolean {
  return Math.abs(a - b) <= SHAPE_LAB_SAME_TOLERANCE * Math.max(1, Math.abs(a), Math.abs(b));
}

function sameShape(a: unknown, b: unknown): boolean {
  if (typeof a === "number" && typeof b === "number") return sameNumber(a, b);
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((entry, index) => sameShape(entry, b[index]));
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const left = Object.keys(a as object), right = Object.keys(b as object);
    if (left.length !== right.length) return false;
    return left.every((key) => key in (b as object)
      && sameShape((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]));
  }
  return a === b;
}

export interface ShapeLabRung {
  readonly depth: number;
  readonly leaf_m: number;
  readonly records: number;
  /**
   * Whether the records differ from the **previous** rung's by more than ground
   * reseating.
   *
   * Against the previous rather than against the coarsest, because the useful
   * question is *where does this specimen stop re-deriving itself*. Measured on
   * the hero set: the bonsai moves from depth 0 to 2 — its canopy plates are
   * floored at one leaf thick, so they thin until the authored flatten takes
   * over — and then stops; the rosettes go on adding blades to depth 4; the hose
   * never moves at all, because a swept tube derives nothing from the lattice.
   */
  readonly moved: boolean;
}

export function shapeLabDepthLadder(options: {
  readonly specimenId: string;
  readonly nodeOverrides?: Readonly<Record<string, SceneryNode>>;
  readonly vessel?: PondVesselSpec;
}): readonly ShapeLabRung[] {
  let previous: SvoPrimitiveDescriptor[] = [];
  return SHAPE_LAB_DEPTHS.map((depth, index) => {
    const world = shapeLabWorld({ depth, ...options });
    const specimen = world.specimens.find((entry) => entry.id === options.specimenId);
    const descriptors = shapeLabRecords(world, specimen?.nodeIds ?? []).records
      .map((record) => record.descriptor);
    const moved = index > 0 && !sameShape(previous, descriptors);
    previous = descriptors;
    return { depth, leaf_m: world.leaf_m, records: descriptors.length, moved };
  });
}

/** The container the lattice is anchored in, so the lab's voxels land where the scene's do. */
export const SHAPE_LAB_CONTAINER = HERO_GARDEN_CONTAINER;
