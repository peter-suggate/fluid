import { add, scale as scaleVec } from "./math";
import type { Quaternion, Vec3 } from "./model";
import { quaternionMultiply } from "./rigid-body";
import { growSceneryGenerator, type SceneryGeneratorRequest } from "./scenery-generators";
import { sceneryMaximumSwayExcursion_m } from "./scenery-sway";
import {
  isSceneryShellNode,
  type SceneryClusterNode,
  type SceneryGeneratorNode,
  type SceneryGraph,
  type SceneryMaterial,
  type SceneryNode,
  type SceneryPlacement,
  type SceneryPrimitiveNode,
  type SceneryRoomShellNode,
  type SceneryShellNode,
  type SceneryUnits,
} from "./scenery-graph";
import { validateSvoClusterPacking, type SvoSmoothUnionClusterPacking } from "./svo-primitive-abi";
import { svoFieldProgramExtent_m, validateSvoFieldProgram } from "./svo-field-program";
import {
  aabb,
  V,
  type EnvironmentLinearColor,
  type EnvironmentProxyShell,
  type EnvironmentSceneryContext,
  type ProxyBuilder,
} from "./voxel-scenery";
import { emitProceduralTree, planProceduralTree } from "./voxel-scenery/procedural-tree";
import { terrainHeightAt } from "./terrain";

/**
 * Turn a scene's declarative scenery into the primitive catalog every
 * downstream consumer reads.
 *
 * This is the whole of what `lib/voxel-scenery/<id>.ts` used to be: the modules
 * described geometry by running code, and this runs over a description instead.
 * Nothing else in the pipeline changes — the same `ProxyBuilder` emits the same
 * proxies in the same order, so owner indices stay dense and the SVO records,
 * static mips, coverage and sparse domain are all unaffected.
 *
 * Expansion order is document order, depth-first. That is load-bearing: owner
 * indices are the GPU material table's own addresses, so two builds of the same
 * graph must publish the same primitives in the same sequence. A `generator`
 * node is grown in place and its nodes visited where it stands, so the sequence
 * is the same one a hand-spliced set produced — which is what let six baked
 * generators become node kinds without a single owner index moving.
 */

/** A node's resolved place in the world, and the units its children speak in. */
interface SceneryFrame {
  /** World position of this frame's local origin. */
  readonly origin_m: Vec3;
  /** Metres per authored unit for lengths declared in this frame. */
  readonly unit_m: number;
  /** Cumulative uniform scale from every ancestor's `place.scale`. */
  readonly scale: number;
  readonly orientation?: Quaternion;
  readonly units: SceneryUnits;
  /** The selectable object children belong to unless they name their own. */
  readonly group: string;
}

/**
 * The primitives one top-level node published, as a half-open range of indices
 * into `builder.props`.
 *
 * This is what makes a described object a *thing* downstream. A lantern is
 * three primitives and a tree is thirty, and neither has a name in the catalog —
 * so the editor could not say what a click landed on, and the renderer could not
 * outline it, without being told where each object's primitives start and stop.
 * Ranges are contiguous because expansion is depth-first in document order.
 */
export interface ScenerySpan {
  readonly nodeId: string;
  /** Index of this node's first primitive in `builder.props`. */
  readonly from: number;
  /** One past its last. Equal to `from` for a node that published nothing. */
  readonly to: number;
}

/**
 * A dielectric pane the glass path traces, in world metres.
 *
 * Panes leave the catalog by this door rather than as proxies: an opaque
 * primitive standing where the glass is would occlude everything behind it, and
 * the pane's whole job is to be seen through.
 */
export interface SceneryPane {
  /** Unprefixed node id; the catalog namespaces it by environment. */
  readonly id: string;
  readonly center_m: Vec3;
  /** Half width and half height in the pane's own plane. */
  readonly half_m: readonly [number, number];
  readonly orientation?: Quaternion;
}

export interface SceneryExpansion {
  readonly shell: EnvironmentProxyShell;
  /** One entry per top-level node, in document order, shell excluded. */
  readonly spans: readonly ScenerySpan[];
  /** Declared glazing, shell openings first, then document order. */
  readonly panes: readonly SceneryPane[];
}

function rotate(orientation: Quaternion | undefined, v: Vec3): Vec3 {
  if (!orientation) return v;
  const { w, x, y, z } = orientation;
  // q * (0,v) * q^-1, expanded.
  const tx = 2 * (y * v.z - z * v.y);
  const ty = 2 * (z * v.x - x * v.z);
  const tz = 2 * (x * v.y - y * v.x);
  return {
    x: v.x + w * tx + (y * tz - z * ty),
    y: v.y + w * ty + (z * tx - x * tz),
    z: v.z + w * tz + (x * ty - y * tx),
  };
}

/**
 * Compose a node's placement onto its parent frame.
 *
 * `anchor: "terrain"` replaces the vertical datum with the ground height under
 * the node's own root rather than under each part, which is what keeps a tree's
 * canopy attached to its trunk on a slope. It is resolved per node and never
 * inherited: only the object that stands on the ground declares it, and its
 * parts are ordinary offsets from the datum it resolved.
 */
function childFrame(
  parent: SceneryFrame,
  node: SceneryNode,
  context: EnvironmentSceneryContext,
): SceneryFrame {
  const place: SceneryPlacement = node.place ?? {};
  const units = place.units ?? parent.units;
  const unit_m = units === "metres" ? 1 : context.s;
  const scale = parent.scale * (place.scale ?? 1);
  const local = place.position ?? V(0, 0, 0);
  const offset = scaleVec(local, unit_m * parent.scale);
  let origin_m = add(parent.origin_m, rotate(parent.orientation, offset));
  if (place.anchor === "terrain" || place.anchor === "floor") {
    const [groundX, groundZ] = place.ground ?? [local.x, local.z];
    // No terrain means a flat floor at the environment's own datum, which is
    // what a terrain-anchored node on a plain lawn should sit on.
    const datum = place.anchor === "floor" || !context.scene.terrain
      ? context.floorY_m
      : terrainHeightAt(context.scene.terrain, groundX * unit_m, groundZ * unit_m);
    origin_m = { x: origin_m.x, y: datum + offset.y, z: origin_m.z };
  }
  return {
    origin_m,
    unit_m,
    scale,
    orientation: place.orientation
      ? quaternionMultiply(parent.orientation ?? { w: 1, x: 0, y: 0, z: 0 }, place.orientation)
      : parent.orientation,
    units,
    group: node.group ?? (node.kind === "group" ? node.id : parent.group),
  };
}

function resolveMaterial(
  material: SceneryMaterial,
  graph: SceneryGraph,
): { color: EnvironmentLinearColor; emission: number } {
  const emission = material.emission ?? 0;
  if ("colorLinear" in material) {
    return { color: [material.colorLinear[0], material.colorLinear[1], material.colorLinear[2]], emission };
  }
  const palette = graph.palettes[material.palette];
  if (!palette) throw new Error(`Scenery names unknown palette ${material.palette}`);
  const v = material.value;
  return { color: [v * palette.tint[0], v * palette.tint[1], v * palette.tint[2]], emission };
}

/** A length authored in this frame's units, in metres. */
function metres(frame: SceneryFrame, value: number): number {
  return value * frame.unit_m * frame.scale;
}

function point(frame: SceneryFrame, local: Vec3): Vec3 {
  return add(frame.origin_m, rotate(frame.orientation, scaleVec(local, frame.unit_m * frame.scale)));
}

/**
 * Publish one node, inside its authored surface scope when it named one.
 *
 * The scope wraps rather than being passed because the builder's own emitters
 * take colour and emission and not a material — see `ProxyBuilder.surface`.
 * A node with no `surface` emits exactly as it did before this field existed,
 * so every set that has not been migrated keeps the name-regex inference.
 */
function emitPrimitive(
  builder: ProxyBuilder,
  node: SceneryPrimitiveNode,
  frame: SceneryFrame,
  graph: SceneryGraph,
): void {
  const surface = node.material.surface;
  if (surface) {
    builder.surface(surface, () => emitPrimitiveGeometry(builder, node, frame, graph));
    return;
  }
  emitPrimitiveGeometry(builder, node, frame, graph);
}

/**
 * The render ABI's packing for one aggregate node, in metres.
 *
 * Every length scales with the object and every proportion is left alone: an
 * aggregate at half its size wants lattice lobes half as large on a period half
 * as fine, and the same count and arrangement of them. Control points are
 * scaled but *not* placed — they are offsets inside the record's own frame,
 * which the builder centres at `frame.origin_m` and rotates by
 * `frame.orientation`, so running them through `point` would place them twice.
 *
 * The one validator lives in the render ABI, because every bound it enforces is
 * a property of the march rather than of scenery. It is called here anyway, and
 * its message re-thrown with the node's id, so an author sees which node in a
 * four-thousand-primitive graph is the one that will not publish.
 */
function clusterPacking(node: SceneryClusterNode, frame: SceneryFrame, envelope_m: Vec3): SvoSmoothUnionClusterPacking {
  const shared = { smoothRadius_m: metres(frame, node.smoothRadius), seed: node.seed };
  const packing: SvoSmoothUnionClusterPacking = node.field === "seeded-lobes"
    ? {
      ...shared,
      field: "seeded-lobes",
      lobeCount: node.lobeCount,
      anisotropy: node.anisotropy,
      // Spread rather than defaulted here: an omitted placement parameter has
      // exactly one default and it lives in the render ABI beside the field
      // that reads it.
      ...(node.lobeSpan === undefined ? {} : { lobeSpan: node.lobeSpan }),
      ...(node.lobeSpanSpread === undefined ? {} : { lobeSpanSpread: node.lobeSpanSpread }),
      ...(node.displacement === undefined ? {} : { displacement: node.displacement }),
    }
    : node.field === "tapered-sweep"
      ? {
        ...shared,
        field: "tapered-sweep",
        points: node.points.map((control) => ({
          position_m: scaleVec(control.position, frame.unit_m * frame.scale),
          radius_m: metres(frame, control.radius),
        })),
      }
      : {
        ...shared,
        latticeLobeRadius_m: metres(frame, node.floretRadius),
        latticePeriod_m: metres(frame, node.latticePeriod),
        jitter: node.jitter,
        ...(node.octaves === undefined ? {} : { octaves: node.octaves }),
      };
  try {
    validateSvoClusterPacking(packing, envelope_m);
  } catch (error) {
    throw new RangeError(`Scenery cluster node "${node.id}": ${(error as Error).message}`);
  }
  return packing;
}

function emitPrimitiveGeometry(
  builder: ProxyBuilder,
  node: SceneryPrimitiveNode,
  frame: SceneryFrame,
  graph: SceneryGraph,
): void {
  const { color, emission } = resolveMaterial(node.material, graph);
  const tags = node.tags ?? [];
  const group = node.group ?? frame.group;
  const centre = frame.origin_m;
  switch (node.kind) {
    case "box":
      builder.box(node.id, group, centre,
        scaleVec(node.halfSize, frame.unit_m * frame.scale), color, emission, tags, false, frame.orientation);
      return;
    case "cylinder":
      builder.cylinder(node.id, group, centre,
        metres(frame, node.radius), metres(frame, node.halfHeight), color, emission, tags, frame.orientation,
        node.edgeRadius === undefined ? 0 : metres(frame, node.edgeRadius));
      return;
    case "ellipsoid":
      builder.ellipsoid(node.id, group, centre,
        scaleVec(node.radius, frame.unit_m * frame.scale), color, emission, tags, frame.orientation);
      return;
    case "capsule":
      // Endpoints, not a centre: the builder derives the centre and the
      // orientation that takes local +Y onto the run, so a hose is authored as
      // the line it actually follows.
      builder.capsule(node.id, group, point(frame, node.from), point(frame, node.to),
        metres(frame, node.radius), color, emission, tags);
      return;
    case "torus":
      builder.torus(node.id, group, centre,
        metres(frame, node.majorRadius), metres(frame, node.minorRadius), color, emission, tags, frame.orientation);
      return;
    case "cone":
      builder.cone(node.id, group, centre,
        metres(frame, node.baseRadius), metres(frame, node.topRadius), metres(frame, node.halfHeight),
        color, emission, tags, frame.orientation, node.roundedEnds);
      return;
    case "cluster": {
      const envelope_m = scaleVec(node.lobe, frame.unit_m * frame.scale);
      builder.cluster(node.id, group, centre, envelope_m,
        clusterPacking(node, frame, envelope_m), color, emission, tags, frame.orientation);
      return;
    }
    case "field-program": {
      // The one node kind whose lengths this function does not scale, because
      // they are not its lengths: which of an op's five scalars is a length is
      // the op's own business, and a scaling rule here would be a second copy of
      // the op table that drifts from the first. See `SceneryFieldProgramNode`.
      // The check is on the *resolved* metres-per-unit, so a metres node under a
      // scaled group is caught as well as one authored in scene-scale.
      const unitScale = frame.unit_m * frame.scale;
      if (Math.abs(unitScale - 1) > 1e-9) {
        throw new RangeError(`Scenery field-program node "${node.id}" resolves to ${unitScale} metres per authored unit;`
          + ` a tape is authored in metres, so it needs \`units: "metres"\` and no enclosing \`place.scale\``);
      }
      // The extent bound is derived here and nowhere else, so the proxy box and
      // the tape can never be two opinions. The validator runs first: a tape the
      // machine cannot run has no extent, and a bound computed from one would be
      // a number rather than a box.
      try {
        validateSvoFieldProgram(node.program);
      } catch (error) {
        throw new RangeError(`Scenery field-program node "${node.id}": ${(error as Error).message}`);
      }
      const [x, y, z] = svoFieldProgramExtent_m(node.program);
      builder.fieldProgram(node.id, group, centre, V(x, y, z), node.program, color, emission, tags, frame.orientation);
      return;
    }
    default:
      node satisfies never;
  }
}

/**
 * The finite six-face room, from the declared faces outward.
 *
 * The imperative version took an override callback for the one wall that
 * carries an opening. Here that wall is simply the boxes around the hole, which
 * is what a union-only catalog can express and what the callback was standing
 * in for.
 */
function emitRoomShell(
  builder: ProxyBuilder,
  node: SceneryRoomShellNode,
  context: EnvironmentSceneryContext,
  graph: SceneryGraph,
  panes: SceneryPane[],
): EnvironmentProxyShell {
  const { floorY_m: floorY, shellThickness_m: thickness, scene, s } = context;
  const roomHalf = node.halfSize ? V(
    Math.max(node.halfSize.x * s, scene.container.width_m * .5 + thickness),
    Math.max(node.halfSize.y * s, scene.container.height_m * .5 + thickness),
    Math.max(node.halfSize.z * s, scene.container.depth_m * .5 + thickness),
  ) : context.roomHalf_m;
  const centre = V(0, floorY + roomHalf.y, 0);
  const t = thickness * .5;
  const face = (
    key: string, group: string, position: Vec3, half: Vec3, material: SceneryMaterial, tag: string,
  ): void => {
    const { color, emission } = resolveMaterial(material, graph);
    builder.box(key, group, position, half, color, emission, ["shell", tag], true);
  };
  face("shell/floor", "shell-floor", V(0, floorY - t, 0), V(roomHalf.x, t, roomHalf.z), node.floor, "floor");
  face("shell/ceiling", "shell-ceiling", V(0, floorY + 2 * roomHalf.y + t, 0), V(roomHalf.x, t, roomHalf.z), node.ceiling, "ceiling");
  face("shell/wall-left", "shell-wall", V(-roomHalf.x - t, centre.y, 0), V(t, roomHalf.y, roomHalf.z), node.wall, "wall");
  face("shell/wall-right", "shell-wall", V(roomHalf.x + t, centre.y, 0), V(t, roomHalf.y, roomHalf.z), node.wall, "wall");
  if (node.backWall) {
    // The four boxes around an exact rectangular hole. A single union-only wall
    // box would conceal the authored thin-glass pane behind it and force the
    // whole set off the analytic path.
    const opening = node.backWall;
    const openingHalfWidth = Math.min(opening.halfWidth * s, roomHalf.x - thickness);
    const openingHalfHeight = Math.min(opening.halfHeight * s, roomHalf.y - thickness);
    const openingCentreY = floorY + opening.centerY * s;
    const openingBottom = Math.max(floorY, openingCentreY - openingHalfHeight);
    const openingTop = Math.min(floorY + 2 * roomHalf.y, openingCentreY + openingHalfHeight);
    const sideHalfWidth = .5 * (roomHalf.x - openingHalfWidth);
    const sideCentreX = openingHalfWidth + sideHalfWidth;
    const bottomHalfHeight = .5 * (openingBottom - floorY);
    const topHalfHeight = .5 * (floorY + 2 * roomHalf.y - openingTop);
    const cut = (key: string, position: Vec3, half: Vec3): void => {
      const { color, emission } = resolveMaterial(node.wall, graph);
      builder.box(key, "shell-wall", position, half, color, emission,
        ["shell", "wall", "window-cutout"], true);
    };
    const frame = opening.frame ?? "shell/wall-back";
    cut(`${frame}-left`, V(-sideCentreX, centre.y, -roomHalf.z - t), V(sideHalfWidth, roomHalf.y, t));
    cut(`${frame}-right`, V(sideCentreX, centre.y, -roomHalf.z - t), V(sideHalfWidth, roomHalf.y, t));
    cut(`${frame}-bottom`, V(0, floorY + bottomHalfHeight, -roomHalf.z - t), V(openingHalfWidth, bottomHalfHeight, t));
    cut(`${frame}-top`, V(0, openingTop + topHalfHeight, -roomHalf.z - t), V(openingHalfWidth, topHalfHeight, t));
    // The pane sits in the wall's inner face, and the backing immediately
    // outside it. Both are derived from the same opening as the frame, so a
    // resized room can never leave glass hanging beside its hole.
    if (opening.glazing) {
      panes.push({
        id: opening.glazing,
        center_m: V(0, openingCentreY, -roomHalf.z),
        half_m: [openingHalfWidth, openingHalfHeight],
      });
    }
    if (opening.backing) {
      const { color, emission } = resolveMaterial(opening.backing.material, graph);
      builder.box(opening.backing.id, opening.backing.group ?? "window-backing",
        V(0, openingCentreY, -roomHalf.z - 2 * t - .01 * s),
        V(openingHalfWidth, openingHalfHeight, .01 * s),
        color, emission, opening.backing.tags ?? ["window", "backing"], true);
    }
  } else {
    face("shell/wall-back", "shell-wall", V(0, centre.y, -roomHalf.z - t), V(roomHalf.x, roomHalf.y, t), node.wall, "wall");
  }
  face("shell/wall-front", "shell-wall", V(0, centre.y, roomHalf.z + t), V(roomHalf.x, roomHalf.y, t), node.wall, "wall");
  return {
    kind: "room", floorY_m: floorY, bounds_m: aabb(centre, roomHalf),
    primitives: builder.shell, materialModel: node.materialModel,
  };
}

function emitShell(
  builder: ProxyBuilder,
  node: SceneryShellNode,
  context: EnvironmentSceneryContext,
  graph: SceneryGraph,
  panes: SceneryPane[],
): EnvironmentProxyShell {
  if (node.kind === "room-shell") return emitRoomShell(builder, node, context, graph, panes);
  // The garden's heightfield is the authority — the same surface the solver
  // collides against — so this shell publishes no boxes of its own.
  const { roomHalf_m: roomHalf, floorY_m: floorY, scene } = context;
  const terrainTop = Math.max(scene.container.height_m, scene.terrain?.baseHeight_m ?? 0);
  return {
    kind: "terrain-heightfield", floorY_m: floorY,
    bounds_m: { min: V(-roomHalf.x, 0, -roomHalf.z), max: V(roomHalf.x, terrainTop, roomHalf.z) },
    primitives: builder.shell, materialModel: node.materialModel,
  };
}

function emitTree(
  builder: ProxyBuilder,
  node: Extract<SceneryNode, { kind: "tree" }>,
  frame: SceneryFrame,
  context: EnvironmentSceneryContext,
  graph: SceneryGraph,
): void {
  const ramp = (name: string) => (value: number): EnvironmentLinearColor =>
    resolveMaterial({ palette: name, value }, graph).color;
  const plan = planProceduralTree({
    key: node.id,
    root_m: frame.origin_m,
    height_m: metres(frame, node.height),
    rootRadius_m: metres(frame, node.rootRadius),
    spread_m: metres(frame, node.spread),
    seed: node.seed,
    leanXZ: node.lean,
    bark: ramp(node.bark),
    leaf: ramp(node.leaf),
    // The same number the generator catalog hands every species, and for the
    // same reason — see `SceneryGeneratorRequest.detailCellSize_m`. `tree` is
    // the one node kind that grows its own geometry without going through that
    // catalog, so it was the one that never received a leaf at all.
    leafSize_m: context.detailCellSize_m,
  });
  // The *render* lattice, not the solver's. The excursion is the margin inside
  // which a re-posed analytic surface still belongs to the voxel that owns it,
  // and the voxel that owns it is a leaf of the tree the prop was rasterised
  // into — which on a scene that spent `environmentRefinementDepth` levels is
  // `finestCellSize_m / 2^depth`. Taking the solver's number there granted a
  // gust up to eight times the margin it actually had, which is why
  // `DEFAULT_SVO_RENDER_TUNING` keeps refinement off by default and names this
  // as the reason.
  emitProceduralTree(builder, plan, node.sway
    ? { excursion_m: sceneryMaximumSwayExcursion_m(context.detailCellSize_m) }
    : undefined);
}

/**
 * Grow the nodes a generator node stands for, in publication order.
 *
 * The three things a document cannot hold are supplied here and nowhere else.
 *
 * The **ground query** is the whole reason a generator is a node kind rather
 * than a serialized function: a bonsai seats its root fingers on the heightfield
 * and a coping is set into it, and neither can carry `(x, z) => number` in a
 * JSON document. It resolves exactly as `childFrame`'s `anchor: "terrain"` does
 * — the scene's own baked grid, falling back to the environment's floor datum
 * when there is no terrain — so a generated set and an anchored node can never
 * disagree about where the ground is.
 *
 * The **vessel** is resolved from the graph's own table, so a run is named
 * rather than enumerated. `validateSceneryGraph` refuses a node naming one the
 * graph does not declare; the throw below is for a generator that needs a vessel
 * and was given none, which no valid document reaches.
 *
 * The **leaf** — `detailCellSize_m` — is the finest voxel the set will be drawn
 * at, and it is here for the same reason: a species carries legibility floors
 * measured in voxels across a feature, and a document that pinned a leaf would
 * be a bake of the lattice it was saved at. It reaches the context from
 * `voxelDomain.detailCellSize_m`, and is `finestCellSize_m` when a scene has not
 * asked for extra octree levels under the solver's.
 *
 * The generated nodes are visited in the generator node's own frame, so a
 * generator with no `place` publishes exactly what the same nodes would have
 * published spliced into the document by hand — which is what makes converting
 * a baked set to a generator node a no-op for owner indices.
 */
function growGenerator(
  node: SceneryGeneratorNode,
  context: EnvironmentSceneryContext,
  graph: SceneryGraph,
): SceneryNode[] {
  const request: SceneryGeneratorRequest = {
    key: node.id,
    seed: node.seed,
    groundHeightAt: (x_m, z_m) => context.scene.terrain
      ? terrainHeightAt(context.scene.terrain, x_m, z_m)
      : context.floorY_m,
    detailCellSize_m: context.detailCellSize_m,
    vessel: () => {
      const spec = node.vessel === undefined ? undefined : graph.vessels?.[node.vessel];
      if (!spec) {
        throw new Error(`Scenery generator ${node.id} (${node.generator}) needs a vessel the graph declares`);
      }
      return spec;
    },
  };
  return growSceneryGenerator(node.generator, node.params, request);
}

/**
 * Expand a scene's scenery graph onto the builder.
 *
 * Shell faces publish before props, exactly as the imperative modules did, so
 * owner indices keep the order every consumer already depends on.
 */
export function expandSceneryGraph(
  builder: ProxyBuilder,
  graph: SceneryGraph,
  context: EnvironmentSceneryContext,
): SceneryExpansion {
  const shellNode = graph.nodes.find(isSceneryShellNode);
  if (!shellNode) throw new Error("A scenery graph needs exactly one shell node");
  const panes: SceneryPane[] = [];
  const shell = emitShell(builder, shellNode, context, graph, panes);
  const root: SceneryFrame = {
    origin_m: V(0, 0, 0), unit_m: context.s, scale: 1, units: "scene-scale", group: "scenery",
  };
  const visit = (nodes: readonly SceneryNode[], parent: SceneryFrame): void => {
    for (const node of nodes) {
      if (isSceneryShellNode(node)) continue;
      const frame = childFrame(parent, node, context);
      if (node.kind === "group") { visit(node.children, frame); continue; }
      if (node.kind === "generator") { visit(growGenerator(node, context, graph), frame); continue; }
      if (node.kind === "tree") { emitTree(builder, node, frame, context, graph); continue; }
      if (node.kind === "glazing") {
        panes.push({
          id: node.id, center_m: frame.origin_m, orientation: frame.orientation,
          half_m: [metres(frame, node.half[0]), metres(frame, node.half[1])],
        });
        continue;
      }
      emitPrimitive(builder, node, frame, graph);
    }
  };
  const spans: ScenerySpan[] = [];
  for (const node of graph.nodes) {
    if (isSceneryShellNode(node)) continue;
    const from = builder.props.length;
    visit([node], root);
    spans.push({ nodeId: node.id, from, to: builder.props.length });
  }
  return { shell, spans, panes };
}
