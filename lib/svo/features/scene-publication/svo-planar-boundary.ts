import {
  PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO,
  type PlanarBoundaryPatch,
  type PlanarBoundaryVec3,
} from "../../../core/planar-boundary";
import type { EnvironmentBoxProxy, EnvironmentProxyPrimitive } from "../../../core/voxel-environments";
import type { SceneDescription } from "../../../core/model";
import { sceneLatticeDimensions, solidVoxelShellForScene } from "../../../core/scene-lattice";
import { appendedSolidPatches } from "../../../core/solid-patch-list";
import { terrainContentStamp } from "../../../core/terrain";
import { vesselCutawayPatches } from "../../../core/vessel-cutaway";
import {
  createSolidWorld,
  planarBoundaryForSolidWorldVoxelPatch,
  solidWorldVoxelPatchBounds_m,
  terrainSolidWorldWithPatches,
  withSolidWorldPatches,
  type SolidWorld,
  type SolidWorldVoxelPatch,
} from "../../../core/solid-world";
import {
  SPARSE_BRICK_LEAF_TERMINAL,
  SPARSE_BRICK_VOXEL_TERMINAL,
  type SparseBrickCoordinate,
  type SparseBrickLeafTerminal,
} from "../construction/sparse-brick-octree";

/** A thin box is promoted only when its next-smallest dimension is decisively larger. */
export const SVO_PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO =
  PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO;

export interface SvoPlanarBoundarySource {
  readonly patch: PlanarBoundaryPatch;
  /** Dense index in the immutable accepted planar-record catalogue. */
  readonly sourceIndex: number;
  readonly bounds_m: {
    readonly minimum: PlanarBoundaryVec3;
    readonly maximum: PlanarBoundaryVec3;
  };
}

export interface SvoPlanarBoundaryCatalog {
  readonly sources: readonly SvoPlanarBoundarySource[];
  readonly patchIndexByOwner: ReadonlyMap<number, number>;
}

export interface SvoSolidWorldPlanarBoundaryCatalog {
  readonly sources: readonly SvoPlanarBoundarySource[];
  readonly patchIndexByPatch: ReadonlyMap<number, number>;
  /**
   * SolidWorld sources omitted from the opaque render voxel field. This is the
   * union of opaque analytic records and canonical tank-shell faces whose
   * presentation is the cheap volume wireframe or hidden entirely.
   */
  readonly residualExcludedPatchIndices: ReadonlySet<number>;
}

/**
 * The render voxel field is the residual of the accepted analytic catalogue.
 * A promoted source is never submitted to both representations.
 */
export function svoPlanarResidualEnvironmentPrimitives(
  primitives: readonly EnvironmentProxyPrimitive[],
  catalog: SvoPlanarBoundaryCatalog,
): readonly EnvironmentProxyPrimitive[] {
  return primitives.filter((primitive) =>
    !catalog.patchIndexByOwner.has(primitive.ownerIndex));
}

/**
 * Rebuild the render-only SolidWorld after removing every fill owned by the
 * exact planar catalogue, and the near walls a scene's cutaway opens. Fluid
 * keeps its canonical SolidWorld separately.
 *
 * `scene` is the document. Its terrain is baked into pages, not carried as a
 * patch, so the surviving fills have to be laid back over it rather than over
 * nothing; its cutaway is cut from what survives, since that is what is drawn.
 */
export function svoPlanarResidualSolidWorld(
  world: SolidWorld,
  catalog: SvoSolidWorldPlanarBoundaryCatalog | undefined,
  scene?: SceneDescription,
): SolidWorld {
  const excluded = catalog?.residualExcludedPatchIndices;
  const drawn = excluded?.size
    ? world.patches.filter((_, patchIndex) => !excluded.has(patchIndex))
    : world.patches;
  const cutaway = scene ? vesselCutawayPatches(scene, drawn) : [];
  const drawTerrain = scene?.scenery?.terrainSurface !== "scenery";
  if (drawn === world.patches && cutaway.length === 0 && drawTerrain) return world;
  const patches = cutaway.length === 0 ? drawn : [...drawn, ...cutaway];
  return scene?.terrain && drawTerrain
    ? terrainSolidWorldWithPatches(scene, patches)
    : createSolidWorld(patches);
}

/**
 * One published solid snapshot's render residual, with what the next snapshot
 * needs to extend it instead of replaying the document.
 */
export interface SvoRenderResidual {
  /** The canonical snapshot this was taken from. */
  readonly world: SolidWorld;
  /** What is drawn: `world` without its hidden shell and with the cutaway cut. */
  readonly residual: SolidWorld;
  /** `residual` before the cutaway; the image a later stroke is laid over. */
  readonly drawn: SolidWorld;
  readonly shell: CanonicalShellSources;
  readonly excluded: ReadonlySet<number>;
  readonly groundKey: string;
  readonly cutawayKey: string;
}

/** Everything but the patches that the drawn image is composed from. */
const residualGroundKey = (scene: SceneDescription): string =>
  `${scene.scenery?.terrainSurface ?? "terrain"}:${terrainContentStamp(scene.terrain)}:${scene.container.width_m}:${scene.container.height_m}:`
  + `${scene.container.depth_m}:${sceneLatticeDimensions(scene).join(":")}:`
  + `${scene.container.shape ?? "box"}:${scene.container.top ?? "open"}`;

/**
 * Residuals by the snapshot they were taken from, for exactly as long as
 * something else holds that snapshot. An edit asks for its residual twice, to
 * be validated and to be staged; and undo and redo return to snapshots history
 * still holds, so they take those residuals back rather than replay a document.
 */
const residualBySnapshot = new WeakMap<SolidWorld, SvoRenderResidual>();

/**
 * The render residual of `world`, the snapshot `scene` publishes.
 *
 * `previous` is the residual of the snapshot published before it. A stroke
 * appends patches, and the residual of an appended document is the previous
 * drawn image with the stroke laid over it: the edit costs its own pages, where
 * replaying the document costs every patch ever authored and, on a terrain
 * scene, a bake of the ground. Pages the stroke leaves alone keep their
 * identity, which is what lets the display upload only the ones that moved.
 *
 * The document is replayed when there is nothing to extend: a first
 * publication, an undo or a branch, different ground, or an edit that changes
 * which shell faces are hidden — the first cut into a wall reveals the whole
 * slab. A scene's cutaway is a flood of the drawn solids and is taken afresh
 * either way.
 */
export function svoRenderResidual(
  scene: SceneDescription,
  world: SolidWorld,
  previous?: SvoRenderResidual,
): SvoRenderResidual {
  const groundKey = residualGroundKey(scene);
  const eye = scene.cutaway?.eye_m;
  const cutawayKey = eye ? `${eye.x}:${eye.y}:${eye.z}` : "";
  const known = residualBySnapshot.get(world);
  if (known && known.groundKey === groundKey && known.cutawayKey === cutawayKey) return known;
  const appended = previous && previous.groundKey === groundKey
    ? appendedSolidPatches(previous.world.patches, world.patches) : undefined;
  const shell = canonicalShellSources(scene, world.patches, appended ? previous!.shell : undefined);
  const excluded = canonicalShellPatches(shell).hidden;
  const firstAppended = world.patches.length - (appended?.length ?? 0);
  let drawn: SolidWorld;
  const drawTerrain = scene.scenery?.terrainSurface !== "scenery";
  if (excluded.size === 0 && drawTerrain) {
    drawn = world;
  } else if (appended && sameHiddenBefore(previous!.excluded, excluded, firstAppended)) {
    drawn = withSolidWorldPatches(previous!.drawn,
      appended.filter((_, index) => !excluded.has(firstAppended + index)));
  } else {
    const patches = world.patches.filter((_, patchIndex) => !excluded.has(patchIndex));
    drawn = scene.terrain && drawTerrain ? terrainSolidWorldWithPatches(scene, patches) : createSolidWorld(patches);
  }
  const cutaway = vesselCutawayPatches(scene, drawn.patches);
  const residual: SvoRenderResidual = { world, residual: withSolidWorldPatches(drawn, cutaway),
    drawn, shell, excluded, groundKey, cutawayKey };
  residualBySnapshot.set(world, residual);
  return residual;
}

/** Whether both snapshots hide the same patches among the first `count`. */
function sameHiddenBefore(previous: ReadonlySet<number>, next: ReadonlySet<number>,
  count: number): boolean {
  let nextBefore = 0;
  for (const patchIndex of next) if (patchIndex < count) nextBefore += 1;
  if (nextBefore !== previous.size) return false;
  for (const patchIndex of previous) if (!next.has(patchIndex)) return false;
  return true;
}

/** Bounds of visible solid geometry, retaining analytic owners as well as voxels. */
export function svoPlanarSolidWorldBlockers(
  scene: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[],
  catalog: SvoSolidWorldPlanarBoundaryCatalog | undefined,
): SvoPlanarLeafClassifierOptions["blockers"] {
  return patches.flatMap((patch, patchIndex) => {
    const planarSourceIndex = catalog?.patchIndexByPatch.get(patchIndex);
    // Physics-only geometry is absent from both render representations. It
    // cannot disqualify an otherwise isolated analytic surface.
    if (planarSourceIndex === undefined
      && catalog?.residualExcludedPatchIndices.has(patchIndex)) return [];
    return [{ ...solidWorldVoxelPatchBounds_m(scene, patch), planarSourceIndex }];
  });
}

const quaternionRotate = (
  vector: PlanarBoundaryVec3,
  orientation: EnvironmentBoxProxy["orientation"],
): PlanarBoundaryVec3 => {
  if (!orientation) return vector;
  const { w, x, y, z } = orientation;
  const dotUV = x * vector[0] + y * vector[1] + z * vector[2];
  const dotUU = x * x + y * y + z * z;
  const cross: PlanarBoundaryVec3 = [
    y * vector[2] - z * vector[1],
    z * vector[0] - x * vector[2],
    x * vector[1] - y * vector[0],
  ];
  return [
    2 * dotUV * x + (w * w - dotUU) * vector[0] + 2 * w * cross[0],
    2 * dotUV * y + (w * w - dotUU) * vector[1] + 2 * w * cross[1],
    2 * dotUV * z + (w * w - dotUU) * vector[2] + 2 * w * cross[2],
  ];
};

function normalized(value: PlanarBoundaryVec3): PlanarBoundaryVec3 {
  const magnitude = Math.hypot(...value);
  if (!(magnitude > 0) || !Number.isFinite(magnitude)) {
    throw new RangeError("Planar boundary orientation produced a zero axis");
  }
  return [value[0] / magnitude, value[1] / magnitude, value[2] / magnitude];
}

export function isSvoPlanarBoundaryProxy(
  primitive: EnvironmentProxyPrimitive,
): primitive is EnvironmentBoxProxy {
  if (primitive.kind !== "box" || primitive.sway) return false;
  // Interior presentation deliberately opens this authored shell face. A macro
  // terminal must not bypass the same visibility policy through a direct hit.
  if (primitive.tags.includes("shell") && primitive.tags.includes("wall")
    && primitive.key.endsWith("/shell/wall-front")) return false;
  const extents = [primitive.halfSize_m.x, primitive.halfSize_m.y, primitive.halfSize_m.z]
    .sort((left, right) => left - right);
  return extents[0] > 0
    && extents[1] >= extents[0] * SVO_PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO;
}

/** Convert an authored thin oriented box without changing any of its geometry. */
export function svoPlanarBoundaryForProxy(
  primitive: EnvironmentProxyPrimitive,
  identity: { readonly materialId: number; readonly ownerId: number },
): PlanarBoundaryPatch | null {
  if (!isSvoPlanarBoundaryProxy(primitive)) return null;
  const half = [primitive.halfSize_m.x, primitive.halfSize_m.y, primitive.halfSize_m.z] as const;
  let normalAxis = 0;
  if (half[1] < half[normalAxis]) normalAxis = 1;
  if (half[2] < half[normalAxis]) normalAxis = 2;
  const tangentAxes = [0, 1, 2].filter((axis) => axis !== normalAxis);
  const basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as const;
  return {
    center_m: [primitive.center_m.x, primitive.center_m.y, primitive.center_m.z],
    normal: normalized(quaternionRotate(basis[normalAxis], primitive.orientation)),
    tangentU: normalized(quaternionRotate(basis[tangentAxes[0]], primitive.orientation)),
    tangentV: normalized(quaternionRotate(basis[tangentAxes[1]], primitive.orientation)),
    halfExtentU_m: half[tangentAxes[0]],
    halfExtentV_m: half[tangentAxes[1]],
    halfThickness_m: half[normalAxis],
    materialId: identity.materialId,
    ownerId: identity.ownerId,
  };
}

/** Build the compact accepted-record order independently of primitive owner order. */
export function buildSvoPlanarBoundaryCatalog(
  primitives: readonly EnvironmentProxyPrimitive[],
  identityFor: (primitive: EnvironmentProxyPrimitive) => {
    readonly materialId: number;
    readonly ownerId: number;
  },
): SvoPlanarBoundaryCatalog {
  const sources: SvoPlanarBoundarySource[] = [];
  const patchIndexByOwner = new Map<number, number>();
  for (const primitive of primitives) {
    const patch = svoPlanarBoundaryForProxy(primitive, identityFor(primitive));
    if (!patch) continue;
    const sourceIndex = sources.length;
    patchIndexByOwner.set(primitive.ownerIndex, sourceIndex);
    sources.push({
      patch,
      sourceIndex,
      bounds_m: {
        minimum: [primitive.aabb_m.min.x, primitive.aabb_m.min.y, primitive.aabb_m.min.z],
        maximum: [primitive.aabb_m.max.x, primitive.aabb_m.max.y, primitive.aabb_m.max.z],
      },
    });
  }
  return { sources, patchIndexByOwner };
}

/** A shell slab is a whole-cell box; a patch of any other kind is never one. */
const shellPatchSignature = (patch: SolidWorldVoxelPatch): string =>
  `${patch.operation}:${patch.kind ?? "box"}:${patch.materialId ?? 1}:`
  + `${patch.minimum.join(",")}:${patch.maximumExclusive.join(",")}`;

const overlapsVolume = (left: SolidWorldVoxelPatch, right: SolidWorldVoxelPatch): boolean =>
  left.maximumExclusive[0] > right.minimum[0]
    && left.minimum[0] < right.maximumExclusive[0]
    && left.maximumExclusive[1] > right.minimum[1]
    && left.minimum[1] < right.maximumExclusive[1]
    && left.maximumExclusive[2] > right.minimum[2]
    && left.minimum[2] < right.maximumExclusive[2];

/**
 * Where a document keeps its canonical tank shell, and which of those slabs an
 * edit has cut.
 *
 * Both answers only ever move one way as patches are appended: a slab is the
 * first document patch equal to it, and a slab once cut stays cut. So the
 * record for a longer document is the shorter one's continued over the new
 * patches, and a stroke pays for itself rather than for the document.
 */
export interface CanonicalShellSources {
  /** The shell `solidVoxelShellForScene` authors for this lattice. */
  readonly expected: readonly SolidWorldVoxelPatch[];
  /** Per expected slab, the first document patch that is it, or -1. */
  readonly source: Int32Array;
  /** Per expected slab, whether a clear overlaps it. */
  readonly cut: Uint8Array;
  /** Expected slabs the document has not supplied yet, by signature. */
  readonly unmatched: ReadonlyMap<string, readonly number[]>;
  readonly anyClear: boolean;
  /** A sphere's shell is hidden whole or not at all. */
  readonly atomic: boolean;
  /** How many document patches this record has read. */
  readonly patchCount: number;
}

/** `previous` must describe a prefix of `patches` on the same lattice. */
function canonicalShellSources(
  scene: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[],
  previous?: CanonicalShellSources,
): CanonicalShellSources {
  const expected = previous?.expected ?? solidVoxelShellForScene(scene);
  const source = previous ? previous.source.slice() : new Int32Array(expected.length).fill(-1);
  const cut = previous ? previous.cut.slice() : new Uint8Array(expected.length);
  let unmatched: ReadonlyMap<string, readonly number[]>;
  if (previous) unmatched = previous.unmatched;
  else {
    const bySignature = new Map<string, number[]>();
    expected.forEach((slab, slabIndex) => {
      const signature = shellPatchSignature(slab);
      const slabs = bySignature.get(signature);
      if (slabs) slabs.push(slabIndex); else bySignature.set(signature, [slabIndex]);
    });
    unmatched = bySignature;
  }
  let anyClear = previous?.anyClear ?? false;
  for (let patchIndex = previous?.patchCount ?? 0; patchIndex < patches.length; patchIndex += 1) {
    const patch = patches[patchIndex]!;
    const slabs = unmatched.size > 0 ? unmatched.get(shellPatchSignature(patch)) : undefined;
    if (slabs) {
      for (const slabIndex of slabs) source[slabIndex] = patchIndex;
      // The previous record keeps its own map; only a snapshot that finds a slab pays for a copy.
      const remaining = new Map(unmatched);
      remaining.delete(shellPatchSignature(patch));
      unmatched = remaining;
    }
    if (patch.operation !== "clear") continue;
    anyClear = true;
    for (let slabIndex = 0; slabIndex < expected.length; slabIndex += 1) {
      if (cut[slabIndex] === 0 && overlapsVolume(patch, expected[slabIndex]!)) cut[slabIndex] = 1;
    }
  }
  return { expected, source, cut, unmatched, anyClear,
    atomic: scene.container.shape === "sphere", patchCount: patches.length };
}

/**
 * The document patches that are canonical shell, and the ones presentation
 * hides. `candidates` is the list being published, when it is not the document
 * the record was read from: a slab counts only where both hold it.
 */
function canonicalShellPatches(
  shell: CanonicalShellSources,
  candidates?: readonly SolidWorldVoxelPatch[],
): { canonical: Set<number>; hidden: Set<number> } {
  const canonical = new Set<number>(), uncut = new Set<number>();
  shell.expected.forEach((slab, slabIndex) => {
    const patchIndex = shell.source[slabIndex]!;
    if (patchIndex < 0) return;
    const candidate = candidates?.[patchIndex];
    if (candidates && (!candidate
      || shellPatchSignature(candidate) !== shellPatchSignature(slab))) return;
    canonical.add(patchIndex);
    if (shell.cut[slabIndex] === 0) uncut.add(patchIndex);
  });
  // A canonical sphere is replaced as one atomic presentation source.
  // Any clear edit makes the substitution ineligible so the exact edited
  // voxel surface remains visible.
  const atomicallyHidden = shell.atomic && canonical.size === shell.expected.length
    && !shell.anyClear;
  return { canonical, hidden: atomicallyHidden ? canonical : uncut };
}

/** Compile accepted SolidWorld edit boxes into a caller's compact record range. */
export function buildSvoSolidWorldPlanarBoundaryCatalog(
  scene: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[],
  firstSourceIndex = 0,
  options: { promoteEditablePatches?: boolean } = {},
): SvoSolidWorldPlanarBoundaryCatalog {
  if (!Number.isSafeInteger(firstSourceIndex) || firstSourceIndex < 0) {
    throw new RangeError("Planar boundary source base must be a non-negative integer");
  }
  const sources: SvoPlanarBoundarySource[] = [];
  const patchIndexByPatch = new Map<number, number>();
  // Canonical shell fills remain voxels for physics. Presentation may omit an
  // unedited fill in favour of the vessel outline, but it must never promote a
  // shell slab into the explicit embedded-plane catalogue.
  const { canonical: canonicalTankShellSources, hidden: canonicalTankShell } =
    canonicalShellPatches(canonicalShellSources(scene, scene.solidVoxels), patches);
  // Tank glass is retired. Keep the canonical shell as physical SolidWorld
  // authority, but remove every unedited wall from the render residual. The
  // vessel outline supplies the visual cue.
  const residualExcludedPatchIndices = new Set<number>(canonicalTankShell);
  patches.forEach((voxelPatch, patchIndex) => {
    // The canonical vessel shell is physical SolidWorld authority for the
    // solver. Presentation omits it from the ray field, and edited shell pieces
    // remain visible as ordinary voxel geometry. Neither case may promote it to an
    // opaque embedded-plane hit, which caused the black tank floor.
    if (canonicalTankShellSources.has(patchIndex)) return;
    if (options.promoteEditablePatches === false) return;
    const patch = planarBoundaryForSolidWorldVoxelPatch(scene, voxelPatch);
    if (!patch) return;
    const sourceIndex = firstSourceIndex + sources.length;
    patchIndexByPatch.set(patchIndex, sourceIndex);
    residualExcludedPatchIndices.add(patchIndex);
    sources.push({
      patch,
      sourceIndex,
      bounds_m: solidWorldVoxelPatchBounds_m(scene, voxelPatch),
    });
  });
  return { sources, patchIndexByPatch, residualExcludedPatchIndices };
}

export interface SvoPlanarLeafClassifierOptions {
  readonly sources: readonly SvoPlanarBoundarySource[];
  /** Visible authored bounds, including non-planar sources and static solid pages. */
  readonly blockers: readonly {
    readonly minimum: PlanarBoundaryVec3;
    readonly maximum: PlanarBoundaryVec3;
    /** Source index when this bound is the same planar primitive. */
    readonly planarSourceIndex?: number;
  }[];
  readonly worldOrigin_m: PlanarBoundaryVec3;
  readonly nodeEdge_m: readonly (readonly number[])[];
}

export interface SvoPlanarLeafClassifier {
  (level: number, coordinate: SparseBrickCoordinate): SparseBrickLeafTerminal;
  /** True where planar geometry is present but cannot replace the voxel payload. */
  requiresFineVoxelResidual(level: number, coordinate: SparseBrickCoordinate): boolean;
}

const overlaps = (
  a: { readonly minimum: PlanarBoundaryVec3; readonly maximum: PlanarBoundaryVec3 },
  b: { readonly minimum: PlanarBoundaryVec3; readonly maximum: PlanarBoundaryVec3 },
): boolean => a.maximum[0] >= b.minimum[0] && a.minimum[0] <= b.maximum[0]
  && a.maximum[1] >= b.minimum[1] && a.minimum[1] <= b.maximum[1]
  && a.maximum[2] >= b.minimum[2] && a.minimum[2] <= b.maximum[2];

/**
 * Select a macro terminal only where one planar record is the complete authored
 * geometry set for the node. Intersections and corners remain voxel residuals.
 */
export function createSvoPlanarLeafClassifier(
  options: SvoPlanarLeafClassifierOptions,
): SvoPlanarLeafClassifier {
  const byIndex = new Map(options.sources.map((source) => [source.sourceIndex, source]));
  const classify = (level: number, coordinate: SparseBrickCoordinate): {
    readonly terminal: SparseBrickLeafTerminal;
    readonly planarOverlap: boolean;
  } => {
    const edge = options.nodeEdge_m[level];
    if (!edge) return { terminal: SPARSE_BRICK_VOXEL_TERMINAL, planarOverlap: false };
    const minimum: PlanarBoundaryVec3 = [
      options.worldOrigin_m[0] + coordinate.x * edge[0],
      options.worldOrigin_m[1] + coordinate.y * edge[1],
      options.worldOrigin_m[2] + coordinate.z * edge[2],
    ];
    const node = { minimum, maximum: [minimum[0] + edge[0], minimum[1] + edge[1],
      minimum[2] + edge[2]] as PlanarBoundaryVec3 };
    let selected: number | undefined;
    let planarOverlap = false;
    let blocked = false;
    for (const blocker of options.blockers) {
      if (!overlaps(node, blocker)) continue;
      const sourceIndex = blocker.planarSourceIndex;
      if (sourceIndex === undefined || !byIndex.has(sourceIndex)) {
        blocked = true;
        continue;
      }
      planarOverlap = true;
      if (selected !== undefined && selected !== sourceIndex) blocked = true;
      selected = sourceIndex;
    }
    return {
      terminal: selected === undefined || blocked ? SPARSE_BRICK_VOXEL_TERMINAL : {
        kind: SPARSE_BRICK_LEAF_TERMINAL.planarBoundary,
        index: selected,
      },
      planarOverlap,
    };
  };
  const result = ((level: number, coordinate: SparseBrickCoordinate) =>
    classify(level, coordinate).terminal) as SvoPlanarLeafClassifier;
  result.requiresFineVoxelResidual = (level, coordinate) => {
    const classification = classify(level, coordinate);
    return classification.planarOverlap
      && classification.terminal.kind === SPARSE_BRICK_LEAF_TERMINAL.voxels;
  };
  return result;
}
