import {
  PLANAR_BOUNDARY_MINIMUM_ASPECT_RATIO,
  type PlanarBoundaryPatch,
  type PlanarBoundaryVec3,
} from "../core/planar-boundary";
import type { EnvironmentBoxProxy, EnvironmentProxyPrimitive } from "../core/voxel-environments";
import type { SceneDescription } from "../core/model";
import { compileE0PlanarFluidBoundaries } from "../core/planar-fluid-boundary";
import { solidVoxelShellForScene } from "../core/scene-lattice";
import {
  createSolidWorld,
  planarBoundaryForSolidWorldVoxelPatch,
  solidWorldVoxelPatchBounds_m,
  type SolidWorld,
  type SolidWorldVoxelPatch,
} from "../core/solid-world";
import {
  SPARSE_BRICK_LEAF_TERMINAL,
  SPARSE_BRICK_VOXEL_TERMINAL,
  type SparseBrickCoordinate,
  type SparseBrickLeafTerminal,
} from "./sparse-brick-octree";

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
 * exact planar catalogue. Fluid keeps its canonical SolidWorld separately.
 */
export function svoPlanarResidualSolidWorld(
  world: SolidWorld,
  catalog: SvoSolidWorldPlanarBoundaryCatalog | undefined,
): SolidWorld {
  if (!catalog || catalog.residualExcludedPatchIndices.size === 0) return world;
  return createSolidWorld(world.patches.filter((_, patchIndex) =>
    !catalog.residualExcludedPatchIndices.has(patchIndex)));
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

/** Compile accepted SolidWorld edit boxes into a caller's compact record range. */
export function buildSvoSolidWorldPlanarBoundaryCatalog(
  scene: SceneDescription,
  patches: readonly SolidWorldVoxelPatch[],
  firstSourceIndex = 0,
): SvoSolidWorldPlanarBoundaryCatalog {
  if (!Number.isSafeInteger(firstSourceIndex) || firstSourceIndex < 0) {
    throw new RangeError("Planar boundary source base must be a non-negative integer");
  }
  const sources: SvoPlanarBoundarySource[] = [];
  const patchIndexByPatch = new Map<number, number>();
  const sameCoordinate = (left: SolidWorldVoxelPatch["minimum"],
    right: SolidWorldVoxelPatch["minimum"]) => left.every((value, axis) =>
    value === right[axis]);
  const samePatch = (authored: SolidWorldVoxelPatch | undefined,
    candidate: SolidWorldVoxelPatch | undefined) => authored !== undefined
      && candidate !== undefined
      && authored.operation === candidate.operation
      && sameCoordinate(authored.minimum, candidate.minimum)
      && sameCoordinate(authored.maximumExclusive, candidate.maximumExclusive);
  // Membership and admission are different facts. Even a cut canonical face
  // remains a vessel source and must never fall through into the generic
  // opaque thin-plate promoter; admission below only decides whether its
  // unedited voxel payload may be replaced by the cheap outline.
  const canonicalTankShellSources = new Set<number>();
  const expectedTankShell = solidVoxelShellForScene(scene);
  expectedTankShell.forEach((expected, expectedIndex) => {
    const authored = scene.solidVoxels[expectedIndex];
    const candidate = patches[expectedIndex];
    if (samePatch(authored, expected) && samePatch(candidate, expected)) {
      canonicalTankShellSources.add(expectedIndex);
    }
  });
  const e0CanonicalTankShell = new Set(compileE0PlanarFluidBoundaries(scene)
    .admitted.filter(({ sourcePatchIndex }) => {
      const authored = scene.solidVoxels[sourcePatchIndex];
      const candidate = patches[sourcePatchIndex];
      return samePatch(authored, candidate);
    }).map(({ sourcePatchIndex }) => sourcePatchIndex));
  // A canonical sphere has no planar E0 faces, but outline/hidden presentation
  // still replaces its expensive dielectric exterior as one atomic source.
  // Any clear edit makes the substitution ineligible so the exact edited
  // voxel surface remains visible.
  const canonicalTankShell = (scene.container.shape === "sphere"
    && canonicalTankShellSources.size === expectedTankShell.length
    && !scene.solidVoxels.some(({ operation }) => operation === "clear"))
    ? canonicalTankShellSources
    : e0CanonicalTankShell;
  // Explicit glass keeps the historical voxel/dielectric presentation. The
  // default outline and hidden modes have no filled wall surface, so their
  // canonical shell is removed from the opaque/voxel render residual entirely.
  const residualExcludedPatchIndices = new Set<number>(
    (scene.container.vessel ?? "outline") === "glass" ? [] : canonicalTankShell,
  );
  patches.forEach((voxelPatch, patchIndex) => {
    // The canonical vessel shell is physical SolidWorld authority for the
    // solver. Glass retains the dielectric voxel surface; outline/hidden omit
    // it from the ray field. None of those modes may also promote it to an
    // opaque planar hit, which caused the black tank floor.
    if (canonicalTankShellSources.has(patchIndex)) return;
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
  /** All authored bounds, including non-planar sources and static solid pages. */
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
