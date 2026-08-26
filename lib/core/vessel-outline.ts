import type { SceneDescription } from "./model";
import { compileE0PlanarFluidBoundaries } from "./planar-fluid-boundary";
import { sceneLatticeDimensions, solidVoxelShellForScene } from "./scene-lattice";
import { solidWorldVoxelPatchBounds_m } from "./solid-world";
import {
  containerDecorationSpace,
  DecorationBuilder,
  linearFromHex,
  type DecorationVec3,
} from "./visualization-decorations";
import type { DecorationGeometry } from "./webgpu-decoration-overlay";

export type VesselPresentation = "outline" | "glass" | "none";

/** Missing is the cheap product default; garden terrain never gains a tank. */
export function sceneVesselPresentation(scene: SceneDescription): VesselPresentation {
  return scene.environment === "garden" ? "none"
    : scene.container.vessel ?? "outline";
}

export interface VesselOutlineGeometry {
  readonly key: string;
  readonly geometry: DecorationGeometry;
}

const BOX_EDGES: readonly (readonly [number, number])[] = Object.freeze([
  [0, 1], [1, 3], [3, 2], [2, 0],
  [4, 5], [5, 7], [7, 6], [6, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
]);

const boxCorner = (
  minimum: DecorationVec3,
  maximum: DecorationVec3,
  index: number,
): DecorationVec3 => [
  (index & 1) === 0 ? minimum[0] : maximum[0],
  (index & 2) === 0 ? minimum[1] : maximum[1],
  (index & 4) === 0 ? minimum[2] : maximum[2],
];

const pointKey = (point: DecorationVec3) => point.map((value) =>
  Object.is(value, -0) ? "0" : value.toPrecision(12)).join(",");

/** Add the union's crease graph without drawing shared slab edges twice. */
function appendBoxEdges(
  builder: DecorationBuilder,
  minimum: DecorationVec3,
  maximum: DecorationVec3,
  seen: Set<string>,
  style: Parameters<DecorationBuilder["worldSegment"]>[2],
): void {
  for (const [from, to] of BOX_EDGES) {
    const start = boxCorner(minimum, maximum, from);
    const end = boxCorner(minimum, maximum, to);
    const a = pointKey(start), b = pointKey(end);
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    if (seen.has(key)) continue;
    seen.add(key);
    builder.worldSegment(start, end, style);
  }
}

/**
 * Build the persistent cue from the same admitted voxel slabs CM12 samples.
 *
 * The box is deliberately not the ideal container envelope. Each accepted
 * floor/wall/lid patch contributes the exact metre-space bounds of its finite
 * one-voxel thickness. If a shell patch was cut or replaced, the E0 compiler
 * rejects it and it contributes no line work; that region remains owned by the
 * ordinary residual voxel renderer instead.
 */
export function buildVesselOutlineGeometry(
  scene: SceneDescription,
): VesselOutlineGeometry | undefined {
  if (sceneVesselPresentation(scene) !== "outline") return undefined;
  const extent_m = [scene.container.width_m, scene.container.height_m,
    scene.container.depth_m] as const;
  const dimensions = sceneLatticeDimensions(scene);
  const builder = new DecorationBuilder(containerDecorationSpace(dimensions,
    extent_m));
  const style = {
    colorLinear: linearFromHex("#78c8c0"),
    width_px: 1.35,
    intensity: 0.58,
  } as const;
  const shape = scene.container.shape ?? "box";
  if (shape === "box") {
    const seen = new Set<string>();
    const admitted = compileE0PlanarFluidBoundaries(scene).admitted;
    for (const boundary of admitted) {
      const patch = scene.solidVoxels[boundary.sourcePatchIndex];
      if (!patch) continue;
      const bounds = solidWorldVoxelPatchBounds_m(scene, patch);
      appendBoxEdges(builder, bounds.minimum, bounds.maximum, seen, style);
    }
    if (builder.segmentCount === 0) return undefined;
    const stamp = admitted.map(({ face, sourcePatchIndex }) => {
      const patch = scene.solidVoxels[sourcePatchIndex]!;
      return `${face}:${patch.minimum.join(",")}:${patch.maximumExclusive.join(",")}`;
    }).join("|");
    return {
      key: `vessel-voxel-volume:${stamp}`,
      geometry: builder.finish(),
    };
  } else {
    const expected = solidVoxelShellForScene(scene);
    const sameCoordinate = (left: readonly number[], right: readonly number[]) =>
      left.every((value, axis) => value === right[axis]);
    const completeCanonicalShell = expected.every((patch, index) => {
      const authored = scene.solidVoxels[index];
      return authored?.operation === patch.operation
        && sameCoordinate(authored.minimum, patch.minimum)
        && sameCoordinate(authored.maximumExclusive, patch.maximumExclusive);
    }) && !scene.solidVoxels.some(({ operation }) => operation === "clear");
    if (!completeCanonicalShell) return undefined;
    const center: DecorationVec3 = [0.5 * dimensions[0],
      0.5 * dimensions[1], 0.5 * dimensions[2]];
    const radius = 0.5 * Math.min(...dimensions);
    const seen = new Set<string>();
    // Three central lattice sections expose the exact staircase where empty
    // vessel cells meet the canonical filled exterior. This is intentionally
    // not a smooth authored sphere: changing the solve lattice changes the cue.
    for (const [axes, fixedAxis] of [
      [[0, 1], 2], [[0, 2], 1], [[1, 2], 0],
    ] as const) {
      const fixedCell = Math.max(0, Math.min(dimensions[fixedAxis]! - 1,
        Math.floor(center[fixedAxis]!)));
      const fixedOffset = fixedCell + 0.5 - center[fixedAxis]!;
      const inside = (u: number, v: number) => u >= 0 && v >= 0
        && u < dimensions[axes[0]]! && v < dimensions[axes[1]]!
        && Math.hypot(u + 0.5 - center[axes[0]]!,
          v + 0.5 - center[axes[1]]!, fixedOffset) < radius;
      const emit = (startU: number, startV: number,
        endU: number, endV: number) => {
        const start = [...center] as [number, number, number];
        const end = [...center] as [number, number, number];
        start[axes[0]] = startU; start[axes[1]] = startV;
        end[axes[0]] = endU; end[axes[1]] = endV;
        const a = pointKey(start), b = pointKey(end);
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        if (seen.has(key)) return;
        seen.add(key);
        builder.segment(start, end, style);
      };
      for (let v = 0; v < dimensions[axes[1]]!; v += 1) {
        for (let u = 0; u < dimensions[axes[0]]!; u += 1) {
          if (!inside(u, v)) continue;
          if (!inside(u - 1, v)) emit(u, v, u, v + 1);
          if (!inside(u + 1, v)) emit(u + 1, v, u + 1, v + 1);
          if (!inside(u, v - 1)) emit(u, v, u + 1, v);
          if (!inside(u, v + 1)) emit(u, v + 1, u + 1, v + 1);
        }
      }
    }
  }
  return {
    key: `vessel-voxel-volume:${shape}:${dimensions.join("x")}:${extent_m.join(",")}`,
    geometry: builder.finish(),
  };
}
