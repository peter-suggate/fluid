import {
  createBiasedSvoVisibilityRay,
  traceSvoVisibilityRay,
  type SvoVisibilityStepSource,
} from "./svo-visibility-rays";
import type { SvoVec3 } from "./webgpu-svo-traversal";

/**
 * Bounded short-range diffuse visibility. The production gate remains off
 * until live timing proves that secondary traversal fits the dry-scene budget.
 */
export const SVO_CONTACT_VISIBILITY_CONTRACT = Object.freeze({
  enabledByDefault: false,
  sampleCount: 2,
  maximumNodeVisitsPerSample: 64,
  maximumLeafVisitsPerSample: 16,
  maximumWorkItemsPerSample: 256,
  maximumIntersectionsPerSample: 2,
  smoothBiasCells: 0.025,
  hardFeatureBiasCells: 0.05,
  radiusCells: 6,
  minimumSceneRadiusFraction: 0.01,
  maximumSceneRadiusFraction: 0.06,
} as const);

export interface SvoContactVisibilityInput {
  surfacePosition_m: SvoVec3;
  geometricNormal: SvoVec3;
  featureId: number;
  cellSize_m: SvoVec3;
  sceneExtent_m: SvoVec3;
}

export type SvoContactVisibilityResult =
  | { status: "resolved"; visibility: SvoVec3; radius_m: number }
  | { status: "invalid" | "exhausted"; visibility: readonly [0, 0, 0]; radius_m: number };

function normalized(value: SvoVec3): SvoVec3 {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) throw new RangeError("Contact normal must be finite");
  const length = Math.hypot(...value);
  if (!(length > 1e-12)) throw new RangeError("Contact normal must be nonzero");
  return [value[0] / length, value[1] / length, value[2] / length];
}

function cross(a: SvoVec3, b: SvoVec3): SvoVec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function svoContactVisibilityRadius_m(cellSize_m: SvoVec3, sceneExtent_m: SvoVec3): number {
  if ([...cellSize_m, ...sceneExtent_m].some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("Contact visibility cell and scene extents must be positive and finite");
  }
  const cellScale = Math.max(...cellSize_m), sceneScale = Math.max(...sceneExtent_m);
  return Math.min(
    sceneScale * SVO_CONTACT_VISIBILITY_CONTRACT.maximumSceneRadiusFraction,
    Math.max(
      cellScale * SVO_CONTACT_VISIBILITY_CONTRACT.radiusCells,
      sceneScale * SVO_CONTACT_VISIBILITY_CONTRACT.minimumSceneRadiusFraction,
    ),
  );
}

/** Stable two-direction cosine-hemisphere pattern; no frame-varying noise. */
export function svoContactVisibilityDirections(normalIn: SvoVec3, featureId: number): readonly SvoVec3[] {
  if (!Number.isSafeInteger(featureId) || featureId < 0) throw new RangeError("Contact feature ID must be non-negative");
  const normal = normalized(normalIn);
  const helper: SvoVec3 = Math.abs(normal[1]) > 0.9 ? [1, 0, 0] : [0, 1, 0];
  let tangent = normalized(cross(helper, normal));
  let bitangent = cross(normal, tangent);
  if ((featureId & 1) !== 0) [tangent, bitangent] = [bitangent, [-tangent[0], -tangent[1], -tangent[2]]];
  const directions: SvoVec3[] = [
    [normal[0] + 0.55 * tangent[0] + 0.2 * bitangent[0], normal[1] + 0.55 * tangent[1] + 0.2 * bitangent[1], normal[2] + 0.55 * tangent[2] + 0.2 * bitangent[2]],
    [normal[0] - 0.55 * tangent[0] - 0.2 * bitangent[0], normal[1] - 0.55 * tangent[1] - 0.2 * bitangent[1], normal[2] - 0.55 * tangent[2] - 0.2 * bitangent[2]],
  ];
  return directions.map(normalized);
}

/** CPU contract/oracle mirroring the gated production shader. */
export function traceSvoContactVisibility(
  input: SvoContactVisibilityInput,
  source: SvoVisibilityStepSource,
): SvoContactVisibilityResult {
  const radius_m = svoContactVisibilityRadius_m(input.cellSize_m, input.sceneExtent_m);
  const biasCells = input.featureId === 0
    ? SVO_CONTACT_VISIBILITY_CONTRACT.smoothBiasCells
    : SVO_CONTACT_VISIBILITY_CONTRACT.hardFeatureBiasCells;
  const total: [number, number, number] = [0, 0, 0];
  for (const direction of svoContactVisibilityDirections(input.geometricNormal, input.featureId)) {
    const ray = createBiasedSvoVisibilityRay({
      surfacePosition_m: input.surfacePosition_m,
      geometricNormal: input.geometricNormal,
      directionToLight: direction,
      maximumLightDistance_m: radius_m,
      cellSize_m: input.cellSize_m,
    }, { originBiasCells: biasCells });
    const result = traceSvoVisibilityRay(ray, source, {
      maximumNodeVisits: SVO_CONTACT_VISIBILITY_CONTRACT.maximumNodeVisitsPerSample,
      maximumLeafVisits: SVO_CONTACT_VISIBILITY_CONTRACT.maximumLeafVisitsPerSample,
      maximumWorkItems: SVO_CONTACT_VISIBILITY_CONTRACT.maximumWorkItemsPerSample,
      maximumIntersections: SVO_CONTACT_VISIBILITY_CONTRACT.maximumIntersectionsPerSample,
      allowTransmission: true,
      minimumTransmittance: 1e-3,
    });
    if (result.status === "invalid" || result.status === "exhausted") {
      return { status: result.status, visibility: [0, 0, 0], radius_m };
    }
    for (let channel = 0; channel < 3; channel += 1) total[channel] += result.transmittance[channel];
  }
  return {
    status: "resolved",
    visibility: [
      Math.min(1, Math.max(0, total[0] / SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount)),
      Math.min(1, Math.max(0, total[1] / SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount)),
      Math.min(1, Math.max(0, total[2] / SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount)),
    ],
    radius_m,
  };
}
