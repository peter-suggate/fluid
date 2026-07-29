import type { SvoAabb, SvoVec3 } from "./webgpu-svo-traversal";

/**
 * Screen-space termination is diagnostic-only until the render hierarchy
 * publishes a representative material and normal for every internal node.
 * The canonical topology currently proves only that descendants exist.
 */
export const SVO_SCREEN_SPACE_TERMINATION_CONTRACT = Object.freeze({
  disabledThresholdPixels: 0,
  defaultTanHalfVerticalFov: 0.72,
  shading: "conservative-aabb-proxy" as const,
  exactShadows: true,
  hasRepresentativeMaterial: false,
  hasRepresentativeNormal: false,
});

export interface SvoScreenSpaceTerminationOptions {
  /** Zero disables termination and preserves exact traversal. */
  thresholdPixels: number;
  viewportHeightPixels: number;
  /** Must match the camera ray generator. The dry renderer currently uses 0.72. */
  tanHalfVerticalFov?: number;
  /** Never approximate nodes shallower than this level. */
  minimumLevel?: number;
}

function finiteVec3(value: SvoVec3, label: string): void {
  if (value.length !== 3 || value.some((component) => !Number.isFinite(component))) {
    throw new RangeError(`${label} must contain three finite components`);
  }
}

function validatedOptions(options: SvoScreenSpaceTerminationOptions): Required<SvoScreenSpaceTerminationOptions> {
  const tanHalfVerticalFov = options.tanHalfVerticalFov ?? SVO_SCREEN_SPACE_TERMINATION_CONTRACT.defaultTanHalfVerticalFov;
  const minimumLevel = options.minimumLevel ?? 0;
  if (!Number.isFinite(options.thresholdPixels) || options.thresholdPixels < 0) {
    throw new RangeError("SVO screen-space threshold must be a non-negative finite pixel count");
  }
  if (!Number.isFinite(options.viewportHeightPixels) || options.viewportHeightPixels <= 0) {
    throw new RangeError("SVO screen-space viewport height must be positive and finite");
  }
  if (!Number.isFinite(tanHalfVerticalFov) || tanHalfVerticalFov <= 0) {
    throw new RangeError("SVO screen-space tan-half-FOV must be positive and finite");
  }
  if (!Number.isInteger(minimumLevel) || minimumLevel < 0 || minimumLevel > 21) {
    throw new RangeError("SVO screen-space minimum level must be an integer from 0 to 21");
  }
  return { ...options, tanHalfVerticalFov, minimumLevel };
}

/**
 * Conservative pixel diameter of the sphere enclosing a node AABB.
 *
 * The sphere avoids view-dependent corner underestimation. If the camera is
 * inside or on the sphere, the footprint is infinite and traversal continues.
 */
export function projectedSvoNodeFootprintPixels(
  bounds: SvoAabb,
  cameraPosition: SvoVec3,
  options: Pick<SvoScreenSpaceTerminationOptions, "viewportHeightPixels" | "tanHalfVerticalFov">,
): number {
  finiteVec3(bounds.minimum, "SVO node minimum");
  finiteVec3(bounds.maximum, "SVO node maximum");
  finiteVec3(cameraPosition, "SVO camera position");
  if (bounds.maximum.some((value, axis) => value < bounds.minimum[axis])) {
    throw new RangeError("SVO node maximum must not be below its minimum");
  }
  const checked = validatedOptions({ thresholdPixels: 0, minimumLevel: 0, ...options });
  const halfExtent = bounds.maximum.map((value, axis) => (value - bounds.minimum[axis]) * 0.5) as [number, number, number];
  const center = bounds.minimum.map((value, axis) => value + halfExtent[axis]) as [number, number, number];
  const radius = Math.hypot(...halfExtent);
  if (radius === 0) return 0;
  const distance = Math.hypot(...center.map((value, axis) => value - cameraPosition[axis]));
  if (!(distance > radius)) return Number.POSITIVE_INFINITY;
  const focalLengthPixels = checked.viewportHeightPixels / (2 * checked.tanHalfVerticalFov);
  // The angular radius of a sphere is asin(radius / distance). Its projected
  // screen radius is f*tan(angle) = f*r/sqrt(d^2-r^2).
  return 2 * focalLengthPixels * radius / Math.sqrt(distance * distance - radius * radius);
}

/** Exact traversal remains the mandatory result for threshold zero. */
export function shouldTerminateSvoNodeScreenSpace(
  bounds: SvoAabb,
  cameraPosition: SvoVec3,
  level: number,
  options: SvoScreenSpaceTerminationOptions,
): boolean {
  const checked = validatedOptions(options);
  if (checked.thresholdPixels === 0 || level < checked.minimumLevel) return false;
  return projectedSvoNodeFootprintPixels(bounds, cameraPosition, checked) <= checked.thresholdPixels;
}

/** Generic WGSL helper shared by future canonical and wide diagnostic paths. */
export const svoScreenSpaceTerminationWGSL = /* wgsl */ `
const SVO_STATUS_SCREEN_SPACE_PROXY: u32 = 7u;
fn svoProjectedNodeFootprintPixels(
  bounds: mat2x3f,
  cameraPosition: vec3f,
  viewportHeightPixels: f32,
  tanHalfVerticalFov: f32,
) -> f32 {
  let halfExtent = (bounds[1] - bounds[0]) * 0.5;
  let radius = length(halfExtent);
  if (radius == 0.0) { return 0.0; }
  let distanceSquared = dot((bounds[0] + halfExtent) - cameraPosition,
    (bounds[0] + halfExtent) - cameraPosition);
  let radiusSquared = radius * radius;
  if (distanceSquared <= radiusSquared) { return 3.402823e38; }
  let focalLengthPixels = viewportHeightPixels / (2.0 * tanHalfVerticalFov);
  return 2.0 * focalLengthPixels * radius / sqrt(distanceSquared - radiusSquared);
}

fn svoShouldTerminateNodeScreenSpace(
  bounds: mat2x3f,
  cameraPosition: vec3f,
  viewportHeightPixels: f32,
  tanHalfVerticalFov: f32,
  thresholdPixels: f32,
  level: u32,
  minimumLevel: u32,
) -> bool {
  return thresholdPixels > 0.0 && level >= minimumLevel
    && svoProjectedNodeFootprintPixels(bounds, cameraPosition, viewportHeightPixels, tanHalfVerticalFov)
      <= thresholdPixels;
}
`;

/**
 * Derive an opt-in primary-ray continuation from the canonical implementation.
 * Keeping this as a generated sibling rather than changing the exact function
 * is the bit-exact safety boundary: threshold zero never calls this variant.
 */
export function createSvoScreenSpaceTraversalWGSL(canonicalTraversalWGSL: string): string {
  const begin = canonicalTraversalWGSL.indexOf("fn svoTraversalContinuationNext(");
  const end = canonicalTraversalWGSL.indexOf("\nfn svoTraverseWithDepthLimit", begin);
  if (begin < 0 || end < 0) throw new Error("Canonical SVO continuation function was not found");
  const canonical = canonicalTraversalWGSL.slice(begin, end);
  const leafBranch = "    if (node.links.z != SVO_INVALID) {";
  if (!canonical.includes(leafBranch)) throw new Error("Canonical SVO leaf branch was not found");
  const leafBacklinkValidation = `      if (leaf.topology.x != current.nodeIndex) {
        (*continuation).status = SVO_STATUS_INVALID_TOPOLOGY;
        return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
      }`;
  if (!canonical.includes(leafBacklinkValidation)) throw new Error("Canonical SVO leaf validation was not found");
  const proxyBranch = /* wgsl */ `    // Diagnostic-only: internal nodes have no representative material or normal.
    if (node.links.z == SVO_INVALID) {
      var proxyBounds = (*continuation).currentBounds;
      if ((*continuation).currentBoundsValid == 0u) { proxyBounds = svoNodeBounds(node, mapping); }
      if (drySvoShouldTerminateNodeScreenSpace(proxyBounds, node.address.z)) {
        let hit = SvoTraversalHit(SVO_STATUS_SCREEN_SPACE_PROXY, visits, current.nodeIndex,
          SVO_INVALID, 0u, node.address.z, max(current.tEnter, ray.tMin), min(current.tExit, ray.tMax));
        svoTraversalContinuationAdvance(continuation);
        return hit;
      }
    }
`;
  const leafProxyBranch = /* wgsl */ `
      // A leaf-level proxy also skips the 8^3 payload DDA, but only after the
      // leaf record and backlink have been validated so LOD cannot hide damage.
      var proxyBounds = (*continuation).currentBounds;
      if ((*continuation).currentBoundsValid == 0u) { proxyBounds = svoNodeBounds(node, mapping); }
      if (drySvoShouldTerminateNodeScreenSpace(proxyBounds, node.address.z)) {
        let hit = SvoTraversalHit(SVO_STATUS_SCREEN_SPACE_PROXY, visits, current.nodeIndex,
          node.links.z, leaf.topology.y, node.address.z, max(current.tEnter, ray.tMin), min(current.tExit, ray.tMax));
        svoTraversalContinuationAdvance(continuation);
        return hit;
      }`;
  const derived = canonical
    .replace("fn svoTraversalContinuationNext(", "fn svoTraversalContinuationNextScreenSpace(")
    .replace(leafBranch, `${proxyBranch}${leafBranch}`)
    .replace(leafBacklinkValidation, `${leafBacklinkValidation}${leafProxyBranch}`);
  return `${svoScreenSpaceTerminationWGSL}\n${derived}`;
}

export interface SvoScreenSpaceImageComparison {
  totalPixels: number;
  changedPixels: number;
  changedPercent: number;
  silhouetteDisagreementPixels: number;
  silhouetteDisagreementPercent: number;
  silhouetteFalsePositivePixels: number;
  silhouetteFalseNegativePixels: number;
  depthSilhouetteDisagreementPixels: number;
  depthSilhouetteDisagreementPercent: number;
  depthSilhouetteFalsePositivePixels: number;
  depthSilhouetteFalseNegativePixels: number;
  absoluteDepthError: { mean: number; p50: number; p95: number; p99: number; maximum: number };
  absoluteLuminanceError: { mean: number; p50: number; p95: number; p99: number; maximum: number };
  relativeLuminanceError: { mean: number; p50: number; p95: number; p99: number; maximum: number; denominatorFloor: number };
}

function percentile(sorted: readonly number[], fraction: number): number {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(fraction * sorted.length) - 1))];
}

function summarize(values: number[]): { mean: number; p50: number; p95: number; p99: number; maximum: number } {
  values.sort((left, right) => left - right);
  return {
    mean: values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length),
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95),
    p99: percentile(values, 0.99),
    maximum: values[values.length - 1] ?? 0,
  };
}

/**
 * Compare two linear RGBA radiance/depth images. Alpha <= 0 is a miss, matching
 * the dry-scene G-buffer contract; RGB uses Rec.709 relative luminance.
 */
export function compareSvoScreenSpaceImages(
  reference: Float32Array,
  candidate: Float32Array,
  options: { changedEpsilon?: number; relativeDenominatorFloor?: number; width?: number; height?: number; depthEdgeRelativeThreshold?: number } = {},
): SvoScreenSpaceImageComparison {
  if (reference.length !== candidate.length || reference.length % 4 !== 0) {
    throw new RangeError("SVO screen-space image inputs must be equal-length RGBA arrays");
  }
  const changedEpsilon = options.changedEpsilon ?? 1e-4;
  const denominatorFloor = options.relativeDenominatorFloor ?? 0.01;
  const width = options.width ?? reference.length / 4, height = options.height ?? 1;
  const depthEdgeRelativeThreshold = options.depthEdgeRelativeThreshold ?? 0.02;
  if (!Number.isFinite(changedEpsilon) || changedEpsilon < 0 || !Number.isFinite(denominatorFloor) || denominatorFloor <= 0) {
    throw new RangeError("SVO image comparison tolerances must be finite and non-negative (with a positive denominator floor)");
  }
  if (!Number.isSafeInteger(width) || width <= 0 || !Number.isSafeInteger(height) || height <= 0 || width * height * 4 !== reference.length) {
    throw new RangeError("SVO image comparison dimensions must match the RGBA arrays");
  }
  if (!Number.isFinite(depthEdgeRelativeThreshold) || depthEdgeRelativeThreshold < 0) {
    throw new RangeError("SVO depth-edge threshold must be non-negative and finite");
  }
  const absoluteErrors: number[] = [], relativeErrors: number[] = [], absoluteDepthErrors: number[] = [];
  let changedPixels = 0, silhouetteDisagreementPixels = 0;
  let silhouetteFalsePositivePixels = 0, silhouetteFalseNegativePixels = 0;
  const totalPixels = reference.length / 4;
  for (let pixel = 0; pixel < totalPixels; pixel += 1) {
    const base = pixel * 4;
    const referenceY = 0.2126 * reference[base] + 0.7152 * reference[base + 1] + 0.0722 * reference[base + 2];
    const candidateY = 0.2126 * candidate[base] + 0.7152 * candidate[base + 1] + 0.0722 * candidate[base + 2];
    const absolute = Math.abs(candidateY - referenceY);
    absoluteErrors.push(absolute);
    relativeErrors.push(absolute / Math.max(Math.abs(referenceY), denominatorFloor));
    if (absolute > changedEpsilon || Math.abs(candidate[base + 3] - reference[base + 3]) > changedEpsilon) changedPixels += 1;
    const referenceHit = reference[base + 3] > 0, candidateHit = candidate[base + 3] > 0;
    if (referenceHit && candidateHit) absoluteDepthErrors.push(Math.abs(candidate[base + 3] - reference[base + 3]));
    if (referenceHit !== candidateHit) {
      silhouetteDisagreementPixels += 1;
      if (candidateHit) silhouetteFalsePositivePixels += 1;
      else silhouetteFalseNegativePixels += 1;
    }
  }
  const depthEdge = (pixels: Float32Array, pixel: number): boolean => {
    const depth = pixels[pixel * 4 + 3];
    const x = pixel % width, y = Math.floor(pixel / width);
    for (const neighbor of [x + 1 < width ? pixel + 1 : -1, y + 1 < height ? pixel + width : -1]) {
      if (neighbor < 0) continue;
      const other = pixels[neighbor * 4 + 3];
      if ((depth > 0) !== (other > 0)) return true;
      if (depth > 0 && other > 0 && Math.abs(depth - other) / Math.max(1e-4, Math.min(depth, other)) > depthEdgeRelativeThreshold) return true;
    }
    return false;
  };
  let depthSilhouetteDisagreementPixels = 0, depthSilhouetteFalsePositivePixels = 0, depthSilhouetteFalseNegativePixels = 0;
  for (let pixel = 0; pixel < totalPixels; pixel += 1) {
    const referenceEdge = depthEdge(reference, pixel), candidateEdge = depthEdge(candidate, pixel);
    if (referenceEdge === candidateEdge) continue;
    depthSilhouetteDisagreementPixels += 1;
    if (candidateEdge) depthSilhouetteFalsePositivePixels += 1;
    else depthSilhouetteFalseNegativePixels += 1;
  }
  return {
    totalPixels,
    changedPixels,
    changedPercent: 100 * changedPixels / Math.max(1, totalPixels),
    silhouetteDisagreementPixels,
    silhouetteDisagreementPercent: 100 * silhouetteDisagreementPixels / Math.max(1, totalPixels),
    silhouetteFalsePositivePixels,
    silhouetteFalseNegativePixels,
    depthSilhouetteDisagreementPixels,
    depthSilhouetteDisagreementPercent: 100 * depthSilhouetteDisagreementPixels / Math.max(1, totalPixels),
    depthSilhouetteFalsePositivePixels,
    depthSilhouetteFalseNegativePixels,
    absoluteDepthError: summarize(absoluteDepthErrors),
    absoluteLuminanceError: summarize(absoluteErrors),
    relativeLuminanceError: { ...summarize(relativeErrors), denominatorFloor },
  };
}
