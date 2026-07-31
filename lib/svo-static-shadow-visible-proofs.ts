import type { SvoLightRecord } from "./svo-light-abi";
import type { SvoNodeMipCoordinate } from "./svo-node-mip-pyramid";
import type { SvoStaticNodeMipPublication } from "./svo-static-node-mips";
import type { SvoStaticShadowFieldPlan } from "./svo-static-shadow-field";
import type { WebGpuSvoStaticShadowProof } from "./webgpu-svo-static-shadow-field";

type Triple = readonly [number, number, number];

export interface SvoStaticShadowAabb {
  minimum: Triple;
  maximum: Triple;
}

export interface SvoStaticShadowVisibleProofOptions {
  /**
   * Optional complete conservative cover of static blockers. This permits safe
   * proofs when the sampled node-mip capacity omitted proxy pages. Terrain must
   * have no omitted pages, or be included in this cover by the caller.
   */
  completeStaticBlockerBounds?: readonly SvoStaticShadowAabb[];
  /** Level-zero is the useful receiver lookup today; defaults to level zero only. */
  receiverLevels?: readonly number[];
}

export interface SvoStaticShadowVisibleProofLightCoverage {
  lightId: number;
  kind: SvoLightRecord["kind"];
  visiblePages: number;
  totalPages: number;
  visibleFraction: number;
}

export interface SvoStaticShadowVisibleProofPublication {
  cacheKey: string;
  sourceGeneration: number;
  lightRevision: number;
  /**
   * These certificates cover static blockers outside the receiver page only.
   * They certify hard rays/segments only: the swept volume is not inflated by
   * the production cone aperture or node-mip sampling support. They therefore
   * cannot skip `dryConeVisibility`. The consumer must also retain an exact
   * receiver-page trace and the live rigid overlay.
   */
  semantics: "remote-static-hard-ray-visible";
  maximumCertifiedConeApertureRadians: 0;
  includesNodeMipSamplingSupport: false;
  requiresExactLocalPageTrace: true;
  requiresCurrentFrameRigidOverlay: true;
  proofs: readonly WebGpuSvoStaticShadowProof[];
  coverage: readonly SvoStaticShadowVisibleProofLightCoverage[];
  visiblePageLightPairs: number;
  totalPageLightPairs: number;
  visibleFraction: number;
  failClosedReason?: "omitted-static-pages";
}

function aabbForPage(
  coordinate: SvoNodeMipCoordinate,
  level: number,
  origin: Triple,
  basePageSize: Triple,
): SvoStaticShadowAabb {
  const scale = 2 ** level;
  const minimum = coordinate.map((value, axis) => origin[axis] + value * basePageSize[axis] * scale) as unknown as Triple;
  const maximum = minimum.map((value, axis) => value + basePageSize[axis] * scale) as unknown as Triple;
  return { minimum, maximum };
}

function contains(outer: SvoStaticShadowAabb, inner: SvoStaticShadowAabb): boolean {
  return outer.minimum.every((value, axis) => inner.minimum[axis] >= value && inner.maximum[axis] <= outer.maximum[axis]);
}

function overlaps(left: SvoStaticShadowAabb, right: SvoStaticShadowAabb): boolean {
  return left.minimum.every((value, axis) => left.maximum[axis] >= right.minimum[axis]
    && right.maximum[axis] >= value);
}

/**
 * Exact intersection test for the infinite bundle of parallel rays originating
 * anywhere in receiver AABB. It is a ray/slab test against blocker-receiver,
 * the Minkowski difference of the two boxes.
 */
export function svoDirectionalPageBeamIntersectsAabb(
  receiver: SvoStaticShadowAabb,
  blocker: SvoStaticShadowAabb,
  towardLightDirection: Triple,
): boolean {
  const length = Math.hypot(...towardLightDirection);
  if (!(length > 1e-12) || !Number.isFinite(length)) throw new RangeError("Directional proof requires a finite non-zero direction");
  const direction = towardLightDirection.map((value) => value / length);
  let first = 0;
  let last = Number.POSITIVE_INFINITY;
  for (let axis = 0; axis < 3; axis += 1) {
    const minimum = blocker.minimum[axis] - receiver.maximum[axis];
    const maximum = blocker.maximum[axis] - receiver.minimum[axis];
    const component = direction[axis];
    if (Math.abs(component) <= 1e-12) {
      // Sparse page ownership is half-open. Boundary-only contact belongs to
      // one local page and cannot certify a remote volume blocker.
      if (minimum >= 0 || maximum <= 0) return false;
      continue;
    }
    const a = minimum / component, b = maximum / component;
    first = Math.max(first, Math.min(a, b));
    last = Math.min(last, Math.max(a, b));
    if (last < first) return false;
  }
  return last > first;
}

function emitterBounds(light: SvoLightRecord): SvoStaticShadowAabb {
  if (light.kind === "point" || light.kind === "sphereArea") {
    const radius = light.radius_m;
    return {
      minimum: light.position_m.map((value) => value - radius) as unknown as Triple,
      maximum: light.position_m.map((value) => value + radius) as unknown as Triple,
    };
  }
  if (light.kind === "rectangleArea") {
    const extent = [0, 1, 2].map((axis) =>
      Math.abs(light.axisU[axis]) * light.halfWidth_m + Math.abs(light.axisV[axis]) * light.halfHeight_m);
    return {
      minimum: light.position_m.map((value, axis) => value - extent[axis]) as unknown as Triple,
      maximum: light.position_m.map((value, axis) => value + extent[axis]) as unknown as Triple,
    };
  }
  throw new Error("Directional lights do not have finite emitter bounds");
}

/**
 * Conservative superset of every segment from the receiver box to the finite
 * emitter support. A false result is therefore a mathematical clear proof;
 * a true result remains unknown and takes the exact path.
 */
function finiteLightBeamIntersectsAabb(receiver: SvoStaticShadowAabb, blocker: SvoStaticShadowAabb, light: SvoLightRecord): boolean {
  const emitter = emitterBounds(light);
  const sweptBounds: SvoStaticShadowAabb = {
    minimum: receiver.minimum.map((value, axis) => Math.min(value, emitter.minimum[axis])) as unknown as Triple,
    maximum: receiver.maximum.map((value, axis) => Math.max(value, emitter.maximum[axis])) as unknown as Triple,
  };
  return overlaps(sweptBounds, blocker);
}

function blockerPagesFromBounds(
  bounds: readonly SvoStaticShadowAabb[],
  origin: Triple,
  basePageSize: Triple,
): SvoStaticShadowAabb[] {
  const coordinates = new Map<string, SvoNodeMipCoordinate>();
  for (const bound of bounds) {
    if ([...bound.minimum, ...bound.maximum].some((value) => !Number.isFinite(value))) {
      throw new RangeError("Static-shadow blocker bounds must be finite");
    }
    const first = bound.minimum.map((value, axis) => Math.max(0, Math.floor((value - origin[axis]) / basePageSize[axis])));
    const last = bound.maximum.map((value, axis) =>
      Math.max(0, Math.ceil((value - origin[axis]) / basePageSize[axis]) - 1));
    for (let z = first[2]; z <= last[2]; z += 1) for (let y = first[1]; y <= last[1]; y += 1) {
      for (let x = first[0]; x <= last[0]; x += 1) coordinates.set(`${x},${y},${z}`, [x, y, z]);
    }
  }
  return [...coordinates.values()].map((coordinate) => aabbForPage(coordinate, 0, origin, basePageSize));
}

function validatesExactTopology(staticMips: SvoStaticNodeMipPublication, shadowPlan: SvoStaticShadowFieldPlan): boolean {
  if (staticMips.generation !== shadowPlan.sourceGeneration
      || staticMips.plan.pages.length !== shadowPlan.pages.length
      || staticMips.plan.atlas.capacity !== shadowPlan.atlasCapacity) return false;
  return staticMips.plan.pages.every((page, index) =>
    page.keyString === shadowPlan.pages[index]?.keyString && page.slot === shadowPlan.pages[index]?.slot);
}

/**
 * Produces mathematically conservative VISIBLE certificates for hard rays.
 *
 * Level-zero node-mip pages are a conservative AABB cover of every selected
 * authored/terrain blocker. The receiver page is excluded because integration
 * retains an exact local trace until that page is exited. Any remote page whose
 * conservative AABB intersects the page-to-light hard-ray swept volume makes
 * the result unknown. Unknown pairs publish nothing and therefore use the exact
 * path. The production cone marcher needs an aperture/filter-footprint inflated
 * proof and must not consume these certificates.
 */
export function buildSvoStaticShadowVisibleProofs(
  staticMips: SvoStaticNodeMipPublication,
  lights: readonly SvoLightRecord[],
  shadowPlan: SvoStaticShadowFieldPlan,
  options: SvoStaticShadowVisibleProofOptions = {},
): SvoStaticShadowVisibleProofPublication {
  if (!validatesExactTopology(staticMips, shadowPlan)) {
    throw new Error("Static-shadow visible proofs require the exact node-mip topology and generation");
  }
  if (lights.length !== shadowPlan.lightChannels.length
      || lights.some((light, index) => light.lightId !== shadowPlan.lightChannels[index].lightId
        || light.revision !== shadowPlan.lightRevision)) {
    throw new Error("Static-shadow visible proofs require the exact authored-light revision and order");
  }
  const totalPageLightPairs = shadowPlan.pages.length * lights.length;
  const completeBlockerBounds = options.completeStaticBlockerBounds;
  const suppliedCompleteCover = completeBlockerBounds !== undefined
    && (staticMips.terrainCandidatePageCount === 0 || staticMips.omittedBasePageCount === 0);
  if (staticMips.omittedBasePageCount !== 0 && !suppliedCompleteCover) {
    return {
      cacheKey: shadowPlan.cacheKey,
      sourceGeneration: shadowPlan.sourceGeneration,
      lightRevision: shadowPlan.lightRevision,
      semantics: "remote-static-hard-ray-visible",
      maximumCertifiedConeApertureRadians: 0,
      includesNodeMipSamplingSupport: false,
      requiresExactLocalPageTrace: true,
      requiresCurrentFrameRigidOverlay: true,
      proofs: [],
      coverage: lights.map((light) => ({
        lightId: light.lightId,
        kind: light.kind,
        visiblePages: 0,
        totalPages: shadowPlan.pages.length,
        visibleFraction: 0,
      })),
      visiblePageLightPairs: 0,
      totalPageLightPairs,
      visibleFraction: 0,
      failClosedReason: "omitted-static-pages",
    };
  }

  const origin = staticMips.worldOrigin_m;
  const baseSize = staticMips.basePageSize_m;
  const blockerBounds = completeBlockerBounds
    ? blockerPagesFromBounds(completeBlockerBounds, origin, baseSize)
    : staticMips.plan.pages.filter(({ key }) => key.level === 0)
      .map((page) => aabbForPage(page.key.coordinate, 0, origin, baseSize));
  const receiverLevels = new Set(options.receiverLevels ?? [0]);
  const proofs: WebGpuSvoStaticShadowProof[] = [];
  const visibleByLight = new Uint32Array(lights.length);
  for (const receiver of shadowPlan.pages) {
    if (!receiverLevels.has(receiver.key.level)) continue;
    const receiverBounds = aabbForPage(receiver.key.coordinate, receiver.key.level, origin, baseSize);
    for (let lightIndex = 0; lightIndex < lights.length; lightIndex += 1) {
      const light = lights[lightIndex];
      let remoteBlocker = false;
      for (const blocker of blockerBounds) {
        // The local exact trace owns every blocker page wholly inside the receiver page.
        if (contains(receiverBounds, blocker)) continue;
        const intersects = light.kind === "directional"
          ? svoDirectionalPageBeamIntersectsAabb(receiverBounds, blocker, light.direction)
          : finiteLightBeamIntersectsAabb(receiverBounds, blocker, light);
        if (intersects) {
          remoteBlocker = true;
          break;
        }
      }
      if (!remoteBlocker) {
        proofs.push({ page: receiver.keyString, lightId: light.lightId, certificate: "visible" });
        visibleByLight[lightIndex] += 1;
      }
    }
  }
  const coverage = lights.map((light, index): SvoStaticShadowVisibleProofLightCoverage => ({
    lightId: light.lightId,
    kind: light.kind,
    visiblePages: visibleByLight[index],
    totalPages: shadowPlan.pages.length,
    visibleFraction: shadowPlan.pages.length === 0 ? 1 : visibleByLight[index] / shadowPlan.pages.length,
  }));
  return {
    cacheKey: shadowPlan.cacheKey,
    sourceGeneration: shadowPlan.sourceGeneration,
    lightRevision: shadowPlan.lightRevision,
    semantics: "remote-static-hard-ray-visible",
    maximumCertifiedConeApertureRadians: 0,
    includesNodeMipSamplingSupport: false,
    requiresExactLocalPageTrace: true,
    requiresCurrentFrameRigidOverlay: true,
    proofs,
    coverage,
    visiblePageLightPairs: proofs.length,
    totalPageLightPairs,
    visibleFraction: totalPageLightPairs === 0 ? 1 : proofs.length / totalPageLightPairs,
  };
}
