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
  /**
   * How far the consumer promises to keep tracing exactly, in page diagonals.
   * Two is the smallest sound choice that excuses a receiver's whole 26-page
   * neighbourhood, because a corner neighbour's farthest point is exactly two
   * diagonals away. Raising it buys more certificates and costs a longer local
   * trace; see docs/plan-cached-static-visibility.md for the measured curve.
   */
  localTraceReachPageDiagonals?: number;
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
  /**
   * Metres of exact tracing a certified receiver still owes. Every blocker the
   * proof excused lies wholly within this distance of the receiver page, so a
   * consumer that clips its static traversal any shorter reintroduces the light
   * leaks the certificate was supposed to be free of.
   */
  localTraceReach_m: number;
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

/**
 * Farthest any point of `blocker` can lie from any point of `receiver`. A
 * blocker within the consumer's local-trace reach by this measure is found by
 * that trace from *every* point of the receiver page, which is what makes
 * excusing it from the proof sound — the shaded point is not known at bake time.
 */
function maximumSeparation_m(receiver: SvoStaticShadowAabb, blocker: SvoStaticShadowAabb): number {
  let total = 0;
  for (let axis = 0; axis < 3; axis += 1) {
    total += Math.max(
      Math.abs(blocker.maximum[axis] - receiver.minimum[axis]),
      Math.abs(receiver.maximum[axis] - blocker.minimum[axis]),
    ) ** 2;
  }
  return Math.sqrt(total);
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
 * How far a certified receiver still has to trace.
 *
 * A `visible` certificate withholds nothing about the receiver page's own
 * contents — every blocker it excused was strictly inside that page — so the
 * consumer keeps tracing locally and only skips the remainder. A ray leaving
 * any interior point of a box has left the box by its diameter, so the page
 * diagonal is a sufficient clip for every ray direction and every light kind.
 * This is the whole return on a certificate: centimetres of traversal in place
 * of a scene-length ray.
 */
export function svoStaticShadowLocalTraceReach_m(pageSize_m: Triple, level = 0): number {
  if (!Number.isSafeInteger(level) || level < 0) throw new RangeError("Static-shadow page level must be a non-negative integer");
  if (pageSize_m.length !== 3 || pageSize_m.some((value) => !Number.isFinite(value) || value <= 0)) {
    throw new RangeError("Static-shadow page size must contain three positive finite metres");
  }
  return Math.hypot(...pageSize_m) * 2 ** level;
}

/**
 * Shader twin of `svoStaticShadowLocalTraceReach_m`. Both sides must agree on
 * the metre count, so the formula is authored once and asserted in both.
 */
export const svoStaticShadowLocalTraceWGSL = /* wgsl */ `
fn svoStaticShadowLocalTraceReach_m(cellSize_m:vec3f,interiorSize:u32,level:u32)->f32{
  return length(cellSize_m*f32(interiorSize))*exp2(f32(level));
}
`;

function centreAndHalfExtent(bounds: SvoStaticShadowAabb): readonly [Triple, Triple] {
  const centre = bounds.minimum.map((value, axis) => (value + bounds.maximum[axis]) / 2) as unknown as Triple;
  const half = bounds.maximum.map((value, axis) => (value - bounds.minimum[axis]) / 2) as unknown as Triple;
  return [centre, half];
}

/**
 * Conservative test for the bundle of segments joining the receiver box to a
 * finite emitter's support. A false result is a mathematical clear proof; a
 * true result remains unknown and takes the exact path.
 *
 * Every hull point is `(1-t)a + t b` for `a` in the receiver and `b` in the
 * emitter, so its offset from the centre segment is bounded per axis by
 * `(1-t)receiverHalf + t emitterHalf <= max(receiverHalf, emitterHalf)`. The
 * hull therefore sits inside the centre segment grown by that per-axis margin,
 * which turns the bundle into one ray/slab test against the grown blocker.
 *
 * The union AABB of receiver and emitter is also a superset, but a ruinously
 * loose one: for a floor page under a ceiling fixture it spans the entire
 * column between them, so every intervening page vetoes the proof and finite
 * lights certify nothing. That looseness is what made point and area fixtures —
 * the lights whose count actually drives frame time — pay a full scene ray per
 * receiver.
 */
export function svoFinitePageBeamIntersectsAabb(
  receiver: SvoStaticShadowAabb,
  blocker: SvoStaticShadowAabb,
  light: SvoLightRecord,
): boolean {
  const emitter = emitterBounds(light);
  const [receiverCentre, receiverHalf] = centreAndHalfExtent(receiver);
  const [emitterCentre, emitterHalf] = centreAndHalfExtent(emitter);
  const offset = emitterCentre.map((value, axis) => value - receiverCentre[axis]);
  const separation = Math.hypot(...offset);
  // An emitter inside the receiver page has no single shadow direction; the
  // exact trace owns it.
  if (!(separation > 1e-12) || !Number.isFinite(separation)) return true;
  const margin = receiverHalf.map((value, axis) => Math.max(value, emitterHalf[axis]));
  let first = 0;
  let last = separation;
  for (let axis = 0; axis < 3; axis += 1) {
    const minimum = blocker.minimum[axis] - margin[axis];
    const maximum = blocker.maximum[axis] + margin[axis];
    const component = offset[axis] / separation;
    if (Math.abs(component) <= 1e-12) {
      // Half-open page ownership: boundary-only contact belongs to one local
      // page and cannot certify a remote volume blocker.
      if (receiverCentre[axis] <= minimum || receiverCentre[axis] >= maximum) return false;
      continue;
    }
    const a = (minimum - receiverCentre[axis]) / component;
    const b = (maximum - receiverCentre[axis]) / component;
    first = Math.max(first, Math.min(a, b));
    last = Math.min(last, Math.max(a, b));
    if (last < first) return false;
  }
  return last > first;
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
 * authored/terrain blocker. Blockers within `localTraceReach_m` of the receiver
 * page are excused, because integration keeps tracing exactly that far and will
 * find them itself. Any remaining page whose conservative AABB intersects the
 * page-to-light hard-ray swept volume makes the result unknown. Unknown pairs
 * publish nothing and therefore use the exact path. The production cone marcher
 * needs an aperture/filter-footprint inflated proof and must not consume these
 * certificates.
 *
 * The reach is the whole economics of this producer, and the measured curve is
 * unkind: short reaches certify almost nothing in a furnished room, and the
 * reaches that certify most of the scene are a large fraction of the scene
 * diagonal, so the local trace they mandate costs about what the full ray did.
 * docs/plan-cached-static-visibility.md records the numbers.
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
  const diagonals = options.localTraceReachPageDiagonals ?? 2;
  if (!Number.isFinite(diagonals) || diagonals < 1) {
    // Below one diagonal the receiver's own page is not even excused, so the
    // proof would contradict the local trace it depends on.
    throw new RangeError("Static-shadow local-trace reach must be at least one page diagonal");
  }
  const localTraceReach_m = svoStaticShadowLocalTraceReach_m(staticMips.basePageSize_m) * diagonals;
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
      localTraceReach_m,
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
        // The local exact trace owns every blocker it is guaranteed to reach
        // from whichever point of this page ends up being shaded.
        if (maximumSeparation_m(receiverBounds, blocker) <= localTraceReach_m) continue;
        const intersects = light.kind === "directional"
          ? svoDirectionalPageBeamIntersectsAabb(receiverBounds, blocker, light.direction)
          : svoFinitePageBeamIntersectsAabb(receiverBounds, blocker, light);
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
    localTraceReach_m,
    proofs,
    coverage,
    visiblePageLightPairs: proofs.length,
    totalPageLightPairs,
    visibleFraction: totalPageLightPairs === 0 ? 1 : proofs.length / totalPageLightPairs,
  };
}
