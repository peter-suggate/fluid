import type { SparseSceneDomainPlan } from "./sparse-scene-domain";
import {
  SVO_NODE_MIP_LAYOUT,
  svoNodeMipPageKey,
  type SvoNodeMipCoordinate,
  type SvoNodeMipPageKey,
  type SvoNodeMipPyramidPlan,
} from "./svo-node-mip-pyramid";
import type { SvoStaticNodeMipPublication } from "./svo-static-node-mips";
import {
  SVO_TETRAHEDRAL_RADIANCE_LAYOUT,
  packSvoRadianceRgb9e5,
  svoTetrahedralLambertianEmission,
  type SvoRadianceRgb,
} from "./svo-tetrahedral-radiance";
import type { EnvironmentProxyPrimitive } from "./voxel-environments";

type Triple = [number, number, number];

export interface SvoStaticEmissiveRadianceOptions {
  /** Surface samples per base-voxel axis. Two is 8 samples/voxel. */
  samplesPerAxis?: 1 | 2 | 4;
  /** Defaults to every proxy with finite, positive authored emission. */
  includeProxy?: (proxy: EnvironmentProxyPrimitive) => boolean;
  /** Optional first-bounce direct illumination; omission retains the emissive-only fast path. */
  primaryDirectionalLight?: SvoStaticPrimaryDirectionalLight;
}

export interface SvoStaticPrimaryDirectionalLight {
  /** Normalized internally; points from the shaded surface toward the light. */
  towardLightDirection: readonly [number, number, number];
  /** Scene-linear incident-light colour. */
  colorLinear: SvoRadianceRgb;
  intensity: number;
  /** Defaults to the scene-domain diagonal, beyond which no resident geometry can matter. */
  shadowDistance_m?: number;
}

export interface SvoStaticEmissiveRadianceInterior {
  /** Identical virtual key (including generation) to the corresponding opacity page. */
  key: SvoNodeMipPageKey;
  /**
   * Tightly packed 8^3 interior in the atlas owner's native layout:
   * four consecutive RGB9E5 words per spatial texel.
   */
  packedInterleaved: Uint32Array<ArrayBuffer>;
  /** True is an explicit promise that all four interiors contain only black. */
  certifiedBlack: boolean;
  /** Non-black texels in this level, useful for upload and injection telemetry. */
  nonBlackTexelCount: number;
}

export interface SvoStaticEmissiveRadiancePublication {
  generation: number;
  /** Shared topology, slots and directory with the opacity publication. */
  plan: SvoNodeMipPyramidPlan;
  interiors: readonly SvoStaticEmissiveRadianceInterior[];
  worldOrigin_m: readonly [number, number, number];
  baseVoxelSize_m: readonly [number, number, number];
  emissiveProxyCount: number;
  injectedBaseTexelCount: number;
  directLightSampleCount: number;
  shadowedDirectLightSampleCount: number;
  blackPageCount: number;
  packedInteriorBytes: number;
}

const DIRECTION_TAGS = [
  ["emits-positive-x", 0, 1], ["emits-negative-x", 0, -1],
  ["emits-positive-y", 1, 1], ["emits-negative-y", 1, -1],
  ["emits-positive-z", 2, 1], ["emits-negative-z", 2, -1],
] as const;

function coordinateKey(value: SvoNodeMipCoordinate): string { return `${value[0]},${value[1]},${value[2]}`; }
function interiorKey(level: number, value: SvoNodeMipCoordinate): string { return `${level}:${coordinateKey(value)}`; }
function dot(a: readonly number[], b: readonly number[]): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function length(value: readonly number[]): number { return Math.hypot(value[0], value[1], value[2]); }
function normalized(value: Triple, fallback: Triple = [0, 1, 0]): Triple {
  const magnitude = length(value);
  return magnitude > 1e-12 ? value.map((component) => component / magnitude) as Triple : fallback;
}

/** World point transformed into proxy-local space by the inverse quaternion. */
function localPoint(proxy: EnvironmentProxyPrimitive, point: Triple): Triple {
  const offset: Triple = [point[0] - proxy.center_m.x, point[1] - proxy.center_m.y, point[2] - proxy.center_m.z];
  const q = proxy.orientation;
  if (!q) return offset;
  const [x, y, z] = offset, ux = -q.x, uy = -q.y, uz = -q.z;
  const tx = 2 * (uy * z - uz * y), ty = 2 * (uz * x - ux * z), tz = 2 * (ux * y - uy * x);
  return [
    x + q.w * tx + (uy * tz - uz * ty),
    y + q.w * ty + (uz * tx - ux * tz),
    z + q.w * tz + (ux * ty - uy * tx),
  ];
}

function worldVector(proxy: EnvironmentProxyPrimitive, vector: Triple): Triple {
  const q = proxy.orientation;
  if (!q) return normalized(vector);
  const [x, y, z] = vector, { w, x: ux, y: uy, z: uz } = q;
  const tx = 2 * (uy * z - uz * y), ty = 2 * (uz * x - ux * z), tz = 2 * (ux * y - uy * x);
  return normalized([
    x + w * tx + (uy * tz - uz * ty),
    y + w * ty + (uz * tx - ux * tz),
    z + w * tz + (ux * ty - uy * tx),
  ]);
}

function authoredNormal(proxy: EnvironmentProxyPrimitive): Triple | undefined {
  const matches = DIRECTION_TAGS.filter(([tag]) => proxy.tags.includes(tag));
  if (matches.length > 1) throw new Error(`Emissive proxy ${proxy.key} has conflicting emission directions`);
  const match = matches[0];
  if (!match) return undefined;
  const result: Triple = [0, 0, 0];
  result[match[1]] = match[2];
  // Direction tags are authored in world axes, matching the light ABI.
  return result;
}

interface SurfaceSample { distance_m: number; normal: Triple }

interface CanonicalDirectionalLight {
  towardLightDirection: Triple;
  colorLinear: SvoRadianceRgb;
  intensity: number;
  shadowDistance_m: number;
}

interface InjectionCounts {
  directLightSamples: number;
  shadowedDirectLightSamples: number;
}

/** Signed-distance estimates are exact for boxes/cylinders/capsules/tori and conservative for ellipsoids/cones. */
function proxySurface(proxy: EnvironmentProxyPrimitive, world: Triple): SurfaceSample {
  const p = localPoint(proxy, world);
  let distance_m = 0, normal: Triple = [0, 1, 0];
  if (proxy.kind === "box") {
    const h = [proxy.halfSize_m.x, proxy.halfSize_m.y, proxy.halfSize_m.z] as Triple;
    const q = p.map((value, axis) => Math.abs(value) - h[axis]) as Triple;
    const outside: Triple = q.map((value) => Math.max(value, 0)) as Triple;
    distance_m = length(outside) + Math.min(Math.max(q[0], q[1], q[2]), 0);
    if (length(outside) > 1e-12) normal = normalized(outside.map((value, axis) => value * Math.sign(p[axis] || 1)) as Triple);
    else {
      const axis = q.indexOf(Math.max(...q));
      normal = [0, 0, 0]; normal[axis] = Math.sign(p[axis] || 1);
    }
  } else if (proxy.kind === "ellipsoid") {
    const r = [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] as Triple;
    const scaled = p.map((value, axis) => value / Math.max(r[axis], 1e-12)) as Triple;
    const k0 = length(scaled);
    const k1 = Math.hypot(p[0] / (r[0] * r[0]), p[1] / (r[1] * r[1]), p[2] / (r[2] * r[2]));
    distance_m = k1 > 1e-12 ? k0 * (k0 - 1) / k1 : -Math.min(...r);
    normal = normalized(p.map((value, axis) => value / (r[axis] * r[axis])) as Triple);
  } else if (proxy.kind === "cylinder") {
    const radial = Math.hypot(p[0], p[2]), side = radial - proxy.radius_m, cap = Math.abs(p[1]) - proxy.halfHeight_m;
    distance_m = Math.hypot(Math.max(side, 0), Math.max(cap, 0)) + Math.min(Math.max(side, cap), 0);
    normal = side > cap ? normalized([p[0], 0, p[2]]) : [0, Math.sign(p[1] || 1), 0];
  } else if (proxy.kind === "capsule") {
    const segmentY = Math.max(-proxy.halfLength_m, Math.min(proxy.halfLength_m, p[1]));
    const delta: Triple = [p[0], p[1] - segmentY, p[2]];
    distance_m = length(delta) - proxy.radius_m;
    normal = normalized(delta);
  } else if (proxy.kind === "torus") {
    const radial = Math.hypot(p[0], p[2]);
    const ring = radial - proxy.majorRadius_m;
    distance_m = Math.hypot(ring, p[1]) - proxy.minorRadius_m;
    const radialNormal: Triple = radial > 1e-12 ? [p[0] / radial, 0, p[2] / radial] : [1, 0, 0];
    normal = normalized([radialNormal[0] * ring, p[1], radialNormal[2] * ring]);
  } else {
    const radial = Math.hypot(p[0], p[2]), height = Math.max(0, Math.min(1, (p[1] + proxy.halfHeight_m) / (2 * proxy.halfHeight_m)));
    const radius = proxy.baseRadius_m + (proxy.topRadius_m - proxy.baseRadius_m) * height;
    const side = radial - radius, cap = Math.abs(p[1]) - proxy.halfHeight_m;
    distance_m = Math.hypot(Math.max(side, 0), Math.max(cap, 0)) + Math.min(Math.max(side, cap), 0);
    const slope = (proxy.baseRadius_m - proxy.topRadius_m) / Math.max(2 * proxy.halfHeight_m, 1e-12);
    normal = side > cap ? normalized([p[0] / Math.max(radial, 1e-12), slope, p[2] / Math.max(radial, 1e-12)]) : [0, Math.sign(p[1] || 1), 0];
  }
  return { distance_m, normal: authoredNormal(proxy) ?? worldVector(proxy, normal) };
}

function overlaps(proxy: EnvironmentProxyPrimitive, minimum: Triple, maximum: Triple): boolean {
  const axes = ["x", "y", "z"] as const;
  return axes.every((axis, index) => proxy.aabb_m.max[axis] >= minimum[index] && proxy.aabb_m.min[axis] <= maximum[index]);
}

/** Conservative slab test: proxy AABB overlap is sufficient to classify a directional sample as shadowed. */
function rayIntersectsProxyBounds(origin: Triple, direction: Triple, maximumDistance: number, proxy: EnvironmentProxyPrimitive): boolean {
  const axes = ["x", "y", "z"] as const;
  let entry = 0, exit = maximumDistance;
  for (let axis = 0; axis < 3; axis += 1) {
    const minimum = proxy.aabb_m.min[axes[axis]], maximum = proxy.aabb_m.max[axes[axis]];
    if (Math.abs(direction[axis]) < 1e-12) {
      if (origin[axis] < minimum || origin[axis] > maximum) return false;
      continue;
    }
    const inverse = 1 / direction[axis];
    const first = (minimum - origin[axis]) * inverse, second = (maximum - origin[axis]) * inverse;
    entry = Math.max(entry, Math.min(first, second));
    exit = Math.min(exit, Math.max(first, second));
    if (entry > exit) return false;
  }
  return exit > 1e-5 && entry < maximumDistance;
}

function visibleToDirectionalLight(
  point: Triple,
  receiver: EnvironmentProxyPrimitive,
  occluders: readonly EnvironmentProxyPrimitive[],
  light: CanonicalDirectionalLight,
): boolean {
  const origin = point.map((value, axis) => value + light.towardLightDirection[axis] * 1e-5) as Triple;
  return !occluders.some((proxy) => proxy !== receiver
    && rayIntersectsProxyBounds(origin, light.towardLightDirection, light.shadowDistance_m, proxy));
}

function canonicalDirectionalLight(
  input: SvoStaticPrimaryDirectionalLight | undefined,
  domain: SparseSceneDomainPlan,
): CanonicalDirectionalLight | undefined {
  if (!input) return undefined;
  if (input.towardLightDirection.some((component) => !Number.isFinite(component))) {
    throw new RangeError("Static primary-light direction must contain three finite components");
  }
  if (!(length(input.towardLightDirection) > 1e-12)) throw new RangeError("Static primary-light direction must be non-zero");
  if (input.colorLinear.some((channel) => channel < 0 || !Number.isFinite(channel))) {
    throw new RangeError("Static primary-light colour must contain three non-negative finite channels");
  }
  if (input.intensity < 0 || !Number.isFinite(input.intensity)) throw new RangeError("Static primary-light intensity must be non-negative and finite");
  const domainDiagonal = Math.hypot(
    domain.worldBounds_m.max.x - domain.worldBounds_m.min.x,
    domain.worldBounds_m.max.y - domain.worldBounds_m.min.y,
    domain.worldBounds_m.max.z - domain.worldBounds_m.min.z,
  );
  const shadowDistance_m = input.shadowDistance_m ?? domainDiagonal;
  if (!(shadowDistance_m > 0) || !Number.isFinite(shadowDistance_m)) throw new RangeError("Static primary-light shadow distance must be positive and finite");
  return {
    towardLightDirection: normalized([...input.towardLightDirection] as Triple),
    colorLinear: [...input.colorLinear],
    intensity: input.intensity,
    shadowDistance_m,
  };
}

function createFloatInterior(): Float32Array<ArrayBuffer> {
  return new Float32Array(SVO_NODE_MIP_LAYOUT.interiorSize ** 3 * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount * 3);
}

function texelOffset(x: number, y: number, z: number): number {
  return ((z * SVO_NODE_MIP_LAYOUT.interiorSize + y) * SVO_NODE_MIP_LAYOUT.interiorSize + x) * 12;
}

function buildBaseInterior(
  page: SvoNodeMipCoordinate,
  domain: SparseSceneDomainPlan,
  surfaces: readonly EnvironmentProxyPrimitive[],
  emissive: ReadonlySet<EnvironmentProxyPrimitive>,
  samplesPerAxis: number,
  light: CanonicalDirectionalLight | undefined,
  occluders: readonly EnvironmentProxyPrimitive[],
  counts: InjectionCounts,
): Float32Array<ArrayBuffer> {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize, result = createFloatInterior();
  const sampleRadius = .5 * Math.hypot(...domain.cellSize_m.map((size) => size / samplesPerAxis));
  const sampleCount = samplesPerAxis ** 3;
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const cell = [page[0] * n + x, page[1] * n + y, page[2] * n + z] as Triple;
    if (cell.some((value, axis) => value < 0 || value >= domain.sceneDimensionsCells[axis])) continue;
    const minimum = cell.map((value, axis) => domain.worldOrigin_m[["x", "y", "z"][axis] as "x" | "y" | "z"] + value * domain.cellSize_m[axis]) as Triple;
    const maximum = minimum.map((value, axis) => value + domain.cellSize_m[axis]) as Triple;
    const candidates = surfaces.filter((proxy) => overlaps(proxy, minimum, maximum));
    if (candidates.length === 0) continue;
    const offset = texelOffset(x, y, z);
    for (let sz = 0; sz < samplesPerAxis; sz += 1) for (let sy = 0; sy < samplesPerAxis; sy += 1) for (let sx = 0; sx < samplesPerAxis; sx += 1) {
      const point = minimum.map((value, axis) => value + ([sx, sy, sz][axis] + .5) * domain.cellSize_m[axis] / samplesPerAxis) as Triple;
      for (const proxy of candidates) {
        const surface = proxySurface(proxy, point);
        if (Math.abs(surface.distance_m) > sampleRadius) continue;
        const emission = emissive.has(proxy) ? proxy.material.emission : 0;
        const emitted: Triple = [
          proxy.material.colorLinear[0] * emission,
          proxy.material.colorLinear[1] * emission,
          proxy.material.colorLinear[2] * emission,
        ];
        if (light) {
          const incidence = Math.max(0, dot(surface.normal, light.towardLightDirection));
          if (incidence > 0) {
            counts.directLightSamples += 1;
            if (visibleToDirectionalLight(point, proxy, occluders, light)) {
              for (let channel = 0; channel < 3; channel += 1) {
                emitted[channel] += proxy.material.colorLinear[channel] * light.colorLinear[channel]
                  * light.intensity * incidence / Math.PI;
              }
            } else counts.shadowedDirectLightSamples += 1;
          }
        }
        const sample = svoTetrahedralLambertianEmission(emitted, surface.normal, 1 / sampleCount);
        for (let direction = 0; direction < 4; direction += 1) for (let channel = 0; channel < 3; channel += 1) {
          result[offset + direction * 3 + channel] += sample[direction][channel];
        }
      }
    }
  }
  return result;
}

function reduceParent(page: SvoNodeMipCoordinate, level: number, values: ReadonlyMap<string, Float32Array>): Float32Array<ArrayBuffer> {
  const n = SVO_NODE_MIP_LAYOUT.interiorSize, result = createFloatInterior();
  for (let z = 0; z < n; z += 1) for (let y = 0; y < n; y += 1) for (let x = 0; x < n; x += 1) {
    const output = texelOffset(x, y, z);
    for (let cz = 0; cz < 2; cz += 1) for (let cy = 0; cy < 2; cy += 1) for (let cx = 0; cx < 2; cx += 1) {
      const fine = [(page[0] * n + x) * 2 + cx, (page[1] * n + y) * 2 + cy, (page[2] * n + z) * 2 + cz] as Triple;
      const childPage = fine.map((value) => Math.floor(value / n)) as Triple;
      const childTexel = fine.map((value) => value % n) as Triple;
      const child = values.get(interiorKey(level - 1, childPage));
      if (!child) continue; // An absent child is explicitly black for radiance.
      const input = texelOffset(childTexel[0], childTexel[1], childTexel[2]);
      for (let lane = 0; lane < 12; lane += 1) result[output + lane] += child[input + lane] / 8;
    }
  }
  return result;
}

function packInterior(key: SvoNodeMipPageKey, interior: Float32Array): SvoStaticEmissiveRadianceInterior {
  const count = SVO_NODE_MIP_LAYOUT.interiorSize ** 3;
  const packedInterleaved = new Uint32Array(count * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount);
  let nonBlackTexelCount = 0;
  for (let texel = 0; texel < count; texel += 1) {
    let nonBlack = false;
    for (let direction = 0; direction < 4; direction += 1) {
      const offset = texel * 12 + direction * 3;
      const word = packSvoRadianceRgb9e5([interior[offset], interior[offset + 1], interior[offset + 2]]);
      packedInterleaved[texel * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.directionCount + direction] = word;
      nonBlack ||= word !== 0;
    }
    if (nonBlack) nonBlackTexelCount += 1;
  }
  return { key, packedInterleaved, certifiedBlack: nonBlackTexelCount === 0, nonBlackTexelCount };
}

/**
 * Publishes authored emissive exitance over an already-complete static opacity
 * topology. Every opacity page receives either packed data or an explicit black
 * certificate, so the atlas owner never has to infer missing-child semantics.
 */
export function buildSvoStaticEmissiveRadiancePublication(
  opacity: SvoStaticNodeMipPublication,
  domain: SparseSceneDomainPlan,
  environmentPrimitives: readonly EnvironmentProxyPrimitive[],
  options: SvoStaticEmissiveRadianceOptions = {},
): SvoStaticEmissiveRadiancePublication {
  if (!opacity.plan.complete) throw new Error("Static emissive radiance requires a complete opacity page plan");
  if (opacity.generation !== opacity.plan.generation) throw new Error("Static opacity publication generation does not match its page plan");
  const samplesPerAxis = options.samplesPerAxis ?? 2;
  if (![1, 2, 4].includes(samplesPerAxis)) throw new RangeError("Static emissive radiance samples per axis must be 1, 2, or 4");
  const include = options.includeProxy ?? ((proxy: EnvironmentProxyPrimitive) => proxy.material.emission > 0 && Number.isFinite(proxy.material.emission));
  const emissive = environmentPrimitives.filter(include);
  for (const proxy of emissive) {
    if (!(proxy.material.emission > 0) || !Number.isFinite(proxy.material.emission)) throw new RangeError(`Emissive proxy ${proxy.key} must have positive finite emission`);
    if (proxy.material.colorLinear.some((channel) => channel < 0 || !Number.isFinite(channel))) throw new RangeError(`Emissive proxy ${proxy.key} has invalid linear colour`);
    authoredNormal(proxy); // Validate conflicts even if capacity omitted its page.
  }
  const light = canonicalDirectionalLight(options.primaryDirectionalLight, domain);
  // Direct bounce visits every authored surface. With no light, retain the
  // original emissive-only set and avoid all visibility work.
  const surfaces = light ? environmentPrimitives : emissive;
  const emissiveSet = new Set(emissive);
  const counts: InjectionCounts = { directLightSamples: 0, shadowedDirectLightSamples: 0 };

  const values = new Map<string, Float32Array<ArrayBuffer>>();
  for (const { key } of opacity.plan.pages.filter(({ key }) => key.level === 0)) {
    values.set(interiorKey(0, key.coordinate), buildBaseInterior(
      key.coordinate, domain, surfaces, emissiveSet, samplesPerAxis, light, environmentPrimitives, counts,
    ));
  }
  const maximumLevel = Math.max(0, ...opacity.plan.pages.map(({ key }) => key.level));
  for (let level = 1; level <= maximumLevel; level += 1) for (const { key } of opacity.plan.pages.filter(({ key }) => key.level === level)) {
    values.set(interiorKey(level, key.coordinate), reduceParent(key.coordinate, level, values));
  }
  const interiors = opacity.plan.pages.map(({ key }) => {
    const value = values.get(interiorKey(key.level, key.coordinate));
    if (!value) throw new Error(`Missing static emissive radiance interior ${svoNodeMipPageKey(key)}`);
    return packInterior(key, value);
  });
  const base = interiors.filter(({ key }) => key.level === 0);
  return {
    generation: opacity.generation,
    plan: opacity.plan,
    interiors,
    worldOrigin_m: opacity.worldOrigin_m,
    baseVoxelSize_m: opacity.baseVoxelSize_m,
    emissiveProxyCount: emissive.length,
    injectedBaseTexelCount: base.reduce((sum, page) => sum + page.nonBlackTexelCount, 0),
    directLightSampleCount: counts.directLightSamples,
    shadowedDirectLightSampleCount: counts.shadowedDirectLightSamples,
    blackPageCount: interiors.filter(({ certifiedBlack }) => certifiedBlack).length,
    packedInteriorBytes: interiors.length * SVO_NODE_MIP_LAYOUT.interiorSize ** 3 * SVO_TETRAHEDRAL_RADIANCE_LAYOUT.bytesPerTexel,
  };
}
