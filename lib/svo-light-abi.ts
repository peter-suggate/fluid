import type { EnvironmentId } from "./environments";
import type { SceneDescription } from "./model";
import {
  cachedSvoStaticPublication,
  hashSvoStaticPublication,
  internSvoStaticPublication,
} from "./svo-static-publication-cache";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
  type EnvironmentProxyPrimitive,
} from "./voxel-environments";

export const SVO_LIGHT_RECORD_STRIDE_BYTES = 112;
export const SVO_LIGHT_RECORD_WORDS = SVO_LIGHT_RECORD_STRIDE_BYTES / Uint32Array.BYTES_PER_ELEMENT;
export const SVO_LIGHT_MAXIMUM_RECORDS = 32;
export const SVO_SCENE_LIGHT_VERSION = "1" as const;

const sceneLightCache = new Map<string, SvoSceneLights>();

export const SVO_LIGHT_KINDS = Object.freeze({
  directional: 1,
  point: 2,
  sphereArea: 3,
  rectangleArea: 4,
} as const);

export type SvoLightKind = keyof typeof SVO_LIGHT_KINDS;
type Vec3Tuple = readonly [number, number, number];

export interface SvoLightRecord {
  lightId: number;
  ownerId: number;
  revision: number;
  kind: SvoLightKind;
  position_m: Vec3Tuple;
  range_m: number;
  direction: Vec3Tuple;
  colorLinear: Vec3Tuple;
  intensity: number;
  axisU: Vec3Tuple;
  halfWidth_m: number;
  axisV: Vec3Tuple;
  halfHeight_m: number;
  /** Area radius, or the finite emissive endpoint radius for a point fixture. */
  radius_m: number;
  sourceKey: string;
}

function vec3(value: Vec3Tuple, label: string, nonNegative = false): [number, number, number] {
  if (value.length !== 3 || value.some((entry) => !Number.isFinite(entry) || (nonNegative && entry < 0))) {
    throw new RangeError(`${label} must contain three ${nonNegative ? "non-negative " : ""}finite values`);
  }
  return [...value];
}

function normalized(value: Vec3Tuple, label: string): [number, number, number] {
  const input = vec3(value, label);
  const length = Math.hypot(...input);
  if (!(length > 1e-9)) throw new RangeError(`${label} must be non-zero`);
  return input.map((entry) => entry / length) as [number, number, number];
}

export function canonicalSvoLightRecord(input: SvoLightRecord): SvoLightRecord {
  for (const [value, label] of [[input.lightId, "ID"], [input.ownerId, "owner ID"], [input.revision, "revision"]] as const) {
    if (!Number.isSafeInteger(value) || value < 0 || value > 0xffff_ffff) throw new RangeError(`SVO light ${label} must be an unsigned 32-bit integer`);
  }
  if (input.lightId === 0) throw new RangeError("SVO light ID zero is reserved");
  const range_m = input.kind === "directional" ? 0 : input.range_m;
  if (!Number.isFinite(range_m) || range_m < 0) throw new RangeError("SVO light range must be finite and non-negative");
  if (!Number.isFinite(input.intensity) || input.intensity < 0) throw new RangeError("SVO light intensity must be finite and non-negative");
  const radius_m = input.kind === "sphereArea" || input.kind === "point" ? input.radius_m : 0;
  const halfWidth_m = input.kind === "rectangleArea" ? input.halfWidth_m : 0;
  const halfHeight_m = input.kind === "rectangleArea" ? input.halfHeight_m : 0;
  if (![radius_m, halfWidth_m, halfHeight_m].every((value) => Number.isFinite(value) && value >= 0)) {
    throw new RangeError("SVO light shape dimensions must be finite and non-negative");
  }
  if ((input.kind === "sphereArea" && !(radius_m > 0))
      || (input.kind === "rectangleArea" && (!(halfWidth_m > 0) || !(halfHeight_m > 0)))) {
    throw new RangeError("SVO area lights require positive shape dimensions");
  }
  return Object.freeze({
    ...input,
    position_m: vec3(input.position_m, "SVO light position"),
    range_m,
    direction: normalized(input.direction, "SVO light direction"),
    colorLinear: vec3(input.colorLinear, "SVO light color", true),
    intensity: input.intensity,
    axisU: normalized(input.axisU, "SVO light U axis"),
    halfWidth_m,
    axisV: normalized(input.axisV, "SVO light V axis"),
    halfHeight_m,
    radius_m,
  });
}

function proxyPhysicalLight(proxy: EnvironmentProxyPrimitive, ownerBase: number, revision: number): SvoLightRecord | undefined {
  if (!(proxy.material.emission > 0) || !proxy.tags.includes("light")) return undefined;
  const common = {
    lightId: proxy.ownerIndex + 2,
    ownerId: ownerBase + proxy.ownerIndex,
    revision,
    position_m: [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z] as Vec3Tuple,
    // Point fixtures use a deliberately finite influence radius. This avoids
    // spending shadow work on negligible contributions outside the authored
    // garden composition and keeps their inverse-square energy bounded.
    range_m: proxy.tags.includes("point-light")
      ? Math.min(4.5, Math.max(1, 3 * Math.sqrt(proxy.material.emission)))
      : Math.max(1, 6 * Math.sqrt(proxy.material.emission)),
    colorLinear: proxy.material.colorLinear,
    intensity: proxy.material.emission,
    sourceKey: proxy.key,
  };
  if (proxy.tags.includes("point-light")) return canonicalSvoLightRecord({
    ...common,
    kind: "point",
    direction: [0, -1, 0], axisU: [1, 0, 0], axisV: [0, 0, 1],
    halfWidth_m: 0, halfHeight_m: 0,
    // The shader still samples one point at the center, but stops its shadow
    // ray at the visible emitter surface so the lantern cannot shadow itself.
    radius_m: proxy.kind === "ellipsoid"
      ? Math.max(proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z)
      : proxy.kind === "cylinder"
        ? Math.max(proxy.radius_m, proxy.halfHeight_m)
        : Math.max(proxy.halfSize_m.x, proxy.halfSize_m.y, proxy.halfSize_m.z),
  });
  if (proxy.kind !== "box") {
    const radius_m = proxy.kind === "ellipsoid"
      ? Math.cbrt(proxy.radius_m.x * proxy.radius_m.y * proxy.radius_m.z)
      : Math.cbrt(proxy.radius_m * proxy.radius_m * proxy.halfHeight_m);
    return canonicalSvoLightRecord({
      ...common,
      kind: "sphereArea",
      direction: [0, -1, 0], axisU: [1, 0, 0], axisV: [0, 0, 1],
      halfWidth_m: 0, halfHeight_m: 0, radius_m,
    });
  }
  const dimensions = [proxy.halfSize_m.x, proxy.halfSize_m.y, proxy.halfSize_m.z] as const;
  const normalAxis = dimensions.indexOf(Math.min(...dimensions));
  const axes: Vec3Tuple[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
  const directionTags = [
    ["emits-positive-x", 0, 1], ["emits-negative-x", 0, -1],
    ["emits-positive-y", 1, 1], ["emits-negative-y", 1, -1],
    ["emits-positive-z", 2, 1], ["emits-negative-z", 2, -1],
  ] as const;
  const authoredDirection = directionTags.filter(([tag]) => proxy.tags.includes(tag));
  if (authoredDirection.length > 1) throw new Error(`Emissive proxy ${proxy.key} has conflicting emission directions`);
  if (authoredDirection[0]?.[1] !== undefined && authoredDirection[0][1] !== normalAxis) {
    throw new Error(`Emissive proxy ${proxy.key} direction is not normal to its thinnest surface`);
  }
  const directionSign = authoredDirection[0]?.[2] ?? -1;
  const surfaceAxes = [0, 1, 2].filter((axis) => axis !== normalAxis);
  return canonicalSvoLightRecord({
    ...common,
    kind: "rectangleArea",
    direction: axes[normalAxis].map((value) => value === 0 ? 0 : directionSign * value) as [number, number, number],
    axisU: axes[surfaceAxes[0]], halfWidth_m: dimensions[surfaceAxes[0]],
    axisV: axes[surfaceAxes[1]], halfHeight_m: dimensions[surfaceAxes[1]], radius_m: 0,
  });
}

export interface BuildSvoSceneLightsOptions {
  environmentId?: EnvironmentId;
  revision?: number;
  maximumRecords?: number;
  directionalDirection?: Vec3Tuple;
  directionalColor?: Vec3Tuple;
  directionalIntensity?: number;
}

export interface SvoSceneLights {
  records: readonly SvoLightRecord[];
  packedRecords: Uint32Array<ArrayBuffer>;
  omittedFixtureKeys: readonly string[];
  revision: number;
  staticRevision: string;
  cacheKey: string;
}

function importance(light: SvoLightRecord): number {
  const luminance = 0.2126 * light.colorLinear[0] + 0.7152 * light.colorLinear[1] + 0.0722 * light.colorLinear[2];
  const area = light.kind === "rectangleArea" ? 4 * light.halfWidth_m * light.halfHeight_m
    : light.kind === "sphereArea" ? 4 * Math.PI * light.radius_m * light.radius_m : 1;
  return luminance * light.intensity * area;
}

export function buildSvoSceneLights(scene: SceneDescription, options: BuildSvoSceneLightsOptions = {}): SvoSceneLights {
  const revision = options.revision ?? 1;
  const maximumRecords = options.maximumRecords ?? SVO_LIGHT_MAXIMUM_RECORDS;
  if (!Number.isSafeInteger(maximumRecords) || maximumRecords < 1 || maximumRecords > SVO_LIGHT_MAXIMUM_RECORDS) {
    throw new RangeError(`SVO light capacity must be from 1 to ${SVO_LIGHT_MAXIMUM_RECORDS}`);
  }
  const directional = canonicalSvoLightRecord({
    lightId: 1, ownerId: 0xffff_ffff, revision, kind: "directional",
    position_m: [0, 0, 0], range_m: 0,
    direction: options.directionalDirection ?? scene.lighting?.directional?.direction ?? [-0.45, 0.86, 0.28],
    colorLinear: options.directionalColor ?? scene.lighting?.directional?.colorLinear ?? [1.04, 1, 0.91],
    intensity: options.directionalIntensity ?? scene.lighting?.directional?.intensity ?? 1,
    axisU: [1, 0, 0], halfWidth_m: 0, axisV: [0, 0, 1], halfHeight_m: 0, radius_m: 0,
    sourceKey: "authored/directional",
  });
  const catalog = buildEnvironmentProxyCatalog(scene, options.environmentId ?? scene.environment ?? "default");
  const fixtures = environmentProxyPrimitives(catalog).map((proxy) => proxyPhysicalLight(proxy, scene.rigidBodies.length, revision))
    .filter((light): light is SvoLightRecord => Boolean(light));
  const selected = fixtures.slice().sort((a, b) => importance(b) - importance(a) || a.lightId - b.lightId).slice(0, maximumRecords - 1);
  const selectedIds = new Set(selected.map((light) => light.lightId));
  const omittedFixtureKeys = fixtures.filter((light) => !selectedIds.has(light.lightId)).map((light) => light.sourceKey).sort();
  const records = [directional, ...selected.sort((a, b) => a.lightId - b.lightId)];
  const staticRevision = hashSvoStaticPublication(new Uint32Array(), JSON.stringify({
    records,
    omittedFixtureKeys,
  }));
  const cacheKey = `svo-scene-lights-v${SVO_SCENE_LIGHT_VERSION}:${catalog.environmentId}:${staticRevision}`;
  const cached = cachedSvoStaticPublication(sceneLightCache, cacheKey);
  if (cached) return cached;
  const packedRecords = packSvoLightRecords(records);
  return internSvoStaticPublication(sceneLightCache, cacheKey, {
    records,
    packedRecords,
    omittedFixtureKeys,
    revision,
    staticRevision,
    cacheKey,
  });
}

export function packSvoLightRecords(records: readonly SvoLightRecord[]): Uint32Array<ArrayBuffer> {
  if (records.length > SVO_LIGHT_MAXIMUM_RECORDS) throw new RangeError("SVO light table exceeds its fixed capacity");
  const canonical = records.map(canonicalSvoLightRecord);
  const ids = new Set<number>();
  const buffer = new ArrayBuffer(canonical.length * SVO_LIGHT_RECORD_STRIDE_BYTES);
  const words = new Uint32Array(buffer), floats = new Float32Array(buffer);
  canonical.forEach((light, index) => {
    if (ids.has(light.lightId)) throw new RangeError(`Duplicate SVO light ID ${light.lightId}`);
    ids.add(light.lightId);
    const offset = index * SVO_LIGHT_RECORD_WORDS;
    floats.set([...light.position_m, light.range_m], offset);
    floats.set([...light.direction, 0], offset + 4);
    floats.set([...light.colorLinear, light.intensity], offset + 8);
    floats.set([...light.axisU, light.halfWidth_m], offset + 12);
    floats.set([...light.axisV, light.halfHeight_m], offset + 16);
    floats.set([light.radius_m, 0, 0, 0], offset + 20);
    words.set([SVO_LIGHT_KINDS[light.kind], light.lightId, light.ownerId, light.revision], offset + 24);
  });
  return words;
}

export const svoLightWGSL = /* wgsl */ `
struct SvoLightRecord {
  positionRange:vec4f,
  directionCone:vec4f,
  colorIntensity:vec4f,
  axisUWidth:vec4f,
  axisVHeight:vec4f,
  shape:vec4f,
  identity:vec4u,
}
const SVO_LIGHT_DIRECTIONAL:u32=1u;
const SVO_LIGHT_POINT:u32=2u;
const SVO_LIGHT_SPHERE_AREA:u32=3u;
const SVO_LIGHT_RECTANGLE_AREA:u32=4u;
fn svoLightRadiance(light:SvoLightRecord)->vec3f{return max(light.colorIntensity.xyz,vec3f(0.0))*max(light.colorIntensity.w,0.0);}
`;
