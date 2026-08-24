import type { EnvironmentId } from "../core/environments";
import type { SceneDescription } from "../core/model";
import { SCENE_ENVIRONMENT_OWNER_BASE } from "../core/webgpu-rigid-body";
import { svoSceneLighting } from "./svo-dry-scene-lighting";
import {
  cachedSvoPublication,
  hashSvoPublication,
  internSvoPublication,
} from "./svo-publication-cache";
import {
  buildEnvironmentProxyCatalog,
  environmentProxyPrimitives,
  type EnvironmentProxyPrimitive,
} from "../core/voxel-environments";

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
  spot: 5,
} as const);

/**
 * Every code the kind table assigns, as a set a validator can test against.
 *
 * The publication check on the renderer side has to reject a garbage kind word
 * without knowing what the kinds mean, and it used to do that with a literal
 * upper bound. That bound is a copy of this table, and a copy of a table is a
 * thing that lags it: adding `spot` above made every spot-bearing scene fail
 * its lighting contract with no message naming the kind, because 5 was one past
 * a 4 written when 4 was the last code. Deriving the set here means the next
 * kind is legal the moment it is declared.
 */
export const SVO_LIGHT_KIND_CODES: ReadonlySet<number> = new Set<number>(Object.values(SVO_LIGHT_KINDS));

/**
 * The angular skirt a spot's beam softens over, as a fraction of its own cone.
 *
 * A reflector has one angle — the taper its geometry describes — and a beam
 * with only that angle has a stencil edge no real fixture produces. The inner
 * cone is therefore derived rather than authored: it is the same cone at
 * `1 - SVO_SPOT_PENUMBRA_FRACTION` of the half-angle, so a wide floodlight gets
 * a proportionally wide penumbra and a tight pin-spot a tight one, and the
 * author still sets exactly one number by drawing the fixture.
 */
export const SVO_SPOT_PENUMBRA_FRACTION = 0.35;

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
  /**
   * A spot's beam, as the two cosines the shader interpolates between.
   *
   * Cosines rather than angles because the shader compares against a dot
   * product and would otherwise pay an `acos` per light per sample. Absent on
   * every other kind: those records pack the full sphere into the same two
   * lanes and never read them back, because the beam term is taken under the
   * spot branch rather than multiplied in unconditionally.
   */
  cone?: { cosInner: number; cosOuter: number };
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
  const radius_m = input.kind === "sphereArea" || input.kind === "point" || input.kind === "spot" ? input.radius_m : 0;
  const halfWidth_m = input.kind === "rectangleArea" ? input.halfWidth_m : 0;
  const halfHeight_m = input.kind === "rectangleArea" ? input.halfHeight_m : 0;
  if (![radius_m, halfWidth_m, halfHeight_m].every((value) => Number.isFinite(value) && value >= 0)) {
    throw new RangeError("SVO light shape dimensions must be finite and non-negative");
  }
  if ((input.kind === "sphereArea" && !(radius_m > 0))
      || (input.kind === "rectangleArea" && (!(halfWidth_m > 0) || !(halfHeight_m > 0)))) {
    throw new RangeError("SVO area lights require positive shape dimensions");
  }
  const cone = input.kind === "spot" ? spotCone(input.cone) : undefined;
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
    ...(cone ? { cone } : {}),
  });
}

/**
 * A spot's two cosines, ordered and bounded.
 *
 * `cosInner >= cosOuter` is the whole contract: the shader divides by their
 * difference, so an inverted pair is a beam that brightens outwards and a
 * coincident pair is a division by zero. Both are refused here rather than
 * clamped in the shader, where the record has already lost the name of the
 * fixture that produced it.
 */
function spotCone(cone: SvoLightRecord["cone"]): { cosInner: number; cosOuter: number } {
  if (!cone) throw new RangeError("SVO spot light needs a beam cone");
  const { cosInner, cosOuter } = cone;
  if (![cosInner, cosOuter].every((value) => Number.isFinite(value) && value >= -1 && value <= 1)) {
    throw new RangeError("SVO spot cone cosines must be from -1 to 1");
  }
  if (!(cosInner > cosOuter)) throw new RangeError("SVO spot inner cone must be tighter than its outer cone");
  return { cosInner, cosOuter };
}

/**
 * Emitter size for a proxy of any kind: the radius that stops a point fixture's
 * shadow ray at its own visible surface, or the equal-volume sphere an area
 * emitter is sampled as.
 */
function proxyEmitterRadius(proxy: EnvironmentProxyPrimitive, policy: "bounding" | "equivalent-sphere"): number {
  if (proxy.kind === "ellipsoid") {
    const { x, y, z } = proxy.radius_m;
    return policy === "bounding" ? Math.max(x, y, z) : Math.cbrt(x * y * z);
  }
  if (proxy.kind === "cylinder") {
    return policy === "bounding"
      ? Math.max(proxy.radius_m, proxy.halfHeight_m)
      : Math.cbrt(proxy.radius_m * proxy.radius_m * proxy.halfHeight_m);
  }
  if (proxy.kind === "capsule") {
    return policy === "bounding" ? proxy.halfLength_m + proxy.radius_m : proxy.radius_m;
  }
  if (proxy.kind === "torus") {
    return policy === "bounding" ? proxy.majorRadius_m + proxy.minorRadius_m : proxy.minorRadius_m;
  }
  if (proxy.kind === "cone") {
    const widest = Math.max(proxy.baseRadius_m, proxy.topRadius_m);
    return policy === "bounding" ? Math.max(widest, proxy.halfHeight_m) : Math.cbrt(widest * widest * proxy.halfHeight_m);
  }
  // An aggregate emits as its lobe, for the same reason it voxelizes as one.
  if (proxy.kind === "cluster") {
    const { x, y, z } = proxy.radius_m;
    return policy === "bounding" ? Math.max(x, y, z) : Math.cbrt(x * y * z);
  }
  // A tape emits as its conservative box, for the same reason. The bounding
  // policy takes the box's corner rather than its longest half-extent — the
  // `field-program` kind's own `boundingRadius_m`, which is a corner because the
  // solid is a box and not an ellipsoid.
  if (proxy.kind === "field-program") {
    const { x, y, z } = proxy.halfExtent_m;
    return policy === "bounding" ? Math.hypot(x, y, z) : Math.cbrt(x * y * z);
  }
  return Math.max(proxy.halfSize_m.x, proxy.halfSize_m.y, proxy.halfSize_m.z);
}

function rotatedByProxy(proxy: EnvironmentProxyPrimitive, v: Vec3Tuple): Vec3Tuple {
  const orientation = proxy.orientation;
  if (!orientation) return v;
  const { w, x, y, z } = orientation;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy);
  const ty = 2 * (z * vx - x * vz);
  const tz = 2 * (x * vy - y * vx);
  return [
    vx + w * tx + (y * tz - z * ty),
    vy + w * ty + (z * tx - x * tz),
    vz + w * tz + (x * ty - y * tx),
  ];
}

/**
 * A spot, read off the reflector that produces it.
 *
 * The fixture's own taper *is* the beam: a truncated cone opens from a throat
 * to a mouth, and the half-angle between them is the only angle a real can
 * light has. Deriving it means the author sets the beam by drawing the lamp
 * — no second number that can disagree with the geometry standing in frame.
 *
 * The record is placed at the **mouth**, not at the fixture centre, and that is
 * load-bearing rather than tidy. A cone proxy is a solid, so a shadow ray aimed
 * at its centre would have to cross its own body to arrive; aimed at the mouth
 * cap it arrives at the surface, and every receiver inside the beam is by
 * construction on the open side of that cap. The mouth radius rides along as
 * `radius_m` so the ray stops at the emitting disc rather than in it.
 */
function proxySpotLight(proxy: EnvironmentProxyPrimitive, common: SvoLightCommon): SvoLightRecord {
  if (proxy.kind !== "cone") {
    throw new Error(`Spot fixture ${proxy.key} must be a cone; its taper is the beam angle`);
  }
  const { baseRadius_m: base, topRadius_m: top, halfHeight_m: half } = proxy;
  const mouthRadius_m = Math.max(base, top);
  const throatRadius_m = Math.min(base, top);
  if (!(mouthRadius_m > throatRadius_m)) {
    throw new Error(`Spot fixture ${proxy.key} has no taper, so it describes no beam`);
  }
  if (!(half > 0)) throw new Error(`Spot fixture ${proxy.key} needs a positive half height`);
  // `baseRadius_m` is the -Y cap and `topRadius_m` the +Y one, so the beam runs
  // from the narrow end toward the wide one along the proxy's own axis.
  const localAxis: Vec3Tuple = base > top ? [0, -1, 0] : [0, 1, 0];
  const direction = normalized(rotatedByProxy(proxy, localAxis), `Spot fixture ${proxy.key} axis`);
  const halfAngle_rad = Math.atan2(mouthRadius_m - throatRadius_m, 2 * half);
  return canonicalSvoLightRecord({
    ...common,
    kind: "spot",
    position_m: [
      proxy.center_m.x + direction[0] * half,
      proxy.center_m.y + direction[1] * half,
      proxy.center_m.z + direction[2] * half,
    ],
    direction,
    axisU: [1, 0, 0], axisV: [0, 0, 1],
    halfWidth_m: 0, halfHeight_m: 0,
    radius_m: mouthRadius_m,
    cone: {
      cosOuter: Math.cos(halfAngle_rad),
      cosInner: Math.cos(halfAngle_rad * (1 - SVO_SPOT_PENUMBRA_FRACTION)),
    },
  });
}

interface SvoLightCommon {
  lightId: number;
  ownerId: number;
  revision: number;
  position_m: Vec3Tuple;
  range_m: number;
  colorLinear: Vec3Tuple;
  intensity: number;
  sourceKey: string;
}

function proxyPhysicalLight(proxy: EnvironmentProxyPrimitive, ownerBase: number, revision: number): SvoLightRecord | undefined {
  if (!(proxy.material.emission > 0) || !proxy.tags.includes("light")) return undefined;
  const common: SvoLightCommon = {
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
  if (proxy.tags.includes("spot-light")) return proxySpotLight(proxy, common);
  if (proxy.tags.includes("point-light")) return canonicalSvoLightRecord({
    ...common,
    kind: "point",
    direction: [0, -1, 0], axisU: [1, 0, 0], axisV: [0, 0, 1],
    halfWidth_m: 0, halfHeight_m: 0,
    // The shader still samples one point at the center, but stops its shadow
    // ray at the visible emitter surface so the lantern cannot shadow itself.
    radius_m: proxyEmitterRadius(proxy, "bounding"),
  });
  if (proxy.kind !== "box") {
    const radius_m = proxyEmitterRadius(proxy, "equivalent-sphere");
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
  contentRevision: string;
  cacheKey: string;
}

function importance(light: SvoLightRecord): number {
  const luminance = 0.2126 * light.colorLinear[0] + 0.7152 * light.colorLinear[1] + 0.0722 * light.colorLinear[2];
  const area = light.kind === "rectangleArea" ? 4 * light.halfWidth_m * light.halfHeight_m
    : light.kind === "sphereArea" ? 4 * Math.PI * light.radius_m * light.radius_m
    // A spot emits from its mouth disc and only into its cone, so its emitting
    // area is that disc rather than the sphere the same radius would imply.
    : light.kind === "spot" ? Math.PI * light.radius_m * light.radius_m : 1;
  return luminance * light.intensity * area;
}

export function buildSvoSceneLights(scene: SceneDescription, options: BuildSvoSceneLightsOptions = {}): SvoSceneLights {
  const revision = options.revision ?? 1;
  const maximumRecords = options.maximumRecords ?? SVO_LIGHT_MAXIMUM_RECORDS;
  if (!Number.isSafeInteger(maximumRecords) || maximumRecords < 1 || maximumRecords > SVO_LIGHT_MAXIMUM_RECORDS) {
    throw new RangeError(`SVO light capacity must be from 1 to ${SVO_LIGHT_MAXIMUM_RECORDS}`);
  }
  const sceneLighting = svoSceneLighting(scene);
  const directional = canonicalSvoLightRecord({
    lightId: 1, ownerId: 0xffff_ffff, revision, kind: "directional",
    position_m: [0, 0, 0], range_m: 0,
    direction: options.directionalDirection ?? sceneLighting?.directional?.direction ?? [-0.45, 0.86, 0.28],
    colorLinear: options.directionalColor ?? sceneLighting?.directional?.colorLinear ?? [1.04, 1, 0.91],
    intensity: options.directionalIntensity ?? sceneLighting?.directional?.intensity ?? 1,
    axisU: [1, 0, 0], halfWidth_m: 0, axisV: [0, 0, 1], halfHeight_m: 0, radius_m: 0,
    sourceKey: "authored/directional",
  });
  const catalog = buildEnvironmentProxyCatalog(scene, options.environmentId ?? scene.environment ?? "default");
  const fixtures = environmentProxyPrimitives(catalog).map((proxy) => proxyPhysicalLight(proxy, SCENE_ENVIRONMENT_OWNER_BASE, revision))
    .filter((light): light is SvoLightRecord => Boolean(light));
  const selected = fixtures.slice().sort((a, b) => importance(b) - importance(a) || a.lightId - b.lightId).slice(0, maximumRecords - 1);
  const selectedIds = new Set(selected.map((light) => light.lightId));
  const omittedFixtureKeys = fixtures.filter((light) => !selectedIds.has(light.lightId)).map((light) => light.sourceKey).sort();
  const records = [directional, ...selected.sort((a, b) => a.lightId - b.lightId)];
  const contentRevision = hashSvoPublication(new Uint32Array(), JSON.stringify({
    records,
    omittedFixtureKeys,
  }));
  const cacheKey = `svo-scene-lights-v${SVO_SCENE_LIGHT_VERSION}:${catalog.environmentId}:${contentRevision}`;
  const cached = cachedSvoPublication(sceneLightCache, cacheKey);
  if (cached) return cached;
  const packedRecords = packSvoLightRecords(records);
  return internSvoPublication(sceneLightCache, cacheKey, {
    records,
    packedRecords,
    omittedFixtureKeys,
    revision,
    contentRevision,
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
    // The full sphere on every kind that is not a spot. Nothing reads these
    // lanes there — the beam term lives under the spot branch — but a defined
    // value keeps two records that describe the same fixture byte-identical.
    const cosOuter = light.cone?.cosOuter ?? -1;
    const cosInner = light.cone?.cosInner ?? 1;
    floats.set([...light.position_m, light.range_m], offset);
    floats.set([...light.direction, cosOuter], offset + 4);
    floats.set([...light.colorLinear, light.intensity], offset + 8);
    floats.set([...light.axisU, light.halfWidth_m], offset + 12);
    floats.set([...light.axisV, light.halfHeight_m], offset + 16);
    floats.set([light.radius_m, cosInner, 0, 0], offset + 20);
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
const SVO_LIGHT_SPOT:u32=5u;
fn svoLightRadiance(light:SvoLightRecord)->vec3f{return max(light.colorIntensity.xyz,vec3f(0.0))*max(light.colorIntensity.w,0.0);}
// The beam term, for the spot branch alone. \`towardLight\` points from the
// receiver at the emitter, so the beam axis is compared against its negation.
// Squared so the skirt falls off as a smooth shoulder rather than a linear ramp.
fn svoLightConeFalloff(light:SvoLightRecord,towardLight:vec3f)->f32{
  let cosOuter=light.directionCone.w;let cosInner=light.shape.y;
  let alignment=dot(normalize(light.directionCone.xyz),-towardLight);
  let t=clamp((alignment-cosOuter)/max(cosInner-cosOuter,1e-4),0.0,1.0);
  return t*t;
}
`;

/**
 * The scene's dominant light, expressed as the directional the water can use.
 *
 * The raster water pipeline has no light table. Its entire rig is one key,
 * resolved by `resolveWaterKeyLight` from `lighting.directional` and nothing
 * else — so on a set whose illumination is a fixture rather than a sun, the
 * water is keyed by a fill while everything around it is lit by a lamp. On the
 * stage that gap is measured rather than argued: the practical delivers about
 * eighteen at the middle of the tank against the fill's 0.25, which is the same
 * fifty-fold the frame already shows between the lit floor and a wall the beam
 * cannot reach.
 *
 * This closes it by evaluating the table at one point and handing the winner
 * back in the shape `lighting.directional` already has, so nothing downstream
 * has to learn what a fixture is. The evaluation is `dryLightSample`'s,
 * deliberately restated here in the small — centre sample rather than the
 * penumbra jitter, same range fade, same shape scale, same beam term — because
 * the whole point is that the lamp the water is keyed by is the lamp the dry
 * shader is lighting the floor with, and a second model of the same fixture is
 * exactly how those two come to disagree about its strength.
 *
 * One point, because a directional has one direction. The receiver is the
 * middle of the container: a fixture is hung to light a vessel, so the vessel's
 * centre is where its beam is aimed and where the inverse square it arrives
 * attenuated by should be read.
 */
export interface WaterKeyDirectional {
  readonly direction: [number, number, number];
  readonly colorLinear: [number, number, number];
  readonly intensity: number;
}

const REC709 = [0.2126, 0.7152, 0.0722] as const;

function rec709Luminance(color: Vec3Tuple): number {
  return REC709[0] * color[0] + REC709[1] * color[1] + REC709[2] * color[2];
}

interface LightArrival {
  direction: [number, number, number];
  radiance: [number, number, number];
}

/** `dryLightSample` on the CPU, for one receiver and the record's own centre. */
function lightArrivalAt(light: SvoLightRecord, receiver_m: Vec3Tuple): LightArrival | undefined {
  const base = light.colorLinear.map((channel) => channel * light.intensity) as [number, number, number];
  if (!(Math.max(...base) > 0)) return undefined;
  if (light.kind === "directional") {
    const length = Math.hypot(...light.direction);
    if (!(length > 1e-6)) return undefined;
    return { direction: light.direction.map((axis) => axis / length) as [number, number, number], radiance: base };
  }
  const offset = light.position_m.map((axis, index) => axis - receiver_m[index]) as [number, number, number];
  const distanceSquared = offset[0] ** 2 + offset[1] ** 2 + offset[2] ** 2;
  if (!(distanceSquared > 1e-10)) return undefined;
  if (light.range_m > 0 && distanceSquared >= light.range_m ** 2) return undefined;
  const distance = Math.sqrt(distanceSquared);
  const towardLight = offset.map((axis) => axis / distance) as [number, number, number];
  const rangeFade = light.range_m > 0
    ? Math.min(1, Math.max(0, 1 - distance / light.range_m)) ** 2
    : 1;
  let shapeScale = 1 / Math.max(1, distanceSquared);
  if (light.kind === "sphereArea") {
    const area = 4 * Math.PI * light.radius_m ** 2;
    shapeScale = area / Math.max(area, distanceSquared);
  }
  if (light.kind === "rectangleArea") {
    const area = 4 * light.halfWidth_m * light.halfHeight_m;
    const axisLength = Math.hypot(...light.direction);
    const facing = axisLength > 1e-9
      ? Math.max(0, -(light.direction[0] * towardLight[0] + light.direction[1] * towardLight[1]
        + light.direction[2] * towardLight[2]) / axisLength)
      : 0;
    shapeScale = facing * area / Math.max(area, distanceSquared);
  }
  if (light.kind === "spot") {
    const axisLength = Math.hypot(...light.direction);
    const alignment = axisLength > 1e-9
      ? -(light.direction[0] * towardLight[0] + light.direction[1] * towardLight[1]
        + light.direction[2] * towardLight[2]) / axisLength
      : -1;
    const cosOuter = light.cone?.cosOuter ?? -1;
    const cosInner = light.cone?.cosInner ?? 1;
    const beam = Math.min(1, Math.max(0, (alignment - cosOuter) / Math.max(cosInner - cosOuter, 1e-4)));
    shapeScale *= beam * beam;
  }
  const radiance = base.map((channel) => channel * rangeFade * shapeScale) as [number, number, number];
  if (!(Math.max(...radiance) > 0)) return undefined;
  return { direction: towardLight, radiance };
}

/**
 * Reduce a published light table to the one key the water is entitled to.
 *
 * The authored directional is in the table as a record like any other, so it
 * competes on the same footing and wins on any set where no fixture reaches the
 * water — which is what keeps every sunlit scene exactly where it was.
 */
export function waterKeyDirectionalFromSceneLights(
  records: readonly SvoLightRecord[],
  receiver_m: Vec3Tuple,
): WaterKeyDirectional | undefined {
  let best: LightArrival | undefined;
  let bestLuminance = 0;
  let bestIsDirectional = false;
  for (const light of records) {
    const arrival = lightArrivalAt(light, receiver_m);
    if (!arrival) continue;
    const luminance = rec709Luminance(arrival.radiance);
    if (!(luminance > bestLuminance)) continue;
    bestLuminance = luminance;
    best = arrival;
    bestIsDirectional = light.kind === "directional";
  }
  // Declining is not the same as answering with the directional, and the
  // difference is a bit. Round-tripping that record through here — normalise
  // its direction, divide its radiance by its own luminance, let
  // `resolveWaterKeyLight` normalise and multiply both back — is the identity
  // in exact arithmetic and a last-place drift in floating point, so a set with
  // no fixture over its water would have had its key move by an ULP for no
  // reason at all. Saying nothing hands the caller back to the authored path
  // untouched, which is what "every sunlit scene is where it was" has to mean
  // if it is to be worth asserting.
  if (bestIsDirectional) return undefined;
  if (!best || !(bestLuminance > 0)) return undefined;
  // Split back into a hue and a strength rather than handing over the radiance
  // as a colour: `resolveWaterKeyLight` guards the intensity and not the
  // channels, and a split that puts all of the magnitude in the guarded field
  // is the only one that leaves that guard meaning anything. Their product is
  // the radiance again, exactly.
  return {
    direction: best.direction,
    colorLinear: best.radiance.map((channel) => channel / bestLuminance) as [number, number, number],
    intensity: bestLuminance,
  };
}
