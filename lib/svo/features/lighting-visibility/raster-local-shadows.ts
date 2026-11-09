import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../../contracts/svo-light-abi";
import { RASTER_MAX_SPOT_LIGHTS, rasterLightValues, rasterSpotLights, type RasterSpotLight } from "./raster-spot-lights";

export const RASTER_POINT_SHADOW_SIZE = 512;
export const RASTER_MAX_LOCAL_SHADOW_VIEWS = RASTER_MAX_SPOT_LIGHTS + 32 * 6;
export const RASTER_LOCAL_LOOKUP_WORD = 16 + RASTER_MAX_LOCAL_SHADOW_VIEWS * 12;
export const RASTER_LOCAL_LAYOUT_WORD = RASTER_LOCAL_LOOKUP_WORD + 32 * 4;
export const RASTER_LIGHTING_PARAM_BYTES = Math.ceil((RASTER_LOCAL_LAYOUT_WORD + 4) * 4 / 256) * 256;
export const RASTER_CUBE_DIRECTIONS = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] as const;
export interface RasterLocalShadowView extends RasterSpotLight {
  readonly pointFace: number;
  /** Cube faces of an area emitter: contact-hardening width (a point light's faces are hard). */
  readonly penumbraRadius?: number;
}
export interface RasterShadowBounds { readonly origin: readonly number[]; readonly extent: readonly number[] }

/** A draw-time proof for removing native/SVO fallback stacks from visibility
 * kernels. Unsupported or newly edited lights keep the generic pipeline. */
export function rasterLocalMapsComplete(records: Uint32Array | undefined, views: readonly RasterLocalShadowView[]): boolean {
  if(!records?.length||records.length%SVO_LIGHT_RECORD_WORDS||records.length>32*SVO_LIGHT_RECORD_WORDS)return false;
  const mapped=new Set(views.map(view=>view.index)),f=rasterLightValues(records);
  for(let index=0;index<records.length/SVO_LIGHT_RECORD_WORDS;index++){
    if(mapped.has(index))continue;
    const at=index*SVO_LIGHT_RECORD_WORDS;
    if(index!==0||records[at+24]!==SVO_LIGHT_KINDS.directional)return false;
    const direction=Array.from(f.subarray(at+4,at+7));
    if(!direction.every(Number.isFinite)||Math.hypot(...direction)<=1e-6)return false;
  }
  return true;
}

/** Shadow coverage is independent of the emitter's radiance cutoff. */
export function rasterShadowFar(position: readonly number[], bounds: RasterShadowBounds): number {
  return Math.max(.01, Math.hypot(...position.map((p, axis) => Math.max(
    Math.abs(bounds.origin[axis]! - p), Math.abs(bounds.origin[axis]! + bounds.extent[axis]! - p)))) * 1.001);
}
/** Widest single perspective view of an area emitter (about 76 degrees off axis). */
export const RASTER_AREA_MAX_TAN = 4;
/** The square-frustum basis the WGSL shadow projection uses (rasterSpotBasis). */
function rasterSpotBasis(forward: readonly number[]): [number[], number[]] {
  const helper = Math.abs(forward[1]!) > .95 ? [1, 0, 0] : [0, 1, 0];
  const cross = (a: readonly number[], b: readonly number[]) => [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!];
  const r = cross(forward, helper), length = Math.hypot(...r), right = r.map(v => v / length);
  return [right, cross(right, forward)];
}
interface RasterAreaEmitter { readonly index: number; readonly position: number[]; readonly radius: number; readonly sphere: boolean; readonly view?: RasterSpotLight }
/**
 * A finite area emitter is mapped from its centre, filtered by the existing
 * contact-hardening lookup over its equivalent disc radius. When the whole
 * domain lies in front of it, one perspective view (`view`) covers it;
 * otherwise (an emitter hanging inside the room) it takes a soft cube.
 */
function rasterAreaEmitter(records: Uint32Array, f: Float32Array, at: number, bounds: RasterShadowBounds): RasterAreaEmitter | undefined {
  const kind = records[at + 24];
  if (kind !== SVO_LIGHT_KINDS.rectangleArea && kind !== SVO_LIGHT_KINDS.sphereArea) return undefined;
  for (let i = 0; i < 7; i++) if (!Number.isFinite(f[at + i])) return undefined;
  if (f[at + 3]! < 0) return undefined;
  const position = Array.from(f.subarray(at, at + 3)), sphere = kind === SVO_LIGHT_KINDS.sphereArea;
  let direction: number[], radius: number;
  if (!sphere) {
    const halfWidth = f[at + 15]!, halfHeight = f[at + 19]!;
    if (!(halfWidth >= 0) || !(halfHeight >= 0) || !Number.isFinite(halfWidth * halfHeight)) return undefined;
    direction = Array.from(f.subarray(at + 4, at + 7));
    radius = Math.sqrt(4 * halfWidth * halfHeight / Math.PI);
  } else {
    radius = f[at + 20]!;
    if (!(radius >= 0) || !Number.isFinite(radius)) return undefined;
    direction = bounds.origin.map((v, axis) => v + bounds.extent[axis]! * .5 - position[axis]!);
  }
  const emitter = { index: at / SVO_LIGHT_RECORD_WORDS, position, radius, sphere };
  const length = Math.hypot(...direction);
  if (!(length >= 1e-6)) return emitter;
  direction = direction.map(v => v / length);
  const far = rasterShadowFar(position, bounds), [right, up] = rasterSpotBasis(direction);
  // Near stays inside the emitter's own extent without costing depth precision.
  const near = Math.max(.001, Math.min(radius * 1.01, far * .01));
  let widest = 0;
  for (let corner = 0; corner < 8; corner++) {
    const delta = bounds.origin.map((v, axis) => v + ((corner >> axis) & 1) * bounds.extent[axis]! - position[axis]!);
    const dot = (axis: readonly number[]) => delta.reduce((sum, v, i) => sum + v * axis[i]!, 0);
    const axial = dot(direction);
    if (!(axial > near)) return emitter;
    widest = Math.max(widest, Math.max(Math.abs(dot(right)), Math.abs(dot(up))) / axial);
  }
  if (widest > RASTER_AREA_MAX_TAN || !(far > near)) return emitter;
  // The square frustum is convex: all eight corners inside cover the domain.
  return { ...emitter, view: { index: emitter.index, position, direction, near, far, radius, tanHalfAngle: Math.max(1e-3, widest * 1.001) } };
}
export function rasterLocalShadowViews(records: Uint32Array | undefined, bounds: RasterShadowBounds): RasterLocalShadowView[] {
  if (!records || records.length % SVO_LIGHT_RECORD_WORDS) return [];
  // Resolve projection bounds in a copy; never change the live light's energy,
  // falloff or range. Existing finite-spot eligibility remains authoritative.
  const bounded = records.slice(), b = new Float32Array(bounded.buffer), f = rasterLightValues(records);
  for (let at = 0; at < records.length; at += SVO_LIGHT_RECORD_WORDS) {
    if (records[at + 24] === SVO_LIGHT_KINDS.spot && f[at + 3] === 0) {
      b[at + 3] = rasterShadowFar(Array.from(f.subarray(at, at + 3)), bounds);
    }
  }
  const result: RasterLocalShadowView[] = rasterSpotLights(bounded).map(spot => ({ ...spot, pointFace: -1 }));
  // Spot-layer views precede point faces: the atlas allots one layer per spot.
  // Area emitters past the spot-layer budget take a cube like the rest.
  const cubes: RasterAreaEmitter[] = [];
  for (let at = 0; at < Math.min(records.length, 32 * SVO_LIGHT_RECORD_WORDS); at += SVO_LIGHT_RECORD_WORDS) {
    const emitter = rasterAreaEmitter(records, f, at, bounds);
    if (emitter?.view && result.length < RASTER_MAX_SPOT_LIGHTS) result.push({ ...emitter.view, pointFace: -1 });
    else if (emitter) cubes.push(emitter);
  }
  let face = 0;
  for (let at = 0; at < Math.min(records.length, 32 * SVO_LIGHT_RECORD_WORDS); at += SVO_LIGHT_RECORD_WORDS) {
    if (records[at + 24] !== SVO_LIGHT_KINDS.point) continue;
    const position = Array.from(f.subarray(at, at + 3)), radius = f[at + 20]!;
    if (!position.every(Number.isFinite) || !Number.isFinite(radius) || radius < 0 || !Number.isFinite(f[at + 3]) || f[at + 3]! < 0) continue;
    const far = rasterShadowFar(position, bounds);
    const near = Math.max(.001, radius / Math.sqrt(3) * .99);
    if (far <= near) continue;
    for (const direction of RASTER_CUBE_DIRECTIONS) result.push({ index: at / SVO_LIGHT_RECORD_WORDS,
      position, radius, direction: [...direction], near, far, tanHalfAngle: 1, pointFace: face++ });
  }
  for (const { index, position, radius, sphere } of cubes) {
    const far = rasterShadowFar(position, bounds);
    // `radius` is the caster clearance about the centre (a sphere's own
    // surface); the PCSS width is the emitter's equivalent disc radius.
    const clearance = sphere ? radius * 1.01 : Math.max(.001, Math.min(radius * 1.01, far * .01));
    const near = Math.max(.001, clearance / Math.sqrt(3) * .99);
    if (!(far > near)) continue;
    for (const direction of RASTER_CUBE_DIRECTIONS) result.push({ index, position, radius: clearance,
      direction: [...direction], near, far, tanHalfAngle: 1, pointFace: face++, penumbraRadius: radius });
  }
  return result;
}
export function rasterShadowTile(view: RasterLocalShadowView | undefined, virtualLayer: number, spots: number, size: number) {
  if (!view || view.pointFace < 0) return { layer: virtualLayer, x: 0, y: 0, size };
  const across = size / RASTER_POINT_SHADOW_SIZE, face = view.pointFace;
  return { layer: 2 + spots + Math.floor(face / (across * across)),
    x: face % across * RASTER_POINT_SHADOW_SIZE,
    y: Math.floor(face / across) % across * RASTER_POINT_SHADOW_SIZE, size: RASTER_POINT_SHADOW_SIZE };
}

export const rasterShadowAtlasWGSL = /* wgsl */ `
fn rasterMapResolution(layer:u32)->f32{
  if(layer>=2u&&rasterLighting.spots[layer-2u].rangeRadiusIndex.w>0.0){return f32(rasterLighting.localLayout.z);}
  return f32(max(rasterLighting.control.w>>16u,2048u));
}
fn rasterAtlasTile(layer:u32)->vec3u{
  if(layer<2u){return vec3u(0,0,layer);}
  let encoded=u32(rasterLighting.spots[layer-2u].rangeRadiusIndex.w);
  if(encoded==0u){return vec3u(0,0,layer);}
  let face=encoded-1u;let across=rasterLighting.localLayout.y;let size=rasterLighting.localLayout.z;
  return vec3u((face%across)*size,((face/across)%across)*size,2u+rasterLighting.localLayout.x+face/(across*across));
}
fn rasterSpotSlot(lightIndex:u32,position:vec3f)->i32{
  if(lightIndex>=32u){return -1;}
  let lookup=rasterLighting.lightSlots[lightIndex];if(lookup.x==0u){return -1;}
  let base=lookup.x-1u;if(lookup.y==1u){return i32(base);}
  let delta=position-rasterLighting.spots[base].positionNear.xyz;let magnitude=abs(delta);
  var axis=0u;if(magnitude.y>magnitude.x){axis=1u;}if(magnitude.z>magnitude[axis]){axis=2u;}
  return i32(base+axis*2u+select(1u,0u,delta[axis]>=0.0));
}
`;
