import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../../contracts/svo-light-abi";
import { RASTER_MAX_SPOT_LIGHTS, rasterLightValues, rasterSpotLights, type RasterSpotLight } from "./raster-spot-lights";

export const RASTER_POINT_SHADOW_SIZE = 512;
export const RASTER_MAX_LOCAL_SHADOW_VIEWS = RASTER_MAX_SPOT_LIGHTS + 32 * 6;
export const RASTER_LOCAL_LOOKUP_WORD = 16 + RASTER_MAX_LOCAL_SHADOW_VIEWS * 12;
export const RASTER_LOCAL_LAYOUT_WORD = RASTER_LOCAL_LOOKUP_WORD + 32 * 4;
export const RASTER_LIGHTING_PARAM_BYTES = Math.ceil((RASTER_LOCAL_LAYOUT_WORD + 4) * 4 / 256) * 256;
export const RASTER_CUBE_DIRECTIONS = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]] as const;
export interface RasterLocalShadowView extends RasterSpotLight { readonly pointFace: number }
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
