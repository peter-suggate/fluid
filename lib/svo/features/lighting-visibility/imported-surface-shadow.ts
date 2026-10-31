/** Finest imported surfaces are shared by primary visibility and shadow maps.
 * Every instance is retained here: camera-selected LOD/visibility is never a
 * valid caster list, because an off-camera tree can shade a visible receiver. */
export interface ImportedSurfaceShadowGroup {
  readonly asset: string;
  readonly firstInstance: number;
  readonly instanceCount: number;
}

export function importedSurfaceShadowInstances(
  instances: readonly { readonly asset: string; readonly index: number }[],
  radiusFor: (asset: string) => number,
): { readonly groups: readonly ImportedSurfaceShadowGroup[]; readonly records: Uint32Array<ArrayBuffer> } {
  const byAsset = new Map<string, number[]>();
  for (const instance of instances) {
    if (!Number.isInteger(instance.index) || instance.index < 0 || instance.index > 0xffffffff) throw new RangeError("Invalid native shadow primitive identity");
    let entries = byAsset.get(instance.asset);
    if (!entries) { entries = []; byAsset.set(instance.asset, entries); }
    entries.push(instance.index);
  }
  const records = new Uint32Array(instances.length * 2), floats = new Float32Array(records.buffer);
  const groups: ImportedSurfaceShadowGroup[] = [];
  let offset = 0;
  for (const [asset, indices] of byAsset) {
    const radius = radiusFor(asset);
    if (!Number.isFinite(radius) || radius <= 0) throw new RangeError(`Invalid native shadow radius for ${asset}`);
    groups.push({ asset, firstInstance: offset, instanceCount: indices.length });
    for (const index of indices) { records[offset * 2] = index; floats[offset * 2 + 1] = radius; offset++; }
  }
  return { groups, records };
}

/** Appended to the shadow producer, sharing its exact sun/spot projection and
 * edit rectangle. Level zero is the complete native mesh, with no camera LOD. */
export const importedSurfaceShadowWGSL = /* wgsl */ `
fn rasterNativeCasterVisible(centre:vec3f,radius:f32,layer:u32)->bool{
  let mode=rasterShadowRedraw.minimum.w;
  if(mode==0u){return false;}
  if(layer>=2u){
    let spot=rasterLighting.spots[layer-2u];
    let depth=dot(centre-spot.positionNear.xyz,spot.directionTan.xyz);
    if(depth+radius<spot.positionNear.w||depth-radius>spot.rangeRadiusIndex.x){return false;}
  }else{
    let z=rasterShadowPosition(centre,layer).z;let reach=radius/rasterLighting.depthCell.x;
    if(z+reach<0.0||z-reach>1.0){return false;}
  }
  let bounds=rasterShadowRectangle(centre,vec3f(radius),layer);
  if(any(abs(bounds.xy)>vec2f(1.0)+bounds.zw)){return false;}
  if(!rasterShadowRedrawWhole()){
    let redraw=rasterShadowRectangle(rasterShadowRedrawCentre(),rasterShadowRedrawHalf(),layer);
    if(any(abs(bounds.xy-redraw.xy)>bounds.zw+redraw.zw)){return false;}
  }
  return true;
}
struct RasterNativeShadowVertex { @builtin(position) position:vec4f, @location(0) world:vec3f }
@vertex fn rasterNativeShadowVertex(@location(0) local:vec3f,@location(1) primitive:u32,@location(2) radius:f32)->RasterNativeShadowVertex{
  let record=dryPrimitive(primitive);let centre=svoPrimitiveCenter_m(record);let layer=rasterLighting.control.y;
  if(!rasterNativeCasterVisible(centre,radius,layer)){return RasterNativeShadowVertex(vec4f(2,2,2,1),vec3f(0));}
  let world=centre+svoQuaternionRotate(record.orientation,local);
  if(layer>=2u){return RasterNativeShadowVertex(rasterSpotClip(world,rasterLighting.spots[layer-2u]),world);}
  return RasterNativeShadowVertex(vec4f(rasterShadowPosition(world,layer),1),world);
}
@fragment fn rasterNativeShadowFragment(input:RasterNativeShadowVertex)->@builtin(frag_depth) f32{
  if(importedSurfaceRemoved(input.world,0xffffffffu)){discard;}
  let layer=rasterLighting.control.y;
  if(layer>=2u){let spot=rasterLighting.spots[layer-2u];if(spot.rangeRadiusIndex.w>0.0&&distance(input.world,spot.positionNear.xyz)<=spot.rangeRadiusIndex.y){discard;}}
  return input.position.z;
}
struct RasterNativeCutVertex { @builtin(position) position:vec4f, @location(0) @interpolate(flat) cut:u32, @location(1) world:vec3f }
@vertex fn rasterNativeCutVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) cut:u32)->RasterNativeCutVertex{
  let bounds=importedCutBounds(cut);let layer=rasterLighting.control.y;
  if(!rasterNativeCasterVisible(.5*(bounds[0]+bounds[1]),.5*length(bounds[1]-bounds[0]),layer)){return RasterNativeCutVertex(vec4f(2,2,2,1),cut,vec3f(0));}
  let world=importedCutProxy(vertex,cut);
  if(layer>=2u){return RasterNativeCutVertex(rasterSpotClip(world,rasterLighting.spots[layer-2u]),cut,world);}
  return RasterNativeCutVertex(vec4f(rasterShadowPosition(world,layer),1),cut,world);
}
@fragment fn rasterNativeCutFragment(input:RasterNativeCutVertex)->@builtin(frag_depth) f32{
  let layer=rasterLighting.control.y;
  if(layer>=2u){
    let spot=rasterLighting.spots[layer-2u];let basis=rasterSpotBasis(spot);
    let ray=normalize(input.world-spot.positionNear.xyz);let cosine=dot(ray,basis[2]);
    let start=select(spot.positionNear.w/cosine,max(spot.positionNear.w/cosine,spot.rangeRadiusIndex.y),spot.rangeRadiusIndex.w>0.0);
    let hit=importedCutHit(input.cut,spot.positionNear.xyz,ray,start,spot.rangeRadiusIndex.x/cosine);
    if(hit.t>=DRY_MISS){discard;}let clip=rasterSpotClip(spot.positionNear.xyz+ray*hit.t,spot);return clamp(clip.z/clip.w,0.0,1.0);
  }
  let basis=rasterSunBasis();let centre=rasterMapCenter(layer);
  let origin=input.world+basis[2]*(rasterLighting.depthCell.x*.5-dot(input.world-centre,basis[2]));
  let hit=importedCutHit(input.cut,origin,-basis[2],0.0,rasterLighting.depthCell.x);
  if(hit.t>=DRY_MISS){discard;}return clamp(hit.t/rasterLighting.depthCell.x,0.0,1.0);
}
`;

/** A GPU cache hit suppresses vertex work even before the host has its receipt. */
export const importedSurfaceShadowGateWGSL = /* wgsl */ `
struct Redraw {minimum:vec4u,maximum:vec4u}
@group(0) @binding(0) var<uniform> redraw:Redraw;
@group(0) @binding(1) var<storage,read_write> args:array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3u){
 let at=id.x*6u;if(at+5u>=arrayLength(&args)){return;}
 args[at+1u]=select(0u,args[at+5u],redraw.minimum.w!=0u);
}`;
