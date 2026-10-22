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
@vertex fn rasterNativeShadowVertex(@location(0) local:vec3f,@location(1) primitive:u32,@location(2) radius:f32)->@builtin(position) vec4f{
  let record=dryPrimitive(primitive);let centre=svoPrimitiveCenter_m(record);let layer=rasterLighting.control.y;
  if(!rasterNativeCasterVisible(centre,radius,layer)){return vec4f(2,2,2,1);}
  let world=centre+svoQuaternionRotate(record.orientation,local);
  if(layer>=2u){return rasterSpotClip(world,rasterLighting.spots[layer-2u]);}
  return vec4f(rasterShadowPosition(world,layer),1);
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
