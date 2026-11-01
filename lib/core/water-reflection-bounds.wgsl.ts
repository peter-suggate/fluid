/** Conservative radial-depth intervals for 8x8 dry-scene tiles. Empty tiles
 * have an inverted interval. Nothing is downsampled in the reflection itself. */
export const waterReflectionBoundsWGSL = /* wgsl */ `
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var bounds:texture_storage_2d<rg32float,write>;
var<workgroup> minimum:array<f32,64>;
var<workgroup> maximum:array<f32,64>;
@compute @workgroup_size(8,8) fn main(@builtin(workgroup_id) tile:vec3u,@builtin(local_invocation_id) local:vec3u,@builtin(local_invocation_index) lane:u32){
  let pixel=tile.xy*8u+local.xy;
  var depth=65504.0;
  if(all(pixel<textureDimensions(source))){depth=textureLoad(source,vec2i(pixel),0).a;}
  let valid=depth>0.0&&depth<60000.0;
  minimum[lane]=select(65504.0,depth,valid);maximum[lane]=select(0.0,depth,valid);
  workgroupBarrier();
  for(var stride=32u;stride>0u;stride>>=1u){
    if(lane<stride){minimum[lane]=min(minimum[lane],minimum[lane+stride]);maximum[lane]=max(maximum[lane],maximum[lane+stride]);}
    workgroupBarrier();
  }
  if(lane==0u){textureStore(bounds,vec2i(tile.xy),vec4f(minimum[0],maximum[0],0,0));}
}
`;
