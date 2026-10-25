/** Normalize local geometric variation against the current detailed receivers.
 * Only the next sample fades in: intermediate budgets also skip real work. */
export const rasterHorizonBudgetWGSL = /* wgsl */ `
fn rasterHorizonProjectedRadius(radius:f32,viewDepth:f32,height:f32,tanHalfFov:f32)->f32{
  return radius*height/(2.0*max(viewDepth,.001)*tanHalfFov);
}
fn rasterHorizonRange(encoded:vec2u)->vec2f{
  if(encoded.y==0u){return vec2f(0);}
  return vec2f(bitcast<f32>(~encoded.x),bitcast<f32>(encoded.y));
}
fn rasterHorizonBudget(projected:f32,range:vec2f)->f32{
  // Coplanar regions use the minimum independently of depth. End plateaus
  // make both slider endpoints attainable despite floating-point roundoff.
  if(projected<=0.0||range.y-range.x<=max(range.y*1e-5,1e-6)){return 0.0;}
  return smoothstep(.1,.9,(projected-range.x)/(range.y-range.x));
}
fn rasterHorizonTargetTaps(budget:f32,farTaps:u32)->f32{
  // Avoid ghost taps from float roundoff at an integer contact floor.
  return floor(mix(f32(farTaps),36.0,budget)*1024.0+.5)/1024.0;
}
fn rasterHorizonTapCount(budget:f32,farTaps:u32)->u32{
  return u32(ceil(rasterHorizonTargetTaps(budget,farTaps)));
}
fn rasterHorizonStepWeight(direction:u32,step:u32,budget:f32,farTaps:u32)->f32{
  // Spend taps from the inside out, alternating opposite directions. At a
  // minimum of 24 taps this retains all four inner samples in every direction.
  let order=(step-1u)*6u+(direction%3u)*2u+direction/3u;
  return clamp(rasterHorizonTargetTaps(budget,farTaps)-f32(order),0.0,1.0);
}
`;

/** Reuse a 16x16 group's geometry in local quads to estimate normal/plane
 * variation without extra texture reads. Flat background gets score zero and
 * does not set the detailed receivers' range. Positive floats sort by bits. */
export const rasterHorizonRangeProducerWGSL = /* wgsl */ `
@group(2) @binding(22) var<storage,read_write> rasterAoRangeOutput:array<atomic<u32>>;
@group(2) @binding(24) var rasterAoImportanceOutput:texture_storage_2d<r32float,write>;
var<workgroup> rasterAoRangeMinimum:atomic<u32>;
var<workgroup> rasterAoRangeMaximum:atomic<u32>;
var<workgroup> rasterAoPositions:array<vec4f,256>;
var<workgroup> rasterAoNormals:array<vec3f,256>;
@compute @workgroup_size(16,16) fn rasterAoRangeMain(
  @builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&rasterAoRangeMinimum,0u);atomicStore(&rasterAoRangeMaximum,0u);}
  let dims=vec2i(textureDimensions(drySplitGeometryRead));
  let stride=i32(rasterVisibilityStride());
  let inBounds=all(vec2i(id.xy)<(dims+stride-1)/stride);
  let coord=min(vec2i(id.xy)*stride+vec2i(stride/2),dims-1);
  var g=vec4f(0);var valid=false;
  if(inBounds){g=drySplitGeometryAt(coord);valid=g.w>0.0&&g.w<DRY_MISS&&dot(g.xyz,g.xyz)>=.1;}
  var position=vec3f(0);var normal=vec3f(0);
  if(valid){position=rasterWorldAt(vec2f(coord)+.5,g.w);normal=normalize(g.xyz);}
  rasterAoPositions[lane]=vec4f(position,select(0.0,1.0,valid));
  rasterAoNormals[lane]=normal;
  workgroupBarrier();
  var importance=0.0;
  if(valid){
    let radius=dryContactVisibilityRadius();var variation=0.0;
    // The other three receivers in this local quad. Every lane already paid
    // for its geometry read; no neighbourhood texture fetches are added.
    let neighbours=array<u32,3>(lane^1u,lane^16u,lane^17u);
    for(var i=0u;i<3u;i++){
      let other=rasterAoPositions[neighbours[i]];
      if(other.w==0.0){continue;}
      let plane=abs(dot(normal,other.xyz-position))/max(radius,1e-6);
      let bend=1.0-clamp(dot(normal,rasterAoNormals[neighbours[i]]),-1.0,1.0);
      variation=max(variation,max(bend,plane*4.0));
    }
    // Suppress reconstruction roundoff on large planes. Discontinuities and
    // curved receivers retain graded importance, bounded against outliers.
    variation=clamp((variation-.01)/.99,0.0,1.0);
    let ndc=(vec2f(coord)+.5)/uniforms.viewport.xy*2.0-1.0;
    let ray=vec2f(ndc.x*uniforms.viewport.x/uniforms.viewport.y,ndc.y)*cameraTanHalfFov();
    let depth=g.w*inverseSqrt(1.0+dot(ray,ray));
    importance=variation*rasterHorizonProjectedRadius(radius,depth,uniforms.viewport.y,cameraTanHalfFov());
    if(importance>0.0){
      let bits=bitcast<u32>(importance);
      atomicMax(&rasterAoRangeMinimum,~bits);atomicMax(&rasterAoRangeMaximum,bits);
    }
  }
  if(inBounds){textureStore(rasterAoImportanceOutput,vec2i(id.xy),vec4f(importance));}
  workgroupBarrier();
  if(lane==0u){
    atomicMax(&rasterAoRangeOutput[0],atomicLoad(&rasterAoRangeMinimum));
    atomicMax(&rasterAoRangeOutput[1],atomicLoad(&rasterAoRangeMaximum));
  }
}
`;

/** Low minima need neighbourhood protection: a flat receiver can have a
 * nearby blocker. Four max-mip reads conservatively cover its AO footprint. */
export const rasterHorizonImportanceWGSL = /* wgsl */ `
fn rasterHorizonImportanceAt(coordinate:vec2f,radius:f32,minimum:u32)->f32{
  if(minimum>=24u||radius<=0.0){return textureLoad(rasterAoImportance,vec2i(coordinate),0).x;}
  // One full-resolution pixel also covers the horizon tap rounding.
  let support=radius+.5;
  let level=min(u32(ceil(log2(max(1.0,2.0*support)))),textureNumLevels(rasterAoImportance)-1u);
  let scale=exp2(f32(level));let dims=vec2i(textureDimensions(rasterAoImportance,level));
  let lo=clamp(vec2i(floor((coordinate-support)/scale)),vec2i(0),dims-1);
  let hi=clamp(vec2i(floor((coordinate+support)/scale)),vec2i(0),dims-1);
  return max(max(textureLoad(rasterAoImportance,lo,i32(level)).x,textureLoad(rasterAoImportance,hi,i32(level)).x),
    max(textureLoad(rasterAoImportance,vec2i(lo.x,hi.y),i32(level)).x,textureLoad(rasterAoImportance,vec2i(hi.x,lo.y),i32(level)).x));
}
// A contact floor independent of the slider preserves monotonic tap counts:
// raising the minimum cannot remove work when footprint protection switches off.
fn rasterHorizonReceiverBudget(coordinate:vec2f,radius:f32,minimum:u32,range:vec2f)->f32{
  let raw=textureLoad(rasterAoImportance,vec2i(coordinate),0).x;
  let budget=rasterHorizonBudget(raw,range);
  // Keep the minimum positive-importance receivers as anchors, so even a
  // view filled with detail reaches the requested slider count.
  if(minimum>=24u||(raw>0.0&&raw<=range.x)){return budget;}
  let nearby=rasterHorizonImportanceAt(coordinate,radius,minimum);
  let protection=smoothstep(0.0,max(range.y*.05,1e-6),nearby)*24.0;
  let taps=max(mix(f32(minimum),36.0,budget),protection);
  return clamp((taps-f32(minimum))/(36.0-f32(minimum)),0.0,1.0);
}
`;

/** Preserve the final odd row/column when conventional mip extents round down. */
export const rasterAoImportanceMipWGSL = /* wgsl */ `
@group(0) @binding(0) var source:texture_2d<f32>;
@group(0) @binding(1) var destination:texture_storage_2d<r32float,write>;
@compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3u){
  let dims=textureDimensions(destination);if(any(id.xy>=dims)){return;}
  let sourceDims=textureDimensions(source);let start=id.xy*2u;
  let end=select(min(start+2u,sourceDims),sourceDims,id.xy==dims-1u);
  var value=0.0;
  for(var y=start.y;y<end.y;y++){for(var x=start.x;x<end.x;x++){
    value=max(value,textureLoad(source,vec2i(vec2u(x,y)),0).x);
  }}
  textureStore(destination,vec2i(id.xy),vec4f(value));
}
`;
