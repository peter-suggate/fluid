import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../../contracts/svo-light-abi";

/** The existing shader budget counts emitter samples, not texture layers per
 * authored light. Allocate only the samples that this publication can consume. */
export function rasterVisibilityLayers(records: Uint32Array | undefined, budget: number, areaSamples: number, gi: boolean): number {
  let samples = 0;
  for (let i = 0; records && i < Math.min(budget, records.length / SVO_LIGHT_RECORD_WORDS); i++) {
    const kind = records[i * SVO_LIGHT_RECORD_WORDS + 24];
    samples += !gi && (kind === SVO_LIGHT_KINDS.spot || kind === SVO_LIGHT_KINDS.sphereArea || kind === SVO_LIGHT_KINDS.rectangleArea) ? areaSamples : 1;
  }
  return Math.max(1, Math.min(budget, samples));
}

export const rasterVisibilityTilesWGSL = /* wgsl */ `
fn rasterVisibilityStride()->u32{return 2u;}
@group(2) @binding(16) var<storage,read> rasterTileRequests:array<u32>;
@group(2) @binding(17) var<storage,read_write> rasterTileRequestsOutput:array<atomic<u32>>;
var<workgroup> rasterTileOccupied:atomic<u32>;
// One request per 16x16 full-resolution tile. Test all four pixels so a thin
// receiver absent from the reduced lattice still reaches the exact edge pass.
@compute @workgroup_size(8,8) fn rasterClassifyTiles(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&rasterTileOccupied,0u);}workgroupBarrier();
  let dims=textureDimensions(drySplitGeometryRead);let stride=rasterVisibilityStride();let origin=(group.xy*8u+local.xy)*stride;
  for(var y=0u;y<stride;y++){for(var x=0u;x<stride;x++){
    let c=origin+vec2u(x,y);if(any(c>=dims)){continue;}
    let g=rasterVisibilityGeometryAt(vec2i(c));
    if(g.w<DRY_MISS&&dot(g.xyz,g.xyz)>=.1){atomicStore(&rasterTileOccupied,1u);}
  }}
  workgroupBarrier();
  if(lane==0u&&atomicLoad(&rasterTileOccupied)!=0u){
    let index=atomicAdd(&rasterTileRequestsOutput[0],1u);
    atomicStore(&rasterTileRequestsOutput[4u+index],group.y*((dims.x+8u*stride-1u)/(8u*stride))+group.x);
  }
}
@compute @workgroup_size(1) fn rasterFinalizeTiles(){
  let count=atomicLoad(&rasterTileRequestsOutput[0]);
  atomicStore(&rasterTileRequestsOutput[1],min(count,65535u));
  atomicStore(&rasterTileRequestsOutput[2],max(1u,(count+65534u)/65535u));
  atomicStore(&rasterTileRequestsOutput[3],1u);
}
fn rasterQueuedCoordinate(group:vec3u,local:vec3u)->vec2u{
  if((rasterLighting.control.w&2u)==0u){return group.xy*8u+local.xy;}
  let index=group.x+group.y*65535u;
  if(index>=rasterTileRequests[0]){return vec2u(0xffffffffu);}
  let tile=rasterTileRequests[4u+index];let columns=(textureDimensions(drySplitGeometryRead).x+8u*rasterVisibilityStride()-1u)/(8u*rasterVisibilityStride());
  return vec2u(tile%columns,tile/columns)*8u+local.xy;
}
`;

export const rasterVisibilityPackingWGSL = /* wgsl */ `
fn rasterPackVisibility(value:vec3f)->u32{
  let v=vec3u(round(clamp(value,vec3f(0),vec3f(1))*1023.0));
  return v.x|(v.y<<10u)|(v.z<<20u);
}
fn rasterUnpackVisibility(value:u32)->vec3f{
  return vec3f(vec3u(value&1023u,(value>>10u)&1023u,(value>>20u)&1023u))/1023.0;
}
`;

/** Count and indirect dispatch share one buffer; the dispatch is copied out
 * before its INDIRECT usage scope. The two-dimensional dispatch covers Retina
 * frames even when every pixel needs exact visibility. */
export const rasterExactQueueWGSL = /* wgsl */ `
@group(2) @binding(20) var<storage,read_write> rasterExactRequests:array<atomic<u32>>;
@compute @workgroup_size(1) fn rasterExactFinalize(){
  let groups=(atomicLoad(&rasterExactRequests[0])+63u)/64u;
  atomicStore(&rasterExactRequests[1],min(groups,65535u));
  atomicStore(&rasterExactRequests[2],max(1u,(groups+65534u)/65535u));
  atomicStore(&rasterExactRequests[3],1u);
}
`;

/** Bijective atlas permutation: representative samples occupy a compact
 * rectangle; exact receiver samples occupy the remaining three rectangles. */
export const rasterAtlasLayoutWGSL = /* wgsl */ `
fn rasterCompactAtlasCoordinate(c:vec2i,dims:vec2i)->vec2i{
  let reduced=(dims+1)/2;let cell=c/2;
  let representative=min(cell*2+1,dims-1);
  return select(reduced+cell,cell,c==representative);
}
`;

/** One current-frame visibility atlas. Odd/odd pixels hold reduced samples;
 * otherwise unrepresented receivers occupy their own pixel. The edge producer
 * never overwrites a representative, including clamped odd-sized viewports.
 * RGB10 preserves coloured transmission with <= 1/2046 quantization error. */
export const rasterLightVisibilityWGSL = /* wgsl */ `
@group(2) @binding(14) var rasterLightVisibility:texture_2d_array<u32>;
${rasterExactQueueWGSL}
@group(2) @binding(18) var rasterGuideOutput:texture_storage_2d<rg32uint,write>;
@group(2) @binding(19) var rasterGuide:texture_2d<u32>;
@group(2) @binding(15) var rasterLightVisibilityOutput:texture_storage_2d_array<r32uint,write>;
// Use the same optional silhouette receiver as full-resolution material shading.
fn rasterVisibilityGeometryAt(c:vec2i)->vec4f{
  if((dry.materialPublication.w&128u)!=0u){let seam=dryPrimarySeamSample(c);if(seam.valid!=0u){return seam.geometry;}}
  return drySplitGeometryAt(c);
}
fn rasterVisibilityIdentityAt(c:vec2i)->vec2u{
  if((dry.materialPublication.w&128u)!=0u){let seam=dryPrimarySeamSample(c);if(seam.valid!=0u){return seam.identity;}}
  return drySplitIdentityAt(c).xy;
}
${rasterVisibilityTilesWGSL}
var<private> rasterVisibilityWeights:vec4f;
var<private> rasterVisibilityCoordinates:array<vec2i,4>;
var<private> rasterVisibilityWeight:f32;
var<private> rasterCurrentSample:u32;
${rasterVisibilityPackingWGSL}
${rasterAtlasLayoutWGSL}
fn rasterAtlasCoordinate(c:vec2i)->vec2i{
  return rasterCompactAtlasCoordinate(c,vec2i(textureDimensions(drySplitGeometryRead)));
}
fn rasterRepresentative(c:vec2i)->vec2i{
  let stride=i32(rasterVisibilityStride());return min(c*stride+vec2i(stride/2),vec2i(textureDimensions(drySplitGeometryRead))-1);
}
// Shared by the edge producer and material consumer. A zero-weight receiver
// is evaluated exactly in compute, never silently replaced by unshadowed light.
fn rasterPrepareVisibility(position:vec3f,normal:vec3f)->vec2f{
  let depth=length(position-uniforms.cameraPosition.xyz);
  let stride=f32(rasterVisibilityStride());let q=(rasterPixel-(floor(stride*.5)+.5))/stride;let base=vec2i(floor(q));let f=fract(q);
  let dims=vec2i(textureDimensions(rasterContact));
  let identity=rasterVisibilityIdentityAt(vec2i(rasterPixel));
  rasterVisibilityWeights=vec4f(0);rasterVisibilityWeight=0.0;
  var total=vec2f(0);
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){
    let index=u32(y*2+x);
    let bilinear=select(1.0-f.x,f.x,x==1)*select(1.0-f.y,f.y,y==1);if(bilinear==0.0){continue;}
    let c=clamp(base+vec2i(x,y),vec2i(0),dims-1);
    let representative=rasterRepresentative(c);rasterVisibilityCoordinates[index]=representative;
    // Identity includes owner, feature and motion class, preventing reuse
    // across touching surfaces whose depth and normal happen to agree.
    let representativeIdentity=drySplitIdentityAt(representative).xy;
    if(representativeIdentity.x==0u||any(representativeIdentity!=identity)){continue;}
    let v=textureLoad(rasterContact,c,0);
    let tolerance=max(depth*.005,rasterLighting.depthCell.y*.5);
    let w=bilinear*exp(-abs(v.y-depth)/max(tolerance,1e-4))*pow(max(dot(normal,rasterOctDecode(v.zw)),0.0),8.0);
    rasterVisibilityWeights[index]=w;rasterVisibilityWeight+=w;
    total+=vec2f(v.x,textureLoad(rasterShadowVisibility,c,0).x)*w;
  }}
  if(rasterVisibilityWeight>1e-4){return total/rasterVisibilityWeight;}
  return vec2f(1);
}
fn rasterCachedLightVisibility(sample:u32)->vec3f{
  if(rasterVisibilityWeight<=1e-4){return rasterUnpackVisibility(textureLoad(rasterLightVisibility,rasterAtlasCoordinate(vec2i(rasterPixel)),i32(sample),0).x);}
  var value=vec3f(0);
  for(var i=0u;i<4u;i++){
    if(rasterVisibilityWeights[i]>0.0){value+=rasterUnpackVisibility(textureLoad(rasterLightVisibility,rasterAtlasCoordinate(rasterVisibilityCoordinates[i]),i32(sample),0).x)*rasterVisibilityWeights[i];}
  }
  return value/rasterVisibilityWeight;
}
fn rasterVisibilityHit(coord:vec2i,g:vec4f)->DryHit{
  let identity=rasterVisibilityIdentityAt(coord);let metadata=identity.y;
  return DryHit(g.w,normalize(g.xyz),identity.x,dryOpaqueOwner(metadata),(metadata>>16u)&15u,(metadata>>20u)&15u,(metadata>>24u)&3u,(metadata>>26u)&1u,0.0,vec3u(0u,metadata,0u));
}
fn rasterProduceLightVisibility(coord:vec2i,g:vec4f,sunVisibility:f32){
  let hit=rasterVisibilityHit(coord,g);let position=rasterWorldAt(vec2f(coord)+.5,g.w);
  let geometricNormal=dryGeometricNormal(hit);
  dryVisibilityIgnoredBody=DRY_OWNER_NONE;
  drySampledReceiver=drySampledMaterial(hit.materialId);
  rasterResolvedVisibility=vec2f(1,sunVisibility);
  var sampleBudget=0u;
  let lightCount=min(dryLighting.metadata.x,min(dry.tuningCounts0.z,32u));
  let gi=(dry.materialPublication.w&16u)!=0u;
  for(var lightIndex=0u;lightIndex<8u;lightIndex++){
    if(lightIndex>=lightCount||sampleBudget>=dry.tuningCounts0.z){break;}
    let light=dryLighting.lights[lightIndex];if(light.identity.w!=dryLighting.metadata.y){continue;}
    let area=light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_RECTANGLE_AREA||light.identity.x==SVO_LIGHT_SPOT;
    let sampleCount=select(select(1u,max(dry.tuningCounts1.x,dry.tuningCounts0.w),area),1u,gi);
    // Cached spot/area maps use the receiver plane, not the emitter sample.
    // Body rays and liquid attenuation do depend on its direction, so those
    // variants keep evaluating every sample independently.
    let sharedMap=!RASTER_VISIBILITY_WATER_SHADOWS&&uniforms.options.z<.5&&rasterSpotSlot(lightIndex)>=0;
    var mapReady=false;var mapVisibility=vec3f(1);
    for(var sampleIndex=0u;sampleIndex<2u;sampleIndex++){
      if(sampleIndex>=sampleCount||sampleBudget>=dry.tuningCounts0.z){break;}
      let sample=dryLightSample(light,sampleIndex,position);var visibility=vec3f(1);
      // Range, emitter facing, zero radiance and receiver backfaces all reject
      // before a shadow-map lookup or SVO ray. Neutral visibility at a sample
      // outside a beam does not invent a dark fringe inside its zero-energy edge.
      if(sample.valid!=0u&&dot(hit.normal,sample.towardLight)>0.0){
        if(sharedMap&&mapReady&&dot(geometricNormal,sample.towardLight)>0.0){visibility=mapVisibility;}
        else{
          visibility=rasterEvaluateLightVisibility(position,geometricNormal,hit.ownerId,sample.towardLight,sample.finiteDistance_m,lightIndex);
          if(sharedMap&&dot(geometricNormal,sample.towardLight)>0.0){mapReady=true;mapVisibility=visibility;}
        }
      }
      textureStore(rasterLightVisibilityOutput,rasterAtlasCoordinate(coord),i32(sampleBudget),vec4u(rasterPackVisibility(visibility)));
      sampleBudget++;
    }
  }
}
fn rasterLoadVisibilityGuide()->vec2f{
  let guide=textureLoad(rasterGuide,vec2i(rasterPixel),0).xy;
  rasterVisibilityWeights=unpack4x8unorm(guide.y);
  rasterVisibilityWeight=dot(rasterVisibilityWeights,vec4f(1));
  let stride=f32(rasterVisibilityStride());let base=vec2i(floor((rasterPixel-(floor(stride*.5)+.5))/stride));let dims=vec2i(textureDimensions(rasterContact));
  rasterVisibilityCoordinates[0]=rasterRepresentative(clamp(base,vec2i(0),dims-1));
  rasterVisibilityCoordinates[1]=rasterRepresentative(clamp(base+vec2i(1,0),vec2i(0),dims-1));
  rasterVisibilityCoordinates[2]=rasterRepresentative(clamp(base+vec2i(0,1),vec2i(0),dims-1));
  rasterVisibilityCoordinates[3]=rasterRepresentative(clamp(base+vec2i(1,1),vec2i(0),dims-1));
  return unpack2x16float(guide.x);
}
@compute @workgroup_size(8,8) fn rasterLightsMain(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u){
  let id=rasterQueuedCoordinate(group,local);if(any(id>=textureDimensions(rasterContact))){return;}
  let coord=rasterRepresentative(vec2i(id));let g=rasterVisibilityGeometryAt(coord);
  if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){return;}
  var shadow=textureLoad(rasterShadowVisibility,vec2i(id),0).x;
  if((dry.materialPublication.w&128u)!=0u&&any(g!=drySplitGeometryAt(coord))){shadow=rasterSunVisibility(rasterWorldAt(vec2f(coord)+.5,g.w),dryGeometricNormal(rasterVisibilityHit(coord,g)));}
  rasterProduceLightVisibility(coord,g,shadow);
}
fn rasterLightEdgeAt(coord:vec2i){
  let dims=vec2i(textureDimensions(drySplitGeometryRead));if(any(coord>=dims)){return;}
  let g=rasterVisibilityGeometryAt(coord);if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){return;}
  // An ordinary lattice receiver reconstructs solely from itself. Preserve
  // that exact result without identity/depth/normal bilateral arithmetic.
  // Clamped even border pixels and seam receivers retain the general path.
  if(all((coord&vec2i(1))==vec2i(1))&&(dry.materialPublication.w&128u)==0u&&drySplitIdentityAt(coord).x!=0u){
    let c=coord/2;let resolved=vec2f(textureLoad(rasterContact,c,0).x,textureLoad(rasterShadowVisibility,c,0).x);
    textureStore(rasterGuideOutput,coord,vec4u(pack2x16float(resolved),255u,0u,0u));return;
  }
  rasterPixel=vec2f(coord)+.5;
  let position=rasterWorldAt(rasterPixel,g.w);let normal=normalize(g.xyz);
  var resolved=rasterPrepareVisibility(position,normal);var weights=0u;
  if(rasterVisibilityWeight>1e-4){weights=pack4x8unorm(rasterVisibilityWeights/rasterVisibilityWeight);}
  else{
    let index=atomicAdd(&rasterExactRequests[0],1u);
    atomicStore(&rasterExactRequests[4u+index],u32(coord.y)*u32(dims.x)+u32(coord.x));
    return;
  }
  textureStore(rasterGuideOutput,coord,vec4u(pack2x16float(resolved),weights,0u,0u));
}
fn rasterInlineEdgeAt(coord:vec2i){
  let dims=vec2i(textureDimensions(drySplitGeometryRead));if(any(coord>=dims)){return;}
  // Never overwrite the reduced lattice, even at an odd viewport boundary.
  let representative=all(coord==rasterRepresentative(coord/i32(rasterVisibilityStride())));
  let g=rasterVisibilityGeometryAt(coord);if(!(g.w<DRY_MISS)||dot(g.xyz,g.xyz)<.1){return;}
  rasterPixel=vec2f(coord)+.5;
  let position=rasterWorldAt(rasterPixel,g.w);let normal=normalize(g.xyz);
  var resolved=rasterPrepareVisibility(position,normal);var weights=0u;
  if(rasterVisibilityWeight>1e-4){weights=pack4x8unorm(rasterVisibilityWeights/rasterVisibilityWeight);}
  else{
    if((dry.materialPublication.w&1u)!=0u){resolved.x=min(rasterHorizonVisibility(rasterPixel,position,normal),rasterCoarseVisibility(position,normal));}
    if((dry.materialPublication.w&2u)!=0u){
      resolved.y=rasterSunVisibility(position,dryGeometricNormal(rasterVisibilityHit(coord,g)));
      if(!representative){rasterProduceLightVisibility(coord,g,resolved.y);}
    }
  }
  textureStore(rasterGuideOutput,coord,vec4u(pack2x16float(resolved),weights,0u,0u));
}
@compute @workgroup_size(8,8) fn rasterInlineEdgesMain(@builtin(global_invocation_id) id:vec3u){
  let origin=vec2i(id.xy)*2;
  for(var y=0;y<2;y++){for(var x=0;x<2;x++){rasterInlineEdgeAt(origin+vec2i(x,y));}}
}
// Exact traversal lives in a separate entry point: the common full-screen
// resolve must not carry its private stacks/register pressure on every lane.
@compute @workgroup_size(64) fn rasterExactEdgesMain(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_index) lane:u32){
  let index=(group.x+group.y*65535u)*64u+lane;
  if(index>=atomicLoad(&rasterExactRequests[0])){return;}
  let dims=textureDimensions(drySplitGeometryRead);let address=atomicLoad(&rasterExactRequests[4u+index]);
  let coord=vec2i(i32(address%dims.x),i32(address/dims.x));
  let g=rasterVisibilityGeometryAt(coord);rasterPixel=vec2f(coord)+.5;
  let position=rasterWorldAt(rasterPixel,g.w);let normal=normalize(g.xyz);
  var resolved=vec2f(1);
  if((dry.materialPublication.w&1u)!=0u){resolved.x=min(rasterHorizonVisibility(rasterPixel,position,normal),rasterCoarseVisibility(position,normal));}
  if((dry.materialPublication.w&2u)!=0u){
    resolved.y=rasterSunVisibility(position,dryGeometricNormal(rasterVisibilityHit(coord,g)));
    if(!all(coord==rasterRepresentative(coord/i32(rasterVisibilityStride())))){rasterProduceLightVisibility(coord,g,resolved.y);}
  }
  textureStore(rasterGuideOutput,coord,vec4u(pack2x16float(resolved),0u,0u,0u));
}
@compute @workgroup_size(8,8) fn rasterLightEdgesMain(@builtin(workgroup_id) group:vec3u,@builtin(local_invocation_id) local:vec3u){
  if((rasterLighting.control.w&2u)==0u){
    let origin=vec2i(group.xy*8u+local.xy)*2;
    for(var y=0;y<2;y++){for(var x=0;x<2;x++){rasterLightEdgeAt(origin+vec2i(x,y));}}return;
  }
  let id=rasterQueuedCoordinate(group,local);if(any(id>=textureDimensions(rasterContact))){return;}
  let stride=i32(rasterVisibilityStride());let origin=vec2i(id)*stride;
  for(var y=0;y<stride;y++){for(var x=0;x<stride;x++){rasterLightEdgeAt(origin+vec2i(x,y));}}
}
`;
