/** Producer-only support fitting against the voxelizer's authored fields. */
export function svoCellContourFitWGSL(renderTerrain: boolean, solidWorld: boolean): string {
  const staticBound = renderTerrain ? /* wgsl */ `fn contourStaticMayOccupy(minimum:vec3f,maximum:vec3f)->bool{
  // Edits can replace the heightfield with exact voxel boxes. Keep their cubes.
  let patches=min(atomicLoad(&maintenance[RT_BASE+7u]),RT_PATCH_CAPACITY);
  for(var i=0u;i<patches;i+=1u){
    let base=RT_PATCH_BASE+i*8u;
    let low=vec3f(rtFloat(base),rtFloat(base+1u),rtFloat(base+2u));
    let high=vec3f(rtFloat(base+4u),rtFloat(base+5u),rtFloat(base+6u));
    if(all(maximum>=low)&&all(minimum<=high)){return true;}
  }
  let origin=vec2f(rtFloat(RT_BASE),rtFloat(RT_BASE+1u));
  let cell=max(vec2f(rtFloat(RT_BASE+2u),rtFloat(RT_BASE+3u)),vec2f(1e-8));
  // Limit the reconstructed field to its authored XZ domain.
  let low=max(minimum.xz,origin);let high=min(maximum.xz,origin+cell*vec2f(f32(RT_WIDTH),f32(RT_DEPTH)));
  if(any(low>high)){return false;}
  return minimum.y<=rtSurfaceRange(low,high).y;
}` : solidWorld ? /* wgsl */ `
fn contourStaticMayOccupy(minimum:vec3f,maximum:vec3f)->bool{
  // Canonical voxel-only geometry cannot be narrowed, but empty canonical
  // cells must not disable contours on unrelated procedural geometry.
  let cell=max(swCell(),vec3f(1e-8));
  let low=vec3i(floor((minimum-swOrigin())/cell));
  let high=vec3i(floor((maximum-swOrigin())/cell));
  if(any(high-low>vec3i(8))){return true;}
  for(var z=low.z;z<=high.z;z+=1){for(var y=low.y;y<=high.y;y+=1){for(var x=low.x;x<=high.x;x+=1){
    if(swFractionQ8(vec3i(x,y,z))!=0u){return true;}
  }}}
  return false;
}` : `fn contourStaticMayOccupy(minimum:vec3f,maximum:vec3f)->bool{return false;}`;
  return staticBound + /* wgsl */ `
// A one-sided Laine slab: the lower plane is the cube's own support plane.
// Only the upper offset needs storage; its normal is the accepted oct8 identity.
fn contourPrimitiveLowerBound(p:ScenePrimitive,world:vec3f)->f32{
  if(scenePrimitiveType(p)==3u){
    let q=inverseRotate(world-p.centerType.xyz,p.rotation);let r=p.extentIdentity.xyz;
    return (length(q/r)-1.0)*min(r.x,min(r.y,r.z));
  }
  return primitiveDistance(p,world);
}
fn fitSceneContour(world:vec3f,cell:vec3f,normal:vec3f,fraction:f32,dirtyIndex:u32,candidates:u32)->u32{
  if(!(fraction>0.0&&fraction<1.0)||dot(normal,normal)<0.5){return 0u;}
  // Fit to the decoded normal. Quantizing it later could cut into the source.
  let n=svoGBufferUnpackNormalOct8(svoGBufferPackNormalOct8(normal));
  let radius=0.5*dot(abs(n),cell);
  let small=cell/4.0;let smallRadius=0.5*length(small);
  var support=-radius;var found=false;
  for(var i=0u;i<candidates;i+=1u){
    let p=primitives[atomicLoad(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+i])];
    // A threshold field's occupancy rule is not distance-expanded geometry.
    if(primitiveUsesThresholdOccupancy(p)){return 0u;}
  }
  for(var z=0u;z<4u;z+=1u){for(var y=0u;y<4u;y+=1u){for(var x=0u;x<4u;x+=1u){
    let offset=(vec3f(f32(x),f32(y),f32(z))+vec3f(0.5))*small-0.5*cell;
    let p=world+offset;
    var occupied=contourStaticMayOccupy(p-0.5*small,p+0.5*small);
    for(var i=0u;i<candidates&&!occupied;i+=1u){
      let primitive=primitives[atomicLoad(&maintenance[candidateOffset()+dirtyIndex*candidatesPerBrick()+i])];
      occupied=contourPrimitiveLowerBound(primitive,p)<=smallRadius+1e-6;
    }
    if(occupied){found=true;support=max(support,dot(n,offset)+0.5*dot(abs(n),small));}
  }}}
  // Producer-only empty result. Code zero means retain the cube, so using it
  // here resurrects a coverage-expanded cell whose entire domain proved empty.
  // The publisher consumes 255 by clearing occupancy; it is never stored.
  if(!found){return 255u;}
  // Round outward and reserve another step for packed triangle coordinates.
  // 0 means absent; 255 (the whole cube) is also written as absent.
  let q=u32(clamp(ceil((support/radius+1.0)*127.5)+1.0,1.0,255.0));
  return select(q,0u,q>=255u);
}
`;
}
