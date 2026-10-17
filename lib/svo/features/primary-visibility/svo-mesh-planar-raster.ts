import { PLANAR_BOUNDARY_PATCH_BYTES } from "../../../core/planar-boundary";

/** Rasterize each accepted planar slab's conservative screen rectangle. The
 * fragment still uses the catalogue's exact intersector and identity; only
 * candidate discovery changes from every slab at every screen pixel. */
export const svoMeshPlanarRasterWGSL = /* wgsl */ `
struct MeshPlanarVertexOut {
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) index:u32,
}
@vertex fn meshPlanarVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) index:u32)->MeshPlanarVertexOut{
  let hidden=MeshPlanarVertexOut(vec4f(2,2,0,1),index);
  if(dry.planarBoundaries.y==0u||dry.planarBoundaries.z!=${PLANAR_BOUNDARY_PATCH_BYTES}u||index>=dry.planarBoundaries.x){return hidden;}
  let boundary=dryPlanarBoundaries[index];
  if(dryOpaqueOwnerSuppressed(planarBoundaryIdentity(boundary)>>16u)){return hidden;}
  let camera=dryRasterPrimaryCamera();
  let u=boundary.tangentUExtentV.xyz*boundary.normalExtentU.w;
  let v=planarBoundaryTangentV(boundary)*boundary.tangentUExtentV.w;
  let n=boundary.normalExtentU.xyz*boundary.centerThickness.w;
  let scale=vec2f(cameraTanHalfFov()*uniforms.viewport.x/max(uniforms.viewport.y,1.0),cameraTanHalfFov());
  var low=vec2f(1e30);var high=vec2f(-1e30);var crossesEye=false;var furthest=-1e30;
  for(var corner=0u;corner<8u;corner++){
    let signs=select(vec3f(-1),vec3f(1),(vec3u(corner)&vec3u(1,2,4))!=vec3u(0));
    let relative=boundary.centerThickness.xyz+u*signs.x+v*signs.y+n*signs.z-camera[0];
    let z=dot(relative,camera[1]);furthest=max(furthest,z);
    if(z<=DRY_REVERSED_Z_NEAR_M){crossesEye=true;continue;}
    let xy=vec2f(dot(relative,camera[2]),dot(relative,camera[3]))/(z*scale);
    low=min(low,xy);high=max(high,xy);
  }
  if(furthest<=0.0){return hidden;}
  // Near-plane crossings and an eye inside the slab must not lose coverage.
  if(crossesEye){low=vec2f(-1);high=vec2f(1);}
  let margin=2.0/max(uniforms.viewport.xy,vec2f(1));
  low=max(low-margin,vec2f(-1));high=min(high+margin,vec2f(1));
  if(any(low>=high)){return hidden;}
  let corners=array<vec2f,4>(vec2f(0,0),vec2f(1,0),vec2f(0,1),vec2f(1,1));
  return MeshPlanarVertexOut(vec4f(mix(low,high,corners[vertex]),1,1),index);
}
@fragment fn meshPlanarFragment(input:MeshPlanarVertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  let boundary=dryPlanarBoundaries[input.index];
  let exact=intersectPlanarBoundary(boundary,camera[0],rd,1e-4,DRY_MISS);
  if(exact.valid==0u){discard;}
  let identity=planarBoundaryIdentity(boundary);
  let hit=DryHit(exact.tHit,exact.normal,identity&0xffffu,identity>>16u,
    SVO_FEATURE_BOX_X+exact.featureAxis,DRY_GBUFFER_FIELD_ANALYTIC,
    DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
  return dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND);
}
`;
