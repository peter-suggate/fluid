/** A native asset is extracted once. Hardware depth handles visibility; the
 * fragment performs one source normal/PBR lookup, never a field ray march. */
export const importedSurfaceWGSL = /* wgsl */ `
@group(1) @binding(80) var<storage,read> importedLodRecords:array<vec4u>;
@group(1) @binding(81) var<storage,read_write> importedDrawArgs:array<atomic<u32>>;
@group(1) @binding(82) var<storage,read_write> importedInstanceIds:array<u32>;
@group(1) @binding(83) var<storage,read> importedGroups:array<vec4u>;
@compute @workgroup_size(64) fn importedSurfaceResetGroups(@builtin(global_invocation_id) id:vec3u){
  if(id.x>=arrayLength(&importedGroups)){return;}
  let group=importedGroups[id.x];let at=id.x*5u;
  atomicStore(&importedDrawArgs[at],group.x);atomicStore(&importedDrawArgs[at+1u],0u);
  atomicStore(&importedDrawArgs[at+2u],group.y);atomicStore(&importedDrawArgs[at+3u],0u);atomicStore(&importedDrawArgs[at+4u],0u);
}
fn importedSurfaceSelectInstance(instance:u32,batched:bool){
  if(instance>=arrayLength(&importedLodRecords)/8u){return;}
  let base=instance*8u;let header=importedLodRecords[base];let primitive=header.x;
  let record=dryPrimitive(primitive);let camera=dryRasterPrimaryCamera();
  let centre=svoPrimitiveCenter_m(record);let radius=bitcast<f32>(header.z);
  let relative=centre-camera[0];let z=dot(relative,camera[1]);let ty=cameraTanHalfFov();let tx=ty*uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let visible=z+radius>=DRY_REVERSED_Z_NEAR_M
    &&abs(dot(relative,camera[2]))<=z*tx+radius*sqrt(1.0+tx*tx)
    &&abs(dot(relative,camera[3]))<=z*ty+radius*sqrt(1.0+ty*ty);
  let depth=max(dot(centre-camera[0],camera[1])-radius,DRY_REVERSED_Z_NEAR_M);
  let pixelsPerMetre=uniforms.viewport.y/(2.0*cameraTanHalfFov()*depth);
  var selected=importedLodRecords[base+1u];var level=0u;
  for(var i=1u;i<header.y;i+=1u){let candidate=importedLodRecords[base+1u+i];if(bitcast<f32>(candidate.z)*pixelsPerMetre<=0.25){selected=candidate;level=i;}}
  if(batched){
    if(!visible){return;}
    let groupId=header.w+level;let group=importedGroups[groupId];
    let slot=atomicAdd(&importedDrawArgs[groupId*5u+1u],1u);
    importedInstanceIds[group.z+slot]=primitive;
  }else{
    let at=instance*5u;atomicStore(&importedDrawArgs[at],select(0u,selected.y,visible));atomicStore(&importedDrawArgs[at+1u],1u);
    atomicStore(&importedDrawArgs[at+2u],selected.x);atomicStore(&importedDrawArgs[at+3u],0u);atomicStore(&importedDrawArgs[at+4u],0u);
    importedInstanceIds[instance]=primitive;
  }
}
@compute @workgroup_size(64) fn importedSurfaceSelect(@builtin(global_invocation_id) id:vec3u){importedSurfaceSelectInstance(id.x,false);}
@compute @workgroup_size(64) fn importedSurfaceSelectBatched(@builtin(global_invocation_id) id:vec3u){importedSurfaceSelectInstance(id.x,true);}
struct ImportedSurfaceVertex {
  @builtin(position) position:vec4f,
  @location(0) local:vec3f,
  @location(1) world:vec3f,
  @location(2) @interpolate(flat) primitive:u32,
}
@vertex fn importedSurfaceVertex(@location(0) local:vec3f,@location(1) primitive:u32)->ImportedSurfaceVertex{
  let record=dryPrimitive(primitive);
  let world=svoPrimitiveCenter_m(record)+svoQuaternionRotate(record.orientation,local);
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let clip=vec4f(dot(relative,camera[2])/(aspect*cameraTanHalfFov()),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,dot(relative,camera[1]));
  return ImportedSurfaceVertex(clip,local,world,primitive);
}
struct ImportedSurfaceOut {
  @location(0) packedSurface:vec4u,@location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,@location(3) opaqueIdentity:vec2u,
}
@fragment fn importedSurfaceFragment(input:ImportedSurfaceVertex)->ImportedSurfaceOut{
  let record=dryPrimitive(input.primitive);
  let reference=svoPrimitiveFieldProgramReference(record);
  let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,input.local);
  let normal=normalize(svoQuaternionRotate(record.orientation,surface.normal));
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);
  let hit=DryHit(length(input.world-camera[0]),normal,select(svoPrimitiveMaterialId(record),surface.material,surface.material!=0u),svoPrimitiveOwnerId(record),SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
  let out=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
  return ImportedSurfaceOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity);
}
`;
