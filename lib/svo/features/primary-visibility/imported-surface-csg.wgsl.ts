import { SVO_SCULPT_MATERIAL_OFFSET } from "../scene-publication/svo-sculpt-cuts";

/** The boundary of source minus cut-union has two parts: surviving source
 * surface and cut walls inside the source. Both are evaluated from the same
 * published fields on the GPU. No edited mesh, CPU extraction or readback. */
export const importedSurfaceCsgWGSL = /* wgsl */ `
fn importedSurfaceRemoved(world:vec3f,ownCut:u32)->bool{
  if(svoSculptBaked()){return svoSculptCutDistance(world)<-1e-6;}
  for(var i=0u;i<svoSculptCutCount();i++){
    if(i==ownCut){continue;}
    let bounds=importedCutBounds(i);
    if(all(world>=bounds[0])&&all(world<=bounds[1])&&svoSculptCutRecordDistance(i,world)<-1e-6){return true;}
  }
  return false;
}
fn importedCutBounds(cut:u32)->mat2x3f{
  let at=4u+cut*8u;
  return mat2x3f(bitcast<vec3f>(vec3u(drySculptCutWord(at),drySculptCutWord(at+1u),drySculptCutWord(at+2u))),
    bitcast<vec3f>(vec3u(drySculptCutWord(at+4u),drySculptCutWord(at+5u),drySculptCutWord(at+6u))));
}
fn importedCutProxy(vertex:u32,cut:u32)->vec3f{
  let indices=array<u32,36>(0,2,1,1,2,3,4,5,6,5,7,6,0,1,4,1,5,4,2,6,3,3,6,7,0,4,2,2,4,6,1,3,5,3,7,5);
  let corner=indices[vertex];let bounds=importedCutBounds(cut);
  return select(bounds[0],bounds[1],vec3u(corner&1u,(corner>>1u)&1u,corner>>2u)!=vec3u(0));
}
fn importedCutInterval(cut:u32,ro:vec3f,rd:vec3f)->vec2f{
  let bounds=importedCutBounds(cut);let centre=.5*(bounds[0]+bounds[1]);let radius=.5*(bounds[1]-bounds[0]);
  if(drySculptCutWord(7u+cut*8u)==2u){
    let o=(ro-centre)/radius;let d=rd/radius;let a=dot(d,d);let middle=-dot(o,d)/a;
    // Closest approach avoids subtracting two camera-distance-sized squares
    // when a tiny brush is viewed from far away.
    let closest=o+d*middle;let discriminant=1.0-dot(closest,closest);
    if(discriminant<0.0){return vec2f(1e20,-1e20);}
    let span=sqrt(discriminant/a);return vec2f(middle-span,middle+span);
  }
  var near=-1e20;var far=1e20;
  for(var axis=0u;axis<3u;axis++){
    if(abs(rd[axis])<1e-10){if(ro[axis]<bounds[0][axis]||ro[axis]>bounds[1][axis]){return vec2f(1e20,-1e20);}}
    else{let a=(bounds[0][axis]-ro[axis])/rd[axis];let b=(bounds[1][axis]-ro[axis])/rd[axis];near=max(near,min(a,b));far=min(far,max(a,b));}
  }
  return vec2f(near,far);
}
fn importedBakedCutHit(page:u32,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit{
  let interval=importedCutInterval(page,ro,rd);let end=min(interval.y,tMax);var t=max(interval.x,tMin);
  if(t>end){return missHit();}
  let h=sculptFieldCell();let epsilon=max(1e-6,min(h.x,min(h.y,h.z))*0.001);
  for(var step=0u;step<256u&&t<=end;step++){
    let world=ro+rd*t;let cut=svoSculptCutSample(world);
    if(abs(cut.distance)<epsilon&&dot(cut.normal,cut.normal)>0.5){
      var owner=0xffffffffu;var distance=0.0;
      for(var i=0u;i<dry.metadata.x;i++){
        let record=dryPrimitive(i);if(svoPrimitiveKind(record)!=SVO_KIND_FIELD_PROGRAM){continue;}
        let reference=svoPrimitiveFieldProgramReference(record);
        if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET||drySceneWord(reference)!=0xfffffffeu){continue;}
        let base=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET+drySceneWord(reference+1u);
        let d=dryFieldProgramBlockVolumeDistance(base,svoPrimitiveLocalPoint(record,world));
        if(d<distance){distance=d;owner=i;}
      }
      if(owner!=0xffffffffu){
        let record=dryPrimitive(owner);let reference=svoPrimitiveFieldProgramReference(record);
        let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,svoPrimitiveLocalPoint(record,world));
        let material=select(svoPrimitiveMaterialId(record),surface.material,surface.material!=0u)+${SVO_SCULPT_MATERIAL_OFFSET}u;
        return DryHit(t,-cut.normal,material,svoPrimitiveOwnerId(record),SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
      }
    }
    // Trilinear distance is sampled at the substrate's resolution. A bounded
    // step cannot jump across a thin exposed wall between lattice samples.
    t+=max(epsilon, min(abs(cut.distance)*0.5,min(h.x,min(h.y,h.z))));
  }
  return missHit();
}
fn importedCutHit(cut:u32,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit{
  if(svoSculptBaked()){return importedBakedCutHit(cut,ro,rd,tMin,tMax);}
  let interval=importedCutInterval(cut,ro,rd);
  if(interval.x>interval.y){return missHit();}
  for(var side=0u;side<2u;side++){
    let t=interval[side];if(t<tMin||t>tMax){continue;}
    let world=ro+rd*t;if(importedSurfaceRemoved(world,cut)){continue;}
    var owner=0xffffffffu;var distance=0.0;
    for(var i=0u;i<dry.metadata.x;i++){
      let record=dryPrimitive(i);
      if(svoPrimitiveKind(record)!=SVO_KIND_FIELD_PROGRAM){continue;}
      let reference=svoPrimitiveFieldProgramReference(record);
      if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET||drySceneWord(reference)!=0xfffffffeu){continue;}
      let base=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET+drySceneWord(reference+1u);
      let d=dryFieldProgramBlockVolumeDistance(base,svoPrimitiveLocalPoint(record,world));
      if(d<distance){distance=d;owner=i;}
    }
    if(owner==0xffffffffu){continue;}
    let record=dryPrimitive(owner);let reference=svoPrimitiveFieldProgramReference(record);
    let surface=dryFieldProgramBlockSurface((reference-DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET)/DRY_SCENE_FIELD_PROGRAM_BLOCK_WORDS,svoPrimitiveLocalPoint(record,world));
    let material=select(svoPrimitiveMaterialId(record),surface.material,surface.material!=0u)+${SVO_SCULPT_MATERIAL_OFFSET}u;
    return DryHit(t,-svoSculptCutValue(cut,world).normal,material,svoPrimitiveOwnerId(record),SVO_FEATURE_SMOOTH,DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
  }
  return missHit();
}
`;

export const importedCutPrimaryWGSL = /* wgsl */ `
struct ImportedCutVertex {
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) cut:u32,
}
@vertex fn importedCutVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) cut:u32)->ImportedCutVertex{
  let world=importedCutProxy(vertex,cut);let camera=dryRasterPrimaryCamera();let relative=world-camera[0];
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  return ImportedCutVertex(vec4f(dot(relative,camera[2])/(aspect*cameraTanHalfFov()),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,dot(relative,camera[1])),cut);
}
struct ImportedCutOut {
  @location(0) packedSurface:vec4u,@location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,@location(3) opaqueIdentity:vec2u,
  @builtin(frag_depth) depth:f32,
}
@fragment fn importedCutFragment(input:ImportedCutVertex)->ImportedCutOut{
  let camera=dryRasterPrimaryCamera();let ndc=input.position.xy/uniforms.viewport.xy*vec2f(2,-2)+vec2f(-1,1);
  let rd=normalize(camera[1]+camera[2]*(ndc.x*uniforms.viewport.x/uniforms.viewport.y*cameraTanHalfFov())+camera[3]*(ndc.y*cameraTanHalfFov()));
  let hit=importedCutHit(input.cut,camera[0],rd,DRY_REVERSED_Z_NEAR_M/dot(rd,camera[1]),DRY_MISS);
  if(hit.t>=DRY_MISS){discard;}
  let out=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_SCENE_PRIMITIVE);
  return ImportedCutOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity,dryHardwareDepth(hit.t,rd,camera[1]));
}
`;
