/** Reconcile an extracted FVX triangle with the same trilinear scalar field
 * used by secondary rays. This changes only the ray origin, never geometry or
 * shading normals. The derivative is analytic, independent of normal maps. */
export function sampledSurfaceProjectionWGSL(load: string): string {
  return /* wgsl */ `
struct SampledSurfaceField { distance:f32, gradient:vec3f, cellSize:f32, valid:u32 }
fn sampledSurfaceField(base:u32,p:vec3f)->SampledSurfaceField{
  var result:SampledSurfaceField;
  let origin=vec3f(bitcast<f32>(${load}(base)),bitcast<f32>(${load}(base+1u)),bitcast<f32>(${load}(base+2u)));
  let h=bitcast<f32>(${load}(base+3u));
  let dims=vec3u(${load}(base+4u),${load}(base+5u),${load}(base+6u));
  if(!(h>0.0)||any(dims==vec3u(0u))||any(p!=p)){return result;}
  let q=(p-origin)/h;
  if(any(q<vec3f(0))||any(q>vec3f(dims))){return result;}
  let cell=min(vec3u(floor(q)),dims-vec3u(1));let b=cell>>vec3u(3);
  let bd=vec3u(${load}(base+8u),${load}(base+9u),${load}(base+10u));
  let relative=${load}(base+16u+2u*(b.x+bd.x*(b.y+bd.y*b.z)));
  if(relative==0u){return result;}
  let c=cell&vec3u(7);let f=q-vec3f(cell);
  for(var i=0u;i<8u;i++){
    let corner=vec3u(i&1u,(i>>1u)&1u,i>>2u);let n=c+corner;
    let d=bitcast<f32>(${load}(base+relative+2u*(n.x+9u*(n.y+9u*n.z))));
    let w=select(vec3f(1)-f,f,corner!=vec3u(0));let sign=select(vec3f(-1),vec3f(1),corner!=vec3u(0));
    result.distance+=d*w.x*w.y*w.z;
    result.gradient+=d*sign*vec3f(w.y*w.z,w.x*w.z,w.x*w.y)/h;
  }
  result.cellSize=h;
  result.valid=select(0u,1u,result.distance==result.distance&&all(result.gradient==result.gradient)&&all(abs(result.gradient)<vec3f(1e10)));
  return result;
}
struct SampledSurfaceProjection { position:vec3f, normal:vec3f, valid:u32 }
fn sampledSurfaceProjection(base:u32,original:vec3f)->SampledSurfaceProjection{
  var point=original;var result:SampledSurfaceProjection;
  let initial=sampledSurfaceField(base,point);if(initial.valid==0u){return result;}
  // The extracted triangle is confined to one source cell. Refuse a jump to
  // another sheet instead of silently moving an unrelated receiver.
  let travelLimit=.5*initial.cellSize;
  let tolerance=max(max(1e-6,initial.cellSize*1e-5),2e-7*max(abs(point.x),max(abs(point.y),abs(point.z))));
  for(var iteration=0u;iteration<9u;iteration++){
    let field=sampledSurfaceField(base,point);let squared=dot(field.gradient,field.gradient);
    if(field.valid==0u||!(squared>1e-10)){return result;}
    if(abs(field.distance)<=tolerance){return SampledSurfaceProjection(point,field.gradient*inverseSqrt(squared),1u);}
    if(iteration==8u){break;}
    let step=field.gradient*(field.distance/squared);
    let boundedStep=step*min(1.0,.25*initial.cellSize/max(length(step),1e-20));
    point-=boundedStep;
    if(length(point-original)>travelLimit){return result;}
  }
  return result;
}
`;
}

/** Host adapter: only identified, uncut imported receivers have a source field. */
export const sampledShadowSurfaceWGSL = /* wgsl */ `
fn drySampledShadowSurface(position:vec3f,normal:vec3f,owner:u32)->SampledSurfaceProjection{
  // Structural SVO receivers may carry sampled materials after owners have
  // been merged. They are valid voxel surfaces, not a particular imported
  // triangle; only an identified source owner can be reconciled.
  if(owner==DRY_OWNER_NONE){return SampledSurfaceProjection(position,normal,1u);}
  // Edited cut caps belong to the retained CSG field, not the source volume.
  // Keep their existing origin contract; do not project a cap onto a removed
  // original surface. Unedited scenes use the source field below.
  if(svoSculptCutCount()>0u){return SampledSurfaceProjection(position,normal,1u);}
  let index=drySampledMaterialOwner(owner);if(index==0xffffffffu){return SampledSurfaceProjection();}
  let record=dryPrimitive(index);if(svoPrimitiveKind(record)!=SVO_KIND_FIELD_PROGRAM){return SampledSurfaceProjection();}
  let reference=svoPrimitiveFieldProgramReference(record);
  if(reference<DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET||drySceneWord(reference)!=0xfffffffeu){return SampledSurfaceProjection();}
  let base=DRY_SCENE_FIELD_PROGRAM_WORD_OFFSET+drySceneWord(reference+1u);
  let projected=sampledSurfaceProjection(base,svoPrimitiveLocalPoint(record,position));
  if(projected.valid==0u){return projected;}
  let q=record.orientation;
  return SampledSurfaceProjection(svoPrimitiveCenter_m(record)+svoQuaternionRotate(q,projected.position),svoQuaternionRotate(q,projected.normal),1u);
}
`;
