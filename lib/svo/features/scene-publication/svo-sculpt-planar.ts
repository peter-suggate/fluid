/** Exact finite slabs retain their geometry and material after CSG edits.
 * Shared by raster primary visibility, structural traversal and shadow rays. */
export const svoSculptPlanarWGSL = /* wgsl */ `
fn dryPlanarBoundaryHit(boundary:PlanarBoundaryPatch,ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->DryHit{
  let exact=intersectPlanarBoundary(boundary,ro,rd,tMin,tMax);
  let identity=planarBoundaryIdentity(boundary);
  let axes=mat3x3f(boundary.tangentUExtentV.xyz,planarBoundaryTangentV(boundary),boundary.normalExtentU.xyz);
  let extent=vec3f(boundary.normalExtentU.w,boundary.tangentUExtentV.w,boundary.centerThickness.w);
  let startLocal=transpose(axes)*(ro+rd*tMin-boundary.centerThickness.xyz);
  let inside=all(abs(startLocal)<extent);
  if(exact.valid==0u&&!inside){return missHit();}
  var t=exact.tHit;var normal=exact.normal;var feature=SVO_FEATURE_BOX_X+exact.featureAxis;
  // Untouched pixels take precisely the original intersection, without a
  // tessellation/resampling change anywhere else on the source surface.
  if(svoSculptCutCount()>0u&&(inside||svoSculptCutDistance(ro+rd*t)<0.0)){
    t=select(exact.tEnter,tMin,inside);
    let limit=select(exact.tExit,tMax,exact.valid==0u);
    var found=false;
    for(var step=0u;step<512u;step++){
      let point=ro+rd*t;let local=transpose(axes)*(point-boundary.centerThickness.xyz);
      let q=abs(local)-extent;
      let source=length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);
      let cut=svoSculptCutSample(point);let retained=max(source,-cut.distance);
      if(abs(retained)<1e-5){
        if(-cut.distance>=source){normal=-cut.normal;feature=SVO_FEATURE_SMOOTH;}
        else{var axis=0u;if(q.y>q.x){axis=1u;}if(q.z>q[axis]){axis=2u;}
          normal=axes[axis]*sign(local[axis]);feature=SVO_FEATURE_BOX_X+axis;}
        found=true;break;
      }
      t+=max(abs(retained),1e-5);if(t>limit){break;}
    }
    if(!found){return missHit();}
  }else if(exact.valid==0u){return missHit();}
  return DryHit(t,normal,identity&0xffffu,identity>>16u,feature,
    DRY_GBUFFER_FIELD_ANALYTIC,DRY_GBUFFER_MOTION_STATIC,1u,0.0,vec3u(0u));
}
`;
