/** Shared sampling of prepared h/4h velocity fields, retaining the local
 * owner width for characteristic substeps. Negative h planes escape. A
 * sample resolves its tile once (umSampleTile): the tile's key classifies it
 * and names the 4h cache block its three components read, and its unit
 * taps' halo home (umUnitHaloHome: the module binds unitHalo). */
export const uniformCachedMomentumSamplingWGSL = /* wgsl */ `
var<private> umUnitEscaped:bool;
var<private> umUnitWidth:f32;
var<private> umUnitFineWeight:f32;
fn umUnitInterpolant(p:vec3f,axis:u32)->f32 {
 var offset=vec3f(0.5);offset[axis]=1.0;
 let b=umHold;var lower=b.lo;lower[axis]-=1.0;
 let q=clamp(p-offset,lower,b.hi-vec3f(1.0));
 let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
 // The sample's page's halo block holds every tap, the low-wall faces
 // included: plain offsets from the home its sample resolved once.
 if(umUnitHome.w>=0){
  for(var k=0u;k<8u;k++){
   terms[k]=textureLoad(unitHalo,base+vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u))+umUnitHome.xyz,0)[axis];
  }
  return umVelocityLerp8(terms,fraction);
 }
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  terms[k]=textureLoad(unitVelocity,base+bit,0)[axis];
 }
 return umVelocityLerp8(terms,fraction);
}
// 0 native fine stencil, 1 prepared blend, 2 negative-plane escape: a fine
// sample within a cell of a low wall plane (umWallLow) taps the negative planes.
// A native fine stencil with no halo home escapes too: the general sampler
// stays out of this module (its inlined code cost the common path occupancy).
fn umUnitSampleKind(p:vec3f,s:UMSampleTile,axis:u32)->u32 {
 if(umFineStencilTile(s.key)){umUnitWidth=1.0;umUnitFineWeight=1.0;return 0u;}
 umUnitWidth=f32(umTileWidth(s.key));umUnitFineWeight=umVelocitySamplingWeightsAt(s);
 let wall=vec3f(umWallLow())+vec3f(1.0);let low=select(p[min(axis,2u)]<wall[min(axis,2u)],any(p<wall),axis>=3u);
 return select(1u,2u,umUnitFineWeight>0.0&&low);
}
fn umUnitWeighted(p:vec3f,home:UMCoarseHome,axis:u32,fine:f32)->f32 {
 var value=0.0;
 if(fine>0.0){value+=fine*umUnitInterpolant(p,axis);}
 let coarse=max(0.0,1.0-fine);
 if(coarse>0.0){value+=coarse*umSampleVelocity4In(home,axis);}
 return value;
}
// A native fine stencil's taps are stored unit faces, which its halo
// holds: with a home it is the unit interpolant at weight one (exactly).
fn umUnitSampleComponent(p:vec3f,axis:u32)->f32 {
 let s=umSampleTile(p);let kind=umUnitSampleKind(p,s,axis);umUnitHome=umUnitHaloHome(s);
 if(kind==2u||(kind==0u&&umUnitHome.w<0)){umUnitEscaped=true;return 0.0;}
 return umUnitWeighted(p,umCoarseHome(p,s),axis,umUnitFineWeight);
}
// The three components in one loop, each interpolant at one call site
// (Metal inlines every call: one written three times compiles three times).
fn umUnitSample(p:vec3f)->vec3f {
 let s=umSampleTile(p);let kind=umUnitSampleKind(p,s,3u);
 umUnitHome=umUnitHaloHome(s);
 if(kind==2u||(kind==0u&&umUnitHome.w<0)){umUnitEscaped=true;return vec3f(0);}
 let fine=umUnitFineWeight;let home=umCoarseHome(p,s);var v=vec3f(0.0);
 for(var a=0u;a<3u;a++){
  v[a]=umUnitWeighted(p,home,a,fine);
 }
 return v;
}
// Each substep's two samples (the start, then the midpoint) at one call site.
fn umUnitDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f {
  var point=position;var remaining=abs(dt);let direction=select(-1.0,1.0,dt>=0.0);
  var at=position;var mid=false;var stepSeconds=0.0;var signedStep=0.0;
  for(var stage=0;stage<64;stage+=1){
    if(!mid&&remaining<=1e-7){break;}
    let sampled=umUnitSample(at);if(umUnitEscaped){break;}
    if(!mid){
      let rate=max(abs(sampled.x)/h.x,max(abs(sampled.y)/h.y,abs(sampled.z)/h.z))/umUnitWidth;
      stepSeconds=min(remaining,1.5/max(rate,1e-6));signedStep=direction*stepSeconds;
      at=umClampMomentum(point-0.5*sampled*signedStep/h);mid=true;continue;
    }
    point=umClampMomentum(point-sampled*signedStep/h);remaining-=stepSeconds;at=point;mid=false;
  }
  return point;
}
`;
