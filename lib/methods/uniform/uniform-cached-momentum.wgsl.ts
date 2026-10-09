/** Shared sampling of prepared h/4h velocity fields, retaining the local
 * owner width for characteristic substeps. Negative h planes escape. */
export const uniformCachedMomentumSamplingWGSL = /* wgsl */ `
var<private> umUnitEscaped:bool;
var<private> umUnitWidth:f32;
var<private> umUnitFineWeight:f32;
fn umUnitInterpolant(p:vec3f,axis:u32)->f32 {
 var offset=vec3f(0.5);offset[axis]=1.0;var lower=vec3f(0.0);lower[axis]=-1.0;
 let q=clamp(p-offset,lower,vec3f(UM_D)-vec3f(1.0));
 let base=vec3i(floor(q));let fraction=fract(q);var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let bit=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));
  terms[k]=textureLoad(unitVelocity,base+bit,0)[axis];
 }
 return umVelocityLerp8(terms,fraction);
}
// 0 native fine stencil, 1 prepared blend, 2 negative-plane escape.
fn umUnitSampleKind(p:vec3f,axis:u32)->u32 {
 if(umFineStencilSample(p)){umUnitWidth=1.0;umUnitFineWeight=1.0;return 0u;}
 let q=clamp(p,vec3f(0),vec3f(UM_D));
 let tile=umTileAt(vec3u(clamp(vec3i(floor(q/4.0)),vec3i(0),vec3i(UM_T)-1)));
 umUnitWidth=f32(umTileWidth(tile));umUnitFineWeight=umVelocitySamplingWeights(p);
 let low=select(p[min(axis,2u)]<1.0,any(p<vec3f(1.0)),axis>=3u);
 return select(1u,2u,umUnitFineWeight>0.0&&low);
}
fn umUnitWeighted(p:vec3f,axis:u32,fine:f32)->f32 {
 var value=0.0;
 if(fine>0.0){value+=fine*umUnitInterpolant(p,axis);}
 let coarse=max(0.0,1.0-fine);
 if(coarse>0.0){value+=coarse*umSampleVelocity4(p,axis);}
 return value;
}
fn umUnitSampleComponent(p:vec3f,axis:u32)->f32 {
 let kind=umUnitSampleKind(p,axis);
 if(kind==0u){return umSampleVelocityFine(p,axis);}
 if(kind==2u){umUnitEscaped=true;return 0.0;}
 return umUnitWeighted(p,axis,umUnitFineWeight);
}
fn umUnitSample(p:vec3f)->vec3f {
 let kind=umUnitSampleKind(p,3u);
 if(kind==0u){return vec3f(umSampleVelocityFine(p,0u),umSampleVelocityFine(p,1u),umSampleVelocityFine(p,2u));}
 if(kind==2u){umUnitEscaped=true;return vec3f(0);}
 let fine=umUnitFineWeight;
 return vec3f(umUnitWeighted(p,0u,fine),umUnitWeighted(p,1u,fine),umUnitWeighted(p,2u,fine));
}
fn umUnitDeparture(position:vec3f,dt:f32,h:vec3f)->vec3f {
  var point=position;var remaining=abs(dt);let direction=select(-1.0,1.0,dt>=0.0);
  for(var step=0;step<32;step+=1){
    if(remaining<=1e-7){break;}
    let first=umUnitSample(point);if(umUnitEscaped){break;}
    let rate=max(abs(first.x)/h.x,max(abs(first.y)/h.y,abs(first.z)/h.z))/umUnitWidth;
    let stepSeconds=min(remaining,1.5/max(rate,1e-6));let signedStep=direction*stepSeconds;
    let midpoint=umClampMomentum(point-0.5*first*signedStep/h);
    let second=umUnitSample(midpoint);if(umUnitEscaped){break;}
    point=umClampMomentum(point-second*signedStep/h);remaining-=stepSeconds;
  }
  return point;
}
`;
