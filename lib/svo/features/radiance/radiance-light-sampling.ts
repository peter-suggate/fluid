/** Finite-source quadrature matching the direct surface-light contract.
 * Kept independently composable so the radiance builder does not import the
 * full surface shader. GPU parity tests exercise both implementations.
 */
export const RADIANCE_AREA_LIGHT_SAMPLES = 2;
export const radianceLightSamplingWGSL = /* wgsl */ `
struct RadianceLightSample{towardLight:vec3f,finiteDistance_m:f32,radiance:vec3f,valid:u32}
fn radianceInvalidLightSample()->RadianceLightSample{return RadianceLightSample(vec3f(0.0,1.0,0.0),0.0,vec3f(0.0),0u);}
fn radianceLightSample(light:SvoLightRecord,sampleIndex:u32,position:vec3f)->RadianceLightSample {
  let baseRadiance=svoLightRadiance(light);if(max(max(baseRadiance.x,baseRadiance.y),baseRadiance.z)<=0.0){return radianceInvalidLightSample();}
  if(light.identity.x==SVO_LIGHT_DIRECTIONAL){let lengthSquared=dot(light.directionCone.xyz,light.directionCone.xyz);if(lengthSquared<=1e-12){return radianceInvalidLightSample();}return RadianceLightSample(light.directionCone.xyz*inverseSqrt(lengthSquared),0.0,baseRadiance,1u);}
  var samplePosition=light.positionRange.xyz;
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA||light.identity.x==SVO_LIGHT_SPOT){
    // A spot jitters across its mouth disc for the same reason a sphere jitters
    // across its silhouette: the shadow edge of a finite emitter is a penumbra,
    // and a single centre sample renders it as a hard line.
    let towardCenter=normalize(light.positionRange.xyz-position);let helper=select(vec3f(0.0,1.0,0.0),vec3f(1.0,0.0,0.0),abs(towardCenter.y)>.9);let tangent=normalize(cross(towardCenter,helper));let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=tangent*(signValue*.45*light.shape.x);
  }else if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){
    let signValue=select(-1.0,1.0,sampleIndex!=0u);samplePosition+=light.axisUWidth.xyz*(signValue*.45*light.axisUWidth.w)+light.axisVHeight.xyz*(signValue*.2*light.axisVHeight.w);
  }
  let offset=samplePosition-position;let distanceSquared=dot(offset,offset);if(distanceSquared<=1e-10){return radianceInvalidLightSample();}if(light.positionRange.w>0.0&&distanceSquared>=light.positionRange.w*light.positionRange.w){return radianceInvalidLightSample();}let distance=sqrt(distanceSquared);let towardLight=offset/distance;
  let rangeFade=select(1.0,pow(clamp(1.0-distance/max(light.positionRange.w,1e-6),0.0,1.0),2.0),light.positionRange.w>0.0);
  var shapeScale=1.0/max(1.0,distanceSquared);
  if(light.identity.x==SVO_LIGHT_SPHERE_AREA){let area=4.0*3.141592653589793*light.shape.x*light.shape.x;shapeScale=area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_RECTANGLE_AREA){let area=4.0*light.axisUWidth.w*light.axisVHeight.w;let emitterFacing=max(dot(normalize(light.directionCone.xyz),-towardLight),0.0);shapeScale=emitterFacing*area/max(area,distanceSquared);}
  if(light.identity.x==SVO_LIGHT_SPOT){shapeScale*=svoLightConeFalloff(light,towardLight);}
  let radiance=baseRadiance*(rangeFade*shapeScale);if(max(max(radiance.x,radiance.y),radiance.z)<=0.0){return radianceInvalidLightSample();}
  // Point fixtures retain the finite radius of their visible emissive proxy.
  // Attenuation uses center distance, while visibility ends at the globe's
  // near surface so the source geometry cannot occlude its own light.
  // Zero is reserved for directional lights. A receiver inside a finite
  // emitter envelope has an empty local shadow segment, never a scene-exit ray.
  let visibilityDistance=select(distance,max(1e-6,distance-light.shape.x),light.identity.x==SVO_LIGHT_POINT||light.identity.x==SVO_LIGHT_SPOT);
  return RadianceLightSample(towardLight,visibilityDistance,radiance,1u);
}
`;
