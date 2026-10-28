export const waterVerifiedLiquidIntervalWGSL = /* wgsl */ `
fn waterWorldTransmissionEligible(meshExitValid:bool,opaqueSolidExit:bool,worldDistance:f32,opaqueDistance:f32)->bool{
  return !meshExitValid&&worldDistance>0.0&&(!opaqueSolidExit||worldDistance<=opaqueDistance);
}
// Check the actual resident liquid field up to the opaque receiver. A nearby
// extracted contact may differ by less than one cell; the last half interval
// still receives a sample. Leaving liquid/storage or exhausting the fixed
// budget is unknown, not permission to absorb an air path as water.
fn waterVerifiedLiquidInterval(front:vec3f,direction:vec3f,distance:f32,cell:f32)->bool{
  if(!(distance>0.0)||distance>1e6){return false;}
  let end=max(distance*.5,distance-.75*cell);let count=u32(ceil(end/max(.5*cell,1e-5)));
  if(count==0u||count>128u){return false;}
  let lower=vec3f(-.5*u.container.x,0,-.5*u.container.z);let upper=lower+u.container.xyz;
  let budget=min(count,contactBound(128u));if(budget<count){return false;}
  for(var i=0u;i<budget;i+=1u){let t=end*(f32(i)+.5)/f32(count);let p=front+direction*t;
    if(any(p<lower)||any(p>upper)){return false;}
    let value=contactFluidValue(p);if(!(value>=.5)){return false;}
  }
  return true;
}
`;

/** Optional world-space opaque transmission requests and bounded liquid proof.
 * No screen-edge colour is used as a substitute for an offscreen receiver. */
export const waterTransmissionWGSL = /* wgsl */ `
fn waterTransmissionRequest(positions:texture_2d<f32>,normals:texture_2d<f32>,backs:texture_2d<f32>,uv:vec2f)->vec4f{
  let front=safePositionSample(positions,uv);if(front.a<.5){return vec4f(0);}
  let ro=u.cameraPosition.xyz;let incoming=normalize(front.xyz-ro);let cell=min(u.container.x/u.gridInfo.x,min(u.container.y/u.gridInfo.y,u.container.z/u.gridInfo.z));
  if(resolvedDrySceneDepth(safeSample(sceneTexture,uv).a)+max(.0015,.18*cell)<length(front.xyz-ro)){return vec4f(0);}
  let packedNormal=safeInterfaceSample(normals,uv);if(dot(packedNormal.xyz,packedNormal.xyz)<1e-12){return vec4f(0);}
  var normal=normalize(packedNormal.xyz);if(dot(normal,incoming)>0.0){normal=-normal;}
  let direction=refract(incoming,normal,1.0/waterIndexOfRefraction());if(dot(direction,direction)<.5){return vec4f(0);}
  let forward=normalize(u.cameraTarget.xyz-ro);var exitUV=uv;
  for(var i=0;i<3;i+=1){let back=safePositionSample(backs,exitUV);if(back.a<.5){break;}
    let travel=max(0.0,dot(back.xyz-front.xyz,forward)/max(dot(direction,forward),.001));exitUV=project(front.xyz+direction*travel);}
  let back=safePositionSample(backs,exitUV);if(back.a>=.5&&length(back.xyz-front.xyz)>=1e-4){return vec4f(0);}
  return vec4f(direction,1);
}
${waterVerifiedLiquidIntervalWGSL}
fn waterVerifyTransmission(radiance:texture_2d_array<f32>,positions:texture_2d<f32>,normals:texture_2d<f32>,uv:vec2f)->vec4f{
  let at=clamp(vec2i(uv*vec2f(textureDimensions(radiance))),vec2i(0),vec2i(textureDimensions(radiance))-1);
  let query=textureLoad(radiance,at,1,0);if(query.a<=0.0){return query;}
  let front=safePositionSample(positions,uv);var n=normalize(safeInterfaceSample(normals,uv).xyz);let incoming=normalize(front.xyz-u.cameraPosition.xyz);if(dot(n,incoming)>0.0){n=-n;}
  let rawDirection=refract(incoming,n,1.0/waterIndexOfRefraction());
  // The query direction crossed an rgba16float request target. Verify the
  // same quantized ray that the SVO traced, not a nearby full-float ray.
  let xy=unpack2x16float(pack2x16float(rawDirection.xy));let z=unpack2x16float(pack2x16float(vec2f(rawDirection.z,0))).x;
  let direction=normalize(vec3f(xy,z));let cell=min(u.container.x/u.gridInfo.x,min(u.container.y/u.gridInfo.y,u.container.z/u.gridInfo.z));
  if(front.a<.5||!waterVerifiedLiquidInterval(front.xyz,direction,query.a,cell)){return vec4f(0,0,0,-4);}
  return query;
}
@fragment fn transmissionRequestMain(input:VOut)->ReflectionOut{
  let uv=vec2f(input.uv.x,1.0-input.uv.y);
  return ReflectionOut(waterTransmissionRequest(frontPosition,frontNormal,backPosition,uv),waterTransmissionRequest(rearFrontPosition,rearFrontNormal,rearBackPosition,uv));
}
@fragment fn transmissionVerifyMain(input:VOut)->ReflectionOut{
  // Each entry point owns its private sampler state. Without this, the mixed
  // lattice has zero extents even though the resident textures are bound.
  udrInit();
  let uv=vec2f(input.uv.x,1.0-input.uv.y);
  return ReflectionOut(waterVerifyTransmission(frontReflections,frontPosition,frontNormal,uv),waterVerifyTransmission(rearReflections,rearFrontPosition,rearFrontNormal,uv));
}
`;
