/** Source geometry shared by native and canonical mixed execution. */
export const uniformDropSourceWGSL = /* wgsl */ `
fn dropSource(q:vec3i)->f32{
  let radius=params.drop.w;if(radius<=0.0){return 0.0;}
  let h=params.cellGravity.xyz;
  let minimum=vec3f(-0.5*params.container.x,0.0,-0.5*params.container.z);
  var covered=0.0;
  for(var sample=0u;sample<8u;sample+=1u){
    let offset=vec3f(f32(sample&1u),f32((sample>>1u)&1u),f32((sample>>2u)&1u))*0.5+vec3f(0.25);
    let point=minimum+(vec3f(q)+offset)*h;
    let d=point-params.drop.xyz;
    let inside=select(length(d)<=radius,
      length(d.xy)<=radius&&abs(d.z)<=params.dropExtent.x,
      params.dropExtent.x>0.0);
    if(inside){covered+=0.125;}
  }
  return covered;
}
`;
export const uniformSourcePhiWGSL = /* wgsl */ `
fn uvSourcePhi(p:vec3f,phi:f32)->f32{
  var result=phi;
  if(params.drop.w>0.0){let delta=traceWorld(p)-params.drop.xyz;
    let ball=select(length(delta)-params.drop.w,
      max(length(delta.xy)-params.drop.w,abs(delta.z)-params.dropExtent.x),params.dropExtent.x>0.0);
    result=min(result,ball);}
  let speed=length(params.inflowVelocityLength.xyz)*inflowStrength();
  if(speed>1e-6){let direction=normalize(params.inflowVelocityLength.xyz);
    let delta=traceWorld(p)-params.inflowPositionRadius.xyz;let axial=dot(delta,direction);
    let plug=max(length(delta-axial*direction)-params.inflowPositionRadius.w,
      max(-axial,axial-speed*params.dimsDt.w));result=min(result,plug);}
  return result;
}
`;
