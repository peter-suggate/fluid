/** The deposited light map describes a heightfield under the water. A dry
 * screen pixel can supply colour from an occluding wall, but that wall has
 * no area in the floor projection on which to collect the deposit. */
export const waterCausticReceiverWGSL = /* wgsl */ `
fn waterCausticTexel()->f32{
  let pitch=u.container.xz/vec2f(textureDimensions(causticMap));
  return max(pitch.x,pitch.y);
}
fn floorCaustic(xz:vec2f,weight:f32)->vec3f{
  if(weight<=0.0){return vec3f(1);}
  let uv=.5+vec2f(xz.x,-xz.y)/u.container.xz;
  if(any(uv<vec2f(0))||any(uv>vec2f(1))){return vec3f(1);}
  let sample=textureSampleLevel(causticMap,linearSampler,uv,0);
  if(sample.a<=1e-4){return vec3f(1);}
  // RGB is premultiplied by coverage; excess coverage is multiple bundles
  // landing at the same point. Add the uncovered share of ordinary light.
  let modulation=max(sample.rgb,vec3f(0))+1.0-clamp(sample.a,0.0,1.0);
  return mix(vec3f(1),modulation,weight*waterCausticStrength());
}
fn dryReceiverCaustic(world:vec3f)->vec3f{
  let height=waterReceiverHeight(world.x,world.z);
  let texel=waterCausticTexel();
  let matched=1.0-smoothstep(texel,2.0*texel,abs(world.y-height));
  return floorCaustic(world.xz,matched);
}
fn solidReceiverCaustic(world:vec3f,normal:vec3f)->vec3f{
  if(waterCausticStrength()<=0.0){return vec3f(1);}
  let height=waterReceiverHeight(world.x,world.z);
  let texel=waterCausticTexel();
  let offset=world.y-height;
  let matched=1.0-smoothstep(texel,2.0*texel,abs(offset));
  // On the mapped terrain the deposit already accounts for its slope. For
  // displaced solid floors, follow the mean refracted light to the reference
  // height and weight by projected area. Vertical walls collect no floor flux.
  let light=refract(-waterAuthoredKeyDirection(),vec3f(0,1,0),1.0/waterIndexOfRefraction());
  let xz=world.xz+light.xz*(offset/max(-light.y,1e-4));
  let area=abs(normal.y);
  return floorCaustic(xz,mix(area,1.0,matched));
}
`;
