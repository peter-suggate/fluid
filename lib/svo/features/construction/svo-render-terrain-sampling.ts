/** Shared continuous reconstruction of the renderer's centre-sampled heights.
 * rtHeight clamps at the field edge. Neither sampling nor bounds touches physics.
 */
export const svoRenderTerrainSamplingWGSL = /* wgsl */ `
fn rtSurface(xz:vec2f)->vec3f{
  let origin=vec2f(rtFloat(RT_BASE),rtFloat(RT_BASE+1u));
  let cell=max(vec2f(rtFloat(RT_BASE+2u),rtFloat(RT_BASE+3u)),vec2f(1e-8));
  let p=(xz-origin)/cell-vec2f(0.5);let q=vec2i(floor(p));let t=fract(p);
  let a=rtHeight(q);let b=rtHeight(q+vec2i(1,0));
  let c=rtHeight(q+vec2i(0,1));let d=rtHeight(q+vec2i(1,1));
  return vec3f(mix(mix(a,b,t.x),mix(c,d,t.x),t.y),
    mix(b-a,d-c,t.y)/cell.x,mix(c-a,d-b,t.x)/cell.y);
}
// A bilinear patch attains its extrema on a rectangle at its corners. Split
// at every sample knot so a subcell crossing a patch boundary stays bounded.
fn rtSurfaceRange(minimum:vec2f,maximum:vec2f)->vec2f{
  let origin=vec2f(rtFloat(RT_BASE),rtFloat(RT_BASE+1u));
  let cell=max(vec2f(rtFloat(RT_BASE+2u),rtFloat(RT_BASE+3u)),vec2f(1e-8));
  let lo=vec2i(floor((minimum-origin)/cell-vec2f(0.5)));
  let hi=vec2i(floor((maximum-origin)/cell-vec2f(0.5)));
  if(any(hi-lo>vec2i(8))){return vec2f(-1e20,1e20);}
  var range=vec2f(1e20,-1e20);
  for(var z=lo.y;z<=hi.y;z+=1){for(var x=lo.x;x<=hi.x;x+=1){
    let a=max(minimum,origin+(vec2f(f32(x),f32(z))+vec2f(0.5))*cell);
    let b=min(maximum,origin+(vec2f(f32(x),f32(z))+vec2f(1.5))*cell);
    let h=vec4f(rtSurface(a).x,rtSurface(vec2f(a.x,b.y)).x,rtSurface(vec2f(b.x,a.y)).x,rtSurface(b).x);
    range.x=min(range.x,min(min(h.x,h.y),min(h.z,h.w)));
    range.y=max(range.y,max(max(h.x,h.y),max(h.z,h.w)));
  }}
  return range;
}
`;
