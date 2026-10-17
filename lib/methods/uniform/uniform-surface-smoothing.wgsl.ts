/** Fourth-difference smoothing only where accepted taps contain an interface.
 * The including module supplies umWallWithin; h is the cell spacing.
 * Reuses the foot stencil, adding no texture reads or dispatches. */
export const uniformSurfaceSmoothingWGSL=/* wgsl */`
fn umSurfaceSmoothing(vertex:vec3i,wi:i32,c:f32,t:array<f32,10>,closed:u32,h:vec3f,eps:f32)->f32{
 let w=f32(wi);let sign=bitcast<u32>(c);var crossing=0u;
 // Reject a single-phase stencil before its wall walks. The per-axis
 // test below still excludes crossings supplied only by invalid taps.
 let near=bitcast<vec4u>(vec4f(t[0],t[1],t[4],t[5]));let far=bitcast<vec4u>(vec4f(t[6],t[7],t[8],t[9]));
 if(all((((near^vec4u(sign))|(far^vec4u(sign)))&vec4u(0x80000000u))==vec4u(0u))){return 0.0;}
 var sum=0.0;
 for(var axis=0u;axis<3u;axis+=2u){
  if(umWallWithin(vertex,axis,2*wi)){continue;}
  var part=6.0*c;var valid=((closed>>(1u+2u*axis))&15u)==0u;var different=0u;
  for(var k=0u;k<4u;k++){
   let s=t[select(2u*axis+k,6u+axis+k-2u,k>1u)];let span=select(w,2.0*w,k>1u);
   // A tap further from the vertex's value than its span allows is no
   // distance: past the band, far air holds a marker. Its fourth difference
   // drew the air beside the band onto the liquid every step.
   if(abs(s-c)>1.5*span*h[axis]){valid=false;}
   different|=bitcast<u32>(s)^sign;
   part+=select(-4.0,1.0,k>1u)*s;
  }
  if(valid){sum+=part;crossing|=different;}
 }
 // A fourth difference can overshoot even when every tap is air (Letters,
 // frame 94 at the back wall). That invents liquid outside the residency
 // certificate. A sign difference in an accepted stencil certifies an
 // interface; invalid distance taps cannot supply one. Comparing sign bits
 // reuses the loaded taps without keeping extra floating-point extrema live.
 // Bounding against c alone is insufficient: advection can return a smaller
 // positive value before this correction is applied.
 return select(0.0,-eps*sum,(crossing&0x80000000u)!=0u);
}
`;
