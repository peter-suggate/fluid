/** Sato 2018 §3.2 / Jiang 2015: quadratic MAC APIC on the virtual h grid.
 * Rows store velocity change per h-cell displacement (not per metre).
 * Thus D = I/4 and C = 4 sum(w u r^T), including anisotropic physical cells.
 * The first 12 floats retain the FLIP presentation/diagnostic ABI. */
export type NarrowBandTransfer = "flip" | "apic";
export const narrowBandParticleStride = (apic: boolean) => apic ? 28 : 16;
export const narrowBandParticleWGSL = (apic: boolean) =>
 `struct Particle{position:vec4f,velocity:vec4f,before:vec4f${apic?",cx:vec4f,cy:vec4f,cz:vec4f":""},lifetime:vec4f}`;

/** nbApicTap reads the projected/extrapolated MAC component at an h-face
 * anchor. Its boundary extension must leave the virtual stencil positions
 * intact: clamping r would invalidate D = I/4 and amplify wall noise. */
export const narrowBandApicMomentWGSL = /* wgsl */`
fn nbApicWeight(x:f32)->f32{
 let a=abs(x);if(a<0.5){return 0.75-a*a;}let b=max(0.0,1.5-a);return 0.5*b*b;
}
fn nbApicRow(p:vec3f,axis:u32)->vec4f{
 var offset=vec3f(0.5);offset[axis]=1.0;
 let base=vec3i(floor(p-offset-0.5));var v=0.0;var c=vec3f(0);
 for(var z=0;z<3;z++){for(var y=0;y<3;y++){for(var x=0;x<3;x++){
  let index=base+vec3i(x,y,z);let r=vec3f(index)+offset-p;
  let w=nbApicWeight(r.x)*nbApicWeight(r.y)*nbApicWeight(r.z);
  if(w>0.0){let u=nbApicTap(index,axis);v+=w*u;c+=4.0*w*u*r;}
 }}}
 return vec4f(c,v);
}
`;
