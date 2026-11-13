/** Sato 2018 §3.2 / Jiang 2015: quadratic MAC APIC on the virtual h grid.
 * Rows store velocity change per h-cell displacement (not per metre).
 * Thus D = I/4 and C = 4 sum(w u r^T), including anisotropic physical cells.
 * FLIP packs nonnegative sparse age and the ballistic bit into before.w.
 * Its sign bit is the flag (including negative zero); its magnitude keeps
 * every age bit. APIC keeps its existing affine/presentation record.
 * merge (crowdedCells "merge"): a sample's mass, a whole number of seeded
 * units from 1 to 2^massBits (8; 64 under surfaceSamples 64, uniform-narrow-band-sites.ts).
 * FLIP keeps mass - 1 in before.w's low massBits bits (the age keeps the rest
 * of its mantissa); APIC keeps it in lifetime.y. Without merge every mass is 1
 * and the record is unchanged. */
export type NarrowBandTransfer = "flip" | "apic";
export const narrowBandParticleStride = (apic: boolean) => apic ? 28 : 12;
export const narrowBandParticleWGSL = (apic: boolean, merge = false, massBits = 3) => {
 const low = (1 << massBits) - 1, ageBits = merge ? `0x${(0x7fffffff & ~low).toString(16)}u` : "0x7fffffffu";
 return `struct Particle{position:vec4f,velocity:vec4f,before:vec4f${apic?",cx:vec4f,cy:vec4f,cz:vec4f,lifetime:vec4f":""}}
fn nbParticleMode(p:Particle)->f32{return ${apic?"p.before.w":"f32(bitcast<u32>(p.before.w)>>31u)"};}
fn nbParticleAge(p:Particle)->f32{return ${apic?"p.lifetime.x":`bitcast<f32>(bitcast<u32>(p.before.w)&${ageBits})`};}
fn nbParticleState(age:f32,mode:f32)->f32{return ${apic?"mode":`bitcast<f32>((bitcast<u32>(age)&${ageBits})|(u32(mode)<<31u))`};}
fn nbParticleMass(p:Particle)->f32{return ${!merge?"1.0":apic?"p.lifetime.y+1.0":`f32((bitcast<u32>(p.before.w)&${low}u)+1u)`};}
// before.w for age and mode, keeping p's mass.
fn nbParticleRestate(p:Particle,age:f32,mode:f32)->f32{return ${merge&&!apic?`bitcast<f32>(bitcast<u32>(nbParticleState(age,mode))|(bitcast<u32>(p.before.w)&${low}u))`:"nbParticleState(age,mode)"};}
fn nbWithMass(p:Particle,mass:f32)->Particle{var r=p;${!merge?"":apic?"r.lifetime.y=mass-1.0;":`r.before.w=bitcast<f32>((bitcast<u32>(p.before.w)&0x${((0xffffffff & ~low) >>> 0).toString(16)}u)|((u32(mass)-1u)&${low}u));`}return r;}
`;
};

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
