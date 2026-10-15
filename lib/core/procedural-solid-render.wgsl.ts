/** Read the same node field and edit-validity mask used by the fluid boundary.
 * The image lives in the voxelizer's maintenance arena; no extra binding or
 * scene-specific shader is needed. Missing/edited cells use the voxel field. */
export function proceduralSolidRenderWGSL(baseWords: number | string, read = "atomicLoad(&maintenance[index])", cell = "swCell()", origin = "swOrigin()", name = "swBoundary"): string {
  return /* wgsl */ `
fn ${name}Word(index:u32)->u32{return ${read};}
fn ${name}Sample(world:vec3f)->vec4f{
  let base=${typeof baseWords === "number" ? `${baseWords}u` : baseWords};
  if(base==0u){return vec4f(0,0,0,1e20);}
  let h=${cell};let q=(world-(${origin}))/h;let cell=vec3i(floor(q));
  let brick=cell>>vec3u(3);let mask=${name}Word(base);
  var slot=((u32(brick.x)*73856093u)^(u32(brick.y)*19349663u)^(u32(brick.z)*83492791u))&mask;
  for(var probe=0u;probe<=mask;probe++){
    let at=base+4u+4u*slot;let relative=${name}Word(at+3u);
    if(relative==0u){break;}
    let key=vec3i(bitcast<i32>(${name}Word(at)),bitcast<i32>(${name}Word(at+1u)),bitcast<i32>(${name}Word(at+2u)));
    if(all(key==brick)){
      let record=base+relative;let c=vec3u(cell)&vec3u(7);let i=c.x+8u*(c.y+8u*c.z);
      if((${name}Word(record+(i>>5u))&(1u<<(i&31u)))==0u){break;}
      let f=q-vec3f(cell);var result=vec4f(0);
      for(var k=0u;k<8u;k++){
        let b=vec3u(k&1u,(k>>1u)&1u,k>>2u);let n=c+b;let w=select(1.0-f,f,b!=vec3u(0));
        let v=bitcast<f32>(${name}Word(record+16u+n.x+9u*(n.y+9u*n.z)));
        let sign=select(vec3f(-1),vec3f(1),b!=vec3u(0));
        result+=v*vec4f(sign.x*w.y*w.z,sign.y*w.x*w.z,sign.z*w.x*w.y,w.x*w.y*w.z);
      }
      let unit=min(h.x,min(h.y,h.z));return result*vec4f(vec3f(unit)/h,unit);
    }
    slot=(slot+1u)&mask;
  }
  return vec4f(0,0,0,1e20);
}
// Meshing joins the sampled field to a bounded voxel-density field. Use
// the same plateau on both sides of a sparse-brick boundary; otherwise the
// graph fitter can extrapolate a positive distance jump into negative air.
// Keep Sample unbounded: normal stencils still need the original gradients.
fn ${name}Field(world:vec3f,limit:f32)->f32{
  let value=${name}Sample(world).w;
  if(value>=1e19){return value;}
  return clamp(value,-limit,limit);
}
// Central differences over one node interval blend the piecewise trilinear
// gradients continuously. They preserve affine faces and remove lattice bands
// on a curved surface without changing its position or the solver boundary.
fn ${name}Normal(world:vec3f,centre:vec4f)->vec3f{
  let e=.5*(${cell});var gradient=centre.xyz;
  for(var axis=0u;axis<3u;axis++){
    var delta=vec3f(0);delta[axis]=e[axis];
    let a=${name}Sample(world-delta);let b=${name}Sample(world+delta);
    if(a.w<1e19&&b.w<1e19){gradient[axis]=(b.w-a.w)/(2.0*e[axis]);}
  }
  return gradient;
}
`;
}
