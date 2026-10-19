/** Sampler shared by every host of the field-program arena. Pointers within a
 * volume are relative to its header, so the same image works in both arenas.
 * Marching uses the scalar entry point; only an accepted surface decodes PBR. */
export function meshVolumeWGSL(name: string, load: string): string {
  const sampler = (distanceOnly: boolean) => /* wgsl */ `
fn ${distanceOnly ? `${name}Distance` : name}(base:u32,p:vec3f)->${distanceOnly ? "f32" : `${name}Value`}{
  let origin=vec3f(bitcast<f32>(${load}(base)),bitcast<f32>(${load}(base+1u)),bitcast<f32>(${load}(base+2u)));
  let h=bitcast<f32>(${load}(base+3u));
  let dims=vec3u(${load}(base+4u),${load}(base+5u),${load}(base+6u));
  let band=bitcast<f32>(${load}(base+7u));
  ${distanceOnly ? "" : `var result=${name}Value(band,vec3f(0,1,0),0u);`}
  let q=(p-origin)/h;
  if(any(q<vec3f(0))||any(q>vec3f(dims))){${distanceOnly ? "return" : "result.distance="} max(band,length(max(max(origin-p,p-(origin+vec3f(dims)*h)),vec3f(0))));${distanceOnly ? "" : "return result;"}}
  let cell=min(vec3u(floor(q)),dims-vec3u(1));let b=cell>>vec3u(3);
  let bd=vec3u(${load}(base+8u),${load}(base+9u),${load}(base+10u));
  let directory=base+16u+2u*(b.x+bd.x*(b.y+bd.y*b.z));
  let relative=${load}(directory);
  if(relative==0u){${distanceOnly ? "return" : "result.distance="} bitcast<f32>(${load}(directory+1u));${distanceOnly ? "" : "return result;"}}
  let c=cell&vec3u(7);let f=q-vec3f(cell);var value=0.0;${distanceOnly ? "" : "var normal=vec3f(0);var ids:array<u32,8>;var votes:array<f32,8>;"}
  for(var i=0u;i<8u;i++){
    let corner=vec3u(i&1u,(i>>1u)&1u,i>>2u);let n=c+corner;
    let at=base+relative+2u*(n.x+9u*(n.y+9u*n.z));
    let d=bitcast<f32>(${load}(at));${distanceOnly ? "" : `let identity=${load}(at+1u);let m=identity>>16u;`}
    let weights=select(vec3f(1)-f,f,corner!=vec3u(0));let w=weights.x*weights.y*weights.z;
    value+=w*d;
    ${distanceOnly ? "" : `ids[i]=m;votes[i]=w;
    if(m!=0u){normal+=w*${name}Normal(identity);}`}
  }
  ${distanceOnly ? "return value;" : `var best=-1.0;
  // Constant indices avoid dynamic local-array access in material election.
  // Preserve the original corner order, addition order and first-wins ties.
  ${Array.from({ length: 8 }, (_, i) => `
  if(ids[${i}]!=0u){
    var vote=0.0;
    ${Array.from({ length: 8 }, (_, j) => `if(ids[${j}]==ids[${i}]){vote+=votes[${j}];}`).join("\n    ")}
    if(vote>best){best=vote;result.material=ids[${i}];}
  }`).join("\n")}
  result.distance=value;if(dot(normal,normal)>1e-12){result.normal=normalize(normal);}return result;`}
}
`;
  return /* wgsl */ `
struct ${name}Value { distance:f32, normal:vec3f, material:u32 }
fn ${name}Normal(word:u32)->vec3f{
  let p=vec2f(f32(word&255u),f32((word>>8u)&255u))/255.0*2.0-1.0;
  var n=vec3f(p,1.0-abs(p.x)-abs(p.y));
  if(n.z<0.0){n=vec3f((1.0-abs(p.yx))*select(vec2f(-1),vec2f(1),p>=vec2f(0)),n.z);}
  return normalize(n);
}
${sampler(true)}
${sampler(false)}
`;
}
