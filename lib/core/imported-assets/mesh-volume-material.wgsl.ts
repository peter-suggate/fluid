/** The same trilinear cell as the distance/normal sampler, exposing material
 * samples without electing a single ID. Air contributes no optical attribute;
 * the consumer renormalizes the remaining valid sample weights. */
export function meshVolumeMaterialWeightsWGSL(name: string, load: string): string {
  return /* wgsl */ `
struct ${name}Value { ids:array<u32,8>, weights:array<f32,8> }
fn ${name}(base:u32,p:vec3f)->${name}Value{
  var result:${name}Value;
  let origin=vec3f(bitcast<f32>(${load}(base)),bitcast<f32>(${load}(base+1u)),bitcast<f32>(${load}(base+2u)));
  let h=bitcast<f32>(${load}(base+3u));
  let dims=vec3u(${load}(base+4u),${load}(base+5u),${load}(base+6u));
  let q=(p-origin)/h;
  if(any(q<vec3f(0))||any(q>vec3f(dims))){return result;}
  let cell=min(vec3u(floor(q)),dims-vec3u(1));let b=cell>>vec3u(3);
  let bd=vec3u(${load}(base+8u),${load}(base+9u),${load}(base+10u));
  let relative=${load}(base+16u+2u*(b.x+bd.x*(b.y+bd.y*b.z)));
  if(relative==0u){return result;}
  let c=cell&vec3u(7);let f=q-vec3f(cell);
  for(var i=0u;i<8u;i++){
    let corner=vec3u(i&1u,(i>>1u)&1u,i>>2u);let n=c+corner;
    let at=base+relative+2u*(n.x+9u*(n.y+9u*n.z));
    let material=${load}(at+1u)>>16u;
    let weights=select(vec3f(1)-f,f,corner!=vec3u(0));
    result.ids[i]=material;
    result.weights[i]=select(0.0,weights.x*weights.y*weights.z,material!=0u);
  }
  return result;
}
`;
}
