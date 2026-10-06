/** Fixed h MAC sites of a coarse seam tile. Geometry is an integer recipe:
 * only a positive face next to h is a stored unit patch; every other site
 * interpolates the prepared coarse cache. No owner/face discovery is needed.
 * The eighth-cell fractions are exact, including clipped boundary taps. */
export const uniformCompiledVelocityTapWGSL = /* wgsl */ `
fn umCompiledVelocityTap(tile:u32,local:vec3u,axis:u32,negativePlane:bool)->f32{
 let origin=umTileCoord(tile);var index=vec3i(origin*4u+local);
 if(negativePlane){index[axis]=-1;}
 if(!negativePlane&&local[axis]==3u&&(umFineFaceSides(tile)&(1u<<(2u*axis)))!=0u){return umLoadMixedFace(index,axis);}
 // q = p/4 - MAC offset, in eighths. Transverse coordinates are
 // tile + local/4 - 3/8; the normal coordinate is tile + local/4 - 3/4.
 var q=vec3i(origin)*8+vec3i(local)*2-vec3i(3);q[axis]-=3;
 if(negativePlane){q[axis]=-8;}
 var lower=vec3i(0);lower[axis]=-8;q=clamp(q,lower,(vec3i(UM_D/4u)-vec3i(1))*8);
 let base=q>>vec3u(3);let fraction=vec3f(q-base*8)/8.0;var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let c=umCorner(k,2u);let weights=select(vec3f(1)-fraction,fraction,c!=vec3u(0));let weight=weights.x*weights.y*weights.z;
  if(weight>0.0){terms[k]=weight*umLoadCoarseFace(base+vec3i(c),axis);}
 }
 return ((terms[0]+terms[5])+(terms[1]+terms[4]))+((terms[2]+terms[7])+(terms[3]+terms[6]));
}
`;
