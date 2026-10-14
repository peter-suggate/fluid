/** Fixed h MAC sites of a coarse seam tile. Geometry is an integer recipe:
 * only a positive face next to h is a stored unit patch; every other site
 * interpolates the prepared coarse cache (`coarse`, by block:
 * uniform-coarse-velocity-cache.ts). No owner/face discovery is needed, and
 * the tile's key names its cache block: no tap reads a table or clamps.
 * The eighth-cell fractions are exact; a tap past the tile's page is the
 * block's halo tile. */
export const uniformCompiledVelocityTapWGSL = /* wgsl */ `
fn umCompiledVelocityTap(tile:u32,local:vec3u,axis:u32,negativePlane:bool)->f32{
 if(!negativePlane&&local[axis]==3u&&(umFineFaceSides(tile)&(1u<<(2u*axis)))!=0u){return umLoadMixedFace(vec3i(umTileCoord(tile)*4u+local),axis);}
 // q = p/4 - MAC offset from the page's low corner, in eighths. Transverse
 // coordinates are tile + local/4 - 3/8; the normal coordinate is
 // tile + local/4 - 3/4. A negative plane is the page's own low plane (only
 // a tile on it is asked).
 var q=vec3i(umLocalTile(tile))*8+vec3i(local)*2-vec3i(3);q[axis]-=3;
 if(negativePlane){q[axis]=-8;}
 let base=q>>vec3u(3);let fraction=vec3f(q-base*8)/8.0;let at=base+umCoarseOrigin(tile>>9u);var terms:array<f32,8>;
 for(var k=0u;k<8u;k++){
  let c=umCorner(k,2u);let weights=select(vec3f(1)-fraction,fraction,c!=vec3u(0));let weight=weights.x*weights.y*weights.z;
  if(weight>0.0){terms[k]=weight*textureLoad(coarse,at+vec3i(c),0)[axis];}
 }
 return ((terms[0]+terms[5])+(terms[1]+terms[4]))+((terms[2]+terms[7])+(terms[3]+terms[6]));
}
`;
