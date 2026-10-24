/** A point on a face belongs to both incident cells. Testing only floor(q)
 * pins high-side wall contact while allowing the reflected low-side contact
 * to slide. No tolerance: points strictly inside a closed cell stay closed.
 * Requires umCellIsOpen, which also rejects non-resident solid pages.
 * k=0 is floor(q) itself: one umCellIsOpen site, because Metal inlines the
 * solid geometry into every call site and compile time follows the copies. */
export const uniformSolidPointOpenWGSL=/* wgsl */`
fn umOpenAt(q:vec3f)->bool{
 let base=vec3i(floor(q));
 let on=q==floor(q);let faces=u32(on.x)|(u32(on.y)<<1u)|(u32(on.z)<<2u);
 for(var k=0u;k<=faces;k++){
  if((k&faces)!=k){continue;}
  if(umCellIsOpen(base-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u)))){return true;}
 }
 return false;
}
`;

/** The surface gradient uses capacity at the owning resolution. Keep the
 * same incident-cell convention as the characteristic walk. */
export const uniformSolidSampleClosedWGSL=/* wgsl */`
fn umSampleClosed(q:vec3f)->bool{
 if(!umSolidEnabled()){return false;}
 let base=vec3i(floor(q));let on=q==floor(q);
 let faces=u32(on.x)|(u32(on.y)<<1u)|(u32(on.z)<<2u);
 for(var k=0u;k<=faces;k++){
  if((k&faces)!=k){continue;}
  let c=base-vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let tile=c>>vec3u(2u);let t=umTileKey(tile);
  if(!umKeyStored(t)){if(umTileOpenAt(tile)>1e-5){return false;}}
  else if(umTileWidth(t)==4u){if(umTileOpen(t)>1e-5){return false;}}
  else if(umCellIsOpen(c)){return false;}
 }
 return true;
}
`;

/** A characteristic starting on an aligned solid face keeps its tangential
 * motion. Clamp only components directed into the solid, as the exterior
 * domain clamp does. Probe incident cells at the starting point so corners
 * treat every wall independently; the subsequent walk still checks the
 * entire segment for other obstacles. Probes run as one flat loop (ahead,
 * then behind only when ahead is closed) so umOpenAt has a single site. */
export const uniformSolidSlideStartWGSL=/* wgsl */`
fn umSlideStart(p:vec3f,end:vec3f)->vec3f{
 var q=end;var axes=0u;var count=0u;
 for(var axis=0u;axis<3u;axis++){
  if(p[axis]==floor(p[axis])&&end[axis]!=p[axis]){axes|=axis<<(2u*count);count++;}
 }
 var aheadClosed=false;
 for(var j=0u;j<2u*count;j++){
  let axis=(axes>>(2u*(j>>1u)))&3u;let behind=(j&1u)!=0u;
  if(behind&&!aheadClosed){continue;}
  var step=vec3f(0);step[axis]=select(-0.25,0.25,end[axis]>p[axis]);
  let open=umOpenAt(select(p+step,p-step,behind));
  if(!behind){aheadClosed=!open;}else if(open){q[axis]=p[axis];}
 }
 return q;
}
`;
/** Resolve a point exactly on an aligned high wall into the open-side cell.
 * Its interpolation weights are then evaluated at the cell's high face,
 * exactly as low-wall points use the first open cell's low face. */
export const uniformSolidPointCellWGSL=/* wgsl */`
fn umOpenPointCell(q:vec3f)->vec3i{
 let base=vec3i(floor(q));if(umCellIsOpen(base)){return base;}
 let probe=umSlideStart(q,q+vec3f(0.25));var cell=base;
 for(var axis=0u;axis<3u;axis++){if(probe[axis]==q[axis]){cell[axis]-=1;}}
 return select(base,cell,umCellIsOpen(cell));
}
`;
