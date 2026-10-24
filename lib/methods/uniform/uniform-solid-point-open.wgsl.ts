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
 * same incident-cell convention as the characteristic walk. Requires
 * umStoredCellIsOpen (umCellIsOpen of a cell of a stored tile). */
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
  else if(umStoredCellIsOpen(c,t)){return false;}
 }
 return true;
}
`;

/** A characteristic starting on an aligned solid face keeps its tangential
 * motion. Clamp only components directed into the solid, as the exterior
 * domain clamp does. Probe incident cells at the starting point so corners
 * treat every wall independently; the subsequent walk still checks the
 * entire segment for other obstacles.
 *
 * umSlideWalk runs the slide's probes (per sliding axis ahead, then behind
 * only where ahead is closed), then by mode: 0 nothing, 1 the slid end, 2
 * every half-cell step of the walk to it (umWalk), stopping at the first
 * closed one. It is one loop with one umOpenAt site: Metal inlines the
 * solid geometry at every call site, and compile time follows the copies. */
export const uniformSolidSlideStartWGSL=/* wgsl */`
struct UMSlideWalk{end:vec3f,previous:vec3f,closed:bool}
fn umSlideWalk(p:vec3f,destination:vec3f,mode:u32)->UMSlideWalk{
 var end=destination;var axes=0u;var count=0u;
 for(var axis=0u;axis<3u;axis++){
  if(p[axis]==floor(p[axis])&&destination[axis]!=p[axis]){axes|=axis<<(2u*count);count++;}
 }
 let probes=2u*count;var aheadClosed=false;var steps=0u;var previous=p;
 for(var j=0u;;j++){
  let sliding=j<probes;let behind=(j&1u)!=0u;var axis=0u;var x=end;
  if(sliding){
   axis=(axes>>(2u*(j>>1u)))&3u;
   if(behind&&!aheadClosed){continue;}
   var step=vec3f(0);step[axis]=select(-0.25,0.25,destination[axis]>p[axis]);
   x=select(p+step,p-step,behind);
  }else{
   if(j==probes){
    if(mode==0u){break;}
    steps=select(1u,max(1u,u32(ceil(2.0*max(abs(end.x-p.x),max(abs(end.y-p.y),abs(end.z-p.z)))))),mode==2u);
   }
   let s=j-probes+1u;if(s>steps){break;}
   if(mode==2u){x=mix(p,end,f32(s)/f32(steps));}
  }
  let open=umOpenAt(x);
  if(sliding){if(!behind){aheadClosed=!open;}else if(open){end[axis]=p[axis];}}
  else{if(!open){return UMSlideWalk(end,previous,true);}previous=x;}
 }
 return UMSlideWalk(end,previous,false);
}
fn umSlideStart(p:vec3f,end:vec3f)->vec3f{return umSlideWalk(p,end,0u).end;}
`;
/** Resolve a point exactly on an aligned high wall into the open-side cell.
 * Its interpolation weights are then evaluated at the cell's high face,
 * exactly as low-wall points use the first open cell's low face. */
export const uniformSolidPointCellWGSL=/* wgsl */`
// The base cell, else the slid cell, else the base: one umCellIsOpen site.
fn umOpenPointCell(q:vec3f)->vec3i{
 let base=vec3i(floor(q));var cell=base;
 for(var j=0u;j<2u;j++){
  if(umCellIsOpen(cell)){return cell;}
  if(j==1u){break;}
  let probe=umSlideStart(q,q+vec3f(0.25));
  for(var axis=0u;axis<3u;axis++){if(probe[axis]==q[axis]){cell[axis]-=1;}}
 }
 return base;
}
`;
