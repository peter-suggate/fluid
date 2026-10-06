/** Canonical MAC patches for the ungraded h/4h mixed Uniform grid. A
 * geometric face has one patch at equal/coarser neighbours and
 * (owner/neighbour width)^2 at a finer one: sixteen where a 4h owner meets h.
 * Both incident cells derive the same anchor; no incidence buffer is needed.
 * Coordinates use native positive-MAC indexing (normal coordinate plane-1).
 * The negative domain boundary continues to use the native boundary planes.
 */
export const uniformMixedFacesWGSL = /* wgsl */ `
struct UMFace {
 neighbor:UMOwner,
 anchor:vec3i,
 width:u32,
 count:u32,
 axis:u32,
 sign:i32,
}
fn umOwnerAt(p:vec3i)->UMOwner {
 if(any(p<vec3i(0))||any(p>=vec3i(UM_D))){return UMOwner();}
 let q=vec3u(p);let tile=umTileAt(q/4u);let width=umTileWidth(tile);
 let local=(q%4u)/width;let n=4u/width;let lane=local.x+n*(local.y+n*local.z);
 return UMOwner(tile,lane,width,(umTopology[tile]&0x3fffffffu)+lane);
}
// A first patch is also the compiled face link: all parts share its
// neighbor tile word, width and owner base. Resolve that link once, then
// derive every child patch arithmetically without a second owner lookup.
fn umFaceFirst(owner:UMOwner,axis:u32,sign:i32)->UMFace {
 let origin=vec3i(umOrigin(owner));var probe=origin;
 probe[axis]+=select(-1,i32(owner.width),sign>0);
 var neighbor=UMOwner();
 if(all(probe>=vec3i(0))&&all(probe<vec3i(UM_D))){
  // Fine interior faces never leave their tile or need another tile word.
  if(owner.width==1u&&all(vec3u(probe)/4u==umTileCoord(owner.tile))){
   let stride=select(select(16,4,axis==1u),1,axis==0u);
   neighbor=UMOwner(owner.tile,u32(i32(owner.lane)+sign*stride),1u,u32(i32(owner.index)+sign*stride));
  }else{neighbor=umOwnerAt(probe);}
 }
 let width=min(owner.width,select(owner.width,neighbor.width,neighbor.width!=0u));
 let side=owner.width/width;var anchor=probe;anchor[axis]-=select(0,1,sign>0);
 return UMFace(neighbor,anchor,width,side*side,axis,sign);
}
fn umFacePatch(first:UMFace,part:u32)->UMFace {
 if(part>=first.count){return UMFace();}
 if(part==0u){return first;}
 var face=first;let side=4u;let u=(first.axis+1u)%3u;let v=(first.axis+2u)%3u;
 face.anchor[u]+=i32(part%side);face.anchor[v]+=i32(part/side);
 // Only a 4h-to-h face has several parts. Its h neighbors all belong to
 // one tile, in the canonical x-fastest 4³ local ordering.
 let stride=vec3u(1u,4u,16u);let offset=(part%side)*stride[u]+(part/side)*stride[v];
 face.neighbor.lane+=offset;face.neighbor.index+=offset;
 return face;
}
fn umFace(owner:UMOwner,axis:u32,sign:i32,part:u32)->UMFace {
 return umFacePatch(umFaceFirst(owner,axis,sign),part);
}
fn umFaceCenter(face:UMFace)->vec3f {
 var center=vec3f(face.anchor)+vec3f(0.5*f32(face.width));
 center[face.axis]=f32(face.anchor[face.axis]+1);
 return center;
}
`;
