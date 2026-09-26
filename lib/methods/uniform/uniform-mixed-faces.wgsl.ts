/** Canonical MAC patches for the strongly graded Uniform grid. A geometric
 * face has one patch at equal/coarser neighbours, four at a finer neighbour.
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
fn umFace(owner:UMOwner,axis:u32,sign:i32,part:u32)->UMFace {
 let origin=vec3i(umOrigin(owner));var probe=origin;
 probe[axis]+=select(-1,i32(owner.width),sign>0);
 var neighbor=umOwnerAt(probe);
 let width=min(owner.width,select(owner.width,neighbor.width,neighbor.width!=0u));
 let side=owner.width/width;let count=side*side;
 if(part>=count){return UMFace();}
 let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 probe[u]+=i32((part%side)*width);probe[v]+=i32((part/side)*width);
 if(part!=0u){neighbor=umOwnerAt(probe);}
 var anchor=probe;anchor[axis]-=select(0,1,sign>0);
 return UMFace(neighbor,anchor,width,count,axis,sign);
}
fn umFaceCenter(face:UMFace)->vec3f {
 var center=vec3f(face.anchor)+vec3f(0.5*f32(face.width));
 center[face.axis]=f32(face.anchor[face.axis]+1);
 return center;
}
`;
