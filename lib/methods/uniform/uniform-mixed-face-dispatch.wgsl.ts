/** Shared, single-writer traversal of canonical mixed MAC patches. A native
 * RGBA texel can contain patches from several axes; one owner packs them all.
 * Callers bind output (RGBA texture) and boundary (negative face buffer). */
export const uniformMixedFaceAddressWGSL = /* wgsl */ `
fn umNegativeBoundaryIndex(p:vec3u,axis:u32)->u32 {
 if(axis==0u){return p.y+UM_D.y*p.z;}
 if(axis==1u){return UM_D.y*UM_D.z+p.x+UM_D.x*p.z;}
 return UM_D.y*UM_D.z+UM_D.x*UM_D.z+p.x+UM_D.x*p.y;
}
fn umPositiveFaceAtAnchor(owner:UMOwner,axis:u32,anchor:vec3i)->UMFace {
 let origin=vec3i(umOrigin(owner));let local=anchor-origin;
 if(owner.width==1u){if(all(local==vec3i(0))){return umFace(owner,axis,1,0u);}return UMFace();}
 if(local[axis]!=i32(owner.width)-1){return UMFace();}
 let first=umFace(owner,axis,1,0u);let u=(axis+1u)%3u;let v=(axis+2u)%3u;
 if(local[u]%i32(first.width)!=0||local[v]%i32(first.width)!=0){return UMFace();}
 return umFace(owner,axis,1,u32(local[u])/first.width+(owner.width/first.width)*(u32(local[v])/first.width));
}
`;

export function uniformMixedFaceDispatchWGSL(entry: string, evaluate: string, allWidths = false, metadata = ""): string {
  return /* wgsl */ `
@compute @workgroup_size(64) fn ${entry}(@builtin(global_invocation_id) gid:vec3u){
 let owner=${allWidths ? "umAllOwner" : "umOwner"}(gid);if(owner.width==0u){return;}
 let origin=umOrigin(owner);
 for(var axis=0u;axis<3u;axis++){
  if(origin[axis]==0u){let face=umFace(owner,axis,-1,0u);boundary[umNegativeBoundaryIndex(origin,axis)]=${evaluate};}
 }
 // Unit owners pack their three positive components at the same anchor.
 // Visit that anchor once without cloning the (large) evaluator call graph.
 for(var axis=0u;axis<select(3u,1u,owner.width==1u);axis++){
  let first=umFace(owner,axis,1,0u);
  for(var part=0u;part<first.count;part++){
   let ownedFace=umFace(owner,axis,1,part);var earlier=false;
   for(var other=0u;other<axis;other++){earlier=earlier||umPositiveFaceAtAnchor(owner,other,ownedFace.anchor).width!=0u;}
   if(earlier){continue;}
   var value=vec4f(0);
   for(var other=0u;other<3u;other++){
    let face=umPositiveFaceAtAnchor(owner,other,ownedFace.anchor);
    if(face.width!=0u){value[other]=${evaluate};}
   }
   ${metadata}
   textureStore(output,ownedFace.anchor,value);
  }
 }
}`;
}
