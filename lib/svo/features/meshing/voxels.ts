/** Greedy voxel faces and voxel LOD, with an optional clipped-cell hook. */
export function svoVoxelMeshWGSL(errorFlags:number, contours:boolean):string { return /* wgsl */ `
// Greedy rectangles over one face layer's exposed-cell mask. \`m\` cells per
// side, each \`cell\` lattice units wide; the mask is consumed as it is merged.
fn meshEmitMask(mask:ptr<function,array<u32,64>>,m:u32,base:vec3u,cell:u32,face:u32,layer:u32,packedFace:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
    let identity=(*mask)[x+y*m];if(identity==0u){continue;}
    let agreement=meshMaskAgreement[x+y*m];
    var width=1u;loop{if(x+width>=m){break;}if((*mask)[x+width+y*m]!=identity||meshMaskAgreement[x+width+y*m]!=agreement){break;}width+=1u;}
    var height=1u;loop{if(y+height>=m){break;}var same=true;for(var k=0u;k<width;k+=1u){if((*mask)[x+k+(y+height)*m]!=identity||meshMaskAgreement[x+k+(y+height)*m]!=agreement){same=false;}}
      if(!same){break;}height+=1u;}
    for(var j=0u;j<height;j+=1u){for(var k=0u;k<width;k+=1u){(*mask)[x+k+(y+j)*m]=0u;}}
    var origin=base;origin[axis]+=(layer+select(0u,1u,(face&1u)!=0u))*cell;origin[u]+=x*cell;origin[v]+=y*cell;
    var extent=vec3u(0u);extent[u]=width*cell;extent[v]=height*cell;meshAppend(origin,extent,packedFace|(agreement<<11u),identity);
  }}
}
// Neighbour coverage at mixed-resolution boundaries is resolved recursively,
// including partially empty fine children. A DFS needs at most 3*depth+1 slots.
fn meshBoundary(origin:vec3u,size:u32,face:u32,identity:u32,packedFace:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  var pending:array<vec4u,64>;var count=1u;pending[0]=vec4u(origin,size);
  loop {
    if(count==0u){break;}
    count-=1u;let item=pending[count];let o=item.xyz;let s=item.w;
    var p=vec3f(o);p[u]+=f32(s)*0.5;p[v]+=f32(s)*0.5;
    p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
    let neighbour=meshRegionAt(p);
    let fits=f32(o[u])>=neighbour.origin[u]&&f32(o[v])>=neighbour.origin[v]
      &&f32(o[u]+s)<=neighbour.origin[u]+neighbour.size
      &&f32(o[v]+s)<=neighbour.origin[v]+neighbour.size;
    if(fits||s==1u){
      if((!sceneIdentitySolid(neighbour.identity)||neighbour.contour!=0u)){var extent=vec3u(0u);extent[u]=s;extent[v]=s;meshAppend(o,extent,packedFace,identity);}
    }else{
      if(count+4u>64u){atomicOr(&meshState[${errorFlags}],2u);break;}
      let half=s/2u;
      for(var child=0u;child<4u;child+=1u){var q=o;q[u]+=(child&1u)*half;q[v]+=((child>>1u)&1u)*half;pending[count]=vec4u(q,half);count+=1u;}
    }
  }
}
// One coarse level of one brick in a single invocation: every cell's material
// is derived once from the brick's own voxels, then all six faces' layers are
// masked from that table. Within the brick both sides of a face use the same
// dilated cells; across a brick boundary the face hides only behind complete
// coverage, so mixed levels between neighbours stay watertight. Exposure is a
// property of the cell, so it is decided from the table, but the identity a
// quad carries is the exposed face's own mean normal, read from the cell's
// voxels once per exposed face; an interior cell reads none.
fn meshBuildLevel(payload:u32,base:vec3u,scale:u32,depth:u32,level:u32,n:u32){
  let m=n>>level;let cell=scale<<level;
  var cells:array<u32,64>;
  for(var i=0u;i<m*m*m;i+=1u){cells[i]=meshCellMaterial(payload,vec3u(i%m,(i/m)%m,i/(m*m)),level,n);}
  let packedBase=meshPackFace(0u,level,depth);
  for(var face=0u;face<6u;face+=1u){
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let positive=(face&1u)!=0u;
    for(var layer=0u;layer<m;layer+=1u){
      var mask:array<u32,64>;
      for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
        let index=x+y*m;mask[index]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
        if(cells[c.x+c.y*m+c.z*m*m]==0u){continue;}
        let boundary=select(layer==0u,layer==m-1u,positive);
        var exposed=false;
        if(boundary){
          var p=vec3f(base+c*cell);p[u]+=f32(cell)*0.5;p[v]+=f32(cell)*0.5;
          p[axis]+=select(-0.25,f32(cell)+0.25,positive);
          exposed=!meshNeighbourCovered(p,cell);
        }else{
          var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,positive));
          exposed=cells[adjacent.x+adjacent.y*m+adjacent.z*m*m]==0u;
        }
        if(exposed){mask[index]=meshFaceIdentity(payload,c,level,n,face);meshMaskAgreement[index]=meshFaceAgreement;}
      }}
      meshEmitMask(&mask,m,base,cell,face,layer,packedBase|face);
    }
  }
}
// One face layer or one coarse level of one brick.
fn meshVoxelExtractJob(leafIndex:u32,local:u32){
  let n=dry.mapping.brickSize;
  let leaf=svoLeafLoad(leafIndex).topology;
  let node=svoNodeLoad(leaf.x);
  let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*n);
  if(local>=6u*n){meshBuildLevel(leaf.y,base,scale,node.address.z,local-6u*n+1u,n);return;}
  // Face/layer masks are independent: each invocation owns exactly one.
  // Greedy rectangles and mixed-level boundary subdivision are unchanged;
  // only append order within a brick's range may differ.
  let face=local/n;let layer=local%n;
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let packedFace=meshPackFace(face,0u,node.address.z);
  var mask:array<u32,64>;
  for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
    let m=x+y*n;mask[m]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
    let voxel=svoBrickVoxelIndex(leaf.y,c,n);if(voxel>=dryVoxelCapacity()){continue;}
    let identity=meshMergeIdentity(sceneIdentityAt(voxel));if(!sceneIdentitySolid(identity)){continue;}
    ${contours ? `let code=meshContourCode(voxel);
    if(code!=0u){
      // The first face job for this cell emits its entire closed clipped cube.
      if(face==0u){meshEmitContour(base+c*scale,scale,node.address.z,sceneIdentityAt(voxel),code);}
      continue;
    }` : ""}
    var origin=base+c*scale;origin[axis]+=select(0u,scale,(face&1u)!=0u);
    let boundary=select(layer==0u,layer==n-1u,(face&1u)!=0u);
    if(boundary){
      var p=vec3f(origin);p[u]+=f32(scale)*0.5;p[v]+=f32(scale)*0.5;p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
      let neighbour=meshRegionAt(p);
      let fits=f32(origin[u])>=neighbour.origin[u]&&f32(origin[v])>=neighbour.origin[v]
        &&f32(origin[u]+scale)<=neighbour.origin[u]+neighbour.size&&f32(origin[v]+scale)<=neighbour.origin[v]+neighbour.size;
      if(fits){if((!sceneIdentitySolid(neighbour.identity)||neighbour.contour!=0u)){mask[m]=identity;}}
      else{meshBoundary(origin,scale,face,identity,packedFace);}
      continue;
    }
    var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,(face&1u)!=0u));
    let other=svoBrickVoxelIndex(leaf.y,adjacent,n);
    if(!sceneIdentitySolid(sceneIdentityAt(other))||meshContourCode(other)!=0u){mask[m]=identity;}
  }}
  meshEmitMask(&mask,n,base,scale,face,layer,packedFace);
}
`; }
