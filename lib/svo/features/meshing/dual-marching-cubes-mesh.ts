/** Marching Cubes on function-fitted dual hexahedra. A face graph replaces
 * the fixed case table so both owners use the same bilinear face decider.
 * Uses the shared 16-byte attachment reader and packed-triangle append ABI.
 */
export const svoDualMarchingCubesMeshWGSL = /* wgsl */ `
fn dmcCorner(i:u32)->vec3f{return vec3f(f32(i&1u),f32((i>>1u)&1u),f32((i>>2u)&1u));}
fn dmcEdge(axis:u32,c:u32)->u32{return axis*4u+(c&((1u<<axis)-1u))+((c>>(axis+1u))<<axis);}
fn dmcEdgeStart(edge:u32)->u32{
  let axis=edge/4u;let other=edge%4u;
  return (other&((1u<<axis)-1u))|((other>>axis)<<(axis+1u));
}
fn dmcConnect(adjacency:ptr<function,array<vec2u,12>>,a:u32,b:u32,face:u32,signs:u32){
  let ca=dmcEdgeStart(a);let cb=dmcEdgeStart(b);
  let da=dmcCorner(ca|(1u<<(a/4u)))-dmcCorner(ca);
  let db=dmcCorner(cb|(1u<<(b/4u)))-dmcCorner(cb);
  let middleA=dmcCorner(ca)+.5*da;let middleB=dmcCorner(cb)+.5*db;
  var faceNormal=vec3f(0);faceNormal[face/2u]=select(-1.,1.,(face&1u)!=0u);
  let outward=da*select(-1.,1.,(signs&(1u<<ca))!=0u);
  // Orient in cube parameter space, never from a possibly collapsed world
  // triangle. The opposite owner reverses its face normal and thus this edge.
  if(dot(middleB-middleA,cross(outward,faceNormal))>0.){(*adjacency)[a].y=b;(*adjacency)[b].x=a;}
  else{(*adjacency)[b].y=a;(*adjacency)[a].x=b;}
}
fn meshDmcExtract(base:vec3u,scale:u32,depth:u32,local:u32,n:u32){
  if(local>=n){return;}
  for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
    let vertex=vec3f(base+vec3u(x,y,local)*scale);let origin=vertex-vec3f(f32(scale));
    if(any(origin<vec3f(0))){continue;}
    var positions:array<vec3f,8>;var values:array<f32,8>;var signs=0u;var valid=true;var identity=0u;var sharp=0u;
    for(var c=0u;c<8u;c+=1u){
      let region=meshRegionAt(vertex+(dmcCorner(c)-vec3f(.5))*f32(scale));
      if(region.voxel==SVO_INVALID||region.size!=f32(scale)){valid=false;break;}
      positions[c]=meshDcPoint(region.voxel);values[c]=bitcast<f32>(meshDcWord(region.voxel,3u));
      if(values[c]==0.){sharp=1u;}
      if(values[c]<0.){signs|=1u<<c;identity=region.identity;}
    }
    if(!valid||signs==0u||signs==255u){continue;}
    var points:array<vec3f,12>;var adjacency:array<vec2u,12>;var crossingMask=0u;
    for(var edge=0u;edge<12u;edge+=1u){
      adjacency[edge]=vec2u(SVO_INVALID);let axis=edge/4u;let other=edge%4u;
      let a=(other&((1u<<axis)-1u))|((other>>axis)<<(axis+1u));let b=a|(1u<<axis);
      if((values[a]<0.)==(values[b]<0.)){continue;}
      crossingMask|=1u<<edge;
      let t=clamp(values[a]/(values[a]-values[b]),0.,1.);
      points[edge]=mix(positions[a],positions[b],t);
    }
    // Face ordering uses global axes on both sides; ties are deterministic.
    for(var face=0u;face<6u;face+=1u){
      let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let c=(face&1u)<<axis;
      let corners=array<u32,4>(c,c|(1u<<u),c|(1u<<u)|(1u<<v),c|(1u<<v));
      let edges=array<u32,4>(dmcEdge(u,c),dmcEdge(v,c|(1u<<u)),dmcEdge(u,c|(1u<<v)),dmcEdge(v,c));
      var found:array<u32,4>;var count=0u;
      for(var i=0u;i<4u;i+=1u){if((crossingMask&(1u<<edges[i]))!=0u){found[count]=edges[i];count+=1u;}}
      if(count==2u){dmcConnect(&adjacency,found[0],found[1],face,signs);}
      if(count==4u){
        let f=vec4f(values[corners[0]],values[corners[1]],values[corners[2]],values[corners[3]]);
        let q=f/max(max(abs(f.x),abs(f.y)),max(abs(f.z),abs(f.w)));
        if(q.x*q.z>=q.y*q.w){dmcConnect(&adjacency,edges[0],edges[1],face,signs);dmcConnect(&adjacency,edges[2],edges[3],face,signs);}
        else{dmcConnect(&adjacency,edges[0],edges[3],face,signs);dmcConnect(&adjacency,edges[1],edges[2],face,signs);}
      }
    }
    var visited=0u;
    for(var seed=0u;seed<12u;seed+=1u){
      if((crossingMask&(1u<<seed))==0u||(visited&(1u<<seed))!=0u){continue;}
      var ring:array<u32,12>;var count=0u;var at=seed;
      loop{
        if(at==SVO_INVALID||count==12u||(visited&(1u<<at))!=0u){break;}
        ring[count]=at;count+=1u;visited|=1u<<at;
        at=adjacency[at].y;
      }
      if(at!=seed||count<3u){continue;}
      // Zero-valued fitted vertices make several cube edges meet at exactly
      // one point. Collapse those repeated packed vertices before triangulation.
      var polygon:array<vec3f,12>;var packed:array<u32,12>;var faceMasks:array<u32,12>;var size=0u;
      for(var j=0u;j<count;j+=1u){
        let word=meshDcPack((points[ring[j]]-origin)/f32(scale));var duplicate=false;
        let edge=ring[j];let axis=edge/4u;let other=edge%4u;
        let corner=(other&((1u<<axis)-1u))|((other>>axis)<<(axis+1u));var faceMask=0u;
        for(var a=0u;a<3u;a+=1u){if(a!=axis){faceMask|=1u<<(2u*a+((corner>>a)&1u));}}
        for(var k=0u;k<size;k+=1u){if(packed[k]==word){duplicate=true;faceMasks[k]|=faceMask;}}
        if(!duplicate){packed[size]=word;faceMasks[size]=faceMask;polygon[size]=origin+vec3f(f32(word&1023u),f32((word>>10u)&1023u),f32((word>>20u)&1023u))*f32(scale)/256.;size+=1u;}
      }
      if(size<3u){continue;}
      // Use n-2 triangles unless the fan would introduce a diagonal lying on
      // a shared cube face; those cases get an interior fan instead.
      var interior=false;
      for(var j=2u;j+1u<size;j+=1u){if((faceMasks[0]&faceMasks[j])!=0u){interior=true;}}
      var centre=polygon[0];var triangles=size-2u;
      if(interior){centre=vec3f(0);for(var j=0u;j<size;j+=1u){centre+=polygon[j];}centre/=f32(size);triangles=size;}
      for(var j=0u;j<triangles;j+=1u){
        let a=centre;var b=polygon[select(j+1u,j,interior)];var c=polygon[select(j+2u,(j+1u)%size,interior)];
        let p=meshDcPack((a-origin)/f32(scale));let q=meshDcPack((b-origin)/f32(scale));let r=meshDcPack((c-origin)/f32(scale));
        if(p==q||q==r||r==p){continue;}
        meshAppend(vec3u(origin),vec3u(p,q,r),0xc0000000u|(sharp<<29u)|meshPackFace(0u,0u,depth),identity);
      }
    }
  }}
}
`;
