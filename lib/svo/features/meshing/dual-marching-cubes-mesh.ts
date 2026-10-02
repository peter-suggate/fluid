import { SVO_GBUFFER_NORMAL_OCT8_ABSENT } from "../../contracts/svo-gbuffer";
/** Marching Cubes on function-fitted dual hexahedra. A face graph replaces
 * the fixed case table so both owners use the same bilinear face decider.
 * Uses the shared 16-byte attachment reader and packed-triangle append ABI.
 *
 * The dual grid is the octree's own (Schaefer & Warren): one dual cell per
 * lattice vertex that is a corner of some leaf cell, whose eight corners are
 * the fitted vertices of the leaves holding the vertex's eight octants. Where
 * leaves of different sizes meet, a coarse leaf holds several octants and the
 * hexahedron degenerates; the face graph needs no case for that, because a
 * repeated corner never changes sign along its own edge. Adjacent dual cells
 * share whole faces at every resolution, so the surface closes across a
 * refinement boundary without transition geometry.
 *
 * Each dual cell is emitted once, by the finest leaf cell at its vertex and,
 * among equally fine ones, the one in the highest octant. On a uniform grid
 * that is the cell whose minimum corner is the vertex.
 *
 * Triangle vertices are 12 bits per axis in 1/256 of the owner's cell from the
 * record origin: the low 10 bits in the extent word, the high 2 in face bits
 * 11-28 (6 per vertex). A vertex on a dual edge is first rounded to 1/256 of
 * the smaller of the edge's two leaves, the grid every dual cell sharing the
 * edge can represent, so both sides store the identical point. Leaves more
 * than four times the owner's cell exceed the 12 bits and are reported through
 * the host's \`meshDmcUnsupported\`.
 *
 * Each vertex's shading normal rides in the high 16 bits of one origin word
 * (vertex j in axis j), packed by the host's \`meshDmcPackNormal\`. A vertex
 * lies on the dual edge between a solid leaf's fit and an air leaf's, and
 * takes the solid leaf's fitted normal: a property of the edge, so every
 * triangle and every dual cell sharing the vertex stores the same one.
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
// The dual cell being assembled: the fitted vertex, value and cell size of the
// leaf in each octant of its lattice vertex.
var<private> dmcPositions:array<vec3f,8>;
var<private> dmcValues:array<f32,8>;
var<private> dmcSizes:array<f32,8>;
// The fitted normal of each octant's leaf; zero where it publishes none.
var<private> dmcNormals:array<vec3f,8>;
var<private> dmcGradients:array<vec3f,8>;
var<private> dmcGradientCount:u32;
var<private> dmcSigns:u32;
var<private> dmcIdentity:u32;
var<private> dmcSharp:u32;
fn dmcReset(){dmcGradientCount=0u;dmcSigns=0u;dmcIdentity=0u;dmcSharp=0u;}
fn dmcLoad(c:u32,voxel:u32,size:f32){
  dmcPositions[c]=meshDcPoint(voxel);dmcValues[c]=bitcast<f32>(meshDcWord(voxel,3u));dmcSizes[c]=size;
  let material=sceneIdentityAt(voxel);
  dmcNormals[c]=vec3f(0);
  // A zero fitted value is sliver elimination, not evidence of a crease.
  // Compare published field gradients instead. Smooth patches keep their
  // interpolated normals; corners with >~37 degrees of disagreement retain
  // the geometric triangle normal. Absent air normals contribute nothing.
  if(sceneIdentitySolid(material)&&sceneIdentityHasNormal(material)){
    let gradient=sceneIdentityNormal(material);
    dmcNormals[c]=gradient;
    for(var j=0u;j<dmcGradientCount;j+=1u){if(dot(gradient,dmcGradients[j])<0.8){dmcSharp=1u;}}
    dmcGradients[dmcGradientCount]=gradient;dmcGradientCount+=1u;
  }
  if(dmcValues[c]<0.){dmcSigns|=1u<<c;dmcIdentity=material;}
}
fn dmcQuantize(p:vec3f,origin:vec3f,scale:f32,limit:f32)->vec3u{
  return vec3u(round(clamp((p-origin)/scale*256.,vec3f(0),vec3f(limit))));
}
fn dmcLow(q:vec3u)->u32{return (q.x&1023u)|((q.y&1023u)<<10u)|((q.z&1023u)<<20u);}
fn dmcHigh(q:vec3u)->u32{return (q.x>>10u)|((q.y>>10u)<<2u)|((q.z>>10u)<<4u);}
fn dmcNormalWord(n:vec3f)->u32{
  if(dot(n,n)<1e-12){return ${SVO_GBUFFER_NORMAL_OCT8_ABSENT}u;}
  return meshDmcPackNormal(normalize(n))&0xffffu;
}
// Polygonise the loaded dual cell. \`mixed\` cells hold leaves coarser than the
// owner and use the 12-bit range from \`origin\`.
fn dmcEmit(origin:vec3f,scale:u32,depth:u32,mixed:bool){
  let signs=dmcSigns;
  if(signs==0u||signs==255u){return;}
  // The origin words keep their high halves for the vertex normals.
  if(any(origin>=vec3f(65536.))){meshDmcUnsupported();return;}
  let s=f32(scale);let limit=select(1023.,4095.,mixed);
  var points:array<vec3f,12>;var normals:array<vec3f,12>;var adjacency:array<vec2u,12>;var crossingMask=0u;
  for(var edge=0u;edge<12u;edge+=1u){
    adjacency[edge]=vec2u(SVO_INVALID);let axis=edge/4u;let other=edge%4u;
    let a=(other&((1u<<axis)-1u))|((other>>axis)<<(axis+1u));let b=a|(1u<<axis);
    if((dmcValues[a]<0.)==(dmcValues[b]<0.)){continue;}
    crossingMask|=1u<<edge;
    let t=clamp(dmcValues[a]/(dmcValues[a]-dmcValues[b]),0.,1.);
    var p=mix(dmcPositions[a],dmcPositions[b],t);
    if(mixed){let m=min(dmcSizes[a],dmcSizes[b]);p=round(p*(256./m))*(m/256.);}
    points[edge]=p;
    // The solid end's normal, unless the vertex sits on the other end's fit
    // (a zero value): that leaf is then the one every sharer of the vertex sees.
    var n=dmcNormals[select(b,a,dmcValues[a]<0.)];
    let on=dmcNormals[select(b,a,t<=0.)];
    if((t<=0.||t>=1.)&&dot(on,on)>0.){n=on;}
    normals[edge]=n;
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
      let f=vec4f(dmcValues[corners[0]],dmcValues[corners[1]],dmcValues[corners[2]],dmcValues[corners[3]]);
      let q=f/max(max(abs(f.x),abs(f.y)),max(abs(f.z),abs(f.w)));
      if(q.x*q.z>=q.y*q.w){dmcConnect(&adjacency,edges[0],edges[1],face,signs);dmcConnect(&adjacency,edges[2],edges[3],face,signs);}
      else{dmcConnect(&adjacency,edges[0],edges[3],face,signs);dmcConnect(&adjacency,edges[1],edges[2],face,signs);}
    }
  }
  let faceBase=0xc0000000u|(dmcSharp<<29u)|meshPackFace(0u,0u,depth);
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
    // Zero-valued fitted vertices, and the repeated corners of a mixed cell,
    // make several cube edges meet at exactly one point. Collapse those
    // repeated packed vertices before triangulation.
    var polygon:array<vec3f,12>;var packed:array<vec3u,12>;var shading:array<vec3f,12>;var faceMasks:array<u32,12>;var size=0u;
    for(var j=0u;j<count;j+=1u){
      let word=dmcQuantize(points[ring[j]],origin,s,limit);var duplicate=false;
      let edge=ring[j];let axis=edge/4u;let other=edge%4u;
      let corner=(other&((1u<<axis)-1u))|((other>>axis)<<(axis+1u));var faceMask=0u;
      for(var a=0u;a<3u;a+=1u){if(a!=axis){faceMask|=1u<<(2u*a+((corner>>a)&1u));}}
      for(var k=0u;k<size;k+=1u){if(all(packed[k]==word)){duplicate=true;faceMasks[k]|=faceMask;}}
      if(!duplicate){packed[size]=word;shading[size]=normals[edge];faceMasks[size]=faceMask;polygon[size]=origin+vec3f(word)*s/256.;size+=1u;}
    }
    if(size<3u){continue;}
    // Use n-2 triangles unless the fan would introduce a diagonal lying on
    // a shared cube face; those cases get an interior fan instead.
    var interior=false;
    for(var j=2u;j+1u<size;j+=1u){if((faceMasks[0]&faceMasks[j])!=0u){interior=true;}}
    var p=packed[0];var pn=shading[0];var triangles=size-2u;
    if(interior){var centre=vec3f(0);pn=vec3f(0);for(var j=0u;j<size;j+=1u){centre+=polygon[j];pn+=shading[j];}p=dmcQuantize(centre/f32(size),origin,s,limit);triangles=size;}
    for(var j=0u;j<triangles;j+=1u){
      let qi=select(j+1u,j,interior);let ri=select(j+2u,(j+1u)%size,interior);
      let q=packed[qi];let r=packed[ri];
      if(all(p==q)||all(q==r)||all(r==p)){continue;}
      meshAppend(vec3u(origin)|(vec3u(dmcNormalWord(pn),dmcNormalWord(shading[qi]),dmcNormalWord(shading[ri]))<<vec3u(16u)),vec3u(dmcLow(p),dmcLow(q),dmcLow(r)),
        faceBase|(dmcHigh(p)<<11u)|(dmcHigh(q)<<17u)|(dmcHigh(r)<<23u),dmcIdentity);
    }
  }
}
// Emit a mixed cell whose largest leaf is \`largest\`, or report a grading the
// 12-bit record cannot hold.
fn dmcEmitMixed(vertex:vec3f,scale:u32,depth:u32,largest:f32){
  if(largest>4.*f32(scale)){meshDmcUnsupported();return;}
  dmcEmit(max(vertex-vec3f(largest),vec3f(0)),scale,depth,true);
}
fn meshDmcExtract(base:vec3u,scale:u32,depth:u32,local:u32,n:u32){
  if(local>=n){return;}
  let s=f32(scale);let span=f32(n*scale);
  // Uniform dual grids have contiguous payloads within a brick. Resolve the
  // eight incident bricks once per slice instead of walking the tree eight
  // times for each cell. Only an octant whose brick is not this brick's size
  // is looked up on its own.
  var bricks:array<MeshRegion,8>;
  for(var c=0u;c<8u;c+=1u){
    bricks[c]=meshRegionAt(vec3f(base)+(dmcCorner(c)-vec3f(1))*span+vec3f(.5*s));
  }
  let own=bricks[7];
  if(own.voxel==SVO_INVALID||own.size!=s){return;}
  // Vertices at a cell's minimum corner: this cell holds the highest octant,
  // so it owns the dual cell unless a finer leaf is present.
  for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
    let vertex=vec3f(base+vec3u(x,y,local)*scale);
    if(any(vertex<vec3f(s))){continue;}
    dmcReset();var valid=true;var largest=s;
    for(var c=0u;c<8u;c+=1u){
      let q=vec3i(vec3u(x,y,local))+vec3i(dmcCorner(c))-vec3i(1);
      let upper=q>=vec3i(0);
      var region=bricks[select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z)];
      if(region.voxel!=SVO_INVALID&&region.size==s){
        let cell=vec3u((q+vec3i(i32(n)))%vec3i(i32(n)));
        dmcLoad(c,region.voxel+cell.x+n*cell.y+n*n*cell.z,s);continue;
      }
      // A whole absent brick leaves nothing to look up.
      if(region.voxel==SVO_INVALID&&region.size>=span){valid=false;break;}
      region=meshRegionAt(vertex+(dmcCorner(c)-vec3f(.5))*s);
      if(region.voxel==SVO_INVALID||region.size<s){valid=false;break;}
      largest=max(largest,region.size);dmcLoad(c,region.voxel,region.size);
    }
    if(!valid){continue;}
    if(largest>s){dmcEmitMixed(vertex,scale,depth,largest);}else{dmcEmit(vertex-vec3f(s),scale,depth,false);}
  }}
  // Vertices on this brick's maximum faces. The cell beyond is another
  // brick's; this brick owns the dual cell only where every higher octant is a
  // coarser leaf, which needs the brick diagonally beyond the vertex to be one.
  var plus:array<bool,8>;var graded=false;
  for(var c=1u;c<8u;c+=1u){
    let region=meshRegionAt(vec3f(base)+dmcCorner(c)*span+vec3f(.5*s));
    plus[c]=region.voxel!=SVO_INVALID&&region.size>s;graded=graded||plus[c];
  }
  if(!graded){return;}
  let layers=select(1u,2u,local+1u==n);
  for(var layer=0u;layer<layers;layer+=1u){for(var y=0u;y<=n;y+=1u){for(var x=0u;x<=n;x+=1u){
    let v=vec3u(x,y,local+layer);
    let beyond=select(0u,1u,v.x==n)|select(0u,2u,v.y==n)|select(0u,4u,v.z==n);
    if(beyond==0u||!plus[beyond]){continue;}
    let owner=7u&~beyond;let vertex=vec3f(base+v*scale);
    dmcReset();var valid=true;var largest=s;
    for(var c=0u;c<8u;c+=1u){
      let q=vec3i(v)+vec3i(dmcCorner(c))-vec3i(1);
      if(all(q>=vec3i(0))&&all(q<vec3i(i32(n)))){
        let cell=vec3u(q);dmcLoad(c,own.voxel+cell.x+n*cell.y+n*n*cell.z,s);continue;
      }
      let region=meshRegionAt(vertex+(dmcCorner(c)-vec3f(.5))*s);
      if(region.voxel==SVO_INVALID||region.size<s||(region.size==s&&c>owner)){valid=false;break;}
      largest=max(largest,region.size);dmcLoad(c,region.voxel,region.size);
    }
    if(!valid){continue;}
    dmcEmitMixed(vertex,scale,depth,largest);
  }}}
}
`;
