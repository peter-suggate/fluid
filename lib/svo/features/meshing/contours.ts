/** Clipped-cell extraction; allocation and publication belong to the shared mesh scheduler. */
export const svoContourMeshWGSL = /* wgsl */ `
fn meshAppendContourPolygon(base:vec3u,depth:u32,identity:u32,polygon:ContourPolygon){
  for(var i=1u;i+1u<polygon.count;i+=1u){
    let a=contourPackPoint(polygon.points[0]);let b=contourPackPoint(polygon.points[i]);let c=contourPackPoint(polygon.points[i+1u]);
    if(a==b||b==c||c==a){continue;}
    let n=cross(contourUnpackPoint(b)-contourUnpackPoint(a),contourUnpackPoint(c)-contourUnpackPoint(a));
    if(dot(n,n)<1e-12){continue;}
    meshAppend(base,vec3u(a,b,c),0x80000000u|(meshInflationCode()<<11u)|meshPackFace(0u,0u,depth),identity);
  }
}
fn meshEmitContour(base:vec3u,scale:u32,depth:u32,identity:u32,code:u32){
  var contour=cellContour(sceneIdentityNormal(identity),dry.mapping.cellSize*f32(scale),code);
  let inflation=f32(meshInflationCode())/100.0;
  // Work in the expanded cube's normalized coordinates. Scaling the support
  // inversely keeps the world-space plane fixed instead of inflating the solid.
  contour.high/=1.0+2.0*inflation;
  if(contour.valid==0u){return;}
  for(var face=0u;face<6u;face+=1u){
    var polygon=contourClipPolygon(contourCubeFace(face),contour);
    if(polygon.count<3u){continue;}
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    var p=vec3f(base)+vec3f(0.5*f32(scale));
    p[axis]=f32(base[axis])+select(-0.25,f32(scale)+0.25,(face&1u)!=0u);
    let neighbour=meshRegionAt(p);
    let fits=f32(base[u])>=neighbour.origin[u]&&f32(base[v])>=neighbour.origin[v]
      &&f32(base[u]+scale)<=neighbour.origin[u]+neighbour.size
      &&f32(base[v]+scale)<=neighbour.origin[v]+neighbour.size;
    if(inflation==0.0&&fits&&sceneIdentitySolid(neighbour.identity)){
      if(neighbour.contour==0u){continue;}
      // Keep only the face outside the neighbour's clipped solid. Translate
      // its support plane into this cell, including coarse/fine scale changes.
      let other=cellContour(sceneIdentityNormal(neighbour.identity),dry.mapping.cellSize*neighbour.size,neighbour.contour);
      let ratio=f32(scale)/neighbour.size;
      let offset=dot(other.normal,(vec3f(base)-neighbour.origin)/neighbour.size+vec3f(0.5*ratio-0.5));
      polygon=contourClipPolygon(polygon,CellContour(-other.normal,(offset-other.high)/ratio,1u));
    }
    // A face spanning several finer neighbours remains conservative; their
    // closed surfaces hide its covered portions in the depth test.
    meshAppendContourPolygon(base,depth,identity,polygon);
  }
  meshAppendContourPolygon(base,depth,identity,contourCap(contour));
}
`;
