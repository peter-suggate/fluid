/** Surface normals for continuous terrain and swept tubes. Geometry, occupancy
 * and materials remain the published voxel mesh's; unrelated props keep their
 * fitted normals. Sampling at shared vertices makes interpolation continuous. */
export function svoTerrainSurfaceNormalsWGSL(terrainWord: number, editsWord: number): string {
  return /* wgsl */ `
fn meshTerrainFloat(at:u32)->f32{return bitcast<f32>(meshMaintenance[at]);}
fn meshTerrainHeight(base:u32,q:vec2i)->f32{
  let size=vec2u(meshMaintenance[base+4u],meshMaintenance[base+5u]);
  let at=vec2u(clamp(q,vec2i(0),vec2i(size)-vec2i(1)));
  return meshTerrainFloat(base+8u+at.x+size.x*at.y);
}
fn meshTerrainSample(base:u32,xz:vec2f)->f32{
  let origin=vec2f(meshTerrainFloat(base),meshTerrainFloat(base+1u));
  let pitch=vec2f(meshTerrainFloat(base+2u),meshTerrainFloat(base+3u));
  let p=(xz-origin)/pitch-vec2f(.5);let q=vec2i(floor(p));let t=fract(p);
  return mix(mix(meshTerrainHeight(base,q),meshTerrainHeight(base,q+vec2i(1,0)),t.x),
    mix(meshTerrainHeight(base,q+vec2i(0,1)),meshTerrainHeight(base,q+vec2i(1,1)),t.x),t.y);
}
// Edits own their surface. Conservatively retain fitted normals throughout a
// touched page and its one-cell neighbourhood, including erased cells.
fn meshTerrainEditedPage(base:u32,q:vec3i)->bool{
  var hash=0x811c9dc5u;
  for(var axis=0u;axis<4u;axis++){
    var value=0u;if(axis<3u){value=bitcast<u32>(q[axis]);}
    hash=(hash^value)*0x01000193u;hash^=hash>>16u;
  }
  hash|=1u;let mask=meshMaintenance[base+3u];var slot=hash&mask;
  for(var probe=0u;probe<=mask;probe++){
    let at=base+meshMaintenance[base+4u]+slot*6u;let state=meshMaintenance[at];
    if(state==0u){return false;}
    if(state==2u&&meshMaintenance[at+1u]==hash&&all(bitcast<vec3i>(vec3u(meshMaintenance[at+2u],meshMaintenance[at+3u],meshMaintenance[at+4u]))==q)){return true;}
    slot=(slot+1u)&mask;
  }
  return false;
}
fn meshTerrainEdited(world:vec3f)->bool{
  let base=meshHeader[${editsWord}u];if(base==0u||meshMaintenance[base+7u]==0u){return false;}
  let origin=vec3f(meshTerrainFloat(base+16u),meshTerrainFloat(base+17u),meshTerrainFloat(base+18u));
  let cell=vec3f(meshTerrainFloat(base+20u),meshTerrainFloat(base+21u),meshTerrainFloat(base+22u));
  let lo=vec3i(floor((world-origin-dry.mapping.cellSize)/cell))>>vec3u(3);
  let hi=vec3i(floor((world-origin+dry.mapping.cellSize)/cell))>>vec3u(3);
  for(var z=lo.z;z<=hi.z;z++){for(var y=lo.y;y<=hi.y;y++){for(var x=lo.x;x<=hi.x;x++){
    if(meshTerrainEditedPage(base,vec3i(x,y,z))){return true;}
  }}}
  return false;
}
fn meshAuthoredTerrainNormal(world:vec3f,material:u32,face:vec3f)->vec3f{
  let base=meshHeader[${terrainWord}u];if(base==0u){return vec3f(0);}
  if(material!=meshMaintenance[base+6u]){return vec3f(0);}
  let origin=vec2f(meshTerrainFloat(base),meshTerrainFloat(base+1u));
  let pitch=vec2f(meshTerrainFloat(base+2u),meshTerrainFloat(base+3u));
  let size=vec2f(f32(meshMaintenance[base+4u]),f32(meshMaintenance[base+5u]));
  // The bounded heightfield's vertical sides are real edges, not ground.
  if(any(world.xz<=origin+pitch)||any(world.xz>=origin+pitch*(size-vec2f(1)))){return vec3f(0);}
  if(meshTerrainEdited(world)){return vec3f(0);}
  // A fitted triangle can sit slightly off the heightfield. On a steep wall
  // that small horizontal error samples a different part of its rolled rim.
  // Project the shading query onto the same bilinear surface before taking its
  // normal; this does not move the mesh or change contact with the water.
  var surface=world;
  for(var iteration=0u;iteration<2u;iteration++){
    let p=(surface.xz-origin)/pitch-vec2f(.5);let q=vec2i(floor(p));let t=fract(p);
    let a=meshTerrainHeight(base,q);let b=meshTerrainHeight(base,q+vec2i(1,0));
    let c=meshTerrainHeight(base,q+vec2i(0,1));let d=meshTerrainHeight(base,q+vec2i(1,1));
    let height=mix(mix(a,b,t.x),mix(c,d,t.x),t.y);
    let g=vec3f(-mix(b-a,d-c,t.y)/pitch.x,1,-mix(c-a,d-b,t.x)/pitch.y);
    surface-=g*((surface.y-height)/dot(g,g));
  }
  // Interpolate unit normals, rather than slopes: a near-vertical sample
  // must not outweigh its neighbours by an arbitrary heightfield gradient.
  // Centre differences at the sample nodes retain flat tops and vary smoothly
  // through rolled edges without the cell-boundary ridges of bilinear slopes.
  let p=(surface.xz-origin)/pitch-vec2f(.5);let q=vec2i(floor(p));let t=fract(p);
  var g=vec3f(0);
  for(var corner=0u;corner<4u;corner++){
    let b=vec2u(corner&1u,corner>>1u);let at=q+vec2i(b);
    let dx=(meshTerrainHeight(base,at+vec2i(1,0))-meshTerrainHeight(base,at-vec2i(1,0)))/(2.0*pitch.x);
    let dz=(meshTerrainHeight(base,at+vec2i(0,1))-meshTerrainHeight(base,at-vec2i(0,1)))/(2.0*pitch.y);
    let weight=select(vec2f(1)-t,t,b!=vec2u(0));g+=normalize(vec3f(-dx,1,-dz))*weight.x*weight.y;
  }
  let normal=normalize(g);
  let distance=abs(world.y-meshTerrainSample(base,world.xz))*normal.y;
  if(distance>2.0*max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))||dot(normal,face)<=0.0){return vec3f(0);}
  return normal;
}
`;
}

export function svoAuthoredSurfaceNormalsWGSL(terrainWord: number, editsWord: number, materialBaseWord: number): string {
  return /* wgsl */ `${svoTerrainSurfaceNormalsWGSL(terrainWord, editsWord)}
fn meshAuthoredSurfaceNormal(world:vec3f,material:u32,face:vec3f)->vec3f{
  let terrain=meshAuthoredTerrainNormal(world,material,face);
  if(dot(terrain,terrain)>0.0){return terrain;}
  // Dense scenery publications admit an O(1) material lookup. Verify the
  // candidate so omitted/reordered records simply keep their fitted normals.
  let base=meshHeader[${materialBaseWord}u];if(base==0u||material<base){return vec3f(0);}
  let index=material-base;if(index>=dry.metadata.x){return vec3f(0);}
  let record=dryPrimitive(index);
  if(svoPrimitiveMaterialId(record)!=material||svoPrimitiveKind(record)!=SVO_KIND_SMOOTH_UNION_CLUSTER){return vec3f(0);}
  let reference=svoPrimitiveClusterReference(record);
  if(reference<DRY_SCENE_CLUSTER_WORD_OFFSET||reference>=DRY_SCENE_CLUSTER_WORD_LIMIT){return vec3f(0);}
  // Read the field tag before loading a packing: foliage/pebbles are numerous
  // in the x10 set and should neither pay for nor receive tube smoothing.
  if(drySceneArena[reference]!=SVO_CLUSTER_FIELD_TAPERED_SWEEP){return vec3f(0);}
  let packing=dryClusterPacking(record);
  let sample=svoClusterSample(svoPrimitiveLocalPoint(record,world),svoPrimitiveDimensions_m(record),packing);
  let g=svoQuaternionRotate(record.orientation,sample.gradient);
  if(dot(g,g)<1e-12||abs(sample.distance_m)>2.0*max(dry.mapping.cellSize.x,max(dry.mapping.cellSize.y,dry.mapping.cellSize.z))){return vec3f(0);}
  let normal=normalize(g);if(dot(normal,face)<=0.0){return vec3f(0);}
  return normal;
}
`;
}
