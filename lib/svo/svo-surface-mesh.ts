/**
 * Cached opaque voxel boundary quads. All coordinates are on the accepted
 * finest-cell lattice, so adjacent bricks produce identical shared vertices.
 * The GPU publication header is also the indirect draw/dispatch buffer.
 * A capacity overflow withdraws the entire mesh and selects current-frame rays.
 */
export interface SvoSurfaceMeshStatus {
  state: "pending" | "ready" | "fallback";
  quads?: number;
  detail?: string;
}

export const SVO_SURFACE_MESH_BYTES = 64 * 1024 * 1024;
export const SVO_SURFACE_MESH_HEADER_BYTES = 64;

export function svoSurfaceMeshWGSL(group: number, flatNormalsFlag: number): string {
  return /* wgsl */ `
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
@group(${group}) @binding(30) var<storage,read_write> meshState:array<atomic<u32>>;
@group(${group}) @binding(32) var<storage,read_write> meshOutput:array<SurfaceQuad>;
@group(${group}) @binding(31) var<storage,read> meshQuads:array<SurfaceQuad>;
@group(${group}) @binding(33) var<storage,read> meshHeader:array<u32>;

struct MeshRegion { origin:vec3f, size:f32, identity:u32 }
// Lookup returns the containing cell OR empty octree region. Its extent lets
// a coarse boundary stop subdividing as soon as its neighbour is uniform.
fn meshRegionAt(p:vec3f)->MeshRegion {
  let rootSize=f32((1u<<dry.mapping.maximumDepth)*dry.mapping.brickSize);
  if(any(p<vec3f(0.0))||any(p>=vec3f(rootSize))){return MeshRegion(vec3f(-rootSize),rootSize*3.0,0u);}
  var origin=vec3f(0.0);var size=rootSize;var index=0u;
  for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
    if(index>=svoControlLoad(0u)){return MeshRegion(origin,size,0u);}
    let node=svoNodeLoad(index);
    if(node.links.z!=SVO_INVALID){
      let leaf=svoLeafLoad(node.links.z).topology;
      if(leaf.x!=index||leaf.z!=0u){return MeshRegion(origin,size,0u);}
      if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return MeshRegion(origin,size,0u);}
      let cellSize=size/f32(dry.mapping.brickSize);
      let local=vec3u(clamp(floor((p-origin)/cellSize),vec3f(0.0),vec3f(f32(dry.mapping.brickSize-1u))));
      let voxel=svoBrickVoxelIndex(leaf.y,local,dry.mapping.brickSize);
      var identity=0u;if(voxel<dryVoxelCapacity()){identity=sceneIdentityAt(voxel);}
      return MeshRegion(origin+vec3f(local)*cellSize,cellSize,identity);
    }
    size*=0.5;let upper=p>=origin+vec3f(size);
    let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);
    origin+=select(vec3f(0.0),vec3f(size),upper);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return MeshRegion(origin,size,0u);}
    index=node.links.x+countOneBits(mask&(bit-1u));
  }
  return MeshRegion(origin,size,0u);
}
fn meshAppend(origin:vec3u,extent:vec3u,face:u32,identity:u32){
  let slot=atomicAdd(&meshState[4],1u);
  if(slot>=arrayLength(&meshOutput)){atomicStore(&meshState[5],1u);return;}
  meshOutput[slot]=SurfaceQuad(origin,identity,extent,face);
}
// Neighbour coverage at mixed-resolution boundaries is resolved recursively,
// including partially empty fine children. A DFS needs at most 3*depth+1 slots.
fn meshBoundary(origin:vec3u,size:u32,face:u32,identity:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  var pending:array<vec4u,64>;var count=1u;pending[0]=vec4u(origin,size);
  loop {
    if(count==0u||atomicLoad(&meshState[5])!=0u){break;}
    count-=1u;let item=pending[count];let o=item.xyz;let s=item.w;
    var p=vec3f(o);p[u]+=f32(s)*0.5;p[v]+=f32(s)*0.5;
    p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
    let neighbour=meshRegionAt(p);
    let fits=f32(o[u])>=neighbour.origin[u]&&f32(o[v])>=neighbour.origin[v]
      &&f32(o[u]+s)<=neighbour.origin[u]+neighbour.size
      &&f32(o[v]+s)<=neighbour.origin[v]+neighbour.size;
    if(fits||s==1u){
      if(!sceneIdentitySolid(neighbour.identity)){var extent=vec3u(0u);extent[u]=s;extent[v]=s;meshAppend(o,extent,face,identity);}
    }else{
      if(count+4u>64u){atomicStore(&meshState[5],1u);break;}
      let half=s/2u;
      for(var child=0u;child<4u;child+=1u){var q=o;q[u]+=(child&1u)*half;q[v]+=((child>>1u)&1u)*half;pending[count]=vec4u(q,half);count+=1u;}
    }
  }
}
@compute @workgroup_size(1)
fn surfaceMeshPrepare(){
  atomicStore(&meshState[0],6u);atomicStore(&meshState[8],0u);
  atomicStore(&meshState[9],1u);atomicStore(&meshState[10],1u);atomicStore(&meshState[11],0u);
  // Smooth reconstruction has a view-dependent face fallback and cannot be
  // represented by these cached boundary quads. Keep its exact ray semantics.
  if((dry.materialPublication.w&${flatNormalsFlag}u)==0u){atomicStore(&meshState[15],2u);atomicStore(&meshState[1],0u);return;}
  let valid=dryPublicationWord(0u)!=0u&&(dryPublicationWord(1u)&REQUIRED_FIELDS)==REQUIRED_FIELDS;
  if(!valid){atomicStore(&meshState[13],0u);atomicStore(&meshState[1],0u);return;}
  if(atomicLoad(&meshState[6])!=dryPublicationWord(2u)||atomicLoad(&meshState[7])!=dryPublicationWord(3u)
    ||atomicLoad(&meshState[12])==0u||(atomicLoad(&meshState[13])==0u&&atomicLoad(&meshState[5])==0u)){
    atomicStore(&meshState[4],0u);atomicStore(&meshState[5],0u);atomicStore(&meshState[13],0u);
    atomicStore(&meshState[11],1u);atomicAdd(&meshState[12],1u);
    let groups=(svoControlLoad(1u)+63u)/64u;
    atomicStore(&meshState[8],min(groups,65535u));atomicStore(&meshState[9],max(1u,(groups+65534u)/65535u));
  }
  let p=(uniforms.cameraPosition.xyz-dry.mapping.worldOrigin)/dry.mapping.cellSize;
  atomicStore(&meshState[15],select(0u,1u,sceneIdentitySolid(meshRegionAt(p).identity)));
}
@compute @workgroup_size(64)
fn surfaceMeshBuild(@builtin(global_invocation_id) id:vec3u){
  let leafIndex=id.x+id.y*65535u*64u;
  if(leafIndex>=svoControlLoad(1u)||atomicLoad(&meshState[5])!=0u){return;}
  let leaf=svoLeafLoad(leafIndex).topology;
  if(leaf.x>=svoControlLoad(0u)||leaf.z!=0u){return;}
  let node=svoNodeLoad(leaf.x);
  if(node.links.z!=leafIndex||!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return;}
  let n=dry.mapping.brickSize;let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*n);
  // One invocation owns a brick. Greedy rectangles never cross identity or
  // brick boundaries; the boundary routine handles neighbouring levels.
  for(var face=0u;face<6u;face+=1u){
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
    for(var layer=0u;layer<n;layer+=1u){
      var mask:array<u32,64>;
      for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
        let m=x+y*n;mask[m]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
        let voxel=svoBrickVoxelIndex(leaf.y,c,n);if(voxel>=dryVoxelCapacity()){continue;}
        let identity=sceneIdentityAt(voxel);if(!sceneIdentitySolid(identity)){continue;}
        var origin=base+c*scale;origin[axis]+=select(0u,scale,(face&1u)!=0u);
        let boundary=select(layer==0u,layer==n-1u,(face&1u)!=0u);
        if(boundary){
          var p=vec3f(origin);p[u]+=f32(scale)*0.5;p[v]+=f32(scale)*0.5;p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
          let neighbour=meshRegionAt(p);
          let fits=f32(origin[u])>=neighbour.origin[u]&&f32(origin[v])>=neighbour.origin[v]
            &&f32(origin[u]+scale)<=neighbour.origin[u]+neighbour.size&&f32(origin[v]+scale)<=neighbour.origin[v]+neighbour.size;
          if(fits){if(!sceneIdentitySolid(neighbour.identity)){mask[m]=identity;}}
          else{meshBoundary(origin,scale,face,identity);}
          continue;
        }
        var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,(face&1u)!=0u));
        if(!sceneIdentitySolid(sceneIdentityAt(svoBrickVoxelIndex(leaf.y,adjacent,n)))){mask[m]=identity;}
      }}
      for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
        let identity=mask[x+y*n];if(identity==0u){continue;}
        var width=1u;loop{if(x+width>=n){break;}if(mask[x+width+y*n]!=identity){break;}width+=1u;}
        var height=1u;loop{if(y+height>=n){break;}var same=true;for(var k=0u;k<width;k+=1u){if(mask[x+k+(y+height)*n]!=identity){same=false;}}
          if(!same){break;}height+=1u;}
        for(var j=0u;j<height;j+=1u){for(var k=0u;k<width;k+=1u){mask[x+k+(y+j)*n]=0u;}}
        var origin=base;origin[axis]+=(layer+select(0u,1u,(face&1u)!=0u))*scale;origin[u]+=x*scale;origin[v]+=y*scale;
        var extent=vec3u(0u);extent[u]=width*scale;extent[v]=height*scale;meshAppend(origin,extent,face,identity);
      }}
    }
  }
}
@compute @workgroup_size(1)
fn surfaceMeshPublish(){
  if(atomicLoad(&meshState[11])!=0u){
    atomicStore(&meshState[6],dryPublicationWord(2u));atomicStore(&meshState[7],dryPublicationWord(3u));
    atomicStore(&meshState[13],select(1u,0u,atomicLoad(&meshState[5])!=0u));
  }
  let usable=atomicLoad(&meshState[13])!=0u&&atomicLoad(&meshState[15])==0u;
  atomicStore(&meshState[1],select(0u,atomicLoad(&meshState[4]),usable));
}
struct MeshVertexOut {
  @builtin(position) position:vec4f,
  @location(0) world:vec3f,
  @location(1) @interpolate(flat) identity:u32,
  @location(2) @interpolate(flat) normal:vec3f,
}
@vertex fn surfaceMeshVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->MeshVertexOut{
  let quad=meshQuads[instance];let axis=quad.face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,6>(vec2u(0u,0u),vec2u(1u,0u),vec2u(1u,1u),vec2u(0u,0u),vec2u(1u,1u),vec2u(0u,1u));
  let corner=corners[vertex];var lattice=quad.origin;lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  let world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];let z=dot(relative,camera[1]);
  let position=vec4f(dot(relative,camera[2])/(cameraTanHalfFov()*uniforms.viewport.x/max(uniforms.viewport.y,1.0)),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,z);
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,(quad.face&1u)!=0u);
  return MeshVertexOut(position,world,quad.identity,normal);
}
struct MeshSurfaceOut {
  @location(0) packedSurface:vec4u,@location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,@location(3) opaqueIdentity:vec2u,
}
@fragment fn surfaceMeshFragment(input:MeshVertexOut)->MeshSurfaceOut{
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);let t=length(input.world-camera[0]);
  let normal=dryShadingNormal(input.identity,input.normal);
  let hit=DryHit(t,normal.normal,sceneIdentityMaterial(input.identity),DRY_OWNER_NONE,normal.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
  let out=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
  return MeshSurfaceOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity);
}
@fragment fn surfaceMeshBackground(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  var hit=missHit();
  if(meshHeader[13]==0u||meshHeader[15]!=0u){hit=traceStatic(camera[0],rd);}
  else{hit=dryPlanarCatalogHit(camera[0],rd,0.0,DRY_MISS);}
  if(hit.t>=DRY_MISS){return dryRasterPrimaryMiss();}
  return dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND);
}
`;
}
