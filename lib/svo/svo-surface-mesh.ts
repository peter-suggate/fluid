import type { WorkProgress } from "../core/work-progress";

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
  requiredQuads?: number;
  capacityQuads?: number;
  allocatedBytes?: number;
  maximumBytes?: number;
  builds?: number;
  requirementComplete?: boolean;
  completedBricks?: number;
  totalBricks?: number;
  buildPhase?: "extracting" | "capacity" | "complete";
  restartReason?: "initial" | "topology" | "geometry" | "publication";
  fallbackReason?: "budget" | "smooth" | "inside-solid" | "publication" | "extraction";

}

/** Mesh-specific meaning stays beside the producer; the panel only renders facts. */
export function surfaceMeshProgress(status?: SvoSurfaceMeshStatus): WorkProgress {
  if (!status) return { label: "Waiting for mesh publication", state: "waiting", detail: "Mesh counters have not arrived from the GPU." };
  const state = status.state === "ready" ? "complete" : status.state === "fallback" ? "waiting" : "active";
  const capacity = status.buildPhase === "capacity";
  const reason = status.restartReason === "topology" ? "Topology changed; rebuilding the current scene."
    : status.restartReason === "geometry" ? "Geometry changed; rebuilding the current scene."
    : status.restartReason === "publication" ? "Source publication changed; rebuilding the current scene." : undefined;
  return {
    label: status.state === "ready" ? "Mesh ready"
      : status.state === "fallback" ? "Using ray fallback"
      : capacity ? "Expanding mesh storage" : "Extracting voxel surfaces",
    state,
    completed: status.completedBricks,
    total: status.totalBricks,
    unit: "bricks",
    generation: status.builds,
    phase: status.state === "ready" ? "complete" : capacity ? "capacity" : "extracting",
    phases: [{ id: "extracting", label: "Extract" }, { id: "capacity", label: "Storage" }, { id: "complete", label: "Ready" }],
    detail: [reason, status.state === "pending" ? "Current-frame rays provide visibility until the complete mesh is published." : undefined,
      status.state === "fallback" ? status.detail : undefined].filter(Boolean).join(" "),
  };
}

export const SVO_SURFACE_MESH_BYTES = 64 * 1024 * 1024;
export const SVO_SURFACE_MESH_HEADER_BYTES = 64;
/** First 64 bytes retain the public draw/diagnostic ABI; tail is builder state. */
export const SVO_SURFACE_MESH_STATE_BYTES = 80;
/** Maximum bricks extracted in one presentation; no whole-world dispatch. */
export const SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME = 128;
/** Separate bounded GPU batches share one presentation, avoiding a complete
 * fallback render between every 128 bricks. Prepare/publish gate each batch. */
export const SVO_SURFACE_MESH_BUILD_BATCHES_PER_PRESENTATION = 16;

export function svoSurfaceMeshWGSL(group: number, flatNormalsFlag: number, culling = true): string {
  return /* wgsl */ `
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
@group(${group}) @binding(30) var<storage,read_write> meshState:array<atomic<u32>>;
@group(${group}) @binding(32) var<storage,read_write> meshOutput:array<SurfaceQuad>;
@group(${group}) @binding(31) var<storage,read> meshQuads:array<SurfaceQuad>;
@group(${group}) @binding(33) var<storage,read> meshHeader:array<u32>;
@group(${group}) @binding(34) var<storage,read_write> meshVisibleOutput:array<u32>;
@group(${group}) @binding(35) var<storage,read> meshVisible:array<u32>;

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
  // Finish the bounded batch even when storage is exhausted. Publication
  // then rolls it back to the preceding complete-brick checkpoint.
  let slot=atomicAdd(&meshState[4],1u);
  if(slot>=arrayLength(&meshOutput)){atomicOr(&meshState[5],1u);return;}
  meshOutput[slot]=SurfaceQuad(origin,identity,extent,face);
}
// Neighbour coverage at mixed-resolution boundaries is resolved recursively,
// including partially empty fine children. A DFS needs at most 3*depth+1 slots.
fn meshBoundary(origin:vec3u,size:u32,face:u32,identity:u32){
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
      if(!sceneIdentitySolid(neighbour.identity)){var extent=vec3u(0u);extent[u]=s;extent[v]=s;meshAppend(o,extent,face,identity);}
    }else{
      if(count+4u>64u){atomicOr(&meshState[5],2u);break;}
      let half=s/2u;
      for(var child=0u;child<4u;child+=1u){var q=o;q[u]+=(child&1u)*half;q[v]+=((child>>1u)&1u)*half;pending[count]=vec4u(q,half);count+=1u;}
    }
  }
}
@compute @workgroup_size(1)
fn surfaceMeshPrepare(){
  atomicStore(&meshState[0],4u);atomicStore(&meshState[8],0u);
  atomicStore(&meshState[9],1u);atomicStore(&meshState[10],1u);
  atomicStore(&meshState[18],svoControlLoad(1u));
  // Smooth reconstruction has a view-dependent face fallback and cannot be
  // represented by these cached boundary quads. Keep its exact ray semantics.
  if((dry.materialPublication.w&${flatNormalsFlag}u)==0u){atomicStore(&meshState[15],2u);atomicStore(&meshState[1],0u);return;}
  let valid=dryPublicationWord(0u)!=0u&&(dryPublicationWord(1u)&REQUIRED_FIELDS)==REQUIRED_FIELDS;
  if(!valid){atomicStore(&meshState[13],0u);atomicStore(&meshState[1],0u);atomicStore(&meshState[11],0u);atomicStore(&meshState[5],0u);return;}
  if(atomicLoad(&meshState[6])!=dryPublicationWord(2u)||atomicLoad(&meshState[7])!=dryPublicationWord(3u)
    ||atomicLoad(&meshState[12])==0u||(atomicLoad(&meshState[13])==0u&&atomicLoad(&meshState[5])==0u&&atomicLoad(&meshState[11])==0u)){
    var reason=4u;
    if(atomicLoad(&meshState[12])==0u){reason=1u;}
    else if(atomicLoad(&meshState[6])!=dryPublicationWord(2u)){reason=2u;}
    else if(atomicLoad(&meshState[7])!=dryPublicationWord(3u)){reason=3u;}
    atomicStore(&meshState[19],reason);
    atomicStore(&meshState[4],0u);atomicStore(&meshState[5],0u);atomicStore(&meshState[13],0u);
    atomicStore(&meshState[11],1u);atomicAdd(&meshState[12],1u);
    atomicStore(&meshState[14],0u);
    atomicStore(&meshState[6],dryPublicationWord(2u));atomicStore(&meshState[7],dryPublicationWord(3u));
  }
  // The host copies the retained prefix into a larger arena. Resume only when
  // that larger binding is actually visible; stale diagnostic reads never
  // clear GPU revision/error state or overwrite the current cursor.
  if((atomicLoad(&meshState[5])&1u)!=0u&&arrayLength(&meshOutput)>atomicLoad(&meshState[17])){
    atomicAnd(&meshState[5],~1u);
  }
  if(atomicLoad(&meshState[11])!=0u&&atomicLoad(&meshState[5])==0u){
    atomicStore(&meshState[16],atomicLoad(&meshState[4]));
    let remaining=svoControlLoad(1u)-min(atomicLoad(&meshState[14]),svoControlLoad(1u));
    let jobs=min(remaining,${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u)*6u*dry.mapping.brickSize;
    let groups=(jobs+63u)/64u;
    atomicStore(&meshState[8],min(groups,65535u));atomicStore(&meshState[9],max(1u,(groups+65534u)/65535u));
  }
  let p=(uniforms.cameraPosition.xyz-dry.mapping.worldOrigin)/dry.mapping.cellSize;
  atomicStore(&meshState[15],select(0u,1u,sceneIdentitySolid(meshRegionAt(p).identity)));
}
@compute @workgroup_size(64)
fn surfaceMeshBuild(@builtin(global_invocation_id) id:vec3u){
  let cursor=atomicLoad(&meshState[14]);
  let n=dry.mapping.brickSize;let jobsPerBrick=6u*n;
  let job=id.x+id.y*65535u*64u;
  let leafIndex=cursor+job/jobsPerBrick;
  if(leafIndex>=min(svoControlLoad(1u),cursor+${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u)){return;}
  let leaf=svoLeafLoad(leafIndex).topology;
  if(leaf.x>=svoControlLoad(0u)||leaf.z!=0u){return;}
  let node=svoNodeLoad(leaf.x);
  if(node.links.z!=leafIndex||!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return;}
  let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*n);
  // Face/layer masks are independent: each invocation owns exactly one.
  // A 128-brick batch of 8^3 bricks now supplies 96 workgroups, instead of
  // two serial-brick workgroups. Greedy rectangles and mixed-level boundary
  // subdivision are unchanged; only atomic append order may differ.
  let face=(job%jobsPerBrick)/n;let layer=job%n;
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
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
@compute @workgroup_size(1)
fn surfaceMeshPublish(){
  if(atomicLoad(&meshState[11])!=0u&&(atomicLoad(&meshState[8])!=0u||svoControlLoad(1u)==0u)){
    if(atomicLoad(&meshState[5])!=0u){
      // An overflowing batch may be partially written. Discard just that
      // batch, retaining every completed brick. Retry after capacity grows.
      atomicStore(&meshState[4],atomicLoad(&meshState[16]));
      atomicStore(&meshState[17],arrayLength(&meshOutput));
    }else{
      let cursor=min(svoControlLoad(1u),atomicLoad(&meshState[14])+${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u);
      atomicStore(&meshState[14],cursor);
      if(cursor==svoControlLoad(1u)){
        atomicStore(&meshState[11],0u);atomicStore(&meshState[13],1u);
      }
    }
  }
  let usable=atomicLoad(&meshState[13])!=0u&&atomicLoad(&meshState[15])==0u;
  let quads=select(0u,atomicLoad(&meshState[4]),usable);
  atomicStore(&meshState[1],${culling ? "0u" : "quads"});
  let groups=(quads+63u)/64u;
  atomicStore(&meshState[8],min(groups,65535u));
  atomicStore(&meshState[9],max(1u,(groups+65534u)/65535u));
  atomicStore(&meshState[10],1u);
}
// Cull exact cached quad bounds before invoking any vertex shader. No
// occlusion or screen-size approximation: subpixel geometry is retained.
fn meshQuadVisible(index:u32)->bool{
  if(index>=atomicLoad(&meshState[4])){return false;}
  let quad=meshOutput[index];let axis=quad.face/2u;
  let origin=dry.mapping.worldOrigin+vec3f(quad.origin)*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();
  let facing=(camera[0][axis]-origin[axis])*select(-1.0,1.0,(quad.face&1u)!=0u);
  // Retain the coplanar tolerance band to avoid rounding-dependent holes.
  let epsilon=max(1e-5,abs(origin[axis])*1e-6);
  if(facing < -epsilon){return false;}
  let halfExtent=vec3f(quad.extent)*dry.mapping.cellSize*0.5;
  let center=origin+halfExtent-camera[0];
  let z=dot(center,camera[1]);
  if(z+dot(halfExtent,abs(camera[1])) < DRY_REVERSED_Z_NEAR_M-epsilon){return false;}
  let ty=cameraTanHalfFov();let tx=ty*uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let planes=array<vec3f,4>(camera[2]-tx*camera[1],-camera[2]-tx*camera[1],
    camera[3]-ty*camera[1],-camera[3]-ty*camera[1]);
  for(var i=0u;i<4u;i+=1u){
    if(dot(center,planes[i])>dot(halfExtent,abs(planes[i]))+epsilon){return false;}
  }
  return true;
}
var<workgroup> meshCullCount:atomic<u32>;
var<workgroup> meshCullBase:u32;
@compute @workgroup_size(64)
fn surfaceMeshCull(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_index) lane:u32){
  if(lane==0u){atomicStore(&meshCullCount,0u);}
  workgroupBarrier();
  let index=id.x+id.y*65535u*64u;
  let visible=meshQuadVisible(index);
  var rank=0u;if(visible){rank=atomicAdd(&meshCullCount,1u);}
  workgroupBarrier();
  if(lane==0u){
    let count=atomicLoad(&meshCullCount);
    meshCullBase=0u;if(count!=0u){meshCullBase=atomicAdd(&meshState[1],count);}
  }
  workgroupBarrier();
  if(visible){meshVisibleOutput[meshCullBase+rank]=index;}
}
struct MeshVertexOut {
  @builtin(position) position:vec4f,
  @location(0) world:vec3f,
  @location(1) @interpolate(flat) identity:u32,
  @location(2) @interpolate(flat) normal:vec3f,
}
@vertex fn surfaceMeshVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->MeshVertexOut{
  let quad=meshQuads[${culling ? "meshVisible[instance]" : "instance"}];let axis=quad.face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1u,0u),vec2u(1u,1u),vec2u(0u,0u),vec2u(0u,1u));
  // Swapping the two in-plane coordinates reverses winding on negative
  // faces while preserving the 00–11 diagonal and the covered rectangle.
  let corner=select(corners[vertex],corners[vertex].yx,(quad.face&1u)==0u);var lattice=quad.origin;lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
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
