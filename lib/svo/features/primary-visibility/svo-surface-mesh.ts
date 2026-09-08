import type { WorkProgress } from "../../../core/work-progress";
import { SVO_SCREEN_SPACE_TERMINATION_CONTRACT } from "../lighting-visibility/svo-screen-space-termination";

/**
 * Cached opaque voxel boundary quads. All coordinates are on the accepted
 * finest-cell lattice, so adjacent bricks produce identical shared vertices.
 * The GPU publication header is also the indirect draw/dispatch buffer.
 * A capacity overflow withdraws the entire mesh until storage can grow.
 */
export interface SvoSurfaceMeshStatus {
  state: "pending" | "ready" | "blocked";
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
  const state = status.state === "ready" ? "complete" : status.state === "blocked" ? "waiting" : "active";
  const capacity = status.buildPhase === "capacity";
  const reason = status.restartReason === "topology" ? "Topology changed; rebuilding the current scene."
    : status.restartReason === "geometry" ? "Geometry changed; rebuilding the current scene."
    : status.restartReason === "publication" ? "Source publication changed; rebuilding the current scene." : undefined;
  return {
    label: status.state === "ready" ? "Mesh ready"
      : status.state === "blocked" ? "Raster unavailable"
      : capacity ? "Expanding mesh storage" : "Extracting voxel surfaces",
    state,
    completed: status.completedBricks,
    total: status.totalBricks,
    unit: "bricks",
    generation: status.builds,
    phase: status.state === "ready" ? "complete" : capacity ? "capacity" : "extracting",
    phases: [{ id: "extracting", label: "Extract" }, { id: "capacity", label: "Storage" }, { id: "complete", label: "Ready" }],
    detail: [reason, status.state === "pending" ? "The current voxel scene is traced while its raster mesh is rebuilt." : undefined,
      status.state === "blocked" ? status.detail : undefined].filter(Boolean).join(" "),
  };
}

export const SVO_SURFACE_MESH_BYTES = 64 * 1024 * 1024;
export const SVO_SURFACE_MESH_HEADER_BYTES = 64;
/** First 64 bytes retain the public draw/diagnostic ABI; tail is builder state. */
export const SVO_SURFACE_MESH_STATE_BYTES = 80;
/** Bricks extracted by one bounded GPU batch; no whole-world dispatch. */
export const SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME = 128;
/**
 * Batches encoded in the first presentation of a build. Prepare/publish gate
 * each batch on the GPU, so several share one presentation instead of paying
 * a complete fallback render between every 128 bricks.
 */
export const SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL = 16;
/**
 * Ceiling on batches per presentation once a build has stayed pending.
 *
 * Every presentation of an incomplete build re-traces the current voxel scene
 * at full resolution as its fallback, and that trace, not the extraction, is
 * what a long build is made of: on `hero-garden-hose-x10` a 2,048-brick
 * presentation spent 8 ms extracting and 83 ms tracing, and 595,825 bricks
 * took ~291 such presentations. The extraction is GPU resident and its cursor
 * is GPU owned, so the host is free to encode more batches per presentation.
 * Doubling while the build stays pending keeps an edit's short rebuild at one
 * cheap presentation and lets a whole-world build reach this ceiling within a
 * few frames, where extraction outweighs the fallback it is paired with.
 * Batches past completion dispatch zero workgroups.
 */
export const SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM = 128;

/**
 * Batches to encode for the given number of consecutive presentations this
 * build has already spent pending. Zero is the first presentation of a build.
 */
export function surfaceMeshBuildBatches(pendingPresentations: number): number {
  const ramp = Math.max(0, Math.floor(pendingPresentations));
  return Math.min(SVO_SURFACE_MESH_BUILD_BATCHES_MAXIMUM, SVO_SURFACE_MESH_BUILD_BATCHES_INITIAL * 2 ** Math.min(ramp, 30));
}

/** Presentations a build of `bricks` bricks needs under the ramp, from a cold start. */
export function surfaceMeshBuildPresentations(bricks: number): number {
  let remaining = Math.max(0, bricks); let presentations = 0;
  while (remaining > 0) {
    remaining -= surfaceMeshBuildBatches(presentations) * SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME;
    presentations += 1;
  }
  return presentations;
}

export function svoSurfaceMeshWGSL(group: number, flatNormalsFlag: number, culling = true): string {
  return /* wgsl */ `
// \`face\` packs the boundary face in bits 0-2, the detail level in bits 3-5 and
// the brick's octree depth in bits 6-10. Level k quads bound cells of 2^k
// resident voxels; level 0 is the exact voxel boundary.
struct SurfaceQuad { origin:vec3u, identity:u32, extent:vec3u, face:u32 }
@group(${group}) @binding(30) var<storage,read_write> meshState:array<atomic<u32>>;
@group(${group}) @binding(32) var<storage,read_write> meshOutput:array<SurfaceQuad>;
@group(${group}) @binding(31) var<storage,read> meshQuads:array<SurfaceQuad>;
@group(${group}) @binding(33) var<storage,read> meshHeader:array<u32>;
@group(${group}) @binding(34) var<storage,read_write> meshVisibleOutput:array<u32>;
@group(${group}) @binding(35) var<storage,read> meshVisible:array<u32>;

fn meshQuadFace(word:u32)->u32{return word&7u;}
fn meshQuadLevel(word:u32)->u32{return (word>>3u)&7u;}
fn meshQuadDepth(word:u32)->u32{return (word>>6u)&31u;}
fn meshPackFace(face:u32,level:u32,depth:u32)->u32{return face|(level<<3u)|(depth<<6u);}
// Levels a brick can offer: the voxel boundary plus one per halving of the
// brick edge down to a single cell.
fn meshLevelCount()->u32{return countTrailingZeros(max(dry.mapping.brickSize,1u))+1u;}
// Filtered-detail threshold in live pixels; zero is the exact voxel mesh.
// Authored at the screen-space contract's reference height so it stays angular.
fn dryMeshLodPixels()->f32{
  return max(dry.lod.w,0.0)*uniforms.viewport.y/${SVO_SCREEN_SPACE_TERMINATION_CONTRACT.referenceViewportHeightPixels};
}
// Quads merge on material alone. A level-0 quad reads its voxel's baked normal
// per fragment, so its normal half is left absent; coarse quads carry the mean
// baked normal of the voxels they stand for.
fn meshMergeIdentity(identity:u32)->u32{return sceneIdentityMaterial(identity)|(SCENE_IDENTITY_NO_NORMAL<<16u);}

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
// Whether the lattice-aligned cube of \`cell\` finest cells around \`p\` is solid
// at every resident voxel. A coarse boundary face hides behind a neighbour only
// when the neighbour's finest detail covers it completely, so a neighbour drawn
// finer than this brick can never open a gap; a partial neighbour leaves the
// face to the depth test. A neighbour subtree finer than the cube answers
// "uncovered" rather than being walked.
fn meshNeighbourCovered(p:vec3f,cell:u32)->bool {
  let n=dry.mapping.brickSize;
  let rootSize=f32((1u<<dry.mapping.maximumDepth)*n);
  if(any(p<vec3f(0.0))||any(p>=vec3f(rootSize))){return false;}
  var origin=vec3f(0.0);var size=rootSize;var index=0u;
  for(var level=0u;level<=dry.mapping.maximumDepth;level+=1u){
    if(index>=svoControlLoad(0u)){return false;}
    let node=svoNodeLoad(index);
    if(node.links.z!=SVO_INVALID){
      let leaf=svoLeafLoad(node.links.z).topology;
      if(leaf.x!=index||leaf.z!=0u){return false;}
      if(!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return false;}
      let voxelSize=size/f32(n);
      if(voxelSize>=f32(cell)){
        let local=vec3u(clamp(floor((p-origin)/voxelSize),vec3f(0.0),vec3f(f32(n-1u))));
        let voxel=svoBrickVoxelIndex(leaf.y,local,n);
        return voxel<dryVoxelCapacity()&&sceneIdentitySolid(sceneIdentityAt(voxel));
      }
      let cubeOrigin=floor(p/f32(cell))*f32(cell);
      let first=vec3u(clamp(floor((cubeOrigin-origin)/voxelSize),vec3f(0.0),vec3f(f32(n-1u))));
      let span=min(u32(round(f32(cell)/voxelSize)),n);
      for(var z=0u;z<span;z+=1u){for(var y=0u;y<span;y+=1u){for(var x=0u;x<span;x+=1u){
        let local=min(first+vec3u(x,y,z),vec3u(n-1u));
        let voxel=svoBrickVoxelIndex(leaf.y,local,n);
        if(voxel>=dryVoxelCapacity()||!sceneIdentitySolid(sceneIdentityAt(voxel))){return false;}
      }}}
      return true;
    }
    if(size<=f32(cell)){return false;}
    size*=0.5;let upper=p>=origin+vec3f(size);
    let octant=select(0u,1u,upper.x)|select(0u,2u,upper.y)|select(0u,4u,upper.z);
    origin+=select(vec3f(0.0),vec3f(size),upper);
    let mask=node.address.w&255u;let bit=1u<<octant;
    if((mask&bit)==0u){return false;}
    index=node.links.x+countOneBits(mask&(bit-1u));
  }
  return false;
}
// Identity of one level-k cell of a brick: the first solid material, with the
// normalised mean of the solid voxels' baked normals in the high half. A cell
// is solid when any voxel in it is, so far detail dilates rather than vanishes.
fn meshCellIdentity(payload:u32,cell:vec3u,level:u32,n:u32)->u32 {
  let span=1u<<level;var material=0u;var sum=vec3f(0.0);
  for(var z=0u;z<span;z+=1u){for(var y=0u;y<span;y+=1u){for(var x=0u;x<span;x+=1u){
    let voxel=svoBrickVoxelIndex(payload,cell*span+vec3u(x,y,z),n);
    if(voxel>=dryVoxelCapacity()){continue;}
    let identity=sceneIdentityAt(voxel);
    if(!sceneIdentitySolid(identity)){continue;}
    if(material==0u){material=sceneIdentityMaterial(identity);}
    if(sceneIdentityHasNormal(identity)){sum+=sceneIdentityNormal(identity);}
  }}}
  if(material==0u){return 0u;}
  var normalWord=SCENE_IDENTITY_NO_NORMAL;
  if(dot(sum,sum)>1e-6){normalWord=svoGBufferPackNormalOct8(normalize(sum));}
  return material|(normalWord<<16u);
}
fn meshAppend(origin:vec3u,extent:vec3u,packedFace:u32,identity:u32){
  // Finish the bounded batch even when storage is exhausted. Publication
  // then rolls it back to the preceding complete-brick checkpoint.
  let slot=atomicAdd(&meshState[4],1u);
  if(slot>=arrayLength(&meshOutput)){atomicOr(&meshState[5],1u);return;}
  meshOutput[slot]=SurfaceQuad(origin,identity,extent,packedFace);
}
// Greedy rectangles over one face layer's exposed-cell mask. \`m\` cells per
// side, each \`cell\` lattice units wide; the mask is consumed as it is merged.
fn meshEmitMask(mask:ptr<function,array<u32,64>>,m:u32,base:vec3u,cell:u32,face:u32,layer:u32,packedFace:u32){
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
    let identity=(*mask)[x+y*m];if(identity==0u){continue;}
    var width=1u;loop{if(x+width>=m){break;}if((*mask)[x+width+y*m]!=identity){break;}width+=1u;}
    var height=1u;loop{if(y+height>=m){break;}var same=true;for(var k=0u;k<width;k+=1u){if((*mask)[x+k+(y+height)*m]!=identity){same=false;}}
      if(!same){break;}height+=1u;}
    for(var j=0u;j<height;j+=1u){for(var k=0u;k<width;k+=1u){(*mask)[x+k+(y+j)*m]=0u;}}
    var origin=base;origin[axis]+=(layer+select(0u,1u,(face&1u)!=0u))*cell;origin[u]+=x*cell;origin[v]+=y*cell;
    var extent=vec3u(0u);extent[u]=width*cell;extent[v]=height*cell;meshAppend(origin,extent,packedFace,identity);
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
      if(!sceneIdentitySolid(neighbour.identity)){var extent=vec3u(0u);extent[u]=s;extent[v]=s;meshAppend(o,extent,packedFace,identity);}
    }else{
      if(count+4u>64u){atomicOr(&meshState[5],2u);break;}
      let half=s/2u;
      for(var child=0u;child<4u;child+=1u){var q=o;q[u]+=(child&1u)*half;q[v]+=((child>>1u)&1u)*half;pending[count]=vec4u(q,half);count+=1u;}
    }
  }
}
// One coarse level of one brick in a single invocation: every cell identity
// is derived once from the brick's own voxels, then all six faces' layers are
// masked from that table. Within the brick both sides of a face use the same
// dilated cells; across a brick boundary the face hides only behind complete
// coverage, so mixed levels between neighbours stay watertight.
fn meshBuildLevel(payload:u32,base:vec3u,scale:u32,depth:u32,level:u32,n:u32){
  let m=n>>level;let cell=scale<<level;
  var cells:array<u32,64>;
  for(var i=0u;i<m*m*m;i+=1u){cells[i]=meshCellIdentity(payload,vec3u(i%m,(i/m)%m,i/(m*m)),level,n);}
  let packedBase=meshPackFace(0u,level,depth);
  for(var face=0u;face<6u;face+=1u){
    let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;let positive=(face&1u)!=0u;
    for(var layer=0u;layer<m;layer+=1u){
      var mask:array<u32,64>;
      for(var y=0u;y<m;y+=1u){for(var x=0u;x<m;x+=1u){
        let index=x+y*m;mask[index]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
        let identity=cells[c.x+c.y*m+c.z*m*m];if(identity==0u){continue;}
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
        if(exposed){mask[index]=identity;}
      }}
      meshEmitMask(&mask,m,base,cell,face,layer,packedBase|face);
    }
  }
}
@compute @workgroup_size(1)
fn surfaceMeshPrepare(){
  atomicStore(&meshState[0],4u);atomicStore(&meshState[8],0u);
  atomicStore(&meshState[9],1u);atomicStore(&meshState[10],1u);
  atomicStore(&meshState[18],svoControlLoad(1u));
  // Smooth reconstruction has a view-dependent face fallback and cannot be
  // represented by these cached boundary quads. Withhold raster geometry for this unsupported representation.
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
    // One job per exact face layer, plus one per coarse level.
    let jobs=min(remaining,${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u)*(6u*dry.mapping.brickSize+meshLevelCount()-1u);
    let groups=(jobs+63u)/64u;
    atomicStore(&meshState[8],min(groups,65535u));atomicStore(&meshState[9],max(1u,(groups+65534u)/65535u));
  }
  let p=(uniforms.cameraPosition.xyz-dry.mapping.worldOrigin)/dry.mapping.cellSize;
  atomicStore(&meshState[15],select(0u,1u,sceneIdentitySolid(meshRegionAt(p).identity)));
}
@compute @workgroup_size(64)
fn surfaceMeshBuild(@builtin(global_invocation_id) id:vec3u){
  let cursor=atomicLoad(&meshState[14]);
  let n=dry.mapping.brickSize;let jobsPerBrick=6u*n+meshLevelCount()-1u;
  let job=id.x+id.y*65535u*64u;
  let leafIndex=cursor+job/jobsPerBrick;
  if(leafIndex>=min(svoControlLoad(1u),cursor+${SVO_SURFACE_MESH_BUILD_BRICKS_PER_FRAME}u)){return;}
  let leaf=svoLeafLoad(leafIndex).topology;
  if(leaf.x>=svoControlLoad(0u)||leaf.z!=0u){return;}
  let node=svoNodeLoad(leaf.x);
  if(node.links.z!=leafIndex||!svoBrickLifecycleCurrent(svoBrickLifecycleDecode(node.links.w))){return;}
  let scale=1u<<(dry.mapping.maximumDepth-node.address.z);
  let base=svoDecodeMorton(node.address.x,node.address.y,node.address.z)*(scale*n);
  let local=job%jobsPerBrick;
  if(local>=6u*n){meshBuildLevel(leaf.y,base,scale,node.address.z,local-6u*n+1u,n);return;}
  // Face/layer masks are independent: each invocation owns exactly one.
  // A 128-brick batch of 8^3 bricks now supplies 96 workgroups, instead of
  // two serial-brick workgroups. Greedy rectangles and mixed-level boundary
  // subdivision are unchanged; only atomic append order may differ.
  let face=local/n;let layer=local%n;
  let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let packedFace=meshPackFace(face,0u,node.address.z);
      var mask:array<u32,64>;
      for(var y=0u;y<n;y+=1u){for(var x=0u;x<n;x+=1u){
        let m=x+y*n;mask[m]=0u;var c=vec3u(0u);c[axis]=layer;c[u]=x;c[v]=y;
        let voxel=svoBrickVoxelIndex(leaf.y,c,n);if(voxel>=dryVoxelCapacity()){continue;}
        let identity=meshMergeIdentity(sceneIdentityAt(voxel));if(!sceneIdentitySolid(identity)){continue;}
        var origin=base+c*scale;origin[axis]+=select(0u,scale,(face&1u)!=0u);
        let boundary=select(layer==0u,layer==n-1u,(face&1u)!=0u);
        if(boundary){
          var p=vec3f(origin);p[u]+=f32(scale)*0.5;p[v]+=f32(scale)*0.5;p[axis]+=select(-0.25,0.25,(face&1u)!=0u);
          let neighbour=meshRegionAt(p);
          let fits=f32(origin[u])>=neighbour.origin[u]&&f32(origin[v])>=neighbour.origin[v]
            &&f32(origin[u]+scale)<=neighbour.origin[u]+neighbour.size&&f32(origin[v]+scale)<=neighbour.origin[v]+neighbour.size;
          if(fits){if(!sceneIdentitySolid(neighbour.identity)){mask[m]=identity;}}
          else{meshBoundary(origin,scale,face,identity,packedFace);}
          continue;
        }
        var adjacent=c;adjacent[axis]=u32(i32(layer)+select(-1,1,(face&1u)!=0u));
        if(!sceneIdentitySolid(sceneIdentityAt(svoBrickVoxelIndex(leaf.y,adjacent,n)))){mask[m]=identity;}
      }}
      meshEmitMask(&mask,n,base,scale,face,layer,packedFace);
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
// Which of a brick's levels this camera draws: the coarsest whose cell still
// projects under the threshold, measured on the brick's enclosing sphere the
// way the traversal contract measures a node. Every quad of one brick answers
// alike, so exactly one level of each brick reaches the rasterizer.
fn meshQuadSelected(quad:SurfaceQuad)->bool{
  let level=meshQuadLevel(quad.face);
  let threshold=dryMeshLodPixels();
  if(threshold<=0.0){return level==0u;}
  let n=dry.mapping.brickSize;
  let face=meshQuadFace(quad.face);let axis=face/2u;
  let brickLattice=n*(1u<<(dry.mapping.maximumDepth-meshQuadDepth(quad.face)));
  var base=(quad.origin/brickLattice)*brickLattice;
  // A positive face on the brick's far side sits on the next brick's origin.
  if((face&1u)!=0u&&quad.origin[axis]%brickLattice==0u){base[axis]-=brickLattice;}
  let camera=dryRasterPrimaryCamera();
  let half=vec3f(f32(brickLattice))*dry.mapping.cellSize*0.5;
  let centre=dry.mapping.worldOrigin+vec3f(base)*dry.mapping.cellSize+half-camera[0];
  let radius=length(half);let distanceSquared=dot(centre,centre);
  if(distanceSquared<=radius*radius){return level==0u;}
  let footprint=uniforms.viewport.y/cameraTanHalfFov()*radius/sqrt(distanceSquared-radius*radius);
  let cellFootprint=footprint/f32(n);
  var selected=0u;
  if(cellFootprint<=threshold){selected=min(meshLevelCount()-1u,u32(floor(log2(threshold/max(cellFootprint,1e-6)))));}
  return level==selected;
}
// Cull exact cached quad bounds before invoking any vertex shader. No
// occlusion or screen-size approximation: subpixel geometry is retained.
fn meshQuadVisible(index:u32)->bool{
  if(index>=atomicLoad(&meshState[4])){return false;}
  let quad=meshOutput[index];
  if(!meshQuadSelected(quad)){return false;}
  let face=meshQuadFace(quad.face);let axis=face/2u;
  let origin=dry.mapping.worldOrigin+vec3f(quad.origin)*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();
  let facing=(camera[0][axis]-origin[axis])*select(-1.0,1.0,(face&1u)!=0u);
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
  @location(3) @interpolate(flat) level:u32,
}
@vertex fn surfaceMeshVertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->MeshVertexOut{
  let quad=meshQuads[${culling ? "meshVisible[instance]" : "instance"}];
  let face=meshQuadFace(quad.face);let axis=face/2u;let u=(axis+1u)%3u;let v=(axis+2u)%3u;
  let corners=array<vec2u,4>(vec2u(1u,0u),vec2u(1u,1u),vec2u(0u,0u),vec2u(0u,1u));
  // Swapping the two in-plane coordinates reverses winding on negative
  // faces while preserving the 00–11 diagonal and the covered rectangle.
  let corner=select(corners[vertex],corners[vertex].yx,(face&1u)==0u);var lattice=quad.origin;lattice[u]+=corner.x*quad.extent[u];lattice[v]+=corner.y*quad.extent[v];
  let world=dry.mapping.worldOrigin+vec3f(lattice)*dry.mapping.cellSize;
  let camera=dryRasterPrimaryCamera();let relative=world-camera[0];let z=dot(relative,camera[1]);
  var position=vec4f(dot(relative,camera[2])/(cameraTanHalfFov()*uniforms.viewport.x/max(uniforms.viewport.y,1.0)),dot(relative,camera[3])/cameraTanHalfFov(),DRY_REVERSED_Z_NEAR_M,z);
  ${culling ? "" : "// Without the cull pass, an unselected level collapses to a zero-area strip.\n  if(!meshQuadSelected(quad)){position=vec4f(0.0,0.0,0.0,1.0);}"}
  var normal=vec3f(0.0);normal[axis]=select(-1.0,1.0,(face&1u)!=0u);
  return MeshVertexOut(position,world,quad.identity,normal,meshQuadLevel(quad.face));
}
struct MeshSurfaceOut {
  @location(0) packedSurface:vec4u,@location(1) identityMedia:vec4u,
  @location(2) geometry:vec4f,@location(3) opaqueIdentity:vec2u,
}
@fragment fn surfaceMeshFragment(input:MeshVertexOut)->MeshSurfaceOut{
  let camera=dryRasterPrimaryCamera();let rd=normalize(input.world-camera[0]);let t=length(input.world-camera[0]);
  var normal=input.normal;
  if(dryMeshLodPixels()>0.0){
    // Filtered detail shades the baked normal: a coarse quad carries its cells'
    // mean, an exact quad reads the voxel just behind its face. The surface
    // itself stays the quad, so depth is unchanged; a baked normal facing away
    // from the face keeps the face.
    var word=input.identity;
    if(input.level==0u){
      let lattice=(input.world-dry.mapping.worldOrigin)/dry.mapping.cellSize-input.normal*0.25;
      word=meshRegionAt(lattice).identity;
    }
    // Air's zero word is not the absent-normal sentinel, so solidity gates the read.
    if(sceneIdentitySolid(word)&&sceneIdentityHasNormal(word)){let baked=sceneIdentityNormal(word);if(dot(baked,input.normal)>-1e-4){normal=baked;}}
  }
  let shading=dryShadingNormal(input.identity,normal);
  let hit=DryHit(t,shading.normal,sceneIdentityMaterial(input.identity),DRY_OWNER_NONE,shading.featureId,DRY_GBUFFER_FIELD_VOXEL,DRY_GBUFFER_MOTION_STATIC,0u,0.0,vec3u(0u));
  let out=dryRasterPrimarySurface(hit,camera[0],rd,camera[1],SVO_GBUFFER_PRODUCER_BRICK);
  return MeshSurfaceOut(out.packedSurface,out.identityMedia,out.geometry,out.opaqueIdentity);
}
@fragment fn surfaceMeshBackground(input:VertexOut)->DryRasterPrimaryOut{
  dryRasterPrimaryReset();let camera=dryRasterPrimaryCamera();let rd=dryRasterPrimaryRay(input.position.xy,camera);
  var hit=missHit();
  // Exact planes are independent of the cached voxel mesh. During bounded
  // extraction, traverse the current published SVO so edits appear immediately
  // and unaffected room geometry remains visible. This uses the same voxel
  // authority, never the preceding mesh or an analytic approximation of it.
  hit=dryPlanarCatalogHit(camera[0],rd,0.0,DRY_MISS);
  var producer=SVO_GBUFFER_PRODUCER_RASTER_BACKGROUND;
  if(meshHeader[13]==0u||meshHeader[15]!=0u){
    let current=traceStatic(camera[0],rd);
    if(current.t<hit.t){hit=current;producer=SVO_GBUFFER_PRODUCER_BRICK;}
  }
  if(hit.t>=DRY_MISS){return dryRasterPrimaryMiss();}
  return dryRasterPrimarySurface(hit,camera[0],rd,camera[1],producer);
}
`;
}
