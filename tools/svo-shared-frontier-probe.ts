import assert from "node:assert/strict";

/** Benchmark-only conservative tile-frustum traversal. No renderer defaults. */
export function sharedFrontierShader(source: string, width: number, height: number, x: number, y: number, depth: number, frontierCapacity = 64, cooperative = false) {
  const begin = source.indexOf("fn svoTraversalContinuationBegin(");
  const end = source.indexOf("fn svoTraversalContinuationNext(", begin);
  assert.ok(begin >= 0 && end > begin);
  const insertion = "  (*continuation).current = SvoStackEntry(0u, rootInterval.y, rootInterval.z);";
  const body = source.slice(begin, end);
  assert.ok(body.includes(insertion));
  source = source.slice(0, begin) + body.replace(insertion, `
  if(probeSharedEnabled && probeFrontierValid!=0u){
    var failed=false;
    for(var i=0u;i<probeFrontierCount;i+=1u){
      let candidate=probeFrontier[i];
      let interval=svoRayAabbWithInverse(ray,(*continuation).inverseDirection,mat2x3f(candidate.lower,candidate.upper));
      if(interval.x==0.0){continue;}
      if((*continuation).stackSize==SVO_STACK_CAPACITY){failed=true;break;}
      let entry=SvoStackEntry(candidate.index,interval.y,interval.z);
      var slot=(*continuation).stackSize;
      loop {
        if(slot==0u){break;}
        let previous=(*continuation).stack[slot-1u];
        let nearer=entry.tEnter<previous.tEnter || (entry.tEnter==previous.tEnter &&
          (entry.tExit<previous.tExit || (entry.tExit==previous.tExit && entry.nodeIndex<=previous.nodeIndex)));
        if(nearer){break;}
        (*continuation).stack[slot]=previous;slot-=1u;
      }
      (*continuation).stack[slot]=entry;(*continuation).stackSize+=1u;
    }
    if(!failed){
      probeSharedUsed=true;
      svoTraversalContinuationAdvance(continuation);
      return;
    }
    (*continuation).stackSize=0u;
  }
${insertion}`) + source.slice(end);
  // An overflow later in fine traversal triggers a complete canonical retrace.
  source = source.replaceAll("(*continuation).status = SVO_STATUS_STACK_OVERFLOW;",
    "probeSharedRetry=probeSharedUsed; (*continuation).status = SVO_STATUS_STACK_OVERFLOW;");
  source += `
struct ProbeFrontierEntry { lower:vec3f,index:u32,upper:vec3f,padding:u32 }
var<workgroup> probeFrontier:array<ProbeFrontierEntry,64>;
var<workgroup> probePending:array<u32,64>;
var<workgroup> probeFrontierCount:u32;
var<workgroup> probeFrontierValid:u32;
var<private> probeSharedEnabled:bool;
var<private> probeSharedUsed:bool;
var<private> probeSharedRetry:bool;

fn probePlaneVisible(bounds:mat2x3f,origin:vec3f,normal:vec3f)->bool{
  let center=(bounds[0]+bounds[1])*0.5-origin;
  let extent=(bounds[1]-bounds[0])*0.5;
  // Outward expansion covers floating point classification error. False positives
  // cost work; false negatives would lose geometry.
  let tolerance=1e-4*(1.0+length(center)+length(extent))*length(normal);
  return dot(center,normal)+dot(extent,abs(normal)) >= -tolerance;
}
fn probeBuildFrontier(groupId:vec3u){
  probeFrontierValid=0u;probeFrontierCount=0u;
  let mapping=dryConfiguredMapping();
  if(mapping.nodeCount==0u || svoControlLoad(12u)!=0u){return;}
  let ro=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0,1,0)));
  let up=normalize(cross(right,forward));
  // Pixel outer edges bound every pixel-center ray, including partial edge tiles.
  let minimum=vec2f(groupId.xy*vec2u(${x}u,${y}u));
  let maximum=min(minimum+vec2f(${x}.0,${y}.0),vec2f(${width}.0,${height}.0));
  let scale=cameraTanHalfFov();
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  let left=(minimum.x/${width}.0*2.0-1.0)*aspect*scale;
  let rightEdge=(maximum.x/${width}.0*2.0-1.0)*aspect*scale;
  let bottom=(1.0-maximum.y/${height}.0*2.0)*scale;
  let top=(1.0-minimum.y/${height}.0*2.0)*scale;
  probePending[0]=0u;var pendingCount=1u;var visits=0u;
  loop{
    if(pendingCount==0u){break;}
    visits+=1u;if(visits>1024u){return;}
    pendingCount-=1u;let index=probePending[pendingCount];
    if(index>=mapping.nodeCount){return;}
    let node=svoNodeLoad(index);let bounds=svoNodeBounds(node,mapping);
    if(!probePlaneVisible(bounds,ro,right-forward*left) ||
       !probePlaneVisible(bounds,ro,forward*rightEdge-right) ||
       !probePlaneVisible(bounds,ro,up-forward*bottom) ||
       !probePlaneVisible(bounds,ro,forward*top-up) ||
       !probePlaneVisible(bounds,ro,forward)){continue;}
    if(node.links.z!=SVO_INVALID || node.address.z>=${depth}u){
      if(probeFrontierCount==${frontierCapacity}u){return;}
      probeFrontier[probeFrontierCount]=ProbeFrontierEntry(bounds[0],index,bounds[1],0u);
      probeFrontierCount+=1u;continue;
    }
    let mask=node.address.w&255u;let count=countOneBits(mask);
    if(count==0u){continue;}
    if(node.links.x==SVO_INVALID || count!=node.links.y || node.links.x+count>mapping.nodeCount){return;}
    if(pendingCount+count>64u){return;}
    for(var child=0u;child<count;child+=1u){probePending[pendingCount]=node.links.x+child;pendingCount+=1u;}
  }
  probeFrontierValid=1u;
}
`;
  if (cooperative) {
    // Fixed-depth breadth-first expansion: all lanes help with the tile's upper
    // nodes. Atomics only allocate workgroup-local slots; no device queue.
    const cameraBegin = source.lastIndexOf("  let ro=uniforms.cameraPosition.xyz;");
    const cameraEnd = source.indexOf("  probePending[0]=0u;", cameraBegin);
    assert.ok(cameraBegin >= 0 && cameraEnd > cameraBegin);
    const camera = source.slice(cameraBegin, cameraEnd);
    source += `
var<workgroup> probePendingOther:array<u32,64>;
var<workgroup> probeLevelCounts:array<atomic<u32>,2>;
var<workgroup> probeFrontierAtomic:atomic<u32>;
var<workgroup> probeBuildError:atomic<u32>;
fn probeBuildFrontierParallel(groupId:vec3u,localId:vec3u){
  let lane=localId.y*${x}u+localId.x;
  let mapping=dryConfiguredMapping();
${camera}
  if(lane==0u){
    probeFrontierValid=0u;probeFrontierCount=0u;probePending[0]=0u;
    atomicStore(&probeLevelCounts[0],select(0u,1u,mapping.nodeCount!=0u && svoControlLoad(12u)==0u));
    atomicStore(&probeLevelCounts[1],0u);atomicStore(&probeFrontierAtomic,0u);atomicStore(&probeBuildError,0u);
  }
  workgroupBarrier();
  for(var level=0u;level<=${depth}u;level+=1u){
    let current=level&1u;let next=current^1u;
    if(lane==0u){atomicStore(&probeLevelCounts[next],0u);}
    workgroupBarrier();
    let count=min(atomicLoad(&probeLevelCounts[current]),64u);
    for(var i=lane;i<count;i+=${x * y}u){
      var index=0u;
      if(current==0u){index=probePending[i];}else{index=probePendingOther[i];}
      if(index>=mapping.nodeCount){atomicStore(&probeBuildError,1u);continue;}
      let node=svoNodeLoad(index);let bounds=svoNodeBounds(node,mapping);
      if(!probePlaneVisible(bounds,ro,right-forward*left) ||
         !probePlaneVisible(bounds,ro,forward*rightEdge-right) ||
         !probePlaneVisible(bounds,ro,up-forward*bottom) ||
         !probePlaneVisible(bounds,ro,forward*top-up) ||
         !probePlaneVisible(bounds,ro,forward)){continue;}
      if(node.links.z!=SVO_INVALID || node.address.z>=${depth}u){
        let slot=atomicAdd(&probeFrontierAtomic,1u);
        if(slot<64u){probeFrontier[slot]=ProbeFrontierEntry(bounds[0],index,bounds[1],0u);}
        else{atomicStore(&probeBuildError,1u);}
        continue;
      }
      let mask=node.address.w&255u;let children=countOneBits(mask);
      if(children==0u){continue;}
      if(node.links.x==SVO_INVALID || children!=node.links.y || node.links.x+children>mapping.nodeCount){atomicStore(&probeBuildError,1u);continue;}
      let slot=atomicAdd(&probeLevelCounts[next],children);
      if(slot+children>64u){atomicStore(&probeBuildError,1u);continue;}
      for(var child=0u;child<children;child+=1u){
        if(next==0u){probePending[slot+child]=node.links.x+child;}
        else{probePendingOther[slot+child]=node.links.x+child;}
      }
    }
    workgroupBarrier();
  }
  if(lane==0u){probeFrontierCount=min(atomicLoad(&probeFrontierAtomic),64u);probeFrontierValid=select(0u,1u,atomicLoad(&probeBuildError)==0u);}
}
`;
  }
  // Other raster entry points share the cursor library but are unused here.
  // Remove their stage annotations because workgroup storage is compute-only.
  return source.replaceAll("@fragment fn ", "fn ").replace(/->\s*@location\(\d+\)/g, "->");
}
