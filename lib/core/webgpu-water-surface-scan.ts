import { ADAPTIVE_SURFACE_TRIANGLE_CODE } from "./webgpu-water-adaptive-mesh";
import { GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE } from "./webgpu-water-global-fine-classify";
import { GLOBAL_FINE_SURFACE_EMIT_LANES, GLOBAL_FINE_SURFACE_TRIANGLES_PER_LANE, globalFineScanBindingsWGSL } from "./webgpu-water-global-fine-tetra";

export const SURFACE_SCAN_BLOCK_SIZE = 64;

/** GPU-authored launches. At least one classifier group visits the empty
 * publication so it can retire the previously drawn mesh. Validation remains
 * in the classifier; an invalid publication must retain that mesh. */
export const surfaceClassifyDispatchShader = /* wgsl */ `
struct P{sample:vec4u,bricks:vec4u,table:vec4u}
@group(0)@binding(0)var<storage,read>worklist:array<u32>;
@group(0)@binding(1)var<uniform>p:P;
@group(0)@binding(2)var<storage,read_write>dispatch:array<u32>;
@compute @workgroup_size(1)fn prepareClassify(){
  let pages=min(worklist[1],min(p.table.x,p.table.z));
  let groups=max(1u,(pages*p.bricks.w+255u)/256u);
  dispatch[0]=min(groups,65535u);dispatch[1]=(groups+65534u)/65535u;dispatch[2]=1u;
}
`;

/** One contour evaluation per cube, a local prefix, then a prefix over block
 * totals. No contour traversal runs in the single-workgroup totals pass. The
 * final add keeps the public offsets and emission shader unchanged. */
export const parallelSurfaceScanShader = /* wgsl */ `
${globalFineScanBindingsWGSL}
@group(0)@binding(11)var<storage,read_write>dispatch:array<u32>;
@group(0)@binding(18)var<storage,read_write>blocks:array<u32>;
var<workgroup>prefix:array<u32,256>;
fn cubeCount()->u32{
  if(atomicLoad(&args.vertexAllocator)==0xffffffffu){return 0u;}
  return min(atomicLoad(&args.activeCubeCount),min(arrayLength(&cubes),min(arrayLength(&offsets)/${GLOBAL_FINE_SURFACE_EMIT_LANES}u,arrayLength(&values)/2u)));
}
fn triangles(i:u32)->u32{
  let descriptor=cubes[i].y>>16u;let raw=descriptor&255u;
  if(raw==${ADAPTIVE_SURFACE_TRIANGLE_CODE}u){return 1u;}
  let lo=values[i*2u];let hi=values[i*2u+1u];
  if(raw==${GLOBAL_FINE_HEIGHTFIELD_DESCRIPTOR_CODE}u){return heightFieldTriangleCount(lo);}
  let samples=array<f32,8>(lo.x,lo.y,lo.z,lo.w,hi.x,hi.y,hi.z,hi.w);
  if(raw>=224u){return wallTriangleCount(samples,descriptor);}
  let transition=(raw&128u)!=0u;
  if(!transition&&((descriptor>>8u)&7u)!=0u){return 2u;}
  return contourTriangleCount(samples,select(0u,(descriptor>>8u)&63u,transition));
}
@compute @workgroup_size(1)fn prepareSurfaceScan(){
  let groups=(cubeCount()+63u)/64u;
  dispatch[3]=min(groups,65535u);dispatch[4]=max(1u,(groups+65534u)/65535u);dispatch[5]=1u;
}
@compute @workgroup_size(64)fn countSurfaceBlocks(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let block=wg.x+wg.y*65535u;let i=block*64u+lid;let count=cubeCount();
  var vertices=0u;if(i<count){vertices=3u*triangles(i);}
  prefix[lid]=vertices;workgroupBarrier();
  for(var stride=1u;stride<64u;stride*=2u){
    var add=0u;if(lid>=stride){add=prefix[lid-stride];}
    workgroupBarrier();prefix[lid]+=add;workgroupBarrier();
  }
  if(lid==63u){blocks[block]=prefix[63];}
  if(i>=count){return;}
  let start=prefix[lid]-vertices;
  for(var lane=0u;lane<${GLOBAL_FINE_SURFACE_EMIT_LANES}u;lane+=1u){offsets[i*${GLOBAL_FINE_SURFACE_EMIT_LANES}u+lane]=start+min(vertices,lane*${3 * GLOBAL_FINE_SURFACE_TRIANGLES_PER_LANE}u);}
}
@compute @workgroup_size(256)fn scanSurfaceBlocks(@builtin(local_invocation_index)lid:u32){
  let count=cubeCount();let n=(count+63u)/64u;
  let base=n/256u;let extra=n%256u;let begin=lid*base+min(lid,extra);
  let end=begin+base+select(0u,1u,lid<extra);var sum=0u;
  for(var i=begin;i<end;i+=1u){let value=blocks[i];blocks[i]=sum;sum+=value;}
  prefix[lid]=sum;workgroupBarrier();
  for(var stride=1u;stride<256u;stride*=2u){
    var add=0u;if(lid>=stride){add=prefix[lid-stride];}
    workgroupBarrier();prefix[lid]+=add;workgroupBarrier();
  }
  let start=prefix[lid]-sum;
  for(var i=begin;i<end;i+=1u){blocks[i]+=start;}
  if(lid==255u){
    dispatch[0]=(count+63u)/64u;dispatch[1]=${GLOBAL_FINE_SURFACE_EMIT_LANES}u;dispatch[2]=1u;
    if(atomicLoad(&args.vertexAllocator)!=0xffffffffu){
      let capacity=arrayLength(&out)-arrayLength(&out)%3u;
      atomicStore(&args.vertexCount,min(prefix[255],capacity));atomicStore(&args.vertexAllocator,prefix[255]);
      if(p.table.y!=6u){atomicStore(&args.meshPublicationGeneration,p.table.w);}
    }
  }
}
@compute @workgroup_size(64)fn addSurfaceBlockOffsets(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){
  let block=wg.x+wg.y*65535u;let i=block*64u+lid;if(i>=cubeCount()){return;}
  for(var lane=0u;lane<${GLOBAL_FINE_SURFACE_EMIT_LANES}u;lane+=1u){offsets[i*${GLOBAL_FINE_SURFACE_EMIT_LANES}u+lane]+=blocks[block];}
}
`;
