import { cameraApertureShaderLibrary } from "../core/webgpu-camera";
import { SPARSE_BRICK_LEAF_TERMINAL } from "./sparse-brick-octree";
import { SVO_BRICK_RASTER_BOX_CORNERS } from "./webgpu-svo-brick-raster";

/**
 * The conservative entry-depth prepass: a rasterized lower bound on where the
 * octree march may begin, computed once per pixel before the megakernel runs.
 *
 * The megakernel starts every ray at the root AABB the camera sits inside, so a
 * flat scene pays a full root-to-leaf descent per pixel to discover that the
 * only thing in front of it is the exact planar catalogue it already holds.
 * This pass answers the same question by projection instead of by descent: draw
 * one box per *voxel-bearing* leaf, keep the nearest box entry per pixel, and
 * hand that distance to `traceStaticFrom` as the cursor's `tMin`.
 *
 * Three properties make it a skipped computation rather than an approximation
 * of one, and each is load-bearing:
 *
 *   - **Only voxel leaves are drawn.** A planar terminal owns no voxel payload;
 *     its surface is already the exact seed the megakernel starts from
 *     (`dryPlanarCatalogHit`). Drawing the stage floor here would pull the
 *     recorded entry back to the floor on every floor pixel and delete the win
 *     the pass exists for.
 *   - **The proxy is the leaf's own node box, padded outward.** The cursor
 *     derives leaf bounds by halving the root box, this kernel by decoding the
 *     node's Morton address; the two agree to a few ULP and not exactly, so the
 *     box is grown by half a fine cell. Growth only ever lowers the recorded
 *     entry, which is the safe direction, and it is ~10^4 times the
 *     disagreement it covers.
 *   - **The recorded entry is biased down.** Reverse-Z resolves the per-pixel
 *     minimum, and two boxes whose entries differ by less than one float ULP of
 *     depth resolve arbitrarily. A 1e-4 relative bias is far above that tie
 *     window and far below one voxel.
 *
 * Together those give the consumer's contract: **the true first voxel hit on a
 * pixel's ray is never nearer than the recorded entry, and a pixel with nothing
 * drawn has no voxel leaf on its ray at all.** The second half is what lets the
 * megakernel skip the cursor outright on a sky or floor pixel.
 *
 * Deliberately standalone, exactly like the brick-raster cull it borrows the
 * box table from: the camera uniform, the published topology and the
 * `SvoMapping` prefix of `DryParams` are all it reads, so it is independent of
 * cone-lighting scale and of the megakernel's fragment-only shading bindings.
 */
export const SVO_PRIMARY_ENTRY_PREPASS_CONTRACT = Object.freeze({
  /** `x` = bitcast entry distance in metres, `y` = winning node index + 1. */
  seedFormat: "rg32uint" as GPUTextureFormat,
  /**
   * `y == 0` is the cleared value and means "no voxel leaf covers this ray".
   * It is a positive statement, not an absence: the pass is encoded whenever
   * the primary is, and the readiness gate withdraws the whole split bundle
   * rather than let a stale or unwritten plane be read as one.
   */
  emptyKey: 0,
  /** vec3f padded proxy minimum + node index, vec3f padded proxy maximum + spare. */
  instanceStrideBytes: 32,
  /** Two triangles per box face, drawn non-indexed like the brick proxies. */
  verticesPerInstance: 36,
  /** The far side is drawn so a camera inside a leaf still produces a fragment. */
  cullMode: "back" as GPUCullMode,
  emitWorkgroupSize: 64,
  /** `[vertexCount, instanceCount, firstVertex, firstInstance]` at buffer offset 0. */
  drawArgsWords: 4,
  /** Group-zero bindings. The cull and the draw share the uniform pair. */
  bindings: Object.freeze({
    uniforms: 0,
    mapping: 1,
    structure: 2,
    /** Read-write draw args + instances, compute stage only. */
    publication: 3,
    /** The same instances, read-only, from the vertex stage. */
    instances: 4,
  }),
  /** `SvoMapping` prefix of `DryParams`; bound directly so the two cannot drift. */
  mappingBindingBytes: 48,
  /**
   * Where the seed plane joins the dry-scene split visibility group.
   *
   * Free in both group layouts that share that index: the visibility outputs
   * hold 0 and 4, the lighting inputs 1, 5 and 6, and the retired raster-primary
   * arm 2, 3, 4 and 6 through 8.
   */
  seedBinding: 10,
  /** Half a fine cell of outward growth on every axis. See the note above. */
  proxyPadCells: 0.5,
  /** Relative bias applied to the recorded entry before it is published. */
  entryBias: 1e-4,
  entryPoints: Object.freeze({
    emit: "svoPrimaryEntryEmitMain" as const,
    vertex: "svoPrimaryEntryVertex" as const,
    fragment: "svoPrimaryEntryFragment" as const,
  }),
});

/** Draw args live at word zero; instances start on the next 256-byte boundary. */
export function svoPrimaryEntryInstanceOffsetBytes(): number {
  return 256;
}

export function svoPrimaryEntryPublicationBytes(leafCapacity: number): number {
  if (!Number.isSafeInteger(leafCapacity) || leafCapacity < 1) {
    throw new RangeError("Primary entry prepass leaf capacity must be a positive safe integer");
  }
  return svoPrimaryEntryInstanceOffsetBytes()
    + leafCapacity * SVO_PRIMARY_ENTRY_PREPASS_CONTRACT.instanceStrideBytes;
}

/** Group-zero layout of the emission pass. */
export function svoPrimaryEntryCullBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_PRIMARY_ENTRY_PREPASS_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.uniforms, visibility, buffer: { type: "uniform" } },
    { binding: bindings.mapping, visibility, buffer: { type: "uniform" } },
    { binding: bindings.structure, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.publication, visibility, buffer: { type: "storage" } },
  ];
}

/** Group-zero layout of the depth draw. */
export function svoPrimaryEntryDrawBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_PRIMARY_ENTRY_PREPASS_CONTRACT;
  return [
    { binding: bindings.uniforms, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    { binding: bindings.mapping, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
    { binding: bindings.instances, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
  ];
}

/**
 * The whole pass: emission, the proxy vertex stage and the depth fragment.
 *
 * One module, three entry points, two pipeline layouts — the compute entry
 * reaches the read-write publication and the draw entries reach the read-only
 * instance view of the same buffer, so neither layout carries a binding its own
 * stage cannot legally declare.
 */
export function createSvoPrimaryEntryPrepassWGSL(options: { reversedZNear_m: number }): string {
  const contract = SVO_PRIMARY_ENTRY_PREPASS_CONTRACT;
  const { bindings } = contract;
  return /* wgsl */ `
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f,environment:vec4f,terrainMeta:vec4f,terrainFeatures:array<vec4f,16>}
struct SvoNode{address:vec4u,links:vec4u}
struct SvoLeaf{topology:vec4u}
struct SvoMapping{worldOrigin:vec3f,brickSize:u32,cellSize:vec3f,maximumDepth:u32,nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32}
struct SvoPrimaryEntryInstance{proxyMinimum:vec3f,nodeIndex:u32,proxyMaximum:vec3f,_spare:u32}
struct SvoPrimaryEntryDrawArgs{vertexCount:u32,instanceCount:atomic<u32>,firstVertex:u32,firstInstance:u32}
struct SvoPrimaryEntryPublication{
  args:SvoPrimaryEntryDrawArgs,
  _alignment:array<u32,${(svoPrimaryEntryInstanceOffsetBytes() - 16) / 4}>,
  instances:array<SvoPrimaryEntryInstance>,
}

@group(0) @binding(${bindings.uniforms}) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(${bindings.mapping}) var<uniform> mapping:SvoMapping;
@group(0) @binding(${bindings.structure}) var<storage,read> svoEntryStructure:array<u32>;
@group(0) @binding(${bindings.publication}) var<storage,read_write> svoEntryPublication:SvoPrimaryEntryPublication;
@group(0) @binding(${bindings.instances}) var<storage,read> svoEntryInstances:array<SvoPrimaryEntryInstance>;

const SVO_ENTRY_NEAR_M:f32=${options.reversedZNear_m};
const SVO_ENTRY_INVALID:u32=0xffffffffu;
const SVO_ENTRY_PLANAR_TERMINAL:u32=${SPARSE_BRICK_LEAF_TERMINAL.planarBoundary}u;
const SVO_ENTRY_PAD_CELLS:f32=${contract.proxyPadCells};
const SVO_ENTRY_BIAS:f32=${contract.entryBias};

fn svoEntryWords4(offset:u32)->vec4u{return vec4u(svoEntryStructure[offset],svoEntryStructure[offset+1u],svoEntryStructure[offset+2u],svoEntryStructure[offset+3u]);}
fn svoEntryControl(index:u32)->u32{return svoEntryStructure[index];}
fn svoEntryNode(index:u32)->SvoNode{let base=128u+index*8u;return SvoNode(svoEntryWords4(base),svoEntryWords4(base+4u));}
fn svoEntryLeaf(index:u32)->SvoLeaf{let base=128u+svoEntryControl(16u)+index*4u;return SvoLeaf(svoEntryWords4(base));}
fn svoEntryTopologyPublished()->bool{return svoEntryStructure[64u]!=0u;}

fn svoEntryCompactMortonBits(value:vec3u)->vec3u{
  var compact=value&vec3u(0x49249249u);
  compact=(compact^(compact>>vec3u(2u)))&vec3u(0xc30c30c3u);
  compact=(compact^(compact>>vec3u(4u)))&vec3u(0x0f00f00fu);
  compact=(compact^(compact>>vec3u(8u)))&vec3u(0xff0000ffu);
  return (compact^(compact>>vec3u(16u)))&vec3u(0x0000ffffu);
}
fn svoEntryDecodeMorton(low:u32,high:u32,level:u32)->vec3u{
  let levelMask=(1u<<level)-1u;
  let lowBits=svoEntryCompactMortonBits(vec3u(low,low>>1u,low>>2u));
  let highBits=svoEntryCompactMortonBits(vec3u(high>>1u,high>>2u,high));
  return (lowBits|(highBits<<vec3u(11u,11u,10u)))&vec3u(levelMask);
}
fn svoEntryNodeBounds(node:SvoNode)->mat2x3f{
  let coordinate=vec3f(svoEntryDecodeMorton(node.address.x,node.address.y,node.address.z));
  let scale=f32((1u<<(mapping.maximumDepth-node.address.z))*mapping.brickSize);
  let minimum=mapping.worldOrigin+coordinate*scale*mapping.cellSize;
  return mat2x3f(minimum,minimum+scale*mapping.cellSize);
}
struct SvoEntryCamera{origin:vec3f,forward:vec3f,right:vec3f,up:vec3f,aspect:f32}
fn svoEntryCamera()->SvoEntryCamera{
  let origin=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-origin);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  return SvoEntryCamera(origin,forward,right,normalize(cross(right,forward)),
    uniforms.viewport.x/max(uniforms.viewport.y,1.0));
}
/** Conservative plane rejection: a box survives unless every corner is outside one plane. */
fn svoEntryFrustumVisible(camera:SvoEntryCamera,bounds:mat2x3f)->bool{
  var outside=vec4<bool>(true,true,true,true);
  var outsideNear=true;
  for(var corner=0u;corner<8u;corner+=1u){
    let point=vec3f(
      select(bounds[0].x,bounds[1].x,(corner&1u)!=0u),
      select(bounds[0].y,bounds[1].y,(corner&2u)!=0u),
      select(bounds[0].z,bounds[1].z,(corner&4u)!=0u));
    let relative=point-camera.origin;
    let viewDepth=dot(relative,camera.forward);
    let lateral=viewDepth*cameraTanHalfFov();
    let x=dot(relative,camera.right);
    let y=dot(relative,camera.up);
    outsideNear=outsideNear&&(viewDepth<SVO_ENTRY_NEAR_M);
    outside=vec4<bool>(
      outside.x&&(x+lateral*camera.aspect<0.0),
      outside.y&&(lateral*camera.aspect-x<0.0),
      outside.z&&(y+lateral<0.0),
      outside.w&&(lateral-y<0.0));
  }
  return !(outsideNear||any(outside));
}

// One instance per leaf the megakernel's cursor can actually report as a voxel
// payload, and no others.
//
// The reachability test is the traversal's own, transcribed: the cursor reports
// a leaf only through the node whose \`links.z\` names it and only when the leaf's
// back-reference agrees (webgpu-svo-traversal.ts, svoTraversalContinuationNext),
// so a leaf that fails either test is unreachable and drawing it would only
// weaken the seed. The planar rejection is \`dryPrimaryLeafResolve\`'s: a leaf
// whose terminal word says planar resolves to the exact catalogue patch or to a
// skip, and neither owns a voxel the cursor could find.
@compute @workgroup_size(${contract.emitWorkgroupSize})
fn ${contract.entryPoints.emit}(@builtin(global_invocation_id) globalId:vec3u){
  let leafIndex=globalId.x;
  if(!svoEntryTopologyPublished()){return;}
  if(leafIndex>=svoEntryControl(1u)){return;}
  let leaf=svoEntryLeaf(leafIndex).topology;
  if(leaf.z==SVO_ENTRY_PLANAR_TERMINAL){return;}
  let nodeIndex=leaf.x;
  if(nodeIndex>=svoEntryControl(0u)){return;}
  let node=svoEntryNode(nodeIndex);
  if(node.links.z!=leafIndex){return;}
  let bounds=svoEntryNodeBounds(node);
  let pad=mapping.cellSize*SVO_ENTRY_PAD_CELLS;
  let proxy=mat2x3f(bounds[0]-pad,bounds[1]+pad);
  let camera=svoEntryCamera();
  if(!svoEntryFrustumVisible(camera,proxy)){return;}
  let slot=atomicAdd(&svoEntryPublication.args.instanceCount,1u);
  if(slot>=arrayLength(&svoEntryPublication.instances)){return;}
  svoEntryPublication.instances[slot]=SvoPrimaryEntryInstance(proxy[0],nodeIndex,proxy[1],0u);
}

fn svoEntryBoxCorner(vertexIndex:u32)->vec3f{
  var table=array<u32,${contract.verticesPerInstance}>(${SVO_BRICK_RASTER_BOX_CORNERS.map((corner) => `${corner}u`).join(",")});
  let corner=table[vertexIndex];
  return vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32((corner>>2u)&1u));
}

struct SvoPrimaryEntryVertexOut{
  @builtin(position) position:vec4f,
  @location(0) @interpolate(flat) proxyMinimum:vec3f,
  @location(1) @interpolate(flat) proxyMaximum:vec3f,
  @location(2) @interpolate(flat) nodeIndex:u32,
}
struct SvoPrimaryEntryOut{
  @location(0) seed:vec2u,
  @builtin(frag_depth) hardwareDepth:f32,
}

@vertex fn ${contract.entryPoints.vertex}(@builtin(vertex_index) vertexIndex:u32,@builtin(instance_index) instanceIndex:u32)->SvoPrimaryEntryVertexOut{
  let record=svoEntryInstances[instanceIndex];
  let ro=uniforms.cameraPosition.xyz;let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));let up=normalize(cross(right,forward));
  let aspect=uniforms.viewport.x/max(uniforms.viewport.y,1.0);
  var position=vec4f(0.0,0.0,0.0,1.0);
  // Back faces let a camera inside a leaf still produce a fragment, but they
  // clip once the box reaches the near plane. Those few instances cover the
  // screen instead; the fragment's own box intersection rejects the pixels
  // their proxy does not actually contain.
  let margin=vec3f(${4 * options.reversedZNear_m});
  if(all(ro>=record.proxyMinimum-margin)&&all(ro<=record.proxyMaximum+margin)){
    var screen=array<vec2f,3>(vec2f(-1.0,-1.0),vec2f(-1.0,3.0),vec2f(3.0,-1.0));
    if(vertexIndex<3u){position=vec4f(screen[vertexIndex],1.0,1.0);}
  }else{
    let world=mix(record.proxyMinimum,record.proxyMaximum,svoEntryBoxCorner(vertexIndex));
    let relative=world-ro;let viewDepth=dot(relative,forward);
    // Constant clip-space z with w = view depth is exactly the reversed-Z
    // infinite-far projection: the interpolated depth is near/viewDepth.
    position=vec4f(dot(relative,right)/(aspect*cameraTanHalfFov()),dot(relative,up)/cameraTanHalfFov(),SVO_ENTRY_NEAR_M,viewDepth);
  }
  return SvoPrimaryEntryVertexOut(position,record.proxyMinimum,record.proxyMaximum,record.nodeIndex);
}

fn svoEntryRay(pixel:vec2f,camera:SvoEntryCamera)->vec3f{
  let uv=vec2f(pixel.x/max(uniforms.viewport.x,1.0),1.0-pixel.y/max(uniforms.viewport.y,1.0));let ndc=uv*2.0-1.0;
  return normalize(camera.forward+camera.right*ndc.x*camera.aspect*cameraTanHalfFov()+camera.up*ndc.y*cameraTanHalfFov());
}
/** The traversal's own slab test, so the box the cursor sees and the box this pass records agree. */
fn svoEntryRayAabb(origin:vec3f,direction:vec3f,bounds:mat2x3f)->vec3f{
  let inverseDirection=1.0/direction;
  var enter=0.0;
  var exit=3.4028234e38;
  for(var axis=0u;axis<3u;axis+=1u){
    if(direction[axis]==0.0){
      if(origin[axis]<bounds[0][axis]||origin[axis]>bounds[1][axis]){return vec3f(0.0);}
    }else{
      let first=(bounds[0][axis]-origin[axis])*inverseDirection[axis];
      let second=(bounds[1][axis]-origin[axis])*inverseDirection[axis];
      enter=max(enter,min(first,second));
      exit=min(exit,max(first,second));
      if(exit<enter){return vec3f(0.0);}
    }
  }
  return vec3f(1.0,enter,exit);
}

// Reverse-Z with \`greater\` resolves the per-pixel minimum: the nearest entry is
// the largest \`near/viewDepth\`, so the surviving fragment's colour is the
// nearest leaf's.
//
// Two values are deliberately floored to zero rather than published as written.
// A fragment whose entry sits inside the near plane would clamp its depth to 1
// and tie with every other such fragment, and the winner among ties is
// unordered — so the entry it publishes must be one no tie can make wrong, and
// that is zero. The published entry is otherwise biased down by a relative
// 1e-4, which is above the depth format's tie window and far below one voxel.
@fragment fn ${contract.entryPoints.fragment}(input:SvoPrimaryEntryVertexOut)->SvoPrimaryEntryOut{
  let camera=svoEntryCamera();
  let rd=svoEntryRay(input.position.xy,camera);
  let interval=svoEntryRayAabb(camera.origin,rd,mat2x3f(input.proxyMinimum,input.proxyMaximum));
  if(interval.x==0.0){discard;}
  let key=input.nodeIndex+1u;
  let viewDepth=interval.y*max(dot(rd,camera.forward),1e-6);
  if(!(viewDepth>SVO_ENTRY_NEAR_M)){return SvoPrimaryEntryOut(vec2u(bitcast<u32>(0.0),key),1.0);}
  let biased=max(interval.y-max(1e-5,interval.y*SVO_ENTRY_BIAS),0.0);
  return SvoPrimaryEntryOut(vec2u(bitcast<u32>(biased),key),SVO_ENTRY_NEAR_M/viewDepth);
}
`;
}
