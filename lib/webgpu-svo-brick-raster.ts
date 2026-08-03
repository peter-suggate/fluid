import { SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import { cameraApertureShaderLibrary } from "./webgpu-camera";

/**
 * Raster-assisted primary visibility: GPU-driven emission, frustum culling and
 * front-to-back ordering of one box instance per resident, occupied SVO leaf
 * brick (docs/SVO_RASTER_PRIMARY_HANDOFF.md).
 *
 * The instance box is the brick's *occupied sub-AABB*, decoded from the
 * occupancy word the sparse-brick producer already publishes in the terminal
 * node's `links.w`. Sub-boxes of disjoint boxes stay disjoint, so a ray still
 * meets each instance over one interval and those intervals remain totally
 * ordered — which is what makes depth resolution exact and makes ordering a
 * pure performance lever rather than a correctness one.
 *
 * Ordering is a bucket counting sort on the instance's minimum view depth.
 * It is deliberately approximate within a bucket: the depth test is the
 * authority, and the sort exists only so tile-based hidden-surface removal
 * gets to reject occluded bricks before shading them.
 */
export const SVO_BRICK_RASTER_CONTRACT = Object.freeze({
  /** vec3f proxy minimum + voxel offset, vec3f proxy maximum + node/key word. */
  instanceStrideBytes: 32,
  /** Two triangles per box face, drawn non-indexed like the rigid impostors. */
  verticesPerInstance: 36,
  /**
   * The box's far side is drawn, so a camera inside a brick still shades it.
   * The table below winds outward-CCW in world space; projection flips that to
   * clockwise in framebuffer coordinates for the near-facing triangles, which
   * WebGPU's default `ccw` front face then classifies as back faces.
   */
  cullMode: "back" as GPUCullMode,
  sortBuckets: 1024,
  sortKeyShift: 22,
  nodeIndexMask: (1 << 22) - 1,
  emitWorkgroupSize: 64,
  scanWorkgroupSize: 256,
  scatterWorkgroupSize: 64,
  /**
   * Fixed per-pixel conservative candidate arena. Garden's measured proxy
   * crossings are p90=9 and max=18 at 1500 square; twenty-four covers that
   * measured tail while an exact direct fallback handles every overflow.
   */
  coverageCandidatesPerPixel: 24,
  /**
   * packedSurface(16) + identityMedia(8) + splitGeometry(16) + splitIdentity(8).
   * Every plane the deferred lighting pass consumes is a depth-tested colour
   * attachment; nothing in this pass writes an untested storage texture.
   */
  colorAttachmentBytesPerSample: 48,
  colorAttachmentCount: 4,
  /** Group-zero bindings of the standalone emission/sort module. */
  bindings: Object.freeze({
    uniforms: 0,
    mapping: 1,
    structure: 2,
    candidates: 3,
    rasterPublication: 4,
  }),
  /** The dry-scene split group carries the sorted list into the vertex stage. */
  instanceDrawBinding: 2,
  coverageCountBinding: 3,
  coverageCandidateBinding: 4,
  /** `SvoMapping` prefix of `DryParams`; bound directly so the two cannot drift. */
  mappingBindingBytes: 48,
  sortStateHeaderWords: 8,
  entryPoints: Object.freeze({
    emit: "svoBrickEmitMain" as const,
    scan: "svoBrickScanMain" as const,
    scatter: "svoBrickScatterMain" as const,
    vertex: "svoBrickRasterVertex" as const,
    fragment: "svoBrickRasterFragment" as const,
    coverage: "svoBrickCoverageFragment" as const,
    resolve: "svoBrickCoverageResolveFragment" as const,
    overflowResolve: "svoBrickCoverageOverflowFragment" as const,
    background: "dryRasterPrimaryBackgroundMain" as const,
  }),
});

export function svoBrickRasterSortStateBytes(): number {
  return (SVO_BRICK_RASTER_CONTRACT.sortStateHeaderWords + SVO_BRICK_RASTER_CONTRACT.sortBuckets)
    * Uint32Array.BYTES_PER_ELEMENT;
}

export function svoBrickRasterPublicationInstanceOffsetBytes(): number {
  return Math.ceil(svoBrickRasterSortStateBytes() / 256) * 256;
}

export function svoBrickRasterInstanceBytes(leafCapacity: number): number {
  if (!Number.isSafeInteger(leafCapacity) || leafCapacity < 1) {
    throw new RangeError("Brick-raster leaf capacity must be a positive safe integer");
  }
  return leafCapacity * SVO_BRICK_RASTER_CONTRACT.instanceStrideBytes;
}

export function svoBrickRasterCoverageCountBytes(width: number, height: number): number {
  if (!Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
    throw new RangeError("Brick-raster coverage dimensions must be positive safe integers");
  }
  return width * height * Uint32Array.BYTES_PER_ELEMENT;
}

export function svoBrickRasterCoverageCandidateBytes(width: number, height: number): number {
  return svoBrickRasterCoverageCountBytes(width, height)
    * SVO_BRICK_RASTER_CONTRACT.coverageCandidatesPerPixel;
}

/** Standalone group-zero layout for the emission, scan and scatter passes. */
export function svoBrickRasterCullBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_BRICK_RASTER_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.uniforms, visibility, buffer: { type: "uniform" } },
    { binding: bindings.mapping, visibility, buffer: { type: "uniform" } },
    { binding: bindings.structure, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.candidates, visibility, buffer: { type: "storage" } },
    { binding: bindings.rasterPublication, visibility, buffer: { type: "storage" } },
  ];
}

/**
 * The direct control reads only the sorted instances. The production coverage
 * arm additionally appends instance indices per pixel, then reads the same
 * arena from its one-fragment-per-pixel resolve. Counts remain atomic because
 * overlapping proxy fragments are unordered by the rasterizer.
 */
export function svoBrickRasterDrawBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [{
    binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
    visibility: GPUShaderStage.VERTEX,
    buffer: { type: "read-only-storage" },
  }];
}

export function svoBrickRasterCoverageBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  return [
    { binding: SVO_BRICK_RASTER_CONTRACT.instanceDrawBinding,
      visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
      buffer: { type: "read-only-storage" } },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCountBinding,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: "storage" },
    },
    {
      binding: SVO_BRICK_RASTER_CONTRACT.coverageCandidateBinding,
      visibility: GPUShaderStage.FRAGMENT,
      buffer: { type: "storage" },
    },
  ];
}

/**
 * Outward, counter-clockwise unit-box corner indices. Corner `i` is
 * `(i & 1, (i >> 1) & 1, (i >> 2) & 1)`; the pipeline culls front faces, so
 * these wind so that the *back* faces survive.
 */
export const SVO_BRICK_RASTER_BOX_CORNERS: readonly number[] = Object.freeze([
  0, 4, 6, 0, 6, 2, // -X
  1, 3, 7, 1, 7, 5, // +X
  0, 1, 5, 0, 5, 4, // -Y
  2, 6, 7, 2, 7, 3, // +Y
  0, 2, 3, 0, 3, 1, // -Z
  4, 5, 7, 4, 7, 6, // +Z
]);

/** Shared declarations: the instance record, the sort state, and the box table. */
export const svoBrickRasterSharedWGSL = /* wgsl */ `
struct SvoBrickInstance{proxyMinimum:vec3f,voxelOffset:u32,proxyMaximum:vec3f,nodeIndexKey:u32}
const SVO_BRICK_SORT_BUCKETS:u32=${SVO_BRICK_RASTER_CONTRACT.sortBuckets}u;
const SVO_BRICK_SORT_KEY_SHIFT:u32=${SVO_BRICK_RASTER_CONTRACT.sortKeyShift}u;
const SVO_BRICK_NODE_INDEX_MASK:u32=${SVO_BRICK_RASTER_CONTRACT.nodeIndexMask}u;
fn svoBrickBoxCorner(vertexIndex:u32)->vec3f{
  var table=array<u32,${SVO_BRICK_RASTER_CONTRACT.verticesPerInstance}>(${SVO_BRICK_RASTER_BOX_CORNERS.map((corner) => `${corner}u`).join(",")});
  let corner=table[vertexIndex];
  return vec3f(f32(corner&1u),f32((corner>>1u)&1u),f32((corner>>2u)&1u));
}
`;

/**
 * Emission, prefix scan and scatter. Deliberately standalone: it needs the
 * camera uniform, the published topology and the `SvoMapping` prefix of
 * `DryParams`, none of the renderer's fragment-only shading bindings.
 */
export function createSvoBrickRasterCullWGSL(options: { reversedZNear_m: number }): string {
  const { bindings } = SVO_BRICK_RASTER_CONTRACT;
  return /* wgsl */ `
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f,environment:vec4f,terrainMeta:vec4f,terrainFeatures:array<vec4f,16>}
struct SvoNode{address:vec4u,links:vec4u}
struct SvoLeaf{topology:vec4u}
struct SvoMapping{worldOrigin:vec3f,brickSize:u32,cellSize:vec3f,maximumDepth:u32,nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32}
struct SvoBrickSortState{
  drawVertexCount:u32,
  drawInstanceCount:atomic<u32>,
  drawFirstVertex:u32,
  drawFirstInstance:u32,
  candidateCount:atomic<u32>,
  culled:atomic<u32>,
  empty:atomic<u32>,
  resident:atomic<u32>,
  buckets:array<atomic<u32>,${SVO_BRICK_RASTER_CONTRACT.sortBuckets}>,
}
${svoBrickRasterSharedWGSL}
${svoBrickOccupancyDecodeWGSL}
struct SvoBrickRasterPublication{
  sort:SvoBrickSortState,
  _instanceAlignment:array<u32,${(svoBrickRasterPublicationInstanceOffsetBytes() - svoBrickRasterSortStateBytes()) / 4}>,
  instances:array<SvoBrickInstance>,
}

@group(0) @binding(${bindings.uniforms}) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(${bindings.mapping}) var<uniform> mapping:SvoMapping;
@group(0) @binding(${bindings.structure}) var<storage,read> svoBrickStructure:array<u32>;
@group(0) @binding(${bindings.candidates}) var<storage,read_write> svoBrickCandidates:array<SvoBrickInstance>;
@group(0) @binding(${bindings.rasterPublication}) var<storage,read_write> svoBrickRaster:SvoBrickRasterPublication;

fn svoBrickStructureWords4(offset:u32)->vec4u{return vec4u(svoBrickStructure[offset],svoBrickStructure[offset+1u],svoBrickStructure[offset+2u],svoBrickStructure[offset+3u]);}
fn svoBrickControl(index:u32)->u32{return svoBrickStructure[index];}
fn svoBrickNode(index:u32)->SvoNode{let base=128u+index*8u;return SvoNode(svoBrickStructureWords4(base),svoBrickStructureWords4(base+4u));}
fn svoBrickLeaf(index:u32)->SvoLeaf{let base=128u+svoBrickControl(16u)+index*4u;return SvoLeaf(svoBrickStructureWords4(base));}

const SVO_BRICK_NEAR_M:f32=${options.reversedZNear_m};

fn svoBrickCompactMortonBits(value:vec3u)->vec3u{
  var compact=value&vec3u(0x49249249u);
  compact=(compact^(compact>>vec3u(2u)))&vec3u(0xc30c30c3u);
  compact=(compact^(compact>>vec3u(4u)))&vec3u(0x0f00f00fu);
  compact=(compact^(compact>>vec3u(8u)))&vec3u(0xff0000ffu);
  return (compact^(compact>>vec3u(16u)))&vec3u(0x0000ffffu);
}
fn svoBrickDecodeMorton(low:u32,high:u32,level:u32)->vec3u{
  let levelMask=(1u<<level)-1u;
  let lowBits=svoBrickCompactMortonBits(vec3u(low,low>>1u,low>>2u));
  let highBits=svoBrickCompactMortonBits(vec3u(high>>1u,high>>2u,high));
  return (lowBits|(highBits<<vec3u(11u,11u,10u)))&vec3u(levelMask);
}
fn svoBrickNodeBounds(node:SvoNode)->mat2x3f{
  let coordinate=vec3f(svoBrickDecodeMorton(node.address.x,node.address.y,node.address.z));
  let scale=f32((1u<<(mapping.maximumDepth-node.address.z))*mapping.brickSize);
  let minimum=mapping.worldOrigin+coordinate*scale*mapping.cellSize;
  return mat2x3f(minimum,minimum+scale*mapping.cellSize);
}
struct SvoBrickCamera{origin:vec3f,forward:vec3f,right:vec3f,up:vec3f,aspect:f32}
fn svoBrickCamera()->SvoBrickCamera{
  let origin=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-origin);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  return SvoBrickCamera(origin,forward,right,normalize(cross(right,forward)),
    uniforms.viewport.x/max(uniforms.viewport.y,1.0));
}
/** Conservative plane rejection: a box survives unless every corner is outside one plane. */
fn svoBrickFrustumVisible(camera:SvoBrickCamera,bounds:mat2x3f)->bool{
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
    outsideNear=outsideNear&&(viewDepth<SVO_BRICK_NEAR_M);
    outside=vec4<bool>(
      outside.x&&(x+lateral*camera.aspect<0.0),
      outside.y&&(lateral*camera.aspect-x<0.0),
      outside.z&&(y+lateral<0.0),
      outside.w&&(lateral-y<0.0));
  }
  return !(outsideNear||any(outside));
}
/** Minimum view depth of the box; the counting-sort key is a quantization of it. */
fn svoBrickSortKey(camera:SvoBrickCamera,bounds:mat2x3f)->u32{
  let center=0.5*(bounds[0]+bounds[1]);
  let halfExtent=0.5*(bounds[1]-bounds[0]);
  let nearDepth=dot(center-camera.origin,camera.forward)-dot(abs(camera.forward),halfExtent);
  let rootScale=f32((1u<<mapping.maximumDepth)*mapping.brickSize);
  let rootHalf=0.5*rootScale*mapping.cellSize;
  let rootCenter=mapping.worldOrigin+rootHalf;
  let farDepth=dot(rootCenter-camera.origin,camera.forward)+dot(abs(camera.forward),rootHalf);
  let normalized=max(nearDepth,0.0)/max(farDepth,1e-3);
  return min(u32(f32(SVO_BRICK_SORT_BUCKETS)*clamp(normalized,0.0,1.0)),SVO_BRICK_SORT_BUCKETS-1u);
}
fn svoBrickTopologyPublished()->bool{
  return svoBrickStructure[64u]!=0u;
}

@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.emitWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.emit}(@builtin(global_invocation_id) globalId:vec3u){
  let leafIndex=globalId.x;
  if(!svoBrickTopologyPublished()){return;}
  if(leafIndex>=svoBrickControl(1u)||leafIndex>=arrayLength(&svoBrickCandidates)){return;}
  let leaf=svoBrickLeaf(leafIndex).topology;
  let nodeIndex=leaf.x;
  if(nodeIndex>=svoBrickControl(0u)||nodeIndex>SVO_BRICK_NODE_INDEX_MASK){return;}
  atomicAdd(&svoBrickRaster.sort.resident,1u);
  let node=svoBrickNode(nodeIndex);
  let occupancy=svoBrickOccupancyDecode(node.links.w);
  // An empty brick can never produce a primary hit, so it is never drawn. The
  // producer publishes this word every topology change, which is why the
  // instance list can be rebuilt from scratch each frame instead of cached.
  if(occupancy.ready!=0u&&occupancy.occupied==0u){atomicAdd(&svoBrickRaster.sort.empty,1u);return;}
  let bounds=svoBrickNodeBounds(node);
  let cellSize=(bounds[1]-bounds[0])/f32(mapping.brickSize);
  var proxy=bounds;
  if(occupancy.ready!=0u){proxy=svoBrickOccupiedBounds(occupancy,bounds[0],cellSize);}
  let camera=svoBrickCamera();
  if(!svoBrickFrustumVisible(camera,proxy)){atomicAdd(&svoBrickRaster.sort.culled,1u);return;}
  let key=svoBrickSortKey(camera,proxy);
  let slot=atomicAdd(&svoBrickRaster.sort.candidateCount,1u);
  if(slot>=arrayLength(&svoBrickCandidates)){return;}
  svoBrickCandidates[slot]=SvoBrickInstance(proxy[0],leaf.y,proxy[1],nodeIndex|(key<<SVO_BRICK_SORT_KEY_SHIFT));
  atomicAdd(&svoBrickRaster.sort.buckets[key],1u);
}

var<workgroup> svoBrickBucketScan:array<u32,${SVO_BRICK_RASTER_CONTRACT.sortBuckets}>;
var<workgroup> svoBrickLaneTotals:array<u32,${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}>;
const SVO_BRICK_BUCKETS_PER_LANE:u32=${SVO_BRICK_RASTER_CONTRACT.sortBuckets / SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}u;

/** Exclusive prefix sum over the bucket histogram; also publishes the draw count. */
@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.scan}(@builtin(local_invocation_id) localId:vec3u){
  let lane=localId.x;
  let base=lane*SVO_BRICK_BUCKETS_PER_LANE;
  var laneTotal=0u;
  for(var offset=0u;offset<SVO_BRICK_BUCKETS_PER_LANE;offset+=1u){
    let value=atomicLoad(&svoBrickRaster.sort.buckets[base+offset]);
    svoBrickBucketScan[base+offset]=laneTotal;
    laneTotal+=value;
  }
  svoBrickLaneTotals[lane]=laneTotal;
  workgroupBarrier();
  // Hillis-Steele over the lane totals; ${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize} lanes is a fixed log2 rounds.
  var stride=1u;
  loop{
    if(stride>=${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize}u){break;}
    var added=svoBrickLaneTotals[lane];
    if(lane>=stride){added+=svoBrickLaneTotals[lane-stride];}
    workgroupBarrier();
    svoBrickLaneTotals[lane]=added;
    workgroupBarrier();
    stride*=2u;
  }
  let laneExclusive=svoBrickLaneTotals[lane]-laneTotal;
  for(var offset=0u;offset<SVO_BRICK_BUCKETS_PER_LANE;offset+=1u){
    atomicStore(&svoBrickRaster.sort.buckets[base+offset],svoBrickBucketScan[base+offset]+laneExclusive);
  }
  if(lane==${SVO_BRICK_RASTER_CONTRACT.scanWorkgroupSize - 1}u){
    atomicStore(&svoBrickRaster.sort.drawInstanceCount,min(svoBrickLaneTotals[lane],arrayLength(&svoBrickRaster.instances)));
  }
}

@compute @workgroup_size(${SVO_BRICK_RASTER_CONTRACT.scatterWorkgroupSize})
fn ${SVO_BRICK_RASTER_CONTRACT.entryPoints.scatter}(@builtin(global_invocation_id) globalId:vec3u){
  let candidate=globalId.x;
  if(candidate>=atomicLoad(&svoBrickRaster.sort.candidateCount)||candidate>=arrayLength(&svoBrickCandidates)){return;}
  let record=svoBrickCandidates[candidate];
  let key=record.nodeIndexKey>>SVO_BRICK_SORT_KEY_SHIFT;
  let slot=atomicAdd(&svoBrickRaster.sort.buckets[key],1u);
  if(slot>=arrayLength(&svoBrickRaster.instances)){return;}
  svoBrickRaster.instances[slot]=record;
}
`;
}

/** Local copy of the occupancy decoder so the cull module stays self-contained. */
const svoBrickOccupancyDecodeWGSL = /* wgsl */ `
struct SvoBrickOccupancy{ready:u32,occupied:u32,macroMask:u32,minInclusive:vec3u,maxInclusive:vec3u}
fn svoBrickOccupancyDecode(packed:u32)->SvoBrickOccupancy{
  return SvoBrickOccupancy(
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.readyBit}u)!=0u),
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.occupiedBit}u)!=0u),
    packed&0xffu,
    vec3u((packed>>8u)&7u,(packed>>11u)&7u,(packed>>14u)&7u),
    vec3u((packed>>17u)&7u,(packed>>20u)&7u,(packed>>23u)&7u));
}
fn svoBrickOccupiedBounds(summary:SvoBrickOccupancy,brickMinimum:vec3f,cellSize:vec3f)->mat2x3f{
  let minimum=brickMinimum+vec3f(summary.minInclusive)*cellSize;
  let maximum=brickMinimum+vec3f(summary.maxInclusive+vec3u(1u))*cellSize;
  return mat2x3f(minimum,maximum);
}
`;
