import { SVO_BRICK_OCCUPANCY } from "./svo-brick-occupancy";
import { svoFieldProgramAbsentWGSL, svoPrimitiveWGSL } from "./svo-primitive-abi";
import {
  SVO_PIXEL_TRACE_FLAGS,
  SVO_PIXEL_TRACE_HEADER,
  SVO_PIXEL_TRACE_HEADER_WORDS,
  SVO_PIXEL_TRACE_KINDS,
  SVO_PIXEL_TRACE_MAGIC,
  SVO_PIXEL_TRACE_PRIMARY_MODE,
  SVO_PIXEL_TRACE_RECORD_WORDS,
  SVO_PIXEL_TRACE_STAGES,
  SVO_PIXEL_TRACE_STATUS,
  SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS,
  decodeSvoPixelTrace,
  type SvoPixelTrace,
} from "./svo-pixel-trace";
import {
  sparseBrickBandedLeafCodecWGSL,
  sparseBrickSceneIdentityCodecWGSL,
  type SparseBrickScenePayloadLanes,
} from "./sparse-brick-octree";
import { cameraApertureShaderLibrary } from "../core/webgpu-camera";
import { SVO_BRICK_RASTER_CONTRACT, svoBrickRasterSharedWGSL } from "./webgpu-svo-brick-raster";

/**
 * The raster primary's answer to "what produced this pixel".
 *
 * Under `raster-primary` there is no per-pixel search to mirror. A compute pass
 * emitted one proxy box per resident occupied leaf, bucket-sorted them by view
 * depth, and the rasterizer's depth test picked the winner. So this probe does
 * not re-derive the search — it *reads the frame's own output*: the sorted
 * instance buffer the draw consumed and the sort-state counters the emission
 * pass wrote. That is the whole point. A probe that re-ran the emission would be
 * a second implementation free to drift from the first, which is exactly how the
 * traced probe came to describe a shader that no longer runs.
 *
 * What it does mirror is small and bounded: the ray/AABB clamp and the in-brick
 * DDA that each covering fragment ran. Those are lifted from
 * `traceLeafPayload`'s own law, over the same lattice, against the same shared
 * analytic intersector.
 *
 * It runs as one workgroup, once per frame, and writes a record buffer in the
 * shared pixel-trace ABI so the host decodes both probes with one decoder.
 */
export const SVO_BRICK_RASTER_PROBE_CONTRACT = Object.freeze({
  /**
   * Proxies recordable at one pixel. Overdraw is the count of bricks whose box a
   * single ray pierces; the raster handoff targets a median below ~4 and a
   * glancing ray through a dense stack is tens, not hundreds. Overflow is
   * counted and reported rather than silently dropped.
   */
  recordCapacity: 192,
  workgroupSize: 64,
  entryPoint: "svoBrickRasterProbeMain" as const,
  bindings: Object.freeze({
    uniforms: 0,
    params: 1,
    request: 2,
    structure: 3,
    /** The whole scene payload arena; identity is decoded through the shared codec. */
    scenePayload: 4,
    scene: 5,
    rasterPublication: 6,
    records: 10,
  }),
});

/** Words the probe's record texture holds, header included. */
export function svoBrickRasterProbeWordCount(): number {
  return SVO_PIXEL_TRACE_HEADER_WORDS
    + SVO_BRICK_RASTER_PROBE_CONTRACT.recordCapacity * SVO_PIXEL_TRACE_RECORD_WORDS;
}

export function svoBrickRasterProbeTextureRows(): number {
  return Math.ceil(svoBrickRasterProbeWordCount() / SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS);
}

export function svoBrickRasterProbeBufferBytes(): number {
  return svoBrickRasterProbeTextureRows() * SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4;
}

export function svoBrickRasterProbeBindGroupLayoutEntries(): GPUBindGroupLayoutEntry[] {
  const { bindings } = SVO_BRICK_RASTER_PROBE_CONTRACT;
  const visibility = GPUShaderStage.COMPUTE;
  return [
    { binding: bindings.uniforms, visibility, buffer: { type: "uniform" } },
    { binding: bindings.params, visibility, buffer: { type: "uniform" } },
    { binding: bindings.request, visibility, buffer: { type: "uniform" } },
    { binding: bindings.structure, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.scenePayload, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.scene, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.rasterPublication, visibility, buffer: { type: "read-only-storage" } },
    { binding: bindings.records, visibility, storageTexture: { access: "write-only", format: "r32uint" } },
  ];
}

export interface SvoBrickRasterProbeOptions {
  /**
   * Whether the shipping brick fragment writes `@builtin(frag_depth)`.
   *
   * This decides whether the covering-proxy count is exact or an upper bound,
   * and it is not a detail. A fragment that writes its own depth cannot be
   * rejected by early depth testing or by tile-based hidden-surface removal —
   * the hardware has to run it to find out what depth it would write. So in the
   * shipping configuration *every* covering proxy shades, and the count this
   * probe reports is the real fragment count. Only the `no-fragment-depth`
   * experiment makes it an upper bound.
   */
  readonly fragmentDepthWritten: boolean;
  readonly primitiveWordOffset?: number;
  readonly sortStateWordOffset?: number;
  readonly instanceWordOffset?: number;
  /**
   * `DryParams` in u32 words, and where scene identity's lane bases sit in it.
   *
   * The probe used to bind only the 64-byte `SvoMapping`+`metadata` prefix. It
   * now needs the payload lane bases too, and they have to be *read* rather than
   * baked: a module compiled against one arena's offsets and later bound to a
   * world rebuilt at another capacity would resolve a plausible word in the wrong
   * lane, which reads as a finding about the traversal rather than as the stale
   * binding it is. Passed in rather than imported because the layout lives with
   * the shader that owns the uniform, and that shader imports this module.
   */
  readonly paramsWordCount: number;
  readonly payloadLaneWordOffset: number;
  /** How this world stores scene identity; decides which decode is compiled. */
  readonly scenePayload?: SparseBrickScenePayloadLanes;
}

/**
 * Local copy of the occupancy decoder, so this module stays independent of the
 * dry-scene fragment composition exactly as the emission module does.
 */
const occupancyWGSL = /* wgsl */ `
struct SvoProbeOccupancy{ready:u32,occupied:u32,minInclusive:vec3u,maxInclusive:vec3u}
fn svoProbeOccupancyDecode(packed:u32)->SvoProbeOccupancy{
  return SvoProbeOccupancy(
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.readyBit}u)!=0u),
    select(0u,1u,(packed&${SVO_BRICK_OCCUPANCY.occupiedBit}u)!=0u),
    vec3u((packed>>8u)&7u,(packed>>11u)&7u,(packed>>14u)&7u),
    vec3u((packed>>17u)&7u,(packed>>20u)&7u,(packed>>23u)&7u));
}
`;

export function createSvoBrickRasterProbeWGSL(options: SvoBrickRasterProbeOptions): string {
  const primitiveWordOffset = options.primitiveWordOffset ?? 0;
  const sortStateWordOffset = options.sortStateWordOffset ?? 0;
  const instanceWordOffset = options.instanceWordOffset ?? 0;
  const { bindings, recordCapacity, workgroupSize, entryPoint } = SVO_BRICK_RASTER_PROBE_CONTRACT;
  // `SvoMapping` (12 words) plus `metadata` (4). Everything between that and the
  // payload lane bases is `DryParams` state this probe does not read, declared as
  // one named span rather than field by field.
  const PROBE_PARAMS_PREFIX_WORDS = 16;
  const reservedVectors = (options.payloadLaneWordOffset - PROBE_PARAMS_PREFIX_WORDS) / 4;
  if (!Number.isSafeInteger(reservedVectors) || reservedVectors < 0
    || options.paramsWordCount < options.payloadLaneWordOffset + 8) {
    throw new RangeError("Brick raster probe parameters must reach two whole vec4u lanes past the prefix");
  }
  const mode = options.scenePayload?.mode ?? "dense";
  const sceneIdentityWGSL = /* wgsl */ `${mode === "dense" ? "" : sparseBrickBandedLeafCodecWGSL({
    occupancyBase: "params.payloadLanes.x", recordMaskBase: "params.payloadLanes.y",
    headerBase: "params.payloadLanes.z", blobBase: "params.payloadLanes.w",
    recordsBase: "params.payloadLanes1.x",
    load: (index) => `scenePayload[${index}]`, mode, records: false,
  })}
${sparseBrickSceneIdentityCodecWGSL({
    mode, materialOwnerBase: "params.payloadLanes1.y",
    load: (index) => `scenePayload[${index}]`,
  })}
fn dryVoxelCapacity()->u32{return params.payloadLanes1.z;}`;
  const header = SVO_PIXEL_TRACE_HEADER;
  const kinds = SVO_PIXEL_TRACE_KINDS;
  const flags = SVO_PIXEL_TRACE_FLAGS;
  return /* wgsl */ `
// ---------------------------------------------------------------------------
// Raster-primary pixel probe. Reads the frame's own instance list and cull
// counters; mirrors only the per-fragment AABB clamp and in-brick DDA.
// ---------------------------------------------------------------------------
struct Uniforms{viewport:vec4f,cameraPosition:vec4f,cameraTarget:vec4f,container:vec4f,options:vec4f,gridInfo:vec4f,debug:vec4f,environment:vec4f,terrainMeta:vec4f,terrainFeatures:array<vec4f,16>}
struct SvoNode{address:vec4u,links:vec4u}
struct SvoLeaf{topology:vec4u}
struct SvoMapping{worldOrigin:vec3f,brickSize:u32,cellSize:vec3f,maximumDepth:u32,nodeCount:u32,leafCount:u32,maxVisits:u32,_padding:u32}
// Reserved: the DryParams lanes between \`metadata\` and the payload lane bases.
struct SvoProbeParams{mapping:SvoMapping,metadata:vec4u,reserved:array<vec4u,${reservedVectors}>,payloadLanes:vec4u,payloadLanes1:vec4u}
struct SvoBrickSortStateRead{
  drawVertexCount:u32,
  drawInstanceCount:u32,
  drawFirstVertex:u32,
  candidateCount:u32,
  culled:u32,
  empty:u32,
  resident:u32,
}
${svoBrickRasterSharedWGSL}
${occupancyWGSL}
// This probe reads the scene arena's primitive records and nothing else, so it
// has no field-program tape arena to resolve against. Saying so is the contract:
// a field-program record here reports itself invalid, exactly as an aggregate
// whose block never arrived does, rather than drawing its conservative box.
${svoFieldProgramAbsentWGSL}
${svoPrimitiveWGSL}

@group(0) @binding(${bindings.uniforms}) var<uniform> uniforms:Uniforms;
${cameraApertureShaderLibrary()}
@group(0) @binding(${bindings.params}) var<uniform> params:SvoProbeParams;
// x,y requested pixel; z request token; w non-zero arms the probe.
@group(0) @binding(${bindings.request}) var<uniform> probeRequest:vec4u;
@group(0) @binding(${bindings.structure}) var<storage,read> svoProbeStructure:array<u32>;
@group(0) @binding(${bindings.scenePayload}) var<storage,read> scenePayload:array<u32>;
@group(0) @binding(${bindings.scene}) var<storage,read> svoProbeScene:array<u32>;
@group(0) @binding(${bindings.rasterPublication}) var<storage,read> svoProbeRaster:array<u32>;
@group(0) @binding(${bindings.records}) var probeRecords:texture_storage_2d<r32uint,write>;
${sceneIdentityWGSL}

fn svoProbeWords4(source:ptr<storage,array<u32>,read>,offset:u32)->vec4u{return vec4u((*source)[offset],(*source)[offset+1u],(*source)[offset+2u],(*source)[offset+3u]);}
fn svoProbeControl(index:u32)->u32{return svoProbeStructure[index];}
fn svoProbeNode(index:u32)->SvoNode{let base=128u+index*8u;return SvoNode(svoProbeWords4(&svoProbeStructure,base),svoProbeWords4(&svoProbeStructure,base+4u));}
fn svoProbePrimitive(index:u32)->SvoPrimitiveRecord{let base=${primitiveWordOffset}u+index*16u;return SvoPrimitiveRecord(svoProbeWords4(&svoProbeScene,base),svoProbeWords4(&svoProbeScene,base+4u),bitcast<vec4f>(svoProbeWords4(&svoProbeScene,base+8u)),svoProbeWords4(&svoProbeScene,base+12u));}
fn svoProbeInstance(index:u32)->SvoBrickInstance{let base=${instanceWordOffset}u+index*8u;return SvoBrickInstance(bitcast<vec3f>(vec3u(svoProbeRaster[base],svoProbeRaster[base+1u],svoProbeRaster[base+2u])),svoProbeRaster[base+3u],bitcast<vec3f>(vec3u(svoProbeRaster[base+4u],svoProbeRaster[base+5u],svoProbeRaster[base+6u])),svoProbeRaster[base+7u]);}
fn svoProbeSortInstanceCount()->u32{return svoProbeRaster[${sortStateWordOffset + 1}u];}
fn svoProbeSortWord(index:u32)->u32{return svoProbeRaster[${sortStateWordOffset}u+index];}
fn svoProbeInstanceCapacity()->u32{return (arrayLength(&svoProbeRaster)-${instanceWordOffset}u)/8u;}

const PROBE_HEADER_WORDS:u32=${SVO_PIXEL_TRACE_HEADER_WORDS}u;
const PROBE_RECORD_WORDS:u32=${SVO_PIXEL_TRACE_RECORD_WORDS}u;
const PROBE_ROW_WORDS:u32=${SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS}u;
const PROBE_CAPACITY:u32=${recordCapacity}u;
const PROBE_MISS:f32=1e30;

fn probeWriteWord(wordIndex:u32,value:u32){
  textureStore(probeRecords,vec2u(wordIndex%PROBE_ROW_WORDS,wordIndex/PROBE_ROW_WORDS),vec4u(value,0u,0u,0u));
}
fn probeWriteFloat(wordIndex:u32,value:f32){probeWriteWord(wordIndex,bitcast<u32>(value));}

/**
 * One covering proxy, as gathered before the winner is known.
 *
 * Lanes scan the instance list strided and append here under a workgroup
 * atomic, so the append order is nondeterministic. That is fine and is in fact
 * the truth of the pass: the fragments were shaded concurrently. Ordering is
 * restored by sorting on the instance index, which is the draw order the
 * counting sort established.
 */
// Laid out vec3f-first so WGSL's 16-byte alignment packs this into 64 bytes.
// The whole array has to fit the *default* 16 KiB workgroup-storage limit,
// because that is what the renderer requests; a probe that only ran on devices
// granting more would be unavailable exactly where the diagnostic is wanted.
// The leaf box is not held per proxy for the same reason: only the winner's is
// ever drawn, so lane zero recomputes that one from the node index.
struct SvoProbeProxy{
  proxyMinimum:vec3f,
  proxyMaximum:vec3f,
  instanceIndex:u32,
  sortBucket:u32,
  cells:u32,
  nodeIndex:u32,
  voxelOffset:u32,
  surfaceT:f32,
  tEnter:f32,
  tExit:f32,
}
var<workgroup> probeProxies:array<SvoProbeProxy,${recordCapacity}>;
var<workgroup> probeProxyCount:atomic<u32>;
/**
 * Proxies past the workgroup array, and records past the buffer, counted apart.
 *
 * They mean different things and must not be added: a dropped *proxy* means the
 * covering count itself is a floor, while a dropped *record* only means the
 * drawing is a prefix. Folding them into one counter would let a long DDA
 * inflate the number of bricks the frame is reported to have drawn here.
 */
var<workgroup> probeOverflow:atomic<u32>;
var<workgroup> probeRecordDrops:atomic<u32>;
var<workgroup> probeCells:atomic<u32>;

fn svoProbeCompactMortonBits(value:vec3u)->vec3u{
  var compact=value&vec3u(0x49249249u);
  compact=(compact^(compact>>vec3u(2u)))&vec3u(0xc30c30c3u);
  compact=(compact^(compact>>vec3u(4u)))&vec3u(0x0f00f00fu);
  compact=(compact^(compact>>vec3u(8u)))&vec3u(0xff0000ffu);
  return (compact^(compact>>vec3u(16u)))&vec3u(0x0000ffffu);
}
fn svoProbeDecodeMorton(low:u32,high:u32,level:u32)->vec3u{
  let levelMask=(1u<<level)-1u;
  let lowBits=svoProbeCompactMortonBits(vec3u(low,low>>1u,low>>2u));
  let highBits=svoProbeCompactMortonBits(vec3u(high>>1u,high>>2u,high));
  return (lowBits|(highBits<<vec3u(11u,11u,10u)))&vec3u(levelMask);
}
/** Full leaf box. The DDA lattice is this, not the occupied sub-box drawn. */
fn svoProbeNodeBounds(node:SvoNode)->mat2x3f{
  let coordinate=vec3f(svoProbeDecodeMorton(node.address.x,node.address.y,node.address.z));
  let scale=f32((1u<<(params.mapping.maximumDepth-node.address.z))*params.mapping.brickSize);
  let minimum=params.mapping.worldOrigin+coordinate*scale*params.mapping.cellSize;
  return mat2x3f(minimum,minimum+scale*params.mapping.cellSize);
}

/** Slab test. Component x is non-zero on intersection; y and z are the interval. */
fn svoProbeRayAabb(ro:vec3f,inverseDirection:vec3f,bounds:mat2x3f)->vec3f{
  let t0=(bounds[0]-ro)*inverseDirection;
  let t1=(bounds[1]-ro)*inverseDirection;
  let near=min(t0,t1);
  let far=max(t0,t1);
  let enter=max(max(near.x,near.y),max(near.z,0.0));
  let exit=min(min(far.x,far.y),far.z);
  return select(vec3f(0.0,0.0,0.0),vec3f(1.0,enter,exit),exit>=enter);
}

/** Mirror of the dry-scene camera basis and the raster fragment's pixel ray. */
fn svoProbeRay()->vec3f{
  let ro=uniforms.cameraPosition.xyz;
  let forward=normalize(uniforms.cameraTarget.xyz-ro);
  let right=normalize(cross(forward,vec3f(0.0,1.0,0.0)));
  let up=normalize(cross(right,forward));
  let viewport=max(uniforms.viewport.xy,vec2f(1.0));
  let pixel=vec2f(f32(probeRequest.x)+.5,f32(probeRequest.y)+.5);
  let uv=vec2f(pixel.x/viewport.x,1.0-pixel.y/viewport.y);
  let ndc=uv*2.0-1.0;
  return normalize(forward+right*ndc.x*viewport.x/viewport.y*cameraTanHalfFov()+up*ndc.y*cameraTanHalfFov());
}

fn svoProbeVoxelIndex(voxelOffset:u32,local:vec3u,brickSize:u32)->u32{
  return voxelOffset+local.x+brickSize*(local.y+brickSize*local.z);
}

/**
 * The in-brick DDA one covering fragment ran.
 *
 * Structurally the shipping traceLeafPayload: the same lattice, the same
 * 32-step ceiling, the same owner-tag gate, the same tolerance, and the same
 * shared analytic intersector. Returns the surface distance, or PROBE_MISS when
 * the brick held nothing — which is a fragment that ran in full and discarded.
 */
/** Append one box-shaped record and advance the cursor, if capacity remains. */
fn svoProbeWriteCell(cursor:ptr<function,u32>,kind:u32,recordFlags:u32,
    minimum:vec3f,maximum:vec3f,tEnter:f32,tExit:f32){
  let index=*cursor;
  if(index>=PROBE_CAPACITY){atomicAdd(&probeRecordDrops,1u);return;}
  *cursor=index+1u;
  let base=PROBE_HEADER_WORDS+index*PROBE_RECORD_WORDS;
  probeWriteWord(base,kind);
  probeWriteWord(base+1u,0u);
  probeWriteWord(base+2u,0u);
  probeWriteWord(base+3u,recordFlags);
  probeWriteFloat(base+4u,minimum.x);
  probeWriteFloat(base+5u,minimum.y);
  probeWriteFloat(base+6u,minimum.z);
  probeWriteFloat(base+7u,maximum.x);
  probeWriteFloat(base+8u,maximum.y);
  probeWriteFloat(base+9u,maximum.z);
  probeWriteFloat(base+10u,tEnter);
  probeWriteFloat(base+11u,tExit);
}

struct SvoProbeDda{t:f32,cells:u32}
fn svoProbeTraceBrick(ro:vec3f,rd:vec3f,bounds:mat2x3f,voxelOffset:u32,tEnter:f32,tExit:f32,
    record:bool,cursor:ptr<function,u32>)->SvoProbeDda{
  let brickSize=params.mapping.brickSize;
  let extent=(bounds[1]-bounds[0])/f32(brickSize);
  var entry=max(tEnter,0.0);
  let point=ro+rd*(entry+1e-5);
  var cell=vec3i(clamp(floor((point-bounds[0])/extent),vec3f(0.0),vec3f(f32(brickSize-1u))));
  let step=select(vec3i(-1),vec3i(1),rd>=vec3f(0.0));
  let nextBoundary=bounds[0]+(vec3f(cell)+select(vec3f(0.0),vec3f(1.0),step>vec3i(0)))*extent;
  var nextT=select(vec3f(PROBE_MISS),(nextBoundary-ro)/rd,abs(rd)>vec3f(1e-9));
  let deltaT=select(vec3f(PROBE_MISS),abs(extent/rd),abs(rd)>vec3f(1e-9));
  let tolerance=length(extent)*1.05;
  var cells=0u;
  // Hoisted exactly as the fragment's own walk hoists it, so the probe mirrors
  // the loads the frame took rather than a second way of reaching the same word.
  let identitySource=sceneIdentitySourceAt(voxelOffset);
  for(var iteration=0u;iteration<32u;iteration+=1u){
    if(any(cell<vec3i(0))||any(cell>=vec3i(i32(brickSize)))||entry>tExit){break;}
    cells+=1u;
    let cellExit=min(min(nextT.x,nextT.y),min(nextT.z,tExit));
    let cellMinimum=bounds[0]+vec3f(cell)*extent;
    let payloadIndex=svoProbeVoxelIndex(voxelOffset,vec3u(cell),brickSize);
    var tagged=false;
    var found=false;
    var surface=PROBE_MISS;
    if(payloadIndex<dryVoxelCapacity()){
      let identity=sceneIdentityOf(identitySource,payloadIndex);
      // The probe used to resolve the cell's owner and march that record's exact
      // surface, mirroring a primary that no longer does either. A voxel carries
      // a baked normal in the high half of this word now, so the first solid cell
      // *is* the surface and the exact test it was tagging has no counterpart to
      // measure. Solidity is the low half, unchanged.
      if((identity&0xffffu)!=0u){
        tagged=true;
        found=true;
        surface=entry;
      }
    }
    // Recording is the winner's second pass only: the scanning lanes run
    // concurrently and the record texture has no atomic cursor, so exactly one
    // invocation may ever append.
    if(record){
      svoProbeWriteCell(cursor,${kinds.brickCell}u,select(0u,${flags.tagged}u,tagged),
        cellMinimum,cellMinimum+extent,entry,cellExit);
      if(tagged){
        svoProbeWriteCell(cursor,${kinds.exactTest}u,select(0u,${flags.hit}u,found),
          cellMinimum,cellMinimum+extent,entry,cellExit);
      }
    }
    if(found){return SvoProbeDda(surface,cells);}
    let advance=min(nextT.x,min(nextT.y,nextT.z));
    if(nextT.x<=advance+1e-6){cell.x+=step.x;nextT.x+=deltaT.x;}
    if(nextT.y<=advance+1e-6){cell.y+=step.y;nextT.y+=deltaT.y;}
    if(nextT.z<=advance+1e-6){cell.z+=step.z;nextT.z+=deltaT.z;}
    entry=advance;
  }
  return SvoProbeDda(PROBE_MISS,cells);
}

@compute @workgroup_size(${workgroupSize})
fn ${entryPoint}(@builtin(local_invocation_id) localId:vec3u){
  let lane=localId.x;
  if(lane==0u){
    atomicStore(&probeProxyCount,0u);
    atomicStore(&probeOverflow,0u);
    atomicStore(&probeRecordDrops,0u);
    atomicStore(&probeCells,0u);
  }
  workgroupBarrier();

  let ro=uniforms.cameraPosition.xyz;
  let rd=svoProbeRay();
  let inverseDirection=1.0/rd;
  let armed=probeRequest.w!=0u;
  // The draw count is the frame's own published indirect instance count, so the
  // set scanned here is exactly the set the rasterizer consumed.
  let drawn=min(svoProbeSortInstanceCount(),svoProbeInstanceCapacity());
  if(armed){
    for(var index=lane;index<drawn;index+=${workgroupSize}u){
      let record=svoProbeInstance(index);
      let proxy=mat2x3f(record.proxyMinimum,record.proxyMaximum);
      let interval=svoProbeRayAabb(ro,inverseDirection,proxy);
      // Not covering: the box's screen projection does not contain this pixel,
      // so the rasterizer produced no fragment here for it.
      if(interval.x==0.0||interval.z<0.0){continue;}
      let nodeIndex=record.nodeIndexKey&SVO_BRICK_NODE_INDEX_MASK;
      if(nodeIndex>=params.mapping.nodeCount){continue;}
      let leafBounds=svoProbeNodeBounds(svoProbeNode(nodeIndex));
      var unusedCursor=0u;
      let dda=svoProbeTraceBrick(ro,rd,leafBounds,record.voxelOffset,max(interval.y,0.0),interval.z,
        false,&unusedCursor);
      atomicAdd(&probeCells,dda.cells);
      let slot=atomicAdd(&probeProxyCount,1u);
      if(slot>=PROBE_CAPACITY){atomicAdd(&probeOverflow,1u);continue;}
      probeProxies[slot]=SvoProbeProxy(proxy[0],proxy[1],index,record.nodeIndexKey>>SVO_BRICK_SORT_KEY_SHIFT,
        dda.cells,nodeIndex,record.voxelOffset,dda.t,max(interval.y,0.0),interval.z);
    }
  }
  workgroupBarrier();

  if(lane!=0u){return;}

  let stored=min(atomicLoad(&probeProxyCount),PROBE_CAPACITY);
  // Restore draw order. Insertion sort: the count is an overdraw depth, so it is
  // small, and a stable order is what makes the drawn colour ramp mean anything.
  for(var i=1u;i<stored;i+=1u){
    let value=probeProxies[i];
    var j=i;
    loop{
      if(j==0u||probeProxies[j-1u].instanceIndex<=value.instanceIndex){break;}
      probeProxies[j]=probeProxies[j-1u];
      j-=1u;
    }
    probeProxies[j]=value;
  }

  // The depth test, replayed. Leaves partition space, so the intervals are
  // disjoint and totally ordered and nearest simply wins; submission order is a
  // performance lever, never a correctness one.
  var winner=0xffffffffu;
  var winnerT=PROBE_MISS;
  var withSurface=0u;
  for(var i=0u;i<stored;i+=1u){
    let candidate=probeProxies[i];
    if(!(candidate.surfaceT<PROBE_MISS)){continue;}
    withSurface+=1u;
    if(candidate.surfaceT<winnerT){winnerT=candidate.surfaceT;winner=i;}
  }

  var cursor=0u;
  var hsrEligible=0u;
  for(var i=0u;i<stored;i+=1u){
    let proxy=probeProxies[i];
    let isWinner=i==winner;
    let discarded=!(proxy.surfaceT<PROBE_MISS);
    var recordFlags=0u;
    if(isWinner){recordFlags|=${flags.depthWinner}u|${flags.hit}u;}
    else if(discarded){recordFlags|=${flags.discarded}u;}
    else{recordFlags|=${flags.depthLoser}u;}
    ${options.fragmentDepthWritten
      ? `// The shipping fragment writes @builtin(frag_depth), so the hardware
    // cannot reject it before it runs: every covering proxy shaded, and this
    // count is exact rather than an upper bound.`
      : `// Without a fragment-written depth the tiler may reject a proxy that
    // sits wholly behind the winner before it ever shades, so anything in that
    // position is reported as an upper bound.
    if(winner!=0xffffffffu&&!isWinner&&proxy.tEnter>winnerT){recordFlags|=${flags.hsrEligible}u;hsrEligible+=1u;}`}
    let base=PROBE_HEADER_WORDS+cursor*PROBE_RECORD_WORDS;
    // level packs the sort bucket in its low half and the cells this fragment
    // stepped in its high half; detail is the instance's index in draw order.
    probeWriteWord(base,${kinds.brickProxy}u);
    probeWriteWord(base+1u,(proxy.sortBucket&0xffffu)|(min(proxy.cells,0xffffu)<<16u));
    probeWriteWord(base+2u,proxy.instanceIndex);
    probeWriteWord(base+3u,recordFlags);
    probeWriteFloat(base+4u,proxy.proxyMinimum.x);
    probeWriteFloat(base+5u,proxy.proxyMinimum.y);
    probeWriteFloat(base+6u,proxy.proxyMinimum.z);
    probeWriteFloat(base+7u,proxy.proxyMaximum.x);
    probeWriteFloat(base+8u,proxy.proxyMaximum.y);
    probeWriteFloat(base+9u,proxy.proxyMaximum.z);
    probeWriteFloat(base+10u,proxy.tEnter);
    probeWriteFloat(base+11u,proxy.tExit);
    cursor+=1u;
    // The winner's full leaf box, so the occupied sub-box actually drawn is
    // legible against the brick it was carved from.
    if(isWinner&&cursor<PROBE_CAPACITY&&proxy.nodeIndex<params.mapping.nodeCount){
      let leafBounds=svoProbeNodeBounds(svoProbeNode(proxy.nodeIndex));
      let leafBase=PROBE_HEADER_WORDS+cursor*PROBE_RECORD_WORDS;
      probeWriteWord(leafBase,${kinds.leafBounds}u);
      probeWriteWord(leafBase+1u,proxy.sortBucket&0xffffu);
      probeWriteWord(leafBase+2u,proxy.instanceIndex);
      probeWriteWord(leafBase+3u,${flags.depthWinner}u);
      probeWriteFloat(leafBase+4u,leafBounds[0].x);
      probeWriteFloat(leafBase+5u,leafBounds[0].y);
      probeWriteFloat(leafBase+6u,leafBounds[0].z);
      probeWriteFloat(leafBase+7u,leafBounds[1].x);
      probeWriteFloat(leafBase+8u,leafBounds[1].y);
      probeWriteFloat(leafBase+9u,leafBounds[1].z);
      probeWriteFloat(leafBase+10u,proxy.tEnter);
      probeWriteFloat(leafBase+11u,proxy.tExit);
      cursor+=1u;
      // Replay the winning fragment's DDA, this time recording each cell it
      // stepped and each analytic test it issued. Replayed rather than captured
      // during the scan because only one invocation may append, and during the
      // scan every lane is walking a different brick at once.
      // The surface it finds is already known; this pass is run for its records,
      // so the result is discarded rather than compared.
      _=svoProbeTraceBrick(ro,rd,leafBounds,proxy.voxelOffset,proxy.tEnter,proxy.tExit,true,&cursor);
    }
  }

  let hasWinner=winner!=0xffffffffu;
  probeWriteWord(${header.magic}u,${SVO_PIXEL_TRACE_MAGIC}u);
  probeWriteWord(${header.status}u,select(${SVO_PIXEL_TRACE_STATUS.miss}u,${SVO_PIXEL_TRACE_STATUS.hit}u,hasWinner));
  // Produced minus dropped is what the host will believe was stored, so both
  // counts are of *records*: a proxy that never reached the array would have
  // produced one record, and the cells past the buffer produced their own.
  let recordDrops=atomicLoad(&probeOverflow)+atomicLoad(&probeRecordDrops);
  probeWriteWord(${header.recordCount}u,cursor+recordDrops);
  probeWriteWord(${header.droppedRecords}u,recordDrops);
  probeWriteWord(${header.pixelX}u,probeRequest.x);
  probeWriteWord(${header.pixelY}u,probeRequest.y);
  probeWriteWord(${header.requestToken}u,probeRequest.z);
  probeWriteFloat(${header.rayOrigin}u,ro.x);
  probeWriteFloat(${header.rayOrigin + 1}u,ro.y);
  probeWriteFloat(${header.rayOrigin + 2}u,ro.z);
  probeWriteFloat(${header.rayDirection}u,rd.x);
  probeWriteFloat(${header.rayDirection + 1}u,rd.y);
  probeWriteFloat(${header.rayDirection + 2}u,rd.z);
  probeWriteFloat(${header.hitDistance}u,select(0.0,winnerT,hasWinner));
  probeWriteFloat(${header.minimumVoxel}u,max(params.mapping.cellSize.x,max(params.mapping.cellSize.y,params.mapping.cellSize.z)));

  probeWriteWord(${header.primaryMode}u,${SVO_PIXEL_TRACE_PRIMARY_MODE.raster}u);
  probeWriteWord(${header.stagesPresent}u,${SVO_PIXEL_TRACE_STAGES.brickCull | SVO_PIXEL_TRACE_STAGES.brickRaster}u);
  // Cull counters come straight off the frame's own sort state: measured, not
  // mirrored. They are frame-wide and the host is careful to badge them so.
  probeWriteWord(${header.residentLeaves}u,svoProbeSortWord(7u));
  probeWriteWord(${header.emptyBricks}u,svoProbeSortWord(6u));
  probeWriteWord(${header.frustumCulled}u,svoProbeSortWord(5u));
  probeWriteWord(${header.candidatesEmitted}u,svoProbeSortWord(4u));
  probeWriteWord(${header.instancesDrawn}u,drawn);
  probeWriteWord(${header.coveringProxies}u,stored+atomicLoad(&probeOverflow));
  probeWriteWord(${header.proxiesWithSurface}u,withSurface);
  probeWriteWord(${header.hsrEligibleProxies}u,hsrEligible);
  probeWriteWord(${header.winnerInstanceIndex}u,select(0u,probeProxies[select(0u,winner,hasWinner)].instanceIndex,hasWinner));
  probeWriteWord(${header.winnerSortBucket}u,select(0u,probeProxies[select(0u,winner,hasWinner)].sortBucket,hasWinner));
  probeWriteWord(${header.ddaCellsAcrossProxies}u,atomicLoad(&probeCells));
  probeWriteWord(${header.prepassState}u,0u);
}
`;
}

/* ------------------------------------------------------------------------- */
/* Host side: the record texture and the readback ring.                        */
/* ------------------------------------------------------------------------- */

/**
 * Owns the raster probe's record texture and the mapped copies it is read back
 * through.
 *
 * The request uniform is *not* owned here: both probes answer the same pixel in
 * the same frame, and giving them one buffer is what guarantees they cannot
 * disagree about which pixel they were asked for. Two staging slots, for the
 * same reason the ray probe has two — the probe runs at most once per frame and
 * a request supersedes rather than queues.
 */
export class SparseVoxelBrickRasterProbeBuffers {
  readonly records: GPUTexture;
  readonly recordsView: GPUTextureView;
  private readonly rows = svoBrickRasterProbeTextureRows();
  private readonly staging: { buffer: GPUBuffer; pending: boolean }[];
  private pendingSlot?: { buffer: GPUBuffer; pending: boolean };
  private destroyed = false;

  constructor(private readonly device: GPUDevice) {
    this.records = device.createTexture({
      label: "Sparse voxel raster-primary probe records",
      size: [SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS, this.rows],
      format: "r32uint",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    this.recordsView = this.records.createView();
    const size = svoBrickRasterProbeBufferBytes();
    this.staging = Array.from({ length: 2 }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Sparse voxel raster-primary probe readback ${index + 1}/2`,
        size,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: false,
    }));
  }

  /** Copy the records into a free staging slot. False when both are in flight. */
  encodeReadback(encoder: GPUCommandEncoder): boolean {
    if (this.destroyed) return false;
    const slot = this.staging.find((candidate) => !candidate.pending);
    if (!slot) return false;
    slot.pending = true;
    encoder.copyTextureToBuffer(
      { texture: this.records },
      { buffer: slot.buffer, bytesPerRow: SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4, rowsPerImage: this.rows },
      { width: SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS, height: this.rows, depthOrArrayLayers: 1 },
    );
    this.pendingSlot = slot;
    return true;
  }

  /**
   * Resolve the most recently encoded readback. Decoded with the shared decoder
   * because the probe writes the shared ABI; a malformed buffer returns
   * `undefined` so the overlay draws nothing rather than drawing a lie.
   */
  async read(isCurrent: () => boolean): Promise<SvoPixelTrace | undefined> {
    const slot = this.pendingSlot;
    this.pendingSlot = undefined;
    if (!slot || this.destroyed) return undefined;
    try {
      await slot.buffer.mapAsync(GPUMapMode.READ);
      if (this.destroyed || !isCurrent()) return undefined;
      const bytes = slot.buffer.getMappedRange();
      return decodeSvoPixelTrace(new Uint32Array(bytes), new Float32Array(bytes));
    } catch {
      return undefined;
    } finally {
      try { slot.buffer.unmap(); } catch { /* Device loss or destruction. */ }
      slot.pending = false;
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    try { this.records.destroy(); } catch { /* Device loss. */ }
    for (const slot of this.staging) { try { slot.buffer.destroy(); } catch { /* Device loss. */ } }
  }
}
