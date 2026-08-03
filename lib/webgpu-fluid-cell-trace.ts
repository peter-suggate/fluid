/**
 * Gathers everything the frame published about the pressure cell under one pixel.
 *
 * The request is a pixel, exactly as in `webgpu-svo-pixel-trace.ts`: the host
 * says where the pointer is and the shader marches that camera ray to decide
 * which cell it means. Reusing the pixel as the selection keeps the gesture the
 * same as the ray probe's — hover to follow, click to pin — and avoids a
 * CPU-side picking path that would have to duplicate the camera transform.
 *
 * What it gathers is state, not history. `fluid-cell-trace.ts` explains why a
 * cell cannot be replayed the way a ray can; this pass supplies the `gathered`
 * half of that contract and the host supplies the `scheduled` half.
 *
 * The fine-band summary is a bounded probe lattice rather than a full sweep: a
 * 32³ leaf at fine factor four contains over two million samples, and no
 * interactive diagnostic should walk them. The probe count is published so the
 * HUD can say so rather than implying an exact census.
 */
import { FINE_FLOOD_SAMPLE_FLAGS, FINE_FLOOD_SAMPLE_FLAG_BITS } from "./fine-flood-provenance";
import type { Vec3 } from "./model";
import { OCTREE_COARSE_PHI_FLAG } from "./octree-coarse-levelset";
import {
  FLUID_CELL_TRACE_FINE_FLAGS,
  FLUID_CELL_TRACE_FINE_RECORD,
  FLUID_CELL_TRACE_FINE_RECORDS_OFFSET,
  FLUID_CELL_TRACE_FINE_RECORD_CAPACITY,
  FLUID_CELL_TRACE_FINE_RECORD_EDGE,
  FLUID_CELL_TRACE_FINE_RECORD_WORDS,
  FLUID_CELL_TRACE_HEADER,
  FLUID_CELL_TRACE_HEADER_WORDS,
  FLUID_CELL_TRACE_HIT,
  FLUID_CELL_TRACE_HITS_OFFSET,
  FLUID_CELL_TRACE_HIT_CAPACITY,
  FLUID_CELL_TRACE_HIT_FLAGS,
  FLUID_CELL_TRACE_HIT_WORDS,
  FLUID_CELL_TRACE_MAGIC,
  FLUID_CELL_TRACE_NEIGHBOR_CAPACITY,
  FLUID_CELL_TRACE_RECORD,
  FLUID_CELL_TRACE_RECORD_FLAGS,
  FLUID_CELL_TRACE_RECORD_WORDS,
  FLUID_CELL_TRACE_STATUS,
  FLUID_CELL_TRACE_WORDS,
  decodeFluidCellTrace,
  type FluidCellTrace,
} from "./fluid-cell-trace";
import { octreeTechniqueSharedWGSL } from "./webgpu-octree-technique-shared";
import { makeFineLevelSetSortedWorklistLookupWGSL } from "./webgpu-octree-fine-levelset-bricks";
import type { OctreeTechniqueDebugSource } from "./octree-technique-debug";
import { PassBroker } from "./webgpu-pass-broker";
import { fineLevelSetPackedSampleWGSL } from "./fine-levelset-packed-sample";

/** Probe lattice edge; 8³ = 512 probes spread over the leaf. */
export const FLUID_CELL_TRACE_FINE_PROBE_EDGE = 8;
export const FLUID_CELL_TRACE_FINE_PROBES =
  FLUID_CELL_TRACE_FINE_PROBE_EDGE ** 3;
const LANES = 64;
const CONFIG_BYTES = 80;

/**
 * The hard storage-buffer budget of every cell-trace compute stage.
 *
 * The trace is an ordered gather now: row authority, coarse correction, fine
 * addressing, fine values, and fine provenance are separate stages.  No stage
 * binds publications it does not read, and no future diagnostic is allowed to
 * grow back toward the device's ten-buffer ceiling.
 */
export const FLUID_CELL_TRACE_STORAGE_BINDINGS: readonly string[] = Object.freeze([
  "authorityA", "authorityB", "scratch", "trace",
]);

export const FLUID_CELL_TRACE_STAGE_STORAGE_BINDINGS = Object.freeze({
  gatherCore: Object.freeze(["headers", "metrics", "pressure", "trace"]),
  gatherCoarse: Object.freeze(["coarsePhi", "trace"]),
  discoverFine: Object.freeze(["fineWorklist", "fineMetadata", "scratch", "trace"]),
  gatherFineValues: Object.freeze(["fineSamples", "scratch", "trace"]),
  gatherFineSeeds: Object.freeze(["fineSeeds", "scratch"]),
  resolveFineSeeds: Object.freeze(["fineMetadata", "fineSamples", "scratch", "trace"]),
} as const);

export const fluidCellTraceGatherShader = /* wgsl */ `
${octreeTechniqueSharedWGSL}
struct TraceConfig {
  dimensions:vec3u, rowCapacity:u32,
  pixel:vec2u, requestToken:u32, hasFine:u32,
  /** Step along the ray run to describe; clamped to the run the march finds. */
  hitIndex:u32, hasAim:u32, pad0:u32, pad1:u32,
  /**
   * The aim, in world space, as the host froze it. A pixel is not a selection:
   * the same pixel names a different cell as soon as the camera moves, which is
   * how a pinned cell used to drift out from under the pin while orbiting. The
   * host builds this with \`viewportRayForPixel\`, the documented inverse of
   * \`cameraRay\`, so a live aim is the pixel's own ray and a frozen one is the
   * ray that pixel had at the moment of the pin.
   */
  aimOrigin:vec3f, pad2:f32,
  aimDirection:vec3f, pad3:f32,
}
struct LeafHeader { cell:u32,entryStart:u32,entryCount:u32,size:u32,diagonal:f32,rhs:f32,pad0:u32,pad1:u32,gradient:vec4f }
struct Metric { topologyCode:u32, transformAndFlags:u32, volume:f32, reserved:u32 }
struct FineParams { brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,domainOrigin:vec3f,fineCellWidth:f32,worklistCapacity:u32,worklistHeaderWords:u32,pageCapacity:u32,generation:u32,activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32 }
/** One record per compact row, as \`webgpu-octree-coarse-levelset.ts\` writes it. */
struct CoarsePhi { phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 }

@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var ownerRows:texture_3d<u32>;
@group(0) @binding(2) var<uniform> config:TraceConfig;
@group(0) @binding(3) var<storage,read> headers:array<LeafHeader>;
@group(0) @binding(4) var<storage,read> metrics:array<Metric>;
@group(0) @binding(5) var<storage,read> pressure:array<f32>;
@group(0) @binding(6) var<storage,read_write> trace:array<u32>;
@group(0) @binding(7) var<uniform> fine:FineParams;
@group(0) @binding(8) var<storage,read> fineWorklist:array<u32>;
@group(0) @binding(9) var<storage,read> fineMetadata:array<u32>;
@group(0) @binding(10) var<storage,read> fineSamples:array<u32>;
@group(0) @binding(11) var<storage,read> fineSeeds:array<u32>;
@group(0) @binding(13) var<storage,read> coarsePhi:array<CoarsePhi>;
@group(0) @binding(14) var<storage,read_write> scratch:array<u32>;
${fineLevelSetPackedSampleWGSL("fineSamples")}

const INVALID:u32=0xffffffffu;
/** Published-row flag in \`Metric.transformAndFlags\`, as \`rowValid\` reads it. */
const VALID:u32=0x80000000u;
const MAGIC:u32=${FLUID_CELL_TRACE_MAGIC}u;
const HEADER_WORDS:u32=${FLUID_CELL_TRACE_HEADER_WORDS}u;
const RECORD_WORDS:u32=${FLUID_CELL_TRACE_RECORD_WORDS}u;
const PROBE_EDGE:u32=${FLUID_CELL_TRACE_FINE_PROBE_EDGE}u;
const RECORD_EDGE:u32=${FLUID_CELL_TRACE_FINE_RECORD_EDGE}u;
const FINE_RECORDS_OFFSET:u32=${FLUID_CELL_TRACE_FINE_RECORDS_OFFSET}u;
const FINE_RECORD_WORDS:u32=${FLUID_CELL_TRACE_FINE_RECORD_WORDS}u;
const FINE_RECORD_CAPACITY:u32=${FLUID_CELL_TRACE_FINE_RECORD_CAPACITY}u;
const HIT_CAPACITY:u32=${FLUID_CELL_TRACE_HIT_CAPACITY}u;
const HIT_WORDS:u32=${FLUID_CELL_TRACE_HIT_WORDS}u;
const HITS_OFFSET:u32=${FLUID_CELL_TRACE_HITS_OFFSET}u;
const FINE_VALID:u32=${FINE_FLOOD_SAMPLE_FLAGS.valid}u;
const FINE_INTERFACE:u32=${FINE_FLOOD_SAMPLE_FLAGS.interface}u;
const FINE_NEGATIVE:u32=${FINE_FLOOD_SAMPLE_FLAGS.negative}u;
const COARSE_READABLE:u32=${OCTREE_COARSE_PHI_FLAG.valid | OCTREE_COARSE_PHI_FLAG.finite}u;
/** Page states \`fineProbeAt\` distinguishes; "absent" and "stale" differ. */
const PAGE_MISSING:u32=0u;
const PAGE_STALE:u32=1u;
const PAGE_RESIDENT:u32=2u;
${makeFineLevelSetSortedWorklistLookupWGSL("fine", "fineMetadata", "fineWorklist", "finePageOf")}

/**
 * Order-preserving u32 key for a float, so phi extremes can be reduced with
 * \`atomicMin\`/\`atomicMax\`.
 *
 * WGSL has no atomic float. Flipping the sign bit for non-negatives and
 * inverting everything for negatives maps IEEE-754 onto unsigned integers with
 * the ordering intact, which is the standard trick and exact for every finite
 * value the band can hold.
 */
fn phiKey(value:f32)->u32 {
  let bits=bitcast<u32>(value);
  return select(~bits, bits|0x80000000u, value>=0.0);
}
fn phiFromKey(key:u32)->f32 {
  return bitcast<f32>(select(~key, key&0x7fffffffu, (key&0x80000000u)!=0u));
}

/**
 * World width of one finest cell.
 *
 * Phi is stored in metres on the fine lattice; every phi this trace publishes
 * is divided by this so the host reads distances in the same finest cells the
 * leaf sizes and origins are already in. Derived exactly as
 * \`decorationCellSize_m\` derives it, so the HUD's numbers and the overlay's
 * geometry cannot disagree.
 */
fn finestCellWidth()->f32 {
  let extent=max(vec3f(config.dimensions),vec3f(1.0));
  let size=u.container.xyz/extent;
  return max(min(size.x,min(size.y,size.z)),1e-9);
}

/** The row's coarse record, or a zeroed one when it was never published. */
fn coarseRecord(row:u32)->CoarsePhi {
  if(row>=arrayLength(&coarsePhi)){return CoarsePhi(0.0,0.0,0.0,0u);}
  let record=coarsePhi[row];
  if((record.flags&COARSE_READABLE)!=COARSE_READABLE){return CoarsePhi(0.0,0.0,0.0,0u);}
  return record;
}

fn direction18(index:u32)->vec3i {
  let d=array<vec3i,18>(
    vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),
    vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),
    vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));
  return d[index];
}

fn leafOriginOf(header:LeafHeader)->vec3u {
  let nx=max(config.dimensions.x,1u);
  let nxy=nx*max(config.dimensions.y,1u);
  return vec3u(header.cell%nx,(header.cell/nx)%max(config.dimensions.y,1u),header.cell/nxy);
}

/**
 * The row that genuinely owns a finest cell, or INVALID.
 *
 * The owner map is a zero-filled 3D texture and zero is a valid row index, so
 * every cell the publication never wrote reads back as row 0. Taking that at
 * face value is why the pick never moved: the march stopped on the first cell
 * the ray entered anywhere inside the container, which is empty space near the
 * near face, and reported the same degenerate row from every pointer position.
 * It is also why the stencil read "0 live of 18" — each air neighbour resolved
 * to row 0 as well, and a neighbour equal to the selected row is treated as
 * interior to the same leaf.
 *
 * The test is the one the technique views already apply in \`rowValid\`: a leaf
 * with extent, the published VALID flag, and a leaf box that actually contains
 * the cell. Containment is what rejects the zero fill, because row 0's leaf does
 * not span the domain.
 */
fn ownerAt(cell:vec3i)->u32 {
  if(any(cell<vec3i(0))||any(cell>=vec3i(config.dimensions))){return INVALID;}
  let row=textureLoad(ownerRows,cell,0).x;
  if(row>=config.rowCapacity||row>=arrayLength(&headers)||row>=arrayLength(&metrics)){return INVALID;}
  let header=headers[row];
  if(header.size==0u||(metrics[row].transformAndFlags&VALID)==0u){return INVALID;}
  let origin=vec3i(leafOriginOf(header));
  if(any(cell<origin)||any(cell>=origin+vec3i(i32(header.size)))){return INVALID;}
  return row;
}

/**
 * The aim this gather answers.
 *
 * The host's frozen ray wins whenever it has one, because that is the only
 * form of the request that survives a camera move. The pixel path stays as the
 * fallback for a frame that has not been given an aim yet, and builds the ray
 * in the same basis every technique view does, so the two agree.
 */
fn requestedRay()->CameraRay {
  if(config.hasAim!=0u){
    return CameraRay(config.aimOrigin,normalize(config.aimDirection));
  }
  let viewport=max(u.viewport.xy,vec2f(1.0));
  let uv=(vec2f(config.pixel)+vec2f(0.5))/viewport;
  // The shared helper takes uv with y up, which is the opposite of pixel order.
  return cameraRay(vec2f(uv.x,1.0-uv.y));
}

/**
 * Every distinct leaf the ray crosses, nearest first, and the one selected.
 *
 * Stepping at half a finest cell is the same law the volume overlays march
 * with, so what the trace names is what those views draw at that pixel. The run
 * is deduplicated by row rather than by cell: a 32³ leaf is one unknown however
 * many finest cells the ray spends inside it, and listing it once per cell would
 * bury the interior leaves the selection exists to reach.
 *
 * Records the run into the trace as it goes, and returns the cell of the
 * requested step. Selecting by pixel alone can only ever name the nearest leaf,
 * which on a liquid is a surface cell; the index is what reaches the interior.
 */
fn collectRayRun()->vec3i {
  let ray=requestedRay();
  let minimum=vec3f(-0.5*u.container.x,0.0,-0.5*u.container.z);
  let maximum=minimum+u.container.xyz;
  let interval=boxInterval(ray,minimum,maximum);
  if(interval.y<=interval.x){return vec3i(-1);}
  let extent=vec3f(max(config.dimensions,vec3u(1u)));
  let span=interval.y-interval.x;
  let cellWidth=min(u.container.x/extent.x,min(u.container.y/extent.y,u.container.z/extent.z));
  let steps=u32(clamp(ceil(span/max(cellWidth*0.5,1e-6)),1.0,1024.0));
  let dt=span/f32(steps);
  var found=0u;
  var overflow=0u;
  var previous=INVALID;
  var selected=vec3i(-1);
  for(var i=0u;i<1024u;i+=1u){
    if(i>=steps){break;}
    let distance=interval.x+(f32(i)+0.5)*dt;
    let point=ray.origin+ray.direction*distance;
    let coordinate=worldToFine(point);
    if(any(coordinate<vec3f(0.0))||any(coordinate>=extent)){continue;}
    let cell=vec3i(floor(coordinate));
    let row=ownerAt(cell);
    if(row==INVALID||row==previous){continue;}
    previous=row;
    if(found>=HIT_CAPACITY){overflow+=1u;continue;}
    let header=headers[row];
    let origin=leafOriginOf(header);
    let base=HITS_OFFSET+found*HIT_WORDS;
    trace[base+${FLUID_CELL_TRACE_HIT.row}u]=row;
    trace[base+${FLUID_CELL_TRACE_HIT.leafSize}u]=header.size;
    trace[base+${FLUID_CELL_TRACE_HIT.leafOrigin}u]=origin.x;
    trace[base+${FLUID_CELL_TRACE_HIT.leafOrigin}u+1u]=origin.y;
    trace[base+${FLUID_CELL_TRACE_HIT.leafOrigin}u+2u]=origin.z;
    trace[base+${FLUID_CELL_TRACE_HIT.distance_m}u]=bitcast<u32>(distance);
    // Coarse classification is attached by the following gather stage. Keeping
    // it out of this row-authority stage is what makes the four-buffer contract
    // structural rather than a device-limit special case.
    trace[base+${FLUID_CELL_TRACE_HIT.flags}u]=0u;
    // Clamped here rather than by the host, because only this pass knows how
    // long the run turned out to be. Holding every hit up to the requested one
    // leaves selected on the last that exists, so an index stranded past the
    // end of a shortened run lands on the deepest cell instead of missing.
    if(found<=min(config.hitIndex,HIT_CAPACITY-1u)){selected=cell;}
    found+=1u;
  }
  let index=select(0u,min(config.hitIndex,found-1u),found>0u);
  trace[${FLUID_CELL_TRACE_HEADER.hitCount}u]=found;
  trace[${FLUID_CELL_TRACE_HEADER.hitIndex}u]=index;
  trace[${FLUID_CELL_TRACE_HEADER.hitOverflow}u]=overflow;
  return selected;
}

fn storeFloat(index:u32,value:f32){trace[index]=bitcast<u32>(value);}

var<workgroup> probeSamples:atomic<u32>;
var<workgroup> probeResolved:atomic<u32>;
var<workgroup> probeInterface:atomic<u32>;
var<workgroup> probeMaximumHop:atomic<u32>;
var<workgroup> probeMissing:atomic<u32>;
var<workgroup> probeStale:atomic<u32>;
var<workgroup> probeNegative:atomic<u32>;
var<workgroup> probeMinimumPhi:atomic<u32>;
var<workgroup> probeMaximumPhi:atomic<u32>;
var<workgroup> probeNearestPhi:atomic<u32>;
var<workgroup> probeRecords:atomic<u32>;

/**
 * Where a fine-lattice coordinate lives, and why it might not.
 *
 * The previous version returned only INVALID, which made "this leaf sits
 * outside the allocated band" indistinguishable from "this leaf's pages are a
 * generation behind". Those are different faults with different fixes, and the
 * gaps decoration draws them differently, so the state travels with the
 * address.
 */
struct FineProbe { address:u32, state:u32 }

fn fineProbeAt(q:vec3u)->FineProbe {
  if(config.hasFine==0u||any(q>=fine.sampleDimensions)){return FineProbe(INVALID,PAGE_MISSING);}
  let brick=q/max(fine.brickResolution,1u);
  let key=brick.x+fine.brickDimensions.x*(brick.y+fine.brickDimensions.y*brick.z);
  let page=finePageOf(key);
  if(page==INVALID||page>=fine.pageCapacity||page*4u+2u>=arrayLength(&fineMetadata)){
    return FineProbe(INVALID,PAGE_MISSING);
  }
  if(fineMetadata[page*4u+2u]!=fine.generation){return FineProbe(INVALID,PAGE_STALE);}
  let local=q-brick*fine.brickResolution;
  let localIndex=local.x+fine.brickResolution*(local.y+fine.brickResolution*local.z);
  let address=page*fine.samplesPerBrick+localIndex;
  // Payload buffers are capacity-stable members of this publication. Their
  // bounds are validated in the value/provenance stages that actually bind
  // them; address discovery deliberately owns only the directory pair.
  if(address>=fine.pageCapacity*fine.samplesPerBrick){return FineProbe(INVALID,PAGE_STALE);}
  return FineProbe(address,PAGE_RESIDENT);
}

/** Fine-lattice cell of a sample address, matching the redistance addressing. */
fn fineSampleCell(address:u32)->vec3u {
  let perBrick=max(fine.samplesPerBrick,1u);
  let id=address/perBrick;
  let local=address-id*perBrick;
  let key=fineMetadata[id*4u+1u];
  let xy=max(fine.brickDimensions.x*fine.brickDimensions.y,1u);
  let bz=key/xy;let rest=key-bz*xy;let by=rest/max(fine.brickDimensions.x,1u);
  let brick=vec3u(rest-by*fine.brickDimensions.x,by,bz);
  let r=max(fine.brickResolution,1u);
  let lz=local/(r*r);let lrest=local-lz*r*r;let ly=lrest/r;
  return brick*r+vec3u(lrest-ly*r,ly,lz);
}

/** Row authority only: four storage bindings, including the trace target. */
@compute @workgroup_size(1)
fn gatherCore() {
  for(var word=0u;word<HEADER_WORDS;word+=1u){trace[word]=0u;}
  for(var slot=0u;slot<FINE_RECORD_CAPACITY;slot+=1u){
    trace[FINE_RECORDS_OFFSET+slot*FINE_RECORD_WORDS+${FLUID_CELL_TRACE_FINE_RECORD.flags}u]=0u;
  }
  trace[${FLUID_CELL_TRACE_HEADER.magic}u]=MAGIC;
  trace[${FLUID_CELL_TRACE_HEADER.pixelX}u]=config.pixel.x;
  trace[${FLUID_CELL_TRACE_HEADER.pixelY}u]=config.pixel.y;
  trace[${FLUID_CELL_TRACE_HEADER.requestToken}u]=config.requestToken;
  trace[${FLUID_CELL_TRACE_HEADER.status}u]=${FLUID_CELL_TRACE_STATUS.miss}u;
  trace[${FLUID_CELL_TRACE_HEADER.dimensions}u]=config.dimensions.x;
  trace[${FLUID_CELL_TRACE_HEADER.dimensions}u+1u]=config.dimensions.y;
  trace[${FLUID_CELL_TRACE_HEADER.dimensions}u+2u]=config.dimensions.z;

  let cell=collectRayRun();
  if(any(cell<vec3i(0))){return;}
  let row=ownerAt(cell);
  if(row==INVALID){return;}
  let header=headers[row];
  if(header.size==0u){
    trace[${FLUID_CELL_TRACE_HEADER.status}u]=${FLUID_CELL_TRACE_STATUS.invalid}u;
    return;
  }
  let origin=leafOriginOf(header);
  trace[${FLUID_CELL_TRACE_HEADER.status}u]=${FLUID_CELL_TRACE_STATUS.resolved}u;
  trace[${FLUID_CELL_TRACE_HEADER.cell}u]=u32(cell.x);
  trace[${FLUID_CELL_TRACE_HEADER.cell}u+1u]=u32(cell.y);
  trace[${FLUID_CELL_TRACE_HEADER.cell}u+2u]=u32(cell.z);
  trace[${FLUID_CELL_TRACE_HEADER.row}u]=row;
  trace[${FLUID_CELL_TRACE_HEADER.leafSize}u]=header.size;
  trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u]=origin.x;
  trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u+1u]=origin.y;
  trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u+2u]=origin.z;
  storeFloat(${FLUID_CELL_TRACE_HEADER.diagonal}u,header.diagonal);
  storeFloat(${FLUID_CELL_TRACE_HEADER.rhs}u,header.rhs);
  trace[${FLUID_CELL_TRACE_HEADER.entryCount}u]=header.entryCount;
  if(row<arrayLength(&metrics)){
    storeFloat(${FLUID_CELL_TRACE_HEADER.volume}u,metrics[row].volume);
    trace[${FLUID_CELL_TRACE_HEADER.topologyCode}u]=metrics[row].topologyCode;
  }
  if(row<arrayLength(&pressure)){
    storeFloat(${FLUID_CELL_TRACE_HEADER.pressure}u,pressure[row]);
  }

  let centre=vec3f(origin)+vec3f(f32(header.size)*0.5);
  let reach=f32(header.size)*0.5+0.5;
  for(var index=0u;index<${FLUID_CELL_TRACE_NEIGHBOR_CAPACITY}u;index+=1u){
    let probe=vec3i(floor(centre+vec3f(direction18(index))*reach));
    let base=HEADER_WORDS+index*RECORD_WORDS;
    var flags=0u;var neighbourRow=INVALID;var neighbourSize=0u;
    var neighbourOrigin=vec3u(0u);var neighbourPressure=0.0;
    if(any(probe<vec3i(0))||any(probe>=vec3i(config.dimensions))){
      flags=${FLUID_CELL_TRACE_RECORD_FLAGS.boundary}u;
    } else {
      neighbourRow=ownerAt(probe);
      if(neighbourRow!=INVALID&&neighbourRow!=row){
        let other=headers[neighbourRow];
        neighbourSize=other.size;neighbourOrigin=leafOriginOf(other);
        flags=${FLUID_CELL_TRACE_RECORD_FLAGS.present}u;
        if(other.size>header.size){flags|=${FLUID_CELL_TRACE_RECORD_FLAGS.coarser}u;}
        if(other.size<header.size){flags|=${FLUID_CELL_TRACE_RECORD_FLAGS.finer}u;}
        if(neighbourRow<arrayLength(&pressure)){neighbourPressure=pressure[neighbourRow];}
      } else { neighbourRow=INVALID; }
    }
    trace[base+${FLUID_CELL_TRACE_RECORD.direction}u]=index;
    trace[base+${FLUID_CELL_TRACE_RECORD.row}u]=neighbourRow;
    trace[base+${FLUID_CELL_TRACE_RECORD.leafSize}u]=neighbourSize;
    trace[base+${FLUID_CELL_TRACE_RECORD.flags}u]=flags;
    trace[base+${FLUID_CELL_TRACE_RECORD.leafOrigin}u]=neighbourOrigin.x;
    trace[base+${FLUID_CELL_TRACE_RECORD.leafOrigin}u+1u]=neighbourOrigin.y;
    trace[base+${FLUID_CELL_TRACE_RECORD.leafOrigin}u+2u]=neighbourOrigin.z;
    storeFloat(base+${FLUID_CELL_TRACE_RECORD.pressure}u,neighbourPressure);
    storeFloat(base+${FLUID_CELL_TRACE_RECORD.phi}u,0.0);
    trace[base+${FLUID_CELL_TRACE_RECORD.phiFlags}u]=0u;
  }
  trace[${FLUID_CELL_TRACE_HEADER.neighborCount}u]=${FLUID_CELL_TRACE_NEIGHBOR_CAPACITY}u;
}

/** Coarse correction annotates row ids already selected by gatherCore. */
@compute @workgroup_size(1)
fn gatherCoarse() {
  if(trace[${FLUID_CELL_TRACE_HEADER.status}u]!=${FLUID_CELL_TRACE_STATUS.resolved}u){return;}
  let inverseWidth=1.0/finestCellWidth();
  let row=trace[${FLUID_CELL_TRACE_HEADER.row}u];
  let selected=coarseRecord(row);
  storeFloat(${FLUID_CELL_TRACE_HEADER.coarsePhi}u,selected.phi*inverseWidth);
  storeFloat(${FLUID_CELL_TRACE_HEADER.coarsePhiMinimum}u,selected.minimumPhi*inverseWidth);
  storeFloat(${FLUID_CELL_TRACE_HEADER.coarsePhiMaximum}u,selected.maximumPhi*inverseWidth);
  trace[${FLUID_CELL_TRACE_HEADER.coarsePhiFlags}u]=selected.flags;
  let hitCount=min(trace[${FLUID_CELL_TRACE_HEADER.hitCount}u],HIT_CAPACITY);
  for(var index=0u;index<hitCount;index+=1u){
    let base=HITS_OFFSET+index*HIT_WORDS;
    let record=coarseRecord(trace[base+${FLUID_CELL_TRACE_HIT.row}u]);
    var flags=0u;
    if(record.flags!=0u){
      flags|=${FLUID_CELL_TRACE_HIT_FLAGS.corrected}u;
      if(record.phi<0.0){flags|=${FLUID_CELL_TRACE_HIT_FLAGS.liquid}u;}
      if(record.minimumPhi<=0.0&&record.maximumPhi>=0.0){flags|=${FLUID_CELL_TRACE_HIT_FLAGS.interface}u;}
    }
    trace[base+${FLUID_CELL_TRACE_HIT.flags}u]=flags;
  }
  let neighbourCount=min(trace[${FLUID_CELL_TRACE_HEADER.neighborCount}u],${FLUID_CELL_TRACE_NEIGHBOR_CAPACITY}u);
  for(var index=0u;index<neighbourCount;index+=1u){
    let base=HEADER_WORDS+index*RECORD_WORDS;
    let neighbourRow=trace[base+${FLUID_CELL_TRACE_RECORD.row}u];
    if(neighbourRow==INVALID){continue;}
    let record=coarseRecord(neighbourRow);
    storeFloat(base+${FLUID_CELL_TRACE_RECORD.phi}u,record.phi*inverseWidth);
    trace[base+${FLUID_CELL_TRACE_RECORD.phiFlags}u]=record.flags;
  }
}

/** Directory lookup writes one compact three-word scratch record per probe. */
@compute @workgroup_size(${LANES})
fn discoverFine(@builtin(local_invocation_index) lid:u32) {
  if(lid==0u){
    atomicStore(&probeMissing,0u);atomicStore(&probeStale,0u);atomicStore(&probeRecords,0u);
    trace[${FLUID_CELL_TRACE_HEADER.fineFactor}u]=select(0u,max(fine.fineFactor,1u),config.hasFine!=0u);
  }
  workgroupBarrier();
  let total=PROBE_EDGE*PROBE_EDGE*PROBE_EDGE;
  if(config.hasFine!=0u&&trace[${FLUID_CELL_TRACE_HEADER.status}u]==${FLUID_CELL_TRACE_STATUS.resolved}u){
    let factor=max(fine.fineFactor,1u);
    let fineOrigin=vec3u(trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u],
      trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u+1u],trace[${FLUID_CELL_TRACE_HEADER.leafOrigin}u+2u])*factor;
    let fineExtent=trace[${FLUID_CELL_TRACE_HEADER.leafSize}u]*factor;
    for(var probe=lid;probe<total;probe+=${LANES}u){
      let pz=probe/(PROBE_EDGE*PROBE_EDGE);let rest=probe-pz*PROBE_EDGE*PROBE_EDGE;
      let py=rest/PROBE_EDGE;let px=rest-py*PROBE_EDGE;
      let q=fineOrigin+(vec3u(px,py,pz)*2u+vec3u(1u))*fineExtent/(2u*PROBE_EDGE);
      let found=fineProbeAt(q);let scratchBase=probe*3u;
      scratch[scratchBase]=found.address;scratch[scratchBase+1u]=found.state;scratch[scratchBase+2u]=INVALID;
      if(found.state==PAGE_MISSING){atomicAdd(&probeMissing,1u);}
      if(found.state==PAGE_STALE){atomicAdd(&probeStale,1u);}
      if(((px|py|pz)&1u)==0u){
        let slot=(px>>1u)+RECORD_EDGE*((py>>1u)+RECORD_EDGE*(pz>>1u));
        let base=FINE_RECORDS_OFFSET+slot*FINE_RECORD_WORDS;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.cell}u]=q.x;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.cell}u+1u]=q.y;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.cell}u+2u]=q.z;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.flags}u]=${FLUID_CELL_TRACE_FINE_FLAGS.probed}u
          |select(0u,${FLUID_CELL_TRACE_FINE_FLAGS.stale}u,found.state==PAGE_STALE)
          |select(0u,${FLUID_CELL_TRACE_FINE_FLAGS.resident}u,found.state==PAGE_RESIDENT);
        atomicMax(&probeRecords,slot+1u);
      }
    }
  }
  workgroupBarrier();
  if(lid==0u){
    trace[${FLUID_CELL_TRACE_HEADER.fineMissing}u]=atomicLoad(&probeMissing);
    trace[${FLUID_CELL_TRACE_HEADER.fineStale}u]=atomicLoad(&probeStale);
    trace[${FLUID_CELL_TRACE_HEADER.fineRecordCount}u]=atomicLoad(&probeRecords);
    trace[${FLUID_CELL_TRACE_HEADER.fineProbes}u]=select(0u,total,config.hasFine!=0u);
  }
}

/** Fine scalar authority: flags and phi only. */
@compute @workgroup_size(${LANES})
fn gatherFineValues(@builtin(local_invocation_index) lid:u32) {
  if(lid==0u){
    atomicStore(&probeSamples,0u);atomicStore(&probeInterface,0u);atomicStore(&probeNegative,0u);
    atomicStore(&probeMinimumPhi,0xffffffffu);atomicStore(&probeMaximumPhi,0u);
    atomicStore(&probeNearestPhi,0x7f800000u);
  }
  workgroupBarrier();
  let inverseWidth=1.0/finestCellWidth();let total=PROBE_EDGE*PROBE_EDGE*PROBE_EDGE;
  if(config.hasFine!=0u&&trace[${FLUID_CELL_TRACE_HEADER.status}u]==${FLUID_CELL_TRACE_STATUS.resolved}u){
    for(var probe=lid;probe<total;probe+=${LANES}u){
      let scratchBase=probe*3u;if(scratch[scratchBase+1u]!=PAGE_RESIDENT){continue;}
      let address=scratch[scratchBase];if(address>=arrayLength(&fineSamples)){continue;}
      let flags=finePackedFlags(address);if((flags&FINE_VALID)==0u){continue;}
      let phi=finePackedPhi(address)*inverseWidth;
      atomicAdd(&probeSamples,1u);atomicMin(&probeMinimumPhi,phiKey(phi));
      atomicMax(&probeMaximumPhi,phiKey(phi));atomicMin(&probeNearestPhi,bitcast<u32>(abs(phi)));
      if((flags&FINE_INTERFACE)!=0u){atomicAdd(&probeInterface,1u);}
      if((flags&FINE_NEGATIVE)!=0u){atomicAdd(&probeNegative,1u);}
      let pz=probe/(PROBE_EDGE*PROBE_EDGE);let rest=probe-pz*PROBE_EDGE*PROBE_EDGE;
      let py=rest/PROBE_EDGE;let px=rest-py*PROBE_EDGE;
      if(((px|py|pz)&1u)==0u){
        let slot=(px>>1u)+RECORD_EDGE*((py>>1u)+RECORD_EDGE*(pz>>1u));
        let base=FINE_RECORDS_OFFSET+slot*FINE_RECORD_WORDS;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.flags}u]|=${FLUID_CELL_TRACE_FINE_FLAGS.valid}u
          |select(0u,${FLUID_CELL_TRACE_FINE_FLAGS.interface}u,(flags&FINE_INTERFACE)!=0u)
          |select(0u,${FLUID_CELL_TRACE_FINE_FLAGS.negative}u,(flags&FINE_NEGATIVE)!=0u);
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.phi}u]=bitcast<u32>(phi);
      }
    }
  }
  workgroupBarrier();
  if(lid==0u){
    let samples=atomicLoad(&probeSamples);
    trace[${FLUID_CELL_TRACE_HEADER.fineSamples}u]=samples;
    trace[${FLUID_CELL_TRACE_HEADER.fineInterface}u]=atomicLoad(&probeInterface);
    trace[${FLUID_CELL_TRACE_HEADER.fineNegative}u]=atomicLoad(&probeNegative);
    trace[${FLUID_CELL_TRACE_HEADER.probeMinimumPhi}u]=select(0u,bitcast<u32>(phiFromKey(atomicLoad(&probeMinimumPhi))),samples>0u);
    trace[${FLUID_CELL_TRACE_HEADER.probeMaximumPhi}u]=select(0u,bitcast<u32>(phiFromKey(atomicLoad(&probeMaximumPhi))),samples>0u);
    trace[${FLUID_CELL_TRACE_HEADER.probeNearestPhi}u]=select(0u,atomicLoad(&probeNearestPhi),samples>0u);
  }
}

/** Seed authority is fetched separately so scalar gather stays at four. */
@compute @workgroup_size(${LANES})
fn gatherFineSeeds(@builtin(local_invocation_index) lid:u32) {
  let total=PROBE_EDGE*PROBE_EDGE*PROBE_EDGE;
  for(var probe=lid;probe<total;probe+=${LANES}u){
    let base=probe*3u;let address=scratch[base];
    if(scratch[base+1u]==PAGE_RESIDENT&&address<arrayLength(&fineSeeds)){
      scratch[base+2u]=fineSeeds[address];
    }
  }
}

/** Resolve fetched seed addresses against directory + seed flags. */
@compute @workgroup_size(${LANES})
fn resolveFineSeeds(@builtin(local_invocation_index) lid:u32) {
  if(lid==0u){atomicStore(&probeResolved,0u);atomicStore(&probeMaximumHop,0u);}
  workgroupBarrier();
  let total=PROBE_EDGE*PROBE_EDGE*PROBE_EDGE;
  if(config.hasFine!=0u&&trace[${FLUID_CELL_TRACE_HEADER.status}u]==${FLUID_CELL_TRACE_STATUS.resolved}u){
    for(var probe=lid;probe<total;probe+=${LANES}u){
      let scratchBase=probe*3u;let address=scratch[scratchBase];let seed=scratch[scratchBase+2u];
      if(scratch[scratchBase+1u]!=PAGE_RESIDENT||address>=arrayLength(&fineSamples)
        ||(finePackedFlags(address)&FINE_VALID)==0u||seed==INVALID||seed>=arrayLength(&fineSamples)){continue;}
      let seedPage=seed/max(fine.samplesPerBrick,1u);
      if(seedPage*4u+1u>=arrayLength(&fineMetadata)){continue;}
      let seedCell=fineSampleCell(seed);let sampleCell=fineSampleCell(address);
      let delta=abs(vec3i(seedCell)-vec3i(sampleCell));let hop=u32(max(max(delta.x,delta.y),delta.z));
      atomicAdd(&probeResolved,1u);atomicMax(&probeMaximumHop,hop);
      let pz=probe/(PROBE_EDGE*PROBE_EDGE);let rest=probe-pz*PROBE_EDGE*PROBE_EDGE;
      let py=rest/PROBE_EDGE;let px=rest-py*PROBE_EDGE;
      if(((px|py|pz)&1u)==0u){
        let slot=(px>>1u)+RECORD_EDGE*((py>>1u)+RECORD_EDGE*(pz>>1u));
        let base=FINE_RECORDS_OFFSET+slot*FINE_RECORD_WORDS;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.flags}u]|=${FLUID_CELL_TRACE_FINE_FLAGS.resolved}u;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.seedCell}u]=seedCell.x;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.seedCell}u+1u]=seedCell.y;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.seedCell}u+2u]=seedCell.z;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.seedCode}u]=finePackedFlags(seed)>>${FINE_FLOOD_SAMPLE_FLAG_BITS}u;
        trace[base+${FLUID_CELL_TRACE_FINE_RECORD.hop}u]=hop;
      }
    }
  }
  workgroupBarrier();
  if(lid==0u){
    trace[${FLUID_CELL_TRACE_HEADER.fineResolved}u]=atomicLoad(&probeResolved);
    trace[${FLUID_CELL_TRACE_HEADER.fineMaximumHop}u]=atomicLoad(&probeMaximumHop);
  }
}
`;

const FLUID_CELL_TRACE_STAGE_ENTRY_POINTS = [
  "gatherCore", "gatherCoarse", "discoverFine", "gatherFineValues",
  "gatherFineSeeds", "resolveFineSeeds",
] as const;

export class WebGPUFluidCellTrace {
  private pipelines?: readonly GPUComputePipeline[];
  private readonly config: GPUBuffer;
  private readonly output: GPUBuffer;
  /**
   * Two staging slots, exactly as the ray probe uses.
   *
   * A buffer with a map pending may not appear in a submit, and `mapAsync` takes
   * at least a frame to resolve. Copying into a single readback buffer every
   * frame therefore invalidates the *whole* command buffer rather than just this
   * pass — the scene stops presenting and the viewport goes black. Alternating
   * slots means the copy always lands in one that is neither mapped nor pending,
   * and a frame with both in flight skips the copy instead of poisoning itself.
   */
  private readonly staging: readonly { readonly buffer: GPUBuffer; pending: boolean }[];
  private pendingSlot?: { readonly buffer: GPUBuffer; pending: boolean };
  private readonly fallback: GPUBuffer;
  private readonly fineFallback: GPUBuffer;
  private readonly scratch: GPUBuffer;
  private groups?: readonly GPUBindGroup[];
  private source?: OctreeTechniqueDebugSource;
  private ownerRows?: GPUTexture;
  private pixel: readonly [number, number] = [0, 0];
  /** The frozen aim, if the host has given one. See `setAim`. */
  private aim?: { readonly origin: Vec3; readonly direction: Vec3 };
  private token = 0;
  private hitIndex = 0;
  private destroyed = false;
  /** One staging block for the config, so the floats and the words share it. */
  private readonly configWords = new ArrayBuffer(CONFIG_BYTES);

  constructor(private readonly device: GPUDevice, private readonly uniformBuffer: GPUBuffer) {
    this.config = device.createBuffer({
      label: "Fluid cell trace config", size: CONFIG_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.output = device.createBuffer({
      label: "Fluid cell trace records", size: FLUID_CELL_TRACE_WORDS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    this.staging = Array.from({ length: 2 }, (_, index) => ({
      buffer: device.createBuffer({
        label: `Fluid cell trace readback ${index + 1}/2`, size: FLUID_CELL_TRACE_WORDS * 4,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      }),
      pending: false,
    }));
    // Bound when a scene carries no fine band, so the shader keeps one layout
    // and reports zero samples instead of the pipeline failing to build.
    this.fallback = device.createBuffer({
      label: "Fluid cell trace absent storage", size: 4, usage: GPUBufferUsage.STORAGE,
    });
    this.fineFallback = device.createBuffer({
      label: "Fluid cell trace absent fine params", size: 80, usage: GPUBufferUsage.UNIFORM,
    });
    this.scratch = device.createBuffer({
      label: "Fluid cell trace fine address scratch",
      size: FLUID_CELL_TRACE_FINE_PROBES * 3 * 4,
      usage: GPUBufferUsage.STORAGE,
    });
  }

  async initialize(): Promise<void> {
    // Four is an architectural ceiling, not a best effort against the device's
    // advertised limit. A lower-limit device gets the same clear failure.
    const available = this.device.limits.maxStorageBuffersPerShaderStage;
    if (available < FLUID_CELL_TRACE_STORAGE_BINDINGS.length) {
      throw new Error(
        `Fluid cell trace stages use at most ${FLUID_CELL_TRACE_STORAGE_BINDINGS.length} storage buffers `
        + `and this device allows ${available}`,
      );
    }
    const module = this.device.createShaderModule({
      label: "Fluid cell trace gather", code: fluidCellTraceGatherShader,
    });
    const compilation = await module.getCompilationInfo();
    const errors = compilation.messages.filter((message) => message.type === "error");
    if (errors.length) {
      throw new Error(errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("\n"));
    }
    this.pipelines = await Promise.all(FLUID_CELL_TRACE_STAGE_ENTRY_POINTS.map((entryPoint) =>
      this.device.createComputePipelineAsync({
        label: `Fluid cell trace ${entryPoint}`, layout: "auto",
        compute: { module, entryPoint },
      })));
    this.rebuild();
  }

  setSource(source: OctreeTechniqueDebugSource | undefined, ownerRows: GPUTexture | undefined): void {
    if (this.source === source && this.ownerRows === ownerRows) return;
    this.source = source;
    this.ownerRows = ownerRows;
    this.rebuild();
  }

  /** The pixel whose cell the next encode gathers. */
  requestPixel(x: number, y: number): void {
    const next: readonly [number, number] = [Math.max(0, Math.round(x)), Math.max(0, Math.round(y))];
    if (next[0] === this.pixel[0] && next[1] === this.pixel[1]) return;
    this.pixel = next;
    this.token += 1;
  }

  /**
   * The world-space ray the next gather marches.
   *
   * Set every frame the selection is live and left alone once it is pinned,
   * which is the whole mechanism: the pin freezes the *aim*, not the pixel.
   * Freezing a pixel froze nothing — the ray behind it swings with the camera,
   * so orbiting a pinned cell quietly reselected whatever had moved under that
   * pixel. Does not bump the request token: the same aim from a moved camera is
   * the same request.
   */
  setAim(origin: Vec3, direction: Vec3): void {
    const current = this.aim;
    if (current
      && current.origin.x === origin.x && current.origin.y === origin.y
      && current.origin.z === origin.z && current.direction.x === direction.x
      && current.direction.y === direction.y && current.direction.z === direction.z) return;
    this.aim = { origin, direction };
  }

  /** Fall back to the requested pixel's own ray. */
  clearAim(): void { this.aim = undefined; }

  /**
   * Step along the ray run the next gather describes.
   *
   * Clamped in the shader rather than here: the run's length is only known once
   * the march has walked it, and the host would otherwise have to keep a mirror
   * of the topology to know how far it may step.
   */
  setHitIndex(index: number): void {
    const next = Math.max(0, Math.min(FLUID_CELL_TRACE_HIT_CAPACITY - 1, Math.floor(index)));
    if (next === this.hitIndex) return;
    this.hitIndex = next;
    this.token += 1;
  }

  get requestedHitIndex(): number { return this.hitIndex; }
  get requestedPixel(): readonly [number, number] { return this.pixel; }
  get ready(): boolean { return this.pipelines !== undefined && this.groups !== undefined; }

  private rebuild(): void {
    this.groups = undefined;
    const source = this.source;
    const ownerRows = this.ownerRows;
    const pipelines = this.pipelines;
    if (!pipelines || !source || !ownerRows) return;
    const fine = source.fineBandLifecycle;
    const create = (index: number, entries: readonly GPUBindGroupEntry[]) => this.device.createBindGroup({
      label: `Fluid cell trace ${FLUID_CELL_TRACE_STAGE_ENTRY_POINTS[index]} bindings`,
      layout: pipelines[index].getBindGroupLayout(0), entries,
    });
    const trace = { buffer: this.output };
    const scratch = { buffer: this.scratch };
    const fineParams = fine ? fine.params : { buffer: this.fineFallback };
    this.groups = [
      create(0, [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 1, resource: ownerRows.createView({ dimension: "3d" }) },
        { binding: 2, resource: { buffer: this.config } },
        { binding: 3, resource: source.leafHeaders },
        { binding: 4, resource: source.topologyMetrics },
        { binding: 5, resource: source.pressure },
        { binding: 6, resource: trace },
      ]),
      create(1, [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 2, resource: { buffer: this.config } },
        { binding: 6, resource: trace },
        { binding: 13, resource: source.coarsePhi ?? { buffer: this.fallback } },
      ]),
      create(2, [
        { binding: 2, resource: { buffer: this.config } },
        { binding: 6, resource: trace },
        { binding: 7, resource: fineParams },
        { binding: 8, resource: fine ? fine.worklist : { buffer: this.fallback } },
        { binding: 9, resource: fine ? fine.metadata : { buffer: this.fallback } },
        { binding: 14, resource: scratch },
      ]),
      create(3, [
        { binding: 0, resource: { buffer: this.uniformBuffer } },
        { binding: 2, resource: { buffer: this.config } },
        { binding: 6, resource: trace },
        { binding: 10, resource: fine ? fine.samples : { buffer: this.fallback } },
        { binding: 14, resource: scratch },
      ]),
      create(4, [
        { binding: 11, resource: fine ? fine.seeds : { buffer: this.fallback } },
        { binding: 14, resource: scratch },
      ]),
      create(5, [
        { binding: 2, resource: { buffer: this.config } },
        { binding: 6, resource: trace },
        { binding: 7, resource: fineParams },
        { binding: 9, resource: fine ? fine.metadata : { buffer: this.fallback } },
        { binding: 10, resource: fine ? fine.samples : { buffer: this.fallback } },
        { binding: 14, resource: scratch },
      ]),
    ];
  }

  encode(encoder: GPUCommandEncoder): void {
    if (this.destroyed || !this.pipelines || !this.groups || !this.source) return;
    const { dimensions, rowCapacity } = this.source.pressureRows;
    const words = new Uint32Array(this.configWords);
    const floats = new Float32Array(this.configWords);
    words.set([
      dimensions[0], dimensions[1], dimensions[2], rowCapacity,
      this.pixel[0], this.pixel[1], this.token, this.source.fineBandLifecycle ? 1 : 0,
      this.hitIndex, this.aim ? 1 : 0, 0, 0,
    ]);
    const aim = this.aim;
    floats.set(aim
      ? [aim.origin.x, aim.origin.y, aim.origin.z, 0,
        aim.direction.x, aim.direction.y, aim.direction.z, 0]
      : [0, 0, 0, 0, 0, 0, 0, 0], 12);
    this.device.queue.writeBuffer(this.config, 0, this.configWords);
    const broker = new PassBroker(encoder);
    for (let index = 0; index < this.pipelines.length; index += 1) {
      const entryPoint = FLUID_CELL_TRACE_STAGE_ENTRY_POINTS[index];
      const pass = broker.compute({ label: `Gather fluid cell trace: ${entryPoint}` });
      pass.setPipeline(this.pipelines[index]);
      pass.setBindGroup(0, this.groups[index]);
      pass.dispatchWorkgroups(1);
      broker.fence(`fluid cell trace ${entryPoint} gathered`);
    }
    // The gather itself is cheap and idempotent, so a frame with both slots in
    // flight still refreshes `output` and simply forgoes this frame's copy.
    const slot = this.staging.find((candidate) => !candidate.pending);
    if (!slot) return;
    slot.pending = true;
    this.pendingSlot = slot;
    encoder.copyBufferToBuffer(this.output, 0, slot.buffer, 0, slot.buffer.size);
  }

  /** Resolve the most recently encoded readback, or nothing if none is waiting. */
  async read(): Promise<FluidCellTrace | undefined> {
    const slot = this.pendingSlot;
    this.pendingSlot = undefined;
    if (!slot || this.destroyed) return undefined;
    try {
      await slot.buffer.mapAsync(GPUMapMode.READ);
      if (this.destroyed) return undefined;
      // Copied off the mapped range: `finally` unmaps it, and a decode that kept
      // a view would then be reading a detached buffer.
      return decodeFluidCellTrace(Uint32Array.from(new Uint32Array(slot.buffer.getMappedRange())));
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
    this.config.destroy();
    this.output.destroy();
    for (const slot of this.staging) { try { slot.buffer.destroy(); } catch { /* Device loss. */ } }
    this.fallback.destroy();
    this.fineFallback.destroy();
    this.scratch.destroy();
  }
}
