import {
  createSparseCM12TransportHomeFrameHaloLayout,
} from "./sparse-cm12-transport-home-frame-halo";

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

export interface SparseCM12TransportHomeFrameHaloWGSLOptions {
  readonly maximumRadius?: 1 | 2 | 3;
  readonly profileBaseWords?: number;
  /** Callback returning CM12TransportOwner { cell, widths, volume }. */
  readonly ownerAtFineName?: string;
  /** Callback performing one native vec4f load from the effective plane. */
  readonly effectiveVelocityName?: string;
  /** Existing transport stencil type with cells[8] and weights[8]. */
  readonly stencilTypeName?: string;
  readonly invalidExpression?: string;
  readonly dimensionsExpression?: string;
  readonly frameDtExpression?: string;
  readonly clampInsideBoundaryName?: string;
  readonly clipBoundarySegmentName?: string;
}

/**
 * Binding-free home-frame transport sampler.
 *
 * The caller invokes cm12StageTransportHomeFrameHalo exactly once at the start
 * of a 64-lane rung-major packet workgroup. `packetOriginFine` is the minimum
 * finest-lattice coordinate of its 4^3 home-cell packet; `homeShift` is
 * log2(home cell span); and `radiusEstimate` is ceil(max(|v|) dt) + 1. A radius
 * above three is safe: the cache remains clamped and every miss uses the exact
 * global resolver. All hot-path fine-to-home conversion is shifts/masks; no
 * integer division is emitted.
 */
export function createSparseCM12TransportHomeFrameHaloWGSL(
  options: SparseCM12TransportHomeFrameHaloWGSLOptions = {},
): string {
  const layout = createSparseCM12TransportHomeFrameHaloLayout(options.maximumRadius);
  const ownerAtFine = identifier(options.ownerAtFineName ?? "cm12TeiOwnerAtFine",
    "ownerAtFineName");
  const velocity = identifier(options.effectiveVelocityName
    ?? "cm12EffectiveTransportVelocity", "effectiveVelocityName");
  const stencil = identifier(options.stencilTypeName ?? "TransportStencil",
    "stencilTypeName");
  const clampInside = identifier(options.clampInsideBoundaryName
    ?? "clampInsideEmbeddedBoundary", "clampInsideBoundaryName");
  const clipSegment = identifier(options.clipBoundarySegmentName
    ?? "clipBoundarySegment", "clipBoundarySegmentName");
  const invalid = options.invalidExpression ?? "INVALID";
  const dimensions = options.dimensionsExpression ?? "p.dimensions.xyz";
  const frameDt = options.frameDtExpression ?? "p.frame.x";
  const profile = options.profileBaseWords === undefined ? undefined
    : options.profileBaseWords;
  const profileAdd = (offset: number, expression = "1u") => profile === undefined
    ? "" : `atomicAdd(&activity[${profile + offset}u],${expression});`;

  return /* wgsl */ `
const CM12_HOME_HALO_PACKET_EDGE:u32=${layout.packetEdge}u;
const CM12_HOME_HALO_MAX_RADIUS:u32=${layout.maximumRadius}u;
const CM12_HOME_HALO_MAX_EDGE:u32=${layout.maximumEdge}u;
const CM12_HOME_HALO_CAPACITY:u32=${layout.capacity}u;

// 1000 * (u32 + vec4f) + metadata = 20064 bytes, below the portable 32 KiB
// compute workgroup-storage limit. Separate arrays preserve native vec4 loads.
var<workgroup> cm12HomeHaloCells:array<u32,${layout.capacity}>;
var<workgroup> cm12HomeHaloVelocities:array<vec4f,${layout.capacity}>;
var<workgroup> cm12HomeHaloPacketOriginFine:vec3i;
var<workgroup> cm12HomeHaloCacheFirstFine:vec3i;
var<workgroup> cm12HomeHaloHomeWidths:vec3u;
var<workgroup> cm12HomeHaloEdge:u32;
var<workgroup> cm12HomeHaloRadius:u32;
var<workgroup> cm12HomeHaloShift:u32;
var<workgroup> cm12HomeHaloProfileFamily:u32;

struct CM12HomeHaloLookup{
  cell:u32,
  widths:vec3u,
  effectiveVelocity:vec4f,
  cached:u32,
}

fn cm12HomeHaloIndex(q:vec3u)->u32{
  return q.x+CM12_HOME_HALO_MAX_EDGE
    *(q.y+CM12_HOME_HALO_MAX_EDGE*q.z);
}

fn cm12HomeHaloGlobalLookup(q:vec3i)->CM12HomeHaloLookup{
  let owner=${ownerAtFine}(q);
  var value=vec4f(0.0);
  if(owner.cell!=${invalid}){value=${velocity}(owner.cell);}
  return CM12HomeHaloLookup(owner.cell,owner.widths,value,0u);
}

// Each lane is interpreted as one coordinate in a 4x4x4 packet using only
// shifts/masks. Its three strided loops cooperatively cover every halo site.
fn cm12StageTransportHomeFrameHalo(packetOriginFine:vec3i,homeShift:u32,
 radiusEstimate:u32,lane:u32,family:u32){
  let shift=min(homeShift,30u);
  let span=1u<<shift;
  let radius=min(radiusEstimate,CM12_HOME_HALO_MAX_RADIUS);
  if(lane==0u){
    cm12HomeHaloPacketOriginFine=packetOriginFine;
    cm12HomeHaloHomeWidths=vec3u(span);
    cm12HomeHaloRadius=radius;
    cm12HomeHaloEdge=CM12_HOME_HALO_PACKET_EDGE+2u*radius;
    cm12HomeHaloShift=shift;
    cm12HomeHaloProfileFamily=family;
    cm12HomeHaloCacheFirstFine=packetOriginFine
      -vec3i(i32(radius*span));
  }
  workgroupBarrier();
  let laneX=lane&3u;
  let laneY=(lane>>2u)&3u;
  let laneZ=(lane>>4u)&3u;
  for(var z=laneZ;z<cm12HomeHaloEdge;z+=4u){
    for(var y=laneY;y<cm12HomeHaloEdge;y+=4u){
      for(var x=laneX;x<cm12HomeHaloEdge;x+=4u){
        let local=vec3u(x,y,z);
        let q=cm12HomeHaloCacheFirstFine+vec3i(local*span);
        let owner=${ownerAtFine}(q);
        let sameHomeSpan=all(owner.widths==cm12HomeHaloHomeWidths);
        let index=cm12HomeHaloIndex(local);
        cm12HomeHaloCells[index]=${invalid};
        cm12HomeHaloVelocities[index]=vec4f(0.0);
        // Do not use select here: WGSL evaluates both operands, which would
        // turn an INVALID ordinal into an out-of-bounds plane load.
        if(owner.cell!=${invalid}&&sameHomeSpan){
          cm12HomeHaloCells[index]=owner.cell;
          cm12HomeHaloVelocities[index]=${velocity}(owner.cell);
          ${profile === undefined ? "" : `atomicAdd(
            &activity[${profile + 9}u+cm12HomeHaloProfileFamily],1u);`}
        }
      }
    }
  }
  workgroupBarrier();
}

fn cm12HomeHaloLookupAtFine(q:vec3i)->CM12HomeHaloLookup{
  ${profileAdd(39, "1u")}
  let relative=q-cm12HomeHaloCacheFirstFine;
  let home=relative>>vec3u(cm12HomeHaloShift);
  let inside=all(home>=vec3i(0))&&all(home<vec3i(i32(cm12HomeHaloEdge)));
  if(inside){
    let index=cm12HomeHaloIndex(vec3u(home));
    let cell=cm12HomeHaloCells[index];
    if(cell!=${invalid}){
      ${profileAdd(40, "1u")}
      return CM12HomeHaloLookup(cell,cm12HomeHaloHomeWidths,
        cm12HomeHaloVelocities[index],1u);
    }
  }
  // Covers underestimated travel, clipped radius > 3, topology seams, and
  // every probe/corner whose accepted span differs from the home packet.
  ${profile === undefined ? "return cm12HomeHaloGlobalLookup(q);" : `
  atomicAdd(&activity[${profile + 41}u],1u);
  if(!inside){atomicAdd(&activity[${profile + 43}u],1u);}
  let global=cm12HomeHaloGlobalLookup(q);
  if(global.cell==${invalid}){atomicAdd(&activity[${profile + 44}u],1u);}
  else if(any(global.widths!=cm12HomeHaloHomeWidths)){
    atomicAdd(&activity[${profile + 42}u],1u);
  }
  return global;`}
}

fn cm12HomeHaloGlobalSampleVelocity(position:vec3f)->vec3f{
  let probe=${ownerAtFine}(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(${dimensions})-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe.cell!=${invalid}){spans=vec3f(probe.widths);}
  let clamped=clamp(position,0.5*spans,vec3f(${dimensions})-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);
  let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let owner=${ownerAtFine}(vec3i(floor(lattice)));
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    if(owner.cell!=${invalid}){
      result+=wx*wy*wz*${velocity}(owner.cell).xyz;
    }
  }}}
  return result;
}

fn cm12HomeHaloSampleVelocity(position:vec3f)->vec3f{
  let probe=cm12HomeHaloLookupAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(${dimensions})-vec3f(1e-4)))));
  if(probe.cell==${invalid}||any(probe.widths!=cm12HomeHaloHomeWidths)){
    return cm12HomeHaloGlobalSampleVelocity(position);
  }
  let spans=vec3f(cm12HomeHaloHomeWidths);
  let clamped=clamp(position,0.5*spans,vec3f(${dimensions})-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);
  let lower=vec3i(floor(shifted));let fraction=fract(shifted);
  var result=vec3f(0.0);
  // Preserve the production dz/dy/dx order and exact weight expressions.
  for(var dz=0;dz<2;dz+=1){for(var dy=0;dy<2;dy+=1){for(var dx=0;dx<2;dx+=1){
    let lattice=spans*(vec3f(lower+vec3i(dx,dy,dz))+vec3f(0.5));
    let sample=cm12HomeHaloLookupAtFine(vec3i(floor(lattice)));
    // A corner owned at another accepted span is a seam crossing. Re-run the
    // complete sample in that corner's production lattice rather than mixing
    // two lattices under the home-frame weights.
    if(sample.cell!=${invalid}&&any(sample.widths!=cm12HomeHaloHomeWidths)){
      return cm12HomeHaloGlobalSampleVelocity(position);
    }
    let wx=select(1.0-fraction.x,fraction.x,dx==1);
    let wy=select(1.0-fraction.y,fraction.y,dy==1);
    let wz=select(1.0-fraction.z,fraction.z,dz==1);
    if(sample.cell!=${invalid}){result+=wx*wy*wz*sample.effectiveVelocity.xyz;}
  }}}
  return result;
}

fn cm12HomeHaloGlobalTransportStencil(position:vec3f)->${stencil}{
  let probe=${ownerAtFine}(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(${dimensions})-vec3f(1e-4)))));
  var spans=vec3f(1.0);if(probe.cell!=${invalid}){spans=vec3f(probe.widths);}
  let clamped=clamp(position,0.5*spans,vec3f(${dimensions})-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:${stencil};
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=clamp(vec3i(floor(spans*(vec3f(lower+offset)+vec3f(0.5)))),
      vec3i(0),vec3i(${dimensions})-vec3i(1));
    let cell=${ownerAtFine}(lattice).cell;
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=cell;
    result.weights[corner]=select(0.0,weight,cell!=${invalid});
  }
  return result;
}

fn cm12HomeHaloTransportStencil(position:vec3f)->${stencil}{
  let probe=cm12HomeHaloLookupAtFine(vec3i(floor(clamp(position,vec3f(0.0),
    vec3f(${dimensions})-vec3f(1e-4)))));
  if(probe.cell==${invalid}||any(probe.widths!=cm12HomeHaloHomeWidths)){
    return cm12HomeHaloGlobalTransportStencil(position);
  }
  let spans=vec3f(cm12HomeHaloHomeWidths);
  let clamped=clamp(position,0.5*spans,vec3f(${dimensions})-0.5*spans);
  let shifted=clamped/spans-vec3f(0.5);let lower=vec3i(floor(shifted));
  let fraction=fract(shifted);var result:${stencil};
  for(var corner=0u;corner<8u;corner+=1u){
    let offset=vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let lattice=clamp(vec3i(floor(spans*(vec3f(lower+offset)+vec3f(0.5)))),
      vec3i(0),vec3i(${dimensions})-vec3i(1));
    let sample=cm12HomeHaloLookupAtFine(lattice);
    if(sample.cell!=${invalid}&&any(sample.widths!=cm12HomeHaloHomeWidths)){
      return cm12HomeHaloGlobalTransportStencil(position);
    }
    let weight=select(1.0-fraction.x,fraction.x,offset.x==1)
      *select(1.0-fraction.y,fraction.y,offset.y==1)
      *select(1.0-fraction.z,fraction.z,offset.z==1);
    result.cells[corner]=sample.cell;
    result.weights[corner]=select(0.0,weight,sample.cell!=${invalid});
  }
  return result;
}

fn cm12HomeHaloTraceCharacteristic(position:vec3f,direction:f32)->vec3f{
  let initialPosition=${clampInside}(position);
  let initial=cm12HomeHaloSampleVelocity(initialPosition);
  let substeps=clamp(i32(ceil(length(initial)*${frameDt})),1,16);
  let subDt=${frameDt}/f32(substeps);var traced=initialPosition;
  let lower=vec3f(0.5);let upper=vec3f(${dimensions})-vec3f(0.5);
  for(var step=0;step<substeps;step+=1){
    var first=initial;if(step>0){first=cm12HomeHaloSampleVelocity(traced);}
    let midpoint=${clipSegment}(traced,
      clamp(traced+direction*0.5*subDt*first,lower,upper));
    let candidate=traced+direction*subDt*cm12HomeHaloSampleVelocity(midpoint);
    traced=${clipSegment}(traced,clamp(candidate,lower,upper));
  }
  return traced;
}
fn cm12HomeHaloTraceDeparture(position:vec3f)->vec3f{
  return cm12HomeHaloTraceCharacteristic(position,-1.0);
}
fn cm12HomeHaloTraceArrival(position:vec3f)->vec3f{
  return cm12HomeHaloTraceCharacteristic(position,1.0);
}
`;
}
