import {
  SPARSE_CM12_VELOCITY_EXTENSION_DEPTH,
  SPARSE_CM12_VELOCITY_EXTENSION_DISPATCH_WIDTH,
  SPARSE_CM12_VELOCITY_EXTENSION_HEADER,
  type SparseCM12VelocityExtensionLayout,
} from "./sparse-cm12-velocity-extension";

export interface SparseCM12VelocityExtensionWGSLOptions {
  readonly layout: SparseCM12VelocityExtensionLayout;
  readonly arenaName?: string;
  readonly stateName?: string;
  readonly topologyGenerationExpression?: string;
  readonly sourceFrameGenerationExpression?: string;
  readonly effectiveVelocityHookPrefix?: string;
  /** QA-only A/B: consume the packet authority's accepted-packet list. */
  readonly compactAcceptedPacketsForQA?: boolean;
  /** Production topology-cached schedule; QA flag forces its compact mode. */
  readonly cacheAcceptedPackets?: boolean;
  /** Bake one recurrence depth into a bounded pipeline slice when requested. */
  readonly fixedRecurrenceDepth?: number;
  /** Consume the generation-sealed CNX connectivity image in the fallback path. */
  readonly compiledTopology?: boolean;
  /** WGSL predicate selecting authoritative liquid seeds. Production passes
   * the accepted adaptive level-set phase; standalone fixtures may retain the
   * legacy scalar predicate until they compose the level-set module. */
  readonly liquidCellPredicate?: (cellExpression: string) => string;
}

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/**
 * VEX2 is one packet initialization and eight Jacobi transforms.
 * Successful sweep publication is the commit. Masks are packet-owned and
 * overwritten on every accepted packet; rebuild initialization clears both
 * retired mask banks before the sweeps consume the accepted schedule.
 */
export function createSparseCM12VelocityExtensionWGSL(
  options: SparseCM12VelocityExtensionWGSLOptions,
): string {
  const layout = options.layout;
  const arena = identifier(options.arenaName ?? "activity", "arenaName");
  const state = identifier(options.stateName ?? "state", "stateName");
  const topologyGeneration = options.topologyGenerationExpression
    ?? `atomicLoad(&${arena}[12])`;
  const frameGeneration = options.sourceFrameGenerationExpression ?? "1u";
  const effectiveHook = options.effectiveVelocityHookPrefix
    ? identifier(options.effectiveVelocityHookPrefix, "effectiveVelocityHookPrefix")
    : undefined;
  const publish = effectiveHook
    ? `${effectiveHook}PublishVexAcceptedEffectiveVelocity(cell,value);` : "";
  const acceptedVelocity = effectiveHook
    ? `let accepted=${effectiveHook}EffectiveTransportVelocity(cell);`
    : `let input=sourceCellVelocity()+4u*cell;
    let accepted=vec4f(${state}[input],${state}[input+1u],${state}[input+2u],1.0);`;
  const fixedRecurrenceDepth = options.fixedRecurrenceDepth;
  const dispatchPacket = (initial: boolean) => options.cacheAcceptedPackets
    ? /* wgsl */ `let dispatchOrdinal=wid.x+cm12ExtensionDispatchWidth*wid.y;
  if(lane==0u){
    var selected=cm12ExtensionStablePacket(dispatchOrdinal);
    if(cm12ExtensionLoad(CM12_VEX_SCHEDULE+3u)!=0u${initial ? "&&cm12ExtensionLoad(CM12_VEX_SCHEDULE+1u)==0u" : ""}){
      selected=cm12ExtensionInvalid;
      if(dispatchOrdinal<cm12ExtensionLoad(CM12_VEX_SCHEDULE+2u)){
        selected=cm12ExtensionLoad(CM12_VEX_PACKET_LIST+dispatchOrdinal);
      }
    }
    cm12ExtensionDispatchPacket=selected;
  }
  let packet=workgroupUniformLoad(&cm12ExtensionDispatchPacket);`
    : options.compactAcceptedPacketsForQA
    ? /* wgsl */ `let dispatchOrdinal=wid.x;
  if(lane==0u){
    let compactOrdinal=cm12TransportPacketOrdinal(dispatchOrdinal);
    cm12ExtensionDispatchPacket=select(cm12ExtensionInvalid,
      cm12TransportDirectStablePacket(compactOrdinal),
      compactOrdinal<CM12_TPA_DIRECT_PACKET_COUNT);
  }
  let packet=workgroupUniformLoad(&cm12ExtensionDispatchPacket);`
    : /* wgsl */ `let dispatchOrdinal=wid.x+cm12ExtensionDispatchWidth*wid.y;
  let packet=cm12ExtensionStablePacket(dispatchOrdinal);`;
  if (fixedRecurrenceDepth !== undefined
    && (!Number.isInteger(fixedRecurrenceDepth) || fixedRecurrenceDepth < 1
      || fixedRecurrenceDepth > SPARSE_CM12_VELOCITY_EXTENSION_DEPTH)) {
    throw new RangeError(`fixedRecurrenceDepth must be in [1, ${SPARSE_CM12_VELOCITY_EXTENSION_DEPTH}]`);
  }
  const recurrenceDepthDeclaration = fixedRecurrenceDepth === undefined
    ? /* wgsl */ `
struct CM12ExtensionDispatch{depth:u32}
@group(0)@binding(5)var<uniform>cm12ExtensionDispatch:CM12ExtensionDispatch;
fn cm12ExtensionDepth()->u32{return clamp(cm12ExtensionDispatch.depth,1u,
  ${SPARSE_CM12_VELOCITY_EXTENSION_DEPTH}u);}`
    : /* wgsl */ `fn cm12ExtensionDepth()->u32{return ${fixedRecurrenceDepth}u;}`;
  const finalValidityBaseWords = (SPARSE_CM12_VELOCITY_EXTENSION_DEPTH & 1) === 0
    ? layout.validityABaseWords : layout.validityBBaseWords;
  const compiledTopologyFence = options.compiledTopology ? /* wgsl */ `
  if(lane==0u){cm12ExtensionCompiledTopologyValid=u32(cnxAccepted());}
  if(workgroupUniformLoad(&cm12ExtensionCompiledTopologyValid)==0u){
    cm12ExtensionPublishFrameReceipt(dispatchOrdinal,lane);return;
  }` : "";
  const liquidCellPredicate = options.liquidCellPredicate?.("cell")
    ?? `${state}[sourceDensity()+cell]>CM12_LIQUID_ISOVALUE`;
  const connectivityTraversal = options.compiledTopology ? /* wgsl */ `
        let incidences=cnxCellIncidenceRangeUnchecked(cell);
        for(var at=incidences.x;at<incidences.y;at+=1u){
          let rowOrdinal=cnxIncidenceRowOrdinalUnchecked(at);
          let row=cnxStableRowUnchecked(rowOrdinal);
          if(row==cm12ExtensionInvalid||!cm12VelocityExtensionRowOpen(row)){continue;}
          let range=cnxRowTermRangeByOrdinalUnchecked(rowOrdinal);
          let termCount=range.y-range.x;
          let ownTerm=cnxIncidenceOwnTermUnchecked(at);
          let own=cnxRowTermCoefficientUnchecked(ownTerm);
          let axis=cnxRowPackedMetadataByOrdinal(rowOrdinal)&3u;
          let side=2u*axis+select(0u,1u,own<0.0);
          if(termCount==2u){
            let ordinal=(ownTerm-range.x)^1u;
            let neighbor=cnxRowTermCellUnchecked(range.x+ordinal);
            if(neighbor==cm12ExtensionInvalid){continue;}
            if(cm12ExtensionLoad(cm12ExtensionAcceptedDepth+neighbor)
              >=cm12ExtensionDepth()){continue;}
            let w=cm12VelocityExtensionNeighborWeight(row,own,
              cnxRowTermCoefficientUnchecked(range.x+ordinal));
            terms[side]+=w*vec4f(cm12EffectiveTransportVelocity(neighbor).xyz,1.0);continue;
          }
          for(var ordinal=0u;ordinal<termCount;ordinal+=1u){
            let neighbor=cnxRowTermCellUnchecked(range.x+ordinal);
            if(neighbor==cell||neighbor==cm12ExtensionInvalid){continue;}
            if(cm12ExtensionLoad(cm12ExtensionAcceptedDepth+neighbor)
              >=cm12ExtensionDepth()){continue;}
            let w=cm12VelocityExtensionNeighborWeight(row,own,
              cnxRowTermCoefficientUnchecked(range.x+ordinal));
            terms[side]+=w*vec4f(cm12EffectiveTransportVelocity(neighbor).xyz,1.0);
          }
        }` : /* wgsl */ `
        let incidences=cm12HotIncidenceRange(cell);
        for(var local=0u;local<incidences.y;local+=1u){
          let at=incidences.x+local;let row=incidenceRow(at);
          if(row==cm12ExtensionInvalid||!cm12VelocityExtensionRowOpen(row)){continue;}
          let range=rowTermRange(row);let termCount=range.y-range.x;
          let ownTerm=incidenceTerm(at);let own=termCoefficient(ownTerm);
          let side=2u*rowAxis(row)+select(0u,1u,own<0.0);
          if(termCount==2u){
            let ordinal=(ownTerm-range.x)^1u;
            let neighbor=termCell(range.x+ordinal);
            if(neighbor==cm12ExtensionInvalid){continue;}
            if(cm12ExtensionLoad(cm12ExtensionAcceptedDepth+neighbor)
              >=cm12ExtensionDepth()){continue;}
            let w=cm12VelocityExtensionNeighborWeight(row,own,termCoefficient(range.x+ordinal));
            terms[side]+=w*vec4f(cm12EffectiveTransportVelocity(neighbor).xyz,1.0);continue;
          }
          for(var ordinal=0u;ordinal<termCount;ordinal+=1u){
            let neighbor=termCell(range.x+ordinal);
            if(neighbor==cell||neighbor==cm12ExtensionInvalid){continue;}
            if(cm12ExtensionLoad(cm12ExtensionAcceptedDepth+neighbor)
              >=cm12ExtensionDepth()){continue;}
            let w=cm12VelocityExtensionNeighborWeight(row,own,termCoefficient(range.x+ordinal));
            terms[side]+=w*vec4f(cm12EffectiveTransportVelocity(neighbor).xyz,1.0);
          }
        }`;
  const h = (word: number) => `${layout.headerBaseWords + word}u`;
  return /* wgsl */ `
const cm12ExtensionInvalid:u32=0xffffffffu;
const cm12ExtensionCapacity:u32=${layout.cellCapacity}u;
const cm12ExtensionPacketCapacity:u32=${layout.packetCapacity}u;
const cm12ExtensionDispatchPacketsPerLeaf:u32=${layout.dispatchPacketsPerLeaf}u;
const cm12ExtensionDispatchPacketCount:u32=${layout.dispatchPacketCount}u;
const cm12ExtensionValidityA:u32=${layout.validityABaseWords}u;
const cm12ExtensionValidityB:u32=${layout.validityBBaseWords}u;
const cm12ExtensionAcceptedDepth:u32=${layout.acceptedDepthBaseWords}u;
const cm12ExtensionDispatchWidth:u32=${SPARSE_CM12_VELOCITY_EXTENSION_DISPATCH_WIDTH}u;
${recurrenceDepthDeclaration}

var<workgroup> cm12ExtensionBallotLow:atomic<u32>;
var<workgroup> cm12ExtensionBallotHigh:atomic<u32>;
var<workgroup> cm12ExtensionPacketFirst:u32;
var<workgroup> cm12ExtensionPacketCounts:vec3u;
var<workgroup> cm12ExtensionPacketStrides:vec2u;
var<workgroup> cm12ExtensionPacketLocal:vec3u;
var<workgroup> cm12ExtensionLeafValid:vec3u;
var<workgroup> cm12ExtensionLeafScale:u32;
var<workgroup> cm12ExtensionInputMaskLow:u32;
var<workgroup> cm12ExtensionInputMaskHigh:u32;
var<workgroup> cm12ExtensionPacketComplete:u32;
var<workgroup> cm12ExtensionDispatchPacket:u32;
${options.compiledTopology
    ? "var<workgroup> cm12ExtensionCompiledTopologyValid:u32;" : ""}

fn cm12ExtensionLoad(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn cm12ExtensionStore(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn cm12ExtensionStablePacket(dispatchOrdinal:u32)->u32{
  if(dispatchOrdinal>=cm12ExtensionDispatchPacketCount){return cm12ExtensionInvalid;}
  let leaf=dispatchOrdinal/cm12ExtensionDispatchPacketsPerLeaf;
  let local=dispatchOrdinal%cm12ExtensionDispatchPacketsPerLeaf;
  let packet=64u*leaf+local;
  return select(packet,cm12ExtensionInvalid,packet>=cm12ExtensionPacketCapacity);
}
${options.cacheAcceptedPackets ? /* wgsl */ `
const CM12_VEX_SCHEDULE:u32=${layout.scheduleBaseWords}u;
const CM12_VEX_PACKET_LIST:u32=${layout.packetListBaseWords}u;
@compute @workgroup_size(1)
fn beginSparseCM12VelocityExtensionSchedule(){
  let changed=cm12ExtensionLoad(CM12_VEX_SCHEDULE)!=${topologyGeneration}
    ||cm12ExtensionLoad(CM12_VEX_SCHEDULE+10u)!=acceptedTopologySlot();
  cm12ExtensionStore(CM12_VEX_SCHEDULE+1u,select(0u,1u,changed));
  if(changed){cm12ExtensionStore(CM12_VEX_SCHEDULE+2u,0u);}
}
@compute @workgroup_size(64)
fn compileSparseCM12VelocityExtensionSchedule(@builtin(global_invocation_id)gid:vec3u){
  if(cm12ExtensionLoad(CM12_VEX_SCHEDULE+1u)==0u){return;}
  let leaf=acceptedLeafInvocation(gid.x);if(leaf==cm12ExtensionInvalid){return;}
  let slot=acceptedTopologySlot();let descriptor=cm12TeiLoadLeaf(slot,leaf);
  if((descriptor.flags&0x80000000u)==0u){return;}
  let axis=max(1u,((descriptor.flags&31u)+3u)/4u);
  let count=min(cm12ExtensionDispatchPacketsPerLeaf,axis*axis*axis);
  for(var local=0u;local<count;local+=1u){
    let packet=64u*leaf+local;
    if(cm12TeiPacket(packet,slot).first==cm12ExtensionInvalid){continue;}
    let at=atomicAdd(&${arena}[CM12_VEX_SCHEDULE+2u],1u);
    if(at<cm12ExtensionDispatchPacketCount){cm12ExtensionStore(CM12_VEX_PACKET_LIST+at,packet);}
  }
}
@compute @workgroup_size(1)
fn sealSparseCM12VelocityExtensionSchedule(){
  let count=min(cm12ExtensionLoad(CM12_VEX_SCHEDULE+2u),cm12ExtensionDispatchPacketCount);
  // Initialization scrubs retired masks once. Sweeps can immediately use the
  // accepted list even after a topology change; dense images stay direct.
  let compact=${options.compactAcceptedPacketsForQA ? 'true' : 'count<cm12ExtensionDispatchPacketCount-cm12ExtensionDispatchPacketCount/4u'};
  cm12ExtensionStore(CM12_VEX_SCHEDULE+3u,select(0u,1u,compact));
  let initialCompact=compact&&cm12ExtensionLoad(CM12_VEX_SCHEDULE+1u)==0u;
  let initialGroups=max(1u,select(cm12ExtensionDispatchPacketCount,count,initialCompact));
  let sweepGroups=max(1u,select(cm12ExtensionDispatchPacketCount,count,compact));
  cm12ExtensionStore(CM12_VEX_SCHEDULE+4u,min(cm12ExtensionDispatchWidth,initialGroups));
  cm12ExtensionStore(CM12_VEX_SCHEDULE+5u,(initialGroups+cm12ExtensionDispatchWidth-1u)/cm12ExtensionDispatchWidth);
  cm12ExtensionStore(CM12_VEX_SCHEDULE+6u,1u);
  cm12ExtensionStore(CM12_VEX_SCHEDULE+7u,min(cm12ExtensionDispatchWidth,sweepGroups));
  cm12ExtensionStore(CM12_VEX_SCHEDULE+8u,(sweepGroups+cm12ExtensionDispatchWidth-1u)/cm12ExtensionDispatchWidth);
  cm12ExtensionStore(CM12_VEX_SCHEDULE+9u,1u);
  cm12ExtensionStore(CM12_VEX_SCHEDULE,${topologyGeneration});
  cm12ExtensionStore(CM12_VEX_SCHEDULE+10u,acceptedTopologySlot());
}
` : ""}
fn cm12ExtensionExpectedMask(counts:vec3u)->vec2u{
  var result=vec2u(0u);let rowMask=(1u<<counts.x)-1u;
  for(var z=0u;z<counts.z;z+=1u){for(var y=0u;y<counts.y;y+=1u){
    let first=4u*y+16u*z;
    if(first<32u){result.x|=rowMask<<first;}else{result.y|=rowMask<<(first-32u);}
  }}
  return result;
}
fn cm12ExtensionBeginInitialPacket(packet:u32,lane:u32)->u32{
  if(lane==0u){
    let item=cm12TeiPacket(packet,acceptedTopologySlot());
    cm12ExtensionPacketFirst=item.first;cm12ExtensionPacketCounts=item.counts;
    cm12ExtensionPacketStrides=vec2u(item.strideY,item.strideZ);
    if(item.first!=cm12ExtensionInvalid){
      atomicStore(&cm12ExtensionBallotLow,0u);
      atomicStore(&cm12ExtensionBallotHigh,0u);
    }
  }
  return workgroupUniformLoad(&cm12ExtensionPacketFirst);
}
fn cm12ExtensionBeginPacket(packet:u32,lane:u32)->u32{
  if(lane==0u){
    let slot=acceptedTopologySlot();let item=cm12TeiPacket(packet,slot);
    cm12ExtensionPacketFirst=item.first;cm12ExtensionPacketCounts=item.counts;
    cm12ExtensionPacketStrides=vec2u(item.strideY,item.strideZ);
    cm12ExtensionPacketComplete=0u;
    if(item.first!=cm12ExtensionInvalid){
      atomicStore(&cm12ExtensionBallotLow,0u);
      atomicStore(&cm12ExtensionBallotHigh,0u);
    }
    if(item.first!=cm12ExtensionInvalid){
      let inputMask=select(cm12ExtensionValidityB,cm12ExtensionValidityA,
        (cm12ExtensionDepth()&1u)==1u);
      if(packet<cm12ExtensionPacketCapacity){
        cm12ExtensionInputMaskLow=cm12ExtensionLoad(inputMask+2u*packet);
        cm12ExtensionInputMaskHigh=cm12ExtensionLoad(inputMask+2u*packet+1u);
      }else{
        cm12ExtensionInputMaskLow=0u;cm12ExtensionInputMaskHigh=0u;
      }
      let expected=cm12ExtensionExpectedMask(item.counts);
      cm12ExtensionPacketComplete=u32(any(expected!=vec2u(0u))
        &&all(vec2u(cm12ExtensionInputMaskLow,cm12ExtensionInputMaskHigh)==expected));
      if(cm12ExtensionPacketComplete==0u){
        let brick=packet/64u;let localPacket=packet%64u;
        let leaf=cm12TeiLoadLeaf(slot,brick);let resolution=leaf.flags&31u;
        let packetAxis=max(1u,(resolution+3u)/4u);
        let pz=localPacket/(packetAxis*packetAxis);
        let remainder=localPacket-pz*packetAxis*packetAxis;
        let py=remainder/packetAxis;let px=remainder-py*packetAxis;
        cm12ExtensionPacketLocal=4u*vec3u(px,py,pz);
        cm12ExtensionLeafValid=leaf.valid;cm12ExtensionLeafScale=leaf.scale;
      }
    }
  }
  return workgroupUniformLoad(&cm12ExtensionPacketFirst);
}
fn cm12ExtensionPacketCell(lane:u32)->u32{
  let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  if(cm12ExtensionPacketFirst==cm12ExtensionInvalid
    ||any(q>=cm12ExtensionPacketCounts)){return cm12ExtensionInvalid;}
  return cm12ExtensionPacketFirst+q.x+cm12ExtensionPacketStrides.x*q.y
    +cm12ExtensionPacketStrides.y*q.z;
}
fn cm12ExtensionPacketInputValid(lane:u32)->bool{
  let word=select(cm12ExtensionInputMaskLow,cm12ExtensionInputMaskHigh,lane>=32u);
  return ((word>>(lane&31u))&1u)!=0u;
}
fn cm12ExtensionPublishBallot(base:u32,packet:u32,lane:u32,valid:bool){
  if(valid){if(lane<32u){_=atomicOr(&cm12ExtensionBallotLow,1u<<lane);}
    else{_=atomicOr(&cm12ExtensionBallotHigh,1u<<(lane-32u));}}
  workgroupBarrier();
  if(lane==0u&&packet<cm12ExtensionPacketCapacity){
    cm12ExtensionStore(base+2u*packet,atomicLoad(&cm12ExtensionBallotLow));
    cm12ExtensionStore(base+2u*packet+1u,atomicLoad(&cm12ExtensionBallotHigh));}}
fn cm12ExtensionPublishFrameReceipt(dispatchOrdinal:u32,lane:u32){
  if(dispatchOrdinal==0u&&lane==0u){
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.sourceFrameGeneration)},
      ${frameGeneration});
    cm12ExtensionStore(${h(SPARSE_CM12_VELOCITY_EXTENSION_HEADER.topologyGeneration)},
      ${topologyGeneration});}}

fn cm12ExtensionOutputMask(depth:u32)->u32{return select(
  cm12ExtensionValidityA,cm12ExtensionValidityB,(depth&1u)==1u);}
fn cm12ExtendedPacketMask(packet:u32)->vec2u{
  if(packet>=cm12ExtensionPacketCapacity){return vec2u(0u);}
  let at=${finalValidityBaseWords}u+2u*packet;
  return vec2u(cm12ExtensionLoad(at),cm12ExtensionLoad(at+1u));
}
fn cm12ExtendedPacketLaneSelected(packet:u32,lane:u32)->bool{
  if(packet>=cm12ExtensionPacketCapacity||lane>=64u){return false;}
  let word=cm12ExtensionLoad(${finalValidityBaseWords}u+2u*packet+(lane>>5u));
  return ((word>>(lane&31u))&1u)!=0u;
}
fn cm12ExtendedCellSelected(cell:u32)->bool{
  return cell<cm12ExtensionCapacity
    &&cm12ExtensionLoad(cm12ExtensionAcceptedDepth+cell)!=cm12ExtensionInvalid;
}

@compute @workgroup_size(64)
fn initializeVelocityExtensionPackets(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  ${dispatchPacket(true)}
  if(packet==cm12ExtensionInvalid){return;}
  // Initialization is embarrassingly packet-local. Reading the four-word TEI
  // descriptor per lane costs a few startup loads, but removes the shared
  // Jacobi topology and its workgroup state from this pipeline altogether.
  let item=cm12TeiPacket(packet,acceptedTopologySlot());
  if(lane==0u){
    cm12ExtensionStore(cm12ExtensionValidityA+2u*packet,0u);
    cm12ExtensionStore(cm12ExtensionValidityA+2u*packet+1u,0u);
    if(item.first==cm12ExtensionInvalid){
      cm12ExtensionStore(cm12ExtensionValidityB+2u*packet,0u);
      cm12ExtensionStore(cm12ExtensionValidityB+2u*packet+1u,0u);
    }
  }
  storageBarrier();workgroupBarrier();
  if(item.first==cm12ExtensionInvalid){return;}
  let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
  let cell=select(item.first+q.x+item.strideY*q.y+item.strideZ*q.z,
    cm12ExtensionInvalid,any(q>=item.counts));
  // TEI packets are emitted only for accepted active leaves; repeating the
  // resident topology predicate here imported the complete cell-ownership
  // graph into both otherwise packet-local programs.
  let selected=cell!=cm12ExtensionInvalid&&cell<cm12ExtensionCapacity;
  // Air-side fractions receive the liquid's extrapolated velocity. Making
  // them independent seeds changes the free-surface transport boundary.
  let wet=selected&&(${liquidCellPredicate});
  if(selected){
    // A hooked resident already has the accepted velocity authority in its
    // effective plane. In particular, the post-pressure rebuild must retain
    // the projection just published there rather than reload the stale source
    // frame bank. Standalone fixtures without a plane keep the legacy input.
    ${acceptedVelocity}
    let value=select(vec4f(0.0),vec4f(accepted.xyz,1.0),wet);${publish}
    cm12ExtensionStore(cm12ExtensionAcceptedDepth+cell,
      select(cm12ExtensionInvalid,0u,wet));}
  if(wet){_=atomicOr(&${arena}[cm12ExtensionValidityA+2u*packet+(lane>>5u)],
    1u<<(lane&31u));}
}

// Use only the opposite side of the physical subface. A mixed gradient
// also contains same-side fine siblings; those are not extension neighbours.
fn cm12VelocityExtensionNeighborWeight(row:u32,own:f32,other:f32)->f32{
  return cm12PhysicalSubfaceArea(row,own,other)/max(rowDistance(row),1e-9);
}

@compute @workgroup_size(64)
fn advanceVelocityExtensionPackets(@builtin(workgroup_id)wid:vec3u,
 @builtin(local_invocation_index)lane:u32){
  ${dispatchPacket(false)}
  if(packet==cm12ExtensionInvalid){
    cm12ExtensionPublishFrameReceipt(dispatchOrdinal,lane);return;}
  let packetFirst=cm12ExtensionBeginPacket(packet,lane);
  if(packetFirst==cm12ExtensionInvalid){
    cm12ExtensionPublishFrameReceipt(dispatchOrdinal,lane);return;}
  ${compiledTopologyFence}
  let packetComplete=workgroupUniformLoad(&cm12ExtensionPacketComplete);
  if(packetComplete!=0u){
    if(lane==0u){let output=cm12ExtensionOutputMask(cm12ExtensionDepth());
      cm12ExtensionStore(output+2u*packet,cm12ExtensionInputMaskLow);
      cm12ExtensionStore(output+2u*packet+1u,cm12ExtensionInputMaskHigh);}
    cm12ExtensionPublishFrameReceipt(dispatchOrdinal,lane);return;
  }
  let cell=cm12ExtensionPacketCell(lane);
  let selected=cell!=cm12ExtensionInvalid&&cell<cm12ExtensionCapacity;
  var valid=false;
  if(selected){
    valid=cm12ExtensionPacketInputValid(lane);
    if(!valid){var terms:array<vec4f,6>;
      let q=vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
      let leafLocal=cm12ExtensionPacketLocal+q;
      if(!hasStaticSolidVoxels()&&cm12ExtensionLeafScale!=0u
        &&all(leafLocal>vec3u(0u))
        &&all(leafLocal+vec3u(1u)<cm12ExtensionLeafValid)){
        // Canonical intra-leaf incidences are x-, x+, y-, y+, z-, z+.
        let neighbors=array<u32,6>(cell-1u,cell+1u,
          cell-cm12ExtensionPacketStrides.x,cell+cm12ExtensionPacketStrides.x,
          cell-cm12ExtensionPacketStrides.y,cell+cm12ExtensionPacketStrides.y);
        let w=1.0/f32(cm12ExtensionLeafScale);
        for(var ordinal=0u;ordinal<6u;ordinal+=1u){
          let neighbor=neighbors[ordinal];
          if(cm12ExtensionLoad(cm12ExtensionAcceptedDepth+neighbor)
            >=cm12ExtensionDepth()){continue;}
          terms[ordinal]=w*vec4f(cm12EffectiveTransportVelocity(neighbor).xyz,1.0);
        }
      }else{
        ${connectivityTraversal}
      }
      let x=min(terms[0],terms[1])+max(terms[0],terms[1]);
      let y=min(terms[2],terms[3])+max(terms[2],terms[3]);
      let z=min(terms[4],terms[5])+max(terms[4],terms[5]);
      let sum=(x+y)+z;
      valid=sum.w>0.0;if(valid){let velocity=sum.xyz/sum.w;
        let value=vec4f(velocity,1.0);${publish}
        cm12ExtensionStore(cm12ExtensionAcceptedDepth+cell,
          cm12ExtensionDepth());}}
  }
  cm12ExtensionPublishBallot(cm12ExtensionOutputMask(
    cm12ExtensionDepth()),packet,lane,valid);
  cm12ExtensionPublishFrameReceipt(dispatchOrdinal,lane);
}
`;
}
