import type { SparseCM12Phase1TransportQALayout } from
  "./sparse-cm12-phase1-transport-receipt";

/** Binding-free QA hooks. `activity` is the resident read/write u32 arena. */
export function createSparseCM12Phase1TransportQAWGSL(options: {
  readonly layout: SparseCM12Phase1TransportQALayout;
  readonly publishEffectiveVelocity: boolean;
  readonly validateExecutionImage: boolean;
}): string {
  const l = options.layout;
  const publish = options.publishEffectiveVelocity
    ? "cm12PublishVexAcceptedEffectiveVelocity(cell,value);" : "_=value;";
  const publishTransferred = options.publishEffectiveVelocity
    ? "cm12PublishTransferredEffectiveVelocity(cell,velocity);" : "_=cell;_=velocity;";
  const validatePacket = options.validateExecutionImage
    ? `let actual=cm12TeiPacketCell(packet.x,packet.y,acceptedTopologySlot());
  if(actual!=cell){atomicAdd(&activity[CM12_P1TQ_HEADER+12u],1u);
    atomicStore(&activity[CM12_P1TQ_HEADER+13u],cell);
    atomicStore(&activity[CM12_P1TQ_HEADER+14u],actual);
    atomicStore(&activity[CM12_P1TQ_HEADER+15u],packet.x|(packet.y<<24u));}` : "";
  return /* wgsl */ `
const CM12_P1TQ_HEADER:u32=${l.baseWords}u;
const CM12_P1TQ_CAPACITY:u32=${l.cellCapacity}u;
const CM12_P1TQ_DEPARTURE:u32=${l.departureBaseWords}u;
const CM12_P1TQ_STENCIL_CELL:u32=${l.stencilCellBaseWords}u;
const CM12_P1TQ_STENCIL_WEIGHT:u32=${l.stencilWeightBaseWords}u;
const CM12_P1TQ_BETA:u32=${l.betaBaseWords}u;
const CM12_P1TQ_DEFICIT_DENSITY:u32=${l.deficitDensityBaseWords}u;
const CM12_P1TQ_DEFICIT_GAMMA:u32=${l.deficitGammaBaseWords}u;
const CM12_P1TQ_MASS_DENSITY:u32=${l.massDensityBaseWords}u;
const CM12_P1TQ_MASS_GAMMA:u32=${l.massGammaBaseWords}u;
const CM12_P1TQ_PACKET_ID:u32=${l.packetIdBaseWords}u;
const CM12_P1TQ_PACKET_LANE:u32=${l.packetLaneBaseWords}u;

fn cm12Phase1QACanonicalPacket(cell:u32)->vec2u{
  let brick=cellBrick(cell);let resolution=cellResolution(cell);
  let origin=cm12WorldLeafCoordinate(brick)*i32(BRICK_FINE_RESOLUTION);
  let scale=BRICK_FINE_RESOLUTION*brickSpan(brick)/resolution;
  let minimum=cellMinimum(cell);
  if(any(minimum<origin)){return vec2u(0xffffffffu);}
  let local=vec3u(minimum-origin)/scale;
  let packetAxis=max(1u,(resolution+3u)/4u);let coordinate=local/4u;
  let packet=64u*brick+coordinate.x+packetAxis
    *(coordinate.y+packetAxis*coordinate.z);
  let lane=(local.x&3u)+4u*((local.y&3u)+4u*(local.z&3u));
  return vec2u(packet,lane);
}

@compute @workgroup_size(64)
fn captureSparseCM12Phase1TransportPackets(
 @builtin(global_invocation_id)gid:vec3u){
  let cell=acceptedTemplateCellInvocation(gid.x);
  if(cell==0xffffffffu||!cellActive(cell)){return;}
  let packet=cm12Phase1QACanonicalPacket(cell);
  atomicStore(&activity[CM12_P1TQ_PACKET_ID+cell],packet.x);
  atomicStore(&activity[CM12_P1TQ_PACKET_LANE+cell],packet.y+1u);
  atomicStore(&activity[CM12_P1TQ_HEADER+4u],ptrTopologyGeneration());
  atomicAdd(&activity[CM12_P1TQ_HEADER+11u],1u);
  ${validatePacket}
}

fn cm12Phase1QAPublishVexAcceptedEffectiveVelocity(cell:u32,value:vec4f){
  ${publish}
  atomicStore(&activity[CM12_P1TQ_HEADER+5u],cm12FCCandidateGeneration());
  atomicStore(&activity[CM12_P1TQ_HEADER+6u],ptrTopologyGeneration());
}
fn cm12Phase1QAPublishTransferredEffectiveVelocity(cell:u32,velocity:vec3f){
  ${publishTransferred}
  atomicStore(&activity[CM12_P1TQ_HEADER+6u],ptrTopologyGeneration());
}
fn cm12Phase1QACaptureTrace(cell:u32,departure:vec3f,stencil:TransportStencil){
  if(cell>=CM12_P1TQ_CAPACITY){return;}
  atomicStore(&activity[CM12_P1TQ_HEADER+3u],cm12FCCandidateGeneration());
  atomicStore(&activity[CM12_P1TQ_HEADER+4u],ptrTopologyGeneration());
  for(var axis=0u;axis<3u;axis+=1u){
    atomicStore(&activity[CM12_P1TQ_DEPARTURE+3u*cell+axis],bitcast<u32>(departure[axis]));
  }
  for(var corner=0u;corner<8u;corner+=1u){
    atomicStore(&activity[CM12_P1TQ_STENCIL_CELL+8u*cell+corner],stencil.cells[corner]);
    atomicStore(&activity[CM12_P1TQ_STENCIL_WEIGHT+8u*cell+corner],
      bitcast<u32>(stencil.weights[corner]));
  }
  atomicAdd(&activity[CM12_P1TQ_HEADER+7u],1u);
}
fn cm12Phase1QACaptureBeta(cell:u32,value:i32){
  if(cell>=CM12_P1TQ_CAPACITY){return;}
  atomicStore(&activity[CM12_P1TQ_BETA+cell],bitcast<u32>(value));
  atomicAdd(&activity[CM12_P1TQ_HEADER+8u],1u);
}
fn cm12Phase1QACaptureDeficit(cell:u32,density:i32,gamma:i32){
  if(cell>=CM12_P1TQ_CAPACITY){return;}
  atomicStore(&activity[CM12_P1TQ_DEFICIT_DENSITY+cell],bitcast<u32>(density));
  atomicStore(&activity[CM12_P1TQ_DEFICIT_GAMMA+cell],bitcast<u32>(gamma));
  atomicAdd(&activity[CM12_P1TQ_HEADER+9u],1u);
}
fn cm12Phase1QACaptureMass(cell:u32,density:f32,gamma:f32){
  if(cell>=CM12_P1TQ_CAPACITY){return;}
  atomicStore(&activity[CM12_P1TQ_MASS_DENSITY+cell],bitcast<u32>(density));
  atomicStore(&activity[CM12_P1TQ_MASS_GAMMA+cell],bitcast<u32>(gamma));
  atomicAdd(&activity[CM12_P1TQ_HEADER+10u],1u);
}
`;
}

export const SPARSE_CM12_PHASE1_TRANSPORT_QA_WGSL_STUBS = /* wgsl */ `
fn cm12Phase1QACaptureTrace(cell:u32,departure:vec3f,stencil:TransportStencil){
  _=cell;_=departure;_=stencil;
}
fn cm12Phase1QACaptureBeta(cell:u32,value:i32){_=cell;_=value;}
fn cm12Phase1QACaptureDeficit(cell:u32,density:i32,gamma:i32){
  _=cell;_=density;_=gamma;
}
fn cm12Phase1QACaptureMass(cell:u32,density:f32,gamma:f32){
  _=cell;_=density;_=gamma;
}
`;
