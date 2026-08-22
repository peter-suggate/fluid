import {
  SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER as H,
  SPARSE_CM12_TOPOLOGY_EFFECTS_INVALID,
  SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC,
  SPARSE_CM12_TOPOLOGY_EFFECTS_PHASE as PHASE,
  SPARSE_CM12_TOPOLOGY_EFFECTS_VERSION,
  type SparseCM12TopologyEffectsAuthorityLayout,
} from "./sparse-cm12-topology-effects-authority";

const id = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError("WGSL identifier");
  return value;
};

/**
 * Binding-free candidate-effects authority. Required downstream hooks:
 * tfxPTRTargetGeneration(), tfxPTRReady(generation,newCount,newLeafCount),
 * tfxPTRWillAppend(brick,generation), tfxPTRDirtyLeafWillAppend(leaf,generation),
 * tfxPTRCompatible(brick,oldState,newState),
 * tfxPTRPublish(brick,oldState,newState,cause,ownsLeaf,generation).
 * The publication entry point may be dispatched only after the enclosing
 * topology transaction authorizes. Queue ordering then makes the singleton
 * finish entry point infallible; coveredEffects is QA observability, never a
 * post-write rejection gate.
 */
export function createSparseCM12TopologyEffectsAuthorityWGSL(options: Readonly<{
  layout: SparseCM12TopologyEffectsAuthorityLayout;
  arenaName?: string;
  prefix?: string;
  authorizationExpression: string;
}>): string {
  const l = options.layout, arena = id(options.arenaName ?? "topologyArena");
  const p = id(options.prefix ?? "tfx");
  const a = options.authorizationExpression;
  const h = (word: number) => `${l.baseWords + word}u`;
  return /* wgsl */ `
const ${p}Invalid:u32=${SPARSE_CM12_TOPOLOGY_EFFECTS_INVALID}u;
const ${p}GenerationBusy:u32=0x80000000u;
const ${p}PtrCapacity:u32=${l.ptrCapacity}u;
const ${p}PtrLeafCapacity:u32=${l.ptrLeafCapacity}u;
fn ${p}HeaderValid()->bool{return atomicLoad(&${arena}[${h(H.magic)}])
 ==0x${SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC.toString(16)}u
 &&atomicLoad(&${arena}[${h(H.version)}])==${SPARSE_CM12_TOPOLOGY_EFFECTS_VERSION}u
 &&atomicLoad(&${arena}[${h(H.totalWords)}])==${l.totalWords}u;}
fn ${p}Hash(value:u32,hash:u32)->u32{return (hash^value)*0x01000193u;}
fn ${p}Fail(code:u32,id:u32){if(atomicCompareExchangeWeak(&${arena}[${h(H.fault)}],0u,code).exchanged){
 atomicStore(&${arena}[${h(H.firstFaultId)}],id);}atomicStore(&${arena}[${h(H.phase)}],${PHASE.fault}u);}
@compute @workgroup_size(1) fn beginSparseCM12TopologyEffectsPreflight(){
 let generation=atomicLoad(&${arena}[${h(H.generation)}])+1u;
 if(!${p}HeaderValid()||generation==0u||generation>=${p}GenerationBusy){${p}Fail(1u,${p}Invalid);return;}
 atomicStore(&${arena}[${h(H.generation)}],generation);atomicStore(&${arena}[${h(H.fault)}],0u);
 atomicStore(&${arena}[${h(H.firstFaultId)}],${p}Invalid);
 atomicStore(&${arena}[${h(H.ptrTargetGeneration)}],${p}PTRTargetGeneration());
 atomicStore(&${arena}[${h(H.ptrCount)}],0u);atomicStore(&${arena}[${h(H.ptrLeafCount)}],0u);
 atomicStore(&${arena}[${h(H.ptrNewCount)}],0u);atomicStore(&${arena}[${h(H.ptrNewLeafCount)}],0u);
 atomicStore(&${arena}[${h(H.coveredEffects)}],0u);atomicStore(&${arena}[${h(H.phase)}],${PHASE.recording}u);}
fn ${p}RecordPTRBrick(brick:u32,oldState:u32,newState:u32,cause:u32)->bool{
 if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.recording}u||brick>=${p}PtrCapacity||cause==0u){
  ${p}Fail(5u,brick);return false;}let generation=atomicLoad(&${arena}[${h(H.generation)}]);
 for(var attempt=0u;attempt<64u;attempt+=1u){let old=atomicLoad(&${arena}[${l.ptrStampBaseWords}u+brick]);
  if(old==generation){if(atomicLoad(&${arena}[${l.ptrOldStateBaseWords}u+brick])!=oldState
    ||atomicLoad(&${arena}[${l.ptrNewStateBaseWords}u+brick])!=newState){${p}Fail(6u,brick);return false;}
    atomicOr(&${arena}[${l.ptrCauseBaseWords}u+brick],cause);return true;}
  if((old&${p}GenerationBusy)==0u){let claim=atomicCompareExchangeWeak(&${arena}[
    ${l.ptrStampBaseWords}u+brick],old,generation|${p}GenerationBusy);if(claim.exchanged){
    atomicStore(&${arena}[${l.ptrOldStateBaseWords}u+brick],oldState);
    atomicStore(&${arena}[${l.ptrNewStateBaseWords}u+brick],newState);
    atomicStore(&${arena}[${l.ptrCauseBaseWords}u+brick],cause);
    let leaf=brick/256u;let owns=atomicExchange(&${arena}[${l.ptrLeafStampBaseWords}u+leaf],generation)!=generation;
    atomicStore(&${arena}[${l.ptrOwnsLeafBaseWords}u+brick],select(0u,1u,owns));
    if(owns){let leafSlot=atomicAdd(&${arena}[${h(H.ptrLeafCount)}],1u);
      if(leafSlot>=${p}PtrLeafCapacity){${p}Fail(7u,leaf);return false;}
      atomicStore(&${arena}[${l.ptrLeafListBaseWords}u+leafSlot],leaf);}
    let slot=atomicAdd(&${arena}[${h(H.ptrCount)}],1u);if(slot>=${p}PtrCapacity){${p}Fail(8u,brick);return false;}
    atomicStore(&${arena}[${l.ptrListBaseWords}u+slot],brick);
    atomicStore(&${arena}[${l.ptrStampBaseWords}u+brick],generation);return true;}}}
 ${p}Fail(9u,brick);return false;}
@compute @workgroup_size(1) fn finalizeSparseCM12TopologyEffectsPreflight(){
 if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.recording}u||atomicLoad(&${arena}[${h(H.fault)}])!=0u){return;}
 let ptrGeneration=atomicLoad(&${arena}[${h(H.ptrTargetGeneration)}]);
 let ptrCount=atomicLoad(&${arena}[${h(H.ptrCount)}]);let leafCount=atomicLoad(&${arena}[${h(H.ptrLeafCount)}]);
 var ptrNew=0u;var ptrHash=0u;for(var rank=0u;rank<ptrCount;rank+=1u){
  let brick=atomicLoad(&${arena}[${l.ptrListBaseWords}u+rank]);let oldState=atomicLoad(&${arena}[${l.ptrOldStateBaseWords}u+brick]);
  let newState=atomicLoad(&${arena}[${l.ptrNewStateBaseWords}u+brick]);let cause=atomicLoad(&${arena}[${l.ptrCauseBaseWords}u+brick]);
  if(!${p}PTRCompatible(brick,oldState,newState)){${p}Fail(10u,brick);return;}
  ptrNew+=select(0u,1u,${p}PTRWillAppend(brick,ptrGeneration));
  ptrHash^=${p}Hash(cause,${p}Hash(newState,${p}Hash(oldState,
    ${p}Hash(brick,0x811c9dc5u))));}
 var leafNew=0u;for(var rank=0u;rank<leafCount;rank+=1u){let leaf=atomicLoad(&${arena}[
   ${l.ptrLeafListBaseWords}u+rank]);leafNew+=select(0u,1u,${p}PTRDirtyLeafWillAppend(leaf,ptrGeneration));}
 atomicStore(&${arena}[${h(H.ptrNewCount)}],ptrNew);
 atomicStore(&${arena}[${h(H.ptrNewLeafCount)}],leafNew);
 if(!${p}PTRReady(ptrGeneration,ptrNew,leafNew)){${p}Fail(11u,${p}Invalid);return;}
 atomicStore(&${arena}[${h(H.ptrHash)}],ptrHash);atomicStore(&${arena}[${h(H.expectedEffects)}],ptrCount);
 atomicStore(&${arena}[${h(H.ptrDispatchX)}],(ptrCount+63u)/64u);
 atomicStore(&${arena}[${h(H.phase)}],${PHASE.preflighted}u);}
fn ${p}PreflightReady(generation:u32,ptrGeneration:u32,ptrCount:u32,ptrHash:u32)->bool{
 return atomicLoad(&${arena}[${h(H.phase)}])==${PHASE.preflighted}u
  &&atomicLoad(&${arena}[${h(H.fault)}])==0u&&atomicLoad(&${arena}[${h(H.generation)}])==generation
  &&atomicLoad(&${arena}[${h(H.ptrTargetGeneration)}])==ptrGeneration
  &&atomicLoad(&${arena}[${h(H.ptrCount)}])==ptrCount&&atomicLoad(&${arena}[${h(H.ptrHash)}])==ptrHash;}
fn ${p}Authorize(){atomicStore(&${arena}[${h(H.phase)}],${PHASE.authorized}u);}
@compute @workgroup_size(64) fn publishSparseCM12TopologyPTREffects(
 @builtin(global_invocation_id)gid:vec3u){if(!(${a})||atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.authorized}u){return;}
 let rank=gid.x;let count=atomicLoad(&${arena}[${h(H.ptrCount)}]);if(rank>=count){return;}
 let brick=atomicLoad(&${arena}[${l.ptrListBaseWords}u+rank]);
 let generation=atomicLoad(&${arena}[${h(H.ptrTargetGeneration)}]);
 ${p}PTRPublish(brick,atomicLoad(&${arena}[${l.ptrOldStateBaseWords}u+brick]),
  atomicLoad(&${arena}[${l.ptrNewStateBaseWords}u+brick]),atomicLoad(&${arena}[${l.ptrCauseBaseWords}u+brick]),
  atomicLoad(&${arena}[${l.ptrOwnsLeafBaseWords}u+brick])!=0u,generation);
 atomicAdd(&${arena}[${h(H.coveredEffects)}],1u);}
@compute @workgroup_size(1) fn finishSparseCM12TopologyEffectsPublication(){
 if(atomicLoad(&${arena}[${h(H.phase)}])!=${PHASE.authorized}u){return;}
 atomicStore(&${arena}[${h(H.phase)}],${PHASE.published}u);}
fn ${p}Published()->bool{return atomicLoad(&${arena}[${h(H.phase)}])==${PHASE.published}u;}
`;
}
