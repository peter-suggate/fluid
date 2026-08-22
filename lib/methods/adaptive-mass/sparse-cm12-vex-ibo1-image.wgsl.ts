import {
  SPARSE_CM12_VEX_IBO1_IMAGE_HEADER,
  SPARSE_CM12_VEX_IBO1_IMAGE_MAGIC,
  SPARSE_CM12_VEX_IBO1_IMAGE_VERSION,
  type SparseCM12VexIBO1ImageLayout,
} from "./sparse-cm12-vex-ibo1-image";

export interface SparseCM12VexIBO1ImageWGSLOptions {
  readonly layout: SparseCM12VexIBO1ImageLayout;
  readonly baseWords?: number;
  readonly arenaName?: string;
  /** Prefix of the immutable/slot IBO1 accessor contract, default `cm12`. */
  readonly iboPrefix?: string;
  /** Dynamic exact packet-root batch accessors, with `(rank)` and `(packet)`. */
  readonly sourcePacketCountFunction: string;
  readonly sourcePacketFunction: string;
  readonly sourceMaskFunction: string;
  readonly rootCauseExpression: string;
  readonly entryPointPrefix?: string;
}

const identifier = (value: string, label: string) => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be an identifier`);
  }
  return value;
};
const expression = (value: string, label: string) => {
  if (value.trim().length === 0) throw new TypeError(`${label} must not be empty`);
  return value;
};

/**
 * Exact bounded IBO1 local-operator bridge for VXP1.
 *
 * The root entry consumes only an already compact exact source packet list.
 * The recurrence entry consumes VXP1's compact prior frontier. Both resolve a
 * canonical core recipe plus at most 6*4 selected IBO face refs. They never
 * traverse cells, authoritative rows, or a global incidence structure.
 */
export function createSparseCM12VexIBO1ImageWGSL(
  options: SparseCM12VexIBO1ImageWGSLOptions,
): string {
  const layout = options.layout, base = options.baseWords ?? 0;
  if (!Number.isSafeInteger(base) || base < 0) throw new RangeError("baseWords is invalid");
  const arena = identifier(options.arenaName ?? "topologyArena", "arenaName");
  const ibo = identifier(options.iboPrefix ?? "cm12", "iboPrefix");
  const count = identifier(options.sourcePacketCountFunction, "sourcePacketCountFunction");
  const packet = identifier(options.sourcePacketFunction, "sourcePacketFunction");
  const mask = identifier(options.sourceMaskFunction, "sourceMaskFunction");
  const prefix = identifier(options.entryPointPrefix ?? "vxi1", "entryPointPrefix");
  const cause = expression(options.rootCauseExpression, "rootCauseExpression");
  const at = (word: number) => `${base + word}u`;
  const h = SPARSE_CM12_VEX_IBO1_IMAGE_HEADER;
  return /* wgsl */ `
const ${prefix}Magic:u32=0x${SPARSE_CM12_VEX_IBO1_IMAGE_MAGIC.toString(16)}u;
const ${prefix}Version:u32=${SPARSE_CM12_VEX_IBO1_IMAGE_VERSION}u;
const ${prefix}Base:u32=${base}u;
const ${prefix}CanonicalTemplate:u32=${at(layout.canonicalTemplateBaseWords)};
const ${prefix}Directory:u32=${at(layout.directoryBaseWords)};
const ${prefix}Operation:u32=${at(layout.operationBaseWords)};
const ${prefix}CanonicalCapacity:u32=${layout.canonicalCapacity}u;
const ${prefix}CoreTemplateCount:u32=${layout.coreTemplateCount}u;
const ${prefix}TemplateCount:u32=${layout.templateCount}u;
var<workgroup> ${prefix}SourcePacket:array<u32,4>;
var<workgroup> ${prefix}SourceLow:array<u32,4>;
var<workgroup> ${prefix}SourceHigh:array<u32,4>;
var<workgroup> ${prefix}SourceLeaf:array<u32,4>;
var<workgroup> ${prefix}SourceCore:array<u32,4>;
var<workgroup> ${prefix}SourceValid:array<u32,4>;
var<workgroup> ${prefix}RefTemplate:array<u32,96>;
var<workgroup> ${prefix}RefTarget:array<u32,96>;

fn ${prefix}Load(word:u32)->u32{return atomicLoad(&${arena}[word]);}
fn ${prefix}Valid()->bool{return ${prefix}Load(${at(h.magic)})==${prefix}Magic
  &&${prefix}Load(${at(h.version)})==${prefix}Version
  &&${prefix}Load(${at(h.totalWords)})==${layout.totalWords}u
  &&${prefix}Load(${at(h.operationCount)})==${layout.operationCount}u;}
fn ${prefix}Shift(mask:vec2u,delta:i32)->vec2u{
  if(delta==0){return mask;}
  if(delta>0){let d=u32(delta);if(d<32u){return vec2u(mask.x<<d,
      (mask.y<<d)|(mask.x>>(32u-d)));}
    return vec2u(0u,mask.x<<(d-32u));}
  let d=u32(-delta);if(d<32u){return vec2u((mask.x>>d)|(mask.y<<(32u-d)),mask.y>>d);}
  return vec2u(mask.y>>(d-32u),0u);
}
fn ${prefix}ApplyTemplate(recipe:u32,sourcePacket:u32,sourceMask:vec2u,
    sourceLeaf:u32,targetLeaf:u32,worker16:u32,root:bool,cause:u32){
  if(recipe>=${prefix}TemplateCount||(sourceMask.x|sourceMask.y)==0u){return;}
  let localPacket=sourcePacket&63u;
  let directory=${prefix}Load(${prefix}Directory+64u*recipe+localPacket);
  let first=directory&0x000fffffu;let operationCount=directory>>20u;
  for(var local=worker16;local<operationCount;local+=16u){
    let operation=${prefix}Operation+3u*(first+local);
    let encoded=${prefix}Load(operation);let domain=encoded>>31u;
    let targetLocalPacket=encoded&63u;
    let targetKey=encoded&0x8000003fu;
    if(local>0u&&(${prefix}Load(operation-3u)&0x8000003fu)==targetKey){continue;}
    var shifted=vec2u(0u);var grouped=local;
    loop{let groupedAt=${prefix}Operation+3u*(first+grouped);
      let groupedEncoded=${prefix}Load(groupedAt);
      if((groupedEncoded&0x8000003fu)!=targetKey){break;}
      let groupedDelta=i32((groupedEncoded>>6u)&127u)-63;
      let selected=sourceMask&vec2u(${prefix}Load(groupedAt+1u),${prefix}Load(groupedAt+2u));
      shifted|=${prefix}Shift(selected,groupedDelta);grouped+=1u;
      if(grouped>=operationCount){break;}}
    if((shifted.x|shifted.y)==0u){continue;}
    let leaf=select(sourceLeaf,targetLeaf,domain==1u);
    if(leaf==0xffffffffu){vxp1Fault(sourcePacket,0xffffffffu);continue;}
    let targetPacket=leaf*64u+targetLocalPacket;
    if(root){_ = vxp1RecordRootPacketMask(targetPacket,shifted.x,shifted.y,cause);
    }else{_ = vxp1MergeFrontierTarget(targetPacket,shifted.x,shifted.y);}
  }
}
fn ${prefix}CacheSelected(sub:u32,worker16:u32){
  workgroupBarrier();
  if(worker16==0u){
    let sourcePacket=${prefix}SourcePacket[sub];let sourceMask=vec2u(
      ${prefix}SourceLow[sub],${prefix}SourceHigh[sub]);let sourceLeaf=sourcePacket/64u;
    ${prefix}SourceLeaf[sub]=sourceLeaf;${prefix}SourceCore[sub]=0xffffffffu;
    ${prefix}SourceValid[sub]=0u;
    if((sourceMask.x|sourceMask.y)!=0u){let slot=${ibo}IBOAcceptedSlot();
      let generation=${ibo}IBOAcceptedGeneration();
      if(!${prefix}Valid()||slot!=vxp1TopologySlot()
        ||!${ibo}IBOLeafActive(slot,sourceLeaf)
        ||${ibo}IBOLeafGeneration(slot,sourceLeaf)!=generation){
        vxp1Fault(sourcePacket,0xffffffffu);
      }else{let canonical=${ibo}IBOLeafDescriptorId(slot,sourceLeaf);
        if(canonical>=${prefix}CanonicalCapacity){vxp1Fault(sourcePacket,0xffffffffu);
        }else{let core=${prefix}Load(${prefix}CanonicalTemplate+canonical);
          if(core==0xffffffffu){vxp1Fault(sourcePacket,0xffffffffu);
          }else{${prefix}SourceCore[sub]=core;${prefix}SourceValid[sub]=1u;}}}}
  }
  workgroupBarrier();
  for(var flat=worker16;flat<24u;flat+=16u){let at=24u*sub+flat;
    ${prefix}RefTemplate[at]=0xffffffffu;${prefix}RefTarget[at]=0xffffffffu;
    if(${prefix}SourceValid[sub]!=0u){let side=flat>>2u;let localRef=flat&3u;
      let slot=${ibo}IBOAcceptedSlot();let sourceLeaf=${prefix}SourceLeaf[sub];
      let refCount=${ibo}IBOFaceRefCount(slot,sourceLeaf,side);
      if(refCount>4u){vxp1Fault(${prefix}SourcePacket[sub],0xffffffffu);
      }else if(localRef<refCount){let faceRef=${ibo}IBORef(slot,sourceLeaf,side,localRef);
        ${prefix}RefTemplate[at]=faceRef.x;${prefix}RefTarget[at]=faceRef.y;}}
  }
  workgroupBarrier();
}
fn ${prefix}ApplySelected(sub:u32,worker16:u32,root:bool,cause:u32){
  if(${prefix}SourceValid[sub]==0u){return;}
  let sourcePacket=${prefix}SourcePacket[sub];let sourceMask=vec2u(
    ${prefix}SourceLow[sub],${prefix}SourceHigh[sub]);
  let sourceLeaf=${prefix}SourceLeaf[sub];
  ${prefix}ApplyTemplate(${prefix}SourceCore[sub],sourcePacket,sourceMask,sourceLeaf,sourceLeaf,
    worker16,root,cause);
  for(var flat=0u;flat<24u;flat+=1u){let at=24u*sub+flat;
    let faceTemplate=${prefix}RefTemplate[at];if(faceTemplate==0xffffffffu){continue;}
    ${prefix}ApplyTemplate(${prefix}CoreTemplateCount+faceTemplate,sourcePacket,sourceMask,
      sourceLeaf,${prefix}RefTarget[at],worker16,root,cause);
  }
}

@compute @workgroup_size(64) fn ${prefix}CompileDepthZeroRoots(
    @builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let sub=lane>>4u;let worker16=lane&15u;let rank=4u*wid.x+sub;
  if(worker16==0u){${prefix}SourcePacket[sub]=0xffffffffu;
    ${prefix}SourceLow[sub]=0u;${prefix}SourceHigh[sub]=0u;
    if(rank<${count}()){let sourcePacket=${packet}(rank);let sourceMask=${mask}(sourcePacket);
      ${prefix}SourcePacket[sub]=sourcePacket;${prefix}SourceLow[sub]=sourceMask.x;
      ${prefix}SourceHigh[sub]=sourceMask.y;}}
  ${prefix}CacheSelected(sub,worker16);
  ${prefix}ApplySelected(sub,worker16,true,${cause});
}

// Specialize VXP1_FRONTIER_DEPTH to 1..8. Preparation/finalization remain the
// existing bounded three-dispatch-per-depth VXP1 chronology.
@compute @workgroup_size(64) fn ${prefix}ExpandFrontier(
    @builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lane:u32){
  let sub=lane>>4u;let worker16=lane&15u;let rank=4u*wid.x+sub;
  let sourceDepth=VXP1_FRONTIER_DEPTH-1u;
  if(worker16==0u){${prefix}SourcePacket[sub]=0xffffffffu;
    ${prefix}SourceLow[sub]=0u;${prefix}SourceHigh[sub]=0u;
    if(rank<vxp1FrontierPacketCount(sourceDepth)){
      let sourcePacket=vxp1FrontierPacket(sourceDepth,rank);
      let sourceMask=vxp1FrontierMask(sourceDepth,sourcePacket);
      ${prefix}SourcePacket[sub]=sourcePacket;${prefix}SourceLow[sub]=sourceMask.x;
      ${prefix}SourceHigh[sub]=sourceMask.y;}}
  ${prefix}CacheSelected(sub,worker16);
  ${prefix}ApplySelected(sub,worker16,false,0u);
}
`;
}
