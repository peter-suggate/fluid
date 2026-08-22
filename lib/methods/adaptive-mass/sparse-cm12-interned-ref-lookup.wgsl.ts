import type { SparseCM12InternedRefLookupLayout } from
  "./sparse-cm12-interned-ref-lookup";

const identifier = (value: string, label: string) => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Bounded GPU lookup from a scheduled canonical face to its exact IBO ref. */
export function createSparseCM12InternedRefLookupWGSL(options: Readonly<{
  layout: SparseCM12InternedRefLookupLayout;
  arenaName?: string;
  iboPrefix?: string;
  /** Absolute shared-arena base of the composed IBO image. */
  baseWords?: number;
}>): string {
  const { layout: l } = options;
  const arena = identifier(options.arenaName ?? "topologyArena", "IRL1 arenaName");
  const ibo = identifier(options.iboPrefix ?? "cm12", "IRL1 iboPrefix");
  const base = options.baseWords ?? 0;
  if (!Number.isSafeInteger(base) || base < 0 || base + l.totalWords > 0xffff_ffff) {
    throw new RangeError("IRL1 relocation exceeds u32 word addressing");
  }
  return /* wgsl */ `
const IRL1_INVALID:u32=0xffffffffu;
const IRL1_INVALID16:u32=0xffffu;
const IRL1_EXTERIOR_DELTA16:u32=0x8000u;
const IRL1_CANONICAL_CAPACITY:u32=${l.canonicalCapacity}u;
const IRL1_TEMPLATE_COUNT:u32=${l.templateCount}u;
const IRL1_TEMPLATE_ENTRY_COUNT:u32=${l.templateEntryCount}u;
const IRL1_MAX_PER_SIDE:u32=${l.maximumEntriesPerSide}u;
const IRL1_DIRECTORY:u32=${base + l.directoryBaseWords}u;
const IRL1_TEMPLATE_DIRECTORY:u32=${base + l.templateDirectoryBaseWords}u;
const IRL1_ENTRIES:u32=${base + l.entryBaseWords}u;
const IRL1_FALLBACK_ANCHORS:u32=${base + l.fallbackAnchorBaseWords}u;
fn irl1Load(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn irl1U16(base:u32,index:u32)->u32{
  let packed=irl1Load(base+(index>>1u));
  return (packed>>(16u*(index&1u)))&0xffffu;}
fn ${ibo}IBOInstantiationCount(sourceCanonical:u32,side:u32)->u32{
  if(sourceCanonical>=IRL1_CANONICAL_CAPACITY||side>=6u){return 0u;}
  let group=sourceCanonical*6u+side;
  let templateId=irl1U16(IRL1_DIRECTORY,group);
  if(templateId>=IRL1_TEMPLATE_COUNT){return 0u;}
  let begin=irl1U16(IRL1_TEMPLATE_DIRECTORY,templateId);
  let end=irl1U16(IRL1_TEMPLATE_DIRECTORY,templateId+1u);
  return select(0u,end-begin,end>=begin&&end<=IRL1_TEMPLATE_ENTRY_COUNT
    &&end-begin<=IRL1_MAX_PER_SIDE);}
fn ${ibo}IBOInstantiationEntry(sourceCanonical:u32,side:u32,local:u32)->vec3u{
  let count=${ibo}IBOInstantiationCount(sourceCanonical,side);
  if(local>=count){return vec3u(IRL1_INVALID);}
  let group=sourceCanonical*6u+side;
  let templateId=irl1U16(IRL1_DIRECTORY,group);
  let entry=irl1U16(IRL1_TEMPLATE_DIRECTORY,templateId)+local;
  let at=IRL1_ENTRIES+2u*entry;let packed=irl1Load(at);
  let encodedDelta=packed&0xffffu;var targetCanonical=IRL1_INVALID;
  if(encodedDelta!=IRL1_EXTERIOR_DELTA16){
    let delta=bitcast<i32>(encodedDelta<<16u)>>16;
    let candidate=i32(sourceCanonical)+delta;
    if(candidate<0||candidate>=i32(IRL1_CANONICAL_CAPACITY)){return vec3u(IRL1_INVALID);}
    targetCanonical=u32(candidate);}
  var canonicalAnchor=${ibo}IBOCanonicalRowBase(sourceCanonical,side>>1u);
  for(var axis=0u;canonicalAnchor==IRL1_INVALID&&axis<3u;axis+=1u){
    canonicalAnchor=${ibo}IBOCanonicalRowBase(sourceCanonical,axis);}
  let anchor=select(canonicalAnchor,irl1Load(IRL1_FALLBACK_ANCHORS+sourceCanonical),
    canonicalAnchor==IRL1_INVALID);
  return vec3u(targetCanonical,packed>>16u,anchor+irl1Load(at+1u));}
fn ${ibo}IBOFindScheduledRef(sourceCanonical:u32,side:u32,
  targetCanonical:u32,targetLeaf:u32)->vec3u{
  if(targetCanonical!=IRL1_INVALID&&(targetCanonical>=IRL1_CANONICAL_CAPACITY
    ||${ibo}IBOCanonicalWord(targetCanonical,1u)!=targetLeaf)){
    return vec3u(IRL1_INVALID);}
  if(targetCanonical==IRL1_INVALID&&targetLeaf!=IRL1_INVALID){return vec3u(IRL1_INVALID);}
  let count=${ibo}IBOInstantiationCount(sourceCanonical,side);
  for(var local=0u;local<count;local+=1u){
    let entry=${ibo}IBOInstantiationEntry(sourceCanonical,side,local);
    if(entry.x==targetCanonical){return vec3u(entry.y,targetLeaf,entry.z);}}
  return vec3u(IRL1_INVALID);}
`;
}
