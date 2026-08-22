const identifier = (value: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) throw new TypeError("ISA1 invalid prefix");
  return value;
};

export function createSparseCM12GeometryFaceNeighborsWGSL(options: Readonly<{
  baseWords: number; leafCapacity: number; offsetBaseWords: number;
  neighborBaseWords: number; arenaName?: string; hookPrefix?: string;
}>): string {
  const arena = identifier(options.arenaName ?? "topologyArena");
  const p = identifier(options.hookPrefix ?? "cm12");
  return /* wgsl */ `
const IGN1_BASE:u32=${options.baseWords}u;const IGN1_LEAF_CAPACITY:u32=${options.leafCapacity}u;
const IGN1_OFFSETS:u32=${options.baseWords + options.offsetBaseWords}u;
const IGN1_NEIGHBORS:u32=${options.baseWords + options.neighborBaseWords}u;
fn ${p}ISAGeometryNeighborCount(leaf:u32)->u32{
  if(leaf>=IGN1_LEAF_CAPACITY){return 0u;}return atomicLoad(&${arena}[IGN1_OFFSETS+leaf+1u])
    -atomicLoad(&${arena}[IGN1_OFFSETS+leaf]);}
fn ${p}ISAGeometryNeighbor(leaf:u32,local:u32)->u32{
  if(leaf>=IGN1_LEAF_CAPACITY){return 0xffffffffu;}
  let begin=atomicLoad(&${arena}[IGN1_OFFSETS+leaf]);
  return atomicLoad(&${arena}[IGN1_NEIGHBORS+begin+local]);}
`;
}

/**
 * Binding-free semantic authority. Required prefix hooks expose SCMT row words,
 * candidate-face ranges, scheduled-row membership, and immutable geometry CSR.
 * IBO accessors come from IBO1. Neither path calls an IRL accessor.
 */
export function createSparseCM12IBOSemanticAuthorityWGSL(options: Readonly<{
  hookPrefix?: string; iboPrefix?: string;
}> = {}): string {
  const p = identifier(options.hookPrefix ?? "cm12");
  const ibo = identifier(options.iboPrefix ?? "cm12");
  return /* wgsl */ `
fn ${p}ISAFNV(hash:u32,value:u32)->u32{return (hash^value)*0x01000193u;}
fn ${p}ISAMixRow(hash:u32,row:u32)->u32{
  return ${p}ISAFNV(${p}ISAFNV(0x9e3779b9u,hash),row);}
fn ${p}ISAFold(acc:vec3u,rowHash:u32,row:u32)->vec3u{
  let mixed=${p}ISAMixRow(rowHash,row);
  return vec3u(acc.x+1u,acc.y^mixed,acc.z+mixed*0x85ebca6bu);}
fn ${p}ISASCMTStableRowHash(row:u32)->u32{
  var hash=0x811c9dc5u;hash=${p}ISAFNV(hash,row);
  hash=${p}ISAFNV(hash,${p}ISARowWord(row,1u)&0xf0000000u);
  for(var plane=2u;plane<=5u;plane+=1u){hash=${p}ISAFNV(hash,${p}ISARowWord(row,plane));}
  let count=${p}ISARowTermCount(row);hash=${p}ISAFNV(hash,count);
  let first=${p}ISARowTermFirst(row);
  for(var local=0u;local<count;local+=1u){hash=${p}ISAFNV(
    ${p}ISAFNV(hash,${p}ISATermCell(first+local)),${p}ISATermBits(first+local));}
  return hash;}
fn ${p}ISASCMTLeafSemantics(descriptor:u32)->vec3u{
  var result=vec3u(0u);let resolution=${ibo}IBOCanonicalWord(descriptor,2u);
  for(var side=0u;side<6u;side+=1u){for(var boundary=0u;
    boundary<resolution*resolution;boundary+=1u){
    let range=${p}ISACandidateFaceRange(descriptor,side,boundary);
    for(var local=0u;local<range.y;local+=1u){let row=${p}ISACandidateFaceRow(range.x+local);
      if(${p}ISAScheduledRow(row)){result=${p}ISAFold(result,
        ${p}ISASCMTStableRowHash(row),row);}}}}
  return result;}
fn ${p}ISAIBOTemplateRowHash(slot:u32,leaf:u32,reference:vec3u,
  localRow:u32)->vec2u{
  let source=${ibo}IBOLeafDescriptorId(slot,leaf);
  let targetDescriptor=select(IBO1_INVALID,${ibo}IBOLeafDescriptorId(slot,reference.y),
    reference.y!=IBO1_INVALID);
  let row=reference.z+${ibo}IBOTemplateRowWord(reference.x,localRow,0u);
  let packed=${ibo}IBOTemplateRowWord(reference.x,localRow,1u);
  let first=packed&0x007fffffu;let count=packed>>23u;
  var hash=0x811c9dc5u;hash=${p}ISAFNV(hash,row);
  for(var word=2u;word<7u;word+=1u){hash=${p}ISAFNV(hash,
    ${ibo}IBOTemplateRowWord(reference.x,localRow,word));}
  hash=${p}ISAFNV(hash,count);
  for(var local=0u;local<count;local+=1u){let normalized=
      ${ibo}IBOTemplateTermWord(reference.x,first+local,0u);
    let targetDomain=(normalized&0x80000000u)!=0u;
    let descriptor=select(source,targetDescriptor,targetDomain);
    let cell=${ibo}IBOCanonicalWord(descriptor,3u)+(normalized&0x7fffffffu);
    hash=${p}ISAFNV(${p}ISAFNV(hash,cell),
      ${ibo}IBOTemplateTermWord(reference.x,first+local,1u));}
  return vec2u(row,hash);}
fn ${p}ISAIBOLeafSemantics(slot:u32,leaf:u32)->vec3u{
  var result=vec3u(0u);for(var side=0u;side<6u;side+=1u){
    let count=${ibo}IBOFaceRefCount(slot,leaf,side);
    for(var local=0u;local<count;local+=1u){let reference=${ibo}IBORef(slot,leaf,side,local);
      let rows=${ibo}IBOTemplateHeaderWord(reference.x,6u);
      for(var row=0u;row<rows;row+=1u){let semantic=
        ${p}ISAIBOTemplateRowHash(slot,leaf,reference,row);
        result=${p}ISAFold(result,semantic.y,semantic.x);}}}
  return result;}
fn ${p}ISAValidateLeaf(slot:u32,leaf:u32,expectedDescriptor:u32)->vec2u{
  if(${ibo}IBOLeafDescriptorId(slot,leaf)!=expectedDescriptor){return vec2u(3u,leaf);}
  let expected=${p}ISASCMTLeafSemantics(expectedDescriptor);
  let observed=${p}ISAIBOLeafSemantics(slot,leaf);
  return select(vec2u(5u,leaf),vec2u(0u,0xffffffffu),all(expected==observed));}
// Main topology owns stamps/list/count/hash. This helper adds exactly changed+
// immutable geometry face-neighbors; claim/append hooks dedupe without a scan.
fn ${p}ISAAppendGeometryClosure(changedLeaf:u32,generation:u32){
  if(${p}ISAClaimClosureLeaf(changedLeaf,generation)){${p}ISAAppendClosureLeaf(changedLeaf);}
  let count=${p}ISAGeometryNeighborCount(changedLeaf);
  for(var local=0u;local<count;local+=1u){let leaf=${p}ISAGeometryNeighbor(changedLeaf,local);
    if(${p}ISAClaimClosureLeaf(leaf,generation)){${p}ISAAppendClosureLeaf(leaf);}}}
`;
}
