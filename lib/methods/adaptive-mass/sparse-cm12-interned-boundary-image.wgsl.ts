import {
  SPARSE_CM12_INTERNED_BOUNDARY_FAULT,
  SPARSE_CM12_INTERNED_BOUNDARY_VERSION,
} from "./sparse-cm12-interned-boundary-image";
import {
  SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_MAGIC,
  SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF,
  SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_BITS,
  SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK,
  SPARSE_CM12_INTERNED_BOUNDARY_REF_TEMPLATE_MAXIMUM,
  SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS,
  SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS,
  type SparseCM12InternedBoundaryLayout,
} from "./sparse-cm12-interned-boundary-operators";
import { SPARSE_CM12_FACTORED_AEI_INVALID } from
  "./sparse-cm12-factored-aei-topology";

const identifier = (value: string, label: string): string => {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
    throw new TypeError(`${label} must be a WGSL identifier`);
  }
  return value;
};

/** Binding-free IBO1 image accessors and bounded transaction primitives. */
export function createSparseCM12InternedBoundaryImageWGSL(options: Readonly<{
  layout: SparseCM12InternedBoundaryLayout;
  arenaName?: string;
  hookPrefix?: string;
  baseWords?: number;
  packetsPerLeaf: number;
  /** Existing TEI/topology selector hook; IBO never authors a second selector. */
  acceptedSlotHook: string;
  /** Existing accepted-generation hook paired with the shared selector. */
  acceptedGenerationHook: string;
}>): string {
  const { layout: l } = options;
  const arena = identifier(options.arenaName ?? "topologyArena", "IBO1 arenaName");
  const p = identifier(options.hookPrefix ?? "cm12", "IBO1 hookPrefix");
  const acceptedSlotHook = identifier(options.acceptedSlotHook,
    "IBO1 acceptedSlotHook");
  const acceptedGenerationHook = identifier(options.acceptedGenerationHook,
    "IBO1 acceptedGenerationHook");
  const base = options.baseWords ?? 0;
  if (!Number.isSafeInteger(base) || base < 0
    || base + l.totalWords > 0xffff_ffff) {
    throw new RangeError("IBO1 relocated image must fit u32 word addressing");
  }
  if (!Number.isSafeInteger(options.packetsPerLeaf)
    || options.packetsPerLeaf < 1 || options.packetsPerLeaf > 0xffff_ffff) {
    throw new RangeError("IBO1 packetsPerLeaf must be a positive u32");
  }
  return /* wgsl */ `
const IBO1_MAGIC:u32=0x${SPARSE_CM12_INTERNED_BOUNDARY_MAGIC.toString(16)}u;
const IBO1_VERSION:u32=${SPARSE_CM12_INTERNED_BOUNDARY_VERSION}u;
const IBO1_INVALID:u32=${SPARSE_CM12_FACTORED_AEI_INVALID}u;
const IBO1_BASE:u32=${base}u;
const IBO1_TOTAL_WORDS:u32=${l.totalWords}u;
const IBO1_LEAF_CAPACITY:u32=${l.leafCapacity}u;
const IBO1_CANONICAL_CAPACITY:u32=${l.canonicalCapacity}u;
const IBO1_TEMPLATE_COUNT:u32=${l.templateCount}u;
const IBO1_CANONICAL_BASE:u32=${base + l.canonicalBaseWords}u;
const IBO1_TEMPLATE_DIRECTORY_BASE:u32=${base + l.templateDirectoryBaseWords}u;
const IBO1_TEMPLATE_PAYLOAD_BASE:u32=${base + l.templatePayloadBaseWords}u;
const IBO1_IMMUTABLE_END:u32=${base + l.immutableWords}u;
const IBO1_SLOT0:u32=${base + l.slotBaseWords[0]}u;
const IBO1_SLOT1:u32=${base + l.slotBaseWords[1]}u;
const IBO1_SLOT0_LEAVES:u32=${base + l.slotLeafBaseWords[0]}u;
const IBO1_SLOT1_LEAVES:u32=${base + l.slotLeafBaseWords[1]}u;
const IBO1_SLOT0_REFS:u32=${base + l.slotRefBaseWords[0]}u;
const IBO1_SLOT1_REFS:u32=${base + l.slotRefBaseWords[1]}u;
const IBO1_SLOT_LEAF_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_SLOT_LEAF_WORDS}u;
const IBO1_SLOT_REF_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_SLOT_REF_WORDS}u;
const IBO1_REFS_PER_LEAF:u32=${SPARSE_CM12_INTERNED_BOUNDARY_REFS_PER_LEAF}u;
const IBO1_REFS_PER_FACE:u32=4u;
const IBO1_REF_TARGET_BITS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_BITS}u;
const IBO1_REF_TARGET_MASK:u32=0x${SPARSE_CM12_INTERNED_BOUNDARY_REF_TARGET_MASK.toString(16)}u;
const IBO1_REF_TEMPLATE_MAXIMUM:u32=${SPARSE_CM12_INTERNED_BOUNDARY_REF_TEMPLATE_MAXIMUM}u;
const IBO1_DIRECTORY_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_DIRECTORY_WORDS}u;
const IBO1_TEMPLATE_HEADER_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_TEMPLATE_HEADER_WORDS}u;
const IBO1_ROW_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_ROW_WORDS}u;
const IBO1_TERM_WORDS:u32=${SPARSE_CM12_INTERNED_BOUNDARY_TERM_WORDS}u;
const IBO1_PACKETS_PER_LEAF:u32=${options.packetsPerLeaf}u;
const IBO1_FAULT_GENERATION:u32=${SPARSE_CM12_INTERNED_BOUNDARY_FAULT.generation}u;
const IBO1_FAULT_TOPOLOGY:u32=${SPARSE_CM12_INTERNED_BOUNDARY_FAULT.topologyMismatch}u;
const IBO1_FAULT_CANONICAL:u32=${SPARSE_CM12_INTERNED_BOUNDARY_FAULT.canonicalCertificate}u;
const IBO1_FAULT_REFERENCE:u32=${SPARSE_CM12_INTERNED_BOUNDARY_FAULT.reference}u;
const IBO1_FAULT_IMMUTABLE:u32=${SPARSE_CM12_INTERNED_BOUNDARY_FAULT.immutableImage}u;

fn ${p}IBOLoad(at:u32)->u32{return atomicLoad(&${arena}[at]);}
fn ${p}IBOStore(at:u32,value:u32){atomicStore(&${arena}[at],value);}
fn ${p}IBOFNV(hash:u32,value:u32)->u32{return (hash^value)*0x01000193u;}
fn ${p}IBOHeaderValid()->bool{return arrayLength(&${arena})>=IBO1_BASE+IBO1_TOTAL_WORDS
  &&${p}IBOLoad(IBO1_BASE)==IBO1_MAGIC&&${p}IBOLoad(IBO1_BASE+1u)==IBO1_VERSION
  &&${p}IBOLoad(IBO1_BASE+4u)==${SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS}u;}
fn ${p}IBOSelectorMirror()->u32{return ${p}IBOLoad(IBO1_BASE+2u)&1u;}
fn ${p}IBOGenerationMirror()->u32{return ${p}IBOLoad(IBO1_BASE+3u);}
fn ${p}IBOImmutableCertificateHash()->u32{return ${p}IBOLoad(IBO1_BASE+29u);}
fn ${p}IBOImmutableCertificate()->vec4u{return vec4u(${p}IBOLoad(IBO1_BASE+25u),
  ${p}IBOLoad(IBO1_BASE+26u),${p}IBOLoad(IBO1_BASE+27u),${p}IBOLoad(IBO1_BASE+28u));}
fn ${p}IBOImmutableHeaderValid(expectedCertificateHash:u32)->bool{
  return ${p}IBOHeaderValid()&&${p}IBOLoad(IBO1_BASE+5u)==IBO1_LEAF_CAPACITY
    &&${p}IBOLoad(IBO1_BASE+6u)==IBO1_CANONICAL_CAPACITY
    &&${p}IBOLoad(IBO1_BASE+7u)==IBO1_TEMPLATE_COUNT
    &&${p}IBOLoad(IBO1_BASE+8u)==IBO1_CANONICAL_BASE-IBO1_BASE
    &&${p}IBOLoad(IBO1_BASE+9u)==IBO1_TEMPLATE_DIRECTORY_BASE-IBO1_BASE
    &&${p}IBOLoad(IBO1_BASE+10u)==IBO1_TEMPLATE_PAYLOAD_BASE-IBO1_BASE
    &&${p}IBOLoad(IBO1_BASE+11u)==IBO1_SLOT0-IBO1_BASE
    &&${p}IBOLoad(IBO1_BASE+12u)==IBO1_SLOT1-IBO1_BASE
    &&${p}IBOLoad(IBO1_BASE+24u)==IBO1_TOTAL_WORDS
    &&${p}IBOImmutableCertificateHash()==expectedCertificateHash;}
// One-time post-upload certificate pass. Main topology retains its result and the
// delta validator remains O(delta); this is never rerun per topology commit.
fn ${p}IBOValidateImmutableContent(expectedContentHash:u32,
  expectedCertificateHash:u32)->vec2u{
  if(!${p}IBOImmutableHeaderValid(expectedCertificateHash)){
    return vec2u(IBO1_FAULT_IMMUTABLE,0u);}
  var hash=0x811c9dc5u;
  for(var at=IBO1_BASE+${SPARSE_CM12_INTERNED_BOUNDARY_HEADER_WORDS}u;
    at<IBO1_IMMUTABLE_END;at+=1u){hash=${p}IBOFNV(hash,${p}IBOLoad(at));}
  if(hash!=expectedContentHash||${p}IBOLoad(IBO1_BASE+28u)!=expectedContentHash){
    return vec2u(IBO1_FAULT_IMMUTABLE,28u);}
  return vec2u(0u,IBO1_INVALID);}
fn ${p}IBOAcceptedSlot()->u32{return ${acceptedSlotHook}()&1u;}
fn ${p}IBOSlotBase(slot:u32)->u32{return select(IBO1_SLOT0,IBO1_SLOT1,(slot&1u)!=0u);}
fn ${p}IBOSlotLeafBase(slot:u32)->u32{return select(
  IBO1_SLOT0_LEAVES,IBO1_SLOT1_LEAVES,(slot&1u)!=0u);}
fn ${p}IBOSlotRefBase(slot:u32)->u32{return select(
  IBO1_SLOT0_REFS,IBO1_SLOT1_REFS,(slot&1u)!=0u);}
fn ${p}IBOAcceptedGeneration()->u32{return ${acceptedGenerationHook}();}
fn ${p}IBOCanonicalWord(descriptorId:u32,word:u32)->u32{
  if(descriptorId>=IBO1_CANONICAL_CAPACITY||word>=16u){return IBO1_INVALID;}
  return ${p}IBOLoad(IBO1_CANONICAL_BASE+16u*descriptorId+word);}
fn ${p}IBOCanonicalRowBase(descriptorId:u32,axis:u32)->u32{
  if(axis>=3u){return IBO1_INVALID;}return ${p}IBOCanonicalWord(descriptorId,6u+axis);}
fn ${p}IBOCanonicalRowCount(descriptorId:u32,axis:u32)->u32{
  if(axis>=3u){return 0u;}return ${p}IBOCanonicalWord(descriptorId,9u+axis);}
fn ${p}IBOLeafWord(slot:u32,leaf:u32,word:u32)->u32{
  if(leaf>=IBO1_LEAF_CAPACITY||word>=IBO1_SLOT_LEAF_WORDS){return IBO1_INVALID;}
  return ${p}IBOLoad(${p}IBOSlotLeafBase(slot)+IBO1_SLOT_LEAF_WORDS*leaf+word);}
fn ${p}IBOLeafGeneration(slot:u32,leaf:u32)->u32{return ${p}IBOLeafWord(slot,leaf,0u);}
fn ${p}IBOLeafActive(slot:u32,leaf:u32)->bool{return (
  ${p}IBOLeafWord(slot,leaf,1u)&1u)!=0u;}
fn ${p}IBOLeafDescriptorId(slot:u32,leaf:u32)->u32{return ${p}IBOLeafWord(slot,leaf,2u);}
fn ${p}IBOLeafCellFirst(slot:u32,leaf:u32)->u32{return ${p}IBOLeafWord(slot,leaf,3u);}
fn ${p}IBOLeafDimensions(slot:u32,leaf:u32)->vec3u{
  let packed=${p}IBOLeafWord(slot,leaf,4u);return vec3u(
    packed&1023u,(packed>>10u)&1023u,(packed>>20u)&1023u);}
fn ${p}IBOFaceRefCount(slot:u32,leaf:u32,side:u32)->u32{
  if(side>=6u){return 0u;}return (${p}IBOLeafWord(slot,leaf,5u)>>(3u*side))&7u;}
fn ${p}IBORef(slot:u32,leaf:u32,side:u32,local:u32)->vec3u{
  if(leaf>=IBO1_LEAF_CAPACITY||side>=6u||local>=IBO1_REFS_PER_FACE){
    return vec3u(IBO1_INVALID);}let refIndex=leaf*IBO1_REFS_PER_LEAF+side*4u+local;
  let at=${p}IBOSlotRefBase(slot)+IBO1_SLOT_REF_WORDS*refIndex;
  let identity=${p}IBOLoad(at);if(identity==IBO1_INVALID){return vec3u(IBO1_INVALID);}
  let targetId=identity&IBO1_REF_TARGET_MASK;
  return vec3u(identity>>IBO1_REF_TARGET_BITS,
    select(targetId,IBO1_INVALID,targetId==IBO1_REF_TARGET_MASK),${p}IBOLoad(at+1u));}
fn ${p}IBOFaceNeighbor(slot:u32,leaf:u32,side:u32,local:u32)->u32{
  return ${p}IBORef(slot,leaf,side,local).y;}
fn ${p}IBOTemplateDirectory(templateId:u32)->vec4u{
  if(templateId>=IBO1_TEMPLATE_COUNT){return vec4u(IBO1_INVALID);}
  let at=IBO1_TEMPLATE_DIRECTORY_BASE+IBO1_DIRECTORY_WORDS*templateId;
  return vec4u(${p}IBOLoad(at),${p}IBOLoad(at+1u),
    ${p}IBOLoad(at+2u),${p}IBOLoad(at+3u));}
fn ${p}IBOTemplateHeaderWord(templateId:u32,word:u32)->u32{
  let directory=${p}IBOTemplateDirectory(templateId);
  if(directory.x==IBO1_INVALID||word>=IBO1_TEMPLATE_HEADER_WORDS){return IBO1_INVALID;}
  return ${p}IBOLoad(IBO1_BASE+directory.x+word);}
fn ${p}IBOTemplateRowWord(templateId:u32,row:u32,word:u32)->u32{
  let directory=${p}IBOTemplateDirectory(templateId);
  if(directory.x==IBO1_INVALID||row>=directory.z||word>=IBO1_ROW_WORDS){return IBO1_INVALID;}
  return ${p}IBOLoad(IBO1_BASE+directory.x+IBO1_TEMPLATE_HEADER_WORDS
    +IBO1_ROW_WORDS*row+word);}
fn ${p}IBOTemplateTermWord(templateId:u32,term:u32,word:u32)->u32{
  let directory=${p}IBOTemplateDirectory(templateId);
  if(directory.x==IBO1_INVALID||term>=directory.w||word>=IBO1_TERM_WORDS){
    return IBO1_INVALID;}return ${p}IBOLoad(IBO1_BASE+directory.x
      +IBO1_TEMPLATE_HEADER_WORDS+IBO1_ROW_WORDS*directory.z
      +IBO1_TERM_WORDS*term+word);}
fn ${p}IBOStablePacketLeaf(packet:u32)->u32{return packet/IBO1_PACKETS_PER_LEAF;}
fn ${p}IBOStablePacketLocal(packet:u32)->u32{return packet%IBO1_PACKETS_PER_LEAF;}

// One invocation owns one scheduled leaf. Main topology scheduling supplies the
// target descriptor and exact compact refs; no catalog or domain scan occurs here.
fn ${p}IBOBeginDeltaLeaf(slot:u32,leaf:u32,generation:u32,
  activeFlag:bool,descriptor:u32)->vec2u{
  if(leaf>=IBO1_LEAF_CAPACITY||descriptor>=IBO1_CANONICAL_CAPACITY
    ||${p}IBOCanonicalWord(descriptor,0u)==0u
    ||${p}IBOCanonicalWord(descriptor,1u)!=leaf){
    return vec2u(IBO1_FAULT_CANONICAL,leaf);}
  let at=${p}IBOSlotLeafBase(slot)+IBO1_SLOT_LEAF_WORDS*leaf;
  ${p}IBOStore(at,generation);${p}IBOStore(at+1u,select(2u,3u,activeFlag));
  ${p}IBOStore(at+2u,descriptor);${p}IBOStore(at+3u,${p}IBOCanonicalWord(descriptor,3u));
  ${p}IBOStore(at+4u,${p}IBOCanonicalWord(descriptor,4u));${p}IBOStore(at+5u,0u);
  ${p}IBOStore(at+6u,0x811c9dc5u);${p}IBOStore(at+7u,${p}IBOCanonicalWord(descriptor,12u));
  let first=leaf*IBO1_REFS_PER_LEAF*IBO1_SLOT_REF_WORDS;
  for(var word=0u;word<IBO1_REFS_PER_LEAF*IBO1_SLOT_REF_WORDS;word+=1u){
    ${p}IBOStore(${p}IBOSlotRefBase(slot)+first+word,IBO1_INVALID);}
  return vec2u(0u,IBO1_INVALID);}
fn ${p}IBOWriteDeltaRef(slot:u32,leaf:u32,side:u32,local:u32,
  reference:vec3u)->vec2u{
  let record=leaf*IBO1_REFS_PER_LEAF+side*4u+local;
  if(leaf>=IBO1_LEAF_CAPACITY||side>=6u||local>=4u
    ||reference.x>=IBO1_TEMPLATE_COUNT||reference.x>=IBO1_REF_TEMPLATE_MAXIMUM
    ||(reference.y!=IBO1_INVALID&&reference.y>=IBO1_REF_TARGET_MASK)
    ||reference.z==IBO1_INVALID){
    return vec2u(IBO1_FAULT_REFERENCE,record);}
  let at=${p}IBOSlotRefBase(slot)+IBO1_SLOT_REF_WORDS*record;
  let targetId=select(reference.y,IBO1_REF_TARGET_MASK,reference.y==IBO1_INVALID);
  ${p}IBOStore(at,(reference.x<<IBO1_REF_TARGET_BITS)|targetId);
  ${p}IBOStore(at+1u,reference.z);return vec2u(0u,IBO1_INVALID);}
fn ${p}IBOSealDeltaLeaf(slot:u32,leaf:u32)->vec2u{
  if(leaf>=IBO1_LEAF_CAPACITY){return vec2u(IBO1_FAULT_TOPOLOGY,leaf);}
  var packedCounts=0u;var hash=0x811c9dc5u;
  let activeFlag=${p}IBOLeafActive(slot,leaf);
  for(var side=0u;side<6u;side+=1u){var count=0u;var ended=false;
    for(var local=0u;local<4u;local+=1u){let reference=${p}IBORef(slot,leaf,side,local);
      let present=reference.x!=IBO1_INVALID;
      if(!present){ended=true;continue;}let record=leaf*IBO1_REFS_PER_LEAF+side*4u+local;
      if(!activeFlag||ended||reference.x>=IBO1_TEMPLATE_COUNT||reference.z==IBO1_INVALID){
        return vec2u(IBO1_FAULT_REFERENCE,record);}
      count+=1u;hash=${p}IBOFNV(${p}IBOFNV(${p}IBOFNV(${p}IBOFNV(hash,
        side*4u+local),reference.x),reference.y),reference.z);}
    packedCounts|=count<<(3u*side);}
  let at=${p}IBOSlotLeafBase(slot)+IBO1_SLOT_LEAF_WORDS*leaf;
  ${p}IBOStore(at+5u,packedCounts);${p}IBOStore(at+6u,hash);
  return vec2u(0u,IBO1_INVALID);}

// Bounded one-leaf validation used by a receipt reduction. It never scans the world.
fn ${p}IBOValidateDeltaLeaf(slot:u32,leaf:u32,requiredGeneration:u32)->vec2u{
  if(leaf>=IBO1_LEAF_CAPACITY){return vec2u(IBO1_FAULT_TOPOLOGY,leaf);}
  if(${p}IBOLeafGeneration(slot,leaf)!=requiredGeneration){
    return vec2u(IBO1_FAULT_GENERATION,leaf);}
  let descriptor=${p}IBOLeafDescriptorId(slot,leaf);
  if(descriptor>=IBO1_CANONICAL_CAPACITY||${p}IBOCanonicalWord(descriptor,0u)==0u
    ||${p}IBOCanonicalWord(descriptor,1u)!=leaf
    ||${p}IBOCanonicalWord(descriptor,3u)!=${p}IBOLeafCellFirst(slot,leaf)){
    return vec2u(IBO1_FAULT_CANONICAL,leaf);}
  var hash=0x811c9dc5u;
  for(var side=0u;side<6u;side+=1u){let count=${p}IBOFaceRefCount(slot,leaf,side);
    if(count>4u){return vec2u(IBO1_FAULT_REFERENCE,leaf);}
    for(var local=0u;local<4u;local+=1u){let reference=${p}IBORef(slot,leaf,side,local);
      let record=leaf*IBO1_REFS_PER_LEAF+side*4u+local;
      if(local>=count){if(any(reference!=vec3u(IBO1_INVALID))){
          return vec2u(IBO1_FAULT_REFERENCE,record);}continue;}
      if(reference.x>=IBO1_TEMPLATE_COUNT||reference.z==IBO1_INVALID){
        return vec2u(IBO1_FAULT_REFERENCE,record);}
      let directory=${p}IBOTemplateDirectory(reference.x);
      if(directory.x<${l.templatePayloadBaseWords}u||directory.x+directory.y>${l.immutableWords}u
        ||directory.z!=${p}IBOTemplateHeaderWord(reference.x,6u)
        ||directory.w!=${p}IBOTemplateHeaderWord(reference.x,7u)){
        return vec2u(IBO1_FAULT_IMMUTABLE,reference.x);}
      if(reference.y!=IBO1_INVALID&&(reference.y>=IBO1_LEAF_CAPACITY
        ||!${p}IBOLeafActive(slot,reference.y))){return vec2u(IBO1_FAULT_REFERENCE,record);}
      hash=${p}IBOFNV(${p}IBOFNV(${p}IBOFNV(${p}IBOFNV(hash,
        side*4u+local),reference.x),reference.y),reference.z);}}
  if(hash!=${p}IBOLeafWord(slot,leaf,6u)){
    return vec2u(IBO1_FAULT_REFERENCE,leaf);}
  return vec2u(0u,IBO1_INVALID);}

// Main topology validation supplies an independent scheduled digest. Fold this
// result into its existing preflip fault receipt before the shared selector flip.
fn ${p}IBOValidateScheduledDeltaLeaf(slot:u32,leaf:u32,generation:u32,
  expectedActive:bool,expectedDescriptor:u32,expectedFaceCounts:u32,
  expectedRefHash:u32)->vec2u{
  let physical=${p}IBOValidateDeltaLeaf(slot,leaf,generation);
  if(physical.x!=0u){return physical;}
  if(${p}IBOLeafActive(slot,leaf)!=expectedActive
    ||${p}IBOLeafDescriptorId(slot,leaf)!=expectedDescriptor){
    return vec2u(IBO1_FAULT_TOPOLOGY,leaf);}
  if(${p}IBOLeafWord(slot,leaf,5u)!=expectedFaceCounts
    ||${p}IBOLeafWord(slot,leaf,6u)!=expectedRefHash){
    return vec2u(IBO1_FAULT_REFERENCE,leaf);}
  return vec2u(0u,IBO1_INVALID);}

// Caller invokes this only for the exact changed-leaf + face-neighbor closure.
fn ${p}IBOReplayDeltaLeaf(sourceSlot:u32,targetSlot:u32,leaf:u32,generation:u32){
  if(leaf>=IBO1_LEAF_CAPACITY){return;}
  let sourceLeaf=${p}IBOSlotLeafBase(sourceSlot)+IBO1_SLOT_LEAF_WORDS*leaf;
  let targetLeaf=${p}IBOSlotLeafBase(targetSlot)+IBO1_SLOT_LEAF_WORDS*leaf;
  ${p}IBOStore(targetLeaf,generation);
  for(var word=1u;word<IBO1_SLOT_LEAF_WORDS;word+=1u){
    ${p}IBOStore(targetLeaf+word,${p}IBOLoad(sourceLeaf+word));}
  let first=leaf*IBO1_REFS_PER_LEAF*IBO1_SLOT_REF_WORDS;
  for(var word=0u;word<IBO1_REFS_PER_LEAF*IBO1_SLOT_REF_WORDS;word+=1u){
    ${p}IBOStore(${p}IBOSlotRefBase(targetSlot)+first+word,
      ${p}IBOLoad(${p}IBOSlotRefBase(sourceSlot)+first+word));}}
`;
}
