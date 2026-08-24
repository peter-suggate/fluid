/**
 * The compact fine level-set page lookup and its signed-distance sampler.
 *
 * A buffer-native solver publishes its surface as a key-sorted page set rather
 * than a dense field: there is no texture to sample, so every consumer that
 * wants phi at a lattice coordinate has to resolve the page first. That
 * resolution is not a one-liner — it validates the publication transaction,
 * binary-searches the sorted metadata, and re-probes for a Sparse CM12 macro
 * leaf whose single native page covers many logical page slots — and it was
 * hand-copied into each consumer that needed it.
 *
 * Both halves live here so a new consumer adopts the publication ABI instead of
 * re-deriving it. Nothing is declared but functions: the including shader owns
 * the bindings (`fineWorklist`, `fineSamples`, `metadata`, `params`), the
 * packed-sample decoders (`fineLevelSetPackedSampleWGSL`), `finite`, and the
 * coarse fallback the sampler falls through to.
 *
 * The fallback is a parameter rather than a fixed call because what lies
 * outside the compact page set is a property of the publisher, not of this
 * lookup: a Section 5 narrow band falls through to its background octree, while
 * Sparse CM12 publishes a self-contained page set with a dry apron, so a
 * missing page there is authoritative air.
 */

/**
 * `fn pageLookup(key:u32)->u32` — the validated logical-to-physical page map.
 *
 * INVALID unless the whole publication transaction agrees: worklist header
 * shape, generation, capacity, and the page's own metadata identity. Compact
 * publishers (worklist word 3, bit 31) binary-search a key-sorted physical-page
 * list; physical storage itself may be allocated in any order.
 * ordinary publishers index the domain-sized directory that follows the
 * worklist body.
 */
export const compactFineLevelSetPageLookupWGSL = /* wgsl */ `fn pageLookup(key:u32)->u32{
  if(params.table.y!=7u||arrayLength(&fineWorklist)<7u||fineWorklist[0]!=params.table.w
    ||fineWorklist[2]!=params.table.z||(fineWorklist[3]&3u)!=3u||fineWorklist[5]!=1u||fineWorklist[6]!=1u){return INVALID;}
  let logicalCount=params.brickDimensions.x*params.brickDimensions.y*params.brickDimensions.z;
  if(key>=logicalCount){return INVALID;}
  if((fineWorklist[3]&0x80000000u)!=0u){
    let count=min(fineWorklist[1],params.table.z);var low=0u;var high=count;
    loop{if(low>=high){break;}let middle=low+(high-low)/2u;
      let id=fineWorklist[7u+middle];let base=id*4u;
      if(id>=params.table.z||base+2u>=arrayLength(&metadata)){return INVALID;}
      let candidate=metadata[base+1u];
      if(candidate<key){low=middle+1u;}else{high=middle;}}
    var id=INVALID;if(low<count){id=fineWorklist[7u+low];}let base=id*4u;
    return select(INVALID,id,id<params.table.z&&base+2u<arrayLength(&metadata)
      &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==params.table.w);
  }
  let directoryBase=7u+params.table.z;
  if(directoryBase+key>=arrayLength(&fineWorklist)){return INVALID;}
  let id=fineWorklist[directoryBase+key];let base=id*4u;
  return select(INVALID,id,id<params.table.z&&base+2u<arrayLength(&metadata)
    &&metadata[base]==id&&metadata[base+1u]==key&&metadata[base+2u]==params.table.w);
}`;

/**
 * `fn phi(qi:vec3i)->f32` and `fn fineValid(q:vec3u)->bool`, plus the address
 * resolution both rest on.
 *
 * `coarseFallback` names an already-declared `fn (vec3i)->f32` used wherever
 * the compact set cannot answer: out of lattice, no page, an unset sample flag,
 * or a non-finite stored value.
 */
export function makeCompactFineLevelSetPhiWGSL(coarseFallback: string): string {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(coarseFallback)) {
    throw new RangeError("Compact fine level-set coarse fallback must be a WGSL identifier");
  }
  return /* wgsl */ `// Resolve a fine coordinate to either its ordinary 4^3 page or the single
// native-scale page owned by a containing Sparse CM12 macro leaf. Word 3 of
// compact metadata stores [spanLog:5, brick:24, octant:3], and worklist word 3
// advertises the largest span log. The bounded logarithmic probe avoids a
// domain-sized direct directory and never expands a macro leaf by volume.
fn compactBrickFineResolution()->u32{
  let encoded=(fineWorklist[3]>>16u)&31u;return select(8u,encoded,encoded!=0u);
}
fn compactPagesPerSolverAxis()->u32{
  return max(1u,compactBrickFineResolution()/max(1u,params.brickResolution));
}
fn compactSourceSpanLog(source:u32)->u32{
  if(compactPagesPerSolverAxis()==4u){
    return select(0u,(source>>24u)&31u,(source&0x80000000u)!=0u);
  }
  return source>>27u;
}
fn compactSampleAddress(q:vec3u)->vec2u{
  let r=max(1u,params.brickResolution);let pageCoordinate=q/r;
  let exactKey=pageCoordinate.x+params.brickDimensions.x
    *(pageCoordinate.y+params.brickDimensions.y*pageCoordinate.z);
  let exact=pageLookup(exactKey);
  if(exact!=INVALID){
    if((fineWorklist[3]&0x80000000u)==0u){let local=q-pageCoordinate*r;
      return vec2u(exact,local.x+r*(local.y+r*local.z));}
    let source=metadata[4u*exact+3u];let spanLog=compactSourceSpanLog(source);
    if(spanLog==0u){let local=q-pageCoordinate*r;
      return vec2u(exact,local.x+r*(local.y+r*local.z));}
    let scale=compactPagesPerSolverAxis()*(1u<<spanLog);let origin=pageCoordinate*r;
    let local=min((q-origin)/scale,vec3u(r-1u));
    return vec2u(exact,local.x+r*(local.y+r*local.z));
  }
  let maximumSpanLog=(fineWorklist[3]>>8u)&31u;
  for(var spanLog=1u;spanLog<=maximumSpanLog;spanLog+=1u){
    let span=1u<<spanLog;let pageSpan=compactPagesPerSolverAxis()*span;
    let originPage=(pageCoordinate/pageSpan)*pageSpan;
    let key=originPage.x+params.brickDimensions.x
      *(originPage.y+params.brickDimensions.y*originPage.z);
    let id=pageLookup(key);if(id==INVALID){continue;}
    let source=metadata[4u*id+3u];if(compactSourceSpanLog(source)!=spanLog){continue;}
    let scale=pageSpan;let origin=originPage*r;
    let local=min((q-origin)/scale,vec3u(r-1u));
    return vec2u(id,local.x+r*(local.y+r*local.z));
  }
  return vec2u(INVALID);
}
fn phi(qi:vec3i)->f32{
  if(any(qi<vec3i(0))||any(qi>=vec3i(params.sampleDimensions))){return ${coarseFallback}(qi);}
  let address=compactSampleAddress(vec3u(qi));if(address.x==INVALID){return ${coarseFallback}(qi);}
  let index=address.x*params.samplesPerBrick+address.y;
  if(index>=arrayLength(&fineSamples)||(finePackedFlags(index)&1u)==0u||!finite(finePackedPhi(index))){return ${coarseFallback}(qi);}
  return finePackedPhi(index);
}
fn fineValid(q:vec3u)->bool{if(any(q>=params.sampleDimensions)){return false;}let address=compactSampleAddress(q);if(address.x==INVALID){return false;}let index=address.x*params.samplesPerBrick+address.y;return index<arrayLength(&fineSamples)&&(finePackedFlags(index)&1u)!=0u&&finite(finePackedPhi(index));}`;
}

/**
 * `fn compactSampleSpanScale(id:u32)->u32` — fine cells per stored sample.
 *
 * One for an ordinary page and for every non-compact publisher; for a Sparse
 * CM12 macro leaf, the width of the fine block that one stored sample answers
 * for. A consumer that sweeps whole pages needs this to know when neighbouring
 * fine coordinates collapse onto the same sample, which is the difference
 * between one memory read and a redundant fan of identical ones.
 *
 * Emit after `makeCompactFineLevelSetPhiWGSL`, whose `compactSourceSpanLog` and
 * `compactPagesPerSolverAxis` it reads.
 */
export const compactFineLevelSetSpanScaleWGSL = /* wgsl */ `fn compactSampleSpanScale(id:u32)->u32{
  if((fineWorklist[3]&0x80000000u)==0u||4u*id+3u>=arrayLength(&metadata)){return 1u;}
  let spanLog=compactSourceSpanLog(metadata[4u*id+3u]);
  return select(1u,compactPagesPerSolverAxis()<<spanLog,spanLog!=0u);
}`;

/** Words in the `FineParams` uniform the sampler above reads. */
export const COMPACT_FINE_LEVELSET_PARAM_WORDS = 28;

/** WGSL declaration of that uniform, so a consumer cannot mis-order its lanes. */
export const compactFineLevelSetParamsWGSL = /* wgsl */ `struct FineParams{sampleDimensions:vec3u,brickResolution:u32,brickDimensions:vec3u,samplesPerBrick:u32,table:vec4u,settings:vec4f,cellAndDt:vec4f,sizing:vec4f,physical:vec4f}`;

/** Everything the sampler needs from a live compact publication. */
export interface CompactFineLevelSetParamInputs {
  readonly sampleDimensions: readonly [number, number, number];
  readonly brickDimensions: readonly [number, number, number];
  readonly brickResolution: number;
  readonly samplesPerBrick: number;
  readonly pageCapacity: number;
  readonly fineFactor: number;
  readonly fineCellWidth: number;
  readonly domainOrigin: readonly [number, number, number];
  readonly generation: number;
}

/**
 * Pack the uniform exactly as the water pipeline's compact render params do.
 *
 * Table state 7 is the fine-band publication; `table.z` is the page capacity
 * the worklist header must agree with, and `table.w` is the generation every
 * page's metadata must carry before the lookup admits it.
 */
export function packCompactFineLevelSetParams(input: CompactFineLevelSetParamInputs): ArrayBuffer {
  const positive = (value: number, label: string) => {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
    return value;
  };
  input.sampleDimensions.forEach((value, axis) => positive(value, `Compact sample dimension ${axis}`));
  input.brickDimensions.forEach((value, axis) => positive(value, `Compact brick dimension ${axis}`));
  positive(input.brickResolution, "Compact brick resolution");
  positive(input.samplesPerBrick, "Compact samples per brick");
  // A publication may legitimately hold no pages at all — see
  // `validateGlobalFineLevelSetConsumerSource`. `table.z` then agrees with a
  // worklist header whose count is also zero, and every lookup fails its
  // `id < pageCapacity` guard, which is the correct "no fluid here" answer.
  if (!Number.isSafeInteger(input.pageCapacity) || input.pageCapacity < 0) {
    throw new RangeError("Compact page capacity must be a non-negative integer");
  }
  if (!Number.isFinite(input.fineCellWidth) || input.fineCellWidth <= 0) {
    throw new RangeError("Compact fine cell width must be positive and finite");
  }
  if (!Number.isSafeInteger(input.generation) || input.generation < 0) {
    throw new RangeError("Compact publication generation must be a non-negative integer");
  }
  const bytes = new ArrayBuffer(COMPACT_FINE_LEVELSET_PARAM_WORDS * 4);
  const u32 = new Uint32Array(bytes), f32 = new Float32Array(bytes);
  u32.set([...input.sampleDimensions, input.brickResolution], 0);
  u32.set([...input.brickDimensions, input.samplesPerBrick], 4);
  u32.set([input.pageCapacity, 7, input.pageCapacity, input.generation], 8);
  f32.set([...input.domainOrigin, input.fineCellWidth], 12);
  f32[16] = input.fineFactor;
  return bytes;
}
