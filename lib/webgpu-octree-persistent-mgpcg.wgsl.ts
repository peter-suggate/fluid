/**
 * Single-dispatch persistent MGPCG for small pressure systems.
 *
 * The kernel is a transcription, not a redesign: every arithmetic statement
 * below is copied from the three shaders whose dispatch chain it replaces —
 * `octreePipelinedMGPCGShader` (outer CG + compensated reductions),
 * `octreeSection43HybridPreconditionerShader` (k=8 band shell), and
 * `octreeSPGridVCycleShader` / `octreeSPGridAccurateOperatorShader` (the
 * first-order V-cycle and the Section 6.3 A2 apply). Phases that were
 * separate dispatches become `storageBarrier(); workgroupBarrier();`-separated
 * regions of one 256-lane workgroup; every loop whose body contains a barrier
 * is bounded by a `workgroupUniformLoad`, exactly as
 * `marchAirSupportFacesToFixedPoint` does in
 * `webgpu-octree-air-velocity-support-gpu.ts`.
 *
 * Deliberate, documented deviations from a literal transcription — each one
 * either provably value-neutral or strictly more fail-closed — are marked
 * `TRANSCRIPTION NOTE` in the source below.
 */
import { octreeCompensatedF32WGSL } from "./octree-compensated-f32.wgsl";
import { OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY } from "./webgpu-octree-section43-contract";
import { octreeSection63DirectionChannelWGSL } from "./webgpu-octree-spgrid-vcycle";

/** Row-shaped arena channels. Every channel is `rowCapacity` words. */
export const OCTREE_PERSISTENT_MGPCG_CHANNEL = Object.freeze({
  pressure: 0,
  residual: 1,
  preconditioned: 2,
  preconditionedImage: 3,
  direction: 4,
  directionImage: 5,
  hybridA: 6,
  hybridB: 7,
  innerRhs: 8,
  innerCorrection: 9,
  operatorImage: 10,
  bandA: 11,
  bandB: 12,
  bandList: 13,
  rhs: 14,
  pressureSeed: 15,
} as const);

export const OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT = 16;

/** Word offsets inside the arena's fixed 1024-word staging header. */
export const OCTREE_PERSISTENT_MGPCG_HEADER = Object.freeze({
  /** Six-word accepted structured control `[flags, firstError, rows, generation, bank, slots]`. */
  accepted: 0,
  acceptedWords: 16,
  /** `levelCount * 12 + 2` SPGrid worklist counts / published dispatch words. */
  dispatch: 16,
  dispatchWords: 256,
  /** Eight `(epoch, count)` pairs: `(bank * 4 + rowClass)`. */
  worksetHeaders: 272,
  worksetHeaderWords: 16,
  /**
   * `[marker, bandRows, regularBandRows, coarseRegularBandRows]`, authored once
   * per solve by the class-0 band census and never read by the kernel. It sits
   * above every host-staged region (`encodeSolve` copies the dispatch header
   * into `[16, 272)` and nothing else below `totalWords`), so it is the only
   * kernel arena write below `ARENA_HEADER`. Emitted only when
   * `FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS` selects a mode.
   */
  bandCensus: 288,
  bandCensusWords: 4,
  totalWords: 1024,
} as const);

/** Sentinel proving `bandCensus` was authored by this dispatch, not left over. */
export const OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER = 0x4241_4e44;

/** 128-lane virtual reduction groups, matching the hierarchical partial shape. */
export const OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES = 128;
/** Eight words per partial: four `CompensatedF32` accumulators. */
export const OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS = 8;

export interface OctreePersistentMGPCGShaderOptions {
  /** Encoded outer-iteration budget; mirrors `plan.maximumIterations`. */
  readonly maximumIterations: number;
  /**
   * Address row-shaped channels with the accepted live-row count rather than
   * the provisioned capacity. The two CPU/GPU copy-staged input channels are
   * repacked into this live prefix once at kernel entry.
   */
  readonly compactLiveRows?: boolean;
  /** Diagnostic A/B: retain the class-0 arithmetic but use the full page-slot
   * resolver, localizing address reduction independently of the hybrid split. */
  readonly regularAddressOracle?: boolean;
  /** Diagnostic A/B: add the full fine-level adjoint to class-0 rows. */
  readonly regularAdjointOracle?: boolean;
  /** Diagnostic A/B: retain reduced addressing but visit all 18 channels. */
  readonly regularAllChannelsOracle?: boolean;
  /** Diagnostic A/B: classify the Section 4.3 shell from the original row
   * predicate while leaving the hybrid A2 apply enabled. */
  readonly section63BandOracle?: boolean;
  /** Diagnostic A/B: route class 0 through the unmodified full A2 row. */
  readonly fullApplyOracle?: boolean;
  /** Diagnostic A/B: restore the original linear full-row apply traversal. */
  readonly linearApplyOracle?: boolean;
  /**
   * Memory-only smoother staging: hoist the two solve-invariant header regions
   * into workgroup storage and issue `applied`'s eighteen spokes as independent
   * load waves instead of one dependent chain. Value-neutral by construction;
   * see `stagedSmootherApplied` below for the term-by-term equality argument.
   */
  readonly stagedSmoother?: boolean;
  /**
   * Route class-0 rows of the Section 4.3 band sweep through the same cheap
   * regular apply `applyAllRows` already uses for them.
   *
   * `"census"` publishes the class-0 row map and the band census but leaves
   * every band row on `applyRow`, so the hit rate and the map's own cost can be
   * measured before the routing is trusted. `"route"` additionally takes the
   * regular path. See `octreePersistentMGPCGRegularBandRowsMode` for the
   * numerical-equivalence argument and why this defaults off.
   */
  readonly regularBandRows?: "census" | "route";
  /** Diagnostic-only: publish one internal channel through pressureOut after
   * the solve so the Dawn symmetry oracle can localize a broken sub-map. */
  readonly diagnosticOutputChannel?: number;
  /** Diagnostic-only snapshots reuse the two staged-input regions after P1. */
  readonly diagnosticStageCapture?: "initialPreconditioned" | "vcyclePhase0"
    | "vcyclePresmooth" | "vcycleRestrict" | "vcycleProlong" | "vcyclePostsmooth"
    | "section43Boundary" | "section43First" | "section43Inputs";
  /** Compile-time-only profiler hooks. Omitted production output remains
   * byte-for-byte free of profiler declarations, builtins, and calls. */
  readonly activity?: Readonly<{
    parameters: string;
    enter: string;
    exit: string;
  }>;
}

/**
 * Word count of the row arena for a given provisioned capacity: header, the
 * sixteen row channels, one partial slot per 128-lane virtual group, then two
 * capacity-strided input channels. Keeping the staged RHS and seed outside the
 * compact hot channels makes them immutable while the live-row stride is
 * repacked in place.
 */
export function octreePersistentMGPCGArenaWords(rowCapacity: number): number {
  const partials = Math.ceil(rowCapacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES);
  return OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
    + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * rowCapacity
    + partials * OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS
    + 2 * rowCapacity;
}

/**
 * Workgroup-storage footprint the kernel declares. Checked against
 * `maxComputeWorkgroupStorageSize` before any pipeline is created, because
 * an overflow here is a validation error the `skip_validation` harness turns
 * into a SIGSEGV rather than a message.
 */
export const OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES =
  // merged: array<MergedScalars, 128>, eight f32 each
  128 * 8 * 4
  // scan: array<u32, 256> for the deterministic band compaction
  + 256 * 4
  // uniform-load scalars (rows, levels, partials, counts, halt flags)
  + 64;

/**
 * Extra workgroup storage the staged smoother declares.
 *
 * Both regions are immutable for the whole dispatch, so one copy replaces every
 * subsequent storage re-read:
 *  - `wDispatch`: the 256-word dispatch header `arena[16..272)`, authored by
 *    the host `copyBufferToBuffer` before the dispatch and never written by the
 *    kernel (every kernel arena write is at `ARENA_HEADER` or above, at
 *    `partialBase()`, or at `stagedCh()`), re-read today by `count()`,
 *    `pageCount()` and `transferCount()` — the last two from inside
 *    `regularPageSlot`, i.e. twice per stencil channel per band row per sweep.
 *  - `wAccepted`: the eight-word accepted-authority prefix of a `read` binding,
 *    re-read today by `acceptedBank()` and therefore by `coefficientBase()` —
 *    once per `diagonalAt`, and 144 times per `applyRow` whose `finerAdjoint`
 *    runs (eight children by eighteen `coefficientForDirection` probes).
 *
 * 1,056 bytes on top of the 5,184-byte baseline, well inside the 16 KiB
 * `maxComputeWorkgroupStorageSize` floor Dawn exposes on Metal. The deleted
 * page-blocked halo needed 9,600 bytes on its own and is not restored here;
 * see `stagedSmootherApplied`.
 */
export const OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES =
  // wDispatch: array<u32, 256>
  OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords * 4
  // wAccepted: array<u32, 8>
  + 8 * 4;

/** Declared workgroup footprint for a given staged-smoother selection. */
export function octreePersistentMGPCGWorkgroupBytes(stagedSmoother = false): number {
  return OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES
    + (stagedSmoother ? OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES : 0);
}

/**
 * The staged `applied()` body: the same eighteen terms, in the same order, fed
 * to the same `canonical18Sum`, with only the load schedule changed.
 *
 * HEAD fuses one dependent chain per spoke — coefficient load, compare,
 * neighbour-column load, bounds compare, scattered value load — and repeats it
 * eighteen times. Inside a single-workgroup kernel there is no other resident
 * work to hide any of those round trips, so the eighteen spokes serialize into
 * roughly thirty-six memory latencies per cell per sweep. Every spoke is
 * independent of every other, so the identical values can be fetched as three
 * waves of eighteen independent loads.
 *
 * BIT-IDENTITY, term by term (`t{k}` here versus `terms[k]` at HEAD):
 *  - `c{k}` is the identical `loadf(S_XP+k,l,slot)`; `k` is a literal here and
 *    a fully-unrollable loop counter at HEAD, so the address is the same word.
 *  - `n{k}` is the identical `topology[columnBase+k*span]`, now issued for
 *    every `k` rather than only for nonzero coefficients. The stencil neighbour
 *    table is exactly `18 * totalLevelSlots` words (see `stencilNeighbourWords`
 *    in the SPGrid plan) and `columnBase + 17*span` is its last column, so the
 *    unconditional loads are in bounds and their results are discarded.
 *  - `g{k}` is HEAD's acceptance predicate verbatim: nonzero coefficient AND a
 *    resolvable in-capacity column.
 *  - the report loop fires `reportAt(OVERFLOW,75u,slot)` for exactly the same
 *    `k`, in ascending `k`, exactly as HEAD's `continue` arm does.
 *  - `x{k}` reads slot zero when `g{k}` is false. That address is
 *    `at(source,l,0u)`, always in bounds, and the value is discarded by the
 *    following `select`, so no term changes and no report is added or lost.
 *  - `t{k}` is `c{k}*x{k}` when accepted and the default `0.0` otherwise —
 *    HEAD's zero-initialized `terms[k]` for both the `c==0` and the
 *    invalid-column arms.
 *  - the eighteen scalars enter `canonical18Sum` in ascending `k`, so the
 *    axis/diagonal pairing and the `sorted3Sum`/`sorted6Sum` association order
 *    are untouched. This is what keeps the exact-D4 window where it is.
 *
 * Replacing `var terms:array<f32,18>` with eighteen named scalars is the second
 * half of the change and is free: HEAD indexes that array with a loop counter,
 * which Metal backs with thread-local scratch whenever it declines to unroll,
 * and the only consumer takes it by value at constant indices.
 */
function stagedSmootherApplied(): string {
  const spokes = Array.from({ length: 18 }, (_, k) => k);
  const wave = (build: (k: number) => string) => spokes.map(build).join("");
  return ` let cap=levelCapacity(l);
 ${wave((k) => `let c${k}=loadf(S_XP+${k}u,l,slot);`)}
 ${wave((k) => `let n${k}=topology[columnBase+${k}u*span];`)}
 ${wave((k) => `let g${k}=c${k}!=0.0&&n${k}!=INVALID&&n${k}<cap;`)}
 ${wave((k) => `if(c${k}!=0.0&&!g${k}){reportAt(OVERFLOW,75u,slot);}`)}
 ${wave((k) => `let x${k}=loadf(source,l,select(0u,n${k},g${k}));`)}
 ${wave((k) => `let t${k}=select(0.0,c${k}*x${k},g${k});`)}
 return loadf(S_DIAG,l,slot)*loadf(source,l,slot)-canonical18Sum(array<f32,18>(${
    spokes.map((k) => `t${k}`).join(",")}));`;
}

export function octreePersistentMGPCGWGSL(
  options: OctreePersistentMGPCGShaderOptions,
): string {
  const iterations = options.maximumIterations;
  const rowStride = options.compactLiveRows === false ? "capacity()" : "wRows";
  const regularSlotLookup = options.regularAddressOracle
    ? "opPageSlot(l,page,q,vec3u(targetQ),row)"
    : "regularPageSlot(l,page,q,vec3u(targetQ),row)";
  const regularAdjoint = options.regularAdjointOracle
    ? " + finerAdjoint(row,q,l,x,inCh)" : "";
  const regularChannelCount = options.regularAllChannelsOracle ? 18 : 6;
  const bandInitialization = options.section63BandOracle
    ? `for(var row=lane;row<liveRows;row+=LANES){if(!stopped()){
       ustore(CH_BANDA,row,select(0u,1u,section63Class(row)>=2u));}}`
    : `for(var cls=0u;cls<5u;cls+=1u){let n=worksetCount(cls);
     for(var item=lane;item<n;item+=LANES){let row=worksetRow(cls,item);
      if(stopped()){continue;}
      if(row>=liveRows||!validDiagonal(row)||!validSection63Row(row)){
       reportAt(ERR_ROW,10u,row);continue;}
      if(cls==0u&&!regularCoefficientTailValid(row)){continue;}
      ustore(CH_BANDA,row,select(0u,1u,cls!=0u&&cls!=4u));}}`;
  const classZeroApply = options.fullApplyOracle
    ? "applyRow(row,inCh,outCh)" : "applyRegularRow(row,inCh,outCh)";
  const allRowsApply = options.linearApplyOracle
    ? `for(var row=lane;row<liveRows;row+=LANES){if(!stopped()){applyRow(row,inCh,outCh);}}`
    : `for(var cls=0u;cls<5u;cls+=1u){let n=worksetCount(cls);
     for(var item=lane;item<n;item+=LANES){let row=worksetRow(cls,item);
      if(!stopped()&&row<liveRows){if(cls==0u){${classZeroApply};}else if(cls==4u){applyIdentityRow(row,inCh,outCh);}else{applyRow(row,inCh,outCh);}}}}`;
  // The band list carries row indices only; unlike `applyAllRows` it has no
  // class in hand, because the class lives in the five accepted worksets and
  // nowhere else. P4b republishes the class-0 membership as a per-row map in
  // CH_BANDA — dead from the last dilation to the end of the dispatch — and the
  // sweep takes the SAME `classZeroApply` string `applyAllRows` takes, so the
  // full-apply oracle continues to disable both routings together.
  const regularBandRows = options.regularBandRows;
  if (regularBandRows !== undefined
    && regularBandRows !== "census" && regularBandRows !== "route") {
    throw new RangeError("Persistent MGPCG regular band-row mode is invalid");
  }
  const bandRowApply = regularBandRows === "route"
    ? `if(uload(CH_BANDA,row)!=0u){${classZeroApply};}else{applyRow(row,inCh,outCh);}`
    : "applyRow(row,inCh,outCh);";
  // Gated so the declined option leaves the emitted kernel byte-identical,
  // comments included: a shader module is hashed and cached by its source.
  const bandRowsNote = regularBandRows === undefined ? "" : `// The band shell is
// the one sweep that runs 2 * boundarySweeps - 1 times per Section 4.3
// correction, and at HEAD every one of its rows takes the full eighteen-channel
// power apply, including the rows the accepted publication already proved
// regular. CH_BANDA carries that proof forward as a per-row map (P4b).
`;
  const bandCensusDeclarations = regularBandRows === undefined ? ""
    : `const H_BAND_CENSUS=${OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus}u;\n`
      + `const BAND_CENSUS_MARKER=${
        OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER >>> 0}u;\n`;
  // Ordering: after the third dilation's barrier (nothing reads CH_BANDA again)
  // and before `compactBand`, so the map is visible to every later band sweep.
  // The membership is exactly `applyAllRows`' `cls==0u` arm — P0 already proved
  // the five lists partition `[0, rows())` exactly, so the bounds test below is
  // belt-and-braces, and P4's `regularCoefficientTailValid` has already failed
  // the solve closed for any class-0 row whose edge coefficients are nonzero.
  const bandRowClassMap = regularBandRows === undefined ? "" : `
  // --- P4b: class-0 row map for the Section 4.3 band sweep -----------------
  for(var row=lane;row<liveRows;row+=LANES){if(!stopped()){ustore(CH_BANDA,row,0u);}}
  storageBarrier();workgroupBarrier();
  let regularRowCount=worksetCount(0u);
  for(var item=lane;item<regularRowCount;item+=LANES){
   let row=worksetRow(0u,item);
   if(!stopped()&&row<liveRows){ustore(CH_BANDA,row,1u);}}
  storageBarrier();workgroupBarrier();`;
  // One serial lane-zero pass over a list bounded by the live row count, once
  // per solve. It reads only what the sweep reads and writes only the four
  // host-visible census words, so it cannot perturb any solve value.
  const bandRowCensus = regularBandRows === undefined ? "" : `
  if(lane==0u){
   var censusRegular=0u;var censusCoarse=0u;let censusBand=wBand;
   for(var item=0u;item<censusBand;item+=1u){
    let row=uload(CH_BANDLIST,item);
    if(row==INVALID||row>=liveRows||uload(CH_BANDA,row)==0u){continue;}
    censusRegular+=1u;
    if(row<arrayLength(&geometry)&&firstTrailingBit(geometry[row].y)!=0u){
     censusCoarse+=1u;}}
   arena[H_BAND_CENSUS]=BAND_CENSUS_MARKER;
   arena[H_BAND_CENSUS+1u]=censusBand;
   arena[H_BAND_CENSUS+2u]=censusRegular;
   arena[H_BAND_CENSUS+3u]=censusCoarse;}`;
  const diagnosticOutputChannel = options.diagnosticOutputChannel
    ?? OCTREE_PERSISTENT_MGPCG_CHANNEL.pressure;
  const captureInitialResidual = options.diagnosticStageCapture
    && options.diagnosticStageCapture !== "section43Boundary"
    && options.diagnosticStageCapture !== "section43First"
    && options.diagnosticStageCapture !== "section43Inputs"
    ? "arena[stagedCh(CH_RHS,row)]=bitcast<u32>(value);" : "";
  const captureInitialPreconditioned = options.diagnosticStageCapture === "initialPreconditioned"
    ? `for(var row=lane;row<liveRows;row+=LANES){if(!stopped()){
       arena[stagedCh(CH_SEED,row)]=bitcast<u32>(vload(CH_PRE,row));}}
      storageBarrier();workgroupBarrier();` : "";
  const captureNative = (stage: OctreePersistentMGPCGShaderOptions["diagnosticStageCapture"],
    level: string, channel: string) => options.diagnosticStageCapture === stage
    ? `if(atomicLoad(&control[3])==0u){for(var r=lane;r<rows();r+=LANES){
       let native=firstTrailingBit(geometry[r].y);if(native==${level}){
        arena[stagedCh(CH_SEED,r)]=state[at(${channel},native,rowMap(native,r))];}}}` : "";
  const captureVcyclePhase0 = options.diagnosticStageCapture === "vcyclePhase0"
    ? `if(l==0u){${captureNative("vcyclePhase0", "l", "S_B")}}` : "";
  const captureVcyclePresmooth = captureNative("vcyclePresmooth", "l", "S_A");
  const captureVcycleRestrict = captureNative("vcycleRestrict", "l+1u", "S_RHS");
  const captureVcycleProlong = captureNative("vcycleProlong", "l", "S_A");
  const captureVcyclePostsmooth = captureNative("vcyclePostsmooth", "l", "S_A");
  const captureSection43Presmooth = options.diagnosticStageCapture === "section43Boundary"
    ? `if(atomicLoad(&control[3])==0u){for(var r=lane;r<liveRows;r+=LANES){
       arena[stagedCh(CH_RHS,r)]=bitcast<u32>(vload(CH_HA,r));}}
      storageBarrier();workgroupBarrier();` : "";
  const captureSection43InnerRhs = options.diagnosticStageCapture === "section43Boundary"
    ? `if(atomicLoad(&control[3])==0u){for(var r=lane;r<liveRows;r+=LANES){
       arena[stagedCh(CH_SEED,r)]=bitcast<u32>(vload(CH_INNER_RHS,r));}}
      storageBarrier();workgroupBarrier();` : "";
  const captureSection43Initial = options.diagnosticStageCapture === "section43First"
    ? `if(atomicLoad(&control[3])==0u){for(var r=lane;r<liveRows;r+=LANES){
       arena[stagedCh(CH_RHS,r)]=bitcast<u32>(vload(CH_HB,r));}}
      storageBarrier();workgroupBarrier();` : "";
  const captureSection43First = options.diagnosticStageCapture === "section43First"
    ? `if(iteration==1u&&atomicLoad(&control[3])==0u){for(var r=lane;r<liveRows;r+=LANES){
       arena[stagedCh(CH_SEED,r)]=bitcast<u32>(vload(CH_HA,r));}}
      storageBarrier();workgroupBarrier();` : "";
  const captureSection43Inputs = options.diagnosticStageCapture === "section43Inputs"
    ? `if(atomicLoad(&control[3])==0u){for(var r=lane;r<liveRows;r+=LANES){
       arena[stagedCh(CH_RHS,r)]=bitcast<u32>(vload(rhsCh,r));
       arena[stagedCh(CH_SEED,r)]=bitcast<u32>(diagonalAt(r));}}
      storageBarrier();workgroupBarrier();` : "";
  if (!Number.isSafeInteger(diagnosticOutputChannel)
    || diagnosticOutputChannel < 0
    || diagnosticOutputChannel >= OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT) {
    throw new RangeError("Persistent MGPCG diagnostic output channel is invalid");
  }
  const stagedSmoother = options.stagedSmoother === true;
  const stagedSmootherDeclarations = stagedSmoother
    ? `var<workgroup> wDispatch:array<u32,${OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords}>;\n`
      + "var<workgroup> wAccepted:array<u32,8>;\n"
    : "";
  const acceptedRead = stagedSmoother ? "wAccepted[word]" : "acceptedBoundary[word]";
  const dispatchRead = stagedSmoother ? "wDispatch[word]" : "arena[H_DISPATCH+word]";
  const appliedBody = stagedSmoother ? stagedSmootherApplied()
    : ` var terms:array<f32,18>;
 for(var k=0u;k<18u;k+=1u){let c=loadf(S_XP+k,l,slot);if(c==0.0){continue;}
  let other=topology[columnBase+k*span];
  if(other==INVALID||other>=levelCapacity(l)){reportAt(OVERFLOW,75u,slot);continue;}
  terms[k]=c*loadf(source,l,other);}
 return loadf(S_DIAG,l,slot)*loadf(source,l,slot)-canonical18Sum(terms);`;
  // Emitted ahead of BOTH P0 and the profiler's enter checkpoint. P0's
  // lane-zero block snapshots the accepted authority through `acc()`, and the
  // enter hook's sampled-lane predicate is `lane < rows() && !failed()`, which
  // is another `acc()` — either one running before the copy would read
  // uninitialized workgroup storage. An out-of-range staging read takes the
  // identical WGSL bounds clamp the direct `acc()`/`dispatchWord()` read took,
  // so a malformed authority binding still lands on the same validation code.
  const stagedSmootherPrologue = stagedSmoother ? `
 // --- P-1: solve-invariant header staging --------------------------------
 for(var w=lane;w<${OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords}u;w+=LANES){
  wDispatch[w]=arena[H_DISPATCH+w];}
 if(lane<8u){wAccepted[lane]=acceptedBoundary[lane];}
 storageBarrier();workgroupBarrier();` : "";
  const activityParameters = options.activity?.parameters
    ? `,\n ${options.activity.parameters}` : "";
  const activityEnter = options.activity?.enter ? `\n ${options.activity.enter}` : "";
  const activityExit = options.activity?.exit ? `\n ${options.activity.exit}` : "";
  return /* wgsl */ `
${octreeCompensatedF32WGSL}
struct MergedScalars{gamma:CompensatedF32,delta:CompensatedF32,rr:CompensatedF32,bb:CompensatedF32}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct TransferTarget{coarse:u32,weight:f32}
struct Params{
 dims:vec4u,       // domain x/y/z, rowCapacity
 hierarchy:vec4u,  // levelCount, levelStride, totalLevelSlots, coefficientBankStrideWords
 shape:vec4u,      // encodedIterations, chebyshevDegree, boundarySweeps, bandLayers
 sizes:vec4u,      // transferStride, totalBrickCount, pageDirectoryWords, stagedInputBase
 numerics:vec4f,   // relativeTolerance, absoluteTolerance, tiny, damping
 worksets:vec4u,   // seven-word class stride, A/B bank stride, reserved
 levelCaps:array<vec4u,4>,
 levelBases:array<vec4u,4>,
 brickOffsets:array<vec4u,4>,
 pageOffsets:array<vec4u,4>,
 transferOffsets:array<vec4u,4>,
}
@group(0) @binding(0) var<uniform> p:Params;
@group(0) @binding(1) var<storage,read_write> arena:array<u32>;
@group(0) @binding(2) var<storage,read_write> state:array<u32>;
@group(0) @binding(3) var<storage,read> topology:array<u32>;
@group(0) @binding(4) var<storage,read> coefficients:array<f32>;
@group(0) @binding(5) var<storage,read> geometry:array<vec4u>;
@group(0) @binding(6) var<storage,read> metrics:array<Metric>;
@group(0) @binding(7) var<storage,read_write> control:array<atomic<u32>>;
@group(0) @binding(8) var<storage,read_write> pressureOut:array<f32>;
@group(0) @binding(9) var<storage,read> dynamicWorksets:array<u32>;
@group(0) @binding(10) var<storage,read> acceptedBoundary:array<u32>;

const INVALID=0xffffffffu;
const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u;
// State-arena channels (webgpu-octree-spgrid-vcycle.ts, verbatim ordinals).
const S_KEY=0u;const S_FLAGS=1u;const S_DIAG=2u;const S_XP=3u;
const S_YZMM=20u;const S_RHS=21u;const S_A=22u;const S_B=23u;
const S_OWNER=24u;const S_SPECTRAL=25u;const STATE_CHANNELS=26u;
const PAGE_RECORD_WORDS=28u;
const DISPATCH_WORDS=12u;
const LANES=256u;
const REDUCTION_LANES=${OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES}u;
const MAX_LIVE_ROWS=${OCTREE_PERSISTENT_MGPCG_MAXIMUM_ROW_CAPACITY}u;

// Solve-control error flags. The three producers already share these bit
// values; keeping the identical encodings is what lets the snapshot ring and
// the Part-A tripwires read this path unchanged.
const ERR_AUTHORITY=1u;const ERR_ROW=2u;const ERR_NONFINITE=4u;
const ERR_NONPOSITIVE_PRE=8u;const ERR_NONPOSITIVE_CURVATURE=16u;const ERR_NONCONVERGENCE=32u;
// SPGrid V-cycle report flags reuse the same bits (OVERFLOW=2, NONFINITE=4,
// NONPOSITIVE=8); the stage ordinals below are the ones each source kernel
// already writes, so failureStage/failureRow stay comparable across paths.
const OVERFLOW=2u;const NONFINITE=4u;const NONPOSITIVE=8u;

// Arena geometry.
const H_ACCEPTED=${OCTREE_PERSISTENT_MGPCG_HEADER.accepted}u;
const H_DISPATCH=${OCTREE_PERSISTENT_MGPCG_HEADER.dispatch}u;
const H_WORKSET=${OCTREE_PERSISTENT_MGPCG_HEADER.worksetHeaders}u;
const ARENA_HEADER=${OCTREE_PERSISTENT_MGPCG_HEADER.totalWords}u;
const CHANNELS=${OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT}u;
const CH_PRESSURE=${OCTREE_PERSISTENT_MGPCG_CHANNEL.pressure}u;
const CH_RESIDUAL=${OCTREE_PERSISTENT_MGPCG_CHANNEL.residual}u;
const CH_PRE=${OCTREE_PERSISTENT_MGPCG_CHANNEL.preconditioned}u;
const CH_PREIMG=${OCTREE_PERSISTENT_MGPCG_CHANNEL.preconditionedImage}u;
const CH_DIR=${OCTREE_PERSISTENT_MGPCG_CHANNEL.direction}u;
const CH_DIRIMG=${OCTREE_PERSISTENT_MGPCG_CHANNEL.directionImage}u;
const CH_HA=${OCTREE_PERSISTENT_MGPCG_CHANNEL.hybridA}u;
const CH_HB=${OCTREE_PERSISTENT_MGPCG_CHANNEL.hybridB}u;
const CH_INNER_RHS=${OCTREE_PERSISTENT_MGPCG_CHANNEL.innerRhs}u;
const CH_INNER_CORR=${OCTREE_PERSISTENT_MGPCG_CHANNEL.innerCorrection}u;
const CH_OPIMG=${OCTREE_PERSISTENT_MGPCG_CHANNEL.operatorImage}u;
const CH_BANDA=${OCTREE_PERSISTENT_MGPCG_CHANNEL.bandA}u;
const CH_BANDB=${OCTREE_PERSISTENT_MGPCG_CHANNEL.bandB}u;
const CH_BANDLIST=${OCTREE_PERSISTENT_MGPCG_CHANNEL.bandList}u;
const CH_RHS=${OCTREE_PERSISTENT_MGPCG_CHANNEL.rhs}u;
const CH_SEED=${OCTREE_PERSISTENT_MGPCG_CHANNEL.pressureSeed}u;
${bandCensusDeclarations}
${stagedSmootherDeclarations}var<workgroup> merged:array<MergedScalars,${OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES}>;
var<workgroup> scan:array<u32,256>;
var<workgroup> wRows:u32;
var<workgroup> wLevels:u32;
var<workgroup> wPartials:u32;
var<workgroup> wBand:u32;
var<workgroup> wHalt:u32;
var<workgroup> wInitial:u32;

fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn failed()->bool{return atomicLoad(&control[0])!=0u;}
fn stopped()->bool{return failed()||atomicLoad(&control[1])!=0u;}
// Verbatim from all three producers: one claimant owns failureStage/failureRow.
fn reportAt(flag:u32,stage:u32,index:u32){
 atomicOr(&control[0],flag);
 for(var retry=0u;retry<16u;retry+=1u){
  let claim=atomicCompareExchangeWeak(&control[6],0u,stage);
  if(claim.exchanged){atomicStore(&control[7],index);return;}
  if(claim.old_value!=0u){return;}}
}
fn pairAt(word:u32)->CompensatedF32{
 return CompensatedF32(bitcast<f32>(atomicLoad(&control[word])),
  bitcast<f32>(atomicLoad(&control[word+1u])));}
fn storePair(word:u32,value:CompensatedF32){
 atomicStore(&control[word],bitcast<u32>(value.hi));
 atomicStore(&control[word+1u],bitcast<u32>(value.lo));}

// ---------------------------------------------------------------------------
// Accepted boundary authority and its banked memberships are read directly.
// This removes nine staging copies and, more importantly, keeps the class
// membership and coefficient bank under one live epoch selection.
// ---------------------------------------------------------------------------
fn acc(word:u32)->u32{return ${acceptedRead};}
fn capacity()->u32{return p.dims.w;}
fn rows()->u32{return min(acc(2u),capacity());}
fn acceptedEpoch()->u32{return acc(4u);}
fn acceptedBank()->u32{return acc(5u)&1u;}
fn worksetBase(cls:u32)->u32{return acceptedBank()*p.worksets.y+cls*p.worksets.x;}
fn worksetCount(cls:u32)->u32{return dynamicWorksets[worksetBase(cls)+1u];}
fn worksetRow(cls:u32,item:u32)->u32{return dynamicWorksets[worksetBase(cls)+7u+item];}
fn validateAcceptedRowPartition()->u32{
 if(arrayLength(&acceptedBoundary)<7u||acc(0u)!=0u||acc(1u)!=INVALID
  ||acc(2u)==0u||acc(2u)>capacity()||acc(3u)==0u||acceptedEpoch()==0u
  ||acc(5u)>1u||acc(6u)!=acceptedEpoch()){return 1u;}
 var counts:array<u32,5>;var cursor:array<u32,5>;var total=0u;
 for(var cls=0u;cls<5u;cls+=1u){
  let base=worksetBase(cls);if(base+7u>arrayLength(&dynamicWorksets)){return 0x100u+cls*16u;}
  let n=dynamicWorksets[base+1u];let blocks=(n+63u)/64u;let dx=max(1u,min(65535u,blocks));
  if(dynamicWorksets[base]!=acceptedEpoch()){return 0x101u+cls*16u;}
  if(n>rows()){return 0x102u+cls*16u;}
  if(dynamicWorksets[base+2u]<rows()){return 0x103u+cls*16u;}
  if(dynamicWorksets[base+3u]!=3u){return 0x104u+cls*16u;}
  if(dynamicWorksets[base+4u]!=dx){return 0x105u+cls*16u;}
  if(dynamicWorksets[base+5u]!=(blocks+dx-1u)/dx){return 0x106u+cls*16u;}
  if(dynamicWorksets[base+6u]!=1u){return 0x107u+cls*16u;}
  if(base+7u+n>arrayLength(&dynamicWorksets)){return 0x108u+cls*16u;}
  counts[cls]=n;cursor[cls]=0u;total+=n;}
 if(total!=rows()){return 0x200u;}
 // Each class list is index-ordered. A five-way merge therefore proves the
 // complete exact partition in O(rows) reads: a duplicate leaves a value
 // below the expected row, and an omission advances the minimum above it.
 for(var expected=0u;expected<rows();expected+=1u){
  var best=INVALID;var owner=INVALID;
  for(var cls=0u;cls<5u;cls+=1u){if(cursor[cls]<counts[cls]){
   let candidate=worksetRow(cls,cursor[cls]);if(candidate<best){best=candidate;owner=cls;}}}
  if(best!=expected||owner==INVALID){return 0x201u;}cursor[owner]+=1u;}
 for(var cls=0u;cls<5u;cls+=1u){if(cursor[cls]!=counts[cls]){return 0x202u+cls;}}
 return 0u;}
fn dispatchWord(word:u32)->u32{return ${dispatchRead};}
fn count(l:u32)->u32{return dispatchWord(l*DISPATCH_WORDS);}
fn transferCount(l:u32)->u32{return dispatchWord(l*DISPATCH_WORDS+1u);}
fn pageCount(l:u32)->u32{return dispatchWord(l*DISPATCH_WORDS+8u);}

// The source copies land after the capacity-shaped hot arena and reduction
// partials. All hot
// vectors use a live-row stride in the production variant, keeping the exact
// adaptive solve in a contiguous working set instead of spacing consecutive
// channels by the conservative provisioned capacity. The legacy
// capacity-strided variant is retained as a process-local A/B oracle.
fn stagedCh(c:u32,r:u32)->u32{return p.sizes.w+(c-CH_RHS)*capacity()+r;}
fn stagedVload(c:u32,r:u32)->f32{return bitcast<f32>(arena[stagedCh(c,r)]);}
fn rowStride()->u32{return ${rowStride};}
fn ch(c:u32,r:u32)->u32{return ARENA_HEADER+c*rowStride()+r;}
fn vload(c:u32,r:u32)->f32{return bitcast<f32>(arena[ch(c,r)]);}
fn vstore(c:u32,r:u32,v:f32){arena[ch(c,r)]=bitcast<u32>(v);}
fn uload(c:u32,r:u32)->u32{return arena[ch(c,r)];}
fn ustore(c:u32,r:u32,v:u32){arena[ch(c,r)]=v;}
fn partialBase()->u32{return ARENA_HEADER+CHANNELS*rowStride();}

// ---------------------------------------------------------------------------
// Address helpers. These are the SPGrid V-cycle's memoized-table forms; the
// accurate operator and the Section 4.3 shell recompute the identical values
// with closed-form loops, so substituting the table changes no address.
// ---------------------------------------------------------------------------
fn levels()->u32{return p.hierarchy.x;}
fn totalLevelSlots()->u32{return p.hierarchy.z;}
fn transferStride()->u32{return p.sizes.x;}
fn totalBrickCount()->u32{return p.sizes.y;}
fn dims(l:u32)->vec3u{let s=1u<<l;return (p.dims.xyz+vec3u(s-1u))/s;}
fn levelTable(l:u32)->vec2u{let c=min(l,15u);return vec2u(c>>2u,c&3u);}
fn levelCapacity(l:u32)->u32{let t=levelTable(l);return p.levelCaps[t.x][t.y];}
fn levelBase(l:u32)->u32{let t=levelTable(l);return p.levelBases[t.x][t.y];}
fn brickLevelOffset(l:u32)->u32{let t=levelTable(l);return p.brickOffsets[t.x][t.y];}
fn pageLevelOffset(l:u32)->u32{let t=levelTable(l);return p.pageOffsets[t.x][t.y];}
fn transferLevelOffset(l:u32)->u32{let t=levelTable(l);return p.transferOffsets[t.x][t.y];}
fn at(c:u32,l:u32,s:u32)->u32{return c*totalLevelSlots()+levelBase(l)+s;}
fn loadf(c:u32,l:u32,s:u32)->f32{return bitcast<f32>(state[at(c,l,s)]);}
fn storef(c:u32,l:u32,s:u32,v:f32){state[at(c,l,s)]=bitcast<u32>(v);}
fn rowMapBase()->u32{return 16u;}
fn workBase()->u32{return rowMapBase()+levels()*capacity();}
fn pageWorkBase()->u32{return workBase()+totalLevelSlots();}
fn logicalPageDims(l:u32)->vec3u{return(dims(l)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);}
fn pageDirectoryBase()->u32{return pageWorkBase()+PAGE_RECORD_WORDS*totalLevelSlots();}
fn pageRecord(l:u32,i:u32)->u32{return pageWorkBase()+(levelBase(l)+i)*PAGE_RECORD_WORDS;}
fn pageKey(l:u32,i:u32)->u32{return topology[pageRecord(l,i)];}
fn pageNeighbour(l:u32,i:u32,ordinal:u32)->u32{return topology[pageRecord(l,i)+1u+ordinal];}
fn transferCapacity(l:u32)->u32{return min(transferStride(),levelCapacity(l)*8u);}
fn transferBase()->u32{return pageDirectoryBase()+pageLevelOffset(levels());}
fn rowMap(l:u32,r:u32)->u32{return topology[rowMapBase()+l*capacity()+r];}
fn workSlot(l:u32,i:u32)->u32{return topology[workBase()+levelBase(l)+i];}
fn transferWord(l:u32,i:u32,w:u32)->u32{return transferBase()+transferLevelOffset(l)+i*4u+w;}
fn parentHeadBase(l:u32)->u32{return transferBase()+transferLevelOffset(l)+transferCapacity(l)*4u;}
fn parentTailBase(l:u32)->u32{return parentHeadBase(l)+levelCapacity(l);}
fn fineHeadBase(l:u32)->u32{return parentTailBase(l)+levelCapacity(l);}
fn fineCountBase(l:u32)->u32{return fineHeadBase(l)+levelCapacity(l);}
fn directoryBase()->u32{return transferBase()+transferLevelOffset(levels()-1u);}
fn brickDims(l:u32)->vec3u{return(dims(l)+vec3u(3u))/4u;}
fn brickRecord(l:u32,q:vec3u)->u32{let d=brickDims(l);let b=q/4u;let dense=b.x+d.x*(b.y+d.y*b.z);
 return directoryBase()+16u+(brickLevelOffset(l)+dense)*4u;}
fn rankedSlotsBase()->u32{return directoryBase()+16u+totalBrickCount()*4u;}
fn neighbourBase()->u32{return rankedSlotsBase()+totalLevelSlots();}
fn neighbourAt(k:u32,l:u32,s:u32)->u32{return neighbourBase()+k*totalLevelSlots()+levelBase(l)+s;}
fn localBit(q:vec3u)->u32{let local=q&vec3u(3u);return local.x+4u*local.y+16u*local.z;}
fn decode(key:u32,l:u32)->vec3u{let d=dims(l);let v=key-1u;return vec3u(v%d.x,(v/d.x)%d.y,v/(d.x*d.y));}
fn coordKey(q:vec3u,l:u32)->u32{let d=dims(l);return q.x+d.x*(q.y+d.y*q.z)+1u;}
fn originOf(h:vec4u)->vec3u{return vec3u(h.x%p.dims.x,(h.x/p.dims.x)%p.dims.y,h.x/(p.dims.x*p.dims.y));}
fn pageFor(l:u32,q:vec3u)->u32{let pages=logicalPageDims(l);let v=q/vec3u(8u,8u,4u);
 return topology[pageDirectoryBase()+pageLevelOffset(l)+v.x+pages.x*(v.y+pages.y*v.z)];}

fn canonicalDirection(channel:u32)->vec3i{let d=array<vec3i,18>(
 vec3i(1,0,0),vec3i(-1,0,0),vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1),
 vec3i(1,1,0),vec3i(1,-1,0),vec3i(-1,1,0),vec3i(-1,-1,0),vec3i(1,0,1),vec3i(1,0,-1),
 vec3i(-1,0,1),vec3i(-1,0,-1),vec3i(0,1,1),vec3i(0,1,-1),vec3i(0,-1,1),vec3i(0,-1,-1));return d[channel];}
fn worldDirection(value:vec3i,code:u32)->vec3i{
 let signs=vec3i(select(1,-1,(code&1u)!=0u),select(1,-1,(code&2u)!=0u),select(1,-1,(code&4u)!=0u));
 let q=value*signs;let permutation=(code/8u)%6u;
 if(permutation==0u){return q.xyz;}if(permutation==1u){return q.xzy;}if(permutation==2u){return q.yxz;}
 if(permutation==3u){return q.zxy;}if(permutation==4u){return q.yzx;}return q.zyx;}
${octreeSection63DirectionChannelWGSL}
fn coefficientBase(row:u32)->u32{return acceptedBank()*p.hierarchy.w+row*19u;}
// TRANSCRIPTION NOTE: the accurate operator still carries the eighteen-step
// scan; this is the memoized table the Section 4.3 shell already uses, proved
// equal to that scan's first match by buildSection63DirectionChannelTable's
// four import-time self-checks. Pure lookup replacement (change C4).
fn coefficientForDirection(row:u32,direction:vec3i)->f32{
 let channel=section63ChannelForDirection(metrics[row].transformAndFlags&63u,direction);
 if(channel>=18u){return 0.0;}
 return coefficients[coefficientBase(row)+1u+channel];}
fn diagonalAt(row:u32)->f32{return coefficients[coefficientBase(row)];}
fn validDiagonal(row:u32)->bool{let d=diagonalAt(row);return finite(d)&&d>0.0;}
fn sorted3Sum(values:array<f32,3>)->f32{var sorted=values;
 for(var i=1u;i<3u;i+=1u){let value=sorted[i];var j=i;
  loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}
 return sorted[0]+sorted[1]+sorted[2];}
fn sorted4Sum(values:array<f32,4>)->f32{var sorted=values;
 for(var i=1u;i<4u;i+=1u){let value=sorted[i];var j=i;
  loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}
 return (sorted[0]+sorted[1])+(sorted[2]+sorted[3]);}
fn sorted6Sum(values:array<f32,6>)->f32{var sorted=values;
 for(var i=1u;i<6u;i+=1u){let value=sorted[i];var j=i;
  loop{if(j==0u||sorted[j-1u]<=value){break;}sorted[j]=sorted[j-1u];j-=1u;}sorted[j]=value;}
 return ((sorted[0]+sorted[1])+(sorted[2]+sorted[3]))+(sorted[4]+sorted[5]);}
// Signed axis permutations preserve opposite-direction pairs. Canonicalizing
// three axial pair sums and six diagonal pair sums is therefore invariant to
// every transform code, while doing 18 comparisons instead of sorting all
// eighteen values (153 comparisons). The arithmetic remains grouped by the
// two stencil orbits, so the symmetry fix does not turn each sparse spoke into
// a quadratic amount of work.
fn canonical18Sum(v:array<f32,18>)->f32{
 let axes=array<f32,3>(v[0]+v[1],v[2]+v[3],v[4]+v[5]);
 let diagonals=array<f32,6>(v[6]+v[9],v[7]+v[8],v[10]+v[13],
  v[11]+v[12],v[14]+v[17],v[15]+v[16]);
 return sorted3Sum(axes)+sorted6Sum(diagonals);}
// The eight interpolation corners similarly form four opposite pairs under
// signed axis permutations. Six comparisons replace the full 28-comparison
// corner sort without weakening transform invariance.
fn canonical8Sum(v:array<f32,8>)->f32{
 return sorted4Sum(array<f32,4>(v[0]+v[7],v[1]+v[6],v[2]+v[5],v[3]+v[4]));}
fn sorted64PrefixSum(values:array<f32,64>,n:u32)->f32{
 var sorted=values;for(var i=n;i<64u;i+=1u){sorted[i]=3.402823e38;}
 for(var width=2u;width<=64u;width*=2u){for(var stride=width/2u;stride>0u;stride/=2u){
  for(var i=0u;i<64u;i+=1u){let other=i^stride;if(other<=i){continue;}
   let ascending=(i&width)==0u;let a=sorted[i];let b=sorted[other];
   if((a>b)==ascending){sorted[i]=b;sorted[other]=a;}}}}
 var sum=0.0;for(var i=0u;i<n;i+=1u){sum+=sorted[i];}return sum;}

// --- three pageSlot variants, one per producer, so failureStage stays exact ---
fn opPageSlot(l:u32,page:u32,origin:vec3u,q:vec3u,row:u32)->u32{
 let shape=vec3u(8u,8u,4u);let delta=vec3i(q/shape)-vec3i(origin/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(ERR_ROW,21u,row);return INVALID;}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return INVALID;}let physicalOrigin=decode(topology[pageRecord(l,physical)],l);
 if(any(physicalOrigin/shape!=q/shape)){reportAt(ERR_ROW,22u,row);return INVALID;}
 let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];
 if(slot>=levelCapacity(l)){reportAt(ERR_ROW,23u,row);return INVALID;}return slot;}

fn bandPageSlot(l:u32,page:u32,origin:vec3u,q:vec3u)->u32{
 let shape=vec3u(8u,8u,4u);let delta=vec3i(q/shape)-vec3i(origin/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(ERR_ROW,10u,INVALID);return INVALID;}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));let physical=pageNeighbour(l,page,ordinal);
 if(physical==INVALID){return INVALID;}let physicalOrigin=decode(topology[pageRecord(l,physical)],l);
 if(any(physicalOrigin/shape!=q/shape)){reportAt(ERR_ROW,10u,INVALID);return INVALID;}
 let record=brickRecord(l,q);let bit=localBit(q);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let slot=topology[rankedSlotsBase()+levelBase(l)+topology[record+3u]+rank];
 if(slot>=levelCapacity(l)){reportAt(ERR_ROW,10u,INVALID);return INVALID;}return slot;}

// Dynamically proven class 0 has exactly the six same-level Cartesian faces.
// The accepted SPGrid publication already proved page adjacency and brick
// ranks, so this path omits the general page-origin/catalog chase while
// retaining every slot, flag, owner, and coefficient guard used by A2.
fn regularPageSlot(l:u32,page:u32,q:vec3u,targetCell:vec3u,row:u32)->u32{
 let shape=vec3u(8u,8u,4u);let delta=vec3i(targetCell/shape)-vec3i(q/shape);
 if(any(delta<vec3i(-1))||any(delta>vec3i(1))){reportAt(ERR_ROW,21u,row);return INVALID;}
 let ordinal=u32(delta.x+1)+3u*(u32(delta.y+1)+3u*u32(delta.z+1));
 let physical=pageNeighbour(l,page,ordinal);if(physical==INVALID){return INVALID;}
 if(physical>=pageCount(l)){reportAt(ERR_ROW,22u,row);return INVALID;}
 let record=brickRecord(l,targetCell);let bit=localBit(targetCell);let low=topology[record+1u];let high=topology[record+2u];
 if(((select(low,high,bit>=32u)>>(bit&31u))&1u)==0u){return INVALID;}
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}
 let ranked=topology[record+3u]+rank;if(ranked>=count(l)){reportAt(ERR_ROW,23u,row);return INVALID;}
 let slot=topology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)||state[at(S_KEY,l,slot)]!=coordKey(targetCell,l)){reportAt(ERR_ROW,23u,row);return INVALID;}
 return slot;}

fn directoryLookup(l:u32,q:vec3u)->u32{
 if(l>=levels()||any(q>=dims(l))){return INVALID;}
 let generation=topology[directoryBase()+2u+l];if(generation==0u){reportAt(OVERFLOW,61u,l);return INVALID;}
 let record=brickRecord(l,q);if(topology[record]!=generation){reportAt(OVERFLOW,62u,record);return INVALID;}
 let bit=localBit(q);let word=topology[record+1u+(bit>>5u)];
 let flag=1u<<(bit&31u);if((word&flag)==0u){return INVALID;}
 let low=topology[record+1u];let high=topology[record+2u];
 let lower=select((1u<<(bit&31u))-1u,0xffffffffu,bit>=32u);var rank=countOneBits(low&lower);
 if(bit>=32u){rank+=countOneBits(high&((1u<<(bit-32u))-1u));}let ranked=topology[record+3u]+rank;
 if(ranked>=count(l)||ranked>=levelCapacity(l)){reportAt(OVERFLOW,63u,ranked);return INVALID;}
 let slot=topology[rankedSlotsBase()+levelBase(l)+ranked];
 if(slot>=levelCapacity(l)||state[at(S_KEY,l,slot)]!=coordKey(q,l)){reportAt(OVERFLOW,64u,slot);return INVALID;}
 return slot;}

// ---------------------------------------------------------------------------
// Section 6.3 accurate A2 apply (octreeSPGridAccurateOperatorShader::applyRow)
// ---------------------------------------------------------------------------
fn finerAdjoint(row:u32,q:vec3u,l:u32,x:f32,inCh:u32)->f32{
 if(l==0u){return 0.0;}var children:array<f32,8>;
 for(var child=0u;child<8u;child+=1u){var terms:array<f32,18>;
  let fine=l-1u;let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
  let ghostPage=pageFor(fine,ghostQ);
  if(ghostPage!=INVALID&&ghostPage<levelCapacity(fine)){
   let ghost=opPageSlot(fine,ghostPage,ghostQ,ghostQ,row);
   if(ghost!=INVALID&&(state[at(S_FLAGS,fine,ghost)]&GHOST)!=0u
    &&state[at(S_OWNER,fine,ghost)]==row+1u){
    for(var direction=0u;direction<18u;direction+=1u){let delta=canonicalDirection(direction);
     let activeQ=vec3i(ghostQ)-delta;
     if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}
     let activeSlot=opPageSlot(fine,ghostPage,ghostQ,vec3u(activeQ),row);
     if(activeSlot==INVALID||(state[at(S_FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}
     let encoded=state[at(S_OWNER,fine,activeSlot)];
     if(encoded==0u||encoded>capacity()){reportAt(ERR_ROW,24u,row);continue;}
     let other=encoded-1u;let c=coefficientForDirection(other,delta);
     terms[direction]=select(0.0,c*(x-vload(inCh,other)),c>0.0);}}}
  children[child]=canonical18Sum(terms);}
 return canonical8Sum(children);}

fn applyRow(row:u32,inCh:u32,outCh:u32){
 if(row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)
  ||ch(inCh,row)>=arrayLength(&arena)){reportAt(ERR_ROW,25u,row);return;}
 let h=geometry[row];let m=metrics[row];let base=coefficientBase(row);
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u
  ||base+19u>arrayLength(&coefficients)){reportAt(ERR_AUTHORITY,26u,row);return;}
 let l=firstTrailingBit(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
 if(page==INVALID||page>=levelCapacity(l)){reportAt(ERR_ROW,31u,row);return;}
 let x=vload(inCh,row);var coefficientTerms:array<f32,18>;var directTerms:array<f32,18>;
 for(var channel=0u;channel<18u;channel+=1u){let c=coefficients[base+1u+channel];if(c==0.0){continue;}
  coefficientTerms[channel]=c;
  let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);
  if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(ERR_ROW,27u,row);continue;}
  let slot=opPageSlot(l,page,q,vec3u(targetQ),row);
  if(slot==INVALID){reportAt(ERR_ROW,28u,row);continue;}
  let flags=state[at(S_FLAGS,l,slot)];
  if((flags&MG_ONLY)!=0u){continue;}
  let encoded=state[at(S_OWNER,l,slot)];
  if(encoded==0u||encoded>capacity()){reportAt(ERR_ROW,29u,row);continue;}
  directTerms[channel]=c*(x-vload(inCh,encoded-1u));}
 var value=max(0.0,coefficients[base]-canonical18Sum(coefficientTerms))*x+canonical18Sum(directTerms);
 value+=finerAdjoint(row,q,l,x,inCh);
 if(!finite(value)){reportAt(ERR_NONFINITE,30u,row);}else{vstore(outCh,row,value);}}

fn regularCoefficientTailValid(row:u32)->bool{
 let base=coefficientBase(row);
 if(base+19u>arrayLength(&coefficients)){reportAt(ERR_AUTHORITY,26u,row);return false;}
 for(var channel=6u;channel<18u;channel+=1u){
  if(coefficients[base+1u+channel]!=0.0){reportAt(ERR_AUTHORITY,32u,row);return false;}}
 return true;}

fn applyRegularRow(row:u32,inCh:u32,outCh:u32){
 if(row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)
  ||ch(inCh,row)>=arrayLength(&arena)){reportAt(ERR_ROW,25u,row);return;}
 let h=geometry[row];let m=metrics[row];let base=coefficientBase(row);
 if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u||m.caseId!=0u
  ||(m.transformAndFlags&0x3f00u)!=0u||base+19u>arrayLength(&coefficients)){
  reportAt(ERR_AUTHORITY,26u,row);return;}
 let l=firstTrailingBit(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
 if(page==INVALID||page>=levelCapacity(l)){reportAt(ERR_ROW,31u,row);return;}
 let x=vload(inCh,row);var coefficientTerms:array<f32,18>;var directTerms:array<f32,18>;
 for(var channel=0u;channel<${regularChannelCount}u;channel+=1u){
  let c=coefficients[base+1u+channel];if(c==0.0){continue;}coefficientTerms[channel]=c;
  let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);
  if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){reportAt(ERR_ROW,27u,row);continue;}
  let slot=${regularSlotLookup};
  if(slot==INVALID){reportAt(ERR_ROW,28u,row);continue;}
  let flags=state[at(S_FLAGS,l,slot)];if((flags&MG_ONLY)!=0u){continue;}
  let encoded=state[at(S_OWNER,l,slot)];if(encoded==0u||encoded>capacity()){
   reportAt(ERR_ROW,29u,row);continue;}
  directTerms[channel]=c*(x-vload(inCh,encoded-1u));}
 let value=max(0.0,coefficients[base]-canonical18Sum(coefficientTerms))*x
  +canonical18Sum(directTerms)${regularAdjoint};
 if(!finite(value)){reportAt(ERR_NONFINITE,30u,row);}else{vstore(outCh,row,value);}}

// Class 4 is outside the liquid pressure system. The boundary publisher
// admits it only for the exact identity row; re-prove that invariant here,
// including the dynamics-authored zero RHS, so stale or corrupt membership
// rejects instead of silently dropping pressure coupling.
fn applyIdentityRow(row:u32,inCh:u32,outCh:u32){
 if(row>=capacity()||row>=arrayLength(&geometry)||row>=arrayLength(&metrics)
  ||ch(inCh,row)>=arrayLength(&arena)){reportAt(ERR_ROW,25u,row);return;}
 // Distinct stages per violated term: 32 arena bounds, 33 non-identity
 // diagonal, 34 nonzero dynamics RHS, 35 nonzero tail coefficient. The class-4
 // proof in dynamicRowClass reads the diagonal and tails at classification
 // time but never the RHS, so the split names which input went stale.
 let base=coefficientBase(row);if(base+19u>arrayLength(&coefficients)){reportAt(ERR_AUTHORITY,32u,row);return;}
 if(coefficients[base]!=1.0){reportAt(ERR_AUTHORITY,33u,row);return;}
 if(vload(CH_RHS,row)!=0.0){
  // Stage-34 forensics: the offending staged RHS and the published diagonal,
  // plus the class-4 workset header generation against the accepted
  // generation and bank — a header/generation mismatch means the producers'
  // generation-guarded worklist readers no-oped while this unguarded
  // iteration consumed the stale list.
  atomicStore(&control[8],bitcast<u32>(vload(CH_RHS,row)));
  atomicStore(&control[9],bitcast<u32>(coefficients[base]));
  atomicStore(&control[10],dynamicWorksets[worksetBase(4u)]);
  atomicStore(&control[11],acceptedEpoch());
  atomicStore(&control[12],acceptedBank());
  atomicStore(&control[13],worksetCount(4u));
  // Words 14/15 split the two surviving theories: sweep the whole class-4
  // list and count rows with nonzero staged RHS. Wholesale (≈count) means the
  // producer's zero dispatch never landed on this row space; isolated (1–2)
  // means a per-row wet/dry disagreement between divergence and classifier.
  var dirty=0u;var firstOther=0xFFFFFFFFu;let n4=worksetCount(4u);
  for(var item=0u;item<n4;item+=1u){let r=worksetRow(4u,item);
   if(r<capacity()&&ch(CH_RHS,r)<arrayLength(&arena)&&vload(CH_RHS,r)!=0.0){
    dirty+=1u;if(r!=row&&firstOther==0xFFFFFFFFu){firstOther=r;}}}
  atomicStore(&control[14],dirty);
  atomicStore(&control[15],firstOther);
  reportAt(ERR_AUTHORITY,34u,row);return;}
 for(var channel=0u;channel<18u;channel+=1u){if(coefficients[base+1u+channel]!=0.0){
  reportAt(ERR_AUTHORITY,35u,row);return;}}
 let value=vload(inCh,row);if(!finite(value)){reportAt(ERR_NONFINITE,32u,row);}
 else{vstore(outCh,row,value);}}

// TRANSCRIPTION NOTE: the hierarchical exact apply dispatches the five
// disjoint accepted row classes, whose lists partition [0, rows()) exactly
// (finalizeStructuredPublication scatters every row into exactly one class).
// applyRow is a pure gather into its own output slot, so iterating the rows
// directly is set-identical and order-independent. P0 verifies the partition
// against the staged class headers and fails closed if it does not hold.
fn applyAllRows(lane:u32,liveRows:u32,inCh:u32,outCh:u32){
 ${allRowsApply}}

${bandRowsNote}fn applyBandRows(lane:u32,bandCount:u32,inCh:u32,outCh:u32){
 for(var item=lane;item<bandCount;item+=LANES){
  let row=uload(CH_BANDLIST,item);
  if(!stopped()&&row!=INVALID&&row<rows()){${bandRowApply}}}}

// ---------------------------------------------------------------------------
// Section 4.3 boundary-band classification (encodeSetup's six stages)
// ---------------------------------------------------------------------------
fn validSection63Row(row:u32)->bool{
 if(row>=rows()||row>=arrayLength(&metrics)){return false;}
 let base=coefficientBase(row);if(base+19u>arrayLength(&coefficients)){return false;}
 let m=metrics[row];if(m.error!=0u||(m.transformAndFlags&0x80000000u)==0u){return false;}
 for(var channel=0u;channel<19u;channel+=1u){let c=coefficients[base+channel];
  if(!finite(c)||c<0.0){return false;}}
 return coefficients[base]>0.0;}
fn section63Class(row:u32)->u32{
 let base=coefficientBase(row);var terms:array<f32,18>;
 for(var channel=0u;channel<18u;channel+=1u){terms[channel]=coefficients[base+1u+channel];}
 let off=canonical18Sum(terms);
 let physicalBoundary=(metrics[row].transformAndFlags&0x3f00u)!=0u;
 let boundary=physicalBoundary||coefficients[base]>off+max(1e-6,1e-5*abs(off));
 let transition=metrics[row].caseId!=0u;
 return select(select(0u,2u,boundary),select(1u,3u,boundary),transition);}
fn bandAt(row:u32,useB:bool)->u32{return select(uload(CH_BANDA,row),uload(CH_BANDB,row),useB);}
fn dilatedBand(row:u32,useB:bool)->u32{
 var value=bandAt(row,useB);let h=geometry[row];let m=metrics[row];
 let l=firstTrailingBit(h.y);let q=originOf(h)/(1u<<l);let page=pageFor(l,q);
 let base=coefficientBase(row);
 for(var channel=0u;channel<18u&&value==0u;channel+=1u){
  if(coefficients[base+1u+channel]==0.0){continue;}
  let targetQ=vec3i(q)+worldDirection(canonicalDirection(channel),m.transformAndFlags&63u);
  if(any(targetQ<vec3i(0))||any(targetQ>=vec3i(dims(l)))){continue;}
  let slot=bandPageSlot(l,page,q,vec3u(targetQ));
  if(slot==INVALID||(state[at(S_FLAGS,l,slot)]&MG_ONLY)!=0u){continue;}
  let encoded=state[at(S_OWNER,l,slot)];
  if(encoded>0u&&encoded<=rows()){value=max(value,bandAt(encoded-1u,useB));}}
 if(l>0u&&value==0u){let fine=l-1u;
  for(var child=0u;child<8u&&value==0u;child+=1u){
   let ghostQ=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
   let ghostPage=pageFor(fine,ghostQ);if(ghostPage==INVALID){continue;}
   let ghost=bandPageSlot(fine,ghostPage,ghostQ,ghostQ);
   if(ghost==INVALID||(state[at(S_FLAGS,fine,ghost)]&GHOST)==0u
    ||state[at(S_OWNER,fine,ghost)]!=row+1u){continue;}
   for(var candidate=0u;candidate<18u&&value==0u;candidate+=1u){
    let delta=canonicalDirection(candidate);let activeQ=vec3i(ghostQ)-delta;
    if(any(activeQ<vec3i(0))||any(activeQ>=vec3i(dims(fine)))){continue;}
    let activeSlot=bandPageSlot(fine,ghostPage,ghostQ,vec3u(activeQ));
    if(activeSlot==INVALID||(state[at(S_FLAGS,fine,activeSlot)]&ACTIVE)==0u){continue;}
    let encoded=state[at(S_OWNER,fine,activeSlot)];
    if(encoded>0u&&encoded<=rows()&&coefficientForDirection(encoded-1u,delta)>0.0){
     value=max(value,bandAt(encoded-1u,useB));}}}}
 return value;}

// ---------------------------------------------------------------------------
// Section 4.3 damped-Jacobi band shell
// ---------------------------------------------------------------------------
fn hybridValue(row:u32,useB:bool)->f32{return select(vload(CH_HA,row),vload(CH_HB,row),useB);}
fn smoothValue(row:u32,useB:bool,rhsCh:u32)->f32{
 let current=hybridValue(row,useB);
 if(uload(CH_BANDB,row)==0u){return current;}
 let next=current+p.numerics.w*(vload(rhsCh,row)-vload(CH_OPIMG,row))/diagonalAt(row);
 if(!finite(next)){reportAt(ERR_NONFINITE,11u,row);return current;}
 return next;}
fn smoothZeroValue(row:u32,rhsCh:u32)->f32{
 if(uload(CH_BANDB,row)==0u){return 0.0;}
 if(!validDiagonal(row)){reportAt(ERR_ROW,10u,row);return 0.0;}
 let next=p.numerics.w*vload(rhsCh,row)/diagonalAt(row);
 if(!finite(next)){reportAt(ERR_NONFINITE,11u,row);return 0.0;}
 return next;}

// ---------------------------------------------------------------------------
// First-order SPGrid V-cycle
// ---------------------------------------------------------------------------
fn smoothable(l:u32,s:u32)->bool{return(state[at(S_FLAGS,l,s)]&GHOST)==0u;}
fn applied(l:u32,slot:u32,source:u32)->f32{
 let base=levelBase(l);let span=totalLevelSlots();let columnBase=neighbourBase()+base+slot;
${appliedBody}}
fn chebyshevWeight(l:u32,phase:u32,degree:u32)->f32{
 let upper=loadf(S_SPECTRAL,l,0u);let lower=upper/30.0;
 if(!(lower>0.0)||!(upper>lower)||!finite(upper)){reportAt(NONPOSITIVE,76u,l);return 0.0;}
 let centre=0.5*(upper+lower);let radius=0.5*(upper-lower);
 return 1.0/(centre-radius*cos(3.141592653589793*(2.0*f32(phase)+1.0)/(2.0*f32(degree))));}
fn sparseSmoothPhase(l:u32,source:u32,destination:u32,phase:u32,degree:u32,lane:u32){
 let weight=chebyshevWeight(l,phase,degree);let n=count(l);
 for(var i=lane;i<n;i+=LANES){let slot=workSlot(l,i);let value=loadf(source,l,slot);
  if(!smoothable(l,slot)){storef(destination,l,slot,value);continue;}
  let d=loadf(S_DIAG,l,slot);if(!(d>0.0)){reportAt(NONPOSITIVE,79u,slot);continue;}
  let next=value+weight*(loadf(S_RHS,l,slot)-applied(l,slot,source))/d;
  if(!finite(next)){reportAt(NONFINITE,80u,slot);}else{storef(destination,l,slot,next);}}}
fn smoothLevel(l:u32,reverse:bool,lane:u32){
 let degree=p.shape.y;
 for(var step=0u;step<degree;step+=1u){let phase=select(step,degree-1u-step,reverse);
  if((step&1u)==0u){sparseSmoothPhase(l,S_A,S_B,phase,degree,lane);}
  else{sparseSmoothPhase(l,S_B,S_A,phase,degree,lane);}
  storageBarrier();workgroupBarrier();
  if(!reverse&&step==0u){${captureVcyclePhase0}}}}

fn correctionTransfer(l:u32,fine:u32,corner:u32)->TransferTarget{
 if(l+1u>=levels()||fine>=levelCapacity(l)){reportAt(OVERFLOW,81u,fine);return TransferTarget(INVALID,0.0);}
 let first=topology[fineHeadBase(l)+fine];let n=topology[fineCountBase(l)+fine];
 if(first==INVALID||corner>=n||first+corner>=transferCount(l)){
  reportAt(OVERFLOW,82u,fine);return TransferTarget(INVALID,0.0);}
 let record=first+corner;
 if(topology[transferWord(l,record,0u)]!=fine){reportAt(OVERFLOW,83u,fine);return TransferTarget(INVALID,0.0);}
 let coarse=topology[transferWord(l,record,1u)];
 let weight=bitcast<f32>(topology[transferWord(l,record,2u)]);
 if(coarse>=levelCapacity(l+1u)||!finite(weight)){reportAt(OVERFLOW,84u,fine);return TransferTarget(INVALID,0.0);}
 return TransferTarget(coarse,weight);}

fn restrictLevel(l:u32,lane:u32){
 if(l+1u>=levels()){return;}
 let n=count(l+1u);
 for(var i=lane;i<n;i+=LANES){
  if(stopped()){continue;}
  let coarse=workSlot(l+1u,i);var terms:array<f32,64>;var termCount=0u;
  var record=topology[parentHeadBase(l)+coarse];var bad=false;
  for(var visited=0u;record!=INVALID&&visited<transferCount(l);visited+=1u){
   if(record>=transferCount(l)){reportAt(OVERFLOW,85u,coarse);bad=true;break;}
   let fine=topology[transferWord(l,record,0u)];let parent=topology[transferWord(l,record,1u)];
   if(parent!=coarse){reportAt(OVERFLOW,86u,coarse);bad=true;break;}
   let ghost=(state[at(S_FLAGS,l,fine)]&GHOST)!=0u;
   let product=applied(l,fine,S_A);
   let residual=select(-product,loadf(S_RHS,l,fine)-product,!ghost);
   if(termCount>=64u){reportAt(OVERFLOW,87u,coarse);bad=true;break;}
   terms[termCount]=bitcast<f32>(topology[transferWord(l,record,2u)])*residual;termCount+=1u;
   record=topology[transferWord(l,record,3u)];}
  if(bad){continue;}
  let sum=sorted64PrefixSum(terms,termCount);
  if(record!=INVALID||!finite(sum)){reportAt(OVERFLOW,87u,coarse);continue;}
  storef(S_RHS,l+1u,coarse,loadf(S_RHS,l+1u,coarse)+sum);}}

fn exactBottom(l:u32,lane:u32){
 if(lane!=0u||stopped()){return;}
 if(count(l)!=1u){reportAt(NONPOSITIVE,88u,l);return;}
 let s=workSlot(l,0u);let d=loadf(S_DIAG,l,s);
 if(!(d>0.0)){reportAt(NONPOSITIVE,89u,s);return;}
 let x=loadf(S_RHS,l,s)/d;
 if(!finite(x)){reportAt(NONFINITE,90u,s);}else{storef(S_A,l,s,x);}}

fn prolongLevel(l:u32,lane:u32){
 let n=count(l);
 for(var i=lane;i<n;i+=LANES){
  if(stopped()){continue;}
  let fine=workSlot(l,i);let ghost=(state[at(S_FLAGS,l,fine)]&GHOST)!=0u;
  let targetCount=select(8u,1u,ghost);var value=select(loadf(S_A,l,fine),0.0,ghost);
  var terms:array<f32,8>;var bad=false;
  for(var corner=0u;corner<targetCount;corner+=1u){
   let transfer=correctionTransfer(l,fine,corner);
   if(transfer.coarse==INVALID){bad=true;break;}
   terms[corner]=transfer.weight*loadf(S_A,l+1u,transfer.coarse);}
  if(bad){continue;}
  value+=canonical8Sum(terms);
  if(!finite(value)){reportAt(NONFINITE,91u,fine);}else{storef(S_A,l,fine,value);}}}

fn vcycleCorrection(lane:u32,liveRows:u32,rhsCh:u32,corrCh:u32){
 // Every loop below whose body contains a barrier is bounded by a
 // workgroupUniformLoad so Tint's uniformity analysis sees uniform control
 // flow without relying on inter-procedural parameter uniformity.
 let levelCount=workgroupUniformLoad(&wLevels);
 for(var r=lane;r<liveRows;r+=LANES){if(!stopped()){vstore(corrCh,r,0.0);}}
 for(var l=0u;l<levelCount;l+=1u){
  let n=count(l);
  for(var i=lane;i<n;i+=LANES){
   if(stopped()){continue;}
   let s=workSlot(l,i);
   for(var c=S_RHS;c<=S_B;c+=1u){storef(c,l,s,0.0);}}}
 storageBarrier();workgroupBarrier();
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  let v=vload(rhsCh,r);let native=firstTrailingBit(geometry[r].y);
  if(!finite(v)){reportAt(NONFINITE,73u,r);}else{storef(S_RHS,native,rowMap(native,r),v);}}
 storageBarrier();workgroupBarrier();
 for(var l=0u;l+1u<levelCount;l+=1u){
  smoothLevel(l,false,lane);
  ${captureVcyclePresmooth}
  restrictLevel(l,lane);
  storageBarrier();workgroupBarrier();
  ${captureVcycleRestrict}}
 exactBottom(levelCount-1u,lane);
 storageBarrier();workgroupBarrier();
 for(var back=0u;back+1u<levelCount;back+=1u){
  let l=levelCount-2u-back;
  prolongLevel(l,lane);
  storageBarrier();workgroupBarrier();
  ${captureVcycleProlong}
  smoothLevel(l,true,lane);
  ${captureVcyclePostsmooth}}
 storageBarrier();workgroupBarrier();
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  let native=firstTrailingBit(geometry[r].y);
  let v=loadf(S_A,native,rowMap(native,r));
  if(!finite(v)){reportAt(NONFINITE,92u,r);}else{vstore(corrCh,r,v);}}
 storageBarrier();workgroupBarrier();}

// ---------------------------------------------------------------------------
// Section 4.3 correction: J^k(0,q) -> V1 -> J^k, exactly as encodeCorrection
// ---------------------------------------------------------------------------
fn section43Correction(lane:u32,liveRows:u32,bandCount:u32,rhsCh:u32,corrCh:u32){
 let sweeps=p.shape.z;
 ${captureSection43Inputs}
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  vstore(CH_HA,r,0.0);vstore(CH_HB,r,smoothZeroValue(r,rhsCh));}
 storageBarrier();workgroupBarrier();
 ${captureSection43Initial}
 for(var iteration=1u;iteration<sweeps;iteration+=1u){
  let fromB=(iteration&1u)==1u;
  applyBandRows(lane,bandCount,select(CH_HA,CH_HB,fromB),CH_OPIMG);
  storageBarrier();workgroupBarrier();
  for(var item=lane;item<bandCount;item+=LANES){
   let row=uload(CH_BANDLIST,item);
   if(!stopped()&&row!=INVALID&&row<liveRows){
    if(fromB){vstore(CH_HA,row,smoothValue(row,true,rhsCh));}
    else{vstore(CH_HB,row,smoothValue(row,false,rhsCh));}}}
  storageBarrier();workgroupBarrier();
  ${captureSection43First}}
 ${captureSection43Presmooth}
 applyAllRows(lane,liveRows,CH_HA,CH_OPIMG);
 storageBarrier();workgroupBarrier();
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  let value=vload(rhsCh,r)-vload(CH_OPIMG,r);
  if(!finite(value)){reportAt(ERR_NONFINITE,12u,r);}else{vstore(CH_INNER_RHS,r,value);}}
 storageBarrier();workgroupBarrier();
 ${captureSection43InnerRhs}
 vcycleCorrection(lane,liveRows,CH_INNER_RHS,CH_INNER_CORR);
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  let value=vload(CH_HA,r)+vload(CH_INNER_CORR,r);
  if(!finite(value)){reportAt(ERR_NONFINITE,13u,r);}
  else{vstore(CH_HA,r,value);vstore(CH_HB,r,value);}}
 storageBarrier();workgroupBarrier();
 for(var iteration=0u;iteration<sweeps;iteration+=1u){
  let fromB=(iteration&1u)==0u;
  applyBandRows(lane,bandCount,select(CH_HA,CH_HB,fromB),CH_OPIMG);
  storageBarrier();workgroupBarrier();
  for(var item=lane;item<bandCount;item+=LANES){
   let row=uload(CH_BANDLIST,item);
   if(!stopped()&&row!=INVALID&&row<liveRows){
    if(fromB){vstore(CH_HA,row,smoothValue(row,true,rhsCh));}
    else{vstore(CH_HB,row,smoothValue(row,false,rhsCh));}}}
  storageBarrier();workgroupBarrier();}
 for(var r=lane;r<liveRows;r+=LANES){
  if(stopped()){continue;}
  let value=vload(CH_HB,r);
  if(!finite(value)){reportAt(ERR_NONFINITE,14u,r);}else{vstore(corrCh,r,value);}}
 storageBarrier();workgroupBarrier();}

// ---------------------------------------------------------------------------
// Compensated reductions. The tree below is the exact width-halving
// shared-memory merge the hierarchical reduction already performs over its
// 128 slots; the persistent kernel emulates 128-lane virtual workgroups so the
// association order of every f32 addition is unchanged.
// ---------------------------------------------------------------------------
fn zeroMergedScalars()->MergedScalars{
 let zero=CompensatedF32(0.0,0.0);
 return MergedScalars(zero,zero,zero,zero);}
fn mergeScalars(a:MergedScalars,b:MergedScalars)->MergedScalars{
 return MergedScalars(
  mergeCompensatedF32(a.gamma,b.gamma),
  mergeCompensatedF32(a.delta,b.delta),
  mergeCompensatedF32(a.rr,b.rr),
  mergeCompensatedF32(a.bb,b.bb));}
fn storePartial(vg:u32,m:MergedScalars){
 let b=partialBase()+vg*8u;
 arena[b]=bitcast<u32>(m.gamma.hi);arena[b+1u]=bitcast<u32>(m.gamma.lo);
 arena[b+2u]=bitcast<u32>(m.delta.hi);arena[b+3u]=bitcast<u32>(m.delta.lo);
 arena[b+4u]=bitcast<u32>(m.rr.hi);arena[b+5u]=bitcast<u32>(m.rr.lo);
 arena[b+6u]=bitcast<u32>(m.bb.hi);arena[b+7u]=bitcast<u32>(m.bb.lo);}
fn loadPartial(vg:u32)->MergedScalars{
 let b=partialBase()+vg*8u;
 return MergedScalars(
  CompensatedF32(bitcast<f32>(arena[b]),bitcast<f32>(arena[b+1u])),
  CompensatedF32(bitcast<f32>(arena[b+2u]),bitcast<f32>(arena[b+3u])),
  CompensatedF32(bitcast<f32>(arena[b+4u]),bitcast<f32>(arena[b+5u])),
  CompensatedF32(bitcast<f32>(arena[b+6u]),bitcast<f32>(arena[b+7u])));}
fn reductionTree(lane:u32,local:MergedScalars){
 if(lane<REDUCTION_LANES){merged[lane]=local;}
 for(var width=REDUCTION_LANES/2u;width>0u;width>>=1u){
  workgroupBarrier();
  if(lane<width){merged[lane]=mergeScalars(merged[lane],merged[lane+width]);}}
 workgroupBarrier();}

fn reduceMerged(lane:u32,liveRows:u32,initial:bool){
 let livePartials=workgroupUniformLoad(&wPartials);
 for(var vg=0u;vg<livePartials;vg+=1u){
  var local=zeroMergedScalars();
  let row=vg*REDUCTION_LANES+lane;
  if(lane<REDUCTION_LANES&&row<liveRows&&!stopped()){
   let r=vload(CH_RESIDUAL,row);
   let u=vload(CH_PRE,row);
   let w=vload(CH_PREIMG,row);
   let b=-vload(CH_RHS,row);
   if(!finite(r)||!finite(u)||(initial&&!finite(w))){reportAt(ERR_NONFINITE,4u,row);}
   else{
    local.gamma=addCompensatedF32(local.gamma,r*u);
    if(initial){local.delta=addCompensatedF32(local.delta,u*w);}
    local.rr=addCompensatedF32(local.rr,r*r);
    if(initial){local.bb=addCompensatedF32(local.bb,b*b);}}}
  reductionTree(lane,local);
  if(lane==0u){storePartial(vg,merged[0]);}
  storageBarrier();workgroupBarrier();}}

fn reduceCurvature(lane:u32,liveRows:u32){
 let livePartials=workgroupUniformLoad(&wPartials);
 for(var vg=0u;vg<livePartials;vg+=1u){
  var local=zeroMergedScalars();
  let row=vg*REDUCTION_LANES+lane;
  if(lane<REDUCTION_LANES&&row<liveRows&&!stopped()){
   let d=vload(CH_DIR,row);
   let image=vload(CH_DIRIMG,row);
   if(!finite(d)||!finite(image)){reportAt(ERR_NONFINITE,15u,row);}
   else{local.delta=addCompensatedF32(local.delta,d*image);}}
  reductionTree(lane,local);
  if(lane==0u){storePartial(vg,merged[0]);}
  storageBarrier();workgroupBarrier();}}

fn mergePartials(lane:u32){
 let livePartials=workgroupUniformLoad(&wPartials);
 var local=zeroMergedScalars();
 if(lane<REDUCTION_LANES){
  for(var partial=lane;partial<livePartials;partial+=REDUCTION_LANES){
   local=mergeScalars(local,loadPartial(partial));}}
 reductionTree(lane,local);}

// The persistent path has no indirect outer records to zero; these keep the
// GPU-authored zeroedDispatches accounting word identical to the
// hierarchical run so a lockstep A/B compares the full control record.
fn accountZeroAll(){atomicAdd(&control[5],4u*p.shape.x);}
fn accountZeroRemaining(iteration:u32){
 var total=2u;
 if(iteration+1u<p.shape.x){total+=4u*(p.shape.x-1u-iteration);}
 atomicAdd(&control[5],total);}

// ---------------------------------------------------------------------------
// finishMergedReduction's scalar tail, verbatim, with the indirect-record
// zeroing replaced by the workgroup halt flag.
// ---------------------------------------------------------------------------
fn finishMerged(){
 let initial=atomicLoad(&control[3])==0u;
 if(failed()){
  if(initial){accountZeroAll();}else{accountZeroRemaining(atomicLoad(&control[2]));}
  return;}
 if(atomicLoad(&control[1])!=0u){return;}
 let total=merged[0];
 let gamma=compensatedValue(total.gamma);
 let delta=compensatedValue(total.delta);
 let rr=compensatedValue(total.rr);
 if(!finite(gamma)||!finite(delta)||!finite(rr)||rr<0.0){
  reportAt(ERR_NONFINITE,5u,INVALID);
  if(initial){accountZeroAll();}else{accountZeroRemaining(atomicLoad(&control[2]));}
  return;}
 if(initial){
  let bb=compensatedValue(total.bb);
  if(!finite(bb)||bb<0.0){reportAt(ERR_NONFINITE,5u,INVALID);accountZeroAll();return;}
  storePair(8u,total.bb);}
 storePair(10u,total.rr);
 atomicAdd(&control[3],1u);
 let bb=max(compensatedValue(pairAt(8u)),p.numerics.z);
 let threshold=max(p.numerics.y*p.numerics.y,p.numerics.x*p.numerics.x*bb);
 if(!initial){atomicAdd(&control[2],1u);}
 if(rr<=threshold){
  atomicStore(&control[1],1u);
  if(initial){accountZeroAll();}else{accountZeroRemaining(atomicLoad(&control[2])-1u);}
  return;}
 if(!(gamma>0.0)){
  reportAt(ERR_NONPOSITIVE_PRE,5u,INVALID);
  if(initial){accountZeroAll();}else{accountZeroRemaining(atomicLoad(&control[2])-1u);}
  return;}
 var alpha=0.0;
 if(initial){
  if(!(delta>0.0)){reportAt(ERR_NONPOSITIVE_CURVATURE,5u,INVALID);accountZeroAll();return;}
  alpha=gamma/delta;
  storePair(18u,CompensatedF32(0.0,0.0));
 }else{
  let previousGamma=compensatedValue(pairAt(12u));
  let beta=gamma/previousGamma;
  if(!(previousGamma>0.0)||!finite(beta)||beta<0.0){
   reportAt(ERR_NONPOSITIVE_PRE,5u,INVALID);
   accountZeroRemaining(atomicLoad(&control[2])-1u);
   return;}
  storePair(18u,CompensatedF32(beta,0.0));}
 if(initial&&(!finite(alpha)||!(alpha>0.0))){
  reportAt(ERR_NONFINITE,5u,INVALID);accountZeroAll();return;}
 storePair(12u,total.gamma);
 storePair(14u,total.delta);
 storePair(16u,CompensatedF32(alpha,0.0));
 if(!initial&&atomicLoad(&control[2])>=p.shape.x){
  reportAt(ERR_NONCONVERGENCE,6u,INVALID);
  accountZeroRemaining(p.shape.x-1u);}}

fn finishCurvature(){
 if(stopped()){return;}
 atomicAdd(&control[3],1u);
 let curvature=merged[0].delta;
 let direct=compensatedValue(curvature);
 let gamma=compensatedValue(pairAt(12u));
 if(!finite(direct)||!(direct>0.0)){
  reportAt(ERR_NONPOSITIVE_CURVATURE,15u,INVALID);
  accountZeroRemaining(atomicLoad(&control[2])-1u);
  return;}
 let alpha=gamma/direct;
 if(!finite(alpha)||!(alpha>0.0)){
  reportAt(ERR_NONFINITE,15u,INVALID);
  accountZeroRemaining(atomicLoad(&control[2])-1u);
  return;}
 storePair(14u,curvature);
 storePair(16u,CompensatedF32(alpha,0.0));}

// ---------------------------------------------------------------------------
// Deterministic band compaction. The shipped compactBandIntersections appends
// through an atomicAdd, so its list order is scheduling-dependent; every
// consumer (the merged-band apply and the ping-pong smooth) writes only its
// own row, so the order is immaterial and this stable ascending compaction is
// set-identical and reproducible.
// ---------------------------------------------------------------------------
fn compactBand(lane:u32,liveRows:u32){
 let chunk=(liveRows+LANES-1u)/LANES;
 let first=min(lane*chunk,liveRows);
 let last=min(first+chunk,liveRows);
 var local=0u;
 for(var r=first;r<last;r+=1u){if(uload(CH_BANDB,r)!=0u){local+=1u;}}
 scan[lane]=local;
 workgroupBarrier();
 for(var width=1u;width<LANES;width<<=1u){
  var add=0u;
  if(lane>=width){add=scan[lane-width];}
  workgroupBarrier();
  scan[lane]+=add;
  workgroupBarrier();}
 var rank=0u;
 if(lane>0u){rank=scan[lane-1u];}
 let total=scan[LANES-1u];
 for(var r=first;r<last;r+=1u){
  if(uload(CH_BANDB,r)==0u){continue;}
  if(rank<capacity()){ustore(CH_BANDLIST,rank,r);}else{reportAt(ERR_ROW,10u,r);}
  rank+=1u;}
 if(lane==0u){wBand=min(total,capacity());}
 storageBarrier();workgroupBarrier();}

// ---------------------------------------------------------------------------
// The single persistent entry point.
// ---------------------------------------------------------------------------
@compute @workgroup_size(256)
fn persistentMGPCG(@builtin(local_invocation_index) lane:u32${activityParameters}){${stagedSmootherPrologue}${activityEnter}
 // --- P0: authority validation, uniform bootstrap -------------------------
 if(lane==0u){
 atomicStore(&control[7],INVALID);
  // Persistent-executor marker so a control readback can tell the two paths
  // apart without changing any word the snapshot ring or tripwires read.
  atomicStore(&control[21],0x50455253u);
  // On rejection these reserved census words retain an exact authority
  // snapshot and validation code. A successful solve overwrites them with
  // the live hybrid work census below.
  atomicStore(&control[22],acc(0u));atomicStore(&control[23],acc(1u));
  atomicStore(&control[24],acc(2u));atomicStore(&control[25],acc(3u));
  atomicStore(&control[26],acc(4u));atomicStore(&control[27],acc(5u));
  atomicStore(&control[28],acc(6u));atomicStore(&control[29],p.worksets.x);
  atomicStore(&control[30],p.worksets.y);atomicStore(&control[31],0u);
  if(acc(2u)==0u||acc(2u)>capacity()){reportAt(ERR_AUTHORITY,1u,INVALID);}
  else if(acc(2u)>MAX_LIVE_ROWS){reportAt(ERR_ROW,1u,acc(2u));}
  else {let authorityError=validateAcceptedRowPartition();
   if(authorityError!=0u){atomicStore(&control[31],authorityError);reportAt(ERR_AUTHORITY,1u,INVALID);}}
  atomicStore(&control[4],rows());
  if(!failed()){
   let regular=worksetCount(0u);let identity=worksetCount(4u);let power=rows()-regular-identity;
   atomicStore(&control[22],regular);atomicStore(&control[23],identity);
   let liquid=regular+power;
   atomicStore(&control[24],power);atomicStore(&control[25],liquid);
   atomicStore(&control[26],power);atomicStore(&control[27],18u*liquid);
   atomicStore(&control[28],18u*power);
   atomicStore(&control[29],acceptedEpoch());
   atomicStore(&control[30],5u);atomicStore(&control[31],0x48344234u);}
  wRows=rows();
  wLevels=levels();
  wPartials=(rows()+REDUCTION_LANES-1u)/REDUCTION_LANES;
  wBand=0u;
  wInitial=0u;
  wHalt=select(0u,1u,failed());}
 storageBarrier();workgroupBarrier();
 let liveRows=workgroupUniformLoad(&wRows);
 var halt=workgroupUniformLoad(&wHalt);

 if(halt==0u){
  // --- P1: initializeState ------------------------------------------------
  for(var row=lane;row<liveRows;row+=LANES){
   if(failed()){continue;}
   // RHS and seed were copied before the live count was GPU-visible to the
   // host. Read them once from their stable staging offsets, then place them
   // in the compact channel layout used by every subsequent phase.
   let seed=stagedVload(CH_SEED,row);
   let rhs=stagedVload(CH_RHS,row);
   let diagonal=diagonalAt(row);
   if(!finite(diagonal)||diagonal<=0.0||!finite(rhs)||!finite(seed)){
    reportAt(ERR_ROW,2u,row);continue;}
   vstore(CH_RHS,row,rhs);
   vstore(CH_SEED,row,seed);
   vstore(CH_PRESSURE,row,seed);
   vstore(CH_RESIDUAL,row,0.0);}
  storageBarrier();workgroupBarrier();

  // --- P2: A2(pressure) -> directionImage ---------------------------------
  applyAllRows(lane,liveRows,CH_PRESSURE,CH_DIRIMG);
  storageBarrier();workgroupBarrier();

  // --- P3: formInitialResidual --------------------------------------------
  for(var row=lane;row<liveRows;row+=LANES){
   if(failed()){continue;}
   let value=-vload(CH_RHS,row)-vload(CH_DIRIMG,row);
   if(!finite(value)){reportAt(ERR_NONFINITE,3u,row);}else{vstore(CH_RESIDUAL,row,value);${captureInitialResidual}}}
  storageBarrier();workgroupBarrier();

  // --- P4: Section 4.3 band publication (encodeSetup) ----------------------
  ${bandInitialization}
  storageBarrier();workgroupBarrier();
  for(var row=lane;row<liveRows;row+=LANES){
   if(!stopped()){ustore(CH_BANDB,row,dilatedBand(row,false));}}
  storageBarrier();workgroupBarrier();
  for(var row=lane;row<liveRows;row+=LANES){
   if(!stopped()){ustore(CH_BANDA,row,dilatedBand(row,true));}}
  storageBarrier();workgroupBarrier();
  for(var row=lane;row<liveRows;row+=LANES){
   if(!stopped()){ustore(CH_BANDB,row,dilatedBand(row,false));}}
  storageBarrier();workgroupBarrier();${bandRowClassMap}
  compactBand(lane,liveRows);${bandRowCensus}}
 let bandCount=workgroupUniformLoad(&wBand);

 if(lane==0u){wHalt=select(0u,1u,failed());}
 storageBarrier();workgroupBarrier();
 halt=workgroupUniformLoad(&wHalt);

 if(halt==0u){
  // --- P5: M(residual) -> preconditioned ----------------------------------
  section43Correction(lane,liveRows,bandCount,CH_RESIDUAL,CH_PRE);
  ${captureInitialPreconditioned}
  // --- P6: A2(preconditioned) -> preconditionedImage ----------------------
  applyAllRows(lane,liveRows,CH_PRE,CH_PREIMG);
  storageBarrier();workgroupBarrier();

  // --- P7: initial merged reduction and its scalar finish ------------------
  if(lane==0u){wInitial=select(0u,1u,atomicLoad(&control[3])==0u);}
  storageBarrier();workgroupBarrier();
  reduceMerged(lane,liveRows,workgroupUniformLoad(&wInitial)!=0u);
  mergePartials(lane);
  if(lane==0u){finishMerged();wHalt=select(0u,1u,stopped());}
  storageBarrier();workgroupBarrier();
  halt=workgroupUniformLoad(&wHalt);}

 if(halt==0u){
  // --- P8: initializeDirections -------------------------------------------
  for(var row=lane;row<liveRows;row+=LANES){
   if(stopped()){continue;}
   vstore(CH_DIR,row,vload(CH_PRE,row));
   vstore(CH_DIRIMG,row,vload(CH_PREIMG,row));}
  storageBarrier();workgroupBarrier();

  // --- P9: the outer CG loop ----------------------------------------------
  for(var iteration=0u;iteration<${iterations}u;iteration+=1u){
   if(halt!=0u){break;}
   for(var row=lane;row<liveRows;row+=LANES){
    if(stopped()){continue;}
    let alpha=compensatedValue(pairAt(16u));
    let nextPressure=vload(CH_PRESSURE,row)+alpha*vload(CH_DIR,row);
    let nextResidual=vload(CH_RESIDUAL,row)-alpha*vload(CH_DIRIMG,row);
    if(!finite(nextPressure)||!finite(nextResidual)){reportAt(ERR_NONFINITE,7u,row);continue;}
    vstore(CH_PRESSURE,row,nextPressure);
    vstore(CH_RESIDUAL,row,nextResidual);}
   storageBarrier();workgroupBarrier();

   section43Correction(lane,liveRows,bandCount,CH_RESIDUAL,CH_PRE);

   if(lane==0u){wInitial=select(0u,1u,atomicLoad(&control[3])==0u);}
   storageBarrier();workgroupBarrier();
   reduceMerged(lane,liveRows,workgroupUniformLoad(&wInitial)!=0u);
   mergePartials(lane);
   if(lane==0u){finishMerged();wHalt=select(0u,1u,stopped());}
   storageBarrier();workgroupBarrier();
   halt=workgroupUniformLoad(&wHalt);
   if(halt!=0u){break;}

   for(var row=lane;row<liveRows;row+=LANES){
    if(stopped()){continue;}
    let beta=compensatedValue(pairAt(18u));
    let nextDirection=vload(CH_PRE,row)+beta*vload(CH_DIR,row);
    if(!finite(nextDirection)){reportAt(ERR_NONFINITE,8u,row);continue;}
    vstore(CH_DIR,row,nextDirection);}
   storageBarrier();workgroupBarrier();

   applyAllRows(lane,liveRows,CH_DIR,CH_DIRIMG);
   storageBarrier();workgroupBarrier();

   reduceCurvature(lane,liveRows);
   mergePartials(lane);
   if(lane==0u){finishCurvature();wHalt=select(0u,1u,stopped());}
   storageBarrier();workgroupBarrier();
   halt=workgroupUniformLoad(&wHalt);}}

 // --- P10: finalizeAndPublish ---------------------------------------------
 if(lane==0u&&!failed()&&atomicLoad(&control[1])==0u){reportAt(ERR_NONCONVERGENCE,9u,INVALID);}
 storageBarrier();workgroupBarrier();
 let success=!failed()&&atomicLoad(&control[1])!=0u;
 for(var row=lane;row<liveRows;row+=LANES){
  let rawSeed=vload(CH_SEED,row);
  let seed=select(0.0,rawSeed,finite(rawSeed));
  let candidate=vload(${diagnosticOutputChannel}u,row);
  pressureOut[row]=select(seed,candidate,success&&finite(candidate));}
 if(lane==0u&&success){atomicStore(&control[20],1u);}
 ${activityExit}
}
`;
}
