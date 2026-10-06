/** The sticky GPU failure record of one mixed frame sequence ("Uniform mixed
 * frame status", 16 u32, owned by UniformMixedFrame). The first failing
 * writer latches the cause and nothing clears it: every later slot gate,
 * band launch and transfer back to the simulation reads word 0 and does no
 * work. No readback is needed for the GPU to stop; the host learns the cause
 * from its receipt copy.
 *
 * Words: 0 first cause (0 healthy; first writer wins), 1 its frame, 2 its
 * layout generation, 3-4 cause detail, 5 every cause seen (bit per cause,
 * atomicOr), 6 last accepted frame, 7 the current frame (the host writes it
 * before each advance), 8 the current layout generation (written on the GPU
 * by whoever adopts a layout: the remap's markListed, from the receipt),
 * 9-15 reserved. */
export const UNIFORM_MIXED_STATUS_WORDS=16;
export const UNIFORM_MIXED_STATUS={cause:0,frame:1,generation:2,detail:3,causes:5,accepted:6,currentFrame:7,currentGeneration:8} as const;
/** Latched causes. The final pressure schedule gate latches 1-4 (its verdict);
 * 6-8 are for the layout and transport stages. 5 is retired. */
export const UNIFORM_MIXED_FAILURE={pressureRejected:1,pressureUnconverged:2,pressureNonfinite:3,bandCapacity:4,layoutCapacity:6,invalidSupport:7,transportCapacity:8} as const;
const CAUSE_NAMES:Record<number,string>={1:"pressure rejected a non-improving cycle",2:"pressure did not converge",3:"pressure went nonfinite",4:"pressure band over capacity",
 6:"layout or hanging-tap capacity exceeded",7:"invalid support",8:"transport capacity exceeded"};

/** WGSL for the record at @group(group) @binding(binding). "read" gives
 * umFrameFailed(); "read_write" also gives umLatchFailure(cause, a, b), which
 * any number of invocations may call: the first cause wins words 0-4, every
 * cause lands in word 5. */
export function uniformMixedFrameStatusWGSL(group:number,binding:number,access:"read"|"read_write"):string{
 if(access==="read")return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read> umStatus:array<u32,${UNIFORM_MIXED_STATUS_WORDS}>;
fn umFrameFailed()->bool{return umStatus[0]!=0u;}
`;
 return /* wgsl */`
@group(${group}) @binding(${binding}) var<storage,read_write> umStatus:array<atomic<u32>,${UNIFORM_MIXED_STATUS_WORDS}>;
fn umFrameFailed()->bool{return atomicLoad(&umStatus[0])!=0u;}
fn umLatchFailure(cause:u32,a:u32,b:u32){
 atomicOr(&umStatus[5],1u<<cause);
 loop{
  let r=atomicCompareExchangeWeak(&umStatus[0],0u,cause);
  if(r.exchanged){atomicStore(&umStatus[1],atomicLoad(&umStatus[7]));atomicStore(&umStatus[2],atomicLoad(&umStatus[8]));atomicStore(&umStatus[3],a);atomicStore(&umStatus[4],b);break;}
  if(r.old_value!=0u){break;}
 }
}
`;
}

/** The host's reading of a copied record: undefined when healthy. */
export function describeUniformMixedFrameStatus(words:Uint32Array):string|undefined{
 const cause=words[0]!;if(cause===0)return undefined;
 const f=new Float32Array(words.buffer,words.byteOffset,words.length);
 const detail=cause===1||cause===3?`candidate ${f[3]}, accepted ${f[4]}`:cause===2?`accepted ${f[3]} after ${words[4]} cycles`:cause===4?`${words[3]} tiles`:cause===6||cause===7?relayoutDetail(words[3]!,words[4]!):`detail ${words[3]} ${words[4]}`;
 const all=Array.from({length:31},(_,i)=>i+1).filter(i=>words[5]!&(1<<i)&&i!==cause).map(i=>CAUSE_NAMES[i]??`cause ${i}`);
 return `GPU failure record: ${CAUSE_NAMES[cause]??`cause ${cause}`} (${detail}) at frame ${words[1]}, layout generation ${words[2]}; last accepted frame ${words[6]}${all.length?`; also ${all.join(", ")}`:""}`;
}

/** Causes 6 and 7 from a GPU relayout (UniformMixedRemap.applyGpu): the
 * builder's sticky fatal bits (UNIFORM_MIXED_RELAYOUT_FATAL: 1 hanging tap
 * capacity, 2 tier sum, 4 tile words, 16 h-tile capacity) and the build that first raised one. */
function relayoutDetail(fatal:number,build:number):string{
 const names=[[1,"hanging tap slots over the cache's capacity"],[2,"h + 4h tiles differ from the lattice's"],[4,"a tile word, owner index or worklist entry disagrees with the receipt counts"],
  [16,"h tiles over the owner-indexed storage's capacity"]] as const;
 const causes=names.filter(([bit])=>fatal&bit).map(([,name])=>name);
 return `layout build ${build}: ${causes.length?causes.join("; "):`fatal bits ${fatal}`}`;
}
