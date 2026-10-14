/** Moving-window decision primitive, embedded by UniformPagePool at commit.
 *
 * Coordinates here are signed world pages, relative to the initial scene
 * frame (whose metre origin need not be page-aligned). A lattice page is
 * worldPage - F. All resident pages, including idle ones, and all non-solid
 * requests must be included before deciding. Excluding idle residents could
 * wrap a live block onto a different world page before it retires.
 *
 * Addressing and world-solid policy live in uniform-window-address.ts.
 * See docs/plans/moving-window-handoff-2026-10-14.md for integration evidence.
 */
import { UNIFORM_WINDOW_EDGE_FATAL } from "./uniform-window-address";
const UNIFORM_PAGE_POOL_FATAL = { edge: UNIFORM_WINDOW_EDGE_FATAL };

export type UniformWindowPage = readonly [number, number, number];
export const UNIFORM_MOVING_WINDOW_PAGES = 64;
export const UNIFORM_MOVING_WINDOW_MARGIN = 2;
/** State and bounds are separate: gather writes only bounds; decision writes
 * only state, then resets bounds for the next gather. F is never host-written
 * after construction. delta is this boundary's shift, revision counts shifts,
 * boundary counts decisions, fatal is sticky (the pool's named edge bit).
 */
export const UNIFORM_MOVING_WINDOW_STATE = { frame: 0, revision: 3, delta: 4, fatal: 7, boundary: 8, words: 12 } as const;
/** Sign-bit-biased i32 values allow signed min/max using atomic<u32>, like
 * the page pool's request buffer. Empty: low=0xffffffff, high=0, seen=0. */
export const UNIFORM_MOVING_WINDOW_BOUNDS = { low: 0, high: 3, seen: 6, words: 8 } as const;
const I32_MIN = -2147483648, I32_MAX = 2147483647;
const int32 = (v: number) => Number.isInteger(v) && v >= I32_MIN && v <= I32_MAX;

export interface UniformWindowBounds { readonly low: UniformWindowPage; readonly high: UniformWindowPage }
export interface UniformWindowDecision { readonly frame: UniformWindowPage; readonly delta: UniformWindowPage; readonly fatal: number }
/** Probe-only forced boundary, one-based; omitted means automatic decisions.
 * The force never bypasses fit or signed-coordinate checks. */
export interface UniformWindowForce { readonly boundary: number; readonly delta: UniformWindowPage }

/** Host twin of mwDecide. No caller may publish a partial-axis decision:
 * either all axes fit or the original frame stays intact with edge fatal.
 * A two-page margin on either side reserves room for the next census;
 * a forced probe shift may use that margin but must keep all pages inside.
 */
export function uniformWindowDecision(frame: UniformWindowPage, bounds?: UniformWindowBounds, force?: UniformWindowPage): UniformWindowDecision {
  if (!frame.every(int32) || (force && !force.every(int32))) throw new RangeError("Moving-window coordinates must be i32 pages");
  if (bounds && (!bounds.low.every(int32) || !bounds.high.every(int32) || bounds.low.some((v, k) => v > bounds.high[k]!)))
    throw new RangeError("Moving-window bounds must be ordered i32 pages");
  const next: [number, number, number] = [frame[0], frame[1], frame[2]];
  for (let axis = 0; axis < 3; axis++) {
    const lo = bounds?.low[axis], hi = bounds?.high[axis];
    if (lo !== undefined && hi !== undefined) {
      const width = hi - lo + 1;
      if (width > UNIFORM_MOVING_WINDOW_PAGES - 2 * UNIFORM_MOVING_WINDOW_MARGIN)
        return { frame, delta: [0, 0, 0], fatal: UNIFORM_PAGE_POOL_FATAL.edge };
      if (lo - frame[axis]! < UNIFORM_MOVING_WINDOW_MARGIN || hi - frame[axis]! >= UNIFORM_MOVING_WINDOW_PAGES - UNIFORM_MOVING_WINDOW_MARGIN)
        next[axis] = Math.max(I32_MIN, Math.min(I32_MAX - 63, lo - Math.floor((64 - width) / 2)));
    }
    if (force) next[axis] = frame[axis]! + force[axis]!;
    // The full window, and the particle rebase delta, must be representable.
    if (!int32(next[axis]!) || next[axis]! > I32_MAX - 63 || !int32(next[axis]! - frame[axis]!) ||
        (lo !== undefined && hi !== undefined && (lo < next[axis]! || hi - next[axis]! >= 64)))
      return { frame, delta: [0, 0, 0], fatal: UNIFORM_PAGE_POOL_FATAL.edge };
  }
  return { frame: next, delta: [next[0] - frame[0], next[1] - frame[1], next[2] - frame[2]], fatal: 0 };
}

/** Include in an existing residency/request pass. The includer declares
 * mwBounds:array<atomic<u32>> at a binding of its choice. Call once per
 * requested page or reduce a request box to its two corners. Include pins,
 * queued drops and every still-resident page as well as liquid requests.
 * SOLID requests do not contribute. No pass is added for the gather. */
export const uniformMovingWindowGatherWGSL = /* wgsl */ `
fn mwInclude(q:vec3i){
 let key=bitcast<vec3u>(q)^vec3u(0x80000000u);
 for(var a=0u;a<3u;a++){atomicMin(&mwBounds[a],key[a]);atomicMax(&mwBounds[3u+a],key[a]);}
 atomicOr(&mwBounds[6u],1u);
}`;

/** Standalone fixed-size decision launch. It must run after all gathers and
 * before any consumer of the new frame; it does NOT rebase those consumers.
 * No world-page subtraction uses signed arithmetic until its small range is
 * known. This also keeps INT_MIN/INT_MAX rejection identical to the twin. */
export const uniformMovingWindowDecisionWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> mwState:array<u32>;
@group(0) @binding(1) var<storage,read_write> mwBounds:array<atomic<u32>>;
struct MWForce{boundary:u32,x:i32,y:i32,z:i32}
@group(0) @binding(2) var<uniform> mwForce:MWForce;
@compute @workgroup_size(1) fn mwDecide(){
 let old=vec3u(mwState[0],mwState[1],mwState[2])^vec3u(0x80000000u);
 var next=old;var fail=false;
 let seen=atomicLoad(&mwBounds[6u])!=0u;
 let forced=mwForce.boundary!=0u&&mwForce.boundary==mwState[8u]+1u;
 let force=vec3i(mwForce.x,mwForce.y,mwForce.z);
 for(var a=0u;a<3u;a++){
  let lo=atomicLoad(&mwBounds[a]);let hi=atomicLoad(&mwBounds[3u+a]);
  if(seen){
   // Check the difference before adding 1 (a full i32 extent is 2^32).
   if(hi<lo||hi-lo>=60u){fail=true;}
   else{
    let width=hi-lo+1u;
    if(lo<old[a]||lo-old[a]<2u||hi<old[a]||hi-old[a]>=62u){
     let pad=(64u-width)/2u;
     next[a]=min(lo-min(lo,pad),0xffffffc0u);
    }
   }
  }
  if(forced){
   // Two's-complement magnitude, including INT_MIN, without signed negate.
   let bits=bitcast<u32>(force[a]);let negative=force[a]<0;
   let magnitude=select(bits,0u-bits,negative);
   if(negative){if(magnitude>old[a]){fail=true;}next[a]=old[a]-magnitude;}
   else{if(magnitude>0xffffffffu-old[a]){fail=true;}next[a]=old[a]+magnitude;}
  }
  if(next[a]>0xffffffc0u){fail=true;}
  if(next[a]>=old[a]){if(next[a]-old[a]>0x7fffffffu){fail=true;}}
  else{if(old[a]-next[a]>0x80000000u){fail=true;}}
  if(seen&&(lo<next[a]||hi<next[a]||hi-next[a]>=64u)){fail=true;}
  atomicStore(&mwBounds[a],0xffffffffu);atomicStore(&mwBounds[3u+a],0u);
 }
 atomicStore(&mwBounds[6u],0u);
 mwState[8u]+=1u;
 mwState[4u]=0u;mwState[5u]=0u;mwState[6u]=0u;
 if(fail){mwState[7u]|=${UNIFORM_PAGE_POOL_FATAL.edge}u;}
 if(mwState[7u]!=0u){return;}
 if(any(next!=old)){mwState[3u]+=1u;}
 for(var a=0u;a<3u;a++){mwState[a]=next[a]^0x80000000u;mwState[4u+a]=next[a]-old[a];}
}`;

/** Preallocated GPU state for the decision primitive. Construction and
 * initialize belong to scene build. encodeBoundary neither allocates JS
 * objects nor reads back, compiles, resizes or rebinds anything.
 * This standalone wrapper is for the primitive oracle; the solver embeds
 * the same decision in its pool commit and mirrors the header there. */
export class UniformMovingWindow {
  readonly state: GPUBuffer;
  readonly bounds: GPUBuffer;
  readonly allocatedBytes = 4 * (UNIFORM_MOVING_WINDOW_STATE.words + UNIFORM_MOVING_WINDOW_BOUNDS.words + 4);
  private readonly force: GPUBuffer;
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;
  private static readonly PASS: GPUComputePassDescriptor = { label: "Moving window boundary decision" };
  constructor(private readonly device: GPUDevice, frame: UniformWindowPage = [0, 0, 0], force?: UniformWindowForce) {
    if (!frame.every(v => int32(v) && v <= I32_MAX - 63)) throw new RangeError("Moving-window frame must hold 64 signed pages on each axis");
    if (force && (!Number.isInteger(force.boundary) || force.boundary < 1 || force.boundary > 0xffffffff || !force.delta.every(int32)))
      throw new RangeError("Moving-window probe force needs a positive u32 boundary and i32 page delta");
    this.state = device.createBuffer({ label: "Moving window GPU frame", size: 4 * UNIFORM_MOVING_WINDOW_STATE.words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.bounds = device.createBuffer({ label: "Moving window gathered bounds", size: 4 * UNIFORM_MOVING_WINDOW_BOUNDS.words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.force = device.createBuffer({ label: "Moving window probe force (off by default)", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const initial = new Int32Array(UNIFORM_MOVING_WINDOW_STATE.words);initial.set(frame);
    const bounds = new Uint32Array(UNIFORM_MOVING_WINDOW_BOUNDS.words);bounds.fill(0xffffffff, 0, 3);
    const command = new Int32Array(4);if (force) { command[0] = force.boundary;command.set(force.delta, 1); }
    device.queue.writeBuffer(this.state, 0, initial);device.queue.writeBuffer(this.bounds, 0, bounds);device.queue.writeBuffer(this.force, 0, command);
  }
  async initialize(): Promise<void> {
    if (this.pipeline) throw new Error("Moving-window decision is already initialized");
    const module = this.device.createShaderModule({ label: "Moving window decision", code: uniformMovingWindowDecisionWGSL });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    this.pipeline = await this.device.createComputePipelineAsync({ label: "Moving window mwDecide", layout: "auto", compute: { module, entryPoint: "mwDecide" } });
    this.group = this.device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.state } }, { binding: 1, resource: { buffer: this.bounds } }, { binding: 2, resource: { buffer: this.force } },
    ] });
  }
  encodeBoundary(encoder: GPUCommandEncoder): void {
    if (!this.pipeline || !this.group) throw new Error("Moving-window decision must compile at scene build");
    const pass = encoder.beginComputePass(UniformMovingWindow.PASS);
    pass.setPipeline(this.pipeline);pass.setBindGroup(0, this.group);pass.dispatchWorkgroups(1);pass.end();
  }
  destroy(): void { this.state.destroy();this.bounds.destroy();this.force.destroy(); }
}
