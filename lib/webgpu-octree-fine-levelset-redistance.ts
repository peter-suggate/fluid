import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";

export interface FineLevelSetGPURedistanceOptions {
  /** Required signed-distance width, measured in fine cells. */
  bandCells: number;
  residualTolerance?: number;
  /** JFA-CPT is the fixed-dispatch product path; FMM is the validation oracle. */
  method?: FineLevelSetRedistanceMethod;
}

export type FineLevelSetRedistanceMethod = "jfa-cpt" | "fmm";

export function resolveFineLevelSetRedistanceMethod(value: unknown): FineLevelSetRedistanceMethod {
  if (value === undefined || value === "jfa" || value === "jfa-cpt") return "jfa-cpt";
  if (value === "fmm") return "fmm";
  throw new RangeError(`Unknown fine redistance method: ${String(value)}`);
}

/** Descending JFA strides followed by the 1+JFA repair pass. */
export function planFineLevelSetJFAStrides(bandCells: number): readonly number[] {
  if (!Number.isSafeInteger(bandCells) || bandCells < 1 || bandCells > 256) {
    throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
  }
  let stride = 1;
  // Round up so the sum of the descending jumps covers the complete signed
  // band from either side of an interface, including a seed near a resident
  // support boundary. Rounding down leaves a one-sided annulus unreachable.
  while (stride < bandCells) stride *= 2;
  const strides: number[] = [];
  for (; stride >= 1; stride /= 2) strides.push(stride);
  strides.push(1);
  return strides;
}

export interface FineLevelSetGPURedistanceControl {
  unresolvedCells: number;
  maximumResidualScaled: number;
  seedCount: number;
  committed: boolean;
  flags: number;
  firstError: number;
  activatedPages: number;
  acceptedCells: number;
  initialPages: number;
  finalPages: number;
}

export const FINE_LEVELSET_REDISTANCE_CONTROL_BYTES = 48;
/** The opt-in FMM remains a tiny-scene diagnostic until its Dawn/Metal fault
 * is resolved. Four B4 pages are enough for local oracle fixtures without
 * allowing an accidental production-domain dispatch. */
export const FINE_LEVELSET_FMM_MAX_DIAGNOSTIC_SAMPLES = 256;
const FINE_LEVELSET_JFA_MAX_PASSES = 10;
export const FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES = FINE_LEVELSET_REDISTANCE_CONTROL_BYTES
  + 80 + FINE_LEVELSET_JFA_MAX_PASSES * 80;

export function unpackFineLevelSetGPURedistanceControl(words: ArrayLike<number>): FineLevelSetGPURedistanceControl {
  if (words.length < 4) throw new RangeError("Fine redistance control requires four words");
  return {
    unresolvedCells: Number(words[0]) >>> 0,
    maximumResidualScaled: Number(words[1]) >>> 0,
    seedCount: Number(words[2]) >>> 0,
    committed: Number(words[3]) !== 0,
    flags: words.length > 4 ? Number(words[4]) >>> 0 : 0,
    firstError: words.length > 5 ? Number(words[5]) >>> 0 : 0xffff_ffff,
    activatedPages: words.length > 6 ? Number(words[6]) >>> 0 : 0,
    acceptedCells: words.length > 7 ? Number(words[7]) >>> 0 : 0,
    initialPages: words.length > 9 ? Number(words[9]) >>> 0 : 0,
    finalPages: words.length > 10 ? Number(words[10]) >>> 0 : 0,
  };
}

/**
 * Fixed-resident fine-grid redistance. JFA-CPT is the default, parallel path;
 * exact causal bucketed FMM remains selectable as the validation oracle. Both
 * consume the complete support generation published by topology and never
 * allocate, link, or publish a page while redistancing.
 */
export class WebGPUFineLevelSetRedistance {
  readonly control: GPUBuffer;
  readonly allocatedBytes = FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES;
  private readonly params: GPUBuffer;
  private readonly jfaParams: readonly GPUBuffer[];
  private readonly controlPipeline: GPUComputePipeline;
  private readonly initializePipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly snapshotPipeline: GPUComputePipeline;
  private readonly marchPipeline: GPUComputePipeline;
  private readonly advancePipeline: GPUComputePipeline;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private readonly jfaControlPipeline: GPUComputePipeline;
  private readonly jfaSeedPipeline: GPUComputePipeline;
  private readonly jfaABPipeline: GPUComputePipeline;
  private readonly jfaBAPipeline: GPUComputePipeline;
  private readonly jfaResolveAToBPipeline: GPUComputePipeline;
  private readonly jfaResolveBToAPipeline: GPUComputePipeline;
  private readonly jfaValidatePipeline: GPUComputePipeline;
  private readonly jfaFinalizePipeline: GPUComputePipeline;
  private readonly jfaCommitPipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource) {
    this.control = device.createBuffer({ label: "fine-levelset fast-march control",
      size: FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "fine-levelset fast-march params", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.jfaParams = Array.from({ length: FINE_LEVELSET_JFA_MAX_PASSES }, (_, index) =>
      device.createBuffer({ label: `fine-levelset JFA-CPT params ${index}`, size: 80,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }));
    const module = device.createShaderModule({ label: "fine-levelset uniform fast march", code: fineLevelSetRedistanceWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `fine fast march ${entryPoint}`,
      layout: "auto", compute: { module, entryPoint } });
    this.controlPipeline = pipeline("initializeControl");
    this.initializePipeline = pipeline("initializeDistances");
    this.seedPipeline = pipeline("seedDistances");
    this.snapshotPipeline = pipeline("snapshotKnownDistances");
    this.marchPipeline = pipeline("marchBucket");
    this.advancePipeline = pipeline("advanceBucket");
    this.validatePipeline = pipeline("validateDistances");
    this.finalizePipeline = pipeline("finalizeDistances");
    this.commitPipeline = pipeline("commitDistances");
    const jfaModule = device.createShaderModule({ label: "fine-levelset JFA closest-point transform",
      code: fineLevelSetJFACPTWGSL });
    const jfaPipeline = (entryPoint: string) => device.createComputePipeline({
      label: `fine JFA-CPT ${entryPoint}`, layout: "auto", compute: { module: jfaModule, entryPoint },
    });
    this.jfaControlPipeline = jfaPipeline("initializeJFAControl");
    this.jfaSeedPipeline = jfaPipeline("seedClosestPoints");
    this.jfaABPipeline = jfaPipeline("jumpFloodAToB");
    this.jfaBAPipeline = jfaPipeline("jumpFloodBToA");
    this.jfaResolveAToBPipeline = jfaPipeline("resolveClosestPointsAToB");
    this.jfaResolveBToAPipeline = jfaPipeline("resolveClosestPointsBToA");
    this.jfaValidatePipeline = jfaPipeline("validateJFADistances");
    this.jfaFinalizePipeline = jfaPipeline("finalizeJFADistances");
    this.jfaCommitPipeline = jfaPipeline("commitJFADistances");
  }

  encode(encoder: GPUCommandEncoder, options: FineLevelSetGPURedistanceOptions): void {
    if ((this.source.plan.fineFactor !== 4 && this.source.plan.fineFactor !== 8)
      || this.source.plan.brickResolution !== 4) {
      throw new RangeError("GPU fine fast marching requires a factor-4/factor-8 B4 generation");
    }
    if (!Number.isSafeInteger(options.bandCells) || options.bandCells < 1 || options.bandCells > 256) {
      throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
    }
    const method = resolveFineLevelSetRedistanceMethod(options.method);
    const tolerance = options.residualTolerance ?? 0.1;
    if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance > 1) {
      throw new RangeError("Fine redistance residual tolerance must be in (0, 1]");
    }
    const bytes = new ArrayBuffer(80); const u32 = new Uint32Array(bytes); const f32 = new Float32Array(bytes);
    u32.set([...this.source.plan.brickDimensions, this.source.plan.brickResolution,
      ...this.source.plan.sampleDimensions, this.source.plan.samplesPerBrick,
      this.source.plan.hashCapacity, this.source.plan.maximumHashProbes,
      this.source.plan.maximumResidentBricks, this.source.generation, options.bandCells], 0);
    f32[13] = this.source.plan.fineCellWidth; f32[14] = tolerance;
    u32[15] = this.source.plan.maximumResidentBricks * this.source.plan.samplesPerBrick;
    u32[16] = this.device.limits.maxComputeWorkgroupsPerDimension;
    u32[17] = options.bandCells * 2;
    const sampleCount = this.source.plan.sampleDimensions.reduce((product, value) => product * value, 1);
    if (!Number.isSafeInteger(sampleCount) || sampleCount >= 0xffff_ffff) {
      throw new RangeError("Fine redistance logical sample keys must fit in 32 bits");
    }
    if (method === "jfa-cpt") {
      this.encodeJFA(encoder, bytes, options.bandCells);
      return;
    }
    if (sampleCount > FINE_LEVELSET_FMM_MAX_DIAGNOSTIC_SAMPLES) {
      throw new RangeError(`Fine FMM oracle is limited to ${FINE_LEVELSET_FMM_MAX_DIAGNOSTIC_SAMPLES} logical samples until its Dawn backend gate is cleared`);
    }
    this.device.queue.writeBuffer(this.params, 0, bytes); encoder.clearBuffer(this.control);
    const bind = (pipeline: GPUComputePipeline, bindings: readonly (readonly [number, GPUBuffer])[]) =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindings.map(([binding, buffer]) =>
        ({ binding, resource: { buffer } })) });
    const bindings = [[0, this.params], [1, this.source.hash], [2, this.source.metadata],
      [3, this.source.worklist], [4, this.source.flags], [5, this.source.phi],
      [6, this.source.workA], [7, this.source.workB], [8, this.control]] as const;
    const pick = (...wanted: number[]) => bindings.filter(([binding]) => wanted.includes(binding));
    const run = (pipeline: GPUComputePipeline, selected: readonly (readonly [number, GPUBuffer])[], groups: number,
      pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind(pipeline, selected));
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      if (dispatch.workgroups > 0) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    };
    // Each bucket snapshots the previously accepted distances before the
    // parallel march. This makes candidate reads race-free while preserving
    // the half-cell causal bucket ordering without queue fences.
    const pass = encoder.beginComputePass({ label: "Fixed-resident fine FMM oracle" });
    run(this.controlPipeline, pick(0, 3, 8), 1, pass);
    run(this.initializePipeline, pick(0, 2, 3, 4, 5, 6, 8), this.source.plan.maximumResidentBricks, pass);
    run(this.seedPipeline, pick(0, 2, 3, 4, 5, 6, 8), this.source.plan.maximumResidentBricks, pass);
    for (let bucket = 0; bucket < options.bandCells * 2; bucket += 1) {
      run(this.snapshotPipeline, pick(0, 2, 3, 4, 6, 7), this.source.plan.maximumResidentBricks, pass);
      run(this.marchPipeline, pick(0, 2, 3, 4, 6, 7, 8), this.source.plan.maximumResidentBricks, pass);
      run(this.advancePipeline, pick(8), 1, pass);
    }
    run(this.validatePipeline, pick(0, 2, 3, 4, 6, 8), this.source.plan.maximumResidentBricks, pass);
    run(this.finalizePipeline, [[0, this.params], [3, this.source.worklist], [8, this.control]], 1, pass);
    run(this.commitPipeline, pick(0, 2, 3, 4, 5, 6, 8), this.source.plan.maximumResidentBricks, pass);
    pass.end();
  }

  private encodeJFA(encoder: GPUCommandEncoder, baseBytes: ArrayBuffer, bandCells: number): void {
    const strides = planFineLevelSetJFAStrides(bandCells);
    if (strides.length > this.jfaParams.length) throw new RangeError("Fine JFA pass budget exceeded");
    const distanceInB = (strides.length & 1) === 0;
    strides.forEach((stride, index) => {
      const bytes = baseBytes.slice(0); const words = new Uint32Array(bytes);
      words[17] = stride; words[18] = distanceInB ? 1 : 0;
      this.device.queue.writeBuffer(this.jfaParams[index], 0, bytes);
    });
    encoder.clearBuffer(this.control);
    const buffers = [[1, this.source.hash], [2, this.source.metadata], [3, this.source.worklist],
      [4, this.source.flags], [5, this.source.phi], [6, this.source.workA], [7, this.source.workB],
      [8, this.control]] as const;
    const pass = encoder.beginComputePass({ label: "Fine level-set JFA closest-point redistance" });
    const dispatch = planFineLevelSetDispatch2D(this.source.plan.maximumResidentBricks,
      this.device.limits.maxComputeWorkgroupsPerDimension);
    const run = (pipeline: GPUComputePipeline, params: GPUBuffer, wanted: readonly number[], groups = true) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: params } },
        ...buffers.filter(([binding]) => wanted.includes(binding)).map(([binding, buffer]) =>
          ({ binding, resource: { buffer } })),
      ] }));
      if (groups) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
      else pass.dispatchWorkgroups(1);
    };
    const params = this.jfaParams[0];
    run(this.jfaControlPipeline, params, [3, 8], false);
    run(this.jfaSeedPipeline, params, [1, 2, 3, 4, 5, 6, 7, 8]);
    let inA = true;
    strides.forEach((_stride, index) => {
      run(inA ? this.jfaABPipeline : this.jfaBAPipeline, this.jfaParams[index],
        [1, 2, 3, 4, 6, 7]);
      inA = !inA;
    });
    run(inA ? this.jfaResolveAToBPipeline : this.jfaResolveBToAPipeline, params,
      [2, 3, 4, 5, 6, 7, 8]);
    // Resolve writes magnitudes to the opposite channel.
    run(this.jfaValidatePipeline, params, [1, 2, 3, 4, 6, 7, 8]);
    run(this.jfaFinalizePipeline, params, [3, 8], false);
    run(this.jfaCommitPipeline, params, [2, 3, 4, 5, 6, 7, 8]);
    pass.end();
  }

  destroy(): void { this.control.destroy(); this.params.destroy();
    this.jfaParams.forEach((buffer) => buffer.destroy()); }
}

/** Fixed-resident FMM validation oracle. This shader has no request arena,
 * page-table mutation, or indirect-dispatch binding. Each half-cell bucket
 * snapshots accepted distances before a race-free parallel update. */
export const fineLevelSetRedistanceWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const KNOWN:u32=4u;const NEGATIVE:u32=16u;const LARGE:f32=3.402823e38;
const STALE:u32=4u;const NONFINITE:u32=8u;
struct Params{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,hashCapacity:u32,maxProbes:u32,pageCapacity:u32,generation:u32,bandCells:u32,fineWidth:f32,tolerance:f32,scratchWords:u32,maxWorkgroups:u32,bucketCount:u32,pad0:u32,pad1:u32}
struct Control{unresolved:atomic<u32>,residualScaled:atomic<u32>,seeds:atomic<u32>,committed:atomic<u32>,flags:atomic<u32>,firstError:atomic<u32>,activated:atomic<u32>,accepted:atomic<u32>,requests:atomic<u32>,initialPages:u32,finalPages:u32,bucket:atomic<u32>}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(2)var<storage,read>metadata:array<u32>;
@group(0)@binding(3)var<storage,read>worklist:array<u32>;
@group(0)@binding(4)var<storage,read_write>flags:array<u32>;
@group(0)@binding(5)var<storage,read_write>phi:array<u32>;
@group(0)@binding(6)var<storage,read_write>distance:array<f32>;
@group(0)@binding(7)var<storage,read_write>snapshot:array<f32>;
@group(0)@binding(8)var<storage,read_write>control:Control;
fn finite(v:f32)->bool{return v==v&&abs(v)<LARGE;}
fn fail(code:u32,index:u32){atomicOr(&control.flags,code);atomicMin(&control.firstError,index);}
fn bandDistance()->f32{return f32(p.bandCells)*p.fineWidth;}
fn bucketUpper()->f32{return f32(atomicLoad(&control.bucket)+1u)*(.5*p.fineWidth);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}
fn localIndex(q:vec3u)->u32{return q.x+p.brickResolution*(q.y+p.brickResolution*q.z);}
fn neighborIndex(id:u32,local:u32,direction:u32)->u32{var q=localCoord(local);var next=id;let r=p.brickResolution;if(direction==0u){if(q.x>0u){q.x-=1u;}else{next=metadata[id*10u+4u];q.x=r-1u;}}else if(direction==1u){if(q.x+1u<r){q.x+=1u;}else{next=metadata[id*10u+5u];q.x=0u;}}else if(direction==2u){if(q.y>0u){q.y-=1u;}else{next=metadata[id*10u+6u];q.y=r-1u;}}else if(direction==3u){if(q.y+1u<r){q.y+=1u;}else{next=metadata[id*10u+7u];q.y=0u;}}else if(direction==4u){if(q.z>0u){q.z-=1u;}else{next=metadata[id*10u+8u];q.z=r-1u;}}else{if(q.z+1u<r){q.z+=1u;}else{next=metadata[id*10u+9u];q.z=0u;}}if(next==INVALID||next>=p.pageCapacity||metadata[next*10u+2u]!=p.generation){return INVALID;}return next*p.samplesPerBrick+localIndex(q);}
fn activePage(wid:vec3u,nw:vec3u)->u32{let work=fineLinearWorkgroup(wid,nw);let count=min(worklist[0],p.pageCapacity);if(work>=count){return INVALID;}let id=worklist[5u+work];return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}
fn solve(a0:f32,a1:f32,a2:f32)->f32{var a=a0;var b=a1;var c=a2;if(a>b){let t=a;a=b;b=t;}if(b>c){let t=b;b=c;c=t;}if(a>b){let t=a;a=b;b=t;}if(a>=LARGE){return LARGE;}let h=p.fineWidth;var result=a+h;if(result>b){result=.5*(a+b+sqrt(max(0.,2.*h*h-(a-b)*(a-b))));}if(result>c){let sum=a+b+c;result=(sum+sqrt(max(0.,sum*sum-3.*(a*a+b*b+c*c-h*h))))/3.;}return result;}
fn candidate(index:u32)->f32{let id=index/p.samplesPerBrick;let local=index-id*p.samplesPerBrick;var axes=vec3f(LARGE);for(var direction=0u;direction<6u;direction+=1u){let neighbor=neighborIndex(id,local,direction);if(neighbor!=INVALID){axes[direction/2u]=min(axes[direction/2u],snapshot[neighbor]);}}return solve(axes.x,axes.y,axes.z);}
@compute @workgroup_size(1)fn initializeControl(){atomicStore(&control.firstError,INVALID);atomicStore(&control.bucket,0u);atomicStore(&control.requests,0u);let initial=min(worklist[0],p.pageCapacity);control.initialPages=initial;control.finalPages=initial;if(worklist[1]!=p.generation){fail(STALE,worklist[1]);}}
@compute @workgroup_size(64)fn initializeDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*p.brickResolution+localCoord(lid);if(any(q>=p.sampleDims)){flags[index]=0u;return;}let value=bitcast<f32>(phi[index]);if((flags[index]&VALID)==0u||!finite(value)){fail(NONFINITE,index);return;}flags[index]=VALID|select(0u,NEGATIVE,value<0.);distance[index]=bandDistance();}
@compute @workgroup_size(64)fn seedDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let center=bitcast<f32>(phi[index]);var seed=center==0.;var seedDistance=select(bandDistance(),0.,seed);for(var direction=0u;direction<6u;direction+=1u){let neighbor=neighborIndex(id,lid,direction);if(neighbor==INVALID||(flags[neighbor]&VALID)==0u){continue;}let other=bitcast<f32>(phi[neighbor]);if(!finite(other)||(other<0.)==(center<0.)){continue;}let denominator=abs(center)+abs(other);seedDistance=min(seedDistance,select(0.,p.fineWidth*abs(center)/denominator,denominator>0.));seed=true;}if(seed){distance[index]=seedDistance;flags[index]|=KNOWN|INTERFACE;atomicAdd(&control.seeds,1u);atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(64)fn snapshotKnownDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;snapshot[index]=select(LARGE,distance[index],(flags[index]&KNOWN)!=0u);}
@compute @workgroup_size(64)fn marchBucket(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u||atomicLoad(&control.bucket)>=p.bucketCount){return;}let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u||(flags[index]&KNOWN)!=0u){return;}let value=candidate(index);if(finite(value)&&value<bucketUpper()&&value<bandDistance()){distance[index]=value;flags[index]|=KNOWN;atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(1)fn advanceBucket(){atomicAdd(&control.bucket,1u);}
@compute @workgroup_size(64)fn validateDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let flat=id*p.samplesPerBrick+lid;if((flags[flat]&VALID)==0u){return;}let d=distance[flat];if(!finite(d)){atomicAdd(&control.unresolved,1u);fail(NONFINITE,flat);return;}if(d>=bandDistance()){return;}if((flags[flat]&KNOWN)==0u){atomicAdd(&control.unresolved,1u);return;}if((flags[flat]&INTERFACE)!=0u){return;}var sum=0.;for(var axis=0u;axis<3u;axis+=1u){let left=neighborIndex(id,lid,axis*2u);let right=neighborIndex(id,lid,axis*2u+1u);var nearest=d;if(left!=INVALID&&(flags[left]&KNOWN)!=0u){nearest=min(nearest,distance[left]);}if(right!=INVALID&&(flags[right]&KNOWN)!=0u){nearest=min(nearest,distance[right]);}let gradient=max(0.,d-nearest)/p.fineWidth;sum+=gradient*gradient;}let residual=u32(min(4294967295.,abs(sqrt(sum)-1.)*1000000.));atomicMax(&control.residualScaled,residual);if(residual>u32(p.tolerance*1000000.)){atomicAdd(&control.unresolved,1u);}}
@compute @workgroup_size(1)fn finalizeDistances(){control.finalPages=min(worklist[0],p.pageCapacity);if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.unresolved)==0u&&atomicLoad(&control.seeds)>0u&&atomicLoad(&control.bucket)==p.bucketCount){atomicStore(&control.committed,1u);}}
@compute @workgroup_size(64)fn commitDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.committed)==0u){return;}let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let flat=id*p.samplesPerBrick+lid;let d=distance[flat];let sign=select(1.,-1.,(flags[flat]&NEGATIVE)!=0u);phi[flat]=bitcast<u32>(sign*d);if(d>=bandDistance()||(flags[flat]&KNOWN)==0u){flags[flat]=0u;}else{flags[flat]=VALID|(flags[flat]&(INTERFACE|NEGATIVE));}}
`;

/** Sparse 1+JFA closest-point transform. Seed keys are global fine-sample
 * linear indices, so the deterministic secondary ordering is independent of
 * physical page IDs and A/B generation allocation order. */
export const fineLevelSetJFACPTWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const NEGATIVE:u32=16u;const LARGE:f32=3.402823e38;
const SAMPLE_FLAG_BITS:u32=5u;const CP_FRACTION_MASK:u32=0x00ffffffu;const CP_FRACTION_SCALE:f32=16777215.;
const STALE:u32=4u;const NONFINITE:u32=8u;
struct Params{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,hashCapacity:u32,maxProbes:u32,pageCapacity:u32,generation:u32,bandCells:u32,fineWidth:f32,tolerance:f32,scratchWords:u32,maxWorkgroups:u32,stride:u32,distanceInB:u32,pad1:u32}
struct Control{unresolved:atomic<u32>,residualScaled:atomic<u32>,seeds:atomic<u32>,committed:atomic<u32>,flags:atomic<u32>,firstError:atomic<u32>,activated:atomic<u32>,accepted:atomic<u32>,requests:atomic<u32>,initialPages:u32,finalPages:u32,bucket:atomic<u32>}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read>pageHash:array<u32>;@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;@group(0)@binding(4)var<storage,read_write>flags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<u32>;@group(0)@binding(6)var<storage,read_write>workA:array<u32>;@group(0)@binding(7)var<storage,read_write>workB:array<u32>;@group(0)@binding(8)var<storage,read_write>control:Control;
fn finite(v:f32)->bool{return v==v&&abs(v)<LARGE;}fn fail(code:u32,index:u32){atomicOr(&control.flags,code);atomicMin(&control.firstError,index);}fn bandDistance()->f32{return f32(p.bandCells)*p.fineWidth;}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}fn localIndex(q:vec3u)->u32{return q.x+p.brickResolution*(q.y+p.brickResolution*q.z);}
fn sampleKey(q:vec3u)->u32{return q.x+p.sampleDims.x*(q.y+p.sampleDims.y*q.z);}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(p.hashCapacity-1u);}fn pageOf(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let stored=pageHash[slot*2u];if(stored==key){let id=pageHash[slot*2u+1u];return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}if(stored==INVALID){return INVALID;}}return INVALID;}
fn sampleIndex(q:vec3u)->u32{if(any(q>=p.sampleDims)){return INVALID;}let id=pageOf(packBrick(q/p.brickResolution));if(id==INVALID){return INVALID;}let index=id*p.samplesPerBrick+localIndex(q%p.brickResolution);return select(INVALID,index,(flags[index]&VALID)!=0u);}
fn activePage(wid:vec3u,nw:vec3u)->u32{let work=fineLinearWorkgroup(wid,nw);let count=min(worklist[0],p.pageCapacity);if(work>=count){return INVALID;}let id=worklist[5u+work];return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}
fn directionDelta(direction:u32)->vec3i{if(direction==0u){return vec3i(-1,0,0);}if(direction==1u){return vec3i(1,0,0);}if(direction==2u){return vec3i(0,-1,0);}if(direction==3u){return vec3i(0,1,0);}if(direction==4u){return vec3i(0,0,-1);}return vec3i(0,0,1);}
fn physicalSampleQ(index:u32)->vec3u{let id=index/p.samplesPerBrick;let local=index-id*p.samplesPerBrick;return unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(local);}
fn seedStableKey(index:u32)->u32{return sampleKey(physicalSampleQ(index));}
fn seedClosestPointCode(q:vec3u,index:u32)->u32{let center=bitcast<f32>(phi[index]);if(center==0.){return 6u<<24u;}var best=LARGE;var bestDirection=INVALID;var bestFraction=0.;for(var direction=0u;direction<6u;direction+=1u){let nq=vec3i(q)+directionDelta(direction);if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor==INVALID){continue;}let other=bitcast<f32>(phi[neighbor]);if(!finite(other)||(other<0.)==(center<0.)){continue;}let denominator=abs(center)+abs(other);let fraction=select(0.,abs(center)/denominator,denominator>0.);let d2=fraction*fraction;if(d2<best||(d2==best&&direction<bestDirection)){best=d2;bestDirection=direction;bestFraction=fraction;}}if(bestDirection==INVALID){return INVALID;}let quantized=u32(round(clamp(bestFraction,0.,1.)*CP_FRACTION_SCALE));return (bestDirection<<24u)|(quantized&CP_FRACTION_MASK);}
fn materializedClosestPoint(index:u32)->vec3f{let q=physicalSampleQ(index);let code=flags[index]>>SAMPLE_FLAG_BITS;let direction=code>>24u;let fraction=f32(code&CP_FRACTION_MASK)/CP_FRACTION_SCALE;return vec3f(q)+vec3f(.5)+select(vec3f(directionDelta(direction))*fraction,vec3f(0.),direction>=6u);}
fn flood(index:u32,q:vec3u,fromA:bool)->u32{let point=vec3f(q)+vec3f(.5);var best=select(workB[index],workA[index],fromA);var bestD=LARGE;if(best!=INVALID){let delta=point-materializedClosestPoint(best);bestD=dot(delta,delta);}let stride=i32(p.stride);for(var dz=-1;dz<=1;dz+=1){for(var dy=-1;dy<=1;dy+=1){for(var dx=-1;dx<=1;dx+=1){let nq=vec3i(q)+vec3i(dx,dy,dz)*stride;if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let candidateIndex=sampleIndex(vec3u(nq));if(candidateIndex==INVALID){continue;}let candidate=select(workB[candidateIndex],workA[candidateIndex],fromA);if(candidate==INVALID){continue;}let delta=point-materializedClosestPoint(candidate);let d=dot(delta,delta);if(d<bestD||(d==bestD&&seedStableKey(candidate)<seedStableKey(best))){best=candidate;bestD=d;}}}}return best;}
fn resolvedDistance(seed:u32,q:vec3u)->f32{if(seed==INVALID){return bandDistance();}return length((vec3f(q)+vec3f(.5))-materializedClosestPoint(seed))*p.fineWidth;}
fn distanceValue(index:u32)->f32{return bitcast<f32>(select(workA[index],workB[index],p.distanceInB!=0u));}
fn resolvedSeed(index:u32)->u32{return select(workB[index],workA[index],p.distanceInB!=0u);}
@compute @workgroup_size(1)fn initializeJFAControl(){atomicStore(&control.firstError,INVALID);let count=min(worklist[0],p.pageCapacity);control.initialPages=count;control.finalPages=count;if(worklist[1]!=p.generation){fail(STALE,worklist[1]);}}
@compute @workgroup_size(64)fn seedClosestPoints(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*p.brickResolution+localCoord(lid);workA[index]=INVALID;workB[index]=INVALID;if(any(q>=p.sampleDims)){flags[index]=0u;return;}let value=bitcast<f32>(phi[index]);if((flags[index]&VALID)==0u||!finite(value)){fail(NONFINITE,index);return;}flags[index]=VALID|select(0u,NEGATIVE,value<0.);let closest=seedClosestPointCode(q,index);if(closest!=INVALID){workA[index]=index;flags[index]|=INTERFACE|(closest<<SAMPLE_FLAG_BITS);atomicAdd(&control.seeds,1u);}}
@compute @workgroup_size(64)fn jumpFloodAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);workB[index]=flood(index,q,true);}
@compute @workgroup_size(64)fn jumpFloodBToA(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);workA[index]=flood(index,q,false);}
@compute @workgroup_size(64)fn resolveClosestPointsAToB(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);var seed=workA[index];if((flags[index]&INTERFACE)!=0u){seed=index;}if(seed==INVALID&&abs(bitcast<f32>(phi[index]))<bandDistance()){atomicAdd(&control.unresolved,1u);}let d=resolvedDistance(seed,q);workB[index]=bitcast<u32>(d);if(seed!=INVALID&&d<=bandDistance()){atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(64)fn resolveClosestPointsBToA(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);var seed=workB[index];if((flags[index]&INTERFACE)!=0u){seed=index;}if(seed==INVALID&&abs(bitcast<f32>(phi[index]))<bandDistance()){atomicAdd(&control.unresolved,1u);}let d=resolvedDistance(seed,q);workA[index]=bitcast<u32>(d);if(seed!=INVALID&&d<=bandDistance()){atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(64)fn validateJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let d=distanceValue(index);if(!finite(d)){atomicAdd(&control.unresolved,1u);fail(NONFINITE,index);return;}if(d>=bandDistance()||(flags[index]&INTERFACE)!=0u){return;}let q=unpackBrick(metadata[id*10u+1u])*p.brickResolution+localCoord(lid);var sum=0.;for(var axis=0u;axis<3u;axis+=1u){var nearest=d;for(var side=-1;side<=1;side+=2){var nq=vec3i(q);nq[axis]+=side;if(any(nq<vec3i(0))||any(nq>=vec3i(p.sampleDims))){continue;}let neighbor=sampleIndex(vec3u(nq));if(neighbor!=INVALID){nearest=min(nearest,distanceValue(neighbor));}}let gradient=max(0.,d-nearest)/p.fineWidth;sum+=gradient*gradient;}let residual=u32(min(4294967295.,abs(sqrt(sum)-1.)*1000000.));atomicMax(&control.residualScaled,residual);if(residual>u32(p.tolerance*1000000.)){atomicAdd(&control.unresolved,1u);}}
@compute @workgroup_size(1)fn finalizeJFADistances(){control.finalPages=min(worklist[0],p.pageCapacity);if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.unresolved)==0u&&atomicLoad(&control.seeds)>0u){atomicStore(&control.committed,1u);}}
@compute @workgroup_size(64)fn commitJFADistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.committed)==0u){return;}let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;let d=distanceValue(index);let sign=select(1.,-1.,(flags[index]&NEGATIVE)!=0u);phi[index]=bitcast<u32>(sign*d);if(resolvedSeed(index)==INVALID||d>bandDistance()){flags[index]=0u;}else{flags[index]=VALID|(flags[index]&(INTERFACE|NEGATIVE));}}
`;
