import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";

export interface FineLevelSetGPURedistanceOptions {
  /** Required signed-distance width, measured in fine cells. */
  bandCells: number;
  residualTolerance?: number;
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
export const FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES = FINE_LEVELSET_REDISTANCE_CONTROL_BYTES + 80 + 24;

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
 * Exact causal bucketed fast marching on the uniform factor-4/factor-8 fine
 * lattice using B4 pages.  The factor changes only the physical sample width
 * and logical page dimensions; marching and page activation remain entirely
 * in global fine-sample/page coordinates.
 * A half-cell bucket is strictly narrower than the h/sqrt(3) minimum 3-D
 * upwind increment, so every accepted value depends only on earlier buckets.
 * Sample work is indirect-dispatched over live pages. Missing-page requests
 * are emitted and deduplicated in parallel between buckets.  New pages are
 * capacity-reserved, initialized, and linked before their hash keys and live
 * worklist count are published, after which the next bucket may observe them.
 */
export class WebGPUFineLevelSetRedistance {
  readonly control: GPUBuffer;
  readonly allocatedBytes = FINE_LEVELSET_REDISTANCE_ALLOCATED_BYTES;
  private readonly params: GPUBuffer;
  private readonly indirect: GPUBuffer;
  private readonly controlPipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly initializePipeline: GPUComputePipeline;
  private readonly seedPipeline: GPUComputePipeline;
  private readonly marchPipeline: GPUComputePipeline;
  private readonly requestPipeline: GPUComputePipeline;
  private readonly prepareRequestPipeline: GPUComputePipeline;
  private readonly dedupePipeline: GPUComputePipeline;
  private readonly classifyPipeline: GPUComputePipeline;
  private readonly initializeRequestPipeline: GPUComputePipeline;
  private readonly linkRequestPipeline: GPUComputePipeline;
  private readonly copyPublicationPipeline: GPUComputePipeline;
  private readonly reservePublicationPipeline: GPUComputePipeline;
  private readonly installLinksPipeline: GPUComputePipeline;
  private readonly publishRequestPipeline: GPUComputePipeline;
  private readonly finishActivationPipeline: GPUComputePipeline;
  private readonly advancePipeline: GPUComputePipeline;
  private readonly validatePipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;

  constructor(private readonly device: GPUDevice, readonly source: WebGPUFineLevelSetBrickSource) {
    this.control = device.createBuffer({ label: "fine-levelset fast-march control",
      size: FINE_LEVELSET_REDISTANCE_CONTROL_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.params = device.createBuffer({ label: "fine-levelset fast-march params", size: 80,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.indirect = device.createBuffer({ label: "fine-levelset fast-march live dispatch", size: 24,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ label: "fine-levelset uniform fast march", code: fineLevelSetRedistanceWGSL });
    const pipeline = (entryPoint: string) => device.createComputePipeline({ label: `fine fast march ${entryPoint}`,
      layout: "auto", compute: { module, entryPoint } });
    this.controlPipeline = pipeline("initializeControl");
    this.preparePipeline = pipeline("prepareActiveDispatch");
    this.initializePipeline = pipeline("initializeDistances");
    this.seedPipeline = pipeline("seedDistances");
    this.marchPipeline = pipeline("marchBucket");
    this.requestPipeline = pipeline("requestPages");
    this.prepareRequestPipeline = pipeline("prepareRequestDispatch");
    this.dedupePipeline = pipeline("deduplicateRequests");
    this.classifyPipeline = pipeline("classifyRequests");
    this.initializeRequestPipeline = pipeline("initializeRequestedPages");
    this.linkRequestPipeline = pipeline("linkRequestedPages");
    this.copyPublicationPipeline = pipeline("copyPublicationTable");
    this.reservePublicationPipeline = pipeline("reservePublicationSlots");
    this.installLinksPipeline = pipeline("installReverseLinks");
    this.publishRequestPipeline = pipeline("publishRequestedPages");
    this.finishActivationPipeline = pipeline("finishActivation");
    this.advancePipeline = pipeline("advanceBucket");
    this.validatePipeline = pipeline("validateDistances");
    this.finalizePipeline = pipeline("finalizeDistances");
    this.commitPipeline = pipeline("commitDistances");
  }

  encode(encoder: GPUCommandEncoder, options: FineLevelSetGPURedistanceOptions): void {
    if ((this.source.plan.fineFactor !== 4 && this.source.plan.fineFactor !== 8)
      || this.source.plan.brickResolution !== 4) {
      throw new RangeError("GPU fine fast marching requires a factor-4/factor-8 B4 generation");
    }
    if (!Number.isSafeInteger(options.bandCells) || options.bandCells < 1 || options.bandCells > 256) {
      throw new RangeError("Fine redistance bandCells must be an integer in [1, 256]");
    }
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
    const requestWords = this.source.plan.maximumResidentBricks * 6 * 3;
    const activationWords = requestWords + this.source.plan.hashCapacity * 4;
    if (activationWords > u32[15]) {
      throw new RangeError("Fine redistance activation scratch exceeds the shared work buffer");
    }
    this.device.queue.writeBuffer(this.params, 0, bytes); encoder.clearBuffer(this.control);
    const bind = (pipeline: GPUComputePipeline, bindings: readonly (readonly [number, GPUBuffer])[]) =>
      this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: bindings.map(([binding, buffer]) =>
        ({ binding, resource: { buffer } })) });
    const bindings = [[0, this.params], [1, this.source.hash], [2, this.source.metadata],
      [3, this.source.worklist], [4, this.source.flags], [5, this.source.phi],
      [6, this.source.workA], [7, this.source.workB], [8, this.control], [9, this.indirect]] as const;
    const pick = (...wanted: number[]) => bindings.filter(([binding]) => wanted.includes(binding));
    const run = (pipeline: GPUComputePipeline, selected: readonly (readonly [number, GPUBuffer])[], groups: number,
      pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind(pipeline, selected));
      const dispatch = planFineLevelSetDispatch2D(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
      if (dispatch.workgroups > 0) pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    };
    const indirectRun = (pipeline: GPUComputePipeline, selected: readonly (readonly [number, GPUBuffer])[],
      pass: GPUComputePassEncoder, offset = 0) => {
      pass.setPipeline(pipeline); pass.setBindGroup(0, bind(pipeline, selected));
      pass.dispatchWorkgroupsIndirect(this.indirect, offset);
    };
    const computePass = (label: string, encode: (pass: GPUComputePassEncoder) => void) => {
      const pass = encoder.beginComputePass({ label }); encode(pass); pass.end();
    };
    // A compute dispatch is its own WebGPU usage scope. Keeping dependent
    // dispatches in command order inside one pass preserves the exact causal
    // bucket algorithm while avoiding a pass transition per small kernel.
    computePass("Initialize and seed fine fast march", (pass) => {
      run(this.controlPipeline, pick(0, 3, 8), 1, pass);
      run(this.preparePipeline, pick(0, 3, 9), 1, pass);
      indirectRun(this.initializePipeline, pick(0, 2, 3, 4, 5, 6, 8), pass);
      indirectRun(this.seedPipeline, pick(0, 2, 3, 4, 5, 6, 8), pass);
    });
    for (let bucket = 0; bucket < options.bandCells * 2; bucket += 1) {
      computePass(`March and request fine distance bucket ${bucket}`, (pass) => {
        indirectRun(this.marchPipeline, pick(0, 2, 3, 4, 6, 8), pass);
        indirectRun(this.requestPipeline, pick(0, 2, 3, 4, 6, 7, 8), pass);
      });
      // clearBuffer cannot be encoded inside a compute pass, so it is the one
      // required command boundary within a bucket.
      encoder.clearBuffer(this.source.workB, requestWords * 4, this.source.plan.hashCapacity * 4 * 4);
      computePass(`Activate pages and advance fine distance bucket ${bucket}`, (pass) => {
        run(this.prepareRequestPipeline, pick(0, 8, 9), 1, pass);
        indirectRun(this.dedupePipeline, pick(0, 7, 8), pass, 12);
        indirectRun(this.classifyPipeline, pick(0, 1, 4, 7, 8), pass, 12);
        indirectRun(this.initializeRequestPipeline, pick(0, 2, 4, 5, 6, 7, 8), pass, 12);
        indirectRun(this.linkRequestPipeline, pick(0, 1, 2, 7, 8), pass, 12);
        run(this.copyPublicationPipeline, pick(0, 1, 7, 8), Math.ceil(this.source.plan.hashCapacity / 64), pass);
        indirectRun(this.reservePublicationPipeline, pick(0, 7, 8), pass, 12);
        indirectRun(this.installLinksPipeline, pick(0, 2, 7, 8), pass, 12);
        indirectRun(this.publishRequestPipeline, pick(0, 1, 3, 7, 8), pass, 12);
        run(this.finishActivationPipeline, pick(0, 3, 8), 1, pass);
        run(this.preparePipeline, pick(0, 3, 9), 1, pass);
        run(this.advancePipeline, pick(8), 1, pass);
      });
    }
    computePass("Validate, finalize, and commit fine fast march", (pass) => {
      indirectRun(this.validatePipeline, pick(0, 2, 3, 4, 6, 8), pass);
      run(this.finalizePipeline, [[0, this.params], [3, this.source.worklist], [8, this.control]], 1, pass);
      indirectRun(this.commitPipeline, pick(0, 2, 3, 4, 5, 6, 8), pass);
    });
  }

  destroy(): void { this.control.destroy(); this.params.destroy(); this.indirect.destroy(); }
}

export const fineLevelSetRedistanceWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const INTERFACE:u32=2u;const KNOWN:u32=4u;const NEGATIVE:u32=16u;const FRONTIER:u32=32u;const LARGE:f32=3.402823e38;
const CAPACITY:u32=1u;const HASH:u32=2u;const STALE:u32=4u;const NONFINITE:u32=8u;const REQUEST:u32=16u;
struct Params{brickDims:vec3u,brickResolution:u32,sampleDims:vec3u,samplesPerBrick:u32,hashCapacity:u32,maxProbes:u32,pageCapacity:u32,generation:u32,bandCells:u32,fineWidth:f32,tolerance:f32,scratchWords:u32,maxWorkgroups:u32,bucketCount:u32,pad0:u32,pad1:u32}
struct Control{unresolved:atomic<u32>,residualScaled:atomic<u32>,seeds:atomic<u32>,committed:atomic<u32>,flags:atomic<u32>,firstError:atomic<u32>,activated:atomic<u32>,accepted:atomic<u32>,requests:atomic<u32>,initialPages:u32,finalPages:u32,bucket:atomic<u32>}
@group(0)@binding(0)var<uniform>p:Params;@group(0)@binding(1)var<storage,read_write>pageHash:array<atomic<u32>>;@group(0)@binding(2)var<storage,read_write>metadata:array<u32>;@group(0)@binding(3)var<storage,read_write>worklist:array<atomic<u32>>;@group(0)@binding(4)var<storage,read_write>flags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<u32>;@group(0)@binding(6)var<storage,read_write>distance:array<f32>;@group(0)@binding(7)var<storage,read_write>requests:array<atomic<u32>>;@group(0)@binding(8)var<storage,read_write>control:Control;@group(0)@binding(9)var<storage,read_write>indirect:array<atomic<u32>>;
fn finite(v:f32)->bool{return v==v&&abs(v)<LARGE;}fn fail(code:u32,index:u32){atomicOr(&control.flags,code);atomicMin(&control.firstError,index);}fn bandDistance()->f32{return f32(p.bandCells)*p.fineWidth;}fn bucketUpper()->f32{return f32(atomicLoad(&control.bucket)+1u)*(.5*p.fineWidth);}
fn unpackBrick(key:u32)->vec3u{let xy=p.brickDims.x*p.brickDims.y;let z=key/xy;let rem=key-z*xy;let y=rem/p.brickDims.x;return vec3u(rem-y*p.brickDims.x,y,z);}fn packBrick(q:vec3u)->u32{return q.x+p.brickDims.x*(q.y+p.brickDims.y*q.z);}fn localCoord(local:u32)->vec3u{let r=p.brickResolution;let z=local/(r*r);let rem=local-z*r*r;let y=rem/r;return vec3u(rem-y*r,y,z);}fn localIndex(q:vec3u)->u32{return q.x+p.brickResolution*(q.y+p.brickResolution*q.z);}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(p.hashCapacity-1u);}fn pageOf(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let stored=atomicLoad(&pageHash[slot*2u]);if(stored==key){return atomicLoad(&pageHash[slot*2u+1u]);}if(stored==INVALID){return INVALID;}}return INVALID;}
fn brickAcross(key:u32,direction:u32)->u32{var q=vec3i(unpackBrick(key));if(direction==0u){q.x-=1;}else if(direction==1u){q.x+=1;}else if(direction==2u){q.y-=1;}else if(direction==3u){q.y+=1;}else if(direction==4u){q.z-=1;}else{q.z+=1;}if(any(q<vec3i(0))||any(q>=vec3i(p.brickDims))){return INVALID;}return packBrick(vec3u(q));}
fn neighborIndex(id:u32,local:u32,direction:u32)->u32{var q=localCoord(local);var next=id;let r=p.brickResolution;if(direction==0u){if(q.x>0u){q.x-=1u;}else{next=metadata[id*10u+4u];q.x=r-1u;}}else if(direction==1u){if(q.x+1u<r){q.x+=1u;}else{next=metadata[id*10u+5u];q.x=0u;}}else if(direction==2u){if(q.y>0u){q.y-=1u;}else{next=metadata[id*10u+6u];q.y=r-1u;}}else if(direction==3u){if(q.y+1u<r){q.y+=1u;}else{next=metadata[id*10u+7u];q.y=0u;}}else if(direction==4u){if(q.z>0u){q.z-=1u;}else{next=metadata[id*10u+8u];q.z=r-1u;}}else{if(q.z+1u<r){q.z+=1u;}else{next=metadata[id*10u+9u];q.z=0u;}}if(next==INVALID||next>=p.pageCapacity||metadata[next*10u+2u]!=p.generation){return INVALID;}let result=next*p.samplesPerBrick+localIndex(q);return select(INVALID,result,(flags[result]&VALID)!=0u);}
fn activePage(wid:vec3u,nw:vec3u)->u32{let work=fineLinearWorkgroup(wid,nw);let count=min(atomicLoad(&worklist[0]),p.pageCapacity);if(work>=count){return INVALID;}let id=atomicLoad(&worklist[5u+work]);return select(INVALID,id,id<p.pageCapacity&&metadata[id*10u+2u]==p.generation);}
fn solve(a0:f32,a1:f32,a2:f32)->f32{var a=a0;var b=a1;var c=a2;if(a>b){let t=a;a=b;b=t;}if(b>c){let t=b;b=c;c=t;}if(a>b){let t=a;a=b;b=t;}if(a>=LARGE){return LARGE;}let h=p.fineWidth;var result=a+h;if(result>b){result=.5*(a+b+sqrt(max(0.,2.*h*h-(a-b)*(a-b))));}if(result>c){let sum=a+b+c;result=(sum+sqrt(max(0.,sum*sum-3.*(a*a+b*b+c*c-h*h))))/3.;}return result;}
fn candidate(index:u32)->f32{let id=index/p.samplesPerBrick;let local=index-id*p.samplesPerBrick;var axes=vec3f(LARGE);for(var direction=0u;direction<6u;direction+=1u){let neighbor=neighborIndex(id,local,direction);if(neighbor!=INVALID&&(flags[neighbor]&KNOWN)!=0u){axes[direction/2u]=min(axes[direction/2u],distance[neighbor]);}}return solve(axes.x,axes.y,axes.z);}
fn faceLocal(direction:u32,u:u32,v:u32)->u32{let r=p.brickResolution;var q=vec3u(0u);if(direction<2u){q=vec3u(select(0u,r-1u,direction==1u),u,v);}else if(direction<4u){q=vec3u(u,select(0u,r-1u,direction==3u),v);}else{q=vec3u(u,v,select(0u,r-1u,direction==5u));}return localIndex(q);}
fn requestCapacity()->u32{return p.pageCapacity*6u;}fn requestWords()->u32{return requestCapacity()*3u;}fn dedupeBase()->u32{return requestWords();}fn dedupeAt(slot:u32,word:u32)->u32{return dedupeBase()+slot*3u+word;}fn publicationBase()->u32{return dedupeBase()+p.hashCapacity*3u;}fn requestInvocation(wid:vec3u,nw:vec3u,lid:u32)->u32{return fineLinearWorkgroup(wid,nw)*64u+lid;}
fn publishRequest(key:u32,anchor:u32,direction:u32,negative:bool){let request=atomicAdd(&control.requests,1u);let base=request*3u;if(request>=requestCapacity()||base+2u>=p.scratchWords){fail(REQUEST,key);return;}atomicStore(&requests[base],key);atomicStore(&requests[base+1u],anchor);atomicStore(&requests[base+2u],direction|select(0u,8u,negative));}
fn activationSlot(key:u32)->u32{let encoded=key+1u;let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let stored=atomicLoad(&requests[dedupeAt(slot,0u)]);if(stored==encoded){return slot;}if(stored==0u){return INVALID;}}return INVALID;}
fn representativeRequest(request:u32,key:u32)->bool{let slot=activationSlot(key);return slot!=INVALID&&atomicLoad(&requests[dedupeAt(slot,1u)])==request+1u;}
@compute @workgroup_size(1)fn initializeControl(){atomicStore(&control.firstError,INVALID);atomicStore(&control.bucket,0u);atomicStore(&control.requests,0u);let initial=min(atomicLoad(&worklist[0]),p.pageCapacity);control.initialPages=initial;control.finalPages=initial;if(atomicLoad(&worklist[1])!=p.generation){fail(STALE,atomicLoad(&worklist[1]));}}
@compute @workgroup_size(1)fn prepareActiveDispatch(){let count=min(atomicLoad(&worklist[0]),p.pageCapacity);let x=min(count,p.maxWorkgroups);var y=1u;if(x>0u){y=(count+x-1u)/x;}atomicStore(&indirect[0],x);atomicStore(&indirect[1],y);atomicStore(&indirect[2],1u);}
@compute @workgroup_size(64)fn initializeDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;let brick=unpackBrick(metadata[id*10u+1u]);let q=brick*p.brickResolution+localCoord(lid);if(any(q>=p.sampleDims)){flags[index]=0u;return;}let value=bitcast<f32>(phi[index]);if((flags[index]&VALID)==0u||!finite(value)){fail(NONFINITE,index);return;}flags[index]=VALID|select(0u,NEGATIVE,value<0.);distance[index]=bandDistance();}
@compute @workgroup_size(64)fn seedDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u){return;}let center=bitcast<f32>(phi[index]);var seed=center==0.;var seedDistance=select(bandDistance(),0.,seed);for(var direction=0u;direction<6u;direction+=1u){let neighbor=neighborIndex(id,lid,direction);if(neighbor==INVALID){continue;}let other=bitcast<f32>(phi[neighbor]);if(!finite(other)||(other<0.)==(center<0.)){continue;}let denominator=abs(center)+abs(other);seedDistance=min(seedDistance,select(0.,p.fineWidth*abs(center)/denominator,denominator>0.));seed=true;}if(seed){distance[index]=seedDistance;flags[index]|=KNOWN|INTERFACE;atomicAdd(&control.seeds,1u);atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(64)fn marchBucket(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u||atomicLoad(&control.bucket)>=p.bucketCount){return;}let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let index=id*p.samplesPerBrick+lid;if((flags[index]&VALID)==0u||(flags[index]&KNOWN)!=0u){return;}let value=candidate(index);if(finite(value)&&value<bucketUpper()&&value<bandDistance()){distance[index]=value;flags[index]|=KNOWN|FRONTIER;atomicAdd(&control.accepted,1u);}}
@compute @workgroup_size(64)fn requestPages(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(lid!=0u||atomicLoad(&control.flags)!=0u){return;}let id=activePage(wid,nw);if(id==INVALID){return;}let key=metadata[id*10u+1u];let r=p.brickResolution;for(var direction=0u;direction<6u;direction+=1u){if(metadata[id*10u+4u+direction]!=INVALID){continue;}var signMask=0u;for(var u=0u;u<r;u+=1u){for(var v=0u;v<r;v+=1u){let local=faceLocal(direction,u,v);let index=id*p.samplesPerBrick+local;if((flags[index]&FRONTIER)!=0u&&distance[index]+p.fineWidth<bandDistance()){signMask|=select(1u,2u,(flags[index]&NEGATIVE)!=0u);}}}if(signMask!=0u){let requestKey=brickAcross(key,direction);if(requestKey!=INVALID){if(signMask==3u){fail(REQUEST,requestKey);}else{publishRequest(requestKey,id,direction,signMask==2u);}}}}for(var local=0u;local<p.samplesPerBrick;local+=1u){flags[id*p.samplesPerBrick+local]&=(~FRONTIER);}}
@compute @workgroup_size(1)fn prepareRequestDispatch(){let count=atomicLoad(&control.requests);if(count>requestCapacity()){fail(REQUEST,count);}let groups=(min(count,requestCapacity())+63u)/64u;let x=min(groups,p.maxWorkgroups);var y=1u;if(x>0u){y=(groups+x-1u)/x;}atomicStore(&indirect[3],x);atomicStore(&indirect[4],y);atomicStore(&indirect[5],1u);}
@compute @workgroup_size(64)fn deduplicateRequests(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let base=request*3u;let key=atomicLoad(&requests[base]);if(key==INVALID){fail(REQUEST,request);return;}let encoded=key+1u;let sign=select(1u,2u,(atomicLoad(&requests[base+2u])&8u)!=0u);let start=hashKey(key);var probe=0u;for(var attempt=0u;attempt<128u;attempt+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let at=dedupeAt(slot,0u);let observed=atomicLoad(&requests[at]);if(observed==encoded){atomicMax(&requests[at+1u],request+1u);atomicOr(&requests[at+2u],sign);return;}if(observed==0u){let claim=atomicCompareExchangeWeak(&requests[at],0u,encoded);if(claim.exchanged||claim.old_value==encoded){atomicMax(&requests[at+1u],request+1u);atomicOr(&requests[at+2u],sign);return;}if(claim.old_value==0u){continue;}}probe+=1u;}fail(HASH,key);}
@compute @workgroup_size(64)fn classifyRequests(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let base=request*3u;let key=atomicLoad(&requests[base]);let slot=activationSlot(key);if(slot==INVALID){fail(HASH,key);return;}if(atomicLoad(&requests[dedupeAt(slot,1u)])!=request+1u){return;}let signMask=atomicLoad(&requests[dedupeAt(slot,2u)]);if(signMask==0u||signMask==3u){fail(REQUEST,key);return;}let negative=signMask==2u;let existing=pageOf(key);if(existing!=INVALID){var found=false;var existingNegative=false;for(var local=0u;local<p.samplesPerBrick;local+=1u){let index=existing*p.samplesPerBrick+local;if((flags[index]&VALID)!=0u&&!found){found=true;existingNegative=(flags[index]&NEGATIVE)!=0u;}}if(!found||existingNegative!=negative){fail(REQUEST,key);}atomicStore(&requests[base+1u],INVALID);atomicStore(&requests[base+2u],INVALID);return;}let rank=atomicAdd(&control.activated,1u);let id=control.initialPages+rank;if(id>=p.pageCapacity){fail(CAPACITY,key);return;}atomicStore(&requests[base+1u],id);atomicStore(&requests[base+2u],signMask);}
@compute @workgroup_size(64)fn initializeRequestedPages(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let requestBase=request*3u;let key=atomicLoad(&requests[requestBase]);if(!representativeRequest(request,key)){return;}let id=atomicLoad(&requests[requestBase+1u]);if(id==INVALID){return;}let signMask=atomicLoad(&requests[requestBase+2u]);let negative=signMask==2u;let base=id*10u;for(var word=0u;word<10u;word+=1u){metadata[base+word]=INVALID;}metadata[base]=id;metadata[base+1u]=key;metadata[base+2u]=p.generation;metadata[base+3u]=1u;let brick=unpackBrick(key);for(var local=0u;local<p.samplesPerBrick;local+=1u){let index=id*p.samplesPerBrick+local;let q=brick*p.brickResolution+localCoord(local);let state=select(VALID,VALID|NEGATIVE,negative);flags[index]=select(0u,state,all(q<p.sampleDims));distance[index]=bandDistance();phi[index]=bitcast<u32>(select(bandDistance(),-bandDistance(),negative));}}
@compute @workgroup_size(64)fn linkRequestedPages(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let requestBase=request*3u;let key=atomicLoad(&requests[requestBase]);if(!representativeRequest(request,key)){return;}let id=atomicLoad(&requests[requestBase+1u]);if(id==INVALID){return;}let base=id*10u;for(var direction=0u;direction<6u;direction+=1u){let neighborKey=brickAcross(key,direction);if(neighborKey==INVALID){continue;}var neighbor=pageOf(neighborKey);if(neighbor==INVALID){let neighborSlot=activationSlot(neighborKey);if(neighborSlot!=INVALID){let neighborRequest=atomicLoad(&requests[dedupeAt(neighborSlot,1u)]);if(neighborRequest!=0u){neighbor=atomicLoad(&requests[(neighborRequest-1u)*3u+1u]);}}}if(neighbor==INVALID){continue;}if(neighbor>=p.pageCapacity||metadata[neighbor*10u+2u]!=p.generation){fail(STALE,neighbor);return;}if(neighbor<control.initialPages&&metadata[neighbor*10u+4u+(direction^1u)]!=INVALID){fail(REQUEST,neighborKey);return;}metadata[base+4u+direction]=neighbor;}}
@compute @workgroup_size(64)fn copyPublicationTable(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let slot=requestInvocation(wid,nw,lid);if(slot>=p.hashCapacity){return;}let key=atomicLoad(&pageHash[slot*2u]);atomicStore(&requests[publicationBase()+slot],select(0u,key+1u,key!=INVALID));}
@compute @workgroup_size(64)fn reservePublicationSlots(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let base=request*3u;let key=atomicLoad(&requests[base]);if(!representativeRequest(request,key)||atomicLoad(&requests[base+1u])==INVALID){return;}let encoded=key+1u;let start=hashKey(key);var probe=0u;for(var attempt=0u;attempt<128u;attempt+=1u){if(probe>=p.maxProbes){break;}let slot=(start+probe)&(p.hashCapacity-1u);let at=publicationBase()+slot;var observed=atomicLoad(&requests[at]);if(observed==0u){let claim=atomicCompareExchangeWeak(&requests[at],0u,encoded);if(claim.exchanged){atomicStore(&requests[base+2u],slot);return;}if(claim.old_value==0u){continue;}observed=claim.old_value;}if(observed==encoded){fail(REQUEST,key);return;}probe+=1u;}fail(HASH,key);}
@compute @workgroup_size(64)fn installReverseLinks(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let base=request*3u;let key=atomicLoad(&requests[base]);if(!representativeRequest(request,key)){return;}let id=atomicLoad(&requests[base+1u]);if(id==INVALID){return;}for(var direction=0u;direction<6u;direction+=1u){let neighbor=metadata[id*10u+4u+direction];if(neighbor!=INVALID&&neighbor<control.initialPages){metadata[neighbor*10u+4u+(direction^1u)]=id;}}}
@compute @workgroup_size(64)fn publishRequestedPages(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.flags)!=0u){return;}let request=requestInvocation(wid,nw,lid);let count=min(atomicLoad(&control.requests),requestCapacity());if(request>=count){return;}let base=request*3u;let key=atomicLoad(&requests[base]);if(!representativeRequest(request,key)){return;}let id=atomicLoad(&requests[base+1u]);if(id==INVALID){return;}let slot=atomicLoad(&requests[base+2u]);atomicStore(&worklist[5u+id],id);atomicStore(&pageHash[slot*2u+1u],id);atomicStore(&pageHash[slot*2u],key);}
@compute @workgroup_size(1)fn finishActivation(){if(atomicLoad(&control.flags)!=0u){return;}let count=control.initialPages+atomicLoad(&control.activated);if(count>p.pageCapacity){fail(CAPACITY,count);return;}atomicStore(&worklist[0],count);atomicStore(&worklist[1],p.generation);atomicStore(&worklist[2],(count+64u)/64u);atomicStore(&worklist[3],1u);atomicStore(&worklist[4],1u);control.finalPages=count;}
@compute @workgroup_size(1)fn advanceBucket(){atomicStore(&control.requests,0u);atomicAdd(&control.bucket,1u);}
@compute @workgroup_size(64)fn validateDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let flat=id*p.samplesPerBrick+lid;if((flags[flat]&VALID)==0u){return;}let d=distance[flat];if(!finite(d)){atomicAdd(&control.unresolved,1u);fail(NONFINITE,flat);return;}if(d>=bandDistance()){return;}if((flags[flat]&KNOWN)==0u){atomicAdd(&control.unresolved,1u);return;}if((flags[flat]&INTERFACE)!=0u){return;}var sum=0.;for(var axis=0u;axis<3u;axis+=1u){let left=neighborIndex(id,lid,axis*2u);let right=neighborIndex(id,lid,axis*2u+1u);var nearest=d;if(left!=INVALID&&(flags[left]&KNOWN)!=0u){nearest=min(nearest,distance[left]);}if(right!=INVALID&&(flags[right]&KNOWN)!=0u){nearest=min(nearest,distance[right]);}let gradient=max(0.,d-nearest)/p.fineWidth;sum+=gradient*gradient;}let residual=u32(min(4294967295.,abs(sqrt(sum)-1.)*1000000.));atomicMax(&control.residualScaled,residual);if(residual>u32(p.tolerance*1000000.)){atomicAdd(&control.unresolved,1u);}}
@compute @workgroup_size(1)fn finalizeDistances(){control.finalPages=min(atomicLoad(&worklist[0]),p.pageCapacity);if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.unresolved)==0u&&atomicLoad(&control.seeds)>0u&&atomicLoad(&control.bucket)==p.bucketCount){atomicStore(&control.committed,1u);}}
@compute @workgroup_size(64)fn commitDistances(@builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)nw:vec3u){if(atomicLoad(&control.committed)==0u){return;}let id=activePage(wid,nw);if(id==INVALID||lid>=p.samplesPerBrick){return;}let flat=id*p.samplesPerBrick+lid;let d=distance[flat];let sign=select(1.,-1.,(flags[flat]&NEGATIVE)!=0u);phi[flat]=bitcast<u32>(sign*d);if(d>=bandDistance()||(flags[flat]&KNOWN)==0u){flags[flat]=0u;}else{flags[flat]=VALID|(flags[flat]&(INTERFACE|NEGATIVE));}}
`;
