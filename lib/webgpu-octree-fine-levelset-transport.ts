import type { WebGPUFineLevelSetBrickSource } from "./webgpu-octree-fine-levelset-bricks";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import type {
  WebGPUOctreePowerVelocityPrepass,
} from "./webgpu-octree-power-velocity-prepass";
import type {
  OctreeFaceBandSampleOptions,
  WebGPUOctreeFaceFastMarch,
} from "./webgpu-octree-face-fast-march";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const FINE_LEVELSET_TRANSPORT_CONTROL_BYTES = 64;

export interface FineLevelSetGPUTransportControl {
  departureOutsideBand: number;
  nonfiniteVelocity: number;
  processed: number;
  committed: boolean;
  extrapolatedVelocity: number;
  maximumDisplacementFineCells: number;
  faceBandUnavailable: number;
  velocityUnavailable: number;
  invalidVelocityStatus?: number;
  nonpositiveVelocityResult?: number;
  velocityStatusReasonOr?: number;
  firstInvalidVelocityStatus?: number;
  firstInvalidVelocityLocalIndex?: number;
  firstInvalidVelocityPosition?: readonly [number, number, number];
}

export type FineLevelSetGPUBoundaryPolicy = "strict" | "closed-neumann";

export interface FineLevelSetGPUTransportOptions {
  timestep: number;
  headers: GPUBuffer;
  rowVelocities: GPUBuffer;
  dimensions: readonly [number, number, number];
  physicalCellSize: number;
  maximumLeafSize: number;
  /** Adaptive owner authority used to resolve Section 5 regular-face rows. */
  ownerTopology?: GPUBuffer;
  /** Current catalog/Delaunay authority used by the Section 5 point field. */
  powerTopology?: OctreePowerTopologySource;
  generation?: number;
  maximumHashProbes?: number;
  /** Destination band transported this step; outer valid samples are interpolation support only. */
  transportBandCells?: number;
  /** Closed boundaries extend cell-centred phi constantly through the in-domain wall half-cell. */
  boundaryPolicy?: FineLevelSetGPUBoundaryPolicy;
  /** Open ceilings extend the top air field normally for outflow characteristics. */
  openTopBoundary?: boolean;
}

export interface FineLevelSetGPUTransportPlan {
  readonly queryCapacity: number;
  readonly velocityChunkCapacity: number;
  readonly positionCapacity: number;
  readonly positionBytes: number;
  readonly chunkCount: number;
  readonly chunkParameterStride: number;
  readonly chunkParameterBytes: number;
  readonly controlBytes: number;
  readonly allocatedBytes: number;
}

export interface FineLevelSetGPUTransportPassPlan {
  readonly chunkCount: number;
  readonly segmentCount: number;
  readonly passesPerSegment: number;
  readonly passesPerChunk: number;
  readonly encodedPasses: number;
}

export function planFineLevelSetGPUTransport(queryCapacity: number,
  velocityChunkCapacity: number, chunkParameterAlignment = 256): FineLevelSetGPUTransportPlan {
  if (!Number.isSafeInteger(queryCapacity) || queryCapacity < 1
    || !Number.isSafeInteger(velocityChunkCapacity) || velocityChunkCapacity < 1
    || !Number.isSafeInteger(chunkParameterAlignment) || chunkParameterAlignment < 4
    || chunkParameterAlignment % 4 !== 0) {
    throw new RangeError("Fine transport capacities must be positive integers");
  }
  const positionCapacity = velocityChunkCapacity;
  const positionBytes = positionCapacity * 16;
  const chunkCount = Math.ceil(queryCapacity / velocityChunkCapacity);
  const chunkParameterStride = chunkParameterAlignment, chunkParameterBytes = chunkCount * chunkParameterStride;
  return { queryCapacity, velocityChunkCapacity, positionCapacity, positionBytes,
    chunkCount, chunkParameterStride, chunkParameterBytes,
    controlBytes: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES,
    allocatedBytes: positionBytes + chunkParameterBytes + FINE_LEVELSET_TRANSPORT_CONTROL_BYTES };
}

/**
 * Static command-count contract for Section 5's piecewise-linear trace.
 * Each segment retains a fresh Stage-B query build, prepare/sample/publish,
 * optional air-band override, and trajectory advance.  Batching changes only
 * how many starting samples share those passes; it never removes a velocity
 * evaluation from the paper's m-segment trace.
 */
export function planFineLevelSetGPUTransportPasses(
  plan: Pick<FineLevelSetGPUTransportPlan, "chunkCount">,
  segmentCount: 4 | 8,
  includesFaceBand = true,
): FineLevelSetGPUTransportPassPlan {
  if (!Number.isSafeInteger(plan.chunkCount) || plan.chunkCount < 1) {
    throw new RangeError("Fine transport pass plan requires at least one chunk");
  }
  const passesPerSegment = 5 + (includesFaceBand ? 1 : 0);
  const passesPerChunk = 2 + segmentCount * passesPerSegment;
  return {
    chunkCount: plan.chunkCount,
    segmentCount,
    passesPerSegment,
    passesPerChunk,
    encodedPasses: plan.chunkCount * passesPerChunk + 2,
  };
}

export function unpackFineLevelSetGPUTransportControl(words: ArrayLike<number>): FineLevelSetGPUTransportControl {
  if (words.length < 8) throw new RangeError("Fine transport control needs eight words");
  return { departureOutsideBand: Number(words[0]) >>> 0, nonfiniteVelocity: Number(words[1]) >>> 0,
    processed: Number(words[2]) >>> 0, committed: Number(words[3]) !== 0,
    extrapolatedVelocity: Number(words[4]) >>> 0, maximumDisplacementFineCells: Number(words[5]) >>> 0,
    faceBandUnavailable: Number(words[6]) >>> 0, velocityUnavailable: Number(words[7]) >>> 0,
    ...(words.length >= 12 ? { invalidVelocityStatus: Number(words[8]) >>> 0,
      nonpositiveVelocityResult: Number(words[9]) >>> 0,
      velocityStatusReasonOr: Number(words[10]) >>> 0,
      firstInvalidVelocityStatus: Number(words[11]) >>> 0 } : {}),
    ...(words.length >= 16 ? { firstInvalidVelocityLocalIndex: Number(words[12]) >>> 0,
      firstInvalidVelocityPosition: [new Float32Array(new Uint32Array([Number(words[13]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[14]) >>> 0]).buffer)[0],
        new Float32Array(new Uint32Array([Number(words[15]) >>> 0]).buffer)[0]] as const } : {}) };
}

/**
 * Paper section 5 transport on the global uniform fine lattice.  Trajectory
 * positions stay GPU-resident and Stage-B octree velocity is queried again
 * before every one of the m=fineFactor piecewise-linear segments.
 */
export class WebGPUFineLevelSetTransport {
  readonly positions: GPUBuffer;
  readonly control: GPUBuffer;
  readonly queryCapacity: number;
  readonly plan: FineLevelSetGPUTransportPlan;
  private readonly positionCapacity: number;
  private readonly chunkParameters: GPUBuffer;
  private readonly chunkParameterWords: Uint32Array<ArrayBuffer>;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly advancePipeline: GPUComputePipeline;
  private readonly samplePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly commitPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    readonly source: WebGPUFineLevelSetBrickSource,
    private readonly velocityPrepass: Pick<WebGPUOctreePowerVelocityPrepass, "encodeFromPositions" | "source">,
    /** Paper Section 5 face-band velocity authority. Stage B remains the
     * primary liquid interpolant; positive-air samples and exact local-catalog
     * coverage misses are completed from fast-marched regular octree faces. */
    private readonly faceBand?: Pick<WebGPUOctreeFaceFastMarch, "encodeAirSamples">,
  ) {
    this.queryCapacity = source.plan.maximumResidentBricks * source.plan.samplesPerBrick;
    if (velocityPrepass.source.queryCapacity < this.queryCapacity
      && (velocityPrepass.source.queryCapacity * 16) % device.limits.minStorageBufferOffsetAlignment !== 0) {
      throw new RangeError("Fine transport velocity chunk must satisfy storage-buffer offset alignment");
    }
    this.plan = planFineLevelSetGPUTransport(this.queryCapacity, velocityPrepass.source.queryCapacity,
      device.limits.minUniformBufferOffsetAlignment);
    this.positionCapacity = this.plan.positionCapacity;
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.positions = device.createBuffer({ label: "fine-levelset trajectory positions",
      size: this.plan.positionBytes, usage: storage });
    this.chunkParameters = device.createBuffer({ label: "fine-levelset trajectory chunk parameters",
      size: this.plan.chunkParameterBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const chunkWords = new Uint32Array(this.plan.chunkParameterBytes / 4);
    for (let chunk = 0; chunk < this.plan.chunkCount; chunk += 1) {
      chunkWords[chunk * this.plan.chunkParameterStride / 4] = chunk * this.plan.velocityChunkCapacity;
    }
    this.chunkParameterWords = chunkWords;
    device.queue.writeBuffer(this.chunkParameters, 0, chunkWords);
    this.control = device.createBuffer({ label: "fine-levelset transport control",
      size: FINE_LEVELSET_TRANSPORT_CONTROL_BYTES, usage: storage });
    const shaderModule = device.createShaderModule({ label: "fine-levelset GPU query transport",
      code: fineLevelSetGPUQueryTransportWGSL });
    const pipeline = (entryPoint: string, label: string) => device.createComputePipeline({ label, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.preparePipeline = pipeline("prepareFineTrajectories", "Prepare fine trajectories");
    this.advancePipeline = pipeline("advanceFineTrajectories", "Advance fine trajectories");
    this.samplePipeline = pipeline("sampleFineDepartures", "Sample fine departures");
    this.publishPipeline = pipeline("publishFineTransport", "Publish fine transport status");
    this.commitPipeline = pipeline("commitFineTransport", "Commit fine transport");
  }

  encode(encoder: GPUCommandEncoder, options: FineLevelSetGPUTransportOptions): void {
    if (this.destroyed) throw new Error("Fine level-set transport is destroyed");
    if (!Number.isFinite(options.timestep) || options.timestep < 0) {
      throw new RangeError("Fine level-set transport timestep must be finite and non-negative");
    }
    if (this.faceBand && (!options.ownerTopology || !options.powerTopology)) {
      throw new RangeError("Fine transport face-band sampling requires owner and power topology authority");
    }
    const transportBandCells = options.transportBandCells ?? 0xffff;
    if (!Number.isSafeInteger(transportBandCells) || transportBandCells < 1 || transportBandCells > 0xffff) {
      throw new RangeError("Fine level-set transport band must be a positive integer");
    }
    if (this.faceBand && !options.ownerTopology) {
      throw new RangeError("Fine transport face-band sampling requires adaptive owner topology");
    }
    const chunkFloats = new Float32Array(this.chunkParameterWords.buffer);
    const closedDomainBoundary = options.boundaryPolicy === "closed-neumann" ? 1 : 0;
    for (let chunk = 0; chunk < this.plan.chunkCount; chunk += 1) {
      chunkFloats[chunk * this.plan.chunkParameterStride / 4 + 1]
        = transportBandCells * this.source.plan.fineCellWidth;
      this.chunkParameterWords[chunk * this.plan.chunkParameterStride / 4 + 2] = closedDomainBoundary;
      this.chunkParameterWords[chunk * this.plan.chunkParameterStride / 4 + 3] = options.openTopBoundary ? 1 : 0;
    }
    this.device.queue.writeBuffer(this.chunkParameters, 0, this.chunkParameterWords);
    this.device.queue.writeBuffer(this.source.params, 76, new Float32Array([options.timestep]));
    const resetControl = new Uint32Array(16); resetControl[11] = 0xffff_ffff; resetControl[12] = 0xffff_ffff;
    this.device.queue.writeBuffer(this.control, 0, resetControl);
    const binding = (buffer: GPUBuffer): GPUBufferBinding => ({ buffer });
    const run = (pipeline: GPUComputePipeline, entries: readonly GPUBindGroupEntry[], label: string,
      workgroups = Math.ceil(this.positionCapacity / 64), tiled = false) => {
      const pass = encoder.beginComputePass({ label });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }));
      if (tiled) { const dispatch = planFineLevelSetDispatch2D(workgroups,
        this.device.limits.maxComputeWorkgroupsPerDimension); pass.dispatchWorkgroups(dispatch.x, dispatch.y); }
      else pass.dispatchWorkgroups(workgroups);
      pass.end();
    };
    const common: GPUBindGroupEntry[] = [
      { binding: 0, resource: binding(this.source.params) },
      { binding: 2, resource: binding(this.source.metadata) },
      { binding: 3, resource: binding(this.source.worklist) },
      { binding: 4, resource: binding(this.source.flags) },
    ];
    const prepassOptions = { dimensions: options.dimensions, physicalCellSize: options.physicalCellSize,
      maximumLeafSize: options.maximumLeafSize, queryCount: this.queryCapacity,
      generation: options.generation, maximumHashProbes: options.maximumHashProbes };
    const chunkCapacity = this.velocityPrepass.source.queryCapacity;
    for (let chunk = 0; chunk < this.plan.chunkCount; chunk += 1) {
      const chunkBinding: GPUBufferBinding = { buffer: this.chunkParameters,
        offset: chunk * this.plan.chunkParameterStride, size: 16 };
      run(this.preparePipeline, [...common, { binding: 5, resource: binding(this.source.phi) },
        { binding: 6, resource: binding(this.source.workA) },
        { binding: 10, resource: binding(this.positions) },
        { binding: 11, resource: chunkBinding }],
        `Prepare global fine trajectory chunk ${chunk + 1}/${this.plan.chunkCount}`);
      for (let segment = 0; segment < this.source.plan.fineFactor; segment += 1) {
        const count = chunkCapacity;
        const positionSlice: GPUBufferBinding = { buffer: this.positions, offset: 0, size: count * 16 };
        this.velocityPrepass.encodeFromPositions(encoder, positionSlice, options.headers, options.rowVelocities,
          { ...prepassOptions, queryCount: count });
        this.faceBand?.encodeAirSamples(encoder, positionSlice,
          this.velocityPrepass.source.results, this.velocityPrepass.source.statuses, {
            dimensions: options.dimensions,
            maximumLeafSize: options.maximumLeafSize,
            queryCount: count,
            physicalCellSize: options.physicalCellSize,
            owners: options.ownerTopology!,
            fineGeneration: this.source.generation,
            powerTopology: options.powerTopology!,
          } satisfies OctreeFaceBandSampleOptions);
        run(this.advancePipeline, [
          { binding: 0, resource: binding(this.source.params) },
          { binding: 7, resource: binding(this.control) },
          { binding: 8, resource: binding(this.velocityPrepass.source.results) },
          { binding: 9, resource: binding(this.velocityPrepass.source.statuses) },
          { binding: 10, resource: positionSlice },
        ], `Advance global fine trajectories ${segment + 1}/${this.source.plan.fineFactor}`);
      }
      run(this.samplePipeline, [
        { binding: 0, resource: binding(this.source.params) }, { binding: 1, resource: binding(this.source.hash) },
        { binding: 2, resource: binding(this.source.metadata) }, { binding: 3, resource: binding(this.source.worklist) },
        { binding: 4, resource: binding(this.source.flags) }, { binding: 5, resource: binding(this.source.phi) },
        { binding: 6, resource: binding(this.source.workA) }, { binding: 7, resource: binding(this.control) },
        { binding: 10, resource: binding(this.positions) }, { binding: 11, resource: chunkBinding },
      ], `Sample global fine departure chunk ${chunk + 1}/${this.plan.chunkCount}`);
    }
    run(this.publishPipeline, [{ binding: 7, resource: binding(this.control) }],
      "Publish global fine transport", 1);
    run(this.commitPipeline, [...common,
      { binding: 5, resource: binding(this.source.phi) }, { binding: 6, resource: binding(this.source.workA) },
      { binding: 7, resource: binding(this.control) }],
    "Commit global fine transport", Math.ceil(this.queryCapacity / 64), true);
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.positions.destroy();
    this.chunkParameters.destroy();
    this.control.destroy();
  }
}

export const fineLevelSetGPUQueryTransportWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
const INVALID:u32=0xffffffffu;const VALID:u32=1u;const VELOCITY_VALID:u32=0x80000000u;const LARGE:f32=3.402823e38;
struct Params{brickDimensions:vec3u,brickResolution:u32,sampleDimensions:vec3u,samplesPerBrick:u32,
 domainOrigin:vec3f,fineCellWidth:f32,hashCapacity:u32,maximumHashProbes:u32,pageCapacity:u32,generation:u32,
 activeCount:u32,invalid:u32,fineFactor:u32,timestep:f32}
struct Control{departureOutsideBand:atomic<u32>,nonfiniteVelocity:atomic<u32>,processed:atomic<u32>,committed:atomic<u32>,extrapolatedVelocity:atomic<u32>,maximumDisplacementFineCells:atomic<u32>,faceBandUnavailable:atomic<u32>,velocityUnavailable:atomic<u32>,invalidVelocityStatus:atomic<u32>,nonpositiveVelocityResult:atomic<u32>,velocityStatusReasonOr:atomic<u32>,firstInvalidVelocityStatus:atomic<u32>,firstInvalidVelocityLocalIndex:atomic<u32>,firstInvalidVelocityX:atomic<u32>,firstInvalidVelocityY:atomic<u32>,firstInvalidVelocityZ:atomic<u32>}
struct Chunk{base:u32,transportBandDistance:f32,closedDomainBoundary:u32,openTopBoundary:u32}
@group(0)@binding(0)var<uniform>params:Params;@group(0)@binding(1)var<storage,read>pageHash:array<u32>;
@group(0)@binding(2)var<storage,read>metadata:array<u32>;@group(0)@binding(3)var<storage,read>worklist:array<u32>;
@group(0)@binding(4)var<storage,read>sampleFlags:array<u32>;@group(0)@binding(5)var<storage,read_write>phi:array<f32>;
@group(0)@binding(6)var<storage,read_write>workA:array<f32>;@group(0)@binding(7)var<storage,read_write>control:Control;
@group(0)@binding(8)var<storage,read>velocities:array<vec4f>;@group(0)@binding(9)var<storage,read>velocityStatus:array<u32>;
@group(0)@binding(10)var<storage,read_write>positions:array<vec4f>;
@group(0)@binding(11)var<uniform>chunk:Chunk;
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<vec3f(LARGE));}
fn activeSample(flat:u32)->vec2u{let count=min(worklist[0],params.pageCapacity);if(flat>=count*params.samplesPerBrick){return vec2u(INVALID);}
 let w=flat/params.samplesPerBrick;let local=flat-w*params.samplesPerBrick;let id=worklist[5u+w];
 if(id>=params.pageCapacity||metadata[id*10u+2u]!=params.generation){return vec2u(INVALID);}return vec2u(id,local);}
fn unpackBrick(key:u32)->vec3u{let xy=params.brickDimensions.x*params.brickDimensions.y;let z=key/xy;let r=key-z*xy;let y=r/params.brickDimensions.x;return vec3u(r-y*params.brickDimensions.x,y,z);}
fn localCoord(local:u32)->vec3u{let r=params.brickResolution;let z=local/(r*r);let q=local-z*r*r;let y=q/r;return vec3u(q-y*r,y,z);}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(params.hashCapacity-1u);}fn packBrick(q:vec3u)->u32{return q.x+params.brickDimensions.x*(q.y+params.brickDimensions.y*q.z);}
fn lookup(key:u32)->u32{let start=hashKey(key);for(var probe=0u;probe<32u;probe+=1u){if(probe>=params.maximumHashProbes){break;}let slot=(start+probe)&(params.hashCapacity-1u);let stored=pageHash[slot*2u];if(stored==key){let id=pageHash[slot*2u+1u];if(id>=params.pageCapacity){return INVALID;}let base=id*10u;if(metadata[base]!=id||metadata[base+1u]!=key||metadata[base+2u]!=params.generation){return INVALID;}return id;}if(stored==INVALID){return INVALID;}}return INVALID;}
fn loadFine(q:vec3i)->vec3f{if(any(q<vec3i(0))||any(q>=vec3i(params.sampleDimensions))){return vec3f(0.,0.,1.);}
 let uq=vec3u(q);let brick=uq/params.brickResolution;let local=uq-brick*params.brickResolution;let id=lookup(packBrick(brick));if(id==INVALID){return vec3f(0.,0.,2.);}
 let i=id*params.samplesPerBrick+local.x+params.brickResolution*(local.y+params.brickResolution*local.z);if((sampleFlags[i]&VALID)==0u){return vec3f(0.,0.,3.);}
 let v=phi[i];if(v!=v||abs(v)>=LARGE){return vec3f(0.,0.,4.);}return vec3f(v,1.,0.);}
fn trilinear(x:vec3f)->vec3f{let raw=(x-params.domainOrigin)/params.fineCellWidth-vec3f(.5);let wall=vec3f(params.sampleDimensions)-vec3f(.5);let wallTolerance=1e-3;
 // Section 5 traces and interpolates the old level set.  A closed-domain
 // characteristic may land on the physical wall, half a sample beyond the
 // outer cell centre.  Accept small integration drift around that wall.  An
 // authored open ceiling extends the top value normally for outflow; every
 // other materially exterior trace still fails publication instead of
 // sampling an absent sparse brick.
 let below=raw<vec3f(-.5-wallTolerance);let above=raw>wall+vec3f(wallTolerance);
 if(any(below)||above.x||above.z||(above.y&&chunk.openTopBoundary==0u)){return vec3f(0.,0.,1.);}let sampleMax=vec3f(params.sampleDimensions)-vec3f(1.);if(chunk.closedDomainBoundary==0u&&(any(raw<vec3f(0.))||raw.x>sampleMax.x||raw.z>sampleMax.z||(raw.y>sampleMax.y&&chunk.openTopBoundary==0u))){return vec3f(0.,0.,1.);}let extendBoundary=chunk.closedDomainBoundary!=0u||chunk.openTopBoundary!=0u;let lattice=select(raw,clamp(raw,vec3f(0.),sampleMax),extendBoundary);let base=vec3i(floor(lattice));let f=fract(lattice);var v=0.;
 for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x0=0;x0<2;x0+=1){let w=select(1.-f.x,f.x,x0==1)*select(1.-f.y,f.y,y==1)*select(1.-f.z,f.z,z==1);if(w==0.){continue;}let q=loadFine(base+vec3i(x0,y,z));if(q.y==0.){return q;}v+=w*q.x;}}}return vec3f(v,1.,0.);}
@compute @workgroup_size(64)fn prepareFineTrajectories(@builtin(global_invocation_id)g:vec3u){let local=g.x;if(local>=arrayLength(&positions)){return;}positions[local]=vec4f(0);let flat=chunk.base+local;let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*params.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)==0u){return;}
 // Section 5 assigns an interpolated value only to traced starting cells.
 // Preserve the outer valid halo verbatim so the all-valid commit below can
 // never publish stale shared scratch for support-only samples.
 workA[index]=phi[index];
 if(abs(phi[index])>=chunk.transportBandDistance){return;}
 let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*params.brickResolution+localCoord(a.y);positions[local]=vec4f(params.domainOrigin+(vec3f(q)+.5)*params.fineCellWidth,1);}
@compute @workgroup_size(64)fn advanceFineTrajectories(@builtin(global_invocation_id)g:vec3u){let i=g.x;if(i>=arrayLength(&positions)||positions[i].w<=0.){return;}
 if(i>=arrayLength(&velocities)||i>=arrayLength(&velocityStatus)){atomicAdd(&control.velocityUnavailable,1u);positions[i].w=0.;return;}let status=velocityStatus[i];if((status&0x08000000u)!=0u){atomicAdd(&control.faceBandUnavailable,1u);}if((status&VELOCITY_VALID)==0u){atomicAdd(&control.invalidVelocityStatus,1u);atomicOr(&control.velocityStatusReasonOr,status);let claim=atomicCompareExchangeWeak(&control.firstInvalidVelocityLocalIndex,INVALID,i);if(claim.exchanged){atomicStore(&control.firstInvalidVelocityStatus,status);atomicStore(&control.firstInvalidVelocityX,bitcast<u32>(positions[i].x));atomicStore(&control.firstInvalidVelocityY,bitcast<u32>(positions[i].y));atomicStore(&control.firstInvalidVelocityZ,bitcast<u32>(positions[i].z));}atomicAdd(&control.velocityUnavailable,1u);positions[i].w=0.;return;}if(velocities[i].w<=0.){atomicAdd(&control.nonpositiveVelocityResult,1u);atomicAdd(&control.velocityUnavailable,1u);positions[i].w=0.;return;}if(!finite3(velocities[i].xyz)){atomicAdd(&control.nonfiniteVelocity,1u);positions[i].w=0.;return;}
 if((velocityStatus[i]&0x10000000u)!=0u){atomicAdd(&control.extrapolatedVelocity,1u);}
 let next=positions[i].xyz-(params.timestep/f32(params.fineFactor))*velocities[i].xyz;if(!finite3(next)){atomicAdd(&control.nonfiniteVelocity,1u);positions[i].w=0.;return;}positions[i]=vec4f(next,1);}
@compute @workgroup_size(64)fn sampleFineDepartures(@builtin(global_invocation_id)g:vec3u){let local=g.x;if(local>=arrayLength(&positions)||positions[local].w<=0.){return;}let flat=chunk.base+local;let a=activeSample(flat);if(a.x==INVALID){return;}let brick=unpackBrick(metadata[a.x*10u+1u]);let q=brick*params.brickResolution+localCoord(a.y);let origin=params.domainOrigin+(vec3f(q)+.5)*params.fineCellWidth;let displacement=u32(ceil(length(positions[local].xyz-origin)/params.fineCellWidth));atomicMax(&control.maximumDisplacementFineCells,displacement);let value=trilinear(positions[local].xyz);if(value.y==0.){atomicAdd(&control.departureOutsideBand,1u);let claim=atomicCompareExchangeWeak(&control.firstInvalidVelocityLocalIndex,INVALID,local);if(claim.exchanged){atomicStore(&control.firstInvalidVelocityStatus,0x04000000u|u32(value.z));atomicStore(&control.firstInvalidVelocityX,bitcast<u32>(positions[local].x));atomicStore(&control.firstInvalidVelocityY,bitcast<u32>(positions[local].y));atomicStore(&control.firstInvalidVelocityZ,bitcast<u32>(positions[local].z));}return;}workA[a.x*params.samplesPerBrick+a.y]=value.x;atomicAdd(&control.processed,1u);}
@compute @workgroup_size(1)fn publishFineTransport(){if(atomicLoad(&control.departureOutsideBand)==0u&&atomicLoad(&control.nonfiniteVelocity)==0u&&atomicLoad(&control.velocityUnavailable)==0u){atomicStore(&control.committed,1u);}}
@compute @workgroup_size(64)fn commitFineTransport(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){if(atomicLoad(&control.committed)==0u){return;}let flat=fineLinearWorkgroup(w,n)*64u+lid;let a=activeSample(flat);if(a.x==INVALID){return;}let index=a.x*params.samplesPerBrick+a.y;if((sampleFlags[index]&VALID)!=0u){phi[index]=workA[index];}}
`;

/**
 * Generates the global fine-grid transport shader around an injected octree
 * velocity interpolant. The injected source must define:
 *
 *   fn sampleOctreeVelocity(position: vec3f) -> vec3f
 *
 * This keeps velocity on the octree/power-Delaunay representation. No fine
 * velocity channel or compatibility texture is introduced.
 */
export function makeFineLevelSetTransportWGSL(velocitySamplerWGSL: string,
  boundaryPolicy: FineLevelSetGPUBoundaryPolicy = "strict"): string {
  if (!/fn\s+sampleOctreeVelocity\s*\(/.test(velocitySamplerWGSL)) {
    throw new RangeError("Fine level-set transport requires an injected sampleOctreeVelocity function");
  }
  if (boundaryPolicy !== "strict" && boundaryPolicy !== "closed-neumann") {
    throw new RangeError("Fine level-set boundary policy is invalid");
  }
  return /* wgsl */ `
const INVALID:u32=0xffffffffu;
const VALID:u32=1u;
const CLOSED_DOMAIN_BOUNDARY:bool=${boundaryPolicy === "closed-neumann"};
struct Params {
  brickDimensions:vec3u, brickResolution:u32,
  sampleDimensions:vec3u, samplesPerBrick:u32,
  domainOrigin:vec3f, fineCellWidth:f32,
  hashCapacity:u32, maximumHashProbes:u32, pageCapacity:u32, generation:u32,
  activeCount:u32, invalid:u32, fineFactor:u32, timestep:f32,
}
struct Control { departureOutsideBand:atomic<u32>, nonfiniteVelocity:atomic<u32>, processed:atomic<u32>, committed:atomic<u32> }
@group(0) @binding(0) var<uniform> params:Params;
@group(0) @binding(1) var<storage,read> pageHash:array<u32>;
@group(0) @binding(2) var<storage,read> metadata:array<u32>;
@group(0) @binding(3) var<storage,read> worklist:array<u32>;
@group(0) @binding(4) var<storage,read> sampleFlags:array<u32>;
@group(0) @binding(5) var<storage,read_write> phi:array<f32>;
@group(0) @binding(6) var<storage,read_write> workA:array<f32>;
@group(0) @binding(7) var<storage,read_write> control:Control;

${velocitySamplerWGSL}

fn finite3(value:vec3f)->bool{return all(value==value)&&all(abs(value)<=vec3f(3.402823e38));}
fn hashKey(key:u32)->u32{return ((key^(key>>16u))*0x9e3779b1u)&(params.hashCapacity-1u);}
fn packBrick(coord:vec3u)->u32{return coord.x+params.brickDimensions.x*(coord.y+params.brickDimensions.y*coord.z);}
fn unpackBrick(key:u32)->vec3u{
  let xy=params.brickDimensions.x*params.brickDimensions.y;
  let z=key/xy; let remainder=key-z*xy; let y=remainder/params.brickDimensions.x;
  return vec3u(remainder-y*params.brickDimensions.x,y,z);
}
fn lookupBrick(key:u32)->u32{
  let start=hashKey(key);
  for(var probe=0u;probe<32u;probe+=1u){
    if(probe>=params.maximumHashProbes){break;}
    let slot=(start+probe)&(params.hashCapacity-1u); let stored=pageHash[slot*2u];
    if(stored==key){
      let id=pageHash[slot*2u+1u]; if(id>=params.pageCapacity){return INVALID;}
      let base=id*10u;
      if(metadata[base]!=id||metadata[base+1u]!=key||metadata[base+2u]!=params.generation){return INVALID;}
      return id;
    }
    if(stored==INVALID){return INVALID;}
  }
  return INVALID;
}
fn loadFine(q:vec3i)->vec2f{
  if(any(q<vec3i(0))||any(q>=vec3i(params.sampleDimensions))){return vec2f(0.0,0.0);}
  let uq=vec3u(q); let brick=uq/params.brickResolution; let local=uq-brick*params.brickResolution;
  let id=lookupBrick(packBrick(brick)); if(id==INVALID){return vec2f(0.0,0.0);}
  let localIndex=local.x+params.brickResolution*(local.y+params.brickResolution*local.z);
  let index=id*params.samplesPerBrick+localIndex;
  if((sampleFlags[index]&VALID)==0u){return vec2f(0.0,0.0);}
  let value=phi[index]; if(value!=value||abs(value)>3.402823e38){return vec2f(0.0,0.0);}
  return vec2f(value,1.0);
}
fn sampleFineTrilinear(position:vec3f)->vec2f{
  let raw=(position-params.domainOrigin)/params.fineCellWidth-vec3f(0.5);
  let wall=vec3f(params.sampleDimensions)-vec3f(0.5);
  let wallTolerance=1e-3;
  if(any(raw<vec3f(-0.5-wallTolerance))||any(raw>wall+vec3f(wallTolerance))){return vec2f(0.0,0.0);}
  let sampleMaximum=vec3f(params.sampleDimensions)-vec3f(1.0);
  if(!CLOSED_DOMAIN_BOUNDARY&&(any(raw<vec3f(0.0))||any(raw>sampleMaximum))){return vec2f(0.0,0.0);}
  let lattice=select(raw,clamp(raw,vec3f(0.0),sampleMaximum),CLOSED_DOMAIN_BOUNDARY);
  let base=vec3i(floor(lattice)); let fraction=fract(lattice); var value=0.0;
  for(var z=0;z<2;z+=1){for(var y=0;y<2;y+=1){for(var x=0;x<2;x+=1){
    let weight=select(1.0-fraction.x,fraction.x,x==1)*select(1.0-fraction.y,fraction.y,y==1)*select(1.0-fraction.z,fraction.z,z==1);
    if(weight==0.0){continue;}
    let loaded=loadFine(base+vec3i(x,y,z)); if(loaded.y==0.0){return vec2f(0.0,0.0);}
    value+=weight*loaded.x;
  }}}
  return vec2f(value,1.0);
}

@compute @workgroup_size(64)
fn transportFinePhi(@builtin(global_invocation_id) invocation:vec3u){
  let flat=invocation.x; let total=worklist[0]*params.samplesPerBrick; if(flat>=total){return;}
  let workIndex=flat/params.samplesPerBrick; let localIndex=flat-workIndex*params.samplesPerBrick;
  let id=worklist[5u+workIndex]; if(id>=params.pageCapacity){atomicAdd(&control.departureOutsideBand,1u);return;}
  let base=id*10u; if(metadata[base+2u]!=params.generation){atomicAdd(&control.departureOutsideBand,1u);return;}
  let key=metadata[base+1u]; let brick=unpackBrick(key);
  let localZ=localIndex/(params.brickResolution*params.brickResolution);
  let remainder=localIndex-localZ*params.brickResolution*params.brickResolution;
  let localY=remainder/params.brickResolution; let localX=remainder-localY*params.brickResolution;
  let q=brick*params.brickResolution+vec3u(localX,localY,localZ);
  let destinationIndex=id*params.samplesPerBrick+localIndex;
  if((sampleFlags[destinationIndex]&VALID)==0u){return;}
  var position=params.domainOrigin+(vec3f(q)+vec3f(0.5))*params.fineCellWidth;
  let segmentDt=params.timestep/f32(params.fineFactor);
  for(var segment=0u;segment<8u;segment+=1u){
    if(segment>=params.fineFactor){break;}
    let velocity=sampleOctreeVelocity(position); if(!finite3(velocity)){atomicAdd(&control.nonfiniteVelocity,1u);return;}
    position-=segmentDt*velocity;
  }
  let transported=sampleFineTrilinear(position);
  if(transported.y==0.0){atomicAdd(&control.departureOutsideBand,1u);return;}
  workA[destinationIndex]=transported.x;
  atomicAdd(&control.processed,1u);
}

@compute @workgroup_size(64)
fn commitFinePhi(@builtin(global_invocation_id) invocation:vec3u){
  let flat=invocation.x; let total=worklist[0]*params.samplesPerBrick; if(flat>=total){return;}
  if(atomicLoad(&control.departureOutsideBand)!=0u||atomicLoad(&control.nonfiniteVelocity)!=0u){return;}
  let workIndex=flat/params.samplesPerBrick; let localIndex=flat-workIndex*params.samplesPerBrick;
  let id=worklist[5u+workIndex]; if(id>=params.pageCapacity){return;}
  let index=id*params.samplesPerBrick+localIndex; if((sampleFlags[index]&VALID)==0u){return;} phi[index]=workA[index];
  if(flat==0u){atomicStore(&control.committed,1u);}
}
`;
}
