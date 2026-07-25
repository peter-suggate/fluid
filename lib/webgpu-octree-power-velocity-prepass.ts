/** GPU-only direct Stage-B sampling for fine-level-set trajectories. */
import { OCTREE_POWER_CATALOG_ENTRY_UNIFORM } from "./octree-power-catalog";
import { fineLevelSetLinearWorkgroupWGSL, planFineLevelSetDispatch2D } from "./webgpu-fine-levelset-dispatch";
import {
  OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
  type OctreePowerFaceSource,
} from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_POWER_SAMPLE_CONTROL_BYTES = 32;
export const OCTREE_POWER_PREPASS_QUERY_BYTES = 0;
export const OCTREE_POWER_PREPASS_VERTEX_COUNT = 0;
export const OCTREE_POWER_PREPASS_ROW_DESCRIPTOR_BYTES = 16;
/** Reserved tail of the existing status buffer used by face-band diagnostics.
 * It is outside the public query range and does not add a sampler binding. */
export const OCTREE_POWER_PREPASS_STATUS_COUNTER_BYTES = 32;
export const OCTREE_POWER_PREPASS_SUMMARY_WORDS = 8;
/** Direct sampling is bounded by device buffers/dispatch, not a per-query 76-vector arena. */
export const OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY = Number.MAX_SAFE_INTEGER;

export interface OctreePowerVelocityPrepassPlan {
  readonly queryCapacity: number;
  readonly rowCapacity: number;
  readonly queryBytes: number;
  readonly vertexVelocityBytes: number;
  readonly rowDescriptorBytes: number;
  readonly scratchBytes: number;
  readonly samplerBytes: number;
  readonly allocatedBytes: number;
}

const positive = (value: number, label: string) => {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be positive`);
  return value;
};

export function planOctreePowerVelocityPrepass(queryCapacityValue: number,
  storageOffsetAlignment = 256, rowCapacityValue = 1): OctreePowerVelocityPrepassPlan {
  const queryCapacity = positive(queryCapacityValue, "Power prepass query capacity");
  const rowCapacity = positive(rowCapacityValue, "Power prepass row capacity");
  const alignment = positive(storageOffsetAlignment, "Power prepass storage alignment");
  const rowDescriptorBytes = Math.ceil(
    rowCapacity * OCTREE_POWER_PREPASS_ROW_DESCRIPTOR_BYTES / alignment,
  ) * alignment;
  const summaryBytes = Math.ceil(queryCapacity / 64) * OCTREE_POWER_PREPASS_SUMMARY_WORDS * 4;
  const samplerBytes = queryCapacity * 20 + summaryBytes + OCTREE_POWER_PREPASS_STATUS_COUNTER_BYTES
    + OCTREE_POWER_SAMPLE_CONTROL_BYTES + 48;
  return {
    queryCapacity,
    rowCapacity,
    queryBytes: 0,
    vertexVelocityBytes: 0,
    rowDescriptorBytes,
    scratchBytes: rowDescriptorBytes,
    samplerBytes,
    allocatedBytes: rowDescriptorBytes + samplerBytes,
  };
}

export interface OctreePowerVelocityPrepassSource {
  readonly results: GPUBuffer;
  readonly statuses: GPUBuffer;
  readonly control: GPUBuffer;
  readonly queryCapacity: number;
}

/** Producer-owned grouped direct authority consumed in place by fine transport.
 * Dynamic velocity records are published once per advance; immutable catalog
 * regions are installed once when this producer is constructed. */
export interface OctreePowerVelocityFusedSamplerSource {
  readonly params: GPUBuffer;
  readonly authority: GPUBuffer;
  readonly regions: Readonly<Record<"rows" | "velocities"
    | "tetraHeaders" | "tetraVertices" | "tetrahedra", { readonly offsetBytes: number; readonly sizeBytes: number }>>;
}

export interface OctreePowerVelocityChunkLimits {
  readonly maxStorageBufferBindingSize: number;
  readonly maxBufferSize: number;
  readonly maxComputeWorkgroupsPerDimension: number;
  readonly minStorageBufferOffsetAlignment: number;
}

const gcd = (a: number, b: number): number => {
  while (b !== 0) [a, b] = [b, a % b];
  return a;
};

/** Largest direct Stage-B batch admitted by result/position bindings and a tiled 2-D dispatch. */
export function planOctreePowerVelocityChunkCapacity(
  queryCapacityValue: number,
  limits: OctreePowerVelocityChunkLimits,
  targetQueryCapacity = OCTREE_POWER_PREPASS_TARGET_QUERY_CAPACITY,
): number {
  const queryCapacity = positive(queryCapacityValue, "Power prepass total query capacity");
  const target = positive(targetQueryCapacity, "Power prepass target query capacity");
  const bindingLimit = positive(limits.maxStorageBufferBindingSize, "Power prepass storage binding limit");
  const bufferLimit = positive(limits.maxBufferSize, "Power prepass buffer limit");
  const dispatchLimit = positive(limits.maxComputeWorkgroupsPerDimension, "Power prepass dispatch limit");
  const alignment = positive(limits.minStorageBufferOffsetAlignment, "Power prepass storage offset alignment");
  const maximumBytes = Math.min(bindingLimit, bufferLimit);
  const dispatchSamples = dispatchLimit * dispatchLimit * 64;
  const maximum = Math.min(queryCapacity, target, Math.floor(maximumBytes / 16), dispatchSamples);
  if (!Number.isSafeInteger(maximum) || maximum < 1) {
    throw new RangeError("Power prepass device limits cannot fit one direct velocity query");
  }
  if (maximum === queryCapacity) return queryCapacity;
  const alignmentSamples = alignment / gcd(alignment, 16);
  const aligned = Math.floor(maximum / alignmentSamples) * alignmentSamples;
  if (aligned < 1) throw new RangeError("Power prepass device limits cannot fit one aligned velocity chunk");
  return aligned;
}

export class WebGPUOctreePowerVelocityPrepass {
  readonly queryCapacity: number;
  readonly plan: OctreePowerVelocityPrepassPlan;
  readonly source: OctreePowerVelocityPrepassSource;
  private readonly fusedAuthority: GPUBuffer;
  private readonly fusedRegions: OctreePowerVelocityFusedSamplerSource["regions"];
  private readonly fusedPackParams: GPUBuffer;
  private readonly fusedPackPipeline: GPUComputePipeline;
  private readonly params: GPUBuffer;
  private readonly results: GPUBuffer;
  private readonly statuses: GPUBuffer;
  private readonly control: GPUBuffer;
  private readonly rowDescriptorPipeline: GPUComputePipeline;
  private readonly samplePipeline: GPUComputePipeline;
  private readonly summarizePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private readonly rowDescriptorGroups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private readonly fusedAuthorityGroups = new WeakMap<GPUBuffer, GPUBindGroup>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, queryCapacityValue: number,
    private readonly topology: OctreePowerTopologySource, private readonly faces: OctreePowerFaceSource) {
    this.queryCapacity = positive(queryCapacityValue, "Power prepass query capacity");
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Power prepass requires tetrahedron catalog data");
    }
    const basePlan = planOctreePowerVelocityPrepass(this.queryCapacity,
      device.limits.minStorageBufferOffsetAlignment, topology.plan.rowCapacity);
    const regionSizes = {
      rows: basePlan.rowDescriptorBytes,
      velocities: topology.plan.rowCapacity * 16,
      tetraHeaders: topology.catalogTetrahedronHeaders.size,
      tetraVertices: topology.catalogTetrahedronVertices.size,
      tetrahedra: topology.catalogTetrahedra.size,
    };
    let fusedBytes = 0;
    const fusedRegions = {} as Record<keyof typeof regionSizes, { offsetBytes: number; sizeBytes: number }>;
    for (const key of Object.keys(regionSizes) as (keyof typeof regionSizes)[]) {
      const sizeBytes = regionSizes[key];
      fusedRegions[key] = { offsetBytes: fusedBytes, sizeBytes };
      fusedBytes += sizeBytes;
    }
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (fusedBytes > maximumBinding) {
      throw new RangeError("Grouped direct Stage-B authority exceeds the adapter storage binding limit");
    }
    this.fusedRegions = Object.freeze(fusedRegions);
    this.plan = { ...basePlan,
      allocatedBytes: basePlan.allocatedBytes - basePlan.rowDescriptorBytes + fusedBytes + 32 };
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.fusedAuthority = device.createBuffer({ label: "Grouped direct Stage-B transport authority",
      size: fusedBytes, usage: storage });
    this.fusedPackParams = device.createBuffer({ label: "Grouped direct Stage-B authority parameters",
      size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.fusedPackParams, 0, new Uint32Array([
      fusedRegions.velocities.offsetBytes / 4, fusedRegions.velocities.sizeBytes / 4,
      0, 0,
    ]));
    this.results = device.createBuffer({ label: "Octree power sampled velocities",
      size: this.queryCapacity * 16, usage: storage });
    this.statuses = device.createBuffer({ label: "Octree power sample statuses",
      size: this.queryCapacity * 4
        + Math.ceil(this.queryCapacity / 64) * OCTREE_POWER_PREPASS_SUMMARY_WORDS * 4
        + OCTREE_POWER_PREPASS_STATUS_COUNTER_BYTES, usage: storage });
    this.control = device.createBuffer({ label: "Octree power sample control",
      size: OCTREE_POWER_SAMPLE_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Power direct Stage-B parameters", size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.rowDescriptorPipeline = device.createComputePipeline({ label: "Pack direct Stage-B row descriptors",
      layout: "auto", compute: { module: device.createShaderModule({ code: powerVelocityRowDescriptorWGSL }),
        entryPoint: "packPowerVelocityRows" } });
    this.fusedPackPipeline = device.createComputePipeline({
      label: "Publish grouped direct Stage-B transport authority", layout: "auto",
      compute: { module: device.createShaderModule({ code: powerVelocityFusedAuthorityWGSL }),
        entryPoint: "publishDirectFusedAuthority" },
    });
    const immutablePack = device.createCommandEncoder({ label: "Publish immutable direct Stage-B catalog authority" });
    for (const [buffer, region] of [
      [topology.catalogTetrahedronHeaders, fusedRegions.tetraHeaders],
      [topology.catalogTetrahedronVertices, fusedRegions.tetraVertices],
      [topology.catalogTetrahedra, fusedRegions.tetrahedra],
    ] as const) immutablePack.copyBufferToBuffer(buffer, 0, this.fusedAuthority, region.offsetBytes, buffer.size);
    device.queue.submit([immutablePack.finish()]);
    const shaderModule = device.createShaderModule({ label: "Direct power trajectory Stage-B sampler",
      code: directPowerVelocitySampleWGSL });
    const pipeline = (entryPoint: string, label: string) => device.createComputePipeline({ label, layout: "auto",
      compute: { module: shaderModule, entryPoint } });
    this.samplePipeline = pipeline("sampleDirectPowerVelocity", "Sample direct power trajectory velocity");
    this.summarizePipeline = pipeline("summarizeDirectPowerVelocitySamples", "Summarize direct power velocity samples");
    this.publishPipeline = pipeline("publishDirectPowerVelocitySamples", "Publish direct power velocity samples");
    this.source = { results: this.results, statuses: this.statuses,
      control: this.control, queryCapacity: this.queryCapacity };
  }

  /** Pack immutable-per-trace row metadata once before all m trajectory segments. */
  encodeRowDescriptors(broker: PassBroker, headers: GPUBuffer): void {
    if (this.destroyed) throw new Error("Power velocity prepass is destroyed");
    const pipeline = this.rowDescriptorPipeline;
    const pass = broker.compute({ label: "Pack direct Stage-B row descriptors" });
    pass.setPipeline(pipeline);
    let group = this.rowDescriptorGroups.get(headers);
    if (!group) {
      group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: headers } },
        { binding: 1, resource: { buffer: this.topology.metrics } },
        { binding: 2, resource: { buffer: this.fusedAuthority,
          offset: this.fusedRegions.rows.offsetBytes, size: this.fusedRegions.rows.sizeBytes } },
      ] });
      this.rowDescriptorGroups.set(headers, group);
    }
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(this.topology.plan.rowCapacity / 64));
  }

  /** Publish the generation-varying direct sampler fields into the producer's
   * grouped transport authority. Immutable catalog fields were published once. */
  encodeFusedAuthority(broker: PassBroker, rowVelocities: GPUBuffer): void {
    if (rowVelocities.size > this.fusedRegions.velocities.sizeBytes) {
      throw new RangeError("Direct Stage-B velocity publication exceeds its grouped authority");
    }
    const pipeline = this.fusedPackPipeline;
    const pass = broker.compute({ label: "Publish grouped direct Stage-B transport authority" });
    pass.setPipeline(pipeline);
    let group = this.fusedAuthorityGroups.get(rowVelocities);
    if (!group) {
      group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: this.fusedPackParams } },
        { binding: 1, resource: { buffer: rowVelocities } },
        { binding: 2, resource: { buffer: this.fusedAuthority } },
      ] });
      this.fusedAuthorityGroups.set(rowVelocities, group);
    }
    pass.setBindGroup(0, group);
    // One 16-byte velocity record is published per committed power row. The
    // fused sampler resolves identity through the face-band table, so the old
    // duplicate 64 KiB sorted directory is deliberately absent.
    pass.dispatchWorkgroupsIndirect(
      this.faces.liveFaceDispatch,
      OCTREE_POWER_FACE_LIVE_ROW_DISPATCH_OFFSET_BYTES,
    );
  }

  encodeFromPositions(broker: PassBroker, positions: GPUBuffer | GPUBufferBinding,
    _headers: GPUBuffer, rowVelocities: GPUBuffer, options: {
      dimensions: readonly [number, number, number]; physicalCellSize: number; maximumLeafSize: number;
      queryCount?: number; generation?: number;
    }): void {
    if (this.destroyed) throw new Error("Power velocity prepass is destroyed");
    const count = options.queryCount ?? this.queryCapacity;
    this.prepareFusedSampling(options, count);
    const bind = (buffer: GPUBuffer | GPUBufferBinding): GPUBufferBinding => "buffer" in buffer ? buffer : { buffer };
    const params = { binding: 0, resource: { buffer: this.params } };
    // These stages retain identical storage/uniform roles. WebGPU orders
    // dispatches within one compute pass, so the status reductions and final
    // control publication observe exactly the preceding stage's writes.
    this.encodeDirectoryCount(broker);
    const pass = broker.compute({ label: "Sample, summarize, and publish direct power velocity" });
    if (count > 0) {
      pass.setPipeline(this.samplePipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.samplePipeline.getBindGroupLayout(0), entries: [
        params,
        { binding: 1, resource: bind(positions) },
        { binding: 2, resource: { buffer: this.fusedAuthority,
          offset: this.fusedRegions.rows.offsetBytes, size: this.fusedRegions.rows.sizeBytes } },
        { binding: 3, resource: { buffer: this.faces.rowDirectory } },
        { binding: 4, resource: { buffer: rowVelocities } },
        { binding: 5, resource: { buffer: this.topology.catalogTetrahedronHeaders! } },
        { binding: 6, resource: { buffer: this.topology.catalogTetrahedronVertices! } },
        { binding: 7, resource: { buffer: this.topology.catalogTetrahedra! } },
        { binding: 8, resource: { buffer: this.results } },
        { binding: 9, resource: { buffer: this.statuses } },
      ] }));
      const dispatch = planFineLevelSetDispatch2D(Math.ceil(count / 64),
        this.device.limits.maxComputeWorkgroupsPerDimension);
      pass.dispatchWorkgroups(dispatch.x, dispatch.y);

      pass.setPipeline(this.summarizePipeline);
      pass.setBindGroup(0, this.device.createBindGroup({ layout: this.summarizePipeline.getBindGroupLayout(0), entries: [
        params, { binding: 9, resource: { buffer: this.statuses } },
      ] }));
      pass.dispatchWorkgroups(Math.ceil(count / 64));
    }
    pass.setPipeline(this.publishPipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.publishPipeline.getBindGroupLayout(0), entries: [
      params, { binding: 9, resource: { buffer: this.statuses } },
      { binding: 10, resource: { buffer: this.control } },
    ] }));
    pass.dispatchWorkgroups(1);
    broker.fence("direct power velocity publication complete");
  }

  /** Publish the exact direct-sampler uniform without encoding a query pass. */
  prepareFusedSampling(options: {
      dimensions: readonly [number, number, number]; physicalCellSize: number; maximumLeafSize: number;
      generation?: number;
    }, countValue: number): void {
    const count = countValue;
    if (!Number.isSafeInteger(count) || count < 0 || count > this.queryCapacity) {
      throw new RangeError("Power prepass query count exceeds capacity");
    }
    const dims = options.dimensions.map(value => positive(value, "Power prepass dimension"));
    const leaf = positive(options.maximumLeafSize, "Power prepass leaf size");
    if (!(options.physicalCellSize > 0) || !Number.isFinite(options.physicalCellSize)) {
      throw new RangeError("Power prepass cell size is invalid");
    }
    const data = new ArrayBuffer(48), words = new Uint32Array(data), floats = new Float32Array(data);
    // Word 2 is overwritten from the GPU-published face control immediately
    // before dispatch. It is the live prefix length of the sorted directory.
    words.set([count, this.queryCapacity, 0, this.topology.plan.rowCapacity,
      ...dims, leaf, 0, options.generation ?? 0]);
    floats[10] = options.physicalCellSize;
    this.device.queue.writeBuffer(this.params, 0, data);
  }

  /** Copy the exact live directory prefix published by the face transaction. */
  encodeDirectoryCount(broker: PassBroker): void {
    broker.copyBufferToBuffer(this.faces.control, 0, this.params, 8, 4);
  }

  get fusedSamplerSource(): OctreePowerVelocityFusedSamplerSource {
    return {
      params: this.params, authority: this.fusedAuthority, regions: this.fusedRegions,
    };
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.fusedAuthority.destroy();
    this.fusedPackParams.destroy();
    this.params.destroy();
    this.results.destroy();
    this.statuses.destroy();
    this.control.destroy();
  }
}

export const powerVelocityFusedAuthorityWGSL = /* wgsl */ `
struct P{velocityOffset:u32,velocityWords:u32,p0:u32,p1:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read>velocities:array<u32>;
@group(0)@binding(2)var<storage,read_write>authority:array<u32>;
@compute @workgroup_size(64)fn publishDirectFusedAuthority(@builtin(global_invocation_id)id:vec3u){
 let i=id.x;
 let word=i*4u;
 if(word+3u<p.velocityWords&&word+3u<arrayLength(&velocities)
    &&p.velocityOffset+word+3u<arrayLength(&authority)){
  authority[p.velocityOffset+word]=velocities[word];
  authority[p.velocityOffset+word+1u]=velocities[word+1u];
  authority[p.velocityOffset+word+2u]=velocities[word+2u];
  authority[p.velocityOffset+word+3u]=velocities[word+3u];
 }
}`;

export const powerVelocityRowDescriptorWGSL = /* wgsl */ `
struct H{cell:u32,a:u32,b:u32,size:u32,x:f32,y:f32,z:u32,w:u32,g:vec4f}
struct M{topology:u32,flags:u32,volume:f32,pad:u32}
struct R{cell:u32,size:u32,topology:u32,flags:u32}
@group(0)@binding(0)var<storage,read>headers:array<H>;
@group(0)@binding(1)var<storage,read>metrics:array<M>;
@group(0)@binding(2)var<storage,read_write>rows:array<R>;
@compute @workgroup_size(64)fn packPowerVelocityRows(@builtin(global_invocation_id)id:vec3u){
 let i=id.x;if(i>=arrayLength(&rows)){return;}if(i>=arrayLength(&headers)||i>=arrayLength(&metrics)){rows[i]=R(0u,0u,0xffffffffu,0u);return;}
 let h=headers[i];let m=metrics[i];rows[i]=R(h.cell,h.size,m.topology,m.flags);
}`;

/** Exact direct equivalent of the former query-builder + 76-vector materialization + indexed sampler. */
export const directPowerVelocitySampleWGSL = /* wgsl */ `
${fineLevelSetLinearWorkgroupWGSL}
struct P{count:u32,capacity:u32,directoryCount:u32,rowCapacity:u32,dimensions:vec3u,maximumLeafSize:u32,reserved:u32,generation:u32,cellSize:f32,pad:u32}
struct R{cell:u32,size:u32,topology:u32,flags:u32}
struct DirectoryEntry{cellPlusOne:u32,size:u32,row:u32,morton:u32}
struct TH{first:u32,count:u32,flags:u32}
struct V{v:vec4f}
struct Control{flags:u32,firstError:u32,queryCount:u32,interpolated:u32,uniform:u32,tetrahedron:u32,noContainingSimplex:u32,generation:u32}
@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read>positions:array<vec4f>;
@group(0)@binding(2)var<storage,read>rows:array<R>;
@group(0)@binding(3)var<storage,read>rowDirectory:array<DirectoryEntry>;
@group(0)@binding(4)var<storage,read>velocities:array<vec4f>;
@group(0)@binding(5)var<storage,read>tetraHeaders:array<TH>;
@group(0)@binding(6)var<storage,read>vertices:array<V>;
@group(0)@binding(7)var<storage,read>tetrahedra:array<u32>;
@group(0)@binding(8)var<storage,read_write>results:array<vec4f>;
@group(0)@binding(9)var<storage,read_write>statuses:array<u32>;
@group(0)@binding(10)var<storage,read_write>control:Control;
const VALID:u32=0x80000000u;const CAPACITY:u32=1u;const INVALID_QUERY:u32=2u;const NONFINITE:u32=4u;const NO_CONTAINING_SIMPLEX:u32=8u;const INACTIVE:u32=0x20000000u;const STATUS_UNIFORM:u32=0x40000000u;const UNIFORM:u32=${OCTREE_POWER_CATALOG_ENTRY_UNIFORM}u;const INVALID:u32=0xffffffffu;
fn finite(v:f32)->bool{return(bitcast<u32>(v)&0x7f800000u)!=0x7f800000u;}fn velocityValid(v:vec4f)->bool{return v.w>0.&&finite(v.x)&&finite(v.y)&&finite(v.z);}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn rowMorton(cell:u32)->u32{let q=vec3u(cell%p.dimensions.x,(cell/p.dimensions.x)%p.dimensions.y,cell/(p.dimensions.x*p.dimensions.y));return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn level(size:u32)->u32{return 31u-countLeadingZeros(size);}
fn keyLess(aLevel:u32,aMorton:u32,bLevel:u32,bMorton:u32)->bool{return aLevel<bLevel||(aLevel==bLevel&&aMorton<bMorton);}
fn find(c:u32,s:u32)->u32{let count=min(p.directoryCount,arrayLength(&rowDirectory));let wantedLevel=level(s);let wantedMorton=rowMorton(c);var low=0u;var high=count;while(low<high){let middle=low+(high-low)/2u;let candidate=rowDirectory[middle];if(keyLess(level(candidate.size),candidate.morton,wantedLevel,wantedMorton)){low=middle+1u;}else{high=middle;}}if(low<count){let candidate=rowDirectory[low];if(candidate.cellPlusOne==c+1u&&candidate.size==s){return candidate.row;}}return INVALID;}
fn owner(x:vec3f)->u32{let g=x/p.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(p.dimensions))){return 0xffffffffu;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let r=find(o.x+p.dimensions.x*(o.y+p.dimensions.y*o.z),s);if(r!=0xffffffffu){return r;}if(s>=p.maximumLeafSize){break;}s*=2u;}return 0xffffffffu;}
fn inverseTransform(x:vec3f,c:u32)->vec3f{let bits=c&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(c/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn powerTransform(value:vec3f,code:u32)->vec3f{let permutation=(code/8u)%6u;var result=value;if(permutation==1u){result=value.xzy;}else if(permutation==2u){result=value.yxz;}else if(permutation==3u){result=value.yzx;}else if(permutation==4u){result=value.zxy;}else if(permutation==5u){result=value.zyx;}let bits=code&7u;return result*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));}
fn tetraWeights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(-2.);}let wa=dot(point,cross(b,c))/determinant;let wb=dot(a,cross(point,c))/determinant;let wc=dot(a,cross(b,point))/determinant;return vec4f(1.-wa-wb-wc,wa,wb,wc);}fn contained(w:vec4f)->bool{return all(w>=vec4f(-2e-6))&&all(w<=vec4f(1.000002));}
fn fail(i:u32,flag:u32){if(i<arrayLength(&results)){results[i]=vec4f(0.);}if(i<arrayLength(&statuses)){statuses[i]=flag;}}
fn neighborVelocity(center:vec3f,size:u32,transform:u32,q:vec4f)->vec4f{let point=center+f32(size)*p.cellSize*inverseTransform(q.xyz,transform);let ns=u32(round(f32(size)*q.w));let no=round(point/p.cellSize-.5*f32(ns));if(any(no<vec3f(0))){return vec4f(0.);}let n=vec3u(no);let row=find(n.x+p.dimensions.x*(n.y+p.dimensions.y*n.z),ns);if(row==0xffffffffu||row>=arrayLength(&velocities)){return vec4f(0.);}return velocities[row];}
@compute @workgroup_size(64)fn sampleDirectPowerVelocity(@builtin(workgroup_id)w:vec3u,@builtin(local_invocation_index)lid:u32,@builtin(num_workgroups)n:vec3u){let i=fineLinearWorkgroup(w,n)*64u+lid;if(i>=p.count||i>=p.capacity){return;}if(p.count>p.capacity||i>=arrayLength(&positions)||i>=arrayLength(&results)||i>=arrayLength(&statuses)){return;}let x=positions[i];if(x.w<=0.){results[i]=vec4f(0.,0.,0.,1.);statuses[i]=VALID|INACTIVE;return;}let row=owner(x.xyz);if(row==0xffffffffu){fail(i,NO_CONTAINING_SIMPLEX);return;}if(row>=arrayLength(&rows)||row>=arrayLength(&velocities)){fail(i,NONFINITE);return;}let r=rows[row];if(r.size==0u||r.topology>=arrayLength(&tetraHeaders)){fail(i,NONFINITE);return;}let th=tetraHeaders[r.topology];let o=vec3u(r.cell%p.dimensions.x,(r.cell/p.dimensions.x)%p.dimensions.y,r.cell/(p.dimensions.x*p.dimensions.y));let center=(vec3f(o)+.5*f32(r.size))*p.cellSize;let marker=th.flags&0x10000000u;
 if((th.flags&UNIFORM)!=0u){let g=x.xyz/p.cellSize;let lowi=select(vec3i(o)-vec3i(i32(r.size)),vec3i(o),g>=vec3f(o)+.5*f32(r.size));let low=max(lowi,vec3i(0));let t=clamp((g-(vec3f(low)+.5*f32(r.size)))/f32(r.size),vec3f(0),vec3f(1));let anchor=velocities[row];var value=vec3f(0.);for(var corner=0u;corner<8u;corner+=1u){let co=vec3u(low)+vec3u(corner&1u,(corner>>1u)&1u,(corner>>2u)&1u)*r.size;var v=anchor;if(all(co<p.dimensions)){let nr=find(co.x+p.dimensions.x*(co.y+p.dimensions.y*co.z),r.size);if(nr!=0xffffffffu&&nr<arrayLength(&velocities)){v=velocities[nr];}}if(!velocityValid(v)){fail(i,NONFINITE);return;}let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);value+=weight*v.xyz;}results[i]=vec4f(value,1.);statuses[i]=VALID|STATUS_UNIFORM|marker;return;}
 let local=powerTransform((x.xyz-center)/(f32(r.size)*p.cellSize),r.flags&63u);let anchor=velocities[row];if(!velocityValid(anchor)){fail(i,NONFINITE);return;}if(th.first>arrayLength(&tetrahedra)||th.count>arrayLength(&tetrahedra)-th.first){fail(i,INVALID_QUERY);return;}for(var ti=0u;ti<th.count;ti+=1u){let packed=tetrahedra[th.first+ti];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(min(75u,arrayLength(&vertices))))){fail(i,INVALID_QUERY);return;}let a=vertices[selectors.x].v;let b=vertices[selectors.y].v;let c=vertices[selectors.z].v;let weights=tetraWeights(local,a.xyz,b.xyz,c.xyz);if(!contained(weights)){continue;}let va=neighborVelocity(center,r.size,r.flags&63u,a);let vb=neighborVelocity(center,r.size,r.flags&63u,b);let vc=neighborVelocity(center,r.size,r.flags&63u,c);if(!velocityValid(va)||!velocityValid(vb)||!velocityValid(vc)){fail(i,NONFINITE);return;}results[i]=vec4f(weights.x*anchor.xyz+weights.y*va.xyz+weights.z*vb.xyz+weights.w*vc.xyz,1.);statuses[i]=VALID|ti|marker;return;}fail(i,NO_CONTAINING_SIMPLEX);statuses[i]=NO_CONTAINING_SIMPLEX|((r.topology&0xffffu)<<8u);}
var<workgroup> rf:array<u32,64>;var<workgroup> re:array<u32,64>;var<workgroup> ri:array<u32,64>;var<workgroup> ru:array<u32,64>;var<workgroup> rt:array<u32,64>;var<workgroup> rn:array<u32,64>;
@compute @workgroup_size(64)fn summarizeDirectPowerVelocitySamples(@builtin(workgroup_id)wg:vec3u,@builtin(local_invocation_index)lid:u32){let i=wg.x*64u+lid;var status=0u;if(i<p.count&&i<arrayLength(&statuses)){status=statuses[i];}let errors=status&15u;rf[lid]=errors;re[lid]=select(INVALID,i,errors!=0u);ri[lid]=select(0u,1u,(status&VALID)!=0u);ru[lid]=select(0u,1u,(status&STATUS_UNIFORM)!=0u);rt[lid]=select(0u,1u,(status&VALID)!=0u&&(status&(STATUS_UNIFORM|INACTIVE))==0u);rn[lid]=select(0u,1u,(status&NO_CONTAINING_SIMPLEX)!=0u);workgroupBarrier();var stride=32u;loop{if(stride==0u){break;}if(lid<stride){rf[lid]|=rf[lid+stride];re[lid]=min(re[lid],re[lid+stride]);ri[lid]+=ri[lid+stride];ru[lid]+=ru[lid+stride];rt[lid]+=rt[lid+stride];rn[lid]+=rn[lid+stride];}workgroupBarrier();stride/=2u;}if(lid==0u){let base=p.capacity+wg.x*8u;statuses[base]=rf[0];statuses[base+1u]=re[0];statuses[base+2u]=ri[0];statuses[base+3u]=ru[0];statuses[base+4u]=rt[0];statuses[base+5u]=rn[0];statuses[base+6u]=0u;statuses[base+7u]=0u;}}
@compute @workgroup_size(64)fn publishDirectPowerVelocitySamples(@builtin(local_invocation_index)lid:u32){let groups=(p.count+63u)/64u;var flags=0u;var first=INVALID;var interpolated=0u;var uniform=0u;var tetrahedron=0u;var noContaining=0u;for(var group=lid;group<groups;group+=64u){let base=p.capacity+group*8u;flags|=statuses[base];first=min(first,statuses[base+1u]);interpolated+=statuses[base+2u];uniform+=statuses[base+3u];tetrahedron+=statuses[base+4u];noContaining+=statuses[base+5u];}rf[lid]=flags;re[lid]=first;ri[lid]=interpolated;ru[lid]=uniform;rt[lid]=tetrahedron;rn[lid]=noContaining;workgroupBarrier();var stride=32u;loop{if(stride==0u){break;}if(lid<stride){rf[lid]|=rf[lid+stride];re[lid]=min(re[lid],re[lid+stride]);ri[lid]+=ri[lid+stride];ru[lid]+=ru[lid+stride];rt[lid]+=rt[lid+stride];rn[lid]+=rn[lid+stride];}workgroupBarrier();stride/=2u;}if(lid==0u){let capacityFailure=p.count>p.capacity||p.capacity+groups*8u>arrayLength(&statuses);control.flags=select(rf[0],CAPACITY,capacityFailure);control.firstError=select(re[0],min(p.count,p.capacity),capacityFailure);control.queryCount=p.count;control.interpolated=ri[0];control.uniform=ru[0];control.tetrahedron=rt[0];control.noContainingSimplex=rn[0];control.generation=p.generation;if(!capacityFailure&&rf[0]==0u&&ri[0]==p.count){control.flags=VALID;}}}
`;

/** Production direct sampler source, exposed to constrained-device portability tests. */
export function makePowerVelocityPrepassBuilderWGSL(): string { return directPowerVelocitySampleWGSL; }
