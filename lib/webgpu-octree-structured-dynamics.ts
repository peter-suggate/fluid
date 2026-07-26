/** Class-specialized timestep kernels over the packed six-family authority. */

import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import {
  OCTREE_STRUCTURED_GPU_ERROR,
  type DirectStructuredVelocitySource,
} from "./webgpu-octree-structured-velocity-gpu";
import {
  OCTREE_AIR_SUPPORT_LAYOUT_VERSION,
  OCTREE_AIR_SUPPORT_VALID,
  type OctreeAirVelocitySupportLayout,
} from "./webgpu-octree-air-velocity-support";
import type { PassBroker } from "./webgpu-pass-broker";

const ROW_CLASSES = [0, 1, 2, 3] as const;
const FAMILY_CLASSES = [5, 6, 7, 8] as const;
/**
 * Bounded GPU relaxation budget for the Section 5 ordinary-face march.
 *
 * Aanjaneya et al. prescribe copying from the face geometrically closest to
 * the free surface, not widening the extrapolation domain until a particular
 * scene happens to pass. The final encoded wave remains a fixed-point proof:
 * support that cannot converge within this production budget is rejected by
 * the publication rather than admitted through a wider numerical stencil.
 */
export const STRUCTURED_VELOCITY_EXTENSION_LAYERS = 6;
export const STRUCTURED_PROJECTION_ENERGY_WORDS = 8;

export interface StructuredProjectionEnergySample {
  readonly epoch: number;
  readonly activeBank: 0 | 1;
  readonly familySampleCount: number;
  readonly preProjectionKineticEnergyProxy: number;
  readonly postProjectionKineticEnergyProxy: number;
  readonly projectionEnergyRatio: number;
}

export interface StructuredProjectionEnergyDecode {
  readonly sample: StructuredProjectionEnergySample | null;
  readonly blocker: string | null;
}

/** Decode the one-generation paired reduction. An initialized/partial buffer
 * never becomes a zero-energy observation. */
export function decodeStructuredProjectionEnergy(
  words: ArrayLike<number>,
): StructuredProjectionEnergyDecode {
  if (words.length < STRUCTURED_PROJECTION_ENERGY_WORDS) {
    return Object.freeze({ sample: null, blocker: "structured projection-energy report is truncated" });
  }
  const flags = Number(words[0]) >>> 0;
  const epoch = Number(words[1]) >>> 0;
  const activeBank = Number(words[2]) >>> 0;
  const preCount = Number(words[3]) >>> 0;
  const postCount = Number(words[4]) >>> 0;
  const pairedEpoch = Number(words[7]) >>> 0;
  if (flags !== 0) return Object.freeze({ sample: null,
    blocker: `structured projection-energy reduction failed with flags ${flags}` });
  if (epoch === 0 || pairedEpoch !== epoch || activeBank > 1) return Object.freeze({ sample: null,
    blocker: "structured projection-energy pair is unpublished or generation-incoherent" });
  if (preCount === 0 || postCount !== preCount) return Object.freeze({ sample: null,
    blocker: "structured projection-energy pair has incomplete family coverage" });
  const bits = new Uint32Array([Number(words[5]) >>> 0, Number(words[6]) >>> 0]);
  const energy = new Float32Array(bits.buffer);
  const pre = energy[0]!;
  const post = energy[1]!;
  if (!Number.isFinite(pre) || pre < 0 || !Number.isFinite(post) || post < 0
    || (pre === 0 && post !== 0)) return Object.freeze({ sample: null,
    blocker: "structured projection-energy pair contains invalid energy" });
  return Object.freeze({ sample: Object.freeze({
    epoch, activeBank: activeBank as 0 | 1, familySampleCount: preCount,
    preProjectionKineticEnergyProxy: pre,
    postProjectionKineticEnergyProxy: post,
    projectionEnergyRatio: pre === 0 ? 1 : post / pre,
  }), blocker: null });
}

export interface StructuredDynamicsResources {
  readonly structured: DirectStructuredVelocitySource;
  readonly topology: OctreePowerTopologySource;
  readonly pressure: GPUBuffer;
  readonly divergenceRhs: GPUBuffer;
  /** Accepted dynamic-boundary classification, one u32 liquid bit per banked row. */
  readonly liquidMask: GPUBuffer;
  /** Required normal velocity of the solid boundary, one f32 per family handle
   * in each of the two structured banks. The boundary producer writes the bank
   * named by the accepted structured publication, so every read here must apply
   * the same `bank()*slotCapacity` base. */
  readonly solidNormalVelocities: GPUBuffer;
  /** Current-step boundary-class worksets and their fail-closed publication. */
  readonly boundaryWorksets: GPUBuffer;
  readonly boundaryControl: GPUBuffer;
  /** Current compact-row adjacency for every byte-packed Delaunay selector. */
  readonly selectorRows: GPUBuffer;
  readonly selectorStride: number;
  readonly selectorOffsetWords: number;
  readonly airSupportLayout: OctreeAirVelocitySupportLayout;
  readonly dimensions: readonly [number, number, number];
  readonly physicalCellSize: number;
  readonly closedBoundaryMask: number;
}

/**
 * No stage consumes a general face or incidence record. The only recurring
 * writes are destination-owned family values, one RHS per row, and one
 * projected row vector. Section 5 face extrapolation is published by the
 * dedicated air-support producer after projection.
 */
export class WebGPUStructuredVelocityDynamics {
  readonly encodedAdvectionDispatchCount = 5;
  readonly encodedForceDivergenceDispatchCount = 10;
  readonly encodedProjectionDispatchCount = 9;
  readonly allocatedBytes: number;
  /** Per-slot vec4f: transported momentum xyz and kinetic dissipation w. */
  readonly transportMetrics: GPUBuffer;
  /** Failure-only topology readback metadata for the shared selector-row buffer. */
  readonly selectorStride: number;
  readonly selectorOffsetWords: number;
  readonly dimensions: readonly [number, number, number];
  /** Eight-word coherent pre/post projection kinetic-energy report. */
  readonly projectionEnergyStats: GPUBuffer;
  /** Immutable packed volume/slot/tetra catalog shared by direct consumers. */
  readonly catalog: GPUBuffer;
  readonly catalogOffsetsWords: readonly [number, number, number, number, number];
  /** Nine accepted class dispatch records, one vec3u per workset class. */
  readonly dispatch: GPUBuffer;
  private readonly params: readonly [GPUBuffer, GPUBuffer, GPUBuffer];
  private readonly prepare: GPUComputePipeline;
  private readonly advection: readonly GPUComputePipeline[];
  private readonly force: readonly GPUComputePipeline[];
  private readonly divergence: readonly GPUComputePipeline[];
  private readonly projection: readonly GPUComputePipeline[];
  private readonly reconstruct: readonly GPUComputePipeline[];
  private readonly summarizePreProjectionEnergy: GPUComputePipeline;
  private readonly summarizePostProjectionEnergy: GPUComputePipeline;
  private readonly groups = new WeakMap<GPUComputePipeline,
    WeakMap<GPUBuffer, WeakMap<GPUBuffer, GPUBindGroup>>>();
  private destroyed = false;

  constructor(private readonly device: GPUDevice, private readonly resources: StructuredDynamicsResources) {
    const { structured, topology } = resources;
    if (!(resources.physicalCellSize > 0) || !Number.isFinite(resources.physicalCellSize)
      || resources.dimensions.some((value) => !Number.isSafeInteger(value) || value < 1)
      || resources.pressure.size < structured.plan.rowCapacity * 4
      || resources.divergenceRhs.size < structured.plan.rowCapacity * 4
      || resources.liquidMask.size < structured.plan.rowCapacity * 2 * 4
      || resources.solidNormalVelocities.size < structured.plan.slotCapacity * 2 * 4
      || !Number.isSafeInteger(resources.selectorStride) || resources.selectorStride < 1
      || !Number.isSafeInteger(resources.selectorOffsetWords) || resources.selectorOffsetWords < 0
      || resources.selectorRows.size < (resources.selectorOffsetWords
        + structured.plan.rowCapacity * resources.selectorStride) * 4
      || resources.airSupportLayout.rowCapacity !== structured.plan.rowCapacity
      || resources.airSupportLayout.slotCapacity !== structured.plan.slotCapacity
      || resources.airSupportLayout.selectorTagOffsetWords !== resources.selectorOffsetWords
      || resources.airSupportLayout.selectorStride !== resources.selectorStride
      || resources.selectorRows.size < resources.airSupportLayout.totalBytes
      || !topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra
      || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Structured dynamics resources are invalid or undersized");
    }
    this.selectorStride = resources.selectorStride;
    this.selectorOffsetWords = resources.selectorOffsetWords;
    this.dimensions = resources.dimensions;
    const pieces = [topology.catalogVolumes, topology.catalogFaces,
      topology.catalogTetrahedronHeaders, topology.catalogTetrahedronVertices,
      topology.catalogTetrahedra] as const;
    const offsets: number[] = []; let catalogBytes = 0;
    for (const piece of pieces) { offsets.push(catalogBytes); catalogBytes += piece.size; }
    this.catalogOffsetsWords = offsets.map((offset) => offset / 4) as unknown as
      readonly [number, number, number, number, number];
    const maximumBinding = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    if (catalogBytes > maximumBinding) throw new RangeError("Structured dynamics catalog exceeds binding limits");
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.catalog = device.createBuffer({ label: "Structured dynamics immutable catalog", size: catalogBytes, usage: storage });
    const copy = device.createCommandEncoder({ label: "Install structured dynamics catalog" });
    pieces.forEach((piece, index) => copy.copyBufferToBuffer(piece, 0, this.catalog, offsets[index]!, piece.size));
    device.queue.submit([copy.finish()]);
    this.dispatch = device.createBuffer({ label: "Structured dynamics accepted indirect arguments",
      size: 9 * 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    this.transportMetrics = resources.selectorRows;
    this.projectionEnergyStats = device.createBuffer({
      label: "Structured paired projection kinetic energy",
      size: STRUCTURED_PROJECTION_ENERGY_WORDS * 4,
      usage: storage,
    });
    this.params = [0, 1, 2].map((stage) => device.createBuffer({
      label: `Structured dynamics parameters ${stage}`, size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })) as unknown as readonly [GPUBuffer, GPUBuffer, GPUBuffer];
    const words = new Uint32Array(64), floats = new Float32Array(words.buffer);
    words.set([structured.plan.rowCapacity, structured.plan.slotCapacity,
      structured.plan.maximumCaseSlots, structured.plan.authorityWords,
      structured.plan.worksetStrideWords, structured.worksetBankStrideWords,
      ...resources.dimensions, resources.closedBoundaryMask >>> 0,
      offsets[0]! / 4, offsets[1]! / 4, offsets[2]! / 4, offsets[3]! / 4,
      offsets[4]! / 4, topology.catalogTetrahedronVertexCount ?? 0,
      topology.rowTemplateHeaderOffsetBytes / 4,
      topology.reconstructionDataOffsetBytes / 4,
    ], 0);
    words.set(Object.values(structured.plan.offsets), 18);
    floats[40] = resources.physicalCellSize;
    words[48] = resources.selectorOffsetWords;
    words[49] = resources.selectorStride;
    words[50] = resources.airSupportLayout.regularTagOffsetWords;
    words[51] = resources.airSupportLayout.controlOffsetWords;
    words[52] = resources.airSupportLayout.supportVectorOffsetWords;
    words[53] = resources.airSupportLayout.supportCapacity;
    for (const buffer of this.params) device.queue.writeBuffer(buffer, 0, words.buffer);
    const shaderModule = device.createShaderModule({ label: "Structured velocity dynamics", code: structuredVelocityDynamicsWGSL });
    const make = (name: string) => device.createComputePipeline({ label: name,
      layout: "auto", compute: { module: shaderModule, entryPoint: name } });
    this.prepare = make("prepareStructuredDynamics");
    this.advection = FAMILY_CLASSES.map((value) => make(`advectStructuredClass${value}`));
    this.force = FAMILY_CLASSES.map((value) => make(`forceStructuredClass${value}`));
    this.divergence = ROW_CLASSES.map((value) => make(`divergenceStructuredClass${value}`));
    this.projection = FAMILY_CLASSES.map((value) => make(`projectStructuredClass${value}`));
    this.reconstruct = ROW_CLASSES.map((value) => make(`reconstructStructuredClass${value}`));
    this.summarizePreProjectionEnergy = make("summarizeStructuredPreProjectionEnergy");
    this.summarizePostProjectionEnergy = make("summarizeStructuredPostProjectionEnergy");
    this.allocatedBytes = catalogBytes + this.dispatch.size
      + this.projectionEnergyStats.size
      + 3 * 256;
  }

  private update(stage: 0 | 1 | 2, dt: number, density: number,
    gravity: readonly [number, number, number]): GPUBuffer {
    if (!(dt >= 0) || !Number.isFinite(dt) || !(density > 0) || !Number.isFinite(density)
      || gravity.some((value) => !Number.isFinite(value))) throw new RangeError("Invalid structured dynamics parameters");
    const bytes = new ArrayBuffer(28), floats = new Float32Array(bytes);
    floats[0] = dt; floats[1] = density; floats.set(gravity, 3);
    this.device.queue.writeBuffer(this.params[stage], 164, bytes);
    return this.params[stage];
  }

  private entries(params: GPUBuffer, pressure: GPUBuffer): Readonly<Record<number, GPUBuffer>> {
    const { structured, topology, divergenceRhs, liquidMask } = this.resources;
    return { 0: params, 1: structured.control, 2: structured.authority,
      3: structured.rowGeometry, 4: structured.rowVelocities, 5: topology.metrics,
      6: this.catalog, 11: this.resources.boundaryWorksets,
      12: this.dispatch, 13: pressure, 14: divergenceRhs, 16: liquidMask,
      17: this.resources.boundaryControl, 18: this.transportMetrics,
      22: this.resources.solidNormalVelocities, 23: this.projectionEnergyStats };
  }

  private group(pipeline: GPUComputePipeline, entries: readonly number[], params: GPUBuffer,
    pressure = this.resources.pressure): GPUBindGroup {
    let byParams = this.groups.get(pipeline);
    if (!byParams) { byParams = new WeakMap(); this.groups.set(pipeline, byParams); }
    let byPressure = byParams.get(params);
    if (!byPressure) { byPressure = new WeakMap(); byParams.set(params, byPressure); }
    const cached = byPressure.get(pressure); if (cached) return cached;
    const resources = this.entries(params, pressure);
    const group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: entries.map((binding) => ({
      binding, resource: { buffer: resources[binding]! },
    })) });
    byPressure.set(pressure, group); return group;
  }

  private encodePrepare(broker: PassBroker, params: GPUBuffer): void {
    const pass = broker.compute({ label: "Prepare accepted structured dynamics worksets" });
    pass.setPipeline(this.prepare); pass.setBindGroup(0, this.group(this.prepare, [0, 1, 11, 12, 17], params));
    pass.dispatchWorkgroups(1); broker.fence("structured indirect arguments published");
  }

  private encodeProjectionEnergy(broker: PassBroker, params: GPUBuffer,
    phase: "pre" | "post"): void {
    const pipeline = phase === "pre"
      ? this.summarizePreProjectionEnergy : this.summarizePostProjectionEnergy;
    const pass = broker.compute({ label: phase === "pre"
      ? "Structured dynamics report pre-projection kinetic energy"
      : "Structured dynamics report post-projection kinetic energy" });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.group(pipeline, [0, 1, 2, 11, 17, 23], params));
    pass.dispatchWorkgroups(1);
  }

  private encodeClasses(broker: PassBroker, pipelines: readonly GPUComputePipeline[], classes: readonly number[],
    bindings: readonly number[], params: GPUBuffer, label: string, pressure?: GPUBuffer): void {
    pipelines.forEach((pipeline, index) => {
      const pass = broker.compute({ label: `${label} ${classes[index]}` }); pass.setPipeline(pipeline);
      pass.setBindGroup(0, this.group(pipeline, bindings, params, pressure));
      pass.dispatchWorkgroupsIndirect(this.dispatch, classes[index]! * 12);
    });
  }

  encodeAdvection(broker: PassBroker, dt: number): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    const params = this.update(0, dt, 1, [0, 0, 0]); this.encodePrepare(broker, params);
    this.encodeClasses(broker, this.advection, FAMILY_CLASSES, [0, 1, 2, 3, 4, 5, 6, 11, 17, 18, 22], params,
      "Advect structured family class");
  }

  // The divergence RHS is dimensional: rhs = rho*flux/dt. `encodeProjection`
  // undoes it as v -= dt*grad(p)/rho, so both halves must be given the SAME
  // density. Passing 1 here while the projection received the scene density
  // scaled every pressure gradient by 1/rho, so the projection removed ~0.1%
  // of the divergence: the column free-fell under gravity but no horizontal
  // momentum was ever generated and the dam front did not advance.
  encodeForcesAndDivergence(broker: PassBroker, dt: number, density: number,
    gravity: readonly [number, number, number]): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    const params = this.update(1, dt, density, gravity); this.encodePrepare(broker, params);
    this.encodeClasses(broker, this.force, FAMILY_CLASSES, [0, 1, 2, 11, 17, 22], params,
      "Force and constrain structured family class");
    this.encodeProjectionEnergy(broker, params, "pre");
    this.encodeClasses(broker, this.divergence, ROW_CLASSES, [0, 1, 2, 5, 6, 11, 14, 16, 17, 22], params,
      "Fuse structured divergence RHS class");
  }

  encodeProjection(broker: PassBroker, dt: number, density: number,
    pressure = this.resources.pressure): void {
    if (this.destroyed) throw new Error("Structured dynamics is destroyed");
    const params = this.update(2, dt, density, [0, 0, 0]); this.encodePrepare(broker, params);
    this.encodeClasses(broker, this.projection, FAMILY_CLASSES,
      [0, 1, 2, 11, 13, 16, 17, 22], params,
      "Project structured family class", pressure);
    this.encodeProjectionEnergy(broker, params, "post");
    this.encodeClasses(broker, this.reconstruct, ROW_CLASSES,
      [0, 1, 2, 4, 5, 6, 11, 17], params,
      "Reconstruct projected structured rows");
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.catalog.destroy(); this.dispatch.destroy();
    this.projectionEnergyStats.destroy();
    this.params.forEach((buffer) => buffer.destroy());
  }
}

export const structuredVelocityDynamicsWGSL = /* wgsl */ `
struct P{rowCapacity:u32,slotCapacity:u32,maxSlots:u32,authorityWords:u32,worksetStride:u32,worksetBankStride:u32,dimensionX:u32,dimensionY:u32,dimensionZ:u32,closedMask:u32,denseOffset:u32,slotGeometryOffset:u32,tetraHeaderOffset:u32,tetraVertexOffset:u32,tetraOffset:u32,tetraVertexCount:u32,templateHeaderOffset:u32,reconstructionOffset:u32,valuesOffset:u32,ownerOffset:u32,neighborOffset:u32,metadataOffset:u32,areaOffset:u32,inverseOffset:u32,fractionOffset:u32,pressureScaleOffset:u32,normalOffset:u32,centroidOffset:u32,rowNeighborOffset:u32,rowReciprocalOffset:u32,rowOwnerMetadataOffset:u32,rowHandleOffset:u32,rowSignOffset:u32,rowCatalogOffset:u32,rowAxisOffset:u32,rowFamilyPrefixOffset:u32,rowFamilyHandleOffset:u32,rowFamilySlotOffset:u32,padA:u32,padB:u32,physical:vec4f,gravity:vec4f,selectorOffsetWords:u32,selectorStride:u32,regularTagOffsetWords:u32,supportControlOffsetWords:u32,supportVectorOffsetWords:u32,supportCapacity:u32}
struct Metric{caseId:u32,transformAndFlags:u32,volume:f32,error:u32}
struct SlotGeometry{neighborOffsetSize:vec4f,areaCentroid:vec4f,normalInverseDistance:vec4f}

@group(0)@binding(0)var<uniform>p:P;
@group(0)@binding(1)var<storage,read_write>accepted:array<atomic<u32>>;
@group(0)@binding(2)var<storage,read_write>a:array<u32>;
@group(0)@binding(3)var<storage,read>rowGeometry:array<vec4u>;
@group(0)@binding(4)var<storage,read_write>rowVelocity:array<vec4f>;
@group(0)@binding(5)var<storage,read>metrics:array<Metric>;
@group(0)@binding(6)var<storage,read>catalog:array<u32>;
@group(0)@binding(11)var<storage,read>worksets:array<u32>;
@group(0)@binding(12)var<storage,read_write>indirect:array<u32>;
@group(0)@binding(13)var<storage,read>pressure:array<f32>;
@group(0)@binding(14)var<storage,read_write>rhs:array<f32>;
@group(0)@binding(16)var<storage,read>liquid:array<u32>;
@group(0)@binding(17)var<storage,read>boundaryControl:array<u32>;
@group(0)@binding(18)var<storage,read_write>transportMetrics:array<vec4f>;
@group(0)@binding(22)var<storage,read>solidNormalVelocities:array<f32>;
@group(0)@binding(23)var<storage,read_write>projectionEnergyStats:array<u32>;

const INVALID:u32=0xffffffffu;
const ERROR_SAMPLE:u32=${OCTREE_STRUCTURED_GPU_ERROR.carry}u;
const SUPPORT_TAG:u32=0x80000000u;
const SUPPORT_VALID:u32=${OCTREE_AIR_SUPPORT_VALID}u;
const SUPPORT_LAYOUT_VERSION:u32=${OCTREE_AIR_SUPPORT_LAYOUT_VERSION}u;

fn acc(index:u32)->u32{return atomicLoad(&accepted[index]);}
fn rejectSample(stage:u32,index:u32){
  atomicOr(&accepted[0],ERROR_SAMPLE);
  atomicMin(&accepted[1],(stage<<24u)|(index&0x00ffffffu));
}
fn rejectVector(stage:u32,index:u32,detail:vec4f,cls:u32){
  atomicOr(&accepted[0],ERROR_SAMPLE);
  let packed=(stage<<24u)|(index&0x00ffffffu);
  let previous=atomicMin(&accepted[1],packed);
  if(packed<previous&&arrayLength(&accepted)>=11u){
    atomicStore(&accepted[6],bitcast<u32>(detail.x));
    atomicStore(&accepted[7],bitcast<u32>(detail.y));
    atomicStore(&accepted[8],bitcast<u32>(detail.z));
    atomicStore(&accepted[9],bitcast<u32>(detail.w));
    atomicStore(&accepted[10],cls);
  }
}
fn bank()->u32{return acc(4u)&1u;}
fn abase()->u32{return bank()*p.authorityWords;}
fn rbase()->u32{return bank()*p.rowCapacity;}
fn lbase()->u32{return bank()*p.rowCapacity;}
fn sbase()->u32{return bank()*p.slotCapacity;}
fn wbase(cls:u32)->u32{return bank()*p.worksetBankStride+cls*p.worksetStride;}
fn finite(v:f32)->bool{return v==v&&abs(v)<=3.402823e38;}
fn finite3(v:vec3f)->bool{return all(v==v)&&all(abs(v)<=vec3f(3.402823e38));}
fn invalidVector()->vec4f{return vec4f(0.,0.,0.,-1.);}
fn vectorValid(v:vec4f)->bool{return v.w>0.&&finite3(v.xyz);}
fn bitsf(at:u32)->f32{return bitcast<f32>(catalog[at]);}
fn dimensions()->vec3u{return vec3u(p.dimensionX,p.dimensionY,p.dimensionZ);}
fn signs(code:u32)->vec3f{let b=code&7u;return vec3f(select(1.,-1.,(b&1u)!=0u),select(1.,-1.,(b&2u)!=0u),select(1.,-1.,(b&4u)!=0u));}
fn inverseTransform(v:vec3f,code:u32)->vec3f{let q=v*signs(code);let k=(code/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn powerTransform(v:vec3f,code:u32)->vec3f{let k=(code/8u)%6u;var q=v;if(k==1u){q=v.xzy;}else if(k==2u){q=v.yxz;}else if(k==3u){q=v.yzx;}else if(k==4u){q=v.zxy;}else if(k==5u){q=v.zyx;}return q*signs(code);}
fn caseHeader(caseId:u32)->vec2u{let at=p.denseOffset+p.templateHeaderOffset+4u*caseId;return vec2u(catalog[at],catalog[at+1u]);}
fn geom(global:u32)->SlotGeometry{let at=p.slotGeometryOffset+12u*global;return SlotGeometry(vec4f(bitsf(at),bitsf(at+1u),bitsf(at+2u),bitsf(at+3u)),vec4f(bitsf(at+4u),bitsf(at+5u),bitsf(at+6u),bitsf(at+7u)),vec4f(bitsf(at+8u),bitsf(at+9u),bitsf(at+10u),bitsf(at+11u)));}
fn boundaryValid()->bool{return arrayLength(&boundaryControl)>=7u&&boundaryControl[0]==0u&&boundaryControl[2]==acc(2u)&&boundaryControl[4]==acc(3u)&&boundaryControl[5]==bank()&&boundaryControl[6]==acc(3u);}
fn workItem(cls:u32,index:u32)->u32{let base=wbase(cls);if(!boundaryValid()||worksets[base]!=acc(3u)||index>=worksets[base+1u]){return INVALID;}return worksets[base+7u+index];}

@compute @workgroup_size(1)
fn prepareStructuredDynamics(){
  for(var cls=0u;cls<9u;cls+=1u){
    let base=wbase(cls);
    let valid=acc(0u)==0u&&acc(3u)!=0u&&boundaryValid()&&worksets[base]==acc(3u);
    let out=3u*cls;
    indirect[out]=select(0u,worksets[base+4u],valid);
    indirect[out+1u]=1u;
    indirect[out+2u]=1u;
  }
}

fn value(handle:u32)->f32{return bitcast<f32>(a[abase()+p.valuesOffset+handle]);}
fn setValue(handle:u32,v:f32){a[abase()+p.valuesOffset+handle]=bitcast<u32>(v);}
fn owner(handle:u32)->u32{return a[abase()+p.ownerOffset+handle];}
fn neighbor(handle:u32)->u32{return a[abase()+p.neighborOffset+handle];}
fn normal(handle:u32)->vec3f{let at=abase()+p.normalOffset+4u*handle;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}
fn centroid(handle:u32)->vec3f{let at=abase()+p.centroidOffset+4u*handle;return vec3f(bitcast<f32>(a[at]),bitcast<f32>(a[at+1u]),bitcast<f32>(a[at+2u]));}
fn rowSlotCount(row:u32)->u32{return caseHeader(metrics[row].caseId).y;}
fn liquidAt(row:u32)->u32{return liquid[lbase()+row];}
fn solidVelocityAt(handle:u32)->f32{return solidNormalVelocities[sbase()+handle];}
fn velocitySample(row:u32)->vec4f{
  if(row>=acc(2u)){return invalidVector();}
  // Section 5 advection consumes the previous projected and extended field.
  // The final destination-gather layer publishes air rows back into this
  // canonical full-vector field. Keeping the choice out of this hot sampler
  // preserves the device's ten-storage-buffer advection contract.
  let sample=rowVelocity[rbase()+row];
  if(sample.w<=0.||!finite3(sample.xyz)){return invalidVector();}
  return vec4f(sample.xyz,1.);
}
fn supportWord(word:u32)->u32{
  if(word>=4u*arrayLength(&transportMetrics)){return INVALID;}
  return bitcast<vec4u>(transportMetrics[word/4u])[word&3u];
}
fn supportCount()->u32{return supportWord(p.supportControlOffsetWords+6u);}
fn supportPublicationValid()->bool{
  let base=p.supportControlOffsetWords;
  if(base+16u>4u*arrayLength(&transportMetrics)||!boundaryValid()){return false;}
  let count=supportWord(base+6u);let capacity=supportWord(base+7u);
  let faces=supportWord(base+10u);let seeds=supportWord(base+11u);
  return supportWord(base)==0u&&supportWord(base+1u)==INVALID
    &&supportWord(base+2u)==acc(3u)&&supportWord(base+3u)==bank()
    &&supportWord(base+4u)==boundaryControl[4u]&&supportWord(base+5u)==acc(2u)
    &&capacity==p.supportCapacity&&count<=capacity
    &&supportWord(base+13u)==SUPPORT_VALID
    &&supportWord(base+14u)==SUPPORT_LAYOUT_VERSION
    &&seeds<=faces&&(count==0u||(supportWord(base+8u)>0u&&faces>0u&&seeds>0u));
}
fn taggedVelocity(tag:u32)->vec4f{
  if(tag==INVALID||!supportPublicationValid()){return invalidVector();}
  if((tag&SUPPORT_TAG)==0u){return velocitySample(tag);}
  let support=tag&0x7fffffffu;let count=supportCount();
  if(support>=count||support>=p.supportCapacity){return invalidVector();}
  let at=p.supportVectorOffsetWords/4u+support;
  if(at>=arrayLength(&transportMetrics)){return invalidVector();}
  let sample=transportMetrics[at];
  return select(invalidVector(),vec4f(sample.xyz,1.),vectorValid(sample));
}
fn axisNeighbor(row:u32,axis:u32,positive:bool)->u32{
  if(row>=acc(2u)||axis>=3u){return INVALID;}
  let direction=2u*axis+select(0u,1u,positive);
  let other=a[abase()+p.rowAxisOffset+6u*row+direction];
  if(other>=acc(2u)){return INVALID;}
  return other;
}
fn regularTag(row:u32,offset:vec3i)->u32{
  if(row>=acc(2u)||any(offset<vec3i(-1))||any(offset>vec3i(1))){return INVALID;}
  let local=u32(offset.x+1)+3u*u32(offset.y+1)+9u*u32(offset.z+1);
  return supportWord(p.regularTagOffsetWords+27u*row+local);
}
fn regularSample(anchor:u32,x:vec3f)->vec4f{
  if(anchor>=acc(2u)||!finite3(x)){return invalidVector();}
  let rg=rowGeometry[rbase()+anchor];
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let h=f32(rg.y)*p.physical.x;
  if(!finite(h)||h<=0.){return invalidVector();}
  // Cell-centred velocity has a constant physical-boundary extension. This
  // is the production basis at a domain wall: a missing live interior row
  // still rejects, while the basis never requests an exterior row.
  let sampleX=clamp(x,vec3f(.5*h),vec3f(d)*p.physical.x-vec3f(.5*h));
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  var lowOffset=vec3i(0);
  for(var axis=0u;axis<3u;axis+=1u){
    if(sampleX[axis]<center[axis]){lowOffset[axis]=-1;}
  }
  let lowCenter=center+vec3f(lowOffset)*h;
  let t=clamp((sampleX-lowCenter)/h,vec3f(0),vec3f(1));
  var result=vec3f(0.);
  for(var corner=0u;corner<8u;corner+=1u){
    let weight=select(1.-t.x,t.x,(corner&1u)!=0u)*select(1.-t.y,t.y,(corner&2u)!=0u)*select(1.-t.z,t.z,(corner&4u)!=0u);
    if(weight<=0.){continue;}
    var offset=lowOffset+vec3i(i32(corner&1u),i32((corner>>1u)&1u),i32((corner>>2u)&1u));
    let requestedCenter=center+vec3f(offset)*h;
    for(var axis=0u;axis<3u;axis+=1u){
      if(requestedCenter[axis]<.5*h||requestedCenter[axis]>f32(d[axis])*p.physical.x-.5*h){offset[axis]=0;}
    }
    let sample=taggedVelocity(regularTag(anchor,offset));
    if(!vectorValid(sample)){return invalidVector();}
    result+=weight*sample.xyz;
  }
  if(!finite3(result)){return invalidVector();}
  return vec4f(result,1.);
}
fn selectorVelocity(row:u32,selectorIndex:u32,selector:vec4f)->vec4f{
  if(row>=acc(2u)||selectorIndex>=p.tetraVertexCount||selectorIndex>=p.selectorStride
    ||!finite3(selector.xyz)||!finite(selector.w)){return vec4f(f32(selectorIndex),10.,f32(row),-1.);}
  if(length(selector.xyz)<1e-7){return velocitySample(row);}
  let selectorAt=p.selectorOffsetWords+row*p.selectorStride+selectorIndex;
  if(selectorAt>=4u*arrayLength(&transportMetrics)){return vec4f(f32(selectorIndex),1.,0.,-1.);}
  let other=supportWord(selectorAt);
  if(other!=INVALID){
    let sample=taggedVelocity(other);
    if(!vectorValid(sample)){return vec4f(f32(selectorIndex),2.,f32(other),-1.);}
    return sample;
  }
  // The paper's local Delaunay mesh includes face and edge neighbors, so a
  // missing selector inside the domain is incomplete live topology. Only a
  // catalog vertex proven exterior receives the physical-boundary extension.
  let rg=rowGeometry[rbase()+row];
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let extent=f32(rg.y)*p.physical.x;
  let neighborExtent=selector.w*extent;
  if(!finite(extent)||extent<=0.||!finite(neighborExtent)||neighborExtent<=0.){return invalidVector();}
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  let selectorCenter=center+inverseTransform(selector.xyz,metrics[row].transformAndFlags&63u)*extent;
  let lower=vec3f(.5*neighborExtent);
  let upper=vec3f(d)*p.physical.x-lower;
  let tolerance=max(1e-5,neighborExtent*2e-5);
  if(all(selectorCenter>=lower-vec3f(tolerance))&&all(selectorCenter<=upper+vec3f(tolerance))){
    return vec4f(f32(selectorIndex),3.,f32(other),-1.);
  }
  return velocitySample(row);
}
fn tetraWeights(point:vec3f,x:vec3f,y:vec3f,z:vec3f)->vec4f{
  let d=dot(x,cross(y,z));
  if(!finite(d)||abs(d)<1e-10){return vec4f(-2.);}
  let a0=dot(point,cross(y,z))/d;
  let a1=dot(x,cross(point,z))/d;
  let a2=dot(x,cross(y,point))/d;
  return vec4f(1.-a0-a1-a2,a0,a1,a2);
}
fn transitionSample(row:u32,x:vec3f)->vec4f{
  if(row>=acc(2u)||!finite3(x)){return vec4f(255.,20.,f32(row),-1.);}
  let rg=rowGeometry[rbase()+row];
  let d=dimensions();let q=vec3u(rg.x%d.x,(rg.x/d.x)%d.y,rg.x/(d.x*d.y));
  let extent=f32(rg.y)*p.physical.x;
  if(!finite(extent)||extent<=0.){return vec4f(255.,21.,extent,-1.);}
  let center=(vec3f(q)+.5*f32(rg.y))*p.physical.x;
  let local=powerTransform((x-center)/extent,metrics[row].transformAndFlags&63u);
  let thAt=p.tetraHeaderOffset+3u*metrics[row].caseId;
  let first=catalog[thAt];
  let count=catalog[thAt+1u];
  for(var ti=0u;ti<count;ti+=1u){
    let packed=catalog[p.tetraOffset+first+ti];
    let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);
    if(any(selectors>=vec3u(p.tetraVertexCount))){return vec4f(f32(ti),22.,f32(p.tetraVertexCount),-1.);}
    let va=p.tetraVertexOffset+4u*selectors.x;
    let vb=p.tetraVertexOffset+4u*selectors.y;
    let vc=p.tetraVertexOffset+4u*selectors.z;
    let sa=vec4f(bitsf(va),bitsf(va+1u),bitsf(va+2u),bitsf(va+3u));
    let sb=vec4f(bitsf(vb),bitsf(vb+1u),bitsf(vb+2u),bitsf(vb+3u));
    let sc=vec4f(bitsf(vc),bitsf(vc+1u),bitsf(vc+2u),bitsf(vc+3u));
    let weights=tetraWeights(local,sa.xyz,sb.xyz,sc.xyz);
    if(all(weights>=vec4f(-2e-6))&&all(weights<=vec4f(1.000002))){
      // Barycentric vertices with zero weight do not contribute to the
      // paper's tetrahedral interpolant. Avoid requiring an extrapolated air
      // row for those vertices, but keep every positive contributor strict.
      var positive=max(weights,vec4f(0.));
      let positiveSum=dot(positive,vec4f(1.));
      if(!finite(positiveSum)||positiveSum<=0.){return vec4f(f32(ti),25.,positiveSum,-1.);}
      positive/=positiveSum;
      var result=vec3f(0.);
      if(positive.x>0.){let v0=velocitySample(row);if(!vectorValid(v0)){return vec4f(f32(ti),23.,f32(row),-1.);}result+=positive.x*v0.xyz;}
      if(positive.y>0.){let v1=selectorVelocity(row,selectors.x,sa);if(!vectorValid(v1)){return v1;}result+=positive.y*v1.xyz;}
      if(positive.z>0.){let v2=selectorVelocity(row,selectors.y,sb);if(!vectorValid(v2)){return v2;}result+=positive.z*v2.xyz;}
      if(positive.w>0.){let v3=selectorVelocity(row,selectors.z,sc);if(!vectorValid(v3)){return v3;}result+=positive.w*v3.xyz;}
      if(!finite3(result)){return invalidVector();}
      return vec4f(result,1.);
    }
  }
  return vec4f(255.,24.,f32(count),-1.);
}
fn advect(cls:u32,index:u32,transition:bool){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  if(!supportPublicationValid()){transportMetrics[handle]=invalidVector();rejectSample(1u,handle);return;}
  let row=owner(handle);
  let useTransition=metrics[row].caseId!=0u;
  let x=centroid(handle);
  let prior=value(handle);
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  if(!finite(aperture)||aperture<0.||aperture>1.){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
  // A fully prescribed face has no transported degree of freedom. Publish
  // the exact solid boundary value before any characteristic sampling.
  if(aperture==0.){
    let solid=solidVelocityAt(handle);
    let n=normal(handle);
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    if(!finite(solid)||!finite3(n)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
    setValue(handle,solid);
    transportMetrics[handle]=vec4f(solid*n*area,.5*area*max(0.,prior*prior-solid*solid));
    return;
  }
  var adv=invalidVector();
  if(useTransition){adv=transitionSample(row,x);}else{adv=regularSample(row,x);}
  if(!vectorValid(adv)){transportMetrics[handle]=adv;rejectVector(1u,handle,adv,cls);return;}
  let departure=x-p.physical.y*adv.xyz;
  var transported=invalidVector();
  if(useTransition){transported=transitionSample(row,departure);}else{transported=regularSample(row,departure);}
  if(!vectorValid(transported)){transportMetrics[handle]=transported;rejectVector(2u,handle,transported,cls);return;}
  let n=normal(handle);
  let projected=dot(transported.xyz,n);
  let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
  if(!finite(projected)||!finite(area)||area<=0.||!finite(prior)){transportMetrics[handle]=invalidVector();rejectSample(3u,handle);return;}
  setValue(handle,projected);
  transportMetrics[handle]=vec4f(projected*n*area,.5*area*max(0.,prior*prior-projected*projected));
}
@compute @workgroup_size(64)fn advectStructuredClass5(@builtin(global_invocation_id)g:vec3u){advect(5u,g.x,false);}
@compute @workgroup_size(64)fn advectStructuredClass6(@builtin(global_invocation_id)g:vec3u){advect(6u,g.x,true);}
@compute @workgroup_size(64)fn advectStructuredClass7(@builtin(global_invocation_id)g:vec3u){advect(7u,g.x,false);}
@compute @workgroup_size(64)fn advectStructuredClass8(@builtin(global_invocation_id)g:vec3u){advect(8u,g.x,true);}

var<workgroup> projectionEnergyPartial:array<f32,128>;
var<workgroup> projectionEnergyCount:array<u32,128>;
var<workgroup> projectionEnergyInvalid:array<u32,128>;
fn summarizeProjectionEnergy(lane:u32,post:bool){
  var energy=0.;
  var count=0u;
  var invalid=0u;
  if(acc(0u)!=0u||acc(3u)==0u||!boundaryValid()){invalid=1u;}
  for(var cls=5u;cls<9u;cls+=1u){
    let base=wbase(cls);
    if(worksets[base]!=acc(3u)){invalid=1u;continue;}
    let size=worksets[base+1u];
    for(var index=lane;index<size;index+=128u){
      let handle=worksets[base+7u+index];
      if(handle>=acc(5u)){invalid=1u;continue;}
      let sample=value(handle);
      let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
      let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
      let inverseDistance=bitcast<f32>(a[abase()+p.inverseOffset+handle]);
      if(!finite(sample)||!finite(area)||area<=0.||!finite(aperture)||aperture<0.||aperture>1.
        ||!finite(inverseDistance)||inverseDistance<=0.){invalid=1u;continue;}
      let dualVolume=area/inverseDistance;
      let contribution=.5*aperture*dualVolume*sample*sample;
      if(!finite(contribution)||contribution<0.){invalid=1u;continue;}
      energy+=contribution;
      count+=1u;
    }
  }
  projectionEnergyPartial[lane]=energy;
  projectionEnergyCount[lane]=count;
  projectionEnergyInvalid[lane]=invalid;
  workgroupBarrier();
  for(var width=64u;width>0u;width>>=1u){
    if(lane<width){
      projectionEnergyPartial[lane]+=projectionEnergyPartial[lane+width];
      projectionEnergyCount[lane]+=projectionEnergyCount[lane+width];
      projectionEnergyInvalid[lane]|=projectionEnergyInvalid[lane+width];
    }
    workgroupBarrier();
  }
  if(lane==0u){
    let generation=acc(3u);
    let failed=projectionEnergyInvalid[0]!=0u||projectionEnergyCount[0]==0u
      ||!finite(projectionEnergyPartial[0]);
    if(!post){
      projectionEnergyStats[0]=select(0u,1u,failed);
      projectionEnergyStats[1]=generation;
      projectionEnergyStats[2]=bank();
      projectionEnergyStats[3]=projectionEnergyCount[0];
      projectionEnergyStats[4]=0u;
      projectionEnergyStats[5]=bitcast<u32>(projectionEnergyPartial[0]);
      projectionEnergyStats[6]=0u;
      projectionEnergyStats[7]=0u;
    }else{
      let pairMatches=projectionEnergyStats[0]==0u&&projectionEnergyStats[1]==generation
        &&projectionEnergyStats[2]==bank()&&projectionEnergyStats[3]==projectionEnergyCount[0];
      projectionEnergyStats[0]=select(0u,2u,failed||!pairMatches);
      projectionEnergyStats[4]=projectionEnergyCount[0];
      projectionEnergyStats[6]=bitcast<u32>(projectionEnergyPartial[0]);
      projectionEnergyStats[7]=select(generation,0u,failed||!pairMatches);
    }
  }
}
@compute @workgroup_size(128)
fn summarizeStructuredPreProjectionEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,false);
}
@compute @workgroup_size(128)
fn summarizeStructuredPostProjectionEnergy(@builtin(local_invocation_index)lane:u32){
  summarizeProjectionEnergy(lane,true);
}

fn closedWorld(handle:u32)->bool{
  if(neighbor(handle)!=INVALID){return false;}
  let x=centroid(handle)/p.physical.x;
  let n=normal(handle);
  for(var axis=0u;axis<3u;axis+=1u){
    if(n[axis]<-.5&&x[axis]<=1e-4&&(p.closedMask&(1u<<(2u*axis)))!=0u){return true;}
    if(n[axis]>.5&&x[axis]>=f32(dimensions()[axis])-1e-4&&(p.closedMask&(1u<<(2u*axis+1u)))!=0u){return true;}
  }
  return false;
}
fn forceFamily(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  let solid=solidVelocityAt(handle);
  if(!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)){rejectSample(10u,handle);return;}
  if(aperture==0.){setValue(handle,solid);return;}
  let forced=value(handle)+p.physical.y*dot(p.gravity.xyz,normal(handle));
  if(!finite(forced)){rejectSample(11u,handle);return;}
  setValue(handle,forced);
}
@compute @workgroup_size(64)fn forceStructuredClass5(@builtin(global_invocation_id)g:vec3u){forceFamily(5u,g.x);}
@compute @workgroup_size(64)fn forceStructuredClass6(@builtin(global_invocation_id)g:vec3u){forceFamily(6u,g.x);}
@compute @workgroup_size(64)fn forceStructuredClass7(@builtin(global_invocation_id)g:vec3u){forceFamily(7u,g.x);}
@compute @workgroup_size(64)fn forceStructuredClass8(@builtin(global_invocation_id)g:vec3u){forceFamily(8u,g.x);}

fn divergenceRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  if(liquidAt(row)==0u){rhs[row]=0.;return;}
  var flux=0.;
  let count=rowSlotCount(row);
  if(count>p.maxSlots){rhs[row]=0.;rejectSample(20u,row);return;}
  for(var local=0u;local<count;local+=1u){
    let at=row*p.maxSlots+local;
    let handle=a[abase()+p.rowHandleOffset+at];
    if(handle==INVALID||handle>=acc(5u)){rhs[row]=0.;rejectSample(21u,row);return;}
    let sign=f32(bitcast<i32>(a[abase()+p.rowSignOffset+at]));
    let area=bitcast<f32>(a[abase()+p.areaOffset+handle]);
    let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
    let solid=solidVelocityAt(handle);
    let sample=value(handle);
    if(!finite(sign)||!finite(area)||!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)||!finite(sample)){rhs[row]=0.;rejectSample(22u,row);return;}
    flux+=sign*area*(aperture*sample+(1.-aperture)*solid);
  }
  if(!finite(flux)||!finite(p.physical.y)||p.physical.y<0.
    ||!finite(p.physical.z)||p.physical.z<=0.){rhs[row]=0.;rejectSample(23u,row);return;}
  // The fenced t=0 publication is the exact zero-length member of the same
  // projection operator family.  It validates every geometric/velocity input
  // above, but its integrated divergence RHS is identically zero; dividing by
  // dt would be undefined.  Positive-time substeps retain the physical
  // integrated Eq. (3)/(4) equation below, and invalid inputs still fail closed.
  if(p.physical.y==0.){rhs[row]=0.;return;}
  // A stores area/distance and projection applies dt/rho times the pressure
  // gradient. Therefore A p = -(rho/dt) integral(u.n dA), with this solver's
  // residual convention. Dividing by cell volume would make the operator,
  // RHS, and projection describe different equations at different leaf sizes.
  rhs[row]=p.physical.z*flux/p.physical.y;
}
@compute @workgroup_size(64)fn divergenceStructuredClass0(@builtin(global_invocation_id)g:vec3u){divergenceRow(0u,g.x);}
@compute @workgroup_size(64)fn divergenceStructuredClass1(@builtin(global_invocation_id)g:vec3u){divergenceRow(1u,g.x);}
@compute @workgroup_size(64)fn divergenceStructuredClass2(@builtin(global_invocation_id)g:vec3u){divergenceRow(2u,g.x);}
@compute @workgroup_size(64)fn divergenceStructuredClass3(@builtin(global_invocation_id)g:vec3u){divergenceRow(3u,g.x);}

fn projectFamily(cls:u32,index:u32){
  let handle=workItem(cls,index);
  if(handle==INVALID||handle>=acc(5u)){return;}
  let lo=owner(handle);
  let hi=neighbor(handle);
  if(lo>=acc(2u)||(hi!=INVALID&&hi>=acc(2u))){rejectSample(30u,handle);return;}
  let n=normal(handle);
  if(!finite3(n)){rejectSample(31u,handle);return;}
  let aperture=bitcast<f32>(a[abase()+p.fractionOffset+handle]);
  let solid=solidVelocityAt(handle);
  if(!finite(aperture)||aperture<0.||aperture>1.||!finite(solid)){rejectSample(32u,handle);return;}
  var projected=solid;
  if(aperture>0.){
    var pressureLo=0.;
    var pressureHi=0.;
    if(liquidAt(lo)!=0u){pressureLo=pressure[lo];}
    if(hi!=INVALID&&liquidAt(hi)!=0u){pressureHi=pressure[hi];}
    let inv=bitcast<f32>(a[abase()+p.inverseOffset+handle]);
    let scale=bitcast<f32>(a[abase()+p.pressureScaleOffset+handle]);
    projected=value(handle)-p.physical.y*(pressureHi-pressureLo)*inv*scale/p.physical.z;
  }
  if(!finite(projected)){rejectSample(33u,handle);return;}
  setValue(handle,projected);
}
@compute @workgroup_size(64)fn projectStructuredClass5(@builtin(global_invocation_id)g:vec3u){projectFamily(5u,g.x);}
@compute @workgroup_size(64)fn projectStructuredClass6(@builtin(global_invocation_id)g:vec3u){projectFamily(6u,g.x);}
@compute @workgroup_size(64)fn projectStructuredClass7(@builtin(global_invocation_id)g:vec3u){projectFamily(7u,g.x);}
@compute @workgroup_size(64)fn projectStructuredClass8(@builtin(global_invocation_id)g:vec3u){projectFamily(8u,g.x);}

fn reconstructRow(cls:u32,index:u32){
  let row=workItem(cls,index);
  if(row==INVALID||row>=acc(2u)){return;}
  let header=caseHeader(metrics[row].caseId);
  if(header.y>p.maxSlots){rejectSample(50u,row);return;}
  var canonical=vec3f(0.);
  for(var local=0u;local<header.y;local+=1u){
    let at=row*p.maxSlots+local;
    let handle=a[abase()+p.rowHandleOffset+at];
    if(handle==INVALID||handle>=acc(5u)){rejectSample(51u,row);return;}
    let other=a[abase()+p.rowNeighborOffset+at];
    if(other!=INVALID&&other>=acc(2u)){rejectSample(52u,row);return;}
    let sample=f32(bitcast<i32>(a[abase()+p.rowSignOffset+at]))*value(handle);
    let global=a[abase()+p.rowCatalogOffset+at];
    let reconstruction=p.denseOffset+p.reconstructionOffset+3u*global;
    let coefficient=vec3f(bitsf(reconstruction),bitsf(reconstruction+1u),bitsf(reconstruction+2u));
    if(!finite(sample)||!finite3(coefficient)){rejectSample(53u,row);return;}
    canonical+=coefficient*sample;
  }
  let projected=inverseTransform(canonical,metrics[row].transformAndFlags&63u);
  if(!finite3(projected)){rejectSample(54u,row);return;}
  rowVelocity[rbase()+row]=vec4f(projected,1.);
}
@compute @workgroup_size(64)fn reconstructStructuredClass0(@builtin(global_invocation_id)g:vec3u){reconstructRow(0u,g.x);}
@compute @workgroup_size(64)fn reconstructStructuredClass1(@builtin(global_invocation_id)g:vec3u){reconstructRow(1u,g.x);}
@compute @workgroup_size(64)fn reconstructStructuredClass2(@builtin(global_invocation_id)g:vec3u){reconstructRow(2u,g.x);}
@compute @workgroup_size(64)fn reconstructStructuredClass3(@builtin(global_invocation_id)g:vec3u){reconstructRow(3u,g.x);}
`;
