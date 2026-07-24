/**
 * Aanjaneya et al. (2017), Section 5 velocity advection across an octree
 * topology change.
 *
 * The projected old generalized-face field is first reconstructed to full
 * vectors at old power-cell centres by `WebGPUOctreePowerVelocity`.  This
 * class snapshots that complete old interpolation mesh.  After the next
 * topology has been built, every new generalized-face centroid is traced
 * backward through the old full-vector interpolant and the sampled vector is
 * projected on the new face normal.  This retained mesh is the native
 * transition-region authority. Regular-region samples are deliberately left
 * pending for the mandatory Section 5 regular-face completion transaction.
 */

import {
  OCTREE_POWER_FACE_DELTA_CARRIED,
  type OctreePowerFaceSource,
} from "./webgpu-octree-power-faces";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";
import type { PassBroker } from "./webgpu-pass-broker";

export const OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES = 64;
/** WGSL `P` is 80 bytes because `vec3u dims` begins at a 16-byte boundary. */
export const OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES = 80;
export const OCTREE_POWER_OLD_MESH_ADVECTION_VALID = 0x8000_0000;
export const OCTREE_POWER_OLD_MESH_ADVECTION_ERROR = Object.freeze({
  source: 1 << 0,
  generation: 1 << 1,
  capacity: 1 << 2,
  interpolation: 1 << 3,
  nonfinite: 1 << 4,
} as const);
export const OCTREE_POWER_OLD_MESH_CAPTURE_BINDINGS =
  Object.freeze([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const);
export const OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS = Object.freeze([0, 1, 4, 5, 6, 7, 8, 9] as const);
export const OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS = Object.freeze([0, 1, 2, 5, 6, 7, 10, 11, 12] as const);
export const OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS = Object.freeze([1, 7, 8] as const);

export interface OctreePowerOldMeshAdvectionPlan {
  readonly rowCapacity: number;
  readonly faceCapacity: number;
  readonly rowDirectoryCapacity: number;
  readonly headerBytes: number;
  readonly metricOffsetBytes: number;
  readonly velocityOffsetBytes: number;
  readonly rowDirectoryOffsetBytes: number;
  readonly arenaBytes: number;
  readonly rowDirectoryBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerOldMeshAdvection(
  rowCapacity: number,
  faceCapacity: number,
  rowDirectoryCapacity: number,
): OctreePowerOldMeshAdvectionPlan {
  for (const [value, label] of [[rowCapacity, "row"], [faceCapacity, "face"],
    [rowDirectoryCapacity, "row directory"]] as const) {
    if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`Old-mesh ${label} capacity must be positive`);
  }
  const headerBytes = rowCapacity * 48;
  const metricOffsetBytes = headerBytes;
  const velocityOffsetBytes = metricOffsetBytes + rowCapacity * 16;
  const rowDirectoryBytes = rowDirectoryCapacity * 16;
  const rowDirectoryOffsetBytes = velocityOffsetBytes + rowCapacity * 16;
  const arenaBytes = rowDirectoryOffsetBytes + rowDirectoryBytes;
  return { rowCapacity, faceCapacity, rowDirectoryCapacity, headerBytes, metricOffsetBytes,
    velocityOffsetBytes, rowDirectoryOffsetBytes, arenaBytes, rowDirectoryBytes,
    allocatedBytes: arenaBytes + OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES
      + OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES };
}

export function packOctreePowerOldMeshAdvectionParameters(input: {
  rowCapacity: number; faceCapacity: number; rowDirectoryCapacity: number;
  metricOffsetWords: number; velocityOffsetWords: number;
  dimensions: readonly [number, number, number]; maximumLeafSize: number; generation: number;
  physicalCellSize: number; timestep: number;
  rowDirectoryOffsetWords?: number;
}): ArrayBuffer {
  const data = new ArrayBuffer(OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES);
  const words = new Uint32Array(data), floats = new Float32Array(data);
  words.set([input.rowCapacity, input.faceCapacity, input.rowDirectoryCapacity,
    input.metricOffsetWords, input.velocityOffsetWords]);
  // WGSL uniform layout: vec3u has 16-byte alignment, so words 5--7 are
  // explicit padding and `dims` starts at word 8.
  words.set(input.dimensions, 8);
  words[11] = input.maximumLeafSize; words[12] = input.generation >>> 0;
  floats[13] = input.physicalCellSize; floats[14] = input.timestep;
  // Word 15 is explicit padding before the retained row-directory offset.
  words[15] = 0;
  words[16] = input.rowDirectoryOffsetWords ?? 0;
  return data;
}

/** CPU oracle for the characteristic and final normal projection. */
export function advectPowerFaceFromOldVector(
  centroid: readonly [number, number, number],
  normal: readonly [number, number, number],
  dt: number,
  sampleOld: (point: readonly [number, number, number]) => readonly [number, number, number] | undefined,
): number | undefined {
  if (![...centroid, ...normal, dt].every(Number.isFinite) || dt < 0) return undefined;
  const first = sampleOld(centroid); if (!first?.every(Number.isFinite)) return undefined;
  const midpoint = centroid.map((value, axis) => value - 0.5 * dt * first[axis]) as [number, number, number];
  const middle = sampleOld(midpoint); if (!middle?.every(Number.isFinite)) return undefined;
  const departure = centroid.map((value, axis) => value - dt * middle[axis]) as [number, number, number];
  const advected = sampleOld(departure); if (!advected?.every(Number.isFinite)) return undefined;
  const projected = advected[0] * normal[0] + advected[1] * normal[1] + advected[2] * normal[2];
  return Number.isFinite(projected) ? projected : undefined;
}

export class WebGPUOctreePowerFaceAdvection {
  readonly plan: OctreePowerOldMeshAdvectionPlan;
  readonly control: GPUBuffer;
  private readonly oldArena: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly capturePipeline: GPUComputePipeline;
  private readonly preparePipeline: GPUComputePipeline;
  private readonly advectPipeline: GPUComputePipeline;
  private readonly finalizePipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(
    private readonly device: GPUDevice,
    private readonly topology: OctreePowerTopologySource,
    private readonly faces: OctreePowerFaceSource,
  ) {
    if (!topology.catalogTetrahedronHeaders || !topology.catalogTetrahedra || !topology.catalogTetrahedronVertices) {
      throw new RangeError("Old-mesh velocity advection requires the Section 5 Delaunay catalog");
    }
    this.plan = planOctreePowerOldMeshAdvection(topology.plan.rowCapacity, faces.plan.faceCapacity,
      faces.plan.rowDirectoryCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.oldArena = device.createBuffer({ label: "Old power interpolation mesh", size: this.plan.arenaBytes,
      usage: storage });
    this.control = device.createBuffer({ label: "Old-mesh power-face advection control",
      size: OCTREE_POWER_OLD_MESH_ADVECTION_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Old-mesh power-face advection parameters",
      size: OCTREE_POWER_OLD_MESH_ADVECTION_PARAMETER_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const captureModule = device.createShaderModule({ label: "Capture 2017 old-mesh authority",
      code: octreePowerOldMeshCaptureWGSL });
    const module = device.createShaderModule({ label: "2017 old-mesh full-vector face advection",
      code: octreePowerOldMeshAdvectionWGSL });
    this.capturePipeline = device.createComputePipeline({ label: "Capture old power interpolation authority",
      layout: "auto", compute: { module: captureModule, entryPoint: "captureOldMeshAuthority" } });
    this.preparePipeline = device.createComputePipeline({ label: "Prepare old-mesh power-face advection",
      layout: "auto", compute: { module, entryPoint: "prepareNewPowerFaceAdvection" } });
    this.advectPipeline = device.createComputePipeline({ label: "Advect new power faces through old mesh",
      layout: "auto", compute: { module, entryPoint: "advectNewPowerFaces" } });
    this.finalizePipeline = device.createComputePipeline({ label: "Publish old-mesh power-face advection",
      layout: "auto", compute: { module, entryPoint: "publishNewPowerFaceAdvection" } });
  }

  /** Capture generation N only after its projected full cell vectors publish.
   * One cooperative workgroup copies the exact live row prefix; no capacity
   * sized transfer or host-authored live count is retained. */
  encodeCapture(broker: PassBroker, input: {
    leafHeaders: GPUBuffer;
    rowVelocities: GPUBuffer;
    velocityControl: GPUBuffer;
  }): void {
    this.assertLive();
    const pass = broker.compute({ label: "Capture live old power interpolation rows" });
    pass.setPipeline(this.capturePipeline);
    pass.setBindGroup(0, this.device.createBindGroup({ layout: this.capturePipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: this.faces.control } },
      { binding: 1, resource: { buffer: input.velocityControl } },
      { binding: 2, resource: { buffer: this.topology.control } },
      { binding: 3, resource: { buffer: this.control } },
      { binding: 4, resource: { buffer: this.params } },
      { binding: 5, resource: { buffer: input.leafHeaders } },
      { binding: 6, resource: { buffer: this.topology.metrics } },
      { binding: 7, resource: { buffer: input.rowVelocities } },
      { binding: 8, resource: { buffer: this.faces.rowDirectory } },
      { binding: 9, resource: { buffer: this.oldArena } },
    ] }));
    pass.dispatchWorkgroups(1);
  }

  /** Overwrite generation N+1 face velocities from the captured generation N mesh. */
  encodeAdvect(broker: PassBroker, input: {
    seedControl: GPUBuffer;
    dimensions: readonly [number, number, number];
    physicalCellSize: number;
    maximumLeafSize: number;
    generation: number;
    timestep: number;
  }): void {
    this.assertLive();
    if (!input.dimensions.every((value) => Number.isSafeInteger(value) && value > 0)
      || !Number.isSafeInteger(input.maximumLeafSize) || input.maximumLeafSize < 1
      || !Number.isSafeInteger(input.generation) || input.generation < 1
      || !Number.isFinite(input.physicalCellSize) || input.physicalCellSize <= 0
      || !Number.isFinite(input.timestep) || input.timestep < 0) {
      throw new RangeError("Old-mesh face-advection parameters are invalid");
    }
    const data = packOctreePowerOldMeshAdvectionParameters({
      rowCapacity: this.plan.rowCapacity, faceCapacity: this.plan.faceCapacity,
      rowDirectoryCapacity: this.plan.rowDirectoryCapacity,
      metricOffsetWords: this.plan.metricOffsetBytes / 4,
      velocityOffsetWords: this.plan.velocityOffsetBytes / 4,
      dimensions: input.dimensions, maximumLeafSize: input.maximumLeafSize,
      generation: input.generation, physicalCellSize: input.physicalCellSize, timestep: input.timestep,
      rowDirectoryOffsetWords: this.plan.rowDirectoryOffsetBytes / 4,
    });
    this.device.queue.writeBuffer(this.params, 0, data);
    const t = this.topology;
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.params } },
      { binding: 1, resource: { buffer: this.control } },
      { binding: 2, resource: { buffer: this.oldArena } },
      { binding: 4, resource: { buffer: this.faces.control } },
      { binding: 5, resource: { buffer: this.faces.faces } },
      { binding: 6, resource: { buffer: this.faces.faceNormals } },
      { binding: 7, resource: { buffer: this.faces.faceCentroids } },
      { binding: 8, resource: { buffer: input.seedControl } },
      { binding: 9, resource: { buffer: this.faces.deltaStats } },
      { binding: 10, resource: { buffer: t.catalogTetrahedronHeaders! } },
      { binding: 11, resource: { buffer: t.catalogTetrahedra! } },
      { binding: 12, resource: { buffer: t.catalogTetrahedronVertices! } },
    ];
    const group = (pipeline: GPUComputePipeline, bindings: readonly number[]) => this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0), entries: entries.filter((entry) => bindings.includes(entry.binding)),
    });
    const prepare = broker.compute({ label: "Prepare Section 5 old-mesh face advection" });
    prepare.setPipeline(this.preparePipeline);
    prepare.setBindGroup(0, group(this.preparePipeline, OCTREE_POWER_OLD_MESH_PREPARE_BINDINGS));
    prepare.dispatchWorkgroups(1);
    const advect = broker.compute({ label: "Section 5 advect new power faces through old mesh" });
    advect.setPipeline(this.advectPipeline);
    advect.setBindGroup(0, group(this.advectPipeline, OCTREE_POWER_OLD_MESH_ADVECT_BINDINGS));
    advect.dispatchWorkgroupsIndirect(this.faces.liveFaceDispatch, 0);
    // Generation one is complete after the validated seed transaction. Every
    // recurring generation is completed and committed by the mandatory
    // regular-face transaction after this transition-specialized dispatch.
    if (input.generation === 1) {
      const finalize = broker.compute({ label: "Publish Section 5 old-mesh face advection" });
      finalize.setPipeline(this.finalizePipeline);
      finalize.setBindGroup(0, group(this.finalizePipeline, OCTREE_POWER_OLD_MESH_FINALIZE_BINDINGS));
      finalize.dispatchWorkgroups(1);
    }
  }

  destroy(): void {
    if (this.destroyed) return; this.destroyed = true;
    this.oldArena.destroy(); this.control.destroy(); this.params.destroy();
  }
  private assertLive(): void { if (this.destroyed) throw new Error("Old-mesh power-face advection is destroyed"); }
}

export const octreePowerOldMeshCaptureWGSL = /* wgsl */ `
@group(0)@binding(0)var<storage,read>faceControl:array<u32>;
@group(0)@binding(1)var<storage,read>velocityControl:array<u32>;
@group(0)@binding(2)var<storage,read>topologyControl:array<u32>;
@group(0)@binding(3)var<storage,read_write>snapshot:array<u32>;
struct P{rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,metricBase:u32,velocityBase:u32,dims:vec3u,maximumLeafSize:u32,generation:u32,cellSize:f32,dt:f32,pad15:u32,rowDirectoryBase:u32,p2:u32,p3:u32}
@group(0)@binding(4)var<uniform>p:P;
@group(0)@binding(5)var<storage,read>headers:array<u32>;
@group(0)@binding(6)var<storage,read>metrics:array<u32>;
@group(0)@binding(7)var<storage,read>velocities:array<vec4f>;
@group(0)@binding(8)var<storage,read>directory:array<vec4u>;
@group(0)@binding(9)var<storage,read_write>arena:array<u32>;
const VALID=0x80000000u;const SOURCE=1u;const CAPACITY=4u;
var<workgroup>captureState:vec3u;
@compute @workgroup_size(256)fn captureOldMeshAuthority(@builtin(local_invocation_index)lid:u32){
 if(lid==0u){
  var flags=0u;var rows=0u;var generation=0u;
  if(arrayLength(&faceControl)<9u||arrayLength(&velocityControl)<8u||arrayLength(&topologyControl)<5u||arrayLength(&snapshot)<16u){flags|=CAPACITY;}
  else{rows=faceControl[0];generation=faceControl[7];
   let arenaFits=p.rowDirectoryBase<=arrayLength(&arena)&&rows<=p.rowDirectoryCapacity
    &&rows<=p.rowCapacity&&rows*12u<=arrayLength(&headers)&&rows*4u<=arrayLength(&metrics)
    &&rows<=arrayLength(&velocities)&&rows<=arrayLength(&directory)
    &&p.rowDirectoryBase+rows*4u<=arrayLength(&arena);
   if(!arenaFits){flags|=CAPACITY;}
   if(faceControl[8]!=VALID||faceControl[3]!=0u||velocityControl[0]!=VALID
    ||velocityControl[2]!=rows||velocityControl[5]!=rows||velocityControl[7]!=generation
    ||topologyControl[0]!=0u||topologyControl[2]!=0u||topologyControl[3]!=rows){flags|=SOURCE;}
  }
  snapshot[0]=flags;snapshot[1]=0xffffffffu;snapshot[6]=0u;
  captureState=vec3u(flags,rows,generation);
 }
 workgroupBarrier();
 let flags=captureState.x;let rows=captureState.y;
 if(flags!=0u){return;}
 for(var row=lid;row<rows;row+=256u){
  let headerBase=row*12u;for(var word=0u;word<12u;word+=1u){arena[headerBase+word]=headers[headerBase+word];}
  let metricBase=p.metricBase+row*4u;let sourceMetric=row*4u;
  for(var word=0u;word<4u;word+=1u){arena[metricBase+word]=metrics[sourceMetric+word];}
  let velocityBase=p.velocityBase+row*4u;let velocity=velocities[row];
  arena[velocityBase]=bitcast<u32>(velocity.x);arena[velocityBase+1u]=bitcast<u32>(velocity.y);
  arena[velocityBase+2u]=bitcast<u32>(velocity.z);arena[velocityBase+3u]=bitcast<u32>(velocity.w);
  let directoryBase=p.rowDirectoryBase+row*4u;let entry=directory[row];
  arena[directoryBase]=entry.x;arena[directoryBase+1u]=entry.y;
  arena[directoryBase+2u]=entry.z;arena[directoryBase+3u]=entry.w;
 }
 if(lid==0u){snapshot[2]=rows;snapshot[4]=captureState.z;snapshot[5]=captureState.z;
  snapshot[6]=select(0u,VALID,rows>0u);}
}
`;

export const octreePowerOldMeshAdvectionWGSL = /* wgsl */ `
struct P{rowCapacity:u32,faceCapacity:u32,rowDirectoryCapacity:u32,metricBase:u32,velocityBase:u32,dims:vec3u,maximumLeafSize:u32,generation:u32,cellSize:f32,dt:f32,pad15:u32,rowDirectoryBase:u32,p2:u32,p3:u32}
struct Face{negativeRow:u32,positiveRow:u32,geometryCode:u32,flags:u32,normalVelocity:f32,area:f32,inverseDistance:f32,openFraction:f32}
struct TH{first:u32,count:u32,flags:u32}struct TV{v:vec4f}
struct C{flags:u32,firstError:u32,faceCount:u32,advected:u32,generation:u32,oldGeneration:u32,valid:u32,mode:u32,p0:u32,p1:u32,p2:u32,p3:u32,p4:u32,firstInterpolationStage:u32,firstInterpolationReason:u32,p7:u32}
struct Seed{flags:u32,firstError:u32,rowCount:u32,faceCount:u32,seededCount:u32,generation:u32,valid:u32}
@group(0)@binding(0)var<uniform>p:P;@group(0)@binding(1)var<storage,read_write>control:C;@group(0)@binding(2)var<storage,read>arena:array<u32>;@group(0)@binding(4)var<storage,read>faceControl:array<u32>;@group(0)@binding(5)var<storage,read_write>faces:array<Face>;@group(0)@binding(6)var<storage,read>normals:array<vec4f>;@group(0)@binding(7)var<storage,read_write>centroids:array<vec4f>;@group(0)@binding(8)var<storage,read_write>seed:Seed;@group(0)@binding(10)var<storage,read>tetraHeaders:array<TH>;@group(0)@binding(11)var<storage,read>tetrahedra:array<u32>;@group(0)@binding(12)var<storage,read>vertices:array<TV>;
@group(0)@binding(9)var<storage,read>faceDelta:array<u32>;
const INVALID=0xffffffffu;const VALID=0x80000000u;const STATUS_VALID=0x3f800000u;
const BOUNDARY=1u;const OPEN=2u;const DELTA_CARRIED:u32=${OCTREE_POWER_FACE_DELTA_CARRIED}u;
const FACE_DELTA_VALID=0x46444c54u;const INITIALIZE=1u;const ADVECT=2u;
const SOURCE=1u;const GENERATION=2u;const CAPACITY=4u;const INTERPOLATION=8u;const NONFINITE=16u;
const STATUS_NONFINITE=254u;const STATUS_INVALID_FACE=253u;const STATUS_MISSING=252u;
fn finite(x:f32)->bool{return x==x&&abs(x)<=3.402823e38;}
fn fail(code:u32,index:u32){control.flags|=code;control.firstError=min(control.firstError,index);seed.flags|=code;seed.firstError=min(seed.firstError,index);seed.valid=0u;}
fn mortonPart10(value:u32)->u32{var x=value&1023u;x=(x|(x<<16u))&0x030000ffu;x=(x|(x<<8u))&0x0300f00fu;x=(x|(x<<4u))&0x030c30c3u;x=(x|(x<<2u))&0x09249249u;return x;}
fn morton(c:u32)->u32{let q=vec3u(c%p.dims.x,(c/p.dims.x)%p.dims.y,c/(p.dims.x*p.dims.y));return mortonPart10(q.x)|(mortonPart10(q.y)<<1u)|(mortonPart10(q.z)<<2u);}
fn less(levelA:u32,mortonA:u32,levelB:u32,mortonB:u32)->bool{return levelA<levelB||(levelA==levelB&&mortonA<mortonB);}
fn find(c:u32,s:u32)->u32{let cap=min(control.faceCount,min(p.rowDirectoryCapacity,(arrayLength(&arena)-min(p.rowDirectoryBase,arrayLength(&arena)))/4u));let level=31u-countLeadingZeros(s);let wanted=morton(c);var low=0u;var high=cap;while(low<high){let middle=low+(high-low)/2u;let base=p.rowDirectoryBase+middle*4u;let candidateSize=arena[base+1u];let candidateMorton=arena[base+3u];if(less(31u-countLeadingZeros(candidateSize),candidateMorton,level,wanted)){low=middle+1u;}else{high=middle;}}if(low<cap){let base=p.rowDirectoryBase+low*4u;if(arena[base]==c+1u&&arena[base+1u]==s){return arena[base+2u];}}return INVALID;}
fn owner(x:vec3f)->u32{let g=x/p.cellSize;if(any(g<vec3f(0))||any(g>=vec3f(p.dims))){return INVALID;}let q=vec3u(floor(g));var s=1u;loop{let o=(q/s)*s;let c=o.x+p.dims.x*(o.y+p.dims.y*o.z);let r=find(c,s);if(r!=INVALID){return r;}if(s>=p.maximumLeafSize){break;}s*=2u;}return INVALID;}
fn header(row:u32)->vec4u{let b=row*12u;return vec4u(arena[b],arena[b+1u],arena[b+2u],arena[b+3u]);}fn metric(row:u32)->vec2u{let b=p.metricBase+row*4u;return vec2u(arena[b],arena[b+1u]);}fn velocity(row:u32)->vec4f{let b=p.velocityBase+row*4u;return vec4f(bitcast<f32>(arena[b]),bitcast<f32>(arena[b+1u]),bitcast<f32>(arena[b+2u]),bitcast<f32>(arena[b+3u]));}
fn inv(x:vec3f,c:u32)->vec3f{let bits=c&7u;let q=x*vec3f(select(1.,-1.,(bits&1u)!=0u),select(1.,-1.,(bits&2u)!=0u),select(1.,-1.,(bits&4u)!=0u));let k=(c/8u)%6u;if(k==0u){return q;}if(k==1u){return q.xzy;}if(k==2u){return q.yxz;}if(k==3u){return q.zxy;}if(k==4u){return q.yzx;}return q.zyx;}
fn weights(point:vec3f,a:vec3f,b:vec3f,c:vec3f)->vec4f{let d=dot(a,cross(b,c));if(abs(d)<=1e-9){return vec4f(-1.);}let q=vec3f(dot(point,cross(b,c)),dot(a,cross(point,c)),dot(a,cross(b,point)))/d;return vec4f(1.-q.x-q.y-q.z,q);}
fn contained(w:vec4f)->bool{return all(w>=vec4f(-2e-5))&&all(w<=vec4f(1.00002));}
fn bad(code:f32)->vec4f{return vec4f(0.,0.,0.,-code);}fn sampleOld(x:vec3f)->vec4f{let oldRows=control.faceCount;let row=owner(x);if(row==INVALID||row>=oldRows||row>=p.rowCapacity){return bad(1.);}let h=header(row);let m=metric(row);let anchor=velocity(row);if(anchor.w<=0.||m.x>=arrayLength(&tetraHeaders)||(m.y&VALID)==0u){return bad(2.);}let o=vec3u(h.x%p.dims.x,(h.x/p.dims.x)%p.dims.y,h.x/(p.dims.x*p.dims.y));let size=h.w;let center=(vec3f(o)+.5*f32(size))*p.cellSize;let th=tetraHeaders[m.x];if((th.flags&1u)!=0u){return bad(13.);}
 let point=inv((x-center)/(f32(size)*p.cellSize),m.y&63u);if(th.first>arrayLength(&tetrahedra)||th.count>arrayLength(&tetrahedra)-th.first){return bad(7.);}for(var local=0u;local<th.count;local+=1u){let packed=tetrahedra[th.first+local];let s=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(s>=vec3u(arrayLength(&vertices)))){return bad(8.);}let w=weights(point,vertices[s.x].v.xyz,vertices[s.y].v.xyz,vertices[s.z].v.xyz);if(!contained(w)){continue;}var vv:array<vec4f,3>;for(var k=0u;k<3u;k+=1u){let q=vertices[s[k]].v;let ns=u32(round(f32(size)*q.w));let nc=center+f32(size)*p.cellSize*inv(q.xyz,m.y&63u);let no=round(nc/p.cellSize-.5*f32(ns));if(any(no<vec3f(0))){return bad(9.);}let n=vec3u(no);let r=find(n.x+p.dims.x*(n.y+p.dims.y*n.z),ns);if(r==INVALID||r>=oldRows){return bad(10.);}vv[k]=velocity(r);if(vv[k].w<=0.){return bad(11.);}}let result=w.x*anchor.xyz+w.y*vv[0].xyz+w.z*vv[1].xyz+w.w*vv[2].xyz;return vec4f(result,1.);}return bad(12.);}
// A power-face centroid lies on the boundary of two dual interpolation
// elements.  The paper's dual cube/tetrahedral interpolant includes that
// boundary, whereas the half-open integer owner map assigns an exact grid
// plane to only one side.  At a liquid-air face that selected side need not
// be a pressure row.  Resolve only that measure-zero ambiguity through the
// incident liquid-side element; a genuinely air-side point still fails.
fn sampleOldIncident(x:vec3f,n:vec3f)->vec4f{let direct=sampleOld(x);if(direct.w>0.||direct.w!=-1.){return direct;}let epsilon=p.cellSize*1e-4;let incident=sampleOld(x-epsilon*n);if(incident.w>0.){return incident;}return direct;}
@compute @workgroup_size(1)fn prepareNewPowerFaceAdvection(){
 let oldGeneration=control.generation;let oldValid=control.valid;let oldRows=control.faceCount;
 control.p1=control.flags;control.p2=control.firstError;control.p3=control.advected;control.p4=control.valid;
 control.flags=0u;control.firstError=INVALID;control.advected=0u;
 control.firstInterpolationStage=INVALID;control.firstInterpolationReason=INVALID;control.p7=INVALID;
 control.oldGeneration=oldGeneration;control.generation=p.generation;control.valid=0u;control.mode=0u;
 let count=select(0u,faceControl[1],arrayLength(&faceControl)>=9u);control.p0=count;
 let facePublication=arrayLength(&faceControl)>=9u&&faceControl[8]==VALID&&faceControl[3]==0u
  &&faceControl[7]==p.generation&&count>0u&&count<=p.faceCapacity&&count<=arrayLength(&faces)
  &&count<=arrayLength(&normals)&&count<=arrayLength(&centroids);
 let deltaPublication=arrayLength(&faceDelta)>=4u&&faceDelta[3]==FACE_DELTA_VALID
  &&faceDelta[0]<=count&&faceDelta[1]<=count&&faceDelta[0]+faceDelta[1]==count;
 if(!facePublication||!deltaPublication){fail(CAPACITY,0u);return;}
 if(p.generation==1u){
  // Generation one is an explicit seed transaction: it has no predecessor,
  // every face is new, and the validated authored seed is already in-place.
  if(faceDelta[0]!=0u||faceDelta[1]!=count||seed.flags!=0u||seed.faceCount!=count
    ||seed.seededCount!=count||seed.generation!=p.generation||seed.valid!=VALID){fail(SOURCE,0u);return;}
  control.mode=INITIALIZE;control.faceCount=0u;return;
 }
 if(oldValid!=VALID||oldRows==0u){fail(SOURCE,0u);return;}
 if(oldGeneration+1u!=p.generation){fail(GENERATION,0u);return;}
 control.mode=ADVECT;control.faceCount=oldRows;
}
fn storeStatus(i:u32,status:u32){centroids[i].w=bitcast<f32>(status);}
fn pending(i:u32,stage:u32,value:vec4f){let reason=u32(max(1.,-value.w));storeStatus(i,(stage<<8u)|min(reason,255u));}
@compute @workgroup_size(64)fn advectNewPowerFaces(@builtin(global_invocation_id)id:vec3u){
 let i=id.x;let count=control.p0;if(i>=count||i>=arrayLength(&centroids)||control.flags!=0u){return;}
 storeStatus(i,0u);var f=faces[i];
 if(control.mode==INITIALIZE){f.flags&=~DELTA_CARRIED;faces[i]=f;storeStatus(i,STATUS_VALID);return;}
 if((f.flags&DELTA_CARRIED)!=0u){
  f.flags&=~DELTA_CARRIED;
  if(!finite(f.normalVelocity)){faces[i]=f;storeStatus(i,(3u<<8u)|STATUS_NONFINITE);return;}
  faces[i]=f;storeStatus(i,STATUS_VALID);return;
 }
 let n=normals[i].xyz;let x=centroids[i].xyz;
 if((f.flags&VALID)==0u||!finite(x.x)||!finite(x.y)||!finite(x.z)
  ||!finite(n.x)||!finite(n.y)||!finite(n.z)){storeStatus(i,(1u<<8u)|STATUS_INVALID_FACE);return;}
 if(f.positiveRow==INVALID&&(f.flags&BOUNDARY)!=0u&&(f.flags&OPEN)==0u){
  f.normalVelocity=0.;faces[i]=f;storeStatus(i,STATUS_VALID);return;
 }
 let v0=sampleOldIncident(x,n);if(v0.w<=0.){pending(i,1u,v0);return;}
 let vm=sampleOldIncident(x-.5*p.dt*v0.xyz,n);if(vm.w<=0.){pending(i,2u,vm);return;}
 let va=sampleOldIncident(x-p.dt*vm.xyz,n);if(va.w<=0.){pending(i,3u,va);return;}
 let value=dot(va.xyz,n);if(!finite(value)){storeStatus(i,(3u<<8u)|STATUS_NONFINITE);return;}
 f.normalVelocity=value;faces[i]=f;storeStatus(i,STATUS_VALID);
}
var<workgroup>advectionReduce:array<vec4u,256>;
var<workgroup>reasonReduce:array<u32,256>;
@compute @workgroup_size(256)fn publishNewPowerFaceAdvection(@builtin(local_invocation_index)lid:u32){
 let count=control.p0;var flags=0u;var first=INVALID;var completed=0u;var firstStage=INVALID;var firstReason=INVALID;
 if(control.flags==0u){
  for(var i=lid;i<count;i+=256u){
   var status=STATUS_MISSING;if(i<arrayLength(&centroids)){status=bitcast<u32>(centroids[i].w);}
   if(status==STATUS_VALID){completed+=1u;continue;}
   let stage=(status>>8u)&3u;let reason=status&255u;
   flags|=select(INTERPOLATION,NONFINITE,reason>=STATUS_INVALID_FACE);
   first=min(first,i);firstStage=min(firstStage,i*4u+stage);firstReason=min(firstReason,i*16u+min(reason,15u));
  }
 }
 advectionReduce[lid]=vec4u(flags,first,completed,firstStage);reasonReduce[lid]=firstReason;
 for(var width=128u;width>0u;width>>=1u){workgroupBarrier();if(lid<width){
  advectionReduce[lid].x|=advectionReduce[lid+width].x;
  advectionReduce[lid].y=min(advectionReduce[lid].y,advectionReduce[lid+width].y);
  advectionReduce[lid].z+=advectionReduce[lid+width].z;
  advectionReduce[lid].w=min(advectionReduce[lid].w,advectionReduce[lid+width].w);
  reasonReduce[lid]=min(reasonReduce[lid],reasonReduce[lid+width]);
 }}
 workgroupBarrier();if(lid==0u){
  let result=advectionReduce[0];control.flags|=result.x;control.firstError=min(control.firstError,result.y);
  control.advected=result.z;control.firstInterpolationStage=result.w;control.firstInterpolationReason=reasonReduce[0];
  control.valid=select(0u,VALID,control.flags==0u&&count>0u&&result.z==count);
  if(control.valid!=VALID){seed.flags|=control.flags;seed.firstError=min(seed.firstError,control.firstError);seed.valid=0u;}
  control.p1=control.flags;control.p2=control.firstError;control.p3=control.advected;control.p4=control.valid;
 }}
`;
