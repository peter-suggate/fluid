/** Transition-only coarse-octree redistance over catalog local tetrahedra. */

import type { OctreePowerCatalogEntry } from "./octree-power-catalog";
import type { PowerVec3 } from "./octree-power-geometry";
import type { OctreePowerTopologySource } from "./webgpu-octree-power-topology";

export const OCTREE_POWER_REDISTANCE_QUERY_BYTES = 48;
export const OCTREE_POWER_REDISTANCE_CONTROL_BYTES = 32;
export const OCTREE_POWER_REDISTANCE_VALID = 0x8000_0000;
export const OCTREE_POWER_REDISTANCE_ERROR = Object.freeze({
  capacity: 1,
  invalidQuery: 2,
  noCausalSimplex: 4,
  /** @deprecated ABI alias retained for older diagnostic consumers. */
  noKnownNeighbor: 4,
} as const);

export interface OctreePowerRedistanceResult {
  readonly signedDistance: number;
  readonly mode: "tetrahedron";
  readonly tetrahedron: number;
}

const cross = (a: PowerVec3, b: PowerVec3): PowerVec3 => [
  a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
];
const dot = (a: PowerVec3, b: PowerVec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function solveTranspose(columns: readonly [PowerVec3, PowerVec3, PowerVec3], rhs: PowerVec3): PowerVec3 | undefined {
  const determinant = dot(columns[0], cross(columns[1], columns[2]));
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12) return undefined;
  const terms = [cross(columns[1], columns[2]), cross(columns[2], columns[0]), cross(columns[0], columns[1])];
  return [0, 1, 2].reduce((sum, index) => sum.map((value, axis) =>
    value + rhs[index] * terms[index][axis] / determinant) as [number, number, number], [0, 0, 0] as [number, number, number]);
}

function solveColumns(columns: readonly [PowerVec3, PowerVec3, PowerVec3], rhs: PowerVec3): PowerVec3 | undefined {
  const determinant = dot(columns[0], cross(columns[1], columns[2]));
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= 1e-12) return undefined;
  return [dot(rhs, cross(columns[1], columns[2])) / determinant,
    dot(columns[0], cross(rhs, columns[2])) / determinant,
    dot(columns[0], cross(columns[1], rhs)) / determinant];
}

/**
 * Solves |grad phi|=1 at the anchor from accepted magnitudes on an incident
 * tetrahedron. The larger causal quadratic root is accepted only when its
 * characteristic intersects the opposite triangle. Catalog order breaks ties.
 */
export function redistanceOctreePowerCatalogCell(
  entry: OctreePowerCatalogEntry,
  tetrahedronVertexData: ArrayLike<number>,
  neighborMagnitudes: readonly (number | undefined)[],
  anchorSize: number,
  sign: -1 | 1,
  tolerance = 2e-6,
): OctreePowerRedistanceResult {
  const selectorCount = Math.floor(tetrahedronVertexData.length / 4);
  if (entry.uniform) throw new RangeError("Catalog tetrahedron redistance is transition-only");
  if (tetrahedronVertexData.length % 4 !== 0 || selectorCount > 256 || neighborMagnitudes.length < selectorCount
    || !(anchorSize > 0) || !Number.isFinite(anchorSize)
    || (sign !== -1 && sign !== 1) || !Number.isFinite(tolerance) || tolerance < 0) {
    throw new RangeError("Invalid power redistance input");
  }
  let best = Infinity, bestTetrahedron = 0xffff_ffff;
  entry.tetrahedra.forEach((selectors, tetrahedron) => {
    const values = selectors.map((selector) => neighborMagnitudes[selector]);
    if (values.some((value) => value === undefined || !Number.isFinite(value) || value < 0)) return;
    const known = values as number[];
    const positions = selectors.map((selector) => [0, 1, 2].map((axis) => tetrahedronVertexData[selector * 4 + axis] * anchorSize));
    const columns: [PowerVec3, PowerVec3, PowerVec3] = [
      positions[0] as [number, number, number], positions[1] as [number, number, number], positions[2] as [number, number, number],
    ];
    const lengths = columns.map((value) => Math.sqrt(dot(value, value)));
    const solidAngleDenominator = lengths[0] * lengths[1] * lengths[2] + dot(columns[0], columns[1]) * lengths[2]
      + dot(columns[0], columns[2]) * lengths[1] + dot(columns[1], columns[2]) * lengths[0];
    const solidAngleNumerator = Math.abs(dot(columns[0], cross(columns[1], columns[2])));
    const angleTolerance = tolerance * Math.max(1, Math.abs(solidAngleDenominator), solidAngleNumerator);
    if (solidAngleDenominator + angleTolerance < solidAngleNumerator) return;
    const a = solveTranspose(columns, known as [number, number, number]);
    const b = solveTranspose(columns, [1, 1, 1]);
    if (!a || !b) return;
    const quadratic = dot(b, b), projection = dot(a, b), constant = dot(a, a) - 1;
    const discriminant = projection * projection - quadratic * constant;
    if (!(quadratic > 0) || discriminant < -tolerance) return;
    const candidate = (projection + Math.sqrt(Math.max(0, discriminant))) / quadratic;
    if (!Number.isFinite(candidate) || candidate + tolerance < Math.max(...known)) return;
    const gradient = a.map((value, axis) => value - candidate * b[axis]) as [number, number, number];
    const ray = solveColumns(columns, gradient.map((value) => -value) as [number, number, number]);
    if (!ray) return;
    const sum = ray[0] + ray[1] + ray[2];
    if (!(sum > tolerance) || ray.some((value) => value / sum < -tolerance)) return;
    if (candidate < best) { best = candidate; bestTetrahedron = tetrahedron; }
  });
  if (Number.isFinite(best)) return { signedDistance: sign * best, mode: "tetrahedron", tetrahedron: bestTetrahedron };
  throw new RangeError("Power redistance has no causal nonobtuse Delaunay simplex");
}

export interface OctreePowerRedistancePlan {
  readonly queryCapacity: number;
  readonly resultBytes: number;
  readonly statusBytes: number;
  readonly allocatedBytes: number;
}

export function planOctreePowerRedistance(queryCapacityValue: number): OctreePowerRedistancePlan {
  if (!Number.isSafeInteger(queryCapacityValue) || queryCapacityValue < 1) throw new RangeError("Power redistance capacity must be positive");
  const resultBytes = queryCapacityValue * 4, statusBytes = queryCapacityValue * 4;
  return { queryCapacity: queryCapacityValue, resultBytes, statusBytes,
    allocatedBytes: resultBytes + statusBytes + OCTREE_POWER_REDISTANCE_CONTROL_BYTES + 16 };
}

export interface OctreePowerRedistanceControl {
  readonly flags: number;
  readonly firstError: number;
  readonly queryCount: number;
  readonly updatedCount: number;
  readonly tetrahedronCount: number;
  readonly nearestFallbackCount: number;
  readonly reserved: number;
  readonly generation: number;
}

export function unpackOctreePowerRedistanceControl(words: ArrayLike<number>): OctreePowerRedistanceControl {
  if (words.length < 8) throw new RangeError("Power redistance control needs eight words");
  return { flags: Number(words[0]) >>> 0, firstError: Number(words[1]) >>> 0, queryCount: Number(words[2]) >>> 0,
    updatedCount: Number(words[3]) >>> 0, tetrahedronCount: Number(words[4]) >>> 0,
    nearestFallbackCount: Number(words[5]) >>> 0, reserved: Number(words[6]) >>> 0, generation: Number(words[7]) >>> 0 };
}

export class WebGPUOctreePowerRedistance {
  readonly plan: OctreePowerRedistancePlan;
  readonly results: GPUBuffer;
  readonly statuses: GPUBuffer;
  readonly control: GPUBuffer;
  private readonly params: GPUBuffer;
  private readonly updatePipeline: GPUComputePipeline;
  private readonly publishPipeline: GPUComputePipeline;
  private destroyed = false;

  constructor(private readonly device: GPUDevice, queryCapacity: number, private readonly topology: OctreePowerTopologySource) {
    if (!topology.catalogTetrahedra || !topology.catalogTetrahedronVertices) throw new RangeError("Coarse power redistance requires catalog tetrahedra");
    this.plan = planOctreePowerRedistance(queryCapacity);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.results = device.createBuffer({ label: "Octree power redistance results", size: this.plan.resultBytes, usage: storage });
    this.statuses = device.createBuffer({ label: "Octree power redistance statuses", size: this.plan.statusBytes, usage: storage });
    this.control = device.createBuffer({ label: "Octree power redistance control", size: OCTREE_POWER_REDISTANCE_CONTROL_BYTES, usage: storage });
    this.params = device.createBuffer({ label: "Octree power redistance params", size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shader = device.createShaderModule({ label: "Octree power tetrahedral redistance", code: octreePowerRedistanceShader });
    this.updatePipeline = device.createComputePipeline({ label: "Update octree power distance", layout: "auto", compute: { module: shader, entryPoint: "updatePowerDistance" } });
    this.publishPipeline = device.createComputePipeline({ label: "Publish octree power redistance", layout: "auto", compute: { module: shader, entryPoint: "publishPowerDistance" } });
  }

  encode(encoder: GPUCommandEncoder, queries: GPUBuffer, neighborMagnitudes: GPUBuffer, queryCountValue: number, generationValue = 0): void {
    if (this.destroyed) throw new Error("Octree power redistance is destroyed");
    if (!Number.isSafeInteger(queryCountValue) || queryCountValue < 0 || queryCountValue > 0xffff_ffff
      || !Number.isSafeInteger(generationValue) || generationValue < 0 || generationValue > 0xffff_ffff) {
      throw new RangeError("Power redistance counts must be unsigned u32 integers");
    }
    this.device.queue.writeBuffer(this.params, 0, new Uint32Array([queryCountValue, this.plan.queryCapacity, generationValue, 0]));
    this.device.queue.writeBuffer(this.control, 0, new Uint32Array([0, 0xffff_ffff, queryCountValue, 0, 0, 0, 0, generationValue]));
    const resource = (buffer: GPUBuffer) => ({ buffer });
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: resource(this.params) }, { binding: 1, resource: resource(queries) },
      { binding: 2, resource: resource(neighborMagnitudes) }, { binding: 3, resource: resource(this.topology.catalogTetrahedronVertices!) },
      { binding: 4, resource: resource(this.topology.catalogTetrahedra!) }, { binding: 5, resource: resource(this.results) },
      { binding: 6, resource: resource(this.statuses) }, { binding: 7, resource: resource(this.control) },
    ];
    if (queryCountValue > 0) {
      const group = this.device.createBindGroup({ layout: this.updatePipeline.getBindGroupLayout(0), entries });
      const pass = encoder.beginComputePass({ label: "Tetrahedral coarse-octree redistance" });
      pass.setPipeline(this.updatePipeline); pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(Math.ceil(Math.min(queryCountValue, this.plan.queryCapacity) / 64)); pass.end();
    }
    const publishGroup = this.device.createBindGroup({ layout: this.publishPipeline.getBindGroupLayout(0),
      entries: entries.filter((entry) => [0, 1, 5, 6, 7].includes(entry.binding)) });
    const publish = encoder.beginComputePass({ label: "Publish tetrahedral coarse-octree redistance" });
    publish.setPipeline(this.publishPipeline); publish.setBindGroup(0, publishGroup); publish.dispatchWorkgroups(1); publish.end();
  }

  destroy(): void { if (this.destroyed) return; this.destroyed = true;
    this.results.destroy(); this.statuses.destroy(); this.control.destroy(); this.params.destroy(); }
}

export const octreePowerRedistanceShader = /* wgsl */ `
struct Params { queryCount:u32, queryCapacity:u32, generation:u32, pad:u32 }
struct Query { firstFace:u32, faceCount:u32, firstTetrahedron:u32, tetrahedronCount:u32, flags:u32, vertexStart:u32, vertexCount:u32, output:u32, data:vec4f }
struct TetraVertex { offsetSize:vec4f }
struct Control { flags:atomic<u32>, firstError:atomic<u32>, queryCount:u32, updated:atomic<u32>, tetrahedron:atomic<u32>, nearest:atomic<u32>, reserved:u32, generation:u32 }
@group(0) @binding(0) var<uniform> params:Params;@group(0) @binding(1) var<storage,read> queries:array<Query>;
@group(0) @binding(2) var<storage,read> magnitudes:array<f32>;@group(0) @binding(3) var<storage,read> vertices:array<TetraVertex>;
@group(0) @binding(4) var<storage,read> tetrahedra:array<u32>;@group(0) @binding(5) var<storage,read_write> results:array<f32>;
@group(0) @binding(6) var<storage,read_write> statuses:array<u32>;@group(0) @binding(7) var<storage,read_write> control:Control;
const VALID:u32=0x80000000u;const CAPACITY:u32=1u;const INVALID_QUERY:u32=2u;const NO_KNOWN:u32=4u;const UNIFORM:u32=1u;
fn finite(value:f32)->bool{return (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;}fn fail(index:u32,flag:u32){atomicOr(&control.flags,flag);atomicMin(&control.firstError,index);statuses[index]=flag;}
fn solveTranspose(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.0);}let value=(rhs.x*cross(b,c)+rhs.y*cross(c,a)+rhs.z*cross(a,b))/determinant;return vec4f(value,1.0);}
fn solveColumns(a:vec3f,b:vec3f,c:vec3f,rhs:vec3f)->vec4f{let determinant=dot(a,cross(b,c));if(!finite(determinant)||abs(determinant)<=1e-10){return vec4f(0.0);}return vec4f(dot(rhs,cross(b,c)),dot(a,cross(rhs,c)),dot(a,cross(b,rhs)),determinant);}
fn nonobtuseAnchorSolidAngle(a:vec3f,b:vec3f,c:vec3f,tolerance:f32)->bool{let denominator=length(a)*length(b)*length(c)+dot(a,b)*length(c)+dot(a,c)*length(b)+dot(b,c)*length(a);let numerator=abs(dot(a,cross(b,c)));return denominator+tolerance*max(1.,max(abs(denominator),numerator))>=numerator;}
@compute @workgroup_size(64) fn updatePowerDistance(@builtin(global_invocation_id) gid:vec3u){let index=gid.x;if(index>=params.queryCount||index>=params.queryCapacity){return;}if(params.queryCount>params.queryCapacity||index>=arrayLength(&queries)||index>=arrayLength(&results)||index>=arrayLength(&statuses)){return;}let query=queries[index];if(query.output>=arrayLength(&results)||query.output>=arrayLength(&statuses)||(query.flags&UNIFORM)!=0u||query.firstFace>arrayLength(&vertices)||query.faceCount>arrayLength(&vertices)-query.firstFace||query.firstTetrahedron>arrayLength(&tetrahedra)||query.tetrahedronCount>arrayLength(&tetrahedra)-query.firstTetrahedron||query.vertexStart>arrayLength(&magnitudes)||query.vertexCount>arrayLength(&magnitudes)-query.vertexStart||query.vertexCount<query.faceCount||!finite(query.data.x)||query.data.x<=0.0||abs(query.data.y)!=1.0){fail(index,INVALID_QUERY);return;}let tolerance=max(0.0,query.data.z);var best=1e30;var bestTetra=0xffffffffu;for(var local=0u;local<query.tetrahedronCount;local+=1u){let packed=tetrahedra[query.firstTetrahedron+local];let selectors=vec3u(packed&255u,(packed>>8u)&255u,(packed>>16u)&255u);if(any(selectors>=vec3u(query.faceCount))){fail(index,INVALID_QUERY);return;}let known=vec3f(magnitudes[query.vertexStart+selectors.x],magnitudes[query.vertexStart+selectors.y],magnitudes[query.vertexStart+selectors.z]);if(any(known<vec3f(0.0))||!finite(known.x)||!finite(known.y)||!finite(known.z)){continue;}let a=query.data.x*vertices[query.firstFace+selectors.x].offsetSize.xyz;let b=query.data.x*vertices[query.firstFace+selectors.y].offsetSize.xyz;let c=query.data.x*vertices[query.firstFace+selectors.z].offsetSize.xyz;if(!nonobtuseAnchorSolidAngle(a,b,c,tolerance)){continue;}let av=solveTranspose(a,b,c,known);let bv=solveTranspose(a,b,c,vec3f(1.0));if(av.w==0.0||bv.w==0.0){continue;}let quadratic=dot(bv.xyz,bv.xyz);let projection=dot(av.xyz,bv.xyz);let constant=dot(av.xyz,av.xyz)-1.0;let discriminant=projection*projection-quadratic*constant;if(quadratic<=0.0||discriminant < -tolerance){continue;}let candidate=(projection+sqrt(max(0.0,discriminant)))/quadratic;if(!finite(candidate)||candidate+tolerance<max(known.x,max(known.y,known.z))){continue;}let gradient=av.xyz-candidate*bv.xyz;let ray=solveColumns(a,b,c,-gradient);if(ray.w==0.0){continue;}let coefficients=ray.xyz/ray.w;let sum=coefficients.x+coefficients.y+coefficients.z;if(sum<=tolerance||any(coefficients/sum<vec3f(-tolerance))){continue;}if(candidate<best){best=candidate;bestTetra=local;}}
if(bestTetra==0xffffffffu){fail(index,NO_KNOWN);return;}results[query.output]=query.data.y*best;statuses[query.output]=VALID|bestTetra;atomicAdd(&control.tetrahedron,1u);atomicAdd(&control.updated,1u);}
@compute @workgroup_size(1) fn publishPowerDistance(){if(params.queryCount>params.queryCapacity||params.queryCount>arrayLength(&queries)||params.queryCount>arrayLength(&results)||params.queryCount>arrayLength(&statuses)){atomicOr(&control.flags,CAPACITY);atomicMin(&control.firstError,min(params.queryCount,params.queryCapacity));return;}if(atomicLoad(&control.flags)==0u&&atomicLoad(&control.updated)==params.queryCount){atomicStore(&control.flags,VALID);}}
`;
