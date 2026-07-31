import { pathToFileURL } from "node:url";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

export const DENSE_CACHE_MINI_DIMENSIONS = Object.freeze([
  [16, 16, 16], [8, 8, 8], [4, 4, 4], [2, 2, 2], [1, 1, 1],
] as const);
export const DENSE_CACHE_MINI_OCCUPIED_COUNTS = Object.freeze([1475, 214, 35, 8, 1] as const);
export const DENSE_CACHE_WORKGROUP_SIZES = Object.freeze([32, 64, 128, 256] as const);
export const DENSE_CACHE_LAYOUTS = Object.freeze([
  "xfast-full", "xfast-worklist", "tile4-physical", "tile4-logical", "stage8-logical",
] as const);
export const DENSE_CACHE_KERNELS = Object.freeze(["apply", "restrict", "prolong"] as const);

export type DenseCacheLayout = typeof DENSE_CACHE_LAYOUTS[number];
export type DenseCacheKernel = typeof DENSE_CACHE_KERNELS[number];
type Dimensions = readonly [number, number, number];

export interface DenseCacheFrozenMini {
  readonly dimensions: readonly Dimensions[];
  readonly logicalBases: readonly number[];
  readonly volumes: readonly number[];
  readonly occupiedCounts: readonly number[];
  readonly occupiedLocals: readonly (readonly number[])[];
  readonly workBases: readonly number[];
  readonly worklist: Uint32Array;
  readonly canonicalFlags: Uint32Array;
  readonly canonicalValues: Float32Array;
  readonly canonicalDiagonal: Float32Array;
  readonly totalLogicalSlots: number;
}

export interface DenseCachePhysicalPlan {
  readonly layout: DenseCacheLayout;
  readonly physicalBases: readonly number[];
  readonly physicalVolumes: readonly number[];
  readonly totalPhysicalSlots: number;
}

const volume = (dimensions: Dimensions) =>
  dimensions[0] * dimensions[1] * dimensions[2];

const coordinate = (local: number, dimensions: Dimensions): [number, number, number] => {
  const yz = Math.floor(local / dimensions[0]);
  return [local - yz * dimensions[0], yz % dimensions[1], Math.floor(yz / dimensions[1])];
};

const localIndex = (q: readonly [number, number, number], dimensions: Dimensions) =>
  q[0] + dimensions[0] * (q[1] + dimensions[1] * q[2]);

const tileCount = (dimensions: Dimensions, shape: Dimensions) =>
  Math.ceil(dimensions[0] / shape[0])
  * Math.ceil(dimensions[1] / shape[1])
  * Math.ceil(dimensions[2] / shape[2]);

/** Frozen, spatially coherent mini hierarchy with exact q/2 parent closure. */
export function createDenseCacheFrozenMini(): DenseCacheFrozenMini {
  const dimensions = DENSE_CACHE_MINI_DIMENSIONS.map((value) => [...value] as Dimensions);
  const volumes = dimensions.map(volume);
  const logicalBases: number[] = [];
  let totalLogicalSlots = 0;
  for (const size of volumes) {
    logicalBases.push(totalLogicalSlots);
    totalLogicalSlots += size;
  }

  const occupied = Array.from({ length: dimensions.length }, () => new Set<number>());
  occupied[4]!.add(0);
  for (let level = 3; level >= 0; level -= 1) {
    const fineDimensions = dimensions[level]!;
    const coarseDimensions = dimensions[level + 1]!;
    const candidates: number[] = [];
    for (const parentLocal of occupied[level + 1]!) {
      const parent = coordinate(parentLocal, coarseDimensions);
      for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) {
        for (let x = 0; x < 2; x += 1) {
          const child = [2 * parent[0] + x, 2 * parent[1] + y, 2 * parent[2] + z] as const;
          if (child.every((value, axis) => value < fineDimensions[axis])) {
            candidates.push(localIndex(child, fineDimensions));
          }
        }
      }
    }
    candidates.sort((left, right) => left - right);
    for (const local of candidates.slice(0, DENSE_CACHE_MINI_OCCUPIED_COUNTS[level])) {
      occupied[level]!.add(local);
    }
    if (occupied[level]!.size !== DENSE_CACHE_MINI_OCCUPIED_COUNTS[level]) {
      throw new Error(`Frozen mini level ${level} cannot satisfy its occupied count`);
    }
  }

  const occupiedLocals = occupied.map((entries) => [...entries].sort((a, b) => a - b));
  const workBases: number[] = [];
  const work: number[] = [];
  for (const locals of occupiedLocals) {
    workBases.push(work.length);
    work.push(...locals);
  }
  const canonicalFlags = new Uint32Array(totalLogicalSlots);
  const canonicalValues = new Float32Array(totalLogicalSlots);
  const canonicalDiagonal = new Float32Array(totalLogicalSlots);
  for (let level = 0; level < dimensions.length; level += 1) {
    const base = logicalBases[level]!;
    for (let local = 0; local < volumes[level]!; local += 1) {
      // Binary fractions and integral diagonals keep every test expression
      // exactly representable, independent of backend FMA contraction.
      canonicalValues[base + local] = ((13 * level + local) % 31 - 15) / 16;
      canonicalDiagonal[base + local] = 8 + (level & 1);
    }
    for (const local of occupiedLocals[level]!) canonicalFlags[base + local] = 1;
  }
  return Object.freeze({
    dimensions: Object.freeze(dimensions),
    logicalBases: Object.freeze(logicalBases),
    volumes: Object.freeze(volumes),
    occupiedCounts: DENSE_CACHE_MINI_OCCUPIED_COUNTS,
    occupiedLocals: Object.freeze(occupiedLocals.map((locals) =>
      Object.freeze(locals))) as readonly (readonly number[])[],
    workBases: Object.freeze(workBases),
    worklist: new Uint32Array(work),
    canonicalFlags,
    canonicalValues,
    canonicalDiagonal,
    totalLogicalSlots,
  });
}

export function denseCachePhysicalPlan(
  hierarchy: DenseCacheFrozenMini,
  layout: DenseCacheLayout,
): DenseCachePhysicalPlan {
  const shape: Dimensions | undefined = layout === "tile4-physical" ? [4, 4, 4] : undefined;
  const physicalBases: number[] = [];
  const physicalVolumes: number[] = [];
  let totalPhysicalSlots = 0;
  for (let level = 0; level < hierarchy.dimensions.length; level += 1) {
    physicalBases.push(totalPhysicalSlots);
    const slots = shape
      ? tileCount(hierarchy.dimensions[level]!, shape) * volume(shape)
      : hierarchy.volumes[level]!;
    physicalVolumes.push(slots);
    totalPhysicalSlots += slots;
  }
  return Object.freeze({
    layout, physicalBases: Object.freeze(physicalBases),
    physicalVolumes: Object.freeze(physicalVolumes), totalPhysicalSlots,
  });
}

export function denseCachePhysicalIndex(
  plan: DenseCachePhysicalPlan,
  dimensions: Dimensions,
  level: number,
  q: readonly [number, number, number],
): number {
  if (plan.layout !== "tile4-physical") {
    return plan.physicalBases[level]! + localIndex(q, dimensions);
  }
  const tx = Math.ceil(dimensions[0] / 4);
  const ty = Math.ceil(dimensions[1] / 4);
  const tile = Math.floor(q[0] / 4)
    + tx * (Math.floor(q[1] / 4) + ty * Math.floor(q[2] / 4));
  const within = q[0] % 4 + 4 * (q[1] % 4 + 4 * (q[2] % 4));
  return plan.physicalBases[level]! + 64 * tile + within;
}

function canonicalExpected(
  hierarchy: DenseCacheFrozenMini,
  kernel: DenseCacheKernel,
): Float32Array {
  const output = new Float32Array(hierarchy.totalLogicalSlots);
  const directions = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0],
    [0, -1, 0], [0, 0, 1], [0, 0, -1],
  ] as const;
  const levelBegin = kernel === "restrict" ? 1 : 0;
  const levelEnd = kernel === "prolong" ? 4 : 5;
  for (let level = levelBegin; level < levelEnd; level += 1) {
    const dimensions = hierarchy.dimensions[level]!;
    const base = hierarchy.logicalBases[level]!;
    for (const local of hierarchy.occupiedLocals[level]!) {
      const q = coordinate(local, dimensions);
      if (kernel === "apply") {
        let result = hierarchy.canonicalDiagonal[base + local]!
          * hierarchy.canonicalValues[base + local]!;
        for (const d of directions) {
          const n = [q[0] + d[0], q[1] + d[1], q[2] + d[2]] as const;
          if (n.some((value, axis) => value < 0 || value >= dimensions[axis])) continue;
          const neighbour = localIndex(n, dimensions);
          if (hierarchy.canonicalFlags[base + neighbour] !== 0) {
            result -= hierarchy.canonicalValues[base + neighbour]!;
          }
        }
        output[base + local] = result;
      } else if (kernel === "restrict") {
        const fineLevel = level - 1;
        const fineDimensions = hierarchy.dimensions[fineLevel]!;
        const fineBase = hierarchy.logicalBases[fineLevel]!;
        let result = 0;
        for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) {
          for (let x = 0; x < 2; x += 1) {
            const child = [2 * q[0] + x, 2 * q[1] + y, 2 * q[2] + z] as const;
            if (child.some((value, axis) => value >= fineDimensions[axis])) continue;
            const childLocal = localIndex(child, fineDimensions);
            if (hierarchy.canonicalFlags[fineBase + childLocal] !== 0) {
              result += hierarchy.canonicalValues[fineBase + childLocal]!;
            }
          }
        }
        output[base + local] = result;
      } else {
        const coarseLevel = level + 1;
        const coarseDimensions = hierarchy.dimensions[coarseLevel]!;
        const parent = [q[0] >> 1, q[1] >> 1, q[2] >> 1] as const;
        output[base + local] = hierarchy.canonicalValues[
          hierarchy.logicalBases[coarseLevel]! + localIndex(parent, coarseDimensions)
        ]!;
      }
    }
  }
  return output;
}

export function denseCacheCoordinateHash(values: Float32Array): string {
  const words = new Uint32Array(values.buffer, values.byteOffset, values.length);
  let hash = 0x811c9dc5;
  for (let index = 0; index < words.length; index += 1) {
    hash ^= index >>> 0; hash = Math.imul(hash, 0x01000193);
    hash ^= words[index]!; hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

interface LevelDispatch {
  readonly level: number;
  readonly params: Uint32Array;
  readonly workgroups: number;
}

function dispatches(
  hierarchy: DenseCacheFrozenMini,
  plan: DenseCachePhysicalPlan,
  layout: DenseCacheLayout,
  kernel: DenseCacheKernel,
  workgroupSize: number,
): readonly LevelDispatch[] {
  const first = kernel === "restrict" ? 1 : 0;
  const end = kernel === "prolong" ? 4 : 5;
  const records: LevelDispatch[] = [];
  for (let level = first; level < end; level += 1) {
    const d = hierarchy.dimensions[level]!;
    const otherLevel = kernel === "restrict" ? level - 1 : Math.min(4, level + 1);
    const other = hierarchy.dimensions[otherLevel]!;
    const itemCount = layout === "xfast-worklist"
      ? hierarchy.occupiedCounts[level]!
      : layout === "tile4-physical" || layout === "tile4-logical"
        ? tileCount(d, [4, 4, 4])
        : layout === "stage8-logical"
          ? tileCount(d, [8, 8, 4])
          : hierarchy.volumes[level]!;
    const workgroups = layout === "xfast-full" || layout === "xfast-worklist"
      ? Math.ceil(itemCount / workgroupSize) : itemCount;
    records.push({
      level,
      params: new Uint32Array([
        level, d[0], d[1], d[2],
        hierarchy.logicalBases[level]!, plan.physicalBases[level]!,
        hierarchy.workBases[level]!, hierarchy.occupiedCounts[level]!,
        other[0], other[1], other[2], plan.physicalBases[otherLevel]!,
        hierarchy.volumes[level]!, itemCount, 0, 0,
      ]),
      workgroups,
    });
  }
  return records;
}

function benchmarkShader(layout: DenseCacheLayout, workgroupSize: number): string {
  const physicalTiled4 = layout === "tile4-physical";
  const scheduled4 = physicalTiled4 || layout === "tile4-logical";
  const staged8 = layout === "stage8-logical";
  const worklist = layout === "xfast-worklist";
  const itemSpan = scheduled4 ? 64 : staged8 ? 256 : workgroupSize;
  const selector = scheduled4 ? `
    let td=(vec3u(p.nx,p.ny,p.nz)+vec3u(3u))/4u;
    let tile=wid.x;let tq=vec3u(tile%td.x,(tile/td.x)%td.y,tile/(td.x*td.y));
    q=tq*4u+vec3u(ordinal%4u,(ordinal/4u)%4u,ordinal/16u);
    if(all(q<vec3u(p.nx,p.ny,p.nz))){local=q.x+p.nx*(q.y+p.ny*q.z);}` : staged8 ? `
    let td=(vec3u(p.nx,p.ny,p.nz)+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);
    let tile=wid.x;let tq=vec3u(tile%td.x,(tile/td.x)%td.y,tile/(td.x*td.y));
    q=tq*vec3u(8u,8u,4u)+vec3u(ordinal%8u,(ordinal/8u)%8u,ordinal/64u);
    if(all(q<vec3u(p.nx,p.ny,p.nz))){local=q.x+p.nx*(q.y+p.ny*q.z);}` : worklist ? `
    let item=wid.x*${workgroupSize}u+ordinal;
    if(item<p.workCount){local=worklist[p.workBase+item];q=coord(local,vec3u(p.nx,p.ny,p.nz));}` : `
    let item=wid.x*${workgroupSize}u+ordinal;
    if(item<p.volume){local=item;q=coord(local,vec3u(p.nx,p.ny,p.nz));}`;
  const physical = physicalTiled4 ? `
fn at(q:vec3u,d:vec3u,base:u32)->u32{let td=(d+vec3u(3u))/4u;
 let tile=q/4u;let t=tile.x+td.x*(tile.y+td.y*tile.z);let w=q%4u;
 return base+64u*t+w.x+4u*(w.y+4u*w.z);}` : `
fn at(q:vec3u,d:vec3u,base:u32)->u32{return base+q.x+d.x*(q.y+d.y*q.z);}`;
  const declarations = worklist
    ? "@group(0)@binding(5)var<storage,read>worklist:array<u32>;" : "";
  const stagedApply = staged8 ? `
var<workgroup> stageValues:array<f32,600>;
var<workgroup> stageFlags:array<u32,600>;
fn stageIndex(q:vec3u)->u32{return q.x+10u*(q.y+10u*q.z);}
@compute @workgroup_size(${workgroupSize}) fn apply(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 let d=vec3u(p.nx,p.ny,p.nz);let td=(d+vec3u(7u,7u,3u))/vec3u(8u,8u,4u);
 let tile=wid.x;let tq=vec3u(tile%td.x,(tile/td.x)%td.y,tile/(td.x*td.y));
 let origin=vec3i(tq*vec3u(8u,8u,4u));
 for(var item=lane;item<600u;item+=${workgroupSize}u){let h=vec3u(item%10u,(item/10u)%10u,item/100u);
  let q=origin+vec3i(h)-vec3i(1);stageValues[item]=0.0;stageFlags[item]=0u;
  if(all(q>=vec3i(0))&&all(q<vec3i(d))){let s=at(vec3u(q),d,p.physicalBase);
   stageValues[item]=values[s];stageFlags[item]=flags[s];}}
 workgroupBarrier();
 for(var ordinal=lane;ordinal<256u;ordinal+=${workgroupSize}u){let l=vec3u(ordinal%8u,(ordinal/8u)%8u,ordinal/64u);
  let q=tq*vec3u(8u,8u,4u)+l;if(any(q>=d)){continue;}let h=stageIndex(l+vec3u(1u));if(stageFlags[h]==0u){continue;}
  let s=at(q,d,p.physicalBase);var result=diagonal[s]*stageValues[h];
  let offsets=array<i32,6>(-1,1,-10,10,-100,100);for(var k=0u;k<6u;k+=1u){
   let n=u32(i32(h)+offsets[k]);if(stageFlags[n]!=0u){result-=stageValues[n];}}
  output[s]=result;}}` : `
@compute @workgroup_size(${workgroupSize}) fn apply(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 for(var ordinal=lane;ordinal<${itemSpan}u;ordinal+=${workgroupSize}u){var local=INVALID;var q=vec3u(0u);${selector}
  if(local==INVALID){continue;}let d=vec3u(p.nx,p.ny,p.nz);let s=at(q,d,p.physicalBase);if(flags[s]==0u){continue;}
  var result=diagonal[s]*values[s];let directions=array<vec3i,6>(vec3i(1,0,0),vec3i(-1,0,0),
   vec3i(0,1,0),vec3i(0,-1,0),vec3i(0,0,1),vec3i(0,0,-1));
  for(var k=0u;k<6u;k+=1u){let n=vec3i(q)+directions[k];if(any(n<vec3i(0))||any(n>=vec3i(d))){continue;}
   let ns=at(vec3u(n),d,p.physicalBase);if(flags[ns]!=0u){result-=values[ns];}}
  output[s]=result;}}`;
  return `
const INVALID:u32=0xffffffffu;
struct Params{level:u32,nx:u32,ny:u32,nz:u32,logicalBase:u32,physicalBase:u32,
 workBase:u32,workCount:u32,otherNx:u32,otherNy:u32,otherNz:u32,otherPhysicalBase:u32,
 volume:u32,itemCount:u32,pad0:u32,pad1:u32}
@group(0)@binding(0)var<uniform>p:Params;
@group(0)@binding(1)var<storage,read>flags:array<u32>;
@group(0)@binding(2)var<storage,read>values:array<f32>;
@group(0)@binding(3)var<storage,read>diagonal:array<f32>;
@group(0)@binding(4)var<storage,read_write>output:array<f32>;
${declarations}
fn coord(local:u32,d:vec3u)->vec3u{return vec3u(local%d.x,(local/d.x)%d.y,local/(d.x*d.y));}
${physical}
${stagedApply}
@compute @workgroup_size(${workgroupSize}) fn restrictKernel(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 for(var ordinal=lane;ordinal<${itemSpan}u;ordinal+=${workgroupSize}u){var local=INVALID;var q=vec3u(0u);${selector}
  if(local==INVALID){continue;}let d=vec3u(p.nx,p.ny,p.nz);let s=at(q,d,p.physicalBase);if(flags[s]==0u){continue;}
  let fd=vec3u(p.otherNx,p.otherNy,p.otherNz);var result=0.0;
  for(var child=0u;child<8u;child+=1u){let cq=2u*q+vec3u(child&1u,(child>>1u)&1u,(child>>2u)&1u);
   if(any(cq>=fd)){continue;}let cs=at(cq,fd,p.otherPhysicalBase);if(flags[cs]!=0u){result+=values[cs];}}
  output[s]=result;}}
@compute @workgroup_size(${workgroupSize}) fn prolongKernel(
 @builtin(workgroup_id)wid:vec3u,@builtin(local_invocation_index)lane:u32){
 for(var ordinal=lane;ordinal<${itemSpan}u;ordinal+=${workgroupSize}u){var local=INVALID;var q=vec3u(0u);${selector}
  if(local==INVALID){continue;}let d=vec3u(p.nx,p.ny,p.nz);let s=at(q,d,p.physicalBase);if(flags[s]==0u){continue;}
  let cd=vec3u(p.otherNx,p.otherNy,p.otherNz);output[s]=values[at(q/2u,cd,p.otherPhysicalBase)];}}
@compute @workgroup_size(${workgroupSize}) fn empty(){}
`;
}

interface RuntimeVariant {
  readonly layout: DenseCacheLayout;
  readonly plan: DenseCachePhysicalPlan;
  readonly flags: GPUBuffer;
  readonly values: GPUBuffer;
  readonly diagonal: GPUBuffer;
  readonly output: GPUBuffer;
  readonly worklist?: GPUBuffer;
}

const median = (values: readonly number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)]!;
};

async function main(): Promise<void> {
  const repeats = Number(process.env.FLUID_DENSE_CACHE_REPEATS ?? 8);
  const samples = Number(process.env.FLUID_DENSE_CACHE_SAMPLES ?? 5);
  if (!Number.isSafeInteger(repeats) || repeats < 1
    || !Number.isSafeInteger(samples) || samples < 1) {
    throw new RangeError("Dense cache benchmark repeats and samples must be positive integers");
  }
  await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/benchmark-dense-hierarchy-cache.ts");
  try {
    const modulePath = process.env.WEBGPU_NODE_MODULE
      ?? `${process.cwd()}/node_modules/webgpu/index.js`;
    const dawn = await import(pathToFileURL(modulePath).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const backend = process.env.WEBGPU_BACKEND ?? "metal";
    const gpu = dawn.create([`backend=${backend}`]);
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter?.features.has("timestamp-query")) {
      throw new Error("Dense hierarchy cache benchmark requires timestamp-query");
    }
    const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
    const hierarchy = createDenseCacheFrozenMini();
    const makeStorage = (label: string, size: number, readback = false) => device.createBuffer({
      label, size: Math.max(4, size), usage: readback
        ? GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    const variants: RuntimeVariant[] = [];
    for (const layout of DENSE_CACHE_LAYOUTS) {
      const plan = denseCachePhysicalPlan(hierarchy, layout);
      const packedFlags = new Uint32Array(plan.totalPhysicalSlots);
      const packedValues = new Float32Array(plan.totalPhysicalSlots);
      const packedDiagonal = new Float32Array(plan.totalPhysicalSlots);
      for (let level = 0; level < hierarchy.dimensions.length; level += 1) {
        const d = hierarchy.dimensions[level]!;
        const logicalBase = hierarchy.logicalBases[level]!;
        for (let local = 0; local < hierarchy.volumes[level]!; local += 1) {
          const physical = denseCachePhysicalIndex(plan, d, level, coordinate(local, d));
          packedFlags[physical] = hierarchy.canonicalFlags[logicalBase + local]!;
          packedValues[physical] = hierarchy.canonicalValues[logicalBase + local]!;
          packedDiagonal[physical] = hierarchy.canonicalDiagonal[logicalBase + local]!;
        }
      }
      const flags = makeStorage(`${layout} flags`, packedFlags.byteLength);
      const values = makeStorage(`${layout} values`, packedValues.byteLength);
      const diagonal = makeStorage(`${layout} diagonal`, packedDiagonal.byteLength);
      const output = makeStorage(`${layout} output`, plan.totalPhysicalSlots * 4);
      device.queue.writeBuffer(flags, 0, packedFlags);
      device.queue.writeBuffer(values, 0, packedValues);
      device.queue.writeBuffer(diagonal, 0, packedDiagonal);
      let worklist: GPUBuffer | undefined;
      if (layout === "xfast-worklist") {
        worklist = makeStorage("Dense cache sorted occupied worklist", hierarchy.worklist.byteLength);
        device.queue.writeBuffer(worklist, 0,
          hierarchy.worklist as Uint32Array<ArrayBuffer>);
      }
      variants.push({ layout, plan, flags, values, diagonal, output, worklist });
    }

    const querySet = device.createQuerySet({ type: "timestamp", count: 2 });
    const resolve = device.createBuffer({
      size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const timestampRead = makeStorage("Dense cache timestamp readback", 16, true);
    const results: Array<Record<string, unknown>> = [];
    const expected = Object.fromEntries(DENSE_CACHE_KERNELS.map((kernel) => [
      kernel, canonicalExpected(hierarchy, kernel),
    ])) as Record<DenseCacheKernel, Float32Array>;
    const expectedHashes = Object.fromEntries(DENSE_CACHE_KERNELS.map((kernel) => [
      kernel, denseCacheCoordinateHash(expected[kernel]),
    ]));

    for (const workgroupSize of DENSE_CACHE_WORKGROUP_SIZES) for (const runtime of variants) {
      const module = device.createShaderModule({
        label: `${runtime.layout} wg${workgroupSize}`,
        code: benchmarkShader(runtime.layout, workgroupSize),
      });
      const compilation = await module.getCompilationInfo();
      const errors = compilation.messages.filter((message) => message.type === "error");
      if (errors.length > 0) {
        throw new Error(`${runtime.layout} wg${workgroupSize} shader: `
          + errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join(" | "));
      }
      const pipelines = Object.fromEntries(await Promise.all(
        [...DENSE_CACHE_KERNELS, "empty"].map(async (entryPoint) => [entryPoint,
          await device.createComputePipelineAsync({
            label: `${runtime.layout} ${entryPoint} wg${workgroupSize}`,
            layout: "auto", compute: {
              module,
              entryPoint: entryPoint === "restrict" ? "restrictKernel"
                : entryPoint === "prolong" ? "prolongKernel"
                  : entryPoint,
            },
          })]),
      )) as Record<DenseCacheKernel | "empty", GPUComputePipeline>;

      for (const kernel of DENSE_CACHE_KERNELS) {
        const levelDispatches = dispatches(
          hierarchy, runtime.plan, runtime.layout, kernel, workgroupSize,
        );
        const parameterBuffers: GPUBuffer[] = [];
        const groups: GPUBindGroup[] = [];
        for (const record of levelDispatches) {
          const params = device.createBuffer({
            label: `${runtime.layout} ${kernel} L${record.level} params`,
            size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(params, 0, record.params as Uint32Array<ArrayBuffer>);
          parameterBuffers.push(params);
          const entries: GPUBindGroupEntry[] = [
            { binding: 0, resource: { buffer: params } },
            { binding: 1, resource: { buffer: runtime.flags } },
            { binding: 2, resource: { buffer: runtime.values } },
            { binding: 4, resource: { buffer: runtime.output } },
          ];
          if (kernel === "apply") {
            entries.push({ binding: 3, resource: { buffer: runtime.diagonal } });
          }
          if (runtime.worklist) entries.push({ binding: 5, resource: { buffer: runtime.worklist } });
          groups.push(device.createBindGroup({
            layout: pipelines[kernel].getBindGroupLayout(0), entries,
          }));
        }

        const run = async (empty: boolean, iterations: number, readOutput: boolean) => {
          const encoder = device.createCommandEncoder();
          if (readOutput) encoder.clearBuffer(runtime.output);
          const pass = encoder.beginComputePass({
            timestampWrites: {
              querySet,
              beginningOfPassWriteIndex: 0,
              endOfPassWriteIndex: 1,
            },
          });
          pass.setPipeline(empty ? pipelines.empty : pipelines[kernel]);
          for (let repeat = 0; repeat < iterations; repeat += 1) {
            for (let index = 0; index < levelDispatches.length; index += 1) {
              if (!empty) pass.setBindGroup(0, groups[index]!);
              pass.dispatchWorkgroups(levelDispatches[index]!.workgroups);
            }
          }
          pass.end();
          encoder.resolveQuerySet(querySet, 0, 2, resolve, 0);
          encoder.copyBufferToBuffer(resolve, 0, timestampRead, 0, 16);
          let outputRead: GPUBuffer | undefined;
          if (readOutput) {
            outputRead = makeStorage("Dense cache output readback", runtime.output.size, true);
            encoder.copyBufferToBuffer(runtime.output, 0, outputRead, 0, runtime.output.size);
          }
          device.queue.submit([encoder.finish()]);
          await timestampRead.mapAsync(GPUMapMode.READ);
          const times = new BigUint64Array(timestampRead.getMappedRange().slice(0));
          timestampRead.unmap();
          const elapsed = Number(times[1]! - times[0]!) / 1e6 / iterations;
          let physicalOutput: Float32Array | undefined;
          if (outputRead) {
            await outputRead.mapAsync(GPUMapMode.READ);
            physicalOutput = new Float32Array(outputRead.getMappedRange().slice(0));
            outputRead.unmap(); outputRead.destroy();
          }
          return { elapsed, physicalOutput };
        };
        // Large single command buffers trigger a native Dawn/Metal failure for this
        // dispatch-heavy matrix. Preserve the requested sample population while
        // accumulating timestamp-only chunks below that backend limit.
        const runBatch = async (empty: boolean, iterations: number) => {
          let weightedElapsed = 0;
          for (let remaining = iterations; remaining > 0;) {
            const chunk = Math.min(8, remaining);
            weightedElapsed += (await run(empty, chunk, false)).elapsed * chunk;
            remaining -= chunk;
          }
          return weightedElapsed / iterations;
        };

        const validation = await run(false, 1, true);
        const canonical = new Float32Array(hierarchy.totalLogicalSlots);
        for (let level = 0; level < hierarchy.dimensions.length; level += 1) {
          const d = hierarchy.dimensions[level]!;
          const logicalBase = hierarchy.logicalBases[level]!;
          for (let local = 0; local < hierarchy.volumes[level]!; local += 1) {
            canonical[logicalBase + local] = validation.physicalOutput![
              denseCachePhysicalIndex(runtime.plan, d, level, coordinate(local, d))
            ]!;
          }
        }
        const actualWords = new Uint32Array(canonical.buffer);
        const expectedWords = new Uint32Array(expected[kernel].buffer);
        for (let index = 0; index < actualWords.length; index += 1) {
          if (actualWords[index] !== expectedWords[index]) {
            throw new Error(`${runtime.layout} ${kernel} wg${workgroupSize} differs at canonical slot ${index}: `
              + `0x${actualWords[index]!.toString(16)} != 0x${expectedWords[index]!.toString(16)}`);
          }
        }
        const actualHash = denseCacheCoordinateHash(canonical);
        await runBatch(false, repeats); await runBatch(true, repeats);
        const kernelSamples: number[] = [], emptySamples: number[] = [];
        for (let sample = 0; sample < samples; sample += 1) {
          if ((sample & 1) === 0) {
            kernelSamples.push(await runBatch(false, repeats));
            emptySamples.push(await runBatch(true, repeats));
          } else {
            emptySamples.push(await runBatch(true, repeats));
            kernelSamples.push(await runBatch(false, repeats));
          }
        }
        const gross = median(kernelSamples), overhead = median(emptySamples);
        results.push({
          layout: runtime.layout, kernel, workgroupSize,
          levels: levelDispatches.length,
          workgroupsPerApplication: levelDispatches.reduce((sum, record) => sum + record.workgroups, 0),
          grossMicroseconds: gross * 1000,
          emptyMicroseconds: overhead * 1000,
          netMicroseconds: (gross - overhead) * 1000,
          timingResolved: gross > overhead,
          coordinateHash: actualHash,
          bitExact: actualHash === expectedHashes[kernel],
        });
        parameterBuffers.forEach((buffer) => buffer.destroy());
      }
    }

    const winners = Object.fromEntries(DENSE_CACHE_KERNELS.map((kernel) => {
      const candidates = results.filter((result) => result.kernel === kernel)
        .filter((result) => result.timingResolved)
        .sort((left, right) => Number(left.netMicroseconds) - Number(right.netMicroseconds));
      return [kernel, candidates[0] ?? { unresolved: true }];
    }));
    console.log(JSON.stringify({
      benchmark: "factor1-dense-hierarchy-cache",
      backend,
      adapter: Object.fromEntries(Object.entries(adapter.info)),
      configuration: {
        dimensions: hierarchy.dimensions,
        occupiedCounts: hierarchy.occupiedCounts,
        totalLogicalSlots: hierarchy.totalLogicalSlots,
        repeats, samples,
        methodology: "median timestamp batch per application minus matched empty-dispatch batch; "
          + "batches chunked to at most 8 applications for Dawn/Metal stability",
      },
      physicalSlots: Object.fromEntries(variants.map((variant) =>
        [variant.layout, variant.plan.totalPhysicalSlots])),
      expectedCoordinateHashes: expectedHashes,
      results,
      winners,
    }, null, 2));

    for (const variant of variants) {
      variant.flags.destroy(); variant.values.destroy(); variant.diagonal.destroy();
      variant.output.destroy(); variant.worklist?.destroy();
    }
    timestampRead.destroy(); resolve.destroy(); querySet.destroy(); device.destroy();
  } finally {
    await releaseWebGPUExclusiveLock();
  }
}

const invokedPath = process.argv[1];
if (invokedPath && import.meta.url === pathToFileURL(invokedPath).href) await main();
