import type { WebGPUUniformPressureMultigrid } from "./webgpu-uniform-pressure-multigrid";
import { uniformPressureHierarchyWGSL } from "./webgpu-uniform-pressure-multigrid.wgsl";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformBufferedWork } from "./uniform-buffered-work";
import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { UNIFORM_PAGE_TABLE_WORDS, uniformBlockFaceWGSL } from "./uniform-lattice-address";
import { mixedCellWidth } from "./uniform-mixed-layout";
import { uniformMixedPressureStorage } from "./uniform-mixed-pressure-boundary.wgsl";
import { uniformMixedPressureLiquidWGSL, uniformMixedSurfaceThetaWGSL } from "./uniform-mixed-pressure-surface.wgsl";
import type { UniformMixedPressureTopology } from "./uniform-mixed-pressure-topology.wgsl";
import { UNIFORM_MIXED_ROOT_FAR_CELLS } from "./uniform-mixed-root-owners";
import { UNIFORM_PRESSURE_PAGE_LEVELS } from "./pressure-plan";

type Continuation = ReturnType<WebGPUUniformPressureMultigrid["prepareMixedContinuation"]>;

/** One smoothing visit of the root solve: the weights of its simultaneous
 * (Jacobi) iterations, in order. Seven undamped iterations carry the
 * smoothing and two half-weight ones remove the alternating-sign error
 * undamped Jacobi only flips. Every level below the mixed operator runs it
 * before and after its coarse correction. */
export const UNIFORM_MIXED_ROOT_SMOOTHING: readonly number[] = [1, 1, 1, 1, 1, 1, 1, 0.5, 0.5];
/** Iterations of a visit one tiled launch carries (its core's halo is this
 * wide, and the kernel's workgroup arrays are sized by it). */
const ROOT_LAUNCH_ITERATIONS = 3;
/** Lanes of a block's smoothing workgroup below the root lattice: a launch
 * there waits for a lane's chain of loads, not for its total work. The root
 * lattice's launches are bound by their work (and, in a closed slot, by the
 * threads they start), so its workgroups are a lane to an owned cell. */
const ROOT_TILE_LANES = 256;
/** The coarsest level's solve: this many red-black projected Gauss-Seidel
 * sweeps at this over-relaxation. Nothing lies below that level, so the
 * sweeps are all that carries a correction across it. The relaxation is the
 * one that needs the fewest sweeps on a level six to eight cells deep: the
 * launch's time is its sweeps, and more of them at a lower relaxation buy the
 * same reduction. */
const ROOT_PAGE_SWEEPS = 16;
const ROOT_PAGE_RELAXATION = 1.6;
/** Lanes of the one workgroup that runs the small levels: a lane to a row of
 * every level it visits; they stride the rows of the coarsest. */
const ROOT_COARSE_LANES = 1024;
/** The root's levels: 4h, 8h, 16h and 32h, the last a cell to a page and the
 * coarsest. Their word offsets belong to the solve's data, never its
 * compiled shader. */
const ROOT_LEVELS = UNIFORM_PRESSURE_PAGE_LEVELS;
const ROOT_CONFIG_WORDS = 16 + 12 * ROOT_LEVELS;
const ROOT_LAUNCH_KEYS = ["UM_RHS", "UM_MIN", "UM_FULL", "UM_BL", "UM_BRHS", "UM_BSRC", "UM_BFIRST", "UM_BCOUNT", "UM_BEND",
  "UM_CRHS", "UM_CFULL", "UM_TL", "UM_TSRC", "UM_TRHS", "UM_TFULL", "UM_PL", "UM_PSRC", "UM_PTO", "UM_PCOARSE", "UM_PADD", "UM_SL"] as const;
const ROOT_LAUNCH_BYTES = 4 * ROOT_LAUNCH_KEYS.length;
/** A level of more rows than this is run a block to a workgroup; the rest
 * share the one-workgroup launch (a workgroup is one core's worth of the
 * GPU, so it only pays where a launch's fixed cost exceeds the level's work). */
const ROOT_SINGLE_MAX_ROWS = 1000;
/** Threads a capacity slot's tiled launch starts at most. A slot the plan
 * holds beyond the cycles it expects is closed in all but a few frames, and a
 * closed launch costs its fixed 5 microseconds plus the threads it starts
 * (each reads the gate and returns); a launch this narrow still completes
 * when the slot does open, since every workgroup strides the jobs. Narrower
 * buys nothing closed and costs the open slot its parallelism (pool 128^3,
 * a V slot: +0.16 ms at 16384, +1.5 at 4096, +6.5 at 1024). */
const ROOT_CAPACITY_THREADS = 16384;

/** The all-4h pressure root: owner-indexed fields (owner index = tile key,
 * then a wall slot a tile of each block face) that RHS assembly, projection
 * and the band read and write. */
export interface UniformMixedPressureCycleLevel {
  ownership: UniformMixedOwnership;
  pressure: GPUBufferBinding;
  /** The RHS, then a Full-Cycle's correction RHS. */
  rhs: readonly [GPUBufferBinding, GPUBufferBinding];
  frozen: GPUBufferBinding;
  residual: GPUBufferBinding;
  /** The bound, then a Full-Cycle's shifted bound. */
  minimum: readonly [GPUBufferBinding, GPUBufferBinding];
  phi: GPUBufferBinding;
  /** Static solids: the all-4h (open, V+) record. */
  topology?: UniformMixedPressureTopology;
}

/** One compute pass shared by consecutive mixed pressure dispatches. A
 * dispatch is its own WebGPU usage scope, so dependent dispatches may share a
 * pass; encoder commands (copies, clears, foreign passes) end it first.
 * One instance serves every frame's batch of its label (begin, then finish):
 * the encode path constructs nothing, the pass descriptor included. */
export class UniformMixedPressurePasses {
  private open?: GPUComputePassEncoder;
  private readonly descriptor: GPUComputePassDescriptor;
  constructor(private encoder: GPUCommandEncoder | undefined, readonly label: string) { this.descriptor = { label }; }
  /** The next batch, on `encoder` (a pass an abandoned batch left open is
   * that encoder's, and is dropped with it). */
  begin(encoder: GPUCommandEncoder): this { this.open = undefined; this.encoder = encoder; return this; }
  get pass(): GPUComputePassEncoder {
    if (this.open) return this.open;
    if (!this.encoder) throw new Error(`${this.label}: no batch is open`);
    return this.open = this.encoder.beginComputePass(this.descriptor);
  }
  /** The encoder with no pass open. */
  get commands(): GPUCommandEncoder {
    this.end(); if (!this.encoder) throw new Error(`${this.label}: no batch is open`);
    return this.encoder;
  }
  end(): void { if (this.open) { this.open.end(); this.open = undefined; } }
  /** Ends the batch; the encoder is not kept. */
  finish(): void { this.end(); this.encoder = undefined; }
}

/** A launch: its fixed workgroup grid, or (tiled) its job count, its
 * workgroup's lanes and whether its jobs are the cycle list's. offsets: its
 * record's dynamic offset, as setBindGroup takes it. */
interface RootLaunch { pipeline: GPUComputePipeline; groups?: readonly [number, number, number]; tiles: number; lanes: number; listed: boolean; offset: number; offsets: number[] }
/** Every traversal's launches in encode order. */
interface RootTraversals { setup: readonly RootLaunch[]; measure: readonly RootLaunch[]; vCycle: readonly RootLaunch[]; fullCycle: readonly RootLaunch[] }

/** How a root-lattice smoothing launch stores its last iterate: in the other
 * parity of the level, or straight into the mixed pressure (added, assigned,
 * or added with the Full-Cycle backup restored). */
type RootEnd = 0 | 1 | 2 | 3;

/** CM11a cycles of the all-4h pressure root, whose correction is solved on
 * four levels below it: 4h, 8h, 16h and 32h. The mixed operator stays
 * authoritative: every residual, the measure and the correction bounds are
 * the mixed rows (umPressureTheta's ghost-fluid surface rule, wall slots as
 * unknowns), which the levels' baked operator does not reproduce; the levels
 * are the correction's solver.
 *
 * Every level is rows of blocks, as the root is (UniformPressureLevelWords):
 * a level's cells partition a page (8, 4, 2 and 1 an axis), a block's cells
 * are its rows wherever the pool puts its page, and a wall slot follows them
 * for each cell of each block face. A row's neighbour across a face of its
 * block is what the block neighbour table names there: a stored block's row;
 * far air where the page is AIR (no row: its phi is the far air's, its
 * fractions 1, its pressure zero); the face's wall slot where it is SOLID. No
 * kernel reads the lattice box, so a block held outside it is solved as any
 * other and a page table's AIR and SOLID mean the same on every level.
 * Level 0's row is the mixed row itself (the tile key, or the wall slot).
 *
 * The setup (level 0's geometry here, then the continuation's pyramid, baked
 * coefficients and cycle list, uniformPressureHierarchyWGSL) runs once a
 * solve, a lane to a row over the whole pool. A row's baked mask says
 * whether it is a row of the solve at all: a block that holds no page has
 * none, so a lane on one returns after that load.
 *
 * The traversal is in few launches because these levels are small and a
 * launch costs more than its work on most of them. A level of more than
 * ROOT_SINGLE_MAX_ROWS rows is tiled: a smoothing visit is
 * ROOT_LAUNCH_ITERATIONS Jacobi iterations to a launch (a workgroup takes a
 * core of cells, an octant of a block on level 0 and a block below, loads it
 * and a halo as wide as its iteration count into workgroup memory and
 * relaxes a region that shrinks a cell an iteration, so the stored values
 * are those of one launch per iteration), and its descent and ascent are a
 * launch each. The levels below share one launch of one workgroup (their
 * visits, descents, the coarsest solve and ascents, with a storage barrier
 * wherever a pass boundary was); a visit there is a lane to a row, which
 * keeps the row's coefficients and passes its iterates through workgroup
 * memory, so only the visit's last iterate is stored. The 32h level is
 * solved there by ROOT_PAGE_SWEEPS red-black projected Gauss-Seidel sweeps
 * in workgroup memory, its lanes striding its rows (see umLSolve): seven rows
 * a block, so the pool is bounded by what one workgroup's memory holds, and
 * a larger one is refused at construction. The last launch of a traversal
 * adds its iterate to the mixed pressure itself.
 *
 * A tiled launch's jobs are fixed by the pool's capacity (level 0's listed
 * ones), and workgroup w of a launch of W runs jobs w, w+W, ...: the width
 * is the host's parallelism estimate (uniformBufferedWork of the list count
 * two frames old, see observeWork), never a bound on the work. A capacity
 * slot launches narrower still (ROOT_CAPACITY_THREADS).
 *
 * A tiled level's halo holds a cell to a place, and a wall slot stands at
 * the place across its wall face. Two wall slots would share a place along
 * the edge of a SOLID page with stored pages across faces of two axes: the
 * root's fatal (UNIFORM_MIXED_ROOT_UNREACHED) is latched for exactly that.
 *
 * One kernel computes the residual straight into level 0's RHS with the
 * correction bound and a zero correction; the last smoothing launch adds (or
 * assigns) the correction back. The pressure layout is all-4h with no seams,
 * so every owner is a regular row and reconstruction slopes are zero. The
 * pressure fields are borrowed; low rounding words and the cycle list are
 * owned here. Convergence validation and publication remain the frame's job. */
export class UniformMixedPressureCycles {
  readonly rounding:GPUBuffer;
  private readonly configuration: GPUBuffer;
  private launches?: GPUBuffer;
  /** The count (word 0), the listed jobs from word 4, then a class word a job. */
  private readonly list: GPUBuffer;
  get allocatedBytes():number{return this.rounding.size + this.configuration.size + (this.launches?.size ?? 0) + this.list.size;}
  destroy():void{this.rounding.destroy();this.configuration.destroy();this.launches?.destroy();this.list.destroy();}
  private readonly resources: GPUBindGroupLayout;
  private group!: GPUBindGroup;
  private readonly resourceEntries: GPUBindGroupEntry[];
  private readonly pipelines = new Map<string, RootLaunch>();
  /** Resolved by initialize(): the sequences are fixed from there on
   * (cycle() and opening() run once), and a frame replays them. */
  private traversals?: RootTraversals;
  private readonly setupPasses = new UniformMixedPressurePasses(undefined, "Uniform mixed pressure setup");
  private readonly measurePasses = new UniformMixedPressurePasses(undefined, "Uniform mixed pressure measure");
  private readonly vCyclePasses = new UniformMixedPressurePasses(undefined, "Uniform mixed pressure V-cycle");
  private readonly fullCyclePasses = new UniformMixedPressurePasses(undefined, "Uniform mixed pressure Full-cycle");
  /** Levels run a block to a workgroup (the leading ones); the rest are the
   * one-workgroup launch's. */
  private readonly tiled: number;
  /** Each level's rows over the whole pool (cells, then wall slots) and its cells. */
  private readonly levelRows: readonly number[];
  private readonly levelCells: readonly number[];
  /** Rows of the one-workgroup launch's iterate (workgroup memory): the
   * coarsest level holds no more. */
  private readonly rows: number;
  /** A visit's launches: the iterations each carries. */
  private readonly visit: { first: number; count: number }[];
  private readonly tileLanes: number;
  private boundPass?: GPUComputePassEncoder;
  private readonly lanes: number;
  private readonly blocks: number;
  /** Level 0's jobs (an octant of each block): the listed launches' widest. */
  private readonly rootJobs: number;
  private readonly maxWorkgroups: number;
  /** The root lattice's kernels walk the listed jobs. */
  private readonly listed: boolean;
  /** The listed launches' width (see observeWork); every job without one. */
  private work?: number;
  private readonly words: Record<"p" | "b0" | "b1" | "m0" | "m1" | "backup" | "residual", number>;
  private readonly solid: boolean;
  constructor(private readonly device: GPUDevice, readonly level: UniformMixedPressureCycleLevel, backup: GPUBufferBinding, native: Continuation) {
    const layout = level.ownership.layout, address = layout.address, blocks = address.blocks;
    // A row a tile key: an owner's index is its tile's key.
    if (layout.tiles.some(word => mixedCellWidth(word) !== 4) || layout.tiles.length !== address.tileKeys || layout.cellCount !== address.tileKeys)
      throw new Error("Mixed pressure root requires the uniform all-4h layout");
    const count = uniformMixedPressureStorage(layout).count;
    const levels = native.levels, arena = native.arena;
    if (levels.length !== ROOT_LEVELS || levels.some((l, i) => l.side !== 8 >> i || l.cells !== l.side ** 3 * blocks || l.slots !== 6 * l.side ** 2 * blocks)
      || levels[0]!.cells + levels[0]!.slots !== count)
      throw new Error("Mixed pressure root: the correction's levels are not rows of this pool's blocks");
    const mixed = [level.pressure, level.rhs[0], level.rhs[1], level.minimum[0], level.minimum[1], backup, level.residual];
    if (mixed.some(f => f.buffer !== arena || (f.size ?? f.buffer.size - (f.offset ?? 0)) < 4 * count)) throw new Error("Mixed pressure root fields must be arena views of the live words");
    // Two disjoint arena ranges: binding either whole would alias the other.
    const mixedOffset = Math.floor(Math.min(...mixed.map(f => f.offset ?? 0)) / 256) * 256;
    const mixedRange = { offset: mixedOffset, size: Math.max(...mixed.map(f => (f.offset ?? 0) + 4 * count)) - mixedOffset };
    const last = ROOT_LEVELS - 1;
    const nativeRange = { offset: 4 * levels[0]!.pressure[0], size: 4 * (levels[last]!.wall + levels[last]!.slots - levels[0]!.pressure[0]) };
    if (nativeRange.offset % 256 !== 0 || levels.some(l => [...l.pressure, ...l.rhs, ...l.minimum, l.coefficients, ...l.phi, l.topology, l.wall].some(w => w < levels[0]!.pressure[0])))
      throw new Error("Mixed pressure root: the correction's levels are not one aligned arena range");
    if (mixedRange.offset < nativeRange.offset + nativeRange.size && nativeRange.offset < mixedRange.offset + mixedRange.size)
      throw new Error("Mixed pressure root and its correction's levels overlap");
    this.blocks = blocks; this.rootJobs = 8 * blocks;
    this.levelCells = levels.map(l => l.cells); this.levelRows = levels.map(l => l.cells + l.slots);
    this.tileLanes = Math.min(ROOT_TILE_LANES, device.limits.maxComputeInvocationsPerWorkgroup);
    const smoothing = UNIFORM_MIXED_ROOT_SMOOTHING, M = ROOT_LAUNCH_ITERATIONS;
    if (!smoothing.length || smoothing.some(w => !(w > 0 && w <= 1))) throw new Error("Mixed pressure root smoothing weights must lie in (0, 1]");
    this.rounding=device.createBuffer({label:"Uniform root pressure rounding and Full-Cycle backup",size:8*(count+1),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST});
    this.visit = Array.from({ length: Math.ceil(smoothing.length / M) }, (_, k) => ({
      first: k * M, count: Math.min(M, smoothing.length - k * M) }));
    this.lanes = Math.min(ROOT_COARSE_LANES, device.limits.maxComputeInvocationsPerWorkgroup);
    // A one-workgroup level above the coarsest is a lane to a row (see
    // umLSmooth); the coarsest is bounded by the iterate alone (see umLSolve),
    // which is all the workgroup memory the device holds beside the launch's
    // one scalar (Tint pads each variable to 16 bytes).
    const rows = this.levelRows;
    const single = rows.findIndex(n => n <= Math.min(ROOT_SINGLE_MAX_ROWS, this.lanes));
    this.tiled = Math.max(1, single < 0 ? last : single);
    this.rows = 4 * Math.floor((device.limits.maxComputeWorkgroupStorageSize - 16) / 16);
    const visits = Math.max(0, ...rows.slice(this.tiled, last));
    if (visits > this.lanes) throw new Error(`Mixed pressure root: a ${visits}-row level does not fit one workgroup of ${this.lanes} lanes`);
    // A visit keeps two iterates of ROOT_COARSE_LANES rows there.
    if (this.rows < 2 * ROOT_COARSE_LANES) throw new Error(`Mixed pressure root: one workgroup's memory holds ${this.rows} rows, fewer than the ${2 * ROOT_COARSE_LANES} a visit's two iterates take`);
    const perBlock = rows[last]! / blocks;
    if (rows[last]! > this.rows)
      throw new Error(`Mixed pressure root: a pool of ${blocks} blocks has ${rows[last]} rows on its coarsest level (${perBlock} a block), and one workgroup's memory holds ${this.rows}: at most ${Math.floor(this.rows / perBlock)} blocks`);
    const [p, b0, b1, m0, m1, backupWord, residual] = mixed.map(f => ((f.offset ?? 0) - mixedRange.offset) / 4) as number[];
    this.words = { p: p!, b0: b0!, b1: b1!, m0: m0!, m1: m1!, backup: backupWord!, residual: residual! };
    const phi: GPUBufferBinding = { buffer: level.phi.buffer, offset: level.phi.offset ?? 0, size: 4 * layout.cellCount };
    // The static solid record, read in place from its own binding.
    const topology = level.topology?.buffer;
    if (topology && (topology.size ?? topology.buffer.size - (topology.offset ?? 0)) < 16 * count)
      throw new Error("Mixed pressure root topology is smaller than its owners");
    this.solid = !!topology;
    this.maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;
    this.listed = native.listed;
    this.list = device.createBuffer({ label: "Uniform root pressure job list", size: 4 * (4 + 2 * this.rootJobs),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.configuration = device.createBuffer({ label: "Uniform root pressure hierarchy", size: 4 * ROOT_CONFIG_WORDS,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const config = new Uint32Array(ROOT_CONFIG_WORDS), scalar = new Float32Array(config.buffer);
    // h.w: the phi of a page nothing stores, as the root phi's block 0 rows hold it.
    scalar.set(layout.lattice.cellSize_m, 0); scalar[3] = UNIFORM_MIXED_ROOT_FAR_CELLS * Math.max(...layout.lattice.cellSize_m);
    // flags.x: the tile key count, which places the ownership's support words.
    config.set([layout.tiles.length, this.tiled, 0, this.listed ? 1 : 0], 4);
    config.set([this.words.p, this.words.b0, this.words.b1, this.words.m0], 8);
    config.set([this.words.m1, this.words.backup, this.words.residual, 0], 12);
    const nativeWord = (word: number) => word - nativeRange.offset / 4;
    levels.forEach((l, i) => {
      const at = 16 + 12 * i;
      config.set([...l.pressure, ...l.rhs].map(nativeWord), at);
      config.set([...l.minimum, l.coefficients, l.wall].map(nativeWord), at + 4);
      config.set([...l.phi, l.topology].map(nativeWord), at + 8);
    });
    device.queue.writeBuffer(this.configuration, 0, config);
    this.resources = device.createBindGroupLayout({ entries: [
      ...[0, 1, 2, 3, ...(topology ? [4] : []), 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 4 ? "read-only-storage" as const : "storage" as const } })),
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", minBindingSize: this.configuration.size } },
      { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: ROOT_LAUNCH_BYTES } },
    ] });
    this.resourceEntries = [
      { binding: 0, resource: { buffer: arena, ...mixedRange } }, { binding: 1, resource: { buffer: arena, ...nativeRange } }, { binding: 2, resource: phi },
      { binding: 3, resource: { buffer: this.list } },
      ...(topology ? [{ binding: 4, resource: topology }] : []), { binding: 5, resource: { buffer: this.rounding } },
      { binding: 6, resource: { buffer: this.configuration } },
    ];
  }
  async initialize(): Promise<void> {
    const w = this.words, solid = this.solid, listed = this.listed;
    const weights = UNIFORM_MIXED_ROOT_SMOOTHING, M = ROOT_LAUNCH_ITERATIONS;
    // The page table's first word in the topology, and its block records' (the block neighbour rows follow them).
    const pages = "5u*UM_KEYS", records = `${pages}+${UNIFORM_PAGE_TABLE_WORDS}u`;
    const declarations = /* wgsl */ `
@group(0) @binding(0) var<storage,read> umTopology:array<u32>;
@group(0) @binding(2) var<storage,read_write> umSupport:array<u32>;
@group(1) @binding(0) var<storage,read_write> mixed:array<f32>;
@group(1) @binding(1) var<storage,read_write> native:array<f32>;
@group(1) @binding(2) var<storage,read_write> phi:array<f32>;
@group(1) @binding(5) var<storage,read_write> rounding:array<vec2f>;
struct UMLevelWords{pb:vec4u,mc:vec4u,g:vec4u}
struct UMConfiguration{h:vec4f,flags:vec4u,mixed0:vec4u,mixed1:vec4u,levels:array<UMLevelWords,${ROOT_LEVELS}>}
@group(1) @binding(6) var<uniform> umConfig:UMConfiguration;
struct UMLaunch{${ROOT_LAUNCH_KEYS.map(k => `${k}:u32`).join(",")}}
@group(1) @binding(7) var<uniform> umLaunch:UMLaunch;
@group(1) @binding(3) var<storage,read_write> umList:array<atomic<u32>>;
const UM_NONE=0xffffffffu;
// A row's mask beyond its liquid bits: a row of the solve; odd by colour (the coarsest level's).
const UM_ROW=256u;const UM_ODD=128u;
const UM_LAST=${ROOT_LEVELS - 1}u;
`;
    const body = /* wgsl */ `
// A tiled launch's jobs. Level 0's are the octants of the blocks, 4^3 tiles
// each and 8 block + octant by number (the listed ones where a list is
// built); a smoothing launch's below it are the blocks; a descent's and an
// ascent's below it are runs of 64 rows of the level they write. Workgroup w
// of a launch of W runs jobs w, w+W, ..., so the width is the host's
// parallelism estimate and no job is ever beyond a launch.
fn umJobs0()->u32{if(UM_LISTED){return atomicLoad(&umList[0]);}return 8u*UM_BLOCKS;}
fn umJob0(job:u32)->u32{if(UM_LISTED){return atomicLoad(&umList[4u+job]);}return job;}
// Tile \`lane\` of the octant \`id\`: its key.
fn umJobKey(id:u32,lane:u32)->u32{
 let t=4u*vec3u(id&1u,(id>>1u)&1u,(id>>2u)&1u)+vec3u(lane&3u,(lane>>2u)&3u,lane>>4u);
 return 512u*(id>>3u)+(t.x|(t.y<<3u)|(t.z<<6u));
}

// The mixed pressure schedule's slot gate (support 9n+24, 0 open).
fn umSlotClosed()->bool{return umSupport[9u*UM_KEYS+24u]!=0u;}
// The residency certificate the transfer copies into this ownership's support
// (uniformMixedResidencyWord): a flag per page of 4^3 owners, eight pages a
// block. A row's page of 4^3 owners is the octant of its block its key names.
fn umAbsent(k:u32)->bool{
 return umSupport[UM_RESIDENCY+4u+8u*(k>>9u)+(((k>>2u)&1u)|((k>>4u)&2u)|((k>>6u)&4u))]==0u;
}
${uniformBlockFaceWGSL}
// The page table's words follow the topology's five a key: a word a page of
// the window, a record a block (its page, or all ones where it holds none),
// then the block neighbour table, 27 words a block. A table word is a stored
// block, 0 (AIR) or a word past the stored blocks (SOLID).
${uniformWindowAddressWGSL("umTopology",pages,"UM_BLOCKS")}
fn umPageAt(page:vec3i)->u32{return umPageWord(page);}
fn umBlockPage(b:u32)->vec3i{let r=umBlockWord(b);return vec3i(vec3u(r&63u,(r>>6u)&63u,(r>>12u)&63u));}
// The block holds a page: neither sentinel, nor free.
fn umStored(b:u32)->bool{return b-1u<UM_BLOCKS-2u&&umTopology[${records}+b]!=UM_NONE;}
// The table word of the page at block b's page + d, d within one page.
fn umAround(b:u32,d:vec3i)->u32{return umTopology[${records}+UM_BLOCKS+27u*b+u32(13+d.x+3*d.y+9*d.z)];}
// The same across face 2 axis + side of block b.
fn umAcross(b:u32,axis:u32,side:u32)->u32{
 let step=select(select(9u,3u,axis==1u),1u,axis==0u);
 return umTopology[${records}+UM_BLOCKS+27u*b+select(13u-step,13u+step,side==1u)];
}
// umBoundaryIndex of the all-4h layout: the wall slot of row k on face 2 axis + side of its block.
fn umSlot(k:u32,axis:u32,side:u32)->u32{return UM_KEYS+umFaceTile(k>>9u,2u*axis+side,vec3u(k&7u,(k>>3u)&7u,(k>>6u)&7u));}
fn umP(i:u32)->f32{return mixed[UM_P+i];}
// Keep the rounding lost when a small correction joins the hydrostatic
// pressure. Stencils subtract the high and low words independently.
// The immutable zero header forces each result through its float bit pattern.
// Without this fence Metal fast-math reassociates TwoSum to zero and expands
// pressure differences before their multiplication, losing the low word again.
fn umRounded(x:f32)->f32{return bitcast<f32>(bitcast<u32>(x)^bitcast<u32>(rounding[0].x));}
fn umAddPressure(i:u32,value:f32){
 let a=umP(i);let b=umRounded(value+rounding[i+1u].x);let sum=umRounded(a+b);
 let recovered=umRounded(sum-a);let rest=umRounded(sum-recovered);
 let error=umRounded(a-rest)+umRounded(b-recovered);
 mixed[UM_P+i]=sum;rounding[i+1u].x=error;
}
fn umDifference(a:u32,b:u32)->f32{return umRounded(umRounded(umP(a)-umP(b))+umRounded(rounding[a+1u].x-rounding[b+1u].x));}
// umPressureLiquid of a 4h owner.
fn umLiquid(k:u32)->bool{return ${uniformMixedPressureLiquidWGSL("phi[k]", "4.0*UM_MIN_H")};}
${solid ? `@group(1) @binding(4) var<storage,read> umRecord:array<vec4f>;
fn umTopo(i:u32)->vec4f{return umRecord[i];}` : ""}
fn umSum6(v:array<f32,6>)->f32{return ((v[0]+v[1])+(v[4]+v[5]))+(v[2]+v[3]);}
// umPressureSurfaceTheta / umPressureTheta of two 4h owners.
fn umSurfaceTheta(liquidPhi:f32,airPhi:f32)->f32{return ${uniformMixedSurfaceThetaWGSL("liquidPhi", "airPhi")};}
fn umTheta(a:u32,b:u32)->f32{
 let la=umLiquid(a);if(la==umLiquid(b)){return 1.0;}
 var liquid=a;var air=b;if(!la){liquid=b;air=a;}
 return umSurfaceTheta(phi[liquid],phi[air]);
}
// umBoundaryCoefficient: row k's coupling to its wall slot s.
fn umWall(k:u32,s:u32,axis:u32)->f32{
 if(!umLiquid(k)){return 0.0;}
 let distance=4.0*UM_H[axis];var fraction=0.5;
 ${solid ? "fraction=umTopo(s).x;if(fraction<=1e-6){return 0.0;}" : ""}
 return fraction/(distance*distance);
}
// The regular row's (diagonal, A p): umPressureRegularTerms. Across a face of
// its block a row meets the wall slot where the page there is SOLID, and
// otherwise the row the neighbour table names: a stored block's, or block
// 0's, which is far air.
fn umRow(k:u32)->vec2f{
 var diagonal:array<f32,6>;var values:array<f32,6>;let own=umP(k);
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  let at=2u*axis+side;let stride=1u<<(3u*axis);
  var n=select(k-stride,k+stride,side==1u);
  if(((k>>(3u*axis))&7u)==select(0u,7u,side==1u)){
   let across=umAcross(k>>9u,axis,side);
   if(across>=UM_BLOCKS-1u){
    let s=umSlot(k,axis,side);let weight=umWall(k,s,axis);
    diagonal[at]=weight;values[at]=weight*umDifference(k,s);continue;
   }
   n=512u*across+(select(k+7u*stride,k-7u*stride,side==1u)&511u);
  }
  let distance=4.0*UM_H[axis];
  ${solid ? `let volume=select(umTopo(n)[axis+1u],umTopo(k)[axis+1u],side==1u);
  let coefficient=select(volume/(distance*distance*umTheta(k,n)),0.0,volume<=1e-6);` : "let coefficient=1.0/(distance*distance*umTheta(k,n));"}
  diagonal[at]=coefficient;values[at]=coefficient*select(own+rounding[k+1u].x,umDifference(k,n),umLiquid(n));
 }}
 return vec2f(umSum6(diagonal),umSum6(values));
}
fn umNonfinite(v:f32)->bool{return (bitcast<u32>(v)&0x7f800000u)==0x7f800000u;}
// The CM11a projected residual of one row; a maximal sentinel on failure.
fn umProjected(p:f32,r:f32,low:f32,diagonal:f32)->f32{
 let gap=max(0.0,p-low);
 let projected=max(select(abs(r),gap*diagonal,r<0.0&&-r>=gap*diagonal),max(0.0,low-p)*diagonal);
 return select(3.402823e38,projected,!umNonfinite(p)&&!umNonfinite(r)&&!umNonfinite(projected)&&projected>=0.0);
}

// ---- The correction's levels. p, b, m: both parities of the pressure, RHS
// and bound (the Full-Cycle's own RHS and bound below level 0 are the
// second); c: seven words a row, its -x,+x,-y,+y,-z,+z coefficients and its
// mask; sv: a wall slot's face fraction; phi: a cell's phi as coarsened, then
// continued; v: a cell's open and +x,+y,+z face fractions. Each is a word
// offset of native. A level's rows are its cells by block then local cell,
// then its wall slots by block face (umFaceLevel); level 0's are the mixed
// rows themselves.
struct UMLevel{l:u32,sh:u32,side:u32,cells:u32,p:vec2u,b:vec2u,m:vec2u,c:u32,sv:u32,phi:vec2u,v:u32}
fn umLevel(l:u32)->UMLevel{
 let W=umConfig.levels[l];let sh=3u-l;
 return UMLevel(l,sh,1u<<sh,UM_BLOCKS<<(3u*sh),W.pb.xy,W.pb.zw,W.mc.xy,W.mc.z,W.mc.w,W.g.xy,W.g.z);
}
fn umLRows(L:UMLevel)->u32{return L.cells+6u*(UM_BLOCKS<<(2u*L.sh));}
fn umLBlock(L:UMLevel,j:u32)->u32{if(j<L.cells){return j>>(3u*L.sh);}return ((j-L.cells)>>(2u*L.sh))/6u;}
// A cell row's place in its block, and the row of a block's cell.
fn umLLocal(L:UMLevel,j:u32)->vec3u{return vec3u(j,j>>L.sh,j>>(2u*L.sh))&vec3u(L.side-1u);}
fn umLCell(L:UMLevel,b:u32,c:vec3u)->u32{return (b<<(3u*L.sh))+c.x+L.side*(c.y+L.side*c.z);}
// A slot row's face, the cell row it lies behind, and the slot row of cell c of block b on a face.
fn umLFace(L:UMLevel,j:u32)->u32{return ((j-L.cells)>>(2u*L.sh))%6u;}
fn umLOwner(L:UMLevel,j:u32)->u32{
 let s=j-L.cells;let plane=s>>(2u*L.sh);let axis=(plane%6u)>>1u;
 var c=vec3u(0u);c[axis]=select(0u,L.side-1u,(plane&1u)==1u);c[(axis+1u)%3u]=s&(L.side-1u);c[(axis+2u)%3u]=(s>>L.sh)&(L.side-1u);
 return umLCell(L,plane/6u,c);
}
fn umLSlot(L:UMLevel,b:u32,face:u32,c:vec3u)->u32{return L.cells+umFaceLevel(b,face,L.l,c);}
// Bit 0 is the row's own liquid flag, bits 1..6 its -x,+x,-y,+y,-z,+z neighbours'.
fn umLMask(L:UMLevel,row:u32)->u32{return u32(native[L.c+7u*row+6u]);}
fn umStep(n:u32)->vec3i{let e=array<vec3i,6>(vec3i(-1,0,0),vec3i(1,0,0),vec3i(0,-1,0),vec3i(0,1,0),vec3i(0,0,-1),vec3i(0,0,1));return e[n];}
fn umCorner(i:u32)->vec3i{return vec3i(i32(i&1u),i32((i>>1u)&1u),i32((i>>2u)&1u));}
fn umSum8(v:array<f32,8>)->f32{return ((v[0]+v[5])+(v[1]+v[4]))+((v[2]+v[7])+(v[3]+v[6]));}
// The row across face n of a row: its block's own, or across a face of the
// block the stored block's there or the face's wall slot. A wall slot's one
// neighbour is the cell it lies behind. Where there is none (far air, or a
// slot's other faces) the row itself: no mask names such a neighbour.
fn umLNear(L:UMLevel,row:u32,n:u32)->u32{
 if(row>=L.cells){if(n==(umLFace(L,row)^1u)){return umLOwner(L,row);}return row;}
 let axis=n>>1u;let shift=L.sh*axis;let c=(row>>shift)&(L.side-1u);let stride=1u<<shift;
 if((n&1u)==1u){if(c+1u<L.side){return row+stride;}}else if(c>0u){return row-stride;}
 let b=row>>(3u*L.sh);let across=umAcross(b,axis,n&1u);
 if(across==0u){return row;}
 if(across>=UM_BLOCKS-1u){return umLSlot(L,b,n,umLLocal(L,row));}
 let wrap=(L.side-1u)*stride;
 return (across<<(3u*L.sh))+(select(row+wrap,row-wrap,(n&1u)==1u)&((1u<<(3u*L.sh))-1u));
}
const UM_WN=${weights.length}u;
fn umWeight(i:u32)->f32{let w=array<f32,${weights.length}>(${weights.map(w => w.toFixed(7)).join(",")});return w[i];}
// One weighted simultaneous update of a liquid row from its neighbours'
// iterate (only liquid neighbours enter), projected onto its bound.
fn umLRelax(L:UMLevel,row:u32,mask:u32,old:f32,nb:array<f32,6>,rhs:u32,minimum:f32,weight:f32)->f32{
 var a:array<f32,6>;var s:array<f32,6>;
 for(var n=0u;n<6u;n++){a[n]=native[L.c+7u*row+n];s[n]=select(0.0,a[n]*nb[n],((mask>>(n+1u))&1u)!=0u);}
 let diagonal=umSum6(a);let sum=umSum6(s);
 let p=select(0.0,(sum+native[rhs+row])/diagonal,diagonal>0.0);
 return max(mix(old,p,weight),minimum);
}
// b - A p of a liquid row; an air row has no unknown and carries zero.
fn umLResidual(L:UMLevel,row:u32,par:u32,rhs:u32)->f32{
 let mask=umLMask(L,row);if((mask&1u)==0u){return 0.0;}
 let centre=native[L.p[par]+row];var t:array<f32,6>;
 for(var n=0u;n<6u;n++){
  var other=0.0;if(((mask>>(n+1u))&1u)!=0u){other=native[L.p[par]+umLNear(L,row,n)];}
  t[n]=native[L.c+7u*row+n]*(centre-other);
 }
 return native[rhs+row]-umSum6(t);
}

// Per solve: level 0's phi and fractions, which the setup below coarsens,
// continues and bakes. A lane to a row of level 0. A cell's are the root's
// own; a wall slot's fraction is the wall face's (V=1/2 on a closed domain
// face without a solid record).
@compute @workgroup_size(64) fn setup(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let L=umLevel(0u);let j=gid.x+64u*launched.x*gid.y;if(j>=umLRows(L)){return;}
 let b=umLBlock(L,j);if(!umStored(b)){return;}
 let cell=j<L.cells;var k=j;var row=true;
 if(!cell){k=umLOwner(L,j);let face=umLFace(L,j);row=umAcross(b,face>>1u,face&1u)>=UM_BLOCKS-1u;}
 // An absent page's owners and their wall slots: the rows RHS assembly writes
 // for certified far air (V=0, air: zero RHS and pressure, the bound). RHS
 // assembly and projection stride the resident pages only, and these arena
 // words are shared scratch, rewritten by other stages every frame.
 if(row&&umAbsent(k)){
  var bound=-3.402823e38;
  if(!cell){bound=0.0;}
  ${solid ? "else if(umTopo(k).x<=1e-5){bound=0.0;}" : ""}
  mixed[UM_P+j]=0.0;mixed[UM_B0+j]=0.0;mixed[UM_M0+j]=bound;
 }
 if(cell){
  ${solid ? "let topology=umTopo(j);" : `var topology=vec4f(1.0);
  for(var axis=0u;axis<3u;axis++){if(((j>>(3u*axis))&7u)==7u&&umAcross(b,axis,1u)>=UM_BLOCKS-1u){topology[axis+1u]=0.5;}}`}
  native[L.phi.x+j]=phi[j];for(var component=0u;component<4u;component++){native[L.v+4u*j+component]=topology[component];}
 }else{native[L.sv+j-L.cells]=${solid ? "umTopo(j).x" : "0.5"};}
 if(UM_LISTED){
  // The far-field seeds of the listed cycle kernels: every value a dense
  // cycle leaves at a row outside the list that anything listed reads. Air
  // rows restrict a zero residual (air slots carry a zero RHS), take a zero
  // correction, and keep the bound restrictRoot would give them at p = 0
  // (-FLT_MAX absorbs any finite p where it is read); their measured residual
  // is zero. The seeded bound is also what the list's classification reads
  // to list every bounded job.
  var lower=-3.402823e38;if(row){lower=mixed[UM_M0+j];mixed[UM_RES+j]=0.0;}
  native[L.p.x+j]=0.0;native[L.p.y+j]=0.0;native[L.b.x+j]=0.0;native[L.m.x+j]=lower;
 }
}
// The correction problem: its RHS is the mixed residual, its bound the row's
// bound less the pressure, its initial value zero. A Full-Cycle also keeps
// the pressure, the shifted bound and the residual for its closing V-cycle.
fn restrictRow(s:u32,b:f32){
 let own=umP(s);let lower=mixed[UM_MIN+s]-own;
 if(UM_FULL){mixed[UM_BACKUP+s]=own;rounding[s+1u].y=rounding[s+1u].x;mixed[UM_M1+s]=lower;mixed[UM_B1+s]=b;}
 native[MG_P+s]=0.0;native[MG_B+s]=b;native[MG_MIN+s]=lower;
}
fn restrictCell(k:u32){
 // Branch, not select: an air row must not assemble its stencil.
 var b=0.0;if(umLiquid(k)){b=mixed[UM_RHS+k]-umRow(k).y;}
 restrictRow(k,b);
}
// A slot row couples to its owner alone.
fn restrictSlot(k:u32,s:u32,axis:u32){restrictRow(s,mixed[UM_RHS+s]-umWall(k,s,axis)*umDifference(s,k));}
// The acceptance norm's rows, owners and wall slots.
fn measureCell(k:u32){
 let q=umP(k);
 if(!umLiquid(k)){mixed[UM_RES+k]=select(0.0,3.402823e38,umNonfinite(q));return;}
 let row=umRow(k);
 mixed[UM_RES+k]=umProjected(q,mixed[UM_B0+k]-row.y,mixed[UM_M0+k],row.x);
}
fn measureSlot(k:u32,s:u32,axis:u32){
 let coefficient=umWall(k,s,axis);let q=umP(s);
 mixed[UM_RES+s]=umProjected(q,mixed[UM_B0+s]-coefficient*umDifference(s,k),mixed[UM_M0+s],coefficient);
}
// A lane to a tile of the job: its row, and the wall slot behind each face of its block it lies on.
${["restrict", "measure"].map(name => `
@compute @workgroup_size(64) fn ${name === "measure" ? name : `${name}Root`}(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let count=umJobs0();
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  let id=umJob0(job);if(!UM_LISTED&&!umStored(id>>3u)){continue;}
  let k=umJobKey(id,lane);${name}Cell(k);
  for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
   if(((k>>(3u*axis))&7u)==select(0u,7u,side==1u)&&umAcross(k>>9u,axis,side)>=UM_BLOCKS-1u){${name}Slot(k,umSlot(k,axis,side),axis);}
  }}
 }
}`).join("")}

// ---- A tiled level's smoothing launch: UM_BCOUNT iterations of the visit
// (from its UM_BFIRST-th) on every job of level UM_BL (the root lattice's
// listed ones), a workgroup to a job at a time. UM_BRHS: the level's V-cycle
// RHS and bound, or its Full-Cycle ones. A job's core is its cells: an
// octant of a block on level 0, a block below.
override UM_BLANES:u32=64u;
var<workgroup> umGo:u32;var<workgroup> umAny:u32;var<workgroup> umFlag:array<u32,64>;
// The table words of the pages a core's halo reaches (5^3 where a page is
// two cells an axis, 3^3 otherwise), and the row at each place of the halo.
var<workgroup> umNear:array<u32,125>;var<workgroup> umRegion:array<u32,${(4 + 2 * M) ** 3}>;
// The core and its halo, iterate by iterate: sides c+2m, then c-2+2m, then c-4+2m.
var<workgroup> umBlockA:array<f32,${(4 + 2 * M) ** 3}>;var<workgroup> umBlockB:array<f32,${(2 + 2 * M) ** 3}>;
fn umBlock(k:i32,i:i32)->f32{if((k&1)==0){return umBlockA[i];}return umBlockB[i];}
fn umBlockEnd(row:u32,value:f32,to:u32){
 if(UM_BEND==0u){native[to+row]=value;return;}
 if(UM_BEND==2u){mixed[UM_P+row]=value;rounding[row+1u].x=0.0;}
 else if(UM_BEND==3u){umAddPressure(row,value);umAddPressure(row,rounding[row+1u].y);umAddPressure(row,mixed[UM_BACKUP+row]);}
 else{umAddPressure(row,value);}
}
fn umNearWord(reach:i32,d:vec3i)->u32{let w=2*reach+1;return umNear[(d.x+reach)+w*((d.y+reach)+w*(d.z+reach))];}
// The row at place q of a halo (in cells from the job's block's corner): a
// stored page's cell; none in far air; in a SOLID page the wall slot of the
// stored cell across the page face the place lies against, if there is one.
fn umRegionRow(L:UMLevel,reach:i32,q:vec3i)->u32{
 let d=q>>vec3u(L.sh);let word=umNearWord(reach,d);let local=vec3i(i32(L.side)-1);
 if(word==0u){return UM_NONE;}
 if(word<UM_BLOCKS-1u){return umLCell(L,word,vec3u(q&local));}
 for(var n=0u;n<6u;n++){
  let r=q+umStep(n);let e=r>>vec3u(L.sh);
  if(all(e==d)||any(abs(e)>vec3i(reach))){continue;}
  let across=umNearWord(reach,e);
  if(across-1u<UM_BLOCKS-2u){return umLSlot(L,across,n^1u,vec3u(r&local));}
 }
 return UM_NONE;
}
@compute @workgroup_size(UM_BLANES) fn smoothTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 let R=umLevel(UM_BL);let rhs=R.b[UM_BRHS];let low=R.m[UM_BRHS];let src=R.p[UM_BSRC];let to=R.p[UM_BSRC^1u];
 let core=min(R.side,4u);let reach=select(1,2,R.side==2u);let pages=u32((2*reach+1)*(2*reach+1)*(2*reach+1));
 // A closed slot has no jobs.
 if(lane==0u){umGo=select(select(UM_BLOCKS,umJobs0(),UM_BL==0u),0u,umSlotClosed());}
 let count=workgroupUniformLoad(&umGo);
 // Jobs read one parity and store the other (or the mixed pressure), so a
 // workgroup's jobs are as independent as two workgroups'.
 for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
  var b=job;var origin=vec3i(0);
  if(UM_BL==0u){let id=umJob0(job);b=id>>3u;origin=4*vec3i(vec3u(id&1u,(id>>1u)&1u,(id>>2u)&1u));}
  // The core's own cells are the first lanes'.
  let inside=lane<core*core*core;var row=0u;var mask=0u;
  if(inside){row=umLCell(R,b,vec3u(origin)+vec3u(lane%core,(lane/core)%core,lane/(core*core)));mask=umLMask(R,row);}
  if(lane<64u){umFlag[lane]=mask;}workgroupBarrier();
  if(lane==0u){var flags=0u;for(var i=0u;i<64u;i++){flags|=umFlag[i];}umAny=flags;}
  let live=workgroupUniformLoad(&umAny);
  // A block that holds no page has no rows.
  if((live&UM_ROW)==0u){continue;}
  if((live&1u)==0u){
   // A core without liquid only projects: no row of it reads a neighbour.
   if(inside){
    umBlockEnd(row,max(native[src+row],native[low+row]),to);
    for(var f=0u;f<6u;f++){let s=umLNear(R,row,f);if(s>=R.cells){umBlockEnd(s,max(native[src+s],native[low+s]),to);}}
   }
   continue;
  }
  if(lane<pages){
   let w=2*reach+1;let i=i32(lane);
   umNear[lane]=umPageAt(umBlockPage(b)+vec3i(i%w,(i/w)%w,i/(w*w))-vec3i(reach));
  }
  workgroupBarrier();
  let m=i32(UM_BCOUNT);let cs=i32(core);let s0=cs+2*m;let b0=origin-vec3i(m);
  for(var j=i32(lane);j<s0*s0*s0;j+=i32(UM_BLANES)){
   let at=umRegionRow(R,reach,b0+vec3i(j%s0,(j/s0)%s0,j/(s0*s0)));
   // A place with no row is never liquid and nothing reads it.
   var v=0.0;if(at!=UM_NONE){v=native[src+at];}
   umRegion[j]=at;umBlockA[j]=v;
  }
  workgroupBarrier();
  for(var k=1;k<=m;k++){
   let sp=cs+2+2*(m-k);let sn=sp-2;let weight=umWeight(UM_BFIRST+u32(k-1));let last=k==m;
   for(var j=i32(lane);j<sn*sn*sn;j+=i32(UM_BLANES)){
    let o=vec3i(j%sn,(j/sn)%sn,j/(sn*sn));let jp=(o.x+1)+sp*((o.y+1)+sp*(o.z+1));let jr=(o.x+k)+s0*((o.y+k)+s0*(o.z+k));
    let old=umBlock(k-1,jp);var next=old;let at=umRegion[jr];
    if(at!=UM_NONE){
     let minimum=native[low+at];let own=umLMask(R,at);next=max(old,minimum);
     if((own&1u)!=0u){
      let nb=array<f32,6>(umBlock(k-1,jp-1),umBlock(k-1,jp+1),umBlock(k-1,jp-sp),umBlock(k-1,jp+sp),umBlock(k-1,jp-sp*sp),umBlock(k-1,jp+sp*sp));
      next=umLRelax(R,at,own,old,nb,rhs,minimum,weight);
     }
     if(last){
      umBlockEnd(at,next,to);
      // The core's wall slots stand in its halo, where no job has them for
      // its own: the lane of a cell relaxes its slots from the same iterate.
      let ahead=array<i32,6>(-1,1,-sp,sp,-sp*sp,sp*sp);let around=array<i32,6>(-1,1,-s0,s0,-s0*s0,s0*s0);
      for(var f=0u;f<6u;f++){
       let s=umRegion[jr+around[f]];if(s==UM_NONE||s<R.cells){continue;}
       let before=umBlock(k-1,jp+ahead[f]);let bound=native[low+s];let wall=umLMask(R,s);var value=max(before,bound);
       if((wall&1u)!=0u){var nb:array<f32,6>;nb[f^1u]=old;value=umLRelax(R,s,wall,before,nb,rhs,bound,weight);}
       umBlockEnd(s,value,to);
      }
     }
    }
    if(!last){if((k&1)==1){umBlockB[j]=next;}else{umBlockA[j]=next;}}
   }
   workgroupBarrier();
  }
 }
}

// ---- The small levels (UM_SINGLE and below), in one workgroup. Each helper
// is a former pass, ended by a storage barrier.
const UM_ROWS=${ROOT_COARSE_LANES}u;
override UM_LANES:u32=256u;
// A visit from parity start; the parity it ends on. A lane owns one row: it
// reads the row's coefficients, RHS and bound once, the iterates pass between
// lanes through workgroup memory, and only the visit's last is stored.
var<workgroup> umIterate:array<f32,${this.rows}>;
const UM_HALVES=${2 * ROOT_PAGE_SWEEPS}u;const UM_OMEGA=${ROOT_PAGE_RELAXATION.toFixed(7)};
fn umLSmooth(l:u32,start:u32,rhs:u32,low:u32,lane:u32)->u32{
 let L=umLevel(l);var mask=0u;if(lane<umLRows(L)){mask=umLMask(L,lane);}
 let own=(mask&UM_ROW)!=0u;
 var old=0.0;var minimum=0.0;var b=0.0;var diagonal=0.0;
 var a:array<f32,6>;var at:array<u32,6>;
 if(own){
  old=native[L.p[start]+lane];minimum=native[low+lane];
  if((mask&1u)!=0u){
   for(var n=0u;n<6u;n++){a[n]=native[L.c+7u*lane+n];at[n]=lane;if(((mask>>(n+1u))&1u)!=0u){at[n]=umLNear(L,lane,n);}}
   diagonal=umSum6(a);b=native[rhs+lane];
  }
  umIterate[lane]=old;
 }
 workgroupBarrier();
 var cur=0u;
 for(var i=0u;i<UM_WN;i++){
  let weight=umWeight(i);
  if(own){
   var next=max(old,minimum);
   if((mask&1u)!=0u){
    var s:array<f32,6>;
    for(var n=0u;n<6u;n++){s[n]=select(0.0,a[n]*umIterate[cur*UM_ROWS+at[n]],((mask>>(n+1u))&1u)!=0u);}
    let p=select(0.0,(umSum6(s)+b)/diagonal,diagonal>0.0);
    next=max(mix(old,p,weight),minimum);
   }
   umIterate[(cur^1u)*UM_ROWS+lane]=next;old=next;
  }
  workgroupBarrier();cur^=1u;
 }
 let end=start^(UM_WN&1u);
 if(own){native[L.p[end]+lane]=old;}
 storageBarrier();
 return end;
}
// The coarsest level's solve, in place in parity start: red-black projected
// Gauss-Seidel sweeps, over-relaxed. A cell's colour is the parity of its
// page's coordinates and a wall slot's is the other one (baked into the
// mask), so the two rows of every face differ: a half-sweep reads only the
// other colour's iterate, and a wall slot (coupled to its cell alone) is
// relaxed in the half-sweep after its cell's. Every update is projected onto
// its bound, as a visit's is.
//
// Row j's relaxed value, everything read where it lies.
fn umLSolveRow(L:UMLevel,rhs:u32,low:u32,j:u32,mask:u32)->f32{
 let old=umIterate[j];var a:array<f32,6>;var s:array<f32,6>;
 for(var n=0u;n<6u;n++){a[n]=native[L.c+7u*j+n];s[n]=0.0;if(((mask>>(n+1u))&1u)!=0u){s[n]=a[n]*umIterate[umLNear(L,j,n)];}}
 let diagonal=umSum6(a);let p=select(0.0,(umSum6(s)+native[rhs+j])/diagonal,diagonal>0.0);
 return max(mix(old,p,UM_OMEGA),native[low+j]);
}
// A lane takes rows lane, lane+UM_LANES, ...: the first it holds as a visit
// holds its row (a level of no more rows than lanes is all such rows), the
// others it reads again each half-sweep of their colour. The iterate lives in
// workgroup memory and only the last is stored, by the lane that relaxed the
// row (so no barrier follows the last half-sweep).
fn umLSolve(l:u32,start:u32,rhs:u32,low:u32,lane:u32)->u32{
 let L=umLevel(l);let rows=umLRows(L);var mask=0u;if(lane<rows){mask=umLMask(L,lane);}
 let own=(mask&UM_ROW)!=0u;
 var old=0.0;var minimum=0.0;var b=0.0;var diagonal=0.0;
 var a:array<f32,6>;var at:array<u32,6>;
 if(own){
  old=native[L.p[start]+lane];minimum=native[low+lane];
  if((mask&1u)!=0u){
   for(var n=0u;n<6u;n++){a[n]=native[L.c+7u*lane+n];at[n]=lane;if(((mask>>(n+1u))&1u)!=0u){at[n]=umLNear(L,lane,n);}}
   diagonal=umSum6(a);b=native[rhs+lane];
  }else{old=max(old,minimum);}
  umIterate[lane]=old;
 }
 // An air row past the lanes is projected here, once.
 for(var j=lane+UM_LANES;j<rows;j+=UM_LANES){
  let other=umLMask(L,j);if((other&UM_ROW)==0u){continue;}
  var v=native[L.p[start]+j];if((other&1u)==0u){v=max(v,native[low+j]);}
  umIterate[j]=v;
 }
 let relaxed=(mask&1u)!=0u;let colour=(mask/UM_ODD)&1u;
 for(var half=0u;half<UM_HALVES;half++){
  workgroupBarrier();
  if(relaxed&&colour==(half&1u)){
   var s:array<f32,6>;
   for(var n=0u;n<6u;n++){s[n]=select(0.0,a[n]*umIterate[at[n]],((mask>>(n+1u))&1u)!=0u);}
   let p=select(0.0,(umSum6(s)+b)/diagonal,diagonal>0.0);
   old=max(mix(old,p,UM_OMEGA),minimum);umIterate[lane]=old;
  }
  for(var j=lane+UM_LANES;j<rows;j+=UM_LANES){
   let other=umLMask(L,j);if((other&1u)==0u||((other/UM_ODD)&1u)!=(half&1u)){continue;}
   umIterate[j]=umLSolveRow(L,rhs,low,j,other);
  }
 }
 if(own){native[L.p[start]+lane]=old;}
 for(var j=lane+UM_LANES;j<rows;j+=UM_LANES){if((umLMask(L,j)&UM_ROW)!=0u){native[L.p[start]+j]=umIterate[j];}}
 storageBarrier();
 return start;
}
// The rows under row j of the level above: the eight cells of a cell (2c+o,
// in its own block), and the four slots of a wall slot on its face, each
// twice, so one eight-term fold averages either.
fn umLChild(F:UMLevel,C:UMLevel,j:u32,corner:u32)->u32{
 if(j<C.cells){return umLCell(F,umLBlock(C,j),2u*umLLocal(C,j)+vec3u(umCorner(corner)));}
 let s=j-C.cells;
 return F.cells+((s>>(2u*C.sh))<<(2u*F.sh))+2u*(s&(C.side-1u))+(corner&1u)+F.side*(2u*((s>>C.sh)&(C.side-1u))+((corner>>1u)&1u));
}
// A V-cycle's descent to row j of level C from level F above it: each
// child's residual restricted into the row's RHS (the average; air rows carry
// zero), its correction cleared, and its bound the largest of the children's
// bounds less their pressure (bound-active residuals must not enter an
// unconstrained coarse solve).
fn umLTransitionAt(F:UMLevel,C:UMLevel,par:u32,rhs:u32,low:u32,j:u32){
 if((umLMask(C,j)&UM_ROW)==0u){return;}
 var terms:array<f32,8>;var lower=-3.402823e38;
 for(var corner=0u;corner<8u;corner++){
  if(j>=C.cells&&corner>=4u){terms[corner]=terms[corner-4u];continue;}
  let at=umLChild(F,C,j,corner);
  terms[corner]=umLResidual(F,at,par,rhs);lower=max(lower,native[low+at]-native[F.p[par]+at]);
 }
 native[C.b.x+j]=umSum8(terms)/8.0;native[C.p.x+j]=0.0;native[C.m.x+j]=lower;
}
fn umLTransition(l:u32,par:u32,rhs:u32,low:u32,lane:u32){
 let F=umLevel(l);let C=umLevel(l+1u);let rows=umLRows(C);
 for(var j=lane;j<rows;j+=UM_LANES){umLTransitionAt(F,C,par,rhs,low,j);}
 storageBarrier();
}
// A Full-Cycle's descent: level F's RHS and bound are corrections already,
// restricted unmasked into row j's Full-Cycle fields.
fn umLRestrictAt(F:UMLevel,C:UMLevel,rhs:u32,low:u32,j:u32){
 if((umLMask(C,j)&UM_ROW)==0u){return;}
 var terms:array<f32,8>;var lower=-3.402823e38;
 for(var corner=0u;corner<8u;corner++){
  if(j>=C.cells&&corner>=4u){terms[corner]=terms[corner-4u];continue;}
  let at=umLChild(F,C,j,corner);
  terms[corner]=native[rhs+at];lower=max(lower,native[low+at]);
 }
 native[C.b.y+j]=umSum8(terms)/8.0;native[C.m.y+j]=lower;
}
fn umLRestrict(l:u32,rhs:u32,low:u32,lane:u32){
 let F=umLevel(l);let C=umLevel(l+1u);let rows=umLRows(C);
 for(var j=lane;j<rows;j+=UM_LANES){umLRestrictAt(F,C,rhs,low,j);}
 storageBarrier();
}
// Either descent out of a tiled level UM_TL, a workgroup to a run of 64 rows
// of the level below at a time: a V-cycle's from the iterate in parity
// UM_TSRC, or a Full-Cycle's (UM_TFULL). UM_TRHS: the level's V-cycle RHS
// and bound, or its Full-Cycle ones.
@compute @workgroup_size(64) fn descendTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let F=umLevel(UM_TL);let C=umLevel(UM_TL+1u);let rhs=F.b[UM_TRHS];let low=F.m[UM_TRHS];let rows=umLRows(C);
 for(var j=64u*(g.x+n.x*g.y)+lane;j<rows;j+=64u*n.x*n.y){
  if(UM_TFULL){umLRestrictAt(F,C,rhs,low,j);}else{umLTransitionAt(F,C,UM_TSRC,rhs,low,j);}
 }
}
// Level C's pressure (parity coarse) onto row \`row\` of level F above it:
// added to parity src, or assigned; stored in parity to. Cell-centred
// trilinear over the eight cells of C around the row's cell, each found
// through the block neighbour table where it lies in another page: a stored
// page's pressure; zero in far air; a wall has none there, and the rest
// renormalise. A wall slot takes its cell's interpolation with the taps held
// to the cell's own layer along the wall's axis.
fn umLProlongAt(F:UMLevel,C:UMLevel,src:u32,to:u32,coarse:u32,add:bool,row:u32){
 var cell=row;var held=3u;if(row>=F.cells){cell=umLOwner(F,row);held=umLFace(F,row)>>1u;}
 let b=umLBlock(F,cell);let c=vec3i(umLLocal(F,cell));let odd=(c&vec3i(1))==vec3i(1);
 var f=select(vec3f(0.75),vec3f(0.25),odd);var base=(c>>vec3u(1u))-select(vec3i(1),vec3i(0),odd);
 if(held<3u){f[held]=0.25;base[held]=(c[held]>>1u)-1;}
 let m=i32(C.side);var values:array<f32,8>;var weights:array<f32,8>;
 for(var corner=0u;corner<8u;corner++){
  let o=umCorner(corner);let p=base+o;values[corner]=0.0;weights[corner]=0.0;
  if(held<3u&&o[held]==0){continue;}
  let d=vec3i(p>=vec3i(m))-vec3i(p<vec3i(0));
  var word=b;if(any(d!=vec3i(0))){word=umAround(b,d);}
  if(word>=UM_BLOCKS-1u){continue;}
  let w=select(1.0-f.x,f.x,o.x==1)*select(1.0-f.y,f.y,o.y==1)*select(1.0-f.z,f.z,o.z==1);
  weights[corner]=w;if(word!=0u){values[corner]=w*native[C.p[coarse]+umLCell(C,word,vec3u(p-m*d))];}
 }
 let value=umSum8(values);let total=umSum8(weights);
 var v=select(0.0,value/total,total>0.0);if(add){v=native[F.p[src]+row]+v;}
 native[F.p[to]+row]=v;
}
fn umLProlong(l:u32,src:u32,to:u32,coarse:u32,add:bool,lane:u32){
 let F=umLevel(l);let C=umLevel(l+1u);let rows=umLRows(F);
 for(var j=lane;j<rows;j+=UM_LANES){if((umLMask(F,j)&UM_ROW)!=0u){umLProlongAt(F,C,src,to,coarse,add,j);}}
 storageBarrier();
}
// The same onto a tiled level UM_PL, a workgroup to a job at a time: the
// root lattice's listed octants (its far field keeps the zero correction its
// seeds hold), a lane to a tile and its wall slots; below it a run of 64
// rows.
@compute @workgroup_size(64) fn ascendTiles(@builtin(workgroup_id) g:vec3u,@builtin(num_workgroups) n:vec3u,@builtin(local_invocation_index) lane:u32){
 if(umSlotClosed()){return;}
 let F=umLevel(UM_PL);let C=umLevel(UM_PL+1u);
 if(UM_PL==0u){
  let count=umJobs0();
  for(var job=g.x+n.x*g.y;job<count;job+=n.x*n.y){
   let id=umJob0(job);if(!UM_LISTED&&!umStored(id>>3u)){continue;}
   let k=umJobKey(id,lane);umLProlongAt(F,C,UM_PSRC,UM_PTO,UM_PCOARSE,UM_PADD,k);
   for(var f=0u;f<6u;f++){let s=umLNear(F,k,f);if(s>=F.cells){umLProlongAt(F,C,UM_PSRC,UM_PTO,UM_PCOARSE,UM_PADD,s);}}
  }
  return;
 }
 let rows=umLRows(F);
 for(var j=64u*(g.x+n.x*g.y)+lane;j<rows;j+=64u*n.x*n.y){
  if((umLMask(F,j)&UM_ROW)!=0u){umLProlongAt(F,C,UM_PSRC,UM_PTO,UM_PCOARSE,UM_PADD,j);}
 }
}
// A V-cycle of level a and everything below it, on (rhs, low) at level a
// and each level's V-cycle fields below; the parity level a's result is in.
fn umLV(a:u32,rhsA:u32,lowA:u32,start:u32,lane:u32)->u32{
 var par:array<u32,${ROOT_LEVELS}>;par[a]=start;
 for(var l=a;l<UM_LAST;l++){
  let L=umLevel(l);let rhs=select(L.b.x,rhsA,l==a);let low=select(L.m.x,lowA,l==a);
  par[l]=umLSmooth(l,par[l],rhs,low,lane);
  umLTransition(l,par[l],rhs,low,lane);par[l+1u]=0u;
 }
 {
  let L=umLevel(UM_LAST);let rhs=select(L.b.x,rhsA,UM_LAST==a);let low=select(L.m.x,lowA,UM_LAST==a);
  par[UM_LAST]=umLSolve(UM_LAST,par[UM_LAST],rhs,low,lane);
 }
 for(var i=UM_LAST;i>a;i--){
  let l=i-1u;let L=umLevel(l);let rhs=select(L.b.x,rhsA,l==a);let low=select(L.m.x,lowA,l==a);
  umLProlong(l,par[l],par[l]^1u,par[i],true,lane);par[l]^=1u;
  par[l]=umLSmooth(l,par[l],rhs,low,lane);
 }
 return par[a];
}
// The small levels' part of a traversal: level UM_SINGLE's V-cycle from the
// zero correction a descent left in its parity 0, on its V-cycle RHS and
// bound or (UM_CRHS) its Full-Cycle ones. UM_CFULL: a Full-Cycle's opening
// instead (Algorithm 3 on these levels): the level's Full-Cycle RHS and bound
// restricted to the coarsest, solved there, and each level up assigned the
// prolonged solution and given a V-cycle. The result's parity is static
// (singleParity).
@compute @workgroup_size(UM_LANES) fn coarse(@builtin(local_invocation_index) lane:u32){
 if(lane==0u){umGo=select(1u,0u,umSlotClosed());}
 if(workgroupUniformLoad(&umGo)==0u){return;}
 if(UM_CFULL){
  for(var l=UM_SINGLE;l<UM_LAST;l++){let L=umLevel(l);umLRestrict(l,L.b.y,L.m.y,lane);}
  let B=umLevel(UM_LAST);let rows=umLRows(B);
  for(var j=lane;j<rows;j+=UM_LANES){native[B.p.x+j]=0.0;}
  storageBarrier();
  var top=umLSolve(UM_LAST,0u,B.b.y,B.m.y,lane);
  for(var i=UM_LAST;i>UM_SINGLE;i--){
   let l=i-1u;let L=umLevel(l);
   umLProlong(l,0u,0u,top,false,lane);
   top=umLV(l,L.b.y,L.m.y,0u,lane);
  }
 }else{
  let L=umLevel(UM_SINGLE);let top=umLV(UM_SINGLE,L.b[UM_CRHS],L.m[UM_CRHS],0u,lane);
 }
}
${uniformPressureHierarchyWGSL}`;
    const aliases: Record<string, string> = {
      UM_H: "umConfig.h.xyz", UM_MIN_H: "min(umConfig.h.x,min(umConfig.h.y,umConfig.h.z))", UM_FAR: "umConfig.h.w",
      UM_P: "umConfig.mixed0.x", UM_B0: "umConfig.mixed0.y", UM_B1: "umConfig.mixed0.z", UM_M0: "umConfig.mixed0.w",
      UM_M1: "umConfig.mixed1.x", UM_BACKUP: "umConfig.mixed1.y", UM_RES: "umConfig.mixed1.z",
      // Level 0's correction, V-cycle RHS and bound.
      MG_P: "umConfig.levels[0].pb.x", MG_B: "umConfig.levels[0].pb.z", MG_MIN: "umConfig.levels[0].mc.x",
      UM_SINGLE: "umConfig.flags.y", UM_LISTED: "(umConfig.flags.w!=0u)",
      UM_KEYS: "umConfig.flags.x", UM_BLOCKS: "(umConfig.flags.x/512u)", UM_RESIDENCY: "(9u*umConfig.flags.x+28u)",
    };
    for (const key of ROOT_LAUNCH_KEYS) aliases[key] = ["UM_FULL", "UM_CFULL", "UM_TFULL", "UM_PADD"].includes(key) ? `(umLaunch.${key}!=0u)` : `umLaunch.${key}`;
    const code = declarations + body.replace(/\b(?:UM|MG)_[A-Z0-9_]+\b/g, key => aliases[key] ?? key);
    const module = this.device.createShaderModule({ label: "Uniform mixed pressure root", code });
    const info = await module.getCompilationInfo(), errors = info.messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(errors.map(m => `${m.lineNum}: ${m.message}`).join("\n"));
    const pipelineLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.level.ownership.bindLayout, this.resources] });
    // A fixed workgroup grid, or a tiled launch's jobs, lanes and whether the jobs are the list's.
    type Shape = readonly [number, number, number] | { jobs: number; lanes: number; listed: boolean };
    type Variant = [string, Record<string, number>, Shape];
    const rows = this.levelRows;
    // A launch of a lane to an item over the whole pool, 64 lanes a workgroup, wrapped into a second dimension.
    const grid = (groups: number): readonly [number, number, number] => { const x = Math.min(groups, this.maxWorkgroups); return [x, Math.ceil(groups / x), 1]; };
    const lanes = (items: number) => grid(Math.ceil(items / 64));
    const root = { jobs: this.rootJobs, lanes: 64, listed }, runs = (level: number) => ({ jobs: Math.ceil(rows[level]! / 64), lanes: 64, listed: false });
    const variants: Record<string, Variant> = {
      setup: ["setup", {}, lanes(rows[0]!)], measure: ["measure", {}, root],
      restrictV: ["restrictRoot", {}, root], restrictInner: ["restrictRoot", { UM_RHS: w.b1, UM_MIN: w.m1 }, root], restrictFull: ["restrictRoot", { UM_FULL: 1 }, root],
    };
    // The setup below level 0's geometry: the pyramid, then each level's continuation and coefficients, and the list.
    const setup: string[] = ["setup"];
    for (let l = 1; l < ROOT_LEVELS; l++) { variants[`down.${l}`] = ["mgDownsample", { UM_SL: l }, lanes(rows[l]!)]; setup.push(`down.${l}`); }
    for (let l = 0; l < ROOT_LEVELS; l++) {
      variants[`extend.${l}`] = ["mgExtend", { UM_SL: l }, lanes(this.levelCells[l]!)];
      variants[`bake.${l}`] = ["mgBake", { UM_SL: l }, lanes(rows[l]!)];
      setup.push(`extend.${l}`, `bake.${l}`);
      if (l === 0 && listed) {
        // A workgroup a job classifies it; a lane a job lists it from its neighbours' classes.
        variants.classify = ["mgClassify", {}, grid(this.rootJobs)]; variants.list = ["mgList", {}, lanes(this.rootJobs)];
        setup.push("classify", "list");
      }
    }
    // Every launch the traversals issue (see cycle() and opening()).
    const add = (entry: string, constants: Record<string, number>) => {
      const level = constants.UM_BL ?? constants.UM_PL ?? 0;
      variants[entry] ??= entry.startsWith("coarse") ? ["coarse", constants, [1, 1, 1]]
        : entry.startsWith("descend") ? ["descendTiles", constants, runs(constants.UM_TL! + 1)]
        : entry.startsWith("ascend") ? ["ascendTiles", constants, level ? runs(level) : root]
        : ["smoothTiles", constants, level ? { jobs: this.blocks, lanes: this.tileLanes, listed: false } : root];
    };
    for (const end of [1, 2, 3] as const) this.cycle(0, 0, 0, end, add);
    this.opening(add);
    // Every encoded launch has its own immutable record. Dynamic offsets
    // select records within one bind group; submitting several cycles cannot
    // overwrite the parameters of an earlier dispatch.
    const stride = Math.ceil(ROOT_LAUNCH_BYTES / this.device.limits.minUniformBufferOffsetAlignment) * this.device.limits.minUniformBufferOffsetAlignment;
    const entries = Object.entries(variants), launchWords = new Uint32Array(stride / 4 * entries.length);
    this.launches = this.device.createBuffer({ label: "Uniform root pressure launches", size: launchWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const shared = new Map<string, Promise<GPUComputePipeline>>();
    // Both tiled widths are part of the installed path, independent of how
    // many of this pool's levels happen to fit the coarse workgroup.
    const deepConstants = { UM_BLANES: this.tileLanes };
    const deepKey = `smoothTiles:${JSON.stringify(deepConstants)}`;
    shared.set(deepKey, this.device.createComputePipelineAsync({ layout: pipelineLayout,
      compute: { module, entryPoint: "smoothTiles", constants: deepConstants } }));
    await Promise.all(entries.map(async ([entry, [entryPoint, parameters, shape]], i) => {
      const defaults: Record<string, number> = { UM_RHS: w.b0, UM_MIN: w.m0, UM_BCOUNT: M, UM_PADD: 1, ...parameters };
      ROOT_LAUNCH_KEYS.forEach((key, k) => { launchWords[i * stride / 4 + k] = defaults[key] ?? 0; });
      const constants: Record<string, number> = {};
      if (entryPoint === "coarse") constants.UM_LANES = this.lanes;
      if (entryPoint === "smoothTiles") constants.UM_BLANES = "jobs" in shape ? shape.lanes : 64;
      const key = `${entryPoint}:${JSON.stringify(constants)}`;
      let compilation = shared.get(key);
      if (!compilation) { compilation = this.device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint, constants } }); shared.set(key, compilation); }
      const pipeline = await compilation, offset = i * stride;
      this.pipelines.set(entry, "jobs" in shape
        ? { pipeline, offset, offsets: [offset], tiles: shape.jobs, lanes: shape.lanes, listed: shape.listed }
        : { pipeline, offset, offsets: [offset], groups: shape, tiles: 0, lanes: 0, listed: false });
    }));
    await Promise.all(shared.values());
    this.device.queue.writeBuffer(this.launches, 0, launchWords);
    this.group = this.device.createBindGroup({ layout: this.resources, entries: [...this.resourceEntries,
      { binding: 7, resource: { buffer: this.launches, size: ROOT_LAUNCH_BYTES } },
    ] });
    const resolve = (entry: string) => {
      const launch = this.pipelines.get(entry); if (!launch) throw new Error(`Mixed pressure root has no launch ${entry}`);
      return launch;
    };
    const traversal = (body: (launch: (entry: string) => void) => void) => {
      const launches: RootLaunch[] = []; body(entry => launches.push(resolve(entry))); return launches;
    };
    this.traversals = { setup: setup.map(resolve), measure: [resolve("measure")],
      vCycle: traversal(launch => { launch("restrictV"); this.cycle(0, 0, 0, 1, launch); }),
      // Algorithm 3 at the root: the Full-Cycle on the residual under the
      // shifted bound, a closing V-cycle, then the saved pressure added back.
      fullCycle: traversal(launch => {
        launch("restrictFull"); this.opening(launch); this.cycle(0, 0, 0, 2, launch);
        launch("restrictInner"); this.cycle(0, 0, 0, 3, launch);
      }) };
  }
  /** The parity the one-workgroup launch leaves level `tiled`'s result in
   * from parity 0: the coarsest solve is in place; a V-cycle is two visits
   * and the prolongation's store. */
  private get singleParity(): number {
    return this.tiled === ROOT_LEVELS - 1 ? 0 : 1;
  }
  /** A V-cycle of level `l` (a tiled one, or the one-workgroup launch's
   * first) from the iterate in parity `start`, on the level's V-cycle RHS and
   * bound or (`set` 1) its Full-Cycle ones: a visit, the level below, a
   * visit. The root lattice's last launch stores into the mixed pressure
   * (`end`). Returns the result's parity. */
  private cycle(l: number, set: number, start: number, end: RootEnd, launch: (entry: string, constants: Record<string, number>) => void): number {
    if (l === this.tiled) {
      if (start !== 0) throw new Error("The one-workgroup V-cycle starts from parity 0");
      launch(`coarse.${set}`, { UM_CRHS: set }); return this.singleParity;
    }
    let parity = start;
    const visit = (last: RootEnd) => this.visit.forEach((v, k) => {
      const to = k === this.visit.length - 1 ? last : 0;
      launch(`smooth.${l}.${set}.${parity}.${v.first}.${v.count}.${to}`, { UM_BL: l, UM_BRHS: set, UM_BSRC: parity, UM_BFIRST: v.first, UM_BCOUNT: v.count, UM_BEND: to });
      parity ^= 1;
    });
    visit(0);
    launch(`descend.${l}.${set}.${parity}`, { UM_TL: l, UM_TRHS: set, UM_TSRC: parity });
    const below = this.cycle(l + 1, 0, 0, 0, launch);
    launch(`ascend.${l}.${parity}.${below}.add`, { UM_PL: l, UM_PSRC: parity, UM_PTO: parity ^ 1, UM_PCOARSE: below, UM_PADD: 1 }); parity ^= 1;
    visit(l === 0 ? end : 0);
    return parity;
  }
  /** A Full-Cycle's opening below the root lattice (Algorithm 3): the root
   * RHS and bound restricted level by level, the coarsest solved, and each
   * level up assigned the prolonged solution and given a V-cycle; the root
   * lattice's parity 0 is assigned the result. */
  private opening(launch: (entry: string, constants: Record<string, number>) => void): void {
    for (let l = 0; l < this.tiled; l++) launch(`descend.${l}.full`, { UM_TL: l, UM_TRHS: l ? 1 : 0, UM_TFULL: 1 });
    launch("coarse.full", { UM_CFULL: 1 });
    let top = this.singleParity;
    for (let l = this.tiled - 1; l >= 0; l--) {
      launch(`ascend.${l}.${top}.assign`, { UM_PL: l, UM_PSRC: 0, UM_PTO: 0, UM_PCOARSE: top, UM_PADD: 0 });
      if (l > 0) top = this.cycle(l, 1, 0, 0, launch);
    }
  }
  /** capacity: a launch of a slot the plan holds in reserve (see
   * ROOT_CAPACITY_THREADS). */
  private dispatch(passes: UniformMixedPressurePasses, launch: RootLaunch, capacity: boolean): void {
    const pass = passes.pass;
    pass.setPipeline(launch.pipeline);
    if (this.boundPass !== pass) { this.boundPass = pass; pass.setBindGroup(0, this.level.ownership.bindGroup); }
    pass.setBindGroup(1, this.group, launch.offsets);
    const groups = launch.groups;
    if (groups) { pass.dispatchWorkgroups(groups[0], groups[1], groups[2]); return; }
    let width = launch.listed ? Math.min(launch.tiles, this.work ?? launch.tiles) : launch.tiles;
    if (capacity) width = Math.min(width, Math.max(1, Math.floor(ROOT_CAPACITY_THREADS / launch.lanes)));
    const x = Math.min(width, this.maxWorkgroups);
    pass.dispatchWorkgroups(x, Math.ceil(width / x));
  }
  /** The listed launches' width from a completed frame's list count
   * (uniformBufferedWork: parallelism only, every kernel strides the live
   * list). */
  observeWork(listed: number): void {
    this.work = uniformBufferedWork(this.work ?? this.rootJobs, listed, this.rootJobs);
  }
  /** The cycle list's count (this solve's) into a frame receipt: one word,
   * zero without a list. */
  encodeWorkReceipt(encoder: GPUCommandEncoder, target: GPUBuffer, offset: number): void {
    if (this.listed) encoder.copyBufferToBuffer(this.list, 0, target, offset, 4);
    else encoder.clearBuffer(target, offset, 4);
  }
  /** One pass of a traversal's launches, in order. */
  private replay(passes: UniformMixedPressurePasses, encoder: GPUCommandEncoder, launches: readonly RootLaunch[], capacity: boolean): void {
    passes.begin(encoder);
    for (let i = 0; i < launches.length; i++) this.dispatch(passes, launches[i]!, capacity);
    passes.finish();
  }
  private get resolved(): RootTraversals {
    if (!this.traversals) throw new Error("Mixed pressure root is not initialized");
    return this.traversals;
  }
  /** Per solve, after RHS assembly and before the initial measure: every
   * level's geometry and coefficients, and the cycle list. Ungated. */
  encodeSetup(encoder: GPUCommandEncoder): void {
    encoder.clearBuffer(this.rounding);
    // The list's count; the setup's last launch appends to it.
    if (this.listed) encoder.clearBuffer(this.list, 0, 4);
    this.replay(this.setupPasses, encoder, this.resolved.setup, false);
  }
  /** capacity (here and on the cycles): the slot is one the plan holds in
   * reserve, launched narrow. */
  encodeMeasure(encoder: GPUCommandEncoder, capacity = false): void { this.replay(this.measurePasses, encoder, this.resolved.measure, capacity); }
  encodeVCycle(encoder: GPUCommandEncoder, capacity = false): void { this.replay(this.vCyclePasses, encoder, this.resolved.vCycle, capacity); }
  /** Algorithm 3 at the root (see initialize). */
  encodeFullCycle(encoder: GPUCommandEncoder, capacity = false): void { this.replay(this.fullCyclePasses, encoder, this.resolved.fullCycle, capacity); }
}
