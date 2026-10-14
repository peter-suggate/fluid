import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { compileUniformStencil, uniformCompileStencilWGSL } from "./uniform-compiled-topology";
import {
  UNIFORM_KIND, UNIFORM_PAGE_TABLE_WORDS, UNIFORM_PAGE_WINDOW, uniformFaceTile, uniformFaceTileSlots, uniformLatticeAddressWGSL,
  uniformTileCoord, uniformTileKey, uniformTileKind, uniformTileLive, type UniformLatticeAddress,
} from "./uniform-lattice-address";
import { UNIFORM_MIXED_DETAIL_RING } from "./uniform-mixed-layout";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedDetailViolationWord } from "./uniform-mixed-topology.wgsl";

/** The pressure root's owners, built on the GPU from the page table every
 * step.
 *
 * The root is all-4h and its owner index is the tile's key: every per-owner
 * root array (the arena rows, the wall slots after them, the root phi, the
 * root record, the presented pressure and phi) holds a row a tile key, 512 a
 * block, so a block has its rows wherever the pool puts its page. What
 * follows the page set is the topology the root's kernels reach those rows
 * through: a key's tile word (its owner index, the key itself) and its two
 * stencil words (which of its 26 neighbours are stored). A host-built root
 * names the owners of the seed's blocks only.
 *
 * A tile nothing stores has no row of its own. A read of it resolves, as
 * every keyed read does, to a row of block 0 where its page is AIR and of
 * the last block where it is SOLID: the two sentinel blocks' rows are the
 * image of every such tile.
 *
 * roKeys, one lane a tile key: a live key takes its own key and the all-4h
 * stencil of the stored neighbours it has now; a free block's key takes zero
 * words, as the host seed held them. A claimed block's keys, a retired
 * block's keys and both sides of every changed page face are therefore right
 * in the step of the commit, with no list of what changed. Bit 0 of a live
 * key's second stencil word (the detail ring, written by the detail
 * storage's ring kernels) is kept. A sentinel block's key takes zero words
 * and its row the image: far-air phi, and a record open with V=1 on every
 * face for block 0 and closed with V=0 for the last block. A free block's
 * rows take the far-air phi too: a claim finds the air its page was. No certificate
 * flag is ever set for a sentinel block, so no kernel strides those rows and
 * the root's solve writes none of them: their pressure is the zero the
 * presented pressure starts with.
 *
 * The launch is fixed by capacity (tile keys) and strided by nothing: no
 * count is read back and none is needed. */

/** Sticky bit of the root support's violation word (beside the detail
 * storage's, which the frame receipt already carries): a block record names a
 * page whose table word is another block. Two keys would hold one tile. */
export const UNIFORM_MIXED_ROOT_FATAL = 0x10000;
export const UNIFORM_MIXED_ROOT_FATAL_MESSAGE = "Uniform pressure root: a block record names a page the page table gives to another block (two tile keys would hold one tile); the root's owners were withheld for that block";
/** Sticky bit of the same word: a wall slot the pressure root's tiled levels
 * do not reach. A tiled level smooths a block's cells with a halo around
 * them, a cell of the halo to a place, and a wall slot is the place across
 * its wall face. A SOLID page with stored pages across faces of two axes
 * holds, along the edge between those faces, places that are the wall slot
 * of two cells; the halo holds one. roKeys latches the bit when a held block
 * has a SOLID page across a face that a stored page lies across another
 * axis's face of. */
export const UNIFORM_MIXED_ROOT_UNREACHED = 0x20000;
export const UNIFORM_MIXED_ROOT_UNREACHED_MESSAGE = "Uniform pressure root: a SOLID page has stored pages across faces of two axes, so two wall slots share a cell of the root's smoothing halo along its edge and one of them would not be solved";
/** A tile with nothing stored reads this phi, in cell sizes (the largest
 * axis's): the far air of a claimed block's fields (the page initialise's
 * `near`), which a certified-absent page's owners hold. */
export const UNIFORM_MIXED_ROOT_FAR_CELLS = 16;

/** roKeys on the host: the root's tile words, stencil pairs and blend masks
 * (the topology's words [0,n), [2n,4n) and [4n,5n); the lists at [n,2n) have
 * no reader on the root and are left alone) for the address's page table and
 * block records. held: the words before, for the ring bit. */
export function uniformMixedRootOwnerWords(address: UniformLatticeAddress, held?: Uint32Array): { words: Uint32Array<ArrayBuffer>; fatal: number } {
  const n = address.tileKeys, words = new Uint32Array(5 * n), c: [number, number, number] = [0, 0, 0];
  let fatal = 0;
  if (held) words.set(held.subarray(n, 2 * n), n);
  for (let k = 0; k < n; k++) {
    let word = 0, s0 = 0, s1 = 0;
    if (uniformTileLive(address, k)) {
      uniformTileCoord(address, k, c);
      if ((k & 511) === 0) for (let f = 0; f < 6; f++) {
        const q: [number, number, number] = [c[0], c[1], c[2]];
        q[f >> 1]! += f & 1 ? 8 : -8;
        if (uniformTileKind(address, q[0], q[1], q[2]) !== UNIFORM_KIND.solid) continue;
        for (let g = 0; g < 6; g++) {
          if (g >> 1 === f >> 1) continue;
          const r: [number, number, number] = [q[0], q[1], q[2]];
          r[g >> 1]! += g & 1 ? 8 : -8;
          if (uniformTileKind(address, r[0], r[1], r[2]) === UNIFORM_KIND.resident) fatal |= UNIFORM_MIXED_ROOT_UNREACHED;
        }
      }
      if (uniformTileKey(address, c[0], c[1], c[2]) !== k) fatal |= UNIFORM_MIXED_ROOT_FATAL;
      else {
        let valid = 0;
        for (let i = 0; i < 27; i++) {
          if (uniformTileKind(address, c[0] + i % 3 - 1, c[1] + Math.floor(i / 3) % 3 - 1, c[2] + Math.floor(i / 9) - 1) === UNIFORM_KIND.resident) valid |= 1 << i;
        }
        word = k;s0 = (4 << 27) >>> 0;
        s1 = ((4 << 27) | compileUniformStencil(s0, valid) | ((held?.[2 * n + 2 * k + 1] ?? 0) & UNIFORM_MIXED_DETAIL_RING)) >>> 0;
      }
    }
    words[k] = word;words[2 * n + 2 * k] = s0;words[2 * n + 2 * k + 1] = s1;words[4 * n + k] = 0;
  }
  return { words, fatal };
}

/** A wall slot's row (umBoundaryIndex of the all-4h layout): after the tile
 * keys, the slot of tile `key` on face 2 axis + side of its block. */
export function uniformMixedRootSlot(address: UniformLatticeAddress, key: number, axis: number, side: number): number {
  return address.tileKeys + uniformFaceTile(key >> 9, 2 * axis + side, key & 7, (key >> 3) & 7, (key >> 6) & 7);
}

/** The rows roKeys writes itself: the sentinel blocks' phi and record (four
 * floats a row) and a free block's phi, by tile key. far:
 * UNIFORM_MIXED_ROOT_FAR_CELLS cell sizes, in metres. */
export function uniformMixedRootSentinelRows(address: UniformLatticeAddress, phi: Float32Array, record: Float32Array | undefined, far: number): void {
  const n = address.tileKeys;
  for (let k = 0; k < n; k++) {
    const sentinel = k < 512 || k >= n - 512;
    if (sentinel || !uniformTileLive(address, k)) phi[k] = far;
    if (sentinel) record?.fill(k < 512 ? 1 : 0, 4 * k, 4 * k + 4);
  }
}

const float = (x: number) => Number.isInteger(x) ? x.toFixed(1) : String(x);

/** The kernels. record: the root record (a vec4 a row) is bound, at binding 3. */
export function uniformMixedRootOwnersWGSL(address: UniformLatticeAddress, lattice: { readonly dimensions: readonly number[]; readonly cellSize_m: readonly number[] }, record: boolean): string {
  const n = address.tileKeys, table = 5 * n;
  const paged = { blocks: `${n / 512}u` };
  const far = UNIFORM_MIXED_ROOT_FAR_CELLS * Math.max(...lattice.cellSize_m);
  return /* wgsl */ `
const RO_KEYS:u32=${n}u;
const RO_VIOLATION:u32=${uniformMixedDetailViolationWord(n)}u;
const RO_FAR:f32=${float(far)};
// The root topology: five words a tile key, then the page table and the block records the pool mirrors.
@group(0) @binding(0) var<storage,read_write> roTopology:array<u32>;
// The root support: the violation word.
@group(0) @binding(1) var<storage,read_write> roSupport:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> roPhi:array<f32>;
${record ? "@group(0) @binding(3) var<storage,read_write> roRecord:array<vec4f>;" : ""}
${uniformWindowAddressWGSL("roTopology",`${table}u`,"0u")}
${uniformLatticeAddressWGSL({ constant: true, paged })}
${uniformCompileStencilWGSL}
@compute @workgroup_size(64) fn roKeys(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let k=gid.x+64u*launched.x*gid.y;if(k>=RO_KEYS){return;}
 var word=0u;var first=0u;var second=0u;let live=umTileLive(k);
 if(live){
  let c=vec3i(umTileCoord(k));
  // Once a block: a SOLID page across one of its faces with a stored page
  // across another axis's face has two wall slots on one cell of the tiled
  // levels' halo (UNIFORM_MIXED_ROOT_UNREACHED).
  if((k&511u)==0u){for(var f=0u;f<6u;f++){
   var q=c;q[f>>1u]+=select(-8,8,(f&1u)==1u);
   if(umTileKind(q)!=UM_SOLID){continue;}
   for(var g=0u;g<6u;g++){
    if((g>>1u)==(f>>1u)){continue;}
    var r=q;r[g>>1u]+=select(-8,8,(g&1u)==1u);
    if(umTilePresent(r)){atomicOr(&roSupport[RO_VIOLATION],${UNIFORM_MIXED_ROOT_UNREACHED}u);}
   }
  }}
  // The record and the table name each other, or two keys would hold one tile.
  if(umTileKey(c)!=k){atomicOr(&roSupport[RO_VIOLATION],${UNIFORM_MIXED_ROOT_FATAL}u);}
  else{
   var valid=0u;
   for(var i=0u;i<27u;i++){
    let q=c+vec3i(vec3u(i%3u,(i/3u)%3u,i/9u))-vec3i(1);
    if(umTilePresent(q)){valid|=1u<<i;}
   }
   word=k;first=4u<<27u;
   second=(4u<<27u)|umCompileStencil(first,valid)|(roTopology[2u*RO_KEYS+2u*k+1u]&${UNIFORM_MIXED_DETAIL_RING}u);
  }
 }
 roTopology[k]=word;roTopology[2u*RO_KEYS+2u*k]=first;roTopology[2u*RO_KEYS+2u*k+1u]=second;roTopology[4u*RO_KEYS+k]=0u;
 // A sentinel block's row: what a tile nothing stores reads, AIR in block 0
 // and SOLID in the last. A free block's rows are far air as well, which is
 // what a claimed block's hold until its page is resident.
 let sentinel=k<512u||k>=RO_KEYS-512u;
 if(sentinel||!live){roPhi[k]=RO_FAR;}
 ${record ? "if(sentinel){roRecord[k]=vec4f(select(0.0,1.0,k<512u));}" : ""}
}
`;
}

const PASS: GPUComputePassDescriptor = { label: "Uniform pressure root owners" };

export class UniformMixedRootOwners {
  readonly label = "Uniform pressure root owners";
  private pipeline?: GPUComputePipeline;
  private group?: GPUBindGroup;
  private readonly keyGroups: readonly [number, number];
  /** ownership: the pressure root's. phi: the root phi, by tile key. record:
   * the root record (a vec4 a row, by tile key), where the scene has one. */
  constructor(private readonly device: GPUDevice, private readonly ownership: UniformMixedOwnership, private readonly phi: GPUBufferBinding, private readonly record?: GPUBufferBinding) {
    const c = ownership.capacity, limit = device.limits.maxComputeWorkgroupsPerDimension;
    if (c.fineTiles !== 0) throw new Error(`Uniform pressure root owners: the ownership holds ${c.fineTiles} h tiles; the root is all-4h`);
    const rows = c.tiles, slots = uniformFaceTileSlots(c.address.blocks);
    if ((phi.size ?? phi.buffer.size - (phi.offset ?? 0)) < 4 * rows) throw new Error(`Uniform pressure root owners: the root phi holds fewer than the ${rows} tile keys' rows`);
    if (record && (record.size ?? record.buffer.size - (record.offset ?? 0)) < 16 * (rows + slots)) throw new Error(`Uniform pressure root owners: the root record holds fewer than the ${rows} tile keys' rows and ${slots} wall slots`);
    const grid = (groups: number): readonly [number, number] => { const x = Math.min(groups, limit); return [x, Math.ceil(groups / x)]; };
    this.keyGroups = grid(Math.ceil(c.tiles / 64));
  }
  async initialize(): Promise<void> {
    if (this.pipeline) throw new Error("Uniform pressure root owners is already initialized");
    const device = this.device, c = this.ownership.capacity, C = GPUShaderStage.COMPUTE, storage = { type: "storage" as const };
    const module = device.createShaderModule({ label: this.label, code: uniformMixedRootOwnersWGSL(c.address, c.lattice, !!this.record) });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`${this.label}: ${errors.map(m => `${m.lineNum}: ${m.message}`).join("\n")}`);
    const layout = device.createBindGroupLayout({ label: this.label, entries: [0, 1, 2, ...(this.record ? [3] : [])].map(binding => ({ binding, visibility: C, buffer: storage })) });
    this.group = device.createBindGroup({ label: this.label, layout, entries: [
      { binding: 0, resource: { buffer: this.ownership.presentation.buffer } }, { binding: 1, resource: { buffer: this.ownership.support } },
      { binding: 2, resource: this.phi }, ...(this.record ? [{ binding: 3, resource: this.record }] : [])] });
    this.pipeline = await device.createComputePipelineAsync({ label: `${this.label} roKeys`, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint: "roKeys" } });
  }
  /** After the step's page commit, before the first root reader (the split
   * authority). */
  encode(encoder: GPUCommandEncoder): void {
    const p = this.pipeline;
    if (!p) throw new Error("Uniform pressure root owners is not initialized");
    const pass = encoder.beginComputePass(PASS);
    pass.setBindGroup(0, this.group!);
    pass.setPipeline(p);pass.dispatchWorkgroups(this.keyGroups[0], this.keyGroups[1]);
    pass.end();
  }
}
