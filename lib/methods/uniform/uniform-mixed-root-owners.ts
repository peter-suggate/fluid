import { compileUniformStencil, uniformCompileStencilWGSL } from "./uniform-compiled-topology";
import {
  UNIFORM_KIND, UNIFORM_PAGE_CELLS, UNIFORM_PAGE_TABLE_WORDS, UNIFORM_PAGE_WINDOW, uniformBoxTileIndex, uniformLatticeAddressWGSL,
  uniformTileCoord, uniformTileKey, uniformTileKind, uniformTileLive, uniformTileStored, type UniformLatticeAddress,
} from "./uniform-lattice-address";
import { UNIFORM_MIXED_DETAIL_RING } from "./uniform-mixed-layout";
import type { UniformMixedOwnership } from "./uniform-mixed-ownership";
import { uniformMixedDetailViolationWord, uniformMixedPageCount, uniformMixedResidencyWord } from "./uniform-mixed-topology.wgsl";

/** The pressure root's owners, built on the GPU from the page table every
 * step.
 *
 * The root is all-4h and its owner index is the tile's box index: every
 * per-owner root array (the arena rows, the wall slots after them, the root
 * phi, the root record, the presented pressure and phi) is one box and never
 * moves when a page changes block. What does follow the page set is the
 * topology the root's kernels reach those rows through, which is per tile
 * KEY: a key's tile word (its owner index) and its two stencil words (which
 * of its 26 neighbours are stored). A host-built root names the owners of the
 * seed's blocks only: a block claimed later has none, and a retired one
 * keeps its own.
 *
 * roKeys, one lane a tile key: a live key takes its box index and the
 * all-4h stencil of the stored neighbours it has now; every other key (a
 * free block's, a sentinel block's) takes zero words, as the host seed held
 * them. A claimed block's keys, a retired block's keys and both sides of
 * every changed page face are therefore right in the step of the commit,
 * with no list of what changed. Bit 0 of a live key's second stencil word
 * (the detail ring, written by the detail storage's ring kernels) is kept.
 *
 * roPages, one group a 16³-cell residency page, a lane a tile: a box tile
 * with nothing stored is a row no key reaches, so its row is written here by
 * box index. Its phi is far air, its record is open with V=1 on every face
 * where the page is AIR and closed with V=0 where it is SOLID, its wall
 * slots' records (a tile on a face of the box) are closed, and the root's
 * copy of its residency flag is cleared, so the root cycles' setup gives the
 * row and its slots the zero pressure, zero RHS and bound of an absent page
 * whatever the census certified. A tile's rows are so the same the step its
 * page retires as if the page had never been stored.
 *
 * Both launches are fixed by capacity (tile keys, residency pages) and strided
 * by nothing: no count is read back and none is needed. */

/** Sticky bit of the root support's violation word (beside the detail
 * storage's, which the frame receipt already carries): a block record names a
 * page whose table word is another block. Two keys would hold one owner. */
export const UNIFORM_MIXED_ROOT_FATAL = 0x10000;
export const UNIFORM_MIXED_ROOT_FATAL_MESSAGE = "Uniform pressure root: a block record names a page the page table gives to another block (two tile keys would hold one root owner); the root's owners were withheld for that block";
/** A box tile with nothing stored holds this phi, in cell sizes (the largest
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
      if (uniformTileKey(address, c[0], c[1], c[2]) !== k || !uniformTileStored(address, c[0], c[1], c[2])) fatal |= UNIFORM_MIXED_ROOT_FATAL;
      else {
        let valid = 0;
        for (let i = 0; i < 27; i++) {
          if (uniformTileStored(address, c[0] + i % 3 - 1, c[1] + Math.floor(i / 3) % 3 - 1, c[2] + Math.floor(i / 9) - 1)) valid |= 1 << i;
        }
        word = uniformBoxTileIndex(address, c[0], c[1], c[2]);s0 = (4 << 27) >>> 0;
        s1 = ((4 << 27) | compileUniformStencil(s0, valid) | ((held?.[2 * n + 2 * k + 1] ?? 0) & UNIFORM_MIXED_DETAIL_RING)) >>> 0;
      }
    }
    words[k] = word;words[2 * n + 2 * k] = s0;words[2 * n + 2 * k + 1] = s1;words[4 * n + k] = 0;
  }
  return { words, fatal };
}

/** A wall slot's row (umBoundaryIndex of the all-4h layout): the box's rows,
 * then the two x planes, the two y planes and the two z planes. */
export function uniformMixedRootSlot(address: UniformLatticeAddress, x: number, y: number, z: number, axis: number, side: number): number {
  const [tx, ty, tz] = address.tiles, rows = address.boxTiles;
  if (axis === 0) return rows + side * ty * tz + y + ty * z;
  if (axis === 1) return rows + 2 * ty * tz + side * tx * tz + x + tx * z;
  return rows + 2 * (ty * tz + tx * tz) + side * tx * ty + x + tx * y;
}

/** roPages on the host. flags: the root's residency flags, one per 16³-cell
 * page (cleared in place where nothing is stored); phi and record (four
 * floats a row) by box index, the record's wall slots after its box rows.
 * far: UNIFORM_MIXED_ROOT_FAR_CELLS cell sizes, in metres. Returns the box
 * rows it wrote. */
export function uniformMixedRootUnstoredRows(address: UniformLatticeAddress, flags: Uint32Array, phi: Float32Array, record: Float32Array | undefined, far: number): number {
  const [tx, ty, tz] = address.tiles, px = Math.ceil(tx / 4), py = Math.ceil(ty / 4), pz = Math.ceil(tz / 4);
  let rows = 0;
  for (let page = 0; page < px * py * pz; page++) for (let lane = 0; lane < 64; lane++) {
    const x = 4 * (page % px) + lane % 4, y = 4 * (Math.floor(page / px) % py) + Math.floor(lane / 4) % 4, z = 4 * Math.floor(page / (px * py)) + Math.floor(lane / 16);
    if (x >= tx || y >= ty || z >= tz) continue;
    const kind = uniformTileKind(address, x, y, z);
    if (kind === UNIFORM_KIND.resident) continue;
    if (lane === 0) flags[page] = 0;
    const i = uniformBoxTileIndex(address, x, y, z), v = kind === UNIFORM_KIND.air ? 1 : 0;
    phi[i] = far;rows++;
    if (!record) continue;
    record.fill(v, 4 * i, 4 * i + 4);
    const c = [x, y, z], t = address.tiles;
    for (let axis = 0; axis < 3; axis++) for (let side = 0; side < 2; side++) {
      if (c[axis] !== (side ? t[axis]! - 1 : 0)) continue;
      const s = uniformMixedRootSlot(address, x, y, z, axis, side);
      record.fill(0, 4 * s, 4 * s + 4);
    }
  }
  return rows;
}

const float = (x: number) => Number.isInteger(x) ? x.toFixed(1) : String(x);

/** The kernels. record: the root record (a vec4 a row) is bound, at binding 3. */
export function uniformMixedRootOwnersWGSL(address: UniformLatticeAddress, lattice: { readonly dimensions: readonly number[]; readonly cellSize_m: readonly number[] }, record: boolean): string {
  const n = address.tileKeys, cells = lattice.dimensions, pages = uniformMixedPageCount(lattice), table = 5 * n;
  const paged = { blocks: `${n / 512}u`, whole: cells.every(c => c % UNIFORM_PAGE_CELLS === 0) };
  const far = UNIFORM_MIXED_ROOT_FAR_CELLS * Math.max(...lattice.cellSize_m);
  return /* wgsl */ `
const RO_KEYS:u32=${n}u;
const RO_PAGES:u32=${pages}u;
const RO_RESIDENCY:u32=${uniformMixedResidencyWord(n)}u;
const RO_VIOLATION:u32=${uniformMixedDetailViolationWord(n, lattice)}u;
const RO_FAR:f32=${float(far)};
// The root topology: five words a tile key, then the page table and the block records the pool mirrors.
@group(0) @binding(0) var<storage,read_write> roTopology:array<u32>;
// The root support: the residency flags (the transfer's copy of this step's certificate) and the violation word.
@group(0) @binding(1) var<storage,read_write> roSupport:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> roPhi:array<f32>;
${record ? "@group(0) @binding(3) var<storage,read_write> roRecord:array<vec4f>;" : ""}
const UM_D=vec3u(${cells.map(c => `${c}u`).join(",")});const UM_T=UM_D/4u;
fn umPageWord(page:vec3i)->u32{let e=vec3u(page)&vec3u(${UNIFORM_PAGE_WINDOW - 1}u);return roTopology[${table}u+(e.x|(e.y<<6u)|(e.z<<12u))];}
fn umBlockWord(b:u32)->u32{return roTopology[${table + UNIFORM_PAGE_TABLE_WORDS}u+b];}
${uniformLatticeAddressWGSL({ cells: "UM_D", tiles: "UM_T", constant: true, paged })}
${uniformCompileStencilWGSL}
@compute @workgroup_size(64) fn roKeys(@builtin(global_invocation_id) gid:vec3u,@builtin(num_workgroups) launched:vec3u){
 let k=gid.x+64u*launched.x*gid.y;if(k>=RO_KEYS){return;}
 var word=0u;var first=0u;var second=0u;
 if(umTileLive(k)){
  let c=vec3i(umTileCoord(k));
  // The record and the table name each other, or two keys would hold one owner.
  if(umTileKey(c)!=k){atomicOr(&roSupport[RO_VIOLATION],${UNIFORM_MIXED_ROOT_FATAL}u);}
  else{
   var valid=0u;
   for(var i=0u;i<27u;i++){
    let q=c+vec3i(vec3u(i%3u,(i/3u)%3u,i/9u))-vec3i(1);
    if(umTileInBox(q)&&umTilePresent(q)){valid|=1u<<i;}
   }
   word=umBoxTileIndex(vec3u(c));first=4u<<27u;
   second=(4u<<27u)|umCompileStencil(first,valid)|(roTopology[2u*RO_KEYS+2u*k+1u]&${UNIFORM_MIXED_DETAIL_RING}u);
  }
 }
 roTopology[k]=word;roTopology[2u*RO_KEYS+2u*k]=first;roTopology[2u*RO_KEYS+2u*k+1u]=second;roTopology[4u*RO_KEYS+k]=0u;
}
const RO_PD:vec3u=(UM_T+vec3u(3u))/4u;
// umBoundaryIndex of the all-4h layout: the box's rows, then its six planes.
fn roSlot(c:vec3u,axis:u32,side:u32)->u32{
 let d=UM_T;let rows=d.x*d.y*d.z;
 if(axis==0u){return rows+side*d.y*d.z+c.y+d.y*c.z;}
 if(axis==1u){return rows+2u*d.y*d.z+side*d.x*d.z+c.x+d.x*c.z;}
 return rows+2u*(d.y*d.z+d.x*d.z)+side*d.x*d.y+c.x+d.x*c.y;
}
@compute @workgroup_size(64) fn roPages(@builtin(workgroup_id) group:vec3u,@builtin(num_workgroups) launched:vec3u,@builtin(local_invocation_index) lane:u32){
 let page=group.x+launched.x*group.y;if(page>=RO_PAGES){return;}
 let c=4u*vec3u(page%RO_PD.x,(page/RO_PD.x)%RO_PD.y,page/(RO_PD.x*RO_PD.y))+vec3u(lane%4u,(lane/4u)%4u,lane/16u);
 if(any(c>=UM_T)){return;}
 let kind=umTileKind(vec3i(c));if(kind==UM_RESIDENT){return;}
 // A residency page lies in one pool page, and its first tile in the box.
 if(lane==0u){atomicStore(&roSupport[RO_RESIDENCY+4u+page],0u);}
 let row=umBoxTileIndex(c);
 roPhi[row]=RO_FAR;
 ${record ? `roRecord[row]=vec4f(select(0.0,1.0,kind==UM_AIR));
 // Its wall slots: closed, as a tile no page ever stored holds them.
 for(var axis=0u;axis<3u;axis++){for(var side=0u;side<2u;side++){
  if(c[axis]==select(0u,UM_T[axis]-1u,side==1u)){roRecord[roSlot(c,axis,side)]=vec4f(0.0);}
 }}` : ""}
}
`;
}

const PASS: GPUComputePassDescriptor = { label: "Uniform pressure root owners" };

export class UniformMixedRootOwners {
  readonly label = "Uniform pressure root owners";
  private pipelines?: { readonly keys: GPUComputePipeline; readonly pages: GPUComputePipeline };
  private group?: GPUBindGroup;
  private readonly keyGroups: readonly [number, number];
  private readonly pageGroups: readonly [number, number];
  /** ownership: the pressure root's. phi: the root phi, by box index. record:
   * the root record (a vec4 a row, by box index), where the scene has one. */
  constructor(private readonly device: GPUDevice, private readonly ownership: UniformMixedOwnership, private readonly phi: GPUBufferBinding, private readonly record?: GPUBufferBinding) {
    const c = ownership.capacity, limit = device.limits.maxComputeWorkgroupsPerDimension;
    if (c.fineTiles !== 0) throw new Error(`Uniform pressure root owners: the ownership holds ${c.fineTiles} h tiles; the root is all-4h`);
    const rows = c.address.boxTiles, t = c.address.tiles, slots = 2 * (t[0] * t[1] + t[0] * t[2] + t[1] * t[2]);
    if ((phi.size ?? phi.buffer.size - (phi.offset ?? 0)) < 4 * rows) throw new Error(`Uniform pressure root owners: the root phi holds fewer than the box's ${rows} rows`);
    if (record && (record.size ?? record.buffer.size - (record.offset ?? 0)) < 16 * (rows + slots)) throw new Error(`Uniform pressure root owners: the root record holds fewer than the box's ${rows} rows and ${slots} wall slots`);
    const grid = (groups: number): readonly [number, number] => { const x = Math.min(groups, limit); return [x, Math.ceil(groups / x)]; };
    this.keyGroups = grid(Math.ceil(c.tiles / 64));this.pageGroups = grid(uniformMixedPageCount(c.lattice));
  }
  async initialize(): Promise<void> {
    if (this.pipelines) throw new Error("Uniform pressure root owners is already initialized");
    const device = this.device, c = this.ownership.capacity, C = GPUShaderStage.COMPUTE, storage = { type: "storage" as const };
    const module = device.createShaderModule({ label: this.label, code: uniformMixedRootOwnersWGSL(c.address, c.lattice, !!this.record) });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`${this.label}: ${errors.map(m => `${m.lineNum}: ${m.message}`).join("\n")}`);
    const layout = device.createBindGroupLayout({ label: this.label, entries: [0, 1, 2, ...(this.record ? [3] : [])].map(binding => ({ binding, visibility: C, buffer: storage })) });
    this.group = device.createBindGroup({ label: this.label, layout, entries: [
      { binding: 0, resource: { buffer: this.ownership.presentation.buffer } }, { binding: 1, resource: { buffer: this.ownership.support } },
      { binding: 2, resource: this.phi }, ...(this.record ? [{ binding: 3, resource: this.record }] : [])] });
    const pipeline = (entryPoint: string) => device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint } });
    const [keys, pages] = await Promise.all([pipeline("roKeys"), pipeline("roPages")]);
    this.pipelines = { keys, pages };
  }
  /** After the step's page commit and the transfer's copy of its residency
   * certificate into the root support, before the first root reader (the
   * split authority). */
  encode(encoder: GPUCommandEncoder): void {
    const p = this.pipelines;
    if (!p) throw new Error("Uniform pressure root owners is not initialized");
    const pass = encoder.beginComputePass(PASS);
    pass.setBindGroup(0, this.group!);
    pass.setPipeline(p.keys);pass.dispatchWorkgroups(this.keyGroups[0], this.keyGroups[1]);
    pass.setPipeline(p.pages);pass.dispatchWorkgroups(this.pageGroups[0], this.pageGroups[1]);
    pass.end();
  }
}
