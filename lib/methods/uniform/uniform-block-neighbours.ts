import { uniformWindowAddressWGSL } from "./uniform-window-address";
import { UNIFORM_PAGE_TABLE_WORDS, UNIFORM_PAGE_WINDOW, uniformBlockNeighbourLayout } from "./uniform-lattice-address";
import type { UniformPagePoolMirror } from "./uniform-page-pool";

/** The block neighbour rows and the wall-face list of a topology's tail
 * (uniformBlockNeighbourLayout), rebuilt on the GPU from the page table and
 * the block records the page pool mirrors there.
 *
 * bnRows, one lane a block: the 27 table words around the block's page, a
 * page outside the window SOLID; a block with no page (free, quarantined, a
 * sentinel) takes one word throughout. bnWalls, one workgroup over every
 * block, each lane a run of them: counts each stored block's faces whose row
 * word is SOLID, scans the counts and writes 6 b + f in ascending order,
 * then the count. It reads the rows bnRows has just written, so it follows
 * it in the pass.
 *
 * Every row and the whole list are rewritten after every commit, whatever
 * the commit changed: a claim or a retire changes a word in up to 27 rows
 * and moves the list, and nothing the host could read back names them. Both
 * launches are fixed by the pool's blocks. The list's order follows block
 * number, so it is the same in every topology and stays put while the stored
 * set does; which block a claimed page takes is the pool's choice, so
 * storage is addressed by 6 b + f, never by a face's place in the list.
 *
 * A topology derives its own rows from its own copy of the table: one
 * storage binding a launch, and no stage of the commit gains one.
 *
 * table: the buffer's word that holds the page table's first. */
export function uniformBlockNeighboursWGSL(blocks: number, table: number): string {
  const l = uniformBlockNeighbourLayout(blocks), w = UNIFORM_PAGE_WINDOW;
  return /* wgsl */ `
// A topology: from BN_TABLE, the page table, the block records, then the rows, the wall count and the wall list.
@group(0) @binding(0) var<storage,read_write> bnTail:array<u32>;
const BN_BLOCKS:u32=${blocks}u;const BN_TABLE:u32=${table}u;
const BN_RECORD:u32=${table + UNIFORM_PAGE_TABLE_WORDS}u;const BN_ROWS:u32=${table + l.rows}u;const BN_WALL_COUNT:u32=${table + l.wallCount}u;const BN_WALLS:u32=${table + l.walls}u;
const BN_HOMES:u32=${table + l.homes}u;
const BN_FREE:u32=0xffffffffu;const BN_SOLID:u32=0xffffffffu;
${uniformWindowAddressWGSL("bnTail","BN_TABLE","BN_BLOCKS")}
@compute @workgroup_size(64) fn bnRows(@builtin(global_invocation_id) gid:vec3u){
 let b=gid.x;if(b>=BN_BLOCKS){return;}
 let record=umBlockWord(b);let row=BN_ROWS+27u*b;
 // No page: AIR all round, or SOLID for the block that stands for it.
 if(record==BN_FREE){let word=select(0u,BN_SOLID,b==BN_BLOCKS-1u);for(var i=0u;i<27u;i++){bnTail[row+i]=word;}return;}
 let page=vec3i(vec3u(record&63u,(record>>6u)&63u,(record>>12u)&63u));
 for(var i=0u;i<27u;i++){
  let q=page+vec3i(vec3u(i%3u,(i/3u)%3u,i/9u))-vec3i(1);
  let word=umPageWord(q);
  bnTail[row+i]=word;
 }
 // Canonical homes for the page's 27 interior/face/edge/corner families.
 // This lane owns its row, so no inter-workgroup dependency is introduced.
 for(var f=0u;f<27u;f++){
  let side=vec3u(f%3u,(f/3u)%3u,f/9u);let own=select(vec3i(0),vec3i(-1),side==vec3u(0));
  var result=0u;
  for(var k=0u;k<8u;k++){
   let step=vec3u(k&1u,(k>>1u)&1u,k>>2u);
   if(any((step!=vec3u(0))&(side==vec3u(1)))){continue;}
   let d=own+vec3i(step);let home=bnTail[row+u32(d.x+3*d.y+9*d.z+13)];
   if(home>0u&&home<BN_BLOCKS-1u){
    let code=select(vec3u(d),vec3u(2u),d==vec3i(-1));
    result=home|(code.x<<26u)|(code.y<<28u)|(code.z<<30u);break;
   }
  }
  bnTail[BN_HOMES+27u*b+f]=result;
 }
}
const BN_LANES:u32=256u;const BN_PER:u32=${Math.ceil(blocks / 256)}u;
var<workgroup> bnScan:array<u32,256>;
// Face f = 2 axis + side of block b meets a SOLID page: the row word 1, 3 or 9 below or above its own.
fn bnWall(b:u32,f:u32)->bool{
 let stride=select(select(9u,3u,(f>>1u)==1u),1u,(f>>1u)==0u);
 return bnTail[BN_ROWS+27u*b+select(13u-stride,13u+stride,(f&1u)==1u)]==BN_SOLID;
}
// Neither sentinel holds a page, and both their records are free.
fn bnStored(b:u32)->bool{return bnTail[BN_RECORD+b]!=BN_FREE;}
@compute @workgroup_size(256) fn bnWalls(@builtin(local_invocation_index) lane:u32){
 let first=min(lane*BN_PER,BN_BLOCKS);let last=min(first+BN_PER,BN_BLOCKS);
 var count=0u;
 for(var b=first;b<last;b++){
  if(!bnStored(b)){continue;}
  for(var f=0u;f<6u;f++){if(bnWall(b,f)){count+=1u;}}
 }
 bnScan[lane]=count;workgroupBarrier();
 // Inclusive Hillis-Steele scan over the lanes.
 for(var stride=1u;stride<BN_LANES;stride*=2u){
  var add=0u;if(lane>=stride){add=bnScan[lane-stride];}
  workgroupBarrier();bnScan[lane]+=add;workgroupBarrier();
 }
 var at=bnScan[lane]-count;
 for(var b=first;b<last;b++){
  if(!bnStored(b)){continue;}
  for(var f=0u;f<6u;f++){if(bnWall(b,f)){bnTail[BN_WALLS+at]=6u*b+f;at+=1u;}}
 }
 if(lane==BN_LANES-1u){bnTail[BN_WALL_COUNT]=at;}
}
`;
}

const PASS: GPUComputePassDescriptor = { label: "Uniform block neighbours" };

export class UniformBlockNeighbours {
  readonly label = "Uniform block neighbours";
  /** A launch pair a topology, in the tails' order. */
  private launches?: readonly { readonly group: GPUBindGroup; readonly rows: GPUComputePipeline; readonly walls: GPUComputePipeline }[];
  private readonly rowGroups: number;
  private started = false;
  /** blocks: the pool's, the two sentinels included. tails: where each
   * topology holds the page table (a page pool mirror: its buffer and the
   * table's first word), the rows and the list following its block records. */
  constructor(private readonly device: GPUDevice, private readonly blocks: number, private readonly tails: readonly UniformPagePoolMirror[]) {
    if (!Number.isInteger(blocks) || blocks < 3) throw new RangeError(`Uniform block neighbours: a pool has three blocks at least, not ${blocks}`);
    const l = uniformBlockNeighbourLayout(blocks);
    tails.forEach((t, i) => {
      if (!Number.isSafeInteger(t.offset) || t.offset < 0 || 4 * (t.offset + l.end) > t.buffer.size) throw new RangeError(`Uniform block neighbours: tail ${i} holds ${t.buffer.size / 4 - t.offset} words; the page table, ${blocks} block records, their rows and wall faces take ${l.end}`);
      if (!(t.buffer.usage & GPUBufferUsage.STORAGE)) throw new Error(`Uniform block neighbours: tail ${i}: its buffer needs STORAGE usage`);
      if (tails.findIndex(o => o.buffer === t.buffer) !== i) throw new Error(`Uniform block neighbours: tail ${i}: its buffer is listed already`);
    });
    this.rowGroups = Math.ceil(blocks / 64);
  }
  async initialize(): Promise<void> {
    if (this.started) throw new Error("Uniform block neighbours is already initialized");
    this.started = true;
    const device = this.device;
    const layout = device.createBindGroupLayout({ label: this.label, entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    // The table's word is shader text: tails at one word (the frame's topologies) share a module and its pipelines.
    const built = new Map<number, Promise<readonly [GPUComputePipeline, GPUComputePipeline]>>();
    const build = async (table: number): Promise<readonly [GPUComputePipeline, GPUComputePipeline]> => {
      const module = device.createShaderModule({ label: this.label, code: uniformBlockNeighboursWGSL(this.blocks, table) });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (errors.length) throw new Error(`${this.label}: ${errors.map(m => `${m.lineNum}: ${m.message}`).join("\n")}`);
      const pipeline = (entryPoint: string) => device.createComputePipelineAsync({ label: `${this.label} ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } });
      return Promise.all([pipeline("bnRows"), pipeline("bnWalls")]);
    };
    this.launches = await Promise.all(this.tails.map(async t => {
      let pipelines = built.get(t.offset);
      if (!pipelines) built.set(t.offset, pipelines = build(t.offset));
      const [rows, walls] = await pipelines;
      return { group: device.createBindGroup({ label: this.label, layout, entries: [{ binding: 0, resource: { buffer: t.buffer } }] }), rows, walls };
    }));
  }
  /** After the page pool's commit, which leaves every mirror's table and
   * block records current, and before the first reader of the new stored
   * set. */
  encode(encoder: GPUCommandEncoder): void {
    const launches = this.launches;
    if (!launches) throw new Error("Uniform block neighbours is not initialized");
    const pass = encoder.beginComputePass(PASS);
    for (let i = 0; i < launches.length; i++) {
      const l = launches[i]!;
      pass.setBindGroup(0, l.group);
      pass.setPipeline(l.rows);pass.dispatchWorkgroups(this.rowGroups);
      pass.setPipeline(l.walls);pass.dispatchWorkgroups(1);
    }
    pass.end();
  }
}
