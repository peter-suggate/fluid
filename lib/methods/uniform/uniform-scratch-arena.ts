/** Shared f32 backing of the pressure solve: the mixed pressure root, then
 * the levels its correction is solved on. Both follow the page pool: rows a
 * block. Hierarchy geometry/coefficient ranges are retained derived state;
 * only pressure setup writes them. Iterates, RHS and bounds remain scratch.
 * Persistent/published textures remain ordinary textures.
 *
 * The scratch the non-pressure stages overlap follows the h-tile capacity
 * and is the mixed frame's own buffer (UniformMixedFrame.stageBytesAt). */
import { planUniformPressureLevels, type UniformPressureLevelWords } from "./pressure-plan";
import type { UniformLatticeAddress } from "./uniform-lattice-address";

/** Allocation-free layout; different resolutions can share backing when their
 * stages are serialized. No field in this layout is persistent simulation state. */
export class UniformScratchLayout {
  readonly byteLength: number;
  /** The mixed pressure root's range (planUniformMixedPressureMemory). */
  readonly rootOffset: number;
  readonly rootBytes: number;
  readonly edgeBytes: number;
  readonly sharpenBaseWords: number;
  readonly conditioningBytes: number;
  /** The correction's levels, 4h first: arena words (planUniformPressureLevels). */
  readonly levels: readonly UniformPressureLevelWords[];
  constructor(readonly address: UniformLatticeAddress, edgeBytes: number, retainDiagnostics=false, rootBytes=0) {
    if (!Number.isSafeInteger(edgeBytes) || edgeBytes < 0 || edgeBytes % 4 !== 0)
      throw new RangeError("Uniform scratch edge bytes must be a nonnegative multiple of four");
    const tiles=address.tileKeys;
    this.sharpenBaseWords=Math.ceil((6*tiles+3)/4)*4;
    // The tile stages' prefix; balance/page records remain beyond it. No
    // per-cell plane: Geometric advances only through the mixed frame, whose
    // pressure phi and solid deposits are its own owner-capacity buffers.
    this.conditioningBytes=Math.ceil((this.sharpenBaseWords+8+tiles)/4)*16;
    if (!Number.isSafeInteger(rootBytes) || rootBytes < 0)
      throw new RangeError("Uniform scratch root bytes must be a nonnegative safe integer");
    // The dense receiver-stencil bytes of the lattice: reported, not allocated.
    this.edgeBytes=edgeBytes;
    this.rootOffset=0;this.rootBytes=Math.ceil(rootBytes/256)*256;
    const plan=planUniformPressureLevels(address.blocks,(this.rootOffset+this.rootBytes)/4);
    this.levels=plan.levels;this.byteLength=plan.end*4;
    if (!Number.isSafeInteger(this.byteLength))
      throw new RangeError("Uniform scratch layout exceeds safe integer addressing");
  }
  /** Where a texture of this label lies in the arena: nowhere, since the
   * arena's fields are rows of blocks and none is a texture's. What
   * UniformTexturePages asks of every texture it creates. */
  offset(_label:string):number|undefined{return undefined;}
}

/** A layout plus owned or borrowed backing. Borrowed views are valid only while
 * their owner is alive, and must never execute concurrently with another view.
 * Every stage must initialize its own scratch after changing layouts. */
export class UniformScratchArena extends UniformScratchLayout {
  readonly buffer: GPUBuffer;
  private readonly ownsBuffer: boolean;
  constructor(device: GPUDevice, address: UniformLatticeAddress, edgeBytes: number,
    retainDiagnostics=false, backing?: GPUBuffer, rootBytes=0) {
    super(address, edgeBytes, retainDiagnostics, rootBytes);
    const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC|GPUBufferUsage.COPY_DST;
    if (backing && (backing.size < this.byteLength || (backing.usage & usage) !== usage))
      throw new RangeError("Shared uniform scratch backing has insufficient capacity or usage");
    this.ownsBuffer=backing===undefined;
    this.buffer=backing ?? device.createBuffer({label:"Uniform shared pressure scratch",size:this.byteLength,usage});
  }
  destroy():void{if(this.ownsBuffer)this.buffer.destroy();}
}

/** Keep the original f32 loads/stores and bounds behavior; only their backing
 * changes. w uses its high bit for scratch, low two bits for component count. */
export function uniformScratchAccessWGSL(name:string, metadata:string, type:string, atomic=false, components?:number):string {
  const uint=type.includes('uint')||type.includes('u32');
  const vector=uint?'vec4u':'vec4f';
  const raw=(i:number)=>atomic?`atomicLoad(&uniformScratch[at+${i}u])`:`uniformScratch[at+${i}u]`;
  const load=(i:number)=>uint?raw(i):`bitcast<f32>(${raw(i)})`;
  const store=(i:number)=>uint?`value[${i}]`:`bitcast<u32>(value[${i}])`;
  const address=`let fieldLayout=${metadata};let count=${components?`${components}u`:"select(1u,4u,(fieldLayout.w&3u)==3u)"};let at=(fieldLayout.w&0x7ffffffcu)+(u32(p.x)+fieldLayout.x*(u32(p.y)+fieldLayout.y*u32(p.z)))*count;`;
  const assignment=(i:number)=>atomic?`atomicStore(&uniformScratch[at+${i}u],${store(i)});`:`uniformScratch[at+${i}u]=${store(i)};`;
  const guard=`any(p<vec3i(0))||any(p>=vec3i(${metadata}.xyz))`;
  return type.startsWith('texture_storage')
    ? `if((${metadata}.w&0x80000000u)!=0u){if(${guard}){return;}${address}${assignment(0)}if(count==4u){${[1,2,3].map(assignment).join('')}}return;}`
    : `if((${metadata}.w&0x80000000u)!=0u){if(${guard}){return ${vector}(0);}${address}if(count==1u){return ${vector}(${load(0)},0,0,1);}return ${vector}(${[0,1,2,3].map(load).join(',')});}`;
}
