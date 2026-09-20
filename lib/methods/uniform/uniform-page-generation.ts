import type { UniformPageEdge } from "./uniform-page-layout";

/** GPU-owned residency transaction. Requests are the complete desired support set,
 * produced by GPU operators (or uploaded once for initialization). This mechanism
 * does not decide fluid support, nor authorize retirement of a mass-bearing page.
 * Those policies must finish before prepare. It never reads counts on the host.
 */
export interface UniformPageGenerationOptions {
  capacity: number;
  requestCapacity: number;
  edge: UniformPageEdge;
  /** Canonically owned cell/vertex/face fields, interleaved per lattice coordinate. */
  initialCell: readonly number[];
}
export const PAGE_GENERATION_HEADER = 16;
export const PAGE_GENERATION_RECORD = 16;
export const PAGE_GENERATION_FAULT = { none: 0, requests: 1, capacity: 2 } as const;

export function uniformPageGenerationLayout(options: UniformPageGenerationOptions) {
  const { capacity, requestCapacity, edge, initialCell } = options;
  if (!Number.isSafeInteger(capacity) || capacity < 1 || capacity > 4096
    || !Number.isSafeInteger(requestCapacity) || requestCapacity < 1 || requestCapacity > 65536
    || (edge !== 16 && edge !== 32) || !initialCell.length || initialCell.length > 32
    || !initialCell.every(n => Number.isFinite(Math.fround(n)))) throw new RangeError("Invalid GPU page pool budget");
  const directoryCapacity = 2 ** Math.ceil(Math.log2(2 * capacity));
  const activeBase = PAGE_GENERATION_HEADER + capacity * PAGE_GENERATION_RECORD;
  const freshBase = activeBase + capacity;
  const directoryBase = freshBase + capacity;
  return { directoryCapacity, activeBase, freshBase, directoryBase,
    metadataWords: directoryBase + directoryCapacity * 4,
    requestWords: 4 + requestCapacity * 4,
    fieldWords: capacity * edge ** 3 * initialCell.length };
}

export function uniformPageGenerationShader(options: UniformPageGenerationOptions): string {
  const l = uniformPageGenerationLayout(options);
  // Bitcasts preserve all finite f32 values, including signed zero.
  const initial = new Uint32Array(new Float32Array(options.initialCell).buffer);
  return /* wgsl */ `
const CAP:u32=${options.capacity}u;
const REQUEST_CAP:u32=${options.requestCapacity}u;
const DIRECTORY_CAP:u32=${l.directoryCapacity}u;
const ACTIVE:u32=${l.activeBase}u;
const FRESH:u32=${l.freshBase}u;
const DIRECTORY:u32=${l.directoryBase}u;
const WORDS:u32=${l.metadataWords}u;
const PAGE_CELLS:u32=${options.edge ** 3}u;
const FIELDS:u32=${initial.length}u;
const MISSING:u32=0xffffffffu;
const INITIAL=array<u32,${initial.length}>(${Array.from(initial, n => `${n}u`).join(",")});
@group(0) @binding(0) var<storage,read_write> accepted:array<u32>;
@group(0) @binding(1) var<storage,read_write> candidate:array<u32>;
// count, reserved x3, then signed xyz and nonzero role mask per request.
@group(0) @binding(2) var<storage,read> requests:array<u32>;
@group(0) @binding(3) var<storage,read_write> fields:array<u32>;
fn record(slot:u32)->u32{return 16u+16u*slot;}
fn hash(q:vec3u)->u32{
 var h=0x811c9dc5u;let v=vec4u(q,0u);
 for(var a=0u;a<4u;a++){h=(h^v[a])*0x01000193u;h=h^(h>>16u);}
 return (h|1u)&(DIRECTORY_CAP-1u);
}

fn oldSlot(q:vec3u)->u32{
 if(accepted[1u]==0u){return MISSING;}
 var bucket=hash(q);
 for(var probe=0u;probe<DIRECTORY_CAP;probe++){
  let at=DIRECTORY+4u*bucket;let slot=accepted[at+3u];
  if(slot==MISSING){return MISSING;}
  if(all(vec3u(accepted[at],accepted[at+1u],accepted[at+2u])==q)){return slot;}
  bucket=(bucket+1u)&(DIRECTORY_CAP-1u);
 }return MISSING;
}
fn newBucket(q:vec3u)->u32{
 var bucket=hash(q);
 for(var probe=0u;probe<DIRECTORY_CAP;probe++){
  let at=DIRECTORY+4u*bucket;
  if(candidate[at+3u]==MISSING || all(vec3u(candidate[at],candidate[at+1u],candidate[at+2u])==q)){return bucket;}
  bucket=(bucket+1u)&(DIRECTORY_CAP-1u);
 }return MISSING;
}
// Serial topology builder is deliberately bounded by the pool budget. Field work
// is parallel and indirect. This avoids cross-workgroup publication spinlocks;
// a parallel sort/unique builder can replace it without changing the transaction ABI.
@compute @workgroup_size(1) fn plan(){
 for(var i=0u;i<WORDS;i++){candidate[i]=0u;}
 for(var b=0u;b<DIRECTORY_CAP;b++){candidate[DIRECTORY+4u*b+3u]=MISSING;}
 if(requests[0u]>REQUEST_CAP){candidate[4u]=1u;return;}
 var freeCursor=0u;
 for(var i=0u;i<requests[0u];i++){
  let at=4u+4u*i;let roles=requests[at+3u];if(roles==0u){continue;}
  let q=vec3u(requests[at],requests[at+1u],requests[at+2u]);
  let bucket=newBucket(q);
  if(bucket==MISSING){candidate[4u]=2u;return;}
  let entry=DIRECTORY+4u*bucket;var slot=candidate[entry+3u];
  if(slot!=MISSING){candidate[record(slot)+3u]|=roles;continue;}
  slot=oldSlot(q);
  if(slot==MISSING){
   // Never recycle a slot belonging to the accepted generation in this transition.
   loop {
    if(freeCursor>=CAP){candidate[4u]=2u;return;}
    let r=record(freeCursor);
    if(accepted[r+10u]==0u && candidate[r+10u]==0u){break;}
    freeCursor++;
   }
   slot=freeCursor;freeCursor++;
   candidate[FRESH+candidate[2u]]=slot;candidate[2u]++;
  }
  candidate[entry]=q.x;candidate[entry+1u]=q.y;candidate[entry+2u]=q.z;candidate[entry+3u]=slot;
  let r=record(slot);candidate[r]=q.x;candidate[r+1u]=q.y;candidate[r+2u]=q.z;
  candidate[r+3u]=roles;candidate[r+10u]=1u;
  candidate[ACTIVE+candidate[1u]]=slot;candidate[1u]++;
 }
 for(var slot=0u;slot<CAP;slot++){
  let r=record(slot);
  if(accepted[r+10u]!=0u && candidate[r+10u]==0u){candidate[3u]++;}
  if(candidate[r+10u]==0u){continue;}
  let q=bitcast<vec3i>(vec3u(candidate[r],candidate[r+1u],candidate[r+2u]));
  for(var face=0u;face<6u;face++){
   let axis=face/2u;let delta=select(-1,1,(face&1u)!=0u);var n=q;
   candidate[r+4u+face]=MISSING;
   if((delta<0 && q[axis]==(-2147483647-1)) || (delta>0 && q[axis]==2147483647)){continue;}
   n[axis]+=delta;let bucket=newBucket(bitcast<vec3u>(n));
   if(bucket!=MISSING){candidate[r+4u+face]=candidate[DIRECTORY+4u*bucket+3u];}
  }
 }
 candidate[0u]=accepted[0u]+1u;
 // Header 8: cell dispatch (one y workgroup per resident page).
 candidate[8u]=(PAGE_CELLS+63u)/64u;candidate[9u]=candidate[1u];candidate[10u]=1u;
 // Header 12: initialize new pages only. Overflow paths leave this zero.
 candidate[12u]=(PAGE_CELLS+63u)/64u;candidate[13u]=candidate[2u];candidate[14u]=1u;
}
@compute @workgroup_size(64) fn initialize(@builtin(global_invocation_id) id:vec3u){
 if(candidate[4u]!=0u || id.y>=candidate[2u] || id.x>=PAGE_CELLS){return;}
 let slot=candidate[FRESH+id.y];let at=(slot*PAGE_CELLS+id.x)*FIELDS;
 for(var f=0u;f<FIELDS;f++){fields[at+f]=INITIAL[f];}
}
@compute @workgroup_size(64) fn publish(@builtin(global_invocation_id) id:vec3u){
 if(candidate[4u]==0u && id.x<WORDS){accepted[id.x]=candidate[id.x];}
}
`;
}

/** Field access for consumers binding `accepted:array<u32>` and
 * `fields:array<u32>`. Addresses are page-coordinate + local-coordinate pairs;
 * no world-sized linear index or conversion to imprecise world-space f32.
 * Missing support is explicit. The operator decides whether it is valid ambient
 * air or a fault; this layer never substitutes a solid wall or zero field value.
 */
export function uniformPageFieldAccessWGSL(options: UniformPageGenerationOptions): string {
  const l = uniformPageGenerationLayout(options);
  return /* wgsl */ `
const PAGE_FIELD_MISSING:u32=0xffffffffu;
fn pageFieldSlot(q:vec3i)->u32{
 if(accepted[1u]==0u){return PAGE_FIELD_MISSING;}
 var h=0x811c9dc5u;let words=vec4u(bitcast<vec3u>(q),0u);
 for(var a=0u;a<4u;a++){h=(h^words[a])*0x01000193u;h=h^(h>>16u);}
 var bucket=(h|1u)&${l.directoryCapacity - 1}u;
 for(var probe=0u;probe<${l.directoryCapacity}u;probe++){
  let at=${l.directoryBase}u+4u*bucket;let slot=accepted[at+3u];
  if(slot==PAGE_FIELD_MISSING){return slot;}
  if(all(bitcast<vec3i>(vec3u(accepted[at],accepted[at+1u],accepted[at+2u]))==q)){return slot;}
  bucket=(bucket+1u)&${l.directoryCapacity - 1}u;
 }return PAGE_FIELD_MISSING;
}
fn pageFieldAddress(page:vec3i,local:vec3i,field:u32)->u32{
 if(field>=${options.initialCell.length}u){return PAGE_FIELD_MISSING;}
 let edge=${options.edge};
 // WGSL division truncates toward zero; correct it to floor for negative cells.
 let quotient=local/edge;let remainder=local%edge;
 let offset=quotient-select(vec3i(0),vec3i(1),remainder<vec3i(0));
 let cell=remainder+select(vec3i(0),vec3i(edge),remainder<vec3i(0));
 var q=page;
 for(var a=0u;a<3u;a++){
  if(offset[a]>0 && page[a]>2147483647-offset[a]){return PAGE_FIELD_MISSING;}
  if(offset[a]<0 && page[a]<(-2147483647-1)-offset[a]){return PAGE_FIELD_MISSING;}
  q[a]+=offset[a];
 }
 let slot=pageFieldSlot(q);if(slot==PAGE_FIELD_MISSING){return slot;}
 let index=u32(cell.x+edge*(cell.y+edge*cell.z));
 return (slot*${options.edge ** 3}u+index)*${options.initialCell.length}u+field;
}
`;
}

/** A residency transaction, not yet a complete fluid-step transaction. Consumers
 * must preserve accepted field values until their own validation/commit succeeds.
 * Calls must be queue-ordered: all old consumers before prepare/publish, all new
 * consumers after publish. A retired slot is only reusable by a later transaction.
 */
export class UniformPageGeneration {
  readonly layout: ReturnType<typeof uniformPageGenerationLayout>;
  readonly accepted: GPUBuffer;
  readonly candidate: GPUBuffer;
  readonly requests: GPUBuffer;
  readonly fields: GPUBuffer;
  private readonly initializationDispatch: GPUBuffer;
  readonly allocatedBytes: number;
  private readonly bindGroup: GPUBindGroup;
  private constructor(device: GPUDevice, readonly options: UniformPageGenerationOptions,
    private readonly pipelines: readonly GPUComputePipeline[], bindGroupLayout: GPUBindGroupLayout) {
    this.layout = uniformPageGenerationLayout(options);
    const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    const make = (label: string, words: number, usage = storage) => device.createBuffer({ label, size: words * 4, usage });
    this.accepted = make("uniform accepted page generation", this.layout.metadataWords, storage | GPUBufferUsage.INDIRECT);
    this.candidate = make("uniform candidate page generation", this.layout.metadataWords, storage | GPUBufferUsage.INDIRECT);
    this.requests = make("uniform desired page support", this.layout.requestWords);
    this.fields = make("uniform persistent page fields", this.layout.fieldWords);
    this.initializationDispatch = make("uniform page initialization dispatch", 3, GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST);
    this.allocatedBytes = 12 + 4 * (2 * this.layout.metadataWords + this.layout.requestWords + this.layout.fieldWords);
    this.bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries:
      [this.accepted, this.candidate, this.requests, this.fields].map((buffer, binding) => ({ binding, resource: { buffer } })) });
  }
  static async create(device: GPUDevice, options: UniformPageGenerationOptions): Promise<UniformPageGeneration> {
    options = Object.freeze({ ...options, initialCell: Object.freeze([...options.initialCell]) });
    const layout = uniformPageGenerationLayout(options);
    if (Math.max(layout.fieldWords, layout.metadataWords, layout.requestWords) * 4
      > Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize)) {
      throw new RangeError("Uniform page pool exceeds device buffer budget");
    }
    const bindGroupLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3].map(binding => ({
      binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding === 2 ? "read-only-storage" as const : "storage" as const },
    })) });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const module = device.createShaderModule({ label: "Uniform GPU page residency", code: uniformPageGenerationShader(options) });
    const pipelines = await Promise.all(["plan", "initialize", "publish"].map(entryPoint => device.createComputePipelineAsync({
      label: `uniform pages ${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint },
    })));
    return new UniformPageGeneration(device, options, pipelines, bindGroupLayout);
  }
  encodePrepare(encoder: GPUCommandEncoder): void {
    const plan = encoder.beginComputePass({ label: "uniform page residency plan" });
    plan.setPipeline(this.pipelines[0]!); plan.setBindGroup(0, this.bindGroup); plan.dispatchWorkgroups(1); plan.end();
    // Indirect arguments cannot share a writable storage binding in one pass.
    encoder.copyBufferToBuffer(this.candidate, 12 * 4, this.initializationDispatch, 0, 12);
    const init = encoder.beginComputePass({ label: "uniform new page fields" });
    init.setPipeline(this.pipelines[1]!); init.setBindGroup(0, this.bindGroup);
    init.dispatchWorkgroupsIndirect(this.initializationDispatch, 0); init.end();
  }
  encodePublish(encoder: GPUCommandEncoder): void {
    const pass = encoder.beginComputePass({ label: "uniform page generation commit" });
    pass.setPipeline(this.pipelines[2]!); pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.layout.metadataWords / 64)); pass.end();
  }
  destroy(): void { for (const buffer of [this.accepted, this.candidate, this.requests, this.fields, this.initializationDispatch]) buffer.destroy(); }
}
