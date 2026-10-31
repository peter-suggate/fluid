import { SCULPT_META, SCULPT_TABLE, SCULPT_VALUES, SCULPT_PAGE_SAMPLES,
  planSculptBake, type SculptBakePlan } from "./svo-sculpt-substrate";
import { PublishedWords } from "./svo-table-upload";

/** One workgroup per page row. The existing field is the starting point for
 * append edits; history is consulted only to restore a page on undo/load. */
export const sculptBakeWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage,read_write> arena:array<u32>;
@group(0) @binding(1) var<storage,read> jobs:array<u32>;
@compute @workgroup_size(64) fn bake(@builtin(global_invocation_id) id:vec3u){
  let job=id.y;let sample=id.x;if(job>=jobs[0]||sample>=${SCULPT_PAGE_SAMPLES}u){return;}
  let base=jobs[1];let at=4u+job*8u;let slot=jobs[at];let reset=jobs[at+1u];
  let coord=bitcast<vec3i>(vec3u(jobs[at+2u],jobs[at+3u],jobs[at+4u]));
  let h=bitcast<vec3f>(vec3u(arena[base+${SCULPT_META}u],arena[base+${SCULPT_META + 1}u],arena[base+${SCULPT_META + 2}u]));
  let band=bitcast<f32>(arena[base+${SCULPT_META + 3}u]);
  let origin=bitcast<vec3f>(vec3u(arena[base+${SCULPT_META + 4}u],arena[base+${SCULPT_META + 5}u],arena[base+${SCULPT_META + 6}u]));
  let point=origin+(vec3f(coord)*8.0+vec3f(0.5)+vec3f(f32(sample%9u),f32((sample/9u)%9u),f32(sample/81u)))*h;
  let address=base+${SCULPT_VALUES}u+slot*${SCULPT_PAGE_SAMPLES}u+sample;
  var distance=band;if(reset==0u){distance=bitcast<f32>(arena[address]);}
  for(var i=0u;i<jobs[at+6u];i++){
    let record=jobs[at+5u]+i*8u;
    let lo=bitcast<vec3f>(vec3u(jobs[record],jobs[record+1u],jobs[record+2u]));
    let hi=bitcast<vec3f>(vec3u(jobs[record+4u],jobs[record+5u],jobs[record+6u]));
    let r=.5*(hi-lo);let p=point-.5*(hi+lo);var d=0.0;
    if(jobs[record+3u]==2u){d=(length(p/r)-1.0)*min(r.x,min(r.y,r.z));}
    else{let q=abs(p)-r;d=length(max(q,vec3f(0)))+min(max(q.x,max(q.y,q.z)),0.0);}
    distance=min(distance,d);
  }
  arena[address]=bitcast<u32>(distance);
}
`;

export class WebgpuSculptSubstrate {
  private plan?: SculptBakePlan;
  private pending?: SculptBakePlan;
  private validated?: { base?: SculptBakePlan; next: SculptBakePlan };
  private pipeline?: GPUComputePipeline;
  private arena?: GPUBuffer;
  private offset = 0;
  private readonly proxies = new PublishedWords();
  private readonly directory = new PublishedWords();
  private readonly metadata = new PublishedWords();
  private initialization?: Promise<void>;
  private input?: GPUBuffer;
  private group?: GPUBindGroup;
  private groupArena?: GPUBuffer;
  /** Useful in diagnostics: sampling cost depends on pages, never brush count. */
  lastUpdate = { pages: 0, records: 0, samples: 0 };
  constructor(private readonly device: GPUDevice) {}
  get bakePlan(): SculptBakePlan | undefined { return this.pending ?? this.plan; }
  get proxyCount(): number { return (this.pending ?? this.plan)?.proxies[0] ?? 0; }
  get pageCount(): number { return (this.pending ?? this.plan)?.pages.size ?? 0; }
  validate(words: Uint32Array): void {
    if (words === this.plan?.words || (this.validated?.base === this.plan && this.validated?.next.words === words)) return;
    this.validated = { base: this.plan, next: planSculptBake(this.plan, words) };
  }
  initialize(): Promise<void> {
    return this.initialization ??= (async () => {
      const module = this.device.createShaderModule({ label: "Bake sculpt substrate", code: sculptBakeWGSL });
      this.pipeline = await this.device.createComputePipelineAsync({ label: "Bake sculpt substrate", layout: "auto",
        compute: { module, entryPoint: "bake" } });
      this.flush();
    })();
  }
  publish(arena: GPUBuffer, offsetBytes: number, words: Uint32Array): number {
    if (words === this.plan?.words && arena === this.arena && offsetBytes / 4 === this.offset) return 0;
    const pending = this.validated?.base === this.plan && this.validated?.next.words === words
      ? this.validated.next : planSculptBake(this.plan, words);
    this.arena = arena; this.offset = offsetBytes / 4; this.pending = pending;
    if (this.pipeline) this.flush();
    return pending.jobs.length;
  }
  private flush(): void {
    const next = this.pending, arena = this.arena, pipeline = this.pipeline;
    if (!next || !arena || !pipeline) return;
    const queue = this.device.queue, base = this.offset;
    const h = next.cell, band = Math.min(...h) * 1.5;
    const meta = new Float32Array([...h, band, ...next.origin, 0]);
    // No CPU voxel image: only page addresses, physical lattice and brush operands.
    this.proxies.publish(queue, arena, base * 4, next.proxies);
    this.directory.publish(queue, arena, (base + SCULPT_TABLE) * 4, next.directory);
    this.metadata.publish(queue, arena, (base + SCULPT_META) * 4, new Uint32Array(meta.buffer));
    const records = next.jobs.reduce((sum, job) => sum + job.cuts.length, 0);
    this.lastUpdate = { pages: next.jobs.length, records, samples: next.jobs.length * SCULPT_PAGE_SAMPLES };
    // Bounded batches also cover long-history undo/load without allocating an
    // operand arena proportional to history. A page occurs at most once in a
    // dispatch; later rounds continue its min reduction in queue order.
    const maxRecords = 256, maxPages = 256;
    const rounds = Math.ceil(Math.max(0, ...next.jobs.map(job => job.cuts.length)) / maxRecords);
    for (let round = 0; round < rounds; round++) {
      const jobs = next.jobs.filter(job => job.cuts.length > round * maxRecords);
      for (let first = 0; first < jobs.length; first += maxPages) {
        const batch = jobs.slice(first, first + maxPages);
        const count = batch.reduce((sum, job) => sum + Math.min(maxRecords, job.cuts.length - round * maxRecords), 0);
        const words = new Uint32Array(4 + batch.length * 8 + count * 8);
        words[0] = batch.length; words[1] = base;
        let cursor = 4 + batch.length * 8;
        batch.forEach((job, index) => {
          const cuts = job.cuts.slice(round * maxRecords, (round + 1) * maxRecords);
          words.set([job.page.slot, Number(job.reset && round === 0), ...job.page.coordinate, cursor, cuts.length, 0], 4 + index * 8);
          for (const cut of cuts) { words.set(next.words.subarray(4 + cut * 8, 12 + cut * 8), cursor); cursor += 8; }
        });
        if (!this.input || this.input.size < words.byteLength) {
          this.input?.destroy();
          this.input = this.device.createBuffer({ label: "Sculpt page update operands", size: 2 ** Math.ceil(Math.log2(Math.max(4096, words.byteLength))),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
          this.group = undefined;
        }
        queue.writeBuffer(this.input, 0, words);
        if (!this.group || this.groupArena !== arena) {
          this.group = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
            { binding: 0, resource: { buffer: arena } }, { binding: 1, resource: { buffer: this.input } },
          ] });
          this.groupArena = arena;
        }
        const encoder = this.device.createCommandEncoder({ label: "Bake accepted sculpt pages" });
        const pass = encoder.beginComputePass({ label: "Bake accepted sculpt pages" });
        pass.setPipeline(pipeline); pass.setBindGroup(0, this.group);
        pass.dispatchWorkgroups(Math.ceil(SCULPT_PAGE_SAMPLES / 64), batch.length); pass.end();
        queue.submit([encoder.finish()]);
      }
    }
    this.plan = next; this.pending = undefined; this.validated = undefined;
  }
  destroy(): void { this.input?.destroy(); this.input = undefined; this.group = undefined; }
}
