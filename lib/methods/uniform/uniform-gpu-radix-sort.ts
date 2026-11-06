/**
 * GPU-resident ordering primitives for the cloth coupling: an exclusive u32
 * scan and a stable LSD radix sort of (key, value) u32 pairs. The element
 * count lives in a GPU buffer written by the producing kernel, so a sort is
 * encoded with no host round trip; elements at or past the count are absent.
 * Work and memory scale with the declared capacity (the cloth's own size).
 *
 * Stability is what the coupling needs: entries emitted in the CPU oracle's
 * insertion order and sorted by cell key come out grouped by cell with each
 * group in that insertion order, the order the oracle sums them in.
 */

const SCAN_BLOCK = 1024, SORT_BLOCK = 256, DIGIT_BITS = 4, DIGITS = 1 << DIGIT_BITS;

const scanCode = /* wgsl */`
struct ScanParams {length:u32,unused0:u32,unused1:u32,unused2:u32};
@group(0) @binding(0) var<storage,read_write> data:array<u32>;
@group(0) @binding(1) var<storage,read_write> sums:array<u32>;
@group(0) @binding(2) var<uniform> params:ScanParams;
var<workgroup> partial:array<u32,256>;
// Exclusive scan of 1024 values a workgroup, four a thread; the block's total to sums.
@compute @workgroup_size(256) fn scanBlocks(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) w:vec3u){
 let base=${SCAN_BLOCK}u*w.x+4u*t;var v=vec4u(0u);
 for(var k=0u;k<4u;k++){if(base+k<params.length){v[k]=data[base+k];}}
 let total=v.x+v.y+v.z+v.w;partial[t]=total;
 for(var offset=1u;offset<256u;offset*=2u){workgroupBarrier();let other=select(0u,partial[t-offset],t>=offset);workgroupBarrier();partial[t]+=other;}
 workgroupBarrier();
 var run=partial[t]-total;
 for(var k=0u;k<4u;k++){if(base+k<params.length){data[base+k]=run;}run+=v[k];}
 if(t==255u){sums[w.x]=partial[255];}
}
// Add each block's scanned offset to its values.
@compute @workgroup_size(256) fn addBlocks(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) w:vec3u){
 let base=${SCAN_BLOCK}u*w.x+4u*t;let add=sums[w.x];
 for(var k=0u;k<4u;k++){if(base+k<params.length){data[base+k]+=add;}}
}`;

const sortCode = /* wgsl */`
struct SortParams {shift:u32,blocks:u32,unused0:u32,unused1:u32};
@group(0) @binding(0) var<storage,read> inKeys:array<u32>;
@group(0) @binding(1) var<storage,read> inValues:array<u32>;
@group(0) @binding(2) var<storage,read_write> outKeys:array<u32>;
@group(0) @binding(3) var<storage,read_write> outValues:array<u32>;
@group(0) @binding(4) var<storage,read_write> histogram:array<u32>;
@group(0) @binding(5) var<storage,read> count:array<u32>;
@group(0) @binding(6) var<uniform> params:SortParams;
var<workgroup> bins:array<atomic<u32>,${DIGITS}>;
var<workgroup> ranks:array<vec4u,${SORT_BLOCK}>;
fn digitOf(key:u32)->u32{return (key>>params.shift)&${DIGITS - 1}u;}
// Digit counts by block, digit-major: their exclusive scan is each block's base.
@compute @workgroup_size(${SORT_BLOCK}) fn histogramBlocks(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) w:vec3u){
 if(t<${DIGITS}u){atomicStore(&bins[t],0u);}workgroupBarrier();
 let i=${SORT_BLOCK}u*w.x+t;if(i<count[0]){atomicAdd(&bins[digitOf(inKeys[i])],1u);}
 workgroupBarrier();if(t<${DIGITS}u){histogram[t*params.blocks+w.x]=atomicLoad(&bins[t]);}
}
// Stable scatter: rank among the block's earlier entries of the same digit,
// byte-packed counters (an exclusive prefix within 256 entries fits a byte).
@compute @workgroup_size(${SORT_BLOCK}) fn scatterBlocks(@builtin(local_invocation_index) t:u32,@builtin(workgroup_id) w:vec3u){
 let n=count[0];let i=${SORT_BLOCK}u*w.x+t;
 var key=0u;var digit=0u;if(i<n){key=inKeys[i];digit=digitOf(key);}
 var previous=vec4u(0u);if(t>0u&&i-1u<n){let d=digitOf(inKeys[i-1u]);previous[d>>2u]=1u<<(8u*(d&3u));}
 ranks[t]=previous;
 for(var offset=1u;offset<${SORT_BLOCK}u;offset*=2u){workgroupBarrier();let other=select(vec4u(0u),ranks[t-offset],t>=offset);workgroupBarrier();ranks[t]+=other;}
 workgroupBarrier();if(i>=n){return;}
 let rank=(ranks[t][digit>>2u]>>(8u*(digit&3u)))&255u;let at=histogram[digit*params.blocks+w.x]+rank;
 outKeys[at]=key;outValues[at]=inValues[i];
}`;

interface ScanLevel { data: GPUBuffer; sums: GPUBuffer; length: number; blocks: number; group: GPUBindGroup }

/** In-place exclusive scan of a fixed-length u32 buffer. */
export class UniformGPUScan {
  private readonly levels: ScanLevel[] = [];
  private readonly owned: GPUBuffer[] = [];
  /** `offset` (bytes, a multiple of the storage offset alignment) places the values inside `data`. */
  constructor(private readonly kernels: UniformGPUOrderingKernels, readonly data: GPUBuffer, readonly length: number, offset = 0) {
    const d = kernels.device;
    if (offset % d.limits.minStorageBufferOffsetAlignment) throw new Error('GPU scan offset is not storage aligned');
    let buffer: GPUBufferBinding = { buffer: data, offset, size: 4 * Math.max(1, length) }, n = length;
    for (;;) {
      const blocks = Math.max(1, Math.ceil(n / SCAN_BLOCK));
      if (blocks > d.limits.maxComputeWorkgroupsPerDimension) throw new Error(`GPU scan of ${n} values exceeds one dispatch dimension`);
      const sums = d.createBuffer({ label: 'GPU scan block sums', size: 4 * Math.max(4, blocks), usage: GPUBufferUsage.STORAGE });
      const params = d.createBuffer({ label: 'GPU scan length', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(params, 0, new Uint32Array([n, 0, 0, 0])); this.owned.push(sums, params);
      const group = d.createBindGroup({ layout: kernels.scanLayout, entries: [buffer, { buffer: sums }, { buffer: params }].map((resource, binding) => ({ binding, resource })) });
      this.levels.push({ data: buffer.buffer, sums, length: n, blocks, group });
      if (blocks === 1) break;
      buffer = { buffer: sums }; n = blocks;
    }
  }
  encode(pass: GPUComputePassEncoder): void {
    for (const level of this.levels) { pass.setPipeline(this.kernels.scanBlocks); pass.setBindGroup(0, level.group); pass.dispatchWorkgroups(level.blocks); }
    for (let k = this.levels.length - 2; k >= 0; k--) { const level = this.levels[k]!; pass.setPipeline(this.kernels.addBlocks); pass.setBindGroup(0, level.group); pass.dispatchWorkgroups(level.blocks); }
  }
  destroy(): void { for (const b of this.owned) b.destroy(); this.owned.length = 0; }
}

/** Stable sort of `capacity` (key, value) pairs by the low `keyBits` key bits.
 * `keys`/`values` hold the input and, after encode, the sorted output;
 * `count` holds the live element count in its first word. */
export class UniformGPURadixSort {
  readonly keys: GPUBuffer; readonly values: GPUBuffer;
  private readonly spareKeys: GPUBuffer; private readonly spareValues: GPUBuffer;
  private readonly histogram: GPUBuffer; private readonly scan: UniformGPUScan;
  private readonly passes: GPUBindGroup[] = [];
  private readonly owned: GPUBuffer[] = [];
  private readonly blocks: number;
  constructor(private readonly kernels: UniformGPUOrderingKernels, readonly capacity: number, readonly keyBits: number, readonly count: GPUBuffer, label = 'GPU radix sort') {
    if (!Number.isInteger(capacity) || capacity < 1) throw new Error('GPU radix sort needs a positive capacity');
    if (!Number.isInteger(keyBits) || keyBits < 1 || keyBits > 32) throw new Error('GPU radix sort keys are 1 to 32 bits');
    const d = kernels.device, size = 4 * Math.max(4, capacity), usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    this.blocks = Math.ceil(capacity / SORT_BLOCK);
    if (this.blocks > d.limits.maxComputeWorkgroupsPerDimension) throw new Error(`GPU radix sort of ${capacity} pairs exceeds one dispatch dimension`);
    const make = (name: string, bytes = size) => { const b = d.createBuffer({ label: `${label} ${name}`, size: bytes, usage }); this.owned.push(b); return b; };
    this.keys = make('keys'); this.values = make('values'); this.spareKeys = make('spare keys'); this.spareValues = make('spare values');
    this.histogram = make('digit histogram', 4 * Math.max(4, DIGITS * this.blocks));
    this.scan = new UniformGPUScan(kernels, this.histogram, DIGITS * this.blocks);
    const passes = Math.ceil(keyBits / DIGIT_BITS);
    for (let k = 0; k < passes; k++) {
      const params = d.createBuffer({ label: `${label} digit ${k}`, size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      d.queue.writeBuffer(params, 0, new Uint32Array([DIGIT_BITS * k, this.blocks, 0, 0])); this.owned.push(params);
      const [ik, iv, ok, ov] = k % 2 === 0 ? [this.keys, this.values, this.spareKeys, this.spareValues] : [this.spareKeys, this.spareValues, this.keys, this.values];
      this.passes.push(d.createBindGroup({ layout: kernels.sortLayout, entries: [ik, iv, ok, ov, this.histogram, count, params].map((b, binding) => ({ binding, resource: { buffer: b } })) }));
    }
  }
  encode(encoder: GPUCommandEncoder): void {
    for (const group of this.passes) {
      const pass = encoder.beginComputePass({ label: 'GPU radix sort digit' });
      pass.setPipeline(this.kernels.histogramBlocks); pass.setBindGroup(0, group); pass.dispatchWorkgroups(this.blocks);
      this.scan.encode(pass);
      pass.setPipeline(this.kernels.scatterBlocks); pass.setBindGroup(0, group); pass.dispatchWorkgroups(this.blocks);
      pass.end();
    }
    // An odd digit count leaves the result in the spare pair.
    if (this.passes.length % 2 === 1) {
      encoder.copyBufferToBuffer(this.spareKeys, 0, this.keys, 0, this.keys.size);
      encoder.copyBufferToBuffer(this.spareValues, 0, this.values, 0, this.values.size);
    }
  }
  destroy(): void { this.scan.destroy(); for (const b of this.owned) b.destroy(); this.owned.length = 0; }
}

/** The compiled scan and sort kernels, shared by every scan and sort on a device. */
export class UniformGPUOrderingKernels {
  readonly scanLayout: GPUBindGroupLayout; readonly sortLayout: GPUBindGroupLayout;
  scanBlocks!: GPUComputePipeline; addBlocks!: GPUComputePipeline; histogramBlocks!: GPUComputePipeline; scatterBlocks!: GPUComputePipeline;
  private constructor(readonly device: GPUDevice) {
    const b = (binding: number, type: GPUBufferBindingType): GPUBindGroupLayoutEntry => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    this.scanLayout = device.createBindGroupLayout({ entries: [b(0, 'storage'), b(1, 'storage'), b(2, 'uniform')] });
    this.sortLayout = device.createBindGroupLayout({ entries: [b(0, 'read-only-storage'), b(1, 'read-only-storage'), b(2, 'storage'), b(3, 'storage'), b(4, 'storage'), b(5, 'read-only-storage'), b(6, 'uniform')] });
  }
  static async create(device: GPUDevice): Promise<UniformGPUOrderingKernels> {
    const k = new UniformGPUOrderingKernels(device);
    const scan = device.createShaderModule({ label: 'GPU exclusive scan', code: scanCode }), sort = device.createShaderModule({ label: 'GPU stable radix sort', code: sortCode });
    for (const module of [scan, sort]) {
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === 'error');
      if (errors.length) throw new Error(errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join('\n'));
    }
    const scanLayout = device.createPipelineLayout({ bindGroupLayouts: [k.scanLayout] }), sortLayout = device.createPipelineLayout({ bindGroupLayouts: [k.sortLayout] });
    [k.scanBlocks, k.addBlocks, k.histogramBlocks, k.scatterBlocks] = await Promise.all([
      device.createComputePipelineAsync({ layout: scanLayout, compute: { module: scan, entryPoint: 'scanBlocks' } }),
      device.createComputePipelineAsync({ layout: scanLayout, compute: { module: scan, entryPoint: 'addBlocks' } }),
      device.createComputePipelineAsync({ layout: sortLayout, compute: { module: sort, entryPoint: 'histogramBlocks' } }),
      device.createComputePipelineAsync({ layout: sortLayout, compute: { module: sort, entryPoint: 'scatterBlocks' } }),
    ]);
    return k;
  }
}
