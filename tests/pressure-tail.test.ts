import assert from "node:assert/strict";
import test from "node:test";
import { planUniformPressureSuper, uniformPressureSuperGroup, uniformPressureSuperKey, UNIFORM_PRESSURE_SUPER_BULK, UNIFORM_PRESSURE_SUPER7_SIDE, type UniformPressureSuperLattice } from "../lib/methods/uniform/uniform-pressure-super";
import { UNIFORM_PRESSURE_ARENA_SCHEDULE, uniformPressureAggregateWGSL } from "../lib/methods/uniform/uniform-pressure-arena";
import { tailExact, tailHierarchy, tailIterate, tailPages, tailPool, tailSuperblocks, tailLevel, tailResidual, uniformPressureTailAggregate, type TailLevel, type TailV3 } from "./helpers/pressure-tail-twin";

test("the super-block tail's words follow the arena's and never overlap", () => {
  const W = planUniformPressureSuper(9000, 1000, 50_000);
  assert.equal(W.capacity6, 9000);
  assert.equal(W.capacity7, UNIFORM_PRESSURE_SUPER7_SIDE ** 3);
  const index = [W.index.count6, W.index.count7, W.index.list6, W.index.list7, W.index.map6, W.index.map7, W.index.end];
  assert.equal(index[0], 1000);
  assert.deepEqual([...index].sort((a, b) => a - b), index);
  assert.equal(W.index.map7 - W.index.map6, 32 ** 3);
  assert.equal(W.index.end - W.index.map7, 16 ** 3);
  const pool = [W.pool.x6, W.pool.y6, W.pool.r6, W.pool.d6, W.pool.k6, W.pool.c6, W.pool.x7, W.pool.r7, W.pool.d7, W.pool.k7, W.pool.c7, W.pool.end];
  assert.equal(pool[0], 50_000);
  assert.equal(W.pool.c6 - W.pool.k6, 6 * W.capacity6);
  assert.equal(W.pool.end - W.pool.c7, 8 * W.capacity7);
  assert.deepEqual([...pool].sort((a, b) => a - b), pool);
  const small = planUniformPressureSuper(12, 0, 0);
  assert.equal(small.capacity6, 12);
  assert.equal(small.capacity7, 12);
  assert.throws(() => planUniformPressureSuper(0, 0, 0), RangeError);
});

test("a page's group keys count from the box's low page, mod the window", () => {
  assert.equal(uniformPressureSuperKey([0, 0, 0], [0, 0, 0], 1), 0);
  assert.equal(uniformPressureSuperKey([3, 5, 9], [0, 0, 0], 1), 1 | (2 << 5) | (4 << 10));
  assert.equal(uniformPressureSuperKey([3, 5, 9], [0, 0, 0], 2), 0 | (1 << 4) | (2 << 8));
  // A box whose low page sits at window page 62: its pages 62, 63, 0, 1 group together.
  assert.equal(uniformPressureSuperKey([62, 0, 0], [62, 0, 0], 2), uniformPressureSuperKey([1, 0, 0], [62, 0, 0], 2));
  assert.notEqual(uniformPressureSuperKey([61, 0, 0], [62, 0, 0], 2), uniformPressureSuperKey([62, 0, 0], [62, 0, 0], 2));
});

const rhs = (coord: readonly TailV3[]) => {
  let seed = 7;
  const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
  return Float64Array.from(coord, c => rand() - 0.5 + 0.3 * Math.sin(c[0] / 5));
};

test("an aggregate preserves its children's Dirichlet row sum while scaling links", () => {
  const pages = tailPool([6, 4, 4]), { l6, l7 } = tailHierarchy(pages, 0.5);
  assert.equal(l6.coord.length, 3 * 2 * 2);
  assert.equal(l7.coord.length, 2);
  // On a constant, a row's A·1 (D less its couplings to rows) is the sum
  // of its children's: the couplings across the group's faces cancel.
  const ones = (L: typeof pages, i: number) => { let v = L.D[i]!; for (let f = 0; f < 6; f++) if (L.near[6 * i + f]! >= 0) v -= L.K[6 * i + f]!; return v; };
  for (const [fine, coarse] of [[pages, l6], [l6, l7]] as const) {
    const sum = new Float64Array(coarse.coord.length);
    fine.coord.forEach((_, i) => { sum[fine.parent[i]!]! += ones(fine, i); });
    coarse.coord.forEach((_, e) => assert.ok(Math.abs(ones(coarse, e) - sum[e]!) < 1e-12));
  }
});

const errorOf = (exact: Float64Array) => (X: Float64Array) => Math.sqrt(X.reduce((n, x, i) => n + (x - exact[i]!) ** 2, 0) / exact.reduce((n, x) => n + x * x, 0));

test("the super-block tail solves a pool of more than 4096 pages better than the page tail would", () => {
  // The tail's three corrections a cycle (the schedule's tailCycles) at its scale.
  const pages = tailPool([64, 16, 8]);
  assert.ok(pages.coord.length > 4096);
  const h = tailHierarchy(pages, UNIFORM_PRESSURE_ARENA_SCHEDULE.superScale);
  assert.ok(h.l7.coord.length <= 4096);
  const R = rhs(pages.coord), error = errorOf(tailExact(pages, R)), n = UNIFORM_PRESSURE_ARENA_SCHEDULE.tailCycles;
  const superblocks = error(tailIterate(pages, R, r => tailSuperblocks(h, r, UNIFORM_PRESSURE_ARENA_SCHEDULE.sweeps, 32), n).at(-1)!);
  const paged = error(tailIterate(pages, R, r => tailPages(pages, r, 32), n).at(-1)!);
  assert.ok(superblocks < 0.25, `super-block tail error ${superblocks}`);
  assert.ok(paged > 2 * superblocks, `page tail error ${paged} against ${superblocks}`);
});

test("the super-block tail contracts on ragged pools (part groups at every face)", () => {
  for (const box of [[17, 5, 3], [7, 5, 3], [9, 3, 1], [21, 9, 5], [33, 7, 5]] as TailV3[]) {
    const pages = tailPool(box), h = tailHierarchy(pages, UNIFORM_PRESSURE_ARENA_SCHEDULE.superScale), R = rhs(pages.coord), error = errorOf(tailExact(pages, R));
    const steps = tailIterate(pages, R, r => tailSuperblocks(h, r, UNIFORM_PRESSURE_ARENA_SCHEDULE.sweeps, 32), 3).map(error);
    assert.ok(steps[0]! < 0.5 && steps[2]! < steps[0]! && steps[2]! < 0.15, `${box}: ${steps}`);
  }
});

/** A weak free-surface connection leaves an almost constant pressure mode.
 * Interior smoothing barely changes it; a coarse correction must remove it,
 * rather than weaken its Dirichlet constraint once per hierarchy level. */
function weakSurfacePool(epsilon:number):TailLevel {
 const coords:TailV3[]=[],diag:number[]=[],links:number[]=[];
 for(let z=0;z<4;z++)for(let y=0;y<4;y++)for(let x=0;x<4;x++){
  const p:TailV3=[x,y,z];coords.push(p);let d=0;
  for(let axis=0;axis<3;axis++)for(const sign of [-1,1]){
   const inside=p[axis]!+sign>=0&&p[axis]!+sign<4;
   links.push(inside?1:0);d+=inside?1:epsilon;
  }
  diag.push(d);
 }
 return tailLevel(coords,Float64Array.from(diag),Float64Array.from(links));
}
function assertWeakModeContracts(h:ReturnType<typeof tailHierarchy>):void {
 const L=h.pages,zero=new Float64Array(L.coord.length),ones=new Float64Array(L.coord.length).fill(1);
 const rhs=Float64Array.from(L.coord,(_,i)=>-tailResidual(L,zero,ones,i));
 const solved=tailSuperblocks(h,rhs,3,32);
 const amplification=Math.max(...solved.map((_,i)=>Math.abs(tailResidual(L,rhs,solved,i))))/Math.max(...rhs);
 assert.ok(amplification<.4,`weak free-surface residual amplification ${amplification}`);
 assert.ok(Math.max(...solved.map(x=>Math.abs(x-1)))<.08,'near-constant pressure error must be removed');
}
test('multilevel correction contracts weak free-surface pressure modes',()=>{
 for(const epsilon of [.1,.03,.001])assertWeakModeContracts(tailHierarchy(weakSurfacePool(epsilon),.5));
});

test('GPU aggregate operator preserves Dirichlet rows and contracts a manufactured multilevel solve',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {withUniformDevice}=await import('./helpers/uniform-geometric');
 await withUniformDevice('pressure aggregate Dirichlet compatibility',async device=>{
  const module=device.createShaderModule({code:`${uniformPressureAggregateWGSL}
   @group(0) @binding(0) var<storage,read> input:array<vec4f>;
   @group(0) @binding(1) var<storage,read_write> output:array<f32>;
   @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id:vec3u){let v=input[id.x];output[id.x]=pAggregateDiagonal(v.x,v.y,v.z);}`});
  const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  const aggregate=async(fine:TailLevel)=>{
   const coarse=uniformPressureTailAggregate(fine,.5),count=coarse.coord.length;
   const data=new Float32Array(count*4);
   coarse.coord.forEach((_,i)=>{let scaledOutward=0;for(let f=0;f<6;f++)scaledOutward+=coarse.K[i*6+f]!;
    const outward=scaledOutward/.5;data.set([coarse.D[i]!+.5*outward,outward,.5,0],i*4);});
   const input=device.createBuffer({size:data.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,data);
   const output=device.createBuffer({size:count*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
   const read=device.createBuffer({size:count*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   try{
    const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);
    pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}));
    pass.dispatchWorkgroups(count);pass.end();encoder.copyBufferToBuffer(output,0,read,0,count*4);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
    const result=new Float32Array(read.getMappedRange()).slice();read.unmap();
    result.forEach((value,i)=>{assert.ok(Math.abs(value-coarse.D[i]!)<2e-6,`GPU row ${i}: ${value}, expected ${coarse.D[i]}`);coarse.D[i]=value;});
   }finally{input.destroy();output.destroy();read.destroy();}
   return coarse;
  };
  for(const epsilon of [.1,.03,.001]){const pages=weakSurfacePool(epsilon),l6=await aggregate(pages),l7=await aggregate(l6);assertWeakModeContracts({pages,l6,l7});}
 });
});

// These two rows are a principal submatrix of the actual step-7 lighthouse
// pressure operator (snapshot v16, rows 163454/163462). The first row has
// reached its pressure bound; its negative residual must not be prolonged.
test('GPU coarse correction holds active bounds, retains their Dirichlet links, and releases them next cycle',{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 const {uniformPressureCorrectionWGSL}=await import('../lib/methods/uniform/uniform-pressure-arena');
 const {withUniformDevice}=await import('./helpers/uniform-geometric');
 await withUniformDevice('pressure truncated active correction',async device=>{
  const code=`${uniformPressureCorrectionWGSL}${uniformPressureAggregateWGSL}
  @group(0) @binding(0) var<storage,read> input:array<vec4f>;
  @group(0) @binding(1) var<storage,read_write> output:array<vec4f>;
  @compute @workgroup_size(1) fn main(){
   let a=input[0];let b=input[1];let c=input[2].x;
   let fa=pCorrectionFree(a.w!=0.0,a.z,a.y);let fb=pCorrectionFree(b.w!=0.0,b.z,b.y);
   let ra=select(a.y,max(a.y,0.0),a.w!=0.0&&a.z<=0.0);
   let rb=select(b.y,max(b.y,0.0),b.w!=0.0&&b.z<=0.0);
   let link=select(0.0,c,fa&&fb);
   let diagonal=pAggregateDiagonal(select(0.0,a.x,fa)+select(0.0,b.x,fb)-2.0*link,0.0,0.5);
   let y=(ra+rb)/diagonal;
   let delta=vec2f(select(0.0,y,fa),select(0.0,y,fb));
   let rd=dot(vec2f(a.y,b.y),delta);
   let ada=a.x*delta.x*delta.x+b.x*delta.y*delta.y-2.0*c*delta.x*delta.y;
   output[0]=vec4f(delta,rd,0.5*ada-rd);
   output[1]=vec4f(diagonal,select(0.0,1.0,fa),select(0.0,1.0,pCorrectionFree(true,0.0,input[2].y)),select(0.0,1.0,pCorrectionFree(false,b.z,a.y)));
  }`;
  const module=device.createShaderModule({code});const pipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module,entryPoint:'main'}});
  const values=new Float32Array([39.99999713897705,-3068615.396694243,0,1,94.4000015258789,1231195.7966521978,-197906.90625,0,14.399999618530273,1000,0,0]);
  const input=device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});device.queue.writeBuffer(input,0,values);
  const output=device.createBuffer({size:32,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
  const read=device.createBuffer({size:32,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{
   const encoder=device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:input}},{binding:1,resource:{buffer:output}}]}));pass.dispatchWorkgroups(1);pass.end();encoder.copyBufferToBuffer(output,0,read,0,32);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const result=new Float32Array(read.getMappedRange()).slice();read.unmap();
   assert.equal(result[0],0,'active pressure bound must remain fixed during the coarse correction');
   assert.ok(result[1]!>0&&result[2]!>0&&result[3]!<0,'correction must be a pressure-energy descent step');
   assert.ok(Math.abs(result[4]!-values[4]!)<1e-5,'coupling to the held row must remain in the free diagonal');
   assert.equal(result[5],0);assert.equal(result[6],1,'a positive residual releases the bound on the next classification');
   assert.equal(result[7],1,'unbounded bulk rows stay free even at negative pressure and residual');
  }finally{input.destroy();output.destroy();read.destroy();}
 });
});

test("a 64h group's members resolve one set of children and one leader across blocks and BULK16 pages", () => {
  // A hand-built lattice: each page a live block, a block with no rows, a
  // BULK16 page, AIR or SOLID, from an odd box origin (groups split at odd
  // coordinates) whose window coordinates wrap past 63.
  const box: TailV3 = [61, 2, 5], side: TailV3 = [7, 6, 5], at = (p: readonly number[]) => p.map((v, a) => v - box[a]!);
  let seed = 12345;
  const random = () => { seed = (seed * 1103515245 + 12345) >>> 0; return seed / 2 ** 32; };
  const words = new Map<string, number>(), blockPages = new Map<number, TailV3>(), slotPages = new Map<number, TailV3>(), dead = new Set<number>();
  let blocks = 0, slots = 0;
  for (let z = 0; z < side[2]; z++) for (let y = 0; y < side[1]; y++) for (let x = 0; x < side[0]; x++) {
    const page: TailV3 = [box[0] + x, box[1] + y, box[2] + z], r = random();
    let word = 0;
    if (r < 0.4) { word = ++blocks; blockPages.set(word, page); if (r < 0.08) dead.add(word); }
    else if (r < 0.8) { const slot = ++slots; word = (UNIFORM_PRESSURE_SUPER_BULK | slot) >>> 0; slotPages.set(slot, page); }
    else if (r < 0.9) word = 0xffffffff;
    words.set(at(page).join(), word);
  }
  const capacity = blocks + 2, stored = (w: number) => ((w - 1) >>> 0) < capacity - 2;
  const wordAt = (page: readonly number[]) => words.get(at(page).join()) ?? 0;
  const near = (block: number, k: number) => { const p = blockPages.get(block)!; return [p[0] + (k % 3) - 1, p[1] + (Math.floor(k / 3) % 3) - 1, p[2] + Math.floor(k / 9) - 1]; };
  const L: UniformPressureSuperLattice = {
    record: (block, k) => { const w = wordAt(near(block, k)); return stored(w) ? w : undefined; },
    neighbourWord: (block, k) => wordAt(near(block, k)),
    pageWord: wordAt, stored,
    blockPage: block => blockPages.get(block)!, slotPage: slot => slotPages.get(slot)!,
    live: item => (item & UNIFORM_PRESSURE_SUPER_BULK) !== 0 || !dead.has(item),
  };
  const items = [...[...blockPages.keys()].filter(b => !dead.has(b)), ...[...slotPages.keys()].map(s => (UNIFORM_PRESSURE_SUPER_BULK | s) >>> 0)];
  const groups = new Map<number, { children: (number | undefined)[]; leaders: number; members: Set<number> }>();
  let mixed = 0;
  for (const item of items) {
    const g = uniformPressureSuperGroup(L, item, box);
    const seen = groups.get(g.key);
    if (seen) assert.deepEqual(g.children, seen.children, `group ${g.key}: item ${item} resolves other children`);
    const entry = seen ?? { children: g.children, leaders: 0, members: new Set<number>() };
    entry.leaders += g.leader ? 1 : 0; entry.members.add(item); groups.set(g.key, entry);
  }
  for (const [key, g] of groups) {
    assert.equal(g.leaders, 1, `group ${key}: ${g.leaders} leaders`);
    assert.deepEqual(new Set(g.children.filter(c => c !== undefined)), g.members, `group ${key}: children are its live items`);
    const kinds = new Set([...g.members].map(m => (m & UNIFORM_PRESSURE_SUPER_BULK) !== 0));
    if (kinds.size === 2) mixed++;
  }
  // Each child sits at its own octant of the group.
  for (const g of groups.values()) g.children.forEach((c, j) => {
    if (c === undefined) return;
    const p = (c & UNIFORM_PRESSURE_SUPER_BULK) !== 0 ? L.slotPage(c & 0x7fffffff) : L.blockPage(c);
    assert.equal(p.map((v, a) => ((v + 64 - box[a]!) & 63) & 1).reduce((s, b, a) => s | (b << a), 0), j);
  });
  assert.ok(mixed >= 4, `${mixed} groups mix blocks and BULK16 pages`);
});
