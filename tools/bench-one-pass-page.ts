/** The one-pass page microbenchmark (docs/plans/one-pass-m1max-2026-11-02.md §7).
 *
 * node --import tsx tools/bench-one-pass-page.ts [--pages=64,128,512,1536] [--reps=60] [--out=PATH.json]
 *
 * No solver. N resident 4h pages of 8^3 cells, each reached through a page
 * directory (page coordinate -> block, a shuffled permutation, as umPageWord
 * maps a page to its block). Liquid everywhere below a flat surface one cell
 * under the top. One step's chain, the same arithmetic in both variants:
 *   1. semi-Lagrangian MAC advection (trilinear backtrace of the previous field),
 *   2. gravity, 3. divergence, 4. four red-black Gauss-Seidel sweeps of the
 *   7-point Poisson (air p = 0, walls Neumann), 5. the pressure gradient,
 *   6. two velocity extension sweeps into the air.
 * A (today's shape): one dispatch a stage and a half sweep, 64-thread groups
 *   a 4^3 tile, every field in global memory: 14 dispatches.
 * B (proposed): one dispatch, one 512-thread group a page, the page's fields
 *   and a one-cell halo in threadgroup memory, barriers between stages. Its
 *   Poisson is page-local (the halo frozen): the cost of the shape, not a
 *   solver. Go/no-go: B under 5x better than A at N = 64 and 512 stops the plan.
 * E: 14 dispatches of a one-load, one-store kernel over A's grid; E1: the
 *   same with one workgroup each (the per-dispatch floor alone).
 * Times are GPU timestamps around one compute pass a step (A's dispatches
 * share it, as the solver's stages share passes), median over --reps steps. */
import { pathToFileURL } from "node:url";
import { writeFileSync } from "node:fs";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const PAGES = (option("pages") ?? "64,128,512,1536").split(",").map(Number);
const REPS = Number(option("reps") ?? 60), WARM = 10;
const SHAPES: Record<number, [number, number, number]> = { 64: [4, 4, 4], 128: [8, 4, 4], 256: [8, 8, 4], 512: [8, 8, 8], 1024: [16, 8, 8], 1536: [16, 12, 8] };

const common = (px: number, py: number, pz: number) => /* wgsl */`
const PX:i32=${px};const PY:i32=${py};const PZ:i32=${pz};
const NX:i32=PX*8;const NY:i32=PY*8;const NZ:i32=PZ*8;
const DT:f32=${(1 / 60).toFixed(8)};const G:f32=9.81;const INV_H:f32=25.0;
@group(0) @binding(0) var<storage,read> dir:array<u32>;
@group(0) @binding(1) var<storage,read> blockPage:array<u32>;
@group(0) @binding(2) var<storage,read> prev:array<vec4f>;
@group(0) @binding(3) var<storage,read_write> vel:array<vec4f>;
@group(0) @binding(4) var<storage,read_write> pres:array<f32>;
@group(0) @binding(5) var<storage,read_write> rhs:array<f32>;
// Liquid below the surface one cell under the top; outside the box is wall.
fn inside(g:vec3i)->bool{return all(g>=vec3i(0))&&all(g<vec3i(NX,NY,NZ));}
fn liquid(g:vec3i)->bool{return inside(g)&&g.y<NY-1;}
// The page directory: a dependent load before every field load.
fn cellAt(g:vec3i)->i32{
 if(!inside(g)){return -1;}
 let pg=g>>vec3u(3u);let block=dir[u32(pg.x+PX*(pg.y+PY*pg.z))];
 let l=g&vec3i(7);return i32(block)*512+l.x+8*(l.y+8*l.z);
}
fn pageCoord(block:u32)->vec3i{let p=i32(blockPage[block]);return vec3i(p%PX,(p/PX)%PY,p/(PX*PY));}
fn cellCoord(id:u32)->vec3i{let l=i32(id%512u);return 8*pageCoord(id/512u)+vec3i(l%8,(l/8)%8,l/64);}
`;

// Trilinear sample of component c of a MAC field at x (cell units, cell
// centres at i + 0.5): face j of axis c sits at j + 0.5 + 0.5 e_c.
const sampler = (name: string, load: string) => /* wgsl */`
fn ${name}(c:u32,x:vec3f)->f32{
 var o=vec3f(0.5);o[c]=1.0;let q=x-o;let b=vec3i(floor(q));let f=q-floor(q);var s=0.0;
 for(var k=0u;k<8u;k++){
  let e=vec3i(i32(k&1u),i32((k>>1u)&1u),i32(k>>2u));let w=select(1.0-f,f,e!=vec3i(0));
  s+=w.x*w.y*w.z*${load};
 }
 return s;
}`;

const variantA = (px: number, py: number, pz: number) => `${common(px, py, pz)}
fn loadPrev(g:vec3i,c:u32)->f32{let i=cellAt(g);if(i<0){return 0.0;}return prev[i][c];}
${sampler("samplePrev", "loadPrev(b+e,c)")}
fn ids(gid:vec3u)->u32{return gid.x;}
@compute @workgroup_size(64) fn advect(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);
 let x=vec3f(g)+0.5;let u=vec3f(samplePrev(0u,x),samplePrev(1u,x),samplePrev(2u,x));
 let d=x-DT*INV_H*u;var r=vec4f(0);
 for(var c=0u;c<3u;c++){var o=vec3f(0.0);o[c]=0.5;r[c]=samplePrev(c,d+o);}
 vel[cellAt(g)]=r;
}
@compute @workgroup_size(64) fn gravity(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);let i=cellAt(g);
 if(liquid(g)){var v=vel[i];v.y-=G*DT;vel[i]=v;}
}
@compute @workgroup_size(64) fn divergence(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);let i=cellAt(g);
 if(!liquid(g)){rhs[i]=0.0;pres[i]=0.0;return;}
 let v=vel[i];var d=v.x+v.y+v.z;
 for(var a=0;a<3;a++){var n=g;n[a]-=1;let j=cellAt(n);if(j>=0){d-=vel[j][a];}}
 rhs[i]=d;
}
fn relax(g:vec3i,i:i32){
 var s=0.0;var n=0.0;
 for(var a=0;a<6;a++){var q=g;q[a>>1]+=select(-1,1,(a&1)!=0);
  if(!inside(q)){continue;}n+=1.0;if(liquid(q)){s+=pres[cellAt(q)];}}
 pres[i]=(s-rhs[i])/max(n,1.0);
}
@compute @workgroup_size(64) fn red(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);
 if(liquid(g)&&((g.x+g.y+g.z)&1)==0){relax(g,cellAt(g));}
}
@compute @workgroup_size(64) fn black(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);
 if(liquid(g)&&((g.x+g.y+g.z)&1)==1){relax(g,cellAt(g));}
}
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);let i=cellAt(g);
 var v=vel[i];let p=select(0.0,pres[i],liquid(g));
 for(var a=0;a<3;a++){var q=g;q[a]+=1;if(!inside(q)){v[a]=0.0;continue;}
  if(liquid(g)||liquid(q)){v[a]-=select(0.0,pres[cellAt(q)],liquid(q))-p;}}
 vel[i]=v;
}
// The launch floor: one load and one store a thread (variants E, E1).
@compute @workgroup_size(64) fn touch(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}rhs[id]=pres[id]+1.0;
}
@compute @workgroup_size(64) fn extend(@builtin(global_invocation_id) gid:vec3u){
 let id=ids(gid);if(id>=arrayLength(&pres)){return;}let g=cellCoord(id);
 if(liquid(g)){return;}
 var s=vec4f(0);var n=0.0;
 for(var a=0;a<6;a++){var q=g;q[a>>1]+=select(-1,1,(a&1)!=0);if(liquid(q)){s+=vel[cellAt(q)];n+=1.0;}}
 if(n>0.0){vel[cellAt(g)]=s/n;}
}`;

// One 512-thread group a page. Shared arrays hold the page and a one-cell
// halo (10^3): S(l) for local l in [-1, 8].
const variantB = (px: number, py: number, pz: number) => `${common(px, py, pz)}
var<workgroup> su:array<vec4f,1000>;
var<workgroup> sp:array<f32,1000>;
var<workgroup> sr:array<f32,1000>;
var<private> origin:vec3i;
fn S(l:vec3i)->u32{let c=clamp(l,vec3i(-1),vec3i(8))+1;return u32(c.x+10*(c.y+10*c.z));}
fn loadShared(l:vec3i,c:u32)->f32{return su[S(l)][c];}
${sampler("sampleShared", "loadShared(b+e-origin,c)")}
fn relaxShared(g:vec3i,l:vec3i){
 var s=0.0;var n=0.0;
 for(var a=0;a<6;a++){var q=g;q[a>>1]+=select(-1,1,(a&1)!=0);var m=l;m[a>>1]+=select(-1,1,(a&1)!=0);
  if(!inside(q)){continue;}n+=1.0;if(liquid(q)){s+=sp[S(m)];}}
 sp[S(l)]=(s-sr[S(l)])/max(n,1.0);
}
@compute @workgroup_size(512) fn page(@builtin(workgroup_id) wg:vec3u,@builtin(local_invocation_index) lane:u32){
 let block=wg.x+65535u*wg.y;if(block>=arrayLength(&pres)/512u){return;}
 origin=8*pageCoord(block);
 let l=vec3i(i32(lane%8u),i32((lane/8u)%8u),i32(lane/64u));let g=origin+l;let me=i32(block)*512+i32(lane);
 // Load the previous field with its halo, and the previous pressure (warm start).
 for(var k=lane;k<1000u;k+=512u){
  let h=vec3i(i32(k%10u),i32((k/10u)%10u),i32(k/100u))-1;let i=cellAt(origin+h);
  su[k]=select(vec4f(0),prev[max(i,0)],i>=0);sp[k]=select(0.0,pres[max(i,0)],i>=0);
 }
 workgroupBarrier();
 // 1-2. Advect and gravity, in registers.
 let x=vec3f(g)+0.5;let u=vec3f(sampleShared(0u,x),sampleShared(1u,x),sampleShared(2u,x));
 let d=x-DT*INV_H*u;var v=vec4f(0);
 for(var c=0u;c<3u;c++){var o=vec3f(0.0);o[c]=0.5;v[c]=sampleShared(c,d+o);}
 if(liquid(g)){v.y-=G*DT;}
 workgroupBarrier();
 su[S(l)]=v;
 workgroupBarrier();
 // 3. Divergence.
 if(liquid(g)){var dv=v.x+v.y+v.z;for(var a=0;a<3;a++){var m=l;m[a]-=1;var q=g;q[a]-=1;if(inside(q)){dv-=su[S(m)][a];}}sr[S(l)]=dv;}
 else{sr[S(l)]=0.0;sp[S(l)]=0.0;}
 workgroupBarrier();
 // 4. Four red-black sweeps.
 for(var sweep=0;sweep<4;sweep++){
  if(liquid(g)&&((g.x+g.y+g.z)&1)==0){relaxShared(g,l);}
  workgroupBarrier();
  if(liquid(g)&&((g.x+g.y+g.z)&1)==1){relaxShared(g,l);}
  workgroupBarrier();
 }
 // 5. Project.
 let p=select(0.0,sp[S(l)],liquid(g));
 for(var a=0;a<3;a++){var q=g;q[a]+=1;var m=l;m[a]+=1;if(!inside(q)){v[a]=0.0;continue;}
  if(liquid(g)||liquid(q)){v[a]-=select(0.0,sp[S(m)],liquid(q))-p;}}
 workgroupBarrier();
 su[S(l)]=v;
 workgroupBarrier();
 // 6. Two extension sweeps.
 for(var e=0;e<2;e++){
  var w=su[S(l)];
  if(!liquid(g)){var s=vec4f(0);var n=0.0;
   for(var a=0;a<6;a++){var q=g;q[a>>1]+=select(-1,1,(a&1)!=0);var m=l;m[a>>1]+=select(-1,1,(a&1)!=0);if(liquid(q)){s+=su[S(m)];n+=1.0;}}
   if(n>0.0){w=s/n;}}
  workgroupBarrier();
  su[S(l)]=w;
  workgroupBarrier();
 }
 vel[me]=su[S(l)];pres[me]=sp[S(l)];rhs[me]=sr[S(l)];
}`;

async function main() {
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href); Object.assign(globalThis, dawn.globals);
  const adapter = (await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter())!;
  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
    requiredLimits: { maxComputeWorkgroupStorageSize: 32768, maxComputeInvocationsPerWorkgroup: 512, maxComputeWorkgroupSizeX: 512 },
  });
  const errors: string[] = []; device.addEventListener("uncapturederror", (e: any) => { errors.push(e.error.message); });
  const results: Record<string, unknown>[] = [];
  for (const n of PAGES) {
    const shape = SHAPES[n]; if (!shape) throw new Error(`no page box for ${n}`);
    const [px, py, pz] = shape, cells = n * 512;
    // A shuffled page -> block directory and its inverse.
    const perm = Array.from({ length: n }, (_, i) => i); let seed = 12345;
    for (let i = n - 1; i > 0; i--) { seed = (seed * 1103515245 + 12345) >>> 0; const j = seed % (i + 1); [perm[i], perm[j]] = [perm[j]!, perm[i]!]; }
    const inverse = new Uint32Array(n); perm.forEach((b, p) => { inverse[b] = p; });
    const buffer = (data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer> | number, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) => {
      const size = typeof data === "number" ? data : data.byteLength; const b = device.createBuffer({ size, usage });
      if (typeof data !== "number") device.queue.writeBuffer(b, 0, data); return b;
    };
    const dir = buffer(new Uint32Array(perm)), blockPage = buffer(inverse);
    // A gentle swirl so the backtrace samples off-lattice.
    const init = new Float32Array(4 * cells);
    for (let i = 0; i < cells; i++) { init[4 * i] = 0.05 * Math.sin(i * 0.37); init[4 * i + 1] = 0.05 * Math.cos(i * 0.23); init[4 * i + 2] = 0.02; }
    const prev = buffer(init), vel = buffer(4 * 4 * cells), pres = buffer(4 * cells), rhs = buffer(4 * cells);
    const timings: Record<string, number[]> = {};
    for (const variant of ["A", "B", "E", "E1"] as const) {
      const module = device.createShaderModule({ code: variant === "B" ? variantB(px, py, pz) : variantA(px, py, pz) });
      const info = await module.getCompilationInfo(); for (const m of info.messages) if (m.type === "error") throw new Error(`${variant}: ${m.message} (line ${m.lineNum})`);
      const bindLayout = device.createBindGroupLayout({ entries: [0, 1, 2, 3, 4, 5].map(binding => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: binding < 3 ? "read-only-storage" as const : "storage" as const } })) });
      const layout = device.createPipelineLayout({ bindGroupLayouts: [bindLayout] });
      const pipe = (entryPoint: string) => device.createComputePipeline({ layout, compute: { module, entryPoint } });
      const stages = variant === "A" ? ["advect", "gravity", "divergence", "red", "black", "red", "black", "red", "black", "red", "black", "project", "extend", "extend"] : variant === "B" ? ["page"] : Array<string>(14).fill("touch");
      const pipelines = new Map<string, GPUComputePipeline>(); for (const s of new Set(stages)) pipelines.set(s, pipe(s));
      const groups = new Map([...pipelines].map(([s, p]) => [s, device.createBindGroup({ layout: bindLayout, entries: [dir, blockPage, prev, vel, pres, rhs].map((b, binding) => ({ binding, resource: { buffer: b } })) })]));
      const querySet = device.createQuerySet({ type: "timestamp", count: 2 * (REPS + WARM) });
      const resolve = buffer(16 * (REPS + WARM), GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
      const read = device.createBuffer({ size: 16 * (REPS + WARM), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      for (let r = 0; r < REPS + WARM; r++) {
        const encoder = device.createCommandEncoder();
        const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
        for (const s of stages) {
          pass.setPipeline(pipelines.get(s)!); pass.setBindGroup(0, groups.get(s)!);
          if (variant === "A" || variant === "E") { const w = Math.ceil(cells / 64); pass.dispatchWorkgroups(Math.min(w, 65535), Math.ceil(w / 65535)); }
          else if (variant === "E1") pass.dispatchWorkgroups(1);
          else pass.dispatchWorkgroups(Math.min(n, 65535), Math.ceil(n / 65535));
        }
        pass.end();
        // The next step reads this one's velocity.
        encoder.copyBufferToBuffer(vel, 0, prev, 0, 16 * cells);
        device.queue.submit([encoder.finish()]);
      }
      const encoder = device.createCommandEncoder(); encoder.resolveQuerySet(querySet, 0, 2 * (REPS + WARM), resolve, 0); encoder.copyBufferToBuffer(resolve, 0, read, 0, 16 * (REPS + WARM)); device.queue.submit([encoder.finish()]);
      await read.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(read.getMappedRange().slice(0)); read.unmap();
      timings[variant] = Array.from({ length: REPS }, (_, r) => Number(t[2 * (r + WARM) + 1]! - t[2 * (r + WARM)]!) / 1e3);
      // Sanity: the step produced finite velocity.
      const check = device.createBuffer({ size: 16 * Math.min(cells, 4096), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      { const e = device.createCommandEncoder(); e.copyBufferToBuffer(vel, 0, check, 0, check.size); device.queue.submit([e.finish()]); }
      await check.mapAsync(GPUMapMode.READ); const v = new Float32Array(check.getMappedRange().slice(0)); check.unmap();
      if (!v.every(Number.isFinite)) throw new Error(`${variant} N=${n}: non-finite velocity`);
      querySet.destroy(); resolve.destroy(); read.destroy(); check.destroy();
    }
    const med = (a: number[]) => [...a].sort((x, y) => x - y)[a.length >> 1]!;
    const a = med(timings.A!), b = med(timings.B!);
    const row = { pages: n, cells, A_us: +a.toFixed(1), B_us: +b.toFixed(1), E_us: +med(timings.E!).toFixed(1), E1_us: +med(timings.E1!).toFixed(1), ratio: +(a / b).toFixed(2), A_perDispatch_us: +(a / 14).toFixed(1), A_p10: +[...timings.A!].sort((x, y) => x - y)[Math.floor(REPS / 10)]!.toFixed(1), B_p10: +[...timings.B!].sort((x, y) => x - y)[Math.floor(REPS / 10)]!.toFixed(1) };
    console.log(JSON.stringify(row)); results.push(row);
    for (const b of [dir, blockPage, prev, vel, pres, rhs]) b.destroy();
  }
  if (errors.length) throw new Error(errors.join("\n"));
  // Slopes between the first and last N: microseconds a page.
  if (results.length > 1) {
    const f = results[0]! as any, l = results[results.length - 1]! as any;
    console.log(JSON.stringify({ slopeA_usPerPage: +((l.A_us - f.A_us) / (l.pages - f.pages)).toFixed(3), slopeB_usPerPage: +((l.B_us - f.B_us) / (l.pages - f.pages)).toFixed(3) }));
  }
  const out = option("out"); if (out) writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), reps: REPS, results }, null, 1));
  device.destroy();
}
await main();
