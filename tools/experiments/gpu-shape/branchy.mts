// GPU-shape M0e validation (docs/plans/gpu-shape/m0e-chains.md): does a load-guarded branch per lookup serialise
// independent lookups on the M1 Max? The brick kernels' common paths are runs of independent lookups (stencil taps,
// corners, faces), each a dependent chain page -> child -> data whose helper branches on the child (slot hit / miss,
// owner bit). chains.mts brackets their critical path between "address dependence only" and "every load-guarded branch
// resolves before the next load issues"; this measures which bound the hardware and compiler give.
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/branchy.mts <out.jsonl>
//
// Per lane: R rounds; round r's taps start from the previous round's sum (rounds are serial). Each round has N = 16
// taps; tap k is  page = dir[h & 1023] -> child = dir[1024 + (page ^ hk) & 16383] -> value = data[(child ^ hk) & mask]
// (dir 68 KB, data 16 MB). Variants:
//   free   acc += value: taps independent, no branch (a round = 3 links)
//   guard  if (child != NONE) { acc += data[...] } else { acc += data[other] }: a load in both arms, the miss arm
//          never taken (fgPhiWord / lPhi0 / fgFace's slot-hit test)
//   exit   for k { ...; if (page == NONE) { break; } ... } (foVertexRead4's early return: the next tap is gated)
//   owner  if (bit(child)) { acc += data[a] } else { acc += data[b] }: an owner-bit select between two loads
//   serial tap k's hash takes tap k-1's value (a true dependent chain, a round = 3N links: the calibration)
import { appendFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const root = resolve(import.meta.dirname, "../../..");
const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
const out = process.argv[2] ?? "branchy.jsonl";
writeFileSync(out, "");
const emit = (o: Record<string, unknown>) => { appendFileSync(out, JSON.stringify(o) + "\n"); };
paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve(root, "node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
const device: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
let gpuErrors = 0;
device.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors++; console.log("GPU ERROR", ev.error?.message); emit({ kind: "gpuError", message: String(ev.error?.message) }); });
const U = GPUBufferUsage;

const N = 16, DIRW = 1024 + 16384, DATAW = 1 << 22;
const HASH = `fn hsh(x0: u32) -> u32 { var x = x0 * 747796405u + 2891336453u; x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u; return (x >> 22u) ^ x; }`;
function wgsl(variant: string, L: number) {
  const tap = (k: number) => {
    const hk = variant === "serial" ? `hsh(s ^ ${k}u)` : `hsh(seed ^ ${k * 0x9e3779b1 >>> 0}u)`;
    const pre = `{ let hk = ${hk}; let page = dir[hk & 1023u]; let child = dir[1024u + ((page ^ hk) & 16383u)];`;
    switch (variant) {
      case "free": return `${pre} acc += data[(child ^ hk) & ${DATAW - 1}u]; }`;
      case "serial": return `${pre} s = data[(child ^ hk) & ${DATAW - 1}u]; acc += s; }`;
      case "guard": return `${pre} if (child != 0xffffffffu) { acc += data[(child ^ hk) & ${DATAW - 1}u]; } else { acc += data[(hk * 3u) & ${DATAW - 1}u] + dir[hk & 2047u]; } }`;
      case "owner": return `${pre} if ((child & 1u) != 0u) { acc += data[(child ^ hk) & ${DATAW - 1}u]; } else { acc += data[(child * 7u) & ${DATAW - 1}u]; } }`;
      default: throw new Error(variant);
    }
  };
  const body = variant === "exit"
    ? `for (var k = 0u; k < ${N}u; k++) { let hk = hsh(seed ^ (k * 0x9e3779b1u)); let page = dir[hk & 1023u]; if (page == 0xffffffffu) { break; }
         let child = dir[1024u + ((page ^ hk) & 16383u)]; acc += data[(child ^ hk) & ${DATAW - 1}u]; }`
    : [...Array(N).keys()].map(tap).join("\n    ");
  return `@group(0) @binding(0) var<storage, read> dir: array<u32>;
@group(0) @binding(1) var<storage, read> data: array<u32>;
@group(0) @binding(2) var<storage, read_write> outb: array<u32>;
@group(0) @binding(3) var<uniform> rounds: vec4u;
${HASH}
@compute @workgroup_size(${L}) fn main(@builtin(global_invocation_id) g: vec3u) {
  var acc = hsh(g.x);
  for (var r = 0u; r < rounds.x; r++) {
    let seed = hsh(acc ^ r); var s = seed;
    ${body}
  }
  outb[g.x] = acc;
}`;
}
const fill = (n: number, f: (i: number) => number) => { const a = new Uint32Array(n); for (let i = 0; i < n; i++) a[i] = f(i) >>> 0; return a; };
const rnd = (i: number) => { let x = Math.imul(i ^ 0x5bd1e995, 0x9e3779b1); x ^= x >>> 15; return Math.imul(x, 0x85ebca6b) >>> 0; };
// dir values never 0xffffffff (the NONE tests never hit); child words random (owner bit ~50/50).
const dirA = fill(DIRW, i => (rnd(i) & 0x7fffffff)), dataA = fill(DATAW, i => rnd(i + 99));
const mk = (a: Uint32Array, u: number) => { const b = device.createBuffer({ size: a.byteLength, usage: u | U.COPY_DST }); device.queue.writeBuffer(b, 0, a.buffer as ArrayBuffer); return b; };
const dir = mk(dirA, U.STORAGE), data = mk(dataA, U.STORAGE);
const outb = device.createBuffer({ size: 4 << 16, usage: U.STORAGE });
const uni = (r: number) => mk(new Uint32Array([r, 0, 0, 0]), U.UNIFORM);
const qs = device.createQuerySet({ type: "timestamp", count: 32 });
const qb = device.createBuffer({ size: 8 * 32, usage: U.QUERY_RESOLVE | U.COPY_SRC });
const qr = device.createBuffer({ size: 8 * 32, usage: U.MAP_READ | U.COPY_DST });
const heatWGSL = `@group(0) @binding(0) var<storage, read_write> o: array<u32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  var x = f32(g.x) * 1e-6; var y = x + 1.0;
  for (var i = 0u; i < 7450u; i++) { x = fma(x, 0.999, 0.25); y = fma(y, x, -0.5); }
  o[g.x & 65535u] = bitcast<u32>(x + y);
}`;
const heat = device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code: heatWGSL }), entryPoint: "main" } });
const heatBG = device.createBindGroup({ layout: heat.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outb } }] });

const REPS = 5;
async function time(pipe: GPUComputePipeline, bg: GPUBindGroup, G: number) {
  const enc = device.createCommandEncoder();
  { const p = enc.beginComputePass(); p.setPipeline(heat); p.setBindGroup(0, heatBG); p.dispatchWorkgroups(256); p.end(); }
  for (let r = 0; r < REPS; r++) {
    const p = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
    p.setPipeline(pipe); p.setBindGroup(0, bg); p.dispatchWorkgroups(G); p.end();
  }
  enc.resolveQuerySet(qs, 0, 2 * REPS, qb, 0); enc.copyBufferToBuffer(qb, 0, qr, 0, 16 * REPS);
  device.queue.submit([enc.finish()]);
  await qr.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qr.getMappedRange().slice(0, 16 * REPS)); qr.unmap();
  const us = [...Array(REPS).keys()].map(r => Number(t[2 * r + 1]! - t[2 * r]!) / 1e3);
  return Math.min(...us.slice(1));
}
emit({ kind: "setup", device: (adapter as any).info?.description, N, dirKB: DIRW * 4 / 1024, dataMB: DATAW * 4 / 2 ** 20 });
// Geometry: one 32-lane workgroup (one SIMD, latency alone) and 32 x 256 lanes (a group per core, the kernels' shape).
for (const [L, G] of [[32, 1], [256, 32]] as const) {
  for (const variant of ["free", "guard", "exit", "owner", "serial"]) {
    const module = device.createShaderModule({ code: wgsl(variant, L) });
    for (const m of (await module.getCompilationInfo()).messages) if (m.type === "error") throw new Error(`${variant}: ${m.lineNum}: ${m.message}`);
    const pipe = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });
    const res: Record<string, number> = {};
    for (const R of [4, 12]) {
      const bg = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [dir, data, outb, uni(R)].map((b, i) => ({ binding: i, resource: { buffer: b } })) });
      res[`R${R}`] = await time(pipe, bg, G);
    }
    // per round (slope between R = 4 and 12: the dispatch floor drops out)
    const perRound = (res.R12! - res.R4!) / 8;
    const rec = { kind: "branchy", variant, L, G, us4: res.R4, us12: res.R12, perRoundUs: perRound, gpuErrors };
    emit(rec); console.log(`${variant.padEnd(6)} L${L} G${G}: R4 ${res.R4!.toFixed(1)} us, R12 ${res.R12!.toFixed(1)} us, per round ${perRound.toFixed(2)} us`);
  }
}
emit({ kind: "done", gpuErrors });
console.log(gpuErrors ? `GPU ERRORS ${gpuErrors}` : "ok");
process.exit(0);
