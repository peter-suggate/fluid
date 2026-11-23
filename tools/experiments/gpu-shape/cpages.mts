// GPU-shape M3 (lane B2): are the FC passes' C pages on the critical path, and does reshaping them fix it?
// (docs/plans/gpu-shape/m3-cpages.md)
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/cpages.mts <scene> [outdir]
//   env STEPS=100 (warm-up)  R=7 (interleaved samples)  CR=3 (clocked samples)  MODE=crit,shape,clock,step (any subset)
//
// Warm the brick engine up STEPS steps (fixed dt 1/60, as census.mts), freeze it (a GPU shadow of every buffer), then:
//  dump   every FC placement (FC_G1 / FA_G1 / FCA_G1: F_ALL's bricks then C pages, a workgroup each) and every C-list
//         placement (C_ALL / C_ACT), with its workgroup count (the args word read in situ) and workgroup size.
//  crit   in situ (restore, ring, the step's placements before it untimed, the variant alone in a timestamped pass):
//         full / fonly (the C groups return at entry) / conly (the F groups return at entry) for every FC placement,
//         and full for every C-list placement.
//  shape  uTransfer and uAdvect reshaped (text variants of their C path): "first" (the C pages before the F bricks,
//         k = 1), and each page split over k = 2, 4, 8 workgroups (sub-blocks of its 512 tiles) either appended after
//         the F bricks ("appK") or ordered first ("firstK"); dispatched from an args word nF + k nC written by an
//         untimed one-lane kernel. Exactness from an in-situ snapshot: base, base again, every variant; every
//         read-write binding of the kernel word for word. Timing interleaved with base.
//  clock  per-workgroup start / end on a GPU tick counter: one extra workgroup (group 0, launched first) increments an
//         atomic until every payload group has finished (bounded); each payload group records the counter and an
//         entry / exit ticket at its start and end (the entry body moved into a helper, a barrier after it). Ticks
//         are calibrated against the pass's timestamps. For every FC placement's full, the shape variants, uExtend.
//  step   the whole step (both parts) with uTransfer / uAdvect swapped for a shape variant, walls interleaved.
// No lib/ change: variants are the schedule's own kernel text with substitutions, compiled as the engine compiles,
// bound with its bind() (m3clk joins the engine's buffer map in this process only).
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const scratch = resolve(homedir(), "code/me/fluid-scratch/gpu-shape");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats" };
const log = (...a: unknown[]) => console.log(...a);

function fnAt(code: string, name: string, from = 0): [number, number] {
  const m = new RegExp(`fn\\s+${name}\\s*\\(`).exec(code.slice(from)); if (!m) throw new Error(`no fn ${name}`);
  const i = from + m.index;
  let d = 0; for (let k = code.indexOf("{", i); k < code.length; k++) { if (code[k] === "{") d++; else if (code[k] === "}" && --d === 0) return [i, k + 1]; }
  throw new Error(`unterminated ${name}`);
}
/** The brace-matched block starting at the first "{" at or after i. */
function blockAt(code: string, i: number): [number, number] {
  let d = 0; for (let k = code.indexOf("{", i); k < code.length; k++) { if (code[k] === "{") d++; else if (code[k] === "}" && --d === 0) return [i, k + 1]; }
  throw new Error("unterminated block");
}
const count = (s: string, x: string) => s.split(x).length - 1;
const sub1 = (s: string, from: string, to: string, what: string) => {
  const n = count(s, from); if (n !== 1) throw new Error(`${what}: "${from.slice(0, 80)}" hits ${n}`);
  return s.replace(from, () => to);
};
/** The entry's parameter names (workgroup id, local index), its workgroup size and the span of its body. */
function entryOf(code: string, entry: string) {
  const [a, b] = fnAt(code, entry), head = code.slice(a, code.indexOf("{", a));
  const wid = /@builtin\(workgroup_id\)\s*(\w+)\s*:\s*vec3u/.exec(head)?.[1], lid = /@builtin\(local_invocation_index\)\s*(\w+)\s*:\s*u32/.exec(head)?.[1];
  const params = head.slice(head.indexOf("(") + 1, head.lastIndexOf(")")).split(",").filter(p => p.trim());
  if (!wid || !lid || params.length !== 2) throw new Error(`${entry}: entry parameters ${head}`);
  const before = code.slice(Math.max(0, a - 200), a), size = /@workgroup_size\((\d+)\)\s*$/.exec(before)?.[1];
  if (!size) throw new Error(`${entry}: no workgroup size`);
  const bodyStart = code.indexOf("{", a) + 1;
  return { a, b, wid, lid, size: Number(size), bodyStart, bodyEnd: b - 1, attrStart: a - (before.length - before.lastIndexOf("@compute")) };
}
/** Insert text at the start of the entry's body (plus module-scope declarations before the entry's attributes). */
function atEntryStart(code: string, entry: string, text: string, decl = ""): string {
  const e = entryOf(code, entry);
  return code.slice(0, e.attrStart) + decl + code.slice(e.attrStart, e.bodyStart) + text + code.slice(e.bodyStart);
}
// fonly / conly: the F groups are those below F_ALL's count (every FC kernel's split).
const groupGate = (code: string, entry: string, keep: "F" | "C") => {
  const e = entryOf(code, entry);
  return atEntryStart(code, entry, `\n if(${e.lid}==0u){m3S=nbListCount(NB_LIST_F_ALL);}if(nbGroup(${e.wid})${keep === "F" ? ">=" : "<"}workgroupUniformLoad(&m3S)){return;}`, "var<workgroup> m3S:u32;\n");
};

// ---- the shape variants (uTransfer, uAdvect) ----
const PAGE_CALL = /if\(g-split<workgroupUniformLoad\(&sSplit\)\)\{(\w+)\(nbListSlot\(NB_LIST_C_ACT,g-split\),lid\);\}/g;
/** The page function a staged FC head calls. */
function pageOf(code: string): string {
  const m = [...code.matchAll(PAGE_CALL)]; if (m.length !== 1) throw new Error(`page call hits ${m.length}`);
  return m[0]![1]!;
}
/** The page body split over k workgroups (m3Sub the sub-block): uTransfer's vcTransferPage, uAdvect's vcAdvectPage. */
function splitPage(code: string, k: number): string {
  const nt = 512 / k;
  if (code.includes("fn vcTransferPage(")) {
    const [a, b] = fnAt(code, "vcTransferPage"), f = code.slice(a, b);
    const LOOP = "for(var r=0u;r<2u;r++){\n  let j=lid+256u*r;let T=vcTile(slot,j);let occ=vOcc4(T,h.y);";
    if (count(f, LOOP) !== 1) throw new Error("vcTransferPage changed shape (its tile loop)");
    if (k === 2) return code.slice(0, a) + f.replace(LOOP, `for(var r=0u;r<1u;r++){\n  let j=lid+256u*m3Sub;let T=vcTile(slot,j);let occ=vOcc4(T,h.y);`) + code.slice(b);
    // k >= 4: a lane a (tile, axis) item, the tile's kick bits joined in workgroup memory, its flags word after a barrier.
    const head = f.slice(0, f.indexOf(LOOP)), AX = "for(var a=0u;a<3u;a++){";
    const ai = f.indexOf(AX); if (ai < 0 || count(f, AX) !== 1) throw new Error("vcTransferPage changed shape (its axis loop)");
    const [, ae] = blockAt(f, ai), axis = f.slice(ai, ae);
    const FLAGS = "let fw=NB_PHI_C_FLAGS+NB_STRIDE*slot+j;phi[fw]=(phi[fw]&~(7u*NB_FLAG_KICK0))|kick;";
    if (count(f.slice(ae), FLAGS) !== 1 || count(f, "var kick=0u;") !== 1) throw new Error("vcTransferPage changed shape (its flags)");
    const body = `${head}for(var m3e=lid;m3e<${3 * nt}u;m3e+=256u){
  let m3j=m3e%${nt}u;let m3a=m3e/${nt}u;let j=${nt}u*m3Sub+m3j;let T=vcTile(slot,j);let occ=vOcc4(T,h.y);
  var kick=0u;
  ${axis.replace(AX, "for(var a=m3a;a<m3a+1u;a++){")}
  atomicOr(&m3Kick[m3j],kick);
 }
 workgroupBarrier();
 for(var m3j=lid;m3j<${nt}u;m3j+=256u){let j=${nt}u*m3Sub+m3j;${FLAGS.replace("|kick;", "|atomicLoad(&m3Kick[m3j]);")}}
}`;
    return code.slice(0, a) + `var<workgroup> m3Kick:array<atomic<u32>,${nt}>;\n` + body + code.slice(b);
  }
  if (code.includes("fn vcAdvectPage(")) {
    const [a, b] = fnAt(code, "vcAdvectPage"), f = code.slice(a, b);
    const LOOP = "for(var r=0u;r<512u/A_LANES;r++){\n  let j=lid+A_LANES*r;";
    return code.slice(0, a) + sub1(f, LOOP, `for(var m3e=lid;m3e<${nt}u;m3e+=A_LANES){\n  let j=${nt}u*m3Sub+m3e;`, "vcAdvectPage") + code.slice(b);
  }
  throw new Error("no page body to split");
}
/** A shape variant: C pages appended (the engine's order) or first, each split over k workgroups. */
function shapeVariant(code: string, entry: string, order: "app" | "first", k: number): string {
  const page = pageOf(code);
  let c = k > 1 ? splitPage(code, k) : code;
  c = c.replace(/var<workgroup> sSplit:u32;/, "var<workgroup> sSplit:u32;\nvar<private> m3Sub:u32;");
  if (!c.includes("var<private> m3Sub:u32;")) throw new Error("no sSplit declaration");
  if (order === "app") {
    return c.replace(PAGE_CALL, () => `{let m3q=g-split;if(m3q<${k}u*workgroupUniformLoad(&sSplit)){m3Sub=m3q%${k}u;${page}(nbListSlot(NB_LIST_C_ACT,m3q/${k}u),lid);}}`);
  }
  // first: groups below k nC run the pages; the rest are the F bricks at nbGroup - k nC (and their own split's C
  // branch is never reached: the dispatch is exactly nF + k nC).
  const e = entryOf(c, entry);
  const body = c.slice(e.bodyStart, e.bodyEnd), n = count(body, `nbGroup(${e.wid})`);
  if (n < 1) throw new Error(`${entry}: no nbGroup in the entry`);
  const shifted = body.replaceAll(`nbGroup(${e.wid})`, `(nbGroup(${e.wid})-m3nc)`);
  const pre = `\n if(${e.lid}==0u){m3S=nbListCount(NB_LIST_C_ACT);}let m3nc=${k}u*workgroupUniformLoad(&m3S);let m3g=nbGroup(${e.wid});
 if(m3g<m3nc){m3Sub=m3g%${k}u;${page}(nbListSlot(NB_LIST_C_ACT,m3g/${k}u),${e.lid});return;}`;
  return c.slice(0, e.attrStart) + "var<workgroup> m3S:u32;\n" + c.slice(e.attrStart, e.bodyStart) + pre + shifted + c.slice(e.bodyEnd);
}

// ---- the clock ----
const CLK_HEAD = 8, CLK_REC = 4, CLK_MAX_GROUPS = 1 << 16, CLK_MAX_TICKS = 1 << 26;
// header: 0 the clock, 1 groups completed, 2 payload groups (the args kernel), 3 ticker state (1 capped, 2 done), 4 tickets
/** The clocked kernel: group 0 ticks; the entry body becomes m3Body (wid shifted down by one). */
function clocked(code: string, entry: string): string {
  const e = entryOf(code, entry);
  const body = code.slice(e.bodyStart, e.bodyEnd);
  const decl = `@group(0) @binding(${Math.max(-1, ...[...code.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map(m => Number(m[1]))) + 1}) var<storage,read_write> m3clk:array<atomic<u32>>;\n`;
  const helper = `fn m3Body(${e.wid}:vec3u,${e.lid}:u32){${body}}\n`;
  const entryText = `@compute @workgroup_size(${e.size}) fn ${entry}(@builtin(workgroup_id) m3wid:vec3u,@builtin(local_invocation_index) ${e.lid}:u32){
 if(m3wid.x==0u&&m3wid.y==0u){
  if(${e.lid}==0u){
   atomicStore(&m3clk[3],0u);var t=0u;
   loop{atomicAdd(&m3clk[0],1u);t++;if((t&255u)==0u&&atomicLoad(&m3clk[1])>=atomicLoad(&m3clk[2])){atomicStore(&m3clk[3],2u);break;}if(t>=${CLK_MAX_TICKS}u){atomicStore(&m3clk[3],1u);break;}}
  }
  return;
 }
 let m3l=m3wid.x+NB_FOLD*m3wid.y-1u;let m3r=${CLK_HEAD}u+${CLK_REC}u*min(m3l,${CLK_MAX_GROUPS - 1}u);
 if(${e.lid}==0u){atomicStore(&m3clk[m3r],atomicLoad(&m3clk[0]));atomicStore(&m3clk[m3r+2u],atomicAdd(&m3clk[4],1u));}
 m3Body(vec3u(m3l%NB_FOLD,m3l/NB_FOLD,0u),${e.lid});
 workgroupBarrier();
 if(${e.lid}==0u){atomicStore(&m3clk[m3r+1u],atomicLoad(&m3clk[0]));atomicStore(&m3clk[m3r+3u],atomicAdd(&m3clk[4],1u));atomicAdd(&m3clk[1],1u);}
}`;
  return decl + code.slice(0, e.attrStart) + helper + entryText + code.slice(e.b);
}
// The args of a variant: n = args[x] + z args[y] + w (w = 1: the ticker), one-dimensional (fold 32768).
const ARGS_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage,read> ia:array<u32>;
@group(0) @binding(1) var<storage,read_write> oa:array<u32>;
@group(0) @binding(2) var<storage,read_write> oc:array<u32>;
@group(0) @binding(3) var<uniform> u:vec4u;
@compute @workgroup_size(1) fn main(){let n=ia[u.x]+u.z*ia[u.y];oa[0]=n+u.w;oa[1]=1u;oa[2]=1u;oc[2]=n;}`;
const COMPARE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage,read> ca:array<u32>;
@group(0) @binding(1) var<storage,read> cb:array<u32>;
@group(0) @binding(2) var<storage,read_write> co:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cn:vec4u;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
 for(var i=g.x;i<cn.x;i+=cn.y){if(ca[i]!=cb[i]){atomicAdd(&co[0],1u);atomicMin(&co[1],i+cn.z);}}
}`;
const SHAPES: [string, "app" | "first", number][] = [["first", "first", 1], ["app2", "app", 2], ["app4", "app", 4], ["app8", "app", 8], ["first2", "first", 2], ["first4", "first", 4], ["first8", "first", 8]];

if (process.argv[2] === "--check") {
  // Offline: node cpages.mts --check <kernel.wgsl> <entry> : every variant applies (writes them next to the file).
  const [f, entry] = process.argv.slice(3) as [string, string]; const c = readFileSync(f, "utf8");
  const out: Record<string, string> = { fonly: groupGate(c, entry, "F"), conly: groupGate(c, entry, "C"), clk: clocked(c, entry) };
  try { pageOf(c); for (const [n, o, k] of SHAPES) { out[n] = shapeVariant(c, entry, o, k); out[`${n}.clk`] = clocked(out[n]!, entry); } } catch (e) { log(`no shape variants: ${(e as Error).message}`); }
  for (const [n, t] of Object.entries(out)) writeFileSync(`${f}.${n}.wgsl`, t);
  log(`${f}: ${Object.keys(out).join(",")}`); process.exit(0);
}
await main();

async function main() {
  await import(`${root}/lib/methods/index.ts`);
  const { getMethod } = await import(`${root}/lib/core/method-registry.ts`);
  const { resolveMethodValues } = await import(`${root}/lib/core/method-contract.ts`);
  const { sceneDocument } = await import(`${root}/lib/core/scene-definition.ts`);
  const { getSceneDefinition } = await import(`${root}/lib/core/scenes.ts`);
  const { managedGPUDevice } = await import(`${root}/lib/core/gpu-compilation-manager.ts`);
  const { requiredFluidDeviceLimits } = await import(`${root}/lib/core/webgpu-device-limits.ts`);
  const { createProcessRetainedDawnGPU, paceDawnEventPolling } = await import(`${root}/lib/harness/node-dawn-provider.ts`);
  const { mooredMotorBodies } = await import(`${root}/tools/moored-motor-bodies.ts`);
  const { RING_SLOTS, RING_WORDS, ARGS, LIST } = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
  const { bindGroupLayoutEntries } = await import(`${root}/lib/methods/narrow-band/brick-kernel.ts`);
  const { brickModuleCode, brickLowerOn } = await import(`${root}/lib/methods/narrow-band/brick-lower.ts`);
  const { createClockProbe } = await import(`${here}/clock-probe.ts`);

  const [sceneKey = "dam", outArg] = process.argv.slice(2), sceneId = SCENES[sceneKey] ?? sceneKey;
  const out = resolve(outArg ?? resolve(scratch, "m3", sceneKey)); mkdirSync(resolve(out, "wgsl"), { recursive: true });
  const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 7), CR = Number(process.env.CR ?? 3);
  const modes = new Set((process.env.MODE ?? "crit,shape,clock").split(","));
  const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
  const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
  const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
  const quality = profile?.methodId === methodId ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
  const head = (() => { try { return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(); } catch { return "?"; } })();
  log(`cpages scene ${sceneId} (${sceneKey}) HEAD ${head} warm-up ${N}, R ${R}, CR ${CR}, modes ${[...modes].join(",")}, out ${out}`);
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
  const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const gpuErrors: string[] = [];
  raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors.push(String(ev.error?.message)); console.log("GPU ERROR", ev.error?.message); });
  const fail = (where: string) => { if (gpuErrors.length) throw new Error(`GPU errors ${where}: ${gpuErrors[0]}`); };
  const device = managedGPUDevice(raw, { requireWorkerRealm: false });
  const s: any = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {});
  await s.waitForSimulationReady?.();
  s.setPresentationEnabled(false);
  const dt = 1 / 60;
  const bodies: (t: number) => unknown[] = sceneKey === "paddleboats" || (scene.rigidBodies?.length ?? 0) > 0 ? mooredMotorBodies(scene, dt) : () => [];
  const t0 = performance.now();
  for (let k = 1; k <= N; k++) {
    while (!s.advanceTo(k * dt + 1e-12, bodies(k * dt))) {
      if (s.info.simulationFailure) throw new Error(`step ${k}: ${s.info.simulationFailure.message}`);
      await new Promise<void>(r => setImmediate(r));
    }
    if (k % 25 === 0) { await s.brickEngine.fence(); log(`warm-up ${k} ${((performance.now() - t0) / k).toFixed(1)} ms/step`); }
  }
  const P = s.brickEngine;
  await P.fence(); fail("in the warm-up");

  const L = P.layout, slot = P.steps % RING_SLOTS, rec = P.recorded[slot], args: GPUBuffer = P.buffers.get("args");
  const sch = [...P.schedule.part1, ...P.schedule.part2], all = [...rec.part1, ...rec.part2], n1 = rec.part1.length;
  if (sch.length !== all.length) throw new Error("recorded placements differ from the schedule");
  const result: any = { scene: sceneId, key: sceneKey, head, warmup: N, R, CR, steps: P.steps, date: new Date().toISOString(),
    status: (() => { try { return execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }); } catch { return ""; } })() };
  const save = () => writeFileSync(resolve(out, "cpages.json"), JSON.stringify(result, null, 1));

  // ---- dump ----
  const FCN: Record<number, string> = { [ARGS.FC_G1]: "FC_G1", [ARGS.FA_G1]: "FA_G1", [ARGS.FCA_G1]: "FCA_G1", [ARGS.FC_G8]: "FC_G8", [ARGS.FCA_G8]: "FCA_G8", [ARGS.FA_G8]: "FA_G8" };
  const listName = (k: number) => Object.entries(LIST).find(([, v]) => v === k)?.[0] ?? `L${k}`;
  const tagOf = (d: any) => {
    if (!("indirect" in d)) return { tag: `direct ${JSON.stringify(d.direct)}`, kind: "" };
    const x = d.indirect as number; if (FCN[x]) return { tag: FCN[x]!, kind: "FC" };
    if (x < ARGS.FC_G1) { const l = Math.floor(x / ARGS.PER_LIST), g = ["G1", "", "", "G8", "", "", "G64", "", "", "X8"][x % ARGS.PER_LIST]; const n = listName(l); return { tag: `${n}.${g}`, kind: n.startsWith("C_") ? "C" : "" }; }
    return { tag: `args ${x}`, kind: "" };
  };
  const fcs: number[] = [], cs: number[] = [];
  result.placements = {};
  for (let i = 0; i < sch.length; i++) {
    const p = sch[i], { tag, kind } = tagOf(p.dispatch); if (!kind) continue;
    let size = 0; try { size = entryOf(p.kernel.code, p.kernel.entry).size; } catch { /* */ }
    (kind === "FC" ? fcs : cs).push(i);
    result.placements[i] = { label: p.kernel.label, entry: p.kernel.entry, part: i < n1 ? 1 : 2, tag, kind, size };
  }
  const kdir = resolve(out, "kernels"); mkdirSync(kdir, { recursive: true });
  for (const i of [...fcs, ...cs]) writeFileSync(resolve(kdir, `${sch[i].kernel.label}.wgsl`), sch[i].kernel.code);
  { const ue = sch.findIndex((p: any) => p.kernel.label === "uExtend"); if (ue >= 0) writeFileSync(resolve(kdir, "uExtend.wgsl"), sch[ue].kernel.code); }

  // ---- the frozen state ----
  const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const live = [...[...P.buffers.values()] as GPUBuffer[], ...[...new Set([...P.external.values()])].filter((b: GPUBuffer) => (b.usage & COPY) === COPY)] as GPUBuffer[];
  const shadow = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
  const copyAll = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  copyAll(live, shadow);
  const restore = () => copyAll(shadow, live);
  const ring = () => raw.queue.writeBuffer(P.buffers.get("set"), 4 * (L.set.ring + RING_WORDS * slot), P.ringWords(P.steps));
  const dispatchInto = (pass: GPUComputePassEncoder, r: any) => {
    pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
    if (r.argsBuf) pass.dispatchWorkgroupsIndirect(r.argsBuf, 0);
    else if ("direct" in r.dispatch) pass.dispatchWorkgroups(...(r.dispatch.direct as [number, number, number])); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
  };
  const prefixInto = (enc: GPUCommandEncoder, i: number) => {
    const p1 = all.slice(0, Math.min(i, n1)), p2 = i > n1 ? all.slice(n1, i) : [];
    for (const list of [p1, p2]) if (list.length) { const pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end(); }
  };
  const QMAX = 8, qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
  const qResolve = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readBack = async (enc: GPUCommandEncoder, n: number, extra?: { from: GPUBuffer; to: GPUBuffer; size: number }) => {
    enc.resolveQuerySet(qs, 0, n, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 8 * n);
    if (extra) enc.copyBufferToBuffer(extra.from, 0, extra.to, 0, extra.size);
    raw.queue.submit([enc.finish()]);
    await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange(0, 8 * n).slice(0)); qRead.unmap();
    let w: Uint32Array | undefined; if (extra) { await extra.to.mapAsync(GPUMapMode.READ); w = new Uint32Array(extra.to.getMappedRange(0, extra.size).slice(0)); extra.to.unmap(); }
    return { t, w };
  };
  const probe = createClockProbe(raw, { capacity: 4 });
  const clock = async () => { const enc = raw.createCommandEncoder(); probe.encode(enc); raw.queue.submit([enc.finish()]); return (await probe.read())[0]!; };

  // The args words in situ at each placement (the head writes them; part 2's set stage may rewrite them).
  const argsRead = raw.createBuffer({ size: 4 * ARGS.WORDS, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const argsAt = async (i: number) => {
    restore(); ring(); const enc = raw.createCommandEncoder(); prefixInto(enc, i); enc.copyBufferToBuffer(args, 0, argsRead, 0, 4 * ARGS.WORDS); raw.queue.submit([enc.finish()]);
    await argsRead.mapAsync(GPUMapMode.READ); const w = new Uint32Array(argsRead.getMappedRange().slice(0)); argsRead.unmap(); return w;
  };
  const G = (w: Uint32Array, at: number) => w[at]! * w[at + 1]!;
  for (const i of [...fcs, ...cs]) {
    const w = await argsAt(i), e = result.placements[i], d = sch[i].dispatch;
    e.groups = G(w, d.indirect); e.nF = G(w, ARGS.PER_LIST * LIST.F_ALL); e.nCAll = G(w, ARGS.PER_LIST * LIST.C_ALL); e.nCAct = G(w, ARGS.PER_LIST * LIST.C_ACT);
  }
  fail("reading the args");
  log(`schedule ${sch.length} placements (part1 ${n1}); FC and C-list placements (groups, workgroup size; nF / nC_ALL / nC_ACT):`);
  for (const i of [...fcs, ...cs].sort((a, b) => a - b)) { const e = result.placements[i]; log(`  ${String(i).padStart(3)} p${e.part} ${e.kind.padEnd(2)} ${e.label.padEnd(18)} ${e.tag.padEnd(10)} groups ${String(e.groups).padStart(6)} x ${e.size}  (${e.nF} / ${e.nCAll} / ${e.nCAct})`); }
  save();

  const lower = brickLowerOn(P.options.lowerDynamicWrites);
  const clk = raw.createBuffer({ label: "m3clk", size: 4 * (CLK_HEAD + CLK_REC * CLK_MAX_GROUPS), usage: GPUBufferUsage.STORAGE | COPY });
  const clkRead = raw.createBuffer({ size: clk.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  P.buffers.set("m3clk", clk);
  const argsMod = raw.createShaderModule({ code: ARGS_WGSL }), argsPipe = raw.createComputePipeline({ layout: "auto", compute: { module: argsMod, entryPoint: "main" } });
  /** A one-lane kernel writing a variant's args (n = args[x] + z args[y] + w) and the clock's payload count. */
  const argsRec = (x: number, y: number, z: number, w: number) => {
    const argsBuf = raw.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | COPY });
    const u = raw.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true }); new Uint32Array(u.getMappedRange()).set([x, y, z, w]); u.unmap();
    const group = raw.createBindGroup({ layout: argsPipe.getBindGroupLayout(0), entries: [args, argsBuf, clk, u].map((b, k) => ({ binding: k, resource: { buffer: b } })) });
    return { pre: { pipeline: argsPipe, group, dispatch: { direct: [1, 1, 1] } }, argsBuf };
  };
  /** Compile a kernel variant (extra: the m3clk binding) and bind it at placement i. */
  const variantRec = async (i: number, label: string, code: string, clockOn = false) => {
    const k0 = sch[i].kernel, bindings = clockOn ? [...k0.bindings, { name: "m3clk", kind: "rw", type: "array<u32>" }] : k0.bindings;
    const kv = { ...k0, label, code, bindings };
    const text = brickModuleCode(label, code, lower);
    writeFileSync(resolve(out, "wgsl", `${label}.wgsl`), text);
    const module = raw.createShaderModule({ label, code: text });
    const errs = (await module.getCompilationInfo()).messages.filter((m: any) => m.type === "error");
    if (errs.length) throw new Error(`${label} does not compile:\n${errs.slice(0, 5).map((m: any) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
    const entries = bindGroupLayoutEntries(k0.bindings);
    if (clockOn) entries.push({ binding: entries.length, visibility: 4, buffer: { type: "storage" } });
    const layout = raw.createPipelineLayout({ bindGroupLayouts: [raw.createBindGroupLayout({ entries })] });
    P.pipelines.set(kv, await raw.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: kv.entry } }));
    return P.bind({ ...sch[i], kernel: kv }, slot);
  };
  /** In situ: restore, ring, placements [0, i) untimed, `pre` untimed, then `timed` alone in a timestamped pass. */
  const sample = async (i: number, timed: any, pre: any[] = [], clockOn = false) => {
    restore(); ring();
    const enc = raw.createCommandEncoder(); prefixInto(enc, i);
    if (clockOn) enc.clearBuffer(clk);
    if (pre.length) { const pass = enc.beginComputePass(); for (const x of pre) dispatchInto(pass, x); pass.end(); }
    const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }); dispatchInto(pass, timed); pass.end();
    const { t, w } = await readBack(enc, 2, clockOn ? { from: clk, to: clkRead, size: clk.size } : undefined);
    return { us: Number(t[1]! - t[0]!) / 1e3, w };
  };
  const stats = (v: number[]) => { const a = [...v].sort((x, y) => x - y); return { min: a[0]!, med: a[a.length >> 1]! }; };
  const rounds = async (cases: { key: string; run: () => Promise<number> }[], n = R) => {
    const t = new Map<string, number[]>(cases.map(c => [c.key, []])); const probes: number[] = [];
    for (let r = 0; r < n; r++) { for (const c of r % 2 ? [...cases].reverse() : cases) t.get(c.key)!.push(await c.run()); probes.push(await clock()); fail("in the timing"); }
    return { t, probes };
  };
  // A variant record with its own args (the shape variants' nF + k nC; the clock's + 1).
  const shapeArgs = (k: number, w: number) => argsRec(ARGS.PER_LIST * LIST.F_ALL, ARGS.PER_LIST * LIST.C_ACT, k, w);
  const sameArgs = (i: number, w: number) => argsRec(sch[i].dispatch.indirect, 0, 0, w);
  const shapeIs = (i: number) => ["uTransfer", "uAdvect"].includes(sch[i].kernel.label) && sch[i].dispatch.indirect === ARGS.FCA_G1;
  const variants = new Map<string, { rec: any; pre: any[] }>();
  const V = async (i: number, name: string, make: () => string, a?: { pre: any; argsBuf: GPUBuffer }, clockOn = false) => {
    const key = `${i}\t${name}`; let v = variants.get(key);
    if (!v) { const r = await variantRec(i, `${sch[i].kernel.label}.m3.${name}.${i}`, make(), clockOn); v = { rec: a ? { ...r, argsBuf: a.argsBuf } : r, pre: a ? [a.pre] : [] }; variants.set(key, v); }
    return v;
  };

  // ---- crit: full / fonly / conly; C-list placements full ----
  if (modes.has("crit")) {
    const cases: { key: string; run: () => Promise<number> }[] = [];
    for (const i of fcs) {
      const k = sch[i].kernel;
      const fo = await V(i, "fonly", () => groupGate(k.code, k.entry, "F")), co = await V(i, "conly", () => groupGate(k.code, k.entry, "C"));
      cases.push({ key: `${i}\tfull`, run: async () => (await sample(i, all[i])).us }, { key: `${i}\tfonly`, run: async () => (await sample(i, fo.rec)).us }, { key: `${i}\tconly`, run: async () => (await sample(i, co.rec)).us });
    }
    for (const i of cs) cases.push({ key: `${i}\tfull`, run: async () => (await sample(i, all[i])).us });
    fail("building the crit variants");
    log(`crit: ${cases.length} cases x ${R}`);
    const { t, probes } = await rounds(cases);
    result.crit = { probes };
    let cSum = 0;
    for (const i of [...fcs, ...cs].sort((a, b) => a - b)) {
      const e = result.placements[i], us: any = {};
      for (const v of ["full", "fonly", "conly"]) { const x = t.get(`${i}\t${v}`); if (x) us[v] = { ...stats(x), samples: x }; }
      e.crit = us;
      if (e.kind === "C") { cSum += us.full.min; log(`  ${String(i).padStart(3)} C  ${e.label.padEnd(18)} full ${us.full.min.toFixed(1).padStart(7)} us  (${e.groups} groups)`); }
      else log(`  ${String(i).padStart(3)} FC ${e.label.padEnd(18)} full ${us.full.min.toFixed(1).padStart(7)}  fonly ${us.fonly.min.toFixed(1).padStart(7)}  conly ${us.conly.min.toFixed(1).padStart(7)}  full-fonly ${(us.full.min - us.fonly.min).toFixed(1).padStart(7)} us  (nF ${e.nF}, groups ${e.groups})`);
    }
    result.crit.cListSumUs = cSum;
    log(`crit: C-list placements sum ${cSum.toFixed(1)} us; clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }

  // ---- shape: exactness, then timing ----
  const shapeAt = fcs.filter(shapeIs);
  if (modes.has("shape")) {
    result.shape = {};
    const cmpMod = raw.createShaderModule({ code: COMPARE_WGSL }), cmpPipe = raw.createComputePipeline({ layout: "auto", compute: { module: cmpMod, entryPoint: "main" } });
    const cmpOut = raw.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const cmpRead = raw.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const compare = async (a: GPUBuffer, b: GPUBuffer): Promise<[number, number]> => {
      raw.queue.writeBuffer(cmpOut, 0, new Uint32Array([0, 0xffffffff, 0, 0]));
      const enc = raw.createCommandEncoder(); const CH = 2 ** 30;
      for (let off = 0; off < a.size; off += CH) {
        const size = Math.min(CH, a.size - off), u = raw.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Uint32Array(u.getMappedRange()).set([size / 4, 1024 * 256, off / 4, 0]); u.unmap();
        const bg = raw.createBindGroup({ layout: cmpPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a, offset: off, size } }, { binding: 1, resource: { buffer: b, offset: off, size } }, { binding: 2, resource: { buffer: cmpOut } }, { binding: 3, resource: { buffer: u } }] });
        const pass = enc.beginComputePass(); pass.setPipeline(cmpPipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1024); pass.end();
      }
      enc.copyBufferToBuffer(cmpOut, 0, cmpRead, 0, 16); raw.queue.submit([enc.finish()]);
      await cmpRead.mapAsync(GPUMapMode.READ); const w = new Uint32Array(cmpRead.getMappedRange().slice(0)); cmpRead.unmap(); return [w[0]!, w[1]!];
    };
    const bufOf = (name: string): GPUBuffer => P.buffers.get(name) ?? P.external.get(name);
    const refs = new Map<GPUBuffer, GPUBuffer>();
    const refOf = (b: GPUBuffer) => { let r = refs.get(b); if (!r) { r = raw.createBuffer({ size: b.size, usage: COPY | GPUBufferUsage.STORAGE }); refs.set(b, r); } return r; };
    const snap = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
    const fromSnap = async (list: any[]) => {
      copyAll(snap, live); ring();
      const enc = raw.createCommandEncoder(), pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end();
      raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone();
    };
    for (const i of shapeAt) {
      const k = sch[i].kernel, vs: Record<string, { rec: any; pre: any[] }> = {};
      for (const [name, order, kk] of SHAPES) vs[name] = await V(i, name, () => shapeVariant(k.code, k.entry, order, kk), shapeArgs(kk, 0));
      fail("building the shape variants");
      // Exactness from one in-situ snapshot.
      restore(); ring(); { const enc = raw.createCommandEncoder(); prefixInto(enc, i); raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone(); }
      copyAll(live, snap); await raw.queue.onSubmittedWorkDone();
      const names = k.bindings.filter((b: any) => b.kind === "rw").map((b: any) => b.name as string), bufs: GPUBuffer[] = names.map(bufOf);
      await fromSnap([all[i]]); { const enc = raw.createCommandEncoder(); for (const b of new Set(bufs)) enc.copyBufferToBuffer(b, 0, refOf(b), 0, b.size); raw.queue.submit([enc.finish()]); }
      const ex: Record<string, number> = {};
      for (const [v, list] of [["base", [all[i]]], ...Object.entries(vs).map(([n, x]) => [n, [...x.pre, x.rec]])] as [string, any[]][]) {
        await fromSnap(list); let d = 0; for (const b of bufs) d += (await compare(b, refOf(b)))[0]; ex[v] = d;
      }
      result.shape[i] = { label: k.label, rw: names, exact: ex };
      log(`  exact ${i} ${k.label} (${names.join(",")}): ${Object.entries(ex).map(([v, d]) => `${v} ${d}`).join(", ")} words differ`);
    }
    for (const b of snap) b.destroy();
    fail("in the exactness check");
    const cases: { key: string; run: () => Promise<number> }[] = [];
    for (const i of shapeAt) {
      cases.push({ key: `${i}\tbase`, run: async () => (await sample(i, all[i])).us });
      for (const [name] of SHAPES) { const v = variants.get(`${i}\t${name}`)!; cases.push({ key: `${i}\t${name}`, run: async () => (await sample(i, v.rec, v.pre)).us }); }
    }
    const { t, probes } = await rounds(cases);
    result.shapeProbes = probes;
    for (const i of shapeAt) {
      const c = result.shape[i]; c.us = {};
      for (const v of ["base", ...SHAPES.map(x => x[0])]) c.us[v] = { ...stats(t.get(`${i}\t${v}`)!), samples: t.get(`${i}\t${v}`) };
      log(`  ${String(i).padStart(3)} ${c.label.padEnd(10)} base ${c.us.base.min.toFixed(1)}  ${SHAPES.map(([v]) => `${v} ${c.us[v].min.toFixed(1)} (${(c.us[v].min - c.us.base.min).toFixed(1)})`).join("  ")} us`);
    }
    log(`shape: clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }

  // ---- clock: per-workgroup durations ----
  if (modes.has("clock")) {
    result.clock = {};
    const ue = sch.findIndex((p: any) => p.kernel.label === "uExtend");
    const targets: { i: number; name: string; code: () => string; a: () => { pre: any; argsBuf: GPUBuffer }; cFirst: number }[] = [];
    const which = new Set((process.env.CLOCK ?? "fc,shape,uext,clist").split(","));
    if (which.has("clist")) for (const i of cs) { const e = result.placements[i]; if ((e.crit?.full?.min ?? 1e9) < Number(process.env.CLIST_MIN_US ?? 150)) continue; const k = sch[i].kernel; targets.push({ i, name: "full", code: () => clocked(k.code, k.entry), a: () => sameArgs(i, 1), cFirst: 0 }); }
    if (which.has("fc")) for (const i of fcs) { const k = sch[i].kernel; targets.push({ i, name: "full", code: () => clocked(k.code, k.entry), a: () => sameArgs(i, 1), cFirst: 0 }); }
    if (which.has("shape")) for (const i of shapeAt) { const k = sch[i].kernel; for (const [name, order, kk] of SHAPES) targets.push({ i, name, code: () => clocked(shapeVariant(k.code, k.entry, order, kk), k.entry), a: () => shapeArgs(kk, 1), cFirst: order === "first" ? kk : 0 }); }
    if (which.has("uext") && ue >= 0) { const k = sch[ue].kernel; targets.push({ i: ue, name: "full", code: () => clocked(k.code, k.entry), a: () => sameArgs(ue, 1), cFirst: 0 }); }
    const unclocked = new Map<string, number>();
    for (const tg of targets) {
      const v = await V(tg.i, `${tg.name}.clk`, tg.code, tg.a(), true);
      const e = result.placements[tg.i] ?? { label: sch[tg.i].kernel.label }, runs: any[] = [];
      for (let r = 0; r < CR; r++) {
        const { us, w } = await sample(tg.i, v.rec, v.pre, true); fail(`clocking ${e.label}`);
        const n = w![2]!, ticks = w![0]!, state = w![3]!, done = w![1]!;
        const nF = e.nF ?? n, nC = n - nF, recs: { g: number; s: number; e: number; a: number; b: number; c: boolean }[] = [];
        for (let g = 0; g < Math.min(n, CLK_MAX_GROUPS); g++) {
          const o = CLK_HEAD + CLK_REC * g, isC = tg.cFirst ? g < tg.cFirst * (e.nCAct ?? 0) : g >= nF;
          recs.push({ g, s: w![o]!, e: w![o + 1]!, a: w![o + 2]!, b: w![o + 3]!, c: e.kind === "FC" ? isC : false });
        }
        const rate = ticks / us; // ticks a µs
        const T = (x: number) => x / rate;
        const span = (xs: typeof recs) => xs.length ? { start: Math.min(...xs.map(x => x.s)), end: Math.max(...xs.map(x => x.e)) } : undefined;
        const dist = (xs: typeof recs) => { const d = xs.map(x => T(x.e - x.s)).sort((p, q) => p - q); return d.length ? { n: d.length, p10: d[Math.floor(0.1 * (d.length - 1))]!, p50: d[d.length >> 1]!, p90: d[Math.floor(0.9 * (d.length - 1))]!, max: d[d.length - 1]!, sum: d.reduce((p, q) => p + q, 0) } : undefined; };
        const F = recs.filter(x => !x.c), C = recs.filter(x => x.c), sF = span(F), sC = span(C), sAll = span(recs)!;
        // Groups in flight over time (16 bins of the span).
        const bins = 16, inflight = Array.from({ length: bins }, (_, b) => { const t0b = sAll.start + (b + 0.5) * (sAll.end - sAll.start) / bins; return [recs.filter(x => !x.c && x.s <= t0b && x.e > t0b).length, recs.filter(x => x.c && x.s <= t0b && x.e > t0b).length]; });
        const run = { passUs: us, ticks, rate, state, done, n, nF: F.length, nC: C.length, spanUs: T(sAll.end - sAll.start),
          F: sF && { startUs: T(sF.start - sAll.start), endUs: T(sF.end - sAll.start), dur: dist(F) }, C: sC && { startUs: T(sC.start - sAll.start), endUs: T(sC.end - sAll.start), dur: dist(C) },
          tailUs: sF && sC ? T(Math.max(0, sC.end - sF.end)) : 0, inflight };
        runs.push(run);
        if (r === 0) writeFileSync(resolve(out, `clock-${e.label}-${tg.name}-${tg.i}.tsv`), "g\tc\tstartUs\tendUs\tentry\texit\n" + recs.map(x => `${x.g}\t${x.c ? 1 : 0}\t${T(x.s - sAll.start).toFixed(2)}\t${T(x.e - sAll.start).toFixed(2)}\t${x.a}\t${x.b}`).join("\n"));
      }
      // The same variant unclocked's pass (the clock's overhead): full is the schedule's own.
      let plain = unclocked.get(`${tg.i}\t${tg.name}`);
      if (plain === undefined) {
        const pv = tg.name === "full" ? { rec: all[tg.i], pre: [] } : variants.get(`${tg.i}\t${tg.name}`);
        if (pv) { const xs: number[] = []; for (let r = 0; r < CR; r++) xs.push((await sample(tg.i, pv.rec, pv.pre)).us); plain = Math.min(...xs); unclocked.set(`${tg.i}\t${tg.name}`, plain); }
      }
      const best = runs.reduce((p, q) => (q.passUs < p.passUs ? q : p));
      result.clock[`${tg.i}\t${tg.name}`] = { label: e.label, name: tg.name, plainUs: plain, runs };
      const f = (x?: number) => (x === undefined ? "-" : x.toFixed(1));
      log(`  clock ${String(tg.i).padStart(3)} ${e.label.padEnd(10)} ${tg.name.padEnd(7)} pass ${f(best.passUs)} (plain ${f(plain)}) us, state ${best.state}, ${best.done}/${best.n} groups, ${best.rate.toFixed(1)} ticks/us; ` +
        `F ${best.nF} start ${f(best.F?.startUs)} end ${f(best.F?.endUs)} dur p50 ${f(best.F?.dur?.p50)} p90 ${f(best.F?.dur?.p90)} max ${f(best.F?.dur?.max)}; ` +
        `C ${best.nC} start ${f(best.C?.startUs)} end ${f(best.C?.endUs)} dur p50 ${f(best.C?.dur?.p50)} p90 ${f(best.C?.dur?.p90)} max ${f(best.C?.dur?.max)}; tail ${f(best.tailUs)} us`);
      log(`        in flight (F,C) over 16 bins: ${best.inflight.map((x: number[]) => `${x[0]},${x[1]}`).join(" ")}`);
      save();
    }
    log(`clock: probe ${(await clock()).toFixed(1)} us`);
    save();
  }

  // ---- step: whole step walls with a shape variant swapped in ----
  if (modes.has("step")) {
    const pick = (process.env.STEP_VARIANTS ?? "first2,uTransfer=first2+uAdvect=app4").split(",");
    const lists: Record<string, any[]> = { base: [] };
    for (const v of pick) lists[v] = [];
    for (let i = 0; i < all.length; i++) {
      lists.base!.push(all[i]);
      // A config is a shape name for both kernels, or "uTransfer=first2+uAdvect=app4" (a kernel not named keeps base).
      for (const v of pick) {
        const per = v.includes("=") ? Object.fromEntries(v.split("+").map(x => x.split("="))) : undefined;
        const name = !shapeAt.includes(i) ? undefined : per ? per[sch[i].kernel.label] : v;
        const x = name ? variants.get(`${i}\t${name}`) : undefined; if (name && !x) throw new Error(`step: no shape variant ${name} at ${i} (MODE needs shape)`);
        if (x) lists[v]!.push(...x.pre, x.rec); else lists[v]!.push(all[i]);
      }
    }
    const wall = async (v: string) => {
      restore(); ring();
      const enc = raw.createCommandEncoder(), l = lists[v]!, cut = n1;
      const a = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } }); for (const x of l.slice(0, cut)) dispatchInto(a, x); a.end();
      const b = enc.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: 1 } }); for (const x of l.slice(cut)) dispatchInto(b, x); b.end();
      const { t } = await readBack(enc, 2); return Number(t[1]! - t[0]!) / 1e3;
    };
    if (shapeAt.some(i => i < n1)) throw new Error("step: a shape placement in part 1");
    const { t, probes } = await rounds(Object.keys(lists).map(v => ({ key: v, run: () => wall(v) })));
    result.step = { probes };
    for (const v of Object.keys(lists)) result.step[v] = { ...stats(t.get(v)!), samples: t.get(v) };
    const b0 = result.step.base.min;
    log(`step wall (min, delta, median): ${Object.keys(lists).map(v => `${v} ${(result.step[v].min / 1e3).toFixed(3)} (${((result.step[v].min - b0) / 1e3).toFixed(3)}) med ${(result.step[v].med / 1e3).toFixed(3)}`).join(" | ")} ms; probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }
  for (const b of shadow) b.destroy();
  s.destroy?.();
  process.exit(gpuErrors.length ? 3 : 0);
}
