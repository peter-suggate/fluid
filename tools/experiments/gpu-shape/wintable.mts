// GPU-shape M2 (lane B2): a per-step window table go/no-go (docs/plans/gpu-shape/m2-window-table.md).
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/wintable.mts <scene> [outdir]
//   env STEPS=100 (warm-up)  R=5 (interleaved samples)  MODE=dump,ceiling,proto,step (any subset; default all)
//       ONLY=<label,...> (staging kernels to measure; default every one in the schedule)
//
// Warm the brick engine up STEPS steps (fixed dt 1/60, as census.mts), freeze it (a GPU shadow of every buffer), then:
//  dump     the schedule: every placement's label, part, dispatch, bindings, whether its entry calls the window
//           (fgWindowStage/fgWindowStageAt) or ring (fgRingStage) staging, and whether it binds `dir` read-write.
//  ceiling  per staging placement, in situ (restore, ring words, the step's placements before it untimed, the variant
//           alone in a timestamped pass): text variants that end the entry before the window call ("pre"), after the
//           barrier that follows the window ("win"), and after the ring's barrier ("ring"); window = win - pre,
//           ring = ring - win (staged.mts's truncation method, generic cuts).
//  proto    a window table m2win (256 words an F slot: 27 brick slots, 216 tile words, 7 coarse-bit words, as
//           fgWindowStageAt stages them), written by one kernel m2Build (a workgroup an F_ALL brick, the directory
//           walks of fgWindowStageAt once), and two variants of each staging kernel: "tabA" stages the window's
//           workgroup arrays from the table (coalesced loads, no walks); "tabB" drops the arrays and reads the table
//           in place. Exactness: from an in-situ snapshot (the placements before it, with the single build before
//           the first window user), base / base again / tabA / tabB, every read-write buffer of the kernel word for
//           word, and the snapshot's table against a fresh build there. Timing: in situ, base / tabA / tabB
//           interleaved (the table built untimed just before), and m2Build alone at the same point; the clock probe
//           after every round. EXACT_ONLY=1 stops after the exactness check.
//  step     the whole step (both parts, the engine's two passes) base vs every staging placement swapped for tabA /
//           tabB, with m2Build before the first window user after each `dir` writer (the step's start counts) or once
//           (tabA1 / tabB1); walls interleaved, the clock probe after each. Its bitwise compare of every buffer is a
//           control only: two replays of the frozen step differ (the head claims slots by atomics).
// No lib/ change: variants are the schedule's own kernel text with substitutions, compiled as the engine compiles,
// bound with its bind() (m2win joins the engine's buffer map in this process only).
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { createHash } from "node:crypto";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const scratch = resolve(homedir(), "code/me/fluid-scratch/gpu-shape");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats" };
const log = (...a: unknown[]) => console.log(...a);

/** The text of `fn name(...){...}` in code (brace matched), and its start. */
function fnAt(code: string, name: string, from = 0): [number, number] {
  const m = new RegExp(`fn\\s+${name}\\s*\\(`).exec(code.slice(from)); if (!m) throw new Error(`no fn ${name}`);
  const i = from + m.index;
  let d = 0; for (let k = code.indexOf("{", i); k < code.length; k++) { if (code[k] === "{") d++; else if (code[k] === "}" && --d === 0) return [i, k + 1]; }
  throw new Error(`unterminated ${name}`);
}
const count = (s: string, x: string) => s.split(x).length - 1;
const sub1 = (s: string, from: string, to: string, what: string) => {
  const n = count(s, from); if (n !== 1) throw new Error(`${what}: "${from.slice(0, 70)}" hits ${n}`);
  return s.replace(from, () => to);
};
/** Replace inside the entry's body only. */
const inEntry = (code: string, entry: string, f: (body: string) => string) => { const [a, b] = fnAt(code, entry); return code.slice(0, a) + f(code.slice(a, b)) + code.slice(b); };
const WIN_CALL = /fgWindowStage(?:At)?\(/g, RING_CALL = /fgRingStage\(/g;
const stagingOf = (code: string, entry: string) => {
  const [a, b] = fnAt(code, entry), body = code.slice(a, b);
  const all = (re: RegExp, defs: string[]) => { let c = code; for (const d of defs) { try { const [x, y] = fnAt(c, d); c = c.slice(0, x) + c.slice(y); } catch { /* absent */ } } return (c.match(re) ?? []).length; };
  return { win: (body.match(WIN_CALL) ?? []).length, ring: (body.match(RING_CALL) ?? []).length,
    winAll: all(WIN_CALL, ["fgWindowStage", "fgWindowStageAt"]), ringAll: all(RING_CALL, ["fgRingStage"]) };
};
/** The statement `call(...);` from the n-th match of re in body, and the end of the first `workgroupBarrier();` after it. */
function callSpan(body: string, re: RegExp): { start: number; end: number; arg: string; barrierEnd: number } {
  const m = new RegExp(re.source).exec(body); if (!m) throw new Error(`no ${re.source}`);
  const start = m.index, end = body.indexOf(";", start) + 1;
  // First argument: up to the top-level comma.
  let d = 0, k = start + m[0].length; const a0 = k;
  for (; k < end; k++) { const ch = body[k]; if (ch === "(") d++; else if (ch === ")") d--; else if (ch === "," && d === 0) break; }
  const bar = body.indexOf("workgroupBarrier();", end); if (bar < 0) throw new Error("no barrier after the staging call");
  return { start, end, arg: body.slice(a0, k), barrierEnd: bar + "workgroupBarrier();".length };
}
/** A sink write into the kernel's first plain read-write word buffer (keeps the cut's work alive). */
function sinkWrite(code: string): string {
  for (const m of code.matchAll(/@group\(0\) @binding\(\d+\) var<storage,read_write> (\w+):array<([^>]+(?:>)?)>;/g)) {
    const [, n, t] = m; if (t === "atomic<u32>") return `atomicStore(&${n}[0],$);`; if (t === "u32") return `${n}[0]=$;`; if (t === "f32") return `${n}[0]=bitcast<f32>($);`;
  }
  throw new Error("no word buffer to sink into");
}
const cut = (sink: string, write: string) => `{let m2Acc=${sink};if(m2Acc==0x9e3779b9u){${write.replaceAll("$", "m2Acc")}}return;}`;
const WIN_SINK = "fgWinSlot[lid%27u]^fgWinTile[lid%216u]^atomicLoad(&fgWinCoarse[lid%7u])";

/** Truncation variants of a staging kernel: pre / win / ring. */
function cutVariants(code: string, entry: string): Record<string, string> {
  const st = stagingOf(code, entry), write = sinkWrite(code), out: Record<string, string> = {};
  if (st.win !== 1 || st.ring > 1) throw new Error(`${entry}: ${st.win} window and ${st.ring} ring calls in the entry`);
  out.pre = inEntry(code, entry, body => { const w = callSpan(body, WIN_CALL); return body.slice(0, w.start) + cut(`dot(vec3u(${w.arg}),vec3u(1u,7u,13u))`, write) + body.slice(w.start); });
  out.win = inEntry(code, entry, body => { const w = callSpan(body, WIN_CALL); return body.slice(0, w.barrierEnd) + cut(WIN_SINK, write) + body.slice(w.barrierEnd); });
  if (st.ring) out.ring = inEntry(code, entry, body => { const r = callSpan(body, RING_CALL); return body.slice(0, r.barrierEnd) + cut("bitcast<u32>(fgRingPhi(lid))", write) + body.slice(r.barrierEnd); });
  return out;
}

// ---- the window table ----
const REC = 256;
const STAGE_FROM_TABLE = /* wgsl */`fn fgWindowStageAt(low:vec3i,lid:u32,lanes:u32){
 fgWinOn=false;fgWinLow=low;
 let base=${REC}u*m2Slot;
 for(var j=lid;j<27u;j+=lanes){fgWinSlot[j]=m2win[base+j];}
 for(var j=lid;j<216u;j+=lanes){fgWinTile[j]=m2win[base+27u+j];}
 for(var j=lid;j<7u;j+=lanes){atomicStore(&fgWinCoarse[j],m2win[base+243u+j]);}
 fgWinOn=true;
}`;
const STAGE_DIRECT = `fn fgWindowStageAt(low:vec3i,lid:u32,lanes:u32){fgWinLow=low;m2Base=${REC}u*m2Slot;fgWinOn=true;}`;
const tableDecl = (code: string, kind: "read" | "read_write") => {
  const n = Math.max(-1, ...[...code.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map(m => Number(m[1]))) + 1;
  return { n, decl: `@group(0) @binding(${n}) var<storage,${kind}> m2win:array<u32>;\nvar<private> m2Slot:u32;\nvar<private> m2Base:u32;\n` };
};
function checkWindowShape(code: string) {
  for (const d of ["var<workgroup> fgWinSlot:array<u32,27>;", "var<workgroup> fgWinTile:array<u32,216>;", "var<workgroup> fgWinCoarse:array<atomic<u32>,7>;"])
    if (!code.includes(d)) throw new Error(`window shape changed: ${d}`);
}
function tableVariants(code: string, entry: string): { tabA: string; tabB: string; binding: number } {
  checkWindowShape(code);
  const [a, b] = fnAt(code, "fgWindowStageAt"); const stage = code.slice(a, b);
  if (!stage.includes("fgWinSlot[j]=fgBrickSlotOfDir(") || !stage.includes("atomicOr(&fgWinCoarse[j>>5u],1u<<(j&31u))")) throw new Error("fgWindowStageAt changed shape");
  const { n, decl } = tableDecl(code, "read");
  const withSlot = (c: string) => inEntry(c, entry, body => body.replace(/(?<![\w.])fgWindowStage\(/g, "m2Slot=slot;fgWindowStage("));
  let A = code.slice(0, a) + STAGE_FROM_TABLE + code.slice(b);
  A = withSlot(decl + A);
  let B = code.slice(0, a) + STAGE_DIRECT + code.slice(b);
  B = B.replaceAll("atomicLoad(&fgWinCoarse[i>>5u])", "m2win[m2Base+243u+(i>>5u)]").replaceAll("fgWinSlot[", "m2win[m2Base+").replaceAll("fgWinTile[", "m2win[m2Base+27u+");
  for (const bad of ["fgWinCoarse[", "fgWinSlot[", "fgWinTile["]) if (count(B, bad) !== 0) throw new Error(`tabB: ${bad} left`);
  B = withSlot(decl + B);
  return { tabA: A, tabB: B, binding: n };
}
/** m2Build: the window of every F_ALL brick into its slot's record, appended to a window kernel's module. */
function buildCode(code: string): { code: string; binding: number } {
  checkWindowShape(code);
  const { n, decl } = tableDecl(code, "read_write");
  return { binding: n, code: decl + code + /* wgsl */`
var<workgroup> m2H:u32;
var<workgroup> m2C:array<atomic<u32>,7>;
@compute @workgroup_size(256) fn m2Build(@builtin(workgroup_id) wid:vec3u,@builtin(local_invocation_index) lid:u32){
 if(lid==0u){m2H=nbListCount(NB_LIST_F_ALL);}
 let g=nbGroup(wid);if(g>=workgroupUniformLoad(&m2H)){return;}
 let slot=nbListSlot(NB_LIST_F_ALL,g);let low=(nbFOrigin(slot)>>vec3u(3u))-vec3i(1);let base=${REC}u*slot;
 if(lid<27u){m2win[base+lid]=fgBrickSlotOfDir(low+vec3i(i32(lid%3u),i32((lid/3u)%3u),i32(lid/9u)));}
 if(lid<216u){
  let j=lid;let t=2*low+vec3i(i32(j%6u),i32((j/6u)%6u),i32(j/36u));
  let cs=nbCSlotOfPage(t>>vec3u(3u));let h=fgTileHDir(t);
  m2win[base+27u+j]=select(cs,0x7fffffffu,cs==NB_NONE)|select(0u,0x80000000u,h);
  if(foTileInLattice(t)&&cs!=NB_NONE&&!h){atomicOr(&m2C[j>>5u],1u<<(j&31u));}
 }
 workgroupBarrier();
 if(lid<7u){m2win[base+243u+lid]=atomicLoad(&m2C[lid]);}
}` };
}

const COMPARE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage,read> ca:array<u32>;
@group(0) @binding(1) var<storage,read> cb:array<u32>;
@group(0) @binding(2) var<storage,read_write> co:array<atomic<u32>>;
@group(0) @binding(3) var<uniform> cn:vec4u;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g:vec3u){
 for(var i=g.x;i<cn.x;i+=cn.y){if(ca[i]!=cb[i]){atomicAdd(&co[0],1u);atomicMin(&co[1],i+cn.z);}}
}`;

if (process.argv[2] === "--check") {
  // Offline: node wintable.mts --check <entry> <kernel.wgsl>... : every cut and table variant applies.
  const [entry, ...files] = process.argv.slice(3);
  for (const f of files) { const { readFileSync } = await import("node:fs"); const c = readFileSync(f, "utf8"); const v = cutVariants(c, entry!); const t = tableVariants(c, entry!); buildCode(c);
    writeFileSync(`${f}.tabB.wgsl`, t.tabB); log(`${f}: cuts ${Object.keys(v).join(",")}, table binding ${t.binding}`); }
  process.exit(0);
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
  const { RING_SLOTS, RING_WORDS } = await import(`${root}/lib/methods/narrow-band/brick-abi.ts`);
  const { bindGroupLayoutEntries } = await import(`${root}/lib/methods/narrow-band/brick-kernel.ts`);
  const { brickModuleCode, brickLowerOn } = await import(`${root}/lib/methods/narrow-band/brick-lower.ts`);
  const { createClockProbe } = await import(`${here}/clock-probe.ts`);

  const [sceneKey = "dam", outArg] = process.argv.slice(2), sceneId = SCENES[sceneKey] ?? sceneKey;
  const out = resolve(outArg ?? resolve(scratch, "m2", sceneKey)); mkdirSync(resolve(out, "wgsl"), { recursive: true });
  const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 5);
  const modes = new Set((process.env.MODE ?? "dump,ceiling,proto,step").split(","));
  const ONLY = process.env.ONLY?.split(",");
  const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
  const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
  const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
  const quality = profile?.methodId === methodId ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
  const head = (() => { try { return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(); } catch { return "?"; } })();
  log(`wintable scene ${sceneId} (${sceneKey}) HEAD ${head} warm-up ${N}, R ${R}, modes ${[...modes].join(",")}, out ${out}`);
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
  const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  log(`maxStorageBuffersPerShaderStage ${raw.limits.maxStorageBuffersPerShaderStage}, maxStorageBufferBindingSize ${raw.limits.maxStorageBufferBindingSize}`);
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

  const L = P.layout, slot = P.steps % RING_SLOTS, rec = P.recorded[slot], args = P.buffers.get("args");
  const sch = [...P.schedule.part1, ...P.schedule.part2], all = [...rec.part1, ...rec.part2], n1 = rec.part1.length;
  if (sch.length !== all.length) throw new Error("recorded placements differ from the schedule");
  const result: any = { scene: sceneId, key: sceneKey, head, warmup: N, R, steps: P.steps, fCap: L.cap.f, date: new Date().toISOString(),
    status: (() => { try { return execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }); } catch { return ""; } })() };
  const save = () => writeFileSync(resolve(out, "wintable.json"), JSON.stringify(result, null, 1));

  // ---- dump: the schedule ----
  const stagers: number[] = [];
  result.schedule = sch.map((p: any, i: number) => {
    const k = p.kernel; let st = { win: 0, ring: 0, winAll: 0, ringAll: 0 };
    try { st = stagingOf(k.code, k.entry); } catch { /* entry not found */ }
    if ((st.win || st.ring || st.winAll || st.ringAll) && (!ONLY || ONLY.includes(k.label))) stagers.push(i);
    const dir = k.bindings.find((b: any) => b.name === "dir")?.kind ?? "";
    return { i, part: i < n1 ? 1 : 2, label: k.label, entry: k.entry, dispatch: p.dispatch, dir, rw: k.bindings.filter((b: any) => b.kind === "rw").map((b: any) => b.name),
      sha: createHash("sha1").update(k.code).digest("hex").slice(0, 10), ...st };
  });
  const kdir = resolve(out, "kernels"); mkdirSync(kdir, { recursive: true });
  for (const p of sch) writeFileSync(resolve(kdir, `${p.kernel.label}.wgsl`), p.kernel.code);
  save();
  log(`schedule ${sch.length} placements (part1 ${n1}); staging placements:`);
  for (const i of stagers) { const e = result.schedule[i]; log(`  ${i} p${e.part} ${e.label} win ${e.win}/${e.winAll} ring ${e.ring}/${e.ringAll} dispatch ${JSON.stringify(e.dispatch)}`); }
  log(`dir read-write placements: ${result.schedule.filter((e: any) => e.dir === "rw").map((e: any) => `${e.i}:${e.label}`).join(" ")}`);
  if (!modes.has("ceiling") && !modes.has("proto") && !modes.has("step")) { s.destroy?.(); process.exit(0); }

  // ---- the frozen state ----
  const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const liveNames = [...P.buffers.keys()] as string[];
  const live = [...[...P.buffers.values()] as GPUBuffer[], ...[...new Set([...P.external.values()])].filter((b: GPUBuffer) => (b.usage & COPY) === COPY)] as GPUBuffer[];
  const shadow = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
  const copyAll = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  copyAll(live, shadow);
  const restore = () => copyAll(shadow, live);
  const ring = () => raw.queue.writeBuffer(P.buffers.get("set"), 4 * (L.set.ring + RING_WORDS * slot), P.ringWords(P.steps));
  const dispatchInto = (pass: GPUComputePassEncoder, r: any) => {
    pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
    if ("direct" in r.dispatch) pass.dispatchWorkgroups(...(r.dispatch.direct as [number, number, number])); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
  };
  const QMAX = 64, qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
  const qResolve = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readTimes = async (enc: GPUCommandEncoder, n: number) => {
    enc.resolveQuerySet(qs, 0, n, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 8 * n); raw.queue.submit([enc.finish()]);
    await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange(0, 8 * n).slice(0)); qRead.unmap(); return t;
  };
  const probe = createClockProbe(raw, { capacity: 4 });
  const clock = async () => { const enc = raw.createCommandEncoder(); probe.encode(enc); raw.queue.submit([enc.finish()]); return (await probe.read())[0]!; };
  const lower = brickLowerOn(P.options.lowerDynamicWrites);
  /** Compile a kernel variant (extra: the m2win binding) and bind it at placement i. */
  const variantRec = async (i: number, label: string, code: string, extra?: "ro" | "rw", entry?: string, bindFrom = i) => {
    const k0 = sch[bindFrom].kernel, bindings = extra ? [...k0.bindings, { name: "m2win", kind: extra, type: "array<u32>" }] : k0.bindings;
    const kv = { ...k0, label, code, bindings, entry: entry ?? k0.entry };
    const text = brickModuleCode(label, code, lower);
    writeFileSync(resolve(out, "wgsl", `${label}.wgsl`), text); writeFileSync(resolve(out, "wgsl", `${label}.entry`), kv.entry);
    const module = raw.createShaderModule({ label, code: text });
    const errs = (await module.getCompilationInfo()).messages.filter((m: any) => m.type === "error");
    if (errs.length) throw new Error(`${label} does not compile:\n${errs.slice(0, 5).map((m: any) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
    const entries = bindGroupLayoutEntries(k0.bindings);
    if (extra) entries.push({ binding: entries.length, visibility: 4, buffer: { type: extra === "ro" ? "read-only-storage" : "storage" } });
    const layout = raw.createPipelineLayout({ bindGroupLayouts: [raw.createBindGroupLayout({ entries })] });
    P.pipelines.set(kv, await raw.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: kv.entry } }));
    return P.bind({ ...sch[i], kernel: kv }, slot);
  };
  /** In situ: restore, ring, placements [0, i) untimed (a pass a part), `pre` untimed, then `timed` alone in a timestamped pass. */
  const sample = async (i: number, timed: any, pre: any[] = []) => {
    restore(); ring();
    const enc = raw.createCommandEncoder();
    const p1 = all.slice(0, Math.min(i, n1)), p2 = i > n1 ? all.slice(n1, i) : [];
    for (const list of [p1, p2, pre]) if (list.length) { const pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end(); }
    const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }); dispatchInto(pass, timed); pass.end();
    const t = await readTimes(enc, 2); return Number(t[1]! - t[0]!) / 1e3;
  };
  const stats = (v: number[]) => { const a = [...v].sort((x, y) => x - y); return { min: a[0]!, med: a[a.length >> 1]! }; };
  const rounds = async (cases: { key: string; run: () => Promise<number> }[]) => {
    const t = new Map<string, number[]>(cases.map(c => [c.key, []])); const probes: number[] = [];
    for (let r = 0; r < R; r++) { for (const c of r % 2 ? [...cases].reverse() : cases) t.get(c.key)!.push(await c.run()); probes.push(await clock()); fail("in the timing"); }
    return { t, probes };
  };

  // ---- ceiling: truncation ----
  if (modes.has("ceiling")) {
    result.ceiling = {};
    const built = new Map<string, Record<string, any>>();
    const cases: { key: string; run: () => Promise<number> }[] = [];
    for (const i of stagers) {
      const k = sch[i].kernel; let recs = built.get(k.label);
      if (!recs) { recs = {}; built.set(k.label, recs); }
      const vs = cutVariants(k.code, k.entry), perI: Record<string, any> = { full: all[i] };
      for (const [name, code] of Object.entries(vs)) perI[name] = await variantRec(i, `${k.label}.m2.${name}.${i}`, code);
      for (const [name, r] of Object.entries(perI)) cases.push({ key: `${i}\t${name}`, run: () => sample(i, r) });
      result.ceiling[i] = { label: k.label, variants: Object.keys(perI) };
    }
    fail("building the cuts");
    log(`ceiling: ${cases.length} cases`);
    const { t, probes } = await rounds(cases);
    let winSum = 0, ringSum = 0, fullSum = 0;
    for (const i of stagers) {
      const c = result.ceiling[i]; c.us = {};
      for (const v of c.variants) c.us[v] = { ...stats(t.get(`${i}\t${v}`)!), samples: t.get(`${i}\t${v}`) };
      const win = c.us.win.min - c.us.pre.min, rng = c.us.ring ? c.us.ring.min - c.us.win.min : 0;
      c.windowUs = win; c.ringUs = rng; winSum += win; ringSum += rng; fullSum += c.us.full.min;
      log(`  ${String(i).padStart(3)} ${c.label.padEnd(14)} full ${c.us.full.min.toFixed(1).padStart(7)}  pre ${c.us.pre.min.toFixed(1).padStart(7)}  win ${c.us.win.min.toFixed(1).padStart(7)}${c.us.ring ? `  ring ${c.us.ring.min.toFixed(1).padStart(7)}` : "              "}  window ${win.toFixed(1).padStart(6)} ring ${rng.toFixed(1).padStart(6)} us`);
    }
    result.ceilingSum = { windowUs: winSum, ringUs: ringSum, fullUs: fullSum, probes };
    log(`ceiling sum: window ${winSum.toFixed(1)} us, ring ${ringSum.toFixed(1)} us over ${stagers.length} placements (their full ${fullSum.toFixed(1)} us); clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }

  // ---- prototype: the table ----
  if (!modes.has("proto") && !modes.has("step")) { s.destroy?.(); process.exit(gpuErrors.length ? 3 : 0); }
  const table = raw.createBuffer({ label: "m2win", size: 4 * REC * L.cap.f, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  P.buffers.set("m2win", table);
  log(`m2win ${(table.size / 2 ** 20).toFixed(1)} MB (${L.cap.f} F slots x ${REC} words)`);
  const bc = buildCode(sch[stagers[0]!].kernel.code);
  const buildAt = new Map<number, any>(); // the build bound with placement i's dispatch
  const buildRec = async (i: number) => { let r = buildAt.get(i); if (!r) { r = await variantRec(i, `m2Build.${i}`, bc.code, "rw", "m2Build", stagers[0]!); buildAt.set(i, r); } return r; };
  const tab = new Map<number, { tabA: any; tabB: any }>();
  for (const i of stagers) {
    const k = sch[i].kernel, v = tableVariants(k.code, k.entry);
    tab.set(i, { tabA: await variantRec(i, `${k.label}.m2.tabA.${i}`, v.tabA, "ro"), tabB: await variantRec(i, `${k.label}.m2.tabB.${i}`, v.tabB, "ro") });
    await buildRec(i);
  }
  fail("building the table variants");

  // Exactness helpers: GPU word compare of two buffers (chunked to the binding limit).
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
  const refs = new Map<GPUBuffer, GPUBuffer>();
  const refOf = (b: GPUBuffer) => { let r = refs.get(b); if (!r) { r = raw.createBuffer({ size: b.size, usage: COPY | GPUBufferUsage.STORAGE }); refs.set(b, r); } return r; };
  const bufOf = (name: string): GPUBuffer => P.buffers.get(name) ?? P.external.get(name);

  if (modes.has("proto")) {
    result.proto = {};
    // Exactness per placement, from an in-situ snapshot (the step's head claims slots by atomics, so two replays of
    // the prefix differ; base and variants must start from one state): restore, ring, placements [0, i) with the
    // single build before the first window user, snapshot every buffer; then from the snapshot base (the reference),
    // base again (control), tabA and tabB on the snapshot's table, each kernel's read-write buffers compared; and the
    // snapshot's table against a fresh build at i (the single build still valid there).
    const first = stagers.find(j => result.schedule[j].win)!;
    const snap = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
    const tableRef = raw.createBuffer({ size: table.size, usage: COPY | GPUBufferUsage.STORAGE });
    const fromSnap = async (list: any[]) => {
      copyAll(snap, live); ring();
      const enc = raw.createCommandEncoder(), pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end();
      raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone();
    };
    const b0 = await buildRec(first);
    for (const i of stagers) {
      restore(); ring();
      { const enc = raw.createCommandEncoder(); const pre = all.slice(0, i); if (i > first) pre.splice(first, 0, b0);
        const m = Math.min(n1 + (i > first && first < n1 ? 1 : 0), pre.length);
        for (const list of [pre.slice(0, m), pre.slice(m)]) if (list.length) { const pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end(); }
        raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone(); }
      copyAll(live, snap); await raw.queue.onSubmittedWorkDone();
      const k = sch[i].kernel, names = k.bindings.filter((b: any) => b.kind === "rw").map((b: any) => b.name as string);
      const bufs: GPUBuffer[] = names.map(bufOf);
      await fromSnap([all[i]]); { const enc = raw.createCommandEncoder(); for (const b of new Set(bufs)) enc.copyBufferToBuffer(b, 0, refOf(b), 0, b.size); raw.queue.submit([enc.finish()]); }
      const ex: Record<string, Record<string, [number, number]>> = {};
      for (const [v, list] of [["base", [all[i]]], ["tabA", [tab.get(i)!.tabA]], ["tabB", [tab.get(i)!.tabB]]] as [string, any[]][]) {
        if (v !== "base" && i <= first) list.unshift(await buildRec(i));
        await fromSnap(list); ex[v] = {};
        for (const [j, b] of bufs.entries()) ex[v]![names[j]!] = await compare(b, refOf(b));
      }
      // The snapshot's table (built at the first window user) against a fresh build here.
      let tableDiff: [number, number] = [0, 0];
      if (i > first) { copyAll(snap, live); { const enc = raw.createCommandEncoder(); enc.copyBufferToBuffer(table, 0, tableRef, 0, table.size); raw.queue.submit([enc.finish()]); }
        await fromSnap([await buildRec(i)]); tableDiff = await compare(table, tableRef); }
      result.proto[i] = { label: k.label, exact: ex, tableVsFresh: tableDiff };
      const diffs = Object.entries(ex).map(([v, m]) => `${v} ${Object.values(m).reduce((a, x) => a + x[0], 0)}`).join(", ");
      log(`  exact ${i} ${k.label}: words differing in ${names.join(",")}: ${diffs}; table vs fresh build ${tableDiff[0]}`);
    }
    for (const b of snap) b.destroy();
    fail("in the exactness check");
    if (process.env.EXACT_ONLY === "1") { save(); s.destroy?.(); process.exit(gpuErrors.length ? 3 : 0); }
    const cases: { key: string; run: () => Promise<number> }[] = [];
    for (const i of stagers) {
      const b = await buildRec(i), tv = tab.get(i)!;
      cases.push({ key: `${i}\tbase`, run: () => sample(i, all[i]) });
      cases.push({ key: `${i}\ttabA`, run: () => sample(i, tv.tabA, [b]) });
      cases.push({ key: `${i}\ttabB`, run: () => sample(i, tv.tabB, [b]) });
      cases.push({ key: `${i}\tbuild`, run: () => sample(i, b) });
    }
    const { t, probes } = await rounds(cases);
    let sb = 0, sa = 0, sB = 0, sbuild = 0;
    for (const i of stagers) {
      const c = result.proto[i]; c.us = {};
      for (const v of ["base", "tabA", "tabB", "build"]) c.us[v] = { ...stats(t.get(`${i}\t${v}`)!), samples: t.get(`${i}\t${v}`) };
      sb += c.us.base.min; sa += c.us.tabA.min; sB += c.us.tabB.min; sbuild += c.us.build.min;
      log(`  ${String(i).padStart(3)} ${c.label.padEnd(14)} base ${c.us.base.min.toFixed(1).padStart(7)}  tabA ${c.us.tabA.min.toFixed(1).padStart(7)} (${(c.us.tabA.min - c.us.base.min).toFixed(1)})  tabB ${c.us.tabB.min.toFixed(1).padStart(7)} (${(c.us.tabB.min - c.us.base.min).toFixed(1)})  build ${c.us.build.min.toFixed(1)} us`);
    }
    result.protoSum = { baseUs: sb, tabAUs: sa, tabBUs: sB, buildUsMean: sbuild / stagers.length, probes };
    log(`proto sum: base ${sb.toFixed(1)}  tabA ${sa.toFixed(1)} (${(sa - sb).toFixed(1)})  tabB ${sB.toFixed(1)} (${(sB - sb).toFixed(1)}) us; build mean ${(sbuild / stagers.length).toFixed(1)} us; clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }

  // ---- whole step ----
  if (modes.has("step")) {
    // Builds: before the first staging placement after each dir writer (the step's start counts as one).
    const builds: number[] = []; let dirty = true;
    for (let i = 0; i < sch.length; i++) {
      if (result.schedule[i].dir === "rw") dirty = true;
      if (stagers.includes(i) && result.schedule[i].win && dirty) { builds.push(i); dirty = false; }
    }
    // Configurations: "<variant>" with the rule's builds, "<variant>1" with one build before the first window user.
    const one = [stagers.find(i => result.schedule[i].win)!];
    log(`step: rule builds before ${builds.join(",")} (${builds.length}); single build before ${one[0]}`);
    const configs: Record<string, { v: "base" | "tabA" | "tabB"; builds: number[] }> = {
      base: { v: "base", builds: [] }, tabA: { v: "tabA", builds }, tabB: { v: "tabB", builds }, tabA1: { v: "tabA", builds: one }, tabB1: { v: "tabB", builds: one } };
    const names = Object.keys(configs);
    const lists: Record<string, { l: any[]; m: number }> = {};
    for (const [name, c] of Object.entries(configs)) {
      const l: any[] = [];
      for (let i = 0; i < all.length; i++) { if (c.builds.includes(i)) l.push(await buildRec(i)); l.push(c.v === "base" || !tab.has(i) ? all[i] : tab.get(i)![c.v]); }
      lists[name] = { l, m: n1 + c.builds.filter(b => b < n1).length };
    }
    const wall = async (name: string) => {
      restore(); ring();
      const enc = raw.createCommandEncoder(), { l, m } = lists[name]!;
      const a = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } }); for (const x of l.slice(0, m)) dispatchInto(a, x); a.end();
      const b = enc.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: 1 } }); for (const x of l.slice(m)) dispatchInto(b, x); b.end();
      const t = await readTimes(enc, 2); return Number(t[1]! - t[0]!) / 1e3;
    };
    // Exactness: every engine buffer (and external) after the step, against base's.
    const cmpBufs = live.filter(b => b !== table);
    await wall("base"); await raw.queue.onSubmittedWorkDone();
    { const enc = raw.createCommandEncoder(); for (const b of cmpBufs) enc.copyBufferToBuffer(b, 0, refOf(b), 0, b.size); raw.queue.submit([enc.finish()]); }
    result.stepExact = {};
    for (const v of names) {
      await wall(v); await raw.queue.onSubmittedWorkDone();
      const d: Record<string, [number, number]> = {}; let tot = 0;
      for (const [j, b] of cmpBufs.entries()) { const c = await compare(b, refOf(b)); if (c[0]) d[liveNames[j] ?? `ext${j}`] = c; tot += c[0]; }
      result.stepExact[v] = { total: tot, diffs: d };
      log(`  step exact ${v}: ${tot} words differ${tot ? ` ${JSON.stringify(d).slice(0, 300)}` : ""}`);
    }
    fail("in the step exactness");
    const { t, probes } = await rounds(names.map(v => ({ key: v, run: () => wall(v) })));
    result.step = { builds, one, probes };
    for (const v of names) result.step[v] = { ...stats(t.get(v)!), samples: t.get(v) };
    const b0 = result.step.base.min;
    log(`step wall (min, delta vs base, median): ${names.map(v => `${v} ${(result.step[v].min / 1e3).toFixed(3)} (${((result.step[v].min - b0) / 1e3).toFixed(3)}) med ${(result.step[v].med / 1e3).toFixed(3)}`).join(" | ")} ms; clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }
  for (const b of shadow) b.destroy();
  s.destroy?.();
  process.exit(gpuErrors.length ? 3 : 0);
}
