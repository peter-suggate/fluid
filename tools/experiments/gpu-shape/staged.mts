// GPU-shape M1b (lane B): phase attribution of the staged kernels (docs/plans/gpu-shape/m1b-staged.md).
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/staged.mts <scene> [outdir]
//   env STEPS=100 (warm-up, as the census)  R=7 (interleaved samples)  CTR_SECS=4 (0: no counters)  CAPS=0 (skip regcap)
//       KERNELS=uExtend,uTransfer,uAdvect,pDeposit
//
// Warm the brick engine up STEPS steps (fixed dt 1/60, census.mts), freeze it (a GPU shadow of every buffer), and for
// each kernel build text variants that end it after phase k: the entry's text up to a `workgroupBarrier()` section,
// then a sink (an XOR of the phase's workgroup words and live registers, written to one word only when it equals a
// constant no state takes, so the compiler keeps the work) and `return`. No lib/ change: the variants are the
// schedule's own kernel text with substitutions inside the entry (each must hit once), compiled as the engine
// compiles (brickModuleCode, its bindings), bound to the engine's buffers (its bind()).
// Timing: per sample, the frozen state is restored, the ring written, the step's recorded placements before the
// kernel's run untimed (one pass a part, as the step), then the variant alone in a timestamped pass; R samples a
// variant, interleaved; min and median. Phase cost = t(k) - t(k-1).
// Counters: the child prints READY and cycles every variant back to back (passes labelled `m1b.<kernel>.<variant>`,
// each with timestamp writes so Dawn keeps it its own Metal encoder) while this process attaches Metal GPU Counters.
// Caps: each variant's module (as compiled) goes through msl.mts (Dawn's MSL) and psoinfo (Metal's
// maxTotalThreadsPerThreadgroup with the size attribute stripped: the register-bound cap), as census.mts --caps.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { execFileSync, spawn } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync, openSync, closeSync, readdirSync } from "node:fs";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const scratch = resolve(homedir(), "code/me/fluid-scratch/gpu-shape");
const SCENES: Record<string, string> = { seep: "reservoir-seep", dam: "high-resolution-dam-break", letters: "nbflip-figure-8-letters", paddleboats: "paddleboats" };
const log = (...a: unknown[]) => console.log(...a);

// ---- the variants: [from, to] substitutions inside the entry; a cut appends `sink` (u32 expression) and returns ----
type Sub = [string, string];
type Variant = { name: string; subs: Sub[]; probe?: true };
const F = (x: string) => `bitcast<u32>(${x})`;
const cutAt = (anchor: string, sink: string, write: string, guard?: string): Sub => {
  const body = `{let m1bAcc=${sink};if(m1bAcc==0x9e3779b9u){${write.replaceAll("$", "m1bAcc")}}return;}`;
  return [anchor, `${anchor}${guard ? `if(${guard})` : "if(true)"}${body}`];
};
const VEL = "vel[0]=bitcast<f32>($);", FDEP = "atomicStore(&fdep[0],$);";
const WIN = "fgWinSlot[lid%27u]^fgWinTile[lid%216u]^atomicLoad(&fgWinCoarse[lid%7u])";
const RING = `${F("fgRing[lid]")}^${F("fgRing[(lid+744u)%1000u]")}`;
const KERNELS: Record<string, { write: string; variants: Variant[] }> = {
  uExtend: { write: VEL, variants: (() => {
    const seed = "    wV[k]=s.v;wD[k]=s.d;\n   }\n  }\n  workgroupBarrier();";
    const sweep = "held[r]=xSweepAt(xInset(k),h);}}}\n  workgroupBarrier();";
    const back = "wV[i]=held[r].v;wD[i]=held[r].d;}}}\n  workgroupBarrier();";
    const fin = "vel[vAt(NB_VEL_F_US0,slot,a,l)]=out.x;vel[iu]=out.y;\n  }\n  workgroupBarrier();";
    return [
      { name: "1-ring", subs: [cutAt("if(workgroupUniformLoad(&sRing)!=0u){return;}", "slot^u32(o.x)", VEL)] },
      { name: "2-window", subs: [cutAt("fgWindowStage(o>>vec3u(3u),lid,512u);workgroupBarrier();", WIN, VEL)] },
      { name: "3-stage", subs: [cutAt("  wS4[lid]=vec4f(st,0.0);\n }\n workgroupBarrier();", `wLiqW[lid%432u]^wTile4[lid%64u]^${F("wS4[lid%375u].z")}`, VEL)] },
      { name: "4-reach", subs: [cutAt(" let reach=atomicLoad(&wReach);", "reach^atomicLoad(&wLiq[lid%69u])^atomicLoad(&wSwept[lid%27u])", VEL)] },
      { name: "5-a0seed", subs: [cutAt(seed, `${F("wV[lid]")}^${F("wD[lid]")}`, VEL, "a==0u")] },
      { name: "6-a0sweep", subs: [cutAt(sweep, `${F("held[0].v")}^${F("held[1].d")}`, VEL, "a==0u")] },
      { name: "7-a0back", subs: [cutAt(back, F("wV[lid]"), VEL, "a==0u")] },
      { name: "8-a0finish", subs: [cutAt(fin, "0u", VEL, "a==0u")] },
      { name: "9-a1finish", subs: [cutAt(fin, "0u", VEL, "a==1u")] },
      { name: "full", subs: [] },
      // Probe (timing only, racy: the sweep's barriers removed; wrong values, the barriers' cost bound).
      { name: "nobar", probe: true, subs: [[seed, seed.replace("workgroupBarrier();", "")], [sweep, sweep.replace("workgroupBarrier();", "")], [back, back.replace("workgroupBarrier();", "")], [fin, fin.replace("workgroupBarrier();", "")]] },
    ];
  })() },
  uTransfer: { write: VEL, variants: [
    // The C pages (the groups past F_ALL: vcTransferPage / vcAdvectPageOf) and the F groups' prologue only.
    { name: "0-cpages", subs: [cutAt("let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);", "slot^u32(o.x)", VEL)] },
    { name: "1-window", subs: [cutAt("fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();", WIN, VEL)] },
    { name: "2-ring", subs: [cutAt("fgRingStage(o+vec3i(-1),lid,256u);workgroupBarrier();", RING, VEL)] },
    // Inside `if(live){` (live is the constant true there): after the ring bricks' return, before the stage.
    { name: "3-ringmark", subs: [cutAt("  return;\n }\n if(live){", "atomicLoad(&sRingOwned)^slot", VEL)] },
    { name: "4-stage", subs: [cutAt(" workgroupBarrier();\n if(!live){return;}", `${F("wDepth[lid]")}^${F("wHeat[lid%125u]")}^${F("wHot[lid%125u]")}^${F("wCentre[lid]")}^wSeam[lid%125u]^${F("wOcc4[lid%125u]")}`, VEL)] },
    { name: "5-liqstore", subs: [cutAt(" vLiquidStore(slot,o,lid);\n let dt=vDt();", "0u", VEL)] },
    { name: "6-half", subs: [["for(var r=0u;r<2u;r++){\n  let l=lid+256u*r;", "for(var r=0u;r<2u;r++){if(r==1u){return;}\n  let l=lid+256u*r;"]] },
    { name: "full", subs: [] },
  ] },
  uAdvect: { write: VEL, variants: [
    // The C pages (the groups past F_ALL: vcTransferPage / vcAdvectPageOf) and the F groups' prologue only.
    { name: "0-cpages", subs: [cutAt("let slot=nbListSlot(NB_LIST_F_ALL,g);let o=nbFOrigin(slot);", "slot^u32(o.x)", VEL)] },
    { name: "1-window", subs: [cutAt("fgWindowStage(o>>vec3u(3u),lid,256u);workgroupBarrier();", WIN, VEL)] },
    { name: "2-ringmark", subs: [cutAt("if(workgroupUniformLoad(&sRing)!=0u&&g!=0u){return;}", "atomicLoad(&sRingOwned)^slot", VEL)] },
    { name: "3-ring", subs: [cutAt("fgRingStage(o+vec3i(-1),lid,256u);workgroupBarrier();", RING, VEL)] },
    { name: "4-live4", subs: [cutAt("vLive4(T));}\n workgroupBarrier();", "wLive4[lid%27u]", VEL)] },
    { name: "5-classify", subs: [cutAt("vel[vAt(NB_VEL_F_UP0,slot,a,l)]=0.0;\n  }\n }\n workgroupBarrier();", "atomicLoad(&wTraced)^wTrace[lid]", VEL)] },
    { name: "full", subs: [] },
  ] },
  pDeposit: { write: FDEP, variants: [
    { name: "1-window", subs: [cutAt("fgWindowStage(pdOrigin>>vec3u(3u),lid,256u);\n workgroupBarrier();", `${WIN}^atomicLoad(&wLiquid[lid%32u])`, FDEP)] },
    { name: "2-ring", subs: [cutAt("fgRingStage(pdOrigin-vec3i(1),lid,256u);\n workgroupBarrier();", "atomicLoad(&wMom[lid])^atomicLoad(&wMom[lid+1024u])", FDEP)] },
    { name: "3-liquid", subs: [cutAt("atomicOr(&wLiquid[j>>5u],1u<<(j&31u));}\n }\n workgroupBarrier();", "atomicLoad(&wLiquid[lid%32u])", FDEP)] },
    { name: "4-zero", subs: [cutAt("atomicStore(&wWeight[j],0u);}\n workgroupBarrier();", "atomicLoad(&wMom[lid])^atomicLoad(&wWeight[lid+2048u])", FDEP)] },
    { name: "5-scatter", subs: [cutAt("nbfP2GScatter(q,s.velocity.xyz,nbParticleMass(s));}\n }\n workgroupBarrier();", "atomicLoad(&wMom[lid])^atomicLoad(&wWeight[lid+2048u])", FDEP)] },
    { name: "full", subs: [] },
  ] },
};

function variantCode(code: string, entry: string, v: Variant): string {
  const at = code.search(new RegExp(`fn\\s+${entry}\\s*\\(`));
  if (at < 0) throw new Error(`staged: no entry ${entry}`);
  let body = code.slice(at);
  for (const [from, to] of v.subs) {
    const n = body.split(from).length - 1;
    if (n !== 1) throw new Error(`staged: ${entry}/${v.name}: substitution hits ${n} times: ${JSON.stringify(from.slice(0, 80))}`);
    body = body.replace(from, () => to);
  }
  return code.slice(0, at) + body;
}

if (process.argv[2] === "--check") {
  // Offline: every variant's substitutions against kernel texts (node staged.mts --check <kernel> <file.wgsl>...).
  const [K, ...files] = process.argv.slice(3);
  for (const f of files) for (const v of KERNELS[K!]!.variants) { variantCode(readFileSync(f, "utf8"), K!, v); }
  log(`${K}: ${KERNELS[K!]!.variants.length} variants hit in ${files.length} file(s)`);
  process.exit(0);
}
if (process.env.M1B_CHILD !== "1") await parent();
else await child();

// ---------------------------------------------------------------------------------------------------------------
async function parent() {
  const [scene = "dam", outArg] = process.argv.slice(2);
  const out = resolve(outArg ?? resolve(scratch, "m1b", scene)); mkdirSync(out, { recursive: true });
  const secs = Number(process.env.CTR_SECS ?? 4), ctr = secs > 0;
  const trace = `${out}/run.trace`; if (existsSync(trace)) rmSync(trace, { recursive: true });
  const env = { ...process.env, M1B_CHILD: "1", SCENE: scene, OUT: out, LOOP_MS: String(ctr ? 1000 * secs + 3500 : 0),
    ...(ctr ? { FLUID_WEBGPU_DAWN_FEATURES: "use_user_defined_labels_in_backend" } : {}) };
  const kid = spawn(process.execPath, ["--import", "tsx", fileURLToPath(import.meta.url)], { cwd: root, env, stdio: ["ignore", "pipe", "inherit"] });
  let attached = false;
  const recorded = new Promise<void>((done, fail) => {
    kid.stdout.setEncoding("utf8");
    kid.stdout.on("data", (c: string) => {
      process.stdout.write(c);
      if (!ctr || attached || !c.includes("READY")) return;
      attached = true;
      setTimeout(() => {
        log(`attaching to ${kid.pid}`);
        const blank = resolve(execFileSync("xcode-select", ["-p"]).toString().trim(), "../Applications/Instruments.app/Contents/Packages/Base.instrdst/Contents/Templates/Blank.tracetemplate");
        const x = spawn("xcrun", ["xctrace", "record", "--template", blank, "--instrument", "Metal Application", "--instrument", "GPU", "--instrument", "Metal GPU Counters",
          "--output", trace, "--no-prompt", "--time-limit", `${secs}s`, "--attach", String(kid.pid)], { stdio: ["ignore", "pipe", "pipe"] });
        x.stdout.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
        x.stderr.on("data", (d: Buffer) => process.stdout.write(`[xctrace] ${d}`));
        x.once("exit", code => code === 0 ? done() : fail(new Error(`xctrace ${code}`)));
      }, 1000);
    });
  });
  const finished = new Promise<number>(r => kid.once("exit", c => r(c ?? 1)));
  if (ctr) await recorded;
  const rc = await finished; log("child rc", rc);
  if (rc !== 0) process.exit(rc);
  if (ctr) await counters(out, trace, kid.pid!);
  if (process.env.CAPS !== "0") await caps(out);
}

async function counters(out: string, trace: string, pid: number) {
  const { parseTraceTable } = await import(`${root}/tools/xctrace-trace-tables.ts`);
  const tables = [
    ["metal-gpu-intervals", "gpu-intervals", ["start", "duration", "channel-name", "event-label", "process", "encoder-id"]],
    ["metal-application-encoders-list", "encoders", ["encoder-id", "encoder-label", "process"]],
    ["gpu-counter-info", "counter-info", ["counter-id", "name", "type", "sample-interval"]],
    ["gpu-counter-value", "counter-value", ["timestamp", "counter-id", "value", "ring-buffer-index"]],
  ] as const;
  for (const [schema, file, columns] of tables) {
    const p = spawn("xcrun", ["xctrace", "export", "--input", trace, "--xpath", `/trace-toc/run[@number="1"]/data/table[@schema="${schema}"]`], { stdio: ["ignore", "pipe", "inherit"] });
    p.stdout.setEncoding("utf8");
    const sink = createWriteStream(`${out}/${file}.ndjson`); let n = 0;
    for await (const row of parseTraceTable(p.stdout as AsyncIterable<string>, { columns })) { sink.write(JSON.stringify(row) + "\n"); n++; }
    await new Promise(r => sink.end(r)); log(file, n);
  }
  execFileSync(process.execPath, ["--import", "tsx", `${root}/tools/xctrace-hardware-limiters.ts`, `--artifact=${out}`, `--pid=${pid}`], { cwd: root, stdio: "inherit" });
  rmSync(`${out}/counter-value.ndjson`, { force: true });
  const C = ["Compute Occupancy", "ALU Limiter", "ALU Utilization", "Buffer Read Limiter", "Buffer Write Limiter", "Threadgroup/Imageblock Load Limiter", "Threadgroup/Imageblock Store Limiter", "GPU Last Level Cache Limiter", "MMU Limiter", "MMU TLB Miss Rate", "GPU Read Bandwidth"];
  const short = ["occ", "aluL", "aluU", "bufRL", "bufWL", "tgLdL", "tgStL", "llcL", "mmuL", "tlbMiss", "rdGBs"];
  const d = JSON.parse(readFileSync(`${out}/hardware-limiters.json`, "utf8"));
  const agg = new Map<string, { us: number; n: number; c: Map<string, [number, number]> }>();
  for (const st of d.stages) {
    const m = /m1b\.(\w+)\.([\w-]+)/.exec(String(st.label)); if (!m) continue;
    const lab = `${m[1]}\t${m[2]}`;
    const a = agg.get(lab) ?? { us: 0, n: 0, c: new Map() }; agg.set(lab, a); a.us += st.exclusiveUs; a.n++;
    for (const k of C) if (st.counters[k]) { const v = a.c.get(k) ?? [0, 0]; v[0] += st.counters[k].meanAcrossRings * st.exclusiveUs; v[1] += st.exclusiveUs; a.c.set(k, v); }
  }
  const lines = [["kernel", "variant", "us", "stages", ...short].join("\t")];
  for (const [lab, a] of [...agg].sort((x, y) => x[0].localeCompare(y[0])))
    lines.push([lab, a.us.toFixed(0), a.n, ...C.map(k => { const v = a.c.get(k); return v && v[1] ? (v[0] / v[1]).toFixed(1) : "nan"; })].join("\t"));
  writeFileSync(`${out}/counters.tsv`, lines.join("\n") + "\n");
  log(lines.join("\n"));
}

// Workgroup bytes of the var<workgroup> declarations the MSL still uses (census.mts's wgslWorkgroupBytes).
function wgslWorkgroupBytes(code: string, used: (name: string) => boolean): Record<string, number> {
  const c = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const consts = new Map<string, string>();
  for (const m of c.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([^;]+);/g)) consts.set(m[1]!, m[2]!);
  const evalN = (e: string, depth = 0): number => {
    const t = e.replace(/\b([A-Za-z_]\w*)\b/g, id => { const v = consts.get(id); if (v === undefined || depth > 8) throw new Error(`const ${id}`); return `(${evalN(v, depth + 1)})`; }).replace(/(\d+)[ui]\b/g, "$1");
    if (!/^[\d\s+\-*/()%<>]+$/.test(t)) throw new Error(`expr ${e}`);
    return Math.trunc(Function(`return (${t});`)() as number);
  };
  const structs = new Map<string, string>();
  for (const m of c.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) structs.set(m[1]!, m[2]!);
  const splitTop = (t: string) => { const o: string[] = []; let d = 0, cur = ""; for (const ch of t) { if (ch === "<") d++; if (ch === ">") d--; if (ch === "," && d === 0) { o.push(cur); cur = ""; } else cur += ch; } if (cur.trim()) o.push(cur); return o; };
  const layout = (t: string): [number, number] => {
    t = t.trim();
    if (/^(f32|u32|i32|bool)$/.test(t) || /^atomic<.*>$/.test(t)) return [4, 4];
    let m = /^vec([234])(?:<\w+>|[fiu])$/.exec(t); if (m) { const n = Number(m[1]); return n === 2 ? [8, 8] : n === 3 ? [12, 16] : [16, 16]; }
    m = /^array<(.+)>$/.exec(t);
    if (m) { const [el, n] = splitTop(m[1]!); const [sz, al] = layout(el!); return [Math.ceil(sz / al) * al * evalN(n!), al]; }
    const body = structs.get(t);
    if (body !== undefined) { let off = 0, al = 1; for (const f of splitTop(body)) { const i = f.indexOf(":"); if (i < 0) continue; const [sz, a] = layout(f.slice(i + 1)); off = Math.ceil(off / a) * a + sz; al = Math.max(al, a); } return [Math.ceil(off / al) * al, al]; }
    throw new Error(`type ${t}`);
  };
  const outB: Record<string, number> = {};
  for (const m of c.matchAll(/var<workgroup>\s*(\w+)\s*:\s*([^;]+);/g)) { if (!used(m[1]!)) continue; try { outB[m[1]!] = layout(m[2]!)[0]; } catch { outB[m[1]!] = -1; } }
  return outB;
}

async function caps(out: string) {
  const psoinfo = resolve(scratch, "psoinfo");
  if (!existsSync(psoinfo)) execFileSync("swiftc", ["-O", "-o", psoinfo, resolve(here, "psoinfo.swift")], { stdio: "inherit" });
  const dir = resolve(out, "wgsl"), files = readdirSync(dir).filter(f => f.endsWith(".wgsl")).sort();
  const res: Record<string, unknown> = {};
  const one = (f: string) => new Promise<void>(done => {
    const base = f.replace(/\.wgsl$/, ""), entry = readFileSync(resolve(dir, `${base}.entry`), "utf8").trim();
    const logf = resolve(dir, `${base}.log`), fd = openSync(logf, "w");
    const k = spawn("node", ["--import", "tsx", resolve(here, "msl.mts"), resolve(dir, f), entry], { cwd: root, stdio: ["ignore", fd, fd] });
    k.on("close", () => {
      closeSync(fd); const txt = readFileSync(logf, "utf8"); const i = txt.lastIndexOf("/* Dumped generated MSL */");
      if (i < 0 || !txt.includes("kernel void dawn_entry_point", i)) { res[base] = { error: txt.slice(0, 300) }; return done(); }
      let m = txt.slice(i + 26); m = m.slice(0, m.lastIndexOf("}") + 1);
      const wg = /\[\[max_total_threads_per_threadgroup\((\d+)\)\]\]/.exec(m);
      const mf = resolve(dir, `${base}.metal`); writeFileSync(mf, m.replace(/\[\[max_total_threads_per_threadgroup\(\d+\)\]\] */g, ""));
      rmSync(logf, { force: true });
      const vars = wgslWorkgroupBytes(readFileSync(resolve(dir, f), "utf8"), n => m.includes(`.${n}=`) || m.includes(`.${n}[`) || m.includes(`.${n})`));
      try {
        const o = execFileSync(psoinfo, [mf], { encoding: "utf8" }).trim().split(/\s+/);
        res[base] = o[1] === "ERR" || o[1] === "nofn" ? { error: o.slice(1).join(" ") } : { wg: wg ? Number(wg[1]) : -1, cap: Number(o[1]), simd: Number(o[3]), tgBytes: Object.values(vars).reduce((a, b) => a + Math.max(b, 0), 0), vars };
      } catch (e) { res[base] = { error: String(e).slice(0, 200) }; }
      done();
    });
  });
  let next = 0; const jobs = Number(process.env.JOBS ?? 6);
  await Promise.all(Array.from({ length: jobs }, async () => { while (next < files.length) await one(files[next++]!); }));
  writeFileSync(resolve(out, "caps.json"), JSON.stringify(res, null, 1));
  for (const [k, v] of Object.entries(res)) log(`cap ${k} ${JSON.stringify(v).slice(0, 200)}`);
}

// ---------------------------------------------------------------------------------------------------------------
async function child() {
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

  const sceneKey = process.env.SCENE ?? "dam", sceneId = SCENES[sceneKey] ?? sceneKey;
  const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 7), LOOP_MS = Number(process.env.LOOP_MS ?? 0), out = resolve(process.env.OUT!);
  const targets = (process.env.KERNELS ?? "uExtend,uTransfer,uAdvect,pDeposit").split(",");
  const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
  const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
  const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
  const quality = profile?.methodId === methodId ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
  log(`staged scene ${sceneId} (${sceneKey}) warm-up ${N}, R ${R}, loop ${LOOP_MS} ms, kernels ${targets.join(",")}, out ${out}`);
  paceDawnEventPolling();
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization",
    ...(process.env.FLUID_WEBGPU_DAWN_FEATURES ? [`enable-dawn-features=${process.env.FLUID_WEBGPU_DAWN_FEATURES}`] : [])]).requestAdapter();
  const raw: GPUDevice = await adapter.requestDevice({ requiredFeatures: ["subgroups", "float32-filterable", "timestamp-query"].filter((f: string) => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const gpuErrors: string[] = [];
  raw.addEventListener?.("uncapturederror", (ev: any) => { gpuErrors.push(String(ev.error?.message)); console.log("GPU ERROR", ev.error?.message); });
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
    if (k % 20 === 0) { await s.brickEngine.fence(); log(`warm-up ${k} ${((performance.now() - t0) / k).toFixed(1)} ms/step`); }
  }
  const P = s.brickEngine;
  await P.fence();
  if (gpuErrors.length) throw new Error(`GPU errors during warm-up: ${gpuErrors[0]}`);

  // ---- the frozen state (census.mts's shadow) ----
  const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const live = [...[...P.buffers.values()] as GPUBuffer[], ...[...new Set([...P.external.values()])].filter((b: GPUBuffer) => (b.usage & COPY) === COPY)] as GPUBuffer[];
  const shadow = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
  const copyAll = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  copyAll(live, shadow);
  const restore = () => copyAll(shadow, live);
  const L = P.layout, slot = P.steps % RING_SLOTS, rec = P.recorded[slot], args = P.buffers.get("args");
  const ring = () => raw.queue.writeBuffer(P.buffers.get("set"), 4 * (L.set.ring + RING_WORDS * slot), P.ringWords(P.steps));
  const sch = [...P.schedule.part1, ...P.schedule.part2], all = [...rec.part1, ...rec.part2], n1 = rec.part1.length;
  const dispatchInto = (pass: GPUComputePassEncoder, r: any) => {
    pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
    if ("direct" in r.dispatch) pass.dispatchWorkgroups(...(r.dispatch.direct as [number, number, number])); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
  };
  const QMAX = 4096, qs = raw.createQuerySet({ type: "timestamp", count: QMAX });
  const qResolve = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const qRead = raw.createBuffer({ size: 8 * QMAX, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const readTimes = async (enc: GPUCommandEncoder, n: number) => {
    enc.resolveQuerySet(qs, 0, n, qResolve, 0); enc.copyBufferToBuffer(qResolve, 0, qRead, 0, 8 * n); raw.queue.submit([enc.finish()]);
    await qRead.mapAsync(GPUMapMode.READ); const t = new BigUint64Array(qRead.getMappedRange(0, 8 * n).slice(0)); qRead.unmap(); return t;
  };
  // ---- the variants: compiled as the engine compiles, bound with its bind() ----
  const lower = brickLowerOn(P.options.lowerDynamicWrites);
  mkdirSync(resolve(out, "wgsl"), { recursive: true });
  type Built = { kernel: string; variant: string; probe: boolean; places: { index: number; rec: any; groups: string }[] };
  const built: Built[] = [];
  const result: any = { scene: sceneId, key: sceneKey, warmup: N, R, date: new Date().toISOString(), steps: P.steps,
    git: (() => { try { return { head: execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(), status: execFileSync("git", ["status", "--short"], { cwd: root, encoding: "utf8" }) }; } catch { return null; } })(),
    kernels: {} };
  for (const K of targets) {
    const spec = KERNELS[K]; if (!spec) throw new Error(`staged: no variants for ${K}`);
    const idx = sch.map((p: any, i: number) => (p.kernel.label === K ? i : -1)).filter((i: number) => i >= 0);
    if (!idx.length) throw new Error(`staged: ${K} is not in the schedule`);
    const k0 = sch[idx[0]!].kernel;
    if (idx.some((i: number) => sch[i].kernel !== k0)) throw new Error(`staged: ${K}'s placements bind different kernels`);
    result.kernels[K] = { entry: k0.entry, placements: idx.map((i: number) => ({ index: i, part: i < n1 ? 1 : 2, dispatch: sch[i].dispatch })), variants: {} };
    for (const v of [...spec.variants, { name: "copy", subs: [] } as Variant]) {
      const code = variantCode(k0.code, k0.entry, v), label = `${K}.m1b.${v.name}`;
      const kv = { ...k0, label, code };
      const text = brickModuleCode(label, code, lower);
      writeFileSync(resolve(out, "wgsl", `${K}.${v.name}.wgsl`), text); writeFileSync(resolve(out, "wgsl", `${K}.${v.name}.entry`), k0.entry);
      if (v.name === "full") { built.push({ kernel: K, variant: v.name, probe: false, places: idx.map((i: number) => ({ index: i, rec: all[i], groups: "" })) }); continue; }
      const module = raw.createShaderModule({ label, code: text });
      const errs = (await module.getCompilationInfo()).messages.filter((m: any) => m.type === "error");
      if (errs.length) throw new Error(`staged: ${label} does not compile:\n${errs.map((m: any) => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
      const layout = raw.createPipelineLayout({ bindGroupLayouts: [raw.createBindGroupLayout({ entries: bindGroupLayoutEntries(k0.bindings) })] });
      P.pipelines.set(kv, await raw.createComputePipelineAsync({ label, layout, compute: { module, entryPoint: k0.entry } }));
      built.push({ kernel: K, variant: v.name, probe: v.probe === true, places: idx.map((i: number) => ({ index: i, rec: P.bind({ ...sch[i], kernel: kv }, slot), groups: "" })) });
    }
    log(`${K}: ${idx.length} placement(s) at ${idx.join(",")}, ${spec.variants.length + 1} variants built`);
  }
  if (gpuErrors.length) throw new Error(`GPU errors building the variants: ${gpuErrors[0]}`);

  // ---- timing: restore, ring, the step's placements before the kernel (untimed, a pass a part), the variant ----
  const sample = async (b: Built, p: Built["places"][number]) => {
    restore(); ring();
    const enc = raw.createCommandEncoder();
    const pre1 = all.slice(0, Math.min(p.index, n1)), pre2 = p.index > n1 ? all.slice(n1, p.index) : [];
    for (const list of [pre1, pre2]) if (list.length) { const pass = enc.beginComputePass(); for (const x of list) dispatchInto(pass, x); pass.end(); }
    const pass = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } }); dispatchInto(pass, p.rec); pass.end();
    const t = await readTimes(enc, 2); return Number(t[1]! - t[0]!) / 1e3;
  };
  const times = new Map<Built, number[][]>(built.map(b => [b, b.places.map(() => [])]));
  for (let r = 0; r < R; r++) {
    const order = r % 2 ? [...built].reverse() : built;
    for (const b of order) for (let j = 0; j < b.places.length; j++) times.get(b)![j]!.push(await sample(b, b.places[j]!));
    if (gpuErrors.length) throw new Error(`GPU error in the timing: ${gpuErrors[0]}`);
    log(`round ${r + 1}/${R}`);
  }
  const min = (v: number[]) => Math.min(...v), med = (v: number[]) => { const a = [...v].sort((x, y) => x - y); return a[a.length >> 1]!; };
  for (const b of built) {
    const t = times.get(b)!, m = t.map(min).reduce((a, x) => a + x, 0), md = t.map(med).reduce((a, x) => a + x, 0);
    result.kernels[b.kernel].variants[b.variant] = { probe: b.probe, minUs: m, medUs: md, samples: t };
  }
  for (const K of targets) {
    const V = result.kernels[K].variants; let prev = 0;
    log(`== ${K} (${result.kernels[K].placements.length} placement(s))`);
    for (const [name, v] of Object.entries(V) as [string, any][]) {
      log(`  ${name.padEnd(12)} min ${(v.minUs / 1000).toFixed(3)} ms  med ${(v.medUs / 1000).toFixed(3)} ms${v.probe || name === "copy" ? "" : `  phase ${((v.minUs - prev) / 1000).toFixed(3)} ms`}`);
      if (!v.probe && name !== "copy") prev = v.minUs;
    }
  }
  writeFileSync(resolve(out, "staged.json"), JSON.stringify(result, null, 1));

  // ---- counters loop: every variant back to back, labelled, timestamped (the parent attaches Instruments) ----
  if (LOOP_MS > 0) {
    restore(); ring(); await raw.queue.onSubmittedWorkDone();
    log("READY");
    const tEnd = performance.now() + LOOP_MS; let cycles = 0;
    while (performance.now() < tEnd) {
      const enc = raw.createCommandEncoder(); let q = 0;
      for (const b of built) for (const p of b.places) {
        const pass = enc.beginComputePass({ label: `m1b.${b.kernel}.${b.variant}`, timestampWrites: { querySet: qs, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } }); q += 2;
        dispatchInto(pass, p.rec); pass.end();
      }
      raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone(); cycles++;
      if (cycles % 10 === 0) { restore(); ring(); }
    }
    log(`DONE ${cycles} cycles`);
    result.counterCycles = cycles; writeFileSync(resolve(out, "staged.json"), JSON.stringify(result, null, 1));
  }
  for (const b of shadow) b.destroy();
  s.destroy?.();
  process.exit(gpuErrors.length ? 3 : 0);
}
