// GPU-shape M6 (lane B2): the whole-step residency audit (docs/plans/gpu-shape/m6-residency.md).
//
// 1. Measure (GPU, exclusive): m4.mts in audit mode, one run a scene:
//      AUDIT=50 MODE=time,clock R=3 CR=3 ONLY=none tools/gpu/fluid-gpu-run.sh timing \
//        node --import tsx tools/experiments/gpu-shape/m4.mts <scene> <out>
//    (every indirect placement timed alone in situ; those over 50 us clocked: groups in flight per pass.)
// 2. Caps (compile only): node --import tsx tools/experiments/gpu-shape/m6.mts caps <out>
//    each <out>/wgsl/<label>.base.wgsl through Dawn (msl.mts) to MSL, the size attribute stripped, psoinfo's
//    maxTotalThreadsPerThreadgroup (the register-bound cap) and the bytes of the var<workgroup> the MSL still uses.
// 3. Table: node --import tsx tools/experiments/gpu-shape/m6.mts table <out-scene-a> [<out-scene-b> ...]
//
// Residency model (the brief's): cap-implied groups in flight = 32 cores x floor(min(cap, 3072) / L) (cap <= 1024, so
// cap / L; cap = 1024 is the API ceiling and the true register ceiling may be higher, up to 3072 / L). Workgroup-memory
// implied = 32 x floor(65536 / wgmem): 64 KB a core, measured by m6-wglimit.mts (floor(64 KB / W) groups a core, and
// at most 3072 threads a core). Resident = the clocked pass's peak groups in flight (a sweep over group start/end;
// the ticker group is not a record); avg = sum of group durations / span (fill, with ramp and tail).
// Class: few (G < 0.8 x cap-implied: the list cannot fill the GPU), wgmem (wgmem-implied < cap-implied and peak >=
// 0.85 x wgmem-implied), registers (peak >= 0.85 x cap-implied, cap < 1024), full (the same with cap 1024),
// launch (peak < 0.85 x cap-implied and median group under 2 us: groups end before more are launched), other
// (peak below both bounds). "+tail" when the span after 95% of groups ended is over 35% of the span.
// Residency deficit = 1 - peak / cap-implied (>= 0); ranked by us x deficit; fill = avg / peak.
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { execFileSync, spawnSync } from "node:child_process";
import { homedir } from "node:os";

const here = dirname(fileURLToPath(import.meta.url)), root = resolve(here, "../../..");
const psoinfo = resolve(homedir(), "code/me/fluid-scratch/gpu-shape/psoinfo");
const WG_LIMIT = 64 * 1024, CORES = 32;

type Cap = { cap: number; wgmem: number; error?: string };
function caps(out: string) {
  const dir = resolve(out, "wgsl"), cdir = resolve(out, "caps"); mkdirSync(cdir, { recursive: true });
  for (const f of readdirSync(dir).filter(f => f.endsWith(".base.wgsl"))) {
    const label = f.slice(0, -".base.wgsl".length), cf = resolve(cdir, `${label}.json`);
    if (existsSync(cf) && JSON.parse(readFileSync(cf, "utf8")).cap > 0) continue;
    const code = readFileSync(resolve(dir, f), "utf8");
    const entries = [...code.matchAll(/@compute\s+@workgroup_size\([^)]*\)\s*fn\s+(\w+)/g)].map(m => m[1]!);
    const entry = entries.includes(label) ? label : entries[0]!;
    // The child's output goes to a file (a pipe loses the tail of large dumps, census.mts); a dump without the entry is retried.
    let text = "", i = -1;
    for (let attempt = 0; attempt < 3 && (i < 0 || !text.includes("kernel void dawn_entry_point", i)); attempt++) {
      const logf = resolve(cdir, `${label}.log`), fd = openSync(logf, "w");
      spawnSync("node", ["--import", "tsx", resolve(here, "msl.mts"), resolve(dir, f), entry], { cwd: root, stdio: ["ignore", fd, fd] });
      closeSync(fd); text = readFileSync(logf, "utf8"); i = text.lastIndexOf("/* Dumped generated MSL */");
    }
    let c: Cap = { cap: -1, wgmem: -1 };
    if (i < 0) c.error = text.slice(0, 300);
    else {
      let m = text.slice(i + 26); m = m.slice(0, m.lastIndexOf("}") + 1);
      const mf = resolve(cdir, `${label}.metal`); writeFileSync(mf, m.replace(/\[\[max_total_threads_per_threadgroup\(\d+\)\]\] */g, ""));
      try {
        const o = execFileSync(psoinfo, [mf], { encoding: "utf8" }).trim().split(/\s+/);
        if (o[1] === "ERR" || o[1] === "nofn") c.error = o.slice(1).join(" "); else c.cap = Number(o[1]);
      } catch (e) { c.error = String(e).slice(0, 200); }
      c.wgmem = wgslWorkgroupBytes(code, n => m.includes(`.${n}=`));
    }
    writeFileSync(cf, JSON.stringify(c)); console.log(`${label} cap ${c.cap} wgmem ${c.wgmem}${c.error ? ` ERR ${c.error.slice(0, 120)}` : ""}`);
  }
}

function table(outs: string[]) {
  type Row = { scene: string; i: number; label: string; us: number; G: number; L: number; cap: number; wgmem: number; capRes: number; wgRes: number; meas: number; peak: number; tailFrac: number; durMax: number; span: number; cls: string; deficit: number; score: number };
  const rows: Row[] = [];
  for (const out of outs) {
    const j = JSON.parse(readFileSync(resolve(out, "m4.json"), "utf8"));
    for (const [k, p] of Object.entries<any>(j.placements)) {
      const runs = p.clock?.base?.runs; if (!runs) continue;
      const best = runs.reduce((a: any, x: any) => (x.passUs < a.passUs ? x : a));
      const c: Cap = JSON.parse(readFileSync(resolve(out, "caps", `${p.label}.json`), "utf8"));
      const code = readFileSync(resolve(out, "wgsl", `${p.label}.base.wgsl`), "utf8");
      const L = Number(new RegExp(`@workgroup_size\\((\\d+)\\)\\s*fn\\s+${p.label}\\b`).exec(code)?.[1] ?? /@workgroup_size\((\d+)\)/.exec(code)![1]);
      const G = best.n - 1, us = p.us.base.min;
      const capRes = CORES * Math.floor(Math.min(c.cap, 3072) / L), wgRes = c.wgmem > 0 ? CORES * Math.floor(WG_LIMIT / c.wgmem) : Infinity;
      let meas = best.avgInflight, tailFrac = best.tailUs / best.spanUs, capped = "";
      if (best.state !== 2) {
        // The ticker gave up (a group queued behind it on its core): the stuck groups' ends are the frozen clock. In
        // flight over the window in which 95% of groups ended, from run 0's per-group records.
        const tsv = resolve(out, `clock-${p.label}-base-${k}.tsv`);
        const g = readFileSync(tsv, "utf8").trim().split("\n").slice(1).map(l => l.split("\t").map(Number) as [number, number, number]);
        const ends = g.map(x => x[2]).sort((x, y) => x - y), e95 = ends[Math.floor(0.95 * (ends.length - 1))]!;
        meas = g.filter(x => x[2] <= e95).reduce((t, x) => t + x[2] - x[1], 0) / e95; tailFrac = NaN; capped = " (ticker capped)";
      }
      const peak = best.peakInflight, p50 = best.dur.p50;
      let cls = "other";
      if (G < 0.8 * capRes) cls = "few";
      else if (wgRes < capRes && peak >= 0.85 * wgRes) cls = "wgmem";
      else if (peak >= 0.85 * capRes) cls = c.cap < 1024 ? "registers" : "full";
      else if (p50 < 2) cls = "launch";
      if (tailFrac > 0.35) cls += "+tail";
      const deficit = Math.max(0, 1 - peak / capRes);
      rows.push({ scene: j.key, i: Number(k), label: p.label + capped, us, G, L, cap: c.cap, wgmem: c.wgmem, capRes, wgRes, meas, peak, tailFrac, durMax: best.dur.max, span: best.spanUs, cls, deficit, score: us * deficit });
    }
  }
  rows.sort((a, b) => b.score - a.score);
  const f1 = (x: number) => (Number.isFinite(x) ? x.toFixed(0) : "-");
  const md = ["| scene | # | pass | us | G | L | cap | wgmem B | cap-implied | wgmem-implied | peak in flight | fill | tail % | class | deficit | us x deficit |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"];
  for (const r of rows) md.push(`| ${r.scene} | ${r.i} | ${r.label} | ${r.us.toFixed(0)} | ${r.G} | ${r.L} | ${r.cap} | ${r.wgmem} | ${f1(r.capRes)} | ${f1(r.wgRes)} | ${r.peak} | ${(r.meas / r.peak).toFixed(2)} | ${Number.isNaN(r.tailFrac) ? "-" : (100 * r.tailFrac).toFixed(0)} | ${r.cls} | ${r.deficit.toFixed(2)} | ${r.score.toFixed(0)} |`);
  // Families: a scene's placements of one kernel (the PCG passes repeat), ranked by summed us x deficit.
  const fam = new Map<string, Row[]>(); for (const r of rows) { const key = `${r.scene}\t${r.label}`; fam.set(key, [...(fam.get(key) ?? []), r]); }
  const fr = [...fam.values()].map(v => ({ ...v[0]!, n: v.length, us: v.reduce((t, r) => t + r.us, 0), meas: v.reduce((t, r) => t + r.meas, 0) / v.length, peak: Math.max(...v.map(r => r.peak)), tailFrac: v.reduce((t, r) => t + r.tailFrac, 0) / v.length, deficit: v.reduce((t, r) => t + r.deficit, 0) / v.length, score: v.reduce((t, r) => t + r.score, 0), cls: [...new Set(v.map(r => r.cls))].join("/") })).sort((a, b) => b.score - a.score);
  md.push("", "| scene | pass (placements) | us | G | L | cap | wgmem B | cap-implied | wgmem-implied | peak in flight | fill | tail % | class | deficit | us x deficit |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
  for (const r of fr) md.push(`| ${r.scene} | ${r.label} (${r.n}) | ${r.us.toFixed(0)} | ${r.G} | ${r.L} | ${r.cap} | ${r.wgmem} | ${f1(r.capRes)} | ${f1(r.wgRes)} | ${r.peak} | ${(r.meas / r.peak).toFixed(2)} | ${Number.isNaN(r.tailFrac) ? "-" : (100 * r.tailFrac).toFixed(0)} | ${r.cls} | ${r.deficit.toFixed(2)} | ${r.score.toFixed(0)} |`);
  console.log(md.join("\n"));
  writeFileSync(resolve(outs[0]!, "..", "m6-table.tsv"), ["scene\ti\tlabel\tus\tG\tL\tcap\twgmem\tcapRes\twgRes\tmeas\tpeak\ttailFrac\tclass\tdeficit\tscore", ...rows.map(r => [r.scene, r.i, r.label, r.us.toFixed(1), r.G, r.L, r.cap, r.wgmem, r.capRes, r.wgRes, r.meas.toFixed(1), r.peak, r.tailFrac.toFixed(3), r.cls, r.deficit.toFixed(3), r.score.toFixed(1)].join("\t"))].join("\n") + "\n");
}

/** Bytes of a module's var<workgroup> declarations that `used` keeps (census.mts's wgslWorkgroupBytes). */
function wgslWorkgroupBytes(code: string, used: (name: string) => boolean): number {
  const c = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "");
  const consts = new Map<string, string>();
  for (const m of c.matchAll(/\bconst\s+(\w+)\s*(?::\s*\w+)?\s*=\s*([^;]+);/g)) consts.set(m[1]!, m[2]!);
  const evalN = (e: string, depth = 0): number => {
    const t = e.replace(/\b([A-Za-z_]\w*)\b/g, id => { const v = consts.get(id); if (v === undefined || depth > 8) throw new Error(`const ${id}`); return `(${evalN(v, depth + 1)})`; })
      .replace(/(\d+)[ui]\b/g, "$1");
    if (!/^[\d\s+\-*/()%<>]+$/.test(t)) throw new Error(`expr ${e}`);
    return Math.trunc(Function(`return (${t});`)() as number);
  };
  const structs = new Map<string, string>();
  for (const m of c.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) structs.set(m[1]!, m[2]!);
  const splitTop = (t: string) => { const out: string[] = []; let d = 0, cur = ""; for (const ch of t) { if (ch === "<") d++; if (ch === ">") d--; if (ch === "," && d === 0) { out.push(cur); cur = ""; } else cur += ch; } if (cur.trim()) out.push(cur); return out; };
  const layout = (t: string): [number, number] => {
    t = t.trim();
    if (/^(f32|u32|i32|bool)$/.test(t) || /^atomic<.*>$/.test(t)) return [4, 4];
    let m = /^vec([234])(?:<\w+>|[fiu])$/.exec(t); if (m) { const n = Number(m[1]); return n === 2 ? [8, 8] : n === 3 ? [12, 16] : [16, 16]; }
    m = /^array<(.+)>$/.exec(t);
    if (m) { const [el, n] = splitTop(m[1]!); const [sz, al] = layout(el!); const stride = Math.ceil(sz / al) * al; return [stride * evalN(n!), al]; }
    const body = structs.get(t);
    if (body !== undefined) {
      let off = 0, al = 1;
      for (const f of splitTop(body)) { const i = f.indexOf(":"); if (i < 0) continue; const [sz, a] = layout(f.slice(i + 1)); off = Math.ceil(off / a) * a + sz; al = Math.max(al, a); }
      return [Math.ceil(off / al) * al, al];
    }
    throw new Error(`type ${t}`);
  };
  let bytes = 0;
  for (const m of c.matchAll(/var<workgroup>\s*(\w+)\s*:\s*([^;]+);/g)) {
    if (!used(m[1]!)) continue;
    try { bytes += layout(m[2]!)[0]; } catch { bytes += 0; }
  }
  return bytes;
}

const [cmd, ...rest] = process.argv.slice(2);
if (cmd === "caps") for (const o of rest) caps(resolve(o));
else if (cmd === "table") table(rest.map(o => resolve(o)));
else { console.error("usage: m6.mts caps <out>... | table <out>..."); process.exit(2); }
