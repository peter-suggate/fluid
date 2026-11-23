// GPU-shape M4 (lane B2): the C-list passes reshaped (docs/plans/gpu-shape/m4-clist.md).
//
//   tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/m4.mts <scene> [outdir]
//   env STEPS=100 (warm-up)  R=7 (interleaved samples)  CR=3 (clocked samples)  MODE=exact,time,clock,step (any subset)
//       ONLY=label.name,...   (variants to run; default all)   STEP_VARIANTS=a+b,c (whole-step combos of variant keys)
//       REF_DIR=dir           (a "ref" variant per placement: dir/<label>.wgsl at dir/dispatch.json's args word;
//                              written by any run as <out>/kernels: the exactness of a landing against its parent)
//
// Warm the brick engine up STEPS steps (fixed dt 1/60), freeze it (a GPU shadow of every buffer), then for each
// placement whose kernel a variant names (VARIANTS: text substitutions of the schedule's own kernel text, compiled as
// the engine compiles them, bound with its bind(); a split variant's args word k n, n the placement's own, written by
// an untimed one-lane kernel):
//  exact  from one in-situ snapshot: base, base again, every variant; every read-write binding word for word.
//  time   in situ (restore, ring, the step's placements before it untimed, the variant alone in a timestamped pass),
//         interleaved with base, the clock probe after each round.
//  clock  per-workgroup start / end on a GPU tick counter (cpages.mts's clock): duration p50 / p90 / max, in flight.
//  step   the whole step (both parts) with the named variants swapped in, walls interleaved with base.
// AUDIT=<us> (M6, the residency audit): every indirect placement is a base target; MODE=time,clock times them all and
//  clocks those whose min is over <us>; each audited kernel's lowered module goes to <out>/wgsl/<label>.base.wgsl
//  (for regcap.sh), the clock line adds avg (sum of group durations / span) and peak (sweep) groups in flight.
import { resolve, dirname } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync, readFileSync, existsSync } from "node:fs";
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

// ---- the clock ----
const CLK_HEAD = 8, CLK_REC = 4, CLK_MAX_GROUPS = 1 << 16, CLK_MAX_TICKS = 1 << 16; // ticks: 1-5 a us, so the ticker gives up after 13-53 ms (a group queued behind the ticker on its core deadlocks it: operator on paddleboats)
// Diagnostic variants (name ending in "diag"): 4 words a lane (64-lane kernels) after the group records.
const CLK_LANE_BASE = CLK_HEAD + CLK_REC * CLK_MAX_GROUPS, CLK_LANE_GROUPS = 8192, CLK_LANE_WORDS = 4;
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
 for(var i=g.x;i<cn.x;i+=cn.y){if(ca[i]!=cb[i]){atomicAdd(&co[0],1u);atomicMin(&co[1],i+cn.z);if(ca[i]==0x4aed5be2u){atomicAdd(&co[2],1u);}}}
}`;

// ---- the M4 variants ----
type Variant = { label: string; name: string; k: number; at?: number; make: (code: string, entry: string) => string };
/** A page of a G1 list over k workgroups: the gate on k n, the slot of g / k, m4Sub = g % k. */
const splitGate = (c: string, gate: string, slot: string, list: string, k: number, what: string) => {
  c = sub1(c, gate, gate.replace("workgroupUniformLoad(", `${k}u*workgroupUniformLoad(`), `${what} gate`);
  return sub1(c, slot, slot.replace(`nbListSlot(${list},g)`, `nbListSlot(${list},g/${k}u)`) + `let m4Sub=g%${k}u;`, `${what} slot`);
};
const uExt4SeedSplit = (k: number) => (c0: string) => {
  let c = splitGate(c0, "if(g>=workgroupUniformLoad(&sCount)){return;}", "let slot=nbListSlot(NB_LIST_C_ACT,g);", "NB_LIST_C_ACT", k, "uExt4Seed");
  const nt = 512 / k;
  return sub1(c, "for(var r=0u;r<2u;r++){\n  let j=lid+256u*r;let T=P+", `for(var j=lid+${nt}u*m4Sub;j<${nt}u*(m4Sub+1u);j+=256u){\n  let T=P+`, "uExt4Seed tiles");
};
/** uExt4Seed over k workgroups of `lanes` lanes, its workgroup arrays sized to the sub-block. */
const uExt4SeedLanes = (k: number, lanes: number) => (c0: string) => {
  const nt = 512 / k;
  let c = splitGate(c0, "if(g>=workgroupUniformLoad(&sCount)){return;}", "let slot=nbListSlot(NB_LIST_C_ACT,g);", "NB_LIST_C_ACT", k, "uExt4Seed");
  c = sub1(c, "for(var r=0u;r<2u;r++){\n  let j=lid+256u*r;let T=P+", `for(var j=lid+${nt}u*m4Sub;j<${nt}u*(m4Sub+1u);j+=${lanes}u){\n  let T=P+`, "uExt4Seed tiles");
  c = sub1(c, "@compute @workgroup_size(256) fn uExt4Seed(", `@compute @workgroup_size(${lanes}) fn uExt4Seed(`, "uExt4Seed size");
  c = sub1(c, "var<workgroup> wHeavy:array<u32,1536>;", `var<workgroup> wHeavy:array<u32,${3 * nt}>;`, "wHeavy");
  c = sub1(c, "var<workgroup> wSub:array<vec2f,256>;", `var<workgroup> wSub:array<vec2f,${lanes}>;`, "wSub");
  c = sub1(c, "var<workgroup> wSubLiquid:array<u32,256>;", `var<workgroup> wSubLiquid:array<u32,${lanes}>;`, "wSubLiquid");
  return sub1(c, "for(var b=0u;b<n;b+=16u){", `for(var b=0u;b<n;b+=${lanes / 16}u){`, "uExt4Seed rounds");
};
/** pressureRowsC over k workgroups, its workgroup arrays sized to the sub-block (rcR by the lane). */
const rowsCSized = (k: number) => (c0: string) => {
  const w = 512 / k;
  let c = rowsCSplit(k)(c0);
  c = sub1(c, "var<workgroup> rcE:array<u32,3072>;", `var<workgroup> rcE:array<u32,${6 * w}>;`, "rcE");
  c = sub1(c, "var<workgroup> rcR:array<f32,512>;", `var<workgroup> rcR:array<f32,${w}>;`, "rcR");
  c = sub1(c, "var<workgroup> rcD:array<f32,512>;var<workgroup> rcX:array<f32,512>;var<workgroup> rcP:array<f32,512>;", `var<workgroup> rcD:array<f32,${w}>;var<workgroup> rcX:array<f32,${w}>;var<workgroup> rcP:array<f32,${w}>;`, "rcD");
  c = sub1(c, "rcR[i]=root;", "rcR[lid]=root;", "rcR store");
  return sub1(c, "poSeamFace(m,pm,lm,g2,rcR[ti])", `poSeamFace(m,pm,lm,g2,rcR[ti-${w}u*m4Sub])`, "rcR load");
};
const rowsCSplit = (k: number) => (c0: string) => {
  const w = 512 / k, f = w / 16;
  let c = sub1(c0, "@compute @workgroup_size(512) fn pressureRowsC(", `@compute @workgroup_size(${w}) fn pressureRowsC(`, "rowsC size");
  c = splitGate(c, "let g=nbGroup(wid);if(g>=workgroupUniformLoad(&rcCount)){return;}", "let cs=nbListSlot(NB_LIST_C_ALL,g);", "NB_LIST_C_ALL", k, "rowsC");
  c = sub1(c, "let i=lid;let T=O+", `let i=lid+${w}u*m4Sub;let T=O+`, "rowsC tile");
  c = sub1(c, "for(var c0=0u;c0<n;c0+=32u){", `for(var c0=0u;c0<n;c0+=${f}u){`, "rowsC chunk");
  return sub1(c, "e<min(sBase+sCount,c0+32u);e++){", `e<min(sBase+sCount,c0+${f}u);e++){`, "rowsC sums");
};
/** lcRetire: the projection's cell corners resolved once a cell (PHI is read only: the kernel writes PHI1). */
const lcGcache = (c0: string) => {
  const c = sub1(c0, " var v:array<f32,8>;var terms:array<f32,8>;\n for(var k=0u;k<8u;k++){\n  v[k]=nbVertexPhi(origin+width*lcCorner(k));",
    " var v:array<f32,8>;var terms:array<f32,8>;\n if(!(all(origin==m4GO)&&width==m4GW)){for(var k=0u;k<8u;k++){m4GV[k]=nbVertexPhi(origin+width*lcCorner(k));}m4GO=origin;m4GW=width;}\n for(var k=0u;k<8u;k++){\n  v[k]=m4GV[k];", "lcGradient");
  return sub1(c, "fn lcGradient(", "var<private> m4GO:vec3i=vec3i(-2147483647);\nvar<private> m4GW:i32=0;\nvar<private> m4GV:array<f32,8>;\nfn lcGradient(", "lcGradient decl");
};
/** lcRetire: gcache's code shape without the cache (the corners through m4GV every call): is gcache's difference the cache? */
const lcGfetch = (c0: string) => {
  const c = sub1(c0, " var v:array<f32,8>;var terms:array<f32,8>;\n for(var k=0u;k<8u;k++){\n  v[k]=nbVertexPhi(origin+width*lcCorner(k));",
    " var v:array<f32,8>;var terms:array<f32,8>;\n {for(var k=0u;k<8u;k++){m4GV[k]=nbVertexPhi(origin+width*lcCorner(k));}m4GO=origin;m4GW=width;}\n for(var k=0u;k<8u;k++){\n  v[k]=m4GV[k];", "lcGradient");
  return sub1(c, "fn lcGradient(", "var<private> m4GO:vec3i=vec3i(-2147483647);\nvar<private> m4GW:i32=0;\nvar<private> m4GV:array<f32,8>;\nfn lcGradient(", "lcGradient decl");
};
/** lcRetire race check: gfetch's values (always fetched), plus the cache kept beside them; a cache hit whose cached
 * corners differ from the fresh ones (bit for bit) marks the lane, whose PHI1 word becomes 7777777. Exact against
 * base: no corner changed between two reads in the pass (gcache's difference is not a race). */
const lcGcheck = (c0: string) => {
  let c = sub1(c0, " var v:array<f32,8>;var terms:array<f32,8>;\n for(var k=0u;k<8u;k++){\n  v[k]=nbVertexPhi(origin+width*lcCorner(k));",
    " var v:array<f32,8>;var terms:array<f32,8>;\n {let m4Hit=all(origin==m4GO)&&width==m4GW;for(var k=0u;k<8u;k++){let m4x=nbVertexPhi(origin+width*lcCorner(k));if(m4Hit&&bitcast<u32>(m4x)!=bitcast<u32>(m4GV[k])){m4Bad=true;}m4GV[k]=m4x;}m4GO=origin;m4GW=width;}\n for(var k=0u;k<8u;k++){\n  v[k]=m4GV[k];", "lcGradient");
  c = sub1(c, "  lcSetC(NB_PHI_C_PHI1,slot,l,lcRetired(v,lcPhiC(NB_PHI_C_PHI,slot,l)));", "  {let m4r=lcRetired(v,lcPhiC(NB_PHI_C_PHI,slot,l));lcSetC(NB_PHI_C_PHI1,slot,l,select(m4r,7777777.0,m4Bad));}", "lcRetirePage store");
  return sub1(c, "fn lcGradient(", "var<private> m4GO:vec3i=vec3i(-2147483647);\nvar<private> m4GW:i32=0;\nvar<private> m4GV:array<f32,8>;\nvar<private> m4Bad:bool=false;\nfn lcGradient(", "lcGradient decl");
};
/** lcRetire: the 3^3 stencil's h owners of the workgroup's plane (10 x 10 x 3 tiles) staged once a workgroup. */
const lcStencil = (c0: string) => {
  let c = sub1(c0, "if(nbTileOwnerH(n)){return true;}}\n return false;\n}",
    "let m4r=vec3u(n-m4StLo);if(all(m4r<vec3u(10u,10u,3u))){if(m4St[m4r.x+10u*(m4r.y+10u*m4r.z)]!=0u){return true;}}else if(nbTileOwnerH(n)){return true;}}\n return false;\n}", "lcStencilFine");
  c = sub1(c, "fn lcStencilFine(", "var<workgroup> m4St:array<u32,300>;\nvar<private> m4StLo:vec3i;\nfn lcStencilFine(", "lcStencilFine decl");
  return sub1(c, "fn lcRetirePage(slot:u32,live:bool,lid:u32){\n lcCheckLattice(lid);\n if(!live){return;}",
    "fn lcRetirePage(slot:u32,live:bool,lid:u32){\n lcCheckLattice(lid);\n m4StLo=(nbCOrigin(slot)>>vec3u(2u))+vec3i(0,0,i32(lcPart))-vec3i(1);\n for(var m4s=lid;m4s<300u;m4s+=64u){m4St[m4s]=select(0u,1u,live&&nbTileOwnerH(m4StLo+vec3i(i32(m4s%10u),i32((m4s/10u)%10u),i32(m4s/100u))));}\n workgroupBarrier();\n if(!live){return;}", "lcRetirePage");
};
/** censusDecide: TRIG's 8 children x 8 tiles a lane each (lanes 0..7 walked 8 tiles each), joined in workgroup memory. */
const cdTrig64 = (c0: string) => {
  const from = c0.indexOf("  let child=8u*r+lane;\n  if(lane<8u){"); if (from < 0 || count(c0, "  let child=8u*r+lane;") !== 1) throw new Error("censusDecide: TRIG changed shape");
  const to = c0.indexOf("  workgroupBarrier();\n", from) + "  workgroupBarrier();\n".length;
  const old = c0.slice(from, to);
  for (const x of ["let fs=nbChild(cs,child);", "let b=vec3u(child&3u,(child>>2u)&3u,child>>4u);var bits=0u;", "let q=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);let qi=q.x|(q.y<<3u)|(q.z<<6u);let w1=ncFin(cs,qi);",
    "bits|=select(0u,1u<<j,ncNeedTriggered(xTileOf(cs,qi),cs,qi,(w1>>16u)&63u));", "bits|=select(0u,1u<<(8u+j),((w1>>17u)&1u)!=0u);", "atomicStore(&nbSet[NB_S_TRIG+fs],bits);", "if((bits&0xffu)!=0u){atomicAdd(&wgCount[3],countOneBits(bits&0xffu));}"])
    if (!old.includes(x)) throw new Error(`censusDecide TRIG: no ${x}`);
  const neu = `  let child=8u*r+(lane>>3u);let j=lane&7u;
  let fs=nbChild(cs,child);
  if(fs<NB_SEAM){
    let b=vec3u(child&3u,(child>>2u)&3u,child>>4u);var bits=0u;
    let q=2u*b+vec3u(j&1u,(j>>1u)&1u,j>>2u);let qi=q.x|(q.y<<3u)|(q.z<<6u);let w1=ncFin(cs,qi);
    bits|=select(0u,1u<<j,ncNeedTriggered(xTileOf(cs,qi),cs,qi,(w1>>16u)&63u));
    bits|=select(0u,1u<<(8u+j),((w1>>17u)&1u)!=0u);
    if(bits!=0u){atomicOr(&m4Trig[lane>>3u],bits);}
  }
  workgroupBarrier();
  if(j==0u&&fs<NB_SEAM){
    let bits=atomicLoad(&m4Trig[lane>>3u]);
    atomicStore(&nbSet[NB_S_TRIG+fs],bits);
    if((bits&0xffu)!=0u){atomicAdd(&wgCount[3],countOneBits(bits&0xffu));}
  }
  workgroupBarrier();
`;
  let c = c0.slice(0, from) + neu + c0.slice(to);
  c = sub1(c, "if(lane<4u){atomicStore(&wgCount[lane],0u);}", "if(lane<4u){atomicStore(&wgCount[lane],0u);}if(lane<8u){atomicStore(&m4Trig[lane],0u);}", "censusDecide init");
  return sub1(c, "var<workgroup> wgGo:u32;", "var<workgroup> wgGo:u32;\nvar<workgroup> m4Trig:array<atomic<u32>,8>;", "censusDecide decl");
};
/** A diagnostic: each lane's ticks over `around` and three counters (m4A, m4B, m4C) into the lane records. */
const laneDiag = (c0: string, gLine: string, around: string, counters: [string, string][]) => {
  let c = sub1(c0, gLine, gLine + "m4G=g;", "diag group");
  c = sub1(c, around, `let m4t0=atomicLoad(&m3clk[0]);${around}let m4t1=atomicLoad(&m3clk[0]);{let m4o=${CLK_LANE_BASE}u+${CLK_LANE_WORDS}u*(min(m4G,${CLK_LANE_GROUPS - 1}u)*64u+lid);atomicStore(&m3clk[m4o],m4t1-m4t0);atomicStore(&m3clk[m4o+1u],m4A);atomicStore(&m3clk[m4o+2u],m4B);atomicStore(&m3clk[m4o+3u],m4C);}`, "diag around");
  for (const [from, to] of counters) c = sub1(c, from, to, `diag ${from.slice(0, 30)}`);
  const first = c.search(/\nfn /);
  return c.slice(0, first) + "\nvar<private> m4G:u32;var<private> m4A:u32;var<private> m4B:u32;var<private> m4C:u32;" + c.slice(first);
};
const cdDiag = (c: string) => laneDiag(c.replace("cdTile(cs,lane+64u*r,lane,reach,base);", "{let lid=lane;cdTile(cs,lane+64u*r,lane,reach,base);}"), "let g=nbGroup(wid);let e=g>>3u;let r=g&7u;", "cdTile(cs,lane+64u*r,lane,reach,base);", [
  // m4A: ncRangeKeys' cells; m4B: ncInterfaceAny's pages (low 16) and list entries (high 16); m4C: calls of ncInterfaceAny (low 16), ncRangeKeys (high 16).
  ["   var at:array<u32,6>;\n   if(j==0u)", "   m4A+=1u;var at:array<u32,6>;\n   if(j==0u)"],
  ["fn ncInterfaceAny(low:vec3i,high:vec3i)->bool{\n", "fn ncInterfaceAny(low:vec3i,high:vec3i)->bool{\n m4C+=1u;"],
  ["   let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}\n   if(ncInterfaceIn(", "   m4B+=65536u;let s=ncListSlot(e);let q=ncSlotPage(s);if(any(q<pa)||any(q>pb)){continue;}\n   if(ncInterfaceIn("],
  ["  let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}\n  if(ncInterfaceIn(", "  m4B+=1u;let q=vec3i(x,y,z);let s=ncPageSlot(q);if(s==0xffffffffu){continue;}\n  if(ncInterfaceIn("],
  [" var keys=ncIdentityKeys();\n", " m4C+=65536u;var keys=ncIdentityKeys();\n"],
]);
const lrDiag = (c: string) => laneDiag(c.replace("let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;", "let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;"), "let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;", "lcSetC(NB_PHI_C_PHI1,slot,l,lcRetired(v,lcPhiC(NB_PHI_C_PHI,slot,l)));", [
  // m4A: the path (1 cold/hot, 2 coarse build, 3 reach build, 4 sentinel) | projection iterations << 4; m4B: nbVertexPhi calls; m4C: lcSample calls.
  [" if(lcVertexCold(v)||lcHotCorner(v)){return cur;}", " if(lcVertexCold(v)||lcHotCorner(v)){m4A|=1u;return cur;}"],
  [" if(!lcStencilFine(c>>vec3u(2u))){return", " if(!lcStencilFine(c>>vec3u(2u))){m4A|=2u;return"],
  [" if(fgReach2(c)){return", " if(fgReach2(c)){m4A|=3u;return"],
  [" return nbfFarSentinel(cur,1.0);\n}", " m4A|=4u;return nbfFarSentinel(cur,1.0);\n}"],
  ["fn nbVertexPhi(v:vec3i)->f32{", "fn nbVertexPhi(v:vec3i)->f32{m4B+=1u;"],
  ["fn lcSample(f:u32,p:vec3f)->f32{", "fn lcSample(f:u32,p:vec3f)->f32{m4C+=1u;"],
  ["    let s=lcGradient(cell,q);", "    m4A+=16u;let s=lcGradient(cell,q);"],
]);
/** censusDecide: the calm and candidate counters a workgroup's (workgroup atomics, one global add a counter a
 * workgroup after the last barrier) instead of a global atomic a lane on six shared words. */
const CD_COUNTERS = ["NB_H_CENSUS_CALM_HELD", "NB_H_CENSUS_CALM", "NB_H_CENSUS_CALM_MOVING", "NB_H_CENSUS_CALM_ZERO", "NB_H_CENSUS_CALM_CROSSING", "NB_H_CENSUS_CAND"];
const cdCount = (c0: string) => {
  let c = c0;
  CD_COUNTERS.forEach((name, k) => { c = sub1(c, `atomicAdd(&nbSet[${name}],1u);`, `atomicAdd(&m4Cnt[${k}u],1u);`, name); });
  c = sub1(c, "if(lane<4u){atomicStore(&wgCount[lane],0u);}", `if(lane<4u){atomicStore(&wgCount[lane],0u);}if(lane<${CD_COUNTERS.length}u){atomicStore(&m4Cnt[lane],0u);}`, "censusDecide count init");
  c = sub1(c, "    if(c.w!=0u){atomicAdd(&nbSet[NB_H_CENSUS_TRIGGERED],c.w);}", "    if(c.w!=0u){atomicAdd(&nbSet[NB_H_CENSUS_TRIGGERED],c.w);}\n" + CD_COUNTERS.map((n, k) => `    {let m4x=atomicLoad(&m4Cnt[${k}u]);if(m4x!=0u){atomicAdd(&nbSet[${n}],m4x);}}`).join("\n"), "censusDecide count flush");
  return sub1(c, "var<workgroup> wgGo:u32;", `var<workgroup> wgGo:u32;\nvar<workgroup> m4Cnt:array<atomic<u32>,${CD_COUNTERS.length}>;`, "censusDecide count decl");
};
/** lcAdvect over P z planes a workgroup (64 P lanes, 8 / P workgroups a page on C_ACT's G1 words): one ring of
 * 13 x 13 x (P + 4) a workgroup instead of P rings of 13 x 13 x 5. Each vertex's path is unchanged. */
const laPlanes = (P: number) => (c0: string) => {
  const K = 8 / P, z = P + 4;
  let c = sub1(c0, "@compute @workgroup_size(64) fn lcAdvect(", `@compute @workgroup_size(${64 * P}) fn lcAdvect(`, "lcAdvect size");
  c = sub1(c, "let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;let live=e<nbListCount(NB_LIST_C_ACT);\n var slot=0u;if(live){slot=nbListSlot(NB_LIST_C_ACT,e);}\n lcAdvectPage(",
    `let g=nbGroup(wid);let e=g/${K}u;lcPart=${P}u*(g%${K}u);let live=e<nbListCount(NB_LIST_C_ACT);\n var slot=0u;if(live){slot=nbListSlot(NB_LIST_C_ACT,e);}\n lcAdvectPage(`, "lcAdvect entry");
  c = sub1(c, "for(var j=lid;j<845u;j+=64u){lcRing[j]", `for(var j=lid;j<${169 * z}u;j+=${64 * P}u){lcRing[j]`, "lcAdvect stage");
  c = sub1(c, "var<workgroup> lcRing:array<f32,845>;", `var<workgroup> lcRing:array<f32,${169 * z}>;`, "lcRing decl");
  return sub1(c, "if(all(r<vec3u(13u,13u,5u))){return lcRing[", `if(all(r<vec3u(13u,13u,${z}u))){return lcRing[`, "lcPhiTap bound");
};
const LA_RING = " lcRingLow=(o>>vec3u(2u))+vec3i(-2,-2,i32(lcPart)-2);\n for(var j=lid;j<845u;j+=64u){lcRing[j]=lcFResolved(4*(lcRingLow+vec3i(i32(j%13u),i32((j/13u)%13u),i32(j/169u))));}\n workgroupBarrier();lcRingOn=true;\n";
/** Exact: no ring (every tap resolved where it is read). */
const laNoRing = (c: string) => sub1(c, LA_RING, " workgroupBarrier();\n", "lcAdvect ring");
/** Time only (not exact): the exits alone, or the exits and the staging, no full path. */
const laExits = (c: string) => sub1(c, LA_RING + " // Exit 5: the full path.\n if(full){", " return;\n if(full){", "lcAdvect exits");
const laStage = (c: string) => sub1(c, " // Exit 5: the full path.\n if(full){", " if(lcRing[lid]==1.2345e-37){lcSetC(NB_PHI_C_PHI1,slot,l,0.0);}return;\n if(full){", "lcAdvect stage only");
/** The full path's lane count and ticks (diag): m4A = 1 on a lane taking it. */
const laDiag = (c: string) => laneDiag(c, "let g=nbGroup(wid);let e=g>>3u;lcPart=g&7u;", " if(full){lcSetC(NB_PHI_C_PHI1,slot,l,lcFullAdvected(v,c,dt,hm));}", [
  [" // Exit 5: the full path.\n", " // Exit 5: the full path.\n m4A=select(0u,1u,full);\n"],
]);
/** censusCoarse ablations (time only, not exact): one job phase dropped. */
const ccDrop = (from: string, to: string) => (c: string) => sub1(c, from, to, "censusCoarse drop");
/** censusCoarse: a face between two bounded tiles of the slab (x and y) read once, by the lower tile's high job, and
 * bounding both (both probes coarse: one 4h patch at the same anchor). */
const ccShare = (c0: string) => {
  let c = sub1(c0, "      let axis=(j%6u)/2u;let sign=select(-1,1,(j&1u)!=0u);\n      let probe=cqFaceProbe(cs,r,j);",
    "      let axis=(j%6u)/2u;let sign=select(-1,1,(j&1u)!=0u);\n      let m4x=vec2u(owner&7u,owner>>3u);let m4d=select(8u,1u,axis==0u);\n      if(axis<2u&&sign<0&&m4x[min(axis,1u)]>0u&&cqBounded[owner-m4d]!=0u){continue;}\n      var m4o=64u;if(axis<2u&&sign>0&&m4x[min(axis,1u)]<7u&&cqBounded[owner+m4d]!=0u){m4o=owner+m4d;}\n      let probe=cqFaceProbe(cs,r,j);", "cc share job");
  c = sub1(c, "      if(side*side==1u){cqBound(owner,axis,anchor);}", "      if(side*side==1u){m4Bound2(owner,m4o,axis,anchor);}", "cc share bound");
  return sub1(c, "fn cqBound(owner:u32,axis:u32,anchor:vec3i){", `fn m4Bound2(owner:u32,other:u32,axis:u32,anchor:vec3i){
  if(ncFaceLow(anchor,axis)){return;}
  let u=ncFace(anchor,axis);
  let finite=abs(u)<=3.0e38;let lo=ncOrderKey(select(-3.0e38,u,finite));let hi=ncOrderKey(select(3.0e38,u,finite));
  atomicMin(&cqLow[owner][axis],lo);atomicMax(&cqHigh[owner][axis],hi);
  if(other<64u){atomicMin(&cqLow[other][axis],lo);atomicMax(&cqHigh[other][axis],hi);}
}
fn cqBound(owner:u32,axis:u32,anchor:vec3i){`, "cc share fn");
};
/** censusCoarse: the bounded tiles' face jobs listed as they classify (a workgroup list), the face loop over the list
 * instead of all 384 jobs a slab (the unbounded tiles' lanes idled through their six). */
const ccCompact = (c0: string) => {
  let c = sub1(c0, "      classified=true;cqBounded[lane]=1u;", "      classified=true;cqBounded[lane]=1u;\n      {let m4b=atomicAdd(&m4Jobs,6u);for(var f=0u;f<6u;f++){m4Job[m4b+f]=6u*lane+f;}}", "cc compact push");
  c = sub1(c, "if(lane==0u){atomicStore(&cqErrors,0u);", "if(lane==0u){atomicStore(&m4Jobs,0u);atomicStore(&cqErrors,0u);", "cc compact init");
  c = sub1(c, "    for(var j=lane;j<6u*64u;j+=64u){\n      let owner=j/6u;if(cqBounded[owner]==0u){continue;}", "    let m4n=atomicLoad(&m4Jobs);\n    for(var m4k=lane;m4k<m4n;m4k+=64u){\n      let j=m4Job[m4k];let owner=j/6u;", "cc compact loop");
  return sub1(c, "var<workgroup> cqParts:atomic<u32>;", "var<workgroup> cqParts:atomic<u32>;\nvar<workgroup> m4Jobs:atomic<u32>;\nvar<workgroup> m4Job:array<u32,384>;", "cc compact decl");
};
/** censusCoarse's workgroup memory 7148 -> 4532 B (its residency was its workgroup memory's: ~8 groups a core against
 * a 704-thread cap): the job lists packed (u8 tiles, u16 faces and vertices), the tile flags bit masks. */
const CC_PACK: [string, string][] = [
  ["var<workgroup> cqError:array<u32,64>;", "var<workgroup> cqError:array<atomic<u32>,16>;"],
  ["var<workgroup> cqDeep:array<u32,64>;", "var<workgroup> cqDeep:array<atomic<u32>,16>;"],
  ["var<workgroup> cqIsDeep:array<u32,64>;", "var<workgroup> cqIsDeep:array<atomic<u32>,2>;"],
  ["var<workgroup> cqVertex:array<u32,484>;", "var<workgroup> cqVertex:array<atomic<u32>,242>;"],
  ["var<workgroup> cqBounded:array<u32,64>;", "var<workgroup> cqBounded:array<atomic<u32>,2>;"],
  ["var<workgroup> cqPart:array<u32,384>;", "var<workgroup> cqPart:array<atomic<u32>,192>;"],
  ["    cqIsDeep[lane]=0u;\n", "    if(lane<2u){atomicStore(&cqIsDeep[lane],0u);atomicStore(&cqBounded[lane],0u);}if(lane<16u){atomicStore(&cqError[lane],0u);atomicStore(&cqDeep[lane],0u);}\n    for(var z=lane;z<242u;z+=64u){atomicStore(&cqVertex[z],0u);if(z<192u){atomicStore(&cqPart[z],0u);}}\n"],
  ["    cqBounded[lane]=0u;\n", ""],
  ["classified=true;cqBounded[lane]=1u;", "classified=true;atomicOr(&cqBounded[lane>>5u],1u<<(lane&31u));"],
  ["cqError[atomicAdd(&cqErrors,1u)]=lane;", "let k=atomicAdd(&cqErrors,1u);atomicOr(&cqError[k>>2u],lane<<(8u*(k&3u)));"],
  ["cqDeep[atomicAdd(&cqDeeps,1u)]=lane;cqIsDeep[lane]=1u;", "let k=atomicAdd(&cqDeeps,1u);atomicOr(&cqDeep[k>>2u],lane<<(8u*(k&3u)));atomicOr(&cqIsDeep[lane>>5u],1u<<(lane&31u));"],
  ["let owner=j/6u;if(cqBounded[owner]==0u){continue;}", "let owner=j/6u;if(((atomicLoad(&cqBounded[owner>>5u])>>(owner&31u))&1u)==0u){continue;}"],
  ["else{cqPart[atomicAdd(&cqParts,1u)]=j;}", "else{let k=atomicAdd(&cqParts,1u);atomicOr(&cqPart[k>>1u],j<<(16u*(k&1u)));}"],
  ["let owner=cqError[j/8u];", "let owner=(atomicLoad(&cqError[(j/8u)>>2u])>>(8u*((j/8u)&3u)))&255u;"],
  ["let owner=cqDeep[j/64u];", "let owner=(atomicLoad(&cqDeep[(j/64u)>>2u])>>(8u*((j/64u)&3u)))&255u;"],
  ["cqVertex[atomicAdd(&cqVertices,1u)]=v;", "let k=atomicAdd(&cqVertices,1u);atomicOr(&cqVertex[k>>1u],v<<(16u*(k&1u)));"],
  ["let v=cqVertex[j];", "let v=(atomicLoad(&cqVertex[j>>1u])>>(16u*(j&1u)))&0xffffu;"],
  ["if(cqIsDeep[owner]==0u){continue;}", "if(((atomicLoad(&cqIsDeep[owner>>5u])>>(owner&31u))&1u)==0u){continue;}"],
  ["let face=cqPart[j/16u];", "let face=(atomicLoad(&cqPart[(j/16u)>>1u])>>(16u*((j/16u)&1u)))&0xffffu;"]
];
const ccPack = (c0: string) => CC_PACK.reduce((c, [a, b]) => sub1(c, a, b, `cc pack ${a.slice(0, 30)}`), c0);
/** certError's workgroup memory 6.3 -> 1.6 KB: the seed flags and the x and y passes (3 bits an entry) four entries
 * a word, a lane a word (no atomics). */
const CE_PACK: [string, string][] = [
  ["var<workgroup> sSeed:array<u32,1000>;\nvar<workgroup> sX:array<u32,400>;\nvar<workgroup> sY:array<u32,160>;", "var<workgroup> sSeed:array<u32,250>;\nvar<workgroup> sX:array<u32,100>;\nvar<workgroup> sY:array<u32,40>;\nfn ceSeed(c:u32)->u32{return (sSeed[c>>2u]>>(8u*(c&3u)))&255u;}\nfn ceX(c:u32)->u32{return (sX[c>>2u]>>(8u*(c&3u)))&255u;}\nfn ceY(c:u32)->u32{return (sY[c>>2u]>>(8u*(c&3u)))&255u;}"],
  ["  for(var k=lane;k<1000u;k+=64u){\n    let T=origin+vec3i(i32(k%10u),i32((k/10u)%10u),i32(k/100u));var w=0u;", "  for(var m=lane;m<250u;m+=64u){var word=0u;for(var b=0u;b<4u;b++){let k=4u*m+b;\n    let T=origin+vec3i(i32(k%10u),i32((k/10u)%10u),i32(k/100u));var w=0u;"],
  ["    sSeed[k]=w;\n  }", "    word|=w<<(8u*b);}\n    sSeed[m]=word;\n  }"],
  ["  for(var k=lane;k<400u;k+=64u){\n    let c=3u+(k%4u)+10u*(k/4u);var f=0u;\n    if((sSeed[c]&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(sSeed[u32(i32(c)+d)],d);}}\n    sX[k]=f;\n  }", "  for(var m=lane;m<100u;m+=64u){var word=0u;for(var b=0u;b<4u;b++){let k=4u*m+b;\n    let c=3u+(k%4u)+10u*(k/4u);var f=0u;\n    if((ceSeed(c)&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(ceSeed(u32(i32(c)+d)),d);}}\n    word|=f<<(8u*b);}\n    sX[m]=word;\n  }"],
  ["  for(var k=lane;k<160u;k+=64u){\n    let c=(k%4u)+4u*(3u+((k/4u)%4u)+10u*(k/16u));var f=0u;\n    if((sX[c]&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(sX[u32(i32(c)+4*d)],d);}}\n    sY[k]=f;\n  }", "  for(var m=lane;m<40u;m+=64u){var word=0u;for(var b=0u;b<4u;b++){let k=4u*m+b;\n    let c=(k%4u)+4u*(3u+((k/4u)%4u)+10u*(k/16u));var f=0u;\n    if((ceX(c)&4u)!=0u){f=4u;for(var d=-3;d<=3;d++){f|=ceDilate(ceX(u32(i32(c)+4*d)),d);}}\n    word|=f<<(8u*b);}\n    sY[m]=word;\n  }"],
  ["support|=ceDilate(sY[u32(i32(c)+16*d)],d);", "support|=ceDilate(ceY(u32(i32(c)+16*d)),d);"]
];
// ---- M6 (the residency audit's wgmem-bound passes) ----
const all1 = (s: string, from: string, to: string, what: string) => { if (!s.includes(from)) throw new Error(`${what}: "${from.slice(0, 80)}" absent`); return s.split(from).join(to); };
/** pDeposit 27.5 -> ~9.9 KB: three workgroups a brick, one a face axis (its 11 x 10 x 10 window, momentum then
 * weight in one 2200-word array; the 11^3 vertex ring fits in it), each reads the brick's records, only axis 0
 * restates them. The fdep sums are integer atomics, so the order is free. */
const PD_AXIS3: [string, string][] = [
  ["var<workgroup> wMom:array<atomic<u32>,3300>;\nvar<workgroup> wWeight:array<atomic<u32>,3300>;", "var<workgroup> wMom:array<atomic<u32>,2200>;\nvar<private> pdAxis:u32;"],
  [" return a*PD_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);", " return u32((r.z*d.y+r.y)*d.x+r.x);"],
  ["fn pdFace(s:u32)->vec3i{let a=s/PD_WINDOW;let r=i32(s%PD_WINDOW);", "fn pdFace(s:u32)->vec3i{let a=pdAxis;let r=i32(s);"],
  ["atomicAdd(&wMom[s],bitcast<u32>(momentum));atomicAdd(&wWeight[s],weight);", "atomicAdd(&wMom[s],bitcast<u32>(momentum));atomicAdd(&wMom[1100u+s],weight);"],
  ["let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);", "let g=nbGroup(wid)/3u;pdAxis=nbGroup(wid)%3u;let live=g<nbListCount(NB_LIST_F_ALL);"],
  ["for(var j=lid;j<3300u;j+=256u){atomicStore(&wMom[j],0u);atomicStore(&wWeight[j],0u);}", "for(var j=lid;j<2200u;j+=256u){atomicStore(&wMom[j],0u);}"],
  ["  parts[i].before.w=nbParticleRestate(s,nbParticleAge(s),select(1.0,0.0,supported));", "  if(pdAxis==0u){parts[i].before.w=nbParticleRestate(s,nbParticleAge(s),select(1.0,0.0,supported));}"],
  ["for(var j=lid;j<3300u;j+=256u){\n  let w=atomicLoad(&wWeight[j]);", "for(var j=lid;j<1100u;j+=256u){\n  let w=atomicLoad(&wMom[1100u+j]);"],
  ["let f=pdFace(j);let a=j/PD_WINDOW;let fs=fgSlot(f);", "let f=pdFace(j);let a=pdAxis;let fs=fgSlot(f);"],
];
const pdAxis3 = (c0: string) => {
  const c = PD_AXIS3.reduce((c, [a, b]) => sub1(c, a, b, `pd axis3 ${a.slice(0, 30)}`), c0);
  const d = all1(c, "for(var axis=0u;axis<3u;axis++){\n  let base=nbfP2GBase(p,axis);", "for(var axis=pdAxis;axis<=pdAxis;axis++){\n  let base=nbfP2GBase(p,axis);", "pd axis3 scatter");
  if (d.includes("wWeight")) throw new Error("pd axis3: wWeight left");
  return d;
};
/** pDeposit with two face axes resident at a time (4400 words, 17.6 KB + the window: 3 a core): "axis2" two
 * workgroups a brick (axes 0-1, axis 2), "seq2" one workgroup a brick, the records read twice (axes 0-1, then 2). */
const PD_TWO = (split: boolean): [string, string][] => [
  ["var<workgroup> wMom:array<atomic<u32>,3300>;\nvar<workgroup> wWeight:array<atomic<u32>,3300>;", "var<workgroup> wMom:array<atomic<u32>,4400>;\nvar<private> pdA0:u32;var<private> pdA1:u32;"],
  [" return a*PD_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);", " return (a-pdA0)*PD_WINDOW+u32((r.z*d.y+r.y)*d.x+r.x);"],
  ["fn pdFace(s:u32)->vec3i{let a=s/PD_WINDOW;", "fn pdFace(s:u32)->vec3i{let a=pdA0+s/PD_WINDOW;"],
  ["atomicAdd(&wMom[s],bitcast<u32>(momentum));atomicAdd(&wWeight[s],weight);", "atomicAdd(&wMom[s],bitcast<u32>(momentum));atomicAdd(&wMom[2200u+s],weight);"],
  ...(split ? [["let g=nbGroup(wid);let live=g<nbListCount(NB_LIST_F_ALL);", "let g=nbGroup(wid)/2u;pdA0=2u*(nbGroup(wid)%2u);pdA1=select(2u,3u,pdA0==2u);let live=g<nbListCount(NB_LIST_F_ALL);"],
    ["for(var j=lid;j<3300u;j+=256u){atomicStore(&wMom[j],0u);atomicStore(&wWeight[j],0u);}", "for(var j=lid;j<4400u;j+=256u){atomicStore(&wMom[j],0u);}"]] as [string, string][]
    : [["for(var j=lid;j<3300u;j+=256u){atomicStore(&wMom[j],0u);atomicStore(&wWeight[j],0u);}", "for(var pdRound=0u;pdRound<2u;pdRound++){pdA0=2u*pdRound;pdA1=select(2u,3u,pdRound==1u);\n for(var j=lid;j<4400u;j+=256u){atomicStore(&wMom[j],0u);}"],
      ["  atomicAdd(&fdep[at],atomicLoad(&wMom[j]));atomicAdd(&fdep[at+1u],w);\n }\n}", "  atomicAdd(&fdep[at],atomicLoad(&wMom[j]));atomicAdd(&fdep[at+1u],w);\n }\n workgroupBarrier();}\n}"]] as [string, string][]),
  ["  parts[i].before.w=nbParticleRestate(s,nbParticleAge(s),select(1.0,0.0,supported));", "  if(pdA0==0u){parts[i].before.w=nbParticleRestate(s,nbParticleAge(s),select(1.0,0.0,supported));}"],
  ["for(var j=lid;j<3300u;j+=256u){\n  let w=atomicLoad(&wWeight[j]);", "for(var j=lid;j<1100u*(pdA1-pdA0);j+=256u){\n  let w=atomicLoad(&wMom[2200u+j]);"],
  ["let f=pdFace(j);let a=j/PD_WINDOW;let fs=fgSlot(f);", "let f=pdFace(j);let a=pdA0+j/PD_WINDOW;let fs=fgSlot(f);"],
];
const pdTwo = (split: boolean) => (c0: string) => {
  const c = PD_TWO(split).reduce((c, [a, b]) => sub1(c, a, b, `pd two ${a.slice(0, 30)}`), c0);
  const d = all1(c, "for(var axis=0u;axis<3u;axis++){\n  let base=nbfP2GBase(p,axis);", "for(var axis=pdA0;axis<pdA1;axis++){\n  let base=nbfP2GBase(p,axis);", "pd two scatter");
  if (d.includes("wWeight")) throw new Error("pd two: wWeight left");
  return d;
};
/** The vec3f workgroup arrays (16-byte stride) as planes of f32: pmgSetupF 25.1 -> 20.5 KB, pmgCoarsenC 18.2 -> 15.6. */
const planes = (decls: [string, number][]) => (c0: string) => {
  let c = c0;
  for (const [n, len] of decls) {
    c = sub1(c, `var<workgroup> ${n}:array<vec3f,${len}>;`, `var<workgroup> ${n}:array<f32,${3 * len}>;`, `planes ${n}`);
    c = c.replace(new RegExp(`\\b${n}\\[([^\\]]+)\\]\\[a\\]`, "g"), (_, i) => `${n}[${len}u*a+${i}]`);
    c = c.replace(new RegExp(`\\b${n}\\[([^\\]]+)\\]=([A-Za-z0-9_]+);`, "g"), (_, i, v) => `${n}[${i}]=${v}.x;${n}[${len}u+${i}]=${v}.y;${n}[${2 * len}u+${i}]=${v}.z;`);
    if (new RegExp(`\\b${n}\\[[^\\]]+\\](?!=)(?!\\[)`).test(c.replace(new RegExp(`\\b${n}\\[\\d+u\\*a\\+[^\\]]+\\]|\\b${n}\\[(\\d+u\\+)?[^\\]]+\\]=[^;]+;`, "g"), ""))) throw new Error(`planes ${n}: a use left`);
  }
  return c;
};
const cePack = (c0: string) => CE_PACK.reduce((c, [a, b]) => sub1(c, a, b, `ce pack ${a.slice(0, 30)}`), c0);
const VARIANTS: Variant[] = [
  { label: "censusDecide", name: "count", k: 1, make: cdCount },
  { label: "censusDecide", name: "count+trig64", k: 1, make: c => cdTrig64(cdCount(c)) },
  { label: "censusDecide", name: "diag", k: 1, make: cdDiag },
  { label: "lcRetire", name: "diag", k: 1, make: lrDiag },
  { label: "lcRetire", name: "gcache", k: 1, make: lcGcache },
  { label: "lcRetire", name: "gfetch", k: 1, make: lcGfetch },
  { label: "lcRetire", name: "gcheck", k: 1, make: lcGcheck },
  { label: "lcAdvect", name: "noring", k: 1, make: laNoRing },
  { label: "lcAdvect", name: "exits", k: 1, make: laExits },
  { label: "lcAdvect", name: "stage", k: 1, make: laStage },
  { label: "lcAdvect", name: "diag", k: 1, make: laDiag },
  { label: "lcAdvect", name: "p2", k: 4, at: 12 * 18, make: laPlanes(2) },
  { label: "lcAdvect", name: "p4", k: 2, at: 12 * 18, make: laPlanes(4) },
  { label: "lcAdvect", name: "p8", k: 1, at: 12 * 18, make: laPlanes(8) },
  { label: "censusCoarse", name: "share", k: 1, make: ccShare },
  { label: "censusCoarse", name: "compact", k: 1, make: ccCompact },
  { label: "censusCoarse", name: "pack", k: 1, make: ccPack },
  { label: "certError", name: "pack", k: 1, make: cePack },
  { label: "censusCoarse", name: "noface", k: 1, make: ccDrop("for(var j=lane;j<6u*64u;j+=64u){", "for(var j=lane;j<0u;j+=64u){") },
  { label: "censusCoarse", name: "noerr", k: 1, make: ccDrop("for(var j=lane;j<8u*errors;j+=64u){", "for(var j=lane;j<0u*errors;j+=64u){") },
  { label: "censusCoarse", name: "nodepth", k: 1, make: ccDrop("for(var j=lane;j<vertices;j+=64u){", "for(var j=lane;j<0u*vertices;j+=64u){") },
  { label: "censusCoarse", name: "noparts", k: 1, make: ccDrop("for(var j=lane;j<16u*parts;j+=64u){", "for(var j=lane;j<0u*parts;j+=64u){") },
  { label: "pDeposit", name: "id", k: 1, make: c => c },
  { label: "pDeposit", name: "axis3", k: 3, make: pdAxis3 },
  { label: "pDeposit", name: "axis2", k: 2, make: pdTwo(true) },
  { label: "pDeposit", name: "seq2", k: 1, make: pdTwo(false) },
  { label: "pmgSetupF", name: "planes", k: 1, make: planes([["wK", 512], ["wKU", 512], ["w1K", 64], ["w1KU", 64]]) },
  { label: "pmgCoarsenC", name: "planes", k: 1, make: planes([["w2K", 512], ["w3K", 64], ["w4K", 8]]) },
  { label: "lcAdvect", name: "id", k: 1, make: c => c },
  { label: "censusCoarse", name: "id", k: 1, make: c => c },
  { label: "uExt4", name: "id", k: 1, make: c => c },
  { label: "certError", name: "id", k: 1, make: c => c },
  { label: "certCoarse", name: "id", k: 1, make: c => c },
  { label: "pressureProjectC", name: "id", k: 1, make: c => c },
  { label: "lcCouple", name: "id", k: 1, make: c => c },
  { label: "lcPublish", name: "id", k: 1, make: c => c },
  { label: "uCache4", name: "id", k: 1, make: c => c },
  { label: "lcRetire", name: "stencil", k: 1, make: lcStencil },
  { label: "lcRetire", name: "both", k: 1, make: c => lcStencil(lcGcache(c)) },
  { label: "censusDecide", name: "trig64", k: 1, make: cdTrig64 },
  ...[2, 4, 8].map(k => ({ label: "uExt4Seed", name: `k${k}`, k, make: uExt4SeedSplit(k) })),
  ...[2, 4].map(k => ({ label: "pressureRowsC", name: `k${k}`, k, make: rowsCSplit(k) })),
  ...[2, 4].map(k => ({ label: "pressureRowsC", name: `k${k}s`, k, make: rowsCSized(k) })),
  { label: "uExt4Seed", name: "k2s", k: 2, make: uExt4SeedLanes(2, 256) },
  { label: "uExt4Seed", name: "k4w128", k: 4, make: uExt4SeedLanes(4, 128) },
  { label: "uExt4Seed", name: "k8w64", k: 8, make: uExt4SeedLanes(8, 64) },
];

if (process.argv[2] === "--check") {
  // Offline: node m4.mts --check <kernels dir> : every variant applies to its kernel (writes them next to it).
  const dir = process.argv[3]!;
  for (const v of VARIANTS) {
    const f = resolve(dir, `${v.label}.wgsl`); if (!existsSync(f)) { log(`${v.label}: no ${f}`); continue; }
    const c = readFileSync(f, "utf8"), entry = v.label;
    try { const t = v.make(c, entry); writeFileSync(`${f}.${v.name}.wgsl`, t); writeFileSync(`${f}.${v.name}.clk.wgsl`, clocked(t, entry)); log(`${v.label}.${v.name}: ok`); }
    catch (e) { log(`${v.label}.${v.name}: ${(e as Error).message}`); }
  }
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
  const out = resolve(outArg ?? resolve(scratch, "m4", sceneKey)); mkdirSync(resolve(out, "wgsl"), { recursive: true });
  const N = Number(process.env.STEPS ?? 100), R = Number(process.env.R ?? 7), CR = Number(process.env.CR ?? 3);
  const modes = new Set((process.env.MODE ?? "exact,time,clock").split(","));
  const only = process.env.ONLY ? new Set(process.env.ONLY.split(",")) : undefined;
  const methodId = "uniform-narrow-band-flip", method = getMethod(methodId);
  const definition = getSceneDefinition(sceneId), profile = definition.methodProfile, base = sceneDocument(definition);
  const scene = { ...base, numerics: { ...base.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 } };
  const quality = profile?.methodId === methodId ? profile.quality : "balanced";
  const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === methodId ? profile.overrides : {}), engine: "brick" });
  const head = (() => { try { return execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root, encoding: "utf8" }).trim(); } catch { return "?"; } })();
  log(`m4 scene ${sceneId} (${sceneKey}) HEAD ${head} warm-up ${N}, R ${R}, CR ${CR}, modes ${[...modes].join(",")}, out ${out}`);
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
  const save = () => writeFileSync(resolve(out, "m4.json"), JSON.stringify(result, null, 1));
  void L;

  // ---- dump: every placement's kernel text and dispatch (a later run's REF_DIR) ----
  const kdir = resolve(out, "kernels"); mkdirSync(kdir, { recursive: true });
  const dispatch: Record<string, unknown> = {};
  for (let i = 0; i < sch.length; i++) { const p = sch[i]; if (dispatch[p.kernel.label] !== undefined) continue; dispatch[p.kernel.label] = p.dispatch; writeFileSync(resolve(kdir, `${p.kernel.label}.wgsl`), p.kernel.code); }
  writeFileSync(resolve(kdir, "dispatch.json"), JSON.stringify(dispatch, null, 1));

  // ---- the frozen state ----
  const COPY = GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const live = [...[...P.buffers.values()] as GPUBuffer[], ...[...new Set([...P.external.values()])].filter((b: GPUBuffer) => (b.usage & COPY) === COPY)] as GPUBuffer[];
  const shadow = live.map(b => raw.createBuffer({ size: b.size, usage: COPY }));
  const copyAll = (from: GPUBuffer[], to: GPUBuffer[]) => { const enc = raw.createCommandEncoder(); from.forEach((b, i) => enc.copyBufferToBuffer(b, 0, to[i]!, 0, b.size)); raw.queue.submit([enc.finish()]); };
  copyAll(live, shadow);
  const restore = () => copyAll(shadow, live);
  const ring = () => raw.queue.writeBuffer(P.buffers.get("set"), 4 * (P.layout.set.ring + RING_WORDS * slot), P.ringWords(P.steps));
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

  const lower = brickLowerOn(P.options.lowerDynamicWrites);
  const clk = raw.createBuffer({ label: "m3clk", size: 4 * (CLK_LANE_BASE + CLK_LANE_WORDS * 64 * CLK_LANE_GROUPS), usage: GPUBufferUsage.STORAGE | COPY });
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

  // ---- the targets: (placement, variant) ----
  const ref = process.env.REF_DIR ? { dir: process.env.REF_DIR, dispatch: JSON.parse(readFileSync(resolve(process.env.REF_DIR, "dispatch.json"), "utf8")) as Record<string, any> } : undefined;
  type Target = { i: number; key: string; name: string; code: () => string; a: (w: number) => { pre: any; argsBuf: GPUBuffer } | undefined };
  const targets: Target[] = [];
  const argsOf = (i: number, k: number, w: number, at = sch[i].dispatch.indirect) => (k === 1 && w === 0 && at === sch[i].dispatch.indirect ? undefined : argsRec(at, at, k - 1, w));
  for (let i = 0; i < sch.length; i++) {
    const k0 = sch[i].kernel;
    for (const v of VARIANTS) if (v.label === k0.label && (!only || only.has(`${v.label}.${v.name}`))) targets.push({ i, key: `${v.label}.${v.name}`, name: v.name, code: () => v.make(k0.code, k0.entry), a: w => argsOf(i, v.k, w, v.at) });
    if (ref && ref.dispatch[k0.label] && existsSync(resolve(ref.dir, `${k0.label}.wgsl`)) && (!only || only.has(`${k0.label}.ref`))) {
      const d = ref.dispatch[k0.label]; if (!("indirect" in d)) continue;
      const text = readFileSync(resolve(ref.dir, `${k0.label}.wgsl`), "utf8");
      if (text === k0.code && d.indirect === sch[i].dispatch.indirect) continue;
      targets.push({ i, key: `${k0.label}.ref`, name: "ref", code: () => text, a: w => argsRec(d.indirect, d.indirect, 0, w) });
    }
  }
  const audit = process.env.AUDIT !== undefined ? Number(process.env.AUDIT) : undefined;
  const auditSet = new Set<number>();
  if (audit !== undefined) for (let i = 0; i < sch.length; i++) {
    const k = sch[i].kernel; if (!("indirect" in sch[i].dispatch)) continue;
    auditSet.add(i); const f = resolve(out, "wgsl", `${k.label}.base.wgsl`);
    if (!existsSync(f)) writeFileSync(f, brickModuleCode(k.label, k.code, lower));
  }
  const placements = [...new Set([...targets.map(t => t.i), ...auditSet])].sort((a, b) => a - b);
  log(`targets: ${targets.map(t => `${t.i}:${t.key}`).join(" ")}`);
  const variants = new Map<string, { rec: any; pre: any[] }>();
  const V = async (t: Target, clockOn = false) => {
    const key = `${t.i}\t${t.name}${clockOn ? ".clk" : ""}`; let v = variants.get(key);
    if (!v) {
      const code = clockOn ? clocked(t.code(), sch[t.i].kernel.entry) : t.code(), a = t.a(clockOn ? 1 : 0);
      const r = await variantRec(t.i, `${sch[t.i].kernel.label}.m4.${t.name}${clockOn ? ".clk" : ""}.${t.i}`, code, clockOn);
      v = { rec: a ? { ...r, argsBuf: a.argsBuf } : r, pre: a ? [a.pre] : [] }; variants.set(key, v);
    }
    return v;
  };
  const isDiag = (t: { name: string }) => t.name.endsWith("diag");
  for (const t of targets) if (!isDiag(t)) await V(t);
  fail("building the variants");
  result.placements = Object.fromEntries(placements.map(i => [i, { label: sch[i].kernel.label, part: i < n1 ? 1 : 2, dispatch: sch[i].dispatch, variants: {} as any }]));

  // ---- exact ----
  if (modes.has("exact")) {
    const cmpMod = raw.createShaderModule({ code: COMPARE_WGSL }), cmpPipe = raw.createComputePipeline({ layout: "auto", compute: { module: cmpMod, entryPoint: "main" } });
    const cmpOut = raw.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const cmpRead = raw.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const compare = async (a: GPUBuffer, b: GPUBuffer): Promise<[number, number, number]> => {
      raw.queue.writeBuffer(cmpOut, 0, new Uint32Array([0, 0xffffffff, 0, 0]));
      const enc = raw.createCommandEncoder(); const CH = 2 ** 30;
      for (let off = 0; off < a.size; off += CH) {
        const size = Math.min(CH, a.size - off), u = raw.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
        new Uint32Array(u.getMappedRange()).set([size / 4, 1024 * 256, off / 4, 0]); u.unmap();
        const bg = raw.createBindGroup({ layout: cmpPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: a, offset: off, size } }, { binding: 1, resource: { buffer: b, offset: off, size } }, { binding: 2, resource: { buffer: cmpOut } }, { binding: 3, resource: { buffer: u } }] });
        const pass = enc.beginComputePass(); pass.setPipeline(cmpPipe); pass.setBindGroup(0, bg); pass.dispatchWorkgroups(1024); pass.end();
      }
      enc.copyBufferToBuffer(cmpOut, 0, cmpRead, 0, 16); raw.queue.submit([enc.finish()]);
      await cmpRead.mapAsync(GPUMapMode.READ); const w = new Uint32Array(cmpRead.getMappedRange().slice(0)); cmpRead.unmap(); return [w[0]!, w[1]!, w[2]!];
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
    for (const i of placements) {
      const k = sch[i].kernel;
      restore(); ring(); { const enc = raw.createCommandEncoder(); prefixInto(enc, i); raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone(); }
      copyAll(live, snap); await raw.queue.onSubmittedWorkDone();
      const names = k.bindings.filter((b: any) => b.kind === "rw").map((b: any) => b.name as string), bufs = [...new Set(names.map(bufOf))] as GPUBuffer[];
      await fromSnap([all[i]]); { const enc = raw.createCommandEncoder(); for (const b of bufs) enc.copyBufferToBuffer(b, 0, refOf(b), 0, b.size); raw.queue.submit([enc.finish()]); }
      const ex: Record<string, { words: number; first: number; sentinel: number }> = {};
      const cases: [string, any[]][] = [["base", [all[i]]], ...targets.filter(t => t.i === i && !isDiag(t)).map(t => { const v = variants.get(`${i}\t${t.name}`)!; return [t.name, [...v.pre, v.rec]] as [string, any[]]; })];
      for (const [v, list] of cases) {
        await fromSnap(list); let d = 0, first = -1, sent = 0; for (const b of bufs) { const [n, f, s] = await compare(b, refOf(b)); if (n && first < 0) first = f; d += n; sent += s; } ex[v] = { words: d, first, sentinel: sent };
      }
      result.placements[i].rw = names; result.placements[i].exact = ex;
      log(`  exact ${i} ${k.label} (${names.join(",")}): ${Object.entries(ex).map(([v, d]) => `${v} ${d.words}${d.sentinel ? ` (${d.sentinel} the gcheck sentinel 7777777)` : ""}`).join(", ")} words differ`);
    }
    for (const b of snap) b.destroy();
    fail("in the exactness check");
    save();
  }

  // ---- time ----
  if (modes.has("time")) {
    const cases: { key: string; run: () => Promise<number> }[] = [];
    for (const i of placements) {
      cases.push({ key: `${i}\tbase`, run: async () => (await sample(i, all[i])).us });
      for (const t of targets.filter(t => t.i === i && !isDiag(t))) { const v = variants.get(`${i}\t${t.name}`)!; cases.push({ key: `${i}\t${t.name}`, run: async () => (await sample(i, v.rec, v.pre)).us }); }
    }
    const { t, probes } = await rounds(cases);
    result.timeProbes = probes;
    for (const i of placements) {
      const e = result.placements[i]; e.us = {};
      for (const name of ["base", ...targets.filter(x => x.i === i && !isDiag(x)).map(x => x.name)]) e.us[name] = { ...stats(t.get(`${i}\t${name}`)!), samples: t.get(`${i}\t${name}`) };
      log(`  time ${String(i).padStart(3)} ${e.label.padEnd(14)} base ${e.us.base.min.toFixed(1)} med ${e.us.base.med.toFixed(1)}  ${Object.keys(e.us).filter(x => x !== "base").map(v => `${v} ${e.us[v].min.toFixed(1)} (${(e.us[v].min - e.us.base.min).toFixed(1)}) med ${e.us[v].med.toFixed(1)}`).join("  ")} us`);
    }
    log(`time: clock probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }

  // ---- clock ----
  if (modes.has("clock")) {
    const ct: Target[] = [];
    for (const i of placements) { const k = sch[i].kernel; ct.push({ i, key: `${k.label}.base`, name: "base", code: () => k.code, a: w => argsRec(sch[i].dispatch.indirect, 0, 0, w) }, ...targets.filter(t => t.i === i)); }
    for (const tg of ct) {
      const e = result.placements[tg.i], runs: any[] = [];
      if (audit !== undefined && tg.name === "base" && auditSet.has(tg.i) && !targets.some(t => t.i === tg.i) && !((e.us?.base?.min ?? Infinity) > audit)) continue;
      let v: { rec: any; pre: any[] };
      try {
        raw.pushErrorScope("validation"); let got: any, bad: unknown;
        try { got = await V(tg, true); } catch (x) { bad = x; }
        const err = await raw.popErrorScope(); if (bad) throw bad; if (err) throw new Error(err.message); v = got;
      }
      catch (x) { log(`  clock ${String(tg.i).padStart(3)} ${e.label.padEnd(14)} ${tg.name.padEnd(8)} skipped: ${String((x as Error).message).split("\n")[0]}`); e.clockSkipped = String(x); continue; }
      for (let r = 0; r < CR; r++) {
        const { us, w } = await sample(tg.i, v.rec, v.pre, true); fail(`clocking ${e.label}`);
        const n = w![2]!, ticks = w![0]!, state = w![3]!, done = w![1]!, recs: { s: number; e: number }[] = [];
        for (let g = 0; g < Math.min(n, CLK_MAX_GROUPS); g++) { const o = CLK_HEAD + CLK_REC * g; recs.push({ s: w![o]!, e: w![o + 1]! }); }
        const rate = ticks / us, T = (x: number) => x / rate;
        const start = Math.min(...recs.map(x => x.s)), end = Math.max(...recs.map(x => x.e));
        const d = recs.map(x => T(x.e - x.s)).sort((p, q) => p - q), ends = recs.map(x => T(x.e - start)).sort((p, q) => p - q);
        const q = (a: number[], f: number) => a[Math.min(a.length - 1, Math.floor(f * (a.length - 1)))]!;
        const bins = 16, inflight = Array.from({ length: bins }, (_, b) => { const tb = start + (b + 0.5) * (end - start) / bins; return recs.filter(x => x.s <= tb && x.e > tb).length; });
        const ev = recs.flatMap(x => [[x.s, 1], [x.e, -1]] as [number, number][]).sort((p, q) => p[0] - q[0] || p[1] - q[1]);
        let cur = 0, peak = 0; for (const [, dlt] of ev) { cur += dlt; peak = Math.max(peak, cur); }
        const avg = d.reduce((p, x) => p + x, 0) / Math.max(1e-9, T(end - start));
        runs.push({ passUs: us, rate, state, done, n, spanUs: T(end - start), avgInflight: avg, peakInflight: peak, dur: { p50: q(d, 0.5), p90: q(d, 0.9), max: d[d.length - 1]!, sum: d.reduce((p, x) => p + x, 0) }, end95Us: q(ends, 0.95), tailUs: T(end - start) - q(ends, 0.95), inflight });
        if (r === 0 && isDiag(tg)) {
          const lanes: { g: number; l: number; d: number; a: number; b: number; c: number }[] = [];
          for (let g = 0; g < Math.min(n, CLK_LANE_GROUPS); g++) for (let l = 0; l < 64; l++) { const o = CLK_LANE_BASE + CLK_LANE_WORDS * (g * 64 + l); lanes.push({ g, l, d: T(w![o]!), a: w![o + 1]!, b: w![o + 2]!, c: w![o + 3]! }); }
          writeFileSync(resolve(out, `lanes-${e.label}-${tg.i}.tsv`), "g\tl\tus\ta\tb\tc\n" + lanes.map(x => `${x.g}\t${x.l}\t${x.d.toFixed(2)}\t${x.a}\t${x.b}\t${x.c}`).join("\n"));
          const sorted = [...lanes].sort((p, q) => q.d - p.d), sum = (xs: typeof lanes, f: (x: (typeof lanes)[0]) => number) => xs.reduce((p, x) => p + f(x), 0);
          for (const [lo, hi] of [[0, 0.001], [0.001, 0.01], [0.01, 0.1], [0.1, 0.5], [0.5, 1]] as [number, number][]) {
            const xs = sorted.slice(Math.floor(lo * sorted.length), Math.max(Math.floor(lo * sorted.length) + 1, Math.floor(hi * sorted.length)));
            log(`    lanes top ${lo * 100}-${hi * 100}%: n ${xs.length} us mean ${(sum(xs, x => x.d) / xs.length).toFixed(1)} max ${xs[0]!.d.toFixed(1)}; a ${(sum(xs, x => x.a & 0xffff) / xs.length).toFixed(1)}/${(sum(xs, x => x.a >>> 16) / xs.length).toFixed(1)} b ${(sum(xs, x => x.b & 0xffff) / xs.length).toFixed(1)}/${(sum(xs, x => x.b >>> 16) / xs.length).toFixed(1)} c ${(sum(xs, x => x.c & 0xffff) / xs.length).toFixed(1)}/${(sum(xs, x => x.c >>> 16) / xs.length).toFixed(1)}`);
          }
          log(`    slowest lanes: ${sorted.slice(0, 8).map(x => `g${x.g}.${x.l} ${x.d.toFixed(0)}us a${x.a.toString(16)} b${x.b.toString(16)} c${x.c.toString(16)}`).join(", ")}`);
        }
        if (r === 0) writeFileSync(resolve(out, `clock-${e.label}-${tg.name}-${tg.i}.tsv`), "g\tstartUs\tendUs\n" + recs.map((x, g) => `${g}\t${T(x.s - start).toFixed(2)}\t${T(x.e - start).toFixed(2)}`).join("\n"));
      }
      const best = runs.reduce((p, x) => (x.passUs < p.passUs ? x : p));
      (e.clock ??= {})[tg.name] = { runs };
      const f = (x: number) => x.toFixed(1);
      log(`  clock ${String(tg.i).padStart(3)} ${e.label.padEnd(14)} ${tg.name.padEnd(8)} pass ${f(best.passUs)} us state ${best.state} ${best.done}/${best.n}; dur p50 ${f(best.dur.p50)} p90 ${f(best.dur.p90)} max ${f(best.dur.max)}; span ${f(best.spanUs)}, 95% ended ${f(best.end95Us)}, tail ${f(best.tailUs)} us; avg ${f(best.avgInflight)} peak ${best.peakInflight}; in flight ${best.inflight.join(" ")}`);
      save();
    }
    log(`clock: probe ${(await clock()).toFixed(1)} us`);
  }

  // ---- step ----
  if (modes.has("step")) {
    const pick = (process.env.STEP_VARIANTS ?? "").split(",").filter(x => x);
    const lists: Record<string, any[]> = { base: [] };
    for (const v of pick) lists[v] = [];
    const cuts: Record<string, number> = {};
    for (let i = 0; i < all.length; i++) {
      if (i === n1) for (const v of Object.keys(lists)) cuts[v] = lists[v]!.length;
      lists.base!.push(all[i]);
      for (const v of pick) {
        const keys = v.split("+"), t = targets.find(x => x.i === i && keys.includes(x.key));
        const x = t ? variants.get(`${i}\t${t.name}`) : undefined;
        if (x) lists[v]!.push(...x.pre, x.rec); else lists[v]!.push(all[i]);
      }
    }
    const wall = async (v: string) => {
      restore(); ring();
      const enc = raw.createCommandEncoder(), l = lists[v]!, cut = cuts[v]!;
      const a = enc.beginComputePass({ timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0 } }); for (const x of l.slice(0, cut)) dispatchInto(a, x); a.end();
      const b = enc.beginComputePass({ timestampWrites: { querySet: qs, endOfPassWriteIndex: 1 } }); for (const x of l.slice(cut)) dispatchInto(b, x); b.end();
      const { t } = await readBack(enc, 2); return Number(t[1]! - t[0]!) / 1e3;
    };
    const { t, probes } = await rounds(Object.keys(lists).map(v => ({ key: v, run: () => wall(v) })));
    result.step = { probes };
    for (const v of Object.keys(lists)) result.step[v] = { ...stats(t.get(v)!), samples: t.get(v) };
    const b0 = result.step.base.min;
    log(`step wall (min, delta, median): ${Object.keys(lists).map(v => `${v} ${(result.step[v].min / 1e3).toFixed(3)} (${((result.step[v].min - b0) / 1e3).toFixed(3)}) med ${(result.step[v].med / 1e3).toFixed(3)}`).join(" | ")} ms; probe ${probes.map(x => x.toFixed(1)).join(" ")}`);
    save();
  }
  save();
  for (const b of shadow) b.destroy();
  s.destroy?.();
  process.exit(gpuErrors.length ? 3 : 0);
}
