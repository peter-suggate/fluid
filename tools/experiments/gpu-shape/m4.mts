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
const CLK_HEAD = 8, CLK_REC = 4, CLK_MAX_GROUPS = 1 << 16, CLK_MAX_TICKS = 1 << 26;
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
 for(var i=g.x;i<cn.x;i+=cn.y){if(ca[i]!=cb[i]){atomicAdd(&co[0],1u);atomicMin(&co[1],i+cn.z);}}
}`;

// ---- the M4 variants ----
type Variant = { label: string; name: string; k: number; make: (code: string, entry: string) => string };
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
const VARIANTS: Variant[] = [
  { label: "censusDecide", name: "count", k: 1, make: cdCount },
  { label: "censusDecide", name: "count+trig64", k: 1, make: c => cdTrig64(cdCount(c)) },
  { label: "censusDecide", name: "diag", k: 1, make: cdDiag },
  { label: "lcRetire", name: "diag", k: 1, make: lrDiag },
  { label: "lcRetire", name: "gcache", k: 1, make: lcGcache },
  { label: "lcRetire", name: "gfetch", k: 1, make: lcGfetch },
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
    for (const v of VARIANTS) if (v.label === k0.label && (!only || only.has(`${v.label}.${v.name}`))) targets.push({ i, key: `${v.label}.${v.name}`, name: v.name, code: () => v.make(k0.code, k0.entry), a: w => argsOf(i, v.k, w) });
    if (ref && ref.dispatch[k0.label] && existsSync(resolve(ref.dir, `${k0.label}.wgsl`)) && (!only || only.has(`${k0.label}.ref`))) {
      const d = ref.dispatch[k0.label]; if (!("indirect" in d)) continue;
      const text = readFileSync(resolve(ref.dir, `${k0.label}.wgsl`), "utf8");
      if (text === k0.code && d.indirect === sch[i].dispatch.indirect) continue;
      targets.push({ i, key: `${k0.label}.ref`, name: "ref", code: () => text, a: w => argsRec(d.indirect, d.indirect, 0, w) });
    }
  }
  const placements = [...new Set(targets.map(t => t.i))].sort((a, b) => a - b);
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
    for (const i of placements) {
      const k = sch[i].kernel;
      restore(); ring(); { const enc = raw.createCommandEncoder(); prefixInto(enc, i); raw.queue.submit([enc.finish()]); await raw.queue.onSubmittedWorkDone(); }
      copyAll(live, snap); await raw.queue.onSubmittedWorkDone();
      const names = k.bindings.filter((b: any) => b.kind === "rw").map((b: any) => b.name as string), bufs = [...new Set(names.map(bufOf))] as GPUBuffer[];
      await fromSnap([all[i]]); { const enc = raw.createCommandEncoder(); for (const b of bufs) enc.copyBufferToBuffer(b, 0, refOf(b), 0, b.size); raw.queue.submit([enc.finish()]); }
      const ex: Record<string, { words: number; first: number }> = {};
      const cases: [string, any[]][] = [["base", [all[i]]], ...targets.filter(t => t.i === i && !isDiag(t)).map(t => { const v = variants.get(`${i}\t${t.name}`)!; return [t.name, [...v.pre, v.rec]] as [string, any[]]; })];
      for (const [v, list] of cases) {
        await fromSnap(list); let d = 0, first = -1; for (const b of bufs) { const [n, f] = await compare(b, refOf(b)); if (n && first < 0) first = f; d += n; } ex[v] = { words: d, first };
      }
      result.placements[i].rw = names; result.placements[i].exact = ex;
      log(`  exact ${i} ${k.label} (${names.join(",")}): ${Object.entries(ex).map(([v, d]) => `${v} ${d.words}`).join(", ")} words differ`);
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
      const v = await V(tg, true), e = result.placements[tg.i], runs: any[] = [];
      for (let r = 0; r < CR; r++) {
        const { us, w } = await sample(tg.i, v.rec, v.pre, true); fail(`clocking ${e.label}`);
        const n = w![2]!, ticks = w![0]!, state = w![3]!, done = w![1]!, recs: { s: number; e: number }[] = [];
        for (let g = 0; g < Math.min(n, CLK_MAX_GROUPS); g++) { const o = CLK_HEAD + CLK_REC * g; recs.push({ s: w![o]!, e: w![o + 1]! }); }
        const rate = ticks / us, T = (x: number) => x / rate;
        const start = Math.min(...recs.map(x => x.s)), end = Math.max(...recs.map(x => x.e));
        const d = recs.map(x => T(x.e - x.s)).sort((p, q) => p - q), ends = recs.map(x => T(x.e - start)).sort((p, q) => p - q);
        const q = (a: number[], f: number) => a[Math.min(a.length - 1, Math.floor(f * (a.length - 1)))]!;
        const bins = 16, inflight = Array.from({ length: bins }, (_, b) => { const tb = start + (b + 0.5) * (end - start) / bins; return recs.filter(x => x.s <= tb && x.e > tb).length; });
        runs.push({ passUs: us, rate, state, done, n, spanUs: T(end - start), dur: { p50: q(d, 0.5), p90: q(d, 0.9), max: d[d.length - 1]!, sum: d.reduce((p, x) => p + x, 0) }, end95Us: q(ends, 0.95), tailUs: T(end - start) - q(ends, 0.95), inflight });
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
      log(`  clock ${String(tg.i).padStart(3)} ${e.label.padEnd(14)} ${tg.name.padEnd(8)} pass ${f(best.passUs)} us state ${best.state} ${best.done}/${best.n}; dur p50 ${f(best.dur.p50)} p90 ${f(best.dur.p90)} max ${f(best.dur.max)}; span ${f(best.spanUs)}, 95% ended ${f(best.end95Us)}, tail ${f(best.tailUs)} us; in flight ${best.inflight.join(" ")}`);
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
