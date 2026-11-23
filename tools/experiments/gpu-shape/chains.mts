// Dependent-load critical path of a WGSL compute kernel, per lane (M0e, docs/plans/gpu-shape/m0e-chains.md).
// Offline, no GPU.
//   node --import tsx tools/experiments/gpu-shape/chains.mts <module.wgsl> <entry> [--paths=N] [--trip=LINE:N,...]
//        [--cold=LINE,...] [--loops]
//
// Model: an in-order SIMT lane with ideal scheduling inside a basic block. Time is counted in device-load latencies
// (ALU free). Every value carries the time it is ready (a load is ready one latency after both its address and the
// lane's clock); the lane's clock advances only at a branch whose arms contain device loads (the lane cannot issue past
// it until the condition is known; arms without loads are taken as selects) and at every loop iteration's condition.
// Calls are inlined (WGSL has no recursion), so a callee's branches advance the caller's clock. Both arms of a branch are
// simulated and the clock after it is the later arm (the longest path); `--cold=LINE` drops the then-arm of the if at
// LINE (a rare branch). Loops with a constant trip (for i = 0..N) run exactly (up to 512); other loops run once unless
// `--trip=LINE:N`. Stores and atomics whose result is unused do not stall; their operands' readiness bounds the end.
// The lane's critical path = max(clock at the end, operands of its last stores).
//
// A device load is an index into a storage binding, or an atomic whose result is used. A load is lane-uniform when its
// address does not depend on a lane builtin (local_invocation_*, global_invocation_id): every lane of the workgroup
// performs the same load, which a per-workgroup/simdgroup hoist would do once.
import { readFileSync } from "node:fs";

type Tok = { t: string; line: number };
type V = { d: number; path: string[]; lane: boolean; ld?: boolean }; // ld: derived from a device load
const ZERO: V = { d: 0, path: [], lane: false };
const vmax = (...vs: V[]): V => vs.reduce((a, b) => (b.d > a.d ? { ...b, lane: a.lane || b.lane, ld: a.ld || b.ld } : { ...a, lane: a.lane || b.lane, ld: a.ld || b.ld }), ZERO);

const file = process.argv[2]!, entry = process.argv[3]!;
const arg = (n: string) => process.argv.find(a => a.startsWith(`--${n}=`))?.slice(n.length + 3);
const NPATHS = Number(arg("paths") ?? 4);
const TRIPS = new Map((arg("trip") ?? "").split(",").filter(Boolean).map(s => s.split(":").map(Number) as [number, number]));
const COLD = new Set((arg("cold") ?? "").split(",").filter(Boolean).map(Number));
const SHOW_LOOPS = process.argv.includes("--loops");
// --data: address dependence only (a lane that speculates every branch: the lower bound). Default: control dependence
// too (no load issues past an unresolved branch that guards loads or exits: the upper bound).
const DATA = process.argv.includes("--data");
const MAXTRIP = Number(arg("maxtrip") ?? 64);
// The common path (default; --all-arms takes every arm of every branch whose condition is not a compile-time constant):
// branch conditions are folded where a term is a bool constant, a slot hit (x != NB_NONE, x < NB_SEAM: the lookup finds a
// resident slot), an in-range guard (…InLattice, …InRange, lVertexIn, all(c >= 0), any(c < 0) false), or a call in the
// --assume list (default: the h-owned interior of a tiered F region; `--assume=` empties it, `--assume=+f:true` adds).
// A folded condition's loads still count; only the arm not taken is dropped. Calls in --cold-fn are free.
const ALL_ARMS = process.argv.includes("--all-arms");
const ASSUME_DEFAULT = "foCellOwnerH:true,foTileOwnerH:true,fgTileH:true,fgTileHDir:true,foVertexOwnerH:true,fgVertexOwned:true,fgFaceOwned:true,fgCellOwned:true,fgCellCoarse:false,fgWinCovers:false";
const ASSUME = new Map<string, boolean>();
{ const a = arg("assume"); const list = a === undefined ? ASSUME_DEFAULT : a.startsWith("+") ? `${ASSUME_DEFAULT},${a.slice(1)}` : a;
  for (const x of list.split(",").filter(Boolean)) { const [f, v] = x.split(":"); ASSUME.set(f!, v === "true"); } }
const COLD_FN = new Set((arg("cold-fn") ?? "nbFail,nbxTraceFatal").split(",").filter(Boolean));
const INRANGE = /InLattice$|InRange$|^lVertexIn$|^nbInLattice$/;
const src = readFileSync(file, "utf8").replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, m => m.replace(/[^\n]/g, " "));

// ---- tokenizer ----
const ops = ["<<=", ">>=", "->", "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "++", "--"];
function tokenize(s: string): Tok[] {
  const out: Tok[] = []; let line = 1;
  for (let i = 0; i < s.length;) {
    const c = s[i]!;
    if (c === "\n") { line++; i++; continue; }
    if (/\s/.test(c)) { i++; continue; }
    const id = /^[A-Za-z_][A-Za-z0-9_]*/.exec(s.slice(i, i + 80));
    if (id) { out.push({ t: id[0], line }); i += id[0].length; continue; }
    const num = /^(0[xX][0-9a-fA-F]+[ui]?|(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?[uifh]?)/.exec(s.slice(i, i + 40));
    if (num && /[\d.]/.test(c)) { out.push({ t: num[0], line }); i += num[0].length; continue; }
    const op = ops.find(o => s.startsWith(o, i));
    if (op) { out.push({ t: op, line }); i += op.length; continue; }
    out.push({ t: c, line }); i++;
  }
  return out;
}
const T = tokenize(src);
const match = (i: number, open: string, close: string) => { let d = 0; for (let k = i; k < T.length; k++) { if (T[k]!.t === open) d++; else if (T[k]!.t === close && --d === 0) return k; } throw new Error(`unmatched ${open} at line ${T[i]!.line}`); };

// ---- module scan ----
const boolConsts = new Map<string, boolean>();
const storage = new Set<string>(), uniforms = new Set<string>(), moduleVars = new Map<string, V>(), consts = new Map<string, number>();
type Fn = { name: string; params: string[]; body: [number, number] };
const fns = new Map<string, Fn>();
{
  let depth = 0;
  for (let i = 0; i < T.length; i++) {
    const t = T[i]!.t;
    if (t === "{") depth++; else if (t === "}") depth--;
    if (depth !== 0) continue;
    if (t === "var" && T[i + 1]?.t === "<") {
      let close = i + 2; while (T[close]!.t !== ">") close++;
      const space = T.slice(i + 2, close).map(x => x.t).join(""), name = T[close + 1]!.t;
      if (space.startsWith("storage")) storage.add(name); else if (space.startsWith("uniform")) uniforms.add(name); else moduleVars.set(name, ZERO);
    } else if (t === "var") moduleVars.set(T[i + 1]!.t, ZERO);
    else if (t === "const") {
      const name = T[i + 1]!.t; let k = i + 2; while (T[k]!.t !== "=") k++;
      const v = T[k + 1]!.t; if (/^(0x)?[\da-f]+[ui]?$/i.test(v) && T[k + 2]?.t === ";") consts.set(name, parseInt(v.replace(/[ui]$/, ""), v.startsWith("0x") ? 16 : 10));
      if ((v === "true" || v === "false") && T[k + 2]?.t === ";") boolConsts.set(name, v === "true");
    } else if (t === "fn") {
      const name = T[i + 1]!.t; const pOpen = i + 2, pClose = match(pOpen, "(", ")");
      const params: string[] = [];
      for (let k = pOpen + 1; k < pClose; k++) if (T[k + 1]?.t === ":" && /^[A-Za-z_]/.test(T[k]!.t) && [",", "(", ")"].includes(T[k - 1]!.t)) params.push(T[k]!.t);
      let b = pClose; while (T[b]!.t !== "{") b++;
      const e = match(b, "{", "}"); fns.set(name, { name, params, body: [b, e] }); i = e;
    }
  }
}
// Which functions (transitively) perform device loads; whether a token range does.
const loadsFn = new Map<string, boolean>();
function rangeLoads(i: number, j: number): boolean {
  for (let k = i; k < j; k++) {
    const t = T[k]!.t;
    if (storage.has(t) && ![";", "{", "}"].includes(T[k - 1]!.t)) return true; // not a statement-initial store
    if (storage.has(t) && T[k - 1]!.t !== "&") { // statement-initial: a store, unless it is read (x = buf[..] handled above)
      const e = match(k + 1, "[", "]"); if (!["=", "+=", "-=", "|=", "&=", "^=", "*="].includes(T[e + 1]?.t ?? "") && T[e + 1]?.t !== ".") return true;
    }
    if (fns.has(t) && T[k + 1]?.t === "(" && fnLoads(t)) return true;
    if ((t === "atomicLoad") && T[k + 1]?.t === "(") return true;
  }
  return false;
}
// An arm that exits (return, or break/continue of the enclosing loop) gates everything after it.
function rangeExits(i: number, j: number): boolean {
  let loops = 0; const ends: number[] = [];
  for (let k = i; k < j; k++) {
    while (ends.length && k > ends[ends.length - 1]!) { ends.pop(); loops--; }
    const t = T[k]!.t;
    if (t === "return") return true;
    if ((t === "break" || t === "continue") && loops === 0) return true;
    if (t === "for" || t === "while" || t === "loop") { let b = k + 1; while (T[b]!.t !== "{") b++; ends.push(match(b, "{", "}")); loops++; }
  }
  return false;
}
function fnLoads(n: string): boolean { if (!loadsFn.has(n)) { loadsFn.set(n, false); const f = fns.get(n)!; loadsFn.set(n, rangeLoads(f.body[0] + 1, f.body[1])); } return loadsFn.get(n)!; }

// ---- statistics ----
type Load = { d: number; path: string[]; buf: string; fn: string; line: number; lane: boolean };
class Stats {
  total = 0; uni = 0; uniMax = 0; byBuf = new Map<string, number>(); byFn = new Map<string, number>(); uniByFn = new Map<string, number>(); top: Load[] = [];
  add(l: Load) {
    this.total++; this.byBuf.set(l.buf, (this.byBuf.get(l.buf) ?? 0) + 1); this.byFn.set(l.fn, (this.byFn.get(l.fn) ?? 0) + 1);
    if (!l.lane) { this.uni++; this.uniMax = Math.max(this.uniMax, l.d); this.uniByFn.set(l.fn, (this.uniByFn.get(l.fn) ?? 0) + 1); }
    this.keep(l);
  }
  keep(l: Load) {
    if (this.top.length >= 12 && l.d <= this.top[this.top.length - 1]!.d) return;
    const key = l.path.join(">"); if (this.top.some(x => x.path.join(">") === key)) return;
    this.top.push(l); this.top.sort((a, b) => b.d - a.d); if (this.top.length > 12) this.top.pop();
  }
  mergeShift(o: Stats, dd: number, pre: string[], cut: number) {
    if (dd === 0) { this.merge(o); return; }
    this.total += o.total; this.uni += o.uni; if (o.uni) this.uniMax = Math.max(this.uniMax, o.uniMax + dd);
    for (const [b, n] of o.byBuf) this.byBuf.set(b, (this.byBuf.get(b) ?? 0) + n);
    for (const [b, n] of o.byFn) this.byFn.set(b, (this.byFn.get(b) ?? 0) + n);
    for (const [b, n] of o.uniByFn) this.uniByFn.set(b, (this.uniByFn.get(b) ?? 0) + n);
    for (const l of o.top) this.keep({ ...l, d: l.d + dd, path: [...pre, ...l.path.slice(cut)] });
  }
  merge(o: Stats) {
    this.total += o.total; this.uni += o.uni; this.uniMax = Math.max(this.uniMax, o.uniMax);
    for (const [b, n] of o.byBuf) this.byBuf.set(b, (this.byBuf.get(b) ?? 0) + n);
    for (const [b, n] of o.byFn) this.byFn.set(b, (this.byFn.get(b) ?? 0) + n);
    for (const [b, n] of o.uniByFn) this.uniByFn.set(b, (this.uniByFn.get(b) ?? 0) + n);
    for (const l of o.top) this.keep(l);
  }
}
let stats = new Stats();
let clk: V = ZERO;          // the lane's clock
let tail: V = ZERO;         // readiness bound from stores / unused atomics
const loopLog: { fn: string; line: number; trip: string; iters: number; adv: number; per: number }[] = [];

// ---- interpreter ----
const LANE_BUILTINS = new Set(["local_invocation_index", "local_invocation_id", "global_invocation_id"]);
const TEMPLATED = new Set(["vec2", "vec3", "vec4", "array", "bitcast", "atomic", "ptr", "mat2x2", "mat3x3", "mat4x4", "mat3x4", "mat4x3", "mat2x3", "mat3x2"]);
const ATOMIC_RMW = new Set(["atomicAdd", "atomicSub", "atomicMax", "atomicMin", "atomicOr", "atomicAnd", "atomicXor", "atomicExchange", "atomicCompareExchangeWeak"]);
const ASSIGN = ["=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="];
class Ctx {
  scopes: Map<string, V>[] = [new Map()]; ret: V = ZERO; sel: V[] = []; flow: Flow = "";
  constructor(public fn: string, public known = new Map<string, string>()) {}
  get(n: string): V { for (let s = this.scopes.length - 1; s >= 0; s--) { const v = this.scopes[s]!.get(n); if (v) return v; } return moduleVars.get(n) ?? ZERO; }
  set(n: string, v: V, merge: boolean) {
    const w = vmax(v, ...this.sel);
    for (let s = this.scopes.length - 1; s >= 0; s--) if (this.scopes[s]!.has(n)) { this.scopes[s]!.set(n, merge ? vmax(this.scopes[s]!.get(n)!, w) : w); return; }
    if (moduleVars.has(n)) { moduleVars.set(n, vmax(moduleVars.get(n)!, w)); return; }
    this.scopes[this.scopes.length - 1]!.set(n, w);
  }
  decl(n: string, v: V) { this.scopes[this.scopes.length - 1]!.set(n, vmax(v, ...this.sel)); }
}
function skipTemplate(i: number): number {
  let d = 0;
  for (let k = i; k < T.length; k++) { const t = T[k]!.t; if (t === "<") d++; else if (t === ">") { if (--d === 0) return k + 1; } else if (t === ">>") { d -= 2; if (d <= 0) return k + 1; } else if (t === ">=") { if (--d === 0) return k; } }
  throw new Error("template");
}
function splitArgs(i: number, j: number, sep = ","): [number, number][] {
  const out: [number, number][] = []; let d = 0, s = i;
  for (let k = i; k < j; k++) { const t = T[k]!.t; if (t === "(" || t === "[" || t === "{") d++; else if (t === ")" || t === "]" || t === "}") d--; else if (t === sep && d === 0) { out.push([s, k]); s = k + 1; } }
  if (s < j || sep === ";") out.push([s, j]);
  return out;
}
function load(cx: Ctx, buf: string, idx: V, line: number): V {
  const base = vmax(idx, clk);
  const v: V = { d: base.d + 1, path: [...base.path, `${cx.fn}:${buf}`], lane: idx.lane };
  stats.add({ d: v.d, path: v.path, buf, fn: cx.fn, line, lane: idx.lane });
  return { ...v, lane: idx.lane || clk.lane, ld: true };
}
function expr(cx: Ctx, i: number, j: number): V {
  let acc = ZERO;
  for (let k = i; k < j; k++) {
    const t = T[k]!.t;
    if (t === "(") { const e = match(k, "(", ")"); acc = vmax(acc, expr(cx, k + 1, e)); k = e; continue; }
    if (t === "[") { const e = match(k, "[", "]"); acc = vmax(acc, expr(cx, k + 1, e)); k = e; continue; }
    if (!/^[A-Za-z_]/.test(t)) continue;
    let n = k + 1;
    if (T[n]?.t === "<" && TEMPLATED.has(t)) n = skipTemplate(n);
    if (T[n]?.t === "(") {
      const e = match(n, "(", ")"); const args = splitArgs(n + 1, e);
      if (COLD_FN.has(t)) { /* rare: free */ }
      else if (fns.has(t)) acc = vmax(acc, call(cx, t, args.map(([a, b]) => expr(cx, a, b)), args.map(([a, b]) => constText(cx, a, b))));
      else if (t === "arrayLength") { /* free */ }
      else if (ATOMIC_RMW.has(t) || t === "atomicLoad") {
        const [a0, ...rest] = args; const restV = rest.map(([a, b]) => expr(cx, a, b));
        const { buf, idx } = storageRef(cx, a0![0], a0![1]);
        acc = vmax(acc, load(cx, t === "atomicLoad" ? buf : `${buf}(${t})`, vmax(idx, ...restV), T[k]!.line));
      } else acc = vmax(acc, ...args.map(([a, b]) => expr(cx, a, b)));
      k = e; continue;
    }
    if (T[n]?.t === "[" && storage.has(t)) { const e = match(n, "[", "]"); acc = vmax(acc, load(cx, t, expr(cx, n + 1, e), T[k]!.line)); k = e; continue; }
    if (T[k - 1]?.t === ".") continue;
    if (uniforms.has(t) || storage.has(t)) continue;
    acc = vmax(acc, cx.get(t));
  }
  return acc;
}
function storageRef(cx: Ctx, i: number, j: number): { buf: string; idx: V } {
  let k = i; while (k < j && !storage.has(T[k]!.t)) k++;
  if (k >= j) return { buf: "?", idx: expr(cx, i, j) };
  if (T[k + 1]?.t !== "[") return { buf: T[k]!.t, idx: ZERO };
  const e = match(k + 1, "[", "]"); return { buf: T[k]!.t, idx: expr(cx, k + 2, e) };
}
// Calls are memoised shift-invariantly: every consumer of a value maxes it with the lane clock, which never falls below
// the clock at the call, so inputs are floored to that clock and the callee's results are stored relative to it; a hit
// at a later clock replays them shifted (witness paths keep the callee's suffix after the current clock's path).
type Memo = { base: number; cut: number; ret: V; st: Stats; clk: V; tail: V; mv: Map<string, V>; loops: typeof loopLog };
const memo = new Map<string, Memo>();
function shiftV(v: V, d: number, prefix: string[], cut: number): V { return { d: v.d + d, path: [...prefix, ...v.path.slice(cut)], lane: v.lane, ld: v.ld }; }
// The compile-time constant an argument is (a named const, a literal, or a known parameter), or "".
function constText(cx: Ctx, a: number, b: number): string {
  if (b - a !== 1) return "";
  const t = T[a]!.t;
  if (/^\d/.test(t)) return String(parseInt(t));
  if (consts.has(t)) return String(consts.get(t)); if (boolConsts.has(t)) return String(boolConsts.get(t));
  if (/^[A-Z][A-Z0-9_]+$/.test(t)) return t;
  return cx.known.get(t) ?? "";
}
function call(cx: Ctx, name: string, args: V[], argText: string[] = []): V {
  const f = fns.get(name)!;
  const base = clk.d; const rel = (v: V) => `${Math.max(v.d, base) - base}${v.lane ? "L" : ""}${v.ld ? "M" : ""}`;
  const sel = vmax(...cx.sel);
  const key = `${name}|${args.map(rel).join(",")}|${argText.join(",")}|${clk.lane ? "L" : ""}|${rel(sel)}|${[...moduleVars.values()].map(rel).join(",")}`;
  const floor = (v: V): V => (v.d >= base ? v : { d: clk.d, path: clk.path, lane: v.lane, ld: v.ld });
  const hit = memo.get(key);
  if (hit) {
    const dd = base - hit.base, pre = clk.path, cut = hit.cut;
    stats.mergeShift(hit.st, dd, pre, cut);
    clk = vmax(clk, shiftV(hit.clk, dd, pre, cut)); tail = vmax(tail, shiftV(hit.tail, dd, pre, cut));
    for (const [n, v] of hit.mv) moduleVars.set(n, shiftV(v, dd, pre, cut));
    loopLog.push(...hit.loops);
    return shiftV(hit.ret, dd, pre, cut);
  }
  const outer = stats; stats = new Stats(); const tailBefore = tail; tail = { ...clk, lane: false }; const loopsBefore = loopLog.length;
  const mvBefore = new Map(moduleVars); for (const [n, v] of moduleVars) moduleVars.set(n, floor(v));
  const c2 = new Ctx(name); if (sel.d > 0) c2.sel.push(floor(sel));
  f.params.forEach((p, i) => { if (argText[i]) c2.known.set(p, argText[i]!); });
  f.params.forEach((p, i) => c2.decl(p, floor(args[i] ?? ZERO)));
  const cut = clk.path.length;
  block(c2, f.body[0] + 1, f.body[1]);
  const mv = new Map<string, V>(); for (const [n, v] of moduleVars) if (v !== mvBefore.get(n)) mv.set(n, floor(v));
  const ret = floor(vmax(c2.ret, ...c2.sel));
  memo.set(key, { base, cut, ret, st: stats, clk, tail, mv, loops: loopLog.slice(loopsBefore) });
  outer.merge(stats); stats = outer; tail = vmax(tail, tailBefore);
  for (const [n, v] of mv) moduleVars.set(n, v);
  return ret;
}
function stmtEnd(k: number, j: number) { let e = k, d = 0; while (e < j && !(T[e]!.t === ";" && d === 0)) { const tt = T[e]!.t; if (tt === "(" || tt === "[" || tt === "{") d++; else if (tt === ")" || tt === "]" || tt === "}") d--; e++; } return e; }

// Fold a condition token range: true / false / undefined (unknown).
let foldCx: Ctx | null = null;
function fold(cx: Ctx, i: number, j: number): boolean | undefined { foldCx = cx; return foldTerm(i, j, !ALL_ARMS); }
function splitTop(i: number, j: number, op: string): [number, number][] {
  const out: [number, number][] = []; let d = 0, s = i;
  for (let k = i; k < j; k++) { const t = T[k]!.t; if (t === "(" || t === "[") d++; else if (t === ")" || t === "]") d--; else if (t === op && d === 0) { out.push([s, k]); s = k + 1; } }
  out.push([s, j]); return out;
}
function foldTerm(i: number, j: number, common: boolean): boolean | undefined {
  while (T[i]?.t === "(" && match(i, "(", ")") === j - 1) { i++; j--; }
  const ors = splitTop(i, j, "||");
  if (ors.length > 1) { const v = ors.map(([a, b]) => foldTerm(a, b, common)); if (v.includes(true)) return true; if (v.every(x => x === false)) return false; return undefined; }
  const ands = splitTop(i, j, "&&");
  if (ands.length > 1) { const v = ands.map(([a, b]) => foldTerm(a, b, common)); if (v.includes(false)) return false; if (v.every(x => x === true)) return true; return undefined; }
  if (T[i]!.t === "!") { const v = foldTerm(i + 1, j, common); return v === undefined ? undefined : !v; }
  const toks = T.slice(i, j).map(x => x.t);
  if (toks.length === 1 && boolConsts.has(toks[0]!)) return boolConsts.get(toks[0]!);
  if (toks.length === 1 && foldCx?.known.get(toks[0]!) === "true") return true;
  if (toks.length === 1 && foldCx?.known.get(toks[0]!) === "false") return false;
  // a == / != b between compile-time constants (named consts, literals, parameters bound to constants)
  if (toks.length === 3 && (toks[1] === "==" || toks[1] === "!=")) {
    const val = (x: string) => /^\d/.test(x) ? String(parseInt(x)) : consts.has(x) ? String(consts.get(x)) : foldCx?.known.get(x) ?? (/^[A-Z][A-Z0-9_]+$/.test(x) ? x : "");
    const x = val(toks[0]!), y = val(toks[2]!);
    if (x && y && (/^\d+$/.test(x) === /^\d+$/.test(y))) return (x === y) === (toks[1] === "==");
  }
  if (!common) return undefined;
  const s = toks.join("");
  // slot hits: only for a value that came from a load (a constant NB_NONE, e.g. a disabled ring, stays unknown)
  const loaded = foldCx ? !!foldCx.get(toks[0]!).ld : false;
  if (loaded && /^[A-Za-z_]\w*(\.\w+)?(!=NB_NONE|<NB_SEAM)$/.test(s)) return true;
  if (loaded && /^[A-Za-z_]\w*(\.\w+)?(==NB_NONE|>=NB_SEAM)$/.test(s)) return false;
  if (/^any\(.*<vec3i\(0\)\)$/.test(s)) return false;
  if (/^all\(.*>=vec3i\(0\)\)$/.test(s)) return true;
  if (/^[A-Za-z_]\w*$/.test(toks[0]!) && toks[1] === "(" && match(i + 1, "(", ")") === j - 1) {
    if (ASSUME.has(toks[0]!)) return ASSUME.get(toks[0]!);
    if (INRANGE.test(toks[0]!)) return true;
  }
  return undefined;
}
type Flow = "" | "return" | "break" | "continue";
function block(cx: Ctx, i: number, j: number): void {
  let k = i;
  while (k < j && !cx.flow) {
    const t = T[k]!.t;
    if (t === ";") { k++; continue; }
    if (t === "{") { const e = match(k, "{", "}"); cx.scopes.push(new Map()); block(cx, k + 1, e); cx.scopes.pop(); k = e + 1; continue; }
    if (t === "let" || t === "var" || t === "const") {
      let n = k + 1; if (T[n]!.t === "<") n = skipTemplate(n);
      const name = T[n]!.t; let e = n + 1, init = -1;
      while (e < j) { const tt = T[e]!.t; if (tt === "<") { e = skipTemplate(e); if (T[e]!.t === "=") { init = e + 1; break; } if (T[e - 1]!.t === ">=") { init = e; break; } continue; } if (tt === "=") { init = e + 1; break; } if (tt === ";") break; e++; }
      const end = stmtEnd(init >= 0 ? init : e, j);
      cx.decl(name, init >= 0 ? expr(cx, init, end) : ZERO);
      k = end + 1; continue;
    }
    if (t === "return") { const e = stmtEnd(k, j); cx.ret = vmax(cx.ret, expr(cx, k + 1, e), ...cx.sel); cx.flow = "return"; k = e + 1; continue; }
    if (t === "break" || t === "continue" || t === "discard") {
      const e = stmtEnd(k, j);
      if (T[k + 1]?.t === "if") { const c = expr(cx, k + 2, e); if (!DATA) clk = vmax(clk, c); const f = fold(cx, k + 2, e); if (f === true) cx.flow = t === "continue" ? "continue" : "break"; }
      else cx.flow = t === "continue" ? "continue" : t === "discard" ? "return" : "break";
      k = e + 1; continue;
    }
    if (t === "if") {
      const arms: { cond: [number, number] | null; body: [number, number]; line: number }[] = [];
      let p = k;
      while (true) {
        let b = p + 1; while (T[b]!.t !== "{") b++;
        const e = match(b, "{", "}"); arms.push({ cond: [p + 1, b], body: [b + 1, e], line: T[p]!.line });
        p = e + 1;
        if (T[p]?.t === "else" && T[p + 1]?.t === "if") { p = p + 1; continue; }
        if (T[p]?.t === "else") { const e2 = match(p + 1, "{", "}"); arms.push({ cond: null, body: [p + 2, e2], line: T[p]!.line }); p = e2 + 1; }
        break;
      }
      // Which arms run: walk the chain, folding conditions; a known-true arm ends it, known-false arms are skipped.
      const run: typeof arms = []; let covered = false;
      let condAcc = ZERO;
      for (const a of arms) {
        if (!a.cond) { run.push(a); covered = true; break; }
        condAcc = vmax(condAcc, expr(cx, a.cond[0], a.cond[1]));
        if (COLD.has(a.line)) continue; // rare branch: not taken
        const f = fold(cx, a.cond[0], a.cond[1]);
        if (f === false) continue;
        run.push(a); if (f === true) { covered = true; break; }
      }
      const single = covered && run.length === 1;
      const branchy = !DATA && run.some(a => rangeLoads(a.body[0], a.body[1]) || rangeExits(a.body[0], a.body[1]));
      const clk0 = branchy ? vmax(clk, condAcc) : clk; let clkOut = clk0;
      const flows: Flow[] = [];
      for (const a of run) {
        clk = clk0; if (!branchy && !single) cx.sel.push(condAcc);
        cx.scopes.push(new Map()); block(cx, a.body[0], a.body[1]); cx.scopes.pop();
        if (!branchy && !single) cx.sel.pop();
        flows.push(cx.flow); cx.flow = "";
        clkOut = vmax(clkOut, clk);
      }
      clk = clkOut;
      // After the if: an exit only when every path exits (the arms run cover all cases).
      if (covered && flows.length && flows.every(f => f)) cx.flow = flows.includes("continue") || flows.includes("break") ? (flows.every(f => f === "return") ? "return" : flows.includes("break") ? "break" : "continue") : "return";
      else if (!DATA && flows.some(f => f)) clk = vmax(clk, condAcc); // a lane that exits on a load-dependent condition gates the rest
      k = p; continue;
    }
    if (t === "switch") {
      let b = k + 1; while (T[b]!.t !== "{") b++; const cond = expr(cx, k + 1, b); const e = match(b, "{", "}");
      const clk0 = vmax(clk, cond); let out = clk0;
      for (let q = b + 1; q < e; q++) if (T[q]!.t === "{") { const qe = match(q, "{", "}"); clk = clk0; cx.scopes.push(new Map()); block(cx, q + 1, qe); cx.scopes.pop(); out = vmax(out, clk); q = qe; }
      clk = out; k = e + 1; continue;
    }
    if (t === "for" || t === "while" || t === "loop") {
      let b = k + 1; while (T[b]!.t !== "{") b++; const e = match(b, "{", "}");
      cx.scopes.push(new Map());
      let condR: [number, number] | null = null, updR: [number, number] | null = null, trip = -1, label = "loop";
      if (t === "for") {
        const pc = match(k + 1, "(", ")"); const parts = splitArgs(k + 2, pc, ";");
        block(cx, parts[0]![0], parts[0]![1] + 1); condR = parts[1]!; updR = parts[2]!;
        const it = T.slice(parts[0]![0], parts[0]![1]).map(x => x.t).join(" "), ct = T.slice(condR[0], condR[1]).map(x => x.t);
        const iv = /^var (\w+) (?:: \w+ )?= 0[ui]?$/.exec(it);
        if (iv && ct[0] === iv[1] && (ct[1] === "<" || ct[1] === "<=") && ct.length === 3) {
          const lim = /^\d+[ui]?$/.test(ct[2]!) ? parseInt(ct[2]!) : consts.get(ct[2]!) ?? -1; if (lim > 0) trip = lim + (ct[1] === "<=" ? 1 : 0);
        }
        // a workgroup-strided loop: for (var j = lid; j < N; j += S): ceil(N / S) iterations for lane 0
        const sv = /^var (\w+) (?:: \w+ )?= (\w+)$/.exec(it), ut = T.slice(updR[0], updR[1]).map(x => x.t);
        if (trip < 0 && sv && ct[0] === sv[1] && ct[1] === "<" && ct.length === 3 && ut[0] === sv[1] && ut[1] === "+=" && ut.length === 3) {
          const num = (x: string) => /^\d+[ui]?$/.test(x) ? parseInt(x) : consts.get(x) ?? -1;
          const lim = num(ct[2]!), st = num(ut[2]!); if (lim > 0 && st > 0) trip = Math.ceil(lim / st);
        }
        label = `for(${T.slice(parts[0]![0], pc).map(x => x.t).join("")})`;
      } else if (t === "while") { condR = [k + 1, b]; label = "while " + T.slice(k + 1, b).map(x => x.t).join(""); }
      const line = T[k]!.line;
      const iters = TRIPS.get(line) ?? (trip > 0 ? Math.min(trip, MAXTRIP) : 1);
      const branchy = !DATA && (rangeLoads(b + 1, e) || rangeExits(b + 1, e) || (condR ? rangeLoads(condR[0], condR[1]) : false));
      const c0 = clk.d; let c1 = c0;
      let ran = 0;
      for (let r = 0; r < iters; r++) {
        if (condR) { const cv = expr(cx, condR[0], condR[1]); if (branchy) clk = vmax(clk, cv); }
        cx.scopes.push(new Map()); block(cx, b + 1, e); cx.scopes.pop(); ran++;
        const f = cx.flow as Flow; if (f === "continue") cx.flow = "";
        if (f === "break") { cx.flow = ""; break; }
        if (f === "return") break;
        if (updR) block(cx, updR[0], updR[1] + 1);
        if (r === 0) c1 = clk.d;
      }
      loopLog.push({ fn: cx.fn, line, trip: trip > 0 ? String(trip) + (iters < trip ? " (CAPPED)" : "") : label.slice(0, 60), iters: ran, adv: clk.d - c0, per: iters > 1 ? (clk.d - c1) / (iters - 1) : clk.d - c0 });
      cx.scopes.pop(); k = e + 1; continue;
    }
    if (t === "continuing") { const e = match(k + 1, "{", "}"); block(cx, k + 2, e); if ((cx.flow as Flow) === "continue") cx.flow = ""; k = e + 1; continue; }
    if (t === "workgroupBarrier" || t === "storageBarrier" || t === "textureBarrier") { const e = stmtEnd(k, j); k = e + 1; continue; }
    const e = stmtEnd(k, j);
    let asg = -1; { let d = 0; for (let q = k; q < e; q++) { const tt = T[q]!.t; if (tt === "(" || tt === "[" || tt === "{") d++; else if (tt === ")" || tt === "]" || tt === "}") d--; else if (d === 0 && ASSIGN.includes(tt)) { asg = q; break; } } }
    if (asg >= 0) {
      const rhs = expr(cx, asg + 1, e);
      const root = T[k]!.t === "*" ? T[k + 1]!.t : T[k]!.t;
      if (storage.has(root)) { let idx = ZERO; if (T[k + 1]?.t === "[") idx = expr(cx, k + 2, match(k + 1, "[", "]")); tail = vmax(tail, rhs, idx, clk); }
      else if (root !== "_") { let idx = ZERO; if (T[k + 1]?.t === "[") idx = expr(cx, k + 2, match(k + 1, "[", "]")); cx.set(root, vmax(rhs, idx), T[asg]!.t !== "=" || T[k + 1]?.t === "[" || T[k + 1]?.t === "."); }
    } else if (T[e - 1]?.t === "++" || T[e - 1]?.t === "--") { cx.set(T[k]!.t, cx.get(T[k]!.t), false); }
    else if (ATOMIC_RMW.has(T[k]!.t) || T[k]!.t === "atomicStore") {
      const pOpen = k + 1, pClose = match(pOpen, "(", ")"); const args = splitArgs(pOpen + 1, pClose);
      const r = storageRef(cx, args[0]![0], args[0]![1]); tail = vmax(tail, r.idx, clk, ...args.slice(1).map(([a, b]) => expr(cx, a, b)));
    } else tail = vmax(tail, expr(cx, k, e), clk);
    k = e + 1;
  }
}

// ---- run ----
const ef = fns.get(entry); if (!ef) throw new Error(`no entry ${entry}`);
const top = new Ctx(entry);
{ let k = ef.body[0]; while (!(T[k]!.t === "fn" && T[k + 1]!.t === entry)) k--; const pc = match(k + 2, "(", ")");
  for (let q = k + 3; q < pc; q++) if (T[q]!.t === "builtin") top.decl(T[q + 4]!.t, { ...ZERO, lane: LANE_BUILTINS.has(T[q + 2]!.t) });
  // A helper as the entry: its parameters are per-lane inputs ready at time 0.
  if (T[k - 1]?.t !== ")" ) for (const p of ef.params) if (!top.scopes[0]!.has(p)) top.decl(p, { ...ZERO, lane: true }); }
block(top, ef.body[0] + 1, ef.body[1]);
tail = vmax(tail, top.ret);
const end = vmax(clk, tail);

const compress = (p: string[]) => { const out: string[] = []; for (const s of p) { const last = out[out.length - 1]; const m = last && /^(.*) x(\d+)$/.exec(last); if (last === s) out[out.length - 1] = `${s} x2`; else if (m && m[1] === s) out[out.length - 1] = `${s} x${+m[2]! + 1}`; else out.push(s); } return out.join(" > "); };
console.log(`# ${entry} (${file.split("/").pop()}${DATA ? ", data dependence only" : ""}${COLD.size ? `, cold ${[...COLD].join(",")}` : ""}${TRIPS.size ? `, trips ${[...TRIPS].map(x => x.join(":")).join(",")}` : ""})`);
console.log(`critical path ${end.d} load latencies (lane clock ${clk.d}, store tail ${tail.d}); ${stats.total} device loads; MLP ${(stats.total / Math.max(1, end.d)).toFixed(1)}`);
console.log(`  path: ${compress(end.path)}`);
console.log(`lane-uniform loads ${stats.uni} of ${stats.total} (deepest ${stats.uniMax}); by function: ${[...stats.uniByFn].sort((a, b) => b[1] - a[1]).slice(0, 10).map(([b, n]) => `${b} ${n}`).join(", ")}`);
console.log(`loads by binding: ${[...stats.byBuf].sort((a, b) => b[1] - a[1]).map(([b, n]) => `${b} ${n}`).join(", ")}`);
console.log(`loads by function: ${[...stats.byFn].sort((a, b) => b[1] - a[1]).slice(0, 18).map(([b, n]) => `${b} ${n}`).join(", ")}`);
for (const l of stats.top.slice(0, NPATHS)) console.log(`  deepest load ${l.d}${l.lane ? "" : " (uniform)"}: ${compress(l.path)}`);
const agg = new Map<string, { fn: string; line: number; trip: string; iters: number; adv: number; per: number; n: number }>();
for (const l of loopLog) { const key = `${l.fn}:${l.line}`; const o = agg.get(key); if (!o) agg.set(key, { ...l, n: 1 }); else { o.n++; o.adv = Math.max(o.adv, l.adv); o.per = Math.max(o.per, l.per); } }
const loopsShown = [...agg.values()].filter(l => SHOW_LOOPS || l.adv > 0).sort((a, b) => b.adv - a.adv);
if (loopsShown.length) { console.log(`loops that advance the lane clock (line: trip, iterations run, max clock advance, per iteration, times entered):`); for (const l of loopsShown.slice(0, 16)) console.log(`  ${l.fn} line ${l.line}: trip ${l.trip}, ran ${l.iters}, +${l.adv}, ${l.per.toFixed(1)}/iter, x${l.n}`); }
