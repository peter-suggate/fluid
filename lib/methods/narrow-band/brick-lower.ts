/**
 * The dynamic vector-write lowering: a diagnostic switch at the brick engine's compile points (brick-engine.ts,
 * brick-present.ts), off by default. With NB_BRICK_LOWER=1 in the environment, or the engine option
 * `lowerDynamicWrites`, every module's dynamic vector-component writes are lowered to whole-vector selects and
 * checked before its shader is created: a one-flag A/B when a result looks miscompiled. Off, every module is created
 * from its text exactly as written.
 *
 * Why. Metal (Dawn) once miscompiled `v[axis] op= x` in a loop with heavy calls (couple's face-area loop: −6.71 N·s
 * against +10.84; bodies-shapes.wgsl.ts pbE, m5-bodies-design §8.2), on the morning the GPU hung and restarted. The
 * verbatim loop gave the right impulse on 2026-11-17 (10.8421 N·s, as the select form), and the pass costs the dam
 * 0.6–0.9 ms a step, so it is not on by default (the coordinator's ruling). Writing new code without dynamic vector
 * writes is guidance, not a requirement.
 *
 * The lowering (exact: select is componentwise, so every other component keeps its bits and the written one gets
 * exactly `v[i] op x`; the splat's other lanes are discarded):
 * - `v[i] op= x` becomes `v=select(v,v op vecN(x),nbDynAxisN(u32(i)))` (op one of + - * / % & | ^ << >>),
 *   `v[i] = x` becomes `v=select(v,vecN(x),...)`, and `v[i]++` / `v[i]--` add or subtract vecN(1);
 * - a literal index (`v[0]`) is a static swizzle and is left alone, as is every array (its nearest declaration
 *   names `array<` or `atomic<`), a storage or workgroup buffer included.
 * N is the width of the target's nearest declaration in its function (a `var`/`let`, a parameter, a pointer's
 * pointee, the field's type in its base's struct, a called function's return type, or what it is copied from);
 * closed blocks and other functions' locals are not visible. An undeclared target is taken as vec3: a wrong guess is
 * a type error at shader creation, never a silent change. With the switch on, a module that still holds such a write
 * after lowering fails at engine creation (brickLowerChecked).
 */

const OPS = ["<<", ">>", "+", "-", "*", "/", "%", "&", "|", "^", ""] as const;

function closeAt(s: string, from: number, open: string, close: string): number {
  let depth = 0;
  for (let i = from; i < s.length; i++) { if (s[i] === open) depth++; else if (s[i] === close) { depth--; if (depth === 0) return i; } }
  return -1;
}

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** One module's memo (the module scope a function sees, its structs): the lowering asks per write. */
interface Memo { code: string; scope: Map<number, string>; structs?: Map<string, Map<string, string>> }
let memo: Memo | undefined;
const memoFor = (code: string): Memo => (memo?.code === code ? memo : (memo = { code, scope: new Map() }));
const moduleScope = (code: string, fnStart: number) => {
  const m = memoFor(code); let v = m.scope.get(fnStart);
  if (v === undefined) m.scope.set(fnStart, v = visible(code, 0, fnStart));
  return v;
};

/** The text between `from` and `pos` with every block that closes before `pos` blanked (positions kept): what a
 * declaration at `pos` can see. From 0, every function and struct body is blanked (module scope). */
function visible(code: string, from: number, pos: number): string {
  const out = code.slice(from, pos).split(""), open: number[] = [];
  for (let i = 0; i < out.length; i++) {
    if (out[i] === "{") open.push(i);
    else if (out[i] === "}") { const o = open.pop(); if (o !== undefined) out.fill(" ", o, i + 1); }
  }
  return out.join("");
}

/** A name's nearest visible declaration before `pos`: a var/let/const in its function (closed blocks excluded), then
 * its function's parameters, then the module's own scope. A call's argument is not a declaration. */
function declaration(code: string, name: string, pos: number): { type?: string; init?: string; at: number } | undefined {
  const n = escape(name), type = "((?:[^=;,(){}<>]|<[^;{}]*?>)+)";
  const local = new RegExp(`\\b(?:var|let|const|override)(?:<[^>]*>)?\\s+${n}\\s*(?::\\s*${type})?\\s*(?:=\\s*([^;]+))?;`, "g");
  const param = new RegExp(`[(,]\\s*(?:@\\w+(?:\\([^)]*\\))?\\s*)*${n}\\s*:\\s*${type}\\s*[,)]`, "g");
  const last = (re: RegExp, text: string, base: number) => { const m = [...text.matchAll(re)].pop(); return m && { type: m[1]?.trim(), init: m[2]?.trim(), at: base + m.index! }; };
  const fns = [...code.slice(0, pos).matchAll(/\bfn\s+\w+\s*\(/g)], fnStart = fns.length ? fns[fns.length - 1]!.index! : -1;
  if (fnStart >= 0) {
    const open = fnStart + fns[fns.length - 1]![0].length - 1, close = closeAt(code, open, "(", ")");
    if (close > 0 && close < pos) {
      return last(local, visible(code, close, pos), close) ?? last(param, code.slice(open, close + 1), open) ?? last(local, moduleScope(code, fnStart), 0);
    }
  }
  return last(local, visible(code, 0, pos), 0);
}

/** Each struct's fields and their type text. */
function structs(code: string): Map<string, Map<string, string>> {
  const memoed = memoFor(code); if (memoed.structs) return memoed.structs;
  const out = new Map<string, Map<string, string>>();
  for (const m of code.matchAll(/\bstruct\s+(\w+)\s*\{([^}]*)\}/g)) {
    out.set(m[1]!, new Map([...m[2]!.matchAll(/(\w+)\s*:\s*((?:[^,<>]|<(?:[^<>]|<[^<>]*>)*>)+)/g)].map(f => [f[1]!, f[2]!.trim()])));
  }
  return (memoed.structs = out);
}

type Kind = { array: true } | { array: false; width: number };

function kindOf(type: string): Kind | undefined {
  if (/^(array|atomic)\b/.test(type)) return { array: true };
  const v = type.match(/^vec([234])/); return v ? { array: false, width: Number(v[1]) } : undefined;
}

/** A type text's pointee (ptr<space,T[,access]>) or itself. */
const pointee = (type: string) => type.match(/^ptr\s*<\s*\w+\s*,\s*(.*?)(?:\s*,\s*(?:read|write|read_write))?\s*>$/)?.[1]?.trim() ?? type;

/** A function's declared return type text, or undefined. */
const fnReturn = (code: string, name: string) => code.match(new RegExp(`\\bfn\\s+${escape(name)}\\s*\\([^{]*?->\\s*([^{]+?)\\s*\\{`))?.[1];

/** The struct type a name holds: its declared type (a pointer's pointee), the constructor or function it is
 * initialised from, or that of what it is copied from. */
function baseStruct(code: string, all: Map<string, Map<string, string>>, name: string, pos: number, depth = 0): string | undefined {
  const d = declaration(code, name, pos);
  if (!d) return undefined;
  if (d.type) return pointee(d.type);
  const call = d.init?.match(/^(\w+)\s*\(/)?.[1];
  if (call && all.has(call)) return call;
  const r = call && fnReturn(code, call); if (r && all.has(r)) return r;
  const id = d.init?.match(/^\(?\*?([A-Za-z_]\w*)\)?$/)?.[1];
  return id && depth < 8 ? baseStruct(code, all, id, d.at, depth + 1) : undefined;
}

function targetKind(code: string, target: string, pos: number, depth = 0): Kind {
  const path = target.replace(/[()*]/g, "").split(".");
  if (path.length > 1) {
    // A field path: walk the base's declared struct type; else the field's one type across every struct.
    const all = structs(code);
    let type = baseStruct(code, all, path[0]!, pos);
    for (const f of path.slice(1)) type = type ? all.get(type)?.get(f) : undefined;
    if (type) return kindOf(type) ?? { array: false, width: 3 };
    const field = path[path.length - 1]!, types = new Set([...all.values()].map(s => s.get(field)).filter((t): t is string => !!t));
    if (types.size > 1) throw new Error(`brick-lower: field ${field} has several declared types (${[...types].join(", ")}) and ${path[0]}'s struct is not declared (at "${code.slice(pos, pos + 48)}")`);
    return (types.size ? kindOf([...types][0]!) : undefined) ?? { array: false, width: 3 };
  }
  const d = declaration(code, path[0]!, pos);
  if (!d) return { array: false, width: 3 };
  if (d.type) {
    if (/\b(array|atomic)\s*</.test(d.type)) return { array: true };
    const v = d.type.match(/vec([234])/); if (v) return { array: false, width: Number(v[1]) };
  }
  const init = d.init ?? "";
  if (/^array\s*</.test(init)) return { array: true };
  const v = init.match(/^vec([234])/); if (v) return { array: false, width: Number(v[1]) };
  const call = init.match(/^(\w+)\s*\(/)?.[1], ret = call && fnReturn(code, call);
  if (ret) { const k = kindOf(ret); if (k) return k; }
  const id = init.match(/^\(?\*?([A-Za-z_]\w*(?:\.\w+)*)\)?$/) ?? init.match(/^([A-Za-z_]\w*(?:\.\w+)*)\s*[-+*/&|^%]/);
  return id && depth < 8 ? targetKind(code, id[1]!, d.at, depth + 1) : { array: false, width: 3 };
}

interface Write { start: number; end: number; target: string; index: string; op: string; rhs: string; width: number }

/** Each dynamic vector-component write in a module: its target, index, operator ("" plain, "++"/"--"), span. */
function writes(code: string): Write[] {
  const out: Write[] = [];
  const re = /((?:\(\*\w+\)|\b[A-Za-z_]\w*)(?:\.\w+)*)\[/g;
  for (let m = re.exec(code); m; m = re.exec(code)) {
    const start = m.index, target = m[1]!, open = start + target.length, close = closeAt(code, open, "[", "]");
    if (close < 0) continue;
    const index = code.slice(open + 1, close);
    let j = close + 1; while (code[j] === " ") j++;
    let op: string | undefined, rhsFrom = 0;
    if (code.startsWith("++", j) || code.startsWith("--", j)) { op = code.slice(j, j + 2); rhsFrom = j + 2; }
    else {
      const o = OPS.find(o => code.startsWith(o + "=", j) && code[j + o.length + 1] !== "=");
      if (o !== undefined && !(o === "" && "<>!=".includes(code[j - 1]!))) { op = o; rhsFrom = j + o.length + 1; }
    }
    if (op === undefined || /^\s*\d+[ui]?\s*$/.test(index)) continue;
    const kind = targetKind(code, target, start);
    if (kind.array) continue;
    // The right side: to the statement's ';' (or a for-header's ')') at depth 0.
    let k = rhsFrom, depth = 0;
    if (op !== "++" && op !== "--") for (; k < code.length; k++) { const ch = code[k]!; if ("([".includes(ch)) depth++; else if (")]".includes(ch)) { if (depth === 0) break; depth--; } else if (ch === ";" && depth === 0) break; }
    out.push({ start, end: k, target, index, op, rhs: code.slice(rhsFrom, k).trim(), width: kind.width });
    re.lastIndex = k;
  }
  return out;
}

/** The selects' axis masks a lowered text calls (nbDynAxis2..4), for a module that holds a lowered fragment. */
export const dynAxisWGSL = (widths: Iterable<number>) =>
  [...new Set(widths)].sort().map(n => `fn nbDynAxis${n}(a:u32)->vec${n}<bool>{return vec${n}<bool>(${Array.from({ length: n }, (_, i) => `a==${i}u`).join(",")});}`).join("\n");

export interface LoweredModule { readonly code: string; readonly sites: number; readonly widths: readonly number[] }

/**
 * The text with every dynamic vector-component write lowered to a select (the header's rule). `helpers: false`
 * leaves out the axis masks (a fragment spliced into another module, which adds dynAxisWGSL(widths) itself).
 */
export function lowerDynamicVectorWrites(code: string, options: { helpers?: boolean } = {}): LoweredModule {
  let out = "", at = 0; const widths = new Set<number>(); const sites = writes(code);
  for (const w of sites) {
    const n = w.width, t = w.target; widths.add(n);
    const value = w.op === "++" ? `${t}+vec${n}(1)` : w.op === "--" ? `${t}-vec${n}(1)` : w.op ? `${t}${w.op}vec${n}(${w.rhs})` : `vec${n}(${w.rhs})`;
    out += code.slice(at, w.start) + `${t}=select(${t},${value},nbDynAxis${n}(u32(${w.index})))`;
    at = w.end;
  }
  out += code.slice(at);
  if (widths.size && options.helpers !== false) out += "\n" + dynAxisWGSL(widths) + "\n";
  return { code: out, sites: sites.length, widths: [...widths].sort() };
}

/** The dynamic vector-component writes a text holds (the guard, and the scanner's count). */
export function scanDynamicVectorWrites(code: string): string[] {
  return writes(code).map(w => `${w.target}[${w.index}]${w.op}${w.op === "++" || w.op === "--" ? "" : "="}`);
}

/** The lowered module, or an explicit failure naming the kernel when a dynamic write survives the lowering. */
export function brickLowerChecked(label: string, code: string): string {
  const lowered = lowerDynamicVectorWrites(code).code, left = scanDynamicVectorWrites(lowered);
  if (left.length) throw new Error(`Brick engine: ${label} keeps ${left.length} dynamic vector-component writes after lowering (${left.slice(0, 4).join(" ")})`);
  return lowered;
}

/** The switch: the engine option when given, else NB_BRICK_LOWER=1 in the environment (node: tools and tests). */
export const brickLowerOn = (option?: boolean): boolean =>
  option ?? (typeof process !== "undefined" && process.env?.NB_BRICK_LOWER === "1");

/** A compile point's module text: exactly as written, or lowered and checked when the switch is on. */
export const brickModuleCode = (label: string, code: string, on: boolean): string => (on ? brickLowerChecked(label, code) : code);
