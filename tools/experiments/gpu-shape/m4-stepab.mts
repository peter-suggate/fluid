// GPU-shape M4 (lane B2): stepab.mts (lane A2's frozen whole-step A/B) for a kernel change that also moves its
// placement to another indirect args word (a split page: the dispatch is k n). stepab requires every placement's
// dispatch to agree; this runs stepab's own text with two changes, generated into OUTDIR (default
// ~/code/me/fluid-scratch/gpu-shape/m4/stepab), stepab.mts itself untouched:
//  - the shape check accepts an indirect dispatch for an indirect one (label, entry, bindings, z/w must still agree);
//  - a changed placement runs with the candidate's dispatch (its text and its args word).
// The args words must be written by both trees: BASE is the candidate with the kernels and placements reverted but
// the ABI's words and setArgs kept (docs/plans/gpu-shape/m4-clist.md), so setArgs is the same text in both.
//
//   SCENE=... BASE=tree CAND=tree [stepab's env] tools/gpu/fluid-gpu-run.sh timing node --import tsx tools/experiments/gpu-shape/m4-stepab.mts
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

const dir = import.meta.dirname, root = resolve(dir, "../../..");
const sub1 = (t: string, a: string, b: string) => { if (t.split(a).length !== 2) throw new Error(`m4-stepab: stepab.mts changed shape (${a.slice(0, 60)})`); return t.replace(a, () => b); };
let t = readFileSync(resolve(dir, "stepab.mts"), "utf8");
t = sub1(t, `const here = resolve(import.meta.dirname, "../../..");`, `const here = ${JSON.stringify(root)};`);
t = sub1(t, `from "./clock-probe.js";`, `from ${JSON.stringify(pathToFileURL(resolve(dir, "clock-probe.ts")).href)};`);
t = sub1(t, "const shape = (t: Text) => JSON.stringify([t.label, t.entry, t.bindings, t.dispatch, t.z ?? 0, t.w ?? 0]);",
  `const shape = (t: Text) => JSON.stringify([t.label, t.entry, t.bindings, (t.dispatch as any)?.indirect !== undefined ? "indirect" : t.dispatch, t.z ?? 0, t.w ?? 0]);`);
t = sub1(t, "const code = cand![i]!.code; if (code === p.kernel.code) return p;",
  "const code = cand![i]!.code, cd = cand![i]!.dispatch; if (code === p.kernel.code && JSON.stringify(cd) === JSON.stringify(p.dispatch)) return p;");
t = sub1(t, "return { ...p, kernel: memo.get(key) };", "return { ...p, kernel: memo.get(key), dispatch: cd };");
const out = resolve(process.env.OUTDIR ?? resolve(homedir(), "code/me/fluid-scratch/gpu-shape/m4/stepab"));
mkdirSync(out, { recursive: true });
const file = resolve(out, "stepab-dispatch.mts");
writeFileSync(file, t);
await import(pathToFileURL(file).href);
