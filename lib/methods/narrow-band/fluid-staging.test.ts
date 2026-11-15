/** Lane F's staged owner windows and vertex rings (fluid-grid.wgsl.ts: fgWindowStage / fgWindowStageAt,
 * fgRingStage): a lane's window or ring lookups read entries other lanes stage, so every staging call in a kernel
 * must be followed by a workgroup barrier before any other statement (uExtend once read the window in its 4h state
 * staging before the barrier, a race that changed UP on the dam's frozen state). Checked on every lane F kernel,
 * tiered and all-h, with and without viscosity. */
import test from "node:test";
import assert from "node:assert/strict";
import { BrickLayout } from "./brick-abi";
import { fluidLayout } from "./fluid-grid.wgsl";
import { levelSetKernels } from "./levelset-kernels.wgsl";
import { particleKernels } from "./particles-kernels.wgsl";
import { velocityKernels } from "./velocity-kernels.wgsl";

const STAGE = /\b(fgWindowStage|fgWindowStageAt|fgRingStage)\(/g;

/** The staging calls of code outside their own definitions, each with the statement after it. */
const stagingCalls = (code: string) => {
  const calls: { call: string; next: string }[] = [];
  for (const m of code.matchAll(STAGE)) {
    const at = m.index!;
    if (/fn\s+$/.test(code.slice(Math.max(0, at - 4), at))) continue;
    // The call's closing parenthesis, then its semicolon.
    let depth = 0, j = at + m[0].length - 1;
    for (; j < code.length; j++) { if (code[j] === "(") depth++; else if (code[j] === ")" && --depth === 0) break; }
    const end = code.indexOf(";", j);
    const call = code.slice(at, end + 1);
    // Calls inside another staging function's body (fgWindowStage calls fgWindowStageAt) are that function's.
    const before = code.slice(0, at), fnAt = before.lastIndexOf("fn "), name = /fn\s+(\w+)/.exec(before.slice(fnAt))?.[1] ?? "";
    if (/^fg(WindowStage|WindowStageAt|RingStage)$/.test(name)) continue;
    calls.push({ call, next: code.slice(end + 1).trimStart().slice(0, 40) });
  }
  return calls;
};

test("lane F kernels: every owner window and vertex ring staging call is followed by a workgroup barrier", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  let staged = 0;
  for (const tiered of [false, true]) {
    const F = fluidLayout(L, { tiered });
    for (const viscosity_Pa_s of [0, 1e-3]) {
      const kernels = { ...particleKernels(L, F), ...levelSetKernels(L, F), ...velocityKernels(L, F, { viscosity_Pa_s, density_kg_m3: 1000 }) };
      for (const [name, k] of Object.entries(kernels)) {
        if (!k || typeof k !== "object" || !("code" in k)) continue;
        for (const { call, next } of stagingCalls((k as { code: string }).code)) {
          staged++;
          assert.ok(next.startsWith("workgroupBarrier();"), `${name} (tiered ${tiered}): ${call} is followed by "${next}", not a barrier`);
        }
      }
    }
  }
  // The staged kernels exist (the check is not vacuous): pMove, uAdvect, uTransfer, uViscous, uExtend, uCache,
  // pDeposit, uExt4Seed, lTiles, lClear, lDistance.
  assert.ok(staged >= 20, `${staged} staging calls`);
});

test("the staging check catches a window read before the barrier", () => {
  const bad = "fn k(){ fgWindowStage(o>>vec3u(3u),lid,512u);xLow=o; workgroupBarrier(); }";
  assert.deepEqual(stagingCalls(bad).map(c => c.next.startsWith("workgroupBarrier();")), [false]);
  const good = "fn fgWindowStage(c:vec3i,l:u32,n:u32){fgWindowStageAt(c-vec3i(1),l,n);}\nfn k(){ fgWindowStage(o,lid,512u); workgroupBarrier(); }";
  assert.deepEqual(stagingCalls(good).map(c => c.next.startsWith("workgroupBarrier();")), [true]);
});
