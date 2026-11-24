/** Lane F's staged owner windows and vertex rings (fluid-grid.wgsl.ts: fgWindowStage / fgWindowStageAt,
 * fgRingStage): a lane's window or ring lookups read entries other lanes stage, so every staging call in a kernel
 * must be followed by a workgroup barrier before any other statement (uExtend once read the window in its 4h state
 * staging before the barrier, a race that changed UP on the dam's frozen state). Checked on every lane F kernel,
 * tiered and all-h, with and without viscosity. */
import test from "node:test";
import assert from "node:assert/strict";
import { BrickLayout } from "./brick-abi";
import { fluidStagesAt } from "./brick-fluid-stages";
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

/** FC_LIQ (fluid-grid.wgsl.ts) is a per-step cache of each cell's liquid phase: uTransfer (uKick at boot) writes it
 * from φ, and the tail's extension reads it in place of recomputing the cell's corners. It is exact only when every
 * reader runs after a writer of the same schedule with no φ writer between them. */
test("FC_LIQ: every reader runs after its writer, with no φ writer between", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  type K = { label: string; code: string; bindings: readonly { name: string; kind: string }[] };
  const writes = (k: K) => k.code.includes("fcStore(FC_LIQ+"), reads = (k: K) => k.code.includes("fcLoad(FC_LIQ+");
  const phiWriter = (k: K) => k.bindings.some(b => b.name === "phi" && b.kind === "rw");
  let readers = 0;
  for (const tiered of [false, true]) {
    for (const viscosity_Pa_s of [0, 1e-3]) {
      const stages = fluidStagesAt(tiered)({ layout: L, values: { volumeControlSeconds: 0 }, fluid: { viscosity_Pa_s, density_kg_m3: 1000, surfaceTension_N_m: 0 } } as never) as unknown as Record<string, unknown>;
      const of = (list: unknown) => ((list as { kernel: K }[] | undefined) ?? []).map(p => p.kernel);
      const kick = stages.kick as Record<string, unknown>;
      const schedules = {
        step: ["move", "rebin", "levelSet", "velocity", "tail"].flatMap(s => of(stages[s])),
        kick: [...of(kick.force), ...of(kick.after), ...of(kick.seed)],
      };
      for (const [name, list] of Object.entries(schedules)) {
        let written = false;
        for (const k of list) {
          if (phiWriter(k)) written = false;
          if (writes(k)) written = true;
          if (reads(k)) { readers++; assert.ok(written, `${name} (tiered ${tiered}): ${k.label} reads FC_LIQ before its writer or after a φ write`); }
        }
      }
    }
  }
  // uExtend at every schedule, and tiered uExt4Seed and uExt4.
  assert.ok(readers >= 12, `${readers} FC_LIQ readers`);
});

/** fcell.UNIT (fluid-grid.wgsl.ts, brick-abi.md): uCache's resolved h taps of US, a slot's 9^3 with the +side apron.
 * The sampler reads a stencil's eight taps through its base slot, so every reader must run after the schedule's
 * uCache and before the next US writer (uTransfer, uExtend, uExt4, uKick): a later reader would take taps of a US
 * the apron no longer matches. */
test("FC_UNIT: every reader runs after uCache, with no US writer between", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  type K = { label: string; code: string };
  const writes = (k: K) => k.code.includes("fcStoreF(FC_UNIT+"), reads = (k: K) => !writes(k) && k.code.includes("FC_UNIT+");
  const usWriter = (k: K) => /vel\[vc?At\(NB_VEL_[FC]_US0[^\]]*\]\s*=[^=]/.test(k.code);
  let readers = 0, writers = 0, usWriters = 0;
  for (const viscosity_Pa_s of [0, 1e-3]) {
    const stages = fluidStagesAt(true)({ layout: L, values: { volumeControlSeconds: 0 }, fluid: { viscosity_Pa_s, density_kg_m3: 1000, surfaceTension_N_m: 0 } } as never) as unknown as Record<string, unknown>;
    const of = (list: unknown) => ((list as { kernel: K }[] | undefined) ?? []).map(p => p.kernel);
    const kick = stages.kick as Record<string, unknown>;
    // The kick's seed follows its after (between them only E's TRIG-only census, which writes no US).
    const schedules = { step: ["move", "rebin", "levelSet", "velocity", "tail"].flatMap(s => of(stages[s])), boot: of(stages.boot), kick: [...of(kick.force), ...of(kick.after), ...of(kick.seed)] };
    for (const [name, list] of Object.entries(schedules)) {
      let fresh = false;
      for (const k of list) {
        if (writes(k)) { fresh = true; writers++; }
        if (reads(k)) { readers++; assert.ok(fresh, `${name}: ${k.label} reads FC_UNIT before uCache or after a US writer`); }
        if (usWriter(k)) { usWriters++; fresh = false; }
      }
    }
  }
  // uAdvect, lAdvect, lcAdvect, pMove and pSeed (twice) a step; uTransfer and uExtend among the US writers.
  assert.ok(readers >= 12 && writers >= 2 && usWriters >= 6, `${readers} FC_UNIT readers, ${writers} writers, ${usWriters} US writers`);
});

/** Resolved φ (fluid-grid.wgsl.ts FG_PHI_WGSL): tiered, a non-owned vertex's PHI word is exact only between the
 * lResolve after the last writer of what it depends on (a C PHI writer, or the set update before a schedule) and
 * the next such writer. Every kernel that reads resolved words (FG_PHI_RESOLVED on, or a staged ring) must run in
 * such a span. */
test("resolved φ: no resolved read before the schedule's lResolve or after a C PHI write without one", () => {
  const L = new BrickLayout({ lattice: [128, 128, 128], f: 600, c: 40, b: 0, k: 0, particles: 1 << 16 });
  type K = { label: string; code: string };
  // lcAdvect's taps read F's PHI words (lcFResolved); the C kernels share one module text, so it counts by entry.
  const resolved = (k: K) => k.code.includes("const FG_PHI_RESOLVED:bool=true;") || /\bfgRingStage\(/.test(k.code.replace(/fn fgRingStage\(/g, ""))
    || (k.label === "lcAdvect" && k.code.includes("fn lcFResolved("));
  // Lane E's C kernels share one module text, so their writes are classified by entry (levelset-coarse.wgsl.ts):
  // C PHI by lcCouple, lcClear and lcPublish; PHI1 only by lcAdvect, lcRetire and lcRetireLong (lcRetire's listed
  // vertices). Any other C writer fails here.
  const C_PHI = new Set(["lcCouple", "lcClear", "lcPublish"]), C_PHI1 = new Set(["lcAdvect", "lcRetire", "lcRetireLong"]);
  const writesC = (k: K) => {
    if (C_PHI.has(k.label)) return true;
    if (C_PHI1.has(k.label)) return false;
    assert.ok(!/lcSetC\(/.test(k.code) && !/phi\[NB_PHI_C_PHI1?\+[^\]]*\]\s*=[^=]/.test(k.code), `${k.label} writes C vertices and is not classified`);
    return false;
  };
  let readers = 0, resolves = 0;
  for (const viscosity_Pa_s of [0, 1e-3]) {
    const stages = fluidStagesAt(true)({ layout: L, values: { volumeControlSeconds: 0 }, fluid: { viscosity_Pa_s, density_kg_m3: 1000, surfaceTension_N_m: 0 } } as never) as unknown as Record<string, unknown>;
    const of = (list: unknown) => ((list as { kernel: K }[] | undefined) ?? []).map(p => p.kernel);
    const kick = stages.kick as Record<string, unknown>;
    // Each list starts after a possible set update or census (stale until its lResolve).
    const schedules = { step: ["move", "rebin", "levelSet", "velocity", "tail"].flatMap(s => of(stages[s])), boot: of(stages.boot), kickForce: [...of(kick.force), ...of(kick.after)], kickSeed: of(kick.seed) };
    for (const [name, list] of Object.entries(schedules)) {
      let fresh = false, prev = "", phiWriters: string[] = [];
      const writesPhi = (k: K) => /phi\[NB_PHI_[FC]_PHI\+[^\]]*\]\s*=[^=]|lSetPhi\(NB_PHI_F_PHI,|lcSetC\(NB_PHI_C_PHI,/.test(k.code);
      for (const k of list) {
        // lResolveD re-resolves only what the analysis's clear changed: since the lResolve before it, PHI's only
        // writers are lClear and lcClear (whose stamps it reads), and it follows lcClear.
        if (k.label === "lResolveD") {
          assert.ok(prev === "lcClear", `${name}: lResolveD follows ${prev}, not lcClear`);
          assert.deepEqual([...new Set(phiWriters)].sort(), ["lClear", "lcClear"], `${name}: PHI writers since the last lResolve: ${phiWriters.join(",")}`);
          fresh = true; resolves++; prev = k.label; phiWriters = []; continue;
        }
        if (k.label === "lResolve") { fresh = true; resolves++; prev = k.label; phiWriters = []; continue; }
        if (writesPhi(k)) phiWriters.push(k.label);
        if (resolved(k)) { readers++; assert.ok(fresh, `${name}: ${k.label} reads resolved φ while a non-owned word may be stale`); }
        if (writesC(k)) fresh = false;
        prev = k.label;
      }
    }
  }
  assert.ok(readers >= 20 && resolves >= 8, `${readers} resolved readers, ${resolves} lResolve placements`);
});
