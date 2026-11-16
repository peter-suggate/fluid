/**
 * The old engine's refinement join box (nbRefineBox, uniform-particle-brick.wgsl.ts) is bounded: a travel at or past
 * the lattice's extent (2048 cells an axis, the move's trace limit) raises the move's fatal (state[3] = 1) and joins
 * nothing, as the brick engine's pJoin does (the GPU wedge of 2026-11-17: a sentinel-sized travel swept the whole
 * 2048^3-tile window for one record). A finite travel's box is unchanged.
 *
 * The function runs as written (its text cut from particleBrickDepositWGSL) over stubs for what it reads.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { particleBrickDepositWGSL } from "../lib/methods/uniform/uniform-particle-brick.wgsl";
import { parityGPU, runParity, withParityDevice, type ParityKernel } from "../lib/methods/narrow-band/parity/harness";

const text = particleBrickDepositWGSL(false, { joins: 0, allocation: 0 }, "bins", false, 0, 3);
const cut = (from: string, to: string) => {
  const a = text.indexOf(from), b = text.indexOf(to, a);
  assert.ok(a >= 0 && b > a, `${from} .. ${to} in particleBrickDepositWGSL`);
  return text.slice(a, b);
};

test("nbRefineBox latches a travel at the lattice's extent before it sizes the box", () => {
  const box = cut("fn nbRefineBox(", "fn nbRefineJoin(");
  const latch = box.indexOf("if(!all(abs(travel)<vec3f(NB_REFINE_REACH))){atomicStore(&state[3],1u);return NbRefineBox(vec3i(0),vec3i(-1));}");
  assert.ok(latch > 0 && latch < box.indexOf("let lo="), "the latch precedes the box");
  assert.match(text, /const NB_REFINE_REACH:f32=2048\.0;/);
});

// inp a thread: position (3), before (3), mode (ballistic), padding; 8 words. out: low (3), high (3), the fatal; 8.
const IN = 8, OUT = 8;
const KERNEL: ParityKernel = {
  prelude: /* wgsl */`
struct Particle{position:vec4f,velocity:vec4f,before:vec4f}
struct Params{hDt:vec4f,activity:vec4f}
var<private> params:Params;
@group(0) @binding(3) var<storage,read_write> state:array<atomic<u32>>;
@group(0) @binding(4) var<storage,read_write> bins:array<atomic<u32>>;
const NB_DEPTH_B:u32=0u;
fn umCellCoord(w:u32)->vec3u{return vec3u(w&1023u,(w>>10u)&1023u,w>>20u);}
${cut("const NB_REFINE_WINDOW", "// A tile's samples reach")}
${cut("struct NbRefineBox", "fn nbRefineJoin(")}`,
  body: /* wgsl */`let i=${IN}u*t;let o=${OUT}u*t;_=atomicLoad(&bins[0]);
 params=Params(vec4f(0.0125,0.0125,0.0125,1.0/60.0),vec4f(0.0,0.0,f(i+7u),0.0));
 let s=Particle(vec4f(v3(i),1.0),vec4f(v3(i+3u),0.0),vec4f(v3(i+3u),0.0));
 let box=nbRefineBox(s,inp[i+6u]!=0u,0u,false);
 for(var a=0u;a<3u;a++){outp[o+a]=bitcast<u32>(box.low[a]);outp[o+3u+a]=bitcast<u32>(box.high[a]);}
 outp[o+6u]=atomicLoad(&state[3]);outp[o+7u]=0u;`,
};

test("nbRefineBox (GPU): a sentinel or past-lattice travel is fatal and joins nothing; a finite one keeps its box", { skip: !parityGPU }, async () => {
  // before in m/s: travel = before dt / h cells = before / 0.75.
  const cases: { p: number[]; before: number[]; ballistic: number; fatal: boolean }[] = [
    { p: [40.5, 20.5, 30.5], before: [0.3, -1.2, 0.6], ballistic: 0, fatal: false },
    { p: [40.5, 20.5, 30.5], before: [6, 0, -3], ballistic: 1, fatal: false },
    { p: [40.5, 20.5, 30.5], before: [0, 0, 3e38], ballistic: 0, fatal: true },
    { p: [40.5, 20.5, 30.5], before: [-2100 * 0.75, 0, 0], ballistic: 0, fatal: true },
    { p: [40.5, 20.5, 30.5], before: [0, Number.NaN, 0], ballistic: 1, fatal: true },
  ];
  await withParityDevice(async device => {
    for (const [n, c] of cases.entries()) {
      const inp = new Float32Array(IN), w = new Uint32Array(inp.buffer);
      inp.set([...c.p, ...c.before]); w[6] = c.ballistic; inp[7] = 0.5;
      const out = await runParity(device, KERNEL, { inp, threads: 1, outWords: OUT, extra: { 3: new Uint32Array(16), 4: new Uint32Array(64) } });
      const low = [...new Int32Array(out.buffer, 0, 3)], high = [...new Int32Array(out.buffer, 12, 3)];
      if (c.fatal) {
        assert.equal(out[6], 1, `case ${n}: state[3] = 1`);
        assert.ok(low.some((x, a) => high[a]! < x), `case ${n}: the box is empty (${low} .. ${high})`);
      } else {
        assert.equal(out[6], 0, `case ${n}: no fatal`);
        // The CPU's box: the surface radius about the start, swept by the travel, padded, in tiles.
        const r = Number(/var low=start-([0-9.]+);/.exec(text)![1]);
        for (let a = 0; a < 3; a++) {
          const t = (c.before[a]! / 60) / 0.0125;
          assert.equal(low[a], Math.max(0, Math.floor((c.p[a]! - r + Math.min(t, 0) - 0.5) / 4)), `case ${n} low ${a}`);
          assert.ok(high[a]! >= low[a]! && high[a]! <= Math.floor((c.p[a]! + r + Math.max(t, 0) + 0.5) / 4), `case ${n} high ${a}`);
        }
      }
    }
  });
});
