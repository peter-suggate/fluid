/** The brick engine's static solid queries (brick-static.wgsl.ts, M3 stage 1, the box rule): the Float64 mirror's
 * semantics, and the WGSL against the mirror bit for bit (GPU; skips without WEBGPU_NODE_MODULE). */
import test from "node:test";
import assert from "node:assert/strict";
import { brickStaticMirror, brickStaticWGSL, NB_STATIC } from "./brick-static.wgsl";
import { parityGPU, prng, runParity, withParityDevice } from "./parity/harness";

const BOX = [12, 8, 4] as const, FRAME = [0.1, -0.2, 0.3] as const, H = [0.05, 0.05, 0.05] as const;

test("statics: the box rule, closed and open top", () => {
  for (const openTop of [false, true]) {
    const m = brickStaticMirror({ box: BOX, openTop, frame: FRAME, h: H });
    assert.equal(m.cellKind([0, 0, 0]), NB_STATIC.PLAIN);
    assert.equal(m.cellKind([11, 7, 3]), NB_STATIC.PLAIN);
    for (const c of [[-1, 0, 0], [0, -1, 0], [0, 0, -1], [12, 0, 0], [0, 0, 4]]) {
      assert.equal(m.cellKind(c), NB_STATIC.VOXEL, `${c}`); assert.equal(m.cellOpen(c), 0);
    }
    // Above the box: solid under a closed top, air under an open one.
    assert.equal(m.cellOpen([3, 8, 1]), openTop ? 1 : 0);
    assert.equal(m.cellOpen([3, 40, 1]), openTop ? 1 : 0);
    // Faces: the +axis face of c; a wall face is closed.
    assert.equal(m.faceOpen([10, 3, 1], 0), 1);
    assert.equal(m.faceOpen([11, 3, 1], 0), 0);
    assert.equal(m.faceOpen([3, 7, 1], 1), openTop ? 1 : 0);
    assert.equal(m.faceOpen([-1, 3, 1], 0), 0);
    assert.equal(m.faceBoxed([3, 3, 1], 2), false);
    // World points: cell (0,0,0) spans frame .. frame + h.
    assert.equal(m.solidAt([0.1 + 0.025, -0.2 + 0.025, 0.3 + 0.025]), false);
    assert.equal(m.solidAt([0.1 - 0.001, -0.2 + 0.025, 0.3 + 0.025]), true);
    assert.equal(m.solidAt([0.1 + 0.025, -0.2 + 8 * 0.05 + 0.01, 0.3 + 0.025]), !openTop);
  }
});

test("statics: WGSL against the mirror", { skip: !parityGPU }, async () => {
  await withParityDevice(async device => {
    for (const openTop of [false, true]) {
      const m = brickStaticMirror({ box: BOX, openTop, frame: FRAME.map(Math.fround) as unknown as [number, number, number], h: H.map(Math.fround) as unknown as [number, number, number] });
      const prelude = brickStaticWGSL({ box: BOX, openTop, frame: `vec3f(${FRAME.join(",")})`, h: `vec3f(${H.join(",")})` });
      // Every cell of the box grown by 2 on each side (and 30 above), each axis; then random world points.
      const cells: number[][] = [];
      for (let z = -2; z < BOX[2] + 2; z++) for (let y = -2; y < BOX[1] + 30; y++) for (let x = -2; x < BOX[0] + 2; x++) cells.push([x, y, z]);
      const r = prng(7), worlds: number[][] = [];
      for (let i = 0; i < cells.length; i++) worlds.push(FRAME.map((f, a) => Math.fround(f + (r() * (BOX[a]! + (a === 1 ? 34 : 4)) - 2) * H[a]!)));
      const n = cells.length, inp = new Uint32Array(6 * n), fw = new Float32Array(inp.buffer);
      cells.forEach((c, i) => { inp[6 * i] = c[0]! >>> 0; inp[6 * i + 1] = c[1]! >>> 0; inp[6 * i + 2] = c[2]! >>> 0; fw.set(worlds[i]!, 6 * i + 3); });
      const OUT = 9;
      const body = `let c=vec3i(bitcast<i32>(inp[6u*t]),bitcast<i32>(inp[6u*t+1u]),bitcast<i32>(inp[6u*t+2u]));let w=v3(6u*t+3u);let o=${OUT}u*t;
 outp[o]=nbStaticCellKind(c);put(o+1u,nbStaticCellOpen(c));
 for(var a=0u;a<3u;a++){put(o+2u+a,nbStaticFaceOpen(c,a));outp[o+5u+a]=select(0u,1u,nbStaticFaceBoxed(c,a));}
 outp[o+8u]=select(0u,1u,nbStaticSolidAt(w));`;
      const got = await runParity(device, { prelude, body }, { inp, threads: n, outWords: OUT * n });
      const gf = new Float32Array(got.buffer);
      let bad = 0;
      cells.forEach((c, i) => {
        const o = OUT * i, want = [m.cellKind(c), m.cellOpen(c), m.faceOpen(c, 0), m.faceOpen(c, 1), m.faceOpen(c, 2), 0, 0, 0, m.solidAt(worlds[i]!) ? 1 : 0];
        const have = [got[o]!, gf[o + 1]!, gf[o + 2]!, gf[o + 3]!, gf[o + 4]!, got[o + 5]!, got[o + 6]!, got[o + 7]!, got[o + 8]!];
        if (want.some((v, j) => v !== have[j])) { if (bad++ < 3) console.error(c, worlds[i], want, have); }
      });
      assert.equal(bad, 0, `openTop ${openTop}: ${bad} of ${n} differ`);
    }
  });
});
