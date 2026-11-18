/**
 * An open world on the brick engine (container.walls "none", 2026-11-18). CPU only.
 * The old engine runs the box (sceneFluidBox) in its 64-page window; the brick lattice is the box scene
 * (sceneFluidBoxScene) widened by fluid.brickTiers.openWorldMargin pages a side on x and z, its sides the box rule's
 * walls, and liquid that reaches their last brick layer latches FAIL.REACH (setMark, info bit 31) before it meets them.
 * - the three open-ground scenes set up (no gap), the lattice the box plus the margin, the far φ the old engine's;
 * - the slope's columns are the analytic ground at the lattice's world cells (origin_m), past the old container;
 * - the margin is validated as whole pages in [1, 64] and widens the lattice;
 * - setMark latches on the outer x and z layers only with openSides, the open top's latch kept;
 * - a layout refuses open sides narrower than the lattice; a free-space boundary keeps its explicit gap.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getSceneDefinition } from "../../core/scenes";
import { sceneDocument } from "../../core/scene-definition";
import { validateScene } from "../../core/model";
import { terrainHeightAt } from "../../core/terrain";
import { sceneFluidBox } from "../../core/scene-lattice-dimensions";
import { BrickLayout } from "./brick-abi";
import { BRICK_OPEN_WORLD_MARGIN, brickSceneSetup } from "./brick-scene";
import { brickSetKernels } from "./brick-set.wgsl";

const OPEN = ["nbflip-open-ground", "nbflip-open-ground-faces", "nbflip-open-ground-slope"] as const;
const sixtieth = { timeStep: "sixtieth" };

test("brick open world: the open-ground scenes run on the box widened by the margin", () => {
  for (const id of OPEN) {
    const scene = sceneDocument(getSceneDefinition(id)), r = brickSceneSetup(scene, sixtieth), s = r.setup!;
    assert.deepEqual(r.gaps, [], id);
    const margin = scene.fluid.brickTiers?.openWorldMargin ?? BRICK_OPEN_WORLD_MARGIN;
    assert.equal(margin, 10, `${id}: the front's 6 s need ten pages a side`);
    const box = sceneFluidBox(scene, Number.MAX_SAFE_INTEGER).cells, side = 64 * margin;
    assert.deepEqual(s.lattice, [box[0]! + side, box[1]! + 32, box[2]! + side], `${id}: the box, the margin a side and one page of headroom`);
    assert.equal(s.box, undefined, `${id}: whole pages, the lattice is the box rule's box`);
    assert.deepEqual([s.openTop, s.openSides, s.sceneId], [true, true, id]);
    assert.ok(s.initial && s.liquid.length === 0, `${id}: the old engine's t = 0 φ`);
    assert.equal(s.farPhi, Math.max(...box), `${id}: the far φ is the old box's`);
  }
});

test("brick open world: the slope's columns are the analytic ground at the lattice's world cells", () => {
  const scene = sceneDocument(getSceneDefinition("nbflip-open-ground-slope")), s = brickSceneSetup(scene, sixtieth).setup!;
  const t = s.terrain!, [nx, nz] = t.columns, h = s.h_m;
  assert.deepEqual([nx, nz], [s.lattice[0], s.lattice[2]]);
  for (const [i, k] of [[0, 0], [nx - 1, 0], [nx >> 1, nz >> 1], [17, nz - 1], [nx - 1, nz - 1]] as const) {
    const x = s.origin_m[0] + (i + 0.5) * h, z = s.origin_m[2] + (k + 0.5) * h;
    assert.ok(Math.abs(t.heights[i + nx * k]! * h - terrainHeightAt(scene.terrain, x, z)) < 1e-4, `column ${i},${k}`);
  }
  assert.ok(t.heights.reduce((a, b) => Math.max(a, b), 0) * h > 4, "the mound rises inside the margin, past the old container");
});

test("brick open world: the margin is whole pages in [1, 64] and widens the lattice", () => {
  const scene = sceneDocument(getSceneDefinition("nbflip-open-ground"));
  for (const [v, ok] of [[1, true], [8, true], [64, true], [0, false], [65, false], [2.5, false]] as const) {
    const s = structuredClone(scene); s.fluid.brickTiers = { openWorldMargin: v };
    assert.equal(validateScene(s).some(e => e.includes("openWorldMargin")), !ok, `${v}`);
  }
  const wide = structuredClone(scene); wide.fluid.brickTiers = { openWorldMargin: 8 };
  assert.deepEqual(brickSceneSetup(wide, sixtieth).setup!.lattice, [64 + 512, 64, 64 + 512]);
  const plain = structuredClone(scene); delete plain.fluid.brickTiers;
  assert.deepEqual(brickSceneSetup(plain, sixtieth).setup!.lattice, [64 + 64 * BRICK_OPEN_WORLD_MARGIN, 64, 64 + 64 * BRICK_OPEN_WORLD_MARGIN], "the default margin");
});

test("brick open world: setMark latches FAIL.REACH on the outer x and z brick layers with open sides", () => {
  const base = { lattice: [128, 64, 128] as const, f: 64, c: 8, b: 16, k: 8, particles: 4096 };
  const mark = (o: object) => brickSetKernels(new BrickLayout({ ...base, ...o })).setMark.code;
  const sides = "o.x<8||o.z<8||o.x+8>=NB_LATTICE.x||o.z+8>=NB_LATTICE.z";
  const both = mark({ openTop: true, openSides: true }), top = mark({ openTop: true }), closed = mark({});
  assert.ok(both.includes(`if(atomicLoad(&wNeg)>0u&&(o.y+8>=NB_LATTICE.y||${sides})){nbFail(NB_FAIL_REACH,0x80000000u|nbKey(o>>vec3u(3u)));}`));
  assert.ok(both.includes(`if(lid<8u&&(o.y+8>=NB_LATTICE.y||${sides})){`), "the tile latch too");
  assert.ok(top.includes("if(atomicLoad(&wNeg)>0u&&o.y+8>=NB_LATTICE.y){nbFail(NB_FAIL_REACH,") && !top.includes("o.x<8"), "the open top alone");
  assert.ok(!closed.includes("NB_FAIL_REACH,0x80000000u"), "closed: no latch");
});

test("brick open world: open sides hold the lattice's x and z; a free-space boundary keeps its gap", () => {
  const base = { lattice: [128, 64, 128] as const, f: 64, c: 8, b: 16, k: 8, particles: 4096 };
  assert.throws(() => new BrickLayout({ ...base, box: [120, 64, 128], openSides: true }), /open world's box/);
  const scene = sceneDocument(getSceneDefinition("nbflip-open-ground"));
  const free = structuredClone(scene); free.container = { ...free.container, boundary: "free-space" };
  assert.match(brickSceneSetup(free, sixtieth).gaps.join(";"), /free-space boundary/);
});
