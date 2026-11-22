/**
 * The brick engine's detail engine (brick-detail*.ts), on the CPU:
 * - the plan: no engine without automatic detail or a detail region; automatic detail is a heat budget over the
 *   container's 4h tiles (the default, or the scene's); authored regions are forced tiles (snapped outward, clipped),
 *   and two factors in one scene are refused;
 * - the pools and the mask's stores follow the budget, never the container: the same budget on a bigger container
 *   gives the same fine pools, the same `src` and the same allocator words but the bit planes;
 * - the fine scene: the whole container at h / factor, empty, the parent's sources and regions stripped, the solid
 *   patches scaled; what the engine cannot carry is an explicit gap;
 * - the slot map: a power of two at least twice the slots, and the WGSL hash (mirrored here) finds every key of a full
 *   map by linear probing;
 * - the parent's shader text without a detail layout carries no detail hook (resolveBox's tests, the census,
 *   residency, set and region-heat kernels, the source), and a detail layout routes the box tests through the mask;
 * - the allocator's dispatches cover the candidates and the forced tiles, the budget's slots, and its heat store is
 *   the parent's pools';
 * - the fine surface caps the side walls and the lid, leaves the floor open, and its pool is the budget's;
 * - the scene document: automatic detail and its budget validate, a detail region survives the editor's round trip,
 *   a detail change re-seeds the solve, and the uniform engine refuses detail.
 */
import assert from "node:assert/strict";
import test from "node:test";
import { getMethod } from "../../core/method-registry";
import { resolveMethodValues } from "../../core/method-contract";
import { sceneDocument } from "../../core/scene-definition";
import { getSceneDefinition } from "../../core/scenes";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { NBFLIP_TEASER_SCENE_ID } from "../../core/nbflip-paper-scenes";
import { validateScene, type FluidRefinementRegion, type SceneDescription } from "../../core/model";
import { detailRegionRefusal, refinementRegionLattice, validateRefinementRegions } from "../../core/refinement-regions";
import { refinementRegionFromRecord, refinementRegionRecord, studioRegionSpaceForMethod } from "../../core/editor-refinement-region";
import { sceneEditRequiresReset } from "../../core/webgpu-renderer";
import { brickSceneSetup } from "./brick-scene";
import { BRICK_STATIC_EDIT_K, brickStaticCutBricks } from "./brick-static.wgsl";
import { brickDetailPools, brickDetailSpecs, DETAIL_RING } from "./brick-detail";
import { brickDetailPlan, DETAIL_DEFAULT_BUDGET } from "./brick-detail-regions";
import { brickDetailSceneGaps, brickDetailSubScene } from "./brick-detail-scene";
import { brickDetailHash, brickDetailPlaneWords, brickDetailSrc, brickDetailWGSL, DETAIL_HASH_WGSL, type BrickDetailLayout } from "./brick-detail.wgsl";
import { brickDetailAllocDispatches, brickDetailAllocWords } from "./brick-detail-alloc.wgsl";
import { brickDetailSurfaceCapacity, brickDetailSurfaceWGSL } from "./brick-detail-surface";
import { BRICK_SOURCE_WGSL, brickSourceWGSL } from "./brick-source.wgsl";
import { brickResidencyKernels } from "./brick-residency.wgsl";
import { brickSetKernels } from "./brick-set.wgsl";
import { brickRegionHeatKernels } from "./brick-region-heat.wgsl";
import { BrickLayout, FLAG_DTL_OUT, resolveBox } from "./brick-abi";
import type { Vec3 } from "./brick-engine";
import "../index";

const METHOD = "uniform-narrow-band-flip";

function teaser(fluid: Partial<SceneDescription["fluid"]> = {}): SceneDescription {
  const scene = sceneDocument(getSceneDefinition(NBFLIP_TEASER_SCENE_ID));
  return { ...scene, fluid: { ...scene.fluid, ...fluid } };
}

function crown(fluid: Partial<SceneDescription["fluid"]> = {}): SceneDescription {
  const scene = sceneDocument(getSceneDefinition("cm12-figure-6"));
  return { ...scene, fluid: { ...scene.fluid, ...fluid } };
}

function specs(scene: SceneDescription) {
  const method = getMethod(METHOD), values = resolveMethodValues(method, "balanced", { ...method.appDefaults, engine: "brick" });
  const { setup, gaps } = brickSceneSetup(scene, values);
  assert.deepEqual(gaps, [], "the scene runs on the brick engine");
  return brickDetailSpecs(scene, setup!, values);
}

/** A detail region over parent cells [lo, hi) of `scene`. */
function region(scene: SceneDescription, id: string, lo: readonly number[], hi: readonly number[], cells: 0.5 | 0.25): FluidRefinementRegion {
  const { cellSize_m: h, origin_m: o } = refinementRegionLattice(scene);
  const at = (v: readonly number[]) => ({ x: o.x + v[0]! * h[0]!, y: o.y + v[1]! * h[1]!, z: o.z + v[2]! * h[2]! });
  return { id, rule: "minimum-cell-size", minimumCellSize_cells: cells, maximumCellSize_cells: cells, min_m: at(lo), max_m: at(hi) };
}

test("the plan: a heat budget over the container's tiles, authored regions forced, one factor", () => {
  assert.equal(specs(teaser()).length, 0, "no automatic detail, no regions: no engine");
  const cells = sceneLatticeDimensions(teaser()) as unknown as Vec3;
  const [spec, ...rest] = specs(teaser({ automaticDetail: { factor: 2 } }));
  assert.equal(rest.length, 0, "one engine");
  assert.deepEqual(spec!.plan.tiles, cells.map(n => Math.ceil(n / 4)), "the mask is the container's 4h tiles");
  assert.equal(spec!.plan.budget, DETAIL_DEFAULT_BUDGET);
  assert.equal(spec!.plan.forced.length, 0);
  assert.deepEqual(spec!.plan.origins, ["auto"]);
  assert.equal(spec!.layout.budget, DETAIL_DEFAULT_BUDGET);
  assert.equal(spec!.layout.slots, DETAIL_DEFAULT_BUDGET * (1 + DETAIL_RING));
  assert.ok(Math.abs(spec!.setup.h_m * 2 - refinementRegionLattice(teaser()).cellSize_m[0]!) < 1e-12, "fine h is h / 2");
  for (let a = 0; a < 3; a++) assert.ok(spec!.setup.lattice[a]! >= 2 * cells[a]!, "the fine lattice holds the container at h / 2");
  assert.equal(specs(teaser({ automaticDetail: { factor: 4, budget: 64 } }))[0]!.layout.budget, 64, "the scene's budget");

  // An authored region: its tiles forced, snapped outward to whole tiles and clipped to the container.
  const base = teaser(), r = region(base, "a", [41, 2, 10], [70, 30, 50], 0.5);
  const plan = brickDetailPlan({ ...base, fluid: { ...base.fluid, refinementRegions: [r] } }, cells)!;
  assert.equal(plan.budget, 0, "no heat without automatic detail");
  assert.equal(plan.factor, 2);
  assert.equal(plan.forced.length, (Math.ceil(70 / 4) - 10) * (Math.ceil(30 / 4) - 0) * (Math.ceil(50 / 4) - 2), "outward to whole tiles");
  for (let k = 1; k < plan.forced.length; k++) assert.ok(plan.forced[k]! > plan.forced[k - 1]!, "sorted, no repeats");
  const both = brickDetailPlan({ ...base, fluid: { ...base.fluid, refinementRegions: [r], automaticDetail: { factor: 2, budget: 100 } } }, cells)!;
  assert.deepEqual(both.origins, ["a", "auto"]);
  assert.equal(specs({ ...base, fluid: { ...base.fluid, refinementRegions: [r], automaticDetail: { factor: 2, budget: 100 } } })[0]!.layout.budget, 100 + plan.forced.length, "forced tiles on top of the heat's");
  assert.throws(() => brickDetailPlan({ ...base, fluid: { ...base.fluid, refinementRegions: [r], automaticDetail: { factor: 4 } } }, cells), /one factor/);
  assert.throws(() => brickDetailPlan({ ...base, fluid: { ...base.fluid, refinementRegions: [r, region(base, "b", [0, 0, 0], [8, 8, 8], 0.25)] } }, cells), /one factor/);
});

test("the stores follow the budget, never the container", () => {
  const small = specs(teaser({ automaticDetail: { factor: 2, budget: 128 } }))[0]!, big = specs(crown({ automaticDetail: { factor: 2, budget: 128 } }))[0]!;
  assert.notDeepEqual(small.plan.tiles, big.plan.tiles, "different containers");
  const pools = ({ k: _k, ...rest }: { readonly k: number }) => rest;
  assert.deepEqual(pools(small.setup.capacities), pools(big.setup.capacities), "the fine pools are the budget's (the cut slots, k, the solids')");
  assert.deepEqual(brickDetailSrc(small.layout), brickDetailSrc(big.layout), "src is the budget's");
  const parent = { cap: { f: 1000, c: 100 } } as unknown as BrickLayout;
  const a = brickDetailAllocWords(small.layout, parent, 0), b = brickDetailAllocWords(big.layout, parent, 0);
  assert.equal(a.words, b.words, "the allocator's words are the budget's and the parent's pools'");
  assert.equal(brickDetailPlaneWords(big.layout), Math.ceil(big.plan.tiles.reduce((p, n) => p * n, 1) / 32), "a bit a tile, the one container-sized store");
  // The pools grow with the budget and the factor (a lone tile's bricks and the layer past its + sides).
  const p64 = brickDetailPools({ factor: 2, budget: 64 }, 64), p512 = brickDetailPools({ factor: 2, budget: 512 }, 64), q512 = brickDetailPools({ factor: 4, budget: 512 }, 64);
  assert.ok(p512.f >= 512 * 8 && q512.f >= 512 * 27 && p64.f < p512.f);
  assert.ok(p512.c <= p512.f && p512.k <= p512.f);
  // K follows the solids the budget's F bricks may cut, not the empty sub-scene's own (liquid-sized) pool: the
  // teaser's rods cut more fine bricks than that pool's 128 (2026-11-23, POOL_K at step ~60 on the teaser).
  const cut = brickStaticCutBricks(small.setup.staticMask!);
  assert.equal(small.setup.capacities.k, Math.min(small.setup.capacities.f, cut + BRICK_STATIC_EDIT_K), "K: the cut bricks and the edit slack, at most F");
  assert.ok(small.setup.capacities.k > 128, `the teaser's rods need more than 128 K slots (${small.setup.capacities.k})`);
});

test("the fine residency claims a reached tile's keep block (all-h), grown by the parent's flow on a fresh tile", () => {
  // resDetail's box, in bricks: b0 = floor((lo - rlo) / 8), b1 = floor((lo + U - 1 + rhi) / 8) + 1 per axis, U = 4 factor.
  const box = (factor: number, T: number, rlo: number, rhi: number) => {
    const U = 4 * factor, lo = U * T;
    return [Math.floor((lo - rlo) / 8), Math.floor((lo + U - 1 + rhi) / 8) + 1] as const;
  };
  for (const factor of [2, 4] as const) {
    const detail: BrickDetailLayout = { factor, tiles: [64 / (4 * factor), 64 / (4 * factor), 64 / (4 * factor)], budget: 64, slots: 192 };
    const L = new BrickLayout({ lattice: [64, 64, 64], box: [64, 64, 64], f: 512, c: 8, b: 16, k: 0, particles: 1024, detail });
    const code = brickResidencyKernels(L).resDetail!.code;
    assert.ok(code.includes("let b0=vec3i(floor((lo-rlo)/8.0));let b1=vec3i(floor((lo+vec3f(NB_DTL_U-1.0)+rhi)/8.0))+vec3i(1);"), "the box the arithmetic below checks");
    assert.ok(code.includes("nbDtlBit(1u,T)") && code.includes("NB_REACH_MARGIN"), "a fresh tile's reach rule");
    // A workgroup a tile, its lanes over the fresh tile's node block and the keep block's bricks: no lane walks the
    // active tiles in turn (the 8-workgroup stride cost 0.3 ms a step at budget 32, 2026-11-23).
    assert.ok(code.includes("let j=nbGroup(wid)") && code.includes("for(var n=lane;") && code.includes("for(var k=lane;"), "a tile a workgroup, the lanes split it");
    assert.ok(!code.includes("num_workgroups"), "no stride over the active tiles");
    for (const T of [0, 1, 3]) {
      const [b0, b1] = box(factor, T, 0, 0);
      assert.equal(b1 - b0 + 1, factor / 2 + 1, `factor ${factor}: no flow, (factor/2 + 1) bricks an axis, as nbDtlBrickKeep and the F pool`);
      assert.equal(b0, (4 * factor * T) >> 3, "starting at the tile's own lowest brick");
    }
    // A falling drop at a 1/60 s step: ~3 m/s down at fine h 2 mm is ~25 cells, plus the reach margin, below the tile.
    const [d0, d1] = box(factor, 3, 25 + 7, 7), [k0, k1] = box(factor, 3, 0, 0);
    assert.ok(k0 - d0 >= 4 && d1 - k1 === 1, `factor ${factor}: the box reaches 4 bricks down, a margin's up`);
  }
});

test("the fine scene: the container at h / factor, empty, the solids scaled", () => {
  const scene = teaser({ automaticDetail: { factor: 2 } }), sub = brickDetailSubScene(scene, 2), h = refinementRegionLattice(scene).cellSize_m[0]!;
  assert.equal(sub.container.width_m, scene.container.width_m, "the whole container");
  assert.ok(Math.abs(sub.voxelDomain.finestCellSize_m - h / 2) < 1e-12);
  assert.equal(sub.fluid.automaticDetail, undefined);
  assert.equal(sub.fluid.refinementRegions, undefined);
  assert.equal(sub.fluid.brickPools, undefined);
  assert.equal(sub.container.fillFraction, 0);
  assert.deepEqual(sub.fluid.initialLiquidVolumes, []);
  const parentSolids = brickDetailSubScene(scene, 2).solidVoxels ?? [];
  assert.ok(parentSolids.length > 0, "the rods");
  const four = brickDetailSubScene(scene, 4).solidVoxels!;
  for (let k = 0; k < parentSolids.length; k++) for (let a = 0; a < 3; a++) assert.equal(four[k]!.minimum[a], 2 * parentSolids[k]!.minimum[a]!, "scaled by the factor");
  assert.deepEqual(brickDetailSceneGaps(scene), []);
  const gapped = { ...scene, container: { ...scene.container, walls: "none" as const } };
  assert.ok(brickDetailSceneGaps(gapped).some(g => /open world/.test(g)));
  assert.throws(() => brickDetailSubScene(gapped, 2), /open world/);
});

test("the slot map finds every key of a full map", () => {
  const d = { factor: 2, tiles: [32, 32, 32], budget: 100, slots: 300 } as const satisfies BrickDetailLayout;
  const n = brickDetailHash(d), shift = 32 - Math.log2(n);
  assert.equal(n & (n - 1), 0, "a power of two");
  assert.ok(n >= 2 * d.slots);
  assert.match(DETAIL_HASH_WGSL(n), new RegExp(`\\(i\\*2654435761u\\)>>${shift}u`));
  const hash = (i: number) => Math.imul(i, 2654435761) >>> shift, keys = new Int32Array(n).fill(-1);
  const tiles = Array.from({ length: d.slots }, (_, k) => (k * 7919) % (32 ** 3));
  for (const i of tiles) { let e = hash(i); while (keys[e] !== -1) e = (e + 1) & (n - 1); keys[e] = i; }
  for (const i of tiles) { let e = hash(i), found = false; for (let k = 0; k < n && keys[e] !== -1; k++, e = (e + 1) & (n - 1)) if (keys[e] === i) { found = true; break; } assert.ok(found, `tile ${i}`); }
});

test("the parent's shader text carries no detail hook without a detail layout", () => {
  const options = { lattice: [64, 64, 64] as [number, number, number], box: [64, 64, 64] as [number, number, number], f: 125, c: 8, b: 16, k: 0, particles: 1024 };
  const plain = new BrickLayout(options);
  const detail: BrickDetailLayout = { factor: 2, tiles: [16, 16, 16], budget: 64, slots: 192 };
  const nested = new BrickLayout({ ...options, detail });
  assert.equal(brickSourceWGSL(plain), BRICK_SOURCE_WGSL);
  assert.throws(() => brickDetailWGSL(plain), /detail layout/);
  assert.equal(resolveBox(plain, "nbInWallBox(c)&&nbTileInWallBox(t)"), "nbInLattice(c)&&nbTileInLattice(t)");
  assert.equal(resolveBox(nested, "nbInWallBox(c)&&nbInDtlBox(c)&&nbTileInWallBox(t)"), "nbInDtl(c)&&nbInLattice(c)&&nbTileInDtl(t)", "the box tests through the mask");
  const texts = (L: BrickLayout) => [
    ...Object.values(brickResidencyKernels(L, { tiered: true })), ...Object.values(brickSetKernels(L, { tiered: true })),
    ...Object.values(brickRegionHeatKernels(L)),
  ].map(k => (k as { code: string }).code).join("\n");
  assert.doesNotMatch(texts(plain), /Dtl|DTL/, "no hook without a detail layout");
  const hooked = texts(nested);
  for (const hook of ["fn resDetail(", "fn detailRefresh(", "slpMasked(", "nbDtlBrickKeep(", "nbDtlOn("]) assert.ok(hooked.includes(hook), hook);
  assert.equal(FLAG_DTL_OUT & (FLAG_DTL_OUT - 1), 0, "one flag bit");
});

test("the allocator: its dispatches cover the candidates, the forced tiles and the slots", () => {
  const d: BrickDetailLayout = { factor: 2, tiles: [32, 32, 32], budget: 70, slots: 210 }, parent = { cap: { f: 1000, c: 100 } } as unknown as BrickLayout;
  const w = brickDetailAllocWords(d, parent, 6), g = brickDetailAllocDispatches(parent, d, 6);
  assert.ok(64 * g.dtlMark![0] >= w.cand + 6, "every candidate and forced tile a thread");
  assert.ok(64 * g.dtlRing![0] >= d.budget, "every active a thread");
  assert.ok(64 * g.dtlClear![0] >= Math.max(brickDetailHash(d), d.budget), "the map and the last actives");
  assert.deepEqual(g.dtlThreshold, [1, 1, 1]);
  assert.equal(w.forced + 6, w.words, "the forced list last");
  assert.equal(w.heat + 3 * 512 * 100, w.forced, "the heat: three words a tile of the parent's C pool");
});

test("the fine surface: the floor open, the side walls and lid capped, the budget's pool", () => {
  const mask = { tiles: [16, 16, 16] as [number, number, number], unit: 8, word: 1234, budget: 64 };
  const text = brickDetailSurfaceWGSL({ cells: [128, 128, 128], bricks: [16, 16, 16], h_m: 0.01, origin_m: [0, 0, 0], mask, far_m: 1 });
  assert.match(text, /DS_CLOSED_LO:vec3<bool>=vec3<bool>\(true,false,true\)/, "the floor is open");
  assert.match(text, /DS_CLOSED_HI:vec3<bool>=vec3<bool>\(true,true,true\)/);
  assert.match(text, /DS_SHIFT:u32=3u;const DS_ON:u32=1234u;/);
  assert.equal(brickDetailSurfaceCapacity([128, 128, 128], 2 ** 40, mask), 6 * 2 * 3 * 8 * 8 * 64, "the budget's, not the lattice's");
  assert.equal(brickDetailSurfaceCapacity([128, 128, 128], 2 ** 40) % 3, 0, "whole triangles");
  assert.equal(brickDetailSurfaceCapacity([128, 128, 128], 32 * 300), 300, "held to the device's binding");
});

test("the scene document: validation, the editor's tiers, re-seeding, refusal", () => {
  const scene = teaser();
  assert.deepEqual(validateScene(teaser({ automaticDetail: { factor: 2 } })), validateScene(scene));
  assert.deepEqual(validateScene(teaser({ automaticDetail: { factor: 2, budget: 256 } })), validateScene(scene));
  assert.ok(validateScene(teaser({ automaticDetail: { factor: 3 as 2 } })).some(e => /Automatic detail factor/.test(e)));
  for (const budget of [0, 1.5, 70000]) assert.ok(validateScene(teaser({ automaticDetail: { factor: 2, budget } })).some(e => /Automatic detail budget/.test(e)), `budget ${budget}`);
  const r = region(scene, "d", [40, 0, 40], [80, 40, 80], 0.5);
  assert.deepEqual(validateRefinementRegions([r], scene.container), []);
  const tiers = studioRegionSpaceForMethod(METHOD).heldTiers!.map(t => t.cells);
  assert.ok(tiers.includes(0.5) && tiers.includes(0.25), "the editor offers the detail tiers");
  assert.ok(studioRegionSpaceForMethod(METHOD).cellSizes.includes(0.5));
  const back = refinementRegionFromRecord(scene, refinementRegionRecord(scene, r));
  assert.equal(back.minimumCellSize_cells, 0.5);
  for (const k of ["x", "y", "z"] as const) assert.ok(Math.abs(back.min_m[k] - r.min_m[k]) < 1e-9 && Math.abs(back.max_m[k] - r.max_m[k]) < 1e-9);
  // The detail engine is built with the solver: a detail change re-seeds; an ordinary region stays a live edit.
  assert.ok(sceneEditRequiresReset(scene, teaser({ automaticDetail: { factor: 2 } }), METHOD));
  assert.ok(sceneEditRequiresReset(teaser({ automaticDetail: { factor: 2 } }), teaser({ automaticDetail: { factor: 2, budget: 64 } }), METHOD), "a budget change re-seeds");
  assert.ok(sceneEditRequiresReset(scene, teaser({ refinementRegions: [r] }), METHOD));
  assert.ok(!sceneEditRequiresReset(scene, teaser({ refinementRegions: [{ ...r, minimumCellSize_cells: 1, maximumCellSize_cells: 1 }] }), METHOD));
  assert.equal(detailRegionRefusal(scene, "x"), undefined);
  assert.match(detailRegionRefusal(teaser({ refinementRegions: [r] }), "The uniform engine")!, /only the brick NB-FLIP engine/);
  assert.match(detailRegionRefusal(teaser({ automaticDetail: { factor: 2 } }), "The uniform engine")!, /automatic detail/);
});
