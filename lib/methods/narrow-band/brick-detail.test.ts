/**
 * The brick engine's nested detail engines (brick-detail*.ts), on the CPU:
 * - placement: the automatic heuristic puts one box about the NB-FLIP teaser's rods and none in a still tank or with
 *   automatic detail off; authored detail regions snap to tiles, merge (the finer factor, both origins) and a coupled
 *   high face whose fine extent is whole pages pads by a tile;
 * - the box's scene: the container is the box at h / factor, the parent's regions, sources and pools are its alone,
 *   the solid patches are the parent's translated and scaled; what the box cannot carry is an explicit gap;
 * - the shell's slab index is a bijection onto [0, brickDetailNodes) and its WGSL is defined only for detail layouts;
 * - the parent's shader text without a detail layout is unchanged (the hooks are all conditional);
 * - the fine surface closes the container's walls and leaves the coupled faces open;
 * - the scene document: automatic detail validates, a detail region validates and survives the editor's record round
 *   trip, a detail change re-seeds the solve while an ordinary region stays live, and the uniform engine refuses detail.
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
import { brickDetailSpecs, detailCoupledFaces } from "./brick-detail";
import { automaticDetailBoxes, brickDetailBoxes } from "./brick-detail-regions";
import { brickDetailSceneGaps, brickDetailSubScene } from "./brick-detail-scene";
import { DETAIL_DEPTH, brickDetailIndex, brickDetailNodes, brickDetailWGSL } from "./brick-detail.wgsl";
import { brickDetailSurfaceCapacity, brickDetailSurfaceWGSL } from "./brick-detail-surface";
import { BRICK_SOURCE_WGSL, brickSourceWGSL } from "./brick-source.wgsl";
import type { BrickLayout } from "./brick-abi";
import type { Vec3 } from "./brick-engine";
import "../index";

const METHOD = "uniform-narrow-band-flip";

function teaser(fluid: Partial<SceneDescription["fluid"]> = {}): SceneDescription {
  const scene = sceneDocument(getSceneDefinition(NBFLIP_TEASER_SCENE_ID));
  return { ...scene, fluid: { ...scene.fluid, ...fluid } };
}

function brickValues() {
  const definition = getSceneDefinition(NBFLIP_TEASER_SCENE_ID), method = getMethod(METHOD), profile = definition.methodProfile;
  const quality = profile?.methodId === METHOD ? profile.quality : "balanced";
  return resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === METHOD ? profile.overrides : {}), engine: "brick" });
}

function specs(scene: SceneDescription) {
  const values = brickValues(), { setup, gaps } = brickSceneSetup(scene, values);
  assert.deepEqual(gaps, [], "the teaser runs on the brick engine");
  return brickDetailSpecs(scene, setup!, values);
}

/** A detail region over parent cells [lo, hi) of `scene`. */
function region(scene: SceneDescription, id: string, lo: readonly number[], hi: readonly number[], cells: 0.5 | 0.25): FluidRefinementRegion {
  const { cellSize_m: h, origin_m: o } = refinementRegionLattice(scene);
  const at = (v: readonly number[]) => ({ x: o.x + v[0]! * h[0]!, y: o.y + v[1]! * h[1]!, z: o.z + v[2]! * h[2]! });
  return { id, rule: "minimum-cell-size", minimumCellSize_cells: cells, maximumCellSize_cells: cells, min_m: at(lo), max_m: at(hi) };
}

test("automatic detail: one box about the teaser's rods, none when off", () => {
  assert.equal(specs(teaser()).length, 0, "no automatic detail, no regions: no engine");
  const [box, ...rest] = specs(teaser({ automaticDetail: { factor: 2 } }));
  assert.equal(rest.length, 0, "the rods are one merged box");
  const cells = sceneLatticeDimensions(teaser()) as unknown as Vec3;
  assert.equal(box!.box.factor, 2);
  assert.ok(box!.box.origins.every(o => o.startsWith("auto:")));
  for (let a = 0; a < 3; a++) {
    assert.equal(box!.box.lo[a]! % 4, 0, "whole tiles");
    assert.ok(box!.box.hi[a]! <= cells[a]!, "inside the container");
  }
  assert.equal(box!.box.lo[1], 0, "the box reaches the floor (the rods stand on it)");
  assert.equal(box!.layout.coupled, detailCoupledFaces(box!.box, cells));
  assert.equal(box!.layout.coupled & 0b100, 0, "the floor is the container's wall, not coupled");
  assert.ok(Math.abs(box!.setup.h_m * 2 - refinementRegionLattice(teaser()).cellSize_m[0]!) < 1e-12, "fine h is h / 2");
});

test("automatic detail: a still tank has no energy to splash", () => {
  const scene = teaser({ automaticDetail: { factor: 2 } }), cells = sceneLatticeDimensions(scene) as unknown as Vec3;
  assert.equal(automaticDetailBoxes(scene, cells, { top: 40, rest: 40, sources: false }).length, 0);
  assert.ok(automaticDetailBoxes(scene, cells, { top: 120, rest: 40, sources: false }).length > 0, "raised liquid splashes");
  assert.ok(automaticDetailBoxes(scene, cells, { top: 40, rest: 40, sources: true }).length > 0, "a source adds energy");
});

test("authored detail regions: tiles, merging, the high face's padding", () => {
  const base = teaser(), cells = sceneLatticeDimensions(base) as unknown as Vec3, reach = { top: 0, rest: 0, sources: false };
  // Two overlapping regions merge: the finer factor, both origins.
  const merged = brickDetailBoxes({ ...base, fluid: { ...base.fluid, refinementRegions: [
    region(base, "a", [41, 2, 10], [70, 30, 50], 0.5), region(base, "b", [60, 2, 10], [90, 30, 50], 0.25)] } }, cells, reach);
  assert.equal(merged.length, 1);
  assert.equal(merged[0]!.factor, 4);
  assert.deepEqual([...merged[0]!.origins].sort(), ["a", "b"]);
  assert.equal(merged[0]!.lo[0], 40, "snapped outward to a tile");
  // A fine extent of whole 32-cell pages on a coupled high face grows by a tile, so its storage brick exists.
  const [padded] = brickDetailBoxes({ ...base, fluid: { ...base.fluid, refinementRegions: [region(base, "c", [32, 0, 32], [64, 32, 64], 0.5)] } }, cells, reach);
  for (let a = 0; a < 3; a++) {
    if (padded!.hi[a]! >= cells[a]!) continue;
    assert.notEqual((2 * (padded!.hi[a]! - padded!.lo[a]!)) % 32, 0, `axis ${a}: no whole-page extent on a coupled high face`);
  }
  assert.equal(padded!.hi[0], 68);
});

test("the box's scene: the box at h / factor, the parent's patches translated and scaled", () => {
  const scene = teaser({ automaticDetail: { factor: 2 } }), [spec] = specs(scene), box = spec!.box;
  const sub = brickDetailSubScene(scene, box), h = refinementRegionLattice(scene).cellSize_m[0]!;
  assert.ok(Math.abs(sub.container.width_m - (box.hi[0] - box.lo[0]) * h) < 1e-9);
  assert.ok(Math.abs(sub.voxelDomain.finestCellSize_m - h / 2) < 1e-12);
  assert.equal(sub.fluid.automaticDetail, undefined);
  assert.equal(sub.fluid.refinementRegions, undefined);
  assert.equal(sub.fluid.brickPools, undefined);
  assert.equal(sub.container.fillFraction, 0);
  assert.ok((sub.solidVoxels?.length ?? 0) > 0, "the rods reach the box");
  for (const p of sub.solidVoxels!) for (let a = 0; a < 3; a++) assert.ok(p.maximumExclusive[a]! > 0 && p.minimum[a]! < 2 * (box.hi[a]! - box.lo[a]!));
  assert.deepEqual(brickDetailSceneGaps(scene, box), []);
  const gapped = { ...scene, container: { ...scene.container, walls: "none" as const } };
  assert.ok(brickDetailSceneGaps(gapped, box).some(g => /open world/.test(g)));
  assert.throws(() => brickDetailSubScene(gapped, box), /open world/);
});

test("the shell's slab index is a bijection onto its nodes", () => {
  const d = { nodes: [2 * DETAIL_DEPTH + 3, 2 * DETAIL_DEPTH + 1, 2 * DETAIL_DEPTH + 5] as const };
  const n = brickDetailNodes(d), seen = new Set<number>();
  let inside = 0;
  for (let z = 0; z < d.nodes[2]; z++) for (let y = 0; y < d.nodes[1]; y++) for (let x = 0; x < d.nodes[0]; x++) {
    const i = brickDetailIndex(d, [x, y, z]);
    const inner = [x, y, z].every((v, a) => v >= DETAIL_DEPTH && v < d.nodes[a]! - DETAIL_DEPTH);
    if (inner) { assert.equal(i, -1); inside++; continue; }
    assert.ok(i >= 0 && i < n, `node ${x},${y},${z}`);
    assert.ok(!seen.has(i), `index ${i} twice`);
    seen.add(i);
  }
  assert.equal(seen.size, n);
  assert.equal(inside + n, d.nodes[0] * d.nodes[1] * d.nodes[2]);
  assert.throws(() => brickDetailNodes({ nodes: [2 * DETAIL_DEPTH, 40, 40] }), /under/);
});

test("the parent's shader text is unchanged without a detail layout", () => {
  const L = { box: [64, 64, 64], lattice: [64, 64, 64] } as unknown as BrickLayout;
  assert.equal(brickSourceWGSL(L), BRICK_SOURCE_WGSL);
  assert.throws(() => brickDetailWGSL(L), /not a detail layout/);
  const D = { ...L, detail: { factor: 2, nodes: [35, 35, 35], coupled: 0b11 } } as unknown as BrickLayout;
  const text = brickSourceWGSL(D);
  assert.match(text, /fn nbSourcePhi\(p:vec3f,value:f32\)->f32\{return nbDetailPhi\(p,nbSourcePhiCore\(p,value\)\);\}/);
  assert.match(brickDetailWGSL(D), /const NB_DTL_CLO:vec3<bool>=vec3<bool>\(true,false,false\)/);
});

test("the fine surface: the container's walls closed, the coupled faces open", () => {
  const text = brickDetailSurfaceWGSL({ cells: [200, 272, 368], bricks: [28, 36, 48], h_m: 0.01, origin_m: [0, 0, 0], coupled: 59, far_m: 1 });
  assert.match(text, /DS_CLOSED_LO:vec3<bool>=vec3<bool>\(false,true,false\)/, "the floor is closed");
  assert.match(text, /DS_CLOSED_HI:vec3<bool>=vec3<bool>\(false,false,false\)/);
  assert.match(text, /DS_GROUPS:vec3i=vec3i\(26,35,47\)/);
  assert.equal(brickDetailSurfaceCapacity([200, 272, 368], 2 ** 40) % 3, 0, "whole triangles");
  assert.equal(brickDetailSurfaceCapacity([200, 272, 368], 32 * 300), 300, "held to the device's binding");
});

test("the scene document: validation, the editor's tiers, re-seeding, refusal", () => {
  const scene = teaser();
  assert.deepEqual(validateScene(teaser({ automaticDetail: { factor: 2 } })), validateScene(scene));
  assert.ok(validateScene(teaser({ automaticDetail: { factor: 3 as 2 } })).some(e => /Automatic detail factor/.test(e)));
  const r = region(scene, "d", [40, 0, 40], [80, 40, 80], 0.5);
  assert.deepEqual(validateRefinementRegions([r], scene.container), []);
  const tiers = studioRegionSpaceForMethod(METHOD).heldTiers!.map(t => t.cells);
  assert.ok(tiers.includes(0.5) && tiers.includes(0.25), "the editor offers the detail tiers");
  assert.ok(studioRegionSpaceForMethod(METHOD).cellSizes.includes(0.5));
  const back = refinementRegionFromRecord(scene, refinementRegionRecord(scene, r));
  assert.equal(back.minimumCellSize_cells, 0.5);
  for (const k of ["x", "y", "z"] as const) assert.ok(Math.abs(back.min_m[k] - r.min_m[k]) < 1e-9 && Math.abs(back.max_m[k] - r.max_m[k]) < 1e-9);
  // Detail engines are built with the solver: a detail change re-seeds; an ordinary region stays a live edit.
  assert.ok(sceneEditRequiresReset(scene, teaser({ automaticDetail: { factor: 2 } }), METHOD));
  assert.ok(sceneEditRequiresReset(scene, teaser({ refinementRegions: [r] }), METHOD));
  assert.ok(!sceneEditRequiresReset(scene, teaser({ refinementRegions: [{ ...r, minimumCellSize_cells: 1, maximumCellSize_cells: 1 }] }), METHOD));
  assert.equal(detailRegionRefusal(scene, "x"), undefined);
  assert.match(detailRegionRefusal(teaser({ refinementRegions: [r] }), "The uniform engine")!, /only the brick NB-FLIP engine/);
  assert.match(detailRegionRefusal(teaser({ automaticDetail: { factor: 2 } }), "The uniform engine")!, /automatic detail/);
});
