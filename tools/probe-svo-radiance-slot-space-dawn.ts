#!/usr/bin/env node
/**
 * Why a scene falls back to exact SVO visibility at the *first* publication.
 *
 * The withdrawal this lane is about is not the one
 * `tools/run-svo-live-voxelization-smoke.ts` covers. That one happens on a
 * later edit, when a page outside the plan is activated. This one happens
 * before a single frame: `WebGpuLiveSvoTetrahedralRadiance`'s constructor
 * rejects the address plan it was handed, the whole derived-lighting block
 * unwinds, and the render panel reads
 *
 *   Lighting visibility: EXACT FALLBACK · Live radiance slot offset must
 *   leave its atlas inside the slot index space
 *
 * The radiance atlas rides the opacity pyramid's slot numbering: slots below
 * the radiance floor own no radiance page, so the atlas is the suffix of the
 * slot space shifted down by `radianceSlotOffset`. The atlas therefore has to
 * fit in `pageCapacity - radianceSlotOffset` slots. It is sized from the
 * *domain* above the floor, while the slot space is sized from today's
 * occupancy plus a fixed reserve — two different quantities, and on a partial
 * (non-`total`) plan with the floor above level zero the domain figure wins and
 * overflows the space.
 *
 * That combination needs a lattice fine enough to put the radiance floor above
 * zero and a domain too large to plan totally. `twin-dam-collision` at its
 * authored 50 mm has the floor at level 0 and never trips; the same scene at
 * 25 mm does, which is the sweep this lane runs.
 *
 * Environment
 *   WEBGPU_NODE_MODULE           path to the Dawn node module
 *   FLUID_SVO_SLOT_SPACE_SCENE   scene preset id (default twin-dam-collision)
 *   FLUID_SVO_SLOT_SPACE_CELLS   comma-separated cell sizes in metres
 *                                (default 0.05,0.025,0.0125)
 *   FLUID_SVO_SLOT_SPACE_OUT     optional JSON report path
 */
// A renderer resolves a method by id on any path that reaches a scene, even
// when — as here — no solver is ever created.
import "../lib/methods";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { getScenePreset } from "../lib/core/scenes";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import type { OctreeSparseBrickWorld } from "../lib/svo/webgpu-svo-sparse-bricks";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";

const scenePresetId = process.env.FLUID_SVO_SLOT_SPACE_SCENE ?? "twin-dam-collision";
const cellSizes = (process.env.FLUID_SVO_SLOT_SPACE_CELLS ?? "0.05,0.025,0.0125")
  .split(",").map((value) => Number(value.trim())).filter((value) => value > 0);
const outPath = process.env.FLUID_SVO_SLOT_SPACE_OUT;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

const log = (message: string) => process.stderr.write(`${message}\n`);

const { device, adapterInfo } = await createDawnRenderDevice({
  modulePath, label: "SVO radiance slot-space probe",
});
log(`Adapter: ${JSON.stringify(adapterInfo)}`);

interface Rung {
  cellSize_m: number;
  lattice: readonly [number, number, number];
  state: string;
  reason?: string;
  detail?: string;
  radianceFloorLevel?: number;
  radianceSlotOffset?: number;
  radiancePageCapacity?: number;
  pageCapacity?: number;
  total?: boolean;
  activePages?: number;
}

const rungs: Rung[] = [];
for (const cellSize_m of cellSizes) {
  const scene = getScenePreset(scenePresetId).create();
  scene.voxelDomain = { ...scene.voxelDomain, finestCellSize_m: cellSize_m };
  const solver = await WebGPULiveSvoScene.create(device, scene, "balanced", () => {});
  // The address plan is the world's, and the world is private. This lane is a
  // diagnostic and reaches for it deliberately.
  const world = (solver as unknown as { world: OctreeSparseBrickWorld }).world;
  const source = solver.sparseVoxelSceneSource;
  const plan = (world as unknown as {
    liveDerivedAddressPlan?: {
      radianceFloorLevel: number; radianceSlotOffset: number; radiancePageCapacity: number;
      pageCapacity: number; total: boolean; plan: { pages: readonly unknown[] };
    };
  }).liveDerivedAddressPlan;
  const rung: Rung = {
    cellSize_m,
    lattice: source.structural?.domain.dimensionsCells ?? [0, 0, 0],
    state: source.derivedLighting?.state ?? "absent",
    reason: source.derivedLighting?.reason,
    detail: source.derivedLighting?.detail,
    radianceFloorLevel: plan?.radianceFloorLevel,
    radianceSlotOffset: plan?.radianceSlotOffset,
    radiancePageCapacity: plan?.radiancePageCapacity,
    pageCapacity: plan?.pageCapacity,
    total: plan?.total,
    activePages: plan?.plan.pages.length,
  };
  rungs.push(rung);
  log(`  ${(cellSize_m * 1000).toFixed(2)} mm  ${rung.lattice.join("x")}`
    + `  floor=${rung.radianceFloorLevel} offset=${rung.radianceSlotOffset}`
    + ` atlas=${rung.radiancePageCapacity} capacity=${rung.pageCapacity}`
    + ` total=${rung.total}  -> ${rung.state}${rung.detail ? ` (${rung.detail})` : ""}`);
  solver.destroy();
}

const withdrawn = rungs.filter((rung) => rung.state === "unavailable");
const report = { lane: "svo-radiance-slot-space", scene: scenePresetId, adapter: adapterInfo, rungs,
  state: withdrawn.length === 0 ? "pass" : "fail" };
if (outPath) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
}
process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
if (withdrawn.length > 0) {
  log(`${withdrawn.length} rung(s) withdrew derived lighting: `
    + withdrawn.map((rung) => `${(rung.cellSize_m * 1000).toFixed(2)} mm`).join(", "));
}
// Explicit, like every other Dawn lane here: the retained Dawn instance keeps
// the loop alive, and a lane that never returns holds the exclusive GPU lock
// for every run after it.
process.exit(withdrawn.length > 0 ? 1 : 0);
