#!/usr/bin/env node
/**
 * The W2 gate: ABI-true incremental voxelization.
 *
 *   npm run test:webgpu:svo-live-voxelization
 *   FLUID_SVO_VOXELIZATION_SCENE=<preset> node --import tsx tools/run-webgpu-exclusive.ts \
 *     --import tsx tools/run-svo-live-voxelization-smoke.ts
 *
 * `tools/run-svo-dry-render-smoke.ts` proves one *published* frame is healthy.
 * It cannot prove anything about editing, because it publishes once and never
 * moves a record — and every failure this workstream is about happens on the
 * second publication:
 *
 *   1. **The address plan.** A scene edit that activated a derived page outside
 *      the plan fixed at first publish used to set `liveDerivedAddressPlanValid`
 *      false, null the opacity pyramid and the radiance atlas, and drop cone
 *      visibility onto exact traversal — roughly a 15x frame cost, reported in
 *      a status string and in no console output at all. Incremental
 *      voxelization activates pages continuously, so it trips by construction.
 *      This lane runs a real edit script and fails if the pyramid ever
 *      withdraws, on the source object, the plan, and the warning stream.
 *
 *   2. **Steady state.** Voxelization is an accelerator; a scene nobody is
 *      editing must cost nothing. The lane counts the voxelizer's own compute
 *      passes across still frames and requires zero.
 *
 *   3. **The per-frame budget.** A publication used to voxelize in one go, so
 *      one teleport was one frame however many bricks it touched. It is now
 *      chunked, and the danger in chunking is the completion contract: a
 *      half-repaired publication must never be observable as current, because
 *      under-reporting coverage is what once painted a black slab across the
 *      garden. The lane reads `requestedRevision` and `completedRevision`
 *      straight off the maintenance arena on every frame of a convergence and
 *      requires them to differ until the last chunk lands.
 *
 *   4. **The bonsai crown.** The voxelizer used to downgrade a
 *      `smooth-union-cluster` to its solid envelope ellipsoid, so the crown —
 *      65 % of the hero frame's distance-function evaluations — was a blob in
 *      voxel space, and therefore in every shadow and GI ray that reads the
 *      opacity pyramid. The lane reads the published scene material lane back
 *      off the GPU and measures what fraction of the crown's envelope is
 *      actually solid. A blob is 1.0.
 *
 * Everything before the device is a fixture check, so a layout drift fails
 * before a lock is taken.
 *
 * Environment
 *   WEBGPU_NODE_MODULE                     path to the Dawn node module
 *   FLUID_SVO_VOXELIZATION_SCENE           scene preset id (default hero-garden-hose)
 *   FLUID_SVO_VOXELIZATION_STILL_FRAMES    still frames sampled (default 3)
 *   FLUID_SVO_VOXELIZATION_CROWN_AIR_FLOOR minimum air fraction inside the crown envelope (default 0.15)
 *   FLUID_SVO_VOXELIZATION_BRICK_BUDGET    bricks repaired per frame (this lane defaults to 2 so the budget bites)
 *   FLUID_SVO_VOXELIZATION_CONVERGE_FRAMES authored ceiling on frames one teleport may take (default 64)
 *   FLUID_SVO_VOXELIZATION_OUT             optional JSON report path
 */
// These lanes render without a solver, but they construct the renderer, and
// a renderer resolves a method by id on any path that reaches a scene.
import "../lib/methods";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { SCENE_ENVIRONMENT_OWNER_BASE } from "../lib/core/webgpu-rigid-body";
import { getScenePreset } from "../lib/core/scenes";
import { SVO_CLUSTER_ARENA_BLOCK, packSvoClusterArena } from "../lib/svo/svo-cluster-arena";
import {
  growSvoNodeMipAddressPlan,
  pagesOutsideSvoNodeMipAddressPlan,
  planSvoNodeMipAddresses,
  svoNodeMipDomainBasePages,
  svoNodeMipDomainPyramidPageCount,
} from "../lib/svo/svo-node-mip-address-plan";
import type { SvoNodeMipCoordinate } from "../lib/svo/svo-node-mip-pyramid";
import { SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS, type SvoSmoothUnionClusterPacking } from "../lib/svo/svo-primitive-abi";
import {
  SPARSE_BRICK_PAYLOAD_PROFILES, sparseBrickScenePayloadIdentityAt, unpackMaterialOwner,
} from "../lib/svo/sparse-brick-octree";
import { packSvoDrySceneClusters } from "../lib/svo/webgpu-svo-dry-scene";
import { WebGPULiveSvoScene } from "../lib/svo/webgpu-live-svo-scene";
import {
  ENVIRONMENT_VOXEL_MATERIAL_BASE,
  OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET,
  octreeLiveSceneDryPayloadProfile,
  type OctreeSparseBrickWorld,
} from "../lib/svo/webgpu-svo-sparse-bricks";
import {
  SPARSE_SCENE_MAINTENANCE_STATE_WORDS,
  sparseScenePrimitiveBounds,
  sparseScenePrimitiveForProxy,
  sparseSceneRevisionIncomplete,
  type SparseScenePrimitive,
} from "../lib/core/webgpu-sparse-scene-proxies";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/core/voxel-environments";
import { createDawnRenderDevice } from "./svo-dry-frame-harness";

const scenePresetId = process.env.FLUID_SVO_VOXELIZATION_SCENE ?? "hero-garden-hose";
const stillFrames = Number(process.env.FLUID_SVO_VOXELIZATION_STILL_FRAMES ?? 3);
const crownAirFloor = Number(process.env.FLUID_SVO_VOXELIZATION_CROWN_AIR_FLOOR ?? 0.15);
// Deliberately far below the production default. A budget the hero garden never
// reaches proves nothing about budgeting: a teleported record here dirties eight
// to twelve bricks, so two a frame is what makes every teleport in the script
// below take several, which is the state the completion contract has to survive.
// Set before the world is built: the world reads it.
process.env.FLUID_SVO_VOXELIZATION_BRICK_BUDGET ??= "2";
const brickBudget = Number(process.env.FLUID_SVO_VOXELIZATION_BRICK_BUDGET);
const convergenceFrameCeiling = Number(process.env.FLUID_SVO_VOXELIZATION_CONVERGE_FRAMES ?? 64);
const outPath = process.env.FLUID_SVO_VOXELIZATION_OUT;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

const log = (message: string) => process.stderr.write(`${message}\n`);

interface Check { id: string; state: "pass" | "fail"; detail: string; measured?: unknown; limit?: unknown }
const checks: Check[] = [];
function record(id: string, ok: boolean, detail: string, measured?: unknown, limit?: unknown): boolean {
  checks.push({ id, state: ok ? "pass" : "fail", detail, measured, limit });
  log(`  [${ok ? "pass" : "FAIL"}] ${id}: ${detail}`);
  return ok;
}

// The one warning path the address plan still has, plus the older one. Captured
// from before the world is built, exactly as the dry-render lane does.
const capturedWarnings: string[] = [];
const originalWarn = console.warn.bind(console);
console.warn = (...args: unknown[]) => {
  capturedWarnings.push(args.map((value) => (value instanceof Error ? value.message : String(value))).join(" "));
  originalWarn(...args);
};

// ---------------------------------------------------------------------------
// (A) Fixtures. No device: a layout drift should fail before the GPU lock.
// ---------------------------------------------------------------------------
/**
 * The live voxelizer packs aggregate blocks itself, against its own maintenance
 * arena, because it is driven by keyed live updates rather than by a whole
 * scene publication. Two transcriptions of one layout is the exact failure the
 * primitive ABI keeps warning about, so the agreement is executed here.
 */
const clusterFixtures: readonly SvoSmoothUnionClusterPacking[] = [
  { field: "lattice", seed: 0x51ed5eed, smoothRadius_m: 0.012, latticeLobeRadius_m: 0.03, latticePeriod_m: 0.08, jitter: 0.35, octaves: 2 },
  { field: "seeded-lobes", seed: 0x1234abcd, smoothRadius_m: 0.02, lobeCount: 11, anisotropy: 1.7, lobeSpan: 0.41, lobeSpanSpread: 0.19, displacement: 0.72 },
  {
    field: "tapered-sweep", seed: 7, smoothRadius_m: 0.006,
    points: [
      { position_m: { x: 0, y: -0.4, z: 0 }, radius_m: 0.05 },
      { position_m: { x: 0.03, y: 0, z: 0.01 }, radius_m: 0.035 },
      { position_m: { x: 0.05, y: 0.4, z: -0.02 }, radius_m: 0.012 },
    ],
  },
];
const arenaMatches = clusterFixtures.every((packing) => {
  const mine = packSvoClusterArena([packing], 1);
  const theirs = packSvoDrySceneClusters([packing]).subarray(0, SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS);
  return mine.length === theirs.length && mine.every((word, index) => word === theirs[index]);
});
record("cluster-arena-layout", arenaMatches,
  arenaMatches
    ? `live and dry-scene aggregate blocks are byte-identical across ${clusterFixtures.length} fields`
    + ` (${SVO_CLUSTER_ARENA_BLOCK.strideWords}-word slots)`
    : "live voxelizer aggregate blocks have drifted from the renderer's; every cluster would voxelize as a different field");

/**
 * The address plan, on the hero garden's own page grid.
 *
 * `total` is the shape that discharges the gate outright: every base page the
 * domain can hold already has a slot, so no edit can name one outside the plan
 * and the withdrawal branch is unreachable. The growth branch is exercised
 * against a deliberately starved reserve, because a plan that fits will never
 * take it and an untested fallback is not a fallback.
 */
const heroBasePages = [12, 4, 8] as [number, number, number];
const heroLevels = 5;
const heroDomainPages = svoNodeMipDomainPyramidPageCount(heroBasePages, heroLevels);
const occupiedSubset = svoNodeMipDomainBasePages(heroBasePages)
  .filter(([x, y, z]) => y < 2 || (x + z) % 2 === 0) as SvoNodeMipCoordinate[];
const totalPlan = planSvoNodeMipAddresses({
  occupiedBasePages: occupiedSubset, basePageDimensions: heroBasePages,
  levelCount: heroLevels, addressCapacity: 134_217_728,
});
record("address-plan-total", totalPlan.total && totalPlan.plan.complete
  && pagesOutsideSvoNodeMipAddressPlan(totalPlan, svoNodeMipDomainBasePages(heroBasePages)).length === 0,
  `a ${heroBasePages.join("x")} page domain plans totally at ${totalPlan.plan.pages.length}/${heroDomainPages} pages`
  + ` (${totalPlan.reservePages} beyond the ${totalPlan.plan.pages.length - totalPlan.reservePages} occupied);`
  + " no edit can activate a page outside it",
  { total: totalPlan.total, pages: totalPlan.plan.pages.length }, heroDomainPages);

const starved = planSvoNodeMipAddresses({
  occupiedBasePages: occupiedSubset, basePageDimensions: heroBasePages, levelCount: heroLevels,
  addressCapacity: 134_217_728, maximumReserveBytes: 40 * 72_000,
});
const starvedMissing = pagesOutsideSvoNodeMipAddressPlan(starved, svoNodeMipDomainBasePages(heroBasePages));
const grown = growSvoNodeMipAddressPlan(starved, starvedMissing.slice(0, 3));
const refused = growSvoNodeMipAddressPlan(starved, starvedMissing);
const growthSound = !starved.total && grown !== undefined && grown.plan.plan.complete
  && grown.addedBasePages.length === 3 && grown.rebuildRequired
  && grown.plan.atlasPages.join(",") === starved.atlasPages.join(",")
  && grown.plan.pageCapacity === starved.pageCapacity
  && refused === undefined;
record("address-plan-growth", growthSound,
  growthSound
    ? `a starved plan (${starved.pageCapacity} pages, ${starved.reservePages} reserved) grows in place without moving its`
    + ` atlas, and refuses the ${starvedMissing.length}-page request it cannot hold`
    : "growth into the reserve is unsound: it either reshaped the atlas, produced an incomplete plan, or accepted an overflow");

// ---------------------------------------------------------------------------
// The scene, and what its aggregates now voxelize as.
// ---------------------------------------------------------------------------
const scene = getScenePreset(scenePresetId).create();
const environmentId = (scene.environment ?? "default") as Parameters<typeof buildEnvironmentProxyCatalog>[1];
const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
const proxies = environmentProxyPrimitives(catalog, true);
const ownerFor = (ownerIndex: number) => SCENE_ENVIRONMENT_OWNER_BASE + ownerIndex;
const clusterProxies = proxies.filter((proxy) => proxy.kind === "cluster");
const crownProxies = clusterProxies.filter((proxy) => proxy.tags.includes("crown") || proxy.key.includes("crown"));
const censusProxies = crownProxies.length > 0 ? crownProxies : clusterProxies;

const voxelized: SparseScenePrimitive[] = proxies.map((proxy) =>
  sparseScenePrimitiveForProxy(proxy, {
    materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex, ownerId: ownerFor(proxy.ownerIndex),
  }));
const abiTrue = voxelized.filter((primitive) =>
  primitive.kind === "smooth-union-cluster" || primitive.kind === "round-cone" || primitive.kind === "rounded-cylinder");
record("abi-true-vocabulary", clusterProxies.length === 0
  || voxelized.filter((primitive) => primitive.kind === "smooth-union-cluster").length === clusterProxies.length,
  `${abiTrue.length} of ${voxelized.length} live primitives now voxelize through the render ABI`
  + ` (${clusterProxies.length} aggregates, ${abiTrue.length - voxelized.filter((p) => p.kind === "smooth-union-cluster").length} tapered/filleted);`
  + " an aggregate that fell back to its envelope would be counted as a plain ellipsoid here",
  abiTrue.length, voxelized.length);

/**
 * The swap must not move a single dirty region.
 *
 * Every bound in the tree — invalidation, binning, topology activation — reads
 * a primitive's conservative extent, and the render ABI clips an aggregate's
 * field to exactly the envelope the old downgrade used. So the ABI-true
 * primitive and the shape it replaced must bound identically. If they did not,
 * this change would be a topology change wearing a distance function's clothes,
 * and the address-plan result above would be measuring something else.
 *
 * The comparison is against the *previous adapter's own output*, reconstructed
 * here, rather than against the authored AABB: the authored AABB is a rotated
 * box around a local extent and the two agree only up to the rotation bound, so
 * comparing to it would test the quaternion algebra rather than the swap.
 */
function legacyVoxelizedProxy(proxy: (typeof proxies)[number], identity: { materialId: number; ownerId: number }): SparseScenePrimitive {
  const center: [number, number, number] = [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z];
  const orientation = proxy.orientation
    ? ([proxy.orientation.x, proxy.orientation.y, proxy.orientation.z, proxy.orientation.w] as [number, number, number, number])
    : undefined;
  const base = { ...identity, center, orientation };
  if (proxy.kind === "cone") {
    // Was a truncated cone whatever the ends were; a round cone's own extent is
    // the same box because the endpoint spheres are inside the authored height.
    return { ...base, kind: "cone", baseRadius: proxy.baseRadius_m, topRadius: proxy.topRadius_m, halfHeight: proxy.halfHeight_m };
  }
  if (proxy.kind === "cylinder") return { ...base, kind: "cylinder", radius: proxy.radius_m, halfHeight: proxy.halfHeight_m };
  if (proxy.kind === "cluster") return { ...base, kind: "ellipsoid", radii: [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] };
  return sparseScenePrimitiveForProxy(proxy, identity);
}
const boundsDrift = voxelized.map((primitive, index) => {
  const proxy = proxies[index];
  const identity = { materialId: ENVIRONMENT_VOXEL_MATERIAL_BASE + proxy.ownerIndex, ownerId: ownerFor(proxy.ownerIndex) };
  const now = sparseScenePrimitiveBounds(primitive);
  const legacy = sparseScenePrimitiveBounds(legacyVoxelizedProxy(proxy, identity));
  const axes = [0, 1, 2];
  // Signed, because the two directions are not the same fact. A widened bound
  // dirties more bricks and bins more candidates than it needs to, which costs
  // maintenance and nothing else. A narrowed one silently drops geometry out of
  // invalidation and out of the opacity pyramid.
  const shrink = Math.max(...axes.map((axis) =>
    Math.max(now.minimum[axis] - legacy.minimum[axis], legacy.maximum[axis] - now.maximum[axis])));
  const grow = Math.max(...axes.map((axis) =>
    Math.max(legacy.minimum[axis] - now.minimum[axis], now.maximum[axis] - legacy.maximum[axis])));
  const authoredGap = Math.max(...(["x", "y", "z"] as const).map((axis, component) =>
    Math.max(proxy.aabb_m.min[axis] - now.minimum[component], now.maximum[component] - proxy.aabb_m.max[axis])));
  return { key: proxy.key, kind: primitive.kind, shrink, grow, authoredGap };
});
const narrowed = boundsDrift.filter(({ shrink }) => shrink > 1e-9);
const widened = boundsDrift.filter(({ grow }) => grow > 1e-9);
/**
 * The swap must not narrow a single dirty region.
 *
 * Every bound in the tree — invalidation, binning, topology activation — reads a
 * primitive's conservative extent. An aggregate is bounded by the envelope its
 * field is clipped to, so it does not move at all; a filleted cylinder is
 * bounded by its outer, so it does not either. A round cone *does*: its
 * predecessor here voxelized as a truncated `cone`, whose extent stops at the
 * authored half height and therefore cut the swept endpoint spheres out of
 * voxel space entirely. Widening is the correction, and the widened bound is
 * the authored AABB to the last bit.
 */
record("voxel-bounds-never-narrow", narrowed.length === 0,
  narrowed.length === 0
    ? `no voxelized primitive's bound narrowed; ${widened.length} widened`
    + (widened.length > 0
      ? ` (worst ${(1000 * Math.max(...widened.map(({ grow }) => grow))).toFixed(1)} mm on ${widened[0].kind}`
      + " — the truncated-cone downgrade was cutting the swept caps out of voxel space)"
      : "")
    : `${narrowed.length} primitives narrowed, worst ${(1000 * Math.max(...narrowed.map(({ shrink }) => shrink))).toFixed(3)} mm`
    + ` (e.g. ${narrowed[0].kind} ${narrowed[0].key}); geometry has silently left invalidation and the opacity pyramid`,
  narrowed.length, 0);
const cappedKinds = new Set(["round-cone", "rounded-cylinder"]);
const cappedGap = boundsDrift.filter(({ kind }) => cappedKinds.has(kind));
record("capped-kinds-match-authored-aabb", cappedGap.every(({ authoredGap }) => Math.abs(authoredGap) <= 1e-9),
  `${cappedGap.length} tapered/filleted primitives bound exactly to their authored AABB`
  + ` (worst deviation ${(1000 * Math.max(0, ...cappedGap.map(({ authoredGap }) => Math.abs(authoredGap)))).toFixed(4)} mm)`,
  cappedGap.length, 0);

// ---------------------------------------------------------------------------
// (B) The device.
// ---------------------------------------------------------------------------
const { adapterInfo, device, validationErrors } = await createDawnRenderDevice({
  modulePath, label: "SVO live voxelization smoke",
});
log(`Adapter: ${JSON.stringify(adapterInfo)}`);
log(`Scene ${scenePresetId}: ${proxies.length} proxies, ${clusterProxies.length} aggregates, census over ${censusProxies.length}`);

const solver = await WebGPULiveSvoScene.create(device, scene, "balanced",
  ({ label, completed, total }) => log(`  [world] ${label} (${completed}/${total})`));
// The private world carries the address-plan status. This lane is a diagnostic
// and reaches for it deliberately; `WebGPULiveSvoScene` is another workstream's
// file and gets a handoff item rather than a getter added behind its back.
const world = (solver as unknown as { world: OctreeSparseBrickWorld }).world;

/** Compute passes the live-scene voxelizer encodes, by label. */
const VOXELIZER_PASS_LABELS = new Set([
  "Invalidate live scene dirty bricks",
  "Prepare live scene maintenance dispatches",
  "Bin live scene primitives into dirty bricks",
  "Rebuild live scene dirty brick payloads",
  "Finalize live scene dirty bricks",
]);

function countingEncoder(label: string): { encoder: GPUCommandEncoder; passes: string[] } {
  const raw = device.createCommandEncoder({ label });
  const passes: string[] = [];
  const encoder = new Proxy(raw, {
    get(target, property, receiver) {
      const value = Reflect.get(target, property, receiver);
      if (typeof value !== "function") return value;
      const bound = (value as (...args: unknown[]) => unknown).bind(target);
      if (property !== "beginComputePass") return bound;
      return (descriptor?: GPUComputePassDescriptor) => {
        if (descriptor?.label) passes.push(descriptor.label);
        return bound(descriptor);
      };
    },
  }) as GPUCommandEncoder;
  return { encoder, passes };
}

function submitMaintenance(label: string): { encoded: boolean; voxelizerPasses: number; passes: string[] } {
  const { encoder, passes } = countingEncoder(label);
  const encoded = solver.encodeSceneMaintenance(encoder);
  device.queue.submit([encoder.finish()]);
  return { encoded, voxelizerPasses: passes.filter((name) => VOXELIZER_PASS_LABELS.has(name)).length, passes };
}

async function readBuffer(buffer: GPUBuffer, offset: number, size: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ label: "Voxelization census readback", size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder({ label: "Voxelization census copy" });
  encoder.copyBufferToBuffer(buffer, offset, readback, 0, size);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  await readback.mapAsync(GPUMapMode.READ);
  const copy = new Uint32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return copy;
}

/**
 * The completion contract, read off the device.
 *
 * `finalizeScene` withholds the live scene generation unless
 * `completedRevision == requestedRevision` and no incomplete overflow is
 * raised, so these three words *are* the statement that a publication is
 * observable. A budgeted publication must fail that test on every frame but its
 * last; asserting the CPU's own chunk arithmetic instead would prove nothing.
 */
async function maintenanceState(): Promise<{
  dirtyBricks: number; overflow: number; requested: number; completed: number;
  chunkBegin: number; chunkEnd: number; cursor: number; maximumBrickCandidates: number; complete: boolean;
}> {
  const binding = world.sceneMaintenanceBinding;
  const words = await readBuffer(binding.buffer, binding.stateOffsetBytes,
    SPARSE_SCENE_MAINTENANCE_STATE_WORDS.wordCount * 4);
  const requested = words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.requestedRevision];
  const completed = words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.completedRevision];
  const overflow = words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.overflowFlags];
  return {
    dirtyBricks: words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.dirtyBrickCount],
    overflow, requested, completed,
    chunkBegin: words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkBegin],
    chunkEnd: words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkEnd],
    cursor: words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.chunkCursor],
    maximumBrickCandidates: words[SPARSE_SCENE_MAINTENANCE_STATE_WORDS.maximumBrickCandidates],
    complete: requested !== 0 && completed === requested && !sparseSceneRevisionIncomplete(overflow),
  };
}

const initial = submitMaintenance("Live voxelization initial publication");
await device.queue.onSubmittedWorkDone();
record("initial-publication", initial.encoded && initial.voxelizerPasses === VOXELIZER_PASS_LABELS.size,
  `the first publication runs all ${VOXELIZER_PASS_LABELS.size} voxelizer passes (${initial.voxelizerPasses} seen)`,
  initial.voxelizerPasses, VOXELIZER_PASS_LABELS.size);

const initialState = await maintenanceState();
/**
 * The publication that brings the scene into existence is exempt from the
 * budget on purpose: nothing downstream has a generation to keep showing while
 * it converges, and a scene generation of zero is what the black slab is made
 * of. So it must complete in the one frame it was encoded in.
 */
record("initial-publication-completes-unbudgeted", initialState.complete,
  initialState.complete
    ? `the first publication repaired ${initialState.dirtyBricks} bricks in one frame and stored revision`
    + ` ${initialState.completed} (budget ${brickBudget} applies to edits only)`
    : `the first publication did not complete: requested=${initialState.requested} completed=${initialState.completed}`
    + ` overflow=${initialState.overflow}; every consumer is reading a scene generation of zero`,
  { requested: initialState.requested, completed: initialState.completed, dirtyBricks: initialState.dirtyBricks });

/**
 * Per-brick candidate density, which the plan's capacity table mis-states.
 *
 * It calls a per-brick candidate overflow "image-preserving", and at the one
 * brick over 64 the hero garden actually has, it nearly is. At ten times the
 * records the busiest brick binds hundreds, and the surplus is written nowhere:
 * those primitives leave the opacity pyramid and the radiance atlas while still
 * drawing in primary visibility, so the geometry keeps its silhouette and stops
 * casting a shadow. The check is the measured maximum against the arena, not
 * the overflow bit, because the bit cannot tell one lost primitive from most of
 * them.
 */
record("per-brick-candidate-density",
  initialState.maximumBrickCandidates > 0
  && initialState.maximumBrickCandidates <= OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK,
  `the busiest brick binds ${initialState.maximumBrickCandidates} primitives against`
  + ` ${OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK} slots`
  + ` (planner splits above ${OCTREE_LIVE_SCENE_REFINEMENT_CANDIDATE_TARGET}; the arena is the backstop for`
  + " where refinement runs out of depth)"
  + (initialState.maximumBrickCandidates > OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK
    ? " — the surplus is absent from every shadow and GI ray that reads the pyramid"
    : ""),
  initialState.maximumBrickCandidates, OCTREE_LIVE_SCENE_CANDIDATES_PER_BRICK);

// ---------------------------------------------------------------------------
// Steady state: a scene nobody edits must cost nothing.
// ---------------------------------------------------------------------------
let stillPasses = 0;
for (let frame = 0; frame < Math.max(1, stillFrames); frame += 1) {
  stillPasses += submitMaintenance(`Live voxelization still frame ${frame}`).voxelizerPasses;
}
await device.queue.onSubmittedWorkDone();
record("still-scene-zero-maintenance", stillPasses === 0,
  `${stillPasses} voxelizer compute passes across ${Math.max(1, stillFrames)} still frames`
  + " (the pyramid's bounded radiance-feedback window is deliberately excluded; it is convergence, not maintenance)",
  stillPasses, 0);

// ---------------------------------------------------------------------------
// The edit script. This is the check the workstream exists for.
// ---------------------------------------------------------------------------
function pyramidState() {
  const source = solver.sparseVoxelSceneSource;
  const nodeMip = source?.nodeMipPyramid;
  const radiance = source?.tetrahedralRadiance;
  return {
    healthy: source?.derivedLighting?.state !== "unavailable"
      && Boolean(nodeMip && radiance && nodeMip.generation > 0
        && nodeMip.generation === radiance.generation
        && nodeMip.plan.complete && radiance.plan.complete),
    pages: nodeMip?.plan.pages.length ?? 0,
    reason: source?.derivedLighting?.reason,
    detail: source?.derivedLighting?.detail,
    plan: world.liveDerivedAddressPlanStatus,
  };
}

const before = pyramidState();
/**
 * A record teleported the length of the domain, then to each far corner, then
 * home. Teleports rather than a smooth path on purpose: a smooth path stays
 * inside bricks that are already covered, and covered motion never touched the
 * address plan. What trips it is a *newly covered* brick, so the script goes
 * out of its way to activate them.
 */
const domain = solver.sparseVoxelSceneSource!.structural!.domain;
const extent = domain.dimensionsCells.map((cells, axis) => cells * domain.cellSize_m[axis]);
const origin = domain.worldOrigin_m;
const editTarget = [...world.liveScenePrimitiveKeys][0];
const editScript: Array<readonly [number, number, number]> = [
  [0.12, 0.5, 0.12], [0.88, 0.5, 0.12], [0.88, 0.5, 0.88], [0.12, 0.5, 0.88],
  [0.5, 0.85, 0.5], [0.5, 0.15, 0.5], [0.5, 0.5, 0.5],
].map(([u, v, w]) => [origin[0] + u * extent[0], origin[1] + v * extent[1], origin[2] + w * extent[2]] as const);

interface EditStep {
  step: number;
  staged: boolean;
  encoded: boolean;
  /** Frames from staging to the revision being observable, inclusive. */
  frames: number;
  /** Voxelizer compute passes across those frames. */
  passes: number;
  dirtyBricks: number;
  /** Frames on which the publication was still, correctly, not observable. */
  incompleteFrames: number;
  /** A frame that reported complete while the cursor had not reached the end. This must be zero. */
  prematureCompletions: number;
  healthy: boolean;
  pages: number;
}
const editSteps: EditStep[] = [];
if (editTarget) {
  const template = world.liveScenePrimitive(editTarget)!;
  for (let step = 0; step < editScript.length; step += 1) {
    const staged = solver.stageLivePrimitiveUpdates([
      { key: editTarget, primitive: { ...template, center: editScript[step] } },
    ]);
    let frames = 0;
    let passes = 0;
    let incompleteFrames = 0;
    let prematureCompletions = 0;
    let encoded = false;
    let state = await maintenanceState();
    // Drive the budgeted publication to convergence, checking the contract on
    // every frame of it rather than only at the end: a chunked revision that
    // reported itself complete halfway through would pass an end-state check
    // and still be the regression this budget is most able to reintroduce.
    while (frames < convergenceFrameCeiling) {
      const result = submitMaintenance(`Live voxelization edit ${step} frame ${frames}`);
      await device.queue.onSubmittedWorkDone();
      frames += 1;
      passes += result.voxelizerPasses;
      encoded = encoded || result.encoded;
      state = await maintenanceState();
      const reachedEnd = state.cursor >= Math.min(state.dirtyBricks, world.sceneMaintenanceBinding.dirtyBrickCapacity);
      if (state.complete && !reachedEnd) prematureCompletions += 1;
      if (state.complete) break;
      incompleteFrames += 1;
      if (!world.sceneVoxelizationBudget.converging) break;
    }
    const pyramid = pyramidState();
    editSteps.push({
      step, staged, encoded, frames, passes,
      dirtyBricks: state.dirtyBricks, incompleteFrames, prematureCompletions,
      healthy: pyramid.healthy, pages: pyramid.pages,
    });
  }
}
const everyStepHealthy = editSteps.length > 0 && editSteps.every(({ healthy }) => healthy);
const after = pyramidState();
record("pyramid-survives-edit-script", everyStepHealthy && after.pages === before.pages,
  editTarget === undefined
    ? "the scene published no live primitive to move; the edit script did not run"
    : everyStepHealthy && after.pages === before.pages
      ? `${editSteps.length} teleports across the domain; the opacity pyramid held ${after.pages} pages throughout`
      + ` and the address plan stayed ${after.plan?.total ? "total" : "valid"}`
      : `the pyramid withdrew during the edit script — reason=${after.reason ?? "n/a"} detail=${after.detail ?? "n/a"}`
      + ` pages ${before.pages} -> ${after.pages}`,
  { pagesBefore: before.pages, pagesAfter: after.pages, steps: editSteps.length }, 0);
record("address-plan-valid-after-edits", after.plan?.valid === true,
  after.plan
    ? `plan ${after.plan.total ? "total" : "reserved"}: ${after.plan.activePages}/${after.plan.pageCapacity} pages,`
    + ` ${after.plan.basePages}/${after.plan.domainBasePages} base pages addressed, valid=${after.plan.valid}`
    : "the world published no derived address plan at all",
  after.plan ?? null);
const worstFrames = Math.max(0, ...editSteps.map(({ frames }) => frames));
const convergedSteps = editSteps.filter(({ frames }) => frames < convergenceFrameCeiling).length;
record("budgeted-teleport-converges", editSteps.length > 0 && convergedSteps === editSteps.length,
  `${convergedSteps}/${editSteps.length} teleports converged under a ${brickBudget}-brick budget;`
  + ` worst took ${worstFrames} frames of the authored ${convergenceFrameCeiling}`
  + ` (${editSteps.map(({ frames, dirtyBricks }) => `${frames}f/${dirtyBricks}b`).join(" ")})`,
  { worstFrames, converged: convergedSteps }, convergenceFrameCeiling);

/**
 * The contract the budget is most able to break.
 *
 * `SPARSE_SCENE_MAINTENANCE_INCOMPLETE_OVERFLOW` exists because a publication
 * that reports itself complete with bricks still holding the previous revision
 * takes the whole opacity pyramid down with it — the documented failure is a
 * hard-edged black slab across the container footprint. Chunking creates that
 * state deliberately and holds it for several frames, so the lane demands both
 * halves: nobody ever saw it complete early, and somebody did see it incomplete
 * (otherwise the budget never bit and the check is vacuous).
 */
const premature = editSteps.reduce((sum, { prematureCompletions }) => sum + prematureCompletions, 0);
const observedIncomplete = editSteps.reduce((sum, { incompleteFrames }) => sum + incompleteFrames, 0);
const chunkedSteps = editSteps.filter(({ frames }) => frames > 1).length;
record("incomplete-until-the-last-chunk",
  premature === 0 && observedIncomplete > 0 && chunkedSteps === editSteps.length,
  premature > 0
    ? `${premature} frame(s) reported a publication complete before its cursor reached the end of the dirty list;`
    + " this is the black-slab state"
    : chunkedSteps === editSteps.length
      ? `every one of the ${editSteps.length} teleports spanned several frames, and across the`
      + ` ${observedIncomplete} intermediate ones the publication reported itself not observable every time`
      : `only ${chunkedSteps}/${editSteps.length} teleports took more than one frame at a ${brickBudget}-brick budget,`
      + " so the contract was barely exercised; lower FLUID_SVO_VOXELIZATION_BRICK_BUDGET or move a larger record",
  { premature, observedIncomplete, chunkedSteps }, 0);

const indexStatus = world.sceneVoxelizationIndex;
record("record-index-active", indexStatus.enabled && indexStatus.indexedInvalidation,
  indexStatus.enabled
    ? `binning reads a coarse grid (${indexStatus.gridItems} scattered slots, ${indexStatus.globalRecords} records too large`
    + ` to localize) and the last invalidation dispatched ${indexStatus.regionCells} dirty brick cells`
    + `${indexStatus.indexedInvalidation ? "" : " through the unindexed leaf scan"}`
    : "the world published no record index; maintenance is still O(leaves x regions) and O(bricks x records)",
  indexStatus);

// ---------------------------------------------------------------------------
// The bonsai crown: fissured, or a blob?
// ---------------------------------------------------------------------------
const structural = solver.sparseVoxelSceneSource!.structural!;
const control = await readBuffer(structural.control.buffer, 0, 128);
const leafCount = control[1];
const brickSize = control[11];
const leafBaseWord = control[16];
const structure = await readBuffer(structural.structure.buffer, 0, structural.structure.size ?? structural.structure.buffer.size);
// Leaf and node records are addressed *relative to the topology arena*, exactly
// as `topologyLoad` does in the voxelization shader; the flat readback is not.
const topologyBase = structural.structureOffsetsWords.nodes;
const topology = (word: number) => structure[topologyBase + word];
// The whole payload arena, decoded through the block the shaders address it by:
// the banded leaf payload has no owner lane to slice, and a census that read one
// would be counting the absent-lane page rather than the scene.
const scenePayload = await readBuffer(structural.scenePayload.buffer,
  structural.scenePayload.offset ?? 0,
  structural.scenePayload.size ?? structural.scenePayload.buffer.size);
const sceneGeometry = await readBuffer(structural.sceneGeometry.buffer,
  structural.sceneGeometry.offset ?? 0, structural.sceneGeometry.size!);
/**
 * The published signed distance lane, as floats.
 *
 * Both the stride and the channel come from the active profile: `dry` prunes
 * the two channels no scene writer touches, so `solidSignedDistance` sits at
 * word 0 of a 2-word voxel there and word 1 of a 4-word voxel under `full`. A
 * hardcoded index reads a neighbour's bytes and reports the difference as NaN.
 */
const sceneGeometryChannels = SPARSE_BRICK_PAYLOAD_PROFILES[octreeLiveSceneDryPayloadProfile()]
  .find((lane) => lane.name === "sceneGeometry")?.channels;
if (!sceneGeometryChannels) throw new Error("Active payload profile has no sceneGeometry lane");
const solidDistanceChannel = sceneGeometryChannels.indexOf("solidSignedDistance");
if (solidDistanceChannel < 0) throw new Error("Active sceneGeometry lane carries no solidSignedDistance channel");
const sceneDistance = new Float32Array(sceneGeometry.buffer);

/** Morton decode mirroring `decodeMorton` in the voxelization shader. */
function decodeMorton(low: number, high: number, level: number): [number, number, number] {
  const bitAt = (bit: number) => (bit >= 32 ? (high >>> (bit - 32)) & 1 : (low >>> bit) & 1);
  const result: [number, number, number] = [0, 0, 0];
  for (let bit = 0; bit < level; bit += 1) {
    for (let axis = 0; axis < 3; axis += 1) result[axis] += bitAt(3 * bit + axis) * 2 ** bit;
  }
  return result;
}

const finestLevel = domain.maximumDepth;
const censusOwners = new Set(censusProxies.map((proxy) => ownerFor(proxy.ownerIndex)));
const envelopes = censusProxies.map((proxy) => {
  const q = proxy.orientation;
  return {
    centre: [proxy.center_m.x, proxy.center_m.y, proxy.center_m.z] as const,
    radii: [proxy.radius_m.x, proxy.radius_m.y, proxy.radius_m.z] as const,
    conjugate: q ? ([-q.x, -q.y, -q.z, q.w] as const) : ([0, 0, 0, 1] as const),
  };
});
/**
 * Signed distance to the nearest census *envelope* — the exact shape the old
 * downgrade voxelized. Negative inside, and the Lipschitz-1 scaled-radius form
 * the ABI itself uses to clip a field, so it is comparable term for term with
 * the distance the GPU published.
 */
function envelopeDistance(point: readonly [number, number, number]): number {
  let nearest = Number.POSITIVE_INFINITY;
  for (const { centre, radii, conjugate } of envelopes) {
    const d = [point[0] - centre[0], point[1] - centre[1], point[2] - centre[2]] as const;
    const [ux, uy, uz, w] = conjugate;
    const tx = 2 * (uy * d[2] - uz * d[1]), ty = 2 * (uz * d[0] - ux * d[2]), tz = 2 * (ux * d[1] - uy * d[0]);
    const local = [
      d[0] + w * tx + (uy * tz - uz * ty),
      d[1] + w * ty + (uz * tx - ux * tz),
      d[2] + w * tz + (ux * ty - uy * tx),
    ];
    const scaled = Math.hypot(local[0] / radii[0], local[1] / radii[1], local[2] / radii[2]);
    nearest = Math.min(nearest, (scaled - 1) * Math.min(radii[0], radii[1], radii[2]));
  }
  return nearest;
}

let envelopeVoxels = 0, solidVoxels = 0, ownedVoxels = 0, airVoxels = 0, distanceDrift_m = 0;
for (let leaf = 0; leaf < leafCount; leaf += 1) {
  const leafWord = leafBaseWord + leaf * 4;
  const nodeIndex = topology(leafWord);
  const voxelOffset = topology(leafWord + 1);
  const nodeWord = nodeIndex * 8;
  const level = topology(nodeWord + 2);
  const brick = decodeMorton(topology(nodeWord), topology(nodeWord + 1), level);
  const scale = finestLevel > level ? 2 ** (finestLevel - level) : 1;
  for (let local = 0; local < brickSize ** 3; local += 1) {
    const lx = local % brickSize, ly = Math.floor(local / brickSize) % brickSize, lz = Math.floor(local / (brickSize * brickSize));
    const cell = [(brick[0] * brickSize + lx) * scale, (brick[1] * brickSize + ly) * scale, (brick[2] * brickSize + lz) * scale];
    const world_m = [
      origin[0] + (cell[0] + 0.5 * scale) * domain.cellSize_m[0],
      origin[1] + (cell[1] + 0.5 * scale) * domain.cellSize_m[1],
      origin[2] + (cell[2] + 0.5 * scale) * domain.cellSize_m[2],
    ] as const;
    const envelope_m = envelopeDistance(world_m);
    if (envelope_m > 0) continue;
    envelopeVoxels += 1;
    const published_m = sceneDistance[(voxelOffset + local) * sceneGeometryChannels.length + solidDistanceChannel];
    // Strictly outside every published surface, while strictly inside the
    // envelope: air the blob claimed as solid.
    if (published_m > 0) airVoxels += 1;
    distanceDrift_m += Math.abs(published_m - envelope_m);
    const { materialId, ownerId } = unpackMaterialOwner(
      sparseBrickScenePayloadIdentityAt(scenePayload, structural.scenePayloadLanes, voxelOffset + local));
    if (materialId !== 0) solidVoxels += 1;
    if (materialId !== 0 && censusOwners.has(ownerId)) ownedVoxels += 1;
  }
}
const solidFraction = envelopeVoxels === 0 ? 0 : solidVoxels / envelopeVoxels;
const ownedFraction = envelopeVoxels === 0 ? 0 : ownedVoxels / envelopeVoxels;
const airFraction = envelopeVoxels === 0 ? 0 : airVoxels / envelopeVoxels;
const meanDrift_mm = envelopeVoxels === 0 ? 0 : 1000 * distanceDrift_m / envelopeVoxels;
/**
 * The gate, and why it reads the distance lane rather than the coverage lane.
 *
 * Coverage is `clamp(0.5 - d / 2r)` against the cell's own radius, so a voxel
 * counts as solid whenever the surface is anywhere within half a cell diagonal.
 * The hero garden's crown is a three-octave lattice of 17 mm lobes on a 45 mm
 * period inside a 100 mm envelope, and its lattice gaps are about 10 mm — under
 * the 21.7 mm cell radius of a 25 mm lattice. So the *coverage* of the crown is
 * legitimately near 100 % at this resolution however exactly it is voxelized,
 * and under-reporting it would be the failure this whole pass exists to avoid:
 * an under-populated page is indistinguishable from empty space to every
 * consumer, and the geometry would stop casting shadows.
 *
 * What the swap changed, and what is therefore measurable here, is the *content*
 * of the distance lane — the field's own signed distance instead of the
 * envelope ellipsoid's. That lane is what the derived builder differentiates
 * for radiance normals and what W3's owner-to-exact hybrid will refine against,
 * and it is exactly zero-drift if the aggregate is still a blob.
 */
record("crown-carries-the-field", envelopeVoxels > 0 && airFraction >= crownAirFloor && meanDrift_mm > 1,
  envelopeVoxels === 0
    ? "no voxel centre fell inside an aggregate envelope; the census proved nothing"
    : `${(100 * airFraction).toFixed(1)} % of the ${envelopeVoxels} voxels inside the aggregate envelopes are strictly outside`
    + ` the published surface, and the published distance differs from the envelope's by ${meanDrift_mm.toFixed(1)} mm on average.`
    + " Both are 0 for the envelope downgrade this replaces, by construction",
  { airFraction: Number(airFraction.toFixed(4)), meanDrift_mm: Number(meanDrift_mm.toFixed(2)) },
  { airFloor: crownAirFloor, driftFloor_mm: 1 });
log(`  [note] coverage of those envelopes is ${(100 * solidFraction).toFixed(1)} % (${(100 * ownedFraction).toFixed(1)} % owned by`
  + " the aggregates themselves). At a 25 mm lattice a 10 mm lattice gap is inside one cell radius, so conservative"
  + " coverage reports it solid on purpose; the fissures resolve when the lattice does.");

// ---------------------------------------------------------------------------
// Hygiene.
// ---------------------------------------------------------------------------
record("gpu-validation", validationErrors.length === 0,
  validationErrors.length === 0 ? "no uncaptured Dawn validation errors" : validationErrors.join(" | "),
  validationErrors.length, 0);
const svoWarnings = capturedWarnings.filter((message) => message.includes("[svo]"));
record("no-svo-warnings", svoWarnings.length === 0,
  svoWarnings.length === 0 ? "no [svo] console warnings across build and the whole edit script" : svoWarnings.join(" | "),
  svoWarnings.length, 0);

const failures = checks.filter(({ state }) => state === "fail");
const report = {
  lane: "svo-live-voxelization-smoke",
  scene: scenePresetId,
  adapter: adapterInfo,
  state: failures.length === 0 ? "pass" : "fail",
  world: {
    proxies: proxies.length,
    aggregates: clusterProxies.length,
    abiTruePrimitives: abiTrue.length,
    censusAggregates: censusProxies.length,
    dimensionsCells: domain.dimensionsCells,
    cellSize_mm: domain.cellSize_m.map((value) => Number((value * 1000).toFixed(3))),
    brickSize,
    leafCount,
  },
  addressPlan: after.plan ?? null,
  maintenance: {
    initial: initial.voxelizerPasses,
    initialState,
    stillFramePasses: stillPasses,
    brickBudget,
    convergenceFrameCeiling,
    worstConvergenceFrames: worstFrames,
    recordIndex: indexStatus,
    editSteps,
  },
  crownCensus: {
    envelopeVoxels, solidVoxels, ownedVoxels, airVoxels,
    solidFraction: Number(solidFraction.toFixed(4)),
    ownedFraction: Number(ownedFraction.toFixed(4)),
    airFraction: Number(airFraction.toFixed(4)),
    meanDistanceDriftFromEnvelope_mm: Number(meanDrift_mm.toFixed(2)),
    airFloor: crownAirFloor,
  },
  checks,
  warnings: capturedWarnings,
};
if (outPath) {
  mkdirSync(path.dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`);
  log(`Report written to ${outPath}`);
}
console.log(JSON.stringify(report, null, 2));

solver.destroy();
device.destroy();
if (failures.length > 0) log(`\n${failures.length} check(s) failed: ${failures.map(({ id }) => id).join(", ")}`);
assert.ok(true);
process.exit(failures.length > 0 ? 1 : 0);
