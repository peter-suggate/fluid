#!/usr/bin/env node
/** Pure CPU/Naga gate for the integrated production B16/P16 resident shader. */
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import { createWebgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import { createSparseCM12IncrementalActivityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";
import { createSparseCM12CanonicalMembershipLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-canonical-membership";
import { createSparseCM12FramePlanLayout } from "../lib/core/sparse-cm12-frame-plan";
import { createSparseCM12FramePlanPresentationLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12PressureTopologyRepairLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12ResidentPersistentPressureCacheLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-persistent-pressure-cache";
import { createSparseCM12FaceProjectionAuthorityLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority";
import { createSparseCM12VexActivityBatchLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-activity-batch";
import { createSparseCM12PressureAddressingABLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-addressing-ab";

const pressure = { edgeCoefficientBaseWords: 4096, cellSlotBaseWords: 8192,
  rowSlotBaseWords: 9216, cellChangeBaseWords: 10240, rowChangeBaseWords: 11264,
  brickStateBaseWords: 12288, rowTopologyStampBaseWords: 12352,
  aggregateEdgeForFineEdgeBaseWords: 13376, aggregateEdgeSourceBaseWords: 14400,
  hierarchyEdgeForAggregateBaseWords: [14464], headerBaseWords: 15488,
  totalWords: 15496 };
const activity = createSparseCM12IncrementalActivityLayout({
  baseWords: 4168, brickCount: 8 });
const membership = createSparseCM12CanonicalMembershipLayout({
  baseWords: activity.totalWords, cellCapacity: 1024, rowCapacity: 2048 });
const framePlan = createSparseCM12FramePlanLayout({
  baseWords: Math.ceil(membership.totalWords / 64) * 64,
  brickCapacity: 8, brickFineResolution: 16, packetCount: 6 });
const presentation = createSparseCM12FramePlanPresentationLayout({
  baseWords: framePlan.totalWords, pageCapacity: 8, brickFineResolution: 16,
  pageResolution: 16, packetIndex: 5 });
const frameControl = createSparseCM12FrameControl({ baseWords: 32768,
  cellWorkgroups: 16, rowWorkgroups: 32, bodyCapacity: 0, d4Capable: true,
  rigidCapable: false, boundaryCapable: false });
const ptr = createSparseCM12PressureTopologyRepairLayout({
  baseWords: frameControl.layout.totalWords, brickCapacity: 8, rowCapacity: 2048,
  brickFineResolution: 16, presentationPageResolution: 16 });
const pcf = createSparseCM12ResidentPersistentPressureCacheLayout({
  baseWords: ptr.totalWords, cellCount: 1024, rowCount: 2048,
  directedEdgeCount: 1024, brickCount: 8, aggregateEdgeCount: 32,
  hierarchyLevelCounts: [8], hierarchyEdgeLevelCounts: [32] });
const fpa = createSparseCM12FaceProjectionAuthorityLayout({
  baseWords: pcf.bufferSizeWords, rowCapacity: 2048, cellCapacity: 1024 });
const batch = createSparseCM12VexActivityBatchLayout({
  activityTailWords: presentation.totalWords, stateTailFloats: 65536,
  cellCapacity: 1024 });
const pressureAddressing = process.argv.includes("--pab")
  ? createSparseCM12PressureAddressingABLayout({
    baseWords: batch.totalActivityWords, cellCapacity: 1024,
    brickFineResolution: 16, presentationPageResolution: 16,
    constructionMode: "qa-pressure-addressing-ab",
  }) : undefined;
const source = createWebgpuSparseCM12ResidentWGSL(16, 16, pressure,
  activity, membership, framePlan, presentation, frameControl.layout,
  ptr, pcf, fpa, batch, pressureAddressing, undefined);
if (process.argv.includes("--emit-wgsl")) {
  console.log(source);
  process.exit(0);
}
const directory = mkdtempSync(join(tmpdir(), "fluid-resident-vex-"));
try {
  const path = join(directory, "resident.wgsl"); writeFileSync(path, source);
  const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
  if (result.status !== 0) throw new Error(result.stderr || result.stdout);
  const entries = [...source.matchAll(/@compute[^\n]*\nfn\s+([A-Za-z0-9_]+)/g)].length;
  console.log(`Sparse CM12 integrated VEX2${pressureAddressing ? "+PAB1" : ""}`
    + ` resident: Naga PASS (${entries} entry points)`);
} finally { rmSync(directory, { recursive: true, force: true }); }
