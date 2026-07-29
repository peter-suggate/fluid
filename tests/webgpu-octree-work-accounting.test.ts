import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import {
  OCTREE_WORK_STAGES,
  auditOctreeProductionSource,
  decodeOctreePressureSolveWork,
  OctreeWorkAccounting,
  recordOctreeRuntimeGPUWork,
} from "../lib/webgpu-octree-work-accounting";
import {
  auditOctreeProductionRepository,
  discoverOctreeProductionSources,
  extractEmbeddedWgslSources,
} from "../tools/audit-octree-production-source";

function pressureControls(iterations = 1, maximumIterations = 2) {
  const acceptedRows = new Uint32Array([0, 0xffff_ffff, 1, 7, 0, 1]);
  const classWorksets = [
    new Uint32Array([7, 1, 1, 3, 1, 1, 1]),
    ...Array.from({ length: 3 }, () => new Uint32Array([7, 0, 1, 3, 0, 1, 1])),
  ];
  const hybridBandWorksets = classWorksets.map((header) => header.slice());
  const spgridDispatch = new Uint32Array(26);
  for (let level = 0; level < 2; level += 1) {
    const base = level * 12;
    spgridDispatch[base] = 1;
    spgridDispatch.set([1, 1, 1], base + 2);
    spgridDispatch.set([level + 1 < 2 ? 1 : 0, 1, 1], base + 5);
    spgridDispatch[base + 8] = 1;
    spgridDispatch.set([1, 1, 1], base + 9);
  }
  spgridDispatch[24] = 1; spgridDispatch[25] = 1;
  const outerControl = new Uint32Array(32);
  outerControl[1] = 1; outerControl[2] = iterations;
  outerControl[3] = iterations === 0 ? 1 : 2 * iterations; outerControl[4] = 1;
  outerControl[5] = iterations === 0
    ? 4 * maximumIterations
    : 2 + 4 * (maximumIterations - iterations);
  outerControl[20] = 1;
  const outerIndirectTail = new Uint32Array(maximumIterations * 16);
  for (let record = 0; record < maximumIterations * 4; record += 1) {
    outerIndirectTail.set([1, 1, 1, 0], record * 4);
  }
  if (iterations > 0) {
    outerIndirectTail.fill(0, ((iterations - 1) * 4 + 1) * 4,
      ((iterations - 1) * 4 + 2) * 4);
    outerIndirectTail.fill(0, ((iterations - 1) * 4 + 3) * 4,
      ((iterations - 1) * 4 + 4) * 4);
  }
  outerIndirectTail.fill(0, iterations * 16);
  return { acceptedRows, classWorksets, hybridBandWorksets,
    spgridDispatch, outerControl, outerIndirectTail };
}

const pressurePlan = Object.freeze({
  rowCapacity: 64,
  maximumOuterIterations: 2,
  maximumNeighborSlots: 18,
  classApplyEncodedDispatches: 4,
  hybridBoundarySweeps: 2,
  hybridEncodedCorrectionDispatches: 26,
  spgridLevelCount: 2,
  spgridEncodedCorrectionDispatches: 10,
  spgridLevelCapacities: [128, 2],
  reductionPartialCount: 1,
  reductionLanes: 128 as const,
});

test("octree work accounting closes only a fully observed stage graph", () => {
  const counters = new OctreeWorkAccounting();
  counters.publishTopologyEpoch(7);
  counters.beginSubstep();
  for (const stage of OCTREE_WORK_STAGES) counters.recordIdleStage(stage);
  counters.recordStage("topology-publication", {
    scheduledLanes: 128,
    activeLanes: 80,
    activePages: 2,
    logicalPages: 8,
    worksets: 1,
    estimatedBytesMoved: 800,
  });
  counters.recordStage("pressure-solve", {
    scheduledLanes: 640,
    activeLanes: 400,
    activePages: 7,
    logicalPages: 10,
    worksets: 4,
    estimatedBytesMoved: 8_000,
  });
  counters.recordCatalogStamps(80);
  counters.recordSolver(6, 7, 8);
  counters.setAuthorityBytes("pressure", 4_096);
  counters.setAuthorityBytes("topology", 2_048);
  counters.setScratchBytes("solver-vectors", 1_024);
  counters.sealAllocationInventory();

  const snapshot = counters.snapshot();
  assert.equal(snapshot.stagesComplete, true);
  assert.deepEqual(snapshot.missingStageMetrics,
    Object.fromEntries(OCTREE_WORK_STAGES.map((stage) => [stage, []])));
  assert.equal(snapshot.scheduledLanes, 768);
  assert.equal(snapshot.activeLanes, 480);
  assert.equal(snapshot.activePages, 9);
  assert.equal(snapshot.logicalPages, 18);
  assert.equal(snapshot.worksetCount, 5);
  assert.equal(snapshot.activeLaneRatio, 0.625);
  assert.equal(snapshot.estimatedBytesMoved, 8_800);
  assert.equal(snapshot.solverIterations, 6);
  assert.equal(snapshot.reductionPasses, 7);
  assert.equal(snapshot.stages["pressure-solve"].encodedIterations, 8);
  assert.equal(snapshot.stages["pressure-solve"].executedIterations, 6);
  assert.equal(snapshot.stages["pressure-solve"].estimatedBytesMovedPerActiveElement, 20);
  assert.deepEqual(snapshot.allocatedBytesByAuthority, {
    pressure: 4_096,
    topology: 2_048,
  });
  assert.deepEqual(snapshot.allocatedScratchBytesByArena, {
    "solver-vectors": 1_024,
  });
  assert.equal(snapshot.authoritativeBytes, 6_144);
  assert.equal(snapshot.scratchBytes, 1_024);
  assert.equal(snapshot.allocatedBytes, 7_168);
  assert.equal(snapshot.allocationInventoryComplete, true);
});

test("missing runtime observations remain unavailable instead of becoming zero", () => {
  const counters = new OctreeWorkAccounting();
  counters.recordStage("rhs", { scheduledLanes: 0, activeLanes: 0 });
  const snapshot = counters.snapshot();
  assert.equal(snapshot.stagesComplete, false);
  assert.equal(snapshot.scheduledLanes, null);
  assert.equal(snapshot.activeLanes, null);
  assert.equal(snapshot.activeLaneRatio, null);
  assert.equal(snapshot.solverIterations, null);
  assert.equal(snapshot.reductionPasses, null);
  assert.equal(snapshot.stages.rhs.activePages, null);
  assert.ok(snapshot.missingStageMetrics.rhs.includes("activePages"));
  assert.equal(snapshot.allocationInventoryComplete, false);
});

test("octree work accounting rejects impossible or regressing observations", () => {
  const counters = new OctreeWorkAccounting();
  counters.publishTopologyEpoch(3);
  assert.throws(() => counters.publishTopologyEpoch(2), /monotonic/);
  assert.throws(() => counters.recordStage("rhs", {
    scheduledLanes: 10, activeLanes: 11,
  }), /cannot exceed/);
  assert.throws(() => counters.recordStage("projection", {
    logicalPages: 2, activePages: 3,
  }), /cannot exceed/);
  assert.throws(() => counters.recordStage("fine-redistance", {
    encodedIterations: 2, executedIterations: 3,
  }), /cannot exceed/);
  assert.throws(() => counters.recordStage("fine-transport", {
    estimatedBytesMoved: Number.NaN,
  }), /finite/);
});

test("positively observed empty schedules have a defined perfect active-lane ratio", () => {
  const counters = new OctreeWorkAccounting();
  for (const stage of OCTREE_WORK_STAGES) counters.recordIdleStage(stage);
  assert.equal(counters.snapshot().activeLaneRatio, 1);
});

test("pressure solve telemetry decodes all fixed layers and a zero-work convergence tail", () => {
  const decoded = decodeOctreePressureSolveWork(pressureControls(), pressurePlan);
  assert.equal(decoded.blocker, null);
  const report = decoded.report!;
  assert.equal(report.outer.executedIterations, 1);
  assert.equal(report.outer.outerReductionPasses, 1);
  assert.equal(report.outer.initialReductionPasses, 1);
  assert.equal(report.outer.tailExecutedRowLanes, 0);
  assert.equal(report.outer.zeroedTailDispatches, 6);
  assert.equal(report.layers["class-applies"]?.worksets, 40,
    "only class worksets reached by observed operator applications are credited");
  assert.equal(report.layers["class-applies"]?.encodedIterations, 16,
    "the immutable graph still reports its full encoded operator budget");
  assert.equal(report.layers["class-applies"]?.executedIterations, 10);
  assert.equal(report.layers["class-applies"]?.encodedDispatches, 64);
  assert.equal(report.layers["class-applies"]?.executedDispatches, 10,
    "zero-indirect nested tail applications launch no class payload");
  assert.equal(report.layers["class-applies"]?.scheduledLanes, 640);
  assert.equal(report.layers["class-applies"]?.activeLanes, 10);
  assert.ok(report.layers["class-applies"]!.scheduledLanes
    > report.layers["class-applies"]!.activeLanes,
  "scheduled lanes retain only workgroup padding, never convergence-tail gates");
  assert.equal(report.layers["hybrid-shell"]?.encodedIterations, 3);
  assert.equal(report.layers["hybrid-shell"]?.executedIterations, 2);
  assert.equal(report.layers["hybrid-shell"]?.worksets, 2);
  assert.equal(report.layers["hybrid-shell"]?.executedDispatches, 16);
  assert.equal(report.layers["spgrid-vcycle"]?.encodedIterations, 3);
  assert.equal(report.layers["spgrid-vcycle"]?.executedIterations, 2);
  assert.equal(report.layers["spgrid-vcycle"]?.worksets, 4);
  assert.equal(report.layers["spgrid-vcycle"]?.executedDispatches, 20,
    "both parent-owned restriction dispatches are credited per executed correction");
  assert.equal(report.layers["outer-pipelined-mgpcg"]?.encodedDispatches, 20);
  assert.equal(report.layers["outer-pipelined-mgpcg"]?.executedDispatches, 14);
  assert.equal(report.layers["spgrid-vcycle"]?.activePages, 4);
  assert.equal(report.layers["persistent-coarse-solve"], null);
  assert.ok(report.stage.estimatedBytesMoved > 0);

  const accounting = new OctreeWorkAccounting();
  accounting.recordPressureSolveReport(report);
  assert.equal(accounting.snapshot().solverIterations, 1);
  assert.equal(accounting.snapshot().stages["pressure-solve"].reductionPasses, 2);
});

test("pressure telemetry gives an initially converged solve no nested tail credit", () => {
  const decoded = decodeOctreePressureSolveWork(pressureControls(0), pressurePlan);
  assert.equal(decoded.blocker, null);
  const report = decoded.report!;
  const classes = report.layers["class-applies"]!;
  const hybrid = report.layers["hybrid-shell"]!;
  const spgrid = report.layers["spgrid-vcycle"]!;

  assert.equal(report.outer.executedIterations, 0);
  assert.equal(report.outer.zeroedTailDispatches, 8);
  assert.equal(classes.encodedIterations, 16);
  assert.equal(classes.executedIterations, 6,
    "initial A/M/A work remains, but neither encoded outer tail is credited");
  assert.equal(classes.worksets, 24);
  assert.equal(classes.scheduledLanes, 384);
  assert.equal(classes.activeLanes, 6);
  assert.equal(classes.executedDispatches, 6);
  assert.equal(hybrid.encodedIterations, 3);
  assert.equal(hybrid.executedIterations, 1);
  assert.equal(hybrid.worksets, 1);
  assert.equal(hybrid.scheduledLanes, 448);
  assert.equal(hybrid.executedDispatches, 8);
  assert.equal(spgrid.encodedIterations, 3);
  assert.equal(spgrid.executedIterations, 1);
  assert.equal(spgrid.worksets, 2);
  assert.equal(spgrid.scheduledLanes, 768);
  assert.equal(spgrid.executedDispatches, 10,
    "the initial correction includes its parent-owned restriction dispatch");
});

test("nested payload credit grows only with GPU-executed outer iterations", () => {
  const reports = [0, 1, 2].map((iterations) => {
    const decoded = decodeOctreePressureSolveWork(
      pressureControls(iterations), pressurePlan,
    );
    assert.equal(decoded.blocker, null);
    return decoded.report!;
  });
  assert.deepEqual(reports.map((report) => report.layers["class-applies"]!.executedIterations),
    [6, 10, 15]);
  assert.deepEqual(reports.map((report) => report.layers["class-applies"]!.scheduledLanes),
    [384, 640, 960]);
  assert.deepEqual(reports.map((report) => report.layers["hybrid-shell"]!.executedIterations),
    [1, 2, 3]);
  assert.deepEqual(reports.map((report) => report.layers["spgrid-vcycle"]!.executedIterations),
    [1, 2, 3]);
  assert.deepEqual(reports.map((report) => report.outer.outerReductionPasses),
    [0, 1, 3], "each nonterminal update adds one direct-curvature reduction");
  assert.deepEqual(reports.map((report) => report.layers["outer-pipelined-mgpcg"]!.reductionPasses),
    [1, 2, 4]);
  assert.deepEqual(reports.map((report) => report.layers["outer-pipelined-mgpcg"]!.executedDispatches),
    [12, 14, 18]);
  assert.ok(reports.every((report) => report.layers["class-applies"]!.encodedIterations === 16));
  assert.ok(reports.every((report) => report.layers["hybrid-shell"]!.encodedIterations === 3));
  assert.ok(reports.every((report) => report.layers["spgrid-vcycle"]!.encodedIterations === 3));
});

test("pressure telemetry credits repeated L2 work only to the compact Section 4.3 shell", () => {
  const compact = pressureControls();
  compact.acceptedRows[2] = 4;
  compact.classWorksets[0] = new Uint32Array([7, 3, 64, 3, 1, 1, 1]);
  compact.classWorksets[2] = new Uint32Array([7, 1, 64, 3, 1, 1, 1]);
  compact.hybridBandWorksets[0] = new Uint32Array([7, 0, 64, 3, 0, 1, 1]);
  compact.hybridBandWorksets[2] = new Uint32Array([7, 1, 64, 3, 1, 1, 1]);
  compact.outerControl[4] = 4;
  compact.spgridDispatch[25] = 4;
  const allBand = pressureControls();
  allBand.acceptedRows[2] = 4;
  allBand.classWorksets = compact.classWorksets.map((header) => header.slice());
  allBand.hybridBandWorksets = compact.classWorksets.map((header) => header.slice());
  allBand.outerControl[4] = 4;
  allBand.spgridDispatch[25] = 4;
  const compactReport = decodeOctreePressureSolveWork(compact, pressurePlan).report;
  const allBandReport = decodeOctreePressureSolveWork(allBand, pressurePlan).report;
  assert.ok(compactReport && allBandReport);
  assert.ok(compactReport.layers["class-applies"]!.activeLanes
    < allBandReport.layers["class-applies"]!.activeLanes);
  assert.ok(compactReport.layers["hybrid-shell"]!.activeLanes
    < allBandReport.layers["hybrid-shell"]!.activeLanes);
});

test("pressure telemetry blocks missing, stale, failed, and unproven controls", () => {
  assert.match(decodeOctreePressureSolveWork({}, pressurePlan).blocker!, /accepted-row/);
  const stale = pressureControls(); stale.classWorksets[0]![0] = 6;
  assert.match(decodeOctreePressureSolveWork(stale, pressurePlan).blocker!, /stale/);
  const failed = pressureControls(); failed.outerControl[0] = 1;
  assert.match(decodeOctreePressureSolveWork(failed, pressurePlan).blocker!, /successful/);
  const tail = pressureControls(); tail.outerControl[5] = 0;
  assert.match(decodeOctreePressureSolveWork(tail, pressurePlan).blocker!, /zero-work/);
  assert.equal(decodeOctreePressureSolveWork(tail, pressurePlan).report, null,
    "an invalid control never fabricates zero work");
  assert.match(decodeOctreePressureSolveWork(pressureControls(), {
    ...pressurePlan, reductionPartialCount: 2,
  }).blocker!, /plan is invalid/);
  const skippedExecutedRow = pressureControls();
  skippedExecutedRow.outerIndirectTail.fill(0, 0, 4);
  assert.match(decodeOctreePressureSolveWork(skippedExecutedRow, pressurePlan).blocker!,
    /executed indirect/);
  const transferCountDispatch = pressureControls();
  transferCountDispatch.spgridDispatch.set([0, 1, 1], 5);
  assert.match(decodeOctreePressureSolveWork(transferCountDispatch, pressurePlan).blocker!,
    /compact publication/,
    "restriction dispatch must be sized by published coarse parents, not transfer records");
  assert.match(decodeOctreePressureSolveWork(pressureControls(), {
    ...pressurePlan, persistentEnabled: true, persistentMaximumIterations: 12,
  }).blocker!, /persistent solve control unavailable/);
});

test("pressure telemetry accounts the selected persistent single-workgroup solve", () => {
  const controls = pressureControls();
  const persistentControl = new Uint32Array(13);
  persistentControl[1] = 1; persistentControl[2] = 2; persistentControl[4] = 1;
  const decoded = decodeOctreePressureSolveWork({ ...controls, persistentControl }, {
    ...pressurePlan, persistentEnabled: true, persistentMaximumIterations: 12,
  });
  assert.equal(decoded.blocker, null);
  assert.equal(decoded.report?.layers["persistent-coarse-solve"]?.scheduledLanes, 256);
  assert.equal(decoded.report?.layers["persistent-coarse-solve"]?.executedIterations, 2);
  assert.equal(decoded.report?.layers["persistent-coarse-solve"]?.reductionPasses, 3);
  assert.equal(decoded.report?.stage.reductionPasses, 5,
    "the aggregate retains two outer and three persistent reductions");
});

test("runtime capture records only metrics proved by compact GPU controls", () => {
  const accounting = new OctreeWorkAccounting();
  accounting.beginSubstep();
  const controls = pressureControls();
  const liveRowDispatch = new Uint32Array([1, 1, 1, 1, 1, 1]);
  const ownerControl = new Uint32Array(16);
  ownerControl.set([3, 5, 0, 8, 64, 0, 0, 7, 2, 1, 1, 7, 0, 8, 8, 0x4f57_4e52]);
  const fineWorkset = new Uint32Array([7, 2, 8, 3, 1, 1, 1]);
  const fineTopologyControl = new Uint32Array(12);
  fineTopologyControl[2] = 2; fineTopologyControl[4] = 1;
  const fineTransportControl = new Uint32Array(16);
  fineTransportControl[2] = 128; fineTransportControl[3] = 1;
  const fineTransportGovernor = new Uint32Array(68);
  fineTransportGovernor[1] = 1;
  const fineRedistanceControl = new Uint32Array(10);
  fineRedistanceControl[3] = 1; fineRedistanceControl[6] = 120;
  fineRedistanceControl[8] = 2;
  const pressure = recordOctreeRuntimeGPUWork(accounting, {
    ...controls,
    liveRowDispatch,
    runtimeClassWorksets: controls.classWorksets,
    runtimeFamilyWorksets: controls.classWorksets,
    ownerControl,
    fineWorkset,
    fineTopologyControl,
    fineTransportControl,
    fineTransportWorkset: fineWorkset,
    fineTransportGovernor,
    fineRedistanceControl,
  }, { pressure: pressurePlan, ownerLogicalPages: 64, ownerBytesPerPage: 2_048,
    fineLogicalPages: 16, fineSamplesPerPage: 64, fineMaximumPages: 8,
    fineTransportEncodedIterations: 64, fineRedistanceEncodedIterations: 8 });
  assert.equal(pressure.blocker, null);
  const snapshot = accounting.snapshot();
  assert.equal(snapshot.topologyEpoch, 7);
  assert.equal(snapshot.catalogStamps, 1);
  assert.equal(snapshot.stages["topology-publication"].activePages, 5);
  assert.equal(snapshot.stages["forces-and-solids"].activeLanes, 1);
  assert.equal(snapshot.stages.rhs.scheduledLanes, 64);
  assert.equal(snapshot.stages.projection.worksets, 8,
    "projection accounts four family updates and four row reconstructions");
  assert.equal(snapshot.stages["pressure-solve"].executedIterations, 1);
  assert.equal(snapshot.stages["candidate-build"].activePages, 2);
  assert.equal(snapshot.stages["fine-transport"].activeLanes, 128);
  assert.equal(snapshot.stages["fine-transport"].scheduledLanes, 128);
  assert.equal(snapshot.stages["fine-redistance"].activeLanes, 1_024);
  assert.equal(snapshot.stages["fine-redistance"].logicalPages, 16);
  assert.equal(snapshot.stagesComplete, true,
    "a complete valid GPU packet closes all eight stage records");
});

test("projection capture selects the GPU-accepted workset bank and smoke awaits it", () => {
  const projection = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
  assert.match(projection, /classWorksetBanks:[\s\S]*\[0, 1\]\.map/,
    "both immutable workset banks must be exposed for post-submit capture");
  assert.match(projection,
    /acceptedBank\s*=\s*acceptedRows[\s\S]*classWorkset\$\{acceptedBank\}_\$\{index\}/,
    "readback must select the bank named by the accepted GPU control");
  assert.match(projection, /await this\.device\.queue\.onSubmittedWorkDone\(\)[\s\S]*Capture octree GPU work accounting/,
    "the accounting copy must be observational and occur only after submitted simulation work completes");
  assert.match(projection,
    /setAuthorityBytes\("surface-state", this\.surfaceStateAccountingBytes\)[\s\S]*setAuthorityBytes\("diagnostic-overlays"[\s\S]*sealAllocationInventory\(\)/,
    "the post-submit snapshot must refresh named lazy owners and seal the allocation inventory");
  assert.doesNotMatch(projection,
    /setAuthorityBytes\("projection-resident",\s*this\.info\.allocatedBytes\)/,
    "an aggregate compensating bucket could conceal double-counted named arenas");
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  assert.match(smoke,
    /capturedWorkAccounting\s*=\s*accountingOwner\.captureWorkAccounting[\s\S]*await accountingOwner\.captureWorkAccounting\(\)[\s\S]*octreeWorkAccounting:\s*capturedWorkAccounting\?\.snapshot/,
    "the smoke artifact must await GPU accounting capture before snapshotting the ledger");
});

test("production source guards reject all forbidden recurring constructs with locations", () => {
  assert.deepEqual(auditOctreeProductionSource("clean.ts",
    "pass.dispatchWorkgroupsIndirect(workset, 0);", "host"), []);
  const violations = auditOctreeProductionSource("bad.ts", [
    "pass.dispatchWorkgroups(Math.ceil(this.rowCapacity / 64));",
    "while (true) {}",
    "atomicAdd(&faceVelocity[index], contribution);",
    "new WebGPUOctreeParticles(); // primary APIC P2G/G2P",
  ].join("\n"), "host");
  assert.deepEqual(violations.map((violation) => [violation.line, violation.rule]), [
    [1, "capacity-dispatch"],
    [2, "unbounded-lookup"],
    [3, "floating-velocity-scatter"],
    [4, "primary-particle-transport"],
    [4, "primary-particle-transport"],
    [4, "primary-particle-transport"],
    [4, "primary-particle-transport"],
  ]);
  assert.equal(auditOctreeProductionSource("mg.wgsl",
    "fn assembleGalerkin() { tripleProduct(); }", "shader")[0]?.rule,
  "galerkin-rap");
  assert.equal(auditOctreeProductionSource("lookup.wgsl",
    "loop { let row = lookup(key); if (row != INVALID) { break; } }", "shader")[0]?.rule,
  "unbounded-lookup");
  assert.deepEqual(auditOctreeProductionSource("bounded.ts",
    "loop { processOneItem(); }", "host"), [],
  "the WGSL loop spelling must not create host-source false positives");
});

test("capacity scan guard separates per-item rescans from lane-partitioned sweeps", () => {
  // A helper called once per work item that rescans the whole changed list is
  // the quadratic shape; it terminates, so no loop-keyword guard sees it.
  const perItem = auditOctreeProductionSource("rescan.wgsl", [
    "fn affectedRadii(key:u32)->u32{",
    "  let total=min(pageDelta[0],2u*params.pageCapacity);var hit=0u;",
    "  for(var item=0u;item<total;item+=1u){hit|=probe(item);}",
    "  return hit;}",
  ].join("\n"), "shader");
  assert.deepEqual(perItem.map(({ line, rule }) => ({ line, rule })),
    [{ line: 3, rule: "capacity-scan" }]);

  // A binary search nested inside another loop pays the same product but has a
  // logarithmic inner cost, so it is reported under its own rule.
  const nested = auditOctreeProductionSource("probe.wgsl", [
    "fn ownerRow(cell:vec3u)->u32{let count=min(accepted[2],p.rowCapacity);var size=1u;",
    "  loop{var lo=0u;var hi=count;",
    "    while(lo<hi){let mid=lo+(hi-lo)/2u;if(less(mid)){lo=mid+1u;}else{hi=mid;}}",
    "    if(size>=p.maxLeaf){break;}size*=2u;}",
    "  return INVALID;}",
  ].join("\n"), "shader");
  assert.ok(nested.some((violation) => violation.rule === "nested-capacity-scan"),
    "a capacity-bounded binary search inside a loop must be reported");

  // `min(live, capacity)` is a fail-closed clamp, and `first..last` divides the
  // live list across lanes. Total work is the live count, not the capacity.
  assert.deepEqual(auditOctreeProductionSource("partitioned.wgsl", [
    "@compute @workgroup_size(64)fn sweep(@builtin(local_invocation_index)lane:u32){",
    "  let rows=min(control.rowCount,p.rowCapacity);let chunk=(rows+63u)/64u;",
    "  let first=lane*chunk;let last=min(first+chunk,rows);",
    "  for(var row=first;row<last;row+=1u){accumulate(row);}}",
  ].join("\n"), "shader"), [],
  "a lane-partitioned sweep over a capacity-clamped live count is the intended shape");

  // A compile-time trip count inside a capacity loop is fixed work per item.
  assert.deepEqual(auditOctreeProductionSource("fixed.wgsl", [
    "@compute @workgroup_size(64)fn classify(@builtin(local_invocation_index)lane:u32){",
    "  let rows=min(control.rowCount,p.rowCapacity);",
    "  for(var row=lane;row<rows;row+=64u){for(var cls=0u;cls<9u;cls+=1u){tally(row,cls);}}}",
  ].join("\n"), "shader"), [],
  "a fixed-width inner loop is not a capacity scan");
});

test("capacity dispatch guard follows local and field aliases", () => {
  const violations = auditOctreeProductionSource("aliased.ts", [
    "const groups = Math.ceil(rowCapacity / 64);",
    "pass.dispatchWorkgroups(groups);",
    "this.workgroups = Math.ceil(this.pageCapacity / 64);",
    "pass.dispatchWorkgroups(this.workgroups);",
  ].join("\n"), "host");
  assert.deepEqual(violations.map(({ line, rule }) => ({ line, rule })), [
    { line: 2, rule: "capacity-dispatch" },
    { line: 4, rule: "capacity-dispatch" },
  ]);
});

test("shader deletion gate rejects retired identifiers and general timestep authorities", () => {
  const source = [
    "// fallbackLegacyCompatibility is commentary, not executable WGSL",
    "@group(0) @binding(0) var<storage, read> faces: array<PowerFace>;",
    "@compute @workgroup_size(1) fn compatibilityFallbackMain() {",
    "  let legacyBranch = 1u;",
    "}",
  ].join("\n");
  const violations = auditOctreeProductionSource("embedded.wgsl", source, "shader");
  assert.deepEqual(violations.map(({ line, rule, excerpt }) => ({ line, rule, excerpt })), [
    { line: 2, rule: "general-timestep-authority",
      excerpt: "var<storage, read> faces:" },
    { line: 3, rule: "retired-shader-terminology", excerpt: "compatibilityFallbackMain" },
    { line: 4, rule: "retired-shader-terminology", excerpt: "legacyBranch" },
  ]);
  assert.deepEqual(auditOctreeProductionSource(
    "webgpu-octree-power-galerkin-persistent3.ts", "export {};", "host",
  ).map(({ line, rule }) => ({ line, rule })), [
    { line: 1, rule: "deleted-production-module" },
  ]);
});

test("repository audit extracts every WGSL template and reports original TypeScript lines", () => {
  const root = mkdtempSync(join(tmpdir(), "fluid-octree-wgsl-audit-"));
  const library = join(root, "lib"); mkdirSync(library);
  const sourcePath = join(library, "webgpu-octree-embedded.ts");
  const source = [
    "export const prose = `ordinary host text`;",
    "export const clean = /* wgsl */ `@compute @workgroup_size(1) fn cleanMain() {}`;",
    "export const rejected = /* wgsl */ `",
    "@compute @workgroup_size(1) fn legacyCompatibilityMain() {}",
    "`;",
  ].join("\n");
  writeFileSync(sourcePath, source);
  const embedded = extractEmbeddedWgslSources("lib/webgpu-octree-embedded.ts", source);
  assert.equal(embedded.length, 2);
  assert.deepEqual(embedded.map(({ bodyLine }) => bodyLine), [2, 3]);
  const audit = auditOctreeProductionRepository(root);
  assert.deepEqual(audit.violations.map(({ source: name, line, rule, excerpt }) =>
    ({ source: name, line, rule, excerpt })), [{
    source: "lib/webgpu-octree-embedded.ts",
    line: 4,
    rule: "retired-shader-terminology",
    excerpt: "legacyCompatibilityMain",
  }]);
});

test("repository audit discovers new production octree sources instead of scanning a synthetic list", () => {
  const root = mkdtempSync(join(tmpdir(), "fluid-octree-audit-"));
  const library = join(root, "lib");
  mkdirSync(library);
  const clean = join(library, "webgpu-octree-clean.ts");
  const bad = join(library, "webgpu-octree-new-hot-loop.ts");
  const unrelated = join(library, "webgpu-uniform-eulerian.ts");
  writeFileSync(clean, "pass.dispatchWorkgroupsIndirect(workset, 0);\n");
  writeFileSync(bad, "pass.dispatchWorkgroups(Math.ceil(rowCapacity / 64));\n");
  writeFileSync(unrelated, "while (true) {}\n");

  assert.deepEqual(discoverOctreeProductionSources(root), [clean, bad]);
  const audit = auditOctreeProductionRepository(root);
  assert.deepEqual(audit.scannedSources, [
    "lib/webgpu-octree-clean.ts",
    "lib/webgpu-octree-new-hot-loop.ts",
  ]);
  assert.deepEqual(audit.violations.map(({ source, line, rule }) => ({ source, line, rule })), [{
    source: "lib/webgpu-octree-new-hot-loop.ts",
    line: 1,
    rule: "capacity-dispatch",
  }]);
});

test("repository audit includes the real production source tree by default", () => {
  const root = new URL("..", import.meta.url).pathname;
  const sources = discoverOctreeProductionSources(root);
  assert.ok(sources.length > 30);
  assert.ok(sources.some((source) => source.endsWith("/lib/webgpu-octree.ts")),
    "the primary orchestration owner and every embedded shader must be audited");
  assert.ok(sources.some((source) => source.endsWith("/lib/webgpu-octree-structured-dynamics.ts")));
  assert.ok(!sources.some((source) => source.endsWith("/lib/webgpu-octree-voxel-inspection.ts")));
  const audit = auditOctreeProductionRepository(root);
  assert.equal(audit.scannedSources.length, sources.length);
  assert.ok(audit.violations.every((violation) => sources.some((source) =>
    source.endsWith(violation.source))), "every violation must belong to an automatically discovered source");
});
