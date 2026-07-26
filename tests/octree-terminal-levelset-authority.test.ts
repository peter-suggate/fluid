import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { WebGPUOctreeProjection } from "../lib/webgpu-octree";
import { WebGPUOctreePowerCoarseLevelSet } from "../lib/webgpu-octree-power-coarse-levelset";
import {
  applyGlobalFineTransportDiagnostics,
  WebGPUUniformEulerianSolver,
} from "../lib/webgpu-uniform-eulerian";
import type { GPUEulerianInfo } from "../lib/webgpu-eulerian";

const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

function compact(value: string | Function): string {
  return value.toString().replace(/\s+/g, "");
}

test("fine redistance retains the outer pressure-centre trilinear shell", () => {
  const source = compact(octreeSource);
  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  assert.match(source,
    /constredistanceBandFineCells=Math\.min\(256,transportBandFineCells\+globalFineFactor\+3\)/,
    "allocation planning must reserve one cell beyond the backtrace and trilinear reach");
  assert.match(surface,
    /constredistanceBandCells=Math\.min\(256,bandCells\+this\.globalFineLevelSet\.plan\.fineFactor\+3\)/,
    "recurring redistance must retain the same pressure-centre support shell");
  const settlement = compact(octreeSource.slice(
    octreeSource.indexOf("  private encodePendingFineSettlement("),
    octreeSource.indexOf("\n  encodeSurface(", octreeSource.indexOf("  private encodePendingFineSettlement(")),
  ));
  assert.match(surface, /redistanceBandFineCells:redistanceBandCells/);
  assert.match(settlement, /bandCells:pending\.redistanceBandCells/,
    "topology allocation and deferred redistance must consume the identical authored radius");
});

test("production surface recurrence transports before forces and settles after projection", () => {
  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  const transport = surface.indexOf("transport.encode");
  const topology = surface.indexOf("publicationTopology.encode", transport);
  const pending = surface.indexOf("this.pendingFinePublication={", topology);
  const coarseBeforeForces = surface.indexOf("this.encodeCoarsePhiCorrection", pending);
  assert.ok(transport >= 0 && topology > transport && pending > topology
    && coarseBeforeForces > pending,
  "fine transport/topology must precede validated-provisional coarse advection and correction");
  assert.match(surface.slice(coarseBeforeForces), /publicationTopology,dt_s,true/,
    "the pre-force restriction must explicitly consume the validated provisional fine transaction");
  assert.doesNotMatch(surface.slice(topology, coarseBeforeForces), /publicationRedistance\.encode|publicationVolume\?\.encode/,
    "recurring fine redistance and volume correction must not run before current-step forces");
  assert.match(surface.slice(topology, pending), /,true,this\.globalFineBootstrapped/,
    "fine topology must remain provisional until redistance and volume validation finish");
  const assembly = compact(octreeSource.slice(
    octreeSource.indexOf("  private encodeNativePowerAssembly("),
    octreeSource.indexOf("\n  private encodeStructuredProjection(", octreeSource.indexOf("  private encodeNativePowerAssembly(")),
  ));
  assert.ok(assembly.indexOf("dynamics.encodeAdvection") < assembly.indexOf("dynamics.encodeForcesAndDivergence"),
    "structured velocity destination advection must precede current-step forces and RHS");
  const solve = compact(WebGPUOctreeProjection.prototype.encode);
  assert.ok(solve.indexOf("this.pipelinedMGPCG.encode") < solve.indexOf("this.encodeStructuredProjection")
    && solve.indexOf("this.encodeStructuredProjection") < solve.indexOf("this.encodePendingFineSettlement"),
  "MGPCG/project/CPT seeding must precede fine redistance and coarse re-correction");
  const settlement = compact(octreeSource.slice(
    octreeSource.indexOf("  private encodePendingFineSettlement("),
    octreeSource.indexOf("\n  encodeSurface(", octreeSource.indexOf("  private encodePendingFineSettlement(")),
  ));
  const redistance = settlement.indexOf("pending.redistance.encode");
  const volume = settlement.indexOf("pending.volume?.encode", redistance);
  const finalize = settlement.indexOf("pending.topology.encodeFinalizePublication", volume);
  const recorrect = settlement.indexOf("this.encodeCoarsePhiCorrection", finalize);
  const summaries = settlement.indexOf("this.globalFineSummaries?.encode", recorrect);
  assert.ok(redistance >= 0 && volume > redistance && finalize > volume
    && recorrect > finalize && summaries > recorrect);
  assert.match(settlement.slice(recorrect), /pending\.target,pending\.topology,0/,
    "post-projection coarse correction must not advect the interval twice");
  const schedule = compact(WebGPUOctreePowerCoarseLevelSet.prototype.encode);
  const prepare = schedule.indexOf("prepare.setPipeline(this.preparePipeline");
  const selectors = schedule.indexOf("runRows(this.buildSelectorRowsPipeline");
  const advect = schedule.indexOf("runRows(this.advectPipeline");
  const correct = schedule.indexOf("runRows(this.correctPipeline");
  const publish = schedule.indexOf("runRows(this.publishPipeline");
  const commit = schedule.indexOf("pass.setPipeline(this.commitPipeline");
  assert.ok(prepare >= 0 && prepare < selectors && selectors < advect
    && advect < correct && correct < publish && publish < commit,
  "coarse evolution must remain prepare -> selector rows -> advect -> fine correction -> publish -> commit");
});

test("production substep consumes one immutable epoch and builds the next candidate at its tail", () => {
  const advance = compact(WebGPUUniformEulerianSolver.prototype.advanceTo);
  const flip = advance.indexOf("this.octreeProjection?.encodeReadyTopologyFlip(encoder)");
  const surface = advance.indexOf("this.octreeProjection.encodeSurface(encoder");
  const projection = advance.indexOf("this.octreeProjection.encode(encoder", surface);
  const candidate = advance.indexOf("this.octreeProjection.encodeInactiveTopologyCandidate(encoder", projection);
  assert.ok(flip >= 0 && flip < surface && surface < projection && projection < candidate,
    "each production substep must flip only at its boundary, consume one epoch, then build the next inactive candidate");
  assert.doesNotMatch(advance.slice(flip, candidate), /queue\.submit|mapAsync|getMappedRange/,
    "the vertical slice must rely on command-stream ordering rather than a CPU authority handoff");
  const finish = advance.indexOf("constsubmittedEncoder=encoder", candidate);
  const submit = advance.indexOf("this.device.queue.submit([submittedEncoder.finish()])", finish);
  const retire = advance.indexOf("this.octreeProjection?.retireSubmittedEncoder(submittedEncoder)", submit);
  assert.ok(finish > surface && submit > finish && retire > submit,
    "invocation-stable coarse parameters may be reused only after the owning encoder is submitted");
});

test("octree simulation encode entrypoints never map or submit CPU readback", () => {
  const methods = [
    ["encode", WebGPUOctreeProjection.prototype.encode],
    ["encodeSurface", WebGPUOctreeProjection.prototype.encodeSurface],
    ["encodeSparseBrickWorld", WebGPUOctreeProjection.prototype.encodeSparseBrickWorld],
  ] as const;
  for (const [name, method] of methods) {
    assert.doesNotMatch(method.toString(), /mapAsync|getMappedRange|GPUMapMode\.READ|queue\.submit|createCommandEncoder/,
      `${name} must only append GPU work to its caller's encoder`);
  }
});

test("diagnostic readback remains explicitly separated from simulation encoding", () => {
  const simulation = [
    WebGPUOctreeProjection.prototype.encode,
    WebGPUOctreeProjection.prototype.encodeSurface,
    WebGPUOctreeProjection.prototype.encodeSparseBrickWorld,
  ].map(compact).join("");
  assert.doesNotMatch(simulation, /readSolveDiagnostics|readSurfaceDiagnostics|readPagedPhiDifferential/);
  assert.match(octreeSource, /async readSolveDiagnostics\(\)[\s\S]*?mapAsync\(GPUMapMode\.READ\)/,
    "readback is permitted only through an explicit asynchronous diagnostic API");
});

test("global-fine QA diagnostics publish one bounded structured authority packet", () => {
  const diagnostics = compact(WebGPUOctreeProjection.prototype.readGlobalFineLevelSetDiagnostics);
  assert.match(diagnostics, /label:"GlobalfinestructuredQAdiagnostics",size:576/);
  assert.match(diagnostics, /this\.globalFineSeeds\.buffer,0,readback,0,8/);
  assert.match(diagnostics, /topology\.control,0,readback,8,36/);
  assert.match(diagnostics, /fine\.worklist,0,readback,44,20/);
  assert.match(diagnostics, /this\.powerCoarseLevelSetSchedule\.control,0,readback,64,64/);
  assert.match(diagnostics, /this\.fineToPowerCoarseLevelSet\.control,0,readback,128,32/);
  assert.match(diagnostics, /this\.structuredVelocity\.control,0,readback,160,24/);
  assert.match(diagnostics, /this\.structuredBoundary\.control,0,readback,184,64/);
  assert.match(diagnostics, /this\.fineSeedAdapter\.source\.candidateCount,0,readback,248,36/);
  assert.match(diagnostics,
    /structuredVelocityControl:Array\.from\(words\.slice\(40,46\)\)[\s\S]*structuredBoundaryControl:Array\.from\(words\.slice\(46,62\)\)/);
  assert.doesNotMatch(diagnostics, /faceBand|powerFace|globalFinePowerVelocity/i);
  const simulation = [
    WebGPUOctreeProjection.prototype.encode,
    WebGPUOctreeProjection.prototype.encodeSurface,
    WebGPUOctreeProjection.prototype.encodeSparseBrickWorld,
  ].map(compact).join("");
  assert.doesNotMatch(simulation, /readGlobalFineLevelSetDiagnostics/,
    "QA readback must never steer topology, publication, or scheduling");
});

test("global-fine transport diagnostics expose exact invalid velocity evidence and suppress sentinels", () => {
  const bytes = new ArrayBuffer(64);
  const words = new Uint32Array(bytes);
  const floats = new Float32Array(bytes);
  words.set([1, 2, 31, 0, 9, 4, 5, 6, 7, 8, 0x1400_0008, 0x0800_0008, 12]);
  floats.set([0.125, 0.25, 0.375], 13);
  const info = {} as GPUEulerianInfo;

  applyGlobalFineTransportDiagnostics(info, Array.from(words));
  assert.equal(info.globalFineTransportDepartureOutsideBand, 1);
  assert.equal(info.globalFineTransportNonfiniteVelocity, 2);
  assert.equal(info.globalFineTransportCommitted, false);
  assert.equal(info.globalFineTransportStructuredAuthorityUnavailable, 5);
  assert.equal(info.globalFineTransportVelocityUnavailable, 6);
  assert.equal(info.globalFineTransportInvalidVelocityStatus, 7);
  assert.equal(info.globalFineTransportNonpositiveVelocityResult, 8);
  assert.equal(info.globalFineTransportVelocityStatusReasonOr, 0x1400_0008);
  assert.equal(info.globalFineTransportFirstInvalidVelocityStatus, 0x0800_0008);
  assert.equal(info.globalFineTransportFirstInvalidVelocityLocalIndex, 12);
  assert.deepEqual(info.globalFineTransportFirstInvalidVelocityPosition_m,
    { x: 0.125, y: 0.25, z: 0.375 });

  const clear = new Uint32Array(16);
  clear[12] = 0xffff_ffff;
  applyGlobalFineTransportDiagnostics(info, Array.from(clear));
  assert.equal(info.globalFineTransportFirstInvalidVelocityStatus, undefined);
  assert.equal(info.globalFineTransportFirstInvalidVelocityLocalIndex, undefined);
  assert.equal(info.globalFineTransportFirstInvalidVelocityPosition_m, undefined);
});
