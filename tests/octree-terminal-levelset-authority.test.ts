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
const powerFaceSource = readFileSync(new URL("../lib/webgpu-octree-power-faces.ts", import.meta.url), "utf8");

function compact(value: string | Function): string {
  return value.toString().replace(/\s+/g, "");
}

test("recurring coarse restriction uses candidate rows without changing committed power source", () => {
  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  const restriction = surface.indexOf("this.fineToPowerCoarseLevelSet.encode");
  const summaries = surface.indexOf("this.globalFineSummaries?.encode", restriction);
  assert.ok(restriction >= 0 && summaries > restriction);
  assert.equal(surface.slice(restriction, summaries)
    .split("rowDirectory:this.powerFaces.candidateRowDirectoryForCoarse").length - 1, 2,
  "fine restriction and recurring coarse publication must share the current candidate row identities");

  const sourceGetter = compact(powerFaceSource.slice(
    powerFaceSource.indexOf("get source(): OctreePowerFaceSource"),
    powerFaceSource.indexOf("get candidateRowDirectoryForCoarse"),
  ));
  assert.match(sourceGetter, /rowDirectory:this\.rowDirectory/,
    "the public power-face source retains the immutable committed row directory");
  assert.doesNotMatch(sourceGetter, /candidateRowDirectory/,
    "the candidate directory cannot leak into admitted downstream consumers");
  assert.match(compact(powerFaceSource),
    /getcandidateRowDirectoryForCoarse\(\):GPUBuffer\{returnthis\.candidateRowDirectory;\}/,
    "the pre-commit coarse transaction has an explicit candidate-only directory authority");
});

test("fine redistance retains the outer pressure-centre trilinear shell", () => {
  const source = compact(octreeSource);
  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  assert.match(source,
    /constredistanceBandFineCells=Math\.min\(256,transportBandFineCells\+globalFineFactor\+3\)/,
    "allocation planning must reserve one cell beyond the backtrace and trilinear reach");
  assert.match(surface,
    /constredistanceBandCells=Math\.min\(256,bandCells\+this\.globalFineLevelSet\.plan\.fineFactor\+3\)/,
    "recurring redistance must retain the same pressure-centre support shell");
  assert.match(surface,
    /redistanceBandFineCells:redistanceBandCells[\s\S]*publicationRedistance\.encode\(redistanceBroker,\{bandCells:redistanceBandCells/,
    "topology allocation and redistance must consume the identical authored radius");
});

test("production surface recurrence advects, redistances, conserves volume, and fine-corrects compact coarse phi", () => {
  const surface = compact(WebGPUOctreeProjection.prototype.encodeSurface);
  const bootstrap = surface.indexOf("this.powerCoarseLevelSet.encodeBootstrapFromSurfaceLeaves");
  const bootstrapSchedule = surface.indexOf("this.powerCoarseLevelSetSchedule.encode", bootstrap);
  const fineTopologyAB = surface.indexOf("publicationTopology=this.globalFineTopologyAB", bootstrapSchedule);
  const fineTopologyBA = surface.indexOf("publicationTopology=this.globalFineTopologyBA", fineTopologyAB);
  const fineTopologyEncode = surface.indexOf("publicationTopology.encode", fineTopologyBA);
  const fineRedistance = surface.indexOf("publicationRedistance.encode", fineTopologyEncode);
  const fineVolume = surface.indexOf("publicationVolume?.encode", fineRedistance);
  const publicationGate = surface.indexOf("publicationTopology.encodeFinalizePublication", fineVolume);
  const restriction = surface.indexOf("this.fineToPowerCoarseLevelSet.encode", publicationGate);
  const recurringSchedule = surface.indexOf("this.powerCoarseLevelSetSchedule.encode", bootstrapSchedule + 1);
  const summaries = surface.indexOf("this.globalFineSummaries?.encode", recurringSchedule);

  assert.ok(bootstrap >= 0 && bootstrapSchedule > bootstrap, "coarse phi must be initialized from the bootstrap leaves");
  assert.match(surface.slice(bootstrapSchedule, summaries), /dt:0,/, "bootstrap must not perform an artificial advection step");
  assert.ok(fineTopologyAB > bootstrapSchedule && fineTopologyBA > fineTopologyAB
    && fineTopologyEncode > fineTopologyBA && fineRedistance > fineTopologyEncode && fineVolume > fineRedistance
    && publicationGate > fineVolume && restriction > publicationGate
    && recurringSchedule > restriction && summaries > recurringSchedule,
  "fine evolution -> atomic publication -> fine restriction -> recurring coarse evolution -> merged summary must remain ordered in one command stream");
  assert.match(surface.slice(fineTopologyEncode, fineRedistance), /,true,(?:this\.)?globalFineBootstrapped/,
    "fine topology must remain provisional until redistance and volume validation finish");
  assert.match(surface.slice(fineRedistance, publicationGate),
    /publicationVolume\?\.encode\(redistanceBroker\)/,
    "production must apply and remeasure the bounded volume correction before admitting the fine generation");
  assert.doesNotMatch(surface.slice(fineRedistance, publicationGate), /encodeMeasurement/,
    "telemetry-only measurement cannot satisfy the mini-dam conservation gate");
  assert.match(surface.slice(publicationGate, restriction), /redistance:publicationRedistance\.control/);
  assert.match(surface.slice(restriction, recurringSchedule), /topologyControl:publicationTopology\.control/,
    "fine restriction must consume the exact topology transaction that produced its source");
  assert.match(surface.slice(bootstrapSchedule, fineTopologyAB),
    /rowDirectory:this\.powerFaces\.source\.rowDirectory/,
    "cold bootstrap continues to consume the last admitted row directory");
  assert.doesNotMatch(surface, /globalFineBootstrapped\?undefined:this\.globalFineSeeds/,
    "cold bootstrap seeds must be retryable after a GPU-rejected first generation");
  assert.match(surface.slice(recurringSchedule), /fineCorrection:\{rowOffsets:correction\.rowOffsets/);
  assert.match(surface.slice(recurringSchedule), /aggregated:correction\.aggregated/);
  assert.match(surface.slice(recurringSchedule), /dt:coarseBootstrappedThisStep\?0:dt_s,/,
    "a cold coarse bootstrap and its same-command corrected republication must not advect the same interval twice");
  assert.doesNotMatch(surface, /coarseInitializedThisStep/,
    "cold bootstrap coarse phi is an initialization dependency, not a substitute for restricting the accepted fine generation");
  assert.match(surface.slice(recurringSchedule), /generation:correctedFine\.generation&1073741823/,
    "the compact-coarse publication epoch must come from the fine source it restricts, not an optimistic host counter");
  const schedule = compact(WebGPUOctreePowerCoarseLevelSet.prototype.encode);
  const prepare = schedule.indexOf("run(this.preparePipeline");
  const selectors = schedule.indexOf("run(this.buildSelectorRowsPipeline");
  const advect = schedule.indexOf("run(this.advectPipeline");
  const correct = schedule.indexOf("run(this.correctPipeline");
  const redistance = schedule.indexOf("for(constredistanceofthis.redistancePipelines)");
  const publish = schedule.indexOf("run(this.publishPipeline");
  const commit = schedule.indexOf("run(this.commitPipeline");
  assert.ok(prepare >= 0 && prepare < selectors && selectors < advect
    && advect < correct && correct < redistance
    && redistance < publish && publish < commit,
  "coarse evolution must remain prepare -> selector rows -> advect -> fine correction -> redistance -> publish -> commit");
});

test("production substep keeps topology, pressure/power projection, and fine surface recurrence in one command stream", () => {
  const advance = compact(WebGPUUniformEulerianSolver.prototype.advanceTo);
  const rebuild = advance.indexOf("this.octreeProjection.encodeInlineRebuild(encoder");
  const projection = advance.indexOf("this.octreeProjection.encode(encoder", rebuild);
  const surface = advance.indexOf("this.adaptiveProjection.encodeSurface(encoder", projection);
  assert.ok(rebuild >= 0 && projection > rebuild && surface > projection,
    "each production substep must rebuild coarse topology, solve/project power rows, then transport fine phi");
  assert.doesNotMatch(advance.slice(rebuild, surface), /queue\.submit|mapAsync|getMappedRange/,
    "the vertical slice must rely on command-stream ordering rather than a CPU authority handoff");
  const finish = advance.indexOf("constsubmittedEncoder=encoder", surface);
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

test("global-fine QA diagnostics read the published GPU controls without steering simulation", () => {
  const diagnostics = compact(WebGPUOctreeProjection.prototype.readGlobalFineLevelSetDiagnostics);
  assert.match(diagnostics, /this\.globalFineSeeds\.buffer,0,readback,0,8/,
    "seed count and seed fault must come from the actual GPU seed header");
  assert.match(diagnostics, /topology\.control,0,readback,8,32/,
    "topology discovery and publication state must come from the matching generation gate");
  assert.match(diagnostics, /fine\.worklist,0,readback,40,20/,
    "active count and generation must come from the published fine worklist");
  assert.match(diagnostics,
    /this\.powerCoarseLevelSetSchedule\.sampleSource\.directory,0,readback,64,32/,
    "generation diagnostics must include the actual compact-coarse directory header");
  assert.match(diagnostics, /this\.powerCoarseLevelSetSchedule\.control,0,readback,96,64/,
    "generation diagnostics must distinguish a stale tag from a rejected coarse transaction");
  assert.match(diagnostics, /this\.fineToPowerCoarseLevelSet\.control,0,readback,160,32/,
    "generation diagnostics must expose rejection between fine publication and coarse correction");
  assert.match(diagnostics, /this\.lastGlobalFineTransport\.control,0,readback,192,32/,
    "the ABI-stable transport prefix must remain at its established offset");
  assert.match(diagnostics, /this\.lastGlobalFineTransport\.control,32,readback,864,32/,
    "the exact invalid-velocity status and position suffix must be appended without moving existing controls");
  assert.match(diagnostics, /redistance\.control,0,readback,224,16/,
    "the stable redistance prefix remains available at its established packet offset");
  assert.match(diagnostics, /this\.globalFineVolumeA\.control,0,readback,240,64/,
    "volume rejection telemetry must retain its complete shared control");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.transitionControl,0,readback,496,64/,
    "face-band rejection telemetry must retain the catalog-Delaunay gate preceding face emission");
  assert.match(diagnostics,
    /redistance\.control,0,readback,720,FINE_LEVELSET_REDISTANCE_CONTROL_BYTES/,
    "redistance rejection telemetry must retain its complete ten-word control");
  assert.match(diagnostics, /label:"GlobalfineQAdiagnostics",size:1540/,
    "the compact evidence packet accounts for transport detail, Section 5 completion, and native Power construction controls");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.control,64,readback,768,32/,
    "the packet retains direct closest-point and liquid-interpolation metrics without legacy scheduler controls");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.control,96,readback,900,16/,
    "the packet appends the four mutually exclusive CPT interpolation-failure reasons");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.control,112,readback,1108,16/,
    "the packet appends direct face slots for every CPT interpolation-failure reason");
  assert.match(diagnostics, /this\.powerDescriptor\.control,0,readback,1124,64/,
    "the packet attributes rejected native descriptors without a second diagnostic transaction");
  assert.match(diagnostics, /this\.powerTopology\.control,0,readback,1188,64/,
    "the packet attributes rejected native topology metrics without a second diagnostic transaction");
  assert.match(diagnostics,
    /this\.globalFineFaceExtension\.candidateControlForDiagnostics,0,readback,1252,128/,
    "the packet preserves the rejected Section 5 candidate rather than reporting an empty retained authority");
  assert.match(diagnostics,
    /this\.globalFineFaceExtension\.candidateTransitionControlForDiagnostics,0,readback,1380,160/,
    "the packet preserves the complete rejected transition candidate");
  assert.match(diagnostics,
    /faceBandControl:\[\.\.\.words\.slice\(76,92\),\.\.\.words\.slice\(192,200\),\.\.\.words\.slice\(225,229\),\.\.\.words\.slice\(277,281\)\]/,
    "the face-band decoder includes the appended CPT reason suffix");
  assert.match(diagnostics, /topology\.control,32,readback,896,4/,
    "the pre-dilation interface prefix is attributable without moving the stable topology header");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.pointFieldControl,0,readback,560,32/,
    "final cell-centre LS failures must be attributable independently of graph construction");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.transientPowerControl,0,readback,592,64/,
    "all-band physical graph failures must retain their complete fixed header");
  assert.match(diagnostics, /this\.globalFineFaceExtension\.transitionControl,64,readback,656,64/,
    "the first exact-owner mismatch must be available without a second unbounded diagnostic scan");
  assert.match(diagnostics, /published:words\[6\]!==0,rolledBack:words\[7\]!==0/);
  assert.match(diagnostics, /interfaceSeedBricks:words\[224\]/);
  assert.match(diagnostics, /activeBricks:words\[10\],generation:words\[11\]/);
  assert.match(diagnostics,
    /coarseDirectoryState:words\[16\],coarseDirectoryGeneration:words\[17\][\s\S]*coarseControlGeneration:words\[34\],coarseControlValid:words\[35\]/);
  assert.match(diagnostics,
    /fineRestrictionCount:words\[40\][\s\S]*fineRestrictionFlags:words\[42\][\s\S]*fineRestrictionValid:words\[45\]/);
  assert.match(diagnostics,
    /transportControl:this\.lastGlobalFineTransport\?\[\.\.\.words\.slice\(48,56\),\.\.\.words\.slice\(216,224\)\]:Array\.from\(words\.slice\(48,56\)\)[\s\S]*redistanceControl:Array\.from\(words\.slice\(56,60\)\)[\s\S]*redistanceControlDetailed:Array\.from\(words\.slice\(180,190\)\)[\s\S]*volumeControl:Array\.from\(words\.slice\(60,76\)\)[\s\S]*faceBandTransitionControl:Array\.from\(words\.slice\(124,140\)\)[\s\S]*faceBandTransitionOwnerFailure:Array\.from\(words\.slice\(164,180\)\)[\s\S]*faceBandPointFieldControl:Array\.from\(words\.slice\(140,148\)\)[\s\S]*faceBandTransientPowerControl:Array\.from\(words\.slice\(148,164\)\)/);
  assert.match(diagnostics,
    /faceBandPointField:unpackOctreeFaceBandPointFieldControl\(words\.slice\(140,148\)\)[\s\S]*faceBandTransientPower:unpackOctreeFaceBandTransientPowerControl\(words\.slice\(148,164\)\)/,
    "raw evidence must also expose decoded stage attribution without granting CPU authority");

  const simulation = [
    WebGPUOctreeProjection.prototype.encode,
    WebGPUOctreeProjection.prototype.encodeSurface,
    WebGPUOctreeProjection.prototype.encodeSparseBrickWorld,
  ].map(compact).join("");
  assert.doesNotMatch(simulation, /readGlobalFineLevelSetDiagnostics/,
    "QA counters must never influence topology, publication, or scheduling decisions");
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
  assert.equal(info.globalFineTransportFaceBandUnavailable, 5);
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
