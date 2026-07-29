import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { OCTREE_TECHNIQUE_OVERLAY_CODES, OCTREE_TECHNIQUE_OVERLAY_MODES,
  isOctreeTechniqueOverlayMode } from "../lib/octree-technique-debug";
import { OctreeTechniqueOverlayPipeline } from "../lib/webgpu-octree-technique-overlay";
import { OctreeTechniqueAuditOverlayPipeline } from "../lib/webgpu-octree-technique-audit-overlay";

const overlaySource = readFileSync(new URL("../lib/webgpu-octree-technique-overlay.ts", import.meta.url), "utf8");
const auditOverlaySource = readFileSync(new URL("../lib/webgpu-octree-technique-audit-overlay.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const performancePanelSource = readFileSync(new URL("../components/PerformancePanel.tsx", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");
const fineBricksSource = readFileSync(
  new URL("../lib/webgpu-octree-fine-levelset-bricks.ts", import.meta.url), "utf8");

test("paper-technique modes have unique stable uniform codes", () => {
  for (const mode of ["power-cells", "power-faces", "delaunay-tetrahedra", "transition-band",
    "power-operator",
    "octree-lifecycle", "fine-band-lifecycle", "operator-diagonal", "operator-rhs",
    "operator-reciprocity", "operator-open-fraction", "tetra-validity",
    "global-fine-phi", "band-residency", "evaluated-velocity", "projection-update",
    "divergence-closure", "structured-velocity"] as const) {
    assert.equal(isOctreeTechniqueOverlayMode(mode), true, mode);
    assert.ok(OCTREE_TECHNIQUE_OVERLAY_CODES[mode] >= 12, mode);
  }
  assert.equal(new Set(Object.values(OCTREE_TECHNIQUE_OVERLAY_CODES)).size,
    OCTREE_TECHNIQUE_OVERLAY_MODES.length);
  assert.equal(isOctreeTechniqueOverlayMode("structure"), false);
});

test("the band-residency view compares both authored bands against actual residency", () => {
  // Both widths are authored in finest cells, so the shader has to convert with
  // the fine plan's own factor rather than assume fine cells.
  assert.match(overlaySource,
    /let finest=max\(fine\.fineCellWidth,1e-9\)\*f32\(max\(fine\.fineFactor,1u\)\);/,
    "the pressure reach must be measured in finest cells, the unit it is authored in");
  assert.match(overlaySource,
    /if\(distance<=f32\(bands\.pressureBandCells\)\*finest\)\{/,
    "pressure-band membership is a distance test against the authored pressure width");
  // The derived widths are in fine cells, the authored ones in finest cells.
  // Scaling either with the wrong unit silently mis-draws the nesting.
  assert.match(overlaySource,
    /if\(distance<=f32\(bands\.transportBandFineCells\)\*h\)\{/,
    "the transported surface band is a fine-cell width");
  assert.match(overlaySource,
    /if\(distance<=f32\(bands\.redistanceBandFineCells\)\*h\)\{/,
    "the redistance support margin is a fine-cell width");
  assert.match(octreeSource,
    /\.\.\.planFineLevelSetBandFineCells\(this\.fineLevelSetBandCells, fine\.plan\.fineFactor\)/,
    "the view must consume the solver's own planner, not re-derive the widths");
  // The state the view exists for: resident phi inside the pressure reach whose
  // neighbour is not resident. That is where the fine-to-coarse restriction
  // loses the phi a pressure cell centre needs.
  assert.match(overlaySource,
    /if\(fineAddress\(q\+delta\)==INVALID\|\|fineAddress\(q-delta\)==INVALID\)\{truncated=true;\}/,
    "a truncated pressure reach must be detected from residency, not assumed");
  assert.match(overlaySource, /\{binding:4,resource:\{buffer:this\.bandConfig\}\}/,
    "the authored widths must reach the shader");
  assert.equal(OCTREE_TECHNIQUE_OVERLAY_CODES["band-residency"], 26);
  assert.match(overlaySource, /modeCode===18\|\|modeCode===25\|\|modeCode===26/,
    "the mode must route to the fine publication pipeline that owns phi residency");
  assert.match(performancePanelSource, /mode: "band-residency", axis: "volume"/);
});

test("fine lifecycle uses the generation word rather than the active-page count", () => {
  assert.match(fineBricksSource, /\$\{worklist\}\[0\]!=\$\{params\}\.generation/,
    "the shared page lookup must validate generation from header word zero");
  assert.match(overlaySource,
    /makeFineLevelSetSortedWorklistLookupWGSL\("fine", "metadata", "worklist", "pageOf"\)/,
    "the lifecycle shader must use the shared generation-validating lookup");
  assert.doesNotMatch(overlaySource, /worklist\[1\]!=fine\.generation/,
    "header word one is the active-page count, not the publication generation");
});

test("sparse technique slices do not report absent pressure rows as corrupt", () => {
  assert.match(overlaySource, /previous=row;if\(row==INVALID\)\{continue;\}/,
    "power-cell volume traversal must skip unowned sparse space");
  assert.match(overlaySource, /if\(row==INVALID\)\{discard;\}let fault=topologyFault\(row,pointFine\)/,
    "power-cell slices must leave unowned sparse space transparent");
  assert.equal((overlaySource.match(/if\(row==INVALID\)\{discard;\}if\(!rowValid/g) ?? []).length, 2,
    "face and structured-field slices must also distinguish absence from corruption");
  assert.match(auditOverlaySource, /if\(row==INVALID\)\{discard;\}/,
    "audit slices must preserve the same sparse-owner semantics");
});

test("categorical power volumes preserve their slice palettes", () => {
  assert.match(overlaySource, /fn compositeDisplay\(/,
    "categorical volume samples should be converted with the same display mapping as slices");
  assert.match(overlaySource, /return finishDisplayVolume\(accum\)/,
    "categorical volumes must not apply a second display transform after compositing");
  assert.match(overlaySource, /0\.24\+0\.56\*site/,
    "power-cell volume classification should retain enough opacity to preserve categorical color");
  assert.match(overlaySource, /alpha\*select\(0\.96,1\.0,volume\)/,
    "geometry volume opacity should be controlled once at composition, not squared");
  assert.match(overlaySource, /if\(planeInk>=dualInk&&planeInk>=normalInk\)/,
    "overlapping geometry should retain categorical colors instead of adding to white");
});

test("technique overlay composes directly from compact topology and fine publications", () => {
  assert.match(overlaySource, /loadOp:"load",storeOp:"store"/);
  assert.match(overlaySource, /tetrahedronHeaders/);
  assert.match(overlaySource, /@binding\(8\) var<storage,read> finePhi:array<f32>/,
    "the paper phi view must read the direct factor-m field without a CPU mirror");
  assert.match(overlaySource, /abs\(length\(gradient\)-1\.0\)/,
    "the paper phi view must expose signed-distance Eikonal residual");
  assert.match(overlaySource,
    /fn renderWorldToFine[\s\S]*vec3f\(-0\.5\*u\.container\.x,0\.0,-0\.5\*u\.container\.z\)[\s\S]*\(point-minimum\)/);
  assert.doesNotMatch(overlaySource, /mapAsync|copyBufferToBuffer|readback/i);
  assert.doesNotMatch(auditOverlaySource, /mapAsync|copyBufferToBuffer|readback/i);
  assert.match(overlaySource, /i32\(round\(u\.debug\.x\)\)==4/);
  assert.match(overlaySource, /return globalFinePhiVolume\(input\.uv\)/,
    "fine phi must retain its own volume presentation");
  assert.match(overlaySource, /modeCode===13\|\|modeCode===16/,
    "power faces and the power operator must route to their real catalog pipeline");
  assert.match(overlaySource, /modeCode>=27&&modeCode<=30/,
    "compact velocity, projection, and divergence fields must route to structured authority");
  assert.match(overlaySource, /if\(mode==30\)\{let direction=/,
    "structured velocity must show vector direction instead of duplicating evaluated speed");
  assert.match(overlaySource, /source\.catalogEntryHeaders/);
  assert.match(overlaySource, /source\.structuredAuthority/);
  assert.match(overlaySource, /source\.pressure/);
  assert.match(auditOverlaySource, /let volume=i32\(round\(u\.debug\.x\)\)==4/);
  assert.match(rendererSource,
    /techniqueOverlayPipeline\?\.setSource\(this\.gpuFluid\?\.octreeTechniqueDebugSource\)/);
  assert.match(rendererSource,
    /techniqueOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
  assert.match(rendererSource,
    /techniqueAuditOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
});

test("performance observatory owns topology, fine, pressure, and velocity inspection", () => {
  for (const label of ["Fine SDF layer", "Adaptive coarse grid", "Fine signed distance",
    "Band slice", "Power cells", "Power face geometry", "Sparse topology lifecycle",
    "Evaluated pressure", "Evaluated velocity", "Pressure update Δu", "Divergence closure",
    "Structured velocity", "Power operator"]) assert.match(performancePanelSource, new RegExp(`label: "${label}"`));
  assert.match(performancePanelSource, /aria-label="Paper field view plane"/);
  assert.match(performancePanelSource, />VOLUME<\/button>/);
  assert.match(performancePanelSource, />HIDE<\/button>/);
  assert.match(performancePanelSource, /setOverlayAxis\("volume"\)/);
  assert.match(performancePanelSource, /VOLUME OPACITY/);
});

test("Dawn compiles every technique and audit pipeline at portable binding counts", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU technique-overlay checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) =>
    errors.push((event as { error: { message: string } }).error.message));
  const uniform = device.createBuffer({ size: 400, usage: GPUBufferUsage.UNIFORM });
  const overlay = new OctreeTechniqueOverlayPipeline(device, "rgba8unorm", uniform);
  const auditOverlay = new OctreeTechniqueAuditOverlayPipeline(device, "rgba8unorm", uniform);
  try {
    await Promise.all([overlay.initialize(), auditOverlay.initialize()]);
  } catch (error) {
    await new Promise((resolve) => setImmediate(resolve));
    assert.fail(`Technique pipeline creation failed: ${error instanceof Error ? error.message : String(error)} ${errors.join(" | ")}`);
  }
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  uniform.destroy(); device.destroy();
});
