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
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
const octreeSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("paper-technique modes have unique stable uniform codes", () => {
  for (const mode of ["power-cells", "delaunay-tetrahedra", "transition-band",
    "octree-lifecycle", "fine-band-lifecycle", "operator-diagonal", "operator-rhs",
    "operator-reciprocity", "operator-open-fraction", "tetra-validity",
    "global-fine-phi", "band-residency"] as const) {
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
  assert.match(panelSource, /"band-residency": "PRESSURE vs SURFACE BAND"/);
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
  assert.match(auditOverlaySource, /let volume=i32\(round\(u\.debug\.x\)\)==4/);
  assert.match(rendererSource,
    /techniqueOverlayPipeline\?\.setSource\(this\.gpuFluid\?\.octreeTechniqueDebugSource\)/);
  assert.match(rendererSource,
    /techniqueOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
  assert.match(rendererSource,
    /techniqueAuditOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
});

test("render panel exposes topology, fine, pressure, and validity inspection", () => {
  for (const label of ["Power cells", "Tetrahedra", "Transitions", "Operator",
    "Octree lifecycle", "Fine band", "Diagonal", "RHS", "Reciprocity",
    "Open fraction", "Tetra validity"]) assert.match(panelSource, new RegExp(`>${label}<`));
  assert.match(panelSource, /aria-label="Diagnostic geometry"/);
  assert.match(panelSource, />Slice</);
  assert.match(panelSource, />Full volume</);
  assert.match(panelSource, /setGridOverlayAxis\("volume"\)/);
  assert.match(panelSource, /Volume opacity/);
  assert.match(panelSource,
    /Residual next to the white zero crossing audits redistancing; interior equal-distance ridges are nondifferentiable/);
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
