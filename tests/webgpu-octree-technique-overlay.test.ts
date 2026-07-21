import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_TECHNIQUE_OVERLAY_CODES,
  OCTREE_TECHNIQUE_OVERLAY_MODES,
  isOctreeTechniqueOverlayMode,
} from "../lib/octree-technique-debug";
import { OctreeTechniqueOverlayPipeline } from "../lib/webgpu-octree-technique-overlay";
import { OctreeTechniqueAuditOverlayPipeline } from "../lib/webgpu-octree-technique-audit-overlay";

const overlaySource = readFileSync(new URL("../lib/webgpu-octree-technique-overlay.ts", import.meta.url), "utf8");
const auditOverlaySource = readFileSync(new URL("../lib/webgpu-octree-technique-audit-overlay.ts", import.meta.url), "utf8");
const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");

test("paper-technique modes have stable non-legacy uniform codes", () => {
  assert.deepEqual([...OCTREE_TECHNIQUE_OVERLAY_MODES], [
    "power-cells", "power-faces", "delaunay-tetrahedra", "transition-band", "power-operator",
    "octree-lifecycle", "fine-band-lifecycle", "operator-diagonal", "operator-rhs",
    "operator-reciprocity", "operator-open-fraction", "tetra-validity", "section5-face-band", "global-fine-phi",
  ]);
  assert.deepEqual(Object.values(OCTREE_TECHNIQUE_OVERLAY_CODES), [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]);
  for (const mode of OCTREE_TECHNIQUE_OVERLAY_MODES) assert.equal(isOctreeTechniqueOverlayMode(mode), true);
  assert.equal(isOctreeTechniqueOverlayMode("structure"), false);
});

test("technique overlay composes independently from live compact GPU buffers", () => {
  assert.match(overlaySource, /loadOp:"load",storeOp:"store"/);
  assert.match(overlaySource, /tetrahedronHeaders/);
  assert.match(overlaySource, /faceNormals/);
  assert.match(overlaySource, /faceCentroids/);
  assert.match(overlaySource, /incidenceRows/);
  assert.match(overlaySource, /Octree Section 5 face-band overlay/);
  assert.match(overlaySource, /@binding\(7\) var<storage,read> transitionControl/,
    "the Section 5 overlay must stay within seven fragment storage bindings");
  assert.match(overlaySource, /@binding\(8\) var<storage,read> finePhi:array<f32>/,
    "the paper φ view must read the direct factor-m field without a CPU mirror");
  assert.match(overlaySource, /abs\(length\(gradient\)-1\.0\)/,
    "the paper φ view must expose the signed-distance Eikonal residual");
  assert.match(overlaySource, /first==f\.globalFace\|\|first==index/,
    "bounded owner telemetry must preserve the existing first-error spatial face highlight");
  assert.match(overlaySource, /var acceptedInk=0\.0;var trialInk=0\.0;var unresolvedInk=0\.0/,
    "accepted, heap-trial, and unresolved faces must remain spatially distinct inside a mixed row");
  assert.match(overlaySource, /if\(firstMatch\)\{color=vec3f\(1\.0,0\.01,0\.05\);alpha=0\.98;\}/,
    "the exact first-error key must win over row and per-face march-state colors");
  assert.doesNotMatch(overlaySource, /Absence of ROW_PHI on them is therefore valid/,
    "endpoint rows now carry parent-edge φ and must pass the same scalar validity audit");
  assert.doesNotMatch(overlaySource, /mapAsync|copyBufferToBuffer|readback/i,
    "the display must not read topology back or create a dense diagnostic mirror");
  assert.doesNotMatch(auditOverlaySource, /mapAsync|copyBufferToBuffer|readback/i,
    "validity audits must remain observational GPU consumers");
  assert.match(overlaySource, /i32\(round\(u\.debug\.x\)\)==4/,
    "the primary technique views must have a true camera-ray volume path");
  assert.match(auditOverlaySource, /let volume=i32\(round\(u\.debug\.x\)\)==4/,
    "operator and tetra audits must have an adaptive volume path");
  assert.match(rendererSource, /techniqueOverlayPipeline\?\.setSource\(this\.gpuFluid\?\.octreeTechniqueDebugSource\)/,
    "the warmed compact source must attach independently of first-step water presentation");
  assert.match(rendererSource, /if\(techniqueModeCode\)\{/);
  assert.match(rendererSource, /techniqueOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
  assert.match(rendererSource, /techniqueAuditOverlayPipeline\?\.encode\(encoder,overlayView,techniqueModeCode\)/);
  assert.match(rendererSource, /gridOverlay\?\.axis === "volume" \? 4 : 0/,
    "volume must have a distinct shader geometry code rather than masquerading as a slice");
  assert.match(rendererSource, /if\(gridOverlay\.axis!=="volume"\)this\.gridOverlayPipeline\?\.encode/,
    "the planar legacy overlay must not run underneath a full-volume paper view");
});

test("render panel exposes and explains exact paper structures", () => {
  for (const label of ["Power cells", "Power faces", "Tetrahedra", "Transitions", "Operator"]) {
    assert.match(panelSource, new RegExp(`>${label}<`));
  }
  assert.match(panelSource, /exact GPU power catalog, generalized faces, incidence rows, and local Delaunay tetrahedra/);
  assert.match(panelSource, /red · missing owner, descriptor, or catalog metric/);
});

test("render panel exposes the second-tranche lifecycle and validity audits", () => {
  for (const label of [
    "Octree lifecycle", "Fine band", "Face march", "Diagonal", "RHS", "Reciprocity", "Open fraction", "Tetra validity",
  ]) assert.match(panelSource, new RegExp(`>${label}<`));
  for (const mode of [
    "octree-lifecycle", "fine-band-lifecycle", "section5-face-band", "operator-diagonal", "operator-rhs",
    "operator-reciprocity", "operator-open-fraction", "tetra-validity",
  ]) assert.match(panelSource, new RegExp(`gridOverlayMode === "${mode}"`));
  assert.match(panelSource, /Full volume exposes every active and retired rebuild tile at once/);
  assert.match(panelSource, /live sparse hash, page metadata, worklist, sample flags, and transaction controls/);
  assert.match(panelSource, /paired endpoint\/sign and reverse-CSR incidence agree/);
  assert.match(panelSource, /area-weighted incident-face fraction/);
  assert.match(panelSource, /selector, degeneracy, or reconstructed-volume mismatch/);
  assert.match(panelSource, /\{globalFineVolumeEstimate \? "pre-correction occupancy" : "volume"\} \{driftLabel\(physicalVolumeDrift\)\} · \{volumeSource\}/,
    "the render cockpit must identify global-fine drift as a pre-correction occupancy estimate");
  assert.match(panelSource, /represented \{driftLabel\(representedVolumeDrift\)\}/,
    "the render cockpit must retain represented-volume drift only when it is independent");
  assert.match(panelSource, /engineering supplement, not part of the paper&apos;s Section 5 algorithm/,
    "the implementation's optional global phi shift must not be attributed to the paper");
  assert.match(panelSource, /exact unresolved split: heap-bound trial, accepted-predecessor scheduler defect, or disconnected/,
    "the Section 5 legend must identify the solver-owned unresolved classifier");
});

test("render panel makes paper structures explorable as slices or a full volume", () => {
  assert.match(panelSource, /aria-label="Diagnostic geometry"/);
  assert.match(panelSource, />Slice</);
  assert.match(panelSource, />Full volume</);
  assert.match(panelSource, /setGridOverlayAxis\("volume"\)/);
  assert.match(panelSource, /disabled=\{!octree \|\| !paperVolumeCapable\}/,
    "legacy texture fields must remain slice-only");
  assert.match(panelSource, /Volume opacity/);
  assert.match(panelSource, /camera ray through the complete live structure with front-to-back alpha compositing/);
  assert.match(panelSource, /Volume opacity scales front-to-back compositing/);
  assert.match(panelSource, /inverted, degenerate, non-finite, or catalog-selector mismatch/);
  assert.match(panelSource, /non-positive diagonal, or non-finite coefficient/);
  assert.match(panelSource, /desired → newly activated → resident core\/halo → valid coarse authority or fault/);
});

test("Dawn compiles every technique and audit pipeline at portable binding counts", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU technique-overlay checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
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
  uniform.destroy();
  device.destroy();
});
