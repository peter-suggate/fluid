import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { useDiagnosticsStore } from "../lib/stores/diagnostics-store";

test("water presentation diagnostics default to unavailable rather than claiming authority", () => {
  assert.equal(useDiagnosticsStore.getInitialState().waterSurfacePresentation, null);
});

test("renderer presentation source reaches the diagnostics store and panel with honest fallback copy", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const pipeline = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  const controller = readFileSync(new URL("../lib/simulation/controller.ts", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../components/DiagnosticsPanel.tsx", import.meta.url), "utf8");
  const methodPanel = readFileSync(new URL("../components/MethodPanel.tsx", import.meta.url), "utf8");

  for (const field of ["surfaceGeometrySource", "globalFineAttachedGeneration", "meshPublicationGeneration", "globalFineCrossingPublished", "presentationFallbackActive"]) {
    assert.match(renderer, new RegExp(field), `${field} must cross the renderer metrics boundary`);
  }
  assert.match(controller, /waterSurfacePresentation:\s*metrics\.waterSurfacePresentation\s*\?\?\s*null/);
  assert.doesNotMatch(pipeline, /surfaceDiagnosticsReadbackEnabled|performanceReadbacksEnabled|setPerformanceReadbacksEnabled/,
    "functional presentation evidence must not be controlled by retired performance instrumentation");
  assert.match(pipeline, /if \(this\.surfaceDiagnosticPending \|\| !this\.indirectBuffer\) return;/,
    "bounded presentation evidence remains available independently of performance traces");
  assert.match(pipeline, /lastSurfaceDiagnosticEncodeAt_ms < 250/,
    "the ordinary presentation proof must use the same bounded cadence as solver telemetry");
  assert.match(pipeline, /surfaceDiagnosticsFullRateRequested/,
    "explicit diagnostics and Dawn captures may request per-capture presentation evidence");
  assert.match(pipeline, /query\.get\("panel"\) === "diagnostics"/,
    "the diagnostics panel is the browser authority for full-rate presentation evidence");
  assert.doesNotMatch(pipeline, /query\.get\("(?:waterdiag|diagnostics)"\)/,
    "retired browser switches must not retain renderer behavior");
  assert.match(panel, /testId="water-surface-presentation-source"/);
  assert.match(panel, /GLOBAL FINE \/ COARSE/);
  assert.match(panel, /VOLUME FIELD/);
  assert.doesNotMatch(panel, /ADAPTIVE FALLBACK/);
  assert.match(panel, /RETAINED PREVIOUS MESH/);
  assert.match(panel, /presentation fallback only · solver authority unchanged/);
  assert.match(panel, /isOctreePersistentMGPCGSolverLabel\(gpuInfo\?\.pressureSolver\)/);
  assert.match(methodPanel, /isOctreePersistentMGPCGSolverLabel\(gpuInfo\?\.pressureSolver\)/);
  assert.doesNotMatch(`${panel}\n${methodPanel}`, /Section 4\.3 hybrid/);
  assert.match(panel, /POWER \+ SECTION 4\.3/,
    "the sole production paper solver must be visible");
  assert.doesNotMatch(panel, /Galerkin|POWER \+ CHEBYSHEV/,
    "retired pressure authorities must not remain selectable or labeled");
  assert.match(panel, /authoritative global-fine field/);
  assert.match(panel, /unavailable — no authoritative field published/);
  assert.match(panel, /frontTelemetrySource/);
  assert.match(panel, /volumeTelemetrySource/);
});
