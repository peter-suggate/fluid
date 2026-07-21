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

  for (const field of ["surfaceGeometrySource", "globalFineCrossingPublished", "presentationFallbackActive"]) {
    assert.match(renderer, new RegExp(field), `${field} must cross the renderer metrics boundary`);
  }
  assert.match(controller, /waterSurfacePresentation:\s*metrics\.waterSurfacePresentation\s*\?\?\s*null/);
  assert.match(pipeline, /query\.get\("diagnostics"\) === "1"/,
    "opening the canonical diagnostics panel must enable its presentation readback");
  assert.match(panel, /testId="water-surface-presentation-source"/);
  assert.match(panel, /GLOBAL FINE \/ COARSE/);
  assert.match(panel, /ADAPTIVE FALLBACK/);
  assert.match(panel, /RETAINED PREVIOUS MESH/);
  assert.match(panel, /presentation fallback only · solver authority unchanged/);
  assert.match(panel, /pressureSolver\?\.includes\("Section 4\.3 hybrid"\)/);
  assert.match(panel, /POWER \+ SECTION 4\.3/,
    "the balanced authoritative solver must not be mislabeled as Chebyshev in acceptance diagnostics");
  assert.match(panel, /authoritative global-fine field/);
  assert.match(panel, /unavailable — no authoritative field published/);
  assert.match(panel, /frontTelemetrySource/);
  assert.match(panel, /volumeTelemetrySource/);
});
