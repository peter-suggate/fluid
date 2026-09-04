import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_PIXEL_TRACE_KINDS,
  svoPixelTraceNarrative,
  type SvoPixelTrace,
} from "../lib/svo/svo-pixel-trace";
import {
  createSvoDrySceneFragmentWGSL,
  svoDryScenePixelProbeOptions,
} from "../lib/svo/webgpu-svo-dry-scene";
import { createSvoPixelTraceProbeWGSL } from "../lib/svo/webgpu-svo-pixel-trace";

const functionBody = (shader: string, name: string) => {
  const start = shader.indexOf(`fn ${name}(`);
  assert.notEqual(start, -1, `${name} must be present`);
  const end = shader.indexOf("\nfn ", start + 1);
  return shader.slice(start, end < 0 ? undefined : end);
};

test("the traced-pixel brick walk mirrors production occupancy and contour clamps", () => {
  const shader = createSvoDrySceneFragmentWGSL(1, "canonical", "bounds", "inline", 0, true);
  const probe = functionBody(shader, "probeTraceLeafPayload");
  assert.match(probe, /svoBrickOccupancyDecode\(leafNode\.links\.w\)/);
  assert.match(probe, /svoBrickOccupiedBounds\(brickSummary,bounds\[0\],extent\)/);
  assert.match(probe, /cellLower=vec3i\(brickSummary\.minInclusive\)/);
  assert.match(probe, /svoBrickContourClamp\(contour/);
  assert.match(probe, /entry=contourSpan\.y/);
  assert.match(probe, /brickExit=contourSpan\.z/);
});

test("the probe follows an explicitly disabled production brick accelerator", () => {
  const probe = createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions("traced", {
    brickOccupancyMode: "off",
    brickContour: false,
  }));
  const walk = functionBody(probe, "probeTraceLeafPayload");
  assert.doesNotMatch(walk, /svoBrickOccupancyDecode/);
  assert.doesNotMatch(walk, /svoBrickContourClamp/);
});

test("the traced narrative reports aggregate and per-brick cell work", () => {
  const trace = {
    primaryMode: "traced",
    counters: {
      nodeVisits: 23, leafVisits: 8, emptyBrickSkips: 7, voxelWork: 56,
      exactTests: 0, maximumDepth: 0, shadowNodeVisits: 0, shadowLeafVisits: 0,
      shadowWork: 0, mipSteps: 0, traversalFailure: 0, shadedLights: 0,
    },
    records: Array.from({ length: 56 }, (_, order) => ({
      kind: SVO_PIXEL_TRACE_KINDS.brickCell, order, level: 0, detail: 0, flags: 0,
      a: [0, 0, 0], b: [0, 0, 0], tEnter_m: 0, tExit_m: 0,
    })),
  } as unknown as SvoPixelTrace;
  const cells = svoPixelTraceNarrative(trace).find((step) => step.id === "cells");
  assert.equal(cells?.label, "Walk leaf bricks");
  assert.match(cells?.detail ?? "", /56 recorded cells across 8 reached bricks \(7\.0 steps per brick on average\)/);
});
