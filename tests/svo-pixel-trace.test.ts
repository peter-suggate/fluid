import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY,
  SVO_PIXEL_TRACE_FLAGS,
  SVO_PIXEL_TRACE_GI_STATE,
  SVO_PIXEL_TRACE_HEADER,
  SVO_PIXEL_TRACE_HEADER_WORDS,
  SVO_PIXEL_TRACE_KINDS,
  SVO_PIXEL_TRACE_LAYERS,
  SVO_PIXEL_TRACE_LAYER_DEFINITIONS,
  SVO_PIXEL_TRACE_MAGIC,
  SVO_PIXEL_TRACE_RECORD_WORDS,
  SVO_PIXEL_TRACE_SEGMENT_FLOATS,
  SVO_PIXEL_TRACE_STATUS,
  SVO_PIXEL_TRACE_PIN_PATIENCE,
  SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS,
  buildSvoPixelTraceGeometry,
  decodeSvoPixelTrace,
  resolveSvoPixelTracePin,
  resolveSvoPixelTracePinnedFrame,
  svoPixelTraceMipColorLinear,
  svoPixelTraceMipFootprint_m,
  svoPixelTraceMipLadder,
  svoPixelTraceMipSwatch,
  svoPixelTracePinClick,
  svoPixelTraceBufferBytes,
  svoPixelTraceTextureRows,
  svoPixelTraceWordCount,
  svoPixelTraceLayerForKind,
  svoPixelTraceNarrative,
  svoPixelTraceTotalWork,
  type SvoPixelTraceLayer,
} from "../lib/svo-pixel-trace";
import {
  SVO_PIXEL_TRACE_OVERLAY_INSTANCE_BYTES,
  svoPixelTraceOverlayShader,
} from "../lib/webgpu-svo-pixel-trace-overlay";
import { createSvoPixelTraceProbeWGSL, svoPixelTraceProbeRecordCapacity } from "../lib/webgpu-svo-pixel-trace";
import {
  createSvoDrySceneFragmentWGSL,
  svoDryScenePixelProbeOptions,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";

interface RecordInput {
  kind: number;
  level?: number;
  detail?: number;
  flags?: number;
  a?: readonly [number, number, number];
  b?: readonly [number, number, number];
  tEnter?: number;
  tExit?: number;
}

/** Assemble a probe buffer exactly as the shader lays it out. */
function encodeTraceBuffer(options: {
  status?: number;
  pixel?: readonly [number, number];
  origin?: readonly [number, number, number];
  direction?: readonly [number, number, number];
  hitDistance?: number;
  hitNormal?: readonly [number, number, number];
  counters?: Partial<Record<keyof typeof SVO_PIXEL_TRACE_HEADER, number>>;
  /** Level-0 voxel width. Left unset the trace reports no grid, as a frame can. */
  minimumVoxel?: number;
  records?: readonly RecordInput[];
  capacity?: number;
  magic?: number;
  producedOverride?: number;
  droppedOverride?: number;
} = {}) {
  const capacity = options.capacity ?? 32;
  const buffer = new ArrayBuffer(svoPixelTraceBufferBytes(capacity));
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  const header = SVO_PIXEL_TRACE_HEADER;
  words[header.magic] = options.magic ?? SVO_PIXEL_TRACE_MAGIC;
  words[header.status] = options.status ?? SVO_PIXEL_TRACE_STATUS.hit;
  const records = options.records ?? [];
  words[header.recordCount] = options.producedOverride ?? records.length;
  words[header.droppedRecords] = options.droppedOverride ?? 0;
  words[header.pixelX] = options.pixel?.[0] ?? 12;
  words[header.pixelY] = options.pixel?.[1] ?? 34;
  floats.set(options.origin ?? [0, 1, -4], header.rayOrigin);
  floats.set(options.direction ?? [0, 0, 1], header.rayDirection);
  floats[header.hitDistance] = options.hitDistance ?? 4;
  floats[header.minimumVoxel] = options.minimumVoxel ?? 0;
  floats.set(options.hitNormal ?? [0, 0, -1], header.hitNormal);
  for (const [key, value] of Object.entries(options.counters ?? {})) {
    words[SVO_PIXEL_TRACE_HEADER[key as keyof typeof SVO_PIXEL_TRACE_HEADER]] = value;
  }
  records.forEach((record, index) => {
    const base = SVO_PIXEL_TRACE_HEADER_WORDS + index * SVO_PIXEL_TRACE_RECORD_WORDS;
    words[base] = record.kind;
    words[base + 1] = record.level ?? 0;
    words[base + 2] = record.detail ?? 0;
    words[base + 3] = record.flags ?? 0;
    floats.set(record.a ?? [0, 0, 0], base + 4);
    floats.set(record.b ?? [1, 1, 1], base + 7);
    floats[base + 10] = record.tEnter ?? 0;
    floats[base + 11] = record.tExit ?? 1;
  });
  return { words, floats };
}

test("records live in whole texture rows so the readback needs no padding", () => {
  assert.equal(svoPixelTraceWordCount(1), SVO_PIXEL_TRACE_HEADER_WORDS + SVO_PIXEL_TRACE_RECORD_WORDS);
  // A row is 1024 bytes: both a legal copy bytesPerRow and padding-free, which
  // is what lets the copied buffer be decoded as one flat word array.
  assert.equal(SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4 % 256, 0);
  for (const capacity of [64, 1000, SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY]) {
    const rows = svoPixelTraceTextureRows(capacity);
    assert.equal(svoPixelTraceBufferBytes(capacity), rows * SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS * 4);
    assert.ok(rows * SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS >= svoPixelTraceWordCount(capacity),
      "every record word must fit inside the allocated rows");
  }
  assert.throws(() => svoPixelTraceBufferBytes(0), RangeError);
  assert.throws(() => svoPixelTraceBufferBytes(1.5), RangeError);
  assert.throws(() => svoPixelTraceProbeRecordCapacity({ recordCapacity: 8 }), RangeError);
  assert.equal(svoPixelTraceProbeRecordCapacity({}), SVO_PIXEL_TRACE_DEFAULT_RECORD_CAPACITY);
});

test("a trace decodes its header, counters, and records", () => {
  const { words, floats } = encodeTraceBuffer({
    pixel: [640, 360],
    origin: [1, 2, 3],
    direction: [0, 0, 1],
    hitDistance: 7.5,
    hitNormal: [0, 1, 0],
    counters: { nodeVisits: 41, leafVisits: 3, voxelWork: 22, exactTests: 2, maximumDepth: 6, mipSteps: 48, hitOwnerId: 5, hitMaterialId: 9 },
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.hierarchyNode, level: 2, a: [0, 0, 0], b: [8, 8, 8], tEnter: 1, tExit: 9 },
      { kind: SVO_PIXEL_TRACE_KINDS.childRejected, level: 3, a: [0, 0, 0], b: [4, 4, 4] },
      { kind: SVO_PIXEL_TRACE_KINDS.exactTest, flags: SVO_PIXEL_TRACE_FLAGS.hit, a: [1, 1, 1], b: [2, 2, 2] },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  assert.equal(trace.status, SVO_PIXEL_TRACE_STATUS.hit);
  assert.deepEqual(trace.pixel, [640, 360]);
  assert.deepEqual(trace.ray.origin_m, [1, 2, 3]);
  assert.equal(trace.counters.nodeVisits, 41);
  assert.equal(trace.counters.mipSteps, 48);
  assert.equal(trace.records.length, 3);
  assert.equal(trace.records[0].kind, SVO_PIXEL_TRACE_KINDS.hierarchyNode);
  assert.deepEqual(trace.records[0].b, [8, 8, 8]);
  assert.equal(trace.records[2].flags & SVO_PIXEL_TRACE_FLAGS.hit, SVO_PIXEL_TRACE_FLAGS.hit);
  // The hit position is reconstructed from the ray, exactly as the G-buffer does.
  assert.deepEqual(trace.hit?.position_m, [1, 2, 3 + 7.5]);
  assert.equal(trace.hit?.ownerId, 5);
  assert.equal(trace.hit?.materialId, 9);
  assert.equal(svoPixelTraceTotalWork(trace), 41 + 3 + 22 + 2 + 48);
});

test("a trace reports the shipping GI gather state, energy, visibility, and wide-cone taps", () => {
  const encoded = encodeTraceBuffer({
    counters: { mipSteps: 3 },
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample, level: 1, detail: 0, a: [1, 2, 4], b: [0, 1, 0], tEnter: 0.25 },
      { kind: SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample, level: 2, detail: 0, a: [1, 3, 4], b: [0, 1, 0], tEnter: 0.5 },
    ],
  });
  encoded.words[SVO_PIXEL_TRACE_HEADER.giState] = SVO_PIXEL_TRACE_GI_STATE.enabled | SVO_PIXEL_TRACE_GI_STATE.ready;
  encoded.words[SVO_PIXEL_TRACE_HEADER.giConeCount] = 4;
  encoded.words[SVO_PIXEL_TRACE_HEADER.giConeTaps] = 31;
  encoded.floats[SVO_PIXEL_TRACE_HEADER.giVisibility] = 0.42;
  encoded.floats.set([0.3, 0.2, 0.1], SVO_PIXEL_TRACE_HEADER.giRadiance);
  const trace = decodeSvoPixelTrace(encoded.words, encoded.floats);
  assert.ok(trace?.globalIllumination?.ready);
  assert.equal(trace.globalIllumination.coneCount, 4);
  assert.equal(trace.globalIllumination.coneTaps, 31);
  assert.ok(Math.abs(trace.globalIllumination.visibility - 0.42) < 1e-6);
  assert.deepEqual(trace.records.map(({ kind }) => kind), [13, 13]);
  assert.equal(svoPixelTraceLayerForKind(SVO_PIXEL_TRACE_KINDS.globalIlluminationConeSample), "gi-cones");
  assert.deepEqual(svoPixelTraceMipLadder(trace).map(({ level }) => level), [1, 2]);
  const gi = svoPixelTraceNarrative(trace).find(({ id }) => id === "gi");
  assert.match(gi?.detail ?? "", /4 wide cones across the upper hemisphere; bounced RGB 0\.30 \/ 0\.20 \/ 0\.10; broad diffuse visibility 42%/);
  assert.equal(gi?.layer, "gi-cones");
  const giGeometry = buildSvoPixelTraceGeometry(trace, { layers: ["gi-cones"] });
  assert.ok(giGeometry.countsByLayer["gi-cones"] > 0);
  assert.ok(Math.abs(giGeometry.segments[11] - 0.8) < 1e-6,
    "the apex stays bright even though the replay deliberately omits per-tap opacity");
  SVO_PIXEL_TRACE_LAYER_DEFINITIONS["gi-cones"].colorLinear.forEach((channel, index) => {
    assert.ok(Math.abs(giGeometry.segments[8 + index] - channel) < 1e-6, "the hemisphere rails use their own warm colour");
  });
  assert.equal(buildSvoPixelTraceGeometry(trace, { layers: ["cones"] }).countsByLayer["gi-cones"], 0);
});

test("a requested but unavailable GI atlas is explicit instead of masquerading as cone lighting", () => {
  const encoded = encodeTraceBuffer();
  encoded.words[SVO_PIXEL_TRACE_HEADER.giState] = SVO_PIXEL_TRACE_GI_STATE.enabled;
  const trace = decodeSvoPixelTrace(encoded.words, encoded.floats);
  assert.ok(trace?.globalIllumination && !trace.globalIllumination.ready);
  const gi = svoPixelTraceNarrative(trace).find(({ id }) => id === "gi");
  assert.equal(gi?.value, "fallback");
  assert.match(gi?.detail ?? "", /radiance atlas was unavailable/);
});

test("a miss decodes without a hit even when a distance word survives", () => {
  const { words, floats } = encodeTraceBuffer({ status: SVO_PIXEL_TRACE_STATUS.miss, hitDistance: 9 });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  assert.equal(trace.hit, undefined);
});

test("a foreign or truncated buffer decodes to nothing rather than to a guess", () => {
  const foreign = encodeTraceBuffer({ magic: 0xdeadbeef });
  assert.equal(decodeSvoPixelTrace(foreign.words, foreign.floats), undefined);
  const shortBuffer = new ArrayBuffer(16);
  assert.equal(decodeSvoPixelTrace(new Uint32Array(shortBuffer), new Float32Array(shortBuffer)), undefined);
  const badStatus = encodeTraceBuffer({ status: 99 });
  assert.equal(decodeSvoPixelTrace(badStatus.words, badStatus.floats), undefined);
});

test("records past the capture buffer are reported, and the stored prefix stays exact", () => {
  // The shader counts what it could not store; the readback's row padding means
  // its byte length can never be the authority on capacity.
  const { words, floats } = encodeTraceBuffer({
    capacity: 64,
    producedOverride: 5,
    droppedOverride: 3,
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.hierarchyNode },
      { kind: SVO_PIXEL_TRACE_KINDS.leafBrick },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  assert.equal(trace.records.length, 2);
  assert.equal(trace.droppedRecords, 3);
});

test("every record kind maps to exactly one legend layer", () => {
  const kinds = Object.values(SVO_PIXEL_TRACE_KINDS);
  for (const kind of kinds) {
    const layer = svoPixelTraceLayerForKind(kind);
    assert.ok(SVO_PIXEL_TRACE_LAYERS.includes(layer), `${kind} maps into the legend`);
    assert.ok(SVO_PIXEL_TRACE_LAYER_DEFINITIONS[layer].label.length > 0);
  }
});

test("layer swatches convert to linear colour inside the unit range", () => {
  for (const layer of SVO_PIXEL_TRACE_LAYERS) {
    const definition = SVO_PIXEL_TRACE_LAYER_DEFINITIONS[layer];
    assert.match(definition.swatch, /^#[0-9a-f]{6}$/);
    assert.equal(definition.colorLinear.length, 3);
    for (const channel of definition.colorLinear) {
      assert.ok(channel >= 0 && channel <= 1, `${layer} channel ${channel} is normalized`);
    }
    assert.ok(definition.width_px > 0);
  }
});

test("a box record becomes twelve wireframe edges with the layer's colour", () => {
  const { words, floats } = encodeTraceBuffer({
    status: SVO_PIXEL_TRACE_STATUS.miss,
    records: [{ kind: SVO_PIXEL_TRACE_KINDS.hierarchyNode, level: 1, a: [0, 0, 0], b: [2, 2, 2], tEnter: 1, tExit: 3 }],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const geometry = buildSvoPixelTraceGeometry(trace, { layers: ["hierarchy"] });
  assert.equal(geometry.countsByLayer.hierarchy, 12);
  assert.equal(geometry.segmentCount, 12);
  assert.equal(geometry.segments.length, geometry.segmentCount * SVO_PIXEL_TRACE_SEGMENT_FLOATS);
  const expected = SVO_PIXEL_TRACE_LAYER_DEFINITIONS.hierarchy.colorLinear;
  expected.forEach((channel, index) => {
    assert.ok(Math.abs(geometry.segments[8 + index] - channel) < 1e-6, `channel ${index} carries the layer colour`);
  });
  // Every emitted endpoint is a corner of the recorded box.
  for (let segment = 0; segment < geometry.segmentCount; segment += 1) {
    const base = segment * SVO_PIXEL_TRACE_SEGMENT_FLOATS;
    for (const axis of [0, 1, 2]) {
      assert.ok([0, 2].includes(geometry.segments[base + axis]));
      assert.ok([0, 2].includes(geometry.segments[base + 4 + axis]));
    }
  }
});

test("disabled layers emit nothing and the primary ray is its own layer", () => {
  const { words, floats } = encodeTraceBuffer({
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.hierarchyNode, a: [0, 0, 0], b: [1, 1, 1], tEnter: 1, tExit: 2 },
      { kind: SVO_PIXEL_TRACE_KINDS.childRejected, a: [0, 0, 0], b: [1, 1, 1] },
      { kind: SVO_PIXEL_TRACE_KINDS.brickCell, a: [0, 0, 0], b: [1, 1, 1] },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const only = buildSvoPixelTraceGeometry(trace, { layers: ["rejected"] });
  assert.equal(only.countsByLayer.hierarchy, 0);
  assert.equal(only.countsByLayer.cells, 0);
  assert.equal(only.countsByLayer.rejected, 12);
  const none = buildSvoPixelTraceGeometry(trace, { layers: [] });
  assert.equal(none.segmentCount, 0);
  const ray = buildSvoPixelTraceGeometry(trace, { layers: ["primary-ray"] });
  // Camera-to-entry and entry-to-terminus, each a shaft plus its arrowhead; a
  // hit needs no dashed continuation.
  assert.equal(ray.countsByLayer["primary-ray"], 4);
});

test("a cone sample becomes a ring at the recorded radius, facing the march", () => {
  const facets = 8;
  const { words, floats } = encodeTraceBuffer({
    status: SVO_PIXEL_TRACE_STATUS.miss,
    records: [{
      kind: SVO_PIXEL_TRACE_KINDS.coneSample,
      level: 2,
      a: [0, 0, 0],
      b: [0, 1, 0],
      tEnter: 0.5,
      tExit: 0.75,
    }],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const geometry = buildSvoPixelTraceGeometry(trace, { layers: ["cones"], coneRingFacets: facets });
  assert.equal(geometry.countsByLayer.cones, facets);
  for (let segment = 0; segment < geometry.segmentCount; segment += 1) {
    const base = segment * SVO_PIXEL_TRACE_SEGMENT_FLOATS;
    const start = [0, 1, 2].map((axis) => geometry.segments[base + axis]);
    // The ring lies in the plane normal to the march direction (+y here) and on
    // the circle of the recorded radius.
    assert.ok(Math.abs(start[1]) < 1e-6, "ring stays in the plane normal to the cone axis");
    assert.ok(Math.abs(Math.hypot(start[0], start[2]) - 0.5) < 1e-5, "ring sits at the sample radius");
    // Coverage 0.75 drives intensity: 0.28 + 0.72 * 0.75.
    assert.ok(Math.abs(geometry.segments[base + 11] - (0.28 + 0.72 * 0.75)) < 1e-6);
  }
});

test("width scaling is bounded and applies to every emitted segment", () => {
  const { words, floats } = encodeTraceBuffer({
    status: SVO_PIXEL_TRACE_STATUS.miss,
    records: [{ kind: SVO_PIXEL_TRACE_KINDS.leafBrick, a: [0, 0, 0], b: [1, 1, 1] }],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const base = buildSvoPixelTraceGeometry(trace, { layers: ["bricks"] });
  const wide = buildSvoPixelTraceGeometry(trace, { layers: ["bricks"], widthScale: 2 });
  assert.ok(wide.segments[3] > base.segments[3]);
  const clamped = buildSvoPixelTraceGeometry(trace, { layers: ["bricks"], widthScale: 500 });
  assert.ok(clamped.segments[3] <= base.segments[3] * 4 + 1e-6);
});

test("the narrative reports what the shader did, and names a failure when it failed", () => {
  const { words, floats } = encodeTraceBuffer({
    counters: { nodeVisits: 30, leafVisits: 2, emptyBrickSkips: 1, voxelWork: 18, exactTests: 2, maximumDepth: 5, mipSteps: 40 },
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.childAccepted },
      { kind: SVO_PIXEL_TRACE_KINDS.childRejected },
      { kind: SVO_PIXEL_TRACE_KINDS.brickCell },
      { kind: SVO_PIXEL_TRACE_KINDS.shadowRay },
      { kind: SVO_PIXEL_TRACE_KINDS.coneSample },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const steps = svoPixelTraceNarrative(trace);
  const ids = steps.map((step) => step.id);
  assert.deepEqual(ids, ["descend", "leaves", "cells", "exact", "shadow", "cones"]);
  assert.match(steps[0].value, /30 nodes/);
  assert.match(steps[1].detail, /1 crossed without a surface/);
  assert.equal(steps.find((step) => step.id === "failure"), undefined);

  const failed = encodeTraceBuffer({
    status: SVO_PIXEL_TRACE_STATUS.invalid,
    counters: { traversalFailure: 2 },
  });
  const failedTrace = decodeSvoPixelTrace(failed.words, failed.floats);
  assert.ok(failedTrace);
  const failure = svoPixelTraceNarrative(failedTrace).find((step) => step.id === "failure");
  assert.ok(failure);
  assert.match(failure.detail, /fails closed/);
});

test("every narrative step's layer is drawable, so the readout and overlay agree", () => {
  const { words, floats } = encodeTraceBuffer({
    counters: { shadowNodeVisits: 4, mipSteps: 9 },
    records: [{ kind: SVO_PIXEL_TRACE_KINDS.shadowRay }, { kind: SVO_PIXEL_TRACE_KINDS.occlusionConeSample }],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  for (const step of svoPixelTraceNarrative(trace)) {
    if (!step.layer) continue;
    assert.ok(SVO_PIXEL_TRACE_LAYERS.includes(step.layer as SvoPixelTraceLayer));
  }
});

/* ------------------------------------------------------------------------- */
/* Shader composition                                                        */
/* ------------------------------------------------------------------------- */

test("the probe is absent from every production dry-scene composition", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1, "hybrid"), svoDrySceneShader);
  assert.doesNotMatch(svoDrySceneShader, /dryProbeMain|probeRecords|probeRequest/);
  for (const scale of [0.5, 0.25, 0.125] as const) {
    assert.doesNotMatch(createSvoDrySceneFragmentWGSL(scale), /dryProbeMain|probeRecords|probeRequest/);
  }
  assert.doesNotMatch(createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "split"), /dryProbeMain|probeRecords|probeRequest/);
});

test("the probe composition appends one entry point and keeps the production ones", () => {
  const probe = createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "inline", 0, true);
  assert.ok(probe.startsWith(svoDrySceneShader), "the probe is appended, never woven into the production source");
  assert.match(probe, /@fragment fn dryProbeMain/);
  assert.match(probe, /@fragment fn fragmentMain/);
  // One writer, guarded by pixel coordinate rather than by discard semantics.
  assert.match(probe, /if\(u32\(input\.position\.x\)!=0u\|\|u32\(input\.position\.y\)!=0u\)\{return vec4f\(0\.0\);\}/);
  assert.doesNotMatch(probe.slice(svoDrySceneShader.length), /(^|[^a-zA-Z])discard\s*;/);
});

test("the probe requires the inline full-rate composition it mirrors", () => {
  assert.throws(() => createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "inline", 0, true), RangeError);
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "hybrid", "off", "split", 0, true), RangeError);
});

test("the probe records every kind of work it claims to record", () => {
  const probe = createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions());
  for (const [name, kind] of Object.entries(SVO_PIXEL_TRACE_KINDS)) {
    // Terrain and rigid intersectors are reserved kinds: the primary hit they
    // produce is recorded, but their internal steps are not yet instrumented.
    if (name === "terrainStep" || name === "rigidTest") continue;
    assert.match(probe, new RegExp(`${kind}u`), `${name} is written by the probe`);
  }
  assert.match(probe, /fn probeTraceStatic/);
  assert.match(probe, /fn probeTraceLeafPayload/);
  assert.match(probe, /fn probeConeMarch/);
  assert.match(probe, /fn probeLightVisibility/);
  assert.match(probe, /fn probeContactVisibility/);
  assert.match(probe, /fn probeGlobalIllumination/);
  assert.match(probe, /fn probeGiConeMarch/);
});

test("the probe mirrors the production cone step law and calls production helpers", () => {
  const probe = createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions());
  // Step distances, LOD, and coverage blending must be the shipping arithmetic.
  assert.match(probe, /var distance=minimumVoxel\*\.75;/);
  assert.match(probe, /let diameter=max\(minimumVoxel,2\.0\*distance\*tangent\);/);
  assert.match(probe, /distance\+=max\(stepWidth,minimumVoxel\*\.25\);/);
  assert.match(probe, /svoNodeMipCoverageOpacity\(conservativeCoverage,stepWidth\/diameter\)/);
  assert.match(probe, /emitterOffset\*=1\.5;/);
  assert.match(probe, /let result=svoTetraRadianceConeTrace\(config\)/,
    "the diagnostic must report the production radiance gather rather than a second approximation");
  assert.match(probe, /boundedStep>=result\.coneTaps/,
    "the drawn GI cone contains exactly the tap count taken by the shipping integrator");
  // Exact intersection, light sampling, and bias come from the shipping shader.
  for (const helper of ["primitiveHit", "dryLightSample", "dryBiasedVisibilityRayUnit", "dryNodeMipAt", "traceTerrain", "nearestBody", "svoTraceVisibility"]) {
    assert.match(probe, new RegExp(`${helper}\\(`), `${helper} is reused rather than reimplemented`);
  }
  assert.doesNotMatch(probe, /let opaque=traceOpaqueScene\(ro,rd\)/,
    "the probe must not run the complete static primary traversal a second time");
});

test("records go to a storage texture, and the request arrives in a uniform", () => {
  const probe = createSvoPixelTraceProbeWGSL(svoDryScenePixelProbeOptions());
  // The dry pass already spends the whole storage-buffer budget browsers report
  // on Apple silicon, so an eleventh storage buffer would make the diagnostic
  // unavailable exactly where it matters.
  assert.doesNotMatch(probe, /var<storage/);
  assert.match(probe, /var probeRecords:texture_storage_2d<r32uint,write>/);
  assert.match(probe, /var<uniform> probeRequest:vec4u/);
  // A write-only texture cannot be read, so identity is written, never checked.
  assert.match(probe, /if\(probeRequest\.w==0u\)\{return vec4f\(0\.0\);\}/);
  assert.match(probe, new RegExp(`probeWriteWord\\(0u,${SVO_PIXEL_TRACE_MAGIC}u\\)`));
  assert.match(probe, new RegExp(`PROBE_ROW_WORDS:u32=${SVO_PIXEL_TRACE_TEXTURE_ROW_WORDS}u`));
});

test("the probe's record group is separate from group zero and configurable", () => {
  const probe = createSvoPixelTraceProbeWGSL({ ...svoDryScenePixelProbeOptions(), group: 3 });
  assert.match(probe, /@group\(3\) @binding\(0\) var probeRecords/);
  assert.match(probe, /@group\(3\) @binding\(1\) var<uniform> probeRequest/);
  assert.throws(() => createSvoPixelTraceProbeWGSL({ ...svoDryScenePixelProbeOptions(), group: 0 }), RangeError);
  assert.throws(() => createSvoPixelTraceProbeWGSL({ ...svoDryScenePixelProbeOptions(), coneLodBlendBandWidth: 0 }), RangeError);
});

test("the overlay instance layout matches the geometry the host builds", () => {
  assert.equal(SVO_PIXEL_TRACE_OVERLAY_INSTANCE_BYTES, SVO_PIXEL_TRACE_SEGMENT_FLOATS * 4);
  // Four vec4 attributes cover the sixteen interleaved floats.
  for (const location of [0, 1, 2, 3]) {
    assert.match(svoPixelTraceOverlayShader, new RegExp(`@location\\(${location}\\) `));
  }
  // Occlusion ghosts rather than hides, and reversed-Z zero means "nothing here".
  assert.match(svoPixelTraceOverlayShader, /if \(stored>0\.0\)/);
  assert.match(svoPixelTraceOverlayShader, /alpha\*=max\(overlay\.viewport\.w,0\.0\)/);
});

test("a viewport click pins the pixel it aimed at, and the next one goes live", () => {
  const click = { normalizedX: 0.4, normalizedY: 0.7, cameraKey: "view-a", revision: 12 };
  const first = svoPixelTracePinClick({ pinned: false, pending: false, ...click });
  assert.deepEqual(first.request, { normalizedX: 0.4, normalizedY: 0.7, cameraKey: "view-a", revision: 12 });
  // Clicking a pinned ray cannot re-aim it: the camera has been free to orbit
  // away from the pixel that produced it, so a new pixel means nothing yet.
  assert.equal(svoPixelTracePinClick({ pinned: true, pending: false, ...click }).request, undefined);
  // A second click while the first is still in flight cancels it rather than
  // stacking a request the probe would have to disambiguate.
  assert.equal(svoPixelTracePinClick({ pinned: false, pending: true, ...click }).request, undefined);
  // A click outside the viewport rectangle still names a pixel inside it.
  const clamped = svoPixelTracePinClick({
    pinned: false, pending: false, normalizedX: -0.2, normalizedY: 1.4, cameraKey: "view-a", revision: 0,
  });
  assert.deepEqual(clamped.request, { normalizedX: 0, normalizedY: 1, cameraKey: "view-a", revision: 0 });
});

test("a pin waits for its own pixel, and abandons the aim the camera invalidated", () => {
  const request = { normalizedX: 0.5, normalizedY: 0.5, cameraKey: "view-a", revision: 4 };
  const state = { answered: false, cameraKey: "view-a", probeCanAnswer: true, revision: 4 };
  // The readback runs a frame behind its request, so the trace present at the
  // instant of the click belongs to a neighbouring pixel: waiting is the point.
  assert.equal(resolveSvoPixelTracePin(request, state), "wait");
  assert.equal(resolveSvoPixelTracePin(request, { ...state, answered: true }), "pin");
  // An orbit begun before the click landed re-aims the pixel at a different ray,
  // so honouring the click would freeze a ray nobody pointed at. This is the case
  // a heavy scene makes reachable: one frame there can outlast the drag.
  assert.equal(resolveSvoPixelTracePin(request, { ...state, cameraKey: "view-b" }), "abandon");
  assert.equal(resolveSvoPixelTracePin(request, { ...state, answered: true, cameraKey: "view-b" }), "abandon");
  // A path that cannot answer at all must not hold the pointer hostage.
  assert.equal(resolveSvoPixelTracePin(request, { ...state, probeCanAnswer: false }), "abandon");
  // Patience is counted in traces, not milliseconds: a frame that takes seconds
  // is slow, not broken, and a deadline would abandon every click on it.
  assert.equal(resolveSvoPixelTracePin(request, { ...state, revision: 4 + SVO_PIXEL_TRACE_PIN_PATIENCE - 1 }), "wait");
  assert.equal(resolveSvoPixelTracePin(request, { ...state, revision: 4 + SVO_PIXEL_TRACE_PIN_PATIENCE }), "abandon");
});

test("a directional segment emits a shaft and one screen-space arrowhead", () => {
  const { words, floats } = encodeTraceBuffer({
    records: [{
      kind: SVO_PIXEL_TRACE_KINDS.shadowRay,
      a: [0, 0, 0],
      b: [0, 3, 0],
      flags: SVO_PIXEL_TRACE_FLAGS.hit,
    }],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const geometry = buildSvoPixelTraceGeometry(trace, { layers: ["shadow-rays"] });
  assert.equal(geometry.segmentCount, 2);
  const [shaft, head] = [0, 1].map((index) => geometry.segments
    .slice(index * SVO_PIXEL_TRACE_SEGMENT_FLOATS, (index + 1) * SVO_PIXEL_TRACE_SEGMENT_FLOATS));
  // Both instances carry the same endpoints: the head is built from them in the
  // shader, so it cannot drift away from the shaft it terminates.
  assert.deepEqual([...shaft.slice(0, 3)], [...head.slice(0, 3)]);
  assert.deepEqual([...shaft.slice(4, 7)], [...head.slice(4, 7)]);
  // The head width lives in the fourteenth float, and only the head has one.
  assert.equal(shaft[14], 0);
  assert.ok(head[14] >= 4.5 && head[14] <= 13, `head is bounded in device pixels, got ${head[14]}`);
});

test("the primary ray becomes one arrow per brick walked and per gap skipped", () => {
  const { words, floats } = encodeTraceBuffer({
    hitDistance: 9,
    records: [
      // Entered at 1 m, walked 1..2, skipped 2..4, walked 4..9 to the hit.
      { kind: SVO_PIXEL_TRACE_KINDS.hierarchyNode, a: [0, 0, 0], b: [8, 8, 8], tEnter: 1, tExit: 9 },
      { kind: SVO_PIXEL_TRACE_KINDS.leafBrick, flags: SVO_PIXEL_TRACE_FLAGS.emptySkip, tEnter: 1, tExit: 2 },
      { kind: SVO_PIXEL_TRACE_KINDS.leafBrick, flags: SVO_PIXEL_TRACE_FLAGS.hit, tEnter: 4, tExit: 12 },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const geometry = buildSvoPixelTraceGeometry(trace, { layers: ["primary-ray"] });
  // Approach, first brick, the empty gap, the last brick: four arrows, each a
  // shaft plus a head.
  assert.equal(geometry.countsByLayer["primary-ray"], 8);
  const shafts: number[][] = [];
  for (let segment = 0; segment < geometry.segmentCount; segment += 1) {
    const base = segment * SVO_PIXEL_TRACE_SEGMENT_FLOATS;
    if (geometry.segments[base + 14] > 0) continue;
    shafts.push([...geometry.segments.slice(base, base + 8)]);
  }
  // The ray runs +z from z = -4, so an endpoint's z is its distance minus four.
  const spans = shafts.map((shaft) => [shaft[2] + 4, shaft[6] + 4]);
  assert.deepEqual(spans, [[0, 1], [1, 2], [2, 4], [4, 9]]);
  // The chain is gapless and monotone: every arrow starts where the last ended,
  // which is what makes it read as one ray advancing rather than four strokes.
  assert.ok(spans.every(([from, to], index) => to > from && (index === 0 || from === spans[index - 1][1])));
});

test("a cone tap draws the mip voxel it read, at that level's true width", () => {
  const facets = 8;
  const sample = {
    kind: SVO_PIXEL_TRACE_KINDS.coneSample,
    level: 3,
    a: [1, 1, 1] as const,
    b: [0, 1, 0] as const,
    tEnter: 0.5,
    tExit: 0.5,
  };
  const withGrid = encodeTraceBuffer({ minimumVoxel: 0.05, records: [sample] });
  const trace = decodeSvoPixelTrace(withGrid.words, withGrid.floats);
  assert.ok(trace);
  // A float32 round trip through the buffer, so widths compare with a tolerance.
  assert.ok(Math.abs((trace.minimumVoxel_m ?? 0) - 0.05) < 1e-6);
  // Level 3 is the finest width doubled three times.
  assert.ok(Math.abs((svoPixelTraceMipFootprint_m(trace, 3) ?? 0) - 0.4) < 1e-6);
  const geometry = buildSvoPixelTraceGeometry(trace, { layers: ["cones"], coneRingFacets: facets });
  assert.equal(geometry.countsByLayer.cones, facets + 12, "the ring, plus the footprint cube's twelve edges");
  // The cube is centred on the tap and spans the level's voxel width.
  const cube = [...geometry.segments.slice(facets * SVO_PIXEL_TRACE_SEGMENT_FLOATS)];
  for (let segment = 0; segment < 12; segment += 1) {
    const base = segment * SVO_PIXEL_TRACE_SEGMENT_FLOATS;
    for (const axis of [0, 1, 2]) {
      assert.ok([0.8, 1.2].some((value) => Math.abs(cube[base + axis] - value) < 1e-6));
      assert.ok([0.8, 1.2].some((value) => Math.abs(cube[base + 4 + axis] - value) < 1e-6));
    }
  }
  // Cone colour is the level, not the layer: the ladder is what makes the climb
  // from fine to coarse readable along one cone.
  const expected = svoPixelTraceMipColorLinear(3);
  expected.forEach((channel, index) => {
    assert.ok(Math.abs(geometry.segments[8 + index] - channel) < 1e-6);
    assert.ok(Math.abs(channel - SVO_PIXEL_TRACE_LAYER_DEFINITIONS.cones.colorLinear[index]) > 1e-6
      || svoPixelTraceMipSwatch(3) === SVO_PIXEL_TRACE_LAYER_DEFINITIONS.cones.swatch);
  });
  // Without a reported grid there is no footprint to draw, and none is invented.
  const noGrid = encodeTraceBuffer({ records: [sample] });
  const ungridded = decodeSvoPixelTrace(noGrid.words, noGrid.floats);
  assert.ok(ungridded);
  assert.equal(ungridded.minimumVoxel_m, undefined);
  assert.equal(svoPixelTraceMipFootprint_m(ungridded, 3), undefined);
  assert.equal(buildSvoPixelTraceGeometry(ungridded, { layers: ["cones"], coneRingFacets: facets }).countsByLayer.cones, facets);
});

test("consecutive taps of one march stitch into a cone; separate marches do not", () => {
  const facets = 4;
  // Occlusion taps, because only an occlusion march draws its own axis: a shadow
  // march already has a shadow ray along the same line.
  const tap = (detail: number, level: number, radius: number, flags = 0) => ({
    kind: SVO_PIXEL_TRACE_KINDS.occlusionConeSample, detail, level, flags,
    a: [0, radius * 4, 0] as const, b: [0, 1, 0] as const, tEnter: radius, tExit: 0,
  });
  const single = encodeTraceBuffer({ records: [tap(0, 0, 0.1), tap(0, 1, 0.2), tap(0, 2, 0.3)] });
  const one = decodeSvoPixelTrace(single.words, single.floats);
  assert.ok(one);
  // Three rings, two envelope steps of four rails each, and one axis arrow.
  assert.equal(
    buildSvoPixelTraceGeometry(one, { layers: ["cones"], coneRingFacets: facets }).countsByLayer.cones,
    3 * facets + 2 * 4 + 2,
  );
  // Two marches of the same light, or the two phases of one split march, are not
  // one cone: stitching them would draw a cone that reverses halfway along.
  const split = encodeTraceBuffer({
    records: [tap(0, 0, 0.1), tap(1, 1, 0.2), tap(0, 2, 0.3, SVO_PIXEL_TRACE_FLAGS.emitterAnchored)],
  });
  const separate = decodeSvoPixelTrace(split.words, split.floats);
  assert.ok(separate);
  assert.equal(
    buildSvoPixelTraceGeometry(separate, { layers: ["cones"], coneRingFacets: facets }).countsByLayer.cones,
    3 * facets,
  );
});

test("the mip ladder reports the levels the cones read, finest first", () => {
  const { words, floats } = encodeTraceBuffer({
    minimumVoxel: 0.04,
    counters: { mipSteps: 4 },
    records: [
      { kind: SVO_PIXEL_TRACE_KINDS.coneSample, level: 2, detail: 0 },
      { kind: SVO_PIXEL_TRACE_KINDS.occlusionConeSample, level: 0, detail: 0 },
      { kind: SVO_PIXEL_TRACE_KINDS.coneSample, level: 2, detail: 1 },
      { kind: SVO_PIXEL_TRACE_KINDS.leafBrick },
    ],
  });
  const trace = decodeSvoPixelTrace(words, floats);
  assert.ok(trace);
  const ladder = svoPixelTraceMipLadder(trace);
  assert.deepEqual(ladder.map((rung) => [rung.level, rung.taps]), [[0, 1], [2, 2]]);
  // Each level covers twice the width of the one below it.
  assert.ok(Math.abs((ladder[0].footprint_m ?? 0) - 0.04) < 1e-6);
  assert.ok(Math.abs((ladder[1].footprint_m ?? 0) - 0.16) < 1e-6);
  assert.deepEqual(ladder.map((rung) => rung.swatch), [svoPixelTraceMipSwatch(0), svoPixelTraceMipSwatch(2)]);
  // The narrative names the span so the readout says what the colours mean.
  const cones = svoPixelTraceNarrative(trace).find((step) => step.id === "cones");
  assert.match(cones?.detail ?? "", /levels 0–2/);
  // A ray that took no cone samples reports no ladder rather than a level zero.
  const bare = encodeTraceBuffer({ records: [{ kind: SVO_PIXEL_TRACE_KINDS.leafBrick }] });
  const unshaded = decodeSvoPixelTrace(bare.words, bare.floats);
  assert.ok(unshaded);
  assert.deepEqual(svoPixelTraceMipLadder(unshaded), []);
});

test("a scene change re-traces a pinned ray from its own view, and is owned up to from any other", () => {
  const aimed = { pinned: true, aimCameraKey: "view-a", cameraKey: "view-a" };
  // Nothing changed: a pinned ray stays frozen and stays silent.
  assert.deepEqual(resolveSvoPixelTracePinnedFrame({ ...aimed, sceneChanged: false }), { refresh: false, stale: false });
  // Another light, a republished topology, a stepped simulation: the same ray now
  // does different work, and from the view it was aimed from that is exactly the
  // comparison worth showing, so re-trace it in place.
  assert.deepEqual(resolveSvoPixelTracePinnedFrame({ ...aimed, sceneChanged: true }), { refresh: true, stale: false });
  // Orbited since the pin: the pinned pixel names some other ray now, so refreshing
  // would swap rays behind the user's back. Report the numbers as old instead.
  assert.deepEqual(
    resolveSvoPixelTracePinnedFrame({ ...aimed, cameraKey: "view-b", sceneChanged: true }),
    { refresh: false, stale: true },
  );
  // No aim on record is not the same as a matching aim; it cannot be refreshed.
  assert.deepEqual(
    resolveSvoPixelTracePinnedFrame({ pinned: true, aimCameraKey: undefined, cameraKey: "view-a", sceneChanged: true }),
    { refresh: false, stale: true },
  );
  // A live trace needs neither: it re-traces every frame by itself.
  assert.deepEqual(
    resolveSvoPixelTracePinnedFrame({ ...aimed, pinned: false, sceneChanged: true }),
    { refresh: false, stale: false },
  );
});

test("one gate decides whether the probe runs, so a pinned ray cannot half-probe", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  // Requesting a pixel, encoding the probe pass, and pumping its readback must
  // agree: encoding without pumping strands the readback, and pumping without
  // encoding adopts a trace of some other pixel as though it were the pinned one.
  const gate = /pixelTraceProbing/g;
  assert.ok((renderer.match(gate) ?? []).length >= 4, "the probing gate is defined once and used at each site");
  assert.match(renderer, /if \(pixelTraceProbing\) this\.pumpPixelTraceReadback\(\);/);
  assert.match(renderer, /if \(pixelTraceProbing && this\.svoDryScenePipeline\?\.encodePixelTrace\(encoder\)\)/);
  // Nothing may gate probe work on `pinned` alone any more: a pinned ray whose
  // scene changed does probe, exactly once per change.
  assert.doesNotMatch(renderer, /if \(!pixelTrace\.pinned\) this\.svoDryScenePipeline/);
  assert.doesNotMatch(renderer, /!pixelTrace\?\.pinned\) this\.pumpPixelTraceReadback/);
  // The trace carries the scene revision it was encoded against, not whichever
  // one is current when its readback happens to resolve.
  assert.match(renderer, /const encodedSceneRevision = this\.pixelTraceEncodedSceneRevision;/);
  assert.match(renderer, /this\.pixelTraceSceneRevisionOfTrace = encodedSceneRevision;/);
  // The scene key covers the epoch, the presentation, and the traversal tuning.
  assert.match(renderer, /sceneEpoch \?\? 0\}\|\$\{presentationContext\}\|\$\{diagnosticsKey\}/);
});

test("a pinned ray's aim is only ever recorded from a request, never invented", () => {
  const viewport = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
  const panel = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");
  // The aim is written in exactly one place: where a pin request resolves. Any
  // other writer would be inventing one from the current pointer and camera, and
  // a refresh against an invented aim answers a ray the user never pinned.
  const aims = viewport.match(/tracePinnedRef\.current = \{/g) ?? [];
  assert.equal(aims.length, 1, "exactly one place records an aim");
  const clears = viewport.match(/tracePinnedRef\.current = null/g) ?? [];
  assert.equal(clears.length, 2, "cleared when unpinned and when the diagnostic closes");
  assert.match(viewport, /if \(!pinnedNow\) tracePinnedRef\.current = null;/);
  // Neither pin control declares a pin; both ask, so the viewport supplies the aim.
  assert.match(panel, /on \? requestPixelTracePin\(\) : setPixelTracePinned\(false\)/);
  assert.match(viewport, /pixelTracePinned \? setPixelTracePinned\(false\) : requestPixelTracePin\(\)/);
  assert.match(viewport, /ui\.pixelTracePinRequested && !ui\.pixelTracePinned/);
  // While pinned the probe traces the pinned pixel, not wherever the pointer went.
  assert.match(viewport, /const pointer = pinnedAt \?\? tracePinRequestRef\.current \?\? tracePointerRef\.current;/);
});
