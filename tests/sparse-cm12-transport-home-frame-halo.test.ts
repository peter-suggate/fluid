import assert from "node:assert/strict";
import test from "node:test";

import {
  createSparseCM12TransportHomeFrameHaloLayout,
  SPARSE_CM12_HOME_FRAME_HALO_WGSL_API,
} from "../lib/methods/adaptive-mass/sparse-cm12-transport-home-frame-halo";
import { createSparseCM12TransportHomeFrameHaloWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-transport-home-frame-halo.wgsl";

const functionBody = (source: string, name: string, nextName: string) =>
  source.slice(source.indexOf(`fn ${name}`), source.indexOf(`fn ${nextName}`));

test("home-frame halo fits the portable workgroup-storage budget", () => {
  const layout = createSparseCM12TransportHomeFrameHaloLayout();
  assert.deepEqual(layout, {
    packetEdge: 4, maximumRadius: 3, maximumEdge: 10, capacity: 1000,
    workgroupSize: 64, cellOrdinalBytes: 4000, effectiveVelocityBytes: 16000,
    metadataBytes: 64, totalWorkgroupBytes: 20064,
  });
  assert.ok(layout.totalWorkgroupBytes <= 32 * 1024);
});

test("staging covers the 4^3 packet plus a clamped radius-three halo", () => {
  const wgsl = createSparseCM12TransportHomeFrameHaloWGSL();
  assert.match(wgsl, /array<u32,1000>/);
  assert.match(wgsl, /array<vec4f,1000>/);
  const stage = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.stage,
    SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.lookupAtFine);
  assert.match(stage, /min\(radiusEstimate,CM12_HOME_HALO_MAX_RADIUS\)/);
  assert.match(stage, /laneX=lane&3u[\s\S]*laneY=\(lane>>2u\)&3u[\s\S]*laneZ=\(lane>>4u\)&3u/);
  assert.match(stage, /cm12TeiOwnerAtFine\(q\)/);
  assert.match(stage, /cm12EffectiveTransportVelocity\(owner\.cell\)/);
  assert.match(stage,
    /if\(owner\.cell!=INVALID&&sameHomeSpan\)[\s\S]*cm12EffectiveTransportVelocity\(owner\.cell\)/);
  assert.equal((stage.match(/workgroupBarrier\(\)/g) ?? []).length, 2);
  assert.doesNotMatch(stage, /\b(?:u32|i32)\([^)]*\)\s*\/|\/\s*(?:CM12|[0-9]+u)/,
    "packet staging must not perform integer division");
});

test("cache misses, underestimated travel, and span seams use the exact resolver", () => {
  const wgsl = createSparseCM12TransportHomeFrameHaloWGSL();
  const lookup = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.lookupAtFine,
    "cm12HomeHaloGlobalSampleVelocity");
  assert.match(lookup, /relative>>vec3u\(cm12HomeHaloShift\)/);
  assert.match(lookup, /return cm12HomeHaloGlobalLookup\(q\)/);
  assert.doesNotMatch(lookup.replace(/\/\/.*$/gm, ""), /\//,
    "the hot fine-to-home lookup uses shifts only");

  const sample = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.sampleVelocity,
    "cm12HomeHaloGlobalTransportStencil");
  assert.match(sample,
    /any\(probe\.widths!=cm12HomeHaloHomeWidths\)[\s\S]*cm12HomeHaloGlobalSampleVelocity/);
  assert.match(sample, /cm12HomeHaloLookupAtFine\(vec3i\(floor\(lattice\)\)\)/);
});

test("cached velocity and stencil preserve production corner order and weights", () => {
  const wgsl = createSparseCM12TransportHomeFrameHaloWGSL();
  const sample = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.sampleVelocity,
    "cm12HomeHaloGlobalTransportStencil");
  assert.match(sample,
    /for\(var dz=0;dz<2;dz\+=1\)\{for\(var dy=0;dy<2;dy\+=1\)\{for\(var dx=0;dx<2;dx\+=1\)/);
  assert.match(sample, /let wx=select\(1\.0-fraction\.x,fraction\.x,dx==1\)[\s\S]*let wy=[\s\S]*let wz=/);

  const stencil = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.stencil,
    SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.traceCharacteristic);
  assert.match(stencil,
    /offset=vec3i\(i32\(corner&1u\),i32\(\(corner>>1u\)&1u\),i32\(\(corner>>2u\)&1u\)\)/);
  assert.match(stencil,
    /select\(1\.0-fraction\.x,fraction\.x,offset\.x==1\)[\s\S]*select\(1\.0-fraction\.y,fraction\.y,offset\.y==1\)[\s\S]*select\(1\.0-fraction\.z,fraction\.z,offset\.z==1\)/);
});

test("trace API uses the cached sampler for RK2 midpoint and endpoints", () => {
  const wgsl = createSparseCM12TransportHomeFrameHaloWGSL();
  const trace = functionBody(wgsl, SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.traceCharacteristic,
    SPARSE_CM12_HOME_FRAME_HALO_WGSL_API.traceDeparture);
  assert.match(trace, /initial=cm12HomeHaloSampleVelocity\(initialPosition\)/);
  assert.match(trace, /first=cm12HomeHaloSampleVelocity\(traced\)/);
  assert.match(trace, /cm12HomeHaloSampleVelocity\(midpoint\)/);
});
