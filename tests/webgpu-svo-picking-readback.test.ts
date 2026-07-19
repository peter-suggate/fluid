import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  packSvoGBufferPixel,
  SVO_GBUFFER_LAYOUT,
  SVO_GBUFFER_FIELD_SOURCES,
  SVO_GBUFFER_MOTION_KINDS,
  type SvoGBufferHit,
} from "../lib/svo-gbuffer";
import {
  decodeSvoGpuPickingSample,
  SVO_GPU_PICKING_BUFFER_BYTES,
  SVO_GPU_PICKING_BYTES_PER_ROW,
  SVO_GPU_PICKING_OFFSETS,
  SVO_GPU_PICKING_READBACK_SLOTS,
  svoPickingPixelFromNormalized,
} from "../lib/webgpu-svo-picking-readback";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const viewportSource = readFileSync(new URL("../components/WebGPUViewport.tsx", import.meta.url), "utf8");
const targetsSource = readFileSync(new URL("../lib/webgpu-svo-gbuffer-targets.ts", import.meta.url), "utf8");
const drySource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const readbackSource = readFileSync(new URL("../lib/webgpu-svo-picking-readback.ts", import.meta.url), "utf8");
const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");

const hit: SvoGBufferHit = {
  status: "hit",
  radianceLinear: [0.2, 0.3, 0.4],
  depth_m: 2.5,
  geometricNormal: [0, 1, 0],
  shadingNormal: [0, 1, 0],
  materialId: 5,
  ownerId: 2,
  mediumBefore: 0,
  mediumAfter: 3,
  velocity_m_s: [0, 0, 0],
  motionKind: SVO_GBUFFER_MOTION_KINDS.rigid,
  motionValid: true,
  fieldSource: SVO_GBUFFER_FIELD_SOURCES.analyticPrimitive,
  localTopologyGeneration: 17,
  featureId: 0,
};

const request = {
  pixelX: 7, pixelY: 9,
  rayOrigin_m: [1, 2, 3] as const,
  rayDirection: [0, 0, -2] as const,
  rigidBodyCount: 4,
  materialCount: 12,
  frameToken: 8,
};

test("screen coordinates map exactly to the internal G-buffer without a Y flip", () => {
  assert.deepEqual(svoPickingPixelFromNormalized(0, 0, 100, 50), [0, 0]);
  assert.deepEqual(svoPickingPixelFromNormalized(1, 1, 100, 50), [99, 49]);
  assert.deepEqual(svoPickingPixelFromNormalized(0.505, 0.51, 100, 50), [50, 25]);
  assert.equal(svoPickingPixelFromNormalized(-0.01, 0.5, 100, 50), undefined);
});

test("compact G-buffer decode validates owner/material/generation and reconstructs the exact primary ray", () => {
  assert.match(SVO_GBUFFER_LAYOUT.radianceDepth.encoding, /linear metres along the normalized primary ray/);
  const result = decodeSvoGpuPickingSample(packSvoGBufferPixel(hit), request);
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  assert.equal(result.bodyIndex, 2);
  assert.equal(result.materialId, 5);
  assert.equal(result.localTopologyGeneration, 17);
  assert.equal(result.depth_m, 2.5);
  assert.deepEqual(result.position_m, [1, 2, 0.5]);
  assert.ok(Math.abs(result.geometricNormal[1] - 1) < 1e-6);
});

test("background, environment owner, malformed material, and unpublished generation fail closed", () => {
  const miss = packSvoGBufferPixel({ status: "miss", radianceLinear: [0, 0, 0] });
  assert.deepEqual(decodeSvoGpuPickingSample(miss, request), { status: "miss", reason: "background" });
  assert.deepEqual(decodeSvoGpuPickingSample(packSvoGBufferPixel({ ...hit, ownerId: 7 }), request), {
    status: "miss", reason: "non-interactive-owner",
  });
  assert.deepEqual(decodeSvoGpuPickingSample(packSvoGBufferPixel({ ...hit, materialId: 12 }), request), {
    status: "invalid", reason: "identity",
  });
  assert.deepEqual(decodeSvoGpuPickingSample(packSvoGBufferPixel({ ...hit, localTopologyGeneration: 0 }), request), {
    status: "invalid", reason: "generation",
  });
});

test("readback ABI is a bounded three-slot ring with aligned one-pixel rows", () => {
  assert.equal(SVO_GPU_PICKING_READBACK_SLOTS, 3);
  assert.equal(SVO_GPU_PICKING_BYTES_PER_ROW, 256);
  assert.equal(SVO_GPU_PICKING_BUFFER_BYTES, 768);
  assert.deepEqual(SVO_GPU_PICKING_OFFSETS, { radianceDepth: 0, packedSurface: 256, identityMedia: 512 });
  assert.match(targetsSource, /GPUTextureUsage\.COPY_SRC/);
  assert.match(waterSource, /label: "Dry scene HDR"[\s\S]*?GPUTextureUsage\.COPY_SRC/);
  assert.match(rendererSource, /label:"Water presentation target"[\s\S]*?GPUTextureUsage\.COPY_SRC/);
  assert.match(readbackSource, /Array\.from\(\{ length: SVO_GPU_PICKING_READBACK_SLOTS \}/,
    "readback buffers are allocated once in the ring constructor");
  assert.match(readbackSource, /this\.slots\.find\(\(candidate\) => !candidate\.pending\)/);
  assert.match(readbackSource, /if \(!slot\) return \{ status: "busy" \}/,
    "the click path must stay bounded instead of allocating or waiting for a slot");
  assert.match(readbackSource, /queue\.submit[\s\S]*?await slot\.buffer\.mapAsync/,
    "mapping is asynchronous and happens only after the one-pixel copy is submitted");
  assert.match(drySource, /SparseVoxelGpuPickingReadbackRing/);
  assert.match(drySource, /this\.pickingFrameToken === frameToken && this\.lastPickingTarget === radianceDepth/,
    "resize/source replacement must invalidate an in-flight readback");
});

test("default SVO picking uses the G-buffer while raster keeps the resident rigid fallback", () => {
  assert.match(rendererSource, /this\.svoPickingAvailable && screen && this\.svoDryScenePipeline/);
  assert.match(rendererSource, /pipeline\.pickGBuffer/);
  assert.match(rendererSource, /fluid\.pickRigidBody\(origin,direction\)/,
    "raster and unavailable-SVO paths retain the existing GPU rigid picker");
  assert.match(viewportSource, /normalizedX:\(event\.clientX-rect\.left\)\/Math\.max\(rect\.width,1\)/);
  assert.match(viewportSource, /normalizedY:\(event\.clientY-rect\.top\)\/Math\.max\(rect\.height,1\)/);
  assert.match(viewportSource, /surfacePosition = position/);
  assert.match(viewportSource, /planeHit\(ray\.origin, ray\.direction, surfacePosition, basis\.forward\)/,
    "the exact reconstructed surface point anchors the established camera-facing drag plane");
  assert.match(viewportSource, /beginBodyDrag\(pointerId,timeStamp,ray,body,picked\.position_m,picked\.orientation,"surfacePosition_m" in picked\?picked\.surfacePosition_m:picked\.position_m\)/,
    "SVO uses exact surface unprojection while the resident raster picker retains its center-position fallback");
  assert.match(viewportSource, /useUIStore\.getState\(\)\.selectBody\(body\.description\.id\)/);
  assert.match(viewportSource, /const pausedPresentation = runtime\.runState === "paused" \? \[[\s\S]*?sceneState, ui, method/,
    "selection changes the paused presentation key and schedules the material override repaint");
  assert.match(drySource, /selectedEmission=body\.colorSelected\.w\*vec3f\(\.12,\.42,\.32\)/,
    "the production SVO material path visibly consumes selected rigid state");
  assert.match(drySource, /if \(shape==0\)[\s\S]*?discriminant=b\*b-dot\(localOrigin,localOrigin\)\+radius\*radius/,
    "sphere identity comes from the analytic sphere intersection");
  assert.match(drySource, /else if \(shape==1\)[\s\S]*?slabHit\(localOrigin,localDirection,body\.halfSizeShape\.xyz\)/,
    "box identity comes from the exact oriented slab intersection");
  assert.match(drySource, /if\(shape==2\)\{for\(var side=-1\.0;side<=1\.0;side\+=2\.0\)[\s\S]*?let disc=hb\*hb-dot\(offset,offset\)\+radius\*radius/,
    "capsules close the shared cylindrical side with analytic hemispheres");
  assert.match(drySource, /else if\(abs\(localDirection\.y\)>1e-7\)[\s\S]*?dot\(p\.xz,p\.xz\)<=radius\*radius[\s\S]*?SVO_FEATURE_CYLINDER_CAP/,
    "capped cylinders close the shared cylindrical side with exact planar discs");
});
