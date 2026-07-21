import assert from "node:assert/strict";
import test from "node:test";
import {
  SVO_GBUFFER_BYTES_PER_PIXEL,
  SVO_GBUFFER_COLOR_ATTACHMENT_COUNT,
  SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE,
  SVO_GBUFFER_DEBUG_SIDECAR_BYTES_PER_PIXEL,
  SVO_GBUFFER_FAILURES,
  SVO_GBUFFER_FEATURES,
  SVO_GBUFFER_FIELD_SOURCES,
  SVO_GBUFFER_FLAGS,
  SVO_GBUFFER_LAYOUT,
  SVO_GBUFFER_MOTION_KINDS,
  SVO_GBUFFER_PRECISION,
  decodeSvoGBufferFloat16,
  encodeSvoGBufferFloat16,
  makeSvoGBufferTemporalKey,
  packSvoGBufferPixel,
  reconstructSvoGBufferWorldPosition,
  svoGBufferHardBoxFeatureNormal,
  svoGBufferWGSL,
  unpackSvoGBufferPixel,
  type SvoGBufferHit,
} from "../lib/svo-gbuffer";
import { packSvoTemporalHitKey } from "../lib/svo-temporal-history";

function close(actual: number, expected: number, tolerance: number) {
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} should be within ${tolerance} of ${expected}`);
}

function closeVector(actual: readonly number[], expected: readonly number[], tolerance: number) {
  actual.forEach((value, index) => close(value, expected[index], tolerance));
}

function angularErrorDegrees(actual: readonly number[], expected: readonly number[]) {
  const denominator = Math.hypot(...actual) * Math.hypot(...expected);
  const cosine = actual.reduce((sum, value, index) => sum + value * expected[index], 0) / denominator;
  return Math.acos(Math.max(-1, Math.min(1, cosine))) * 180 / Math.PI;
}

const hit: SvoGBufferHit = {
  status: "hit",
  radianceLinear: [12.5, 0.125, 1024.25],
  depth_m: 123.456789,
  geometricNormal: [0.25, -0.5, -0.8291561976],
  shadingNormal: [-0.7, 0.2, 0.68556546],
  materialId: 0xabcd,
  ownerId: 0xfffe,
  mediumBefore: 12,
  mediumAfter: 0xab,
  velocity_m_s: [1.25, -0.03125, 27.7],
  motionKind: SVO_GBUFFER_MOTION_KINDS.rigid,
  motionValid: true,
  fieldSource: SVO_GBUFFER_FIELD_SOURCES.analyticPrimitive,
  localTopologyGeneration: 0xffff_fffe,
  featureId: SVO_GBUFFER_FEATURES.smooth,
};

test("core production targets fit the baseline 32-byte WebGPU color-attachment limit", () => {
  assert.equal(SVO_GBUFFER_COLOR_ATTACHMENT_COUNT, 3);
  assert.equal(SVO_GBUFFER_COLOR_BYTES_PER_SAMPLE, 32);
  assert.equal(SVO_GBUFFER_BYTES_PER_PIXEL, 36, "hardware depth is separate from the 32-byte MRT budget");
  assert.equal(SVO_GBUFFER_DEBUG_SIDECAR_BYTES_PER_PIXEL, 16);
  assert.deepEqual([
    SVO_GBUFFER_LAYOUT.radianceDepth.format,
    SVO_GBUFFER_LAYOUT.packedSurface.format,
    SVO_GBUFFER_LAYOUT.identityMedia.format,
    SVO_GBUFFER_LAYOUT.hardwareDepth.format,
  ], ["rgba16float", "rgba32uint", "rgba16uint", "depth32float"]);
  assert.deepEqual([
    SVO_GBUFFER_LAYOUT.radianceDepth.location,
    SVO_GBUFFER_LAYOUT.packedSurface.location,
    SVO_GBUFFER_LAYOUT.identityMedia.location,
  ], [0, 1, 2]);
  assert.match(SVO_GBUFFER_LAYOUT.debugSidecar.encoding, /diagnostic-only passes/);
});

test("binary16 CPU mirror is round-to-nearest-even and declares depth precision", () => {
  for (const value of [0, -0, 2 ** -24, 2 ** -14, 1, -2.5, 31.337, 65_504, 1e9]) {
    const decoded = decodeSvoGBufferFloat16(encodeSvoGBufferFloat16(value));
    const expected = Math.max(-65_504, Math.min(65_504, value));
    const tolerance = Math.max(SVO_GBUFFER_PRECISION.halfMinimumSubnormal, Math.abs(expected) * SVO_GBUFFER_PRECISION.halfRelativeTolerance);
    close(decoded, expected, tolerance);
  }
  assert.equal(encodeSvoGBufferFloat16(1 + 2 ** -11), 0x3c00);
  assert.equal(SVO_GBUFFER_PRECISION.maximumLinearDepth_m, 65_504);
});

test("core packing preserves HDR, depth, two normals, exact identity/generation, bounded motion, and source flags", () => {
  const packed = packSvoGBufferPixel(hit);
  assert.deepEqual({
    radianceDepth: packed.radianceDepth.byteLength,
    packedSurface: packed.packedSurface.byteLength,
    identityMedia: packed.identityMedia.byteLength,
    debugSidecar: packed.debugSidecar.byteLength,
  }, { radianceDepth: 8, packedSurface: 16, identityMedia: 8, debugSidecar: 16 });
  const decoded = unpackSvoGBufferPixel(packed);
  assert.equal(decoded.status, "hit");
  closeVector(decoded.radianceLinear, hit.radianceLinear, 1.1);
  close(decoded.depth_m, hit.depth_m, hit.depth_m * SVO_GBUFFER_PRECISION.halfRelativeTolerance);
  assert.ok(angularErrorDegrees(decoded.geometricNormal, hit.geometricNormal) <= SVO_GBUFFER_PRECISION.maximumNormalAngularError_deg);
  assert.ok(angularErrorDegrees(decoded.shadingNormal, hit.shadingNormal) <= SVO_GBUFFER_PRECISION.maximumNormalAngularError_deg);
  closeVector(decoded.velocity_m_s, hit.velocity_m_s, SVO_GBUFFER_PRECISION.maximumVelocityAbsoluteError_m_s + 1e-12);
  assert.deepEqual({
    materialId: decoded.materialId, ownerId: decoded.ownerId,
    mediumBefore: decoded.mediumBefore, mediumAfter: decoded.mediumAfter,
    fieldSource: decoded.fieldSource, generation: decoded.localTopologyGeneration,
    featureId: decoded.featureId, motionKind: decoded.motionKind,
  }, {
    materialId: hit.materialId, ownerId: hit.ownerId,
    mediumBefore: hit.mediumBefore, mediumAfter: hit.mediumAfter,
    fieldSource: hit.fieldSource, generation: hit.localTopologyGeneration,
    featureId: hit.featureId, motionKind: hit.motionKind,
  });
  assert.deepEqual(Array.from(packed.debugSidecar), [0, 0, 0, 0]);
});

test("velocity saturates to the declared range and preserves motion kind", () => {
  const decoded = unpackSvoGBufferPixel(packSvoGBufferPixel({
    ...hit, velocity_m_s: [100, -100, 0] as const, motionKind: SVO_GBUFFER_MOTION_KINDS.rigid,
  }));
  assert.equal(decoded.status, "hit");
  assert.deepEqual(decoded.velocity_m_s, [64, -64, 0]);
  assert.equal(decoded.motionKind, SVO_GBUFFER_MOTION_KINDS.rigid);
});

test("miss encoding uses zero distance and validity while retaining failure identity", () => {
  const packed = packSvoGBufferPixel({
    status: "miss", radianceLinear: [0.1, 0.25, 2],
    fieldSource: SVO_GBUFFER_FIELD_SOURCES.structuralDiscrete,
    localTopologyGeneration: 19,
    failure: SVO_GBUFFER_FAILURES.staleGeneration,
  });
  assert.equal(packed.radianceDepth[3], 0);
  assert.equal(packed.packedSurface[0], 0);
  assert.deepEqual(Array.from(packed.identityMedia), [0, 0, 0, 0]);
  const decoded = unpackSvoGBufferPixel(packed);
  assert.equal(decoded.status, "miss");
  assert.equal(decoded.failure, SVO_GBUFFER_FAILURES.staleGeneration);
  assert.equal(decoded.localTopologyGeneration, 19);
  assert.ok((decoded.additionalFlags! & SVO_GBUFFER_FLAGS.validSurface) === 0);
  assert.ok((decoded.additionalFlags! & SVO_GBUFFER_FLAGS.miss) !== 0);
  assert.ok((decoded.additionalFlags! & SVO_GBUFFER_FLAGS.staleGeneration) !== 0);
  assert.deepEqual(Array.from(packed.debugSidecar), [0, 0, 0, 0]);
});

test("linear ray distance reconstructs world position independently of direction magnitude", () => {
  closeVector(reconstructSvoGBufferWorldPosition([1, 2, 3], [0, 20, 0], 2), [1, 4, 3], 1e-12);
  assert.throws(() => reconstructSvoGBufferWorldPosition([0, 0, 0], [0, 0, 0], 1), /nonzero/);
});

test("hard boxes select one stable authored feature at faces, edges, and corners", () => {
  assert.deepEqual(svoGBufferHardBoxFeatureNormal([1, 1, 1], [1, 1, 1]), {
    normal: [1, 0, 0], featureId: SVO_GBUFFER_FEATURES.boxFaceX,
  });
  assert.deepEqual(svoGBufferHardBoxFeatureNormal([-1, -1, 0], [1, 1, 1]), {
    normal: [-1, 0, 0], featureId: SVO_GBUFFER_FEATURES.boxFaceX,
  });
  assert.deepEqual(svoGBufferHardBoxFeatureNormal([0, -1, 0], [1, 1, 1]), {
    normal: [0, -1, 0], featureId: SVO_GBUFFER_FEATURES.boxFaceY,
  });
});

test("temporal construction remains the exact existing six-word 24-byte ABI after core quantization", () => {
  const quantized = unpackSvoGBufferPixel(packSvoGBufferPixel(hit));
  assert.equal(quantized.status, "hit");
  const expected = packSvoTemporalHitKey({
    depth_m: quantized.depth_m, geometricNormal: quantized.geometricNormal, shadingNormal: quantized.shadingNormal,
    materialId: quantized.materialId, ownerId: quantized.ownerId,
    mediumBefore: quantized.mediumBefore, mediumAfter: quantized.mediumAfter,
    localTopologyGeneration: quantized.localTopologyGeneration,
  });
  const key = makeSvoGBufferTemporalKey(hit);
  assert.equal(key.byteLength, 24);
  assert.deepEqual(key, expected);
});

test("invalid core values and attachment shapes fail at the ABI boundary", () => {
  assert.throws(() => packSvoGBufferPixel({ ...hit, depth_m: 65_505 }), /depth/);
  assert.throws(() => packSvoGBufferPixel({ ...hit, materialId: 0x1_0000 }), /Material ID/);
  assert.throws(() => packSvoGBufferPixel({ ...hit, geometricNormal: [0, 0, 0] as const }), /nonzero/);
  const malformed = packSvoGBufferPixel(hit);
  assert.throws(() => unpackSvoGBufferPixel({ ...malformed, debugSidecar: new Uint32Array(3) }), /attachment lengths/);
});

test("WGSL mirror is binding-free, three-target, temporally compatible, and preserves hard box ties", () => {
  assert.doesNotMatch(svoGBufferWGSL, /@group|@binding/);
  for (let location = 0; location < 3; location += 1) assert.match(svoGBufferWGSL, new RegExp(`@location\\(${location}\\)`));
  assert.doesNotMatch(svoGBufferWGSL, /@location\([3-9]/);
  assert.match(svoGBufferWGSL, /struct SvoGBufferTemporalKey\{[\s\S]*depth_m:f32[\s\S]*localTopologyGeneration:u32/);
  assert.match(svoGBufferWGSL, /origin_m\+normalize\(rayDirection\)\*linearDepth_m/);
  assert.match(svoGBufferWGSL, /if\(q\.y>q\.x\)\{axis=1u;\}if\(q\.z>q\[axis\]\)\{axis=2u;\}/);
  assert.match(svoGBufferWGSL, /SVO_GBUFFER_FEATURE_BOX_X\+axis/);
  assert.match(svoGBufferWGSL, /targets\.identityMedia\.y<<16u/);
});
