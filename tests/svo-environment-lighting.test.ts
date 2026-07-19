import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds, environmentIndex } from "../lib/environments";
import {
  buildAllSvoEnvironmentLightingRecords,
  buildSvoEnvironmentLighting,
  canonicalSvoEnvironmentLightingRecord,
  evaluateSvoEnvironmentDiffuseIrradiance,
  evaluateSvoEnvironmentPrefilteredSpecular,
  packSvoEnvironmentLightingRecords,
  SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES,
  svoEnvironmentLightingRecord,
  svoEnvironmentLightingWGSL,
  unpackSvoEnvironmentLightingRecord,
} from "../lib/svo-environment-lighting";
import { environmentShaderLibrary } from "../lib/webgpu-environments";

test("compact ABI covers every EnvironmentId in canonical direct order", () => {
  assert.equal(SVO_ENVIRONMENT_LIGHTING_RECORD_STRIDE_BYTES, 96);
  const records = buildAllSvoEnvironmentLightingRecords(7);
  assert.deepEqual(records.map(({ environmentId }) => environmentId), environmentIds);
  const packed = packSvoEnvironmentLightingRecords(records);
  assert.equal(packed.byteLength, environmentIds.length * 96);
  records.forEach((record, index) => {
    const unpacked = unpackSvoEnvironmentLightingRecord(packed, index);
    assert.equal(unpacked.environmentId, record.environmentId);
    assert.equal(unpacked.revision, 7);
    assert.equal(environmentIndex(unpacked.environmentId), index);
    assert.ok(Math.abs(Math.hypot(...unpacked.keyLightDirection) - 1) < 1e-6);
  });
});

test("fallback palettes retain the existing raster environment values", () => {
  const conservatory = svoEnvironmentLightingRecord("conservatory");
  assert.deepEqual(conservatory.lowerRadianceLinear, [0.035, 0.055, 0.044]);
  assert.deepEqual(conservatory.upperRadianceLinear, [0.46, 0.58, 0.43]);
  assert.deepEqual(conservatory.accentRadianceLinear, [0.24, 0.55, 0.39]);
  const lab = svoEnvironmentLightingRecord("night-lab");
  assert.deepEqual(lab.lowerRadianceLinear, [0.016, 0.017, 0.020]);
  assert.deepEqual(lab.keyLightColorLinear, [1, 0.94, 0.80]);
  const garden = svoEnvironmentLightingRecord("garden");
  assert.deepEqual(garden.upperRadianceLinear, [0.52, 0.60, 0.72]);
  assert.deepEqual(garden.keyLightDirection, canonicalSvoEnvironmentLightingRecord(garden).keyLightDirection);
  assert.match(environmentShaderLibrary, /mix\(vec3f\(\.035,\.055,\.044\),vec3f\(\.46,\.58,\.43\),t\)/);
  assert.match(environmentShaderLibrary, /vec3f\(1\.0,\.94,\.80\)/);
  assert.match(environmentShaderLibrary, /vec3f\(\.52,\.60,\.72\)/);
});

test("diffuse irradiance remains finite linear HDR and distinguishes authored looks", () => {
  const normal = [0.2, 0.96, -0.1] as const;
  const lab = evaluateSvoEnvironmentDiffuseIrradiance(svoEnvironmentLightingRecord("night-lab"), normal);
  const garden = evaluateSvoEnvironmentDiffuseIrradiance(svoEnvironmentLightingRecord("garden"), normal);
  const conservatory = evaluateSvoEnvironmentDiffuseIrradiance(svoEnvironmentLightingRecord("conservatory"), normal);
  for (const color of [lab, garden, conservatory]) assert.ok(color.every((channel) => Number.isFinite(channel) && channel >= 0));
  assert.notDeepEqual(lab, garden);
  assert.notDeepEqual(garden, conservatory);
  assert.ok(garden[2] > lab[2]);
  assert.ok(conservatory[1] > lab[1]);
});

test("prefiltered specular broadens with roughness while preserving HDR sun response", () => {
  const record = svoEnvironmentLightingRecord("garden");
  const mirror = evaluateSvoEnvironmentPrefilteredSpecular(record, record.keyLightDirection, 0);
  const rough = evaluateSvoEnvironmentPrefilteredSpecular(record, record.keyLightDirection, 1);
  assert.ok(Math.max(...mirror) > 1, "authored sun remains linear HDR");
  assert.ok(Math.max(...rough) < Math.max(...mirror), "energy-normalized prefilter lowers the lobe peak");
  assert.ok(rough.every((channel) => channel >= 0 && Number.isFinite(channel)));
  assert.throws(() => evaluateSvoEnvironmentPrefilteredSpecular(record, [0, 1, 0], 1.1), /zero to one/);
});

test("revision and cache identity are deterministic and content-sensitive", () => {
  const first = buildSvoEnvironmentLighting("night-lab", 3);
  const second = buildSvoEnvironmentLighting("night-lab", 3);
  assert.equal(second.staticRevision, first.staticRevision);
  assert.equal(second.cacheKey, first.cacheKey);
  assert.deepEqual(second.packedRecord, first.packedRecord);
  assert.notEqual(buildSvoEnvironmentLighting("night-lab", 4).cacheKey, first.cacheKey);
  assert.notEqual(buildSvoEnvironmentLighting("garden", 3).cacheKey, first.cacheKey);
  assert.match(first.cacheKey, /^svo-environment-lighting-v1:night-lab:[0-9a-f]{8}$/);
});

test("invalid records and duplicate coverage fail before publication", () => {
  const record = svoEnvironmentLightingRecord("default");
  assert.throws(() => canonicalSvoEnvironmentLightingRecord({ ...record, revision: 0 }), /positive uint32/);
  assert.throws(() => canonicalSvoEnvironmentLightingRecord({ ...record, keyLightDirection: [0, 0, 0] }), /non-zero/);
  assert.throws(() => packSvoEnvironmentLightingRecords([record, record]), /Duplicate/);
  assert.throws(() => unpackSvoEnvironmentLightingRecord(new Uint32Array(0)), /exceeds/);
});

test("WGSL evaluation is binding-free and parameterizes direction and roughness", () => {
  assert.match(svoEnvironmentLightingWGSL, /struct SvoEnvironmentLightingRecord/);
  assert.match(svoEnvironmentLightingWGSL, /fn svoEnvironmentDiffuseIrradiance/);
  assert.match(svoEnvironmentLightingWGSL, /fn svoEnvironmentPrefilteredSpecular/);
  assert.match(svoEnvironmentLightingWGSL, /roughnessIn:f32/);
  assert.match(svoEnvironmentLightingWGSL, /worldDirectionIn:vec3f/);
  assert.doesNotMatch(svoEnvironmentLightingWGSL, /@group|@binding|texture_|sampler/);
});
