import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDefaultSvoMaterialRecords,
  canonicalSvoMaterialRecord,
  packSvoMaterialTable,
  SVO_MATERIAL_FLAGS,
  SVO_MATERIAL_FUNCTION_IDS,
  SVO_MATERIAL_RECORD_STRIDE_BYTES,
  svoMaterialWGSL,
  unpackSvoMaterialRecord,
} from "../lib/svo-material-abi";
import { WATER_OPTICS } from "../lib/webgpu-lighting";
import { VOXEL_MATERIAL_IDS, VOXEL_MATERIALS } from "../lib/voxel-scene";

test("96-byte material ABI preserves PBR, optical, procedural, and identity lanes", () => {
  assert.equal(SVO_MATERIAL_RECORD_STRIDE_BYTES, 96);
  const records = buildDefaultSvoMaterialRecords(7);
  const packed = packSvoMaterialTable(records);
  assert.equal(packed.byteLength, (Math.max(...VOXEL_MATERIALS.map((entry) => entry.id)) + 1) * 96);
  for (const expected of records) {
    const actual = unpackSvoMaterialRecord(packed, expected.materialId);
    assert.equal(actual.materialId, expected.materialId);
    assert.equal(actual.revision, 7);
    assert.equal(actual.flags, expected.flags);
    assert.equal(actual.materialFunctionId, expected.materialFunctionId);
    assert.deepEqual(actual.baseColorLinear.map((value) => Number(value.toFixed(5))), expected.baseColorLinear.map((value) => Number(value.toFixed(5))));
    assert.ok(Math.abs(actual.roughness - expected.roughness) < 1e-6);
    assert.ok(Math.abs(actual.indexOfRefraction - expected.indexOfRefraction) < 1e-6);
  }
});

test("water and glass keep distinct bounded dielectric semantics", () => {
  const records = buildDefaultSvoMaterialRecords();
  const water = records.find((entry) => entry.materialId === VOXEL_MATERIAL_IDS.fluid)!;
  const glass = records.find((entry) => entry.materialId === VOXEL_MATERIAL_IDS.containerGlass)!;
  assert.equal(water.flags, SVO_MATERIAL_FLAGS.dielectric);
  assert.deepEqual(water.absorption_mInv, WATER_OPTICS.absorption);
  assert.ok(water.scattering_mInv > 0);
  assert.equal(glass.flags, SVO_MATERIAL_FLAGS.dielectric | SVO_MATERIAL_FLAGS.thinWall);
  assert.equal(glass.opacity, 0.24);
  assert.deepEqual(glass.absorption_mInv, [0, 0, 0]);
});

test("terrain carries a stable material function while rigid materials remain plain opaque PBR", () => {
  const records = buildDefaultSvoMaterialRecords();
  const terrain = records.find((entry) => entry.materialId === VOXEL_MATERIAL_IDS.terrain)!;
  const sphere = records.find((entry) => entry.materialId === VOXEL_MATERIAL_IDS.sphere)!;
  assert.equal(terrain.materialFunctionId, SVO_MATERIAL_FUNCTION_IDS.gardenTerrain);
  assert.equal(terrain.flags, SVO_MATERIAL_FLAGS.opaque);
  assert.equal(sphere.materialFunctionId, SVO_MATERIAL_FUNCTION_IDS.none);
  assert.equal(sphere.flags, SVO_MATERIAL_FLAGS.opaque);
});

test("invalid values and duplicate/reserved identities fail before upload", () => {
  const valid = buildDefaultSvoMaterialRecords()[0];
  assert.throws(() => canonicalSvoMaterialRecord({ ...valid, roughness: Number.NaN }), /roughness/);
  assert.throws(() => canonicalSvoMaterialRecord({ ...valid, indexOfRefraction: 0.9 }), /IOR/);
  assert.throws(() => canonicalSvoMaterialRecord({ ...valid, scatteringAnisotropy: 1 }), /anisotropy/);
  assert.throws(() => packSvoMaterialTable([{ ...valid, materialId: 0 }]), /slot zero/);
  assert.throws(() => packSvoMaterialTable([valid, valid]), /Duplicate/);
});

test("WGSL record stays binding-free and exposes direct-index validation and dielectric F0", () => {
  assert.match(svoMaterialWGSL, /struct SvoMaterialRecord/);
  assert.match(svoMaterialWGSL, /baseColorOpacity:vec4f/);
  assert.match(svoMaterialWGSL, /identity:vec4u/);
  assert.match(svoMaterialWGSL, /fn svoMaterialValid/);
  assert.match(svoMaterialWGSL, /fn svoMaterialDielectricF0/);
  assert.doesNotMatch(svoMaterialWGSL, /@group|@binding|while\s*\(|loop\s*\{/);
});
