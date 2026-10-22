import "../lib/core/empty-scene";
import assert from "node:assert/strict";
import test from "node:test";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS, packSvoMaterialTable, svoMaterialFromEnvironmentProxyMaterial } from "../lib/svo/contracts/svo-material-abi";
import { sculptMaterialTable } from "../lib/svo/features/scene-publication/svo-render-scene-publication";
import { SVO_SCULPT_MATERIAL_OFFSET } from "../lib/svo/features/scene-publication/svo-sculpt-cuts";

test("sculpted instances retain full PBR with independent native visibility identities", () => {
  const source = svoMaterialFromEnvironmentProxyMaterial(7, { colorLinear: [0.2, 0.4, 0.8], roughness: 0.3, emission: 0.2 }, 1);
  const base = packSvoMaterialTable([{ ...source, metallic: 0.7, flags: SVO_MATERIAL_FLAGS.opaque | SVO_MATERIAL_FLAGS.sampledVolume }]);
  const untouched = base.slice(), mirrored = sculptMaterialTable(base);
  const from = 7 * SVO_MATERIAL_RECORD_WORDS, to = (7 + SVO_SCULPT_MATERIAL_OFFSET) * SVO_MATERIAL_RECORD_WORDS;
  assert.deepEqual(base, untouched, "source palette stays immutable for untouched instances");
  assert.deepEqual(mirrored.subarray(from, from + 24), base.subarray(from, from + 24));
  assert.deepEqual(mirrored.subarray(to, to + 20), base.subarray(from, from + 20), "all PBR parameters survive");
  assert.equal(mirrored[to + 20], 7 + SVO_SCULPT_MATERIAL_OFFSET);
  assert.equal(mirrored[to + 23], SVO_MATERIAL_FLAGS.opaque);
  assert.equal(sculptMaterialTable(base), mirrored, "subsequent stroke samples reuse the mirror");
});
