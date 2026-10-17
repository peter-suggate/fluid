import { meshAssetHeader } from "../../../core/imported-assets/asset-library";
import type { SvoFieldProgram } from "../scene-publication/svo-field-program";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS, canonicalSvoMaterialRecord, svoMaterialFromEnvironmentProxyMaterial, type SvoMaterialRecord } from "../../contracts/svo-material-abi";

/** Baked texture palette entries are ordinary PBR records, shared by all
 * instances. The voxel selects a full material, including roughness/metallic. */
export function sampledVolumeMaterialRecords(programs: Iterable<SvoFieldProgram>, revision: number): SvoMaterialRecord[] {
  const records: SvoMaterialRecord[] = [], seen = new Set<number>();
  for (const program of programs) {
    const field = program.sampledVolume;
    if (!field) continue;
    const base = field.materialBase;
    if (!base) throw new Error("Sampled field has no published PBR palette");
    if (seen.has(base)) continue;
    seen.add(base);
    meshAssetHeader(field.asset).materials.forEach((m, i) => records.push(canonicalSvoMaterialRecord({
      ...svoMaterialFromEnvironmentProxyMaterial(base + i, { colorLinear: m.baseColorLinear, emission: 0, roughness: m.roughness }, revision),
      metallic: m.metallic, flags: SVO_MATERIAL_FLAGS.opaque | SVO_MATERIAL_FLAGS.sampledVolume,
    })));
  }
  return records;
}

/** Mark fallback owner colours as well: a coarse lighting cell can miss every
 * fine material node, but must still yield primary visibility to its source. */
export function markSampledVolumeOwnerMaterials(table: Uint32Array,
  primitives: readonly { kind: string; materialId: number; program?: SvoFieldProgram }[]): Uint32Array<ArrayBuffer> {
  for (const p of primitives) if (p.kind === "field-program" && p.program?.sampledVolume) {
    table[p.materialId * SVO_MATERIAL_RECORD_WORDS + 23]! |= SVO_MATERIAL_FLAGS.sampledVolume;
  }
  return table as Uint32Array<ArrayBuffer>;
}
