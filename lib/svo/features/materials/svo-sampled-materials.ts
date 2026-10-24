import { SVO_SCULPT_MATERIAL_OFFSET } from "../scene-publication/svo-sculpt-cuts";
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

const sculptMaterialMirrors = new WeakMap<Uint32Array, Uint32Array<ArrayBuffer>>();
/** Both primary shading and radiance must resolve the voxelizer's mirrored IDs. */
export function writeSculptMaterialMirrors(table: Uint32Array, count: number): number {
  let end = count;
  for (let id = 1; id < count; id++) {
    const from = id * SVO_MATERIAL_RECORD_WORDS;
    if (!(table[from + 23]! & SVO_MATERIAL_FLAGS.sampledVolume)) continue;
    if (id >= SVO_SCULPT_MATERIAL_OFFSET) throw new RangeError("Scenery sculpt material palette capacity exceeded");
    const mirror = id + SVO_SCULPT_MATERIAL_OFFSET, to = mirror * SVO_MATERIAL_RECORD_WORDS;
    if (to + SVO_MATERIAL_RECORD_WORDS > table.length) throw new RangeError("Scenery sculpt material table is too small");
    table.set(table.subarray(from, from + SVO_MATERIAL_RECORD_WORDS), to);
    table[to + 20] = mirror;
    table[to + 23]! &= ~SVO_MATERIAL_FLAGS.sampledVolume;
    end = Math.max(end, mirror + 1);
  }
  return end;
}
/** Independent palette identities let an edited instance use the SVO mesh
 * while other instances of the same asset keep their original native mesh. */
export function sculptMaterialTable(base: Uint32Array<ArrayBuffer>): Uint32Array<ArrayBuffer> {
  const cached = sculptMaterialMirrors.get(base); if (cached) return cached;
  const count = base.length / SVO_MATERIAL_RECORD_WORDS;
  if (count > SVO_SCULPT_MATERIAL_OFFSET) throw new RangeError("Scenery sculpt material palette capacity exceeded");
  const table = new Uint32Array((SVO_SCULPT_MATERIAL_OFFSET + count) * SVO_MATERIAL_RECORD_WORDS);
  table.set(base);
  writeSculptMaterialMirrors(table, count);
  sculptMaterialMirrors.set(base, table); return table;
}
