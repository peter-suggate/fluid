import { svoPrimitiveLocalExtent_m, type SvoPrimitiveDescriptor } from "../../contracts/svo-primitive-abi";
import { intersectsSvoSculptPrimitiveBounds, SVO_SCULPT_CUT_HEADER_WORDS, SVO_SCULPT_CUT_RECORD_WORDS } from "./svo-sculpt-cuts";

export interface SvoSculptInstanceClassification {
  readonly primitives: readonly SvoPrimitiveDescriptor[];
  readonly cuts: Uint32Array;
  readonly indices: readonly number[];
}

/** Appended pen samples cannot restore an affected instance. Test only the new
 * cuts against the remaining instances; undo and changed scenery recompute. */
export function classifySvoSculptInstances(primitives: readonly SvoPrimitiveDescriptor[], cuts: Uint32Array,
  previous?: SvoSculptInstanceClassification): readonly number[] {
  if (previous?.primitives === primitives && previous.cuts === cuts) return previous.indices;
  const count = cuts[0] ?? 0;
  if (count === 0) return [];
  let retained: ReadonlySet<number> | undefined;
  let testedCuts = cuts;
  if (previous?.primitives === primitives && count >= (previous.cuts[0] ?? 0)) {
    const oldCount = previous.cuts[0] ?? 0;
    const oldEnd = SVO_SCULPT_CUT_HEADER_WORDS + oldCount * SVO_SCULPT_CUT_RECORD_WORDS;
    let prefix = true;
    for (let word = SVO_SCULPT_CUT_HEADER_WORDS; word < oldEnd; word++) {
      if (cuts[word] !== previous.cuts[word]) { prefix = false; break; }
    }
    if (prefix) {
      if (count === oldCount) return previous.indices;
      retained = new Set(previous.indices);
      testedCuts = new Uint32Array(SVO_SCULPT_CUT_HEADER_WORDS + (count - oldCount) * SVO_SCULPT_CUT_RECORD_WORDS);
      testedCuts[0] = count - oldCount;
      testedCuts.set(cuts.subarray(oldEnd), SVO_SCULPT_CUT_HEADER_WORDS);
    }
  }
  const indices: number[] = [];
  primitives.forEach((p, index) => {
    if (retained?.has(index)) { indices.push(index); return; }
    if (p.kind !== "field-program" || !p.program?.sampledVolume) return;
    const extent = svoPrimitiveLocalExtent_m(p), center = p.center_m, q = p.orientation;
    if (intersectsSvoSculptPrimitiveBounds(testedCuts, [center.x, center.y, center.z], [extent.x, extent.y, extent.z],
      q ? [q.x, q.y, q.z, q.w] : undefined)) indices.push(index);
  });
  return indices;
}
