import {
  SVO_CLUSTER_FIELD_TABLE,
  SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT,
  SVO_CLUSTER_LOBE_DEFAULT_SPAN,
  SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD,
  SVO_CLUSTER_SWEEP_MAXIMUM_POINTS,
  SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,
  svoClusterFieldName,
  type SvoSmoothUnionClusterPacking,
} from "./svo-primitive-abi";

/**
 * The aggregate-cluster parameter block, packed for a consumer that is not the
 * dry-scene arena.
 *
 * The renderer publishes cluster packings inside its one big scene arena
 * (`packSvoDrySceneClusters`, `lib/webgpu-svo-dry-scene.ts:597`). The live
 * voxelizer needs the same blocks against a different buffer, at a different
 * base, on a different revision cadence — it is fed by keyed live primitive
 * updates, not by a whole-scene publication — so it packs its own arena rather
 * than reaching into that one.
 *
 * Two transcriptions of one layout is exactly the failure this ABI keeps
 * warning about, so the agreement is *executed* rather than asserted in a
 * comment: `tools/run-svo-live-voxelization-smoke.ts` packs the same fixtures
 * through both packers and compares the bytes. If the dry scene's layout moves,
 * that lane goes red before a frame is drawn.
 */
export const SVO_CLUSTER_ARENA_BLOCK = Object.freeze({
  fieldWord: 0,
  seedWord: 1,
  countWord: 2,
  smoothRadiusWord: 3,
  latticeLobeRadiusWord: 4,
  latticePeriodWord: 5,
  jitterWord: 6,
  anisotropyWord: 7,
  lobeSpanWord: 8,
  lobeSpanSpreadWord: 9,
  displacementWord: 10,
  /** Eight `vec4f`, sixteen-word aligned so the shader reads the polyline as vectors. */
  pointsWord: 16,
  strideWords: SVO_SMOOTH_UNION_CLUSTER_ARENA_WORDS,
} as const);

/** Pack `packings` into a dense arena; block `index` starts at `index * strideWords`. */
export function packSvoClusterArena(
  packings: readonly SvoSmoothUnionClusterPacking[],
  capacity = packings.length,
): Uint32Array<ArrayBuffer> {
  if (!Number.isSafeInteger(capacity) || capacity < packings.length || capacity < 0) {
    throw new RangeError("Cluster arena capacity must hold every packing");
  }
  const block = SVO_CLUSTER_ARENA_BLOCK;
  const buffer = new ArrayBuffer(Math.max(1, capacity) * block.strideWords * 4);
  const words = new Uint32Array(buffer);
  const floats = new Float32Array(buffer);
  packings.forEach((packing, index) => {
    const base = index * block.strideWords;
    words[base + block.fieldWord] = SVO_CLUSTER_FIELD_TABLE[svoClusterFieldName(packing)].code;
    words[base + block.seedWord] = packing.seed >>> 0;
    floats[base + block.smoothRadiusWord] = packing.smoothRadius_m;
    if (packing.field === "seeded-lobes") {
      words[base + block.countWord] = packing.lobeCount >>> 0;
      floats[base + block.anisotropyWord] = packing.anisotropy;
      // Defaulted by the packer, never by the shader: the block is the ABI.
      floats[base + block.lobeSpanWord] = packing.lobeSpan ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN;
      floats[base + block.lobeSpanSpreadWord] = packing.lobeSpanSpread ?? SVO_CLUSTER_LOBE_DEFAULT_SPAN_SPREAD;
      floats[base + block.displacementWord] = packing.displacement ?? SVO_CLUSTER_LOBE_DEFAULT_DISPLACEMENT;
      return;
    }
    if (packing.field === "tapered-sweep") {
      words[base + block.countWord] = packing.points.length >>> 0;
      packing.points.forEach((point, pointIndex) => {
        floats.set([point.position_m.x, point.position_m.y, point.position_m.z, point.radius_m],
          base + block.pointsWord + pointIndex * 4);
      });
      return;
    }
    words[base + block.countWord] = (packing.octaves ?? 1) >>> 0;
    floats[base + block.latticeLobeRadiusWord] = packing.latticeLobeRadius_m;
    floats[base + block.latticePeriodWord] = packing.latticePeriod_m;
    floats[base + block.jitterWord] = packing.jitter;
  });
  return words;
}

export interface SvoClusterArenaDecodeOptions {
  /** Name of the generated `fn(blockIndex:u32)->SvoClusterPacking`. */
  functionName: string;
  /** Name of an existing `fn(word:u32)->u32` over whatever buffer holds the arena. */
  loadWord: string;
  /** WGSL expression for the arena's first word. */
  baseWordExpression: string;
  /** WGSL expression for the number of blocks the arena holds. */
  capacityExpression: string;
}

/**
 * WGSL decode of one block into the shared ABI's `SvoClusterPacking`.
 *
 * Reads through a caller-supplied word loader rather than indexing a binding
 * directly, because the live voxelizer keeps its arena inside the same fixed
 * maintenance buffer as its dirty regions, brick list and candidate bins — that
 * pass holds itself to four storage bindings, and one more buffer for a table
 * that is usually a few kilobytes would spend a scarce binding on nothing.
 *
 * `blockIndex` is a slot, not a word offset, because the arena is dense and
 * owned by one pass. A slot past the capacity decodes as the invalid packing,
 * which the ABI evaluates as a miss rather than as the envelope — the same
 * refusal the renderer makes, and for the same reason: an aggregate that
 * quietly became an ellipsoid is a shape nobody authored.
 */
export function svoClusterArenaDecodeWGSL(options: SvoClusterArenaDecodeOptions): string {
  const block = SVO_CLUSTER_ARENA_BLOCK;
  const { functionName, loadWord, baseWordExpression, capacityExpression } = options;
  const words4 = (offset: string) =>
    `vec4u(${loadWord}(${offset}),${loadWord}(${offset}+1u),${loadWord}(${offset}+2u),${loadWord}(${offset}+3u))`;
  return /* wgsl */ `
fn ${functionName}(blockIndex:u32)->SvoClusterPacking{
  if(blockIndex>=(${capacityExpression})){return svoInvalidClusterPacking();}
  let base=(${baseWordExpression})+blockIndex*${block.strideWords}u;
  let head=${words4("base")};
  let shape=bitcast<vec4f>(${words4(`base+${block.latticeLobeRadiusWord}u`)});
  let placement=bitcast<vec4f>(${words4(`base+${block.lobeSpanWord}u`)});
  // The polyline is copied whatever the field is: a uniform read costs the same
  // for every aggregate, and branching on the field here would diverge the wave
  // for the one arm that needs it.
  var points=array<vec4f,${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}>();
  for(var index=0u;index<${SVO_CLUSTER_SWEEP_MAXIMUM_POINTS}u;index+=1u){
    points[index]=bitcast<vec4f>(${words4(`base+${block.pointsWord}u+index*4u`)});
  }
  return SvoClusterPacking(head[${block.fieldWord}],head[${block.seedWord}],head[${block.countWord}],
    bitcast<f32>(head[${block.smoothRadiusWord}]),shape.x,shape.y,shape.z,shape.w,
    placement.x,placement.y,placement.z,points);
}
`;
}
