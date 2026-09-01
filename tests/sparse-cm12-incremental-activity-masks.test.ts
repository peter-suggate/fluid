import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import {
  createSparseCM12IncrementalActivityInitialWords,
  createSparseCM12IncrementalActivityLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";
import { createSparseCM12IncrementalActivityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity.wgsl";

test("ACT1 uses stamp masks without allocating a brick list", () => {
  const layout = createSparseCM12IncrementalActivityLayout({
    baseWords: 128,
    brickCount: 8,
  });
  assert.equal(layout.headerBaseWords, 128);
  assert.equal(layout.brickStampBaseWords, 144);
  assert.equal(layout.brickVelocityStampBaseWords, 152);
  assert.equal(layout.brickTopologyStateBaseWords, 160);
  assert.equal(layout.brickCensusStateBaseWords, 168);
  assert.equal(layout.scoreHistogramBaseWords, 176);
  assert.equal(layout.brickBoundaryLiquidFaceBaseWords, 432);
  assert.equal(layout.totalWords, 440);
  assert.equal("brickListBaseWords" in layout, false);

  const words = createSparseCM12IncrementalActivityInitialWords(layout);
  assert.equal(words.length, layout.totalWords - layout.headerBaseWords);
  assert.deepEqual([...words.subarray(32, 40)], Array(8).fill(0xffff_ffff));
});

test("ACT1 claims and consumes brick stamps directly", () => {
  const layout = createSparseCM12IncrementalActivityLayout({
    baseWords: 0,
    brickCount: 32,
  });
  const wgsl = createSparseCM12IncrementalActivityWGSL(layout, 2);
  assert.match(wgsl, /atomicExchange\(&activity\[ACTIVITY_BRICK_STAMP\+brick\],generation\)/);
  assert.match(wgsl,
    /atomicExchange\(&activity\[ACTIVITY_BRICK_VELOCITY_STAMP\+brick\],\s*generation\)/);
  assert.match(wgsl,
    /if\(previous!=generation\)\{incrementalActivityPublishFaceBrickClosure\(brick\);\}/);
  assert.match(wgsl,
    /if\(dx>=0&&dx<span&&dy>=0&&dy<span&&dz>=0&&dz<span\)\{continue;\}/);
  assert.doesNotMatch(wgsl,
    /generation\);\s*_\s*=incrementalActivityClaimBrick\(brick\);\s*\/\/ Closure depends/);
  assert.match(wgsl, /fn finalizeIncrementalActivityMasks\(\)/);
  assert.match(wgsl, /return select\(INVALID,invocation,invocation<ACTIVITY_BRICK_COUNT/);
  assert.doesNotMatch(wgsl, /ACTIVITY_BRICK_LIST/);
  assert.doesNotMatch(wgsl, /atomicCompareExchangeWeak/);
});

test("resident activity consumers use direct brick-domain dispatch", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /dispatch\("measureBrickActivity", this\.incrementalActivityLayout\.brickCount\)/);
  assert.match(source, /dispatch\("clearSparseCM12RetiredFaceVelocitySupport",\s*this\.incrementalActivityLayout\.brickCount\)/);
  assert.doesNotMatch(source, /activityIndirectArguments/);
  assert.doesNotMatch(source, /dispatchActivity\(/);

  const shader = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const measure = shader.slice(shader.indexOf("fn measureBrickActivity"),
    shader.indexOf("fn brickHasTransportDemand", shader.indexOf("fn measureBrickActivity")));
  // Empty stamp slots are expected. Their zero count must be tested before a
  // global base is added, or INVALID+lane can wrap into a watchdog-scale loop.
  assert.match(shader,
    /for\(var localCell=lane;localCell<measuredCount;localCell\+=64u\)\{\s*let cell=first\+localCell;/);
  assert.doesNotMatch(shader, /for\(var cell=first\+lane;/);
  assert.match(measure, /let incidenceRangeForCell=incidenceRange\(cell\);/);
  assert.match(measure,
    /let firstIncidence=incidenceRangeForCell\.x;let incidenceLimit=incidenceRangeForCell\.y;/);
  assert.doesNotMatch(measure,
    /for\(var incidence=incidenceBegin\(cell\);incidence<incidenceEnd\(cell\)/);
  assert.match(measure, /let packedTerms=rowPackedTerms\(row\);/);
  assert.match(measure, /let packedMetadata=rowPackedMetadata\(row\);/);
  assert.match(measure,
    /let incidenceEntry=incidenceRecord\(incidence\);let row=incidenceEntry\.x;/);
  assert.match(measure,
    /let termEntry=termRecord\(term\);let coefficient=bitcast<f32>\(termEntry\.y\);/);
  assert.match(measure, /boundaryLiquidFaces\|=1u<<\(2u\*axis\+side\);/);
  assert.match(measure,
    /activityBoundaryLiquidFaces\[lane\]\|=activityBoundaryLiquidFaces\[lane\+width\];/);
  assert.match(measure, /ACTIVITY_BRICK_BOUNDARY_LIQUID_FACES\+brick/);
  assert.doesNotMatch(measure, /activityPhaseSampleAt|cellLower|cellUpper/,
    "activity must consume compiled row terms, not expand cells to the finest lattice");
  assert.equal((measure.match(/rowDistance\(row\)/g) ?? []).length, 1,
    "row distance must be decoded once outside the indirect term loop");
  assert.match(shader,
    /resolution>1u&&\(x&1u\)==0u&&\(y&1u\)==0u&&\(z&1u\)==0u/);
  assert.match(shader, /var childFixed:array<i32,8>;/);
  assert.doesNotMatch(shader, /let group=2u\*\(vec3u\(x,y,z\)\/2u\)/,
    "all eight children must not repeat the same sibling-octet gather");

  const classify = shader.slice(shader.indexOf("fn classifyAcceptedLiquidFrontier"),
    shader.indexOf("fn activitySignalsEnabled", shader.indexOf(
      "fn classifyAcceptedLiquidFrontier")));
  assert.match(classify, /let sourceFace=2u\*axis\+sourceSide;/);
  assert.match(classify, /ACTIVITY_BRICK_BOUNDARY_LIQUID_FACES\+neighbor/);
  assert.match(classify, /let reciprocalSupportBit=26u-neighborBit;/);
  assert.match(classify, /neighborOutput\+32u/);
  assert.match(classify, /acceptedLiquidDeeplyEnclosed/);
  assert.match(classify, /ACTIVITY_REFINEMENT_POLICY_DEEPLY_ENCLOSED/);
  assert.doesNotMatch(classify, /BRICK_FINE_RESOLUTION\*BRICK_FINE_RESOLUTION/,
    "frontier classification must reduce census receipts, not resample brick faces");
  assert.doesNotMatch(shader, /fn brickDeeplyEnclosed/,
    "planning must not repeat the classifier's six indirect face-neighbour walk");

  const transportDemand = shader.slice(shader.indexOf("fn brickHasTransportDemand"),
    shader.indexOf("fn refinementPolicyTileScale", shader.indexOf(
      "fn brickHasTransportDemand")));
  assert.match(transportDemand, /brickTouchesAcceptedLiquid\(brick\)/);
  assert.doesNotMatch(transportDemand, /cm12WorldOwnerAt|neighborOutput\+32u/,
    "planning must consume the shared frontier receipt without another neighbour walk");
});

test("one sibling-octet owner preserves the exhaustive detail maximum", () => {
  for (const resolution of [2, 4, 8]) {
    for (let count = 1; count <= resolution ** 3; count += 1) {
      const fixed = Array.from({ length: count }, (_, cell) =>
        (97 * cell + 13 * count) % 1000 - 500);
      let repeatedMaximum = 0;
      let electedMaximum = 0;
      for (let local = 0; local < count; local += 1) {
        const x = local % resolution;
        const y = Math.floor(local / resolution) % resolution;
        const z = Math.floor(local / (resolution * resolution));
        const group = [x & ~1, y & ~1, z & ~1];
        let sum = 0;
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const child = group[0]! + dx + resolution * (group[1]! + dy
              + resolution * (group[2]! + dz));
            if (child < count) sum += fixed[child]!;
          }
        }
        repeatedMaximum = Math.max(repeatedMaximum, Math.abs(8 * fixed[local]! - sum));
      }
      for (let local = 0; local < count; local += 1) {
        const x = local % resolution;
        const y = Math.floor(local / resolution) % resolution;
        const z = Math.floor(local / (resolution * resolution));
        if ((x & 1) !== 0 || (y & 1) !== 0 || (z & 1) !== 0) continue;
        const children: number[] = [];
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const child = x + dx + resolution * (y + dy + resolution * (z + dz));
            if (child < count) children.push(fixed[child]!);
          }
        }
        const sum = children.reduce((total, value) => total + value, 0);
        for (const value of children) {
          electedMaximum = Math.max(electedMaximum, Math.abs(8 * value - sum));
        }
      }
      assert.equal(electedMaximum, repeatedMaximum, `${resolution}^3 count ${count}`);
    }
  }
});

test("BFA1 prepares and projects accepted rows without a transient DFRM plane", () => {
  const resident = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(resident, /dispatch\("prepareSparseCM12InteriorFaceTiles"/);
  assert.match(resident, /dispatch\("prepareSparseCM12SeamFacePackets"/);
  assert.doesNotMatch(resident, /DirtyFaceRowMask|compileSparseCM12DirtyFaceRowMasks/);
  assert.doesNotMatch(resident, /dispatchAccepted\("measureDivergenceDiagnostics"/);
  assert.match(resident, /dispatchAccepted\("collocateAndDiagnose", "cell"\)/);
  assert.match(resident, /dispatch\("reduceDivergenceDiagnostics", 1\)/);
});

test("SparseWorld frontier allocation covers all 26 activity-support neighbours", () => {
  const resident = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(resident,
    /dispatchAcceptedFrontierNeighbors\("allocateSparseWorldFrontier"\)/);
  assert.match(resident,
    /acceptedLeafManifestBaseBytes \+ 4 \* 20[\s\S]*acceptedIndirectArguments, 120, 12/);

  const shader = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const allocation = shader.slice(shader.indexOf("fn allocateSparseWorldFrontier"),
    shader.indexOf("fn synthesizeSparseWorldFrontierPages"));
  assert.match(allocation,
    /let brick=acceptedLeafInvocation\(gid\.x\/26u\);let localNeighbor=gid\.x%26u;/);
  assert.match(allocation,
    /let supportBit=select\(localNeighbor,localNeighbor\+1u,localNeighbor>=13u\);/);
  assert.match(allocation, /cm12FluidNeighborReachable\(sourceCoordinate,offset\)/);
  assert.match(allocation,
    /ACTIVITY_FRONTIER_RESOLVED_MASK_WORD[\s\S]*atomicOr\(&activity\[output\+ACTIVITY_FRONTIER_RESOLVED_MASK_WORD\],resolvedBit\)/);

  assert.match(shader,
    /acceptedActive!=candidateActive[\s\S]*candidateActive[\s\S]*atomicAnd\(&activity\[activityRecord\(neighbor\)[\s\S]*ACTIVITY_FRONTIER_RESOLVED_MASK_WORD\],~\(1u<<\(26u-bit\)\)\)/);
  assert.match(resident,
    /setPipeline\(this\.pipelines\.clearSparseWorldFrontierResolutionCache!\)/);
  assert.match(resident, /const ACTIVITY_RECORD_WORDS = 43;/);

  const mapped = Array.from({ length: 26 }, (_, local) => local >= 13 ? local + 1 : local);
  assert.equal(new Set(mapped).size, 26);
  assert.ok(!mapped.includes(13), "the center bit must not allocate the source page");
  assert.deepEqual(mapped, Array.from({ length: 27 }, (_, bit) => bit)
    .filter((bit) => bit !== 13));
});
