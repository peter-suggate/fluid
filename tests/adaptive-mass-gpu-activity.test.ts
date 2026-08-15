import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";
import {
  evaluateSparseCM12Performance,
  SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  type SparseCM12BenchmarkArm,
} from "../lib/methods/adaptive-mass/adaptive-mass-performance";
import { sceneDamBreakBox } from "../lib/core/initial-fluid";
import { cm12Scene } from "../lib/core/cm12-paper-scenes";
import { FINE_LEVELSET_METADATA_WORDS } from
  "../lib/core/fine-levelset-brick-abi";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  createMinimalPowerDamBreak64Scene,
  createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE,
} from "../lib/core/scenes";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  sparseBrickKey,
  sparseBrickAtlasStats,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import {
  dormantReceiverDomain,
  dormantReceiverResolution,
  adaptiveMassPresentationDimensionsForScene,
  residentSupportAtlas,
  SPARSE_CM12_RECEIVER_SUPPORT_RINGS,
} from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { sparseCM12FinePresentationPlan } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("resident activity measurement is GPU-owned and disjoint from accepted fields", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /@group\(0\)@binding\(12\)var<storage,read_write>activity:array<atomic<u32>>/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn advanceActivityClock\(\)/);
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn measureBrickActivity");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn classifyPresentationBricks", begin,
  );
  assert.ok(begin >= 0 && end > begin, "activity kernel must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /atomicStore\(&activity\[/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted physics state");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/,
    "measurement must not mutate accepted topology");
});

test("resident sharpening converts the CM12 pseudo-time to finest-cell units", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn sharpeningDelta");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn scatterSharpeningMass", begin,
  );
  assert.ok(begin >= 0 && end > begin, "sharpening kernel must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /pseudoTimeFineCells=3\.0\*p\.frame\.x\/p\.frame\.y/,
    "3 dt must be divided by finest-cell metres before using grid-coordinate distances");
  assert.match(kernel, /pseudoTimeFineCells\/beforeDistance/);
  assert.doesNotMatch(kernel, /courant\*width\/beforeDistance/,
    "local cell width must not cancel the physical density gradient");
});

test("resident sharpening implements CM12 Algorithm 2 trace and scatter", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn traceSharpeningMass");
  const end = webgpuSparseCM12ResidentWGSL.indexOf("fn preserveHorizontalD4", begin);
  assert.ok(begin >= 0 && end > begin, "sharpening return must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /maximumDistance=p\.sharpening\.x\*sourceWidth/);
  assert.match(kernel, /step<40u/);
  assert.match(kernel, /step>=u32\(p\.sharpening\.y\)/);
  assert.match(kernel, /sampleSharpeningDensity\(position\)>=CM12_LIQUID_ISOVALUE/);
  assert.match(kernel, /sharpeningDensityGradient\(position,owner\)/);
  assert.match(kernel, /0\.5\*cellMinimumWidth\(owner\)/,
    "the trace step must adapt after crossing a 2:1 seam");
  assert.match(kernel, /gradient\/magnitude\*distance/);
  assert.match(kernel, /f32\(removedFixed\)\*term\.weight\/total/,
    "removed integrated mass must be scattered trilinearly at the traced point");
  assert.match(kernel, /offeredFixed=remainingFixed/,
    "fixed-point scatter rounding must remain exactly conservative");
  assert.match(kernel, /\+incoming\/cellVolume\(cell\)/,
    "every fixed-point deposit must be resolved into receiver density");
  assert.doesNotMatch(kernel, /uphillConductance|capacity\/incoming/,
    "the paper trace must not fall back to the old one-face/capacity shortcut");
});

test("resident D4 preservation has disjoint density and gamma scratch", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(source, /sharpeningDelta: cells\(\), symmetryGamma: cells\(\)/,
    "density and gamma symmetry values must not share storage");
  assert.match(source,
    /u\.set\(\[l\.sharpeningDelta, l\.symmetryGamma, 0, 0\], 36\)/,
    "stateOffsets5.y must address allocated gamma scratch rather than densityA");
});

test("adaptive construction leaves interface resolution to the atlas initializer", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf('id: "adaptive-mass.atlas"');
  const end = source.indexOf('id: "adaptive-mass.presentation"', begin);
  const construction = source.slice(begin, end);
  assert.match(construction, /options\.resolutionMode === "all-coarse"/);
  assert.match(construction, /:\s*undefined/);
  assert.match(construction, /\.\.\.\(resolutionForBrick \? \{ resolutionForBrick \} : \{\}\)/);
  assert.doesNotMatch(construction,
    /options\.resolutionMode === "all-fine"[\s\S]*?\? \(\) => 8[\s\S]*?: \(\) => 4,/,
    "adaptive mode must not blanket-force every interface brick to 4 cubed");
});

test("frame scheduling never waits for or consumes activity readback", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  const advanceBegin = source.indexOf("  advanceTo(");
  const advanceEnd = source.indexOf("  private finishFrameCapture(", advanceBegin);
  const advance = source.slice(advanceBegin, advanceEnd);
  assert.doesNotMatch(advance, /readGPUActivityPolicy|readActivitySnapshot|mapAsync/);
  assert.match(source, /Explicit acceptance\/debug readback; never consulted by advanceTo/);
});

test("GPU candidate planning is epoch-gated and cannot mutate accepted state", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn planBrickResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn classifyPresentationBricks", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate planner must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /requested=atomicLoad\(&activity\[output\+8u\]\)/,
    "candidate requests must persist between topology epochs");
  assert.match(kernel, /surface\|\|predicted\|\|score>=224u/);
  assert.match(kernel, /else if\(atomicLoad\(&activity\[5\]\)!=0u\)/);
  assert.match(kernel, /if\(hotEpochs>=2u\)/);
  assert.match(kernel, /requested=min\(8u,2u\*current\)/);
  assert.match(kernel, /quietEpochs>=8u&&!detail/);
  assert.match(kernel, /requested=current\/2u/);
  assert.match(kernel, /atomicStore\(&activity\[output\+8u\],requested\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("GPU candidate levels close the full 1/2/4/8 ladder to 2:1", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn closePlannedResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn activateSweptReceivers", begin,
  );
  assert.ok(begin >= 0 && end > begin, "grading closure must be independently inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /neighborResolution\/2u/);
  assert.match(kernel, /atomicMax\(&activity\[activityRecord\(brick\)\+8u\],required\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(residentSource,
    /for \(let gradingPass = 0; gradingPass < 3; gradingPass \+= 1\)/);
});

test("GPU candidate validation is isolated from accepted level and topology", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn validateCandidateResolution");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn activateSweptReceivers", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate validator must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /validBrickResolution\(accepted\)/);
  assert.match(kernel, /larger>2u\*smaller/);
  assert.match(kernel, /atomicStore\(&activity\[output\+13u\],candidate\)/);
  assert.match(kernel, /atomicStore\(&activity\[output\+14u\]/);
  assert.doesNotMatch(kernel, /atomicStore\(&activity\[output\+12u\]/,
    "validation cannot publish the accepted logical level");
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/);
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("GPU candidate cell transfer is conservative and remains non-authoritative", () => {
  assert.match(webgpuSparseCM12ResidentWGSL,
    /@group\(0\)@binding\(13\)var<storage,read_write>candidateState:array<f32>/);
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn transferCandidateCells");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn activateSweptReceivers", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate transfer must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /candidate<accepted/);
  assert.match(kernel, /momentumSum\/massSum/);
  assert.match(kernel, /candidateState\[candidateFieldIndex\(0u,brick,local\)\]=rho/);
  assert.match(kernel, /atomicStore\(&activity\[output\+18u\],bitcast<u32>\(massError\)\)/);
  assert.match(kernel, /if\(!valid\)\{atomicOr\(&activity\[7\],2u\);\}/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "candidate transfer cannot write accepted fields");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("GPU candidate face transfer area-averages authoritative exterior flux", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn transferCandidateFaces");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn activateSweptReceivers", begin,
  );
  assert.ok(begin >= 0 && end > begin, "candidate face transfer must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(kernel, /state\[destinationFaceVelocity\(\)\+row\]\*rowAreaValue/);
  assert.match(kernel, /candidateState\[candidateFieldIndex\(6u\+side,brick,lane\)\]/);
  assert.match(kernel, /reduceB\[0\]-reduceA\[0\]/);
  assert.match(kernel, /atomicOr\(&activity\[7\],4u\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "face transfer cannot write accepted velocity");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/);
});

test("swept receiver activation is GPU-published and dormant cells stay inert", () => {
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(webgpuSparseCM12ResidentWGSL,
    /atomicCompareExchangeWeak\(&activity\[output\+10u\],0u,1u\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /atomicAdd\(&activity\[11\],topology\[record\+1u\]\)/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /if\(!cellActive\(id\)\)\{\s*state\[destinationDensity\(\)\+id\]=0\.0/);
  const plan = residentSource.indexOf('dispatch("planBrickResolution"');
  const activate = residentSource.indexOf('dispatch("activateSweptReceivers"');
  const present = residentSource.indexOf('dispatch("classifyPresentationBricks"');
  assert.ok(plan >= 0 && activate > plan && present > activate,
    "GPU planning must publish receiver activation before presentation");
  const encodeBegin = residentSource.indexOf("  encode(\n");
  const encodeEnd = residentSource.indexOf("  /** Publish generation zero", encodeBegin);
  const encode = residentSource.slice(encodeBegin, encodeEnd);
  assert.doesNotMatch(encode, /mapAsync|readActivitySnapshot|readGPUActivityPolicy/);
});

test("GPU retirement uses directional surface support without discarding exact mass", () => {
  const begin = webgpuSparseCM12ResidentWGSL.indexOf("fn retireUnsupportedEmptyBricks");
  const end = webgpuSparseCM12ResidentWGSL.indexOf(
    "fn classifyPresentationBricks", begin,
  );
  assert.ok(begin >= 0 && end > begin, "retirement kernel must be inspectable");
  const kernel = webgpuSparseCM12ResidentWGSL.slice(begin, end);
  assert.match(webgpuSparseCM12ResidentWGSL, /occupiedCell=occupiedCell\|\|rho>0\.0/);
  assert.match(kernel, /activity\[output\+1u\]\)&64u/,
    "any positive mass must retain its own brick");
  assert.match(kernel, /activityRecord\(neighbor\)\+32u\]\)&\(1u<<bit\)/,
    "only the neighbor's directional surface/swept bit may retain this air brick");
  assert.match(kernel, /for\(var dz=-1;dz<=1;dz\+=1\)/);
  assert.match(kernel, /for\(var dy=-1;dy<=1;dy\+=1\)/);
  assert.match(kernel, /for\(var dx=-1;dx<=1;dx\+=1\)/);
  assert.match(kernel, /atomicCompareExchangeWeak\(&activity\[output\+10u\],1u,0u\)/);
  assert.match(kernel, /atomicSub\(&activity\[8\],1u\)/);
  assert.doesNotMatch(kernel, /state\[[^\]]+\]\s*=(?!=)/,
    "retirement must not erase accepted fields");
  assert.doesNotMatch(kernel, /topology\[[^\]]+\]\s*=(?!=)/,
    "retirement must not mutate immutable packed topology");
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const activate = residentSource.indexOf('dispatch("activateSweptReceivers"');
  const retire = residentSource.indexOf('dispatch("retireUnsupportedEmptyBricks"');
  const present = residentSource.indexOf('dispatch("classifyPresentationBricks"');
  assert.ok(activate >= 0 && retire > activate && present > retire,
    "receiver publication and support closure must precede retirement and presentation");
});

test("new Sparse CM12 receivers are fine with a full graded support ladder", () => {
  assert.equal(dormantReceiverResolution("adaptive"), 8);
  assert.equal(dormantReceiverResolution("adaptive", 1), 4);
  assert.equal(dormantReceiverResolution("adaptive", 2), 2);
  assert.equal(dormantReceiverResolution("adaptive", 3), 1);
  assert.equal(dormantReceiverResolution("adaptive", 4), 1);
  assert.equal(dormantReceiverResolution("adaptive", 1, 4), 2);
  assert.equal(dormantReceiverResolution("all-coarse"), 4);
  assert.equal(dormantReceiverResolution("all-fine"), 8);
});

test("Figure 7 promotes only fluid-bearing face receivers at construction", () => {
  const scene = cm12Scene("cm12-figure-7");
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: adaptiveMassPresentationDimensionsForScene(scene),
  });
  const supported = residentSupportAtlas(initial, "adaptive");
  const dormant = dormantReceiverDomain(supported, "adaptive");
  const initialStats = sparseBrickAtlasStats(initial);
  const supportStats = sparseBrickAtlasStats(supported);
  const dormantStats = sparseBrickAtlasStats(dormant);
  assert.equal(supportStats.residentBrickCount - initialStats.residentBrickCount, 8,
    "the spherical source must not promote its whole 3x3x3 neighbor shell to 8-cubed");
  assert.equal(supportStats.fineBrickCount - initialStats.fineBrickCount, 8);
  assert.ok(dormantStats.leafCount < 100_000,
    `Figure 7 construction regressed to ${dormantStats.leafCount} packed cells`);
});

test("the 64-cubed mini dam has dormant receivers on every domain axis", () => {
  const scene = createMinimalPowerDamBreak64Scene();
  const dimensions = sceneLatticeDimensions(scene);
  const initial = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: dimensions,
    resolutionForBrick: () => 4,
  });
  const domain = dormantReceiverDomain(initial, "adaptive");
  assert.equal(domain.bricks.length, 8 ** 3);
  assert.ok(domain.bricks.every((brick) => brick.resolution >= 4),
    "a boundary-fed dam needs a traversable 4-cubed immutable receiver pool");
  for (const coordinate of [[7, 0, 7], [0, 7, 7], [7, 7, 7]] as const) {
    const key = coordinate[0] + 8 * (coordinate[1] + 8 * coordinate[2]);
    assert.ok(domain.directory.has(key),
      `missing dormant mini-dam receiver ${coordinate.join(",")}`);
  }
});

test("Sparse CM12 receiver and compact page capacity do not grow with empty world volume", () => {
  const seeded = (dimensions: readonly [number, number, number],
    coordinate: readonly [number, number, number]) => {
    const brickDimensions = dimensions.map((value) => Math.ceil(value / 8)) as
      [number, number, number];
    return createSparseAdaptiveMassAtlas(dimensions, [{
      key: sparseBrickKey(coordinate, brickDimensions),
      coordinate,
      resolution: 8,
      density: new Float64Array(8 ** 3).fill(1),
      gamma: new Float64Array(8 ** 3).fill(1),
    }]);
  };
  const local = dormantReceiverDomain(seeded([256, 32, 32], [16, 2, 2]), "adaptive");
  const vast = dormantReceiverDomain(
    seeded([1_000_000, 32, 32], [62_500, 2, 2]), "adaptive",
  );
  assert.equal(SPARSE_CM12_RECEIVER_SUPPORT_RINGS, 9);
  assert.equal(vast.bricks.length, local.bricks.length,
    "empty world width must not add receiver records");
  assert.equal(sparseBrickAtlasStats(vast).leafCount, sparseBrickAtlasStats(local).leafCount,
    "empty world width must not add solver cells");
  const localPresentation = sparseCM12FinePresentationPlan(local);
  const vastPresentation = sparseCM12FinePresentationPlan(vast);
  assert.equal(vastPresentation.plan.maximumResidentBricks,
    localPresentation.plan.maximumResidentBricks,
    "empty world width must not add compact pages");
  assert.equal(vastPresentation.worklist.length,
    7 + vastPresentation.plan.maximumResidentBricks,
    "compact worklist must omit the logical-domain direct table");
  assert.ok(vastPresentation.plan.logicalBrickCount
    > 1_000 * vastPresentation.plan.maximumResidentBricks,
  "the test world must be materially larger than its retained page set");
});

test("compact Sparse CM12 presentation pages retain their direct source address", () => {
  const atlas = dormantReceiverDomain(createSparseAdaptiveMassAtlas([80, 24, 16], [{
    key: sparseBrickKey([4, 1, 1], [10, 3, 2]),
    coordinate: [4, 1, 1],
    resolution: 8,
    density: new Float64Array(8 ** 3).fill(1),
    gamma: new Float64Array(8 ** 3).fill(1),
  }]), "adaptive", 2);
  const publication = sparseCM12FinePresentationPlan(atlas);
  const [pagesX, pagesY] = publication.plan.brickDimensions;
  let previousKey = -1;
  for (let page = 0; page < publication.plan.maximumResidentBricks; page += 1) {
    const at = FINE_LEVELSET_METADATA_WORDS * page;
    const key = publication.metadata[at + 1]!;
    const source = publication.metadata[at + 3]!;
    const brick = atlas.bricks[source >> 3]!;
    const octant = source & 7;
    const coordinate = [2 * brick.coordinate[0] + (octant & 1),
      2 * brick.coordinate[1] + ((octant >> 1) & 1),
      2 * brick.coordinate[2] + ((octant >> 2) & 1)] as const;
    const expected = coordinate[0] + pagesX * (coordinate[1] + pagesY * coordinate[2]);
    assert.equal(key, expected, "page metadata must directly address its packed source octant");
    assert.ok(key > previousKey, "compact metadata must remain sorted for consumer lookup");
    previousKey = key;
  }
});

test("Sparse CM12 scientific overlays consume accepted compact state directly", () => {
  const overlay = readFileSync(new URL("../lib/core/webgpu-grid-overlay.ts", import.meta.url),
    "utf8");
  const renderer = readFileSync(new URL("../lib/core/webgpu-renderer.ts", import.meta.url),
    "utf8");
  assert.match(overlay, /@binding\(11\) var<storage,read> sparseTopology/);
  assert.match(overlay, /fn sparseBrickLookup\(key:u32\)->u32/);
  assert.match(overlay, /if\(sparseGridEnabled\(\)\)\{return sparseDensityAt/,
    "surface density must come from the live compact resident state");
  assert.match(overlay,
    /select\(sparseP\.stateOffsets0\.x,sparseP\.stateOffsets0\.y,sparseP\.frame\.w>0\.5\)/,
    "density visualization must select the solver's accepted ping-pong bank");
  assert.match(overlay,
    /select\(sparseP\.stateOffsets1\.x,sparseP\.stateOffsets1\.y,sparseP\.frame\.w>0\.5\)/,
    "velocity visualization must select the solver's accepted ping-pong bank");
  assert.match(overlay, /if\(sparseGridEnabled\(\)&&sparseOwner\(cell\)\.x==SPARSE_INVALID\)/,
    "unrepresented world space must remain transparent");
  assert.match(renderer, /setSparseSource\(this\.gpuFluid\.sparseAdaptiveGridSource\)/);
  assert.doesNotMatch(renderer, /readDiagnosticFields/,
    "rendering must never trigger dense QA materialization");
});

test("analytic initial volumes do not trigger an empty-domain brick scan", () => {
  const base = createSparseCM12LongDamBreakScene();
  const seed = base.fluid.initialBrickSeeds_m?.[0] ?? { x: 0, y: 0, z: 0 };
  const h = base.container.width_m / 1_000_000;
  const scene = { ...base, fluid: { ...base.fluid,
    initialBrickSeeds_m: [seed],
    initialBrickSeedsAdditive: false,
    initialLiquidVolumes: [{ shape: "box" as const,
      min_m: { x: seed.x - h, y: seed.y, z: seed.z - h },
      max_m: { x: seed.x + h, y: seed.y + h, z: seed.z + h } }],
  } };
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [1_000_000, 32, 32],
  });
  assert.ok(atlas.bricks.length > 0);
  assert.ok(atlas.bricks.length < 16,
    "local authored sources must not retain bricks from the empty million-cell corridor");
});

test("Sparse CM12 frame publication is page-shaped, not domain-shaped", () => {
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL, /texture_storage_3d/);
  assert.doesNotMatch(webgpuSparseCM12ResidentWGSL, /fn publishPresentation/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn brickDirectoryLookup\(key:u32\).*var low=0u;var high=p\.dispatch\.w/s);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /fn publishSparseLevelSet[\s\S]*pageCount=arrayLength\(&fineMetadata\)\/4u/);
  const residentSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.doesNotMatch(residentSource, /denseCount|brickDirectoryOffset|ownerOffset/);
  assert.doesNotMatch(residentSource, /dispatch\("publishPresentation"/);
  assert.match(residentSource,
    /dispatch\("publishSparseLevelSet",\s*this\.globalFineLevelSetSource\.plan\.maximumResidentBricks\)/);
  const solverSource = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver.ts",
    import.meta.url,
  ), "utf8");
  assert.match(solverSource, /new WebGPUAdaptiveMassSparsePresentation\(device\)/);
  assert.doesNotMatch(solverSource, /new WebGPUAdaptiveMassAtlasPresentation\(/,
    "the sparse solver must not construct full-domain presentation textures");
  assert.match(solverSource, /get globalFineLevelSetSource\(\)/,
    "compact pages must be the renderer's surface authority");
});

test("canonical Sparse CM12 dam defines a tall sparse 96x48x16 traversal", () => {
  const scene = createSparseCM12LongDamBreakScene();
  assert.deepEqual(sceneLatticeDimensions(scene), [96, 48, 16]);
  assert.equal(SPARSE_CM12_LONG_DAM_METHOD_PROFILE.methodId, "adaptive-mass");
  assert.equal(SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides?.timeStep, "paper");
  const dam = sceneDamBreakBox(scene);
  assert.equal(dam.min.x, 0);
  assert.ok(Math.abs(dam.max.x - 1 / 6) < 1e-12);
  assert.ok(Math.abs(dam.max.y - 5 / 12) < 1e-12);
  assert.equal(dam.max.z, 1);
  const atlas = initializeSparseBrickAtlasFromScene(scene, {
    finestDimensions: [96, 48, 16],
    resolutionForBrick: () => 4,
  });
  const stats = sparseBrickAtlasStats(atlas);
  assert.equal(stats.logicalBrickCount, 144);
  assert.equal(stats.residentBrickCount, 12);
  assert.equal(stats.omittedEmptyBrickCount, 132);
  assert.equal(atlas.brickDimensions[0] - 2, 10,
    "the front must cross ten initially dry brick columns");
});

test("mini32 performance accepts a Rung-A fine set without freezing the old one-seed count", () => {
  const arm = (methodId: "uniform" | "adaptive-mass"): SparseCM12BenchmarkArm => ({
    methodId,
    sceneId: "minimal-power-dam-break-32",
    finestDimensions: [32, 32, 32],
    dt_s: 0.004,
    constructionExcluded: true,
    endToEndFrame_ms: new Array(30).fill(methodId === "uniform" ? 10 : 9),
    cpuTraces: [],
    gpuTraces: [],
    initialTopology: methodId === "adaptive-mass" ? {
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    } : undefined,
    evolvedTopology: methodId === "adaptive-mass" ? [{
      fineBricks: 28,
      coarseBricks: 36,
      fineCoarseFaceConnectedPairs: 24,
      mixedSeamRows: 384,
    }] : undefined,
  });
  const verdict = evaluateSparseCM12Performance(
    arm("uniform"), arm("adaptive-mass"),
    SPARSE_CM12_MINI_DAM_32_PERFORMANCE_ACCEPTANCE,
  );
  assert.equal(verdict.passed, true, verdict.failures.join("\n"));
});
