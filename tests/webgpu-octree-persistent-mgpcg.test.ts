import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { planOctreePressureCapacity } from "../lib/webgpu-octree";
import {
  OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
  WebGPUOctreePersistentMGPCG,
  decodeOctreePersistentMGPCGBandCensus,
  octreePersistentMGPCGRegularBandRowsMode,
  octreePersistentMGPCGStagedSmootherEnabled,
  octreePersistentMGPCGLanes,
  octreePersistentMGPCGPhaseRepeatProbe,
  octreePersistentMGPCGRestrictedPrefixNetworkEnabled,
  octreePersistentMGPCGStencilColumnsEnabled,
} from "../lib/webgpu-octree-persistent-mgpcg";
import {
  OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER,
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS,
  OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
  OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES,
  OCTREE_PERSISTENT_MGPCG_STENCIL_COLUMNS,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
  octreePersistentMGPCGWorkgroupBytes,
} from "../lib/webgpu-octree-persistent-mgpcg.wgsl";

test("persistent MGPCG packs hot channels by live rows after stable input staging", () => {
  const compact = octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    compactLiveRows: true,
  });
  assert.match(compact, /fn rowStride\(\)->u32\{return wRows;\}/);
  assert.match(compact,
    /let seed=stagedVload\(CH_SEED,row\);\s*let rhs=stagedVload\(CH_RHS,row\);/);
  assert.match(compact,
    /vstore\(CH_RHS,row,rhs\);\s*vstore\(CH_SEED,row,seed\);/);
  assert.match(compact,
    /fn partialBase\(\)->u32\{return ARENA_HEADER\+CHANNELS\*rowStride\(\);\}/);
  assert.match(compact,
    /fn stagedCh\(c:u32,r:u32\)->u32\{return p\.sizes\.w\+\(c-CH_RHS\)\*capacity\(\)\+r;\}/);
  assert.match(compact,
    /else if\(acc\(2u\)>MAX_LIVE_ROWS\)\{reportAt\(ERR_ROW,1u,acc\(2u\)\);\}/);
});

test("persistent MGPCG keeps staged inputs disjoint from compact live channels", () => {
  const capacity = 148_480;
  const partials = Math.ceil(capacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES);
  assert.equal(octreePersistentMGPCGArenaWords(capacity),
    OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
      + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * capacity
      + OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS * partials
      + 2 * capacity);
});

test("persistent MGPCG retains the capacity-strided arena as an A/B oracle", () => {
  const legacy = octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    compactLiveRows: false,
  });
  assert.match(legacy, /fn rowStride\(\)->u32\{return capacity\(\);\}/);
});

test("persistent A2 folds transformed direct and fine-adjoint terms permutation-invariantly", () => {
  const shader = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  assert.match(shader, /fn canonical18Sum\(v:array<f32,18>\)->f32/);
  assert.match(shader, /fn canonical8Sum\(v:array<f32,8>\)->f32/);
  assert.match(shader,
    /children\[child\]=canonical18Sum\(terms\);\}\s*return canonical8Sum\(children\);/,
    "a D4 transform permutes both the eighteen fine directions and eight children");
  assert.match(shader,
    /coefficients\[base\]-canonical18Sum\(coefficientTerms\)\)\*x\+canonical18Sum\(directTerms\)/,
    "the persistent executor must preserve the hierarchical A2 row fold");
  assert.doesNotMatch(shader, /result\+=c\*\(x-vload\(inCh,other\)\)/,
    "catalog iteration order must not determine an A2 result");
  assert.match(shader,
    /fn regularCoefficientTailValid[^]*for\(var channel=6u;channel<18u;channel\+=1u\)[^]*reportAt\(ERR_AUTHORITY,32u,row\)/,
    "class-0 rows must fail closed if an omitted edge coefficient is nonzero");
  assert.match(shader, /if\(cls==0u&&!regularCoefficientTailValid\(row\)\)\{continue;\}/,
    "the class-0 coefficient contract must be checked once during P4 publication");
});

test("opposite-pair stencil orbits are closed under every signed axis transform", () => {
  const directions = [
    [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
    [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
    [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
  ] as const;
  const pairClass = new Map([
    ["0,1", "axis"], ["2,3", "axis"], ["4,5", "axis"],
    ["6,9", "diagonal"], ["7,8", "diagonal"], ["10,13", "diagonal"],
    ["11,12", "diagonal"], ["14,17", "diagonal"], ["15,16", "diagonal"],
  ]);
  const key = (v: readonly number[]) => v.join(",");
  const index = new Map(directions.map((direction, i) => [key(direction), i]));
  const permutations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [2, 0, 1], [1, 2, 0], [2, 1, 0]];
  for (let code = 0; code < 48; code += 1) {
    const signs = [code & 1 ? -1 : 1, code & 2 ? -1 : 1, code & 4 ? -1 : 1];
    const permutation = permutations[Math.floor(code / 8) % 6]!;
    for (const [pair, orbit] of pairClass) {
      const transformed = pair.split(",").map((entry) => {
        const direction = directions[Number(entry)]!.map((value, axis) => value * signs[axis]!);
        return index.get(key(permutation.map((axis) => direction[axis]!)))!;
      }).sort((a, b) => a - b).join(",");
      assert.equal(pairClass.get(transformed), orbit, `transform ${code} maps ${pair} to ${transformed}`);
    }
  }
});

test("persistent M1 uses sparse live-slot Jacobi and the exact transfer adjoint", () => {
  const shader = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  assert.equal(OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES, 5_184);
  assert.match(shader,
    /fn sparseSmoothPhase\([^]*let n=count\(l\)[^]*workSlot\(l,i\)/,
    "the smoother enumerates published sparse slots, never the domain or capacity");
  assert.doesNotMatch(shader, /fn smoothGlobalPhase\(/,
    "the sparse traversal must not regress to a domain-shaped global helper");
  assert.match(shader,
    /fn applied\([^]*let columnBase=neighbourBase\(\)\+base\+slot[^]*topology\[columnBase\+k\*span\][^]*canonical18Sum\(terms\)/,
    "restriction consumes the prepublished linear neighbor columns");
  const applied = shader.slice(shader.indexOf("fn applied("), shader.indexOf("fn chebyshevWeight("));
  assert.doesNotMatch(applied, /directoryLookup|pageFor|brickRecord/,
    "a restriction spoke must not rediscover sparse topology");
  assert.match(shader,
    /for\(var step=0u;step<degree;step\+=1u\)[^]*sparseSmoothPhase\([^]*storageBarrier\(\);workgroupBarrier\(\);\}/,
    "each sparse Chebyshev phase is globally visible to the persistent workgroup");
  assert.doesNotMatch(shader, /HALO_ELEMENTS|blockRepresentative|optionalSparseSlot/,
    "M1 smoothing must do no page-halo discovery beyond the published slot list");
  assert.match(shader,
    /storef\(S_RHS,l\+1u,coarse,loadf\(S_RHS,l\+1u,coarse\)\+sum\)/,
    "E^T adds finer residuals to native coarse pressure variables");
  assert.match(shader, /fn sorted64PrefixSum\([^]*width=2u[^]*other=i\^stride/,
    "canonical E^T order uses a bounded data-independent network");
  assert.match(shader, /value\+=canonical8Sum\(terms\)/,
    "prolongation must be invariant to the transformed corner order");
});

test("persistent MGPCG selects exact live rows, not provisioned capacity", () => {
  assert.equal(OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD, 65_536);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(4_096), true);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(9_216), true);
  assert.equal(WebGPUOctreePersistentMGPCG.acceptsLiveRows(65_537), false);
});

test("large provisioned pressure headroom does not reject adaptive construction", () => {
  const capacity = planOctreePressureCapacity(
    { nx: 96, ny: 64, nz: 96 }, 32, 4, undefined, false, 0.22,
  ).rowCapacity;
  assert.ok(capacity > OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
    "fixture must distinguish conservative capacity from exact adaptive rows");
  const projectionSource = readFileSync(
    new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8",
  );
  assert.doesNotMatch(projectionSource,
    /acceptsLiveRows\(this\.pressureCapacity\.rowCapacity\)/,
    "provisioned capacity exists before the GPU publishes adaptive rows");
});

// ---------------------------------------------------------------------------
// FLUID_OCTREE_MGPCG_STAGED_SMOOTHER — memory-only smoother staging
// ---------------------------------------------------------------------------

const SPOKES = Array.from({ length: 18 }, (_, k) => k);
const stagedShader = () => octreePersistentMGPCGWGSL({
  maximumIterations: 10, stagedSmoother: true,
});
const appliedBodyOf = (shader: string) =>
  shader.slice(shader.indexOf("fn applied("), shader.indexOf("fn chebyshevWeight("));

test("the staged smoother is on unless its flag is exactly 0", () => {
  // Default ON since its bitwise gate was witnessed on device: the
  // `symmetric-expansion` D4 window is unmoved at 68/69 with it enabled.
  assert.equal(octreePersistentMGPCGStagedSmootherEnabled({}), true);
  assert.equal(octreePersistentMGPCGStagedSmootherEnabled(
    { FLUID_OCTREE_MGPCG_STAGED_SMOOTHER: undefined }), true);
  assert.equal(octreePersistentMGPCGStagedSmootherEnabled(
    { FLUID_OCTREE_MGPCG_STAGED_SMOOTHER: "1" }), true);
  assert.equal(octreePersistentMGPCGStagedSmootherEnabled(
    { FLUID_OCTREE_MGPCG_STAGED_SMOOTHER: "0" }), false,
  "exactly \"0\" is the documented rollback to the unstaged smoother");
  assert.equal(octreePersistentMGPCGStagedSmootherEnabled(
    { FLUID_OCTREE_MGPCG_STAGED_SMOOTHER: "" }), true);
});

test("the default persistent kernel is unchanged by the staged-smoother option", () => {
  const authored = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  assert.equal(octreePersistentMGPCGWGSL({ maximumIterations: 10, stagedSmoother: false }),
    authored, "declining the option must emit the identical shader text");
  assert.doesNotMatch(authored, /wDispatch|wAccepted/,
    "the default kernel declares no staged workgroup regions");
  assert.match(authored, /fn acc\(word:u32\)->u32\{return acceptedBoundary\[word\];\}/);
  assert.match(authored, /fn dispatchWord\(word:u32\)->u32\{return arena\[H_DISPATCH\+word\];\}/);
  assert.match(appliedBodyOf(authored), /var terms:array<f32,18>;/);
});

test("staged smoother stages only regions no phase of the solve writes", () => {
  const shader = stagedShader();
  assert.match(shader, /var<workgroup> wDispatch:array<u32,256>;/);
  assert.match(shader, /var<workgroup> wAccepted:array<u32,8>;/);
  assert.equal(OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords, 256,
    "the staged dispatch window must cover the whole authored header");
  // One reader each, and it is the staging copy: every later `count()`,
  // `pageCount()`, `transferCount()` and `acceptedBank()` now hits workgroup
  // storage, so the copy is the kernel's only view of either region.
  assert.equal(shader.match(/arena\[H_DISPATCH/g)?.length, 1);
  assert.equal(shader.match(/acceptedBoundary\[/g)?.length, 1);
  assert.match(shader, /fn acc\(word:u32\)->u32\{return wAccepted\[word\];\}/);
  assert.match(shader, /fn dispatchWord\(word:u32\)->u32\{return wDispatch\[word\];\}/);
  // The copy must precede P0, which snapshots the accepted authority itself.
  const prologue = shader.indexOf("P-1: solve-invariant header staging");
  assert.ok(prologue > 0 && prologue < shader.indexOf("P0: authority validation"),
    "staging must be visible before the lane-zero authority snapshot reads it");
  assert.match(shader,
    /P-1: solve-invariant header staging[^]*wDispatch\[w\]=arena\[H_DISPATCH\+w\];\}\s*if\(lane<8u\)\{wAccepted\[lane\]=acceptedBoundary\[lane\];\}\s*storageBarrier\(\);workgroupBarrier\(\);/,
    "both copies must complete behind one barrier before any consumer runs");
});

test("staged smoother workgroup budget stays inside the guaranteed device floor", () => {
  assert.equal(OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES, 256 * 4 + 8 * 4);
  assert.equal(octreePersistentMGPCGWorkgroupBytes(), OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES);
  assert.equal(octreePersistentMGPCGWorkgroupBytes(false), 5_184);
  assert.equal(octreePersistentMGPCGWorkgroupBytes(true), 6_240);
  // WebGPU guarantees 16,384 bytes of maxComputeWorkgroupStorageSize. The
  // deleted 10x10x6 page halo alone was 4 * 600 * 4 = 9,600 bytes; the staged
  // smoother deliberately buys locality that fits with 10 KiB to spare.
  assert.ok(octreePersistentMGPCGWorkgroupBytes(true) <= 16_384);
});

test("staged applied keeps the eighteen canonical terms in their authored order", () => {
  const applied = appliedBodyOf(stagedShader());
  assert.doesNotMatch(applied, /var terms:array<f32,18>|terms\[k\]/,
    "a dynamically indexed local array is what Metal backs with scratch");
  assert.doesNotMatch(applied, /for\(var k=/,
    "the eighteen spokes must not re-enter a dependent loop");
  assert.doesNotMatch(applied, /directoryLookup|pageFor|brickRecord/,
    "a restriction spoke must not rediscover sparse topology");
  for (const k of SPOKES) {
    assert.ok(applied.includes(`let c${k}=loadf(S_XP+${k}u,l,slot);`),
      `spoke ${k} must read the same coefficient channel HEAD reads`);
    assert.ok(applied.includes(`let n${k}=topology[columnBase+${k}u*span];`),
      `spoke ${k} must read the same prepublished neighbour column`);
    assert.ok(applied.includes(`let g${k}=c${k}!=0.0&&n${k}!=INVALID&&n${k}<cap;`),
      `spoke ${k} must keep HEAD's nonzero-coefficient and in-capacity guard`);
    assert.ok(applied.includes(`if(c${k}!=0.0&&!g${k}){reportAt(OVERFLOW,75u,slot);}`),
      `spoke ${k} must report the identical unresolvable column`);
    assert.ok(applied.includes(`let x${k}=loadf(source,l,select(0u,n${k},g${k}));`),
      `spoke ${k} must fetch the value from an unconditionally in-bounds slot`);
    assert.ok(applied.includes(`let t${k}=select(0.0,c${k}*x${k},g${k});`),
      `spoke ${k} must fall back to HEAD's zero-initialized term`);
  }
  // The whole bitwise contract of this change: the same eighteen terms reach
  // the same summation tree in ascending channel order, so the axis/diagonal
  // pairing in canonical18Sum — and the exact-D4 window it buys — cannot move.
  assert.ok(applied.includes(
    `-canonical18Sum(array<f32,18>(${SPOKES.map((k) => `t${k}`).join(",")}));`),
  "terms must enter canonical18Sum in ascending channel order");
  assert.match(applied,
    /return loadf\(S_DIAG,l,slot\)\*loadf\(source,l,slot\)-canonical18Sum\(/,
    "the diagonal term and its subtraction are untouched");
  // Ascending emission order also pins the report sequence a single lane makes.
  const reportOrder = [...applied.matchAll(/if\(c(\d+)!=0\.0&&!g\d+\)\{reportAt/g)]
    .map((match) => Number(match[1]));
  assert.deepEqual(reportOrder, SPOKES);
});

test("staged smoother changes no other arithmetic in the kernel", () => {
  const authored = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  const staged = stagedShader();
  const withoutApplied = (shader: string) =>
    shader.replace(appliedBodyOf(shader), "")
      .replace(/var<workgroup> wDispatch:array<u32,256>;\nvar<workgroup> wAccepted:array<u32,8>;\n/, "")
      .replace(/\n *\/\/ --- P-1[^]*?storageBarrier\(\);workgroupBarrier\(\);/, "")
      .replace(/return wAccepted\[word\];/, "return acceptedBoundary[word];")
      .replace(/return wDispatch\[word\];/, "return arena[H_DISPATCH+word];");
  assert.equal(withoutApplied(staged), withoutApplied(authored),
    "the staged variant may differ only in applied() and the header staging");
});

// ---------------------------------------------------------------------------
// FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS — class-0 routing for the band sweep
// ---------------------------------------------------------------------------

const bandShader = (
  regularBandRows: "census" | "route",
  stagedSmoother = false,
) => octreePersistentMGPCGWGSL({ maximumIterations: 10, regularBandRows, stagedSmoother });

const bandRowsBodyOf = (shader: string) =>
  shader.slice(shader.indexOf("fn applyBandRows("),
    shader.indexOf("// Section 4.3 boundary-band classification"));

/** Strip exactly the three regions this option is allowed to add. */
const withoutBandRows = (shader: string) => shader
  .replace(/const H_BAND_CENSUS=\d+u;\nconst BAND_CENSUS_MARKER=\d+u;\n/, "")
  .replace(/\/\/ The band shell is\n[^]*?\(P4b\)\.\n(?=fn applyBandRows\()/, "")
  .replace(
    "if(uload(CH_BANDA,row)!=0u){applyRegularRow(row,inCh,outCh);}else{applyRow(row,inCh,outCh);}",
    "applyRow(row,inCh,outCh);")
  .replace(/\n *\/\/ --- P4b[^]*?storageBarrier\(\);workgroupBarrier\(\);(?=\n *compactBand)/, "")
  .replace(/\n *if\(lane==0u\)\{\n *var censusRegular[^]*?arena\[H_BAND_CENSUS\+3u\]=censusCoarse;\}/, "");

test("the regular band-row option is off unless its flag names a mode", () => {
  assert.equal(octreePersistentMGPCGRegularBandRowsMode({}), "off");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: undefined }), "off");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: "0" }), "off");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: "true" }), "off",
  "a Gate-B opt-in must not accept a loose truthy");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: "CENSUS" }), "off");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: "census" }), "census");
  assert.equal(octreePersistentMGPCGRegularBandRowsMode(
    { FLUID_OCTREE_MGPCG_REGULAR_BAND_ROWS: "1" }), "route");
});

test("the default persistent kernel is unchanged by the regular band-row option", () => {
  const authored = octreePersistentMGPCGWGSL({ maximumIterations: 10 });
  assert.equal(
    octreePersistentMGPCGWGSL({ maximumIterations: 10, regularBandRows: undefined }),
    authored, "declining the option must emit the identical shader text");
  assert.doesNotMatch(authored, /H_BAND_CENSUS|BAND_CENSUS_MARKER|--- P4b|CH_BANDA,row,0u/,
    "the default kernel declares no census words and publishes no class map");
  assert.match(bandRowsBodyOf(authored),
    /if\(!stopped\(\)&&row!=INVALID&&row<rows\(\)\)\{applyRow\(row,inCh,outCh\);\}/,
    "HEAD routes every band row through the full eighteen-channel apply");
  assert.throws(() => octreePersistentMGPCGWGSL({
    maximumIterations: 10,
    regularBandRows: "regular" as unknown as "route",
  }), /regular band-row mode is invalid/);
});

test("census mode publishes the class map and the census without moving arithmetic", () => {
  const census = bandShader("census");
  assert.match(bandRowsBodyOf(census),
    /if\(!stopped\(\)&&row!=INVALID&&row<rows\(\)\)\{applyRow\(row,inCh,outCh\);\}/,
    "the measurement arm must leave every band row on the authored apply");
  assert.match(census, /--- P4b: class-0 row map for the Section 4\.3 band sweep/);
  assert.match(census, /arena\[H_BAND_CENSUS\]=BAND_CENSUS_MARKER;/);
  assert.equal(withoutBandRows(census), octreePersistentMGPCGWGSL({ maximumIterations: 10 }),
    "census mode may differ only in the declarations, the map and the census");
});

test("route mode sends class-0 band rows down the same path applyAllRows uses", () => {
  const routed = bandShader("route");
  assert.match(bandRowsBodyOf(routed),
    /if\(uload\(CH_BANDA,row\)!=0u\)\{applyRegularRow\(row,inCh,outCh\);\}else\{applyRow\(row,inCh,outCh\);\}/,
    "class-0 band rows take applyRegularRow, everything else keeps applyRow");
  assert.match(routed,
    /if\(!stopped\(\)&&row<liveRows\)\{if\(cls==0u\)\{applyRegularRow\(row,inCh,outCh\);\}/,
    "applyAllRows keeps its own class-0 routing unchanged");
  assert.equal(withoutBandRows(routed), octreePersistentMGPCGWGSL({ maximumIterations: 10 }),
    "route mode may differ only in the declarations, the map, the census and the sweep");
  // Class 4 is deliberately NOT rerouted: applyBandRows never called
  // applyIdentityRow, and admitting it here would be a separate change.
  assert.doesNotMatch(bandRowsBodyOf(routed), /applyIdentityRow/);
});

test("the class-0 map is published where CH_BANDA is provably dead", () => {
  const routed = bandShader("route");
  const lastDilation = routed.lastIndexOf("ustore(CH_BANDB,row,dilatedBand(row,false));");
  const map = routed.indexOf("--- P4b: class-0 row map");
  const compaction = routed.indexOf("compactBand(lane,liveRows);");
  assert.ok(lastDilation > 0 && lastDilation < map && map < compaction,
    "the map must land after the third dilation and before the band compaction");
  // Only `dilatedBand(row,false)` reads CH_BANDA (through `bandAt`), and the
  // ping-pong ends before the map, so nothing observes the repurposed channel.
  assert.equal(routed.slice(map).match(/dilatedBand\(|bandAt\(/g), null,
    "no band dilation may read CH_BANDA after the map overwrites it");
  assert.match(routed, /fn bandAt\(row:u32,useB:bool\)->u32\{return select\(uload\(CH_BANDA,row\),uload\(CH_BANDB,row\),useB\);\}/);
  assert.match(routed, /if\(uload\(CH_BANDB,row\)==0u\)\{return current;\}/,
    "the smoother's own band gate reads CH_BANDB, which the map never touches");
  // The map is exactly applyAllRows' cls==0 membership, not a re-derivation.
  assert.match(routed,
    /let regularRowCount=worksetCount\(0u\);\s*for\(var item=lane;item<regularRowCount;item\+=LANES\)\{\s*let row=worksetRow\(0u,item\);/);
});

test("the full-apply oracle disables the band routing with applyAllRows", () => {
  const oracle = octreePersistentMGPCGWGSL({
    maximumIterations: 10, regularBandRows: "route", fullApplyOracle: true,
  });
  assert.match(bandRowsBodyOf(oracle),
    /if\(uload\(CH_BANDA,row\)!=0u\)\{applyRow\(row,inCh,outCh\);\}else\{applyRow\(row,inCh,outCh\);\}/,
    "both arms must be the full apply so the oracle still isolates one variable");
  assert.doesNotMatch(oracle, /applyRegularRow\(row,inCh,outCh\)/);
});

test("band-row routing composes with the staged smoother in all four combinations", () => {
  const off = (staged: boolean) =>
    octreePersistentMGPCGWGSL({ maximumIterations: 10, stagedSmoother: staged });
  for (const staged of [false, true]) {
    for (const mode of ["census", "route"] as const) {
      assert.equal(withoutBandRows(bandShader(mode, staged)), off(staged),
        `${mode} + ${staged ? "staged" : "direct"} smoother must be separable`);
    }
  }
  // The staged smoother stages arena words [16, 272); the census words live at
  // 288, so neither option can read or write the other's region.
  assert.equal(OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus, 288);
  assert.equal(OCTREE_PERSISTENT_MGPCG_HEADER.bandCensusWords, 4);
  assert.ok(OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus
    >= OCTREE_PERSISTENT_MGPCG_HEADER.dispatch + OCTREE_PERSISTENT_MGPCG_HEADER.dispatchWords);
  assert.ok(OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus
    >= OCTREE_PERSISTENT_MGPCG_HEADER.worksetHeaders
      + OCTREE_PERSISTENT_MGPCG_HEADER.worksetHeaderWords);
  assert.ok(OCTREE_PERSISTENT_MGPCG_HEADER.bandCensus
    + OCTREE_PERSISTENT_MGPCG_HEADER.bandCensusWords
    <= OCTREE_PERSISTENT_MGPCG_HEADER.totalWords);
  const staged = bandShader("route", true);
  assert.equal(staged.match(/arena\[H_DISPATCH/g)?.length, 1,
    "the staged dispatch copy stays the kernel's only view of that header");
  assert.equal(staged.match(/acceptedBoundary\[/g)?.length, 1);
});

test("the band census decodes only words this dispatch authored", () => {
  assert.equal(OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 0x4241_4e44);
  assert.equal(decodeOctreePersistentMGPCGBandCensus(Uint32Array.from([0, 1, 1, 0])), null,
    "an unwritten arena must never read back as a measurement");
  assert.equal(decodeOctreePersistentMGPCGBandCensus(
    Uint32Array.from([OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 4])), null,
  "a short readback is not a census");
  assert.equal(decodeOctreePersistentMGPCGBandCensus(
    Uint32Array.from([OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 10, 11, 0])), null,
  "more regular band rows than band rows is impossible");
  assert.equal(decodeOctreePersistentMGPCGBandCensus(
    Uint32Array.from([OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 10, 4, 5])), null,
  "more coarse regular rows than regular rows is impossible");
  assert.deepEqual(decodeOctreePersistentMGPCGBandCensus(
    Uint32Array.from([OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 1_000, 250, 40])), {
    bandRows: 1_000, regularBandRows: 250, coarseRegularBandRows: 40, regularShare: 0.25,
  });
  assert.equal(decodeOctreePersistentMGPCGBandCensus(
    Uint32Array.from([OCTREE_PERSISTENT_MGPCG_BAND_CENSUS_MARKER, 0, 0, 0]))?.regularShare, 0,
  "an empty band reports no share rather than a NaN");
});

test("the stencil column cache is on by default and rolls back on exactly zero", () => {
  assert.equal(octreePersistentMGPCGStencilColumnsEnabled({}), true);
  assert.equal(octreePersistentMGPCGStencilColumnsEnabled(
    { FLUID_OCTREE_MGPCG_STENCIL_COLUMNS: undefined }), true);
  assert.equal(octreePersistentMGPCGStencilColumnsEnabled(
    { FLUID_OCTREE_MGPCG_STENCIL_COLUMNS: "1" }), true);
  assert.equal(octreePersistentMGPCGStencilColumnsEnabled(
    { FLUID_OCTREE_MGPCG_STENCIL_COLUMNS: "0" }), false,
  "exactly \"0\" is the documented rollback to the per-sweep page chase");
  assert.equal(octreePersistentMGPCGStencilColumnsEnabled(
    { FLUID_OCTREE_MGPCG_STENCIL_COLUMNS: "" }), true);
});

test("persistent MGPCG publishes solve-invariant stencil columns above the staged inputs", () => {
  const capacity = 148_480;
  const partials = Math.ceil(capacity / OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES);
  const baseline = OCTREE_PERSISTENT_MGPCG_HEADER.totalWords
    + OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT * capacity
    + OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS * partials
    + 2 * capacity;
  assert.equal(octreePersistentMGPCGArenaWords(capacity, false), baseline,
    "the declined cache must not move a single arena word");
  assert.equal(octreePersistentMGPCGArenaWords(capacity, true),
    baseline + OCTREE_PERSISTENT_MGPCG_STENCIL_COLUMNS * capacity);

  const cached = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, stencilColumnCache: true,
  });
  // The region starts exactly where the two capacity-strided staged input
  // channels end, so `channelByteOffset` and the external staging ABI are
  // unmoved and the arena still ends on the last column.
  assert.match(cached, /fn stencilBase\(\)->u32\{return p\.sizes\.w\+2u\*capacity\(\);\}/);
  assert.match(cached,
    /fn stencilColumnAt\(row:u32,k:u32\)->u32\{return stencilBase\(\)\+k\*capacity\(\)\+row;\}/);
  assert.equal(cached.includes(`const STENCIL_COLUMNS=${OCTREE_PERSISTENT_MGPCG_STENCIL_COLUMNS}u;`), true);
  // Addressing only: the apply keeps its eighteen terms and its fold, and
  // loses only the per-sweep page/brick/rank chase.
  assert.match(cached,
    /let owner=stencilColumn\(row,channel\);if\(owner==0u\)\{continue;\}\s*directTerms\[channel\]=c\*\(x-vload\(inCh,owner-1u\)\);/);
  assert.equal(cached.includes("let slot=opPageSlot(l,page,q,vec3u(targetQ),row);\n  if(slot==INVALID){reportAt(ERR_ROW,28u,row);continue;}\n  let flags=state[at(S_FLAGS,l,slot)];\n  if((flags&MG_ONLY)!=0u){continue;}\n  let encoded=state[at(S_OWNER,l,slot)];\n  if(encoded==0u||encoded>capacity()){reportAt(ERR_ROW,29u,row);continue;}\n  directTerms[channel]"), false,
    "the chased arm must not survive inside the cached apply");
  // The zero sweep covers every live row; only classes 0..3 resolve, so a
  // class-4 identity row reads an all-zero cache exactly as HEAD reads
  // eighteen zero coefficients.
  assert.match(cached,
    /for\(var k=0u;k<STENCIL_COLUMNS;k\+=1u\)\{arena\[stencilColumnAt\(row,k\)\]=0u;\}/);
  assert.match(cached, /for\(var cls=0u;cls<4u;cls\+=1u\)\{let n=worksetCount\(cls\);/);

  const declined = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, stencilColumnCache: false,
  });
  assert.equal(declined,
    octreePersistentMGPCGWGSL({ maximumIterations: 10, compactLiveRows: true }),
    "declining the cache must emit a byte-identical module, comments included");
  assert.equal(declined.includes("stencilColumn"), false);
});

test("the restriction sum's sorting network narrows to next_pow2 without moving a term", () => {
  assert.equal(octreePersistentMGPCGRestrictedPrefixNetworkEnabled({}), true);
  assert.equal(octreePersistentMGPCGRestrictedPrefixNetworkEnabled(
    { FLUID_OCTREE_MGPCG_RESTRICTED_PREFIX_SORT: "1" }), true);
  assert.equal(octreePersistentMGPCGRestrictedPrefixNetworkEnabled(
    { FLUID_OCTREE_MGPCG_RESTRICTED_PREFIX_SORT: "0" }), false,
  "exactly \"0\" is the documented rollback to the full 64-wide network");

  const restricted = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, restrictedPrefixNetwork: true,
  });
  // Six doublings cover the caller's own termCount ceiling of 64.
  assert.match(restricted,
    /var m=1u;for\(var step=0u;step<6u;step\+=1u\)\{if\(m>=n\)\{break;\}m=m\*2u;\}/);
  assert.match(restricted, /for\(var i=n;i<m;i\+=1u\)\{sorted\[i\]=3\.402823e38;\}/);
  assert.match(restricted, /for\(var width=2u;width<=m;width\*=2u\)/);
  assert.equal(restricted.includes("for(var width=2u;width<=64u;width*=2u)"), false,
    "the full-width network must not survive alongside the restricted one");
  // The compare-exchange rule and the accumulation are untouched: this is a
  // loop bound, not a different sum.
  assert.match(restricted,
    /let ascending=\(i&width\)==0u;let a=sorted\[i\];let b=sorted\[other\];/);
  assert.match(restricted, /var sum=0\.0;for\(var i=0u;i<n;i\+=1u\)\{sum\+=sorted\[i\];\}return sum;/);

  const full = octreePersistentMGPCGWGSL({ maximumIterations: 10, compactLiveRows: true });
  assert.equal(octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, restrictedPrefixNetwork: false,
  }), full, "declining the option must emit a byte-identical module");
  assert.match(full, /for\(var width=2u;width<=64u;width\*=2u\)/);
});

test("the phase repeat probe is idempotent, opaque to the compiler, and absent by default", () => {
  assert.equal(octreePersistentMGPCGPhaseRepeatProbe({}), undefined);
  assert.deepEqual(octreePersistentMGPCGPhaseRepeatProbe(
    { FLUID_PERSISTENT_MGPCG_PHASE_REPEAT: "band:4" }), { phase: "band", repeats: 4 });
  assert.deepEqual(octreePersistentMGPCGPhaseRepeatProbe(
    { FLUID_PERSISTENT_MGPCG_PHASE_REPEAT: "all-rows" }), { phase: "all-rows", repeats: 1 });
  assert.deepEqual(octreePersistentMGPCGPhaseRepeatProbe(
    { FLUID_PERSISTENT_MGPCG_PHASE_REPEAT: "smooth:6" }), { phase: "smooth", repeats: 6 });
  assert.throws(() => octreePersistentMGPCGPhaseRepeatProbe(
    { FLUID_PERSISTENT_MGPCG_PHASE_REPEAT: "restrict:2" }), /Unknown persistent MGPCG phase repeat/);
  assert.throws(() => octreePersistentMGPCGPhaseRepeatProbe(
    { FLUID_PERSISTENT_MGPCG_PHASE_REPEAT: "band:0" }), /integer in \[1,64\]/);

  const band = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, phaseRepeatProbe: "band",
  });
  // The bound is a uniform read, never a literal: a folded repeat would measure
  // nothing while looking like a null result.
  assert.match(band,
    /for\(var probeRepeat=0u;probeRepeat<=p\.worksets\.z;probeRepeat\+=1u\)/);
  assert.equal(band.includes("probeRepeat<=1u"), false);

  const authored = octreePersistentMGPCGWGSL({ maximumIterations: 10, compactLiveRows: true });
  assert.equal(authored.includes("probeRepeat"), false,
    "the absent probe must leave the production kernel free of it");
  const allRows = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, phaseRepeatProbe: "all-rows",
  });
  assert.equal(allRows.includes("probeRepeat"), true);
  assert.notEqual(allRows, band, "the two probes select different phases");

  // The smoother's value-neutrality rests on source != destination, which
  // `smoothLevel` guarantees by alternating S_A->S_B and S_B->S_A. Pin that
  // alternation here: if it ever became an in-place sweep the probe would
  // silently stop being idempotent.
  const smooth = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, phaseRepeatProbe: "smooth",
  });
  assert.match(smooth,
    /for\(var probeRepeat=0u;probeRepeat<=p\.worksets\.z;probeRepeat\+=1u\)\{ for\(var i=lane;i<n;i\+=LANES\)/);
  assert.match(smooth,
    /sparseSmoothPhase\(l,S_A,S_B,phase,degree,lane\);.*\s*.*sparseSmoothPhase\(l,S_B,S_A,phase,degree,lane\);/);
  assert.notEqual(smooth, band);
  assert.notEqual(smooth, allRows);
});

test("the persistent workgroup width is a work-distribution stride only", () => {
  assert.equal(octreePersistentMGPCGLanes({}), 256);
  assert.equal(octreePersistentMGPCGLanes({ FLUID_OCTREE_MGPCG_LANES: "512" }), 512);
  assert.equal(octreePersistentMGPCGLanes({ FLUID_OCTREE_MGPCG_LANES: "1024" }), 1024);
  assert.throws(() => octreePersistentMGPCGLanes({ FLUID_OCTREE_MGPCG_LANES: "384" }),
    /must be 256, 512 or 1024/, "non-power-of-two widths break compactBand's scan");

  const authored = octreePersistentMGPCGWGSL({ maximumIterations: 10, compactLiveRows: true });
  assert.equal(octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, lanes: 256,
  }), authored, "the authored width must emit a byte-identical module");

  const wide = octreePersistentMGPCGWGSL({
    maximumIterations: 10, compactLiveRows: true, lanes: 1024,
  });
  assert.match(wide, /const LANES=1024u;/);
  assert.match(wide, /@compute @workgroup_size\(1024\)/);
  // The scan array must track the width or compactBand reads out of bounds.
  assert.match(wide, /var<workgroup> scan:array<u32,1024>;/);
  // The reductions must NOT track it: their association order is what the
  // compensated sum depends on, and it is indexed by REDUCTION_LANES alone.
  assert.match(wide, /let row=vg\*REDUCTION_LANES\+lane;/);
  assert.match(wide, /if\(lane<REDUCTION_LANES&&row<liveRows&&!stopped\(\)\)/);
  assert.match(wide, /for\(var width=REDUCTION_LANES\/2u;width>0u;width>>=1u\)/);

  // Workgroup storage has to grow with the scan array and stay inside the
  // 16 KiB floor Dawn exposes on Metal; an overflow here is a validation error
  // the skip_validation harness turns into a SIGSEGV rather than a message.
  for (const lanes of [256, 512, 1024] as const) {
    const bytes = octreePersistentMGPCGWorkgroupBytes(true, lanes);
    assert.equal(bytes, OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES + (lanes - 256) * 4
      + OCTREE_PERSISTENT_MGPCG_STAGED_SMOOTHER_WORKGROUP_BYTES);
    assert.ok(bytes < 16384, `workgroup storage at ${lanes} lanes must fit the floor`);
  }
});

test("every band-row and staged-smoother combination is accepted by naga", () => {
  const probe = spawnSync(process.env.NAGA ?? "naga", ["--version"], { encoding: "utf8" });
  if (probe.error) {
    // `npm run test:water-shaders` is the tree's naga gate and does not yet
    // carry this kernel at all; this keeps the combinations covered wherever
    // naga exists without making the CPU suite depend on it.
    console.log("skipped: naga is not on PATH");
    return;
  }
  const directory = mkdtempSync(join(tmpdir(), "fluid-persistent-mgpcg-wgsl-"));
  try {
    for (const stagedSmoother of [false, true]) {
      for (const stencilColumnCache of [false, true]) {
        for (const restrictedPrefixNetwork of [false, true]) {
        for (const phaseRepeatProbe of [undefined, "band", "all-rows", "smooth"] as const) {
        for (const regularBandRows of [undefined, "census", "route"] as const) {
          const name = `mgpcg-${regularBandRows ?? "off"}-${
            stagedSmoother ? "staged" : "direct"}-${
            stencilColumnCache ? "columns" : "chased"}-${
            restrictedPrefixNetwork ? "narrow" : "wide"}-${phaseRepeatProbe ?? "noprobe"}`;
          const path = join(directory, `${name}.wgsl`);
          writeFileSync(path, octreePersistentMGPCGWGSL({
            maximumIterations: 10, stagedSmoother, stencilColumnCache, restrictedPrefixNetwork,
            ...(phaseRepeatProbe === undefined ? {} : { phaseRepeatProbe }),
            ...(regularBandRows === undefined ? {} : { regularBandRows }),
          }));
          const result = spawnSync(process.env.NAGA ?? "naga", [path], { encoding: "utf8" });
          assert.equal(result.status, 0, `${name}:\n${result.stderr || result.stdout}`);
        }
        }
        }
      }
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});
