import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  structuredBoundaryCoefficientWGSL,
  structuredBoundaryExactRowCarryEnabled,
  structuredBoundaryExactRowCarryMode,
} from "../lib/webgpu-octree-structured-boundary";

const boundaryEntryPoints = ["prepareStructuredBoundaryCandidate", "prepareStructuredBoundaryAccepted",
  "classifyStructuredLiquidRows",
  "resolveStructuredBoundarySlots", "resolveStructuredSolidSlots", "commitStructuredBoundarySlots",
  "rebuildStructuredBoundaryRows", "countStructuredRowClasses", "countStructuredFamilyClasses",
  "scanStructuredWorksetBlocks", "scatterStructuredRowWorksets", "scatterStructuredFamilyWorksets",
  "acceptStructuredBoundary"] as const;

function reachableStorageBindings(entryPoint: string): number[] {
  const globals = [...structuredBoundaryCoefficientWGSL.matchAll(
    /@group\(0\)\s*@binding\((\d+)\)\s*var<storage,[^>]+>\s*(\w+)/g,
  )].map((match) => ({ binding: Number(match[1]), name: match[2] }));
  const functions = new Map<string, string>();
  const signature = /\bfn\s+(\w+)\s*\(/g;
  for (let match = signature.exec(structuredBoundaryCoefficientWGSL); match;
    match = signature.exec(structuredBoundaryCoefficientWGSL)) {
    const open = structuredBoundaryCoefficientWGSL.indexOf("{", signature.lastIndex);
    assert.notEqual(open, -1, `missing body for ${match[1]}`);
    let cursor = open + 1, depth = 1;
    while (cursor < structuredBoundaryCoefficientWGSL.length && depth > 0) {
      if (structuredBoundaryCoefficientWGSL[cursor] === "{") depth += 1;
      else if (structuredBoundaryCoefficientWGSL[cursor] === "}") depth -= 1;
      cursor += 1;
    }
    assert.equal(depth, 0, `unterminated body for ${match[1]}`);
    functions.set(match[1]!, structuredBoundaryCoefficientWGSL.slice(open + 1, cursor - 1));
    signature.lastIndex = cursor;
  }
  const pending = [entryPoint], reachable = new Set<string>();
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reachable.has(name)) continue;
    reachable.add(name);
    const body = functions.get(name); assert.ok(body, `missing ${name}`);
    for (const candidate of functions.keys()) {
      if (!reachable.has(candidate) && new RegExp(`\\b${candidate}\\s*\\(`).test(body)) pending.push(candidate);
    }
  }
  // Strip comments before matching: binding names appearing in prose
  // ("accepted", "rows", "liquid", "solid") are not shader references.
  const bodies = [...reachable].map((name) => functions.get(name)!
    .replace(/\/\/[^\n]*/g, ""));
  return globals.filter(({ name }) => bodies.some((body) =>
    new RegExp(`(?<![.\\w])${name}\\b`).test(body)))
    .map(({ binding }) => binding).sort((a, b) => a - b);
}

test("structured boundary update is canonical, transactional, and face-graph free", () => {
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL, /PowerFaceRecord|incidence/i);
  // The theta histogram is the one permitted accumulation: it counts crossing
  // buckets for diagnostics and touches no face graph. Banning atomicAdd
  // outright also banned that, so the ban is now on the target instead --
  // any other atomic accumulation here would be face-contribution gathering,
  // which is exactly what this update must not do.
  const atomicTargets = [...structuredBoundaryCoefficientWGSL
    .matchAll(/atomicAdd\(&([A-Za-z0-9_.]+)(\[[^\]]*\])?/g)]
    .map((match) => `${match[1]}${match[2] ? "[]" : ""}`);
  assert.deepEqual([...new Set(atomicTargets)].sort(), ["control.theta[]"],
    "only the theta histogram may accumulate atomically in the boundary update");
  assert.match(structuredBoundaryCoefficientWGSL, /fn fineSample\(/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn bootstrapTexturePhi[\s\S]*point\/p\.physical\.x-vec3f\(\.5\)[\s\S]*round\(2\.\*raw\)[\s\S]*for\(var corner=0u;corner<8u;corner\+=1u\)/,
    "bootstrap texture sampling must use the snapped cell-centred trilinear lattice");
  assert.match(structuredBoundaryCoefficientWGSL,
    /var lattice=\(x-fp\.origin\)\/fp\.width-vec3f\(\.5\)[\s\S]*for\(var corner=0u;corner<8u;corner\+=1u\)[\s\S]*if\(sample\.y==0\.\)\{return vec2f\(coarsePhi,0\.\);\}/,
    "face pressure must use complete trilinear samples on the cell-centred fine lattice");
  assert.match(structuredBoundaryCoefficientWGSL, /fn solidAt\(/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(!world&&\(hi==INVALID\|\|\(\(plo\.x<0\.\)!=\(phi\.x<0\.\)\)\)\)[\s\S]*let theta=max\(-plo\.x\/\(phi\.x-plo\.x\),1e-2\);[\s\S]*scale=1\.\/theta/,
    "the ghost-fluid pressure scale must use the dual-edge crossing with the Gibou robustness floor");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL, /clamp\([^;]*theta|theta=clamp/,
    "the wall-impact pressure impulse must not be capped by a minimum theta");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let loPoint=rowCenter\(lo\);let world=worldBoundaryBit\(h\)!=0u;var hiPoint=handleCenter\(h\);if\(hi!=INVALID\)\{hiPoint=rowCenter\(hi\);\}else if\(!world\)\{let inv=[\s\S]*hiPoint=loPoint\+handleNormal\(h\)\/max\(inv,1e-20\);\}/,
    "ghost-fluid theta must sample the two actual power-cell centres");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL,
    /let half=\.5\/max\(inv,1e-20\);let loPoint=x-half\*n;let hiPoint=x\+half\*n/,
    "a power-face centroid is not generally the midpoint of its dual edge");
  assert.match(structuredBoundaryCoefficientWGSL, /resolveStructuredSolidSlots/);
  assert.match(structuredBoundaryCoefficientWGSL, /commitStructuredBoundarySlots/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /let lo=handleOwner\(h\);let hi=handleNeighbor\(h\);let other=select\(lo,hi,lo==row\)/,
    "both incident rows must rebuild from the opposite canonical family owner");
  assert.match(structuredBoundaryCoefficientWGSL, /fn dynamicRowClass\(/);
  assert.match(structuredBoundaryCoefficientWGSL, /fn dynamicFamilyClass\(/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn rowTransition\(row:u32\)->bool\{let at=rbase\(\)\+row;[\s\S]*geometry\[at\]\.z!=0u;/,
    "dynamic worksets must classify from the banked accepted row geometry, not mutable candidate metrics");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let direction=vec3i\(round\(slot\.areaCentroid\.yzw\+\.5\*slot\.normalInverseDistance\.xyz\)\)/,
    "dynamic rows must use the same centroid-plus-half-normal channel map as the generated 19-channel catalog");
  assert.match(structuredBoundaryCoefficientWGSL,
    /var diagonalTerms:array<f32,31>[\s\S]*diagonalTerms\[local\]=coefficient[\s\S]*diagonalTerms\[j-1u\]<=value[\s\S]*sum\+=diagonalTerms\[i\]/,
    "the Section 6.3 diagonal must be independent of incident-slot permutation");
  assert.match(structuredBoundaryCoefficientWGSL,
    /scatterStructuredRowWorksets[\s\S]*dynamicWorksets\[worksetBase\(cls\)\+7u\+worksetBlocks\[cls\*worksetBlockStride\(\)\+wg\.x\]\+worksetBlockOffset\(lane\)\]=row/,
    "current free-surface coefficients must publish actual disjoint row and family worksets");
  // Family work is slot-shaped, so its block index folds through `foldedBlock`
  // while the row scatter above stays on `wg.x`. The prefix arithmetic is
  // otherwise identical, which is the property under test: both phases must
  // read the same block-ordered prefix table.
  assert.match(structuredBoundaryCoefficientWGSL,
    /scatterStructuredFamilyWorksets[\s\S]*dynamicWorksets\[worksetBase\(cls\)\+7u\+worksetBlocks\[cls\*worksetBlockStride\(\)\+foldedBlock\(wg\)\]\+worksetBlockOffset\(lane\)\]=h/,
    "family worksets must scatter from the same block-ordered prefix as the row worksets");
});

test("dynamic workset publication is wide and its destinations stay index-ordered", () => {
  // The destination of an element is the exclusive prefix of its class over
  // preceding blocks plus the count of lower lanes in its own block carrying
  // the same class. That is the element's rank in ascending index order, which
  // is exactly what the superseded single-workgroup kernel produced -- the
  // published worksets must be bit-identical, not merely equivalent.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn worksetBlockOffset\(lane:u32\)->u32\{let cls=worksetBlockClass\[lane\];var n=0u;for\(var j=0u;j<lane;j\+=1u\)/,
    "the intra-block offset must count lower lanes of the same class, never bump a shared cursor");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL, /publishStructuredBoundaryWorksets/,
    "the single-workgroup workset publication must not survive alongside the wide one");
  // Row classes are 0..4 and family classes 5..8, so the two phases can share
  // one block table only while each writes its own disjoint class range.
  assert.match(structuredBoundaryCoefficientWGSL,
    /countStructuredRowClasses[\s\S]*?if\(lane<5u\)\{worksetBlocks\[lane\*worksetBlockStride\(\)\+wg\.x\]/,
    "the row count must publish only classes 0..4 so it cannot clobber family blocks");
  // The family phase indexes its block table through `foldedBlock` while the
  // row phase above stays on `wg.x`. That asymmetry is deliberate: family work
  // is slot-shaped, so its block count can saturate one workgroup dimension and
  // its record is published two-dimensionally; row work is bounded by
  // rowCapacity and never folds. Both still write disjoint class ranges.
  assert.match(structuredBoundaryCoefficientWGSL,
    /countStructuredFamilyClasses[\s\S]*?if\(lane>=5u&&lane<9u\)\{worksetBlocks\[lane\*worksetBlockStride\(\)\+foldedBlock\(wg\)\]/,
    "the family count must publish only classes 5..8 so it cannot clobber row blocks");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let blocks=select\(rowBlocks,slotBlocks,cls>=5u\);[\s\S]*for\(var start=0u;start<blocks;start\+=256u\)/,
    "the ordered scan must visit live blocks rather than its capacity stride");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL,
    /for\(var start=0u;start<stride;start\+=256u\)/,
    "unused capacity blocks must not consume scan tiles");
  // A row's 19-channel class is the expensive part; it must be read from the
  // cache the row phase filled rather than recomputed per incident slot.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn cachedRowClass\(row:u32\)->u32\{return select\(dynamicRowClass\(row\),worksetClasses\[row\],row<control\.rows\);\}/,
    "family classification must read cached row classes for every in-range row");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL,
    /fn dynamicFamilyClass[^\n]*dynamicRowClass/,
    "dynamicFamilyClass must not recompute a row class it can read from the cache");
});

test("candidate-ready and recurring accepted controls have disjoint setup semantics", () => {
  assert.match(structuredBoundaryCoefficientWGSL,
    /prepareStructuredBoundaryCandidate\(\)[\s\S]*?rowCount=accepted\[2\][\s\S]*?slotCount=accepted\[3\][\s\S]*?generation=accepted\[4\][\s\S]*?bank=accepted\[5\][\s\S]*?accepted\[0\]==1398162764u/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /prepareStructuredBoundaryAccepted\(\)[\s\S]*?rowCount=accepted\[2\][\s\S]*?generation=accepted\[3\][\s\S]*?bank=accepted\[4\][\s\S]*?slotCount=accepted\[5\][\s\S]*?accepted\[0\]==0u/);
  assert.match(structuredBoundaryCoefficientWGSL,
    /rowCount<=p\.counts\.x&&slotCount<=p\.counts\.y&&generation!=0u&&bank<=1u/,
    "both authorities must validate counts, generation, and the published bank");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL,
    /accepted\[0\]==(?:1398162764u|0u)\s*\|\||\|\|\s*accepted\[0\]==(?:1398162764u|0u)/,
    "a setup entry point must not infer authority by accepting both layouts");
});

test("boundary classification uses authored SDF only for bootstrap and missing coarse publication", () => {
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(p\.pad\.z!=0u\)\{return vec2f\(analyticInitialPhi\(point\),1\.\);\}/,
    "the selected cold bootstrap must own its sample");
  assert.match(structuredBoundaryCoefficientWGSL,
    /return select\(vec2f\(1\.,0\.\),vec2f\(analyticInitialPhi\(point\),1\.\),authoredAnalyticPhiAvailable\(\)\);/,
    "an authored analytic scene must bridge a missing recurring coarse-directory publication");
  assert.match(structuredBoundaryCoefficientWGSL,
    /powerCoarseSamples\.generation&0x3fffffffu;let expected=control\.epoch&0x3fffffffu/,
    "recurring boundary publication must require the matching coarse generation");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let value=sampleCoarseOctreePhi\(point\);let valid=finite\(value\)&&value<3\.402823e38;return vec2f\(value,select\(0\.,1\.,valid\)\);/,
    "candidate rows must sample the identity-keyed coarse directory, not stale row-indexed values");
  assert.doesNotMatch(structuredBoundaryCoefficientWGSL, /coarse\[row\]/,
    "a newly inserted candidate row has no same-index accepted coarse value");
});

test("every structured boundary entry point fits the hard ten-storage-buffer contract", () => {
  for (const entryPoint of boundaryEntryPoints) {
    const bindings = reachableStorageBindings(entryPoint);
    assert.ok(bindings.length <= 10,
      `${entryPoint} reaches ${bindings.length} storage buffers: ${bindings.join(", ")}`);
  }
  assert.deepEqual(reachableStorageBindings("resolveStructuredBoundarySlots"),
    [2, 3, 4, 6, 7, 8, 9, 11, 16, 25]);
  assert.deepEqual(reachableStorageBindings("resolveStructuredSolidSlots"),
    [2, 3, 10, 11, 16, 19, 21]);
  assert.deepEqual(reachableStorageBindings("countStructuredRowClasses"),
    [3, 13, 14, 16, 27, 28],
    "class-4 dry proof reaches the accepted liquid mask at binding 13");
  const host = readFileSync(new URL("../lib/webgpu-octree-structured-boundary.ts", import.meta.url), "utf8");
  assert.match(host, /group\(this\.countRowClasses, \[0, 3, 13, 14, 16, 27, 28\]\)/,
    "the host bind group must include every storage binding reachable from class-4 proof");
});

test("the exact row carry is an opt-in uniform arm, never a shader-source variant", () => {
  assert.equal(structuredBoundaryExactRowCarryMode({}), 0,
    "the zero-dispatch row carry must default off so Gate A/B owns the flip");
  assert.equal(structuredBoundaryExactRowCarryEnabled({}), false);
  // 1 alternates, 2 chains. The two differ only in whether a carried
  // publication may itself be carried, which is the separation mask's staleness
  // bound and nothing else.
  assert.equal(structuredBoundaryExactRowCarryMode({
    FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY: "1",
  }), 1);
  assert.equal(structuredBoundaryExactRowCarryMode({
    FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY: "2",
  }), 2);
  assert.equal(structuredBoundaryExactRowCarryMode({
    FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY: "0",
  }), 0);
  assert.equal(structuredBoundaryExactRowCarryMode({
    FLUID_STRUCTURED_BOUNDARY_EXACT_ROW_CARRY: "3",
  }), 0, "an unrecognised mode must fail closed to off");
  // The arm lives in the tenth uniform vec4, not in the WGSL text, so both A/B
  // arms compile the identical module and derive the identical auto bind-group
  // layouts. An arm that changed the source could change a layout and would
  // stop being a controlled comparison.
  assert.match(structuredBoundaryCoefficientWGSL,
    /struct P\{counts:vec4u,[^}]*damDimensions:vec4f,carry:vec4u\}/,
    "the carry arm selector must be a uniform word");
  const host = readFileSync(new URL("../lib/webgpu-octree-structured-boundary.ts", import.meta.url), "utf8");
  assert.match(host, /size: 160,\s*\n\s*usage: GPUBufferUsage\.UNIFORM/,
    "the parameter buffer must cover the tenth vec4");
  assert.match(host, /const words = new Uint32Array\(40\)/);
  assert.match(host, /words\[36\] = structuredBoundaryExactRowCarryMode\(\);/,
    "carry.x is uniform word 36");
});

test("the boundary carry gate proves every input, not just the retrospective receipt", () => {
  // The receipt alone is retrospective: it says the liquid mask was unchanged
  // LAST step. These are the forward conjuncts that make a skip exact.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn boundaryExactRowCarryEligible\(eligible:bool\)->bool\{\s*if\(p\.carry\.x==0u\|\|!eligible\)\{return false;\}\s*let receipt=atomicLoad\(&acceptedBoundary\.pad\);\s*if\(\(receipt&BOUNDARY_RECEIPT_IDENTITY\)==0u\)\{return false;\}\s*if\(p\.carry\.x==1u&&\(receipt&BOUNDARY_CARRIED\)!=0u\)\{return false;\}\s*if\(p\.pad\.z!=0u\|\|fp\.width>0\.\|\|p\.dimensions\.w!=0u\)\{return false;\}\s*return boundaryCoarsePhiIdentity\(\);\}/,
    "the carry must require the receipt, a post-bootstrap phi, no fine band and no rigid bodies");
  // The separation mask is the one conjunct that does not chain: a carried step
  // never runs the resolve that observes it. Mode 1 refuses to carry a carried
  // publication, which is what bounds that staleness at a single step. The
  // receipt therefore has to publish the carry marker, not collapse it away.
  assert.match(structuredBoundaryCoefficientWGSL,
    /let pad=atomicLoad\(&control\.pad\);let liquidIdentity=select\(0u,BOUNDARY_RECEIPT_IDENTITY,\(pad&BOUNDARY_DIRTY\)==0u\)\|\(pad&BOUNDARY_CARRIED\);/,
    "the accepted receipt must carry both the identity proof and whether it was itself carried");
  // The phi-side signal. `count == 0` under a valid publication is the coarse
  // publisher's own bitwise proof that no live sample's phi, minimum, maximum
  // or interface/corrected phase moved and that no row was inserted or retired.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn boundaryCoarsePhiIdentity\(\)->bool\{[\s\S]*powerCoarseSamples\.state!=COARSE_PHI_VALID[\s\S]*powerCoarseSamples\.generation&COARSE_PHI_DENSE\)!=0u[\s\S]*coarsePhiDelta\[3\]==COARSE_PHI_VALID\s*&&coarsePhiDelta\[2\]==0u&&coarsePhiDelta\[0\]==0u/,
    "the carry must consume the coarse exact value/phase delta and exclude the uncovered dense complement");
  // `accepted[15]` is the structured publication's own exact row-delta identity
  // bit. It is what certifies the row/slot geometry AND that the family scatter
  // did not reset the fractions/pressureScales this boundary owns.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn boundaryLiquidIdentityEligible\(\)->bool\{return \(atomicLoad\(&control\.pad\)&BOUNDARY_DIRTY\)==0u\s*&&arrayLength\(&accepted\)>15u&&accepted\[15u\]!=0u/,
    "the carry precondition must read the structured identity carry bit and mask only the dirty channel");
  // A live separation mark ages against control.epoch, so an unchanged mask
  // still flips an aperture. Observing one withdraws the next step's carry.
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(p\.carry\.x!=0u&&separationMarked\(lo,boundaryBit\)\)\{atomicOr\(&control\.pad,BOUNDARY_DIRTY\);\}\s*aperture=select\(0\.,1\.,separationFresh\(lo,boundaryBit\)\);/,
    "a live separation mark of any age must publish the dirty bit, and only in the enabled arm");
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn separationMarked\(row:u32,faceBit:u32\)->bool\{[\s\S]*return \(separationMask\[cell\]&faceBit\)!=0u;/,
    "the mark observation must ignore the age the fresh test applies");
  // The receipt keeps its meaning: bit 0 is the dirty channel, bit 1 only
  // records that this publication carried.
  assert.equal([...structuredBoundaryCoefficientWGSL
    .matchAll(/atomicOr\(&control\.pad,(\w+)\)/g)].every((match) => match[1] === "BOUNDARY_DIRTY"), true,
    "every dirty publisher must or the dirty bit alone");
});

test("a carried boundary publishes two exact zero records and still advances publication", () => {
  // publishStructuredBoundarySetup floors the slot record's X at one, so the
  // carry needs its own exact publisher rather than a zero count pushed through
  // that arithmetic.
  assert.match(structuredBoundaryCoefficientWGSL,
    /fn publishExactBoundaryDispatch\(at:u32,blocks:u32\)\{if\(blocks==0u\)\{dispatch\[at\]=0u;dispatch\[at\+1u\]=1u;dispatch\[at\+2u\]=1u;return;\}/,
    "the carry must publish a genuine (0,1,1)");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let slotX=max\(1u,min\(65535u,slotBlocks\)\)/,
    "the ordinary setup keeps its floored slot record, which is why the carry cannot reuse it");
  // Trap: `rebuild` is what normally advances `published`, and it launches zero
  // workgroups under the carry. Without this, worksetPublicationEnabled and
  // acceptStructuredBoundary both fail closed and the boundary never publishes.
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(carry\)\{publishExactBoundaryDispatch\(0u,0u\);publishExactBoundaryDispatch\(3u,0u\);control\.published=control\.epoch;for\(var b=0u;b<8u;b\+=1u\)\{atomicStore\(&control\.theta\[b\],atomicLoad\(&acceptedBoundary\.theta\[b\]\)\);\}\}/,
    "the carry must zero both records, advance publication itself, and carry the theta histogram");
  assert.match(structuredBoundaryCoefficientWGSL,
    /atomicStore\(&control\.pad,select\(select\(BOUNDARY_DIRTY,0u,eligible\),BOUNDARY_CARRIED,carry\)\);/,
    "a carried publication must mark itself so the block scan can tell it from an ordinary clean step");
});

test("the carried workset scan re-stamps the retained headers instead of re-prefixing them", () => {
  // `scanStructuredWorksetBlocks` is the one direct dispatch in the publication,
  // so it still launches when both indirect records are zero -- and it rewrites
  // worksetBlocks in place from counts to exclusive prefixes. Re-running it over
  // an already-prefixed table would prefix it twice.
  assert.match(structuredBoundaryCoefficientWGSL,
    /worksetGate=select\(0u,select\(1u,2u,boundaryRowCarryActive\(\)\),worksetPublicationEnabled\(\)\);/,
    "the scan must distinguish a carried publication from an ordinary one");
  assert.match(structuredBoundaryCoefficientWGSL,
    /let worksetMode=workgroupUniformLoad\(&worksetGate\);\s*if\(worksetMode==0u\)\{return;\}/,
    "the mode must be workgroup-uniform so both early returns keep every lane at every barrier");
  // Exactly the fields the persistent solver's validateAcceptedRowPartition
  // reads. A carry that advanced the epoch word but left a count or a dispatch
  // triple inconsistent would be rejected there with 0x10x/0x200 instead.
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(worksetMode==2u\)\{[\s\S]*dynamicWorksets\[base\]==acceptedEpoch&&n<=dynamicWorksets\[base\+2u\][\s\S]*\(dynamicWorksets\[base\+3u\]&3u\)==3u&&dynamicWorksets\[base\+4u\]==dx[\s\S]*dynamicWorksets\[base\+5u\]==\(blocks\+dx-1u\)\/dx&&dynamicWorksets\[base\+6u\]==1u[\s\S]*valid=valid&&rowTotal==control\.rows;/,
    "the carry arm must revalidate the epoch stamp, the exact dispatch triple and the class 0..4 partition total");
  assert.match(structuredBoundaryCoefficientWGSL,
    /if\(valid\)\{for\(var cls=0u;cls<9u;cls\+=1u\)\{dynamicWorksets\[worksetBase\(cls\)\]=control\.epoch;\}\}\s*else\{fail\(0u,ERROR_CARRY\);\}/,
    "nothing may be stamped until all nine classes validate, so a rejected carry leaves the accepted publication intact");
  // Extract the carry arm by brace balance rather than by regex: the property
  // is that the whole workgroup leaves through it together, so it may contain
  // no barrier of its own and the scan's barriers must all be downstream.
  const armStart = structuredBoundaryCoefficientWGSL.indexOf("if(worksetMode==2u){");
  assert.notEqual(armStart, -1);
  let cursor = structuredBoundaryCoefficientWGSL.indexOf("{", armStart), depth = 0, armEnd = -1;
  for (let at = cursor; at < structuredBoundaryCoefficientWGSL.length; at += 1) {
    if (structuredBoundaryCoefficientWGSL[at] === "{") depth += 1;
    else if (structuredBoundaryCoefficientWGSL[at] === "}" && --depth === 0) { armEnd = at; break; }
  }
  assert.ok(armEnd > armStart);
  const arm = structuredBoundaryCoefficientWGSL.slice(armStart, armEnd + 1);
  assert.doesNotMatch(arm, /workgroupBarrier\(\)/,
    "the carry arm must return before the scan's barriers, never take one of its own");
  assert.match(arm, /return;/, "the carry arm must leave the kernel, not fall through into the scan");
  assert.doesNotMatch(arm, /worksetScan|worksetBlocks/,
    "the carry arm must not touch the block prefix table it would otherwise re-prefix");
});

test("the carry gate reaches only the bindings it adds, and only where it is evaluated", () => {
  // Trap: resolve/resolveSolid already sit at Metal's ten-storage ceiling. The
  // gate's new binding must be reachable from the one-lane prepare and nowhere
  // else.
  assert.deepEqual(reachableStorageBindings("prepareStructuredBoundaryCandidate"),
    [1, 4, 16, 17, 22, 29]);
  assert.deepEqual(reachableStorageBindings("prepareStructuredBoundaryAccepted"), [1, 16, 17],
    "the recurring accepted authority has no carry arm and must not widen");
  assert.deepEqual(reachableStorageBindings("scanStructuredWorksetBlocks"), [16, 18, 22, 28],
    "the block scan reads the accepted boundary only to revalidate a carried publication");
  for (const entryPoint of boundaryEntryPoints) {
    assert.ok(!reachableStorageBindings(entryPoint).includes(29)
      || entryPoint === "prepareStructuredBoundaryCandidate",
      `${entryPoint} must not reach the coarse delta receipt`);
  }
  const host = readFileSync(new URL("../lib/webgpu-octree-structured-boundary.ts", import.meta.url), "utf8");
  assert.match(host, /\[0, 1, 4, 5, 16, 17, 22, 29\] : \[0, 1, 16, 17\]/,
    "the host prepare groups must match the two entry points' reachable sets");
  assert.match(host, /group\(this\.scanWorksetBlocks, \[0, 16, 18, 22, 28\]\)/);
  assert.match(host, /29: this\.resources\.coarse\.delta/);
});

test("Dawn Metal compiles transactional structured boundary update", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const module = device.createShaderModule({ code: structuredBoundaryCoefficientWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter((message) => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of boundaryEntryPoints) {
    device.createComputePipeline({ layout: "auto", compute: { module, entryPoint } });
  }
  const error = await device.popErrorScope(); assert.equal(error, null, error?.message);
  device.destroy();
});
