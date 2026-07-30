import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { structuredBoundaryCoefficientWGSL } from "../lib/webgpu-octree-structured-boundary";

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
    /let lattice=\(x-fp\.origin\)\/fp\.width-vec3f\(\.5\)[\s\S]*for\(var corner=0u;corner<8u;corner\+=1u\)[\s\S]*if\(sample\.y==0\.\)\{return vec2f\(coarsePhi,0\.\);\}/,
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
  // Row classes are 0..3 and family classes 5..8, so the two phases can share
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
