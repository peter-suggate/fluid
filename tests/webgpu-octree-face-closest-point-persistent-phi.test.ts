import assert from "node:assert/strict";
import test from "node:test";

import {
  WebGPUOctreeFaceClosestPointExtension,
  octreeFaceBandWGSL,
} from "../lib/webgpu-octree-face-closest-point";

const compact = (source: { toString(): string }): string => source.toString().replace(/\s+/g, "");

function wgslFunction(name: string): string {
  const source = compact(octreeFaceBandWGSL);
  const start = source.indexOf(`fn${name}(`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let cursor = open; cursor < source.length; cursor += 1) {
    if (source[cursor] === "{") depth += 1;
    else if (source[cursor] === "}" && --depth === 0) return source.slice(start, cursor + 1);
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

test("band-phi relaxation retains the exact canonical parallel Jacobi row kernel", () => {
  const shader = compact(octreeFaceBandWGSL);

  const perRow = wgslFunction("extendBandRowPhi");
  assert.doesNotMatch(perRow, /@builtin|global_invocation_id|provisionalVelocities\[rowIndex\]=/,
    "the exact per-row Eikonal body remains independently testable");
  assert.match(perRow,
    /lowerSimplexCandidate\(rowIndex,sign\).*localTetraEikonal\(rowIndex,sign\)/s);

  assert.match(shader,
    /@compute@workgroup_size\(64\)fnextendBandRowPhiParallel\(@builtin\(global_invocation_id\)g:vec3u\)/);
  const parallel = wgslFunction("extendBandRowPhiParallel");
  assert.match(parallel,
    /count=atomicLoad\(&bandPhiFrontier\[0\]\).*rowIndex=atomicLoad\(&bandPhiFrontier\[item\+1u\]\).*rowIndex!=INVALID.*provisionalVelocities\[rowIndex\]=extendBandRowPhi\(rowIndex\)/s,
    "dependency waves consume the canonical live-row map and skip immutable rows");
});

test("band-phi rounds preserve canonical parallel dispatch and cached ping-pong bindings", () => {
  const constructor = compact(WebGPUOctreeFaceClosestPointExtension);
  assert.match(constructor,
    /extendBandPhi:pipeline\("extendBandRowPhiParallel"\)/);
  assert.match(constructor,
    /collectBandPhiFrontier:pipeline\("collectBandPhiActiveRows"\)/);
  assert.doesNotMatch(constructor, /prepareBandPhiActiveRows|publishBandPhiActiveRows/,
    "the canonical live-row map needs no singleton preparation or publication launch");

  const encode = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const start = encode.indexOf('run("initializeBandPhi"');
  const end = encode.indexOf('run("sampleFaceCoarsePhi"', start);
  const bandPhi = encode.slice(start, end);
  assert.equal(bandPhi.match(/run\("extendBandPhi"/g)?.length, 1);
  assert.match(bandPhi,
    /run\("collectBandPhiFrontier".*\[18,this\.indirect\].*\[74,this\.bandPhiFrontier\].*for\(letround=0;round<this\.bandPhiRelaxationRounds;round\+=1\).*run\("extendBandPhi".*\[19,currentPhi\].*\[44,nextPhi\].*\[74,this\.bandPhiFrontier\].*\[currentPhi,nextPhi\]=\[nextPhi,currentPhi\]/s);
  assert.match(bandPhi, /run\("extendBandPhi"[\s\S]*0,pass,252\)/,
    "the live-row map owns a private indirect triplet and cannot truncate the shared row-prefix dispatch");
  const collect = wgslFunction("collectBandPhiActiveRows");
  assert.match(collect,
    /bandPhiFrontier\[0\],control\.rowCount.*indirect\[63\]=\(control\.rowCount\+63u\)\/64u.*bandPhiFrontier\[rowIndex\+1u\].*select\(rowIndex,INVALID,state\.w==0\.\|\|state\.z>0\.\)/s,
    "one parallel pass publishes stable row identity and the private indirect dispatch");
  assert.doesNotMatch(collect, /atomicAdd/,
    "frontier publication has no contended append or nondeterministic row order");
  assert.match(bandPhi,
    /run\("commitBandPhi",\[\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[19,currentPhi\],\[32,this\.candidateTransitionControl\]\]/,
    "the exact odd/even Jacobi result is committed without a copy");
});

test("one canonical live-face package drives every downstream face stage", () => {
  const publish = wgslFunction("publishFaceBandCounts");
  assert.match(publish,
    /topologyPublishLaneRange\(count,lane\).*topologyPublishOffsets\[sourceLane\]=offset.*liveFaceWorklist\[output\+1u\]=faceIndex.*liveFaceWorklist\[0\]=totals\.x.*indirect\[57\]=\(totals\.x\+63u\)\/64u/s,
    "the parallel contiguous-lane prefix also emits the canonical face package");
  assert.match(octreeFaceBandWGSL,
    /@compute @workgroup_size\(256\)fn publishFaceBandCounts/,
    "face publication must not return to a capacity-sized singleton walk");

  for (const name of ["sampleBandFacePhi", "sampleBandFaceCoarsePhi", "seedFaceCentroids",
    "extendFaceClosestPoints", "gatherDryFaceClosestPointRepairs",
    "commitDryFaceClosestPointRepairs"]) {
    assert.match(wgslFunction(name), /liveFaceIndex\(g\.x\)/,
      `${name} must map dense work-item identity to its immutable face slot`);
  }
  assert.match(wgslFunction("reduceBandPhiFailure"),
    /topologyPublishLaneRange\(liveCount,lane\).*liveFaceIndex\(item\).*topologyPublishSums\[lane\].*lane!=0u.*phiFailureCounts=reduced\.x/s,
    "diagnostics parallel-reduce the same canonical package without a capacity scan");
  assert.match(wgslFunction("finalizeFaceBandClosestPointDiagnostics"),
    /item=lane;item<faceCount;item\+=256u.*liveFaceIndex\(item\).*publicationReductions\[22\].*if\(lane!=0u\)\{return;\}/s,
    "the terminal publication reduction consumes the shared package in parallel");

  const encode = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  for (const stage of ["publishFaceCounts", "sampleFacePhi", "sampleFaceCoarsePhi",
    "reducePhiFailure", "seedCentroids", "extendClosestPoints",
    "gatherClosestPointRepairs", "commitClosestPointRepairs", "finalizeClosestPoint"]) {
    const start = encode.indexOf(`run("${stage}"`);
    assert.notEqual(start, -1);
    const end = encode.indexOf("]],", start);
    assert.match(encode.slice(start, end), /\[75,this\.liveFaceWorklist/,
      `${stage} must receive the canonical live-face package`);
  }
});
