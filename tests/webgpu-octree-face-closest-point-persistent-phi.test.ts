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

test("band-phi relaxation retains the exact sparse parallel Jacobi row kernel", () => {
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
    /count=atomicLoad\(&bandPhiFrontier\[0\]\).*rowIndex=atomicLoad\(&bandPhiFrontier\[item\+1u\]\).*provisionalVelocities\[rowIndex\]=extendBandRowPhi\(rowIndex\)/s,
    "dependency waves consume the GPU-built mutable-row frontier instead of the full arena");
});

test("band-phi rounds preserve sparse parallel dispatch and cached ping-pong bindings", () => {
  const constructor = compact(WebGPUOctreeFaceClosestPointExtension);
  assert.match(constructor,
    /extendBandPhi:pipeline\("extendBandRowPhiParallel"\)/);
  assert.match(constructor,
    /prepareBandPhiFrontier:pipeline\("prepareBandPhiActiveRows"\).*collectBandPhiFrontier:pipeline\("collectBandPhiActiveRows"\).*publishBandPhiFrontier:pipeline\("publishBandPhiActiveRows"\)/s);

  const encode = compact(WebGPUOctreeFaceClosestPointExtension.prototype.encodePhase);
  const start = encode.indexOf('run("initializeBandPhi"');
  const end = encode.indexOf('run("sampleFaceCoarsePhi"', start);
  const bandPhi = encode.slice(start, end);
  assert.equal(bandPhi.match(/run\("extendBandPhi"/g)?.length, 1);
  assert.match(bandPhi,
    /run\("prepareBandPhiFrontier".*run\("collectBandPhiFrontier".*run\("publishBandPhiFrontier".*for\(letround=0;round<this\.bandPhiRelaxationRounds;round\+=1\).*run\("extendBandPhi".*\[19,currentPhi\].*\[44,nextPhi\].*\[74,this\.bandPhiFrontier\].*\[currentPhi,nextPhi\]=\[nextPhi,currentPhi\]/s);
  assert.match(bandPhi, /run\("extendBandPhi"[\s\S]*0,pass,252\)/,
    "the mutable frontier owns a dead indirect triplet and cannot truncate the shared row-prefix dispatch");
  assert.doesNotMatch(wgslFunction("publishBandPhiActiveRows"), /indirect\[5[4-6]\]/,
    "frontier publication must not overwrite indirect slot 54 used by later immutable-row consumers");
  assert.match(bandPhi,
    /run\("commitBandPhi",\[\[5,this\.candidateControl\],\[6,this\.candidateRows\],\[19,currentPhi\],\[32,this\.candidateTransitionControl\]\]/,
    "the exact odd/even Jacobi result is committed without a copy");
});
