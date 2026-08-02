import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { planOctreePressureCapacity } from "../lib/webgpu-octree";
import {
  OCTREE_PERSISTENT_MGPCG_ROW_THRESHOLD,
  WebGPUOctreePersistentMGPCG,
} from "../lib/webgpu-octree-persistent-mgpcg";
import {
  OCTREE_PERSISTENT_MGPCG_CHANNEL_COUNT,
  OCTREE_PERSISTENT_MGPCG_HEADER,
  OCTREE_PERSISTENT_MGPCG_PARTIAL_WORDS,
  OCTREE_PERSISTENT_MGPCG_REDUCTION_LANES,
  OCTREE_PERSISTENT_MGPCG_WORKGROUP_BYTES,
  octreePersistentMGPCGArenaWords,
  octreePersistentMGPCGWGSL,
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
