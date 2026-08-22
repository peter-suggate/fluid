import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12FrameControl } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control";
import { createSparseCM12FrameControlWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-control.wgsl";
import { createSparseCM12VelocityExtensionLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-velocity-extension";
import {
  SPARSE_CM12_VEX_DELTA_AUTHORITY_FAULT as F,
  createSparseCM12VexDeltaAuthorityInitialWords,
  createSparseCM12VexDeltaAuthorityLayout,
  preflightSparseCM12VexDelta,
  sparseCM12VexDeltaRetiredCoverageHash,
  sparseCM12VexDeltaRootCoverageHash,
} from "../lib/methods/adaptive-mass/sparse-cm12-vex-delta-authority";
import { createSparseCM12VexDeltaAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-delta-authority.wgsl";

test("VDA1 is an optional delta-sized allocation and leaves baseline FCA bytes unchanged", () => {
  const baseline = createSparseCM12FrameControl({ cellWorkgroups: 7,
    rowWorkgroups: 9, d4Capable: true });
  const again = createSparseCM12FrameControl({ cellWorkgroups: 7,
    rowWorkgroups: 9, d4Capable: true });
  assert.deepEqual([...baseline.words], [...again.words]);
  const layout = createSparseCM12VexDeltaAuthorityLayout({ baseWords: 128,
    cellCapacity: 64 });
  assert.equal(layout.authorityWords, 32);
  const words = createSparseCM12VexDeltaAuthorityInitialWords(layout);
  assert.equal(words.length, layout.authorityWords);
});

test("VDA1 preflight admits repeated exact incidences and reserves only unique new cells", () => {
  const roots = [{ cell: 5, cause: 8 }, { cell: 2, cause: 8 },
    { cell: 5, cause: 8 }, { cell: 9, cause: 8 }];
  const stamp = new Uint32Array(16);stamp[2] = 11;
  const result = preflightSparseCM12VexDelta({ transactionGeneration: 4,
    topologyGeneration: 4, expectedTopologyGeneration: 4,
    vexGeneration: 11, expectedVexGeneration: 11, vexCollecting: true,
    cellCapacity: 16, vexRootCount: 3, vexRootCapacity: 16, rootStamp: stamp,
    roots, retiredCells: [12, 7, 12], finalCellActive: (cell) => cell < 10,
    finalCellRetired: (cell) => cell >= 10 || cell === 7 });
  assert.equal(result.authorized, true);
  assert.equal(result.existingRootCount, 1);
  assert.equal(result.newRootCount, 2);
  assert.equal(result.reservedRootBase, 3);
  assert.deepEqual([...result.rootSlots], [0, 0xffff_ffff, 0xffff_ffff, 1]);
  assert.equal(sparseCM12VexDeltaRootCoverageHash(roots),
    sparseCM12VexDeltaRootCoverageHash([...roots].reverse()));
  assert.equal(sparseCM12VexDeltaRetiredCoverageHash([12, 7, 12]),
    sparseCM12VexDeltaRetiredCoverageHash([12, 12, 7]));
});

test("VDA1 rejects stale, incomplete, inactive and mixed-cause batches before publication", () => {
  const base = { transactionGeneration: 4, topologyGeneration: 4,
    expectedTopologyGeneration: 4, vexGeneration: 11, expectedVexGeneration: 11,
    vexCollecting: true, cellCapacity: 16, vexRootCount: 0, vexRootCapacity: 16,
    rootStamp: new Uint32Array(16), roots: [{ cell: 3, cause: 8 }],
    retiredCells: [] as number[], finalCellActive: (cell: number) => cell < 10,
    finalCellRetired: (cell: number) => cell >= 10 };
  assert.equal(preflightSparseCM12VexDelta({ ...base, topologyGeneration: 3 }).fault,
    F.staleTopologyGeneration);
  assert.equal(preflightSparseCM12VexDelta({ ...base, rootCoverageCount: 2 }).fault,
    F.missingCoverage);
  assert.equal(preflightSparseCM12VexDelta({ ...base,
    roots: [{ cell: 13, cause: 8 }] }).fault, F.invalidRoot);
  assert.equal(preflightSparseCM12VexDelta({ ...base,
    roots: [{ cell: 3, cause: 8 }, { cell: 3, cause: 16 }] }).fault,
  F.invalidRoot);
  assert.equal(preflightSparseCM12VexDelta({ ...base, rootBatch0Count: 1,
    roots: [{ cell: 3, cause: 8 }, { cell: 3, cause: 16 }] }).authorized, true);
});

test("VDA1 WGSL keeps every failure before the authorized no-fail seams", () => {
  const velocity = createSparseCM12VelocityExtensionLayout({ baseWords: 4096,
    cellCapacity: 64 });
  const layout = createSparseCM12VexDeltaAuthorityLayout({ baseWords: 0,
    cellCapacity: 64 });
  const wgsl = createSparseCM12VexDeltaAuthorityWGSL({ layout,
    velocityExtensionLayout: velocity, finalCellActiveFunction: "candidateActive",
    finalCellRetiredFunction: "candidateRetired",
    topologyGenerationExpression: "candidateTopologyGeneration()",
    frameGenerationExpression: "candidateFrameGeneration()",
    framePhaseValidExpression: "candidateFramePhaseValid()" });
  const root = wgsl.slice(wgsl.indexOf("fn vda1PublishAuthorizedRoot"),
    wgsl.indexOf("fn vda1PublishAuthorizedRetirement"));
  const retired = wgsl.slice(wgsl.indexOf("fn vda1PublishAuthorizedRetirement"),
    wgsl.indexOf("fn vda1SealPublicationNoFail"));
  for (const source of [root, retired]) {
    assert.doesNotMatch(source, /(?:cm12Extension|vda1)Fail|compareExchange|for\(|loop\{/i);
  }
  assert.match(root, /atomicExchange/);
  assert.match(wgsl, /fn vda1TransactionSucceeded/);
  assert.doesNotMatch(wgsl, /global fallback|world scan|incidenceBegin/i);
});

test("candidate FCA exposes a no-fail D4 seam only when explicitly enabled", () => {
  const control = createSparseCM12FrameControl({ cellWorkgroups: 7,
    rowWorkgroups: 9, d4Capable: true });
  const baseline = createSparseCM12FrameControlWGSL({ layout: control.layout });
  const candidate = createSparseCM12FrameControlWGSL({ layout: control.layout,
    authorizedD4Invalidation: true });
  assert.doesNotMatch(baseline, /cm12FCInvalidateD4Authorized/);
  const start = candidate.indexOf("fn cm12FCInvalidateD4Authorized");
  assert.ok(start >= 0);
  const helper = candidate.slice(start);
  assert.doesNotMatch(helper, /cm12FCFail|CompareExchange|capacity|return false/i);
});
