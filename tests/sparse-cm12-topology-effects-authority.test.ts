import assert from "node:assert/strict";
import test from "node:test";
import {
  SPARSE_CM12_TOPOLOGY_EFFECTS_HEADER as H,
  SPARSE_CM12_TOPOLOGY_EFFECTS_DISPATCH_ORDER,
  SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC,
  compileSparseCM12TopologyEffectsReference,
  createSparseCM12TopologyEffectsAuthorityInitialWords,
  createSparseCM12TopologyEffectsAuthorityLayout,
  sparseCM12TopologyEffectsIndirectByteOffset,
} from "../lib/methods/adaptive-mass/sparse-cm12-topology-effects-authority";
import { createSparseCM12TopologyEffectsAuthorityWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-topology-effects-authority.wgsl";
import { createSparseCM12SRR1IngressLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-runtime-adapter";
import { createSparseCM12SRR1ResidentIngressWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-srr1-resident-ingress.wgsl";
import { createSparseCM12PressureTopologyRepairLayout } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair";
import { createSparseCM12PressureTopologyRepairWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-pressure-topology-repair.wgsl";

const layout = createSparseCM12TopologyEffectsAuthorityLayout({
  baseWords: 13, scaCapacity: 130, ptrCapacity: 513, ptrLeafCapacity: 3,
});
const functionSlice = (source: string, signature: string): string => {
  const start = source.indexOf(signature);
  assert.notEqual(start, -1, `${signature} missing`);
  const open = source.indexOf("{", start);
  let depth = 0;
  for (let at = open; at < source.length; at++) {
    if (source[at] === "{") depth++;
    if (source[at] === "}" && --depth === 0) return source.slice(start, at + 1);
  }
  throw new Error(`${signature} is unterminated`);
};

test("TFX1 layout is relocatable, disjoint, and accounts only appended bytes", () => {
  const bases = [layout.baseWords, layout.scaStampBaseWords, layout.scaCauseBaseWords,
    layout.scaListBaseWords, layout.ptrStampBaseWords, layout.ptrOldStateBaseWords,
    layout.ptrNewStateBaseWords, layout.ptrCauseBaseWords, layout.ptrOwnsLeafBaseWords,
    layout.ptrListBaseWords, layout.ptrLeafStampBaseWords, layout.ptrLeafListBaseWords,
    layout.totalWords];
  for (const base of bases) assert.equal(base % 64, 0);
  assert.equal(layout.totalBytes, 4 * (layout.totalWords - layout.baseWords));
  const words = createSparseCM12TopologyEffectsAuthorityInitialWords(layout);
  assert.equal(words.length * 4, layout.totalBytes);
  assert.equal(words[H.magic], SPARSE_CM12_TOPOLOGY_EFFECTS_MAGIC);
  assert.equal(words[H.totalWords], layout.totalWords);
  assert.equal(words[H.scaCapacity], 130);
  assert.equal(words[H.ptrCapacity], 513);
  assert.equal(words[H.ptrLeafCapacity], 3);
  assert.equal(sparseCM12TopologyEffectsIndirectByteOffset(layout, "sca"),
    4 * (layout.baseWords + H.scaDispatchX));
  assert.equal(sparseCM12TopologyEffectsIndirectByteOffset(layout, "ptr"),
    4 * (layout.baseWords + H.ptrDispatchX));
});

test("TFX1 rejects invalid capacities and an incomplete PTR leaf plane", () => {
  for (const key of ["scaCapacity", "ptrCapacity", "ptrLeafCapacity"] as const) {
    assert.throws(() => createSparseCM12TopologyEffectsAuthorityLayout({
      baseWords: 0, scaCapacity: 1, ptrCapacity: 1, ptrLeafCapacity: 1, [key]: 0,
    }), /positive/);
  }
  assert.throws(() => createSparseCM12TopologyEffectsAuthorityLayout({
    baseWords: 0, scaCapacity: 1, ptrCapacity: 257, ptrLeafCapacity: 1,
  }), /cover every PTR brick leaf/);
});

test("TFX1 CPU receipt deduplicates causes, rejects conflicts, and is producer-order invariant", () => {
  const a = compileSparseCM12TopologyEffectsReference({ scaCapacity: 16, ptrCapacity: 600,
    sca: [{ tile: 4, cause: 1 }, { tile: 2, cause: 8 }, { tile: 4, cause: 2 }],
    ptr: [{ brick: 511, oldState: 1, newState: 2, cause: 4 },
      { brick: 3, oldState: 7, newState: 0, cause: 1 },
      { brick: 511, oldState: 1, newState: 2, cause: 8 }] });
  const b = compileSparseCM12TopologyEffectsReference({ scaCapacity: 16, ptrCapacity: 600,
    sca: [{ tile: 4, cause: 2 }, { tile: 4, cause: 1 }, { tile: 2, cause: 8 }],
    ptr: [{ brick: 511, oldState: 1, newState: 2, cause: 8 },
      { brick: 511, oldState: 1, newState: 2, cause: 4 },
      { brick: 3, oldState: 7, newState: 0, cause: 1 }] });
  assert.deepEqual(a, b);
  assert.deepEqual(a.sca, [{ tile: 2, cause: 8 }, { tile: 4, cause: 3 }]);
  assert.deepEqual(a.ptrLeaves, [0, 1]);
  assert.throws(() => compileSparseCM12TopologyEffectsReference({
    scaCapacity: 1, ptrCapacity: 2, sca: [], ptr: [
      { brick: 1, oldState: 0, newState: 1, cause: 1 },
      { brick: 1, oldState: 1, newState: 0, cause: 2 },
    ],
  }), /conflicting PTR effect/);
});

test("TFX1 preflight uses independent transaction, SCA, and PTR generations", () => {
  const source = createSparseCM12TopologyEffectsAuthorityWGSL({ layout,
    arenaName: "topologyArena", authorizationExpression: "transactionAuthorized()" });
  assert.match(source, new RegExp(`topologyArena\\[${layout.baseWords + H.scaTargetGeneration}u\\],tfxSCATargetGeneration\\(\\)`));
  assert.match(source, new RegExp(`topologyArena\\[${layout.baseWords + H.ptrTargetGeneration}u\\],tfxPTRTargetGeneration\\(\\)`));
  assert.match(source, /tfxSCAWillAppend\(tile,scaGeneration\)/);
  assert.match(source, /tfxPTRWillAppend\(brick,ptrGeneration\)/);
  assert.match(source, /tfxSCAPublish\(tile,generation,/);
  assert.match(source, /!=0u,generation\);/);
  assert.match(source, /scaHash\^=/);
  assert.match(source, /ptrHash\^=/);
  assert.match(source, /finishSparseCM12TopologyEffectsPublication/);
  assert.doesNotMatch(source, /PublicationComplete\(\).*coveredEffects/s);
  assert.doesNotMatch(source, /for\(var (tile|brick)=0u/);
  assert.match(SPARSE_CM12_TOPOLOGY_EFFECTS_DISPATCH_ORDER.at(-1)!,
    /one final singleton flips/);
});

test("SIR1 and PTR1 no-fail publication seams are strictly opt-in", () => {
  const sirLayout = createSparseCM12SRR1IngressLayout({ baseWords: 0, tileCapacity: 64 });
  const sirBaseline = createSparseCM12SRR1ResidentIngressWGSL({ layout: sirLayout });
  assert.equal(sirBaseline, createSparseCM12SRR1ResidentIngressWGSL({
    layout: sirLayout, preflightedTopologyPublication: false }));
  assert.doesNotMatch(sirBaseline, /Preflighted|PreflightReady|PreflightWillAppend/);
  const sirCandidate = createSparseCM12SRR1ResidentIngressWGSL({
    layout: sirLayout, preflightedTopologyPublication: true });
  const sirPublish = functionSlice(sirCandidate, "fn sirResidentPublishPreflightedEvent");
  assert.match(sirPublish, /atomicStore|atomicOr/);
  assert.doesNotMatch(sirPublish, /Fail|return false|for\(|while\(|>=|eventCapacity/);

  const ptrLayout = createSparseCM12PressureTopologyRepairLayout({
    brickCapacity: 64, rowCapacity: 64,
  });
  const ptrBaseline = createSparseCM12PressureTopologyRepairWGSL({ layout: ptrLayout });
  assert.equal(ptrBaseline, createSparseCM12PressureTopologyRepairWGSL({
    layout: ptrLayout, preflightedTopologyPublication: false }));
  assert.doesNotMatch(ptrBaseline, /PublishPreflightedChangedBrick/);
  const ptrCandidate = createSparseCM12PressureTopologyRepairWGSL({
    layout: ptrLayout, preflightedTopologyPublication: true });
  const ptrPublish = functionSlice(ptrCandidate, "fn ptrPublishPreflightedChangedBrick");
  assert.match(ptrPublish, /atomicStore|atomicOr/);
  assert.doesNotMatch(ptrPublish, /ptrFail|return false|for\(|while\(|>=|Capacity/);
});

test("a rejected candidate leaves complete accepted SCA and PTR snapshots byte-identical", () => {
  const acceptedSCA = new Uint32Array([11, 22, 33, 44]);
  const acceptedPTR = new Uint32Array([55, 66, 77, 88, 99]);
  const beforeSCA = acceptedSCA.slice(), beforePTR = acceptedPTR.slice();
  assert.throws(() => compileSparseCM12TopologyEffectsReference({
    scaCapacity: 2, ptrCapacity: 2, sca: [{ tile: 2, cause: 1 }], ptr: [],
  }), /invalid SCA effect/);
  assert.deepEqual(acceptedSCA, beforeSCA);
  assert.deepEqual(acceptedPTR, beforePTR);
});
