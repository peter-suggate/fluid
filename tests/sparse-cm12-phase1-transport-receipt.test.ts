import assert from "node:assert/strict";
import test from "node:test";

import {
  SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER_WORDS,
  createSparseCM12Phase1TransportQALayout,
  sparseCM12Phase1Sha256,
} from "../lib/methods/adaptive-mass/sparse-cm12-phase1-transport-receipt";
import { createSparseCM12Phase1TransportQAWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-phase1-transport-receipt.wgsl";

test("Phase-1 QA layout is an exact non-overlapping 26-word-per-cell receipt", () => {
  const layout = createSparseCM12Phase1TransportQALayout({
    baseWords: 64, cellCapacity: 17,
  });
  assert.equal(layout.departureBaseWords,
    64 + SPARSE_CM12_PHASE1_TRANSPORT_QA_HEADER_WORDS);
  assert.equal(layout.totalWords - layout.departureBaseWords, 26 * 17);
  assert.equal(layout.stencilCellBaseWords - layout.departureBaseWords, 3 * 17);
  assert.equal(layout.stencilWeightBaseWords - layout.stencilCellBaseWords, 8 * 17);
  assert.equal(layout.betaBaseWords - layout.stencilWeightBaseWords, 8 * 17);
  assert.equal(layout.massGammaBaseWords - layout.massDensityBaseWords, 17);
});

test("Phase-1 WGSL captures the real pass-boundary bit representations", () => {
  const layout = createSparseCM12Phase1TransportQALayout({
    baseWords: 128, cellCapacity: 64,
  });
  const wgsl = createSparseCM12Phase1TransportQAWGSL({
    layout, publishEffectiveVelocity: true, validateExecutionImage: true,
  });
  assert.match(wgsl, /bitcast<u32>\(departure\[axis\]\)/);
  assert.match(wgsl, /stencil\.cells\[corner\]/);
  assert.match(wgsl, /bitcast<u32>\(stencil\.weights\[corner\]\)/);
  assert.match(wgsl, /CM12_P1TQ_BETA\+cell/);
  assert.match(wgsl, /CM12_P1TQ_DEFICIT_DENSITY\+cell/);
  assert.match(wgsl, /CM12_P1TQ_MASS_DENSITY\+cell/);
  assert.match(wgsl, /cm12PublishVexAcceptedEffectiveVelocity\(cell,value\)/);
  assert.match(wgsl, /cm12TeiPacketCell\(packet\.x,packet\.y,acceptedTopologySlot\(\)\)/);
});

test("Phase-1 receipt hashes raw bytes", async () => {
  assert.equal(await sparseCM12Phase1Sha256(new Uint32Array([0x3f80_0000])),
    "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c");
});
