import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  createSparseCM12FaceProjectionAuthorityLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-face-projection-authority";
import {
  createSparseCM12IncrementalActivityLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity";

test("brick-owned face preparation omits the per-row preparation authority", () => {
  const options = {
    rowCapacity: 1_000_000,
    cellCapacity: 500_000,
    brickFineResolution: 16 as const,
    presentationPageResolution: 16 as const,
  };
  const layout = createSparseCM12FaceProjectionAuthorityLayout(options);
  assert.ok(!("preparation" in layout));
  assert.ok(!("preparedAuthorityBaseWords" in layout));
  assert.ok(!("preparationCertificateBaseWords" in layout));
  assert.ok(layout.projection.activeBitsBaseWords
    >= layout.acceptedPressureBitsBaseWords + options.cellCapacity);
  assert.equal(layout.projection.activeBitWordCount,
    Math.ceil(options.rowCapacity / 32));
});

test("face preparation reuses the single activity brick worklist", () => {
  const layout = createSparseCM12IncrementalActivityLayout({
    baseWords: 0, brickCount: 12,
  });
  assert.equal(layout.brickListBaseWords,
    layout.brickVelocityStampBaseWords + layout.brickCount);
});

test("production face preparation consumes one contiguous owned range per dirty brick", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("fn prepareSparseCM12DirtyBrickFaceRows");
  const end = source.indexOf("fn markSparseCM12FaceProjectionFromDirtyBricks", begin);
  assert.ok(begin >= 0 && end > begin);
  const preparation = source.slice(begin, end);

  assert.match(preparation, /incrementalActivityBrickInvocation\(wid\.x\)/);
  assert.match(preparation,
    /templateRowOwnerRange\(brick,acceptedBrickResolution\(brick\)\)/);
  assert.match(preparation, /let row=range\.x\+local/);
  assert.doesNotMatch(preparation, /incidence(Begin|End)|fpaMark|atomic/);
  assert.doesNotMatch(source, /OwnedRowPacket|incrementalFacePreparationPacketDirty/);
});

test("face preparation traces a dense cache derived from TEI and effective velocity", () => {
  const wgsl = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const publishBegin = wgsl.indexOf("fn publishSparseCM12FaceVelocitySupport");
  const publishEnd = wgsl.indexOf("fn clearSparseCM12RetiredFaceVelocitySupport",
    publishBegin);
  assert.ok(publishBegin >= 0 && publishEnd > publishBegin);
  const publish = wgsl.slice(publishBegin, publishEnd);
  assert.match(publish, /cm12TeiLoadLeaf\(acceptedTopologySlot\(\),brick\)/);
  assert.match(publish, /cm12EffectiveTransportVelocity\(cell\)/);
  assert.match(publish, /leaf\.first\+cellCoordinate\.x\+leaf\.valid\.x/);
  assert.match(publish, /state\[sourceDensity\(\)\+cell\]>CM12_LIQUID_ISOVALUE/);
  assert.doesNotMatch(publish,
    /acceptedBrickResolution|templateBrickCellRange|cm12ExtensionTransportVelocity|ownerCellAt/);

  const supportBegin = wgsl.indexOf("fn faceVelocitySupportAt");
  const supportEnd = wgsl.indexOf("fn publishSparseCM12FaceVelocitySupport", supportBegin);
  const support = wgsl.slice(supportBegin, supportEnd);
  assert.match(support, /FACE_VELOCITY_SUPPORT\+4u\*index/);
  assert.doesNotMatch(support, /cm12TeiOwnerAtFine|cm12EffectiveTransportVelocity/,
    "the RK2 hot path must retain contiguous dense-cache reads");

  const sampleBegin = wgsl.indexOf("fn sampleFaceVelocitySupport");
  const sampleEnd = wgsl.indexOf("fn traceFaceDeparture", sampleBegin);
  const sample = wgsl.slice(sampleBegin, sampleEnd);
  assert.match(sample,
    /for\(var dz=0;dz<2;dz\+=1\)\{for\(var dy=0;dy<2;dy\+=1\)\{for\(var dx=0;dx<2;dx\+=1\)/,
    "the face cutover must preserve the legacy corner accumulation order");

  const prepareBegin = wgsl.indexOf("fn prepareTransportFaceRow");
  const prepareEnd = wgsl.indexOf("fn publishSparseCM12MovingSolidVelocityRoots",
    prepareBegin);
  const prepare = wgsl.slice(prepareBegin, prepareEnd);
  assert.match(prepare, /faceVelocitySupportAt/);
  assert.match(prepare, /traceFaceDeparture/);
  assert.doesNotMatch(prepare,
    /ownerCellAt|sampleVelocity|incidence(Begin|End)|rowTerm|termCell/);

  const host = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const stageBegin = host.indexOf('stage("face-preparation"');
  const stageEnd = host.indexOf('stage("conservative-transport"', stageBegin);
  const stage = host.slice(stageBegin, stageEnd);
  assert.match(stage, /useBindGroup\(this\.transportBindGroup\)/);
  assert.match(stage, /dispatchActivity\("clearSparseCM12RetiredFaceVelocitySupport"\)/);
  assert.match(stage,
    /dispatch\("publishSparseCM12FaceVelocitySupport",\s*this\.incrementalActivityLayout\.brickCount\)/);
  assert.match(stage, /dispatchActivity\("prepareSparseCM12DirtyBrickFaceRows"\)/);
  assert.match(host, /readonly faceVelocitySupport: number/);
});

test("face invalidation publishes a geometric brick closure without incidence walks", () => {
  const source = readFileSync(new URL(
    "../lib/methods/adaptive-mass/sparse-cm12-incremental-activity.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const begin = source.indexOf("fn incrementalActivityPublishFaceBrickClosure");
  const end = source.indexOf("fn incrementalActivityMarkCellClosure", begin);
  assert.ok(begin >= 0 && end > begin);
  const closure = source.slice(begin, end);
  assert.match(closure, /brickDirectoryLookupAtCoordinate/);
  assert.match(closure, /incrementalActivityClaimBrick\(owner\)/);
  assert.doesNotMatch(closure, /incidence(Begin|End)|rowTerm|termCell/);
});
