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
    baseWords: 0, stableTileCount: 32, brickCount: 12,
  });
  assert.equal(layout.brickListBaseWords,
    layout.brickStampBaseWords + layout.brickCount);
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

test("adaptive face support publishes every active packed-brick volume", () => {
  const wgsl = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
    import.meta.url,
  ), "utf8");
  const publishBegin = wgsl.indexOf("fn publishSparseCM12FaceVelocitySupport");
  const publishEnd = wgsl.indexOf("fn sampleFaceVelocitySupport", publishBegin);
  const publish = wgsl.slice(publishBegin, publishEnd);
  assert.match(publish,
    /let brick=wid\.x;if\(brick>=p\.dispatch\.w\|\|!brickActive\(brick\)\)\{return;\}/,
    "the support producer must follow the exact active packed-atlas authority");
  assert.match(publish, /let extent=min\(vec3u\(width\),p\.dimensions\.xyz/);
  assert.match(publish, /for\(var local=lane;local<count;local\+=256u\)/,
    "one workgroup must cooperatively fill each physical active-brick volume");
  assert.match(publish, /let index=q\.x\+p\.dimensions\.x\*\(q\.y\+p\.dimensions\.y\*q\.z\)/);
  assert.match(publish,
    /span=min\(taf\(base\+4u\),min\(taf\(base\+5u\),taf\(base\+6u\)\)\)/);
  assert.match(publish, /flags=1u\|select\(0u,2u,value\.w>0\.5\)/);
  const adaptivePublish = publish.slice(publish.indexOf("let brick=wid.x"),
    publish.indexOf("fn clearSparseCM12RetiredFaceVelocitySupport"));
  assert.doesNotMatch(adaptivePublish,
    /ownerCellAt|incidence(Begin|End)|rowTerm|termCell/);
  assert.match(wgsl,
    /const EXP_FACE_DENSE_SUPPORT:bool=\$\{transportExperiment === "face-characteristic-cache"/,
    "the dense publisher must remain an opt-in timing oracle");
  const clearBegin = wgsl.indexOf("fn clearSparseCM12RetiredFaceVelocitySupport");
  const clearEnd = wgsl.indexOf("fn sampleFaceVelocitySupport", clearBegin);
  const clear = wgsl.slice(clearBegin, clearEnd);
  assert.match(clear, /incrementalActivityBrickInvocation\(wid\.x\)/);
  assert.match(clear, /brick==INVALID\|\|brickActive\(brick\)/,
    "only retired dirty bricks may clear persistent support");
  assert.match(clear, /state\[at\]=0\.0/);

  const prepareBegin = wgsl.indexOf("fn prepareTransportFaceRow");
  const prepareEnd = wgsl.indexOf("fn publishSparseCM12MovingSolidVelocityRoots",
    prepareBegin);
  const prepare = wgsl.slice(prepareBegin, prepareEnd);
  assert.match(prepare, /faceVelocitySupportAt/);
  assert.match(prepare, /traceFaceDeparture/);
  assert.doesNotMatch(prepare,
    /ownerCellAt|sampleVelocity|incidence(Begin|End)|rowTerm|termCell/);
  assert.doesNotMatch(wgsl.slice(wgsl.indexOf("fn faceVelocitySupportAt"),
    wgsl.indexOf("fn presentationPhiAt")), /atomic(Load|Store)/,
  "dense face support must use ordinary contiguous state traffic");

  const host = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  assert.match(host,
    /dispatch\("publishSparseCM12FaceVelocitySupport",\s*this\.incrementalActivityLayout\.brickCount\)/);
  assert.match(host, /dispatchActivity\("clearSparseCM12RetiredFaceVelocitySupport"\)/);
  assert.match(host,
    /transportExperiment === "face-characteristic-cache"[\s\S]{0,240}publishSparseCM12FaceVelocitySupport/);
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
