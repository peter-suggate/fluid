import assert from "node:assert/strict";
import test from "node:test";

import {
  makeOctreeFaceBandAirSampleWGSL,
  octreeFaceBandWGSL,
} from "../lib/webgpu-octree-face-closest-point";

const compact = (source: string): string => source.replace(/\s+/g, "");

function wgslFunction(source: string, name: string): string {
  const shader = compact(source);
  const start = shader.indexOf(`fn${name}(`);
  assert.notEqual(start, -1, `missing WGSL function ${name}`);
  const open = shader.indexOf("{", start);
  assert.notEqual(open, -1, `missing WGSL body for ${name}`);
  let depth = 0;
  for (let cursor = open; cursor < shader.length; cursor += 1) {
    if (shader[cursor] === "{") depth += 1;
    else if (shader[cursor] === "}" && --depth === 0) {
      return shader.slice(start, cursor + 1);
    }
  }
  assert.fail(`unterminated WGSL function ${name}`);
}

test("step 52 permits only the observed symmetric S1-to-S3 carrier exception", () => {
  const carrier = wgslFunction(octreeFaceBandWGSL, "provisionalS1ToS3Carrier");
  assert.match(carrier,
    /s1Reason!=7u\|\|s3Reason!=1u\|\|s1Index==INVALID\|\|s3Index==INVALID/,
    "the carrier requires the exact missing-selector and rejected-S3-anchor reasons");
  assert.match(carrier,
    /\(s1\.flags&ROW_SUPPORT1\)==0u\|\|\(s1\.flags&\(ROW_SUPPORT3_NODE\|ROW_SUPPORT3_ENDPOINT\)\)!=0u\|\|\(s3\.flags&ROW_SUPPORT3_NODE\)==0u/,
    "the carrier accepts only a pure S1 row incident to an S3 node");
  assert.match(carrier,
    /returnprovisionalCellVector\(coord\(s1\.cell\),s1\.size\)/,
    "the bounded fallback reuses only the incident S1 provisional vector");
  assert.doesNotMatch(carrier, /for\(|while\(|loop\{|containingPublishedRow|nearest|catalog/,
    "the exception performs no neighborhood, row-arena, or catalog search");

  const sample = wgslFunction(octreeFaceBandWGSL, "sampleTransientBandPowerFaces");
  assert.match(sample,
    /elseif\(positiveValid\)\{full=positive\.xyz;\}else\{letnegativeReason=.*letpositiveReason=.*provisionalS1ToS3Carrier\(face\.negativeRow,face\.positiveRow,negativeReason,positiveReason\).*provisionalS1ToS3Carrier\(face\.positiveRow,face\.negativeRow,positiveReason,negativeReason\)/s,
    "the symmetric exception is reachable only after both exact endpoint interpolants fail");
  assert.match(sample,
    /if\(velocityValid\(carrier\)\)\{full=carrier\.xyz;\}else\{face\.pad=.*transientFaceFail\(POINT_SAMPLE,faceIndex\);return;\}/s,
    "all other endpoint/reason combinations remain fail-closed");
});

test("fused air sampling accepts only a current band or its exact predecessor", () => {
  const shader = makeOctreeFaceBandAirSampleWGSL();
  const clock = wgslFunction(shader, "sampledBandGenerationValid");
  assert.match(clock,
    /fine=sp\.fineGeneration&mask.*paramsFine=p\.generation&mask.*band=control\.generation&mask.*power=p\.powerGeneration&mask.*predecessor=\(fine\+mask\)&mask.*returnparamsFine==fine&&\(band==fine\|\|\(power==fine&&band==predecessor\)\)/s,
    "the predecessor is accepted only when fine and power clocks are current");

  const classify = wgslFunction(shader, "classifyAirBandVelocity");
  assert.match(classify,
    /!sampledBandGenerationValid\(\)\|\|pointControl\.generation!=sp\.fineGeneration/,
    "classification still requires the current reconstructed point field");
  const finalize = wgslFunction(shader, "finalizeAirBandVelocity");
  assert.match(finalize,
    /control\.valid==VALID&&sampledBandGenerationValid\(\)&&pointControl\.valid==VALID&&pointControl\.flags==0u&&pointControl\.generation==sp\.fineGeneration/,
    "publication rechecks both the bounded predecessor rule and current point-field authority");
});
