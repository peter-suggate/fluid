import assert from "node:assert/strict";
import test from "node:test";
import { initialUniformPageDomain } from "../lib/methods/uniform/uniform-page-domain";
import { uniformPageHasNativeCoordinates } from "../lib/methods/uniform/uniform-page-execution";

test("native coordinates require one complete origin page, including partial payloads", () => {
  for (const d of [[16,16,16],[32,32,32],[24,16,16]] as const)
    assert.equal(uniformPageHasNativeCoordinates(initialUniformPageDomain(d)), true);
  assert.equal(uniformPageHasNativeCoordinates(initialUniformPageDomain([33,16,16])), false);
  const empty=initialUniformPageDomain([16,16,16]);empty.words[8]=0;
  assert.equal(uniformPageHasNativeCoordinates(empty), false);
  const translated=initialUniformPageDomain([16,16,16]);translated.words[16]=1;
  assert.equal(uniformPageHasNativeCoordinates(translated), false);
});
