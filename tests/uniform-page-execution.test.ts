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

test("native rectangular execution requires complete unique accepted coverage", async () => {
  const {uniformPageHasRectangularCoverage:complete}=await import('../lib/methods/uniform/uniform-page-execution');
  for(const reverse of [false,true])assert.equal(complete(initialUniformPageDomain([192,96,32],32,reverse)),true);
  assert.equal(complete(initialUniformPageDomain([40,20,16])),true);
  const truncated=initialUniformPageDomain([32,32,32]);truncated.words=truncated.words.slice(0,16);assert.equal(complete(truncated),false);
  const wrongEdge=initialUniformPageDomain([64,32,32]);wrongEdge.words[9]=16;assert.equal(complete(wrongEdge),false);
  const missing=initialUniformPageDomain([64,32,32]);missing.words[8]=1;assert.equal(complete(missing),false);
  const duplicate=initialUniformPageDomain([64,32,32]);duplicate.words[32]=0;assert.equal(complete(duplicate),false);
  const outside=initialUniformPageDomain([64,32,32]);outside.words[32]=2;assert.equal(complete(outside),false);
});
