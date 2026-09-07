import assert from "node:assert/strict";
import test from "node:test";
import { composeFeatures } from "../composition";
import { configurationChangeImpact } from "../lifecycle";
import { normalizeControlNumber } from "../controls";
test("effective changes select the strongest lifecycle and ignore equal values", () => {
  const before = composeFeatures({ features: [] });
  const settings = [{ key: "strength", impact: "live" }, { key: "size", impact: "rebuild" }] as const;
  assert.equal(configurationChangeImpact(before,before,{strength:1,size:3},{strength:1,size:3},settings), "none");
  assert.equal(configurationChangeImpact(before,before,{strength:1,size:3},{strength:2,size:3},settings), "live");
  assert.equal(configurationChangeImpact(before,before,{strength:1,size:3},{strength:2,size:4},settings), "rebuild");
  const after = composeFeatures({features:[{id:"method",variants:[{id:"new",point:"surface",default:true,update:"reset"}]}]});
  assert.equal(configurationChangeImpact(before,after,{},{},[]), "reset");
});
test("shared numeric metadata clamps values and optionally snaps editor input", () => {
  const control = { min: 1, max: 8, step: 2 };
  assert.equal(normalizeControlNumber(100, 3, control), 8);
  assert.equal(normalizeControlNumber(NaN, 3, control), 3);
  assert.equal(normalizeControlNumber(4.1, 3, control, true), 5);
  assert.equal(normalizeControlNumber(4.1, 3, control), 4.1);
});
