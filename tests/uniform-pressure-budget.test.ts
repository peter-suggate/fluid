import assert from "node:assert/strict";
import test from "node:test";
import { uniformCM11aCycleBudget as budget } from "../lib/methods/uniform/pressure-policy";

test("minimal pressure prefix grows from missed tolerance and shrinks from observed convergence", () => {
 const policy={maxCycles:7,initialCycles:1,headroom:0};
 assert.equal(budget(policy),1);
 assert.equal(budget({...policy,lastExecutedCycles:NaN}),1);
 for(const [executed,next] of [[1,3],[3,6],[6,7],[7,7]])
  assert.equal(budget({...policy,lastExecutedCycles:executed,lastConverged:false}),next);
 for(const executed of [0,1,2,7])
  assert.equal(budget({...policy,lastExecutedCycles:executed,lastConverged:true}),Math.max(1,executed));
 assert.equal(budget({...policy,maxCycles:0}),0);
 assert.equal(budget({...policy,maxCycles:1,lastExecutedCycles:1,lastConverged:false}),1);
 assert.equal(budget({maxCycles:7,headroom:1}),7,"reference startup is unchanged");
 assert.equal(budget({maxCycles:7,headroom:1,lastExecutedCycles:1,lastConverged:true}),2);
});
