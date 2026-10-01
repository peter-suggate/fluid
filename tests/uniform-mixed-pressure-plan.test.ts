import assert from "node:assert/strict";
import test from "node:test";
import {uniformMixedPressureReserve} from "../lib/methods/uniform/uniform-mixed-pressure-schedule";

const maximum={vCycles:4,fullCycles:3};
test("coarse pressure reserve covers a third impact cycle and respects both configured caps",()=>{
 assert.deepEqual(uniformMixedPressureReserve({vCycles:2,fullCycles:0},maximum,1),{vCycles:3,fullCycles:0});
 assert.deepEqual(uniformMixedPressureReserve({vCycles:4,fullCycles:0},maximum,1),{vCycles:4,fullCycles:1});
 assert.deepEqual(uniformMixedPressureReserve(maximum,maximum,1),maximum);
 assert.deepEqual(uniformMixedPressureReserve({vCycles:2,fullCycles:0},maximum,0),{vCycles:2,fullCycles:0});
});
test("reserve retains a stalled V phase's full-cycle fallback",()=>{
 assert.deepEqual(uniformMixedPressureReserve({vCycles:0,fullCycles:2},maximum,1),{vCycles:0,fullCycles:3});
 assert.deepEqual(uniformMixedPressureReserve({vCycles:0,fullCycles:3},maximum,2),{vCycles:0,fullCycles:3});
 assert.deepEqual(uniformMixedPressureReserve({vCycles:1,fullCycles:1},maximum,1),{vCycles:1,fullCycles:2});
});
