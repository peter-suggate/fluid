import assert from "node:assert/strict";
import test from "node:test";
import "../lib/methods";
import { defaultMethodId, getMethod, interactiveSimulationMethods } from "../lib/core/method-registry";
import { resolveMethodValues } from "../lib/core/method-contract";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";

test("narrow-band FLIP is selectable and fixes complete surface support without sharpening",async()=>{
 const method=getMethod("uniform-narrow-band-flip");
 assert.ok(interactiveSimulationMethods().includes(method));assert.equal(defaultMethodId(),"uniform-volume");
 assert.equal((await method.harness!()).methodId,method.id);assert.equal((await method.pipelineGraph!()).methodId,method.id);
 const values=resolveMethodValues(method,"balanced",{detailPolicy:"requested",detailShape:"off",detailShapeTolerance:2,sharpeningSweeps:8});
 assert.equal(values.detailPolicy,"dynamic");assert.equal(values.detailShape,"on");assert.equal(values.detailShapeTolerance,0);
 assert.equal(values.sharpeningSweeps,0);assert.equal(values.sharpeningDistance,0);
 const options=uniformGeometricSolverOptions(values);
 assert.equal(options.sharpeningSweeps,0);assert.equal(options.pressureCycleBudget,"lagged");
 const fixed=new Set(["detailPolicy","detailShape","detailShapeTolerance","sharpeningSweeps","sharpeningDistance"]);
 for(const stage of (await method.pipelineGraph!()).stages)for(const control of stage.controls??[])if("param" in control)assert.ok(!fixed.has(control.param));
});
