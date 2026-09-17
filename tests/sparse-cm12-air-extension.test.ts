import { momentumSnapshotLayout } from "../lib/methods/adaptive-volume/sparse-cm12-momentum-snapshot";
import assert from "node:assert/strict";
import test from "node:test";
import { adaptiveMassMethod, adaptiveMassSolverOptions, ADAPTIVE_MASS_RUNTIME_PARAM_KEYS } from "../lib/methods/adaptive-volume/method";
import { ALGORITHM_PARAMS } from "../lib/methods/adaptive-volume/features/algorithms/definition";
import { airExtensionLayout } from "../lib/methods/adaptive-volume/sparse-cm12-air-extension";

test("air correction defaults on, is normalized, and switchable without resetting the scene",()=>{
  const control=ALGORITHM_PARAMS.find(p=>p.key==="airExtension")!;
  assert.equal(control.default,"on");assert.equal(control.update,"runtime");
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("airExtension"));
  for(const value of [undefined,"off","on","invalid"]){
    const values: Record<string,string>=value===undefined?{}:{airExtension:value};
    const normalized=adaptiveMassMethod.normalizeValues!(values);
    assert.equal(normalized.airExtension,value==="off"?"off":"on");
    assert.equal(adaptiveMassSolverOptions(normalized).airExtensionEnabled,value!=="off");
  }
});

test("air storage keeps the original velocity prefix separate from scratch and receipts",()=>{
  const layout=airExtensionLayout(65,180);
  assert.equal(layout.cellBase,65);assert.equal(layout.rowBase,260);
  assert.equal(layout.header,440);assert.equal(layout.reduction,444);
  assert.equal(layout.byteLength,16*446);
  assert.throws(()=>airExtensionLayout(-1,2),RangeError);
});

test("momentum snapshot owns geometry, incidence and a half-empty spatial hash",()=>{
  const layout=momentumSnapshotLayout(65,180,400);
  assert.equal(layout.hashCapacity,256);
  assert.equal(layout.rowBase,layout.cellBase+8*65);
  assert.equal(layout.incidenceBase,layout.rowBase+8*180);
  assert.equal(layout.hashBase,layout.incidenceBase+400);
  assert.equal(layout.byteLength,4*(layout.hashBase+256));
  assert.throws(()=>momentumSnapshotLayout(-1,2,3),RangeError);
});
