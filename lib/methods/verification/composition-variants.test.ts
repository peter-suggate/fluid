import assert from "node:assert/strict";
import test from "node:test";
import { resolveMethodComposition as uniform } from "../uniform/composition";
import { resolveMethodComposition as adaptive } from "../adaptive-volume/composition";
import { resolveMethodComposition as losasso } from "../losasso/composition";
import { resolveMethodComposition as power } from "../power/composition";
import { ALGORITHM_PARAMS as uniformParams } from "../uniform/features/algorithms/definition";
import { ALGORITHM_PARAMS as adaptiveParams } from "../adaptive-volume/features/algorithms/definition";
import { ADAPTIVITY_PARAMS } from "../adaptive-volume/features/adaptivity/definition";
import { SPARSE_CM12_ACTIVITY_POLICY } from "../adaptive-volume/features/adaptivity/policy";

const selected = (composition: ReturnType<typeof uniform>, point: string) => composition.variants.find(v => v.point === point);

test("uniform algorithm choices compose pairwise with schema-defined lifecycles", () => {
  for (const a of uniformParams) for (const b of uniformParams) {
    if (a.kind !== "select" || b.kind !== "select" || a.key === b.key) continue;
    for (const av of a.options) for (const bv of b.options) {
      const result = uniform({[a.key]:av.value,[b.key]:bv.value});
      assert.equal(selected(result,`simulation.uniform.algorithms.${a.key}`)?.id,av.value);
      assert.equal(selected(result,`simulation.uniform.algorithms.${b.key}`)?.id,bv.value);
      assert.equal(selected(result,`simulation.uniform.algorithms.${a.key}`)?.update,a.update === "runtime" ? "live" : "rebuild");
    }
  }
});

test("all adaptive criteria compose with live conditioning and timestep alternatives", () => {
  for (const selectorMode of ["surface","activity","coarse-first"]) for (const param of adaptiveParams) {
    if (param.kind !== "select") continue;
    for (const option of param.options) {
      const result=adaptive({selectorMode,[param.key]:option.value});
      assert.equal(selected(result,"simulation.adaptive-volume.adaptivity")?.id,selectorMode);
      assert.equal(selected(result,`simulation.adaptive-volume.algorithms.${param.key}`)?.id,option.value);
      assert.equal(selected(result,`simulation.adaptive-volume.algorithms.${param.key}`)?.update,"live");
    }
  }
});

test("Losasso extension alternatives compose with each supported surface representation", () => {
  for (const globalFineLevelSetFactor of ["1","4","8"]) for (const losassoVelocityExtension of ["fixed-jacobi","causal-front"]) {
    const result=losasso({globalFineLevelSetFactor,losassoVelocityExtension});
    assert.equal(selected(result,"simulation.losasso.algorithms.losassoVelocityExtension")?.id,losassoVelocityExtension);
    assert.equal(selected(result,"simulation.losasso.algorithms.losassoVelocityExtension")?.update,"rebuild");
    assert.equal(result.features[0]?.outputs?.[0]?.representation,globalFineLevelSetFactor === "1" ? "octree" : "sparse-fine-levelset-bricks");
  }
});

test("fixed pressure providers cannot become unsupported method combinations", () => {
  assert.equal(selected(uniform(),"simulation.uniform.pressure")?.id,"cm11a-lcp-multigrid");
  assert.equal(selected(adaptive(),"simulation.adaptive-volume.pressure")?.id,"sparse-jacobi-pcg");
  assert.equal(selected(losasso(),"simulation.losasso.pressure")?.id,"vcycle-mgpcg");
  assert.equal(selected(power(),"simulation.power-liquids.pressure")?.id,"power2017-hybrid");
  assert.throws(() => uniform({velocityTransport:"causal-front"}),/supported variant/);
  assert.throws(() => losasso({losassoVelocityExtension:"maccormack"}),/supported variant/);
  assert.throws(() => power({globalFineLevelSetFactor:"16"}),/supported variant/);
  assert.ok(!power().variants.some(v => v.point.includes("maximumLeafSize")));
});

test("adaptivity UI defaults are the policy defaults", () => {
  for (const param of ADAPTIVITY_PARAMS) {
    if (!(param.key in SPARSE_CM12_ACTIVITY_POLICY)) continue;
    assert.equal(param.default,SPARSE_CM12_ACTIVITY_POLICY[param.key as keyof typeof SPARSE_CM12_ACTIVITY_POLICY]);
  }
});

test("boolean overrides preserve the existing explicit-string solver semantics", () => {
  for (const params of [uniformParams, adaptiveParams]) for (const param of params) {
    const resolve = params === uniformParams ? uniform : adaptive;
    const method = params === uniformParams ? "uniform" : "adaptive-volume";
    for (const value of [true,false]) {
      assert.equal(selected(resolve({[param.key]:value}),`simulation.${method}.algorithms.${param.key}`)?.id,param.default);
    }
  }
});
