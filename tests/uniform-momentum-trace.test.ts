import assert from "node:assert/strict";
import test from "node:test";
import { UNIFORM_MOMENTUM_TRACE_SEGMENTS, uniformCachedMomentumSamplingWGSL } from "../lib/methods/uniform/uniform-cached-momentum.wgsl";
import { uniformNearSettled } from "../lib/methods/uniform/uniform-near-settled";
import { narrowBandFlipValues } from "../lib/methods/uniform/uniform-narrow-band-method";

const traceOf = (source: string) => {
  const at = source.indexOf("fn umUnitTrace(");
  assert.ok(at >= 0, "the sampler defines umUnitTrace");
  return source.slice(at, source.indexOf("\n}", at) + 2);
};

test("momentumTrace defaults to the one-step midpoint trace, whose source is unchanged", () => {
  assert.equal(uniformNearSettled({}).momentumTrace, "midpoint");
  assert.equal(uniformNearSettled({ momentumTrace: "bogus" }).momentumTrace, "midpoint");
  assert.equal(uniformNearSettled({ momentumTrace: "substep" }).momentumTrace, "substep");
  assert.equal(narrowBandFlipValues({}).momentumTrace, "midpoint");
  for (const sampler of ["software", "texture32", "texture16"] as const) {
    assert.equal(uniformCachedMomentumSamplingWGSL(sampler), uniformCachedMomentumSamplingWGSL(sampler, "midpoint"));
    assert.equal(traceOf(uniformCachedMomentumSamplingWGSL(sampler)), `fn umUnitTrace(position:vec3f,axis:u32,dt:f32,h:vec3f)->f32 {
 var at=position;
 for(var stage=select(0u,2u,abs(dt)<=1e-7);stage<3u;stage++){
  let last=stage==2u;
  let v=umUnitSample(at,select(0u,axis,last),select(3u,axis+1u,last));
  if(last){return v[axis];}
  at=umClampMomentum(position-select(0.5,1.0,stage==1u)*v*dt/h);
 }
 return 0.0;
}`);
  }
});

test("the substep trace segments one backward trace by travel, at one sampler call site", () => {
  for (const sampler of ["software", "texture32", "texture16"] as const) {
    const source = uniformCachedMomentumSamplingWGSL(sampler, "substep");
    const trace = traceOf(source);
    assert.equal(trace.split("umUnitSample(").length - 1, 1, "one sampler call site (Metal inlines each call)");
    assert.match(trace, /min\(left,1\.0\/max\(speed,1e-30\)\)/, "a segment spans at most one cell of travel");
    assert.match(source, new RegExp(`const UM_TRACE_SEGMENTS:u32=${UNIFORM_MOMENTUM_TRACE_SEGMENTS}u;`));
    assert.doesNotMatch(trace, /clamp\(v|min\(speed|max\(-/, "no speed is limited");
    assert.equal(source.split("fn umUnitTrace(").length - 1, 1);
  }
});
