import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { uniformReferenceComputeShader } from "../lib/webgpu-uniform-reference.wgsl";
import { uniformPressureMultigridWGSL } from "../lib/webgpu-uniform-pressure-multigrid.wgsl";
import { uniformVelocityExtrapolationShader } from "../lib/webgpu-uniform-velocity-extrapolation.wgsl";

test("uniform frame kernels consume the GPU-resident active region", () => {
  for (const entryPoint of ["traceGammaAndBeta", "gatherConservativeDensity",
    "semiLagrangianAdvection", "project", "sharpenCompute", "reduceDiagnostics"]) {
    const start = uniformReferenceComputeShader.indexOf(`fn ${entryPoint}(`);
    const body = start < 0 ? "" : uniformReferenceComputeShader.slice(start, start + 500);
    assert.match(body ?? "", /activeId\(gid\)/, `${entryPoint} must offset its indirect dispatch`);
  }
  assert.match(uniformReferenceComputeShader,
    /fn scanExternalActiveSources[\s\S]*inflowSweptPlugSource[\s\S]*dropSource/,
    "the census must include external mass sources before their first frame");
  assert.match(uniformReferenceComputeShader,
    /fn scanActiveRegion[\s\S]*?let id=activeId\(gid\)/,
    "ordinary census work must scale with the previous conservative box");
  assert.match(uniformReferenceComputeShader,
    /fn buildExtrapolationAuthority[\s\S]*activeId\(gid\)/,
    "runtime rho-prime and face authority must not revisit empty-domain cells");
  assert.match(uniformReferenceComputeShader,
    /currentMaximum=min\(d,observedMax\+padding\)/,
    "the active region must include the velocity-reach safety band");
});

test("active dispatch retains one clearing tail instead of the complete swept history", () => {
  assert.match(uniformReferenceComputeShader,
    /let previousMinimum=vec3u\(activeRegion\[0\],activeRegion\[1\],activeRegion\[2\]\)/);
  assert.match(uniformReferenceComputeShader, /let minimum=min\(previousMinimum,currentMinimum\)/);
  assert.match(uniformReferenceComputeShader,
    /atomicStore\(&activeScratch\[0\],currentMinimum\.x\)[\s\S]*atomicStore\(&activeScratch\[3\],currentMaximum\.x\)/);
  assert.doesNotMatch(uniformReferenceComputeShader, /minimum=min\(minimum,observedMin/,
    "an all-history union eventually turns a sparse moving scene back into a dense one");
});

test("FIM updates and every non-coarsest pressure pass use active indirect bounds", () => {
  assert.match(uniformVelocityExtrapolationShader,
    /fn updateActiveFront[\s\S]*?let p = activeBaseId\(gid\)/);
  assert.match(uniformVelocityExtrapolationShader,
    /dispatchArgs\.x[\s\S]*activeRegion\[13\]/);
  assert.match(uniformPressureMultigridWGSL,
    /fn mgActiveId[\s\S]*activeRegion\[base\]/);

  const pressureHost = readFileSync(new URL("../lib/webgpu-uniform-pressure-multigrid.ts", import.meta.url), "utf8");
  assert.match(pressureHost,
    /dispatch\.entryPoint !== "mgSolveCoarsest"[\s\S]*dispatchWorkgroupsIndirect/,
    "the barrier-synchronous coarsest solve stays direct while hierarchy grids become indirect");
});

test("writable census data is copied before becoming indirect/read-only authority", () => {
  const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
  assert.match(host,
    /copyBufferToBuffer\(this\.activeScratch, 0, this\.activeRegion[\s\S]*copyBufferToBuffer\(this\.activeScratch, 0, this\.activeDispatch/,
    "WebGPU must not bind one writable storage buffer as the same pass's indirect authority");
  assert.match(host, /FLUID_UNIFORM_ACTIVE_REGION !== "0"/,
    "Dawn must retain a dense A\/B arm for performance regression measurements");
});
