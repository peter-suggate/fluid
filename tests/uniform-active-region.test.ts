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
    /activeScratch\[0\]=currentMinimum\.x[\s\S]*activeScratch\[3\]=currentMaximum\.x/);
  assert.doesNotMatch(uniformReferenceComputeShader, /minimum=min\(minimum,observedMin/,
    "an all-history union eventually turns a sparse moving scene back into a dense one");
});

test("active census is a contention-free hierarchical reduction", () => {
  const censusStart = uniformReferenceComputeShader.indexOf("fn scanActiveRegion(");
  const censusEnd = uniformReferenceComputeShader.indexOf("fn activeCeilDiv", censusStart);
  const census = uniformReferenceComputeShader.slice(censusStart, censusEnd);
  assert.doesNotMatch(census, /atomic(?:Min|Max|Add|Store|Load)/,
    "no census cell or workgroup may contend on a device-global atomic");
  assert.match(uniformReferenceComputeShader,
    /var<workgroup> activeMinimumLanes:[\s\S]*writeActiveWorkgroupSummary/,
    "each workgroup must first reduce its 64 cells locally");
  assert.match(uniformReferenceComputeShader,
    /fn reduceActiveSummaryRange[\s\S]*summaryIndex\+=256u[\s\S]*if\(lane==0u\)/,
    "one small second-level pass must merge the uncontended workgroup records");
  assert.match(uniformReferenceComputeShader,
    /if\(wet\)[\s\S]*speedBits=bitcast<u32>\(length\(faceVelocity\(id\)\)\)/,
    "air extrapolation outliers must not inflate the CFL padding");

  const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
  assert.doesNotMatch(host, /pipelines\.resetActiveRegion/);
  assert.match(host,
    /scan prior active liquid bounds[\s\S]*reduce active workgroup summaries[\s\S]*finalize active liquid dispatches/);
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
  assert.match(host, /options\.activeRegion === true[\s\S]*FLUID_UNIFORM_ACTIVE_REGION !== "0"/,
    "dense dispatch must be the default while retaining an explicit sparse A\/B arm");
});
