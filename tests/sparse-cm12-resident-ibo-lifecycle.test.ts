import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const wgsl = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url), "utf8");
const host = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url), "utf8");

const functionSource = (source: string, name: string, next: string): string => {
  const begin = source.indexOf(`fn ${name}`);
  const end = source.indexOf(next, begin);
  assert.ok(begin >= 0 && end > begin, `${name} source range must remain identifiable`);
  return source.slice(begin, end);
};

test("resident topology authorization consumes IBO before the distinct selector flip", () => {
  const validate = functionSource(wgsl, "validateAndAuthorizeShadowTopology",
    "fn finalizeAuthorizedShadowTopology");
  const receipt = validate.indexOf("${internedBoundaryCommitReceipt}");
  const authorize = validate.indexOf("atomicStore(&topologyArena[base+3u],2u)");
  assert.ok(receipt >= 0 && authorize > receipt,
    "IBO must participate in the aggregate decision before phase-2 authorization");
  const finalize = functionSource(wgsl, "finalizeAuthorizedShadowTopology",
    "fn publishSparseCM12TopologyVelocityRootsWork");
  assert.match(finalize, /topologyArena\[base\+3u\]\)!=2u/);
  assert.match(finalize, /atomicStore\(&topologyArena\[base\+2u\],slot\)/);
  assert.match(wgsl, /cm12IBOLoad\(iboHeader\+1u\)==2u/);
  assert.match(wgsl, /cm12IBOLoad\(iboHeader\+5u\)==0u/);
  assert.match(wgsl, /cm12IBOSelectorMirror\(\)==acceptedTopologySlot\(\)/);
});

test("IBO faults cannot disappear on weak CAS and replay mirrors either outcome", () => {
  assert.match(wgsl, /atomicMax\(&topologyArena\[header\+5u\],fault\.x\)/);
  assert.doesNotMatch(wgsl,
    /atomicCompareExchangeWeak\(&topologyArena\[header\+5u\],0u,fault\.x\)/);
  assert.match(wgsl, /let source=acceptedTopologySlot\(\);let retiredSlot=1u-source/);
  assert.match(wgsl, /cm12IBOStore\(header\+1u,0u\)/);
  for (const marker of ["stage(\"candidate-transfer\"", "resident liquid injection topology"]) {
    const begin = host.indexOf(marker);assert.ok(begin >= 0);
    const authorize = host.indexOf("validateAndAuthorizeShadowTopology", begin);
    const finalize = host.indexOf("finalizeAuthorizedShadowTopology", authorize);
    const replay = host.indexOf("replaySparseCM12InternedBoundaryDelta", finalize);
    assert.ok(authorize > begin && finalize > authorize && replay > finalize,
      `${marker} must authorize, publish/flip, then replay only the retired slot`);
  }
});
