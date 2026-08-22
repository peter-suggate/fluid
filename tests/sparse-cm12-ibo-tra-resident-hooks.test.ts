import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12IboTRAResidentHooksWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-resident-hooks.wgsl";

test("IBO TRA resident hooks preserve stable packets and both gamma banks", () => {
  const source = createSparseCM12IboTRAResidentHooksWGSL();
  assert.match(source, /IBOStablePacketLeaf\(packet\)/);
  assert.match(source, /IBOStablePacketLocal\(packet\)/);
  assert.match(source, /leaf\*IBO1_PACKETS_PER_LEAF/);
  assert.match(source, /DynamicClosureSurfaceMask/);
  assert.match(source, /DynamicClosureDensityMask/);
  assert.match(source, /scatterGammaRow\(row,destinationDensity\(\),destinationGamma\(\)\)/);
  assert.match(source, /scatterGammaRow\(row,p\.stateOffsets2\.x,p\.stateOffsets2\.y\)/);
  assert.doesNotMatch(source, /TRA1|tra1|rowTermCount|incidence|ownerCellAt|fallback/i);
});

test("IBO TRA resident hooks reject invalid identifiers", () => {
  assert.throws(() => createSparseCM12IboTRAResidentHooksWGSL({
    residentPrefix: "bad-prefix",
  }), /identifier/);
});
