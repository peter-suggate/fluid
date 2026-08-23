import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12IboTRAResidentHooksWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-ibo-tra-resident-hooks.wgsl";

test("IBO TRA resident hooks preserve stable packet addresses", () => {
  const source = createSparseCM12IboTRAResidentHooksWGSL();
  assert.match(source, /IBOStablePacketLeaf\(packet\)/);
  assert.match(source, /IBOStablePacketLocal\(packet\)/);
  assert.match(source, /leaf\*IBO1_PACKETS_PER_LEAF/);
  assert.match(source, /IBOTRAPacketDescriptor/);
  assert.match(source, /IBOTRAPacketLocal/);
  assert.doesNotMatch(source, /scatterGammaRow/);
  assert.doesNotMatch(source, /TRA1|tra1|rowTermCount|incidence|ownerCellAt|fallback/i);
});

test("IBO TRA resident hooks reject invalid identifiers", () => {
  assert.throws(() => createSparseCM12IboTRAResidentHooksWGSL({
    residentPrefix: "bad-prefix",
  }), /identifier/);
});
