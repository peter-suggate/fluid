import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const source = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

const body = (name: string, next: string): string => {
  const begin = source.indexOf(`fn ${name}`);
  const end = source.indexOf(`fn ${next}`, begin);
  assert.ok(begin >= 0 && end > begin, `${name} source body is present`);
  return source.slice(begin, end);
};

test("sharpening density samples use the staged accepted TEI", () => {
  const sample = body("sampleSharpeningField", "traceSharpeningMass");
  assert.match(sample, /cm12TeiOwnerAtFine/);
  assert.match(sample, /spans=vec3f\(probe\.widths\)/);
  assert.doesNotMatch(sample, /ownerCellAt|acceptedBrickResolution|brickActive/);

  const trace = body("traceSharpeningMass", "scatterSharpeningCell");
  assert.match(trace, /cm12TeiOwnerAtFine\(vec3i\(floor\(position\)\)\)\.cell/);
  assert.match(trace, /cm12TeiOwnerAtFine\(vec3i\(floor\(candidate\)\)\)\.cell/);
  assert.doesNotMatch(trace, /ownerCellAt/);
});

test("the sharpening scatter workgroup stages its spatial tile before tracing", () => {
  const scatterCell = body("scatterSharpeningCell", "scatterSharpeningMass");
  assert.match(scatterCell, /effectiveTransportStencil\(position\)/);
  assert.doesNotMatch(scatterCell, /transportStencil\(position\)/);

  const scatter = body("scatterSharpeningMass", "finalizeSharpeningCell");
  assert.match(scatter, /sirStableMassTileOrigin\(sca1SourceTile\(wid\.x\)\)/);
  assert.match(scatter,
    /cm12TeiStageDirectory\(origin,lane,acceptedTopologySlot\(\)\)/);
  assert.ok(scatter.indexOf("cm12TeiStageDirectory")
    < scatter.indexOf("if(EXP_SHARPENING_CELL_CATALOG)"));
});
