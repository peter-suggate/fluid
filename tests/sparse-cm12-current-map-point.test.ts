import assert from "node:assert/strict";
import test from "node:test";
import { createSparseCM12CurrentMapLayout, createSparseCM12CurrentMapWGSL } from "../lib/methods/adaptive-mass/sparse-cm12-current-map.wgsl";
import { createSparseCM12CurrentMapMeasureWGSL } from "../lib/methods/adaptive-mass/sparse-cm12-current-map-measure.wgsl";

function body(source: string, name: string) {
  const start = source.indexOf("{", source.indexOf(`fn ${name}(`));
  let depth = 1, end = start + 1;
  while (depth) { if (source[end] === "{") depth++; else if (source[end] === "}") depth--; end++; }
  return source.slice(start + 1, end - 1);
}
const normalized = (source: string) => source.replace(/\s+/g, "");

test("point-only map preserves every point operation and reverse chain step from the full evaluator", () => {
  const source = createSparseCM12CurrentMapWGSL(createSparseCM12CurrentMapLayout(16, [4, 4, 4]));
  // The contract is exact floating-point operation order, not merely an
  // algebraically equivalent spline. Remove only unused derivative statements
  // from the authoritative full evaluator and compare the remaining program.
  const pointProgram = (full: string) => full
    .replace("var result:CurrentMapEvaluation;result.point=point;", "var result=point;")
    .replace(/result\.jacobian=mat3x3f\([^;]+;/g, "")
    .replace(/let d[xyz]=cm12CurrentMapBasisDerivative\([^;]+;/g, "")
    .replace(/var gradient[XYZ]=vec3f\([^;]+;/g, "")
    .replace(/gradient[XYZ]\+=[^;]+;/g, "")
    .replace(/result\.jacobian\[[012]\]\+=[^;]+;/g, "")
    .replace(/result\.jacobian=previous\.jacobian\*result\.jacobian;/g, "")
    .replaceAll("cm12CurrentMapEvaluateIncrementAtBase", "cm12CurrentMapEvaluatePointIncrementAtBase")
    .replaceAll("result.point", "result").replaceAll("previous.point", "previous");
  for (const [full, point] of [["cm12CurrentMapEvaluateIncrementAtBase", "cm12CurrentMapEvaluatePointIncrementAtBase"],
    ["cm12CurrentMapEvaluate", "cm12CurrentMapEvaluatePoint"]]) {
    assert.equal(normalized(body(source, point!)), normalized(pointProgram(body(source, full!))));
    assert.doesNotMatch(body(source, point!), /jacobian|gradient|BasisDerivative/);
  }
});

test("only seed-phi cuts use the point-only path; density and momentum keep their full Jacobian", () => {
  const source = createSparseCM12CurrentMapMeasureWGSL({ baseWords: 64, dimensions: [4, 4, 4] });
  const phi = body(source, "cm12CurrentMapSeedPhiAtFine");
  assert.match(phi, /cm12CurrentMapPointOpen\(point\)/);
  assert.match(phi, /cm12CurrentMapEvaluatePoint\(point,bank\)/);
  assert.doesNotMatch(phi, /determinant|cm12CurrentMapPointDensity/);
  for (const name of ["cm12CurrentMapMeasureIntegrationAxis", "cm12CurrentMapMeasureLine"]) {
    assert.match(body(source, name), /cm12CurrentMapSeedPhiAtFine/);
    assert.doesNotMatch(body(source, name), /cm12CurrentMapPointDensity/);
  }
  assert.match(body(source, "cm12CurrentMapPointDensity"), /cm12CurrentMapEvaluate\(point,bank\)/);
  assert.match(body(source, "cm12CurrentMapPointDensity"), /determinant\(mapped.jacobian\)/);
  assert.match(body(source, "cm12CurrentMapMeasureSample"), /cm12CurrentMapDensityAtFine\(point,bank\)/);
  assert.match(body(source, "cm12CurrentMapPhiAtFine"), /cm12CurrentMapPointDensity\(point,bank\)/);
});
