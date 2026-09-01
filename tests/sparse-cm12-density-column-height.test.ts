import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

function columnHeight(fills: readonly number[], base = 0): number {
  return base + fills.reduce((sum, fill) => sum + Math.max(0, Math.min(1, fill)), 0);
}

function restrictedColumn(waterline: number, scale: number,
  cells: number): number[] {
  return Array.from({ length: cells }, (_, y) => {
    const lower = Math.floor(y / scale) * scale;
    const fill = Math.max(0, Math.min(1, (waterline - lower) / scale));
    return fill;
  });
}

function adaptiveFloorReceipt(fills: readonly number[]): boolean {
  let previous = 1;
  let sawAir = false;
  for (const fill of fills) {
    if (fill > previous + 0.01) return false;
    previous = fill;sawAir ||= fill < 1 - 1e-3;
  }
  return sawAir && columnHeight(fills) <= 8.125;
}

test("integrated density gives one waterline at every represented scale", () => {
  for (const waterline of [0.25, 3.75, 8.125, 8.75, 15.5, 23.875]) {
    const heights = [1, 2, 4, 8].map((scale) =>
      columnHeight(restrictedColumn(waterline, scale, 24)));
    for (const height of heights) {
      assert.ok(Math.abs(height - waterline) < 1e-12,
        `${waterline}: ${heights.join(", ")}`);
    }
  }
});

test("height survives conservative refinement that spreads one cut over children", () => {
  const parentLower = 8;
  const parentFill = 0.375;
  const childFills = [0.5625, 0.1875];
  assert.equal(columnHeight(childFills, parentLower),
    parentLower + 2 * parentFill);
});

test("height integrates a monotone sheet split across coarse vertical bricks", () => {
  const brickWidth = 8;
  const fills = [0.4, 0.1, 0];
  const height = fills.reduce((sum, fill) => sum + brickWidth * fill, 0);
  assert.equal(height, 4);
  assert.ok(fills[0]! < 0.5,
    "the regression must remain invisible to per-coarse-cell rho=.5 classification");
});

test("general adaptive receipt admits floor sheets but rejects elevated liquid", () => {
  assert.equal(adaptiveFloorReceipt([0.4, 0.1, 0]), true);
  assert.equal(adaptiveFloorReceipt([0, 1, 0]), false,
    "the falling corner brick must not be projected down to the floor");
  assert.equal(adaptiveFloorReceipt([1, 1, 1, 1, 1, 1, 1, 1, 1, 0]), false,
    "a tall free surface must keep the established local reconstruction");
});

test("cell-centred coarse heights interpolate continuously across their face", () => {
  const left = 1, right = 5, width = 8;
  const sample = (x: number) => left
    + (right - left) * ((x + 0.5 - width / 2) / width);
  assert.equal(sample(4) - sample(3), (right - left) / width);
  assert.equal(sample(7), 2.75);
  assert.equal(sample(8), 3.25);
  assert.equal(sample(8) - sample(7), (right - left) / width);
});

test("a physical waterline gives a rung-independent ghost-fluid boundary", () => {
  const waterline = 15.25;
  for (const width of [1, 2, 4, 8]) {
    const cutLower = Math.floor(waterline / width) * width;
    const cutFill = (waterline - cutLower) / width;
    const liquidCenter = cutFill >= 0.5
      ? cutLower + width / 2
      : cutLower - width / 2;
    const airCenter = liquidCenter + width;
    const theta = (waterline - liquidCenter) / (airCenter - liquidCenter);
    assert.ok(theta > 0 && theta < 1, `width ${width}: theta=${theta}`);
    assert.ok(Math.abs(
      liquidCenter + theta * (airCenter - liquidCenter) - waterline,
    ) < 1e-12, `width ${width}: reconstructed waterline`);
  }
});

test("production publication uses guarded column height and keeps its fallback", () => {
  assert.match(shader,
    /fn presentationIntegratedColumnHeight[\s\S]*massHeight\+=fill\*f32\(width\)/);
  assert.match(shader,
    /fn presentationIntegratedWorldColumnHeight[\s\S]*for\(var y=0;y<i32\(p\.dimensions\.y\)[\s\S]*massHeight\+=fill\*f32\(width\)/,
    "uniform B1 pages must integrate a vertically split floor sheet");
  assert.match(shader,
    /uniformWorldColumn=!halo&&p\.refinementRegionControl\.x>0u[\s\S]*acceptedBrickResolution\(brick\)==1u[\s\S]*presentationIntegratedWorldColumnHeight/,
    "the full-column walk must remain confined to uniform min-8 pages");
  assert.match(shader,
    /fn presentationIntegratedAdaptiveFloorHeight[\s\S]*compactOwnerCellAt[\s\S]*fill>previous\+0\.01[\s\S]*massHeight>f32\(BRICK_FINE_RESOLUTION\)\+0\.125/,
    "ordinary adaptivity must prove a short monotone floor column from accepted owners");
  assert.match(shader,
    /adaptiveFloorColumn=brickOrigin\.y==0[\s\S]*activityReasons&\(1u\|256u\)[\s\S]*presentationIntegratedAdaptiveFloorHeight/,
    "ordinary surface and thin-fluid floor pages must reach the adaptive receipt");
  const adaptiveGate = shader.slice(shader.indexOf("let adaptiveFloorColumn="),
    shader.indexOf("let worldColumnField="));
  assert.doesNotMatch(adaptiveGate, /refinementRegionControl|acceptedBrickResolution/,
    "ordinary floor sheets must not require an authored region or a particular rung");
  assert.match(shader,
    /var neighbours=array<f32,9>[\s\S]*let lower=mix[\s\S]*let upper=mix[\s\S]*let height=mix/,
    "short min-8 films must share a bilinear height gradient across brick faces");
  assert.match(shader,
    /fn preparePresentationColumnHeights[\s\S]*maximumHeight-minimumHeight<=0\.125/);
  assert.match(shader,
    /owner\.x==INVALID\|\|!brickActive\(owner\.y\)[\s\S]*anchoredAbove=cm12SolidVoxelFractionQ8/,
    "unrepresented open air must anchor generation zero before support activates");
  assert.match(shader,
    /fn presentationHeightPhi[\s\S]*height=presentationColumnHeight[\s\S]*q\.y==0&&height>1e-3[\s\S]*return -1e-3\*p\.frame\.y/,
    "a nonzero sub-half-cell floor sheet must retain a representable inside sample");
  assert.match(shader,
    /presentationHeightFieldValid!=0u[\s\S]*presentationHeightPhi[\s\S]*smoothedPresentationDensityAt/,
    "a rejected height field must retain the limited-linear presentation path");
  assert.match(shader,
    /publishSparseCM12SurfaceRepresentabilityReceipts[\s\S]*preparePresentationColumnHeights[\s\S]*presentationHeightPhi/,
    "the one-rung proof must evaluate the same accepted height geometry");
});

test("pressure theta uses guarded physical column height for vertical planar rows", () => {
  assert.match(shader,
    /fn pressureIntegratedColumnHeight[\s\S]*massHeight\+=fill\*f32\(width\)/);
  assert.match(shader,
    /fn pressurePlanarColumnHeight[\s\S]*maximumHeight-minimumHeight<=0\.01/,
    "the pressure correction must require a locally flat five-column patch");
  assert.match(shader,
    /fn pressureHasPartialRefinementRegion[\s\S]*minimumCellSize>1\.0&&!coversDomain[\s\S]*pressureHasPartialRefinementRegion\(\)/,
    "whole-domain min-8 dams must retain their established pressure path");
  assert.match(shader,
    /var theta=select\(1\.0,cm12GhostFluidTheta[\s\S]*if\(cut&&rowAxis\(row\)==1u[\s\S]*theta=clamp\(\(height-liquidCenterY\)\/\(airCenterY-liquidCenterY\)/,
    "eligible vertical rows must replace density interpolation with physical distance");
  assert.match(shader,
    /var theta=select\(1\.0,cm12GhostFluidTheta/,
    "general interfaces must retain the existing CM12 fallback");
});
