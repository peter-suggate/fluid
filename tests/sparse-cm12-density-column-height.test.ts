import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");
const classifier = readFileSync(new URL(
  "../lib/core/webgpu-water-global-fine-classify.ts",
  import.meta.url,
), "utf8");
const tetraEmitter = readFileSync(new URL(
  "../lib/core/webgpu-water-global-fine-tetra.ts",
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

test("floor ghost continuation reconstructs a true sub-half-cell waterline", () => {
  for (const height of [0.0011, 0.01, 0.125, 0.25, 0.499]) {
    const firstCentrePhi = 0.5 - height;
    const ghostCentrePhi = firstCentrePhi - 1;
    const crossing = -ghostCentrePhi / (firstCentrePhi - ghostCentrePhi);
    const reconstructedHeight = crossing - 0.5;
    assert.ok(Math.abs(reconstructedHeight - height) < 1e-12,
      `${height}: ${reconstructedHeight}`);
  }
});

test("one affine row-zero receipt spans thin, regular, and tall floor sheets", () => {
  for (const height of [0.0011, 0.499, 0.501, 4.49, 4.51, 8.125]) {
    const rowZeroPhi = 0.5 - height;
    assert.ok(Math.abs(0.5 - rowZeroPhi - height) < 1e-12,
      `${height}: row-zero receipt changed representation`);
  }
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

test("production publication uses guarded column height and floor continuation", () => {
  assert.match(shader,
    /fn restrictedPresentationDensityAt[\s\S]*owner\.x!=INVALID&&brickActive\(owner\.y\)[\s\S]*ownerScale>=u32\(cellScale\)/,
    "active dry coarse owners must not expand into finest-child presentation lookups");
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
    /fn presentationHeightPhi[\s\S]*signedFineCells=f32\(q\.y\)\+0\.5-height[\s\S]*return signedFineCells\*p\.frame\.y/,
    "floor height publication must preserve its complete affine signed distance");
  assert.doesNotMatch(shader,
    /fn presentationHeightPhi[\s\S]{0,500}return clamp/,
    "regular-height geometry must not fall back when row-zero phi exceeds the narrow band");
  assert.match(shader,
    /fn presentationFloorContinuationFlag[\s\S]*q\.y==0[\s\S]*presentationColumnHeight[\s\S]*>1e-3/,
    "nonzero floor height fields must mark their boundary continuation sample");
  assert.match(classifier,
    /fn sparseFloorGhostPhi[\s\S]*flags&3u[\s\S]*value-physicalCellSize\(\)\.y/,
    "the marked floor sample must continue affinely into the ghost cube");
  assert.match(classifier,
    /compactSignedSparseAddressing\(\)&&q\.y<0[\s\S]*sparseFloorGhostPhi/,
    "wall-adjacent floor films must use the same ghost continuation");
  assert.match(classifier,
    /fn markedFloorHeight[\s\S]*floorFlags&3u[\s\S]*fn heightFieldPatch[\s\S]*if\(heightFieldPatch\(base,scale,&heights\)\)[\s\S]*if\(base\.y==0\)\{emitHeightFieldPatch\(base,heights\);\}return/,
    "every proved unit heightfield column must replace all of its vertical cube owners");
  assert.match(classifier,
    /fn publishedColumnCrossing[\s\S]*round\(\(\.5\*\(abs\(sa\)-abs\(sb\)\)\/denominator\)\*65536\.0\)\/65536\.0[\s\S]*fn markedFloorHeight[\s\S]*publishedColumnCrossing/,
    "height patches must use the volumetric contour's packed-edge interpolation");
  assert.match(classifier,
    /emitHeightFieldPatch[\s\S]*HEIGHTFIELD_DESCRIPTOR_CODE[\s\S]*vec4f\(heights\[0\],heights\[1\],heights\[2\],heights\[3\]\)/,
    "thin and regular floor-connected surfaces must publish the same four-height patch");
  assert.match(classifier,
    /fineOwnsCube[\s\S]*compactSignedSparseAddressing\(\)&&base\.y==0[\s\S]*q\.y=0/,
    "the fine height patch must remain the sole owner of its below-floor cube");
  assert.match(tetraEmitter,
    /if\(heightField\)\{triangleCount=heightFieldTriangleCount\(lo\);\}[\s\S]*fn heightFieldBoundaryVertex[\s\S]*heightFieldPublishedPhi[\s\S]*fn emitHeightFieldLane[\s\S]*heightFieldBoundaryVertex[\s\S]*heightFieldTri/,
    "one descriptor must emit a continuous, vertically conforming fan with shared height normals");
  assert.match(tetraEmitter,
    /if\(heightField\)\{emitHeightFieldLane\(&cursor,base,lo,lane\);return;\}/,
    "the consolidated production emitter must route both thin and regular heights through the same path");
  assert.match(shader,
    /presentationHeightCache\[column\][\s\S]*fn presentationColumnHeightValid[\s\S]*presentationHeightPhi/,
    "height validity must remain column-local instead of rejecting an entire page");
  assert.match(shader,
    /let stencilCandidate=\(presentationCandidates&1u\)!=0u&&!heightReady/,
    "a mixed-validity page must retain the limited-linear fallback cache");
  assert.match(shader,
    /presentationColumnHeightValid\(i32\(localX\),i32\(localZ\),false\)[\s\S]*presentationHeightPhi/,
    "each valid column must publish its own height while rejected neighbours fall back locally");
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
