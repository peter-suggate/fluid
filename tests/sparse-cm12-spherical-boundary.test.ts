import assert from "node:assert/strict";
import test from "node:test";

import { createCm12Figure8, createCm12Figure12 } from
  "../lib/core/cm12-paper-scenes";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import {
  classifyFineBoxAgainstSphericalContainer,
  sphericalContainerOpenFractionAtFineBox,
  type SphericalContainerFineGeometry,
} from "../lib/core/spherical-container";
import {
  createSparseAdaptiveMassAtlas,
  initializeSparseBrickAtlasFromScene,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  dormantReceiverDomain,
  residentSupportAtlas,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { webgpuSparseCM12ResidentWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl";

for (const [figure, makeScene] of [[8, createCm12Figure8],
  [12, createCm12Figure12]] as const) {
  test(`CM12 Figure ${figure} initializes mass against the spherical capacity`, () => {
    const scene = makeScene();
    const dimensions = sceneLatticeDimensions(scene);
    const atlas = initializeSparseBrickAtlasFromScene(scene, {
      finestDimensions: dimensions,
    });
    assert.ok(atlas.boundary, "the sparse atlas must own the embedded boundary");
    let cutBrickCount = 0, wetCutCellCount = 0;
    for (const brick of atlas.bricks) {
      const brickMinimum = brick.coordinate.map((value) => 8 * value) as
        [number, number, number];
      const brickMaximum = brick.coordinate.map((value) => 8 * (value + 1)) as
        [number, number, number];
      const classification = classifyFineBoxAgainstSphericalContainer(
        atlas.boundary, brickMinimum, brickMaximum,
      );
      if (classification === "cut") {
        cutBrickCount += 1;
        assert.equal(brick.resolution, 8,
          "initial liquid touching the vessel uses the finest cut geometry");
      }
      const width = 8 / brick.resolution;
      for (let z = 0; z < brick.resolution; z += 1)
        for (let y = 0; y < brick.resolution; y += 1)
          for (let x = 0; x < brick.resolution; x += 1) {
            const local = x + brick.resolution * (y + brick.resolution * z);
            const center = [brickMinimum[0] + (x + 0.5) * width,
              brickMinimum[1] + (y + 0.5) * width,
              brickMinimum[2] + (z + 0.5) * width] as const;
            const open = sphericalContainerOpenFractionAtFineBox(
              atlas.boundary, center, [width, width, width],
            );
            assert.ok(brick.density[local]! <= open + 1e-12,
              "raw CM12 density cannot exceed the full-volume cut capacity");
            if (open > 0 && open < 1 && brick.density[local]! > 0) wetCutCellCount += 1;
          }
    }
    if (figure === 8) {
      assert.ok(cutBrickCount > 0);
      assert.ok(wetCutCellCount > 0);
    } else {
      assert.equal(cutBrickCount, 0,
        "Figure 12 starts as an interior ball; boundary cut cells are dormant");
    }
  });

  test(`CM12 Figure ${figure} receiver residency excludes exterior sphere bricks`, () => {
    const scene = makeScene();
    const initial = initializeSparseBrickAtlasFromScene(scene, {
      finestDimensions: sceneLatticeDimensions(scene),
    });
    const resident = dormantReceiverDomain(residentSupportAtlas(initial, "adaptive"),
      "adaptive");
    assert.ok(resident.boundary);
    assert.ok(resident.bricks.length < resident.brickDimensions.reduce(
      (product, value) => product * value, 1));
    for (const brick of resident.bricks) {
      const minimum = brick.coordinate.map((value) => 8 * value) as
        [number, number, number];
      const maximum = brick.coordinate.map((value) => 8 * (value + 1)) as
        [number, number, number];
      assert.notEqual(classifyFineBoxAgainstSphericalContainer(
        resident.boundary, minimum, maximum,
      ), "outside");
    }
  });
}

test("composite sparse topology carries cut capacity, aperture, and pressure dual volume", () => {
  const dimensions = [32, 32, 32] as const;
  const brickDimensions = [4, 4, 4] as const;
  const boundary: SphericalContainerFineGeometry = {
    kind: "sphere", centerFine: [16, 16, 16], radiiFine: [15, 15, 15],
  };
  const bricks: SparseAdaptiveMassBrick[] = [];
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1)
    for (let x = 0; x < 4; x += 1) {
      const coordinate = [x, y, z] as const;
      bricks.push({
        key: sparseBrickKey(coordinate, brickDimensions), coordinate, resolution: 4,
        density: new Float64Array(4 ** 3), gamma: new Float64Array(4 ** 3).fill(1),
      });
    }
  const grid = buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(
    dimensions, bricks, 1, boundary,
  ));
  assert.ok(grid.cells.some((cell) => cell.openFraction > 0 && cell.openFraction < 1));
  assert.ok(grid.cells.some((cell) => cell.openVolume === 0));
  assert.ok(grid.cells.some((cell) => cell.separatingPressureMinimum));
  assert.ok(grid.gradientRows.some((row) => row.openFraction > 0 && row.openFraction < 1));
  assert.ok(grid.gradientRows.some((row) => row.openFraction === 0 && row.area === 0));
  assert.ok(grid.gradientRows.some((row) => row.pressureDualOpenFraction > 0
    && row.pressureDualOpenFraction < 1));
  for (const row of grid.gradientRows) {
    assert.equal(row.area, row.geometricArea * row.openFraction);
    assert.equal(row.dualWeight, row.geometricArea * row.distance
      * row.pressureDualOpenFraction);
  }
});

test("resident CM12 uses clipped characteristics and a separating pressure solve", () => {
  assert.match(webgpuSparseCM12ResidentWGSL, /fn clipBoundarySegment/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn cellOpenVolume/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn pressureDensity/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn projectedJacobiValue/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /select\(value,max\(0\.0,value\),cellSeparatingMinimum\(cell\)\)/);
  assert.match(webgpuSparseCM12ResidentWGSL, /fn scatterSolidExcess/);
  assert.match(webgpuSparseCM12ResidentWGSL,
    /insideEmbeddedBoundary\(vec3f\(q\)\+vec3f\(0\.5\)\)/);
});
