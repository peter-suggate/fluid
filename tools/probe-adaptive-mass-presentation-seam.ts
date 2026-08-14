import {
  materializeAdaptiveMassPresentationAtlas,
  type AdaptiveMassAtlasCoordinate,
  type AdaptiveMassPresentationBrick,
} from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-atlas-presentation";

type Axis = 0 | 1 | 2;

interface VariantReceipt {
  axis: Axis;
  coarseSide: "negative" | "positive";
  repeatedRmsError: number;
  reconstructedRmsError: number;
  maximumCoarseMeanError: number;
  maximumSeamDifferenceError: number;
  maximumDensityAuthorityError: number;
}

const CELL_METRES = 0.01;
const COEFFICIENTS = [0.43, -0.29, 0.21] as const;

function linearIndex(dimensions: readonly number[], q: readonly number[]): number {
  return q[0] + dimensions[0] * (q[1] + dimensions[1] * q[2]);
}

function phiAtFineCenter(x: number, y: number, z: number): number {
  return CELL_METRES * (COEFFICIENTS[0] * x + COEFFICIENTS[1] * y
    + COEFFICIENTS[2] * z - 2.75);
}

function makeBrick(
  originFine: AdaptiveMassAtlasCoordinate,
  resolution: 4 | 8,
): AdaptiveMassPresentationBrick {
  const scale = 8 / resolution;
  const density = new Float64Array(resolution ** 3);
  const levelSet = new Float64Array(resolution ** 3);
  for (let z = 0; z < resolution; z += 1) {
    for (let y = 0; y < resolution; y += 1) {
      for (let x = 0; x < resolution; x += 1) {
        const index = x + resolution * (y + resolution * z);
        const phi = phiAtFineCenter(
          originFine[0] + (x + 0.5) * scale,
          originFine[1] + (y + 0.5) * scale,
          originFine[2] + (z + 0.5) * scale,
        );
        levelSet[index] = phi;
        density[index] = Math.max(0, Math.min(1, 0.5 - phi / (4 * CELL_METRES)));
      }
    }
  }
  return { originFine, resolution, fineSpan: 8, density, levelSet };
}

function variant(axis: Axis, coarseSide: "negative" | "positive"): VariantReceipt {
  const dimensions: [number, number, number] = [8, 8, 8];
  dimensions[axis] = 16;
  const negativeOrigin: [number, number, number] = [0, 0, 0];
  const positiveOrigin: [number, number, number] = [0, 0, 0];
  positiveOrigin[axis] = 8;
  const negativeResolution = coarseSide === "negative" ? 4 : 8;
  const positiveResolution = coarseSide === "positive" ? 4 : 8;
  const bricks = [
    makeBrick(negativeOrigin, negativeResolution),
    makeBrick(positiveOrigin, positiveResolution),
  ];
  const materialized = materializeAdaptiveMassPresentationAtlas({
    dimensions,
    bricks,
    emptyLevelSet: 4 * CELL_METRES,
    densityProxyBand: 4 * CELL_METRES,
  });

  let repeatedSquaredError = 0;
  let reconstructedSquaredError = 0;
  let coarseFineSamples = 0;
  let maximumCoarseMeanError = 0;
  let maximumDensityAuthorityError = 0;
  const coarseBrick = coarseSide === "negative" ? bricks[0] : bricks[1];
  for (let localZ = 0; localZ < 4; localZ += 1) {
    for (let localY = 0; localY < 4; localY += 1) {
      for (let localX = 0; localX < 4; localX += 1) {
        const sourceIndex = localX + 4 * (localY + 4 * localZ);
        const sourcePhi = Number(coarseBrick.levelSet![sourceIndex]);
        const sourceDensity = Number(coarseBrick.density[sourceIndex]);
        let reconstructedMean = 0;
        for (let childZ = 0; childZ < 2; childZ += 1) {
          for (let childY = 0; childY < 2; childY += 1) {
            for (let childX = 0; childX < 2; childX += 1) {
              const q = [
                coarseBrick.originFine[0] + 2 * localX + childX,
                coarseBrick.originFine[1] + 2 * localY + childY,
                coarseBrick.originFine[2] + 2 * localZ + childZ,
              ];
              const destination = linearIndex(dimensions, q);
              const expected = phiAtFineCenter(q[0] + 0.5, q[1] + 0.5, q[2] + 0.5);
              const reconstructed = Number(materialized.levelSetOrProxy[destination]);
              repeatedSquaredError += (sourcePhi - expected) ** 2;
              reconstructedSquaredError += (reconstructed - expected) ** 2;
              reconstructedMean += reconstructed / 8;
              maximumDensityAuthorityError = Math.max(maximumDensityAuthorityError,
                Math.abs(Number(materialized.density[destination]) - sourceDensity));
              coarseFineSamples += 1;
            }
          }
        }
        maximumCoarseMeanError = Math.max(
          maximumCoarseMeanError, Math.abs(reconstructedMean - sourcePhi),
        );
      }
    }
  }

  let maximumSeamDifferenceError = 0;
  const q: [number, number, number] = [0, 0, 0];
  const across: [number, number, number] = [0, 0, 0];
  q[axis] = 7;
  across[axis] = 8;
  const tangentA = axis === 0 ? 1 : 0;
  const tangentB = axis === 2 ? 1 : 2;
  for (let b = 0; b < 8; b += 1) {
    for (let a = 0; a < 8; a += 1) {
      q[tangentA] = a;
      q[tangentB] = b;
      across[tangentA] = a;
      across[tangentB] = b;
      const actualDifference = Number(materialized.levelSetOrProxy[linearIndex(dimensions, across)])
        - Number(materialized.levelSetOrProxy[linearIndex(dimensions, q)]);
      const expectedDifference = CELL_METRES * COEFFICIENTS[axis];
      maximumSeamDifferenceError = Math.max(maximumSeamDifferenceError,
        Math.abs(actualDifference - expectedDifference));
    }
  }

  return {
    axis,
    coarseSide,
    repeatedRmsError: Math.sqrt(repeatedSquaredError / coarseFineSamples),
    reconstructedRmsError: Math.sqrt(reconstructedSquaredError / coarseFineSamples),
    maximumCoarseMeanError,
    maximumSeamDifferenceError,
    maximumDensityAuthorityError,
  };
}

const variants = ([0, 1, 2] as const).flatMap((axis) =>
  (["negative", "positive"] as const).map((side) => variant(axis, side)));
const failures: string[] = [];
for (const receipt of variants) {
  const label = `axis ${receipt.axis}, coarse ${receipt.coarseSide}`;
  if (!(receipt.reconstructedRmsError < receipt.repeatedRmsError * 1e-4)) {
    failures.push(`${label}: reconstruction did not remove replicated-leaf stair steps`);
  }
  if (receipt.maximumCoarseMeanError > 2e-8) {
    failures.push(`${label}: coarse-cell phi mean changed by ${receipt.maximumCoarseMeanError}`);
  }
  if (receipt.maximumSeamDifferenceError > 2e-8) {
    failures.push(`${label}: 4^3/8^3 seam slope mismatch ${receipt.maximumSeamDifferenceError}`);
  }
  // The dense publication is intentionally f32; this bound is one rounding
  // ulp around unit density, not a reconstruction allowance.
  if (receipt.maximumDensityAuthorityError > 4e-8) {
    failures.push(`${label}: presentation reconstruction changed physics density`);
  }
}

const result = { passed: failures.length === 0, variants, failures };
if (process.argv.includes("--json")) {
  process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
} else {
  process.stdout.write([
    "Adaptive sparse CM12 presentation seam probe",
    `status: ${result.passed ? "PASS" : "FAIL"}`,
    ...variants.map((receipt) => `axis ${receipt.axis}, coarse ${receipt.coarseSide}: `
      + `RMS ${receipt.repeatedRmsError} -> ${receipt.reconstructedRmsError}; `
      + `mean ${receipt.maximumCoarseMeanError}; seam ${receipt.maximumSeamDifferenceError}`),
    ...failures.map((failure) => `failure: ${failure}`),
  ].join("\n") + "\n");
}
process.exitCode = result.passed ? 0 : 1;
