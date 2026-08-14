import {
  advanceTwoTileConservativeTransport,
  integratedScalar,
  type TwoTileTransportFields,
} from "./two-tile-conservative-transport";
import {
  buildTwoTileCompositeGrid,
  type TwoTileCompositeGrid,
  type TwoTileResolution,
} from "./two-tile-composite-grid";
import {
  diffuseTwoTileGamma,
  sharpenTwoTileDensity,
} from "./two-tile-surface-conditioning";

interface SurfaceArm {
  readonly grid: TwoTileCompositeGrid;
  readonly fields: TwoTileTransportFields;
  readonly initialMass: number;
  readonly finalMass: number;
}

export interface AdaptiveMassM1SurfaceDirectionReceipt {
  readonly direction: -1 | 1;
  readonly adaptiveTopology: "8+4" | "4+8";
  readonly fineControlL1: number;
  readonly coarseControlL1: number;
  readonly adaptiveL1: number;
  readonly adaptiveErrorReductionFromCoarse: number;
  readonly maximumMassAbsoluteError: number;
}

export interface AdaptiveMassM1SurfaceABReceipt {
  readonly schemaVersion: 1;
  readonly scope: string;
  readonly steps: number;
  readonly displacementPerStep: number;
  readonly directions: readonly AdaptiveMassM1SurfaceDirectionReceipt[];
  readonly reflectionL1: number;
  readonly minimumRequiredErrorReduction: number;
  readonly failures: readonly string[];
  readonly passed: boolean;
}

export interface AdaptiveMassM1SurfaceABResult {
  readonly receipt: AdaptiveMassM1SurfaceABReceipt;
  readonly svg: string;
}

function initialPulse(grid: TwoTileCompositeGrid, direction: -1 | 1): TwoTileTransportFields {
  const centerNormal = grid.tileWidth - direction * 0.28 * grid.tileWidth;
  return {
    density: Float64Array.from(grid.cells, (cell) => {
      const normal = (cell.center[0] - centerNormal) / (0.32 * grid.tileWidth);
      const tangent = (cell.center[1] - 0.5 * grid.tileWidth) / (0.28 * grid.tileWidth);
      const depth = (cell.center[2] - 0.5 * grid.tileWidth) / (0.36 * grid.tileWidth);
      return Math.max(0, 1 - normal * normal - tangent * tangent - depth * depth) ** 2;
    }),
    gamma: new Float64Array(grid.cells.length).fill(1),
  };
}

function runArm(
  negativeResolution: TwoTileResolution,
  positiveResolution: TwoTileResolution,
  direction: -1 | 1,
  steps: number,
  displacementPerStep: number,
): SurfaceArm {
  const grid = buildTwoTileCompositeGrid({
    axis: 0,
    negativeResolution,
    positiveResolution,
  });
  let fields = initialPulse(grid, direction);
  const initialMass = integratedScalar(grid, fields.density);
  for (let step = 0; step < steps; step += 1) {
    fields = advanceTwoTileConservativeTransport({
      axis: 0,
      negativeResolution,
      positiveResolution,
      displacement: direction * displacementPerStep,
    }, fields).after;
    fields = diffuseTwoTileGamma(grid, fields, 2).fields;
    fields = sharpenTwoTileDensity(grid, fields).fields;
  }
  return {
    grid,
    fields,
    initialMass,
    finalMass: integratedScalar(grid, fields.density),
  };
}

function sample(arm: SurfaceArm, x: number, y: number, z: number): number {
  const tile: 0 | 1 = x < arm.grid.tileWidth ? 0 : 1;
  const resolution = tile === 0
    ? arm.grid.negativeResolution
    : arm.grid.positiveResolution;
  const local = [x - tile * arm.grid.tileWidth, y, z].map((value) =>
    Math.min(resolution - 1, Math.max(0, Math.floor(value * resolution / arm.grid.tileWidth))));
  const base = tile === 0 ? 0 : arm.grid.negativeResolution ** 3;
  const cellId = base + local[0] + resolution * (local[1] + resolution * local[2]);
  return arm.fields.density[cellId];
}

function commonL1(left: SurfaceArm, right: SurfaceArm, reflectRight = false): number {
  let error = 0;
  let count = 0;
  for (let z = 0; z < 8; z += 1) {
    for (let y = 0; y < 8; y += 1) {
      for (let x = 0; x < 16; x += 1) {
        const worldX = (x + 0.5) / 8;
        const rightX = reflectRight ? 2 - worldX : worldX;
        error += Math.abs(sample(left, worldX, (y + 0.5) / 8, (z + 0.5) / 8)
          - sample(right, rightX, (y + 0.5) / 8, (z + 0.5) / 8));
        count += 1;
      }
    }
  }
  return error / count;
}

function color(density: number): string {
  const t = Math.min(1, Math.max(0, density));
  const red = Math.round(8 + 54 * t);
  const green = Math.round(18 + 205 * t);
  const blue = Math.round(32 + 210 * t);
  return `rgb(${red},${green},${blue})`;
}

function renderPanel(arm: SurfaceArm, x: number, y: number, title: string): string {
  const width = 320;
  const height = 160;
  const columns = 64;
  const rows = 32;
  const parts = [
    `<g transform="translate(${x} ${y})">`,
    `<text x="0" y="-10" fill="#dbeafe" font-size="13" font-family="ui-monospace,monospace">${title}</text>`,
    `<rect width="${width}" height="${height}" fill="#081220"/>`,
  ];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const density = sample(arm, 2 * (column + 0.5) / columns,
        (rows - row - 0.5) / rows, 0.5);
      parts.push(`<rect x="${column * width / columns}" y="${row * height / rows}" `
        + `width="${width / columns + 0.15}" height="${height / rows + 0.15}" `
        + `fill="${color(density)}"/>`);
    }
  }
  for (const tile of [0, 1] as const) {
    const resolution = tile === 0 ? arm.grid.negativeResolution : arm.grid.positiveResolution;
    for (let index = 0; index <= resolution; index += 1) {
      const gridX = (tile + index / resolution) * width / 2;
      const gridY = index * height / resolution;
      parts.push(`<path d="M${gridX} 0V${height}" stroke="#ffffff" stroke-opacity="0.12"/>`);
      parts.push(`<path d="M${tile * width / 2} ${gridY}H${(tile + 1) * width / 2}" `
        + `stroke="#ffffff" stroke-opacity="0.12"/>`);
    }
  }
  parts.push(`<path d="M${width / 2} 0V${height}" stroke="#fbbf24" stroke-width="2"/>`,
    `<rect width="${width}" height="${height}" fill="none" stroke="#64748b"/>`, "</g>");
  return parts.join("");
}

/**
 * A small visual A/B for the seam transport problem. It uses the same physical
 * displacement and analytic pulse for the finest, coarsest, and mixed grids.
 * The second row is the exact reflected experiment, not a separately tuned arm.
 */
export function buildAdaptiveMassM1SurfaceAB(options: {
  readonly steps?: number;
  readonly displacementPerStep?: number;
  readonly minimumRequiredErrorReduction?: number;
} = {}): AdaptiveMassM1SurfaceABResult {
  const steps = options.steps ?? 4;
  const displacementPerStep = options.displacementPerStep ?? 0.75 / 8;
  const minimumRequiredErrorReduction = options.minimumRequiredErrorReduction ?? 0.25;
  const positiveFine = runArm(8, 8, 1, steps, displacementPerStep);
  const positiveCoarse = runArm(4, 4, 1, steps, displacementPerStep);
  const positiveAdaptive = runArm(8, 4, 1, steps, displacementPerStep);
  const negativeFine = runArm(8, 8, -1, steps, displacementPerStep);
  const negativeCoarse = runArm(4, 4, -1, steps, displacementPerStep);
  const negativeAdaptive = runArm(4, 8, -1, steps, displacementPerStep);

  const describe = (direction: -1 | 1, topology: "8+4" | "4+8",
    fine: SurfaceArm, coarse: SurfaceArm, adaptive: SurfaceArm,
  ): AdaptiveMassM1SurfaceDirectionReceipt => {
    const coarseControlL1 = commonL1(coarse, fine);
    const adaptiveL1 = commonL1(adaptive, fine);
    return {
      direction,
      adaptiveTopology: topology,
      fineControlL1: 0,
      coarseControlL1,
      adaptiveL1,
      adaptiveErrorReductionFromCoarse: 1 - adaptiveL1 / coarseControlL1,
      maximumMassAbsoluteError: Math.max(
        Math.abs(fine.finalMass - fine.initialMass),
        Math.abs(coarse.finalMass - coarse.initialMass),
        Math.abs(adaptive.finalMass - adaptive.initialMass),
      ),
    };
  };
  const directions = [
    describe(1, "8+4", positiveFine, positiveCoarse, positiveAdaptive),
    describe(-1, "4+8", negativeFine, negativeCoarse, negativeAdaptive),
  ];
  const reflectionL1 = commonL1(positiveAdaptive, negativeAdaptive, true);
  const failures: string[] = [];
  for (const direction of directions) {
    if (direction.maximumMassAbsoluteError > 1e-11) {
      failures.push(`${direction.adaptiveTopology}: mass error ${direction.maximumMassAbsoluteError}`);
    }
    if (direction.adaptiveErrorReductionFromCoarse < minimumRequiredErrorReduction) {
      failures.push(`${direction.adaptiveTopology}: error reduction `
        + `${direction.adaptiveErrorReductionFromCoarse} is below ${minimumRequiredErrorReduction}`);
    }
  }
  if (reflectionL1 > 1e-11) failures.push(`reflected density L1 ${reflectionL1}`);
  const receipt: AdaptiveMassM1SurfaceABReceipt = {
    schemaVersion: 1,
    scope: "Prescribed uniform translation, persistent-gamma diffusion, and conservative sharpening of a density pulse across the frozen seam; this surface-stage visual is not a fully coupled velocity/pressure liquid step.",
    steps,
    displacementPerStep,
    directions,
    reflectionL1,
    minimumRequiredErrorReduction,
    failures,
    passed: failures.length === 0,
  };
  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="470" viewBox="0 0 1040 470">`,
    `<rect width="1040" height="470" fill="#050b14"/>`,
    `<text x="32" y="32" fill="#f8fafc" font-size="20" font-family="ui-sans-serif,sans-serif">Adaptive mass M1 — symmetric CM12 surface-stage A/B</text>`,
    `<text x="32" y="55" fill="#94a3b8" font-size="12" font-family="ui-monospace,monospace">gold = tile seam · grid overlay = actual leaf resolution · identical world displacement</text>`,
    renderPanel(positiveFine, 32, 92, "A fine 8+8 →"),
    renderPanel(positiveCoarse, 360, 92, "A coarse 4+4 →"),
    renderPanel(positiveAdaptive, 688, 92, "B adaptive 8+4 →"),
    renderPanel(negativeFine, 32, 300, "A fine 8+8 ←"),
    renderPanel(negativeCoarse, 360, 300, "A coarse 4+4 ←"),
    renderPanel(negativeAdaptive, 688, 300, "B adaptive 4+8 ←"),
    `</svg>`,
  ].join("");
  return { receipt, svg };
}
