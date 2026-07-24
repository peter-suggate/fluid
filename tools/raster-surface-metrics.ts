export interface EnclosedSurfaceHoleMetrics {
  readonly count: number;
  readonly pixels: number;
  readonly maximumPixels: number;
  readonly maximumWidth_px: number;
  readonly maximumHeight_px: number;
}

export interface SurfaceStepMetrics {
  readonly neighborPairs: number;
  readonly cellScaleHeightJumps: number;
  readonly cellScaleHeightJumpFraction: number;
  readonly largeWorldJumps: number;
  readonly terraceEdgeSamples: number;
  readonly terraceEdges: number;
  readonly terraceEdgeFraction: number;
  readonly maximumHeightCurvature_m: number;
  readonly maximumHeightJump_m: number;
  readonly maximumWorldJump_m: number;
}

function assertRaster(mask: Uint8Array, width: number, height: number): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
    || mask.length !== width * height) throw new Error("Surface raster dimensions are invalid");
}

/**
 * Counts absent components fully enclosed by a projected interface.
 *
 * Boundary-connected background is discarded first, so wall contacts,
 * silhouettes, and concave exterior notches are not mistaken for missing
 * surface cubes. Any remaining component is a literal screen-space hole.
 */
export function enclosedSurfaceHoleMetrics(
  mask: Uint8Array,
  width: number,
  height: number,
): EnclosedSurfaceHoleMetrics {
  assertRaster(mask, width, height);
  const background = new Uint8Array(mask.length);
  for (let index = 0; index < mask.length; index += 1) {
    if (mask[index] === 0) background[index] = 1;
  }
  const stack: number[] = [];
  const admitExterior = (index: number) => {
    if (background[index] === 0) return;
    background[index] = 0;
    stack.push(index);
  };
  for (let x = 0; x < width; x += 1) {
    admitExterior(x);
    admitExterior(x + (height - 1) * width);
  }
  for (let y = 1; y + 1 < height; y += 1) {
    admitExterior(y * width);
    admitExterior(width - 1 + y * width);
  }
  const visit = () => {
    while (stack.length > 0) {
      const index = stack.pop()!;
      const x = index % width;
      const y = Math.floor(index / width);
      if (x > 0) admitExterior(index - 1);
      if (x + 1 < width) admitExterior(index + 1);
      if (y > 0) admitExterior(index - width);
      if (y + 1 < height) admitExterior(index + width);
    }
  };
  visit();

  let count = 0;
  let pixels = 0;
  let maximumPixels = 0;
  let maximumWidth_px = 0;
  let maximumHeight_px = 0;
  for (let seed = 0; seed < background.length; seed += 1) {
    if (background[seed] === 0) continue;
    background[seed] = 0;
    stack.push(seed);
    let componentPixels = 0;
    let minimumX = width, maximumX = -1, minimumY = height, maximumY = -1;
    while (stack.length > 0) {
      const index = stack.pop()!;
      const x = index % width;
      const y = Math.floor(index / width);
      componentPixels += 1;
      minimumX = Math.min(minimumX, x);
      maximumX = Math.max(maximumX, x);
      minimumY = Math.min(minimumY, y);
      maximumY = Math.max(maximumY, y);
      const admitComponent = (neighbor: number) => {
        if (background[neighbor] === 0) return;
        background[neighbor] = 0;
        stack.push(neighbor);
      };
      if (x > 0) admitComponent(index - 1);
      if (x + 1 < width) admitComponent(index + 1);
      if (y > 0) admitComponent(index - width);
      if (y + 1 < height) admitComponent(index + width);
    }
    count += 1;
    pixels += componentPixels;
    maximumPixels = Math.max(maximumPixels, componentPixels);
    maximumWidth_px = Math.max(maximumWidth_px, maximumX - minimumX + 1);
    maximumHeight_px = Math.max(maximumHeight_px, maximumY - minimumY + 1);
  }
  return { count, pixels, maximumPixels, maximumWidth_px, maximumHeight_px };
}

/**
 * Measures cell-sized terraces between adjacent pixels on the same visible
 * sheet. A small x/z separation rejects ordinary depth discontinuities at an
 * occluding silhouette; a half-cell vertical jump is far above raster noise
 * but catches a surface snapped to successive fine-grid layers.
 */
export function surfaceStepMetrics(
  mask: Uint8Array,
  positions: Float32Array,
  width: number,
  height: number,
  fineCellWidth_m: number,
): SurfaceStepMetrics {
  assertRaster(mask, width, height);
  if (positions.length !== mask.length * 3) throw new Error("Surface position raster dimensions are invalid");
  if (!(fineCellWidth_m > 0) || !Number.isFinite(fineCellWidth_m)) {
    throw new Error("Surface step metrics require a finite positive cell width");
  }
  let neighborPairs = 0;
  let cellScaleHeightJumps = 0;
  let largeWorldJumps = 0;
  let terraceEdgeSamples = 0;
  let terraceEdges = 0;
  let maximumHeightCurvature_m = 0;
  let maximumHeightJump_m = 0;
  let maximumWorldJump_m = 0;
  const measure = (a: number, b: number) => {
    if (mask[a] === 0 || mask[b] === 0) return;
    const a3 = a * 3, b3 = b * 3;
    const dx = positions[a3] - positions[b3];
    const dy = positions[a3 + 1] - positions[b3 + 1];
    const dz = positions[a3 + 2] - positions[b3 + 2];
    if (![dx, dy, dz].every(Number.isFinite)) return;
    const horizontal = Math.hypot(dx, dz);
    const heightJump = Math.abs(dy);
    const worldJump = Math.hypot(dx, dy, dz);
    neighborPairs += 1;
    maximumHeightJump_m = Math.max(maximumHeightJump_m, heightJump);
    maximumWorldJump_m = Math.max(maximumWorldJump_m, worldJump);
    if (horizontal <= 0.35 * fineCellWidth_m && heightJump >= 0.45 * fineCellWidth_m) {
      cellScaleHeightJumps += 1;
    }
    if (worldJump >= 1.5 * fineCellWidth_m) largeWorldJumps += 1;
  };
  const measureCurvature = (a: number, b: number, c: number) => {
    if (mask[a] === 0 || mask[b] === 0 || mask[c] === 0) return;
    const a3 = a * 3, b3 = b * 3, c3 = c * 3;
    const ddx = positions[a3] - 2 * positions[b3] + positions[c3];
    const ddy = positions[a3 + 1] - 2 * positions[b3 + 1] + positions[c3 + 1];
    const ddz = positions[a3 + 2] - 2 * positions[b3 + 2] + positions[c3 + 2];
    if (![ddx, ddy, ddz].every(Number.isFinite)) return;
    terraceEdgeSamples += 1;
    const heightCurvature = Math.abs(ddy);
    maximumHeightCurvature_m = Math.max(maximumHeightCurvature_m, heightCurvature);
    if (Math.hypot(ddx, ddz) <= 0.5 * fineCellWidth_m
      && heightCurvature >= 0.75 * fineCellWidth_m) terraceEdges += 1;
  };
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = x + y * width;
    if (x + 1 < width) measure(index, index + 1);
    if (y + 1 < height) measure(index, index + width);
    if (x > 0 && x + 1 < width) measureCurvature(index - 1, index, index + 1);
    if (y > 0 && y + 1 < height) measureCurvature(index - width, index, index + width);
  }
  return {
    neighborPairs,
    cellScaleHeightJumps,
    cellScaleHeightJumpFraction: neighborPairs === 0 ? 0 : cellScaleHeightJumps / neighborPairs,
    largeWorldJumps,
    terraceEdgeSamples,
    terraceEdges,
    terraceEdgeFraction: terraceEdgeSamples === 0 ? 0 : terraceEdges / terraceEdgeSamples,
    maximumHeightCurvature_m,
    maximumHeightJump_m,
    maximumWorldJump_m,
  };
}
