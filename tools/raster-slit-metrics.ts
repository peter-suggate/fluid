export interface NarrowVerticalSlitMetrics {
  readonly count: number;
  readonly pixels: number;
  readonly maximumLength_px: number;
}

/**
 * Finds thin internal gaps in an otherwise filled projected interface.
 *
 * Only horizontally bounded absent runs are candidates, so legitimate
 * silhouette notches remain excluded. Eight-connected candidates must persist
 * vertically for several rows before they count, rejecting isolated raster
 * noise while retaining slightly diagonal cracks.
 */
export function narrowVerticalSlitMetrics(
  mask: Uint8Array,
  width: number,
  height: number,
  maximumWidth_px = 3,
  minimumLength_px = 4,
): NarrowVerticalSlitMetrics {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0
    || mask.length !== width * height) throw new Error("Raster slit mask dimensions are invalid");
  const candidates = new Uint8Array(mask.length);
  for (let y = 0; y < height; y += 1) {
    let x = 0;
    while (x < width) {
      if (mask[x + y * width] !== 0) { x += 1; continue; }
      const first = x;
      while (x < width && mask[x + y * width] === 0) x += 1;
      const lastExclusive = x;
      if (first === 0 || lastExclusive === width || lastExclusive - first > maximumWidth_px
        || mask[first - 1 + y * width] === 0 || mask[lastExclusive + y * width] === 0) continue;
      candidates.fill(1, first + y * width, lastExclusive + y * width);
    }
  }
  let count = 0, pixels = 0, maximumLength_px = 0;
  const stack: number[] = [];
  for (let seed = 0; seed < candidates.length; seed += 1) {
    if (candidates[seed] === 0) continue;
    candidates[seed] = 0; stack.push(seed);
    let componentPixels = 0, minimumY = height, maximumY = -1;
    while (stack.length > 0) {
      const index = stack.pop()!;
      const x = index % width, y = Math.floor(index / width);
      componentPixels += 1; minimumY = Math.min(minimumY, y); maximumY = Math.max(maximumY, y);
      for (let dy = -1; dy <= 1; dy += 1) for (let dx = -1; dx <= 1; dx += 1) {
        if (dx === 0 && dy === 0) continue;
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
        const neighbor = nx + ny * width;
        if (candidates[neighbor] === 0) continue;
        candidates[neighbor] = 0; stack.push(neighbor);
      }
    }
    const length_px = maximumY - minimumY + 1;
    if (length_px < minimumLength_px) continue;
    count += 1; pixels += componentPixels; maximumLength_px = Math.max(maximumLength_px, length_px);
  }
  return { count, pixels, maximumLength_px };
}
