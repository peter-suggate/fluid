export type ContourMetrics = {
  segmentCount: number;
  perimeterM: number | null;
  circularity: number | null;
  surfaceCentroidM: [number, number] | null;
  surfaceCentroidErrorM: number | null;
  radialRmsErrorM: number | null;
  radialMaximumErrorM: number | null;
  boundaryAxisRatio: number | null;
};

export function contourMetrics(segments: ArrayLike<number>, representedAreaFine: number,
  expectedFine: readonly [number, number], radiusFine: number, h: number,
  dimensionsFine: readonly [number, number]): ContourMetrics {
  let perimeter = 0, cx = 0, cy = 0;
  const samples: Array<{ x: number; y: number; weight: number }> = [];
  for (let at = 0; at + 3 < segments.length; at += 4) {
    const x0 = segments[at]!, y0 = segments[at + 1]!;
    const x1 = segments[at + 2]!, y1 = segments[at + 3]!;
    const weight = Math.hypot(x1 - x0, y1 - y0);
    if (!(weight > 0 && Number.isFinite(weight))) continue;
    const x = 0.5 * (x0 + x1), y = 0.5 * (y0 + y1);
    samples.push({ x, y, weight });
    perimeter += weight; cx += weight * x; cy += weight * y;
  }
  if (!(perimeter > 0)) return { segmentCount: 0, perimeterM: null, circularity: null,
    surfaceCentroidM: null, surfaceCentroidErrorM: null, radialRmsErrorM: null,
    radialMaximumErrorM: null, boundaryAxisRatio: null };
  cx /= perimeter; cy /= perimeter;
  let xx = 0, xy = 0, yy = 0, radialSquared = 0, radialMax = 0;
  for (const sample of samples) {
    const dx = sample.x - cx, dy = sample.y - cy;
    xx += sample.weight * dx * dx; xy += sample.weight * dx * dy; yy += sample.weight * dy * dy;
    const error = Math.abs(Math.hypot(sample.x - expectedFine[0], sample.y - expectedFine[1]) - radiusFine);
    radialSquared += sample.weight * error * error; radialMax = Math.max(radialMax, error);
  }
  xx /= perimeter; xy /= perimeter; yy /= perimeter;
  const trace = xx + yy, discriminant = Math.hypot(xx - yy, 2 * xy);
  const minor = 0.5 * (trace - discriminant), major = 0.5 * (trace + discriminant);
  return {
    segmentCount: samples.length,
    perimeterM: perimeter * h,
    circularity: 4 * Math.PI * representedAreaFine / (perimeter * perimeter),
    surfaceCentroidM: [(cx - dimensionsFine[0] / 2) * h, cy * h],
    surfaceCentroidErrorM: Math.hypot(cx - expectedFine[0], cy - expectedFine[1]) * h,
    radialRmsErrorM: Math.sqrt(radialSquared / perimeter) * h,
    radialMaximumErrorM: radialMax * h,
    boundaryAxisRatio: minor > 0 ? Math.sqrt(major / minor) : null,
  };
}
