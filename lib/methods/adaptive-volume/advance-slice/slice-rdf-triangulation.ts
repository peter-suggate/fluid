export type SliceRdfVertex = readonly [number, number, number];

/** A centre fan avoids privileging either diagonal under axis reflection. */
export function sliceRdfTriangles(x: number, y: number,
  a: number, b: number, c: number, d: number): readonly (readonly SliceRdfVertex[])[] {
  const corners: readonly SliceRdfVertex[] = [
    [x, y, a], [x + 1, y, b], [x + 1, y + 1, c], [x, y + 1, d],
  ];
  const centre: SliceRdfVertex = [x + 0.5, y + 0.5, ((a + c) + (b + d)) * 0.25];
  return corners.map((corner, index) => [corner, corners[(index + 1) % 4]!, centre]);
}
