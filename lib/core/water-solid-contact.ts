/** A camera-depth occluder can nominate a solid exit but cannot prove one.
 * Nearby signed normal separation covers the coarse extraction uncertainty;
 * a farther buried surface requires an actual refracted-ray intersection. */
export function verifiedWaterSolidContact(normalGap: number, cellSize: number, rayHit: boolean): boolean {
  return rayHit || Math.abs(normalGap) <= 1.25 * cellSize;
}
export const waterSolidContactWGSL = /* wgsl */ `
fn verifiedWaterSolidContact(normalGap:f32,cellSize:f32,rayHit:bool)->bool{
  return rayHit||abs(normalGap)<=1.25*cellSize;
}
`;
