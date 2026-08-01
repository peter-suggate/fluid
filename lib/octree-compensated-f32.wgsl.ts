/** Deterministic compensated f32 accumulation shared by persistent kernels. */
export const octreeCompensatedF32WGSL = /* wgsl */ `
struct CompensatedF32 { hi: f32, lo: f32 }
fn twoSumF32(a: f32, b: f32) -> CompensatedF32 {
  let hi = a + b;
  let bVirtual = hi - a;
  let lo = (a - (hi - bVirtual)) + (b - bVirtual);
  return CompensatedF32(hi, lo);
}
fn addCompensatedF32(a: CompensatedF32, value: f32) -> CompensatedF32 {
  let first = twoSumF32(a.hi, value);
  let second = twoSumF32(first.lo, a.lo);
  return twoSumF32(first.hi, second.hi + second.lo);
}
fn mergeCompensatedF32(a: CompensatedF32, b: CompensatedF32) -> CompensatedF32 {
  return addCompensatedF32(addCompensatedF32(a, b.hi), b.lo);
}
fn compensatedValue(a: CompensatedF32) -> f32 { return a.hi + a.lo; }
`;
