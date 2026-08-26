import test from "node:test";

test("water-box keeps coarse Sparse CM12 cells beside exact planar walls", {
  timeout: 240_000,
}, async () => {
  process.env.FLUID_PLANAR_CM12_STEPS ??= "2";
  await import("../tools/probe-water-box-planar-cm12");
});
