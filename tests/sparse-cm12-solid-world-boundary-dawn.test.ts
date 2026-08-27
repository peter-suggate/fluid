import test from "node:test";

test("water-box is sealed by its SolidWorld voxel shell", {
  timeout: 240_000,
}, async () => {
  process.env.FLUID_SOLID_WORLD_CM12_STEPS ??= "2";
  await import("../tools/probe-water-box-solid-world-cm12");
});
