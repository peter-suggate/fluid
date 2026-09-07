import test from "node:test";

test("water-box is sealed by its SolidWorld voxel shell", {
  timeout: 240_000,
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the SolidWorld boundary regression",
}, async () => {
  process.env.FLUID_SOLID_WORLD_CM12_STEPS ??= "2";
  await import("../tools/probe-water-box-solid-world-cm12");
});
