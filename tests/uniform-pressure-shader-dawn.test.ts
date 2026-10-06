import assert from "node:assert/strict";
import test from "node:test";
import { createUniformReferenceComputeShader } from "../lib/methods/uniform/webgpu-uniform-reference.wgsl";
import { uniformPressureInPlaceSmootherWGSL, uniformPressureMultigridWGSL } from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl";
import { uniformPressurePagedShader } from "../lib/methods/uniform/uniform-pressure-pages";
import { withUniformDevice } from "./helpers/uniform-geometric";

(process.env.WEBGPU_NODE_MODULE ? test : test.skip)("pressure shaders compile with and without the optional in-place smoother", async () => {
  await withUniformDevice("Uniform pressure shader variants", async device => {
    const base = createUniformReferenceComputeShader(true);
    for (const [label, fragment] of [
      ["native ping-pong", uniformPressureMultigridWGSL],
      ["native in-place", uniformPressureMultigridWGSL + uniformPressureInPlaceSmootherWGSL],
      ["paged", uniformPressurePagedShader(uniformPressureMultigridWGSL)],
      ["paged logical dispatch", uniformPressurePagedShader(uniformPressureMultigridWGSL, true)],
    ]) {
      const module = device.createShaderModule({ label, code: base + fragment });
      const errors = (await module.getCompilationInfo()).messages
        .filter(message => message.type === "error").map(message => message.message);
      assert.deepEqual(errors, [], label);
    }
  });
});
