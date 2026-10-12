import assert from "node:assert/strict";
import test from "node:test";
import { validateLiveFluidEdit, liveFluidEditMode, liveFluidEditDistance, type LiveFluidEdit } from "../lib/core/live-fluid-edit";
import { FluidLabRenderer } from "../lib/core/webgpu-renderer";
const edit: LiveFluidEdit = { operation: "add", shape: "ball", center_m: { x: 4, y: 4, z: 4 }, radius_m: 2 };

test("fluid descriptors reject malformed inputs and preserve distinct shapes", () => {
  assert.throws(() => validateLiveFluidEdit({ ...edit, radius_m: NaN }), /positive radius/);
  assert.throws(() => validateLiveFluidEdit({ ...edit, shape: "torus", tubeRadius_m: 1 }), /half/);
  const torus = validateLiveFluidEdit({ ...edit, shape: "torus" });
  assert.ok(liveFluidEditDistance(torus, edit.center_m) > 0);
  assert.ok(liveFluidEditDistance(torus, { x: 4 + 4 / 3, y: 4, z: 4 }) < 0);
  const corner = { x: 5.5, y: 5.5, z: 5.5 };
  assert.ok(liveFluidEditDistance(edit, corner) > 0);
  assert.ok(liveFluidEditDistance({ ...edit, shape: "cube" }, corner) < 0);
  assert.deepEqual(["ball", "cube", "torus"].flatMap(shape => ["add", "remove"].map(operation =>
    liveFluidEditMode({ shape, operation } as LiveFluidEdit))), [1, 5, 3, 6, 4, 7]);
});

test("renderer reports unavailable dry worlds without reset and awaits accepted receipt before repaint", async () => {
  const renderer = new FluidLabRenderer({} as HTMLCanvasElement, () => {});
  assert.equal((await renderer.editFluid(edit)).accepted, false);
  let resolve!: (value: { accepted: boolean }) => void, invalidations = 0;
  const solver = { editFluid: () => new Promise<{ accepted: boolean }>(done => { resolve = done; }) };
  Object.assign(renderer, { gpuFluid: solver, sparseDeviceReady: () => true,
    waterPipeline: { invalidateSurface() { invalidations++; } } });
  const pending = renderer.editFluid(edit);
  assert.equal(invalidations, 0);
  resolve({ accepted: true });
  assert.deepEqual(await pending, { accepted: true });
  assert.equal(invalidations, 1);
});
